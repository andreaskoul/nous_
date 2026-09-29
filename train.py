#!/usr/bin/env python3
"""
Train a deployable nous_ v2 LOCATE model (docs/DESIGN_v2.md) — the library entry point.
eval.py is the scientific harness (cross-validated gates); this script fits ONE model
the way it would be shipped:

  1. L1 fusion alpha chosen on the TRAINING conversations (first-evidence MRR)
  2. LOCATE (identity-anchored residual head + per-memory features + null slot) trained
     with multi-positive InfoNCE + KD from a cross-encoder teacher + Brier on P(null)
  3. on separate CALIBRATION conversations: the post-hoc temperature, the abstention
     threshold (P(null) quantile with <= max_false_abstain on answerable questions),
     and the split-conformal threshold for all-evidence recall >= 1 - conformal_alpha
  4. checkpoint = weights + every fitted constant + encoder id + plan hash

Usage:
    python train.py                                   # LoCoMo, train 0-7, calibrate 8-9
    python train.py --dataset locomo_conv --train-convs 0,1,2,3,4,5,6,7 --calib-convs 8,9
"""
from __future__ import annotations
import argparse, dataclasses, os, time

import numpy as np
import torch

from config import Config
from manifold_mvp import store
from manifold_mvp.candidates import Encoder, CandidateGenerator
from manifold_mvp.decide import CrossEncoderDecider
from manifold_mvp.locate import fit_temperature, conformal_threshold
from manifold_mvp.pipeline import (PairCache, build_batch, state_vectors, teacher_scores, train_locate, locate_logits)
from eval import best_alpha, softmax_np, PLAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="locomo", choices=["locomo", "locomo_conv"])
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--train-convs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--calib-convs", default="8,9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/locate_v2.pt")
    args = ap.parse_args()
    cfg, t0 = Config(), time.time()
    tr_c = {int(c) for c in args.train_convs.split(",")}
    ca_c = {int(c) for c in args.calib_convs.split(",")}
    assert not tr_c & ca_c, "training and calibration conversations must be disjoint"

    if args.dataset == "locomo":
        bank, queries = store.load_locomo(cfg.data.locomo_json)
        mode = "last_session"
    else:
        bank, queries = store.load_locomo_conv(cfg.data.locomo_conv_json, cfg.data.locomo_conv_multimem)
        queries = [q for q in queries if q.style in cfg.data.conv_styles and q.answerable]
        mode = cfg.data.state_mode
    enc_id = args.encoder or cfg.data.encoders[0]
    enc = Encoder(enc_id, cache_dir=cfg.data.cache_dir)
    gen = CandidateGenerator(bank, enc, K=cfg.data.K, contiguity=cfg.data.contiguity)
    gen.index()
    doc = enc.encode_docs(bank.texts)
    q_emb = enc.encode_queries([q.text for q in queries])
    tr = [i for i, q in enumerate(queries) if q.conv in tr_c]
    ca = [i for i, q in enumerate(queries) if q.conv in ca_c]
    alpha = best_alpha(gen, queries, q_emb, [i for i in tr if queries[i].answerable])

    def prep(idx):
        qs = [queries[i] for i in idx]
        b = build_batch(gen, bank, qs, q_emb[idx], cfg.data.K, alpha=alpha)
        S, H = state_vectors(bank, doc, qs, mode=mode)
        return qs, b, S, H

    ce = CrossEncoderDecider(cfg.train.teacher)
    cache = PairCache(os.path.join(cfg.data.cache_dir, "pairs_" + ce.name.replace("/", "_") + ".json"))
    qtr, btr, Str, Htr = prep(tr)
    teacher = teacher_scores(ce, bank, qtr, btr, cache=cache)
    cache.save()
    scorer = train_locate(btr, doc, q_emb[tr], Str, Htr, cfg.weights, teacher=teacher, steps=cfg.train.steps,
                          lr=cfg.train.lr, bsz=cfg.train.batch, seed=args.seed, rank=cfg.model.rank,
                          tau_init=cfg.model.tau_init, null_hidden=cfg.model.null_hidden, kd_T=cfg.train.kd_T)

    # ---- calibration on held-out conversations ----
    qca, bca, Sca, Hca = prep(ca)
    lg = locate_logits(scorer, bca, doc, q_emb[ca], Sca, Hca)
    keep = [j for j in range(len(ca)) if bca.pos[j].any() and not bca.unans[j]]
    c = fit_temperature([lg[j] for j in keep], [np.where(bca.pos[j])[0] for j in keep])
    pn = []
    for j in range(len(ca)):
        z = lg[j].copy(); z[:-1] = c * z[:-1]; z[:-1][~bca.mask[j]] = -1e9
        pn.append(softmax_np(z)[-1])
    pn = np.array(pn)
    tau_null = float(np.quantile(pn[~bca.unans], 1 - cfg.prereg.max_false_abstain))
    thr = conformal_threshold([lg[j, :-1][bca.mask[j]] for j in keep],
                              [np.where(bca.pos[j][bca.mask[j]])[0] for j in keep], cfg.model.conformal_alpha)

    try:
        from manifold_mvp import prereg
        plan = prereg.check(PLAN)
    except Exception as e:  # noqa: BLE001 — a missing plan must not block training
        plan = {"error": str(e)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(state_dict=scorer.state_dict(), feature_mask=scorer.feature_mask,
                    config=dataclasses.asdict(cfg), dataset=args.dataset, encoder=enc_id, alpha=alpha,
                    state_mode=mode, temperature=c, tau_null=tau_null, conformal_threshold=thr,
                    train_convs=sorted(tr_c), calib_convs=sorted(ca_c), plan=plan), args.out)
    print(f"alpha={alpha:.2f} temperature={c:.3f} tau_null={tau_null:.3f} conformal_thr={thr:.3f} "
          f"-> {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
