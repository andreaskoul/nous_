#!/usr/bin/env python3
"""
Held-out evaluation against the pre-registered gates (plan stage 5).

Loads a checkpoint from train.py, rebuilds the chosen split WITHOUT leakage, and
prints the gated dashboard: E0 curvature on the LEARNED metric, E2/GATE-1 (the
four mandatory baselines), E3 density top-K, C2 fidelity (recon + separability),
grounding drift, plus Recall@k / MRR. PASS/FAIL is judged against the FROZEN
PREREG in config.py — never edit those post-hoc.

Usage:
    python eval.py --ckpt checkpoints/ckpt.pt --split test
"""
from __future__ import annotations
import argparse, math
import torch

from config import Config
from manifold_mvp import device as dev
from manifold_mvp.metric import PullbackMetric
from manifold_mvp.curvature import curvature_probe
from manifold_mvp.experiments import e1_geodesic_recall, e3_density_sampler
from manifold_mvp.locate import (ContextHead, StaticHead, locate_density,
                                 cosine_topk, filesystem_lookup, accuracy)
from manifold_mvp.conformal import ConformalHead
from manifold_mvp.signature import signature
from manifold_mvp.relative import relative_rep
from manifold_mvp import invert
from train import build_data


def banner(t): print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


@torch.no_grad()
def rank_metrics(scores, target, evidence=None, ks=(1, 5, 10)):
    """scores:(Q,N). Returns MRR + Recall@k vs the gold target (or evidence set)."""
    order = scores.argsort(dim=-1, descending=True)
    ranks = (order == target[:, None]).float().argmax(-1) + 1     # 1-based position
    out = {"MRR": (1.0 / ranks).mean().item()}
    for k in ks:
        if evidence is None:
            out[f"R@{k}"] = (ranks <= k).float().mean().item()
        else:
            topk = order[:, :k]
            hit = [len(set(topk[i].tolist()) & set(evidence[i])) > 0 for i in range(len(evidence))]
            out[f"R@{k}"] = float(sum(hit)) / max(1, len(hit))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = ap.parse_args()

    device = dev.pick_device(args.device)
    device, dtype = dev.resolve(device, torch.float32)
    ck = torch.load(args.ckpt, weights_only=False)
    # restore the frozen contract + data config exactly as trained
    from config import DataCfg
    cfg = Config()
    cfg.data = DataCfg(**ck["cfg"]["data"])
    P = ck["cfg"]["prereg"]
    beta = ck["cfg"]["model"]["beta"]
    print(f"Device: {dev.describe(device)} | data: {cfg.data.kind} | split: {args.split}")

    # rebuild data (reuse the TRAIN pca chart for locomo so lambda_theta transfers)
    pca = None
    if cfg.data.kind == "locomo" and "pca" in ck:
        from manifold_mvp.real import TorchPCA
        pca = TorchPCA(ck["pca"]["d"])
        pca.V = ck["pca"]["V"].to(device, dtype); pca.mean = ck["pca"]["mean"].to(device, dtype)
    corpus, manifold, d_latent = build_data(cfg, args.split, device, dtype, pca=pca)
    M = corpus.keys
    N = M.shape[0]

    # restore models
    head = ContextHead(ck["dim"], hidden=ck["cfg"]["model"]["hidden"]).to(device, dtype)
    head.load_state_dict(ck["head"]); head.eval()
    static = StaticHead(ck["dim"], hidden=ck["cfg"]["model"]["hidden"]).to(device, dtype)
    static.load_state_dict(ck["static"]); static.eval()
    lam = ConformalHead(d_latent, hidden=ck["cfg"]["model"]["conformal_hidden"]).to(device, dtype)
    lam.load_state_dict(ck["lam"]); lam.eval()
    metric = PullbackMetric(manifold.decode, conformal=lam)

    # ---------------- E0 : curvature of the LEARNED metric ----------------
    banner("E0  CURVATURE OF THE LEARNED METRIC  G = lambda_theta * G0")
    lat = manifold.sample_latents(120, seed=1)
    e0 = curvature_probe(metric, lat, n_pairs=24, tau_kappa=P["tau_kappa"], steps=80)
    print(f"  shortcut (1-ratio): {e0['shortcut (1-ratio)']:.3f}  (tau_kappa={P['tau_kappa']})")
    print(f"  metric logdet CV  : {e0['metric_logdet_cv']:.3f}")
    print(f"  E0 verdict        : {'GO (curved)' if e0['go'] else 'NO-GO (flat -> reframe)'}")

    # ---------------- E2 / GATE-1 : the four mandatory baselines ----------------
    banner("E2  CONTEXT RECALL  (GATE-1 — the project floor)")
    acc_cos = accuracy(None, corpus, M, beta=beta, kind="cosine")
    acc_fs = accuracy(None, corpus, M, beta=beta, kind="filesystem")
    acc_static = accuracy(static, corpus, M, beta=beta, kind="head")
    acc_ctx = accuracy(head, corpus, M, beta=beta, kind="head")
    best_base = max(acc_cos, acc_fs, acc_static)
    gate1 = (acc_ctx - best_base) >= P["delta2"]
    print(f"  cosine baseline   : {acc_cos:.3f}   (ignores state s)")
    print(f"  filesystem dumb   : {acc_fs:.3f}   (exact-key landmine)")
    print(f"  static head (no s): {acc_static:.3f}   (ablation)")
    print(f"  CONTEXT head      : {acc_ctx:.3f}   (reads q,s)")
    print(f"  GATE-1            : {'PASS' if gate1 else 'FAIL'} "
          f"(delta2={P['delta2']}; beats best baseline by {acc_ctx - best_base:+.3f})")

    # ---------------- ranking metrics (Recall@k / MRR) ----------------
    banner("RETRIEVAL RANKING  (Recall@k / MRR on held-out queries)")
    if hasattr(corpus, "all_queries"):
        q, s, tgt, ev = corpus.all_queries()
    else:
        q, s, tgt = corpus.query_batch(m=2000, seed=12345); ev = None
    phi = head(q, s)
    rm_ctx = rank_metrics(phi @ M.T, tgt, evidence=ev)
    qn = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    rm_cos = rank_metrics(qn @ (M / M.norm(dim=-1, keepdim=True).clamp_min(1e-9)).T, tgt, evidence=ev)
    print(f"  context head : " + "  ".join(f"{k} {v:.3f}" for k, v in rm_ctx.items()))
    print(f"  cosine base  : " + "  ".join(f"{k} {v:.3f}" for k, v in rm_cos.items()))

    # ---------------- E3 : density as parallel sampler ----------------
    banner("E3  DENSITY AS PARALLEL SAMPLER")
    e3 = e3_density_sampler(corpus, head, M, beta=beta)
    print("  " + " | ".join(f"top-{k}:{v:.3f}" for k, v in e3.items()))
    rising = all(e3[b] >= e3[a] - 1e-6 for a, b in zip(list(e3)[:-1], list(e3)[1:]))
    print(f"  E3 verdict        : {'rising -> density buys parallel search' if rising else 'flat'}")

    # ---------------- E1 : geodesic edge (synthetic ground truth only) -------
    if cfg.data.kind == "synthetic":
        banner("E1  GEODESIC vs STRAIGHT-LINE  (ground truth = ambient dist)")
        e1 = e1_geodesic_recall(manifold, metric, n_queries=20, n_cand=10, steps=80)
        e1_pass = (e1["geodesic_top1"] - e1["straight_top1"]) >= P["delta1"]
        print(f"  straight top1 {e1['straight_top1']:.3f} | geodesic top1 {e1['geodesic_top1']:.3f} "
              f"-> {'PASS' if e1_pass else 'fail'} (delta1={P['delta1']})")

    # ---------------- C2 : fidelity of sigma^-1 (parallel module) ----------------
    banner("C2  TRAJECTORY -> POINT  (invertible bridge, own gate)")
    C, T, depth = ck["cfg"]["model"]["c2_channels"], ck["cfg"]["model"]["c2_steps"], ck["cfg"]["model"]["c2_depth"]
    sampler, protos = invert.repertoire_sampler(C, T, n_proto=ck["cfg"]["model"]["c2_n_proto"],
                                                jitter=ck["cfg"]["model"]["c2_jitter"],
                                                seed=ck["cfg"]["model"]["c2_seed"])
    inv = invert.NeuralInverse(ck["sig_dim"], T, C, hidden=256).to(device, dtype)
    inv.load_state_dict(ck["inv"]); inv.eval()
    c2_rel = invert.reconstruction_relerr(inv, sampler(256, 7_777_777).to(device, dtype), depth)
    n_exec = 12
    execs = torch.stack([protos[k][None] + ck["cfg"]["model"]["c2_jitter"]
                         * torch.randn(n_exec, T, C) for k in range(protos.shape[0])]).to(device, dtype)
    sep = invert.skill_separability(lambda p: signature(p, depth), execs)
    c2_pass = (c2_rel <= P["c2_max_recon"]) and (sep >= P["c2_min_separability"])
    print(f"  recon rel-err : {c2_rel:.3f} (gate <= {P['c2_max_recon']})")
    print(f"  separability  : {sep:.1f}x (gate >= {P['c2_min_separability']})")
    print(f"  C2 verdict    : {'FAITHFUL' if c2_pass else 'fail'}")

    # ---------------- grounding (anti-drift) ----------------
    banner("GROUNDING  (anchor-coordinate drift under a change of encoder)")
    anchors = M[:16]; z = M[16:48]
    Q = torch.linalg.qr(torch.randn(M.shape[1], M.shape[1], dtype=dtype, device=device))[0]
    drift = (relative_rep(z @ Q, anchors @ Q) - relative_rep(z, anchors)).norm(dim=-1).mean().item()
    gnd_pass = drift <= P["max_anchor_drift"]
    print(f"  anchor-coord drift: {drift:.2e}  (gate <= {P['max_anchor_drift']}: "
          f"{'pinned' if gnd_pass else 'DRIFT'})")

    # ---------------- dashboard ----------------
    banner("DASHBOARD")
    print(f"  E0 learned-metric curvature : {'GO' if e0['go'] else 'NO-GO'}")
    print(f"  E2 GATE-1 (decisive)        : {'PASS' if gate1 else 'FAIL'}")
    print(f"  C2 bridge faithful          : {'yes' if c2_pass else 'no'}")
    print(f"  grounding pinned            : {'yes' if gnd_pass else 'no'}")
    print(f"  context Recall@1 / MRR      : {rm_ctx['R@1']:.3f} / {rm_ctx['MRR']:.3f}")
    print("\n  PREREG is the contract (config.py). A borderline result is a fail.")


if __name__ == "__main__":
    main()
