#!/usr/bin/env python3
"""
nous_ v2 evaluation — the pre-registered gate family (prereg/plan_v2.json,
docs/DESIGN_v2.md §5) on LoCoMo and LoCoMo-Conv.

DEV mode (default): 5-fold leave-conversations-out. Every learned arm is trained on the
other conversations with >= 5 seeds and scored out of fold, so the verdicts are
DEVELOPMENT evidence only.
CONFIRMATORY mode (--confirmatory) refuses to run unless the plan is frozen and
unchanged (manifold_mvp.prereg.require_confirmatory), and it records the plan hash.

  X0  harness reproduction: RRF(BM25, bge-base) + cross-encoder@30 hit@1 ~= 0.524,
      MiniLM dense hit@1 ~= 0.176 (cats 1-4, per-conversation scope)
  G1a C-state   LoCoMo-Conv implicit / composed recall@10: LOCATE with the asking
      speaker's state vs max(same-data static adapter, shuffled-speaker control,
      non-learned speaker-name prefix)
  G1b C-recall  LoCoMo multi-hop all-evidence recall@30: LOCATE vs fusion and vs
      fusion + contiguity only; plus non-inferiority overall
  G1c pipeline  hit@3: fusion -> LOCATE -> cross-encoder vs fusion -> cross-encoder
  A1  abstention: LOCATE P(null) AUROC for category 5 against max-rho, score gap
      and a cross-fitted cross-encoder Noul null; false abstention at a threshold
      frozen on the training conversations
  C1  calibration: per-category top-1 ECE with a temperature fitted on the training
      conversations

Usage:
    python eval.py --quick           # 1 encoder, 2 folds, 2 seeds (smoke)
    python eval.py                   # full dev run
    python eval.py --confirmatory    # only after `python -m manifold_mvp.prereg freeze prereg/plan_v2.json`
"""
from __future__ import annotations
import argparse, dataclasses, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import torch

from config import Config
from manifold_mvp import store, stats
from manifold_mvp.candidates import Encoder, CandidateGenerator, fuse
from manifold_mvp.decide import CrossEncoderDecider, PlattCalibrator, noul_null
from manifold_mvp.locate import fit_temperature
from manifold_mvp.pipeline import (FEATURES, STATE_FEATURES, PairCache, build_batch, state_vectors,
                                   teacher_scores, train_locate, locate_logits, full_ranking, rank_metrics)

PLAN = "prereg/plan_v2.json"
BGE_INSTR = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------- helpers

def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def best_alpha(gen, queries, q_emb, idx, grid=np.linspace(0, 1, 11)):
    """Fusion weight maximising mean first-evidence MRR on the TRAINING queries only."""
    pre = [gen.scores(queries[i].text, queries[i].conv, q_emb=q_emb[i]) for i in idx]
    best, best_a = -1, 0.5
    for a in grid:
        s = np.mean([rank_metrics(fuse(p["bm25"], p["dense"], a), np.asarray(p["rows"]), queries[i].evidence)["mrr"]
                     for p, i in zip(pre, idx)])
        if s > best:
            best, best_a = s, float(a)
    return best_a


def cluster_t_pvalue(d, clusters, null=0.0):
    """One-sided p for H0: mean(d) <= null, cluster-level t with G-1 df (CR1 SE)."""
    G = len(np.unique(clusters))
    mean, lo, hi = stats.cluster_t_ci(d, clusters, alpha=0.10)      # 90% two-sided = 95% one-sided
    se = (hi - lo) / (2 * stats.t_ppf(0.95, G - 1))
    if not np.isfinite(se) or se <= 0:
        return 0.0 if mean > null else 1.0
    return float(1 - stats.t_cdf((mean - null) / se, G - 1))


def cluster_boot_pvalue(stat_fn, idx_by_cluster, null, n=2000, seed=0):
    """One-sided percentile p for H0: stat <= null, resampling CONVERSATIONS."""
    rng = np.random.default_rng(seed)
    keys = list(idx_by_cluster)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by_cluster[keys[p]] for p in pick])
        draws.append(stat_fn(idx))
    draws = np.asarray(draws)
    return float((np.sum(draws <= null) + 1) / (n + 1)), float(np.percentile(draws, 5))


def diff_test(a, b, clusters, null=0.0):
    """H1: mean(a - b) > null. Conservative one-sided p = max(cluster-t, cluster bootstrap)."""
    ok = ~(np.isnan(a) | np.isnan(b))
    d, cl = (a - b)[ok], clusters[ok]
    groups = {c: np.where(cl == c)[0] for c in np.unique(cl)}
    p_t = cluster_t_pvalue(d, cl, null)
    p_b, lo_b = cluster_boot_pvalue(lambda i: d[i].mean(), groups, null)
    return dict(mean=round(float(d.mean()), 4), null=null, p_t=p_t, p_boot=p_b, p=max(p_t, p_b),
                boot_lower_95=round(lo_b, 4), n=int(ok.sum()), n_clusters=len(groups))


def auroc_test(scores, labels, clusters, null=0.60):
    """H1: AUROC > null, cluster bootstrap over conversations."""
    groups = {c: np.where(clusters == c)[0] for c in np.unique(clusters)}
    auc = stats.auroc(scores, labels)
    fn = lambda i: stats.auroc(scores[i], labels[i]) if labels[i].any() and (~labels[i]).any() else 0.5
    p_b, lo_b = cluster_boot_pvalue(fn, groups, null)
    return dict(auroc=round(float(auc), 4), null=null, p=p_b, boot_lower_95=round(lo_b, 4))


def summarize(values, clusters):
    m, lo, hi = stats.cluster_t_ci(values, clusters)
    _, blo, bhi = stats.cluster_ci(values, clusters)
    return dict(mean=round(m, 4), t_ci=[round(lo, 4), round(hi, 4)], boot_ci=[round(blo, 4), round(bhi, 4)], n=int(len(values)))


def logit_slope(conf, correct):
    """Calibration slope: logistic regression of correctness on logit(conf)."""
    x = np.log(np.clip(conf, 1e-6, 1 - 1e-6) / np.clip(1 - conf, 1e-6, 1))
    y = np.asarray(correct, float)
    a, b = 1.0, 0.0
    for _ in range(100):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = np.array([((p - y) * x).sum(), (p - y).sum()])
        w = p * (1 - p) + 1e-9
        H = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a)


def ece_mass(conf, correct, bins=10):
    """ECE with equal-MASS bins (less small-n bias than equal-width bins)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    chunks = np.array_split(np.argsort(conf, kind="stable"), min(bins, len(conf)))
    return float(sum(len(c) / len(conf) * abs(conf[c].mean() - correct[c].mean()) for c in chunks if len(c)))


def ece_debiased(conf, correct, bins=10, n=500, seed=0):
    """Observed ECE minus the ECE a PERFECTLY calibrated model would show at the same
    confidences and n (y ~ Bernoulli(conf)); the raw estimator is biased upward at
    small n (~0.18 per bin at n~19), which made the draft per-fold C1 unpassable."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    obs = ece_mass(conf, correct, bins)
    null = np.array([ece_mass(conf, rng.random(len(conf)) < conf, bins) for _ in range(n)])
    return dict(ece=obs, null=float(null.mean()), debiased=obs - float(null.mean()), n=len(conf))


class Collector:
    """method -> metric -> per-question array (NaN where not evaluated); seeds averaged."""

    def __init__(self, n):
        self.n, self.d, self.cnt = n, defaultdict(dict), defaultdict(dict)

    def add(self, method, qidx, metrics):
        for k, v in metrics.items():
            arr = self.d[method].setdefault(k, np.zeros(self.n))
            cnt = self.cnt[method].setdefault(k, np.zeros(self.n))
            arr[qidx] += v
            cnt[qidx] += 1

    def get(self, method, k):
        c = self.cnt[method][k]
        out = np.full(self.n, np.nan)
        out[c > 0] = self.d[method][k][c > 0] / c[c > 0]
        return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--skip-x0", action="store_true")
    ap.add_argument("--out", default="experiments/results/v2_dev_eval.json")
    args = ap.parse_args()
    cfg = Config()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.time()

    plan_info = {"mode": "dev"}
    if args.confirmatory:
        from manifold_mvp import prereg
        prereg.require_confirmatory(PLAN)
        plan_info = {"mode": "confirmatory", **prereg.check(PLAN)}

    encoders = list(cfg.data.encoders)
    folds = [list(f) for f in cfg.data.folds]
    seeds = list(cfg.train.seeds)
    K, steps = cfg.data.K, cfg.train.steps
    if args.quick:
        encoders, folds, seeds, K, steps = encoders[:1], [[0, 1], [8, 9]], seeds[:2], 30, 120
        args.skip_x0 = True

    bank, qs = store.load_locomo(cfg.data.locomo_json)
    bank_c, qc_all = store.load_locomo_conv(cfg.data.locomo_conv_json, cfg.data.locomo_conv_multimem)
    assert bank.texts == bank_c.texts, "LoCoMo-Conv must share the LoCoMo memory bank"
    # G1a measures recall of GOLD evidence: adversarial rewrites (category 5) point at a
    # premise-violating trap turn, not at something to recall, so they are excluded here
    qc = [q for q in qc_all if q.style in cfg.data.conv_styles and q.answerable]
    eval_convs = sorted({c for f in folds for c in f})
    print(f"LoCoMo {len(bank)} memories, {len(qs)} questions; LoCoMo-Conv {len(qc)} queries "
          f"({', '.join(cfg.data.conv_styles)}); folds {folds}; seeds {seeds}; K={K}")

    ce = CrossEncoderDecider(cfg.train.teacher)
    cache = PairCache(os.path.join(cfg.data.cache_dir, "pairs_" + ce.name.replace("/", "_") + ".json"))
    qconv = np.array([q.conv for q in qs]); qcat = np.array([q.category for q in qs])
    cconv = np.array([q.conv for q in qc]); cstyle = np.array([q.style for q in qc])
    out = dict(plan=plan_info, folds=folds, seeds=seeds, K=K, steps=steps, encoders={})
    no_state = np.array([f not in STATE_FEATURES for f in FEATURES], np.float32)

    # ---------------- X0: reproduce the reality-check bar with the v2 modules -------------
    if not args.skip_x0:
        x0_enc = Encoder("BAAI/bge-base-en-v1.5", query_instruction=BGE_INSTR, cache_dir=cfg.data.cache_dir)
        x0_gen = CandidateGenerator(bank, x0_enc, method="rrf", K=30, contiguity=0); x0_gen.index()
        qe = x0_enc.encode_queries([q.text for q in qs])
        m_idx = [i for i, q in enumerate(qs) if q.category != 5]
        x0b = build_batch(x0_gen, bank, [qs[i] for i in m_idx], qe[m_idx], 30)
        t = teacher_scores(ce, bank, [qs[i] for i in m_idx], x0b, cache=cache)
        cache.save()
        hit = [rank_metrics(*full_ranking(x0b, j, t[j]), qs[i].evidence)["hit@1"] for j, i in enumerate(m_idx)]
        mini = Encoder("sentence-transformers/all-MiniLM-L6-v2", cache_dir=cfg.data.cache_dir)
        mg = CandidateGenerator(bank, mini, K=30); mg.index()
        mq = mini.encode_queries([qs[i].text for i in m_idx])
        dh = [rank_metrics(mg.scores(qs[i].text, qs[i].conv, q_emb=mq[j])["dense"],
                           np.asarray(mg.scores(qs[i].text, qs[i].conv, q_emb=mq[j])["rows"]), qs[i].evidence)["hit@1"]
              for j, i in enumerate(m_idx)]
        x0 = dict(rerank_hit1=round(float(np.mean(hit)), 4), minilm_dense_hit1=round(float(np.mean(dh)), 4))
        x0["pass"] = bool(abs(x0["rerank_hit1"] - cfg.prereg.x0_rerank_hit1) <= cfg.prereg.x0_tol
                          and abs(x0["minilm_dense_hit1"] - cfg.prereg.x0_minilm_hit1) <= cfg.prereg.x0_tol)
        out["X0"] = x0
        print(f"X0 {x0}  ({time.time()-t0:.0f}s)")

    for enc_id in encoders:
        short = enc_id.split("/")[-1]
        enc = Encoder(enc_id, query_instruction=(BGE_INSTR if cfg.data.query_instruction and "bge" in enc_id else None),
                      cache_dir=cfg.data.cache_dir)
        gen = CandidateGenerator(bank, enc, alpha=0.5, method="zscore", K=K, contiguity=cfg.data.contiguity)
        gen.index()
        doc = enc.encode_docs(bank.texts)
        q_lo = enc.encode_queries([q.text for q in qs])
        q_cv = enc.encode_queries([q.text for q in qc])
        pre = [dataclasses.replace(q, text=store.render_query(q, speaker_prefix=True)) for q in qc]
        q_pre = enc.encode_queries([q.text for q in pre])
        L = Collector(len(qs)); C = Collector(len(qc))
        a1 = defaultdict(list); c1 = defaultdict(list); alphas = []

        for f_i, test in enumerate(folds):
            # ================= LoCoMo: G1b, G1c, A1, C1 =================
            tr = [i for i, q in enumerate(qs) if q.conv not in test]
            te = [i for i, q in enumerate(qs) if q.conv in test]
            alpha = best_alpha(gen, qs, q_lo, [i for i in tr if qs[i].category != 5])
            alphas.append(alpha)
            btr = build_batch(gen, bank, [qs[i] for i in tr], q_lo[tr], K, alpha=alpha)
            bte = build_batch(gen, bank, [qs[i] for i in te], q_lo[te], K, alpha=alpha)
            ttr = teacher_scores(ce, bank, [qs[i] for i in tr], btr, cache=cache)
            tte = teacher_scores(ce, bank, [qs[i] for i in te], bte, cache=cache)
            cache.save()
            Str, Htr = state_vectors(bank, doc, [qs[i] for i in tr], mode="last_session")
            Ste, Hte = state_vectors(bank, doc, [qs[i] for i in te], mode="last_session")
            for j, i in enumerate(te):
                ev = qs[i].evidence
                L.add("fusion", [i], rank_metrics(bte.fused_full[j], bte.conv_rows[j], ev))
                L.add("fusion+contiguity", [i], rank_metrics(*full_ranking(bte, j, -np.arange(K, dtype=float)), ev))
                fz = bte.feats[j, :, 2].copy(); fz[~bte.mask[j]] = -1e9
                top = np.argsort(-fz)[: cfg.data.rerank_depth]
                sc = fz.copy(); sc[top] = 1e3 + tte[j, top]
                L.add("fusion->CE", [i], rank_metrics(*full_ranking(bte, j, sc), ev))
            # cross-fitted CE Noul null (Platt on training pairs)
            pl = PlattCalibrator()
            pl.fit(ttr[btr.mask], btr.pos[btr.mask].astype(float))
            for seed in seeds:
                sc_ = train_locate(btr, doc, q_lo[tr], Str, Htr, cfg.weights, teacher=ttr, steps=steps,
                                   lr=cfg.train.lr, bsz=cfg.train.batch, seed=seed, rank=cfg.model.rank,
                                   tau_init=cfg.model.tau_init, null_hidden=cfg.model.null_hidden, kd_T=cfg.train.kd_T)
                lg_tr = locate_logits(sc_, btr, doc, q_lo[tr], Str, Htr)
                lg_te = locate_logits(sc_, bte, doc, q_lo[te], Ste, Hte)
                keep = [j for j in range(len(tr)) if btr.pos[j].any() and not btr.unans[j]]
                c = fit_temperature([lg_tr[j] for j in keep], [np.where(btr.pos[j])[0] for j in keep])
                def dens(lg, mask):
                    z = lg.copy(); z[:-1] = c * z[:-1]; z[:-1][~mask] = -1e9
                    return softmax_np(z)
                pn_tr = np.array([dens(lg_tr[j], btr.mask[j])[-1] for j in range(len(tr))])
                thr = float(np.quantile(pn_tr[~btr.unans], 1 - cfg.prereg.max_false_abstain))
                for j, i in enumerate(te):
                    ev = qs[i].evidence
                    ms = lg_te[j, :-1].copy(); ms[~bte.mask[j]] = -1e9
                    L.add("LOCATE", [i], rank_metrics(*full_ranking(bte, j, ms), ev))
                    top = np.argsort(-ms)[: cfg.data.rerank_depth]
                    sc = ms.copy(); sc[top] = 1e3 + tte[j, top]
                    L.add("fusion->LOCATE->CE", [i], rank_metrics(*full_ranking(bte, j, sc), ev))
                    p = dens(lg_te[j], bte.mask[j])
                    pm = p[:-1]
                    a1["null_slot"].append((i, float(p[-1]), float(p[-1] >= thr)))
                    a1["one_minus_max_rho"].append((i, float(1 - pm.max()), 0.0))
                    srt = np.sort(ms[bte.mask[j]])[::-1]
                    a1["neg_score_gap"].append((i, float(-(srt[0] - srt[min(len(srt) - 1, 5)])), 0.0))
                    pce = pl.predict(tte[j, bte.mask[j]])
                    a1["ce_noul_null"].append((i, float(noul_null(pce)), 0.0))
                    if qs[i].category != 5:
                        top1 = int(np.argmax(pm))
                        c1["conf"].append((i, float(pm[top1] / max(1e-12, pm.sum())), float(bte.pos[j, top1]), f_i, seed))
            print(f"  [{short}] LoCoMo fold {f_i} done (alpha={alpha:.1f}, {time.time()-t0:.0f}s)")

            # ================= LoCoMo-Conv: G1a =================
            ctr = [i for i, q in enumerate(qc) if q.conv not in test]
            cte = [i for i, q in enumerate(qc) if q.conv in test]
            ca = best_alpha(gen, qc, q_cv, ctr)
            arms = {}
            shuf = store.shuffled_speakers(qc, seed=f_i)
            for arm, spk in [("context", None), ("static", [""] * len(qc)), ("shuffled", shuf)]:
                spk_tr = None if spk is None else [spk[i] for i in ctr]
                spk_te = None if spk is None else [spk[i] for i in cte]
                b_tr = build_batch(gen, bank, [qc[i] for i in ctr], q_cv[ctr], K, speakers=spk_tr, alpha=ca)
                b_te = build_batch(gen, bank, [qc[i] for i in cte], q_cv[cte], K, speakers=spk_te, alpha=ca)
                mode = "none" if arm == "static" else "speaker_persona"
                S_tr, H_tr = state_vectors(bank, doc, [qc[i] for i in ctr], mode=mode, speakers=spk_tr)
                S_te, H_te = state_vectors(bank, doc, [qc[i] for i in cte], mode=mode, speakers=spk_te)
                arms[arm] = (b_tr, b_te, S_tr, H_tr, S_te, H_te)
            t_tr = teacher_scores(ce, bank, [qc[i] for i in ctr], arms["context"][0], cache=cache)
            cache.save()
            b_te = arms["context"][1]
            bp = build_batch(gen, bank, [pre[i] for i in cte], q_pre[cte], K, alpha=ca)
            for j, i in enumerate(cte):
                ev = qc[i].evidence
                C.add("fusion", [i], rank_metrics(b_te.fused_full[j], b_te.conv_rows[j], ev))
                sc = gen.scores(qc[i].text, qc[i].conv, q_emb=q_cv[i])
                C.add("bm25", [i], rank_metrics(sc["bm25"], np.asarray(sc["rows"]), ev))
                C.add("dense", [i], rank_metrics(sc["dense"], np.asarray(sc["rows"]), ev))
                C.add("speaker_prefix_fusion", [i], rank_metrics(bp.fused_full[j], bp.conv_rows[j], ev))
            for arm, (b_tr, b_te2, S_tr, H_tr, S_te, H_te) in arms.items():
                feats_mask = no_state if arm == "static" else None
                for seed in seeds:
                    sc_ = train_locate(b_tr, doc, q_cv[ctr], S_tr, H_tr, cfg.weights, teacher=t_tr, steps=steps,
                                       lr=cfg.train.lr, bsz=cfg.train.batch, seed=seed, rank=cfg.model.rank,
                                       tau_init=cfg.model.tau_init, null_hidden=cfg.model.null_hidden,
                                       kd_T=cfg.train.kd_T, use_feats=feats_mask)
                    lg = locate_logits(sc_, b_te2, doc, q_cv[cte], S_te, H_te)
                    for j, i in enumerate(cte):
                        ms = lg[j, :-1].copy(); ms[~b_te2.mask[j]] = -1e9
                        C.add(f"LOCATE_{arm}", [i], rank_metrics(*full_ranking(b_te2, j, ms), qc[i].evidence))
            print(f"  [{short}] LoCoMo-Conv fold {f_i} done ({time.time()-t0:.0f}s)")

        # ---------------- aggregate for this encoder ----------------
        E = dict(alphas=alphas)
        lmask = np.isin(qconv, eval_convs)
        main = lmask & (qcat != 5)
        mh = lmask & (qcat == 1)
        E["locomo"] = {m: {k: summarize(L.get(m, k)[main], qconv[main]) for k in ("hit@1", "hit@3", "mrr", "recall@30", "cover@30")}
                       for m in L.d}
        E["locomo_multihop"] = {m: {k: summarize(L.get(m, k)[mh], qconv[mh]) for k in ("recall@30", "cover@30")} for m in L.d}
        # Confirmatory tests of this encoder. Each: one-sided p (conservative of cluster-t and
        # cluster bootstrap) + the plan's point criterion; Holm is applied over ALL encoders' tests below.
        T = {}
        for b in ("fusion", "fusion+contiguity"):
            t = diff_test(L.get("LOCATE", "recall@30")[mh], L.get(b, "recall@30")[mh], qconv[mh], 0.0)
            t["criterion"] = bool(t["mean"] >= cfg.prereg.delta_recall_mh)
            T[f"G1b:multihop_recall30_vs_{b}"] = t
        t = diff_test(L.get("LOCATE", "recall@30")[main], L.get("fusion", "recall@30")[main], qconv[main], -cfg.prereg.ni_margin)
        t["criterion"] = True
        T["G1b:overall_recall30_noninferior"] = t
        t = diff_test(L.get("fusion->LOCATE->CE", "hit@3")[main], L.get("fusion->CE", "hit@3")[main], qconv[main], -cfg.prereg.ni_margin)
        t["criterion"] = True
        T["G1c:overall_hit3_noninferior"] = t
        t = diff_test(L.get("fusion->LOCATE->CE", "hit@3")[mh], L.get("fusion->CE", "hit@3")[mh], qconv[mh], 0.0)
        t["criterion"] = True
        T["G1c:multihop_hit3_superior"] = t
        # A1 (descriptive arms + the gated null slot)
        A = {}
        for name, rows in a1.items():
            idx = np.array([r[0] for r in rows]); sc = np.array([r[1] for r in rows]); flag = np.array([r[2] for r in rows])
            y = qcat[idx] == 5
            A[name] = dict(auroc_cat5=round(stats.auroc(sc, y), 4))
            if name == "null_slot":
                A[name].update(false_abstain=round(float(flag[~y].mean()), 4), correct_abstain=round(float(flag[y].mean()), 4))
                t = auroc_test(sc, y, qconv[idx], cfg.prereg.null_auroc_lo)
                t["criterion"] = bool(t["auroc"] >= cfg.prereg.null_auroc and A[name]["false_abstain"] <= cfg.prereg.max_false_abstain)
                T["A1:null_slot_auroc"] = t
        E["A1"] = A
        # C1 — pooled out-of-fold per category, equal-mass bins, DEBIASED against the ECE a
        # perfectly calibrated model would show at the same n (parametric bootstrap); per seed.
        rows = np.array(c1["conf"], dtype=float)          # (qidx, conf, correct, fold, seed)
        per_seed, slopes = defaultdict(list), []
        for sd in np.unique(rows[:, 4]):
            r = rows[rows[:, 4] == sd]
            cats = qcat[r[:, 0].astype(int)]
            for cat in (1, 2, 3, 4):
                m = cats == cat
                if m.sum() >= 10:
                    per_seed[store_cat(cat)].append(ece_debiased(r[m, 1], r[m, 2]))
            slopes.append(logit_slope(r[:, 1], r[:, 2]))
        C1 = {cat: dict(ece=round(float(np.mean([v["ece"] for v in vs])), 4),
                        null_ece=round(float(np.mean([v["null"] for v in vs])), 4),
                        debiased=round(float(np.mean([v["debiased"] for v in vs])), 4),
                        debiased_sd_over_seeds=round(float(np.std([v["debiased"] for v in vs])), 4),
                        n=int(np.mean([v["n"] for v in vs])))
              for cat, vs in per_seed.items()}
        slope = float(np.mean(slopes))
        r0 = rows[rows[:, 4] == rows[0, 4]]                # descriptive: plan-v2-draft per-fold ECE (seed 0)
        per_fold = {store_cat(cat): [round(stats.ece(r0[(qcat[r0[:, 0].astype(int)] == cat) & (r0[:, 3] == f), 1],
                                                     r0[(qcat[r0[:, 0].astype(int)] == cat) & (r0[:, 3] == f), 2]), 4)
                                     for f in range(len(folds))] for cat in (1, 2, 3, 4)}
        E["C1"] = dict(by_category=C1, calibration_slope=round(slope, 3), slope_sd_over_seeds=round(float(np.std(slopes)), 3),
                       per_fold_equal_width_ece_seed0=per_fold,
                       pass_=bool(C1 and all(v["debiased"] <= cfg.prereg.max_ece for v in C1.values())
                                  and cfg.prereg.calib_slope[0] <= slope <= cfg.prereg.calib_slope[1]))
        # G1a
        cmask = np.isin(cconv, eval_convs)
        E["locomo_conv"], E["G1a_rivals"] = {}, {}
        for style in cfg.data.conv_styles:
            sm = cmask & (cstyle == style)
            E["locomo_conv"][style] = {m: summarize(C.get(m, "recall@10")[sm], cconv[sm]) for m in C.d}
            rivals = {r: float(np.nanmean(C.get(r, "recall@10")[sm])) for r in ("LOCATE_static", "LOCATE_shuffled", "speaker_prefix_fusion")}
            best = max(rivals, key=rivals.get)       # best comparator = highest pooled out-of-fold mean
            t = diff_test(C.get("LOCATE_context", "recall@10")[sm], C.get(best, "recall@10")[sm], cconv[sm], 0.0)
            t["criterion"] = bool(t["mean"] >= cfg.prereg.delta_state)
            t["best_rival"] = best
            T[f"G1a:{style}_recall10_vs_best_rival"] = t
            E["G1a_rivals"][style] = dict(rivals=rivals, context=float(np.nanmean(C.get("LOCATE_context", "recall@10")[sm])))
        E["tests"] = T
        out["encoders"][short] = E
        print(f"[{short}] done ({time.time()-t0:.0f}s)")

    # Holm over the WHOLE confirmatory family (all encoders, all gates with a test)
    fam = {f"{s}|{n}": t["p"] for s, E in out["encoders"].items() for n, t in E["tests"].items()}
    adj = stats.holm(fam) if fam else {}
    gates = defaultdict(dict)
    for s, E in out["encoders"].items():
        for n, t in E["tests"].items():
            t["holm_p"] = adj[f"{s}|{n}"]
            t["pass_"] = bool(t["criterion"] and t["holm_p"] < cfg.prereg.alpha)
        for g in ("G1a", "G1b", "G1c", "A1"):
            ts = [t for n, t in E["tests"].items() if n.startswith(g + ":")]
            gates[g][s] = bool(ts) and all(t["pass_"] for t in ts)
        gates["C1"][s] = E["C1"]["pass_"]
    out["gates"] = {g: dict(v, any_encoder=any(v.values())) for g, v in gates.items()}
    out["holm_family_size"] = len(fam)
    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.quick:
        args.out = args.out.replace(".json", "_quick.json")
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print_dashboard(out)
    print(f"\nwrote {args.out} ({out['runtime_s']}s)")


def store_cat(c):
    return {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}[c]


def print_dashboard(out):
    print("\n" + "=" * 72 + "\n  nous_ v2 — " + out["plan"]["mode"].upper() + " gate dashboard\n" + "=" * 72)
    if "X0" in out:
        print(f"  X0  harness reproduction       : {'PASS' if out['X0']['pass'] else 'FAIL'}  {out['X0']}")
    for short, E in out["encoders"].items():
        print(f"  --- encoder {short} ---")
        lo = E["locomo"]
        for m in ("fusion", "fusion+contiguity", "LOCATE", "fusion->CE", "fusion->LOCATE->CE"):
            if m in lo:
                print(f"    {m:22s} hit@1 {lo[m]['hit@1']['mean']:.3f}  hit@3 {lo[m]['hit@3']['mean']:.3f}  "
                      f"MRR {lo[m]['mrr']['mean']:.3f}  recall@30 {lo[m]['recall@30']['mean']:.3f}")
        for n, t in E["tests"].items():
            val = f"AUROC {t['auroc']:.3f}" if "auroc" in t else f"diff {t['mean']:+.4f} (null {t['null']:+.2f})"
            print(f"    {'PASS' if t['pass_'] else 'fail'}  {n:42s} {val}  criterion={t['criterion']}  Holm p={t['holm_p']:.3g}")
        a = E["A1"]
        print("    A1 arms: " + ", ".join(f"{k} AUROC {v['auroc_cat5']:.3f}" for k, v in a.items())
              + f"; null-slot false-abstain {a['null_slot']['false_abstain']:.3f}, correct-abstain {a['null_slot']['correct_abstain']:.3f}")
        print(f"    C1 {'PASS' if E['C1']['pass_'] else 'fail'}: slope {E['C1']['calibration_slope']:.2f}; "
              + ", ".join(f"{k} ECE {v['ece']:.3f} (null {v['null_ece']:.3f}, debiased {v['debiased']:+.3f}, n={v['n']})"
                          for k, v in E['C1']['by_category'].items()))
        for style, r in E["G1a_rivals"].items():
            print(f"    G1a [{style}] context {r['context']:.3f} vs " + ", ".join(f"{k} {v:.3f}" for k, v in r["rivals"].items()))
    print(f"\n  GATES (Holm over {out['holm_family_size']} tests): "
          + ", ".join(f"{g} {'PASS' if v['any_encoder'] else 'FAIL'}" for g, v in out["gates"].items()))
    print("  DEV mode verdicts are development evidence only.")


if __name__ == "__main__":
    main()
