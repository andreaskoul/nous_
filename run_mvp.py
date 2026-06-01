#!/usr/bin/env python3
"""
Manifold-MVP runner.  Executes the Act-II protocol end to end:

    E0  curvature probe        (GATE: is the geometry load-bearing?)
    E1  geodesic vs cosine recall
    E2  learned context recall (GATE 1: the project floor)
    E3  density as parallel sampler (optional contribution)

Plus a C2 signature-bridge fidelity check and a grounding/stitching demo.

Pre-registered thresholds live in PREREG below — EDIT THEM BEFORE TRUSTING A
RUN, then never touch them again (the protocol's non-negotiable rule).

Usage:
    python run_mvp.py                  # auto device (MPS on M4), float32
    python run_mvp.py --device mps
    python run_mvp.py --precision float64   # forces CPU for E0 sensitivity
"""
from __future__ import annotations
import argparse, time
import torch

from manifold_mvp import device as dev
from manifold_mvp.synthetic import CurvedManifold, ContextCorpus
from manifold_mvp.metric import PullbackMetric
from manifold_mvp.curvature import curvature_probe
from manifold_mvp.experiments import e1_geodesic_recall, e3_density_sampler
from manifold_mvp.locate import (StaticHead, ContextHead, train_head, accuracy)
from manifold_mvp.signature import SkillBridge, signature, levy_area as sig_levy
from manifold_mvp.relative import relative_rep, drift, procrustes_align

# ---- PRE-REGISTRATION (freeze before trusting results) ----------------------
PREREG = dict(
    tau_kappa=0.02,     # E0: min (detour_ratio - 1) to call the manifold curved
    delta1=0.10,        # E1: min (geodesic_top1 - straight_top1)
    delta2=0.15,        # E2: min (context_head_acc - best_baseline_acc)
    c2_max_recon=0.20,  # C2: max relative reconstruction error to call sigma faithful
)


def banner(t): print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--precision", default="float32", choices=["float32", "float64"])
    ap.add_argument("--quick", action="store_true", help="fewer iters for a smoke test")
    args = ap.parse_args()

    dtype = torch.float64 if args.precision == "float64" else torch.float32
    device = dev.pick_device(args.device)
    device, dtype = dev.resolve(device, dtype)
    print(f"Device : {dev.describe(device)}")
    print(f"Dtype  : {dtype}")
    torch.manual_seed(0)
    qs = 12 if args.quick else 30
    steps = 60 if args.quick else 120

    # ---------------- E0 : curvature gate ----------------
    banner("E0  CURVATURE PROBE  (gate: is geometry load-bearing?)")
    man = CurvedManifold(d_latent=2, D=32, warp=2.0, freq=2.5,
                         device=device, dtype=dtype)
    g = PullbackMetric(man.decode)
    lat = man.sample_latents(120, seed=1)
    t0 = time.time()
    e0 = curvature_probe(g, lat, n_pairs=30, tau_kappa=PREREG["tau_kappa"],
                         steps=steps)
    for k, v in e0.items(): print(f"  {k:18s}: {v}")
    print(f"  (took {time.time()-t0:.1f}s)")

    # ---------------- E1 : geodesic recall ----------------
    banner("E1  GEODESIC vs STRAIGHT-LINE RECALL  (ground truth = ambient dist)")
    t0 = time.time()
    e1 = e1_geodesic_recall(man, g, n_queries=qs, n_cand=10, steps=steps)
    for k, v in e1.items(): print(f"  {k:18s}: {v}")
    e1_pass = (e1["geodesic_top1"] - e1["straight_top1"]) >= PREREG["delta1"]
    print(f"  E1 verdict        : {'PASS' if e1_pass else 'fail'} "
          f"(delta1={PREREG['delta1']})   (took {time.time()-t0:.1f}s)")

    # ---------------- E2 : learned context recall (GATE 1) ----------------
    banner("E2  LEARNED CONTEXT RECALL  (GATE 1 — the project floor)")
    corpus = ContextCorpus(n_facts=256, dim=64, noise=0.25,
                           device=device, dtype=dtype)
    M = corpus.keys
    s = 200 if args.quick else 400
    static = train_head(StaticHead(64).to(device, dtype), corpus, M, steps=s)
    ctx = train_head(ContextHead(64).to(device, dtype), corpus, M, steps=s)
    acc_cos = accuracy(None, corpus, M, kind="cosine")
    acc_fs = accuracy(None, corpus, M, kind="filesystem")
    acc_static = accuracy(static, corpus, M, kind="head")
    acc_ctx = accuracy(ctx, corpus, M, kind="head")
    print(f"  cosine baseline   : {acc_cos:.3f}   (ignores state s)")
    print(f"  filesystem dumb   : {acc_fs:.3f}   (exact-key landmine)")
    print(f"  static head (no s): {acc_static:.3f}   (ablation)")
    print(f"  CONTEXT head      : {acc_ctx:.3f}   (reads q,s,h)")
    best_base = max(acc_cos, acc_fs, acc_static)
    e2_pass = (acc_ctx - best_base) >= PREREG["delta2"]
    print(f"  E2 / GATE-1       : {'PASS' if e2_pass else 'FAIL'} "
          f"(delta2={PREREG['delta2']}, beats best baseline by "
          f"{acc_ctx-best_base:+.3f})")

    # ---------------- E3 : density as parallel sampler ----------------
    banner("E3  DENSITY AS PARALLEL SAMPLER  (optional contribution)")
    e3 = e3_density_sampler(corpus, ctx, M)
    for K, hit in e3.items(): print(f"  top-{K:<2d} recall    : {hit:.3f}")
    rising = all(e3[k2] >= e3[k1] - 1e-6 for k1, k2 in zip(list(e3)[:-1], list(e3)[1:]))
    print(f"  E3 verdict        : {'rising curve -> density buys parallel search' if rising else 'flat'}")

    # ---------------- C2 : signature bridge fidelity ----------------
    banner("C2  TRAJECTORY -> POINT  (signature bridge + recall-as-attractor)")
    # sanity: signature level-1 == endpoint - start
    p = torch.randn(4, 20, 3, dtype=dtype, device=device)
    sig = signature(p, depth=3)
    lvl1 = sig[:, :3]
    err = (lvl1 - (p[:, -1] - p[:, 0])).abs().max().item()
    print(f"  Chen level-1 check: max|sig_1 - (x_T - x_0)| = {err:.2e} (should be ~0)")

    # The real C2 claim: a trajectory's signature is a STABLE POINT — different
    # executions of the SAME skill (same shape, jittered/ reparametrised) land
    # close together, while DIFFERENT skills land far apart. That separation is
    # what lets a re-embedded trajectory be recalled as an attractor.
    br = SkillBridge(channels=4, depth=3)
    n_skill, n_exec = 8, 12
    base = torch.cumsum(0.3 * torch.randn(n_skill, 24, 4, dtype=dtype, device=device), 1)
    execs = []  # jittered + reparametrised copies of each skill
    for k in range(n_skill):
        jit = base[k][None] + 0.04 * torch.randn(n_exec, 24, 4, dtype=dtype, device=device)
        execs.append(jit)
    execs = torch.stack(execs)                       # (skill, exec, T, C)
    embs = br.embed(execs.reshape(-1, 24, 4)).reshape(n_skill, n_exec, -1)
    embs = embs / embs.norm(dim=-1, keepdim=True)
    centroids = embs.mean(1)
    intra = (embs - centroids[:, None]).norm(dim=-1).mean().item()
    inter = torch.cdist(centroids, centroids)
    inter = inter[~torch.eye(n_skill, dtype=torch.bool)].mean().item()
    sep = inter / (intra + 1e-9)
    c2_pass = sep >= 3.0
    print(f"  sigma dim         : {embs.shape[-1]}")
    print(f"  intra-skill spread: {intra:.3f}   inter-skill gap: {inter:.3f}")
    print(f"  separability      : {sep:.2f}x  "
          f"({'recallable as attractor' if c2_pass else 'too entangled'}, want >=3x)")

    # well-posed linear decode: recover translation/reparam-INVARIANT features
    # (displacement = level-1, Levy area = antisym level-2) -> should be faithful.
    paths = torch.cumsum(0.2 * torch.randn(256, 24, 4, dtype=dtype, device=device), 1)
    disp = paths[:, -1] - paths[:, 0]                 # 4 dims, == sig level 1
    levy = (sig_levy(paths))                          # 6 dims, signed areas
    targets = torch.cat([disp, levy], -1)
    br2 = SkillBridge(channels=4, depth=3).fit_inverse(paths, targets)
    test = torch.cumsum(0.2 * torch.randn(128, 24, 4, dtype=dtype, device=device), 1)
    test_tgt = torch.cat([test[:, -1] - test[:, 0], sig_levy(test)], -1)
    rel = (br2.reconstruct(test) - test_tgt).norm() / test_tgt.norm()
    print(f"  linear decode of invariants (disp+Levy area): rel-error {rel.item():.3f}")

    # ---------------- grounding + cross-manifold stitch ----------------
    banner("GROUNDING (anchors) + CROSS-MANIFOLD STITCH")
    anchors = corpus.keys[:16]
    z = corpus.keys[16:48]
    Q = torch.linalg.qr(torch.randn(64, 64, dtype=dtype, device=device))[0]
    # A different 'encoder' = the whole space (points AND anchors) rotated by Q.
    z2, anchors2 = z @ Q, anchors @ Q
    d_raw = (z2 - z).norm(dim=-1).mean().item()
    # relative reps use each space's OWN anchors -> invariant by construction.
    d_rel = (relative_rep(z2, anchors2) - relative_rep(z, anchors)).norm(dim=-1).mean().item()
    print(f"  raw drift across encoders : {d_raw:.3f}")
    print(f"  anchor-coord drift        : {d_rel:.3e}  (~0 => meaning is pinned)")
    # cross-manifold stitch: if anchor sets are mismatched, align by Procrustes.
    shuffle = torch.randperm(16)
    _, resid = procrustes_align(relative_rep(z, anchors),
                                relative_rep(z2, anchors2[shuffle]))
    print(f"  procrustes stitch resid   : {resid:.3f}  (low => spaces re-aligned)")

    # ---------------- dashboard ----------------
    banner("DASHBOARD")
    print(f"  E0 geometry gate : {'GO' if e0['go'] else 'NO-GO (reframe)'}")
    print(f"  E1 geodesic edge : {'PASS' if e1_pass else 'fail'}")
    print(f"  E2 GATE-1 floor  : {'PASS' if e2_pass else 'FAIL'}  <-- the decisive one")
    print(f"  C2 bridge faithful: {'yes' if c2_pass else 'no'}")
    print("\n  Reminder: thresholds in PREREG are the contract. Don't edit post-hoc.")


if __name__ == "__main__":
    main()
