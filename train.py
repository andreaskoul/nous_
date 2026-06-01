#!/usr/bin/env python3
"""
Train the geometric-memory model end to end (plan stages 2-4).

Composite loss  L = w_ret*L_ret + w_geo*L_geo + w_ent*L_ent + w_gnd*L_gnd, with a
staged curriculum, plus a PARALLEL C2 sigma^-1 trained by reconstruction:

  Stage A  recall head only, metric FROZEN            -> establishes Gate-1
  Stage B  unfreeze lambda_theta, anneal in L_geo     -> learnable warped metric
  C2       train NeuralInverse on smooth skills        -> invertible bridge

Runs on the synthetic world (no network) by default; --data locomo swaps in the
real corpus. Saves a checkpoint + a metrics log for eval.py.

Usage:
    python train.py                       # synthetic, auto device
    python train.py --quick               # fast smoke
    python train.py --data locomo         # real corpus (needs sentence-transformers + data)
"""
from __future__ import annotations
import argparse, json, math, os, time
from dataclasses import asdict
import torch

from config import Config
from manifold_mvp import device as dev
from manifold_mvp.synthetic import CurvedManifold, ContextCorpus
from manifold_mvp.metric import PullbackMetric
from manifold_mvp.locate import ContextHead, StaticHead, locate_density, train_head
from manifold_mvp.conformal import ConformalHead
from manifold_mvp.signature import signature
from manifold_mvp import losses, invert


def build_data(cfg, split, device, dtype, embed_fn=None, pca=None):
    """Returns (corpus, manifold, d_latent). For synthetic these are two distinct
    objects (ContextCorpus + CurvedManifold); for LoCoMo one object plays both."""
    if cfg.data.kind == "synthetic":
        corpus = ContextCorpus(n_facts=cfg.data.n_facts, dim=cfg.data.corpus_dim,
                               noise=cfg.data.noise, device=device, dtype=dtype)
        manifold = CurvedManifold(d_latent=2, D=32, warp=2.0, freq=2.5,
                                  device=device, dtype=dtype)
        return corpus, manifold, 2
    from manifold_mvp.real import LoCoMoData, make_sentence_encoder
    ef = embed_fn or make_sentence_encoder(cfg.data.encoder)
    data = LoCoMoData(cfg.data.locomo_json, split=split,
                      conv_ids=cfg.data.convs_for(split), embed_fn=ef,
                      pca_dim=cfg.data.pca_dim, emb_cache=cfg.data.emb_cache,
                      device=device, dtype=dtype, pca=pca)
    return data, data, data.pca.d


def geo_minibatch(manifold, bs, seed, device, dtype):
    """On-manifold latents + GAGA-style off-manifold negatives (perturbed far)
    + a pair with a target ambient distance."""
    z_on = manifold.sample_latents(bs, seed=seed)
    scale = (3.0 * z_on.std()).clamp_min(1e-3)
    gn = torch.Generator().manual_seed(seed + 7)
    z_off = z_on + scale * torch.randn(z_on.shape, generator=gn).to(device, dtype)
    a = manifold.sample_latents(bs, seed=seed + 2)
    b = manifold.sample_latents(bs, seed=seed + 3)
    return z_on, z_off, a, b, manifold.ambient_dist(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "locomo"])
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--out", default="checkpoints/ckpt.pt")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.kind = args.data
    if args.quick:
        cfg.train.steps_warmup = 80
        cfg.train.steps_joint = 80
        cfg.train.steps_c2 = 120
        cfg.train.batch = 256

    device = dev.pick_device(args.device)
    device, dtype = dev.resolve(device, torch.float32)
    torch.manual_seed(cfg.train.seed)
    print(f"Device: {dev.describe(device)} | data: {cfg.data.kind}")

    corpus, manifold, d_latent = build_data(cfg, "train", device, dtype)
    M = corpus.keys
    N, dim = M.shape
    beta = cfg.model.beta
    h_floor = cfg.prereg.entropy_floor_frac * math.log(N)
    anchors = M[: min(16, N)]
    geo_bs = min(48, cfg.train.batch)            # Jacobians are pricey -> small batch

    head = ContextHead(dim, hidden=cfg.model.hidden).to(device, dtype)
    lam = ConformalHead(d_latent, hidden=cfg.model.conformal_hidden).to(device, dtype)
    metric = PullbackMetric(manifold.decode, conformal=lam)
    opt_head = torch.optim.Adam(head.parameters(), lr=cfg.train.lr)
    opt_metric = torch.optim.Adam(lam.parameters(), lr=cfg.train.lr_metric)

    logs = []

    def recall_parts(it):
        q, s, tgt = corpus.query_batch(m=cfg.train.batch, seed=cfg.train.seed + it)
        phi = head(q, s)
        neg = (losses.mine_hard_negatives(phi.detach(), M, tgt, k=cfg.train.n_hard_neg)
               if cfg.train.n_hard_neg > 0 else None)
        parts = {"ret": losses.retrieval_infonce(phi, M, tgt, beta=beta, neg_idx=neg)}
        rho = locate_density(phi, M, beta=beta)
        parts["ent"] = losses.entropy_floor(rho, h_floor)
        gi = torch.randint(0, N, (cfg.train.batch,))
        parts["gnd"] = losses.grounding_loss(M[gi], anchors)
        acc = (phi.detach() @ M.T).argmax(-1).eq(tgt).float().mean().item()
        return parts, rho.detach(), acc

    # ---------------- Stage A: recall head, frozen metric ----------------
    t0 = time.time()
    for it in range(cfg.train.steps_warmup):
        parts, rho, acc = recall_parts(it)
        loss = losses.composite_loss(parts, cfg.weights)     # no 'geo' key yet
        opt_head.zero_grad(); loss.backward(); opt_head.step()
        if it % max(1, cfg.train.steps_warmup // 4) == 0 or it == cfg.train.steps_warmup - 1:
            H = -(rho.clamp_min(1e-12) * rho.clamp_min(1e-12).log()).sum(-1).mean().item()
            logs.append(dict(stage="A", it=it, ret=parts["ret"].item(),
                             ent=parts["ent"].item(), top1=acc, entropy=H))
            print(f"  A {it:4d} | ret {parts['ret'].item():.3f} | top1 {acc:.3f} "
                  f"| H(rho) {H:.2f} (floor {h_floor:.2f})")

    # ---------------- Stage B: + learnable metric (anneal L_geo) ----------------
    for it in range(cfg.train.steps_joint):
        parts, rho, acc = recall_parts(cfg.train.steps_warmup + it)
        anneal = min(1.0, (it + 1) / max(1, cfg.train.steps_joint // 2))
        z_on, z_off, a, b, d_tgt = geo_minibatch(manifold, geo_bs,
                                                 cfg.train.seed + 10_000 + it, device, dtype)
        l_geo, ginfo = losses.geometry_loss(metric, z_on, z_off, a, b, d_tgt)
        parts["geo"] = anneal * l_geo
        loss = losses.composite_loss(parts, cfg.weights)
        opt_head.zero_grad(); opt_metric.zero_grad()
        loss.backward()
        opt_head.step(); opt_metric.step()
        if it % max(1, cfg.train.steps_joint // 4) == 0 or it == cfg.train.steps_joint - 1:
            with torch.no_grad():
                lam_mean = lam(manifold.sample_latents(64, seed=it)).mean().item()
            logs.append(dict(stage="B", it=it, ret=parts["ret"].item(),
                             geo=l_geo.item(), top1=acc, lam_mean=lam_mean, **ginfo))
            print(f"  B {it:4d} | ret {parts['ret'].item():.3f} | top1 {acc:.3f} "
                  f"| geo {l_geo.item():.3f} (warp {ginfo['warp']:.2f}, "
                  f"match {ginfo['dist_match']:.2f}) | lambda~{lam_mean:.2f}")

    # ---------------- C2 parallel: train sigma^-1 over the skill repertoire ----
    C, T, depth = cfg.model.c2_channels, cfg.model.c2_steps, cfg.model.c2_depth
    sampler, _ = invert.repertoire_sampler(C, T, n_proto=cfg.model.c2_n_proto,
                                           jitter=cfg.model.c2_jitter, seed=cfg.model.c2_seed)
    sig_dim = signature(torch.zeros(1, T, C, dtype=dtype), depth).shape[-1]
    inv = invert.NeuralInverse(sig_dim, T, C, hidden=256).to(device, dtype)
    inv = invert.train_inverse(inv, sampler, steps=cfg.train.steps_c2, lr=2e-3,
                               depth=depth, batch=128, device=device, dtype=dtype)
    # held-out executions of the SAME repertoire (distinct exec seed)
    c2_rel = invert.reconstruction_relerr(inv, sampler(256, 9_999_999).to(device, dtype), depth)
    print(f"  C2 sigma^-1 trained | held-out recon rel-err {c2_rel:.3f} "
          f"(gate <= {cfg.prereg.c2_max_recon})")
    print(f"  total train time {time.time() - t0:.1f}s")

    # ---------------- static-head ablation baseline (E2 contender) ----------
    # plain cross-entropy == full-bank InfoNCE; reuse the existing trainer.
    static = train_head(StaticHead(dim, hidden=cfg.model.hidden).to(device, dtype),
                        corpus, M, beta=beta, steps=cfg.train.steps_warmup)
    print("  static-head ablation trained (no state s)")

    # ---------------- checkpoint ----------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = dict(cfg=asdict(cfg), head=head.state_dict(), static=static.state_dict(),
                lam=lam.state_dict(), inv=inv.state_dict(), sig_dim=sig_dim,
                dim=dim, d_latent=d_latent)
    if cfg.data.kind == "locomo":
        ckpt["pca"] = dict(V=manifold.pca.V, mean=manifold.pca.mean, d=manifold.pca.d)
    torch.save(ckpt, args.out)
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w") as f:
        json.dump(logs, f, indent=2)
    print(f"  saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
