"""
The composite training objective (plan stage 3).

Declarative model:
    L = w_ret*L_ret + w_geo*L_geo + w_ent*L_ent + w_gnd*L_gnd

  L_ret  InfoNCE retrieval. Full-bank InfoNCE == cross-entropy over all memories
         (every non-target is a negative; the softmax is dominated by the HARD
         negatives automatically). Optionally restrict the denominator to mined
         hard negatives (ANCE 2007.00808 / NV-Retriever 2407.15831). This is the
         gradient of the modern-Hopfield logsumexp energy (Hopfield-Fenchel-Young
         2411.08590), so the recall head and the energy are the same object.
  L_geo  GAGA-style (2410.12779): make off-manifold regions metrically EXPENSIVE
         (so geodesics stay on-manifold) + match first-order geodesic length to a
         target semantic distance. Trains lambda_theta.
  L_ent  entropy floor: keep the retrieval density rho_theta from collapsing to a
         point (a named risk in the register).
  L_gnd  anchor-coordinate stability under nuisance noise (Moschella-Rodola
         relative reps) so meaning cannot drift.

The C2 reconstruction term L_C2 is a *parallel* objective with its own optimizer
and gate; it lives in invert.py, not in this sum.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .relative import relative_rep


# ---- L_ret : InfoNCE retrieval ----------------------------------------------

def retrieval_infonce(phi, M, target, beta=8.0, neg_idx=None):
    """phi:(B,dim) query field, M:(N,dim) memory bank, target:(B,) gold index.
    neg_idx:(B,k) optional mined hard negatives -> restrict the denominator."""
    if neg_idx is None:
        return F.cross_entropy(beta * (phi @ M.T), target)        # full-bank
    pos = (phi * M[target]).sum(-1, keepdim=True)                 # (B,1)
    negs = torch.einsum("bd,bkd->bk", phi, M[neg_idx])            # (B,k)
    logits = beta * torch.cat([pos, negs], dim=-1)                # col 0 = positive
    tgt = torch.zeros(phi.shape[0], dtype=torch.long, device=phi.device)
    return F.cross_entropy(logits, tgt)


@torch.no_grad()
def mine_hard_negatives(phi, M, target, k=16):
    """Top-k highest-similarity WRONG memories per query (ANCE-style)."""
    sims = phi @ M.T                                              # (B,N)
    sims.scatter_(1, target[:, None], float("-inf"))             # mask positive
    return sims.topk(min(k, sims.shape[1] - 1), dim=-1).indices  # (B,k)


# ---- L_geo : GAGA-style warp + local distance-matching -----------------------

def geometry_loss(metric, z_on, z_off, pairs_a, pairs_b, d_target, margin=1.0):
    """(a) warp: metric volume (log det G) should be SMALLER on-manifold than off
    -manifold by at least `margin`, so traversing off-manifold is costly and
    geodesics hug the data. (b) distance-match: the first-order squared geodesic
    dz^T G(mid) dz between a pair matches the target semantic distance^2 (no inner
    geodesic solve needed -> cheap & differentiable; the full solve is reserved
    for E0/E1 evaluation)."""
    logdet_on = torch.logdet(metric.metric(z_on)).clamp(-30, 30)
    logdet_off = torch.logdet(metric.metric(z_off)).clamp(-30, 30)
    warp = F.softplus(margin + logdet_on.mean() - logdet_off.mean())

    mid = 0.5 * (pairs_a + pairs_b)
    dz = pairs_b - pairs_a
    d_local = metric.speed(mid, dz)                               # sqrt(dz^T G dz)
    match = F.mse_loss(d_local, d_target)
    return warp + match, {"warp": warp.item(), "dist_match": match.item()}


# ---- L_ent : entropy floor (anti-collapse) -----------------------------------

def entropy_floor(rho, h_floor):
    """rho:(B,N) density over memories. One-sided penalty when per-query entropy
    drops below h_floor (nats)."""
    p = rho.clamp_min(1e-12)
    H = -(p * p.log()).sum(-1)                                    # (B,)
    return F.relu(h_floor - H).mean()


# ---- L_gnd : anchor-coordinate stability (anti-drift) ------------------------

def grounding_loss(z, anchors, noise=0.05):
    """Anchor-similarity coords r(z)=[cos(z,a_k)] should be stable under a small
    nuisance perturbation -> pins meaning (Moschella-Rodola)."""
    z2 = z + noise * torch.randn_like(z)
    return (relative_rep(z2, anchors) - relative_rep(z, anchors)).norm(dim=-1).mean()


# ---- weighted sum ------------------------------------------------------------

def composite_loss(parts: dict, weights):
    """parts: name -> scalar tensor. weights: LossWeights (.ret/.geo/.ent/.gnd).
    Missing terms are simply skipped (curriculum: geo/gnd can be off early)."""
    total = torch.zeros((), device=next(iter(parts.values())).device)
    for name in ("ret", "geo", "ent", "gnd"):
        if name in parts and parts[name] is not None:
            total = total + getattr(weights, name) * parts[name]
    return total
