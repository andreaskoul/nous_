"""
GROUNDING (Open Q2) + CROSS-MANIFOLD (Layer 5) — relative representations.

Moschella & Rodola: represent a point by its similarity to a fixed ANCHOR set,
    r(z) = [ cos(z, a_1), ..., cos(z, a_K) ].
This is invariant to rotations/rescalings of the latent space (the anchors move
with it), which (a) pins meaning so it cannot drift, and (b) lets TWO different
manifolds communicate: if they share anchors, their relative reps are directly
comparable; if not, align anchor sets by orthogonal Procrustes.
"""
from __future__ import annotations
import torch


def relative_rep(z: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    zn = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    an = anchors / anchors.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return zn @ an.T                       # (B, K) anchor-similarity coords


def drift(z_before, z_after, anchors):
    """How far the anchor-coords moved (should be ~0 if grounding holds)."""
    rb = relative_rep(z_before, anchors)
    ra = relative_rep(z_after, anchors)
    return (ra - rb).norm(dim=-1).mean().item()


def procrustes_align(X, Y):
    """Best orthogonal R minimising ||X R - Y||. Returns R and residual."""
    U, _, Vt = torch.linalg.svd(X.T @ Y)
    R = U @ Vt
    resid = (X @ R - Y).norm().item() / (Y.norm().item() + 1e-9)
    return R, resid
