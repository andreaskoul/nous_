"""
LAYER 0 / 3.2 — the metric.  Analytic, not learned (the hybrid backbone the
formalism recommends: pullback of a frozen decoder + an optional conformal
correction lambda(z)).

    G(z) = J_g(z)^T J_g(z)            (Shao-Kumar-Fletcher 2018)
    <u,v>_z = u^T G(z) v
    G_conf(z) = lambda(z) * G(z)      (Arvanitidis-style conformal factor)

Jacobians are taken with torch.func (vmap + jacrev). On MPS a couple of the
underlying ops can fall back to CPU but the math is identical; for the E0 probe
we run on CPU/float64 anyway. Everything here is differentiable, so the same
code path supports a *learned* metric later (swap g for g_theta, or make lambda
a small MLP) with no structural change.
"""
from __future__ import annotations
import torch
from torch.func import jacrev, vmap


class PullbackMetric:
    def __init__(self, decode_fn, conformal=None, eps=1e-6):
        """decode_fn: g: (d,) -> (D,) for a SINGLE point (vmap handles batching).
        conformal: optional lambda(z)->scalar>0 multiplying the whole metric."""
        self.g = decode_fn
        self.conformal = conformal
        self.eps = eps
        self._jac = jacrev(self.g)           # (d,) -> (D, d)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        return vmap(self._jac)(z)            # (B, D, d)

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """G(z): (B, d, d), SPD."""
        J = self.jacobian(z)                 # (B, D, d)
        G = J.transpose(-1, -2) @ J          # (B, d, d)
        d = G.shape[-1]
        G = G + self.eps * torch.eye(d, device=G.device, dtype=G.dtype)
        if self.conformal is not None:
            lam = self.conformal(z).reshape(-1, 1, 1).clamp_min(self.eps)
            G = lam * G
        return G

    def inner(self, z, u, v) -> torch.Tensor:
        """<u,v>_z for batched tangent vectors u,v at z. -> (B,)"""
        G = self.metric(z)
        return torch.einsum("bi,bij,bj->b", u, G, v)

    def speed(self, z, v) -> torch.Tensor:
        """sqrt(<v,v>_z), the length element along a velocity v."""
        return torch.sqrt(self.inner(z, v, v).clamp_min(0.0) + self.eps)
