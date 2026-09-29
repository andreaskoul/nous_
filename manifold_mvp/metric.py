"""
EXPLORATORY (v2) — the pullback metric. Not in the retrieval path.

    G(z) = J_g(z)^T J_g(z)            (Shao-Kumar-Fletcher 2018)
    <u,v>_z = u^T G(z) v
    G_conf(z) = lambda(z) * G(z)      (Arvanitidis-style conformal factor)

Status after the September 2026 reality check (docs/REALITY_CHECK_2026-09.md):
on a real encoder the chart g is a LINEAR PCA map, so J is constant and
G0 = V^T V = I; a learned conformal lambda trained to match chart distances then
collapses to 1 (measured 1.01 +/- 0.07) and geodesics equal straight lines, which
lost to full-dimension cosine. Pullback geodesics through linear or same-dimension
decoders are straight (arXiv 2505.17517). Kept only for the exploratory E1' test,
which must beat identity, whitening and random-metric controls.

Jacobians are taken with torch.func (vmap + jacrev).
"""
from __future__ import annotations
import torch
from torch.func import jacrev, vmap


class TorchPCA:
    """Linear chart x in R^D <-> z in R^d; decode(z) = z V^T + mean is the chart g.
    Because V has orthonormal columns, its pullback metric is exactly the identity."""

    def __init__(self, d: int):
        self.d = d
        self.mean = None
        self.V = None          # (D, d), orthonormal columns

    def fit(self, X: torch.Tensor):
        self.mean = X.mean(0)                            # (D,) so decode((d,)) -> (D,) for jacrev
        _, _, Vh = torch.linalg.svd(X - self.mean, full_matrices=False)
        self.V = Vh[: self.d].T.contiguous()             # (D, d)
        return self

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) @ self.V                  # (.., d)

    def decode(self, Z: torch.Tensor) -> torch.Tensor:
        return Z @ self.V.T + self.mean                  # (.., D)


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
