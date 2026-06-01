"""
TRAVERSE — the geodesic operator.

We do NOT integrate the geodesic ODE with Christoffel symbols (that needs
derivatives of G => fragile double-backward on MPS). Instead we use the
variational definition directly: a geodesic minimises curve energy

    E[gamma] = sum_t  <Delta_t, Delta_t>_{z_t}              (discrete energy)

over the interior control points of a piecewise-linear curve with fixed
endpoints. This is exactly stochman's approach, needs only FORWARD evaluations
of G plus first-order autograd, and runs cleanly on MPS in float32.

Returned length uses the Riemannian length functional
    L[gamma] = sum_t sqrt(<Delta_t, Delta_t>_{midpoint_t}).
"""
from __future__ import annotations
import torch


def _curve_energy(metric, z0, z1, interior):
    """interior: (T-1, d) free points. Endpoints fixed. Returns scalar energy."""
    pts = torch.cat([z0[None], interior, z1[None]], dim=0)  # (T+1, d)
    deltas = pts[1:] - pts[:-1]                              # (T, d)
    mids = 0.5 * (pts[1:] + pts[:-1])                        # (T, d)
    # <Delta, Delta>_{mid} for each segment, summed
    sq = metric.inner(mids, deltas, deltas)                 # (T,)
    return sq.sum()


def geodesic(metric, z0, z1, n_segments=16, steps=150, lr=5e-2):
    """Solve for the energy-minimising curve from z0 to z1.
    Returns (points (T+1,d), length scalar)."""
    z0 = z0.detach(); z1 = z1.detach()
    T = n_segments
    t = torch.linspace(0, 1, T + 1, device=z0.device, dtype=z0.dtype)[1:-1, None]
    interior = (z0[None] + t * (z1 - z0)[None]).clone().requires_grad_(True)
    opt = torch.optim.Adam([interior], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        E = _curve_energy(metric, z0, z1, interior)
        E.backward()
        opt.step()
    with torch.no_grad():
        pts = torch.cat([z0[None], interior.detach(), z1[None]], dim=0)
        deltas = pts[1:] - pts[:-1]
        mids = 0.5 * (pts[1:] + pts[:-1])
        length = metric.speed(mids, deltas).sum()
    return pts, length


def straight_length(metric, z0, z1, n_segments=16):
    """Length of the straight latent line under the SAME metric (the cosine/linear
    baseline 'sees' the manifold as flat, but we still measure honestly)."""
    T = n_segments
    t = torch.linspace(0, 1, T + 1, device=z0.device, dtype=z0.dtype)[:, None]
    pts = z0[None] + t * (z1 - z0)[None]
    deltas = pts[1:] - pts[:-1]
    mids = 0.5 * (pts[1:] + pts[:-1])
    return metric.speed(mids, deltas).sum()
