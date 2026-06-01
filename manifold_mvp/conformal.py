"""
LAYER 3.2 (learnable) — the conformal warp lambda_theta(z).

The frozen pullback gives G0(z) = J_g(z)^T J_g(z). We make the metric LEARNABLE
with a conformal factor:

    G(z) = lambda_theta(z) * G0(z),   lambda_theta(z) > 0.

This is the cheapest learnable-metric route the formalism recommends and the
GAGA (2410.12779) lever: with a *linear* PCA chart, J_g is constant => G0 is flat,
so lambda_theta(z) is the ONLY thing that makes the metric vary across space
(i.e. produces curvature). softplus keeps lambda strictly positive => G stays SPD.

Drop-in: metric.PullbackMetric already accepts a `conformal` callable, so

    PullbackMetric(decode_fn, conformal=ConformalHead(d_latent))

needs no structural change to the existing metric code.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus(0.5413) ~= 1.0  -> initialise lambda ~ 1 so training starts at the
# frozen analytic metric and only *warps* it from there.
_INIT_BIAS = 0.5413


class ConformalHead(nn.Module):
    def __init__(self, d_latent: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        with torch.no_grad():               # start near lambda == 1 (~frozen metric)
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.fill_(_INIT_BIAS)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z:(B, d) -> lambda:(B,) strictly positive."""
        return F.softplus(self.net(z).squeeze(-1)) + 1e-4
