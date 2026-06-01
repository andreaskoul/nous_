"""
C2 — a genuinely invertible sigma^{-1} (the kill criterion).

run_mvp.py's SkillBridge fits a LINEAR map onto translation/reparam-INVARIANTS
(displacement + Levy area). That is faithful for those summaries but is not a
path inverse. This module provides the upgrade the plan calls for:

 1) NeuralInverse: a learned decoder  sigma(gamma) -> increments -> path, trained
    by reconstruction MSE (Kidger-style neural inversion). This is what L_C2
    trains and is the operational "regenerate the trajectory" test.
 2) LinearSignatureInverse: the principled INSERTION METHOD baseline (Chang &
    Lyons, arXiv 2304.01862). The insertion method shows each path-derivative
    slot is recoverable from the next signature level by LINEAR least-squares;
    a single global least-squares map sigma -> increments is the simplest
    consistent realization and needs NO gradient training.

Signatures are translation-invariant (they depend only on increments), so we
recover the path SHAPE (increments) and re-anchor at the given start x0; this is
the honest content of "invert the signature".

Fidelity gate (kill criterion):
    reconstruction rel-error <= c2_max_recon AND skill separability >= c2_min_sep.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .signature import signature


def smooth_skill_sampler(channels, n_steps, n_modes=2, amp=0.6):
    """A 'skill' = a smooth low-frequency curve (a few Fourier modes), so it lives
    on a low-dim manifold a depth-K signature can actually capture/invert. This is
    the honest test substrate for the C2 kill criterion (random walks are NOT
    invertible from a truncated signature; structured skills are)."""
    t = torch.linspace(0.0, 1.0, n_steps)

    def sample(n, seed):
        g = torch.Generator().manual_seed(seed)
        # genuinely low frequency (<=1.5 oscillations over the window) so the path
        # is well-sampled and smooth -> increments small -> signature is invertible
        freq = 0.5 + 1.0 * torch.rand(n, channels, n_modes, generator=g)
        a = amp * torch.randn(n, channels, n_modes, generator=g)
        ph = 2 * math.pi * torch.rand(n, channels, n_modes, generator=g)
        # path[n,s,c] = sum_k a*sin(2pi f t + ph)
        arg = 2 * math.pi * freq[:, :, None, :] * t[None, None, :, None] + ph[:, :, None, :]
        path = (a[:, :, None, :] * torch.sin(arg)).sum(-1)                 # (n,C,S)
        return path.transpose(1, 2).contiguous()                          # (n,S,C)

    return sample


def repertoire_sampler(channels, n_steps, n_proto=12, jitter=0.03, n_modes=2, seed=0):
    """A finite LIBRARY of smooth skills + per-execution jitter. This is the
    honest C2 setting: skills are recalled from a repertoire (recall-as-attractor),
    so sigma^-1 only has to regenerate executions of KNOWN skills -> path
    reconstruction is well-posed and generalizes. Returns (sample_fn, prototypes)."""
    prototypes = smooth_skill_sampler(channels, n_steps, n_modes)(n_proto, seed)

    def sample(n, s):
        g = torch.Generator().manual_seed(s)
        idx = torch.randint(0, n_proto, (n,), generator=g)
        return prototypes[idx] + jitter * torch.randn(n, n_steps, channels, generator=g)

    return sample, prototypes


def _path_from_increments(incr, x0):
    """incr:(B,T-1,C), x0:(B,C) -> path:(B,T,C) anchored at x0."""
    start = x0[:, None]
    return torch.cat([start, start + torch.cumsum(incr, dim=1)], dim=1)


class NeuralInverse(nn.Module):
    """sigma(gamma):(B,S) -> increments (B,T-1,C) -> path (B,T,C). Trainable."""

    def __init__(self, sig_dim, n_steps, channels, hidden=256):
        super().__init__()
        self.n_steps, self.channels = n_steps, channels
        # signature levels span very different scales -> LayerNorm conditions the
        # input and makes optimization tractable.
        self.net = nn.Sequential(
            nn.LayerNorm(sig_dim),
            nn.Linear(sig_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, (n_steps - 1) * channels),
        )

    def forward(self, sig, x0):
        incr = self.net(sig).reshape(sig.shape[0], self.n_steps - 1, self.channels)
        return _path_from_increments(incr, x0)


class LinearSignatureInverse:
    """Insertion-method-style ANALYTIC inverse: one global least-squares map
    signature -> increments. No training; the closed-form linear solve."""

    def __init__(self, depth=3):
        self.depth, self.W, self.n_steps, self.C = depth, None, None, None

    def fit(self, paths):
        B, T, C = paths.shape
        self.n_steps, self.C = T, C
        S = signature(paths, self.depth)
        S1 = torch.cat([S, torch.ones(B, 1, device=S.device, dtype=S.dtype)], -1)
        incr = (paths[:, 1:] - paths[:, :-1]).reshape(B, -1)
        self.W = torch.linalg.lstsq(S1, incr).solution
        return self

    def reconstruct(self, paths):
        B = paths.shape[0]
        S = signature(paths, self.depth)
        S1 = torch.cat([S, torch.ones(B, 1, device=S.device, dtype=S.dtype)], -1)
        incr = (S1 @ self.W).reshape(B, self.n_steps - 1, self.C)
        return _path_from_increments(incr, paths[:, 0])


# ---- training + the kill-criterion tests ------------------------------------

def c2_reconstruction_loss(inv: NeuralInverse, paths, depth=3):
    sig = signature(paths, depth=depth)
    return F.mse_loss(inv(sig, x0=paths[:, 0]), paths)


def train_inverse(inv, sample_paths_fn, steps=400, lr=1e-3, depth=3, batch=128,
                  device="cpu", dtype=torch.float32, seed=0):
    """sample_paths_fn(batch, seed) -> (batch, T, C). Trains L_C2 by MSE."""
    opt = torch.optim.Adam(inv.parameters(), lr=lr)
    for it in range(steps):
        paths = sample_paths_fn(batch, seed + it).to(device, dtype)
        loss = c2_reconstruction_loss(inv, paths, depth)
        opt.zero_grad(); loss.backward(); opt.step()
    return inv


@torch.no_grad()
def reconstruction_relerr(inv, paths_test, depth=3):
    sig = signature(paths_test, depth=depth)
    recon = (inv(sig, x0=paths_test[:, 0]) if isinstance(inv, nn.Module)
             else inv.reconstruct(paths_test))
    return (recon - paths_test).norm().item() / (paths_test.norm().item() + 1e-9)


@torch.no_grad()
def skill_separability(embed_fn, skills_execs):
    """skills_execs:(n_skill, n_exec, T, C). inter/intra centroid ratio (>=3x good)."""
    n_skill, n_exec, T, C = skills_execs.shape
    embs = embed_fn(skills_execs.reshape(-1, T, C)).reshape(n_skill, n_exec, -1)
    embs = embs / embs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    centroids = embs.mean(1)
    intra = (embs - centroids[:, None]).norm(dim=-1).mean().item()
    inter = torch.cdist(centroids, centroids)
    inter = inter[~torch.eye(n_skill, dtype=torch.bool, device=embs.device)].mean().item()
    return inter / (intra + 1e-9)
