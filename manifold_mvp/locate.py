"""
LAYER 2 — LOCATE as a density, not a point.

Classical retrieval: zhat = argmax_m cos(q, m).
Ours: a density over the manifold,
    rho_theta(z | q,s,h) ∝ exp(-E(z | q,s,h)),
with the modern-Hopfield energy whose landscape is SHAPED BY THE INPUT:
    E(z) = -beta^{-1} logsumexp_m( beta <phi(q,s,h), m> ) + 1/2 ||z||^2.
One Hopfield update = attention = softmax(beta * phi M^T) M, which is the mean
of rho_theta. The novelty is not the energy form (Ramsauer 2020) but that the
state s and history h enter phi, so the SAME cue q descends into DIFFERENT
basins (the 'input shapes the landscape' result, turned into a design lever).

This module also provides the protocol's four E2 contenders:
  cosine_topk      — baseline, ignores s
  filesystem       — exact-key lookup landmine (dumb storage)
  StaticHead       — learns phi(q) only (ablation: no s,h)
  ContextHead      — learns phi(q,s,h) (the thesis)
"""
from __future__ import annotations
import torch
import torch.nn as nn


# ---- analytic Locate: density + one-step Hopfield retrieval -----------------

def hopfield_retrieve(phi, M, beta=8.0):
    """phi:(B,dim) query field, M:(N,dim) stored patterns. Returns (weights, zhat)
    where weights = rho_theta over memories (a density!), zhat = its mean."""
    logits = beta * (phi @ M.T)              # (B, N)
    w = torch.softmax(logits, dim=-1)        # density over stored memories
    zhat = w @ M                             # Frechet/Euclidean mean of rho_theta
    return w, zhat


def locate_density(phi, M, beta=8.0):
    """Just the density rho_theta(.|q,s,h) over the N stored memories."""
    return torch.softmax(beta * (phi @ M.T), dim=-1)


# ---- E2 baselines -----------------------------------------------------------

def cosine_topk(q, M, k=1):
    qn = q / q.norm(dim=-1, keepdim=True)
    Mn = M / M.norm(dim=-1, keepdim=True)
    sims = qn @ Mn.T
    return sims.topk(k, dim=-1).indices

def filesystem_lookup(q, keys, tol=0.15):
    """Exact-ish key match: returns the stored key only if q is within tol,
    else -1 (a miss). Models 'dumb storage' that dies under query noise."""
    d = torch.cdist(q, keys)                 # (B, N)
    best = d.argmin(dim=-1)
    hit = d.gather(1, best[:, None]).squeeze(1) <= tol
    return torch.where(hit, best, torch.full_like(best, -1))


# ---- learned recall heads ---------------------------------------------------

class StaticHead(nn.Module):
    """phi(q): ignores state/history (ablation)."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
    def forward(self, q, s=None, h=None):
        phi = self.net(q)
        return phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-9)


class ContextHead(nn.Module):
    """phi(q,s,h): the cognitive-state-conditioned field. s reshapes the basin."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
    def forward(self, q, s, h=None):
        if h is None:
            h = torch.zeros_like(q)
        phi = self.net(torch.cat([q, s, h], dim=-1))
        return phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def train_head(head, corpus, M, beta=8.0, steps=400, lr=2e-3, m=512, seed=11):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for it in range(steps):
        q, s, tgt = corpus.query_batch(m=m, seed=seed + it)
        phi = head(q, s)
        logits = beta * (phi @ M.T)
        loss = nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


@torch.no_grad()
def accuracy(head_or_fn, corpus, M, beta=8.0, m=2000, seed=99, kind="head"):
    q, s, tgt = corpus.query_batch(m=m, seed=seed)
    if kind == "head":
        phi = head_or_fn(q, s)
        pred = (beta * (phi @ M.T)).argmax(-1)
    elif kind == "cosine":
        pred = cosine_topk(q, M, k=1).squeeze(-1)
    elif kind == "filesystem":
        pred = filesystem_lookup(q, corpus.keys)
    return (pred == tgt).float().mean().item()
