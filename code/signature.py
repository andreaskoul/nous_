"""
C2 — THE BRIDGE (your novel core).  Turn a trajectory (a 'skill', a reasoning
path) into a single POINT that can be Located and recalled like a fact.

signatory is unusable on M4 (pinned to torch 1.8-1.11 / py3.7-3.9, no recent
macOS wheels, source build must match torch exactly). The truncated signature is
a closed-form Chen recursion, so we implement it directly in ~40 lines of pure
torch — fully differentiable, runs on MPS.

    sigma(gamma) = ( 1, ∫dgamma, ∫∫ dgamma⊗dgamma, ... )   truncated at depth K

Computed via Chen's identity: the signature of a concatenation is the tensor
product of signatures, and the signature of one straight segment with increment
x is exp_⊗(x) = ( 1, x, x⊗x/2!, ..., x^{⊗K}/K! ). We fold left-to-right.

C2's KILL CRITERION is the fidelity of sigma^{-1} ∘ sigma on held-out skills:
if from the point sigma(gamma) we can regenerate a trajectory that reproduces
the skill, declarative and procedural memory are one store under two read-heads.
We test the linear-decodability proxy here (fit sigma^{-1} by least squares to
recover trajectory summary stats), which is the cheapest honest version.
"""
from __future__ import annotations
import math
import torch


def _seg_exp(x: torch.Tensor, depth: int):
    """exp_⊗ of a single increment x:(B,C). Returns list level k -> (B, C^k)."""
    B, C = x.shape
    levels = [torch.ones(B, 1, device=x.device, dtype=x.dtype)]  # level 0
    term = torch.ones(B, 1, device=x.device, dtype=x.dtype)
    for k in range(1, depth + 1):
        # term_k = term_{k-1} ⊗ x / k   (gives x^{⊗k}/k!)
        term = torch.einsum("bi,bj->bij", term, x).reshape(B, -1) / k
        levels.append(term)
    return levels


def _chen_product(A, B, depth):
    """Tensor-algebra product of two truncated tensors (lists of levels)."""
    out = []
    for n in range(depth + 1):
        acc = 0.0
        for i in range(n + 1):
            j = n - i
            Bn, _ = A[i].shape
            Ci = round(A[i].shape[1] ** (1.0 / i)) if i > 0 else 1
            # outer product of level-i and level-j, summed into level-n
            prod = torch.einsum("bi,bj->bij", A[i], B[j]).reshape(Bn, -1)
            acc = prod if isinstance(acc, float) else acc + prod
        out.append(acc)
    return out


def signature(path: torch.Tensor, depth: int = 3) -> torch.Tensor:
    """path:(B, T, C) -> signature vector (B, C + C^2 + ... + C^depth).
    Differentiable. Level 1 == path[:,-1]-path[:,0] (sanity-checkable)."""
    B, T, C = path.shape
    incr = path[:, 1:] - path[:, :-1]              # (B, T-1, C)
    sig = _seg_exp(incr[:, 0], depth)
    for t in range(1, T - 1):
        sig = _chen_product(sig, _seg_exp(incr[:, t], depth), depth)
    # drop level 0 (constant 1), concat levels 1..depth
    return torch.cat(sig[1:], dim=-1)


def levy_area(path: torch.Tensor) -> torch.Tensor:
    """Signed (Levy) areas A_ij = 1/2 (S_ij - S_ji) for i<j, from level-2 of the
    signature. Translation- and reparam-invariant -> a well-posed decode target.
    path:(B,T,C) -> (B, C*(C-1)/2)."""
    B, T, C = path.shape
    sig = signature(path, depth=2)
    lvl2 = sig[:, C:C + C * C].reshape(B, C, C)
    out = []
    for i in range(C):
        for j in range(i + 1, C):
            out.append(0.5 * (lvl2[:, i, j] - lvl2[:, j, i]))
    return torch.stack(out, dim=-1)


class SkillBridge:
    """Embed trajectories as points; fit a linear inverse for the fidelity test."""
    def __init__(self, channels, depth=3, proj_dim=None, seed=0):
        self.depth = depth
        self.channels = channels
        g = torch.Generator().manual_seed(seed)
        # optional random projection of channels to keep C^depth small
        self.P = None
        if proj_dim is not None and proj_dim < channels:
            P = torch.randn(channels, proj_dim, generator=g)
            self.P = P / P.norm(dim=0, keepdim=True)
            self.channels = proj_dim
        self.W = None  # linear sigma^{-1}

    def embed(self, paths: torch.Tensor) -> torch.Tensor:
        if self.P is not None:
            paths = paths @ self.P.to(paths.device, paths.dtype)
        return signature(paths, self.depth)

    def fit_inverse(self, paths, targets):
        """Least-squares sigma^{-1}: signature -> trajectory summary (targets)."""
        S = self.embed(paths)
        S1 = torch.cat([S, torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype)], -1)
        self.W = torch.linalg.lstsq(S1, targets).solution
        return self

    def reconstruct(self, paths):
        S = self.embed(paths)
        S1 = torch.cat([S, torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype)], -1)
        return S1 @ self.W
