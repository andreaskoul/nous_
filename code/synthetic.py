"""
Synthetic worlds with a KNOWN ground truth, so every experiment can be scored
without a real benchmark. Two generators:

1) `CurvedManifold`: a smooth decoder g: R^d_latent -> R^D whose pullback metric
   G(z)=J^T J genuinely varies in space (sinusoidal warp) => non-zero curvature.
   This is what makes E0/E1 meaningful: the "true" semantic distance between two
   latents is the AMBIENT distance ||g(a)-g(b)||, which the pullback geodesic is
   supposed to recover and the straight latent line is not.

2) `ContextCorpus`: an ambiguous-query retrieval task where the correct memory
   depends on a state vector `s`. Cosine-on-q alone is at chance; a head that
   reads `s` can win. This is the E2 (Gate-1) arena and its filesystem baseline.

A real encoder (sentence-transformers) can be swapped in later via `real.py`;
the synthetic path exists so the whole pipeline runs offline on the M4 today.
"""
from __future__ import annotations
import torch


class CurvedManifold:
    """g(z) = R @ [ z ; A*sin(f z1)*sin(f z2) ; A*cos(f z1) ] padded to D, with R
    a fixed orthonormal lift. The sinusoidal bump makes J_g(z) (hence G) vary."""

    def __init__(self, d_latent=2, D=32, warp=1.5, freq=2.0, seed=0,
                 device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        self.d, self.D, self.A, self.f = d_latent, D, warp, freq
        self.device, self.dtype = device, dtype
        # fixed orthonormal lift from a small intrinsic space into R^D
        raw = torch.randn(D, d_latent + 2, generator=g)
        q, _ = torch.linalg.qr(raw)
        self.R = q[:, : d_latent + 2].to(device=device, dtype=dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., d) -> (..., D). Smooth & differentiable (autograd-friendly)."""
        z1, z2 = z[..., 0], z[..., 1]
        bump = self.A * torch.sin(self.f * z1) * torch.sin(self.f * z2)
        twist = self.A * torch.cos(self.f * z1)
        feat = torch.stack([z1, z2, bump, twist], dim=-1)  # (..., d+2)
        return feat @ self.R.T  # (..., D)

    def sample_latents(self, n, spread=2.0, seed=1) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        z = (torch.rand(n, self.d, generator=g) * 2 - 1) * spread
        return z.to(device=self.device, dtype=self.dtype)

    def ambient_dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Ground-truth 'semantic' distance = chordal distance in ambient space."""
        return torch.linalg.norm(self.decode(a) - self.decode(b), dim=-1)


class ContextCorpus:
    """N facts. Each fact has a key embedding. We build AMBIGUOUS queries: a query
    sits between two facts that share a 'topic' but differ by 'state'. The state
    vector s says which one is correct. Encodes the protocol's mandatory baselines:
      - cosine(q)        : ignores s        -> ~chance on ambiguous items
      - filesystem       : exact key match  -> fails under any query noise
      - static head      : learns from data but never reads s (ablation)
      - context head     : reads (q, s, h)  -> should win
    """

    def __init__(self, n_facts=256, dim=64, noise=0.25, seed=3,
                 device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(seed)
        self.dim, self.noise = dim, noise
        self.device, self.dtype = device, dtype
        # facts come in ambiguous PAIRS that share a topic direction but differ
        # along a state direction.
        n_pairs = n_facts // 2
        topic = torch.randn(n_pairs, dim, generator=g)
        topic = topic / topic.norm(dim=-1, keepdim=True)
        state_dir = torch.randn(n_pairs, dim, generator=g)
        state_dir = state_dir / state_dir.norm(dim=-1, keepdim=True)
        plus = topic + 0.6 * state_dir
        minus = topic - 0.6 * state_dir
        keys = torch.stack([plus, minus], dim=1).reshape(2 * n_pairs, dim)
        self.keys = (keys / keys.norm(dim=-1, keepdim=True)).to(device, dtype)
        # the state cue for each fact: +state_dir for plus, -state_dir for minus
        cue = torch.stack([state_dir, -state_dir], dim=1).reshape(2 * n_pairs, dim)
        self.cues = (cue / cue.norm(dim=-1, keepdim=True)).to(device, dtype)
        self.topic_of = torch.arange(n_pairs).repeat_interleave(2).to(device)
        self._g = g

    def query_batch(self, m=512, seed=7):
        """Return (q, s, target_idx). q is a noisy topic vector (ambiguous between
        the pair); s is the correct fact's state cue (the disambiguator)."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.randint(0, self.keys.shape[0], (m,), generator=g)
        q = self.keys[idx] + self.noise * torch.randn(m, self.dim, generator=g).to(
            self.device, self.dtype)
        q = q / q.norm(dim=-1, keepdim=True)
        s = self.cues[idx]
        return q.to(self.device, self.dtype), s.to(self.device, self.dtype), idx.to(self.device)
