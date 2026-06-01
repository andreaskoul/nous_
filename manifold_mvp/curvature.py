"""
E0 — THE CURVATURE PROBE.  Runs first; can flip the whole framing in minutes.

If the semantic manifold is near-flat (Shao et al. found learned IMAGE manifolds
were near-zero curvature), then geodesic == straight line and the geometry earns
nothing — the honest move is to reframe as a learned-recall paper. So we measure,
before building anything geometric, two cheap signals:

  (1) shortcut ratio  rho = mean_pairs L_geodesic / L_straight.
      The geodesic is the SHORTEST curve, so rho <= 1 always; the gap is the
      signal:  shortcut = 1 - rho.
      shortcut ~ 0 => flat (straight latent line already IS the geodesic)
                      => geometry decorative.
      shortcut > 0 => curved => the geodesic finds a shorter route the straight
                      line misses.

  (2) metric variation  = how much G(z) changes across the space (coefficient of
      variation of log det G). 0 => constant metric => flat.

Pre-registered gate tau_kappa: if shortcut < tau_kappa => NO-GO on geometry.
"""
from __future__ import annotations
import torch
from .geodesic import geodesic, straight_length


def curvature_probe(metric, latents, n_pairs=40, tau_kappa=0.02,
                    n_segments=16, steps=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = latents.shape[0]
    ii = torch.randint(0, N, (n_pairs,), generator=g)
    jj = torch.randint(0, N, (n_pairs,), generator=g)

    ratios = []
    for a, b in zip(ii.tolist(), jj.tolist()):
        if a == b:
            continue
        z0, z1 = latents[a], latents[b]
        _, Lg = geodesic(metric, z0, z1, n_segments=n_segments, steps=steps)
        Ls = straight_length(metric, z0, z1, n_segments=n_segments)
        ratios.append((Lg / Ls.clamp_min(1e-9)).item())
    ratios = torch.tensor(ratios)
    shortcut_ratio = ratios.mean().item()        # = L_geo / L_straight, <= 1
    shortcut = 1.0 - shortcut_ratio              # the curvature signal

    # metric variation across space
    G = metric.metric(latents)                       # (N, d, d)
    logdet = torch.logdet(G).clamp(-30, 30)
    metric_cv = (logdet.std() / (logdet.abs().mean() + 1e-9)).item()

    curved = shortcut >= tau_kappa
    return {
        "shortcut_ratio": shortcut_ratio,
        "shortcut (1-ratio)": shortcut,
        "shortcut_std": ratios.std().item(),
        "metric_logdet_cv": metric_cv,
        "tau_kappa": tau_kappa,
        "verdict": "CURVED -> geometry is load-bearing; proceed to E1/E2"
        if curved else
        "FLAT -> reframe as learned-recall; geodesics buy little",
        "go": bool(curved),
    }
