"""
E1 (geodesic recall vs cosine) and E3 (density-as-parallel-sampler).
E0 lives in curvature.py; E2 in locate.py. run_mvp.py wires all four together.
"""
from __future__ import annotations
import torch
from .geodesic import geodesic, straight_length


def e1_geodesic_recall(manifold, metric, n_queries=30, n_cand=12,
                       n_segments=14, steps=100, seed=5):
    """For each query latent, rank candidates by (a) straight latent distance and
    (b) geodesic distance under G. Ground truth = AMBIENT distance ||g(.)-g(.)||.
    Score = how often each method's nearest matches the true nearest (top-1),
    plus mean Spearman-ish rank agreement."""
    g = torch.Generator().manual_seed(seed)
    lat = manifold.sample_latents(n_queries + n_cand * n_queries, seed=seed)
    queries = lat[:n_queries]
    pool = lat[n_queries:].reshape(n_queries, n_cand, -1)

    cos_hit = geo_hit = 0
    for i in range(n_queries):
        q = queries[i]
        cand = pool[i]
        true_d = manifold.ambient_dist(q[None].expand_as(cand), cand)  # (n_cand,)
        straight = torch.stack([straight_length(metric, q, c, n_segments) for c in cand])
        geo = torch.stack([geodesic(metric, q, c, n_segments, steps)[1] for c in cand])
        cos_hit += int(straight.argmin() == true_d.argmin())
        geo_hit += int(geo.argmin() == true_d.argmin())
    return {
        "straight_top1": cos_hit / n_queries,
        "geodesic_top1": geo_hit / n_queries,
        "n_queries": n_queries,
    }


def e3_density_sampler(corpus, head, M, beta=8.0, Ks=(1, 2, 4, 8, 16),
                       m=1500, seed=21):
    """Density as a parallel-exploration mechanism. Draw K samples from
    rho_theta as K frontiers; a query is 'solved' if the true target is in the
    top-K of the density. Rising accuracy(K) = the density buys parallel search
    (the contribution the latent-CoT field flags as missing)."""
    from .locate import locate_density
    q, s, tgt = corpus.query_batch(m=m, seed=seed)
    with torch.no_grad():
        phi = head(q, s)
        rho = locate_density(phi, M, beta=beta)        # (m, N)
    out = {}
    for K in Ks:
        topk = rho.topk(K, dim=-1).indices             # (m, K)
        hit = (topk == tgt[:, None]).any(dim=-1).float().mean().item()
        out[K] = hit
    return out
