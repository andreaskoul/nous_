"""
Statistics for the pre-registered gates.

A gate is a claim about a population, not about one run. LoCoMo has only 10
conversations and questions inside a conversation are correlated, so the honest
resampling unit is the CONVERSATION (cluster bootstrap); the question-level
bootstrap is reported alongside but is optimistic.

  cluster_ci      mean + 95% CI, resampling clusters
  paired_diff     paired contrast a-b with question- and cluster-level CIs
  gate_verdict    PASS only if the whole cluster CI of (a-b) clears delta
                  ("a borderline result is a fail", made precise)
  auroc / ece     is a decision score calibrated / discriminative?
                  (System-One view: a Locate density is a typed decision)
"""
from __future__ import annotations
import numpy as np


def cluster_ci(values, clusters, n=2000, seed=0, alpha=0.05):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    rng = np.random.default_rng(seed)
    uc = np.unique(clusters)
    groups = [values[clusters == c] for c in uc]
    stats = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uc), len(uc))
        stats[b] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_diff(a, b, clusters, n=2000, seed=1):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    q = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    mean, clo, chi = cluster_ci(d, clusters, n=n, seed=seed)
    qlo, qhi = np.percentile(q, [2.5, 97.5])
    return dict(mean=mean, q_ci=[float(qlo), float(qhi)], cluster_ci=[clo, chi])


def gate_verdict(a, b, clusters, delta, n=2000, seed=1):
    """Pre-registered superiority test: PASS iff the lower end of the cluster CI
    of mean(a-b) is >= delta. Point estimates above delta with a CI crossing it
    are reported as INCONCLUSIVE, not PASS."""
    r = paired_diff(a, b, clusters, n=n, seed=seed)
    lo = r["cluster_ci"][0]
    r["verdict"] = "PASS" if lo >= delta else ("INCONCLUSIVE" if r["mean"] >= delta else "FAIL")
    return r


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties averaged (Mann-Whitney)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, correct, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)
