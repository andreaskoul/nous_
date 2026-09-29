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

v2 additions (frozen pre-registration, G = 10 conversations):
  cluster_t_ci    ratio-estimator mean, CR1 cluster-robust SE, Student-t with G-1
                  df. With few clusters the normal / bootstrap-percentile interval
                  undercovers; t(G-1) + CR1 is the standard few-cluster fix
                  (Bell & McCaffrey 2002; Cameron & Miller 2015, J. Hum. Resour.).
                  At G=10 the percentile cluster bootstrap is ~18% too narrow:
                  1.96*sqrt(9/10) / t_{.975,9} = 0.82 (checked in the tests).
  wild_cluster_bootstrap_ci  Rademacher signs on centred cluster totals
                  (Cameron, Gelbach & Miller 2008, REStat); when 2^G <= n all sign
                  patterns are enumerated, so the G=10 case is exact (Webb 2014).
                  Not studentized, so at G=10 it covers ~0.90 for nominal 0.95
                  (like the pairs bootstrap; cluster-t ~0.94, 2000 simulated
                  draws, unequal sizes): a companion diagnostic, not a gate bound.
  mcnemar_exact   exact paired test on discordant pairs (hit@1 is binary)
  holm            step-down FWER control over the pre-registered gate family
                  (Holm 1979) -- several gates share one alpha
  power_mde       what effect CAN 10 conversations detect? (report before running)
  noninferiority_verdict / superiority_verdict
                  one-sided gates that must clear BOTH the cluster-t bound and the
                  percentile cluster-bootstrap bound: the verdict never rests on
                  the more optimistic of two small-sample approximations.
  t_cdf / t_ppf   Student-t via the regularized incomplete beta (no scipy).
"""
from __future__ import annotations
import math
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


# ---- Student-t without scipy -------------------------------------------------

def _betacf(a, b, x, itmax=10000, eps=1e-15):
    """Continued fraction of the incomplete beta (modified Lentz, NR 6.4)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, itmax + 1):
        for num in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                    -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + num * d; d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + num / c; c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lf = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lf) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lf) * _betacf(b, a, 1.0 - x) / b


def _t_tail(t, df):
    """P(T > t) for t >= 0, computed directly (no 1-p cancellation)."""
    return 0.5 * _betainc(df / 2, 0.5, df / (df + t * t))


def t_cdf(t, df):
    tail = _t_tail(abs(t), df)
    return 1.0 - tail if t > 0 else tail


def t_ppf(p, df):
    """Student-t quantile by bisection on the exact tail (|err| ~ 1e-12 * t)."""
    if not 0.0 < p < 1.0:
        return math.copysign(math.inf, p - 0.5) if p in (0.0, 1.0) else math.nan
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    q = 1.0 - p
    lo, hi = 0.0, 1.0
    while _t_tail(hi, df) > q:
        lo, hi = hi, 2.0 * hi
    while hi - lo > 1e-12 * hi:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _t_tail(mid, df) > q else (lo, mid)
    return 0.5 * (lo + hi)


# ---- few-cluster inference -----------------------------------------------------

def _cluster_totals(values, clusters):
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    inv = np.unique(clusters, return_inverse=True)[1].ravel()
    return values.mean(), np.bincount(inv, weights=values), np.bincount(inv)


def cluster_t_ci(values, clusters, alpha=0.05):
    """mean = sum(y)/N (cluster means weighted by size: the ratio estimator),
    SE = CR1 sandwich, interval from t(G-1). Equal sizes -> exactly the one-sample
    t-test on the G cluster means. G < 2 -> (-inf, inf): nothing is identified."""
    mu, tot, cnt = _cluster_totals(values, clusters)
    G, N = len(cnt), cnt.sum()
    if G < 2:
        return float(mu), -math.inf, math.inf
    u = tot - cnt * mu                                   # centred cluster totals (scores)
    se = math.sqrt(G / (G - 1) * float((u ** 2).sum())) / N   # CR1; (N-1)/(N-K)=1 at K=1
    h = t_ppf(1 - alpha / 2, G - 1) * se
    return float(mu), float(mu - h), float(mu + h)


def wild_cluster_bootstrap_ci(values, clusters, n=2000, seed=0, alpha=0.05):
    """mean* = mean + sum_g w_g u_g / N, w_g = +-1 (Rademacher), u_g the centred
    cluster totals scaled by sqrt(G/(G-1)) so Var* matches CR1. Symmetric
    percentile interval mean +- q_{1-alpha}(|mean* - mean|). Exact enumeration of
    all 2^G sign patterns when that is no more than n draws."""
    mu, tot, cnt = _cluster_totals(values, clusters)
    G, N = len(cnt), cnt.sum()
    if G < 2:
        return float(mu), -math.inf, math.inf
    u = (tot - cnt * mu) * math.sqrt(G / (G - 1))
    if 2 ** G <= n:
        W = 1.0 - 2.0 * ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1)
        dev = np.abs(W @ u)
    else:
        rng, blk = np.random.default_rng(seed), max(1, (1 << 22) // G)   # bound memory
        dev = np.concatenate([np.abs(rng.choice([-1.0, 1.0], (min(blk, n - i), G)) @ u)
                              for i in range(0, n, blk)])
    h = float(np.quantile(dev, 1 - alpha)) / N
    return float(mu), float(mu - h), float(mu + h)


# ---- paired tests, multiplicity, power ---------------------------------------------

def _binom_half_sf(k, n):
    """P(X >= k), X ~ Bin(n, 1/2), summed in log space."""
    if k <= 0:
        return 1.0
    lc = math.lgamma(n + 1) - n * math.log(2)
    lp = np.array([lc - math.lgamma(j + 1) - math.lgamma(n - j + 1) for j in range(k, n + 1)])
    m = lp.max()
    return float(min(1.0, math.exp(m) * np.exp(lp - m).sum()))


def mcnemar_exact(a, b):
    """Exact McNemar on paired successes. n10 = #(a & ~b) (a right, b wrong),
    n01 = #(~a & b). Under H0 the discordant pairs split Bin(n10+n01, 1/2).
    p_greater: one-sided 'a better than b' = P(X >= n10); p_less the mirror;
    p_two_sided = min(1, 2 * smaller tail). No discordant pairs -> all p = 1."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p_ge = _binom_half_sf(n10, n) if n else 1.0
    p_le = _binom_half_sf(n01, n) if n else 1.0     # P(X <= n10) = P(n-X >= n01)
    return dict(n10=n10, n01=n01, p_greater=p_ge, p_less=p_le,
                p_two_sided=min(1.0, 2 * min(p_ge, p_le)))


def holm(pvals):
    """Holm step-down adjusted p-values: p_(i) -> max_{j<=i} min(1, (m-j+1) p_(j))."""
    keys = sorted(pvals, key=pvals.get)
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * float(pvals[k])))
        out[k] = run
    return {k: out[k] for k in pvals}


def power_mde(sd_cluster_diff, n_clusters, alpha=0.05, power=0.8, one_sided=True):
    """Minimum detectable mean paired difference of a cluster-level t-test:
    (t_{1-alpha,G-1} + t_{power,G-1}) * sd / sqrt(G), sd = SD of cluster-mean diffs."""
    G = int(n_clusters)
    if G < 2:
        return math.inf
    ta = t_ppf(1 - (alpha if one_sided else alpha / 2), G - 1)
    return float((ta + t_ppf(power, G - 1)) * sd_cluster_diff / math.sqrt(G))


# ---- one-sided gates (both bounds must clear) ----------------------------------------

def _lower_bounds(a, b, clusters, alpha, n, seed):
    """One-sided (1-alpha) lower bounds on mean(a-b): the ends of 1-2alpha intervals."""
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, t_lo, _ = cluster_t_ci(d, clusters, alpha=2 * alpha)
    _, b_lo, _ = cluster_ci(d, clusters, n=n, seed=seed, alpha=2 * alpha)
    return dict(mean=mean, t_lower=t_lo, boot_lower=b_lo, alpha=alpha,
                n=int(len(d)), n_clusters=int(len(np.unique(clusters))))


def noninferiority_verdict(a, b, clusters, margin, alpha=0.05, n=2000, seed=1):
    """H1: mean(a-b) > -margin. PASS iff the cluster-t AND the percentile
    cluster-bootstrap one-sided lower bounds both exceed -margin; INCONCLUSIVE if
    only the point estimate does; else FAIL."""
    r = _lower_bounds(a, b, clusters, alpha, n, seed)
    r["margin"] = margin
    ok = min(r["t_lower"], r["boot_lower"]) > -margin
    r["verdict"] = "PASS" if ok else ("INCONCLUSIVE" if r["mean"] > -margin else "FAIL")
    return r


def superiority_verdict(a, b, clusters, delta, alpha=0.05, n=2000, seed=1):
    """H1: mean(a-b) >= delta. PASS iff both one-sided lower bounds are >= delta;
    INCONCLUSIVE if the mean is >= delta but a bound is below; else FAIL."""
    r = _lower_bounds(a, b, clusters, alpha, n, seed)
    r["delta"] = delta
    ok = min(r["t_lower"], r["boot_lower"]) >= delta
    r["verdict"] = "PASS" if ok else ("INCONCLUSIVE" if r["mean"] >= delta else "FAIL")
    return r
