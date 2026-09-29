"""
Unit checks for the v2 gate statistics (manifold_mvp/stats.py): Student-t without
scipy, few-cluster CIs, exact McNemar, Holm, power/MDE, one-sided verdicts, and
back-compat of the v1 functions. Offline, deterministic, a few seconds.
Run with:  python tests/test_stats_v2.py   (or pytest).
"""
from __future__ import annotations
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_mvp import stats


def _sim(rng, G=10, effect=0.0, tau=0.5, sigma=1.0, sizes=(20, 80)):
    """Clustered data: y = effect + cluster effect + noise, unequal cluster sizes."""
    n_g = rng.integers(sizes[0], sizes[1] + 1, G)
    cl = np.repeat(np.arange(G), n_g)
    y = effect + rng.normal(0, tau, G)[cl] + rng.normal(0, sigma, len(cl))
    return y, cl


def test_t_quantiles():
    """t_ppf matches tables to 1e-3 (df>=1) and inverts t_cdf."""
    table = {(9, .975): 2.262, (4, .975): 2.776, (1, .975): 12.706, (2, .975): 4.303,
             (9, .95): 1.833, (9, .8): 0.883, (30, .975): 2.042, (3, .9995): 12.924}
    for (df, p), ref in table.items():
        assert abs(stats.t_ppf(p, df) - ref) < 1e-3, (df, p, stats.t_ppf(p, df))
        assert abs(stats.t_ppf(1 - p, df) + ref) < 1e-3                  # symmetry
    assert abs(stats.t_ppf(0.975, 1e5) - 1.95996) < 1e-4                 # -> normal
    for df in (1, 2.5, 7, 40):
        for p in (0.01, 0.3, 0.5, 0.9, 0.999):
            assert abs(stats.t_cdf(stats.t_ppf(p, df), df) - p) < 1e-9
    assert stats.t_ppf(0.5, 5) == 0.0 and stats.t_ppf(1.0, 5) == math.inf
    print("  t quantiles OK (t_.975: df9 2.262, df4 2.776, df1 12.706; cdf round-trip)")


def test_mcnemar_exact():
    """Hand example n10=9, n01=1: P(X>=9 | Bin(10,.5)) = 11/1024."""
    a = np.array([1] * 9 + [0] + [1] * 5 + [0] * 3, bool)
    b = np.array([0] * 9 + [1] + [1] * 5 + [0] * 3, bool)
    r = stats.mcnemar_exact(a, b)
    assert (r["n10"], r["n01"]) == (9, 1)
    assert abs(r["p_greater"] - 11 / 1024) < 1e-12 and abs(r["p_greater"] - 0.0107) < 1e-4
    assert abs(r["p_two_sided"] - 22 / 1024) < 1e-12
    s = stats.mcnemar_exact(b, a)                                          # mirror
    assert (s["n10"], s["n01"]) == (1, 9) and abs(s["p_less"] - r["p_greater"]) < 1e-12
    z = stats.mcnemar_exact([1, 0, 1], [1, 0, 1])                          # no discordance
    assert z["p_greater"] == z["p_two_sided"] == 1.0
    big = stats.mcnemar_exact([1] * 600 + [0] * 400, [0] * 600 + [1] * 400)  # log-space safe
    zc = (600 - 500) / math.sqrt(250)                                      # normal approx
    assert 0 < big["p_greater"] < 1e-8 and abs(math.log10(big["p_greater"])
                                                - math.log10(0.5 * math.erfc(zc / math.sqrt(2)))) < 0.3
    tie = stats.mcnemar_exact([1, 0], [0, 1])
    assert tie["p_two_sided"] == 1.0                                       # capped at 1
    print("  McNemar exact OK (n10=9,n01=1 -> p_greater=0.0107, mirror, n=0, n=1000)")


def test_holm():
    adj = stats.holm({"a": .01, "b": .04, "c": .03})
    for k, v in {"a": .03, "b": .06, "c": .06}.items():
        assert abs(adj[k] - v) < 1e-12, adj
    assert list(adj) == ["a", "b", "c"]                                    # key order kept
    adj = stats.holm({"x": .2, "y": .6, "z": .9})
    assert adj["z"] == 1.0 and adj["x"] <= adj["y"] <= adj["z"]           # capped, monotone
    assert stats.holm({}) == {} and stats.holm({"only": .04}) == {"only": .04}
    print("  Holm OK ({a:.01,b:.04,c:.03} -> {a:.03,b:.06,c:.06}, monotone, capped)")


def test_cluster_t_coverage():
    """Zero-mean clustered effect, G=10, unequal sizes: the cluster-t CI covers 0
    in >= 90% of 200 simulations; the percentile cluster bootstrap is narrower."""
    rng = np.random.default_rng(123)
    cover_t = cover_w = 0
    ratio = []
    for s in range(200):
        y, cl = _sim(rng)
        m, lo, hi = stats.cluster_t_ci(y, cl)
        cover_t += lo <= 0 <= hi
        _, wlo, whi = stats.wild_cluster_bootstrap_ci(y, cl, seed=s)
        cover_w += wlo <= 0 <= whi
        if s < 20:
            _, blo, bhi = stats.cluster_ci(y, cl, n=500, seed=s)
            ratio.append((bhi - blo) / (hi - lo))
    assert cover_t / 200 >= 0.90, cover_t
    assert cover_w / 200 >= 0.85, cover_w
    assert np.mean(ratio) < 0.95, np.mean(ratio)                          # ~0.82 expected
    # equal sizes: exactly the one-sample t-test on cluster means
    y, cl = _sim(rng, sizes=(30, 30))
    cm = np.array([y[cl == g].mean() for g in range(10)])
    h = stats.t_ppf(0.975, 9) * cm.std(ddof=1) / math.sqrt(10)
    m, lo, hi = stats.cluster_t_ci(y, cl)
    assert abs(m - cm.mean()) < 1e-12 and abs(lo - (cm.mean() - h)) < 1e-9 and abs(hi - (cm.mean() + h)) < 1e-9
    assert stats.cluster_t_ci([1.0, 2.0], [0, 0])[1] == -math.inf            # G=1: unidentified
    print(f"  cluster-t CI OK (coverage t={cover_t / 200:.3f}, wild={cover_w / 200:.3f}; "
          f"pairs-bootstrap/t width ratio {np.mean(ratio):.2f})")


def test_wild_cluster_bootstrap():
    rng = np.random.default_rng(7)
    y, cl = _sim(rng, G=10, effect=0.3)
    r1, r2 = stats.wild_cluster_bootstrap_ci(y, cl, seed=0), stats.wild_cluster_bootstrap_ci(y, cl, seed=5)
    assert r1 == r2                                                         # G=10: exact enumeration
    m, lo, hi = r1
    assert lo < m < hi and abs((m - lo) - (hi - m)) < 1e-12 and abs(m - y.mean()) < 1e-12
    y, cl = _sim(rng, G=40, effect=0.3)                                     # sampling path
    a, b = stats.wild_cluster_bootstrap_ci(y, cl, seed=0), stats.wild_cluster_bootstrap_ci(y, cl, seed=0)
    assert a == b and a != stats.wild_cluster_bootstrap_ci(y, cl, seed=1)
    _, tlo, thi = stats.cluster_t_ci(y, cl)
    assert abs((a[2] - a[1]) / (thi - tlo) - 1) < 0.1                       # agrees with t at G=40
    print("  wild cluster bootstrap OK (deterministic, symmetric, exact at G=10, ~t at G=40)")


def test_power_mde():
    mdes = [stats.power_mde(0.1, G) for G in (3, 5, 10, 20, 50, 200)]
    assert all(x > y for x, y in zip(mdes, mdes[1:])), mdes                 # monotone in G
    ref = (1.833113 + 0.883404) * 0.1 / math.sqrt(10)
    assert abs(stats.power_mde(0.1, 10) - ref) < 1e-5
    assert stats.power_mde(0.1, 10, one_sided=False) > stats.power_mde(0.1, 10)
    assert stats.power_mde(0.1, 10, power=0.9) > stats.power_mde(0.1, 10)
    assert stats.power_mde(0.1, 1) == math.inf
    print(f"  power/MDE OK (sd=0.1: G=10 -> {mdes[2]:.4f}, G=50 -> {mdes[4]:.4f}, decreasing)")


def test_verdicts():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    base = rng.random(500) < 0.3
    better = base | (rng.random(500) < 0.4)                                 # large real gain
    noisy = rng.random(500) < 0.32
    same = base.copy(); flip = rng.random(500) < 0.03; same[flip] = ~same[flip]
    worse = base & (rng.random(500) < 0.5)                                  # big loss
    r = stats.superiority_verdict(better, base, clusters, 0.15)
    assert r["verdict"] == "PASS" and min(r["t_lower"], r["boot_lower"]) >= 0.15
    assert stats.superiority_verdict(noisy, base, clusters, 0.15)["verdict"] == "FAIL"
    assert stats.noninferiority_verdict(same, base, clusters, 0.05)["verdict"] == "PASS"
    assert stats.noninferiority_verdict(better, base, clusters, 0.0)["verdict"] == "PASS"
    assert stats.noninferiority_verdict(worse, base, clusters, 0.05)["verdict"] == "FAIL"
    # mean above delta but heterogeneous across conversations -> not a PASS
    gains = np.array([0.4, -0.2, 0.3, -0.1, 0.5, -0.2, 0.2, 0.0, 0.1, -0.3])  # mean 0.07
    het = np.repeat(gains, 50) + rng.normal(0, 0.05, 500)
    r = stats.superiority_verdict(het, np.zeros(500), clusters, 0.05)
    assert r["mean"] >= 0.05 and r["verdict"] == "INCONCLUSIVE", r
    assert stats.noninferiority_verdict(het - 0.12, np.zeros(500), clusters, 0.1)["verdict"] == "INCONCLUSIVE"
    # the one-sided bound is the end of the (1-2alpha) two-sided interval
    d = better.astype(float) - base
    assert abs(r["alpha"] - 0.05) < 1e-12
    r = stats.noninferiority_verdict(better, base, clusters, 0.05)
    assert abs(r["t_lower"] - stats.cluster_t_ci(d, clusters, alpha=0.10)[1]) < 1e-12
    assert abs(r["boot_lower"] - stats.cluster_ci(d, clusters, seed=1, alpha=0.10)[1]) < 1e-12
    assert r["n"] == 500 and r["n_clusters"] == 10 and r["margin"] == 0.05
    print("  verdicts OK (superiority PASS/FAIL/INCONCLUSIVE, non-inferiority PASS/FAIL/INCONCLUSIVE)")


def test_back_compat():
    """v1 functions unchanged: same assertions as test_pipeline.test_stats plus
    golden outputs recorded before the v2 extension."""
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    base = rng.random(500) < 0.3
    better = base | (rng.random(500) < 0.4)
    noisy = rng.random(500) < 0.32
    assert stats.gate_verdict(better, base, clusters, 0.15)["verdict"] == "PASS"
    assert stats.gate_verdict(noisy, base, clusters, 0.15)["verdict"] == "FAIL"
    assert abs(stats.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert abs(stats.auroc([0.5, 0.5], [1, 0]) - 0.5) < 1e-9
    assert stats.ece([1.0, 1.0], [1, 1]) < 1e-9 and abs(stats.ece([0.9] * 10, [0] * 10) - 0.9) < 1e-9
    close = lambda x, y: np.allclose(np.asarray(x, float), np.asarray(y, float), atol=1e-12)
    assert close(stats.cluster_ci(better.astype(float), clusters), (0.542, 0.52, 0.566))
    r = stats.paired_diff(better, base, clusters)
    assert close([r["mean"], *r["q_ci"], *r["cluster_ci"]], [0.284, 0.244, 0.324, 0.252, 0.314])
    g = stats.gate_verdict(noisy, base, clusters, 0.15)
    assert close([g["mean"], *g["q_ci"], *g["cluster_ci"]], [0.07, 0.012, 0.126, 0.024, 0.114])
    assert abs(stats.auroc([0.3, 0.7, 0.7, 0.1, 0.9], [0, 1, 0, 0, 1]) - 0.9166666666666666) < 1e-12
    assert abs(stats.ece([0.2, 0.4, 0.6, 0.8, 0.95], [0, 1, 1, 1, 0], bins=5) - 0.47) < 1e-12
    print("  back-compat OK (cluster_ci, paired_diff, gate_verdict, auroc, ece unchanged)")


if __name__ == "__main__":
    for fn in [test_t_quantiles, test_mcnemar_exact, test_holm, test_cluster_t_coverage,
               test_wild_cluster_bootstrap, test_power_mde, test_verdicts, test_back_compat]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nALL STATS v2 CHECKS PASSED")
