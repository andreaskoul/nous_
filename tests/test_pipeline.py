"""
Integration checks for nous_ v2 plus the surviving exploratory pieces.
Offline and deterministic: fixtures + fake encoders, no network, no dataset.
Run with:  python tests/test_pipeline.py   (or pytest). Module-level unit tests live
in tests/test_<module>.py.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from manifold_mvp.synthetic import CurvedManifold, ContextCorpus  # noqa: E402
from manifold_mvp.metric import PullbackMetric  # noqa: E402
from manifold_mvp.curvature import curvature_probe  # noqa: E402
from manifold_mvp.conformal import ConformalHead  # noqa: E402
from manifold_mvp.locate import ContextHead  # noqa: E402
from manifold_mvp import losses, invert  # noqa: E402
from manifold_mvp.signature import signature  # noqa: E402

torch.manual_seed(0)
DT = torch.float32


# ---------------------------------------------------------------- exploratory / legacy

def test_exploratory_metric_spd():
    """The (exploratory) conformal pullback metric stays SPD; the probe runs."""
    man = CurvedManifold(d_latent=2, D=32, warp=2.0, freq=2.5, dtype=DT)
    lam = ConformalHead(2, hidden=32)
    g = PullbackMetric(man.decode, conformal=lam)
    z = man.sample_latents(20, seed=1)
    G = g.metric(z)
    assert (torch.linalg.eigvalsh(G) > 0).all() and torch.isfinite(G).all()
    probe = curvature_probe(g, z, n_pairs=12, steps=40)
    assert np.isfinite(probe["shortcut (1-ratio)"])
    print(f"  exploratory metric SPD + curvature probe OK (shortcut={probe['shortcut (1-ratio)']:.3f})")


def test_linear_chart_is_flat():
    """The reality-check finding, as a test: a linear PCA chart has G0 = I."""
    from manifold_mvp.metric import TorchPCA
    X = torch.randn(200, 24)
    pca = TorchPCA(6).fit(X)
    G = PullbackMetric(pca.decode).metric(torch.randn(5, 6))
    assert torch.allclose(G, torch.eye(6).expand(5, 6, 6), atol=1e-4)
    print("  linear PCA chart pullback metric == identity (geometry cannot help there)")


def test_legacy_losses():
    """Legacy/exploratory loss terms stay finite and differentiable; grounding is gone."""
    from config import LossWeights
    corpus = ContextCorpus(n_facts=128, dim=64, dtype=DT)
    M = corpus.keys
    head = ContextHead(64)
    q, s, tgt = corpus.query_batch(m=64, seed=1)
    phi = head(q, s)
    l_full = losses.retrieval_infonce(phi, M, tgt, beta=8.0)
    negs = losses.mine_hard_negatives(phi.detach(), M, tgt, k=8)
    assert torch.isfinite(losses.retrieval_infonce(phi, M, tgt, beta=8.0, neg_idx=negs))
    l_full.backward()
    assert any(p.grad is not None for p in head.parameters())
    man = CurvedManifold(d_latent=2, D=32, dtype=DT)
    g = PullbackMetric(man.decode, conformal=ConformalHead(2, hidden=32))
    a, b = man.sample_latents(16, seed=4), man.sample_latents(16, seed=5)
    l_geo, _ = losses.geometry_loss(g, man.sample_latents(16, seed=2), man.sample_latents(16, spread=6.0, seed=3),
                                    a, b, man.ambient_dist(a, b))
    assert torch.isfinite(l_geo)
    assert not hasattr(losses, "grounding_loss"), "grounding_loss had zero gradient and must stay removed"
    total = losses.composite_loss({"ret": l_full.detach(), "geo": l_geo.detach()}, LossWeights())
    assert torch.isfinite(total) and abs(float(total) - float(l_full)) < 1e-5   # geo weight is 0 in v2
    print("  legacy losses finite; grounding removed; v2 weights keep geo out of the objective")


def test_signature_inverse_is_regression():
    """C2 honesty: the global linear inverse is a regression; training the neural
    conditional-mean decoder reduces error; distinct skills stay separable."""
    C, T, depth = 4, 24, 3
    sig_dim = signature(torch.zeros(1, T, C), depth).shape[-1]
    sample = lambda n, seed: torch.cumsum(0.2 * torch.randn(n, T, C, generator=torch.Generator().manual_seed(seed)), 1)
    assert hasattr(invert, "GlobalLinearRegressionInverse") and not hasattr(invert, "LinearSignatureInverse")
    lin = invert.GlobalLinearRegressionInverse(depth).fit(sample(512, 0))
    assert np.isfinite(invert.reconstruction_relerr(lin, sample(128, 99), depth))
    inv = invert.NeuralInverse(sig_dim, T, C, hidden=128)
    rel0 = invert.reconstruction_relerr(inv, sample(128, 99), depth)
    inv = invert.train_inverse(inv, sample, steps=200, depth=depth, batch=128)
    assert invert.reconstruction_relerr(inv, sample(128, 99), depth) < rel0
    base = torch.cumsum(0.3 * torch.randn(6, T, C), dim=1)
    execs = torch.stack([base[k][None] + 0.04 * torch.randn(10, T, C) for k in range(6)])
    assert invert.skill_separability(lambda p: signature(p, depth), execs) > 1.0
    print("  C2 regression inverse + conditional-mean decoder + separability OK")


def test_stats():
    """Gate statistics behave: clear wins PASS, noise does not, calibration math."""
    from manifold_mvp import stats
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    base = rng.random(500) < 0.3
    better = base | (rng.random(500) < 0.4)
    noisy = rng.random(500) < 0.32
    assert stats.gate_verdict(better, base, clusters, 0.15)["verdict"] == "PASS"
    assert stats.gate_verdict(noisy, base, clusters, 0.15)["verdict"] == "FAIL"
    assert abs(stats.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert stats.ece([1.0, 1.0], [1, 1]) < 1e-9 and abs(stats.ece([0.9] * 10, [0] * 10) - 0.9) < 1e-9
    print("  gate statistics OK")


# ---------------------------------------------------------------- v2 end-to-end (offline)

def _write_locomo_fixture(path, n_conv=3):
    convs = []
    for c in range(n_conv):
        convs.append({
            "conversation": {
                "speaker_a": "Ana", "speaker_b": "Ben",
                "session_1_date_time": "1 Jan 2023",
                "session_1": [
                    {"speaker": "Ana", "dia_id": "D1:1", "text": f"I adopted a dog named Rex{c} last week."},
                    {"speaker": "Ben", "dia_id": "D1:2", "text": "Nice, what breed is it?"},
                    {"speaker": "Ana", "dia_id": "D1:3", "text": f"Rex{c} is a golden retriever puppy."},
                    {"speaker": "Ben", "dia_id": "D1:4", "text": "I started a pottery class on Mondays."},
                ],
                "session_2_date_time": "5 Feb 2023",
                "session_2": [
                    {"speaker": "Ana", "dia_id": "D2:1", "text": "We hiked the coastal trail with Rex{c}.".replace("{c}", str(c))},
                    {"speaker": "Ben", "dia_id": "D2:2", "text": "My pottery bowl finally came out of the kiln."},
                    {"speaker": "Ana", "dia_id": "D2:3", "text": "I am thinking about learning the violin."},
                ],
            },
            "qa": [
                {"question": "What breed is Ana's dog?", "answer": "golden retriever", "evidence": ["D1:3"], "category": 4},
                {"question": "What did Ben make in pottery class?", "answer": "a bowl", "evidence": ["D1:4; D2:2"], "category": 1},
                {"question": "Where did Ana hike with her dog?", "answer": "coastal trail", "evidence": ["D2:1"], "category": 2},
                {"question": "What instrument is Ben learning?", "evidence": ["D2:3"], "category": 5,
                 "adversarial_answer": "violin"},
            ],
        })
    with open(path, "w") as fh:
        json.dump(convs, fh)


def _bow_embedder(dim=64):
    """Deterministic bag-of-words hashing embedder: lexical overlap -> cosine."""
    def vec(tok):
        g = torch.Generator().manual_seed(int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little"))
        return torch.randn(dim, generator=g)

    def f(texts):
        out = []
        for t in texts:
            toks = re.findall(r"\w+", t.lower())
            v = torch.stack([vec(w) for w in toks]).sum(0) if toks else torch.zeros(dim)
            out.append(v / v.norm().clamp_min(1e-9))
        return torch.stack(out)
    return f


class _OverlapDecider:
    name = "fake-overlap"

    def relevance(self, query, memories):
        qt = set(re.findall(r"\w+", query.lower()))
        return np.array([len(qt & set(re.findall(r"\w+", m.lower()))) for m in memories], float)


def test_v2_pipeline_smoke():
    """STORE -> CANDIDATES -> LOCATE (+ teacher, state, null) end to end on a fixture."""
    from config import LossWeights
    from manifold_mvp import store
    from manifold_mvp.candidates import Encoder, CandidateGenerator
    from manifold_mvp.pipeline import (FEATURES, PairCache, build_batch, state_vectors, teacher_scores,
                                       train_locate, locate_logits, full_ranking, rank_metrics)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "locomo.json")
        _write_locomo_fixture(p)
        bank, qs = store.load_locomo(p)
    assert len(bank) == 21 and len(qs) == 12
    enc = Encoder("fake-bow", fn=_bow_embedder(), cache_dir=None)
    gen = CandidateGenerator(bank, enc, alpha=0.5, K=4, contiguity=1)
    gen.index()
    doc = enc.encode_docs(bank.texts)
    q_emb = enc.encode_queries([q.text for q in qs])
    K = 5
    b = build_batch(gen, bank, qs, q_emb, K)
    assert b.rows.shape == (12, K) and b.feats.shape == (12, K, len(FEATURES)) and b.mask.any(1).all()
    assert np.isfinite(b.feats).all() and b.unans.sum() == 3
    for i, q in enumerate(qs):                   # per-conversation scope
        assert all(bank.memories[r].conv == q.conv for r in b.rows[i, b.mask[i]])
    # premise feature: the adversarial "What instrument is Ben learning?" marks exactly Ben's turns
    fi = FEATURES.index("name_match")
    adv = [i for i, q in enumerate(qs) if not q.answerable]
    for i in adv:
        spk = [bank.memories[r].speaker for r in b.rows[i, b.mask[i]]]
        assert list(b.feats[i, b.mask[i], fi]) == [float(s == "Ben") for s in spk]
    S, H = state_vectors(bank, doc, qs, mode="last_session")
    assert S.shape == H.shape == (12, doc.shape[1]) and torch.isfinite(H).all()
    cache = PairCache()
    t1 = teacher_scores(_OverlapDecider(), bank, qs, b, cache=cache)
    n_pairs = len(cache.d)
    t2 = teacher_scores(_OverlapDecider(), bank, qs, b, cache=cache)
    assert np.array_equal(t1, t2) and len(cache.d) == n_pairs      # second call served from cache
    w = LossWeights()
    scorer = train_locate(b, doc, q_emb, S, H, w, teacher=t1, steps=60, bsz=8, seed=0, rank=4)
    lg = locate_logits(scorer, b, doc, q_emb, S, H)
    assert lg.shape == (12, K + 1) and np.isfinite(lg).all()
    for i, q in enumerate(qs):
        m = rank_metrics(*full_ranking(b, i, lg[i, :-1]), q.evidence)
        assert 0.0 <= m["mrr"] <= 1.0 and m["hit@1"] <= m["hit@3"] <= m["hit@10"]
        assert m["cover@10"] <= m["recall@10"] + 1e-9
    print("  v2 pipeline end-to-end on a fixture OK (shortlist, features, state, cached teacher, LOCATE)")


def test_config_matches_plan():
    """config.PreReg is the typed view of prereg/plan_v2.json: the numbers must agree."""
    from config import PreReg
    plan = json.load(open(os.path.join(ROOT, "prereg", "plan_v2.json")))
    text = {g["id"]: json.dumps(g) for g in plan["gates"]}
    P = PreReg()
    checks = {"X0": [P.x0_rerank_hit1, P.x0_minilm_hit1, P.x0_tol], "G1a": [P.delta_state],
              "G1b": [P.delta_recall_mh, P.ni_margin], "A1": [P.null_auroc, P.null_auroc_lo, P.max_false_abstain],
              "C1": [P.max_ece, *P.calib_slope]}
    for gid, vals in checks.items():
        for v in vals:
            s = f"{v:g}"
            assert s in text[gid] or s.lstrip("0") in text[gid], f"{gid}: {s} missing from the plan"
    print("  config.PreReg agrees with prereg/plan_v2.json")


if __name__ == "__main__":
    for fn in [test_exploratory_metric_spd, test_linear_chart_is_flat, test_legacy_losses,
               test_signature_inverse_is_regression, test_stats, test_v2_pipeline_smoke, test_config_matches_plan]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nALL PIPELINE CHECKS PASSED")
