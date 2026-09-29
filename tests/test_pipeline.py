"""
Unit checks for the end-to-end pipeline pieces (plan stage: Verification).
Runs entirely on the synthetic path / closed-form fixtures -> no network, no
dataset, fast. Run with:  python tests/test_pipeline.py   (or pytest).
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_mvp.synthetic import CurvedManifold, ContextCorpus
from manifold_mvp.metric import PullbackMetric
from manifold_mvp.curvature import curvature_probe
from manifold_mvp.conformal import ConformalHead
from manifold_mvp.locate import ContextHead
from manifold_mvp import losses
from manifold_mvp import invert
from manifold_mvp.signature import signature

torch.manual_seed(0)
DT = torch.float32


def test_conformal_metric_spd():
    """G(z)=lambda_theta(z) G0(z) stays SPD and the learned metric is curved."""
    man = CurvedManifold(d_latent=2, D=32, warp=2.0, freq=2.5, dtype=DT)
    lam = ConformalHead(2, hidden=32)
    g = PullbackMetric(man.decode, conformal=lam)
    z = man.sample_latents(20, seed=1)
    G = g.metric(z)
    eig = torch.linalg.eigvalsh(G)
    assert (eig > 0).all(), "metric not SPD"
    assert torch.isfinite(G).all()
    # lambda is strictly positive, ~1 at init
    l = lam(z)
    assert (l > 0).all() and l.shape == (20,)
    probe = curvature_probe(g, z, n_pairs=12, steps=40)
    assert torch.isfinite(torch.tensor(probe["shortcut (1-ratio)"]))
    print("  conformal/metric SPD + curvature probe OK "
          f"(shortcut={probe['shortcut (1-ratio)']:.3f}, lambda~{l.mean():.2f})")


def test_losses_differentiable():
    """Every loss term is finite and back-propagates to its parameters."""
    corpus = ContextCorpus(n_facts=128, dim=64, dtype=DT)
    M = corpus.keys
    head = ContextHead(64)
    q, s, tgt = corpus.query_batch(m=64, seed=1)
    phi = head(q, s)

    # L_ret full-bank and mined
    l_full = losses.retrieval_infonce(phi, M, tgt, beta=8.0)
    negs = losses.mine_hard_negatives(phi.detach(), M, tgt, k=8)
    l_mined = losses.retrieval_infonce(phi, M, tgt, beta=8.0, neg_idx=negs)
    assert torch.isfinite(l_full) and torch.isfinite(l_mined)
    l_full.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters()), "no grad to recall head"

    # L_geo wrt conformal params
    man = CurvedManifold(d_latent=2, D=32, dtype=DT)
    lam = ConformalHead(2, hidden=32)
    g = PullbackMetric(man.decode, conformal=lam)
    z_on = man.sample_latents(16, seed=2)
    z_off = man.sample_latents(16, spread=6.0, seed=3)   # off-manifold (far out)
    a = man.sample_latents(16, seed=4); b = man.sample_latents(16, seed=5)
    d_tgt = man.ambient_dist(a, b)
    l_geo, info = losses.geometry_loss(g, z_on, z_off, a, b, d_tgt)
    assert torch.isfinite(l_geo)
    l_geo.backward()
    assert any(p.grad is not None for p in lam.parameters()), "no grad to lambda_theta"

    # L_ent and L_gnd
    rho = torch.softmax(8.0 * (phi.detach() @ M.T), dim=-1)
    h_floor = 0.25 * torch.log(torch.tensor(float(M.shape[0])))
    assert torch.isfinite(losses.entropy_floor(rho, h_floor))
    assert torch.isfinite(losses.grounding_loss(corpus.keys[:32], corpus.keys[:16]))

    # composite sum
    from config import LossWeights
    total = losses.composite_loss(
        {"ret": l_full.detach(), "geo": l_geo.detach()}, LossWeights())
    assert torch.isfinite(total)
    print(f"  losses finite + differentiable OK (warp={info['warp']:.3f}, "
          f"dist_match={info['dist_match']:.3f})")


def test_signature_inverse():
    """NeuralInverse beats the naive baseline and the linear insertion inverse
    recovers a path; both are sane; separability > 1 for distinct skills."""
    C, T, depth = 4, 24, 3
    sig_dim = signature(torch.zeros(1, T, C), depth).shape[-1]

    def sample(n, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.cumsum(0.2 * torch.randn(n, T, C, generator=g), dim=1)

    # linear insertion-style inverse (training-free)
    lin = invert.LinearSignatureInverse(depth).fit(sample(512, 0))
    rel_lin = invert.reconstruction_relerr(lin, sample(128, 99), depth)

    # learned inverse
    inv = invert.NeuralInverse(sig_dim, T, C, hidden=128)
    rel0 = invert.reconstruction_relerr(inv, sample(128, 99), depth)
    inv = invert.train_inverse(inv, sample, steps=250, depth=depth, batch=128)
    rel1 = invert.reconstruction_relerr(inv, sample(128, 99), depth)
    assert rel1 < rel0, "training did not improve reconstruction"

    # separability of distinct skills via the raw signature embedding
    n_skill, n_exec = 6, 10
    base = torch.cumsum(0.3 * torch.randn(n_skill, T, C), dim=1)
    execs = torch.stack([base[k][None] + 0.04 * torch.randn(n_exec, T, C)
                         for k in range(n_skill)])
    sep = invert.skill_separability(lambda p: signature(p, depth), execs)
    assert sep > 1.0
    print(f"  sigma^-1 OK (linear rel-err {rel_lin:.3f}; learned {rel0:.3f}->{rel1:.3f}; "
          f"separability {sep:.1f}x)")


def _fake_embedder(dim=48):
    """Deterministic per-text embedding (stable hash) — no network/model."""
    import hashlib

    def f(texts):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.md5(t.encode()).digest()[:4], "little")
            g = torch.Generator().manual_seed(seed)
            out.append(torch.randn(dim, generator=g))
        return torch.stack(out)
    return f


def _write_fixture(path):
    import json
    convs = []
    for c in range(3):
        convs.append({
            "conversation": {
                "session_1_date_time": "1 Jan 2023",
                "session_1": [
                    {"speaker": "A", "dia_id": f"D{c}1:1", "text": f"conv{c} pet is a dog named Rex{c}."},
                    {"speaker": "B", "dia_id": f"D{c}1:2", "text": f"conv{c} nice, what breed is it?"},
                ],
                "session_2_date_time": "5 Feb 2023",
                "session_2": [
                    {"speaker": "A", "dia_id": f"D{c}2:1", "text": f"conv{c} Rex{c} is a golden retriever, very fluffy."},
                    {"speaker": "B", "dia_id": f"D{c}2:2", "text": f"conv{c} we went to the park last weekend."},
                ],
            },
            "qa": [
                {"question": f"What is the name of the pet in conversation {c}?",
                 "answer": f"Rex{c}", "evidence": [f"D{c}1:1"], "category": 1},
                {"question": f"What breed is the pet in conversation {c}?",
                 "answer": "golden retriever", "evidence": [f"D{c}2:1"], "category": 1},
            ],
        })
    with open(path, "w") as fh:
        json.dump(convs, fh)


def test_locomo_loader():
    """Fixture LoCoMo: parsing, tuples, PCA chart, and leakage-free split."""
    import tempfile
    from manifold_mvp.real import LoCoMoData
    emb = _fake_embedder(48)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fix.json")
        _write_fixture(p)
        train = LoCoMoData(p, split="train", conv_ids=(0, 1), embed_fn=emb, pca_dim=8, n_anchors=4)
        test = LoCoMoData(p, split="test", conv_ids=(2,), embed_fn=emb, pca_dim=8, n_anchors=4)

    # bank + tuples well-formed
    assert train.keys.shape[0] == 8 and train.keys.shape[1] == 48   # 2 convs * 4 turns
    q, s, tgt = train.query_batch(m=6, seed=0)
    assert q.shape == (6, 48) and s.shape == (6, 48) and tgt.shape == (6,)
    assert int(tgt.max()) < train.keys.shape[0] and int(tgt.min()) >= 0
    qa_q, qa_s, qa_t, ev = train.all_queries()
    assert qa_q.shape[0] == 4 and all(len(e) >= 1 for e in ev)        # 2 convs * 2 QA

    # manifold role: PCA chart shapes + ambient distance
    z = train.sample_latents(5, seed=1)
    assert z.shape == (5, 8)
    assert train.decode(z).shape == (5, 48)
    assert train.ambient_dist(z, train.sample_latents(5, seed=2)).shape == (5,)

    # leakage-free: train and test banks share no memory text vectors
    inter = (torch.cdist(train.keys, test.keys) < 1e-6).any().item()
    assert not inter, "train/test memory overlap -> leakage"

    # h is query-time state: two questions in the same conversation with
    # DIFFERENT evidence must get the SAME history (no label leakage)
    same = (train.qa_conv == train.qa_conv[0]).nonzero().flatten()
    assert len(same) >= 2 and len(set(train.target[same].tolist())) >= 2
    assert torch.allclose(train.h[same[0]], train.h[same[1]]), "h depends on the evidence -> leakage"
    # memories carry their speaker
    assert all(t.split(":", 1)[0] in ("A", "B") for t in train.texts)

    # the metric must build on the PCA chart (jacrev needs decode((d,))->(D,))
    g = PullbackMetric(train.decode, conformal=ConformalHead(z.shape[1], hidden=16))
    G = g.metric(train.sample_latents(8, seed=3))
    assert G.shape == (8, z.shape[1], z.shape[1])
    assert (torch.linalg.eigvalsh(G) > 0).all() and torch.isfinite(G).all()
    print(f"  LoCoMo loader OK (bank N={train.keys.shape[0]}, QA={qa_q.shape[0]}, "
          f"latent d={z.shape[1]}, split leakage-free, metric SPD on chart)")


def test_stats():
    """Gate statistics behave: clear wins PASS, noise does not, calibration math."""
    import numpy as np
    from manifold_mvp import stats
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(10), 50)
    base = rng.random(500) < 0.3
    better = base | (rng.random(500) < 0.4)                  # large real gain
    noisy = rng.random(500) < 0.32                           # ~no gain
    assert stats.gate_verdict(better, base, clusters, 0.15)["verdict"] == "PASS"
    assert stats.gate_verdict(noisy, base, clusters, 0.15)["verdict"] == "FAIL"
    assert abs(stats.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) - 1.0) < 1e-9
    assert abs(stats.auroc([0.5, 0.5], [1, 0]) - 0.5) < 1e-9
    assert stats.ece([1.0, 1.0], [1, 1]) < 1e-9 and abs(stats.ece([0.9] * 10, [0] * 10) - 0.9) < 1e-9
    print("  gate statistics OK (cluster-bootstrap PASS/FAIL, AUROC, ECE)")


def test_locomo_training_step():
    """The real-data object drives the actual training loop: every loss term is
    finite, the metric builds on the PCA chart, and both optimizers step."""
    import tempfile, math
    from manifold_mvp.real import LoCoMoData
    from manifold_mvp.locate import locate_density
    from config import LossWeights
    from train import geo_minibatch
    emb = _fake_embedder(48)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "fix.json"); _write_fixture(p)
        data = LoCoMoData(p, conv_ids=(0, 1, 2), embed_fn=emb, pca_dim=6, n_anchors=4)
    M = data.keys; N, dim = M.shape; dlat = data.pca.d
    head = ContextHead(dim, hidden=16)
    lam = ConformalHead(dlat, hidden=16)
    g = PullbackMetric(data.decode, conformal=lam)
    oh = torch.optim.Adam(head.parameters(), lr=1e-2)
    om = torch.optim.Adam(lam.parameters(), lr=1e-2)
    h_floor = 0.25 * math.log(N)
    for it in range(4):
        q, s, tgt = data.query_batch(m=8, seed=it)
        phi = head(q, s)
        parts = {"ret": losses.retrieval_infonce(phi, M, tgt, beta=8.0)}
        parts["ent"] = losses.entropy_floor(locate_density(phi, M, 8.0), h_floor)
        parts["gnd"] = losses.grounding_loss(M[torch.randint(0, N, (8,))], M[:4])
        z_on, z_off, a, b, dt = geo_minibatch(data, 6, 100 + it, "cpu", DT)
        lg, _ = losses.geometry_loss(g, z_on, z_off, a, b, dt)
        parts["geo"] = lg
        loss = losses.composite_loss(parts, LossWeights())
        assert torch.isfinite(loss), "non-finite composite loss on LoCoMo object"
        oh.zero_grad(); om.zero_grad(); loss.backward(); oh.step(); om.step()
    print("  LoCoMo training-loop step OK (all loss terms finite, optimizers step)")


if __name__ == "__main__":
    for fn in [test_conformal_metric_spd, test_losses_differentiable,
               test_signature_inverse, test_locomo_loader, test_stats,
               test_locomo_training_step]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nALL PIPELINE UNIT CHECKS PASSED")
