"""
Unit checks for L1 CANDIDATES (manifold_mvp/candidates.py): tokenizer + BM25, z-score
and RRF fusion, leave-one-group-out alpha, contiguity expansion, the per-conversation
shortlist, and the Encoder's disk cache. Offline and deterministic: a hash-seeded fake
encoder replaces sentence-transformers, and a tiny duck-typed bank replaces store.py.
Run with:  python tests/test_candidates.py   (or pytest).
"""
from __future__ import annotations
import hashlib, os, sys, tempfile
from dataclasses import dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_mvp import candidates as C

D = 32
SHARED = np.random.default_rng(123).normal(size=D)


def fake_fn(texts):
    """Hash-seeded random vectors; texts mentioning the planted pair ("zebra" /
    "striped animal") also share one strong direction, so dense retrieval finds the
    pair even without lexical overlap."""
    out = []
    for t in texts:
        v = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=D)
        if "zebra" in t or "striped animal" in t:
            v = v + 6.0 * SHARED
        out.append(v)
    return np.stack(out)


@dataclass
class Mem:
    conv: int
    session: int
    speaker: str
    text: str


class FakeBank:
    """Duck-typed MemoryBank: rows are in conversation order."""

    def __init__(self, mems):
        self.memories, self.texts = mems, [m.text for m in mems]

    def __len__(self):
        return len(self.memories)

    def rows_of_conv(self, conv):
        return np.array([i for i, m in enumerate(self.memories) if m.conv == conv], np.int64)

    def neighbors(self, row, n=1, same_session=True):
        """Nearest first, earlier turn on ties (as store.MemoryBank.neighbors)."""
        m, M = self.memories[row], self.memories
        return sorted((j for j in range(row - n, row + n + 1)
                       if j != row and 0 <= j < len(M) and M[j].conv == m.conv
                       and (not same_session or M[j].session == m.session)), key=lambda j: (abs(j - row), j))


QUERY = "Which striped animal did Alice adopt last spring?"


def make_bank():
    t0 = ["Alice: I went hiking with my sister last spring.",
          "Bob: Which animal did Alice adopt? She said it was striped.",     # lexical decoy
          "Alice: The weather was lovely.",
          "Bob: Nice, I have been busy with work.",
          "Alice: We finally brought home our zebra from the sanctuary!",    # planted (dense)
          "Bob: Congratulations, that is amazing.",
          "Alice: Thanks, the kids love it.",
          "Bob: Let us catch up next week."]
    t1 = ["Carol: Which striped animal did Alice adopt last spring?",        # exact query, OTHER conv
          "Dan: No idea, ask her.",
          "Carol: I adopted a cat.",
          "Dan: Cats are great."]
    mems = [Mem(0, 1 + (i >= 4), "Alice" if t.startswith("Alice") else "Bob", t) for i, t in enumerate(t0)]
    mems += [Mem(1, 1, t.split(":")[0], t) for t in t1]
    return FakeBank(mems)


def order_of(x):
    return np.argsort(-np.asarray(x, float), kind="stable")


# ---------------------------------------------------------------------------

def test_tokenize_bm25():
    assert C.tokenize("Hello, World! It's 2 o'clock") == ["hello", "world", "it", "s", "2", "o", "clock"]
    assert C.TOKEN_RE.pattern == r"\w+"
    idx = C.BM25Index(["the cat sat on the mat", "dogs bark loudly", "a cat and a dog"])
    s = idx.scores("CAT sat")
    assert s.shape == (3,) and s.argmax() == 0 and s[1] == 0.0 and s[2] > 0
    assert C.BM25Index([]).scores("x").shape == (0,)
    print("  tokenize + BM25 OK (lowercase \\w+, best lexical match on top, no overlap -> 0)")


def test_zscore_rrf_fuse():
    z = C.zscore([1.0, 2.0, 3.0, 4.0])
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    assert np.all(C.zscore([5.0, 5.0, 5.0]) == 0)                                   # std floor
    r = C.rrf([np.array([0, 1, 2]), np.array([2, 0, 1])], k=60)
    assert np.allclose(r, [1 / 60 + 1 / 62, 1 / 61 + 1 / 60, 1 / 62 + 1 / 61])
    b, d = np.array([3.0, 0.0, 5.0, 1.0]), np.array([0.1, 0.9, 0.2, 0.5])
    f = C.fuse(b, d, 0.3, method="rrf")                     # ranks b: [1,3,0,2], d: [3,0,2,1]
    assert np.allclose(f, [1 / 61 + 1 / 63, 1 / 63 + 1 / 60, 1 / 60 + 1 / 62, 1 / 62 + 1 / 61])
    assert np.allclose(f, C.fuse(b, d, 0.9, method="rrf"))                          # alpha ignored
    rng = np.random.default_rng(0)
    for _ in range(20):
        b, d, a = rng.random(30) * 10, rng.normal(size=30), rng.random()
        assert np.array_equal(order_of(C.fuse(b, d, 1.0)), order_of(b))              # pure BM25
        assert np.array_equal(order_of(C.fuse(b, d, 0.0)), order_of(d))              # pure dense
        assert np.allclose(C.fuse(b, d, a), a * C.zscore(b) + (1 - a) * C.zscore(d))
    try:
        C.fuse(b, d, 0.5, method="max"); raise AssertionError("unknown method accepted")
    except ValueError:
        pass
    print("  zscore / rrf / fuse OK (alpha=1 -> BM25 order, alpha=0 -> dense order, rrf hand math)")


def _query(rng, n, kind):
    """kind 'lex': evidence barely tops BM25 but is last on dense -> only alpha=1 ranks
    it first. kind 'sem': the mirror -> only alpha=0 does."""
    top, bottom = rng.random(n), rng.random(n)
    ev = int(rng.integers(n))
    top[ev], bottom[ev] = top.max() + 1e-3, -5.0
    return (top, bottom, [ev]) if kind == "lex" else (bottom, top, [ev])


def test_fit_fusion_alpha_logo():
    rng = np.random.default_rng(7)
    groups = [10] * 6 + [20] * 3                   # group 10 lexical (majority), 20 semantic
    qs = [_query(rng, 12, "lex") for _ in range(6)] + [_query(rng, 12, "sem") for _ in range(3)]
    B, Dn, E = zip(*qs)
    res = C.fit_fusion_alpha(B, Dn, E, np.array(groups))
    assert res["grid"] == np.linspace(0, 1, 11).tolist() and len(res["score_by_alpha"]) == 11
    assert set(res["alpha_by_group"]) == {10, 20} and all(type(k) is int for k in res["alpha_by_group"])
    # pooled (in-sample) prefers alpha=1 because the lexical group is larger ...
    assert res["alpha_all"] == 1.0 and np.argmax(res["score_by_alpha"]) == 10
    # ... but held-out group 10 must get the alpha of group 20 alone, and vice versa
    assert res["alpha_by_group"] == {10: 0.0, 20: 1.0}, res["alpha_by_group"]
    # each fold equals a fit on the other groups only
    g = np.array(groups)
    for h in (10, 20):
        keep = np.where(g != h)[0]
        sub = C.fit_fusion_alpha([B[i] for i in keep], [Dn[i] for i in keep], [E[i] for i in keep], g[keep])
        assert sub["alpha_all"] == res["alpha_by_group"][h]
    # perturbing the held-out group's data never changes its own alpha
    for h in (10, 20):
        B2, D2, E2 = list(B), list(Dn), list(E)
        for i in np.where(g == h)[0]:
            B2[i], D2[i], E2[i] = rng.random(12), rng.random(12), [int(rng.integers(12))]
        assert C.fit_fusion_alpha(B2, D2, E2, groups)["alpha_by_group"][h] == res["alpha_by_group"][h]
    # recall@k / hit@k parsing; a single group has no "others" -> 0.5 prior
    r3 = C.fit_fusion_alpha(B, Dn, E, groups, metric="recall@1", grid=[0.0, 0.5, 1.0])
    assert r3["grid"] == [0.0, 0.5, 1.0] and r3["alpha_by_group"] == {10: 0.0, 20: 1.0}
    assert abs(r3["score_by_alpha"][2] - 6 / 9) < 1e-12                           # 6 lexical hits at alpha=1
    assert C.fit_fusion_alpha(B, Dn, E, groups, metric="hit@1")["alpha_by_group"] == {10: 0.0, 20: 1.0}
    assert C.fit_fusion_alpha(B[:2], Dn[:2], E[:2], ["x", "x"])["alpha_by_group"] == {"x": 0.5}
    assert C.fit_fusion_alpha(B[:2], Dn[:2], [[], []], ["x", "y"])["score_by_alpha"] == [0.0] * 11
    for bad in ("ndcg", "recall@", "recall@k"):
        try:
            C.fit_fusion_alpha(B, Dn, E, groups, metric=bad); raise AssertionError(bad)
        except ValueError:
            pass
    print("  LOGO alpha OK (pooled 1.0, held-out groups get the OTHER group's alpha: {10: 0.0, 20: 1.0})")


def test_expand_contiguity():
    bank = make_bank()          # conv 0: rows 0-3 session 1, rows 4-7 session 2; conv 1: rows 8-11
    ec = C.expand_contiguity
    assert ec(bank, [5, 3, 6]).tolist() == [5, 4, 6, 3, 2, 7]           # anchor, then its new neighbours
    assert ec(bank, [3]).tolist() == [3, 2]                              # 4 is in the next session
    assert ec(bank, [3], same_session=False).tolist() == [3, 2, 4]
    assert ec(bank, [7], same_session=False).tolist() == [7, 6]          # never crosses into conv 1
    assert ec(bank, [5, 3, 6], limit=4).tolist() == [5, 4, 6, 3]         # truncation
    assert ec(bank, [5, 3, 6], limit=0).tolist() == []
    assert ec(bank, [5, 5, 3, 5], n=0).tolist() == [5, 3]                # n=0: dedup only
    assert ec(bank, [1], n=2, same_session=False).tolist() == [1, 0, 2, 3]
    assert ec(bank, [2], n=2).tolist() == [2, 1, 3, 0]                   # bank order: nearest first
    assert ec(bank, [2, 1], n=1).tolist() == [2, 1, 3, 0]                # 1 already in: add only 0
    out = ec(bank, np.array([9, 10]))
    assert out.dtype == np.int64 and out.tolist() == [9, 8, 10, 11]
    print("  contiguity OK (order, dedup, truncation, same_session, conversation boundary)")


def test_shortlist():
    bank = make_bank()
    enc = C.Encoder("fake", fn=fake_fn, cache_dir=None)
    gen = C.CandidateGenerator(bank, enc, alpha=0.5, K=3).index()
    assert set(gen._conv) == {0, 1} and gen.doc_emb.shape == (12, D)
    s = gen.scores(QUERY, 0)
    assert s["rows"].tolist() == list(range(8)) and all(len(s[k]) == 8 for k in ("bm25", "dense", "fused"))
    assert order_of(s["bm25"])[0] == 1 and order_of(s["dense"])[0] == 4    # decoy vs planted pair
    # alpha extremes reproduce the pure BM25 / pure dense rankings
    assert gen.shortlist(QUERY, 0, alpha=1.0)["rows"].tolist() == order_of(s["bm25"])[:3].tolist()
    assert gen.shortlist(QUERY, 0, alpha=0.0, K=8)["rows"].tolist() == order_of(s["dense"]).tolist()
    # per-conversation scope: the verbatim copy of the query lives in conv 1 and never appears
    for K in (1, 3, 8, 50):
        sl = gen.shortlist(QUERY, 0, K=K)
        assert sl["rows"].dtype == np.int64 and len(sl["rows"]) == min(K, 8)
        assert set(sl["rows"].tolist()) <= set(range(8))
    assert gen.shortlist(QUERY, 1, K=1)["rows"].tolist() == [8]
    # returned fields align with the full conversation scores
    sl = gen.shortlist(QUERY, 0, K=4, alpha=0.3)
    f = C.fuse(s["bm25"], s["dense"], 0.3)
    assert np.allclose(sl["fused"], f[sl["rows"]]) and np.allclose(sl["dense"], s["dense"][sl["rows"]])
    assert np.allclose(sl["bm25_z"], C.zscore(s["bm25"])[sl["rows"]])
    assert sl["rank"].tolist() == [0, 1, 2, 3] and sl["rows"].tolist() == order_of(f)[:4].tolist()
    # precomputed query embedding == on-the-fly encoding
    q = enc.encode_queries([QUERY])[0]
    assert all(np.array_equal(a, b) for a, b in zip(gen.shortlist(QUERY, 0, q_emb=q).values(),
                                                    gen.shortlist(QUERY, 0).values()))
    assert np.allclose(gen.scores(QUERY, 0, q_emb=3.0 * q)["dense"], s["dense"], atol=1e-6)  # cosine
    # contiguity: anchors = top-K by fused, each followed by same-session neighbours, cut to K
    gc = C.CandidateGenerator(bank, enc, alpha=0.0, K=4, contiguity=1).index()
    cs = gc.shortlist(QUERY, 0)
    dord = order_of(s["dense"])
    anchors = dord[:4]
    assert cs["rows"].tolist() == C.expand_contiguity(bank, anchors, n=1, limit=4).tolist()
    assert cs["rows"][0] == 4 and len(cs["rows"]) == 4
    rk = {int(r): p for p, r in enumerate(dord)}
    assert cs["rank"].tolist() == [rk[int(r)] for r in cs["rows"]]
    contig_only = [int(r) for r in cs["rows"] if int(r) not in set(anchors.tolist())]
    assert [int(r) for r, k in zip(cs["rows"], cs["rank"]) if k >= 4] == contig_only   # pipeline's flag
    for r in cs["rows"]:
        assert r in anchors or any(r in bank.neighbors(int(a), 1, True) for a in anchors)
    # lazy index: scores() before index(); bank without .memories
    lazy = C.CandidateGenerator(FakeBank.__new__(FakeBank), enc, K=2)
    lazy.bank.texts, lazy.bank.rows_of_conv = bank.texts, bank.rows_of_conv
    assert np.array_equal(lazy.shortlist(QUERY, 0, alpha=0.5)["rows"], gen.shortlist(QUERY, 0, K=2)["rows"])
    # rrf generator runs and ignores alpha
    gr = C.CandidateGenerator(bank, enc, method="rrf", K=3).index()
    assert np.array_equal(gr.shortlist(QUERY, 0, alpha=0.0)["rows"], gr.shortlist(QUERY, 0, alpha=1.0)["rows"])
    print(f"  shortlist OK (conv-scoped, alpha extremes, aligned fields, q_emb, "
          f"contiguity {cs['rows'].tolist()})")


class CountingEncoder(C.Encoder):
    """Exercises the fn=None disk-cache path with a fake model call."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = []

    def _embed(self, texts):
        self.calls.append(list(texts))
        return torch.as_tensor(fake_fn(texts), dtype=torch.float64)


def test_encoder_cache():
    docs, qs = ["Alice: hi", "Bob: a zebra!"], ["who said hi?"]
    # fn path: normalised float32, instruction prefixed, nothing written to disk
    with tempfile.TemporaryDirectory() as tmp:
        seen = []
        enc = C.Encoder("fake/model", query_instruction="Q: ", cache_dir=tmp,
                        fn=lambda t: (seen.append(list(t)), fake_fn(t))[1])
        v = enc.encode_docs(docs)
        assert v.dtype == torch.float32 and v.shape == (2, D) and torch.allclose(v.norm(dim=1), torch.ones(2))
        enc.encode_queries(qs)
        assert seen == [docs, ["Q: who said hi?"]] and os.listdir(tmp) == [] and enc.name == "fake/model"
        assert enc.encode_queries(["x"]).shape == (1, D)
    # disk cache round trip (fn=None, model call injected by subclass)
    with tempfile.TemporaryDirectory() as tmp:
        cdir = os.path.join(tmp, "cache")                                   # created on demand
        a = CountingEncoder("org/fake-v1", cache_dir=cdir)
        v1 = a.encode_docs(docs)
        p = a.cache_path("d", docs)
        md5 = hashlib.md5(("org/fake-v1" + "d" + "\n".join(docs)).encode()).hexdigest()
        assert os.path.exists(p) and p.endswith(md5 + ".pt") and os.path.dirname(p) == cdir
        assert len(a.calls) == 1 and torch.allclose(v1.norm(dim=1), torch.ones(2))
        b = CountingEncoder("org/fake-v1", cache_dir=cdir)
        assert torch.equal(b.encode_docs(docs), v1) and b.calls == []          # loaded from disk
        # query cache keys include the instruction; docs and queries never collide
        q0 = b.encode_queries(qs)
        c = CountingEncoder("org/fake-v1", query_instruction="Represent: ", cache_dir=cdir)
        q1 = c.encode_queries(qs)
        assert c.calls == [["Represent: who said hi?"]] and not torch.equal(q0, q1)
        keys = {a.cache_path("d", qs), a.cache_path("q", qs), c.cache_path("q", ["Represent: " + qs[0]])}
        assert len(keys) == 3
        assert CountingEncoder("org/fake-v1", cache_dir=cdir).encode_queries(qs).equal(q0)
        assert CountingEncoder("org/fake-v2", cache_dir=cdir).cache_path("d", docs) != p   # model in key
        assert not [f for f in os.listdir(cdir) if f.endswith(".tmp")]
        assert len(os.listdir(cdir)) == 3
    print("  Encoder OK (fn path uncached + normalised; disk cache round trip; instruction in key)")


def test_store_bank():
    """Same checks on the real store.MemoryBank (skipped if store.py is absent)."""
    try:
        from manifold_mvp.store import Memory, MemoryBank
    except ImportError:
        print("  store.MemoryBank not importable: skipped"); return
    fb = make_bank()
    order = {0: 0, 1: 0}
    mems = []
    for i, m in enumerate(fb.memories):
        mems.append(Memory(row=i, conv=m.conv, session=m.session, order=order[m.conv],
                           dia_id=f"D{m.session}:{i}", speaker=m.speaker, date="", text=m.text))
        order[m.conv] += 1
    bank = MemoryBank(mems)
    for rows, n, ss in [([5, 3, 6], 1, True), ([3], 1, False), ([7], 1, False), ([2], 2, True),
                        ([9, 10], 1, True)]:
        assert C.expand_contiguity(bank, rows, n, ss).tolist() == C.expand_contiguity(fb, rows, n, ss).tolist()
    enc = C.Encoder("fake", fn=fake_fn, cache_dir=None)
    for kw in (dict(K=3), dict(K=4, contiguity=1, alpha=0.0), dict(K=5, method="rrf", contiguity=2)):
        a, b = C.CandidateGenerator(bank, enc, **kw).index(), C.CandidateGenerator(fb, enc, **kw).index()
        assert set(a._conv) == {0, 1}
        for conv in (0, 1):
            sa, sb = a.shortlist(QUERY, conv), b.shortlist(QUERY, conv)
            assert all(np.array_equal(sa[k], sb[k]) for k in sa)
            assert set(sa["rows"].tolist()) <= set(bank.rows_of_conv(conv).tolist())
    print("  store.MemoryBank OK (same contiguity and shortlists as the duck-typed fake)")


if __name__ == "__main__":
    for fn in [test_tokenize_bm25, test_zscore_rrf_fuse, test_fit_fusion_alpha_logo,
               test_expand_contiguity, test_shortlist, test_encoder_cache, test_store_bank]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nALL CANDIDATES CHECKS PASSED")
