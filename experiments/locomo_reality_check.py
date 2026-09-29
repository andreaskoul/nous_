#!/usr/bin/env python3
"""
LoCoMo REALITY CHECK — run nous_ on real data against honest baselines.

Question -> evidence-turn retrieval on LoCoMo (snap-research, CC BY-NC 4.0),
5-fold leave-conversations-out cross-validation (every question is scored by a
model that never saw its conversation). Everything is reported with 95% CIs
from a CLUSTER bootstrap over conversations (questions within a conversation are
not independent), plus a question-level paired bootstrap for the key contrasts.

Blocks
  A  retrieval   : random / filesystem / BM25 / dense (several encoders) /
                   hybrid (RRF) / hybrid + cross-encoder rerank /
                   nous_ heads (static, context-as-implemented, residual-context)
                   in two scopes:
                     per-conversation  (realistic: an agent searches its own user's memory)
                     pooled            (the scope Gate-1 was written for) + oracle conversation filter
  B  Gate-1      : the pre-registered rule (context - best baseline >= 0.15 top-1),
                   as implemented and against the honest baseline set
  C  geometry    : learned conformal metric on a real PCA chart — does lambda_theta
                   collapse to 1 (flat)? E0 curvature, E1 geodesic-vs-straight rerank
  D  System-One  : is the Locate density rho a calibrated decision? ECE, selective
                   prediction AUROC, and can max-rho flag adversarial (cat-5) questions?

Usage:
    python experiments/locomo_reality_check.py                 # full run
    python experiments/locomo_reality_check.py --quick         # 2 folds, no rerank
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, sys, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from manifold_mvp.locate import ContextHead, StaticHead, filesystem_lookup  # noqa: E402
from manifold_mvp import losses  # noqa: E402
from manifold_mvp.metric import PullbackMetric  # noqa: E402
from manifold_mvp.conformal import ConformalHead  # noqa: E402
from manifold_mvp.geodesic import geodesic, straight_length  # noqa: E402
from manifold_mvp.curvature import curvature_probe  # noqa: E402
from manifold_mvp.real import TorchPCA  # noqa: E402

# Category ids as used by LoCoMo evaluations (e.g. Mem0): mapping is conventional,
# the raw file only stores integers.
CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
SESSION_RE = re.compile(r"^session_(\d+)$")
DIA_RE = re.compile(r"D\d+:\d+")
BETA = 8.0  # config.ModelCfg.beta — the head is evaluated as designed


# ---------------------------------------------------------------- data

def load_locomo(path):
    data = json.load(open(path))
    turns, conv_of, sess_of, qa = [], [], [], []
    unresolved = 0
    for ci, s in enumerate(data):
        conv = s["conversation"]
        off = len(turns)
        local = {}
        sess = sorted((k for k in conv if SESSION_RE.match(k)),
                      key=lambda k: int(SESSION_RE.match(k).group(1)))
        for sk in sess:
            sidx = int(SESSION_RE.match(sk).group(1))
            for t in conv[sk]:
                text = t.get("text", "")
                if t.get("blip_caption"):
                    text = f"{text} [shares image: {t['blip_caption']}]"
                local[t["dia_id"]] = len(turns) - off
                turns.append(f"{t['speaker']}: {text}")
                conv_of.append(ci)
                sess_of.append(sidx)
        n = len(turns) - off
        for item in s["qa"]:
            ids = []
            for e in item.get("evidence") or []:
                ids += DIA_RE.findall(str(e))
            ev = sorted({off + local[e] for e in ids if e in local})
            unresolved += sum(1 for e in ids if e not in local)
            if ev:
                qa.append(dict(conv=ci, q=item["question"], cat=int(item["category"]),
                               ev=np.array(ev), off=off, n=n))
    return turns, np.array(conv_of), np.array(sess_of), qa, unresolved


# ---------------------------------------------------------------- encoders

_MODELS = {}


def encode(name, texts, is_query, cache_dir):
    h = hashlib.md5(("\n".join(texts) + name + str(is_query)).encode()).hexdigest()[:12]
    path = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9]+", "_", name) + f"_{'q' if is_query else 'd'}_{h}.pt")
    if os.path.exists(path):
        return torch.load(path)
    from sentence_transformers import SentenceTransformer
    if name not in _MODELS:
        _MODELS[name] = SentenceTransformer(name, device="cpu")
    m = _MODELS[name]
    kw = {}
    if is_query and "bge-" in name.lower():
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    if is_query and "qwen3-embedding" in name.lower():
        kw["prompt_name"] = "query"
    v = m.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True,
                 show_progress_bar=False, **kw).float().cpu()
    torch.save(v, path)
    return v


# ---------------------------------------------------------------- metrics

KS = (1, 5, 10)


def per_question(scores, cands, ev):
    """scores over `cands` (np arrays) -> dict of per-question metrics."""
    order = np.argsort(-scores, kind="stable")
    pos = {int(c): r for r, c in enumerate(cands[order])}
    ranks = np.array([pos[int(e)] for e in ev if int(e) in pos])
    if ranks.size == 0:
        out = {"mrr": 0.0}
        out.update({f"hit@{k}": 0.0 for k in KS})
        out.update({f"recall@{k}": 0.0 for k in KS})
        return out
    out = {"mrr": 1.0 / (ranks.min() + 1)}
    for k in KS:
        out[f"hit@{k}"] = float(ranks.min() < k)
        out[f"recall@{k}"] = float((ranks < k).sum() / len(ev))
    return out


def cluster_ci(values, convs, n=2000, seed=0):
    """95% CI of the mean by resampling CONVERSATIONS (the honest unit)."""
    rng = np.random.default_rng(seed)
    uc = np.unique(convs)
    groups = [values[convs == c] for c in uc]
    stats = []
    for _ in range(n):
        pick = rng.integers(0, len(uc), len(uc))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def paired_diff(a, b, convs, n=2000, seed=1):
    d = a - b
    rng = np.random.default_rng(seed)
    q = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    mean, clo, chi = cluster_ci(d, convs, n=n, seed=seed)
    qlo, qhi = np.percentile(q, [2.5, 97.5])
    return dict(mean=mean, q_ci=[float(qlo), float(qhi)], cluster_ci=[clo, chi])


def auroc(scores, labels):
    """P(score_pos > score_neg) via ranks (Mann-Whitney)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# ---------------------------------------------------------------- nous_ heads

class ResidualContextHead(nn.Module):
    """Candidate FIX: start at the pretrained geometry (phi = q) and only learn a
    state-conditioned residual. The as-implemented ContextHead maps [q,s,h] -> dim
    from scratch, which throws away the encoder's similarity structure."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q, s, h):
        phi = q + self.net(torch.cat([q, s, h], dim=-1))
        return phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def question_state(T, conv_of, sess_of, qa):
    """Honest, query-time-available state. s = conversation fingerprint;
    h = mean of the LAST session's turns (all LoCoMo questions are asked after the
    conversation ends). NOTE: manifold_mvp/real.py defines h from turns *before
    the evidence*, which leaks the label; that is not used here."""
    S, H = [], []
    cache = {}
    for item in qa:
        c = item["conv"]
        if c not in cache:
            rows = np.where(conv_of == c)[0]
            last = rows[sess_of[rows] == sess_of[rows].max()]
            s = T[rows].mean(0)
            h = T[last].mean(0)
            cache[c] = (s / s.norm().clamp_min(1e-9), h / h.norm().clamp_min(1e-9))
        S.append(cache[c][0]); H.append(cache[c][1])
    return torch.stack(S), torch.stack(H)


def train_head(kind, Q, S, H, tgt, bank, steps=400, seed=0):
    torch.manual_seed(seed)
    dim = bank.shape[1]
    head = {"static": lambda: StaticHead(dim), "context": lambda: ContextHead(dim),
            "residual": lambda: ResidualContextHead(dim)}[kind]()
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, Q.shape[0], (256,), generator=g)
        phi = head(Q[idx], S[idx], H[idx])
        loss = losses.retrieval_infonce(phi, bank, tgt[idx], beta=BETA)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/locomo10.json"))
    ap.add_argument("--encoders", default="sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5")
    ap.add_argument("--head-encoders", default="sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--rerank-depth", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments/results"))
    args = ap.parse_args()
    if args.quick:
        args.folds, args.reranker = 2, ""
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    cache_dir = os.path.join(ROOT, "data/cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    turns, conv_of, sess_of, qa, unresolved = load_locomo(args.data)
    nconv = int(conv_of.max()) + 1
    qconv = np.array([x["conv"] for x in qa])
    qcat = np.array([x["cat"] for x in qa])
    questions = [x["q"] for x in qa]
    print(f"LoCoMo: {nconv} conversations, {len(turns)} turns, {len(qa)} QA with resolvable evidence "
          f"({unresolved} unresolved evidence ids dropped)")
    folds = [list(range(i, nconv, args.folds)) for i in range(args.folds)] if args.folds != 5 else \
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    if args.quick:
        folds = [[0, 1], [8, 9]]
    evalq = np.array([x["conv"] in {c for f in folds for c in f} for x in qa])

    def per_conv_cands(i):
        o, n = qa[i]["off"], qa[i]["n"]
        return np.arange(o, o + n)

    results = defaultdict(dict)   # method -> per-question metric arrays (NaN where not evaluated)
    rhostats = {}

    def store(method, i, m):
        for k, v in m.items():
            results[method].setdefault(k, np.full(len(qa), np.nan))[i] = v

    # ---- A1: trivial floors -------------------------------------------------
    rng = np.random.default_rng(0)
    for i in np.where(evalq)[0]:
        c = per_conv_cands(i)
        store("random", i, per_question(rng.random(len(c)), c, qa[i]["ev"]))

    # ---- A2: BM25 -----------------------------------------------------------
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm25_scores = {}
    for c in range(nconv):
        rows = np.where(conv_of == c)[0]
        bm = BM25Okapi([tok(turns[r]) for r in rows])
        for i in np.where((qconv == c) & evalq)[0]:
            sc = bm.get_scores(tok(questions[i]))
            bm25_scores[i] = sc
            store("bm25", i, per_question(sc, rows, qa[i]["ev"]))
    print(f"  BM25 done ({time.time()-t0:.0f}s)")

    # ---- A3: dense encoders (+ filesystem on the first) ---------------------
    embs = {}
    for enc in [e for e in args.encoders.split(",") if e]:
        T = encode(enc, turns, False, cache_dir)
        Qe = encode(enc, questions, True, cache_dir)
        embs[enc] = (T, Qe)
        sims = (Qe @ T.T).numpy()
        short = enc.split("/")[-1]
        for i in np.where(evalq)[0]:
            c = per_conv_cands(i)
            store(f"dense:{short}", i, per_question(sims[i, c], c, qa[i]["ev"]))
        if enc == args.encoders.split(",")[0]:
            # dumb exact-key storage returns at most ONE item (or a miss)
            for i in np.where(evalq)[0]:
                c = per_conv_cands(i)
                hit = filesystem_lookup(Qe[i:i + 1], T[c])[0].item()
                ok = float(hit >= 0 and int(c[hit]) in set(qa[i]["ev"].tolist()))
                store("filesystem", i, {"mrr": ok, **{f"hit@{k}": ok for k in KS},
                                        **{f"recall@{k}": ok / len(qa[i]["ev"]) for k in KS}})
        print(f"  dense {short} done ({time.time()-t0:.0f}s)")

    # ---- A4: hybrid RRF (BM25 + strongest dense) + cross-encoder rerank ------
    strongest = args.encoders.split(",")[-1]
    T, Qe = embs[strongest]
    sims = (Qe @ T.T).numpy()
    hybrid_order = {}
    for i in np.where(evalq)[0]:
        c = per_conv_cands(i)
        rb = np.argsort(np.argsort(-bm25_scores[i]))
        rd = np.argsort(np.argsort(-sims[i, c]))
        rrf = 1.0 / (60 + rb) + 1.0 / (60 + rd)
        hybrid_order[i] = rrf
        store("hybrid:bm25+" + strongest.split("/")[-1], i, per_question(rrf, c, qa[i]["ev"]))
    if args.reranker:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(args.reranker, device="cpu")
        pairs, owners = [], []
        tops = {}
        for i in np.where(evalq)[0]:
            c = per_conv_cands(i)
            top = np.argsort(-hybrid_order[i])[: args.rerank_depth]
            tops[i] = top
            pairs += [(questions[i], turns[c[j]]) for j in top]
            owners += [i] * len(top)
        ce_sc = ce.predict(pairs, batch_size=128, show_progress_bar=False)
        ptr = 0
        for i in np.where(evalq)[0]:
            c = per_conv_cands(i)
            top = tops[i]
            sc = hybrid_order[i] - 10.0          # everything below the reranked block
            sc[top] = 10.0 + ce_sc[ptr: ptr + len(top)]
            ptr += len(top)
            store("hybrid+rerank", i, per_question(sc, c, qa[i]["ev"]))
        print(f"  rerank done ({time.time()-t0:.0f}s)")

    # ---- A5: nous_ heads, leave-conversations-out ----------------------------
    gate_rows = []
    for enc in [e for e in args.head_encoders.split(",") if e]:
        if enc not in embs:
            embs[enc] = (encode(enc, turns, False, cache_dir), encode(enc, questions, True, cache_dir))
        T, Qe = embs[enc]
        S, H = question_state(T, conv_of, sess_of, qa)
        short = enc.split("/")[-1]
        for f_i, test in enumerate(folds):
            train_c = [c for c in range(nconv) if c not in test]
            tr_rows = np.where(np.isin(conv_of, train_c))[0]
            te_rows = np.where(np.isin(conv_of, test))[0]
            remap = {int(r): j for j, r in enumerate(tr_rows)}
            tr_q = [i for i in range(len(qa)) if qa[i]["conv"] in train_c]
            tgt = torch.tensor([remap[int(qa[i]["ev"][0])] for i in tr_q])
            bank_tr = T[tr_rows]
            te_q = np.where(np.isin(qconv, test))[0]
            for kind in ("static", "context", "residual"):
                head = train_head(kind, Qe[tr_q], S[tr_q], H[tr_q], tgt, bank_tr, seed=f_i)
                with torch.no_grad():
                    phi = head(Qe[te_q], S[te_q], H[te_q])
                    sc_all = (phi @ T.T).numpy()
                for j, i in enumerate(te_q):
                    store(f"{kind}-head:{short}", i, per_question(sc_all[j, per_conv_cands(i)], per_conv_cands(i), qa[i]["ev"]))
                    store(f"{kind}-head:{short}@pooled", i, per_question(sc_all[j, te_rows], te_rows, qa[i]["ev"]))
            # pooled-scope baselines for the SAME fold (the scope Gate-1 was written for)
            for j, i in enumerate(te_q):
                store(f"dense:{short}@pooled", i, per_question((Qe[i] @ T[te_rows].T).numpy(), te_rows, qa[i]["ev"]))
        print(f"  heads on {short} done ({time.time()-t0:.0f}s)")

    # ---- summarise A ---------------------------------------------------------
    main_mask = evalq & (qcat != 5)
    summary = {}
    for method, mets in results.items():
        row = {}
        m = main_mask & ~np.isnan(mets["mrr"])
        if not m.any():
            continue
        for k in ["hit@1", "hit@5", "hit@10", "recall@10", "mrr"]:
            mean, lo, hi = cluster_ci(mets[k][m], qconv[m])
            row[k] = [round(mean, 4), round(lo, 4), round(hi, 4)]
        row["by_category_hit@5"] = {CATS[c]: round(float(np.nanmean(mets["hit@5"][evalq & (qcat == c)])), 4)
                                   for c in CATS if (evalq & (qcat == c) & ~np.isnan(mets["hit@5"])).any()}
        row["n"] = int(m.sum())
        summary[method] = row

    # ---- B: Gate-1 as pre-registered (top-1, delta2 = 0.15) ------------------
    gate = {}
    for enc in [e for e in args.head_encoders.split(",") if e]:
        short = enc.split("/")[-1]
        ctx = f"context-head:{short}@pooled"
        base_impl = {k: summary[k]["hit@1"][0] for k in [f"dense:{short}@pooled", "filesystem", f"static-head:{short}@pooled"] if k in summary}
        honest = dict(base_impl)
        honest.update({k: summary[k]["hit@1"][0] for k in [f"dense:{short}", "bm25", "hybrid+rerank",
                                                           "hybrid:bm25+" + strongest.split("/")[-1]] if k in summary})
        c_acc = summary[ctx]["hit@1"][0]
        gate[short] = dict(
            context_top1=c_acc,
            as_implemented=dict(baselines=base_impl, best=max(base_impl.values()),
                                delta=round(c_acc - max(base_impl.values()), 4),
                                pass_=bool(c_acc - max(base_impl.values()) >= 0.15)),
            honest=dict(baselines=honest, best=max(honest.values()),
                        delta=round(c_acc - max(honest.values()), 4),
                        pass_=bool(c_acc - max(honest.values()) >= 0.15)),
        )
    contrasts = {}
    for enc in [e for e in args.head_encoders.split(",") if e]:
        short = enc.split("/")[-1]
        pairs_ = {
            f"context@pooled - dense@pooled ({short})": (f"context-head:{short}@pooled", f"dense:{short}@pooled"),
            f"context@pooled - dense per-conv [oracle conv filter] ({short})": (f"context-head:{short}@pooled", f"dense:{short}"),
            f"residual per-conv - dense per-conv ({short})": (f"residual-head:{short}", f"dense:{short}"),
            f"context per-conv - dense per-conv ({short})": (f"context-head:{short}", f"dense:{short}"),
        }
        for name, (a, b) in pairs_.items():
            if a in results and b in results:
                m = main_mask & ~np.isnan(results[a]["mrr"]) & ~np.isnan(results[b]["mrr"])
                contrasts[name] = {k: paired_diff(results[a][k][m], results[b][k][m], qconv[m]) for k in ("hit@1", "mrr")}

    # ---- C: geometry on a REAL chart ----------------------------------------
    geo = {}
    enc = args.head_encoders.split(",")[0]
    T, Qe = embs[enc]
    test = folds[-1]
    tr_rows = np.where(~np.isin(conv_of, test))[0]
    pca = TorchPCA(16).fit(T[tr_rows])
    lat_tr = pca.encode(T[tr_rows])
    torch.manual_seed(0)
    lam = ConformalHead(16, hidden=64)
    metric = PullbackMetric(pca.decode, conformal=lam)
    opt = torch.optim.Adam(lam.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(0)
    for it in range(300):
        pick = lambda n: lat_tr[torch.randint(0, lat_tr.shape[0], (n,), generator=gen)]
        z_on = pick(32)
        z_off = z_on + 3.0 * z_on.std() * torch.randn(z_on.shape, generator=gen)
        a, b = pick(32), pick(32)
        d_t = (pca.decode(a) - pca.decode(b)).norm(dim=-1)
        loss, info = losses.geometry_loss(metric, z_on, z_off, a, b, d_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        l_on = lam(lat_tr)
        l_off = lam(lat_tr + 3.0 * lat_tr.std() * torch.randn(lat_tr.shape, generator=gen))
    geo["lambda_on_data"] = dict(mean=float(l_on.mean()), std=float(l_on.std()), min=float(l_on.min()), max=float(l_on.max()))
    geo["lambda_off_manifold_mean"] = float(l_off.mean())
    geo["final_warp"], geo["final_dist_match"] = info["warp"], info["dist_match"]
    te_rows = np.where(np.isin(conv_of, test))[0]
    lat_te = pca.encode(T[te_rows])
    e0 = curvature_probe(metric, lat_te[:200], n_pairs=16, steps=60)
    geo["E0_shortcut"] = e0["shortcut (1-ratio)"]
    geo["E0_go"] = e0["go"]
    # E1 on real questions: rerank the 20 nearest turns (latent) by straight vs geodesic length
    te_q = [i for i in np.where(np.isin(qconv, test))[0] if qa[i]["cat"] != 5][:60]
    hits = defaultdict(int)
    for i in te_q:
        c = per_conv_cands(i)
        zq = pca.encode(Qe[i:i + 1])[0]
        zc = pca.encode(T[c])
        top = torch.argsort((zc - zq).norm(dim=-1))[:20]
        ev = set(int(e) for e in qa[i]["ev"])
        cos = (Qe[i] @ T[c[top.numpy()]].T)
        eu = (zc[top] - zq).norm(dim=-1)
        st = torch.stack([straight_length(metric, zq, zc[j], n_segments=10) for j in top])
        ge = torch.stack([geodesic(metric, zq, zc[j], n_segments=10, steps=50)[1] for j in top])
        for name, sc in [("cosine_full_dim", -cos), ("euclid_latent", eu), ("straight_metric", st), ("geodesic_metric", ge)]:
            hits[name] += int(int(c[top[int(torch.argmin(sc))]]) in ev)
    geo["E1_real_top1_among_20"] = {k: round(v / len(te_q), 4) for k, v in hits.items()}
    geo["E1_n_questions"] = len(te_q)
    print(f"  geometry done ({time.time()-t0:.0f}s)")

    # ---- D: is rho a calibrated System-One decision? ------------------------
    sysone = {}
    for enc in [e for e in args.encoders.split(",") if e]:
        T, Qe = embs[enc]
        sims = (Qe @ T.T).numpy()
        short = enc.split("/")[-1]

        def rho_stats(beta, idxs):
            conf, corr, marg, nll = [], [], [], []
            for i in idxs:
                c = per_conv_cands(i)
                z = beta * sims[i, c]
                p = np.exp(z - z.max()); p /= p.sum()
                o = np.argsort(-p)
                conf.append(p[o[0]]); marg.append(p[o[0]] - p[o[1]])
                ev_local = np.array(qa[i]["ev"]) - qa[i]["off"]
                corr.append(o[0] in set(ev_local.tolist()))
                nll.append(-np.log(p[ev_local].sum() + 1e-12))
            return np.array(conf), np.array(corr), np.array(marg), float(np.mean(nll))

        per_fold = []
        confs, corrs, margs, cats_ = [], [], [], []
        for test in folds:
            tr = [i for i in np.where(evalq)[0] if qa[i]["conv"] not in test and qa[i]["cat"] != 5]
            te = [i for i in np.where(evalq)[0] if qa[i]["conv"] in test]
            grid = [1, 2, 4, 8, 12, 16, 20, 30, 40, 60, 80, 100]
            beta = min(grid, key=lambda b: rho_stats(b, tr)[3])    # temperature fit on OTHER conversations
            cf, co, mg, _ = rho_stats(beta, te)
            per_fold.append(beta)
            confs.append(cf); corrs.append(co); margs.append(mg); cats_.append(qcat[te])
        cf, co, mg, ct = map(np.concatenate, (confs, corrs, margs, cats_))
        cf8, co8, _, _ = rho_stats(BETA, np.where(evalq)[0])
        ans = ct != 5
        sysone[short] = dict(
            fitted_beta_per_fold=per_fold,
            ece_fitted=round(ece(cf[ans], co[ans]), 4),
            ece_beta8=round(ece(cf8[qcat[evalq] != 5], co8[qcat[evalq] != 5]), 4),
            top1_acc=round(float(co[ans].mean()), 4),
            mean_conf=round(float(cf[ans].mean()), 4),
            auroc_conf_predicts_correct=round(auroc(cf[ans], co[ans]), 4),
            auroc_margin_predicts_correct=round(auroc(mg[ans], co[ans]), 4),
            auroc_lowconf_flags_adversarial=round(auroc(-cf, ct == 5), 4),
        )
    print(f"  system-one calibration done ({time.time()-t0:.0f}s)")

    out = dict(dataset=dict(conversations=nconv, turns=len(turns), qa=len(qa), unresolved_evidence=unresolved,
                            evaluated=int(evalq.sum()), main_eval_excludes="category 5 (adversarial)"),
               folds=folds, retrieval=summary, gate1=gate, contrasts=contrasts, geometry=geo,
               system_one=sysone, runtime_s=round(time.time() - t0, 1))
    tag = "quick" if args.quick else "full"
    jpath = os.path.join(args.out, f"locomo_reality_check_{tag}.json")
    json.dump(out, open(jpath, "w"), indent=2)

    # ---- print ---------------------------------------------------------------
    print("\n== A  retrieval (cats 1-4, cluster-bootstrap 95% CI) ==")
    print(f"{'method':46s} {'hit@1':>20s} {'hit@5':>20s} {'MRR':>20s}")
    for mth, row in sorted(summary.items(), key=lambda kv: -kv[1]["mrr"][0]):
        f = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
        print(f"{mth:46s} {f(row['hit@1']):>20s} {f(row['hit@5']):>20s} {f(row['mrr']):>20s}")
    print("\n== B  Gate-1 ==")
    print(json.dumps(gate, indent=1))
    print("\n== contrasts ==")
    print(json.dumps(contrasts, indent=1))
    print("\n== C  geometry on a real chart ==")
    print(json.dumps(geo, indent=1))
    print("\n== D  System-One view of rho ==")
    print(json.dumps(sysone, indent=1))
    print(f"\nwrote {jpath}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
