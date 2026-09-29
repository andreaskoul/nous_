"""
LAYER 1 — CANDIDATES: a training-free, per-conversation shortlist.

Nothing downstream (Locate, a System-One reranker, the LLM reader) can recover a
memory that is not in the shortlist, so L1 is judged by RECALL, and it has to be
strong without training:

  lexical   BM25 over lowercase \\w+ tokens. LoCoMo is overwhelmingly lexical: 98.9%
            of its questions are substring-solvable (arXiv 2603.15599).
  dense     cosine of L2-normalised sentence embeddings. By default there is no
            query instruction, because instructions can HURT on dialogue memory
            (LMEB, 2603.12572); it stays an explicit, cache-keyed option.
  fusion    f(m) = alpha * z(BM25) + (1 - alpha) * z(cos), z = within-query z-score
            over the conversation. Fusing BM25 with dense similarity is a strong
            training-free first stage (2606.04194). In-repo reality check (hit@1):
            RRF(BM25, bge-base) 0.304 > bge-small 0.284 > BM25 0.264. The z-score
            form keeps score MAGNITUDES that RRF throws away and exposes one weight,
            alpha, fitted leave-one-conversation-out (fit_fusion_alpha). "rrf" is
            kept, with the reality check's 0-based ranks, so X0 can reproduce it.
  contiguity each anchor is followed by its +-n same-session neighbours: evidence
            clusters in time, and temporal/contiguity neighbours help recall (CueMem
            2609.12354, EM-LLM 2407.09450). The expanded list is truncated back to K,
            so contiguity spends shortlist budget and does not grow it.
  scope     ONE conversation per query. Every production memory system partitions
            memory by user or conversation.

The bank is duck-typed (texts, rows_of_conv(conv), neighbors(row, n, same_session),
optionally memories[*].conv) so this module does not import store.py.
"""
from __future__ import annotations
import hashlib
import os
import re

import numpy as np
import torch

TOKEN_RE = re.compile(r"\w+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


# ---- encoders -----------------------------------------------------------------

class Encoder:
    """Frozen text encoder -> (N, D) L2-normalised float32, cached on disk.
    fn: optional callable list[str] -> array/tensor used INSTEAD of
    sentence-transformers (tests, custom models); its outputs are not disk-cached."""

    def __init__(self, model_id, query_instruction=None, cache_dir="data/cache", device="cpu",
                 batch_size=64, fn=None):
        self.model_id, self.query_instruction, self.cache_dir = model_id, query_instruction, cache_dir
        self.device, self.batch_size, self.fn = device, batch_size, fn
        self._model = None

    @property
    def name(self):
        return self.model_id

    def cache_path(self, kind, texts):
        """kind 'd' (documents) or 'q' (queries). Key = md5(model_id + kind + the
        strings actually encoded), so a query instruction is part of the key."""
        h = hashlib.md5((self.model_id + kind + "\n".join(texts)).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{re.sub(r'[^A-Za-z0-9]+', '_', self.model_id)}_{kind}_{h}.pt")

    def _embed(self, texts):
        """The raw model call (no cache). sentence-transformers is imported lazily."""
        if self.fn is not None:
            return self.fn(texts)
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)

    def _encode(self, kind, texts):
        texts = list(texts)
        path = None if self.fn is not None or not self.cache_dir else self.cache_path(kind, texts)
        if path and os.path.exists(path):
            return torch.load(path)
        v = self._embed(texts)
        v = v.detach().cpu() if isinstance(v, torch.Tensor) else torch.as_tensor(np.asarray(v))
        v = torch.nn.functional.normalize(torch.atleast_2d(v.float()), dim=-1)
        if path:
            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save(v, path + ".tmp")
            os.replace(path + ".tmp", path)          # atomic: no half-written cache files
        return v

    def encode_docs(self, texts):
        return self._encode("d", texts)

    def encode_queries(self, texts):
        pre = self.query_instruction or ""
        return self._encode("q", [pre + t for t in texts])


# ---- lexical index ------------------------------------------------------------

class BM25Index:
    def __init__(self, texts, k1=1.5, b=0.75):
        from rank_bm25 import BM25Okapi
        self.n = len(texts)
        self.bm25 = BM25Okapi([tokenize(t) for t in texts], k1=k1, b=b) if self.n else None

    def scores(self, query):
        if self.bm25 is None:
            return np.zeros(0)
        return np.asarray(self.bm25.get_scores(tokenize(query)), float)


# ---- fusion ---------------------------------------------------------------------

def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / max(float(x.std()), 1e-9) if x.size else x


def _ranks(x):
    """0-based descending ordinal ranks; ties keep index order (stable sort)."""
    order = np.argsort(-np.asarray(x, float), kind="stable")
    r = np.empty(len(order), np.int64)
    r[order] = np.arange(len(order))
    return r


def rrf(rank_arrays, k=60):
    """Reciprocal-rank fusion sum_i 1/(k + rank_i). Ranks as given (0-based here,
    as in experiments/locomo_reality_check.py; Cormack et al. use 1-based)."""
    return sum(1.0 / (k + np.asarray(r, float)) for r in rank_arrays)


def fuse(bm25_scores, dense_scores, alpha, method="zscore"):
    """zscore: alpha*z(bm25) + (1-alpha)*z(dense). rrf: 1/(k+rank_bm25) + 1/(k+rank_dense)
    (unweighted: alpha is ignored)."""
    if method == "zscore":
        return alpha * zscore(bm25_scores) + (1 - alpha) * zscore(dense_scores)
    if method == "rrf":
        return rrf([_ranks(bm25_scores), _ranks(dense_scores)])
    raise ValueError(f"unknown fusion method {method!r}")


def _metric(metric):
    """metric(positions of the evidence in the fused order) -> float."""
    if metric == "mrr":
        return lambda p: 1.0 / (1.0 + p.min())
    name, _, k = metric.partition("@")
    if name in ("recall", "hit") and k.isdigit():
        k = int(k)
        return (lambda p: float((p < k).mean())) if name == "recall" else (lambda p: float(p.min() < k))
    raise ValueError(f"unknown metric {metric!r} (use 'mrr', 'recall@k' or 'hit@k')")


def fit_fusion_alpha(bm25_list, dense_list, evidence_list, groups, grid=None, metric="mrr"):
    """Leave-one-group-out alpha for z-score fusion. Per query i, bm25_list[i] and
    dense_list[i] score the same candidates and evidence_list[i] holds LOCAL positions
    of the gold evidence in them. alpha_by_group[g] is the grid argmax of the
    query-weighted mean metric over queries of all OTHER groups (first max on ties;
    0.5 when there is no other group), so no group ever selects its own alpha.
    Also returns score_by_alpha and alpha_all over all groups (in-sample: report only)."""
    grid = np.linspace(0, 1, 11) if grid is None else np.asarray(grid, float)
    f = _metric(metric)
    S = np.zeros((len(bm25_list), len(grid)))          # (queries, grid)
    for i, (b, d, ev) in enumerate(zip(bm25_list, dense_list, evidence_list)):
        ev = np.asarray(ev, np.int64).ravel()
        if ev.size == 0:
            continue                                      # contributes 0 to every alpha
        F = grid[:, None] * zscore(b)[None] + (1 - grid[:, None]) * zscore(d)[None]
        pos = np.argsort(np.argsort(-F, axis=1, kind="stable"), axis=1, kind="stable")
        S[i] = [f(p[ev]) for p in pos]
    g = np.asarray(groups).tolist()
    out = {}
    for h in dict.fromkeys(g):
        other = np.array([x != h for x in g])
        out[h] = float(grid[int(np.argmax(S[other].mean(0)))]) if other.any() else 0.5
    score = S.mean(0) if len(S) else np.zeros(len(grid))
    return dict(alpha_by_group=out, grid=grid.tolist(), score_by_alpha=score.tolist(),
                alpha_all=float(grid[int(np.argmax(score))]))


# ---- contiguity -------------------------------------------------------------------

def expand_contiguity(bank, rows, n=1, same_session=True, limit=None):
    """Anchors in their given order, each immediately followed by its not-yet-included
    neighbours (in bank.neighbors order); deduplicated; truncated to `limit`."""
    out, seen = [], set()
    for r in np.asarray(rows, np.int64).ravel().tolist():
        for x in [r] + ([int(v) for v in bank.neighbors(r, n, same_session)] if n > 0 else []):
            if x not in seen:
                seen.add(x); out.append(x)
        if limit is not None and len(out) >= limit:
            break
    return np.asarray(out if limit is None else out[:limit], np.int64)


# ---- the generator ------------------------------------------------------------------

class CandidateGenerator:
    """BM25 + dense fusion (+ contiguity) over ONE conversation's memories."""

    def __init__(self, bank, encoder, alpha=0.5, method="zscore", K=50, contiguity=0):
        self.bank, self.encoder = bank, encoder
        self.alpha, self.method, self.K, self.contiguity = alpha, method, K, contiguity
        self.same_session = True                      # design: +-n turns, same session
        self.doc_emb, self._conv = None, {}

    def index(self):
        """Encode every memory once; one BM25Index per conversation (built lazily
        for conversations the bank does not enumerate through memories[*].conv)."""
        self.doc_emb = self.encoder.encode_docs(self.bank.texts)
        self._conv = {}
        for c in dict.fromkeys(getattr(m, "conv", None) for m in getattr(self.bank, "memories", [])):
            if c is not None:
                self._conv_index(c)
        return self

    def _conv_index(self, conv):
        if conv not in self._conv:
            rows = np.asarray(self.bank.rows_of_conv(conv), np.int64)
            self._conv[conv] = (rows, {int(r): j for j, r in enumerate(rows)},
                                BM25Index([self.bank.texts[r] for r in rows]),
                                self.doc_emb[torch.as_tensor(rows)])
        return self._conv[conv]

    def scores(self, query_text, conv, q_emb=None):
        """Scores over all rows of `conv`: dict(rows, bm25, dense, fused)."""
        if self.doc_emb is None:
            self.index()
        rows, _, bm, E = self._conv_index(conv)
        q = self.encoder.encode_queries([query_text])[0] if q_emb is None else \
            torch.as_tensor(q_emb).detach().float().cpu().reshape(-1)
        q = q / q.norm().clamp_min(1e-9)
        bm25 = bm.scores(query_text)
        dense = (E @ q).numpy().astype(np.float64)
        return dict(rows=rows, bm25=bm25, dense=dense, fused=fuse(bm25, dense, self.alpha, self.method))

    def shortlist(self, query_text, conv, K=None, alpha=None, q_emb=None):
        """Top-K anchors by fused score, contiguity-expanded, truncated to K.
        rank = position of each row in the conversation's fused order, so
        rank >= K marks a row that entered only through contiguity."""
        K = self.K if K is None else int(K)
        s = self.scores(query_text, conv, q_emb=q_emb)
        fused = fuse(s["bm25"], s["dense"], self.alpha if alpha is None else alpha, self.method)
        rank = _ranks(fused)
        sel = s["rows"][np.argsort(rank)[:K]]
        if self.contiguity > 0:
            sel = expand_contiguity(self.bank, sel, n=self.contiguity, same_session=self.same_session)
            sel = sel[np.isin(sel, s["rows"])][:K]    # never leave the conversation
        pos = self._conv[conv][1]
        loc = np.array([pos[int(r)] for r in sel], np.int64)
        return dict(rows=np.asarray(sel, np.int64), fused=fused[loc], bm25_z=zscore(s["bm25"])[loc],
                    dense=s["dense"][loc], rank=rank[loc])
