"""
v2 PIPELINE — STORE -> CANDIDATES -> LOCATE (-> DECIDE).  See docs/DESIGN_v2.md.

Glue between the layers, kept separate from them so each layer stays testable alone:

  build_batch      per-query shortlist (L1) as fixed-width arrays + the per-memory
                   feature vector f_m that Locate adds to its logits
  state_vectors    QUERY-TIME state only: s = persona of the asking speaker (mean of
                   their own turns), h = the conversation's last session. Neither may
                   look at the evidence (v1 leaked the label through h).
  teacher_scores   System-One / cross-encoder relevance over the shortlist, cached
                   (the distillation teacher and the DECIDE-stage arm)
  train_locate     the v2 objective: multi-positive InfoNCE over shortlist U {null}
                   + KD from the teacher + Brier on P(null) for unanswerable questions
  rank_metrics     any-evidence hit@k, all-evidence recall@k and coverage@k, MRR

Features f_m, in order (FEATURES):
  bm25_z        within-conversation z-scored BM25
  dense         cosine of query and memory
  fused         the L1 fusion score (z-scored inside the shortlist)
  contig        1 if the memory entered only through contiguity expansion
  speaker_match 1 if the memory was spoken by the asking speaker (a state feature:
                LoCoMo-Conv users say "I", so who is asking decides what "I" refers to)
  name_match    1 if the memory's speaker is NAMED in the query text — a premise-
                consistency cue: LoCoMo's adversarial questions attribute a real turn to
                the wrong person ("What is Ben learning?" when Ana said it), which
                confidence alone cannot flag (in-harness AUROC 0.46-0.48)
  rank_frac     L1 rank / conversation length
"""
from __future__ import annotations
import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import torch

from . import losses
from .candidates import fuse
from .locate import LocateScorer, evidence_density

FEATURES = ("bm25_z", "dense", "fused", "contig", "speaker_match", "name_match", "rank_frac")
STATE_FEATURES = ("speaker_match",)


@dataclass
class Batch:
    rows: np.ndarray            # (Q, K) int64 memory rows, -1 = padding
    mask: np.ndarray            # (Q, K) bool
    feats: np.ndarray           # (Q, K, F) float32
    pos: np.ndarray             # (Q, K) bool — gold evidence inside the shortlist
    unans: np.ndarray           # (Q,) bool — unanswerable (LoCoMo category 5)
    conv_rows: list             # per query: all memory rows of its conversation
    fused_full: list            # per query: L1 fused score over conv_rows (for ranking the tail)


def build_batch(gen, bank, queries, q_emb, K, speakers=None, alpha=None):
    """gen: candidates.CandidateGenerator (indexed). q_emb: (Q, D) query embeddings.
    speakers: the speaker each query is attributed to (defaults to query.speaker; pass a
    shuffled list for the shuffled-state control, or [""]*Q for no state)."""
    Q, F = len(queries), len(FEATURES)
    speakers = [q.speaker for q in queries] if speakers is None else speakers
    rows = np.full((Q, K), -1, np.int64)
    feats = np.zeros((Q, K, F), np.float32)
    pos = np.zeros((Q, K), bool)
    conv_rows, fused_full = [], []
    a = getattr(gen, "alpha", 0.5) if alpha is None else alpha
    for i, q in enumerate(queries):
        sl = gen.shortlist(q.text, q.conv, K=K, alpha=a, q_emb=q_emb[i])
        full = gen.scores(q.text, q.conv, q_emb=q_emb[i])
        full["fused"] = fuse(full["bm25"], full["dense"], a, method=getattr(gen, "method", "zscore"))
        n = len(sl["rows"])
        rows[i, :n] = sl["rows"]
        fz = np.asarray(sl["fused"], np.float32)
        fz = (fz - fz.mean()) / (fz.std() + 1e-9)
        feats[i, :n, 0] = sl["bm25_z"]
        feats[i, :n, 1] = sl["dense"]
        feats[i, :n, 2] = fz
        feats[i, :n, 3] = (np.asarray(sl["rank"]) >= K).astype(np.float32)
        if speakers[i]:
            feats[i, :n, 4] = np.array([bank.memories[r].speaker == speakers[i] for r in sl["rows"]], np.float32)
        feats[i, :n, 5] = [_named(bank.memories[r].speaker, q.text) for r in sl["rows"]]
        feats[i, :n, 6] = np.asarray(sl["rank"], np.float32) / max(1, len(full["rows"]))
        ev = set(q.evidence)
        pos[i, :n] = [int(r) in ev for r in sl["rows"]]
        conv_rows.append(np.asarray(full["rows"]))
        fused_full.append(np.asarray(full["fused"], np.float32))
    unans = np.array([not q.answerable for q in queries])
    return Batch(rows, rows >= 0, feats, pos, unans, conv_rows, fused_full)


def _named(speaker, text):
    """1.0 if `speaker` appears as a whole word in `text` (case-insensitive)."""
    return float(bool(speaker) and re.search(r"\b" + re.escape(speaker) + r"\b", text, re.I) is not None)


def state_vectors(bank, doc_emb, queries, mode="speaker_persona", speakers=None):
    """(S, H) as float tensors (Q, D). Built only from query-time information."""
    D = doc_emb.shape[1]
    speakers = [q.speaker for q in queries] if speakers is None else speakers
    S = torch.zeros(len(queries), D)
    H = torch.zeros(len(queries), D)
    if mode == "none":
        return S, H
    for i, q in enumerate(queries):
        if mode == "speaker_persona" and speakers[i]:
            r = bank.speaker_rows(q.conv, speakers[i])
            if len(r):
                S[i] = torch.nn.functional.normalize(doc_emb[torch.as_tensor(r)].mean(0), dim=0)
        last = bank.last_session_rows(q.conv)
        if len(last):
            H[i] = torch.nn.functional.normalize(doc_emb[torch.as_tensor(last)].mean(0), dim=0)
    return S, H


class PairCache:
    """(decider, query text, memory row) -> relevance, persisted as JSON. Shortlists move
    with the per-fold fusion alpha, so caching PAIRS (not batches) avoids re-scoring the
    same query-memory pair in every fold and arm."""

    def __init__(self, path=None):
        self.path, self.d = path, {}
        if path and os.path.exists(path):
            self.d = json.load(open(path))

    @staticmethod
    def key(name, text, row):
        return hashlib.md5(f"{name}\x1f{text}\x1f{int(row)}".encode()).hexdigest()

    def save(self):
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            json.dump(self.d, open(self.path, "w"))


def teacher_scores(decider, bank, queries, batch, cache=None, query_texts=None):
    """(Q, K) raw relevance from a DECIDE arm over each shortlist; -1e4 on padding.
    Only pairs missing from `cache` (a PairCache) reach the decider."""
    texts = query_texts or [q.text for q in queries]
    cache = cache if cache is not None else PairCache()
    out = np.full(batch.rows.shape, -1e4, np.float32)
    for i in range(len(queries)):
        rows = batch.rows[i, batch.mask[i]]
        keys = [PairCache.key(decider.name, texts[i], r) for r in rows]
        todo = [j for j, k in enumerate(keys) if k not in cache.d]
        if todo:
            sc = decider.relevance(texts[i], [bank.texts[rows[j]] for j in todo])
            for j, v in zip(todo, np.asarray(sc, float)):
                cache.d[keys[j]] = float(v)
        out[i, :len(rows)] = [cache.d[k] for k in keys]
    return out


def _gather(batch, idx, doc_emb, q_emb, S, H, teacher=None):
    rows = torch.as_tensor(batch.rows[idx]).clamp_min(0)
    t = None if teacher is None else torch.as_tensor(teacher[idx])
    return (q_emb[idx], doc_emb[rows], torch.as_tensor(batch.feats[idx]), torch.as_tensor(batch.mask[idx]),
            S[idx], H[idx], torch.as_tensor(batch.pos[idx]), torch.as_tensor(batch.unans[idx]), t)


def train_locate(batch, doc_emb, q_emb, S, H, weights, teacher=None, steps=300, lr=2e-3,
                 bsz=64, seed=0, rank=32, tau_init=16.0, null_hidden=32, kd_T=1.0, use_feats=None):
    """Fit LocateScorer with the v2 objective. use_feats: optional boolean mask over
    FEATURES (e.g. drop STATE_FEATURES for the static-adapter control)."""
    torch.manual_seed(seed)
    D, F = doc_emb.shape[1], len(FEATURES)
    scorer = LocateScorer(D, F, rank=rank, tau_init=tau_init, null_hidden=null_hidden)
    opt = torch.optim.Adam(scorer.parameters(), lr=lr)
    fmask = torch.ones(F) if use_feats is None else torch.as_tensor(np.asarray(use_feats, np.float32))
    usable = np.where(batch.pos.any(1) | batch.unans)[0]
    g = np.random.default_rng(seed)
    for _ in range(steps):
        idx = g.choice(usable, min(bsz, len(usable)), replace=False)
        q, mem, f, m, s, h, pos, un, t = _gather(batch, idx, doc_emb, q_emb, S, H, teacher)
        logits = scorer(q, mem, f * fmask, m, s=s, h=h)
        parts = {"ret": losses.multi_positive_infonce(logits, pos, null_positive=un)}
        if t is not None and getattr(weights, "kd", 0) > 0:
            parts["kd"] = losses.distill_kl(logits[:, :-1], t, m, T=kd_T)
        if getattr(weights, "null", 0) > 0:
            parts["null"] = losses.null_brier(evidence_density(logits)[:, -1], un.float())
        loss = losses.composite_loss(parts, weights)
        opt.zero_grad(); loss.backward(); opt.step()
    scorer.eval()
    scorer.feature_mask = fmask
    return scorer


@torch.no_grad()
def locate_logits(scorer, batch, doc_emb, q_emb, S, H, chunk=256):
    fmask = getattr(scorer, "feature_mask", torch.ones(len(FEATURES)))
    out = []
    for a in range(0, len(batch.rows), chunk):
        idx = np.arange(a, min(a + chunk, len(batch.rows)))
        q, mem, f, m, s, h, *_ = _gather(batch, idx, doc_emb, q_emb, S, H)
        out.append(scorer(q, mem, f * fmask, m, s=s, h=h).numpy())
    return np.concatenate(out)


def full_ranking(batch, i, shortlist_scores):
    """Scores over the whole conversation: the shortlist ordered by `shortlist_scores`
    (K,), placed above the tail, which keeps the L1 fused order."""
    cands = batch.conv_rows[i]
    sc = batch.fused_full[i].astype(np.float64) - 1e6
    pos = {int(r): j for j, r in enumerate(cands)}
    for j in np.where(batch.mask[i])[0]:
        sc[pos[int(batch.rows[i, j])]] = 1e6 + float(shortlist_scores[j])
    return cands, sc


def rank_metrics(scores, cands, evidence, ks=(1, 3, 5, 10, 20, 30, 50)):
    """any-evidence hit@k, all-evidence recall@k (fraction of gold found), coverage@k
    (ALL gold in top-k), MRR (first gold)."""
    order = cands[np.argsort(-np.asarray(scores), kind="stable")]
    pos = {int(r): p for p, r in enumerate(order)}
    ranks = np.array([pos[int(e)] for e in evidence if int(e) in pos])
    out = {"mrr": float(1.0 / (ranks.min() + 1)) if ranks.size else 0.0}
    for k in ks:
        hit = ranks.size > 0 and ranks.min() < k
        out[f"hit@{k}"] = float(hit)
        out[f"recall@{k}"] = float((ranks < k).sum() / max(1, len(evidence))) if ranks.size else 0.0
        out[f"cover@{k}"] = float(ranks.size == len(evidence) and (ranks < k).all())
    return out
