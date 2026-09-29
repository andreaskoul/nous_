"""
LAYER 0 — STORE: verbatim turns + metadata, the LoCoMo / LoCoMo-Conv loaders, and
the QUERY-TIME state a query may be conditioned on.

Memories are stored VERBATIM, one per dialogue turn. No LLM fact extraction or
summarisation sits between the conversation and retrieval: verbatim chunks beat
LLM-extracted facts by +15.9 pts on LoCoMo (arXiv 2601.00821), and an extractor
is one more un-calibrated model whose misses nothing downstream can recover.
The speaker name is part of the memory text ("<speaker>: <text>"): LoCoMo
questions are about named people, and a turn stripped of its speaker cannot
answer them. Shared images keep their BLIP caption (" [shares image: ...]").

Data (snap-research LoCoMo, arXiv 2402.17753, CC BY-NC 4.0; data/ is gitignored):
  locomo10.json      [ {conversation: {speaker_a, speaker_b, session_k: [turn],
                        session_k_date_time}, qa: [{question, evidence, category}]} ]
                     10 conversations, 5882 turns, 1986 QA (1981 with resolvable
                     evidence; 3 evidence ids name no turn, a few strings are
                     malformed, e.g. "D:11:26"). Evidence strings may pack several
                     ids ("D8:6; D9:17"), so ids are pulled out with DIA_RE.
  locomo10_dialog.json   LoCoMo-Conv: the SAME conversations; each QA item gains
                     first-person rewrites asked BY a speaker of the conversation:
                     dialog_query ("Do you remember when I ..."; its error flag is
                     `rewrite_error`), implicit_query (the need is implied, not
                     asked), counterfactual_query (asserts a wrong fact to correct;
                     absent for category 5).
  locomo10_multimem_full.json  composed queries whose answer needs several
                     original QA facts: composed_query, gold_dia_ids, member_q_idxs.

Query-time state (state_rows): v1 built its history h from the turns preceding
the EARLIEST EVIDENCE turn, i.e. from the label; the gain it measured was leakage.
v2 state is a function of (bank, conversation, asking speaker) only. LoCoMo
questions are asked after the conversation ends, so the honest history is the
last session. The shuffled-state control (shuffled_speakers) keeps a real persona
from the same conversation but the WRONG person: a gain from speaker_persona that
survives shuffling is not about who is asking.

  Memory / Query / MemoryBank     the store and its lookups (rows are global ids)
  load_locomo / load_locomo_conv  -> (MemoryBank, [Query]); qid == list position
  conversation_folds              leave-conversations-out CV folds
  state_rows / shuffled_speakers  query-time state and its shuffled control
  render_query                    "<speaker>: <text>", the non-learned metadata baseline
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

DIA_RE = re.compile(r"D\d+:\d+")
_SESSION_RE = re.compile(r"^session_(\d+)$")
STYLES = ("question", "dialog", "implicit", "counterfactual", "composed")
STATE_MODES = ("none", "speaker_persona", "last_session")
# LoCoMo-Conv rewrite fields: style -> (query, error flags, speaker name field)
_REWRITES = {
    "dialog": ("dialog_query", ("rewrite_error", "dialog_error"), "subject_speaker_name"),
    "implicit": ("implicit_query", ("implicit_error",), "implicit_subject_speaker_name"),
    "counterfactual": ("counterfactual_query", ("counterfactual_error",),
                       "counterfactual_subject_speaker_name"),
}


@dataclass(frozen=True)
class Memory:
    row: int            # global row in the bank (== index in MemoryBank.memories)
    conv: int
    session: int
    order: int          # turn index within the conversation (0-based, across sessions)
    dia_id: str
    speaker: str
    date: str           # the session's date_time string
    text: str           # "<speaker>: <text>[ [shares image: <caption>]]"


@dataclass(frozen=True)
class Query:
    qid: int
    conv: int
    text: str
    evidence: tuple     # sorted unique global memory rows
    category: int       # LoCoMo category; -1 for composed
    style: str          # one of STYLES
    speaker: str        # asking speaker; "" for original LoCoMo questions
    answerable: bool    # False iff LoCoMo category 5 (adversarial)
    source_qids: tuple = ()   # indices into the conversation's original qa list


class MemoryBank:
    def __init__(self, memories: list[Memory]):
        self.memories = list(memories)
        if any(m.row != i for i, m in enumerate(self.memories)):
            raise ValueError("Memory.row must equal its position in the bank")
        self.texts = [m.text for m in self.memories]
        self.speakers = [m.speaker for m in self.memories]
        self.conv = np.array([m.conv for m in self.memories], np.int64)
        self.session = np.array([m.session for m in self.memories], np.int64)
        self.order = np.array([m.order for m in self.memories], np.int64)
        self._spk = np.array(self.speakers, dtype=object)
        self.unresolved_evidence = 0       # set by the loaders

    def __len__(self):
        return len(self.memories)

    def rows_of_conv(self, conv) -> np.ndarray:
        return np.flatnonzero(self.conv == conv)

    def neighbors(self, row, n=1, same_session=True) -> np.ndarray:
        """Rows within +-n turns of `row` in its conversation (and session), excluding
        `row`, nearest first; ties (one before, one after) go to the earlier turn."""
        d = self.order - self.order[row]
        m = (self.conv == self.conv[row]) & (np.abs(d) <= n)
        if same_session:
            m &= self.session == self.session[row]
        m[row] = False
        r = np.flatnonzero(m)
        return r[np.lexsort((self.order[r], np.abs(d[r])))]

    def speaker_rows(self, conv, speaker) -> np.ndarray:
        return np.flatnonzero((self.conv == conv) & (self._spk == speaker))

    def last_session_rows(self, conv) -> np.ndarray:
        r = self.rows_of_conv(conv)
        return r[self.session[r] == self.session[r].max()] if len(r) else r


# ---- LoCoMo parsing -------------------------------------------------------------

def _read(path):
    with open(path) as f:
        return json.load(f)


def _bank(samples):
    """-> (MemoryBank of every turn in file order, per-conversation {dia_id: row})."""
    mems, local = [], []
    for ci, s in enumerate(samples):
        conv, loc, order = s.get("conversation", s), {}, 0
        for sk in sorted((k for k in conv if _SESSION_RE.match(k)),     # numeric: 2 < 10
                         key=lambda k: int(_SESSION_RE.match(k).group(1))):
            sidx, date = int(_SESSION_RE.match(sk).group(1)), conv.get(f"{sk}_date_time", "")
            for t in conv[sk]:
                dia, sp, text = t.get("dia_id") or t.get("id") or "", t.get("speaker", ""), t.get("text") or ""
                if t.get("blip_caption"):
                    text = f"{text} [shares image: {t['blip_caption']}]"
                if dia:
                    loc[dia] = len(mems)
                mems.append(Memory(len(mems), ci, sidx, order, dia, sp, date,
                                   f"{sp}: {text}" if sp else text))
                order += 1
        local.append(loc)
    return MemoryBank(mems), local


def _resolve(ev, loc):
    """Evidence field -> (sorted unique rows, #ids naming no turn of this conversation)."""
    ev = [ev] if isinstance(ev, str) else (ev or [])
    ids = [d for e in ev for d in DIA_RE.findall(str(e))]
    return tuple(sorted({loc[d] for d in ids if d in loc})), sum(d not in loc for d in ids)


def _originals(samples, local):
    """-> ([(conv, qa_index, qa, evidence_rows)] with resolvable evidence, #unresolved)."""
    out, unres = [], 0
    for ci, s in enumerate(samples):
        for qi, qa in enumerate(s.get("qa", [])):
            ev, u = _resolve(qa.get("evidence"), local[ci])
            unres += u
            if ev:
                out.append((ci, qi, qa, ev))
    return out, unres


def load_locomo(path):
    """Original LoCoMo questions (style "question", speaker ""). QA whose evidence
    resolves to no turn are skipped; the count of unresolvable evidence ids is kept
    as bank.unresolved_evidence. qid == position in the returned list."""
    return load_locomo_conv(path, styles=("question",))


def load_locomo_conv(dialog_path, multimem_path=None,
                     styles=("dialog", "implicit", "counterfactual", "composed")):
    """LoCoMo-Conv queries, style-major in the order of `styles` (then conversation,
    then QA order). The bank is built from the conversations in the dialog file, so it
    is identical to load_locomo on the same data. Rewrites inherit evidence, category
    and answerability from their original QA and are skipped when the rewrite is
    missing/empty or its error flag is set; "composed" needs multimem_path (silently
    absent otherwise). source_qids indexes the conversation's original qa list: the
    QA itself for question/rewrites, member_q_idxs for composed."""
    bad = set(styles) - set(STYLES)
    if bad:
        raise ValueError(f"unknown styles {sorted(bad)}; expected a subset of {STYLES}")
    samples = _read(dialog_path)
    bank, local = _bank(samples)
    orig, bank.unresolved_evidence = _originals(samples, local)
    bank.unresolved_composed = 0
    queries = []

    def add(conv, text, ev, cat, style, speaker, src):
        queries.append(Query(len(queries), conv, text, ev, int(cat), style, speaker,
                             int(cat) != 5, tuple(int(i) for i in src)))

    for style in styles:
        if style == "question":
            for ci, qi, qa, ev in orig:
                if qa.get("question"):
                    add(ci, qa["question"], ev, qa.get("category", -1), style, "", (qi,))
        elif style == "composed":
            for it in _read(multimem_path) if multimem_path else []:
                ci = int(it["sample_idx"])
                if not 0 <= ci < len(local):
                    raise ValueError(f"multimem sample_idx {ci} not in {dialog_path}")
                ev, u = _resolve(it.get("gold_dia_ids"), local[ci])
                bank.unresolved_composed += u
                if ev and it.get("composed_query") and not it.get("rewrite_error"):
                    add(ci, it["composed_query"], ev, -1, style,
                        it.get("subject_speaker_name") or "", it.get("member_q_idxs") or ())
        else:
            field, errs, spk = _REWRITES[style]
            for ci, qi, qa, ev in orig:
                if qa.get(field) and not any(qa.get(e) for e in errs):
                    add(ci, qa[field], ev, qa.get("category", -1), style,
                        qa.get(spk) or qa.get("subject_speaker_name") or "", (qi,))
    return bank, queries


# ---- folds, query-time state, controls ---------------------------------------------

def conversation_folds(n_conv, k=5):
    """Leave-conversations-out folds. n_conv == 2k -> contiguous pairs [[0,1],[2,3],..]
    (the reality-check folds); otherwise round-robin [i, i+k, ...]. Empty folds
    (n_conv < k) are dropped: a fold with no test conversation scores nothing."""
    if n_conv == 2 * k:
        return [[2 * i, 2 * i + 1] for i in range(k)]
    return [f for f in (list(range(i, n_conv, k)) for i in range(k)) if f]


def state_rows(bank, query, mode) -> np.ndarray:
    """Memory rows forming the QUERY-TIME state of `query`.
      none             nothing
      speaker_persona  the asking speaker's own turns in query.conv (empty if speaker "")
      last_session     the conversation's last session (LoCoMo questions come after it)
    WARNING: must never read query.evidence (or anything derived from it). v1's
    history was 'turns before the earliest evidence turn', which leaked the label."""
    if mode == "none":
        return np.empty(0, np.int64)
    if mode == "speaker_persona":
        return bank.speaker_rows(query.conv, query.speaker) if query.speaker else np.empty(0, np.int64)
    if mode == "last_session":
        return bank.last_session_rows(query.conv)
    raise ValueError(f"unknown state mode {mode!r}; expected one of {STATE_MODES}")


def _derangement(n, rng):
    while True:                           # rejection: P(accept) -> 1/e, n >= 2
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def shuffled_speakers(queries, seed=0, bank=None) -> list[str]:
    """Shuffled-state control: per conversation, a random derangement of its speakers
    applied to every query, so each query is attributed to a DIFFERENT real speaker of
    the same conversation (with 2 speakers: the other one). The speaker pool is the
    queries' speakers in that conversation (plus the bank's, when given); a pool of < 2
    keeps the speaker, and "" stays "" (no state to shuffle). Deterministic in
    (seed, conv), independent of which other conversations are present."""
    out = [q.speaker for q in queries]
    by_conv = defaultdict(list)
    for i, q in enumerate(queries):
        by_conv[q.conv].append(i)
    for c, idx in by_conv.items():
        pool = {queries[i].speaker for i in idx}
        if bank is not None:
            pool |= {bank.speakers[r] for r in bank.rows_of_conv(c)}
        pool = sorted(pool - {""})
        if len(pool) < 2:
            continue
        perm = _derangement(len(pool), np.random.default_rng([seed, int(c)]))
        new = {s: pool[j] for s, j in zip(pool, perm)}
        for i in idx:
            out[i] = new.get(out[i], out[i])
    return out


def render_query(query, speaker_prefix=False) -> str:
    """The non-learned metadata baseline: prefix the asking speaker's name, so lexical
    and dense retrieval can resolve "I" / "my" without any learned state."""
    return f"{query.speaker}: {query.text}" if speaker_prefix and query.speaker else query.text
