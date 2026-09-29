"""
Unit checks for L0 STORE (manifold_mvp/store.py): verbatim memory rendering, LoCoMo
and LoCoMo-Conv parsing on small synthetic fixtures (packed evidence, category 5,
unresolved ids, *_error flags, speaker fallbacks, composed queries), bank lookups,
folds, query-time state (never a function of the evidence), the shuffled-speaker
control, and a smoke test on the real files when they are present locally.
Offline, deterministic, ~3 s. Run with:  python tests/test_store.py   (or pytest).
"""
from __future__ import annotations
import dataclasses, json, os, sys, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from manifold_mvp import store
from manifold_mvp.store import (DIA_RE, Memory, MemoryBank, Query, conversation_folds,
                                load_locomo, load_locomo_conv, render_query,
                                shuffled_speakers, state_rows)


# ---- fixtures ---------------------------------------------------------------------

def _t(sp, dia, text, cap=None):
    return dict(speaker=sp, dia_id=dia, text=text, **({"blip_caption": cap} if cap else {}))


def _samples():
    """2 conversations x 2 sessions. conv 0 inserts session_2 BEFORE session_1 (sessions
    must be ordered by number, not by key order); both reuse dia_ids like "D1:1"."""
    c0 = {"speaker_a": "Alice", "speaker_b": "Bob",
          "session_2_date_time": "9:00 am on 2 June, 2023",
          "session_2": [_t("Bob", "D2:1", "I started pottery."),          # row 3
                        _t("Alice", "D2:2", "Nice! I ran a marathon."),     # row 4
                        _t("Bob", "D2:3", "Congrats!")],                    # row 5
          "session_1_date_time": "1:00 pm on 1 May, 2023",
          "session_1": [_t("Alice", "D1:1", "Hi Bob!"),                     # row 0
                        _t("Bob", "D1:2", "Look at this.", "a dog on a beach"),  # row 1
                        _t("Alice", "D1:3", "I adopted a cat named Luna.")]}     # row 2
    qa0 = [dict(question="What pet did Alice adopt?", answer="a cat", evidence=["D1:3"], category=4),
           dict(question="What did they do?", answer="x", evidence=["D1:1; D2:1"], category=1),
           dict(question="What did Bob realize?", adversarial_answer="y", evidence=["D2:2"], category=5),
           dict(question="Unresolvable?", answer="z", evidence=["D9:9"], category=4),
           dict(question="Dupes?", answer="w", evidence=["D2:3", "D2:3", "D1:2"], category=2),
           dict(question="Partial?", answer="v", evidence=["D1:2", "D7:1"], category=3)]
    c1 = {"speaker_a": "Carol", "speaker_b": "Dan",
          "session_1_date_time": "8:00 am on 3 March, 2023",
          "session_1": [_t("Carol", "D1:1", "Morning Dan."),                # row 6
                        _t("Dan", "D1:2", "I moved to Oslo.")],             # row 7
          "session_2_date_time": "8:00 pm on 4 April, 2023",
          "session_2": [_t("Carol", "D2:1", "How is Oslo?"),                # row 8
                        _t("Dan", "D2:2", "Cold, but I love it.")]}         # row 9
    qa1 = [dict(question="Where did Dan move?", answer="Oslo", evidence=["D1:2"], category=4),
           dict(question="Greeting?", answer="Morning", evidence=["D1:1"], category=4)]
    return [dict(sample_id="s0", conversation=c0, qa=qa0), dict(sample_id="s1", conversation=c1, qa=qa1)]


def _dialog_samples():
    s = _samples()
    r = lambda d, i, c, sp, **kw: dict(dialog_query=d, subject_speaker_name=sp, rewrite_error=None,
                                       implicit_query=i, implicit_subject_speaker_name=sp, implicit_error=None,
                                       counterfactual_query=c, counterfactual_subject_speaker_name=sp,
                                       counterfactual_error=None) | kw
    extra0 = [r("Remember my cat?", "I need to buy pet food for her.", "My dog Luna is fine, right?", "Alice"),
              r("What did we do?", "Planning a recap.", "We did nothing, right?", "Bob",
                implicit_subject_speaker_name="Alice", rewrite_error="bad rewrite"),
              r("What did I realize?", "Thinking about lessons.", None, "Bob"),     # cat 5: no cf
              r("Unresolvable dq", "Unresolvable iq", "Unresolvable cq", "Alice"),  # skipped (no evidence)
              r("", "Clay is fun.", "I never did pottery.", "Bob", implicit_error="flagged"),
              {k: v for k, v in r("Partial dq", "Partial iq", "Partial cq", "Bob",
                                  implicit_subject_speaker_name="").items()
               if k != "counterfactual_subject_speaker_name"}]
    extra1 = [r("Where did I move?", "Packing winter clothes.", "I moved to Rome, right?", "Dan"),
              r("How did I greet you?", "Morning routine.", "I said good night?", "Carol")]
    for sm, extra in zip(s, (extra0, extra1)):
        for qa, e in zip(sm["qa"], extra):
            qa.update(e)
    return s


def _multimem():
    return [dict(sample_idx=0, member_q_idxs=[0, 4], gold_dia_ids=["D1:3", "D2:3; D1:2"],
                 composed_query="Tell me about my cat and your hobbies.", subject_speaker_name="Alice",
                 rewrite_error=None),
            dict(sample_idx=1, member_q_idxs=[0, 1], gold_dia_ids=["D1:2", "D1:1", "D5:5"],
                 composed_query="Where did I move and how did you greet me?", subject_speaker_name="Dan",
                 rewrite_error=None),
            dict(sample_idx=1, member_q_idxs=[0], gold_dia_ids=["D1:2"], composed_query="err",
                 subject_speaker_name="Dan", rewrite_error="failed"),
            dict(sample_idx=0, member_q_idxs=[3], gold_dia_ids=["D9:9"], composed_query="nothing",
                 subject_speaker_name="Bob", rewrite_error=None)]


def _write(d, name, obj):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        json.dump(obj, f)
    return p


def _files(d):
    return (_write(d, "locomo.json", _samples()), _write(d, "dialog.json", _dialog_samples()),
            _write(d, "multimem.json", _multimem()))


# ---- tests ------------------------------------------------------------------------

def test_memory_rendering():
    """Verbatim turns, numeric session order, per-conversation order, dates, captions."""
    with tempfile.TemporaryDirectory() as d:
        bank, _ = load_locomo(_files(d)[0])
    assert len(bank) == 10 and bank.conv.tolist() == [0] * 6 + [1] * 4
    assert [m.dia_id for m in bank.memories[:6]] == ["D1:1", "D1:2", "D1:3", "D2:1", "D2:2", "D2:3"]
    assert bank.session.tolist() == [1, 1, 1, 2, 2, 2, 1, 1, 2, 2]
    assert bank.order.tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    assert all(a.dtype == np.int64 for a in (bank.conv, bank.session, bank.order))
    m = bank.memories[1]
    assert m == Memory(1, 0, 1, 1, "D1:2", "Bob", "1:00 pm on 1 May, 2023",
                       "Bob: Look at this. [shares image: a dog on a beach]")
    assert bank.texts[2] == "Alice: I adopted a cat named Luna." and bank.memories[3].date.startswith("9:00 am")
    assert bank.speakers == [m.speaker for m in bank.memories] and bank.texts[6] == "Carol: Morning Dan."
    try:
        MemoryBank([dataclasses.replace(m, row=5)])
        raise AssertionError("row != position must be rejected")
    except ValueError:
        pass
    print("  memory rendering OK (10 turns, session_2-before-session_1 reordered, caption, dates)")


def test_locomo_queries():
    """Packed evidence, per-conversation dia_id resolution, dedup, cat 5, unresolved."""
    assert DIA_RE.findall("D8:6; D9:17") == ["D8:6", "D9:17"]
    with tempfile.TemporaryDirectory() as d:
        bank, qs = load_locomo(_files(d)[0])
    assert [q.qid for q in qs] == list(range(len(qs))) == list(range(7))
    assert [q.evidence for q in qs] == [(2,), (0, 3), (4,), (1, 5), (1,), (7,), (6,)]
    assert [q.source_qids for q in qs] == [(0,), (1,), (2,), (4,), (5,), (0,), (1,)]
    assert [q.answerable for q in qs] == [True, True, False, True, True, True, True]
    assert all(q.style == "question" and q.speaker == "" for q in qs)
    assert qs[5].conv == 1 and qs[6].evidence == (6,)      # "D1:1" of conv 1, not conv 0's row 0
    assert bank.unresolved_evidence == 2                  # D9:9 (skipped QA) + D7:1 (partial QA)
    assert all(type(r) is int for q in qs for r in q.evidence) and hash(qs[0])
    print("  LoCoMo queries OK (7/8 kept, packed 'D1:1; D2:1' -> (0,3), cat5 unanswerable, 2 unresolved)")


def test_bank_lookups():
    with tempfile.TemporaryDirectory() as d:
        bank, _ = load_locomo(_files(d)[0])
    assert bank.rows_of_conv(1).tolist() == [6, 7, 8, 9] and bank.rows_of_conv(7).tolist() == []
    assert bank.neighbors(1).tolist() == [0, 2]
    assert bank.neighbors(2).tolist() == [1]                                  # session boundary
    assert bank.neighbors(2, same_session=False).tolist() == [1, 3]
    assert bank.neighbors(2, n=2, same_session=False).tolist() == [1, 3, 0, 4]
    assert bank.neighbors(3, n=5).tolist() == [4, 5]
    assert bank.neighbors(5, n=9, same_session=False).tolist() == [4, 3, 2, 1, 0]  # stays in conv 0
    assert bank.neighbors(6, n=3, same_session=False).tolist() == [7, 8, 9]
    assert bank.neighbors(0, n=0).tolist() == []
    assert bank.speaker_rows(0, "Alice").tolist() == [0, 2, 4] and bank.speaker_rows(1, "Alice").tolist() == []
    assert bank.last_session_rows(0).tolist() == [3, 4, 5] and bank.last_session_rows(1).tolist() == [8, 9]
    assert bank.last_session_rows(5).tolist() == []
    for r in (bank.neighbors(1), bank.speaker_rows(0, "Bob"), bank.rows_of_conv(0)):
        assert isinstance(r, np.ndarray) and r.dtype == np.int64
    print("  bank lookups OK (neighbors by |distance| then order, session-bounded; speaker; last session)")


def test_locomo_conv():
    """Same bank as LoCoMo; rewrites inherit labels; error/empty skips; speaker fallbacks."""
    with tempfile.TemporaryDirectory() as d:
        lp, dp, mp = _files(d)
        bank0, q0 = load_locomo(lp)
        bank, qs = load_locomo_conv(dp, mp)
        _, qall = load_locomo_conv(dp, mp, styles=store.STYLES)
        _, qnone = load_locomo_conv(dp)                     # no multimem -> no composed
        try:
            load_locomo_conv(dp, styles=("dialog", "bogus"))
            raise AssertionError("unknown style must raise")
        except ValueError:
            pass
    assert bank.memories == bank0.memories and bank.unresolved_evidence == bank0.unresolved_evidence == 2
    assert bank.unresolved_composed == 2                    # D5:5 (kept item) + D9:9 (skipped item)
    by = {s: [q for q in qs if q.style == s] for s in ("dialog", "implicit", "counterfactual", "composed")}
    assert [q.style for q in qs] == sorted([q.style for q in qs], key=["dialog", "implicit",
                                                                     "counterfactual", "composed"].index)
    assert [q.qid for q in qs] == list(range(len(qs)))
    # dialog: QA1 has rewrite_error, QA4 an empty dialog_query, QA3 no evidence
    assert [q.text for q in by["dialog"]] == ["Remember my cat?", "What did I realize?", "Partial dq",
                                              "Where did I move?", "How did I greet you?"]
    assert [q.source_qids for q in by["dialog"]] == [(0,), (2,), (5,), (0,), (1,)]
    assert by["dialog"][1].answerable is False and by["dialog"][1].category == 5
    # implicit: QA4 flagged; QA1's implicit speaker differs; QA5 falls back to subject_speaker_name
    assert [(q.text, q.speaker) for q in by["implicit"][:3]] == [
        ("I need to buy pet food for her.", "Alice"), ("Planning a recap.", "Alice"),
        ("Thinking about lessons.", "Bob")]
    assert by["implicit"][3].text == "Partial iq" and by["implicit"][3].speaker == "Bob"
    assert by["implicit"][1].evidence == (0, 3)              # inherited from the original QA
    # counterfactual: none for cat 5 (None); QA5 has no cf speaker field -> fallback
    assert [q.source_qids for q in by["counterfactual"]] == [(0,), (1,), (4,), (5,), (0,), (1,)]
    assert by["counterfactual"][3].speaker == "Bob" and all(q.answerable for q in by["counterfactual"])
    # composed: errors / unresolvable items skipped, packed gold ids resolved within sample_idx
    c = by["composed"]
    assert [(q.conv, q.evidence, q.speaker, q.category, q.answerable, q.source_qids) for q in c] == [
        (0, (1, 2, 5), "Alice", -1, True, (0, 4)), (1, (6, 7), "Dan", -1, True, (0, 1))]
    assert len({q.qid for q in qall}) == len(qall) == len(q0) + len(qs)
    assert [dataclasses.replace(q, qid=0) for q in qall[:len(q0)]] == [dataclasses.replace(q, qid=0) for q in q0]
    assert [q.style for q in qnone].count("composed") == 0 and len(qnone) == len(qs) - 2
    print("  LoCoMo-Conv OK (dialog 5, implicit 6, counterfactual 6, composed 2; bank == LoCoMo bank)")


def test_folds():
    assert conversation_folds(10) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert conversation_folds(6, k=3) == [[0, 1], [2, 3], [4, 5]]
    assert conversation_folds(7, k=3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert conversation_folds(3, k=5) == [[0], [1], [2]]                  # empty folds dropped
    for n, k in ((10, 5), (9, 5), (11, 4), (2, 2), (1, 5)):
        f = conversation_folds(n, k)
        assert sorted(c for x in f for c in x) == list(range(n)) and all(f)  # a partition
    print("  folds OK (10 -> contiguous pairs, else round-robin, partition, no empty fold)")


def test_state_rows():
    """Each mode, and invariance to the evidence (the v1 leak)."""
    with tempfile.TemporaryDirectory() as d:
        bank, qs = load_locomo_conv(_files(d)[1], styles=("question", "dialog"))
    q = next(x for x in qs if x.style == "dialog" and x.conv == 0 and x.speaker == "Alice")
    assert state_rows(bank, q, "none").tolist() == []
    assert state_rows(bank, q, "speaker_persona").tolist() == [0, 2, 4]
    assert state_rows(bank, q, "last_session").tolist() == [3, 4, 5]
    orig = next(x for x in qs if x.style == "question")
    assert state_rows(bank, orig, "speaker_persona").tolist() == []       # speaker ""
    assert state_rows(bank, orig, "last_session").tolist() == [3, 4, 5]
    for x in qs:
        for mode in store.STATE_MODES:
            ref = state_rows(bank, x, mode)
            for ev in ((), (9,), tuple(range(len(bank)))):
                assert np.array_equal(state_rows(bank, dataclasses.replace(x, evidence=ev), mode), ref)
            assert ref.dtype == np.int64
    try:
        state_rows(bank, q, "evidence_history")
        raise AssertionError("unknown mode must raise")
    except ValueError:
        pass
    print("  state rows OK (none / persona / last session; identical under any evidence)")


def test_shuffled_speakers():
    mk = lambda i, c, sp: Query(i, c, f"q{i}", (0,), 4, "dialog", sp, True)
    qs = [mk(0, 0, "Alice"), mk(1, 0, "Bob"), mk(2, 0, "Alice"), mk(3, 0, ""), mk(4, 1, "Dan"),
          mk(5, 1, "Carol"), mk(6, 2, "Eve"), mk(7, 3, "X"), mk(8, 3, "Y"), mk(9, 3, "Z"), mk(10, 3, "X")]
    s = shuffled_speakers(qs, seed=0)
    assert s[:6] == ["Bob", "Alice", "Bob", "", "Carol", "Dan"]          # 2 speakers: the other one
    assert s[6] == "Eve"                                                   # pool of 1: kept
    assert all(a != q.speaker for a, q in zip(s[7:], qs[7:])) and s[7] == s[10]  # derangement per speaker
    assert sorted(set(s[7:])) == ["X", "Y", "Z"]                           # a permutation of the pool
    assert s == shuffled_speakers(qs, seed=0) and s[:4] == shuffled_speakers(qs[:4], seed=0)
    seen = {tuple(shuffled_speakers(qs, seed=k)[7:10]) for k in range(20)}
    assert seen == {("Y", "Z", "X"), ("Z", "X", "Y")}                      # both 3-derangements occur
    with tempfile.TemporaryDirectory() as d:
        bank, _ = load_locomo(_files(d)[0])
    one_sided = [mk(0, 1, "Dan"), mk(1, 1, "Dan")]
    assert shuffled_speakers(one_sided) == ["Dan", "Dan"]                  # pool from queries only
    assert shuffled_speakers(one_sided, bank=bank) == ["Carol", "Carol"]   # pool from the bank
    print("  shuffled speakers OK (2-speaker swap, 3-speaker derangement, '' kept, seeded, bank pool)")


def test_render_query():
    q = Query(0, 0, "Where did I move?", (7,), 4, "dialog", "Dan", True)
    assert render_query(q) == "Where did I move?"
    assert render_query(q, speaker_prefix=True) == "Dan: Where did I move?"
    assert render_query(dataclasses.replace(q, speaker=""), speaker_prefix=True) == "Where did I move?"
    print("  render_query OK")


def test_real_data_smoke():
    """Only when the gitignored LoCoMo files are present locally."""
    lp, dp = os.path.join(ROOT, "data", "locomo10.json"), os.path.join(ROOT, "data", "locomo10_dialog.json")
    mp = os.path.join(ROOT, "data", "locomo10_multimem_full.json")
    if not (os.path.exists(lp) and os.path.exists(dp)):
        print("  real-data smoke SKIPPED (data/locomo10.json or data/locomo10_dialog.json absent)")
        return
    bank, qs = load_locomo(lp)
    assert len(np.unique(bank.conv)) == 10 and len(bank) == 5882 and len(qs) == 1981
    assert sum(not q.answerable for q in qs) == 446
    cbank, cq = load_locomo_conv(dp, mp if os.path.exists(mp) else None)
    assert cbank.memories == bank.memories
    n = {s: sum(q.style == s for q in cq) for s in store.STYLES}
    assert n["implicit"] > 1000 and len({q.qid for q in cq}) == len(cq)
    if os.path.exists(mp):
        assert n["composed"] > 1000
    names = set(cbank.speakers)
    assert all(q.speaker in names for q in cq)                             # askers are real speakers
    print(f"  real-data smoke OK (10 convs, 5882 memories, 1981 questions, {bank.unresolved_evidence} "
          f"unresolved ids; LoCoMo-Conv {n})")


if __name__ == "__main__":
    for fn in [test_memory_rendering, test_locomo_queries, test_bank_lookups, test_locomo_conv,
               test_folds, test_state_rows, test_shuffled_speakers, test_render_query,
               test_real_data_smoke]:
        print(f"[{fn.__name__}]")
        fn()
    print("\nALL STORE CHECKS PASSED")
