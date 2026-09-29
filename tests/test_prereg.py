"""
Unit checks for the pre-registration mechanism (manifold_mvp/prereg.py) and the
v2 plan (prereg/plan_v2.json). Offline, stdlib only; every lock is written in a
temp dir (never for the real plan). Run with:  python tests/test_prereg.py
"""
from __future__ import annotations
import contextlib, hashlib, io, json, os, subprocess, sys, tempfile
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from manifold_mvp.prereg import (PlanNotFrozen, canonical_hash, check, freeze,  # noqa: E402
                                 load_plan, log_deviation, main, require_confirmatory)

REAL_PLAN = os.path.join(ROOT, "prereg", "plan_v2.json")
GATE_IDS = {"X0", "G1a", "G1b", "G1c", "A1", "C1", "R1", "E1prime", "P1", "U1"}
FAMILY = {"X0": "measurement", "G1a": "confirmatory", "G1b": "confirmatory",
          "G1c": "confirmatory", "A1": "confirmatory", "C1": "confirmatory",
          "R1": "measurement", "E1prime": "exploratory", "P1": "exploratory",
          "U1": "exploratory"}
GATE_FIELDS = ("id", "family", "hypothesis", "metric", "arms", "pass", "kill")
TOY = {"version": "t", "status": "frozen",
       "gates": [{"id": "G", "pass": "x >= 0.03", "thresholds": {"delta": 0.03}}]}


def _write(path, obj, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def test_hash_stable():
    """Content hash: key order and whitespace do not matter, values and list order do."""
    a = {"b": {"y": "é", "x": [1, 2.5]}, "a": 1}
    b = {"a": 1, "b": {"x": [1, 2.5], "y": "é"}}
    assert canonical_hash(a) == canonical_hash(b)
    assert canonical_hash(a) == hashlib.sha256(
        '{"a":1,"b":{"x":[1,2.5],"y":"é"}}'.encode("utf-8")).hexdigest()
    assert canonical_hash(a) != canonical_hash({"a": 1, "b": {"x": [2.5, 1], "y": "é"}})
    assert canonical_hash(a) != canonical_hash({"a": 2, "b": b["b"]})
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "p1.json"), os.path.join(d, "p2.json")
        _write(p1, a, indent=2); _write(p2, b, separators=(",", ":"))
        assert canonical_hash(load_plan(p1)) == canonical_hash(load_plan(p2))


def test_freeze_check_roundtrip():
    """freeze writes <plan>.lock.json; check matches; reformatting keeps the match;
    re-freezing identical content keeps the original timestamp."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan_v2.json")
        _write(p, TOY, indent=2)
        lock = freeze(p, git_commit="abc123", frozen_at="2026-09-29T00:00:00+00:00")
        lp = os.path.join(d, "plan_v2.lock.json")
        assert os.path.exists(lp)
        on_disk = load_plan(lp)
        assert on_disk == lock and set(lock) == {"plan_path", "sha256", "git_commit", "frozen_at"}
        assert lock["sha256"] == canonical_hash(TOY) and lock["git_commit"] == "abc123"
        r = check(p)
        assert r["frozen"] and r["matches"] and r["sha256"] == r["locked_sha256"] == lock["sha256"]
        assert require_confirmatory(p)["matches"]
        # reformat + reorder keys: same content -> still frozen
        _write(p, dict(reversed(list(TOY.items()))), separators=(",", ":"))
        assert check(p)["matches"]
        again = freeze(p, git_commit="zzz", frozen_at="2030-01-01T00:00:00+00:00")
        assert again["frozen_at"] == "2026-09-29T00:00:00+00:00" and again["git_commit"] == "abc123"
        # explicit lock path
        lp2 = os.path.join(d, "locks", "custom.json")
        freeze(p, lock_path=lp2, git_commit="c", frozen_at="t")
        assert check(p, lp2)["matches"] and require_confirmatory(p, lp2)["frozen"]


def test_edit_breaks_lock():
    """Any content edit after freeze breaks the match; require_confirmatory raises;
    silent re-freeze is refused, a forced one records what it supersedes."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.json")
        _write(p, TOY)
        old = freeze(p, git_commit="a", frozen_at="t0")["sha256"]
        edited = json.loads(json.dumps(TOY)); edited["gates"][0]["thresholds"]["delta"] = 0.02
        _write(p, edited)
        r = check(p)
        assert r["frozen"] and not r["matches"] and r["locked_sha256"] == old != r["sha256"]
        try:
            require_confirmatory(p); raise AssertionError("edited plan passed require_confirmatory")
        except PlanNotFrozen as e:
            assert "changed since it was frozen" in str(e)
        try:
            freeze(p); raise AssertionError("silent re-freeze of an edited plan")
        except FileExistsError:
            pass
        new = freeze(p, git_commit="b", frozen_at="t1", force=True)
        assert new["supersedes"] == old and check(p)["matches"]
        require_confirmatory(p)


def test_missing_lock():
    """No lock (or an unreadable one) -> frozen False, never raises; require raises."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.json")
        _write(p, TOY)
        r = check(p)
        assert r == dict(frozen=False, matches=False, sha256=canonical_hash(TOY),
                         locked_sha256=None, lock_path=os.path.join(d, "plan.lock.json"),
                         frozen_at=None)
        try:
            require_confirmatory(p); raise AssertionError("unfrozen plan passed")
        except PlanNotFrozen as e:
            assert "no lock" in str(e)
        for bad in ("{not json", "[1, 2]", ""):
            with open(os.path.join(d, "plan.lock.json"), "w") as f:
                f.write(bad)
            assert check(p)["frozen"] is False
        assert issubclass(PlanNotFrozen, Exception)


def test_freeze_defaults():
    """Default git_commit is HEAD or 'unknown'; default frozen_at is tz-aware ISO."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.json")
        _write(p, TOY)
        lock = freeze(p)
        gc = lock["git_commit"]
        assert gc == "unknown" or (len(gc) == 40 and all(c in "0123456789abcdef" for c in gc))
        t = datetime.fromisoformat(lock["frozen_at"])
        assert t.tzinfo is not None and t.microsecond == 0


def test_log_deviation():
    """Ledger lines are '- <iso-date>: <text>', appended, one line each."""
    with tempfile.TemporaryDirectory() as d:
        led = os.path.join(d, "sub", "deviations.md")
        l1 = log_deviation("moved C1 ece_max 0.06 -> 0.07", led, when=date(2026, 9, 30))
        l2 = log_deviation("two\nlines  here", led, when=datetime(2026, 10, 1, 12, 5))
        l3 = log_deviation("string date", led, when="2026-10-02")
        l4 = log_deviation("today", led)
        assert l1 == "- 2026-09-30: moved C1 ece_max 0.06 -> 0.07"
        assert l2 == "- 2026-10-01: two lines here" and l3.startswith("- 2026-10-02: ")
        date.fromisoformat(l4[2:12])
        with open(led) as f:
            assert f.read().splitlines() == [l1, l2, l3, l4]
        # a ledger without a trailing newline still gets a clean new line
        with open(led, "a") as f:
            f.write("tail without newline")
        l5 = log_deviation("after", led, when="2026-10-03")
        with open(led) as f:
            lines = f.read().splitlines()
        assert lines[-2:] == ["tail without newline", l5]


def test_cli():
    """main() prints JSON and returns 0/1/2; `python -m manifold_mvp.prereg` works."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.json")
        _write(p, TOY)

        def run(*argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = main(list(argv))
            return code, json.loads(buf.getvalue())

        code, r = run("check", p)
        assert code == 1 and r["frozen"] is False
        code, r = run("freeze", p)
        assert code == 0 and r["sha256"] == canonical_hash(TOY)
        _write(p, {**TOY, "status": "edited"})
        code, r = run("freeze", p)
        assert code == 2 and "error" in r
        code, r = run("freeze", p, "--force")
        assert code == 0 and "supersedes" in r
        out = subprocess.run([sys.executable, "-m", "manifold_mvp.prereg", "check", p],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout)["matches"] is True


def test_real_plan():
    """prereg/plan_v2.json: loads, draft and unlocked, all gates present and well-formed,
    thresholds encode the spec."""
    plan = load_plan(REAL_PLAN)
    for k in ("version", "status", "created", "datasets", "splits", "seeds", "statistics", "gates"):
        assert k in plan, f"plan missing {k}"
    assert plan["created"] == "2026-09-29"
    r = check(REAL_PLAN)                      # read-only: never freeze the real plan here
    if plan["status"] == "draft":
        assert not r["frozen"], "a draft plan must not have a lock"
    else:
        assert plan["status"] == "frozen" and r["matches"], "status 'frozen' needs a matching lock"
    ds = plan["datasets"]
    assert ds["locomo"] == "data/locomo10.json"
    assert ds["locomo_conv"] == ["data/locomo10_dialog.json", "data/locomo10_multimem_full.json"]
    sp = plan["splits"]
    assert sp["unit"] == "conversation"
    assert sp["dev_folds"] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    assert sorted(c for f in sp["dev_folds"] for c in f) == list(range(10))
    assert len(set(plan["seeds"]["learned"])) >= 5
    st = plan["statistics"]
    assert st["ci"] == ["cluster_t", "cluster_bootstrap"] and st["paired_test"] == "mcnemar_exact"
    assert st["multiplicity"] == "holm" and st["alpha"] == 0.05

    gates = {g["id"]: g for g in plan["gates"]}
    assert len(gates) == len(plan["gates"]), "duplicate gate id"
    assert set(gates) == GATE_IDS, set(gates) ^ GATE_IDS
    for gid, g in gates.items():
        for k in GATE_FIELDS:
            assert g.get(k), f"{gid} missing/empty {k}"
        assert g["family"] == FAMILY[gid], gid
        assert isinstance(g["arms"], list) and all(isinstance(a, str) and a for a in g["arms"])
        assert all(isinstance(g[k], str) for k in ("hypothesis", "metric", "pass", "kill"))
    # the Holm family holds exactly the confirmatory tests the gates declare
    declared = [t for g in gates.values() for t in g.get("tests", [])]
    assert sorted(declared) == sorted(st["holm_family"])
    assert {t.split(".")[0] for t in declared} <= {i for i, f in FAMILY.items() if f == "confirmatory"}

    th = {gid: g["thresholds"] for gid, g in gates.items()}
    assert th["X0"] == {"hybrid_rerank_hit1": 0.524, "dense_minilm_hit1": 0.176, "tolerance": 0.02}
    assert th["G1a"]["delta"] == 0.03 and th["G1a"]["k"] == 10
    assert th["G1b"]["k"] == 30 and th["G1b"]["delta_multihop"] == 0.05 and th["G1b"]["ni_margin"] == 0.02
    assert th["G1c"]["k"] == 3 and th["G1c"]["ni_margin"] == 0.02
    assert (th["A1"]["auroc_min"], th["A1"]["auroc_lower_bound_gt"], th["A1"]["false_abstention_max"]) == (0.70, 0.60, 0.10)
    assert th["C1"]["ece_max"] == 0.06 and th["C1"]["slope_range"] == [0.8, 1.25]
    assert (th["R1"]["flip_max"], th["R1"]["option_name_effect_max_points"], th["R1"]["post_platt_ece_max"]) == (0.02, 3, 0.05)
    assert th["E1prime"]["delta"] == 0.03 and th["P1"]["delta"] == 0.05
    assert th["U1"]["gap_closed_min"] == 0.30


def test_real_ledger():
    """prereg/deviations.md exists, has a header and the v1-retirement entry."""
    with open(os.path.join(ROOT, "prereg", "deviations.md"), encoding="utf-8") as f:
        text = f.read()
    entries = [l for l in text.splitlines() if l.startswith("- ")]
    assert text.startswith("# ") and entries
    first = entries[0]
    assert first.startswith("- 2026-09-29: ") and "delta2" in first and "retired" in first


TESTS = [test_hash_stable, test_freeze_check_roundtrip, test_edit_breaks_lock,
         test_missing_lock, test_freeze_defaults, test_log_deviation, test_cli,
         test_real_plan, test_real_ledger]

if __name__ == "__main__":
    for fn in TESTS:
        fn()
        print(f"  {fn.__name__} OK")
    print("\nALL PREREG CHECKS PASSED")
