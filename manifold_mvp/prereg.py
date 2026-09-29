"""
Pre-registration: freeze the confirmatory plan BEFORE the confirmatory run.

v1 kept its gates in config.PreReg, a dataclass anyone could edit between runs,
and the LoCoMo reality check showed why that is not enough: every v1 gate had to
be re-read against honest baselines after the fact. v2 moves the plan into a
JSON file (prereg/plan_v2.json) and makes "frozen" a checkable fact:

  canonical_hash        sha256 of the plan's CONTENT (sorted keys, no whitespace),
                        so reformatting the file never breaks a lock, any edit does
  freeze                write <plan>.lock.json = {plan_path, sha256, git_commit,
                        frozen_at}; refuses to silently re-freeze edited content
  check                 is there a lock, and does the plan still match it?
  require_confirmatory  what eval scripts call first: raises PlanNotFrozen unless
                        the plan is frozen and unchanged (exploratory runs skip it)
  log_deviation         append-only ledger (prereg/deviations.md) of every change
                        made after a freeze, and of retired gates

Workflow: edit the plan (status "draft") -> set status "frozen" ->
`python -m manifold_mvp.prereg freeze prereg/plan_v2.json` -> commit plan + lock
together -> confirmatory runs call require_confirmatory(). Editing afterwards is
a deviation: log it, then re-freeze with --force (the new lock records the hash
it supersedes).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path


class PlanNotFrozen(Exception):
    """The plan has no lock, or changed since it was frozen."""


def canonical_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_plan(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _lock_path(plan_path, lock_path=None) -> Path:
    return Path(lock_path) if lock_path is not None else Path(plan_path).with_suffix(".lock.json")


def _read_lock(lock_path) -> dict:
    """{} for a missing, unreadable or malformed lock — never raises."""
    try:
        with open(lock_path, encoding="utf-8") as f:
            lock = json.load(f)
    except (OSError, ValueError):
        return {}
    return lock if isinstance(lock, dict) else {}


def _git_head(cwd) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd or ".",
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def freeze(plan_path, lock_path=None, git_commit=None, frozen_at=None, force=False) -> dict:
    """Write the lock for the plan's current content and return it.
    Re-freezing identical content is a no-op (the ORIGINAL frozen_at is kept).
    Re-freezing edited content raises FileExistsError unless force=True: a
    post-freeze edit is a deviation, so log_deviation() it first; the new lock
    then records the hash it supersedes."""
    lp = _lock_path(plan_path, lock_path)
    sha = canonical_hash(load_plan(plan_path))
    old = _read_lock(lp)
    if old.get("sha256") == sha:
        return old
    if old.get("sha256") and not force:
        raise FileExistsError(
            f"{lp} already freezes sha256 {old['sha256'][:12]} but {plan_path} is now "
            f"{sha[:12]}. Editing a frozen plan is a deviation: log it "
            f"(prereg.log_deviation) and re-freeze with force=True / --force.")
    lock = dict(plan_path=str(plan_path), sha256=sha,
                git_commit=git_commit or _git_head(os.path.dirname(os.path.abspath(plan_path))),
                frozen_at=frozen_at or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if old.get("sha256"):
        lock["supersedes"] = old["sha256"]
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    return lock


def check(plan_path, lock_path=None) -> dict:
    lp = _lock_path(plan_path, lock_path)
    sha, lock = canonical_hash(load_plan(plan_path)), _read_lock(lp)
    locked = lock.get("sha256")
    return dict(frozen=locked is not None, matches=locked == sha, sha256=sha,
                locked_sha256=locked, lock_path=str(lp), frozen_at=lock.get("frozen_at"))


def require_confirmatory(plan_path, lock_path=None) -> dict:
    """Gate for confirmatory runs; returns check() so callers can log the hash
    next to their results."""
    r = check(plan_path, lock_path)
    if not r["frozen"]:
        raise PlanNotFrozen(
            f"{plan_path}: no lock at {r['lock_path']}. Freeze the plan "
            f"(python -m manifold_mvp.prereg freeze {plan_path}) before a confirmatory "
            f"run; exploratory runs do not call require_confirmatory.")
    if not r["matches"]:
        raise PlanNotFrozen(
            f"{plan_path} changed since it was frozen (sha256 {r['sha256'][:12]} != "
            f"locked {r['locked_sha256'][:12]}): results would not be pre-registered. "
            f"Revert the edit, or log a deviation and re-freeze with --force.")
    return r


def log_deviation(text, ledger_path="prereg/deviations.md", when=None) -> str:
    """Append '- <iso-date>: <text>' (one line) to the ledger; returns the line."""
    when = when or datetime.now(timezone.utc).date()
    if isinstance(when, datetime):
        when = when.date()
    stamp = when.isoformat() if isinstance(when, date) else str(when)
    line = f"- {stamp}: {' '.join(str(text).split())}\n"
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lead = "\n" if p.exists() and p.stat().st_size and not p.read_bytes().endswith(b"\n") else ""
    with open(p, "a", encoding="utf-8") as f:
        f.write(lead + line)
    return line.rstrip("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m manifold_mvp.prereg",
                                 description="Freeze / check a pre-registration plan.")
    ap.add_argument("cmd", choices=["freeze", "check"])
    ap.add_argument("plan")
    ap.add_argument("--lock", default=None, help="lock path (default: <plan>.lock.json)")
    ap.add_argument("--force", action="store_true",
                    help="freeze: overwrite a lock for different content (log a deviation first)")
    a = ap.parse_args(argv)
    try:
        r = freeze(a.plan, a.lock, force=a.force) if a.cmd == "freeze" else check(a.plan, a.lock)
    except FileExistsError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 2
    print(json.dumps(r, indent=2))
    return 0 if a.cmd == "freeze" or r["matches"] else 1


if __name__ == "__main__":
    sys.exit(main())
