"""Run every theory gate, save every output, print one summary table.

This replaces the shell one-liner that used to live in status.md section 7, which
had two defects that between them cost a day.

  1. It NAMED the gates.  When a gate was added or renamed the list silently went
     stale, so "all gates pass" described a subset -- which is how the
     clamp_inputs regression in verify_theorem4 went unnoticed (section 8b).  This
     script GLOBS `verify_*.py`, so a new gate is included the moment it exists
     and cannot be forgotten.

  2. It wrote output to /tmp.  A gate whose output lands in a temporary directory
     produces no artifact, so its result is unciteable an hour later -- lesson 21,
     the same failure that left the Delta^2 law, the 40x tie result and the
     competing-risks bias as prose in status.md with nothing behind them.  Output
     goes to theory/<gate>_results.txt and stays there.

Exit code is the number of failing gates, so this is usable as a pre-submission
check: `python theory/run_gates.py && echo SAFE`.

Run:  python -u theory/run_gates.py [--quick] [gate ...]
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def discover(names):
    if names:
        want = {n if n.startswith("verify_") else f"verify_{n}" for n in names}
        return sorted(p for p in HERE.glob("verify_*.py") if p.stem in want)
    return sorted(HERE.glob("verify_*.py"))


def verdict(text, code):
    """Read the gate's own self-report, falling back to the exit code.

    Gates print their own verdict line; that is the authoritative statement,
    because a gate can exit 0 while reporting a failed comparison.  Failure words
    are checked first, then the exit status, then affirmative markers.

    UNCLEAR is a distinct outcome, not a silent pass and not a failure.  The
    first version of this function returned "? (exit 0)" for verify_profiled_info
    -- which prints "VERDICT: Delta^2 SURVIVES profiling" and no PASS token --
    and the summary below counted anything not starting with "PASS" as broken,
    so six passing gates were reported as 5/6.  The durable fix was to make that
    gate emit the standard `RESULT: GATE PASSES` line; this branch stays so the
    next gate that forgets is surfaced loudly instead of being miscounted.
    """
    tail = text[-4000:].upper()
    if "GATE FAILED" in tail or "FAILURE" in tail or "DO NOT ASSERT" in tail:
        return "FAIL"
    if "VIOLATED" in tail and "OK" not in tail.split("VIOLATED")[-1][:200]:
        return "FAIL"
    if code != 0:
        return f"FAIL (exit {code})"
    if ("GATE PASSES" in tail or "ALL GATES PASS" in tail
            or "SUPPORTS THE CLAIM" in tail or "SURVIVES" in tail
            or "-> PASS" in tail):
        return "PASS"
    return "UNCLEAR"


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    quick = "--quick" in sys.argv
    gates = discover(argv)
    if not gates:
        print("no gates found")
        return 1

    print("=" * 78)
    print(f"THEORY GATES  ({len(gates)} discovered by glob, not by a hand-written list)")
    print("=" * 78)

    results = []
    for g in gates:
        out = HERE / f"{g.stem}_results.txt"
        cmd = [sys.executable, "-u", str(g)] + (["--quick"] if quick else [])
        t0 = time.time()
        print(f"  running {g.stem} ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        text = proc.stdout + proc.stderr
        out.write_text(text, encoding="utf-8")
        v = verdict(text, proc.returncode)
        results.append((g.stem, v, time.time() - t0, out))
        print(f"    -> {v}   [{time.time() - t0:.0f}s]  -> {out.name}", flush=True)

    print()
    print("=" * 78)
    print(f"  {'gate':<26}{'verdict':<16}{'seconds':>9}  artifact")
    for name, v, secs, out in results:
        print(f"  {name:<26}{v:<16}{secs:>9.0f}  {out.name}")
    bad = [r for r in results if r[1].startswith("FAIL")]
    unclear = [r for r in results if r[1] == "UNCLEAR"]
    print()
    print(f"RESULT: {len(results) - len(bad) - len(unclear)}/{len(results)} gates pass")
    if bad:
        print("  FAILING: " + ", ".join(r[0] for r in bad))
    if unclear:
        print("  UNCLEAR (gate printed no recognisable verdict line -- fix the gate,")
        print("           do not assume it passed): " + ", ".join(r[0] for r in unclear))
    return len(bad) + len(unclear)


if __name__ == "__main__":
    raise SystemExit(main())
