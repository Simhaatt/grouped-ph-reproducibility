"""Verify that the frozen result logs still reproduce from the current code.

WHY THIS EXISTS.  check_numbers.py verifies the manuscript against the frozen
result logs.  Nothing verified the frozen logs against the code that claims to
produce them, and on 2026-09-06 that gap bit: `stable_seed` landed in
kanrel/stats.py on 2026-09-05 to fix non-reproducible seeding, only the E8
simulation log was regenerated, and six others silently kept values the code no
longer produces.  Every check passed the whole time, because each compared the
manuscript to a stale log rather than to reality.  Two manuscript numbers were
wrong as a result.

A checker that validates A against B, where B is itself derived and mutable, is
only as good as its guarantee that B is current.  This supplies that guarantee.

TWO LEVELS, AND THE DIFFERENCE MATTERS.

  SUSPECT   The log is older than code it actually imports.  This is a
            heuristic: it cannot distinguish a comment change from a changed
            estimator, so it OVER-reports by design.  Suspect is a prompt to
            probe, not a verdict.

  STALE     A recompute probe re-ran one representative cell and got a
            different answer.  This is a verdict, and it fails the run.

  SUPERSEDED  A log that fails its probe AND has a recorded replacement.  E5 is
            the case: exact_arm_sim.txt reruns the same design under current
            seeding.  Recorded rather than silently tolerated, so that an
            UNEXPECTED stale log still fails -- otherwise the run would fail
            forever on a known-retired file and a reader would stop looking.

Dependencies are the real import closure of each experiment script, restricted
to project modules, not a blanket list -- a change to kanrel/stats.py should
flag the scripts that import it and leave the rest alone.  The first version of
this file used a blanket list and reported 18 of 22 logs stale, which is the
kind of noise that trains a reader to ignore the report.

Run:  python paper/check_freshness.py [--probe]
      --probe also recomputes a representative cell where one is defined.
"""
from __future__ import annotations

import ast
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "experiments"

sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

# log -> (script that produces it, probe key or None)
SPEC = {
    # Superseded on purpose: exact_arm_sim.py reruns this design on the
    # same cells under current seeding and adds the exact-discrete arm.
    # Expected staleness is recorded so that an UNEXPECTED stale log is
    # still a failure; without this the run fails forever and a reader
    # learns to ignore it.
    "simulations_e5.txt": ("experiments/simulations.py", "e5",
                           "exact_arm_sim.txt"),
    "simulations_e6c.txt": ("experiments/simulations.py", None),
    "simulations_e6d.txt": ("experiments/simulations.py", None),
    "simulations_e6e7e8.txt": ("experiments/simulations.py", None),
    "simulations_e7.txt": ("experiments/simulations.py", None),
    "simulations_e8.txt": ("experiments/simulations.py", None),
    "simulations_e6e6ce6de7_seedfix.txt": ("experiments/simulations.py", None),
    "exact_arm_sim.txt": ("experiments/exact_arm_sim.py", "exact_arm"),
    "exact_arm_real.txt": ("experiments/exact_arm_real.py", None),
    "exact_arm_ladder.txt": ("experiments/exact_arm_ladder.py", None),
    "protocol_decomp.txt": ("experiments/protocol_decomp.py", None),
    "kp_vs_breslow.txt": ("experiments/kp_vs_breslow.py", None),
    "kp_out_of_sample.txt": ("experiments/kp_out_of_sample.py", None),
    "kp_out_of_sample_v2.txt": ("experiments/kp_out_of_sample_v2.py", None),
    "crossover_estimator.txt": ("experiments/crossover_estimator.py", None),
    "crossover_multi.txt": ("experiments/crossover.py", None),
    "resolution.txt": ("experiments/resolution.py", None),
    "pooled_cv.txt": ("experiments/pooled_cv.py", None),
    "formulation_ci.txt": ("experiments/formulation_ci.py", None),
    "gam_comparator.txt": ("experiments/gam_comparator.py", None),
    "competing_risks.txt": ("experiments/competing_risks.py", None),
    "symbolic.txt": ("experiments/symbolic.py", None),
    "baselines_fair.txt": ("experiments/baselines.py", None),
}

PROBE_CELL = dict(shape=1.0, censor=0.2, n=500, T=4)


def module_path(name):
    """Resolve a dotted project module to a file, or None if it is external."""
    parts = name.split(".")
    for cand in (ROOT.joinpath(*parts).with_suffix(".py"),
                 ROOT.joinpath(*parts) / "__init__.py"):
        if cand.exists():
            return cand
    return None


def closure(script, seen=None):
    """Every project module reachable from `script` by import, transitively."""
    seen = set() if seen is None else seen
    p = ROOT / script if not Path(script).is_absolute() else Path(script)
    if not p.exists() or p in seen:
        return seen
    seen.add(p)
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return seen
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            names += [f"{node.module}.{a.name}" for a in node.names]
    for n in names:
        m = module_path(n)
        if m is not None:
            closure(m, seen)
    return seen


def parse_e5_cell(path, c):
    pat = re.compile(
        rf"\s+{c['shape']:.1f}\s+{int(c['censor'] * 100)}%\s+{c['n']}\s+"
        rf"{c['T']}\s+[\d.]+\s+([+-][\d.]+)")
    for ln in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        m = pat.match(ln)
        if m:
            return float(m.group(1))
    return float("nan")


def probe_e5(log_path):
    import numpy as np
    from experiments.simulations import sim_weibull_grouped, one_rep, agg
    from kanrel.stats import stable_seed
    beta = np.array([0.7, -0.5, 0.3, 0.4, -0.6])
    c = PROBE_CELL
    rows = []
    for s in range(20):
        rng = np.random.default_rng(
            stable_seed(c["shape"], c["censor"], c["n"], c["T"], s))
        X, idx, ev = sim_weibull_grouped(c["n"], c["T"], beta, c["shape"],
                                         rng, c["censor"])
        try:
            rows.append(one_rep(X, idx, ev, c["T"], rng))
        except Exception:
            pass
    return agg(rows, "D_T")[0], parse_e5_cell(log_path, c)


def probe_exact_arm(log_path):
    import json
    import numpy as np
    from experiments.exact_arm_sim import four_arms, BETA
    from experiments.simulations import sim_weibull_grouped
    from kanrel.stats import stable_seed
    c = PROBE_CELL
    vals = []
    for s in range(20):
        rng = np.random.default_rng(
            stable_seed(c["shape"], c["censor"], c["n"], c["T"], s))
        X, idx, ev = sim_weibull_grouped(c["n"], c["T"], BETA, c["shape"],
                                         rng, c["censor"])
        try:
            vals.append(four_arms(X, idx, ev, c["T"], rng)["D_A"])
        except Exception:
            pass
    got = float(np.mean(vals)) if vals else float("nan")
    j = log_path.with_suffix(".json")
    want = float("nan")
    if j.exists():
        for b in json.loads(j.read_text(encoding="utf-8")):
            if (b["shape"] == c["shape"] and b["censor"] == c["censor"]
                    and b["n"] == c["n"] and b["T"] == c["T"]):
                want = b["D_A"]
    return got, want


PROBES = {"e5": probe_e5, "exact_arm": probe_exact_arm}


def main():
    run_probes = "--probe" in sys.argv
    print("=" * 96)
    print("FRESHNESS OF THE FROZEN RESULT LOGS")
    print("  SUSPECT = older than code it imports (a prompt to probe, not a "
          "verdict; over-reports by design)")
    print("  STALE   = a recompute probe disagreed with the log (a verdict)")
    print("=" * 96)
    print(f"  {'log':<38}{'written':<18}{'verdict':<10}note")
    print("  " + "-" * 92)

    suspect, stale, missing, verified, fresh = [], [], [], [], []
    superseded = []
    for name in sorted(SPEC):
        path = EXP / name
        row = SPEC[name]
        script, probe = row[0], row[1]
        superseded_by = row[2] if len(row) > 2 else None
        if not path.exists():
            missing.append(name)
            continue
        mt = path.stat().st_mtime
        dep = max((p.stat().st_mtime for p in closure(script)), default=0)
        newest = max(closure(script),
                     key=lambda p: p.stat().st_mtime, default=None)
        is_suspect = mt < dep
        verdict = "SUSPECT" if is_suspect else "fresh"
        note = ""
        if is_suspect and newest is not None:
            note = f"older than {newest.relative_to(ROOT).as_posix()}"
        if probe and run_probes:
            try:
                got, want = PROBES[probe](path)
                if want != want:
                    note = (note + "; " if note else "") + "probe: no reference"
                elif abs(got - want) < 1e-9:
                    verdict = "VERIFIED"
                    note = f"probe reproduces {got:+.6f}"
                    verified.append(name)
                elif superseded_by:
                    verdict = "SUPERSEDED"
                    note = (f"replaced by {superseded_by}; probe "
                            f"{got:+.6f} vs log {want:+.6f}")
                    superseded.append(name)
                else:
                    verdict = "STALE"
                    note = f"probe got {got:+.6f}, log says {want:+.6f}"
                    stale.append(name)
            except Exception as e:
                note = (note + "; " if note else "") + \
                    f"probe error {type(e).__name__}: {str(e)[:40]}"
        elif probe:
            note = (note + "; " if note else "") + "probe available (--probe)"
        print(f"  {name:<38}{datetime.fromtimestamp(mt):%Y-%m-%d %H:%M}  "
              f"{verdict:<12}{note}")
        if verdict == "SUSPECT":
            suspect.append(name)
        elif verdict == "fresh":
            fresh.append(name)

    print()
    print("=" * 96)
    print(f"  {len(fresh)} fresh, {len(verified)} probe-VERIFIED, "
          f"{len(suspect)} suspect, {len(superseded)} superseded, "
          f"{len(stale)} STALE, {len(missing)} missing")
    for name in superseded:
        print(f"  {name} is superseded by {SPEC[name][2]} and is retained only "
              f"as the historical record; nothing should source it.")
    if stale:
        print()
        print("  CONFIRMED STALE -- these do not reproduce from the current "
              "code.  Any manuscript")
        print("  number sourced from them is unverified, and check_numbers.py "
              "will not catch it:")
        for s in stale:
            print(f"    {s}")
    if suspect and not stale:
        print()
        print("  Suspect logs are older than code they import.  That is often "
              "a cosmetic edit;")
        print("  add a probe for any log whose numbers the manuscript "
              "actually quotes.")
    if not run_probes:
        print()
        print("  Run with --probe to turn suspicion into a verdict where a "
              "probe is defined.")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
