"""E4, taken to its conclusion: how much can more splits actually buy?

E4 asked for the split count to rise from 3 to at least 20, on the grounds that
several reported effects are smaller than their standard errors.  Raising it is
right, but the 120-replication confirmation of E6d exposed a limit that has to be
stated rather than discovered by a referee.

The Nadeau-Bengio variance for K overlapping random subsampling splits is

    Var_hat = (1/K + f/(1-f)) s^2 ,     f = test fraction,

with s the standard deviation across splits.  Only the FIRST term shrinks with K.
The second is a constant set by the split geometry, so

    NB SE  ->  s sqrt(f/(1-f))    as K -> infinity,

which at f = 0.3 is 0.6547 s.  An effect smaller than about 1.28 s is therefore
UNRESOLVABLE at the 5% level however many splits are run, because
1.96 x 0.6547 = 1.283.

That is exactly what the E6d confirmation showed: going from 20 to 120
replications -- six times the compute -- moved the multiplier from 0.6918 to
0.6610, a 4.5% reduction, and the negative cells stayed unresolved.

The consequence for the paper is that "not resolved" has two very different
causes and they must not be conflated.  One is too few splits, which more
compute fixes.  The other is an effect below the geometry's floor, which only
more DATA or a genuinely independent test set fixes.  This script says which is
which, per split count and per test fraction.

Run:  python -u experiments/se_floor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT = Path(__file__).resolve().parent / "se_floor.txt"


def mult(K, f):
    """NB SE as a multiple of the across-split standard deviation s."""
    return float(np.sqrt(1.0 / K + f / (1.0 - f)))


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 92)
    log("HOW MUCH CAN MORE SPLITS BUY?  The Nadeau-Bengio floor.")
    log("  NB SE = s * sqrt(1/K + f/(1-f));  only the 1/K term shrinks with K.")
    log("=" * 92)
    log("  NB SE as a multiple of s, and the smallest resolvable effect (1.96 SE):")
    log(f"  {'K':>6}" + "".join(f"{'f=' + f'{f:.2f}':>22}" for f in (0.2, 0.3, 0.4)))
    log(f"  {'':>6}" + "".join(f"{'SE/s':>11}{'MDE/s':>11}" for _ in range(3)))
    for K in (3, 5, 10, 20, 50, 120, 1000, 10 ** 6):
        row = f"  {K:>6}"
        for f in (0.2, 0.3, 0.4):
            m = mult(K, f)
            row += f"{m:>11.4f}{1.959964 * m:>11.4f}"
        log(row)
    log("")
    log("  The last row is the limit.  At the project's f = 0.3 the NB SE never")
    log("  falls below 0.6547 s, so an effect below 1.283 s is unresolvable at the")
    log("  5% level no matter how many splits are run.")
    log("")
    log("  NAIVE SE, for contrast, is s/sqrt(K) and has no floor:")
    log(f"  {'K':>6}{'naive SE/s':>14}{'NB SE/s (f=0.3)':>18}{'ratio':>9}")
    for K in (3, 5, 10, 20, 50, 120):
        n_, b_ = 1.0 / np.sqrt(K), mult(K, 0.3)
        log(f"  {K:>6}{n_:>14.4f}{b_:>18.4f}{b_ / n_:>9.2f}")
    log("")
    log("  At K = 20 the honest interval is 3.09x the naive one, and at K = 120 it")
    log("  is 7.24x.  Reporting sd/sqrt(K) on overlapping splits therefore gets")
    log("  MORE wrong as the split count rises, which is the opposite of the")
    log("  intuition that more splits make a naive error bar safer.")
    log("")
    log("=" * 92)
    log("WHAT THIS MEANS FOR E6d's SIGN FLIP")
    log("=" * 92)
    log("  The two negative cells at 120 replications were")
    log("    T=80   D_T = -0.00972   NB SE 0.04906")
    log("    T=160  D_T = -0.02124   NB SE 0.04145")
    log("  Their across-split standard deviations are therefore")
    for dt, nb, K in ((-0.00972, 0.04906, 120), (-0.02124, 0.04145, 120)):
        s = nb / mult(K, 0.3)
        log(f"    s = {s:.5f}  ->  floor on NB SE = {0.6547 * s:.5f},"
            f"  smallest resolvable |D_T| = {1.283 * s:.5f}"
            f"  (observed {abs(dt):.5f})")
    log("")
    log("  Both observed effects are below their own floors, so NO number of")
    log("  splits resolves them.  The evidence for the crossover is therefore the")
    log("  PATTERN -- a sign change that moves right as n grows, plus E9's")
    log("  independent nuisance-dimension result -- and the paper must present it")
    log("  that way rather than promising significance that this design cannot")
    log("  deliver.")
    log("")
    log("  The honest fix for a specific cell is more DATA or an independent test")
    log("  set, not more resamples of the same cohort.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
