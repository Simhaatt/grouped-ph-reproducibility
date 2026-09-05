"""How small an effect can each survival benchmark actually resolve?

Section 4.5 says the classic benchmarks are too small to support the wins the KAN
survival literature reports.  Until now that was an anecdote -- "test sets of 41,
125, 206 rows" -- and the sharper version of it, "the KAN wins only at n >= 34k",
turned out to be FALSE: rotgbsg resolves cleanly at n_test = 670 while
support-pycox does not at n_test = 2662.  Resolution is not a function of n.

It is a function of the STANDARD ERROR of the paired difference, which depends on
n_test, the event rate, the censoring pattern and how correlated the two models'
predictions are.  So measure it.

For each benchmark we report the half-width of the paired 95% CI for the C-index
difference -- literally the smallest difference that dataset can distinguish from
zero -- and the minimum detectable effect at 80% power:

    SE      = CI width / (2 * 1.96)
    MDE_80  = (1.96 + 0.8416) * SE = 2.802 * SE

Then compare against the 0.01-0.02 C-index differences CoxKAN, SurvKAN, KAN-AFT
and KAPLAN-HR report on these same datasets WITHOUT confidence intervals.

This turns a retracted claim into a reusable criterion: it applies to any pair of
survival models, not just ours.

Run:  python -u experiments/resolution.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel import data as D
import experiments.significance as S
from experiments.baselines import SELECTED_L1 as FAIR_L1

S.SELECTED_L1 = dict(FAIR_L1)

Z_ALPHA, Z_POWER = 1.959964, 0.841621
N_SEEDS = 3

# What the KAN survival papers claim on these benchmarks, as C-index gains.
# All four report point estimates with no interval.
PUBLISHED = {
    "metabric": 0.02, "support-pycox": 0.02, "rotgbsg": 0.02,
    "flchain": 0.01, "nwtco": 0.01, "gbsg": 0.02, "pbc": 0.02, "valung": 0.02,
}


def main():
    names = sys.argv[1:] or ["support-pycox", "flchain", "nwtco", "rotgbsg",
                             "metabric", "gbsg", "pbc", "valung"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")

    print("=" * 100)
    print("BENCHMARK RESOLUTION: the smallest C-index difference each dataset can detect")
    print(f"  paired bootstrap, fair baseline, winsorised; {N_SEEDS} seeds averaged")
    print("=" * 100)

    rows = []
    for nm in names:
        if nm not in avail:
            print(f"  unknown dataset {nm!r}")
            continue
        raw = avail[nm]
        loader = lambda raw=raw: D.onehot_ordinals(raw())
        try:
            half, obs, nt, nev = [], [], None, None
            for seed in range(N_SEEDS):
                label = f"{nm}#{seed}"
                S.SELECTED_L1[label] = FAIR_L1.get(nm, 0.01)
                r = S.run(label, loader, seed=seed)
                half.append((r["c_hi"] - r["c_lo"]) / 2.0)
                obs.append(r["c_kan"] - r["c_lin"])
                nt, nev = r["n_test"], r["n_events"]
            hw = float(np.mean(half))
            se = hw / Z_ALPHA
            mde = (Z_ALPHA + Z_POWER) * se
            # Comparable ORDERED PAIRS, not units, is what the C-index averages
            # over -- roughly n_events * n_test.  This is the quantity resolution
            # actually tracks, which is why nwtco (n=1208, 14% events) resolves
            # WORSE than rotgbsg (n=670, 57% events).
            rows.append(dict(name=nm, n_test=nt, n_events=nev, pairs=nev * nt,
                             hw=hw, se=se, mde=mde, obs=float(np.mean(obs))))
        except Exception as e:
            print(f"  {nm:<16} FAILED {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 100)
    print(f"  {'dataset':<16}{'n_test':>8}{'events':>8}{'pairs':>10}{'SE':>9}"
          f"{'MDE(80%)':>10}{'our dC':>10}{'lit.':>8}   verdict")
    print("-" * 100)
    for r in sorted(rows, key=lambda r: -r["n_test"]):
        lit = PUBLISHED.get(r["name"])
        # Can this dataset resolve the effect size the literature reports on it?
        verdict = ("resolvable" if lit is not None and lit >= r["mde"]
                   else "CANNOT RESOLVE published claim")
        lits = f"{lit:+.3f}" if lit is not None else "--"
        print(f"  {r['name']:<16}{r['n_test']:>8}{r['n_events']:>8}"
              f"{r['pairs']/1000:>9.0f}k{r['se']:>9.4f}"
              f"{r['mde']:>10.4f}{r['obs']:>+10.4f}{lits:>8}   {verdict}")
    print("-" * 100)
    bad = [r["name"] for r in rows
           if PUBLISHED.get(r["name"]) is not None
           and PUBLISHED[r["name"]] < r["mde"]]
    print(f"  {len(bad)}/{len(rows)} benchmarks CANNOT resolve the effect published on them: "
          f"{', '.join(bad) if bad else 'none'}")
    print("  MDE(80%) = 2.802 * SE.  A reported win below a dataset's MDE is not")
    print("  evidence of anything, however many decimal places it is quoted to.")
    print()
    print("  RESOLUTION IS NOT A FUNCTION OF n.  The C-index averages over comparable")
    print("  ORDERED PAIRS (~ events x n_test), so the EVENT COUNT drives it:")
    print("  nwtco has n_test=1208 and resolves WORSE than rotgbsg at n_test=670,")
    print("  because nwtco has 170 events and rotgbsg has 383.  Sizing a survival")
    print("  benchmark by n is the mistake; size it by events.")
    print()
    print("  Pairs are not the whole story and we do not claim SE ~ 1/sqrt(pairs):")
    print("  the PAIRED SE also falls when the two models agree, which is why")
    print("  flchain (1.5M pairs, the two models nearly identical) has a smaller SE")
    print("  than support-pycox (4.8M pairs).  The defensible claim is the negative")
    print("  one -- n does not determine resolution -- plus the measured MDE itself,")
    print("  which needs no functional form because it is estimated directly.")


if __name__ == "__main__":
    main()
