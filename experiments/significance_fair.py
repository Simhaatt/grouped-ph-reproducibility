"""§4.3 redone against the FAIR baseline (one-hot ordinals for every model).

`significance.py` predates the §4.7 encoding fix: it feeds the raw loader output
to both models, so its linear comparator still codes 3-6 level ordinals as
0/1/2.  That is the same weak baseline that inflated the measured KAN advantage
~3x on support-pycox, so every CI in `significance_results.txt` is a comparison
against a model we have since declared indefensible.

This re-runs the identical bootstrap with `onehot_ordinals` applied, and adds
the two datasets that actually carry the KAN claim (rotgbsg, support-pycox),
which the original list never included.

l1 follows baselines.py's SELECTED_L1 so the two tables describe the same fits.

Run:  python -u experiments/significance_fair.py [dataset ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kanrel import data as D
import experiments.significance as S
from experiments.baselines import SELECTED_L1 as FAIR_L1

S.SELECTED_L1 = dict(FAIR_L1)


def main():
    names = sys.argv[1:] or ["rotgbsg", "support-pycox", "metabric",
                             "flchain", "nwtco", "support2/slos",
                             "gbsg", "pbc", "valung"]
    avail = dict(D.LOADERS)
    avail["sparcs/drg302"] = lambda: D.load_sparcs("302")
    avail["drsa/music"] = lambda: D.load_drsa("MUSIC", max_rows=200000)

    print("=" * 88)
    print("PAIRED BOOTSTRAP (FAIR BASELINE: one-hot ordinals for both models)")
    print("  positive difference = KAN assigns higher likelihood = KAN better")
    print("=" * 88)
    out = []
    for nm in names:
        if nm not in avail:
            print(f"  unknown dataset {nm!r}")
            continue
        raw = avail[nm]
        try:
            out.append(S.run(nm, lambda raw=raw: D.onehot_ordinals(raw())))
        except Exception as e:
            print(f"  {nm:<16} FAILED {type(e).__name__}: {str(e)[:60]}")

    print()
    print("=" * 88)
    print(f"  {'dataset':<16}{'n_test':>8}{'NLL diff':>12}{'95% CI':>26}{'verdict':>18}")
    for r in sorted(out, key=lambda r: -r["n_test"]):
        print(f"  {r['name']:<16}{r['n_test']:>8}{r['mean']:>+12.5f}"
              f"   [{r['lo']:+.5f}, {r['hi']:+.5f}]{r['sig']:>18}")
    print(f"\n  {'dataset':<16}{'C linear':>10}{'C KAN':>10}{'delta':>10}")
    for r in sorted(out, key=lambda r: -r["n_test"]):
        print(f"  {r['name']:<16}{r['c_lin']:>10.4f}{r['c_kan']:>10.4f}"
              f"{r['c_kan']-r['c_lin']:>+10.4f}")


if __name__ == "__main__":
    main()
