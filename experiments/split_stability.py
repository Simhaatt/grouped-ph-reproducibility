"""Is a KAN-vs-linear win stable across splits, or a seed-0 artefact?

`significance_fair.py` found rotgbsg SIGNIFICANT at +0.0426 NLL on a 670-row
test set -- the only resolvable KAN-vs-linear win below n=34k once the fair
one-hot baseline is in place.  It is therefore about to become a headline, and
its paired bootstrap resamples ONLY the test rows at seed 0: it says nothing
about training variability.  A paper that criticises the literature for
reporting unresolvable differences (section 4.5) cannot rest a claim on one split.

This refits both models on 5 independent splits and bootstraps each.

Also reports TAIL CONCENTRATION: the share of the mean difference carried
by the 5 most extreme test rows.  flchain's fair-baseline CI came out
[+0.001, +0.089] around a mean of +0.033 -- a skew that badly asymmetric
means a handful of rows are driving the verdict, which is the valung
extrapolation pathology of section 4.8 rather than a real effect.

Run:  python -u experiments/split_stability.py [dataset ...]
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

# Stash each split's per-unit difference vector so we can look at its tail.
# S.paired_bootstrap is the only place it is visible without refitting.
_LAST = {}
_orig_boot = S.paired_bootstrap


def _boot(diff, *a, **k):
    _LAST["diff"] = np.asarray(diff)
    return _orig_boot(diff, *a, **k)


S.paired_bootstrap = _boot


def drop_top_k(diff, ks=(5, 20, 50)):
    """Does the effect SURVIVE deleting the k most extreme rows?

    Strictly better than tail_share, which divides by the mean and therefore
    returns nonsense (-864% on metabric) whenever the mean is near zero.  This
    has no denominator: delete the k largest |d_i| and re-estimate.

    It is what finally settled nwtco.  Its top-50 rows -- 1.2% of 4,028 -- carried
    90.8% of the mean, and dropping them took the gain from +0.0099 (resolved) to
    +0.0009 (not resolved).  So nwtco is a null, not a real improvement.  On
    rotgbsg the same test moves the gain the OTHER way (+0.0226 -> +0.0302, top-20
    rows contribute -7.6%): the effect is broad-based and the extremes understate
    it.  That contrast is the whole point.
    """
    out = []
    order = np.argsort(-np.abs(diff))
    for k in ks:
        if len(diff) <= k + 30:
            continue
        m, lo, hi = _orig_boot(diff[order[k:]])
        out.append((k, m, lo, hi, lo > 0 or hi < 0))
    return out


def tail_share(diff, k=5):
    """Fraction of the mean difference carried by the k most extreme rows.

    UNINTERPRETABLE when the mean is near zero -- the ratio then divides by
    noise and returns things like -864%.  metabric does exactly this.  Only
    read this number on a dataset whose difference is materially nonzero;
    otherwise fall back to max |d_i|, which needs no denominator.
    """
    total = diff.sum()
    if abs(total) < 1e-12:
        return float("nan")
    top = diff[np.argsort(-np.abs(diff))[:k]].sum()
    return float(top / total)


def main():
    args = sys.argv[1:]
    # --noclamp reproduces the OLD unbounded spline extrapolation, so the two
    # passes are an A/B of the kan.py fix on otherwise identical fits.
    if "--noclamp" in args:
        args.remove("--noclamp")
        S.CLAMP_INPUTS = False
    if "--nowins" in args:
        args.remove("--nowins")
        S.WINSORIZE = False
    for a in list(args):
        if a.startswith("--winsq="):
            args.remove(a)
            S.WINSOR_Q = float(a.split("=", 1)[1])
    print(f"### clamping: {S.CLAMP_INPUTS}   winsorising: {S.WINSORIZE} (q={S.WINSOR_Q:g})")
    names = args or ["rotgbsg"]
    avail = dict(D.LOADERS)
    for nm in names:
        raw = avail[nm]
        loader = lambda raw=raw: D.onehot_ordinals(raw())
        print("=" * 88)
        print(f"{nm}: paired bootstrap on 5 INDEPENDENT splits (fair baseline)")
        print("=" * 88)
        # S.run looks l1 up BY NAME, so the per-seed label must be registered
        # too -- otherwise every seed silently falls back to the l1=0 default
        # and we would be reporting an untuned model.
        rows = []
        for seed in range(5):
            label = f"{nm} s{seed}"
            S.SELECTED_L1[label] = FAIR_L1.get(nm, 0.01)
            r = S.run(label, loader, seed=seed)
            r["tail5"] = tail_share(_LAST["diff"])
            r["max_abs"] = float(np.abs(_LAST["diff"]).max())
            r["drop"] = drop_top_k(_LAST["diff"])
            print(f"    top-5 rows carry {r['tail5']*100:5.1f}% of the mean"
                  f"   max |d_i| = {r['max_abs']:.3f}")
            for k, m, lo, hi, sg in r["drop"]:
                print(f"      drop top-{k:<3d} NLL {m:+.5f} [{lo:+.5f}, {hi:+.5f}]"
                      f"  -> {'still resolved' if sg else 'GONE'}")
            rows.append(r)
        d = np.array([r["mean"] for r in rows])
        dc = np.array([r["c_kan"] - r["c_lin"] for r in rows])
        nsig = sum(r["sig"] == "SIGNIFICANT" for r in rows)
        ncsig = sum(r.get("c_sig") == "SIGNIFICANT" for r in rows)
        print()
        print(f"  NLL diff across splits : {d.mean():+.5f} +/- {d.std(ddof=1):.5f}"
              f"   min {d.min():+.5f}  max {d.max():+.5f}")
        print(f"  C   diff across splits : {dc.mean():+.4f} +/- {dc.std(ddof=1):.4f}"
              f"   min {dc.min():+.4f}  max {dc.max():+.4f}")
        print(f"  splits whose OWN 95% NLL CI excludes zero: {nsig}/5")
        print(f"  splits whose OWN 95% C-INDEX CI excludes zero: {ncsig}/5"
              f"   <- the headline metric")
        print(f"  sign consistency: {int((d > 0).sum())}/5 positive")
        # Broad-based means the effect survives deleting the worst rows.
        for k in (5, 20, 50):
            keep = [x for r in rows for x in r["drop"] if x[0] == k]
            if keep:
                surv = sum(x[4] for x in keep)
                print(f"  after dropping top-{k:<3d}: resolved on {surv}/{len(keep)} splits"
                      f"   (mean {np.mean([x[1] for x in keep]):+.5f})")
        t = np.array([r["tail5"] for r in rows])
        mx = np.array([r["max_abs"] for r in rows])
        # Only judge tail-dominance where the effect itself is resolvable;
        # otherwise the share divides by noise and says nothing.
        resolvable = nsig >= 3
        verdict = ("n/a -- effect not resolvable, share divides by noise"
                   if not resolvable else
                   "TAIL-DRIVEN, do not quote" if t.mean() > 0.5 else "broad-based")
        print(f"  top-5-row share of the mean: {t.mean()*100:.1f}% "
              f"(min {t.min()*100:.1f}%, max {t.max()*100:.1f}%)  -> {verdict}")
        print(f"  worst single-row |d_i| across splits: {mx.max():.2f}"
              f"   (>10 means one row can move a {rows[0]['n_test']}-row mean by "
              f"{mx.max()/rows[0]['n_test']:.4f})")


if __name__ == "__main__":
    main()
