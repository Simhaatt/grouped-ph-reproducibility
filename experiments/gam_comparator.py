"""Is the KAN doing anything a spline GAM does not?

THE OBJECTION THIS ANSWERS.  An additive KAN with cubic B-spline edge functions
is, structurally, a generalised additive model whose bases are learned rather
than fixed.  At a statistics journal that is the first question asked, and
"we used a KAN" is not an answer to it.  If a well-specified GAM matches the
KAN, then the KAN effect reported in section 8 is really an ADDITIVITY effect --
the gain comes from dropping linearity, not from the KAN -- and the paper must
say so.

THE COMPARATOR.  A GAM is built here by expanding each genuinely continuous
covariate into a fixed cubic B-spline basis (sklearn SplineTransformer, knots on
data quantiles) and then fitting the SAME discrete cloglog hazard model in
`linear` mode on the expanded design.  That construction is deliberate:

  - identical likelihood, identical optimiser, identical baseline logits, so the
    comparison isolates the basis and nothing else;
  - dummies and low-cardinality ordinals pass through untouched, exactly as they
    do for the KAN, so the fair-baseline correction of section 4.7 applies to
    every arm equally;
  - knots at quantiles, which is what a GAM user would do, rather than at the
    KAN's uniform grid -- the comparator should be the model a competent analyst
    would actually fit, not a handicapped version of it.

Three degrees of freedom are swept, because a GAM with too few knots is a straw
man and one with too many is a different estimator.

WHAT EACH OUTCOME MEANS.
  GAM ~ KAN      the honest finding is "flexibility helps, the architecture does
                 not".  Report it; it strengthens section 10, where the symbolic
                 surrogate already matched the KAN.
  KAN > GAM      the KAN earns its place; say by how much, with the MDE.
  GAM > KAN      report that too.  It would not damage the paper -- the
                 formulation results are independent of the index -- and
                 concealing it would.

Run:  python -u experiments/gam_comparator.py [dataset ...]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.fit import fit, to_tensors
from kanrel.hazard import DiscreteHazardKAN
from kanrel.likelihood import nll as nll_fn
from kanrel.metrics import evaluate, paired_c_bootstrap
from experiments.baselines import SELECTED_L1
from experiments.real_data import continuous_columns, split

SEEDS = (0, 1, 2, 3, 4)
GAM_DF = (4, 6, 8)          # spline degrees of freedom per continuous covariate


def spline_expand(tr, te, cont, n_knots):
    """Expand the continuous columns of BOTH splits on a basis fitted to TRAIN only.

    Fitting the basis on train alone matters: quantile knots read off the pooled
    data would leak the test distribution into the design, which is the same
    class of error as standardising before splitting.
    """
    if not cont:
        return tr, te
    st = SplineTransformer(n_knots=n_knots, degree=3, knots="quantile",
                           include_bias=False, extrapolation="constant")
    st.fit(tr.X[:, cont])

    def rebuild(d):
        keep = [j for j in range(d.X.shape[1]) if j not in set(cont)]
        Z = st.transform(d.X[:, cont]).astype(np.float32)
        X = np.hstack([d.X[:, keep], Z]).astype(np.float32)
        names = ([d.feature_names[j] for j in keep]
                 + [f"s{j}_{k}" for j in cont for k in range(Z.shape[1] // max(len(cont), 1))])
        names = names[:X.shape[1]] + [f"b{i}" for i in range(X.shape[1] - len(names))]
        return D.SurvData(X, d.bin_idx, d.event, d.n_bins, names, name=d.name,
                          intrinsically_discrete=d.intrinsically_discrete,
                          entry_idx=d.entry_idx, bin_edges=d.bin_edges, meta=d.meta)

    return rebuild(tr), rebuild(te)


def fit_score(tr, te, mode, l1=0.0, seed=0, cont_idx=None):
    m = DiscreteHazardKAN(tr.X.shape[1], tr.n_bins, mode=mode, link="cloglog",
                          hidden=(), grid_size=8,
                          cont_idx=cont_idx if mode == "baseline" else None)
    m, _ = fit(m, tr, epochs=400, lr=0.03, val_frac=0.2, patience=60,
               grid_update_epochs=(30, 100) if mode != "linear" else (),
               l1=l1, smooth=1e-3, seed=seed)
    X, mask, y = to_tensors(te)
    with torch.no_grad():
        loss = float(nll_fn(m(X), mask, y, m.link))
        S = m.survival(X).numpy()
    return evaluate(S, te.bin_idx, te.event, nll=loss), S


def run(name, loader):
    d = D.onehot_ordinals(loader())
    l1 = SELECTED_L1.get(name, 0.01)
    print("=" * 96)
    print(f"KAN vs SPLINE GAM -- {name}   n={d.n}  T={d.n_bins}  l1={l1:g}")
    print("=" * 96)

    arms = ["linear", "KAN"] + [f"GAM(df={k})" for k in GAM_DF]
    acc = {a: {"nll": [], "c": []} for a in arms}
    dc_kan_gam = {k: [] for k in GAM_DF}

    for seed in SEEDS:
        tr, te = split(d, seed=seed)
        tr, te = D.clip_to_train_range(tr, te)
        cont = continuous_columns(tr.X)

        r, _ = fit_score(tr, te, "linear", seed=seed)
        acc["linear"]["nll"].append(r["nll"]); acc["linear"]["c"].append(r["c_index"])

        rk, S_kan = fit_score(tr, te, "baseline", l1=l1, seed=seed, cont_idx=cont)
        acc["KAN"]["nll"].append(rk["nll"]); acc["KAN"]["c"].append(rk["c_index"])

        for k in GAM_DF:
            try:
                trg, teg = spline_expand(tr, te, cont, n_knots=k)
                rg, S_gam = fit_score(trg, teg, "linear", seed=seed)
            except Exception as e:
                print(f"    seed {seed} GAM(df={k}) FAILED "
                      f"{type(e).__name__}: {str(e)[:50]}")
                continue
            acc[f"GAM(df={k})"]["nll"].append(rg["nll"])
            acc[f"GAM(df={k})"]["c"].append(rg["c_index"])
            dc_kan_gam[k].append(rk["c_index"] - rg["c_index"])

    print(f"\n  {'model':<16}{'test NLL':>20}{'C-index':>20}   n_cont={len(cont)}")
    for a in arms:
        nl = np.array(acc[a]["nll"]); cc = np.array(acc[a]["c"])
        if not len(nl):
            print(f"  {a:<16}{'--':>20}"); continue
        print(f"  {a:<16}{nl.mean():>11.4f}+/-{nl.std(ddof=1):<8.4f}"
              f"{cc.mean():>11.4f}+/-{cc.std(ddof=1):<8.4f}")

    lin = np.array(acc["linear"]["c"]).mean()
    kan = np.array(acc["KAN"]["c"]).mean()
    print(f"\n  KAN gain over linear      dC = {kan - lin:+.4f}")
    best_k, best_c = None, -np.inf
    for k in GAM_DF:
        v = acc[f"GAM(df={k})"]["c"]
        if v and np.mean(v) > best_c:
            best_k, best_c = k, float(np.mean(v))
    if best_k is None:
        print("  no GAM arm completed")
        return
    print(f"  best GAM (df={best_k}) over linear  dC = {best_c - lin:+.4f}")

    diffs = np.array(dc_kan_gam[best_k])
    m, s = diffs.mean(), diffs.std(ddof=1)
    se = s / np.sqrt(len(diffs))
    print(f"\n  KAN - best GAM   dC = {m:+.4f} +/- {s:.4f}   (SE {se:.4f})")
    print(f"    per split: {', '.join(f'{x:+.4f}' for x in diffs)}")
    # 1.96, not 2.802.  2.802 = z_{0.975} + z_{0.80} is the multiplier for a
    # MINIMUM DETECTABLE EFFECT at 80% power, which answers "how large would a
    # difference have to be for this dataset to find it reliably".  It is not a
    # critical value, and using it to decide whether an observed difference is
    # non-zero makes the test conservative at roughly the 0.5% level rather than
    # the 5% level.  Two scripts here did exactly that.
    if abs(m) < 1.959964 * se:
        print("    -> NOT distinguishable from zero at this split count.")
        print("       The gain is an ADDITIVITY/flexibility effect, not a KAN effect.")
        print("       Report it that way; it is the honest reading and it agrees")
        print("       with section 10, where the closed-form surrogate also matched.")
    elif m > 0:
        print("    -> the KAN beats a well-specified spline GAM. Quote this against")
        print("       the dataset's own MDE before claiming it.")
    else:
        print("    -> the spline GAM BEATS the KAN. Report it; the formulation")
        print("       results in sections 5-6 do not depend on the index.")


def main():
    names = sys.argv[1:] or ["rotgbsg", "support-pycox", "metabric"]
    avail = dict(D.LOADERS)
    for nm in names:
        if nm not in avail:
            print(f"unknown dataset {nm!r}")
            continue
        run(nm, avail[nm])
        print()


if __name__ == "__main__":
    main()
