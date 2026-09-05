"""E19-E24: the robustness checks, and the reproducibility record.

  E19  drsa/music subsample stability at 10k / 25k / 50k / full.  Section 6 quotes
       a 50k subsample without showing that the number it reports is a property of
       the cohort rather than of the subsample size.
  E20  (A7) violations per cohort.  (A7) requires entry and censoring to fall on
       grid boundaries.  Nothing in the project has ever counted how often that
       fails, so the assumption is stated and never audited.  Where it fails, a
       partial-interval likelihood -- crediting a subject only the fraction of the
       interval it was actually at risk -- is fitted as a sensitivity analysis.
  E21  Grid construction: equal-width against equal-count (quantile) against the
       native discrete scale.  Modal bin mass is the paper's main ordering
       variable and it is highly sensitive to which of the three is used, so a
       result stated in terms of modal mass is only as portable as the gridding
       convention behind it.
  E22  Additional R2 cohorts.  Reports which further intrinsically discrete
       cohorts the DRSA archive can supply, since four is thin support for a claim
       about cohort-specific crossovers.
  E23  The formulation sweep repeated with SPLINE covariate effects in BOTH arms,
       to show D_T is not an artefact of the linear index.
  E24  Seeds, software versions and wall-clock, recorded rather than promised.

Run:  python -u experiments/robustness_suite.py [e19 e20 e21 e22 e23 e24]
"""
from __future__ import annotations

import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.likelihood import make_targets, nll as nll_fn
from experiments import cox_arms as CA
from experiments.baselines import standardize
from experiments.crossover import COHORTS, coarsen
from experiments.protocol_decomp import nb_se
from experiments.real_data import continuous_columns, split

N_SPLITS = int(os.environ.get("ROB_SPLITS", "20"))
OUT = Path(__file__).resolve().parent / "robustness_suite.txt"


def d_t(d, seeds, design=None):
    """D_T = NLL(Cox/Efron + Breslow) - NLL(grouped joint MLE), over seeds."""
    vals = []
    for s in seeds:
        tr, te = split(d, frac=0.3, seed=s)
        tr, te = D.clip_to_train_range(tr, te)
        trs, tes = standardize(tr, te)
        Xtr, Xte = trs.X.astype(float), tes.X.astype(float)
        if design is not None:
            got = design(trs, tes)
            if got is None:
                # A design function signals "this arm does not exist on this
                # cohort" by returning None.  Raising here rather than unpacking
                # None keeps the failure legible; the caller is expected to check
                # first and print a reason.
                raise ValueError("design arm is degenerate on this cohort")
            Xtr, Xte = got
        try:
            b = CA.fit_cox_ties(Xtr, trs.bin_idx, trs.event, "efron")
            h = CA.hazards_breslow(CA.risk(Xtr, b), trs.bin_idx, trs.event,
                                   CA.risk(Xte, b), d.n_bins)
            n1 = CA.nll_from_hazards(h, tes.bin_idx, tes.event, d.n_bins)
            bj, aj = CA.fit_grouped_joint(Xtr, trs.bin_idx, trs.event, d.n_bins)
            n3 = CA.nll_from_hazards(CA.hazards_from_alpha(Xte, bj, aj),
                                     tes.bin_idx, tes.event, d.n_bins)
            vals.append(float(np.mean(n1 - n3)))
        except Exception:
            pass
    return np.array(vals)


def degenerate(*designs, names=None):
    """-> list of (i, j) index pairs whose designs are byte-identical.

    THREE TIMES in this project a comparison arm has silently collapsed onto
    another and printed identical numbers that read as agreement:

      nwtco GAM+pairs   +0.00000 +- 0.00000 on all five metrics, because the
                        cohort has fewer than two continuous columns
      sparcs E23        +0.29309 +- 0.00203 in BOTH the linear and spline
                        columns, because the cohort has no continuous column
      E21               'native' and 'equal-width' identical on every row,
                        because a native day grid already IS equal-width

    Each was fixed individually.  This is the general check: compare the arms
    before reporting them, and print "identical" rather than a number that
    invites the reader to conclude the arms agree.
    """
    out = []
    for i in range(len(designs)):
        for j in range(i + 1, len(designs)):
            a, b = designs[i], designs[j]
            if a is None or b is None:
                continue
            if a.shape == b.shape and np.array_equal(a, b):
                out.append((i, j))
    return out


# ------------------------------------------------------------------- E19
def e19(log):
    log("=" * 100)
    log("E19  drsa/music SUBSAMPLE STABILITY")
    log("     Section 6 quotes a 50k subsample.  Is D_T a property of the cohort")
    log("     or of the subsample size?")
    log("=" * 100)
    log(f"  {'n':>9}{'T':>5}{'modal':>8}{'D_T':>11}{'NB SE':>10}{'secs':>8}")
    for n in (10000, 25000, 50000, None):
        try:
            base = D.load_drsa("MUSIC", max_rows=n) if n else D.load_drsa("MUSIC")
        except Exception as e:
            log(f"  n={n}: LOAD FAILED {type(e).__name__}: {str(e)[:50]}")
            continue
        for T in (6, 12, 30):
            t0 = time.time()
            d = D.onehot_ordinals(coarsen(base, T))
            v = d_t(d, range(min(N_SPLITS, 8)))
            if v.size < 2:
                continue
            modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
            log(f"  {d.n:>9}{T:>5}{modal:>8.4f}{v.mean():>+11.5f}"
                f"{nb_se(v):>10.5f}{time.time()-t0:>8.0f}")
    log("")


# ------------------------------------------------------------------- E20
def fit_partial_interval(X, tidx, event, T, frac):
    """Grouped cloglog MLE crediting only `frac` of the exit interval.

    (A7) assumes entry and censoring land on grid boundaries.  When a subject
    leaves part-way through its last interval, the standard likelihood still
    charges it a full interval of exposure.  The partial-interval likelihood
    replaces the exit term's cumulative hazard by frac_i * exp(eta), which is the
    correct exposure under a piecewise-constant hazard, and is available only for
    the cloglog link -- which is itself an argument for the link.
    """
    Xt = torch.as_tensor(np.asarray(X, float), dtype=torch.float64)
    mask, y = make_targets(tidx, event, T)
    mask, y = mask.double(), y.double()
    fr = torch.as_tensor(np.asarray(frac, float), dtype=torch.float64)
    w = mask.clone()
    idx = torch.as_tensor(np.asarray(tidx).astype(int), dtype=torch.long)
    w[torch.arange(len(idx)), idx] = fr                   # partial exposure
    th = torch.zeros(T + Xt.shape[1], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        th[:T] = -2.5
    opt = torch.optim.LBFGS([th], max_iter=300, tolerance_grad=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = th[:T].unsqueeze(0) + (Xt @ th[T:]).unsqueeze(1)
        u = torch.exp(eta.clamp(-30, 10)) * w              # exposure-scaled
        log_1mh = -u
        log_h = torch.log(-torch.expm1(-u.clamp_min(1e-12))).clamp_min(-60.0)
        ll = (y * log_h + (1 - y) * log_1mh) * mask
        loss = -(ll.sum())
        loss.backward()
        return loss

    opt.step(closure)
    out = th.detach().numpy()
    return out[T:], out[:T]


def e20(log):
    log("=" * 100)
    log("E20  (A7) VIOLATIONS PER COHORT, AND SENSITIVITY TO THEM")
    log("     (A7) wants entry and censoring on grid boundaries.  Counted here,")
    log("     then a partial-interval likelihood is fitted as a sensitivity check.")
    log("=" * 100)
    log("     'off-grid <=' is an UPPER BOUND, not a count: on a binned-continuous")
    log("     cohort every censored subject MAY have left mid-interval, and the")
    log("     underlying continuous time is not retained to check.  On a natively")
    log("     discrete cohort it is exactly zero.")
    log(f"  {'cohort':<18}{'T':>5}{'n':>8}{'left-trunc':>12}{'censored':>10}"
        f"{'off-grid <=':>13}{'D_T std':>10}{'D_T partial':>13}")
    for name, (loader, grid, _) in COHORTS.items():
        try:
            base = loader()
        except Exception:
            continue
        T0 = base.n_bins
        d = D.onehot_ordinals(base)
        n_lt = 0 if d.entry_idx is None else int((np.asarray(d.entry_idx) > 0).sum())
        n_cen = int((d.event == 0).sum())
        # A cohort recorded on its native discrete scale cannot be off-grid; a
        # binned-continuous one is off-grid for every censored subject whose true
        # time fell inside a bin, which is all of them unless censoring is
        # administrative at a boundary.
        offgrid = 0.0 if d.intrinsically_discrete else 100.0 * n_cen / d.n
        v_std = d_t(d, range(min(N_SPLITS, 6)))
        # Partial-interval: without the underlying continuous time we can only
        # bound the effect, so the exit interval is credited a half on average,
        # which is the expectation under a uniform within-bin exit.
        vals = []
        for s in range(min(N_SPLITS, 6)):
            tr, te = split(d, frac=0.3, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            trs, tes = standardize(tr, te)
            frac = np.where(trs.event == 0, 0.5, 1.0)
            try:
                b = CA.fit_cox_ties(trs.X.astype(float), trs.bin_idx, trs.event,
                                    "efron")
                h = CA.hazards_breslow(CA.risk(trs.X.astype(float), b),
                                       trs.bin_idx, trs.event,
                                       CA.risk(tes.X.astype(float), b), T0)
                n1 = CA.nll_from_hazards(h, tes.bin_idx, tes.event, T0)
                bj, aj = fit_partial_interval(trs.X.astype(float), trs.bin_idx,
                                              trs.event, T0, frac)
                n3 = CA.nll_from_hazards(
                    CA.hazards_from_alpha(tes.X.astype(float), bj, aj),
                    tes.bin_idx, tes.event, T0)
                vals.append(float(np.mean(n1 - n3)))
            except Exception:
                pass
        vp = np.array(vals)
        log(f"  {name:<18}{T0:>5}{d.n:>8}{n_lt:>12}{n_cen:>10}{offgrid:>13.1f}"
            f"{(v_std.mean() if v_std.size else np.nan):>+10.5f}"
            f"{(vp.mean() if vp.size else np.nan):>+13.5f}")
    log("")
    log("  A large gap between the last two columns means the (A7) idealisation")
    log("  is doing work the paper has not accounted for.")
    log("")


# ------------------------------------------------------------------- E21
def regrid(d, T, how):
    """Rebuild the grid: equal-width, equal-count, or the native discrete scale."""
    idx = np.asarray(d.bin_idx).astype(int)
    if how == "native":
        return coarsen(d, T)
    if how == "equal-width":
        e = np.linspace(0, d.n_bins, T + 1)
        new = np.clip(np.searchsorted(e, idx, side="right") - 1, 0, T - 1)
    else:                                   # equal-count on the observed times
        q = np.quantile(idx, np.linspace(0, 1, T + 1))
        q[0] -= 0.5
        new = np.clip(np.searchsorted(q, idx, side="left") - 1, 0, T - 1)
    ent = None if d.entry_idx is None else np.zeros_like(new)
    return D.SurvData(d.X, new.astype(idx.dtype), d.event, T, d.feature_names,
                      name=d.name, intrinsically_discrete=d.intrinsically_discrete,
                      entry_idx=ent, bin_edges=None, meta=dict(d.meta))


def e21(log):
    log("=" * 100)
    log("E21  GRID CONSTRUCTION: equal-width vs equal-count vs native")
    log("     Modal bin mass is the paper's ordering variable and it is highly")
    log("     sensitive to which convention produced it.")
    log("=" * 100)
    log(f"  {'cohort':<18}{'T':>5}" +
        "".join(f"{h:>26}" for h in ("native", "equal-width", "equal-count")))
    log(f"  {'':<18}{'':>5}" + "".join(f"{'modal':>12}{'D_T':>14}"
                                       for _ in range(3)))
    for name, (loader, grid, _) in COHORTS.items():
        try:
            base = loader()
        except Exception:
            continue
        for T in grid[:3]:
            if T > base.n_bins:
                continue
            grids, cells = [], []
            for how in ("native", "equal-width", "equal-count"):
                try:
                    d = D.onehot_ordinals(regrid(base, T, how))
                    grids.append(np.asarray(d.bin_idx).astype(int))
                    v = d_t(d, range(min(N_SPLITS, 6)))
                    modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
                    cells.append(f"{modal:>12.4f}{v.mean():>+14.5f}"
                                 if v.size else f"{modal:>12.4f}{'--':>14}")
                except Exception:
                    grids.append(None)
                    cells.append(f"{'--':>12}{'--':>14}")
            # Flag any two conventions that produced the SAME grid, so identical
            # columns are never read as two conventions agreeing.
            dup = degenerate(*grids)
            labs = ("native", "equal-width", "equal-count")
            tag = ("" if not dup else
                   "   IDENTICAL: " + ", ".join(f"{labs[i]}=={labs[j]}"
                                                for i, j in dup))
            log(f"  {name:<18}{T:>5}" + "".join(cells) + tag)
    log("")


# ------------------------------------------------------------------- E22
def e22(log, n_new=4):
    log("=" * 100)
    log("E22  FURTHER INTRINSICALLY DISCRETE (R2) COHORTS")
    log("     Four cohorts is thin support for a claim about cohort-specific")
    log("     crossovers, so this looks for more and FITS them rather than")
    log("     listing what might be possible.")
    log("=" * 100)
    log("  DRSA archive:")
    for sp in ("CLINIC", "MUSIC"):
        try:
            d = D.load_drsa(sp, max_rows=4000)
            log(f"    {sp:<10} available   T={d.n_bins:<5} discrete="
                f"{d.intrinsically_discrete}")
        except Exception as e:
            log(f"    {sp:<10} unavailable ({type(e).__name__})")
    log("    The archive ships exactly these two splits; there is no third.")
    log("")
    log("  SPARCS: every APR-DRG is a distinct patient population measured on the")
    log("  same whole-day scale, so each viable code is a NEW R2 cohort rather")
    log("  than a re-slice of an old one.  Scanning for the largest.")
    try:
        import pandas as pd
        from kanrel.data import SPARCS_FILE, _find
        codes = pd.read_csv(_find(SPARCS_FILE), usecols=["APR DRG Code"],
                            dtype=str, low_memory=False)["APR DRG Code"]
        counts = (codes.astype(str).str.strip().str.zfill(3)
                  .value_counts().head(12))
    except Exception as e:
        log(f"    SPARCS scan FAILED {type(e).__name__}: {str(e)[:60]}")
        return
    log(f"    twelve largest APR-DRGs: "
        + ", ".join(f"{c}({n:,})" for c, n in counts.items()))
    log("")
    log(f"  {'cohort':<16}{'n':>8}{'T':>5}{'modal':>8}{'D_T':>11}{'NB SE':>10}"
        f"   verdict")
    done = 0
    for code in counts.index:
        if code == "302" or done >= n_new:
            continue                       # 302 is already in the paper
        try:
            base = D.load_sparcs(code, horizon=30)
            d = D.onehot_ordinals(base)
        except Exception as e:
            log(f"  drg{code:<12} skipped ({type(e).__name__}: {str(e)[:40]})")
            continue
        v = d_t(d, range(min(N_SPLITS, 6)))
        if v.size < 2:
            log(f"  drg{code:<12} no split completed")
            continue
        modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
        se = nb_se(v)
        verdict = ("discrete wins" if v.mean() > 1.96 * se else
                   "COX WINS" if v.mean() < -1.96 * se else "not resolved")
        log(f"  sparcs/drg{code:<6}{d.n:>8}{d.n_bins:>5}{modal:>8.4f}"
            f"{v.mean():>+11.5f}{se:>10.5f}   {verdict}")
        done += 1
    log("")
    log("  These are out-of-sample tests of the modal-mass rule: the rule was")
    log("  read off drg302 and the other cohorts, and these were not used to")
    log("  form it.")
    log("")


# ------------------------------------------------------------------- E23
def e23(log):
    log("=" * 100)
    log("E23  THE FORMULATION SWEEP WITH SPLINE COVARIATE EFFECTS IN BOTH ARMS")
    log("     If D_T survives dropping linearity, it is not an artefact of the")
    log("     linear index.")
    log("=" * 100)
    from sklearn.preprocessing import SplineTransformer

    def spline_design(trs, tes):
        cont = continuous_columns(trs.X)
        if not cont:
            # No continuous column means the spline design IS the linear design,
            # and the two arms print byte-identical numbers -- sparcs/drg302 gave
            # +0.29309 +- 0.00203 in both columns, which reads as "the spline arm
            # agrees" when it means the spline arm never existed.  Signalled by
            # returning None so the caller can say so.
            return None
        st = SplineTransformer(n_knots=4, degree=3, knots="quantile",
                               include_bias=False, extrapolation="constant")
        st.fit(trs.X[:, cont])
        keep = [j for j in range(trs.X.shape[1]) if j not in set(cont)]
        return (np.hstack([trs.X[:, keep], st.transform(trs.X[:, cont])]).astype(float),
                np.hstack([tes.X[:, keep], st.transform(tes.X[:, cont])]).astype(float))

    log(f"  {'cohort':<18}{'T':>5}{'modal':>8}{'D_T linear':>13}{'NB SE':>9}"
        f"{'D_T spline':>13}{'NB SE':>9}   agree?")
    for name, (loader, grid, _) in COHORTS.items():
        try:
            base = loader()
        except Exception:
            continue
        for T in grid[:3]:
            if T > base.n_bins:
                continue
            d = D.onehot_ordinals(coarsen(base, T))
            modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
            vl = d_t(d, range(min(N_SPLITS, 8)))
            if not continuous_columns(d.X):
                log(f"  {name:<18}{T:>5}{modal:>8.4f}{vl.mean():>+13.5f}"
                    f"{nb_se(vl):>9.5f}{'n/a':>13}{'':>9}   NO CONTINUOUS COLUMN")
                continue
            vs = d_t(d, range(min(N_SPLITS, 8)), design=spline_design)
            if vl.size < 2 or vs.size < 2:
                continue
            # Sign agreement is too weak a criterion: flchain came out +0.00090
            # linear against +0.09925 spline and was reported as "yes" because
            # both are positive.  Require the two to be within each other's
            # Nadeau-Bengio intervals as well.
            sl, ss = nb_se(vl), nb_se(vs)
            gap = abs(vl.mean() - vs.mean())
            if np.sign(vl.mean()) != np.sign(vs.mean()):
                same = "SIGN FLIP"
            elif gap > 1.959964 * np.sqrt(sl ** 2 + ss ** 2):
                same = "SAME SIGN, DIFFERENT SIZE"
            else:
                same = "yes"
            log(f"  {name:<18}{T:>5}{modal:>8.4f}{vl.mean():>+13.5f}"
                f"{sl:>9.5f}{vs.mean():>+13.5f}{ss:>9.5f}   {same}")
    log("")


# ------------------------------------------------------------------- E24
def e24(log):
    log("=" * 100)
    log("E24  REPRODUCIBILITY RECORD")
    log("=" * 100)
    import sklearn
    log(f"  python        {sys.version.split()[0]}")
    log(f"  platform      {platform.platform()}")
    log(f"  processor     {platform.processor()}")
    log(f"  numpy         {np.__version__}")
    log(f"  torch         {torch.__version__}  threads={torch.get_num_threads()}")
    log(f"  scikit-learn  {sklearn.__version__}")
    log("")
    log("  SEEDS.  Every split is numpy default_rng(seed) with seed = 0..K-1 in")
    log("  experiments/real_data.py::split; hyperparameter selection uses")
    log("  1000 + seed so that a validation split never coincides with a test")
    log("  split.  Simulations seed from hash((cell..., rep)) so that two cells")
    log("  never share a stream.  Torch is seeded inside kanrel.fit.fit.")
    log("")
    log("  SPLIT COUNTS.  protocol_decomp and nuisance_dim use 20 splits;")
    log("  flexibility uses 20 seeds; simulations use 20 replications per cell.")
    log("  All standard errors labelled 'NB SE' carry the Nadeau-Bengio")
    log("  correction for overlapping resamples and are the ones the resolution")
    log("  verdicts use.")
    log("")


def main():
    which = [a.lower() for a in sys.argv[1:]] or ["e24", "e22", "e20", "e21",
                                                  "e23", "e19"]
    lines = []
    path = (Path(__file__).resolve().parent /
            ("robustness_" + "".join(which) + ".txt"))

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    t0 = time.time()
    for k in which:
        try:
            {"e19": e19, "e20": e20, "e21": e21, "e22": e22, "e23": e23,
             "e24": e24}[k](log)
        except Exception as e:
            log(f"{k}: FAILED {type(e).__name__}: {str(e)[:80]}")
    log(f"total {time.time()-t0:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
