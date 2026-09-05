"""E1-E4: tie approximation, prediction protocol, three-arm decomposition, and
enough repeats to justify the standard errors.

WHAT WAS WRONG WITH THE OLD D_T.  The paper reported one number per grid: Efron
coefficients pushed through a Breslow baseline, minus the grouped cloglog joint
MLE.  Two things were invisible in that number.

  (a) The Breslow protocol makes the Cox arm's fitted hazard
      1 - exp(-dH0_t e^{b'x}), which is the grouped CLOGLOG hazard with
      alpha_t = log dH0_t.  Under that protocol the two arms are the same model
      and D_T measures ESTIMATION, not specification.  See cox_arms.py.
  (b) Neither Cox arm was the method actually designed for intrinsically discrete
      time.  That is the exact-discrete conditional partial likelihood, and it
      was simply absent.

This script sweeps both axes on every configuration and decomposes the result.

  E1  coefficients   Breslow ties | Efron ties | exact-discrete conditional PL
  E2  baseline       Breslow | Nelson-Aalen | Kalbfleisch-Prentice
  E3  three arms     (1) Cox beta + Breslow alpha
                     (2) Cox beta FIXED, alpha profiled by cloglog MLE
                     (3) joint MLE of (alpha, beta)
                     1->2 is the baseline representation, 2->3 is coefficient
                     estimation.  Section 5.4 asserted this split; here it is
                     measured.
  E4  repeats        N_SPLITS raised from 3 to 20, with BOTH the naive standard
                     error and the Nadeau-Bengio correction for overlapping
                     resamples, because 20 random 70/30 splits of one cohort are
                     not 20 independent experiments.

FEASIBILITY IS REPORTED, NOT HIDDEN.  The exact conditional likelihood costs
n x (max_t d_t + 1) sequential autograd steps, and its memory is the same product.
On the R2 cohorts -- which are exactly the ones the paper's claim lives on --
max_t d_t is a large fraction of n, so the cost is quadratic and the method is
genuinely infeasible.  Where that happens the run subsamples the TRAINING split
to the largest feasible size, refits EVERY arm on that identical subsample so the
comparison stays paired, and prints both the full-data and subsampled blocks.
An infeasible cell prints its cost; it is never dropped.

Run:  python -u experiments/protocol_decomp.py [cohort ...]
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.data import SurvData
from experiments import cox_arms as CA
from experiments.baselines import standardize
from experiments.crossover import COHORTS, coarsen
from experiments.real_data import split

N_SPLITS = 20
TEST_FRAC = 0.3
EXACT_BUDGET = 8.0e6          # n * (dmax+1); ~200 MB of autograd tape
OUT = Path(__file__).resolve().parent / "protocol_decomp.txt"

TIE_METHODS = ("breslow", "efron", "exact-discrete")
BASE_METHODS = ("breslow", "nelson-aalen", "kalbfleisch-prentice")


def nb_se(x, test_frac=TEST_FRAC):
    """Nadeau-Bengio standard error for repeated random subsampling.

    K overlapping 70/30 splits of one cohort share most of their training rows,
    so sd/sqrt(K) understates the uncertainty.  The correction inflates the
    variance by (1/K + n_te/n_tr):

        Var_hat = (1/K + f/(1-f)) * s^2 ,  f = test fraction.

    At K=20 and f=0.3 the multiplier is 0.05 + 0.4286 = 0.4786 against the naive
    0.05, so the honest interval is about 3.1x wider.  Several effects the paper
    currently reports as resolved are smaller than that.
    """
    x = np.asarray(x, float)
    K = len(x)
    if K < 2:
        return float("nan")
    s2 = float(np.var(x, ddof=1))
    return float(np.sqrt((1.0 / K + test_frac / (1.0 - test_frac)) * s2))


def subsample_for_exact(tr, budget=EXACT_BUDGET, seed=0):
    """Largest stratified subsample of the training split the exact method can run.

    Cost is n * (max_t d_t + 1) and d_t grows with n, so the cost is quadratic and
    halving n quarters it.  Bisecting on n rather than solving analytically keeps
    this correct whatever the event distribution looks like.
    """
    n = tr.n
    cost, _ = CA.exact_discrete_cost(tr.bin_idx, tr.event, tr.n_bins)
    if cost <= budget:
        return tr, n, cost
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    lo, hi = 50, n
    best = None
    for _ in range(24):
        mid = (lo + hi) // 2
        ix = perm[:mid]
        c, _ = CA.exact_discrete_cost(tr.bin_idx[ix], tr.event[ix], tr.n_bins)
        if c <= budget:
            best, lo = (ix, c), mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break
    if best is None:
        return None, 0, cost
    ix, c = best
    sub = SurvData(tr.X[ix], tr.bin_idx[ix], tr.event[ix], tr.n_bins,
                   tr.feature_names, name=tr.name,
                   intrinsically_discrete=tr.intrinsically_discrete,
                   entry_idx=None if tr.entry_idx is None else tr.entry_idx[ix],
                   bin_edges=tr.bin_edges, meta=tr.meta)
    return sub, len(ix), c


def one_split(tr, te, allow_exact=True):
    """Every arm on one split.  Returns per-unit NLL vectors keyed by arm name."""
    T = tr.n_bins
    Xtr, Xte = tr.X.astype(float), te.X.astype(float)
    out, betas, meta = {}, {}, {}

    betas["breslow"] = CA.fit_cox_ties(Xtr, tr.bin_idx, tr.event, "breslow")
    betas["efron"] = CA.fit_cox_ties(Xtr, tr.bin_idx, tr.event, "efron")
    if allow_exact:
        be, info = CA.fit_cox_exact_discrete(Xtr, tr.bin_idx, tr.event,
                                             budget=EXACT_BUDGET)
        meta["exact"] = info
        if be is not None:
            betas["exact-discrete"] = be
    else:
        meta["exact"] = {"feasible": False, "cost": float("nan")}

    # E1 x E2: every (coefficient, baseline) pair.
    for bm, b in betas.items():
        r_tr, r_te = CA.risk(Xtr, b), CA.risk(Xte, b)
        for base in BASE_METHODS:
            h = CA.BASELINES[base](r_tr, tr.bin_idx, tr.event, r_te, T)
            out[f"cox:{bm}+{base}"] = CA.nll_from_hazards(h, te.bin_idx, te.event, T)

    # E3 arm 2: Cox beta held fixed, alpha profiled by MLE.
    #
    # Both links are profiled.  The cloglog profile is arm 2 proper.  The logit
    # profile exists because the exact-discrete coefficient is a log ODDS ratio:
    # scoring it through a cloglog baseline pairs a coefficient from one model
    # with a hazard from another, and on metabric that mismatch alone cost it
    # +0.0022 against Efron.  Judging the exact method in its own metric is the
    # only way the E1 comparison is a comparison of tie handling rather than of
    # link choice.
    for bm, b in betas.items():
        a = CA.profile_alpha(Xtr, tr.bin_idx, tr.event, T, b, "cloglog")
        out[f"arm2:{bm}"] = CA.nll_from_hazards(
            CA.hazards_from_alpha(Xte, b, a), te.bin_idx, te.event, T)
        al_ = CA.profile_alpha(Xtr, tr.bin_idx, tr.event, T, b, "logit")
        out[f"arm2-logit:{bm}"] = CA.nll_from_hazards(
            CA.hazards_from_alpha(Xte, b, al_, "logit"), te.bin_idx, te.event, T)

    # E3 arm 3 / the reference: joint MLE.  Also the logit-link joint MLE, which
    # is the unconditional counterpart of the exact conditional likelihood.
    bj, aj = CA.fit_grouped_joint(Xtr, tr.bin_idx, tr.event, T, "cloglog")
    out["ours:joint-cloglog"] = CA.nll_from_hazards(
        CA.hazards_from_alpha(Xte, bj, aj), te.bin_idx, te.event, T)
    bl, al = CA.fit_grouped_joint(Xtr, tr.bin_idx, tr.event, T, "logit")
    out["ours:joint-logit"] = CA.nll_from_hazards(
        CA.hazards_from_alpha(Xte, bl, al, "logit"), te.bin_idx, te.event, T)

    meta["betas"] = {k: v.tolist() for k, v in betas.items()}
    meta["beta_joint_cloglog"] = bj.tolist()
    return out, meta


def run_config(name, d, T, log):
    """All N_SPLITS splits of one (cohort, grid) configuration."""
    modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
    cost_full, dmax = CA.exact_discrete_cost(d.bin_idx, d.event, d.n_bins)
    log(f"\n{'-'*100}")
    log(f"{name} @ T={d.n_bins}   n={d.n}  rows/bin={d.n/d.n_bins:.0f}  "
        f"modal bin mass={modal:.4f}")
    log(f"  exact-discrete cost on the full cohort = {cost_full:,.0f} "
        f"(max_t d_t = {dmax}); budget = {EXACT_BUDGET:,.0f} -> "
        f"{'FEASIBLE' if cost_full <= EXACT_BUDGET else 'INFEASIBLE, subsampling'}")

    full, sub = [], []
    n_sub_used, sub_cost = None, None
    t0 = time.time()
    for s in range(N_SPLITS):
        tr, te = split(d, frac=TEST_FRAC, seed=s)
        tr, te = D.clip_to_train_range(tr, te)
        trs, tes = standardize(tr, te)
        try:
            r, m = one_split(trs, tes, allow_exact=True)
            full.append(r)
        except Exception as e:
            log(f"    split {s}: FULL FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        if "cox:exact-discrete+breslow" not in r:
            # Exact was infeasible at full size: rerun EVERY arm on the largest
            # feasible subsample so the three tie methods stay paired.
            sub_tr, n_used, c = subsample_for_exact(trs, seed=1000 + s)
            if sub_tr is None:
                continue
            n_sub_used, sub_cost = n_used, c
            try:
                rs, _ = one_split(sub_tr, tes, allow_exact=True)
                sub.append(rs)
            except Exception as e:
                log(f"    split {s}: SUB FAILED {type(e).__name__}: {str(e)[:60]}")
    log(f"  {len(full)} full splits, {len(sub)} subsampled splits, "
        f"{time.time()-t0:.0f}s")
    return {"cohort": name, "T": int(d.n_bins), "n": int(d.n), "modal": modal,
            "cost_full": cost_full, "dmax": int(dmax),
            "n_sub": n_sub_used, "sub_cost": sub_cost,
            "full": full, "sub": sub}


def summarise(block, log):
    """D_T for every arm against the grouped cloglog joint MLE."""
    for tag, runs in (("FULL COHORT", block["full"]),
                      (f"SUBSAMPLE n_train={block['n_sub']}", block["sub"])):
        if not runs:
            continue
        keys = sorted(set().union(*[set(r) for r in runs]))
        ref = "ours:joint-cloglog"
        if ref not in keys:
            continue
        log(f"\n  {tag}  ({len(runs)} splits)   D_T = NLL(arm) - NLL(grouped cloglog)")
        log(f"    {'arm':<40}{'D_T':>11}{'naive SE':>11}{'N-B SE':>10}"
            f"{'|D|/NB':>9}   verdict")
        for k in keys:
            if k == ref:
                continue
            per = [float(np.mean(r[k] - r[ref])) for r in runs if k in r and ref in r]
            if len(per) < 2:
                continue
            a = np.array(per)
            se = float(a.std(ddof=1) / np.sqrt(len(a)))
            nb = nb_se(a)
            z = abs(a.mean()) / nb if nb > 0 else float("nan")
            verdict = ("grouped wins" if a.mean() > 0 else "arm wins") if z >= 1.96 \
                else "not resolved"
            log(f"    {k:<40}{a.mean():>+11.5f}{se:>11.5f}{nb:>10.5f}"
                f"{z:>9.2f}   {verdict}")

        # E1 proper: the COEFFICIENT effect alone.  alpha is profiled by MLE in
        # each arm, in the link that arm's coefficient belongs to, so nothing
        # here is contaminated by how the baseline was formed or by pairing an
        # odds ratio with a cloglog hazard.
        log(f"    {'-'*84}")
        log("    E1 coefficient effect, alpha profiled by MLE in the matching link")
        log(f"      {'tie method':<20}{'vs cloglog MLE':>16}{'NB SE':>10}"
            f"{'vs logit MLE':>15}{'NB SE':>10}")
        for bm in TIE_METHODS:
            row = [f"      {bm:<20}"]
            for pre, rf in (("arm2", "ours:joint-cloglog"),
                            ("arm2-logit", "ours:joint-logit")):
                k = f"{pre}:{bm}"
                if k in keys and rf in keys:
                    v = np.array([float(np.mean(r[k] - r[rf])) for r in runs
                                  if k in r and rf in r])
                    row.append(f"{v.mean():>+16.5f}{nb_se(v):>10.5f}"
                               if pre == "arm2" else
                               f"{v.mean():>+15.5f}{nb_se(v):>10.5f}")
                else:
                    row.append(f"{'infeasible':>16}{'':>10}" if pre == "arm2"
                               else f"{'infeasible':>15}{'':>10}")
            log("".join(row))

        # E3 decomposition, quoted on the Efron coefficients.
        if all(x in keys for x in ("cox:efron+breslow", "arm2:efron")):
            a1 = np.array([float(np.mean(r["cox:efron+breslow"] - r[ref])) for r in runs])
            a2 = np.array([float(np.mean(r["arm2:efron"] - r[ref])) for r in runs])
            log(f"    {'-'*84}")
            log(f"    E3 decomposition of D_T (Efron coefficients)")
            log(f"      baseline representation  arm1 -> arm2 = "
                f"{(a1-a2).mean():+.5f}  (NB SE {nb_se(a1-a2):.5f})")
            log(f"      coefficient estimation   arm2 -> arm3 = "
                f"{a2.mean():+.5f}  (NB SE {nb_se(a2):.5f})")
            log(f"      total                    arm1 -> arm3 = "
                f"{a1.mean():+.5f}  (NB SE {nb_se(a1):.5f})")
            tot = a1.mean()
            if abs(tot) > 1e-9:
                log(f"      share of D_T attributable to the baseline = "
                    f"{100*(a1-a2).mean()/tot:.1f}%")


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(COHORTS)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 100)
    log("E1-E4  TIE APPROXIMATION x PREDICTION PROTOCOL x THREE-ARM DECOMPOSITION")
    log(f"  {N_SPLITS} splits per configuration; test fraction {TEST_FRAC}")
    log("  D_T > 0 favours the grouped cloglog joint MLE.")
    log("  'N-B SE' is Nadeau-Bengio corrected for overlapping resamples;")
    log("  it is the standard error the resolution verdict uses.")
    log("=" * 100)

    blocks = []
    for name in want:
        if name not in COHORTS:
            log(f"unknown cohort {name!r}")
            continue
        loader, grid, _ = COHORTS[name]
        try:
            base = loader()
        except Exception as e:
            log(f"\n{name}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        # Match crossover.py exactly: coarsen the RAW cohort, then one-hot.
        # One-hotting first and coarsening after would expand the ordinal time
        # columns on the fine grid and then merge bins underneath them.
        for T in [t for t in grid if t <= base.n_bins]:
            d = D.onehot_ordinals(coarsen(base, T))
            try:
                b = run_config(name, d, T, log)
            except Exception as e:
                log(f"  {name}@T{T}: FAILED {type(e).__name__}: {str(e)[:70]}")
                continue
            summarise(b, log)
            blocks.append({k: v for k, v in b.items() if k not in ("full", "sub")})
            (OUT.with_suffix(".json")).write_text(json.dumps(blocks, indent=1),
                                                  encoding="utf-8")
    log("\nwrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
