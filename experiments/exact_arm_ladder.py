"""1c: where the exact-discrete method stops being computable on SPARCS.

The full drg302 cohort costs 1.06e9 recursion steps against a budget of 8e6 --
130x over -- so "run the exact method on SPARCS" is not a matter of waiting
longer.  Rather than fight that, this measures the boundary: fixed subsamples of
the SAME cohort at n = 500 .. 10000, each run through the identical four arms,
reporting for every rung whether the exact method was feasible, what it cost,
how long it took, and how far its coefficient sat from the grouped one.

The result is a practical contrast the paper can state plainly: exact tie
handling addresses the statistical approximation, and is unavailable at the
sample sizes where heavy ties actually occur.

Subsamples are nested by construction -- one permutation, prefixes of it -- so
the rungs are a sequence of growing datasets rather than unrelated draws, and
the trend is not confounded with resampling noise.

Run:  python -u experiments/exact_arm_ladder.py [drg] [T]
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
from experiments.crossover import coarsen
from experiments.exact_arm_real import compare
from experiments.protocol_decomp import (EXACT_BUDGET, TEST_FRAC, nb_se,
                                         one_split)
from experiments.real_data import split

OUT = Path(__file__).resolve().parent / "exact_arm_ladder.txt"
RUNGS = (500, 1000, 2000, 5000, 10000)
N_SPLITS = 20
SEED = 20260906


def take(base, ix):
    return SurvData(base.X[ix], base.bin_idx[ix], base.event[ix], base.n_bins,
                    base.feature_names, name=base.name,
                    intrinsically_discrete=base.intrinsically_discrete,
                    entry_idx=None if base.entry_idx is None
                    else base.entry_idx[ix],
                    bin_edges=base.bin_edges, meta=base.meta)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    drg = args[0] if args else "302"
    T = int(args[1]) if len(args) > 1 else 6

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 116)
    log("1c  SCALABILITY OF THE EXACT-DISCRETE METHOD ON SPARCS")
    log(f"  cohort sparcs/drg{drg} @ T={T}, nested subsamples, "
        f"{N_SPLITS} splits per rung")
    log(f"  cost = n_train x (max_t d_t + 1); budget = {EXACT_BUDGET:,.0f}")
    log("  R = ||beta_arm - beta_grouped|| / ||beta_grouped||, standardised.")
    log("=" * 116)

    try:
        base = D.load_sparcs(drg, horizon=30)
    except Exception as e:
        log(f"  LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
        return 1
    full_cost, full_dmax = CA.exact_discrete_cost(
        coarsen(base, T).bin_idx, base.event, T)
    log(f"  full cohort n={base.n:,}  exact cost {full_cost:,.0f} "
        f"(max_t d_t = {full_dmax:,})  =  "
        f"{full_cost / EXACT_BUDGET:,.0f}x the budget")
    log("")
    log(f"  {'n':>7}{'events':>8}{'modalEv':>9}{'dmax':>7}{'cost':>14}"
        f"  {'feas':<6}{'sec':>8}{'R Efron':>10}{'R exact':>10}"
        f"{'D_A':>10}{'D_B':>10}{'D_C':>10}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(base.n)
    blocks = []
    for n in RUNGS:
        if n > base.n:
            continue
        d = D.onehot_ordinals(coarsen(take(base, perm[:n]), T))
        ev = d.event.astype(bool)
        modal_ev = (float(np.bincount(d.bin_idx.astype(int)[ev]).max()
                          / ev.sum()) if ev.any() else float("nan"))
        cost, dmax = CA.exact_discrete_cost(d.bin_idx, d.event, d.n_bins)
        t0 = time.time()
        Re, Rx, dA, dB, dC, nfeas, nfail = [], [], [], [], [], 0, 0
        for s in range(N_SPLITS):
            # Heartbeat.  A configuration can run for half an hour, and a
            # script that prints only on completion is indistinguishable from a
            # hung one -- which is exactly how this looked from the outside on
            # the first long SPARCS config.  One line per split costs nothing.
            print(f"      n={n} split {s + 1}/{N_SPLITS} "
                  f"[{time.time() - t0:.0f}s elapsed]", flush=True)
            tr, te = split(d, frac=TEST_FRAC, seed=s)
            tr, te = D.clip_to_train_range(tr, te)
            trs, tes = standardize(tr, te)
            try:
                r, m = one_split(trs, tes, allow_exact=True)
            except Exception as e:
                nfail += 1
                if nfail <= 2:
                    log(f"    n={n} split {s}: FAILED {type(e).__name__}: "
                        f"{str(e)[:50]}")
                continue
            bg = m["beta_joint_cloglog"]
            ref = r["ours:joint-cloglog"]
            dA.append(float(np.mean(r["cox:efron+breslow"] - ref)))
            dB.append(float(np.mean(
                r["cox:efron+kalbfleisch-prentice"] - ref)))
            ce = compare(m["betas"]["efron"], bg)
            if ce:
                Re.append(ce["R"])
            if "exact-discrete" in m["betas"]:
                nfeas += 1
                cx = compare(m["betas"]["exact-discrete"], bg)
                if cx:
                    Rx.append(cx["R"])
                dC.append(float(np.mean(
                    r["cox:exact-discrete+kalbfleisch-prentice"] - ref)))
        el = time.time() - t0
        feas = cost <= EXACT_BUDGET
        b = {"n": int(n), "T": T, "events": int(ev.sum()),
             "modal_event": modal_ev, "dmax": int(dmax), "cost": float(cost),
             "feasible": bool(feas), "n_exact_splits": nfeas,
             "seconds": el, "failures": nfail,
             "R_efron": float(np.mean(Re)) if Re else None,
             "R_efron_se": float(nb_se(np.array(Re))) if len(Re) > 1 else None,
             "R_exact": float(np.mean(Rx)) if Rx else None,
             "R_exact_se": float(nb_se(np.array(Rx))) if len(Rx) > 1 else None,
             "D_A": float(np.mean(dA)) if dA else None,
             "D_B": float(np.mean(dB)) if dB else None,
             "D_C": float(np.mean(dC)) if dC else None}
        blocks.append(b)
        f = lambda v, w=10, p=5: (format(v, f">+{w}.{p}f") if v is not None
                                  else format("--", f">{w}"))
        g = lambda v, w=10, p=5: (format(v, f">{w}.{p}f") if v is not None
                                  else format("--", f">{w}"))
        log(f"  {n:>7}{int(ev.sum()):>8}{modal_ev:>9.4f}{dmax:>7}"
            f"{cost:>14,.0f}  {'yes' if feas else 'NO':<6}{el:>8.0f}"
            f"{g(b['R_efron'])}{g(b['R_exact'])}"
            f"{f(b['D_A'])}{f(b['D_B'])}{f(b['D_C'])}")
        OUT.with_suffix(".json").write_text(json.dumps(blocks, indent=1),
                                            encoding="utf-8")
        del d

    log("")
    log("=" * 116)
    log("READING")
    log("=" * 116)
    ok = [b for b in blocks if b["n_exact_splits"] > 0]
    if ok:
        top = max(b["n"] for b in ok)
        log(f"  exact-discrete ran at n <= {top:,}; the full cohort is "
            f"{base.n:,} rows, {full_cost / EXACT_BUDGET:,.0f}x over budget.")
        pairs = [b for b in ok if b["R_efron"] and b["R_exact"]]
        if pairs:
            n_x = sum(1 for b in pairs if b["R_exact"] < b["R_efron"])
            log(f"  exact is closer to the grouped coefficient on "
                f"{n_x} of {len(pairs)} rungs")
    else:
        log("  the exact method was infeasible at every rung tried")
    bad = [b for b in blocks if b["n_exact_splits"] == 0]
    if bad:
        log(f"  infeasible rungs: "
            f"{', '.join(f'n={b['n']:,}' for b in bad)}")
    log("")
    log("  wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
