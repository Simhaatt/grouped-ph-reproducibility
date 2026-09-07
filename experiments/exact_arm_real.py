"""1b + 3: the exact-discrete arm and the coefficient comparison on real data.

protocol_decomp.py already computes every arm the review asks for, including
cox:exact-discrete+kalbfleisch-prentice.  What it does NOT do is keep the
coefficients: run_config drops the per-split payload before the JSON is written,
so meta["betas"] never reaches disk and no coefficient comparison is possible
from the frozen run.  This script re-runs the same splits -- same seeds, same
standardisation, the same one_split() -- and persists beta.

For every split it stores, on the STANDARDISED scale (which is the scale the
review asks the comparison to be reported on):

    beta_breslow, beta_efron, beta_exact (where feasible), beta_grouped

and derives

    R_E      = ||beta_E     - beta_G|| / ||beta_G||
    R_exact  = ||beta_exact - beta_G|| / ||beta_G||
    cos_E, cos_exact                 cosine similarity with beta_G
    maxabs_E, maxabs_exact           largest single-coefficient discrepancy

PAIRING.  On cohorts where the exact method is infeasible at full size the
subsample pass refits EVERY arm on the identical subsample, so beta_exact is
never compared against a beta_G estimated from more data.  Configurations are
reported at both levels where both exist.

Run:  python -u experiments/exact_arm_real.py [cohort@T ...]
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
from experiments import cox_arms as CA
from experiments.baselines import standardize
from experiments.crossover import COHORTS, coarsen
from experiments.protocol_decomp import (EXACT_BUDGET, N_SPLITS, TEST_FRAC,
                                         nb_se, one_split, subsample_for_exact)
from experiments.real_data import split

OUT = Path(__file__).resolve().parent / "exact_arm_real.txt"

# The five configurations the review names, then the two fine-grid companions
# that show whether coefficient disagreement contracts as ties weaken.
TARGETS = [
    ("drsa/clinic", 5),
    ("support2/slos", 6),
    ("sparcs/drg302", 6),
    ("sparcs/drg302", 10),
    ("sparcs/drg302", 15),
    ("drsa/clinic", 50),
    ("sparcs/drg302", 30),
]


def compare(b, bg):
    """Discrepancy of one coefficient vector against the grouped joint MLE."""
    b, bg = np.asarray(b, float), np.asarray(bg, float)
    ng = np.linalg.norm(bg)
    if ng < 1e-12 or b.shape != bg.shape:
        return {}
    cos = float(b @ bg / max(np.linalg.norm(b) * ng, 1e-12))
    return {"R": float(np.linalg.norm(b - bg) / ng),
            "cos": cos,
            "maxabs": float(np.abs(b - bg).max())}


def summarise_level(tag, recs, log):
    """R_E and R_exact across the splits of one configuration at one size."""
    if not recs:
        return None
    out = {"level": tag, "splits": len(recs)}
    log(f"  {tag}  ({len(recs)} splits)")
    log(f"    {'arm':<16}{'R vs grouped':>14}{'NB SE':>9}"
        f"{'cosine':>9}{'max|dbeta|':>12}{'n':>5}")
    for arm, key in (("breslow", "breslow"), ("efron", "efron"),
                     ("exact-discrete", "exact-discrete")):
        vals = [r[key] for r in recs if key in r]
        if not vals:
            log(f"    {arm:<16}{'infeasible':>14}")
            out[key] = None
            continue
        R = np.array([v["R"] for v in vals])
        cs = np.array([v["cos"] for v in vals])
        mx = np.array([v["maxabs"] for v in vals])
        log(f"    {arm:<16}{R.mean():>14.5f}{nb_se(R):>9.5f}"
            f"{cs.mean():>9.5f}{mx.mean():>12.5f}{len(vals):>5}")
        out[key] = {"R": float(R.mean()), "R_se": float(nb_se(R)),
                    "cos": float(cs.mean()), "maxabs": float(mx.mean()),
                    "n": len(vals)}
    return out


def run_config(name, d, T, log):
    modal_all = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
    ev = d.event.astype(bool)
    modal_ev = (float(np.bincount(d.bin_idx.astype(int)[ev]).max() / ev.sum())
                if ev.any() else float("nan"))
    cost_full, dmax = CA.exact_discrete_cost(d.bin_idx, d.event, d.n_bins)
    log("")
    log("-" * 100)
    log(f"{name} @ T={T}   n={d.n}   modal event mass={modal_ev:.4f}   "
        f"modal all-exit mass={modal_all:.4f}")
    log(f"  exact cost = {cost_full:,.0f} (max_t d_t = {dmax}); "
        f"budget = {EXACT_BUDGET:,.0f} -> "
        f"{'FEASIBLE' if cost_full <= EXACT_BUDGET else 'INFEASIBLE, subsampling'}")

    full, sub = [], []
    n_sub_used, nfail = None, 0
    t0 = time.time()
    for s in range(N_SPLITS):
        tr, te = split(d, frac=TEST_FRAC, seed=s)
        tr, te = D.clip_to_train_range(tr, te)
        trs, tes = standardize(tr, te)
        try:
            r, m = one_split(trs, tes, allow_exact=True)
        except Exception as e:
            log(f"    split {s}: FULL FAILED {type(e).__name__}: {str(e)[:60]}")
            nfail += 1
            continue
        bg = m["beta_joint_cloglog"]
        rec = {k: compare(v, bg) for k, v in m["betas"].items()}
        rec["_beta_grouped"] = bg
        rec["_betas"] = m["betas"]
        full.append(rec)

        if "exact-discrete" not in m["betas"]:
            sub_tr, n_used, c = subsample_for_exact(trs, seed=1000 + s)
            if sub_tr is None:
                continue
            n_sub_used = n_used
            try:
                rs, ms = one_split(sub_tr, tes, allow_exact=True)
            except Exception as e:
                log(f"    split {s}: SUB FAILED {type(e).__name__}: "
                    f"{str(e)[:60]}")
                nfail += 1
                continue
            bgs = ms["beta_joint_cloglog"]
            recs = {k: compare(v, bgs) for k, v in ms["betas"].items()}
            recs["_beta_grouped"] = bgs
            recs["_betas"] = ms["betas"]
            sub.append(recs)

    el = time.time() - t0
    log(f"  {len(full)} full splits, {len(sub)} subsampled splits, "
        f"{nfail} failures, {el:.0f}s")
    blocks = []
    b = summarise_level("FULL COHORT", full, log)
    if b:
        blocks.append(b)
    if sub:
        b = summarise_level(f"SUBSAMPLE n_train={n_sub_used}", sub, log)
        if b:
            blocks.append(b)
    return {"cohort": name, "T": int(T), "n": int(d.n),
            "modal_event": modal_ev, "modal_all": modal_all,
            "cost_full": cost_full, "dmax": int(dmax),
            "n_sub": n_sub_used, "seconds": el, "failures": nfail,
            "levels": blocks,
            "betas_full": [r["_betas"] for r in full],
            "beta_grouped_full": [r["_beta_grouped"] for r in full],
            "betas_sub": [r["_betas"] for r in sub],
            "beta_grouped_sub": [r["_beta_grouped"] for r in sub]}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    want = TARGETS
    if args:
        want = []
        for a in args:
            c, _, t = a.partition("@")
            want.append((c, int(t)))

    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 100)
    log("1b + 3  EXACT-DISCRETE ARM AND COEFFICIENT COMPARISON ON REAL DATA")
    log(f"  {N_SPLITS} splits per configuration, seeds 0..{N_SPLITS - 1}, "
        f"identical to protocol_decomp.py")
    log("  R = ||beta_arm - beta_grouped|| / ||beta_grouped||, standardised "
        "scale.  0 means the arms agree.")
    log("  Modal EVENT mass is reported alongside modal all-exit mass; the "
        "former is the tie-severity measure.")
    log("=" * 100)

    blocks = []
    loaded = {}
    for name, T in want:
        if name not in COHORTS:
            log(f"unknown cohort {name!r}")
            continue
        if name not in loaded:
            try:
                loaded[name] = COHORTS[name][0]()
            except Exception as e:
                log(f"\n{name}: LOAD FAILED {type(e).__name__}: {str(e)[:70]}")
                loaded[name] = None
        base = loaded[name]
        if base is None or T > base.n_bins:
            continue
        d = D.onehot_ordinals(coarsen(base, T))
        try:
            blocks.append(run_config(name, d, T, log))
        except Exception as e:
            log(f"  {name}@T{T}: FAILED {type(e).__name__}: {str(e)[:70]}")
            continue
        OUT.with_suffix(".json").write_text(json.dumps(blocks, indent=1),
                                            encoding="utf-8")
        del d

    log("")
    log("=" * 100)
    log("READING")
    log("=" * 100)
    rows = []
    for b in blocks:
        for lv in b["levels"]:
            e, x = lv.get("efron"), lv.get("exact-discrete")
            if e and x:
                rows.append((b["cohort"], b["T"], lv["level"], e["R"], x["R"]))
    if rows:
        log(f"  {'cohort':<16}{'T':>4}  {'level':<26}{'R Efron':>10}"
            f"{'R exact':>10}   closer to grouped")
        for c, T, lv, re_, rx in rows:
            log(f"  {c:<16}{T:>4}  {lv:<26}{re_:>10.5f}{rx:>10.5f}   "
                f"{'exact' if rx < re_ else 'Efron'}")
        n_ex = sum(1 for r in rows if r[4] < r[3])
        log("")
        log(f"  exact is closer to the grouped coefficient in {n_ex} of "
            f"{len(rows)} paired comparisons")
    else:
        log("  no configuration produced both an Efron and an exact "
            "coefficient at the same size")
    log("")
    log("  wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
