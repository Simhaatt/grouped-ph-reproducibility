"""2: the out-of-sample SPARCS validation, rerun to match the main experiment.

The first version of this test had two inconsistencies with the experiment it is
supposed to validate, both avoidable and both reviewer-visible:

  1. it used 10 splits where protocol_decomp.py uses 20, so its standard errors
     are not on the same footing as the ones the claim rests on;
  2. it reported modal ALL-EXIT mass -- the largest share of subjects leaving in
     any interval, censored or not -- where the quantity the argument is about
     is modal EVENT mass, the largest share of FAILURES in any interval.  Tie
     severity is a property of the failures; censored rows break no ties.

This rerun fixes both and stores the diagnostics that let a reader check the
grid rather than take it on trust: how many intervals survived coarsening, how
many contain any event at all, and the largest event count in a single interval.

Everything else is deliberately unchanged from kp_out_of_sample.py -- same four
APR-DRG cohorts, same MAX_ROWS cap with the same seed, same coarsening, same
Nadeau-Bengio correction -- so the difference between the two logs is the two
fixes and the extra splits, nothing else.

Run:  python -u experiments/kp_out_of_sample_v2.py [drg ...]
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
from experiments.crossover import coarsen
from experiments.kp_out_of_sample import MAX_ROWS, NEW_DRGS, one
from experiments.protocol_decomp import nb_se

N_SPLITS = 20                      # matches protocol_decomp.py
GRIDS = (6, 15, 30)
OUT = Path(__file__).resolve().parent / "kp_out_of_sample_v2.txt"


def diagnostics(d):
    """Grid diagnostics a reader can check the configuration against."""
    idx = d.bin_idx.astype(int)
    ev = d.event.astype(bool)
    n_ev = int(ev.sum())
    cnt_all = np.bincount(idx, minlength=d.n_bins)
    cnt_ev = np.bincount(idx[ev], minlength=d.n_bins) if n_ev else cnt_all * 0
    return {
        "n": int(d.n),
        "events": n_ev,
        "T_retained": int(d.n_bins),
        "intervals_with_events": int((cnt_ev > 0).sum()),
        "intervals_occupied": int((cnt_all > 0).sum()),
        "max_events_in_interval": int(cnt_ev.max()) if n_ev else 0,
        "modal_event_mass": float(cnt_ev.max() / n_ev) if n_ev else float("nan"),
        "modal_all_exit_mass": float(cnt_all.max() / d.n),
    }


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(NEW_DRGS)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 120)
    log("2  OUT-OF-SAMPLE TEST OF THE KP-NULL -- 20 SPLITS, MODAL EVENT MASS")
    log("  New SPARCS APR-DRG cohorts, none used to form any claim in the paper.")
    log(f"  {N_SPLITS} splits per configuration, matching protocol_decomp.py; "
        f"NB SE is Nadeau-Bengio corrected.")
    log(f"  Each cohort capped at {MAX_ROWS:,} rows (see MAX_ROWS).")
    log("  'modalEv' is the largest share of FAILURES in one interval -- the "
        "tie-severity measure.")
    log("  'modalAll' is the largest share of all exits, which the first "
        "version reported by mistake.")
    log("  D_T > 0 favours the grouped joint MLE.  'res' is |D_T| > 1.96 NB SE.")
    log("=" * 120)
    log(f"  {'cohort':<18}{'n':>7}{'ev':>7}{'T':>4}{'Tev':>5}{'maxEv':>7}"
        f"{'modalEv':>9}{'modalAll':>9}"
        f"{'D_T Breslow':>13}{'NB SE':>9}{'res':>5}"
        f"{'D_T KP':>11}{'NB SE':>9}{'res':>5}{'base%':>8}")

    blocks = []
    n_br, n_kp_pos, n_tot = 0, 0, 0
    for drg in want:
        try:
            base = D.load_sparcs(drg, horizon=30)
            if base.n > MAX_ROWS:
                rng = np.random.default_rng(20260904)   # same seed as v1
                ix = rng.choice(base.n, MAX_ROWS, replace=False)
                base = D.SurvData(
                    base.X[ix], base.bin_idx[ix], base.event[ix], base.n_bins,
                    base.feature_names, name=base.name,
                    intrinsically_discrete=base.intrinsically_discrete,
                    entry_idx=None if base.entry_idx is None
                    else base.entry_idx[ix],
                    bin_edges=base.bin_edges, meta=base.meta)
        except Exception as e:
            log(f"  drg{drg:<14} LOAD FAILED {type(e).__name__}: {str(e)[:40]}")
            continue
        for T in GRIDS:
            if T > base.n_bins:
                continue
            t0 = time.time()
            d = D.onehot_ordinals(coarsen(base, T))
            dg = diagnostics(d)
            # Split-at-a-time rather than one(d, range(N_SPLITS)) purely so the
            # run reports progress: each split is independent and seeded by s,
            # so concatenating per-split calls is identical to one batched call.
            # Without this a configuration is silent for many minutes and looks
            # hung, which is how the first long SPARCS config appeared.
            brs, kps, shs = [], [], []
            for s in range(N_SPLITS):
                print(f"      drg{drg}@T{T} split {s + 1}/{N_SPLITS} "
                      f"[{time.time() - t0:.0f}s elapsed]", flush=True)
                b1, k1, s1 = one(d, [s])
                brs.append(b1)
                kps.append(k1)
                shs.append(s1)
            br = np.concatenate(brs) if brs else np.array([])
            kp = np.concatenate(kps) if kps else np.array([])
            sh = np.concatenate(shs) if shs else np.array([])
            if br.size < 2 or kp.size < 2:
                log(f"  sparcs/drg{drg}@T{T}  no split completed")
                del d
                continue
            sb, sk = nb_se(br), nb_se(kp)
            rb = "yes" if abs(br.mean()) > 1.959964 * sb else "no"
            rk = "yes" if abs(kp.mean()) > 1.959964 * sk else "no"
            n_tot += 1
            n_br += (br.mean() > 0 and rb == "yes")
            n_kp_pos += (kp.mean() > 0 and rk == "yes")
            b = dict(dg)
            b.update({"cohort": f"sparcs/drg{drg}", "T_requested": T,
                      "splits": int(br.size),
                      "D_breslow": float(br.mean()), "D_breslow_se": float(sb),
                      "resolved_breslow": rb,
                      "D_kp": float(kp.mean()), "D_kp_se": float(sk),
                      "resolved_kp": rk,
                      "baseline_share": (float(sh.mean()) if sh.size
                                         else None),
                      "seconds": time.time() - t0})
            blocks.append(b)
            log(f"  {'sparcs/drg' + drg + '@T' + str(T):<18}{dg['n']:>7}"
                f"{dg['events']:>7}{dg['T_retained']:>4}"
                f"{dg['intervals_with_events']:>5}"
                f"{dg['max_events_in_interval']:>7}"
                f"{dg['modal_event_mass']:>9.4f}"
                f"{dg['modal_all_exit_mass']:>9.4f}"
                f"{br.mean():>+13.5f}{sb:>9.5f}{rb:>5}"
                f"{kp.mean():>+11.5f}{sk:>9.5f}{rk:>5}"
                f"{(f'{sh.mean():.1f}%' if sh.size else '--'):>8}"
                f"   [{time.time() - t0:.0f}s]")
            OUT.with_suffix(".json").write_text(json.dumps(blocks, indent=1),
                                                encoding="utf-8")
            del d

    log("")
    log("=" * 120)
    log("READING")
    log("=" * 120)
    log(f"  configurations: {n_tot}")
    log(f"  resolved POSITIVE under Breslow: {n_br}/{n_tot}")
    log(f"  resolved POSITIVE under KP:      {n_kp_pos}/{n_tot}")
    if blocks:
        me = np.array([b["modal_event_mass"] for b in blocks])
        ma = np.array([b["modal_all_exit_mass"] for b in blocks])
        log("")
        log(f"  modal EVENT mass spans {me.min():.4f} to {me.max():.4f}; "
            f"modal ALL-EXIT mass spans {ma.min():.4f} to {ma.max():.4f}")
        try:
            from kanrel.stats import spearman
            db = np.array([b["D_breslow"] for b in blocks])
            dk = np.array([b["D_kp"] for b in blocks])
            log(f"  Spearman(modal EVENT mass, D_T Breslow) = "
                f"{spearman(me, db):.4f}")
            log(f"  Spearman(modal ALL-EXIT mass, D_T Breslow) = "
                f"{spearman(ma, db):.4f}   [what v1 reported]")
            log(f"  Spearman(modal EVENT mass, D_T KP)      = "
                f"{spearman(me, dk):.4f}")
        except Exception as e:
            log(f"  correlation skipped: {type(e).__name__}: {str(e)[:50]}")
    log("")
    if n_tot and n_kp_pos == 0 and n_br:
        log("  The Breslow effect replicates on populations the finding has "
            "never seen, and the KP arm shows no resolved advantage on any of")
        log("  them.  The conclusion is about Breslow's baseline estimator, "
            "not about the six cohorts the paper happens to use.")
    elif n_kp_pos:
        log("  SOME configurations DO show a resolved grouped advantage under "
            "KP.  That must be reported rather than reconciled -- it bounds")
        log("  the KP-null rather than confirming it.")
    log("")
    log("  wrote " + str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
