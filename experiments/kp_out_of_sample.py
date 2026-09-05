"""Does the KP-null replicate OUT OF SAMPLE, on cohorts that formed no part of it?

`experiments/kp_vs_breslow.txt` shows D_T zero or negative in 21 of 21
configurations once Cox is given a profile-MLE baseline, against a Breslow effect
reaching 13.7 standard errors on the same data.  Every one of those cohorts was
already in the paper.

E22 supplied genuinely new R2 cohorts: each SPARCS APR-DRG is a distinct patient
population on the same whole-day scale, and the largest are an order of magnitude
bigger than anything the paper has used -- drg640 at 207,764 rows against drg302's
34,233.  They were not used to form any claim here.

So this is the out-of-sample test of the corrected claim, and it is the one that
decides whether the reframing is a property of six cohorts or of the estimator:

    Breslow arm   Cox/Efron beta with a moment-estimated baseline
    KP arm        the SAME beta with the cloglog profile MLE for alpha

If the Breslow column stays large and the KP column stays at zero on populations
the finding has never seen, the conclusion is about Breslow's baseline estimator
and not about these cohorts.

Run:  python -u experiments/kp_out_of_sample.py [drg ...]
"""
from __future__ import annotations

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
from experiments.crossover import coarsen
from experiments.protocol_decomp import nb_se
from experiments.real_data import split

N_SPLITS = 10
GRIDS = (6, 15, 30)
# drg640 has 207,764 rows and the Cox arm's LBFGS did not finish one split in
# 35 minutes, so each cohort is capped.  50,000 is still a NEW population the
# paper has never used and still 1.5x larger than drg302, so the out-of-sample
# character of the test is intact; only the precision is reduced, and the cap is
# printed with every row rather than left implicit.
MAX_ROWS = 50000
OUT = Path(__file__).resolve().parent / "kp_out_of_sample.txt"
# The four largest APR-DRGs after 302, which is already in the paper.
NEW_DRGS = ("640", "560", "540", "720")


def one(d, seeds):
    """Breslow and KP arms on identical splits, plus the E3 decomposition."""
    br, kp, base_share = [], [], []
    for s in seeds:
        tr, te = split(d, frac=0.3, seed=s)
        tr, te = D.clip_to_train_range(tr, te)
        trs, tes = standardize(tr, te)
        Xtr, Xte = trs.X.astype(float), tes.X.astype(float)
        T = d.n_bins
        try:
            b = CA.fit_cox_ties(Xtr, trs.bin_idx, trs.event, "efron")
            r_tr, r_te = CA.risk(Xtr, b), CA.risk(Xte, b)
            n1 = CA.nll_from_hazards(
                CA.hazards_breslow(r_tr, trs.bin_idx, trs.event, r_te, T),
                tes.bin_idx, tes.event, T)
            n2 = CA.nll_from_hazards(
                CA.hazards_kalbfleisch_prentice(r_tr, trs.bin_idx, trs.event,
                                                r_te, T),
                tes.bin_idx, tes.event, T)
            bj, aj = CA.fit_grouped_joint(Xtr, trs.bin_idx, trs.event, T)
            n3 = CA.nll_from_hazards(CA.hazards_from_alpha(Xte, bj, aj),
                                     tes.bin_idx, tes.event, T)
        except Exception as e:
            print(f"      split {s} FAILED {type(e).__name__}: {str(e)[:50]}",
                  flush=True)
            continue
        d1, d2 = float(np.mean(n1 - n3)), float(np.mean(n2 - n3))
        br.append(d1)
        kp.append(d2)
        if abs(d1) > 1e-9:
            base_share.append(100.0 * (d1 - d2) / d1)
    return np.array(br), np.array(kp), np.array(base_share)


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("-")] or list(NEW_DRGS)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    log("=" * 104)
    log("OUT-OF-SAMPLE TEST OF THE KP-NULL")
    log("  New SPARCS APR-DRG cohorts, none used to form any claim in the paper.")
    log(f"  {N_SPLITS} splits per configuration; NB SE is Nadeau-Bengio corrected.")
    log(f"  Each cohort capped at {MAX_ROWS:,} rows (see MAX_ROWS); still a new population.")
    log("  D_T > 0 favours the grouped joint MLE.  'res' is |D_T| > 1.96 NB SE.")
    log("=" * 104)
    log(f"  {'cohort':<18}{'n':>9}{'T':>4}{'modal':>8}"
        f"{'D_T Breslow':>13}{'NB SE':>9}{'res':>5}"
        f"{'D_T KP':>11}{'NB SE':>9}{'res':>5}{'base%':>8}")
    n_br, n_kp_pos, n_tot = 0, 0, 0
    for drg in want:
        try:
            base = D.load_sparcs(drg, horizon=30)
            if base.n > MAX_ROWS:
                rng = np.random.default_rng(20260904)
                ix = rng.choice(base.n, MAX_ROWS, replace=False)
                base = D.SurvData(
                    base.X[ix], base.bin_idx[ix], base.event[ix], base.n_bins,
                    base.feature_names, name=base.name,
                    intrinsically_discrete=base.intrinsically_discrete,
                    entry_idx=None if base.entry_idx is None else base.entry_idx[ix],
                    bin_edges=base.bin_edges, meta=base.meta)
        except Exception as e:
            log(f"  drg{drg:<14} LOAD FAILED {type(e).__name__}: {str(e)[:40]}")
            continue
        for T in GRIDS:
            if T > base.n_bins:
                continue
            t0 = time.time()
            d = D.onehot_ordinals(coarsen(base, T))
            br, kp, sh = one(d, range(N_SPLITS))
            if br.size < 2 or kp.size < 2:
                log(f"  sparcs/drg{drg}@T{T}  no split completed")
                continue
            modal = float(np.bincount(d.bin_idx.astype(int)).max() / d.n)
            sb, sk = nb_se(br), nb_se(kp)
            rb = "yes" if abs(br.mean()) > 1.959964 * sb else "no"
            rk = "yes" if abs(kp.mean()) > 1.959964 * sk else "no"
            n_tot += 1
            n_br += (br.mean() > 0 and rb == "yes")
            n_kp_pos += (kp.mean() > 0 and rk == "yes")
            log(f"  {'sparcs/drg' + drg + '@T' + str(T):<18}{d.n:>9}{T:>4}"
                f"{modal:>8.4f}{br.mean():>+13.5f}{sb:>9.5f}{rb:>5}"
                f"{kp.mean():>+11.5f}{sk:>9.5f}{rk:>5}"
                f"{(f'{sh.mean():.1f}%' if sh.size else '--'):>8}"
                f"   [{time.time()-t0:.0f}s]")

    log("")
    log("=" * 104)
    log("READING")
    log("=" * 104)
    log(f"  configurations: {n_tot}")
    log(f"  resolved POSITIVE under Breslow: {n_br}/{n_tot}")
    log(f"  resolved POSITIVE under KP:      {n_kp_pos}/{n_tot}")
    log("")
    if n_tot and n_kp_pos == 0 and n_br:
        log("  The Breslow effect replicates on populations the finding has never")
        log("  seen, and the KP arm shows no resolved advantage on any of them.")
        log("  The conclusion is therefore about Breslow's baseline estimator, not")
        log("  about the six cohorts the paper happens to use.")
    elif n_kp_pos:
        log("  SOME configurations DO show a resolved grouped advantage under KP.")
        log("  That contradicts the 21/21 in-sample result and must be reported")
        log("  rather than reconciled -- it would mean the KP-null is a property")
        log("  of the original cohorts and not of the estimator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
