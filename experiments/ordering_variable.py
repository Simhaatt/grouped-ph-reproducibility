"""Which observable scalar orders D_T?  Section 6.3 picks modal bin mass.

WHY THIS EXISTS.  Section 6.3 orders the formulation effect by the fraction of
subjects in the modal bin, and reports a Spearman correlation to support it.  E8
shows that choice is wrong in a controlled design: with the baseline hazard level
crossed against n and T, the cell with the HIGHEST modal bin mass (0.807) has
D_T = -0.00013 while a cell at 0.505 has D_T = +0.088.  Modal bin mass counts
every subject, censored included, and it is largest exactly where administrative
censoring piles subjects into the last bin -- where there are almost no EVENT ties
for Cox's approximation to mishandle.

Three candidates are separable in principle and near-collinear on real cohorts:

    modal bin mass    max_t #{k_i = t} / n                  what section 6.3 uses
    modal event mass  max_t #{k_i = t, d_i = 1} / #events   event ties only
    events per bin    #events / T                           density, not concentration

This script computes all three for every configuration in the real sweep, pairs
them with the D_T already measured in protocol_decomp.txt, and reports the
Spearman correlation of each against D_T -- pooled, and within regime.  Pooling
across regimes is pseudoreplication (section 6.3 records that lesson), so the
within-regime figures are the ones that decide the question.

Run:  python -u experiments/ordering_variable.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from kanrel import data as D
from kanrel.stats import spearman
from experiments.crossover import COHORTS, coarsen
# REGIME lives in paper/figures.py, which is not an importable package, so it is
# loaded by path rather than duplicated -- a second copy would be free to drift
# out of step with the figures.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_figs", Path(__file__).resolve().parent.parent / "paper" / "figures.py")
_figs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_figs)
REGIME = _figs.REGIME

SRC = Path(__file__).resolve().parent / "protocol_decomp.txt"
OUT = Path(__file__).resolve().parent / "ordering_variable.txt"


def parse_dt(path):
    """-> {(cohort, T): D_T} for cox:efron+breslow against the grouped joint MLE.

    Reads the FULL COHORT block only.  The subsampled block is a different
    sample size and mixing the two would compare cohorts of different n.
    """
    out = {}
    if not path.exists():
        return out
    cur, in_full = None, False
    for line in path.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^(\S+) @ T=(\d+)\s", line)
        if m:
            cur, in_full = (m.group(1), int(m.group(2))), False
            continue
        if line.strip().startswith("FULL COHORT"):
            in_full = True
            continue
        if line.strip().startswith("SUBSAMPLE"):
            in_full = False
            continue
        m2 = re.match(r"^\s+cox:efron\+breslow\s+([-+][\d.]+)", line)
        if m2 and cur and in_full:
            out[cur] = float(m2.group(1))
    return out


def scalars(d):
    idx = np.asarray(d.bin_idx).astype(int)
    ev = np.asarray(d.event).astype(float)
    T = d.n_bins
    modal = float(np.bincount(idx, minlength=T).max() / len(idx))
    ec = np.bincount(idx[ev == 1], minlength=T)
    tot = int(ec.sum())
    ev_modal = float(ec.max() / tot) if tot else float("nan")
    return modal, ev_modal, (tot / T if T else float("nan")), tot


def main():
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dts = parse_dt(SRC)
    log("=" * 104)
    log("WHICH SCALAR ORDERS D_T ON THE REAL COHORTS?")
    log(f"  D_T read from {SRC.name} (FULL COHORT blocks, cox:efron+breslow)")
    log(f"  {len(dts)} configurations available")
    log("=" * 104)
    log(f"  {'cohort':<18}{'T':>4}{'reg':>5}{'modal':>9}{'ev modal':>10}"
        f"{'ev/bin':>9}{'events':>9}{'D_T':>11}")
    rows = []
    for name, (loader, grid, _) in COHORTS.items():
        # Check for a measured D_T BEFORE loading.  sparcs is a 1.15 GB CSV and
        # drsa/music a 2.2 GB archive; the first version read both in full and
        # then skipped every one of their configurations because the sweep had
        # not reached them yet.
        if not any((name, t) in dts for t in grid):
            log(f"  {name:<18} no measured D_T yet; not loaded")
            continue
        try:
            base = loader()
        except Exception as e:
            log(f"  {name:<18} LOAD FAILED {type(e).__name__}: {str(e)[:40]}")
            continue
        reg = REGIME.get(name, "?")
        for T in [t for t in grid if t <= base.n_bins]:
            if (name, T) not in dts:
                continue
            d = D.onehot_ordinals(coarsen(base, T))
            mo, em, eb, tot = scalars(d)
            dt = dts[(name, T)]
            log(f"  {name:<18}{T:>4}{reg:>5}{mo:>9.4f}{em:>10.4f}{eb:>9.0f}"
                f"{tot:>9}{dt:>+11.5f}")
            rows.append((name, T, reg, mo, em, eb, dt))

    if len(rows) < 6:
        log("")
        log("  Too few configurations parsed to correlate; rerun once")
        log("  protocol_decomp.py has finished.")
        return 0

    a = np.array([[r[3], r[4], r[5], r[6]] for r in rows], float)
    regs = np.array([r[2] for r in rows])
    log("")
    log("=" * 104)
    log("SPEARMAN AGAINST D_T")
    log("=" * 104)
    names = ("modal bin mass (section 6.3)", "modal EVENT mass", "events per bin")
    log(f"  {'subset':<22}{'n cfg':>7}" + "".join(f"{n:>32}" for n in names))
    for lbl, mask in (("pooled (pseudorep.)", np.ones(len(rows), bool)),
                      ("R2 only", regs == "R2"),
                      ("R1 only", regs == "R1")):
        if mask.sum() < 3:
            continue
        cells = "".join(f"{spearman(a[mask, j], a[mask, 3]):>32.4f}"
                        for j in range(3))
        log(f"  {lbl:<22}{int(mask.sum()):>7}" + cells)
    log("")
    log("  Pooling across regimes is pseudoreplication -- section 6.3 records that")
    log("  lesson -- so the R2 row is the one that decides which scalar to use.")
    best = int(np.argmax([abs(spearman(a[regs == 'R2', j], a[regs == 'R2', 3]))
                          for j in range(3)])) if (regs == "R2").sum() >= 3 else None
    if best is not None:
        log(f"  Largest |rho| within R2: {names[best]}.")
        if best != 0:
            log("  That is NOT the variable section 6.3 uses.  The section must be")
            log("  restated around the winner, and E8 gives the controlled")
            log("  demonstration of why modal bin mass fails.")
    log("")
    log("  CAVEAT.  These three are strongly collinear on real cohorts, so a")
    log("  ranking among them here is weak evidence on its own.  E8 is the")
    log("  identified comparison; this table is the check that the real data is")
    log("  consistent with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
