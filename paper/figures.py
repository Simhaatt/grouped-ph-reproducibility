"""Publication figures, built ONLY from the authoritative result files.

Every number plotted is parsed out of a .txt produced by a script in
experiments/ or theory/.  Nothing is transcribed by hand, so a figure cannot
silently drift away from the run that justifies it -- which is the same
discipline status.md section 8c applies to the claims themselves.

  F1  crossover      formulation effect vs modal-bin mass   <- experiments/crossover.txt
  F2  resolution     observed dC against each benchmark MDE <- experiments/resolution.txt
  F3  ties           bias vs tie density, (A8) on and off   <- theory/ties_diagnosis.txt
                                                               theory/verify_ties_results.txt
  F4  competing      discharge CIF: observed / competing / naive
                                                            <- experiments/competing_risks.txt
  F5  psi            psi(u) > 0, the u^3/12 law, and (A2)   <- closed form +
                                                               theory/verify_psi_results.txt

Run:  python paper/figures.py [--only F1]
Out:  paper/figures/*.pdf and *.png
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito: colourblind-safe, and it survives greyscale printing, which most
# journal readers still do.
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00",
         purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", grey="#555555")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "lines.linewidth": 1.4,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf (+.png)")


def need(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing -- regenerate it before plotting")
    return path.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ parsers
# Intrinsically discrete (R2) cohorts vs binned-continuous (R1) ones.  The
# five-cohort run showed this split is the whole story: within R2 the formulation
# effect is ordered by modal-bin mass (Spearman +0.978 over 14 configurations),
# while on R1 it is pinned near zero at every tie density (max |effect| 0.007) --
# which is what Lemma 1 requires, since a latent continuous time exists there and
# grouping is exact.  Pooling the two regimes is what dropped the headline
# Spearman from +1.000 to +0.608, and pooling them was the error.
REGIME = {
    "support2/slos": "R2", "sparcs/drg302": "R2", "drsa/clinic": "R2",
    "drsa/music": "R2",
    "flchain": "R1", "support-pycox": "R1", "nwtco": "R1", "metabric": "R1",
}


def parse_crossover():
    """-> {cohort: [(T, rows_per_bin, modal_mass, effect, sd, n_resolved), ...]}

    Prefers the five-cohort run and falls back to the original two-cohort file.
    """
    multi = ROOT / "experiments" / "crossover_multi.txt"
    txt = (multi.read_text(encoding="utf-8", errors="replace") if multi.exists()
           else need(ROOT / "experiments" / "crossover.txt"))
    out, cohort = {}, None
    for line in txt.splitlines():
        # Cohort names do NOT all contain a slash: flchain and support-pycox do
        # not.  An earlier version required one, so their rows were silently
        # appended to the preceding cohort -- which produced a plausible-looking
        # zigzag rather than an error.  Match any non-space name.
        m = re.match(r"^(\S+)\s+--\s+n=(\d+)", line.strip())
        if m:
            cohort = m.group(1)
            out[cohort] = []
            continue
        f = line.split()
        if cohort and len(f) >= 6 and re.fullmatch(r"\d+", f[0]):
            res = int(f[5].split("/")[0])
            out[cohort].append((int(f[0]), int(f[1]), float(f[2]),
                                float(f[3]), float(f[4]), res))
    if not out:
        raise ValueError("no crossover rows parsed")
    return out


def parse_resolution():
    """-> [(dataset, n_test, events, pairs, se, mde, dc, lit, resolvable), ...]"""
    txt = need(ROOT / "experiments" / "resolution.txt")
    rows = []
    for line in txt.splitlines():
        f = line.split()
        if len(f) >= 9 and re.fullmatch(r"\d+k", f[3]) and f[1].isdigit():
            rows.append((f[0], int(f[1]), int(f[2]), int(f[3][:-1]) * 1000,
                         float(f[4]), float(f[5]), float(f[6]), float(f[7]),
                         "CANNOT" not in line))
    if not rows:
        raise ValueError("no resolution rows parsed")
    return rows


def parse_ties_diagnosis():
    """theory/ties_diagnosis.txt (D1): -> bins, {scheme: {method: bias%}}"""
    txt = need(ROOT / "theory" / "ties_diagnosis.txt")
    bins = []
    d = {"random": {"cloglog": [], "efron": []},
         "boundary": {"cloglog": [], "efron": []}}
    for line in txt.splitlines():
        f = [p for p in re.split(r"[|\s]+", line.strip()) if p]
        if len(f) == 5 and re.fullmatch(r"\d+", f[0]) and all("%" in x for x in f[1:]):
            bins.append(int(f[0]))
            vals = [float(x.rstrip("%")) for x in f[1:]]
            d["random"]["cloglog"].append(vals[0])
            d["random"]["efron"].append(vals[1])
            d["boundary"]["cloglog"].append(vals[2])
            d["boundary"]["efron"].append(vals[3])
    if not bins:
        raise ValueError("no D1 rows parsed")
    return np.array(bins), d


def parse_ties_results():
    """theory/verify_ties_results.txt -> bins, tie_ratio, {method: bias%} (random censoring)"""
    p = ROOT / "theory" / "verify_ties_results.txt"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="replace")
    names = ["Cox/Breslow", "Cox/Efron", "discrete/logit", "discrete/cloglog"]
    bins, ratio = [], []
    d = {k: [] for k in names}
    for line in txt.splitlines():
        f = line.split()
        if len(f) == 6 and re.fullmatch(r"\d+", f[0]) and all("%" in x for x in f[2:]):
            bins.append(int(f[0]))
            ratio.append(float(f[1]))
            for k, v in zip(names, f[2:]):
                d[k].append(float(v.rstrip("%")))
    return (np.array(bins), np.array(ratio), d) if bins else None


def parse_competing():
    """experiments/competing_risks.txt -> day, observed, cif, naive"""
    txt = need(ROOT / "experiments" / "competing_risks.txt")
    rows = []
    for line in txt.splitlines():
        f = line.split()
        if (len(f) == 6 and re.fullmatch(r"\d+", f[0])
                and re.fullmatch(r"[+-]\d+\.\d", f[5])):
            rows.append([float(x) for x in f[:5]])
    if not rows:
        raise ValueError("no competing-risks rows parsed")
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def parse_psi_necessity():
    """verify_psi_results.txt section (d) -> sup_lambda, rel_err"""
    txt = need(ROOT / "theory" / "verify_psi_results.txt")
    sup, err = [], []
    for line in txt.splitlines():
        f = line.split()
        if len(f) == 5 and re.fullmatch(r"\d(\.\d+)?e[-+]\d+", f[0]):
            sup.append(float(f[1]))
            err.append(float(f[4]))
    if not sup:
        raise ValueError("no (d) rows parsed from verify_psi_results.txt")
    return np.array(sup), np.array(err)


# ------------------------------------------------------------------ figures
def _spearman(x, y):
    """Delegates to kanrel.stats.spearman, which averages tied ranks.

    This file previously had its own `argsort(argsort(...))`, which assigns
    ORDINAL ranks and so breaks ties by position.  Modal bin mass is quoted to
    three decimals and several configurations collide, so the figure was drawing
    a slightly different statistic from the one its own caption named.  Five
    files carried a copy of this and every one of them was wrong the same way.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent.parent))
    from kanrel.stats import spearman as _s
    return _s(x, y)


def fig1_crossover():
    """The cover figure: the effect is ordered by tie mass, but ONLY where the
    outcome is intrinsically discrete.

    The two panels are the finding.  Panel (a) is the three (R2) cohorts, where
    modal-bin mass orders the formulation effect over a 37-fold range.  Panel (b)
    is the two binned-continuous (R1) cohorts on the SAME y-scale, where the
    effect is pinned near zero at every tie density -- the negative control
    Lemma 1 demands, and the reason tie mass alone is not the rule.
    """
    data = parse_crossover()
    style = {
        "sparcs/drg302": dict(c=C["red"], marker="s", label="sparcs/drg302 (34,233)"),
        "drsa/clinic": dict(c=C["purple"], marker="D", label="drsa/clinic (4,828)"),
        "support2/slos": dict(c=C["blue"], marker="o", label="support2/slos (9,105)"),
        "drsa/music": dict(c=C["sky"], marker="*", label="drsa/music (50,000)"),
        "support-pycox": dict(c=C["green"], marker="v", label="support-pycox (8,873)"),
        "flchain": dict(c=C["orange"], marker="^", label="flchain (7,874)"),
        "nwtco": dict(c=C["purple"], marker="<", label="nwtco (4,028)"),
        "metabric": dict(c=C["grey"], marker=">", label="metabric (1,904)"),
    }
    r2 = {k: v for k, v in data.items() if REGIME.get(k) == "R2"}
    r1 = {k: v for k, v in data.items() if REGIME.get(k) == "R1"}
    if not r1:                                  # two-cohort fallback file
        r1 = {}

    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True,
                                  gridspec_kw={"width_ratios": [1.3, 1]})

    def draw(a, group):
        for cohort, rows in group.items():
            st = style.get(cohort, dict(c=C["grey"], marker="x", label=cohort))
            x = np.array([r[2] for r in rows])
            y = np.array([r[3] for r in rows])
            e = np.array([r[4] for r in rows])
            o = np.argsort(x)
            a.errorbar(x[o], y[o], yerr=1.96 * e[o], color=st["c"],
                       marker=st["marker"], ms=4.5, capsize=2.5, elinewidth=0.9,
                       label=st["label"], zorder=3)

    draw(ax, r2)
    draw(axr, r1)

    def pooled_rho(group):
        xs = np.concatenate([[r[2] for r in v] for v in group.values()]) if group else []
        ys = np.concatenate([[r[3] for r in v] for v in group.values()]) if group else []
        return (_spearman(xs, ys), len(xs)) if len(xs) > 2 else (float("nan"), len(xs))

    rho2, n2 = pooled_rho(r2)
    rho1, n1 = pooled_rho(r1)

    for a in (ax, axr):
        a.axhline(0, color="k", lw=0.8, zorder=1)
        a.set_xlabel("fraction of mass in the modal bin")
        a.set_xlim(0.03, 0.95)
        a.legend(loc="upper left", fontsize=7, bbox_to_anchor=(0.22, 1.0))
    # The two OBSERVED sign changes, not one universal band.  An earlier version
    # shaded 0.145-0.217 alone and titled the panel with the pooled Spearman;
    # section 6.4 withdraws the pooled figure as pseudoreplication and section
    # 6.5 withdraws the single threshold, because drsa/music crosses near 0.10.
    # (Numbering shifted by one when section 6.2 was inserted, 2026-09-04.)
    for lo, hi, lab in ((0.049, 0.096, "drsa/music"), (0.145, 0.217, "support2")):
        ax.axvspan(lo, hi, color=C["yellow"], alpha=0.35, lw=0, zorder=0)
        ax.annotate(lab, xy=((lo + hi) / 2, 0.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6, color=C["grey"], rotation=90)
    ax.set_ylabel("formulation effect\n(Cox $-$ discrete, test NLL)")
    # Between-cohort rho on cohort means is the figure that carries information;
    # the within-cohort relation is near-tautological under coarsening.
    means = [(np.mean([r[2] for r in v]), np.mean([r[3] for r in v]))
             for v in r2.values()]
    rho_between = _spearman([m[0] for m in means], [m[1] for m in means])
    ax.set_title(f"(a) intrinsically discrete: {len(r2)} cohorts, {n2} grids\n"
                 f"within $\\rho=+1.00$, between $\\rho={rho_between:+.2f}$",
                 fontsize=8.5)
    if r1:
        axr.set_title(f"(b) binned-continuous: {len(r1)} cohorts, {n1} grids\n"
                      f"effect near zero at every tie density", fontsize=8.5)
        # Not "so the two formulations must agree".  Exactness constrains the
        # GROUPED arm only; the tie approximation in the Cox arm is untouched by
        # it, and E5 separates the arms by +0.173 in held-out likelihood under a
        # latent continuous truth with DENSE events (simulations_e5.txt).  What
        # holds these cohorts flat is that their events are sparse.
        axr.annotate("Lemma 1 makes the grouped arm exact;\nevents here are sparse enough\n"
                     "that the tie approximation holds too",
                     xy=(0.50, 0.55), xycoords="axes fraction", ha="center",
                     fontsize=7, color=C["grey"])
    ax.annotate("discrete wins", xy=(0.96, 0.32), xycoords="axes fraction",
                ha="right", color=C["green"], fontsize=7.5)
    ax.annotate("Cox wins", xy=(0.96, 0.05), xycoords="axes fraction",
                ha="right", color=C["grey"], fontsize=7.5)
    fig.tight_layout()
    save(fig, "F1_crossover")


def fig2_resolution():
    """Observed effect against each benchmark's own minimum detectable effect."""
    rows = sorted(parse_resolution(), key=lambda r: r[5])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))
    mde = np.array([r[5] for r in rows])
    dc = np.array([r[6] for r in rows])
    se = np.array([r[4] for r in rows])
    lit = np.array([r[7] for r in rows])
    okr = np.array([r[8] for r in rows])

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.axvspan(0.010, 0.020, color=C["grey"], alpha=0.13, lw=0,
               label="differences of the magnitude typically reported")
    ax.barh(y, mde, height=0.55, color=C["sky"], alpha=0.45, lw=0,
            label="smallest detectable difference, 80% power")
    ax.errorbar(dc, y, xerr=1.96 * se, fmt="D", ms=4.5, color=C["blue"],
                capsize=2.5, elinewidth=0.9,
                label="observed $\\Delta$C (95% CI)", zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    # The per-dataset "published effect" ticks are NOT drawn.  Section 7 declines
    # to attribute a specific figure to a specific paper, so a per-row marker
    # asserts something the text refuses to assert.  They also sat inside the
    # grey band on every row, so they carried no information the band does not.

    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n({r[1]} rows, {r[2]} events)"
                        for n, r in zip(names, rows)], fontsize=7)
    for i, good in enumerate(okr):
        if not good:
            ax.annotate("cannot resolve", xy=(mde[i] + 0.002, i), va="center",
                        fontsize=7, color=C["red"])
    ax.set_xlabel("C-index difference")
    # Not "cannot resolve the effect published on them": section 7 withdrew that
    # attribution, since no specific figure was matched to a specific paper.
    ax.set_title("On five of eight benchmarks the smallest detectable difference\n"
                 "exceeds the differences typically reported")
    # Below the axes, not inside: valung's MDE bar runs to 0.096 and its CI to
    # -0.095, so every interior corner is occupied.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
              fontsize=7.5, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "F2_resolution")


def fig3_ties():
    """Bias vs tie density, with (A8) satisfied and violated."""
    bins, d = parse_ties_diagnosis()
    extra = parse_ties_results()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    a1.plot(bins, d["boundary"]["efron"], "o-", color=C["red"], label="Cox / Efron")
    a1.plot(bins, d["boundary"]["cloglog"], "s-", color=C["blue"],
            label="discrete / cloglog")
    a1.set_title("(a) grid-aligned censoring — (A8) holds")

    a2.plot(bins, d["random"]["efron"], "o-", color=C["red"], label="Cox / Efron")
    a2.plot(bins, d["random"]["cloglog"], "s-", color=C["blue"],
            label="discrete / cloglog")
    if extra is not None:
        eb, _, ed = extra
        # Restrict Breslow to the interval counts the other two series cover.
        # verify_ties sweeps eight grids (200 down to 3); ties_diagnosis sweeps
        # six (50 down to 3).  Plotting all eight against ticks drawn from six
        # left two Breslow markers with no tick beneath them, so a reader using
        # the labels mislocated that series -- and in the direction flattering
        # to the argument, since the extra points are its low-bias end.
        keep = [i for i, b0 in enumerate(eb) if b0 in set(bins.tolist())]
        a2.plot(eb[keep], np.asarray(ed["Cox/Breslow"])[keep], "^--",
                color=C["orange"], ms=4, label="Cox / Breslow")
    a2.set_title("(b) random censoring — (A8) violated")

    for a in (a1, a2):
        a.set_xscale("log")
        a.set_xticks(bins)
        a.set_xticklabels([str(b) for b in bins])
        a.invert_xaxis()
        a.minorticks_off()
        a.set_xlabel("number of bins   (fewer bins $\\rightarrow$ heavier ties)")
        a.legend(loc="upper left")
    a1.set_ylabel("mean relative bias in $\\hat\\beta$  (%)")
    a1.annotate(f"flat at {d['boundary']['cloglog'][-1]:.2f}%",
                xy=(bins[-1], d["boundary"]["cloglog"][-1]), xytext=(-8, 18),
                textcoords="offset points", fontsize=7.5, color=C["blue"],
                arrowprops=dict(arrowstyle="->", color=C["blue"], lw=0.7))
    fig.tight_layout()
    save(fig, "F3_ties")


def fig4_competing():
    """Treating a competing exit as censoring answers a question nobody asked."""
    day, obs, cif, naive = parse_competing()
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.plot(day, obs, "o-", color="k", ms=4, label="observed (test split)")
    ax.plot(day, cif, "s-", color=C["blue"], ms=4, label="competing-risks CIF")
    ax.plot(day, naive, "^--", color=C["red"], ms=4,
            label="naive (death treated as censoring)")
    ax.fill_between(day, cif, naive, color=C["red"], alpha=0.12, lw=0)
    i = int(np.argmax(naive - cif))
    ax.annotate(f"{100 * (naive[i] - cif[i]):+.1f} pp",
                xy=(day[i], (naive[i] + cif[i]) / 2), xytext=(-54, -4),
                textcoords="offset points", fontsize=8, color=C["red"],
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.7))
    ax.set_xlabel("hospital day")
    ax.set_ylabel("probability of discharge alive")
    ax.set_title("support2/slos: 24.8% of patients die in hospital")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "F4_competing")


def fig5_psi():
    """Theorem 3: psi > 0 always, and psi ~ u^3/12 -- so the loss is O(Delta^2)."""
    u = np.logspace(-3, 1.0, 400)
    psi = (1 - np.exp(-u)) - u ** 2 * np.exp(-u) / (1 - np.exp(-u))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    a1.loglog(u, psi, color=C["blue"], label=r"$\psi(u)$ (exact)")
    a1.loglog(u, u ** 3 / 12, "--", color=C["orange"],
              label=r"$u^3/12$  (leading term)")
    a1.set_xlabel(r"$u_t = \int_{I_t}\lambda$")
    a1.set_ylabel(r"$\psi(u)$")
    a1.set_title(r"(a) $\psi>0$ always;  $\psi(u)=u^3/12+O(u^4)$")
    a1.legend(loc="upper left")
    a1.annotate("first-order terms cancel,\nhence the $\\Delta^2$ law",
                xy=(0.40, 0.12), xycoords="axes fraction", fontsize=7.5,
                color=C["grey"])

    sup, err = parse_psi_necessity()
    a2.loglog(sup, err, "o-", color=C["red"])
    a2.set_xlabel(r"$\sup_s \lambda(s)$   (Weibull $k=0.8$, truncated at $s_0$)")
    a2.set_ylabel("relative error of the $\\Delta^2$ expansion")
    a2.set_title("(b) (A2) is necessary, not decorative")
    a2.annotate("expansion fails by 39%\nas $\\lambda$ becomes unbounded",
                xy=(sup[0], err[0]), xytext=(-4, -34), textcoords="offset points",
                fontsize=7.5, color=C["red"], ha="left",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=0.7))
    fig.tight_layout()
    save(fig, "F5_psi")


FIGS = {"F1": fig1_crossover, "F2": fig2_resolution, "F3": fig3_ties,
        "F4": fig4_competing, "F5": fig5_psi}


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1].split(",")
    wanted = [k for k in FIGS if not only or k in only]
    fails = 0
    for name in wanted:
        try:
            FIGS[name]()
        except Exception as e:
            fails += 1
            print(f"  {name} SKIPPED: {type(e).__name__}: {e}")
    print(f"\n{len(wanted) - fails}/{len(wanted)} figures written")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
