"""Dataset loaders and time discretisation for the discrete-hazard KAN.

Every loader returns a :class:`SurvData` whose ``bin_idx`` is a 0-based bin index,
ready for ``kanrel.likelihood.make_targets``.

The distinction that matters throughout is ``SurvData.intrinsically_discrete``:

  True   the outcome is counted (days, months).  Regime (R2) of ``paper/setup.md``:
         the discrete hazard IS the likelihood, and continuous-time models are
         misspecified rather than approximate.  Bins are the natural units.

  False  the outcome is a continuous duration we choose to bin.  Regime (R1):
         Theorem 3 governs how much information the binning costs, and its
         corollary gives the bin count.

`tie_ratio` (distinct times / n) is the diagnostic: near 1.0 means continuous,
near 0 means intrinsically discrete with heavy ties.
"""

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Search order for raw files; override with KANREL_DATA.
DATA_DIRS = [
    Path(os.environ.get("KANREL_DATA", "")),
    Path(__file__).resolve().parent.parent / "data",
    Path(__file__).resolve().parent.parent,
    Path.home() / "Downloads",
]


def _find(filename: str) -> Path:
    for d in DATA_DIRS:
        if d and (d / filename).exists():
            return d / filename
    raise FileNotFoundError(
        f"{filename!r} not found in {[str(d) for d in DATA_DIRS if d]}. "
        f"Set KANREL_DATA to the directory holding it."
    )


@dataclass
class SurvData:
    X: np.ndarray                      # [n, p] float32
    bin_idx: np.ndarray                # [n]   int64, 0-based exit bin
    event: np.ndarray                  # [n]   float32, 1 = event, 0 = censored
    n_bins: int
    feature_names: list[str]
    name: str = ""
    intrinsically_discrete: bool = True
    entry_idx: np.ndarray | None = None
    bin_edges: np.ndarray | None = None     # None when bins are native units
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.bin_idx)
        assert self.X.shape[0] == n == len(self.event), "ragged SurvData"
        assert self.bin_idx.min() >= 0 and self.bin_idx.max() < self.n_bins

    @property
    def n(self) -> int:
        return len(self.bin_idx)

    def summary(self) -> str:
        ev = float(self.event.mean())
        top3 = np.sort(np.bincount(self.bin_idx))[::-1][:3].sum() / self.n
        kind = "discrete" if self.intrinsically_discrete else "binned-continuous"
        return (
            f"{self.name:<22} n={self.n:>7}  p={self.X.shape[1]:>3}  T={self.n_bins:>4}  "
            f"events={ev:.3f}  top3bins={top3:.3f}  [{kind}]"
        )


# --------------------------------------------------------------- discretisation
def discretize_native(t, horizon: int, origin: int = 1):
    """Counted outcome (days/months) -> bin index, administratively censored.

    Bin t holds the count ``origin + t``.  Anything at or beyond ``horizon`` is
    right-censored in the final bin -- exactly the SPARCS "120 +" convention.
    """
    t = np.asarray(t, dtype=float)
    idx = np.clip(np.floor(t).astype(np.int64) - origin, 0, horizon - 1)
    beyond = t >= (origin + horizon - 1)
    return idx, beyond


def discretize_quantile(t, event, n_bins: int):
    """Continuous duration -> equal-event-count bins (the pycox convention)."""
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(t[event == 1] if event.any() else t, qs))
    idx = np.searchsorted(edges, t, side="left").astype(np.int64)
    return np.clip(idx, 0, len(edges)), np.concatenate([[0.0], edges, [t.max()]])


def tie_ratio(t) -> float:
    t = np.asarray(t)
    return float(pd.Series(t).nunique() / len(t))


# ------------------------------------------------------------------- covariates
def _prepare(df: pd.DataFrame, cols: list[str], max_missing: float = 0.25):
    """Drop high-missing columns, one-hot categoricals, median-impute the rest."""
    sub = df[cols].copy()
    keep = [c for c in cols if sub[c].isna().mean() <= max_missing]
    dropped = sorted(set(cols) - set(keep))
    sub = sub[keep]

    num = sub.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in keep if c not in num]
    for c in num:
        sub[c] = sub[c].fillna(sub[c].median())
    if cat:
        sub = pd.get_dummies(sub, columns=cat, drop_first=True, dummy_na=False)
    X = sub.to_numpy(dtype=np.float32)
    return X, list(sub.columns), dropped


# ----------------------------------------------------------------------- SUPPORT
SUPPORT_COVARIATES = [
    "age", "sex", "dzgroup", "dzclass", "num.co", "scoma", "race",
    "sps", "aps", "meanbp", "hrt", "resp", "temp", "wblc", "crea", "sod",
    "diabetes", "dementia", "ca", "adlsc",
]


def load_support2(outcome: str = "slos", horizon: int = 60, n_bins: int = 20):
    """SUPPORT (n=9105).  Two outcomes from one cohort.

    outcome="slos"   length of hospital stay in DAYS -- the Nawata analogue.
                     Event = discharged alive; in-hospital death censors.
                     REGIME (R2): intrinsically discrete, tie ratio ~0.018.
    outcome="dtime"  follow-up to death in days, quantile-binned.
                     REGIME (R1): the KAPLAN-HR head-to-head outcome.

    Caveat for "slos": in-hospital death is a COMPETING RISK, not independent
    censoring.  Treating it as censoring is the standard first cut and biases
    the discharge hazard upward.  Competing-risks extension is A7.
    """
    p = _find("support2csv.zip") if _find_ok("support2csv.zip") else _find("support2.csv")
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            df = pd.read_csv(z.open("support2.csv"))
    else:
        df = pd.read_csv(p)

    X, names, dropped = _prepare(df, SUPPORT_COVARIATES)

    if outcome == "slos":
        t = pd.to_numeric(df["slos"], errors="coerce").fillna(1).to_numpy()
        idx, beyond = discretize_native(t, horizon=horizon, origin=1)
        event = (1.0 - df["hospdead"].to_numpy(dtype=np.float32))
        event[beyond] = 0.0
        return SurvData(X, idx, event.astype(np.float32), horizon, names,
                        name="support2/slos", intrinsically_discrete=True,
                        meta={"tie_ratio": tie_ratio(t), "dropped": dropped,
                              "unit": "days", "competing_risk": "hospdead"})

    if outcome == "dtime":
        t = pd.to_numeric(df["d.time"], errors="coerce").fillna(1).to_numpy()
        event = df["death"].to_numpy(dtype=np.float32)
        idx, edges = discretize_quantile(t, event, n_bins)
        return SurvData(X, idx, event, int(idx.max()) + 1, names,
                        name="support2/dtime", intrinsically_discrete=False,
                        bin_edges=edges,
                        meta={"tie_ratio": tie_ratio(t), "dropped": dropped,
                              "unit": "days"})

    raise ValueError("outcome must be 'slos' or 'dtime'")


def _find_ok(fn: str) -> bool:
    try:
        _find(fn)
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------- small classic cohorts
def load_prostate(n_bins: int = 0, horizon: int = 60):
    """Vanderbilt prostate (n=502).  dtime is in MONTHS -> intrinsically discrete."""
    df = pd.read_excel(_find("prostate.xls"), engine="xlrd")
    cols = ["age", "wt", "sbp", "dbp", "hg", "sz", "sg", "ap", "stage", "rx",
            "pf", "hx", "ekg", "bm"]
    X, names, dropped = _prepare(df, [c for c in cols if c in df.columns])
    t = pd.to_numeric(df["dtime"], errors="coerce").fillna(0).to_numpy()
    status = df["status"].astype(str).str.lower()
    event = (~status.str.contains("alive")).to_numpy(dtype=np.float32)
    idx, beyond = discretize_native(t, horizon=horizon, origin=0)
    event[beyond] = 0.0
    return SurvData(X, idx, event, horizon, names, name="prostate",
                    intrinsically_discrete=True,
                    meta={"tie_ratio": tie_ratio(t), "unit": "months",
                          "dropped": dropped})


def load_pbc(n_bins: int = 20):
    df = pd.read_excel(_find("pbc.xls"), engine="xlrd")
    cols = ["age", "bili", "albumin", "protime", "sex", "stage", "alk.phos",
            "sgot", "chol", "trig", "platelet", "copper", "drug", "edema",
            "spiders", "hepatom", "ascites"]
    X, names, dropped = _prepare(df, [c for c in cols if c in df.columns])
    t = pd.to_numeric(df["fu.days"], errors="coerce").to_numpy()
    event = (pd.to_numeric(df["status"], errors="coerce") > 0).to_numpy(dtype=np.float32)
    idx, edges = discretize_quantile(t, event, n_bins)
    return SurvData(X, idx, event, int(idx.max()) + 1, names, name="pbc",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"tie_ratio": tie_ratio(t), "dropped": dropped})


def load_gbsg(n_bins: int = 20):
    df = pd.read_csv(_find("gbsg_ba_ca.dat"), sep=r"\s+")
    cols = ["age", "meno", "size", "grade", "nodes", "enodes", "pgr", "er", "hormon"]
    X, names, dropped = _prepare(df, [c for c in cols if c in df.columns])
    t = pd.to_numeric(df["rectime"], errors="coerce").to_numpy()
    event = pd.to_numeric(df["censrec"], errors="coerce").to_numpy(dtype=np.float32)
    idx, edges = discretize_quantile(t, event, n_bins)
    return SurvData(X, idx, event, int(idx.max()) + 1, names, name="gbsg",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"tie_ratio": tie_ratio(t), "dropped": dropped})


def load_valung(n_bins: int = 12):
    df = pd.read_csv(_find("valung.csv"))
    X, names, dropped = _prepare(df, ["kps", "diagtime", "age", "therapy", "cell", "prior"])
    t = df["t"].to_numpy(dtype=float)
    event = (df["dead"].astype(str) == "dead").to_numpy(dtype=np.float32)
    idx, edges = discretize_quantile(t, event, n_bins)
    return SurvData(X, idx, event, int(idx.max()) + 1, names, name="valung",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"tie_ratio": tie_ratio(t), "dropped": dropped})


# -------------------------------------------------------------------- SPARCS
SPARCS_FILE = "Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2012_20260821.csv"
AGE_ORDER = {"0 to 17": 0, "18 to 29": 1, "30 to 49": 2, "50 to 69": 3, "70 or Older": 4}


def load_sparcs(drg: str = "302", horizon: int = 30, path: str | Path | None = None):
    """SPARCS inpatient discharges.  LOS in whole DAYS, top-coded at "120 +".

    Default cohort is APR-DRG 302 (knee joint replacement, n=33,219 across 162
    facilities): elective surgical, short stay, heavy ties -- the structural
    analogue of Nawata's cataract cohort.  The eye/cataract DRGs are unusable
    here (only 494 eye procedures; cataract surgery is outpatient in the US).

    NOTE age is BANDED into 5 groups, so no smooth age curve is recoverable.
    Use support2 for anything needing continuous age.
    """
    p = Path(path) if path else _find(SPARCS_FILE)
    cols = ["APR DRG Code", "Length of Stay", "Age Group", "Gender",
            "APR Severity of Illness Code", "APR Risk of Mortality",
            "Type of Admission", "Emergency Department Indicator",
            "Facility ID", "APR Medical Surgical Description"]
    df = pd.read_csv(p, usecols=cols, dtype=str, low_memory=False)
    df = df[df["APR DRG Code"].astype(str).str.strip().str.zfill(3) == str(drg).zfill(3)]
    if df.empty:
        raise ValueError(f"no rows for APR-DRG {drg}")

    raw = df["Length of Stay"].astype(str).str.replace("+", "", regex=False).str.strip()
    t = pd.to_numeric(raw, errors="coerce").fillna(120).to_numpy()

    feat = pd.DataFrame({
        "age_band": df["Age Group"].map(AGE_ORDER).astype(float),
        "severity": pd.to_numeric(df["APR Severity of Illness Code"], errors="coerce"),
        "gender": df["Gender"].astype(str),
        "risk_mortality": df["APR Risk of Mortality"].astype(str),
        "admission": df["Type of Admission"].astype(str),
        "ed": df["Emergency Department Indicator"].astype(str),
    })
    X, names, dropped = _prepare(feat, list(feat.columns))
    idx, beyond = discretize_native(t, horizon=horizon, origin=1)
    event = np.ones(len(t), dtype=np.float32)
    event[beyond] = 0.0
    return SurvData(X, idx, event, horizon, names, name=f"sparcs/drg{drg}",
                    intrinsically_discrete=True,
                    meta={"tie_ratio": tie_ratio(t), "unit": "days",
                          "facilities": df["Facility ID"].nunique(),
                          "dropped": dropped})


# ----------------------------------------------------------------- synthetic
def make_grouped_weibull(
    n: int = 4000, p: int = 5, shape: float = 1.5, horizon: float = 2.0,
    n_bins: int = 20, censor_rate: float = 0.3, change_point: bool = False,
    seed: int = 0,
):
    """Grouped Weibull PH with a KNOWN log-hazard-ratio g(x).

    Under Lemma 1 the grouped cloglog model is EXACTLY correct here, so any
    departure of the fitted beta from the truth is estimation error, not
    misspecification -- which is what makes this the estimator-correctness gate.

    change_point=True plants a break in x1 at 0, mimicking the age-40 break
    Nawata et al. report, so recovery can be tested against a known truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)

    if change_point:
        g = 0.9 * np.where(X[:, 0] < 0.0, 0.0, X[:, 0]) + 0.8 * np.tanh(2 * X[:, 1])
    else:
        g = 0.8 * (X[:, 0] ** 2 - 1.0) + 1.0 * np.sin(np.pi * X[:, 1])
    g = g + 0.6 * X[:, 2]                       # x3 linear; x4, x5 are noise

    u = rng.uniform(size=n)
    t = (-np.log(u) / np.exp(g)) ** (1.0 / shape)      # Weibull PH, scale 1
    c = rng.uniform(0, horizon / max(censor_rate, 1e-6), size=n)
    obs = np.minimum(t, np.minimum(c, horizon))
    event = ((t <= c) & (t <= horizon)).astype(np.float32)

    edges = np.linspace(0, horizon, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, obs, side="left") - 1, 0, n_bins - 1)
    return SurvData(X, idx.astype(np.int64), event, n_bins,
                    [f"x{i+1}" for i in range(p)], name="synthetic/weibull-ph",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"g_true": g, "t_true": t, "shape": shape,
                          "change_point": change_point})


LOADERS = {
    "support2/slos": lambda: load_support2("slos"),
    "support2/dtime": lambda: load_support2("dtime"),
    "prostate": load_prostate,
    "pbc": load_pbc,
    "gbsg": load_gbsg,
    "valung": load_valung,
}


# ------------------------------------------------------------------- DRSA
def _drsa_field_map(featindex: Path) -> dict[int, int]:
    """featindex lines are '<field>:<value>\t<index>' -> {index: field}."""
    m = {}
    with open(featindex, encoding="utf-8", errors="replace") as f:
        for line in f:
            key, _, idx = line.rstrip("\n").partition("\t")
            if not idx:
                continue
            field = key.split(":", 1)[0]
            try:
                m[int(idx)] = int(field)
            except ValueError:
                continue
    return m


def load_drsa(split: str = "CLINIC", horizon: int | None = None,
              max_rows: int | None = None, max_cardinality: int = 100,
              root: str | Path = "data/drsa-data"):
    """Ren et al. (2019) DRSA benchmarks, `.yzbx` format.

    Line layout is ``y z b idx:1 idx:1 ...`` where ``y`` is an unused placeholder
    (always 0 in these files).  The survival semantics are

        observed time = min(z, b),   event = 1{z <= b}

    so ``b`` is a KNOWN administrative censoring time -- which means IPCW needs no
    censoring-distribution estimate here.

    split="MUSIC"  is NOT KKBox.  Ren et al. describe MUSIC as a user-lifetime
                   analysis over the Last.fm 1K-user listening dataset (Celma,
                   2010): the tracked event is a user's return visit to the
                   service, and the duration is the time from one visit to the
                   next.  This comment previously said "KKBox churn", which an
                   external audit flagged and the DRSA documentation contradicts.
                   2,796,646 rows, tie ratio 7e-4.

                   ⚠️ UNRESOLVED AND CONSEQUENTIAL.  If ~1,000 users contribute
                   2.8M visit intervals, rows are REPEATED MEASURES WITHIN USER,
                   not independent subjects, and every standard error computed on
                   this cohort is understated -- our splits randomise over rows,
                   so the same user appears in train and test.  The .yzbx format
                   carries no user identifier, so this cannot be checked from the
                   distributed files.  Until it is, treat drsa/music standard
                   errors as lower bounds and do not quote them as resolved.
    split="CLINIC" is a small clinical cohort, 4,828 rows.

    CAVEAT: DRSA one-hot encoded every covariate, so there are no continuous
    features here -- only a handful of categorical FIELDS with many levels each.
    We recover the field structure and return one ordinal column per field, which
    is what a KAN can actually use.  For continuous covariates (where a raw
    plan days, n_prev_churns) the raw WSDM archive must be processed instead.
    """
    root = Path(root)
    fmap = _drsa_field_map(root / split / "featindex.txt")
    n_fields = max(fmap.values()) + 1

    rows, zs, bs = [], [], []
    with open(root / split / "train.yzbx.txt", encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            if max_rows and k >= max_rows:
                break
            parts = line.split()
            zs.append(int(parts[1])); bs.append(int(parts[2]))
            vec = np.zeros(n_fields, dtype=np.float32)
            for tok in parts[3:]:
                idx = int(tok.split(":", 1)[0])
                fld = fmap.get(idx)
                if fld is not None:
                    vec[fld] = idx
            rows.append(vec)

    X = np.vstack(rows)
    z = np.asarray(zs, dtype=float)
    b = np.asarray(bs, dtype=float)
    obs = np.minimum(z, b)
    event = (z <= b).astype(np.float32)

    H = horizon or int(obs.max())
    idx, beyond = discretize_native(obs, horizon=H, origin=int(obs.min()))
    event = event.copy()
    event[beyond] = 0.0
    # Index IDs are arbitrary labels, so a spline over them is meaningless for
    # high-cardinality fields.  Keep only fields a univariate function can
    # sensibly act on; target-encoding or embeddings for the rest is future work.
    card = np.array([len(np.unique(X[:, j])) for j in range(X.shape[1])])
    keep = (X.std(0) > 0) & (card <= max_cardinality)
    if not keep.any():
        raise ValueError(f"no field with cardinality <= {max_cardinality}; card={card}")
    return SurvData(X[:, keep], idx, event, H,
                    [f"f{i}" for i in np.where(keep)[0]],
                    name=f"drsa/{split.lower()}", intrinsically_discrete=True,
                    meta={"tie_ratio": tie_ratio(obs), "unit": "days",
                          "known_censoring_times": True, "n_fields": int(keep.sum()),
                          "cardinality_all": card.tolist()})


# --------------------------------------------------------- simulation grid
def load_sim_grid(event_pct: int = 50, censor_pct: int = 30, n_cov: int = 5,
                  n_bins: int = 20, root: str | Path = "data/MLtoSurvival-Data"):
    """Designed simulation benchmark: event% x censor% x covariate count.

    Grid is event% in {10,30,50,70} x censor% in {0,10,30,50,70} x {5,25} covariates,
    n=3000 each.  Counting-process layout (start, stop, status) with administrative
    censoring at stop=60.

    Useful as an EXTERNAL, citable robustness design -- our own generator has known
    ground truth, but a referee will want performance mapped over event and
    censoring rates that we did not choose ourselves.
    """
    root = Path(root)
    f = root / f"{n_cov} Covariates" / f"{event_pct}E_{censor_pct}C_3000N_{n_cov}Cov.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} (grid: event 10/30/50/70, censor 0/10/30/50/70, cov 5/25)")
    df = pd.read_csv(f)
    xcols = [c for c in df.columns if c.startswith("x")] + (["z"] if "z" in df else [])
    X, names, dropped = _prepare(df, xcols)
    t = pd.to_numeric(df["stop"], errors="coerce").to_numpy()
    event = pd.to_numeric(df["status"], errors="coerce").to_numpy(dtype=np.float32)
    idx, edges = discretize_quantile(t, event, n_bins)
    return SurvData(X, idx, event, int(idx.max()) + 1, names,
                    name=f"sim/{event_pct}E_{censor_pct}C_{n_cov}cov",
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"tie_ratio": tie_ratio(t), "event_pct": event_pct,
                          "censor_pct": censor_pct, "dropped": dropped})


LOADERS["drsa/clinic"] = lambda: load_drsa("CLINIC")
LOADERS["sim/50E_30C"] = lambda: load_sim_grid(50, 30, 5)


def load_support2_competing(horizon: int = 60):
    """SUPPORT length of stay with BOTH exits, for the competing-risks model (A7).

    cause 1 = discharged alive (70.9%), cause 2 = died in hospital (25.9%),
    cause 0 = still in hospital at the horizon (genuinely censored).

    `load_support2("slos")` treats cause 2 as censoring, which assumes those
    patients would eventually have been discharged.  They would not have been,
    so that analysis overstates the discharge hazard.  Returns `cause` in
    `meta` alongside the usual event indicator so the two analyses can be run on
    identical rows.
    """
    p = _find("support2csv.zip") if _find_ok("support2csv.zip") else _find("support2.csv")
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            df = pd.read_csv(z.open("support2.csv"))
    else:
        df = pd.read_csv(p)

    X, names, dropped = _prepare(df, SUPPORT_COVARIATES)
    t = pd.to_numeric(df["slos"], errors="coerce").fillna(1).to_numpy()
    idx, beyond = discretize_native(t, horizon=horizon, origin=1)
    died = df["hospdead"].to_numpy(dtype=np.float32)

    cause = np.where(died > 0, 2, 1).astype(np.int64)
    cause[beyond] = 0                       # still in hospital at the horizon
    event = (cause > 0).astype(np.float32)

    return SurvData(X, idx, event, horizon, names, name="support2/slos-competing",
                    intrinsically_discrete=True,
                    meta={"tie_ratio": tie_ratio(t), "dropped": dropped, "unit": "days",
                          "cause": cause, "n_causes": 2,
                          "cause_names": ["discharged alive", "died in hospital"]})


LOADERS["support2/slos-competing"] = load_support2_competing


# ------------------------------------------------------- pycox benchmark set
def _pycox_cache(name: str) -> pd.DataFrame:
    """Read pycox's cached feather directly.

    pycox's own `read_df()` drops an 'Unnamed: 0' column that pandas 3.0 no
    longer creates, so flchain and nwtco raise KeyError.  The cached feather is
    the raw source and reading it ourselves sidesteps the incompatibility --
    and removes a dependency on their preprocessing besides.
    """
    import pycox
    p = Path(pycox.__file__).parent / "datasets" / "data" / f"{name}.feather"
    if not p.exists():                       # trigger pycox's download once
        from pycox import datasets
        try:
            getattr(datasets, name).read_df()
        except Exception:
            pass
    if not p.exists():
        raise FileNotFoundError(f"pycox cache for {name!r} not found at {p}")
    return pd.read_feather(p)


def _bench(name, df, xcols, dur, ev, n_bins=20, event_true=None):
    X, names, dropped = _prepare(df, [c for c in xcols if c in df.columns])
    t = pd.to_numeric(df[dur], errors="coerce").fillna(0).to_numpy()
    e = pd.to_numeric(df[ev], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    if event_true is not None:
        e = (df[ev] == event_true).to_numpy(dtype=np.float32)
    idx, edges = discretize_quantile(t, e, n_bins)
    return SurvData(X, idx, e, int(idx.max()) + 1, names, name=name,
                    intrinsically_discrete=False, bin_edges=edges,
                    meta={"tie_ratio": tie_ratio(t), "dropped": dropped})


def load_metabric(n_bins: int = 20):
    """METABRIC (n=1904) -- one of KAPLAN-HR's six benchmarks."""
    df = _pycox_cache("metabric")
    return _bench("metabric", df, [f"x{i}" for i in range(9)],
                  "duration", "event", n_bins)


def load_support_pycox(n_bins: int = 20):
    """SUPPORT as pycox preprocesses it (n=8873) -- the version the KAN survival
    papers actually benchmark on, distinct from our support2/slos cohort."""
    df = _pycox_cache("support")
    xc = [c for c in df.columns if c.startswith("x")]
    return _bench("support-pycox", df, xc, "duration", "event", n_bins)


def load_rotgbsg(n_bins: int = 20):
    """RotGBSG (n=2232) as used by KAPLAN-HR -- larger than the Vanderbilt
    gbsg_ba_ca.dat cohort (n=686) already loaded by `load_gbsg`."""
    df = _pycox_cache("gbsg")
    xc = [c for c in df.columns if c.startswith("x")]
    return _bench("rotgbsg", df, xc, "duration", "event", n_bins)


def load_flchain(n_bins: int = 20):
    """FLCHAIN (n=7874).  Free light chain assay cohort."""
    df = _pycox_cache("flchain")
    xc = ["age", "sex", "sample.yr", "kappa", "lambda", "flc.grp", "creatinine", "mgus"]
    return _bench("flchain", df, xc, "futime", "death", n_bins)


def load_nwtco(n_bins: int = 20):
    """NWTCO (n=4028).  National Wilms Tumor Study."""
    df = _pycox_cache("nwtco")
    xc = ["instit", "histol", "stage", "study", "age", "in.subcohort"]
    return _bench("nwtco", df, xc, "edrel", "rel", n_bins)


LOADERS.update({
    "metabric": load_metabric,
    "support-pycox": load_support_pycox,
    "rotgbsg": load_rotgbsg,
    "flchain": load_flchain,
    "nwtco": load_nwtco,
})


def onehot_ordinals(d: SurvData, max_card: int = 6) -> SurvData:
    """Expand low-cardinality ordinal columns into dummies.

    THIS IS REQUIRED FOR A FAIR LINEAR BASELINE.  Coding a 3-level variable as
    0/1/2 forces the effect of level 0->1 to equal that of 1->2, which is rarely
    defensible -- SUPPORT's cancer variable (none / present / metastatic) is a
    clear case.  A spline over such a column can represent the unequal spacing,
    so an all-spline KAN gets credit for repairing an encoding mistake rather
    than for discovering nonlinearity.

    Measured on support-pycox: of a +0.047 C-index "KAN advantage" over a linear
    model with 0/1/2 coding, **+0.031 (66%) was recovered by one-hot encoding
    alone**, leaving +0.016 attributable to the KAN.  Comparing against the
    unexpanded baseline overstates the KAN by roughly 3x.
    """
    cols, names = [], []
    for j, nm in enumerate(d.feature_names):
        u = np.unique(d.X[:, j])
        if 2 < len(u) <= max_card:
            for v in u[1:]:                       # drop first level as reference
                cols.append((d.X[:, j] == v).astype(np.float32))
                names.append(f"{nm}={v:g}")
        else:
            cols.append(d.X[:, j])
            names.append(nm)
    return SurvData(np.stack(cols, 1).astype(np.float32), d.bin_idx, d.event,
                    d.n_bins, names, name=d.name, entry_idx=d.entry_idx,
                    intrinsically_discrete=d.intrinsically_discrete,
                    bin_edges=d.bin_edges, meta=d.meta)


def clip_to_train_range(tr: SurvData, te: SurvData, quantile: float = 0.0):
    """Clip TEST covariates to the range seen in TRAINING, for EVERY model.

    THIS IS REQUIRED FOR A FAIR ROBUSTNESS COMPARISON, and it is the same class
    of error as `onehot_ordinals`.

    flchain's free light chains reach 25 standard deviations.  A linear
    predictor x'beta is unbounded, so the linear baseline is asked to
    extrapolate to 25 sigma and produces a per-unit NLL of 68; the KAN, whose
    spline edge functions are bounded, produces 12.9 on the same row.  Five such
    rows out of 2,362 then carried 97.8% of a "significant" mean NLL difference.

    Crediting the KAN for that is crediting it for boundedness that ANY sanely
    specified baseline also has -- winsorising is the standard remedy and costs
    the linear model nothing.  Clip both arms to the same box and the comparison
    measures what it claims to measure.

    Fitted on TRAIN only (per column min/max, or the given two-sided quantile),
    so no test information leaks.  quantile=0 clips to the exact training range;
    quantile=0.005 winsorises the training tails too, which also bounds what the
    models see during fitting.
    """
    lo = tr.X.min(axis=0) if quantile <= 0 else np.quantile(tr.X, quantile, axis=0)
    hi = tr.X.max(axis=0) if quantile <= 0 else np.quantile(tr.X, 1 - quantile, axis=0)

    def clipped(d, X):
        return SurvData(np.clip(X, lo, hi).astype(np.float32), d.bin_idx, d.event,
                        d.n_bins, d.feature_names, name=d.name,
                        entry_idx=d.entry_idx,
                        intrinsically_discrete=d.intrinsically_discrete,
                        bin_edges=d.bin_edges, meta=d.meta)

    # Training data is only altered when a quantile is requested; with
    # quantile=0 the train clip is the identity by construction.
    return clipped(tr, tr.X), clipped(te, te.X)
