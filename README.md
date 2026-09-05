# Grouped Proportional Hazards under Coarse Time: Information Loss and Baseline Estimation

**Simhaa T. T. · Sudheesh Kumar Kattumannil**

Reproducibility package, version 1.0.0, published 5 September 2026. [Archived release and version DOI](https://doi.org/10.5281/zenodo.22347056). Frozen-output reproduction is available. [Outstanding scientific review items and limitations](docs/RELEASE_BLOCKERS.md) remain documented; publication does not establish a complete independent model refit.

The study separates information lost by grouping survival times from the effects of reconstructing a Cox baseline after estimating coefficients. It develops an exact finite-grid information-loss identity, a small-bin quadratic expansion, and a profiled coefficient-information result. Empirical comparisons hold the Efron Cox coefficient fixed while comparing Breslow and Kalbfleisch-Prentice/profile baselines with grouped complementary-log-log joint maximum likelihood. Event-concentration experiments distinguish concentration of failures from concentration of all exits.

The main frozen sweep contains 34 configurations from eight cohorts. The supplied paper reports 29 positive Breslow differences, 31 non-positive profile differences, and no resolved positive profile differences. An independent SPARCS analysis contains 12 additional configurations. This repository verifies those aggregate statements without downloading subject-level data.

## Install and reproduce frozen outputs

Use Python 3.14, as recorded in the original experiment log. Run commands from this repository's root. The offline workflow was installed and run successfully in a new Windows/Python 3.14 environment during preparation. The full model-refitting environment still requires validation. Pins were observed during preparation, not recovered from an original lockfile.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts/reproduce_main_results.py
```

On Linux/macOS use `.venv/bin/python` instead. Conda alternative: `conda env create -f environment.yml`, then `conda activate grouped-ph`.

Minimal command in an already installed environment:

```text
python scripts/reproduce_main_results.py
```

This reads frozen logs and regenerates CSV/LaTeX tables, PDF/PNG figures and a verification report. It does not refit experiments. Outputs go to `results/`; supplied figures under `manuscript/figures/` remain unchanged. The new figure code reconstructs the numerical content; the original plotting program for the three supplied figures was not included, so their original styling is not claimed to be reproduced exactly.

## Full experiments

```text
python -m pip install -r requirements-experiments.txt
python scripts/run_experiment.py experiments/protocol_decomp.py
python scripts/run_experiment.py experiments/simulations.py e8
python scripts/run_experiment.py experiments/simulations.py e5
python scripts/run_experiment.py experiments/kp_out_of_sample.py
```

First acquire the data described in [data/DATA_SOURCES.md](data/DATA_SOURCES.md). Set `KANREL_DATA` to your external data directory. DRSA files must be in its `drsa-data/CLINIC` and `drsa-data/MUSIC` subdirectories, or set `KANREL_DRSA_DATA`. Full commands, theory checks, cache requirements and limitations are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Each runner invocation uses a fresh ignored directory under `results/reruns/`.

## Organisation

| Path | Purpose |
|---|---|
| manuscript/ | Supplied TeX, bibliography, supplement PDF and original figures |
| src/kanrel/ | Original statistical model/data implementation; compatibility imports retained |
| src/data, models, likelihood, metrics, utils/ | Access to existing implementation |
| experiments/ | Original entry points with thematic directory guides |
| theory/ | Original numerical theory checks; layout preserved for compatibility |
| scripts/ | Frozen-output regeneration, verification, safe fresh-run launcher |
| results_raw/ | Frozen aggregate logs; documented local-path redactions only |
| archive/results_raw/ | Pointer explaining archive policy |
| results/ | Regenerated figures, tables, summaries and verification report |
| data/ | Acquisition documentation; no third-party subject-level data |
| docs/ | Experiment map, provenance, release instructions and limitations |

The exact original log bytes are kept outside the distributable repository in a private preparation archive. `docs/FILE_INVENTORY.json` records original and release-copy SHA-256 hashes and which files needed path redaction.

## Hardware, runtime and seeds

Original E24 log: Windows 11, Python 3.14.3, AMD64 processor, CPU PyTorch with one thread. Exact CPU model and RAM were not recorded. Frozen-output generation takes seconds on the preparation machine; allow a few minutes for installation. The E8 frozen log reports 1,454 seconds; model refits and secondary flexibility runs can take substantially longer. No blanket full-run runtime or memory guarantee is available.

Main experiments use split seeds 0–19, a 70/30 split and Nadeau-Bengio corrected standard errors. Independent SPARCS uses 10 splits. Simulation cells target 20 replications, but exceptions can be silently skipped by the original code. Some simulation seeds use Python `hash()` on tuples containing strings; the original hash seed is unknown. New runs fix `PYTHONHASHSEED=0`, but this does not recover the missing original seed. Frozen results retain their original values.

## Data and traceability

Original third-party datasets are not redistributed. The SPARCS loader refers to the public de-identified 2012 extract, not restricted identifiable administrative data. Restricted or uncertain-licence sources must be acquired from their providers and kept outside this repository. Access does not establish permission to redistribute. All dataset licences remain **verify before redistribution**; see source-specific notes, especially raw versus processed FLCHAIN/NWTCO caches and the DRSA MUSIC source discrepancy.

Every main figure and table has a source map in [docs/EXPERIMENT_MAP.md](docs/EXPERIMENT_MAP.md). [MANIFEST.md](MANIFEST.md) identifies exact source files and row/line references. It distinguishes verified numbers from claims still needing review. Passing the frozen checks is not evidence that all models have been independently refitted.

## Citation, licence and contact

Cite: Simhaa T. T. and Sudheesh Kumar Kattumannil (2026). *Grouped Proportional Hazards under Coarse Time: Information Loss and Baseline Estimation* (version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.22347056.

Use the **version DOI [10.5281/zenodo.22347056](https://doi.org/10.5281/zenodo.22347056)** for the paper's exact archived package. The [concept DOI](https://doi.org/10.5281/zenodo.22347055) identifies the version series. [GitHub source](https://github.com/Simhaatt/grouped-ph-reproducibility) includes post-release citation updates; the v1.0.0 tag and archived files remain unchanged. See CITATION.cff and docs/ZENODO_RECORD.json.

The authors' original material is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see LICENSE and NOTICE.md for scope. Third-party material retains its own terms. Author names, affiliations and ORCID have been confirmed by the repository owner. Contact: Simhaa T. T., simhaa2310510@ssn.edu.in. Editable supplement source is not required at the owner's direction; the supplied supplement PDF is retained for numerical provenance.

Start publication with [docs/GITHUB_ZENODO_STEPS.md](docs/GITHUB_ZENODO_STEPS.md). Do not initialise Git in the parent research folder, which contains large third-party datasets.
