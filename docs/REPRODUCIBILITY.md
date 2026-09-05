# Reproducibility

There are two distinct workflows. Frozen-output reproduction is offline and does not refit models; fresh experiments require the original data access and preprocessing. Neither workflow may overwrite results_raw.

## Frozen outputs

```text
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
python scripts/verify_repository.py
```

Or run `python scripts/reproduce_main_results.py` for all three. The main table is results/main/baseline_decomposition.csv. Supplement S1–S4 have generated TeX tables under results/supplementary. Representative rows and independent SPARCS have separate TeX tables. All numerical formatting is derived from frozen logs. F1–F3 are written as both PDF and PNG under results/figures; review these before using them in the manuscript.

The verification script checks archive hashes, all expected configurations, stored split counts, the corrected-SE relationship within printed rounding uncertainty, headline counts, exact SPARCS values, E8 dimensions/correlations, all 170 numerical cells in supplied supplement S1/S2 and all 25 S3/S4 paired differences with SEs. Generated CSVs are compared cell by cell with a fresh parse. It scans common secret/path signatures; this is not a substitute for a human data-rights review.

## Fresh experiments

Install requirements-experiments.txt and acquire documented external data. In PowerShell set `$env:KANREL_DATA` to your own absolute external data directory. No example personal filesystem path is distributed. The original pycox loader specifically reads its installation's datasets/data/*.feather caches; setting PYCOX_DATA_DIR alone will not repair that assumption. The raw-cache acquisition route is implemented in scripts/acquire_pycox_data.py; observed hashes and row counts are recorded in docs/DATA_CACHE_MANIFEST.json. Run it in an external environment after provider-terms review; it refuses to cache data inside this repository.

```text
python scripts/run_experiment.py experiments/protocol_decomp.py
python scripts/run_experiment.py experiments/ordering_variable.py
python scripts/run_experiment.py experiments/kp_out_of_sample.py
python scripts/run_experiment.py experiments/simulations.py e8
python scripts/run_experiment.py experiments/simulations.py e5
python scripts/run_experiment.py experiments/simulations.py e6c
python scripts/run_experiment.py experiments/simulations.py e6d
python scripts/run_experiment.py theory/verify_lemmas.py
python scripts/run_experiment.py theory/verify_psi.py
python scripts/run_experiment.py theory/verify_profiled_info.py
python scripts/run_experiment.py experiments/robustness_suite.py e24 e20 e21 e23 e22 e19
python scripts/run_experiment.py experiments/flexibility.py
```

The runner creates a new ignored working directory for each command, copies code and frozen prerequisites there, fixes PYTHONHASHSEED=0 for new runs, and launches the unchanged numerical algorithm. A run_metadata.json records the command and interpreter. Original scripts may write files beside themselves; isolation protects the frozen repository. Some original scripts catch exceptions and return success, so inspect the resulting logs for failed fits and omitted cells. Different runs do not automatically replace each other's prerequisites: ordering_variable initially reads the frozen main sweep. To analyse a new sweep, explicitly copy its output into that fresh run's experiment directory and document the new provenance.

## Scientific limits

The main log stores rounded aggregate means and SEs, not all per-split NLLs. The verifier checks the NB multiplier inferred from the two printed SE columns, not an independent recomputation from missing split-level values. Signed rounded zero cannot distinguish exact zero from a tiny negative value. E8 ordinal-rank statistics and tie-aware Spearman are both reported in results/main/summary.json. The historical E8 PYTHONHASHSEED is unrecoverable from available files. E5 targets 20 replications but successful counts are not logged and failures may be skipped. Hardware record: Python 3.14.3, Windows 11, AMD64, CPU PyTorch 2.12.0, one thread; RAM and exact CPU model unknown. E8 log runtime: 1,454 s. Fresh-run total duration is unknown.

## Manuscript build

The supplied manuscript requires a TeX distribution, natbib, booktabs, graphicx, amsmath, amssymb, placeins and the publisher's svjour3 class/style files. Obtain svjour3.cls, svglov3.clo and spbasic.bst from the authorised publisher template (or your original ZIP for local use); these are excluded pending redistribution review. With those installed locally, run from manuscript/:

```text
pdflatex template.tex
bibtex template
pdflatex template.tex
pdflatex template.tex
```

The supplement wrapper requires pdfpages and merely includes the original PDF; it does not recover editable proof source. No manuscript numerical values were edited. The supplied manuscript refers to original PDFs; switching to regenerated figures requires explicit author review of appearance.

## Preparation validation record

The offline environment was installed from requirements.txt in a new Windows/Python 3.14 virtual environment. All 24 frozen-output checks passed after complete figure/table regeneration. requirements-lock.txt records all packages in that tested offline environment. This does not validate the much larger full experiment environment.
