# Preparation report

Prepared locally from the provided manuscript ZIP and research workspace. Nothing was pushed, published, tagged or sent to another person. No experiments were rerun.

## Files created and copied

Created root documentation/metadata, environment specifications, public module access points, thematic experiment guides, frozen-output parsers, table/figure reproduction scripts, verifier, isolated fresh-run launcher, data sources/dictionary, experiment map, manifest, release checklist and GitHub/Zenodo instructions. Machine-readable per-file inventory: FILE_INVENTORY.json; full prepared tree: FILES_CREATED.txt.

Copied original Python modules into the release tree. The model implementation lives in src/kanrel with a root compatibility namespace; only the relocated data loader's project-root calculation changes. Original experiment/theory entry-point layout is kept to preserve imports and output path behaviour. src thematic packages provide access to the existing implementation, avoiding a numerical refactor. Historical paper/figures.py is retained solely because ordering_variable.py imports its cohort regime map; it is not the new F1–F3 plotter.

## Moved, unchanged and excluded

**Moved: none.** All original workspace files and the supplied ZIP remain unchanged. 93 result/log/JSON files were preserved byte for byte in a separate private archive with hashes. Release copies of 32 of those files redact local machine paths; numerical text is unchanged and both hashes are recorded. Supplied manuscript template, bibliography, supplement PDF and figure PDFs were copied unchanged. supplement.tex is a labelled PDF wrapper, not fabricated editable source.

Excluded: all third-party raw datasets and archives, the large SPARCS CSV, DRSA subject records, KKBox archives, cached feather data, processed subject rows, Python caches, checkpoints, private logs/notes/status history and personal local paths. Publisher class/style/example files are excluded pending their separate redistribution review. Original scholarly author contact/ORCID metadata supplied in the manuscript is retained for author review.

## Findings and missing dependencies

The frozen main counts reproduce as 29/34 positive Breslow, 31/34 non-positive profile and 0/34 resolved positive profile. SPARCS T=6 matches all four requested values. The independent validation has 3 resolved profile residuals among 12 cells. All 34 supplement baseline rows and 25 flexibility differences have keyed numerical comparisons in the verifier.

E8 archived ordinal-rank correlations reproduce -0.1422/+0.8742; average-rank Spearman from rounded cells differs (-0.1469/+0.8745). This is reported, not silently corrected. Missing original PYTHONHASHSEED prevents claiming an exact E8 refit. Missing per-split/per-replication metrics limit independent variance and fit-failure checks. FLCHAIN/NWTCO raw-cache construction was recovered from upstream pycox; observed hashes and an explicit acquisition script are included. Fresh downloaded-cache identity, full experimental environment validation, editable supplement source and original F1–F3 plotting code remain unresolved. Official DRSA documentation identifies MUSIC as Last.fm rather than the legacy code comment's KKBox; this needs author source review. New figures reconstruct the numerical content and retain the supplied originals for comparison. Numerical prose not fully checked is exhaustively inventoried in NUMBER_AUDIT.csv.

## Reproduction and publication

All main tables and S1–S4: `python scripts/reproduce_tables.py`. F1/F2/F3 PDF and PNG: `python scripts/reproduce_figures.py`. Combined offline rebuild and checks: `python scripts/reproduce_main_results.py`. Full experiments and their inputs: REPRODUCIBILITY.md and EXPERIMENT_MAP.md. Exact Git init/commit/push commands and the ordered GitHub → Zenodo connection → v1.0.0 release → DOI workflow: GITHUB_ZENODO_STEPS.md.

This is a prepared draft with an explicit release gate, not a claim of publication readiness. Verification results are recorded in results/verification.json; publication-specific checks intentionally fail while decisions and provenance remain open.

## Completed validation

Offline requirements installed successfully in a fresh Python 3.14 Windows environment. The combined reproduction command regenerated tables and figures and passed 24/24 checks. Both generated PNGs and rendered PDF figures were visually reviewed. Official CFF 1.2.0 schema validation passed; name segmentation remains an author metadata decision. All 93 original raw output hashes still match their private preserved copies. Full model refits and full experiment-environment installation were not performed.
