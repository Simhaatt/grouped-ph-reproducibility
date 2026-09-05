# Data sources and cohort construction

All entries: **redistribution permission = verify before redistribution**. No third-party raw data are distributed. Source citations and identifiers below come from the supplied bibliography and loader code. Access availability is not a licence finding. The [official pycox documentation](https://github.com/havakv/pycox) and [SPARCS access overview](https://www.health.ny.gov/statistics/sparcs/) were consulted during preparation. DRSA download availability and terms still require provider review.

## METABRIC

- Manuscript cohort: `metabric`.
- Original source/citation: Curtis et al. (2012), Nature; curtis2012metabric.
- Official source or DOI: https://doi.org/10.1038/nature10983
- Access/input: pycox METABRIC benchmark cache metabric.feather. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: x0–x8.
- Time/outcome/event: duration; event column (death event; verify coding with benchmark source).
- Final analysed n in frozen records: 1904.
- Analysis grids T: 4,6,10,20.
- Preprocessing/discretisation: load_metabric / _bench; finer recorded durations grouped by event-quantile grid then coarsen.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## SUPPORT (pycox)

- Manuscript cohort: `support-pycox`.
- Original source/citation: SUPPORT investigators (1995); support1995; pycox benchmark preprocessing.
- Official source or DOI: https://github.com/havakv/pycox
- Access/input: support.feather, distinct from support2/slos. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: all x-prefixed columns.
- Time/outcome/event: duration; event (death in benchmark).
- Final analysed n in frozen records: 8873.
- Analysis grids T: 4,6,10,20.
- Preprocessing/discretisation: load_support_pycox / _bench; event-quantile grid then coarsen.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## SUPPORT2

- Manuscript cohort: `support2/slos`.
- Original source/citation: SUPPORT investigators (1995); Vanderbilt distribution; vanderbiltSupport2026.
- Official source or DOI: https://hbiostat.org/data/
- Access/input: support2.csv or support2csv.zip containing support2.csv. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: age, sex, dzgroup, dzclass, num.co, scoma, race, sps, aps, meanbp, hrt, resp, temp, wblc, crea, sod, diabetes, dementia, ca, adlsc.
- Time/outcome/event: slos in days; event = 1 - hospdead, discharge alive; beyond 60-day horizon censored.
- Final analysed n in frozen records: 9105.
- Analysis grids T: 6,10,15,20,30,60.
- Preprocessing/discretisation: load_support2(slos,horizon=60); native-day bins origin 1, coarsen fixed cohort. In-hospital death treated as censoring; competing-risk limitation retained.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## FLCHAIN

- Manuscript cohort: `flchain`.
- Original source/citation: Dispenzieri et al. (2012); dispenzieri2012flchain.
- Official source or DOI: https://doi.org/10.1016/j.mayocp.2012.03.009
- Access/input: raw flchain.feather cached by pycox from the survival package via Rdatasets; use processed=False. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: age, sex, sample.yr, kappa, lambda, flc.grp, creatinine, mgus.
- Time/outcome/event: futime; death.
- Final analysed n in frozen records: 7874.
- Analysis grids T: 4,6,10,20.
- Preprocessing/discretisation: load_flchain / _bench; reads raw n=7874 cache. The processed pycox frame drops missing-creatinine rows to n=6524; do not use that frame for this paper. The raw-cache mechanism is confirmed in upstream pycox from_rdatasets.py.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## NWTCO

- Manuscript cohort: `nwtco`.
- Original source/citation: Breslow and Chatterjee (1999); breslow1999nwtco.
- Official source or DOI: https://doi.org/10.1111/1467-9876.00165
- Access/input: raw nwtco.feather cached by pycox from survival via Rdatasets; use processed=False. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: instit, histol, stage, study, age, in.subcohort.
- Time/outcome/event: edrel; rel (relapse indicator).
- Final analysed n in frozen records: 4028.
- Analysis grids T: 4,6,10,20.
- Preprocessing/discretisation: load_nwtco / _bench; raw-cache columns retained, unlike default pycox processing; event-quantile grid then coarsen.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## DRSA CLINIC

- Manuscript cohort: `drsa/clinic`.
- Original source/citation: Ren et al. (2019); ren2019drsa.
- Official source or DOI: https://github.com/rk2900/DRSA
- Access/input: CLINIC/featindex.txt and CLINIC/train.yzbx.txt; verify current download and original dataset rights. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: recovered categorical fields fN with nonzero variance and cardinality <=100; further low-cardinality one-hot encoding.
- Time/outcome/event: observed time min(z,b); event = 1[z<=b], censored beyond native horizon.
- Final analysed n in frozen records: 4828.
- Analysis grids T: 5,10,25,50.
- Preprocessing/discretisation: load_drsa(CLINIC); encoded field semantics not supplied. Retrieve original variable dictionary; do not infer clinical labels.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## DRSA MUSIC

- Manuscript cohort: `drsa/music`.
- Original source/citation: Ren et al. (2019); ren2019drsa; Last.fm-derived MUSIC benchmark per official DRSA README.
- Official source or DOI: https://github.com/rk2900/DRSA
- Access/input: MUSIC/featindex.txt and MUSIC/train.yzbx.txt; upstream music/churn data terms require review. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: same encoded field selection as CLINIC; original labels unavailable.
- Time/outcome/event: observed time min(z,b); event = 1[z<=b].
- Final analysed n in frozen records: 50000 (first 50000 training-file rows, not a random sample).
- Analysis grids T: 6,12,30,60.
- Preprocessing/discretisation: load_drsa(MUSIC,max_rows=50000); native-time coarsening; data ordering affects selected cohort.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## SPARCS 2012 public de-identified

- Manuscript cohort: `sparcs/drg302; validation drg640,560,540,720`.
- Original source/citation: New York State Department of Health; nysdoh2012sparcs.
- Official source or DOI: https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/3m9u-ws8e
- Access/input: 2012 public extract exported as Hospital_Inpatient_Discharges_(SPARCS_De-Identified)__2012_20260821.csv. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: Age Group, Gender, APR Severity of Illness Code, APR Risk of Mortality, Type of Admission, Emergency Department Indicator; APR DRG Code for selection; Facility ID for aggregate count.
- Time/outcome/event: Length of Stay in whole days, 120+ top code; event=1 for exit within horizon, otherwise administrative censoring; this loader does not distinguish discharge alive from death.
- Final analysed n in frozen records: 34233 main; each validation group capped at 50000.
- Analysis grids T: main 6,10,15,30; validation 6,15,30.
- Preprocessing/discretisation: load_sparcs filters APR DRG; strips plus sign and fills unparsed LOS with 120; native bins horizon=30 then coarsen. kp_out_of_sample caps cohorts. No restricted SPARCS files are included.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## Rotterdam/GBSG

- Manuscript cohort: `rotgbsg (supplement only)`.
- Original source/citation: Foekens et al. (2000), Schumacher et al. (1994), DeepSurv benchmark; bibliography keys foekens2000rotterdam, schumacher1994gbsg, katzman2018deepsurv.
- Official source or DOI: https://github.com/havakv/pycox
- Access/input: gbsg.feather benchmark cache. Obtain under the provider's current terms; credentials are never included.
- Redistribution: **verify before redistribution**; excluded from package.
- Variables: all x-prefixed columns.
- Time/outcome/event: duration; event, per benchmark coding (confirm original outcome semantics).
- Final analysed n in frozen records: 2232.
- Analysis grids T: 20-bin loader default; flexibility run configuration is authoritative.
- Preprocessing/discretisation: load_rotgbsg / _bench; distinct from the smaller Vanderbilt GBSG cohort.
- Exclusions: `_prepare` drops covariate columns with >25% missingness, median-imputes numeric columns, one-hot encodes categoricals; it does not apply a general complete-case row exclusion. Loader-specific selection and horizon censoring are described above; exact source-row provenance is not frozen. See `src/kanrel/data.py` and `experiments/crossover.py` for executable definitions.

## Shared processing and unresolved records

`discretize_quantile` builds a grid using event-time quantiles; `coarsen` maps an old bin to floor(old_bin * T_new / T_old). This differs from independently recomputing every grid. The original preprocessing (including initial imputation and grid construction) occurs before the repeated train/test split; standardisation and clipping then use training information. This is described as implemented, not retrospectively repaired. Full retained feature lists, source file hashes and cache-generation procedures were not supplied as project scripts. The pycox mechanism was recovered from its source and scripts/acquire_pycox_data.py now documents the explicit raw-cache acquisition route. Observed hashes, columns and row counts are in docs/DATA_CACHE_MANIFEST.json. Fresh downloads must be checked against these records. Restricted versions of any dataset require provider authorisation and must never be placed in this repository.

## Explicit pycox acquisition

Use an external Conda/Python environment for full data experiments; a virtual environment inside this repository would place pycox's default raw cache inside the repository. After reviewing provider terms, unset PYCOX_DATA_DIR and run `python scripts/acquire_pycox_data.py`. It refuses to create an in-repository cache, requests raw FLCHAIN/NWTCO frames with processed=False, verifies cohort sizes and reports hashes. It was not run to download new data during preparation.

Sources: [pycox raw/processed loader code](https://raw.githubusercontent.com/havakv/pycox/master/pycox/datasets/from_rdatasets.py), [official DRSA data specification](https://github.com/rk2900/DRSA#data-preparation). DRSA supplies split archives; its README identifies MUSIC as Last.fm, not KKBox. The original loader's KKBox and day-unit comments therefore require source review. The 2012 SPARCS URL is taken from the supplied bibliography; direct automated access failed during preparation, so navigate from the official SPARCS overview if it is unavailable.
