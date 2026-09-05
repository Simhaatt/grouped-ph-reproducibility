# Dataset access

No third-party subject-level datasets are included, including public de-identified data, restricted administrative/health records, pycox caches, DRSA files or processed subject-level extracts. Keep all acquired files outside this repository. See DATA_SOURCES.md for exact cohort definitions and unresolved provenance. Raw data must never be committed, even to a private draft repository.

Set KANREL_DATA to an external directory for SUPPORT2 and the named 2012 SPARCS CSV. Put DRSA CLINIC/MUSIC files in drsa-data subdirectories there, or set KANREL_DRSA_DATA. The isolated runner supports this external DRSA location. The pycox loader retains its original package-cache behaviour; resolve that separately as documented.

data/processed is intentionally empty and ignored. If new legally shareable aggregate summaries are approved, record their generating script and dataset terms before including them. No dataset redistribution licence is inferred from a software repository licence.
