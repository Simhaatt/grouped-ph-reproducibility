# Data dictionary

The authoritative executable definitions are in src/kanrel/data.py. Dataset-specific inputs, variable names, outcomes, event coding and exclusions are listed in data/DATA_SOURCES.md. Encoded DRSA fN and pycox xN labels cannot be assigned clinical meanings without the provider dictionary.

| Field | Meaning |
|---|---|
| X | n by p float32 covariate matrix after original preprocessing |
| bin_idx | Zero-based interval containing exit; integer from 0 to T-1 |
| event | 1 for the loader-defined event, 0 for censoring; see source-specific coding |
| n_bins / T | Number of analysis intervals |
| entry_idx | Optional zero-based delayed-entry index |
| bin_edges | Optional fine-grid boundaries; removed by original coarsen operation |
| cohort | Exact frozen cohort identifier; use with T as the result key |
| n | Whole cohort size before repeated train/test splitting |
| splits | Number of successful full-cohort splits reported in the main log |
| modal_all | Largest number of exits in one interval divided by n |
| modal_event | Largest number of event exits in one interval divided by total events |
| D_B | Test NLL(Efron coefficient + Breslow baseline) minus NLL(grouped joint MLE) |
| D_KP | Same difference using profile/Kalbfleisch-Prentice baseline |
| SE_B / SE_KP | Nadeau-Bengio corrected repeated-split SE, not a confidence interval half-width |
| D_base | Baseline contribution D_B - D_KP |
| D_coef | Residual coefficient contribution |
| bias_cox_pct / bias_grouped_pct | Mean absolute relative coefficient bias, percentage |
| difference | Supplement matched KAN versus selected GAM, oriented positive for KAN |
| source_line, line_B, line_KP, ordering_line | One-based lines in the named frozen source files |

Correction: SE_NB = sqrt((1/K + 0.3/0.7) * sample_variance). A positive difference is declared resolved when D > 1.96 SE_NB. Printed values may contain signed zero. No per-subject identifiers are present in generated result tables.
