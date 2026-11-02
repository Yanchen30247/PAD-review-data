# Perceived aggressive driving and road safety: data and code

Extracted data and analysis code for the systematic review and meta-analysis "Perceived aggressive driving and road safety: A systematic review and meta-analysis" by Yan Chen, Anthony D. McDonald, and Yunmei Liu. Study characteristics, quality ratings, and all pooled estimates are in Supplementary Tables S1 to S3 of the article. Section 3 of the article describes how the data were extracted, verified, and analyzed.

Contact: Yunmei Liu, yunmei.liu@louisville.edu

## Contents

| Path | Content |
| --- | --- |
| `analysis/extracted_effects/` | Extracted effect sizes, one file for each part of the evidence (RQ1, RQ2, RQ4, RQ5) |
| `analysis/verification/` | Second-verification records; `corrections.py` applies the corrections they contain |
| `analysis/rq3_frequency/` | Behavior frequency and co-occurrence data for RQ3, and the within-sample comparison of violative and non-violative frequency |
| `analysis/results/all_effects_converted.csv` | All extracted rows after conversion to r and log OR, with the analysis flags |
| `analysis/results/results_table_s3.csv` | Table S3 of the article at full precision |
| `analysis/results/vote_counts.csv`, `vote_by_study.csv` | Direction-of-effect synthesis, by factor and by study |
| `analysis/results/rq5_*.csv` | Path models for RQ5 |
| `analysis/results/sensitivity_runs.csv` | Sensitivity analyses in Section 4.7 |
| `analysis/figures/figure_values.csv` | Values drawn in Figures 5 to 9 |
| `analysis/*.py` | Analysis code |

Scripts write Figures 5 to 10 to `analysis/figures/`. The source articles are not included because of copyright. The columns `file` and `line` point to working extraction files that reproduce passages of the source articles, so these files are not shared either.

## Reproducing the results

Tested with Python 3.12.4:

```
pip install -r requirements.txt
cd analysis
python run_all.py
```

## Main columns in the effect-size files

| Column | Content |
| --- | --- |
| `agent`, `es_id` | Extraction batch and row identifier |
| `study_id`, `sample_id` | Study and independent sample |
| `rq` | Research question |
| `x_stage`, `y_stage` | Stage of each variable: S1 triggering factors, S2i intentional processes, S2u unintentional processes, S3 PAD behaviors, S4 safety outcomes |
| `x_code`, `x_sub`, `x_label` | Construct code, subcode, and label of the first variable |
| `y_code`, `y_label` | Code and label of the second variable; V violative, NV non-violative, CRASH_SELF self-reported crash, CRASH_REC recorded crash |
| `x_source`, `y_source` | Measurement: self-report or observed or recorded |
| `design`, `level` | Study design; driver level or crash level |
| `stat_type` | Reported statistic |
| `r_value`, `m1` to `n2`, `d_value`, `t_value`, `F_value`, `df`, `chi2`, `a` to `d_cell`, `p1`, `p2`, `or_value`, `or_lo`, `or_hi`, `n_total` | Values as reported in the source |
| `sign_x`, `sign_y` | Risk alignment of each variable (Section 3.3.1) |
| `adjusted`, `latent`, `zero_order`, `primary` | Type of estimate and use in the primary analyses |
| `direction`, `significant` | Direction and significance for the direction-of-effect synthesis |
| `value_text` | Short quotation of the value from the source |
| `note` | Coding notes |

Added in `all_effects_converted.csv`: `r`, `z`, `vz` (correlation, Fisher z, sampling variance), `logor`, `vlogor` (log odds ratio and its sampling variance), `n_eff`, `conv` (conversion used), and the flags used to select effect sizes.

## License

The data are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The code is licensed under the MIT License.
