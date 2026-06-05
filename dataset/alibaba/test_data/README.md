# Alibaba Evaluation Splits

This folder contains the Alibaba test datasets used by the Stage II evaluation package.

## Files

- `smart.csv`
  Standalone SMART-only 1000-row evaluation set. This file is balanced across healthy and failed windows and is used for the SMART-only path.

- `smart_workload.csv`
  Standalone SMART+Workload 1000-row evaluation set. This file contains the same SMART-style 30-day windows plus the Alibaba `app` workload label.

- `fig7_smart.csv`
  SMART-only matched split derived from `smart_workload.csv`. It preserves the exact same 1000 rows and sample IDs as `smart_workload.csv`, but the workload label is blanked so Figure 7-style SMART-only predictive baselines can be compared on the same sample set.

- `smart_workload_all.csv`
  Larger workload-aware source pool used to build the compact 1000-row package.

- `evaluation_package.json`
  Manifest describing row counts, balance, and the relationship between the packaged files.

## Important Note

`smart.csv` and `smart_workload.csv` are different balanced 1000-row sample sets. The matched apples-to-apples SMART-only split for `smart_workload.csv` is `fig7_smart.csv`, not `smart.csv`.

## Recommended Usage

- Figure 7-style SMART-only predictive comparison:
  use `fig7_smart.csv`

- SMART-only Table I / demo runs:
  use `smart.csv`

- SMART+Workload Table I / Table II / FCM multimodal runs:
  use `smart_workload.csv`

## Example Commands

Figure 7-style matched SMART-only predictive run:

```bash
python -m stage_II.cli \
  --dataset_type SMART_ALIBABA_FIG7 \
  --tasks predictive \
  --limit_rows 100 \
  --out_name fig7_smart_compare
```

SMART+Workload per-sample run:

```bash
python -m stage_II.cli \
  --dataset_type SMART_WORKLOAD \
  --input_csv dataset/alibaba/test_data/smart_workload.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --limit_rows 100 \
  --out_name demo_smart_workload
```

SMART+Workload fleet run:

```bash
python -m stage_II.fleet_cli \
  --dataset_type SMART_WORKLOAD \
  --input_csv dataset/alibaba/test_data/smart_workload.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name fleet_smart_workload_100x5
```
