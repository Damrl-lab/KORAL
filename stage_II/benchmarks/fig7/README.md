# Figure 7 Package

This folder contains the packaged SMART-only predictive benchmark assets used for the Figure 7-style Alibaba baseline comparison.

## Contents

- `fig7_sample_ids.csv`
  Compact list of the exact sample IDs and labels used in the matched Figure 7 split.

- `fig7_predictive_corpus.csv`
  Predictive corpus derived from the matched Figure 7 split. Each row includes the predictive query together with the dataset-backed failure and TTF labels.

- `manifest.json`
  Metadata for the packaged Figure 7 artifacts, including the source dataset path.

## Source Dataset

The matched Figure 7 SMART-only split is stored at:

- `dataset/alibaba/test_data/fig7_smart.csv`

This file is derived from `dataset/alibaba/test_data/smart_workload.csv` so that SMART-only baselines and SMART+Workload methods can be compared on the same 1000 windows.

## Example Command

```bash
python -m stage_II.cli \
  --dataset_type SMART_ALIBABA_FIG7 \
  --tasks predictive \
  --limit_rows 100 \
  --out_name fig7_smart_compare
```
