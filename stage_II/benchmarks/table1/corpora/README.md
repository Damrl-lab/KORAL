# Pre-Materialized Table I Corpora

This folder contains compact pre-materialized benchmark corpora for the Alibaba Stage II evaluation settings.

## Purpose

These files provide a stable, committed ground-truth package for users who need to inspect or reuse the evaluation corpus without regenerating it from temporary run outputs.

## Included Corpora

- `smart_alibaba_*.csv`
  SMART-only benchmark corpus derived from `dataset/alibaba/test_data/smart.csv`

- `smart_workload_*.csv`
  SMART+Workload benchmark corpus derived from `dataset/alibaba/test_data/smart_workload.csv`

- `manifest.json`
  File list, source dataset paths, and row counts for each task

## Schema

Each compact corpus row includes:

- sample identifiers and source row keys
- dataset and benchmark metadata
- the task query text
- the scenario text for what-if rows
- the row-aligned reference text for non-predictive tasks
- dataset-backed labels for predictive rows

These corpora are compact on purpose. They are designed for sharing, review, and reproducibility, while the full raw 30-day telemetry remains in the source datasets.
