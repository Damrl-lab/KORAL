# Table I Benchmark Pack

This folder contains the benchmark assets used to reproduce the Table I style evaluation in Stage II.

The benchmark package includes:

- A query bank for predictive, descriptive, prescriptive, and what-if analysis
- Dataset-aware ground-truth answers for non-predictive tasks
- Utilities to materialize benchmark CSV files from the supported Stage II datasets

## Contents

- `query_bank.json`
  Stores the benchmark queries and the reference ground truth used for evaluation. Predictive ground truth comes from the input dataset. Descriptive, prescriptive, and what-if ground truth is stored directly in the JSON file.

- `__init__.py`
  Loads the query bank, filters queries by dataset type, and materializes task-specific benchmark rows with the correct query, reference answer, and retrieval hints.

## Supported Benchmark Coverage

The query bank covers the main Stage II dataset settings, including:

- SMART-only samples
- Environmental samples
- SMART + workload samples
- SMART + environmental samples
- Workload + environmental samples
- SMART + flash-type samples
- SMART + controller-policy samples
- Mixed multimodal samples

Each materialized row keeps a unique `sample_id` so the same base sample can be evaluated against multiple benchmark queries.

## Usage

Materialize benchmark CSV files only:

```bash
python stage_II/scripts/run_table1.py \
  --dataset ENV=dataset/env/env_effects.csv \
  --materialize_only
```

Run materialization and inference:

```bash
python stage_II/scripts/run_table1.py \
  --dataset SMART_ALIBABA=dataset/alibaba/test_data/smart.csv \
  --dataset ENV=dataset/env/env_effects.csv \
  --out_name table1_run
```

## Output

The generated benchmark files are written under `stage_II/runs/<out_name>/benchmarks/`.

For non-predictive tasks, the materialized CSVs include:

- the user query
- the reference ground-truth answer
- the benchmark query id
- the benchmark task type
- retrieval terms used by the Stage II pipeline

## Notes

- Predictive evaluation uses dataset labels and regression targets from the input rows.
- Non-predictive evaluation uses the curated reference answers stored in `query_bank.json`.
- The benchmark pack is designed to be inspectable and reusable without needing to read the code first.
