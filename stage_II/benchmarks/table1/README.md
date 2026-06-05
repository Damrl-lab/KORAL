# Table I Benchmark Pack

This folder contains the benchmark assets used to reproduce the Table I style evaluation in Stage II.

The benchmark package includes:

- A query bank for predictive, descriptive, prescriptive, and what-if analysis
- Dataset-aware ground-truth answers for non-predictive tasks
- Utilities to materialize benchmark CSV files from the supported Stage II datasets
- A committed compact corpus package for Alibaba SMART and SMART+Workload evaluation
- A query coverage manifest that identifies which modality and subsystem families each query exercises

## Contents

- `query_bank.json`
  Stores the benchmark queries and the reference ground truth used for evaluation. Predictive ground truth comes from the input dataset. Descriptive, prescriptive, and what-if ground truth is stored directly in the JSON file.

- `__init__.py`
  Loads the query bank, filters queries by dataset type, and materializes task-specific benchmark rows with the correct query, reference answer, and retrieval hints.

- `corpora/`
  Stable compact benchmark corpora with row-level ground truth that can be shared directly with collaborators.

- `query_coverage.json`
  Query coverage manifest with modality and subsystem labels for each benchmark query.

## Supported Benchmark Coverage

The query bank covers the main Stage II dataset settings, including:

- SMART-only samples
- matched Figure 7 SMART-only samples
- Environmental samples
- SMART + workload samples
- SMART + environmental samples
- Workload + environmental samples
- SMART + flash-type samples
- SMART + controller-policy samples
- Mixed multimodal samples

Each materialized row keeps a unique `sample_id` so the same base sample can be evaluated against multiple benchmark queries.

For the committed Alibaba package, the compact corpora live in `corpora/`, while the full raw telemetry remains in `dataset/alibaba/test_data/`.

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

Regenerate the packaged Alibaba evaluation assets:

```bash
python stage_II/scripts/package_alibaba_eval_assets.py
```

Materialize the matched Figure 7 SMART-only split:

```bash
python stage_II/scripts/run_table1.py \
  --dataset SMART_ALIBABA_FIG7=dataset/alibaba/test_data/fig7_smart.csv \
  --tasks predictive \
  --materialize_only
```

## Output

The generated benchmark files are written under `stage_II/runs/<out_name>/benchmarks/`.

The committed compact corpora are written under `stage_II/benchmarks/table1/corpora/`.

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
- The query coverage manifest can be used to filter or scope comparisons when a baseline only models a subset of subsystems.
