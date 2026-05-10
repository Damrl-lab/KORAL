# Table I Benchmark Pack

This folder contains the curated query bank and ground-truth materializer used to reproduce the per-sample `Table I` style evaluation for KORAL.

Files:

- `query_bank.json`: curated predictive, descriptive, prescriptive, and what-if queries. Non-predictive entries now carry full human-readable `ground_truth` answers directly in the JSON itself, while predictive stays dataset-backed.
- Query coverage: `1` predictive query plus `39` descriptive, `39` prescriptive, and `39` what-if queries.
- The benchmark content is grounded in:
  - Alibaba SSD reliability/correlation study (`fast21-han.pdf`)
  - temperature/humidity SSD study (`temphum.pdf`)
  - vibration SSD study (`vib.pdf`)
  - the KORAL paper evaluation intent
- `__init__.py`: materializes query-specific benchmark CSVs with row-level reference texts (`ref_descriptive`, `ref_prescriptive`, `ref_whatif`) and retrieval hints.

Design notes:

- Predictive uses one benchmark question per sample.
- Descriptive, prescriptive, and what-if use multiple curated questions per sample.
- Ground-truth text is declared directly in `query_bank.json` as readable benchmark answers so the file is self-contained for inspection and reuse.
- The bank now includes SMART-only, workload-aware, SMART+Env, workload+Env, flash-type-aware, and controller-policy-aware queries in addition to the environmental and vibration prompts.
- Every materialized row gets a unique `sample_id` so Stage II can evaluate repeated queries against the same base telemetry window.

Typical usage:

```bash
python stage_II/scripts/run_table1.py \
  --dataset ENV=dataset/env/env_effects.csv \
  --materialize_only
```

To run inference as well, omit `--materialize_only` and pass additional datasets as needed.
