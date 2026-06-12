KORAL Stage II
================================================

This package provides Stage II pipeline for SSD operational analysis.

Stage II has TWO modes:
  (A) Per-sample mode (Table I): analyze one drive/window at a time.
  (B) Fleet mode (Table II): analyze a cohort (e.g., 100 drives) in a single call.

------------------------------------------------
A) Per-sample mode (Table I style)
------------------------------------------------
Per-sample mode:
1) Reads a prepared input CSV (SMART / SMART+Workload / SMART+Env / etc.)
2) Builds an Intermediate Representation (IR) for SMART + optional modalities
3) Materializes a lightweight DataKG artifact per sample (TTL if rdflib is available)
4) Retrieves lightweight evidence from a Literature KG TTL (SPARQL via rdflib when available)
5) Calls GPT-4o (OpenAI Chat Completions) for:
     - predictive
     - descriptive
     - prescriptive
     - what-if
6) Records responses + computes metrics:
     Predictive: Precision/Recall/Accuracy (+ optional TTF_MSE, TL_MSE)
     Text: BLEU-4 (B4), ROUGE-L (RL)
     Grounding: FiP for descriptive/prescriptive, CFV for what-if

How to run (per-sample)
-----------------------
1) Install dependencies:
   pip install pandas numpy requests
   (optional but recommended) pip install rdflib

2) Export OpenAI key:
   export OPENAI_API_KEY="sk-..."

3) Run:
   python -m stage_II.cli \
     --dataset_type SMART_ALIBABA \
     --input_csv dataset/alibaba/test_data/smart.csv \
     --tasks predictive,descriptive,prescriptive,whatif \
     --limit_rows 100 \
     --out_name demo_smart

   For SMART+Workload Alibaba, use:
   python -m stage_II.cli \
     --dataset_type SMART_WORKLOAD \
     --input_csv dataset/alibaba/test_data/smart_workload.csv \
     --tasks predictive,descriptive,prescriptive,whatif \
     --limit_rows 100 \
     --out_name demo_smart_workload

   For Figure 7-style SMART-only predictive comparison on the same rows as
   the workload-aware package, use:
   python -m stage_II.cli \
     --dataset_type SMART_ALIBABA_FIG7 \
     --tasks predictive \
     --limit_rows 100 \
     --out_name fig7_smart_compare

   Predictive evaluation note:
   - `failure`, `label`, `ttf_days`, and tail-latency targets are stripped from
     the LLM-visible payload before prompt construction.
   - Precision/recall/accuracy are scored over the full labeled split.
   - Null, invalid, or missing `predicted_failure` outputs are counted as 0 and
     reported separately in the summary.

Outputs (per-sample)
--------------------
stage_II/runs/<RUN_NAME>/
  input_samples.csv
  responses.jsonl
  metrics_per_sample.csv
  metrics_summary.json
  data_kg_ttl/<sample_id>.ttl   (if rdflib available)

Table I benchmark materialization
---------------------------------
Use the curated query bank and ground-truth templates:

python stage_II/scripts/run_table1.py \
  --dataset ENV=dataset/env/env_effects.csv \
  --materialize_only

This writes benchmark CSVs with per-query references. Omit
--materialize_only to also run KORAL inference and aggregate a
Table I-style results CSV.

Committed compact corpora and query-coverage manifests live at:

- `stage_II/benchmarks/table1/corpora/`
- `stage_II/benchmarks/table1/query_coverage.json`

------------------------------------------------
B) Fleet mode (Table II style)
------------------------------------------------
Fleet mode performs collective analysis over N drives at once (one LLM call per task per cohort).

Supported datasets (as configured for this project request)
----------------------------------------------------------
- SMART_ALIBABA         (Alibaba SMART-only, no app)
- SMART_GOOGLE          (Google SMART-only)
- SMART_WORKLOAD        (Alibaba SMART + app workload tag)

Fleet mode workflow
-------------------
1) Reads the input CSV and selects N drives per cohort (de-duplicates by disk_id/drive_id if possible).
2) Builds per-drive compact IR summaries (top signals) and fleet-wide aggregates.
3) Materializes a Fleet DataKG artifact (TTL if rdflib is available).
4) Retrieves literature evidence from the global LitKG TTL.
5) Calls GPT-4o once per task for the entire cohort.
6) Computes metrics:
   - Predictive: precision/recall/accuracy at drive-level by comparing predicted_failing_drives vs GT labels.
   - Grounding: FiP (descriptive/prescriptive), CFV (what-if).
   - (Optional) Text overlap: B4/RL if you provide fleet reference columns (rare in practice).

How to run (fleet)
------------------
Example: 100-drive cohorts, 5 cohorts:

python -m stage_II.fleet_cli \
  --dataset_type SMART_ALIBABA \
  --input_csv dataset/alibaba/test_data/smart.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name fleet_alibaba_100x5

For SMART+Workload Alibaba, use:

python -m stage_II.fleet_cli \
  --dataset_type SMART_WORKLOAD \
  --input_csv dataset/alibaba/test_data/smart_workload.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name fleet_smart_workload_100x5

If you need the matched SMART-only Figure 7 split for baseline predictors,
the packaged sample IDs and predictive corpus are under:

- `stage_II/benchmarks/fig7/`

Fleet outputs
-------------
stage_II/runs/<RUN_NAME>/
  cohort_composition.csv
  responses_fleet.jsonl
  metrics_fleet.csv
  metrics_summary_fleet.json
  fleet_kg_ttl/<cohort_id>.ttl   (if rdflib available)

Table II results generation (all 3 datasets)
--------------------------------------------
Use the provided script (stage_II/scripts/run_table2_fleet.py):

python stage_II/scripts/run_table2_fleet.py \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_dir_name table2_fleet

The script uses dataset defaults from stage_II/config.py, where
SMART_WORKLOAD now points to:
  dataset/alibaba/test_data/smart_workload.csv

To regenerate the packaged Alibaba evaluation assets, run:

python stage_II/scripts/package_alibaba_eval_assets.py

This creates:
  stage_II/runs/table2_fleet/table_II_fleet_results.csv
and keeps detailed run artifacts under stage_II/runs/table2_fleet/.

Notes on input CSV schema
-------------------------
- SMART columns: any header matching r_<number> will be treated as SMART.
  Values can be scalar or a JSON list string like "[...]" for 30-day windows.
- Labels:
    - classification ground truth: 'failure' or 'label' (0/1)
    - optional regression: 'ttf_days' and 'tail_latency_ms'
- Workload (SMART_WORKLOAD):
    - expects 'app' column (Alibaba workload tag)
- Packaged Alibaba split guidance:
- `smart.csv`: standalone SMART-only 1000-row set
- `smart_workload.csv`: standalone SMART+Workload 1000-row set
- `fig7_smart.csv`: SMART-only matched split derived from `smart_workload.csv`

Reference packages committed in the repo:
- `dataset/alibaba/test_data/evaluation_package.json`
- `stage_II/benchmarks/fig7/`
- `stage_II/benchmarks/table1/corpora/`
- `stage_II/benchmarks/table1/query_coverage.json`
