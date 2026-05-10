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
  --alibaba_csv dataset/alibaba/test_data/smart.csv \
  --google_csv dataset/google/test_data/smart.csv \
  --workload_csv dataset/alibaba/test_data/smart_workload.csv \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name table2_fleet

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
