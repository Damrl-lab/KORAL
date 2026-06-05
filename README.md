# KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis

KORAL is a **two-stage pipeline** for SSD operational analysis:

- **Stage I (Literature KG):** Extract an **evidence-backed knowledge graph** from SSD research papers, aligned to a curated SSD taxonomy.
- **Stage II (Operational Analysis):** Summarize telemetry (SMART, workload, environment, etc.) using a rule base, retrieve relevant literature evidence from the Stage I KG, and call an LLM to perform SSD analysis (**predictive / descriptive / prescriptive / what-if**) with automatic evaluation.
- **Stage II Fleet Mode (Table II):** Run **collective / fleet-level analysis** over a cohort (e.g., **100 drives at once**) and compute fleet metrics.

---

## Repository layout

```text
KORAL/
├─ data_preparation/                  # Data prep scripts (Alibaba/Google/env/workload)
├─ dataset/                           # Datasets (Alibaba, Google, env, fio_workload, ...)
├─ stage_I/                           # Stage I: paper → LitKG pipeline
│  ├─ out/                            # Stage I outputs (TTL/JSON/global KG)
│  ├─ __init__.py
│  ├─ ssd_cot_prompt.txt              # Stage I extraction prompt (strict JSON)
│  ├─ ssd_kg_pipeline.py              # Stage I pipeline (papers → TTL/JSON/global KG)
│  └─ taxonomy.json                   # SSD taxonomy (vocabulary)
├─ stage_II/                          # Stage II: operational pipeline + evaluation
│  ├─ evaluation/
│  ├─ features/
│  ├─ kg/
│  ├─ llm/
│  ├─ prompts/
│  ├─ scripts/
│  ├─ utils/
│  ├─ cli.py                          # per-sample Stage II CLI (Table I style)
│  ├─ pipeline.py                     # per-sample pipeline runner
│  ├─ fleet_cli.py                    # fleet-level Stage II CLI (Table II style)
│  ├─ fleet_pipeline.py               # fleet-level runner
│  ├─ config.py
│  ├─ README.md                       # stage II overview
│  └─ README_STAGE_II.txt             # stage II detailed text readme
└─ rule_base.json                     # Stage II rule base (summarization/mapping rules)
```

---

## Installation

Create a fresh environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install pandas numpy pyarrow fastparquet tqdm python-dateutil
pip install rdflib PyPDF2 python-dotenv openai requests
```

Set your OpenAI key (Stage I + Stage II use it):

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

---

# Stage I: Build the Literature Knowledge Graph (papers → TTL)

Stage I reads a **folder of papers** (`.pdf`, `.txt`, `.md`) and produces:

- **per-paper**: `*.ttl` and `*.kg.json`
- **merged**: `global_knowledge_graph.ttl` (accumulates across runs)
- **updated taxonomy**: `taxonomy.json` (if the model proposes new concepts)

## Inputs

- **papers folder**: a directory containing SSD papers (`.pdf`, `.txt`, `.md`)
- **taxonomy**: `stage_I/taxonomy.json`
- **prompt**: `stage_I/ssd_cot_prompt.txt`

## Configure prompt paths

`stage_I/ssd_kg_pipeline.py` defaults to reading the prompt from `prompts/ssd_cot_prompt.txt`.
Since this repo keeps the prompt inside `stage_I/`, set:

```bash
export KG_PROMPT_PATH="stage_I/ssd_cot_prompt.txt"
export KG_PROMPT_ADDENDA_PATH="stage_I/out/ssd_prompt_addenda_auto.txt"
```

## Run Stage I

Example:

```bash
python stage_I/ssd_kg_pipeline.py \
  --papers_dir dataset/papers \
  --taxonomy stage_I/taxonomy.json \
  --out_dir stage_I/out \
  --model gpt-4o
```

### Outputs (Stage I)

```text
stage_I/out/
├─ <paper_slug>.ttl
├─ <paper_slug>.kg.json
└─ global_knowledge_graph.ttl
```

Stage I **merges** the current run into `stage_I/out/global_knowledge_graph.ttl`.

---

# Data preparation (Alibaba / Google / Workload)

This repo includes scripts that prepare Alibaba and Google datasets and create test CSVs.
Place data prep code under `data_preparation/` and keep datasets under `dataset/`.

For Table II fleet evaluation, you only need these **three prepared datasets**:
- **Alibaba SMART** (no `app`)
- **Google SMART**
- **SMART + Workload** (Alibaba with `app`)

For the packaged Alibaba evaluation assets, the repo now includes:
- `dataset/alibaba/test_data/smart.csv` for the standalone SMART-only path
- `dataset/alibaba/test_data/smart_workload.csv` for the fused SMART+Workload path
- `dataset/alibaba/test_data/fig7_smart.csv` for the SMART-only matched Figure 7 split derived from the workload-aware rows
- `dataset/alibaba/test_data/evaluation_package.json` describing row counts, balance, and split relationships

---

# Stage II: Per-sample analysis

Stage II consumes one **input CSV** and produces per-sample:
- prompts,
- LLM responses,
- parsed outputs,
- metrics (predictive + text overlap + grounding).

## Stage II prerequisites: copy KG + taxonomy to repo root

By default, Stage II looks for these files in the **repo root**:

- `taxonomy.json`
- `global_knowledge_graph.ttl`
- `rule_base.json`

If you ran Stage I, copy:

```bash
cp stage_I/taxonomy.json taxonomy.json
cp stage_I/out/global_knowledge_graph.ttl global_knowledge_graph.ttl
```

## Run Stage II (per-sample)

Example:

```bash
python -m stage_II.cli \
  --dataset_type SMART_ALIBABA \
  --input_csv dataset/alibaba/test_data/smart.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --model gpt-4o \
  --limit_rows 100 \
  --out_name demo_smart_alibaba
```

For SMART+Workload Alibaba, use:

```bash
python -m stage_II.cli \
  --dataset_type SMART_WORKLOAD \
  --input_csv dataset/alibaba/test_data/smart_workload.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --model gpt-4o \
  --limit_rows 100 \
  --out_name demo_smart_workload
```

For a Figure 7-style SMART-only predictive comparison on the same 1000 rows as
the workload-aware package, use:

```bash
python -m stage_II.cli \
  --dataset_type SMART_ALIBABA_FIG7 \
  --tasks predictive \
  --model gpt-4o \
  --limit_rows 100 \
  --out_name fig7_smart_compare
```

Outputs go to:

```text
stage_II/runs/<RUN_NAME>/
  input_samples.csv
  responses.jsonl
  metrics_per_sample.csv
  metrics_summary.json
  data_kg_ttl/<sample_id>.ttl   (if rdflib available)
```

## Generate Table I Benchmarks

Use the curated Table I query bank to materialize per-sample descriptive, prescriptive, and what-if references:

```bash
python stage_II/scripts/run_table1.py \
  --dataset ENV=dataset/env/env_effects.csv \
  --materialize_only
```

To run KORAL inference and aggregate a Table I-style CSV, omit `--materialize_only` and add any prepared dataset mappings such as `SMART_ALIBABA=...`, `SMART_WORKLOAD=...`, or `SMART_ENV=...`.

Committed benchmark support files are available at:
- `stage_II/benchmarks/table1/corpora/` for the pre-materialized SMART-only and SMART+Workload corpora
- `stage_II/benchmarks/table1/query_coverage.json` for modality and subsystem coverage labels per query

---

# Stage II Fleet Mode: Collective analysis

Fleet mode evaluates **a cohort of N drives at once** (e.g., N=100).

### Supported datasets (as requested)
- `SMART_ALIBABA`
- `SMART_GOOGLE`
- `SMART_WORKLOAD`

Fleet mode expects the input CSV to contain one row per drive (or it will de-duplicate by `disk_id/drive_id` when possible).

## Run fleet evaluation (one dataset)

Example (100-drive cohorts, 5 cohorts):

```bash
python -m stage_II.fleet_cli \
  --dataset_type SMART_ALIBABA \
  --input_csv dataset/alibaba/test_data/smart.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name fleet_alibaba_100x5
```

For SMART+Workload Alibaba, use:

```bash
python -m stage_II.fleet_cli \
  --dataset_type SMART_WORKLOAD \
  --input_csv dataset/alibaba/test_data/smart_workload.csv \
  --tasks predictive,descriptive,prescriptive,whatif \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_name fleet_smart_workload_100x5
```

Fleet outputs go to:

```text
stage_II/runs/<RUN_NAME>/
  cohort_composition.csv
  responses_fleet.jsonl
  metrics_fleet.csv
  metrics_summary_fleet.json
  fleet_kg_ttl/<cohort_id>.ttl   (if rdflib available)
```

## Generate Table II CSV (all 3 datasets)

Use the script under `stage_II/scripts/`:

```bash
python stage_II/scripts/run_table2_fleet.py \
  --cohort_size 100 \
  --num_cohorts 5 \
  --out_dir_name table2_fleet
```

This writes:
- `stage_II/runs/table2_fleet/table_II_fleet_results.csv`
- and per-dataset fleet run folders under `stage_II/runs/table2_fleet/` (or nested depending on script settings).

To regenerate the packaged Alibaba evaluation assets, run:

```bash
python stage_II/scripts/package_alibaba_eval_assets.py
```

---

---

## Ontology & KG examples

Stage I and Stage II share a consistent “classes vs instances” design:

- **Classes** come from the taxonomy (e.g., `Temperature`, `IOPS`, `TLC`, `Garbage Collection`).
- **Instances** represent paper-specific or scenario-specific objects (e.g., `SSD_X`, `EC1`, `WP1`, `EXP1`).

Common relation patterns you’ll see in the Literature KG (Stage I) and Data KG (Stage II):

- `SSD_X operatesUnder EC1`
- `EC1 hasTemperature {"@value": 45, "unit": "C"}`
- `EC1 hasWorkloadProfile WP1`
- `WP1 hasReadWriteMix "Write-Heavy"`
- `Temperature degrades 99th Percentile Latency` (directional effect)
- `Workload impactsMetric Latency`
- Assertions always carry **evidence text** and a **confidence** score.

---

## Citation

```bibtex
@inproceedings{AkewarEtAl_IPDPS_2026,
  author    = {Akewar, Mayur and Madireddy, Sandeep and Luo, Dongsheng and Bhimani, Janki},
  title     = {KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis},
  booktitle = {IEEE International Parallel \\& Distributed Processing Symposium (IPDPS)},
  year      = {2026},
  note      = {To appear}
}
```
