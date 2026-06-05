#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line interface for Stage II."""

from __future__ import annotations
import argparse
from pathlib import Path

from stage_II.config import Stage2Config, resolve_path
from stage_II.pipeline import Stage2Runner

def main():
    ap = argparse.ArgumentParser(description="KORAL Stage II (operational analysis + evaluation)")
    ap.add_argument("--dataset_type", type=str, required=True,
                    help="Dataset type key (e.g., SMART_ALIBABA, SMART_ALIBABA_FIG7, SMART_GOOGLE, ENV, SMART_WORKLOAD, SMART_ENV, WORKLOAD_ENV, SMART_ENV_WORKLOAD, SMART_FT, SMART_AL).")
    ap.add_argument("--input_csv", type=str, default=None,
                    help="Optional explicit path to input CSV. If omitted, uses a default path for dataset_type.")
    ap.add_argument("--tasks", type=str, default="predictive,descriptive,prescriptive,whatif",
                    help="Comma-separated tasks from {predictive,descriptive,prescriptive,whatif}.")
    ap.add_argument("--limit_rows", type=int, default=None, help="Process only first N rows (debugging).")
    ap.add_argument("--out_name", type=str, default=None, help="Run output folder name (defaults to dataset_type + timestamp).")
    ap.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model name (default: gpt-4o).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    cfg = Stage2Config(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    runner = Stage2Runner(cfg)

    if args.input_csv:
        input_csv = Path(args.input_csv)
    else:
        if args.dataset_type not in cfg.dataset_type_to_csv:
            raise SystemExit(f"Unknown dataset_type: {args.dataset_type}. Provide --input_csv.")
        input_csv = resolve_path(cfg.repo_root, cfg.dataset_type_to_csv[args.dataset_type])

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    out_name = args.out_name
    if not out_name:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{args.dataset_type}_{ts}"

    outs = runner.run(
        input_csv=input_csv,
        tasks=tasks,
        out_name=out_name,
        limit_rows=args.limit_rows,
        seed=args.seed,
    )

    print(f"Run saved under: {outs.run_dir}")
    print(f"Responses: {outs.responses_jsonl}")
    print(f"Per-sample metrics: {outs.metrics_csv}")
    print(f"Summary: {outs.summary_json}")
    print(f"DataKG TTLs: {outs.data_kg_dir}")

if __name__ == "__main__":
    main()

# --------------------------
# HOW TO RUN (from repo root)
# --------------------------
# 1) Export your OpenAI key:
#    export OPENAI_API_KEY="sk-..."
#
# 2) Place these files at repo root (or update Stage2Config paths):
#    - rule_base.json
#    - taxonomy.json
#    - global_knowledge_graph.ttl
#    - ssd_cot_prompt.txt
#
# 3) Run SMART-only:
#    python -m stage_II.cli --dataset_type SMART_ALIBABA --input_csv dataset/alibaba/test_data/smart.csv --tasks predictive,descriptive,prescriptive,whatif --limit_rows 50
#
#    Run SMART+Workload:
#    python -m stage_II.cli --dataset_type SMART_WORKLOAD --input_csv dataset/alibaba/test_data/smart_workload.csv --tasks predictive,descriptive,prescriptive,whatif --limit_rows 50
#
#    Run matched Figure 7 SMART-only predictive comparison:
#    python -m stage_II.cli --dataset_type SMART_ALIBABA_FIG7 --tasks predictive --limit_rows 50
#
# 4) Outputs appear in:
#    stage_II/runs/<RUN_NAME>/
