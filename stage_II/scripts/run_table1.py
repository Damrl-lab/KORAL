#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Table I-style benchmarks and results from curated query packs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_II.benchmarks.table1 import materialize_table1_task_inputs
from stage_II.config import Stage2Config, resolve_path
from stage_II.pipeline import Stage2Runner
from stage_II.utils.io import ensure_dir, read_csv, write_csv, write_json

TABLE1_DATASET_LABELS = {
    "SMART_ALIBABA": "Alibaba",
    "SMART_GOOGLE": "Google",
    "ENV": "Exp. Studies",
    "SMART_WORKLOAD": "Alibaba",
    "SMART_ENV": "Alibaba + Exp. Studies",
    "WORKLOAD_ENV": "FIO + Exp. Studies",
    "SMART_ENV_WORKLOAD": "Alibaba + Exp. Studies",
    "SMART_FT": "Alibaba",
    "SMART_AL": "Alibaba",
    "ENV_FT": "Exp. Studies",
    "ENV_AL": "Exp. Studies",
}


def _parse_dataset_arg(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Expected DATASET_TYPE=/path/to/input.csv")
    dataset_type, csv_path = raw.split("=", 1)
    dataset_type = dataset_type.strip()
    csv_path = csv_path.strip()
    if not dataset_type or not csv_path:
        raise argparse.ArgumentTypeError("Expected DATASET_TYPE=/path/to/input.csv")
    return dataset_type, Path(csv_path)


def _pct(value):
    if value is None:
        return None
    try:
        return round(float(value) * 100.0, 2)
    except Exception:
        return None


def _num(value):
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except Exception:
        return None


def _build_table_row(dataset_type: str, input_csv: Path, summaries: Dict[str, Dict]) -> Dict[str, object]:
    pred = summaries.get("predictive", {}).get("predictive", {})
    desc = summaries.get("descriptive", {}).get("descriptive", {})
    pres = summaries.get("prescriptive", {}).get("prescriptive", {})
    wif = summaries.get("whatif", {}).get("whatif", {})
    return {
        "Dataset Type": dataset_type,
        "Dataset Used": TABLE1_DATASET_LABELS.get(dataset_type, "User Supplied"),
        "Input CSV": str(input_csv.resolve()),
        "Predictive_P": _pct(pred.get("P")),
        "Predictive_R": _pct(pred.get("R")),
        "Predictive_A": _pct(pred.get("A")),
        "Predictive_TL_MSE": _num(pred.get("TL_MSE")),
        "Predictive_TTF_MSE": _num(pred.get("TTF_MSE")),
        "Descriptive_B4": _pct(desc.get("B4")),
        "Descriptive_RL": _pct(desc.get("RL")),
        "Descriptive_FiP": _pct(desc.get("FiP")),
        "Prescriptive_B4": _pct(pres.get("B4")),
        "Prescriptive_RL": _pct(pres.get("RL")),
        "Prescriptive_FiP": _pct(pres.get("FiP")),
        "WhatIf_B4": _pct(wif.get("B4")),
        "WhatIf_RL": _pct(wif.get("RL")),
        "WhatIf_CFV": _pct(wif.get("CFV")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize and run Table I query/ground-truth benchmarks.")
    ap.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Repeat with DATASET_TYPE=/path/to/input.csv, e.g. SMART_ALIBABA=dataset/alibaba/test_data/smart.csv",
    )
    ap.add_argument(
        "--tasks",
        default="predictive,descriptive,prescriptive,whatif",
        help="Comma-separated tasks from {predictive,descriptive,prescriptive,whatif}.",
    )
    ap.add_argument("--limit_rows", type=int, default=None, help="Limit base rows before query expansion.")
    ap.add_argument("--out_name", default="table1_benchmark", help="Output folder under stage_II/runs/.")
    ap.add_argument("--model", default="gpt-4o", help="OpenAI model name.")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--materialize_only",
        action="store_true",
        help="Only write the benchmark CSVs (queries + references); do not run LLM inference.",
    )

    args = ap.parse_args()

    dataset_pairs = [_parse_dataset_arg(raw) for raw in args.dataset]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    cfg = Stage2Config(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens)
    runner = Stage2Runner(cfg) if not args.materialize_only else None

    top_run_dir = ensure_dir(resolve_path(cfg.repo_root, cfg.runs_dir) / args.out_name)
    benchmark_dir = ensure_dir(top_run_dir / "benchmarks")

    manifest: Dict[str, Dict[str, object]] = {}
    table_rows: List[Dict[str, object]] = []
    seed = int(args.seed)

    for dataset_type, input_csv in dataset_pairs:
        df = read_csv(input_csv)
        manifest[dataset_type] = {"input_csv": str(input_csv.resolve()), "tasks": {}}
        task_summaries: Dict[str, Dict] = {}

        for task in tasks:
            materialized = materialize_table1_task_inputs(
                dataset_type=dataset_type,
                input_df=df,
                task=task,
                limit_rows=args.limit_rows,
            )
            if materialized.empty:
                manifest[dataset_type]["tasks"][task] = {"status": "skipped", "reason": "no_applicable_queries"}
                continue

            benchmark_csv = benchmark_dir / f"{dataset_type.lower()}_{task}.csv"
            write_csv(benchmark_csv, materialized)

            task_entry = {
                "status": "materialized",
                "benchmark_csv": str(benchmark_csv.resolve()),
                "rows": int(len(materialized)),
            }

            if not args.materialize_only:
                assert runner is not None
                outs = runner.run(
                    input_csv=benchmark_csv,
                    tasks=[task],
                    out_name=f"{args.out_name}/{dataset_type.lower()}_{task}",
                    limit_rows=None,
                    seed=seed,
                )
                seed += len(materialized) + 1
                summary = json.loads(outs.summary_json.read_text(encoding="utf-8"))
                task_summaries[task] = summary
                task_entry.update(
                    {
                        "status": "completed",
                        "run_dir": str(outs.run_dir.resolve()),
                        "summary_json": str(outs.summary_json.resolve()),
                    }
                )

            manifest[dataset_type]["tasks"][task] = task_entry

        if not args.materialize_only:
            table_rows.append(_build_table_row(dataset_type, input_csv, task_summaries))

    manifest_path = top_run_dir / "materialization_manifest.json"
    write_json(manifest_path, manifest)

    if args.materialize_only:
        print(f"Materialized benchmark CSVs under: {benchmark_dir}")
        print(f"Manifest: {manifest_path}")
        return

    table_df = pd.DataFrame(table_rows)
    table_csv = top_run_dir / "table_I_results.csv"
    write_csv(table_csv, table_df)
    print(f"Table I results saved to: {table_csv}")
    print(f"Materialization manifest: {manifest_path}")


if __name__ == "__main__":
    main()
