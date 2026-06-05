#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package Alibaba evaluation assets for Figure 7, Table I, and Table II.

This script creates three kinds of reproducible artifacts:

1. A matched SMART-only Figure 7 split derived from the SMART+Workload set
2. A compact pre-materialized Table I corpus with row-level ground truth
3. A query coverage manifest that labels modality and subsystem scope
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_II.benchmarks.table1 import load_query_bank, materialize_table1_task_inputs
from stage_II.utils.io import ensure_dir, read_csv, write_csv, write_json

DEFAULT_SMART = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart.csv"
DEFAULT_SMART_WORKLOAD = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart_workload.csv"
DEFAULT_FIG7_SMART = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "fig7_smart.csv"
DEFAULT_TEST_DATA_README = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "README.md"
DEFAULT_EVAL_MANIFEST = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "evaluation_package.json"
DEFAULT_CORPORA_DIR = REPO_ROOT / "stage_II" / "benchmarks" / "table1" / "corpora"
DEFAULT_QUERY_COVERAGE = REPO_ROOT / "stage_II" / "benchmarks" / "table1" / "query_coverage.json"
DEFAULT_FIG7_DIR = REPO_ROOT / "stage_II" / "benchmarks" / "fig7"

TASKS = ("predictive", "descriptive", "prescriptive", "whatif")
PREFIX_BY_DATASET = {
    "SMART_ALIBABA": "smart_alibaba",
    "SMART_WORKLOAD": "smart_workload",
}
QUERY_COLUMN_BY_TASK = {
    "predictive": "predictive_query",
    "descriptive": "descriptive_query",
    "prescriptive": "prescriptive_query",
    "whatif": "whatif_query",
}
REFERENCE_COLUMN_BY_TASK = {
    "predictive": None,
    "descriptive": "ref_descriptive",
    "prescriptive": "ref_prescriptive",
    "whatif": "ref_whatif",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Package Alibaba evaluation assets for repo users.")
    ap.add_argument("--smart_csv", default=str(DEFAULT_SMART), help="SMART-only 1000-row CSV path.")
    ap.add_argument("--smart_workload_csv", default=str(DEFAULT_SMART_WORKLOAD), help="SMART+Workload 1000-row CSV path.")
    ap.add_argument("--fig7_smart_csv", default=str(DEFAULT_FIG7_SMART), help="Output SMART-only matched Figure 7 CSV path.")
    ap.add_argument("--corpora_dir", default=str(DEFAULT_CORPORA_DIR), help="Output directory for compact benchmark corpora.")
    ap.add_argument("--query_coverage_json", default=str(DEFAULT_QUERY_COVERAGE), help="Output path for query coverage manifest.")
    ap.add_argument("--fig7_dir", default=str(DEFAULT_FIG7_DIR), help="Output directory for Figure 7 packaging artifacts.")
    ap.add_argument("--eval_manifest_json", default=str(DEFAULT_EVAL_MANIFEST), help="Output dataset package manifest path.")
    return ap.parse_args()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def _count_summary(df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "rows": int(len(df)),
    }
    if "failure" in df.columns:
        counts = df["failure"].value_counts().to_dict()
        summary["failure_counts"] = {str(k): int(v) for k, v in counts.items()}
    if "disk_id" in df.columns:
        summary["unique_disk_id"] = int(df["disk_id"].nunique())
    if "app" in df.columns:
        apps = df["app"].fillna("").astype(str).str.strip().value_counts().head(10).to_dict()
        summary["top_app_counts"] = {str(k): int(v) for k, v in apps.items()}
    return summary


def _base_identity_columns(df: pd.DataFrame) -> List[str]:
    wanted = [
        "sample_id",
        "base_sample_id",
        "disk_id",
        "drive_id",
        "ds",
        "model",
        "app",
        "failure",
        "ttf_days",
        "source_split",
        "source_index",
        "dataset_type",
        "benchmark_task",
        "benchmark_query_id",
        "benchmark_query_group",
        "benchmark_dataset_profile",
        "benchmark_ground_truth_mode",
        "retrieval_terms",
    ]
    return [col for col in wanted if col in df.columns]


def _compact_benchmark_df(materialized: pd.DataFrame, task: str) -> pd.DataFrame:
    compact = materialized[_base_identity_columns(materialized)].copy()
    compact["query_text"] = materialized[QUERY_COLUMN_BY_TASK[task]].fillna("").astype(str)
    compact["scenario_text"] = (
        materialized["whatif_scenario"].fillna("").astype(str) if task == "whatif" and "whatif_scenario" in materialized.columns else ""
    )
    ref_col = REFERENCE_COLUMN_BY_TASK[task]
    compact["reference_text"] = materialized[ref_col].fillna("").astype(str) if ref_col else ""
    compact["reference_kind"] = "dataset_label" if task == "predictive" else "text_reference"
    return compact


def _write_compact_corpora(dataset_type: str, input_csv: Path, corpora_dir: Path) -> Dict[str, object]:
    df = read_csv(input_csv)
    prefix = PREFIX_BY_DATASET[dataset_type]
    manifest: Dict[str, object] = {
        "input_csv": _rel(input_csv),
        "rows": int(len(df)),
        "tasks": {},
    }

    for task in TASKS:
        materialized = materialize_table1_task_inputs(dataset_type=dataset_type, input_df=df, task=task, limit_rows=None)
        compact = _compact_benchmark_df(materialized, task)
        out_csv = corpora_dir / f"{prefix}_{task}.csv"
        write_csv(out_csv, compact)
        manifest["tasks"][task] = {
            "csv": _rel(out_csv),
            "rows": int(len(compact)),
            "unique_base_samples": int(compact["base_sample_id"].nunique()) if "base_sample_id" in compact.columns else int(compact["sample_id"].nunique()),
            "unique_queries": int(compact["benchmark_query_id"].nunique()) if "benchmark_query_id" in compact.columns else 0,
        }
    return manifest


def _write_fig7_split(smart_workload_csv: Path, fig7_smart_csv: Path, fig7_dir: Path) -> Dict[str, object]:
    df = read_csv(smart_workload_csv)
    fig7_df = df.copy()
    if "app" in fig7_df.columns:
        fig7_df["app"] = ""
    write_csv(fig7_smart_csv, fig7_df)

    sample_cols = [
        "sample_id",
        "disk_id",
        "ds",
        "model",
        "failure",
        "ttf_days",
        "source_split",
        "source_index",
    ]
    sample_cols = [col for col in sample_cols if col in fig7_df.columns]
    sample_ids_csv = fig7_dir / "fig7_sample_ids.csv"
    write_csv(sample_ids_csv, fig7_df[sample_cols].copy())

    predictive = materialize_table1_task_inputs("SMART_ALIBABA", fig7_df, "predictive", limit_rows=None)
    predictive_corpus = _compact_benchmark_df(predictive, "predictive")
    predictive_csv = fig7_dir / "fig7_predictive_corpus.csv"
    write_csv(predictive_csv, predictive_corpus)

    fig7_manifest = {
        "input_source_csv": _rel(smart_workload_csv),
        "matched_smart_only_csv": _rel(fig7_smart_csv),
        "sample_ids_csv": _rel(sample_ids_csv),
        "predictive_corpus_csv": _rel(predictive_csv),
        "row_count": int(len(fig7_df)),
        "same_sample_ids_as_workload_csv": True,
        "note": "This SMART-only matched split is derived from smart_workload.csv by blanking workload labels while preserving the same 1000 rows.",
    }
    write_json(fig7_dir / "manifest.json", fig7_manifest)
    return fig7_manifest


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _classify_query(query: Dict[str, object]) -> Dict[str, object]:
    text = " ".join(
        str(query.get(key, "")).lower()
        for key in ("id", "group", "question", "scenario", "ground_truth")
    )
    dataset_types = list(query.get("dataset_types", []))

    modalities: List[str] = []
    if any(ds.startswith("SMART") or ds == "SMART_GOOGLE" for ds in dataset_types) or "smart" in text:
        modalities.append("SMART")
    if any("WORKLOAD" in ds for ds in dataset_types) or _contains_any(text, ["workload", "app tag", "write-heavy", "read-dominant", "queue depth"]):
        modalities.append("WORKLOAD")
    if any(ds.startswith("ENV") or ds in {"SMART_ENV", "WORKLOAD_ENV", "SMART_ENV_WORKLOAD"} for ds in dataset_types) or _contains_any(text, ["temperature", "humidity", "vibration", "environment", "airflow"]):
        modalities.append("ENVIRONMENT")
    if any(ds.endswith("_FT") or ds == "SMART_FT" for ds in dataset_types) or _contains_any(text, ["flash type", "tlc", "qlc", "plc", "slc"]):
        modalities.append("FLASH_TYPE")
    if any(ds.endswith("_AL") or ds == "SMART_AL" for ds in dataset_types) or _contains_any(text, ["controller", "policy", "garbage collection", "wear leveling", "ftl", "refresh", "ecc"]):
        modalities.append("CONTROLLER_POLICY")

    subsystems: List[str] = []
    if query.get("task") == "predictive":
        subsystems.append("predictive_reliability")
    if _contains_any(text, ["temperature", "thermal", "cooling", "airflow", "ambient", "throttle"]):
        subsystems.append("thermal")
    if _contains_any(text, ["humidity", "moisture"]):
        subsystems.append("humidity")
    if _contains_any(text, ["vibration", "vib_"]):
        subsystems.append("vibration")
    if _contains_any(text, ["interface", "ucrc", "crc", "cable", "connector", "link instability", "host-link"]):
        subsystems.append("interface_or_link")
    if _contains_any(text, ["wear", "endurance", "media", "reallocated", "uncorrectable", "wearout", "program fail"]):
        subsystems.append("media_endurance")
    if _contains_any(text, ["workload", "app tag", "write-heavy", "write pressure", "read-dominant", "burst", "queue depth", "write amplification"]):
        subsystems.append("workload_pressure")
    if _contains_any(text, ["age", "lifetime", "aging", "ageing", "replace", "replacement", "ttf"]):
        subsystems.append("age_lifecycle")
    if _contains_any(text, ["fail-slow", "tail latency", "latency", "throttling"]):
        subsystems.append("fail_slow_or_latency")
    if _contains_any(text, ["correlated", "rack", "node", "same-model", "model mix", "placement", "eager recovery", "lazy recovery", "cluster"]):
        subsystems.append("correlated_failure_and_placement")
    if _contains_any(text, ["flash type", "controller", "policy", "firmware", "ftl", "garbage collection", "wear leveling", "refresh", "ecc"]):
        subsystems.append("static_device_context")

    seen = set()
    modalities = [item for item in modalities if not (item in seen or seen.add(item))]
    seen.clear()
    subsystems = [item for item in subsystems if not (item in seen or seen.add(item))]

    beyond_core = any(
        subsystem not in {"thermal", "interface_or_link", "predictive_reliability"}
        for subsystem in subsystems
    )

    return {
        "id": str(query.get("id", "")),
        "task": str(query.get("task", "")),
        "group": str(query.get("group", "")),
        "question": str(query.get("question", "")),
        "scenario": str(query.get("scenario", "")),
        "dataset_types": dataset_types,
        "modalities": modalities,
        "subsystem_families": subsystems,
        "exercises_beyond_thermal_or_interface": beyond_core,
    }


def _write_query_coverage_manifest(out_json: Path) -> Dict[str, object]:
    queries = load_query_bank()
    coverage_rows = [_classify_query(query) for query in queries]

    dataset_counts = Counter()
    subsystem_counts = Counter()
    smart_workload_subsystems = Counter()
    smart_workload_query_ids: List[str] = []

    for row in coverage_rows:
        for dataset_type in row["dataset_types"]:
            dataset_counts[dataset_type] += 1
        for subsystem in row["subsystem_families"]:
            subsystem_counts[subsystem] += 1
        if "SMART_WORKLOAD" in row["dataset_types"]:
            smart_workload_query_ids.append(row["id"])
            for subsystem in row["subsystem_families"]:
                smart_workload_subsystems[subsystem] += 1

    manifest = {
        "version": 1,
        "query_count": len(coverage_rows),
        "dataset_query_counts": dict(sorted(dataset_counts.items())),
        "subsystem_query_counts": dict(sorted(subsystem_counts.items())),
        "smart_workload_scope_summary": {
            "query_count": len(smart_workload_query_ids),
            "query_ids": smart_workload_query_ids,
            "subsystem_query_counts": dict(sorted(smart_workload_subsystems.items())),
            "note": "SMART_WORKLOAD queries cover SMART plus workload reasoning and also touch media endurance, interface/link issues, age/lifecycle, fail-slow, and correlated-failure scope. They are not limited to thermal or electrical-only reasoning.",
        },
        "queries": coverage_rows,
    }
    write_json(out_json, manifest)
    return manifest


def _write_dataset_package_manifest(
    smart_csv: Path,
    smart_workload_csv: Path,
    fig7_smart_csv: Path,
    out_json: Path,
) -> None:
    smart_df = read_csv(smart_csv)
    smart_workload_df = read_csv(smart_workload_csv)
    fig7_df = read_csv(fig7_smart_csv)
    manifest = {
        "version": 1,
        "files": {
            "smart_csv": {
                "path": _rel(smart_csv),
                "role": "Standalone SMART-only Alibaba 1000-row evaluation set.",
                **_count_summary(smart_df),
            },
            "smart_workload_csv": {
                "path": _rel(smart_workload_csv),
                "role": "Standalone SMART+Workload Alibaba 1000-row evaluation set.",
                **_count_summary(smart_workload_df),
            },
            "fig7_smart_csv": {
                "path": _rel(fig7_smart_csv),
                "role": "SMART-only matched split derived from smart_workload.csv for Figure 7-style predictive baseline comparison.",
                **_count_summary(fig7_df),
            },
        },
        "notes": [
            "smart.csv and smart_workload.csv are different balanced 1000-row sample sets.",
            "fig7_smart.csv preserves the same 1000 windows as smart_workload.csv but blanks the workload label so baseline predictors can run on a SMART-only view of the exact same rows.",
            "Use smart.csv for the legacy SMART-only packaging path, smart_workload.csv for fused SMART+Workload evaluation, and fig7_smart.csv for apples-to-apples Figure 7 style predictive baseline comparison on the workload-matched rows.",
        ],
    }
    write_json(out_json, manifest)


def main() -> None:
    args = parse_args()

    smart_csv = Path(args.smart_csv).expanduser().resolve()
    smart_workload_csv = Path(args.smart_workload_csv).expanduser().resolve()
    fig7_smart_csv = Path(args.fig7_smart_csv).expanduser().resolve()
    corpora_dir = ensure_dir(Path(args.corpora_dir).expanduser().resolve())
    query_coverage_json = Path(args.query_coverage_json).expanduser().resolve()
    fig7_dir = ensure_dir(Path(args.fig7_dir).expanduser().resolve())
    eval_manifest_json = Path(args.eval_manifest_json).expanduser().resolve()

    fig7_manifest = _write_fig7_split(smart_workload_csv, fig7_smart_csv, fig7_dir)

    corpora_manifest = {
        "version": 1,
        "datasets": {
            "SMART_ALIBABA": _write_compact_corpora("SMART_ALIBABA", smart_csv, corpora_dir),
            "SMART_WORKLOAD": _write_compact_corpora("SMART_WORKLOAD", smart_workload_csv, corpora_dir),
        },
        "notes": [
            "These are compact pre-materialized benchmark corpora intended for inspection, sharing, and reuse.",
            "Predictive corpus rows use dataset labels and regression targets from the source CSVs.",
            "Descriptive, prescriptive, and what-if corpus rows carry the row-aligned reference text produced from query_bank.json and the materialization helpers.",
        ],
    }
    write_json(corpora_dir / "manifest.json", corpora_manifest)

    _write_query_coverage_manifest(query_coverage_json)
    _write_dataset_package_manifest(smart_csv, smart_workload_csv, fig7_smart_csv, eval_manifest_json)

    print("Packaged Alibaba evaluation assets:")
    print(f"- Figure 7 matched SMART-only split: {_rel(fig7_smart_csv)}")
    print(f"- Figure 7 manifest: {_rel(fig7_dir / 'manifest.json')}")
    print(f"- Compact Table I corpora: {_rel(corpora_dir)}")
    print(f"- Query coverage manifest: {_rel(query_coverage_json)}")
    print(f"- Dataset package manifest: {_rel(eval_manifest_json)}")
    print(f"- Figure 7 rows: {fig7_manifest['row_count']}")


if __name__ == "__main__":
    main()
