#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Table II-style results (fleet-level) for the 3 core datasets.

Runs fleet-level Stage II for:
- SMART_ALIBABA
- SMART_GOOGLE
- SMART_WORKLOAD

and writes a single CSV table with aggregated metrics.

Usage:
  python stage_II/scripts/run_table2_fleet.py --cohort_size 100 --num_cohorts 1 --out_dir_name table2_fleet_runs

Optional:
  --refs_csv <path>  (cohort-level references for BLEU/ROUGE; otherwise only FiP/CFV computed)

Dataset CSV paths are resolved from stage_II/config.py. The SMART_WORKLOAD
default points to dataset/alibaba/test_data/smart_workload.csv.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from stage_II.config import Stage2Config, resolve_path
from stage_II.fleet_pipeline import FleetStage2Runner
from stage_II.utils.io import ensure_dir, write_json, write_csv

DATASETS = [
    "SMART_ALIBABA",
    "SMART_GOOGLE",
    "SMART_WORKLOAD",
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_size", type=int, default=100)
    ap.add_argument("--num_cohorts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--top_k_signals", type=int, default=2)
    ap.add_argument("--out_dir_name", required=True, help="Parent folder name under stage_II/runs.")
    ap.add_argument("--refs_csv", default=None, help="Optional cohort-level references CSV (shared across datasets).")

    args = ap.parse_args()

    cfg = Stage2Config(repo_root=Path("."))
    runner = FleetStage2Runner(cfg)

    parent = ensure_dir(resolve_path(cfg.repo_root, cfg.runs_dir) / args.out_dir_name)

    rows = []
    for ds in DATASETS:
        input_csv = resolve_path(cfg.repo_root, cfg.dataset_type_to_csv[ds])
        out_name = f"{args.out_dir_name}_{ds.lower()}"
        out = runner.run(
            dataset_type=ds,
            input_csv=input_csv,
            tasks=["predictive","descriptive","prescriptive","whatif"],
            out_name=out_name,
            cohort_size=args.cohort_size,
            num_cohorts=args.num_cohorts,
            seed=args.seed,
            top_k_signals=args.top_k_signals,
            refs_csv=Path(args.refs_csv) if args.refs_csv else None,
        )

        # Load summary json
        import json
        summary = json.loads(Path(out.summary_json).read_text(encoding="utf-8"))
        pred = summary.get("predictive", {})
        desc = summary.get("descriptive", {})
        pres = summary.get("prescriptive", {})
        wif = summary.get("whatif", {})

        rows.append({
            "dataset_type": ds,
            "P": pred.get("P"),
            "R": pred.get("R"),
            "A": pred.get("A"),
            "TTF_MSE": pred.get("TTF_MSE"),
            "B4_desc": desc.get("B4"),
            "RL_desc": desc.get("RL"),
            "FiP_desc": desc.get("FiP"),
            "B4_pres": pres.get("B4"),
            "RL_pres": pres.get("RL"),
            "FiP_pres": pres.get("FiP"),
            "B4_wif": wif.get("B4"),
            "RL_wif": wif.get("RL"),
            "CFV_wif": wif.get("CFV"),
        })

    table = pd.DataFrame(rows)
    write_csv(parent / "table_II_fleet_results.csv", table)
    print(f"[OK] Wrote {parent/'table_II_fleet_results.csv'}")

if __name__ == "__main__":
    main()
