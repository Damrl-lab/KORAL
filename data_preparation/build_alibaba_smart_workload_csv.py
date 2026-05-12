#!/usr/bin/env python3
"""Create a workload-aware Alibaba SMART CSV from the existing SMART windows.

This is a best-effort fallback when the original Alibaba application tags are
not available in the repository. Workload class is inferred from the 30-day
write/read activity counters:
  - r_241: blocks written
  - r_242: blocks read

Output columns added:
  - workload: write-dominant | read-dominant | mixed or unclear
  - workload_source: inferred_from_r_241_r_242_delta
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stage_II.features.smart import parse_series


DEFAULT_IN = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart.csv"
DEFAULT_OUT = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart_workload.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build smart_workload.csv from smart.csv using inferred read/write dominance.")
    ap.add_argument("--input", default=str(DEFAULT_IN), help="Input smart.csv path.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output smart_workload.csv path.")
    return ap.parse_args()


def _delta(series_text: object) -> float | None:
    values: List[float] = parse_series(series_text)
    if len(values) < 2:
        return None
    return float(values[-1] - values[0])


def _infer_workload_label(row: pd.Series) -> str:
    write_delta = _delta(row.get("r_241"))
    read_delta = _delta(row.get("r_242"))
    if write_delta is None or read_delta is None:
        return "mixed or unclear"
    if write_delta > read_delta * 1.1:
        return "write-dominant"
    if read_delta > write_delta * 1.1:
        return "read-dominant"
    return "mixed or unclear"


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df["workload"] = df.apply(_infer_workload_label, axis=1)
    df["workload_source"] = "inferred_from_r_241_r_242_delta"
    df.to_csv(out_path, index=False)

    counts = df["workload"].value_counts(dropna=False).to_dict()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Workload counts: {counts}")


if __name__ == "__main__":
    main()
