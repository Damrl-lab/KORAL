#!/usr/bin/env python3
"""Create a balanced 1000-row Alibaba SMART+workload dataset.

The source file may contain many windows per physical drive. For a compact
Table I / demo dataset, this script:

1. Keeps one representative window per disk_id
   - failed drives: the nearest-to-failure window (minimum ttf_days)
   - healthy drives: the latest available window
2. Samples exactly 500 failed + 500 healthy rows
3. Preserves workload/model diversity by sampling proportionally over
   (app, model) groups within each label
4. Writes a clean 1000-row CSV with unique disk_id values
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart_workload_all.csv"
DEFAULT_OUT = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart_workload.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a balanced 1000-row Alibaba SMART+workload CSV.")
    ap.add_argument("--input", default=str(DEFAULT_IN), help="Input workload CSV path.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output 1000-row workload CSV path.")
    ap.add_argument("--n", type=int, default=1000, help="Total output rows. Must be even. Default: 1000.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed. Default: 7.")
    return ap.parse_args()


def _allocate_counts(total_needed: int, avail_counts: Sequence[int]) -> List[int]:
    total_available = int(sum(avail_counts))
    if total_needed > total_available:
        raise ValueError(f"Requested {total_needed} rows but only {total_available} are available.")
    if total_needed <= 0:
        return [0 for _ in avail_counts]

    counts = np.asarray(avail_counts, dtype=float)
    raw = counts * float(total_needed) / float(total_available)
    alloc = np.floor(raw).astype(int)
    remainder = int(total_needed - int(alloc.sum()))
    if remainder > 0:
        fractional = raw - alloc
        for idx in np.argsort(-fractional)[:remainder]:
            alloc[idx] += 1
    out = alloc.astype(int).tolist()
    for want, have in zip(out, avail_counts):
        if want > have:
            raise ValueError(f"Allocation error: requested {want}, available {have}.")
    return out


def _normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    if "disk_id" not in df.columns or "failure" not in df.columns or "app" not in df.columns:
        raise ValueError("Input CSV must include disk_id, failure, and app columns.")

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda x: x.replace("\x00", "") if isinstance(x, str) else x)
    out["app"] = out["app"].fillna("none").astype(str).str.strip()
    out.loc[out["app"] == "", "app"] = "none"
    out["_ttf_num"] = pd.to_numeric(out.get("ttf_days"), errors="coerce")
    out["_ds_num"] = pd.to_datetime(out.get("ds"), errors="coerce")
    return out


def _representative_rows(df: pd.DataFrame) -> pd.DataFrame:
    failed = df[df["failure"] == 1].copy()
    failed = failed.sort_values(
        by=["disk_id", "_ttf_num", "_ds_num"],
        ascending=[True, True, False],
        na_position="last",
    )
    failed = failed.groupby("disk_id", as_index=False).first()

    failed_disk_ids = set(failed["disk_id"].astype(str))
    healthy = df[df["failure"] == 0].copy()
    healthy = healthy[~healthy["disk_id"].astype(str).isin(failed_disk_ids)].copy()
    healthy = healthy.sort_values(
        by=["disk_id", "_ds_num"],
        ascending=[True, False],
        na_position="last",
    )
    healthy = healthy.groupby("disk_id", as_index=False).first()

    return pd.concat([healthy, failed], ignore_index=True)


def _sample_balanced(df: pd.DataFrame, per_label: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_parts: List[pd.DataFrame] = []

    for label in (0, 1):
        sub = df[df["failure"] == label].copy()
        if len(sub) < per_label:
            raise ValueError(f"Label {label} only has {len(sub)} rows; need {per_label}.")

        group_keys = ["app", "model"] if "model" in sub.columns else ["app"]
        grouped = list(sub.groupby(group_keys, dropna=False, sort=False))
        alloc = _allocate_counts(per_label, [len(g) for _, g in grouped])

        picks: List[pd.DataFrame] = []
        for (key, group), take in zip(grouped, alloc):
            if take <= 0:
                continue
            chosen_idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
            picks.append(sub.loc[chosen_idx])

        part = pd.concat(picks, ignore_index=True)
        sampled_parts.append(part)

    out = pd.concat(sampled_parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["sample_id"] = [f"smart_workload_1000_{i:04d}" for i in range(len(out))]
    return out


def main() -> None:
    args = parse_args()
    if args.n <= 0 or args.n % 2 != 0:
        raise ValueError("--n must be a positive even number.")

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    norm = _normalize_source(df)
    reps = _representative_rows(norm)
    sampled = _sample_balanced(reps, per_label=args.n // 2, seed=int(args.seed))
    sampled = sampled.drop(columns=["_ttf_num", "_ds_num"], errors="ignore")
    sampled.to_csv(out_path, index=False)

    print(f"Wrote {len(sampled)} rows to {out_path}")
    print(f"Failure counts: {sampled['failure'].value_counts().to_dict()}")
    print(f"Unique disk_id: {sampled['disk_id'].nunique()}")
    if "app" in sampled.columns:
        print(f"Top app counts: {sampled['app'].value_counts().head(10).to_dict()}")
    if "model" in sampled.columns:
        print(f"Model counts: {sampled['model'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
