#!/usr/bin/env python3
"""Build a balanced Alibaba SMART test CSV from the processed MB1/MB2 test archives.

The input archives already contain 30-day windows:
  - X: [N, 30, F]
  - y: binary failure label
  - ttf: days to failure (-1 for healthy)
  - features: SMART attribute names

This script creates a Stage II-friendly CSV where each SMART attribute column
stores the 30-day series as a compact JSON list string.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "dataset" / "alibaba" / "test_data" / "smart.csv"
ARCHIVE_GLOB = "dataset/alibaba/MB*_round*/test.npz"
MODEL_NAMES = ("MB1", "MB2")
SPLIT_PSEUDO_DATES = {
    "MB1_round1": "2021-01-30",
    "MB1_round2": "2021-02-28",
    "MB1_round3": "2021-03-30",
    "MB2_round1": "2021-04-30",
    "MB2_round2": "2021-05-31",
    "MB2_round3": "2021-06-30",
}
STREAM_CHUNK_BYTES = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create dataset/alibaba/test_data/smart.csv from MB1/MB2 test.npz archives.")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="Output CSV path.")
    ap.add_argument("--n", type=int, default=1000, help="Total number of rows to write (default: 1000).")
    ap.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    return ap.parse_args()


def _archive_model(npz_path: Path) -> str:
    return npz_path.parent.name.split("_", 1)[0]


def _archive_split(npz_path: Path) -> str:
    return npz_path.parent.name


def _list_archives(repo_root: Path) -> List[Path]:
    files = sorted(repo_root.glob(ARCHIVE_GLOB))
    if not files:
        raise FileNotFoundError(f"No test archives found under {repo_root / 'dataset/alibaba'}")
    return files


def _proportional_allocation(total_needed: int, per_file_available: Sequence[int]) -> List[int]:
    total_available = int(sum(per_file_available))
    if total_needed > total_available:
        raise ValueError(f"Requested {total_needed} rows but only {total_available} are available.")
    if total_needed == 0:
        return [0 for _ in per_file_available]

    counts = np.asarray(per_file_available, dtype=float)
    raw = counts * float(total_needed) / float(total_available)
    alloc = np.floor(raw).astype(int)
    remainder = int(total_needed - int(alloc.sum()))
    if remainder > 0:
        fractional = raw - alloc
        for idx in np.argsort(-fractional)[:remainder]:
            alloc[idx] += 1
    out = alloc.astype(int).tolist()
    for want, have in zip(out, per_file_available):
        if want > have:
            raise ValueError(f"Allocation error: wanted {want}, available {have}.")
    return out


def _series_json(values: np.ndarray) -> str:
    rounded = [round(float(v), 6) for v in values.tolist()]
    return json.dumps(rounded, separators=(",", ":"))


def _read_npy_header(fh) -> Tuple[Tuple[int, ...], bool, np.dtype]:
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fh)
    elif version == (3, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_3_0(fh)
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    return shape, bool(fortran_order), np.dtype(dtype)


def _drain_bytes(fh, nbytes: int) -> None:
    remaining = int(nbytes)
    while remaining > 0:
        chunk = fh.read(min(STREAM_CHUNK_BYTES, remaining))
        if not chunk:
            raise EOFError("Unexpected EOF while skipping bytes in X.npy stream.")
        remaining -= len(chunk)


def _read_exact(fh, nbytes: int) -> bytes:
    buf = bytearray()
    remaining = int(nbytes)
    while remaining > 0:
        chunk = fh.read(remaining)
        if not chunk:
            raise EOFError("Unexpected EOF while reading X.npy stream.")
        buf.extend(chunk)
        remaining -= len(chunk)
    return bytes(buf)


def _stream_selected_windows(npz_path: Path, selected_indices: Sequence[int]) -> List[np.ndarray]:
    if not selected_indices:
        return []
    wanted = sorted(int(i) for i in selected_indices)
    out: List[np.ndarray] = []

    with zipfile.ZipFile(npz_path) as zf:
        with zf.open("X.npy") as fh:
            shape, fortran_order, dtype = _read_npy_header(fh)
            if fortran_order:
                raise ValueError(f"{npz_path} stores X.npy in Fortran order; expected C order.")
            if len(shape) != 3:
                raise ValueError(f"{npz_path} X.npy has shape {shape}; expected [N, 30, F].")
            row_shape = tuple(int(x) for x in shape[1:])
            row_size = int(math.prod(row_shape) * dtype.itemsize)

            prev_idx = -1
            for idx in wanted:
                if idx < 0 or idx >= int(shape[0]):
                    raise IndexError(f"Index {idx} out of range for {npz_path} with {shape[0]} rows.")
                skip_rows = idx - prev_idx - 1
                if skip_rows > 0:
                    _drain_bytes(fh, skip_rows * row_size)
                raw = _read_exact(fh, row_size)
                arr = np.frombuffer(raw, dtype=dtype).reshape(row_shape)
                out.append(arr.astype(np.float32, copy=False))
                prev_idx = idx
    return out


def _load_archive_counts(archive_path: Path) -> Dict[str, int]:
    with np.load(archive_path, allow_pickle=False) as z:
        y = z["y"]
        return {
            "healthy": int(np.sum(y == 0)),
            "failed": int(np.sum(y == 1)),
        }


def _sample_archive_rows(
    archive_path: Path,
    want_healthy: int,
    want_failed: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    with np.load(archive_path, allow_pickle=False) as z:
        y = z["y"]
        ttf = z["ttf"]
        features = [str(x) for x in z["features"].tolist()]

        healthy_idx = np.flatnonzero(y == 0)
        failed_idx = np.flatnonzero(y == 1)

        chosen_healthy = rng.choice(healthy_idx, size=want_healthy, replace=False) if want_healthy else np.array([], dtype=int)
        chosen_failed = rng.choice(failed_idx, size=want_failed, replace=False) if want_failed else np.array([], dtype=int)
        chosen = np.sort(np.concatenate([chosen_healthy, chosen_failed]).astype(int))
        chosen_y = y[chosen].astype(int)
        chosen_ttf = ttf[chosen].astype(int)

    windows = _stream_selected_windows(archive_path, chosen.tolist())
    split_name = _archive_split(archive_path)
    model_name = _archive_model(archive_path)
    ds = pd.Timestamp(SPLIT_PSEUDO_DATES.get(split_name, "2021-01-30"))

    rows: List[Dict[str, object]] = []
    for idx, label, ttf_days, window in zip(chosen.tolist(), chosen_y.tolist(), chosen_ttf.tolist(), windows):
        if window.shape[0] != 30:
            raise ValueError(f"{archive_path} index {idx} produced window shape {window.shape}; expected 30-day windows.")

        failure_ts = ""
        if int(label) == 1 and int(ttf_days) >= 0:
            failure_ts = (ds + timedelta(days=int(ttf_days))).date().isoformat()

        row: Dict[str, object] = {
            "disk_id": f"{split_name.lower()}_{idx:07d}",
            "ds": ds.date().isoformat(),
            "model": model_name,
            "app": "",
            "failure_time": failure_ts,
            "failure": int(label),
            "ttf_days": int(ttf_days),
            "source_split": split_name,
            "source_index": int(idx),
        }
        for feat_idx, feat_name in enumerate(features):
            row[feat_name] = _series_json(window[:, feat_idx])
        rows.append(row)
    return rows


def _build_balanced_rows(archives: Sequence[Path], total_rows: int, seed: int) -> pd.DataFrame:
    if total_rows <= 0:
        raise ValueError("--n must be > 0")
    if total_rows % (len(MODEL_NAMES) * 2) != 0:
        raise ValueError(f"--n must be divisible by {len(MODEL_NAMES) * 2} to keep MB1/MB2 and healthy/failed balanced.")

    per_bucket = total_rows // (len(MODEL_NAMES) * 2)
    rng = np.random.default_rng(seed)

    counts_by_archive = {path: _load_archive_counts(path) for path in archives}
    archives_by_model: Dict[str, List[Path]] = {model: [] for model in MODEL_NAMES}
    for path in archives:
        model = _archive_model(path)
        if model in archives_by_model:
            archives_by_model[model].append(path)

    all_rows: List[Dict[str, object]] = []
    for model in MODEL_NAMES:
        model_archives = archives_by_model.get(model, [])
        if not model_archives:
            raise ValueError(f"No archives found for {model}.")

        healthy_alloc = _proportional_allocation(
            per_bucket,
            [counts_by_archive[path]["healthy"] for path in model_archives],
        )
        failed_alloc = _proportional_allocation(
            per_bucket,
            [counts_by_archive[path]["failed"] for path in model_archives],
        )

        for path, want_h, want_f in zip(model_archives, healthy_alloc, failed_alloc):
            rows = _sample_archive_rows(path, want_h, want_f, rng)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No rows were generated from the input archives.")

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, "sample_id", [f"smart_alibaba_{i:04d}" for i in range(len(df))])
    return df


def main() -> None:
    args = parse_args()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    archives = _list_archives(REPO_ROOT)
    df = _build_balanced_rows(archives, total_rows=int(args.n), seed=int(args.seed))
    df.to_csv(out_path, index=False)

    label_counts = df["failure"].value_counts().to_dict()
    model_counts = df["model"].value_counts().to_dict()
    print(f"Wrote {len(df)} rows to {out_path}")
    print(f"Failure label counts: {label_counts}")
    print(f"Model counts: {model_counts}")


if __name__ == "__main__":
    main()
