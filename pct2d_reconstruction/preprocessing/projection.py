#!/usr/bin/env python3
"""Generate the retained Stage 5 DDB projections and compact projection QC."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PCTBINNING_SOURCE = REPOSITORY / "applications" / "pctbinning" / "pctbinning.py"
PAIRS = HERE / "pairs_filtered"

RUNS = 720
SIZE = (500, 2, 500)
SPACING_MM = (0.5, 1.0, 0.5)
ORIGIN_MM = (-124.75, -0.5, -124.75)
SOURCE_MM = -1000.0
IONIZATION_POTENTIAL_EV = 78.0
QUADRIC_WATER_CYLINDER = (1, 0, 1, 0, 0, 0, 0, 0, 0, -10000)
DTYPES = {
    "MET_FLOAT": np.dtype("<f4"),
    "MET_UINT": np.dtype("<u4"),
}


@lru_cache(maxsize=1)
def load_pctbinning():
    spec = importlib.util.spec_from_file_location("pctbinning_stage5", PCTBINNING_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PCTBINNING_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_mhd(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def read_mhd(path: Path) -> np.ndarray:
    header = parse_mhd(path)
    size = tuple(int(value) for value in header["DimSize"].split())
    dtype = DTYPES[header["ElementType"]]
    raw = path.parent / header["ElementDataFile"]
    return np.memmap(raw, dtype=dtype, mode="r", shape=size[::-1])


def run_pctbinning(arguments: list[str]) -> None:
    module = load_pctbinning()
    args = module.build_parser().parse_args(arguments)
    module.process(args)


def common_arguments(input_path: Path) -> list[str]:
    return [
        "--input",
        str(input_path),
        "--source",
        f"{SOURCE_MM:g}",
        "--particle",
        "proton",
        "--mlptype",
        "schulte",
        "--ionpot",
        f"{IONIZATION_POTENTIAL_EV:g}",
        "--spacing",
        *(f"{value:g}" for value in SPACING_MM),
    ]


def process_run(
    run_id: int,
    pairs_directory: str,
    output_root: str,
    keep_qc_images: bool,
    ddb_directory_name: str = "projections_ddb",
) -> dict[str, object]:
    root = Path(output_root)
    ddb_directory = root / ddb_directory_name
    ddb_directory.mkdir(parents=True, exist_ok=True)
    input_path = Path(pairs_directory) / f"pairs{run_id:04d}.mhd"
    ddb = ddb_directory / f"proj{run_id:04d}.mhd"

    temporary = None
    if keep_qc_images:
        count_directory = root / "proton_count"
        variance_directory = root / "wepl_variance"
        count_directory.mkdir(parents=True, exist_ok=True)
        variance_directory.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix=f"test0713-stage5-{run_id:04d}-")
        temporary_root = Path(temporary.name)
        count_directory = temporary_root
        variance_directory = temporary_root

    count_path = count_directory / f"count{run_id:04d}.mhd"
    variance_path = variance_directory / f"variance{run_id:04d}.mhd"
    start = time.perf_counter()
    run_pctbinning(
        common_arguments(input_path)
        + [
            "--output",
            str(ddb),
            "--count",
            str(count_path),
            "--variance",
            str(variance_path),
            "--dimension",
            *(str(value) for value in SIZE),
            "--quadricIn",
            *(str(value) for value in QUADRIC_WATER_CYLINDER),
        ]
    )
    elapsed = time.perf_counter() - start

    count = read_mhd(count_path)
    variance = read_mhd(variance_path)
    x = ORIGIN_MM[0] + np.arange(SIZE[0]) * SPACING_MM[0]
    z = ORIGIN_MM[2] + np.arange(SIZE[2]) * SPACING_MM[2]
    mask_2d = z[:, None] ** 2 + x[None, :] ** 2 <= 10000.0
    object_mask = np.broadcast_to(mask_2d[:, None, :], SIZE[::-1])
    row = {
        "run_id": run_id,
        "ddb_seconds": elapsed,
        "count_min": int(count.min()),
        "count_max": int(count.max()),
        "count_sum": int(count.sum(dtype=np.uint64)),
        "zero_count": int(np.count_nonzero(count == 0)),
        "object_pixels": int(object_mask.sum()),
        "object_count_sum": int(count[object_mask].sum(dtype=np.uint64)),
        "object_zero_count": int(np.count_nonzero(count[object_mask] == 0)),
        "variance_min": float(variance.min()),
        "variance_max": float(variance.max()),
        "variance_sum": float(variance.sum(dtype=np.float64)),
        "variance_nonfinite": int(np.count_nonzero(~np.isfinite(variance))),
        "variance_negative": int(np.count_nonzero(variance < 0)),
    }
    del count, variance
    if temporary is not None:
        temporary.cleanup()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=RUNS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--pairs-dir", type=Path, default=PAIRS)
    parser.add_argument(
        "--keep-qc-images",
        action="store_true",
        help="Retain per-angle proton-count and WEPL-variance images.",
    )
    args = parser.parse_args()
    if not 1 <= args.runs <= RUNS:
        parser.error(f"--runs must be between 1 and {RUNS}")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    missing = [
        args.pairs_dir / f"pairs{run_id:04d}.mhd"
        for run_id in range(args.runs)
        if not (args.pairs_dir / f"pairs{run_id:04d}.mhd").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} pair files; first: {missing[0]}")

    qc = args.output_root / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_run,
                run_id,
                str(args.pairs_dir),
                str(args.output_root),
                args.keep_qc_images,
            ): run_id
            for run_id in range(args.runs)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"run {row['run_id']:04d}: ddb={row['ddb_seconds']:.2f}s", flush=True)
    rows.sort(key=lambda row: int(row["run_id"]))
    elapsed = time.perf_counter() - start

    with (qc / "stage5_generation_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pixel_count = args.runs * int(np.prod(SIZE))
    object_pixels = sum(int(row["object_pixels"]) for row in rows)
    snapshot = {
        "status": "PASS" if all(int(row["variance_nonfinite"]) == 0 and int(row["variance_negative"]) == 0 for row in rows) else "FAIL",
        "runs": args.runs,
        "count_variance_images_retained": args.keep_qc_images,
        "ddb_pixels": pixel_count,
        "ddb_zero_count": sum(int(row["zero_count"]) for row in rows),
        "ddb_object_pixels": object_pixels,
        "ddb_object_zero_count": sum(int(row["object_zero_count"]) for row in rows),
        "ddb_count_sum": sum(int(row["count_sum"]) for row in rows),
        "ddb_object_count_sum": sum(int(row["object_count_sum"]) for row in rows),
        "ddb_count_min": min(int(row["count_min"]) for row in rows),
        "ddb_count_max": max(int(row["count_max"]) for row in rows),
        "ddb_variance_min": min(float(row["variance_min"]) for row in rows),
        "ddb_variance_max": max(float(row["variance_max"]) for row in rows),
        "ddb_variance_mean_mm2": sum(float(row["variance_sum"]) for row in rows) / pixel_count,
        "ddb_variance_nonfinite": sum(int(row["variance_nonfinite"]) for row in rows),
        "ddb_variance_negative": sum(int(row["variance_negative"]) for row in rows),
    }
    (qc / "stage5_compact_qc_snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "stopped": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "runs": args.runs,
        "jobs": args.jobs,
        "size": list(SIZE),
        "spacing_mm": list(SPACING_MM),
        "origin_mm": list(ORIGIN_MM),
        "source_mm": SOURCE_MM,
        "ionization_potential_eV": IONIZATION_POTENTIAL_EV,
        "mlp": "Schulte",
        "ddb_envelope": "x^2 + z^2 = 10000 mm^2",
        "hole_filling": False,
        "pairs_directory": str(args.pairs_dir.resolve()),
        "count_variance_images_retained": args.keep_qc_images,
        "outputs": {"ddb": str((args.output_root / "projections_ddb").resolve())},
    }
    if args.keep_qc_images:
        summary["outputs"].update(
            {
                "count": str((args.output_root / "proton_count").resolve()),
                "variance": str((args.output_root / "wepl_variance").resolve()),
            }
        )
    (qc / "stage5_generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"generation": summary, "qc_snapshot": snapshot}, indent=2))
    if snapshot["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit("internal module; use run_preprocessing.py")
