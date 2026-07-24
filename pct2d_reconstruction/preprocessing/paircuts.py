#!/usr/bin/env python3
"""Reproduce the default C++ pctpaircuts energy/angle 3-sigma algorithm."""

from __future__ import annotations

import csv
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = HERE / "pairs"
DEFAULT_FILTERED = HERE / "pairs_filtered"
DEFAULT_QC = HERE / "qc"

RUNS = 720
GRID_SIZE = (125, 2)
GRID_SPACING_MM = (2.0, 2.0)
GRID_ORIGIN_MM = (-124.0, -1.0)
SOURCE_MM = -1000.0
ENERGY_SIGMA_CUT = 3.0
ANGLE_SIGMA_CUT = 3.0


def read_mhd(path: Path) -> np.ndarray:
    header = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    size = [int(value) for value in header["DimSize"].split()]
    if size[0] != 5 or header.get("ElementNumberOfChannels") != "3":
        raise ValueError(f"unexpected pair layout in {path}")
    raw_path = path.parent / header["ElementDataFile"]
    data = np.fromfile(raw_path, dtype="<f4")
    return data.reshape(size[1], size[0], 3)


def write_mhd(path: Path, pairs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    pairs.astype("<f4", copy=False).tofile(raw_path)
    path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 2",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 1",
                "Offset = 0 0",
                "CenterOfRotation = 0 0",
                "ElementSpacing = 1 1",
                f"DimSize = 5 {len(pairs)}",
                "AnatomicalOrientation = ??",
                "ElementNumberOfChannels = 3",
                "ElementType = MET_FLOAT",
                f"ElementDataFile = {raw_path.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def itk_round(values: np.ndarray) -> np.ndarray:
    """Match itk::Math::Round, including half-away-from-zero negatives."""

    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(
        np.int64
    )


def pair_features(pairs: np.ndarray):
    p_in = pairs[:, 0, :]
    p_out = pairs[:, 1, :]
    d_in = pairs[:, 2, :].astype(np.float64)
    d_out = pairs[:, 3, :].astype(np.float64)
    data = pairs[:, 4, :]

    magnification = (SOURCE_MM - p_out[0, 2]) / (SOURCE_MM - p_in[0, 2])
    x_pixel = (p_in[:, 0] * magnification - GRID_ORIGIN_MM[0]) / GRID_SPACING_MM[0]
    y_pixel = (p_in[:, 1] * magnification - GRID_ORIGIN_MM[1]) / GRID_SPACING_MM[1]
    i = itk_round(x_pixel)
    j = itk_round(y_pixel)
    inside = (i >= 0) & (i < GRID_SIZE[0]) & (j >= 0) & (j < GRID_SIZE[1])
    pixel = i + j * GRID_SIZE[0]

    def projected_angle(axis: int) -> np.ndarray:
        dot = d_in[:, axis] * d_out[:, axis] + d_in[:, 2] * d_out[:, 2]
        norm_in = np.sqrt(
            d_in[:, axis] * d_in[:, axis] + d_in[:, 2] * d_in[:, 2]
        )
        norm_out = np.sqrt(
            d_out[:, axis] * d_out[:, axis] + d_out[:, 2] * d_out[:, 2]
        )
        cosine = np.minimum(1.0, dot / (norm_in * norm_out))
        return np.arccos(cosine)

    angle_x = projected_angle(0)
    angle_y = projected_angle(1)
    # Match pctpaircuts exactly: list-mode files may store either
    # (energy_in, energy_out) or (0, precomputed_WEPL) in these slots.
    energy_loss = np.where(
        data[:, 0] == 0.0, data[:, 1], data[:, 0] - data[:, 1]
    ).astype(np.float64)
    return inside, pixel, angle_x, angle_y, energy_loss


def filter_pairs(pairs: np.ndarray):
    inside, pixel, angle_x, angle_y, energy_loss = pair_features(pairs)
    pixel_inside = pixel[inside]
    count = np.bincount(pixel_inside, minlength=GRID_SIZE[0] * GRID_SIZE[1]).astype(
        np.uint32
    )

    sum_energy = np.zeros(len(count), dtype=np.float32)
    sum_energy_sq = np.zeros(len(count), dtype=np.float32)
    sum_angle_sq = np.zeros(len(count), dtype=np.float32)
    np.add.at(sum_energy, pixel_inside, energy_loss[inside].astype(np.float32))
    np.add.at(
        sum_energy_sq,
        pixel_inside,
        np.square(energy_loss[inside]).astype(np.float32),
    )
    # The C++ implementation adds the two projected-angle squares separately
    # to its float accumulator; preserve that rounding order here.
    np.add.at(
        sum_angle_sq, pixel_inside, np.square(angle_x[inside]).astype(np.float32)
    )
    np.add.at(
        sum_angle_sq, pixel_inside, np.square(angle_y[inside]).astype(np.float32)
    )

    occupied = count > 0
    mean_energy = np.zeros(len(count), dtype=np.float32)
    sigma_energy = np.zeros(len(count), dtype=np.float32)
    sigma_angle = np.zeros(len(count), dtype=np.float32)
    mean_energy[occupied] = sum_energy[occupied] / count[occupied]
    variance = (
        sum_energy_sq[occupied] / count[occupied]
        - np.square(mean_energy[occupied])
    )
    sigma_energy[occupied] = np.sqrt(np.maximum(variance, 0.0))
    sigma_angle[occupied] = np.sqrt(
        sum_angle_sq[occupied] / (2.0 * count[occupied])
    )
    sigma_energy *= ENERGY_SIGMA_CUT
    sigma_angle[sigma_angle == 0.0] = 1.0
    sigma_angle *= ANGLE_SIGMA_CUT

    selected = np.zeros(len(pairs), dtype=bool)
    candidate = np.flatnonzero(inside)
    candidate_pixel = pixel[candidate]
    selected[candidate] = (
        (angle_x[candidate] <= sigma_angle[candidate_pixel])
        & (angle_y[candidate] <= sigma_angle[candidate_pixel])
        & (
            np.abs(energy_loss[candidate] - mean_energy[candidate_pixel])
            <= sigma_energy[candidate_pixel]
        )
    )
    diagnostics = {
        "input": len(pairs),
        "inside_grid": int(np.count_nonzero(inside)),
        "outside_grid": int(np.count_nonzero(~inside)),
        "output": int(np.count_nonzero(selected)),
        "removed": int(np.count_nonzero(~selected)),
        "removed_inside_grid_by_3sigma": int(np.count_nonzero(inside & ~selected)),
        "occupied_pixels": int(np.count_nonzero(occupied)),
        "zero_count_pixels": int(np.count_nonzero(~occupied)),
        "input_primary": int(np.count_nonzero(pairs[:, 4, 2] == 1)),
        "output_primary": int(np.count_nonzero(pairs[selected, 4, 2] == 1)),
    }
    return pairs[selected], diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FILTERED)
    parser.add_argument("--qc-dir", type=Path, default=DEFAULT_QC)
    args = parser.parse_args()
    pairs_dir = args.pairs_dir
    filtered_dir = args.output_dir
    qc_dir = args.qc_dir
    filtered_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    start_clock = time.perf_counter()
    rows = []
    for run_id in range(RUNS):
        input_path = pairs_dir / f"pairs{run_id:04d}.mhd"
        pairs = read_mhd(input_path)
        filtered, diagnostics = filter_pairs(pairs)
        write_mhd(filtered_dir / f"pairs{run_id:04d}.mhd", filtered)
        diagnostics["run_id"] = run_id
        diagnostics["retained_fraction"] = len(filtered) / len(pairs)
        rows.append(diagnostics)
        if run_id % 50 == 0 or run_id == RUNS - 1:
            print(f"run {run_id:04d}: {len(pairs)} -> {len(filtered)}")

    elapsed = time.perf_counter() - start_clock
    stopped = datetime.now()
    fieldnames = [
        "run_id",
        "input",
        "inside_grid",
        "outside_grid",
        "output",
        "removed",
        "removed_inside_grid_by_3sigma",
        "occupied_pixels",
        "zero_count_pixels",
        "input_primary",
        "output_primary",
        "retained_fraction",
    ]
    with (qc_dir / "paircuts_runs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    totals = {
        key: sum(row[key] for row in rows)
        for key in fieldnames
        if key not in {"run_id", "retained_fraction"}
    }
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "stopped": stopped.isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "algorithm": "default non-robust applications/pctpaircuts/pctpaircuts.cxx",
        "grid_size": list(GRID_SIZE),
        "grid_spacing_mm": list(GRID_SPACING_MM),
        "grid_origin_mm": list(GRID_ORIGIN_MM),
        "source_mm": SOURCE_MM,
        "energy_sigma_cut": ENERGY_SIGMA_CUT,
        "angle_sigma_cut": ANGLE_SIGMA_CUT,
        "runs": len(rows),
        "totals": totals,
        "retained_fraction": totals["output"] / totals["input"],
        "inside_grid_3sigma_retained_fraction": totals["output"]
        / totals["inside_grid"],
        "input_directory": str(pairs_dir.resolve()),
        "output_directory": str(filtered_dir.resolve()),
        "output_raw_bytes": sum(path.stat().st_size for path in filtered_dir.glob("*.raw")),
    }
    (qc_dir / "paircuts_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    raise SystemExit("internal module; use run_preprocessing.py")
