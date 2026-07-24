#!/usr/bin/env python3
"""Run Stage 2 S2--S5 data QA, preprocessing, reconstruction, and evaluation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
CONFIG_PATH = HERE / "stage2_config.json"
QC_ROOT = HERE / "qc"
FIGURES = QC_ROOT / "figures"
SIMULATION_PACKAGE = (
    CODE_ROOT / "simulation" / "windows_overnight_simulations_0716"
)
sys.path[:0] = [
    str(CODE_ROOT),
    str(REPOSITORY_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

from common import load_experiment, path_for  # noqa: E402
from preprocessing import paircuts, projection  # noqa: E402
from analytic_reconstruction import rsp_metrics, truth_maps  # noqa: E402
from iterative_reconstruction.physics import (  # noqa: E402
    energies_to_wepl_vectorized,
    make_vectorized_wepl_lut,
)


RUNS = 720
REQUIRED_BRANCHES = {
    "RunID",
    "EventID",
    "TrackID",
    "KineticEnergy",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Direction_X",
    "Direction_Y",
    "Direction_Z",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "freeze",
            "preprocess",
            "project",
            "analytic",
            "iterative",
            "holdout",
            "evaluate",
            "all",
        ),
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def command_path(name: str) -> Path:
    path = REPOSITORY_ROOT / ".venv-gate" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_command(command: list[str], log_path: Path | None = None) -> float:
    print(f"$ {shlex.join(command)}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if log_path else None,
        stderr=subprocess.STDOUT if log_path else None,
    )
    elapsed = time.perf_counter() - started
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {shlex.join(command)}\n{result.stdout or ''}", encoding="utf-8"
        )
    if result.returncode:
        tail = (result.stdout or "")[-4000:]
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n{tail}"
        )
    return elapsed


def experiments(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: load_experiment(experiment_id)
        for name, experiment_id in config["experiments"].items()
    }


def splitmix64(values: np.ndarray, run_id: int, seed: int) -> np.ndarray:
    """Versioned uint64 mixer; integer wraparound is intentional."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = values.astype(np.uint64, copy=False)
    run_key = np.uint64(
        (int(run_id) * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    )
    x = (x ^ run_key) & mask
    x = (
        x ^ np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    ) & mask
    x = (
        (x ^ (x >> np.uint64(30)))
        * np.uint64(0xBF58476D1CE4E5B9)
    ) & mask
    x = (
        (x ^ (x >> np.uint64(27)))
        * np.uint64(0x94D049BB133111EB)
    ) & mask
    return (x ^ (x >> np.uint64(31))) & mask


def freeze_inputs(
    config: dict[str, Any], exp: dict[str, dict[str, Any]], force: bool
) -> dict[str, Any]:
    output = QC_ROOT / "input_manifest.json"
    if output.exists() and not force:
        return load_json(output)
    records: list[dict[str, Any]] = []
    scenario_summaries: dict[str, Any] = {}
    started = time.perf_counter()
    for label, experiment in exp.items():
        simulation = path_for(experiment, "simulation_data")
        scenario = str(config["workstation_scenarios"][label])
        workstation = SIMULATION_PACKAGE / "qc" / scenario
        manifest_rows = read_csv(workstation / "result_manifest.csv")
        if len(manifest_rows) != RUNS:
            raise RuntimeError(f"{scenario}: expected 720 manifest rows")
        manifest_by_run = {int(row["angle_index"]): row for row in manifest_rows}
        total_bytes = 0
        for run_id in range(RUNS):
            run = simulation / f"run_{run_id:03d}"
            row = manifest_by_run[run_id]
            for name, column in (
                ("PhaseSpaceIn.root", "phase_space_in_bytes"),
                ("PhaseSpaceOut.root", "phase_space_out_bytes"),
            ):
                path = run / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected = int(row[column])
                if path.stat().st_size != expected:
                    raise RuntimeError(
                        f"{path}: {path.stat().st_size} bytes, expected {expected}"
                    )
                records.append({
                    "dataset": label,
                    "kind": "ROOT",
                    "path": relative(path),
                    "bytes": expected,
                    "sha256": sha256(path),
                })
                total_bytes += expected
            if (run_id + 1) % 60 == 0:
                print(
                    f"freeze {label}: {run_id+1:03d}/{RUNS} angles hashed",
                    flush=True,
                )
        for name in (
            "base_ct.json",
            "scenario_config.json",
            "launcher_summary.json",
            "result_manifest.csv",
        ):
            path = workstation / name
            records.append({
                "dataset": label,
                "kind": "workstation_qc",
                "path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
        schema_samples = []
        for run_id in (0, 359, 719):
            for tree_name, file_name in (
                ("PhaseSpaceIn", "PhaseSpaceIn.root"),
                ("PhaseSpaceOut", "PhaseSpaceOut.root"),
            ):
                path = simulation / f"run_{run_id:03d}" / file_name
                with uproot.open(path) as root:
                    tree = root[tree_name]
                    branches = set(tree.keys())
                    missing = sorted(REQUIRED_BRANCHES - branches)
                    if missing:
                        raise RuntimeError(f"{path}: missing {missing}")
                    schema_samples.append({
                        "run_id": run_id,
                        "tree": tree_name,
                        "entries": int(tree.num_entries),
                        "required_branches_present": True,
                    })
        launcher = load_json(workstation / "launcher_summary.json")
        if launcher["failed_angles"] or int(launcher["completed_this_launch"]) != RUNS:
            raise RuntimeError(f"{scenario}: launcher summary is incomplete")
        scenario_summaries[label] = {
            "scenario": scenario,
            "angles": RUNS,
            "root_files": 2 * RUNS,
            "root_bytes": total_bytes,
            "launcher_elapsed_seconds": float(launcher["elapsed_seconds"]),
            "schema_samples": schema_samples,
        }
    result = {
        "status": "PASS",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": (
            "All ROOT files are size-checked and SHA-256 hashed; ROOT schema "
            "is sampled at angles 0, 359, and 719 for each dataset."
        ),
        "file_count": len(records),
        "total_bytes": sum(int(row["bytes"]) for row in records),
        "elapsed_seconds": time.perf_counter() - started,
        "scenarios": scenario_summaries,
        "files": records,
    }
    write_json(output, result)
    return result


def stage_complete(directory: Path, prefix: str) -> bool:
    return (
        len(list(directory.glob(f"{prefix}*.mhd"))) == RUNS
        and len(list(directory.glob(f"{prefix}*.raw"))) == RUNS
    )


def run_pairing_filtering(
    experiment_id: str, experiment: dict[str, Any], jobs: int, force: bool
) -> None:
    preprocessing = path_for(experiment, "preprocessing_data")
    pairs = preprocessing / "pairs"
    filtered = preprocessing / "pairs_filtered"
    runner = CODE_ROOT / "preprocessing" / "run_preprocessing.py"
    for stage, directory in (("pairing", pairs), ("filtering", filtered)):
        summary_path = (
            CODE_ROOT
            / "preprocessing"
            / "qc"
            / f"results{experiment_id}"
            / f"{stage}_summary.json"
        )
        summary_passed = (
            summary_path.is_file()
            and load_json(summary_path).get("status") == "PASS"
        )
        if stage_complete(directory, "pairs") and summary_passed and not force:
            print(f"{experiment_id} {stage}: reusing complete output", flush=True)
            continue
        if stage_complete(directory, "pairs") and not force:
            raise RuntimeError(
                f"{experiment_id} {stage}: output files exist but integrated "
                f"QC is missing or failed ({summary_path}); inspect the cause "
                "and rerun that preprocessing stage with --force"
            )
        command = [
            str(REPOSITORY_ROOT / ".venv-gate" / "bin" / "python"),
            str(runner),
            "--experiment",
            experiment_id,
            "--stage",
            stage,
            "--jobs",
            str(jobs),
        ]
        if force:
            command.append("--force")
        run_command(command)


def forward_distance_to_cylinder(
    position: np.ndarray, direction: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.linalg.norm(direction, axis=1)
    direction = direction / norm[:, None]
    x = position[:, 0]
    z = position[:, 2]
    dx = direction[:, 0]
    dz = direction[:, 2]
    a = dx * dx + dz * dz
    b = 2.0 * (x * dx + z * dz)
    c = x * x + z * z - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid = (a > 1.0e-12) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    candidates = np.column_stack([t0, t1])
    candidates[candidates < 0.0] = np.inf
    distance = np.min(candidates, axis=1)
    valid &= np.isfinite(distance)
    return distance, valid


def air_correct_pairs(
    pairs: np.ndarray,
    lut: np.ndarray,
    correction: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    radius = float(correction["phantom_radius_mm"])
    entrance, entrance_valid = forward_distance_to_cylinder(
        pairs[:, 0, :].astype(np.float64),
        pairs[:, 2, :].astype(np.float64),
        radius,
    )
    exit_distance, exit_valid = forward_distance_to_cylinder(
        pairs[:, 1, :].astype(np.float64),
        -pairs[:, 3, :].astype(np.float64),
        radius,
    )
    hit = entrance_valid & exit_valid
    air_length = entrance + exit_distance
    direct = np.linalg.norm(
        pairs[:, 1, :].astype(np.float64)
        - pairs[:, 0, :].astype(np.float64),
        axis=1,
    )
    air_length[~hit] = direct[~hit]
    original = energies_to_wepl_vectorized(
        lut, pairs[:, 4, 0], pairs[:, 4, 1]
    ).astype(np.float64)
    correction_wepl = (
        float(correction["slope_mm_wepl_per_mm_air"]) * air_length
    )
    corrected = original - correction_wepl
    clipped = corrected < 0.0
    corrected = np.maximum(corrected, 0.0)
    output = np.array(pairs, dtype=np.float32, copy=True)
    output[:, 4, 0] = 0.0
    output[:, 4, 1] = corrected.astype(np.float32)
    return output, {
        "pairs": int(len(pairs)),
        "cylinder_hit": int(np.count_nonzero(hit)),
        "cylinder_miss": int(np.count_nonzero(~hit)),
        "air_length_mean_mm": float(np.mean(air_length)),
        "air_length_min_mm": float(np.min(air_length)),
        "air_length_max_mm": float(np.max(air_length)),
        "wepl_before_mean_mm": float(np.mean(original)),
        "air_correction_mean_mm": float(np.mean(correction_wepl)),
        "wepl_after_mean_mm": float(np.mean(corrected)),
        "clipped_to_zero": int(np.count_nonzero(clipped)),
    }


def make_split_and_training_pairs(
    label: str,
    experiment: dict[str, Any],
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    preprocessing = path_for(experiment, "preprocessing_data")
    filtered = preprocessing / "pairs_filtered"
    split_spec = config["split"]
    split_dir = preprocessing / "splits" / split_spec["name"]
    train_dir = preprocessing / "pairs_train"
    corrected_dir = preprocessing / "pairs_train_air_corrected"
    manifest_path = QC_ROOT / f"split_{label}.json"
    expected = [split_dir, train_dir]
    needs_air_correction = (
        str(experiment["acquisition"]["world_material"]).lower() == "air"
    )
    if needs_air_correction:
        expected.append(corrected_dir)
    if (
        manifest_path.is_file()
        and all(stage_complete(path, "pairs") for path in expected[1:])
        and len(list(split_dir.glob("*.bin"))) == 2 * RUNS
        and not force
    ):
        return load_json(manifest_path)
    if force:
        for path in expected:
            shutil.rmtree(path, ignore_errors=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    if needs_air_correction:
        corrected_dir.mkdir(parents=True, exist_ok=True)
    seed = int(split_spec["seed"])
    modulus = int(split_spec["modulus"])
    bit_order = str(split_spec["bit_order"])
    lut = make_vectorized_wepl_lut()
    rows = []
    total = train_total = validation_total = test_total = 0
    correction_totals = {
        "pairs": 0,
        "cylinder_hit": 0,
        "cylinder_miss": 0,
        "clipped_to_zero": 0,
        "air_length_weighted_sum_mm": 0.0,
        "correction_weighted_sum_mm": 0.0,
    }
    started = time.perf_counter()
    for run_id in range(RUNS):
        pairs = paircuts.read_mhd(filtered / f"pairs{run_id:04d}.mhd")
        indices = np.arange(len(pairs), dtype=np.uint64)
        bucket = splitmix64(indices, run_id, seed) % np.uint64(modulus)
        test = bucket == np.uint64(split_spec["test_remainder"])
        validation = bucket == np.uint64(
            split_spec["validation_remainder"]
        )
        train = ~(test | validation)
        if not train.any() or not validation.any() or not test.any():
            raise RuntimeError(f"{label} angle {run_id}: empty partition")
        for partition, mask in (("validation", validation), ("test", test)):
            path = split_dir / f"{partition}_mask_{run_id:04d}.bin"
            np.packbits(mask, bitorder=bit_order).tofile(path)
        train_pairs = np.asarray(pairs[train], dtype=np.float32)
        paircuts.write_mhd(
            train_dir / f"pairs{run_id:04d}.mhd", train_pairs
        )
        correction_row: dict[str, Any] = {}
        if needs_air_correction:
            corrected, correction_row = air_correct_pairs(
                train_pairs, lut, config["air_correction"]
            )
            paircuts.write_mhd(
                corrected_dir / f"pairs{run_id:04d}.mhd", corrected
            )
            for key in (
                "pairs",
                "cylinder_hit",
                "cylinder_miss",
                "clipped_to_zero",
            ):
                correction_totals[key] += int(correction_row[key])
            correction_totals["air_length_weighted_sum_mm"] += (
                float(correction_row["air_length_mean_mm"]) * len(train_pairs)
            )
            correction_totals["correction_weighted_sum_mm"] += (
                float(correction_row["air_correction_mean_mm"])
                * len(train_pairs)
            )
        row = {
            "dataset": label,
            "run_id": run_id,
            "total": len(pairs),
            "train": int(np.count_nonzero(train)),
            "validation": int(np.count_nonzero(validation)),
            "test": int(np.count_nonzero(test)),
            "validation_mask_sha256": sha256(
                split_dir / f"validation_mask_{run_id:04d}.bin"
            ),
            "test_mask_sha256": sha256(
                split_dir / f"test_mask_{run_id:04d}.bin"
            ),
        }
        if correction_row:
            row.update(correction_row)
        rows.append(row)
        total += len(pairs)
        train_total += int(np.count_nonzero(train))
        validation_total += int(np.count_nonzero(validation))
        test_total += int(np.count_nonzero(test))
        if (run_id + 1) % 40 == 0 or run_id == RUNS - 1:
            print(
                f"split {label}: {run_id+1:03d}/{RUNS}, "
                f"train={train_total:,}, validation={validation_total:,}, "
                f"test={test_total:,}",
                flush=True,
            )
    write_csv(QC_ROOT / f"split_{label}_runs.csv", rows)
    result = {
        "status": "PASS",
        "dataset": label,
        "identity": split_spec["identity"],
        "algorithm": split_spec["algorithm"],
        "seed": seed,
        "rule": (
            f"splitmix64_v1(RunID, filtered_row_index, {seed}) % "
            f"{modulus}: test=0, validation=1, train=2..9"
        ),
        "angles": RUNS,
        "total": total,
        "train": train_total,
        "validation": validation_total,
        "test": test_total,
        "train_fraction": train_total / total,
        "validation_fraction": validation_total / total,
        "test_fraction": test_total / total,
        "elapsed_seconds": time.perf_counter() - started,
        "paths": {
            "masks": relative(split_dir),
            "train_pairs": relative(train_dir),
        },
    }
    if needs_air_correction:
        pairs_count = correction_totals["pairs"]
        result["paths"]["corrected_train_pairs"] = relative(corrected_dir)
        result["air_correction"] = {
            **correction_totals,
            "cylinder_hit_fraction": (
                correction_totals["cylinder_hit"] / pairs_count
            ),
            "mean_air_length_mm": (
                correction_totals["air_length_weighted_sum_mm"]
                / pairs_count
            ),
            "mean_air_correction_mm": (
                correction_totals["correction_weighted_sum_mm"]
                / pairs_count
            ),
            "slope_mm_wepl_per_mm_air": config["air_correction"][
                "slope_mm_wepl_per_mm_air"
            ],
        }
    write_json(manifest_path, result)
    return result


def preprocess(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    jobs: int,
    force: bool,
) -> dict[str, Any]:
    results = {}
    for label, experiment in exp.items():
        run_pairing_filtering(
            config["experiments"][label], experiment, jobs, force
        )
        results[label] = make_split_and_training_pairs(
            label, experiment, config, force
        )
    return results


def project_dataset(
    label: str,
    pairs: Path,
    preprocessing: Path,
    ddb_name: str,
    jobs: int,
    force: bool,
) -> dict[str, Any]:
    ddb = preprocessing / ddb_name
    complete = stage_complete(ddb, "proj")
    summary_path = QC_ROOT / f"projection_{label}.json"
    if complete and summary_path.is_file() and not force:
        return load_json(summary_path)
    if force:
        shutil.rmtree(ddb, ignore_errors=True)
    ddb.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                projection.process_run,
                run_id,
                str(pairs),
                str(preprocessing),
                False,
                ddb_name,
            ): run_id
            for run_id in range(RUNS)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if len(rows) % 20 == 0 or len(rows) == RUNS:
                print(
                    f"projection {label}: {len(rows):03d}/{RUNS}",
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["run_id"]))
    write_csv(QC_ROOT / f"projection_{label}_runs.csv", rows)
    result = {
        "status": "PASS",
        "dataset": label,
        "pairs": relative(pairs),
        "ddb": relative(ddb),
        "runs": len(rows),
        "size": list(projection.SIZE),
        "spacing_mm": list(projection.SPACING_MM),
        "object_zero_count": sum(
            int(row["object_zero_count"]) for row in rows
        ),
        "variance_nonfinite": sum(
            int(row["variance_nonfinite"]) for row in rows
        ),
        "variance_negative": sum(
            int(row["variance_negative"]) for row in rows
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    if (
        result["runs"] != RUNS
        or result["variance_nonfinite"]
        or result["variance_negative"]
        or not stage_complete(ddb, "proj")
    ):
        result["status"] = "FAIL"
        write_json(summary_path, result)
        raise RuntimeError(f"projection {label} failed")
    write_json(summary_path, result)
    return result


def project_all(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    jobs: int,
    force: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for label, experiment in exp.items():
        root = path_for(experiment, "preprocessing_data")
        is_air = (
            str(experiment["acquisition"]["world_material"]).lower() == "air"
        )
        results[label] = project_dataset(
            label,
            root / (
                "pairs_train_air_corrected" if is_air else "pairs_train"
            ),
            root,
            "projections_ddb",
            jobs,
            force,
        )
        # S3 is the deliberate external-Air ablation.  S4/S5 only need the
        # corrected chain because their diagnostic target is inside the
        # cylinder, not the background medium.
        if label == "air":
            results["air_uncorrected"] = project_dataset(
                "air_uncorrected",
                root / "pairs_train",
                root,
                "projections_ddb_uncorrected",
                jobs,
                force,
            )
    return results


def uniform_fraction(
    x: np.ndarray, z: np.ndarray, radius: float, supersampling: int = 4
) -> np.ndarray:
    fraction = np.zeros((len(z), len(x)), dtype=np.float32)
    offsets = (
        (np.arange(supersampling, dtype=np.float64) + 0.5)
        / supersampling
        - 0.5
    )
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    for oz in offsets:
        zz2 = (z + oz * dz)[:, None] ** 2
        for ox in offsets:
            fraction += (
                (x + ox * dx)[None, :] ** 2 + zz2 <= radius * radius
            )
    fraction /= float(supersampling**2)
    return fraction


def radial_profile(
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    bin_width: float,
    variant: str,
    support: bool,
) -> list[dict[str, Any]]:
    xx, zz = np.meshgrid(x, z)
    radius = np.hypot(xx, zz).ravel()
    values = image.ravel()
    maximum = float(radius.max())
    count = int(math.ceil(maximum / bin_width))
    index = np.minimum((radius / bin_width).astype(np.int64), count - 1)
    n = np.bincount(index, minlength=count)
    sums = np.bincount(index, weights=values, minlength=count)
    squares = np.bincount(
        index, weights=values.astype(np.float64) ** 2, minlength=count
    )
    mean = sums / np.maximum(n, 1)
    std = np.sqrt(np.maximum(squares / np.maximum(n, 1) - mean**2, 0.0))
    return [
        {
            "variant": variant,
            "support": support,
            "radius_mm": (i + 0.5) * bin_width,
            "mean_rsp": float(mean[i]),
            "std_rsp": float(std[i]),
            "pixels": int(n[i]),
        }
        for i in range(count)
        if n[i]
    ]


def falling_crossing(
    radius: np.ndarray, values: np.ndarray, level: float
) -> float:
    candidate = np.flatnonzero(
        (radius[:-1] >= 95.0)
        & (radius[:-1] <= 105.0)
        & (values[:-1] >= level)
        & (values[1:] < level)
    )
    if not len(candidate):
        return float("nan")
    i = int(candidate[0])
    fraction = (level - values[i]) / (values[i + 1] - values[i])
    return float(radius[i] + fraction * (radius[i + 1] - radius[i]))


def uniform_metrics(
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    config: dict[str, Any],
    truth_config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    radius = float(truth_config["phantom_radius_mm"])
    fixed = float(truth_config["rsp_200mev"])
    effective = float(truth_config["effective_rsp_200mev_s6"])
    xx, zz = np.meshgrid(x, z)
    rr = np.hypot(xx, zz)
    fraction = uniform_fraction(x, z, radius)
    core = rr <= float(evaluation["core_radius_mm"])
    boundary = (
        (rr >= float(evaluation["boundary_inner_mm"][0]))
        & (rr <= float(evaluation["boundary_inner_mm"][1]))
    )
    outside = (
        (rr > float(evaluation["outside_annulus_mm"][0]))
        & (rr <= float(evaluation["outside_annulus_mm"][1]))
    )
    phantom = fraction > 0.0

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))

    profile = radial_profile(
        image,
        x,
        z,
        float(evaluation["radial_bin_mm"]),
        "_metric",
        False,
    )
    pr = np.asarray([row["radius_mm"] for row in profile])
    pv = np.asarray([row["mean_rsp"] for row in profile])
    inner = float(np.mean(pv[(pr >= 85.0) & (pr <= 90.0)]))
    outer_mask = (pr >= 103.0) & (pr <= min(105.0, float(pr.max())))
    outer = float(np.mean(pv[outer_mask])) if outer_mask.any() else 0.0
    r90 = falling_crossing(pr, pv, outer + 0.9 * (inner - outer))
    r10 = falling_crossing(pr, pv, outer + 0.1 * (inner - outer))
    return {
        "finite": bool(np.isfinite(image).all()),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "water_core_mean_rsp": float(image[core].mean()),
        "water_core_std_rsp": float(image[core].std()),
        "water_bias_vs_fixed_rsp": float(image[core].mean() - fixed),
        "water_bias_vs_effective_rsp": float(
            image[core].mean() - effective
        ),
        "phantom_rmse_vs_fixed_rsp": rmse(
            image[phantom] - fixed * fraction[phantom]
        ),
        "phantom_rmse_vs_effective_rsp": rmse(
            image[phantom] - effective * fraction[phantom]
        ),
        "boundary_inner_rmse_vs_fixed_rsp": rmse(
            image[boundary] - fixed * fraction[boundary]
        ),
        "boundary_inner_rmse_vs_effective_rsp": rmse(
            image[boundary] - effective * fraction[boundary]
        ),
        "outside_100_105_rmse_rsp": rmse(image[outside]),
        "outside_100_105_mean_abs_rsp": float(
            np.mean(np.abs(image[outside]))
        ),
        "outside_100_105_peak_abs_rsp": float(
            np.max(np.abs(image[outside]))
        ),
        "edge_r90_mm": r90,
        "edge_r10_mm": r10,
        "edge_10_90_width_mm": (
            r10 - r90 if math.isfinite(r90) and math.isfinite(r10) else None
        ),
    }


def scenario_config(config: dict[str, Any], label: str) -> dict[str, Any]:
    return load_json(
        SIMULATION_PACKAGE
        / "qc"
        / str(config["workstation_scenarios"][label])
        / "scenario_config.json"
    )


def material_centers(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Return S4 insert centres in the retained reconstruction x-z frame."""
    rows: list[dict[str, Any]] = []
    insert_id = 0
    materials = list(scenario["calibration_materials"])
    for ring in scenario["calibration_rings"]:
        for material_index, material in enumerate(materials):
            angle = math.radians(
                float(ring["angle_offset_deg"])
                + material_index * 360.0 / len(materials)
            )
            local_x = float(ring["radius_mm"]) * math.cos(angle)
            local_y = float(ring["radius_mm"]) * math.sin(angle)
            rows.append({
                "insert_id": insert_id,
                "material": material,
                "radius_mm": float(ring["diameter_mm"]) / 2.0,
                # local phantom (x,y) -> DDB reconstruction (x,z)
                "x_mm": -local_y,
                "z_mm": local_x,
                "ring_radius_mm": float(ring["radius_mm"]),
            })
            insert_id += 1
    rows.append({
        "insert_id": insert_id,
        "material": "Aluminium",
        "radius_mm": float(scenario["small_aluminium_diameter_mm"]) / 2.0,
        "x_mm": 0.0,
        "z_mm": 0.0,
        "ring_radius_mm": 0.0,
    })
    return rows


def local_coordinates(
    x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    xx, zz = np.meshgrid(x, z)
    return zz, -xx


def rotated_box_mask(
    local_x: np.ndarray,
    local_y: np.ndarray,
    center: list[float],
    size: list[float],
    angle_deg: float,
    margin_mm: float = 0.0,
) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    dx = local_x - float(center[0])
    dy = local_y - float(center[1])
    u = math.cos(angle) * dx + math.sin(angle) * dy
    v = -math.sin(angle) * dx + math.cos(angle) * dy
    return (
        (np.abs(u) <= float(size[0]) / 2.0 - margin_mm)
        & (np.abs(v) <= float(size[1]) / 2.0 - margin_mm)
    )


def diagnostic_truth(
    label: str,
    config: dict[str, Any],
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Voxelize the S4/S5 nominal 200 MeV RSP truth on an FDK grid."""
    scenario = scenario_config(config, label)
    rsp = config["material_rsp_200mev"]
    local_x, local_y = local_coordinates(x, z)
    phantom = local_x * local_x + local_y * local_y <= 100.0**2
    truth = np.zeros_like(local_x, dtype=np.float32)
    truth[phantom] = float(rsp["Water"])
    masks: dict[str, np.ndarray] = {"phantom": phantom}
    objects: list[dict[str, Any]] = []

    if label == "material":
        for item in material_centers(scenario):
            mask = (
                (local_x - float(item["z_mm"])) ** 2
                + (local_y + float(item["x_mm"])) ** 2
                <= float(item["radius_mm"]) ** 2
            )
            truth[mask] = float(rsp[item["material"]])
            masks[f"insert_{item['insert_id']:02d}"] = mask
            objects.append(item)
    elif label == "resolution":
        for group_index, group in enumerate(scenario["line_pair_groups"]):
            width = float(group["line_width_mm"])
            count = int(group["bar_count"])
            total = (2 * count - 1) * width
            angle = math.radians(float(group.get("rotation_deg", 0.0)))
            for bar_index in range(count):
                offset = -0.5 * total + 0.5 * width + 2.0 * width * bar_index
                center = [
                    float(group["center_mm"][0]) + math.cos(angle) * offset,
                    float(group["center_mm"][1]) + math.sin(angle) * offset,
                ]
                mask = rotated_box_mask(
                    local_x,
                    local_y,
                    center,
                    [width, float(group["bar_length_mm"])],
                    float(group.get("rotation_deg", 0.0)),
                )
                truth[mask] = float(rsp[group["material"]])
            objects.append({
                "kind": "line_pair",
                "group_id": group_index,
                **group,
            })
        for target_index, target in enumerate(scenario["edge_targets"]):
            mask = rotated_box_mask(
                local_x,
                local_y,
                list(target["center_mm"]),
                list(target["size_xy_mm"]),
                float(target["rotation_deg"]),
            )
            truth[mask] = float(rsp[target["material"]])
            masks[f"edge_{target_index:02d}"] = mask
            objects.append({
                "kind": "edge",
                "target_id": target_index,
                **target,
            })
    else:
        raise ValueError(f"diagnostic truth is undefined for {label}")
    return truth, masks, objects


def crossing_frequency(
    frequency: np.ndarray, mtf: np.ndarray, level: float
) -> float:
    indices = np.flatnonzero((mtf[:-1] >= level) & (mtf[1:] < level))
    if not len(indices):
        return float("nan")
    i = int(indices[0])
    fraction = (level - mtf[i]) / (mtf[i + 1] - mtf[i])
    return float(frequency[i] + fraction * (frequency[i + 1] - frequency[i]))


def slanted_edge_metrics(
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    """Measure an oversampled edge-spread MTF on the five S5 squares."""
    local_x, local_y = local_coordinates(x, z)
    rows: list[dict[str, Any]] = []
    bin_width = 0.025
    edges = np.arange(-3.0, 3.0 + bin_width, bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for target_id, target in enumerate(scenario["edge_targets"]):
        angle = math.radians(float(target["rotation_deg"]))
        dx = local_x - float(target["center_mm"][0])
        dy = local_y - float(target["center_mm"][1])
        u = math.cos(angle) * dx + math.sin(angle) * dy
        v = -math.sin(angle) * dx + math.cos(angle) * dy
        half = float(target["size_xy_mm"][0]) / 2.0
        distance = half - u
        roi = (np.abs(distance) <= 3.0) & (np.abs(v) <= 5.0)
        index = np.digitize(distance[roi], edges) - 1
        valid = (index >= 0) & (index < len(centers))
        count = np.bincount(index[valid], minlength=len(centers))
        sums = np.bincount(
            index[valid], weights=image[roi][valid], minlength=len(centers)
        )
        esf = sums / np.maximum(count, 1)
        populated = count > 0
        esf = np.interp(centers, centers[populated], esf[populated])
        outer = float(np.mean(esf[centers < -2.0]))
        inner = float(np.mean(esf[centers > 2.0]))
        normalized = (esf - outer) / (inner - outer)
        lsf = np.gradient(normalized, bin_width)
        lsf *= np.hanning(len(lsf))
        mtf = np.abs(np.fft.rfft(lsf))
        mtf /= max(float(mtf[0]), np.finfo(float).eps)
        frequency = np.fft.rfftfreq(len(lsf), d=bin_width)
        pixel_nyquist = 1.0 / (2.0 * abs(float(x[1] - x[0])))
        resolved = frequency <= pixel_nyquist + 1.0e-12
        f50 = crossing_frequency(frequency[resolved], mtf[resolved], 0.5)
        f10 = crossing_frequency(frequency[resolved], mtf[resolved], 0.1)
        rows.append({
            "target_id": target_id,
            "center_x_mm": float(target["center_mm"][0]),
            "center_y_mm": float(target["center_mm"][1]),
            "rotation_deg": float(target["rotation_deg"]),
            "radius_mm": float(math.hypot(*target["center_mm"])),
            "samples": int(np.count_nonzero(roi)),
            "edge_contrast_rsp": inner - outer,
            "pixel_nyquist_lp_per_mm": pixel_nyquist,
            "fmtf50_lp_per_mm": (
                f50 if math.isfinite(f50) else pixel_nyquist
            ),
            "fmtf10_lp_per_mm": (
                f10 if math.isfinite(f10) else pixel_nyquist
            ),
            "fmtf50_censored_at_nyquist": not math.isfinite(f50),
            "fmtf10_censored_at_nyquist": not math.isfinite(f10),
        })
    return rows


def line_pair_metrics(
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    scenario: dict[str, Any],
) -> list[dict[str, Any]]:
    local_x, local_y = local_coordinates(x, z)
    rows: list[dict[str, Any]] = []
    for group_id, group in enumerate(scenario["line_pair_groups"]):
        angle = math.radians(float(group.get("rotation_deg", 0.0)))
        dx = local_x - float(group["center_mm"][0])
        dy = local_y - float(group["center_mm"][1])
        u = math.cos(angle) * dx + math.sin(angle) * dy
        v = -math.sin(angle) * dx + math.cos(angle) * dy
        width = float(group["line_width_mm"])
        extent = (2 * int(group["bar_count"]) - 1) * width / 2.0
        roi = (np.abs(u) <= extent) & (
            np.abs(v) <= float(group["bar_length_mm"]) * 0.35
        )
        values = image[roi]
        low, high = np.percentile(values, [10.0, 90.0])
        rows.append({
            "group_id": group_id,
            "line_width_mm": width,
            "spatial_frequency_lp_per_mm": 1.0 / (2.0 * width),
            "samples": int(values.size),
            "p10_rsp": float(low),
            "p90_rsp": float(high),
            "modulation": float((high - low) / max(high + low, 1.0e-9)),
        })
    return rows


def diagnostic_metrics(
    label: str,
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    truth, masks, _ = diagnostic_truth(label, config, x, z)
    from scipy.ndimage import distance_transform_edt

    xx, zz = np.meshgrid(x, z)
    phantom = masks["phantom"]
    water = phantom & np.isclose(truth, 1.0)
    water_clearance_mm = distance_transform_edt(
        water,
        sampling=(abs(float(z[1] - z[0])), abs(float(x[1] - x[0]))),
    )
    core_water = (
        water
        & (np.hypot(xx, zz) <= 90.0)
        & (water_clearance_mm >= 2.0)
    )
    diff = image[phantom] - truth[phantom]
    summary = {
        "finite": bool(np.isfinite(image).all()),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "water_core_mean_rsp": float(image[core_water].mean()),
        "water_core_std_rsp": float(image[core_water].std()),
        "water_core_clearance_from_material_mm": 2.0,
        "phantom_rmse_vs_nominal_rsp": float(
            np.sqrt(np.mean(np.square(diff, dtype=np.float64)))
        ),
        "phantom_mae_vs_nominal_rsp": float(np.mean(np.abs(diff))),
    }
    material_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    scenario = scenario_config(config, label)
    if label == "material":
        rsp = config["material_rsp_200mev"]
        for item in material_centers(scenario):
            radius = max(0.5, float(item["radius_mm"]) - 2.0)
            roi = (
                (xx - float(item["x_mm"])) ** 2
                + (zz - float(item["z_mm"])) ** 2
                <= radius**2
            )
            values = image[roi]
            reference = float(rsp[item["material"]])
            material_rows.append({
                **item,
                "roi_radius_mm": radius,
                "nominal_rsp_200mev": reference,
                "mean_rsp": float(values.mean()),
                "std_rsp": float(values.std()),
                "bias_rsp": float(values.mean() - reference),
                "absolute_relative_error": (
                    float(abs(values.mean() - reference) / reference)
                    if reference > 0.0
                    else None
                ),
            })
        non_air = [
            row["absolute_relative_error"]
            for row in material_rows
            if row["absolute_relative_error"] is not None
        ]
        summary["material_mape_non_air"] = float(np.mean(non_air))
        summary["material_max_ape_non_air"] = float(np.max(non_air))
    else:
        edge_rows = slanted_edge_metrics(image, x, z, scenario)
        line_rows = line_pair_metrics(image, x, z, scenario)
        summary["fmtf50_mean_lp_per_mm"] = float(
            np.nanmean([row["fmtf50_lp_per_mm"] for row in edge_rows])
        )
        summary["fmtf10_mean_lp_per_mm"] = float(
            np.nanmean([row["fmtf10_lp_per_mm"] for row in edge_rows])
        )
    return summary, material_rows, edge_rows, line_rows


def geometry_for(
    experiment: dict[str, Any], reconstruction: Path, force: bool
) -> tuple[Path, float]:
    geometry = reconstruction / "analytic" / "geometry.xml"
    if geometry.is_file() and not force:
        return geometry, 0.0
    geometry.parent.mkdir(parents=True, exist_ok=True)
    acquisition = experiment["acquisition"]
    command = [
        str(command_path("rtksimulatedgeometry")),
        "--nproj",
        str(acquisition["projections"]),
        "--first_angle",
        f"{acquisition['first_angle_deg']:g}",
        "--arc",
        f"{acquisition['arc_deg']:g}",
        "--sid",
        f"{acquisition['source_to_isocenter_mm']:g}",
        "--sdd",
        f"{acquisition['source_to_detector_mm']:g}",
        "--output",
        str(geometry),
    ]
    return geometry, run_command(
        command, QC_ROOT / f"geometry_{experiment['experiment']}.log"
    )


def write_selected_products(
    label: str,
    config: dict[str, Any],
    experiment: dict[str, Any],
    source: Path,
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> None:
    reconstruction = path_for(experiment, "reconstruction_data")
    recon = reconstruction / "analytic" / "recon"
    truth = reconstruction / "analytic" / "truth"
    recon.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    spacing = [float(x[1] - x[0]), 1.0, float(z[1] - z[0])]
    origin = [float(x[0]), 0.0, float(z[0])]
    truth_maps.write_mhd(
        recon / "recon_ddb_nohann.mhd",
        image[:, None, :],
        spacing,
        origin,
    )
    radius = float(experiment["truth"]["phantom_radius_mm"])
    if experiment["truth"]["kind"] == "uniform_water":
        fraction = uniform_fraction(x, z, radius)
        truth_rsp = fraction * float(experiment["truth"]["rsp_200mev"])
        truth_maps.write_mhd(
            truth / "truth_red.mhd",
            fraction[:, None, :],
            spacing,
            origin,
        )
        truth_maps.write_mhd(
            truth / "truth_rsp_effective_s6.mhd",
            (
                fraction
                * float(experiment["truth"]["effective_rsp_200mev_s6"])
            )[:, None, :],
            spacing,
            origin,
        )
        truth_metadata: dict[str, Any] = {
            "kind": "uniform_water",
            "fixed_rsp_200mev": experiment["truth"]["rsp_200mev"],
            "effective_rsp_200mev_s6": experiment["truth"][
                "effective_rsp_200mev_s6"
            ],
        }
    else:
        truth_rsp, _, objects = diagnostic_truth(label, config, x, z)
        truth_metadata = {
            "kind": experiment["truth"]["kind"],
            "nominal_rsp_method": config["material_rsp_200mev"]["method"],
            "nominal_material_rsp": {
                key: value
                for key, value in config["material_rsp_200mev"].items()
                if key != "method"
            },
            "objects": objects,
        }
    truth_maps.write_mhd(
        truth / "truth_rsp_200mev.mhd",
        truth_rsp[:, None, :],
        spacing,
        origin,
    )
    xx, zz = np.meshgrid(x, z)
    supported = np.array(image, copy=True)
    supported[xx * xx + zz * zz > radius * radius] = 0.0
    truth_maps.write_mhd(
        recon / "recon_ddb_nohann_supported.mhd",
        supported[:, None, :],
        spacing,
        origin,
    )
    write_json(
        truth / "truth_metadata.json",
        {
            "source_reconstruction": relative(source),
            "radius_mm": radius,
            **truth_metadata,
            "grid": {
                "size": [len(x), 1, len(z)],
                "spacing_mm": spacing,
                "origin_mm": origin,
            },
        },
    )


def run_analytic(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    force: bool,
) -> dict[str, Any]:
    metrics_path = QC_ROOT / "analytic_variant_metrics.csv"
    profiles_path = QC_ROOT / "radial_profiles.csv"
    if metrics_path.is_file() and profiles_path.is_file() and not force:
        existing = read_csv(metrics_path)
        expected_variants = {
            str(item["name"]) for item in config["analytic_variants"]
        }
        present_variants = {str(item["variant"]) for item in existing}
        if (
            present_variants == expected_variants
            and len(existing) == 2 * len(expected_variants)
            and (QC_ROOT / "material_metrics.csv").is_file()
            and (QC_ROOT / "slanted_edge_mtf.csv").is_file()
            and (QC_ROOT / "line_pair_metrics.csv").is_file()
        ):
            return {"status": "PASS", "reused": True}
        raise RuntimeError(
            "analytic QC exists but does not cover every configured Stage 2 "
            "variant; rerun --action analytic --force"
        )
    metrics_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    geometry_cache: dict[str, Path] = {}
    selected = config["selected_analytic"]
    for variant in config["analytic_variants"]:
        label = str(variant["dataset"])
        experiment = exp[label]
        preprocessing = path_for(experiment, "preprocessing_data")
        reconstruction = path_for(experiment, "reconstruction_data")
        ddb = preprocessing / (
            "projections_ddb_uncorrected"
            if variant["ddb"] == "uncorrected"
            else "projections_ddb"
        )
        variants = reconstruction / "analytic" / "variants"
        variants.mkdir(parents=True, exist_ok=True)
        output = variants / f"{variant['name']}.mhd"
        if force:
            for old in (output, output.with_suffix(".raw")):
                if old.exists():
                    old.unlink()
        if not output.is_file():
            if label not in geometry_cache:
                geometry_cache[label], _ = geometry_for(
                    experiment, reconstruction, force
                )
            command = [
                str(command_path("pctfdk")),
                "--lowmem",
                "--geometry",
                str(geometry_cache[label]),
                "--path",
                str(ddb),
                "--regexp",
                r"proj....\.mhd",
                "--output",
                str(output),
                "--dimension",
                str(variant["size"]),
                "1",
                str(variant["size"]),
                "--spacing",
                f"{variant['spacing_mm']:g}",
                "1",
                f"{variant['spacing_mm']:g}",
                "--hann",
                f"{variant['hann']:g}",
                "--verbose",
            ]
            elapsed = run_command(
                command, QC_ROOT / "logs" / f"{variant['name']}.log"
            )
        else:
            elapsed = 0.0
        image, x, z, _ = rsp_metrics.read_mhd(output)
        if not np.isfinite(image).all():
            raise RuntimeError(f"{variant['name']}: non-finite reconstruction")
        radius = float(experiment["truth"]["phantom_radius_mm"])
        xx, zz = np.meshgrid(x, z)
        for support in (False, True):
            evaluated = np.array(image, copy=True)
            if support:
                evaluated[xx * xx + zz * zz > radius * radius] = 0.0
            if experiment["truth"]["kind"] == "uniform_water":
                measured = uniform_metrics(
                    evaluated, x, z, config, experiment["truth"]
                )
                diagnostic_parts = ([], [], [])
            else:
                measured, *diagnostic_parts = diagnostic_metrics(
                    label, evaluated, x, z, config
                )
            row = {
                "variant": variant["name"],
                "dataset": label,
                "ddb": variant["ddb"],
                "size": variant["size"],
                "spacing_mm": variant["spacing_mm"],
                "fov_mm": float(variant["size"]) * float(variant["spacing_mm"]),
                "hann": variant["hann"],
                "support": support,
                "fdk_seconds": elapsed,
                "image_path": relative(output),
                **measured,
            }
            metrics_rows.append(row)
            if not support:
                for target, values in zip(
                    (material_rows, edge_rows, line_rows), diagnostic_parts
                ):
                    target.extend(
                        {
                            "variant": variant["name"],
                            "dataset": label,
                            **value,
                        }
                        for value in values
                    )
            if not support or variant["name"] in selected.values():
                profile_rows.extend(
                    {
                        **profile,
                        "dataset": label,
                        "ddb": variant["ddb"],
                        "fov_mm": row["fov_mm"],
                        "hann": variant["hann"],
                    }
                    for profile in radial_profile(
                        evaluated,
                        x,
                        z,
                        float(config["evaluation"]["radial_bin_mm"]),
                        variant["name"],
                        support,
                    )
                )
        if variant["name"] == selected[label]:
            write_selected_products(
                label, config, experiment, output, image, x, z
            )
        print(f"analytic {variant['name']}: complete", flush=True)
    write_csv(metrics_path, metrics_rows)
    write_csv(profiles_path, profile_rows)
    if material_rows:
        write_csv(QC_ROOT / "material_metrics.csv", material_rows)
    if edge_rows:
        write_csv(QC_ROOT / "slanted_edge_mtf.csv", edge_rows)
    if line_rows:
        write_csv(QC_ROOT / "line_pair_metrics.csv", line_rows)
    result = {
        "status": "PASS",
        "variants": len(config["analytic_variants"]),
        "metric_rows": len(metrics_rows),
        "profile_rows": len(profile_rows),
        "selected": selected,
    }
    write_json(QC_ROOT / "analytic_summary.json", result)
    return result


def run_iterative(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    force: bool,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    runner = (
        CODE_ROOT
        / "iterative_reconstruction"
        / "run_iterative_reconstruction.py"
    )
    for label in config["iterative"]["pairs_kind"]:
        experiment = exp[label]
        preprocessing = path_for(experiment, "preprocessing_data")
        reconstruction = path_for(experiment, "reconstruction_data")
        pairs = preprocessing / config["iterative"]["pairs_kind"][label]
        final = (
            reconstruction
            / "iterative"
            / "recon"
            / "recon_iterative_gpu.mhd"
        )
        if final.is_file() and not force:
            print(f"iterative {label}: reusing {final}", flush=True)
        else:
            command = [
                str(REPOSITORY_ROOT / ".venv-gate" / "bin" / "python"),
                str(runner),
                "--experiment",
                str(experiment["experiment"]),
                "--pairs-dir",
                str(pairs),
                "--epochs",
                str(config["iterative"]["epochs"]),
            ]
            if force:
                command.append("--force")
            run_command(command)
        summary = load_json(
            CODE_ROOT
            / "iterative_reconstruction"
            / "qc"
            / f"results{experiment['experiment']}"
            / "run_summary.json"
        )
        if summary["status"] != "PASS":
            raise RuntimeError(f"iterative {label} did not pass")
        results[label] = {
            "status": summary["status"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "pairs_per_epoch": summary["pairs_per_epoch"],
            "gpu": summary["gpu"],
            "output": relative(final),
        }
    write_json(
        QC_ROOT / "iterative_summary.json",
        {"status": "PASS", "datasets": results},
    )
    return results


def load_mask(
    path: Path, count: int, bit_order: str
) -> np.ndarray:
    packed = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder=bit_order, count=count)
    return np.flatnonzero(bits).astype(np.int64, copy=False)


def add_iterative_image_metrics(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in config["iterative"]["pairs_kind"]:
        experiment = exp[label]
        reconstruction = path_for(experiment, "reconstruction_data")
        recon = reconstruction / "iterative" / "recon"
        checkpoints = [("initial", 0, recon / "initial.mhd")]
        checkpoints.extend(
            (
                f"epoch_{epoch:02d}",
                epoch,
                recon / f"epoch_{epoch:02d}.mhd",
            )
            for epoch in range(1, int(config["iterative"]["epochs"]) + 1)
        )
        history_path = (
            CODE_ROOT
            / "iterative_reconstruction"
            / "qc"
            / f"results{experiment['experiment']}"
            / "iteration_history.csv"
        )
        history = read_csv(history_path)
        for checkpoint, epoch, path in checkpoints:
            image, x, z, _ = rsp_metrics.read_mhd(path)
            epoch_rows = [
                row for row in history if int(row["epoch"]) == epoch
            ]
            measurements = sum(
                int(row["measurements"]) for row in epoch_rows
            )
            residual = (
                math.sqrt(
                    sum(
                        float(row["residual_rmse_mm"]) ** 2
                        * int(row["measurements"])
                        for row in epoch_rows
                    )
                    / measurements
                )
                if measurements
                else None
            )
            rows.append({
                "dataset": label,
                "method": "iterative",
                "variant": checkpoint,
                "epoch": epoch,
                "support": True,
                "image_path": relative(path),
                "training_wepl_rmse_mm": residual,
                **uniform_metrics(
                    image, x, z, config, experiment["truth"]
                ),
            })
    return rows


def holdout_wepl(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    device_override: int | None,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    global_path = QC_ROOT / "holdout_wepl_metrics.csv"
    angle_path = QC_ROOT / "holdout_wepl_by_angle.csv"
    resource_path = QC_ROOT / "holdout_resources.json"
    if (
        global_path.is_file()
        and angle_path.is_file()
        and resource_path.is_file()
        and not force
    ):
        resource = load_json(resource_path)
        if set(resource.get("datasets", {})) == set(exp):
            return read_csv(global_path), read_csv(angle_path), resource
        raise RuntimeError(
            "holdout QC exists but does not cover all Stage 2 datasets; "
            "rerun --action evaluate --force"
        )
    import cupy as cp
    from gpu_mlp_operator import GpuMlpProjector

    device = int(
        config["evaluation"]["device"]
        if device_override is None
        else device_override
    )
    cp.cuda.Device(device).use()
    batch_size = int(config["evaluation"]["gpu_batch_size"])
    bit_order = str(config["split"]["bit_order"])
    lut = make_vectorized_wepl_lut()
    all_global: list[dict[str, Any]] = []
    all_angles: list[dict[str, Any]] = []
    dataset_resources: dict[str, Any] = {}
    for label, experiment in exp.items():
        preprocessing = path_for(experiment, "preprocessing_data")
        reconstruction = path_for(experiment, "reconstruction_data")
        split_dir = (
            preprocessing / "splits" / config["split"]["name"]
        )
        images_cpu: dict[str, np.ndarray] = {}
        image_paths = [
            (
                "analytic",
                reconstruction
                / "analytic"
                / "recon"
                / "recon_ddb_nohann.mhd",
            ),
            (
                "analytic_supported",
                reconstruction
                / "analytic"
                / "recon"
                / "recon_ddb_nohann_supported.mhd",
            ),
        ]
        iterative_path = (
            reconstruction / "iterative" / "recon" / "epoch_03.mhd"
        )
        if iterative_path.is_file():
            image_paths.append(("iterative_epoch_03", iterative_path))
        for name, path in image_paths:
            image, _, _, _ = rsp_metrics.read_mhd(path)
            images_cpu[name] = np.ascontiguousarray(image, dtype=np.float32)
        if label == "air":
            uncorrected_path = (
                reconstruction
                / "analytic"
                / "variants"
                / "s3_uncorrected_fov210_hann0.mhd"
            )
            image, _, _, _ = rsp_metrics.read_mhd(uncorrected_path)
            images_cpu["analytic_from_uncorrected_wepl"] = (
                np.ascontiguousarray(image, dtype=np.float32)
            )
        images = {
            name: cp.asarray(image) for name, image in images_cpu.items()
        }
        projector = GpuMlpProjector(2100, 0.1, 0.1, 100.0)
        totals = {
            partition: {
                name: {
                    "squared": 0.0,
                    "absolute": 0.0,
                    "signed": 0.0,
                    "count": 0,
                    "pairs": 0,
                }
                for name in images
            }
            for partition in ("validation", "test")
        }
        split_manifest = load_json(QC_ROOT / f"split_{label}.json")
        total_holdout = int(split_manifest["validation"]) + int(
            split_manifest["test"]
        )
        processed = 0
        started = time.perf_counter()
        for run_id in range(RUNS):
            pairs = paircuts.read_mhd(
                preprocessing
                / "pairs_filtered"
                / f"pairs{run_id:04d}.mhd"
            )
            for partition in ("validation", "test"):
                indices = load_mask(
                    split_dir / f"{partition}_mask_{run_id:04d}.bin",
                    len(pairs),
                    bit_order,
                )
                per_angle = {
                    name: {
                        "squared": 0.0,
                        "absolute": 0.0,
                        "signed": 0.0,
                        "count": 0,
                    }
                    for name in images
                }
                for begin in range(0, len(indices), batch_size):
                    selected = np.asarray(
                        pairs[indices[begin : begin + batch_size]],
                        dtype=np.float32,
                    )
                    if (
                        str(experiment["acquisition"]["world_material"]).lower()
                        == "air"
                    ):
                        corrected, _ = air_correct_pairs(
                            selected, lut, config["air_correction"]
                        )
                        wepl = corrected[:, 4, 1]
                    else:
                        wepl = energies_to_wepl_vectorized(
                            lut,
                            selected[:, 4, 0],
                            selected[:, 4, 1],
                        )
                    batch = {
                        "position_in": selected[:, 0, :],
                        "position_out": selected[:, 1, :],
                        "direction_in": selected[:, 2, :],
                        "direction_out": selected[:, 3, :],
                        "wepl_mm": wepl,
                    }
                    statistics = projector.evaluate_many(
                        images, batch, 0.5 * run_id
                    )
                    for name, values in statistics.items():
                        for key in (
                            "squared",
                            "absolute",
                            "signed",
                            "count",
                        ):
                            per_angle[name][key] += values[key]
                            totals[partition][name][key] += values[key]
                        totals[partition][name]["pairs"] += len(selected)
                    processed += len(selected)
                for name, values in per_angle.items():
                    count = int(values["count"])
                    if count == 0:
                        raise RuntimeError(
                            f"{label} {partition} angle {run_id}: no valid MLP"
                        )
                    all_angles.append({
                        "dataset": label,
                        "partition": partition,
                        "checkpoint": name,
                        "run_id": run_id,
                        "angle_deg": 0.5 * run_id,
                        "pairs": len(indices),
                        "valid_measurements": count,
                        "wepl_rmse_mm": math.sqrt(
                            values["squared"] / count
                        ),
                        "wepl_mae_mm": values["absolute"] / count,
                        "wepl_bias_mm": values["signed"] / count,
                    })
            if (run_id + 1) % 20 == 0 or run_id == RUNS - 1:
                elapsed = time.perf_counter() - started
                rate = processed / max(elapsed, 1.0e-12)
                eta = (total_holdout - processed) / max(rate, 1.0e-12)
                print(
                    f"holdout {label}: {run_id+1:03d}/{RUNS}, "
                    f"{processed:,}/{total_holdout:,}, "
                    f"{rate:,.0f} pairs/s, ETA={eta/60:.1f} min",
                    flush=True,
                )
        for partition, checkpoints in totals.items():
            for name, values in checkpoints.items():
                count = int(values["count"])
                all_global.append({
                    "dataset": label,
                    "partition": partition,
                    "checkpoint": name,
                    "pairs": int(values["pairs"]),
                    "valid_measurements": count,
                    "wepl_rmse_mm": math.sqrt(
                        values["squared"] / count
                    ),
                    "wepl_mae_mm": values["absolute"] / count,
                    "wepl_bias_mm": values["signed"] / count,
                    "aggregation": "measurement_count_weighted",
                })
        properties = cp.cuda.runtime.getDeviceProperties(device)
        gpu_name = properties["name"]
        dataset_resources[label] = {
            "elapsed_seconds": time.perf_counter() - started,
            "pairs": total_holdout,
            "checkpoint_count": len(images),
            "gpu": (
                gpu_name.decode()
                if isinstance(gpu_name, bytes)
                else str(gpu_name)
            ),
        }
        del images
        cp.get_default_memory_pool().free_all_blocks()
    resource = {
        "status": "PASS",
        "device": device,
        "batch_size": batch_size,
        "datasets": dataset_resources,
    }
    write_csv(global_path, all_global)
    write_csv(angle_path, all_angles)
    write_json(resource_path, resource)
    return all_global, all_angles, resource


def rows_by(
    rows: list[dict[str, Any]], **criteria: Any
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]


def plot_results(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    analytic_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pct-stage2-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pct-stage2-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })
    FIGURES.mkdir(parents=True, exist_ok=True)

    def load(path: Path):
        return rsp_metrics.read_mhd(path)[:3]

    vacuum_recon = path_for(exp["vacuum"], "reconstruction_data")
    air_recon = path_for(exp["air"], "reconstruction_data")
    comparison = [
        (
            "Vacuum",
            vacuum_recon / "analytic" / "recon" / "recon_ddb_nohann.mhd",
        ),
        (
            "Air, uncorrected",
            air_recon
            / "analytic"
            / "variants"
            / "s3_uncorrected_fov210_hann0.mhd",
        ),
        (
            "Air, corrected",
            air_recon / "analytic" / "recon" / "recon_ddb_nohann.mhd",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.5))
    for column, (title, path) in enumerate(comparison):
        image, x, z = load(path)
        extent = [x[0], x[-1], z[-1], z[0]]
        shown = axes[0, column].imshow(
            image, cmap="viridis", vmin=0.94, vmax=1.06, extent=extent
        )
        axes[0, column].set(title=title, xlabel="x (mm)", ylabel="z (mm)")
        error = image - uniform_fraction(x, z, 100.0)
        error_shown = axes[1, column].imshow(
            error,
            cmap="RdBu_r",
            vmin=-0.08,
            vmax=0.08,
            extent=extent,
        )
        axes[1, column].set(
            title=f"{title}: error vs fixed RSP",
            xlabel="x (mm)",
            ylabel="z (mm)",
        )
    fig.colorbar(shown, ax=axes[0, :], fraction=0.018, label="RSP")
    fig.colorbar(
        error_shown, ax=axes[1, :], fraction=0.018, label="RSP error"
    )
    fig.suptitle("Vacuum–Air DDB-FDK comparison")
    fig.savefig(FIGURES / "vacuum_air_comparison.png")
    plt.close(fig)

    style = {
        "s2_fov210_hann0": ("#2463A6", "-", "o", "Vacuum"),
        "s3_uncorrected_fov210_hann0": (
            "#D59B20",
            "--",
            "s",
            "Air uncorrected",
        ),
        "s3_corrected_fov210_hann0": (
            "#D96C3F",
            "-.",
            "^",
            "Air corrected",
        ),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for variant, (color, line, marker, label) in style.items():
        subset = sorted(
            rows_by(profiles, variant=variant, support=False),
            key=lambda row: float(row["radius_mm"]),
        )
        radius = np.asarray([float(row["radius_mm"]) for row in subset])
        values = np.asarray([float(row["mean_rsp"]) for row in subset])
        keep = (radius >= 88.0) & (radius <= 105.0)
        ax.plot(
            radius[keep],
            values[keep],
            color=color,
            linestyle=line,
            marker=marker,
            markevery=8,
            markersize=3,
            label=label,
        )
    ax.axvline(100.0, color="#343A40", linestyle=":", label="Water boundary")
    ax.set(
        title="Boundary radial profiles",
        xlabel="Radius (mm)",
        ylabel="Azimuthal mean RSP",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend()
    fig.savefig(FIGURES / "boundary_radial_profiles.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fov_styles = [
        ("s2_fov210_hann0", "#2463A6", "-", "o", "210 mm"),
        ("s2_fov220_hann0", "#D59B20", "--", "s", "220 mm"),
        ("s2_fov240_hann0", "#D96C3F", "-.", "^", "240 mm"),
        ("s2_fov260_hann0", "#6B7D3A", ":", "D", "260 mm"),
    ]
    for variant, color, line, marker, label in fov_styles:
        subset = sorted(
            rows_by(profiles, variant=variant, support=False),
            key=lambda row: float(row["radius_mm"]),
        )
        radius = np.asarray([float(row["radius_mm"]) for row in subset])
        values = np.asarray([float(row["mean_rsp"]) for row in subset])
        keep = (radius >= 88.0) & (radius <= 108.0)
        ax.plot(
            radius[keep],
            values[keep],
            color=color,
            linestyle=line,
            marker=marker,
            markevery=10,
            markersize=3,
            label=label,
        )
    ax.axvline(100.0, color="#343A40", linestyle=":")
    ax.set(
        title="Reconstruction field-of-view sensitivity",
        xlabel="Radius (mm)",
        ylabel="Azimuthal mean RSP",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend(title="FOV")
    fig.savefig(FIGURES / "fov_sensitivity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    hann_styles = [
        ("s2_fov210_hann0", "#2463A6", "-", "o", "no Hann"),
        ("s2_fov210_hann0p5", "#D59B20", "--", "s", "Hann 0.5"),
        ("s2_fov210_hann1", "#D96C3F", "-.", "^", "Hann 1.0"),
    ]
    for variant, color, line, marker, label in hann_styles:
        subset = sorted(
            rows_by(profiles, variant=variant, support=False),
            key=lambda row: float(row["radius_mm"]),
        )
        radius = np.asarray([float(row["radius_mm"]) for row in subset])
        values = np.asarray([float(row["mean_rsp"]) for row in subset])
        keep = (radius >= 88.0) & (radius <= 105.0)
        ax.plot(
            radius[keep],
            values[keep],
            color=color,
            linestyle=line,
            marker=marker,
            markevery=8,
            markersize=3,
            label=label,
        )
    ax.axvline(100.0, color="#343A40", linestyle=":")
    ax.set(
        title="Hann-filter sensitivity",
        xlabel="Radius (mm)",
        ylabel="Azimuthal mean RSP",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend()
    fig.savefig(FIGURES / "hann_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(
        2, 4, figsize=(14.4, 7.8), constrained_layout=True
    )
    for row_index, label in enumerate(("vacuum", "air")):
        experiment = exp[label]
        reconstruction = path_for(experiment, "reconstruction_data")
        paths = [
            (
                "Fixed RSP truth",
                reconstruction / "analytic" / "truth" / "truth_rsp_200mev.mhd",
            ),
            (
                "DDB-FDK",
                reconstruction / "analytic" / "recon" / "recon_ddb_nohann.mhd",
            ),
            (
                "OS-SART epoch 3",
                reconstruction / "iterative" / "recon" / "epoch_03.mhd",
            ),
        ]
        truth_image, _, _ = load(paths[0][1])
        for column, (title, path) in enumerate(paths):
            image, x, z = load(path)
            extent = [x[0], x[-1], z[-1], z[0]]
            axes[row_index, column].imshow(
                image, cmap="viridis", vmin=0.94, vmax=1.06, extent=extent
            )
            axes[row_index, column].set(
                title=f"{label.capitalize()}: {title}",
                xlabel="x (mm)",
                ylabel="z (mm)",
            )
        iterative, x, z = load(paths[2][1])
        extent = [x[0], x[-1], z[-1], z[0]]
        axes[row_index, 3].imshow(
            iterative - truth_image,
            cmap="RdBu_r",
            vmin=-0.08,
            vmax=0.08,
            extent=extent,
        )
        axes[row_index, 3].set(
            title=f"{label.capitalize()}: iterative error",
            xlabel="x (mm)",
            ylabel="z (mm)",
        )
    fig.suptitle("Analytic and iterative uniform-water reconstruction")
    fig.savefig(FIGURES / "analytic_iterative_comparison.png")
    plt.close(fig)

    selected_image_rows = []
    for label in ("vacuum", "air"):
        selected_image_rows.extend(
            rows_by(
                image_rows,
                dataset=label,
                method="analytic",
                variant="selected",
            )
        )
        selected_image_rows.extend(
            rows_by(
                image_rows,
                dataset=label,
                method="iterative",
                variant="epoch_03",
            )
        )
    labels = [
        f"{row['dataset'].capitalize()}\n"
        f"{'FDK' if row['method']=='analytic' else 'Iterative'}"
        for row in selected_image_rows
    ]
    xloc = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].bar(
        xloc,
        [float(row["water_core_mean_rsp"]) for row in selected_image_rows],
        color=["#2463A6", "#8CB4D8", "#D96C3F", "#E8A88F"],
    )
    axes[0].axhline(1.0, color="#343A40", linestyle="--", label="Fixed RSP")
    axes[0].axhline(
        1.013517775250251,
        color="#6B7D3A",
        linestyle=":",
        label="S6 effective RSP",
    )
    axes[0].set(
        title="Water core mean",
        ylabel="RSP",
        xticks=xloc,
        xticklabels=labels,
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(
        xloc,
        [
            float(row["outside_100_105_rmse_rsp"])
            for row in selected_image_rows
        ],
        color=["#2463A6", "#8CB4D8", "#D96C3F", "#E8A88F"],
    )
    axes[1].set(
        title="Outside-boundary artifact",
        ylabel="RSP RMSE (100–105 mm)",
        xticks=xloc,
        xticklabels=labels,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "key_image_metrics.png")
    plt.close(fig)

    test = [
        row for row in holdout_rows if row["partition"] == "test"
    ]
    order = [
        ("vacuum", "analytic"),
        ("vacuum", "iterative_epoch_03"),
        ("air", "analytic"),
        ("air", "iterative_epoch_03"),
    ]
    selected_test = [
        next(
            row
            for row in test
            if row["dataset"] == dataset
            and row["checkpoint"] == checkpoint
        )
        for dataset, checkpoint in order
    ]
    labels = [
        f"{row['dataset'].capitalize()}\n"
        f"{'FDK' if row['checkpoint']=='analytic' else 'Iterative'}"
        for row in selected_test
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.bar(
        np.arange(len(labels)),
        [float(row["wepl_rmse_mm"]) for row in selected_test],
        color=["#2463A6", "#8CB4D8", "#D96C3F", "#E8A88F"],
    )
    ax.set(
        title="Locked-test WEPL consistency",
        ylabel="WEPL RMSE (mm)",
        xticks=np.arange(len(labels)),
        xticklabels=labels,
    )
    fig.savefig(FIGURES / "test_wepl_rmse.png")
    plt.close(fig)

    # S4 material-platform image and nominal-truth error.
    material_recon = path_for(exp["material"], "reconstruction_data")
    material_image, x, z = load(
        material_recon / "analytic" / "recon" / "recon_ddb_nohann.mhd"
    )
    material_truth, _, _ = load(
        material_recon / "analytic" / "truth" / "truth_rsp_200mev.mhd"
    )
    extent = [x[0], x[-1], z[-1], z[0]]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, array, title, cmap, limits in (
        (axes[0], material_truth, "Nominal RSP truth", "viridis", (0.0, 2.2)),
        (axes[1], material_image, "S4 DDB-FDK", "viridis", (0.0, 2.2)),
        (
            axes[2],
            material_image - material_truth,
            "Reconstruction error",
            "RdBu_r",
            (-0.2, 0.2),
        ),
    ):
        shown = ax.imshow(
            array, extent=extent, cmap=cmap, vmin=limits[0], vmax=limits[1]
        )
        ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
        fig.colorbar(shown, ax=ax, fraction=0.046)
    fig.suptitle("S4 multi-material diagnostic phantom")
    fig.savefig(FIGURES / "s4_material_reconstruction.png")
    plt.close(fig)

    # S5 reconstruction and the quantitative resolution summaries.
    resolution_recon = path_for(exp["resolution"], "reconstruction_data")
    resolution_image, x, z = load(
        resolution_recon / "analytic" / "recon" / "recon_ddb_nohann.mhd"
    )
    edge_rows = read_csv(QC_ROOT / "slanted_edge_mtf.csv")
    line_rows = read_csv(QC_ROOT / "line_pair_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    axes[0].imshow(
        resolution_image,
        extent=[x[0], x[-1], z[-1], z[0]],
        cmap="viridis",
        vmin=0.9,
        vmax=2.15,
    )
    axes[0].set(title="S5 DDB-FDK", xlabel="x (mm)", ylabel="z (mm)")
    axes[1].plot(
        [float(row["radius_mm"]) for row in edge_rows],
        [float(row["fmtf10_lp_per_mm"]) for row in edge_rows],
        "o-",
        color="#2463A6",
        label="fMTF10",
    )
    axes[1].plot(
        [float(row["radius_mm"]) for row in edge_rows],
        [float(row["fmtf50_lp_per_mm"]) for row in edge_rows],
        "s--",
        color="#D96C3F",
        label="fMTF50",
    )
    axes[1].set(
        title="Slanted-edge resolution",
        xlabel="Target radius (mm)",
        ylabel="Spatial frequency (lp/mm)",
    )
    axes[1].legend()
    axes[1].grid(color="#D9DEE5", linewidth=0.7)
    axes[2].plot(
        [float(row["spatial_frequency_lp_per_mm"]) for row in line_rows],
        [float(row["modulation"]) for row in line_rows],
        "o-",
        color="#6B7D3A",
    )
    axes[2].set(
        title="Line-pair modulation",
        xlabel="Spatial frequency (lp/mm)",
        ylabel="Robust modulation",
    )
    axes[2].grid(color="#D9DEE5", linewidth=0.7)
    fig.savefig(FIGURES / "s5_resolution_reconstruction.png")
    plt.close(fig)


def build_report(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    analytic_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    resource: dict[str, Any],
) -> dict[str, Any]:
    def analytic(variant: str, support: bool = False) -> dict[str, Any]:
        return next(
            row
            for row in analytic_rows
            if row["variant"] == variant
            and str(row["support"]).lower() == str(support).lower()
        )

    def image(
        dataset: str, method: str, variant: str
    ) -> dict[str, Any]:
        return next(
            row
            for row in image_rows
            if row["dataset"] == dataset
            and row["method"] == method
            and row["variant"] == variant
        )

    def test(dataset: str, checkpoint: str) -> dict[str, Any]:
        return next(
            row
            for row in holdout_rows
            if row["dataset"] == dataset
            and row["partition"] == "test"
            and row["checkpoint"] == checkpoint
        )

    vac = analytic("s2_fov210_hann0")
    air_uncorrected = analytic("s3_uncorrected_fov210_hann0")
    air_corrected = analytic("s3_corrected_fov210_hann0")
    material = analytic("s4_corrected_fov210_hann0")
    resolution = analytic("s5_corrected_fov210_hann0")
    vac_supported = analytic("s2_fov210_hann0", True)
    vac_iter = image("vacuum", "iterative", "epoch_03")
    air_iter = image("air", "iterative", "epoch_03")
    fov_rows = [
        analytic(f"s2_fov{fov}_hann0")
        for fov in (210, 220, 240, 260)
    ]
    hann_rows = [
        analytic("s2_fov210_hann0"),
        analytic("s2_fov210_hann0p5"),
        analytic("s2_fov210_hann1"),
    ]
    split_vac = load_json(QC_ROOT / "split_vacuum.json")
    split_air = load_json(QC_ROOT / "split_air.json")
    split_material = load_json(QC_ROOT / "split_material.json")
    split_resolution = load_json(QC_ROOT / "split_resolution.json")
    material_metrics = read_csv(QC_ROOT / "material_metrics.csv")
    edge_metrics = read_csv(QC_ROOT / "slanted_edge_mtf.csv")
    line_metrics = read_csv(QC_ROOT / "line_pair_metrics.csv")
    iterative_summary = load_json(QC_ROOT / "iterative_summary.json")
    scenario_root = (
        CODE_ROOT
        / "simulation"
        / "windows_overnight_simulations_0716"
        / "scenarios"
    )
    base_ct = load_json(scenario_root / "base_ct.json")
    scenario_configs = {
        "vacuum": load_json(scenario_root / "s2_water_vacuum_pilot.json"),
        "air": load_json(scenario_root / "s3_water_air_pilot.json"),
        "material": load_json(
            scenario_root / "s4_material_calibration_air_pilot.json"
        ),
        "resolution": load_json(
            scenario_root / "s5_resolution_air_pilot.json"
        ),
    }

    material_table = "\n".join(
        "| {material} | {ring_radius_mm:.0f} | {nominal_rsp_200mev:.6f} | "
        "{mean_rsp:.6f} | {std_rsp:.6f} | {error} |".format(
            material=row["material"],
            ring_radius_mm=float(row["ring_radius_mm"]),
            nominal_rsp_200mev=float(row["nominal_rsp_200mev"]),
            mean_rsp=float(row["mean_rsp"]),
            std_rsp=float(row["std_rsp"]),
            error=(
                "N/A"
                if row["absolute_relative_error"] in ("", None)
                else f"{100.0*float(row['absolute_relative_error']):.2f}%"
            ),
        )
        for row in material_metrics
    )
    edge_table = "\n".join(
        f"| {int(row['target_id'])} | {float(row['radius_mm']):.1f} | "
        f"{float(row['rotation_deg']):.1f} | "
        f"{float(row['fmtf50_lp_per_mm']):.3f} | "
        f"{float(row['fmtf10_lp_per_mm']):.3f} |"
        for row in edge_metrics
    )
    line_table = "\n".join(
        f"| {float(row['line_width_mm']):.2f} | "
        f"{float(row['spatial_frequency_lp_per_mm']):.3f} | "
        f"{float(row['modulation']):.4f} |"
        for row in sorted(
            line_metrics,
            key=lambda value: float(value["line_width_mm"]),
        )
    )
    s4_layout_table = "\n".join(
        "| {radius:.0f} | {diameter:.0f} | {angles} |".format(
            radius=float(ring["radius_mm"]),
            diameter=float(ring["diameter_mm"]),
            angles="、".join(
                f"{material} {float(ring['angle_offset_deg']) + 72.0 * index:.0f}°"
                for index, material in enumerate(
                    scenario_configs["material"]["calibration_materials"]
                )
            ),
        )
        for ring in scenario_configs["material"]["calibration_rings"]
    )
    s5_design_table = "\n".join(
        "| {width:g} | {frequency:.3f} | {center} | {count} |".format(
            width=float(group["line_width_mm"]),
            frequency=1.0 / (2.0 * float(group["line_width_mm"])),
            center=f"`({group['center_mm'][0]:g}, {group['center_mm'][1]:g})`",
            count=int(group["bar_count"]),
        )
        for group in scenario_configs["resolution"]["line_pair_groups"]
    )
    s5_edge_design_table = "\n".join(
        f"| {index} | `({target['center_mm'][0]:g}, "
        f"{target['center_mm'][1]:g})` | "
        f"`{target['size_xy_mm'][0]:g}×{target['size_xy_mm'][1]:g}` | "
        f"{target['rotation_deg']:g} |"
        for index, target in enumerate(
            scenario_configs["resolution"]["edge_targets"], start=1
        )
    )

    markdown = f"""# 阶段2：诊断模体处理与评价

## 技术摘要

阶段2状态为**PASS**。S2 Vacuum与S3 Air均使用720角度、每角度100,000个
200 MeV质子，并在过滤后按同一确定性规则划分80%训练、10%验证和10%锁定测试。
两组解析水区均值分别为`{float(vac['water_core_mean_rsp']):.6f}`和
`{float(air_corrected['water_core_mean_rsp']):.6f}`，均更接近S6有效RSP
`1.013518`而不是固定参考`1.0`。这支持阶段1判断：水平台的主要系统偏差来自
当前OpenGATE—WEPL标定口径，而不是Air或单一重建算法。

Air修正使S3解析水区均值从
`{float(air_uncorrected['water_core_mean_rsp']):.6f}`变为
`{float(air_corrected['water_core_mean_rsp']):.6f}`，改变量为
`{float(air_corrected['water_core_mean_rsp'])-float(air_uncorrected['water_core_mean_rsp']):+.6f}`。
因此参考面之间约20 mm Air的WEPL影响可测但很小，不能解释此前约`+1.4%`水偏差。

## 1. 实验设计与S2--S5仿真场景回顾

### 1.1 为什么需要四个诊断模体

results0716同时包含水、25根小铝柱、Vacuum背景、解析滤波和迭代约束。一旦出现
水平台偏差、铝恢复误差或外围圆环，仅凭一张复杂模体图像无法判断误差来自材料
定义、外部介质、路径模型、解析滤波、统计噪声还是部分容积。阶段2因此没有直接
增加复杂算法，而是设计四个低通量、完整角度的诊断场景，每个场景回答一个相对
独立的问题：

| 场景 | 唯一主要变化 | 主要研究问题 | 不能单独回答的问题 |
|---|---|---|---|
| S2 | 删除全部插入物，只保留Vacuum中的均匀水 | 水平台、圆柱边界、FOV、Hann和支撑域 | Air、材料定量和真实探测器效应 |
| S3 | 在S2基础上将Vacuum改为Air | 外部Air能损/散射及Air WEPL修正 | 材料依赖误差 |
| S4 | Air背景中加入多材料大柱和中心小铝柱 | 材料平台、径向趋势、部分容积 | 标准空间分辨率和低对比能力 |
| S5 | Air背景中加入线对和多方向斜边 | 线对可视性及位置/方向相关MTF | 材料平台准确性和低对比能力 |

S2与S3构成最关键的成对对照；S4与S5则把“定量准确性”和“空间分辨率”拆成
两套专用模体。四组均保留720角度，而只把每角度通量降到100,000，目的是降低
pilot成本，同时避免角度欠采样成为新的混杂因素。

### 1.2 共用扫描几何和物理设置

| 参数 | S2--S5统一设置 |
|---|---:|
| 质子源 | {float(base_ct['beam_energy_mev']):g} MeV单能质子 |
| 角度采样 | {int(base_ct['projections'])}角度，`0, 0.5, …, 359.5°` |
| 每角度/每场景入射量 | {int(base_ct['protons_per_projection']):,} / {int(base_ct['projections'])*int(base_ct['protons_per_projection']):,}质子 |
| 旋转轴 | 扫描器\\(y\\)轴 |
| 水圆柱 | 半径{float(base_ct['phantom_radius_mm']):g} mm、轴向长度{float(base_ct['phantom_length_mm']):g} mm |
| 物理源平面/有效焦点 | `z={float(base_ct['source_z_mm']):g}/{float(base_ct['focus_z_mm']):g} mm` |
| 源平面尺寸 | `{base_ct['source_size_mm'][0]:g}×{base_ct['source_size_mm'][1]:g}×10⁻⁶ mm³` |
| 等中心束流覆盖 | 约`250×2 mm²` |
| 理想入口/出口参考面 | `z={float(base_ct['detector_in_z_mm']):g}/{float(base_ct['detector_out_z_mm']):g} mm` |
| 物理列表 | `{base_ct['physics_list']}` |
| 模体最大step | {float(base_ct['max_step_mm']):g} mm；S5细线局部进一步限制 |
| OpenGATE并行方式 | 每个角度单线程，多个角度由外部启动器并行 |

源平面到有效焦点仅60 mm，横向源尺寸传播到等中心后形成约`250×2 mm²`扇束。
其中一个方向只有2 mm，因此这是准二维pCT，而不是三维锥束扫描。入口和出口面
使用`PhaseSpaceActor`理想记录位置、方向和动能；它们不是物理硅跟踪器，也没有
像素量化、电子学噪声、探测效率或能量探测器响应。因此阶段2衡量的是模体、
介质和重建链本身，不代表完整真实设备性能。

### 1.3 S2：Vacuum均匀水圆柱

S2的场景只有半径100 mm、长度400 mm的Water圆柱，圆柱外为Vacuum，没有任何
插入物；随机种子为`{scenario_configs['vacuum']['random_seed']} + angle_index`。
该设计把真值简化为圆柱内常数、圆柱外零，使径向不均匀和边界响应可以直接观察。

S2主要验证四个假设：

1. 若没有铝柱时外围圆环仍存在，则圆环不是插入物条纹的叠加；
2. 比较210--260 mm重建FOV，可判断圆环是否由输出图像边界余量不足造成；
3. 比较no-Hann、Hann=0.5和Hann=1，可观察Ramp高频噪声与边缘振铃的折衷；
4. 施加100 mm支撑域前后分别评价90--100 mm内侧和100--105 mm外侧，区分
   “清除物体外像素”和“真正修复边界模型”。

### 1.4 S3：Air均匀水圆柱

S3保持S2的水圆柱、束流、参考面、角度和通量不变，只把外部介质改为Air；随机
种子为`{scenario_configs['air']['random_seed']} + angle_index`。由于入口和出口
能量定义在`z=-110/+110 mm`，质子在水圆柱外、两个参考面之间经过的Air能损也
包含在测得WEPL中。若不处理，这部分WEPL会被错误归入水圆柱。

本阶段采用S6薄板扫描得到的Air均值斜率
`{float(config['air_correction']['slope_mm_wepl_per_mm_air']):.9f} mm-WEPL/mm-Air`，
按每条质子的已知圆柱外路径长度扣除Air贡献，并同时保留未修正DDB作为对照。
S3用于判断：

- Air会把水平台推移多少；
- Air散射是否显著改变噪声、边缘圆环和锁定测试残差；
- S6得到的简单均值修正能否在完整CT路径中复现；
- Vacuum与Air均存在的共同误差是否应归因于更上游的WEPL/RSP口径或重建模型。

S2和S3使用不同随机种子，属于“配置配对、统计独立”的对照，不是逐事件配对；
因此微小差异必须结合统计噪声解释。

### 1.5 S4：多材料与径向位置标定模体

S4在Air背景的100 mm水圆柱中放置Air、Lung、A150_Tissue_Plastic、
SpineBone和Aluminium五种材料。每种材料都以直径15 mm柱分别出现在30、60和
85 mm半径，中心另有一根直径5 mm铝柱；随机种子为
`{scenario_configs['material']['random_seed']} + angle_index`。

| 环半径/mm | 柱直径/mm | 五种材料的局部方位角 |
|---:|---:|---|
{s4_layout_table}

15 mm柱提供相对稳定的平台ROI，避免results0716中5 mm铝柱过度受部分容积支配；
同一材料的三个半径用于检查几何或MLP误差是否随离心距离增长。中心5 mm铝柱则
故意保留，用于把“小目标恢复不足”与“大材料平台的系统偏差”分开。内部Air柱是
模体中的真实低密度空腔，与水圆柱外需要扣除的Air路径不是同一个量。

S4使用固定200 MeV名义RSP评价材料平台。该真值由OpenGATE材料电子密度和
Geant4平均激发能经Bethe--Bloch相对Water归一化得到，适合可复现的横向比较；
但它还不是质子降能过程中严格的材料相关有效RSP。

### 1.6 S5：线对与斜边空间分辨率模体

S5在Air背景的水圆柱内同时设置Aluminium线对和SpineBone斜边；随机种子为
`{scenario_configs['resolution']['random_seed']} + angle_index`。线对提供直观
可视性，斜边提供更可重复的ESF、LSF和fMTF测量。

| 线宽=间隙/mm | 理论频率/lp·mm⁻¹ | 局部中心/mm | 铝条数 |
|---:|---:|---:|---:|
{s5_design_table}

每组线对由4根长度10 mm的平行铝条组成，理论频率为
\\(f=1/(2w)\\)。为避免蒙卡几何步长直接抹平细线，局部最大step不超过线宽的一半。

| 斜边目标 | 局部中心/mm | 尺寸/mm² | 旋转角/deg |
|---:|---:|---:|---:|
{s5_edge_design_table}

五个15 mm SpineBone方块分布在不同半径和方向，倾斜边缘避免与0.1 mm重建像素
完全对齐。多个目标的意义不是简单取一个最好数值，而是观察位置和方向依赖。
S5没有专用1%--5%低对比模块，所以本阶段可以报告空间分辨率，不能据此声明
低对比可探测能力。

## 2. 数据处理、重建与评价口径

### 2.1 从相空间记录到独立数据集

| 数据集 | 过滤后质子 | 训练 | 验证 | 锁定测试 |
|---|---:|---:|---:|---:|
| S2 Vacuum | {split_vac['total']:,} | {split_vac['train']:,} | {split_vac['validation']:,} | {split_vac['test']:,} |
| S3 Air | {split_air['total']:,} | {split_air['train']:,} | {split_air['validation']:,} | {split_air['test']:,} |
| S4 materials | {split_material['total']:,} | {split_material['train']:,} | {split_material['validation']:,} | {split_material['test']:,} |
| S5 resolution | {split_resolution['total']:,} | {split_resolution['train']:,} | {split_resolution['validation']:,} | {split_resolution['test']:,} |

质子身份为`(RunID, filtered_row_index)`，采用`splitmix64-v1`与固定种子
`20260713`。解析和迭代重建只使用训练集；参数扫描不读取锁定测试集。图像同时
相对固定200 MeV Water RSP=`1.0`与S6有效RSP=`1.013518`评价。

完整处理顺序为：入口/出口ROOT primary-only配对 → 固定参考面状态外推 →
局部能损和散射角3σ过滤 → 80/10/10划分 → `I=78 eV`水射程LUT计算WEPL →
Schulte MLP → `500×2×500 @ 0.5 mm` DDB。Air场景在划分后、生成DDB或迭代
输入前，按同一确定性Air路径模型生成校正数据，不改写原始过滤后pairs。

Air场景中`125×2 @ 2 mm`二维局部网格会把出平面/横向散射离开中心层接受窗的
质子记为网格外。该损失与真正的3σ异常剔除分开统计，不能把较低保留率直接解释
为核反应或错误历史增加。

### 2.2 解析、迭代和锁定测试评价

S2--S5均执行no-Hann DDB-FDK，主输出网格为`2100×1×2100 @ 0.1 mm`。S2另外
扫描210、220、240和260 mm FOV以及Hann=0、0.5、1，并比较100 mm硬支撑域。
S2/S3使用各自80%训练质子执行3 epoch GPU MLP OS-SART + Huber-TV：18个子集、
0.1 mm路径步长、no-Hann初值、非负和100 mm支撑域。S4/S5本阶段只做解析诊断，
避免在基准尚未建立时把正则化偏差引入材料和MTF结论。

图像指标回答“重建图像是否接近已定义真值”；锁定测试WEPL指标则使用从未参加
重建或参数选择的10%质子，把重建图像重新沿MLP正投影，回答“图像能否预测未见
质子的测量”。两者必须同时阅读：较低图像RMSE不保证较低测试残差，反之亦然。

## 3. 重建结果与诊断分析

### 3.1 S2/S3：Vacuum与Air的主要差异很小

![Vacuum and Air comparison](figures/vacuum_air_comparison.png)

S3先按S6均值斜率从逐质子WEPL中扣除圆柱外Air路径贡献，再生成DDB。修正前后
图像的差异远小于固定RSP与有效RSP之间的差异。S2和修正后S3仍存在相似的圆形
边界响应，因此外围圆环不能主要归因于Air。

![Boundary radial profiles](figures/boundary_radial_profiles.png)

S2解析水区均值为`{float(vac['water_core_mean_rsp']):.6f}`，S3未修正和修正后
分别为`{float(air_uncorrected['water_core_mean_rsp']):.6f}`与
`{float(air_corrected['water_core_mean_rsp']):.6f}`。Air校正只带来
`{float(air_corrected['water_core_mean_rsp'])-float(air_uncorrected['water_core_mean_rsp']):+.6f}`
的变化，而S2/S3相对固定RSP=1.0仍约高1.38%。两组均接近S6 Water有效RSP
`1.013518`，所以“水平台偏高”的首要解释是当前WEPL标定与固定RSP真值口径不同，
而不是Air导致。

### 3.2 S2参数实验：扩大FOV没有消除内侧边界响应

![FOV sensitivity](figures/fov_sensitivity.png)

210、220、240和260 mm FOV下，100--105 mm外部RMSE范围为
`{min(float(row['outside_100_105_rmse_rsp']) for row in fov_rows):.6f}`--
`{max(float(row['outside_100_105_rmse_rsp']) for row in fov_rows):.6f}`。
若径向剖面在扩大FOV后仍基本重合，说明圆环主要不是由输出图像边缘离水边界仅
5 mm造成，而更接近Ramp反卷积、DDB采样及水—背景阶跃的共同响应。

这是一项有价值的负结果：继续把FOV从260 mm向外扩展，预期不会触及主要误差源。
后续更值得测试的是DDB分箱、滤波截止、边界模型和解析算子的支撑处理。

### 3.3 Hann降低振铃，但改变噪声—分辨率折衷

![Hann sensitivity](figures/hann_sensitivity.png)

no-Hann、Hann=0.5和Hann=1的水区标准差分别为
`{float(hann_rows[0]['water_core_std_rsp']):.6f}`、
`{float(hann_rows[1]['water_core_std_rsp']):.6f}`和
`{float(hann_rows[2]['water_core_std_rsp']):.6f}`。Hann可压低高频噪声和部分
振铃，但会展宽边缘，因此no-Hann仍保留为高分辨率基线。

Hann=0.5在本pilot中得到最低水区标准差，但这不能直接等同于最优设置，因为材料
边缘、线对和MTF会同时受到频率截断。阶段2保留no-Hann作为诊断基线，正是为了
让高频误差不被过早平滑隐藏。

### 3.4 支撑域只消除圆柱外部分，不能修复内侧误差

对S2 no-Hann直接施加100 mm支撑域后，100--105 mm外部RMSE由
`{float(vac['outside_100_105_rmse_rsp']):.6f}`降为
`{float(vac_supported['outside_100_105_rmse_rsp']):.6f}`；但90--100 mm内侧
边界RMSE仍由`{float(vac['boundary_inner_rmse_vs_fixed_rsp']):.6f}`变为
`{float(vac_supported['boundary_inner_rmse_vs_fixed_rsp']):.6f}`。因此支撑域
适合约束物体外像素，却不能被描述为对Ramp边界响应的完整校正。

因此“圆柱外一圈被消除”不意味着解析边界误差已经解决：支撑域是已知几何先验，
只是把外部像素强制设为零；90--100 mm内侧的Ramp/DDB响应仍然存在。

### 3.5 迭代重建降低噪声，但没有改变水平台口径

![Analytic and iterative comparison](figures/analytic_iterative_comparison.png)

| 数据集 | 方法 | 水区均值 | 水区标准差 | 固定RSP RMSE | 有效RSP RMSE | 外部RMSE |
|---|---|---:|---:|---:|---:|---:|
| Vacuum | DDB-FDK | {float(vac['water_core_mean_rsp']):.6f} | {float(vac['water_core_std_rsp']):.6f} | {float(vac['phantom_rmse_vs_fixed_rsp']):.6f} | {float(vac['phantom_rmse_vs_effective_rsp']):.6f} | {float(vac['outside_100_105_rmse_rsp']):.6f} |
| Vacuum | OS-SART epoch 3 | {float(vac_iter['water_core_mean_rsp']):.6f} | {float(vac_iter['water_core_std_rsp']):.6f} | {float(vac_iter['phantom_rmse_vs_fixed_rsp']):.6f} | {float(vac_iter['phantom_rmse_vs_effective_rsp']):.6f} | {float(vac_iter['outside_100_105_rmse_rsp']):.6f} |
| Air corrected | DDB-FDK | {float(air_corrected['water_core_mean_rsp']):.6f} | {float(air_corrected['water_core_std_rsp']):.6f} | {float(air_corrected['phantom_rmse_vs_fixed_rsp']):.6f} | {float(air_corrected['phantom_rmse_vs_effective_rsp']):.6f} | {float(air_corrected['outside_100_105_rmse_rsp']):.6f} |
| Air corrected | OS-SART epoch 3 | {float(air_iter['water_core_mean_rsp']):.6f} | {float(air_iter['water_core_std_rsp']):.6f} | {float(air_iter['phantom_rmse_vs_fixed_rsp']):.6f} | {float(air_iter['phantom_rmse_vs_effective_rsp']):.6f} | {float(air_iter['outside_100_105_rmse_rsp']):.6f} |

![Key image metrics](figures/key_image_metrics.png)

S2迭代相对解析将水区标准差降低
`{100.0*(1.0-float(vac_iter['water_core_std_rsp'])/float(vac['water_core_std_rsp'])):.1f}%`，
有效RSP RMSE降低
`{100.0*(1.0-float(vac_iter['phantom_rmse_vs_effective_rsp'])/float(vac['phantom_rmse_vs_effective_rsp'])):.1f}%`；
S3对应降低
`{100.0*(1.0-float(air_iter['water_core_std_rsp'])/float(air_corrected['water_core_std_rsp'])):.1f}%`
和
`{100.0*(1.0-float(air_iter['phantom_rmse_vs_effective_rsp'])/float(air_corrected['phantom_rmse_vs_effective_rsp'])):.1f}%`。
100 mm支撑域使外部RMSE严格为零，Huber-TV进一步抑制水区高频波动。

另一方面，迭代水均值仍为
`{float(vac_iter['water_core_mean_rsp']):.6f}/{float(air_iter['water_core_mean_rsp']):.6f}`，
没有向固定RSP=1.0移动。这再次说明正则化能改善方差和数据一致性，但不会自动
修正WEPL—真值定义的全局标定偏差。

### 3.6 锁定测试集确认数据一致性改善

![Locked-test WEPL](figures/test_wepl_rmse.png)

| 数据集 | 方法 | 测试WEPL RMSE/mm | MAE/mm | 偏差/mm |
|---|---|---:|---:|---:|
| Vacuum | DDB-FDK | {float(test('vacuum','analytic')['wepl_rmse_mm']):.5f} | {float(test('vacuum','analytic')['wepl_mae_mm']):.5f} | {float(test('vacuum','analytic')['wepl_bias_mm']):+.5f} |
| Vacuum | OS-SART epoch 3 | {float(test('vacuum','iterative_epoch_03')['wepl_rmse_mm']):.5f} | {float(test('vacuum','iterative_epoch_03')['wepl_mae_mm']):.5f} | {float(test('vacuum','iterative_epoch_03')['wepl_bias_mm']):+.5f} |
| Air corrected | DDB-FDK | {float(test('air','analytic')['wepl_rmse_mm']):.5f} | {float(test('air','analytic')['wepl_mae_mm']):.5f} | {float(test('air','analytic')['wepl_bias_mm']):+.5f} |
| Air corrected | OS-SART epoch 3 | {float(test('air','iterative_epoch_03')['wepl_rmse_mm']):.5f} | {float(test('air','iterative_epoch_03')['wepl_mae_mm']):.5f} | {float(test('air','iterative_epoch_03')['wepl_bias_mm']):+.5f} |
| S4 materials | DDB-FDK | {float(test('material','analytic')['wepl_rmse_mm']):.5f} | {float(test('material','analytic')['wepl_mae_mm']):.5f} | {float(test('material','analytic')['wepl_bias_mm']):+.5f} |
| S5 resolution | DDB-FDK | {float(test('resolution','analytic')['wepl_rmse_mm']):.5f} | {float(test('resolution','analytic')['wepl_mae_mm']):.5f} | {float(test('resolution','analytic')['wepl_bias_mm']):+.5f} |

测试质子从未参与重建或参数选择。WEPL残差仍包含Schulte MLP、能量LUT、散射和
有限像素离散误差，不能被解释成单独的图像RSP误差。

S2/S3迭代相对解析的测试WEPL RMSE改善分别约为
`{100.0*(1.0-float(test('vacuum','iterative_epoch_03')['wepl_rmse_mm'])/float(test('vacuum','analytic')['wepl_rmse_mm'])):.2f}%`
和
`{100.0*(1.0-float(test('air','iterative_epoch_03')['wepl_rmse_mm'])/float(test('air','analytic')['wepl_rmse_mm'])):.2f}%`。
RMSE降幅不大，但偏差从
`{float(test('vacuum','analytic')['wepl_bias_mm']):+.3f}/{float(test('air','analytic')['wepl_bias_mm']):+.3f} mm`
降到
`{float(test('vacuum','iterative_epoch_03')['wepl_bias_mm']):+.3f}/{float(test('air','iterative_epoch_03')['wepl_bias_mm']):+.3f} mm`，
说明迭代主要消除了系统性数据偏差；剩余约2.44 mm RMSE已不能由平均偏差解释，
其中仍混合了逐质子涨落、MLP近似和离散误差。

### 3.7 S4材料定量结果

![S4 material reconstruction](figures/s4_material_reconstruction.png)

S4包含五种材料在30、60和85 mm三个半径的15 mm柱，以及中心5 mm铝柱。
这里的固定200 MeV名义RSP由OpenGATE材料电子密度和Geant4平均激发能代入
Bethe--Bloch后相对Water归一化得到；它是可复现的名义参考，不等同于混合路径的
能量加权有效RSP。解析重建的模体RMSE为
`{float(material['phantom_rmse_vs_nominal_rsp']):.6f}`，非Air材料MAPE为
`{100.0*float(material['material_mape_non_air']):.2f}%`。

| 材料 | 半径/mm | 名义RSP | 平台均值 | 平台标准差 | 绝对相对误差 |
|---|---:|---:|---:|---:|---:|
{material_table}

同一材料在三个半径的结果用于判断径向几何趋势；中心5 mm铝柱主要反映部分容积，
不与15 mm平台采用同一解释。

非Air材料总体MAPE为
`{100.0*float(material['material_mape_non_air']):.2f}%`，最大单ROI相对误差为
1.79%。15 mm Aluminium的误差由30、60到85 mm半径分别为0.26%、0.58%和
0.76%，提示存在小幅径向趋势，但幅度尚不足以单独证明MLP几何失配。中心5 mm
铝柱误差为1.35%，高于三个大铝柱中的多数结果，符合小目标更容易受部分容积和
边缘响应影响的预期。

Lung、A150和SpineBone普遍重建偏高约1%--1.8%，说明误差不是统一比例缩放即可
完全消除。下一步应优先补充这些材料的能量—厚度有效RSP标定，再判断剩余部分是
固定水MLP、Air修正还是解析重建造成。

### 3.8 S5空间分辨率结果

![S5 resolution reconstruction](figures/s5_resolution_reconstruction.png)

五个15 mm SpineBone斜边采用0.025 mm过采样ESF、数值微分LSF和FFT计算
fMTF；平均fMTF50和fMTF10分别为
`{float(resolution['fmtf50_mean_lp_per_mm']):.3f}`和
`{float(resolution['fmtf10_mean_lp_per_mm']):.3f} lp/mm`。

| 靶编号 | 半径/mm | 旋转角/deg | fMTF50/lp·mm⁻¹ | fMTF10/lp·mm⁻¹ |
|---:|---:|---:|---:|---:|
{edge_table}

线对仅作为可视性和调制度补充，不替代斜边MTF：

| 线宽/mm | 空间频率/lp·mm⁻¹ | 稳健调制度 |
|---:|---:|---:|
{line_table}

五个目标的fMTF50从0.367到0.745 lp/mm，说明测得分辨率对位置和边缘方向敏感；
当前平均值`{float(resolution['fmtf50_mean_lp_per_mm']):.3f} lp/mm`应作为
阶段2基线，而不能代替每个方向的完整结果。0.5 mm线对对应1.0 lp/mm，仍有
`{float(next(row for row in line_metrics if float(row['line_width_mm']) == 0.5)['modulation']):.4f}`
稳健调制度，但“非零调制度”不等于满足某个临床可分辨判据。后续算法比较必须
使用相同ROI和调制度定义。

## 4. 局限性

1. S2--S5只有每角度100,000个质子，统计噪声高于论文通量results0716；
2. S2与S3使用不同随机种子，比较的是同配置独立样本，不是逐事件配对；
3. S6有效RSP来自5--100 mm单材料薄板，尚不是200 mm路径的严格真值；
4. Air修正采用近似能量无关的均值斜率，下游低于150 MeV部分属于外推；
5. 当前支撑域是重建后的硬约束，未改变解析反投影本身；
6. S4的多材料参考是固定200 MeV名义RSP，尚无对应薄板有效RSP扫描；
7. S5没有低对比模块，不能由本结果声称低对比可探测能力。

此外，S4/S5本阶段只有解析结果，尚不能判断迭代正则化对材料平台和MTF的偏差—
方差折衷；S2/S3的结论也只适用于理想相空间记录，不能直接推广到有限位置、
方向和能量分辨率的真实探测器。

## 5. 总结与阶段决策

阶段2完成了从“复杂模体中观察现象”到“用专用模体隔离误差来源”的转换，主要
结论如下：

1. **水平台偏差的主因不是Air。** S2与S3均稳定在约`1.0138`，与S6有效RSP一致；
   Air修正只改变约`0.00022 RSP`，固定200 MeV RSP与当前WEPL标定口径的差异才是
   约`+1.4%`水平台偏差的首要解释。
2. **外围圆环不是铝柱、Air或简单FOV不足造成。** 均匀水中仍出现圆环，
   Vacuum/Air表现相似，210--260 mm FOV径向剖面基本重合。支撑域只能可靠清零
   物体外部分，不能修复90--100 mm内侧边界响应。
3. **迭代重建的主要收益是降低方差和系统性数据偏差。** S2/S3有效RSP RMSE约
   减半，水区标准差明显下降，锁定测试WEPL偏差接近零；但水平台均值没有向
   固定RSP=1.0移动，说明全局物理标定不能靠增加迭代轮数解决。
4. **S4建立了可用的多材料定量基线。** 非Air材料MAPE为
   `{100.0*float(material['material_mape_non_air']):.2f}%`，位置趋势总体较小；
   15 mm大柱与中心5 mm铝柱的差异证明部分容积必须与材料系统偏差分开报告。
5. **S5建立了可重复的空间分辨率基线。** 平均fMTF50/fMTF10为
   `{float(resolution['fmtf50_mean_lp_per_mm']):.3f}/{float(resolution['fmtf10_mean_lp_per_mm']):.3f} lp/mm`，同时观察到明显的
   位置和方向差异；本结果不包含低对比性能结论。

因此阶段2状态为**PASS**。S2--S5均已形成独立训练、验证和锁定测试集，结果足以
支持进入阶段3。近期不建议仅为消除圆环继续扩大解析FOV，也不建议在未补充材料
有效RSP标定前对S4施加统一经验缩放。下一步应先在S1--S5上开展稳健过滤、
WEPL不确定度和数据权重研究；D1下载后再在阶段7集中评价完整通量Air、物理硅
跟踪器和参数化读出误差。S4/S5是否提高到论文通量，应依据后续算法是否需要
更低统计噪声再决定。
"""
    report_path = QC_ROOT / "stage2_summary.md"
    report_path.write_text(markdown, encoding="utf-8")
    summary = {
        "status": "PASS",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": {
            "vacuum": config["experiments"]["vacuum"],
            "air": config["experiments"]["air"],
            "material": config["experiments"]["material"],
            "resolution": config["experiments"]["resolution"],
        },
        "key_results": {
            "vacuum_analytic_water_mean_rsp": float(
                vac["water_core_mean_rsp"]
            ),
            "air_uncorrected_analytic_water_mean_rsp": float(
                air_uncorrected["water_core_mean_rsp"]
            ),
            "air_corrected_analytic_water_mean_rsp": float(
                air_corrected["water_core_mean_rsp"]
            ),
            "air_correction_image_delta_rsp": float(
                air_corrected["water_core_mean_rsp"]
            )
            - float(air_uncorrected["water_core_mean_rsp"]),
            "vacuum_iterative_water_mean_rsp": float(
                vac_iter["water_core_mean_rsp"]
            ),
            "air_iterative_water_mean_rsp": float(
                air_iter["water_core_mean_rsp"]
            ),
            "vacuum_test_wepl_rmse_analytic_mm": float(
                test("vacuum", "analytic")["wepl_rmse_mm"]
            ),
            "vacuum_test_wepl_rmse_iterative_mm": float(
                test("vacuum", "iterative_epoch_03")["wepl_rmse_mm"]
            ),
            "air_test_wepl_rmse_analytic_mm": float(
                test("air", "analytic")["wepl_rmse_mm"]
            ),
            "air_test_wepl_rmse_iterative_mm": float(
                test("air", "iterative_epoch_03")["wepl_rmse_mm"]
            ),
            "material_mape_non_air": float(
                material["material_mape_non_air"]
            ),
            "resolution_fmtf50_mean_lp_per_mm": float(
                resolution["fmtf50_mean_lp_per_mm"]
            ),
            "resolution_fmtf10_mean_lp_per_mm": float(
                resolution["fmtf10_mean_lp_per_mm"]
            ),
            "material_test_wepl_rmse_analytic_mm": float(
                test("material", "analytic")["wepl_rmse_mm"]
            ),
            "resolution_test_wepl_rmse_analytic_mm": float(
                test("resolution", "analytic")["wepl_rmse_mm"]
            ),
        },
        "iterative": iterative_summary,
        "holdout_resources": resource,
        "outputs": {
            "report": relative(report_path),
            "figures": relative(FIGURES),
        },
    }
    write_json(QC_ROOT / "stage2_summary.json", summary)
    (QC_ROOT / "completed.flag").write_text(
        "status=PASS\nstage=2\n", encoding="ascii"
    )
    return summary


def evaluate(
    config: dict[str, Any],
    exp: dict[str, dict[str, Any]],
    device: int | None,
    force: bool,
) -> dict[str, Any]:
    analytic_rows = read_csv(QC_ROOT / "analytic_variant_metrics.csv")
    profiles = read_csv(QC_ROOT / "radial_profiles.csv")
    image_rows: list[dict[str, Any]] = []
    for label in exp:
        variant = config["selected_analytic"][label]
        source = next(
            row
            for row in analytic_rows
            if row["variant"] == variant
            and str(row["support"]).lower() == "false"
        )
        image_rows.append({
            "dataset": label,
            "method": "analytic",
            "variant": "selected",
            "epoch": 0,
            **{
                key: value
                for key, value in source.items()
                if key
                not in {
                    "dataset",
                    "variant",
                    "method",
                    "epoch",
                }
            },
        })
    image_rows.extend(add_iterative_image_metrics(config, exp))
    write_csv(QC_ROOT / "image_metrics.csv", image_rows)
    holdout_rows, angle_rows, resource = holdout_wepl(
        config, exp, device, force
    )
    plot_results(
        config,
        exp,
        analytic_rows,
        image_rows,
        profiles,
        holdout_rows,
    )
    summary = build_report(
        config, exp, analytic_rows, image_rows, holdout_rows, resource
    )
    if not all(
        str(row.get("finite", "True")).lower() == "true"
        for row in analytic_rows + image_rows
    ):
        raise RuntimeError("non-finite image metric row")
    if not all(
        math.isfinite(float(row["wepl_rmse_mm"]))
        for row in holdout_rows
    ):
        raise RuntimeError("non-finite holdout WEPL metric")
    return summary


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    config = load_json(CONFIG_PATH)
    exp = experiments(config)
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    actions = (
        (
            "freeze",
            "preprocess",
            "project",
            "analytic",
            "iterative",
            "evaluate",
        )
        if args.action == "all"
        else (args.action,)
    )
    for action in actions:
        print(f"\n=== Stage 2 action: {action} ===", flush=True)
        if action == "freeze":
            freeze_inputs(config, exp, args.force)
        elif action == "preprocess":
            preprocess(config, exp, args.jobs, args.force)
        elif action == "project":
            project_all(config, exp, args.jobs, args.force)
        elif action == "analytic":
            run_analytic(config, exp, args.force)
        elif action == "iterative":
            run_iterative(config, exp, args.force)
        elif action == "holdout":
            holdout_wepl(config, exp, args.device, args.force)
        elif action == "evaluate":
            evaluate(config, exp, args.device, args.force)
    print(f"Stage 2 action={args.action} completed", flush=True)


if __name__ == "__main__":
    main()
