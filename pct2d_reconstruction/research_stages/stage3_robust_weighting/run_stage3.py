#!/usr/bin/env python3
"""Run Stage 3 train-only robust filtering, WEPL weighting, and reconstruction."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
from functools import lru_cache
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
CONFIG_PATH = HERE / "stage3_config.json"
QC_ROOT = HERE / "qc"
STAGE2_PATH = HERE.parent / "stage2_diagnostic_phantoms" / "run_stage2.py"
sys.path[:0] = [
    str(HERE),
    str(CODE_ROOT),
    str(REPOSITORY_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

from common import load_experiment, path_for  # noqa: E402
from preprocessing import paircuts, projection  # noqa: E402
from preprocessing.run_preprocessing import pair_one  # noqa: E402
from analytic_reconstruction import rsp_metrics, truth_maps  # noqa: E402
from iterative_reconstruction.mhd_io import (  # noqa: E402
    read_image_2d,
    write_image_2d,
)
from iterative_reconstruction.physics import (  # noqa: E402
    energies_to_wepl_vectorized,
    make_vectorized_wepl_lut,
)
from iterative_reconstruction.gpu_regularization import (  # noqa: E402
    proximal_regularize,
)
from robust_models import (  # noqa: E402
    FilterModel,
    NoiseModel,
    apply_filter,
    fit_filter,
    fit_noise_model,
    fit_two_component_gmm,
    gmm_clean_posterior,
    normalize_and_clip_weights,
    robust_location_scale,
)
from stage3_io import (  # noqa: E402
    air_correct_pairs,
    effective_sample_size,
    format_duration,
    load_json,
    pair_features,
    partition_masks,
    read_packed_mask,
    relative,
    sha256,
    write_json,
    write_packed_mask,
)


RUNS = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "prepare",
            "filter-screen",
            "weight-screen",
            "confirm",
            "report",
            "smoke",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--datasets",
        default="s2,s3,s4,s5",
        help="comma-separated subset of s1,s2,s3,s4,s5",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help="testing only; formal runs require 720",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    known: set[str] = set()
    for row in rows:
        for key in row:
            if key not in known:
                fields.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


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
            f"$ {shlex.join(command)}\n{result.stdout or ''}",
            encoding="utf-8",
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {shlex.join(command)}\n"
            f"{(result.stdout or '')[-4000:]}"
        )
    return elapsed


def command_path(name: str) -> Path:
    path = REPOSITORY_ROOT / ".venv-gate" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@lru_cache(maxsize=1)
def stage2_module():
    spec = importlib.util.spec_from_file_location("stage2_reference", STAGE2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {STAGE2_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def wepl_lut() -> np.ndarray:
    return make_vectorized_wepl_lut()


def dataset_record(config: dict[str, Any], name: str) -> dict[str, Any]:
    item = dict(config["datasets"][name])
    experiment_id = item.get("experiment_id")
    if experiment_id:
        experiment = load_experiment(str(experiment_id))
        item.update(
            {
                "experiment": experiment,
                "simulation_data": path_for(experiment, "simulation_data"),
                "preprocessing_data": path_for(experiment, "preprocessing_data"),
                "reconstruction_data": path_for(experiment, "reconstruction_data"),
                "acquisition": experiment["acquisition"],
                "truth": experiment["truth"],
            }
        )
    else:
        item.update(
            {
                "experiment": None,
                "simulation_data": REPOSITORY_ROOT / item["simulation_data"],
                "preprocessing_data": REPOSITORY_ROOT / item["preprocessing_data"],
                "reconstruction_data": REPOSITORY_ROOT
                / item["reconstruction_data"],
                "acquisition": {
                    "projections": 720,
                    "first_angle_deg": 0.0,
                    "angle_step_deg": 0.5,
                    "arc_deg": 360.0,
                    "source_to_isocenter_mm": 1000.0,
                    "source_to_detector_mm": 1110.0,
                },
                "truth": {
                    "kind": "aluminium_rods",
                    "phantom_radius_mm": 100.0,
                },
            }
        )
    item["name"] = name
    return item


def parse_datasets(text: str, config: dict[str, Any]) -> list[str]:
    names = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = sorted(set(names) - set(config["datasets"]))
    if unknown:
        raise SystemExit(f"unknown datasets: {', '.join(unknown)}")
    if len(names) != len(set(names)):
        raise SystemExit("--datasets contains duplicates")
    return names


def stage_root(dataset: dict[str, Any]) -> Path:
    return Path(dataset["preprocessing_data"]) / "stage3"


def reconstruction_root(dataset: dict[str, Any]) -> Path:
    return Path(dataset["reconstruction_data"]) / "stage3"


def split_dir(dataset: dict[str, Any], config: dict[str, Any]) -> Path:
    return stage_root(dataset) / "splits" / config["split"]["name"]


def mask_path(
    dataset: dict[str, Any], candidate: str, run_id: int
) -> Path:
    return (
        stage_root(dataset)
        / "filters"
        / candidate
        / f"accepted_mask_{run_id:04d}.bin"
    )


def distance_path(
    dataset: dict[str, Any], candidate: str, run_id: int
) -> Path:
    return (
        stage_root(dataset)
        / "filters"
        / candidate
        / f"distance_{run_id:04d}.npy"
    )


def model_path(dataset: dict[str, Any], candidate: str, run_id: int) -> Path:
    return (
        stage_root(dataset)
        / "filters"
        / candidate
        / "models"
        / f"model_{run_id:04d}.npz"
    )


def ensure_pairs(
    dataset: dict[str, Any], jobs: int, runs: int, force: bool
) -> Path:
    pairs = Path(dataset["preprocessing_data"]) / "pairs"
    existing = [
        pairs / f"pairs{run_id:04d}.mhd" for run_id in range(runs)
    ]
    if all(path.is_file() for path in existing):
        return pairs
    if dataset["name"] != "s1":
        missing = [path for path in existing if not path.is_file()]
        raise FileNotFoundError(
            f"{dataset['name']}: missing {len(missing)} primary pair files; "
            f"first: {missing[0]}"
        )
    pairs.mkdir(parents=True, exist_ok=True)
    simulation = Path(dataset["simulation_data"])
    futures = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        for run_id in range(runs):
            output = pairs / f"pairs{run_id:04d}.mhd"
            if output.is_file() and not force:
                continue
            futures[
                executor.submit(
                    pair_one, run_id, str(simulation), str(pairs)
                )
            ] = run_id
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            completed += 1
            print(
                f"S1 pairing {completed:03d}/{len(futures):03d}: "
                f"angle={row['run_id']:03d}, pairs={row['pairs']:,}",
                flush=True,
            )
    if not all(path.is_file() for path in existing):
        raise RuntimeError("S1 primary pairing is incomplete")
    return pairs


def prepare_dataset(
    dataset: dict[str, Any],
    config: dict[str, Any],
    jobs: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    pairs_dir = ensure_pairs(dataset, jobs, runs, force)
    output = split_dir(dataset, config)
    manifest_path = QC_ROOT / f"prepare_{dataset['name']}.json"
    if manifest_path.is_file() and not force:
        existing = load_json(manifest_path)
        if (
            existing.get("status") == "PASS"
            and int(existing.get("runs", 0)) == runs
        ):
            print(f"prepare {dataset['name']}: reusing complete split", flush=True)
            return existing
    if force:
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    bit_order = str(config["split"]["bit_order"])
    rows = []
    totals = {"total": 0, "train": 0, "validation": 0, "test": 0}
    started = time.perf_counter()
    for run_id in range(runs):
        path = pairs_dir / f"pairs{run_id:04d}.mhd"
        pairs = paircuts.read_mhd(path)
        masks = partition_masks(len(pairs), run_id, config["split"])
        if not all(np.any(mask) for mask in masks.values()):
            raise RuntimeError(f"{dataset['name']} run {run_id}: empty partition")
        for partition in ("validation", "test"):
            write_packed_mask(
                output / f"{partition}_mask_{run_id:04d}.bin",
                masks[partition],
                bit_order,
            )
        row = {
            "dataset": dataset["name"],
            "run_id": run_id,
            "total": len(pairs),
            **{
                partition: int(np.count_nonzero(mask))
                for partition, mask in masks.items()
            },
            "pairs_mhd_sha256": sha256(path),
            "validation_mask_sha256": sha256(
                output / f"validation_mask_{run_id:04d}.bin"
            ),
            "test_mask_sha256": sha256(
                output / f"test_mask_{run_id:04d}.bin"
            ),
        }
        rows.append(row)
        for key in totals:
            totals[key] += int(row[key])
        if (run_id + 1) % 40 == 0 or run_id == runs - 1:
            print(
                f"prepare {dataset['name']}: {run_id+1:03d}/{runs}, "
                f"train={totals['train']:,}, validation={totals['validation']:,}, "
                f"test={totals['test']:,}",
                flush=True,
            )
    write_csv(QC_ROOT / f"prepare_{dataset['name']}_runs.csv", rows)
    result = {
        "status": "PASS",
        "dataset": dataset["name"],
        "runs": runs,
        "identity": config["split"]["identity"],
        "rule": (
            "splitmix64-v1(RunID, paired_row_index, "
            f"{config['split']['seed']}) % 10: "
            "test=0, validation=1, train=2..9"
        ),
        **totals,
        "fractions": {
            key: totals[key] / totals["total"]
            for key in ("train", "validation", "test")
        },
        "elapsed_seconds": time.perf_counter() - started,
        "pairs": relative(pairs_dir),
        "masks": relative(output),
    }
    write_json(manifest_path, result)
    return result


def _filter_one(
    dataset: dict[str, Any],
    config: dict[str, Any],
    run_id: int,
    force: bool,
) -> list[dict[str, Any]]:
    pairs_path = Path(dataset["preprocessing_data"]) / "pairs" / f"pairs{run_id:04d}.mhd"
    pairs = paircuts.read_mhd(pairs_path)
    partitions = partition_masks(len(pairs), run_id, config["split"])
    inside, cells, features = pair_features(pairs, config["filtering"])
    rows = []
    for candidate in config["filtering"]["candidates"]:
        accepted_path = mask_path(dataset, candidate, run_id)
        score_path = distance_path(dataset, candidate, run_id)
        fitted_path = model_path(dataset, candidate, run_id)
        if accepted_path.is_file() and score_path.is_file() and fitted_path.is_file() and not force:
            selected = read_packed_mask(
                accepted_path,
                len(pairs),
                config["split"]["bit_order"],
            )
            distance = np.load(score_path, mmap_mode="r")
        else:
            model = fit_filter(
                candidate,
                features,
                cells,
                inside,
                partitions["train"],
                config["filtering"],
            )
            selected, distance = apply_filter(
                model, features, cells, inside, config["filtering"]
            )
            model.save(fitted_path)
            write_packed_mask(
                accepted_path,
                selected,
                config["split"]["bit_order"],
            )
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(score_path, distance.astype(np.float32))
        row = {
            "dataset": dataset["name"],
            "run_id": run_id,
            "candidate": candidate,
            "total": len(pairs),
            "inside": int(np.count_nonzero(inside)),
        }
        for partition, partition_mask in partitions.items():
            denominator = int(np.count_nonzero(partition_mask))
            numerator = int(np.count_nonzero(partition_mask & selected))
            row[f"{partition}_total"] = denominator
            row[f"{partition}_accepted"] = numerator
            row[f"{partition}_retention"] = numerator / denominator
        finite_distance = np.asarray(distance)[np.isfinite(distance)]
        row["distance_p50"] = float(np.quantile(finite_distance, 0.50))
        row["distance_p99"] = float(np.quantile(finite_distance, 0.99))
        rows.append(row)
    return rows


def fit_filters(
    dataset: dict[str, Any],
    config: dict[str, Any],
    jobs: int,
    runs: int,
    force: bool,
) -> list[dict[str, Any]]:
    prepare_manifest = QC_ROOT / f"prepare_{dataset['name']}.json"
    if not prepare_manifest.is_file():
        raise RuntimeError(
            f"{dataset['name']}: run --action prepare before filter-screen"
        )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(_filter_one, dataset, config, run_id, force): run_id
            for run_id in range(runs)
        }
        for future in as_completed(futures):
            rows.extend(future.result())
            if len(rows) % (20 * len(config["filtering"]["candidates"])) == 0:
                print(
                    f"filter {dataset['name']}: "
                    f"{len(rows)//len(config['filtering']['candidates']):03d}/{runs}",
                    flush=True,
                )
    rows.sort(key=lambda row: (str(row["candidate"]), int(row["run_id"])))
    write_csv(QC_ROOT / f"filter_{dataset['name']}_runs.csv", rows)
    return rows


def gmm_pilot(
    dataset: dict[str, Any],
    config: dict[str, Any],
    runs: int,
) -> list[dict[str, Any]]:
    rows = []
    stride = int(config["filtering"]["gmm_angle_stride"])
    for run_id in range(0, runs, stride):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        partitions = partition_masks(len(pairs), run_id, config["split"])
        inside, cells, features = pair_features(pairs, config["filtering"])
        model = FilterModel.load(
            model_path(dataset, "robust_mahalanobis", run_id)
        )
        standardized = np.zeros_like(features)
        standardized[inside] = (
            features[inside] - model.arrays["center"][cells[inside]]
        ) / model.arrays["scale"][cells[inside]]
        train = partitions["train"] & inside
        # The pilot is angle-global by design; its purpose is only to decide
        # whether a much more expensive full per-cell GMM deserves promotion.
        fitted = fit_two_component_gmm(standardized[train])
        posterior = gmm_clean_posterior(standardized, fitted)
        selected = inside & (
            posterior >= float(config["filtering"]["gmm_posterior_cut"])
        )
        for partition in ("train", "validation"):
            denominator = int(np.count_nonzero(partitions[partition]))
            rows.append(
                {
                    "dataset": dataset["name"],
                    "run_id": run_id,
                    "partition": partition,
                    "total": denominator,
                    "accepted": int(
                        np.count_nonzero(partitions[partition] & selected)
                    ),
                    "retention": float(
                        np.count_nonzero(partitions[partition] & selected)
                        / denominator
                    ),
                    "clean_mixing_fraction": float(
                        fitted["mixing"][int(fitted["clean_component"])]
                    ),
                    "posterior_p01": float(np.quantile(posterior[inside], 0.01)),
                    "posterior_p50": float(np.quantile(posterior[inside], 0.50)),
                }
            )
    write_csv(QC_ROOT / f"gmm_pilot_{dataset['name']}.csv", rows)
    return rows


def _masked_projection_one(
    dataset: dict[str, Any],
    config: dict[str, Any],
    candidate: str,
    run_id: int,
    force: bool,
) -> dict[str, Any]:
    output_root = stage_root(dataset)
    ddb_name = f"ddb/{candidate}"
    output = output_root / ddb_name / f"proj{run_id:04d}.mhd"
    if output.is_file() and output.with_suffix(".raw").is_file() and not force:
        image = projection.read_mhd(output)
        return {
            "dataset": dataset["name"],
            "candidate": candidate,
            "run_id": run_id,
            "reused": True,
            "finite": bool(np.isfinite(image).all()),
        }
    pairs = paircuts.read_mhd(
        Path(dataset["preprocessing_data"])
        / "pairs"
        / f"pairs{run_id:04d}.mhd"
    )
    partitions = partition_masks(len(pairs), run_id, config["split"])
    accepted = read_packed_mask(
        mask_path(dataset, candidate, run_id),
        len(pairs),
        config["split"]["bit_order"],
    )
    selected = np.asarray(pairs[partitions["train"] & accepted], dtype=np.float32)
    if str(dataset["world_material"]).lower() == "air":
        selected = air_correct_pairs(
            selected,
            wepl_lut(),
            config["air_correction"],
            energies_to_wepl_vectorized,
        )
    with tempfile.TemporaryDirectory(prefix=f"pct-stage3-{run_id:04d}-") as temporary:
        temporary_path = Path(temporary)
        paircuts.write_mhd(
            temporary_path / f"pairs{run_id:04d}.mhd", selected
        )
        result = projection.process_run(
            run_id,
            str(temporary_path),
            str(output_root),
            False,
            ddb_name,
        )
    result.update(
        {
            "dataset": dataset["name"],
            "candidate": candidate,
            "selected_train": len(selected),
            "reused": False,
            "finite": True,
        }
    )
    return result


def generate_candidate_ddb(
    dataset: dict[str, Any],
    config: dict[str, Any],
    candidate: str,
    jobs: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    directory = stage_root(dataset) / "ddb" / candidate
    if force:
        shutil.rmtree(directory, ignore_errors=True)
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _masked_projection_one,
                dataset,
                config,
                candidate,
                run_id,
                force,
            ): run_id
            for run_id in range(runs)
        }
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0 or len(rows) == runs:
                print(
                    f"DDB {dataset['name']}/{candidate}: "
                    f"{len(rows):03d}/{runs}",
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["run_id"]))
    write_csv(
        QC_ROOT / f"ddb_{dataset['name']}_{candidate}_runs.csv", rows
    )
    complete = (
        len(list(directory.glob("proj*.mhd"))) == runs
        and len(list(directory.glob("proj*.raw"))) == runs
        and all(bool(row["finite"]) for row in rows)
    )
    result = {
        "status": "PASS" if complete else "FAIL",
        "dataset": dataset["name"],
        "candidate": candidate,
        "runs": runs,
        "ddb": relative(directory),
        "selected_train": sum(int(row.get("selected_train", 0)) for row in rows),
    }
    write_json(QC_ROOT / f"ddb_{dataset['name']}_{candidate}.json", result)
    if not complete:
        raise RuntimeError(f"incomplete DDB for {dataset['name']}/{candidate}")
    return result


def reconstruct_candidate(
    dataset: dict[str, Any],
    candidate: str,
    runs: int,
    force: bool,
) -> tuple[Path, float]:
    if runs != RUNS:
        raise RuntimeError("DDB-FDK reconstruction requires all 720 projections")
    output = reconstruction_root(dataset) / "analytic" / candidate / "recon.mhd"
    if output.is_file() and output.with_suffix(".raw").is_file() and not force:
        return output, 0.0
    output.parent.mkdir(parents=True, exist_ok=True)
    geometry = QC_ROOT / "geometry" / f"{dataset['name']}.xml"
    geometry.parent.mkdir(parents=True, exist_ok=True)
    acquisition = dataset["acquisition"]
    if not geometry.is_file() or force:
        run_command(
            [
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
            ],
            QC_ROOT / "logs" / f"geometry_{dataset['name']}.log",
        )
    command = [
        str(command_path("pctfdk")),
        "--lowmem",
        "--geometry",
        str(geometry),
        "--path",
        str(stage_root(dataset) / "ddb" / candidate),
        "--regexp",
        r"proj....\.mhd",
        "--output",
        str(output),
        "--dimension",
        "2100",
        "1",
        "2100",
        "--spacing",
        "0.1",
        "1",
        "0.1",
        "--hann",
        "0",
        "--verbose",
    ]
    elapsed = run_command(
        command,
        QC_ROOT / "logs" / f"fdk_{dataset['name']}_{candidate}.log",
    )
    image, _, _, _ = rsp_metrics.read_mhd(output)
    if not np.isfinite(image).all():
        raise RuntimeError(f"non-finite FDK output: {output}")
    return output, elapsed


def image_metrics(
    dataset: dict[str, Any], config: dict[str, Any], image_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image, x, z, _ = rsp_metrics.read_mhd(image_path)
    stage2 = stage2_module()
    stage2_config = load_json(
        HERE.parent / "stage2_diagnostic_phantoms" / "stage2_config.json"
    )
    if dataset["name"] in {"s2", "s3"}:
        measured = stage2.uniform_metrics(
            image,
            x,
            z,
            stage2_config,
            dataset["truth"],
        )
        return measured, []
    if dataset["name"] == "s4":
        measured, material, _, _ = stage2.diagnostic_metrics(
            "material", image, x, z, stage2_config
        )
        return measured, material
    if dataset["name"] == "s5":
        measured, _, edge, line = stage2.diagnostic_metrics(
            "resolution", image, x, z, stage2_config
        )
        return measured, edge + line
    # S1 uses the established 25-rod truth definition.
    definition = (
        CODE_ROOT
        / "simulation"
        / "simulation0716"
        / "truth_geometry_definition.json"
    )
    if not definition.is_file():
        return {
            "finite": bool(np.isfinite(image).all()),
            "image_min": float(image.min()),
            "image_max": float(image.max()),
        }, []
    from analytic_reconstruction.run_analytic_reconstruction import generate_truth

    with tempfile.TemporaryDirectory(prefix="pct-stage3-s1-truth-") as temporary:
        truth_dir = Path(temporary)
        generate_truth(definition, image_path, truth_dir)
        truth_rsp, tx, tz, _ = rsp_metrics.read_mhd(
            truth_dir / "truth_rsp_200mev.mhd"
        )
        truth_red, _, _, _ = rsp_metrics.read_mhd(truth_dir / "truth_red.mhd")
        centers = load_json(definition)["geometry"]["insert_centers_xz_mm"]
        measured, inserts = rsp_metrics.metrics_for(
            image, truth_red, truth_rsp, tx, tz, centers
        )
    return measured, inserts


def make_batch(
    pairs: np.ndarray, dataset: dict[str, Any], config: dict[str, Any]
) -> dict[str, np.ndarray]:
    if str(dataset["world_material"]).lower() == "air":
        pairs = air_correct_pairs(
            pairs,
            wepl_lut(),
            config["air_correction"],
            energies_to_wepl_vectorized,
        )
    wepl = energies_to_wepl_vectorized(
        wepl_lut(), pairs[:, 4, 0], pairs[:, 4, 1]
    )
    return {
        "position_in": pairs[:, 0, :],
        "position_out": pairs[:, 1, :],
        "direction_in": pairs[:, 2, :],
        "direction_out": pairs[:, 3, :],
        "wepl_mm": wepl,
    }


def evaluate_fixed_partition(
    dataset: dict[str, Any],
    config: dict[str, Any],
    image_paths: dict[str, Path],
    partition: str,
    device: int,
    runs: int,
    pool_filter: str = "baseline_3sigma",
    output_tag: str | None = None,
) -> list[dict[str, Any]]:
    if partition not in {"validation", "test"}:
        raise ValueError(partition)
    import cupy as cp
    from weighted_gpu import WeightedGpuMlpProjector

    cp.cuda.Device(device).use()
    images = {
        name: cp.asarray(read_image_2d(path)[0])
        for name, path in image_paths.items()
    }
    projector = WeightedGpuMlpProjector(2100, 0.1, 0.1, 100.0)
    residuals: dict[str, list[np.ndarray]] = {name: [] for name in images}
    by_angle: list[dict[str, Any]] = []
    batch_size = int(config["iterative"]["batch_size"])
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        partitions = partition_masks(len(pairs), run_id, config["split"])
        pool = read_packed_mask(
            mask_path(dataset, pool_filter, run_id),
            len(pairs),
            config["split"]["bit_order"],
        )
        indices = np.flatnonzero(partitions[partition] & pool)
        angle_values = {name: [] for name in images}
        for begin in range(0, len(indices), batch_size):
            selected = np.asarray(
                pairs[indices[begin : begin + batch_size]], dtype=np.float32
            )
            batch = make_batch(selected, dataset, config)
            for name, image in images.items():
                values = projector.residuals(image, batch, 0.5 * run_id)
                residuals[name].append(values)
                angle_values[name].append(values)
        for name in images:
            values = (
                np.concatenate(angle_values[name])
                if angle_values[name]
                else np.empty(0)
            )
            by_angle.append(
                {
                    "dataset": dataset["name"],
                    "partition": partition,
                    "run_id": run_id,
                    "candidate": name,
                    "evaluation_pool_filter": pool_filter,
                    "count": len(values),
                    "rmse_mm": float(np.sqrt(np.mean(values**2))),
                    "mae_mm": float(np.mean(np.abs(values))),
                    "bias_mm": float(np.mean(values)),
                    "abs_p99_mm": float(np.quantile(np.abs(values), 0.99)),
                }
            )
        if (run_id + 1) % 20 == 0 or run_id == runs - 1:
            print(
                f"{partition} forward {dataset['name']}: "
                f"{run_id+1:03d}/{runs}",
                flush=True,
            )
    rows = []
    for name, chunks in residuals.items():
        values = np.concatenate(chunks)
        rows.append(
            {
                "dataset": dataset["name"],
                "partition": partition,
                "candidate": name,
                "evaluation_pool_filter": pool_filter,
                "count": len(values),
                "rmse_mm": float(np.sqrt(np.mean(values**2))),
                "mae_mm": float(np.mean(np.abs(values))),
                "bias_mm": float(np.mean(values)),
                "abs_p95_mm": float(np.quantile(np.abs(values), 0.95)),
                "abs_p99_mm": float(np.quantile(np.abs(values), 0.99)),
            }
        )
    suffix = output_tag or pool_filter
    write_csv(
        QC_ROOT
        / f"{partition}_wepl_{dataset['name']}_{suffix}_by_angle.csv",
        by_angle,
    )
    return rows


def s4_path_labels(
    pairs: np.ndarray, run_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate traversed S4 insert from the entrance--exit chord.

    This classifier is used only for retention-fairness QC. Reconstruction
    continues to use Schulte MLP paths.
    """

    stage2 = stage2_module()
    stage2_config = load_json(
        HERE.parent / "stage2_diagnostic_phantoms" / "stage2_config.json"
    )
    scenario = stage2.scenario_config(stage2_config, "material")
    inserts = stage2.material_centers(scenario)
    angle = math.radians(0.5 * run_id)
    cosine, sine = math.cos(angle), math.sin(angle)

    def transform(position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scanner_x = np.asarray(position[:, 0], dtype=np.float64)
        scanner_z = np.asarray(position[:, 2], dtype=np.float64)
        return (
            cosine * scanner_x - sine * scanner_z,
            -sine * scanner_x - cosine * scanner_z,
        )

    x0, z0 = transform(pairs[:, 0, :])
    x1, z1 = transform(pairs[:, 1, :])
    dx, dz = x1 - x0, z1 - z0
    length_squared = np.maximum(dx * dx + dz * dz, 1.0e-12)
    labels = np.full(len(pairs), "Water", dtype="<U24")
    best = np.full(len(pairs), np.inf)
    for item in inserts:
        cx, cz = float(item["x_mm"]), float(item["z_mm"])
        t = np.clip(((cx - x0) * dx + (cz - z0) * dz) / length_squared, 0.0, 1.0)
        distance = np.hypot(x0 + t * dx - cx, z0 + t * dz - cz)
        normalized = distance / float(item["radius_mm"])
        hit = (normalized <= 1.0) & (normalized < best)
        labels[hit] = str(item["material"])
        best[hit] = normalized[hit]
    # Chord impact parameter in reconstruction coordinates for radial QC.
    t0 = np.clip(-(x0 * dx + z0 * dz) / length_squared, 0.0, 1.0)
    impact = np.hypot(x0 + t0 * dx, z0 + t0 * dz)
    labels[(labels == "Water") & (impact >= 100.0)] = "Background"
    return labels, impact


def material_and_radial_retention(
    dataset: dict[str, Any],
    config: dict[str, Any],
    runs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset["name"] != "s4":
        return [], []
    material_totals: dict[tuple[str, str, str], list[int]] = {}
    radial_totals: dict[tuple[str, str, str], list[int]] = {}
    radial_edges = (0.0, 30.0, 60.0, 90.0, float("inf"))
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        partitions = partition_masks(len(pairs), run_id, config["split"])
        labels, impact = s4_path_labels(pairs, run_id)
        for candidate in config["filtering"]["candidates"]:
            accepted = read_packed_mask(
                mask_path(dataset, candidate, run_id),
                len(pairs),
                config["split"]["bit_order"],
            )
            for partition in ("train", "validation"):
                partition_mask = partitions[partition]
                for material in np.unique(labels):
                    group = partition_mask & (labels == material)
                    key = (candidate, partition, str(material))
                    counts = material_totals.setdefault(key, [0, 0])
                    counts[0] += int(np.count_nonzero(group))
                    counts[1] += int(np.count_nonzero(group & accepted))
                for lower, upper in zip(radial_edges[:-1], radial_edges[1:]):
                    group = partition_mask & (impact >= lower) & (impact < upper)
                    label = f"{lower:g}-{upper:g}"
                    key = (candidate, partition, label)
                    counts = radial_totals.setdefault(key, [0, 0])
                    counts[0] += int(np.count_nonzero(group))
                    counts[1] += int(np.count_nonzero(group & accepted))
        if (run_id + 1) % 40 == 0 or run_id == runs - 1:
            print(f"S4 retention tagging: {run_id+1:03d}/{runs}", flush=True)
    material_rows = [
        {
            "candidate": key[0],
            "partition": key[1],
            "material": key[2],
            "total": counts[0],
            "accepted": counts[1],
            "retention": counts[1] / counts[0],
        }
        for key, counts in sorted(material_totals.items())
        if counts[0]
    ]
    radial_rows = [
        {
            "candidate": key[0],
            "partition": key[1],
            "impact_radius_mm": key[2],
            "total": counts[0],
            "accepted": counts[1],
            "retention": counts[1] / counts[0],
        }
        for key, counts in sorted(radial_totals.items())
        if counts[0]
    ]
    return material_rows, radial_rows


def filter_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    jobs: int,
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    if {item["name"] for item in datasets} != {"s2", "s4"}:
        raise SystemExit("filter-screen must be run with --datasets s2,s4")
    all_filter_rows = []
    image_rows = []
    material_rows = []
    validation_rows = []
    candidate_pool_validation_rows = []
    retention_material_rows = []
    retention_radial_rows = []
    for dataset in datasets:
        all_filter_rows.extend(
            fit_filters(dataset, config, jobs, runs, force)
        )
        material_retention, radial_retention = material_and_radial_retention(
            dataset, config, runs
        )
        retention_material_rows.extend(material_retention)
        retention_radial_rows.extend(radial_retention)
        gmm_pilot(dataset, config, runs)
        paths = {}
        for candidate in config["filtering"]["candidates"]:
            generate_candidate_ddb(
                dataset, config, candidate, jobs, runs, force
            )
            path, elapsed = reconstruct_candidate(
                dataset, candidate, runs, force
            )
            paths[candidate] = path
            metrics, materials = image_metrics(dataset, config, path)
            image_rows.append(
                {
                    "dataset": dataset["name"],
                    "candidate": candidate,
                    "fdk_seconds": elapsed,
                    "image_path": relative(path),
                    **metrics,
                }
            )
            material_rows.extend(
                {
                    "dataset": dataset["name"],
                    "candidate": candidate,
                    **row,
                }
                for row in materials
            )
        validation_rows.extend(
            evaluate_fixed_partition(
                dataset,
                config,
                paths,
                "validation",
                device,
                runs,
                pool_filter="baseline_3sigma",
                output_tag="fixed_baseline_pool",
            )
        )
        for candidate, path in paths.items():
            candidate_pool_validation_rows.extend(
                evaluate_fixed_partition(
                    dataset,
                    config,
                    {candidate: path},
                    "validation",
                    device,
                    runs,
                    pool_filter=candidate,
                    output_tag=f"candidate_pool_{candidate}",
                )
            )
    write_csv(QC_ROOT / "filter_retention_runs.csv", all_filter_rows)
    if retention_material_rows:
        write_csv(
            QC_ROOT / "filter_material_retention.csv",
            retention_material_rows,
        )
    if retention_radial_rows:
        write_csv(
            QC_ROOT / "filter_radial_retention.csv",
            retention_radial_rows,
        )
    write_csv(QC_ROOT / "filter_image_metrics.csv", image_rows)
    if material_rows:
        write_csv(QC_ROOT / "filter_material_metrics.csv", material_rows)
    write_csv(QC_ROOT / "filter_validation_wepl.csv", validation_rows)
    write_csv(
        QC_ROOT / "filter_validation_wepl_candidate_pool.csv",
        candidate_pool_validation_rows,
    )
    winner = select_filter(
        config,
        all_filter_rows,
        image_rows,
        material_rows,
        validation_rows,
        candidate_pool_validation_rows,
    )
    write_json(QC_ROOT / "filter_selection.json", winner)
    return winner


def select_filter(
    config: dict[str, Any],
    retention_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    material_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    candidate_pool_validation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = "baseline_3sigma"
    validation = {
        (row["dataset"], row["candidate"]): row for row in validation_rows
    }
    candidate_validation = {
        (row["dataset"], row["candidate"]): row
        for row in candidate_pool_validation_rows
    }
    images = {(row["dataset"], row["candidate"]): row for row in image_rows}
    materials = {}
    for row in material_rows:
        if (
            row.get("dataset") == "s4"
            and float(row.get("ring_radius_mm", -1)) == 0.0
            and row.get("material") == "Aluminium"
        ):
            materials[row["candidate"]] = row
    base_rmse = float(validation[("s2", baseline)]["rmse_mm"])
    base_p99 = float(candidate_validation[("s2", baseline)]["abs_p99_mm"])
    base_mape = float(images[("s4", baseline)]["material_mape_non_air"])
    base_small = float(materials[baseline]["absolute_relative_error"])
    material_retention_path = QC_ROOT / "filter_material_retention.csv"
    material_retention = (
        read_csv(material_retention_path)
        if material_retention_path.is_file()
        else []
    )
    selection = config["selection"]
    decisions = []
    for candidate in config["filtering"]["candidates"]:
        rmse = float(validation[("s2", candidate)]["rmse_mm"])
        p99 = float(candidate_validation[("s2", candidate)]["abs_p99_mm"])
        mape = float(images[("s4", candidate)]["material_mape_non_air"])
        small = float(materials[candidate]["absolute_relative_error"])
        checks = {
            "validation_rmse": rmse <= base_rmse
            * (1.0 + selection["validation_rmse_max_relative_degradation"]),
            "p99": p99
            <= base_p99
            * (1.0 - selection["absolute_residual_p99_min_improvement"]),
            "material_mape": (mape - base_mape) * 100.0
            <= selection[
                "material_mape_max_degradation_percentage_points"
            ],
            "small_aluminium": (small - base_small) * 100.0
            <= selection[
                "small_aluminium_max_degradation_percentage_points"
            ],
        }
        candidate_retention = [
            row
            for row in material_retention
            if row["candidate"] == candidate
            and row["partition"] == "validation"
        ]
        retention_by_material = {
            row["material"]: float(row["retention"])
            for row in candidate_retention
        }
        water_retention = retention_by_material.get("Water", float("nan"))
        dense = [
            retention_by_material[name]
            for name in ("A150_Tissue_Plastic", "SpineBone", "Aluminium")
            if name in retention_by_material
        ]
        retention_difference = (
            max(abs(value - water_retention) for value in dense)
            if dense and math.isfinite(water_retention)
            else float("inf")
        )
        checks["material_retention"] = (
            retention_difference * 100.0
            <= selection[
                "material_retention_max_difference_percentage_points"
            ]
        )
        # Material-specific path tagging is deliberately reported during the
        # full S4 confirmation. At screening time the global per-angle
        # retention guard prevents empty or collapsed angular bins.
        candidate_rows = [
            row
            for row in retention_rows
            if row["dataset"] == "s4" and row["candidate"] == candidate
        ]
        checks["no_empty_angle"] = all(
            int(row["train_accepted"]) > 0 for row in candidate_rows
        )
        passed = candidate == baseline or all(checks.values())
        decisions.append(
            {
                "candidate": candidate,
                "passed": passed,
                "checks": checks,
                "validation_rmse_mm": rmse,
                "validation_pool": "fixed baseline-accepted validation rows",
                "abs_p99_mm": p99,
                "p99_pool": "candidate-accepted validation rows",
                "material_mape": mape,
                "small_aluminium_ape": small,
                "max_dense_water_retention_difference": retention_difference,
            }
        )
    passed = [row for row in decisions if row["passed"]]
    winner = min(
        passed,
        key=lambda row: (
            float(row["validation_rmse_mm"]),
            config["filtering"]["candidates"].index(row["candidate"]),
        ),
    )["candidate"]
    return {
        "status": "PASS",
        "winner": winner,
        "baseline_retained": winner == baseline,
        "decisions": decisions,
        "test_partition_opened": False,
    }


def collect_noise_training(
    dataset: dict[str, Any],
    config: dict[str, Any],
    filter_name: str,
    runs: int,
) -> tuple[np.ndarray, np.ndarray]:
    noise = config["noise_model"]
    width = float(noise["energy_bin_width_mev"])
    maximum = int(noise["reservoir_rows_per_bin"])
    rng = np.random.default_rng(int(config["split"]["seed"]))
    reservoirs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        partitions = partition_masks(len(pairs), run_id, config["split"])
        accepted = read_packed_mask(
            mask_path(dataset, filter_name, run_id),
            len(pairs),
            config["split"]["bit_order"],
        )
        inside, cells, _ = pair_features(pairs, config["filtering"])
        selected = partitions["train"] & accepted & inside
        wepl = energies_to_wepl_vectorized(
            wepl_lut(), pairs[:, 4, 0], pairs[:, 4, 1]
        ).astype(np.float64)
        residual = np.zeros(len(pairs), dtype=np.float64)
        for cell in np.unique(cells[selected]):
            index = selected & (cells == cell)
            residual[index] = wepl[index] - np.median(wepl[index])
        energy = np.asarray(pairs[:, 4, 1], dtype=np.float64)
        bins = np.floor(energy / width).astype(np.int64)
        for bin_id in np.unique(bins[selected]):
            index = selected & (bins == bin_id)
            new_energy = energy[index]
            new_residual = residual[index]
            new_key = rng.random(len(new_energy))
            if bin_id in reservoirs:
                old_energy, old_residual, old_key = reservoirs[bin_id]
                new_energy = np.concatenate((old_energy, new_energy))
                new_residual = np.concatenate((old_residual, new_residual))
                new_key = np.concatenate((old_key, new_key))
            if len(new_key) > maximum:
                keep = np.argpartition(new_key, maximum - 1)[:maximum]
                new_energy = new_energy[keep]
                new_residual = new_residual[keep]
                new_key = new_key[keep]
            reservoirs[bin_id] = (new_energy, new_residual, new_key)
        if (run_id + 1) % 40 == 0 or run_id == runs - 1:
            print(
                f"noise reservoir: {run_id+1:03d}/{runs}, "
                f"stored={sum(len(item[0]) for item in reservoirs.values()):,}",
                flush=True,
            )
    energy = np.concatenate([item[0] for item in reservoirs.values()])
    residual = np.concatenate([item[1] for item in reservoirs.values()])
    return energy, residual


def build_noise_and_weights(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    winner: str,
    runs: int,
    force: bool,
) -> NoiseModel:
    model_path_out = QC_ROOT / "noise_model.npz"
    if model_path_out.is_file() and not force:
        noise_model = NoiseModel.load(model_path_out)
    else:
        s2 = next(
            (item for item in datasets if item["name"] == "s2"),
            dataset_record(config, "s2"),
        )
        energy, residual = collect_noise_training(s2, config, winner, runs)
        noise_model = fit_noise_model(energy, residual, config["noise_model"])
        noise_model.save(model_path_out)
    calibration_rows = []
    for dataset in datasets:
        for run_id in range(runs):
            pairs = paircuts.read_mhd(
                Path(dataset["preprocessing_data"])
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            partitions = partition_masks(len(pairs), run_id, config["split"])
            accepted = read_packed_mask(
                mask_path(dataset, winner, run_id),
                len(pairs),
                config["split"]["bit_order"],
            )
            distance = np.load(distance_path(dataset, winner, run_id))
            sigma = noise_model.predict(pairs[:, 4, 1])
            confidence = np.minimum(
                1.0,
                float(config["weights"]["confidence_distance"])
                / np.maximum(distance, 1.0e-12),
            )
            raw = {
                "equal": np.ones(len(pairs)),
                "inverse_variance": 1.0 / np.square(sigma),
                "robust_confidence": confidence,
                "combined": confidence / np.square(sigma),
            }
            selected_train = partitions["train"] & accepted
            for variant, values in raw.items():
                weights = normalize_and_clip_weights(
                    values,
                    selected_train,
                    tuple(config["weights"]["clip"]),
                )
                output = (
                    stage_root(dataset)
                    / "weights"
                    / winner
                    / variant
                    / f"weights_{run_id:04d}.npy"
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(output, weights)
                accepted_weights = weights[selected_train]
                calibration_rows.append(
                    {
                        "dataset": dataset["name"],
                        "run_id": run_id,
                        "variant": variant,
                        "accepted_train": len(accepted_weights),
                        "weight_min": float(accepted_weights.min()),
                        "weight_median": float(np.median(accepted_weights)),
                        "weight_max": float(accepted_weights.max()),
                        "effective_sample_size": effective_sample_size(
                            accepted_weights
                        ),
                        "effective_fraction": effective_sample_size(
                            accepted_weights
                        )
                        / len(accepted_weights),
                    }
                )
        print(f"weights {dataset['name']}: complete", flush=True)
    write_csv(QC_ROOT / "weight_distribution_by_angle.csv", calibration_rows)
    minimum = float(config["weights"]["minimum_effective_fraction"])
    if any(
        float(row["effective_fraction"]) < minimum
        for row in calibration_rows
    ):
        raise RuntimeError("one or more angular weight distributions collapsed")
    write_json(
        QC_ROOT / "noise_model.json",
        {
            "status": "PASS",
            "training_dataset": "s2",
            "fit_partition": "train only",
            "energy_mev": noise_model.energy_mev.tolist(),
            "sigma_mm": noise_model.sigma_mm.tolist(),
            "bin_count": noise_model.bin_count.tolist(),
            "model_path": relative(model_path_out),
        },
    )
    return noise_model


def noise_calibration(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    filter_name: str,
    noise_model: NoiseModel,
    runs: int,
) -> dict[str, Any]:
    rows = []
    summaries = []
    for dataset in datasets:
        standardized_chunks = []
        sigma_chunks = []
        residual_chunks = []
        for run_id in range(runs):
            pairs = paircuts.read_mhd(
                Path(dataset["preprocessing_data"])
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            partitions = partition_masks(len(pairs), run_id, config["split"])
            accepted = read_packed_mask(
                mask_path(dataset, filter_name, run_id),
                len(pairs),
                config["split"]["bit_order"],
            )
            inside, cells, _ = pair_features(pairs, config["filtering"])
            corrected = (
                air_correct_pairs(
                    pairs,
                    wepl_lut(),
                    config["air_correction"],
                    energies_to_wepl_vectorized,
                )
                if str(dataset["world_material"]).lower() == "air"
                else pairs
            )
            wepl = energies_to_wepl_vectorized(
                wepl_lut(), corrected[:, 4, 0], corrected[:, 4, 1]
            ).astype(np.float64)
            train = partitions["train"] & accepted & inside
            validation = partitions["validation"] & accepted & inside
            residual = np.zeros(len(pairs), dtype=np.float64)
            calibrated = np.zeros(len(pairs), dtype=bool)
            for cell in np.unique(cells[train]):
                train_cell = train & (cells == cell)
                validation_cell = validation & (cells == cell)
                if np.any(validation_cell):
                    residual[validation_cell] = (
                        wepl[validation_cell] - np.median(wepl[train_cell])
                    )
                    calibrated[validation_cell] = True
            sigma = noise_model.predict(pairs[:, 4, 1])
            selected_residual = residual[validation & calibrated]
            selected_sigma = sigma[validation & calibrated]
            residual_chunks.append(selected_residual)
            sigma_chunks.append(selected_sigma)
            standardized_chunks.append(selected_residual / selected_sigma)
        residual = np.concatenate(residual_chunks)
        sigma = np.concatenate(sigma_chunks)
        standardized = np.concatenate(standardized_chunks)
        order = np.argsort(sigma, kind="stable")
        boundaries = np.linspace(0, len(order), 11, dtype=np.int64)
        passing = 0
        for index in range(10):
            selected = order[boundaries[index] : boundaries[index + 1]]
            if len(selected) == 0:
                raise RuntimeError(
                    f"{dataset['name']}: empty predicted-sigma decile {index+1}"
                )
            actual = float(np.sqrt(np.mean(residual[selected] ** 2)))
            predicted = float(np.sqrt(np.mean(sigma[selected] ** 2)))
            ratio = actual / predicted
            lower, upper = config["noise_model"]["calibration_ratio_range"]
            passed = float(lower) <= ratio <= float(upper)
            passing += int(passed)
            rows.append(
                {
                    "dataset": dataset["name"],
                    "sigma_decile": index + 1,
                    "count": int(np.count_nonzero(selected)),
                    "predicted_rms_sigma_mm": predicted,
                    "actual_rmse_mm": actual,
                    "actual_to_predicted": ratio,
                    "passed": passed,
                }
            )
        summaries.append(
            {
                "dataset": dataset["name"],
                "count": len(residual),
                "passing_deciles": passing,
                "coverage_1sigma": float(
                    np.mean(np.abs(standardized) <= 1.0)
                ),
                "coverage_2sigma": float(
                    np.mean(np.abs(standardized) <= 2.0)
                ),
                "coverage_3sigma": float(
                    np.mean(np.abs(standardized) <= 3.0)
                ),
            }
        )
    write_csv(QC_ROOT / "noise_calibration_deciles.csv", rows)
    write_csv(QC_ROOT / "noise_calibration_summary.csv", summaries)
    minimum = int(config["noise_model"]["minimum_passing_deciles"])
    passed = all(int(row["passing_deciles"]) >= minimum for row in summaries)
    result = {
        "status": "PASS" if passed else "FAIL",
        "inverse_variance_eligible": passed,
        "minimum_passing_deciles": minimum,
        "datasets": summaries,
    }
    write_json(QC_ROOT / "noise_calibration.json", result)
    return result


def build_equal_weights(
    dataset: dict[str, Any],
    config: dict[str, Any],
    filter_name: str,
    runs: int,
) -> None:
    output_root = stage_root(dataset) / "weights" / filter_name / "equal"
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        output = output_root / f"weights_{run_id:04d}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.ones(len(pairs), dtype=np.float32))


def weighted_iterative(
    dataset: dict[str, Any],
    config: dict[str, Any],
    filter_name: str,
    weight_name: str,
    device: int,
    runs: int,
    force: bool,
) -> Path:
    import cupy as cp
    from weighted_gpu import WeightedGpuMlpProjector

    if runs != RUNS:
        raise RuntimeError("formal weighted reconstruction requires 720 angles")
    output_root = (
        reconstruction_root(dataset)
        / "iterative"
        / filter_name
        / weight_name
    )
    recon_dir = output_root / "recon"
    final = recon_dir / "recon_iterative_gpu.mhd"
    if final.is_file() and not force:
        return final
    if force:
        shutil.rmtree(output_root, ignore_errors=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    initial_path = (
        reconstruction_root(dataset) / "analytic" / filter_name / "recon.mhd"
    )
    if not initial_path.is_file():
        raise FileNotFoundError(initial_path)
    settings = config["iterative"]
    image_cpu, source_spacing, source_origin = read_image_2d(initial_path)
    # read_image_2d exposes a read-only memmap view. Stage 3 applies support
    # and non-negativity in-place, so explicitly detach a writable float copy.
    image_cpu = np.array(image_cpu, dtype=np.float32, copy=True)
    spacing = float(source_spacing[0])
    origin = float(source_origin[0])
    if not (
        np.isclose(source_spacing[2], spacing)
        and np.isclose(source_origin[2], origin)
    ):
        raise RuntimeError("Stage-3 iterative initialization must use a square x-z grid")
    if image_cpu.shape != (
        int(settings["grid_size"]),
        int(settings["grid_size"]),
    ):
        raise RuntimeError(f"unexpected initial grid: {image_cpu.shape}")
    coordinates = origin + np.arange(len(image_cpu)) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    support_cpu = xx * xx + zz * zz <= float(settings["phantom_radius_mm"]) ** 2
    image_cpu[~support_cpu] = 0.0
    np.maximum(image_cpu, 0.0, out=image_cpu)
    write_image_2d(recon_dir / "initial.mhd", image_cpu, spacing, origin)
    cp.cuda.Device(device).use()
    image = cp.asarray(image_cpu)
    support = cp.asarray(support_cpu)
    projector = WeightedGpuMlpProjector(
        int(settings["grid_size"]),
        float(settings["grid_spacing_mm"]),
        float(settings["path_step_mm"]),
        float(settings["phantom_radius_mm"]),
    )
    rows = []
    total_start = time.perf_counter()
    completed_pairs = 0
    accepted_counts = []
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        partition = partition_masks(len(pairs), run_id, config["split"])["train"]
        accepted = read_packed_mask(
            mask_path(dataset, filter_name, run_id),
            len(pairs),
            config["split"]["bit_order"],
        )
        accepted_counts.append(int(np.count_nonzero(partition & accepted)))
    total_pairs = int(sum(accepted_counts) * int(settings["epochs"]))
    for epoch in range(int(settings["epochs"])):
        relaxation = float(settings["relaxation"]) / (
            1.0 + float(settings["relaxation_decay"]) * epoch
        )
        for subset in range(int(settings["subsets"])):
            subset_start = time.perf_counter()
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            weighted_squared = 0.0
            weight_sum = 0.0
            valid = 0
            subset_pairs = 0
            for run_id in range(subset, runs, int(settings["subsets"])):
                pairs = paircuts.read_mhd(
                    Path(dataset["preprocessing_data"])
                    / "pairs"
                    / f"pairs{run_id:04d}.mhd"
                )
                partition = partition_masks(
                    len(pairs), run_id, config["split"]
                )["train"]
                accepted = read_packed_mask(
                    mask_path(dataset, filter_name, run_id),
                    len(pairs),
                    config["split"]["bit_order"],
                )
                indices = np.flatnonzero(partition & accepted)
                weights = np.load(
                    stage_root(dataset)
                    / "weights"
                    / filter_name
                    / weight_name
                    / f"weights_{run_id:04d}.npy",
                    mmap_mode="r",
                )
                batch_size = int(settings["batch_size"])
                for begin in range(0, len(indices), batch_size):
                    batch_index = indices[begin : begin + batch_size]
                    selected = np.asarray(pairs[batch_index], dtype=np.float32)
                    metrics = projector.accumulate_weighted(
                        image,
                        make_batch(selected, dataset, config),
                        np.asarray(weights[batch_index], dtype=np.float32),
                        0.5 * run_id,
                        numerator,
                        denominator,
                    )
                    subset_pairs += len(selected)
                    completed_pairs += len(selected)
                    weighted_squared += float(metrics["weighted_squared"])
                    weight_sum += float(metrics["weight_sum"])
                    valid += int(metrics["valid"])
            observed = denominator > 0
            update = cp.where(
                observed,
                np.float32(relaxation)
                * numerator
                / cp.maximum(denominator, 1.0e-20),
                0.0,
            )
            image += update
            cp.maximum(image, 0.0, out=image)
            image *= support
            cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter() - total_start
            rate = completed_pairs / elapsed
            eta = (
                (total_pairs - completed_pairs) / rate
                if rate > 0
                else float("inf")
            )
            row = {
                "dataset": dataset["name"],
                "filter": filter_name,
                "weight": weight_name,
                "epoch": epoch + 1,
                "subset": subset,
                "relaxation": relaxation,
                "pairs": subset_pairs,
                "valid": valid,
                "weighted_rmse_mm": math.sqrt(weighted_squared / weight_sum),
                "update_l2": float(cp.linalg.norm(update).get()),
                "update_max_abs": float(cp.max(cp.abs(update)).get()),
                "elapsed_seconds": time.perf_counter() - subset_start,
            }
            rows.append(row)
            print(
                f"{dataset['name']}/{weight_name} epoch {epoch+1}/"
                f"{settings['epochs']} subset {subset+1:02d}/"
                f"{settings['subsets']}: total={completed_pairs/total_pairs:6.2%}, "
                f"weighted_rmse={row['weighted_rmse_mm']:.5f} mm, "
                f"ETA={format_duration(eta)}",
                flush=True,
            )
            del numerator, denominator, update
        image, regularization = proximal_regularize(
            image,
            support,
            kind=str(settings["regularizer"]),
            weight=float(settings["regularization_weight"]),
            iterations=int(settings["regularization_iterations"]),
            huber_delta=float(settings["huber_delta"]),
            primal_step=float(settings["primal_step"]),
            dual_step=float(settings["dual_step"]),
        )
        write_image_2d(
            recon_dir / f"epoch_{epoch+1:02d}.mhd",
            cp.asnumpy(image),
            spacing,
            origin,
        )
        print(
            f"{dataset['name']}/{weight_name} epoch {epoch+1}: "
            f"regularization {regularization['elapsed_seconds']:.2f}s",
            flush=True,
        )
    write_image_2d(final, cp.asnumpy(image), spacing, origin)
    write_csv(output_root / "iteration_history.csv", rows)
    write_json(
        output_root / "run_summary.json",
        {
            "status": "PASS",
            "dataset": dataset["name"],
            "filter": filter_name,
            "weight": weight_name,
            "elapsed_seconds": time.perf_counter() - total_start,
            "pairs_per_epoch": sum(accepted_counts),
            "output": relative(final),
        },
    )
    return final


def weight_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    jobs: int,
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    if {item["name"] for item in datasets} != {"s2", "s4"}:
        raise SystemExit("weight-screen must be run with --datasets s2,s4")
    selection = load_json(QC_ROOT / "filter_selection.json")
    winner_filter = str(selection["winner"])
    noise_model = build_noise_and_weights(
        datasets, config, winner_filter, runs, force
    )
    s3 = dataset_record(config, "s3")
    fit_filters(s3, config, jobs, runs, force)
    calibration = noise_calibration(
        [next(item for item in datasets if item["name"] == "s2"), s3],
        config,
        winner_filter,
        noise_model,
        runs,
    )
    image_rows = []
    validation_rows = []
    for dataset in datasets:
        paths = {}
        for weight in config["weights"]["variants"]:
            path = weighted_iterative(
                dataset,
                config,
                winner_filter,
                weight,
                device,
                runs,
                force,
            )
            paths[weight] = path
            metrics, materials = image_metrics(dataset, config, path)
            row = {
                "dataset": dataset["name"],
                "filter": winner_filter,
                "weight": weight,
                "image_path": relative(path),
                **metrics,
            }
            if dataset["name"] == "s4":
                non_air = [
                    item
                    for item in materials
                    if item.get("absolute_relative_error") is not None
                ]
                row["material_mape_non_air"] = float(
                    np.mean(
                        [item["absolute_relative_error"] for item in non_air]
                    )
                )
            image_rows.append(row)
        validation_rows.extend(
            evaluate_fixed_partition(
                dataset, config, paths, "validation", device, runs
            )
        )
    write_csv(QC_ROOT / "weight_image_metrics.csv", image_rows)
    write_csv(QC_ROOT / "weight_validation_wepl.csv", validation_rows)
    winner = select_weight(config, image_rows, validation_rows, calibration)
    write_json(QC_ROOT / "weight_selection.json", winner)
    return winner


def select_weight(
    config: dict[str, Any],
    image_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    baseline = "equal"
    validation = {
        (row["dataset"], row["candidate"]): row for row in validation_rows
    }
    images = {
        (row["dataset"], row["weight"]): row for row in image_rows
    }
    base_mape = float(images[("s4", baseline)]["material_mape_non_air"])
    decisions = []
    for order, weight in enumerate(config["weights"]["variants"]):
        rmse = np.mean(
            [
                float(validation[(dataset, weight)]["rmse_mm"])
                for dataset in ("s2", "s4")
            ]
        )
        mape = float(images[("s4", weight)]["material_mape_non_air"])
        passed = (
            (mape - base_mape) * 100.0
            <= config["selection"][
                "material_mape_max_degradation_percentage_points"
            ]
        )
        if weight in {"inverse_variance", "combined"}:
            passed &= bool(calibration["inverse_variance_eligible"])
        decisions.append(
            {
                "weight": weight,
                "order": order,
                "passed": passed,
                "mean_validation_rmse_mm": float(rmse),
                "material_mape": mape,
                "noise_calibration_eligible": (
                    True
                    if weight in {"equal", "robust_confidence"}
                    else bool(calibration["inverse_variance_eligible"])
                ),
            }
        )
    valid = [row for row in decisions if row["passed"]]
    minimum = min(float(row["mean_validation_rmse_mm"]) for row in valid)
    tolerance = float(config["selection"]["weight_tie_relative_rmse"])
    tied = [
        row
        for row in valid
        if float(row["mean_validation_rmse_mm"])
        <= minimum * (1.0 + tolerance)
    ]
    winner = min(tied, key=lambda row: int(row["order"]))["weight"]
    return {
        "status": "PASS",
        "winner": winner,
        "decisions": decisions,
        "test_partition_opened": False,
    }


def confirm(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    jobs: int,
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    filter_selection = load_json(QC_ROOT / "filter_selection.json")
    weight_selection = load_json(QC_ROOT / "weight_selection.json")
    winner_filter = str(filter_selection["winner"])
    winner_weight = str(weight_selection["winner"])
    variants = [
        ("baseline_3sigma", "equal"),
        (winner_filter, winner_weight),
    ]
    result_rows = []
    test_rows = []
    for dataset in datasets:
        fit_filters(dataset, config, jobs, runs, force)
        build_noise_and_weights(
            [dataset], config, winner_filter, runs, False
        )
        build_equal_weights(
            dataset, config, "baseline_3sigma", runs
        )
        paths = {}
        for filter_name, weight_name in variants:
            generate_candidate_ddb(
                dataset, config, filter_name, jobs, runs, force
            )
            reconstruct_candidate(dataset, filter_name, runs, force)
            path = weighted_iterative(
                dataset,
                config,
                filter_name,
                weight_name,
                device,
                runs,
                force,
            )
            key = f"{filter_name}__{weight_name}"
            paths[key] = path
            metrics, details = image_metrics(dataset, config, path)
            result_rows.append(
                {
                    "dataset": dataset["name"],
                    "variant": key,
                    "image_path": relative(path),
                    **metrics,
                }
            )
        test_rows.extend(
            evaluate_fixed_partition(
                dataset, config, paths, "test", device, runs
            )
        )
    write_csv(QC_ROOT / "confirmation_image_metrics.csv", result_rows)
    write_csv(QC_ROOT / "confirmation_test_wepl.csv", test_rows)
    result = {
        "status": "PASS",
        "winner_filter": winner_filter,
        "winner_weight": winner_weight,
        "datasets": [item["name"] for item in datasets],
        "test_partition_opened": True,
        "note": (
            "PASS denotes a complete locked comparison; the report action "
            "decides whether the new method replaces or retains the baseline."
        ),
    }
    write_json(QC_ROOT / "confirmation_summary.json", result)
    return result


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    required = [
        QC_ROOT / "filter_selection.json",
        QC_ROOT / "weight_selection.json",
        QC_ROOT / "confirmation_summary.json",
        QC_ROOT / "confirmation_image_metrics.csv",
        QC_ROOT / "confirmation_test_wepl.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage-3 result: {missing[0]}")
    filter_selection = load_json(required[0])
    weight_selection = load_json(required[1])
    confirmation = load_json(required[2])
    test_rows = read_csv(required[4])
    lines = [
        "# 阶段3：稳健过滤、数据加权与噪声模型",
        "",
        f"- 冻结过滤器：`{filter_selection['winner']}`",
        f"- 冻结权重：`{weight_selection['winner']}`",
        f"- 独立确认数据：`{', '.join(confirmation['datasets'])}`",
        "- 测试集只在最终基线—胜出方案比较时打开。",
        "",
        "## 测试集WEPL结果",
        "",
        "| 数据 | 方案 | RMSE/mm | MAE/mm | bias/mm | |residual| p99/mm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['dataset']} | {row['candidate']} | "
            f"{float(row['rmse_mm']):.5f} | {float(row['mae_mm']):.5f} | "
            f"{float(row['bias_mm']):+.5f} | "
            f"{float(row['abs_p99_mm']):.5f} |"
        )
    lines.extend(
        [
            "",
            "完整材料、RSP、MTF、权重有效样本量和逐角度结果见同目录CSV。",
            "",
        ]
    )
    summary_path = QC_ROOT / "stage3_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "status": "PASS",
        "filter": filter_selection["winner"],
        "weight": weight_selection["winner"],
        "summary": relative(summary_path),
    }
    write_json(QC_ROOT / "stage3_summary.json", result)
    (QC_ROOT / "completed.flag").write_text(
        datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8"
    )
    return result


def smoke(config: dict[str, Any], device: int) -> dict[str, Any]:
    """Fast synthetic tests plus a two-angle read-only real-data fit."""

    rng = np.random.default_rng(20260713)
    n = 10000
    clean = rng.normal(size=(n, 3))
    clean[:, 0] = 20.0 + 2.0 * clean[:, 0]
    clean[:, 1:] *= 0.002
    contaminated = clean.copy()
    contaminated[-500:, 0] += 25.0
    contaminated[-500:, 1:] += 0.03
    cells = rng.integers(0, 250, size=n)
    inside = np.ones(n, dtype=bool)
    train = partition_masks(n, 0, config["split"])["train"]
    synthetic = {}
    for name in config["filtering"]["candidates"]:
        model = fit_filter(
            name, contaminated, cells, inside, train, config["filtering"]
        )
        selected, distance = apply_filter(
            model, contaminated, cells, inside, config["filtering"]
        )
        synthetic[name] = {
            "finite_distance": bool(np.isfinite(distance).all()),
            "clean_retention": float(np.mean(selected[:-500])),
            "outlier_retention": float(np.mean(selected[-500:])),
        }
        if not np.isfinite(distance).all():
            raise RuntimeError(f"{name}: non-finite synthetic distance")
    # Verify the historical implementation and the Stage-3 baseline agree
    # when both are deliberately fit on all rows (the formal fit is train-only).
    pairs = paircuts.read_mhd(
        REPOSITORY_ROOT
        / "data"
        / "preprocessing_data"
        / "results0717_s2_water_vacuum_pilot"
        / "pairs"
        / "pairs0000.mhd"
    )
    inside_real, cells_real, features_real = pair_features(
        pairs, config["filtering"]
    )
    legacy_config = dict(config["filtering"])
    legacy_config["minimum_cell_training_rows"] = 1
    legacy_config["maximum_merge_radius_cells"] = 0
    legacy_config["scale_floor_fraction"] = 0.0
    model = fit_filter(
        "baseline_3sigma",
        features_real,
        cells_real,
        inside_real,
        np.ones(len(pairs), dtype=bool),
        legacy_config,
    )
    selected, _ = apply_filter(
        model, features_real, cells_real, inside_real, legacy_config
    )
    historical, _ = paircuts.filter_pairs(pairs)
    reproduction_difference = abs(
        int(np.count_nonzero(selected)) - int(len(historical))
    )
    two_angle_rows = []
    with tempfile.TemporaryDirectory(prefix="pct-stage3-two-angle-") as temporary:
        temporary_root = Path(temporary)
        for run_id in (0, 719):
            run_pairs = paircuts.read_mhd(
                REPOSITORY_ROOT
                / "data"
                / "preprocessing_data"
                / "results0717_s2_water_vacuum_pilot"
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            partitions = partition_masks(
                len(run_pairs), run_id, config["split"]
            )
            run_inside, run_cells, run_features = pair_features(
                run_pairs, config["filtering"]
            )
            run_model = fit_filter(
                "robust_mahalanobis",
                run_features,
                run_cells,
                run_inside,
                partitions["train"],
                config["filtering"],
            )
            run_selected, run_distance = apply_filter(
                run_model,
                run_features,
                run_cells,
                run_inside,
                config["filtering"],
            )
            train_pairs = np.asarray(
                run_pairs[partitions["train"] & run_selected],
                dtype=np.float32,
            )
            pair_directory = temporary_root / f"pairs_{run_id:04d}"
            paircuts.write_mhd(
                pair_directory / f"pairs{run_id:04d}.mhd", train_pairs
            )
            projection.process_run(
                run_id,
                str(pair_directory),
                str(temporary_root),
                False,
                "ddb",
            )
            ddb = projection.read_mhd(
                temporary_root / "ddb" / f"proj{run_id:04d}.mhd"
            )
            two_angle_rows.append(
                {
                    "run_id": run_id,
                    "input": len(run_pairs),
                    "selected_train": len(train_pairs),
                    "distance_finite": bool(
                        np.isfinite(run_distance[run_inside]).all()
                    ),
                    "ddb_finite": bool(np.isfinite(ddb).all()),
                    "ddb_shape": list(ddb.shape),
                }
            )
    # GPU smoke checks W=1 exactly reproduces the base accumulation.
    import cupy as cp
    from gpu_mlp_operator import GpuMlpProjector
    from weighted_gpu import WeightedGpuMlpProjector

    cp.cuda.Device(device).use()
    sample = np.asarray(pairs[:8], dtype=np.float32)
    batch = make_batch(
        sample,
        dataset_record(config, "s2"),
        config,
    )
    image = cp.ones((64, 64), dtype=cp.float32)
    base_num = cp.zeros_like(image)
    base_den = cp.zeros_like(image)
    weighted_num = cp.zeros_like(image)
    weighted_den = cp.zeros_like(image)
    base = GpuMlpProjector(64, 4.0, 4.0, 100.0)
    weighted = WeightedGpuMlpProjector(64, 4.0, 4.0, 100.0)
    base.accumulate(image, batch, 0.0, base_num, base_den)
    weighted.accumulate_weighted(
        image,
        batch,
        np.ones(len(sample), dtype=np.float32),
        0.0,
        weighted_num,
        weighted_den,
    )
    numerator_error = float(cp.max(cp.abs(base_num - weighted_num)).get())
    denominator_error = float(cp.max(cp.abs(base_den - weighted_den)).get())
    scaled_num = cp.zeros_like(image)
    scaled_den = cp.zeros_like(image)
    scale_factor = 2.5
    weighted.accumulate_weighted(
        image,
        batch,
        np.full(len(sample), scale_factor, dtype=np.float32),
        0.0,
        scaled_num,
        scaled_den,
    )
    scaled_numerator_error = float(
        cp.max(cp.abs(scaled_num - scale_factor * base_num)).get()
    )
    scaled_denominator_error = float(
        cp.max(cp.abs(scaled_den - scale_factor * base_den)).get()
    )
    result = {
        "status": "PASS",
        "synthetic": synthetic,
        "historical_count": len(historical),
        "stage3_all_row_baseline_count": int(np.count_nonzero(selected)),
        "historical_count_difference": reproduction_difference,
        "historical_reproduction_tolerance_rows": 5,
        "two_angle_filter_ddb": two_angle_rows,
        "weighted_unity_numerator_max_abs_error": numerator_error,
        "weighted_unity_denominator_max_abs_error": denominator_error,
        "weighted_constant_scale": scale_factor,
        "weighted_scaled_numerator_max_abs_error": scaled_numerator_error,
        "weighted_scaled_denominator_max_abs_error": scaled_denominator_error,
    }
    if (
        reproduction_difference > 5
        or not all(
            row["distance_finite"] and row["ddb_finite"]
            for row in two_angle_rows
        )
        or numerator_error > 1.0e-5
        or denominator_error > 1.0e-5
        or scaled_numerator_error > 1.0e-4
        or scaled_denominator_error > 1.0e-4
    ):
        result["status"] = "FAIL"
    write_json(QC_ROOT / "smoke_test.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("Stage-3 smoke test failed")
    return result


def main() -> None:
    args = parse_args()
    if args.jobs < 1 or not 1 <= args.runs <= RUNS:
        raise SystemExit("--jobs must be positive and --runs must be in [1,720]")
    config = load_json(CONFIG_PATH)
    names = parse_datasets(args.datasets, config)
    datasets = [dataset_record(config, name) for name in names]
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.action == "all":
        required = {"s1", "s2", "s3", "s4", "s5"}
        if set(names) != required:
            raise SystemExit(
                "--action all is intentionally guarded; specify "
                "--datasets s1,s2,s3,s4,s5 explicitly"
            )
        by_name = {item["name"]: item for item in datasets}
        for dataset in datasets:
            prepare_dataset(
                dataset, config, args.jobs, args.runs, args.force
            )
        screen = [by_name["s2"], by_name["s4"]]
        filter_screen(
            screen,
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
        weight_screen(
            screen,
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
        confirm(
            [by_name["s1"], by_name["s3"], by_name["s5"]],
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
        build_report(config)
    if args.action == "prepare":
        for dataset in datasets:
            prepare_dataset(
                dataset, config, args.jobs, args.runs, args.force
            )
    if args.action == "filter-screen":
        filter_screen(
            datasets,
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
    if args.action == "weight-screen":
        weight_screen(
            datasets,
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
    if args.action == "confirm":
        confirm(
            datasets,
            config,
            args.jobs,
            args.device,
            args.runs,
            args.force,
        )
    if args.action == "report":
        build_report(config)
    if args.action == "smoke":
        smoke(config, args.device)
    print(
        json.dumps(
            {
                "status": "PASS",
                "action": args.action,
                "datasets": names,
                "runs": args.runs,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
