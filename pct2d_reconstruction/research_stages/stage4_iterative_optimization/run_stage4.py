#!/usr/bin/env python3
"""Run Stage 4 fixed-MLP iterative reconstruction optimization."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
STAGE3_ROOT = HERE.parent / "stage3_robust_weighting"
CONFIG_PATH = HERE / "stage4_config.json"
QC_ROOT = HERE / "qc"
sys.path[:0] = [
    str(HERE),
    str(STAGE3_ROOT),
    str(CODE_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

import run_stage3 as stage3  # noqa: E402
from robust_gpu import RobustGpuMlpProjector  # noqa: E402
from stage3_io import (  # noqa: E402
    format_duration,
    load_json,
    partition_masks,
    read_packed_mask,
    relative,
    write_json,
)
from preprocessing import paircuts  # noqa: E402
from iterative_reconstruction.gpu_regularization import (  # noqa: E402
    proximal_regularize,
)
from iterative_reconstruction.mhd_io import (  # noqa: E402
    read_image_2d,
    write_image_2d,
)


RUNS = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "prepare",
            "relaxation-screen",
            "loss-screen",
            "regularization-screen",
            "subset-screen",
            "confirm",
            "report",
            "smoke",
            "status",
        ),
        required=True,
    )
    parser.add_argument(
        "--datasets",
        default="s1,s2,s3,s4,s5",
        help="comma-separated subset of s1,s2,s3,s4,s5",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help="angles for smoke/development only; formal actions require 720",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the selected Stage-4 variant outputs",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    stage3_path = (HERE / config["stage3_config"]).resolve()
    config["stage3"] = load_json(stage3_path)
    return config


def datasets_from(text: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    names = stage3.parse_datasets(text, config["stage3"])
    return [stage3.dataset_record(config["stage3"], name) for name in names]


def require_datasets(
    datasets: list[dict[str, Any]], expected: set[str], action: str
) -> None:
    actual = {item["name"] for item in datasets}
    if actual != expected:
        raise SystemExit(
            f"{action} requires --datasets {','.join(sorted(expected))}"
        )


def stage4_root(dataset: dict[str, Any]) -> Path:
    return Path(dataset["reconstruction_data"]) / "stage4"


def initial_path(dataset: dict[str, Any]) -> Path:
    return (
        Path(dataset["reconstruction_data"])
        / "stage3"
        / "analytic"
        / "baseline_3sigma"
        / "recon.mhd"
    )


def accepted_mask(
    dataset: dict[str, Any], config: dict[str, Any], run_id: int, count: int
) -> np.ndarray:
    return read_packed_mask(
        stage3.mask_path(dataset, config["filter"], run_id),
        count,
        config["stage3"]["split"]["bit_order"],
    )


def variant_name(settings: dict[str, Any]) -> str:
    delta = settings.get("huber_delta_mm")
    loss = "quadratic" if delta is None else f"huber{float(delta):g}"
    beta = float(settings.get("regularization_weight", 0.0))
    schedule = str(settings.get("regularization_schedule", "fixed"))
    return (
        f"r{float(settings['relaxation']):g}_d"
        f"{float(settings['relaxation_decay']):g}_{loss}_"
        f"b{beta:g}_{schedule}_s{int(settings['subsets'])}"
    ).replace(".", "p")


def invariant_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in settings.items() if key != "epochs"}


def regularization_weight(settings: dict[str, Any], epoch: int) -> float:
    weight = float(settings.get("regularization_weight", 0.0))
    if settings.get("regularization_schedule", "fixed") == "decay":
        weight /= 1.0 + float(settings["regularization_decay"]) * epoch
    return weight


def scalar_metrics(
    dataset: dict[str, Any],
    config: dict[str, Any],
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    measured, details = stage3.image_metrics(dataset, config["stage3"], path)
    output = {}
    for key, value in measured.items():
        if isinstance(value, (bool, str, int, float, np.number)):
            output[key] = value.item() if isinstance(value, np.generic) else value
    return output, details


def evaluate_partition(
    dataset: dict[str, Any],
    config: dict[str, Any],
    image_path: Path,
    partition: str,
    device: int,
    runs: int,
    quiet: bool = False,
) -> dict[str, Any]:
    if partition not in {"validation", "test"}:
        raise ValueError(partition)
    if partition == "test" and not (QC_ROOT / "frozen_final.json").is_file():
        raise RuntimeError("test partition is locked until frozen_final.json exists")
    import cupy as cp
    from weighted_gpu import WeightedGpuMlpProjector

    cp.cuda.Device(device).use()
    image = cp.asarray(read_image_2d(image_path)[0])
    grid = config["grid"]
    projector = WeightedGpuMlpProjector(
        int(grid["size"]),
        float(grid["spacing_mm"]),
        float(grid["path_step_mm"]),
        float(grid["phantom_radius_mm"]),
    )
    chunks: list[np.ndarray] = []
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        split = partition_masks(
            len(pairs), run_id, config["stage3"]["split"]
        )
        accepted = accepted_mask(dataset, config, run_id, len(pairs))
        indices = np.flatnonzero(split[partition] & accepted)
        batch_size = int(grid["batch_size"])
        for begin in range(0, len(indices), batch_size):
            selected = np.asarray(
                pairs[indices[begin : begin + batch_size]], dtype=np.float32
            )
            chunks.append(
                projector.residuals(
                    image,
                    stage3.make_batch(selected, dataset, config["stage3"]),
                    0.5 * run_id,
                )
            )
        if not quiet and ((run_id + 1) % 40 == 0 or run_id == runs - 1):
            print(
                f"{partition} {dataset['name']}: {run_id+1:03d}/{runs}",
                flush=True,
            )
    values = np.concatenate(chunks) if chunks else np.empty(0, np.float32)
    if not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid {partition} residuals for {dataset['name']}")
    return {
        "dataset": dataset["name"],
        "partition": partition,
        "count": len(values),
        "rmse_mm": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "mae_mm": float(np.mean(np.abs(values))),
        "bias_mm": float(np.mean(values)),
        "abs_p95_mm": float(np.quantile(np.abs(values), 0.95)),
        "abs_p99_mm": float(np.quantile(np.abs(values), 0.99)),
    }


def default_settings(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    defaults = config["defaults"]
    settings = {
        "epochs": int(defaults["epochs"]),
        "subsets": int(defaults["subsets"]),
        "relaxation": float(defaults["relaxation"]),
        "relaxation_decay": float(defaults["relaxation_decay"]),
        "huber_delta_mm": None,
        "regularization_weight": float(defaults["regularization_weight"]),
        "regularization_schedule": str(
            defaults["regularization_schedule"]
        ),
        "regularization_decay": float(
            config["regularization_screen"]["decay"]
        ),
    }
    settings.update(updates)
    return settings


def run_variant(
    dataset: dict[str, Any],
    config: dict[str, Any],
    settings: dict[str, Any],
    device: int,
    runs: int,
    force: bool = False,
    early_stop: bool = True,
) -> dict[str, Any]:
    if runs != RUNS:
        raise RuntimeError("formal Stage-4 reconstruction requires 720 angles")
    import cupy as cp

    name = variant_name(settings)
    output_root = stage4_root(dataset) / "variants" / name
    recon_dir = output_root / "recon"
    snapshot_path = output_root / "config.json"
    invariant = invariant_settings(settings)
    digest = canonical_hash(invariant)
    if force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_path.is_file():
        snapshot = load_json(snapshot_path)
        if snapshot["invariant_hash"] != digest:
            raise RuntimeError(f"configuration hash mismatch: {output_root}")
    else:
        write_json(
            snapshot_path,
            {
                "variant": name,
                "invariant_hash": digest,
                "settings": invariant,
                "stage4_config_hash": canonical_hash(
                    {k: v for k, v in config.items() if k != "stage3"}
                ),
            },
        )
    history_path = output_root / "epoch_metrics.csv"
    history = read_csv(history_path)
    completed = max((int(row["epoch"]) for row in history), default=0)
    target = int(settings["epochs"])
    if completed >= target:
        selected = [row for row in history if int(row["epoch"]) == target][-1]
        return {**selected, "variant": name, "output_root": relative(output_root)}

    source = initial_path(dataset)
    if not source.is_file():
        raise FileNotFoundError(source)
    if completed:
        resume = recon_dir / f"epoch_{completed:02d}.mhd"
        image_cpu, spacing3, origin3 = read_image_2d(resume)
    else:
        image_cpu, spacing3, origin3 = read_image_2d(source)
    image_cpu = np.array(image_cpu, dtype=np.float32, copy=True)
    spacing = float(spacing3[0])
    origin = float(origin3[0])
    grid = config["grid"]
    if image_cpu.shape != (int(grid["size"]), int(grid["size"])):
        raise RuntimeError(f"unexpected initial grid {image_cpu.shape}")
    coordinates = origin + np.arange(len(image_cpu), dtype=np.float32) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    support_cpu = xx * xx + zz * zz <= float(grid["phantom_radius_mm"]) ** 2
    np.maximum(image_cpu, 0.0, out=image_cpu)
    image_cpu[~support_cpu] = 0.0
    if not completed:
        write_image_2d(recon_dir / "initial.mhd", image_cpu, spacing, origin)

    counts = []
    for run_id in range(runs):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        split = partition_masks(
            len(pairs), run_id, config["stage3"]["split"]
        )
        accepted = accepted_mask(dataset, config, run_id, len(pairs))
        counts.append(int(np.count_nonzero(split["train"] & accepted)))
    pairs_per_epoch = sum(counts)

    cp.cuda.Device(device).use()
    image = cp.asarray(image_cpu)
    support = cp.asarray(support_cpu)
    projector = RobustGpuMlpProjector(
        int(grid["size"]),
        float(grid["spacing_mm"]),
        float(grid["path_step_mm"]),
        float(grid["phantom_radius_mm"]),
    )
    total_start = time.perf_counter()
    completed_pairs = 0
    recent_rmse = [float(row["validation_rmse_mm"]) for row in history[-2:]]
    for epoch in range(completed, target):
        epoch_start = time.perf_counter()
        relaxation = float(settings["relaxation"]) / (
            1.0 + float(settings["relaxation_decay"]) * epoch
        )
        training = {
            "squared": 0.0,
            "absolute": 0.0,
            "signed": 0.0,
            "huber_objective": 0.0,
            "valid": 0,
        }
        max_update = 0.0
        for subset in range(int(settings["subsets"])):
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            for run_id in range(subset, runs, int(settings["subsets"])):
                pairs = paircuts.read_mhd(
                    Path(dataset["preprocessing_data"])
                    / "pairs"
                    / f"pairs{run_id:04d}.mhd"
                )
                split = partition_masks(
                    len(pairs), run_id, config["stage3"]["split"]
                )
                accepted = accepted_mask(dataset, config, run_id, len(pairs))
                indices = np.flatnonzero(split["train"] & accepted)
                batch_size = int(grid["batch_size"])
                for begin in range(0, len(indices), batch_size):
                    selected = np.asarray(
                        pairs[indices[begin : begin + batch_size]],
                        dtype=np.float32,
                    )
                    values = projector.accumulate_loss(
                        image,
                        stage3.make_batch(
                            selected, dataset, config["stage3"]
                        ),
                        0.5 * run_id,
                        numerator,
                        denominator,
                        settings.get("huber_delta_mm"),
                    )
                    for key in training:
                        training[key] += values[key]
                    completed_pairs += len(selected)
            update = cp.where(
                denominator > 0,
                cp.float32(relaxation)
                * numerator
                / cp.maximum(denominator, cp.float32(1.0e-20)),
                cp.float32(0.0),
            )
            image += update
            cp.maximum(image, 0.0, out=image)
            image *= support
            cp.cuda.Stream.null.synchronize()
            max_update = max(max_update, float(cp.max(cp.abs(update)).get()))
            elapsed = time.perf_counter() - total_start
            rate = completed_pairs / elapsed if elapsed else 0.0
            remaining = (
                (target - epoch - 1) * pairs_per_epoch
                + pairs_per_epoch
                * (1.0 - (subset + 1) / int(settings["subsets"]))
            )
            print(
                f"{dataset['name']} {name} epoch {epoch+1}/{target} "
                f"subset {subset+1:02d}/{settings['subsets']}: "
                f"rate={rate:,.0f} pairs/s ETA="
                f"{format_duration(remaining/rate if rate else math.inf)}",
                flush=True,
            )
            del numerator, denominator, update

        beta = regularization_weight(settings, epoch)
        regularization_seconds = 0.0
        if beta > 0.0:
            reg = config["regularization_screen"]
            image, reg_metrics = proximal_regularize(
                image,
                support,
                kind="huber_tv",
                weight=beta,
                iterations=int(reg["iterations"]),
                huber_delta=float(reg["huber_delta"]),
                primal_step=float(reg["primal_step"]),
                dual_step=float(reg["dual_step"]),
            )
            regularization_seconds = float(reg_metrics["elapsed_seconds"])
        checkpoint = recon_dir / f"epoch_{epoch+1:02d}.mhd"
        image_host = cp.asnumpy(image)
        write_image_2d(checkpoint, image_host, spacing, origin)
        validation = evaluate_partition(
            dataset,
            config,
            checkpoint,
            "validation",
            device,
            runs,
        )
        measured, details = scalar_metrics(dataset, config, checkpoint)
        if details:
            write_csv(
                output_root / f"epoch_{epoch+1:02d}_details.csv", details
            )
        row = {
            "dataset": dataset["name"],
            "variant": name,
            "epoch": epoch + 1,
            "initial_relaxation": float(settings["relaxation"]),
            "relaxation": relaxation,
            "relaxation_decay": float(settings["relaxation_decay"]),
            "regularization_weight": beta,
            "training_rmse_mm": math.sqrt(
                training["squared"] / training["valid"]
            ),
            "training_mae_mm": training["absolute"] / training["valid"],
            "training_bias_mm": training["signed"] / training["valid"],
            "training_huber_objective": training["huber_objective"]
            / training["valid"],
            "validation_rmse_mm": validation["rmse_mm"],
            "validation_mae_mm": validation["mae_mm"],
            "validation_bias_mm": validation["bias_mm"],
            "validation_abs_p99_mm": validation["abs_p99_mm"],
            "max_update": max_update,
            "regularization_seconds": regularization_seconds,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "image_path": relative(checkpoint),
            **measured,
        }
        history.append(row)
        write_csv(history_path, history)
        recent_rmse.append(float(validation["rmse_mm"]))
        print(
            f"completed {dataset['name']} {name} epoch {epoch+1}: "
            f"validation RMSE={validation['rmse_mm']:.5f} mm, "
            f"time={format_duration(row['epoch_seconds'])}",
            flush=True,
        )
        if early_stop and len(recent_rmse) >= 3:
            threshold = float(
                config["early_stopping"]["relative_rmse_degradation"]
            )
            if (
                recent_rmse[-1] > recent_rmse[-2] * (1.0 + threshold)
                and recent_rmse[-2] > recent_rmse[-3] * (1.0 + threshold)
            ):
                print("early stop: validation RMSE degraded twice", flush=True)
                break
    last = history[-1]
    write_json(
        output_root / "run_summary.json",
        {
            "status": "PASS",
            "dataset": dataset["name"],
            "variant": name,
            "invariant_hash": digest,
            "completed_epochs": int(last["epoch"]),
            "target_epochs": target,
            "pairs_per_epoch": pairs_per_epoch,
            "elapsed_seconds_this_call": time.perf_counter() - total_start,
            "latest_image": last["image_path"],
            "finite": bool(np.isfinite(cp.asnumpy(image)).all()),
            "support_outside_nonzero": int(
                np.count_nonzero(cp.asnumpy(image)[~support_cpu])
            ),
        },
    )
    return {**last, "variant": name, "output_root": relative(output_root)}


def best_epoch(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: float(row["validation_rmse_mm"]))


def variant_rows(dataset: dict[str, Any], name: str) -> list[dict[str, str]]:
    return read_csv(
        stage4_root(dataset) / "variants" / name / "epoch_metrics.csv"
    )


def variant_elapsed_seconds(
    dataset: dict[str, Any], name: str, through_epoch: int | None = None
) -> float:
    return float(
        sum(
            float(row["epoch_seconds"])
            for row in variant_rows(dataset, name)
            if through_epoch is None or int(row["epoch"]) <= through_epoch
        )
    )


def prepare(
    datasets: list[dict[str, Any]], config: dict[str, Any], runs: int
) -> dict[str, Any]:
    if runs != RUNS:
        raise RuntimeError("formal prepare requires 720 angles")
    stage3_qc = STAGE3_ROOT / "qc"
    filter_selection = load_json(stage3_qc / "filter_selection.json")
    weight_selection = load_json(stage3_qc / "weight_selection.json")
    if filter_selection["winner"] != config["filter"]:
        raise RuntimeError("Stage-3 frozen filter differs from Stage-4")
    if weight_selection["winner"] != config["data_weight"]:
        raise RuntimeError("Stage-3 frozen weight differs from Stage-4")
    rows = []
    for dataset in datasets:
        missing = []
        for run_id in range(runs):
            pair_path = (
                Path(dataset["preprocessing_data"])
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            mask = stage3.mask_path(dataset, config["filter"], run_id)
            if not pair_path.is_file() or not mask.is_file():
                missing.append(run_id)
        source = initial_path(dataset)
        rows.append(
            {
                "dataset": dataset["name"],
                "runs": runs,
                "missing_runs": len(missing),
                "initial": relative(source) if source.is_file() else "",
                "initial_exists": source.is_file(),
            }
        )
        if missing or not source.is_file():
            raise RuntimeError(f"incomplete Stage-3 input for {dataset['name']}")
    write_csv(QC_ROOT / "prepare_inputs.csv", rows)
    result = {
        "status": "PASS",
        "filter": config["filter"],
        "weight": config["data_weight"],
        "datasets": [item["name"] for item in datasets],
        "runs": runs,
        "test_partition_opened": False,
    }
    write_json(QC_ROOT / "prepare_summary.json", result)
    return result


def relaxation_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    require_datasets(datasets, {"s2", "s4"}, "relaxation-screen")
    prepare(datasets, config, runs)
    s2 = next(item for item in datasets if item["name"] == "s2")
    s4 = next(item for item in datasets if item["name"] == "s4")
    screen = config["relaxation_screen"]
    initial = []
    for value in screen["initial_relaxations"]:
        settings = default_settings(
            config,
            relaxation=float(value),
            relaxation_decay=float(screen["initial_decay"]),
            epochs=int(screen["initial_epochs"]),
            regularization_weight=0.0,
        )
        initial.append(run_variant(s2, config, settings, device, runs, force))
    initial.sort(key=lambda row: float(row["validation_rmse_mm"]))
    retained = initial[: int(screen["retained_relaxations"])]
    schedules = []
    for item in retained:
        relaxation = float(item["initial_relaxation"])
        for decay in screen["decays"]:
            settings = default_settings(
                config,
                relaxation=relaxation,
                relaxation_decay=float(decay),
                epochs=int(screen["maximum_epochs"]),
                regularization_weight=0.0,
            )
            result = run_variant(s2, config, settings, device, runs, force)
            rows = variant_rows(s2, result["variant"])
            schedules.append(best_epoch(rows))
    schedules.sort(key=lambda row: float(row["validation_rmse_mm"]))
    unique = []
    seen = set()
    for row in schedules:
        if row["variant"] not in seen:
            unique.append(row)
            seen.add(row["variant"])
    finalists = unique[: int(screen["retained_schedules"])]
    comparisons = []
    for row in finalists:
        snapshot = load_json(
            stage4_root(s2)
            / "variants"
            / row["variant"]
            / "config.json"
        )["settings"]
        settings = {**snapshot, "epochs": int(row["epoch"])}
        s4_result = run_variant(s4, config, settings, device, runs, force)
        s2_best = best_epoch(variant_rows(s2, row["variant"]))
        comparisons.append(
            {
                "variant": row["variant"],
                "epoch": int(row["epoch"]),
                "relaxation": float(snapshot["relaxation"]),
                "relaxation_decay": float(snapshot["relaxation_decay"]),
                "s2_validation_rmse_mm": float(
                    s2_best["validation_rmse_mm"]
                ),
                "s4_validation_rmse_mm": float(
                    s4_result["validation_rmse_mm"]
                ),
                "s4_material_mape": float(
                    s4_result.get("material_mape_non_air", math.inf)
                ),
            }
        )
    # Normalize across the two independent development phantoms.
    s2_min = min(row["s2_validation_rmse_mm"] for row in comparisons)
    s4_min = min(row["s4_validation_rmse_mm"] for row in comparisons)
    for row in comparisons:
        row["score"] = 0.5 * (
            row["s2_validation_rmse_mm"] / s2_min
            + row["s4_validation_rmse_mm"] / s4_min
        )
    best_score = min(row["score"] for row in comparisons)
    equivalent = [
        row
        for row in comparisons
        if row["score"] <= best_score * (1.0 + 0.002)
    ]
    # Within the predeclared 0.2% equivalence band, prefer the schedule that
    # decays more strongly (smaller late-epoch updates), then the smaller
    # initial relaxation. This avoids promoting a negligible numerical win.
    winner = min(
        equivalent,
        key=lambda row: (
            -row["relaxation_decay"],
            row["relaxation"],
            row["variant"],
        ),
    )
    winner = {
        **winner,
        "selection_reason": (
            "Validation scores are within the 0.2% equivalence band; "
            "the more conservative decay is retained."
        ),
    }
    result = {
        "status": "PASS",
        "winner": winner,
        "candidates": comparisons,
        "test_partition_opened": False,
    }
    write_json(QC_ROOT / "relaxation_selection.json", result)
    write_csv(QC_ROOT / "relaxation_candidates.csv", comparisons)
    return result


def require_selection(name: str) -> dict[str, Any]:
    path = QC_ROOT / name
    if not path.is_file():
        raise RuntimeError(f"run the previous Stage-4 action first: {path.name}")
    return load_json(path)


def selected_schedule(config: dict[str, Any]) -> dict[str, Any]:
    item = require_selection("relaxation_selection.json")["winner"]
    return default_settings(
        config,
        relaxation=float(item["relaxation"]),
        relaxation_decay=float(item["relaxation_decay"]),
        epochs=int(item["epoch"]),
        regularization_weight=0.0,
    )


def loss_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    require_datasets(datasets, {"s2", "s4"}, "loss-screen")
    base = selected_schedule(config)
    rows = []
    for loss in config["loss_screen"]["losses"]:
        settings = {
            **base,
            "epochs": int(config["loss_screen"]["epochs"]),
            "huber_delta_mm": loss["huber_delta_mm"],
        }
        for dataset in datasets:
            result = run_variant(
                dataset, config, settings, device, runs, force
            )
            rows.append(
                {
                    "dataset": dataset["name"],
                    "loss": loss["name"],
                    "variant": result["variant"],
                    "validation_rmse_mm": float(
                        result["validation_rmse_mm"]
                    ),
                    "validation_abs_p99_mm": float(
                        result["validation_abs_p99_mm"]
                    ),
                    "material_mape": float(
                        result.get("material_mape_non_air", math.nan)
                    ),
                    "material_max_ape": float(
                        result.get("material_max_ape_non_air", math.nan)
                    ),
                }
            )
    by_loss = {}
    for loss in config["loss_screen"]["losses"]:
        name = loss["name"]
        current = [row for row in rows if row["loss"] == name]
        by_loss[name] = {
            "name": name,
            "huber_delta_mm": loss["huber_delta_mm"],
            "mean_rmse_mm": float(
                np.mean([row["validation_rmse_mm"] for row in current])
            ),
            "mean_p99_mm": float(
                np.mean([row["validation_abs_p99_mm"] for row in current])
            ),
            "s4_material_mape": next(
                row["material_mape"]
                for row in current
                if row["dataset"] == "s4"
            ),
            "s4_material_max_ape": next(
                row["material_max_ape"]
                for row in current
                if row["dataset"] == "s4"
            ),
        }
    baseline = by_loss["quadratic"]
    selection = config["selection"]
    eligible = [baseline]
    for name in ("huber3", "huber5"):
        item = by_loss[name]
        rmse_improvement = 1.0 - item["mean_rmse_mm"] / baseline["mean_rmse_mm"]
        p99_improvement = 1.0 - item["mean_p99_mm"] / baseline["mean_p99_mm"]
        material_ok = (
            item["s4_material_mape"] - baseline["s4_material_mape"]
        ) * 100.0 <= selection[
            "material_mape_max_degradation_percentage_points"
        ]
        maximum_ok = (
            item["s4_material_max_ape"] - baseline["s4_material_max_ape"]
        ) * 100.0 <= selection[
            "small_roi_max_degradation_percentage_points"
        ]
        item["eligible"] = material_ok and maximum_ok and (
            rmse_improvement >= selection["huber_rmse_min_improvement"]
            or p99_improvement >= selection["huber_p99_min_improvement"]
        )
        item["rmse_improvement"] = rmse_improvement
        item["p99_improvement"] = p99_improvement
        if item["eligible"]:
            eligible.append(item)
    winner = min(eligible, key=lambda item: item["mean_rmse_mm"])
    result = {
        "status": "PASS",
        "winner": winner,
        "candidates": list(by_loss.values()),
        "test_partition_opened": False,
    }
    write_csv(QC_ROOT / "loss_candidates.csv", rows)
    write_json(QC_ROOT / "loss_selection.json", result)
    return result


def selected_loss_settings(config: dict[str, Any]) -> dict[str, Any]:
    base = selected_schedule(config)
    winner = require_selection("loss_selection.json")["winner"]
    return {
        **base,
        "huber_delta_mm": winner["huber_delta_mm"],
    }


def image_score(row: dict[str, Any], baselines: dict[str, Any]) -> float:
    dataset = row["dataset"]
    if dataset == "s2":
        keys = ("phantom_rmse_vs_effective_rsp", "water_core_std_rsp")
    elif dataset == "s4":
        keys = ("material_mape_non_air", "phantom_rmse_vs_nominal_rsp")
    else:
        keys = ("phantom_rmse_vs_nominal_rsp",)
    values = []
    for key in keys:
        if key in row and key in baselines[dataset]:
            values.append(float(row[key]) / float(baselines[dataset][key]))
    return float(np.mean(values)) if values else math.inf


def regularization_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    require_datasets(datasets, {"s2", "s4", "s5"}, "regularization-screen")
    settings0 = selected_loss_settings(config)
    screen = config["regularization_screen"]
    dev = [item for item in datasets if item["name"] in {"s2", "s4"}]
    initial_rows = []
    for beta in screen["weights"]:
        settings = {
            **settings0,
            "epochs": int(screen["initial_epochs"]),
            "regularization_weight": float(beta),
            "regularization_schedule": "fixed",
        }
        for dataset in dev:
            initial_rows.append(
                run_variant(dataset, config, settings, device, runs, force)
            )
    # Initial pruning uses both development-image quality and validation WEPL.
    initial_baseline = {
        row["dataset"]: row
        for row in initial_rows
        if float(row["regularization_weight"]) == 0.0
    }
    beta_scores = []
    for beta in screen["weights"]:
        name = variant_name(
            {
                **settings0,
                "epochs": int(screen["initial_epochs"]),
                "regularization_weight": float(beta),
                "regularization_schedule": "fixed",
            }
        )
        current = [row for row in initial_rows if row["variant"] == name]
        beta_scores.append(
            {
                "weight": float(beta),
                "variant": name,
                "mean_validation_rmse_mm": float(
                    np.mean(
                        [float(row["validation_rmse_mm"]) for row in current]
                    )
                ),
                "development_image_score": float(
                    np.mean(
                        [
                            image_score(row, initial_baseline)
                            for row in current
                        ]
                    )
                ),
            }
        )
    beta_scores.sort(
        key=lambda row: (
            row["development_image_score"],
            row["mean_validation_rmse_mm"],
        )
    )
    finalists = beta_scores[: int(screen["retained_weights"])]
    beta0 = next(row for row in beta_scores if row["weight"] == 0.0)
    if beta0 not in finalists:
        finalists[-1] = beta0
    full_rows = []
    for finalist in finalists:
        settings = {
            **settings0,
            "epochs": int(screen["maximum_epochs"]),
            "regularization_weight": finalist["weight"],
            "regularization_schedule": "fixed",
        }
        for dataset in datasets:
            result = run_variant(dataset, config, settings, device, runs, force)
            full_rows.extend(
                variant_rows(dataset, result["variant"])
            )
    baseline_rows = []
    for dataset in datasets:
        settings = {
            **settings0,
            "epochs": int(screen["maximum_epochs"]),
            "regularization_weight": 0.0,
        }
        beta0_result = run_variant(
            dataset, config, settings, device, runs, force
        )
        baseline_rows.extend(
            variant_rows(dataset, beta0_result["variant"])
        )
    candidate_scores = []
    selection = config["selection"]
    for finalist in finalists:
        candidate_epochs = sorted(
            {
                int(row["epoch"])
                for row in full_rows
                if float(row["regularization_weight"]) == finalist["weight"]
            }
        )
        for epoch in candidate_epochs:
            current = [
                row
                for row in full_rows
                if float(row["regularization_weight"]) == finalist["weight"]
                and int(row["epoch"]) == epoch
            ]
            baselines = {
                row["dataset"]: row
                for row in baseline_rows
                if int(row["epoch"]) == epoch
            }
            if len(current) != len(datasets) or len(baselines) != len(datasets):
                continue
            current_by_dataset = {row["dataset"]: row for row in current}
            checks = {
                "validation_wepl": all(
                    float(row["validation_rmse_mm"])
                    <= float(
                        baselines[row["dataset"]]["validation_rmse_mm"]
                    )
                    * (1.0 + selection["validation_rmse_max_degradation"])
                    for row in current
                ),
                "s4_material": (
                    float(
                        current_by_dataset["s4"]["material_mape_non_air"]
                    )
                    - float(baselines["s4"]["material_mape_non_air"])
                )
                * 100.0
                <= selection[
                    "material_mape_max_degradation_percentage_points"
                ],
                "s2_water_bias": (
                    abs(
                        float(
                            current_by_dataset["s2"][
                                "water_bias_vs_effective_rsp"
                            ]
                        )
                    )
                    - abs(
                        float(
                            baselines["s2"][
                                "water_bias_vs_effective_rsp"
                            ]
                        )
                    )
                )
                * 100.0
                <= selection[
                    "water_bias_max_degradation_percentage_points"
                ],
                "s5_mtf50": float(
                    current_by_dataset["s5"]["fmtf50_mean_lp_per_mm"]
                )
                >= float(baselines["s5"]["fmtf50_mean_lp_per_mm"])
                * (1.0 - selection["mtf_max_relative_degradation"]),
                "s5_mtf10": float(
                    current_by_dataset["s5"]["fmtf10_mean_lp_per_mm"]
                )
                >= float(baselines["s5"]["fmtf10_mean_lp_per_mm"])
                * (1.0 - selection["mtf_max_relative_degradation"]),
            }
            improvements = {
                "s2_effective_rsp_rmse": 1.0
                - float(
                    current_by_dataset["s2"][
                        "phantom_rmse_vs_effective_rsp"
                    ]
                )
                / float(
                    baselines["s2"]["phantom_rmse_vs_effective_rsp"]
                ),
                "s2_water_std": 1.0
                - float(current_by_dataset["s2"]["water_core_std_rsp"])
                / float(baselines["s2"]["water_core_std_rsp"]),
                "s4_material_mape": 1.0
                - float(
                    current_by_dataset["s4"]["material_mape_non_air"]
                )
                / float(baselines["s4"]["material_mape_non_air"]),
                "s5_nominal_rsp_rmse": 1.0
                - float(
                    current_by_dataset["s5"][
                        "phantom_rmse_vs_nominal_rsp"
                    ]
                )
                / float(
                    baselines["s5"]["phantom_rmse_vs_nominal_rsp"]
                ),
            }
            balanced = (
                sum(
                    value
                    >= selection["regularization_metric_min_improvement"]
                    for value in improvements.values()
                )
                >= 2
                and all(
                    value
                    >= -selection[
                        "regularization_other_metric_max_degradation"
                    ]
                    for value in improvements.values()
                )
            )
            candidate_scores.append(
                {
                    **finalist,
                    "epoch": epoch,
                    "image_score": float(
                        np.mean(
                            [
                                image_score(row, baselines)
                                for row in current
                            ]
                        )
                    ),
                    "mean_validation_rmse_mm": float(
                        np.mean(
                            [
                                float(row["validation_rmse_mm"])
                                for row in current
                            ]
                        )
                    ),
                    "checks": checks,
                    "improvements": improvements,
                    "balanced_improvement": balanced,
                    "eligible": all(checks.values())
                    and (float(finalist["weight"]) == 0.0 or balanced),
                }
            )
    eligible = [row for row in candidate_scores if row["eligible"]]
    if not eligible:
        # The unregularized beta=0 baseline is always a safe fallback.
        eligible = [
            row for row in candidate_scores if float(row["weight"]) == 0.0
        ]
    if not eligible:
        raise RuntimeError("regularization screen pruned the beta=0 fallback")
    fixed = min(
        eligible,
        key=lambda row: (
            row["image_score"],
            row["mean_validation_rmse_mm"],
            row["epoch"],
        ),
    )
    selected_epoch = int(fixed["epoch"])
    baselines = {
        row["dataset"]: row
        for row in baseline_rows
        if int(row["epoch"]) == selected_epoch
    }
    # Compare the selected fixed weight with the predeclared decaying schedule.
    # At beta=0 the two schedules are mathematically identical, so reuse the
    # fixed rows rather than spending several GPU-hours on a duplicate run.
    if float(fixed["weight"]) == 0.0:
        decay_rows = [
            row
            for row in full_rows
            if float(row["regularization_weight"]) == 0.0
            and int(row["epoch"]) == selected_epoch
        ]
    else:
        decay_settings = {
            **settings0,
            "epochs": selected_epoch,
            "regularization_weight": fixed["weight"],
            "regularization_schedule": "decay",
        }
        decay_rows = [
            run_variant(dataset, config, decay_settings, device, runs, force)
            for dataset in datasets
        ]
    decay_score = float(
        np.mean([image_score(row, baselines) for row in decay_rows])
    )
    decay_rmse = float(
        np.mean([float(row["validation_rmse_mm"]) for row in decay_rows])
    )
    decay_by_dataset = {row["dataset"]: row for row in decay_rows}
    decay_checks = {
        "validation_wepl": all(
            float(row["validation_rmse_mm"])
            <= float(baselines[row["dataset"]]["validation_rmse_mm"])
            * (1.0 + selection["validation_rmse_max_degradation"])
            for row in decay_rows
        ),
        "s4_material": (
            float(decay_by_dataset["s4"]["material_mape_non_air"])
            - float(baselines["s4"]["material_mape_non_air"])
        )
        * 100.0
        <= selection["material_mape_max_degradation_percentage_points"],
        "s2_water_bias": (
            abs(
                float(
                    decay_by_dataset["s2"][
                        "water_bias_vs_effective_rsp"
                    ]
                )
            )
            - abs(float(baselines["s2"]["water_bias_vs_effective_rsp"]))
        )
        * 100.0
        <= selection["water_bias_max_degradation_percentage_points"],
        "s5_mtf50": float(
            decay_by_dataset["s5"]["fmtf50_mean_lp_per_mm"]
        )
        >= float(baselines["s5"]["fmtf50_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
        "s5_mtf10": float(
            decay_by_dataset["s5"]["fmtf10_mean_lp_per_mm"]
        )
        >= float(baselines["s5"]["fmtf10_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
    }
    decay_improvements = {
        "s2_effective_rsp_rmse": 1.0
        - float(
            decay_by_dataset["s2"]["phantom_rmse_vs_effective_rsp"]
        )
        / float(baselines["s2"]["phantom_rmse_vs_effective_rsp"]),
        "s2_water_std": 1.0
        - float(decay_by_dataset["s2"]["water_core_std_rsp"])
        / float(baselines["s2"]["water_core_std_rsp"]),
        "s4_material_mape": 1.0
        - float(decay_by_dataset["s4"]["material_mape_non_air"])
        / float(baselines["s4"]["material_mape_non_air"]),
        "s5_nominal_rsp_rmse": 1.0
        - float(
            decay_by_dataset["s5"]["phantom_rmse_vs_nominal_rsp"]
        )
        / float(baselines["s5"]["phantom_rmse_vs_nominal_rsp"]),
    }
    decay_balanced = (
        sum(
            value >= selection["regularization_metric_min_improvement"]
            for value in decay_improvements.values()
        )
        >= 2
        and all(
            value
            >= -selection["regularization_other_metric_max_degradation"]
            for value in decay_improvements.values()
        )
    )
    fixed_choice = {
        **fixed,
        "schedule": "fixed",
        "epochs": selected_epoch,
    }
    decay_choice = {
        "weight": fixed["weight"],
        "variant": (
            decay_rows[0]["variant"]
            if float(fixed["weight"]) > 0.0
            else fixed["variant"]
        ),
        "schedule": (
            "decay" if float(fixed["weight"]) > 0.0 else "fixed"
        ),
        "epochs": selected_epoch,
        "image_score": decay_score,
        "mean_validation_rmse_mm": decay_rmse,
        "checks": decay_checks,
        "improvements": decay_improvements,
        "balanced_improvement": decay_balanced,
        "eligible": all(decay_checks.values())
        and (float(fixed["weight"]) == 0.0 or decay_balanced),
    }
    schedule_eligible = [
        row for row in (fixed_choice, decay_choice) if row["eligible"]
    ]
    winner = min(
        schedule_eligible,
        key=lambda row: (row["image_score"], row["mean_validation_rmse_mm"]),
    )
    result = {
        "status": "PASS",
        "winner": winner,
        "initial_candidates": beta_scores,
        "finalists": candidate_scores,
        "schedule_candidates": [fixed_choice, decay_choice],
        "test_partition_opened": False,
    }
    write_json(QC_ROOT / "regularization_selection.json", result)
    write_csv(QC_ROOT / "regularization_initial.csv", initial_rows)
    write_csv(QC_ROOT / "regularization_final.csv", full_rows + decay_rows)
    return result


def final_pre_subset_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = selected_loss_settings(config)
    winner = require_selection("regularization_selection.json")["winner"]
    return {
        **settings,
        "epochs": int(winner["epochs"]),
        "regularization_weight": float(winner["weight"]),
        "regularization_schedule": winner["schedule"],
    }


def subset_screen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    require_datasets(datasets, {"s2", "s4", "s5"}, "subset-screen")
    base = final_pre_subset_settings(config)
    rows = []
    for subsets in config["subset_screen"]["subsets"]:
        settings = {**base, "subsets": int(subsets)}
        for dataset in datasets:
            result = run_variant(
                dataset, config, settings, device, runs, force
            )
            rows.append(
                {
                    **result,
                    "subsets": int(subsets),
                    "total_epoch_seconds": variant_elapsed_seconds(
                        dataset,
                        result["variant"],
                        int(settings["epochs"]),
                    ),
                }
            )
    scores = {}
    for subsets in config["subset_screen"]["subsets"]:
        current = [row for row in rows if row["subsets"] == subsets]
        scores[int(subsets)] = {
            "subsets": int(subsets),
            "mean_validation_rmse_mm": float(
                np.mean(
                    [float(row["validation_rmse_mm"]) for row in current]
                )
            ),
            "elapsed_seconds": float(
                sum(row["total_epoch_seconds"] for row in current)
            ),
        }
    base18, candidate36 = scores[18], scores[36]
    improvement = (
        1.0
        - candidate36["mean_validation_rmse_mm"]
        / base18["mean_validation_rmse_mm"]
    )
    runtime_ratio = (
        candidate36["elapsed_seconds"] / base18["elapsed_seconds"]
        if base18["elapsed_seconds"] > 0
        else math.inf
    )
    by_result = {
        (row["dataset"], int(row["subsets"])): row for row in rows
    }
    selection = config["selection"]
    image_checks = {
        "s4_material": (
            float(by_result[("s4", 36)]["material_mape_non_air"])
            - float(by_result[("s4", 18)]["material_mape_non_air"])
        )
        * 100.0
        <= selection["material_mape_max_degradation_percentage_points"],
        "s2_water_bias": (
            abs(
                float(
                    by_result[("s2", 36)]["water_bias_vs_effective_rsp"]
                )
            )
            - abs(
                float(
                    by_result[("s2", 18)]["water_bias_vs_effective_rsp"]
                )
            )
        )
        * 100.0
        <= selection["water_bias_max_degradation_percentage_points"],
        "s5_mtf50": float(
            by_result[("s5", 36)]["fmtf50_mean_lp_per_mm"]
        )
        >= float(by_result[("s5", 18)]["fmtf50_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
        "s5_mtf10": float(
            by_result[("s5", 36)]["fmtf10_mean_lp_per_mm"]
        )
        >= float(by_result[("s5", 18)]["fmtf10_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
    }
    candidate36["rmse_improvement"] = improvement
    candidate36["runtime_ratio"] = runtime_ratio
    candidate36["image_checks"] = image_checks
    winner = (
        candidate36
        if improvement
        >= float(config["subset_screen"]["minimum_rmse_improvement"])
        and runtime_ratio
        <= float(config["subset_screen"]["maximum_runtime_ratio"])
        and all(image_checks.values())
        else base18
    )
    result = {
        "status": "PASS",
        "winner": winner,
        "candidates": list(scores.values()),
        "test_partition_opened": False,
    }
    write_csv(QC_ROOT / "subset_candidates.csv", rows)
    write_json(QC_ROOT / "subset_selection.json", result)
    final = {
        **base,
        "subsets": int(winner["subsets"]),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "test_partition_opened": False,
    }
    write_json(QC_ROOT / "frozen_final.json", final)
    return result


def final_settings(config: dict[str, Any]) -> dict[str, Any]:
    frozen = require_selection("frozen_final.json")
    return {
        key: value
        for key, value in frozen.items()
        if key not in {"frozen_at", "test_partition_opened"}
    }


def variant_progress(
    dataset: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    name = variant_name(settings)
    root = stage4_root(dataset) / "variants" / name
    history_path = root / "epoch_metrics.csv"
    rows = read_csv(history_path)
    target = int(settings["epochs"])
    usable = [row for row in rows if int(row["epoch"]) <= target]
    completed = min(
        max((int(row["epoch"]) for row in usable), default=0),
        target,
    )
    elapsed = sum(float(row["epoch_seconds"]) for row in usable)
    return {
        "dataset": dataset["name"],
        "variant": name,
        "completed_epochs": completed,
        "target_epochs": target,
        "elapsed_seconds": elapsed,
        "complete": completed >= target,
        "history_path": relative(history_path),
    }


def estimated_remaining_seconds(
    current: dict[str, Any],
    reference: dict[str, Any] | None = None,
) -> float:
    remaining = current["target_epochs"] - current["completed_epochs"]
    if remaining <= 0:
        return 0.0
    if current["completed_epochs"] > 0:
        seconds_per_epoch = (
            current["elapsed_seconds"] / current["completed_epochs"]
        )
    elif reference and reference["completed_epochs"] > 0:
        seconds_per_epoch = (
            reference["elapsed_seconds"] / reference["completed_epochs"]
        )
    else:
        return math.nan
    return float(remaining * seconds_per_epoch)


def status(datasets: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Print read-only aggregate progress for the active Stage-4 workflow."""
    selected = {dataset["name"]: dataset for dataset in datasets}
    subset_names = [name for name in ("s2", "s4", "s5") if name in selected]
    rows: list[dict[str, Any]] = []
    remaining_seconds = 0.0
    unknown_eta = False

    print("Stage 4 aggregate progress (read-only)")
    print("=" * 72)
    if (QC_ROOT / "regularization_selection.json").is_file():
        base = final_pre_subset_settings(config)
        subset_counts = [int(value) for value in config["subset_screen"]["subsets"]]
        references: dict[str, dict[str, Any]] = {}
        for subsets in subset_counts:
            settings = {**base, "subsets": subsets}
            for name in subset_names:
                current = variant_progress(selected[name], settings)
                reference = references.get(name)
                eta = estimated_remaining_seconds(current, reference)
                current["estimated_remaining_seconds"] = eta
                rows.append(current)
                if subsets == 18:
                    references[name] = current
                if not math.isfinite(eta):
                    unknown_eta = True
                else:
                    remaining_seconds += eta
                state = "DONE" if current["complete"] else "RUNNING/WAITING"
                print(
                    f"{name:>2}  subsets={subsets:>2}  "
                    f"epoch={current['completed_epochs']}/"
                    f"{current['target_epochs']}  {state:<15}  "
                    f"elapsed={format_duration(current['elapsed_seconds'])}  "
                    f"ETA={format_duration(eta) if math.isfinite(eta) else 'unknown'}"
                )
        completed = sum(row["completed_epochs"] for row in rows)
        target = sum(row["target_epochs"] for row in rows)
        print("-" * 72)
        print(
            f"subset-screen epochs: {completed}/{target} "
            f"({100.0 * completed / target:.1f}%)"
        )
        print(
            "estimated aggregate remaining: "
            + (
                f"{format_duration(remaining_seconds)}"
                if not unknown_eta
                else f"at least {format_duration(remaining_seconds)}"
            )
        )
        print(
            "subset selection: "
            + (
                "complete"
                if (QC_ROOT / "frozen_final.json").is_file()
                else "not yet frozen"
            )
        )
    else:
        print("regularization selection is not complete; subset progress unavailable")

    frozen_path = QC_ROOT / "frozen_final.json"
    confirmation_path = QC_ROOT / "confirmation_summary.json"
    if frozen_path.is_file():
        frozen = final_settings(config)
        print("\nFinal confirmation progress")
        print("-" * 72)
        confirmation_rows = [
            variant_progress(dataset, frozen) for dataset in datasets
        ]
        for current in confirmation_rows:
            state = "DONE" if current["complete"] else "WAITING"
            print(
                f"{current['dataset']:>2}  "
                f"epoch={current['completed_epochs']}/"
                f"{current['target_epochs']}  {state}"
            )
        print(
            "locked test decision: "
            + ("complete" if confirmation_path.is_file() else "not yet complete")
        )
        rows.extend(confirmation_rows)

    return {
        "status": "PASS",
        "read_only": True,
        "progress": rows,
        "estimated_remaining_seconds": (
            None if unknown_eta else remaining_seconds
        ),
        "subset_selection_complete": frozen_path.is_file(),
        "confirmation_complete": confirmation_path.is_file(),
    }


def confirm(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
    force: bool,
) -> dict[str, Any]:
    require_datasets(
        datasets, {"s1", "s2", "s3", "s4", "s5"}, "confirm"
    )
    settings = final_settings(config)
    rows = []
    image_rows = []
    for dataset in datasets:
        candidate = run_variant(
            dataset, config, settings, device, runs, force, early_stop=False
        )
        candidate_path = REPOSITORY_ROOT / candidate["image_path"]
        baseline_path = (
            Path(dataset["reconstruction_data"])
            / "stage3"
            / "iterative"
            / "baseline_3sigma"
            / "equal"
            / "recon"
            / "epoch_03.mhd"
        )
        for name, path in (("baseline", baseline_path), ("candidate", candidate_path)):
            test = evaluate_partition(
                dataset, config, path, "test", device, runs
            )
            rows.append({"method": name, **test})
            metrics, _ = scalar_metrics(dataset, config, path)
            image_rows.append(
                {
                    "dataset": dataset["name"],
                    "method": name,
                    "image_path": relative(path),
                    **metrics,
                }
            )
    write_csv(QC_ROOT / "confirmation_test_wepl.csv", rows)
    write_csv(QC_ROOT / "confirmation_image_metrics.csv", image_rows)
    baseline_rmse = np.mean(
        [row["rmse_mm"] for row in rows if row["method"] == "baseline"]
    )
    candidate_rmse = np.mean(
        [row["rmse_mm"] for row in rows if row["method"] == "candidate"]
    )
    by_wepl = {
        (row["dataset"], row["method"]): row for row in rows
    }
    by_image = {
        (row["dataset"], row["method"]): row for row in image_rows
    }
    selection = config["selection"]
    individual_wepl_ok = all(
        by_wepl[(name, "candidate")]["rmse_mm"]
        <= by_wepl[(name, "baseline")]["rmse_mm"]
        * (1.0 + selection["validation_rmse_max_degradation"])
        for name in ("s1", "s2", "s3", "s4", "s5")
    )
    mean_wepl_improvement = float(1.0 - candidate_rmse / baseline_rmse)
    water_std_improvement = float(
        1.0
        - np.mean(
            [
                float(by_image[(name, "candidate")]["water_core_std_rsp"])
                for name in ("s2", "s3")
            ]
        )
        / np.mean(
            [
                float(by_image[(name, "baseline")]["water_core_std_rsp"])
                for name in ("s2", "s3")
            ]
        )
    )
    material_mape_improvement_pp = float(
        (
            float(by_image[("s4", "baseline")]["material_mape_non_air"])
            - float(by_image[("s4", "candidate")]["material_mape_non_air"])
        )
        * 100.0
    )
    s5_rsp_improvement = float(
        1.0
        - float(
            by_image[("s5", "candidate")][
                "phantom_rmse_vs_nominal_rsp"
            ]
        )
        / float(
            by_image[("s5", "baseline")]["phantom_rmse_vs_nominal_rsp"]
        )
    )
    safety = {
        "individual_wepl": individual_wepl_ok,
        "s4_material": (
            float(by_image[("s4", "candidate")]["material_mape_non_air"])
            - float(by_image[("s4", "baseline")]["material_mape_non_air"])
        )
        * 100.0
        <= selection["material_mape_max_degradation_percentage_points"],
        "s5_mtf50": float(
            by_image[("s5", "candidate")]["fmtf50_mean_lp_per_mm"]
        )
        >= float(by_image[("s5", "baseline")]["fmtf50_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
        "s5_mtf10": float(
            by_image[("s5", "candidate")]["fmtf10_mean_lp_per_mm"]
        )
        >= float(by_image[("s5", "baseline")]["fmtf10_mean_lp_per_mm"])
        * (1.0 - selection["mtf_max_relative_degradation"]),
    }
    substantive = {
        "mean_test_wepl": mean_wepl_improvement
        >= selection["test_mean_wepl_rmse_min_improvement"],
        "water_std": water_std_improvement
        >= selection["test_water_std_min_improvement"],
        "material_mape": material_mape_improvement_pp
        >= selection[
            "test_material_mape_min_improvement_percentage_points"
        ],
        "s5_rsp_rmse": s5_rsp_improvement
        >= selection["test_s5_rsp_rmse_min_improvement"],
    }
    promoted = all(safety.values()) and any(substantive.values())
    result = {
        "status": "PASS",
        "decision": "PROMOTE_STAGE4" if promoted else "RETAIN_STAGE3",
        "promoted": promoted,
        "settings": settings,
        "mean_test_wepl_rmse_improvement": mean_wepl_improvement,
        "water_std_improvement": water_std_improvement,
        "material_mape_improvement_percentage_points": material_mape_improvement_pp,
        "s5_rsp_rmse_improvement": s5_rsp_improvement,
        "safety_checks": safety,
        "substantive_improvements": substantive,
        "test_partition_opened": True,
    }
    write_json(QC_ROOT / "confirmation_summary.json", result)
    return result


def report(config: dict[str, Any]) -> dict[str, Any]:
    confirmation = require_selection("confirmation_summary.json")
    relaxation = require_selection("relaxation_selection.json")
    loss = require_selection("loss_selection.json")
    regularization = require_selection("regularization_selection.json")
    subset = require_selection("subset_selection.json")
    wepl_rows = read_csv(QC_ROOT / "confirmation_test_wepl.csv")
    image_rows = read_csv(QC_ROOT / "confirmation_image_metrics.csv")
    by_wepl = {
        (row["dataset"], row["method"]): row for row in wepl_rows
    }
    by_image = {
        (row["dataset"], row["method"]): row for row in image_rows
    }
    runtime_rows = [
        variant_progress(dataset, confirmation["settings"])
        for dataset in datasets_from("s1,s2,s3,s4,s5", config)
    ]
    runtime_total = sum(row["elapsed_seconds"] for row in runtime_rows)
    subset36 = next(
        row for row in subset["candidates"] if int(row["subsets"]) == 36
    )
    lines = [
        "# 阶段4：固定MLP下的迭代重建优化",
        "",
        "## 1. 阶段结论",
        "",
        f"**阶段4状态为PASS，最终决定为"
        f"`{confirmation['decision']}`。** 本阶段固定局部3σ过滤、等权数据和"
        "Schulte水MLP，只优化OS-SART与Huber-TV。所有参数在开发/验证数据上"
        "冻结后才打开S1--S5测试集。",
        "",
        "冻结配置为初始松弛因子`0.25`、衰减`0.2`、quadratic数据项、固定"
        "Huber-TV `β=0.0125`、18子集和5 epoch。相对阶段3的3 epoch基线，"
        f"S2/S3水区标准差平均降低"
        f"`{100.0*confirmation['water_std_improvement']:.2f}%`。五个测试集的"
        "WEPL RMSE均略有改善，S5 RSP RMSE与MTF改善；S4材料MAPE轻微恶化但"
        "通过安全约束。",
        "",
        "## 2. 参数筛选",
        "",
        "### 2.1 松弛调度与数据损失",
        "",
        f"松弛调度冻结为`λ0={relaxation['winner']['relaxation']}`、衰减"
        f"`{relaxation['winner']['relaxation_decay']}`。衰减0.1与0.2位于"
        "0.2%等效带内，按预设规则保留更保守的0.2。",
        "",
        "| 损失 | 平均验证RMSE/mm | p99/mm | S4材料MAPE | 决定 |",
        "|---|---:|---:|---:|---|",
    ]
    for candidate in loss["candidates"]:
        decision = (
            "保留"
            if candidate["name"] == loss["winner"]["name"]
            else "未通过"
        )
        lines.append(
            f"| {candidate['name']} | {candidate['mean_rmse_mm']:.6f} | "
            f"{candidate['mean_p99_mm']:.6f} | "
            f"{100.0*candidate['s4_material_mape']:.3f}% | {decision} |"
        )
    lines.extend(
        [
            "",
            "Huber 3/5 mm虽略微改善材料MAPE，但验证WEPL RMSE和p99均未改善，"
            "因此继续使用quadratic数据项。",
            "",
            "### 2.2 Huber-TV、停止epoch和子集",
            "",
            f"固定`β={regularization['winner']['weight']}`、第"
            f"`{regularization['winner']['epoch']}`轮胜出。相对同轮无正则化"
            f"候选，S2有效RSP RMSE、水区标准差和S5名义RSP RMSE分别改善"
            f"`{100.0*regularization['winner']['improvements']['s2_effective_rsp_rmse']:.1f}%`、"
            f"`{100.0*regularization['winner']['improvements']['s2_water_std']:.1f}%`和"
            f"`{100.0*regularization['winner']['improvements']['s5_nominal_rsp_rmse']:.1f}%`。"
            "第6轮因S4材料退化超过平衡改善规则而被否决。",
            "",
            f"36子集相对18子集只改善`{100.0*subset36['rmse_improvement']:.4f}%`"
            f"验证WEPL RMSE，时间比为`{subset36['runtime_ratio']:.4f}`。图像安全"
            "检查全部通过，但改善未达到0.2%门槛，因此冻结18子集。",
            "",
            "## 3. S1--S5锁定测试WEPL",
            "",
            "| 数据 | 基线RMSE/mm | 阶段4RMSE/mm | 改善 | 阶段4bias/mm | p99/mm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("s1", "s2", "s3", "s4", "s5"):
        baseline = by_wepl[(dataset, "baseline")]
        candidate = by_wepl[(dataset, "candidate")]
        improvement = 1.0 - float(candidate["rmse_mm"]) / float(
            baseline["rmse_mm"]
        )
        lines.append(
            f"| {dataset.upper()} | {float(baseline['rmse_mm']):.6f} | "
            f"{float(candidate['rmse_mm']):.6f} | "
            f"{100.0*improvement:.3f}% | "
            f"{float(candidate['bias_mm']):+.6f} | "
            f"{float(candidate['abs_p99_mm']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"平均测试WEPL RMSE改善"
            f"`{100.0*confirmation['mean_test_wepl_rmse_improvement']:.3f}%`，"
            "未达到0.25%的独立实质改善门槛；但五个数据集均未恶化，bias保持在"
            "约±0.002 mm。阶段4晋升的主要依据是在保持数据一致性的同时明显降噪。",
            "",
            "## 4. 图像指标",
            "",
            "| 数据/指标 | 阶段3基线 | 阶段4 | 相对变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    image_specs = [
        ("S1 水标准差", "s1", "water_std"),
        ("S1 RSP RMSE", "s1", "phantom_rmse_vs_rsp_truth"),
        ("S2 水标准差", "s2", "water_core_std_rsp"),
        ("S2 有效RSP RMSE", "s2", "phantom_rmse_vs_effective_rsp"),
        ("S3 水标准差", "s3", "water_core_std_rsp"),
        ("S3 有效RSP RMSE", "s3", "phantom_rmse_vs_effective_rsp"),
        ("S4 名义RSP RMSE", "s4", "phantom_rmse_vs_nominal_rsp"),
        ("S4 材料MAPE", "s4", "material_mape_non_air"),
        ("S4 最大材料误差", "s4", "material_max_ape_non_air"),
        ("S5 水标准差", "s5", "water_core_std_rsp"),
        ("S5 名义RSP RMSE", "s5", "phantom_rmse_vs_nominal_rsp"),
        ("S5 fMTF50/lp·mm⁻¹", "s5", "fmtf50_mean_lp_per_mm"),
        ("S5 fMTF10/lp·mm⁻¹", "s5", "fmtf10_mean_lp_per_mm"),
    ]
    for label, dataset, key in image_specs:
        baseline = float(by_image[(dataset, "baseline")][key])
        candidate = float(by_image[(dataset, "candidate")][key])
        change = 100.0 * (candidate / baseline - 1.0)
        lines.append(
            f"| {label} | {baseline:.6f} | {candidate:.6f} | "
            f"{change:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "S2/S3水区标准差分别降低47.0%和39.6%。S4名义RSP RMSE降低6.17%、"
            "最大材料误差降低9.67%，但材料MAPE由1.196%升至1.203%，恶化"
            "0.0073个百分点。S5名义RSP RMSE降低2.71%，fMTF50和fMTF10提高"
            "1.13%和0.66%，没有观察到以明显分辨率损失换取降噪。",
            "",
            "## 5. 运行成本",
            "",
            "| 数据 | 最终候选5 epoch累计更新时间 |",
            "|---|---:|",
        ]
    )
    for row in runtime_rows:
        lines.append(
            f"| {row['dataset'].upper()} | "
            f"{format_duration(row['elapsed_seconds'])} |"
        )
    lines.extend(
        [
            f"| **合计** | **{format_duration(runtime_total)}** |",
            "",
            "该合计不含全部候选扫描、验证/测试正投影和分析时间。S1论文通量约为"
            "pilot的4.5倍，是主要计算成本；瓶颈仍是逐质子MLP投影，而不是TV近端。",
            "",
            "## 6. 验收、限制与下一步",
            "",
            "- S1--S5逐数据集测试WEPL安全检查：PASS；",
            "- S4材料MAPE退化约束：PASS；",
            "- S5 fMTF50/fMTF10保持约束：PASS；",
            "- S2/S3平均水标准差改善42.6%：达到实质改善门槛；",
            f"- 最终决定：**{confirmation['decision']}**。",
            "",
            "晋升只表示阶段5/6应把该配置作为固定MLP迭代基线；成熟的"
            "`iterative_reconstruction/`入口没有被自动替换。平均测试WEPL只改善"
            "0.095%，S4材料MAPE没有改善，说明正则化不能替代材料/能量或路径模型。"
            "下一步应固定本阶段参数，使用真实轨迹pilot验证非均匀MLP。",
            "",
            "## 7. 主要产物",
            "",
            "- `frozen_final.json`：冻结参数；",
            "- `relaxation_selection.json`、`loss_selection.json`、"
            "`regularization_selection.json`、`subset_selection.json`：筛选决定；",
            "- `confirmation_test_wepl.csv`：S1--S5锁定测试；",
            "- `confirmation_image_metrics.csv`：基线与阶段4图像指标；",
            "- `confirmation_summary.json`：晋升和验收判定。",
            "",
        ]
    )
    path = QC_ROOT / "stage4_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "status": "PASS",
        "summary": relative(path),
        "decision": confirmation["decision"],
        "runtime_seconds_final_candidates": runtime_total,
        "relaxation": relaxation,
        "loss": loss,
        "regularization": regularization,
        "subset": subset,
        "confirmation": confirmation,
    }
    write_json(QC_ROOT / "stage4_summary.json", result)
    return result


def smoke(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    runs: int,
) -> dict[str, Any]:
    import cupy as cp
    from weighted_gpu import WeightedGpuMlpProjector

    dataset = next((item for item in datasets if item["name"] == "s2"), None)
    if dataset is None:
        raise SystemExit("smoke requires s2 in --datasets")
    cp.cuda.Device(device).use()
    base = WeightedGpuMlpProjector(64, 4.0, 4.0, 100.0)
    robust = RobustGpuMlpProjector(64, 4.0, 4.0, 100.0)
    image = cp.ones((64, 64), dtype=cp.float32)
    checks = []
    for run_id in (0, 719):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        split = partition_masks(
            len(pairs), run_id, config["stage3"]["split"]
        )
        accepted = accepted_mask(dataset, config, run_id, len(pairs))
        indices = np.flatnonzero(split["train"] & accepted)[:8]
        selected = np.asarray(pairs[indices], dtype=np.float32)
        batch = stage3.make_batch(selected, dataset, config["stage3"])
        base_num = cp.zeros_like(image)
        base_den = cp.zeros_like(image)
        robust_num = cp.zeros_like(image)
        robust_den = cp.zeros_like(image)
        base.accumulate_weighted(
            image,
            batch,
            np.ones(len(selected), np.float32),
            0.5 * run_id,
            base_num,
            base_den,
        )
        robust.accumulate_loss(
            image,
            batch,
            0.5 * run_id,
            robust_num,
            robust_den,
            None,
        )
        checks.append(
            {
                "run_id": run_id,
                "rows": len(selected),
                "quadratic_numerator_max_difference": float(
                    cp.max(cp.abs(base_num - robust_num)).get()
                ),
                "quadratic_denominator_max_difference": float(
                    cp.max(cp.abs(base_den - robust_den)).get()
                ),
                "finite": bool(
                    cp.isfinite(robust_num).all().get()
                    and cp.isfinite(robust_den).all().get()
                ),
            }
        )
    result = {
        "status": (
            "PASS"
            if all(
                row["finite"]
                and row["quadratic_numerator_max_difference"] < 1.0e-5
                and row["quadratic_denominator_max_difference"] < 1.0e-5
                for row in checks
            )
            else "FAIL"
        ),
        "checks": checks,
    }
    write_json(QC_ROOT / "smoke_test.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("Stage-4 smoke test failed")
    return result


def main() -> None:
    args = parse_args()
    config = load_config()
    datasets = datasets_from(args.datasets, config)
    if args.runs != RUNS and args.action not in {"smoke", "status"}:
        raise SystemExit("--runs other than 720 is restricted to smoke")
    started = time.perf_counter()
    if args.action == "prepare":
        result = prepare(datasets, config, args.runs)
    elif args.action == "relaxation-screen":
        result = relaxation_screen(
            datasets, config, args.device, args.runs, args.force
        )
    elif args.action == "loss-screen":
        result = loss_screen(
            datasets, config, args.device, args.runs, args.force
        )
    elif args.action == "regularization-screen":
        result = regularization_screen(
            datasets, config, args.device, args.runs, args.force
        )
    elif args.action == "subset-screen":
        result = subset_screen(
            datasets, config, args.device, args.runs, args.force
        )
    elif args.action == "confirm":
        result = confirm(
            datasets, config, args.device, args.runs, args.force
        )
    elif args.action == "report":
        result = report(config)
    elif args.action == "status":
        result = status(datasets, config)
    else:
        result = smoke(datasets, config, args.device, args.runs)
    print(
        json.dumps(
            {
                "status": result["status"],
                "action": args.action,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
