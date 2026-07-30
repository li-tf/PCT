#!/usr/bin/env python3
"""Run the complete Stage-6 advanced-prior study with one resumable command."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
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
STAGE4_ROOT = HERE.parent / "stage4_iterative_optimization"
CONFIG_PATH = HERE / "stage6_config.json"
QC_ROOT = HERE / "qc"
PROGRESS_PATH = QC_ROOT / "progress.json"
RUNS = 720

sys.path[:0] = [
    str(HERE),
    str(STAGE4_ROOT),
    str(STAGE3_ROOT),
    str(CODE_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

import run_stage3 as stage3  # noqa: E402
import run_stage4 as stage4  # noqa: E402
from advanced_priors import (  # noqa: E402
    forward_gradient,
    negative_gradient_adjoint,
    negative_symmetric_gradient_adjoint,
    proximal_advanced,
    symmetric_gradient,
)
from robust_gpu import RobustGpuMlpProjector  # noqa: E402
from stage3_io import (  # noqa: E402
    format_duration,
    load_json,
    partition_masks,
    read_packed_mask,
    relative,
    sha256,
    write_json,
)
from preprocessing import paircuts  # noqa: E402
from iterative_reconstruction.mhd_io import (  # noqa: E402
    read_image_2d,
    write_image_2d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("all", "status", "smoke", "report"),
        required=True,
    )
    parser.add_argument(
        "--datasets",
        default="s1,s2,s3,s4,s5",
        help="formal all requires exactly s1,s2,s3,s4,s5",
    )
    parser.add_argument("--jobs", type=int, default=4, help="reserved I/O worker count")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS,
        help="only smoke may use fewer than 720 angles",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and recalculate Stage-6 outputs only",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    stage4_path = (HERE / config["stage4_config"]).resolve()
    config["stage4"] = stage4.load_json(stage4_path)
    stage3_path = (
        STAGE4_ROOT / config["stage4"]["stage3_config"]
    ).resolve()
    config["stage4"]["stage3"] = stage4.load_json(stage3_path)
    return config


def ensure_snapshot(config: dict[str, Any]) -> None:
    """Refuse to resume if Stage-6 code or configuration changed."""

    path = QC_ROOT / "run_snapshot.json"
    value = {
        "config_hash": stage4.canonical_hash(config),
        "source_hashes": {
            name: sha256(HERE / name)
            for name in (
                "run_stage6.py",
                "advanced_priors.py",
                "stage6_config.json",
            )
        },
        "stage4_frozen_hash": sha256(
            STAGE4_ROOT / "qc" / "frozen_final.json"
        ),
    }
    if path.is_file():
        previous = load_json(path)
        if {
            key: previous[key]
            for key in ("config_hash", "source_hashes", "stage4_frozen_hash")
        } != value:
            raise RuntimeError(
                "Stage-6 code/configuration hash changed; use --force only "
                "if the existing Stage-6 study should be discarded"
            )
    else:
        write_json(
            path,
            {
                **value,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
        )


def datasets_from(text: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return stage4.datasets_from(text, config["stage4"])


def require_formal_datasets(datasets: list[dict[str, Any]]) -> None:
    actual = {item["name"] for item in datasets}
    expected = {"s1", "s2", "s3", "s4", "s5"}
    if actual != expected:
        raise SystemExit("--action all requires --datasets s1,s2,s3,s4,s5")


def stage6_root(dataset: dict[str, Any]) -> Path:
    return Path(dataset["reconstruction_data"]) / "stage6"


def stage4_settings() -> dict[str, Any]:
    path = STAGE4_ROOT / "qc" / "frozen_final.json"
    if not path.is_file():
        raise FileNotFoundError("Stage-4 frozen_final.json is required")
    frozen = load_json(path)
    return {
        key: value
        for key, value in frozen.items()
        if key not in {"frozen_at", "test_partition_opened"}
    }


def stage4_image(dataset: dict[str, Any], *, regularized: bool = True) -> Path:
    settings = stage4_settings()
    if not regularized:
        settings["regularization_weight"] = 0.0
        settings["regularization_schedule"] = "fixed"
    name = stage4.variant_name(settings)
    path = (
        stage4.stage4_root(dataset)
        / "variants"
        / name
        / "recon"
        / f"epoch_{int(settings['epochs']):02d}.mhd"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def support_for(image: np.ndarray, spacing: float, origin: float, radius: float):
    coordinates = origin + np.arange(image.shape[0], dtype=np.float32) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    return xx * xx + zz * zz <= radius * radius


def candidate_name(candidate: dict[str, Any]) -> str:
    method = str(candidate["method"])
    weight = f"{float(candidate['weight']):g}".replace(".", "p")
    if method == "tgv":
        suffix = f"r{float(candidate['second_order_ratio']):g}"
    else:
        suffix = f"m{float(candidate['minimum_weight']):g}"
    return f"{method}_b{weight}_{suffix}".replace(".", "p")


def enumerate_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    screen = config["prescreen"]
    rows: list[dict[str, Any]] = []
    for weight in screen["tgv"]["weights"]:
        for ratio in screen["tgv"]["second_order_ratios"]:
            rows.append(
                {
                    "method": "tgv",
                    "weight": float(weight),
                    "second_order_ratio": float(ratio),
                    "minimum_weight": 1.0,
                }
            )
    for method in ("adaptive_tv", "directional_tv"):
        for weight in screen[method]["weights"]:
            for minimum in screen[method]["minimum_weights"]:
                rows.append(
                    {
                        "method": method,
                        "weight": float(weight),
                        "second_order_ratio": 1.0,
                        "minimum_weight": float(minimum),
                    }
                )
    for row in rows:
        row["name"] = candidate_name(row)
    return rows


def update_progress(
    *,
    status: str,
    phase: str,
    task: str,
    fraction: float,
    started: float,
    **extra: Any,
) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed * (1.0 - fraction) / fraction if fraction > 0 else None
    atomic_json(
        PROGRESS_PATH,
        {
            "status": status,
            "phase": phase,
            "task": task,
            "overall_fraction": float(min(max(fraction, 0.0), 1.0)),
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": eta,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            **extra,
        },
    )


def stage_fraction(config: dict[str, Any], phase: str, local: float) -> float:
    weights = config["runtime"]["progress_phase_weights"]
    order = ("smoke", "prescreen", "validation", "confirmation", "report")
    start = sum(float(weights[name]) for name in order[: order.index(phase)])
    return start + float(weights[phase]) * min(max(local, 0.0), 1.0)


def prior_parameters(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    prox = config["proximal"]
    method = str(candidate["method"])
    return {
        "method": method,
        "weight": float(candidate["weight"]),
        "iterations": int(prox["iterations"]),
        "huber_delta": float(prox["huber_delta"]),
        "primal_step": float(
            prox["tgv_primal_step"] if method == "tgv" else prox["tv_primal_step"]
        ),
        "dual_step": float(
            prox["tgv_dual_step"] if method == "tgv" else prox["tv_dual_step"]
        ),
        "second_order_ratio": float(candidate.get("second_order_ratio", 1.0)),
        "minimum_weight": float(candidate.get("minimum_weight", 1.0)),
        "smoothing_iterations": int(prox["smoothing_iterations"]),
        "kappa_quantile": float(prox["kappa_quantile"]),
    }


def metric_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def apply_prescreen_candidate(
    dataset: dict[str, Any],
    config: dict[str, Any],
    candidate: dict[str, Any],
    device: int,
) -> dict[str, Any]:
    import cupy as cp

    output_root = stage6_root(dataset) / "prescreen" / candidate["name"]
    output = output_root / "recon.mhd"
    summary_path = output_root / "metrics.json"
    if summary_path.is_file() and output.is_file():
        return load_json(summary_path)
    source, spacing3, origin3 = read_image_2d(
        stage4_image(dataset, regularized=False)
    )
    source = np.array(source, dtype=np.float32, copy=True)
    guidance = np.array(
        read_image_2d(stage4.initial_path(dataset))[0],
        dtype=np.float32,
        copy=True,
    )
    grid = config["stage4"]["grid"]
    support = support_for(
        source, float(spacing3[0]), float(origin3[0]), float(grid["phantom_radius_mm"])
    )
    cp.cuda.Device(device).use()
    result, prior_metrics = proximal_advanced(
        cp.asarray(source),
        cp.asarray(support),
        reference_guidance=cp.asarray(guidance),
        **prior_parameters(config, candidate),
    )
    host = cp.asnumpy(result)
    if not np.isfinite(host).all() or np.count_nonzero(host[~support]):
        raise RuntimeError(f"invalid prescreen image: {dataset['name']} {candidate['name']}")
    write_image_2d(output, host, float(spacing3[0]), float(origin3[0]))
    measured, details = stage4.scalar_metrics(
        dataset, config["stage4"], output
    )
    if details:
        write_csv(output_root / "details.csv", details)
    summary = {
        "status": "PASS",
        "dataset": dataset["name"],
        "candidate": candidate["name"],
        "image_path": relative(output),
        **candidate,
        **prior_metrics,
        **measured,
    }
    write_json(summary_path, summary)
    return summary


def baseline_image_metrics(
    dataset: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    metrics, _ = stage4.scalar_metrics(
        dataset, config["stage4"], stage4_image(dataset)
    )
    return {"dataset": dataset["name"], **metrics}


def score_image_set(
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_dataset = {row["dataset"]: row for row in rows}
    s2, s4, s5 = by_dataset["s2"], by_dataset["s4"], by_dataset["s5"]
    b2, b4, b5 = baselines["s2"], baselines["s4"], baselines["s5"]
    improvements = {
        "s2_water_std": 1.0
        - metric_value(s2, "water_core_std_rsp")
        / metric_value(b2, "water_core_std_rsp"),
        "s4_material_mape": 1.0
        - metric_value(s4, "material_mape_non_air")
        / metric_value(b4, "material_mape_non_air"),
        "s5_rsp_rmse": 1.0
        - metric_value(s5, "phantom_rmse_vs_nominal_rsp")
        / metric_value(b5, "phantom_rmse_vs_nominal_rsp"),
    }
    selection = config["selection"]
    safety = {
        "s2_water_bias": abs(metric_value(s2, "water_bias_vs_effective_rsp"))
        - abs(metric_value(b2, "water_bias_vs_effective_rsp"))
        <= selection["water_bias_max_degradation_percentage_points"] / 100.0,
        "s4_material_mape": (
            metric_value(s4, "material_mape_non_air")
            - metric_value(b4, "material_mape_non_air")
        )
        * 100.0
        <= selection["material_mape_max_degradation_percentage_points"],
        "s5_mtf50": metric_value(s5, "fmtf50_mean_lp_per_mm")
        >= metric_value(b5, "fmtf50_mean_lp_per_mm")
        * (1.0 - selection["mtf_max_relative_degradation"]),
        "s5_mtf10": metric_value(s5, "fmtf10_mean_lp_per_mm")
        >= metric_value(b5, "fmtf10_mean_lp_per_mm")
        * (1.0 - selection["mtf_max_relative_degradation"]),
    }
    finite = all(math.isfinite(value) for value in improvements.values())
    return {
        "score": float(np.mean(list(improvements.values()))) if finite else -math.inf,
        "safe": bool(finite and all(safety.values())),
        "improvements": improvements,
        "safety": safety,
    }


def prescreen(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    started: float,
) -> dict[str, Any]:
    selection_path = QC_ROOT / "prescreen_selection.json"
    if selection_path.is_file():
        return load_json(selection_path)
    selected = {item["name"]: item for item in datasets}
    screen_names = list(config["prescreen"]["datasets"])
    baselines = {
        name: baseline_image_metrics(selected[name], config)
        for name in screen_names
    }
    candidates = enumerate_candidates(config)
    rows: list[dict[str, Any]] = []
    total = len(candidates) * len(screen_names)
    done = 0
    for candidate in candidates:
        for name in screen_names:
            row = apply_prescreen_candidate(
                selected[name], config, candidate, device
            )
            rows.append(row)
            done += 1
            update_progress(
                status="RUNNING",
                phase="prescreen",
                task=f"{candidate['name']} {name}",
                fraction=stage_fraction(config, "prescreen", done / total),
                started=started,
                completed=done,
                total=total,
            )
    scored = []
    for candidate in candidates:
        current = [
            row for row in rows if row["candidate"] == candidate["name"]
        ]
        score = score_image_set(current, baselines, config)
        scored.append({**candidate, **score})
    finalists = []
    for family in ("tgv", "adaptive_tv", "directional_tv"):
        safe = [
            row for row in scored if row["method"] == family and row["safe"]
        ]
        if safe:
            finalists.append(max(safe, key=lambda row: row["score"]))
    result = {
        "status": "PASS",
        "candidate_count": len(candidates),
        "finalists": finalists,
        "scores": scored,
        "test_partition_opened": False,
    }
    write_csv(QC_ROOT / "prescreen_metrics.csv", rows)
    write_json(selection_path, result)
    return result


def accepted_mask(
    dataset: dict[str, Any], config: dict[str, Any], run_id: int, count: int
) -> np.ndarray:
    return read_packed_mask(
        stage3.mask_path(dataset, "baseline_3sigma", run_id),
        count,
        config["stage4"]["stage3"]["split"]["bit_order"],
    )


def run_candidate(
    dataset: dict[str, Any],
    config: dict[str, Any],
    candidate: dict[str, Any],
    device: int,
    started: float,
    progress_start: float,
    progress_span: float,
) -> dict[str, Any]:
    import cupy as cp

    name = candidate["name"]
    output_root = stage6_root(dataset) / "variants" / name
    recon_dir = output_root / "recon"
    snapshot_path = output_root / "config.json"
    epochs = int(config["baseline"]["epochs"])
    invariant = {
        **candidate,
        "baseline": config["baseline"],
        "proximal": config["proximal"],
    }
    digest = stage4.canonical_hash(invariant)
    output_root.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_path.is_file():
        if load_json(snapshot_path)["invariant_hash"] != digest:
            raise RuntimeError(f"configuration hash mismatch: {output_root}")
    else:
        write_json(
            snapshot_path,
            {
                "invariant_hash": digest,
                "settings": invariant,
                "stage6_config_hash": stage4.canonical_hash(
                    {key: value for key, value in config.items() if key != "stage4"}
                ),
            },
        )
    history_path = output_root / "epoch_metrics.csv"
    history = read_csv(history_path)
    completed = max((int(row["epoch"]) for row in history), default=0)
    if completed >= epochs:
        return history[epochs - 1]

    source = (
        recon_dir / f"epoch_{completed:02d}.mhd"
        if completed
        else stage4.initial_path(dataset)
    )
    image_cpu, spacing3, origin3 = read_image_2d(source)
    image_cpu = np.array(image_cpu, dtype=np.float32, copy=True)
    guidance_cpu = np.array(
        read_image_2d(stage4.initial_path(dataset))[0],
        dtype=np.float32,
        copy=True,
    )
    grid = config["stage4"]["grid"]
    spacing = float(spacing3[0])
    origin = float(origin3[0])
    support_cpu = support_for(
        image_cpu, spacing, origin, float(grid["phantom_radius_mm"])
    )
    np.maximum(image_cpu, 0.0, out=image_cpu)
    image_cpu[~support_cpu] = 0.0
    if not completed:
        write_image_2d(recon_dir / "initial.mhd", image_cpu, spacing, origin)

    counts = []
    for run_id in range(RUNS):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"]) / "pairs" / f"pairs{run_id:04d}.mhd"
        )
        split = partition_masks(
            len(pairs), run_id, config["stage4"]["stage3"]["split"]
        )
        accepted = accepted_mask(dataset, config, run_id, len(pairs))
        counts.append(int(np.count_nonzero(split["train"] & accepted)))
    pairs_per_epoch = sum(counts)
    cp.cuda.Device(device).use()
    image = cp.asarray(image_cpu)
    guidance = cp.asarray(guidance_cpu)
    support = cp.asarray(support_cpu)
    projector = RobustGpuMlpProjector(
        int(grid["size"]),
        float(grid["spacing_mm"]),
        float(grid["path_step_mm"]),
        float(grid["phantom_radius_mm"]),
    )
    call_start = time.perf_counter()
    processed = 0
    subsets = int(config["baseline"]["subsets"])
    for epoch_index in range(completed, epochs):
        epoch_start = time.perf_counter()
        relaxation = float(config["baseline"]["relaxation"]) / (
            1.0
            + float(config["baseline"]["relaxation_decay"]) * epoch_index
        )
        training = {"squared": 0.0, "absolute": 0.0, "signed": 0.0, "valid": 0}
        max_update = 0.0
        for subset in range(subsets):
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            for run_id in range(subset, RUNS, subsets):
                pairs = paircuts.read_mhd(
                    Path(dataset["preprocessing_data"])
                    / "pairs"
                    / f"pairs{run_id:04d}.mhd"
                )
                split = partition_masks(
                    len(pairs), run_id, config["stage4"]["stage3"]["split"]
                )
                accepted = accepted_mask(dataset, config, run_id, len(pairs))
                indices = np.flatnonzero(split["train"] & accepted)
                batch_size = int(grid["batch_size"])
                for begin in range(0, len(indices), batch_size):
                    selected = np.asarray(
                        pairs[indices[begin : begin + batch_size]], dtype=np.float32
                    )
                    values = projector.accumulate_loss(
                        image,
                        stage3.make_batch(
                            selected, dataset, config["stage4"]["stage3"]
                        ),
                        0.5 * run_id,
                        numerator,
                        denominator,
                        None,
                    )
                    for key in training:
                        training[key] += values[key]
                    processed += len(selected)
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
            local = (
                epoch_index + (subset + 1) / subsets
            ) / epochs
            elapsed = time.perf_counter() - call_start
            rate = processed / elapsed if elapsed else 0.0
            remaining_pairs = (
                (epochs - epoch_index - 1) * pairs_per_epoch
                + pairs_per_epoch * (1.0 - (subset + 1) / subsets)
            )
            eta = remaining_pairs / rate if rate else math.nan
            overall = progress_start + progress_span * local
            update_progress(
                status="RUNNING",
                phase="reconstruction",
                task=f"{dataset['name']} {name}",
                fraction=overall,
                started=started,
                dataset=dataset["name"],
                candidate=name,
                epoch=epoch_index + 1,
                epochs=epochs,
                subset=subset + 1,
                subsets=subsets,
                pairs_per_second=rate,
                current_task_eta_seconds=eta if math.isfinite(eta) else None,
            )
            print(
                f"{dataset['name']} {name} epoch {epoch_index+1}/{epochs} "
                f"subset {subset+1:02d}/{subsets}: rate={rate:,.0f} pairs/s "
                f"ETA={format_duration(eta)}",
                flush=True,
            )
            del numerator, denominator, update
        image, prior_metrics = proximal_advanced(
            image,
            support,
            reference_guidance=guidance,
            **prior_parameters(config, candidate),
        )
        checkpoint = recon_dir / f"epoch_{epoch_index+1:02d}.mhd"
        host = cp.asnumpy(image)
        if not np.isfinite(host).all() or np.count_nonzero(host[~support_cpu]):
            raise RuntimeError(f"invalid image: {dataset['name']} {name}")
        write_image_2d(checkpoint, host, spacing, origin)
        validation = stage4.evaluate_partition(
            dataset,
            config["stage4"],
            checkpoint,
            "validation",
            device,
            RUNS,
        )
        measured, details = stage4.scalar_metrics(
            dataset, config["stage4"], checkpoint
        )
        if details:
            write_csv(
                output_root / f"epoch_{epoch_index+1:02d}_details.csv", details
            )
        valid = max(int(training["valid"]), 1)
        row = {
            "dataset": dataset["name"],
            "candidate": name,
            "method": candidate["method"],
            "epoch": epoch_index + 1,
            "relaxation": relaxation,
            "training_rmse_mm": math.sqrt(training["squared"] / valid),
            "training_mae_mm": training["absolute"] / valid,
            "training_bias_mm": training["signed"] / valid,
            "validation_rmse_mm": validation["rmse_mm"],
            "validation_mae_mm": validation["mae_mm"],
            "validation_bias_mm": validation["bias_mm"],
            "validation_abs_p99_mm": validation["abs_p99_mm"],
            "max_update": max_update,
            "prior_seconds": prior_metrics["elapsed_seconds"],
            "prior_l2_change": prior_metrics["l2_change"],
            "prior_max_abs_change": prior_metrics["max_abs_change"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "image_path": relative(checkpoint),
            **measured,
        }
        history.append(row)
        write_csv(history_path, history)
        print(
            f"completed {dataset['name']} {name} epoch {epoch_index+1}: "
            f"validation RMSE={validation['rmse_mm']:.5f} mm, "
            f"time={format_duration(row['epoch_seconds'])}",
            flush=True,
        )
    write_json(
        output_root / "run_summary.json",
        {
            "status": "PASS",
            "dataset": dataset["name"],
            "candidate": candidate,
            "completed_epochs": epochs,
            "pairs_per_epoch": pairs_per_epoch,
            "latest_image": history[epochs - 1]["image_path"],
            "support_outside_nonzero": 0,
        },
    )
    return history[epochs - 1]


def stage4_final_row(dataset: dict[str, Any]) -> dict[str, Any]:
    settings = stage4_settings()
    name = stage4.variant_name(settings)
    rows = stage4.variant_rows(dataset, name)
    epoch = int(settings["epochs"])
    matching = [row for row in rows if int(row["epoch"]) == epoch]
    if not matching:
        raise RuntimeError(f"missing Stage-4 final metrics for {dataset['name']}")
    return matching[-1]


def full_validation(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    finalists: list[dict[str, Any]],
    device: int,
    started: float,
) -> dict[str, Any]:
    selection_path = QC_ROOT / "validation_selection.json"
    if selection_path.is_file():
        return load_json(selection_path)
    selected = {item["name"]: item for item in datasets}
    names = list(config["prescreen"]["datasets"])
    total = max(1, len(finalists) * len(names))
    rows = []
    for index, candidate in enumerate(finalists):
        for offset, name in enumerate(names):
            task_index = index * len(names) + offset
            phase_start = stage_fraction(
                config, "validation", task_index / total
            )
            phase_end = stage_fraction(
                config, "validation", (task_index + 1) / total
            )
            rows.append(
                run_candidate(
                    selected[name],
                    config,
                    candidate,
                    device,
                    started,
                    phase_start,
                    phase_end - phase_start,
                )
            )
    write_csv(QC_ROOT / "validation_metrics.csv", rows)
    baselines = {name: stage4_final_row(selected[name]) for name in names}
    decisions = []
    gate = config["selection"]
    for candidate in finalists:
        current = {
            row["dataset"]: row
            for row in rows
            if row["candidate"] == candidate["name"]
        }
        if set(current) != set(names):
            continue
        image_score = score_image_set(list(current.values()), baselines, config)
        wepl_safe = all(
            metric_value(current[name], "validation_rmse_mm")
            <= metric_value(baselines[name], "validation_rmse_mm")
            * (1.0 + gate["validation_rmse_max_degradation"])
            for name in names
        )
        b4, c4 = baselines["s4"], current["s4"]
        improvements = {
            **image_score["improvements"],
            "s4_material_max_ape": 1.0
            - metric_value(c4, "material_max_ape_non_air")
            / metric_value(b4, "material_max_ape_non_air"),
        }
        substantive = {
            "water_std": improvements["s2_water_std"]
            >= gate["water_std_min_improvement"],
            "material_mape": (
                metric_value(b4, "material_mape_non_air")
                - metric_value(c4, "material_mape_non_air")
            )
            * 100.0
            >= gate["material_mape_min_improvement_percentage_points"],
            "material_max_ape": improvements["s4_material_max_ape"]
            >= gate["material_max_ape_min_improvement"],
            "s5_rsp_rmse": improvements["s5_rsp_rmse"]
            >= gate["s5_rsp_rmse_min_improvement"],
        }
        eligible = bool(
            wepl_safe and image_score["safe"] and any(substantive.values())
        )
        decisions.append(
            {
                **candidate,
                "score": image_score["score"],
                "wepl_safe": wepl_safe,
                "image_safety": image_score["safety"],
                "improvements": improvements,
                "substantive": substantive,
                "eligible": eligible,
            }
        )
    eligible = [row for row in decisions if row["eligible"]]
    winner = max(eligible, key=lambda row: row["score"]) if eligible else None
    result = {
        "status": "PASS",
        "decision": "CONTINUE_TO_CONFIRM" if winner else "RETAIN_STAGE4_VALIDATION_FAIL",
        "winner": winner,
        "candidates": decisions,
        "test_partition_opened": False,
    }
    write_json(selection_path, result)
    if winner:
        write_json(
            QC_ROOT / "frozen_candidate.json",
            {
                "candidate": {
                    key: winner[key]
                    for key in (
                        "name",
                        "method",
                        "weight",
                        "second_order_ratio",
                        "minimum_weight",
                    )
                },
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "test_partition_opened": False,
            },
        )
    return result


def confirm(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    winner: dict[str, Any],
    device: int,
    started: float,
) -> dict[str, Any]:
    decision_path = QC_ROOT / "confirmation_summary.json"
    if decision_path.is_file():
        return load_json(decision_path)
    frozen_path = QC_ROOT / "frozen_candidate.json"
    if not frozen_path.is_file():
        raise RuntimeError("candidate must be frozen before test confirmation")
    candidate = {
        key: winner[key]
        for key in (
            "name",
            "method",
            "weight",
            "second_order_ratio",
            "minimum_weight",
        )
    }
    total = len(datasets)
    candidate_paths: dict[str, Path] = {}
    for index, dataset in enumerate(datasets):
        phase_start = stage_fraction(config, "confirmation", index / total)
        phase_end = stage_fraction(config, "confirmation", (index + 1) / total)
        row = run_candidate(
            dataset,
            config,
            candidate,
            device,
            started,
            phase_start,
            phase_end - phase_start,
        )
        candidate_paths[dataset["name"]] = REPOSITORY_ROOT / row["image_path"]

    wepl_rows = []
    image_rows = []
    for index, dataset in enumerate(datasets):
        for method, path in (
            ("stage4", stage4_image(dataset)),
            ("stage6", candidate_paths[dataset["name"]]),
        ):
            test = stage4.evaluate_partition(
                dataset, config["stage4"], path, "test", device, RUNS
            )
            wepl_rows.append({"method": method, **test})
            metrics, _ = stage4.scalar_metrics(dataset, config["stage4"], path)
            image_rows.append(
                {
                    "dataset": dataset["name"],
                    "method": method,
                    "image_path": relative(path),
                    **metrics,
                }
            )
        update_progress(
            status="RUNNING",
            phase="confirmation",
            task=f"test metrics {dataset['name']}",
            fraction=stage_fraction(
                config, "confirmation", (index + 1) / total
            ),
            started=started,
        )
    write_csv(QC_ROOT / "confirmation_test_wepl.csv", wepl_rows)
    write_csv(QC_ROOT / "confirmation_image_metrics.csv", image_rows)
    by_wepl = {
        (row["dataset"], row["method"]): row for row in wepl_rows
    }
    by_image = {
        (row["dataset"], row["method"]): row for row in image_rows
    }
    gate = config["selection"]
    individual_wepl = all(
        metric_value(by_wepl[(name, "stage6")], "rmse_mm")
        <= metric_value(by_wepl[(name, "stage4")], "rmse_mm")
        * (1.0 + gate["validation_rmse_max_degradation"])
        for name in ("s1", "s2", "s3", "s4", "s5")
    )
    water_bias = all(
        abs(metric_value(by_image[(name, "stage6")], "water_bias_vs_effective_rsp"))
        - abs(metric_value(by_image[(name, "stage4")], "water_bias_vs_effective_rsp"))
        <= gate["water_bias_max_degradation_percentage_points"] / 100.0
        for name in ("s2", "s3")
    )
    material_mape_pp = (
        metric_value(by_image[("s4", "stage4")], "material_mape_non_air")
        - metric_value(by_image[("s4", "stage6")], "material_mape_non_air")
    ) * 100.0
    material_max_improvement = 1.0 - (
        metric_value(by_image[("s4", "stage6")], "material_max_ape_non_air")
        / metric_value(by_image[("s4", "stage4")], "material_max_ape_non_air")
    )
    water_std_improvement = 1.0 - np.mean(
        [
            metric_value(by_image[(name, "stage6")], "water_core_std_rsp")
            for name in ("s2", "s3")
        ]
    ) / np.mean(
        [
            metric_value(by_image[(name, "stage4")], "water_core_std_rsp")
            for name in ("s2", "s3")
        ]
    )
    s5_rsp_improvement = 1.0 - (
        metric_value(by_image[("s5", "stage6")], "phantom_rmse_vs_nominal_rsp")
        / metric_value(by_image[("s5", "stage4")], "phantom_rmse_vs_nominal_rsp")
    )
    safety = {
        "individual_wepl": individual_wepl,
        "water_bias": water_bias,
        "s4_material_mape": material_mape_pp
        >= -gate["material_mape_max_degradation_percentage_points"],
        "s5_mtf50": metric_value(
            by_image[("s5", "stage6")], "fmtf50_mean_lp_per_mm"
        )
        >= metric_value(by_image[("s5", "stage4")], "fmtf50_mean_lp_per_mm")
        * (1.0 - gate["mtf_max_relative_degradation"]),
        "s5_mtf10": metric_value(
            by_image[("s5", "stage6")], "fmtf10_mean_lp_per_mm"
        )
        >= metric_value(by_image[("s5", "stage4")], "fmtf10_mean_lp_per_mm")
        * (1.0 - gate["mtf_max_relative_degradation"]),
    }
    substantive = {
        "water_std": water_std_improvement >= gate["water_std_min_improvement"],
        "material_mape": material_mape_pp
        >= gate["material_mape_min_improvement_percentage_points"],
        "material_max_ape": material_max_improvement
        >= gate["material_max_ape_min_improvement"],
        "s5_rsp_rmse": s5_rsp_improvement
        >= gate["s5_rsp_rmse_min_improvement"],
    }
    promoted = bool(all(safety.values()) and any(substantive.values()))
    result = {
        "status": "PASS",
        "decision": "PROMOTE_STAGE6" if promoted else "RETAIN_STAGE4",
        "promoted": promoted,
        "winner": candidate,
        "safety_checks": safety,
        "substantive_improvements": substantive,
        "water_std_improvement": float(water_std_improvement),
        "material_mape_improvement_percentage_points": float(material_mape_pp),
        "material_max_ape_improvement": float(material_max_improvement),
        "s5_rsp_rmse_improvement": float(s5_rsp_improvement),
        "test_partition_opened": True,
    }
    write_json(decision_path, result)
    frozen = load_json(frozen_path)
    frozen["test_partition_opened"] = True
    write_json(frozen_path, frozen)
    return result


def smoke(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
) -> dict[str, Any]:
    import cupy as cp

    cp.cuda.Device(device).use()
    size = 64
    coords = cp.arange(size, dtype=cp.float32) - size / 2
    xx, zz = cp.meshgrid(coords, coords)
    support = xx * xx + zz * zz <= 28.0**2
    image = cp.where(support, 1.0 + 0.05 * cp.sin(xx / 3.0), 0.0).astype(
        cp.float32
    )
    checks = []
    examples = (
        {"method": "tgv", "weight": 0.006, "second_order_ratio": 2.0, "minimum_weight": 1.0},
        {"method": "adaptive_tv", "weight": 0.006, "second_order_ratio": 1.0, "minimum_weight": 0.3},
        {"method": "directional_tv", "weight": 0.006, "second_order_ratio": 1.0, "minimum_weight": 0.4},
    )
    for item in examples:
        result, metrics = proximal_advanced(
            image.copy(),
            support,
            reference_guidance=image,
            **prior_parameters(config, item),
        )
        checks.append(
            {
                "method": item["method"],
                "finite": bool(cp.isfinite(result).all().get()),
                "outside_nonzero": int(cp.count_nonzero(result[~support]).get()),
                "max_abs_change": metrics["max_abs_change"],
            }
        )
    dataset = next((row for row in datasets if row["name"] == "s2"), None)
    real_checks = []
    if dataset is not None:
        projector = RobustGpuMlpProjector(64, 4.0, 4.0, 100.0)
        for run_id in (0, 719):
            pairs = paircuts.read_mhd(
                Path(dataset["preprocessing_data"])
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            split = partition_masks(
                len(pairs), run_id, config["stage4"]["stage3"]["split"]
            )
            accepted = accepted_mask(dataset, config, run_id, len(pairs))
            indices = np.flatnonzero(split["train"] & accepted)[:8]
            selected = np.asarray(pairs[indices], dtype=np.float32)
            numerator = cp.zeros((64, 64), cp.float32)
            denominator = cp.zeros_like(numerator)
            values = projector.accumulate_loss(
                cp.ones_like(numerator),
                stage3.make_batch(
                    selected, dataset, config["stage4"]["stage3"]
                ),
                0.5 * run_id,
                numerator,
                denominator,
                None,
            )
            real_checks.append(
                {
                    "run_id": run_id,
                    "rows": len(selected),
                    "valid": values["valid"],
                    "finite": bool(
                        cp.isfinite(numerator).all().get()
                        and cp.isfinite(denominator).all().get()
                    ),
                }
            )
    passed = all(
        row["finite"] and row["outside_nonzero"] == 0 for row in checks
    ) and all(row["finite"] and row["valid"] > 0 for row in real_checks)
    result = {
        "status": "PASS" if passed else "FAIL",
        "prior_checks": checks,
        "real_data_checks": real_checks,
    }
    write_json(QC_ROOT / "smoke.json", result)
    if not passed:
        raise RuntimeError("Stage-6 smoke test failed")
    return result


def build_report(
    config: dict[str, Any],
    validation: dict[str, Any] | None = None,
    confirmation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if validation is None:
        validation_path = QC_ROOT / "validation_selection.json"
        validation = load_json(validation_path) if validation_path.is_file() else {
            "decision": "NOT_RUN",
            "winner": None,
            "candidates": [],
        }
    if confirmation is None:
        confirmation_path = QC_ROOT / "confirmation_summary.json"
        confirmation = (
            load_json(confirmation_path)
            if confirmation_path.is_file()
            else None
        )
    final_decision = (
        confirmation["decision"]
        if confirmation
        else validation.get("decision", "NOT_RUN")
    )
    finalist_lines = []
    for row in validation.get("candidates", []):
        finalist_lines.append(
            f"| {row['name']} | {row['method']} | "
            f"{metric_value(row, 'score'):.4f} | "
            f"{'是' if row.get('wepl_safe') else '否'} | "
            f"{'是' if row.get('eligible') else '否'} |"
        )
    if not finalist_lines:
        finalist_lines.append("| 无 | — | — | — | — |")
    confirmation_text = (
        "验证阶段没有候选满足安全与实质改善门槛，因此未打开锁定测试。"
        if confirmation is None
        else (
            f"锁定测试决定为`{confirmation['decision']}`。水区标准差变化为"
            f"`{100.0*confirmation['water_std_improvement']:+.2f}%`，S4材料"
            f"MAPE改善为`{confirmation['material_mape_improvement_percentage_points']:+.3f}`"
            "个百分点，S5 RSP RMSE变化为"
            f"`{100.0*confirmation['s5_rsp_rmse_improvement']:+.2f}%`。"
        )
    )
    text = f"""# 阶段6：高级图像先验

## 执行状态

- 最终决定：`{final_decision}`
- 固定数据项：局部3σ、等权、quadratic；
- 固定路径：水Schulte MLP；
- 固定迭代：`λ0=0.25`、衰减0.2、18子集、5 epoch；
- 基线先验：阶段4固定`β=0.0125` Huber-TV。

## 方法

本阶段只改变图像域先验，比较二阶Huber-TGV、固定边缘自适应Huber-TV和
弱方向TV。候选先在S2/S4/S5的无正则化检查点上进行近端预筛，每个方法族最多
保留一个方案，再执行完整5 epoch验证重建。只有唯一候选冻结后才允许读取测试集。

## 完整验证候选

| 候选 | 方法 | 综合分数 | WEPL安全 | 可晋升 |
|---|---|---:|---|---|
{chr(10).join(finalist_lines)}

## 结论

{confirmation_text}

机器可读决定见`prescreen_selection.json`、`validation_selection.json`和
`confirmation_summary.json`（若打开锁定测试）。大型检查点位于各数据集
`data/reconstruction_data/.../stage6/`。
"""
    path = QC_ROOT / "stage6_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    result = {
        "status": "PASS",
        "decision": final_decision,
        "summary_path": relative(path),
    }
    write_json(QC_ROOT / "stage6_decision.json", result)
    return result


def status() -> dict[str, Any]:
    if not PROGRESS_PATH.is_file():
        print("Stage 6 has not started.")
        return {"status": "NOT_STARTED", "read_only": True}
    progress = load_json(PROGRESS_PATH)
    fraction = float(progress.get("overall_fraction", 0.0))
    print("Stage 6 aggregate progress (read-only)")
    print("=" * 72)
    print(f"status:  {progress.get('status')}")
    print(f"phase:   {progress.get('phase')}")
    print(f"task:    {progress.get('task')}")
    print(f"overall: {100.0*fraction:.1f}%")
    if "dataset" in progress:
        print(
            f"current: {progress.get('dataset')} / "
            f"{progress.get('candidate')} / epoch "
            f"{progress.get('epoch')}/{progress.get('epochs')} / subset "
            f"{progress.get('subset')}/{progress.get('subsets')}"
        )
    if progress.get("pairs_per_second") is not None:
        print(f"rate:    {float(progress['pairs_per_second']):,.0f} pairs/s")
    print(f"elapsed: {format_duration(float(progress.get('elapsed_seconds', 0.0)))}")
    eta = progress.get("estimated_remaining_seconds")
    print(f"ETA:     {format_duration(float(eta)) if eta is not None else 'unknown'}")
    current_eta = progress.get("current_task_eta_seconds")
    if current_eta is not None:
        print(f"task ETA:{format_duration(float(current_eta))}")
    decision_path = QC_ROOT / "stage6_decision.json"
    if decision_path.is_file():
        print(f"decision:{load_json(decision_path).get('decision')}")
    return {**progress, "read_only": True}


def reset_stage6(datasets: list[dict[str, Any]]) -> None:
    for dataset in datasets:
        root = stage6_root(dataset)
        if root.exists():
            shutil.rmtree(root)
    if QC_ROOT.exists():
        shutil.rmtree(QC_ROOT)


def run_all(
    datasets: list[dict[str, Any]],
    config: dict[str, Any],
    device: int,
    started: float,
) -> dict[str, Any]:
    require_formal_datasets(datasets)
    ensure_snapshot(config)
    update_progress(
        status="RUNNING",
        phase="smoke",
        task="operator and two-angle checks",
        fraction=0.0,
        started=started,
    )
    smoke(datasets, config, device)
    update_progress(
        status="RUNNING",
        phase="smoke",
        task="complete",
        fraction=stage_fraction(config, "smoke", 1.0),
        started=started,
    )
    pre = prescreen(datasets, config, device, started)
    finalists = pre["finalists"]
    if not finalists:
        validation = {
            "status": "PASS",
            "decision": "RETAIN_STAGE4_PRESCREEN_FAIL",
            "winner": None,
            "candidates": [],
            "test_partition_opened": False,
        }
        write_json(QC_ROOT / "validation_selection.json", validation)
        report = build_report(config, validation, None)
        update_progress(
            status="COMPLETE",
            phase="report",
            task="prescreen gate stopped workflow",
            fraction=1.0,
            started=started,
            final_decision=report["decision"],
        )
        return report
    validation = full_validation(
        datasets, config, finalists, device, started
    )
    confirmation = None
    if validation["winner"] is not None:
        confirmation = confirm(
            datasets, config, validation["winner"], device, started
        )
    report = build_report(config, validation, confirmation)
    update_progress(
        status="COMPLETE",
        phase="report",
        task="workflow complete",
        fraction=1.0,
        started=started,
        final_decision=report["decision"],
    )
    return report


def main() -> None:
    args = parse_args()
    config = load_config()
    datasets = datasets_from(args.datasets, config)
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    if args.runs != RUNS and args.action != "smoke":
        raise SystemExit("--runs other than 720 is restricted to smoke")
    started = time.perf_counter()
    if args.force:
        if args.action != "all":
            raise SystemExit("--force is only supported with --action all")
        reset_stage6(datasets)
    if args.action == "all":
        result = run_all(datasets, config, args.device, started)
    elif args.action == "status":
        result = status()
    elif args.action == "smoke":
        result = smoke(datasets, config, args.device)
    else:
        result = build_report(config)
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
