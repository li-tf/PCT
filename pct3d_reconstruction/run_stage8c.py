#!/usr/bin/env python3
"""Stage 8C guarded diagnostics, matched-model closure, and 3-D confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIG_PATH = HERE / "stage8c_config.json"
sys.path.insert(0, str(HERE))

from evaluation3d import image_metrics  # noqa: E402
from gpu_operator3d import GpuMlpProjector3D  # noqa: E402
from io3d import pair_batch, read_pairs, read_partition, read_volume, write_volume  # noqa: E402
from physics3d import (  # noqa: E402
    build_truth,
    coordinates,
    load_mlic,
    ray_finite_cylinder_interval,
    scanner_to_object,
    support_mask,
)
from reconstruction3d import evaluate_partition  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config() -> tuple[dict[str, Any], dict[str, Any]]:
    stage8c = read_json(CONFIG_PATH)
    stage8 = read_json(resolve(stage8c["stage8_config"]))
    stage8["_wepl_model"] = str(resolve(stage8["wepl_model"]))
    return stage8c, stage8


def paths(stage8c: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    qc = resolve(stage8c["stage8c_qc"])
    output = resolve(stage8c["stage8c_reconstruction"])
    preprocessing = resolve(stage8c["stage8_preprocessing"])
    stage8_output = resolve(stage8c["stage8_reconstruction"])
    return qc, output, preprocessing, stage8_output


def digest_sources(stage8c: dict[str, Any], stage8: dict[str, Any]) -> str:
    payload = {
        "stage8c": stage8c,
        "stage8": {key: value for key, value in stage8.items() if not key.startswith("_")},
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(HERE.glob("*.py"))
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def update_progress(qc: Path, **values: Any) -> None:
    path = qc / "progress.json"
    current = read_json(path) if path.is_file() else {}
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(path, current)


def require_stage8(stage8c: dict[str, Any]) -> None:
    qc = resolve(stage8c["stage8_qc"])
    required = {
        "preprocessing": qc / "preprocessing_summary.json",
        "operator smoke": qc / "operator_smoke.json",
        "Stage 8 decision": qc / "stage8_decision.json",
    }
    for label, path in required.items():
        if not path.is_file() or read_json(path).get("status") not in {"PASS", "PASS_WITH_FALLBACK"}:
            raise RuntimeError(f"Stage 8 prerequisite is incomplete: {label}")


def selection(preprocessing: Path, run_id: int, mode: str) -> np.ndarray:
    if mode == "screen":
        code = read_partition(preprocessing / "splits" / f"screen{run_id:04d}.npz")
        return np.flatnonzero(code == 1)
    code = read_partition(preprocessing / "splits" / f"split{run_id:04d}.npz")
    target = {"train": 0, "validation": 1, "test": 2}[mode]
    return np.flatnonzero(code == target)


def iter_batches(
    preprocessing: Path, run_id: int, mode: str, batch_size: int
) -> Iterator[dict[str, np.ndarray]]:
    pairs = read_pairs(preprocessing / "pairs" / f"pairs{run_id:04d}.mhd", mmap=True)
    indexes = selection(preprocessing, run_id, mode)
    for begin in range(0, len(indexes), batch_size):
        yield pair_batch(pairs, indexes[begin : begin + batch_size])


def sphere_core_masks(stage8: dict[str, Any], simulation: dict[str, Any]) -> dict[str, np.ndarray]:
    x, y, z = coordinates(stage8)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    result = {}
    for item in simulation["spheres"]:
        center = np.asarray(item["scanner_center_mm"], dtype=float)
        radius = max(float(item["diameter_mm"]) / 2.0 - 1.0, 0.5)
        result[item["name"]] = (
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
            <= radius**2
        )
    return result


def object_to_scanner(points: np.ndarray, angle_deg: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.array(points, copy=True)
    result[..., 0] = cosine * points[..., 0] + sine * points[..., 2]
    result[..., 2] = -sine * points[..., 0] + cosine * points[..., 2]
    return result


def diagnostic_coordinate_check(stage8c: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray([item["scanner_center_mm"] for item in simulation["spheres"]], dtype=float)
    rows = []
    maximum = 0.0
    for angle in stage8c["diagnostic_runs"]:
        scanner = object_to_scanner(points, float(angle))
        restored = scanner_to_object(scanner, float(angle))
        error = float(np.max(np.abs(restored - points)))
        maximum = max(maximum, error)
        rows.append({"angle_deg": angle, "roundtrip_max_abs_mm": error})
    return {"maximum_mm": maximum, "rows": rows}


def diagnostic_intersections(
    projector: GpuMlpProjector3D, stage8: dict[str, Any]
) -> dict[str, Any]:
    rng = np.random.default_rng(20260804)
    n = 4096
    position = np.column_stack(
        (rng.uniform(-65, 65, n), rng.uniform(-25, 25, n), np.full(n, -60.0))
    ).astype(np.float32)
    direction = np.column_stack(
        (rng.normal(0, 0.08, n), rng.normal(0, 0.03, n), np.ones(n))
    ).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    cpu_enter, cpu_leave, cpu_valid = ray_finite_cylinder_interval(
        position, direction, float(stage8["phantom_radius_mm"]),
        float(stage8["phantom_half_length_y_mm"]),
    )
    gpu_enter, gpu_leave, gpu_valid = projector.debug_cylinder_intervals(position, direction)
    same = cpu_valid == gpu_valid
    common = cpu_valid & gpu_valid
    difference = 0.0 if not np.any(common) else float(
        max(np.max(np.abs(cpu_enter[common] - gpu_enter[common])),
            np.max(np.abs(cpu_leave[common] - gpu_leave[common])))
    )
    return {
        "rays": n,
        "validity_mismatches": int(np.count_nonzero(~same)),
        "maximum_interval_difference_mm": difference,
        "valid_rays": int(np.count_nonzero(common)),
    }


def run_operator_checks(
    projector: GpuMlpProjector3D,
    preprocessing: Path,
    stage8c: dict[str, Any],
    stage8: dict[str, Any],
) -> dict[str, Any]:
    import cupy as cp

    rng = np.random.default_rng(20260804)
    shape = tuple(reversed(stage8["grid"]["size_xyz"]))
    ones = cp.ones(shape, cp.float32)
    random_volume = cp.asarray(rng.normal(size=shape).astype(np.float32))
    rows = []
    max_constant = max_adjoint = 0.0
    for run_id in stage8c["diagnostic_runs"]:
        indexes = selection(preprocessing, int(run_id), "screen")[:256]
        pairs = read_pairs(preprocessing / "pairs" / f"pairs{int(run_id):04d}.mhd", mmap=True)
        batch = pair_batch(pairs, indexes)
        bundle = projector.build_paths(batch, float(run_id) * float(stage8["angle_step_deg"]))
        predicted, valid = projector.predict_from_paths(ones, bundle)
        row_sum = bundle[2]
        mask = valid.astype(cp.bool_)
        relative = cp.max(cp.abs(predicted[mask] - row_sum[mask]) / cp.maximum(row_sum[mask], 1e-6))
        constant_error = float(relative.get())
        y = cp.asarray(rng.normal(size=len(indexes)).astype(np.float32))
        ax, valid_x = projector.predict_from_paths(random_volume, bundle)
        aty = projector.transpose(y, valid_x, bundle[:3])
        lhs = float(cp.sum(ax * y, dtype=cp.float64).get())
        rhs = float(cp.sum(random_volume * aty, dtype=cp.float64).get())
        adjoint = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
        max_constant, max_adjoint = max(max_constant, constant_error), max(max_adjoint, adjoint)
        rows.append({
            "run_id": int(run_id),
            "constant_projection_relative_error": constant_error,
            "adjoint_relative_error": adjoint,
            "valid_rays": int(cp.sum(valid).get()),
        })
    return {
        "constant_projection_max_relative_error": max_constant,
        "adjoint_max_relative_error": max_adjoint,
        "rows": rows,
    }


def coverage_and_model_checks(
    projector: GpuMlpProjector3D,
    preprocessing: Path,
    truth: np.ndarray,
    core_masks: dict[str, np.ndarray],
    stage8c: dict[str, Any],
    stage8: dict[str, Any],
    qc: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    import cupy as cp

    coverage = cp.zeros(tuple(reversed(stage8["grid"]["size_xyz"])), cp.float32)
    truth_gpu = cp.asarray(truth)
    variants = {"correct": [0.0, 0], "reverse": [0.0, 0], "none": [0.0, 0]}
    batch_size = int(stage8c["diagnostic_batch_size"])
    started = time.perf_counter()
    for run_id in range(int(stage8["runs"])):
        for batch in iter_batches(preprocessing, run_id, "screen", batch_size):
            for name, angle in (
                ("correct", run_id * float(stage8["angle_step_deg"])),
                ("reverse", -run_id * float(stage8["angle_step_deg"])),
                ("none", 0.0),
            ):
                bundle = projector.build_paths(batch, angle)
                prediction, valid = projector.predict_from_paths(truth_gpu, bundle)
                measured = cp.asarray(batch["wepl_mm"], dtype=cp.float32)
                values = (measured - prediction)[valid.astype(cp.bool_)]
                variants[name][0] += float(cp.sum(values * values, dtype=cp.float64).get())
                variants[name][1] += int(values.size)
                if name == "correct":
                    projector.accumulate_coverage_from_paths(bundle, coverage)
        if (run_id + 1) % 5 == 0 or run_id + 1 == int(stage8["runs"]):
            elapsed = time.perf_counter() - started
            update_progress(
                qc, stage="diagnose", group="coverage_and_model", completed_runs=run_id + 1,
                total_runs=int(stage8["runs"]), task_elapsed_seconds=elapsed,
                task_eta_seconds=elapsed / (run_id + 1) * (int(stage8["runs"]) - run_id - 1),
            )
    coverage_cpu = cp.asnumpy(coverage)
    write_volume(
        qc / "coverage.mhd", coverage_cpu,
        tuple(stage8["grid"]["spacing_xyz_mm"]), tuple(stage8["grid"]["origin_xyz_mm"]),
    )
    support = support_mask(stage8)
    positive = coverage_cpu[support & (coverage_cpu > 0)]
    water_median = float(np.median(positive)) if len(positive) else 0.0
    cores = []
    for name, mask in core_masks.items():
        values = coverage_cpu[mask]
        cores.append({
            "sphere": name,
            "voxels": int(values.size),
            "zero_voxels": int(np.count_nonzero(values <= 0)),
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "relative_to_support_median": float(np.median(values) / max(water_median, 1e-12)),
        })
    residuals = {
        name: {
            "rmse_mm": float(math.sqrt(squared / max(count, 1))),
            "measurements": int(count),
        }
        for name, (squared, count) in variants.items()
    }
    return {"core_coverage": cores, "rotation_residuals": residuals}, coverage_cpu


def diagnostic_plots(qc: Path, coverage: np.ndarray, checks: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    assets = qc / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = coverage.shape
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    images = [coverage[nz // 2], coverage[:, ny // 2], coverage[:, :, nx // 2]]
    titles = ["Axial coverage", "Central x-z coverage", "Central y-z coverage"]
    for axis, image, title in zip(axes, images, titles):
        shown = axis.imshow(np.log10(np.maximum(image, 1e-6)), cmap="viridis", origin="lower")
        axis.set_title(title)
        fig.colorbar(shown, ax=axis, shrink=0.78, label="log10(Aᵀ1)")
    fig.savefig(assets / "coverage_slices.png", dpi=180)
    plt.close(fig)
    residuals = checks["model"]["rotation_residuals"]
    labels = ["Correct", "Reverse", "No rotation"]
    values = [residuals[key]["rmse_mm"] for key in ("correct", "reverse", "none")]
    fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    axis.bar(labels, values, color=["#2563eb", "#94a3b8", "#94a3b8"])
    axis.set_ylabel("Truth-projection WEPL RMSE (mm)")
    axis.set_title("Rotation convention diagnostic")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(assets / "rotation_residuals.png", dpi=180)
    plt.close(fig)


def diagnose(stage8c: dict[str, Any], stage8: dict[str, Any], device: int, force: bool) -> dict[str, Any]:
    import cupy as cp

    qc, _, preprocessing, stage8_output = paths(stage8c)
    decision_path = qc / "diagnostic_decision.json"
    if decision_path.is_file() and not force:
        return read_json(decision_path)
    require_stage8(stage8c)
    qc.mkdir(parents=True, exist_ok=True)
    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    truth = np.array(read_volume(stage8_output / "truth" / "truth_rsp.mhd")[0], copy=False)
    core_masks = sphere_core_masks(stage8, simulation)
    update_progress(qc, status="RUNNING", stage="diagnose", group="local_geometry")
    coordinate = diagnostic_coordinate_check(stage8c, simulation)
    intersection = diagnostic_intersections(projector, stage8)
    operator = run_operator_checks(projector, preprocessing, stage8c, stage8)
    model, coverage = coverage_and_model_checks(
        projector, preprocessing, truth, core_masks, stage8c, stage8, qc
    )
    gates = stage8c["gates"]
    hard = {
        "coordinate": coordinate["maximum_mm"] <= gates["coordinate_roundtrip_max_mm"],
        "intersection": intersection["validity_mismatches"] == 0
        and intersection["maximum_interval_difference_mm"] <= gates["intersection_cpu_cuda_max_mm"],
        "constant_projection": operator["constant_projection_max_relative_error"]
        <= gates["constant_projection_relative_error"],
        "adjoint": operator["adjoint_max_relative_error"] <= gates["adjoint_relative_error"],
    }
    zero_core = sum(row["zero_voxels"] for row in model["core_coverage"])
    correct_rmse = model["rotation_residuals"]["correct"]["rmse_mm"]
    other_rmse = min(
        model["rotation_residuals"]["reverse"]["rmse_mm"],
        model["rotation_residuals"]["none"]["rmse_mm"],
    )
    if not all(hard.values()) or correct_rmse > other_rmse:
        category = "GEOMETRY_OR_OPERATOR_ERROR"
    elif zero_core > int(gates["coverage_zero_core_voxels"]):
        category = "COVERAGE_DEFICIT"
    else:
        stage8_decision = read_json(resolve(stage8c["stage8_qc"]) / "stage8_decision.json")
        reconstruction_rmse = float(stage8_decision["validation"]["wepl_rmse_mm"])
        category = (
            "PHYSICS_MODEL_MISMATCH"
            if correct_rmse > 1.05 * reconstruction_rmse
            else "LIKELY_UNDERCONVERGED"
        )
    result = {
        "status": "PASS" if all(hard.values()) else "FAIL",
        "category": category,
        "source_sha256": digest_sources(stage8c, stage8),
        "coordinate": coordinate,
        "intersection": intersection,
        "operator": operator,
        "model": model,
        "hard_gates": hard,
        "zero_core_voxels": zero_core,
        "test_partition_opened": False,
    }
    atomic_json(decision_path, result)
    write_csv(qc / "coverage_by_sphere.csv", model["core_coverage"])
    write_csv(qc / "coordinate_checks.csv", coordinate["rows"])
    write_csv(qc / "operator_checks.csv", operator["rows"])
    diagnostic_plots(qc, coverage, {"model": model})
    update_progress(qc, status=result["status"], stage="diagnose", group=category)
    return result


def synthetic_scenarios(
    stage8: dict[str, Any], simulation: dict[str, Any], mlic: dict[str, float]
) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    support = support_mask(stage8)
    uniform = np.zeros(support.shape, np.float32)
    uniform[support] = float(mlic["Water"])
    scenarios: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        ("uniform_water", uniform, {"spheres": []})
    ]
    templates = [
        ("center_air", "Air", 10.0, [0.0, 0.0, 0.0]),
        ("center_high_rsp", "SpineBone", 12.0, [0.0, 0.0, 0.0]),
        ("offaxis_high_rsp", "SpineBone", 12.0, [22.0, 7.0, 18.0]),
    ]
    for name, material, diameter, center in templates:
        scene = {"spheres": [{
            "name": name, "material": material, "diameter_mm": diameter,
            "scanner_center_mm": center,
        }]}
        scenarios.append((name, build_truth(stage8, scene, mlic), scene))
    scenarios.append(("five_sphere", build_truth(stage8, simulation, mlic), simulation))
    return scenarios


def simple_image_metrics(
    image: np.ndarray, target: np.ndarray, stage8: dict[str, Any], scene: dict[str, Any],
    mlic: dict[str, float],
) -> dict[str, float]:
    support = support_mask(stage8)
    water_reference = float(mlic["Water"])
    water = support & (np.abs(target - water_reference) < 1e-6)
    result = {
        "phantom_rmse": float(np.sqrt(np.mean((image[support] - target[support]) ** 2))),
        "water_bias": float(np.mean(image[water] - target[water])),
        "large_sphere_mape_percent": 0.0,
        "air_absolute_rsp_error": 0.0,
        "maximum_core_error_percent": 0.0,
    }
    if not scene["spheres"]:
        return result
    x, y, z = coordinates(stage8)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    errors, large = [], []
    for item in scene["spheres"]:
        center = np.asarray(item["scanner_center_mm"], float)
        radius = max(float(item["diameter_mm"]) / 2 - 1.0, 0.5)
        mask = (xx-center[0])**2 + (yy-center[1])**2 + (zz-center[2])**2 <= radius**2
        reference = 0.0011471876206752695 if item["material"] == "Air" else float(mlic[item["material"]])
        mean = float(np.mean(image[mask]))
        if item["material"] == "Air":
            result["air_absolute_rsp_error"] = abs(mean - reference)
        else:
            error = abs(100.0 * (mean - reference) / reference)
            errors.append(error)
            if float(item["diameter_mm"]) >= 10.0:
                large.append(error)
    result["maximum_core_error_percent"] = max(errors, default=0.0)
    result["large_sphere_mape_percent"] = float(np.mean(large)) if large else 0.0
    return result


def run_iterative_case(
    *, name: str, stage8c: dict[str, Any], stage8: dict[str, Any], preprocessing: Path,
    output: Path, target: np.ndarray, scene: dict[str, Any], mlic: dict[str, float],
    projector: GpuMlpProjector3D, mode: str, synthetic: bool, max_epochs: int,
    device: int, qc: Path, initial_path: Path | None = None, initial_epoch: int = 0,
    force: bool = False, relaxation: float | None = None,
    relaxation_decay: float | None = None, progress_stage: str | None = None,
    enable_validation_early_stop: bool = True,
) -> dict[str, Any]:
    import cupy as cp

    summary_path = output / "run_summary.json"
    relaxation = (
        float(stage8["reconstruction"]["relaxation"])
        if relaxation is None else float(relaxation)
    )
    relaxation_decay = (
        float(stage8["reconstruction"]["relaxation_decay"])
        if relaxation_decay is None else float(relaxation_decay)
    )
    solver = {
        "relaxation": relaxation,
        "relaxation_decay": relaxation_decay,
        "subsets": int(stage8["reconstruction"]["subsets"]),
        "mode": mode,
        "synthetic": synthetic,
    }
    if summary_path.is_file() and not force:
        prior_complete = read_json(summary_path)
        if prior_complete.get("solver") not in (None, solver):
            raise RuntimeError(
                f"solver configuration changed for {name}; use a new output directory"
            )
        if (
            prior_complete.get("status") == "PASS"
            and prior_complete.get("epochs")
            and (
                prior_complete.get("science_gate_pass") is True
                or prior_complete.get("stop_reason") == "validation_plateau"
                or int(prior_complete["epochs"][-1]["epoch"]) >= max_epochs
            )
        ):
            return prior_complete
    output.mkdir(parents=True, exist_ok=True)
    support_cpu = support_mask(stage8)
    support = cp.asarray(support_cpu)
    if initial_path is not None:
        image_cpu = np.array(read_volume(initial_path)[0], copy=True)
    else:
        image_cpu = np.zeros_like(target, np.float32)
        image_cpu[support_cpu] = 1.0
    image = cp.asarray(image_cpu)
    target_gpu = cp.asarray(target)
    rows: list[dict[str, Any]] = []
    if summary_path.is_file() and not force:
        prior = read_json(summary_path)
        rows = list(prior.get("epochs", []))
        if rows:
            latest = int(rows[-1]["epoch"])
            checkpoint = output / "recon" / f"epoch_{latest:02d}.mhd"
            if checkpoint.is_file():
                image = cp.asarray(np.array(read_volume(checkpoint)[0], copy=True))
                initial_epoch = latest
    batch_size = int(stage8c["diagnostic_batch_size"])
    subsets = int(stage8["reconstruction"]["subsets"])
    started = time.perf_counter()
    consecutive_pass = 0
    stop_reason = "max_epochs"

    def matched_validation_rmse() -> float:
        squared = count = 0
        for run_id in stage8c["diagnostic_runs"]:
            for batch in iter_batches(preprocessing, int(run_id), "screen", batch_size):
                bundle = projector.build_paths(
                    batch, float(run_id) * float(stage8["angle_step_deg"])
                )
                target_prediction, target_valid = projector.predict_from_paths(target_gpu, bundle)
                image_prediction, image_valid = projector.predict_from_paths(image, bundle)
                valid = target_valid.astype(cp.bool_) & image_valid.astype(cp.bool_)
                difference = target_prediction[valid] - image_prediction[valid]
                squared += float(cp.sum(difference * difference, dtype=cp.float64).get())
                count += int(difference.size)
        return float(math.sqrt(squared / max(count, 1)))

    for epoch in range(initial_epoch + 1, max_epochs + 1):
        epoch_input = image.copy()
        squared = count = 0
        for subset in range(subsets):
            numerator, denominator = cp.zeros_like(image), cp.zeros_like(image)
            for run_id in range(subset, int(stage8["runs"]), subsets):
                for batch in iter_batches(preprocessing, run_id, mode, batch_size):
                    bundle = projector.build_paths(batch, run_id * float(stage8["angle_step_deg"]))
                    if synthetic:
                        measured, _ = projector.predict_from_paths(target_gpu, bundle)
                    else:
                        measured = cp.asarray(batch["wepl_mm"], cp.float32)
                    part_squared, part_count = projector.accumulate_from_paths(
                        image, measured, bundle, numerator, denominator
                    )
                    squared += part_squared
                    count += part_count
            covered = denominator > 0
            update = cp.zeros_like(image)
            update[covered] = numerator[covered] / denominator[covered]
            epoch_relaxation = relaxation / (
                1.0 + relaxation_decay * (epoch - 1)
            )
            image += epoch_relaxation * update
            cp.maximum(image, 0.0, out=image)
            image *= support
            elapsed = time.perf_counter() - started
            completed_steps = (epoch - initial_epoch - 1) * subsets + subset + 1
            total_steps = max(max_epochs - initial_epoch, 1) * subsets
            update_progress(
                qc, status="RUNNING",
                stage=progress_stage or ("closure" if mode == "screen" else "confirm"),
                group=name, epoch=epoch, total_epochs=max_epochs, subset=subset + 1,
                total_subsets=subsets, task_elapsed_seconds=elapsed,
                task_eta_seconds=(
                    elapsed / max(completed_steps, 1) * max(total_steps - completed_steps, 0)
                ),
            )
        image_cpu = cp.asnumpy(image)
        metric = simple_image_metrics(image_cpu, target, stage8, scene, mlic)
        update_max = float(cp.max(cp.abs(image - epoch_input)).get())
        validation = None
        if not synthetic and mode == "train":
            validation = evaluate_partition(
                projector, image, preprocessing, int(stage8["runs"]), "validation",
                batch_size, float(stage8["angle_step_deg"]),
            )
        row = {
            "epoch": epoch,
            "training_wepl_rmse_mm": float(math.sqrt(squared / max(count, 1))),
            "matched_validation_wepl_rmse_mm": matched_validation_rmse() if synthetic else None,
            "training_measurements": int(count),
            "update_max_abs": update_max,
            "validation_wepl_rmse_mm": (
                validation["wepl_rmse_mm"] if validation is not None else None
            ),
            "relaxation": epoch_relaxation,
            **metric,
        }
        rows.append(row)
        write_volume(
            output / "recon" / f"epoch_{epoch:02d}.mhd", image_cpu,
            tuple(stage8["grid"]["spacing_xyz_mm"]), tuple(stage8["grid"]["origin_xyz_mm"]),
        )
        partial = {
            "status": "RUNNING", "case": name, "synthetic": synthetic,
            "solver": solver, "epochs": rows,
        }
        atomic_json(summary_path, partial)
        gate = stage8c["gates"]
        passed = (
            synthetic
            and row["matched_validation_wepl_rmse_mm"] <= gate["synthetic_wepl_rmse_mm"]
            and abs(row["water_bias"]) <= gate["water_absolute_bias"]
            and row["large_sphere_mape_percent"] <= gate["large_sphere_mape_percent"]
            and row["air_absolute_rsp_error"] <= gate["air_absolute_rsp_error"]
        )
        consecutive_pass = consecutive_pass + 1 if passed else 0
        if consecutive_pass >= 2:
            stop_reason = "synthetic_gate"
            break
        if (
            enable_validation_early_stop
            and not synthetic and mode == "train" and len(rows) >= 3
        ):
            recent = [float(item["validation_wepl_rmse_mm"]) for item in rows[-3:]]
            improvements = [
                (recent[index - 1] - recent[index]) / max(recent[index - 1], 1e-12)
                for index in (1, 2)
            ]
            updates = [float(item["update_max_abs"]) for item in rows[-3:]]
            if (
                max(improvements) < float(stage8c["gates"]["confirm_validation_plateau_relative"])
                and updates[2] < updates[1] < updates[0]
            ):
                stop_reason = "validation_plateau"
                break
    result = {
        "status": "PASS", "case": name, "synthetic": synthetic, "epochs": rows,
        "solver": solver,
        "science_gate_pass": synthetic_stable_pass(rows, stage8c) if synthetic else None,
        "stop_reason": stop_reason,
        "selected_epoch": int(rows[-1]["epoch"]), "elapsed_seconds": time.perf_counter() - started,
        "test_partition_opened": False,
    }
    atomic_json(summary_path, result)
    return result


def exploratory_full_fixed015(
    stage8c: dict[str, Any], stage8: dict[str, Any], device: int, force: bool
) -> dict[str, Any]:
    """Run the best post-hoc OS-SART schedule on the full frozen training partition."""
    import cupy as cp

    qc, output, preprocessing, _ = paths(stage8c)
    convergence_path = qc / "convergence_decision.json"
    if not convergence_path.is_file():
        raise RuntimeError("run Stage 8C convergence screening first")
    convergence_value = read_json(convergence_path)
    candidate = next(
        (row for row in convergence_value.get("candidates", [])
         if row.get("candidate") == "fixed_0p15"),
        None,
    )
    if candidate is None:
        raise RuntimeError("fixed_0p15 candidate is missing from convergence results")

    settings = stage8c["exploratory_full_fixed015"]
    source_sha256 = digest_sources(stage8c, stage8)
    manifest_path = qc / "exploratory_full_fixed015_manifest.json"
    if manifest_path.is_file() and not force:
        previous = read_json(manifest_path)
        if previous.get("source_sha256") != source_sha256:
            raise RuntimeError(
                "exploratory full-run source/configuration changed; use a versioned output "
                "or explicitly restart this action with --force"
            )
    atomic_json(manifest_path, {
        "source_sha256": source_sha256,
        "configuration": settings,
        "screen_candidate": {
            "candidate": candidate["candidate"], "final": candidate["final"],
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_partition_opened": False,
    })

    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    truth = build_truth(stage8, simulation, mlic)
    update_progress(
        qc, status="RUNNING", stage="full-fixed015", group="full80_real_data",
        completed_runs=0, total_runs=1, epoch=0,
        total_epochs=int(settings["max_epochs"]), subset=0,
        total_subsets=int(stage8["reconstruction"]["subsets"]),
    )
    run = run_iterative_case(
        name="full80_fixed_0p15", stage8c=stage8c, stage8=stage8,
        preprocessing=preprocessing,
        output=output / "exploratory_full" / "fixed_0p15",
        target=truth, scene=simulation, mlic=mlic, projector=projector,
        mode=str(settings["partition"]), synthetic=False,
        max_epochs=int(settings["max_epochs"]), device=device, qc=qc, force=force,
        relaxation=float(settings["relaxation"]),
        relaxation_decay=float(settings["relaxation_decay"]),
        progress_stage="full-fixed015", enable_validation_early_stop=False,
    )
    rows = run["epochs"]
    minimum_rmse = min(rows, key=lambda row: row["phantom_rmse"])
    minimum_mape = min(rows, key=lambda row: row["large_sphere_mape_percent"])
    minimum_validation = min(rows, key=lambda row: row["validation_wepl_rmse_mm"])
    result = {
        "status": "COMPLETE",
        "category": "EXPLORATORY_FULL_REAL_DATA",
        "configuration": settings,
        "completed_epoch": int(rows[-1]["epoch"]),
        "stop_reason": run["stop_reason"],
        "minimum_phantom_rmse": minimum_rmse,
        "minimum_large_sphere_mape": minimum_mape,
        "minimum_validation_wepl_rmse": minimum_validation,
        "final": rows[-1],
        "source_sha256": source_sha256,
        "test_partition_opened": False,
        "not_a_frozen_winner": True,
    }
    atomic_json(qc / "exploratory_full_fixed015_decision.json", result)
    write_csv(qc / "exploratory_full_fixed015_metrics.csv", rows)
    update_progress(
        qc, status="COMPLETE", stage="full-fixed015",
        group="EXPLORATORY_FULL_REAL_DATA", completed_runs=1, total_runs=1,
        epoch=int(rows[-1]["epoch"]), total_epochs=int(settings["max_epochs"]),
        subset=int(stage8["reconstruction"]["subsets"]),
        total_subsets=int(stage8["reconstruction"]["subsets"]),
    )
    return result


def test_fixed015(
    stage8c: dict[str, Any], stage8: dict[str, Any], device: int, force: bool
) -> dict[str, Any]:
    """Evaluate the frozen exploratory checkpoint once on the correctional test split."""
    import cupy as cp

    qc, output, preprocessing, _ = paths(stage8c)
    exploratory_path = qc / "exploratory_full_fixed015_decision.json"
    if not exploratory_path.is_file():
        raise RuntimeError("complete --action full-fixed015 before opening the test split")
    exploratory = read_json(exploratory_path)
    if exploratory.get("status") != "COMPLETE":
        raise RuntimeError("the fixed-0.15 full reconstruction is incomplete")
    settings = stage8c["fixed015_test_review"]
    frozen_epoch = int(settings["frozen_epoch"])
    if frozen_epoch != int(exploratory["completed_epoch"]):
        raise RuntimeError(
            "the frozen test epoch does not match the completed exploratory checkpoint"
        )
    decision_path = qc / "fixed015_test_decision.json"
    if decision_path.is_file() and not force:
        return read_json(decision_path)
    checkpoint = (
        output / "exploratory_full" / "fixed_0p15" / "recon"
        / f"epoch_{frozen_epoch:02d}.mhd"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    update_progress(
        qc, status="RUNNING", stage="test-fixed015", group="correctional_test_review",
        completed_runs=0, total_runs=1, epoch=frozen_epoch, total_epochs=frozen_epoch,
        subset=0, total_subsets=0,
    )
    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    image_cpu = np.array(read_volume(checkpoint)[0], copy=True)
    image = cp.asarray(image_cpu)
    test = evaluate_partition(
        projector, image, preprocessing, int(stage8["runs"]), "test",
        int(stage8c["diagnostic_batch_size"]), float(stage8["angle_step_deg"]),
    )
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    truth = build_truth(stage8, simulation, mlic)
    metrics, materials, edges = image_metrics(
        image_cpu, truth, stage8, simulation, mlic
    )
    aluminium = next(row for row in materials if row["material"] == "Aluminium")
    air = next(row for row in materials if row["material"] == "Air")
    gates = stage8c["gates"]
    finite_edges = all(np.isfinite(metrics[key]) for key in (
        "edge_width_x_mm", "edge_width_y_mm", "edge_width_z_mm"
    ))
    performance = {
        "water_bias": abs(metrics["water_bias"]) <= gates["water_absolute_bias"],
        "water_std": metrics["water_std_rsp"] <= 0.01,
        "large_material_mape": metrics["large_material_mape_percent"]
        <= gates["large_sphere_mape_percent"],
        "aluminium": abs(aluminium["error_percent"]) <= 2.0,
        "air": air["absolute_rsp_error"] <= gates["air_absolute_rsp_error"],
        "finite_edges": finite_edges,
        "test_wepl": test["wepl_rmse_mm"] <= gates["confirm_test_wepl_rmse_max_mm"],
        "finite": metrics["nonfinite"] == 0 and metrics["outside_nonzero"] == 0,
    }
    result = {
        "status": "PASS" if all(performance.values()) else "PERFORMANCE_FAIL",
        "category": (
            "RELIABLE_VOXEL_BASELINE_CANDIDATE"
            if all(performance.values()) else "CORRECTIONAL_TEST_FAILED"
        ),
        "frozen_epoch": frozen_epoch,
        "configuration": stage8c["exploratory_full_fixed015"],
        "test": test,
        "image": metrics,
        "materials": materials,
        "edges": edges,
        "performance_gates": performance,
        "test_partition_opened": True,
        "correctional_recheck": bool(settings["correctional_recheck"]),
        "historical_test_was_opened_in_stage8": True,
        "source_sha256": digest_sources(stage8c, stage8),
    }
    atomic_json(decision_path, result)
    write_csv(qc / "fixed015_test_material_metrics.csv", materials)
    write_csv(qc / "fixed015_test_edge_metrics.csv", edges)
    write_volume(
        output / "final" / f"recon_stage8c_fixed015_epoch{frozen_epoch:02d}.mhd",
        image_cpu, tuple(stage8["grid"]["spacing_xyz_mm"]),
        tuple(stage8["grid"]["origin_xyz_mm"]),
    )
    lines = [
        "# Stage 8C固定0.15第30轮修正性测试复核", "",
        f"- 状态：`{result['status']}`；", 
        f"- 测试WEPL RMSE：`{test['wepl_rmse_mm']:.6f} mm`；",
        f"- 水区偏差/标准差：`{100.0 * metrics['water_bias']:.4f}%` / "
        f"`{100.0 * metrics['water_std_rsp']:.4f}%`；",
        f"- 大材料球MAPE：`{metrics['large_material_mape_percent']:.4f}%`；",
        f"- 铝球误差：`{aluminium['error_percent']:.4f}%`；",
        f"- Air绝对RSP误差：`{air['absolute_rsp_error']:.6f}`；", "",
        "该测试集在Stage 8历史运行中已经打开，因此本结果属于修正性复核，"
        "不宣称为从未查看的全新独立测试。", "",
    ]
    (qc / "fixed015_test_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    update_progress(
        qc, status=result["status"], stage="test-fixed015", group=result["category"],
        completed_runs=1, total_runs=1, epoch=frozen_epoch, total_epochs=frozen_epoch,
        subset=0, total_subsets=0,
    )
    return result


def synthetic_gate(row: dict[str, Any], stage8c: dict[str, Any]) -> bool:
    gate = stage8c["gates"]
    return bool(
        row["matched_validation_wepl_rmse_mm"] <= gate["synthetic_wepl_rmse_mm"]
        and abs(row["water_bias"]) <= gate["water_absolute_bias"]
        and row["large_sphere_mape_percent"] <= gate["large_sphere_mape_percent"]
        and row["air_absolute_rsp_error"] <= gate["air_absolute_rsp_error"]
    )


def synthetic_stable_pass(rows: list[dict[str, Any]], stage8c: dict[str, Any]) -> bool:
    return len(rows) >= 2 and all(synthetic_gate(row, stage8c) for row in rows[-2:])


def convergence_score(row: dict[str, Any], stage8c: dict[str, Any]) -> float:
    """Dimensionless distance to the matched-model acceptance region."""
    gate = stage8c["gates"]
    return float(max(
        row["matched_validation_wepl_rmse_mm"] / gate["synthetic_wepl_rmse_mm"],
        abs(row["water_bias"]) / gate["water_absolute_bias"],
        row["large_sphere_mape_percent"] / gate["large_sphere_mape_percent"],
        row["air_absolute_rsp_error"] / gate["air_absolute_rsp_error"],
    ))


def convergence_plots(qc: Path, candidates: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    assets = qc / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fields = [
        ("matched_validation_wepl_rmse_mm", "Matched WEPL RMSE (mm)"),
        ("large_sphere_mape_percent", "Large-sphere MAPE (%)"),
        ("air_absolute_rsp_error", "Air absolute RSP error"),
        ("phantom_rmse", "Volume RSP RMSE"),
    ]
    for candidate in candidates:
        rows = candidate["epochs"]
        for axis, (field, label) in zip(axes.flat, fields):
            axis.plot(
                [row["epoch"] for row in rows], [row[field] for row in rows],
                marker="o", markersize=2.5, linewidth=1.2, label=candidate["candidate"],
            )
            axis.set(xlabel="Epoch", ylabel=label)
            axis.grid(alpha=0.25)
    axes[0, 0].axhline(0.01, color="#dc2626", linestyle="--", linewidth=1)
    axes[0, 1].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.05, color="#dc2626", linestyle="--", linewidth=1)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Stage 8C matched-model convergence screen")
    fig.savefig(assets / "convergence_candidates.png", dpi=180)
    plt.close(fig)


def convergence(
    stage8c: dict[str, Any], stage8: dict[str, Any], device: int, force: bool
) -> dict[str, Any]:
    """Screen relaxation schedules using only matched synthetic measurements."""
    import cupy as cp

    qc, output, preprocessing, _ = paths(stage8c)
    diagnostic = read_json(qc / "diagnostic_decision.json")
    if diagnostic.get("status") != "PASS":
        raise RuntimeError("Stage 8C diagnostic must pass before convergence screening")
    closure_decision = read_json(qc / "closure_decision.json")
    if closure_decision.get("status") == "PASS":
        raise RuntimeError("matched-model closure already passed; convergence repair is unnecessary")
    decision_path = qc / "convergence_decision.json"
    if decision_path.is_file() and not force:
        return read_json(decision_path)
    source_sha256 = digest_sources(stage8c, stage8)
    manifest_path = qc / "convergence_manifest.json"
    if manifest_path.is_file() and not force:
        previous = read_json(manifest_path)
        if previous.get("source_sha256") != source_sha256:
            raise RuntimeError(
                "Stage 8C convergence source/configuration changed; use a versioned output "
                "or explicitly restart this action with --force"
            )
    atomic_json(manifest_path, {
        "source_sha256": source_sha256,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_partition_opened": False,
    })

    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    scenarios = {
        name: (target, scene)
        for name, target, scene in synthetic_scenarios(stage8, simulation, mlic)
    }
    settings = stage8c["convergence"]
    target, scene = scenarios[str(settings["screen_case"])]
    candidate_results = []
    total_candidates = len(settings["candidates"])
    for candidate_index, setting in enumerate(settings["candidates"], start=1):
        candidate = str(setting["name"])
        update_progress(
            qc, status="RUNNING", stage="convergence", group=f"screen:{candidate}",
            completed_runs=candidate_index - 1, total_runs=total_candidates,
            epoch=0, total_epochs=int(setting["max_epochs"]), subset=0,
            total_subsets=int(stage8["reconstruction"]["subsets"]),
        )
        initial_path = None
        initial_epoch = 0
        if "resume_closure_epoch" in setting:
            initial_epoch = int(setting["resume_closure_epoch"])
            initial_path = (
                output / "closure" / str(settings["screen_case"]) / "recon"
                / f"epoch_{initial_epoch:02d}.mhd"
            )
            if not initial_path.is_file():
                raise FileNotFoundError(
                    f"baseline convergence seed is missing: {initial_path}"
                )
        run = run_iterative_case(
            name=f"five_sphere_{candidate}", stage8c=stage8c, stage8=stage8,
            preprocessing=preprocessing, output=output / "convergence" / "screen" / candidate,
            target=target, scene=scene, mlic=mlic, projector=projector, mode="screen",
            synthetic=True, max_epochs=int(setting["max_epochs"]), device=device, qc=qc,
            initial_path=initial_path, initial_epoch=initial_epoch, force=force,
            relaxation=float(setting["relaxation"]),
            relaxation_decay=float(setting["relaxation_decay"]),
            progress_stage="convergence",
        )
        last = run["epochs"][-1]
        candidate_results.append({
            "candidate": candidate,
            "relaxation": float(setting["relaxation"]),
            "relaxation_decay": float(setting["relaxation_decay"]),
            "max_epochs": int(setting["max_epochs"]),
            "passed": synthetic_stable_pass(run["epochs"], stage8c),
            "score": convergence_score(last, stage8c),
            "epochs": run["epochs"],
            "final": last,
        })
        atomic_json(qc / "convergence_partial.json", {
            "status": "RUNNING", "candidates": candidate_results,
            "test_partition_opened": False,
        })

    eligible = [row for row in candidate_results if row["passed"]]
    if not eligible:
        result = {
            "status": "FAIL",
            "category": "SLOW_CONVERGENCE_UNRESOLVED",
            "candidates": candidate_results,
            "verification": [],
            "winner": None,
            "source_sha256": source_sha256,
            "test_partition_opened": False,
        }
    else:
        best_score = min(row["score"] for row in eligible)
        tie = float(settings["score_tie_relative"])
        near = [row for row in eligible if row["score"] <= best_score * (1.0 + tie)]
        winner = min(
            near,
            key=lambda row: (
                row["relaxation"] / (
                    1.0 + row["relaxation_decay"] * (row["final"]["epoch"] - 1)
                ),
                row["score"],
            ),
        )
        verification = []
        total_verification = len(settings["verification_cases"])
        for case_index, case_name in enumerate(settings["verification_cases"], start=1):
            update_progress(
                qc, status="RUNNING", stage="convergence",
                group=f"verify:{case_name}", completed_runs=case_index - 1,
                total_runs=total_verification, epoch=0,
                total_epochs=int(settings["verification_max_epochs"]), subset=0,
                total_subsets=int(stage8["reconstruction"]["subsets"]),
            )
            case_target, case_scene = scenarios[str(case_name)]
            run = run_iterative_case(
                name=f"{case_name}_{winner['candidate']}", stage8c=stage8c,
                stage8=stage8, preprocessing=preprocessing,
                output=output / "convergence" / "verification" / str(case_name),
                target=case_target, scene=case_scene, mlic=mlic, projector=projector,
                mode="screen", synthetic=True,
                max_epochs=int(settings["verification_max_epochs"]), device=device, qc=qc,
                force=force, relaxation=float(winner["relaxation"]),
                relaxation_decay=float(winner["relaxation_decay"]),
                progress_stage="convergence",
            )
            last = run["epochs"][-1]
            verification.append({
                "case": case_name,
                "passed": synthetic_stable_pass(run["epochs"], stage8c),
                "score": convergence_score(last, stage8c), **last,
            })
        verified = all(row["passed"] for row in verification)
        result = {
            "status": "PASS" if verified else "FAIL",
            "category": (
                "MATCHED_MODEL_CONVERGENCE_RESTORED"
                if verified else "SYNTHETIC_VERIFICATION_FAILED"
            ),
            "winner": {
                key: winner[key] for key in (
                    "candidate", "relaxation", "relaxation_decay", "score", "final"
                )
            },
            "candidates": candidate_results,
            "verification": verification,
            "source_sha256": source_sha256,
            "test_partition_opened": False,
        }

    atomic_json(decision_path, result)
    write_csv(qc / "convergence_candidates.csv", [
        {
            "candidate": row["candidate"], "relaxation": row["relaxation"],
            "relaxation_decay": row["relaxation_decay"], "max_epochs": row["max_epochs"],
            "passed": row["passed"], "score": row["score"], **row["final"],
        }
        for row in candidate_results
    ])
    if result["verification"]:
        write_csv(qc / "convergence_verification.csv", result["verification"])
    convergence_plots(qc, candidate_results)
    update_progress(
        qc, status=result["status"], stage="convergence", group=result["category"]
    )
    return result


def closure(stage8c: dict[str, Any], stage8: dict[str, Any], device: int, force: bool) -> dict[str, Any]:
    import cupy as cp

    qc, output, preprocessing, stage8_output = paths(stage8c)
    diagnostic = read_json(qc / "diagnostic_decision.json")
    if diagnostic.get("status") != "PASS" or diagnostic.get("category") in {
        "GEOMETRY_OR_OPERATOR_ERROR", "COVERAGE_DEFICIT"
    }:
        raise RuntimeError(
            "Stage 8C diagnostic found a geometry/operator/coverage failure; inspect and fix it before closure"
        )
    decision_path = qc / "closure_decision.json"
    if decision_path.is_file() and not force:
        return read_json(decision_path)
    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    cases = []
    for name, target, scene in synthetic_scenarios(stage8, simulation, mlic):
        cases.append(run_iterative_case(
            name=name, stage8c=stage8c, stage8=stage8, preprocessing=preprocessing,
            output=output / "closure" / name, target=target, scene=scene, mlic=mlic,
            projector=projector, mode="screen", synthetic=True,
            max_epochs=int(stage8c["closure_max_epochs"]), device=device, qc=qc, force=force,
        ))
    actual_target = build_truth(stage8, simulation, mlic)
    actual = run_iterative_case(
        name="actual_screen", stage8c=stage8c, stage8=stage8, preprocessing=preprocessing,
        output=output / "closure" / "actual_screen", target=actual_target, scene=simulation,
        mlic=mlic, projector=projector, mode="screen", synthetic=False,
        max_epochs=int(stage8c["actual_screen_max_epochs"]), device=device, qc=qc,
        initial_path=stage8_output / "screen" / "b0" / "recon" / "epoch_03.mhd",
        initial_epoch=3, force=force,
    )
    gate = stage8c["gates"]
    synthetic_pass = all(
        case["epochs"][-1]["matched_validation_wepl_rmse_mm"] <= gate["synthetic_wepl_rmse_mm"]
        and abs(case["epochs"][-1]["water_bias"]) <= gate["water_absolute_bias"]
        and case["epochs"][-1]["large_sphere_mape_percent"] <= gate["large_sphere_mape_percent"]
        and case["epochs"][-1]["air_absolute_rsp_error"] <= gate["air_absolute_rsp_error"]
        for case in cases
    )
    actual_rows = actual["epochs"]
    improving = len(actual_rows) >= 3 and all(
        actual_rows[index]["phantom_rmse"] < actual_rows[index - 1]["phantom_rmse"]
        for index in range(len(actual_rows) - 2, len(actual_rows))
    )
    if not synthetic_pass:
        category = "OPERATOR_OR_CONVERGENCE_FAILURE"
    elif improving:
        category = "UNDERCONVERGENCE_CONFIRMED"
    else:
        category = "PHYSICS_MODEL_MISMATCH_CONFIRMED"
    result = {
        "status": "PASS" if synthetic_pass else "FAIL",
        "category": category,
        "synthetic_cases": [
            {"case": case["case"], "selected_epoch": case["selected_epoch"], **case["epochs"][-1]}
            for case in cases
        ],
        "actual_screen": {"selected_epoch": actual["selected_epoch"], **actual_rows[-1]},
        "test_partition_opened": False,
    }
    atomic_json(decision_path, result)
    write_csv(qc / "closure_metrics.csv", result["synthetic_cases"])
    update_progress(qc, status=result["status"], stage="closure", group=category)
    return result


def confirm(
    stage8c: dict[str, Any], stage8: dict[str, Any], raw_root: Path | None,
    jobs: int, device: int, force: bool,
) -> dict[str, Any]:
    import cupy as cp

    del jobs  # Reserved for a versioned Stage 8C re-preprocessing branch.
    qc, output, preprocessing, stage8_output = paths(stage8c)
    closure_decision = read_json(qc / "closure_decision.json")
    if closure_decision.get("status") != "PASS":
        raise RuntimeError("matched-model closure did not pass; full confirmation is forbidden")
    if closure_decision.get("category") == "PHYSICS_MODEL_MISMATCH_CONFIRMED":
        raise RuntimeError(
            "physical-model mismatch was confirmed; implement and validate a versioned model correction before full confirmation"
        )
    decision_path = qc / "stage8c_decision.json"
    if decision_path.is_file() and not force:
        return read_json(decision_path)
    if raw_root is not None and not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    cp.cuda.Device(device).use()
    projector = GpuMlpProjector3D(stage8)
    simulation = read_json(resolve(stage8["simulation_config"]))
    mlic = load_mlic(resolve(stage8["mlic_reference"]))
    truth = build_truth(stage8, simulation, mlic)
    result = run_iterative_case(
        name="full80_beta0", stage8c=stage8c, stage8=stage8, preprocessing=preprocessing,
        output=output / "confirm" / "full80_beta0", target=truth, scene=simulation,
        mlic=mlic, projector=projector, mode="train", synthetic=False,
        max_epochs=int(stage8c["confirm_max_epochs"]), device=device, qc=qc,
        initial_path=stage8_output / "confirm" / "b0" / "recon" / "epoch_03.mhd",
        initial_epoch=3, force=force,
    )
    rows = result["epochs"]
    if len(rows) >= 3:
        recent = [float(row["maximum_core_error_percent"]) for row in rows[-3:]]
        relative_decline = [
            (recent[index - 1] - recent[index]) / max(recent[index - 1], 1e-12)
            for index in (1, 2)
        ]
        if (
            int(rows[-1]["epoch"]) >= int(stage8c["confirm_max_epochs"])
            and min(relative_decline) >= 0.05
        ):
            result = run_iterative_case(
                name="full80_beta0", stage8c=stage8c, stage8=stage8,
                preprocessing=preprocessing, output=output / "confirm" / "full80_beta0",
                target=truth, scene=simulation, mlic=mlic, projector=projector,
                mode="train", synthetic=False,
                max_epochs=int(stage8c["confirm_extension_max_epochs"]), device=device,
                qc=qc, initial_path=None, initial_epoch=0, force=False,
            )
            rows = result["epochs"]
    selected = rows[-1]
    checkpoint = output / "confirm" / "full80_beta0" / "recon" / f"epoch_{selected['epoch']:02d}.mhd"
    image = cp.asarray(np.array(read_volume(checkpoint)[0], copy=True))
    validation = evaluate_partition(
        projector, image, preprocessing, int(stage8["runs"]), "validation",
        int(stage8c["diagnostic_batch_size"]), float(stage8["angle_step_deg"]),
    )
    test = evaluate_partition(
        projector, image, preprocessing, int(stage8["runs"]), "test",
        int(stage8c["diagnostic_batch_size"]), float(stage8["angle_step_deg"]),
    )
    image_cpu = cp.asnumpy(image)
    metrics, materials, edges = image_metrics(image_cpu, truth, stage8, simulation, mlic)
    aluminium = next(row for row in materials if row["material"] == "Aluminium")
    finite_axes = all(np.isfinite(metrics[key]) for key in (
        "edge_width_x_mm", "edge_width_y_mm", "edge_width_z_mm"
    ))
    gates = stage8c["gates"]
    performance = {
        "water_bias": abs(metrics["water_bias"]) <= gates["water_absolute_bias"],
        "large_material_mape": metrics["large_material_mape_percent"] <= gates["large_sphere_mape_percent"],
        "aluminium": abs(aluminium["error_percent"]) <= 2.0,
        "air": next(row for row in materials if row["material"] == "Air")["absolute_rsp_error"]
        <= gates["air_absolute_rsp_error"],
        "finite_edges": finite_axes,
        "test_wepl": test["wepl_rmse_mm"] <= gates["confirm_test_wepl_rmse_max_mm"],
        "finite": metrics["nonfinite"] == 0 and metrics["outside_nonzero"] == 0,
    }
    final = {
        "status": "PASS" if all(performance.values()) else "PERFORMANCE_FAIL",
        "category": "RELIABLE_VOXEL_BASELINE" if all(performance.values()) else "FULL_CONFIRMATION_INCOMPLETE",
        "selected_epoch": int(selected["epoch"]),
        "validation": validation,
        "test": test,
        "image": metrics,
        "materials": materials,
        "edges": edges,
        "performance_gates": performance,
        "raw_root_checked": str(raw_root) if raw_root is not None else None,
        "test_partition_opened": True,
    }
    atomic_json(decision_path, final)
    write_volume(
        output / "final" / "recon_stage8c.mhd", image_cpu,
        tuple(stage8["grid"]["spacing_xyz_mm"]), tuple(stage8["grid"]["origin_xyz_mm"]),
    )
    write_csv(qc / "material_metrics.csv", materials)
    write_csv(qc / "edge_metrics.csv", edges)
    update_progress(qc, status=final["status"], stage="confirm", group=final["category"])
    return final


def report(stage8c: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    qc, output, _, stage8_output = paths(stage8c)
    diagnosis = read_json(qc / "diagnostic_decision.json") if (qc / "diagnostic_decision.json").is_file() else None
    closure_value = read_json(qc / "closure_decision.json") if (qc / "closure_decision.json").is_file() else None
    final = read_json(qc / "stage8c_decision.json") if (qc / "stage8c_decision.json").is_file() else None
    fixed_review = (
        read_json(qc / "fixed015_test_decision.json")
        if (qc / "fixed015_test_decision.json").is_file()
        else None
    )
    displayed_final = final or fixed_review
    assets = qc / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if closure_value:
        actual_summary = read_json(output / "closure" / "actual_screen" / "run_summary.json")
        rows = actual_summary["epochs"]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
        axes[0].plot([row["epoch"] for row in rows], [row["phantom_rmse"] for row in rows], "o-")
        axes[0].set(xlabel="Epoch", ylabel="Phantom RSP RMSE", title="Actual-data convergence")
        axes[1].plot([row["epoch"] for row in rows], [row["training_wepl_rmse_mm"] for row in rows], "o-")
        axes[1].set(xlabel="Epoch", ylabel="Training WEPL RMSE (mm)", title="Data consistency")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.savefig(assets / "closure_convergence.png", dpi=180)
        plt.close(fig)
    exploratory_metrics = qc / "exploratory_full_fixed015_metrics.csv"
    if exploratory_metrics.is_file():
        with exploratory_metrics.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        epochs = [int(row["epoch"]) for row in rows]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
        axes[0].plot(epochs, [float(row["validation_wepl_rmse_mm"]) for row in rows], "o-", markersize=3)
        axes[0].set(xlabel="Epoch", ylabel="Validation WEPL RMSE (mm)", title="Measured-data consistency")
        axes[1].plot(epochs, [float(row["large_sphere_mape_percent"]) for row in rows], "o-", markersize=3)
        axes[1].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1, label="1% target")
        axes[1].set(xlabel="Epoch", ylabel="Large-sphere MAPE (%)", title="Material recovery")
        axes[1].legend()
        axes[2].plot(epochs, [float(row["air_absolute_rsp_error"]) for row in rows], "o-", markersize=3)
        axes[2].axhline(0.05, color="#dc2626", linestyle="--", linewidth=1, label="0.05 target")
        axes[2].set(xlabel="Epoch", ylabel="Absolute RSP error", title="Air-sphere recovery")
        axes[2].legend()
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.savefig(assets / "full_fixed015_convergence.png", dpi=180)
        plt.close(fig)
    if displayed_final:
        _, stage8 = load_config()
        truth = np.array(read_volume(stage8_output / "truth" / "truth_rsp.mhd")[0], copy=False)
        before = np.array(read_volume(stage8_output / "final" / "recon_stage8.mhd")[0], copy=False)
        if final:
            after_path = output / "final" / "recon_stage8c.mhd"
        else:
            epoch = int(fixed_review["frozen_epoch"])
            after_path = output / "final" / f"recon_stage8c_fixed015_epoch{epoch:02d}.mhd"
        after = np.array(read_volume(after_path)[0], copy=False)
        y_index = truth.shape[1] // 2
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        panels = [truth[:, y_index], before[:, y_index], after[:, y_index]]
        titles = ["Truth", "Stage 8", "Stage 8C"]
        for axis, panel, title in zip(axes[0], panels, titles):
            shown = axis.imshow(panel, origin="lower", cmap="viridis", vmin=0, vmax=2.2)
            axis.set_title(title)
            fig.colorbar(shown, ax=axis, shrink=0.75, label="RSP")
        for axis, panel, title in zip(axes[1], panels, titles):
            error = panel - panels[0]
            shown = axis.imshow(error, origin="lower", cmap="coolwarm", vmin=-0.5, vmax=0.5)
            axis.set_title(f"{title} - truth")
            fig.colorbar(shown, ax=axis, shrink=0.75, label="RSP error")
        fig.savefig(assets / "stage8_stage8c_comparison.png", dpi=180)
        plt.close(fig)
        before_materials = read_json(resolve(stage8c["stage8_qc"]) / "stage8_decision.json")["materials"]
        names = [row["material"] for row in displayed_final["materials"]]
        before_error = [next(row["error_percent"] for row in before_materials if row["material"] == name) for name in names]
        after_error = [next(row["error_percent"] for row in displayed_final["materials"] if row["material"] == name) for name in names]
        x = np.arange(len(names))
        fig, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
        axis.bar(x - 0.2, before_error, 0.4, label="Stage 8", color="#94a3b8")
        axis.bar(x + 0.2, after_error, 0.4, label="Stage 8C", color="#2563eb")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, names, rotation=20)
        axis.set_ylabel("Core RSP error (%)")
        axis.set_title("Material recovery before and after Stage 8C")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(assets / "material_recovery_comparison.png", dpi=180)
        plt.close(fig)
    lines = ["# Stage 8C三维系统诊断与重新计算", ""]
    if diagnosis:
        lines += [f"- 系统诊断：`{diagnosis['status']}`；分类：`{diagnosis['category']}`；", ""]
    if closure_value:
        lines += [f"- 匹配算子闭环：`{closure_value['status']}`；分类：`{closure_value['category']}`；", ""]
    if final:
        lines += [
            f"- 全量确认：`{final['status']}`；选择第`{final['selected_epoch']}`轮；",
            f"- 测试WEPL RMSE：`{final['test']['wepl_rmse_mm']:.6f} mm`；",
            f"- 大材料球MAPE：`{final['image']['large_material_mape_percent']:.4f}%`。",
            "",
        ]
    elif fixed_review:
        lines += [
            f"- 固定0.15全量重建与修正性测试：`{fixed_review['status']}`；选择第`{fixed_review['frozen_epoch']}`轮；",
            f"- 测试WEPL RMSE：`{fixed_review['test']['wepl_rmse_mm']:.6f} mm`；",
            f"- 大材料球MAPE：`{fixed_review['image']['large_material_mape_percent']:.4f}%`；",
            f"- 水区偏差/标准差：`{100.0 * fixed_review['image']['water_bias']:.4f}%` / `{100.0 * fixed_review['image']['water_std_rsp']:.4f}%`。",
            "",
        ]
    lines += [
        "![Coverage](assets/coverage_slices.png)", "",
        "![Rotation](assets/rotation_residuals.png)", "",
    ]
    if closure_value:
        lines += ["![Closure convergence](assets/closure_convergence.png)", ""]
    if exploratory_metrics.is_file():
        lines += ["![Fixed 0.15 convergence](assets/full_fixed015_convergence.png)", ""]
    if displayed_final:
        lines += [
            "![Stage 8 and Stage 8C](assets/stage8_stage8c_comparison.png)", "",
            "![Material recovery](assets/material_recovery_comparison.png)", "",
        ]
    lines += ["Stage 8C测试集属于修正性复核：Stage 8历史运行已经打开过同一测试集。"]
    (qc / "stage8c_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def status(stage8c: dict[str, Any]) -> None:
    qc, _, _, _ = paths(stage8c)
    progress = qc / "progress.json"
    if not progress.is_file():
        print("Stage 8C has not started.")
        return
    value = read_json(progress)
    print("Stage 8C status")
    for key in (
        "status", "stage", "group", "completed_runs", "total_runs", "epoch",
        "total_epochs", "subset", "total_subsets", "updated_at",
    ):
        if key in value:
            print(f"  {key}: {value[key]}")
    for key in ("task_elapsed_seconds", "task_eta_seconds"):
        if key in value:
            seconds = max(0, int(value[key]))
            hours, remainder = divmod(seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  {key}: {hours:02d}:{minutes:02d}:{seconds:02d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", required=True,
        choices=[
            "diagnose", "closure", "convergence", "full-fixed015",
            "test-fixed015", "confirm", "report", "status",
        ],
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage8c, stage8 = load_config()
    if args.action == "status":
        status(stage8c)
    elif args.action == "diagnose":
        diagnose(stage8c, stage8, args.device, args.force)
    elif args.action == "closure":
        closure(stage8c, stage8, args.device, args.force)
    elif args.action == "convergence":
        convergence(stage8c, stage8, args.device, args.force)
    elif args.action == "full-fixed015":
        exploratory_full_fixed015(stage8c, stage8, args.device, args.force)
    elif args.action == "test-fixed015":
        test_fixed015(stage8c, stage8, args.device, args.force)
    elif args.action == "confirm":
        confirm(stage8c, stage8, args.raw_root, args.jobs, args.device, args.force)
    elif args.action == "report":
        report(stage8c)


if __name__ == "__main__":
    main()
