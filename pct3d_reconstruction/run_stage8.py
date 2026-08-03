#!/usr/bin/env python3
"""Run the guarded compact three-dimensional pCT reconstruction pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CONFIG_PATH = HERE / "stage8_config.json"
QC = HERE / "qc" / "results0718_compact_3d_pilot"
PROGRESS = QC / "progress.json"
sys.path.insert(0, str(HERE))

from evaluation3d import image_metrics  # noqa: E402
from gpu_operator3d import GpuMlpProjector3D  # noqa: E402
from io3d import pair_batch, read_pairs, read_partition, read_volume, write_volume  # noqa: E402
from physics3d import (  # noqa: E402
    build_truth,
    load_mlic,
    mlp_position_cpu,
    ray_finite_cylinder_interval,
    support_mask,
)
from preprocessing3d import locate_run, process_run  # noqa: E402
from reconstruction3d import evaluate_partition, reconstruct  # noqa: E402
from regularization3d import proximal_huber_tv  # noqa: E402


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


def update_progress(**values: Any) -> None:
    current = read_json(PROGRESS) if PROGRESS.is_file() else {}
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(PROGRESS, current)


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    config["_wepl_model"] = str(resolve(config["wepl_model"]))
    return config


def require_status(path: Path, accepted: tuple[str, ...], prerequisite: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing {prerequisite}: run the preceding Stage 8 action first")
    value = read_json(path)
    if value.get("status") not in accepted:
        raise RuntimeError(f"{prerequisite} is not complete: {value.get('status')}")
    return value


def config_digest(config: dict[str, Any], raw_root: Path) -> str:
    operational = {
        "raw_root_default",
        "preprocessing_output",
        "reconstruction_output",
        "minimum_local_free_gib",
        "wsl_backing_mount",
        "minimum_host_backing_free_gib",
        "minimum_host_backing_free_without_root_cache_gib",
    }
    payload = {
        "config": {
            key: value
            for key, value in config.items()
            if not key.startswith("_") and key not in operational
        },
        "raw_root": str(raw_root.resolve()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_digest() -> tuple[str, dict[str, str]]:
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(HERE.glob("*.py"))
    }
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return digest, sources


def sha256_file(path_text: str) -> tuple[str, int, int, str]:
    path = Path(path_text)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns, digest.hexdigest()


def stage_root_cache(config: dict, raw_root: Path, force: bool) -> tuple[Path, list[dict[str, Any]]]:
    """Sequentially copy external ROOT to local storage and hash that byte stream."""

    output = resolve(config["preprocessing_output"])
    cache = output / "_root_cache"
    files = []
    total = int(config["runs"]) * len(config["required_root"])
    completed = 0
    started = time.perf_counter()
    update_progress(status="RUNNING", stage="stage_root_cache", completed_files=0, total_files=total)
    for run_id in range(int(config["runs"])):
        source_dir = locate_run(raw_root, run_id)
        if source_dir is None:
            raise FileNotFoundError(f"run {run_id:03d}")
        target_dir = cache / f"run_{run_id:03d}"
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in config["required_root"]:
            source, target = source_dir / name, target_dir / name
            cache_record_path = target.with_suffix(target.suffix + ".sha256.json")
            source_stat = source.stat()
            prior = (
                read_json(cache_record_path)
                if cache_record_path.is_file()
                else {}
            )
            reusable = (
                not force
                and target.is_file()
                and target.stat().st_size == source_stat.st_size
                and prior.get("bytes") == source_stat.st_size
                and prior.get("mtime_ns") == source_stat.st_mtime_ns
                and bool(prior.get("sha256"))
            )
            if not reusable:
                temporary = target.with_suffix(target.suffix + ".tmp")
                digest = hashlib.sha256()
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    while chunk := reader.read(16 * 1024 * 1024):
                        writer.write(chunk)
                        digest.update(chunk)
                temporary.replace(target)
                sha = digest.hexdigest()
                atomic_json(
                    cache_record_path,
                    {
                        "source": str(source),
                        "bytes": source_stat.st_size,
                        "mtime_ns": source_stat.st_mtime_ns,
                        "sha256": sha,
                    },
                )
            else:
                _, _, _, sha = sha256_file(str(target))
                if sha != prior["sha256"]:
                    raise RuntimeError(
                        f"local ROOT cache hash mismatch for {target}; rerun with --force"
                    )
            files.append(
                {
                    "source": str(source),
                    "bytes": source_stat.st_size,
                    "mtime_ns": source_stat.st_mtime_ns,
                    "sha256": sha,
                }
            )
            completed += 1
            elapsed = time.perf_counter() - started
            update_progress(
                completed_files=completed,
                task_elapsed_seconds=elapsed,
                task_eta_seconds=elapsed / completed * (total - completed),
            )
    return cache, files


def preflight(config: dict, raw_root: Path, jobs: int, device: int, force: bool) -> dict:
    gate = read_json(resolve(config["wepl_gate"]))
    if gate.get("status") != "PASS":
        raise RuntimeError("Stage 6B WEPL gate is not PASS")
    paths = []
    missing = []
    for run_id in range(int(config["runs"])):
        directory = locate_run(raw_root, run_id)
        for name in config["required_root"]:
            path = None if directory is None else directory / name
            if path is None or not path.is_file() or path.stat().st_size == 0:
                missing.append({"run_id": run_id, "file": name})
            else:
                paths.append(path)
    if missing:
        raise RuntimeError(f"missing compact-3D ROOT: {missing[:5]}")
    output_parent = resolve(config["preprocessing_output"]).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_parent).free
    required = int(config["minimum_local_free_gib"] * 1024**3)
    if free < required:
        raise RuntimeError(f"need {required / 1024**3:.0f} GiB free, found {free / 1024**3:.1f}")
    backing_path = Path(config.get("wsl_backing_mount", ""))
    backing_free = None
    if str(backing_path) and backing_path.is_dir():
        backing_free = shutil.disk_usage(backing_path).free
        cache = resolve(config["preprocessing_output"]) / "_root_cache"
        cached_root_count = sum(
            1 for _ in cache.glob("run_*/*.root")
        ) if cache.is_dir() else 0
        cache_complete = cached_root_count == int(config["runs"]) * len(
            config["required_root"]
        )
        preprocessing_summary_path = QC / "preprocessing_summary.json"
        preprocessing_summary = (
            read_json(preprocessing_summary_path)
            if preprocessing_summary_path.is_file()
            else {}
        )
        prepared_file_sets = all(
            (resolve(config["preprocessing_output"]) / "pairs" / f"pairs{run_id:04d}.mhd").is_file()
            and (resolve(config["preprocessing_output"]) / "pairs" / f"pairs{run_id:04d}.raw").is_file()
            and (resolve(config["preprocessing_output"]) / "events" / f"events{run_id:04d}.npy").is_file()
            and (resolve(config["preprocessing_output"]) / "splits" / f"split{run_id:04d}.npz").is_file()
            and (resolve(config["preprocessing_output"]) / "splits" / f"screen{run_id:04d}.npz").is_file()
            for run_id in range(int(config["runs"]))
        )
        preprocessing_complete = (
            preprocessing_summary.get("status") == "PASS" and prepared_file_sets
        )
        heavy_root_staging_required = not cache_complete and not preprocessing_complete
        backing_required = int(
            config.get(
                "minimum_host_backing_free_without_root_cache_gib"
                if heavy_root_staging_required
                else "minimum_host_backing_free_gib",
                0,
            )
            * 1024**3
        )
        if backing_free < backing_required:
            raise RuntimeError(
                "WSL backing drive has insufficient host space: "
                f"need {backing_required / 1024**3:.0f} GiB, "
                f"found {backing_free / 1024**3:.1f} GiB"
            )
    import cupy as cp

    cp.cuda.Device(device).use()
    cp.asarray([1.0], dtype=cp.float32).sum().get()
    manifest_path = QC / "input_manifest.json"
    digest = config_digest(config, raw_root)
    quick = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    cached = read_json(manifest_path) if manifest_path.is_file() and not force else {}
    existing_pairs = resolve(config["preprocessing_output"]) / "pairs"
    if existing_pairs.is_dir() and any(existing_pairs.glob("pairs*.mhd")) and not cached:
        raise RuntimeError(
            "prepared Stage 8 pairs exist without an input manifest; "
            "move them aside or rerun explicitly with --force"
        )
    prepared = existing_pairs.is_dir() and any(existing_pairs.glob("pairs*.mhd"))
    if prepared and cached:
        frozen_science = cached.get("scientific_config_sha256")
        if frozen_science is not None and frozen_science != digest:
            raise RuntimeError("Stage 8 scientific configuration changed while prepared pairs exist")
        if cached.get("quick") != [list(row) for row in quick]:
            raise RuntimeError("Stage 8 ROOT input changed while prepared pairs exist")
    code_sha, source_hashes = source_digest()
    manifest = {
        "status": "READY_FOR_LOCAL_STAGING",
        "scientific_config_sha256": digest,
        "source_sha256": code_sha,
        "source_files": source_hashes,
        "legacy_manifest_migrated": bool(prepared and cached and "scientific_config_sha256" not in cached),
        "raw_root": str(raw_root.resolve()),
        "runs": config["runs"],
        "files": cached.get("files", []),
        "quick": [list(row) for row in quick],
        "root_bytes": sum(path.stat().st_size for path in paths),
        "local_free_bytes": free,
        "wsl_backing_mount": str(backing_path) if backing_free is not None else None,
        "wsl_backing_free_bytes": backing_free,
        "complete_local_root_cache_present": cache_complete if backing_free is not None else None,
        "preprocessing_complete": preprocessing_complete if backing_free is not None else None,
        "heavy_root_staging_required": heavy_root_staging_required if backing_free is not None else None,
        "wepl_gate": gate,
        "test_partition_opened": False,
    }
    atomic_json(manifest_path, manifest)
    update_progress(status="PASS", stage="preflight")
    return manifest


def prepare(config: dict, raw_root: Path, jobs: int, force: bool) -> dict:
    require_status(
        QC / "input_manifest.json",
        ("READY_FOR_LOCAL_STAGING", "PASS"),
        "preflight manifest",
    )
    output = resolve(config["preprocessing_output"])
    output.mkdir(parents=True, exist_ok=True)
    summary_path = QC / "preprocessing_summary.json"
    truth_path = resolve(config["reconstruction_output"]) / "truth" / "truth_rsp.mhd"
    if not force and summary_path.is_file() and truth_path.is_file():
        previous = read_json(summary_path)
        complete = all(
            (output / "pairs" / f"pairs{run_id:04d}.mhd").is_file()
            and (output / "pairs" / f"pairs{run_id:04d}.raw").is_file()
            and (output / "events" / f"events{run_id:04d}.npy").is_file()
            and (output / "splits" / f"split{run_id:04d}.npz").is_file()
            and (output / "splits" / f"screen{run_id:04d}.npz").is_file()
            for run_id in range(int(config["runs"]))
        )
        if previous.get("status") == "PASS" and complete:
            update_progress(status="PASS", stage="prepare", skipped=True)
            return previous
    cache, hashes = stage_root_cache(config, raw_root, force)
    manifest = read_json(QC / "input_manifest.json") if (QC / "input_manifest.json").is_file() else {}
    manifest["files"] = hashes
    manifest["status"] = "PASS"
    atomic_json(QC / "input_manifest.json", manifest)
    update_progress(status="RUNNING", stage="prepare", completed_runs=0, total_runs=config["runs"])
    rows = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(process_run, run_id, str(cache), str(output), config, force): run_id
            for run_id in range(int(config["runs"]))
        }
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            elapsed = time.perf_counter() - started
            update_progress(
                completed_runs=index,
                task_elapsed_seconds=elapsed,
                task_eta_seconds=elapsed / index * (int(config["runs"]) - index),
            )
    rows.sort(key=lambda row: row["run_id"])
    if [row["run_id"] for row in rows] != list(range(int(config["runs"]))):
        raise RuntimeError("preprocessing did not cover every angle exactly once")
    for row in rows:
        if min(row["train"], row["validation"], row["test"]) <= 0:
            raise RuntimeError(f"empty split at run {row['run_id']}")
        if row["train"] + row["validation"] + row["test"] != row["filtered_support_hit"]:
            raise RuntimeError(f"incomplete split at run {row['run_id']}")
        if row["screen_train"] <= 0 or row["screen_train"] > row["train"]:
            raise RuntimeError(f"invalid screen subset at run {row['run_id']}")
    write_csv(QC / "preprocessing_by_run.csv", rows)
    totals = {
        key: int(sum(row[key] for row in rows))
        for key in ("paired", "physical", "filtered_support_hit", "train", "validation", "test", "screen_train", "air_only_count")
    }
    air_count = max(totals["air_only_count"], 1)
    air_mean = sum(row["air_only_residual_sum_mm"] for row in rows) / air_count
    air_abs = sum(row["air_only_residual_abs_sum_mm"] for row in rows) / air_count
    summary = {
        "status": "PASS",
        **totals,
        "air_only_residual_mean_mm": air_mean,
        "air_only_residual_mae_mm": air_abs,
        "elapsed_seconds": time.perf_counter() - started,
        "test_partition_opened": False,
    }
    if abs(air_mean) > float(config["acceptance"]["air_only_corrected_wepl_abs_mean_max_mm"]):
        summary["status"] = "FAIL"
    atomic_json(summary_path, summary)
    if summary["status"] != "PASS":
        raise RuntimeError(f"external-Air gate failed: {summary}")
    shutil.rmtree(cache)
    simulation = read_json(resolve(config["simulation_config"]))
    mlic = load_mlic(resolve(config["mlic_reference"]))
    truth = build_truth(config, simulation, mlic)
    write_volume(
        truth_path,
        truth,
        tuple(config["grid"]["spacing_xyz_mm"]),
        tuple(config["grid"]["origin_xyz_mm"]),
    )
    update_progress(status="PASS", stage="prepare")
    return summary


def operator_smoke(config: dict, device: int) -> dict:
    import cupy as cp

    require_status(QC / "preprocessing_summary.json", ("PASS",), "preprocessing")
    cp.cuda.Device(device).use()
    preprocessing = resolve(config["preprocessing_output"])
    projector = GpuMlpProjector3D(config)
    rng = np.random.default_rng(20260731)
    x_cpu = rng.normal(size=tuple(reversed(config["grid"]["size_xyz"]))).astype(np.float32)
    x_cpu *= support_mask(config)
    x = cp.asarray(x_cpu)
    angle_checks = []
    for run_id in (0, 90):
        pairs = read_pairs(preprocessing / "pairs" / f"pairs{run_id:04d}.mhd")
        split = read_partition(preprocessing / "splits" / f"split{run_id:04d}.npz")
        indexes = np.flatnonzero(split == 1)[:64]
        if len(indexes) < 16:
            raise RuntimeError(f"insufficient validation pairs for run {run_id}")
        batch = pair_batch(pairs, indexes)
        prediction, valid, paths = projector.predict(x, batch, float(run_id))
        y = cp.asarray(rng.normal(size=len(indexes)).astype(np.float32))
        y *= valid
        aty = projector.transpose(y, valid, paths)
        lhs = float(cp.dot(prediction.astype(cp.float64), y.astype(cp.float64)).get())
        rhs = float(cp.sum(x.astype(cp.float64) * aty.astype(cp.float64)).get())
        relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
        z_values = np.zeros(len(indexes), dtype=np.float32)
        gpu_points, gpu_valid = projector.debug_mlp_points(batch, z_values)
        cpu_points = []
        cpu_valid = []
        for local in range(len(indexes)):
            pin, pout, din, dout = (
                batch[key][local]
                for key in (
                    "position_in",
                    "position_out",
                    "direction_in",
                    "direction_out",
                )
            )
            ti, _, vi = ray_finite_cylinder_interval(
                pin[None],
                din[None],
                config["phantom_radius_mm"],
                config["phantom_half_length_y_mm"],
            )
            to, _, vo = ray_finite_cylinder_interval(
                pout[None],
                -dout[None],
                config["phantom_radius_mm"],
                config["phantom_half_length_y_mm"],
            )
            entry = pin + ti[0] * din / np.linalg.norm(din)
            exit_ = pout - to[0] * dout / np.linalg.norm(dout)
            valid_cpu = bool(vi[0] and vo[0] and entry[2] < 0 < exit_[2])
            cpu_valid.append(valid_cpu)
            cpu_points.append(
                mlp_position_cpu(0.0, entry, exit_, din, dout)
                if valid_cpu
                else np.full(3, np.nan)
            )
        cpu_points = np.asarray(cpu_points)
        common = np.asarray(cpu_valid) & gpu_valid
        mlp_max = (
            float(np.max(np.abs(cpu_points[common] - gpu_points[common])))
            if np.any(common)
            else float("inf")
        )
        angle_checks.append(
            {
                "run_id": run_id,
                "angle_deg": float(run_id),
                "adjoint_lhs": lhs,
                "adjoint_rhs": rhs,
                "adjoint_relative_error": relative,
                "cpu_cuda_mlp_max_abs_mm": mlp_max,
                "cpu_cuda_mlp_compared": int(np.count_nonzero(common)),
                "valid_rays": int(cp.sum(valid).get()),
            }
        )
    constant = cp.asarray(support_mask(config).astype(np.float32))
    regularized, _ = proximal_huber_tv(constant, constant.astype(cp.bool_), 0.006, 0.01, 5)
    regularization_constant_error = float(cp.max(cp.abs(regularized - constant)).get())
    result = {
        "status": "PASS"
        if all(
            row["adjoint_relative_error"]
            <= config["acceptance"]["adjoint_relative_error_max"]
            and row["cpu_cuda_mlp_max_abs_mm"] <= 2e-4
            for row in angle_checks
        )
        and regularization_constant_error <= 1e-7
        else "FAIL",
        "angle_checks": angle_checks,
        "constant_huber_tv_max_abs_change": regularization_constant_error,
        "gpu_memory_peak_bytes": int(cp.get_default_memory_pool().total_bytes()),
    }
    atomic_json(QC / "operator_smoke.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"3-D adjoint smoke failed: {result}")
    update_progress(status="PASS", stage="operator_smoke")
    return result


def truth_context(config: dict):
    simulation = read_json(resolve(config["simulation_config"]))
    mlic = load_mlic(resolve(config["mlic_reference"]))
    truth = np.asarray(read_volume(resolve(config["reconstruction_output"]) / "truth/truth_rsp.mhd")[0])
    return simulation, mlic, truth


def beta_tag(beta: float) -> str:
    return "b" + f"{beta:g}".replace(".", "p")


def select_screen(config: dict, summaries: list[dict]) -> dict:
    rows = []
    for summary in summaries:
        final = summary["epochs"][-1]
        rows.append(
            {
                "beta": float(summary["beta"]),
                "validation_wepl_rmse_mm": float(final["validation"]["wepl_rmse_mm"]),
                "material_mape_percent": float(final["image"]["large_material_mape_percent"]),
                "edge_width_mm": float(final["image"]["edge_width_mean_mm"]),
                "phantom_rmse": float(final["image"]["phantom_rmse"]),
            }
        )
    baseline = next(row for row in rows if row["beta"] == 0)
    best_wepl = min(row["validation_wepl_rmse_mm"] for row in rows)
    eligible = []
    for row in rows:
        row["eligible"] = (
            row["validation_wepl_rmse_mm"]
            <= best_wepl * (1 + config["selection"]["validation_wepl_relative_tolerance"])
            and row["material_mape_percent"]
            <= baseline["material_mape_percent"]
            + config["selection"]["material_mape_degradation_percentage_points"]
            and row["edge_width_mm"]
            <= baseline["edge_width_mm"] * (1 + config["selection"]["edge_width_relative_tolerance"])
        )
        if row["eligible"]:
            eligible.append(row)
    selection_fallback = None
    if not eligible:
        eligible = [baseline]
        selection_fallback = (
            "No candidate simultaneously met all selection tolerances; "
            "the unregularized beta=0 baseline was retained."
        )
    eligible.sort(key=lambda row: (row["phantom_rmse"], row["beta"]))
    winner = eligible[0]
    near = [
        row
        for row in eligible
        if row["phantom_rmse"]
        <= winner["phantom_rmse"] * (1 + config["selection"]["truth_rmse_tie_relative"])
    ]
    winner = min(near, key=lambda row: row["beta"])
    decision = {
        "status": "PASS",
        "winner_beta": winner["beta"],
        "candidates": rows,
        "selection_fallback": selection_fallback,
        "test_partition_opened": False,
    }
    atomic_json(QC / "screen_decision.json", decision)
    write_csv(QC / "screen_metrics.csv", rows)
    return decision


def screen(config: dict, device: int, force: bool) -> dict:
    require_status(QC / "operator_smoke.json", ("PASS",), "operator smoke test")
    preprocessing = resolve(config["preprocessing_output"])
    root = resolve(config["reconstruction_output"]) / "screen"
    simulation, mlic, truth = truth_context(config)
    summaries = []
    candidates = config["reconstruction"]["regularization_candidates"]
    for index, beta in enumerate(candidates):
        output = root / beta_tag(float(beta))
        update_progress(status="RUNNING", stage="screen", completed_groups=index, total_groups=len(candidates), group=output.name)
        summaries.append(
            reconstruct(
                config=config,
                preprocessing=preprocessing,
                output=output,
                mode="screen",
                beta=float(beta),
                truth=truth,
                simulation=simulation,
                mlic=mlic,
                device=device,
                progress=update_progress,
                force=force,
            )
        )
    decision = select_screen(config, summaries)
    update_progress(status="PASS", stage="screen", completed_groups=len(candidates))
    return decision


def choose_epoch(config: dict, summary: dict) -> int:
    rows = summary["epochs"]
    best_wepl = min(row["validation"]["wepl_rmse_mm"] for row in rows)
    best_mape = min(row["image"]["large_material_mape_percent"] for row in rows)
    best_edge = min(row["image"]["edge_width_mean_mm"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["validation"]["wepl_rmse_mm"]
        <= best_wepl * (1 + config["selection"]["validation_wepl_relative_tolerance"])
        and row["image"]["large_material_mape_percent"]
        <= best_mape + config["selection"]["material_mape_degradation_percentage_points"]
        and row["image"]["edge_width_mean_mm"]
        <= best_edge * (1 + config["selection"]["edge_width_relative_tolerance"])
    ]
    selection_fallback = None
    if not eligible:
        # A small 3-D sphere may not yet cross both 10% and 90% levels in
        # every direction, leaving the edge metric undefined or mutually
        # incompatible with the WEPL/material minima. In that case retain
        # validation discipline and choose solely by fixed validation WEPL.
        eligible = [
            row
            for row in rows
            if np.isfinite(row["validation"]["wepl_rmse_mm"])
        ]
        if not eligible:
            raise RuntimeError("no epoch has a finite validation WEPL RMSE")
        winner = min(
            eligible,
            key=lambda row: (row["validation"]["wepl_rmse_mm"], row["epoch"]),
        )
        selection_fallback = (
            "joint WEPL/material/edge tolerances had no intersection; "
            "selected minimum fixed-validation WEPL RMSE"
        )
        atomic_json(
            QC / "epoch_selection.json",
            {
                "status": "PASS_WITH_FALLBACK",
                "selected_epoch": int(winner["epoch"]),
                "reason": selection_fallback,
                "test_partition_opened": False,
            },
        )
        return int(winner["epoch"])
    eligible.sort(key=lambda row: (row["image"]["phantom_rmse"], row["epoch"]))
    winner = eligible[0]
    near = [
        row
        for row in eligible
        if row["image"]["phantom_rmse"]
        <= winner["image"]["phantom_rmse"] * (1 + config["selection"]["truth_rmse_tie_relative"])
    ]
    selected = min(int(row["epoch"]) for row in near)
    atomic_json(
        QC / "epoch_selection.json",
        {
            "status": "PASS",
            "selected_epoch": selected,
            "reason": selection_fallback,
            "test_partition_opened": False,
        },
    )
    return selected


def confirm(config: dict, device: int, force: bool) -> dict:
    import cupy as cp

    final_decision = QC / "stage8_decision.json"
    if final_decision.is_file() and not force:
        previous = read_json(final_decision)
        if previous.get("status") == "PASS" and previous.get("test_partition_opened"):
            update_progress(status="PASS", stage="confirm", skipped=True)
            return previous
    decision = require_status(
        QC / "screen_decision.json",
        ("PASS",),
        "regularization screen",
    )
    beta = float(decision["winner_beta"])
    preprocessing = resolve(config["preprocessing_output"])
    output = resolve(config["reconstruction_output"]) / "confirm" / beta_tag(beta)
    simulation, mlic, truth = truth_context(config)
    summary = reconstruct(
        config=config,
        preprocessing=preprocessing,
        output=output,
        mode="full",
        beta=beta,
        truth=truth,
        simulation=simulation,
        mlic=mlic,
        device=device,
        progress=update_progress,
        force=force,
    )
    epoch = choose_epoch(config, summary)
    volume = np.array(read_volume(output / "recon" / f"epoch_{epoch:02d}.mhd")[0], copy=True)
    final_path = resolve(config["reconstruction_output"]) / "final" / "recon_stage8.mhd"
    write_volume(
        final_path,
        volume,
        tuple(config["grid"]["spacing_xyz_mm"]),
        tuple(config["grid"]["origin_xyz_mm"]),
    )
    projector = GpuMlpProjector3D(config)
    test_metrics = evaluate_partition(
        projector,
        cp.asarray(volume),
        preprocessing,
        int(config["runs"]),
        "test",
        int(summary["batch_size"]),
        float(config["angle_step_deg"]),
    )
    metrics, materials, edges = image_metrics(volume, truth, config, simulation, mlic)
    result = {
        "status": "PASS",
        "winner_beta": beta,
        "selected_epoch": epoch,
        "validation": summary["epochs"][epoch - 1]["validation"],
        "test": test_metrics,
        "image": metrics,
        "materials": materials,
        "edges": edges,
        "resources": {
            "confirm_elapsed_seconds": float(summary["elapsed_seconds"]),
            "batch_size": int(summary["batch_size"]),
            "gpu_memory_pool_peak_bytes": int(summary["gpu_memory_pool_bytes"]),
            "training_measurements": int(
                sum(row["training_measurements"] for row in summary["epochs"])
            ),
        },
        "output": str(final_path),
        "test_partition_opened": True,
    }
    atomic_json(QC / "stage8_decision.json", result)
    update_progress(status="PASS", stage="confirm")
    return result


def report(config: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage8")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    decision = require_status(
        QC / "stage8_decision.json",
        ("PASS",),
        "Stage 8 confirmation",
    )
    truth = np.asarray(read_volume(resolve(config["reconstruction_output"]) / "truth/truth_rsp.mhd")[0])
    image = np.asarray(read_volume(Path(decision["output"]))[0])
    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    x_indices = [truth.shape[2] // 2]
    y_indices = [
        int(round((value - config["grid"]["origin_xyz_mm"][1]) / config["grid"]["spacing_xyz_mm"][1]))
        for value in (-7.0, 0.0, 7.0)
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, iy in enumerate(y_indices):
        for col, (data, title) in enumerate(((truth, "MLIC truth"), (image, "Reconstruction"), (image - truth, "Error"))):
            im = axes[row, col].imshow(
                data[:, iy, :],
                origin="lower",
                cmap="coolwarm" if col == 2 else "viridis",
                vmin=-0.08 if col == 2 else 0.0,
                vmax=0.08 if col == 2 else 2.15,
                extent=(-60, 60, -60, 60),
            )
            axes[row, col].set(title=f"{title}, y={(-7,0,7)[row]} mm", xlabel="x (mm)", ylabel="z (mm)")
            if col == 1:
                y_plane = (-7.0, 0.0, 7.0)[row]
                simulation = read_json(resolve(config["simulation_config"]))
                for sphere in simulation["spheres"]:
                    cx, cy, cz = sphere["scanner_center_mm"]
                    radius = float(sphere["diameter_mm"]) / 2
                    if abs(cy - y_plane) <= radius:
                        projected = np.sqrt(max(radius * radius - (cy - y_plane) ** 2, 0))
                        axes[row, col].add_patch(
                            Circle(
                                (cx, cz),
                                projected,
                                fill=False,
                                edgecolor="white",
                                linewidth=0.8,
                            )
                        )
            fig.colorbar(im, ax=axes[row, col], shrink=0.75)
    fig.savefig(assets / "truth_reconstruction_error.png", dpi=180)
    plt.close(fig)
    center = tuple(value // 2 for value in truth.shape)
    slices = [
        (lambda a: a[:, center[1], :], "x-z at y=0", (-60, 60, -60, 60)),
        (lambda a: a[center[0], :, :], "x-y at z=0", (-60, 60, -20, 20)),
        (lambda a: a[:, :, center[2]], "y-z at x=0", (-20, 20, -60, 60)),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for row, (extract, plane, extent) in enumerate(slices):
        for col, (data, title) in enumerate(((truth, "MLIC truth"), (image, "Reconstruction"), (image - truth, "Error"))):
            im = axes[row, col].imshow(
                extract(data),
                origin="lower",
                cmap="coolwarm" if col == 2 else "viridis",
                vmin=-0.08 if col == 2 else 0.0,
                vmax=0.08 if col == 2 else 2.15,
                extent=extent,
                aspect="equal",
            )
            axes[row, col].set_title(f"{title}: {plane}")
            fig.colorbar(im, ax=axes[row, col], shrink=0.75)
    fig.savefig(assets / "orthogonal_slices.png", dpi=180)
    plt.close(fig)
    epochs = read_json(
        resolve(config["reconstruction_output"])
        / "confirm"
        / beta_tag(float(decision["winner_beta"]))
        / "run_summary.json"
    )["epochs"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    epoch_numbers = [row["epoch"] for row in epochs]
    axes[0].plot(
        epoch_numbers,
        [row["image"]["phantom_rmse"] for row in epochs],
        "o-",
    )
    axes[0].set(xlabel="Epoch", ylabel="RSP RMSE", title="Image error")
    axes[1].plot(
        epoch_numbers,
        [row["validation"]["wepl_rmse_mm"] for row in epochs],
        "o-",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="WEPL RMSE (mm)",
        title="Validation residual",
    )
    axes[2].plot(
        epoch_numbers,
        [row["update_max_abs"] for row in epochs],
        "o-",
    )
    axes[2].set(
        xlabel="Epoch",
        ylabel="Maximum |update|",
        title="Update magnitude",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(assets / "epoch_convergence.png", dpi=180)
    plt.close(fig)
    material_rows = decision["materials"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = [row["material"] for row in material_rows]
    axes[0].bar(labels, [row["error_percent"] for row in material_rows])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(ylabel="Platform error (%)", title="Material core accuracy")
    axes[0].tick_params(axis="x", rotation=35)
    edge_axes = ("x", "y", "z")
    axes[1].bar(
        edge_axes,
        [decision["image"][f"edge_width_{axis}_mm"] for axis in edge_axes],
    )
    axes[1].set(ylabel="10%–90% width (mm)", title="Directional edge width")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(assets / "material_and_edge_metrics.png", dpi=180)
    plt.close(fig)
    write_csv(QC / "material_metrics.csv", decision["materials"])
    write_csv(QC / "edge_metrics.csv", decision["edges"])
    write_csv(
        QC / "runtime_metrics.csv",
        [
            {
                "epoch": row["epoch"],
                "elapsed_seconds": row["elapsed_seconds"],
                "training_measurements": row["training_measurements"],
                "pairs_per_second": row["training_measurements"] / max(row["elapsed_seconds"], 1e-9),
                "update_l2": row["update_l2"],
                "update_max_abs": row["update_max_abs"],
                "batch_size": decision["resources"]["batch_size"],
                "gpu_memory_pool_peak_bytes": decision["resources"][
                    "gpu_memory_pool_peak_bytes"
                ],
            }
            for row in epochs
        ],
    )
    performance = {
        "water_bias_pass": abs(decision["image"]["water_bias"]) <= config["acceptance"]["water_absolute_bias_max"],
        "large_material_mape_pass": decision["image"]["large_material_mape_percent"] <= config["acceptance"]["large_material_mape_max_percent"],
        "aluminium_core_pass": abs(next(row["error_percent"] for row in decision["materials"] if row["material"] == "Aluminium")) <= config["acceptance"]["aluminium_core_error_max_percent"],
    }
    decision["performance_targets"] = performance
    atomic_json(QC / "stage8_decision.json", decision)
    summary = f"""# Stage 8紧凑三维pCT结果

状态：**PASS（THREE_DIMENSIONAL_PIPELINE_COMPLETE）**。该状态表示三维数据链、
MLP算子、严格转置、划分和测试评价均完整，不代表所有性能目标自动通过。

## 冻结配置

- 360角度，每角度2,000,000个质子；
- `240×80×240 @ 0.5 mm`；
- Schulte水MLP、三线性8邻域GPU OS-SART；
- 18子集、3 epoch、均匀水圆柱初值；
- Huber-TV `beta={decision['winner_beta']}`，选择epoch `{decision['selected_epoch']}`。

## 独立测试和图像指标

- test WEPL RMSE：`{decision['test']['wepl_rmse_mm']:.6f} mm`；
- 水区均值/标准差：`{decision['image']['water_mean_rsp']:.6f}` /
  `{decision['image']['water_std_rsp']:.6f}`；
- 模体RSP RMSE：`{decision['image']['phantom_rmse']:.6f}`；
- 非Air材料球MAPE：`{decision['image']['nonair_material_mape_percent']:.4f}%`；
- 10--14 mm大材料球MAPE：`{decision['image']['large_material_mape_percent']:.4f}%`；
- 三方向平均10%--90%边缘宽度：`{decision['image']['edge_width_mean_mm']:.4f} mm`。

![Truth reconstruction error](assets/truth_reconstruction_error.png)

![Orthogonal slices](assets/orthogonal_slices.png)

![Material and edge metrics](assets/material_and_edge_metrics.png)

![Epoch convergence](assets/epoch_convergence.png)
"""
    (QC / "stage8_summary.md").write_text(summary, encoding="utf-8")
    update_progress(status="COMPLETE", stage="report")


def status() -> None:
    if not PROGRESS.is_file():
        print("Stage 8 has not started.")
        return
    value = read_json(PROGRESS)
    print("Stage 8 status")
    for key in ("status", "stage", "group", "completed_files", "total_files", "completed_runs", "total_runs", "completed_groups", "total_groups", "epoch", "total_epochs", "subset", "total_subsets"):
        if key in value:
            print(f"  {key}: {value[key]}")
    if "task_elapsed_seconds" in value:
        print(f"  elapsed: {format_duration(value['task_elapsed_seconds'])}")
    if "task_eta_seconds" in value:
        print(f"  task ETA: {format_duration(value['task_eta_seconds'])}")
    print(f"  updated_at: {value.get('updated_at', 'unknown')}")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=["preflight", "prepare", "operator-smoke", "screen", "confirm", "report", "all", "status"])
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.action == "status":
        status()
        return
    config = load_config()
    raw_root = args.raw_root or Path(config["raw_root_default"])
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    actions = ["preflight", "prepare", "operator-smoke", "screen", "confirm", "report"] if args.action == "all" else [args.action]
    for action in actions:
        if action == "preflight":
            preflight(config, raw_root, args.jobs, args.device, args.force)
        elif action == "prepare":
            prepare(config, raw_root, args.jobs, args.force)
        elif action == "operator-smoke":
            operator_smoke(config, args.device)
        elif action == "screen":
            screen(config, args.device, args.force)
        elif action == "confirm":
            confirm(config, args.device, args.force)
        elif action == "report":
            report(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        details = traceback.format_exc()
        QC.mkdir(parents=True, exist_ok=True)
        (QC / "latest_error.log").write_text(details, encoding="utf-8")
        try:
            update_progress(
                status="INTERRUPTED_RESUMABLE",
                error_type=type(error).__name__,
                error_message=str(error),
                error_log=str(QC / "latest_error.log"),
            )
        except Exception:
            pass
        raise
