#!/usr/bin/env python3
"""Run the complete Stage-5 inhomogeneous and alternating MLP study."""

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
STAGE4_ROOT = HERE.parent / "stage4_iterative_optimization"
CONFIG_PATH = HERE / "stage5_config.json"
QC_ROOT = HERE / "qc"
STATE_PATH = QC_ROOT / "progress.json"
sys.path[:0] = [
    str(HERE), str(STAGE4_ROOT), str(STAGE3_ROOT), str(CODE_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

import run_stage3 as stage3  # noqa: E402
import run_stage4 as stage4  # noqa: E402
from inhomogeneous_mlp import (  # noqa: E402
    catalog_rscp, conditioned_path, energy_profile, material_catalog,
    published_rscp, sample_map, scattering_energy_factor, truth_maps,
)
from trajectory_io import prepare_all, sha256  # noqa: E402
from preprocessing import paircuts  # noqa: E402
from stage3_io import (  # noqa: E402
    format_duration, load_json, partition_masks, read_packed_mask, relative,
    write_json,
)
from iterative_reconstruction.mhd_io import read_image_2d, write_image_2d  # noqa: E402
from iterative_reconstruction.mlp import (  # noqa: E402
    cylinder_intersections, schulte_positions,
)
from iterative_reconstruction.gpu_regularization import proximal_regularize  # noqa: E402


RUNS = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("all", "audit", "prepare-truth", "path-screen", "fixed-screen",
                 "alternating-screen", "confirm", "report", "smoke", "status"),
        default="all",
    )
    parser.add_argument("--datasets", default="s1,s2,s3,s4,s5")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--runs", type=int, default=RUNS, help="smoke testing only")
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    config = load_json(CONFIG_PATH)
    config["stage3"] = load_json((HERE / config["stage3_config"]).resolve())
    config["stage4"] = load_json((HERE / config["stage4_frozen"]).resolve())
    return config


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields, seen = [], set()
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


def update_state(**values: Any) -> None:
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_json(STATE_PATH) if STATE_PATH.is_file() else {}
    state.update(values)
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def datasets_from(text: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    return [stage3.dataset_record(config["stage3"], name)
            for name in stage3.parse_datasets(text, config["stage3"])]


def stage5_root(dataset: dict[str, Any]) -> Path:
    return Path(dataset["reconstruction_data"]) / "stage5"


def make_batch(pairs: np.ndarray, dataset: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    batch = stage3.make_batch(pairs, dataset, config["stage3"])
    batch["energy_in"] = np.asarray(pairs[:, 4, 0], np.float32)
    batch["energy_out"] = np.asarray(pairs[:, 4, 1], np.float32)
    return batch


def accepted_mask(dataset: dict[str, Any], config: dict[str, Any], run_id: int, count: int) -> np.ndarray:
    return read_packed_mask(
        stage3.mask_path(dataset, "baseline_3sigma", run_id), count,
        config["stage3"]["split"]["bit_order"],
    )


def audit(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output = QC_ROOT / "truth_pilot_manifest.json"
    if output.is_file() and not force:
        return load_json(output)
    root = REPOSITORY_ROOT / config["truth_pilot"]["simulation_data"]
    rows, started = [], time.perf_counter()
    expected = int(config["truth_pilot"]["angles"]) * 3
    for run_id in range(int(config["truth_pilot"]["angles"])):
        for name in ("PhaseSpaceIn.root", "PhaseSpaceOut.root", "PrimaryTrajectory.root"):
            path = root / f"run_{run_id:03d}" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({
                "run": run_id, "file": name, "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
            completed = len(rows)
            update_state(
                status="RUNNING", level="audit", task="hash ROOT files",
                completed=completed, total=expected,
                overall_fraction=0.03 * completed / expected,
                elapsed_seconds=time.perf_counter() - started,
            )
            if completed % 12 == 0:
                print(f"audit ROOT: {completed}/{expected}", flush=True)
    manifest = {
        "status": "PASS", "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": rows, "file_count": len(rows), "total_bytes": sum(r["bytes"] for r in rows),
        "config_hash": canonical_hash({k: v for k, v in config.items() if k not in {"stage3", "stage4"}}),
    }
    write_json(output, manifest)
    return manifest


def prepare_truth(config: dict[str, Any], jobs: int) -> dict[str, Any]:
    started = time.perf_counter()
    def progress(done: int, total: int, usable: int) -> None:
        elapsed = time.perf_counter() - started
        update_state(
            status="RUNNING", level="level1", task="prepare truth trajectories",
            completed=done, total=total, trajectories=usable,
            rate=usable / elapsed if elapsed else 0.0,
            overall_fraction=0.03 + 0.07 * done / total,
            elapsed_seconds=elapsed,
        )
        print(f"truth preparation: {done:02d}/{total}, usable={usable:,}", flush=True)
    rows = prepare_all(config, REPOSITORY_ROOT, jobs, progress)
    write_csv(QC_ROOT / "truth_preparation.csv", rows)
    result = {
        "status": "PASS", "runs": len(rows), "usable": sum(r["usable"] for r in rows),
        "paired": sum(r.get("paired", 0) for r in rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(QC_ROOT / "truth_preparation.json", result)
    return result


def _path_batch(
    cache: dict[str, np.ndarray], selected: np.ndarray, run_id: int,
    config: dict[str, Any], maps,
) -> dict[str, np.ndarray]:
    state_in = cache["state_in"][selected]
    state_out = cache["state_out"][selected]
    pin, pout = state_in[:, :3], state_out[:, :3]
    din, dout = state_in[:, 3:6], state_out[:, 3:6]
    entry, exit, endpoint_valid = cylinder_intersections(pin, pout, din, dout, 100.0)
    step = float(config["truth_pilot"]["evaluation_step_mm"])
    samples = int(round(200.0 / step))
    z = -100.0 + (np.arange(samples) + .5) * step
    water_x, water_y, water_valid = schulte_positions(entry, exit, din, dout, z)
    angle = run_id * float(config["truth_pilot"]["angle_step_deg"])
    z_grid = np.broadcast_to(z, water_x.shape)
    truth_rsp_on_water = sample_map(maps.rsp, water_x, z_grid, angle, step)
    inside = water_valid.astype(np.float32)
    numerical_water_energy = energy_profile(
        state_in[:, 6], state_out[:, 6], inside, step
    )
    numerical_water_x, _, _ = conditioned_path(
        entry, exit, din, dout, z,
        inside * scattering_energy_factor(numerical_water_energy), step,
    )
    energy_only_energy = energy_profile(
        state_in[:, 6], state_out[:, 6], truth_rsp_on_water, step
    )
    energy_only_x, _, _ = conditioned_path(
        entry, exit, din, dout, z,
        inside * scattering_energy_factor(energy_only_energy), step,
    )
    x, y = water_x.copy(), water_y.copy()
    for _ in range(int(config["path_screen"]["fixed_point_iterations"])):
        rsp = sample_map(maps.rsp, x, np.broadcast_to(z, x.shape), angle, step)
        rscp = sample_map(maps.rscp, x, np.broadcast_to(z, x.shape), angle, step)
        energy = energy_profile(state_in[:, 6], state_out[:, 6], rsp, step)
        power = rscp * scattering_energy_factor(energy)
        x, y, sigma = conditioned_path(entry, exit, din, dout, z, power, step)
    true_x, true_y = cache["true_x"][selected], cache["true_y"][selected]
    valid = np.isfinite(true_x) & water_valid & endpoint_valid[:, None]
    water_error = np.where(valid, water_x - true_x, np.nan)
    numerical_water_error = np.where(
        valid, numerical_water_x - true_x, np.nan
    )
    energy_only_error = np.where(valid, energy_only_x - true_x, np.nan)
    inhom_error = np.where(valid, x - true_x, np.nan)
    water_rmse = np.sqrt(np.nanmean(water_error ** 2, axis=1))
    numerical_water_rmse = np.sqrt(
        np.nanmean(numerical_water_error ** 2, axis=1)
    )
    energy_only_rmse = np.sqrt(np.nanmean(energy_only_error ** 2, axis=1))
    inhom_rmse = np.sqrt(np.nanmean(inhom_error ** 2, axis=1))
    material = cache["material"][selected]
    heterogeneous = np.any(valid & (material != 2) & (material != 255), axis=1)
    material_presence = np.stack(
        [np.any(valid & (material == code), axis=1) for code in range(6)],
        axis=1,
    )
    coverage = {
        k: np.nanmean(
            np.where(valid, np.abs(inhom_error) <= k * sigma, np.nan),
            axis=1,
        )
        for k in (1, 2, 3)
    }
    return {
        "event_id": cache["event_id"][selected], "split": cache["split"][selected],
        "water_rmse": water_rmse,
        "numerical_water_rmse": numerical_water_rmse,
        "energy_only_rmse": energy_only_rmse,
        "inhom_rmse": inhom_rmse,
        "heterogeneous": heterogeneous,
        "material_presence": material_presence,
        "coverage1": coverage[1], "coverage2": coverage[2], "coverage3": coverage[3],
        "water_depth_sse": np.nansum(water_error ** 2, axis=0),
        "inhom_depth_sse": np.nansum(inhom_error ** 2, axis=0),
        "depth_count": np.sum(valid, axis=0),
    }


def _bootstrap_lower(improvement: np.ndarray, samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    values = np.asarray(improvement[np.isfinite(improvement)], np.float64)
    if not len(values):
        return -math.inf
    means = np.empty(samples)
    for i in range(samples):
        means[i] = np.mean(rng.choice(values, len(values), replace=True))
    return float(np.quantile(means, 0.025))


def path_screen(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    decision_path = QC_ROOT / "level1_decision.json"
    if decision_path.is_file() and not force:
        return load_json(decision_path)
    maps = truth_maps(config, float(config["truth_pilot"]["evaluation_step_mm"]))
    cache_root = REPOSITORY_ROOT / config["truth_pilot"]["cache_data"]
    event_rows, depth_rows = [], []
    started, processed = time.perf_counter(), 0
    for run_id in range(int(config["truth_pilot"]["angles"])):
        with np.load(cache_root / f"truth_{run_id:03d}.npz") as loaded:
            cache = {key: loaded[key] for key in loaded.files}
        depth = None
        for begin in range(0, len(cache["event_id"]), 256):
            selected = np.arange(begin, min(begin + 256, len(cache["event_id"])))
            result = _path_batch(cache, selected, run_id, config, maps)
            material_names = (
                "Air", "Lung", "Water", "A150_Tissue_Plastic",
                "SpineBone", "Aluminium",
            )
            for i in range(len(selected)):
                row = {
                    "run": run_id, "event_id": int(result["event_id"][i]),
                    "partition": ("train", "validation", "test")[int(result["split"][i])],
                    "heterogeneous": bool(result["heterogeneous"][i]),
                    "water_rmse_mm": float(result["water_rmse"][i]),
                    "numerical_water_rmse_mm": float(
                        result["numerical_water_rmse"][i]
                    ),
                    "energy_only_rmse_mm": float(
                        result["energy_only_rmse"][i]
                    ),
                    "inhomogeneous_rmse_mm": float(result["inhom_rmse"][i]),
                    "relative_improvement": float(1.0 - result["inhom_rmse"][i] / result["water_rmse"][i]),
                    "coverage_1sigma": float(result["coverage1"][i]),
                    "coverage_2sigma": float(result["coverage2"][i]),
                    "coverage_3sigma": float(result["coverage3"][i]),
                }
                row.update({
                    f"traversed_{name}": bool(
                        result["material_presence"][i, code]
                    )
                    for code, name in enumerate(material_names)
                })
                event_rows.append(row)
            if depth is None:
                depth = [result["water_depth_sse"], result["inhom_depth_sse"], result["depth_count"]]
            else:
                for i, key in enumerate(("water_depth_sse", "inhom_depth_sse", "depth_count")):
                    depth[i] += result[key]
            processed += len(selected)
        for j in range(len(depth[2])):
            if depth[2][j]:
                depth_rows.append({
                    "run": run_id, "depth_mm": -100 + (j + .5) * float(config["truth_pilot"]["evaluation_step_mm"]),
                    "count": int(depth[2][j]),
                    "water_rms_mm": float(np.sqrt(depth[0][j] / depth[2][j])),
                    "inhomogeneous_rms_mm": float(np.sqrt(depth[1][j] / depth[2][j])),
                })
        elapsed = time.perf_counter() - started
        update_state(
            status="RUNNING", level="level1", task="path screen", completed=run_id + 1,
            total=int(config["truth_pilot"]["angles"]), trajectories=processed,
            rate=processed / elapsed if elapsed else 0, overall_fraction=.10 + .10 * (run_id + 1) / int(config["truth_pilot"]["angles"]),
            elapsed_seconds=elapsed,
        )
        print(f"path screen: {run_id+1:02d}/72, trajectories={processed:,}", flush=True)
    event_path = cache_root / "path_event_metrics.csv"
    write_csv(event_path, event_rows)
    write_csv(QC_ROOT / "path_depth_metrics.csv", depth_rows)
    material_names = (
        "Air", "Lung", "Water", "A150_Tissue_Plastic",
        "SpineBone", "Aluminium",
    )
    summary_rows = []
    for partition in ("train", "validation", "test"):
        partition_rows = [r for r in event_rows if r["partition"] == partition]
        for group in ("all", "heterogeneous"):
            selected_rows = (
                partition_rows
                if group == "all"
                else [r for r in partition_rows if r["heterogeneous"]]
            )
            if selected_rows:
                summary_rows.append({
                    "partition": partition, "group": group,
                    "count": len(selected_rows),
                    "water_rmse_mm": float(np.sqrt(np.mean([
                        r["water_rmse_mm"] ** 2 for r in selected_rows
                    ]))),
                    "inhomogeneous_rmse_mm": float(np.sqrt(np.mean([
                        r["inhomogeneous_rmse_mm"] ** 2 for r in selected_rows
                    ]))),
                    "mean_relative_improvement": float(np.mean([
                        r["relative_improvement"] for r in selected_rows
                    ])),
                })
    write_csv(QC_ROOT / "path_metrics.csv", summary_rows)
    material_rows = []
    for partition in ("validation", "test"):
        partition_rows = [r for r in event_rows if r["partition"] == partition]
        for material in material_names:
            selected_rows = [
                r for r in partition_rows
                if r.get(f"traversed_{material}", False)
            ]
            if selected_rows:
                material_rows.append({
                    "partition": partition, "material": material,
                    "count": len(selected_rows),
                    "water_rmse_mm": float(np.sqrt(np.mean([
                        r["water_rmse_mm"] ** 2 for r in selected_rows
                    ]))),
                    "inhomogeneous_rmse_mm": float(np.sqrt(np.mean([
                        r["inhomogeneous_rmse_mm"] ** 2 for r in selected_rows
                    ]))),
                    "mean_relative_improvement": float(np.mean([
                        r["relative_improvement"] for r in selected_rows
                    ])),
                })
    write_csv(QC_ROOT / "material_path_metrics.csv", material_rows)
    validation = [r for r in event_rows if r["partition"] == "validation"]
    hetero = [r for r in validation if r["heterogeneous"]]
    overall = float(np.mean([r["relative_improvement"] for r in validation]))
    heterogeneous = float(np.mean([r["relative_improvement"] for r in hetero]))
    lower = _bootstrap_lower(
        np.asarray([r["relative_improvement"] for r in hetero or validation]),
        int(config["path_screen"]["bootstrap_samples"]),
        int(config["path_screen"]["bootstrap_seed"]),
    )
    gate = (
        overall >= -float(config["path_screen"]["overall_max_degradation"])
        and (overall >= float(config["path_screen"]["overall_min_improvement"])
             or heterogeneous >= float(config["path_screen"]["heterogeneous_min_improvement"]))
        and lower > 0.0
    )
    test_summary = None
    if gate:
        test = [r for r in event_rows if r["partition"] == "test"]
        test_summary = {
            "count": len(test),
            "mean_relative_improvement": float(np.mean([r["relative_improvement"] for r in test])),
            "water_rmse_mm": float(np.sqrt(np.mean([r["water_rmse_mm"] ** 2 for r in test]))),
            "inhomogeneous_rmse_mm": float(np.sqrt(np.mean([r["inhomogeneous_rmse_mm"] ** 2 for r in test]))),
        }
        gate = gate and test_summary["mean_relative_improvement"] > 0.0
    decision = {
        "status": "PASS" if gate else "FAIL", "continue_to_level2": gate,
        "validation_count": len(validation), "validation_overall_improvement": overall,
        "validation_heterogeneous_count": len(hetero),
        "validation_heterogeneous_improvement": heterogeneous,
        "bootstrap_95_lower": lower, "test": test_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(decision_path, decision)
    return decision


def _mapping(image: np.ndarray, kind: str, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import gaussian_filter
    spacing = float(config["geometry"]["spacing_mm"])
    origin = -0.5 * (image.shape[0] - 1) * spacing
    coordinates = origin + np.arange(image.shape[0]) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    support = xx * xx + zz * zz <= float(config["geometry"]["radius_mm"]) ** 2
    rsp = gaussian_filter(np.maximum(image, 0).astype(np.float32),
                          float(config["mapping"]["gaussian_sigma_mm"]) / spacing)
    rsp[~support] = 0.0
    rscp = (
        published_rscp(rsp, config["mapping"])
        if kind == "published"
        else catalog_rscp(rsp, material_catalog(config))
    )
    rscp[~support] = 0.0
    return rsp, rscp


def _baseline_path(dataset: dict[str, Any], config: dict[str, Any]) -> Path:
    settings = {k: v for k, v in config["stage4"].items() if k not in {"frozen_at", "test_partition_opened"}}
    name = stage4.variant_name(settings)
    return Path(dataset["reconstruction_data"]) / "stage4" / "variants" / name / "recon" / f"epoch_{int(settings['epochs']):02d}.mhd"


def _initial_path(dataset: dict[str, Any]) -> Path:
    return Path(dataset["reconstruction_data"]) / "stage3" / "analytic" / "baseline_3sigma" / "recon.mhd"


def _evaluate(
    dataset: dict[str, Any], config: dict[str, Any], image_path: Path,
    projector, partition: str, device: int,
) -> dict[str, Any]:
    import cupy as cp
    cp.cuda.Device(device).use()
    image = cp.asarray(read_image_2d(image_path)[0])
    residuals = []
    for run_id in range(RUNS):
        pairs = paircuts.read_mhd(Path(dataset["preprocessing_data"]) / "pairs" / f"pairs{run_id:04d}.mhd")
        split = partition_masks(len(pairs), run_id, config["stage3"]["split"])[partition]
        accepted = accepted_mask(dataset, config, run_id, len(pairs))
        indices = np.flatnonzero(split & accepted)
        batch_size = int(config["geometry"]["batch_size"])
        for begin in range(0, len(indices), batch_size):
            selected = np.asarray(pairs[indices[begin:begin + batch_size]], np.float32)
            residuals.append(projector.residuals(image, make_batch(selected, dataset, config), .5 * run_id))
    values = np.concatenate(residuals)
    return {
        "dataset": dataset["name"], "partition": partition, "count": len(values),
        "rmse_mm": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
        "mae_mm": float(np.mean(np.abs(values))), "bias_mm": float(np.mean(values)),
        "abs_p99_mm": float(np.quantile(np.abs(values), .99)),
    }


def run_reconstruction(
    dataset: dict[str, Any], config: dict[str, Any], kind: str, mode: str,
    device: int, force: bool,
) -> dict[str, Any]:
    import cupy as cp
    from inhomogeneous_gpu import InhomogeneousGpuMlpProjector

    root = stage5_root(dataset) / mode / kind
    recon = root / "recon"
    if force and root.exists():
        shutil.rmtree(root)
    recon.mkdir(parents=True, exist_ok=True)
    snapshot = root / "config.json"
    digest = canonical_hash({"kind": kind, "mode": mode, "config": {k: v for k, v in config.items() if k not in {"stage3", "stage4"}}})
    if snapshot.is_file() and load_json(snapshot)["hash"] != digest:
        raise RuntimeError(f"Stage-5 configuration hash mismatch: {root}")
    if not snapshot.is_file():
        write_json(snapshot, {"hash": digest, "kind": kind, "mode": mode})
    history_path = root / "epoch_metrics.csv"
    history = read_csv(history_path)
    completed = max((int(r["epoch"]) for r in history), default=0)
    target = int(config["reconstruction"]["epochs"])
    if completed >= target:
        return dict(history[-1])
    source = recon / f"epoch_{completed:02d}.mhd" if completed else _initial_path(dataset)
    image_cpu, spacing3, origin3 = read_image_2d(source)
    image_cpu = np.array(image_cpu, np.float32, copy=True)
    spacing, origin = float(spacing3[0]), float(origin3[0])
    coords = origin + np.arange(len(image_cpu)) * spacing
    xx, zz = np.meshgrid(coords, coords)
    support_cpu = xx * xx + zz * zz <= float(config["geometry"]["radius_mm"]) ** 2
    image_cpu[~support_cpu] = 0
    np.maximum(image_cpu, 0, out=image_cpu)
    if not completed:
        write_image_2d(recon / "initial.mhd", image_cpu, spacing, origin)
    maps_path = root / "material_maps.npz"
    if maps_path.is_file():
        with np.load(maps_path) as saved:
            rsp, rscp = saved["rsp"], saved["rscp"]
    else:
        initial_image = np.asarray(read_image_2d(recon / "initial.mhd")[0])
        rsp, rscp = _mapping(initial_image, kind, config)
        np.savez_compressed(maps_path, rsp=rsp, rscp=rscp)
    cp.cuda.Device(device).use()
    image, support = cp.asarray(image_cpu), cp.asarray(support_cpu)
    projector = InhomogeneousGpuMlpProjector(
        int(config["geometry"]["grid_size"]), spacing,
        float(config["geometry"]["path_step_mm"]), float(config["geometry"]["radius_mm"]),
        rsp, rscp, int(config["path_screen"]["fixed_point_iterations"]),
    )
    counts = []
    for run_id in range(RUNS):
        pairs = paircuts.read_mhd(Path(dataset["preprocessing_data"]) / "pairs" / f"pairs{run_id:04d}.mhd")
        train = partition_masks(len(pairs), run_id, config["stage3"]["split"])["train"]
        counts.append(int(np.count_nonzero(train & accepted_mask(dataset, config, run_id, len(pairs)))))
    total_pairs, done_pairs = sum(counts) * target, sum(counts) * completed
    started = time.perf_counter()
    for epoch in range(completed, target):
        epoch_start = time.perf_counter()
        map_change = 0.0
        map_updated = False
        if mode == "alternating" and epoch + 1 in config["reconstruction"]["path_update_epochs"]:
            new_rsp, new_rscp = _mapping(cp.asnumpy(image), kind, config)
            damping = float(config["mapping"]["damping"])
            blended = damping * new_rscp + (1.0 - damping) * rscp
            map_change = float(np.sqrt(np.mean((blended - rscp) ** 2)))
            rsp, rscp = new_rsp, blended
            projector.update_maps(rsp, rscp)
            map_updated = True
        training_squared = training_valid = 0.0
        max_update = 0.0
        relaxation = float(config["reconstruction"]["relaxation"]) / (
            1.0 + float(config["reconstruction"]["relaxation_decay"]) * epoch)
        for subset in range(int(config["reconstruction"]["subsets"])):
            numerator, denominator = cp.zeros_like(image), cp.zeros_like(image)
            for run_id in range(subset, RUNS, int(config["reconstruction"]["subsets"])):
                pairs = paircuts.read_mhd(Path(dataset["preprocessing_data"]) / "pairs" / f"pairs{run_id:04d}.mhd")
                train = partition_masks(len(pairs), run_id, config["stage3"]["split"])["train"]
                indices = np.flatnonzero(train & accepted_mask(dataset, config, run_id, len(pairs)))
                bs = int(config["geometry"]["batch_size"])
                for begin in range(0, len(indices), bs):
                    selected = np.asarray(pairs[indices[begin:begin + bs]], np.float32)
                    metrics = projector.accumulate_loss(
                        image, make_batch(selected, dataset, config), .5 * run_id,
                        numerator, denominator, None)
                    training_squared += metrics["squared"]
                    training_valid += metrics["valid"]
                    done_pairs += len(selected)
            update = cp.where(denominator > 0, np.float32(relaxation) * numerator /
                              cp.maximum(denominator, np.float32(1e-20)), np.float32(0))
            image += update
            cp.maximum(image, 0, out=image)
            image *= support
            cp.cuda.Stream.null.synchronize()
            max_update = max(max_update, float(cp.max(cp.abs(update)).get()))
            elapsed = time.perf_counter() - started
            rate = (done_pairs - sum(counts) * completed) / elapsed if elapsed else 0
            eta = (total_pairs - done_pairs) / rate if rate else math.inf
            fraction = done_pairs / total_pairs
            update_state(
                status="RUNNING", level="level2" if mode == "fixed" else "level3",
                task=f"{mode} {kind}", dataset=dataset["name"], epoch=epoch + 1,
                target_epochs=target, subset=subset + 1,
                target_subsets=int(config["reconstruction"]["subsets"]), rate=rate,
                eta_seconds=eta, overall_fraction=.20 + .65 * fraction,
                gpu_memory_peak_bytes=int(cp.get_default_memory_pool().total_bytes()),
                elapsed_seconds=elapsed,
            )
            print(
                f"{dataset['name']} {mode}/{kind} epoch {epoch+1}/{target} "
                f"subset {subset+1:02d}/{config['reconstruction']['subsets']}: "
                f"rate={rate:,.0f} pairs/s ETA={format_duration(eta)}", flush=True)
        image, reg = proximal_regularize(
            image, support, kind="huber_tv",
            weight=float(config["reconstruction"]["regularization_weight"]),
            iterations=int(config["reconstruction"]["regularization_iterations"]),
            huber_delta=float(config["reconstruction"]["huber_delta"]),
            primal_step=float(config["reconstruction"]["primal_step"]),
            dual_step=float(config["reconstruction"]["dual_step"]),
        )
        checkpoint = recon / f"epoch_{epoch+1:02d}.mhd"
        write_image_2d(checkpoint, cp.asnumpy(image), spacing, origin)
        if map_updated:
            # Commit the new operator only after its matching image
            # checkpoint exists.  If an epoch is interrupted, rerunning it
            # starts from the previous committed image/map pair.
            temporary_maps = maps_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary_maps, rsp=rsp, rscp=rscp)
            temporary_maps.replace(maps_path)
        validation = _evaluate(dataset, config, checkpoint, projector, "validation", device)
        measured, details = stage4.scalar_metrics(dataset, {"stage3": config["stage3"]}, checkpoint)
        if details:
            write_csv(root / f"epoch_{epoch+1:02d}_details.csv", details)
        row = {
            "dataset": dataset["name"], "mode": mode, "mapping": kind, "epoch": epoch + 1,
            "training_rmse_mm": math.sqrt(training_squared / training_valid),
            "validation_rmse_mm": validation["rmse_mm"],
            "validation_abs_p99_mm": validation["abs_p99_mm"],
            "map_rscp_rms_change": map_change, "max_update": max_update,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "image_path": relative(checkpoint), **measured,
        }
        history.append(row)
        write_csv(history_path, history)
    return dict(history[-1])


def _baseline_metrics(dataset: dict[str, Any], config: dict[str, Any], device: int) -> dict[str, Any]:
    path = _baseline_path(dataset, config)
    water_config = {
        "stage3": config["stage3"],
        "filter": "baseline_3sigma",
        "grid": {
            "size": config["geometry"]["grid_size"],
            "spacing_mm": config["geometry"]["spacing_mm"],
            "path_step_mm": config["geometry"]["path_step_mm"],
            "phantom_radius_mm": config["geometry"]["radius_mm"],
            "batch_size": config["geometry"]["batch_size"],
        },
    }
    validation = stage4.evaluate_partition(
        dataset, water_config, path, "validation", device, RUNS, quiet=True)
    measured, _ = stage4.scalar_metrics(dataset, {"stage3": config["stage3"]}, path)
    return {"dataset": dataset["name"], "method": "water_stage4",
            "validation_rmse_mm": validation["rmse_mm"], "image_path": relative(path), **measured}


def _candidate_checks(candidate: dict[str, Any], baselines: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, bool]:
    dataset = candidate["dataset"]
    base = baselines[dataset]
    checks = {
        "validation_wepl": float(candidate["validation_rmse_mm"]) <= float(base["validation_rmse_mm"]) * (1 + float(config["selection"]["validation_rmse_max_degradation"]))
    }
    if dataset == "s2":
        checks["water_bias"] = (
            abs(float(candidate["water_bias_vs_effective_rsp"])) - abs(float(base["water_bias_vs_effective_rsp"]))
        ) * 100 <= float(config["selection"]["water_bias_max_degradation_percentage_points"])
    if dataset == "s4":
        mape_improvement_pp = (
            float(base["material_mape_non_air"]) - float(candidate["material_mape_non_air"])
        ) * 100
        max_error_improvement = (
            1.0
            - float(candidate["material_max_ape_non_air"])
            / float(base["material_max_ape_non_air"])
        )
        checks["material_improvement"] = (
            mape_improvement_pp
            >= float(config["selection"]["material_mape_min_improvement_percentage_points"])
            or max_error_improvement
            >= float(config["selection"]["material_max_error_min_relative_improvement"])
        )
    if dataset == "s5":
        checks["mtf50"] = float(candidate["fmtf50_mean_lp_per_mm"]) >= float(base["fmtf50_mean_lp_per_mm"]) * (1 - float(config["selection"]["mtf_max_relative_degradation"]))
        checks["mtf10"] = float(candidate["fmtf10_mean_lp_per_mm"]) >= float(base["fmtf10_mean_lp_per_mm"]) * (1 - float(config["selection"]["mtf_max_relative_degradation"]))
    return checks


def fixed_screen(datasets: list[dict[str, Any]], config: dict[str, Any], device: int, force: bool) -> dict[str, Any]:
    required = {"s2", "s4", "s5"}
    dev = [d for d in datasets if d["name"] in required]
    if {d["name"] for d in dev} != required:
        raise RuntimeError("fixed-screen requires s2,s4,s5")
    baselines = {d["name"]: _baseline_metrics(d, config, device) for d in dev}
    rows = []
    for kind in ("published", "catalog"):
        for dataset in dev:
            row = run_reconstruction(dataset, config, kind, "fixed", device, force)
            row["checks"] = _candidate_checks(row, baselines, config)
            rows.append(row)
    candidates = []
    for kind in ("published", "catalog"):
        current = [r for r in rows if r["mapping"] == kind]
        candidates.append({
            "mapping": kind,
            "eligible": all(all(r["checks"].values()) for r in current),
            "mean_validation_ratio": float(np.mean([
                float(r["validation_rmse_mm"]) / float(baselines[r["dataset"]]["validation_rmse_mm"])
                for r in current])),
            "datasets": current,
        })
    eligible = [r for r in candidates if r["eligible"]]
    result = {
        "status": "PASS" if eligible else "FAIL",
        "continue_to_level3": bool(eligible),
        "winner": min(eligible, key=lambda r: r["mean_validation_ratio"]) if eligible else None,
        "candidates": candidates, "baselines": list(baselines.values()),
    }
    write_json(QC_ROOT / "level2_decision.json", result)
    return result


def alternating_screen(datasets: list[dict[str, Any]], config: dict[str, Any], device: int, force: bool) -> dict[str, Any]:
    fixed = load_json(QC_ROOT / "level2_decision.json")
    kind = fixed["winner"]["mapping"]
    dev = [d for d in datasets if d["name"] in {"s2", "s4", "s5"}]
    baselines = {r["dataset"]: r for r in fixed["baselines"]}
    rows = []
    for dataset in dev:
        row = run_reconstruction(dataset, config, kind, "alternating", device, force)
        row["checks"] = _candidate_checks(row, baselines, config)
        rows.append(row)
    fixed_ratio = float(fixed["winner"]["mean_validation_ratio"])
    alternating_ratio = float(np.mean([
        float(r["validation_rmse_mm"]) / float(baselines[r["dataset"]]["validation_rmse_mm"])
        for r in rows]))
    eligible = all(all(r["checks"].values()) for r in rows)
    winner = "alternating" if eligible and alternating_ratio < fixed_ratio else "fixed"
    result = {
        "status": "PASS", "winner": winner, "mapping": kind,
        "fixed_validation_ratio": fixed_ratio, "alternating_validation_ratio": alternating_ratio,
        "alternating_eligible": eligible, "datasets": rows,
    }
    write_json(QC_ROOT / "level3_decision.json", result)
    write_json(QC_ROOT / "frozen_final.json", {
        "mode": winner, "mapping": kind, "epochs": int(config["reconstruction"]["epochs"]),
        "frozen_at": datetime.now().isoformat(timespec="seconds"), "test_partition_opened": False,
    })
    return result


def confirm(datasets: list[dict[str, Any]], config: dict[str, Any], device: int, force: bool) -> dict[str, Any]:
    frozen_path = QC_ROOT / "frozen_final.json"
    if not frozen_path.is_file():
        raise RuntimeError("Stage-5 method must be frozen before test confirmation")
    frozen = load_json(frozen_path)
    rows, images = [], []
    frozen["test_partition_opened"] = True
    write_json(frozen_path, frozen)
    from inhomogeneous_gpu import InhomogeneousGpuMlpProjector
    for dataset in datasets:
        candidate = run_reconstruction(dataset, config, frozen["mapping"], frozen["mode"], device, force)
        candidate_path = REPOSITORY_ROOT / candidate["image_path"]
        _image, spacing, _ = read_image_2d(candidate_path)
        maps_path = (
            stage5_root(dataset) / frozen["mode"] / frozen["mapping"]
            / "material_maps.npz"
        )
        with np.load(maps_path) as saved:
            rsp, rscp = saved["rsp"], saved["rscp"]
        projector = InhomogeneousGpuMlpProjector(
            int(config["geometry"]["grid_size"]), float(spacing[0]),
            float(config["geometry"]["path_step_mm"]), float(config["geometry"]["radius_mm"]),
            rsp, rscp, int(config["path_screen"]["fixed_point_iterations"]))
        for method, path in (("water_stage4", _baseline_path(dataset, config)), ("stage5", candidate_path)):
            if method == "water_stage4":
                water_config = {
                    "stage3": config["stage3"],
                    "filter": "baseline_3sigma",
                    "grid": {
                        "size": config["geometry"]["grid_size"],
                        "spacing_mm": config["geometry"]["spacing_mm"],
                        "path_step_mm": config["geometry"]["path_step_mm"],
                        "phantom_radius_mm": config["geometry"]["radius_mm"],
                        "batch_size": config["geometry"]["batch_size"],
                    },
                }
                test = stage4.evaluate_partition(
                    dataset, water_config,
                    path, "test", device, RUNS, quiet=True)
            else:
                test = _evaluate(dataset, config, path, projector, "test", device)
            rows.append({"method": method, **test})
            measured, _ = stage4.scalar_metrics(dataset, {"stage3": config["stage3"]}, path)
            images.append({"dataset": dataset["name"], "method": method, "image_path": relative(path), **measured})
    write_csv(QC_ROOT / "confirmation_test_wepl.csv", rows)
    write_csv(QC_ROOT / "confirmation_image_metrics.csv", images)
    by = {(r["dataset"], r["method"]): r for r in rows}
    by_image = {(r["dataset"], r["method"]): r for r in images}
    safety = {
        "individual_test_wepl": all(
            float(by[(d["name"], "stage5")]["rmse_mm"])
            <= float(by[(d["name"], "water_stage4")]["rmse_mm"]) * 1.005
            for d in datasets
        ),
        "s2_s3_water_bias": all(
            (
                abs(float(by_image[(name, "stage5")]["water_bias_vs_effective_rsp"]))
                - abs(float(by_image[(name, "water_stage4")]["water_bias_vs_effective_rsp"]))
            ) * 100.0
            <= float(config["selection"]["water_bias_max_degradation_percentage_points"])
            for name in ("s2", "s3")
        ),
        "s4_material": (
            float(by_image[("s4", "stage5")]["material_mape_non_air"])
            <= float(by_image[("s4", "water_stage4")]["material_mape_non_air"])
        ),
        "s5_mtf50": (
            float(by_image[("s5", "stage5")]["fmtf50_mean_lp_per_mm"])
            >= float(by_image[("s5", "water_stage4")]["fmtf50_mean_lp_per_mm"])
            * (1.0 - float(config["selection"]["mtf_max_relative_degradation"]))
        ),
        "s5_mtf10": (
            float(by_image[("s5", "stage5")]["fmtf10_mean_lp_per_mm"])
            >= float(by_image[("s5", "water_stage4")]["fmtf10_mean_lp_per_mm"])
            * (1.0 - float(config["selection"]["mtf_max_relative_degradation"]))
        ),
    }
    safe = all(safety.values())
    result = {
        "status": "PASS" if safe else "FAIL", "decision": "PROMOTE_STAGE5" if safe else "RETAIN_STAGE4",
        "safety": safety, "method": frozen,
        "mean_rmse_improvement": float(1 - np.mean([by[(d["name"], "stage5")]["rmse_mm"] for d in datasets]) /
                                       np.mean([by[(d["name"], "water_stage4")]["rmse_mm"] for d in datasets])),
    }
    write_json(QC_ROOT / "confirmation_summary.json", result)
    return result


def report(config: dict[str, Any]) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = QC_ROOT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    event_metrics_path = (
        REPOSITORY_ROOT / config["truth_pilot"]["cache_data"]
        / "path_event_metrics.csv"
    )
    event_rows = read_csv(event_metrics_path)
    if event_rows:
        water = np.asarray([float(r["water_rmse_mm"]) for r in event_rows])
        inhom = np.asarray([float(r["inhomogeneous_rmse_mm"]) for r in event_rows])
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.hexbin(water, inhom, gridsize=70, bins="log", mincnt=1)
        limit = float(np.quantile(np.r_[water, inhom], .995))
        ax.plot([0, limit], [0, limit], "k--", lw=1)
        ax.set(xlabel="Water MLP path RMSE (mm)", ylabel="Inhomogeneous MLP path RMSE (mm)",
               xlim=(0, limit), ylim=(0, limit))
        fig.tight_layout()
        fig.savefig(assets / "path_error_comparison.png", dpi=180)
        plt.close(fig)
    depth_rows = read_csv(QC_ROOT / "path_depth_metrics.csv")
    if depth_rows:
        depths = sorted({float(row["depth_mm"]) for row in depth_rows})
        water_depth, inhom_depth = [], []
        for depth in depths:
            current = [row for row in depth_rows if float(row["depth_mm"]) == depth]
            count = sum(int(row["count"]) for row in current)
            water_depth.append(math.sqrt(sum(
                int(row["count"]) * float(row["water_rms_mm"]) ** 2
                for row in current
            ) / count))
            inhom_depth.append(math.sqrt(sum(
                int(row["count"]) * float(row["inhomogeneous_rms_mm"]) ** 2
                for row in current
            ) / count))
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        ax.plot(depths, water_depth, label="Water MLP")
        ax.plot(depths, inhom_depth, label="Truth-map inhomogeneous MLP")
        ax.set(xlabel="Scanner depth z (mm)", ylabel="Lateral RMS error (mm)")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(assets / "path_error_by_depth.png", dpi=180)
        plt.close(fig)
    level1 = load_json(QC_ROOT / "level1_decision.json") if (QC_ROOT / "level1_decision.json").is_file() else None
    level2 = load_json(QC_ROOT / "level2_decision.json") if (QC_ROOT / "level2_decision.json").is_file() else None
    level3 = load_json(QC_ROOT / "level3_decision.json") if (QC_ROOT / "level3_decision.json").is_file() else None
    confirmation = load_json(QC_ROOT / "confirmation_summary.json") if (QC_ROOT / "confirmation_summary.json").is_file() else None
    confirmation_images = read_csv(QC_ROOT / "confirmation_image_metrics.csv")
    if confirmation_images:
        write_csv(QC_ROOT / "reconstruction_metrics.csv", confirmation_images)
        s4 = {
            row["method"]: row
            for row in confirmation_images
            if row["dataset"] == "s4"
        }
        if {"water_stage4", "stage5"} <= set(s4):
            water = np.asarray(read_image_2d(REPOSITORY_ROOT / s4["water_stage4"]["image_path"])[0])
            candidate = np.asarray(read_image_2d(REPOSITORY_ROOT / s4["stage5"]["image_path"])[0])
            fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5), constrained_layout=True)
            for ax, image, title in (
                (axes[0], water, "Stage 4 water MLP"),
                (axes[1], candidate, "Stage 5 selected MLP"),
            ):
                shown = ax.imshow(image, cmap="viridis", vmin=0, vmax=2.2,
                                  extent=(-105, 105, 105, -105))
                ax.set(title=title, xlabel="x (mm)", ylabel="z (mm)")
            difference = candidate - water
            bound = max(float(np.quantile(np.abs(difference), .995)), 1e-4)
            axes[2].imshow(difference, cmap="coolwarm", vmin=-bound, vmax=bound,
                           extent=(-105, 105, 105, -105))
            axes[2].set(title="Stage 5 - Stage 4", xlabel="x (mm)", ylabel="z (mm)")
            fig.colorbar(shown, ax=axes[:2], label="RSP", shrink=.82)
            fig.savefig(assets / "stage5_reconstruction_comparison.png", dpi=180)
            plt.close(fig)
    final = confirmation["decision"] if confirmation else (
        "RETAIN_STAGE4_LEVEL1_FAIL" if level1 and level1["status"] == "FAIL"
        else "RETAIN_STAGE4_LEVEL2_FAIL" if level2 and level2["status"] == "FAIL"
        else "INCOMPLETE")
    if level1 is not None:
        reconstruction_figure = (
            "\n## 重建对比\n\n"
            "![Stage 5 reconstruction comparison]"
            "(assets/stage5_reconstruction_comparison.png)\n"
            if (assets / "stage5_reconstruction_comparison.png").is_file()
            else ""
        )
        summary = f"""# 阶段5：非均匀MLP与迭代MLP

## 执行状态

- 最终决定：`{final}`
- Level 1真实轨迹门槛：`{level1['status'] if level1 else '未执行'}`
- Level 2固定非均匀MLP：`{level2['status'] if level2 else '未执行'}`
- Level 3交替更新MLP：`{level3['status'] if level3 else '未执行'}`
- 锁定测试：`{confirmation['status'] if confirmation else '未执行'}`

## 路径上限实验

真实轨迹pilot使用72个角度和五种异质材料。验证集整体路径改善为
`{level1['validation_overall_improvement']*100:.3f}%`，强异质质子改善为
`{level1['validation_heterogeneous_improvement']*100:.3f}%`，配对bootstrap
95%下限为`{level1['bootstrap_95_lower']*100:.3f}%`。

![Path error comparison](assets/path_error_comparison.png)

![Path error by depth](assets/path_error_by_depth.png)
{reconstruction_figure}

## 结论

自动门控严格遵循预注册阈值。Level 1失败表示真实材料先验本身未证明路径收益；
Level 2失败表示路径上限未稳定转化为图像收益。只有锁定测试安全时才允许晋升。
"""
    else:
        summary = "# 阶段5：非均匀MLP与迭代MLP\n\n尚未完成Level 1。\n"
    (QC_ROOT / "stage5_summary.md").write_text(summary, encoding="utf-8")
    decision = {
        "status": "PASS", "final_decision": final, "level1": level1,
        "level2": level2, "level3": level3, "confirmation": confirmation,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(QC_ROOT / "stage5_decision.json", decision)
    return decision


def status() -> dict[str, Any]:
    state = load_json(STATE_PATH) if STATE_PATH.is_file() else {"status": "NOT_STARTED"}
    print("Stage 5 aggregate progress (read-only)")
    print("=" * 76)
    for key in ("status", "level", "task", "dataset"):
        if key in state:
            print(f"{key:>20}: {state[key]}")
    if "overall_fraction" in state:
        print(f"{'overall':>20}: {100*float(state['overall_fraction']):.1f}%")
    if "completed" in state and "total" in state:
        print(f"{'items':>20}: {state['completed']}/{state['total']}")
    if "trajectories" in state:
        print(f"{'trajectories':>20}: {int(state['trajectories']):,}")
    if "epoch" in state:
        print(f"{'epoch':>20}: {state['epoch']}/{state['target_epochs']}")
    if "subset" in state:
        print(f"{'subset':>20}: {state['subset']}/{state['target_subsets']}")
    if "rate" in state:
        print(f"{'rate':>20}: {float(state['rate']):,.0f} protons/s")
    if "eta_seconds" in state:
        print(f"{'current ETA':>20}: {format_duration(float(state['eta_seconds']))}")
    print(f"{'planned full runtime':>20}: 19:00:00--35:00:00 if all gates pass")
    if "gpu_memory_peak_bytes" in state:
        print(f"{'GPU allocated':>20}: {float(state['gpu_memory_peak_bytes'])/2**30:.2f} GiB")
    if "elapsed_seconds" in state:
        print(f"{'elapsed':>20}: {format_duration(float(state['elapsed_seconds']))}")
    print(f"{'updated':>20}: {state.get('updated_at', '-')}")
    for filename, label in (
        ("level1_decision.json", "Level 1"), ("level2_decision.json", "Level 2"),
        ("level3_decision.json", "Level 3"), ("confirmation_summary.json", "confirmation"),
    ):
        path = QC_ROOT / filename
        if path.is_file():
            decision = load_json(path)
            extra = decision.get("winner")
            if isinstance(extra, dict):
                extra = extra.get("mapping") or extra.get("mode")
            if filename == "level3_decision.json":
                extra = decision.get("winner")
            text = decision.get("status")
            if extra:
                text += f" ({extra})"
        else:
            text = "waiting"
        print(f"{label:>20}: {text}")
    return state


def smoke(config: dict[str, Any], device: int) -> dict[str, Any]:
    maps = truth_maps(config, .5)
    catalog = material_catalog(config)
    if not all(np.isfinite([v["rscp"] for v in catalog.values()])):
        raise RuntimeError("non-finite material catalogue")
    # GPU construction compiles every new kernel without starting a reconstruction.
    from inhomogeneous_gpu import InhomogeneousGpuMlpProjector
    import cupy as cp
    cp.cuda.Device(device).use()
    projector = InhomogeneousGpuMlpProjector(
        400, .5, .5, 100, maps.rsp, maps.rscp, 1
    )
    n = 2
    batch = {
        "position_in": np.tile([0.0, 0.0, -110.0], (n, 1)).astype(np.float32),
        "position_out": np.tile([0.0, 0.0, 110.0], (n, 1)).astype(np.float32),
        "direction_in": np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
        "direction_out": np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32),
        "energy_in": np.full(n, 200.0, np.float32),
        "energy_out": np.full(n, 100.0, np.float32),
        "wepl_mm": np.full(n, 100.0, np.float32),
    }
    residual = projector.residuals(cp.zeros((400, 400), cp.float32), batch, 0.0)
    if len(residual) != n or not np.isfinite(residual).all():
        raise RuntimeError("Stage-5 GPU path/forward smoke test failed")
    _blocks, _threads, pixels, weights, _row_sum, _valid = (
        projector._device_paths(batch, 0.0)
    )
    rng = np.random.default_rng(20260713)
    image = cp.asarray(rng.normal(size=400 * 400).astype(np.float32))
    row = cp.asarray(rng.normal(size=n).astype(np.float32))
    valid_pixel = pixels >= 0
    safe_pixels = cp.where(valid_pixel, pixels, 0)
    forward = cp.sum(
        cp.where(valid_pixel, image[safe_pixels] * weights, 0.0)
        .reshape(n, -1),
        axis=1,
    )
    back = cp.zeros_like(image)
    repeated = cp.repeat(row, projector.samples * 4)
    cp.add.at(
        back,
        safe_pixels[valid_pixel],
        (weights * repeated)[valid_pixel],
    )
    lhs = float(cp.dot(forward, row).get())
    rhs = float(cp.dot(image, back).get())
    adjoint_relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12)
    if adjoint_relative_error > 2e-5:
        raise RuntimeError(
            f"Stage-5 GPU adjoint test failed: {adjoint_relative_error:g}"
        )
    # Two-angle real-data closure: build nonuniform paths for the first and
    # last acquisition angles, apply one paired OS-SART numerator/denominator
    # update, and verify the support invariant.
    dataset = stage3.dataset_record(config["stage3"], "s2")
    initial_cpu, spacing3, origin3 = read_image_2d(_initial_path(dataset))
    initial_cpu = np.array(initial_cpu, np.float32, copy=True)
    real_rsp, real_rscp = _mapping(initial_cpu, "published", config)
    real_projector = InhomogeneousGpuMlpProjector(
        int(config["geometry"]["grid_size"]),
        float(spacing3[0]),
        float(config["geometry"]["path_step_mm"]),
        float(config["geometry"]["radius_mm"]),
        real_rsp,
        real_rscp,
        1,
    )
    real_image = cp.asarray(initial_cpu)
    numerator, denominator = cp.zeros_like(real_image), cp.zeros_like(real_image)
    real_valid = 0
    for run_id in (0, 719):
        pairs = paircuts.read_mhd(
            Path(dataset["preprocessing_data"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        selected = np.asarray(pairs[:16], np.float32)
        metrics = real_projector.accumulate_loss(
            real_image,
            make_batch(selected, dataset, config),
            0.5 * run_id,
            numerator,
            denominator,
            None,
        )
        real_valid += int(metrics["valid"])
    update = cp.where(
        denominator > 0,
        numerator / cp.maximum(denominator, cp.float32(1e-20)),
        cp.float32(0),
    )
    coordinates = float(origin3[0]) + np.arange(initial_cpu.shape[0]) * float(spacing3[0])
    xx, zz = np.meshgrid(coordinates, coordinates)
    support_cpu = xx * xx + zz * zz <= float(config["geometry"]["radius_mm"]) ** 2
    updated = (real_image + update) * cp.asarray(support_cpu)
    if (
        real_valid < 1
        or not bool(cp.isfinite(updated).all().get())
        or int(cp.count_nonzero(updated[cp.asarray(~support_cpu)]).get()) != 0
    ):
        raise RuntimeError("Stage-5 two-angle real-data closure failed")
    result = {
        "status": "PASS", "samples": projector.samples,
        "synthetic_residuals_mm": residual.tolist(),
        "adjoint_relative_error": adjoint_relative_error,
        "real_data_angles": [0, 719],
        "real_data_valid_protons": real_valid,
        "real_data_update_max_abs": float(cp.max(cp.abs(update)).get()),
        "materials": catalog,
    }
    del (
        real_projector, real_image, numerator, denominator, update, updated,
        projector, image, back, pixels, weights,
    )
    cp.get_default_memory_pool().free_all_blocks()
    write_json(QC_ROOT / "smoke.json", result)
    return result


def all_actions(datasets, config, jobs, device, force) -> dict[str, Any]:
    start = time.perf_counter()
    update_state(
        status="RUNNING", level="preflight", task="GPU smoke test",
        overall_fraction=0.0, planned_min_seconds=19 * 3600,
        planned_max_seconds=35 * 3600,
    )
    smoke(config, device)
    audit(config, force)
    prepare_truth(config, jobs)
    level1 = path_screen(config, force)
    if not level1["continue_to_level2"]:
        result = report(config)
        elapsed = time.perf_counter() - start
        update_state(status="COMPLETE", level="level1", task="scientific gate stopped workflow",
                     overall_fraction=1.0, elapsed_seconds=elapsed,
                     final_decision=result["final_decision"])
        result["workflow_elapsed_seconds_this_call"] = elapsed
        result["resource_state"] = load_json(STATE_PATH)
        write_json(QC_ROOT / "stage5_decision.json", result)
        return result
    level2 = fixed_screen(datasets, config, device, force)
    if not level2["continue_to_level3"]:
        result = report(config)
        elapsed = time.perf_counter() - start
        update_state(status="COMPLETE", level="level2", task="scientific gate stopped workflow",
                     overall_fraction=1.0, elapsed_seconds=elapsed,
                     final_decision=result["final_decision"])
        result["workflow_elapsed_seconds_this_call"] = elapsed
        result["resource_state"] = load_json(STATE_PATH)
        write_json(QC_ROOT / "stage5_decision.json", result)
        return result
    alternating_screen(datasets, config, device, force)
    confirm(datasets, config, device, force)
    result = report(config)
    elapsed = time.perf_counter() - start
    update_state(status="COMPLETE", level="complete", task="all actions complete",
                 overall_fraction=1.0, elapsed_seconds=elapsed,
                 final_decision=result["final_decision"])
    result["workflow_elapsed_seconds_this_call"] = elapsed
    result["resource_state"] = load_json(STATE_PATH)
    write_json(QC_ROOT / "stage5_decision.json", result)
    return result


def main() -> None:
    args = parse_args()
    config = load_config()
    datasets = datasets_from(args.datasets, config)
    if args.action == "status":
        status()
        return
    if args.force and args.action == "all":
        for path in (QC_ROOT, REPOSITORY_ROOT / config["truth_pilot"]["cache_data"]):
            shutil.rmtree(path, ignore_errors=True)
        for dataset in datasets:
            shutil.rmtree(stage5_root(dataset), ignore_errors=True)
    try:
        if args.action == "all":
            result = all_actions(datasets, config, args.jobs, args.device, args.force)
        elif args.action == "audit":
            result = audit(config, args.force)
        elif args.action == "prepare-truth":
            result = prepare_truth(config, args.jobs)
        elif args.action == "path-screen":
            result = path_screen(config, args.force)
        elif args.action == "fixed-screen":
            result = fixed_screen(datasets, config, args.device, args.force)
        elif args.action == "alternating-screen":
            result = alternating_screen(datasets, config, args.device, args.force)
        elif args.action == "confirm":
            result = confirm(datasets, config, args.device, args.force)
        elif args.action == "report":
            result = report(config)
        else:
            result = smoke(config, args.device)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    except Exception as error:
        update_state(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
