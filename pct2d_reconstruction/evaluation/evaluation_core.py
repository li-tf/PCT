#!/usr/bin/env python3
"""Implementation of immutable baseline snapshots, splits, and RSP metrics."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / "iterative_reconstruction"))

from common import path_for  # noqa: E402
from analytic_reconstruction import rsp_metrics  # noqa: E402
from mhd_io import read_header, read_pairs  # noqa: E402
from physics import energies_to_wepl_vectorized, make_vectorized_wepl_lut  # noqa: E402


SCHEMA_VERSION = 1
SPLIT_NAME = "baseline90_10"
CHECKPOINTS = (
    ("analytic_nohann", "analytic", 0, "analytic/recon/recon_ddb_nohann.mhd"),
    ("iterative_initial", "iterative", 0, "iterative/recon/initial.mhd"),
    ("iterative_epoch_01", "iterative", 1, "iterative/recon/epoch_01.mhd"),
    ("iterative_epoch_02", "iterative", 2, "iterative/recon/epoch_02.mhd"),
    ("iterative_epoch_03", "iterative", 3, "iterative/recon/epoch_03.mhd"),
)


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def mhd_metadata(path: Path) -> dict[str, object]:
    values = read_header(path)
    result: dict[str, object] = {
        "dim_size": [int(item) for item in values["DimSize"].split()],
        "element_type": values["ElementType"],
        "channels": int(values.get("ElementNumberOfChannels", "1")),
        "element_data_file": values["ElementDataFile"],
    }
    if "ElementSpacing" in values:
        result["spacing"] = [float(item) for item in values["ElementSpacing"].split()]
    origin = values.get("Offset", values.get("Origin"))
    if origin is not None:
        result["origin"] = [float(item) for item in origin.split()]
    return result


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr.strip()}"


def _source_files() -> list[Path]:
    roots = [
        CODE_ROOT / "experiments",
        CODE_ROOT / "simulation" / "simulation0716",
        CODE_ROOT / "preprocessing",
        CODE_ROOT / "analytic_reconstruction",
        CODE_ROOT / "iterative_reconstruction",
        CODE_ROOT / "evaluation",
    ]
    suffixes = {".py", ".json", ".xml", ".bat", ".ps1", ".md"}
    files: list[Path] = [CODE_ROOT / "common.py"]
    for root in roots:
        files.extend(
            path for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in suffixes
            and "__pycache__" not in path.parts
            and "baselines" not in path.parts
            and "qc" not in path.parts
        )
    return sorted(set(files))


def _baseline_artifacts(experiment: dict) -> list[tuple[str, Path]]:
    groups: list[tuple[str, Path]] = []
    simulation_data = path_for(experiment, "simulation_data")
    preprocessing_data = path_for(experiment, "preprocessing_data")
    reconstruction_data = path_for(experiment, "reconstruction_data")
    for path in sorted(simulation_data.rglob("*.root")):
        groups.append(("simulation_root", path))
    for directory, label in (
        ("pairs", "paired_protons"),
        ("pairs_filtered", "filtered_protons"),
        ("projections_ddb", "ddb_projection"),
    ):
        for path in sorted((preprocessing_data / directory).glob("*")):
            if path.is_file() and path.suffix.lower() in {".mhd", ".raw"}:
                groups.append((label, path))
    for path in sorted(reconstruction_data.rglob("*")):
        if path.is_file():
            groups.append(("reconstruction", path))
    return groups


def _qc_files(experiment_id: str) -> list[Path]:
    paths: list[Path] = []
    for stage in ("simulation/simulation0716", "preprocessing", "analytic_reconstruction", "iterative_reconstruction"):
        root = CODE_ROOT / stage / "qc"
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(paths)


def _environment_snapshot() -> dict[str, object]:
    result: dict[str, object] = {
        "captured_at": utc_timestamp(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
    }
    try:
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        result["cupy"] = cp.__version__
        result["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
        result["cuda_runtime"] = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception as error:  # CUDA is required only by validation-WEPL evaluation.
        result["gpu_probe_error"] = str(error)
    return result


def freeze_baseline(experiment: dict, baseline_dir: Path, force: bool) -> dict[str, object]:
    output = baseline_dir / "baseline_manifest.json"
    if output.exists() and not force:
        raise FileExistsError(f"baseline manifest exists: {output}; use --force to replace it")
    artifacts = _baseline_artifacts(experiment)
    if not artifacts:
        raise FileNotFoundError("no baseline artifacts found")
    total_bytes = sum(path.stat().st_size for _, path in artifacts)
    print(f"freeze: hashing {len(artifacts):,} artifacts ({total_bytes / 2**30:.2f} GiB)", flush=True)
    records: list[dict[str, object]] = []
    completed_bytes = 0
    last_report = time.monotonic()
    started = time.monotonic()
    for index, (category, path) in enumerate(artifacts, 1):
        stat = path.stat()
        record: dict[str, object] = {
            "category": category,
            "path": relative(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
        if path.suffix.lower() == ".mhd":
            record["mhd"] = mhd_metadata(path)
        records.append(record)
        completed_bytes += stat.st_size
        now = time.monotonic()
        if now - last_report >= 5.0 or index == len(artifacts):
            elapsed = now - started
            rate = completed_bytes / elapsed if elapsed else 0.0
            eta = (total_bytes - completed_bytes) / rate if rate else math.inf
            print(
                f"freeze: {index:,}/{len(artifacts):,} files, {completed_bytes/total_bytes:6.2%}, "
                f"{rate/2**20:,.1f} MiB/s, ETA={eta/60:.1f} min",
                flush=True,
            )
            last_report = now

    source_records = [
        {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in _source_files()
    ]
    qc_records = [
        {"path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in _qc_files(str(experiment["experiment"]))
    ]
    planned = experiment["iterative"]
    run_summary_path = CODE_ROOT / "iterative_reconstruction" / "qc" / f"results{experiment['experiment']}" / "run_summary.json"
    executed_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN",
        "created_at": utc_timestamp(),
        "experiment": str(experiment["experiment"]),
        "truth_quantity": "RSP at 200 MeV",
        "analytic_baseline": "DDB-FDK no-Hann",
        "iterative_baseline": "GPU MLP OS-SART with Huber-TV, epoch 3",
        "planned_iterative_config": planned,
        "executed_iterative_config": executed_summary["config"],
        "executed_iterative_run": {
            "status": executed_summary["status"],
            "started": executed_summary["started"],
            "stopped": executed_summary["stopped"],
            "elapsed_seconds": executed_summary["elapsed_seconds"],
            "gpu": executed_summary["gpu"],
            "pairs_per_epoch": executed_summary["pairs_per_epoch"],
        },
        "configuration_difference": {
            "planned_epochs": int(planned["epochs"]),
            "executed_epochs": int(executed_summary["config"]["epochs"]),
            "authoritative": "executed_iterative_config",
        },
        "git": {
            "commit": _git(["rev-parse", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "status_porcelain": _git(["status", "--short"]),
        },
        "environment": _environment_snapshot(),
        "artifact_count": len(records),
        "artifact_bytes": total_bytes,
        "freeze_resources": {
            "elapsed_seconds": time.monotonic() - started,
            "throughput_mib_per_second": total_bytes / max(time.monotonic() - started, 1.0e-12) / 2**20,
        },
        "artifacts": records,
        "source_files": source_records,
        "qc_files": qc_records,
    }
    write_json(output, manifest)
    print(f"freeze: wrote {output}", flush=True)
    return manifest


def splitmix64(values: np.ndarray, run_id: int, seed: int) -> np.ndarray:
    """Versioned uint64 mixer; wraparound is intentional and deterministic."""
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = values.astype(np.uint64, copy=False)
    run_key = np.uint64((int(run_id) * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF)
    x = (x ^ run_key) & mask
    x = (x ^ np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)) & mask
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & mask
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & mask
    return (x ^ (x >> np.uint64(31))) & mask


def pair_count(path: Path) -> int:
    header = read_header(path)
    dimensions = [int(value) for value in header["DimSize"].split()]
    if dimensions[0] != 5:
        raise ValueError(f"unexpected pair layout: {path}")
    return dimensions[1]


def make_split(experiment: dict, baseline_dir: Path, config: dict, force: bool) -> dict[str, object]:
    preprocessing = path_for(experiment, "preprocessing_data")
    pair_dir = preprocessing / "pairs_filtered"
    split_dir = preprocessing / "splits" / SPLIT_NAME
    manifest_path = baseline_dir / "split_manifest.json"
    run_csv = baseline_dir / "split_runs.csv"
    if (manifest_path.exists() or split_dir.exists()) and not force:
        raise FileExistsError(f"split outputs exist; use --force to replace {SPLIT_NAME}")
    split_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old in split_dir.glob("validation_mask_*.bin"):
            old.unlink()
    specification = config["split"]
    seed = int(specification["seed"])
    modulus = int(specification["validation_modulus"])
    remainder = int(specification["validation_remainder"])
    bitorder = str(specification["bit_order"])
    rows: list[dict[str, object]] = []
    total = validation_total = 0
    started = time.monotonic()
    for run_id in range(int(experiment["acquisition"]["projections"])):
        header_path = pair_dir / f"pairs{run_id:04d}.mhd"
        count = pair_count(header_path)
        packed_parts: list[np.ndarray] = []
        validation_count = 0
        # Chunk boundaries are byte-aligned so concatenating packbits is exact.
        chunk = 1_000_000
        chunk -= chunk % 8
        for begin in range(0, count, chunk):
            end = min(begin + chunk, count)
            index = np.arange(begin, end, dtype=np.uint64)
            selected = splitmix64(index, run_id, seed) % np.uint64(modulus) == np.uint64(remainder)
            validation_count += int(selected.sum())
            packed_parts.append(np.packbits(selected, bitorder=bitorder))
        packed = np.concatenate(packed_parts) if packed_parts else np.empty(0, dtype=np.uint8)
        mask_path = split_dir / f"validation_mask_{run_id:04d}.bin"
        temporary = mask_path.with_name(mask_path.name + ".tmp")
        packed.tofile(temporary)
        temporary.replace(mask_path)
        train_count = count - validation_count
        if validation_count == 0 or train_count == 0:
            raise RuntimeError(f"empty train or validation partition for RunID {run_id}")
        rows.append({
            "run_id": run_id,
            "total_count": count,
            "train_count": train_count,
            "validation_count": validation_count,
            "validation_fraction": validation_count / count,
            "mask_bytes": mask_path.stat().st_size,
            "mask_sha256": sha256_file(mask_path),
            "mask_path": relative(mask_path),
        })
        total += count
        validation_total += validation_count
        if (run_id + 1) % 40 == 0 or run_id == int(experiment["acquisition"]["projections"]) - 1:
            elapsed = time.monotonic() - started
            print(
                f"split: {run_id+1:03d}/{experiment['acquisition']['projections']} angles, "
                f"{total:,} pairs, validation={validation_total/total:.4%}, elapsed={elapsed:.1f}s",
                flush=True,
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "created_at": utc_timestamp(),
        "experiment": str(experiment["experiment"]),
        "name": SPLIT_NAME,
        "identity": specification["identity"],
        "algorithm": specification["algorithm"],
        "seed": seed,
        "validation_modulus": modulus,
        "validation_remainder": remainder,
        "validation_rule": f"splitmix64_v1(RunID, filtered_row_index, {seed}) % {modulus} == {remainder}",
        "bit_order": bitorder,
        "angle_count": len(rows),
        "total_count": total,
        "train_count": total - validation_total,
        "validation_count": validation_total,
        "validation_fraction": validation_total / total,
        "all_angles_have_train_and_validation": True,
        "mask_directory": relative(split_dir),
        "runs_csv": relative(run_csv),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_csv(run_csv, rows)
    write_json(manifest_path, manifest)
    return manifest


def _truth_and_centers(experiment: dict):
    reconstruction = path_for(experiment, "reconstruction_data")
    simulation_code = path_for(experiment, "simulation_code")
    truth_path = reconstruction / "analytic" / "truth" / "truth_rsp_200mev.mhd"
    truth, x, z, meta = rsp_metrics.read_mhd(truth_path)
    definition = json.loads((simulation_code / "truth_geometry_definition.json").read_text(encoding="utf-8"))
    return truth, x, z, meta, definition["geometry"]["insert_centers_xz_mm"]


def image_metric_rows(experiment: dict) -> list[dict[str, object]]:
    reconstruction = path_for(experiment, "reconstruction_data")
    truth, truth_x, truth_z, _, centers = _truth_and_centers(experiment)
    rows: list[dict[str, object]] = []
    for checkpoint, method, epoch, relative_path in CHECKPOINTS:
        path = reconstruction / relative_path
        image, x, z, _ = rsp_metrics.read_mhd(path)
        if not (np.array_equal(x, truth_x) and np.array_equal(z, truth_z)):
            raise ValueError(f"truth grid differs from {path}")
        if not np.isfinite(image).all():
            raise ValueError(f"non-finite reconstruction: {path}")
        # Supplying RSP truth for both mask parameters prevents RED from entering
        # the evaluation definition while preserving the validated ROI code.
        values, _ = rsp_metrics.metrics_for(image, truth, truth, x, z, centers)
        edges = rsp_metrics.aluminium_edge_widths(image, x, z, centers)
        widths = np.array([
            item["width_10_90_mm"] for item in edges
            if item["distance_from_isocenter_mm"] > 0.0 and item["valid"]
        ])
        inner = np.array([item["inner_value"] for item in edges])
        xx, zz = np.meshgrid(x, z)
        outside_support = np.hypot(xx, zz) > 100.0
        rows.append({
            "experiment": str(experiment["experiment"]),
            "checkpoint": checkpoint,
            "method": method,
            "epoch": epoch,
            "truth_quantity": "RSP_200MeV",
            "water_mean_rsp": values["water_mean"],
            "water_bias_rsp": values["water_mean"] - 1.0,
            "water_std_rsp": values["water_std"],
            "phantom_rsp_rmse": values["phantom_rmse_vs_rsp_truth"],
            "aluminium_platform_rsp": float(inner.mean()),
            "aluminium_platform_recovery": float(inner.mean() / float(truth.max())),
            "roi_cnr_median": values["insert_roi_cnr_median"],
            "edge_10_90_median_mm": float(np.median(widths)),
            "outside_support_nonzero": int(np.count_nonzero(image[outside_support])),
            "mtf_10_lp_per_mm": "",
            "mtf_50_lp_per_mm": "",
            "mtf_status": "not_available_no_dedicated_target",
            "path_error_mm": "",
            "path_error_status": "not_available_no_truth_trajectories",
            "image_path": relative(path),
        })
    return rows


def load_validation_indices(mask_path: Path, count: int, bitorder: str) -> np.ndarray:
    packed = np.fromfile(mask_path, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder=bitorder, count=count)
    return np.flatnonzero(bits).astype(np.int64, copy=False)


def evaluate_validation_wepl(
    experiment: dict,
    baseline_dir: Path,
    config: dict,
    batch_size: int,
    device: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    import cupy as cp
    from gpu_mlp_operator import GpuMlpProjector

    reconstruction = path_for(experiment, "reconstruction_data")
    preprocessing = path_for(experiment, "preprocessing_data")
    split_dir = preprocessing / "splits" / SPLIT_NAME
    split_manifest = json.loads((baseline_dir / "split_manifest.json").read_text(encoding="utf-8"))
    cp.cuda.Device(device).use()
    executed = json.loads(
        (CODE_ROOT / "iterative_reconstruction" / "qc" / f"results{experiment['experiment']}" / "run_summary.json").read_text(encoding="utf-8")
    )["config"]
    size = int(executed["grid_size"])
    spacing = float(executed["grid_spacing_mm"])
    step = float(executed["path_step_mm"])
    radius = float(executed["phantom_radius_mm"])
    images: dict[str, object] = {}
    for checkpoint, _, _, relative_path in CHECKPOINTS:
        image, _, _, _ = rsp_metrics.read_mhd(reconstruction / relative_path)
        images[checkpoint] = cp.asarray(np.ascontiguousarray(image, dtype=np.float32))
    projector = GpuMlpProjector(size, spacing, step, radius)
    lut = make_vectorized_wepl_lut()
    totals = {
        name: {"squared": 0.0, "absolute": 0.0, "signed": 0.0, "count": 0}
        for name in images
    }
    angle_rows: list[dict[str, object]] = []
    total_validation = int(split_manifest["validation_count"])
    processed = 0
    started = time.monotonic()
    bitorder = str(split_manifest["bit_order"])
    projections = int(experiment["acquisition"]["projections"])
    first_angle = float(experiment["acquisition"]["first_angle_deg"])
    angle_step = float(experiment["acquisition"]["angle_step_deg"])
    for run_id in range(projections):
        pair_path = preprocessing / "pairs_filtered" / f"pairs{run_id:04d}.mhd"
        pairs = read_pairs(pair_path)
        indices = load_validation_indices(
            split_dir / f"validation_mask_{run_id:04d}.bin", len(pairs), bitorder
        )
        per_angle = {
            name: {"squared": 0.0, "absolute": 0.0, "signed": 0.0, "count": 0}
            for name in images
        }
        for begin in range(0, len(indices), batch_size):
            selected = np.asarray(pairs[indices[begin : begin + batch_size]], dtype=np.float32)
            wepl = energies_to_wepl_vectorized(lut, selected[:, 4, 0], selected[:, 4, 1])
            batch = {
                "position_in": selected[:, 0, :],
                "position_out": selected[:, 1, :],
                "direction_in": selected[:, 2, :],
                "direction_out": selected[:, 3, :],
                "wepl_mm": wepl,
            }
            statistics = projector.evaluate_many(
                images, batch, first_angle + angle_step * run_id
            )
            for name, values in statistics.items():
                for key in ("squared", "absolute", "signed", "count"):
                    per_angle[name][key] += values[key]
                    totals[name][key] += values[key]
            processed += len(selected)
        for name, values in per_angle.items():
            count = int(values["count"])
            angle_rows.append({
                "experiment": str(experiment["experiment"]),
                "checkpoint": name,
                "run_id": run_id,
                "angle_deg": first_angle + angle_step * run_id,
                "validation_pairs": len(indices),
                "valid_measurements": count,
                "wepl_rmse_mm": math.sqrt(values["squared"] / count),
                "wepl_mae_mm": values["absolute"] / count,
                "wepl_bias_mm": values["signed"] / count,
            })
        elapsed = time.monotonic() - started
        rate = processed / elapsed if elapsed else 0.0
        eta = (total_validation - processed) / rate if rate else math.inf
        if (run_id + 1) % 5 == 0 or run_id == projections - 1:
            print(
                f"validation WEPL: angle {run_id+1:03d}/{projections}, "
                f"{processed:,}/{total_validation:,} pairs ({processed/total_validation:.2%}), "
                f"{rate:,.0f} pairs/s, ETA={eta/60:.1f} min",
                flush=True,
            )
    global_rows: list[dict[str, object]] = []
    for name, values in totals.items():
        count = int(values["count"])
        global_rows.append({
            "experiment": str(experiment["experiment"]),
            "checkpoint": name,
            "validation_pairs": total_validation,
            "valid_measurements": count,
            "wepl_rmse_mm": math.sqrt(values["squared"] / count),
            "wepl_mae_mm": values["absolute"] / count,
            "wepl_bias_mm": values["signed"] / count,
            "aggregation": "measurement_count_weighted",
        })
    props = cp.cuda.runtime.getDeviceProperties(device)
    gpu_name = props["name"]
    resource = {
        "gpu": gpu_name.decode() if isinstance(gpu_name, bytes) else str(gpu_name),
        "device": device,
        "batch_size": batch_size,
        "elapsed_seconds": time.monotonic() - started,
        "validation_pairs": total_validation,
        "checkpoint_count": len(images),
        "throughput_pairs_per_second": total_validation / (time.monotonic() - started),
        "peak_gpu_memory_bytes": int(cp.get_default_memory_pool().total_bytes()),
    }
    return global_rows, angle_rows, resource


def build_summary(
    experiment: dict,
    baseline_dir: Path,
    image_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    resource: dict[str, object],
) -> None:
    split = json.loads((baseline_dir / "split_manifest.json").read_text(encoding="utf-8"))
    validation = {row["checkpoint"]: row for row in validation_rows}
    merged: list[dict[str, object]] = []
    for row in image_rows:
        merged.append({**row, **{
            "validation_wepl_rmse_mm": validation[row["checkpoint"]]["wepl_rmse_mm"],
            "validation_wepl_mae_mm": validation[row["checkpoint"]]["wepl_mae_mm"],
            "validation_wepl_bias_mm": validation[row["checkpoint"]]["wepl_bias_mm"],
            "validation_measurements": validation[row["checkpoint"]]["valid_measurements"],
        }})
    write_csv(baseline_dir / "checkpoint_metrics.csv", merged)
    write_csv(
        baseline_dir / "baseline_metrics.csv",
        [row for row in merged if row["checkpoint"] in {"analytic_nohann", "iterative_epoch_03"}],
    )
    analytic = next(row for row in merged if row["checkpoint"] == "analytic_nohann")
    iterative = next(row for row in merged if row["checkpoint"] == "iterative_epoch_03")
    markdown = f"""# results0716 frozen baseline summary

- Status: PASS
- Generated: {utc_timestamp()}
- Truth: 200 MeV reference RSP
- Split: `{split['validation_rule']}`
- Training pairs: {split['train_count']:,}
- Validation pairs: {split['validation_count']:,} ({split['validation_fraction']:.4%})
- Validation angles: {split['angle_count']}

| Reconstruction | Water mean | Water std | Phantom RSP RMSE | Al platform recovery | CNR | Edge 10-90 (mm) | Validation WEPL RMSE (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-Hann DDB-FDK | {analytic['water_mean_rsp']:.6f} | {analytic['water_std_rsp']:.6f} | {analytic['phantom_rsp_rmse']:.6f} | {analytic['aluminium_platform_recovery']:.4%} | {analytic['roi_cnr_median']:.2f} | {analytic['edge_10_90_median_mm']:.4f} | {analytic['validation_wepl_rmse_mm']:.4f} |
| GPU OS-SART + Huber-TV, epoch 3 | {iterative['water_mean_rsp']:.6f} | {iterative['water_std_rsp']:.6f} | {iterative['phantom_rsp_rmse']:.6f} | {iterative['aluminium_platform_recovery']:.4%} | {iterative['roi_cnr_median']:.2f} | {iterative['edge_10_90_median_mm']:.4f} | {iterative['validation_wepl_rmse_mm']:.4f} |

MTF and path-error fields are intentionally unavailable for this experiment. The WEPL values above are computed on the fixed validation split and are distinct from the online pre-update training residuals saved by reconstruction.

Validation forward projection used `{resource['gpu']}` and took {resource['elapsed_seconds']:.1f} s.
"""
    atomic_text(baseline_dir / "baseline_summary.md", markdown)


def evaluate_metrics(
    experiment: dict,
    baseline_dir: Path,
    config: dict,
    force: bool,
    batch_size: int,
    device: int,
) -> dict[str, object]:
    required = [baseline_dir / "split_manifest.json", baseline_dir / "baseline_manifest.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"run freeze and split first; missing {missing}")
    outputs = [baseline_dir / "checkpoint_metrics.csv", baseline_dir / "validation_wepl.csv"]
    if any(path.exists() for path in outputs) and not force:
        raise FileExistsError("metric outputs exist; use --force to replace them")
    image_rows = image_metric_rows(experiment)
    write_csv(baseline_dir / "image_metrics.csv", image_rows)
    validation_rows, angle_rows, resource = evaluate_validation_wepl(
        experiment, baseline_dir, config, batch_size, device
    )
    write_csv(baseline_dir / "validation_wepl.csv", validation_rows)
    write_csv(baseline_dir / "validation_wepl_by_angle.csv", angle_rows)
    write_json(baseline_dir / "evaluation_resources.json", resource)
    build_summary(experiment, baseline_dir, image_rows, validation_rows, resource)
    return resource


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_baseline(experiment: dict, baseline_dir: Path, qc_dir: Path) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    manifest_path = baseline_dir / "baseline_manifest.json"
    split_path = baseline_dir / "split_manifest.json"
    if not manifest_path.exists() or not split_path.exists():
        raise FileNotFoundError("baseline and split manifests are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"verify: checking {len(manifest['artifacts']):,} frozen artifact hashes", flush=True)
    bad: list[str] = []
    started = time.monotonic()
    for index, record in enumerate(manifest["artifacts"], 1):
        path = REPOSITORY_ROOT / str(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(record["size_bytes"])
            or path.stat().st_mtime_ns != int(record["mtime_ns"])
            or sha256_file(path) != record["sha256"]
        ):
            bad.append(str(record["path"]))
        if index % 250 == 0 or index == len(manifest["artifacts"]):
            print(f"verify: artifacts {index:,}/{len(manifest['artifacts']):,}", flush=True)
    check("frozen_artifact_hashes", not bad, "all unchanged" if not bad else f"mismatch: {bad[:3]}")

    frozen_auxiliary_bad: list[str] = []
    for group in ("source_files", "qc_files"):
        for record in manifest[group]:
            path = REPOSITORY_ROOT / str(record["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(record["size_bytes"])
                or sha256_file(path) != record["sha256"]
            ):
                frozen_auxiliary_bad.append(str(record["path"]))
    check(
        "frozen_source_and_qc_hashes",
        not frozen_auxiliary_bad,
        "source and pre-existing QC unchanged" if not frozen_auxiliary_bad else f"mismatch: {frozen_auxiliary_bad[:3]}",
    )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    run_rows = _read_csv(baseline_dir / "split_runs.csv")
    mask_bad = []
    rule_bad = []
    total = validation = 0
    for row in run_rows:
        path = REPOSITORY_ROOT / row["mask_path"]
        if not path.is_file() or sha256_file(path) != row["mask_sha256"]:
            mask_bad.append(row["run_id"])
        count = int(row["total_count"])
        run_id = int(row["run_id"])
        expected_digest = hashlib.sha256()
        chunk = 1_000_000 - 1_000_000 % 8
        for begin in range(0, count, chunk):
            end = min(begin + chunk, count)
            indices = np.arange(begin, end, dtype=np.uint64)
            selected = (
                splitmix64(indices, run_id, int(split["seed"]))
                % np.uint64(int(split["validation_modulus"]))
                == np.uint64(int(split["validation_remainder"]))
            )
            expected_digest.update(np.packbits(selected, bitorder=str(split["bit_order"])).tobytes())
        if expected_digest.hexdigest() != row["mask_sha256"]:
            rule_bad.append(row["run_id"])
        total += int(row["total_count"])
        validation += int(row["validation_count"])
    check("split_mask_hashes", not mask_bad, "720 byte-stable masks" if not mask_bad else f"bad runs: {mask_bad[:5]}")
    check("split_rule_reproduced", not rule_bad, "all masks reproduced from the versioned rule" if not rule_bad else f"bad runs: {rule_bad[:5]}")
    check("split_partition_counts", total == int(split["total_count"]) and validation == int(split["validation_count"]), f"total={total:,}, validation={validation:,}")
    check("split_angle_coverage", len(run_rows) == 720 and all(int(row["train_count"]) > 0 and int(row["validation_count"]) > 0 for row in run_rows), f"angles={len(run_rows)}")

    checkpoint_path = baseline_dir / "checkpoint_metrics.csv"
    validation_path = baseline_dir / "validation_wepl.csv"
    check("metric_outputs", checkpoint_path.is_file() and validation_path.is_file(), "checkpoint and validation tables exist")
    if checkpoint_path.is_file():
        rows = _read_csv(checkpoint_path)
        finite_columns = [
            "water_mean_rsp", "water_std_rsp", "phantom_rsp_rmse",
            "aluminium_platform_rsp", "aluminium_platform_recovery",
            "roi_cnr_median", "edge_10_90_median_mm", "validation_wepl_rmse_mm",
        ]
        finite = len(rows) == len(CHECKPOINTS) and all(
            math.isfinite(float(row[column])) for row in rows for column in finite_columns
        )
        check("finite_metrics", finite, f"checkpoints={len(rows)}")
        existing = _read_csv(CODE_ROOT / "iterative_reconstruction" / "qc" / f"results{experiment['experiment']}" / "rsp_metrics.csv")
        current = {row["checkpoint"]: row for row in rows}
        tolerance = 2e-6
        mapping = {
            "initial": "iterative_initial", "epoch_01": "iterative_epoch_01",
            "epoch_02": "iterative_epoch_02", "epoch_03": "iterative_epoch_03",
        }
        reproducible = all(
            abs(float(old["phantom_rsp_rmse"]) - float(current[mapping[old["checkpoint"]]]["phantom_rsp_rmse"])) <= tolerance
            and abs(float(old["water_mean"]) - float(current[mapping[old["checkpoint"]]]["water_mean_rsp"])) <= tolerance
            for old in existing
        )
        check("existing_rsp_qc_reproduced", reproducible, f"absolute tolerance={tolerance:g}")

    config_epochs = int(experiment["iterative"]["epochs"])
    executed_epochs = int(manifest["executed_iterative_config"]["epochs"])
    check("planned_executed_config_recorded", config_epochs == 10 and executed_epochs == 3, f"planned={config_epochs}, executed={executed_epochs}")
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "verified_at": utc_timestamp(),
        "elapsed_seconds": time.monotonic() - started,
        "experiment": str(experiment["experiment"]),
        "checks": checks,
    }
    write_json(qc_dir / "evaluation_summary.json", result)
    if status != "PASS":
        raise RuntimeError("baseline verification failed; see evaluation_summary.json")
    return result
