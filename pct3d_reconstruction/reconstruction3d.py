"""GPU OS-SART driver for the compact 3-D list-mode dataset."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from evaluation3d import image_metrics
from gpu_operator3d import GpuMlpProjector3D
from io3d import batch_slices, pair_batch, read_pairs, read_partition, read_volume, write_volume
from physics3d import support_mask
from regularization3d import proximal_huber_tv


Progress = Callable[..., None]


def _selection(preprocessing: Path, run_id: int, mode: str) -> np.ndarray:
    if mode == "screen":
        values = read_partition(preprocessing / "splits" / f"screen{run_id:04d}.npz")
        return np.flatnonzero(values == 1)
    split = read_partition(preprocessing / "splits" / f"split{run_id:04d}.npz")
    code = {"train": 0, "validation": 1, "test": 2}[mode]
    return np.flatnonzero(split == code)


def _iter_batches(preprocessing: Path, run_id: int, mode: str, batch_size: int):
    pairs = read_pairs(preprocessing / "pairs" / f"pairs{run_id:04d}.mhd", mmap=True)
    indexes = _selection(preprocessing, run_id, mode)
    for section in batch_slices(len(indexes), batch_size):
        yield pair_batch(pairs, indexes[section])


def evaluate_partition(
    projector: GpuMlpProjector3D,
    image,
    preprocessing: Path,
    runs: int,
    mode: str,
    batch_size: int,
    angle_step: float,
) -> dict[str, float]:
    totals = {"squared": 0.0, "absolute": 0.0, "signed": 0.0, "count": 0}
    for run_id in range(runs):
        for batch in _iter_batches(preprocessing, run_id, mode, batch_size):
            result = projector.evaluate(image, batch, run_id * angle_step)
            for key in totals:
                totals[key] += result[key]
    count = max(int(totals["count"]), 1)
    return {
        "wepl_rmse_mm": float(np.sqrt(totals["squared"] / count)),
        "wepl_mae_mm": float(totals["absolute"] / count),
        "wepl_bias_mm": float(totals["signed"] / count),
        "wepl_measurements": int(totals["count"]),
    }


def initial_volume(config: dict) -> np.ndarray:
    result = np.zeros(tuple(reversed(config["grid"]["size_xyz"])), dtype=np.float32)
    result[support_mask(config)] = 1.0
    return result


def reconstruct(
    *,
    config: dict,
    preprocessing: Path,
    output: Path,
    mode: str,
    beta: float,
    truth: np.ndarray,
    simulation: dict,
    mlic: dict[str, float],
    device: int,
    progress: Progress,
    force: bool = False,
) -> dict:
    import cupy as cp

    cp.cuda.Device(device).use()
    recon_dir = output / "recon"
    metrics_path = output / "run_summary.json"
    previous_summary = None
    if metrics_path.is_file() and not force:
        previous_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous_summary.get("status") == "PASS":
            return previous_summary
    recon_dir.mkdir(parents=True, exist_ok=True)
    support_cpu = support_mask(config)
    support = cp.asarray(support_cpu)
    epochs = int(config["reconstruction"]["epochs"])
    rows = [] if force else list(previous_summary.get("epochs", [])) if previous_summary else []
    completed_epochs = sorted(
        int(row["epoch"])
        for row in rows
        if (recon_dir / f"epoch_{int(row['epoch']):02d}.mhd").is_file()
    )
    if completed_epochs and not force:
        completed = max(completed_epochs)
        image_cpu = np.array(
            read_volume(recon_dir / f"epoch_{completed:02d}.mhd")[0],
            copy=True,
        )
        start_epoch = completed + 1
        rows = [row for row in rows if int(row["epoch"]) <= completed]
    else:
        image_cpu, start_epoch = initial_volume(config), 1
        write_volume(
            recon_dir / "initial.mhd",
            image_cpu,
            tuple(config["grid"]["spacing_xyz_mm"]),
            tuple(config["grid"]["origin_xyz_mm"]),
        )
    image = cp.asarray(image_cpu)
    projector = GpuMlpProjector3D(config)
    batch_size = int(config["reconstruction"]["batch_size"])
    started = time.perf_counter()
    train_mode = "screen" if mode == "screen" else "train"
    total_subset_steps = epochs * int(config["reconstruction"]["subsets"])

    def accumulate_with_fallback(batch, angle, numerator, denominator):
        nonlocal batch_size
        try:
            return projector.accumulate(image, batch, angle, numerator, denominator)
        except cp.cuda.memory.OutOfMemoryError:
            cp.get_default_memory_pool().free_all_blocks()
            fallback = int(config["reconstruction"]["batch_size_fallback"])
            if len(batch["wepl_mm"]) <= fallback:
                raise
            batch_size = fallback
            middle = len(batch["wepl_mm"]) // 2
            totals = [0.0, 0]
            for begin, end in ((0, middle), (middle, len(batch["wepl_mm"]))):
                part = {key: value[begin:end] for key, value in batch.items()}
                squared, count = accumulate_with_fallback(part, angle, numerator, denominator)
                totals[0] += squared
                totals[1] += count
            return float(totals[0]), int(totals[1])

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_input = image.copy()
        residual_sum = valid_count = 0
        for subset in range(int(config["reconstruction"]["subsets"])):
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            for run_id in range(subset, int(config["runs"]), int(config["reconstruction"]["subsets"])):
                for batch in _iter_batches(preprocessing, run_id, train_mode, batch_size):
                    squared, count = accumulate_with_fallback(
                        batch,
                        run_id * float(config["angle_step_deg"]),
                        numerator,
                        denominator,
                    )
                    residual_sum += squared
                    valid_count += count
            relaxation = float(config["reconstruction"]["relaxation"]) / (
                1.0 + float(config["reconstruction"]["relaxation_decay"]) * (epoch - 1)
            )
            # CuPy 13 does not accept NumPy's ``where=`` keyword for this
            # ufunc signature. Boolean indexing also guarantees that zero-
            # coverage voxels remain exactly zero without evaluating 0/0.
            update = cp.zeros_like(numerator)
            covered = denominator > 0
            update[covered] = numerator[covered] / denominator[covered]
            image += relaxation * update
            cp.maximum(image, 0.0, out=image)
            image *= support
            completed_subset_steps = (
                (epoch - 1) * int(config["reconstruction"]["subsets"])
                + subset
                + 1
            )
            task_elapsed = time.perf_counter() - started
            progress(
                stage="reconstruct",
                group=output.name,
                epoch=epoch,
                total_epochs=epochs,
                subset=subset + 1,
                total_subsets=int(config["reconstruction"]["subsets"]),
                task_elapsed_seconds=task_elapsed,
                task_eta_seconds=(
                    task_elapsed
                    / max(completed_subset_steps, 1)
                    * max(total_subset_steps - completed_subset_steps, 0)
                ),
            )
        image, regularization = proximal_huber_tv(
            image,
            support,
            beta,
            float(config["reconstruction"]["huber_delta"]),
            int(config["reconstruction"]["regularization_iterations"]),
        )
        image_cpu = cp.asnumpy(image)
        epoch_difference = image - epoch_input
        checkpoint = recon_dir / f"epoch_{epoch:02d}.mhd"
        write_volume(
            checkpoint,
            image_cpu,
            tuple(config["grid"]["spacing_xyz_mm"]),
            tuple(config["grid"]["origin_xyz_mm"]),
        )
        validation = evaluate_partition(
            projector,
            image,
            preprocessing,
            int(config["runs"]),
            "validation",
            batch_size,
            float(config["angle_step_deg"]),
        )
        metrics, materials, edges = image_metrics(image_cpu, truth, config, simulation, mlic)
        rows.append(
            {
                "epoch": epoch,
                "training_wepl_rmse_mm": float(np.sqrt(residual_sum / max(valid_count, 1))),
                "training_measurements": valid_count,
                "elapsed_seconds": time.perf_counter() - epoch_start,
                "update_l2": float(cp.linalg.norm(epoch_difference).get()),
                "update_max_abs": float(cp.max(cp.abs(epoch_difference)).get()),
                "regularization": regularization,
                "validation": validation,
                "image": metrics,
                "materials": materials,
                "edges": edges,
            }
        )
        partial = {
            "status": "RUNNING",
            "mode": mode,
            "beta": beta,
            "batch_size": batch_size,
            "epochs": rows,
            "test_partition_opened": False,
        }
        temporary = metrics_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        temporary.replace(metrics_path)
    result = {
        "status": "PASS",
        "mode": mode,
        "beta": beta,
        "batch_size": batch_size,
        "epochs": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_memory_pool_bytes": int(cp.get_default_memory_pool().total_bytes()),
        "test_partition_opened": False,
    }
    temporary = metrics_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metrics_path)
    return result
