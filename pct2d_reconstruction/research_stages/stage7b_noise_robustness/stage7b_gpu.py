"""Weighted/robust Stage-7B OS-SART using the retained Schulte MLP operator."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class Stage7bProjector:
    """Compose base inverse-variance weights with standardized Huber IRLS."""

    def __init__(self, size: int, spacing: float, step: float, radius: float):
        from weighted_gpu import WeightedGpuMlpProjector

        self.inner = WeightedGpuMlpProjector(size, spacing, step, radius)
        self.cp = self.inner.cp

    def accumulate(
        self,
        image,
        batch: dict[str, np.ndarray],
        base_weight: np.ndarray,
        sigma_mm: np.ndarray,
        huber_z: float | None,
        angle_deg: float,
        numerator,
        denominator,
    ) -> dict[str, float | int]:
        cp = self.cp
        n = len(batch["wepl_mm"])
        if n == 0:
            return {"squared": 0.0, "absolute": 0.0, "valid": 0}
        (
            blocks,
            threads,
            pixels,
            path_weights,
            row_sum,
            normalized,
            residual_squared,
            valid,
        ) = self.inner._paths_and_forward(image, batch, angle_deg)
        residual = normalized * row_sum
        weight = cp.asarray(np.ascontiguousarray(base_weight, np.float32))
        if huber_z is not None:
            sigma = cp.asarray(np.ascontiguousarray(sigma_mm, np.float32))
            threshold = cp.float32(huber_z) * sigma
            factor = cp.minimum(
                cp.float32(1.0),
                threshold / cp.maximum(cp.abs(residual), cp.float32(1.0e-12)),
            )
            weight *= factor
        weight *= valid
        self.inner.weighted_back_kernel(
            blocks,
            (threads,),
            (
                pixels,
                path_weights,
                normalized,
                weight,
                valid,
                np.int32(n),
                np.int32(self.inner.samples),
                numerator,
                denominator,
            ),
        )
        absolute = cp.abs(residual)
        return {
            "squared": float(
                cp.sum(residual_squared * valid, dtype=cp.float64).get()
            ),
            "absolute": float(
                cp.sum(absolute * valid, dtype=cp.float64).get()
            ),
            "valid": int(cp.sum(valid, dtype=cp.int64).get()),
        }

    def residuals(self, image, batch, angle_deg: float) -> np.ndarray:
        return self.inner.residuals(image, batch, angle_deg)

    def residuals_with_mask(
        self, image, batch, angle_deg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        cp = self.cp
        (
            _blocks,
            _threads,
            _pixels,
            _path_weights,
            row_sum,
            normalized,
            _residual_squared,
            valid,
        ) = self.inner._paths_and_forward(image, batch, angle_deg)
        return (
            cp.asnumpy(normalized * row_sum),
            cp.asnumpy(valid).astype(bool),
        )


def direct_batch(pairs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "position_in": pairs[:, 0, :],
        "position_out": pairs[:, 1, :],
        "direction_in": pairs[:, 2, :],
        "direction_out": pairs[:, 3, :],
        "wepl_mm": pairs[:, 4, 1],
    }


def load_noise_model(path: Path) -> dict[str, np.ndarray | float]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def predict_empirical(model: dict, energy: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.exp(
            np.interp(
                energy,
                model["energy_mev"],
                np.log(model["sigma_mm"]),
                left=np.log(model["sigma_mm"][0]),
                right=np.log(model["sigma_mm"][-1]),
            )
        ),
        float(model["minimum_sigma_mm"]),
    )


def predict_analytic(
    energy: np.ndarray, range_energy: np.ndarray, range_mm: np.ndarray,
    fraction: float, minimum: float,
) -> np.ndarray:
    derivative = np.gradient(range_mm, range_energy)
    slope = np.interp(energy, range_energy, derivative)
    return np.maximum(np.abs(slope) * fraction * energy, minimum)


def per_angle_weights(
    kind: str,
    energy: np.ndarray,
    empirical_model: dict,
    range_energy: np.ndarray,
    range_mm: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    empirical = predict_empirical(empirical_model, energy)
    analytic = predict_analytic(
        energy,
        range_energy,
        range_mm,
        0.01,
        float(config["noise_model"]["minimum_sigma_mm"]),
    )
    if kind == "equal":
        raw = np.ones(len(energy), dtype=np.float64)
    elif kind == "analytic":
        raw = 1.0 / np.square(analytic)
    elif kind == "empirical":
        raw = 1.0 / np.square(empirical)
    else:
        raise ValueError(f"unknown Stage-7B weight: {kind}")
    finite = np.isfinite(raw) & (raw > 0)
    if not np.any(finite):
        raise RuntimeError("no finite positive data weights")
    raw /= np.median(raw[finite])
    low, high = (float(v) for v in config["weights"]["clip"])
    weight = np.clip(raw, low, high).astype(np.float32)
    ess = float(np.square(weight.sum()) / np.square(weight).sum() / len(weight))
    minimum_ess = float(config["weights"]["minimum_effective_fraction"])
    # A steep range derivative at low exit energy can make inverse-variance
    # weights too concentrated even after hard clipping.  Temper them toward
    # equal weights until the pre-registered ESS floor is met.
    for _ in range(16):
        if ess >= minimum_ess:
            break
        weight = (0.5 * (weight + np.float32(1.0))).astype(np.float32)
        ess = float(
            np.square(weight.sum()) / np.square(weight).sum() / len(weight)
        )
    if ess < minimum_ess:
        raise RuntimeError(f"weight effective fraction collapsed to {ess:.3f}")
    return weight, empirical.astype(np.float32), ess


def evaluate(
    image_path: Path,
    pair_root: Path,
    device: int,
    config: dict,
    runs: int,
    progress: Callable[..., None] | None = None,
) -> dict[str, float | int]:
    import cupy as cp
    from preprocessing.paircuts import read_mhd
    from iterative_reconstruction.mhd_io import read_image_2d

    settings = config["reconstruction"]
    cp.cuda.Device(device).use()
    image = cp.asarray(read_image_2d(image_path)[0])
    projector = Stage7bProjector(
        int(settings["grid_size"]),
        float(settings["grid_spacing_mm"]),
        float(settings["path_step_mm"]),
        float(config["phantom_radius_mm"]),
    )
    measured_residual, ideal_residual = [], []
    for run_id in range(runs):
        pairs = read_mhd(pair_root / "pairs" / f"pairs{run_id:04d}.mhd")
        meta = np.load(
            pair_root / "metadata" / f"meta{run_id:04d}.npz",
            allow_pickle=False,
        )
        for begin in range(0, len(pairs), int(settings["batch_size"])):
            selected = np.asarray(
                pairs[begin : begin + int(settings["batch_size"])],
                dtype=np.float32,
            )
            residual, valid = projector.residuals_with_mask(
                image, direct_batch(selected), run_id * float(config["angle_step_deg"])
            )
            measured_residual.append(residual[valid])
            predicted = selected[valid, 4, 1] - residual[valid]
            ideal_residual.append(
                np.asarray(
                    meta["ideal_wepl_mm"][begin : begin + len(selected)]
                )[valid]
                - predicted
            )
        if progress and ((run_id + 1) % 20 == 0 or run_id + 1 == runs):
            progress(eval_completed_runs=run_id + 1, eval_total_runs=runs)
    measured = np.concatenate(measured_residual).astype(np.float64)
    ideal = np.concatenate(ideal_residual).astype(np.float64)
    if not len(ideal) or not np.isfinite(ideal).all():
        raise RuntimeError("invalid fixed-partition WEPL evaluation")
    return {
        "count": int(len(ideal)),
        "measured_rmse_mm": float(np.sqrt(np.mean(measured**2))),
        "measured_mae_mm": float(np.mean(np.abs(measured))),
        "measured_bias_mm": float(np.mean(measured)),
        "measured_abs_p99_mm": float(np.quantile(np.abs(measured), 0.99)),
        "ideal_rmse_mm": float(np.sqrt(np.mean(ideal**2))),
        "ideal_mae_mm": float(np.mean(np.abs(ideal))),
        "ideal_bias_mm": float(np.mean(ideal)),
        "ideal_abs_p99_mm": float(np.quantile(np.abs(ideal), 0.99)),
    }


def run_reconstruction(
    name: str,
    candidate: dict[str, Any],
    train_root: Path,
    validation_root: Path,
    initial_path: Path,
    output_root: Path,
    model_path: Path,
    config: dict,
    epochs: int,
    device: int,
    runs: int,
    progress: Callable[..., None],
    force: bool = False,
) -> dict[str, Any]:
    import cupy as cp
    from preprocessing.paircuts import read_mhd
    from iterative_reconstruction.gpu_regularization import proximal_regularize
    from iterative_reconstruction.mhd_io import (
        read_image_2d,
        write_image_2d,
    )
    from iterative_reconstruction.physics import load_wepl_model

    final_summary = output_root / "run_summary.json"
    if final_summary.is_file() and not force:
        return json.loads(final_summary.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    recon_dir = output_root / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    settings = config["reconstruction"]
    image_cpu, source_spacing, source_origin = read_image_2d(initial_path)
    image_cpu = np.array(image_cpu, np.float32, copy=True)
    spacing = float(source_spacing[0])
    origin = float(source_origin[0])
    coordinates = origin + np.arange(len(image_cpu)) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    support_cpu = (
        xx * xx + zz * zz <= float(config["phantom_radius_mm"]) ** 2
    )
    image_cpu[~support_cpu] = 0
    np.maximum(image_cpu, 0, out=image_cpu)
    cp.cuda.Device(device).use()
    image = cp.asarray(image_cpu)
    support = cp.asarray(support_cpu)
    projector = Stage7bProjector(
        int(settings["grid_size"]),
        float(settings["grid_spacing_mm"]),
        float(settings["path_step_mm"]),
        float(config["phantom_radius_mm"]),
    )
    empirical_model = load_noise_model(Path(config["_noise_model_path"]))
    wepl_model = load_wepl_model("g4_water_calibrated", model_path)
    histories: list[dict[str, Any]] = []
    history_path = output_root / "epoch_metrics.csv"
    start_epoch = 0
    if history_path.is_file() and not force:
        with history_path.open(encoding="utf-8") as stream:
            histories = list(csv.DictReader(stream))
        if histories:
            start_epoch = int(histories[-1]["epoch"])
            image_cpu = read_image_2d(
                recon_dir / f"epoch_{start_epoch:02d}.mhd"
            )[0]
            image = cp.asarray(np.asarray(image_cpu, np.float32))
    total_counts = [
        int(read_mhd(train_root / "pairs" / f"pairs{run_id:04d}.mhd").shape[0])
        for run_id in range(runs)
    ]
    pairs_per_epoch = int(sum(total_counts))
    total_target = pairs_per_epoch * epochs
    completed = pairs_per_epoch * start_epoch
    started = time.perf_counter()
    max_memory = 0
    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        squared = absolute = 0.0
        valid = 0
        max_update = 0.0
        ess_values = []
        relaxation = float(settings["relaxation"]) / (
            1.0 + float(settings["relaxation_decay"]) * epoch
        )
        for subset in range(int(settings["subsets"])):
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            for run_id in range(subset, runs, int(settings["subsets"])):
                pairs = read_mhd(
                    train_root / "pairs" / f"pairs{run_id:04d}.mhd"
                )
                with np.load(
                    train_root / "metadata" / f"meta{run_id:04d}.npz",
                    allow_pickle=False,
                ) as meta:
                    energy = np.asarray(meta["measured_eout_mev"])
                weights, sigma, ess = per_angle_weights(
                    str(candidate["weight"]),
                    energy,
                    empirical_model,
                    wepl_model.energy_mev,
                    wepl_model.range_mm,
                    config,
                )
                ess_values.append(ess)
                batch_size = int(settings["batch_size"])
                for begin in range(0, len(pairs), batch_size):
                    selected = np.asarray(
                        pairs[begin : begin + batch_size], np.float32
                    )
                    metrics = projector.accumulate(
                        image,
                        direct_batch(selected),
                        weights[begin : begin + len(selected)],
                        sigma[begin : begin + len(selected)],
                        candidate.get("huber_z"),
                        run_id * float(config["angle_step_deg"]),
                        numerator,
                        denominator,
                    )
                    squared += float(metrics["squared"])
                    absolute += float(metrics["absolute"])
                    valid += int(metrics["valid"])
                    completed += len(selected)
            update = cp.where(
                denominator > 0,
                cp.float32(relaxation)
                * numerator
                / cp.maximum(denominator, cp.float32(1.0e-20)),
                cp.float32(0),
            )
            image += update
            cp.maximum(image, 0, out=image)
            image *= support
            cp.cuda.Stream.null.synchronize()
            max_update = max(max_update, float(cp.max(cp.abs(update)).get()))
            elapsed = time.perf_counter() - started
            rate = (completed - pairs_per_epoch * start_epoch) / max(elapsed, 1e-9)
            eta = max(total_target - completed, 0) / max(rate, 1e-9)
            used = cp.get_default_memory_pool().used_bytes()
            max_memory = max(max_memory, int(used))
            progress(
                stage="reconstruct",
                candidate=name,
                epoch=epoch + 1,
                total_epochs=epochs,
                subset=subset + 1,
                total_subsets=int(settings["subsets"]),
                pairs_per_second=rate,
                task_eta_seconds=eta,
                gpu_memory_bytes=max_memory,
            )
            print(
                f"{name} epoch {epoch+1}/{epochs} subset "
                f"{subset+1:02d}/{settings['subsets']}: "
                f"rate={rate:,.0f} pairs/s ETA={eta/3600:.2f} h",
                flush=True,
            )
            del numerator, denominator, update
        image, reg = proximal_regularize(
            image,
            support,
            kind=str(settings["regularizer"]),
            weight=float(settings["regularization_weight"]),
            iterations=int(settings["regularization_iterations"]),
            huber_delta=float(settings["huber_delta"]),
            primal_step=float(settings["primal_step"]),
            dual_step=float(settings["dual_step"]),
        )
        checkpoint = recon_dir / f"epoch_{epoch+1:02d}.mhd"
        host = cp.asnumpy(image)
        write_image_2d(checkpoint, host, spacing, origin)
        validation = evaluate(
            checkpoint, validation_root, device, config, runs, progress
        )
        row = {
            "candidate": name,
            "epoch": epoch + 1,
            "training_rmse_mm": math.sqrt(squared / max(valid, 1)),
            "training_mae_mm": absolute / max(valid, 1),
            "validation_measured_rmse_mm": validation["measured_rmse_mm"],
            "validation_ideal_rmse_mm": validation["ideal_rmse_mm"],
            "validation_ideal_abs_p99_mm": validation["ideal_abs_p99_mm"],
            "max_update": max_update,
            "minimum_effective_fraction": min(ess_values),
            "regularization_seconds": float(reg["elapsed_seconds"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "image_path": str(checkpoint),
        }
        histories.append(row)
        write_csv(history_path, histories)
    best = min(
        histories, key=lambda row: float(row["validation_ideal_rmse_mm"])
    )
    best_epoch = int(best["epoch"])
    summary = {
        "status": "PASS",
        "candidate": name,
        "weight": candidate["weight"],
        "huber_z": candidate.get("huber_z"),
        "completed_epochs": epochs,
        "best_epoch": best_epoch,
        "best_image": str(recon_dir / f"epoch_{best_epoch:02d}.mhd"),
        "validation_ideal_rmse_mm": float(best["validation_ideal_rmse_mm"]),
        "validation_ideal_abs_p99_mm": float(
            best["validation_ideal_abs_p99_mm"]
        ),
        "minimum_effective_fraction": float(
            min(float(row["minimum_effective_fraction"]) for row in histories)
        ),
        "elapsed_seconds_this_call": time.perf_counter() - started,
        "gpu_memory_peak_bytes": max_memory,
        "finite": bool(np.isfinite(cp.asnumpy(image)).all()),
        "support_outside_nonzero": int(
            np.count_nonzero(cp.asnumpy(image)[~support_cpu])
        ),
    }
    if not summary["finite"] or summary["support_outside_nonzero"]:
        raise RuntimeError(f"invalid reconstruction QC for {name}")
    atomic_json(final_summary, summary)
    return summary
