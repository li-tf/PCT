#!/usr/bin/env python3
"""Run full list-mode Schulte-MLP OS-SART on a CUDA GPU."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

from gpu_mlp_operator import GpuMlpProjector
from gpu_regularization import proximal_regularize
from mhd_io import read_header, read_image_2d, read_pairs, resample_to_grid, write_image_2d
from physics import energies_to_wepl_vectorized, make_vectorized_wepl_lut


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
sys.path.insert(0, str(CODE_ROOT))

from common import load_experiment, path_for  # noqa: E402
from analytic_reconstruction import rsp_metrics  # noqa: E402


def option(cli_value, config: dict, name: str):
    return cli_value if cli_value is not None else config[name]


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def pair_count(path: Path) -> int:
    return int(read_header(path)["DimSize"].split()[1])


def selected_indices(count: int, fraction: float, seed: int, run_id: int):
    if fraction >= 1.0:
        return None
    selected = max(1, int(round(count * fraction)))
    rng = np.random.default_rng(seed + run_id)
    return np.sort(rng.choice(count, size=selected, replace=False))


def initial_image(kind: str, size: int, spacing: float, fdk_nohann: Path) -> tuple[np.ndarray, float]:
    origin = -0.5 * (size - 1) * spacing
    if kind == "zero":
        return np.zeros((size, size), dtype=np.float32), origin
    image, source_spacing, source_origin = read_image_2d(fdk_nohann)
    return resample_to_grid(image, source_spacing, source_origin, size, spacing)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CUDA list-mode MLP OS-SART with integrated RSP QC"
    )
    parser.add_argument("--experiment", default="0716")
    parser.add_argument("--pairs-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--epochs", type=int, help="OS-SART epochs (default: 5)")
    parser.add_argument("--sample-fraction", type=float, help="per-angle proton fraction (default: 1.0)")
    parser.add_argument("--grid-spacing-mm", type=float, help="reconstruction pixel spacing (default: 0.5)")
    parser.add_argument("--grid-size", type=int, help="square reconstruction size (default: 420)")
    parser.add_argument("--path-step-mm", type=float, help="MLP integration step (default: 0.5)")
    parser.add_argument("--batch-size", type=int, help="protons held on GPU per batch (default: 8192)")
    parser.add_argument("--subsets", type=int, help="ordered subsets (default: 36)")
    parser.add_argument("--runs", type=int, help="number of angles from pairs0000 onward (default: 720)")
    parser.add_argument("--relaxation", type=float, help="first-epoch relaxation (default: 0.5)")
    parser.add_argument("--relaxation-decay", type=float, help="epoch relaxation decay (default: 0.1)")
    parser.add_argument("--initialization", choices=["fdk_nohann", "zero"])
    parser.add_argument("--device", type=int, help="CUDA device index (default: 0)")
    parser.add_argument("--progress-every-batches", type=int, help="batch progress interval (default: 10)")
    parser.add_argument(
        "--regularizer", choices=["none", "tv", "huber_tv"],
        help="proximal regularizer applied after selected epochs (default: none)",
    )
    parser.add_argument("--regularization-weight", type=float, help="TV/Huber-TV proximal weight")
    parser.add_argument("--regularization-iterations", type=int, help="proximal iterations per application")
    parser.add_argument("--regularization-every-epochs", type=int, help="apply regularization every N epochs")
    parser.add_argument("--huber-delta", type=float, help="Huber transition in RED/pixel")
    parser.add_argument("--primal-step", type=float, help="Chambolle--Pock primal step")
    parser.add_argument("--dual-step", type=float, help="Chambolle--Pock dual step")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    experiment = load_experiment(args.experiment)
    config = dict(experiment["iterative"])
    preprocessing_data = path_for(experiment, "preprocessing_data")
    reconstruction_data = path_for(experiment, "reconstruction_data")
    simulation_code = path_for(experiment, "simulation_code")
    args.pairs_dir = args.pairs_dir or preprocessing_data / "pairs_filtered"
    args.output_dir = args.output_dir or reconstruction_data / "iterative"
    fdk_nohann = reconstruction_data / "analytic" / "recon" / "recon_ddb_nohann.mhd"
    truth_dir = reconstruction_data / "analytic" / "truth"
    qc_dir = HERE / "qc" / f"results{args.experiment}"
    epochs = int(option(args.epochs, config, "epochs"))
    fraction = float(option(args.sample_fraction, config, "sample_fraction"))
    spacing = float(option(args.grid_spacing_mm, config, "grid_spacing_mm"))
    size = int(option(args.grid_size, config, "grid_size"))
    step = float(option(args.path_step_mm, config, "path_step_mm"))
    batch_size = int(option(args.batch_size, config, "batch_size"))
    subsets = int(option(args.subsets, config, "subsets"))
    runs = int(option(args.runs, config, "runs"))
    relaxation0 = float(option(args.relaxation, config, "relaxation"))
    relaxation_decay = float(option(args.relaxation_decay, config, "relaxation_decay"))
    initialization = args.initialization or str(config["initialization"])
    device = int(option(args.device, config, "device"))
    progress_every = int(option(args.progress_every_batches, config, "progress_every_batches"))
    regularizer = args.regularizer or str(config.get("regularizer", "none"))
    regularization_weight = float(
        args.regularization_weight
        if args.regularization_weight is not None else config.get("regularization_weight", 0.005)
    )
    regularization_iterations = int(
        args.regularization_iterations
        if args.regularization_iterations is not None else config.get("regularization_iterations", 40)
    )
    regularization_every = int(
        args.regularization_every_epochs
        if args.regularization_every_epochs is not None else config.get("regularization_every_epochs", 1)
    )
    huber_delta = float(
        args.huber_delta if args.huber_delta is not None else config.get("huber_delta", 0.01)
    )
    primal_step = float(
        args.primal_step if args.primal_step is not None else config.get("primal_step", 0.25)
    )
    dual_step = float(
        args.dual_step if args.dual_step is not None else config.get("dual_step", 0.25)
    )
    radius = float(config["phantom_radius_mm"])
    seed = int(config["seed"])
    if not (epochs >= 1 and size >= 2 and spacing > 0 and step > 0 and batch_size >= 1):
        raise SystemExit("epochs, grid size, spacing, path step, and batch size must be positive")
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("sample fraction must be in (0, 1]")
    if not 1 <= runs <= 720 or not 1 <= subsets <= runs or progress_every < 1:
        raise SystemExit("require 1 <= subsets <= runs <= 720 and positive progress interval")
    if regularizer != "none" and (
        regularization_weight <= 0.0
        or regularization_iterations < 1
        or regularization_every < 1
        or huber_delta <= 0.0
        or primal_step <= 0.0
        or dual_step <= 0.0
        or 8.0 * primal_step * dual_step >= 1.0
    ):
        raise SystemExit(
            "regularization parameters must be positive and require "
            "8 * primal_step * dual_step < 1"
        )

    import cupy as cp

    cp.cuda.Device(device).use()
    properties = cp.cuda.runtime.getDeviceProperties(device)
    gpu_name = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
    paths = [args.pairs_dir / f"pairs{run_id:04d}.mhd" for run_id in range(runs)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} pair headers; first: {missing[0]}")
    counts = np.array([pair_count(path) for path in paths], dtype=np.int64)
    selected_counts = np.maximum(1, np.rint(counts * fraction).astype(np.int64))
    if fraction >= 1.0:
        selected_counts = counts.copy()
    pairs_per_epoch = int(selected_counts.sum())

    final_path = args.output_dir / "recon" / "recon_iterative_gpu.mhd"
    if final_path.exists() and not args.force:
        raise FileExistsError(f"iterative result exists: {final_path}; use --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = args.output_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in recon_dir.iterdir():
            if path.is_file():
                path.unlink()
        for pattern in ("iteration_history.csv", "regularization_history.csv", "run_summary.json", "rsp_metrics.csv"):
            path = qc_dir / pattern
            if path.exists():
                path.unlink()
    if not fdk_nohann.is_file():
        raise FileNotFoundError(f"no-Hann initialization is missing: {fdk_nohann}")
    image_cpu, origin = initial_image(initialization, size, spacing, fdk_nohann)
    coordinates = origin + np.arange(size, dtype=np.float32) * spacing
    xx, zz = np.meshgrid(coordinates, coordinates)
    support_cpu = xx * xx + zz * zz <= radius * radius
    image_cpu[~support_cpu] = 0.0
    np.maximum(image_cpu, 0.0, out=image_cpu)
    write_image_2d(recon_dir / "initial.mhd", image_cpu, spacing, origin)

    print(f"GPU: {gpu_name} (device {device})", flush=True)
    print(
        f"configuration: pairs={pairs_per_epoch:,}/epoch ({fraction:.3%}), grid={size}x{size} @ {spacing:g} mm, "
        f"path_step={step:g} mm, subsets={subsets}, epochs={epochs}, batch={batch_size:,}",
        flush=True,
    )
    if regularizer != "none":
        print(
            f"regularization: {regularizer}, weight={regularization_weight:g}, "
            f"iterations={regularization_iterations}, every={regularization_every} epoch(s), "
            f"huber_delta={huber_delta:g}",
            flush=True,
        )
    print("building vectorized Bethe--Bloch LUT...", flush=True)
    lut_start = time.perf_counter()
    wepl_lut = make_vectorized_wepl_lut()
    print(f"Bethe--Bloch LUT ready in {time.perf_counter()-lut_start:.1f}s", flush=True)
    print("compiling CUDA MLP kernels (first launch may take a few seconds)...", flush=True)

    image = cp.asarray(image_cpu)
    support = cp.asarray(support_cpu)
    projector = GpuMlpProjector(size, spacing, step, radius)
    rows: list[dict[str, object]] = []
    regularization_rows: list[dict[str, object]] = []
    total_pairs = epochs * pairs_per_epoch
    completed_pairs = 0
    total_start = time.perf_counter()
    started = datetime.now()

    for epoch in range(epochs):
        relaxation = relaxation0 / (1.0 + relaxation_decay * epoch)
        epoch_start = time.perf_counter()
        epoch_residual_squared = 0.0
        epoch_measurements = 0
        for subset in range(subsets):
            subset_start = time.perf_counter()
            numerator = cp.zeros_like(image)
            denominator = cp.zeros_like(image)
            subset_residual_squared = 0.0
            subset_measurements = 0
            subset_pairs = 0
            subset_total = int(selected_counts[subset:runs:subsets].sum())
            batch_number = 0
            for run_id in range(subset, runs, subsets):
                pairs = read_pairs(paths[run_id])
                indices = selected_indices(len(pairs), fraction, seed, run_id)
                count = len(pairs) if indices is None else len(indices)
                for begin in range(0, count, batch_size):
                    if indices is None:
                        selected = np.asarray(pairs[begin : min(begin + batch_size, count)], dtype=np.float32)
                    else:
                        selected = np.asarray(pairs[indices[begin : begin + batch_size]], dtype=np.float32)
                    wepl = energies_to_wepl_vectorized(wepl_lut, selected[:, 4, 0], selected[:, 4, 1])
                    batch = {
                        "position_in": selected[:, 0, :],
                        "position_out": selected[:, 1, :],
                        "direction_in": selected[:, 2, :],
                        "direction_out": selected[:, 3, :],
                        "wepl_mm": wepl,
                    }
                    try:
                        residual_squared, measurements = projector.accumulate(
                            image, batch, 0.5 * run_id, numerator, denominator
                        )
                    except cp.cuda.memory.OutOfMemoryError as error:
                        raise RuntimeError(
                            f"GPU out of memory at batch_size={batch_size}; retry with --batch-size {max(1,batch_size//2)}"
                        ) from error
                    amount = len(selected)
                    subset_pairs += amount
                    completed_pairs += amount
                    subset_residual_squared += residual_squared
                    subset_measurements += measurements
                    batch_number += 1
                    if batch_number % progress_every == 0 or subset_pairs == subset_total:
                        elapsed = time.perf_counter() - total_start
                        rate = completed_pairs / elapsed if elapsed > 0 else 0.0
                        eta = (total_pairs - completed_pairs) / rate if rate > 0 else float("inf")
                        print(
                            f"epoch {epoch+1}/{epochs} subset {subset+1:02d}/{subsets}: "
                            f"{subset_pairs:,}/{subset_total:,} pairs, total={completed_pairs/total_pairs:6.2%}, "
                            f"rate={rate:,.0f} pairs/s, ETA={format_duration(eta)}",
                            flush=True,
                        )

            observed = denominator > 0.0
            update = cp.where(observed, np.float32(relaxation) * numerator / cp.maximum(denominator, 1.0e-20), 0.0)
            image += update
            cp.maximum(image, 0.0, out=image)
            image *= support
            cp.cuda.Stream.null.synchronize()
            subset_rmse = (
                float(np.sqrt(subset_residual_squared / subset_measurements))
                if subset_measurements else float("nan")
            )
            row = {
                "epoch": epoch + 1,
                "subset": subset,
                "relaxation": relaxation,
                "pairs": subset_pairs,
                "measurements": subset_measurements,
                "residual_rmse_mm": subset_rmse,
                "update_l2": float(cp.linalg.norm(update).get()),
                "update_max_abs": float(cp.max(cp.abs(update)).get()),
                "image_min": float(cp.min(image).get()),
                "image_max": float(cp.max(image).get()),
                "elapsed_seconds": time.perf_counter() - subset_start,
            }
            rows.append(row)
            epoch_residual_squared += subset_residual_squared
            epoch_measurements += subset_measurements
            print(
                f"updated epoch {epoch+1}/{epochs} subset {subset+1:02d}/{subsets}: "
                f"valid={subset_measurements:,}, rmse={subset_rmse:.5f} mm, "
                f"max_update={row['update_max_abs']:.5f}, elapsed={format_duration(row['elapsed_seconds'])}",
                flush=True,
            )
            del numerator, denominator, update

        if regularizer != "none" and (epoch + 1) % regularization_every == 0:
            print(
                f"regularization epoch {epoch+1}/{epochs}: starting {regularizer} proximal step",
                flush=True,
            )

            def regularization_progress(iteration: int, total: int, relative_change: float) -> None:
                print(
                    f"regularization epoch {epoch+1}/{epochs}: iteration {iteration:03d}/{total}, "
                    f"relative_change={relative_change:.3e}",
                    flush=True,
                )

            image, regularization_metrics = proximal_regularize(
                image,
                support,
                kind=regularizer,
                weight=regularization_weight,
                iterations=regularization_iterations,
                huber_delta=huber_delta,
                primal_step=primal_step,
                dual_step=dual_step,
                progress_callback=regularization_progress,
            )
            regularization_metrics = {"epoch": epoch + 1, **regularization_metrics}
            regularization_rows.append(regularization_metrics)
            print(
                f"regularization epoch {epoch+1}/{epochs}: completed in "
                f"{regularization_metrics['elapsed_seconds']:.2f}s, "
                f"l2_change={regularization_metrics['l2_change']:.5f}, "
                f"max_change={regularization_metrics['max_abs_change']:.5f}",
                flush=True,
            )

        epoch_image = cp.asnumpy(image)
        if bool(config.get("checkpoint_every_epoch", True)):
            write_image_2d(recon_dir / f"epoch_{epoch+1:02d}.mhd", epoch_image, spacing, origin)
        epoch_rmse = float(np.sqrt(epoch_residual_squared / epoch_measurements))
        print(
            f"completed epoch {epoch+1}/{epochs}: valid={epoch_measurements:,}, rmse={epoch_rmse:.5f} mm, "
            f"epoch_time={format_duration(time.perf_counter()-epoch_start)}",
            flush=True,
        )

    final_image = cp.asnumpy(image)
    write_image_2d(recon_dir / "recon_iterative_gpu.mhd", final_image, spacing, origin)
    with (qc_dir / "iteration_history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if regularization_rows:
        with (qc_dir / "regularization_history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(regularization_rows[0]))
            writer.writeheader()
            writer.writerows(regularization_rows)
    effective = {
        **config,
        "sample_fraction": fraction,
        "runs": runs,
        "subsets": subsets,
        "epochs": epochs,
        "relaxation": relaxation0,
        "relaxation_decay": relaxation_decay,
        "grid_size": size,
        "grid_spacing_mm": spacing,
        "path_step_mm": step,
        "batch_size": batch_size,
        "initialization": initialization,
        "device": device,
        "regularizer": regularizer,
        "regularization_weight": regularization_weight,
        "regularization_iterations": regularization_iterations,
        "regularization_every_epochs": regularization_every,
        "huber_delta": huber_delta,
        "primal_step": primal_step,
        "dual_step": dual_step,
    }
    summary = {
        "status": "PASS" if np.isfinite(final_image).all() and not np.any(final_image[~support_cpu]) else "FAIL",
        "started": started.isoformat(timespec="seconds"),
        "stopped": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - total_start,
        "gpu": gpu_name,
        "config": effective,
        "pairs_dir": str(args.pairs_dir.resolve()),
        "pairs_per_epoch": pairs_per_epoch,
        "grid": {"size": [size, 1, size], "spacing_mm": [spacing, 1.0, spacing],
                 "origin_mm": [origin, 0.0, origin]},
        "finite": bool(np.isfinite(final_image).all()),
        "support_outside_nonzero": int(np.count_nonzero(final_image[~support_cpu])),
        "output": str((recon_dir / "recon_iterative_gpu.mhd").resolve()),
        "regularization_applications": len(regularization_rows),
    }

    red_truth, truth_x, truth_z, _ = rsp_metrics.read_mhd(truth_dir / "truth_red.mhd")
    rsp_truth, rsp_x, rsp_z, _ = rsp_metrics.read_mhd(truth_dir / "truth_rsp_200mev.mhd")
    if not (np.array_equal(truth_x, rsp_x) and np.array_equal(truth_z, rsp_z)):
        raise RuntimeError("analytic truth grids differ")
    truth_config = experiment.get("truth", {})
    uniform_water = truth_config.get("kind") == "uniform_water"
    if uniform_water:
        centers = []
    else:
        definition = json.loads(
            (simulation_code / "truth_geometry_definition.json").read_text(
                encoding="utf-8"
            )
        )
        centers = definition["geometry"]["insert_centers_xz_mm"]
    metric_rows = []
    checkpoints = [("initial", 0, recon_dir / "initial.mhd")]
    checkpoints.extend(
        (f"epoch_{epoch:02d}", epoch, recon_dir / f"epoch_{epoch:02d}.mhd")
        for epoch in range(1, epochs + 1)
    )
    for label, epoch, checkpoint in checkpoints:
        checkpoint_image, image_x, image_z, _ = rsp_metrics.read_mhd(checkpoint)
        if not (np.array_equal(image_x, truth_x) and np.array_equal(image_z, truth_z)):
            raise RuntimeError(f"truth grid differs from {checkpoint}")
        epoch_rows = [row for row in rows if int(row["epoch"]) == epoch]
        measurements = sum(int(row["measurements"]) for row in epoch_rows)
        residual = (
            float(np.sqrt(sum(float(row["residual_rmse_mm"]) ** 2 * int(row["measurements"]) for row in epoch_rows) / measurements))
            if measurements else float("nan")
        )
        xx_metric, zz_metric = np.meshgrid(image_x, image_z)
        rr_metric = np.hypot(xx_metric, zz_metric)
        if uniform_water:
            core = rr_metric <= 90.0
            phantom = red_truth > 1.0e-6
            outside = rr_metric > 102.0
            effective_rsp = float(
                truth_config.get("effective_rsp_200mev_s6", 1.0)
            )
            metric_rows.append({
                "checkpoint": label,
                "epoch": epoch,
                "water_mean": float(checkpoint_image[core].mean()),
                "water_std": float(checkpoint_image[core].std()),
                "water_bias_vs_fixed_rsp": float(
                    checkpoint_image[core].mean()
                    - float(truth_config.get("rsp_200mev", 1.0))
                ),
                "water_bias_vs_s6_effective_rsp": float(
                    checkpoint_image[core].mean() - effective_rsp
                ),
                "phantom_rsp_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                checkpoint_image[phantom] - rsp_truth[phantom],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "phantom_effective_rsp_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                checkpoint_image[phantom]
                                - effective_rsp * red_truth[phantom],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "outside_rmse": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                checkpoint_image[outside], dtype=np.float64
                            )
                        )
                    )
                ),
                "outside_nonzero": int(
                    np.count_nonzero(checkpoint_image[rr_metric > radius])
                ),
                "wepl_residual_rmse_mm": residual,
            })
        else:
            values, _ = rsp_metrics.metrics_for(
                checkpoint_image, red_truth, rsp_truth, image_x, image_z, centers
            )
            edges = rsp_metrics.aluminium_edge_widths(
                checkpoint_image, image_x, image_z, centers
            )
            widths = np.array([
                row["width_10_90_mm"]
                for row in edges
                if row["distance_from_isocenter_mm"] > 0 and row["valid"]
            ])
            inner = np.array([row["inner_value"] for row in edges])
            metric_rows.append({
                "checkpoint": label,
                "epoch": epoch,
                "water_mean": values["water_mean"],
                "water_std": values["water_std"],
                "phantom_rsp_rmse": values["phantom_rmse_vs_rsp_truth"],
                "aluminium_inner_mean": float(inner.mean()),
                "aluminium_platform_rsp_recovery": float(inner.mean() / 2.1189760409708303),
                "roi_cnr_median": values["insert_roi_cnr_median"],
                "edge_10_90_median_mm": float(np.median(widths)),
                "outside_nonzero": int(
                    np.count_nonzero(checkpoint_image[rr_metric > radius])
                ),
                "wepl_residual_rmse_mm": residual,
            })
    with (qc_dir / "rsp_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    summary["rsp_metrics"] = metric_rows
    summary["experiment"] = args.experiment
    (qc_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
