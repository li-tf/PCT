#!/usr/bin/env python3
"""Run the three gated batches of Stage 8B low-fluence adaptation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import numpy as np


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
REPO = CODE.parent
QC = HERE / "qc"
PROGRESS = QC / "progress.json"
CONFIG_PATH = HERE / "stage8b_config.json"
sys.path[:0] = [
    str(HERE), str(CODE), str(CODE / "preprocessing"),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "analytic_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage7_detector_effects"),
    str(CODE / "research_stages/stage7b_noise_robustness"),
    str(CODE / "research_stages/stage7c_fluence_sensitivity"),
]

from stage8b_data import (  # noqa: E402
    fraction_tag, group_root, prepare_angular_noise_condition_run,
    subset_angular_from_stage7c_parent, subset_from_stage7c_parent,
)
from preprocessing import projection  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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
    config["_noise_model"] = str(resolve(config["noise_model"]))
    return config


def hash_inputs(config: dict[str, Any], raw_root: Path) -> str:
    payload = {
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "raw_root": str(raw_root.resolve()),
        "sources": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), HERE / "stage8b_data.py")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preflight(config: dict[str, Any], raw_root: Path, device: int, action: str, force: bool) -> None:
    if not resolve(config["wepl_model"]).is_file():
        raise FileNotFoundError(config["wepl_model"])
    stage7c = resolve(config["stage7c_preprocessing"])
    required = [
        stage7c / "event_ids/combined_0p2mm_1pct/events0000.npy",
        stage7c / "combined_0p2mm_1pct/seed_20260730/f025/pairs/pairs0000.mhd",
        stage7c / "combined_0p2mm_1pct/seed_20260730/f010/pairs/pairs0000.mhd",
    ]
    if action in {"optimize", "noise-weighting"}:
        acquisition = config["optimization_acquisition"]
        required.extend([
            resolve(acquisition["baseline_qc"]),
            resolve(acquisition["baseline_reconstruction"])
            / "iterative/recon/epoch_05.mhd",
            stage7c
            / "combined_0p2mm_1pct"
            / f"seed_{int(config['validation_seed'])}"
            / "f025/pairs/pairs0000.mhd",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    if action in {"optimize", "noise-weighting"} and not raw_root.is_dir():
        missing.append(str(raw_root))
    free = shutil.disk_usage(resolve(config["preprocessing_output"]).parent).free
    required_free = int(float(config["minimum_free_gib"]) * 1024**3)
    try:
        import cupy as cp

        cp.cuda.Device(device).use()
        cp.asarray([1], dtype=cp.float32).sum().get()
        gpu = cp.cuda.runtime.getDeviceProperties(device)["name"]
        if isinstance(gpu, bytes):
            gpu = gpu.decode()
    except Exception as error:
        raise RuntimeError(f"CUDA preflight failed: {error}") from error
    result = {
        "status": "PASS" if not missing and free >= required_free else "FAIL",
        "action": action, "missing": missing, "free_bytes": free,
        "required_free_bytes": required_free, "gpu": str(gpu),
    }
    atomic_json(QC / "preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Stage 8B preflight failed: {result}")
    manifest = (
        QC / "input_manifest.json"
        if action == "transition"
        else QC / f"input_manifest_{action.replace('-', '_')}.json"
    )
    digest = hash_inputs(config, raw_root)
    if manifest.is_file() and read_json(manifest).get("config_sha256") != digest and not force:
        raise RuntimeError("Stage 8B configuration/source changed; use a new output or --force")
    atomic_json(manifest, {
        "config_sha256": digest,
        "raw_root": str(raw_root),
        "version": int(config["version"]),
        "action": action,
    })


def ensure_geometry(
    config: dict[str, Any],
    runs: int | None = None,
    destination_name: str = "geometry.xml",
) -> Path:
    actual_runs = int(runs or config["runs"])
    if actual_runs != int(config["runs"]):
        destination = QC / destination_name
        if destination.is_file():
            return destination
        acquisition = read_json(CODE / "experiments/experiment0716.json")["acquisition"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            str(REPO / ".venv-gate/bin/rtksimulatedgeometry"),
            "--nproj", str(actual_runs), "--first_angle", "0", "--arc", "360",
            "--sid", str(acquisition["source_to_isocenter_mm"]),
            "--sdd", str(acquisition["source_to_detector_mm"]),
            "--output", str(destination),
        ], check=True)
        return destination
    source = CODE / "research_stages/stage7c_fluence_sensitivity/qc/geometry.xml"
    destination = QC / destination_name
    if destination.is_file():
        return destination
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination
    acquisition = read_json(CODE / "experiments/experiment0716.json")["acquisition"]
    subprocess.run([
        str(REPO / ".venv-gate/bin/rtksimulatedgeometry"), "--nproj", "720",
        "--first_angle", str(acquisition["first_angle_deg"]), "--arc", str(acquisition["arc_deg"]),
        "--sid", str(acquisition["source_to_isocenter_mm"]),
        "--sdd", str(acquisition["source_to_detector_mm"]), "--output", str(destination),
    ], check=True)
    return destination


def run_parallel(function: Callable[..., dict[str, Any]], arguments: list[tuple], jobs: int, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(function, *args) for args in arguments]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (len(futures) - done)
            update_progress(stage=label, completed_runs=done, total_runs=len(futures), task_eta_seconds=eta)
            if done % 20 == 0 or done == len(futures):
                print(f"{label}: {done}/{len(futures)}, ETA={eta/60:.1f} min", flush=True)
    return rows


def transition_root(config: dict[str, Any], fraction: float) -> Path:
    return group_root(
        resolve(config["preprocessing_output"]) / "transition",
        int(config["transition_seed"]), fraction,
    )


def stage7c_root(config: dict[str, Any], seed: int, fraction: float, data: bool) -> Path:
    base = resolve(config["stage7c_preprocessing"] if data else config["stage7c_reconstruction"])
    return base / "combined_0p2mm_1pct" / f"seed_{seed}" / f"f{int(round(100*fraction)):03d}"


def source_root(config: dict[str, Any], seed: int, fraction: float) -> Path:
    if seed == int(config["transition_seed"]) and fraction in tuple(float(v) for v in config["stage7c_reused_fractions"]):
        return stage7c_root(config, seed, fraction, True)
    return group_root(resolve(config["preprocessing_output"]) / "transition", seed, fraction)


def reconstruction_root(config: dict[str, Any], seed: int, fraction: float) -> Path:
    if seed == int(config["transition_seed"]) and fraction in tuple(float(v) for v in config["stage7c_reused_fractions"]):
        return stage7c_root(config, seed, fraction, False)
    return group_root(resolve(config["reconstruction_output"]) / "transition", seed, fraction)


def prepare_nested(config: dict[str, Any], seed: int, fractions: list[float], jobs: int, force: bool) -> None:
    stage7c = resolve(config["stage7c_preprocessing"])
    output = resolve(config["preprocessing_output"]) / "transition"
    arguments = [
        (run_id, str(stage7c), str(output), seed, fraction, force)
        for fraction in fractions for run_id in range(int(config["runs"]))
    ]
    rows = run_parallel(subset_from_stage7c_parent, arguments, jobs, "prepare-transition")
    existing_path = QC / "transition_counts_by_run.csv"
    if existing_path.is_file() and not force:
        with existing_path.open(encoding="utf-8") as stream:
            existing = list(csv.DictReader(stream))
        keys = {(int(row["run_id"]), int(row["seed"]), float(row["fraction"])) for row in rows}
        rows = [row for row in existing if (int(row["run_id"]), int(row["seed"]), float(row["fraction"])) not in keys] + rows
    write_csv(QC / "transition_counts_by_run.csv", rows)


def optimization_acquisition(config: dict[str, Any]) -> dict[str, Any]:
    value = dict(config["optimization_acquisition"])
    value["runs"] = int(value["runs"])
    value["angle_step_deg"] = float(value["angle_step_deg"])
    value["per_angle_fraction"] = float(value["per_angle_fraction"])
    return value


def angular_source_root(
    config: dict[str, Any], condition: str, seed: int, fraction: float,
) -> Path:
    return group_root(
        resolve(config["preprocessing_output"])
        / "angular_noise_sources" / condition,
        seed,
        fraction,
    )


def angular_run_mapping(config: dict[str, Any]) -> list[tuple[int, int]]:
    acquisition = optimization_acquisition(config)
    offset = int(acquisition["original_run_offset"])
    stride = int(acquisition["original_run_stride"])
    mapping = [
        (output_run, offset + stride * output_run)
        for output_run in range(int(acquisition["runs"]))
    ]
    if not mapping or mapping[-1][1] >= int(config["runs"]):
        raise RuntimeError(f"invalid angular run mapping: {mapping[-1:]}")
    return mapping


def ensure_angular_seed_data(
    config: dict[str, Any], raw_root: Path, seed: int, fraction: float,
    jobs: int, force: bool, condition_name: str = "combined_0p2mm_1pct",
    require_metadata: bool = False,
) -> Path:
    """Build the frozen 360-view acquisition for one seed and condition."""
    acquisition = optimization_acquisition(config)
    root = angular_source_root(config, condition_name, seed, fraction)
    expected = [
        root / "pairs" / f"pairs{run_id:04d}.mhd"
        for run_id in range(int(acquisition["runs"]))
    ]
    expected_meta = [
        root / "metadata" / f"meta{run_id:04d}.npz"
        for run_id in range(int(acquisition["runs"]))
    ]
    if (
        all(path.is_file() for path in expected)
        and (not require_metadata or all(path.is_file() for path in expected_meta))
        and not force
    ):
        return root

    stage7c = resolve(config["stage7c_preprocessing"])
    parent = (
        stage7c / "combined_0p2mm_1pct" / f"seed_{seed}"
        / "f025/pairs/pairs0000.mhd"
    )
    mapping = angular_run_mapping(config)
    if condition_name == "combined_0p2mm_1pct" and parent.is_file() and not require_metadata:
        output = resolve(config["preprocessing_output"]) / "angular_noise_sources" / condition_name
        tasks = [
            (
                output_run, original_run, str(stage7c), str(output), seed,
                fraction, force,
            )
            for output_run, original_run in mapping
        ]
        rows = run_parallel(
            subset_angular_from_stage7c_parent,
            tasks,
            jobs,
            f"prepare-angular-seed-{seed}",
        )
    else:
        if not raw_root.is_dir():
            raise FileNotFoundError(raw_root)
        condition = dict(config["noise_conditions"][condition_name])
        tasks = [
            (
                output_run, original_run, str(raw_root),
                str(resolve(config["preprocessing_output"])), condition_name,
                condition, seed, fraction, 20260810, config, force,
            )
            for output_run, original_run in mapping
        ]
        rows = run_parallel(
            prepare_angular_noise_condition_run,
            tasks,
            jobs,
            f"prepare-angular-root-{condition_name}-{seed}",
        )
    count_path = QC / "angular_optimization_counts_by_run.csv"
    existing: list[dict[str, Any]] = []
    if count_path.is_file() and not force:
        existing = list(csv.DictReader(count_path.open(encoding="utf-8")))
    new_keys = {
        (str(row.get("condition", condition_name)), int(row["output_run"]), int(seed))
        for row in rows
    }
    existing = [
        row for row in existing
        if (str(row.get("condition", condition_name)), int(row["output_run"]), int(row["seed"]))
        not in new_keys
    ]
    normalized = [
        {"condition": condition_name, "seed": seed, **row}
        for row in rows
    ]
    write_csv(count_path, existing + normalized)
    if not all(path.is_file() for path in expected):
        raise RuntimeError(f"angular data incomplete for {condition_name}, seed {seed}")
    return root


def ensure_ddb_and_initial(
    config: dict[str, Any], source: Path, destination: Path, jobs: int,
    force: bool, label: str, acquisition: dict[str, Any] | None = None,
) -> Path:
    actual = acquisition or {
        "runs": int(config["runs"]),
        "angle_step_deg": float(config["angle_step_deg"]),
    }
    runs = int(actual["runs"])
    ddb = source / "projections_ddb"
    ddb.mkdir(parents=True, exist_ok=True)
    pending = [run_id for run_id in range(runs) if force or not (ddb / f"proj{run_id:04d}.mhd").is_file()]
    if pending:
        args = [(run_id, str(source / "pairs"), str(source), False, "projections_ddb") for run_id in pending]
        run_parallel(projection.process_run, args, jobs, f"ddb-{label}")
    initial = destination / "analytic/recon_ddb_nohann.mhd"
    if initial.is_file() and not force:
        return initial
    initial.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(REPO / ".venv-gate/bin/pctfdk"), "--lowmem", "--geometry", str(
            ensure_geometry(
                config,
                runs=runs,
                destination_name=("geometry.xml" if runs == int(config["runs"]) else f"geometry_{runs}.xml"),
            )
        ),
        "--path", str(ddb), "--regexp", r"proj....\.mhd", "--output", str(initial),
        "--dimension", "2100", "1", "2100", "--spacing", "0.1", "1", "0.1", "--hann", "0", "--verbose",
    ]
    log = QC / "logs" / f"fdk_{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return initial


def run_logged(command: list[str], log: Path, group: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"epoch\s+(\d+)/(\d+).*subset\s+(\d+)/(\d+)")
    rate_pattern = re.compile(r"rate=([\d,]+)\s+pairs/s\s+ETA=([0-9:]+)")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            match = pattern.search(line)
            if match:
                update_progress(stage="reconstruct", group=group, epoch=int(match.group(1)), total_epochs=int(match.group(2)), subset=int(match.group(3)), total_subsets=int(match.group(4)))
            rate = rate_pattern.search(line)
            if rate:
                parts = [int(value) for value in rate.group(2).split(":")]
                eta = 0
                for value in parts:
                    eta = eta * 60 + value
                update_progress(pairs_per_second=int(rate.group(1).replace(",", "")), task_eta_seconds=eta)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def iterative_command(
    config: dict[str, Any], pairs: Path, initial: Path, output: Path,
    qc: Path, settings: dict[str, Any], device: int,
    epochs: int | None = None, acquisition: dict[str, Any] | None = None,
) -> list[str]:
    actual = acquisition or {
        "runs": int(config["runs"]),
        "angle_step_deg": float(config["angle_step_deg"]),
    }
    return [
        sys.executable, str(CODE / "iterative_reconstruction/run_iterative_reconstruction.py"),
        "--experiment", "0716", "--pairs-dir", str(pairs), "--initial-image", str(initial),
        "--output-dir", str(output), "--qc-dir", str(qc), "--runs", str(actual["runs"]),
        "--angle-step-deg", str(actual["angle_step_deg"]), "--phantom-radius-mm", str(config["phantom_radius_mm"]),
        "--air-wepl-slope", "0", "--wepl-model", "g4_water_calibrated", "--wepl-calibration", str(config["_wepl_model"]),
        "--epochs", str(epochs or settings["epochs"]), "--sample-fraction", "1",
        "--grid-size", str(settings["grid_size"]), "--grid-spacing-mm", str(settings["grid_spacing_mm"]),
        "--path-step-mm", str(settings["path_step_mm"]), "--batch-size", str(settings["batch_size"]),
        "--subsets", str(settings["subsets"]), "--relaxation", str(settings["relaxation"]),
        "--relaxation-decay", str(settings["relaxation_decay"]), "--initialization", "fdk_nohann", "--device", str(device),
        "--regularizer", str(settings["regularizer"]), "--regularization-weight", str(settings["regularization_weight"]),
        "--regularization-iterations", str(settings["regularization_iterations"]),
        "--regularization-every-epochs", str(settings["regularization_every_epochs"]),
        "--huber-delta", str(settings["huber_delta"]), "--primal-step", str(settings["primal_step"]),
        "--dual-step", str(settings["dual_step"]), "--skip-truth-metrics",
    ]


def frozen_settings(config: dict[str, Any]) -> dict[str, Any]:
    return dict(read_json(resolve(config["frozen_reconstruction"]))["reconstruction"])


def run_iterative(
    config: dict[str, Any], pairs: Path, initial: Path, root: Path,
    name: str, settings: dict[str, Any], device: int, force: bool,
    epochs: int | None = None, acquisition: dict[str, Any] | None = None,
) -> Path:
    final_epoch = int(epochs or settings["epochs"])
    final = root / "recon" / f"epoch_{final_epoch:02d}.mhd"
    if final.is_file() and not force:
        return final
    command = iterative_command(
        config, pairs, initial, root, QC / "reconstruction" / name,
        settings, device, epochs, acquisition,
    )
    if force:
        command.append("--force")
    run_logged(command, QC / "logs" / f"iterative_{name}.log", name)
    return final


def image_metrics(path: Path) -> dict[str, Any]:
    import run_stage7b

    return run_stage7b.image_metrics(path)


def numeric_metrics(path: Path) -> dict[str, Any]:
    return {key: value for key, value in image_metrics(path).items() if isinstance(value, (int, float, bool, np.number))}


def aluminium_error(row: dict[str, Any]) -> float:
    return float(row["insert_peak_mean"]) / 2.094511207867794 - 1.0


def full_baseline_metrics(config: dict[str, Any]) -> dict[str, Any]:
    path = resolve(config["stage7_reconstruction"]) / "full/energy_1pct/iterative/recon/epoch_05.mhd"
    return numeric_metrics(path)


def qualify(row: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> tuple[bool, dict[str, float]]:
    selection = config["selection"]
    water_bias = abs(float(row["water_mean"]) / 0.9997458098472691 - 1.0)
    al_deg = (abs(aluminium_error(row)) - abs(aluminium_error(baseline))) * 100.0
    edge_ratio = float(row["aluminium_edge_10_90_median_mm"]) / float(baseline["aluminium_edge_10_90_median_mm"])
    rmse_ratio = float(row["phantom_rmse_vs_rsp_truth"]) / float(baseline["phantom_rmse_vs_rsp_truth"])
    detail = {"water_bias_fraction": water_bias, "phantom_rmse_ratio": rmse_ratio, "aluminium_error_degradation_pp": al_deg, "edge_width_ratio": edge_ratio}
    passed = (
        water_bias <= float(selection["water_bias_max_fraction"])
        and float(row["water_std"]) <= float(selection["water_std_max"])
        and rmse_ratio <= float(selection["phantom_rmse_ratio_max"])
        and al_deg <= float(selection["aluminium_error_degradation_max_pp"])
        and float(row["insert_cnr_median"]) >= float(selection["cnr_min"])
        and edge_ratio <= float(selection["edge_width_ratio_max"])
        and float(row.get("finite_fraction", 1.0)) == 1.0
        and float(row.get("outside_mean", 0.0)) == 0.0
    )
    return passed, detail


def transition(config: dict[str, Any], jobs: int, device: int, force: bool) -> dict[str, Any]:
    """Run and close the first Stage-8B batch."""
    update_progress(status="RUNNING", batch="transition", stage="prepare")
    seed = int(config["transition_seed"])
    fractions = [float(v) for v in config["transition_fractions"]]
    reused = set(float(v) for v in config["stage7c_reused_fractions"])
    new = [fraction for fraction in fractions if fraction not in reused]
    settings = frozen_settings(config)
    for index, fraction in enumerate(new, 1):
        # Materialize one virtual EventID subset at a time because the mature
        # pctbinning/OS-SART executables require pair files.  The materialized
        # pairs and DDB are removed after their checkpoints pass QC.
        prepare_nested(config, seed, [fraction], jobs, force)
        source = source_root(config, seed, fraction)
        root = reconstruction_root(config, seed, fraction)
        label = fraction_tag(fraction)
        update_progress(stage="transition", group=label, completed_groups=index - 1, total_groups=len(new))
        initial = ensure_ddb_and_initial(config, source, root, jobs, force, label)
        run_iterative(config, source / "pairs", initial, root / "iterative", f"transition/{label}", settings, device, force)
        shutil.rmtree(source / "pairs", ignore_errors=True)
        shutil.rmtree(source / "projections_ddb", ignore_errors=True)
    baseline = full_baseline_metrics(config)
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        root = reconstruction_root(config, seed, fraction)
        initial = root / "analytic/recon_ddb_nohann.mhd"
        for epoch in range(0, int(settings["epochs"]) + 1):
            path = initial if epoch == 0 else root / "iterative/recon" / f"epoch_{epoch:02d}.mhd"
            if not path.is_file():
                raise FileNotFoundError(path)
            metrics = numeric_metrics(path)
            passed, details = qualify(metrics, baseline, config)
            rows.append({
                "seed": seed, "fraction": fraction,
                "fluence_per_mm2_projection": float(config["nominal_full_fluence_per_mm2_projection"]) * fraction,
                "epoch": epoch, "image_path": str(path), **metrics, **details, "passed": passed,
            })
    write_csv(QC / "transition_metrics.csv", rows)
    final = [row for row in rows if int(row["epoch"]) == int(settings["epochs"])]
    final.sort(key=lambda row: float(row["fraction"]), reverse=True)
    status_sequence = [bool(row["passed"]) for row in final]
    monotonic = not any(
        (not status_sequence[i]) and status_sequence[j]
        for i in range(len(status_sequence)) for j in range(i + 1, len(status_sequence))
    )
    passing = sorted(float(row["fraction"]) for row in final if row["passed"])
    f_pass = min(passing) if passing else None
    failing_below = sorted(
        (float(row["fraction"]) for row in final if not row["passed"] and (f_pass is None or float(row["fraction"]) < f_pass)),
        reverse=True,
    )
    f_dev = failing_below[0] if failing_below else None
    decision = {
        "status": "PASS" if monotonic and f_pass is not None and f_dev is not None else "NEEDS_REPLICATES",
        "monotonic": monotonic, "f_pass": f_pass, "f_dev": f_dev,
        "fractions": [{"fraction": row["fraction"], "passed": row["passed"], "water_std": row["water_std"], "phantom_rmse": row["phantom_rmse_vs_rsp_truth"]} for row in final],
        "optimization_allowed": bool(monotonic and f_dev is not None),
    }
    atomic_json(QC / "transition_decision.json", decision)
    build_transition_figure(rows)
    lines = [
        "# Stage 8B任务1：低通量转折点", "",
        f"状态：**{decision['status']}**。", "",
        f"最低合格通量：{'—' if f_pass is None else f'{100*f_pass:g}%'}。",
        f"任务2开发通量：{'—' if f_dev is None else f'{100*f_dev:g}%'}。", "",
        "完整的初值及epoch 1--5指标见`transition_metrics.csv`。",
    ]
    (QC / "transition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_progress(status="COMPLETE", batch="transition", stage="decision", task_eta_seconds=0)
    return decision


def build_transition_figure(rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage8b")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    final_epoch = max(int(row["epoch"]) for row in rows)
    selected = sorted((row for row in rows if int(row["epoch"]) == final_epoch), key=lambda row: float(row["fraction"]))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    x = [float(row["fluence_per_mm2_projection"]) for row in selected]
    axes[0].plot(x, [float(row["water_std"]) for row in selected], "o-")
    axes[0].axhline(0.01, color="crimson", ls="--", lw=1)
    axes[0].set_ylabel("Water RSP standard deviation")
    axes[1].plot(x, [float(row["phantom_rmse_vs_rsp_truth"]) for row in selected], "o-")
    axes[1].set_ylabel("Phantom RSP RMSE")
    for axis in axes:
        axis.set_xlabel("Nominal fluence (protons/mm²/projection)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "transition_curve.png", dpi=180)
    plt.close(fig)


def candidate_settings(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    value = dict(base)
    value.update(changes)
    return value


def candidate_epoch_rows(name: str, root: Path, settings: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, int(settings["epochs"]) + 1):
        path = root / "recon" / f"epoch_{epoch:02d}.mhd"
        metrics = numeric_metrics(path)
        passed, details = qualify(metrics, baseline, config)
        rows.append({"candidate": name, "epoch": epoch, "image_path": str(path), **settings, **metrics, **details, "constraints_passed": passed})
    return rows


def choose_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if bool(row["constraints_passed"])]
    pool = eligible or rows
    return min(pool, key=lambda row: (float(row["phantom_rmse_vs_rsp_truth"]), float(row["water_std"]), int(row["epoch"])))


def ensure_seed_data(config: dict[str, Any], raw_root: Path, seed: int, fraction: float, jobs: int, force: bool) -> Path:
    """Prepare a combined-noise subset for optimization validation/test."""
    output = resolve(config["preprocessing_output"]) / "optimization_data"
    root = group_root(output, seed, fraction)
    if all((root / "pairs" / f"pairs{run_id:04d}.mhd").is_file() for run_id in range(int(config["runs"]))) and not force:
        return root
    stage7c = resolve(config["stage7c_preprocessing"])
    parent = stage7c / "combined_0p2mm_1pct" / f"seed_{seed}" / "f025/pairs/pairs0000.mhd"
    if parent.is_file():
        args = [(run_id, str(stage7c), str(output), seed, fraction, force) for run_id in range(int(config["runs"]))]
        run_parallel(subset_from_stage7c_parent, args, jobs, f"prepare-seed-{seed}")
        return root
    # A genuinely unseen seed is built from ROOT with the same detector model.
    condition = dict(config["condition"])
    condition_name = "optimization_combined"
    args = [
        (run_id, str(raw_root), str(resolve(config["preprocessing_output"])), condition_name, condition, seed, fraction, 20260713, config, force)
        for run_id in range(int(config["runs"]))
    ]
    run_parallel(prepare_noise_condition_run, args, jobs, f"prepare-locked-{seed}")
    source = resolve(config["preprocessing_output"]) / f"noise_sources/{condition_name}" / f"seed_{seed}" / fraction_tag(fraction)
    return source


def make_lowpass_initial(source: Path, destination: Path) -> Path:
    from iterative_reconstruction.mhd_io import read_image_2d, resample_to_grid, write_image_2d

    image, spacing, origin = read_image_2d(source)
    coarse, coarse_origin = resample_to_grid(image, spacing, origin, 420, 0.5)
    fine, fine_origin = resample_to_grid(coarse, [0.5, 1.0, 0.5], [coarse_origin, 0.0, coarse_origin], 2100, 0.1)
    write_image_2d(destination, fine, 0.1, fine_origin)
    return destination


def run_candidate(
    config: dict[str, Any], pairs_root: Path, initial: Path, candidates_root: Path,
    name: str, settings: dict[str, Any], device: int, force: bool,
    acquisition: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = candidates_root / name
    if name == "multiscale_0p5_to_0p1":
        coarse_settings = candidate_settings(
            settings,
            grid_size=int(config["optimization"]["multiscale_coarse_size"]),
            grid_spacing_mm=float(config["optimization"]["multiscale_coarse_spacing_mm"]),
            path_step_mm=float(config["optimization"]["multiscale_coarse_path_step_mm"]),
            epochs=int(config["optimization"]["multiscale_coarse_epochs"]),
        )
        coarse = run_iterative(
            config, pairs_root / "pairs", initial, root / "coarse", f"optimization/{name}/coarse",
            coarse_settings, device, force, epochs=coarse_settings["epochs"],
            acquisition=acquisition,
        )
        from iterative_reconstruction.mhd_io import read_image_2d, resample_to_grid, write_image_2d

        image, spacing, origin = read_image_2d(coarse)
        fine, fine_origin = resample_to_grid(image, spacing, origin, 2100, 0.1)
        fine_initial = root / "fine_initial.mhd"
        write_image_2d(fine_initial, fine, 0.1, fine_origin)
        fine_settings = candidate_settings(
            settings, epochs=min(
                int(settings.get("epochs", config["optimization"]["multiscale_fine_epochs"])),
                int(config["optimization"]["multiscale_fine_epochs"]),
            ),
            grid_size=2100, grid_spacing_mm=0.1, path_step_mm=0.1,
        )
        final = run_iterative(
            config, pairs_root / "pairs", fine_initial, root / "fine", f"optimization/{name}/fine",
            fine_settings, device, force, epochs=fine_settings["epochs"],
            acquisition=acquisition,
        )
        return root / "fine", fine_settings
    run_initial = initial
    if name == "lowpass_0p5_upsampled":
        run_initial = make_lowpass_initial(initial, root / "lowpass_initial.mhd")
    run_iterative(
        config, pairs_root / "pairs", run_initial, root,
        f"optimization/{name}", settings, device, force,
        acquisition=acquisition,
    )
    return root, settings


def optimize(config: dict[str, Any], raw_root: Path, jobs: int, device: int, force: bool) -> dict[str, Any]:
    decision_path = QC / "optimization_decision.json"
    if decision_path.is_file() and not force:
        decision = read_json(decision_path)
        print("Stage 8B angular optimization already complete; reusing decision.")
        update_progress(
            status="COMPLETE", batch="optimize", stage="decision",
            task_eta_seconds=0,
        )
        return decision
    acquisition = optimization_acquisition(config)
    gate_path = resolve(acquisition["baseline_qc"])
    if not gate_path.is_file():
        raise RuntimeError("run the 360-angle × 20% baseline first")
    gate = read_json(gate_path)
    if (
        gate.get("status") != "PASS"
        or gate.get("recommended_development_condition") != acquisition["name"]
    ):
        raise RuntimeError(f"angular-fluence gate is not frozen: {gate}")
    fraction = float(acquisition["per_angle_fraction"])
    seed = int(config["transition_seed"])
    source = ensure_angular_seed_data(
        config, raw_root, seed, fraction, jobs, force,
    )
    baseline_root = resolve(acquisition["baseline_reconstruction"])
    root = (
        resolve(config["reconstruction_output"]) / "optimization"
        / acquisition["name"]
    )
    initial = baseline_root / "analytic/recon_ddb_nohann.mhd"
    if not initial.is_file():
        raise FileNotFoundError(initial)
    base = frozen_settings(config)
    baseline = full_baseline_metrics(config)
    all_rows: list[dict[str, Any]] = []
    settings_by_name: dict[str, dict[str, Any]] = {}

    def execute(name: str, settings: dict[str, Any], initial_override: Path = initial) -> dict[str, Any]:
        update_progress(status="RUNNING", batch="optimize", stage="screen", group=name)
        candidate_root, actual = run_candidate(
            config, source, initial_override, root / "development", name,
            settings, device, force, acquisition,
        )
        rows = candidate_epoch_rows(name, candidate_root, actual, baseline, config)
        all_rows.extend(rows)
        settings_by_name[name] = settings
        return choose_candidate(rows)

    # The transition baseline is imported as a candidate without recomputation.
    baseline_rows = []
    transition_recon = baseline_root / "iterative/recon"
    for epoch in range(1, int(base["epochs"]) + 1):
        path = transition_recon / f"epoch_{epoch:02d}.mhd"
        metrics = numeric_metrics(path)
        passed, details = qualify(metrics, baseline, config)
        baseline_rows.append({"candidate": "r0p25_d0p2_b0p0125", "epoch": epoch, "image_path": str(path), **base, **metrics, **details, "constraints_passed": passed})
    all_rows.extend(baseline_rows)
    settings_by_name["r0p25_d0p2_b0p0125"] = base

    relaxation_winners = [choose_candidate(baseline_rows)]
    for value in (0.05, 0.10, 0.15):
        name = f"r{value:g}_d0p2_b0p0125".replace(".", "p")
        relaxation_winners.append(execute(name, candidate_settings(base, relaxation=value)))
    best_relaxation = choose_candidate(relaxation_winners)
    current_name = str(best_relaxation["candidate"])
    current_settings = settings_by_name[current_name]

    decay_winners = [best_relaxation]
    for value in (0.1, 0.4):
        name = f"{current_name}_decay_{value:g}".replace(".", "p")
        decay_winners.append(execute(name, candidate_settings(current_settings, relaxation_decay=value)))
    best_decay = choose_candidate(decay_winners)
    current_name = str(best_decay["candidate"])
    current_settings = settings_by_name[current_name]

    beta_winners = [best_decay]
    for value in (0.025, 0.05):
        name = f"{current_name}_beta_{value:g}".replace(".", "p")
        beta_winners.append(execute(name, candidate_settings(current_settings, regularization_weight=value)))
    best_beta = choose_candidate(beta_winners)
    current_name = str(best_beta["candidate"])
    current_settings = settings_by_name[current_name]

    initial_winners = [best_beta]
    initial_winners.append(execute("lowpass_0p5_upsampled", current_settings))
    initial_winners.append(execute("multiscale_0p5_to_0p1", current_settings))
    write_csv(QC / "optimization_candidates.csv", all_rows)

    ranked = sorted(
        initial_winners,
        key=lambda row: (not bool(row["constraints_passed"]), float(row["phantom_rmse_vs_rsp_truth"])),
    )
    top_names = []
    for row in ranked:
        if row["candidate"] not in top_names:
            top_names.append(str(row["candidate"]))
        if len(top_names) == int(config["optimization"]["validation_top_k"]):
            break

    validation_seed = int(config["validation_seed"])
    validation_source = ensure_angular_seed_data(
        config, raw_root, validation_seed, fraction, jobs, force,
    )
    validation_root = (
        resolve(config["reconstruction_output"]) / "optimization_validation"
        / acquisition["name"] / f"seed_{validation_seed}"
    )
    validation_initial = ensure_ddb_and_initial(
        config, validation_source, validation_root, jobs, force,
        f"validation_{acquisition['name']}", acquisition,
    )
    validation_rows: list[dict[str, Any]] = []
    for name in top_names:
        settings = settings_by_name[name]
        candidate_root, actual = run_candidate(
            config, validation_source, validation_initial,
            validation_root / "candidates", name, settings, device, force,
            acquisition,
        )
        rows = candidate_epoch_rows(name, candidate_root, actual, baseline, config)
        validation_rows.extend(rows)
    write_csv(QC / "optimization_validation.csv", validation_rows)
    validation_best = choose_candidate(validation_rows)
    winner_name = str(validation_best["candidate"])
    winner_settings = settings_by_name[winner_name]

    test_seed = int(config["locked_test_seed"])
    test_source = ensure_angular_seed_data(
        config, raw_root, test_seed, fraction, jobs, force,
    )
    test_root = (
        resolve(config["reconstruction_output"]) / "optimization_test"
        / acquisition["name"] / f"seed_{test_seed}"
    )
    test_initial = ensure_ddb_and_initial(
        config, test_source, test_root, jobs, force,
        f"test_{acquisition['name']}", acquisition,
    )
    test_rows: list[dict[str, Any]] = []
    for name, settings in (("stage7c_baseline", base), (winner_name, winner_settings)):
        candidate_root, actual = run_candidate(
            config, test_source, test_initial, test_root / "candidates",
            name, settings, device, force, acquisition,
        )
        test_rows.extend(candidate_epoch_rows(name, candidate_root, actual, baseline, config))
    write_csv(QC / "optimization_test.csv", test_rows)
    test_baseline = choose_candidate([row for row in test_rows if row["candidate"] == "stage7c_baseline"])
    test_winner = choose_candidate([row for row in test_rows if row["candidate"] == winner_name])
    rmse_improvement = 1.0 - float(test_winner["phantom_rmse_vs_rsp_truth"]) / float(test_baseline["phantom_rmse_vs_rsp_truth"])
    noise_improvement = 1.0 - float(test_winner["water_std"]) / float(test_baseline["water_std"])
    promoted = bool(
        test_winner["constraints_passed"]
        and (rmse_improvement >= float(config["selection"]["candidate_rmse_improvement_min"])
             or noise_improvement >= float(config["selection"]["candidate_water_std_improvement_min"]))
    )
    lower_candidates = sorted(
        (float(value) for value in config["transition_fractions"] if float(value) < fraction),
        reverse=True,
    )
    extension_row = None
    if promoted and lower_candidates:
        lower = lower_candidates[0]
        lower_source = ensure_angular_seed_data(
            config, raw_root, seed, lower, jobs, force,
        )
        lower_root = (
            resolve(config["reconstruction_output"]) / "optimization_extension"
            / f"360_angles_x_{100*lower:g}pct"
        )
        lower_initial = ensure_ddb_and_initial(
            config, lower_source, lower_root, jobs, force,
            f"extension_360x{100*lower:g}", acquisition,
        )
        candidate_root, actual = run_candidate(
            config, lower_source, lower_initial, lower_root, winner_name,
            winner_settings, device, force, acquisition,
        )
        extension_row = choose_candidate(
            candidate_epoch_rows(winner_name, candidate_root, actual, baseline, config)
        )
        extension_row = {"fraction": lower, **extension_row}
        write_csv(QC / "optimization_extension.csv", [extension_row])
    else:
        write_csv(QC / "optimization_extension.csv", [])
    decision = {
        "status": "PASS",
        "acquisition": acquisition,
        "f_dev": fraction,
        "nominal_total_fraction": float(acquisition["nominal_total_fraction"]),
        "winner": winner_name,
        "winner_settings": winner_settings, "best_epoch": int(test_winner["epoch"]),
        "rmse_improvement_fraction": rmse_improvement,
        "water_std_improvement_fraction": noise_improvement,
        "promoted": promoted, "locked_test_seed": test_seed,
        "multiscale": winner_name in {"lowpass_0p5_upsampled", "multiscale_0p5_to_0p1"},
        "extension": extension_row,
        "minimum_fraction_after_optimization": (
            float(extension_row["fraction"])
            if extension_row is not None and bool(extension_row["constraints_passed"])
            else (fraction if promoted else None)
        ),
    }
    atomic_json(QC / "optimization_decision.json", decision)
    build_optimization_figure(all_rows, validation_rows, test_rows)
    (QC / "optimization_summary.md").write_text(
        "# Stage 8B任务2：低通量参数优化\n\n"
        f"开发条件：**360角度 × 每角度{100*fraction:g}%通量**，"
        f"总扫描通量为720角度全通量的"
        f"**{100*float(acquisition['nominal_total_fraction']):g}%**。\n\n"
        f"胜出候选：`{winner_name}`，锁定测试晋升：**{'是' if promoted else '否'}**。\n",
        encoding="utf-8",
    )
    for prepared in (source, validation_source, test_source):
        shutil.rmtree(prepared / "pairs", ignore_errors=True)
        shutil.rmtree(prepared / "projections_ddb", ignore_errors=True)
    update_progress(status="COMPLETE", batch="optimize", stage="decision", task_eta_seconds=0)
    return decision


def build_optimization_figure(
    development: list[dict[str, Any]], validation: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage8b")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def best_rows(rows):
        names = list(dict.fromkeys(str(row["candidate"]) for row in rows))
        return [choose_candidate([row for row in rows if row["candidate"] == name]) for name in names]

    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, title, rows in zip(axes, ("Development", "Validation", "Locked test"), (development, validation, test)):
        selected = best_rows(rows)
        labels = [str(row["candidate"]) for row in selected]
        axis.barh(labels, [float(row["phantom_rmse_vs_rsp_truth"]) for row in selected])
        axis.set_title(title)
        axis.set_xlabel("Phantom RSP RMSE")
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "optimization_comparison.png", dpi=180)
    plt.close(fig)


def noise_source_root(config: dict[str, Any], condition: str, seed: int, fraction: float) -> Path:
    return angular_source_root(config, condition, seed, fraction)


def prepare_noise_sources(
    config: dict[str, Any], raw_root: Path, fraction: float, jobs: int, force: bool,
) -> None:
    main = int(config["transition_seed"])
    validation = int(config["validation_seed"])
    test = int(config["locked_test_seed"])
    # All four conditions at the development seed; paired position/combined at
    # the two confirmation seeds.  The common noise seed makes the position
    # perturbation identical inside each paired comparison.
    requested = [(name, main) for name in config["noise_conditions"]]
    requested += [(name, seed) for seed in (validation, test) for name in ("position_0p2mm", "combined_0p2mm_1pct")]
    rows: list[dict[str, Any]] = []
    for condition_name, seed in requested:
        root = ensure_angular_seed_data(
            config, raw_root, seed, fraction, jobs, force,
            condition_name=condition_name, require_metadata=True,
        )
        for output_run, original_run in angular_run_mapping(config):
            with np.load(
                root / "metadata" / f"meta{output_run:04d}.npz",
                allow_pickle=False,
            ) as meta:
                selected = len(meta["event_id"])
            rows.append({
                "condition": condition_name,
                "seed": seed,
                "output_run": output_run,
                "original_run": original_run,
                "selected": selected,
            })
    write_csv(QC / "noise_source_counts_by_run.csv", rows)


def wepl_noise_metrics(root: Path, runs: int) -> dict[str, float | int]:
    residuals = []
    for run_id in range(runs):
        with np.load(root / "metadata" / f"meta{run_id:04d}.npz", allow_pickle=False) as meta:
            residuals.append(np.asarray(meta["measured_wepl_mm"] - meta["ideal_wepl_mm"], np.float32))
    values = np.concatenate(residuals).astype(np.float64)
    return {
        "wepl_count": len(values), "wepl_noise_rmse_mm": float(np.sqrt(np.mean(values**2))),
        "wepl_noise_mae_mm": float(np.mean(np.abs(values))), "wepl_noise_bias_mm": float(np.mean(values)),
    }


def run_noise_source_images(
    config: dict[str, Any], fraction: float, winner_name: str,
    winner_settings: dict[str, Any], jobs: int,
    device: int, force: bool,
) -> list[dict[str, Any]]:
    acquisition = optimization_acquisition(config)
    main, validation, test = (int(config[key]) for key in ("transition_seed", "validation_seed", "locked_test_seed"))
    requested = [(name, main) for name in config["noise_conditions"]]
    requested += [(name, seed) for seed in (validation, test) for name in ("position_0p2mm", "combined_0p2mm_1pct")]
    rows: list[dict[str, Any]] = []
    for index, (condition, seed) in enumerate(requested, 1):
        source = noise_source_root(config, condition, seed, fraction)
        root = resolve(config["reconstruction_output"]) / "noise_sources" / condition / f"seed_{seed}" / fraction_tag(fraction)
        label = f"{condition}_seed_{seed}_{fraction_tag(fraction)}"
        update_progress(status="RUNNING", batch="noise-weighting", stage="noise-source", group=label, completed_groups=index - 1, total_groups=len(requested))
        initial = ensure_ddb_and_initial(
            config, source, root, jobs, force, label, acquisition,
        )
        candidate_root, actual = run_candidate(
            config, source, initial, root / "winner", winner_name,
            winner_settings, device, force, acquisition,
        )
        best = choose_candidate(candidate_epoch_rows(winner_name, candidate_root, actual, full_baseline_metrics(config), config))
        # ``best`` contains the reconstruction RNG seed from the frozen
        # settings.  Keep it separate from the acquisition/subset seed used
        # to pair the three noise realizations; otherwise the later gate sees
        # every row as seed 20260713 and cannot form matched comparisons.
        row = dict(best)
        row["algorithm_seed"] = row.pop("seed", None)
        row.update({
            "condition": condition,
            "seed": seed,
            "acquisition_seed": seed,
            "fraction": fraction,
            "angles": int(acquisition["runs"]),
            **wepl_noise_metrics(source, int(acquisition["runs"])),
        })
        rows.append(row)
    return rows


def energy_gate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    main = int(config["transition_seed"])
    by_key = {
        (
            row["condition"],
            int(row.get("acquisition_seed", row["seed"])),
        ): row
        for row in rows
    }
    position = by_key[("position_0p2mm", main)]
    combined = by_key[("combined_0p2mm_1pct", main)]
    continuous = by_key[("continuous_hits", main)]
    energy = by_key[("energy_1pct_only", main)]
    paired = []
    for seed in (int(config["transition_seed"]), int(config["validation_seed"]), int(config["locked_test_seed"])):
        p = by_key[("position_0p2mm", seed)]
        c = by_key[("combined_0p2mm_1pct", seed)]
        paired.append(float(c["phantom_rmse_vs_rsp_truth"]) / float(p["phantom_rmse_vs_rsp_truth"]) - 1.0)
    combined_trigger = all(value > 0 for value in paired) and float(np.mean(paired)) >= float(config["energy_gate"]["combined_vs_position_rmse_fraction"])
    energy_image = max(
        float(energy["water_std"]) / float(continuous["water_std"]) - 1.0,
        float(energy["phantom_rmse_vs_rsp_truth"]) / float(continuous["phantom_rmse_vs_rsp_truth"]) - 1.0,
    )
    # Position perturbations alter the estimated path but not the measured
    # entrance/exit energy, so their *added measurement WEPL noise* is exactly
    # zero.  Dividing the combined value by that zero baseline would create a
    # meaningless enormous ratio and force the gate to trigger.  In this case
    # the WEPL-ratio criterion is unavailable; image degradation and the
    # three paired position-vs-combined comparisons remain valid gates.
    position_wepl = float(position["wepl_noise_rmse_mm"])
    if position_wepl > 1.0e-8:
        wepl_increase: float | None = (
            float(combined["wepl_noise_rmse_mm"]) / position_wepl - 1.0
        )
        wepl_trigger = (
            wepl_increase
            >= float(config["energy_gate"]["validation_wepl_fraction"])
        )
    else:
        wepl_increase = None
        wepl_trigger = False
    triggered = bool(
        combined_trigger
        or energy_image >= float(config["energy_gate"]["energy_only_image_fraction"])
        or wepl_trigger
    )
    return {
        "triggered": triggered, "paired_combined_vs_position_rmse_fraction": paired,
        "mean_combined_vs_position_rmse_fraction": float(np.mean(paired)),
        "energy_only_image_degradation_fraction": energy_image,
        "wepl_noise_increase_fraction": wepl_increase,
        "reasons": {
            "paired_rmse": combined_trigger,
            "energy_only_image": energy_image >= float(config["energy_gate"]["energy_only_image_fraction"]),
            "wepl": wepl_trigger,
        },
        "wepl_ratio_available": wepl_increase is not None,
    }


def patched_energy_weights(config: dict[str, Any], gamma: float):
    import stage7b_gpu

    def calculate(kind, energy, empirical_model, range_energy, range_mm, _local_config):
        empirical = stage7b_gpu.predict_empirical(empirical_model, energy).astype(np.float64)
        detector = stage7b_gpu.predict_analytic(energy, range_energy, range_mm, 0.01, 0.02).astype(np.float64)
        total = np.sqrt(empirical**2 + detector**2)
        raw = np.power(1.0 / np.maximum(total**2, 1.0e-12), gamma)
        finite = np.isfinite(raw) & (raw > 0)
        raw /= np.median(raw[finite])
        low, high = (float(value) for value in config["energy_weights"]["clip"])
        weight = np.clip(raw, low, high).astype(np.float32)
        ess = float(np.square(weight.sum()) / np.square(weight).sum() / len(weight))
        if ess < float(config["energy_weights"]["minimum_effective_fraction"]):
            raise RuntimeError(f"energy-weight ESS collapsed to {ess:.3f}")
        return weight, total.astype(np.float32), ess

    return calculate


def run_weighted_candidate(
    config: dict[str, Any], source: Path, initial: Path, output: Path, name: str,
    gamma: float, settings: dict[str, Any], device: int, force: bool,
) -> dict[str, Any]:
    import stage7b_gpu

    original = stage7b_gpu.per_angle_weights
    stage7b_gpu.per_angle_weights = patched_energy_weights(config, gamma)
    try:
        local = dict(config)
        acquisition = optimization_acquisition(config)
        local["angle_step_deg"] = float(acquisition["angle_step_deg"])
        local["_noise_model_path"] = config["_noise_model"]
        local["reconstruction"] = settings
        local["weights"] = {
            "clip": config["energy_weights"]["clip"],
            "minimum_effective_fraction": config["energy_weights"]["minimum_effective_fraction"],
        }
        summary = stage7b_gpu.run_reconstruction(
            name, {"weight": name, "huber_z": None}, source, source, initial,
            output, Path(config["_wepl_model"]), local, int(settings["epochs"]),
            device, int(acquisition["runs"]), update_progress, force,
        )
    finally:
        stage7b_gpu.per_angle_weights = original
    metrics = numeric_metrics(Path(summary["best_image"]))
    row = {**summary, **metrics}
    # Keep the Stage 8B weighting label authoritative even if the generic
    # reconstruction summary contains fields with the same names.
    row.update({"candidate": name, "gamma": gamma})
    return row


def noise_weighting(config: dict[str, Any], raw_root: Path, jobs: int, device: int, force: bool) -> dict[str, Any]:
    decision_path = QC / "optimization_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("run --action optimize first")
    optimization = read_json(decision_path)
    if optimization.get("status") != "PASS":
        raise RuntimeError(f"optimization is not frozen: {optimization}")
    fraction = float(optimization["f_dev"])
    settings = dict(optimization["winner_settings"])
    settings["epochs"] = int(optimization["best_epoch"])
    winner_name = str(optimization["winner"])
    update_progress(status="RUNNING", batch="noise-weighting", stage="prepare")
    prepare_noise_sources(config, raw_root, fraction, jobs, force)
    rows = run_noise_source_images(config, fraction, winner_name, settings, jobs, device, force)
    write_csv(QC / "noise_source_metrics.csv", rows)
    gate = energy_gate(rows, config)
    atomic_json(QC / "energy_gate.json", gate)
    weighted_rows: list[dict[str, Any]] = []
    winner = "equal"
    if gate["triggered"]:
        main = int(config["transition_seed"])
        source = noise_source_root(config, "combined_0p2mm_1pct", main, fraction)
        root = resolve(config["reconstruction_output"]) / "energy_weights/development" / fraction_tag(fraction)
        noise_recon_root = resolve(config["reconstruction_output"]) / "noise_sources/combined_0p2mm_1pct" / f"seed_{main}" / fraction_tag(fraction)
        initial = noise_recon_root / "analytic/recon_ddb_nohann.mhd"
        if winner_name == "multiscale_0p5_to_0p1":
            initial = noise_recon_root / "winner/multiscale_0p5_to_0p1/fine_initial.mhd"
        elif winner_name == "lowpass_0p5_upsampled":
            initial = noise_recon_root / "winner/lowpass_0p5_upsampled/lowpass_initial.mhd"
        equal_source = next(
            row for row in rows
            if row["condition"] == "combined_0p2mm_1pct"
            and int(row["seed"]) == main
        )
        equal_row = dict(equal_source)
        equal_row.update({"candidate": "equal", "gamma": 0.0})
        weighted_rows.append(equal_row)
        for gamma in (float(value) for value in config["energy_weights"]["gammas"]):
            name = f"energy_gamma_{gamma:g}".replace(".", "p")
            try:
                weighted_rows.append(
                    run_weighted_candidate(
                        config, source, initial, root / name, name, gamma,
                        settings, device, force,
                    )
                )
            except RuntimeError as error:
                # The ESS floor is a scientific rejection criterion for one
                # weighting candidate, not a fatal error for the entire stage.
                # Unexpected runtime failures must still stop immediately.
                if "energy-weight ESS collapsed" not in str(error):
                    raise
                weighted_rows.append({
                    "candidate": name,
                    "gamma": gamma,
                    "status": "REJECTED",
                    "rejection_reason": str(error),
                })
                update_progress(
                    status="RUNNING",
                    batch="noise-weighting",
                    stage="candidate-rejected",
                    group=name,
                    rejection_reason=str(error),
                )
        eligible_rows = [
            row for row in weighted_rows
            if row.get("status") != "REJECTED"
            and "phantom_rmse_vs_rsp_truth" in row
        ]
        best = min(
            eligible_rows,
            key=lambda row: float(row["phantom_rmse_vs_rsp_truth"]),
        )
        winner = str(best["candidate"])
        if winner != "equal":
            gamma = float(best["gamma"])
            for seed, role in ((int(config["validation_seed"]), "validation"), (int(config["locked_test_seed"]), "test")):
                source = noise_source_root(config, "combined_0p2mm_1pct", seed, fraction)
                result_root = resolve(config["reconstruction_output"]) / "energy_weights" / role / f"seed_{seed}" / fraction_tag(fraction)
                noise_recon_root = resolve(config["reconstruction_output"]) / "noise_sources/combined_0p2mm_1pct" / f"seed_{seed}" / fraction_tag(fraction)
                initial = noise_recon_root / "analytic/recon_ddb_nohann.mhd"
                if winner_name == "multiscale_0p5_to_0p1":
                    initial = noise_recon_root / "winner/multiscale_0p5_to_0p1/fine_initial.mhd"
                elif winner_name == "lowpass_0p5_upsampled":
                    initial = noise_recon_root / "winner/lowpass_0p5_upsampled/lowpass_initial.mhd"
                weighted_rows.append(run_weighted_candidate(config, source, initial, result_root / winner, winner, gamma, settings, device, force))
    write_csv(QC / "energy_weight_candidates.csv", weighted_rows)
    promoted_weight = False
    if winner != "equal":
        confirmation = [row for row in weighted_rows if row["candidate"] == winner]
        equal_by_seed = {
            int(row["seed"]): row for row in rows
            if row["condition"] == "combined_0p2mm_1pct"
        }
        comparisons = []
        for row in confirmation:
            seed = int(row.get("seed", config["transition_seed"]))
            # Weighted summaries do not carry the acquisition seed; recover it
            # from the output path for confirmation runs.
            text = str(row.get("best_image", ""))
            for candidate_seed in equal_by_seed:
                if f"seed_{candidate_seed}" in text:
                    seed = candidate_seed
            equal = equal_by_seed.get(seed)
            if equal is None:
                continue
            improvement = 1.0 - float(row["phantom_rmse_vs_rsp_truth"]) / float(equal["phantom_rmse_vs_rsp_truth"])
            al_degradation = (abs(aluminium_error(row)) - abs(aluminium_error(equal))) * 100.0
            edge_ratio = float(row["aluminium_edge_10_90_median_mm"]) / float(equal["aluminium_edge_10_90_median_mm"])
            comparisons.append({"seed": seed, "rmse_improvement_fraction": improvement, "aluminium_degradation_pp": al_degradation, "edge_ratio": edge_ratio})
        write_csv(QC / "energy_weight_validation.csv", comparisons)
        promoted_weight = bool(
            len(comparisons) == 3
            and all(row["rmse_improvement_fraction"] >= 0 for row in comparisons)
            and np.mean([row["rmse_improvement_fraction"] for row in comparisons]) >= 0.10
            and all(row["aluminium_degradation_pp"] <= 0.5 and row["edge_ratio"] <= 1.10 for row in comparisons)
        )
    else:
        write_csv(QC / "energy_weight_validation.csv", [])
    decision = {
        "status": "PASS",
        "acquisition": optimization_acquisition(config),
        "f_dev": fraction, "optimization": optimization,
        "energy_gate": gate, "energy_weight_winner": winner,
        "energy_weight_promoted": promoted_weight,
        "final_method": winner if promoted_weight else "equal_quadratic",
    }
    atomic_json(QC / "stage8b_decision.json", decision)
    build_noise_figure(rows)
    if weighted_rows:
        build_weight_figure(weighted_rows)
    lines = [
        "# Stage 8B：低通量适配", "",
        f"最终状态：**PASS**。开发条件为**360角度 × 每角度{100*fraction:g}%通量**。", "",
        f"低通量优化候选晋升：**{'是' if optimization.get('promoted') else '否'}**。",
        f"能量加权门控：**{'触发' if gate['triggered'] else '未触发'}**。",
        f"最终数据项：`{decision['final_method']}`。",
    ]
    (QC / "stage8b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_progress(status="COMPLETE", batch="noise-weighting", stage="report", task_eta_seconds=0)
    return decision


def build_noise_figure(rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage8b")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = min(int(row["seed"]) for row in rows)
    selected = [row for row in rows if int(row["seed"]) == main]
    labels = [str(row["condition"]) for row in selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, [float(row["wepl_noise_rmse_mm"]) for row in selected])
    axes[0].set_ylabel("Added WEPL noise RMSE (mm)")
    axes[1].bar(labels, [float(row["phantom_rmse_vs_rsp_truth"]) for row in selected])
    axes[1].set_ylabel("Phantom RSP RMSE")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / "noise_source_separation.png", dpi=180)
    plt.close(fig)


def build_weight_figure(rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage8b")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    development = []
    for row in rows:
        if "phantom_rmse_vs_rsp_truth" not in row:
            continue
        name = str(row["candidate"])
        if name not in [str(value["candidate"]) for value in development]:
            development.append(row)
    fig, axis = plt.subplots(figsize=(7.4, 4.3))
    axis.bar(
        [str(row["candidate"]) for row in development],
        [float(row["phantom_rmse_vs_rsp_truth"]) for row in development],
    )
    axis.set_ylabel("Phantom RSP RMSE")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / "energy_weight_comparison.png", dpi=180)
    plt.close(fig)


def status() -> None:
    if not PROGRESS.is_file():
        print("Stage 8B has not started.")
        return
    value = read_json(PROGRESS)
    print("Stage 8B low-fluence adaptation")
    print(f"  status: {value.get('status', '-')}")
    print(f"  batch/stage: {value.get('batch', '-')}/{value.get('stage', '-')}")
    print(f"  group: {value.get('group', '-')}")
    if "completed_runs" in value:
        print(f"  work units: {value.get('completed_runs', 0)}/{value.get('total_runs', '?')}")
    if "completed_groups" in value:
        print(f"  groups: {value.get('completed_groups', 0)}/{value.get('total_groups', '?')}")
    if "epoch" in value:
        print(
            f"  epoch/subset: {value.get('epoch', '-')}/{value.get('total_epochs', '-')} "
            f"{value.get('subset', '-')}/{value.get('total_subsets', '-')}"
        )
    if value.get("pairs_per_second"):
        print(f"  rate: {float(value['pairs_per_second']):,.0f} pairs/s")
    eta = float(value.get("task_eta_seconds", 0.0))
    if eta > 0:
        print(f"  current task ETA: {eta/3600:.2f} h")
    for filename, label in (
        ("transition_decision.json", "transition"),
        ("optimization_decision.json", "optimization"),
        ("stage8b_decision.json", "final"),
    ):
        path = QC / filename
        if path.is_file():
            decision = read_json(path)
            print(f"  {label}: {decision.get('status', '-')} f_dev={decision.get('f_dev', '-')}")
    print(f"  updated: {value.get('updated_at', '-')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("transition", "optimize", "noise-weighting", "status"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "status":
        status()
        return
    config = load_config()
    raw_root = args.raw_root or Path(config["raw_root_default"])
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    preflight(config, raw_root, args.device, args.action, args.force)
    if args.action == "transition":
        transition(config, args.jobs, args.device, args.force)
    elif args.action == "optimize":
        optimize(config, raw_root, args.jobs, args.device, args.force)
    elif args.action == "noise-weighting":
        noise_weighting(config, raw_root, args.jobs, args.device, args.force)


if __name__ == "__main__":
    main()
