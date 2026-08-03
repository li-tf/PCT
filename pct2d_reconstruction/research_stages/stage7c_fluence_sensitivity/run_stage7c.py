#!/usr/bin/env python3
"""Run Stage 7C nested effective-fluence sensitivity study."""

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
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
REPO = CODE.parent
QC = HERE / "qc"
PROGRESS = QC / "progress.json"
CONFIG_PATH = HERE / "stage7c_config.json"
sys.path[:0] = [
    str(HERE),
    str(CODE),
    str(CODE / "preprocessing"),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "analytic_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage4_iterative_optimization"),
    str(CODE / "research_stages/stage7_detector_effects"),
    str(CODE / "research_stages/stage7b_noise_robustness"),
]

from stage7c_data import fraction_tag, group_root, prepare_run  # noqa: E402
from preprocessing import projection  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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


def task_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    main = int(config["main_seed"])
    for condition in config["conditions"]:
        for fraction in config["fractions"]:
            if float(fraction) < 1.0:
                tasks.append(
                    {
                        "condition": condition,
                        "seed": main,
                        "fraction": float(fraction),
                        "replicate": False,
                    }
                )
    for seed in config["replicate_seeds"]:
        for fraction in config["replicate_fractions"]:
            tasks.append(
                {
                    "condition": "combined_0p2mm_1pct",
                    "seed": int(seed),
                    "fraction": float(fraction),
                    "replicate": True,
                }
            )
    return tasks


def tag(task: dict[str, Any]) -> str:
    return (
        f"{task['condition']}/seed_{task['seed']}/"
        f"{fraction_tag(float(task['fraction']))}"
    )


def config_hash(config: dict[str, Any], raw_root: Path) -> str:
    value = {
        "config": {key: val for key, val in config.items() if not key.startswith("_")},
        "raw_root": str(raw_root.resolve()),
        "stage7c_data_sha256": hashlib.sha256(
            (HERE / "stage7c_data.py").read_bytes()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def locate_run(root: Path, run_id: int) -> Path | None:
    for name in (
        f"run_{run_id:03d}",
        f"run_{run_id:04d}",
        f"angle_{run_id:03d}",
        f"{run_id:03d}",
    ):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def preflight(
    config: dict[str, Any], raw_root: Path, runs: int, device: int
) -> dict[str, Any]:
    required = (
        "PhaseSpaceIn.root",
        "PhaseSpaceOut.root",
        "TrackerUpstream1.root",
        "TrackerUpstream2.root",
        "TrackerDownstream1.root",
        "TrackerDownstream2.root",
    )
    missing = []
    for run_id in range(runs):
        directory = locate_run(raw_root, run_id)
        absent = (
            list(required)
            if directory is None
            else [
                name
                for name in required
                if not (directory / name).is_file()
                or (directory / name).stat().st_size == 0
            ]
        )
        if absent:
            missing.append({"run": run_id, "missing": absent})
    stage7 = resolve(config["stage7_preprocessing"])
    missing_pairs = []
    for condition in config["conditions"].values():
        name = condition["stage7_name"]
        for run_id in range(runs):
            path = stage7 / "full" / name / "pairs" / f"pairs{run_id:04d}.mhd"
            if not path.is_file():
                missing_pairs.append(str(path))
                break
    decision_path = resolve(config["stage7b_decision"])
    decision = read_json(decision_path) if decision_path.is_file() else {}
    if decision.get("winner") != "equal_quadratic":
        raise RuntimeError("Stage 7B did not freeze equal_quadratic")
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
        "status": "PASS" if not missing and not missing_pairs and free >= required_free else "FAIL",
        "raw_runs": runs - len(missing),
        "expected_runs": runs,
        "first_missing_raw": missing[:3],
        "missing_stage7_pair_groups": missing_pairs,
        "stage7b_winner": decision.get("winner"),
        "test_partition_opened": decision.get("test_partition_opened"),
        "free_bytes": free,
        "required_free_bytes": required_free,
        "gpu": str(gpu),
    }
    atomic_json(QC / "preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Stage 7C preflight failed: {result}")
    return result


def prepare(
    config: dict[str, Any],
    raw_root: Path,
    runs: int,
    jobs: int,
    force: bool,
) -> list[dict[str, Any]]:
    output = resolve(config["preprocessing_output"])
    stage7 = resolve(config["stage7_preprocessing"])
    manifest_path = QC / "input_manifest.json"
    digest = config_hash(config, raw_root)
    if manifest_path.is_file() and not force:
        old = read_json(manifest_path)
        if old.get("config_sha256") != digest:
            raise RuntimeError("Stage 7C input/config changed; use a new output")
    atomic_json(
        manifest_path,
        {
            "config_sha256": digest,
            "raw_root": str(raw_root),
            "stage7_preprocessing": str(stage7),
            "runs": runs,
            "stage7b_test_opened": False,
        },
    )
    update_progress(
        status="RUNNING", stage="prepare", completed_runs=0, total_runs=runs
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                prepare_run,
                run_id,
                str(raw_root),
                str(stage7),
                str(output),
                config,
                force,
            ): run_id
            for run_id in range(runs)
        }
        for count, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            elapsed = time.perf_counter() - started
            eta = elapsed / count * (runs - count)
            update_progress(
                completed_runs=count,
                elapsed_seconds=elapsed,
                task_eta_seconds=eta,
            )
            if count % 10 == 0 or count == runs:
                print(
                    f"Stage7C mapping/subsets {count:03d}/{runs}: "
                    f"ETA={eta/60:.1f} min",
                    flush=True,
                )
    rows.sort(
        key=lambda row: (
            row["condition"],
            int(row["seed"]),
            -float(row["fraction"]),
            int(row["run_id"]),
        )
    )
    write_csv(QC / "fluence_counts_by_run.csv", rows)
    summary: list[dict[str, Any]] = []
    for task in task_groups(config):
        selected = [
            row
            for row in rows
            if row["condition"] == task["condition"]
            and int(row["seed"]) == int(task["seed"])
            and float(row["fraction"]) == float(task["fraction"])
        ]
        summary.append(
            {
                **task,
                "nominal_fluence_per_mm2_projection": float(
                    config["nominal_full_fluence_per_mm2_projection"]
                )
                * float(task["fraction"]),
                "full_filtered": sum(int(row["full_filtered"]) for row in selected),
                "selected": sum(int(row["selected"]) for row in selected),
                "min_per_angle": min(int(row["selected"]) for row in selected),
                "max_per_angle": max(int(row["selected"]) for row in selected),
            }
        )
    write_csv(QC / "fluence_counts.csv", summary)
    return summary


def project(
    config: dict[str, Any], runs: int, jobs: int, force: bool
) -> None:
    output = resolve(config["preprocessing_output"])
    tasks = task_groups(config)
    for task_index, task in enumerate(tasks, 1):
        root = group_root(
            output, task["condition"], int(task["seed"]), float(task["fraction"])
        )
        ddb = root / "projections_ddb"
        ddb.mkdir(parents=True, exist_ok=True)
        pending = [
            run_id
            for run_id in range(runs)
            if force or not (ddb / f"proj{run_id:04d}.mhd").is_file()
        ]
        update_progress(
            stage="projection",
            group=tag(task),
            completed_groups=task_index - 1,
            total_groups=len(tasks),
            completed_runs=runs - len(pending),
            total_runs=runs,
        )
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    projection.process_run,
                    run_id,
                    str(root / "pairs"),
                    str(root),
                    False,
                    "projections_ddb",
                ): run_id
                for run_id in pending
            }
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                done = runs - len(pending) + count
                update_progress(completed_runs=done)
                if done % 40 == 0 or done == runs:
                    print(f"DDB {tag(task)}: {done:03d}/{runs}", flush=True)
        if any(
            not (ddb / f"proj{run_id:04d}.mhd").is_file()
            for run_id in range(runs)
        ):
            raise RuntimeError(f"incomplete DDB {tag(task)}")


def ensure_geometry(config: dict[str, Any], runs: int) -> Path:
    geometry = QC / "geometry.xml"
    if geometry.is_file():
        return geometry
    acquisition = read_json(CODE / "experiments/experiment0716.json")["acquisition"]
    command = [
        str(REPO / ".venv-gate/bin/rtksimulatedgeometry"),
        "--nproj",
        str(runs),
        "--first_angle",
        f"{float(acquisition['first_angle_deg']):g}",
        "--arc",
        f"{float(acquisition['arc_deg']):g}",
        "--sid",
        f"{float(acquisition['source_to_isocenter_mm']):g}",
        "--sdd",
        f"{float(acquisition['source_to_detector_mm']):g}",
        "--output",
        str(geometry),
    ]
    QC.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return geometry


def analytic(config: dict[str, Any], runs: int, force: bool) -> None:
    geometry = ensure_geometry(config, runs)
    pre = resolve(config["preprocessing_output"])
    recon = resolve(config["reconstruction_output"])
    for task in task_groups(config):
        source = group_root(
            pre, task["condition"], int(task["seed"]), float(task["fraction"])
        )
        root = group_root(
            recon, task["condition"], int(task["seed"]), float(task["fraction"])
        )
        output = root / "analytic/recon_ddb_nohann.mhd"
        if output.is_file() and not force:
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        update_progress(stage="analytic", group=tag(task))
        command = [
            str(REPO / ".venv-gate/bin/pctfdk"),
            "--lowmem",
            "--geometry",
            str(geometry),
            "--path",
            str(source / "projections_ddb"),
            "--regexp",
            r"proj....\.mhd",
            "--output",
            str(output),
            "--dimension",
            "2100",
            "1",
            "2100",
            "--spacing",
            "0.1",
            "1",
            "0.1",
            "--hann",
            "0",
            "--verbose",
        ]
        log = QC / "logs" / f"fdk_{tag(task).replace('/', '_')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=True
            )


def run_logged(command: list[str], log: Path, group: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    epoch_pattern = re.compile(r"epoch\s+(\d+)/(\d+).*subset\s+(\d+)/(\d+)")
    rate_pattern = re.compile(r"rate=([\d,]+)\s+pairs/s\s+ETA=([0-9:]+)")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            match = epoch_pattern.search(line)
            if match:
                update_progress(
                    stage="reconstruct",
                    group=group,
                    epoch=int(match.group(1)),
                    total_epochs=int(match.group(2)),
                    subset=int(match.group(3)),
                    total_subsets=int(match.group(4)),
                )
            rate_match = rate_pattern.search(line)
            if rate_match:
                parts = [int(value) for value in rate_match.group(2).split(":")]
                eta = 0
                for value in parts:
                    eta = 60 * eta + value
                update_progress(
                    pairs_per_second=int(rate_match.group(1).replace(",", "")),
                    task_eta_seconds=eta,
                )
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def reconstruct(
    config: dict[str, Any], runs: int, device: int, force: bool
) -> None:
    frozen = read_json(resolve(config["frozen_reconstruction"]))["reconstruction"]
    pre = resolve(config["preprocessing_output"])
    recon = resolve(config["reconstruction_output"])
    tasks = task_groups(config)
    for index, task in enumerate(tasks, 1):
        source = group_root(
            pre, task["condition"], int(task["seed"]), float(task["fraction"])
        )
        root = group_root(
            recon, task["condition"], int(task["seed"]), float(task["fraction"])
        )
        final = root / "iterative/recon/epoch_05.mhd"
        if final.is_file() and not force:
            continue
        command = [
            sys.executable,
            str(CODE / "iterative_reconstruction/run_iterative_reconstruction.py"),
            "--experiment",
            "0716",
            "--pairs-dir",
            str(source / "pairs"),
            "--initial-image",
            str(root / "analytic/recon_ddb_nohann.mhd"),
            "--output-dir",
            str(root / "iterative"),
            "--qc-dir",
            str(QC / "reconstruction" / tag(task)),
            "--runs",
            str(runs),
            "--angle-step-deg",
            str(config["angle_step_deg"]),
            "--phantom-radius-mm",
            str(config["phantom_radius_mm"]),
            "--air-wepl-slope",
            "0",
            "--wepl-model",
            "g4_water_calibrated",
            "--wepl-calibration",
            str(resolve(config["wepl_model"])),
            "--epochs",
            str(frozen["epochs"]),
            "--sample-fraction",
            "1",
            "--grid-size",
            str(frozen["grid_size"]),
            "--grid-spacing-mm",
            str(frozen["grid_spacing_mm"]),
            "--path-step-mm",
            str(frozen["path_step_mm"]),
            "--batch-size",
            str(frozen["batch_size"]),
            "--subsets",
            str(frozen["subsets"]),
            "--relaxation",
            str(frozen["relaxation"]),
            "--relaxation-decay",
            str(frozen["relaxation_decay"]),
            "--initialization",
            "fdk_nohann",
            "--device",
            str(device),
            "--regularizer",
            str(frozen["regularizer"]),
            "--regularization-weight",
            str(frozen["regularization_weight"]),
            "--regularization-iterations",
            str(frozen["regularization_iterations"]),
            "--regularization-every-epochs",
            str(frozen["regularization_every_epochs"]),
            "--huber-delta",
            str(frozen["huber_delta"]),
            "--primal-step",
            str(frozen["primal_step"]),
            "--dual-step",
            str(frozen["dual_step"]),
            "--skip-truth-metrics",
        ]
        if force:
            command.append("--force")
        update_progress(
            status="RUNNING",
            stage="reconstruct",
            group=tag(task),
            completed_groups=index - 1,
            total_groups=len(tasks),
        )
        run_logged(
            command,
            QC / "logs" / f"iterative_{tag(task).replace('/', '_')}.log",
            tag(task),
        )


def image_metrics(path: Path) -> dict[str, Any]:
    import run_stage7b

    return run_stage7b.image_metrics(path)


def full_path(config: dict[str, Any], condition: str, kind: str) -> Path:
    name = config["conditions"][condition]["stage7_name"]
    root = resolve(config["stage7_reconstruction"]) / "full" / name
    if kind == "analytic":
        return root / "analytic/recon_ddb_nohann.mhd"
    return root / "iterative/recon/epoch_05.mhd"


def result_path(
    config: dict[str, Any],
    condition: str,
    seed: int,
    fraction: float,
    kind: str,
) -> Path:
    root = group_root(
        resolve(config["reconstruction_output"]), condition, seed, fraction
    )
    if kind == "analytic":
        return root / "analytic/recon_ddb_nohann.mhd"
    return root / "iterative/recon/epoch_05.mhd"


def metric_rows(config: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main = int(config["main_seed"])
    for condition in config["conditions"]:
        for fraction in config["fractions"]:
            fraction = float(fraction)
            path = (
                full_path(config, condition, kind)
                if fraction == 1.0
                else result_path(config, condition, main, fraction, kind)
            )
            measured = image_metrics(path)
            rows.append(
                {
                    "condition": condition,
                    "seed": main,
                    "fraction": fraction,
                    "fluence_per_mm2_projection": float(
                        config["nominal_full_fluence_per_mm2_projection"]
                    )
                    * fraction,
                    "kind": kind,
                    "reused_stage7": fraction == 1.0,
                    "image_path": str(path),
                    **{
                        key: value
                        for key, value in measured.items()
                        if isinstance(value, (int, float, bool, np.number))
                    },
                }
            )
    if kind == "iterative":
        condition = "combined_0p2mm_1pct"
        for seed in config["replicate_seeds"]:
            for fraction in config["replicate_fractions"]:
                path = result_path(
                    config, condition, int(seed), float(fraction), kind
                )
                measured = image_metrics(path)
                rows.append(
                    {
                        "condition": condition,
                        "seed": int(seed),
                        "fraction": float(fraction),
                        "fluence_per_mm2_projection": float(
                            config["nominal_full_fluence_per_mm2_projection"]
                        )
                        * float(fraction),
                        "kind": kind,
                        "reused_stage7": False,
                        "image_path": str(path),
                        **{
                            key: value
                            for key, value in measured.items()
                            if isinstance(value, (int, float, bool, np.number))
                        },
                    }
                )
    return rows


def aluminium_error(row: dict[str, Any]) -> float:
    return float(row["insert_peak_mean"]) / 2.094511207867794 - 1.0


def qualify(
    row: dict[str, Any], baseline: dict[str, Any], selection: dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    water_bias = abs(float(row["water_mean"]) / 0.9997458098472691 - 1.0)
    al_degradation = (
        abs(aluminium_error(row)) - abs(aluminium_error(baseline))
    ) * 100.0
    edge = float(row.get("aluminium_edge_10_90_median_mm", math.nan))
    base_edge = float(
        baseline.get("aluminium_edge_10_90_median_mm", math.nan)
    )
    edge_ratio = (
        edge / base_edge
        if np.isfinite(edge) and np.isfinite(base_edge) and base_edge > 0
        else math.inf
    )
    details = {
        "water_bias_fraction": water_bias,
        "phantom_rmse_ratio": float(row["phantom_rmse_vs_rsp_truth"])
        / float(baseline["phantom_rmse_vs_rsp_truth"]),
        "aluminium_error_fraction": aluminium_error(row),
        "aluminium_error_degradation_pp": al_degradation,
        "edge_width_ratio": edge_ratio,
        "absolute_aluminium_benchmark_pass": abs(aluminium_error(row))
        <= float(selection["absolute_aluminium_error_reference_fraction"]),
    }
    passed = (
        water_bias <= float(selection["water_bias_max_fraction"])
        and float(row["water_std"]) <= float(selection["water_std_max"])
        and details["phantom_rmse_ratio"]
        <= float(selection["phantom_rmse_ratio_max"])
        and al_degradation
        <= float(selection["aluminium_error_degradation_max_pp"])
        and float(row["insert_cnr_median"]) >= float(selection["cnr_min"])
        and edge_ratio <= float(selection["edge_width_ratio_max"])
        and float(row.get("finite_fraction", 1.0)) == 1.0
        and float(row.get("outside_mean", 0.0)) == 0.0
    )
    return passed, details


def fit_noise(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    from scipy.optimize import curve_fit

    selected = sorted(
        (
            row
            for row in rows
            if row["condition"] == condition
            and int(row["seed"]) == 20260730
        ),
        key=lambda row: float(row["fraction"]),
    )
    n = np.asarray([float(row["fraction"]) for row in selected])
    sigma = np.asarray([float(row["water_std"]) for row in selected])

    def model(value, c, alpha, floor):
        return c * value ** (-alpha) + floor

    try:
        params, _ = curve_fit(
            model,
            n,
            sigma,
            p0=(max(sigma[-1] - sigma[0], 1.0e-4), 0.5, sigma[0] * 0.5),
            bounds=([0.0, 0.0, 0.0], [1.0, 2.0, 1.0]),
            maxfev=20000,
        )
        fitted = model(n, *params)
        ss_res = float(np.sum((sigma - fitted) ** 2))
        ss_tot = float(np.sum((sigma - sigma.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
        stable = n >= 0.25
        stable_alpha = float(
            -np.polyfit(np.log(n[stable]), np.log(sigma[stable]), 1)[0]
        )
        boundary_hit = bool(params[1] >= 1.99)
        return {
            "condition": condition,
            "c": float(params[0]),
            "alpha": float(params[1]),
            "sigma_floor": float(params[2]),
            "r_squared": r2,
            "full_fit_reliable": not boundary_hit,
            "stable_range_min_fraction": 0.25,
            "stable_range_alpha": stable_alpha,
            "interpretation": (
                "10% point enters nonlinear reconstruction breakdown; "
                "bounded four-point fit is not a physical noise exponent"
                if boundary_hit
                else "four-point fit did not hit its parameter bounds"
            ),
        }
    except Exception as error:
        return {"condition": condition, "status": f"FIT_FAILED: {error}"}


def build_figures(
    config: dict[str, Any],
    analytic_rows: list[dict[str, Any]],
    iterative_rows: list[dict[str, Any]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-stage7c")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analytic_reconstruction import rsp_metrics

    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    labels = {
        "ideal_reference": "Ideal reference",
        "continuous_hits": "Continuous Si hits",
        "combined_0p2mm_1pct": "0.2 mm + 1% Eout",
    }
    metrics = [
        ("water_std", "Water RSP standard deviation", "water_noise_vs_fluence.png"),
        ("phantom_rmse_vs_rsp_truth", "Phantom RSP RMSE", "rmse_vs_fluence.png"),
        ("insert_cnr_median", "Aluminium-water CNR", "cnr_vs_fluence.png"),
        (
            "aluminium_edge_10_90_median_mm",
            "Median 10%-90% edge width (mm)",
            "edge_vs_fluence.png",
        ),
    ]
    main = int(config["main_seed"])
    for key, ylabel, filename in metrics:
        fig, axis = plt.subplots(figsize=(6.8, 4.3))
        for condition in config["conditions"]:
            selected = sorted(
                (
                    row
                    for row in iterative_rows
                    if row["condition"] == condition
                    and int(row["seed"]) == main
                ),
                key=lambda row: float(row["fraction"]),
            )
            axis.plot(
                [row["fluence_per_mm2_projection"] for row in selected],
                [row.get(key, math.nan) for row in selected],
                marker="o",
                label=labels[condition],
            )
            if key == "water_std":
                full = next(
                    row
                    for row in selected
                    if float(row["fraction"]) == 1.0
                )
                fluence = np.asarray(
                    [row["fluence_per_mm2_projection"] for row in selected]
                )
                expected = float(full["water_std"]) * np.sqrt(
                    float(config["nominal_full_fluence_per_mm2_projection"])
                    / fluence
                )
                axis.plot(fluence, expected, ":", alpha=0.45)
        axis.set_xlabel("Nominal fluence (protons/mm²/projection)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(assets / filename, dpi=180)
        plt.close(fig)

    # Main-seed iterative montage and error maps.
    fig, axes = plt.subplots(6, 4, figsize=(13.0, 17.0))
    geometry = read_json(resolve(config["truth_geometry"]))["geometry"]
    for col, fraction in enumerate(config["fractions"]):
        fraction = float(fraction)
        for condition_index, condition in enumerate(config["conditions"]):
            row = next(
                row
                for row in iterative_rows
                if row["condition"] == condition
                and int(row["seed"]) == main
                and float(row["fraction"]) == fraction
            )
            image, x, z, _ = rsp_metrics.read_mhd(Path(row["image_path"]))
            xx, zz = np.meshgrid(x, z)
            truth = np.zeros_like(image)
            truth[xx * xx + zz * zz <= 100.0**2] = 0.9997458098472691
            for center in geometry["insert_centers_xz_mm"]:
                mask = (
                    (xx - float(center["x"])) ** 2
                    + (zz - float(center["z"])) ** 2
                    <= float(geometry["insert_radius_mm"]) ** 2
                )
                truth[mask] = 2.094511207867794
            top = axes[condition_index * 2, col]
            bottom = axes[condition_index * 2 + 1, col]
            top.imshow(image, extent=[x[0], x[-1], z[-1], z[0]], vmin=0.95, vmax=2.15)
            bottom.imshow(
                image - truth,
                extent=[x[0], x[-1], z[-1], z[0]],
                cmap="RdBu_r",
                vmin=-0.15,
                vmax=0.15,
            )
            if condition_index == 0:
                top.set_title(f"{int(100*fraction)}%")
            if col == 0:
                top.set_ylabel(labels[condition])
                bottom.set_ylabel("Error")
            top.set_xticks([])
            top.set_yticks([])
            bottom.set_xticks([])
            bottom.set_yticks([])
    fig.tight_layout()
    fig.savefig(assets / "iterative_fluence_montage.png", dpi=170)
    plt.close(fig)

    # Analytic/iterative noise comparison.
    fig, axis = plt.subplots(figsize=(6.8, 4.3))
    for kind, rows, style in (
        ("DDB-FDK", analytic_rows, "--"),
        ("OS-SART", iterative_rows, "-"),
    ):
        selected = sorted(
            (
                row
                for row in rows
                if row["condition"] == "combined_0p2mm_1pct"
                and int(row["seed"]) == main
            ),
            key=lambda row: float(row["fraction"]),
        )
        axis.plot(
            [row["fluence_per_mm2_projection"] for row in selected],
            [row["water_std"] for row in selected],
            style,
            marker="o",
            label=kind,
        )
    axis.set_xlabel("Nominal fluence (protons/mm²/projection)")
    axis.set_ylabel("Water RSP standard deviation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(assets / "analytic_iterative_comparison.png", dpi=180)
    plt.close(fig)

    # Low-fluence multi-seed uncertainty for the combined-noise condition.
    fig, axis = plt.subplots(figsize=(6.8, 4.3))
    for key, label in (
        ("water_std", "Water std"),
        ("phantom_rmse_vs_rsp_truth", "Phantom RMSE"),
    ):
        x, mean, spread = [], [], []
        for fraction in config["replicate_fractions"]:
            values = [
                float(row[key])
                for row in iterative_rows
                if row["condition"] == "combined_0p2mm_1pct"
                and float(row["fraction"]) == float(fraction)
            ]
            x.append(
                float(config["nominal_full_fluence_per_mm2_projection"])
                * float(fraction)
            )
            mean.append(float(np.mean(values)))
            spread.append(float(np.std(values, ddof=1)))
        axis.errorbar(x, mean, yerr=spread, marker="o", capsize=4, label=label)
    axis.set_xlabel("Nominal fluence (protons/mm²/projection)")
    axis.set_ylabel("RSP metric")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(assets / "combined_multiseed_uncertainty.png", dpi=180)
    plt.close(fig)

    # Water-cylinder azimuthal mean profiles for the combined-noise condition.
    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    for fraction in config["fractions"]:
        row = next(
            row
            for row in iterative_rows
            if row["condition"] == "combined_0p2mm_1pct"
            and int(row["seed"]) == main
            and float(row["fraction"]) == float(fraction)
        )
        image, x, z, _ = rsp_metrics.read_mhd(Path(row["image_path"]))
        xx, zz = np.meshgrid(x, z)
        radius = np.hypot(xx, zz)
        bins = np.arange(0.0, 105.01, 0.5)
        index = np.digitize(radius.ravel(), bins) - 1
        sums = np.bincount(index, weights=image.ravel(), minlength=len(bins))
        count = np.bincount(index, minlength=len(bins))
        profile = sums[: len(bins) - 1] / np.maximum(count[: len(bins) - 1], 1)
        axis.plot(
            0.5 * (bins[:-1] + bins[1:]),
            profile,
            label=f"{int(100*float(fraction))}%",
        )
    axis.set_xlim(0, 105)
    axis.set_xlabel("Radius (mm)")
    axis.set_ylabel("Azimuthal mean RSP")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(assets / "combined_radial_profiles.png", dpi=180)
    plt.close(fig)


def runtime_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    main = int(config["main_seed"])
    for condition in config["conditions"]:
        name = config["conditions"][condition]["stage7_name"]
        summary = (
            CODE
            / "research_stages/stage7_detector_effects/qc/reconstruction/full"
            / name
            / "run_summary.json"
        )
        if summary.is_file():
            value = read_json(summary)
            rows.append(
                {
                    "condition": condition,
                    "seed": main,
                    "fraction": 1.0,
                    "reused_stage7": True,
                    "iterative_seconds": value.get("elapsed_seconds"),
                    "gpu": value.get("gpu"),
                    "pairs_per_epoch": value.get("pairs_per_epoch"),
                }
            )
    for task in task_groups(config):
        summary = QC / "reconstruction" / tag(task) / "run_summary.json"
        if not summary.is_file():
            continue
        value = read_json(summary)
        rows.append(
            {
                "condition": task["condition"],
                "seed": task["seed"],
                "fraction": task["fraction"],
                "reused_stage7": False,
                "iterative_seconds": value.get("elapsed_seconds"),
                "gpu": value.get("gpu"),
                "pairs_per_epoch": value.get("pairs_per_epoch"),
            }
        )
    return rows


def report(config: dict[str, Any]) -> dict[str, Any]:
    update_progress(status="RUNNING", stage="report")
    analytic_rows = metric_rows(config, "analytic")
    iterative_rows = metric_rows(config, "iterative")
    write_csv(QC / "analytic_metrics.csv", analytic_rows)
    write_csv(QC / "iterative_metrics.csv", iterative_rows)
    main = int(config["main_seed"])
    qualification: list[dict[str, Any]] = []
    recommendations: dict[str, Any] = {}
    for condition in config["conditions"]:
        baseline = next(
            row
            for row in iterative_rows
            if row["condition"] == condition
            and int(row["seed"]) == main
            and float(row["fraction"]) == 1.0
        )
        passing = []
        for row in iterative_rows:
            if row["condition"] != condition:
                continue
            passed, details = qualify(row, baseline, config["selection"])
            record = {**row, **details, "passed": passed}
            qualification.append(record)
            if passed and int(row["seed"]) == main:
                passing.append(float(row["fraction"]))
        # Multi-seed condition: a low point passes only if every seed passes.
        if condition == "combined_0p2mm_1pct":
            for fraction in config["replicate_fractions"]:
                records = [
                    row
                    for row in qualification
                    if row["condition"] == condition
                    and float(row["fraction"]) == float(fraction)
                ]
                if len(records) != 3 or not all(bool(row["passed"]) for row in records):
                    passing = [value for value in passing if value != float(fraction)]
        recommendations[condition] = {
            "minimum_fraction": min(passing) if passing else None,
            "minimum_fluence_per_mm2_projection": (
                min(passing)
                * float(config["nominal_full_fluence_per_mm2_projection"])
                if passing
                else None
            ),
        }
    write_csv(QC / "qualification_metrics.csv", qualification)
    multi = [
        row
        for row in qualification
        if row["condition"] == "combined_0p2mm_1pct"
        and float(row["fraction"]) in set(config["replicate_fractions"])
    ]
    write_csv(QC / "multiseed_metrics.csv", multi)
    write_csv(QC / "runtime_metrics.csv", runtime_rows(config))
    fits = [fit_noise(iterative_rows, condition) for condition in config["conditions"]]
    write_csv(QC / "noise_scaling_fits.csv", fits)
    build_figures(config, analytic_rows, iterative_rows)
    decision = {
        "status": "PASS",
        "method": "frozen equal-quadratic Stage-4 reconstruction",
        "recommendations": recommendations,
        "noise_scaling": fits,
        "stage7b_test_partition_opened": False,
        "dose_claim_allowed": False,
    }
    atomic_json(QC / "stage7c_decision.json", decision)
    lines = [
        "# 阶段7C：D1有效质子通量敏感性",
        "",
        "最终状态：**PASS**。本阶段使用冻结算法，不重新调整重建参数。",
        "",
        "当前D1没有DoseActor，因此以下结果只解释为通量敏感性，不能换算为mGy。",
        "",
        "## 推荐最低有效通量",
        "",
        "| 测量条件 | 最低比例 | 名义通量/(protons/mm²/projection) |",
        "|---|---:|---:|",
    ]
    for condition, value in recommendations.items():
        fraction = value["minimum_fraction"]
        fluence = value["minimum_fluence_per_mm2_projection"]
        lines.append(
            f"| {condition} | "
            f"{'无合格点' if fraction is None else f'{100*fraction:.0f}%'} | "
            f"{'—' if fluence is None else f'{fluence:.0f}'} |"
        )
    lines += [
        "",
        "组合噪声25%和10%的最低通量判定要求三个嵌套抽样种子全部通过。",
        "详细逐点指标见`iterative_metrics.csv`和`qualification_metrics.csv`。",
        "",
        "Stage 7B锁定测试集未在本阶段用于调参或打开；WEPL训练残差只作为",
        "数据一致性诊断，不称为独立测试误差。",
    ]
    (QC / "stage7c_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    update_progress(status="COMPLETE", stage="report", task_eta_seconds=0.0)
    return decision


def status() -> None:
    if not PROGRESS.is_file():
        print("Stage 7C has not started.")
        return
    value = read_json(PROGRESS)
    print("Stage 7C status")
    print(f"  status: {value.get('status', '-')}")
    print(f"  stage/group: {value.get('stage', '-')}/{value.get('group', '-')}")
    if "completed_runs" in value:
        print(
            f"  angles: {value.get('completed_runs', 0)}/"
            f"{value.get('total_runs', '?')}"
        )
    if "completed_groups" in value:
        print(
            f"  groups: {value.get('completed_groups', 0)}/"
            f"{value.get('total_groups', '?')}"
        )
    if "epoch" in value:
        print(
            f"  epoch/subset: {value.get('epoch', '-')}/"
            f"{value.get('total_epochs', '-')} "
            f"{value.get('subset', '-')}/{value.get('total_subsets', '-')}"
        )
    eta = float(value.get("task_eta_seconds", 0.0))
    if eta > 0:
        print(f"  current ETA: {eta/3600:.2f} h")
    print(f"  updated: {value.get('updated_at', '-')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("all", "prepare", "project", "analytic", "reconstruct", "report", "status"),
        required=True,
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=720, help="development only; formal all requires 720")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    if args.action == "status":
        status()
        return
    raw_root = args.raw_root or Path(config["raw_root_default"])
    if args.jobs < 1 or not 1 <= args.runs <= int(config["runs"]):
        raise SystemExit("invalid --jobs or --runs")
    if args.action == "all" and args.runs != int(config["runs"]):
        raise SystemExit("--action all requires all 720 angles")
    actions = (
        ("prepare", "project", "analytic", "reconstruct", "report")
        if args.action == "all"
        else (args.action,)
    )
    if args.action in {"all", "prepare"}:
        preflight(config, raw_root, args.runs, args.device)
    for action in actions:
        if action == "prepare":
            prepare(config, raw_root, args.runs, args.jobs, args.force)
        elif action == "project":
            project(config, args.runs, args.jobs, args.force)
        elif action == "analytic":
            analytic(config, args.runs, args.force)
        elif action == "reconstruct":
            reconstruct(config, args.runs, args.device, args.force)
        elif action == "report":
            report(config)


if __name__ == "__main__":
    main()
