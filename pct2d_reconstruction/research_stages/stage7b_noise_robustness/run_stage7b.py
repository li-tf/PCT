#!/usr/bin/env python3
"""Run Stage 7B in two resumable batches: screen, then confirm."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CODE = REPO / "pct2d_reconstruction"
CONFIG_PATH = HERE / "stage7b_config.json"
QC = HERE / "qc"
PROGRESS = QC / "progress.json"
_INITIAL_BUILT_THIS_PROCESS: set[str] = set()
sys.path[:0] = [
    str(HERE),
    str(CODE),
    str(CODE / "preprocessing"),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "analytic_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage4_iterative_optimization"),
    str(CODE / "research_stages/stage7_detector_effects"),
]

from stage7b_data import prepare_run  # noqa: E402
from stage7b_gpu import (  # noqa: E402
    atomic_json,
    evaluate,
    predict_analytic,
    predict_empirical,
    run_reconstruction,
    write_csv,
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    config["_resolved_wepl_model"] = str(resolve(config["wepl_model"]))
    return config


def update_progress(**values: Any) -> None:
    current = read_json(PROGRESS) if PROGRESS.is_file() else {}
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(PROGRESS, current)


def locate_run(root: Path, run_id: int) -> Path | None:
    for path in (
        root / f"run_{run_id:03d}",
        root / f"run_{run_id:04d}",
        root / f"angle_{run_id:03d}",
        root / f"{run_id:03d}",
    ):
        if path.is_dir():
            return path
    return None


def preflight(
    config: dict[str, Any], raw_root: Path, runs: int, device: int
) -> dict[str, Any]:
    gate = read_json(resolve(config["wepl_gate"]))
    missing = []
    root_bytes = 0
    for run_id in range(runs):
        directory = locate_run(raw_root, run_id)
        absent = (
            list(config["required_root"])
            if directory is None
            else [
                name
                for name in config["required_root"]
                if not (directory / name).is_file()
                or (directory / name).stat().st_size == 0
            ]
        )
        if absent:
            missing.append({"run_id": run_id, "files": absent})
        else:
            root_bytes += sum(
                (directory / name).stat().st_size
                for name in config["required_root"]
            )
    output = resolve(config["preprocessing_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output.parent).free
    required = int(float(config["minimum_free_gib"]) * 1024**3)
    gpu = {}
    try:
        import cupy as cp

        cp.cuda.Device(device).use()
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties["name"]
        gpu = {
            "status": "PASS",
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "memory_bytes": int(properties["totalGlobalMem"]),
        }
    except Exception as error:
        gpu = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
    passed = (
        not missing
        and gate.get("status") == "PASS"
        and resolve(config["wepl_model"]).is_file()
        and free >= required
        and gpu["status"] == "PASS"
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "raw_root": str(raw_root),
        "runs": runs,
        "missing_runs": len(missing),
        "first_missing": missing[:5],
        "root_bytes": root_bytes,
        "output_free_bytes": free,
        "required_free_bytes": required,
        "wepl_gate": gate.get("status"),
        "gpu": gpu,
    }
    atomic_json(QC / "preflight.json", result)
    if not passed:
        raise RuntimeError(f"Stage 7B preflight failed: {result}")
    return result


def group_root(
    config: dict[str, Any],
    mode: str,
    condition: str,
    seed: int,
    partition: str,
) -> Path:
    return (
        resolve(config["preprocessing_output"])
        / mode
        / condition
        / f"seed_{seed}"
        / partition
    )


def prepare(
    config: dict[str, Any],
    raw_root: Path,
    runs: int,
    jobs: int,
    force: bool,
) -> dict[str, Any]:
    output = resolve(config["preprocessing_output"])
    if force and output.exists():
        shutil.rmtree(output)
    manifest_path = QC / "input_manifest.json"
    invariant = {
        "config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "raw_root": str(raw_root.resolve()),
        "runs": runs,
        "source_hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                CONFIG_PATH,
                HERE / "stage7b_data.py",
                HERE / "stage7b_gpu.py",
                HERE / "run_stage7b.py",
            )
        },
    }
    digest = canonical_hash(invariant)
    if manifest_path.is_file() and not force:
        old = read_json(manifest_path)
        if old.get("invariant_sha256") != digest:
            # Runtime orchestration and reporting may be patched without
            # invalidating the 32-GiB event-stable preparation.  Reuse it only
            # when the configuration and the actual data-construction module
            # are byte-identical; any change to either still requires force.
            old_sources = old.get("source_hashes", {})
            data_hash = invariant["source_hashes"]["stage7b_data.py"]
            compatible_runtime_patch = (
                old.get("config") == invariant["config"]
                and old_sources.get("stage7b_data.py") == data_hash
                and (QC / "preparation_summary.json").is_file()
                and read_json(QC / "preparation_summary.json").get("status")
                == "PASS"
            )
            if not compatible_runtime_patch:
                raise RuntimeError(
                    "Stage 7B data configuration changed; use --force or a "
                    "new output"
                )
            print(
                "Stage 7B runtime code changed but prepared-data invariant "
                "is unchanged; reusing existing ROOT preparation.",
                flush=True,
            )
    atomic_json(
        manifest_path,
        {**invariant, "invariant_sha256": digest, "test_partition_opened": False},
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    update_progress(
        status="RUNNING",
        batch="screen",
        stage="prepare",
        completed_runs=0,
        total_runs=runs,
    )
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                prepare_run,
                run_id,
                str(raw_root),
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
                stage="prepare",
                completed_runs=count,
                total_runs=runs,
                elapsed_seconds=elapsed,
                task_eta_seconds=eta,
            )
            if count % 5 == 0 or count == runs:
                print(
                    f"prepare ROOT {count:03d}/{runs}: "
                    f"elapsed={elapsed/60:.1f} min ETA={eta/60:.1f} min",
                    flush=True,
                )
    rows.sort(
        key=lambda row: (
            int(row["run_id"]),
            str(row["condition"]),
            int(row["noise_seed"]),
        )
    )
    write_csv(QC / "preparation_runs.csv", rows)
    totals = {}
    for condition in config["noise_conditions"]:
        for seed in (
            config["noise_seeds"]
            if condition == "combined_0p2mm_1pct"
            else [config["noise_seeds"][0]]
        ):
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and int(row["noise_seed"]) == int(seed)
            ]
            key = f"{condition}/seed_{seed}"
            totals[key] = {
                name: sum(int(row[name]) for row in selected)
                for name in (
                    "common_events",
                    "physical",
                    "accepted",
                    "train",
                    "validation",
                    "test",
                    "screen",
                )
            }
            totals[key]["noise_wepl_rmse_mm"] = float(
                np.average(
                    [row["noise_wepl_rmse_mm"] for row in selected],
                    weights=[row["accepted"] for row in selected],
                )
            )
    result = {
        "status": "PASS",
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "groups": totals,
        "split_identity": ["RunID", "EventID"],
        "test_partition_opened": False,
    }
    atomic_json(QC / "preparation_summary.json", result)
    return result


def fit_noise_model(config: dict[str, Any], runs: int) -> dict[str, Any]:
    from robust_models import fit_noise_model as fit_model
    from iterative_reconstruction.physics import load_wepl_model

    seed = int(config["noise_seeds"][0])
    train_root = group_root(
        config, "confirm", "combined_0p2mm_1pct", seed, "train"
    )
    validation_root = group_root(
        config, "confirm", "combined_0p2mm_1pct", seed, "validation"
    )
    energies, residuals = [], []
    for run_id in range(runs):
        with np.load(
            train_root / "metadata" / f"meta{run_id:04d}.npz",
            allow_pickle=False,
        ) as source:
            count = len(source["event_id"])
            stride = max(1, count // 5000)
            index = slice(None, None, stride)
            energies.append(np.asarray(source["measured_eout_mev"][index]))
            residuals.append(
                np.asarray(source["measured_wepl_mm"][index])
                - np.asarray(source["ideal_wepl_mm"][index])
            )
    energy = np.concatenate(energies)
    residual = np.concatenate(residuals)
    model = fit_model(energy, residual, config["noise_model"])
    model_path = QC / "empirical_noise_model.npz"
    model.save(model_path)
    config["_noise_model_path"] = str(model_path)

    val_energy, val_residual = [], []
    for run_id in range(runs):
        with np.load(
            validation_root / "metadata" / f"meta{run_id:04d}.npz",
            allow_pickle=False,
        ) as source:
            count = len(source["event_id"])
            stride = max(1, count // 5000)
            index = slice(None, None, stride)
            val_energy.append(np.asarray(source["measured_eout_mev"][index]))
            val_residual.append(
                np.asarray(source["measured_wepl_mm"][index])
                - np.asarray(source["ideal_wepl_mm"][index])
            )
    val_energy = np.concatenate(val_energy)
    val_residual = np.concatenate(val_residual)
    empirical = predict_empirical(
        {
            "energy_mev": model.energy_mev,
            "sigma_mm": model.sigma_mm,
            "minimum_sigma_mm": model.minimum_sigma_mm,
        },
        val_energy,
    )
    wepl = load_wepl_model(
        "g4_water_calibrated", resolve(config["wepl_model"])
    )
    analytic = predict_analytic(
        val_energy,
        wepl.energy_mev,
        wepl.range_mm,
        0.01,
        float(config["noise_model"]["minimum_sigma_mm"]),
    )
    rows = []
    summaries = {}
    for name, predicted in (
        ("analytic", analytic),
        ("empirical", empirical),
    ):
        order = np.argsort(predicted)
        boundaries = np.linspace(0, len(order), 11, dtype=int)
        passing = 0
        for decile in range(10):
            selected = order[boundaries[decile] : boundaries[decile + 1]]
            actual = float(np.sqrt(np.mean(val_residual[selected] ** 2)))
            expected = float(np.sqrt(np.mean(predicted[selected] ** 2)))
            ratio = actual / expected
            low, high = config["noise_model"]["calibration_ratio_range"]
            passed = float(low) <= ratio <= float(high)
            passing += int(passed)
            rows.append(
                {
                    "model": name,
                    "decile": decile + 1,
                    "count": len(selected),
                    "predicted_sigma_rms_mm": expected,
                    "actual_rmse_mm": actual,
                    "ratio": ratio,
                    "passed": passed,
                }
            )
        summaries[name] = {
            "passing_deciles": passing,
            "eligible": passing
            >= int(config["noise_model"]["minimum_passing_deciles"]),
            "coverage_1sigma": float(
                np.mean(np.abs(val_residual) <= predicted)
            ),
            "coverage_2sigma": float(
                np.mean(np.abs(val_residual) <= 2 * predicted)
            ),
        }
    write_csv(QC / "noise_calibration_deciles.csv", rows)
    result = {
        "status": "PASS",
        "training_sample_count": len(energy),
        "validation_sample_count": len(val_energy),
        "model_path": str(model_path),
        "models": summaries,
        "test_partition_opened": False,
    }
    atomic_json(QC / "noise_model_summary.json", result)
    return result


def ensure_initial(
    config: dict[str, Any],
    root: Path,
    runs: int,
    jobs: int,
    force: bool,
    tag: str,
) -> Path:
    from preprocessing import projection

    effective_force = force and tag not in _INITIAL_BUILT_THIS_PROCESS
    _INITIAL_BUILT_THIS_PROCESS.add(tag)
    ddb = root / "projections_ddb"
    ddb.mkdir(parents=True, exist_ok=True)
    pending = [
        run_id
        for run_id in range(runs)
        if effective_force or not (ddb / f"proj{run_id:04d}.mhd").is_file()
    ]
    update_progress(
        stage="projection",
        group=tag,
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
            update_progress(
                stage="projection",
                group=tag,
                completed_runs=done,
                total_runs=runs,
            )
            if done % 20 == 0 or done == runs:
                print(f"DDB {tag}: {done:03d}/{runs}", flush=True)
    output = root / "analytic/recon_ddb_nohann.mhd"
    if effective_force or not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        log = QC / "logs" / f"fdk_{tag.replace('/', '_')}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        geometry = resolve(config["geometry"])
        if not geometry.is_file():
            geometry = QC / "geometry.xml"
            if not geometry.is_file():
                acquisition = read_json(
                    CODE / "experiments/experiment0716.json"
                )["acquisition"]
                geometry_command = [
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
                geometry_log = QC / "logs/geometry.log"
                with geometry_log.open("w", encoding="utf-8") as stream:
                    subprocess.run(
                        geometry_command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
        command = [
            str(REPO / ".venv-gate/bin/pctfdk"),
            "--lowmem",
            "--geometry",
            str(geometry),
            "--path",
            str(ddb),
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
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=True
            )
    return output


def reconstruction_output(
    config: dict[str, Any], mode: str, tag: str, candidate: str
) -> Path:
    return resolve(config["reconstruction_output"]) / mode / tag / candidate


def image_metrics(path: Path) -> dict[str, Any]:
    import run_stage7
    from analytic_reconstruction import rsp_metrics

    measured = run_stage7.image_metrics(path)
    definition = (
        CODE / "simulation/simulation0716/truth_geometry_definition.json"
    )
    image, x, z, _ = rsp_metrics.read_mhd(path)
    centers = read_json(definition)["geometry"]["insert_centers_xz_mm"]
    edge = rsp_metrics.aluminium_edge_widths(image, x, z, centers)
    valid = np.asarray(
        [
            float(row["width_10_90_mm"])
            for row in edge
            if row.get("valid") and np.isfinite(row["width_10_90_mm"])
        ]
    )
    measured["aluminium_edge_10_90_median_mm"] = (
        float(np.median(valid)) if len(valid) else math.nan
    )
    return measured


def metric_row(
    label: str, summary: dict[str, Any], mode: str, seed: int
) -> dict[str, Any]:
    path = Path(summary["best_image"])
    measured = image_metrics(path)
    output = {
        "label": label,
        "mode": mode,
        "seed": seed,
        "candidate": summary["candidate"],
        "best_epoch": summary["best_epoch"],
        "validation_ideal_rmse_mm": summary["validation_ideal_rmse_mm"],
        "validation_ideal_abs_p99_mm": summary[
            "validation_ideal_abs_p99_mm"
        ],
        "image_path": str(path),
    }
    output.update(
        {
            key: value
            for key, value in measured.items()
            if isinstance(value, (int, float, np.number, bool))
        }
    )
    return output


def aluminium_error(row: dict[str, Any]) -> float:
    reference = 2.094511207867794
    path = (
        CODE
        / "research_stages/stage6a_mlic_reference/qc/"
        "mlic_reference_200mev.csv"
    )
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                if item["material"] == "Aluminium":
                    reference = float(item["mlic_rsp_200mev"])
                    break
    return float(row["insert_peak_mean"]) / reference - 1.0


def candidate_eligible(
    row: dict[str, Any], baseline: dict[str, Any], config: dict[str, Any]
) -> tuple[bool, dict[str, float | bool]]:
    selection = config["selection"]
    rmse_improvement = 1 - float(row["validation_ideal_rmse_mm"]) / float(
        baseline["validation_ideal_rmse_mm"]
    )
    p99_improvement = 1 - float(row["validation_ideal_abs_p99_mm"]) / float(
        baseline["validation_ideal_abs_p99_mm"]
    )
    al_degradation_pp = (
        abs(aluminium_error(row)) - abs(aluminium_error(baseline))
    ) * 100
    edge = float(row.get("aluminium_edge_10_90_median_mm", math.nan))
    base_edge = float(
        baseline.get("aluminium_edge_10_90_median_mm", math.nan)
    )
    edge_degradation = (
        edge / base_edge - 1 if np.isfinite(edge) and np.isfinite(base_edge) else 0
    )
    eligible = (
        rmse_improvement
        >= -float(selection["screen_validation_rmse_max_degradation"])
        and (
            rmse_improvement > 0
            or p99_improvement
            >= float(selection["screen_p99_min_improvement"])
        )
        and al_degradation_pp
        <= float(
            selection["aluminium_error_max_degradation_percentage_points"]
        )
        and edge_degradation
        <= float(selection["edge_width_max_relative_degradation"])
    )
    return eligible, {
        "rmse_improvement": rmse_improvement,
        "p99_improvement": p99_improvement,
        "aluminium_error_degradation_pp": al_degradation_pp,
        "edge_width_degradation": edge_degradation,
        "eligible": eligible,
    }


def run_one(
    config: dict[str, Any],
    mode: str,
    condition: str,
    seed: int,
    candidate: dict[str, Any],
    epochs: int,
    device: int,
    runs: int,
    jobs: int,
    force: bool,
) -> dict[str, Any]:
    train = group_root(config, mode, condition, seed, "train")
    validation = group_root(config, mode, condition, seed, "validation")
    # Every screen reconstruction uses the same primary-seed, train-only
    # DDB-FDK initialization.  This removes initialization as a confounder,
    # avoids validation/test leakage, and prevents six redundant DDB builds.
    initial_root = (
        group_root(
            config,
            "screen",
            "combined_0p2mm_1pct",
            int(config["noise_seeds"][0]),
            "train",
        )
        if mode == "screen"
        else train
    )
    initial = ensure_initial(
        config,
        initial_root,
        runs,
        jobs,
        force,
        (
            "screen/shared_primary_train"
            if mode == "screen"
            else f"{mode}/{condition}/seed_{seed}"
        ),
    )
    tag = f"{condition}/seed_{seed}"
    output = reconstruction_output(
        config, mode, tag, str(candidate["name"])
    )
    return run_reconstruction(
        str(candidate["name"]),
        candidate,
        train,
        validation,
        initial,
        output,
        resolve(config["wepl_model"]),
        config,
        epochs,
        device,
        runs,
        update_progress,
        force,
    )


def screen(
    config: dict[str, Any],
    raw_root: Path,
    runs: int,
    jobs: int,
    device: int,
    force: bool,
) -> dict[str, Any]:
    update_progress(status="RUNNING", batch="screen", stage="preflight")
    preflight(config, raw_root, runs, device)
    prepare(config, raw_root, runs, jobs, force)
    noise = fit_noise_model(config, runs)
    config["_noise_model_path"] = noise["model_path"]
    primary = int(config["noise_seeds"][0])
    equal = {"name": "equal_quadratic", "weight": "equal", "huber_z": None}
    completed_tasks = 0
    estimated_tasks = (
        len(config["noise_conditions"])
        + len(config["screen"]["candidates"])
        + 1
        + 2 * (len(config["noise_seeds"]) - 1)
    )
    update_progress(
        completed_reconstructions=completed_tasks,
        estimated_reconstructions=estimated_tasks,
    )

    source_rows = []
    for condition in config["noise_conditions"]:
        result = run_one(
            config,
            "screen",
            condition,
            primary,
            equal,
            int(config["screen"]["epochs"]),
            device,
            runs,
            jobs,
            force,
        )
        source_rows.append(metric_row(condition, result, "source", primary))
        completed_tasks += 1
        update_progress(completed_reconstructions=completed_tasks)
    write_csv(QC / "noise_source_image_metrics.csv", source_rows)

    combined_rows = [
        next(row for row in source_rows if row["label"] == "combined_0p2mm_1pct")
    ]
    summaries = {"equal_quadratic": combined_rows[0]}
    candidates = list(config["screen"]["candidates"])[1:]
    for candidate in candidates:
        if (
            candidate["weight"] == "analytic"
            and not noise["models"]["analytic"]["eligible"]
        ):
            continue
        if (
            candidate["weight"] == "empirical"
            and not noise["models"]["empirical"]["eligible"]
        ):
            continue
        result = run_one(
            config,
            "screen",
            "combined_0p2mm_1pct",
            primary,
            candidate,
            int(config["screen"]["epochs"]),
            device,
            runs,
            jobs,
            force,
        )
        row = metric_row(
            str(candidate["name"]), result, "candidate", primary
        )
        combined_rows.append(row)
        summaries[str(candidate["name"])] = row
        completed_tasks += 1
        update_progress(completed_reconstructions=completed_tasks)

    baseline = summaries["equal_quadratic"]
    huber_rows = [
        row for row in combined_rows if str(row["candidate"]).startswith("huber_")
    ]
    if huber_rows and noise["models"]["empirical"]["eligible"]:
        best_huber = min(
            huber_rows, key=lambda row: float(row["validation_ideal_rmse_mm"])
        )
        huber_candidate = next(
            item
            for item in config["screen"]["candidates"]
            if item["name"] == best_huber["candidate"]
        )
        combined = {
            "name": f"empirical_huber_z{str(huber_candidate['huber_z']).replace('.', 'p')}",
            "weight": "empirical",
            "huber_z": huber_candidate["huber_z"],
        }
        result = run_one(
            config,
            "screen",
            "combined_0p2mm_1pct",
            primary,
            combined,
            int(config["screen"]["epochs"]),
            device,
            runs,
            jobs,
            force,
        )
        row = metric_row(combined["name"], result, "candidate", primary)
        combined_rows.append(row)
        summaries[combined["name"]] = row
        completed_tasks += 1
        update_progress(completed_reconstructions=completed_tasks)

    eligible = []
    selection_rows = []
    candidate_map = {
        item["name"]: item for item in config["screen"]["candidates"]
    }
    for row in combined_rows:
        if row["candidate"] == "equal_quadratic":
            details = {
                "rmse_improvement": 0.0,
                "p99_improvement": 0.0,
                "aluminium_error_degradation_pp": 0.0,
                "edge_width_degradation": 0.0,
                "eligible": True,
            }
        else:
            passed, details = candidate_eligible(row, baseline, config)
            if passed:
                eligible.append(row)
        selection_rows.append({**row, **details})
    provisional = (
        min(eligible, key=lambda row: float(row["validation_ideal_rmse_mm"]))
        if eligible
        else baseline
    )
    if provisional["candidate"] not in candidate_map:
        output = reconstruction_output(
            config,
            "screen",
            f"combined_0p2mm_1pct/seed_{primary}",
            provisional["candidate"],
        )
        saved = read_json(output / "run_summary.json")
        candidate_map[provisional["candidate"]] = {
            "name": provisional["candidate"],
            "weight": saved["weight"],
            "huber_z": saved["huber_z"],
        }

    seed_rows = []
    if provisional["candidate"] != "equal_quadratic":
        for seed in config["noise_seeds"]:
            for candidate in (equal, candidate_map[provisional["candidate"]]):
                if seed == primary:
                    row = summaries[candidate["name"]]
                else:
                    result = run_one(
                        config,
                        "screen",
                        "combined_0p2mm_1pct",
                        int(seed),
                        candidate,
                        int(config["screen"]["epochs"]),
                        device,
                        runs,
                        jobs,
                        force,
                    )
                    row = metric_row(
                        candidate["name"], result, "seed", int(seed)
                    )
                    completed_tasks += 1
                    update_progress(completed_reconstructions=completed_tasks)
                seed_rows.append(row)
    else:
        seed_rows.append(baseline)
    write_csv(QC / "screen_candidate_metrics.csv", selection_rows)
    write_csv(QC / "seed_robustness_metrics.csv", seed_rows)

    seed_pass = provisional["candidate"] != "equal_quadratic"
    nonworse = improved = 0
    seed_details = []
    if seed_pass:
        for seed in config["noise_seeds"]:
            base = next(
                row
                for row in seed_rows
                if int(row["seed"]) == int(seed)
                and row["candidate"] == "equal_quadratic"
            )
            winner = next(
                row
                for row in seed_rows
                if int(row["seed"]) == int(seed)
                and row["candidate"] == provisional["candidate"]
            )
            gain = 1 - float(winner["validation_ideal_rmse_mm"]) / float(
                base["validation_ideal_rmse_mm"]
            )
            nonworse += int(
                gain
                >= -float(
                    config["selection"][
                        "screen_validation_rmse_max_degradation"
                    ]
                )
            )
            improved += int(gain > 0)
            seed_details.append({"seed": int(seed), "rmse_improvement": gain})
        seed_pass = (
            nonworse
            >= int(config["selection"]["seed_required_nonworse"])
            and improved
            >= int(config["selection"]["seed_required_primary_improvement"])
        )
    winner = provisional["candidate"] if seed_pass else "equal_quadratic"
    status = "PASS" if winner != "equal_quadratic" else "NO_PROMOTION"
    result = {
        "status": status,
        "winner": winner,
        "winner_config": candidate_map.get(winner, equal),
        "provisional_winner": provisional["candidate"],
        "seed_gate_pass": seed_pass,
        "seed_details": seed_details,
        "test_partition_opened": False,
        "next_action": (
            "run --action confirm"
            if winner != "equal_quadratic"
            else "no robust candidate passed; confirm will finalize a negative result"
        ),
    }
    atomic_json(QC / "screen_decision.json", result)
    build_screen_figures(config, source_rows, selection_rows, seed_rows)
    build_screen_summary(result, source_rows, selection_rows)
    update_progress(status="COMPLETE", batch="screen", stage="screen_decision")
    return result


def build_screen_figures(
    config: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    with np.load(QC / "empirical_noise_model.npz", allow_pickle=False) as model:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(
            model["energy_mev"],
            model["sigma_mm"],
            color="#2878b5",
            marker="o",
        )
        ax.set_xlabel("Exit energy (MeV)")
        ax.set_ylabel("Empirical WEPL sigma (mm)")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(assets / "empirical_wepl_noise_model.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = [row["label"] for row in source_rows]
    values = [float(row["phantom_rmse_vs_rsp_truth"]) for row in source_rows]
    ax.bar(labels, values, color="#2878b5")
    ax.set_ylabel("Phantom RSP RMSE")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "noise_source_separation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    labels = [row["candidate"] for row in candidate_rows]
    values = [
        float(row["validation_ideal_rmse_mm"]) for row in candidate_rows
    ]
    colors = ["#2878b5" if row["eligible"] else "#999999" for row in candidate_rows]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Validation ideal-WEPL RMSE (mm)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "candidate_validation.png", dpi=180)
    plt.close(fig)

    if len(seed_rows) > 1:
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        methods = sorted({str(row["candidate"]) for row in seed_rows})
        seeds = sorted({int(row["seed"]) for row in seed_rows})
        x = np.arange(len(seeds))
        width = 0.8 / len(methods)
        for index, method in enumerate(methods):
            values = [
                float(
                    next(
                        row["validation_ideal_rmse_mm"]
                        for row in seed_rows
                        if int(row["seed"]) == seed
                        and row["candidate"] == method
                    )
                )
                for seed in seeds
            ]
            ax.bar(
                x + (index - 0.5 * (len(methods) - 1)) * width,
                values,
                width,
                label=method,
            )
        ax.set_xticks(x, [str(seed) for seed in seeds])
        ax.set_xlabel("Noise seed")
        ax.set_ylabel("Validation ideal-WEPL RMSE (mm)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(assets / "seed_robustness.png", dpi=180)
        plt.close(fig)


def build_screen_summary(
    decision: dict[str, Any],
    source_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    lines = [
        "# 阶段7B第一批：噪声准备与候选筛选",
        "",
        f"状态：**{decision['status']}**。",
        "",
        "数据在过滤前按`(RunID, EventID)`固定划分为80%训练、10%验证和10%测试；",
        "本批未读取测试指标。",
        "",
        "## 噪声源分离",
        "",
        "| 条件 | 水标准差 | 模体RMSE | 验证理想WEPL RMSE/mm |",
        "|---|---:|---:|---:|",
    ]
    for row in source_rows:
        lines.append(
            f"| {row['label']} | {float(row['water_std']):.6f} | "
            f"{float(row['phantom_rmse_vs_rsp_truth']):.6f} | "
            f"{float(row['validation_ideal_rmse_mm']):.5f} |"
        )
    lines += [
        "",
        "## 组合噪声候选",
        "",
        "| 方法 | 验证理想WEPL RMSE/mm | p99/mm | 是否合格 |",
        "|---|---:|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['candidate']} | "
            f"{float(row['validation_ideal_rmse_mm']):.5f} | "
            f"{float(row['validation_ideal_abs_p99_mm']):.5f} | "
            f"{row['eligible']} |"
        )
    lines += [
        "",
        f"冻结候选：`{decision['winner']}`。",
        "",
        "只有冻结候选不是等权基线时，第二批才执行两套80%数据正式重建。",
    ]
    (QC / "screen_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def confirm(
    config: dict[str, Any],
    raw_root: Path,
    runs: int,
    jobs: int,
    device: int,
    force: bool,
) -> dict[str, Any]:
    decision_path = QC / "screen_decision.json"
    if not decision_path.is_file():
        raise RuntimeError("run --action screen first")
    decision = read_json(decision_path)
    if not (QC / "preparation_summary.json").is_file():
        raise RuntimeError("Stage 7B preparation is incomplete")
    if decision["winner"] == "equal_quadratic":
        result = {
            "status": "NO_PROMOTION",
            "winner": "equal_quadratic",
            "reason": (
                "No robust candidate passed the primary screen gate; "
                "multi-seed and locked-test confirmation were not opened."
            ),
            "test_partition_opened": False,
        }
        atomic_json(QC / "stage7b_decision.json", result)
        build_final_summary(result, [], [])
        update_progress(
            status="COMPLETE",
            batch="confirm",
            stage="negative_result",
            candidate="equal_quadratic",
            completed_reconstructions=9,
            estimated_reconstructions=9,
            task_eta_seconds=0.0,
        )
        return result
    preflight(config, raw_root, runs, device)
    noise = read_json(QC / "noise_model_summary.json")
    config["_noise_model_path"] = noise["model_path"]
    seed = int(config["noise_seeds"][0])
    equal = {"name": "equal_quadratic", "weight": "equal", "huber_z": None}
    winner = decision["winner_config"]
    update_progress(status="RUNNING", batch="confirm", stage="formal_reconstruction")
    update_progress(completed_reconstructions=0, estimated_reconstructions=2)
    rows = []
    summaries = []
    for candidate in (equal, winner):
        result = run_one(
            config,
            "confirm",
            "combined_0p2mm_1pct",
            seed,
            candidate,
            int(config["reconstruction"]["epochs"]),
            device,
            runs,
            jobs,
            force,
        )
        summaries.append(result)
        rows.append(metric_row(candidate["name"], result, "confirm", seed))
        update_progress(completed_reconstructions=len(rows))
    write_csv(QC / "confirm_image_metrics.csv", rows)

    # This file is the one-way gate that permits the first test evaluation.
    frozen = {
        "status": "FROZEN",
        "baseline": equal,
        "candidate": winner,
        "selected_from": "validation",
        "test_partition_opened": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_json(QC / "frozen_final.json", frozen)
    manifest = read_json(QC / "input_manifest.json")
    manifest["test_partition_opened"] = True
    atomic_json(QC / "input_manifest.json", manifest)
    test_root = group_root(
        config, "confirm", "combined_0p2mm_1pct", seed, "test"
    )
    test_rows = []
    for row, summary in zip(rows, summaries):
        metrics = evaluate(
            Path(summary["best_image"]),
            test_root,
            device,
            config,
            runs,
            update_progress,
        )
        test_rows.append(
            {
                "candidate": row["candidate"],
                "best_epoch": row["best_epoch"],
                **metrics,
            }
        )
    write_csv(QC / "test_wepl_metrics.csv", test_rows)
    baseline_row = next(row for row in rows if row["candidate"] == "equal_quadratic")
    winner_row = next(row for row in rows if row["candidate"] == winner["name"])
    baseline_test = next(
        row for row in test_rows if row["candidate"] == "equal_quadratic"
    )
    winner_test = next(
        row for row in test_rows if row["candidate"] == winner["name"]
    )
    selection = config["selection"]
    wepl_gain = 1 - float(winner_test["ideal_rmse_mm"]) / float(
        baseline_test["ideal_rmse_mm"]
    )
    water_gain = 1 - float(winner_row["water_std"]) / float(
        baseline_row["water_std"]
    )
    image_gain = 1 - float(
        winner_row["phantom_rmse_vs_rsp_truth"]
    ) / float(baseline_row["phantom_rmse_vs_rsp_truth"])
    al_degradation = (
        abs(aluminium_error(winner_row)) - abs(aluminium_error(baseline_row))
    ) * 100
    winner_edge = float(
        winner_row.get("aluminium_edge_10_90_median_mm", math.nan)
    )
    baseline_edge = float(
        baseline_row.get("aluminium_edge_10_90_median_mm", math.nan)
    )
    edge_degradation = (
        winner_edge / baseline_edge - 1
        if np.isfinite(winner_edge) and np.isfinite(baseline_edge)
        else 0
    )
    image_pass = (
        water_gain
        >= float(selection["image_noise_or_rmse_min_improvement"])
        and image_gain >= 0
    ) or (
        image_gain >= float(selection["image_noise_or_rmse_min_improvement"])
        and water_gain >= 0
    )
    passed = (
        wepl_gain
        >= float(selection["test_ideal_wepl_rmse_min_improvement"])
        and image_pass
        and al_degradation
        <= float(
            selection["aluminium_error_max_degradation_percentage_points"]
        )
        and edge_degradation
        <= float(selection["edge_width_max_relative_degradation"])
    )
    result = {
        "status": "PASS" if passed else "NO_PROMOTION",
        "winner": winner["name"] if passed else "equal_quadratic",
        "candidate": winner["name"],
        "test_partition_opened": True,
        "test_ideal_wepl_rmse_improvement": wepl_gain,
        "water_std_improvement": water_gain,
        "phantom_rmse_improvement": image_gain,
        "aluminium_error_degradation_percentage_points": al_degradation,
        "edge_width_relative_degradation": edge_degradation,
        "acceptance_passed": passed,
    }
    atomic_json(QC / "stage7b_decision.json", result)
    build_figures(config, rows, test_rows)
    build_final_summary(result, rows, test_rows)
    update_progress(status="COMPLETE", batch="confirm", stage="report")
    return result


def build_figures(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    from iterative_reconstruction.mhd_io import read_image_2d

    assets = QC / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    labels = [row["candidate"] for row in test_rows]
    values = [float(row["ideal_rmse_mm"]) for row in test_rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(labels, values, color=["#7f7f7f", "#2878b5"])
    ax.set_ylabel("Test ideal-WEPL RMSE (mm)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "test_wepl_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 4.7))
    if len(rows) == 1:
        axes = [axes]
    for axis, row in zip(axes, rows):
        image = read_image_2d(Path(row["image_path"]))[0]
        shown = axis.imshow(image, cmap="viridis", vmin=0.95, vmax=2.15)
        axis.set_title(row["candidate"])
        axis.set_axis_off()
    fig.colorbar(shown, ax=axes, label="RSP", shrink=0.82)
    fig.savefig(assets / "confirm_reconstruction_comparison.png", dpi=180)
    plt.close(fig)


def build_final_summary(
    decision: dict[str, Any],
    image_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> None:
    test_opened = bool(decision.get("test_partition_opened", False))
    lines = [
        "# 阶段7B：位置与能量噪声鲁棒重建",
        "",
        f"最终状态：**{decision['status']}**。",
        "",
        "阶段7B在过滤前使用`(RunID, EventID)`完成80%/10%/10%固定划分，",
        (
            "筛选只使用训练和验证数据；候选通过筛选并冻结后，测试集仅允许打开一次。"
            if test_opened
            else "筛选只使用训练和验证数据；由于没有候选通过筛选，测试集始终未打开。"
        ),
        "",
    ]
    if image_rows:
        lines += [
            "## 80%训练集正式重建",
            "",
            "| 方法 | 水标准差 | 模体RMSE | 铝平台RSP | 边缘宽度/mm |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in image_rows:
            lines.append(
                f"| {row['candidate']} | {float(row['water_std']):.6f} | "
                f"{float(row['phantom_rmse_vs_rsp_truth']):.6f} | "
                f"{float(row['insert_peak_mean']):.6f} | "
                f"{float(row.get('aluminium_edge_10_90_median_mm', math.nan)):.4f} |"
            )
        lines += [
            "",
            "## 独立测试质子",
            "",
            "| 方法 | ideal-WEPL RMSE/mm | MAE/mm | p99/mm |",
            "|---|---:|---:|---:|",
        ]
        for row in test_rows:
            lines.append(
                f"| {row['candidate']} | {float(row['ideal_rmse_mm']):.5f} | "
                f"{float(row['ideal_mae_mm']):.5f} | "
                f"{float(row['ideal_abs_p99_mm']):.5f} |"
            )
    lines += [
        "",
        f"最终保留方法：`{decision['winner']}`。",
        "",
        "机器可读门槛与逐项改善量见`stage7b_decision.json`。",
    ]
    (QC / "stage7b_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def status(config: dict[str, Any]) -> None:
    if not PROGRESS.is_file():
        print("Stage 7B has not started.")
        return
    value = read_json(PROGRESS)
    print("Stage 7B status")
    print(f"  status: {value.get('status', '-')}")
    print(f"  batch/stage: {value.get('batch', '-')}/{value.get('stage', '-')}")
    if "completed_runs" in value:
        print(
            f"  runs: {value.get('completed_runs', 0)}/"
            f"{value.get('total_runs', '?')}"
        )
    if "candidate" in value:
        print(f"  candidate: {value['candidate']}")
    if "completed_reconstructions" in value:
        print(
            f"  reconstructions: {value.get('completed_reconstructions', 0)}/"
            f"~{value.get('estimated_reconstructions', '?')}"
        )
    if "epoch" in value:
        print(
            f"  epoch/subset: {value['epoch']}/{value.get('total_epochs')} ; "
            f"{value.get('subset')}/{value.get('total_subsets')}"
        )
    if "pairs_per_second" in value:
        print(f"  rate: {float(value['pairs_per_second']):,.0f} pairs/s")
    if "task_eta_seconds" in value:
        print(f"  ETA: {float(value['task_eta_seconds'])/3600:.2f} h")
    if "gpu_memory_bytes" in value:
        print(f"  GPU memory peak: {value['gpu_memory_bytes']/1024**3:.2f} GiB")
    decision = QC / "screen_decision.json"
    if decision.is_file():
        selected = read_json(decision)
        print(
            f"  screen winner: {selected.get('winner')} "
            f"({selected.get('status')})"
        )
    final = QC / "stage7b_decision.json"
    if final.is_file():
        selected = read_json(final)
        print(f"  final: {selected.get('winner')} ({selected.get('status')})")
    print(f"  updated: {value.get('updated_at', '-')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 7B detector-noise robustness workflow"
    )
    parser.add_argument(
        "--action", required=True, choices=["screen", "confirm", "status"]
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    runs = int(args.runs or config["runs"])
    if args.action == "status":
        status(config)
        return
    if args.raw_root is None:
        raise SystemExit("--raw-root is required for screen and confirm")
    raw_root = args.raw_root.resolve()
    if args.jobs < 1 or runs < 1 or runs > int(config["runs"]):
        raise SystemExit("require jobs >= 1 and 1 <= runs <= 720")
    if args.action == "screen":
        result = screen(
            config,
            raw_root,
            runs,
            args.jobs,
            args.device,
            args.force,
        )
    else:
        result = confirm(
            config,
            raw_root,
            runs,
            args.jobs,
            args.device,
            args.force,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
