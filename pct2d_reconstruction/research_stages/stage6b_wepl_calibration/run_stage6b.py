#!/usr/bin/env python3
"""Build, validate and deploy the independent Stage-6B WEPL calibration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares
import uproot


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPO_ROOT = CODE_ROOT.parent
CONFIG_PATH = HERE / "stage6b_config.json"
QC_ROOT = HERE / "qc"
sys.path[:0] = [
    str(HERE),
    str(CODE_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
    str(HERE.parent / "stage3_robust_weighting"),
]

from calibration_model import range_grid, wepl, write_model  # noqa: E402
from iterative_reconstruction.physics import (  # noqa: E402
    ENERGY_BIN_MEV,
    energies_to_wepl_model,
    load_wepl_model,
)
from preprocessing.paircuts import read_mhd, write_mhd  # noqa: E402
from preprocessing import projection  # noqa: E402
from iterative_reconstruction.mhd_io import read_image_2d  # noqa: E402
from stage3_io import partition_masks, read_packed_mask  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
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


def enumerate_cases(sim_config: dict) -> list[dict]:
    # Reproduce the Windows package's predeclared thickness rule without
    # requiring OpenGATE to be importable on the analysis host.
    energies = np.asarray(sim_config["energies_mev"], dtype=np.float64)
    bb78 = load_wepl_model("bb78")
    nominal = bb78.range_mm[
        np.floor(energies / 0.0001 + 0.5).astype(np.int64)
    ]
    split_lookup = {
        int(energy): split
        for split, values in sim_config["split_by_energy"].items()
        for energy in values
    }
    cases = []
    for energy, range_mm in zip(energies, nominal):
        for fraction in sim_config["thickness_fractions"]:
            thickness = round(float(range_mm) * float(fraction) * 2.0) / 2.0
            cases.append(
                {
                    "case_index": len(cases),
                    "case_id": (
                        f"e{int(energy):03d}_"
                        f"f{int(round(100*float(fraction))):02d}"
                    ),
                    "energy_mev": float(energy),
                    "thickness_fraction": float(fraction),
                    "water_thickness_mm": max(0.5, thickness),
                    "split": split_lookup[int(energy)],
                }
            )
    return cases


def find_tree(root: uproot.ReadOnlyDirectory, preferred: str):
    if preferred in root:
        return root[preferred]
    trees = [value for _, value in root.items() if hasattr(value, "num_entries")]
    if len(trees) != 1:
        raise RuntimeError(f"cannot identify {preferred} tree: {root.file.file_path}")
    return trees[0]


def branch_array(tree, name: str) -> np.ndarray:
    """Read a ROOT branch basket-by-basket.

    Some Linux uproot builds can stall while assembling these Windows-written
    TTrees through the high-level ``tree.arrays`` executor.  Direct basket
    assembly is deterministic and gives the same values without depending on
    that executor path.
    """

    branch = tree[name]
    pieces = [
        np.asarray(
            branch.basket(index).array(branch.interpretation)
        )
        for index in range(branch.num_baskets)
    ]
    values = np.concatenate(pieces) if pieces else np.empty(0)
    if len(values) != int(tree.num_entries):
        raise RuntimeError(
            f"{name}: read {len(values)} values, expected {int(tree.num_entries)}"
        )
    return values


def primary_energy(path: Path, tree_name: str) -> tuple[np.ndarray, np.ndarray]:
    with uproot.open(path) as root:
        tree = find_tree(root, tree_name)
        arrays = {
            name: branch_array(tree, name)
            for name in (
                "EventID",
                "TrackID",
                "KineticEnergy",
                "Direction_Z",
                "PreGlobalTime",
            )
        }
    mask = (arrays["TrackID"] == 1) & (arrays["Direction_Z"] > 0.0)
    event = np.asarray(arrays["EventID"][mask], dtype=np.int64)
    energy = np.asarray(arrays["KineticEnergy"][mask], dtype=np.float64)
    time_values = np.asarray(arrays["PreGlobalTime"][mask], dtype=np.float64)
    # Sort by EventID, then by time, so recrossing events deterministically use
    # the first forward measurement at each reference plane.
    order = np.lexsort((time_values, event))
    event, energy = event[order], energy[order]
    unique = np.r_[True, event[1:] != event[:-1]]
    return event[unique], energy[unique]


def ingest(config: dict, force: bool) -> dict:
    sim_config = read_json(resolve(config["simulation_config"]))
    cases = enumerate_cases(sim_config)
    data_root = resolve(config["simulation_data"])
    cache_root = resolve(config["sample_cache"])
    cache_root.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for number, case in enumerate(cases, 1):
        output = cache_root / f"{case['case_id']}.npz"
        if output.is_file() and not force:
            with np.load(output) as cached:
                count = int(cached["energy_in"].size)
                raw_count = int(cached["raw_count"])
        else:
            case_dir = data_root / case["case_id"]
            ein_event, ein = primary_energy(
                case_dir / "PhaseSpaceIn.root", "PhaseSpaceIn"
            )
            eout_event, eout = primary_energy(
                case_dir / "PhaseSpaceOut.root", "PhaseSpaceOut"
            )
            common, iin, iout = np.intersect1d(
                ein_event, eout_event, assume_unique=True, return_indices=True
            )
            if common.size < 100:
                raise RuntimeError(f"{case['case_id']}: too few paired primaries")
            ein, eout = ein[iin], eout[iout]
            loss = ein - eout
            median = float(np.median(loss))
            mad = float(1.4826 * np.median(np.abs(loss - median)))
            scale = max(mad, 1.0e-6)
            core = (
                np.isfinite(ein)
                & np.isfinite(eout)
                & (eout > 0.0)
                & (ein >= eout)
                & (np.abs(loss - median) <= float(config["robust_mad_cut"]) * scale)
            )
            ein, eout = ein[core], eout[core]
            raw_count = int(common.size)
            maximum = int(config["maximum_pairs_per_case"])
            if ein.size > maximum:
                rng = np.random.default_rng(
                    20260728 + int(case["case_index"])
                )
                selected = np.sort(rng.choice(ein.size, maximum, replace=False))
                ein, eout = ein[selected], eout[selected]
            np.savez_compressed(
                output,
                energy_in=ein.astype(np.float32),
                energy_out=eout.astype(np.float32),
                raw_count=np.int64(raw_count),
            )
            count = int(ein.size)
        rows.append({**case, "raw_paired": raw_count, "robust_sample": count})
        print(f"ingest {number:02d}/{len(cases)}: {case['case_id']} ({count:,})", flush=True)
    write_csv(QC_ROOT / "calibration_cases.csv", rows)
    result = {
        "status": "PASS",
        "cases": len(rows),
        "samples": sum(int(row["robust_sample"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
        "cache": str(cache_root),
    }
    write_json(QC_ROOT / "ingest_summary.json", result)
    return result


def load_samples(config: dict, cases: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = resolve(config["sample_cache"])
    samples = {}
    for case in cases:
        path = root / f"{case['case_id']}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing calibration cache: {path}; run --action ingest")
        with np.load(path) as values:
            samples[case["case_id"]] = (
                np.asarray(values["energy_in"], dtype=np.float64),
                np.asarray(values["energy_out"], dtype=np.float64),
            )
    return samples


def bb78_initial_derivative(knot_energy: np.ndarray) -> np.ndarray:
    model = load_wepl_model("bb78")
    # The historical BB78 model stores its full range LUT in ``range_mm`` but
    # only the two domain endpoints in ``energy_mev``.  It is therefore a
    # nearest-bin table, not an (xp, fp) pair suitable for np.interp.
    indices = np.floor(
        np.asarray(knot_energy, dtype=np.float64) / ENERGY_BIN_MEV + 0.5
    ).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= model.range_mm.size):
        raise ValueError("Stage-6B derivative knots exceed the BB78 LUT")
    ranges = model.range_mm[indices]
    derivative = np.gradient(ranges, knot_energy, edge_order=2)
    return np.log(np.maximum(derivative, 1.0e-6))


def case_metrics(cases, samples, parameters, knots, output_energy) -> list[dict]:
    rows = []
    for case in cases:
        ein, eout = samples[case["case_id"]]
        predicted = wepl(parameters, knots, output_energy, ein, eout)
        mean = float(np.mean(predicted))
        thickness = float(case["water_thickness_mm"])
        bias = mean - thickness
        rows.append(
            {
                **case,
                "predicted_wepl_mean_mm": mean,
                "predicted_wepl_std_mm": float(np.std(predicted)),
                "bias_mm": bias,
                "relative_bias": bias / thickness,
                "absolute_relative_bias": abs(bias / thickness),
                "samples": len(predicted),
            }
        )
    return rows


def fit(config: dict, force: bool) -> dict:
    model_path = QC_ROOT / "g4_water_calibrated.json"
    if model_path.is_file() and not force:
        return read_json(QC_ROOT / "fit_summary.json")
    sim_config = read_json(resolve(config["simulation_config"]))
    cases = enumerate_cases(sim_config)
    samples = load_samples(config, cases)
    knots = np.arange(
        0.0,
        float(config["energy_max_mev"]) + 0.5 * float(config["derivative_knot_step_mev"]),
        float(config["derivative_knot_step_mev"]),
    )
    output_energy = np.arange(
        0.0,
        float(config["energy_max_mev"]) + 0.5 * float(config["output_step_mev"]),
        float(config["output_step_mev"]),
    )
    initial = bb78_initial_derivative(knots)
    train = [case for case in cases if case["split"] == "train"]
    validation = [case for case in cases if case["split"] == "validation"]
    candidates = []
    fitted = {}
    for alpha in config["smoothness_candidates"]:
        def residual(parameters):
            physical = []
            for case in train:
                ein, eout = samples[case["case_id"]]
                mean = float(np.mean(wepl(parameters, knots, output_energy, ein, eout)))
                physical.append(
                    (mean - float(case["water_thickness_mm"]))
                    / float(case["water_thickness_mm"])
                )
            smooth = np.sqrt(float(alpha)) * np.diff(parameters, n=2)
            return np.r_[physical, smooth]

        result = least_squares(
            residual, initial, method="trf", max_nfev=300,
            xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10,
        )
        validation_rows = case_metrics(
            validation, samples, result.x, knots, output_energy
        )
        score = float(np.mean([row["absolute_relative_bias"] for row in validation_rows]))
        candidates.append(
            {
                "smoothness": alpha,
                "validation_mean_absolute_relative_bias": score,
                "validation_max_absolute_relative_bias": max(
                    row["absolute_relative_bias"] for row in validation_rows
                ),
                "optimizer_success": bool(result.success),
                "optimizer_cost": float(result.cost),
                "function_evaluations": int(result.nfev),
            }
        )
        fitted[float(alpha)] = result.x
        print(f"fit alpha={alpha:g}: validation MARE={score:.4%}", flush=True)
    passing = [row for row in candidates if row["optimizer_success"]]
    if not passing:
        raise RuntimeError("all monotone calibration fits failed")
    winner = min(
        passing,
        key=lambda row: (
            row["validation_mean_absolute_relative_bias"],
            row["smoothness"],
        ),
    )
    parameters = fitted[float(winner["smoothness"])]
    ranges = range_grid(parameters, knots, output_energy)
    payload = {
        "schema_version": 1,
        "model_name": config["model_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "physics_list": sim_config["physics_list"],
        "water_max_step_mm": sim_config["water_max_step_mm"],
        "fit_split": "train",
        "selection_split": "validation",
        "test_opened": False,
        "smoothness": winner["smoothness"],
        "energy_mev": [round(float(value), 10) for value in output_energy],
        "range_mm": [round(float(value), 10) for value in ranges],
    }
    write_model(model_path, payload)
    write_csv(QC_ROOT / "fit_candidates.csv", candidates)
    train_rows = case_metrics(train, samples, parameters, knots, output_energy)
    validation_rows = case_metrics(
        validation, samples, parameters, knots, output_energy
    )
    write_csv(QC_ROOT / "fit_train_metrics.csv", train_rows)
    write_csv(QC_ROOT / "fit_validation_metrics.csv", validation_rows)
    result = {
        "status": "FIT_FROZEN_TEST_CLOSED",
        "selected_smoothness": winner["smoothness"],
        "train_mean_absolute_relative_bias": float(
            np.mean([row["absolute_relative_bias"] for row in train_rows])
        ),
        "validation_mean_absolute_relative_bias": winner[
            "validation_mean_absolute_relative_bias"
        ],
        "model": str(model_path),
    }
    write_json(QC_ROOT / "fit_summary.json", result)
    return result


def verify(config: dict) -> dict:
    model_path = QC_ROOT / "g4_water_calibrated.json"
    payload = read_json(model_path)
    if payload.get("test_opened"):
        existing = QC_ROOT / "test_summary.json"
        if existing.is_file():
            return read_json(existing)
    sim_config = read_json(resolve(config["simulation_config"]))
    cases = enumerate_cases(sim_config)
    test_cases = [case for case in cases if case["split"] == "test"]
    samples = load_samples(config, test_cases)
    model = load_wepl_model("g4_water_calibrated", model_path)
    rows = []
    for case in test_cases:
        ein, eout = samples[case["case_id"]]
        predicted = energies_to_wepl_model(model, ein, eout)
        thickness = float(case["water_thickness_mm"])
        bias = float(np.mean(predicted)) - thickness
        rows.append(
            {
                **case,
                "predicted_wepl_mean_mm": float(np.mean(predicted)),
                "predicted_wepl_std_mm": float(np.std(predicted)),
                "bias_mm": bias,
                "relative_bias": bias / thickness,
                "absolute_relative_bias": abs(bias / thickness),
                "samples": len(predicted),
            }
        )
    mean_abs = float(np.mean([row["absolute_relative_bias"] for row in rows]))
    max_abs = float(max(row["absolute_relative_bias"] for row in rows))
    acceptance = config["acceptance"]
    passed = (
        mean_abs <= float(acceptance["test_mean_absolute_relative_bias"])
        and max_abs <= float(acceptance["test_max_absolute_relative_bias"])
    )
    write_csv(QC_ROOT / "fit_test_metrics.csv", rows)
    payload["test_opened"] = True
    payload["test_status"] = "PASS" if passed else "FAIL"
    write_model(model_path, payload)
    result = {
        "status": "PASS" if passed else "FAIL",
        "test_mean_absolute_relative_bias": mean_abs,
        "test_max_absolute_relative_bias": max_abs,
        "thresholds": acceptance,
        "model": str(model_path),
    }
    write_json(QC_ROOT / "test_summary.json", result)
    if not passed:
        raise SystemExit(
            "independent WEPL calibration failed locked-test acceptance; "
            "Stage 7/8 remain blocked"
        )
    return result


def direct_pairs_one(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    air_slope: float,
    accepted_mask_path: Path | None = None,
    split: dict | None = None,
    run_id: int | None = None,
) -> dict:
    from iterative_reconstruction.physics import subtract_external_air_wepl

    pairs = np.asarray(read_mhd(input_path), dtype=np.float32)
    if accepted_mask_path is not None:
        if split is None or run_id is None:
            raise ValueError("Stage-3 accepted pairs require split and run_id")
        accepted = read_packed_mask(
            accepted_mask_path, len(pairs), str(split["bit_order"])
        )
        train = partition_masks(len(pairs), run_id, split)["train"]
        pairs = pairs[accepted & train]
    model = load_wepl_model("g4_water_calibrated", model_path)
    converted = np.array(pairs, copy=True)
    values = energies_to_wepl_model(model, pairs[:, 4, 0], pairs[:, 4, 1])
    if air_slope > 0:
        values = subtract_external_air_wepl(pairs, values, 100.0, air_slope)
    converted[:, 4, 0] = 0.0
    converted[:, 4, 1] = values
    converted[:, 4, 2] = 0.0
    write_mhd(output_path, converted)
    return {
        "rows": len(converted),
        "wepl_min_mm": float(values.min()),
        "wepl_max_mm": float(values.max()),
        "wepl_mean_mm": float(values.mean()),
    }


def prepare_pairs(config: dict, datasets: list[str], runs: int, force: bool) -> dict:
    model_path = QC_ROOT / "g4_water_calibrated.json"
    test = read_json(QC_ROOT / "test_summary.json")
    if test["status"] != "PASS":
        raise RuntimeError("locked calibration test has not passed")
    results = {}
    for dataset in datasets:
        root = resolve(config["datasets"][dataset])
        source = root / "pairs_train"
        materialize_stage3 = not source.is_dir()
        if materialize_stage3:
            source = root / "pairs"
            stage3_config = read_json(
                HERE.parent / "stage3_robust_weighting/stage3_config.json"
            )
            split = stage3_config["split"]
        else:
            split = None
        output = root / "stage6b_calibrated" / "direct_wepl_pairs"
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for run_id in range(runs):
            src = source / f"pairs{run_id:04d}.mhd"
            dst = output / f"pairs{run_id:04d}.mhd"
            if dst.is_file() and not force:
                rows.append({"run_id": run_id, "status": "reused"})
                continue
            measured = direct_pairs_one(
                src, dst, model_path,
                float(config["air_wepl_slope_mm_per_mm"])
                if dataset in config["air_datasets"] else 0.0,
                (
                    root
                    / "stage3/filters/baseline_3sigma"
                    / f"accepted_mask_{run_id:04d}.bin"
                    if materialize_stage3 else None
                ),
                split,
                run_id,
            )
            rows.append({"run_id": run_id, "status": "written", **measured})
            if (run_id + 1) % 20 == 0 or run_id + 1 == runs:
                print(f"direct pairs {dataset}: {run_id+1:03d}/{runs}", flush=True)
        write_csv(QC_ROOT / f"direct_pairs_{dataset}.csv", rows)
        results[dataset] = {"runs": len(rows), "output": str(output)}
    write_json(QC_ROOT / "direct_pairs_summary.json", results)
    return results


def project(config: dict, datasets: list[str], runs: int, jobs: int, force: bool) -> dict:
    results = {}
    for dataset in datasets:
        root = resolve(config["datasets"][dataset])
        pairs = root / "stage6b_calibrated" / "direct_wepl_pairs"
        ddb_name = "stage6b_calibrated/projections_ddb"
        ddb = root / ddb_name
        if force and ddb.is_dir():
            shutil.rmtree(ddb)
        ddb.mkdir(parents=True, exist_ok=True)
        rows = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    projection.process_run,
                    run_id,
                    str(pairs),
                    str(root),
                    False,
                    ddb_name,
                ): run_id
                for run_id in range(runs)
                if force or not (ddb / f"proj{run_id:04d}.mhd").is_file()
            }
            reused = runs - len(futures)
            for future in as_completed(futures):
                rows.append(future.result())
                count = reused + len(rows)
                if count % 20 == 0 or count == runs:
                    print(f"DDB {dataset}: {count:03d}/{runs}", flush=True)
        missing = [
            run_id
            for run_id in range(runs)
            if not (ddb / f"proj{run_id:04d}.mhd").is_file()
        ]
        result = {
            "status": "PASS" if not missing else "FAIL",
            "dataset": dataset,
            "runs": runs - len(missing),
            "missing": missing[:20],
            "new_runs": len(rows),
            "ddb": str(ddb),
        }
        write_json(QC_ROOT / f"projection_{dataset}.json", result)
        if missing:
            raise RuntimeError(f"{dataset}: {len(missing)} DDB projections missing")
        results[dataset] = result
    return results


def analytic(config: dict, datasets: list[str], force: bool) -> dict:
    executable = REPO_ROOT / ".venv-gate/bin/pctfdk"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    results = {}
    for dataset in datasets:
        preprocessing = resolve(config["datasets"][dataset])
        reconstruction = resolve(config["reconstruction_data"][dataset])
        ddb = preprocessing / "stage6b_calibrated/projections_ddb"
        output = (
            reconstruction
            / "stage6b_calibrated/analytic/recon_ddb_nohann.mhd"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and not force:
            elapsed = 0.0
        else:
            if force:
                for old in (output, output.with_suffix(".raw")):
                    if old.exists():
                        old.unlink()
            command = [
                str(executable),
                "--lowmem",
                "--geometry", str(resolve(config["geometry"][dataset])),
                "--path", str(ddb),
                "--regexp", r"proj....\.mhd",
                "--output", str(output),
                "--dimension", "2100", "1", "2100",
                "--spacing", "0.1", "1", "0.1",
                "--hann", "0",
                "--verbose",
            ]
            started = time.perf_counter()
            process = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
            )
            elapsed = time.perf_counter() - started
            log = QC_ROOT / "logs" / f"analytic_{dataset}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(process.stdout or "", encoding="utf-8")
            if process.returncode:
                raise RuntimeError(
                    f"pctfdk failed for {dataset}; see {log}\n"
                    f"{(process.stdout or '')[-2000:]}"
                )
        image, _, _ = read_image_2d(output)
        finite = bool(np.isfinite(image).all())
        result = {
            "status": "PASS" if finite else "FAIL",
            "dataset": dataset,
            "elapsed_seconds": elapsed,
            "finite": finite,
            "output": str(output),
        }
        write_json(QC_ROOT / f"analytic_{dataset}.json", result)
        if not finite:
            raise RuntimeError(f"{dataset}: non-finite analytic image")
        results[dataset] = result
    return results


def water_core_mean(path: Path) -> tuple[float, float]:
    image, spacing, origin = read_image_2d(path)
    x = float(origin[0]) + np.arange(image.shape[1]) * float(spacing[0])
    z = float(origin[2]) + np.arange(image.shape[0]) * float(spacing[2])
    xx, zz = np.meshgrid(x, z)
    core = xx * xx + zz * zz <= 90.0**2
    return float(np.mean(image[core])), float(np.std(image[core]))


def reconstruct(config: dict, datasets: list[str], force: bool) -> dict:
    model = QC_ROOT / "g4_water_calibrated.json"
    runner = CODE_ROOT / "iterative_reconstruction/run_best_reconstruction.py"
    ordered = [name for name in ("s2", "s3", "s1", "s4", "s5") if name in datasets]
    results = {}
    gate_checked = False
    for dataset in ordered:
        if dataset in {"s1", "s4", "s5"} and not gate_checked:
            gate_path = QC_ROOT / "s2_s3_gate.json"
            if not gate_path.is_file() or read_json(gate_path)["status"] != "PASS":
                raise RuntimeError("S2/S3 calibrated water gate has not passed")
            gate_checked = True
        preprocessing = resolve(config["datasets"][dataset])
        reconstruction = resolve(config["reconstruction_data"][dataset])
        initial = (
            reconstruction
            / "stage6b_calibrated/analytic/recon_ddb_nohann.mhd"
        )
        output = reconstruction / "stage6b_calibrated/iterative"
        final = output / "recon/recon_iterative_gpu.mhd"
        if final.is_file() and not force:
            elapsed = 0.0
        else:
            command = [
                sys.executable, str(runner),
                "--run-name", f"stage6b_{dataset}",
                "--pairs-dir", str(
                    preprocessing
                    / "stage6b_calibrated/direct_wepl_pairs"
                ),
                "--initial-image", str(initial),
                "--output-dir", str(output),
                "--wepl-model", "g4_water_calibrated",
                "--wepl-calibration", str(model),
                # The direct-WEPL pairs were already corrected for external
                # Air during materialization; never subtract it twice.
                "--air-wepl-slope", "0",
                "--runs", "720",
                "--angle-step-deg", "0.5",
                "--device", "0",
            ]
            if force:
                command.append("--force")
            started = time.perf_counter()
            subprocess.run(command, check=True)
            elapsed = time.perf_counter() - started
        result = {
            "status": "PASS",
            "dataset": dataset,
            "elapsed_seconds": elapsed,
            "output": str(final),
        }
        if dataset in {"s2", "s3"}:
            mean, std = water_core_mean(final)
            result.update(water_mean=mean, water_std=std)
        write_json(QC_ROOT / f"iterative_{dataset}.json", result)
        results[dataset] = result
        if all(name in results for name in ("s2", "s3")):
            tolerance = float(
                config["acceptance"]["s2_water_mean_tolerance"]
            )
            gate = {
                "status": (
                    "PASS"
                    if abs(results["s2"]["water_mean"] - 1.0) <= tolerance
                    else "FAIL"
                ),
                "tolerance": tolerance,
                "s2_water_mean": results["s2"]["water_mean"],
                "s2_water_std": results["s2"]["water_std"],
                "s3_water_mean": results["s3"]["water_mean"],
                "s3_water_std": results["s3"]["water_std"],
            }
            write_json(QC_ROOT / "s2_s3_gate.json", gate)
            if gate["status"] != "PASS":
                raise SystemExit(
                    "S2 calibrated water mean failed 1.000±0.003; "
                    "S1/S4/S5 reconstruction is blocked"
                )
    return results


def report(config: dict) -> dict:
    stage4_root = HERE.parent / "stage4_iterative_optimization"
    stage6a_qc = HERE.parent / "stage6a_mlic_reference/qc"
    sys.path[:0] = [
        str(stage4_root),
        str(HERE.parent / "stage3_robust_weighting"),
    ]
    try:
        import run_stage4 as stage4
        stage4_config = stage4.load_config()
        records = {
            item["name"]: item
            for item in stage4.datasets_from("s1,s2,s3,s4,s5", stage4_config)
        }
        new_metrics = {}
        for name, record in records.items():
            path = (
                resolve(config["reconstruction_data"][name])
                / "stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd"
            )
            measured, details = stage4.scalar_metrics(
                record, stage4_config, path
            )
            new_metrics[name] = {"metrics": measured, "details": details}
    finally:
        del sys.path[:2]

    baseline_rows = []
    with (
        HERE.parent
        / "stage4_iterative_optimization/qc/confirmation_image_metrics.csv"
    ).open(encoding="utf-8-sig", newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    baseline = {
        row["dataset"]: row
        for row in baseline_rows
        if row["method"] == "candidate"
    }
    with (stage6a_qc / "mlic_reference_200mev.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        refs = {
            row["material"]: float(row["mlic_rsp_200mev"])
            for row in csv.DictReader(stream)
        }
    s4_detail = [
        row
        for row in new_metrics["s4"]["details"]
        if str(row.get("material")) != "Air"
    ]
    s4_errors = [
        abs(float(row["mean_rsp"]) - refs[str(row["material"])])
        / refs[str(row["material"])]
        for row in s4_detail
    ]
    s4_large = [
        error
        for error, row in zip(s4_errors, s4_detail)
        if float(row["radius_mm"]) > 3.0
    ]
    old_s4_summary = []
    with (stage6a_qc / "s4_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        old_s4_summary = list(csv.DictReader(stream))
    old_s4 = next(
        row for row in old_s4_summary if row["method"] == "stage4_epoch_05"
    )
    old_large = float(old_s4["mlic_large_insert_mape_percent"]) / 100.0
    new_large = float(np.mean(s4_large))

    s5 = new_metrics["s5"]["metrics"]
    old_s5_f50 = float(baseline["s5"]["fmtf50_mean_lp_per_mm"])
    old_s5_f10 = float(baseline["s5"]["fmtf10_mean_lp_per_mm"])
    new_s5_f50 = float(s5["fmtf50_mean_lp_per_mm"])
    new_s5_f10 = float(s5["fmtf10_mean_lp_per_mm"])
    s1 = new_metrics["s1"]["metrics"]
    aluminium = float(s1["insert_peak_mean"])
    rows = [
        {
            "dataset": "s1",
            "water_mean_old": baseline["s1"]["water_mean"],
            "water_mean_calibrated": s1["water_mean"],
            "aluminium_rsp_calibrated": aluminium,
            "aluminium_mlic_rsp": refs["Aluminium"],
            "aluminium_mlic_error_percent": 100.0
            * (aluminium - refs["Aluminium"])
            / refs["Aluminium"],
        },
        {
            "dataset": "s4",
            "mlic_large_insert_mape_old_percent": 100.0 * old_large,
            "mlic_large_insert_mape_calibrated_percent": 100.0 * new_large,
            "mlic_all_non_air_mape_calibrated_percent": 100.0
            * float(np.mean(s4_errors)),
            "mlic_max_ape_calibrated_percent": 100.0
            * float(np.max(s4_errors)),
        },
        {
            "dataset": "s5",
            "fmtf50_old_lp_per_mm": old_s5_f50,
            "fmtf50_calibrated_lp_per_mm": new_s5_f50,
            "fmtf10_old_lp_per_mm": old_s5_f10,
            "fmtf10_calibrated_lp_per_mm": new_s5_f10,
        },
    ]
    write_csv(QC_ROOT / "three_scene_metrics.csv", rows)
    acceptance = config["acceptance"]
    checks = {
        "s2_s3_gate": read_json(QC_ROOT / "s2_s3_gate.json")["status"]
        == "PASS",
        "s4_large_insert_mape_improves_50_percent": new_large
        <= old_large
        * (
            1.0
            - float(
                acceptance[
                    "s4_large_insert_mape_relative_improvement"
                ]
            )
        ),
        "s5_fmtf50_not_degraded_over_2_percent": new_s5_f50
        >= old_s5_f50
        * (1.0 - float(acceptance["s5_fmtf_max_relative_degradation"])),
        "s5_fmtf10_not_degraded_over_2_percent": new_s5_f10
        >= old_s5_f10
        * (1.0 - float(acceptance["s5_fmtf_max_relative_degradation"])),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": (
            "PROMOTE_G4_WATER_CALIBRATED"
            if all(checks.values())
            else "RETAIN_BB78_AND_DIAGNOSE"
        ),
        "checks": checks,
        "headline": {
            "s1_aluminium_mlic_error_percent": rows[0][
                "aluminium_mlic_error_percent"
            ],
            "s4_large_insert_mape_old_percent": 100.0 * old_large,
            "s4_large_insert_mape_calibrated_percent": 100.0 * new_large,
            "s5_fmtf50_relative_change_percent": 100.0
            * (new_s5_f50 / old_s5_f50 - 1.0),
            "s5_fmtf10_relative_change_percent": 100.0
            * (new_s5_f10 / old_s5_f10 - 1.0),
        },
    }
    write_json(QC_ROOT / "three_scene_comparison.json", result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(
            "calibrated model did not pass three-scene promotion rules"
        )
    return result


def status(config: dict) -> dict:
    files = {
        "ingest": QC_ROOT / "ingest_summary.json",
        "fit": QC_ROOT / "fit_summary.json",
        "locked_test": QC_ROOT / "test_summary.json",
        "direct_pairs": QC_ROOT / "direct_pairs_summary.json",
        "s2_s3_gate": QC_ROOT / "s2_s3_gate.json",
        "three_scene_comparison": QC_ROOT / "three_scene_comparison.json",
    }
    result = {
        name: read_json(path) if path.is_file() else {"status": "PENDING"}
        for name, path in files.items()
    }
    if result["locked_test"].get("status") == "PASS":
        result["fit"]["status"] = "FIT_FROZEN_TEST_PASSED"
    for stage in ("projection", "analytic", "iterative"):
        result[stage] = {
            dataset: (
                read_json(QC_ROOT / f"{stage}_{dataset}.json")
                if (QC_ROOT / f"{stage}_{dataset}.json").is_file()
                else {"status": "PENDING"}
            )
            for dataset in config["datasets"]
        }
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=[
            "ingest", "fit", "verify", "prepare-pairs", "project",
            "analytic", "reconstruct", "report", "all", "status"
        ],
        required=True,
    )
    parser.add_argument("--datasets", default="s2,s3,s1,s4,s5")
    parser.add_argument("--runs", type=int, default=720)
    parser.add_argument("--jobs", type=int, default=4, help="parallel DDB workers")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(CONFIG_PATH)
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    unknown = set(datasets) - set(config["datasets"])
    if unknown:
        raise SystemExit(f"unknown datasets: {sorted(unknown)}")
    if not 1 <= args.runs <= 720:
        raise SystemExit("--runs must be in [1,720]")
    QC_ROOT.mkdir(parents=True, exist_ok=True)
    if args.action == "status":
        status(config)
        return
    if args.action == "all":
        ingest(config, args.force)
        fit(config, args.force)
        verify(config)
        gate_datasets = [name for name in ("s2", "s3") if name in datasets]
        if set(gate_datasets) != {"s2", "s3"}:
            raise SystemExit("--action all requires s2 and s3 for the water gate")
        prepare_pairs(config, gate_datasets, args.runs, args.force)
        project(config, gate_datasets, args.runs, args.jobs, args.force)
        analytic(config, gate_datasets, args.force)
        reconstruct(config, gate_datasets, args.force)
        scenes = [name for name in ("s1", "s4", "s5") if name in datasets]
        if set(scenes) != {"s1", "s4", "s5"}:
            raise SystemExit(
                "--action all requires s1,s4,s5 after the S2/S3 gate"
            )
        prepare_pairs(config, scenes, args.runs, args.force)
        project(config, scenes, args.runs, args.jobs, args.force)
        analytic(config, scenes, args.force)
        reconstruct(config, scenes, args.force)
        report(config)
        status(config)
        return
    if args.action in ("ingest", "all"):
        ingest(config, args.force)
    if args.action in ("fit", "all"):
        fit(config, args.force)
    if args.action in ("verify", "all"):
        verify(config)
    if args.action in ("prepare-pairs", "all"):
        prepare_pairs(config, datasets, args.runs, args.force)
    if args.action in ("project", "all"):
        project(config, datasets, args.runs, args.jobs, args.force)
    if args.action in ("analytic", "all"):
        analytic(config, datasets, args.force)
    if args.action in ("reconstruct", "all"):
        reconstruct(config, datasets, args.force)
    if args.action in ("report", "all"):
        report(config)
    status(config)


if __name__ == "__main__":
    main()
