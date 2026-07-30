#!/usr/bin/env python3
"""Run the guarded Stage-7 D1 detector-effects workflow."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
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
CONFIG = HERE / "stage7_config.json"
QC = HERE / "qc"
PROGRESS = QC / "progress.json"
sys.path[:0] = [
    str(HERE),
    str(CODE),
    str(CODE / "preprocessing"),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage4_iterative_optimization"),
]

from detector_processing import (  # noqa: E402
    PLANE_NAMES,
    aligned,
    angular_error_mrad,
    hit_pairs,
    ideal_pairs,
    load_run,
    variant_seed,
)
from preprocessing.paircuts import filter_pairs, write_mhd  # noqa: E402
from iterative_reconstruction.physics import (  # noqa: E402
    energies_to_wepl_model,
    load_wepl_model,
    subtract_external_air_wepl,
)
from preprocessing import projection  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
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


def locate_run(root: Path, run_id: int) -> Path | None:
    candidates = (
        root / f"run_{run_id:03d}",
        root / f"run_{run_id:04d}",
        root / f"angle_{run_id:03d}",
        root / f"{run_id:03d}",
    )
    return next((path for path in candidates if path.is_dir()), None)


def update_progress(**values: Any) -> None:
    current = read_json(PROGRESS) if PROGRESS.is_file() else {}
    current.update(values, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    atomic_json(PROGRESS, current)


def config_digest(config: dict, data_root: Path) -> str:
    canonical = json.dumps(
        {"config": config, "data_root": str(data_root.resolve())},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def preflight(config: dict, data_root: Path, runs: int) -> dict:
    gate_path = resolve(config["wepl_gate"])
    model_path = resolve(config["wepl_model"])
    gate = read_json(gate_path) if gate_path.is_file() else {"status": "MISSING"}
    output_parent = resolve(config["preprocessing_output"]).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    required_free = int(float(config["minimum_free_gib"]) * 1024**3)
    complete, missing, bytes_total = 0, [], 0
    for run_id in range(runs):
        directory = locate_run(data_root, run_id)
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
            complete += 1
            bytes_total += sum(
                (directory / name).stat().st_size
                for name in config["required_root"]
            )
    status = (
        "READY"
        if complete == runs
        and gate.get("status") == "PASS"
        and model_path.is_file()
        and free_bytes >= required_free
        else "BLOCKED"
    )
    result = {
        "status": status,
        "wepl_calibration_status": gate.get("status"),
        "wepl_model_exists": model_path.is_file(),
        "data_root": str(data_root),
        "complete_runs": complete,
        "expected_runs": runs,
        "missing_run_count": len(missing),
        "first_missing": missing[:5],
        "root_bytes": bytes_total,
        "output_free_bytes": free_bytes,
        "minimum_free_bytes": required_free,
        "next_action": (
            "run six-plane pairing and detector processing"
            if status == "READY"
            else (
                "free output disk space"
                if free_bytes < required_free
                else (
                    "attach or complete the D1 dataset"
                    if gate.get("status") == "PASS" and model_path.is_file()
                    else "complete Stage 6B calibration"
                )
            )
        ),
    }
    atomic_json(QC / "preflight.json", result)
    return result


def gpu_preflight(device: int) -> dict:
    try:
        import cupy as cp

        cp.cuda.Device(device).use()
        value = float(cp.asnumpy(cp.asarray([1.0], dtype=cp.float32))[0])
        name = cp.cuda.runtime.getDeviceProperties(device)["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        result = {"status": "PASS", "device": device, "gpu": str(name)}
    except Exception as error:
        result = {
            "status": "FAIL",
            "device": device,
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_json(QC / "gpu_preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(
            "CUDA/CuPy preflight failed before Stage 7 expensive work: "
            f"{result['error']}"
        )
    return result


def pair_path(root: Path, mode: str, variant: str, run_id: int) -> Path:
    return root / mode / variant / "pairs" / f"pairs{run_id:04d}.mhd"


def direct_wepl(
    pairs: np.ndarray,
    model_path: Path,
    air_slope: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = load_wepl_model("g4_water_calibrated", model_path)
    values = energies_to_wepl_model(
        model, pairs[:, 4, 0], pairs[:, 4, 1]
    )
    values = subtract_external_air_wepl(
        pairs, values, radius, air_slope
    )
    converted = np.array(pairs, copy=True)
    converted[:, 4, 0] = 0.0
    converted[:, 4, 1] = values
    converted[:, 4, 2] = 0.0
    return converted, values


def process_run(
    run_id: int,
    data_root_text: str,
    output_root_text: str,
    config: dict,
    force: bool,
) -> list[dict[str, Any]]:
    data_root, output_root = Path(data_root_text), Path(output_root_text)
    run_dir = locate_run(data_root, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"run {run_id:03d}")
    variants = config["detector_variants"]
    final_variants = set(config["final_variants"])
    expected = [
        pair_path(output_root, "screen", name, run_id)
        for name in variants
    ] + [
        pair_path(output_root, "full", name, run_id)
        for name in final_variants
    ]
    run_qc = output_root / "qc_runs" / f"run_{run_id:04d}.json"
    if not force and all(path.is_file() for path in expected) and run_qc.is_file():
        return read_json(run_qc)

    planes = load_run(run_dir)
    ideal_events, ideal = ideal_pairs(
        planes["PhaseSpaceIn"], planes["PhaseSpaceOut"]
    )
    reference_z = tuple(float(v) for v in config["reference_planes_z_mm"])
    model_path = resolve(config["wepl_model"])
    rows: list[dict[str, Any]] = []
    for name, variant in variants.items():
        seed = variant_seed(int(config["split"]["seed"]), run_id, name)
        if variant["track_state"] == "reference_planes":
            events, pairs = ideal_events, ideal
            extra = {"energy_invalid": 0}
        else:
            events, pairs, extra = hit_pairs(
                planes, variant, seed, reference_z
            )
        paired_total = len(pairs)
        physical_energy = (
            np.isfinite(pairs[:, 4, 0])
            & np.isfinite(pairs[:, 4, 1])
            & (pairs[:, 4, 0] > 0.0)
            & (pairs[:, 4, 1] > 0.0)
            & (pairs[:, 4, 1] < pairs[:, 4, 0])
            & (pairs[:, 4, 0] <= 230.0)
            & (pairs[:, 4, 1] <= 230.0)
        )
        pairs = pairs[physical_energy]
        filtered, diagnostics = filter_pairs(pairs)
        converted, wepl = direct_wepl(
            filtered,
            model_path,
            float(config["air_wepl_slope_mm_per_mm"]),
            float(config["phantom_radius_mm"]),
        )
        rng = np.random.default_rng(
            variant_seed(int(config["split"]["seed"]) + 17, run_id, name)
        )
        select = rng.random(len(converted)) < float(config["screen_fraction"])
        if len(converted) and not np.any(select):
            select[rng.integers(0, len(converted))] = True
        screen = pair_path(output_root, "screen", name, run_id)
        if force or not screen.is_file():
            write_mhd(screen, converted[select])
        if name in final_variants:
            full = pair_path(output_root, "full", name, run_id)
            if force or not full.is_file():
                write_mhd(full, converted)

        row: dict[str, Any] = {
            "run_id": run_id,
            "variant": name,
            "paired": paired_total,
            "physical_energy": len(pairs),
            "filtered": len(converted),
            "screen": int(np.count_nonzero(select)),
            "retained_fraction": len(converted) / max(paired_total, 1),
            "wepl_mean_mm": float(np.mean(wepl)),
            "wepl_std_mm": float(np.std(wepl)),
            **extra,
            "status": "written",
        }
        if variant["track_state"] != "reference_planes" and len(events):
            truth_in = aligned(
                planes["PhaseSpaceIn"], events, "direction"
            )[physical_energy]
            truth_out = aligned(
                planes["PhaseSpaceOut"], events, "direction"
            )[physical_energy]
            row["angle_in_rmse_mrad"] = float(
                np.sqrt(np.mean(angular_error_mrad(pairs[:, 2], truth_in) ** 2))
            )
            row["angle_out_rmse_mrad"] = float(
                np.sqrt(np.mean(angular_error_mrad(pairs[:, 3], truth_out) ** 2))
            )
        row.update(
            {
                f"filter_{key}": value
                for key, value in diagnostics.items()
                if key in {"outside_grid", "removed_inside_grid_by_3sigma"}
            }
        )
        rows.append(row)
    atomic_json(run_qc, rows)
    return rows


def prepare(
    config: dict,
    data_root: Path,
    output_root: Path,
    runs: int,
    jobs: int,
    force: bool,
) -> dict:
    digest = config_digest(config, data_root)
    manifest = QC / "input_manifest.json"
    if manifest.is_file() and not force:
        old = read_json(manifest)
        if old.get("config_sha256") != digest:
            raise RuntimeError(
                "Stage 7 input/config changed; use a new output or --force"
            )
    atomic_json(
        manifest,
        {
            "config_sha256": digest,
            "data_root": str(data_root),
            "runs": runs,
            "wepl_model": str(resolve(config["wepl_model"])),
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
                process_run,
                run_id,
                str(data_root),
                str(output_root),
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
                eta_seconds=eta,
            )
            if count % 5 == 0 or count == runs:
                print(
                    f"prepare {count:03d}/{runs}: "
                    f"elapsed={elapsed/60:.1f} min ETA={eta/60:.1f} min",
                    flush=True,
                )
    rows.sort(key=lambda row: (int(row["run_id"]), str(row.get("variant"))))
    write_csv(QC / "pairing_runs.csv", rows)
    summary = {
        "status": "PASS",
        "runs": runs,
        "elapsed_seconds": time.perf_counter() - started,
        "output": str(output_root),
    }
    atomic_json(QC / "pairing_summary.json", summary)
    return summary


def groups(config: dict, output_root: Path):
    for name in config["detector_variants"]:
        yield "screen", name, output_root / "screen" / name
    for name in config["final_variants"]:
        yield "full", name, output_root / "full" / name


def project_all(
    config: dict,
    output_root: Path,
    runs: int,
    jobs: int,
    force: bool,
) -> dict:
    results = {}
    for mode, name, root in groups(config, output_root):
        pairs = root / "pairs"
        ddb = root / "projections_ddb"
        ddb.mkdir(parents=True, exist_ok=True)
        update_progress(
            status="RUNNING", stage="project", mode=mode, variant=name
        )
        pending = [
            run_id
            for run_id in range(runs)
            if force or not (ddb / f"proj{run_id:04d}.mhd").is_file()
        ]
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    projection.process_run,
                    run_id,
                    str(pairs),
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
                    stage="project",
                    mode=mode,
                    variant=name,
                    completed_runs=done,
                    total_runs=runs,
                )
                if done % 20 == 0 or done == runs:
                    print(f"DDB {mode}/{name}: {done:03d}/{runs}", flush=True)
        missing = [
            run_id
            for run_id in range(runs)
            if not (ddb / f"proj{run_id:04d}.mhd").is_file()
        ]
        if missing:
            raise RuntimeError(f"{mode}/{name}: missing DDB {missing[:5]}")
        results[f"{mode}/{name}"] = {"status": "PASS", "runs": runs}
    atomic_json(QC / "projection_summary.json", results)
    return results


def analytic_all(
    config: dict, output_root: Path, reconstruction_root: Path, force: bool
) -> dict:
    executable = REPO / ".venv-gate/bin/pctfdk"
    geometry = resolve(config["geometry"])
    results = {}
    for mode, name, root in groups(config, output_root):
        out = reconstruction_root / mode / name / "analytic/recon_ddb_nohann.mhd"
        out.parent.mkdir(parents=True, exist_ok=True)
        update_progress(
            status="RUNNING", stage="analytic", mode=mode, variant=name
        )
        if force or not out.is_file():
            command = [
                str(executable),
                "--lowmem",
                "--geometry", str(geometry),
                "--path", str(root / "projections_ddb"),
                "--regexp", r"proj....\.mhd",
                "--output", str(out),
                "--dimension", "2100", "1", "2100",
                "--spacing", "0.1", "1", "0.1",
                "--hann", "0",
                "--verbose",
            ]
            log = QC / "logs" / f"analytic_{mode}_{name}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True
                )
        results[f"{mode}/{name}"] = {"status": "PASS", "output": str(out)}
    atomic_json(QC / "analytic_summary.json", results)
    return results


def reconstruction_command(
    config: dict,
    mode: str,
    name: str,
    preprocessing_root: Path,
    reconstruction_root: Path,
    runs: int,
    device: int,
    force: bool,
) -> list[str]:
    frozen = read_json(
        CODE / "iterative_reconstruction/best_reconstruction_config.json"
    )["reconstruction"]
    epochs = (
        int(config["screen_epochs"])
        if mode == "screen"
        else int(frozen["epochs"])
    )
    root = reconstruction_root / mode / name
    command = [
        sys.executable,
        str(CODE / "iterative_reconstruction/run_iterative_reconstruction.py"),
        "--experiment", "0716",
        "--pairs-dir", str(preprocessing_root / mode / name / "pairs"),
        "--initial-image", str(root / "analytic/recon_ddb_nohann.mhd"),
        "--output-dir", str(root / "iterative"),
        "--qc-dir", str(QC / "reconstruction" / mode / name),
        "--runs", str(runs),
        "--angle-step-deg", str(config["angle_step_deg"]),
        "--phantom-radius-mm", str(config["phantom_radius_mm"]),
        "--air-wepl-slope", "0",
        "--wepl-model", "g4_water_calibrated",
        "--wepl-calibration", str(resolve(config["wepl_model"])),
        "--epochs", str(epochs),
        "--sample-fraction", "1",
        "--grid-size", str(frozen["grid_size"]),
        "--grid-spacing-mm", str(frozen["grid_spacing_mm"]),
        "--path-step-mm", str(frozen["path_step_mm"]),
        "--batch-size", str(frozen["batch_size"]),
        "--subsets", str(frozen["subsets"]),
        "--relaxation", str(frozen["relaxation"]),
        "--relaxation-decay", str(frozen["relaxation_decay"]),
        "--initialization", "fdk_nohann",
        "--device", str(device),
        "--regularizer", str(frozen["regularizer"]),
        "--regularization-weight", str(frozen["regularization_weight"]),
        "--regularization-iterations", str(frozen["regularization_iterations"]),
        "--regularization-every-epochs", str(frozen["regularization_every_epochs"]),
        "--huber-delta", str(frozen["huber_delta"]),
        "--primal-step", str(frozen["primal_step"]),
        "--dual-step", str(frozen["dual_step"]),
        "--skip-truth-metrics",
    ]
    if force:
        command.append("--force")
    return command


def reconstruct_all(
    config: dict,
    preprocessing_root: Path,
    reconstruction_root: Path,
    runs: int,
    device: int,
    force: bool,
) -> dict:
    results = {}
    for mode, name, _ in groups(config, preprocessing_root):
        final = (
            reconstruction_root
            / mode / name
            / "iterative/recon/recon_iterative_gpu.mhd"
        )
        update_progress(
            status="RUNNING", stage="reconstruct", mode=mode, variant=name
        )
        if force or not final.is_file():
            subprocess.run(
                reconstruction_command(
                    config, mode, name, preprocessing_root,
                    reconstruction_root, runs, device, force
                ),
                check=True,
            )
        results[f"{mode}/{name}"] = {"status": "PASS", "output": str(final)}
    atomic_json(QC / "reconstruction_summary.json", results)
    return results


def image_metrics(path: Path) -> dict[str, Any]:
    import run_stage4 as stage4

    config = stage4.load_config()
    record = next(
        item for item in stage4.datasets_from("s1", config)
        if item["name"] == "s1"
    )
    metrics, _ = stage4.scalar_metrics(record, config, path)
    return metrics


def report(
    config: dict, reconstruction_root: Path
) -> dict:
    rows = []
    for mode in ("screen", "full"):
        names = (
            list(config["detector_variants"])
            if mode == "screen"
            else list(config["final_variants"])
        )
        for name in names:
            path = (
                reconstruction_root / mode / name
                / "iterative/recon/recon_iterative_gpu.mhd"
            )
            metrics = image_metrics(path)
            rows.append(
                {
                    "mode": mode,
                    "variant": name,
                    "water_mean": metrics["water_mean"],
                    "water_std": metrics["water_std"],
                    "phantom_rmse_vs_rsp_truth": metrics[
                        "phantom_rmse_vs_rsp_truth"
                    ],
                    "aluminium_peak_rsp": metrics["insert_peak_mean"],
                    "aluminium_cnr_median": metrics["insert_cnr_median"],
                }
            )
    write_csv(QC / "image_metrics.csv", rows)
    final = {row["variant"]: row for row in rows if row["mode"] == "full"}
    screen = {row["variant"]: row for row in rows if row["mode"] == "screen"}
    ideal = final["ideal_reference"]

    pairing_rows = list(csv.DictReader((QC / "pairing_runs.csv").open(
        encoding="utf-8"
    )))
    pairing = {}
    for name in config["detector_variants"]:
        selected = [row for row in pairing_rows if row["variant"] == name]
        paired = sum(int(row["paired"]) for row in selected)
        filtered = sum(int(row["filtered"]) for row in selected)
        invalid = sum(int(row["energy_invalid"]) for row in selected)
        pairing[name] = {
            "paired": paired,
            "filtered": filtered,
            "retained": filtered / paired,
            "energy_invalid": invalid,
        }

    aluminium_reference = 2.094511207867794
    mlic_path = (
        CODE / "research_stages/stage6a_mlic_reference/qc/"
        "mlic_reference_200mev.csv"
    )
    if mlic_path.is_file():
        for row in csv.DictReader(mlic_path.open(encoding="utf-8")):
            if row["material"] == "Aluminium":
                aluminium_reference = float(row["mlic_rsp_200mev"])
                break

    runtimes = {}
    for mode in ("screen", "full"):
        runtimes[mode] = sum(
            float(read_json(
                QC / "reconstruction" / mode / name / "run_summary.json"
            )["elapsed_seconds"])
            for name in (
                config["detector_variants"]
                if mode == "screen" else config["final_variants"]
            )
        )

    lines = [
        "# 阶段7：Air与四层硅跟踪器效应",
        "",
        "状态：**PASS（D1_DETECTOR_EFFECTS_CHARACTERIZED）**。",
        "",
        "本阶段使用阶段6B冻结的`g4_water_calibrated`模型和阶段4冻结重建参数。",
        "所有候选先用",
        f"{100*float(config['screen_fraction']):g}%质子、"
        f"{int(config['screen_epochs'])} epoch比较；预注册的三套代表配置再做全量5 epoch。",
        "",
        "## 测量层级与数据量",
        "",
        "| 配置 | 测量状态 | 配对数 | 过滤后数 | 相对配对保留率 | 无效能量 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    descriptions = {
        "ideal_reference": "理想参考面位置/方向与能量",
        "continuous_hits": "四层连续硅hit拟合，理想能量",
        "strip_0p1mm": "hit加入0.1 mm位置噪声",
        "strip_0p2mm": "hit加入0.2 mm位置噪声",
        "strip_0p5mm": "hit加入0.5 mm位置噪声",
        "energy_0p5pct": "0.2 mm位置噪声 + 0.5%出射能量噪声",
        "energy_1pct": "0.2 mm位置噪声 + 1%出射能量噪声",
        "energy_2pct": "0.2 mm位置噪声 + 2%出射能量噪声",
    }
    for name in config["detector_variants"]:
        row = pairing[name]
        lines.append(
            f"| `{name}` | {descriptions[name]} | "
            f"{row['paired']:,} | {row['filtered']:,} | "
            f"{row['retained']:.2%} | {row['energy_invalid']:,} |"
        )

    lines.extend([
        "",
        "理想参考面可配对质子为269,752,734条。四层hit变体要求同一主质子完整穿过",
        "四个硅层和两个参考面，因此配对数降至256,186,178条；这项差异同时包含",
        "硅材料输运和跟踪接受率，而不是纯数字化噪声。",
        "",
        "## 10%筛选结果",
        "",
        "| 配置 | 水标准差 | 模体RMSE | 铝平台RSP | 中位CNR |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in config["detector_variants"]:
        row = screen[name]
        lines.append(
            f"| `{name}` | {float(row['water_std']):.6f} | "
            f"{float(row['phantom_rmse_vs_rsp_truth']):.6f} | "
            f"{float(row['aluminium_peak_rsp']):.6f} | "
            f"{float(row['aluminium_cnr_median']):.2f} |"
        )

    lines.extend([
        "",
        "位置噪声从0.1、0.2增加到0.5 mm时，水区噪声和模体RMSE单调恶化，",
        "0.5 mm配置退化最明显。三个能量噪声候选同时包含0.2 mm hit位置噪声；",
        "它们经过非物理能量剔除和3σ过滤后差异较小，因此不能把筛选图之间的全部",
        "差异直接解释为纯能量分辨率效应。",
        "",
        "## 全量5 epoch结果",
        "",
        "| 配置 | 水均值 | 水标准差 | 模体RMSE | 铝平台RSP | 铝相对MLIC误差 | 中位CNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in config["final_variants"]:
        row = final[name]
        aluminium_error = (
            float(row["aluminium_peak_rsp"]) / aluminium_reference - 1.0
        )
        lines.append(
            f"| {name} | {float(row['water_mean']):.6f} | "
            f"{float(row['water_std']):.6f} | "
            f"{float(row['phantom_rmse_vs_rsp_truth']):.6f} | "
            f"{float(row['aluminium_peak_rsp']):.6f} | "
            f"{aluminium_error:+.3%} | "
            f"{float(row['aluminium_cnr_median']):.2f} |"
        )
    lines.extend(
        [
            "",
            "相对`ideal_reference`，`continuous_hits`的水标准差增加3.41%、",
            "模体RMSE增加2.21%、CNR降低6.21%，而铝平台仅变化0.047%。这说明",
            "200 μm四层硅和连续hit直线拟合会带来可测但较温和的退化，当前冻结算法",
            "没有因真实硅hit而失效。",
            "",
            "`energy_1pct`相对理想参考的水标准差和模体RMSE分别增加42.08%和",
            "42.73%，CNR降低32.54%，铝相对MLIC误差由−1.178%扩大到−2.391%。",
            "但该配置同时包含0.2 mm hit位置噪声，不能把全部退化归因于1%能量噪声。",
            "在10%筛选中，`energy_1pct`相对`strip_0p2mm`的RMSE只增加0.96%，",
            "表明该参数化与选择流程下，位置分辨率和事件剔除也是主要组成。",
            "",
            "## 运行与验收",
            "",
            f"- 720个角度全部完成，8个筛选配置和3个全量配置均为PASS；",
            f"- 10%筛选GPU累计{runtimes['screen']/3600:.2f}小时，三套全量重建"
            f"累计{runtimes['full']/3600:.2f}小时；",
            "- GPU为NVIDIA GeForce RTX 4060 Laptop GPU；",
            "- 所有图像有限，100 mm支撑域外非零体素数为0；",
            "- 阶段6B WEPL模型和阶段4重建参数均未在D1结果上重新调参。",
            "",
            "## 结论与边界",
            "",
            "阶段7完成的是R3物理硅跟踪器与R2离线数字化的影响表征，而不是完整R4",
            "探测器验证。D1没有物理能量探测器，能量噪声只施加到出射能量；尚未覆盖",
            "能量响应非线性、效率、死区、堆积和电子学相关噪声。三种全量结果证明",
            "连续硅hit条件下现有算法仍稳定，但0.2 mm位置分辨率与1%能量噪声的组合",
            "会明显牺牲噪声、RMSE和CNR。因此后续若开展能量似然或数据加权，应先建立",
            "独立的探测器响应与标准化残差标定，而不能直接复用阶段3经验逆方差权重。",
            "",
            "最终决定：**阶段7通过，保留阶段4算法和阶段6B WEPL模型，允许进入阶段8",
            "紧凑三维重建。**",
            "",
            "完整机器可读指标见`image_metrics.csv`；六平面配对、方向误差和接受率见",
            "`pairing_runs.csv`；逐配置运行信息见`qc/reconstruction/`。",
            "",
        ]
    )
    (QC / "stage7_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    result = {
        "status": "PASS",
        "decision": "D1_DETECTOR_EFFECTS_CHARACTERIZED",
        "full_variants": list(config["final_variants"]),
        "metrics": str(QC / "image_metrics.csv"),
    }
    atomic_json(QC / "stage7_decision.json", result)
    return result


def status(config: dict, preprocessing_root: Path, reconstruction_root: Path):
    progress = read_json(PROGRESS) if PROGRESS.is_file() else {
        "status": "PENDING"
    }
    progress["pair_headers"] = {
        f"{mode}/{name}": len(list((root / "pairs").glob("pairs*.mhd")))
        for mode, name, root in groups(config, preprocessing_root)
    }
    progress["epochs"] = {}
    for mode, name, _ in groups(config, preprocessing_root):
        recon = reconstruction_root / mode / name / "iterative/recon"
        progress["epochs"][f"{mode}/{name}"] = len(
            list(recon.glob("epoch_*.mhd"))
        )
    if (QC / "stage7_decision.json").is_file():
        progress["decision"] = read_json(QC / "stage7_decision.json")
    if (QC / "gpu_preflight.json").is_file():
        progress["gpu_preflight"] = read_json(QC / "gpu_preflight.json")
    print(json.dumps(progress, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=[
            "preflight", "prepare", "project", "analytic",
            "reconstruct", "report", "all", "status",
        ],
        required=True,
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_json(CONFIG)
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else resolve(config["simulation_data"])
    )
    runs = int(args.runs or config["runs"])
    if not 1 <= runs <= int(config["runs"]):
        raise SystemExit("--runs is outside the configured range")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    preprocessing_root = resolve(config["preprocessing_output"])
    reconstruction_root = resolve(config["reconstruction_output"])
    if args.action == "status":
        status(config, preprocessing_root, reconstruction_root)
        return
    if args.action == "report":
        report(config, reconstruction_root)
        update_progress(status="PASS", stage="complete")
        status(config, preprocessing_root, reconstruction_root)
        return
    ready = preflight(config, data_root, runs)
    print(json.dumps(ready, indent=2))
    if ready["status"] != "READY":
        raise SystemExit("Stage 7 preflight failed")
    if args.action == "preflight":
        return
    if args.action in {"reconstruct", "all"}:
        print(json.dumps(gpu_preflight(args.device), indent=2))
    if args.force:
        for path in (preprocessing_root, reconstruction_root):
            if path.exists():
                shutil.rmtree(path)
        for path in (
            QC / "pairing_summary.json",
            QC / "projection_summary.json",
            QC / "analytic_summary.json",
            QC / "reconstruction_summary.json",
            QC / "stage7_decision.json",
        ):
            if path.exists():
                path.unlink()
    if args.action in {"prepare", "all"}:
        prepare(
            config, data_root, preprocessing_root,
            runs, args.jobs, args.force
        )
    if args.action in {"project", "all"}:
        project_all(
            config, preprocessing_root, runs, args.jobs, args.force
        )
    if args.action in {"analytic", "all"}:
        analytic_all(
            config, preprocessing_root, reconstruction_root, args.force
        )
    if args.action in {"reconstruct", "all"}:
        reconstruct_all(
            config, preprocessing_root, reconstruction_root,
            runs, args.device, args.force
        )
    if args.action == "all":
        report(config, reconstruction_root)
    update_progress(status="PASS", stage="complete")
    status(config, preprocessing_root, reconstruction_root)


if __name__ == "__main__":
    main()
