#!/usr/bin/env python3
"""Compare 360 angles x 20% with the equal-total-fluence 720 x 10% baseline.

This is deliberately a separate entry point: the completed Stage-8B task-1
source/config hash is left untouched.  No reconstruction parameter is tuned.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
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
QC = HERE / "qc/angular_fluence_360x20"
PROGRESS = QC / "progress.json"
PRE = REPO / "data/preprocessing_data/results0718_d1_air_tracker_full/stage8b/angular_fluence/angles360_f0200"
RECON = REPO / "data/reconstruction_data/results0718_d1_air_tracker_full/stage8b/angular_fluence/angles360_f0200"
STAGE7C_PRE = REPO / "data/preprocessing_data/results0718_d1_air_tracker_full/stage7c"
STAGE7C_RECON = REPO / "data/reconstruction_data/results0718_d1_air_tracker_full/stage7c"
SEED, FRACTION, RUNS, ANGLE_STEP_DEG = 20260730, 0.20, 360, 1.0

sys.path[:0] = [
    str(HERE), str(CODE), str(CODE / "preprocessing"),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage7b_noise_robustness"),
]

from stage8b_data import nested_mask  # noqa: E402
from preprocessing import paircuts, projection  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(**values: Any) -> None:
    current = read_json(PROGRESS) if PROGRESS.is_file() else {}
    current.update(values)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_json(PROGRESS, current)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "original_runs": list(range(0, 720, 2)),
        "runs": RUNS,
        "angle_step_deg": ANGLE_STEP_DEG,
        "per_angle_fraction": FRACTION,
        "nominal_total_fraction": RUNS * FRACTION / 720.0,
    }


def parent_path(original_run: int) -> Path:
    return STAGE7C_PRE / "combined_0p2mm_1pct" / f"seed_{SEED}" / "f025/pairs" / f"pairs{original_run:04d}.mhd"


def events_path(original_run: int) -> Path:
    return STAGE7C_PRE / "event_ids/combined_0p2mm_1pct" / f"events{original_run:04d}.npy"


def preflight(force: bool, device: int) -> None:
    missing = [str(path) for original in range(0, 720, 2) for path in (parent_path(original), events_path(original)) if not path.is_file()]
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        cp.asarray([1], dtype=cp.float32).sum().get()
        gpu = cp.cuda.runtime.getDeviceProperties(device)["name"]
        if isinstance(gpu, bytes):
            gpu = gpu.decode()
    except Exception as error:
        raise RuntimeError(f"CUDA preflight failed: {error}") from error
    # The angular-fluence branch does not exist before its first run.  Create
    # only its empty parent so disk_usage can inspect the correct filesystem.
    PRE.parent.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(PRE.parent).free / 1024**3
    result = {
        "status": "PASS" if not missing and free_gib >= 10 else "FAIL",
        "missing_count": len(missing), "first_missing": missing[0] if missing else None,
        "free_gib": free_gib, "required_free_gib": 10, "gpu": str(gpu),
    }
    atomic_json(QC / "preflight.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"angular-fluence preflight failed: {result}")
    path = QC / "input_manifest.json"
    value = manifest()
    if path.is_file() and read_json(path) != value and not force:
        raise RuntimeError("angular-fluence source/config changed; use --force")
    atomic_json(path, value)


def prepare_one(output_run: int, force: bool) -> dict[str, Any]:
    original_run = 2 * output_run
    destination = PRE / "pairs" / f"pairs{output_run:04d}.mhd"
    if destination.is_file() and not force:
        return {"output_run": output_run, "original_run": original_run, "angle_deg": float(output_run), "selected": len(paircuts.read_mhd(destination)), "status": "reused"}
    events = np.load(events_path(original_run), mmap_mode="r")
    parent_mask = nested_mask(events, original_run, SEED, 0.25)
    target_mask = nested_mask(events, original_run, SEED, FRACTION)
    if np.any(target_mask & ~parent_mask):
        raise RuntimeError(f"non-nested mask for original run {original_run:04d}")
    parent = paircuts.read_mhd(parent_path(original_run))
    if len(parent) != int(np.count_nonzero(parent_mask)):
        raise RuntimeError(f"EventID alignment failed for original run {original_run:04d}")
    selected = np.asarray(parent[target_mask[parent_mask]], np.float32)
    if not len(selected) or not np.isfinite(selected).all():
        raise RuntimeError(f"invalid subset for original run {original_run:04d}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    paircuts.write_mhd(destination, np.ascontiguousarray(selected))
    return {"output_run": output_run, "original_run": original_run, "angle_deg": float(output_run), "selected": len(selected), "status": "written"}


def prepare(jobs: int, force: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(prepare_one, run, force) for run in range(RUNS)]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (RUNS - done)
            progress(status="RUNNING", stage="prepare", completed_runs=done, total_runs=RUNS, task_eta_seconds=eta)
            if done % 20 == 0 or done == RUNS:
                print(f"prepare: {done}/{RUNS}, ETA={eta/60:.1f} min", flush=True)
    rows.sort(key=lambda row: int(row["output_run"]))
    write_csv(QC / "angle_mapping_and_counts.csv", rows)
    return rows


def ddb(jobs: int, force: bool) -> None:
    directory = PRE / "projections_ddb"
    directory.mkdir(parents=True, exist_ok=True)
    pending = [run for run in range(RUNS) if force or not (directory / f"proj{run:04d}.mhd").is_file()]
    if not pending:
        return
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(projection.process_run, run, str(PRE / "pairs"), str(PRE), False, "projections_ddb") for run in pending]
        base = RUNS - len(pending)
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            done = base + count
            elapsed = time.perf_counter() - started
            eta = elapsed / count * (len(pending) - count)
            progress(stage="ddb", completed_runs=done, total_runs=RUNS, task_eta_seconds=eta)
            if done % 20 == 0 or done == RUNS:
                print(f"DDB: {done}/{RUNS}, ETA={eta/60:.1f} min", flush=True)


def geometry(force: bool) -> Path:
    path = QC / "geometry_360.xml"
    if path.is_file() and not force:
        return path
    acquisition = read_json(CODE / "experiments/experiment0716.json")["acquisition"]
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(REPO / ".venv-gate/bin/rtksimulatedgeometry"), "--nproj", str(RUNS),
        "--first_angle", "0", "--arc", "360",
        "--sid", str(acquisition["source_to_isocenter_mm"]),
        "--sdd", str(acquisition["source_to_detector_mm"]), "--output", str(path),
    ], check=True)
    return path


def analytic(force: bool) -> Path:
    output = RECON / "analytic/recon_ddb_nohann.mhd"
    if output.is_file() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    progress(stage="analytic", task_eta_seconds=None)
    command = [
        str(REPO / ".venv-gate/bin/pctfdk"), "--lowmem", "--geometry", str(geometry(force)),
        "--path", str(PRE / "projections_ddb"), "--regexp", r"proj....\.mhd",
        "--output", str(output), "--dimension", "2100", "1", "2100",
        "--spacing", "0.1", "1", "0.1", "--hann", "0", "--verbose",
    ]
    log = QC / "logs/fdk.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return output


def run_logged(command: list[str], log: Path) -> None:
    epoch_pattern = re.compile(r"epoch\s+(\d+)/(\d+).*subset\s+(\d+)/(\d+)")
    rate_pattern = re.compile(r"rate=([\d,]+)\s+pairs/s\s+ETA=([0-9:]+)")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
            match = epoch_pattern.search(line)
            if match:
                progress(stage="reconstruct", epoch=int(match.group(1)), total_epochs=int(match.group(2)), subset=int(match.group(3)), total_subsets=int(match.group(4)))
            rate = rate_pattern.search(line)
            if rate:
                eta = 0
                for value in (int(part) for part in rate.group(2).split(":")):
                    eta = 60 * eta + value
                progress(pairs_per_second=int(rate.group(1).replace(",", "")), task_eta_seconds=eta)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def iterative(initial: Path, device: int, force: bool) -> Path:
    settings = read_json(CODE / "iterative_reconstruction/best_reconstruction_config.json")["reconstruction"]
    final = RECON / "iterative/recon/epoch_05.mhd"
    if final.is_file() and not force:
        return final
    calibration = CODE / "research_stages/stage6b_wepl_calibration/qc/g4_water_calibrated.json"
    command = [
        sys.executable, str(CODE / "iterative_reconstruction/run_iterative_reconstruction.py"),
        "--experiment", "0716", "--pairs-dir", str(PRE / "pairs"), "--initial-image", str(initial),
        "--output-dir", str(RECON / "iterative"), "--qc-dir", str(QC / "reconstruction"),
        "--runs", str(RUNS), "--angle-step-deg", str(ANGLE_STEP_DEG), "--phantom-radius-mm", "100",
        "--air-wepl-slope", "0", "--wepl-model", "g4_water_calibrated", "--wepl-calibration", str(calibration),
        "--epochs", str(settings["epochs"]), "--sample-fraction", "1",
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
    if force:
        command.append("--force")
    run_logged(command, QC / "logs/iterative.log")
    return final


def numeric_metrics(path: Path) -> dict[str, Any]:
    import run_stage7b
    return {key: value for key, value in run_stage7b.image_metrics(path).items() if isinstance(value, (int, float, bool, np.number))}


def report(result: Path, selected_total: int) -> None:
    reference = STAGE7C_RECON / "combined_0p2mm_1pct/seed_20260730/f010/iterative/recon/epoch_05.mhd"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    rows = []
    for name, angles, fraction, path, count in [
        ("720_angles_x_10pct", 720, 0.10, reference, ""),
        ("360_angles_x_20pct", 360, 0.20, result, selected_total),
    ]:
        rows.append({"configuration": name, "angles": angles, "per_angle_fraction": fraction, "nominal_total_fraction": angles * fraction / 720.0, "selected_pairs": count, "image_path": str(path), **numeric_metrics(path)})
    write_csv(QC / "comparison_metrics.csv", rows)
    old, new = rows
    lines = [
        "# Stage 8B等剂量角度—通量对照", "",
        "两组数据采用相同组合噪声、随机种子和冻结重建参数；名义总质子数相同。", "",
        "| 配置 | 水区标准差 | 模体RSP RMSE | 铝柱CNR | 边缘宽度/mm |", "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['configuration']} | {float(row['water_std']):.6f} | {float(row['phantom_rmse_vs_rsp_truth']):.6f} | {float(row['insert_cnr_median']):.2f} | {float(row['aluminium_edge_10_90_median_mm']):.4f} |")
    water_change = float(new["water_std"]) / float(old["water_std"]) - 1
    rmse_change = float(new["phantom_rmse_vs_rsp_truth"]) / float(old["phantom_rmse_vs_rsp_truth"]) - 1
    lines += ["", f"水区标准差相对变化：{water_change * 100:+.2f}% 。", f"模体RSP RMSE相对变化：{rmse_change * 100:+.2f}% 。"]
    (QC / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(QC / "decision.json", {"status": "PASS", "scope": "equal-total-fluence baseline; no tuning", "water_std_relative_change": water_change, "phantom_rmse_relative_change": rmse_change})


def run(args: argparse.Namespace) -> None:
    preflight(args.force, args.device)
    progress(status="RUNNING", stage="prepare", configuration="360_angles_x_20pct", completed_runs=0, total_runs=RUNS)
    rows = prepare(args.jobs, args.force)
    selected_total = sum(int(row["selected"]) for row in rows)
    ddb(args.jobs, args.force)
    initial = analytic(args.force)
    result = iterative(initial, args.device, args.force)
    report(result, selected_total)
    if not args.keep_intermediates:
        shutil.rmtree(PRE / "pairs", ignore_errors=True)
        shutil.rmtree(PRE / "projections_ddb", ignore_errors=True)
    progress(status="COMPLETE", stage="report", task_eta_seconds=0)


def show_status() -> None:
    if not PROGRESS.is_file():
        print("Stage 8B angular-fluence baseline has not started.")
        return
    print(json.dumps(read_json(PROGRESS), ensure_ascii=False, indent=2))
    summary = QC / "comparison_summary.md"
    if summary.is_file():
        print("\n" + summary.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("run", "status"), default="run")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    show_status() if args.action == "status" else run(args)


if __name__ == "__main__":
    main()
