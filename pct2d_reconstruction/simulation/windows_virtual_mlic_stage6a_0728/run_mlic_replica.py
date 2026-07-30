#!/usr/bin/env python3
"""Run one independent virtual-MLIC depth-dose replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import opengate as gate


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "simulation_config.json"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def enumerate_cases(config: dict) -> list[dict]:
    cases: list[dict] = []
    for energy in config["energies_mev"]:
        for item in config["cases_per_energy"]:
            case_id = f"e{int(energy):03d}_{slug(item['name'])}"
            cases.append(
                {
                    "case_index": len(cases),
                    "case_id": case_id,
                    "energy_mev": float(energy),
                    "name": item["name"],
                    "material": item["material"],
                    "thickness_mm": float(item["thickness_mm"]),
                }
            )
    return cases


def enumerate_tasks(config: dict) -> list[dict]:
    tasks: list[dict] = []
    for case in enumerate_cases(config):
        for replica in range(int(config["replicates_per_case"])):
            tasks.append(
                {
                    **case,
                    "task_index": len(tasks),
                    "replica": replica,
                    "task_id": f"{case['case_id']}_r{replica:02d}",
                }
            )
    return tasks


def validate_config(config: dict) -> None:
    if int(config["replicates_per_case"]) < 2:
        raise ValueError("replicates_per_case must be at least 2")
    if int(config["protons_per_case"]) % int(config["replicates_per_case"]):
        raise ValueError("protons_per_case must be divisible by replicates_per_case")
    tank_z = float(config["water_tank_size_mm"][2])
    bin_z = float(config["depth_bin_mm"])
    bins = tank_z / bin_z
    if abs(bins - round(bins)) > 1e-9:
        raise ValueError("water tank length must be divisible by depth_bin_mm")
    names = [item["name"] for item in config["cases_per_energy"]]
    if names.count("Reference") != 1:
        raise ValueError("exactly one Reference case is required per energy")
    for item in config["cases_per_energy"]:
        if item["name"] == "Reference":
            if item["material"] is not None or float(item["thickness_mm"]) != 0:
                raise ValueError("Reference must have null material and zero thickness")
        elif item["material"] is None or float(item["thickness_mm"]) <= 0:
            raise ValueError(f"invalid material case: {item}")


def build_simulation(
    config: dict,
    task: dict,
    protons: int,
    seed: int,
    output_dir: Path,
    qc_dir: Path,
    verbose: bool,
):
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV

    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.number_of_threads = int(config["number_of_threads"])
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.volume_manager.add_material_database(
        gate.utility.get_contrib_path() / "GateMaterials.db"
    )
    sim.world.material = config["world_material"]
    sim.world.size = [500 * mm, 500 * mm, 1000 * mm]

    tank_size = [float(v) for v in config["water_tank_size_mm"]]
    tank_front = float(config["water_tank_front_z_mm"])
    tank = sim.add_volume("Box", "WaterTank")
    tank.size = [v * mm for v in tank_size]
    tank.translation = [0, 0, (tank_front + tank_size[2] / 2.0) * mm]
    tank.material = "Water"
    tank.set_max_step_size(float(config["water_max_step_mm"]) * mm)

    if task["material"] is not None:
        thickness = float(task["thickness_mm"])
        gap = float(config["sample_downstream_gap_mm"])
        transverse = [float(v) for v in config["sample_transverse_size_mm"]]
        sample = sim.add_volume("Box", "CalibrationSample")
        sample.size = [transverse[0] * mm, transverse[1] * mm, thickness * mm]
        sample.translation = [0, 0, -(gap + thickness / 2.0) * mm]
        sample.material = task["material"]
        sample.set_max_step_size(float(config["sample_max_step_mm"]) * mm)

    source = sim.add_source("GenericSource", "proton_beam")
    source.particle = "proton"
    source.n = int(protons)
    source.energy.type = "mono"
    source.energy.mono = float(task["energy_mev"]) * MeV
    source.position.type = "box"
    source.position.size = [
        float(v) * mm for v in config["source_size_mm"]
    ]
    source.position.translation = [0, 0, float(config["source_z_mm"]) * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]

    sim.physics_manager.physics_list_name = config["physics_list"]
    sim.physics_manager.set_user_limits_particles("proton")

    bins = int(round(tank_size[2] / float(config["depth_bin_mm"])))
    dose = sim.add_actor("DoseActor", "DepthDose")
    dose.attached_to = tank.name
    dose.output_filename = str(output_dir / "depth_dose.mhd")
    dose.size = [1, 1, bins]
    dose.spacing = [
        tank_size[0] * mm,
        tank_size[1] * mm,
        float(config["depth_bin_mm"]) * mm,
    ]
    dose.hit_type = "random"
    dose.edep.active = True
    dose.edep_squared.active = False
    dose.edep_uncertainty.active = False
    dose.dose.active = False

    statistics = sim.add_actor("SimulationStatisticsActor", "Statistics")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim


def read_mhd(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    header: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    raw_name = header.get("ElementDataFile")
    if not raw_name:
        raise ValueError(f"ElementDataFile missing in {path}")
    dtype_map = {
        "MET_FLOAT": np.dtype("<f4"),
        "MET_DOUBLE": np.dtype("<f8"),
        "MET_UINT": np.dtype("<u4"),
    }
    dtype = dtype_map.get(header.get("ElementType", ""))
    if dtype is None:
        raise ValueError(f"unsupported ElementType: {header.get('ElementType')}")
    dims = tuple(int(v) for v in header["DimSize"].split())
    values = np.fromfile(path.parent / raw_name, dtype=dtype)
    if values.size != int(np.prod(dims)):
        raise ValueError(f"MHD/RAW size mismatch: {values.size} != {np.prod(dims)}")
    return values.reshape(tuple(reversed(dims))), header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--write-tasks-json", type=Path)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--protons", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)
    tasks = enumerate_tasks(config)
    if args.write_tasks_json:
        args.write_tasks_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_tasks_json.write_text(
            json.dumps(tasks, indent=2) + "\n", encoding="utf-8"
        )
        return
    if args.list_tasks:
        print(json.dumps(tasks))
        return
    if args.task_index is None or args.output_dir is None or args.qc_dir is None:
        raise SystemExit("--task-index, --output-dir and --qc-dir are required")
    if not 0 <= args.task_index < len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")

    task = tasks[args.task_index]
    default_protons = (
        int(config["protons_per_case"]) // int(config["replicates_per_case"])
    )
    protons = int(args.protons or default_protons)
    seed = int(config["random_seed"]) + int(task["task_index"])
    output_dir = args.output_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    record = {
        **task,
        "scenario_id": config["scenario_id"],
        "status": "building",
        "protons": protons,
        "random_seed": seed,
        "config_sha256": config_hash,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"),
        "process_id": os.getpid(),
    }
    metadata_path = qc_dir / "task_metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        sim = build_simulation(
            config, task, protons, seed, output_dir, qc_dir, args.verbose
        )
        if args.build_only:
            record["status"] = "build_only_completed"
            record["elapsed_seconds"] = time.perf_counter() - start
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"Build-only PASS: {task['task_id']}")
            return
        record["status"] = "running"
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"Starting {task['task_id']}, protons={protons:,}, seed={seed}",
            flush=True,
        )
        sim.run()
        # OpenGATE 10.1.0 appends the active score name to the configured
        # base filename.
        dose_path = output_dir / "depth_dose_edep.mhd"
        stats_path = qc_dir / "protonct.txt"
        if not dose_path.is_file() or not stats_path.is_file():
            raise RuntimeError("depth_dose_edep.mhd or protonct.txt is missing")
        curve, header = read_mhd(dose_path)
        flat = curve.reshape(-1)
        if flat.size != int(round(
            float(config["water_tank_size_mm"][2])
            / float(config["depth_bin_mm"])
        )):
            raise RuntimeError(f"unexpected depth-dose bins: {flat.size}")
        if not np.isfinite(flat).all() or float(flat.sum()) <= 0:
            raise RuntimeError("depth-dose curve is non-finite or empty")
        record.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_seconds": time.perf_counter() - start,
                "depth_bins": int(flat.size),
                "depth_spacing_mm": float(config["depth_bin_mm"]),
                "edep_sum": float(flat.sum()),
                "edep_max": float(flat.max()),
                "mhd_element_type": header.get("ElementType"),
                "output_bytes": {
                    "depth_dose_edep.mhd": dose_path.stat().st_size,
                    header["ElementDataFile"]: (
                        dose_path.parent / header["ElementDataFile"]
                    ).stat().st_size,
                    "protonct.txt": stats_path.stat().st_size,
                },
            }
        )
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (qc_dir / "completed.flag").write_text(
            f"task={task['task_id']}\nconfig_sha256={config_hash}\n",
            encoding="ascii",
        )
        print(
            f"Completed {task['task_id']} in {record['elapsed_seconds']:.1f} s",
            flush=True,
        )
    except Exception:
        record["status"] = "failed"
        record["failed_at"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_seconds"] = time.perf_counter() - start
        record["traceback"] = traceback.format_exc()
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
