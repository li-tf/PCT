#!/usr/bin/env python3
"""Run or enumerate one material/energy/thickness calibration case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import opengate as gate


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "material_scan_config.json"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def enumerate_cases(config: dict) -> list[dict]:
    cases = []
    for energy in config["energies_mev"]:
        for material in config["materials"]:
            for thickness in material["thicknesses_mm"]:
                case_id = f"{material['name'].lower()}_e{int(energy):03d}_t{int(thickness):04d}"
                cases.append({
                    "case_index": len(cases), "case_id": case_id,
                    "material": material["name"], "energy_mev": float(energy),
                    "thickness_mm": float(thickness),
                    "max_step_mm": float(material["max_step_mm"]),
                })
    return cases


def make_particle_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return builder.ParticleName == "proton"


def add_plane(sim, config: dict, name: str, z_mm: float, output_dir: Path) -> None:
    mm = gate.g4_units.mm
    plane = sim.add_volume("Box", f"Plane{name}")
    plane.size = [120 * mm, 20 * mm, 0.000001 * mm]
    plane.translation = [0, 0, z_mm * mm]
    plane.material = "Vacuum"
    actor = sim.add_actor("PhaseSpaceActor", f"PhaseSpace{name}")
    actor.attached_to = plane.name
    actor.attributes = list(config["output_attributes"])
    actor.output_filename = str(output_dir / f"PhaseSpace{name}.root")
    actor.filter = make_particle_filter()


def build_simulation(config: dict, case: dict, protons: int, seed: int,
                     output_dir: Path, qc_dir: Path, verbose: bool):
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    second = gate.g4_units.second
    half = case["thickness_mm"] / 2.0
    entrance_z = -half - 0.5
    exit_z = half + 0.5
    source_z = entrance_z - 5.0
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.number_of_threads = 1
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.run_timing_intervals = [[0 * second, 1 * second]]
    sim.volume_manager.add_material_database(gate.utility.get_contrib_path() / "GateMaterials.db")
    sim.world.material = "Vacuum"
    sim.world.size = [500 * mm, 500 * mm, max(4000.0, case["thickness_mm"] + 100.0) * mm]
    slab = sim.add_volume("Box", "CalibrationSlab")
    slab.size = [100 * mm, 20 * mm, case["thickness_mm"] * mm]
    slab.material = case["material"]
    slab.set_max_step_size(case["max_step_mm"] * mm)
    source = sim.add_source("GenericSource", "calibration_beam")
    source.particle = "proton"
    source.energy.type = "mono"
    source.energy.mono = case["energy_mev"] * MeV
    source.position.type = "box"
    source.position.size = [10 * mm, 0.1 * mm, 0.000001 * mm]
    source.position.translation = [0, 0, source_z * mm]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]
    source.activity = protons * Bq
    sim.physics_manager.set_user_limits_particles("proton")
    sim.physics_manager.physics_list_name = config["physics_list"]
    add_plane(sim, config, "In", entrance_z, output_dir)
    add_plane(sim, config, "Out", exit_z, output_dir)
    statistics = sim.add_actor("SimulationStatisticsActor", "stat")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--write-cases-json", type=Path,
        help="Write the enumerated cases to a JSON file instead of stdout.",
    )
    parser.add_argument("--case-index", type=int)
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
    cases = enumerate_cases(config)
    if args.write_cases_json is not None:
        args.write_cases_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_cases_json.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
        return
    if args.list_cases:
        print(json.dumps(cases))
        return
    if args.case_index is None or args.output_dir is None or args.qc_dir is None:
        raise SystemExit("--case-index, --output-dir and --qc-dir are required")
    if not 0 <= args.case_index < len(cases):
        raise ValueError(f"case-index must be in [0, {len(cases)-1}]")
    case = cases[args.case_index]
    protons = int(args.protons or config["protons_per_case"])
    seed = int(config["random_seed"]) + args.case_index
    output_dir = args.output_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    record = {
        **case, "scenario_id": config["scenario_id"], "status": "building",
        "protons": protons, "random_seed": seed,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"), "process_id": os.getpid(),
    }
    metadata_path = qc_dir / "case_metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    start = time.perf_counter()
    try:
        sim = build_simulation(config, case, protons, seed, output_dir, qc_dir, args.verbose)
        if args.build_only:
            record["status"] = "build_only_completed"
            record["elapsed_seconds"] = time.perf_counter() - start
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"Build-only PASS: {case['case_id']}")
            return
        record["status"] = "running"
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Starting {case['case_id']}, protons={protons:,}", flush=True)
        sim.run()
        required = [output_dir / "PhaseSpaceIn.root", output_dir / "PhaseSpaceOut.root",
                    qc_dir / "protonct.txt"]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing outputs: {missing}")
        record["status"] = "completed"
        record["completed_at"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_seconds"] = time.perf_counter() - start
        record["output_bytes"] = {path.name: path.stat().st_size for path in required}
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (qc_dir / "completed.flag").write_text(f"case={case['case_id']}\n", encoding="ascii")
        print(f"Completed {case['case_id']} in {record['elapsed_seconds']:.1f} s", flush=True)
    except Exception:
        record["status"] = "failed"
        record["failed_at"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_seconds"] = time.perf_counter() - start
        record["traceback"] = traceback.format_exc()
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
