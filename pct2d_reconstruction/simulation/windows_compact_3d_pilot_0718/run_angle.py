#!/usr/bin/env python3
"""Run one projection of the compact three-dimensional pCT pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import opengate as gate
from scipy.spatial.transform import Rotation
import uproot


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "simulation_config.json"
MATERIAL_DATABASE = gate.utility.get_contrib_path() / "GateMaterials.db"
REQUIRED_BRANCHES = {
    "RunID", "EventID", "TrackID", "KineticEnergy", "PreGlobalTime",
    "Position_X", "Position_Y", "Position_Z",
    "Direction_X", "Direction_Y", "Direction_Z",
}


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def base_rotation() -> np.ndarray:
    """Map phantom local z to scanner y at projection angle zero."""
    return Rotation.from_euler("yz", [90, 90], degrees=True).as_matrix()


def validate_spheres(config: dict) -> dict:
    radius = float(config["phantom_radius_mm"])
    half_length = float(config["phantom_length_mm"]) / 2.0
    spheres = list(config["spheres"])
    for sphere in spheres:
        center = np.asarray(sphere["scanner_center_mm"], dtype=float)
        sphere_radius = float(sphere["diameter_mm"]) / 2.0
        if math.hypot(center[0], center[2]) + sphere_radius > radius + 1e-9:
            raise ValueError(f"{sphere['name']} extends outside radial water support")
        if abs(center[1]) + sphere_radius > half_length + 1e-9:
            raise ValueError(f"{sphere['name']} extends outside axial water support")
    for index, first in enumerate(spheres):
        c1 = np.asarray(first["scanner_center_mm"], dtype=float)
        r1 = float(first["diameter_mm"]) / 2.0
        for second in spheres[index + 1:]:
            c2 = np.asarray(second["scanner_center_mm"], dtype=float)
            r2 = float(second["diameter_mm"]) / 2.0
            if np.linalg.norm(c1 - c2) < r1 + r2 - 1e-9:
                raise ValueError(f"spheres overlap: {first['name']} and {second['name']}")
    return {"sphere_count": len(spheres), "overlaps": 0, "inside_support": True}


def validate_config(config: dict, angle: int, protons: int) -> None:
    if int(config["projections"]) != 360 or not 0 <= angle < 360:
        raise ValueError("compact 3-D pilot requires angle in [0, 359]")
    if protons < 1 or int(config.get("number_of_threads", 1)) != 1:
        raise ValueError("protons must be positive and number_of_threads must be one")
    if config["world_material"] != "Air":
        raise ValueError("compact 3-D pilot requires an Air world")
    if len(config["reference_planes"]) != 2:
        raise ValueError("two ideal reference planes are required")
    if [float(item["z_mm"]) for item in config["reference_planes"]] != [-60.0, 60.0]:
        raise ValueError("unexpected 3-D reference-plane positions")
    source_z, focus_z = float(config["source_z_mm"]), float(config["focus_z_mm"])
    scale = abs(focus_z) / abs(focus_z - source_z)
    field = np.asarray(config["source_size_mm"][:2], dtype=float) * scale
    if not np.allclose(field, np.asarray(config["isocenter_field_mm"]), atol=1e-9):
        raise ValueError(f"source does not create configured isocenter field: {field}")
    out_scale = abs(60.0 - focus_z) / abs(focus_z - source_z)
    exit_field = np.asarray(config["source_size_mm"][:2], dtype=float) * out_scale
    if np.any(exit_field > np.asarray(config["reference_size_mm"][:2], dtype=float)):
        raise ValueError("reference plane does not cover the divergent field")
    expected_grid = np.asarray(config["recommended_reconstruction"]["fov_mm"]) / np.asarray(
        config["recommended_reconstruction"]["spacing_mm"]
    )
    if not np.array_equal(expected_grid.astype(int), config["recommended_reconstruction"]["size"]):
        raise ValueError("inconsistent recommended reconstruction grid")
    validate_spheres(config)


def particle_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return builder.ParticleName == "proton"


def angle_transform(config: dict, angle_index: int):
    translations, rotations = gate.geometry.utility.volume_orbiting_transform(
        "y", 0, 360, int(config["projections"]), [0, 0, 0], base_rotation()
    )
    return translations[angle_index], rotations[angle_index]


def scanner_to_phantom_local(scanner_center: list[float]) -> np.ndarray:
    return base_rotation().T @ np.asarray(scanner_center, dtype=float)


def add_phantom(sim: gate.Simulation, config: dict, angle_index: int) -> None:
    mm = gate.g4_units.mm
    translation, rotation = angle_transform(config, angle_index)
    phantom = sim.add_volume("Tubs", name="CompactWaterCylinder")
    phantom.rmin = 0 * mm
    phantom.rmax = float(config["phantom_radius_mm"]) * mm
    phantom.dz = float(config["phantom_length_mm"]) / 2.0 * mm
    phantom.material = str(config["phantom_material"])
    phantom.translation = translation
    phantom.rotation = rotation
    phantom.set_max_step_size(float(config["phantom_max_step_mm"]) * mm)
    for item in config["spheres"]:
        sphere = sim.add_volume("Sphere", name=str(item["name"]))
        sphere.mother = phantom.name
        sphere.rmin = 0 * mm
        sphere.rmax = float(item["diameter_mm"]) / 2.0 * mm
        sphere.material = str(item["material"])
        local = scanner_to_phantom_local(item["scanner_center_mm"])
        sphere.translation = [float(value) * mm for value in local]
        sphere.set_max_step_size(float(config["phantom_max_step_mm"]) * mm)


def add_source(sim: gate.Simulation, config: dict, protons: int) -> None:
    source = sim.add_source("GenericSource", "mybeam")
    source.particle = "proton"
    source.energy.type = "mono"
    source.energy.mono = float(config["beam_energy_mev"]) * gate.g4_units.MeV
    source.position.type = "box"
    source.position.size = [float(v) * gate.g4_units.mm for v in config["source_size_mm"]]
    source.position.translation = [0, 0, float(config["source_z_mm"]) * gate.g4_units.mm]
    source.direction.type = "focused"
    source.direction.focus_point = [0, 0, float(config["focus_z_mm"]) * gate.g4_units.mm]
    source.activity = protons * gate.g4_units.Bq


def add_reference_plane(sim: gate.Simulation, config: dict, item: dict,
                        output_dir: Path) -> None:
    mm = gate.g4_units.mm
    name = str(item["name"])
    plane = sim.add_volume("Box", f"Volume{name}")
    plane.size = [float(v) * mm for v in config["reference_size_mm"]]
    plane.translation = [0, 0, float(item["z_mm"]) * mm]
    plane.material = "Air"
    actor = sim.add_actor("PhaseSpaceActor", name)
    actor.attached_to = plane.name
    actor.attributes = list(config["output_attributes"])
    actor.steps_to_store = "entering"
    actor.output_filename = str(output_dir / f"{name}.root")
    actor.filter = particle_filter()


def build_simulation(config: dict, angle: int, protons: int, output_dir: Path,
                     qc_dir: Path, seed: int, verbose: bool) -> gate.Simulation:
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.number_of_threads = 1
    sim.run_timing_intervals = [[angle * gate.g4_units.second, (angle + 1) * gate.g4_units.second]]
    sim.volume_manager.add_material_database(MATERIAL_DATABASE)
    sim.world.material = str(config["world_material"])
    sim.world.size = [float(config["world_size_mm"]) * gate.g4_units.mm] * 3
    add_phantom(sim, config, angle)
    add_source(sim, config, protons)
    for item in config["reference_planes"]:
        add_reference_plane(sim, config, item, output_dir)
    sim.physics_manager.set_user_limits_particles("proton")
    sim.physics_manager.physics_list_name = str(config["physics_list"])
    statistics = sim.add_actor("SimulationStatisticsActor", "stat")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim


def inspect_root(path: Path, tree_name: str) -> dict:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"missing tree {tree_name} in {path.name}")
        tree = root_file[tree_name]
        missing = sorted(REQUIRED_BRANCHES - set(tree.keys()))
        if missing:
            raise RuntimeError(f"missing branches in {path.name}: {missing}")
        count = int(tree.num_entries)
        if count < 1:
            raise RuntimeError(f"empty ROOT tree: {path.name}")
        identity = tree.arrays(["RunID", "EventID", "TrackID"], library="np")
        if np.any(identity["RunID"] != 0):
            raise RuntimeError(f"nonzero local RunID in {path.name}")
        primary = identity["TrackID"] == 1
        primary_events = identity["EventID"][primary].astype(np.int64, copy=False)
        duplicate_primary = int(len(primary_events) - len(np.unique(primary_events)))
        floating = tree.arrays(
            ["KineticEnergy", "PreGlobalTime", "Position_X", "Position_Y", "Position_Z",
             "Direction_X", "Direction_Y", "Direction_Z"],
            entry_stop=min(count, 20000), library="np",
        )
        if not all(np.isfinite(values).all() for values in floating.values()):
            raise RuntimeError(f"non-finite values in {path.name}")
    return {
        "tree": tree_name, "entries": count,
        "primary_entries": int(np.count_nonzero(primary)),
        "duplicate_primary_event_hits": duplicate_primary,
        "bytes": path.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--angle", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--protons-per-projection", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    protons = int(args.protons_per_projection or config["protons_per_projection"])
    seed = int(args.seed if args.seed is not None else int(config["random_seed"]) + args.angle)
    validate_config(config, args.angle, protons)
    output_dir, qc_dir = args.output_dir.resolve(), args.qc_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = qc_dir / "run_metadata.json"
    record = {
        "status": "building", "scenario_id": config["scenario_id"],
        "angle_index": args.angle, "angle_degrees": args.angle,
        "protons_per_projection": protons, "random_seed": seed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"),
        "config": str(config_path), "config_sha256": config_sha256(config_path),
        "process_id": os.getpid(), "sphere_validation": validate_spheres(config),
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    clock = time.perf_counter()
    try:
        sim = build_simulation(config, args.angle, protons, output_dir, qc_dir, seed, args.verbose)
        if args.build_only:
            record.update(status="build_only_completed", elapsed_seconds=time.perf_counter() - clock)
            metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"Compact 3-D build-only PASS for angle {args.angle:03d}")
            return
        record["status"] = "running"
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Starting compact 3-D angle {args.angle:03d}, protons={protons:,}, seed={seed}", flush=True)
        sim.run()
        summaries = {}
        for name in ("PhaseSpaceIn", "PhaseSpaceOut"):
            path = output_dir / f"{name}.root"
            if not path.is_file():
                raise RuntimeError(f"missing output: {path.name}")
            summaries[path.name] = inspect_root(path, name)
        if not (qc_dir / "protonct.txt").is_file():
            raise RuntimeError("missing SimulationStatisticsActor output")
        record.update(
            status="completed", completed_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=time.perf_counter() - clock, root_qc=summaries,
            output_bytes=sum(item["bytes"] for item in summaries.values()),
        )
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        (qc_dir / "completed.flag").write_text(
            f"scenario={config['scenario_id']}\nangle={args.angle}\nseed={seed}\n"
            f"config_sha256={record['config_sha256']}\n", encoding="ascii"
        )
        print(f"Completed compact 3-D angle {args.angle:03d} in {record['elapsed_seconds']:.1f} s", flush=True)
    except Exception:
        record.update(status="failed", failed_at=datetime.now().isoformat(timespec="seconds"),
                      elapsed_seconds=time.perf_counter() - clock, traceback=traceback.format_exc())
        metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
