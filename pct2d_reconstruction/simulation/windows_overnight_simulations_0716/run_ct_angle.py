#!/usr/bin/env python3
"""Run one projection of a configured 2-D pCT diagnostic scenario."""

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

import opengate as gate
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
MATERIAL_DATABASE = gate.utility.get_contrib_path() / "GateMaterials.db"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    base_name = config.pop("base_config", None)
    if base_name:
        base = json.loads((path.parent / str(base_name)).read_text(encoding="utf-8"))
        base.update(config)
        config = base
    required = {
        "scenario_id", "output_name", "projections", "protons_per_projection",
        "random_seed", "beam_energy_mev", "source_z_mm", "focus_z_mm",
        "source_size_mm", "phantom_radius_mm", "phantom_length_mm",
        "phantom_material", "phantom_kind", "detector_in_z_mm",
        "detector_out_z_mm", "detector_size_mm", "world_material",
        "world_size_mm", "physics_list", "max_step_mm", "output_attributes",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing keys: {missing}")
    return config


def config_sha256(path: Path) -> str:
    effective = json.dumps(load_config(path), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(effective.encode("utf-8")).hexdigest()


def validate_config(config: dict, angle: int, protons: int) -> None:
    projections = int(config["projections"])
    if not 0 <= angle < projections:
        raise ValueError(f"angle must be in [0, {projections - 1}], got {angle}")
    if projections != 720:
        raise ValueError("diagnostic CT scenarios must retain all 720 angles")
    if protons < 1:
        raise ValueError("protons-per-projection must be positive")
    if config["phantom_kind"] not in {
        "aluminium_spiral", "uniform_water", "material_calibration", "resolution"
    }:
        raise ValueError(f"unknown phantom_kind: {config['phantom_kind']}")
    if str(config["world_material"]) not in {"Vacuum", "Air"}:
        raise ValueError("world_material must be Vacuum or Air")
    focus_distance = abs(float(config["focus_z_mm"]) - float(config["source_z_mm"]))
    magnification = abs(float(config["focus_z_mm"])) / focus_distance
    field_x = float(config["source_size_mm"][0]) * magnification
    field_y = float(config["source_size_mm"][1]) * magnification
    if not math.isclose(field_x, 250.0, abs_tol=1e-9):
        raise ValueError(f"unexpected isocenter x field: {field_x}")
    if not math.isclose(field_y, 2.0, abs_tol=1e-9):
        raise ValueError(f"unexpected isocenter y field: {field_y}")


def make_particle_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return builder.ParticleName == "proton"


def angle_transform(config: dict, angle_index: int):
    base_rotation = Rotation.from_euler("yz", [90, 90], degrees=True).as_matrix()
    translations, rotations = gate.geometry.utility.volume_orbiting_transform(
        "y", 0, 360, int(config["projections"]), [0, 0, 0], base_rotation
    )
    return translations[angle_index], rotations[angle_index]


def add_cylinder(
    sim: gate.Simulation, phantom, name: str, material: str,
    center: list[float], diameter_mm: float, length_mm: float, max_step_mm: float,
) -> None:
    mm = gate.g4_units.mm
    radius = float(diameter_mm) / 2.0
    if math.hypot(float(center[0]), float(center[1])) + radius > 100.0 + 1e-8:
        raise ValueError(f"cylinder {name} extends outside the phantom")
    volume = sim.add_volume("Tubs", name=name)
    volume.mother = phantom.name
    volume.rmin = 0 * mm
    volume.rmax = radius * mm
    volume.dz = float(length_mm) / 2.0 * mm
    volume.material = material
    volume.translation = [float(center[0]) * mm, float(center[1]) * mm, 0]
    volume.set_max_step_size(float(max_step_mm) * mm)


def add_box(
    sim: gate.Simulation, phantom, name: str, material: str,
    center: list[float], size_xy_mm: list[float], length_mm: float,
    rotation_deg: float, max_step_mm: float,
) -> None:
    mm = gate.g4_units.mm
    half_diagonal = 0.5 * math.hypot(float(size_xy_mm[0]), float(size_xy_mm[1]))
    if math.hypot(float(center[0]), float(center[1])) + half_diagonal > 100.0 + 1e-8:
        raise ValueError(f"box {name} extends outside the phantom")
    volume = sim.add_volume("Box", name=name)
    volume.mother = phantom.name
    volume.size = [float(size_xy_mm[0]) * mm, float(size_xy_mm[1]) * mm,
                   float(length_mm) * mm]
    volume.material = material
    volume.translation = [float(center[0]) * mm, float(center[1]) * mm, 0]
    volume.rotation = Rotation.from_euler("z", float(rotation_deg), degrees=True).as_matrix()
    volume.set_max_step_size(float(max_step_mm) * mm)


def add_aluminium_spiral(sim: gate.Simulation, phantom, config: dict) -> None:
    step = float(config["insert_angle_step_deg"])
    for insert_id, radius_mm in enumerate(config["insert_radii_mm"]):
        angle = math.radians(step * insert_id)
        add_cylinder(
            sim, phantom, f"AluminiumSpiral{insert_id:02d}", "Aluminium",
            [float(radius_mm) * math.cos(angle), float(radius_mm) * math.sin(angle)],
            float(config["insert_diameter_mm"]), float(config["phantom_length_mm"]),
            float(config["max_step_mm"]),
        )


def add_material_calibration(sim: gate.Simulation, phantom, config: dict) -> None:
    materials = list(config["calibration_materials"])
    insert_id = 0
    for ring_index, ring in enumerate(config["calibration_rings"]):
        radius = float(ring["radius_mm"])
        offset = float(ring["angle_offset_deg"])
        for material_index, material in enumerate(materials):
            angle = math.radians(offset + material_index * 360.0 / len(materials))
            add_cylinder(
                sim, phantom, f"Calibration{insert_id:02d}_{material}", material,
                [radius * math.cos(angle), radius * math.sin(angle)],
                float(ring["diameter_mm"]), float(config["phantom_length_mm"]),
                float(config["max_step_mm"]),
            )
            insert_id += 1
    add_cylinder(
        sim, phantom, "CalibrationSmallAluminium", "Aluminium", [0.0, 0.0],
        float(config["small_aluminium_diameter_mm"]),
        float(config["phantom_length_mm"]), float(config["max_step_mm"]),
    )


def add_resolution_targets(sim: gate.Simulation, phantom, config: dict) -> None:
    for group_index, group in enumerate(config["line_pair_groups"]):
        width = float(group["line_width_mm"])
        count = int(group["bar_count"])
        total = (2 * count - 1) * width
        for bar_index in range(count):
            offset = -0.5 * total + 0.5 * width + 2.0 * width * bar_index
            local = np_rotate([offset, 0.0], float(group.get("rotation_deg", 0.0)))
            center = [float(group["center_mm"][0]) + local[0],
                      float(group["center_mm"][1]) + local[1]]
            add_box(
                sim, phantom, f"LinePair{group_index:02d}_{bar_index:02d}",
                str(group["material"]), center,
                [width, float(group["bar_length_mm"])],
                float(config["phantom_length_mm"]), float(group.get("rotation_deg", 0.0)),
                min(float(config["max_step_mm"]), width / 2.0),
            )
    for target_index, target in enumerate(config["edge_targets"]):
        add_box(
            sim, phantom, f"EdgeTarget{target_index:02d}_{target['material']}",
            str(target["material"]), list(target["center_mm"]),
            list(target["size_xy_mm"]), float(config["phantom_length_mm"]),
            float(target["rotation_deg"]), float(config["max_step_mm"]),
        )


def np_rotate(point: list[float], angle_deg: float) -> list[float]:
    angle = math.radians(angle_deg)
    return [math.cos(angle) * point[0] - math.sin(angle) * point[1],
            math.sin(angle) * point[0] + math.cos(angle) * point[1]]


def add_phantom(sim: gate.Simulation, config: dict, angle_index: int) -> None:
    mm = gate.g4_units.mm
    translation, rotation = angle_transform(config, angle_index)
    phantom = sim.add_volume("Tubs", name="DiagnosticPhantom")
    phantom.rmin = 0 * mm
    phantom.rmax = float(config["phantom_radius_mm"]) * mm
    phantom.dz = float(config["phantom_length_mm"]) / 2.0 * mm
    phantom.material = str(config["phantom_material"])
    phantom.translation = translation
    phantom.rotation = rotation
    phantom.set_max_step_size(float(config["max_step_mm"]) * mm)
    kind = config["phantom_kind"]
    if kind == "aluminium_spiral":
        add_aluminium_spiral(sim, phantom, config)
    elif kind == "material_calibration":
        add_material_calibration(sim, phantom, config)
    elif kind == "resolution":
        add_resolution_targets(sim, phantom, config)


def add_source(sim: gate.Simulation, config: dict, protons: int) -> None:
    mm = gate.g4_units.mm
    MeV = gate.g4_units.MeV
    Bq = gate.g4_units.Bq
    source = sim.add_source("GenericSource", "mybeam")
    source.particle = "proton"
    source.energy.type = "mono"
    source.energy.mono = float(config["beam_energy_mev"]) * MeV
    source.position.type = "box"
    source.position.size = [float(value) * mm for value in config["source_size_mm"]]
    source.position.translation = [0, 0, float(config["source_z_mm"]) * mm]
    source.direction.type = "focused"
    source.direction.focus_point = [0, 0, float(config["focus_z_mm"]) * mm]
    source.activity = protons * Bq


def add_detector(sim: gate.Simulation, config: dict, name: str, z_mm: float,
                 output_dir: Path) -> None:
    mm = gate.g4_units.mm
    plane = sim.add_volume("Box", f"PlanePhaseSpace{name}")
    plane.size = [float(value) * mm for value in config["detector_size_mm"]]
    plane.translation = [0, 0, float(z_mm) * mm]
    plane.material = str(config["world_material"])
    actor = sim.add_actor("PhaseSpaceActor", f"PhaseSpace{name}")
    actor.attached_to = plane.name
    actor.attributes = list(config["output_attributes"])
    actor.output_filename = str(output_dir / f"PhaseSpace{name}.root")
    actor.filter = make_particle_filter()


def build_simulation(config: dict, angle_index: int, protons: int,
                     output_dir: Path, qc_dir: Path, seed: int,
                     verbose: bool) -> gate.Simulation:
    second = gate.g4_units.second
    mm = gate.g4_units.mm
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.number_of_threads = 1
    sim.run_timing_intervals = [[angle_index * second, (angle_index + 1) * second]]
    sim.volume_manager.add_material_database(MATERIAL_DATABASE)
    sim.world.material = str(config["world_material"])
    sim.world.size = [float(config["world_size_mm"]) * mm] * 3
    add_phantom(sim, config, angle_index)
    sim.physics_manager.set_user_limits_particles("proton")
    add_source(sim, config, protons)
    sim.physics_manager.physics_list_name = str(config["physics_list"])
    add_detector(sim, config, "In", float(config["detector_in_z_mm"]), output_dir)
    add_detector(sim, config, "Out", float(config["detector_out_z_mm"]), output_dir)
    statistics = sim.add_actor("SimulationStatisticsActor", "stat")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
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
    output_dir = args.output_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    metadata_path = qc_dir / "run_metadata.json"
    record = {
        "status": "building", "scenario_id": config["scenario_id"],
        "output_name": config["output_name"], "angle_index": args.angle,
        "angle_degrees": args.angle * 360.0 / int(config["projections"]),
        "local_run_id": 0, "global_run_id_after_merge": args.angle,
        "protons_per_projection": protons, "random_engine": "MersenneTwister",
        "random_seed": seed, "started_at": started.isoformat(timespec="seconds"),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"), "process_id": os.getpid(),
        "config": str(config_path), "config_sha256": config_sha256(config_path),
    }
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    start_clock = time.perf_counter()
    try:
        sim = build_simulation(config, args.angle, protons, output_dir, qc_dir, seed, args.verbose)
        if args.build_only:
            record["status"] = "build_only_completed"
            record["elapsed_seconds"] = time.perf_counter() - start_clock
            metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"Build-only PASS: {config['scenario_id']} angle {args.angle:03d}")
            return
        record["status"] = "running"
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"Starting {config['scenario_id']} angle {args.angle:03d}, protons={protons:,}", flush=True)
        sim.run()
        required = [output_dir / "PhaseSpaceIn.root", output_dir / "PhaseSpaceOut.root",
                    qc_dir / "protonct.txt"]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"OpenGATE completed but outputs are missing: {missing}")
        record["status"] = "completed"
        record["completed_at"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_seconds"] = time.perf_counter() - start_clock
        record["output_bytes"] = {path.name: path.stat().st_size for path in required}
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (qc_dir / "completed.flag").write_text(
            f"scenario={config['scenario_id']}\nangle={args.angle}\nseed={seed}\n", encoding="ascii"
        )
        print(f"Completed {config['scenario_id']} angle {args.angle:03d} in {record['elapsed_seconds']:.1f} s", flush=True)
    except Exception:
        record["status"] = "failed"
        record["failed_at"] = datetime.now().isoformat(timespec="seconds")
        record["elapsed_seconds"] = time.perf_counter() - start_clock
        record["traceback"] = traceback.format_exc()
        metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
