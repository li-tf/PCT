#!/usr/bin/env python3
"""Run one test0713 projection with native Windows OpenGATE.

Each process owns one angle and writes an independent pair of ROOT files.  The
companion merger restores the local RunID=0 to the global angle index so that
the combined files have the same RunID convention as the original test0713
720-run simulation.
"""

from __future__ import annotations

import argparse
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
DEFAULT_CONFIG = HERE / "simulation_config.json"


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "projections",
        "protons_per_projection",
        "random_seed",
        "beam_energy_mev",
        "source_z_mm",
        "focus_z_mm",
        "source_size_mm",
        "phantom_radius_mm",
        "phantom_length_mm",
        "insert_radii_mm",
        "detector_in_z_mm",
        "detector_out_z_mm",
        "physics_list",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration is missing keys: {missing}")
    return config


def validate_config(config: dict, angle: int, protons: int) -> None:
    if not 0 <= angle < int(config["projections"]):
        raise ValueError(
            f"angle must be in [0, {int(config['projections']) - 1}], got {angle}"
        )
    if protons < 1:
        raise ValueError("protons-per-projection must be positive")
    if len(config["insert_radii_mm"]) != 25:
        raise ValueError("expected exactly 25 aluminium inserts")
    if int(config.get("number_of_threads", 1)) != 1:
        raise ValueError("native Windows package requires number_of_threads=1")

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


def insert_centers(config: dict):
    step = float(config["insert_angle_step_deg"])
    for insert_id, radius_mm in enumerate(config["insert_radii_mm"]):
        angle = math.radians(step * insert_id)
        yield (
            insert_id,
            float(radius_mm) * math.cos(angle),
            float(radius_mm) * math.sin(angle),
        )


def angle_transform(config: dict, angle_index: int):
    """Return exactly the transform used by the 720-run test0713 simulation."""

    base_rotation = Rotation.from_euler(
        "yz", [90, 90], degrees=True
    ).as_matrix()
    translations, rotations = gate.geometry.utility.volume_orbiting_transform(
        "y",
        0,
        360,
        int(config["projections"]),
        [0, 0, 0],
        base_rotation,
    )
    return translations[angle_index], rotations[angle_index]


def add_phantom(sim: gate.Simulation, config: dict, angle_index: int) -> None:
    mm = gate.g4_units.mm
    translation, rotation = angle_transform(config, angle_index)

    phantom = sim.add_volume("Tubs", name="Spiral")
    phantom.rmin = 0 * mm
    phantom.rmax = float(config["phantom_radius_mm"]) * mm
    phantom.dz = float(config["phantom_length_mm"]) / 2.0 * mm
    phantom.material = str(config["phantom_material"])
    phantom.translation = translation
    phantom.rotation = rotation
    phantom.color = [0.18, 0.55, 0.85, 0.35]
    phantom.set_max_step_size(float(config["max_step_mm"]) * mm)

    radius = float(config["insert_diameter_mm"]) / 2.0
    for insert_id, local_x_mm, local_y_mm in insert_centers(config):
        insert = sim.add_volume("Tubs", name=f"SpiralInsert{insert_id:02d}")
        insert.mother = phantom.name
        insert.rmin = 0 * mm
        insert.rmax = radius * mm
        insert.dz = float(config["phantom_length_mm"]) / 2.0 * mm
        insert.material = str(config["insert_material"])
        insert.translation = [local_x_mm * mm, local_y_mm * mm, 0]
        insert.color = [0.95, 0.68, 0.10, 0.85]
        insert.set_max_step_size(float(config["max_step_mm"]) * mm)


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


def add_detector(
    sim: gate.Simulation,
    config: dict,
    name: str,
    z_mm: float,
    output_dir: Path,
) -> None:
    mm = gate.g4_units.mm
    plane = sim.add_volume("Box", f"PlanePhaseSpace{name}")
    plane.size = [float(value) * mm for value in config["detector_size_mm"]]
    plane.translation = [0, 0, z_mm * mm]
    plane.material = str(config["world_material"])

    actor = sim.add_actor("PhaseSpaceActor", f"PhaseSpace{name}")
    actor.attached_to = plane.name
    actor.attributes = list(config["output_attributes"])
    actor.output_filename = str(output_dir / f"PhaseSpace{name}.root")
    actor.filter = make_particle_filter()


def build_simulation(
    config: dict,
    angle_index: int,
    protons: int,
    output_dir: Path,
    qc_dir: Path,
    seed: int,
    verbose: bool,
) -> gate.Simulation:
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
    # Preserve the original absolute run time for PreGlobalTime even though
    # this process contains only one static projection.
    sim.run_timing_intervals = [
        [angle_index * second, (angle_index + 1) * second]
    ]

    sim.volume_manager.add_material_database(
        gate.utility.get_contrib_path() / "GateMaterials.db"
    )
    sim.world.material = str(config["world_material"])
    sim.world.size = [float(config["world_size_mm"]) * mm] * 3

    add_phantom(sim, config, angle_index)
    sim.physics_manager.set_user_limits_particles("proton")
    add_source(sim, config, protons)
    sim.physics_manager.physics_list_name = str(config["physics_list"])
    add_detector(
        sim, config, "In", float(config["detector_in_z_mm"]), output_dir
    )
    add_detector(
        sim, config, "Out", float(config["detector_out_z_mm"]), output_dir
    )
    statistics = sim.add_actor("SimulationStatisticsActor", "stat")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim


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
    config = load_config(args.config)
    protons = int(
        args.protons_per_projection
        if args.protons_per_projection is not None
        else config["protons_per_projection"]
    )
    seed = int(
        args.seed
        if args.seed is not None
        else int(config["random_seed"]) + args.angle
    )
    validate_config(config, args.angle, protons)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = args.qc_dir.resolve()
    qc_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    metadata_path = qc_dir / "run_metadata.json"
    metadata_record = {
        "status": "building",
        "angle_index": args.angle,
        "angle_degrees": args.angle * 360.0 / int(config["projections"]),
        "local_run_id": 0,
        "global_run_id_after_merge": args.angle,
        "protons_per_projection": protons,
        "random_engine": "MersenneTwister",
        "random_seed": seed,
        "started_at": started.isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"),
        "process_id": os.getpid(),
        "config": str(args.config.resolve()),
    }
    metadata_path.write_text(
        json.dumps(metadata_record, indent=2), encoding="utf-8"
    )

    start_clock = time.perf_counter()
    try:
        sim = build_simulation(
            config, args.angle, protons, output_dir, qc_dir, seed, args.verbose
        )
        if args.build_only:
            metadata_record["status"] = "build_only_completed"
            metadata_record["elapsed_seconds"] = time.perf_counter() - start_clock
            metadata_path.write_text(
                json.dumps(metadata_record, indent=2), encoding="utf-8"
            )
            print(f"Build-only check completed for angle {args.angle:03d}")
            return

        metadata_record["status"] = "running"
        metadata_path.write_text(
            json.dumps(metadata_record, indent=2), encoding="utf-8"
        )
        print(
            f"Starting angle {args.angle:03d} "
            f"({metadata_record['angle_degrees']:.1f} deg), "
            f"protons={protons:,}, seed={seed}",
            flush=True,
        )
        sim.run()

        required = [
            output_dir / "PhaseSpaceIn.root",
            output_dir / "PhaseSpaceOut.root",
            qc_dir / "protonct.txt",
        ]
        missing = [str(path.name) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"OpenGATE completed but outputs are missing: {missing}")

        metadata_record["status"] = "completed"
        metadata_record["completed_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_record["elapsed_seconds"] = time.perf_counter() - start_clock
        metadata_record["output_bytes"] = {
            path.name: path.stat().st_size for path in required
        }
        metadata_path.write_text(
            json.dumps(metadata_record, indent=2), encoding="utf-8"
        )
        (qc_dir / "completed.flag").write_text(
            f"angle={args.angle}\nseed={seed}\n", encoding="ascii"
        )
        print(
            f"Completed angle {args.angle:03d} in "
            f"{metadata_record['elapsed_seconds']:.1f} s",
            flush=True,
        )
    except Exception:
        metadata_record["status"] = "failed"
        metadata_record["failed_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_record["elapsed_seconds"] = time.perf_counter() - start_clock
        metadata_record["traceback"] = traceback.format_exc()
        metadata_path.write_text(
            json.dumps(metadata_record, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
