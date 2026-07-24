#!/usr/bin/env python3
"""Run one angle of the low-fluence heterogeneous MLP truth pilot."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import opengate as gate
from scipy.spatial.transform import Rotation
import uproot


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "simulation_config.json"
MATERIAL_DATABASE = gate.utility.get_contrib_path() / "GateMaterials.db"
REFERENCE_BRANCHES = {
    "RunID",
    "EventID",
    "TrackID",
    "KineticEnergy",
    "PreGlobalTime",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Direction_X",
    "Direction_Y",
    "Direction_Z",
}
TRAJECTORY_BRANCHES = {
    "RunID",
    "EventID",
    "TrackID",
    "ParentID",
    "KineticEnergy",
    "PreGlobalTime",
    "PrePosition_X",
    "PrePosition_Y",
    "PrePosition_Z",
    "PostPosition_X",
    "PostPosition_Y",
    "PostPosition_Z",
    "PreDirection_X",
    "PreDirection_Y",
    "PreDirection_Z",
    "PostDirection_X",
    "PostDirection_Y",
    "PostDirection_Z",
}


def read_branches_by_basket(
    tree, names: list[str], entry_stop: int | None = None
) -> dict[str, np.ndarray]:
    """Read OpenGATE TTrees without the uproot 5.7 whole-branch stall.

    OpenGATE 10.1 may report TTree entry counts as floating-point metadata.
    In uproot 5.7.5, ``branch.array`` can then stall for trees with several
    baskets.  Decoding each TBasket individually is equivalent and fast.
    """

    result = {}
    for name in names:
        branch = tree[name]
        if hasattr(branch, "num_baskets"):
            pieces = [
                branch.basket(index).array(library="np")
                for index in range(branch.num_baskets)
            ]
            if not pieces:
                values = np.empty(0, dtype=np.float32)
            elif len(pieces) == 1:
                values = pieces[0]
            else:
                values = np.concatenate(pieces)
        else:
            values = branch.array(library="np")
        if entry_stop is not None:
            values = values[:entry_stop]
        result[name] = values
    return result


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def base_rotation() -> np.ndarray:
    """Map the local cylinder axis z onto the scanner y axis."""

    return Rotation.from_euler("yz", [90, 90], degrees=True).as_matrix()


def angle_transform(config: dict, angle_index: int):
    translations, rotations = gate.geometry.utility.volume_orbiting_transform(
        "y",
        0,
        360,
        int(config["projections"]),
        [0, 0, 0],
        base_rotation(),
    )
    return translations[angle_index], rotations[angle_index]


def primary_proton_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return (builder.ParticleName == "proton") & (builder.TrackID == 1)


def proton_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return builder.ParticleName == "proton"


def validate_inserts(config: dict) -> dict:
    phantom_radius = float(config["phantom_radius_mm"])
    inserts = list(config["inserts"])
    for item in inserts:
        center = np.asarray(item["center_local_xy_mm"], dtype=float)
        radius = float(item["diameter_mm"]) / 2.0
        if np.linalg.norm(center) + radius > phantom_radius + 1.0e-9:
            raise ValueError(f"{item['name']} extends outside the water cylinder")
    minimum_gap = math.inf
    for index, first in enumerate(inserts):
        center_first = np.asarray(first["center_local_xy_mm"], dtype=float)
        radius_first = float(first["diameter_mm"]) / 2.0
        for second in inserts[index + 1 :]:
            center_second = np.asarray(second["center_local_xy_mm"], dtype=float)
            radius_second = float(second["diameter_mm"]) / 2.0
            gap = float(
                np.linalg.norm(center_first - center_second)
                - radius_first
                - radius_second
            )
            minimum_gap = min(minimum_gap, gap)
            if gap < -1.0e-9:
                raise ValueError(
                    f"inserts overlap: {first['name']} and {second['name']}"
                )
    return {
        "insert_count": len(inserts),
        "inside_support": True,
        "overlaps": 0,
        "minimum_surface_gap_mm": minimum_gap,
    }


def validate_config(config: dict, angle: int, protons: int) -> None:
    projections = int(config["projections"])
    if projections != 72 or not 0 <= angle < projections:
        raise ValueError("MLP truth pilot requires 72 angles indexed 0..71")
    if not math.isclose(float(config["angle_step_deg"]), 5.0):
        raise ValueError("angle step must be 5 degrees")
    if protons < 1 or int(config.get("number_of_threads", 1)) != 1:
        raise ValueError("positive proton count and one thread are required")
    if config["world_material"] != "Air":
        raise ValueError("MLP truth pilot requires an Air world")
    if len(config["reference_planes"]) != 2:
        raise ValueError("two reference planes are required")
    reference_z = [float(item["z_mm"]) for item in config["reference_planes"]]
    if reference_z != [-110.0, 110.0]:
        raise ValueError("reference planes must be z=-110/+110 mm")
    focus_distance = abs(float(config["focus_z_mm"]) - float(config["source_z_mm"]))
    field = (
        np.asarray(config["source_size_mm"][:2], dtype=float)
        * abs(float(config["focus_z_mm"]))
        / focus_distance
    )
    if not np.allclose(field, config["isocenter_field_mm"], atol=1.0e-9):
        raise ValueError(f"inconsistent isocenter field: {field}")
    expected = {
        "PhaseSpaceIn.root",
        "PhaseSpaceOut.root",
        "PrimaryTrajectory.root",
    }
    if set(config["expected_root_files"]) != expected:
        raise ValueError("unexpected ROOT output list")
    validate_inserts(config)


def add_phantom(
    sim: gate.Simulation, config: dict, angle_index: int
) -> list[str]:
    mm = gate.g4_units.mm
    translation, rotation = angle_transform(config, angle_index)
    phantom = sim.add_volume("Tubs", name="HeterogeneousWaterCylinder")
    phantom.rmin = 0 * mm
    phantom.rmax = float(config["phantom_radius_mm"]) * mm
    phantom.dz = float(config["phantom_length_mm"]) / 2.0 * mm
    phantom.material = str(config["phantom_material"])
    phantom.translation = translation
    phantom.rotation = rotation
    phantom.set_max_step_size(float(config["phantom_max_step_mm"]) * mm)
    trajectory_volumes = [phantom.name]
    for item in config["inserts"]:
        insert = sim.add_volume("Tubs", name=str(item["name"]))
        insert.mother = phantom.name
        insert.rmin = 0 * mm
        insert.rmax = float(item["diameter_mm"]) / 2.0 * mm
        insert.dz = float(config["phantom_length_mm"]) / 2.0 * mm
        insert.material = str(item["material"])
        insert.translation = [
            float(item["center_local_xy_mm"][0]) * mm,
            float(item["center_local_xy_mm"][1]) * mm,
            0,
        ]
        insert.set_max_step_size(float(config["phantom_max_step_mm"]) * mm)
        trajectory_volumes.append(insert.name)
    return trajectory_volumes


def add_source(sim: gate.Simulation, config: dict, protons: int) -> None:
    source = sim.add_source("GenericSource", "mybeam")
    source.particle = "proton"
    source.energy.type = "mono"
    source.energy.mono = float(config["beam_energy_mev"]) * gate.g4_units.MeV
    source.position.type = "box"
    source.position.size = [
        float(value) * gate.g4_units.mm for value in config["source_size_mm"]
    ]
    source.position.translation = [
        0,
        0,
        float(config["source_z_mm"]) * gate.g4_units.mm,
    ]
    source.direction.type = "focused"
    source.direction.focus_point = [
        0,
        0,
        float(config["focus_z_mm"]) * gate.g4_units.mm,
    ]
    source.activity = protons * gate.g4_units.Bq


def add_reference_plane(
    sim: gate.Simulation, config: dict, item: dict, output_dir: Path
) -> None:
    mm = gate.g4_units.mm
    name = str(item["name"])
    plane = sim.add_volume("Box", f"Volume{name}")
    plane.size = [float(value) * mm for value in config["reference_size_mm"]]
    plane.translation = [0, 0, float(item["z_mm"]) * mm]
    plane.material = "Air"
    actor = sim.add_actor("PhaseSpaceActor", name)
    actor.attached_to = plane.name
    actor.attributes = list(config["reference_attributes"])
    actor.steps_to_store = "entering"
    actor.output_filename = str(output_dir / f"{name}.root")
    actor.filter = proton_filter()


def add_trajectory_actors(
    sim: gate.Simulation,
    config: dict,
    volume_names: list[str],
    output_dir: Path,
) -> list[tuple[str, Path]]:
    """Create one actor per physical volume.

    OpenGATE 10.1.0 accepts a volume list at configuration time, but its
    PhaseSpaceActor cannot start with steps_to_store="all" and multiple
    attached mother/daughter volumes.  Separate actors are therefore merged
    after the run into the stable public PrimaryTrajectory.root interface.
    """

    result = []
    for index, volume_name in enumerate(volume_names):
        actor_name = f"TrajectoryPart{index:02d}"
        path = output_dir / f".{actor_name}.root"
        actor = sim.add_actor("PhaseSpaceActor", actor_name)
        actor.attached_to = volume_name
        actor.attributes = list(config["trajectory_attributes"])
        actor.steps_to_store = "all"
        actor.output_filename = str(path)
        actor.filter = primary_proton_filter()
        result.append((actor_name, path))
    return result


def merge_trajectory_parts(
    parts: list[tuple[str, Path]], output_path: Path, tree_name: str
) -> dict:
    arrays_by_branch: dict[str, list[np.ndarray]] = {}
    input_rows = {}
    for part_tree, part_path in parts:
        if not part_path.is_file():
            raise RuntimeError(f"missing trajectory part: {part_path.name}")
        with uproot.open(part_path) as root_file:
            if part_tree not in root_file:
                raise RuntimeError(
                    f"missing tree {part_tree} in {part_path.name}"
                )
            tree = root_file[part_tree]
            missing = sorted(TRAJECTORY_BRANCHES - set(tree.keys()))
            if missing:
                raise RuntimeError(
                    f"{part_path.name}: missing branches {missing}"
                )
            input_rows[part_path.name] = int(tree.num_entries)
            arrays = read_branches_by_basket(
                tree, sorted(TRAJECTORY_BRANCHES)
            )
            for branch, values in arrays.items():
                arrays_by_branch.setdefault(branch, []).append(values)
    combined = {
        branch: np.concatenate(values)
        for branch, values in arrays_by_branch.items()
    }
    if not combined or len(combined["EventID"]) < 1:
        raise RuntimeError("all trajectory part trees are empty")
    with uproot.recreate(output_path) as root_file:
        root_file[tree_name] = combined
    for _, part_path in parts:
        part_path.unlink()
    return {
        "part_count": len(parts),
        "part_entries": input_rows,
        "merged_entries": int(len(combined["EventID"])),
    }


def build_simulation(
    config: dict,
    angle: int,
    protons: int,
    output_dir: Path,
    qc_dir: Path,
    seed: int,
    verbose: bool,
) -> tuple[gate.Simulation, list[tuple[str, Path]]]:
    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = seed
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.number_of_threads = 1
    sim.run_timing_intervals = [
        [angle * gate.g4_units.second, (angle + 1) * gate.g4_units.second]
    ]
    sim.volume_manager.add_material_database(MATERIAL_DATABASE)
    sim.world.material = str(config["world_material"])
    sim.world.size = [float(config["world_size_mm"]) * gate.g4_units.mm] * 3
    trajectory_volumes = add_phantom(sim, config, angle)
    add_source(sim, config, protons)
    for item in config["reference_planes"]:
        add_reference_plane(sim, config, item, output_dir)
    trajectory_parts = add_trajectory_actors(
        sim, config, trajectory_volumes, output_dir
    )
    sim.physics_manager.set_user_limits_particles("proton")
    sim.physics_manager.physics_list_name = str(config["physics_list"])
    statistics = sim.add_actor("SimulationStatisticsActor", "stat")
    statistics.output_filename = str(qc_dir / "protonct.txt")
    return sim, trajectory_parts


def inspect_reference(path: Path, tree_name: str) -> dict:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"missing tree {tree_name} in {path.name}")
        tree = root_file[tree_name]
        missing = sorted(REFERENCE_BRANCHES - set(tree.keys()))
        if missing:
            raise RuntimeError(f"{path.name}: missing branches {missing}")
        count = int(tree.num_entries)
        if count < 1:
            raise RuntimeError(f"empty ROOT tree: {path.name}")
        identity = read_branches_by_basket(
            tree, ["RunID", "EventID", "TrackID"]
        )
        if np.any(identity["RunID"] != 0):
            raise RuntimeError(f"{path.name}: local RunID is not zero")
        primary = identity["TrackID"] == 1
        floating = read_branches_by_basket(
            tree,
            [
                "KineticEnergy",
                "PreGlobalTime",
                "Position_X",
                "Position_Y",
                "Position_Z",
                "Direction_X",
                "Direction_Y",
                "Direction_Z",
            ],
            entry_stop=min(count, 20000),
        )
        if not all(np.isfinite(value).all() for value in floating.values()):
            raise RuntimeError(f"{path.name}: non-finite values")
    return {
        "tree": tree_name,
        "entries": count,
        "primary_entries": int(np.count_nonzero(primary)),
        "unique_primary_events": int(np.unique(identity["EventID"][primary]).size),
        "bytes": path.stat().st_size,
    }


def inspect_trajectory(path: Path, tree_name: str, maximum_step_mm: float) -> dict:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"missing tree {tree_name} in {path.name}")
        tree = root_file[tree_name]
        missing = sorted(TRAJECTORY_BRANCHES - set(tree.keys()))
        if missing:
            raise RuntimeError(f"{path.name}: missing branches {missing}")
        count = int(tree.num_entries)
        if count < 1:
            raise RuntimeError(f"empty ROOT tree: {path.name}")
        identity = read_branches_by_basket(
            tree, ["RunID", "EventID", "TrackID", "ParentID"]
        )
        if np.any(identity["RunID"] != 0):
            raise RuntimeError("trajectory local RunID is not zero")
        if np.any(identity["TrackID"] != 1) or np.any(identity["ParentID"] != 0):
            raise RuntimeError("trajectory actor retained a non-primary track")
        events, counts = np.unique(identity["EventID"], return_counts=True)
        floating_names = sorted(
            TRAJECTORY_BRANCHES
            - {"RunID", "EventID", "TrackID", "ParentID"}
        )
        floating = read_branches_by_basket(
            tree, floating_names, entry_stop=min(count, 100000)
        )
        if not all(np.isfinite(value).all() for value in floating.values()):
            raise RuntimeError("trajectory contains non-finite values")
        dx = floating["PostPosition_X"] - floating["PrePosition_X"]
        dy = floating["PostPosition_Y"] - floating["PrePosition_Y"]
        dz = floating["PostPosition_Z"] - floating["PrePosition_Z"]
        step = np.sqrt(dx * dx + dy * dy + dz * dz)
        sampled_max = float(step.max())
        if sampled_max > maximum_step_mm * 1.01 + 1.0e-6:
            raise RuntimeError(
                f"sampled trajectory step {sampled_max:g} mm exceeds limit"
            )
    return {
        "tree": tree_name,
        "entries": count,
        "unique_primary_events": int(len(events)),
        "steps_per_event_min": int(counts.min()),
        "steps_per_event_mean": float(counts.mean()),
        "steps_per_event_max": int(counts.max()),
        "sampled_step_length_min_mm": float(step.min()),
        "sampled_step_length_mean_mm": float(step.mean()),
        "sampled_step_length_max_mm": sampled_max,
        "sampled_rows_for_finiteness": int(len(step)),
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
    protons = int(
        args.protons_per_projection or config["protons_per_projection"]
    )
    seed = int(
        args.seed
        if args.seed is not None
        else int(config["random_seed"]) + args.angle
    )
    validate_config(config, args.angle, protons)
    output_dir = args.output_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = qc_dir / "run_metadata.json"
    record = {
        "status": "building",
        "scenario_id": config["scenario_id"],
        "angle_index": args.angle,
        "angle_degrees": args.angle * float(config["angle_step_deg"]),
        "protons_per_projection": protons,
        "random_seed": seed,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "opengate": package_version("opengate"),
        "opengate_core": package_version("opengate-core"),
        "config": str(config_path),
        "config_sha256": config_sha256(config_path),
        "process_id": os.getpid(),
        "insert_validation": validate_inserts(config),
    }
    metadata_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    clock = time.perf_counter()
    try:
        sim, trajectory_parts = build_simulation(
            config,
            args.angle,
            protons,
            output_dir,
            qc_dir,
            seed,
            args.verbose,
        )
        if args.build_only:
            record.update(
                status="build_only_completed",
                elapsed_seconds=time.perf_counter() - clock,
            )
            metadata_path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            print(f"MLP truth build-only PASS for angle {args.angle:03d}")
            return
        record["status"] = "running"
        metadata_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Starting MLP truth angle {args.angle:03d}, "
            f"protons={protons:,}, seed={seed}",
            flush=True,
        )
        sim.run()
        record.update(
            status="simulation_completed",
            simulation_completed_at=datetime.now().isoformat(timespec="seconds"),
            simulation_elapsed_seconds=time.perf_counter() - clock,
            trajectory_part_files=[
                {"tree": tree, "path": str(path)}
                for tree, path in trajectory_parts
            ],
        )
        metadata_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Simulation phase completed for MLP truth angle "
            f"{args.angle:03d} in "
            f"{record['simulation_elapsed_seconds']:.1f} s",
            flush=True,
        )
    except Exception:
        record.update(
            status="failed",
            failed_at=datetime.now().isoformat(timespec="seconds"),
            elapsed_seconds=time.perf_counter() - clock,
            traceback=traceback.format_exc(),
        )
        metadata_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
