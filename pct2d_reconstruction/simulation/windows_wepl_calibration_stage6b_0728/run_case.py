#!/usr/bin/env python3
"""Run one independent water-slab WEPL calibration case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import opengate as gate
import uproot


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "simulation_config.json"


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bb78_ranges(energies: np.ndarray) -> np.ndarray:
    electron_mass = 0.51099895
    proton_mass = 938.27208816
    radius_mm = 2.8179403262e-12
    density_cm3 = 3.343e23
    ionpot = 78.0e-6
    step = 0.001
    grid = np.arange(step, 240.0 + step, step)
    beta2 = 1.0 - (proton_mass / (grid + proton_mass)) ** 2
    k = 4 * np.pi * radius_mm**2 * electron_mass * density_cm3 / 1000.0
    stopping = k * (
        np.log(2 * electron_mass / ionpot * beta2 / (1 - beta2)) - beta2
    ) / beta2
    ranges = np.cumsum(step / stopping)
    return np.interp(energies, grid, ranges)


def enumerate_cases(config: dict) -> list[dict]:
    energies = np.asarray(config["energies_mev"], dtype=float)
    nominal = bb78_ranges(energies)
    split_lookup = {
        int(energy): split
        for split, values in config["split_by_energy"].items()
        for energy in values
    }
    cases = []
    for energy, range_mm in zip(energies, nominal):
        for fraction in config["thickness_fractions"]:
            thickness = round(float(range_mm) * float(fraction) * 2.0) / 2.0
            cases.append(
                {
                    "case_index": len(cases),
                    "case_id": f"e{int(energy):03d}_f{int(round(100*fraction)):02d}",
                    "energy_mev": float(energy),
                    "thickness_fraction": float(fraction),
                    "water_thickness_mm": max(0.5, thickness),
                    "split": split_lookup[int(energy)],
                }
            )
    return cases


def validate_config(config: dict) -> None:
    energies = [int(v) for v in config["energies_mev"]]
    assigned = [
        int(value)
        for values in config["split_by_energy"].values()
        for value in values
    ]
    if sorted(assigned) != sorted(energies) or len(set(assigned)) != len(assigned):
        raise ValueError("train/validation/test energies must partition energies_mev")
    if int(config["protons_per_case"]) < 1000:
        raise ValueError("protons_per_case is unexpectedly small")
    fractions = np.asarray(config["thickness_fractions"], dtype=float)
    if np.any(fractions <= 0) or np.any(fractions >= 0.8):
        raise ValueError("thickness fractions must be in (0, 0.8)")


def particle_filter():
    try:
        builder = gate.GateFilterBuilder()
    except AttributeError:
        builder = gate.actors.filters.GateFilterBuilder()
    return builder.ParticleName == "proton"


def add_actor(sim, volume_name: str, actor_name: str, output: Path, attributes):
    actor = sim.add_actor("PhaseSpaceActor", actor_name)
    actor.attached_to = volume_name
    actor.attributes = list(attributes)
    actor.steps_to_store = "entering"
    actor.output_filename = str(output / f"{actor_name}.root")
    actor.filter = particle_filter()


def build_simulation(config: dict, case: dict, protons: int, seed: int,
                     output: Path, qc: Path, verbose: bool):
    mm = gate.g4_units.mm
    mev = gate.g4_units.MeV
    thickness = float(case["water_thickness_mm"])
    plane_gap = float(config["reference_plane_gap_mm"])
    plane_t = float(config["reference_plane_thickness_mm"])
    slab_front = 0.0
    slab_back = thickness

    sim = gate.Simulation()
    sim.random_engine = "MersenneTwister"
    sim.random_seed = int(seed)
    sim.number_of_threads = 1
    sim.check_volumes_overlap = False
    sim.visu = False
    sim.g4_verbose = False
    sim.progress_bar = verbose
    sim.volume_manager.add_material_database(
        gate.utility.get_contrib_path() / "GateMaterials.db"
    )
    sim.world.material = config["world_material"]
    sim.world.size = [float(v) * mm for v in config["world_size_mm"]]

    slab = sim.add_volume("Box", "WaterSlab")
    slab.size = [
        float(config["slab_transverse_size_mm"][0]) * mm,
        float(config["slab_transverse_size_mm"][1]) * mm,
        thickness * mm,
    ]
    slab.translation = [0, 0, (slab_front + thickness / 2) * mm]
    slab.material = "Water"
    slab.set_max_step_size(float(config["water_max_step_mm"]) * mm)

    for name, z in (
        ("PhaseSpaceIn", slab_front - plane_gap - plane_t / 2),
        ("PhaseSpaceOut", slab_back + plane_gap + plane_t / 2),
    ):
        volume = sim.add_volume("Box", f"Volume{name}")
        volume.size = [
            float(config["reference_plane_size_mm"][0]) * mm,
            float(config["reference_plane_size_mm"][1]) * mm,
            plane_t * mm,
        ]
        volume.translation = [0, 0, z * mm]
        volume.material = config["world_material"]
        add_actor(sim, volume.name, name, output, config["output_attributes"])

    source = sim.add_source("GenericSource", "proton_beam")
    source.particle = "proton"
    source.n = int(protons)
    source.energy.type = "mono"
    source.energy.mono = float(case["energy_mev"]) * mev
    source.position.type = "box"
    source.position.size = [float(v) * mm for v in config["source_size_mm"]]
    source.position.translation = [
        0, 0, -float(config["source_to_slab_gap_mm"]) * mm
    ]
    source.direction.type = "momentum"
    source.direction.momentum = [0, 0, 1]

    sim.physics_manager.physics_list_name = config["physics_list"]
    sim.physics_manager.set_user_limits_particles("proton")
    stats = sim.add_actor("SimulationStatisticsActor", "Statistics")
    stats.output_filename = str(qc / "protonct.txt")
    return sim


def branch_array(tree, name: str) -> np.ndarray:
    """Read a branch basket-by-basket for cross-platform uproot stability."""
    branch = tree[name]
    chunks = [
        np.asarray(branch.basket(index).array(branch.interpretation))
        for index in range(branch.num_baskets)
    ]
    values = np.concatenate(chunks) if chunks else np.empty(0)
    if len(values) != tree.num_entries:
        raise RuntimeError(
            f"incomplete branch {name}: {len(values)} != {tree.num_entries}"
        )
    return values


def inspect(path: Path, tree_name: str) -> dict:
    with uproot.open(path) as root:
        tree = root[tree_name]
        entries = int(tree.num_entries)
        event_id = branch_array(tree, "EventID")
        track_id = branch_array(tree, "TrackID")
        direction_z = branch_array(tree, "Direction_Z")
        # A proton backscattered from the slab can re-enter the infinitesimal
        # reference plane.  Such recrossings are physical PhaseSpaceActor
        # records, not duplicate simulated events.  Calibration uses only
        # forward-going records and later keeps the earliest hit per EventID.
        primary = (track_id == 1) & (direction_z > 0.0)
        forward_entries = int(np.count_nonzero(primary))
        count = len(np.unique(event_id[primary]))
        duplicate = forward_entries - count
        energy = branch_array(tree, "KineticEnergy")
    duplicate_fraction = duplicate / max(count, 1)
    if count < 1 or duplicate_fraction > 0.001:
        raise RuntimeError(
            f"invalid forward primary records in {path}: "
            f"{count=}, {duplicate=}, {duplicate_fraction=:.3%}"
        )
    if not np.isfinite(energy).all():
        raise RuntimeError(f"non-finite energy in {path}")
    return {
        "entries": entries,
        "forward_primary_entries": forward_entries,
        "primary": count,
        "duplicate_forward_primary_hits": duplicate,
        "bytes": path.stat().st_size,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--write-cases-json", type=Path)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--protons", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_config(config)
    cases = enumerate_cases(config)
    if args.write_cases_json:
        args.write_cases_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_cases_json.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
        return
    if args.list_cases:
        print(json.dumps(cases))
        return
    if args.case_index is None or args.output_dir is None or args.qc_dir is None:
        raise SystemExit("--case-index, --output-dir and --qc-dir are required")
    case = cases[args.case_index]
    output, qc = args.output_dir.resolve(), args.qc_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    qc.mkdir(parents=True, exist_ok=True)
    protons = int(args.protons or config["protons_per_case"])
    seed = int(config["random_seed"]) + int(case["case_index"])
    record = {
        **case,
        "status": "building",
        "protons": protons,
        "seed": seed,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "opengate": version("opengate"),
        "opengate_core": version("opengate-core"),
        "pid": os.getpid(),
    }
    metadata_path = qc / "case_metadata.json"
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    sim = build_simulation(config, case, protons, seed, output, qc, args.verbose)
    if args.build_only:
        print(json.dumps({"status": "BUILD_PASS", "case": case}, indent=2))
        return
    sim.run()
    roots = {
        name: inspect(output / f"{name}.root", name)
        for name in ("PhaseSpaceIn", "PhaseSpaceOut")
    }
    record.update(
        status="completed",
        stopped_at=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds=time.perf_counter() - start,
        roots=roots,
        primary_survival=roots["PhaseSpaceOut"]["primary"] / roots["PhaseSpaceIn"]["primary"],
    )
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    (qc / "completed.flag").write_text(f"{case['case_id']}\n", encoding="ascii")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
