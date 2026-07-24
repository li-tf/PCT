#!/usr/bin/env python3
"""Validate the native-Windows environment and MLP truth geometry."""

from __future__ import annotations

from importlib import metadata
import json
import platform
from pathlib import Path
import sys

import numpy
import opengate
import scipy
import uproot

from run_angle import (
    build_simulation,
    load_config,
    validate_config,
    validate_inserts,
)


HERE = Path(__file__).resolve().parent


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    errors = []
    config = load_config(HERE / "simulation_config.json")
    try:
        validate_config(config, 0, int(config["protons_per_projection"]))
        layout = validate_inserts(config)
        build_dir = HERE / "qc" / "environment_build_only"
        simulation, trajectory_parts = build_simulation(
            config,
            0,
            1,
            build_dir / "data",
            build_dir,
            int(config["random_seed"]),
            False,
        )
        actors = [
            simulation.actor_manager.get_actor(name)
            for name, _ in trajectory_parts
        ]
        trajectory_setup = {
            "actor_count": len(actors),
            "attached_volumes": [
                actor.attached_to for actor in actors
            ],
            "steps_to_store": sorted(
                {actor.steps_to_store for actor in actors}
            ),
            "attributes": list(actors[0].attributes),
            "post_run_merge_tree": config["trajectory_actor_name"],
        }
    except Exception as error:
        errors.append(f"config/build: {error}")
        layout = {}
        trajectory_setup = {}
    for package in ("opengate", "opengate-core"):
        if version(package) != "10.1.0":
            errors.append(
                f"expected {package} 10.1.0, found {version(package)}"
            )
    material_database = opengate.utility.get_contrib_path() / "GateMaterials.db"
    material_text = material_database.read_text(
        encoding="utf-8", errors="replace"
    )
    required_materials = [
        "Air",
        "Water",
        "Lung",
        "A150_Tissue_Plastic",
        "SpineBone",
        "Aluminium",
    ]
    missing = [
        name
        for name in required_materials
        if f"{name}:" not in material_text
    ]
    if missing:
        errors.append(f"materials missing from GateMaterials.db: {missing}")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "opengate": version("opengate"),
        "opengate_core": version("opengate-core"),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "uproot": uproot.__version__,
        "material_database": str(material_database),
        "required_materials": required_materials,
        "scenario_id": config["scenario_id"],
        "projections": config["projections"],
        "protons_per_projection": config["protons_per_projection"],
        "total_protons": (
            int(config["projections"])
            * int(config["protons_per_projection"])
        ),
        "insert_layout": layout,
        "trajectory_setup": trajectory_setup,
    }
    qc = HERE / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "environment_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
