#!/usr/bin/env python3
"""Validate the native-Windows D1 environment and geometry configuration."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy
import opengate
import scipy
import uproot

from run_angle import load_config, validate_config


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
    except Exception as error:
        errors.append(f"config: {error}")
    for package in ("opengate", "opengate-core"):
        if version(package) != "10.1.0":
            errors.append(f"expected {package} 10.1.0, found {version(package)}")
    material_db = opengate.utility.get_contrib_path() / "GateMaterials.db"
    material_text = material_db.read_text(encoding="utf-8", errors="replace")
    required_materials = ["Air", "Water", "Aluminium", "Silicon"]
    missing = [name for name in required_materials if f"{name}:" not in material_text]
    if missing:
        errors.append(f"materials missing from GateMaterials.db: {missing}")
    result = {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "opengate": version("opengate"), "opengate_core": version("opengate-core"),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "uproot": uproot.__version__,
        "material_database": str(material_db), "required_materials": required_materials,
        "scenario_id": config["scenario_id"], "projections": config["projections"],
        "protons_per_projection": config["protons_per_projection"],
        "tracker_thickness_mm": config["tracker_size_mm"][2],
        "tracker_z_mm": [item["z_mm"] for item in config["tracker_planes"]],
        "expected_root_files_per_angle": 6,
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
