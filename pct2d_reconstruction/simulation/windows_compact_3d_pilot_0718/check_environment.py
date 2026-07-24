#!/usr/bin/env python3
"""Validate the native-Windows compact 3-D environment and geometry."""

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

from run_angle import load_config, validate_config, validate_spheres


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
        layout = validate_spheres(config)
    except Exception as error:
        errors.append(f"config: {error}")
        layout = {}
    for package in ("opengate", "opengate-core"):
        if version(package) != "10.1.0":
            errors.append(f"expected {package} 10.1.0, found {version(package)}")
    material_db = opengate.utility.get_contrib_path() / "GateMaterials.db"
    material_text = material_db.read_text(encoding="utf-8", errors="replace")
    required = ["Air", "Water", "Aluminium", "SpineBone", "Lung", "A150_Tissue_Plastic"]
    missing = [name for name in required if f"{name}:" not in material_text]
    if missing:
        errors.append(f"materials missing from GateMaterials.db: {missing}")
    result = {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "opengate": version("opengate"), "opengate_core": version("opengate-core"),
        "numpy": numpy.__version__, "scipy": scipy.__version__, "uproot": uproot.__version__,
        "material_database": str(material_db), "required_materials": required,
        "scenario_id": config["scenario_id"], "projections": config["projections"],
        "protons_per_projection": config["protons_per_projection"],
        "total_protons": int(config["projections"]) * int(config["protons_per_projection"]),
        "isocenter_field_mm": config["isocenter_field_mm"],
        "sphere_layout": layout,
        "recommended_reconstruction": config["recommended_reconstruction"],
    }
    qc = HERE / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "environment_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
