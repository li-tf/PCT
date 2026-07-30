#!/usr/bin/env python3
"""Check the Windows OpenGATE environment and Stage 6A configuration."""

from __future__ import annotations

from importlib import metadata
import json
import platform
from pathlib import Path
import shutil
import sys

import matplotlib
import numpy
import opengate
import scipy

from run_mlic_replica import (
    build_simulation,
    enumerate_cases,
    enumerate_tasks,
    load_config,
    validate_config,
)


HERE = Path(__file__).resolve().parent


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    errors: list[str] = []
    config = load_config(HERE / "simulation_config.json")
    cases: list[dict] = []
    tasks: list[dict] = []
    try:
        validate_config(config)
        cases = enumerate_cases(config)
        tasks = enumerate_tasks(config)
        build_root = HERE / "qc" / "environment_build_only"
        build_simulation(
            config,
            tasks[0],
            1,
            int(config["random_seed"]),
            build_root / "data",
            build_root,
            False,
        )
    except Exception as error:
        errors.append(f"configuration/build: {error}")
    for package in ("opengate", "opengate-core"):
        if version(package) != "10.1.0":
            errors.append(f"expected {package} 10.1.0, found {version(package)}")
    material_database = opengate.utility.get_contrib_path() / "GateMaterials.db"
    material_text = material_database.read_text(
        encoding="utf-8", errors="replace"
    )
    required_materials = sorted(
        {"Water"}
        | {
            item["material"]
            for item in config["cases_per_energy"]
            if item["material"] is not None
        }
    )
    missing = [
        name for name in required_materials if f"{name}:" not in material_text
    ]
    if missing:
        errors.append(f"materials missing from GateMaterials.db: {missing}")
    drive = Path("D:/") if platform.system() == "Windows" else HERE
    free_gb = shutil.disk_usage(drive).free / 2**30
    if free_gb < float(config["minimum_free_gb"]):
        errors.append(
            f"free disk {free_gb:.1f} GB is below "
            f"{config['minimum_free_gb']} GB"
        )
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
        "matplotlib": matplotlib.__version__,
        "material_database": str(material_database),
        "required_materials": required_materials,
        "scenario_id": config["scenario_id"],
        "case_count": len(cases),
        "task_count": len(tasks),
        "protons_per_case": config["protons_per_case"],
        "total_protons": len(cases) * int(config["protons_per_case"]),
        "depth_bins": int(
            round(
                config["water_tank_size_mm"][2] / config["depth_bin_mm"]
            )
        ),
        "free_disk_gb": free_gb,
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
