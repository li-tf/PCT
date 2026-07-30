#!/usr/bin/env python3
"""Validate the Stage-6B Windows/OpenGATE environment and geometry."""

from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

import numpy
import opengate
import scipy
import uproot

from run_case import build_simulation, enumerate_cases, load_config, validate_config


HERE = Path(__file__).resolve().parent


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main():
    errors = []
    config = load_config(HERE / "simulation_config.json")
    cases = []
    try:
        validate_config(config)
        cases = enumerate_cases(config)
        root = HERE / "qc" / "environment_build_only"
        root.mkdir(parents=True, exist_ok=True)
        build_simulation(
            config, cases[0], 1, int(config["random_seed"]),
            root / "data", root, False,
        )
    except Exception as error:
        errors.append(f"configuration/build: {error}")
    for package in ("opengate", "opengate-core"):
        if version(package) != "10.1.0":
            errors.append(f"expected {package} 10.1.0, found {version(package)}")
    free_gb = shutil.disk_usage(Path("D:/") if sys.platform == "win32" else HERE).free / 2**30
    if free_gb < float(config["minimum_free_gb"]):
        errors.append(f"only {free_gb:.1f} GB free")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "python": sys.version.split()[0],
        "opengate": version("opengate"),
        "opengate_core": version("opengate-core"),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "uproot": uproot.__version__,
        "cases": len(cases),
        "total_protons": len(cases) * int(config["protons_per_case"]),
        "free_disk_gb": free_gb,
    }
    qc = HERE / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "environment_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
