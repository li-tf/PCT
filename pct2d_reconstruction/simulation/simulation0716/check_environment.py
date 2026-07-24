#!/usr/bin/env python3
"""Check the native-Windows Python/OpenGATE environment and package config."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path


HERE = Path(__file__).resolve().parent


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    config = json.loads((HERE / "simulation_config.json").read_text(encoding="utf-8"))
    import opengate  # noqa: F401
    import scipy  # noqa: F401
    import uproot  # noqa: F401

    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"OpenGATE: {version('opengate')}")
    print(f"opengate-core: {version('opengate-core')}")
    print(f"SciPy: {version('scipy')}")
    print(f"uproot: {version('uproot')}")
    print(f"Projections: {config['projections']}")
    print(f"Protons/projection: {config['protons_per_projection']:,}")
    total = int(config["projections"]) * int(config["protons_per_projection"])
    print(f"Total planned protons: {total:,}")
    area = 250.0 * 2.0
    fluence = float(config["protons_per_projection"]) / area
    print(f"Nominal fluence: {fluence:g} protons/mm^2/projection")
    if version("opengate") != "10.1.0" or version("opengate-core") != "10.1.0":
        print("WARNING: the validated baseline uses OpenGATE/opengate-core 10.1.0")
    if fluence != 900.0:
        raise SystemExit("ERROR: package is not configured for the paper fluence")
    print("Environment and configuration check passed.")


if __name__ == "__main__":
    main()
