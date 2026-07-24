#!/usr/bin/env python3
"""Shared experiment registry and filesystem helpers for the 2-D pCT pipeline."""

from __future__ import annotations

import json
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = CODE_ROOT.parent


def load_experiment(experiment: str) -> dict:
    experiment = str(experiment)
    path = CODE_ROOT / "experiments" / f"experiment{experiment}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown experiment {experiment!r}: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if str(config.get("experiment")) != experiment:
        raise ValueError(f"experiment id mismatch in {path}")
    for key in (
        "simulation_code",
        "simulation_data",
        "preprocessing_data",
        "reconstruction_data",
        "report_code",
    ):
        value = config.get("paths", {}).get(key)
        if not value:
            raise ValueError(f"missing paths.{key} in {path}")
        config["paths"][key] = str((REPOSITORY_ROOT / value).resolve())
    config["registry_path"] = str(path.resolve())
    return config


def path_for(config: dict, key: str) -> Path:
    return Path(config["paths"][key])


def ensure_empty_or_force(path: Path, force: bool, description: str) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"{description} already contains files: {path}; use --force to replace it"
        )
    path.mkdir(parents=True, exist_ok=True)

