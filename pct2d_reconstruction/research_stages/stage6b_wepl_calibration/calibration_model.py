"""Monotone positive-derivative water range model for Stage 6B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid


def range_grid(
    log_derivative_knots: np.ndarray,
    knot_energy_mev: np.ndarray,
    output_energy_mev: np.ndarray,
) -> np.ndarray:
    """Integrate a positive piecewise-log-linear dR/dE model."""

    logq = np.interp(output_energy_mev, knot_energy_mev, log_derivative_knots)
    derivative = np.exp(logq)
    return cumulative_trapezoid(
        derivative, output_energy_mev, initial=0.0
    )


def wepl(
    log_derivative_knots: np.ndarray,
    knot_energy_mev: np.ndarray,
    output_energy_mev: np.ndarray,
    energy_in: np.ndarray,
    energy_out: np.ndarray,
) -> np.ndarray:
    ranges = range_grid(
        log_derivative_knots, knot_energy_mev, output_energy_mev
    )
    return np.interp(energy_in, output_energy_mev, ranges) - np.interp(
        energy_out, output_energy_mev, ranges
    )


def canonical_model_hash(payload: dict) -> str:
    canonical = json.dumps(
        {
            "model_name": payload["model_name"],
            "energy_mev": payload["energy_mev"],
            "range_mm": payload["range_mm"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def write_model(path: Path, payload: dict) -> None:
    payload = dict(payload)
    payload["model_sha256"] = canonical_model_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
