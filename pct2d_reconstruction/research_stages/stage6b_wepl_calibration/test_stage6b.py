"""Fast numerical tests for the Stage-6B calibration contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(CODE_ROOT / "iterative_reconstruction")]

from calibration_model import canonical_model_hash, range_grid  # noqa: E402
from physics import energies_to_wepl_model, load_wepl_model  # noqa: E402
from run_stage6b import bb78_initial_derivative  # noqa: E402


def main() -> None:
    knots = np.arange(0.0, 231.0, 10.0)
    initial = bb78_initial_derivative(knots)
    if initial.shape != knots.shape or not np.isfinite(initial).all():
        raise AssertionError("invalid BB78 derivative initialization")
    energy = np.arange(0.0, 230.1, 0.1)
    # Constant positive dR/dE gives an exactly linear synthetic table.
    ranges = range_grid(np.zeros_like(knots), knots, energy)
    if not np.all(np.diff(ranges) > 0.0):
        raise AssertionError("range table is not strictly monotone")
    payload = {
        "model_name": "g4_water_calibrated",
        "energy_mev": energy.tolist(),
        "range_mm": ranges.tolist(),
    }
    payload["model_sha256"] = canonical_model_hash(payload)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        model = load_wepl_model("g4_water_calibrated", path)
        predicted = energies_to_wepl_model(
            model,
            np.array([200.0, 100.0, 0.0]),
            np.array([150.0, 90.0, 12.5]),
        )
    if not np.allclose(predicted, [50.0, 10.0, 12.5], atol=1.0e-5):
        raise AssertionError(predicted)
    print("Stage 6B numerical tests PASS")


if __name__ == "__main__":
    main()
