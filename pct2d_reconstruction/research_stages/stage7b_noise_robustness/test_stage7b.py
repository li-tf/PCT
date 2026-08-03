"""Fast deterministic tests for Stage 7B."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(CODE / "iterative_reconstruction"),
    str(CODE / "research_stages/stage3_robust_weighting"),
    str(CODE / "research_stages/stage7_detector_effects"),
]

from stage7b_data import event_partitions  # noqa: E402
from stage7b_gpu import per_angle_weights, predict_analytic  # noqa: E402


class Stage7bTests(unittest.TestCase):
    def setUp(self):
        self.split = {
            "seed": 20260713,
            "modulus": 10,
            "test_remainder": 0,
            "validation_remainder": 1,
            "screen_modulus": 8,
            "screen_remainder": 0,
        }

    def test_event_split_is_reproducible_and_complete(self):
        events = np.arange(100_000, dtype=np.int64)
        first = event_partitions(events, 17, self.split)
        second = event_partitions(events, 17, self.split)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertFalse(np.any(first["train"] & first["validation"]))
        self.assertFalse(np.any(first["train"] & first["test"]))
        self.assertFalse(np.any(first["validation"] & first["test"]))
        self.assertTrue(
            np.all(first["train"] | first["validation"] | first["test"])
        )
        self.assertTrue(np.all(~first["screen"] | first["train"]))
        self.assertAlmostEqual(np.mean(first["train"]), 0.8, delta=0.01)
        self.assertAlmostEqual(np.mean(first["validation"]), 0.1, delta=0.01)
        self.assertAlmostEqual(np.mean(first["test"]), 0.1, delta=0.01)

    def test_analytic_sigma_is_positive(self):
        energy_axis = np.linspace(0, 230, 2301)
        ranges = 0.02 * energy_axis**1.75
        energy = np.array([30.0, 100.0, 200.0])
        sigma = predict_analytic(
            energy, energy_axis, ranges, 0.01, 0.02
        )
        self.assertTrue(np.isfinite(sigma).all())
        self.assertTrue(np.all(sigma > 0))

    def test_weight_clipping_and_effective_fraction(self):
        energy_axis = np.linspace(0, 230, 2301)
        ranges = 0.02 * energy_axis**1.75
        energy = np.linspace(30, 220, 2000)
        model = {
            "energy_mev": np.array([30.0, 220.0]),
            "sigma_mm": np.array([1.0, 2.0]),
            "minimum_sigma_mm": np.array(0.02),
        }
        config = {
            "noise_model": {"minimum_sigma_mm": 0.02},
            "weights": {
                "clip": [0.25, 4.0],
                "minimum_effective_fraction": 0.6,
            },
        }
        for kind in ("equal", "analytic", "empirical"):
            weights, sigma, ess = per_angle_weights(
                kind, energy, model, energy_axis, ranges, config
            )
            self.assertTrue(np.isfinite(weights).all())
            self.assertTrue(np.isfinite(sigma).all())
            self.assertGreaterEqual(float(weights.min()), 0.25)
            self.assertLessEqual(float(weights.max()), 4.0)
            self.assertGreaterEqual(ess, 0.6)


if __name__ == "__main__":
    unittest.main()
