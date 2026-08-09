"""CPU-only tests for Stage 8C decision helpers and synthetic truth geometry."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from physics3d import load_mlic, scanner_to_object, support_mask
from run_stage8c import (
    convergence_score,
    object_to_scanner,
    simple_image_metrics,
    synthetic_gate,
    synthetic_stable_pass,
    synthetic_scenarios,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class Stage8CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage8 = json.loads((HERE / "stage8_config.json").read_text(encoding="utf-8"))
        cls.simulation = json.loads(
            (REPO / cls.stage8["simulation_config"]).read_text(encoding="utf-8")
        )
        cls.mlic = load_mlic(REPO / cls.stage8["mlic_reference"])

    def test_scanner_object_roundtrip_multiple_angles(self):
        points = np.array([[12.0, 7.0, 18.0], [-25.0, -7.0, 12.0]])
        for angle in (0, 37, 90, 179, 271):
            restored = scanner_to_object(object_to_scanner(points, angle), angle)
            self.assertTrue(np.allclose(restored, points, atol=1e-12))

    def test_synthetic_scenarios_are_finite_and_supported(self):
        scenarios = synthetic_scenarios(self.stage8, self.simulation, self.mlic)
        self.assertEqual([item[0] for item in scenarios], [
            "uniform_water", "center_air", "center_high_rsp",
            "offaxis_high_rsp", "five_sphere",
        ])
        support = support_mask(self.stage8)
        for _, truth, _ in scenarios:
            self.assertTrue(np.isfinite(truth).all())
            self.assertEqual(np.count_nonzero(truth[~support]), 0)

    def test_exact_truth_has_zero_simple_error(self):
        _, truth, scene = synthetic_scenarios(
            self.stage8, self.simulation, self.mlic
        )[-1]
        metrics = simple_image_metrics(
            truth, truth, self.stage8, scene, self.mlic
        )
        self.assertAlmostEqual(metrics["phantom_rmse"], 0.0)
        self.assertAlmostEqual(metrics["water_bias"], 0.0)
        self.assertAlmostEqual(metrics["large_sphere_mape_percent"], 0.0, places=4)

    def test_synthetic_gate_and_score(self):
        stage8c = json.loads((HERE / "stage8c_config.json").read_text(encoding="utf-8"))
        passing = {
            "matched_validation_wepl_rmse_mm": 0.009,
            "water_bias": -0.002,
            "large_sphere_mape_percent": 0.9,
            "air_absolute_rsp_error": 0.04,
        }
        self.assertTrue(synthetic_gate(passing, stage8c))
        self.assertLessEqual(convergence_score(passing, stage8c), 1.0)
        failing = dict(passing, large_sphere_mape_percent=1.5)
        self.assertFalse(synthetic_gate(failing, stage8c))
        self.assertAlmostEqual(convergence_score(failing, stage8c), 1.5)
        self.assertTrue(synthetic_stable_pass([passing, passing], stage8c))
        self.assertFalse(synthetic_stable_pass([passing], stage8c))
        self.assertFalse(synthetic_stable_pass([passing, failing], stage8c))


if __name__ == "__main__":
    unittest.main()
