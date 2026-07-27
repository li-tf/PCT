#!/usr/bin/env python3
"""Lightweight Stage-4 tests; formal reconstruction is not executed."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_stage4 as stage4
from robust_gpu import RobustGpuMlpProjector


class Stage4Tests(unittest.TestCase):
    def test_huber_factors_match_definition(self) -> None:
        residual = np.array([-10.0, -3.0, 0.0, 2.0, 6.0], np.float32)
        factors = RobustGpuMlpProjector.huber_factors(
            residual, 3.0, np
        )
        np.testing.assert_allclose(
            factors,
            np.array([0.3, 1.0, 1.0, 1.0, 0.5], np.float32),
        )
        np.testing.assert_array_equal(
            RobustGpuMlpProjector.huber_factors(residual, None, np),
            np.ones_like(residual),
        )

    def test_invalid_huber_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RobustGpuMlpProjector.huber_factors(
                np.ones(3, np.float32), 0.0, np
            )

    def test_variant_identity_allows_epoch_resume(self) -> None:
        config = stage4.load_config()
        first = stage4.default_settings(config, epochs=3)
        second = stage4.default_settings(config, epochs=6)
        self.assertEqual(
            stage4.variant_name(first), stage4.variant_name(second)
        )
        self.assertEqual(
            stage4.canonical_hash(stage4.invariant_settings(first)),
            stage4.canonical_hash(stage4.invariant_settings(second)),
        )

    def test_regularization_decay(self) -> None:
        settings = {
            "regularization_weight": 0.012,
            "regularization_schedule": "decay",
            "regularization_decay": 0.5,
        }
        self.assertAlmostEqual(
            stage4.regularization_weight(settings, 0), 0.012
        )
        self.assertAlmostEqual(
            stage4.regularization_weight(settings, 2), 0.006
        )

    def test_locked_candidates_include_stage3_baseline(self) -> None:
        config = stage4.load_config()
        self.assertEqual(config["filter"], "baseline_3sigma")
        self.assertEqual(config["data_weight"], "equal")
        self.assertIn(0.25, config["relaxation_screen"]["initial_relaxations"])
        self.assertIn(
            0.0125, config["regularization_screen"]["weights"]
        )
        self.assertEqual(config["subset_screen"]["subsets"], [18, 36])


if __name__ == "__main__":
    unittest.main()
