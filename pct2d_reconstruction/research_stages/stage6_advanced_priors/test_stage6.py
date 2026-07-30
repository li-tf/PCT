#!/usr/bin/env python3
"""Lightweight Stage-6 tests; no formal reconstruction is executed."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from advanced_priors import (  # noqa: E402
    forward_gradient,
    negative_gradient_adjoint,
    negative_symmetric_gradient_adjoint,
    symmetric_gradient,
)
import run_stage6 as stage6  # noqa: E402


class Stage6Tests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20260713)
        self.u = rng.normal(size=(17, 19)).astype(np.float64)
        self.vx = rng.normal(size=self.u.shape)
        self.vz = rng.normal(size=self.u.shape)
        self.support = np.ones_like(self.u, dtype=bool)
        self.support[:2, :3] = False
        self.edge_x = self.support[:, :-1] & self.support[:, 1:]
        self.edge_z = self.support[:-1, :] & self.support[1:, :]
        self.rng = rng

    def test_gradient_adjoint(self) -> None:
        gx, gz = forward_gradient(
            np, self.u, self.edge_x, self.edge_z
        )
        px = self.rng.normal(size=self.u.shape)
        pz = self.rng.normal(size=self.u.shape)
        left = np.sum(gx * px + gz * pz)
        right = -np.sum(
            self.u
            * negative_gradient_adjoint(
                np,
                px,
                pz,
                edge_x=self.edge_x,
                edge_z=self.edge_z,
            )
        )
        self.assertLess(abs(left - right) / max(abs(left), 1.0), 1.0e-12)

    def test_symmetric_gradient_adjoint(self) -> None:
        exx, ezz, exz = symmetric_gradient(
            np, self.vx, self.vz, self.edge_x, self.edge_z
        )
        qxx = self.rng.normal(size=self.u.shape)
        qzz = self.rng.normal(size=self.u.shape)
        qxz = self.rng.normal(size=self.u.shape)
        left = np.sum(exx * qxx + ezz * qzz + 2.0 * exz * qxz)
        nx, nz = negative_symmetric_gradient_adjoint(
            np, qxx, qzz, qxz, self.edge_x, self.edge_z
        )
        right = -np.sum(self.vx * nx + self.vz * nz)
        self.assertLess(abs(left - right) / max(abs(left), 1.0), 1.0e-12)

    def test_candidate_grid_is_complete_and_unique(self) -> None:
        config = stage6.load_config()
        candidates = stage6.enumerate_candidates(config)
        self.assertEqual(len(candidates), 14)
        self.assertEqual(len({row["name"] for row in candidates}), 14)
        self.assertEqual(
            {row["method"] for row in candidates},
            {"tgv", "adaptive_tv", "directional_tv"},
        )

    def test_stage4_baseline_is_frozen(self) -> None:
        settings = stage6.stage4_settings()
        self.assertEqual(settings["epochs"], 5)
        self.assertEqual(settings["subsets"], 18)
        self.assertAlmostEqual(settings["relaxation"], 0.25)
        self.assertAlmostEqual(settings["regularization_weight"], 0.0125)


if __name__ == "__main__":
    unittest.main()
