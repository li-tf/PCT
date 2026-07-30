"""Fast unit tests for Stage 5; no formal data are modified."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from inhomogeneous_mlp import (  # noqa: E402
    catalog_rscp, energy_profile, material_catalog, published_rscp,
    relative_scattering_power, stopping_power_water, truth_maps,
)


class Stage5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "stage5_config.json").read_text())

    def test_water_rscp_is_near_one(self):
        self.assertAlmostEqual(relative_scattering_power("Water"), 1.0, delta=0.015)

    def test_material_catalog_is_finite_positive(self):
        catalog = material_catalog(self.config)
        self.assertEqual(set(catalog), set(self.config["materials"]))
        self.assertTrue(np.isfinite([v["rscp"] for v in catalog.values()]).all())
        self.assertTrue(all(v["rscp"] > 0 for v in catalog.values()))

    def test_published_mapping_is_continuous(self):
        split = self.config["mapping"]["published_break_rsp"]
        values = published_rscp(np.array([split - 1e-6, split + 1e-6]), self.config["mapping"])
        self.assertLess(abs(float(values[1] - values[0])), 1e-4)

    def test_catalog_mapping_hits_anchors(self):
        catalog = material_catalog(self.config)
        rsp = np.array([v["rsp"] for v in catalog.values()])
        expected = np.array([v["rscp"] for v in catalog.values()])
        actual = catalog_rscp(rsp, catalog)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_energy_profile_endpoints_and_finiteness(self):
        rsp = np.ones((2, 101), np.float64)
        energy = energy_profile(np.array([200., 200.]), np.array([100., 110.]), rsp, 1.0)
        self.assertTrue(np.isfinite(energy).all())
        np.testing.assert_allclose(energy[:, 0], 200.0, atol=1e-8)
        np.testing.assert_allclose(energy[:, -1], [100., 110.], atol=1e-8)
        self.assertTrue(np.all(np.diff(energy, axis=1) < 0))

    def test_stopping_power_positive(self):
        values = stopping_power_water(np.array([10., 100., 200.]))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.all(values > 0))

    def test_truth_geometry_materials_present(self):
        maps = truth_maps(self.config, 0.5)
        self.assertEqual(maps.rsp.shape, (400, 400))
        self.assertEqual(
            set(np.unique(maps.material)),
            {"Air", "Lung", "Water", "A150_Tissue_Plastic", "SpineBone", "Aluminium"},
        )


if __name__ == "__main__":
    unittest.main()

