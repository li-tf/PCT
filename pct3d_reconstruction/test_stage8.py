"""CPU unit tests for Stage 8; CUDA checks live in --action operator-smoke."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    from .preprocessing3d import load_completed_run
except ImportError:
    from preprocessing3d import load_completed_run

try:
    from .io3d import (
        read_pairs,
        read_partition,
        read_volume,
        write_pairs,
        write_partition,
        write_volume,
    )
    from .physics3d import (
        build_truth,
        identity_hash,
        load_mlic,
        mlp_position_cpu,
        partition_codes,
        ray_finite_cylinder_interval,
        scanner_to_object,
        support_mask,
    )
except ImportError:  # Direct execution from the repository root.
    from io3d import (
        read_pairs,
        read_partition,
        read_volume,
        write_pairs,
        write_partition,
        write_volume,
    )
    from physics3d import (
        build_truth,
        identity_hash,
        load_mlic,
        mlp_position_cpu,
        partition_codes,
        ray_finite_cylinder_interval,
        scanner_to_object,
        support_mask,
    )


HERE = Path(__file__).resolve().parent
REPO = HERE.parent


class Stage8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((HERE / "stage8_config.json").read_text())

    def test_split_repeatable_and_partitioned(self):
        events = np.arange(20000, dtype=np.int64)
        first = partition_codes(17, events, 20260713)
        second = partition_codes(17, events, 20260713)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(set(np.unique(first)), {0, 1, 2})
        self.assertFalse(np.array_equal(identity_hash(17, events, 1), identity_hash(17, events, 2)))

    def test_finite_cylinder_side_cap_tangent_and_miss(self):
        positions = np.array(
            [[0, 0, -60], [0, -20, -60], [50, 0, -60], [60, 0, -60]], dtype=float
        )
        directions = np.array(
            [[0, 0, 1], [0, 0.2, 1], [0, 0, 1], [0, 0, 1]], dtype=float
        )
        enter, leave, valid = ray_finite_cylinder_interval(positions, directions, 50, 15)
        self.assertTrue(valid[0])
        self.assertAlmostEqual(enter[0], 10.0, places=6)
        self.assertTrue(valid[1])  # enters through the axial end cap
        self.assertFalse(valid[2])  # tangent has zero in-support path length
        self.assertFalse(valid[3])
        self.assertGreater(leave[1], enter[1])

    def test_pair_partition_and_volume_roundtrip(self):
        rng = np.random.default_rng(4)
        pairs = rng.normal(size=(31, 5, 3)).astype(np.float32)
        codes = rng.integers(0, 3, size=31, dtype=np.uint8)
        volume = rng.normal(size=(7, 5, 9)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_pairs(root / "pairs.mhd", pairs)
            write_partition(root / "split.npz", codes)
            write_volume(root / "volume.mhd", volume, (0.5, 0.6, 0.7), (-2, -1, -3))
            self.assertTrue(np.array_equal(read_pairs(root / "pairs.mhd"), pairs))
            self.assertTrue(np.array_equal(read_partition(root / "split.npz"), codes))
            restored, spacing, origin = read_volume(root / "volume.mhd")
            self.assertTrue(np.array_equal(restored, volume))
            self.assertEqual(spacing, [0.5, 0.6, 0.7])
            self.assertEqual(origin, [-2, -1, -3])

    def test_empty_interrupted_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("pairs.mhd", "events.npy", "split.npz", "screen.npz", "qc.json")]
            for path in paths:
                path.touch()
            self.assertIsNone(load_completed_run(*paths))

    def test_mlp_straight_symmetric(self):
        entry = np.array([2.0, -1.0, -50.0])
        exit_ = np.array([2.0, -1.0, 50.0])
        direction = np.array([0.0, 0.0, 1.0])
        point = mlp_position_cpu(0.0, entry, exit_, direction, direction)
        self.assertTrue(np.allclose(point, [2.0, -1.0, 0.0], atol=1e-8))

    def test_object_rotation_zero_and_ninety(self):
        reference = np.array([[12.0, 7.0, 18.0]])
        scanner_90 = np.array([[18.0, 7.0, -12.0]])
        self.assertTrue(np.allclose(scanner_to_object(reference, 0), reference))
        self.assertTrue(np.allclose(scanner_to_object(scanner_90, 90), reference, atol=1e-12))

    def test_truth_geometry_and_support(self):
        simulation = json.loads(
            (REPO / "pct2d_reconstruction/simulation/windows_compact_3d_pilot_0718/simulation_config.json").read_text()
        )
        mlic = load_mlic(
            REPO / "pct2d_reconstruction/research_stages/stage6a_mlic_reference/qc/mlic_reference_200mev.csv"
        )
        truth = build_truth(self.config, simulation, mlic, supersample=1)
        support = support_mask(self.config)
        self.assertEqual(truth.shape, (240, 80, 240))
        self.assertEqual(np.count_nonzero(truth[~support]), 0)
        self.assertTrue(np.isfinite(truth).all())
        self.assertGreater(float(truth.max()), 2.0)


if __name__ == "__main__":
    unittest.main()
