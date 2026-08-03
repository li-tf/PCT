"""Fast deterministic tests for Stage 7C."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [
    str(HERE),
    str(CODE),
    str(CODE / "research_stages/stage3_robust_weighting"),
]

from stage7c_data import filter_mask, fraction_tag, nested_mask


class Stage7CTests(unittest.TestCase):
    def test_nested_and_repeatable(self) -> None:
        events = np.arange(250_000, dtype=np.int64)
        masks = [nested_mask(events, 19, 20260730, f) for f in (0.1, 0.25, 0.5, 1.0)]
        self.assertTrue(np.all(~masks[0] | masks[1]))
        self.assertTrue(np.all(~masks[1] | masks[2]))
        self.assertTrue(np.all(~masks[2] | masks[3]))
        self.assertTrue(
            np.array_equal(masks[1], nested_mask(events, 19, 20260730, 0.25))
        )
        self.assertGreater(np.count_nonzero(masks[0]), 24_000)
        self.assertLess(np.count_nonzero(masks[0]), 26_000)

    def test_seed_changes_realization(self) -> None:
        events = np.arange(10_000, dtype=np.int64)
        first = nested_mask(events, 0, 20260730, 0.25)
        second = nested_mask(events, 0, 20260731, 0.25)
        self.assertFalse(np.array_equal(first, second))

    def test_fraction_tags(self) -> None:
        self.assertEqual(fraction_tag(0.5), "f050")
        self.assertEqual(fraction_tag(0.25), "f025")
        self.assertEqual(fraction_tag(0.1), "f010")

    def test_filter_mask_matches_paircuts(self) -> None:
        from preprocessing.paircuts import filter_pairs

        rng = np.random.default_rng(7)
        n = 4000
        pairs = np.zeros((n, 5, 3), dtype=np.float32)
        pairs[:, 0, 0] = rng.uniform(-100, 100, n)
        pairs[:, 0, 1] = rng.uniform(-0.8, 0.8, n)
        pairs[:, 0, 2] = -110
        pairs[:, 1, 0] = pairs[:, 0, 0] + rng.normal(0, 0.5, n)
        pairs[:, 1, 1] = pairs[:, 0, 1] + rng.normal(0, 0.02, n)
        pairs[:, 1, 2] = 110
        pairs[:, 2, 2] = pairs[:, 3, 2] = 1
        pairs[:, 2, 0] = rng.normal(0, 0.002, n)
        pairs[:, 3, 0] = pairs[:, 2, 0] + rng.normal(0, 0.004, n)
        pairs[:, 4, 0] = 200
        pairs[:, 4, 1] = 150 + rng.normal(0, 3, n)
        selected = filter_mask(pairs)
        expected, _ = filter_pairs(pairs)
        self.assertTrue(np.array_equal(pairs[selected], expected))


if __name__ == "__main__":
    unittest.main()
