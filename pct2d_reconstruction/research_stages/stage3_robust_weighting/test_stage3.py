#!/usr/bin/env python3
"""Lightweight Stage-3 unit tests; no formal data products are written."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robust_models import (  # noqa: E402
    apply_filter,
    fit_filter,
    fit_noise_model,
    normalize_and_clip_weights,
)
from stage3_io import (  # noqa: E402
    effective_sample_size,
    load_json,
    partition_masks,
    read_packed_mask,
    write_packed_mask,
)


CONFIG = load_json(HERE / "stage3_config.json")


class Stage3Tests(unittest.TestCase):
    def test_split_is_deterministic_and_complete(self) -> None:
        first = partition_masks(10003, 719, CONFIG["split"])
        second = partition_masks(10003, 719, CONFIG["split"])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertFalse(np.any(first["train"] & first["validation"]))
        self.assertFalse(np.any(first["train"] & first["test"]))
        self.assertFalse(np.any(first["validation"] & first["test"]))
        np.testing.assert_array_equal(
            first["train"] | first["validation"] | first["test"],
            np.ones(10003, dtype=bool),
        )

    def test_packed_mask_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        mask = rng.random(1003) < 0.37
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.bin"
            write_packed_mask(path, mask)
            restored = read_packed_mask(path, len(mask))
        np.testing.assert_array_equal(mask, restored)

    def test_robust_filters_reject_contamination(self) -> None:
        rng = np.random.default_rng(9)
        count = 30000
        features = rng.normal(size=(count, 3))
        features[:, 0] = 20.0 + 1.5 * features[:, 0]
        features[:, 1:] *= 0.002
        features[-1500:, 0] += 20.0
        features[-1500:, 1:] += 0.025
        cells = rng.integers(0, 250, size=count)
        inside = np.ones(count, dtype=bool)
        train = partition_masks(count, 8, CONFIG["split"])["train"]
        for name in CONFIG["filtering"]["candidates"]:
            model = fit_filter(
                name,
                features,
                cells,
                inside,
                train,
                CONFIG["filtering"],
            )
            selected, distance = apply_filter(
                model,
                features,
                cells,
                inside,
                CONFIG["filtering"],
            )
            self.assertTrue(np.isfinite(distance).all())
            self.assertGreater(np.mean(selected[:-1500]), 0.90)
            self.assertLess(np.mean(selected[-1500:]), 0.10)

    def test_noise_model_and_weights_are_finite(self) -> None:
        rng = np.random.default_rng(12)
        energy = rng.uniform(50.0, 180.0, 500000)
        sigma = 0.3 + 0.004 * (180.0 - energy)
        residual = rng.normal(scale=sigma)
        local = dict(CONFIG["noise_model"])
        local["minimum_bin_rows"] = 100
        model = fit_noise_model(energy, residual, local)
        predicted = model.predict(np.array([0.0, 80.0, 200.0]))
        self.assertTrue(np.isfinite(predicted).all())
        selected = np.ones(len(energy), dtype=bool)
        weights = normalize_and_clip_weights(
            1.0 / model.predict(energy) ** 2,
            selected,
            tuple(CONFIG["weights"]["clip"]),
        )
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreaterEqual(weights.min(), CONFIG["weights"]["clip"][0])
        self.assertLessEqual(weights.max(), CONFIG["weights"]["clip"][1])
        self.assertGreater(
            effective_sample_size(weights) / len(weights),
            CONFIG["weights"]["minimum_effective_fraction"],
        )


if __name__ == "__main__":
    unittest.main()
