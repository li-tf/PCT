"""Fast numerical tests for Stage-7 tracker processing."""

from __future__ import annotations

import numpy as np

from detector_processing import extrapolate, line_from_hits, variant_seed


def main() -> None:
    first = np.array([[1.0, 2.0, -160.0], [-3.0, 4.0, 130.0]])
    second = np.array([[4.0, 2.0, -130.0], [-1.0, 5.0, 160.0]])
    position, direction = line_from_hits(first, second, -110.0)
    expected_x = 6.0
    if not np.isclose(position[0, 0], expected_x):
        raise AssertionError((position[0, 0], expected_x))
    if not np.allclose(np.linalg.norm(direction, axis=1), 1.0):
        raise AssertionError("directions are not normalized")
    repeated = extrapolate(second, direction, -110.0)
    if not np.allclose(repeated, position):
        raise AssertionError("extrapolation is inconsistent")
    if variant_seed(20260713, 4, "continuous_hits") != variant_seed(
        20260713, 4, "continuous_hits"
    ):
        raise AssertionError("variant seed is not deterministic")
    if variant_seed(20260713, 4, "continuous_hits") == variant_seed(
        20260713, 4, "energy_1pct"
    ):
        raise AssertionError("variant seed does not distinguish variants")
    print("Stage 7 numerical tests PASS")


if __name__ == "__main__":
    main()
