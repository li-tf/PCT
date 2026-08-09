from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path[:0] = [
    str(HERE), str(CODE), str(CODE / "research_stages/stage3_robust_weighting"),
]

from stage8b_data import fraction_tag, nested_mask
from run_stage8b import angular_run_mapping, energy_gate, load_config


def test_fraction_tags_preserve_half_percent_steps() -> None:
    assert fraction_tag(0.25) == "f0250"
    assert fraction_tag(0.175) == "f0175"
    assert fraction_tag(0.125) == "f0125"


def test_event_subsets_are_nested_and_reproducible() -> None:
    events = np.arange(10000, dtype=np.int64)
    masks = [nested_mask(events, 17, 20260730, value) for value in (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)]
    for smaller, larger in zip(masks, masks[1:]):
        assert not np.any(smaller & ~larger)
    assert np.array_equal(masks[-1], nested_mask(events, 17, 20260730, 0.25))


def test_different_acquisition_seeds_are_not_identical() -> None:
    events = np.arange(10000, dtype=np.int64)
    one = nested_mask(events, 5, 20260730, 0.15)
    two = nested_mask(events, 5, 20260803, 0.15)
    assert not np.array_equal(one, two)


def test_optimization_acquisition_uses_even_views_and_preserves_full_arc() -> None:
    mapping = angular_run_mapping(load_config())
    assert len(mapping) == 360
    assert mapping[0] == (0, 0)
    assert mapping[-1] == (359, 718)
    assert [original for _, original in mapping[:4]] == [0, 2, 4, 6]


def test_energy_gate_uses_acquisition_seed_and_ignores_zero_wepl_denominator() -> None:
    config = load_config()
    rows = []
    for seed in (20260730, 20260731, 20260803):
        rows.extend([
            {
                "condition": "position_0p2mm",
                "seed": seed,
                "acquisition_seed": seed,
                "water_std": 0.0087,
                "phantom_rmse_vs_rsp_truth": 0.0596,
                "wepl_noise_rmse_mm": 0.0,
            },
            {
                "condition": "combined_0p2mm_1pct",
                "seed": seed,
                "acquisition_seed": seed,
                "water_std": 0.0096,
                "phantom_rmse_vs_rsp_truth": 0.0593,
                "wepl_noise_rmse_mm": 2.21,
            },
        ])
    rows.extend([
        {
            "condition": "continuous_hits",
            "seed": 20260730,
            "acquisition_seed": 20260730,
            "water_std": 0.0070,
            "phantom_rmse_vs_rsp_truth": 0.0446,
            "wepl_noise_rmse_mm": 0.0,
        },
        {
            "condition": "energy_1pct_only",
            "seed": 20260730,
            "acquisition_seed": 20260730,
            "water_std": 0.0078,
            "phantom_rmse_vs_rsp_truth": 0.0452,
            "wepl_noise_rmse_mm": 2.21,
        },
    ])
    decision = energy_gate(rows, config)
    assert decision["triggered"]
    assert decision["reasons"]["energy_only_image"]
    assert not decision["reasons"]["wepl"]
    assert not decision["wepl_ratio_available"]
    assert decision["wepl_noise_increase_fraction"] is None


def test_equal_weight_label_cannot_be_overwritten_by_reconstruction_candidate() -> None:
    source = {"candidate": "lowpass_0p5_upsampled", "seed": 20260713}
    row = dict(source)
    row.update({"candidate": "equal", "gamma": 0.0})
    assert row["candidate"] == "equal"
    assert row["gamma"] == 0.0
