"""Rebuild Stage-7 EventID alignment and write nested fluence subsets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def filter_mask(pairs: np.ndarray) -> np.ndarray:
    """Return the exact local-3sigma selection used by Stage 7."""
    from preprocessing import paircuts

    inside, pixel, angle_x, angle_y, energy_loss = paircuts.pair_features(pairs)
    pixel_inside = pixel[inside]
    count = np.bincount(
        pixel_inside, minlength=paircuts.GRID_SIZE[0] * paircuts.GRID_SIZE[1]
    ).astype(np.uint32)
    sum_energy = np.zeros(len(count), dtype=np.float32)
    sum_energy_sq = np.zeros(len(count), dtype=np.float32)
    sum_angle_sq = np.zeros(len(count), dtype=np.float32)
    np.add.at(sum_energy, pixel_inside, energy_loss[inside].astype(np.float32))
    np.add.at(
        sum_energy_sq,
        pixel_inside,
        np.square(energy_loss[inside]).astype(np.float32),
    )
    np.add.at(
        sum_angle_sq,
        pixel_inside,
        np.square(angle_x[inside]).astype(np.float32),
    )
    np.add.at(
        sum_angle_sq,
        pixel_inside,
        np.square(angle_y[inside]).astype(np.float32),
    )
    occupied = count > 0
    mean_energy = np.zeros(len(count), dtype=np.float32)
    sigma_energy = np.zeros(len(count), dtype=np.float32)
    sigma_angle = np.zeros(len(count), dtype=np.float32)
    mean_energy[occupied] = sum_energy[occupied] / count[occupied]
    variance = (
        sum_energy_sq[occupied] / count[occupied]
        - np.square(mean_energy[occupied])
    )
    sigma_energy[occupied] = np.sqrt(np.maximum(variance, 0.0))
    sigma_angle[occupied] = np.sqrt(
        sum_angle_sq[occupied] / (2.0 * count[occupied])
    )
    sigma_energy *= paircuts.ENERGY_SIGMA_CUT
    sigma_angle[sigma_angle == 0.0] = 1.0
    sigma_angle *= paircuts.ANGLE_SIGMA_CUT
    selected = np.zeros(len(pairs), dtype=bool)
    candidate = np.flatnonzero(inside)
    cell = pixel[candidate]
    selected[candidate] = (
        (angle_x[candidate] <= sigma_angle[cell])
        & (angle_y[candidate] <= sigma_angle[cell])
        & (
            np.abs(energy_loss[candidate] - mean_energy[cell])
            <= sigma_energy[cell]
        )
    )
    return selected


def nested_mask(events: np.ndarray, run_id: int, seed: int, fraction: float) -> np.ndarray:
    from stage3_io import splitmix64

    if fraction >= 1.0:
        return np.ones(len(events), dtype=bool)
    if not 0.0 < fraction < 1.0:
        raise ValueError(fraction)
    # Integer buckets make thresholds portable and exactly nested.
    bucket = splitmix64(np.asarray(events, dtype=np.uint64), run_id, seed)
    bucket %= np.uint64(1_000_000)
    return bucket < np.uint64(round(fraction * 1_000_000))


def fraction_tag(fraction: float) -> str:
    return f"f{int(round(100 * fraction)):03d}"


def group_root(
    output: Path, condition: str, seed: int, fraction: float
) -> Path:
    return output / condition / f"seed_{seed}" / fraction_tag(fraction)


def rebuild_filtered(
    run_id: int,
    raw_root: Path,
    condition: dict[str, Any],
    stage7_seed: int,
    reference_z: tuple[float, float],
    model_path: Path,
    air_slope: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Stage-7 converted pairs and their EventIDs in identical order."""
    from detector_processing import (
        hit_pairs,
        ideal_pairs,
        load_run,
        variant_seed,
    )
    from run_stage7 import direct_wepl, locate_run

    run_dir = locate_run(raw_root, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"missing raw D1 angle {run_id:04d}")
    planes = load_run(run_dir)
    if condition["track_state"] == "reference_planes":
        events, pairs = ideal_pairs(
            planes["PhaseSpaceIn"], planes["PhaseSpaceOut"]
        )
    else:
        name = str(condition["stage7_name"])
        events, pairs, _ = hit_pairs(
            planes,
            condition,
            variant_seed(stage7_seed, run_id, name),
            reference_z,
        )
    physical = (
        np.isfinite(pairs[:, 4, 0])
        & np.isfinite(pairs[:, 4, 1])
        & (pairs[:, 4, 0] > 0.0)
        & (pairs[:, 4, 1] > 0.0)
        & (pairs[:, 4, 1] < pairs[:, 4, 0])
        & (pairs[:, 4, 0] <= 230.0)
        & (pairs[:, 4, 1] <= 230.0)
    )
    events = events[physical]
    pairs = pairs[physical]
    selected = filter_mask(pairs)
    converted, _ = direct_wepl(
        pairs[selected], model_path, air_slope, radius
    )
    return events[selected], converted


def prepare_run(
    run_id: int,
    raw_root_text: str,
    stage7_root_text: str,
    output_text: str,
    config: dict[str, Any],
    force: bool,
) -> list[dict[str, Any]]:
    from preprocessing.paircuts import read_mhd, write_mhd

    raw_root = Path(raw_root_text)
    stage7_root = Path(stage7_root_text)
    output = Path(output_text)
    stage7_seed = 20260713
    reference_z = tuple(float(v) for v in config["reference_planes_z_mm"])
    model_path = Path(config["_wepl_model"])
    rows: list[dict[str, Any]] = []
    for condition_name, condition in config["conditions"].items():
        id_path = (
            output / "event_ids" / condition_name / f"events{run_id:04d}.npy"
        )
        existing_path = (
            stage7_root
            / "full"
            / str(condition["stage7_name"])
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
        )
        tasks = [
            (int(config["main_seed"]), float(f))
            for f in config["fractions"]
            if float(f) < 1.0
        ]
        if condition_name == "combined_0p2mm_1pct":
            tasks.extend(
                (int(seed), float(f))
                for seed in config["replicate_seeds"]
                for f in config["replicate_fractions"]
            )
        expected = [
            group_root(output, condition_name, seed, fraction)
            / "pairs"
            / f"pairs{run_id:04d}.mhd"
            for seed, fraction in tasks
        ]
        if (
            not force
            and id_path.is_file()
            and all(path.is_file() for path in expected)
        ):
            events = np.load(id_path, mmap_mode="r")
            full_count = len(events)
            for seed, fraction in tasks:
                selected = nested_mask(events, run_id, seed, fraction)
                rows.append(
                    {
                        "run_id": run_id,
                        "condition": condition_name,
                        "seed": seed,
                        "fraction": fraction,
                        "full_filtered": full_count,
                        "selected": int(np.count_nonzero(selected)),
                        "status": "reused",
                    }
                )
            continue
        events, rebuilt = rebuild_filtered(
            run_id,
            raw_root,
            condition,
            stage7_seed,
            reference_z,
            model_path,
            float(config["air_wepl_slope_mm_per_mm"]),
            float(config["phantom_radius_mm"]),
        )
        existing = read_mhd(existing_path)
        if existing.shape != rebuilt.shape:
            raise RuntimeError(
                f"{condition_name}/{run_id}: Stage7 shape mismatch "
                f"{existing.shape} != {rebuilt.shape}"
            )
        max_abs = (
            float(np.max(np.abs(existing - rebuilt))) if len(existing) else 0.0
        )
        if max_abs > 2.0e-5 or not np.all(np.isfinite(rebuilt)):
            raise RuntimeError(
                f"{condition_name}/{run_id}: Stage7 mapping mismatch {max_abs:g}"
            )
        id_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(id_path, np.asarray(events, dtype=np.int64))
        for seed, fraction in tasks:
            selected = nested_mask(events, run_id, seed, fraction)
            if not np.any(selected):
                raise RuntimeError(
                    f"empty subset {condition_name}/{run_id}/{seed}/{fraction}"
                )
            destination = (
                group_root(output, condition_name, seed, fraction)
                / "pairs"
                / f"pairs{run_id:04d}.mhd"
            )
            write_mhd(destination, np.ascontiguousarray(rebuilt[selected]))
            rows.append(
                {
                    "run_id": run_id,
                    "condition": condition_name,
                    "seed": seed,
                    "fraction": fraction,
                    "full_filtered": len(rebuilt),
                    "selected": int(np.count_nonzero(selected)),
                    "mapping_max_abs": max_abs,
                    "status": "written",
                }
            )
    return rows
