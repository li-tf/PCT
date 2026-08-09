"""Data preparation helpers for Stage 8B.

The transition study is derived from the already verified Stage-7C 25% parent
set.  Noise-source studies are rebuilt from the six-plane ROOT data so that
the local filter is applied after the requested detector perturbation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def fraction_tag(fraction: float) -> str:
    value = int(round(1000.0 * float(fraction)))
    return f"f{value:04d}"


def group_root(root: Path, seed: int, fraction: float) -> Path:
    return root / f"seed_{seed}" / fraction_tag(fraction)


def nested_mask(events: np.ndarray, run_id: int, seed: int, fraction: float) -> np.ndarray:
    from stage3_io import splitmix64

    bucket = splitmix64(np.asarray(events, np.uint64), run_id, seed)
    bucket %= np.uint64(1_000_000)
    return bucket < np.uint64(round(float(fraction) * 1_000_000))


def subset_from_stage7c_parent(
    run_id: int,
    stage7c_root_text: str,
    output_root_text: str,
    seed: int,
    fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Write one nested subset from the matching Stage-7C 25% parent."""
    from preprocessing.paircuts import read_mhd, write_mhd

    stage7c = Path(stage7c_root_text)
    output = Path(output_root_text)
    destination = group_root(output, seed, fraction) / "pairs" / f"pairs{run_id:04d}.mhd"
    events_path = stage7c / "event_ids/combined_0p2mm_1pct" / f"events{run_id:04d}.npy"
    parent = (
        stage7c / "combined_0p2mm_1pct" / f"seed_{seed}" / "f025" / "pairs"
        / f"pairs{run_id:04d}.mhd"
    )
    if destination.is_file() and not force:
        from preprocessing.paircuts import read_mhd

        rows = len(read_mhd(destination))
        return {"run_id": run_id, "seed": seed, "fraction": fraction, "selected": rows, "status": "reused"}
    if not events_path.is_file() or not parent.is_file():
        raise FileNotFoundError(f"missing Stage-7C parent for run {run_id:04d}, seed {seed}")
    events = np.load(events_path, mmap_mode="r")
    parent_mask = nested_mask(events, run_id, seed, 0.25)
    target_mask = nested_mask(events, run_id, seed, fraction)
    if np.any(target_mask & ~parent_mask):
        raise RuntimeError("target subset is not nested inside the 25% parent")
    parent_pairs = read_mhd(parent)
    if len(parent_pairs) != int(np.count_nonzero(parent_mask)):
        raise RuntimeError(f"Stage-7C EventID alignment failed for run {run_id:04d}")
    selected = np.asarray(parent_pairs[target_mask[parent_mask]], np.float32)
    if not len(selected) or not np.isfinite(selected).all():
        raise RuntimeError(f"invalid Stage-8B subset for run {run_id:04d}")
    write_mhd(destination, np.ascontiguousarray(selected))
    return {
        "run_id": run_id,
        "seed": seed,
        "fraction": fraction,
        "parent": len(parent_pairs),
        "selected": len(selected),
        "status": "written",
    }


def subset_angular_from_stage7c_parent(
    output_run: int,
    original_run: int,
    stage7c_root_text: str,
    output_root_text: str,
    seed: int,
    fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Write one reindexed angular subset from a Stage-7C 25% parent.

    The hash continues to use the original RunID so selecting only the even
    0.5-degree views does not alter the deterministic proton subset.
    """
    from preprocessing.paircuts import read_mhd, write_mhd

    stage7c = Path(stage7c_root_text)
    output = Path(output_root_text)
    destination = (
        group_root(output, seed, fraction) / "pairs"
        / f"pairs{output_run:04d}.mhd"
    )
    events_path = (
        stage7c / "event_ids/combined_0p2mm_1pct"
        / f"events{original_run:04d}.npy"
    )
    parent = (
        stage7c / "combined_0p2mm_1pct" / f"seed_{seed}" / "f025"
        / "pairs" / f"pairs{original_run:04d}.mhd"
    )
    if destination.is_file() and not force:
        return {
            "output_run": output_run,
            "original_run": original_run,
            "seed": seed,
            "fraction": fraction,
            "selected": len(read_mhd(destination)),
            "status": "reused",
        }
    if not events_path.is_file() or not parent.is_file():
        raise FileNotFoundError(
            f"missing Stage-7C parent for original run {original_run:04d}, "
            f"seed {seed}"
        )
    events = np.load(events_path, mmap_mode="r")
    parent_mask = nested_mask(events, original_run, seed, 0.25)
    target_mask = nested_mask(events, original_run, seed, fraction)
    if np.any(target_mask & ~parent_mask):
        raise RuntimeError("angular target subset is not nested in 25% parent")
    parent_pairs = read_mhd(parent)
    if len(parent_pairs) != int(np.count_nonzero(parent_mask)):
        raise RuntimeError(
            f"Stage-7C EventID alignment failed for run {original_run:04d}"
        )
    selected = np.asarray(parent_pairs[target_mask[parent_mask]], np.float32)
    if not len(selected) or not np.isfinite(selected).all():
        raise RuntimeError(
            f"invalid angular subset for original run {original_run:04d}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(destination, np.ascontiguousarray(selected))
    return {
        "output_run": output_run,
        "original_run": original_run,
        "seed": seed,
        "fraction": fraction,
        "parent": len(parent_pairs),
        "selected": len(selected),
        "status": "written",
    }


def _physical_energy(pairs: np.ndarray) -> np.ndarray:
    ein, eout = pairs[:, 4, 0], pairs[:, 4, 1]
    return (
        np.isfinite(ein) & np.isfinite(eout) & (ein > 0) & (eout > 0)
        & (eout < ein) & (ein <= 230) & (eout <= 230)
    )


def prepare_noise_condition_run(
    run_id: int,
    raw_root_text: str,
    output_root_text: str,
    condition_name: str,
    condition: dict[str, Any],
    subset_seed: int,
    fraction: float,
    noise_seed: int,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    """Build, locally filter and store one detector-noise condition."""
    from detector_processing import PLANE_NAMES, common_events, load_run
    from preprocessing import paircuts
    from run_stage7 import direct_wepl, locate_run
    from stage7b_data import make_condition

    output = Path(output_root_text)
    root = output / "noise_sources" / condition_name / f"seed_{subset_seed}" / fraction_tag(fraction)
    pair_path = root / "pairs" / f"pairs{run_id:04d}.mhd"
    meta_path = root / "metadata" / f"meta{run_id:04d}.npz"
    if pair_path.is_file() and meta_path.is_file() and not force:
        with np.load(meta_path, allow_pickle=False) as meta:
            count = len(meta["event_id"])
        return {"run_id": run_id, "condition": condition_name, "selected": count, "status": "reused"}
    run_dir = locate_run(Path(raw_root_text), run_id)
    if run_dir is None:
        raise FileNotFoundError(f"missing D1 ROOT run {run_id:04d}")
    planes = load_run(run_dir)
    events = common_events([planes[name] for name in PLANE_NAMES])
    measured, ideal = make_condition(
        planes,
        events,
        condition,
        noise_seed,
        run_id,
        tuple(float(value) for value in config["reference_planes_z_mm"]),
    )
    physical = _physical_energy(measured)
    selected = paircuts.filter_mask(measured) if hasattr(paircuts, "filter_mask") else None
    if selected is None:
        # Stage-7C contains the exact mature local-3sigma implementation.
        from stage7c_data import filter_mask

        selected = filter_mask(measured)
    accepted = physical & selected
    accepted &= nested_mask(events, run_id, subset_seed, fraction)
    safe = np.array(measured, copy=True)
    safe[~physical, 4, 1] = ideal[~physical, 4, 1]
    converted, measured_wepl = direct_wepl(
        safe,
        Path(config["_wepl_model"]),
        float(config["air_wepl_slope_mm_per_mm"]),
        float(config["phantom_radius_mm"]),
    )
    _, ideal_wepl = direct_wepl(
        ideal,
        Path(config["_wepl_model"]),
        float(config["air_wepl_slope_mm_per_mm"]),
        float(config["phantom_radius_mm"]),
    )
    if not np.any(accepted):
        raise RuntimeError(f"empty noise condition {condition_name}/{run_id:04d}")
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    paircuts.write_mhd(pair_path, np.ascontiguousarray(converted[accepted], np.float32))
    np.savez(
        meta_path,
        event_id=np.asarray(events[accepted], np.int64),
        measured_eout_mev=np.asarray(measured[accepted, 4, 1], np.float32),
        measured_wepl_mm=np.asarray(measured_wepl[accepted], np.float32),
        ideal_wepl_mm=np.asarray(ideal_wepl[accepted], np.float32),
    )
    return {
        "run_id": run_id,
        "condition": condition_name,
        "common_events": len(events),
        "physical": int(np.count_nonzero(physical)),
        "accepted": int(np.count_nonzero(physical & selected)),
        "selected": int(np.count_nonzero(accepted)),
        "status": "written",
    }


def prepare_angular_noise_condition_run(
    output_run: int,
    original_run: int,
    raw_root_text: str,
    output_root_text: str,
    condition_name: str,
    condition: dict[str, Any],
    subset_seed: int,
    fraction: float,
    noise_seed: int,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    """Build one ROOT-derived condition and reindex an even view to 0..359."""
    from preprocessing.paircuts import read_mhd, write_mhd

    output = Path(output_root_text)
    destination_root = (
        output / "angular_noise_sources" / condition_name
        / f"seed_{subset_seed}" / fraction_tag(fraction)
    )
    pair_destination = (
        destination_root / "pairs" / f"pairs{output_run:04d}.mhd"
    )
    meta_destination = (
        destination_root / "metadata" / f"meta{output_run:04d}.npz"
    )
    if pair_destination.is_file() and meta_destination.is_file() and not force:
        with np.load(meta_destination, allow_pickle=False) as meta:
            selected = len(meta["event_id"])
        return {
            "output_run": output_run,
            "original_run": original_run,
            "condition": condition_name,
            "selected": selected,
            "status": "reused",
        }

    scratch = output / "_angular_raw_cache"
    result = prepare_noise_condition_run(
        original_run,
        raw_root_text,
        str(scratch),
        condition_name,
        condition,
        subset_seed,
        fraction,
        noise_seed,
        config,
        force,
    )
    source_root = (
        scratch / "noise_sources" / condition_name / f"seed_{subset_seed}"
        / fraction_tag(fraction)
    )
    pair_source = source_root / "pairs" / f"pairs{original_run:04d}.mhd"
    meta_source = source_root / "metadata" / f"meta{original_run:04d}.npz"
    pairs = np.ascontiguousarray(read_mhd(pair_source), dtype=np.float32)
    if not len(pairs) or not np.isfinite(pairs).all():
        raise RuntimeError(
            f"invalid ROOT-derived angular subset {original_run:04d}"
        )
    pair_destination.parent.mkdir(parents=True, exist_ok=True)
    meta_destination.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(pair_destination, pairs)
    with np.load(meta_source, allow_pickle=False) as meta:
        np.savez(
            meta_destination,
            **{key: np.asarray(meta[key]) for key in meta.files},
        )

    for path in (
        pair_source,
        pair_source.with_suffix(".raw"),
        meta_source,
    ):
        path.unlink(missing_ok=True)
    return {
        "output_run": output_run,
        "original_run": original_run,
        "condition": condition_name,
        "selected": int(result["selected"]),
        "status": "written",
    }
