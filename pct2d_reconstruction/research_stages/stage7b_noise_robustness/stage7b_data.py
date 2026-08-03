"""Event-stable D1 preparation for Stage 7B.

The split is made from (RunID, EventID) before any physical or statistical
selection.  A locally fitted 3-sigma model is trained only on the training
partition and then applied unchanged to validation and test rows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def _seed(base: int, run_id: int, stream: str) -> int:
    token = int.from_bytes(hashlib.sha256(stream.encode()).digest()[:8], "little")
    return int((base + 1000003 * run_id + token) % (2**32))


def event_partitions(
    events: np.ndarray, run_id: int, split: dict[str, Any]
) -> dict[str, np.ndarray]:
    from stage3_io import splitmix64

    bucket = splitmix64(
        np.asarray(events, dtype=np.uint64), run_id, int(split["seed"])
    ) % np.uint64(split["modulus"])
    test = bucket == np.uint64(split["test_remainder"])
    validation = bucket == np.uint64(split["validation_remainder"])
    train = ~(test | validation)
    screen_bucket = splitmix64(
        np.asarray(events, dtype=np.uint64),
        run_id,
        int(split["seed"]) ^ 0x51A7B,
    ) % np.uint64(split["screen_modulus"])
    screen = train & (screen_bucket == np.uint64(split["screen_remainder"]))
    return {
        "train": train,
        "validation": validation,
        "test": test,
        "screen": screen,
    }


def _track_pairs(
    planes: dict[str, dict[str, np.ndarray]],
    events: np.ndarray,
    position_sigma_mm: float,
    seed: int,
    reference_z: tuple[float, float],
) -> np.ndarray:
    from detector_processing import aligned, line_from_hits

    names = (
        "TrackerUpstream1",
        "TrackerUpstream2",
        "TrackerDownstream1",
        "TrackerDownstream2",
    )
    hits = {
        name: np.array(aligned(planes[name], events, "position"), copy=True)
        for name in names
    }
    if position_sigma_mm > 0:
        rng = np.random.default_rng(seed)
        for hit in hits.values():
            hit[:, :2] += rng.normal(
                0.0, position_sigma_mm, (len(hit), 2)
            ).astype(np.float32)
    pin, din = line_from_hits(
        hits["TrackerUpstream1"], hits["TrackerUpstream2"], reference_z[0]
    )
    pout, dout = line_from_hits(
        hits["TrackerDownstream1"], hits["TrackerDownstream2"], reference_z[1]
    )
    pairs = np.zeros((len(events), 5, 3), dtype=np.float32)
    pairs[:, 0], pairs[:, 1] = pin, pout
    pairs[:, 2], pairs[:, 3] = din, dout
    return pairs


def make_condition(
    planes: dict[str, dict[str, np.ndarray]],
    events: np.ndarray,
    condition: dict[str, float],
    noise_seed: int,
    run_id: int,
    reference_z: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return measured and same-track ideal-energy pairs."""
    from detector_processing import aligned

    measured = _track_pairs(
        planes,
        events,
        float(condition["position_sigma_mm"]),
        _seed(noise_seed, run_id, "position"),
        reference_z,
    )
    energy_in = np.asarray(
        aligned(planes["PhaseSpaceIn"], events, "energy"), dtype=np.float32
    )
    energy_out = np.asarray(
        aligned(planes["PhaseSpaceOut"], events, "energy"), dtype=np.float32
    )
    measured[:, 4, 0] = energy_in
    measured[:, 4, 1] = energy_out
    fraction = float(condition["energy_sigma_fraction"])
    if fraction > 0:
        rng = np.random.default_rng(_seed(noise_seed, run_id, "energy"))
        measured[:, 4, 1] = (
            energy_out.astype(np.float64)
            * (1.0 + rng.normal(0.0, fraction, len(energy_out)))
        ).astype(np.float32)
    ideal = np.array(measured, copy=True)
    ideal[:, 4, 1] = energy_out
    return measured, ideal


def _physical_energy(pairs: np.ndarray) -> np.ndarray:
    ein, eout = pairs[:, 4, 0], pairs[:, 4, 1]
    return (
        np.isfinite(ein)
        & np.isfinite(eout)
        & (ein > 0)
        & (eout > 0)
        & (eout < ein)
        & (ein <= 230)
        & (eout <= 230)
    )


def _write_group(
    root: Path,
    group: str,
    run_id: int,
    pairs: np.ndarray,
    events: np.ndarray,
    measured_eout: np.ndarray,
    measured_wepl: np.ndarray,
    ideal_wepl: np.ndarray,
    reference_wepl: np.ndarray,
) -> None:
    from preprocessing.paircuts import write_mhd

    directory = root / group
    pair_path = directory / "pairs" / f"pairs{run_id:04d}.mhd"
    meta_path = directory / "metadata" / f"meta{run_id:04d}.npz"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_mhd(pair_path, np.ascontiguousarray(pairs, dtype=np.float32))
    np.savez(
        meta_path,
        event_id=np.asarray(events, dtype=np.int64),
        measured_eout_mev=np.asarray(measured_eout, dtype=np.float32),
        measured_wepl_mm=np.asarray(measured_wepl, dtype=np.float32),
        ideal_wepl_mm=np.asarray(ideal_wepl, dtype=np.float32),
        reference_wepl_mm=np.asarray(reference_wepl, dtype=np.float32),
    )


def prepare_run(
    run_id: int,
    raw_root: str,
    output_root: str,
    config: dict[str, Any],
    force: bool,
) -> list[dict[str, Any]]:
    """Prepare one angle and return one QC row per noise condition/seed."""
    from detector_processing import PLANE_NAMES, common_events, load_run
    from robust_models import apply_filter, fit_filter
    from stage3_io import pair_features
    from run_stage7 import direct_wepl, locate_run

    raw = Path(raw_root)
    output = Path(output_root)
    run_qc = output / "qc_runs" / f"run_{run_id:04d}.json"
    primary_seed = int(config["noise_seeds"][0])
    expected = [
        output
        / "confirm/combined_0p2mm_1pct"
        / f"seed_{primary_seed}"
        / partition
        / "metadata"
        / f"meta{run_id:04d}.npz"
        for partition in ("train", "validation", "test")
    ]
    expected += [
        output
        / "screen/combined_0p2mm_1pct"
        / f"seed_{seed}"
        / partition
        / "metadata"
        / f"meta{run_id:04d}.npz"
        for seed in config["noise_seeds"]
        for partition in ("train", "validation")
    ]
    expected += [
        output
        / "screen"
        / name
        / f"seed_{primary_seed}"
        / partition
        / "metadata"
        / f"meta{run_id:04d}.npz"
        for name in ("continuous", "position_0p2mm", "energy_1pct")
        for partition in ("train", "validation")
    ]
    if not force and run_qc.is_file() and all(path.is_file() for path in expected):
        import json

        return json.loads(run_qc.read_text(encoding="utf-8"))

    run_dir = locate_run(raw, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"missing D1 run {run_id:03d}")
    planes = load_run(run_dir)
    events = common_events([planes[name] for name in PLANE_NAMES])
    partitions = event_partitions(events, run_id, config["split"])
    reference_z = tuple(float(value) for value in config["reference_planes_z_mm"])
    model_path = Path(config["_resolved_wepl_model"])
    filtering = config["filtering"]
    rows: list[dict[str, Any]] = []

    # Reference WEPL uses the continuous four-hit track and ideal energy.
    reference_raw, reference_ideal_raw = make_condition(
        planes,
        events,
        config["noise_conditions"]["continuous"],
        primary_seed,
        run_id,
        reference_z,
    )
    reference_physical = _physical_energy(reference_ideal_raw)
    reference_for_conversion = np.array(reference_ideal_raw, copy=True)
    reference_for_conversion[~reference_physical, 4, 1] = (
        reference_for_conversion[~reference_physical, 4, 0]
    )
    reference_converted, reference_wepl = direct_wepl(
        reference_for_conversion,
        model_path,
        float(config["air_wepl_slope_mm_per_mm"]),
        float(config["phantom_radius_mm"]),
    )
    del reference_raw, reference_converted

    for condition_name, condition in config["noise_conditions"].items():
        seeds = (
            config["noise_seeds"]
            if condition_name == "combined_0p2mm_1pct"
            else [primary_seed]
        )
        for noise_seed in seeds:
            measured, ideal = make_condition(
                planes,
                events,
                condition,
                int(noise_seed),
                run_id,
                reference_z,
            )
            physical = _physical_energy(measured)
            inside, cells, features = pair_features(measured, filtering)
            filter_model = fit_filter(
                "baseline_3sigma",
                features,
                cells,
                inside & physical,
                partitions["train"],
                filtering,
            )
            accepted, distance = apply_filter(
                filter_model, features, cells, inside & physical, filtering
            )
            # The calibrated LUT intentionally rejects non-physical energies.
            # Invalid rows have already been excluded from ``accepted``; copy
            # the ideal exit energy into them only to keep vector conversion
            # finite without changing any retained measurement.
            measured_for_conversion = np.array(measured, copy=True)
            measured_for_conversion[~physical, 4, 1] = ideal[~physical, 4, 1]
            converted, measured_wepl = direct_wepl(
                measured_for_conversion,
                model_path,
                float(config["air_wepl_slope_mm_per_mm"]),
                float(config["phantom_radius_mm"]),
            )
            _, ideal_wepl = direct_wepl(
                ideal,
                model_path,
                float(config["air_wepl_slope_mm_per_mm"]),
                float(config["phantom_radius_mm"]),
            )
            measured_eout = measured[:, 4, 1]

            screen_mask = accepted & partitions["screen"]
            screen_group = (
                f"screen/{condition_name}/seed_{noise_seed}/train"
            )
            _write_group(
                output,
                screen_group,
                run_id,
                converted[screen_mask],
                events[screen_mask],
                measured_eout[screen_mask],
                measured_wepl[screen_mask],
                ideal_wepl[screen_mask],
                reference_wepl[screen_mask],
            )
            if int(noise_seed) == primary_seed or (
                condition_name == "combined_0p2mm_1pct"
            ):
                validation_mask = accepted & partitions["validation"]
                _write_group(
                    output,
                    f"screen/{condition_name}/seed_{noise_seed}/validation",
                    run_id,
                    converted[validation_mask],
                    events[validation_mask],
                    measured_eout[validation_mask],
                    measured_wepl[validation_mask],
                    ideal_wepl[validation_mask],
                    reference_wepl[validation_mask],
                )

            if (
                condition_name == "combined_0p2mm_1pct"
                and int(noise_seed) == primary_seed
            ):
                for partition in ("train", "validation", "test"):
                    selected = accepted & partitions[partition]
                    _write_group(
                        output,
                        f"confirm/{condition_name}/seed_{noise_seed}/{partition}",
                        run_id,
                        converted[selected],
                        events[selected],
                        measured_eout[selected],
                        measured_wepl[selected],
                        ideal_wepl[selected],
                        reference_wepl[selected],
                    )

            rows.append(
                {
                    "run_id": run_id,
                    "condition": condition_name,
                    "noise_seed": int(noise_seed),
                    "common_events": int(len(events)),
                    "physical": int(np.count_nonzero(physical)),
                    "accepted": int(np.count_nonzero(accepted)),
                    "train": int(np.count_nonzero(accepted & partitions["train"])),
                    "validation": int(
                        np.count_nonzero(accepted & partitions["validation"])
                    ),
                    "test": int(np.count_nonzero(accepted & partitions["test"])),
                    "screen": int(np.count_nonzero(screen_mask)),
                    "noise_wepl_rmse_mm": float(
                        np.sqrt(
                            np.mean(
                                (
                                    measured_wepl[
                                        accepted & partitions["train"]
                                    ].astype(np.float64)
                                    - ideal_wepl[
                                        accepted & partitions["train"]
                                    ].astype(np.float64)
                                )
                                ** 2
                            )
                        )
                    ),
                    "reference_wepl_rmse_mm": float(
                        np.sqrt(
                            np.mean(
                                (
                                    measured_wepl[
                                        accepted & partitions["train"]
                                    ].astype(np.float64)
                                    - reference_wepl[
                                        accepted & partitions["train"]
                                    ].astype(np.float64)
                                )
                                ** 2
                            )
                        )
                    ),
                    "filter_distance_p99": float(
                        np.quantile(distance[np.isfinite(distance)], 0.99)
                    ),
                }
            )
    run_qc.parent.mkdir(parents=True, exist_ok=True)
    import json

    temporary = run_qc.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(run_qc)
    return rows
