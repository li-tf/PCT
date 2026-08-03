"""ROOT pairing, leakage-safe local filtering and calibrated WEPL preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import uproot

from io3d import read_pairs, read_partition, write_pairs, write_partition
from physics3d import (
    energies_to_wepl,
    partition_codes,
    screen_train_mask,
    subtract_external_air,
)


BRANCHES = [
    "RunID",
    "EventID",
    "TrackID",
    "KineticEnergy",
    "PreGlobalTime",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Direction_X",
    "Direction_Y",
    "Direction_Z",
]


def branch_array(tree, name: str) -> np.ndarray:
    """Avoid uproot's slow high-level executor for Windows-written TTrees."""

    branch = tree[name]
    chunks = [
        np.asarray(branch.basket(index).array(branch.interpretation))
        for index in range(branch.num_baskets)
    ]
    values = np.concatenate(chunks) if chunks else np.empty(0)
    if values.size != tree.num_entries:
        raise RuntimeError(f"incomplete {name}: {values.size} != {tree.num_entries}")
    return values


def locate_run(root: Path, run_id: int) -> Path | None:
    for name in (f"run_{run_id:03d}", f"run_{run_id:04d}", f"angle_{run_id:03d}", f"{run_id:03d}"):
        path = root / name
        if path.is_dir():
            return path
    return None


def load_primary(path: Path, tree_name: str) -> dict[str, np.ndarray]:
    with uproot.open(path) as root:
        tree = root[tree_name]
        missing = sorted(set(BRANCHES) - set(tree.keys()))
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        arrays = {name: branch_array(tree, name) for name in BRANCHES}
    primary = (
        (arrays["TrackID"] == 1)
        & (arrays["Direction_Z"] > 0)
        & np.isfinite(arrays["KineticEnergy"])
    )
    raw_index = np.flatnonzero(primary)
    events = np.asarray(arrays["EventID"][raw_index], dtype=np.int64)
    times = np.asarray(arrays["PreGlobalTime"][raw_index], dtype=np.float64)
    order = np.lexsort((times, events))
    sorted_events = events[order]
    first = np.r_[True, sorted_events[1:] != sorted_events[:-1]]
    selected = raw_index[order[first]]
    result = {key: np.asarray(value[selected]) for key, value in arrays.items()}
    return result


def _aligned(plane: dict[str, np.ndarray], indexes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.column_stack(
        (plane["Position_X"][indexes], plane["Position_Y"][indexes], plane["Position_Z"][indexes])
    ).astype(np.float32)
    direction = np.column_stack(
        (plane["Direction_X"][indexes], plane["Direction_Y"][indexes], plane["Direction_Z"][indexes])
    ).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("zero direction")
    direction /= norm[:, None]
    energy = plane["KineticEnergy"][indexes].astype(np.float32)
    return position, direction, energy


def _project(position: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray:
    if np.any(np.abs(direction[:, 2]) <= 1e-8):
        raise ValueError("direction parallel to reference plane")
    distance = (z - position[:, 2]) / direction[:, 2]
    result = position + distance[:, None] * direction
    result[:, 2] = z
    return result.astype(np.float32)


def pair_run(run_dir: Path, reference_z: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    incoming = load_primary(run_dir / "PhaseSpaceIn.root", "PhaseSpaceIn")
    outgoing = load_primary(run_dir / "PhaseSpaceOut.root", "PhaseSpaceOut")
    events, ii, io = np.intersect1d(
        incoming["EventID"].astype(np.int64),
        outgoing["EventID"].astype(np.int64),
        assume_unique=True,
        return_indices=True,
    )
    pin, din, ein = _aligned(incoming, ii)
    pout, dout, eout = _aligned(outgoing, io)
    pin = _project(pin, din, reference_z[0])
    pout = _project(pout, dout, reference_z[1])
    pairs = np.zeros((len(events), 5, 3), dtype=np.float32)
    pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3] = pin, pout, din, dout
    pairs[:, 4, 0], pairs[:, 4, 1], pairs[:, 4, 2] = ein, eout, 1.0
    if len(np.unique(events)) != len(events):
        raise RuntimeError(f"{run_dir}: paired primary EventID is not unique")
    if not np.isfinite(pairs).all():
        raise RuntimeError(f"{run_dir}: paired state contains NaN or Inf")
    return events.astype(np.int64), pairs


def _filter_features(pairs: np.ndarray, config: dict[str, Any]):
    filt = config["filter"]
    z0 = -pairs[:, 0, 2] / pairs[:, 2, 2]
    x0 = pairs[:, 0, 0] + z0 * pairs[:, 2, 0]
    y0 = pairs[:, 0, 1] + z0 * pairs[:, 2, 1]
    origin = np.asarray(filt["grid_origin_mm"])
    spacing = np.asarray(filt["grid_spacing_mm"])
    size = np.asarray(filt["grid_size"])
    ix = np.floor((x0 - origin[0]) / spacing[0] + 0.5).astype(np.int64)
    iy = np.floor((y0 - origin[1]) / spacing[1] + 0.5).astype(np.int64)
    inside = (ix >= 0) & (ix < size[0]) & (iy >= 0) & (iy < size[1])
    pixel = ix + size[0] * iy
    angle_x = np.arctan2(pairs[:, 3, 0], pairs[:, 3, 2]) - np.arctan2(
        pairs[:, 2, 0], pairs[:, 2, 2]
    )
    angle_y = np.arctan2(pairs[:, 3, 1], pairs[:, 3, 2]) - np.arctan2(
        pairs[:, 2, 1], pairs[:, 2, 2]
    )
    loss = pairs[:, 4, 0] - pairs[:, 4, 1]
    return inside, pixel, np.column_stack((loss, angle_x, angle_y))


def train_local_filter(
    pairs: np.ndarray, partition: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    inside, pixel, features = _filter_features(pairs, config)
    train = (partition == 0) & inside
    bins = int(np.prod(config["filter"]["grid_size"]))
    minimum = int(config["filter"]["minimum_train_per_bin"])
    count = np.bincount(pixel[train], minlength=bins)
    means = np.zeros((bins, 3), dtype=np.float64)
    sigma = np.zeros((bins, 3), dtype=np.float64)
    global_mean = np.mean(features[train], axis=0)
    global_sigma = np.std(features[train], axis=0)
    for column in range(3):
        sums = np.bincount(pixel[train], weights=features[train, column], minlength=bins)
        sums2 = np.bincount(pixel[train], weights=features[train, column] ** 2, minlength=bins)
        occupied = count > 0
        means[occupied, column] = sums[occupied] / count[occupied]
        variance = np.maximum(
            sums2[occupied] / count[occupied] - means[occupied, column] ** 2, 0.0
        )
        sigma[occupied, column] = np.sqrt(variance)
        fallback = count < minimum
        means[fallback, column] = global_mean[column]
        sigma[fallback, column] = global_sigma[column]
    factors = np.asarray(
        [
            config["filter"]["energy_sigma"],
            config["filter"]["angle_sigma"],
            config["filter"]["angle_sigma"],
        ]
    )
    selected = np.zeros(len(pairs), dtype=bool)
    candidates = np.flatnonzero(inside)
    scale = np.maximum(sigma[pixel[candidates]], 1e-12)
    selected[candidates] = np.all(
        np.abs(features[candidates] - means[pixel[candidates]]) <= factors * scale,
        axis=1,
    )
    return selected, {
        "inside_grid": int(np.count_nonzero(inside)),
        "outside_grid": int(np.count_nonzero(~inside)),
        "selected": int(np.count_nonzero(selected)),
        "train_count": int(np.count_nonzero(partition == 0)),
        "fallback_bins": int(np.count_nonzero(count < minimum)),
        "occupied_bins": int(np.count_nonzero(count)),
    }


def load_completed_run(
    pair_path: Path,
    event_path: Path,
    split_path: Path,
    screen_path: Path,
    qc_path: Path,
) -> dict[str, Any] | None:
    """Return a valid per-angle checkpoint, or None after any interrupted write."""

    try:
        previous = json.loads(qc_path.read_text(encoding="utf-8"))
        count = int(previous["filtered_support_hit"])
        existing_pairs = read_pairs(pair_path, mmap=True)
        existing_events = np.load(event_path, mmap_mode="r", allow_pickle=False)
        existing_split = read_partition(split_path)
        existing_screen = read_partition(screen_path)
        complete = (
            previous.get("status") == "PASS"
            and len(existing_pairs) == count
            and len(existing_events) == count
            and len(existing_split) == count
            and len(existing_screen) == count
            and int(np.count_nonzero(existing_split == 0)) == int(previous["train"])
            and int(np.count_nonzero(existing_split == 1)) == int(previous["validation"])
            and int(np.count_nonzero(existing_split == 2)) == int(previous["test"])
            and int(np.count_nonzero(existing_screen == 1)) == int(previous["screen_train"])
        )
        return previous if complete else None
    except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError):
        return None


def process_run(
    run_id: int,
    raw_root_text: str,
    output_root_text: str,
    config: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    raw_root, output = Path(raw_root_text), Path(output_root_text)
    pair_path = output / "pairs" / f"pairs{run_id:04d}.mhd"
    event_path = output / "events" / f"events{run_id:04d}.npy"
    split_path = output / "splits" / f"split{run_id:04d}.npz"
    screen_path = output / "splits" / f"screen{run_id:04d}.npz"
    qc_path = output / "qc_runs" / f"run_{run_id:04d}.json"
    if not force:
        previous = load_completed_run(
            pair_path, event_path, split_path, screen_path, qc_path
        )
        if previous is not None:
            return previous
    run_dir = locate_run(raw_root, run_id)
    if run_dir is None:
        raise FileNotFoundError(f"run {run_id:03d}")
    events, pairs = pair_run(run_dir, tuple(config["reference_z_mm"]))
    paired = len(pairs)
    physical = (
        np.isfinite(pairs).all(axis=(1, 2))
        & (pairs[:, 4, 0] > pairs[:, 4, 1])
        & (pairs[:, 4, 1] > 0)
        & (pairs[:, 4, 0] <= 230)
    )
    events, pairs = events[physical], pairs[physical]
    partition = partition_codes(run_id, events, int(config["split"]["seed"]))
    selected, diagnostics = train_local_filter(pairs, partition, config)
    events, pairs, partition = events[selected], pairs[selected], partition[selected]
    wepl = energies_to_wepl(Path(config["_wepl_model"]), pairs[:, 4, 0], pairs[:, 4, 1])
    corrected, support_hit = subtract_external_air(
        pairs,
        wepl,
        float(config["phantom_radius_mm"]),
        float(config["phantom_half_length_y_mm"]),
        float(config["air_wepl_slope_mm_per_mm"]),
    )
    direct_length = np.linalg.norm(pairs[:, 1].astype(np.float64) - pairs[:, 0].astype(np.float64), axis=1)
    air_only_residual = wepl[~support_hit] - float(config["air_wepl_slope_mm_per_mm"]) * direct_length[~support_hit]
    events, pairs, partition, corrected = (
        events[support_hit],
        pairs[support_hit],
        partition[support_hit],
        corrected[support_hit],
    )
    pairs[:, 4, 0] = 0.0
    pairs[:, 4, 1] = corrected
    pairs[:, 4, 2] = 0.0
    pairs = np.ascontiguousarray(pairs, dtype="<f4")
    events = np.ascontiguousarray(events, dtype="<i8")
    partition = np.ascontiguousarray(partition, dtype=np.uint8)
    screen = (partition == 0) & screen_train_mask(
        run_id, events, int(config["split"]["seed"])
    )
    write_pairs(pair_path, pairs)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    event_temporary = event_path.with_suffix(event_path.suffix + ".tmp")
    with event_temporary.open("wb") as stream:
        np.save(stream, events)
    event_temporary.replace(event_path)
    write_partition(split_path, partition)
    write_partition(screen_path, screen.astype(np.uint8))
    result = {
        "run_id": run_id,
        "paired": paired,
        "physical": int(np.count_nonzero(physical)),
        "filtered_support_hit": len(pairs),
        "train": int(np.count_nonzero(partition == 0)),
        "validation": int(np.count_nonzero(partition == 1)),
        "test": int(np.count_nonzero(partition == 2)),
        "screen_train": int(np.count_nonzero(screen)),
        "pairs_sha256": hashlib.sha256(pairs).hexdigest(),
        "events_sha256": hashlib.sha256(events).hexdigest(),
        "partition_sha256": hashlib.sha256(partition).hexdigest(),
        "wepl_mean_mm": float(np.mean(corrected)),
        "wepl_min_mm": float(np.min(corrected)),
        "wepl_max_mm": float(np.max(corrected)),
        "air_only_count": int(len(air_only_residual)),
        "air_only_residual_sum_mm": float(np.sum(air_only_residual, dtype=np.float64)),
        "air_only_residual_abs_sum_mm": float(np.sum(np.abs(air_only_residual), dtype=np.float64)),
        **diagnostics,
        "status": "PASS",
    }
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = qc_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(qc_path)
    return result
