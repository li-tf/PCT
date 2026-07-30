"""Six-plane D1 event pairing and deterministic detector digitization."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import uproot


PLANE_NAMES = (
    "PhaseSpaceIn",
    "PhaseSpaceOut",
    "TrackerUpstream1",
    "TrackerUpstream2",
    "TrackerDownstream1",
    "TrackerDownstream2",
)
BRANCHES = (
    "EventID",
    "TrackID",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Direction_X",
    "Direction_Y",
    "Direction_Z",
    "KineticEnergy",
    "PreGlobalTime",
)


def branch_array(tree, name: str) -> np.ndarray:
    """Read Windows-written TTrees without uproot's high-level executor."""
    branch = tree[name]
    chunks = [
        np.asarray(branch.basket(index).array(branch.interpretation))
        for index in range(branch.num_baskets)
    ]
    values = np.concatenate(chunks) if chunks else np.empty(0)
    if values.size != tree.num_entries:
        raise RuntimeError(
            f"incomplete {name}: {values.size} != {tree.num_entries}"
        )
    return values


def read_primary_plane(path: Path, tree_name: str) -> dict[str, np.ndarray]:
    with uproot.open(path) as root:
        tree = root[tree_name]
        values = {name: branch_array(tree, name) for name in BRANCHES}
    mask = (
        (values["TrackID"] == 1)
        & (values["Direction_Z"] > 0.0)
        & np.isfinite(values["KineticEnergy"])
    )
    event = np.asarray(values["EventID"][mask], dtype=np.int64)
    time = np.asarray(values["PreGlobalTime"][mask], dtype=np.float64)
    order = np.lexsort((time, event))
    event = event[order]
    first = np.r_[True, event[1:] != event[:-1]]
    selected = np.flatnonzero(mask)[order[first]]
    position = np.column_stack(
        [values[f"Position_{axis}"][selected] for axis in "XYZ"]
    ).astype(np.float32)
    direction = np.column_stack(
        [values[f"Direction_{axis}"][selected] for axis in "XYZ"]
    ).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return {
        "event": np.asarray(values["EventID"][selected], dtype=np.int64),
        "position": position,
        "direction": direction,
        "energy": np.asarray(
            values["KineticEnergy"][selected], dtype=np.float32
        ),
        "raw_entries": np.int64(len(values["EventID"])),
        "forward_unique": np.int64(len(selected)),
    }


def aligned(plane: dict[str, np.ndarray], events: np.ndarray, key: str) -> np.ndarray:
    index = np.searchsorted(plane["event"], events)
    if (
        np.any(index >= len(plane["event"]))
        or np.any(plane["event"][index] != events)
    ):
        raise RuntimeError("event alignment failed")
    return plane[key][index]


def common_events(planes: list[dict[str, np.ndarray]]) -> np.ndarray:
    result = planes[0]["event"]
    for plane in planes[1:]:
        result = np.intersect1d(
            result, plane["event"], assume_unique=True
        )
    return result


def extrapolate(
    position: np.ndarray, direction: np.ndarray, z_mm: float
) -> np.ndarray:
    dz = direction[:, 2]
    if np.any(np.abs(dz) < 1.0e-8):
        raise RuntimeError("tracker fit is parallel to reference plane")
    scale = (float(z_mm) - position[:, 2]) / dz
    result = position + scale[:, None] * direction
    result[:, 2] = float(z_mm)
    return result.astype(np.float32)


def line_from_hits(
    first: np.ndarray, second: np.ndarray, reference_z_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(second - first, dtype=np.float64)
    norm = np.linalg.norm(direction, axis=1)
    valid = np.isfinite(norm) & (norm > 1.0e-8) & (direction[:, 2] > 0.0)
    if not np.all(valid):
        raise RuntimeError(
            f"invalid four-hit directions: {np.count_nonzero(~valid)}"
        )
    direction /= norm[:, None]
    position = extrapolate(second, direction, reference_z_mm)
    return position, direction.astype(np.float32)


def variant_seed(base: int, run_id: int, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    token = int.from_bytes(digest[:4], "little")
    return int((base + 1000003 * run_id + token) % (2**32))


def ideal_pairs(
    phase_in: dict[str, np.ndarray],
    phase_out: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    events = common_events([phase_in, phase_out])
    pairs = np.zeros((len(events), 5, 3), dtype=np.float32)
    pairs[:, 0] = aligned(phase_in, events, "position")
    pairs[:, 1] = aligned(phase_out, events, "position")
    pairs[:, 2] = aligned(phase_in, events, "direction")
    pairs[:, 3] = aligned(phase_out, events, "direction")
    pairs[:, 4, 0] = aligned(phase_in, events, "energy")
    pairs[:, 4, 1] = aligned(phase_out, events, "energy")
    pairs[:, 4, 2] = 1.0
    return events, pairs


def hit_pairs(
    planes: dict[str, dict[str, np.ndarray]],
    variant: dict[str, Any],
    seed: int,
    reference_z: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    events = common_events([planes[name] for name in PLANE_NAMES])
    hits = {
        name: np.array(aligned(planes[name], events, "position"), copy=True)
        for name in PLANE_NAMES[2:]
    }
    rng = np.random.default_rng(seed)
    sigma = float(variant["position_sigma_mm"])
    if sigma > 0.0:
        for hit in hits.values():
            hit[:, :2] += rng.normal(0.0, sigma, (len(hit), 2)).astype(
                np.float32
            )
    pin, din = line_from_hits(
        hits["TrackerUpstream1"],
        hits["TrackerUpstream2"],
        reference_z[0],
    )
    pout, dout = line_from_hits(
        hits["TrackerDownstream1"],
        hits["TrackerDownstream2"],
        reference_z[1],
    )
    energy_in = np.asarray(
        aligned(planes["PhaseSpaceIn"], events, "energy"), dtype=np.float32
    )
    energy_out = np.asarray(
        aligned(planes["PhaseSpaceOut"], events, "energy"), dtype=np.float32
    )
    energy_sigma = float(variant["energy_sigma_fraction"])
    invalid_energy = 0
    if energy_sigma > 0.0:
        noisy = energy_out.astype(np.float64) * (
            1.0 + rng.normal(0.0, energy_sigma, len(energy_out))
        )
        invalid_energy = int(
            np.count_nonzero((noisy < 0.001) | (noisy >= energy_in))
        )
        energy_out = noisy.astype(np.float32)
    pairs = np.zeros((len(events), 5, 3), dtype=np.float32)
    pairs[:, 0], pairs[:, 1] = pin, pout
    pairs[:, 2], pairs[:, 3] = din, dout
    pairs[:, 4, 0], pairs[:, 4, 1], pairs[:, 4, 2] = (
        energy_in,
        energy_out,
        1.0,
    )
    return events, pairs, {"energy_invalid": invalid_energy}


def load_run(run_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: read_primary_plane(run_dir / f"{name}.root", name)
        for name in PLANE_NAMES
    }


def angular_error_mrad(
    estimated: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    dot = np.sum(
        np.asarray(estimated, dtype=np.float64)
        * np.asarray(truth, dtype=np.float64),
        axis=1,
    )
    return 1000.0 * np.arccos(np.clip(dot, -1.0, 1.0))
