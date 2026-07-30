"""Streaming preparation of the Geant4 truth-trajectory pilot."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import uproot

from inhomogeneous_mlp import sample_map, truth_maps


REFERENCE = [
    "EventID", "TrackID", "KineticEnergy", "Position_X", "Position_Y",
    "Position_Z", "Direction_X", "Direction_Y", "Direction_Z",
]
TRAJECTORY = [
    "EventID", "TrackID", "ParentID", "PreGlobalTime",
    "PrePosition_X", "PrePosition_Y", "PrePosition_Z",
    "PostPosition_X", "PostPosition_Y", "PostPosition_Z",
]


def arrays_by_basket(tree, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for name in names:
        branch = tree[name]
        if hasattr(branch, "num_baskets"):
            pieces = [
                branch.basket(i).array(library="np")
                for i in range(branch.num_baskets)
            ]
            out[name] = (
                np.concatenate(pieces)
                if len(pieces) > 1
                else pieces[0]
            )
        else:
            out[name] = branch.array(library="np")
    return out


def read_tree(path: Path, tree_name: str, fields: list[str]) -> dict[str, np.ndarray]:
    with uproot.open(path) as root:
        if tree_name not in root:
            raise RuntimeError(f"{path}: missing tree {tree_name}")
        tree = root[tree_name]
        missing = sorted(set(fields) - set(tree.keys()))
        if missing:
            raise RuntimeError(f"{path}: missing branches {missing}")
        return arrays_by_basket(tree, fields)


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def _state_rows(values: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    primary = values["TrackID"] == 1
    result = {}
    for index in np.flatnonzero(primary):
        event = int(values["EventID"][index])
        if event in result:
            continue
        result[event] = np.array([
            values["Position_X"][index], values["Position_Y"][index],
            values["Position_Z"][index], values["Direction_X"][index],
            values["Direction_Y"][index], values["Direction_Z"][index],
            values["KineticEnergy"][index],
        ], dtype=np.float32)
    return result


def _joint_cut(states_in: np.ndarray, states_out: np.ndarray) -> np.ndarray:
    delta_e = states_in[:, 6] - states_out[:, 6]
    ain_x = np.arctan2(states_in[:, 3], states_in[:, 5])
    ain_y = np.arctan2(states_in[:, 4], states_in[:, 5])
    out_x = np.arctan2(states_out[:, 3], states_out[:, 5])
    out_y = np.arctan2(states_out[:, 4], states_out[:, 5])
    features = np.column_stack([delta_e, out_x - ain_x, out_y - ain_y])
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0, ddof=1)
    scale = np.maximum(scale, np.array([0.1, 1e-5, 1e-5]))
    return np.all(np.abs(features - center) <= 3.0 * scale, axis=1)


def prepare_one(
    run_id: int,
    input_root: str,
    output_root: str,
    samples: int,
    step_mm: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = Path(input_root) / f"run_{run_id:03d}"
    output = Path(output_root) / f"truth_{run_id:03d}.npz"
    if output.is_file():
        with np.load(output) as cache:
            return {"run": run_id, "usable": int(len(cache["event_id"])), "cached": True}
    state_in = _state_rows(read_tree(source / "PhaseSpaceIn.root", "PhaseSpaceIn", REFERENCE))
    state_out = _state_rows(read_tree(source / "PhaseSpaceOut.root", "PhaseSpaceOut", REFERENCE))
    common = sorted(set(state_in) & set(state_out))
    pin = np.stack([state_in[e] for e in common])
    pout = np.stack([state_out[e] for e in common])
    accepted = _joint_cut(pin, pout)
    accepted_events = np.asarray(common, dtype=np.int64)[accepted]
    pin, pout = pin[accepted], pout[accepted]
    index_for = {int(event): i for i, event in enumerate(accepted_events)}

    trajectory = read_tree(source / "PrimaryTrajectory.root", "PrimaryTrajectory", TRAJECTORY)
    primary = (trajectory["TrackID"] == 1) & (trajectory["ParentID"] == 0)
    order = np.lexsort((trajectory["PreGlobalTime"][primary], trajectory["EventID"][primary]))
    values = {key: value[primary][order] for key, value in trajectory.items()}
    events = values["EventID"]
    boundaries = np.r_[0, 1 + np.flatnonzero(events[1:] != events[:-1]), len(events)]
    z_grid = -100.0 + (np.arange(samples) + 0.5) * step_mm
    true_x = np.full((len(accepted_events), samples), np.nan, np.float32)
    true_y = np.full_like(true_x, np.nan)
    rejected_nonmonotonic = 0
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        event = int(events[begin])
        target = index_for.get(event)
        if target is None:
            continue
        z = np.r_[values["PrePosition_Z"][begin:end], values["PostPosition_Z"][end - 1]]
        x = np.r_[values["PrePosition_X"][begin:end], values["PostPosition_X"][end - 1]]
        y = np.r_[values["PrePosition_Y"][begin:end], values["PostPosition_Y"][end - 1]]
        finite = np.isfinite(z) & np.isfinite(x) & np.isfinite(y)
        z, x, y = z[finite], x[finite], y[finite]
        if len(z) < 3 or np.any(np.diff(z) < -0.05):
            rejected_nonmonotonic += 1
            continue
        keep = np.r_[True, np.diff(z) > 1e-6]
        z, x, y = z[keep], x[keep], y[keep]
        inside = (z_grid >= z[0]) & (z_grid <= z[-1])
        true_x[target, inside] = np.interp(z_grid[inside], z, x)
        true_y[target, inside] = np.interp(z_grid[inside], z, y)
    # Retain peripheral chords as well as central tracks.  Ten percent of the
    # diameter supplies at least 20 mm of truth path at the formal 0.5 mm grid.
    usable = np.sum(np.isfinite(true_x), axis=1) >= max(10, int(0.1 * samples))
    event_id = accepted_events[usable]
    pin, pout = pin[usable], pout[usable]
    true_x, true_y = true_x[usable], true_y[usable]

    maps = truth_maps(config, step_mm)
    material_codes = np.full(true_x.shape, 255, np.uint8)
    for code, material in enumerate(("Air", "Lung", "Water", "A150_Tissue_Plastic", "SpineBone", "Aluminium")):
        mask = sample_map(
            (maps.material == material).astype(np.float32),
            true_x, np.broadcast_to(z_grid, true_x.shape), run_id * 5.0, step_mm,
        ) > 0.5
        material_codes[mask & np.isfinite(true_x)] = code
    split_key = (
        np.uint64(run_id) * np.uint64(0x9E3779B97F4A7C15)
        + event_id.astype(np.uint64) + np.uint64(config["truth_pilot"]["seed"])
    )
    split_key ^= split_key >> np.uint64(30)
    split_key *= np.uint64(0xBF58476D1CE4E5B9)
    split_key ^= split_key >> np.uint64(27)
    split_key *= np.uint64(0x94D049BB133111EB)
    split_key ^= split_key >> np.uint64(31)
    remainder = (split_key % np.uint64(10)).astype(np.uint8)
    split = np.where(remainder == 0, 2, np.where(remainder == 1, 1, 0)).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, event_id=event_id, state_in=pin, state_out=pout,
        true_x=true_x, true_y=true_y, material=material_codes, split=split,
    )
    return {
        "run": run_id, "incident": int(config["truth_pilot"]["incident_protons_per_angle"]),
        "paired": len(common), "accepted": int(np.count_nonzero(accepted)),
        "usable": int(len(event_id)), "rejected_nonmonotonic": rejected_nonmonotonic,
        "cached": False,
    }


def prepare_all(
    config: dict[str, Any], repository_root: Path, jobs: int,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    pilot = config["truth_pilot"]
    source = repository_root / pilot["simulation_data"]
    output = repository_root / pilot["cache_data"]
    expected = int(pilot["angles"])
    missing = [
        str(source / f"run_{i:03d}" / name)
        for i in range(expected)
        for name in ("PhaseSpaceIn.root", "PhaseSpaceOut.root", "PrimaryTrajectory.root")
        if not (source / f"run_{i:03d}" / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"truth pilot is incomplete; first missing file: {missing[0]}")
    samples = int(round(200.0 / float(pilot["evaluation_step_mm"])))
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                prepare_one, i, str(source), str(output), samples,
                float(pilot["evaluation_step_mm"]), config,
            ): i for i in range(expected)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if progress:
                progress(completed, expected, sum(r["usable"] for r in rows))
    return sorted(rows, key=lambda row: row["run"])
