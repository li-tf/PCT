"""Stage-3 paths, deterministic masks, pair features, and Air correction."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def splitmix64(values: np.ndarray, run_id: int, seed: int) -> np.ndarray:
    """Versioned unsigned 64-bit mixer; wraparound is intentional."""

    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    x = values.astype(np.uint64, copy=False)
    run_key = np.uint64(
        (int(run_id) * 0xD6E8FEB86659FD93) & 0xFFFFFFFFFFFFFFFF
    )
    x = (x ^ run_key) & mask
    x = (x ^ np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)) & mask
    x = (
        (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    ) & mask
    x = (
        (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    ) & mask
    return (x ^ (x >> np.uint64(31))) & mask


def partition_masks(
    count: int, run_id: int, split: dict[str, Any]
) -> dict[str, np.ndarray]:
    bucket = (
        splitmix64(np.arange(count, dtype=np.uint64), run_id, int(split["seed"]))
        % np.uint64(split["modulus"])
    )
    test = bucket == np.uint64(split["test_remainder"])
    validation = bucket == np.uint64(split["validation_remainder"])
    return {"train": ~(test | validation), "validation": validation, "test": test}


def write_packed_mask(path: Path, mask: np.ndarray, bit_order: str = "little") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.packbits(np.asarray(mask, dtype=bool), bitorder=bit_order).tofile(path)


def read_packed_mask(
    path: Path, count: int, bit_order: str = "little"
) -> np.ndarray:
    packed = np.fromfile(path, dtype=np.uint8)
    required = (count + 7) // 8
    if len(packed) != required:
        raise ValueError(f"{path}: {len(packed)} bytes, expected {required}")
    return np.unpackbits(packed, bitorder=bit_order, count=count).astype(bool)


def itk_round(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0.0, np.floor(values + 0.5), np.ceil(values - 0.5)).astype(
        np.int64
    )


def pair_features(
    pairs: np.ndarray, filtering: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inside-grid, cell id and [energy loss, signed dtheta-x/y]."""

    p_in = np.asarray(pairs[:, 0, :], dtype=np.float64)
    p_out = np.asarray(pairs[:, 1, :], dtype=np.float64)
    d_in = np.asarray(pairs[:, 2, :], dtype=np.float64)
    d_out = np.asarray(pairs[:, 3, :], dtype=np.float64)
    data = np.asarray(pairs[:, 4, :], dtype=np.float64)
    source = float(filtering["source_mm"])
    grid_size = tuple(int(value) for value in filtering["grid_size"])
    spacing = tuple(float(value) for value in filtering["grid_spacing_mm"])
    origin = tuple(float(value) for value in filtering["grid_origin_mm"])
    magnification = (source - p_out[0, 2]) / (source - p_in[0, 2])
    i = itk_round((p_in[:, 0] * magnification - origin[0]) / spacing[0])
    j = itk_round((p_in[:, 1] * magnification - origin[1]) / spacing[1])
    inside = (i >= 0) & (i < grid_size[0]) & (j >= 0) & (j < grid_size[1])
    cell = i + j * grid_size[0]
    def signed_projected_angle(axis: int) -> np.ndarray:
        dot = d_in[:, axis] * d_out[:, axis] + d_in[:, 2] * d_out[:, 2]
        norm_in = np.hypot(d_in[:, axis], d_in[:, 2])
        norm_out = np.hypot(d_out[:, axis], d_out[:, 2])
        magnitude = np.arccos(np.minimum(1.0, dot / (norm_in * norm_out)))
        theta_in = np.arctan2(d_in[:, axis], d_in[:, 2])
        theta_out = np.arctan2(d_out[:, axis], d_out[:, 2])
        sign = np.sign(
            np.arctan2(
                np.sin(theta_out - theta_in), np.cos(theta_out - theta_in)
            )
        )
        return magnitude * sign

    dtheta_x = signed_projected_angle(0)
    dtheta_y = signed_projected_angle(1)
    energy_loss = np.where(data[:, 0] == 0.0, data[:, 1], data[:, 0] - data[:, 1])
    features = np.column_stack((energy_loss, dtheta_x, dtheta_y))
    return inside, cell, features


def forward_distance_to_cylinder(
    position: np.ndarray, direction: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    position = np.asarray(position, dtype=np.float64)
    x, z = position[:, 0], position[:, 2]
    dx, dz = direction[:, 0], direction[:, 2]
    a = dx * dx + dz * dz
    b = 2.0 * (x * dx + z * dz)
    c = x * x + z * z - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid = (a > 1.0e-12) & (discriminant >= 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    candidates = np.column_stack(
        ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a))
    )
    candidates[candidates < 0.0] = np.inf
    distance = np.min(candidates, axis=1)
    valid &= np.isfinite(distance)
    return distance, valid


def air_correct_pairs(
    pairs: np.ndarray,
    lut: np.ndarray,
    correction: dict[str, Any],
    energies_to_wepl,
) -> np.ndarray:
    """Return a copy storing Air-corrected WEPL as (0, WEPL)."""

    radius = float(correction["phantom_radius_mm"])
    entrance, entrance_valid = forward_distance_to_cylinder(
        pairs[:, 0, :], pairs[:, 2, :], radius
    )
    exit_distance, exit_valid = forward_distance_to_cylinder(
        pairs[:, 1, :], -pairs[:, 3, :], radius
    )
    hit = entrance_valid & exit_valid
    air_length = entrance + exit_distance
    direct = np.linalg.norm(
        pairs[:, 1, :].astype(np.float64) - pairs[:, 0, :].astype(np.float64),
        axis=1,
    )
    air_length[~hit] = direct[~hit]
    original = energies_to_wepl(lut, pairs[:, 4, 0], pairs[:, 4, 1])
    corrected = np.maximum(
        np.asarray(original, dtype=np.float64)
        - float(correction["slope_mm_wepl_per_mm_air"]) * air_length,
        0.0,
    )
    output = np.asarray(pairs, dtype=np.float32).copy()
    output[:, 4, 0] = 0.0
    output[:, 4, 1] = corrected.astype(np.float32)
    return output


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.dot(weights, weights))
    return float(weights.sum() ** 2 / denominator) if denominator > 0 else 0.0


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
