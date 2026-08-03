"""Geometry, deterministic splitting, MLP and voxel truth for compact 3-D pCT."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & UINT64_MASK
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & UINT64_MASK
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & UINT64_MASK
    return x ^ (x >> np.uint64(31))


def identity_hash(run_id: int, event_id: np.ndarray, seed: int) -> np.ndarray:
    event = np.asarray(event_id, dtype=np.uint64)
    key = event ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        run_key = np.multiply(
            np.uint64(run_id), np.uint64(0xD6E8FEB86659FD93), dtype=np.uint64
        )
    key ^= run_key
    return splitmix64(key)


def partition_codes(run_id: int, event_id: np.ndarray, seed: int) -> np.ndarray:
    bucket = identity_hash(run_id, event_id, seed) % np.uint64(10)
    result = np.zeros(len(bucket), dtype=np.uint8)
    result[bucket == 8] = 1
    result[bucket == 9] = 2
    return result  # train=0, validation=1, test=2


def scanner_to_object(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Undo the OpenGATE active +y phantom rotation."""

    points = np.asarray(points, dtype=np.float64)
    angle = np.deg2rad(angle_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    result = np.array(points, copy=True)
    result[..., 0] = cosine * points[..., 0] - sine * points[..., 2]
    result[..., 2] = sine * points[..., 0] + cosine * points[..., 2]
    return result


def screen_train_mask(run_id: int, event_id: np.ndarray, seed: int) -> np.ndarray:
    return identity_hash(run_id, event_id, seed + 0x13579BDF) % np.uint64(10) == 0


def ray_finite_cylinder_interval(
    position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    half_y: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward intersection interval with x²+z²<=R² and |y|<=half_y."""

    p = np.asarray(position, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d, axis=1)
    unit = np.divide(d, norm[:, None], out=np.zeros_like(d), where=norm[:, None] > 1e-12)
    a = unit[:, 0] ** 2 + unit[:, 2] ** 2
    b = 2.0 * (p[:, 0] * unit[:, 0] + p[:, 2] * unit[:, 2])
    c = p[:, 0] ** 2 + p[:, 2] ** 2 - radius**2
    disc = b * b - 4.0 * a * c
    valid = (norm > 1e-12) & (a > 1e-12) & (disc >= 0.0)
    root = np.sqrt(np.maximum(disc, 0.0))
    den = np.where(a > 1e-12, 2.0 * a, 1.0)
    radial_lo = (-b - root) / den
    radial_hi = (-b + root) / den
    dy = unit[:, 1]
    parallel = np.abs(dy) <= 1e-12
    y_lo = np.full(len(p), -np.inf)
    y_hi = np.full(len(p), np.inf)
    moving = ~parallel
    t1 = np.divide(-half_y - p[:, 1], dy, out=np.zeros(len(p)), where=moving)
    t2 = np.divide(half_y - p[:, 1], dy, out=np.zeros(len(p)), where=moving)
    y_lo[moving] = np.minimum(t1[moving], t2[moving])
    y_hi[moving] = np.maximum(t1[moving], t2[moving])
    valid &= ~parallel | (np.abs(p[:, 1]) <= half_y)
    enter = np.maximum.reduce([radial_lo, y_lo, np.zeros(len(p))])
    leave = np.minimum(radial_hi, y_hi)
    valid &= leave > enter
    return enter, leave, valid


def subtract_external_air(
    pairs: np.ndarray,
    wepl: np.ndarray,
    radius: float,
    half_y: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    tin, _, vin = ray_finite_cylinder_interval(pairs[:, 0], pairs[:, 2], radius, half_y)
    tout, _, vout = ray_finite_cylinder_interval(pairs[:, 1], -pairs[:, 3], radius, half_y)
    valid = vin & vout
    corrected = np.asarray(wepl, dtype=np.float64) - slope * (tin + tout)
    return np.maximum(corrected, 0.0).astype(np.float32), valid


def load_wepl_model(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    energy = np.asarray(payload["energy_mev"], dtype=np.float64)
    ranges = np.asarray(payload["range_mm"], dtype=np.float64)
    if len(energy) != len(ranges) or np.any(np.diff(energy) <= 0) or np.any(np.diff(ranges) <= 0):
        raise ValueError("invalid monotonic WEPL model")
    return energy, ranges, str(payload.get("model_sha256", ""))


def energies_to_wepl(path: Path, energy_in: np.ndarray, energy_out: np.ndarray) -> np.ndarray:
    energy, ranges, _ = load_wepl_model(path)
    ein = np.asarray(energy_in, dtype=np.float64)
    eout = np.asarray(energy_out, dtype=np.float64)
    if np.any(ein < energy[0]) or np.any(ein > energy[-1]) or np.any(eout < energy[0]) or np.any(eout > energy[-1]):
        raise ValueError("energy outside calibrated model")
    result = np.interp(ein, energy, ranges) - np.interp(eout, energy, ranges)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("invalid calibrated WEPL")
    return result.astype(np.float32)


def _integrals(u: float) -> tuple[float, float, float]:
    coeff = [7.444724e-6, 5.463937e-8, -9.986645e-10, 2.026409e-11, -1.420501e-13, 3.899100e-16]
    theta = ttheta = trans = 0.0
    power = u
    for k, value in enumerate(coeff):
        theta += value * power / (k + 1)
        ttheta += value * power * u / (k + 2)
        trans += value * power * u * u / (k + 3)
        power *= u
    return theta, ttheta, trans


def mlp_position_cpu(z: float, entry: np.ndarray, exit_: np.ndarray, din: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """Schulte water MLP for both transverse coordinates at scanner z."""

    length = float(exit_[2] - entry[2])
    u = float(np.clip(z - entry[2], 1e-6, length - 1e-6))
    remaining = length - u
    tht, ttht, trans = _integrals(length)
    th1, tth1, t1 = _integrals(u)
    s101 = u * th1 - tth1
    s1 = np.array([[u * (2 * s101 - u * th1) + t1, s101], [s101, th1]])
    c1 = (13.6**2 / 361.0) * (1.0 + 0.038 * np.log(max(u, 1e-3) / 361.0)) ** 2
    s1 *= c1
    th2 = tht - th1
    s201 = length * th2 - ttht + tth1
    s2 = np.array([[length * (2 * s201 - length * th2) + trans - t1, s201], [s201, th2]])
    c2 = (13.6**2 / 361.0) * (
        1.0 + 0.038 * np.log(max(length - u, 1e-3) / 361.0)
    ) ** 2
    s2 *= c2
    r0 = np.array([[1.0, u], [0.0, 1.0]])
    r1 = np.array([[1.0, remaining], [0.0, 1.0]])
    r1i = np.array([[1.0, -remaining], [0.0, 1.0]])
    part1 = r1i @ s2 @ np.linalg.inv(r1i @ s2 + s1 @ r1.T) @ r0
    part2 = s1 @ np.linalg.inv(r1 @ s1 + s2 @ r1i.T)
    result = np.empty(3)
    for axis in (0, 1):
        state_in = np.array([entry[axis], np.arctan(din[axis] / din[2])])
        state_out = np.array([exit_[axis], np.arctan(dout[axis] / dout[2])])
        result[axis] = (part1 @ state_in + part2 @ state_out)[0]
    result[2] = z
    return result


def load_mlic(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {row["material"]: float(row["mlic_rsp_200mev"]) for row in rows}


def coordinates(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = config["grid"]["size_xyz"]
    spacing = config["grid"]["spacing_xyz_mm"]
    origin = config["grid"]["origin_xyz_mm"]
    return tuple(origin[i] + np.arange(size[i]) * spacing[i] for i in range(3))  # type: ignore[return-value]


def support_mask(config: dict) -> np.ndarray:
    x, y, z = coordinates(config)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return (xx * xx + zz * zz <= config["phantom_radius_mm"] ** 2) & (
        np.abs(yy) <= config["phantom_half_length_y_mm"]
    )


def build_truth(config: dict, simulation: dict, mlic: dict[str, float], supersample: int = 2) -> np.ndarray:
    """Voxel-volume average using deterministic subvoxel sampling."""

    x, y, z = coordinates(config)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    support = support_mask(config)
    truth = np.zeros(support.shape, dtype=np.float32)
    truth[support] = float(mlic["Water"])
    spacing = np.asarray(config["grid"]["spacing_xyz_mm"])
    offsets = ((np.arange(supersample) + 0.5) / supersample - 0.5)
    for item in simulation["spheres"]:
        center = np.asarray(item["scanner_center_mm"], dtype=float)
        radius = float(item["diameter_mm"]) / 2.0
        material = item["material"]
        value = 0.0011471876206752695 if material == "Air" else float(mlic[material])
        fraction = np.zeros_like(truth, dtype=np.float32)
        for oz in offsets:
            for oy in offsets:
                for ox in offsets:
                    inside = (
                        (xx + ox * spacing[0] - center[0]) ** 2
                        + (yy + oy * spacing[1] - center[1]) ** 2
                        + (zz + oz * spacing[2] - center[2]) ** 2
                        <= radius**2
                    )
                    fraction += inside.astype(np.float32)
        fraction /= supersample**3
        truth = truth * (1.0 - fraction) + value * fraction
    truth[~support] = 0.0
    return truth
