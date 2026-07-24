"""Vectorized Schulte most-likely-path implementation matching PCT."""

from __future__ import annotations

import numpy as np


# PCT uses millimetres internally. The published coefficients use powers of cm.
COEFFICIENTS = np.array(
    [
        7.444724e-6,
        5.463937e-7 / 10.0,
        -9.986645e-8 / 10.0**2,
        2.026409e-8 / 10.0**3,
        -1.420501e-9 / 10.0**4,
        3.899100e-11 / 10.0**5,
    ],
    dtype=np.float64,
)
RADIATION_LENGTH_MM = 361.0


def _integrals(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    powers = np.stack([u ** (index + 1) for index in range(6)], axis=-1)
    theta = np.sum(powers * COEFFICIENTS / np.arange(1, 7), axis=-1)
    ttheta = np.sum(powers * u[..., None] * COEFFICIENTS / np.arange(2, 8), axis=-1)
    t = np.sum(powers * u[..., None] ** 2 * COEFFICIENTS / np.arange(3, 9), axis=-1)
    return theta, ttheta, t


def _constant(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    distance = np.maximum(uy - ux, 1.0e-3)
    return (13.6**2 / RADIATION_LENGTH_MM) * (1.0 + 0.038 * np.log(distance / RADIATION_LENGTH_MM)) ** 2


def _inverse_2x2(matrix: np.ndarray) -> np.ndarray:
    determinant = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    safe = np.where(np.abs(determinant) < 1.0e-30, np.copysign(1.0e-30, determinant + 1.0e-30), determinant)
    result = np.empty_like(matrix)
    result[..., 0, 0] = matrix[..., 1, 1] / safe
    result[..., 0, 1] = -matrix[..., 0, 1] / safe
    result[..., 1, 0] = -matrix[..., 1, 0] / safe
    result[..., 1, 1] = matrix[..., 0, 0] / safe
    return result


def schulte_positions(
    position_in: np.ndarray,
    position_out: np.ndarray,
    direction_in: np.ndarray,
    direction_out: np.ndarray,
    z_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scanner-frame x/y positions and a valid-depth mask."""

    z = np.asarray(z_mm, dtype=np.float64)[None, :]
    origin = position_in[:, 2:3].astype(np.float64)
    length = (position_out[:, 2] - position_in[:, 2]).astype(np.float64)[:, None]
    u = z - origin
    valid = (u > 1.0e-6) & (u < length - 1.0e-6)
    uc = np.minimum(np.maximum(u, 1.0e-6), length - 1.0e-6)
    remaining = length - uc

    theta1, ttheta1, t1 = _integrals(uc)
    theta2_total, ttheta2_total, t2_total = _integrals(length)

    sigma1 = np.zeros(uc.shape + (2, 2), dtype=np.float64)
    sigma1[..., 1, 1] = theta1
    sigma1[..., 0, 1] = uc * theta1 - ttheta1
    sigma1[..., 1, 0] = sigma1[..., 0, 1]
    sigma1[..., 0, 0] = uc * (2.0 * sigma1[..., 0, 1] - uc * theta1) + t1
    sigma1 *= _constant(np.zeros_like(uc), uc)[..., None, None]

    sigma2 = np.zeros_like(sigma1)
    sigma2[..., 1, 1] = theta2_total - theta1
    sigma2[..., 0, 1] = length * sigma2[..., 1, 1] - ttheta2_total + ttheta1
    sigma2[..., 1, 0] = sigma2[..., 0, 1]
    sigma2[..., 0, 0] = length * (2.0 * sigma2[..., 0, 1] - length * sigma2[..., 1, 1]) + t2_total - t1
    sigma2 *= _constant(uc, length)[..., None, None]

    r0 = np.zeros_like(sigma1)
    r0[..., 0, 0] = 1.0
    r0[..., 0, 1] = uc
    r0[..., 1, 1] = 1.0
    r1 = np.zeros_like(sigma1)
    r1[..., 0, 0] = 1.0
    r1[..., 0, 1] = remaining
    r1[..., 1, 1] = 1.0
    r1_inv = np.zeros_like(r1)
    r1_inv[..., 0, 0] = 1.0
    r1_inv[..., 0, 1] = -remaining
    r1_inv[..., 1, 1] = 1.0
    r1_t = np.swapaxes(r1, -1, -2)
    r1_t_inv = np.swapaxes(r1_inv, -1, -2)

    sum1 = np.matmul(r1_inv, sigma2) + np.matmul(sigma1, r1_t)
    sum2 = np.matmul(r1, sigma1) + np.matmul(sigma2, r1_t_inv)
    part1 = np.matmul(np.matmul(np.matmul(r1_inv, sigma2), _inverse_2x2(sum1)), r0)
    part2 = np.matmul(sigma1, _inverse_2x2(sum2))

    din = direction_in.astype(np.float64)
    dout = direction_out.astype(np.float64)
    slope_in_x = din[:, 0] / din[:, 2]
    slope_in_y = din[:, 1] / din[:, 2]
    slope_out_x = dout[:, 0] / dout[:, 2]
    slope_out_y = dout[:, 1] / dout[:, 2]
    state_in_x = np.stack([position_in[:, 0], np.arctan(slope_in_x)], axis=1)
    state_in_y = np.stack([position_in[:, 1], np.arctan(slope_in_y)], axis=1)
    state_out_x = np.stack([position_out[:, 0], np.arctan(slope_out_x)], axis=1)
    state_out_y = np.stack([position_out[:, 1], np.arctan(slope_out_y)], axis=1)

    def evaluate(state_in: np.ndarray, state_out: np.ndarray) -> np.ndarray:
        first = np.matmul(part1, state_in[:, None, :, None])[..., 0, 0]
        second = np.matmul(part2, state_out[:, None, :, None])[..., 0, 0]
        return first + second

    return evaluate(state_in_x, state_out_x), evaluate(state_in_y, state_out_y), valid


def cylinder_intersections(
    position_in: np.ndarray,
    position_out: np.ndarray,
    direction_in: np.ndarray,
    direction_out: np.ndarray,
    radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect entrance/exit rays with the x-z water-cylinder boundary."""

    def roots(position: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = direction[:, 0] ** 2 + direction[:, 2] ** 2
        b = 2.0 * (position[:, 0] * direction[:, 0] + position[:, 2] * direction[:, 2])
        c = position[:, 0] ** 2 + position[:, 2] ** 2 - radius_mm**2
        discriminant = b * b - 4.0 * a * c
        valid = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        return (-b - root) / (2.0 * a), (-b + root) / (2.0 * a), valid

    in_near, in_far, valid_in = roots(position_in, direction_in)
    out_near, out_far, valid_out = roots(position_out, direction_out)
    entry_t = np.where(in_near >= 0.0, in_near, in_far)
    exit_t = np.where(out_far <= 0.0, out_far, out_near)
    entry = position_in + entry_t[:, None] * direction_in
    exit = position_out + exit_t[:, None] * direction_out
    valid = valid_in & valid_out & (entry_t >= 0.0) & (exit_t <= 0.0) & (exit[:, 2] > entry[:, 2])
    return entry, exit, valid
