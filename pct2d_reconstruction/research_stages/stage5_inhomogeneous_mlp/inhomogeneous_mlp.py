"""CPU reference implementation of the Stage-5 inhomogeneous MLP.

The implementation follows the two-stage construction of Brooke and Penfold:
estimate energy from both measured endpoints, form material-dependent
scattering power, integrate the three Fermi--Eyges moments, and condition the
intermediate state on the measured entrance and exit states.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


PROTON_MASS_MEV = 938.27208816
ELECTRON_MASS_MEV = 0.51099895
IONIZATION_POTENTIAL_MEV = 78.0e-6
SCATTERING_POWER_200MEV_PER_MM = 3.645061873086788e-6

# GateMaterials.db compositions used by the truth pilot.
ELEMENTS = {
    "H": (1.0, 1.01),
    "C": (6.0, 12.01),
    "N": (7.0, 14.01),
    "O": (8.0, 16.00),
    "F": (9.0, 18.998),
    "Na": (11.0, 22.99),
    "Mg": (12.0, 24.305),
    "Al": (13.0, 26.981539),
    "P": (15.0, 30.97),
    "S": (16.0, 32.066),
    "Cl": (17.0, 35.45),
    "Ar": (18.0, 39.95),
    "K": (19.0, 39.098),
    "Ca": (20.0, 40.08),
}
MATERIAL_COMPOSITIONS = {
    "Air": (0.00129, {"N": .755268, "O": .231781, "Ar": .012827, "C": .000124}),
    "Lung": (0.26, {"H": .103, "C": .105, "N": .031, "O": .749, "Na": .002, "P": .002, "S": .003, "Cl": .003, "K": .002}),
    "Water": (1.0, {"H": 2 * 1.01 / (2 * 1.01 + 16.0), "O": 16.0 / (2 * 1.01 + 16.0)}),
    "A150_Tissue_Plastic": (1.127, {"H": .101330, "C": .775498, "N": .035057, "O": .052315, "F": .017423, "Ca": .018377}),
    "SpineBone": (1.42, {"H": .063, "C": .261, "N": .039, "O": .436, "Na": .001, "Mg": .001, "P": .061, "S": .003, "Cl": .001, "K": .001, "Ca": .133}),
    "Aluminium": (2.69890, {"Al": 1.0}),
}


def relative_scattering_power(material: str) -> float:
    """Return Gottschalk scattering-length ratio relative to liquid water."""

    def inverse_scattering_length(name: str) -> float:
        density, composition = MATERIAL_COMPOSITIONS[name]
        value = 0.0
        for symbol, fraction in composition.items():
            z, a = ELEMENTS[symbol]
            value += fraction * z * z / a * (
                2.0 * math.log(33219.0 / (a * z ** (1.0 / 3.0))) - 1.0
            )
        return density * value

    # Express the result explicitly as a ratio.  This is algebraically the
    # published scattering-length definition but avoids sensitivity to the
    # rounded liquid-water normalisation constant printed in different
    # versions of the paper.
    return float(
        inverse_scattering_length(material)
        / inverse_scattering_length("Water")
    )


def material_catalog(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "rsp": float(config["materials"][name]["rsp"]),
            "rscp": relative_scattering_power(name),
        }
        for name in MATERIAL_COMPOSITIONS
    }


def published_rscp(rsp: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    rsp = np.asarray(rsp, dtype=np.float32)
    out = np.where(
        rsp < float(mapping["published_break_rsp"]),
        float(mapping["low_slope"]) * rsp,
        float(mapping["solid_slope"]) * rsp + float(mapping["solid_intercept"]),
    )
    return np.clip(out, *map(float, mapping["rscp_clip"])).astype(np.float32)


def catalog_rscp(rsp: np.ndarray, catalog: dict[str, dict[str, float]]) -> np.ndarray:
    anchors = sorted((v["rsp"], v["rscp"]) for v in catalog.values())
    xp = np.asarray([x for x, _ in anchors], dtype=np.float64)
    fp = np.asarray([y for _, y in anchors], dtype=np.float64)
    return np.interp(np.asarray(rsp), xp, fp, left=fp[0], right=fp[-1]).astype(np.float32)


def stopping_power_water(energy_mev: np.ndarray) -> np.ndarray:
    """PCT Bethe--Bloch water stopping power in MeV/mm."""

    energy = np.maximum(np.asarray(energy_mev, dtype=np.float64), 0.001)
    beta2 = 1.0 - (PROTON_MASS_MEV / (energy + PROTON_MASS_MEV)) ** 2
    # Same constant as iterative_reconstruction.physics.
    k = (
        4.0 * np.pi * (2.8179403262e-12 ** 2) * ELECTRON_MASS_MEV
        * 3.343e23 / 1000.0
    )
    return k * (
        np.log(2.0 * ELECTRON_MASS_MEV / IONIZATION_POTENTIAL_MEV * beta2 / (1.0 - beta2))
        - beta2
    ) / beta2


def scattering_energy_factor(energy_mev: np.ndarray) -> np.ndarray:
    energy = np.maximum(np.asarray(energy_mev, dtype=np.float64), 0.1)
    pv = energy * (energy + 2.0 * PROTON_MASS_MEV) / (energy + PROTON_MASS_MEV)
    ref = 200.0 * (200.0 + 2.0 * PROTON_MASS_MEV) / (200.0 + PROTON_MASS_MEV)
    return SCATTERING_POWER_200MEV_PER_MM * (ref / pv) ** 2


def energy_profile(
    energy_in: np.ndarray,
    energy_out: np.ndarray,
    rsp: np.ndarray,
    step_mm: float,
) -> np.ndarray:
    """Blend forward and backward Euler energy estimates."""

    n, samples = rsp.shape
    forward = np.empty((n, samples), dtype=np.float64)
    backward = np.empty_like(forward)
    forward[:, 0] = energy_in
    for j in range(1, samples):
        forward[:, j] = np.maximum(
            0.1,
            forward[:, j - 1]
            - rsp[:, j - 1] * stopping_power_water(forward[:, j - 1]) * step_mm,
        )
    backward[:, -1] = np.maximum(energy_out, 0.1)
    for j in range(samples - 2, -1, -1):
        backward[:, j] = (
            backward[:, j + 1]
            + rsp[:, j + 1] * stopping_power_water(backward[:, j + 1]) * step_mm
        )
    fraction = np.linspace(0.0, 1.0, samples, dtype=np.float64)[None, :]
    return (1.0 - fraction) * forward + fraction * backward


def _inverse2(matrix: np.ndarray) -> np.ndarray:
    det = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    det = np.where(np.abs(det) < 1e-30, np.copysign(1e-30, det + 1e-30), det)
    out = np.empty_like(matrix)
    out[..., 0, 0] = matrix[..., 1, 1] / det
    out[..., 0, 1] = -matrix[..., 0, 1] / det
    out[..., 1, 0] = -matrix[..., 1, 0] / det
    out[..., 1, 1] = matrix[..., 0, 0] / det
    return out


def _states(position: np.ndarray, direction: np.ndarray, axis: int) -> np.ndarray:
    return np.stack(
        [position[:, axis], np.arctan2(direction[:, axis], direction[:, 2])],
        axis=1,
    )


def conditioned_path(
    entry: np.ndarray,
    exit: np.ndarray,
    direction_in: np.ndarray,
    direction_out: np.ndarray,
    z: np.ndarray,
    scattering_power: np.ndarray,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate x/y MLP and conditional lateral standard deviation."""

    n, samples = scattering_power.shape
    ds = float(step_mm)
    coordinate = z[None, :] - entry[:, 2, None]
    length = exit[:, 2] - entry[:, 2]
    valid = (coordinate > 0.0) & (coordinate < length[:, None])
    t = np.where(valid, scattering_power, 0.0)
    s = coordinate
    p0 = np.cumsum(t, axis=1) * ds
    p1 = np.cumsum(t * s, axis=1) * ds
    p2 = np.cumsum(t * s * s, axis=1) * ds
    total0, total1, total2 = p0[:, -1:], p1[:, -1:], p2[:, -1:]
    a0 = p0
    a1 = s * p0 - p1
    a2 = s * s * p0 - 2.0 * s * p1 + p2
    q0, q1, q2 = total0 - p0, total1 - p1, total2 - p2
    b0 = q0
    b1 = length[:, None] * q0 - q1
    b2 = length[:, None] ** 2 * q0 - 2.0 * length[:, None] * q1 + q2
    sigma1 = np.stack([a2, a1, a1, a0], axis=-1).reshape(n, samples, 2, 2)
    sigma2 = np.stack([b2, b1, b1, b0], axis=-1).reshape(n, samples, 2, 2)
    # The common depth grid contains samples outside some oblique chords.
    # Regularise their empty covariance matrices; those samples are masked
    # from every reported metric below.
    sigma1[..., 0, 0] += 1.0e-18
    sigma1[..., 1, 1] += 1.0e-18
    sigma2[..., 0, 0] += 1.0e-18
    sigma2[..., 1, 1] += 1.0e-18
    remain = length[:, None] - s
    r0 = np.zeros_like(sigma1)
    r0[..., 0, 0] = r0[..., 1, 1] = 1.0
    r0[..., 0, 1] = s
    r1 = np.zeros_like(sigma1)
    r1[..., 0, 0] = r1[..., 1, 1] = 1.0
    r1[..., 0, 1] = remain
    r1i = r1.copy()
    r1i[..., 0, 1] *= -1.0
    r1t = np.swapaxes(r1, -1, -2)
    r1ti = np.swapaxes(r1i, -1, -2)
    part1 = r1i @ sigma2 @ _inverse2(r1i @ sigma2 + sigma1 @ r1t) @ r0
    part2 = sigma1 @ _inverse2(r1 @ sigma1 + sigma2 @ r1ti)

    def evaluate(axis: int) -> np.ndarray:
        a = _states(entry, direction_in, axis)[:, None, :, None]
        b = _states(exit, direction_out, axis)[:, None, :, None]
        return (part1 @ a + part2 @ b)[..., 0, 0]

    precision = _inverse2(sigma1) + r1t @ _inverse2(sigma2) @ r1
    posterior = _inverse2(precision)
    std = np.sqrt(np.maximum(posterior[..., 0, 0], 0.0))
    return evaluate(0), evaluate(1), np.where(valid, std, np.nan)


@dataclass
class GeometryMaps:
    rsp: np.ndarray
    rscp: np.ndarray
    material: np.ndarray


def truth_maps(config: dict[str, Any], spacing_mm: float = 0.5) -> GeometryMaps:
    radius = float(config["geometry"]["radius_mm"])
    count = int(round(2 * radius / spacing_mm))
    coordinates = -radius + (np.arange(count) + 0.5) * spacing_mm
    xx, zz = np.meshgrid(coordinates, coordinates)
    catalog = material_catalog(config)
    material = np.full(xx.shape, "Water", dtype="U24")
    for item in config["geometry"]["inserts"]:
        local_x, local_y = map(float, item["center_local_xy_mm"])
        # The simulation's base rotation maps local x -> scanner -z and
        # local y -> scanner -x.  The verified reconstruction transform at
        # angle zero is (xr,zr)=(scanner x,-scanner z).
        cx, cz = -local_y, local_x
        inside = (xx - cx) ** 2 + (zz - cz) ** 2 <= (float(item["diameter_mm"]) / 2) ** 2
        material[inside] = item["material"]
    rsp = np.vectorize(lambda name: catalog[name]["rsp"], otypes=[np.float32])(material)
    rscp = np.vectorize(lambda name: catalog[name]["rscp"], otypes=[np.float32])(material)
    return GeometryMaps(rsp, rscp, material)


def sample_map(
    image: np.ndarray,
    x_scanner: np.ndarray,
    z_scanner: np.ndarray,
    angle_deg: float,
    spacing_mm: float,
) -> np.ndarray:
    """Nearest sample a phantom-fixed map using the verified Stage-4 transform."""

    angle = np.deg2rad(angle_deg)
    xr = np.cos(angle) * x_scanner - np.sin(angle) * z_scanner
    zr = -np.sin(angle) * x_scanner - np.cos(angle) * z_scanner
    origin = -0.5 * image.shape[0] * spacing_mm
    finite = np.isfinite(xr) & np.isfinite(zr)
    ix = np.zeros(xr.shape, dtype=np.int64)
    iz = np.zeros(zr.shape, dtype=np.int64)
    ix[finite] = np.floor((xr[finite] - origin) / spacing_mm).astype(np.int64)
    iz[finite] = np.floor((zr[finite] - origin) / spacing_mm).astype(np.int64)
    valid = finite & (ix >= 0) & (iz >= 0) & (ix < image.shape[1]) & (iz < image.shape[0])
    out = np.zeros(xr.shape, dtype=np.float32)
    out[valid] = image[iz[valid], ix[valid]]
    return out
