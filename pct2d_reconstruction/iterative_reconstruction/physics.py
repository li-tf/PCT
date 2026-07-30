"""WEPL conversion using the exact wrapped PCT Bethe--Bloch LUT."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


IONIZATION_POTENTIAL_MEV = 78.0e-6
MAX_ENERGY_MEV = 600.0
ENERGY_BIN_MEV = 0.0001

# CLHEP constants in the unit system used by pctBetheBlochFunctor.h.  Keeping
# the LUT construction here makes full list-mode reconstruction vectorized;
# calling the wrapped C++ GetValue once per proton is prohibitively slow.
ELECTRON_MASS_MEV = 0.51099895
PROTON_MASS_MEV = 938.27208816
CLASSICAL_ELECTRON_RADIUS_MM = 2.8179403262e-12
ELECTRON_DENSITY_PER_CM3 = 3.343e23


@dataclass(frozen=True)
class WeplModel:
    """Versioned range--energy model used by list-mode reconstruction."""

    name: str
    energy_mev: np.ndarray
    range_mm: np.ndarray
    sha256: str
    source: str

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sha256": self.sha256,
            "source": self.source,
            "energy_min_mev": float(self.energy_mev[0]),
            "energy_max_mev": float(self.energy_mev[-1]),
            "samples": int(
                self.range_mm.size if self.name == "bb78" else self.energy_mev.size
            ),
        }


def make_wepl_converter():
    import itk
    from itk import PCT as pct

    converter_type = pct.IntegratedBetheBlochProtonStoppingPowerInverse[itk.F, itk.D]
    return converter_type(IONIZATION_POTENTIAL_MEV, MAX_ENERGY_MEV, ENERGY_BIN_MEV, "proton")


def energies_to_wepl(converter, energy_in: np.ndarray, energy_out: np.ndarray) -> np.ndarray:
    if energy_in.shape != energy_out.shape:
        raise ValueError("entrance and exit energies must have identical shapes")
    values = np.fromiter(
        (converter.GetValue(float(eout), float(ein)) for ein, eout in zip(energy_in, energy_out)),
        dtype=np.float64,
        count=energy_in.size,
    )
    return values.astype(np.float32)


def make_vectorized_wepl_lut(
    max_energy_mev: float = MAX_ENERGY_MEV,
    energy_bin_mev: float = ENERGY_BIN_MEV,
) -> np.ndarray:
    """Build the same nearest-bin integral LUT as the PCT C++ functor.

    The expression and accumulation order follow ``pctBetheBlochFunctor.h``.
    The returned array is float64 so subtraction remains accurate before WEPL
    values are converted to float32 for reconstruction.
    """

    number_of_bins = int(np.ceil(max_energy_mev / energy_bin_mev))
    low_bin = int(np.ceil(0.001 / energy_bin_mev))  # C++ low limit: 1 keV
    indices = np.arange(low_bin, number_of_bins, dtype=np.float64)
    energy = indices * energy_bin_mev
    beta_squared = 1.0 - (PROTON_MASS_MEV / (energy + PROTON_MASS_MEV)) ** 2
    k = (
        4.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_MM**2
        * ELECTRON_MASS_MEV
        * ELECTRON_DENSITY_PER_CM3
        / 1000.0  # cm^3 -> mm^3
    )
    stopping_power = k * (
        np.log(
            2.0
            * ELECTRON_MASS_MEV
            / IONIZATION_POTENTIAL_MEV
            * beta_squared
            / (1.0 - beta_squared)
        )
        - beta_squared
    ) / beta_squared
    lut = np.zeros(number_of_bins, dtype=np.float64)
    lut[low_bin:] = np.cumsum(energy_bin_mev / stopping_power, dtype=np.float64)
    return lut


def energies_to_wepl_vectorized(
    lut: np.ndarray,
    energy_in: np.ndarray,
    energy_out: np.ndarray,
    energy_bin_mev: float = ENERGY_BIN_MEV,
) -> np.ndarray:
    """Convert arrays of energies with the PCT LUT's nearest-bin lookup."""

    if energy_in.shape != energy_out.shape:
        raise ValueError("entrance and exit energies must have identical shapes")
    energy_in = np.asarray(energy_in, dtype=np.float64)
    energy_out = np.asarray(energy_out, dtype=np.float64)
    direct = energy_in == 0.0
    in_index = np.floor(energy_in / energy_bin_mev + 0.5).astype(np.int64)
    out_index = np.floor(energy_out / energy_bin_mev + 0.5).astype(np.int64)
    if (
        np.any(in_index < 0)
        or np.any(out_index < 0)
        or np.any(in_index >= lut.size)
        or np.any(out_index >= lut.size)
    ):
        raise ValueError("energy is outside the configured Bethe--Bloch LUT")
    values = lut[in_index] - lut[out_index]
    # Match pctProtonPairsToDistanceDrivenProjection: an entrance-energy slot
    # equal to zero marks a directly stored WEPL in the exit-energy slot.
    values[direct] = energy_out[direct]
    return values.astype(np.float32)


def load_wepl_model(
    name: str = "bb78",
    calibration_path: str | Path | None = None,
) -> WeplModel:
    """Load the historical BB78 model or a frozen calibrated range table.

    The calibrated JSON format is intentionally small and portable.  It must
    contain ``energy_mev`` and ``range_mm`` arrays plus a ``model_name``.
    Strict monotonicity is checked before the table can be used.
    """

    if name == "bb78":
        ranges = make_vectorized_wepl_lut()
        energy = np.array(
            [0.0, (ranges.size - 1) * ENERGY_BIN_MEV], dtype=np.float64
        )
        digest = hashlib.sha256(ranges.tobytes()).hexdigest()
        return WeplModel(
            name="bb78",
            energy_mev=energy,
            range_mm=ranges,
            sha256=digest,
            source="PCT I=78 eV integrated Bethe--Bloch LUT",
        )
    if name != "g4_water_calibrated":
        raise ValueError(f"unknown WEPL model: {name}")
    if calibration_path is None:
        raise ValueError("g4_water_calibrated requires --wepl-calibration")
    path = Path(calibration_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    energy = np.asarray(payload["energy_mev"], dtype=np.float64)
    ranges = np.asarray(payload["range_mm"], dtype=np.float64)
    if (
        energy.ndim != 1
        or ranges.shape != energy.shape
        or energy.size < 2
        or not np.isfinite(energy).all()
        or not np.isfinite(ranges).all()
        or np.any(np.diff(energy) <= 0.0)
        or np.any(np.diff(ranges) <= 0.0)
        or energy[0] > 0.0
        or ranges[0] < 0.0
    ):
        raise ValueError(f"invalid calibrated range table: {path}")
    canonical = json.dumps(
        {
            "model_name": payload.get("model_name"),
            "energy_mev": payload["energy_mev"],
            "range_mm": payload["range_mm"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    expected = payload.get("model_sha256")
    if expected is not None and expected != digest:
        raise ValueError(
            f"calibration hash mismatch: expected {expected}, calculated {digest}"
        )
    return WeplModel(
        name="g4_water_calibrated",
        energy_mev=energy,
        range_mm=ranges,
        sha256=digest,
        source=str(path),
    )


def energies_to_wepl_model(
    model: WeplModel,
    energy_in: np.ndarray,
    energy_out: np.ndarray,
) -> np.ndarray:
    """Convert measured energies using a versioned range--energy model."""

    if model.name == "bb78":
        return energies_to_wepl_vectorized(
            model.range_mm, energy_in, energy_out, ENERGY_BIN_MEV
        )
    energy_in = np.asarray(energy_in, dtype=np.float64)
    energy_out = np.asarray(energy_out, dtype=np.float64)
    if energy_in.shape != energy_out.shape:
        raise ValueError("entrance and exit energies must have identical shapes")
    direct = energy_in == 0.0
    non_direct = ~direct
    if np.any(non_direct):
        low = float(model.energy_mev[0])
        high = float(model.energy_mev[-1])
        values = np.concatenate((energy_in[non_direct], energy_out[non_direct]))
        if np.any(values < low) or np.any(values > high):
            observed = (float(values.min()), float(values.max()))
            raise ValueError(
                f"energy {observed} MeV is outside {model.name} range "
                f"[{low:g}, {high:g}] MeV"
            )
    result = np.empty(energy_in.shape, dtype=np.float64)
    rin = np.interp(energy_in[non_direct], model.energy_mev, model.range_mm)
    rout = np.interp(energy_out[non_direct], model.energy_mev, model.range_mm)
    result[non_direct] = rin - rout
    result[direct] = energy_out[direct]
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("WEPL conversion produced non-finite or negative values")
    return result.astype(np.float32)


def subtract_external_air_wepl(
    pairs: np.ndarray,
    wepl_mm: np.ndarray,
    phantom_radius_mm: float,
    slope_mm_wepl_per_mm_air: float,
) -> np.ndarray:
    """Subtract calibrated Air WEPL outside a cylindrical x-z support."""

    def forward_distance(position: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(position, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction, axis=1)
        valid = norm > 1.0e-12
        unit = np.zeros_like(direction)
        unit[valid] = direction[valid] / norm[valid, None]
        x, z = position[:, 0], position[:, 2]
        dx, dz = unit[:, 0], unit[:, 2]
        a = dx * dx + dz * dz
        b = 2.0 * (x * dx + z * dz)
        c = x * x + z * z - phantom_radius_mm * phantom_radius_mm
        discriminant = b * b - 4.0 * a * c
        valid &= (a > 1.0e-12) & (discriminant >= 0.0)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        safe_denominator = np.where(a > 1.0e-12, 2.0 * a, 1.0)
        candidates = np.column_stack(
            ((-b - root) / safe_denominator, (-b + root) / safe_denominator)
        )
        candidates[candidates < 0.0] = np.inf
        distance = np.min(candidates, axis=1)
        valid &= np.isfinite(distance)
        return distance, valid

    entrance, entrance_valid = forward_distance(pairs[:, 0, :], pairs[:, 2, :])
    exit_distance, exit_valid = forward_distance(pairs[:, 1, :], -pairs[:, 3, :])
    hit = entrance_valid & exit_valid
    air_length = entrance + exit_distance
    direct = np.linalg.norm(
        pairs[:, 1, :].astype(np.float64) - pairs[:, 0, :].astype(np.float64),
        axis=1,
    )
    air_length[~hit] = direct[~hit]
    corrected = np.asarray(wepl_mm, dtype=np.float64) - (
        float(slope_mm_wepl_per_mm_air) * air_length
    )
    return np.maximum(corrected, 0.0).astype(np.float32)
