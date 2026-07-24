"""WEPL conversion using the exact wrapped PCT Bethe--Bloch LUT."""

from __future__ import annotations

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
