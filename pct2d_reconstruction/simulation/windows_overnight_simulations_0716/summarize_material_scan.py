#!/usr/bin/env python3
"""Summarize paired primary energies and water-LUT WEPL for material scans."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import uproot

from run_material_case import enumerate_cases, load_config


IONIZATION_POTENTIAL_MEV = 78.0e-6
ENERGY_BIN_MEV = 0.0001
ELECTRON_MASS_MEV = 0.51099895
PROTON_MASS_MEV = 938.27208816
CLASSICAL_ELECTRON_RADIUS_MM = 2.8179403262e-12
ELECTRON_DENSITY_PER_CM3 = 3.343e23


def make_lut(max_energy_mev: float = 600.0) -> np.ndarray:
    number = int(np.ceil(max_energy_mev / ENERGY_BIN_MEV))
    low = int(np.ceil(0.001 / ENERGY_BIN_MEV))
    indices = np.arange(low, number, dtype=np.float64)
    energy = indices * ENERGY_BIN_MEV
    beta2 = 1.0 - (PROTON_MASS_MEV / (energy + PROTON_MASS_MEV)) ** 2
    k = (4.0 * np.pi * CLASSICAL_ELECTRON_RADIUS_MM**2 * ELECTRON_MASS_MEV
         * ELECTRON_DENSITY_PER_CM3 / 1000.0)
    stopping = k * (np.log(2.0 * ELECTRON_MASS_MEV / IONIZATION_POTENTIAL_MEV
                           * beta2 / (1.0 - beta2)) - beta2) / beta2
    lut = np.zeros(number, dtype=np.float64)
    lut[low:] = np.cumsum(ENERGY_BIN_MEV / stopping, dtype=np.float64)
    return lut


def primary_energy(path: Path, tree_name: str) -> tuple[np.ndarray, np.ndarray]:
    tree = uproot.open(path)[tree_name]
    arrays = tree.arrays(["EventID", "TrackID", "KineticEnergy"], library="np")
    keep = arrays["TrackID"] == 1
    event = arrays["EventID"][keep].astype(np.int64)
    energy = arrays["KineticEnergy"][keep].astype(np.float64)
    order = np.argsort(event, kind="stable")
    event, energy = event[order], energy[order]
    unique, first = np.unique(event, return_index=True)
    return unique, energy[first]


def describe(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not values.size:
        return {f"{prefix}_{name}": float("nan") for name in
                ["mean", "std", "p01", "median", "p99"]}
    return {
        f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std()),
        f"{prefix}_p01": float(np.quantile(values, 0.01)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    cases = enumerate_cases(config)
    lut = make_lut()
    rows = []
    for case in cases:
        case_dir = args.data_dir / case["case_id"]
        metadata_path = args.qc_dir / "cases" / case["case_id"] / "case_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        in_event, in_energy = primary_energy(case_dir / "PhaseSpaceIn.root", "PhaseSpaceIn")
        out_event, out_energy = primary_energy(case_dir / "PhaseSpaceOut.root", "PhaseSpaceOut")
        common, in_index, out_index = np.intersect1d(in_event, out_event, return_indices=True)
        ein, eout = in_energy[in_index], out_energy[out_index]
        energy_loss = ein - eout
        ii = np.floor(ein / ENERGY_BIN_MEV + 0.5).astype(np.int64)
        oi = np.floor(eout / ENERGY_BIN_MEV + 0.5).astype(np.int64)
        wepl = lut[ii] - lut[oi]
        rows.append({
            **case, "requested_protons": int(metadata["protons"]),
            "entrance_primary": int(in_event.size), "exit_primary": int(out_event.size),
            "paired_primary": int(common.size),
            "primary_survival": float(common.size / in_event.size) if in_event.size else 0.0,
            **describe(energy_loss, "energy_loss_mev"), **describe(wepl, "wepl_mm"),
        })
        print(f"Summarized {case['case_id']}: paired={common.size:,}")
    args.qc_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.qc_dir / "material_scan_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "PASS" if len(rows) == len(cases) else "FAIL",
        "case_count": len(rows), "ionization_potential_water_ev": 78.0,
        "summary_csv": str(csv_path.resolve()),
    }
    (args.qc_dir / "material_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
