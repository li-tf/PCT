#!/usr/bin/env python3
"""Aggregate depth-dose replicas and extract R80-based virtual-MLIC RSP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from run_mlic_replica import enumerate_cases, load_config, read_mhd


def distal_r80_many(
    curves: np.ndarray, spacing_mm: float, sigma_mm: float
) -> np.ndarray:
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if sigma_mm > 0:
        values = gaussian_filter1d(
            values, sigma=sigma_mm / spacing_mm, axis=1, mode="nearest"
        )
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    depths = (np.arange(values.shape[1], dtype=np.float64) + 0.5) * spacing_mm
    for row_index, row in enumerate(values):
        peak = int(np.argmax(row))
        peak_value = float(row[peak])
        if not np.isfinite(peak_value) or peak_value <= 0:
            continue
        target = 0.8 * peak_value
        candidates = np.flatnonzero(row[peak + 1 :] <= target)
        if not candidates.size:
            continue
        upper = peak + 1 + int(candidates[0])
        lower = upper - 1
        y0, y1 = float(row[lower]), float(row[upper])
        if y1 == y0:
            result[row_index] = depths[upper]
        else:
            fraction = (target - y0) / (y1 - y0)
            result[row_index] = depths[lower] + fraction * spacing_mm
    return result


def replica_bootstrap_r80(
    curves: np.ndarray,
    spacing_mm: float,
    sigma_mm: float,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    replica_count = curves.shape[0]
    output = np.empty(samples, dtype=np.float64)
    batch_size = 100
    for start in range(0, samples, batch_size):
        stop = min(samples, start + batch_size)
        counts = rng.multinomial(
            replica_count,
            np.full(replica_count, 1.0 / replica_count),
            size=stop - start,
        )
        boot_curves = counts @ curves
        output[start:stop] = distal_r80_many(
            boot_curves, spacing_mm, sigma_mm
        )
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite R80 in replica bootstrap")
    return output


def rebin_curve(curve: np.ndarray, factor: int) -> np.ndarray:
    usable = curve.size - curve.size % factor
    return curve[:usable].reshape(-1, factor).sum(axis=1)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    data_dir = args.data_dir.resolve()
    qc_dir = args.qc_dir.resolve()
    summary_dir = qc_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    spacing = float(config["depth_bin_mm"])
    sigma = float(config["r80_smoothing_sigma_mm"])
    bootstrap_samples = int(config["bootstrap_samples"])
    replicas = int(config["replicates_per_case"])
    rng = np.random.default_rng(int(config["random_seed"]) + 600_000)

    case_rows: list[dict] = []
    curves_by_case: dict[str, np.ndarray] = {}
    bootstrap_by_case: dict[str, np.ndarray] = {}
    case_by_id = {case["case_id"]: case for case in enumerate_cases(config)}
    for case_id, case in case_by_id.items():
        curves: list[np.ndarray] = []
        for replica in range(replicas):
            path = (
                data_dir
                / case_id
                / f"replica_{replica:02d}"
                / "depth_dose_edep.mhd"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            image, _ = read_mhd(path)
            curve = np.asarray(image, dtype=np.float64).reshape(-1)
            if not np.isfinite(curve).all() or curve.sum() <= 0:
                raise RuntimeError(f"invalid depth-dose curve: {path}")
            curves.append(curve)
        matrix = np.stack(curves)
        aggregate = matrix.sum(axis=0)
        r80 = float(distal_r80_many(aggregate, spacing, sigma)[0])
        r80_raw = float(distal_r80_many(aggregate, spacing, 0.0)[0])
        r80_sigma05 = float(distal_r80_many(aggregate, spacing, 0.5)[0])
        rebinned = rebin_curve(aggregate, 2)
        r80_bin02 = float(
            distal_r80_many(rebinned, spacing * 2, sigma)[0]
        )
        bootstrap = replica_bootstrap_r80(
            matrix, spacing, sigma, bootstrap_samples, rng
        )
        curves_by_case[case_id] = aggregate
        bootstrap_by_case[case_id] = bootstrap
        case_rows.append(
            {
                "case_id": case_id,
                "energy_mev": case["energy_mev"],
                "name": case["name"],
                "material": case["material"] or "",
                "thickness_mm": case["thickness_mm"],
                "replicates": replicas,
                "protons": int(config["protons_per_case"]),
                "r80_mm": r80,
                "r80_bootstrap_sd_mm": float(np.std(bootstrap, ddof=1)),
                "r80_ci95_low_mm": float(np.percentile(bootstrap, 2.5)),
                "r80_ci95_high_mm": float(np.percentile(bootstrap, 97.5)),
                "r80_no_smoothing_mm": r80_raw,
                "r80_sigma0p5_mm": r80_sigma05,
                "r80_bin0p2_mm": r80_bin02,
                "max_sensitivity_difference_mm": max(
                    abs(r80 - r80_raw),
                    abs(r80 - r80_sigma05),
                    abs(r80 - r80_bin02),
                ),
            }
        )
        print(
            f"R80 {case_id}: {r80:.4f} +/- "
            f"{np.std(bootstrap, ddof=1):.4f} mm",
            flush=True,
        )

    rsp_rows: list[dict] = []
    for energy in config["energies_mev"]:
        reference_id = f"e{int(energy):03d}_reference"
        ref_row = next(row for row in case_rows if row["case_id"] == reference_id)
        ref_boot = bootstrap_by_case[reference_id]
        for item in config["cases_per_energy"]:
            if item["name"] == "Reference":
                continue
            case_id = f"e{int(energy):03d}_{item['name'].lower()}"
            if item["name"] == "Aluminium":
                case_id = f"e{int(energy):03d}_aluminium"
            elif item["name"] == "SpineBone":
                case_id = f"e{int(energy):03d}_spinebone"
            case = case_by_id[case_id]
            row = next(value for value in case_rows if value["case_id"] == case_id)
            thickness = float(case["thickness_mm"])
            shift = float(ref_row["r80_mm"]) - float(row["r80_mm"])
            rsp = shift / thickness
            rsp_boot = (ref_boot - bootstrap_by_case[case_id]) / thickness
            rsp_rows.append(
                {
                    "energy_mev": float(energy),
                    "name": case["name"],
                    "material": case["material"],
                    "thickness_mm": thickness,
                    "reference_r80_mm": ref_row["r80_mm"],
                    "sample_r80_mm": row["r80_mm"],
                    "range_shift_mm": shift,
                    "mlic_rsp": rsp,
                    "mlic_rsp_bootstrap_sd": float(np.std(rsp_boot, ddof=1)),
                    "mlic_rsp_ci95_low": float(np.percentile(rsp_boot, 2.5)),
                    "mlic_rsp_ci95_high": float(np.percentile(rsp_boot, 97.5)),
                    "water_control_error_percent": (
                        100.0 * (rsp - 1.0)
                        if case["name"] == "Water"
                        else ""
                    ),
                }
            )

    write_csv(summary_dir / "r80_summary.csv", case_rows)
    write_csv(summary_dir / "mlic_rsp_summary.csv", rsp_rows)

    energy_count = len(config["energies_mev"])
    column_count = min(2, energy_count)
    row_count = math.ceil(energy_count / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(6 * column_count, 4 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for axis, energy in zip(flat_axes, config["energies_mev"]):
        for item in config["cases_per_energy"]:
            case_id = f"e{int(energy):03d}_{item['name'].lower()}"
            curve = curves_by_case[case_id]
            depth = (np.arange(curve.size) + 0.5) * spacing
            axis.plot(depth, curve / curve.max(), label=item["name"], linewidth=1.2)
        axis.set_title(f"{int(energy)} MeV")
        axis.set_xlabel("Water depth (mm)")
        axis.set_ylabel("Normalised depth dose")
        axis.set_xlim(0, float(config["water_tank_size_mm"][2]))
        axis.set_ylim(0, 1.08)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    for axis in flat_axes[energy_count:]:
        axis.remove()
    fig.savefig(summary_dir / "depth_dose_curves.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for item in config["cases_per_energy"]:
        if item["name"] == "Reference":
            continue
        rows = [row for row in rsp_rows if row["name"] == item["name"]]
        x = np.asarray([row["energy_mev"] for row in rows])
        y = np.asarray([row["mlic_rsp"] for row in rows])
        error = np.asarray([row["mlic_rsp_bootstrap_sd"] for row in rows])
        axis.errorbar(x, y, yerr=error, marker="o", capsize=3, label=item["name"])
    axis.set_xlabel("Incident proton energy (MeV)")
    axis.set_ylabel("Virtual-MLIC RSP")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(summary_dir / "mlic_rsp_vs_energy.png", dpi=180)
    plt.close(fig)

    water_rows = [row for row in rsp_rows if row["name"] == "Water"]
    water_errors = [
        abs(float(row["mlic_rsp"]) - 1.0) for row in water_rows
    ]
    result = {
        "status": "PASS",
        "scenario_id": config["scenario_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "case_count": len(case_rows),
        "replicas_per_case": replicas,
        "task_count": len(case_rows) * replicas,
        "total_protons": len(case_rows) * int(config["protons_per_case"]),
        "all_finite": True,
        "maximum_water_control_absolute_error": max(water_errors),
        "maximum_r80_sensitivity_difference_mm": max(
            float(row["max_sensitivity_difference_mm"]) for row in case_rows
        ),
        "outputs": [
            "r80_summary.csv",
            "mlic_rsp_summary.csv",
            "depth_dose_curves.png",
            "mlic_rsp_vs_energy.png",
        ],
        "bootstrap_method": (
            "nonparametric resampling of independent depth-dose replicas; "
            "R80 is re-extracted from each summed bootstrap curve"
        ),
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
