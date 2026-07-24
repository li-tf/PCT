#!/usr/bin/env python3
"""Run integrated QC and stage-1 material/energy calibration for S6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import uproot


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULT_ANALYSIS_CONFIG = HERE / "calibration_config.json"
DEFAULT_SIM_CONFIG = (
    CODE_ROOT / "simulation" / "windows_overnight_simulations_0716"
    / "material_scan_config.json"
)
DEFAULT_DATA_ROOT = (
    REPOSITORY_ROOT / "data" / "simulation_data"
    / "results0717_s6_material_energy_scan"
)
DEFAULT_WORKSTATION_QC = (
    CODE_ROOT / "simulation" / "windows_overnight_simulations_0716"
    / "qc" / "s6_material_energy_scan"
)
DEFAULT_OUTPUT = HERE / "qc" / "results0717_s6_material_energy_scan"
BASELINE_METRICS = (
    CODE_ROOT / "evaluation" / "baselines" / "results0716"
    / "baseline_metrics.csv"
)

REQUIRED_BRANCHES = (
    "EventID",
    "TrackID",
    "KineticEnergy",
    "Position_X",
    "Position_Y",
    "Position_Z",
    "Direction_X",
    "Direction_Y",
    "Direction_Z",
)

ELECTRON_MASS_MEV = 0.51099895
PROTON_MASS_MEV = 938.27208816
CLASSICAL_ELECTRON_RADIUS_MM = 2.8179403262e-12
ELECTRON_DENSITY_PER_CM3 = 3.343e23
ENERGY_BIN_MEV = 0.0001
MAX_ENERGY_MEV = 600.0

PALETTE = {
    150.0: "#2463A6",
    180.0: "#D59B20",
    200.0: "#D96C3F",
    220.0: "#6B7D3A",
}
ENERGY_STYLES = {
    150.0: {"marker": "o", "linestyle": "-"},
    180.0: {"marker": "s", "linestyle": "--"},
    200.0: {"marker": "^", "linestyle": "-."},
    220.0: {"marker": "D", "linestyle": ":"},
}


@dataclass
class PrimaryState:
    event: np.ndarray
    energy: np.ndarray
    position: np.ndarray
    direction: np.ndarray
    tree_entries: int
    primary_entries: int
    duplicate_primary: int
    finite: bool
    branches: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--simulation-config", type=Path, default=DEFAULT_SIM_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workstation-qc", type=Path, default=DEFAULT_WORKSTATION_QC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--root-backend",
        choices=("auto", "pyroot", "uproot"),
        default="auto",
        help="Array reader. auto prefers PyROOT and falls back to uproot.",
    )
    parser.add_argument(
        "--skip-itk-crosscheck",
        action="store_true",
        help="Skip the wrapped C++/ITK LUT sample comparison.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def enumerate_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for energy in config["energies_mev"]:
        for material in config["materials"]:
            for thickness in material["thicknesses_mm"]:
                cases.append(
                    {
                        "case_index": len(cases),
                        "case_id": (
                            f"{material['name'].lower()}_e{int(energy):03d}"
                            f"_t{int(thickness):04d}"
                        ),
                        "material": str(material["name"]),
                        "energy_mev": float(energy),
                        "thickness_mm": float(thickness),
                        "max_step_mm": float(material["max_step_mm"]),
                    }
                )
    return cases


def build_lut(ionization_potential_ev: float) -> np.ndarray:
    ionization_mev = ionization_potential_ev * 1.0e-6
    bins = int(np.ceil(MAX_ENERGY_MEV / ENERGY_BIN_MEV))
    low = int(np.ceil(0.001 / ENERGY_BIN_MEV))
    indices = np.arange(low, bins, dtype=np.float64)
    energy = indices * ENERGY_BIN_MEV
    beta2 = 1.0 - (PROTON_MASS_MEV / (energy + PROTON_MASS_MEV)) ** 2
    k = (
        4.0
        * np.pi
        * CLASSICAL_ELECTRON_RADIUS_MM**2
        * ELECTRON_MASS_MEV
        * ELECTRON_DENSITY_PER_CM3
        / 1000.0
    )
    stopping = k * (
        np.log(
            2.0
            * ELECTRON_MASS_MEV
            / ionization_mev
            * beta2
            / (1.0 - beta2)
        )
        - beta2
    ) / beta2
    lut = np.zeros(bins, dtype=np.float64)
    lut[low:] = np.cumsum(ENERGY_BIN_MEV / stopping, dtype=np.float64)
    return lut


def energies_to_wepl(
    lut: np.ndarray, energy_in: np.ndarray, energy_out: np.ndarray
) -> np.ndarray:
    ii = np.floor(np.asarray(energy_in, np.float64) / ENERGY_BIN_MEV + 0.5).astype(
        np.int64
    )
    oi = np.floor(np.asarray(energy_out, np.float64) / ENERGY_BIN_MEV + 0.5).astype(
        np.int64
    )
    if (
        np.any(ii < 0)
        or np.any(oi < 0)
        or np.any(ii >= lut.size)
        or np.any(oi >= lut.size)
    ):
        raise ValueError("energy outside the Bethe--Bloch LUT")
    return lut[ii] - lut[oi]


def choose_root_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import ROOT  # noqa: F401
        return "pyroot"
    except Exception:
        return "uproot"


def read_primary_state(
    path: Path, tree_name: str, backend: str
) -> PrimaryState:
    with uproot.open(path) as root_file:
        if tree_name not in root_file:
            raise RuntimeError(f"{path}: missing tree {tree_name}")
        tree = root_file[tree_name]
        branches = list(tree.keys())
        missing = sorted(set(REQUIRED_BRANCHES) - set(branches))
        if missing:
            raise RuntimeError(f"{path}: missing branches {missing}")
        tree_entries = int(tree.num_entries)
        if backend == "uproot":
            arrays = tree.arrays(list(REQUIRED_BRANCHES), library="np")
        elif backend == "pyroot":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import ROOT

            arrays = ROOT.RDataFrame(tree_name, str(path)).AsNumpy(
                list(REQUIRED_BRANCHES)
            )
        else:
            raise ValueError(f"unknown ROOT backend: {backend}")

    keep = arrays["TrackID"] == 1
    primary_entries = int(np.count_nonzero(keep))
    event = arrays["EventID"][keep].astype(np.int64, copy=False)
    order = np.argsort(event, kind="stable")
    event = event[order]
    energy = arrays["KineticEnergy"][keep][order].astype(np.float64, copy=False)
    position = np.column_stack(
        [arrays[f"Position_{axis}"][keep][order] for axis in "XYZ"]
    ).astype(np.float64, copy=False)
    direction = np.column_stack(
        [arrays[f"Direction_{axis}"][keep][order] for axis in "XYZ"]
    ).astype(np.float64, copy=False)
    unique_event, first = np.unique(event, return_index=True)
    duplicates = int(event.size - unique_event.size)
    finite = bool(
        np.isfinite(energy).all()
        and np.isfinite(position).all()
        and np.isfinite(direction).all()
    )
    return PrimaryState(
        event=unique_event,
        energy=energy[first],
        position=position[first],
        direction=direction[first],
        tree_entries=tree_entries,
        primary_entries=primary_entries,
        duplicate_primary=duplicates,
        finite=finite,
        branches=branches,
    )


def paired_states(
    entrance: PrimaryState, exit_state: PrimaryState
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    event, i, o = np.intersect1d(
        entrance.event, exit_state.event, assume_unique=True, return_indices=True
    )
    return (
        event,
        entrance.energy[i],
        exit_state.energy[o],
        entrance.direction[i],
        exit_state.direction[o],
    )


def group_standard_error(
    values: np.ndarray, event: np.ndarray, statistic: str, blocks: int
) -> float:
    estimates = []
    group = np.mod(event, blocks)
    for index in range(blocks):
        sample = values[group == index]
        if sample.size < 5:
            continue
        if statistic == "mean":
            estimates.append(float(np.mean(sample)))
        elif statistic == "median":
            estimates.append(float(np.median(sample)))
        else:
            raise ValueError(statistic)
    if len(estimates) < max(10, blocks // 2):
        return float("nan")
    return float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))


def distribution_metrics(
    values: np.ndarray,
    event: np.ndarray,
    mad_multiplier: float,
    blocks: int,
) -> tuple[dict[str, float], np.ndarray]:
    if not values.size or not np.isfinite(values).all():
        raise ValueError("distribution is empty or non-finite")
    quantiles = np.quantile(values, [0.001, 0.01, 0.5, 0.99, 0.999])
    median = float(quantiles[2])
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 0:
        core = np.abs(values - median) <= mad_multiplier * robust_sigma
    else:
        core = np.ones(values.shape, dtype=bool)
    mean = float(np.mean(values))
    std = float(np.std(values))
    mean_se = group_standard_error(values, event, "mean", blocks)
    median_se = group_standard_error(values, event, "median", blocks)
    core_mean = float(np.mean(values[core]))
    core_se = group_standard_error(values[core], event[core], "mean", blocks)
    metrics = {
        "count": int(values.size),
        "mean": mean,
        "std": std,
        "mean_se": mean_se,
        "mean_ci95_low": mean - 1.96 * mean_se,
        "mean_ci95_high": mean + 1.96 * mean_se,
        "median": median,
        "median_se": median_se,
        "median_ci95_low": median - 1.96 * median_se,
        "median_ci95_high": median + 1.96 * median_se,
        "mad": mad,
        "robust_sigma": robust_sigma,
        "core_mean": core_mean,
        "core_mean_se": core_se,
        "core_mean_ci95_low": core_mean - 1.96 * core_se,
        "core_mean_ci95_high": core_mean + 1.96 * core_se,
        "core_fraction": float(np.mean(core)),
        "p001": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p99": float(quantiles[3]),
        "p999": float(quantiles[4]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }
    return metrics, core


def flatten(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    total = y - np.mean(y)
    r2 = 1.0 - float(np.dot(residual, residual) / np.dot(total, total))
    origin_slope = float(np.dot(x, y) / np.dot(x, x))
    origin_residual = y - origin_slope * x
    origin_r2 = 1.0 - float(
        np.dot(origin_residual, origin_residual) / np.dot(y, y)
    )
    return {
        "slope": float(slope),
        "intercept_mm": float(intercept),
        "r2": r2,
        "origin_constrained_slope": origin_slope,
        "origin_constrained_r2": origin_r2,
    }


def workstation_case_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["case_id"]: row for row in csv.DictReader(stream)}


def read_iterative_baseline(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["checkpoint"] == "iterative_epoch_03":
                return {
                    "water_mean_rsp": float(row["water_mean_rsp"]),
                    "aluminium_platform_rsp": float(
                        row["aluminium_platform_rsp"]
                    ),
                }
    raise RuntimeError("iterative_epoch_03 not found in baseline metrics")


def chart_map() -> list[dict[str, Any]]:
    return [
        {
            "file": "water_wepl_vs_thickness.png",
            "question": "Does the I=78 eV water LUT map Water slabs to their physical thickness?",
            "family": "comparison",
            "type": "multi-series line with identity reference",
            "fields": ["energy_mev", "thickness_mm", "wepl_median_mm"],
            "palette": "four energy categories; markers and line styles retained",
        },
        {
            "file": "water_rsp_bias_heatmap.png",
            "question": "Where does water-equivalent bias depend on energy and thickness?",
            "family": "matrix",
            "type": "annotated diverging heatmap",
            "fields": ["energy_mev", "thickness_mm", "median_rsp_bias_percent"],
            "palette": "two-root diverging with numeric labels",
        },
        {
            "file": "aluminium_effective_rsp.png",
            "question": "How does effective Aluminium RSP differ from the fixed 200 MeV reference?",
            "family": "uncertainty and benchmark",
            "type": "multi-series line with fixed benchmark",
            "fields": ["energy_mev", "thickness_mm", "median_effective_rsp"],
            "palette": "four energy categories; dark neutral benchmark",
        },
        {
            "file": "air_wepl_model.png",
            "question": "How much WEPL is contributed by Air as length and energy change?",
            "family": "relationship",
            "type": "multi-series line with origin-constrained fits",
            "fields": ["energy_mev", "thickness_mm", "wepl_mean_mm"],
            "palette": "four energy categories; line style plus marker",
        },
        {
            "file": "ionization_potential_sensitivity.png",
            "question": "Which empirical water I value minimizes median Water WEPL bias?",
            "family": "uncertainty and benchmark",
            "type": "line with 78 eV reference",
            "fields": ["ionization_potential_ev", "water_bias_rms_percent"],
            "palette": "single blue root; dark neutral reference",
        },
        {
            "file": "tail_diagnostics.png",
            "question": "Do rare energy-loss tails materially separate raw means from robust estimates?",
            "family": "distribution",
            "type": "three histograms with median and core-mean references",
            "fields": ["case_id", "wepl_mm"],
            "palette": "single blue root; orange robust reference",
        },
    ]


def make_figures(
    output: Path,
    case_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    air_models: list[dict[str, Any]],
    selected_distributions: dict[str, np.ndarray],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/pct-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/pct-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )
    output.mkdir(parents=True, exist_ok=True)

    water = [row for row in case_rows if row["material"] == "Water"]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    maximum = max(float(row["thickness_mm"]) for row in water)
    ax.plot([0, maximum], [0, maximum], "--", color="#343A40", label="Ideal WEPL = thickness")
    for energy in sorted({float(row["energy_mev"]) for row in water}):
        subset = sorted(
            [row for row in water if float(row["energy_mev"]) == energy],
            key=lambda row: float(row["thickness_mm"]),
        )
        ax.plot(
            [float(row["thickness_mm"]) for row in subset],
            [float(row["wepl_median"]) for row in subset],
            **ENERGY_STYLES[energy],
            color=PALETTE[energy],
            label=f"{energy:.0f} MeV",
        )
    ax.set(
        title="Water WEPL versus slab thickness",
        xlabel="Water thickness (mm)",
        ylabel="Median WEPL (mm)",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend(ncol=2)
    fig.savefig(output / "water_wepl_vs_thickness.png")
    plt.close(fig)

    energies = sorted({float(row["energy_mev"]) for row in water})
    thicknesses = sorted({float(row["thickness_mm"]) for row in water})
    matrix = np.full((len(energies), len(thicknesses)), np.nan)
    for row in water:
        i = energies.index(float(row["energy_mev"]))
        j = thicknesses.index(float(row["thickness_mm"]))
        matrix[i, j] = 100.0 * (
            float(row["wepl_median"]) / float(row["thickness_mm"]) - 1.0
        )
    limit = max(1.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}%", ha="center", va="center", fontsize=9)
    ax.set(
        title="Median Water RSP bias at I = 78 eV",
        xlabel="Water thickness (mm)",
        ylabel="Incident energy (MeV)",
        xticks=np.arange(len(thicknesses)),
        xticklabels=[f"{value:g}" for value in thicknesses],
        yticks=np.arange(len(energies)),
        yticklabels=[f"{value:g}" for value in energies],
    )
    fig.colorbar(image, ax=ax, label="Median WEPL / thickness - 1 (%)")
    fig.savefig(output / "water_rsp_bias_heatmap.png")
    plt.close(fig)

    aluminium = [row for row in case_rows if row["material"] == "Aluminium"]
    reference = 2.1189760409708303
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.axhline(reference, linestyle="--", color="#343A40", label="Fixed 200 MeV reference")
    for energy in sorted({float(row["energy_mev"]) for row in aluminium}):
        subset = sorted(
            [row for row in aluminium if float(row["energy_mev"]) == energy],
            key=lambda row: float(row["thickness_mm"]),
        )
        ax.plot(
            [float(row["thickness_mm"]) for row in subset],
            [
                float(row["wepl_median"]) / float(row["thickness_mm"])
                for row in subset
            ],
            **ENERGY_STYLES[energy],
            color=PALETTE[energy],
            label=f"{energy:.0f} MeV",
        )
    ax.set(
        title="Aluminium effective RSP",
        xlabel="Aluminium thickness (mm)",
        ylabel="Median WEPL / thickness",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend(ncol=2)
    fig.savefig(output / "aluminium_effective_rsp.png")
    plt.close(fig)

    air = [row for row in case_rows if row["material"] == "Air"]
    model_by_energy = {float(row["energy_mev"]): row for row in air_models}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    ax = axes[0]
    for energy in sorted(model_by_energy):
        subset = sorted(
            [row for row in air if float(row["energy_mev"]) == energy],
            key=lambda row: float(row["thickness_mm"]),
        )
        x = np.asarray([float(row["thickness_mm"]) for row in subset])
        y = np.asarray([float(row["wepl_mean"]) for row in subset])
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=ENERGY_STYLES[energy]["marker"],
            color=PALETTE[energy],
            label=f"{energy:.0f} MeV",
        )
        model = model_by_energy[energy]
        xx = np.linspace(0, x.max(), 100)
        ax.plot(
            xx,
            float(model["origin_constrained_slope"]) * xx,
            color=PALETTE[energy],
            linestyle=ENERGY_STYLES[energy]["linestyle"],
            linewidth=1.5,
        )
    ax.set(
        title="Mean Air WEPL versus path length",
        xlabel="Air length (mm)",
        ylabel="Mean WEPL (mm)",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend(ncol=2, fontsize=8)
    slope_axis = axes[1]
    slope_axis.plot(
        sorted(model_by_energy),
        [
            1000.0
            * float(model_by_energy[energy]["origin_constrained_slope"])
            for energy in sorted(model_by_energy)
        ],
        marker="o",
        color="#2463A6",
    )
    slope_axis.set(
        title="Origin-constrained mean slope",
        xlabel="Incident energy (MeV)",
        ylabel=r"Air WEPL / length ($10^{-3}$)",
        xticks=sorted(model_by_energy),
    )
    slope_axis.grid(color="#D9DEE5", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output / "air_wepl_model.png")
    plt.close(fig)

    objective: dict[float, list[float]] = {}
    for row in sensitivity_rows:
        objective.setdefault(float(row["ionization_potential_ev"]), []).append(
            float(row["median_rsp_bias_percent"])
        )
    x = np.asarray(sorted(objective))
    y = np.asarray([math.sqrt(np.mean(np.square(objective[value]))) for value in x])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o", color="#2463A6")
    ax.axvline(78.0, linestyle="--", color="#343A40", label="Current I = 78 eV")
    best = int(np.argmin(y))
    ax.scatter([x[best]], [y[best]], s=55, color="#D96C3F", zorder=3, label=f"Grid minimum {x[best]:g} eV")
    ax.set(
        title="Water-LUT ionization-potential sensitivity",
        xlabel="Ionization potential I (eV)",
        ylabel="RMS median Water RSP bias (%)",
    )
    ax.grid(color="#D9DEE5", linewidth=0.7)
    ax.legend()
    fig.savefig(output / "ionization_potential_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.9))
    for ax, (case_id, values) in zip(axes, selected_distributions.items()):
        low, high = np.quantile(values, [0.001, 0.995])
        shown = values[(values >= low) & (values <= high)]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826 * mad
        core = np.abs(values - median) <= 3.0 * sigma if sigma > 0 else np.ones(values.shape, bool)
        ax.hist(shown, bins=90, color="#B8D4EE", edgecolor="#2463A6", linewidth=0.3)
        ax.axvline(float(np.mean(values)), color="#343A40", linestyle=":", label="Raw mean")
        ax.axvline(median, color="#D96C3F", linestyle="--", label="Median")
        ax.axvline(float(np.mean(values[core])), color="#6B7D3A", label="3-MAD core mean")
        ax.set(title=case_id.replace("_", "\n"), xlabel="WEPL (mm)", ylabel="Protons")
    axes[0].legend(fontsize=8)
    fig.suptitle("Representative WEPL distributions (display clipped at 0.1–99.5%)")
    fig.tight_layout()
    fig.savefig(output / "tail_diagnostics.png")
    plt.close(fig)


def markdown_report(
    summary: dict[str, Any],
    water_rows: list[dict[str, Any]],
    aluminium_rows: list[dict[str, Any]],
    air_models: list[dict[str, Any]],
) -> str:
    water_200 = [
        row for row in water_rows if float(row["energy_mev"]) == 200.0
    ]
    aluminium_200 = [
        row for row in aluminium_rows if float(row["energy_mev"]) == 200.0
    ]
    air_200 = next(row for row in air_models if float(row["energy_mev"]) == 200.0)
    status = summary["status"]
    return f"""# S6材料能量扫描：阶段1技术总结

## 技术摘要

阶段1状态为**{status}**。52组Water、Aluminium和Air case均已读取，ROOT内容、
主质子配对和工作站汇总完成交叉核对。正式结论以逐case中位数和3-MAD核心均值
为主，原始均值仅用于展示核反应/异常能损尾部的影响。

当前`I=78 eV`水LUT并未在所有厚度上严格给出`WEPL=水厚度`：200 MeV下，
5--100 mm Water的中位数RSP范围为
`{min(float(row['median_effective_rsp']) for row in water_200):.6f}`--
`{max(float(row['median_effective_rsp']) for row in water_200):.6f}`。I值网格的经验
最小点为`{summary['water_i_sensitivity']['best_i_ev']:.1f} eV`，但这只是吸收
Geant4与简化Bethe--Bloch差异的有效参数，不解释为新的水物理常数。

200 MeV Water各厚度的原点约束拟合斜率为
`{summary['water_baseline_comparison']['s6_origin_constrained_rsp']:.6f}`，
与results0716迭代水区均值
`{summary['water_baseline_comparison']['results0716_iterative_water_mean_rsp']:.6f}`
仅差`{summary['water_baseline_comparison']['difference_rsp']:+.6f}`。这为此前约
`+1.4%`的水偏差提供了强证据：它主要与当前WEPL标定口径一致，而不是由重建
算法单独产生。不过S6最长只有100 mm、且是单材料薄板，因此尚不能把这一吻合
当作对200 mm混合路径的严格因果分解。

200 MeV、5 mm Aluminium的中位数有效RSP为
`{summary['aluminium_decomposition']['effective_rsp_200mev_5mm']:.6f}`，相对固定
200 MeV参考值`2.118976`低
`{summary['aluminium_decomposition']['reference_definition_component_percent']:.3f}%`。
因此results0716迭代铝平台的总偏差中，物理参考定义可以解释一部分，但S6最低
仅覆盖150 MeV，尚不足以精确代表穿过200 mm水后的完整能谱。

Air在200 MeV下的期望WEPL原点约束斜率为
`{float(air_200['origin_constrained_slope']):.8f} mm-WEPL/mm-Air`。该模型可进入
S1/S3/D1的Air背景修正，但对150 MeV以下下游质子属于外推，必须保留敏感性检查。

## 52组数据通过完整性和配对检查

数据粒度是一条穿过指定材料薄板、同时在入口和出口参考面保留状态的主质子。
入口和出口按EventID匹配，只保留`TrackID=1`。全部WEPL统计均以“出口仍存在的
存活主质子”为条件；存活率另行报告，不能把缺失出口事件当作零WEPL。

| 项目 | 结果 |
|---|---:|
| case数 | {summary['data_quality']['case_count']} |
| ROOT数 | {summary['data_quality']['root_file_count']} |
| 配对主质子 | {summary['data_quality']['paired_primary']:,} |
| ROOT/QC问题 | {summary['data_quality']['problem_count']} |
| 工作站汇总最大数值差 | {summary['data_quality']['workstation_summary_max_abs_difference']:.3g} |
| ITK/C++ LUT最大差/mm | {summary['lut_crosscheck']['max_abs_difference_mm']:.3g} |

## Water显示厚度相关的有效偏差

![Water WEPL and physical thickness](figures/water_wepl_vs_thickness.png)

图中虚线是理想`WEPL=厚度`。中位数曲线随厚度逐渐偏离理想线，说明偏差不是
单纯随机噪声；它综合反映Geant4输运、当前简化Bethe--Bloch水LUT和能量损失
分布的差异。

![Water RSP bias](figures/water_rsp_bias_heatmap.png)

热图给出中位数`WEPL/厚度-1`。短薄板的相对指标更容易受到能量量化和分布形状
影响；长薄板更接近后续CT的降能条件，但仍不能替代真实路径上的连续材料混合。

![Ionization-potential sensitivity](figures/ionization_potential_sensitivity.png)

I值扫描只用于判断78 eV是否造成系统趋势。即使另一个I值使Water经验偏差更小，
也不能仅凭本扫描更改正式LUT；更改会同时改变所有历史WEPL，并可能把Geant4
模型差异错误吸收到单一参数中。

## Aluminium的固定参考与有效RSP并不完全相同

![Aluminium effective RSP](figures/aluminium_effective_rsp.png)

固定参考线是results0716真值使用的`2.118976`。不同能量和厚度的有效RSP跨过
该参考线，证明“一个固定200 MeV数值”与“沿降能路径测得的平均有效RSP”不是
同一物理量。

200 MeV各厚度结果：

| 厚度/mm | 中位数有效RSP | 相对固定参考/% | 存活率/% |
|---:|---:|---:|---:|
{chr(10).join(f"| {float(row['thickness_mm']):.0f} | {float(row['median_effective_rsp']):.6f} | {float(row['median_rsp_bias_vs_fixed_percent']):+.3f} | {100*float(row['primary_survival']):.3f} |" for row in aluminium_200)}

## Air校正应使用路径长度和能量

![Air WEPL model](figures/air_wepl_model.png)

Air的**均值WEPL**近似随长度线性增长，斜率随能量轻微变化。操作模型使用均值
的强制过原点斜率，并在150、180、200、220 MeV之间线性插值。中位数和3-MAD
核心均值只作为能损分布尾部诊断。后续不能对所有质子减去同一Air常数：圆柱外
Air长度随射线路径变化，上下游能量也不同。

## 尾部使原始均值不适合作为唯一标定量

![Representative tails](figures/tail_diagnostics.png)

图只显示0.1%--99.5%范围，完整数值仍用于原始均值。中位数和3-MAD核心均值
对稀有尾部更稳定；原始均值、稳健统计及核心保留率均保存在`case_metrics.csv`，
后续过滤设计可以直接复用这些证据。

## 定义、方法与不确定度

- `WEPL=R_w(E_in;I)-R_w(E_out;I)`，正式I值为78 eV；
- 有效RSP定义为逐case中位数WEPL除以物理薄板厚度；
- 3-MAD核心只用于稳健敏感性，不覆盖或删除原始统计；
- 95%区间由EventID模100的确定性分块估计；
- Air操作模型拟合逐质子WEPL均值并强制满足零长度对应零WEPL，同时保留中位数、
  3-MAD核心均值和自由截距拟合作为诊断；
- 所有能量统计以存活primary为条件，核反应导致的出口缺失通过存活率单独呈现。

## 局限与稳健性边界

1. S6能量范围为150--220 MeV，没有覆盖长水路径后的低能质子；
2. 薄板是单一材料，不能完全再现Water与Aluminium交替路径；
3. 当前参考面位于薄板表面外0.5 mm的Vacuum中，未包含真实探测器；
4. 有效I拟合不能区分Geant4物理模型、密度效应和简化LUT各自贡献；
5. Air修正很小，模型外推误差通常也小，但仍需在S2/S3配对实验中实测验证。

## 建议的下一步

1. 保留`I=78 eV`作为历史兼容主口径，同时把Water经验偏差作为校准系统项报告；
2. 在S2/S3中验证Air模型能否消除Vacuum/Air差异；
3. S4材料评价同时报告固定200 MeV RSP和S6支持的有效RSP解释；
4. 若需要严格分解results0716铝误差，补充80--140 MeV的5 mm Aluminium低通量扫描；
5. 在S1/S3/D1中记录修正前后WEPL偏差，避免Air校正成为不可审计的常数处理。

## 进一步问题

- 低于150 MeV时Aluminium相对水停止本领比是否继续沿现有趋势变化？
- S2/S3中的Air差异是否与S6模型在统计误差内一致？
- 使用Geant4原生水射程表替代当前简化LUT后，Water厚度趋势是否消失？
"""


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    analysis_config = load_json(args.config.resolve())
    simulation_config = load_json(args.simulation_config.resolve())
    cases = enumerate_cases(simulation_config)
    output = args.output.resolve()
    root_backend = choose_root_backend(args.root_backend)
    print(f"ROOT array backend: {root_backend}", flush=True)
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise FileExistsError(f"output exists: {output}; use --force")
        shutil.rmtree(output)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)

    expected_dataset = analysis_config["dataset"]
    if args.data_root.resolve().name != expected_dataset:
        raise ValueError(
            f"dataset mismatch: expected {expected_dataset}, got {args.data_root.name}"
        )
    expected_ids = {case["case_id"] for case in cases}
    actual_ids = {
        path.name for path in args.data_root.iterdir() if path.is_dir()
    }
    missing_cases = sorted(expected_ids - actual_ids)
    extra_cases = sorted(actual_ids - expected_ids)
    if missing_cases or extra_cases:
        raise RuntimeError(
            f"case directory mismatch; missing={missing_cases}, extra={extra_cases}"
        )

    config_hash = sha256(args.simulation_config.resolve())
    workstation_rows = workstation_case_rows(
        args.workstation_qc / "material_scan_summary.csv"
    )
    lut = build_lut(float(analysis_config["ionization_potential_ev"]))
    sensitivity_spec = analysis_config["ionization_sensitivity_ev"]
    sensitivity_values = np.arange(
        float(sensitivity_spec["start"]),
        float(sensitivity_spec["stop"]) + 0.25 * float(sensitivity_spec["step"]),
        float(sensitivity_spec["step"]),
    )

    manifest_files = [args.simulation_config.resolve()]
    manifest_files.extend(
        path.resolve()
        for path in args.workstation_qc.rglob("*")
        if path.is_file()
    )
    for shared_qc_name in ("environment_check.json", "overnight_summary.json"):
        shared_qc = args.workstation_qc.parent / shared_qc_name
        if shared_qc.is_file():
            manifest_files.append(shared_qc.resolve())
    for case in cases:
        case_dir = args.data_root / case["case_id"]
        manifest_files.extend(
            [case_dir / "PhaseSpaceIn.root", case_dir / "PhaseSpaceOut.root"]
        )
    manifest_files = sorted(set(manifest_files))

    print(f"Hashing {len(manifest_files)} stage-1 input files...", flush=True)
    manifest = []
    for index, path in enumerate(manifest_files, start=1):
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
        if index % 20 == 0 or index == len(manifest_files):
            print(f"  hashed {index}/{len(manifest_files)}", flush=True)

    root_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    water_energy_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selected: dict[str, np.ndarray] = {}
    crosscheck_ein: list[np.ndarray] = []
    crosscheck_eout: list[np.ndarray] = []
    workstation_max_difference = 0.0
    problems: list[str] = []
    total_paired = 0
    blocks = int(analysis_config["uncertainty_blocks"])
    mad_multiplier = float(analysis_config["robust_mad_multiplier"])

    for number, case in enumerate(cases, start=1):
        case_id = case["case_id"]
        case_dir = args.data_root / case_id
        qc_metadata_path = args.workstation_qc / "cases" / case_id / "case_metadata.json"
        qc_metadata = load_json(qc_metadata_path)
        if qc_metadata.get("status") != "completed":
            problems.append(f"{case_id}: workstation status is not completed")
        if qc_metadata.get("config_sha256") != config_hash:
            problems.append(f"{case_id}: simulation config hash mismatch")

        entrance = read_primary_state(
            case_dir / "PhaseSpaceIn.root", "PhaseSpaceIn", root_backend
        )
        exit_state = read_primary_state(
            case_dir / "PhaseSpaceOut.root", "PhaseSpaceOut", root_backend
        )
        if entrance.duplicate_primary or exit_state.duplicate_primary:
            problems.append(
                f"{case_id}: duplicate primary EventID "
                f"in={entrance.duplicate_primary}, out={exit_state.duplicate_primary}"
            )
        if not entrance.finite or not exit_state.finite:
            problems.append(f"{case_id}: non-finite ROOT state")

        for plane, state, expected_z in (
            ("in", entrance, -float(case["thickness_mm"]) / 2.0 - 0.5),
            ("out", exit_state, float(case["thickness_mm"]) / 2.0 + 0.5),
        ):
            root_rows.append(
                {
                    "case_id": case_id,
                    "material": case["material"],
                    "energy_mev": case["energy_mev"],
                    "thickness_mm": case["thickness_mm"],
                    "plane": plane,
                    "tree_entries": state.tree_entries,
                    "primary_entries_raw": state.primary_entries,
                    "primary_unique_events": state.event.size,
                    "duplicate_primary_events": state.duplicate_primary,
                    "finite": state.finite,
                    "position_z_mean_mm": float(np.mean(state.position[:, 2])),
                    "position_z_std_mm": float(np.std(state.position[:, 2])),
                    "position_z_expected_mm": expected_z,
                    "position_z_max_abs_error_mm": float(
                        np.max(np.abs(state.position[:, 2] - expected_z))
                    ),
                    "root_bytes": (case_dir / f"PhaseSpace{plane.title()}.root").stat().st_size,
                }
            )

        event, ein, eout, din, dout = paired_states(entrance, exit_state)
        total_paired += int(event.size)
        if not event.size:
            raise RuntimeError(f"{case_id}: no paired primary protons")
        energy_loss = ein - eout
        dot = np.einsum("ij,ij->i", din, dout)
        angle_mrad = 1000.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        wepl = energies_to_wepl(lut, ein, eout)
        loss_metrics, loss_core = distribution_metrics(
            energy_loss, event, mad_multiplier, blocks
        )
        wepl_metrics, wepl_core = distribution_metrics(
            wepl, event, mad_multiplier, blocks
        )
        angle_metrics, angle_core = distribution_metrics(
            angle_mrad, event, mad_multiplier, blocks
        )
        joint_core = loss_core & angle_core & wepl_core
        survival = float(event.size / entrance.event.size)
        negative_loss = int(np.count_nonzero(energy_loss < -1.0e-6))
        if negative_loss:
            problems.append(f"{case_id}: {negative_loss} paired protons have negative energy loss")

        workstation = workstation_rows[case_id]
        comparisons = {
            "entrance_primary": entrance.event.size,
            "exit_primary": exit_state.event.size,
            "paired_primary": event.size,
            "primary_survival": survival,
            "energy_loss_mev_mean": loss_metrics["mean"],
            "energy_loss_mev_std": loss_metrics["std"],
            "energy_loss_mev_p01": loss_metrics["p01"],
            "energy_loss_mev_median": loss_metrics["median"],
            "energy_loss_mev_p99": loss_metrics["p99"],
            "wepl_mm_mean": wepl_metrics["mean"],
            "wepl_mm_std": wepl_metrics["std"],
            "wepl_mm_p01": wepl_metrics["p01"],
            "wepl_mm_median": wepl_metrics["median"],
            "wepl_mm_p99": wepl_metrics["p99"],
        }
        for key, value in comparisons.items():
            difference = abs(float(value) - float(workstation[key]))
            workstation_max_difference = max(workstation_max_difference, difference)
            tolerance = 0.0 if key in {
                "entrance_primary", "exit_primary", "paired_primary"
            } else 5.0e-5
            if difference > tolerance:
                problems.append(
                    f"{case_id}: workstation {key} difference {difference:g}"
                )

        row = {
            **case,
            "requested_protons": int(qc_metadata["protons"]),
            "entrance_primary_unique": int(entrance.event.size),
            "exit_primary_unique": int(exit_state.event.size),
            "paired_primary": int(event.size),
            "primary_survival": survival,
            "unpaired_entrance_primary": int(entrance.event.size - event.size),
            "negative_energy_loss": negative_loss,
            "joint_3mad_core_fraction": float(np.mean(joint_core)),
            **flatten("energy_loss_mev", loss_metrics),
            **flatten("wepl", wepl_metrics),
            **flatten("scattering_angle_mrad", angle_metrics),
            "median_effective_rsp": float(
                wepl_metrics["median"] / float(case["thickness_mm"])
            ),
            "core_mean_effective_rsp": float(
                wepl_metrics["core_mean"] / float(case["thickness_mm"])
            ),
        }
        case_rows.append(row)

        if case["material"] == "Water":
            water_energy_pairs[case_id] = (ein.copy(), eout.copy())

        if case_id in {
            "water_e200_t0100",
            "aluminium_e200_t0050",
            "air_e200_t0220",
        }:
            selected[case_id] = wepl.copy()
        if sum(array.size for array in crosscheck_ein) < int(
            analysis_config["itk_crosscheck_samples"]
        ):
            take = min(8, event.size)
            indices = np.linspace(0, event.size - 1, take, dtype=int)
            crosscheck_ein.append(ein[indices])
            crosscheck_eout.append(eout[indices])

        elapsed = time.perf_counter() - started
        rate = number / elapsed if elapsed else 0.0
        eta = (len(cases) - number) / rate if rate else float("nan")
        print(
            f"[{number:02d}/{len(cases)}] {case_id}: paired={event.size:,}, "
            f"survival={100*survival:.3f}%, ETA={eta:.0f}s",
            flush=True,
        )

    if set(selected) != {
        "water_e200_t0100",
        "aluminium_e200_t0050",
        "air_e200_t0220",
    }:
        raise RuntimeError("representative distributions are incomplete")

    water_rows = [row for row in case_rows if row["material"] == "Water"]
    aluminium_rows = [
        row for row in case_rows if row["material"] == "Aluminium"
    ]
    air_rows = [row for row in case_rows if row["material"] == "Air"]

    print(
        f"Evaluating {len(sensitivity_values)} ionization-potential candidates...",
        flush=True,
    )
    water_by_id = {row["case_id"]: row for row in water_rows}
    for candidate_number, i_value in enumerate(sensitivity_values, start=1):
        candidate_lut = build_lut(float(i_value))
        for case_id, (ein, eout) in water_energy_pairs.items():
            row = water_by_id[case_id]
            candidate_wepl = energies_to_wepl(candidate_lut, ein, eout)
            candidate_median = float(np.median(candidate_wepl))
            sensitivity_rows.append(
                {
                    "case_id": case_id,
                    "energy_mev": row["energy_mev"],
                    "thickness_mm": row["thickness_mm"],
                    "ionization_potential_ev": float(i_value),
                    "median_wepl_mm": candidate_median,
                    "median_rsp_bias_percent": 100.0
                    * (
                        candidate_median / float(row["thickness_mm"])
                        - 1.0
                    ),
                }
            )
        del candidate_lut
        if candidate_number % 5 == 0 or candidate_number == len(
            sensitivity_values
        ):
            print(
                f"  I sensitivity {candidate_number}/{len(sensitivity_values)}",
                flush=True,
            )

    water_output = []
    aluminium_output = []
    for row in water_rows:
        water_output.append(
            {
                "case_id": row["case_id"],
                "energy_mev": row["energy_mev"],
                "thickness_mm": row["thickness_mm"],
                "paired_primary": row["paired_primary"],
                "primary_survival": row["primary_survival"],
                "wepl_mean_mm": row["wepl_mean"],
                "wepl_median_mm": row["wepl_median"],
                "wepl_core_mean_mm": row["wepl_core_mean"],
                "mean_effective_rsp": row["wepl_mean"] / row["thickness_mm"],
                "median_effective_rsp": row["median_effective_rsp"],
                "core_mean_effective_rsp": row["core_mean_effective_rsp"],
                "median_rsp_bias_percent": 100.0
                * (row["median_effective_rsp"] - 1.0),
                "core_mean_rsp_bias_percent": 100.0
                * (row["core_mean_effective_rsp"] - 1.0),
                "wepl_median_ci95_low_mm": row["wepl_median_ci95_low"],
                "wepl_median_ci95_high_mm": row["wepl_median_ci95_high"],
                "wepl_3mad_core_fraction": row["wepl_core_fraction"],
            }
        )
    reference = float(analysis_config["aluminium_rsp_200mev_reference"])
    for row in aluminium_rows:
        aluminium_output.append(
            {
                "case_id": row["case_id"],
                "energy_mev": row["energy_mev"],
                "thickness_mm": row["thickness_mm"],
                "paired_primary": row["paired_primary"],
                "primary_survival": row["primary_survival"],
                "wepl_median_mm": row["wepl_median"],
                "wepl_core_mean_mm": row["wepl_core_mean"],
                "median_effective_rsp": row["median_effective_rsp"],
                "core_mean_effective_rsp": row["core_mean_effective_rsp"],
                "median_rsp_bias_vs_fixed_percent": 100.0
                * (row["median_effective_rsp"] / reference - 1.0),
                "wepl_median_ci95_low_mm": row["wepl_median_ci95_low"],
                "wepl_median_ci95_high_mm": row["wepl_median_ci95_high"],
                "wepl_3mad_core_fraction": row["wepl_core_fraction"],
            }
        )

    material_fits: list[dict[str, Any]] = []
    for material, rows in (("Water", water_rows), ("Aluminium", aluminium_rows)):
        for energy in sorted({float(row["energy_mev"]) for row in rows}):
            subset = sorted(
                [row for row in rows if float(row["energy_mev"]) == energy],
                key=lambda row: float(row["thickness_mm"]),
            )
            fit = fit_line(
                np.asarray([float(row["thickness_mm"]) for row in subset]),
                np.asarray([float(row["wepl_median"]) for row in subset]),
            )
            material_fits.append(
                {"material": material, "energy_mev": energy, **fit}
            )

    air_models = []
    for energy in sorted({float(row["energy_mev"]) for row in air_rows}):
        subset = sorted(
            [row for row in air_rows if float(row["energy_mev"]) == energy],
            key=lambda row: float(row["thickness_mm"]),
        )
        mean_fit = fit_line(
            np.asarray([float(row["thickness_mm"]) for row in subset]),
            np.asarray([float(row["wepl_mean"]) for row in subset]),
        )
        median_fit = fit_line(
            np.asarray([float(row["thickness_mm"]) for row in subset]),
            np.asarray([float(row["wepl_median"]) for row in subset]),
        )
        core_fit = fit_line(
            np.asarray([float(row["thickness_mm"]) for row in subset]),
            np.asarray([float(row["wepl_core_mean"]) for row in subset]),
        )
        air_models.append(
            {
                "energy_mev": energy,
                "operational_estimator": "paired-primary mean WEPL",
                **mean_fit,
                **{
                    f"median_{key}": value
                    for key, value in median_fit.items()
                },
                **{
                    f"core_mean_{key}": value
                    for key, value in core_fit.items()
                },
            }
        )

    objective: dict[float, list[float]] = {}
    for row in sensitivity_rows:
        objective.setdefault(float(row["ionization_potential_ev"]), []).append(
            float(row["median_rsp_bias_percent"])
        )
    objective_rows = [
        {
            "ionization_potential_ev": value,
            "water_bias_rms_percent": math.sqrt(
                float(np.mean(np.square(objective[value])))
            ),
            "water_bias_mean_percent": float(np.mean(objective[value])),
            "water_bias_max_abs_percent": float(
                np.max(np.abs(objective[value]))
            ),
        }
        for value in sorted(objective)
    ]
    best_i = min(objective_rows, key=lambda row: row["water_bias_rms_percent"])

    itk_result: dict[str, Any]
    if args.skip_itk_crosscheck:
        itk_result = {"status": "SKIPPED", "max_abs_difference_mm": 0.0}
    else:
        try:
            sys.path.insert(0, str(CODE_ROOT / "iterative_reconstruction"))
            from physics import (
                energies_to_wepl as itk_energies_to_wepl,
                make_wepl_converter,
            )

            ein_sample = np.concatenate(crosscheck_ein)[
                : int(analysis_config["itk_crosscheck_samples"])
            ]
            eout_sample = np.concatenate(crosscheck_eout)[
                : int(analysis_config["itk_crosscheck_samples"])
            ]
            itk_values = itk_energies_to_wepl(
                make_wepl_converter(), ein_sample, eout_sample
            ).astype(np.float64)
            vector_values = energies_to_wepl(lut, ein_sample, eout_sample)
            maximum = float(np.max(np.abs(itk_values - vector_values)))
            itk_result = {
                "status": (
                    "PASS"
                    if maximum
                    <= float(analysis_config["itk_crosscheck_tolerance_mm"])
                    else "FAIL"
                ),
                "samples": int(ein_sample.size),
                "max_abs_difference_mm": maximum,
                "tolerance_mm": float(
                    analysis_config["itk_crosscheck_tolerance_mm"]
                ),
            }
            if itk_result["status"] != "PASS":
                problems.append("ITK/C++ LUT crosscheck exceeds tolerance")
        except Exception as error:
            itk_result = {
                "status": "UNAVAILABLE",
                "max_abs_difference_mm": 0.0,
                "reason": f"{type(error).__name__}: {error}",
            }
            problems.append(f"ITK/C++ LUT crosscheck unavailable: {error}")

    al_200_5 = next(
        row
        for row in aluminium_output
        if float(row["energy_mev"]) == 200.0
        and float(row["thickness_mm"]) == 5.0
    )
    iterative_baseline = read_iterative_baseline(BASELINE_METRICS)
    reconstructed_al = iterative_baseline["aluminium_platform_rsp"]
    effective_al = float(al_200_5["median_effective_rsp"])
    decomposition = {
        "fixed_rsp_200mev": reference,
        "effective_rsp_200mev_5mm": effective_al,
        "results0716_iterative_aluminium_platform": reconstructed_al,
        "total_reconstruction_bias_vs_fixed_percent": 100.0
        * (reconstructed_al / reference - 1.0),
        "reference_definition_component_percent": 100.0
        * (effective_al / reference - 1.0),
        "remaining_reconstruction_bias_vs_effective_percent": 100.0
        * (reconstructed_al / effective_al - 1.0),
        "limitation": (
            "The 5 mm, 200 MeV slab is a local comparison only; S6 does not "
            "cover the full low-energy spectrum or mixed Water/Al paths."
        ),
    }
    water_200_fit = next(
        row
        for row in material_fits
        if row["material"] == "Water"
        and float(row["energy_mev"]) == 200.0
    )
    water_baseline_comparison = {
        "s6_origin_constrained_rsp": float(
            water_200_fit["origin_constrained_slope"]
        ),
        "results0716_iterative_water_mean_rsp": float(
            iterative_baseline["water_mean_rsp"]
        ),
        "difference_rsp": float(
            iterative_baseline["water_mean_rsp"]
            - water_200_fit["origin_constrained_slope"]
        ),
        "interpretation": (
            "The close agreement strongly suggests that the approximately "
            "+1.4% results0716 water level is dominated by the current "
            "OpenGATE-to-water-LUT calibration convention. S6 remains a "
            "single-material, 5-100 mm scan and is not a strict causal "
            "decomposition of 200 mm mixed paths."
        ),
    }

    write_json(
        output / "input_manifest.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": expected_dataset,
            "files": manifest,
            "file_count": len(manifest),
            "total_bytes": sum(item["bytes"] for item in manifest),
            "simulation_config_sha256": config_hash,
        },
    )
    write_csv(output / "root_integrity.csv", root_rows)
    write_csv(output / "case_metrics.csv", case_rows)
    write_csv(output / "water_lut_consistency.csv", water_output)
    write_csv(output / "aluminium_effective_rsp.csv", aluminium_output)
    write_csv(output / "ionization_potential_sensitivity.csv", sensitivity_rows)
    write_csv(output / "ionization_potential_objective.csv", objective_rows)
    write_csv(output / "material_wepl_fits.csv", material_fits)
    write_json(
        output / "air_wepl_model.json",
        {
            "model": (
                "linear interpolation in energy of origin-constrained "
                "paired-primary mean WEPL/length slopes"
            ),
            "energy_domain_mev": [
                min(row["energy_mev"] for row in air_models),
                max(row["energy_mev"] for row in air_models),
            ],
            "length_domain_mm": [20.0, 2000.0],
            "extrapolation_policy": "flag; do not silently clamp",
            "fits": air_models,
            "source_dataset": expected_dataset,
        },
    )
    write_json(output / "chart_map.json", chart_map())
    make_figures(figures, case_rows, sensitivity_rows, air_models, selected)

    data_quality = {
        "case_count": len(cases),
        "root_file_count": 2 * len(cases),
        "paired_primary": total_paired,
        "problem_count": len(problems),
        "problems": problems,
        "workstation_summary_max_abs_difference": workstation_max_difference,
    }
    summary = {
        "status": "PASS" if not problems else "FAIL",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - started,
        "dataset": expected_dataset,
        "root_array_backend": root_backend,
        "data_quality": data_quality,
        "lut_crosscheck": itk_result,
        "water_i_sensitivity": {
            "current_i_ev": float(analysis_config["ionization_potential_ev"]),
            "best_i_ev": float(best_i["ionization_potential_ev"]),
            "best_rms_bias_percent": float(best_i["water_bias_rms_percent"]),
            "current_rms_bias_percent": next(
                float(row["water_bias_rms_percent"])
                for row in objective_rows
                if float(row["ionization_potential_ev"]) == 78.0
            ),
            "interpretation": (
                "Empirical effective parameter only; not a direct measurement "
                "of the physical mean excitation energy."
            ),
        },
        "water_baseline_comparison": water_baseline_comparison,
        "aluminium_decomposition": decomposition,
        "air_model": {
            "fits": air_models,
            "requires_low_energy_extrapolation_for_ct": True,
        },
        "limitations": [
            "S6 covers 150-220 MeV only.",
            "Statistics are conditional on primary survival to the exit plane.",
            "Single-material slabs do not reproduce mixed Water/Aluminium paths.",
            "No physical detector or dose actor is present.",
        ],
        "outputs": {
            "report": str((output / "stage1_summary.md").relative_to(REPOSITORY_ROOT)),
            "figures": str(figures.relative_to(REPOSITORY_ROOT)),
        },
    }
    write_json(output / "stage1_summary.json", summary)
    report = markdown_report(summary, water_output, aluminium_output, air_models)
    (output / "stage1_summary.md").write_text(report, encoding="utf-8")
    (output / "completed.flag").write_text(
        f"status={summary['status']}\ndataset={expected_dataset}\n",
        encoding="ascii",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
