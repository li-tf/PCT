#!/usr/bin/env python3
"""Build reproducible figures for current_research_summary.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY_ROOT = CODE_ROOT.parent
ASSET_ROOT = HERE / "research_stages_summary" / "assets" / "current_summary"
sys.path.insert(0, str(CODE_ROOT / "iterative_reconstruction"))
from mhd_io import read_image_2d  # noqa: E402


def save(fig, name: str) -> None:
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSET_ROOT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def scenario_overview() -> None:
    scenarios = [
        ("S1", "Air + 25 Al rods", "720×450k", "Air effect at paper fluence"),
        ("S2", "Vacuum + uniform water", "720×100k", "Boundary artifact control"),
        ("S3", "Air + uniform water", "720×100k", "Paired Air comparison"),
        ("S4", "Air + 5 materials", "720×100k", "RSP accuracy / partial volume"),
        ("S5", "Air + line pairs", "720×100k", "fMTF / resolution"),
        ("S6", "Water/Al/Air slabs", "52×100k", "Energy-dependent RSP"),
        ("MLP", "Air + 5 materials + steps", "72×5k", "True-path upper bound"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis("off")
    colors = ["#dbeafe", "#e0f2fe", "#cffafe", "#dcfce7", "#fef3c7", "#fae8ff", "#fee2e2"]
    positions = [(0.1, 2.6), (1.4, 2.6), (2.7, 2.6), (0.1, 1.35), (1.4, 1.35), (2.7, 1.35), (1.4, 0.1)]
    for (title, scene, fluence, purpose), color, (x, y) in zip(scenarios, colors, positions):
        patch = FancyBboxPatch(
            (x, y), 1.15, 1.0, boxstyle="round,pad=0.035",
            facecolor=color, edgecolor="#334155", linewidth=1.2
        )
        ax.add_patch(patch)
        ax.text(x + 0.08, y + 0.78, title, fontsize=14, weight="bold")
        ax.text(x + 0.08, y + 0.57, scene, fontsize=9)
        ax.text(x + 0.08, y + 0.38, fluence, fontsize=9, color="#334155")
        ax.text(x + 0.08, y + 0.12, purpose, fontsize=8.5, color="#475569", wrap=True)
    ax.text(2, 3.85, "S1–S6 and true-path pilot", ha="center", fontsize=17, weight="bold")
    ax.text(
        2, 3.62,
        "All CT scenarios: 200 MeV, 100 mm-radius water cylinder, ideal phase-space planes",
        ha="center", fontsize=9.5, color="#475569"
    )
    save(fig, "scenario_overview.png")


def stage_decisions() -> None:
    labels = [
        ("0", "Freeze baseline", "PASS"),
        ("1", "RSP/WEPL calibration", "PASS"),
        ("2", "Diagnostic phantoms", "PASS"),
        ("3", "Robust weighting", "RETAIN"),
        ("4", "Iterative tuning", "PROMOTE"),
        ("5", "Inhomogeneous MLP", "RETAIN"),
        ("6", "Advanced priors", "RETAIN"),
        ("6A", "Virtual MLIC reference", "PASS"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.8, 1.3)
    ax.axis("off")
    ax.plot(range(8), [0] * 8, color="#94a3b8", linewidth=3, zorder=0)
    for index, (number, title, decision) in enumerate(labels):
        color = "#16a34a" if decision == "PROMOTE" else ("#2563eb" if decision == "PASS" else "#d97706")
        ax.scatter(index, 0, s=520, color=color, edgecolor="white", linewidth=2, zorder=2)
        ax.text(index, 0, number, color="white", ha="center", va="center", weight="bold", fontsize=12)
        ax.text(index, 0.42, title, ha="center", va="bottom", fontsize=8.5, rotation=18)
        ax.text(index, -0.4, decision, ha="center", fontsize=9, color=color, weight="bold")
    ax.text(3.5, 1.13, "Stage 0–6A decisions", ha="center", fontsize=16, weight="bold")
    ax.text(
        3.5, -0.72,
        "Stage 4 changed the reconstruction; Stage 6A froze the external RSP reference",
        ha="center", fontsize=10, color="#475569"
    )
    save(fig, "stage_decisions.png")


def benchmark_context() -> None:
    names = ["Current S4\nideal 2-D", "Phase-II\nprototype", "ProtonVDA\nprototype", "2024 pCT\nplastic phantom"]
    mape = [1.1924, 1.14, 0.81, 0.28]
    resolution = [1.1167, 0.61, 0.46, 0.54]
    colors = ["#f59e0b", "#3b82f6", "#8b5cf6", "#10b981"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(names, mape, color=colors)
    axes[0].axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.3, label="~1% prototype target")
    axes[0].set_ylabel("RSP MAPE (%)")
    axes[0].set_title("Reported RSP accuracy (lower is better)")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, resolution, color=colors)
    axes[1].set_ylabel("Reported spatial frequency (lp/mm)")
    axes[1].set_title("Reported spatial resolution (higher is better)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Context only: phantoms, dose, dimensionality and metrics differ", fontsize=13, weight="bold")
    fig.text(
        0.5, -0.02,
        "The current result is an ideal 2-D Monte Carlo study; prototype values include real detector effects.",
        ha="center", fontsize=9, color="#475569"
    )
    save(fig, "benchmark_context.png")


def stage6_tradeoff() -> None:
    roots = {
        "S2": "results0717_s2_water_vacuum_pilot",
        "S5": "results0717_s5_resolution_air_pilot",
    }
    base_name = "r0p25_d0p2_quadratic_b0p0125_fixed_s18"
    candidate = "directional_tv_b0p0125_m0p5"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    for row, (label, dataset) in enumerate(roots.items()):
        root = REPOSITORY_ROOT / "data" / "reconstruction_data" / dataset
        baseline = read_image_2d(root / "stage4" / "variants" / base_name / "recon" / "epoch_05.mhd")[0]
        advanced = read_image_2d(root / "stage6" / "variants" / candidate / "recon" / "epoch_05.mhd")[0]
        difference = advanced - baseline
        extent = [-105, 105, 105, -105]
        for column, (image, title, cmap, vmin, vmax) in enumerate([
            (baseline, f"{label}: Stage 4", "viridis", 0.95, 1.08 if label == "S2" else 2.2),
            (advanced, f"{label}: Directional TV", "viridis", 0.95, 1.08 if label == "S2" else 2.2),
            (difference, f"{label}: difference", "coolwarm", -0.03, 0.03),
        ]):
            shown = axes[row, column].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
            axes[row, column].set_title(title, fontsize=10)
            axes[row, column].set_xlabel("x (mm)")
            axes[row, column].set_ylabel("z (mm)")
            colorbar = fig.colorbar(
                shown, ax=axes[row, column], fraction=0.035, pad=0.025
            )
            colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Stage 6 trade-off: sharper but noisier", fontsize=15, weight="bold")
    save(fig, "stage6_tradeoff.png")


def best_pipeline() -> None:
    blocks = [
        ("Filtered pairs", "primary-only\nlocal 3σ"),
        ("Water MLP", "0.1 mm path\nbilinear weights"),
        ("OS-SART", "18 subsets\nλ₀=0.25, d=0.2"),
        ("Constraints", "nonnegative\n100 mm support"),
        ("Huber-TV", "β=0.0125\n5 epochs"),
        ("RSP image", "0.1 mm grid"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    for index, (title, detail) in enumerate(blocks):
        x = 0.2 + index * 1.95
        patch = FancyBboxPatch(
            (x, 0.8), 1.55, 1.25, boxstyle="round,pad=0.04",
            facecolor="#eff6ff" if index < 5 else "#dcfce7",
            edgecolor="#2563eb" if index < 5 else "#16a34a",
            linewidth=1.4,
        )
        ax.add_patch(patch)
        ax.text(x + 0.775, 1.62, title, ha="center", fontsize=10.5, weight="bold")
        ax.text(x + 0.775, 1.08, detail, ha="center", fontsize=9, color="#475569")
        if index < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.9, 1.42), xytext=(x + 1.58, 1.42),
                        arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.6})
    ax.text(6, 2.65, "Frozen best 2-D reconstruction pipeline", ha="center", fontsize=16, weight="bold")
    save(fig, "best_pipeline.png")


def classic_scenario_results() -> None:
    """Compare the Stage-6B calibrated reconstruction against MLIC truth."""
    water_mlic = 0.9997458098
    aluminium_mlic = 2.0945112079
    material_mapping = {
        1.0: water_mlic,
        0.2574898899: 0.2581452607,
        1.1244583130: 1.1242445861,
        1.3215450048: 1.3222608704,
        2.1021010876: aluminium_mlic,
    }
    scenarios = [
        (
            "S1: 25 aluminium rods",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0716/analytic/truth/truth_rsp_200mev.mhd",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s1_aluminium_air_full"
            / "stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
            "aluminium",
        ),
        (
            "S4: multi-material phantom",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot"
            / "analytic/truth/truth_rsp_200mev.mhd",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot"
            / "stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
            "discrete",
        ),
        (
            "S5: line pairs and slanted edges",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s5_resolution_air_pilot"
            / "analytic/truth/truth_rsp_200mev.mhd",
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s5_resolution_air_pilot"
            / "stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
            "discrete",
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 12.0), constrained_layout=True)
    extent = [-105, 105, 105, -105]
    rsp_show = None
    error_show = None
    for row, (label, truth_path, reconstruction_path, truth_kind) in enumerate(scenarios):
        fixed_truth = np.asarray(read_image_2d(truth_path)[0], dtype=np.float32)
        reconstruction = np.asarray(
            read_image_2d(reconstruction_path)[0], dtype=np.float32
        )
        if truth_kind == "aluminium":
            truth = np.empty_like(fixed_truth)
            water_side = fixed_truth <= 1.0
            truth[water_side] = fixed_truth[water_side] * water_mlic
            aluminium_fraction = (
                (fixed_truth[~water_side] - 1.0) / (2.1189761162 - 1.0)
            )
            truth[~water_side] = (
                water_mlic
                + aluminium_fraction * (aluminium_mlic - water_mlic)
            )
        else:
            truth = np.zeros_like(fixed_truth)
            for old_value, new_value in material_mapping.items():
                truth[np.isclose(fixed_truth, old_value, rtol=0.0, atol=1e-5)] = (
                    new_value
                )

        error = reconstruction - truth
        for column, (image, title) in enumerate(
            [
                (truth, "MLIC-valued truth"),
                (reconstruction, "Stage-6B calibrated reconstruction"),
            ]
        ):
            rsp_show = axes[row, column].imshow(
                image, cmap="turbo", vmin=0.0, vmax=2.2, extent=extent
            )
            axes[row, column].set_title(title, fontsize=10.5)
        error_show = axes[row, 2].imshow(
            error, cmap="coolwarm", vmin=-0.08, vmax=0.08, extent=extent
        )
        axes[row, 2].set_title("Reconstruction − truth", fontsize=10.5)
        axes[row, 0].set_ylabel(f"{label}\nz (mm)", fontsize=10)
        for axis in axes[row]:
            axis.set_xlabel("x (mm)")
            axis.set_aspect("equal")

    fig.colorbar(
        rsp_show, ax=axes[:, :2], fraction=0.025, pad=0.015, label="RSP"
    )
    fig.colorbar(
        error_show, ax=axes[:, 2], fraction=0.05, pad=0.025, label="RSP error"
    )
    fig.suptitle(
        "Calibrated best algorithm in three classic 2-D scenarios",
        fontsize=15,
        weight="bold",
    )
    save(fig, "classic_scenario_results.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-images", action="store_true", help="skip MHD comparison panel")
    args = parser.parse_args()
    scenario_overview()
    stage_decisions()
    benchmark_context()
    best_pipeline()
    if not args.skip_images:
        stage6_tradeoff()
        classic_scenario_results()
    print(ASSET_ROOT)


if __name__ == "__main__":
    main()
