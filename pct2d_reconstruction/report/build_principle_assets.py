#!/usr/bin/env python3
"""Build reproducible principle figures from the frozen results0716 products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent
sys.path[:0] = [
    str(CODE_ROOT),
    str(CODE_ROOT / "iterative_reconstruction"),
]

from common import load_experiment, path_for  # noqa: E402
from preprocessing import paircuts  # noqa: E402
from analytic_reconstruction import rsp_metrics  # noqa: E402
from mhd_io import read_pairs  # noqa: E402
from mlp import _constant, _integrals, cylinder_intersections, schulte_positions  # noqa: E402
from physics import make_vectorized_wepl_lut, energies_to_wepl_vectorized  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-pct2d-principles")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402


ASSET_NAMES = [
    "01_pipeline.png",
    "02_pairing_planes.png",
    "03_filtering_actual.png",
    "04_bethe_bloch_wepl.png",
    "05_schulte_mlp_actual.png",
    "06_ddb_binning.png",
    "07_ddb_actual.png",
    "08_fdk_pipeline.png",
    "09_analytic_actual.png",
    "10_iterative_operator.png",
    "11_iterative_epochs.png",
    "12_huber_regularization.png",
    "13_iterative_runtime.png",
    "14_validation_wepl.png",
    "08a_analytic_dimensions.png",
    "08b_uvd_mapping.png",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig, path: Path, dpi: int = 190) -> None:
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def pipeline_figure(output: Path, counts: dict[str, int]) -> None:
    fig, axis = plt.subplots(figsize=(15.2, 5.2), constrained_layout=True)
    axis.set_xlim(0, 15.2)
    axis.set_ylim(0, 5.2)
    axis.axis("off")
    boxes = [
        (0.15, "Phase-space ROOT", "720 In/Out pairs\nposition, direction, energy"),
        (2.7, "Primary pairing", f"{counts['pairs']/1e6:.2f} M list-mode pairs\nN x 5 x 3"),
        (5.25, "3-sigma filtering", f"{counts['filtered']/1e6:.2f} M retained\nenergy loss + two angles"),
        (7.8, "WEPL + Schulte MLP", "water range LUT\ncurved path at each depth"),
        (10.35, "DDB-FDK", "720 x (500 x 2 x 500)\n2100 x 2100 RSP"),
        (12.9, "List-mode OS-SART", "18 subsets x 3 epochs\nMLP + Huber-TV"),
    ]
    for i, (x, title, body) in enumerate(boxes):
        color = "#EAF2FA" if i < 4 else "#F3EAF7"
        axis.add_patch(Rectangle((x, 1.25), 2.05, 2.55, facecolor=color,
                                 edgecolor="#344054", linewidth=1.25))
        axis.text(x + 1.025, 3.05, title, ha="center", va="center",
                  fontsize=10.5, weight="bold")
        axis.text(x + 1.025, 2.25, body, ha="center", va="center",
                  fontsize=8.7, color="#475467")
        if i < len(boxes) - 1:
            axis.add_patch(FancyArrowPatch((x + 2.07, 2.52), (boxes[i + 1][0] - 0.06, 2.52),
                                           arrowstyle="-|>", mutation_scale=14,
                                           color="#667085", linewidth=1.2))
    axis.text(0.15, 4.65, "results0716: measured states to analytic and iterative RSP reconstruction",
              fontsize=15, weight="bold")
    axis.text(0.15, 0.45,
              "The DDB branch averages WEPL on a depth-indexed projection lattice; the iterative branch keeps individual protons.",
              fontsize=9.5, color="#475467")
    save(fig, output)


def pairing_planes_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
    ax = axes[0]
    ax.add_patch(Circle((0, 0), 100, facecolor="#DCEEF8", edgecolor="#2878A4", linewidth=1.5))
    ax.axvline(-110, color="#344054", linewidth=2)
    ax.axvline(110, color="#344054", linewidth=2)
    ax.scatter([-132, 133], [-35, 28], color="#C8553D", zorder=4)
    ax.arrow(-132, -35, 35, 23, width=0.6, head_width=4, color="#C8553D", length_includes_head=True)
    ax.arrow(133, 28, -34, -20, width=0.6, head_width=4, color="#C8553D", length_includes_head=True)
    ax.plot([-110, 110], [-20.5, 8.0], color="#2463A6", linewidth=2.2)
    ax.text(-110, 106, "entrance reference\nz = -110 mm", ha="center", fontsize=9)
    ax.text(110, 106, "exit reference\nz = +110 mm", ha="center", fontsize=9)
    ax.text(0, -8, "water support", ha="center", color="#2878A4")
    ax.set(xlim=(-155, 155), ylim=(-125, 125), xlabel="z (mm)", ylabel="transverse position (mm)",
           title="Measured states extrapolated to fixed planes")
    ax.set_aspect("equal")
    ax.grid(color="#EAECF0", linewidth=0.7)

    ax = axes[1]
    z0, t0, dz, dt = -145.0, -42.0, 1.0, 0.19
    zstar = -110.0
    tstar = t0 + (zstar - z0) * dt / dz
    ax.plot([z0, zstar + 24], [t0, t0 + (zstar + 24 - z0) * dt], color="#2463A6", linewidth=2)
    ax.scatter([z0, zstar], [t0, tstar], color=["#C8553D", "#2463A6"], s=60, zorder=3)
    ax.axvline(zstar, color="#344054", linestyle="--")
    ax.annotate("recorded state", (z0, t0), xytext=(-140, -25), arrowprops={"arrowstyle": "->"})
    ax.annotate("reference-plane state", (zstar, tstar), xytext=(-105, -43), arrowprops={"arrowstyle": "->"})
    ax.text(-143, 5, r"$\mathbf{p}(z_*)=\mathbf{p}+\frac{z_*-z}{d_z}\mathbf{d}$", fontsize=15)
    ax.set(xlim=(-150, -78), ylim=(-50, 12), xlabel="z (mm)", ylabel="transverse position (mm)",
           title="Straight extrapolation outside the object")
    ax.grid(color="#EAECF0", linewidth=0.7)
    save(fig, output)


def filtering_figure(output: Path, pairs_path: Path, filtered_path: Path) -> None:
    before = np.asarray(read_pairs(pairs_path), dtype=np.float32)
    after = np.asarray(read_pairs(filtered_path), dtype=np.float32)
    _, _, ax0, ay0, de0 = paircuts.pair_features(before)
    _, _, ax1, ay1, de1 = paircuts.pair_features(after)
    scatter0 = 1e3 * np.hypot(ax0, ay0)
    scatter1 = 1e3 * np.hypot(ax1, ay1)
    rng = np.random.default_rng(20260713)
    i0 = rng.choice(len(before), min(120_000, len(before)), replace=False)
    i1 = rng.choice(len(after), min(120_000, len(after)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), constrained_layout=True)
    for ax, energy, angle, title in (
        (axes[0, 0], de0[i0], scatter0[i0], "Before local 3-sigma filtering"),
        (axes[0, 1], de1[i1], scatter1[i1], "After local 3-sigma filtering"),
    ):
        shown = ax.hexbin(energy, angle, gridsize=85, bins="log", mincnt=1, cmap="viridis")
        ax.set(xlabel="Energy loss (MeV)", ylabel="combined projected angle (mrad)", title=title)
        fig.colorbar(shown, ax=ax, label="log count")
    bins_e = np.linspace(np.percentile(de0, 0.1), np.percentile(de0, 99.9), 160)
    axes[1, 0].hist(de0, bins=bins_e, density=True, histtype="step", linewidth=1.5, label="before")
    axes[1, 0].hist(de1, bins=bins_e, density=True, histtype="step", linewidth=1.8, label="after")
    axes[1, 0].set(xlabel="Energy loss (MeV)", ylabel="density", title="Energy-loss marginal")
    axes[1, 0].legend(frameon=False)
    bins_a = np.linspace(0, np.percentile(scatter0, 99.8), 160)
    axes[1, 1].hist(scatter0, bins=bins_a, density=True, histtype="step", linewidth=1.5, label="before")
    axes[1, 1].hist(scatter1, bins=bins_a, density=True, histtype="step", linewidth=1.8, label="after")
    axes[1, 1].set(xlabel="combined projected angle (mrad)", ylabel="density", title="Scattering-angle marginal")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("results0716 run 000: actual pair distributions", fontsize=15, weight="bold")
    save(fig, output)


def bethe_bloch_figure(output: Path, filtered_path: Path) -> None:
    lut = make_vectorized_wepl_lut()
    energy = np.linspace(1.0, 250.0, 1000)
    index = np.floor(energy / 0.0001 + 0.5).astype(np.int64)
    range_mm = lut[index]
    stopping = np.gradient(energy, range_mm)
    pairs = np.asarray(read_pairs(filtered_path), dtype=np.float32)
    wepl = energies_to_wepl_vectorized(lut, pairs[:, 4, 0], pairs[:, 4, 1])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)
    axes[0].plot(energy, stopping, color="#2463A6", linewidth=2)
    axes[0].set(xlabel="Proton energy (MeV)", ylabel="Water stopping power (MeV/mm)",
                title="Bethe-Bloch water stopping power")
    axes[1].plot(energy, range_mm, color="#287A5A", linewidth=2)
    axes[1].set(xlabel="Proton energy (MeV)", ylabel="Integrated water range (mm)",
                title=r"$R_w(E)=\int_0^E dE'/S_w(E')$")
    axes[2].hist(wepl, bins=150, color="#8B5AA6", alpha=0.85)
    axes[2].set(xlabel="WEPL (mm)", ylabel="proton count", title="Actual run 000 WEPL distribution")
    for ax in axes:
        ax.grid(color="#EAECF0", linewidth=0.7)
    fig.suptitle("I = 78 eV water LUT and measured energy-to-WEPL conversion", fontsize=15, weight="bold")
    save(fig, output)


def posterior_position_sigma(length: float, u_values: np.ndarray) -> np.ndarray:
    result = np.empty_like(u_values)
    theta_total, ttheta_total, t_total = _integrals(np.asarray(length))
    for i, u in enumerate(u_values):
        theta1, ttheta1, t1 = _integrals(np.asarray(u))
        s1 = np.array([[u * (2 * (u * theta1 - ttheta1) - u * theta1) + t1,
                        u * theta1 - ttheta1], [u * theta1 - ttheta1, theta1]], dtype=float)
        s1 *= float(_constant(np.asarray(0.0), np.asarray(u)))
        s2theta = theta_total - theta1
        s2cross = length * s2theta - ttheta_total + ttheta1
        s2 = np.array([[length * (2 * s2cross - length * s2theta) + t_total - t1,
                        s2cross], [s2cross, s2theta]], dtype=float)
        s2 *= float(_constant(np.asarray(u), np.asarray(length)))
        remaining = length - u
        r1 = np.array([[1.0, remaining], [0.0, 1.0]])
        precision = np.linalg.pinv(s1) + r1.T @ np.linalg.pinv(s2) @ r1
        covariance = np.linalg.pinv(precision)
        result[i] = math.sqrt(max(covariance[0, 0], 0.0))
    return result


def mlp_figure(output: Path, filtered_path: Path) -> None:
    pairs = np.asarray(read_pairs(filtered_path), dtype=np.float32)
    entry, exitp, valid = cylinder_intersections(
        pairs[:, 0], pairs[:, 1], pairs[:, 2], pairs[:, 3], 100.0
    )
    candidates = np.flatnonzero(valid)
    angle = np.hypot(
        np.arctan(pairs[candidates, 3, 0] / pairs[candidates, 3, 2]) - np.arctan(pairs[candidates, 2, 0] / pairs[candidates, 2, 2]),
        np.arctan(pairs[candidates, 3, 1] / pairs[candidates, 3, 2]) - np.arctan(pairs[candidates, 2, 1] / pairs[candidates, 2, 2]),
    )
    order = candidates[np.argsort(angle)]
    chosen = order[(np.array([0.20, 0.45, 0.70, 0.90, 0.98]) * (len(order) - 1)).astype(int)]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.9, len(chosen)))
    selected_record = None
    for color, idx in zip(colors, chosen):
        z = np.linspace(entry[idx, 2] + 1e-3, exitp[idx, 2] - 1e-3, 500)
        x, _, _ = schulte_positions(entry[idx:idx+1], exitp[idx:idx+1],
                                    pairs[idx:idx+1, 2], pairs[idx:idx+1, 3], z)
        straight = entry[idx, 0] + (z - entry[idx, 2]) * (exitp[idx, 0] - entry[idx, 0]) / (exitp[idx, 2] - entry[idx, 2])
        axes[0].plot(z, x[0] - straight, color=color, linewidth=1.8)
        selected_record = (idx, z, x[0], straight)
    axes[0].axhline(0, color="#98A2B3", linewidth=1)
    axes[0].set(xlabel="depth z (mm)", ylabel="MLP - entrance/exit chord (mm)",
                title="Five actual protons at increasing scattering quantiles")
    axes[0].grid(color="#EAECF0", linewidth=0.7)
    idx, z, x, straight = selected_record
    length = float(exitp[idx, 2] - entry[idx, 2])
    u = z - entry[idx, 2]
    sigma = posterior_position_sigma(length, u)
    axes[1].plot(z, x, color="#2463A6", linewidth=2.2, label="Schulte MLP")
    axes[1].plot(z, straight, "--", color="#C8553D", linewidth=1.5, label="straight chord")
    axes[1].fill_between(z, x - 2 * sigma, x + 2 * sigma, color="#2463A6", alpha=0.18,
                         label=r"approx. $\pm2\sigma_t$")
    axes[1].scatter([entry[idx, 2], exitp[idx, 2]], [entry[idx, 0], exitp[idx, 0]], color="#101828", s=32)
    axes[1].set(xlabel="depth z (mm)", ylabel="transverse x (mm)",
                title="High-scattering measured path and posterior envelope")
    axes[1].legend(frameon=False)
    axes[1].grid(color="#EAECF0", linewidth=0.7)
    fig.suptitle("Schulte most-likely paths computed from results0716 pairs", fontsize=15, weight="bold")
    save(fig, output)


def ddb_binning_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), constrained_layout=True)
    ax = axes[0]
    source = (-170, 0)
    depths = np.array([-100, -50, 0, 50, 100])
    path = 0.0009 * depths**2 + 0.08 * depths - 12
    ax.scatter(*source, marker="*", s=130, color="#C8553D", label="effective source")
    for z, x in zip(depths, path):
        ax.axvline(z, color="#D0D5DD", linewidth=0.8)
        ax.scatter(z, x, color="#2463A6", zorder=3)
        magnified = x * (110 + 1000) / (z + 1000)
        ax.plot([source[0], z], [source[1], magnified], color="#98A2B3", linewidth=0.7)
    zfine = np.linspace(-100, 100, 300)
    ax.plot(zfine, 0.0009*zfine**2 + 0.08*zfine - 12, color="#2463A6", linewidth=2, label="MLP")
    ax.set(xlabel="MLP depth (mm)", ylabel="transverse coordinate (mm)",
           title="Evaluate MLP and apply fan-beam magnification")
    ax.legend(frameon=False)
    ax.grid(color="#EAECF0", linewidth=0.7)

    ax = axes[1]
    for value in np.arange(-2, 3):
        ax.axvline(value, color="#D0D5DD")
        ax.axhline(value, color="#D0D5DD")
    point = (0.63, -0.37)
    nearest = (1, 0)
    ax.scatter(*point, s=85, color="#C8553D", label="magnified MLP sample")
    ax.scatter(*nearest, s=85, marker="s", color="#2463A6", label="nearest lattice bin")
    ax.add_patch(FancyArrowPatch(point, nearest, arrowstyle="->", mutation_scale=14, color="#344054"))
    ax.text(-1.85, 1.55, r"accumulate full $b_p$ and count", fontsize=11)
    ax.text(-1.85, 1.15, r"$g_j=\sum_{p\in j}b_p/n_j$", fontsize=15)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), xlabel="transverse bin", ylabel="thin y bin",
           title="Current DDB uses nearest-lattice accumulation")
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower left")
    save(fig, output)


def read_ddb(path: Path) -> np.memmap:
    header = rsp_metrics.header(path)
    size = tuple(int(v) for v in header["DimSize"].split())
    if size != (500, 2, 500):
        raise ValueError(f"unexpected DDB size {size}: {path}")
    return np.memmap(path.parent / header["ElementDataFile"], dtype="<f4", mode="r", shape=(500, 2, 500))


def ddb_actual_figure(output: Path, projection_dir: Path) -> None:
    origin, spacing = -124.75, 0.5
    ddb0 = np.asarray(read_ddb(projection_dir / "proj0000.mhd")).mean(axis=1)
    depths = (-75.0, 0.0, 75.0)
    indices = [int(round((d - origin) / spacing)) for d in depths]
    stacks = np.empty((3, 720, 500), dtype=np.float32)
    for run_id in range(720):
        raw = read_ddb(projection_dir / f"proj{run_id:04d}.mhd")
        for panel, index in enumerate(indices):
            stacks[panel, run_id] = np.asarray(raw[index]).mean(axis=0)
    extent_map = [origin, origin + 499*spacing, origin, origin + 499*spacing]
    extent_sino = [origin, origin + 499*spacing, 359.5, 0.0]
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 5.8), constrained_layout=True)
    shown = axes[0].imshow(ddb0, origin="lower", extent=extent_map, aspect="auto", cmap="viridis", vmin=0, vmax=220)
    axes[0].set(xlabel="transverse DDB coordinate (mm)", ylabel="MLP depth (mm)", title="Projection angle 0 deg")
    for ax, image, depth in zip(axes[1:], stacks, depths):
        shown = ax.imshow(image, origin="upper", extent=extent_sino, aspect="auto", cmap="viridis", vmin=0, vmax=220)
        ax.set(xlabel="transverse DDB coordinate (mm)", ylabel="projection angle (deg)", title=f"depth {depth:+.0f} mm")
    fig.colorbar(shown, ax=axes, fraction=0.018, pad=0.015, label="mean WEPL (mm)")
    fig.suptitle("Actual depth-indexed DDB data: one projection and three sinograms", fontsize=15, weight="bold")
    save(fig, output)


def fdk_pipeline_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.0), constrained_layout=True)
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [(0.2, "DDB stack"), (3.15, "geometry\nweight"), (6.1, "1-D Ramp\nfilter"), (9.05, "DDB backprojection\n+ perspective weight")]
    for i, (x, label) in enumerate(boxes):
        ax.add_patch(Rectangle((x, 1.15), 2.35, 1.65, facecolor="#EAF2FA", edgecolor="#344054"))
        ax.text(x+1.175, 1.98, label, ha="center", va="center", weight="bold")
        if i < len(boxes)-1:
            ax.add_patch(FancyArrowPatch((x+2.38, 1.98), (boxes[i+1][0]-0.08, 1.98), arrowstyle="-|>", mutation_scale=14))
    ax.text(0.2, 3.45, "no-Hann DDB-FDK operator chain", fontsize=14, weight="bold")
    ax.text(0.2, 0.45, "Backprojection interpolates both transverse and MLP-depth coordinates.", fontsize=9.3, color="#475467")
    frequency = np.linspace(-1, 1, 1001)
    ramp = np.abs(frequency)
    hann = ramp * 0.5 * (1 + np.cos(np.pi * frequency))
    axes[1].plot(frequency, ramp, linewidth=2.2, label="Ramp, Hann = 0 (used)")
    axes[1].plot(frequency, hann, "--", linewidth=1.8, label="Ramp with Hann window (reference)")
    axes[1].set(xlabel="normalized detector frequency", ylabel="filter magnitude", title=r"$H(\omega)=|\omega|$")
    axes[1].grid(color="#EAECF0", linewidth=0.7)
    axes[1].legend(frameon=False)
    save(fig, output)


def analytic_dimensions_figure(output: Path) -> None:
    """Explain the DDB-FDK axis changes without implying 500 independent FBP runs."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), constrained_layout=True)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("1. Two equivalent views of the input", weight="bold")
    ax.add_patch(Rectangle((0.5, 5.9), 8.9, 2.2, facecolor="#EAF2FA", edgecolor="#344054", linewidth=1.5))
    ax.text(4.95, 7.0, "angle-major storage\n720 files x [u=500, v=2, d=500]", ha="center", va="center", fontsize=11, weight="bold")
    ax.add_patch(FancyArrowPatch((4.95, 5.8), (4.95, 4.55), arrowstyle="<|-|>", mutation_scale=16, color="#667085"))
    ax.text(5.35, 5.15, "re-index only", va="center", fontsize=9, color="#667085")
    ax.add_patch(Rectangle((0.5, 2.1), 8.9, 2.2, facecolor="#ECFDF3", edgecolor="#027A48", linewidth=1.5))
    ax.text(4.95, 3.2, "depth-major interpretation\n500 depth-indexed sinograms\neach [theta=720, u=500], for each v", ha="center", va="center", fontsize=10.0, weight="bold")
    ax.text(4.95, 0.8, "No reconstruction has happened yet; the same 4-D samples are just viewed with different fixed axes.", ha="center", va="center", fontsize=9, color="#475467", wrap=True)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("2. Processing one angle (low-memory mode)", weight="bold")
    labels = [
        (8.15, "extract angle theta\n[500, 2, 500]"),
        (5.95, "geometry weight\n[500, 2, 500]"),
        (3.75, "Ramp along u\n1000 lines x length 500"),
        (1.55, "filtered DDB\n[500, 2, 500]"),
    ]
    for i, (y, label) in enumerate(labels):
        ax.add_patch(Rectangle((1.2, y-0.7), 7.6, 1.35, facecolor="#FFF7E8" if i == 2 else "#F2F4F7", edgecolor="#344054"))
        ax.text(5.0, y, label, ha="center", va="center", fontsize=10.5, weight="bold")
        if i < len(labels)-1:
            ax.add_patch(FancyArrowPatch((5.0, y-0.72), (5.0, labels[i+1][0]+0.73), arrowstyle="-|>", mutation_scale=14))
    ax.text(5.0, 0.35, "Ramp does not filter across theta or MLP depth d.", ha="center", fontsize=9.5, color="#B54708")

    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("3. Backproject into one 2-D accumulator", weight="bold")
    ax.add_patch(Rectangle((0.5, 7.2), 9.0, 1.55, facecolor="#EAF2FA", edgecolor="#344054"))
    ax.text(5.0, 7.98, "current filtered angle: [u=500, v=2, d=500]", ha="center", va="center", weight="bold")
    ax.add_patch(FancyArrowPatch((5.0, 7.15), (5.0, 5.9), arrowstyle="-|>", mutation_scale=15))
    ax.add_patch(Rectangle((0.5, 4.2), 9.0, 1.65, facecolor="#F4EBFF", edgecolor="#6941C6"))
    ax.text(5.0, 5.03, "for every output pixel r:\ncompute (u_theta(r), v_theta(r), d_theta(r))\ninterpolate, then multiply by c_theta(r)^2", ha="center", va="center", fontsize=9.3, weight="bold")
    ax.add_patch(FancyArrowPatch((5.0, 4.15), (5.0, 2.95), arrowstyle="-|>", mutation_scale=15))
    ax.add_patch(Rectangle((1.4, 1.3), 7.2, 1.6, facecolor="#ECFDF3", edgecolor="#027A48", linewidth=1.5))
    ax.text(5.0, 2.1, "accumulator x += angle contribution\n[2100, 1, 2100]", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(5.0, 0.55, "Repeat for 720 angles; the array size remains [2100, 1, 2100].", ha="center", fontsize=9.5, color="#475467")
    fig.suptitle("How 500 depth-indexed sinograms become one 2-D RSP image", fontsize=16, weight="bold")
    save(fig, output)


def uvd_mapping_figure(output: Path) -> None:
    """Visual explanation of mapping one reconstruction pixel to DDB (u, v, d)."""
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8), constrained_layout=True)

    # Panel 1: perspective geometry in the current rotated scanner coordinates.
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-0.55, 1.65)
    ax.axis("off")
    ax.set_title("1. Rotate pixel into the current view", weight="bold")
    source = np.array([0.0, 1.42])
    center = np.array([0.0, 0.42])
    pixel = np.array([0.48, 0.76])
    detector_y = 0.31
    alpha = (detector_y-source[1])/(pixel[1]-source[1])
    detector_hit = source + alpha*(pixel-source)
    ax.plot([-1.25, 1.25], [detector_y, detector_y], color="#344054", linewidth=2)
    ax.text(-1.23, detector_y-0.09, "DDB transverse plane", fontsize=9, va="top")
    ax.plot([source[0], detector_hit[0]], [source[1], detector_hit[1]], color="#1570EF", linewidth=2.4)
    ax.plot([0, 0], [-0.25, 1.55], "--", color="#98A2B3", linewidth=1.2)
    ax.scatter(*source, s=65, color="#D92D20", zorder=3)
    ax.scatter(*center, s=45, color="#101828", zorder=3)
    ax.scatter(*pixel, s=70, color="#7F56D9", zorder=3)
    ax.scatter(*detector_hit, s=65, color="#039855", zorder=3)
    ax.text(source[0]+0.06, source[1], "source S", va="center", fontsize=10)
    ax.text(center[0]-0.08, center[1]-0.12, "O", fontsize=10)
    ax.text(pixel[0]+0.06, pixel[1]+0.02, "pixel P", fontsize=10, weight="bold")
    ax.text(detector_hit[0]+0.05, detector_hit[1]-0.12, r"$u_{\rm phys}$", fontsize=11, color="#027A48")
    ax.add_patch(FancyArrowPatch((0, 0.02), (0, 1.2), arrowstyle="-|>", mutation_scale=13, color="#475467"))
    ax.text(0.05, 1.04, r"beam depth $\zeta_\theta$", fontsize=10, rotation=90, va="center")
    ax.add_patch(FancyArrowPatch((0, 0.02), (0.92, 0.02), arrowstyle="-|>", mutation_scale=13, color="#475467"))
    ax.text(0.55, -0.08, r"transverse $t_\theta$", fontsize=10, ha="center")
    ax.plot([0, pixel[0]], [pixel[1], pixel[1]], ":", color="#7F56D9")
    ax.plot([pixel[0], pixel[0]], [0.02, pixel[1]], ":", color="#7F56D9")
    ax.text(0.24, pixel[1]+0.04, r"$t_\theta$", color="#6941C6", ha="center")
    ax.text(pixel[0]+0.04, 0.40, r"$\zeta_\theta$", color="#6941C6", rotation=90)
    ax.text(0, -0.39, r"$u_{\rm phys}=\frac{D_{sd}}{D-\zeta_\theta}\,t_\theta$",
            ha="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.28", fc="#EFF8FF", ec="#84CAFF"))

    # Panel 2: physical coordinates to continuous DDB indices and interpolation.
    ax = axes[1]
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 5.8)
    ax.set_ylim(-0.8, 5.8)
    ax.set_title("2. Locate the continuous point in the DDB", weight="bold")
    for i in range(6):
        ax.plot([i, i], [0, 5], color="#D0D5DD", linewidth=1)
        ax.plot([0, 5], [i, i], color="#D0D5DD", linewidth=1)
    point = (3.35, 1.65)
    ax.scatter(*point, s=90, color="#7F56D9", zorder=5)
    ax.plot([point[0], point[0]], [0, point[1]], "--", color="#7F56D9")
    ax.plot([0, point[0]], [point[1], point[1]], "--", color="#7F56D9")
    ax.text(point[0]+0.12, point[1]+0.14, r"$(i_u,i_d)$", fontsize=11, weight="bold", color="#6941C6")
    corners = [(3,1), (4,1), (3,2), (4,2)]
    for x, y in corners:
        ax.scatter(x, y, s=55, facecolor="#ECFDF3", edgecolor="#027A48", zorder=4)
    ax.add_patch(Rectangle((3, 1), 1, 1, fill=False, edgecolor="#039855", linewidth=2.0))
    ax.set_xlabel(r"continuous transverse index $i_u$")
    ax.set_ylabel(r"continuous depth index $i_d$")
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.text(2.5, -0.62, r"$i_u=(u_{\rm phys}-u_0)/\Delta u$", ha="center", fontsize=10)
    ax.text(-0.67, 2.5, r"$i_d=(-\zeta_\theta-d_0)/\Delta d$", ha="center",
            va="center", rotation=90, fontsize=10)
    ax.text(2.5, 5.35, "Interpolate neighboring DDB samples", ha="center",
            fontsize=10, color="#027A48", weight="bold")
    ax.text(2.5, 4.82, r"2-D case: $i_v=0.5$ between the two v layers",
            ha="center", fontsize=9.5, color="#475467")

    # Panel 3: same pixel at two projection angles.
    ax = axes[2]
    ax.set_xlim(90, 410)
    ax.set_ylim(90, 410)
    ax.set_aspect("equal")
    ax.set_title("3. The lookup moves when the angle changes", weight="bold")
    ax.add_patch(Rectangle((100, 100), 300, 300, facecolor="#F9FAFB", edgecolor="#667085", linewidth=1.5))
    ax.axvline(249.5, color="#D0D5DD", linewidth=1)
    ax.axhline(249.5, color="#D0D5DD", linewidth=1)
    p0 = (360.5, 249.5)
    p90 = (249.5, 149.5)
    ax.scatter(*p0, s=100, color="#1570EF", label=r"$\theta=0^\circ$", zorder=4)
    ax.scatter(*p90, s=100, color="#F79009", label=r"$\theta=90^\circ$", zorder=4)
    ax.add_patch(FancyArrowPatch(p0, p90, arrowstyle="-|>", mutation_scale=16,
                                 color="#7F56D9", connectionstyle="arc3,rad=-0.18", linewidth=2))
    ax.text(p0[0]-4, p0[1]+15, "(360.5, 249.5)", ha="right", color="#175CD3", fontsize=10)
    ax.text(p90[0]+10, p90[1]-2, "(249.5, 149.5)", va="center", color="#B54708", fontsize=10)
    ax.set_xlabel(r"transverse index $i_u$")
    ax.set_ylabel(r"MLP-depth index $i_d$")
    ax.legend(frameon=False, loc="upper left")
    ax.text(250, 108, "same physical pixel P = (50, 0, 0) mm", ha="center", fontsize=9.5)
    fig.suptitle("How one reconstruction pixel selects (u, v, d) in the filtered DDB",
                 fontsize=16, weight="bold")
    save(fig, output)


def radial_mean(image: np.ndarray, x: np.ndarray, z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    xx, zz = np.meshgrid(x, z)
    radius = np.hypot(xx, zz).ravel()
    index = np.digitize(radius, edges) - 1
    valid = (index >= 0) & (index < len(edges)-1)
    sums = np.bincount(index[valid], weights=image.ravel()[valid], minlength=len(edges)-1)
    counts = np.bincount(index[valid], minlength=len(edges)-1)
    return sums / np.maximum(counts, 1)


def analytic_actual_figure(output: Path, truth_path: Path, analytic_path: Path) -> None:
    truth, x, z, _ = rsp_metrics.read_mhd(truth_path)
    analytic, axx, azz, _ = rsp_metrics.read_mhd(analytic_path)
    if not (np.array_equal(x, axx) and np.array_equal(z, azz)):
        raise ValueError("analytic and truth grids differ")
    extent = [x[0], x[-1], z[0], z[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.8), constrained_layout=True)
    axes[0].imshow(truth, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=2.2)
    axes[1].imshow(analytic, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=2.2)
    error = axes[2].imshow(analytic-truth, origin="lower", extent=extent, cmap="coolwarm", vmin=-0.2, vmax=0.2)
    for ax, title in zip(axes[:3], ["200 MeV RSP truth", "DDB no-Hann reconstruction", "reconstruction - truth"]):
        ax.set(xlabel="x (mm)", ylabel="z (mm)", title=title, aspect="equal")
    fig.colorbar(error, ax=axes[2], fraction=0.046, label="RSP error")
    edges = np.arange(85, 135.1, 0.25)
    centers = 0.5*(edges[:-1]+edges[1:])
    axes[3].plot(centers, radial_mean(truth, x, z, edges), label="truth", linewidth=2)
    axes[3].plot(centers, radial_mean(analytic, x, z, edges), label="no-Hann", linewidth=2)
    axes[3].axvline(100, color="#98A2B3", linestyle="--", label="support boundary")
    axes[3].set(xlabel="radius (mm)", ylabel="azimuthal mean RSP", title="Boundary and exterior ring")
    axes[3].grid(color="#EAECF0", linewidth=0.7)
    axes[3].legend(frameon=False)
    fig.suptitle("results0716 analytic reconstruction (FDK time: 181.37 s)", fontsize=15, weight="bold")
    save(fig, output)


def iterative_operator_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    ax = axes[0]
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor="#D0D5DD"))
    point = (2.35, 2.65)
    weights = {(2, 2): (1-.35)*(1-.65), (3, 2): .35*(1-.65), (2, 3): (1-.35)*.65, (3, 3): .35*.65}
    for (i, j), value in weights.items():
        ax.add_patch(Rectangle((i, j), 1, 1, facecolor=plt.cm.Blues(0.25+0.7*value), edgecolor="#2463A6"))
        ax.text(i+.5, j+.5, f"{value:.3f}", ha="center", va="center", fontsize=9)
    z = np.linspace(0, 5, 200)
    ax.plot(z, 0.12*(z-2.5)**2+2.05, color="#C8553D", linewidth=2.2, label="sampled MLP")
    ax.scatter(*point, color="#101828", s=45, zorder=4)
    ax.set(xlim=(0, 5), ylim=(0, 5), xlabel="x pixel coordinate", ylabel="z pixel coordinate",
           title="Four bilinear weights per 0.1 mm path sample", aspect="equal")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.axis("off")
    ax.text(0.03, .85, r"Forward:  $(Ax)_p=\sum_j a_{pj}x_j$", fontsize=16)
    ax.text(0.03, .65, r"Residual:  $r_p=b_p-(Ax)_p$", fontsize=16)
    ax.text(0.03, .45, r"Transpose:  $(A^Tr)_j=\sum_p a_{pj}r_p$", fontsize=16)
    ax.text(0.03, .23, r"Adjoint test:  $\langle Ax,y\rangle=\langle x,A^Ty\rangle$", fontsize=16)
    ax.text(0.03, .06, "The same pixel indices and weights are reused in both directions.", fontsize=10, color="#475467")
    ax.set_title("Strictly paired list-mode operator", fontsize=13)
    save(fig, output)


def iterative_epochs_figure(output: Path, truth_path: Path, recon_dir: Path) -> None:
    truth, x, z, _ = rsp_metrics.read_mhd(truth_path)
    paths = [recon_dir/"initial.mhd", recon_dir/"epoch_01.mhd", recon_dir/"epoch_02.mhd", recon_dir/"epoch_03.mhd"]
    images = [rsp_metrics.read_mhd(path)[0] for path in paths]
    step = 4
    extent = [x[0], x[-1], z[0], z[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.7), constrained_layout=True)
    titles = ["Initial", "Epoch 1", "Epoch 2", "Epoch 3"]
    for col, (title, image) in enumerate(zip(titles, images)):
        axes[0, col].imshow(image[::step, ::step], origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=2.2)
        err = axes[1, col].imshow((image-truth)[::step, ::step], origin="lower", extent=extent, cmap="coolwarm", vmin=-.12, vmax=.12)
        axes[0, col].set_title(title)
        axes[1, col].set_title(f"{title} - truth")
        for ax in axes[:, col]:
            ax.set(xlabel="x (mm)", ylabel="z (mm)", aspect="equal")
    fig.colorbar(err, ax=axes[1], fraction=.018, pad=.01, label="RSP error")
    fig.suptitle("Actual GPU MLP OS-SART checkpoints", fontsize=15, weight="bold")
    save(fig, output)


def huber_figure(output: Path, regularization_csv: Path) -> None:
    rows = read_csv(regularization_csv)
    delta = float(rows[0]["huber_delta"])
    weight = float(rows[0]["weight"])
    t = np.linspace(0, .02, 500)
    huber = np.where(t <= delta, t*t/(2*delta), t-delta/2)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), constrained_layout=True)
    axes[0].plot(t, t, label="TV", linewidth=2)
    axes[0].plot(t, huber, label=f"Huber-TV, delta={delta:g}", linewidth=2)
    axes[0].plot(t, t*t/(2*delta), "--", label="quadratic continuation", linewidth=1.3)
    axes[0].axvline(delta, color="#98A2B3", linestyle=":")
    axes[0].set(xlabel=r"gradient magnitude $t$ (RSP/pixel)", ylabel=r"penalty $\phi(t)$", title="Edge-preserving Huber penalty")
    axes[0].legend(frameon=False)
    epochs = np.array([int(row["epoch"]) for row in rows])
    before = np.array([float(row["regularization_value_before"]) for row in rows])
    after = np.array([float(row["regularization_value_after"]) for row in rows])
    axes[1].plot(epochs, before, "o-", linewidth=2, label="before proximal step")
    axes[1].plot(epochs, after, "s-", linewidth=2, label="after proximal step")
    axes[1].set(xticks=epochs, xlabel="epoch", ylabel="weighted Huber-TV value",
                title=f"Actual regularization, beta={weight:g}")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(color="#EAECF0", linewidth=.7)
    save(fig, output)


def iterative_runtime_figure(output: Path, metric_csv: Path, history_csv: Path) -> None:
    metrics = read_csv(metric_csv)
    history = read_csv(history_csv)
    epochs = np.array([int(row["epoch"]) for row in metrics])
    image_rmse = np.array([float(row["phantom_rsp_rmse"]) for row in metrics])
    water_std = np.array([float(row["water_std"]) for row in metrics])
    subset = np.arange(1, len(history)+1)
    residual = np.array([float(row["residual_rmse_mm"]) for row in history])
    update = np.array([float(row["update_max_abs"]) for row in history])
    elapsed = np.array([float(row["elapsed_seconds"]) for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    axes[0,0].plot(epochs, image_rmse, "o-", label="phantom RSP RMSE")
    axes[0,0].plot(epochs, water_std, "s-", label="water std")
    axes[0,0].set(xticks=epochs, xlabel="checkpoint", ylabel="RSP metric", title="Image-domain convergence")
    axes[0,0].legend(frameon=False)
    axes[0,1].plot(subset, residual, linewidth=1.8)
    axes[0,1].axvline(18.5, color="#98A2B3", linestyle="--")
    axes[0,1].axvline(36.5, color="#98A2B3", linestyle="--")
    axes[0,1].set(xlabel="ordered-subset update", ylabel="online WEPL RMSE (mm)", title="Training residual before each update")
    axes[1,0].plot(subset, update, color="#C8553D", linewidth=1.8)
    axes[1,0].set(xlabel="ordered-subset update", ylabel="maximum absolute update", title="Update magnitude")
    axes[1,1].plot(subset, np.cumsum(elapsed)/60, color="#287A5A", linewidth=2)
    axes[1,1].set(xlabel="ordered-subset update", ylabel="cumulative subset time (min)", title="Actual GPU projection time")
    for ax in axes.ravel():
        ax.grid(color="#EAECF0", linewidth=.7)
    fig.suptitle("results0716 iterative runtime diagnostics", fontsize=15, weight="bold")
    save(fig, output)


def validation_figure(output: Path, by_angle_csv: Path, global_csv: Path) -> None:
    angle_rows = read_csv(by_angle_csv)
    global_rows = read_csv(global_csv)
    names = [row["checkpoint"] for row in global_rows]
    labels = ["Analytic", "Initial", "Epoch 1", "Epoch 2", "Epoch 3"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.1), constrained_layout=True)
    for name, label in zip(names, labels):
        rows = [row for row in angle_rows if row["checkpoint"] == name]
        axes[0].plot([float(row["angle_deg"]) for row in rows],
                     [float(row["wepl_rmse_mm"]) for row in rows],
                     linewidth=1.2, label=label)
    axes[0].set(xlabel="projection angle (deg)", ylabel="fixed-subset WEPL RMSE (mm)",
                title="Per-angle forward-model residual")
    axes[0].legend(frameon=False, ncol=2)
    values = [float(row["wepl_rmse_mm"]) for row in global_rows]
    bars = axes[1].bar(labels, values, color=["#667085", "#98A2B3", "#7AA6C2", "#4B87AF", "#2463A6"])
    axes[1].set(ylabel="measurement-weighted WEPL RMSE (mm)", title="24.43 M fixed-subset protons")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(min(values)-.015, max(values)+.012)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+.001, f"{value:.4f}", ha="center", fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", color="#EAECF0", linewidth=.7)
    fig.suptitle("Fixed 10% subset residual (not independent for the existing full-data reconstruction)", fontsize=14, weight="bold")
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="0716")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    output = args.output_dir or CODE_ROOT / "principle_assets"
    output.mkdir(parents=True, exist_ok=True)
    existing = [output/name for name in ASSET_NAMES if (output/name).exists()]
    if existing and not args.force:
        print(f"regenerating {len(existing)} existing principle assets", flush=True)

    preprocessing = path_for(config, "preprocessing_data")
    reconstruction = path_for(config, "reconstruction_data")
    preprocessing_qc = CODE_ROOT / "preprocessing" / "qc" / f"results{args.experiment}"
    iterative_qc = CODE_ROOT / "iterative_reconstruction" / "qc" / f"results{args.experiment}"
    evaluation = CODE_ROOT / "evaluation" / "baselines" / f"results{args.experiment}"
    pairing = read_json(preprocessing_qc/"pairing_summary.json")
    filtering = read_json(preprocessing_qc/"filtering_summary.json")

    tasks = [
        (pipeline_figure, (output/ASSET_NAMES[0], {"pairs": pairing["total_primary_pairs"], "filtered": filtering["output_pairs"]})),
        (pairing_planes_figure, (output/ASSET_NAMES[1],)),
        (filtering_figure, (output/ASSET_NAMES[2], preprocessing/"pairs"/"pairs0000.mhd", preprocessing/"pairs_filtered"/"pairs0000.mhd")),
        (bethe_bloch_figure, (output/ASSET_NAMES[3], preprocessing/"pairs_filtered"/"pairs0000.mhd")),
        (mlp_figure, (output/ASSET_NAMES[4], preprocessing/"pairs_filtered"/"pairs0000.mhd")),
        (ddb_binning_figure, (output/ASSET_NAMES[5],)),
        (ddb_actual_figure, (output/ASSET_NAMES[6], preprocessing/"projections_ddb")),
        (fdk_pipeline_figure, (output/ASSET_NAMES[7],)),
        (analytic_actual_figure, (output/ASSET_NAMES[8], reconstruction/"analytic"/"truth"/"truth_rsp_200mev.mhd", reconstruction/"analytic"/"recon"/"recon_ddb_nohann.mhd")),
        (iterative_operator_figure, (output/ASSET_NAMES[9],)),
        (iterative_epochs_figure, (output/ASSET_NAMES[10], reconstruction/"analytic"/"truth"/"truth_rsp_200mev.mhd", reconstruction/"iterative"/"recon")),
        (huber_figure, (output/ASSET_NAMES[11], iterative_qc/"regularization_history.csv")),
        (iterative_runtime_figure, (output/ASSET_NAMES[12], iterative_qc/"rsp_metrics.csv", iterative_qc/"iteration_history.csv")),
        (validation_figure, (output/ASSET_NAMES[13], evaluation/"validation_wepl_by_angle.csv", evaluation/"validation_wepl.csv")),
        (analytic_dimensions_figure, (output/ASSET_NAMES[14],)),
        (uvd_mapping_figure, (output/ASSET_NAMES[15],)),
    ]
    for index, (function, arguments) in enumerate(tasks, 1):
        print(f"principle figure {index:02d}/{len(tasks)}: {arguments[0].name}", flush=True)
        function(*arguments)

    from PIL import Image
    records = []
    for name in ASSET_NAMES:
        path = output/name
        with Image.open(path) as image:
            width, height = image.size
        if width < 1200 or height < 650:
            raise RuntimeError(f"principle asset is too small: {name} = {width}x{height}")
        records.append({"name": name, "bytes": path.stat().st_size, "width": width,
                        "height": height, "sha256": sha256(path)})
    manifest = {
        "status": "PASS",
        "experiment": str(args.experiment),
        "asset_count": len(records),
        "read_only_inputs": True,
        "assets": records,
    }
    (output/"asset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
