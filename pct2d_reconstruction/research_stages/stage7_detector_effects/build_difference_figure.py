#!/usr/bin/env python3
"""Plot the three formal Stage-7 reconstructions minus the MLIC RSP truth."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CODE = REPO / "pct2d_reconstruction"
sys.path.insert(0, str(CODE / "iterative_reconstruction"))

from mhd_io import read_image_2d  # noqa: E402


def mlic_values() -> dict[str, float]:
    path = (
        CODE
        / "research_stages/stage6a_mlic_reference/qc/"
        "mlic_reference_200mev.csv"
    )
    with path.open(encoding="utf-8") as stream:
        return {
            row["material"]: float(row["mlic_rsp_200mev"])
            for row in csv.DictReader(stream)
        }


def truth_on_grid(reference: Path) -> tuple[np.ndarray, list[float], list[float]]:
    image, spacing, origin = read_image_2d(reference)
    definition = json.loads(
        (
            CODE
            / "simulation/simulation0716/truth_geometry_definition.json"
        ).read_text(encoding="utf-8")
    )
    geometry = definition["geometry"]
    values = mlic_values()
    x = origin[0] + np.arange(image.shape[1]) * spacing[0]
    z = origin[2] + np.arange(image.shape[0]) * spacing[2]
    xx, zz = np.meshgrid(x, z)
    truth = np.zeros_like(image, dtype=np.float32)
    truth[xx * xx + zz * zz <= float(geometry["phantom_radius_mm"]) ** 2] = (
        values["Water"]
    )
    radius2 = float(geometry["insert_radius_mm"]) ** 2
    for center in geometry["insert_centers_xz_mm"]:
        mask = (
            (xx - float(center["x"])) ** 2
            + (zz - float(center["z"])) ** 2
            <= radius2
        )
        truth[mask] = values["Aluminium"]
    return truth, spacing, origin


def main() -> None:
    root = (
        REPO
        / "data/reconstruction_data/results0718_d1_air_tracker_full/"
        "stage7/full"
    )
    cases = [
        ("ideal_reference", "Ideal reference"),
        ("continuous_hits", "Continuous Si hits"),
        ("energy_1pct", "0.2 mm + 1% Eout noise"),
    ]
    paths = [
        root / name / "iterative/recon/recon_iterative_gpu.mhd"
        for name, _ in cases
    ]
    truth, spacing, origin = truth_on_grid(paths[0])
    extent = (
        origin[0] - 0.5 * spacing[0],
        origin[0] + (truth.shape[1] - 0.5) * spacing[0],
        origin[2] - 0.5 * spacing[2],
        origin[2] + (truth.shape[0] - 0.5) * spacing[2],
    )
    differences = [
        np.asarray(read_image_2d(path)[0], dtype=np.float32) - truth
        for path in paths
    ]
    # Keep one common, symmetric scale.  ±0.15 RSP shows material and water
    # errors clearly; larger boundary discrepancies are intentionally clipped.
    limit = 0.15
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.25), sharex=True, sharey=True)
    shown = None
    for axis, difference, (_, title) in zip(axes, differences, cases):
        shown = axis.imshow(
            difference,
            extent=extent,
            origin="lower",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("z (mm)")
    fig.subplots_adjust(
        left=0.065, right=0.895, bottom=0.13, top=0.88, wspace=0.12
    )
    colorbar_axis = fig.add_axes([0.915, 0.18, 0.015, 0.66])
    colorbar = fig.colorbar(shown, cax=colorbar_axis)
    colorbar.set_label("RSP error (reconstruction - MLIC truth)")
    fig.suptitle("Stage 7 reconstruction error relative to MLIC RSP truth", y=0.995)
    output = HERE / "qc/assets/stage7_difference_vs_mlic_truth.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
