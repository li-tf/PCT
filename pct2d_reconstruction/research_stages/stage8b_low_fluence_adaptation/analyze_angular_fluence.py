#!/usr/bin/env python3
"""Build reproducible diagnostics for the equal-total-fluence comparison."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


WINDOWS_CJK_FONT = Path("/mnt/c/Windows/Fonts/msyh.ttc")
if WINDOWS_CJK_FONT.is_file():
    font_manager.fontManager.addfont(WINDOWS_CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=WINDOWS_CJK_FONT
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False


REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "pct2d_reconstruction"
STAGE7B = CODE / "research_stages/stage7b_noise_robustness"
for directory in (CODE, STAGE7B, CODE / "iterative_reconstruction"):
    sys.path.insert(0, str(directory))

from mhd_io import read_image_2d  # noqa: E402
import run_stage7b  # noqa: E402


QC = Path(__file__).resolve().parent / "qc/angular_fluence_360x20"
ASSETS = QC / "assets"
RECON = REPO / "data/reconstruction_data/results0718_d1_air_tracker_full"
BASE_720 = RECON / "stage7c/combined_0p2mm_1pct/seed_20260730/f010/iterative/recon"
TEST_360 = RECON / "stage8b/angular_fluence/angles360_f0200/iterative/recon"


def metrics(path: Path) -> dict[str, float]:
    values = run_stage7b.image_metrics(path)
    return {
        "water_std": float(values["water_std"]),
        "phantom_rmse": float(values["phantom_rmse_vs_rsp_truth"]),
        "cnr": float(values["insert_cnr_median"]),
        "aluminium_rsp": float(values["insert_peak_mean"]),
    }


def write_epoch_metrics() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for label, root in (("720角度 × 10%", BASE_720), ("360角度 × 20%", TEST_360)):
        for epoch, filename in [(0, "initial.mhd")] + [
            (index, f"epoch_{index:02d}.mhd") for index in range(1, 6)
        ]:
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append({"configuration": label, "epoch": epoch, **metrics(path)})
    output = QC / "epoch_comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_epoch_metrics(rows: list[dict[str, float | int | str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    styles = {
        "720角度 × 10%": dict(color="#777777", marker="o", linestyle="--"),
        "360角度 × 20%": dict(color="#1677b8", marker="s", linestyle="-"),
    }
    for label, style in styles.items():
        selected = [row for row in rows if row["configuration"] == label]
        epochs = [int(row["epoch"]) for row in selected]
        axes[0].plot(epochs, [float(row["water_std"]) for row in selected], label=label, **style)
        axes[1].plot(epochs, [float(row["phantom_rmse"]) for row in selected], label=label, **style)
    for ax, ylabel in zip(axes, ("水区RSP标准差", "模体RSP RMSE")):
        ax.set_xlabel("重建阶段（0为DDB-FDK初值）")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(6))
        ax.grid(alpha=0.22)
    axes[0].legend(frameon=False)
    fig.suptitle("等总质子数条件下各重建阶段的指标变化")
    fig.savefig(ASSETS / "epoch_metric_comparison.png", dpi=220)
    plt.close(fig)


def plot_images() -> None:
    paths = [
        BASE_720 / "initial.mhd",
        TEST_360 / "initial.mhd",
        BASE_720 / "epoch_05.mhd",
        TEST_360 / "epoch_05.mhd",
    ]
    images = [np.asarray(read_image_2d(path)[0], dtype=np.float32) for path in paths]
    panels = [images[0], images[1], images[1] - images[0], images[2], images[3], images[3] - images[2]]
    titles = [
        "DDB-FDK：720角度 × 10%",
        "DDB-FDK：360角度 × 20%",
        "解析初值差值",
        "第5轮：720角度 × 10%",
        "第5轮：360角度 × 20%",
        "第5轮差值",
    ]
    extent = (-104.95, 104.95, -104.95, 104.95)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    rsp_handle = None
    diff_handle = None
    for index, (ax, image, title) in enumerate(zip(axes.flat, panels, titles)):
        if index % 3 == 2:
            diff_handle = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu_r", vmin=-0.20, vmax=0.20)
        else:
            rsp_handle = ax.imshow(image, origin="lower", extent=extent, cmap="viridis", vmin=0.90, vmax=2.20)
        ax.set_title(title)
        ax.set_xlabel("x（mm）")
        ax.set_ylabel("z（mm）")
        ax.set_aspect("equal")
    assert rsp_handle is not None and diff_handle is not None
    fig.colorbar(rsp_handle, ax=[axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], shrink=0.77, label="RSP")
    fig.colorbar(diff_handle, ax=axes[:, 2], shrink=0.77, label="RSP差值（360×20% − 720×10%）")
    fig.suptitle("相同总质子数在不同投影角度间重新分配的影响")
    fig.savefig(ASSETS / "image_and_difference_comparison.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = write_epoch_metrics()
    plot_epoch_metrics(rows)
    plot_images()
    print(f"Wrote diagnostics to {ASSETS}")


if __name__ == "__main__":
    main()
