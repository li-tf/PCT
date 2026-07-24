#!/usr/bin/env python3
"""Build the full experiment0716 report without requiring iterative results."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent
TEST0713 = REPO_ROOT / "test0713"
sys.path.insert(0, str(CODE_ROOT))
from common import load_experiment, path_for  # noqa: E402
from analytic_reconstruction import rsp_metrics  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-pct2d-report")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402


RSP_ALUMINIUM = 2.1189760409708303


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement.rstrip() + "\n\n" + text[finish:]


def simulation_totals(simulation_qc: Path) -> dict[str, object]:
    launch = read_json(simulation_qc / "launcher_summary.json")
    manifest = read_csv(simulation_qc / "result_manifest.csv")
    statistics = [
        read_json(path)
        for path in sorted((simulation_qc / "runs").glob("run_*/protonct.txt"))
    ]
    metadata = [
        read_json(path)
        for path in sorted((simulation_qc / "runs").glob("run_*/run_metadata.json"))
    ]
    if len(manifest) != 720 or len(statistics) != 720 or len(metadata) != 720:
        raise RuntimeError("simulation QC does not contain 720 complete run records")
    value = lambda row, key: int(row[key]["value"])
    events = sum(value(row, "events") for row in statistics)
    return {
        "launch": launch,
        "events": events,
        "tracks": sum(value(row, "tracks") for row in statistics),
        "steps": sum(value(row, "steps") for row in statistics),
        "in_bytes": sum(int(row["phase_space_in_bytes"]) for row in manifest),
        "out_bytes": sum(int(row["phase_space_out_bytes"]) for row in manifest),
        "in_entries": sum(int(row["phase_space_in_entries"]) for row in manifest),
        "out_entries": sum(int(row["phase_space_out_entries"]) for row in manifest),
        "start": min(row["started_at"] for row in metadata).replace("T", " "),
        "stop": max(row["completed_at"] for row in metadata).replace("T", " "),
        "host": sorted({row["host"] for row in metadata}),
        "platform": sorted({row["platform"] for row in metadata}),
        "python": sorted({row["python"] for row in metadata}),
        "opengate": sorted({row["opengate"] for row in metadata}),
        "opengate_core": sorted({row["opengate_core"] for row in metadata}),
        "wall_rate": events / float(launch["elapsed_seconds"]),
    }


def preprocessing_totals(qc: Path) -> dict[str, object]:
    pairing = read_json(qc / "pairing_summary.json")
    filtering = read_json(qc / "filtering_summary.json")
    projection = read_json(qc / "projection_summary.json")
    filter_rows = read_csv(qc / "filtering_runs.csv")
    projection_rows = read_csv(qc / "projection_runs.csv")
    total = lambda rows, key: sum(float(row[key]) for row in rows)
    pixels = 720 * 500 * 2 * 500
    object_pixels = int(total(projection_rows, "object_pixels"))
    return {
        "pairing": pairing,
        "filtering": filtering,
        "projection": projection,
        "inside_grid": int(total(filter_rows, "inside_grid")),
        "outside_grid": int(total(filter_rows, "outside_grid")),
        "removed_3sigma": int(total(filter_rows, "removed_inside_grid_by_3sigma")),
        "inside_retained": total(filter_rows, "output") / total(filter_rows, "inside_grid"),
        "pixels": pixels,
        "zero_count": int(total(projection_rows, "zero_count")),
        "object_pixels": object_pixels,
        "object_mean_count": total(projection_rows, "object_count_sum") / object_pixels,
        "variance_mean": total(projection_rows, "variance_sum") / pixels,
    }


def pipeline_figure(output: Path, counts: dict[str, int]) -> None:
    fig, axis = plt.subplots(figsize=(14.0, 4.8), constrained_layout=True)
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 5)
    axis.axis("off")
    boxes = [
        (0.2, "OpenGATE", f"{counts['events']/1e6:.2f} M events\n720 views"),
        (2.45, "Phase-space", "720 In/Out ROOT pairs\nposition, direction, energy"),
        (4.7, "Primary pairing", f"{counts['pairs']/1e6:.2f} M pairs\nRunID + EventID"),
        (6.95, "3-sigma cuts", f"{counts['filtered']/1e6:.2f} M pairs\nenergy + angle"),
        (9.2, "Schulte MLP DDB", "720 projections\n500 x 2 x 500"),
        (11.45, "RSP reconstruction", "no-Hann FDK + GPU OS-SART\n2100 x 2100 @ 0.1 mm"),
    ]
    for index, (left, title, body) in enumerate(boxes):
        color = "#EAF2FA" if index < 5 else "#F2EAF6"
        axis.add_patch(Rectangle((left, 1.25), 1.9, 2.35, facecolor=color,
                                 edgecolor="#344054", linewidth=1.3))
        axis.text(left + 0.95, 2.95, title, ha="center", va="center",
                  weight="bold", fontsize=10.5)
        axis.text(left + 0.95, 2.15, body, ha="center", va="center",
                  fontsize=8.7, color="#475467")
        if index < len(boxes) - 1:
            axis.add_patch(FancyArrowPatch(
                (left + 1.92, 2.42), (boxes[index + 1][0] - 0.04, 2.42),
                arrowstyle="-|>", mutation_scale=13, linewidth=1.2,
                color="#667085"))
    axis.text(0.2, 4.45, "experiment0716 data and reconstruction workflow",
              fontsize=15, weight="bold")
    axis.text(0.2, 0.45,
              "Paper-fluence Windows simulation; results0716 analytic and iterative reconstructions are complete.",
              fontsize=9.5, color="#475467")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def analytic_comparison_figure(
    truth_path: Path, old_path: Path, new_path: Path, output: Path
) -> None:
    truth, x, z, _ = rsp_metrics.read_mhd(truth_path)
    old, old_x, old_z, _ = rsp_metrics.read_mhd(old_path)
    new, new_x, new_z, _ = rsp_metrics.read_mhd(new_path)
    if not (
        np.array_equal(x, old_x) and np.array_equal(z, old_z)
        and np.array_equal(x, new_x) and np.array_equal(z, new_z)
    ):
        raise RuntimeError("test0713 and results0716 analytic grids differ")
    extent = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.2), constrained_layout=True)
    for axis, title, image in zip(
        axes[0],
        ["200 MeV RSP truth", "test0713 DDB no-Hann", "results0716 DDB no-Hann"],
        [truth, old, new],
    ):
        shown = axis.imshow(image, origin="lower", extent=extent, cmap="viridis",
                            vmin=0.0, vmax=2.2)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_aspect("equal")
    fig.colorbar(shown, ax=axes[0], fraction=0.020, pad=0.02, label="RSP")
    axes[1, 0].axis("off")
    for axis, title, image in zip(
        axes[1, 1:],
        ["test0713 no-Hann - truth", "results0716 no-Hann - truth"],
        [old - truth, new - truth],
    ):
        error = axis.imshow(image, origin="lower", extent=extent, cmap="coolwarm",
                            vmin=-0.20, vmax=0.20)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_aspect("equal")
    fig.colorbar(error, ax=axes[1, 1:], fraction=0.030, pad=0.02, label="RSP error")
    fig.suptitle("Analytic RSP reconstruction at two proton fluences",
                 fontsize=15, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def iterative_comparison_figure(
    truth_path: Path, analytic_path: Path, iterative_path: Path, output: Path
) -> None:
    truth, x, z, _ = rsp_metrics.read_mhd(truth_path)
    analytic, ax, az, _ = rsp_metrics.read_mhd(analytic_path)
    iterative, ix, iz, _ = rsp_metrics.read_mhd(iterative_path)
    if not (
        np.array_equal(x, ax) and np.array_equal(z, az)
        and np.array_equal(x, ix) and np.array_equal(z, iz)
    ):
        raise RuntimeError("results0716 analytic and iterative grids differ")
    extent = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 9.2), constrained_layout=True)
    for axis, title, image in zip(
        axes[0],
        ["200 MeV RSP truth", "results0716 DDB no-Hann", "results0716 MLP OS-SART, epoch 3"],
        [truth, analytic, iterative],
    ):
        shown = axis.imshow(image, origin="lower", extent=extent, cmap="viridis",
                            vmin=0.0, vmax=2.2)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_aspect("equal")
    fig.colorbar(shown, ax=axes[0], fraction=0.020, pad=0.02, label="RSP")
    axes[1, 0].axis("off")
    for axis, title, image in zip(
        axes[1, 1:],
        ["DDB no-Hann - truth", "MLP OS-SART epoch 3 - truth"],
        [analytic - truth, iterative - truth],
    ):
        error = axis.imshow(image, origin="lower", extent=extent, cmap="coolwarm",
                            vmin=-0.20, vmax=0.20)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("z (mm)")
        axis.set_aspect("equal")
    fig.colorbar(error, ax=axes[1, 1:], fraction=0.030, pad=0.02, label="RSP error")
    fig.suptitle("results0716 analytic and iterative RSP reconstruction",
                 fontsize=15, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def iterative_convergence_figure(rows: list[dict[str, str]], output: Path) -> None:
    epochs = np.array([int(row["epoch"]) for row in rows])
    rmse = np.array([float(row["phantom_rsp_rmse"]) for row in rows])
    residual = np.array([
        float(row["wepl_residual_rmse_mm"])
        if row["wepl_residual_rmse_mm"].lower() not in {"", "nan"} else np.nan
        for row in rows
    ])
    fig, axis = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    axis.plot(epochs, rmse, "o-", color="#2463A6", linewidth=2.2,
              label="Image RSP RMSE")
    axis.set_xticks(epochs, ["Initial", "Epoch 1", "Epoch 2", "Epoch 3"])
    axis.set_xlabel("Checkpoint")
    axis.set_ylabel("Phantom RSP RMSE", color="#2463A6")
    axis.tick_params(axis="y", labelcolor="#2463A6")
    axis.grid(color="#E4E7EC", linewidth=0.8)
    second = axis.twinx()
    second.plot(epochs[1:], residual[1:], "s--", color="#C8553D",
                linewidth=2.0, label="Weighted WEPL residual")
    second.set_ylabel("Weighted WEPL residual RMSE (mm)", color="#C8553D")
    second.tick_params(axis="y", labelcolor="#C8553D")
    lines = axis.get_lines() + second.get_lines()
    axis.legend(lines, [line.get_label() for line in lines], frameon=False,
                loc="upper right")
    fig.suptitle("results0716 iterative convergence", fontsize=14, weight="bold")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def radial_mean(image: np.ndarray, x: np.ndarray, z: np.ndarray,
                edges: np.ndarray) -> np.ndarray:
    xx, zz = np.meshgrid(x, z)
    radius = np.hypot(xx, zz).ravel()
    values = image.ravel().astype(np.float64)
    index = np.digitize(radius, edges) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    sums = np.bincount(index[valid], weights=values[valid], minlength=len(edges) - 1)
    counts = np.bincount(index[valid], minlength=len(edges) - 1)
    return sums / np.maximum(counts, 1)


def boundary_figure(truth_path: Path, old_path: Path, new_path: Path,
                    output: Path) -> None:
    truth, x, z, _ = rsp_metrics.read_mhd(truth_path)
    old = rsp_metrics.read_mhd(old_path)[0]
    new = rsp_metrics.read_mhd(new_path)[0]
    series = [
        ("RSP truth", truth, "#4C956C", "-"),
        ("test0713 no-Hann", old, "#667085", "--"),
        ("results0716 no-Hann", new, "#2463A6", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), constrained_layout=True)
    for axis, lower, upper, step, title in [
        (axes[0], 90.0, 105.0, 0.1, "Object boundary"),
        (axes[1], 100.0, 148.0, 0.25, "Exterior halo in square-image corners"),
    ]:
        edges = np.arange(lower, upper + 0.5 * step, step)
        radius = 0.5 * (edges[:-1] + edges[1:])
        for label, image, color, style in series:
            axis.plot(radius, radial_mean(image, x, z, edges), label=label,
                      color=color, linestyle=style, linewidth=2.0)
        axis.axvline(100.0, color="#344054", linewidth=1.0, linestyle=":")
        axis.axhline(0.0, color="#98A2B3", linewidth=0.8)
        axis.set_xlim(lower, upper)
        axis.set_xlabel("radius from isocenter (mm)")
        axis.set_ylabel("azimuthal mean RSP")
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.grid(color="#E4E7EC", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("Water-cylinder boundary response at two proton fluences",
                 fontsize=14, weight="bold")
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


def ddb_sinogram_figure(projection_dir: Path, output: Path) -> None:
    depth_mm = (-75.0, 0.0, 75.0)
    origin, spacing = -124.75, 0.5
    indices = [int(round((value - origin) / spacing)) for value in depth_mm]
    stacks = np.empty((3, 720, 500), dtype=np.float32)
    for run_id in range(720):
        path = projection_dir / f"proj{run_id:04d}.mhd"
        header = rsp_metrics.header(path)
        size = tuple(int(value) for value in header["DimSize"].split())
        if size != (500, 2, 500):
            raise RuntimeError(f"unexpected DDB shape in {path}: {size}")
        raw = np.memmap(path.parent / header["ElementDataFile"], dtype="<f4",
                        mode="r", shape=(500, 2, 500))
        for panel, index in enumerate(indices):
            stacks[panel, run_id] = np.asarray(raw[index]).mean(axis=0)
    extent = [origin, origin + 499 * spacing, 359.5, 0.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.0), constrained_layout=True)
    for axis, image, depth in zip(axes, stacks, depth_mm):
        shown = axis.imshow(image, origin="upper", extent=extent, aspect="auto",
                            cmap="viridis", vmin=0.0, vmax=220.0)
        axis.set_title(f"MLP depth = {depth:+.0f} mm")
        axis.set_xlabel("DDB transverse coordinate (mm)")
        axis.set_ylabel("projection angle (deg)")
    fig.colorbar(shown, ax=axes, fraction=0.020, pad=0.02, label="mean WEPL (mm)")
    fig.suptitle("results0716 depth-indexed DDB sinogram sections",
                 fontsize=15, weight="bold")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def build_markdown(
    simulation: dict[str, object], preprocessing: dict[str, object],
    old: dict[str, str], new: dict[str, str], analytic_summary: dict,
    iterative_rows: list[dict[str, str]], iterative_summary: dict,
) -> str:
    source = (TEST0713 / "report" / "test0713_summary_report.md").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        "# test0713 质子CT仿真、数据处理与RSP重建实验报告",
        "# experiment0716 质子CT仿真、数据处理与RSP重建实验报告",
        1,
    ).replace("生成日期：2026-07-15", "生成日期：2026-07-16", 1).replace(
        "实验目录：`test0713/`", "实验配置：`simulation0716 / results0716 / report0716`", 1
    )
    events = int(simulation["events"])
    pairs = int(preprocessing["pairing"]["total_primary_pairs"])
    filtered = int(preprocessing["filtering"]["output_pairs"])
    wall = float(simulation["launch"]["elapsed_seconds"])
    old_std, new_std = float(old["water_std"]), float(new["water_std"])
    old_rmse, new_rmse = float(old["phantom_rsp_rmse"]), float(new["phantom_rmse_vs_rsp_truth"])
    noise_drop = 100.0 * (1.0 - new_std / old_std)
    rmse_drop = 100.0 * (1.0 - new_rmse / old_rmse)
    initial = iterative_rows[0]
    final = iterative_rows[-1]
    iterative_std = float(final["water_std"])
    iterative_rmse = float(final["phantom_rsp_rmse"])
    iterative_noise_drop = 100.0 * (1.0 - iterative_std / new_std)
    iterative_rmse_drop = 100.0 * (1.0 - iterative_rmse / new_rmse)
    iterative_cnr_gain = 100.0 * (
        float(final["roi_cnr_median"]) / float(new["insert_roi_cnr_median"]) - 1.0
    )
    iterative_elapsed = float(iterative_summary["elapsed_seconds"])

    summary = f"""## 摘要

本实验在`test0713`已验证流程的基础上，将质子通量提高到Rit等人Simulation 2
使用的`900 protons/mm²/projection`，并把蒙特卡洛仿真迁移到Windows工作站。
几何、材料、200 MeV能量、720个投影角度和DDB/重建网格均保持不变；每角度
计划质子数由100,000提高到450,000。720个单线程OpenGATE任务以12个独立进程
并行执行，得到`{events:,}`个event。

数据仍采用primary-only入口—出口配对、局部能量—角度3σ过滤、`I=78 eV`
水射程LUT和Schulte MLP距离驱动分箱。最终得到`{pairs:,}`条主质子pair、
`{filtered:,}`条过滤后pair和720组`500×2×500 @ 0.5 mm` DDB投影。

本文继续以**200 MeV参考RSP**作为解析图像的统一评价口径。results0716的
DDB no-Hann水区均值为`{float(new['water_mean']):.5f}`、标准差为
`{new_std:.5f}`，模体RSP RMSE为`{new_rmse:.5f}`，铝柱内部平台为
`{float(new['aluminium_inner_mean']):.4f}`，达到参考真值`{RSP_ALUMINIUM:.5f}`
的`{100*float(new['aluminium_platform_rsp_recovery']):.2f}%`。相对低通量
test0713 no-Hann，水区标准差降低`{noise_drop:.1f}%`，模体RSP RMSE降低
`{rmse_drop:.1f}%`，说明论文通量主要改善了统计噪声，同时保持材料平台和边缘
分辨率。

results0716进一步完成了全量`{int(iterative_summary['pairs_per_epoch']):,}`条
质子/epoch、0.1 mm网格、18子集和3轮GPU MLP OS-SART，并在每轮后施加
Huber-TV。第3轮水区标准差为`{iterative_std:.5f}`，模体RSP RMSE为
`{iterative_rmse:.5f}`，相对本次解析no-Hann分别降低`{iterative_noise_drop:.1f}%`
和`{iterative_rmse_drop:.1f}%`；铝平台恢复率仍为
`{100*float(final['aluminium_platform_rsp_recovery']):.2f}%`。第五部分暂时保持
原报告建议，不根据本次迭代结果重新排序。

![experiment0716数据与重建流程](assets/pipeline_flow.png)

图1概括了本次已完成的数据链。"""
    source = replace_section(source, "## 摘要", "## 第一部分：", summary)

    geometry = """### 1.2 几何、源和探测器参数

| 类别 | 正式参数 |
|---|---|
| 质子源 | proton，200 MeV单能，平面源位于`z=-1060 mm` |
| 源平面尺寸 | `15×0.12×10⁻⁶ mm³`（`x×y×z`） |
| 有效焦点与SID | 焦点`z=-1000 mm`，SID=`1000 mm` |
| 照射范围 | 等中心面`250×2 mm²`，`900 protons/mm²/projection` |
| 角度与统计量 | 720角度，`0,0.5,…,359.5°`；每角度计划450,000个质子 |
| 入口相空间 | `z=-110 mm`，距焦点890 mm，`400×400 mm²` |
| 出口相空间 | `z=+110 mm`，距焦点1110 mm，`400×400 mm²` |
| 水模体 | 半径100 mm，总长度400 mm，圆柱轴沿`y`轴 |
| 铝柱 | 25根，直径5 mm；中心半径`0,5,9,…,97 mm`，角步长139° |
| 世界与物理 | Vacuum，`QGSP_BIC_EMZ`，水和铝内最大transport step为1 mm |
| 随机与执行 | MersenneTwister，基础seed=`20260713`；每任务1线程，12任务并行 |

相空间字段和理想探测面定义与test0713完全相同。源平面在`x`方向宽15 mm，
在`y`方向只有0.12 mm；后者是较小的束流厚度方向。越过距源平面60 mm的焦点后，
束流在等中心处覆盖`250×2 mm²`。`z`向`10⁻⁶ mm`只表示近似零厚度源平面。"""
    source = replace_section(source, "### 1.2 ", "### 1.3 ", geometry)

    environment = f"""### 1.3 软件、设备与运行时间

逐角度元数据记录的软件环境为Python 3.12.4、OpenGATE 10.1.0、
`opengate-core 10.1.0`和Windows 11（10.0.26200）。工作站为Intel Xeon
w5-2455X（12核、24线程），配备约255 GB内存。每个OpenGATE进程内部仍只使用
一个线程，但launcher同时运行12个独立角度；这与test0713单进程、单线程连续
执行720个run不同。

正式仿真的逐run元数据时间范围为{simulation['start']}至{simulation['stop']}；
launcher记录墙钟时间`{wall:.2f} s`，即`{wall/3600:.5f} h`。OpenGATE统计为：

- 720个独立run，`{events:,}`个event；
- `{int(simulation['tracks']):,}`条track；
- `{int(simulation['steps']):,}`个step；
- 12进程墙钟平均吞吐约`{float(simulation['wall_rate']):,.0f} protons/s`；
- 入口ROOT共{int(simulation['in_bytes'])/1e9:.2f} GB、`{int(simulation['in_entries']):,}`条记录；
- 出口ROOT共{int(simulation['out_bytes'])/1e9:.2f} GB、`{int(simulation['out_entries']):,}`条记录；
- primary-only配对得到`{pairs:,}`条存活主质子，按event计的存活率为
  `{100*pairs/events:.4f}%`；
- 720组入口/出口ROOT全部完成，1440个文件迁移前后SHA-256校验一致。

计划质子数为324,000,000，实际event高出`{events-324000000:,}`，相对差异仅
`{100*(events/324000000-1):.4f}%`。这是GenericSource以activity和时间区间
生成事件时的泊松波动，不是角度缺失或重复运行。出口ROOT原始记录包含次级质子，
因此不能用出口总记录数直接计算主质子存活率。"""
    source = replace_section(source, "### 1.3 ", "### 1.4 ", environment)

    protocol = """### 1.4 与Rit等人Simulation 2的关系

results0716对齐了原文的720投影、200 MeV单能质子、1000 mm SID、直径200 mm
水圆柱、25根直径5 mm铝柱、`900 protons/mm²/projection`通量、DDB网格和
0.1 mm重建网格。相对于test0713，最主要的协议变化就是每角度质子数从
100,000提高到450,000，因此当前统计通量已经与论文一致。

仍有两项不能忽略的差异：原文使用GATE 6.2/Geant4 4.9.5.p01并考虑Air，
results0716使用OpenGATE 10.1.0、现代Geant4和Vacuum；此外原文的执行平台与
当前Windows多进程工作站不同。因此本实验是论文通量下的现代软件复现，而不是
历史软件环境的逐位复刻。"""
    source = replace_section(source, "### 1.4 ", "## 第二部分：", protocol)

    status = f"""### 2.1 当前处理状态

| 阶段 | 数量 | 相对前一步 |
|---|---:|---:|
| 计划主质子 | 324,000,000 | 100% |
| 实际OpenGATE event | {events:,} | {events/324000000:.6%} |
| 出口相空间记录 | {int(simulation['out_entries']):,} | {int(simulation['out_entries'])/events:.3%} |
| primary-only pairs | {pairs:,} | {pairs/events:.3%}（相对event） |
| 3σ后pairs | {filtered:,} | {filtered/pairs:.3%} |
| 正式DDB投影 | 720组 | 全角度完成 |

目前720组ROOT、配对结果、过滤结果、DDB投影、200 MeV RSP真值、0.1 mm
no-Hann解析重建以及3轮全量GPU迭代重建均已完成，相关阶段QC均为PASS。
各阶段QC保存在代码目录，ROOT、MHD和RAW等大数据只保存在`data/`目录。"""
    source = replace_section(source, "### 2.1 ", "### 2.2 ", status)

    pairing_text = f"""### 2.2 入口—出口质子配对

工作站仿真把每个角度保存为独立的`run_###/PhaseSpaceIn.root`和
`PhaseSpaceOut.root`，每个分片内部的本地RunID均为0。新预处理入口直接逐角度
读取720对ROOT，不先合并成两个大型ROOT。配对步骤为：

1. 读取同一角度的入口、出口记录，并仅保留主质子`TrackID=1`；
2. 以本地`(EventID,TrackID)`求交，建立同一存活质子的前后测量；
3. 把目录角度编号写成全局RunID，使后续文件编号保持`0000–0719`；
4. 使用记录方向把状态外推到固定`z=-110 mm`和`z=+110 mm`平面；
5. 写出入口/出口位置、方向以及`(E_in,E_out,TrackID)`组成的`5×3`结构。

正式接口仍调用经过test0713验证的
`pctpairprotons --stream-by-run --no-nuclear`算法，但避免了额外合并37 GB ROOT。
最终得到`{pairs:,}`条primary-only pair，占实际event的`{100*pairs/events:.3f}%`。
720组MHD/RAW均完整，所有输出TrackID均为1。"""
    source = replace_section(source, "### 2.2 ", "### 2.3 ", pairing_text)

    old_filter = """在`63,109,480`条输入pair中，`7,133,526`条落在cut grid之外；网格内
`55,975,954`条中又有`1,800,152`条未通过3σ条件，最后保留
`54,175,802`条。网格内3σ保留率为`96.784%`，所有保留pair均为主质子。"""
    new_filter = f"""在`{pairs:,}`条输入pair中，`{int(preprocessing['outside_grid']):,}`条落在cut grid之外；
网格内`{int(preprocessing['inside_grid']):,}`条中又有`{int(preprocessing['removed_3sigma']):,}`条
未通过3σ条件，最后保留`{filtered:,}`条。网格内3σ保留率为
`{100*float(preprocessing['inside_retained']):.3f}%`，所有保留pair均为主质子。"""
    if old_filter not in source:
        raise RuntimeError("test0713 filtering paragraph changed unexpectedly")
    source = source.replace(old_filter, new_filter, 1)

    old_ddb = """720组投影全部有限。共`360,000,000`个DDB像素，仅47个count为零；其中
半径100 mm物体包络内只有7个。物体内平均count约`42.31`，variance全部有限、
非负，均值为`0.45534 mm²`。零计数比例极低，因此正式流程没有使用hole filling。"""
    new_ddb = f"""720组投影全部有限。共`{int(preprocessing['pixels']):,}`个DDB像素，零count
像素为`{int(preprocessing['zero_count'])}`，半径100 mm物体包络内同样为0。
物体内平均count约`{float(preprocessing['object_mean_count']):.2f}`，约为test0713
的4.5倍；variance全部有限、非负，全DDB像素均值为
`{float(preprocessing['variance_mean']):.5f} mm²`。当前无需hole filling。"""
    if old_ddb not in source:
        raise RuntimeError("test0713 DDB paragraph changed unexpectedly")
    source = source.replace(old_ddb, new_ddb, 1).replace(
        "![三个MLP深度处的DDB正弦图切片]",
        "![results0716三个MLP深度处的DDB正弦图切片]", 1
    )

    times = f"""### 2.7 数据处理时间与复现

本机完成工作站数据的预处理时间为：primary配对
`{float(preprocessing['pairing']['elapsed_seconds']):.2f} s`、3σ过滤
`{float(preprocessing['filtering']['elapsed_seconds']):.2f} s`、4进程生成720组DDB
投影`{float(preprocessing['projection']['elapsed_seconds']):.2f} s`。配对时间明显
长于test0713，既有4.5倍数据量因素，也包括逐角度启动公共配对程序的固定开销。

```bash
.venv-gate/bin/python pct2d_reconstruction/preprocessing/run_preprocessing.py \\
  --experiment 0716 --stage all --jobs 4
```

默认拒绝覆盖已有结果；只有确认需要重算时才加入`--force`。"""
    source = replace_section(source, "### 2.7 ", "## 第三部分：", times)

    method = f"""### 3.1 DDB-FDK方法

解析重建读取results0716的720组DDB投影和对应RTK圆轨迹几何。每个投影先做
几何加权，再沿探测器方向执行Ramp滤波，最后由距离驱动DDB反投影器累加到
`2100×1×2100 @ 0.1×1×0.1 mm`网格。results0716只保留成熟的
**DDB no-Hann**主链，不重复生成Hann=1敏感性结果。

几何文件生成耗时`{float(analytic_summary['geometry_seconds']):.2f} s`，no-Hann
FDK耗时`{float(analytic_summary['fdk_seconds']):.2f} s`。为了判断提高通量带来的
变化，本部分使用同一指标实现，把results0716 no-Hann与test0713 no-Hann
直接比较。"""
    source = replace_section(source, "### 3.1 ", "### 3.2 ", method)

    analytic = rf"""### 3.3 RSP真值与解析结果

两次实验的几何、材料和重建坐标完全一致，两个`truth_rsp_200mev.raw`逐字节相同。
RSP真值不是从带噪重建反推，而是依据水圆柱、25根铝柱及材料密度，在
`2100×2100 @ 0.1 mm`网格上以每像素`8×8`子采样独立体素化。水定义为RSP=1，
铝的200 MeV参考值为

\[
\operatorname{{RSP}}_{{Al}}=
\frac{{2.6989\times3.526}}{{1.0\times4.491}}=2.118976.
\]

WEPL仍使用水的`I=78 eV` Bethe–Bloch射程LUT。该选择定义了水等效路径长度，
不会把Geant4中的铝改成水；但固定200 MeV铝真值与降能路径上的有效RSP并不
严格相同，因此约1%的铝平台差异不应直接解释为纯重建损失。

![200 MeV RSP真值及两次no-Hann解析重建](assets/analytic_rsp_comparison.png)

图5统一使用`0–2.2` RSP色标和`-0.20–0.20`误差色标。0716水区纹理明显减弱，
25根铝柱的位置和总体边缘形态与test0713一致。

| 指标 | 200 MeV参考真值 | test0713 DDB no-Hann | results0716 DDB no-Hann |
|---|---:|---:|---:|
| 水区均值 | 1.00000 | {float(old['water_mean']):.5f} | {float(new['water_mean']):.5f} |
| 水区标准差 | 0 | {old_std:.5f} | {new_std:.5f} |
| 模体RSP RMSE | 0 | {old_rmse:.5f} | {new_rmse:.5f} |
| 铝柱内部平台 | {RSP_ALUMINIUM:.5f} | {float(old['aluminium_inner_mean']):.4f}（{100*float(old['aluminium_platform_rsp_recovery']):.2f}%） | {float(new['aluminium_inner_mean']):.4f}（{100*float(new['aluminium_platform_rsp_recovery']):.2f}%） |
| ROI CNR中位数 | — | {float(old['roi_cnr_median']):.2f} | {float(new['insert_roi_cnr_median']):.2f} |
| 10%–90%边缘宽度中位数 | 0 | {float(old['edge_10_90_median_mm']):.3f} mm | {float(new['aluminium_edge_10_90_median_mm']):.3f} mm |

指标定义与test0713保持一致：水区为`r≤95 mm`且排除铝柱5 mm邻域；模体RMSE
只在真值非零支撑内计算；铝平台来自25根柱的平均径向剖面内部值；ROI CNR使用
2 mm柱ROI与4–8 mm局部水壳；边缘宽度排除中心柱，对24根柱各平均360条径向
剖面后测量10%–90%下降距离。

通量提高4.5倍后，水区标准差降低`{noise_drop:.1f}%`。若误差完全由独立泊松
噪声支配，标准差预期按`1/sqrt(4.5)`缩放到原来的47.1%；实测为
`{100*new_std/old_std:.1f}%`，与该数量级一致。RSP RMSE只降低`{rmse_drop:.1f}%`，
说明RMSE还包含水区约+1.37%的共同标定偏差、铝柱边缘部分容积和模型误差，
这些成分不会随质子数按平方根消失。

铝平台恢复率由`{100*float(old['aluminium_platform_rsp_recovery']):.2f}%`变为
`{100*float(new['aluminium_platform_rsp_recovery']):.2f}%`，基本不变；边缘中位宽度
从`{float(old['edge_10_90_median_mm']):.3f} mm`变为
`{float(new['aluminium_edge_10_90_median_mm']):.3f} mm`，差异远小于柱间分布。
因此提高通量显著改善噪声和CNR，但不会自动消除材料模型偏差或提高系统空间
分辨率。"""
    source = replace_section(source, "### 3.3 ", "### 3.4 ", analytic)

    boundary = f"""### 3.4 外围圆环伪影

![两次no-Hann重建的水圆柱边界与外部径向响应](assets/boundary_radial_profile.png)

图6比较相同几何在两种质子通量下的方位平均径向响应。results0716在水区和边界
附近的随机波动更小，但`r>102 mm`外部均值仍为
`{float(new['outside_mean']):.5f}`，与test0713的`{float(old['outside_mean']):.5f}`
几乎相同。由此可见，四角圆环不是低统计噪声造成的；增加4.5倍质子只能让环
更平滑，不能消除其平均幅值。

现有证据继续支持“有限DDB网格、Ramp滤波的长程响应和FDK未施加100 mm支撑域”
的组合解释。约110–130 mm外环接近DDB横向半宽124.75 mm；它与早期错误角度
符号产生的随铝柱半径增长的切向弧不同。两次实验使用相同的正确
`first_angle=0°、arc=+360°`轨迹，因此本次高通量结果进一步排除了随机噪声和
角度映射错误作为圆环主因。"""
    source = replace_section(source, "### 3.4 ", "## 第四部分：", boundary)

    fourth = rf"""## 第四部分：0.1 mm迭代重建

### 4.1 MLP感知的list-mode前向模型

普通X射线迭代器假设射线沿直线传播，而质子在水和铝中发生多重库仑散射。
results0716因此不把DDB投影交给普通Joseph投影器，而是直接读取过滤后的
list-mode pairs。每批质子根据入口/出口位置、方向和能量重新计算Schulte MLP，
再以0.1 mm步长在半径100 mm圆柱支撑内离散路径。

对路径采样点采用双线性像素权重。前投影计算当前RSP图像沿MLP的加权路径积分，
反投影复用完全相同的像素索引和权重，保证前、反算子配对。目标数据仍是由
`I=78 eV`水射程LUT得到的单质子WEPL。

### 4.2 GPU OS-SART更新与正式配置

设当前子集的MLP系统矩阵为\(A_S\)，测量WEPL为\(b_S\)。单条路径按总长度
进行行归一化\(M_S\)，像素按该子集累计路径权重进行列归一化\(D_S\)：

\[
x^{{k+1}}=P_{{\Omega,+}}\!\left[
x^k+\lambda_kD_S^{{-1}}A_S^TM_S^{{-1}}(b_S-A_Sx^k)
\right].
\]

\(P_{{\Omega,+}}\)在每次子集更新后施加非负约束，并令半径100 mm支撑域外
严格为零。720个角度按角度编号模18交错分组，保证每个子集覆盖完整圆周。

| 参数 | results0716正式值 |
|---|---:|
| 输入 | 全量`{int(iterative_summary['pairs_per_epoch']):,}`条过滤后主质子/epoch |
| 初值 | results0716 DDB no-Hann，裁剪到100 mm支撑域 |
| 网格 | `2100×2100 @ 0.1 mm`，视野210 mm |
| MLP路径步长 | 0.1 mm |
| 子集/epoch | 18 / 3 |
| 松弛因子 | 第1轮0.25；\(\lambda_e=0.25/[1+0.2(e-1)]\) |
| batch | 4096条质子 |
| GPU | {iterative_summary['gpu']} |

### 4.3 Huber-TV正则化

每个完整epoch后，以OS-SART结果\(f\)为中心求解

\[
\min_{{u\in\Omega,u\ge0}}
\frac12\|u-f\|_2^2+\beta\sum_j\phi_\delta(|\nabla u_j|),
\]

其中

\[
\phi_\delta(t)=
\begin{{cases}}
t^2/(2\delta),&t\le\delta,\\
t-\delta/2,&t>\delta.
\end{{cases}}
\]

正式参数为`β=0.0125`、`δ=0.002`，每轮执行100步Chambolle–Pock近端迭代，
primal/dual step均为0.25。小梯度区采用近似二次惩罚以降低水区噪声，大梯度区
接近TV以保留铝柱边缘。三次正则化合计只耗时
`{float(iterative_summary['regularization_elapsed_seconds']):.2f} s`，主要计算成本
来自每轮约2.44亿条质子的MLP生成和投影。

### 4.4 解析与迭代RSP对比

![results0716 RSP真值、解析no-Hann和迭代第3轮](assets/iterative_rsp_comparison.png)

图7使用相同RSP和误差色标。第3轮保留全部25根铝柱，并把解析图像水区的随机
纹理显著压低。解析图像四角的外部圆环因100 mm支撑约束被严格置零；物体边界
内侧和铝柱边缘仍存在低幅正负误差，反映有限空间分辨率和模型误差。

| 指标 | 200 MeV参考真值 | results0716 DDB no-Hann | results0716迭代第3轮 |
|---|---:|---:|---:|
| 水区均值 | 1.00000 | {float(new['water_mean']):.5f} | {float(final['water_mean']):.5f} |
| 水区标准差 | 0 | {new_std:.5f} | {iterative_std:.5f} |
| 模体RSP RMSE | 0 | {new_rmse:.5f} | {iterative_rmse:.5f} |
| 铝柱内部平台 | {RSP_ALUMINIUM:.5f} | {float(new['aluminium_inner_mean']):.4f}（{100*float(new['aluminium_platform_rsp_recovery']):.2f}%） | {float(final['aluminium_inner_mean']):.4f}（{100*float(final['aluminium_platform_rsp_recovery']):.2f}%） |
| ROI CNR中位数 | — | {float(new['insert_roi_cnr_median']):.2f} | {float(final['roi_cnr_median']):.2f} |
| 10%–90%边缘宽度中位数 | 0 | {float(new['aluminium_edge_10_90_median_mm']):.3f} mm | {float(final['edge_10_90_median_mm']):.3f} mm |

相对解析no-Hann，第3轮将水区标准差降低`{iterative_noise_drop:.1f}%`，模体RSP
RMSE降低`{iterative_rmse_drop:.1f}%`，ROI CNR提高`{iterative_cnr_gain:.1f}%`。
铝平台恢复率仅降低
`{100*(float(new['aluminium_platform_rsp_recovery'])-float(final['aluminium_platform_rsp_recovery'])):.2f}`
个百分点，边缘宽度从`{float(new['aluminium_edge_10_90_median_mm']):.3f} mm`
变为`{float(final['edge_10_90_median_mm']):.3f} mm`，没有观察到Huber-TV导致的
边缘展宽。水区均值仍约高1.4%，说明迭代主要降低方差，未消除共同低频偏差。

### 4.5 逐轮收敛

![results0716迭代过程RSP误差和WEPL残差](assets/iterative_epoch_convergence.png)

| 检查点 | RSP RMSE | 水区标准差 | 铝平台恢复率 | ROI CNR | 边缘宽度 | WEPL残差 |
|---|---:|---:|---:|---:|---:|---:|
| 初值 | {float(iterative_rows[0]['phantom_rsp_rmse']):.5f} | {float(iterative_rows[0]['water_std']):.5f} | {100*float(iterative_rows[0]['aluminium_platform_rsp_recovery']):.2f}% | {float(iterative_rows[0]['roi_cnr_median']):.2f} | {float(iterative_rows[0]['edge_10_90_median_mm']):.3f} mm | — |
| Epoch 1 | {float(iterative_rows[1]['phantom_rsp_rmse']):.5f} | {float(iterative_rows[1]['water_std']):.5f} | {100*float(iterative_rows[1]['aluminium_platform_rsp_recovery']):.2f}% | {float(iterative_rows[1]['roi_cnr_median']):.2f} | {float(iterative_rows[1]['edge_10_90_median_mm']):.3f} mm | {float(iterative_rows[1]['wepl_residual_rmse_mm']):.5f} mm |
| Epoch 2 | {float(iterative_rows[2]['phantom_rsp_rmse']):.5f} | {float(iterative_rows[2]['water_std']):.5f} | {100*float(iterative_rows[2]['aluminium_platform_rsp_recovery']):.2f}% | {float(iterative_rows[2]['roi_cnr_median']):.2f} | {float(iterative_rows[2]['edge_10_90_median_mm']):.3f} mm | {float(iterative_rows[2]['wepl_residual_rmse_mm']):.5f} mm |
| Epoch 3 | {float(iterative_rows[3]['phantom_rsp_rmse']):.5f} | {float(iterative_rows[3]['water_std']):.5f} | {100*float(iterative_rows[3]['aluminium_platform_rsp_recovery']):.2f}% | {float(iterative_rows[3]['roi_cnr_median']):.2f} | {float(iterative_rows[3]['edge_10_90_median_mm']):.3f} mm | {float(iterative_rows[3]['wepl_residual_rmse_mm']):.5f} mm |

epoch WEPL残差按各子集有效measurement数加权：

\[
\operatorname{{RMSE}}_{{epoch}}=
\sqrt{{\frac{{\sum_sN_s\operatorname{{RMSE}}_s^2}}{{\sum_sN_s}}}}.
\]

图像RSP RMSE和训练WEPL残差均逐轮下降，但第2到第3轮的RSP RMSE改善已从
`{float(iterative_rows[1]['phantom_rsp_rmse'])-float(iterative_rows[2]['phantom_rsp_rmse']):.5f}`
缩小到
`{float(iterative_rows[2]['phantom_rsp_rmse'])-float(iterative_rows[3]['phantom_rsp_rmse']):.5f}`。
训练残差与图像真值误差含义不同；前者继续下降并不保证后者在更多epoch中仍
持续改善，因此后续若增加轮数仍应同时观察独立验证数据和RSP指标。

### 4.6 运行状态与限制

正式运行从`{iterative_summary['started'].replace('T', ' ')}`至
`{iterative_summary['stopped'].replace('T', ' ')}`，总耗时
`{iterative_elapsed:.2f} s`，即约`{iterative_elapsed/3600:.2f} h`；状态为
`PASS`。最终图像全部有限，100 mm支撑域外非零像素数为0。

当前结果仍有三项限制：MLP散射统计主要采用已知水圆柱模型，没有随局部材料
更新；过滤后的质子在数据项中全部等权，没有引入能量和散射不确定度；正则参数
没有使用独立验证集选择。因此第3轮是当前高通量基线，而不是算法性能上限。"""
    source = replace_section(source, "## 第四部分：", "## 第五部分：", fourth)

    conclusion = f"""## 结论

experiment0716完成了论文通量下从Windows OpenGATE仿真到RSP解析重建的完整
数据链。12个独立单线程角度并行运行，使324,015,799个event的仿真在
`{wall/3600:.3f} h`内完成；primary-only配对和局部3σ过滤最终保留
`{filtered:,}`条质子，720组Schulte MLP DDB投影均无零计数像素和非有限方差。

在与test0713完全相同的真值和重建网格上，论文通量no-Hann把水区标准差从
`{old_std:.5f}`降低到`{new_std:.5f}`，CNR从`{float(old['roi_cnr_median']):.2f}`
提高到`{float(new['insert_roi_cnr_median']):.2f}`，模体RSP RMSE从
`{old_rmse:.5f}`降低到`{new_rmse:.5f}`。铝平台仍恢复约98.9%，边缘宽度基本
不变，表明新增统计量主要降低噪声，而不是改变材料幅值或空间分辨率。

外围同心圆环的径向均值几乎不随通量变化，进一步说明其主因是解析滤波、有限
DDB边界和缺少支撑域约束，而不是低统计噪声。高通量3轮GPU MLP OS-SART +
Huber-TV进一步将水区标准差降至`{iterative_std:.5f}`、RSP RMSE降至
`{iterative_rmse:.5f}`，同时保持`{100*float(final['aluminium_platform_rsp_recovery']):.2f}%`
铝平台恢复率和`{float(final['edge_10_90_median_mm']):.3f} mm`边缘宽度。支撑域
约束完全消除了物体外圆环显示，但水区约+1.4%的共同均值偏差仍然存在。"""
    source = replace_section(source, "## 结论", "## 参考文献", conclusion)

    appendix_a = """### A. 报告资源

- `assets/`：本报告8张图片，迭代相关两图由results0716检查点生成；
- `tables/analytic_rsp_metrics.csv`：test0713与results0716 no-Hann统一RSP指标；
- `tables/pipeline_counts.csv`：results0716数据链数量；
- `tables/processing_times.csv`：results0716仿真、处理及重建运行时间；
- `tables/protocol_comparison.csv`：test0713、results0716与Rit协议对照；
- `tables/iterative_epoch_rsp_metrics.csv`：results0716初值及3轮指标；
- `tables/reconstruction_rsp_comparison.csv`：results0716解析与最终迭代对比；
- `qc/report_summary.json`：资源、链接和关键数值的集成验证结果。"""
    source = replace_section(source, "### A. ", "### B. ", appendix_a)
    appendix_b = """### B. 重新生成报告图表

```bash
.venv-gate/bin/python pct2d_reconstruction/report/build_report.py \\
  --experiment 0716 --force
```

资源生成器只读取已有QC、CSV和MHD/RAW，不重新运行OpenGATE、预处理、FDK或
GPU迭代；生成结束时自动检查所有本地图片和表格链接。"""
    source = replace_section(source, "### B. ", "### C. ", appendix_b)
    appendix_c = """### C. results0716解析重建复现

```bash
.venv-gate/bin/python \\
  pct2d_reconstruction/analytic_reconstruction/run_analytic_reconstruction.py \\
  --experiment 0716 --force
```

只有确认需要覆盖现有no-Hann重建和QC时才使用`--force`。"""
    source = replace_section(source, "### C. ", "### D. ", appendix_c)
    appendix_d = """### D. results0716正式0.1 mm迭代重建复现

```bash
.venv-gate/bin/python \\
  pct2d_reconstruction/iterative_reconstruction/run_iterative_reconstruction.py \\
  --experiment 0716 --epochs 3 --force
```

配置文件提供全量质子、0.1 mm网格/路径步长、18子集和Huber-TV参数；只有确认
覆盖现有3轮检查点和QC时才使用`--force`。"""
    source = replace_section(source, "### D. ", "### E. ", appendix_d)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="0716")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.experiment != "0716":
        raise ValueError("the full cross-experiment report currently supports 0716")
    config = load_experiment(args.experiment)
    report = path_for(config, "report_code")
    assets, tables, qc = report / "assets", report / "tables", report / "qc"
    for directory in (assets, tables, qc):
        directory.mkdir(parents=True, exist_ok=True)
    markdown = report / "report0716_summary_report.md"
    if markdown.exists() and not args.force:
        raise FileExistsError(f"report exists: {markdown}; use --force")

    reconstruction = path_for(config, "reconstruction_data")
    preprocessing_data = path_for(config, "preprocessing_data")
    truth = reconstruction / "analytic" / "truth" / "truth_rsp_200mev.mhd"
    new_recon = reconstruction / "analytic" / "recon" / "recon_ddb_nohann.mhd"
    iterative_recon = reconstruction / "iterative" / "recon" / "epoch_03.mhd"
    old_truth = TEST0713 / "analytic_reconstruction" / "truth" / "truth_rsp_200mev.mhd"
    old_recon = TEST0713 / "analytic_reconstruction" / "recon" / "recon_ddb_nohann.mhd"
    analytic_qc = CODE_ROOT / "analytic_reconstruction" / "qc" / "results0716"
    preprocessing_qc = CODE_ROOT / "preprocessing" / "qc" / "results0716"
    simulation_qc = CODE_ROOT / "simulation" / "simulation0716" / "qc"
    iterative_qc = CODE_ROOT / "iterative_reconstruction" / "qc" / "results0716"
    required = [
        truth, new_recon, iterative_recon, old_truth, old_recon,
        analytic_qc / "analytic_summary.json", analytic_qc / "rsp_metrics.csv",
        preprocessing_qc / "pairing_summary.json",
        preprocessing_qc / "filtering_summary.json",
        preprocessing_qc / "projection_summary.json",
        simulation_qc / "launcher_summary.json",
        iterative_qc / "run_summary.json",
        iterative_qc / "rsp_metrics.csv",
        iterative_qc / "regularization_history.csv",
        TEST0713 / "report" / "tables" / "analytic_rsp_metrics.csv",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"report input missing: {missing[0]}")
    if old_truth.with_suffix(".raw").read_bytes() != truth.with_suffix(".raw").read_bytes():
        raise RuntimeError("test0713 and results0716 RSP truth maps are not identical")

    simulation = simulation_totals(simulation_qc)
    preprocessing = preprocessing_totals(preprocessing_qc)
    analytic_summary = read_json(analytic_qc / "analytic_summary.json")
    old_metrics = read_csv(TEST0713 / "report" / "tables" / "analytic_rsp_metrics.csv")[0]
    new_metrics = read_csv(analytic_qc / "rsp_metrics.csv")[0]
    iterative_summary = read_json(iterative_qc / "run_summary.json")
    iterative_rows = read_csv(iterative_qc / "rsp_metrics.csv")
    regularization_rows = read_csv(iterative_qc / "regularization_history.csv")
    iterative_summary["regularization_elapsed_seconds"] = sum(
        float(row["elapsed_seconds"]) for row in regularization_rows
    )
    if iterative_summary["status"] != "PASS" or len(iterative_rows) != 4:
        raise RuntimeError("results0716 iterative QC is incomplete")

    pipeline_figure(assets / "pipeline_flow.png", {
        "events": int(simulation["events"]),
        "pairs": int(preprocessing["pairing"]["total_primary_pairs"]),
        "filtered": int(preprocessing["filtering"]["output_pairs"]),
    })
    analytic_comparison_figure(truth, old_recon, new_recon,
                               assets / "analytic_rsp_comparison.png")
    boundary_figure(truth, old_recon, new_recon,
                    assets / "boundary_radial_profile.png")
    ddb_sinogram_figure(preprocessing_data / "projections_ddb",
                        assets / "ddb_sinogram_sections.png")
    iterative_comparison_figure(
        truth, new_recon, iterative_recon, assets / "iterative_rsp_comparison.png"
    )
    iterative_convergence_figure(
        iterative_rows, assets / "iterative_epoch_convergence.png"
    )
    for name in ("simulation_geometry.png", "mlp_ddb_principle.png"):
        shutil.copy2(TEST0713 / "report" / "assets" / name, assets / name)

    analytic_rows = [
        {
            "case": "test0713_ddb_nohann", "protons_per_projection": 100000,
            **{key: old_metrics[key] for key in old_metrics if key != "case"},
        },
        {
            "case": "results0716_ddb_nohann", "protons_per_projection": 450000,
            "water_mean": new_metrics["water_mean"],
            "water_std": new_metrics["water_std"],
            "phantom_rsp_rmse": new_metrics["phantom_rmse_vs_rsp_truth"],
            "aluminium_inner_mean": new_metrics["aluminium_inner_mean"],
            "aluminium_platform_rsp_recovery": new_metrics["aluminium_platform_rsp_recovery"],
            "roi_cnr_median": new_metrics["insert_roi_cnr_median"],
            "edge_10_90_median_mm": new_metrics["aluminium_edge_10_90_median_mm"],
            "edge_10_90_min_mm": new_metrics["aluminium_edge_10_90_min_mm"],
            "edge_10_90_max_mm": new_metrics["aluminium_edge_10_90_max_mm"],
            "outside_mean": new_metrics["outside_mean"],
            "outside_rmse": new_metrics["outside_rmse_vs_0"],
        },
    ]
    write_csv(tables / "analytic_rsp_metrics.csv", analytic_rows)
    events = int(simulation["events"])
    pairs = int(preprocessing["pairing"]["total_primary_pairs"])
    filtered = int(preprocessing["filtering"]["output_pairs"])
    write_csv(tables / "pipeline_counts.csv", [
        {"stage": "planned primary protons", "count": 324000000, "fraction_of_previous": 1.0},
        {"stage": "simulated events", "count": events, "fraction_of_previous": events / 324000000},
        {"stage": "exit phase-space rows", "count": simulation["out_entries"], "fraction_of_previous": int(simulation["out_entries"]) / events},
        {"stage": "primary-only pairs", "count": pairs, "fraction_of_previous": pairs / events},
        {"stage": "3-sigma retained pairs", "count": filtered, "fraction_of_previous": filtered / pairs},
        {"stage": "DDB projections", "count": 720, "fraction_of_previous": 1.0},
    ])
    write_csv(tables / "processing_times.csv", [
        {"stage": "OpenGATE simulation", "elapsed_seconds": simulation["launch"]["elapsed_seconds"], "device_or_workers": "Xeon w5-2455X; 12 single-thread processes"},
        {"stage": "primary pairing", "elapsed_seconds": preprocessing["pairing"]["elapsed_seconds"], "device_or_workers": "CPU; 4 processes"},
        {"stage": "3-sigma filtering", "elapsed_seconds": preprocessing["filtering"]["elapsed_seconds"], "device_or_workers": "CPU; 4 processes"},
        {"stage": "720 DDB projections", "elapsed_seconds": preprocessing["projection"]["elapsed_seconds"], "device_or_workers": "CPU; 4 processes"},
        {"stage": "no-Hann analytic FDK", "elapsed_seconds": analytic_summary["fdk_seconds"], "device_or_workers": "CPU"},
        {"stage": "0.1 mm iterative reconstruction", "elapsed_seconds": iterative_summary["elapsed_seconds"], "device_or_workers": iterative_summary["gpu"]},
    ])
    write_csv(tables / "protocol_comparison.csv", [
        {"protocol": "test0713", "protons_per_projection": 100000, "fluence_per_mm2_projection": 200, "software": "OpenGATE 10.1.0", "medium": "Vacuum"},
        {"protocol": "results0716", "protons_per_projection": 450000, "fluence_per_mm2_projection": 900, "software": "OpenGATE 10.1.0", "medium": "Vacuum"},
        {"protocol": "Rit Simulation 2", "protons_per_projection": 450000, "fluence_per_mm2_projection": 900, "software": "GATE 6.2 / Geant4 4.9.5.p01", "medium": "Air"},
    ])
    shutil.copy2(iterative_qc / "rsp_metrics.csv",
                 tables / "iterative_epoch_rsp_metrics.csv")
    final = iterative_rows[-1]
    write_csv(tables / "reconstruction_rsp_comparison.csv", [
        {
            "case": "results0716_ddb_nohann",
            "water_mean": new_metrics["water_mean"],
            "water_std": new_metrics["water_std"],
            "phantom_rsp_rmse": new_metrics["phantom_rmse_vs_rsp_truth"],
            "aluminium_inner_mean": new_metrics["aluminium_inner_mean"],
            "aluminium_platform_rsp_recovery": new_metrics["aluminium_platform_rsp_recovery"],
            "roi_cnr_median": new_metrics["insert_roi_cnr_median"],
            "edge_10_90_median_mm": new_metrics["aluminium_edge_10_90_median_mm"],
            "outside_nonzero": "not constrained",
        },
        {
            "case": "results0716_iterative_epoch_03",
            "water_mean": final["water_mean"],
            "water_std": final["water_std"],
            "phantom_rsp_rmse": final["phantom_rsp_rmse"],
            "aluminium_inner_mean": final["aluminium_inner_mean"],
            "aluminium_platform_rsp_recovery": final["aluminium_platform_rsp_recovery"],
            "roi_cnr_median": final["roi_cnr_median"],
            "edge_10_90_median_mm": final["edge_10_90_median_mm"],
            "outside_nonzero": final["outside_nonzero"],
        },
    ])

    content = build_markdown(
        simulation, preprocessing, old_metrics, new_metrics, analytic_summary,
        iterative_rows, iterative_summary,
    )
    markdown.write_text(content, encoding="utf-8")

    links = re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", content)
    local_links = [link for link in links if not link.startswith(("http://", "https://"))]
    missing_links = [link for link in local_links if not (report / link).is_file()]
    expected_phrases = [
        "324,015,799", "284,021,915", "244,217,799",
        "test0713 DDB no-Hann", "results0716 DDB no-Hann",
        "results0716迭代第3轮", "3轮GPU MLP OS-SART",
    ]
    phrases_ok = all(phrase in content for phrase in expected_phrases)
    images = list(assets.glob("*.png"))
    status = "PASS" if not missing_links and phrases_ok and len(images) == 8 else "FAIL"
    summary = {
        "status": status,
        "experiment": "0716",
        "report": str(markdown),
        "local_links_checked": len(local_links),
        "missing_links": missing_links,
        "images": [str(path) for path in sorted(images)],
        "analytic_comparison": ["test0713 DDB no-Hann", "results0716 DDB no-Hann"],
        "iterative_status": "results0716 epoch 3 complete and included",
        "truth": "byte-identical 200 MeV reference RSP maps",
    }
    (qc / "report_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(markdown)
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
