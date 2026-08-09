#!/usr/bin/env python3
"""Build reproducible, archive-safe figures for project_overview.md."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-project-overview")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np


_WINDOWS_CJK_FONT = Path("/mnt/c/Windows/Fonts/msyh.ttc")
if _WINDOWS_CJK_FONT.exists():
    font_manager.fontManager.addfont(_WINDOWS_CJK_FONT)
    _WINDOWS_CJK_BOLD = Path("/mnt/c/Windows/Fonts/msyhbd.ttc")
    if _WINDOWS_CJK_BOLD.exists():
        font_manager.fontManager.addfont(_WINDOWS_CJK_BOLD)
    _CJK_FAMILY = font_manager.FontProperties(fname=_WINDOWS_CJK_FONT).get_name()
else:
    _CJK_FAMILY = "Droid Sans Fallback"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_CJK_FAMILY, "DejaVu Sans"],
    "axes.unicode_minus": False,
})


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY_ROOT = CODE_ROOT.parent
ASSET_ROOT = HERE / "project_overview_assets"
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
        ("0", "冻结基线", "通过"),
        ("1", "材料与能量诊断", "通过"),
        ("2", "诊断模体", "通过"),
        ("3", "稳健过滤与权重", "保留"),
        ("4", "迭代参数优化", "晋升"),
        ("5", "非均匀MLP", "保留"),
        ("6", "高级先验", "保留"),
        ("6A", "MLIC真值", "通过"),
        ("6B", "WEPL标定", "晋升"),
        ("7", "硅跟踪器", "通过"),
        ("7B", "噪声与权重", "保留"),
        ("7C", "通量敏感性", "通过"),
        ("8", "三维首轮", "性能未过"),
        ("8A", "能量模型评估", "搁置"),
        ("8B", "低通量适配", "晋升"),
        ("8C", "三维诊断复算", "通过"),
    ]
    fig, ax = plt.subplots(figsize=(18, 4.2))
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(-0.8, 1.3)
    ax.axis("off")
    ax.plot(range(len(labels)), [0] * len(labels), color="#94a3b8", linewidth=3, zorder=0)
    for index, (number, title, decision) in enumerate(labels):
        color = {
            "晋升": "#16a34a", "通过": "#2563eb", "保留": "#d97706",
            "搁置": "#64748b", "性能未过": "#dc2626",
        }[decision]
        ax.scatter(index, 0, s=420, color=color, edgecolor="white", linewidth=2, zorder=2)
        ax.text(index, 0, number, color="white", ha="center", va="center", weight="bold", fontsize=9)
        ax.text(index, 0.38, title, ha="center", va="bottom", fontsize=8, rotation=35)
        ax.text(index, -0.4, decision, ha="center", fontsize=8, color=color, weight="bold")
    ax.text((len(labels) - 1) / 2, 1.13, "Stage 0--8C研究决策时间线", ha="center", fontsize=16, weight="bold")
    ax.text(
        (len(labels) - 1) / 2, -0.72,
        "Stage 4与6B形成二维正式算法；Stage 8C形成首个通过性能门槛的三维体素基线",
        ha="center", fontsize=10, color="#475569"
    )
    save(fig, "stage_decisions.png")


def benchmark_context() -> None:
    names = ["Current S4\nideal 2-D", "Phase-II\nprototype", "ProtonVDA\nprototype", "2024 pCT\nplastic phantom"]
    mape = [0.2551, 1.14, 0.81, 0.28]
    resolution = [1.1733, 0.61, 0.46, 0.54]
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
        ("过滤后质子对", "仅主质子\n局部3σ"),
        ("水介质MLP", "0.1 mm路径步长\n双线性权重"),
        ("OS-SART", "18个子集\nλ0=0.25，衰减0.2"),
        ("物理约束", "非负\n100 mm支撑域"),
        ("Huber-TV", "β=0.0125\n5轮迭代"),
        ("RSP图像", "0.1 mm网格"),
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
    ax.text(6, 2.65, "当前冻结的二维最优重建流程", ha="center", fontsize=16, weight="bold")
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


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_key_metrics() -> dict[str, float]:
    stage6b = _csv_rows(
        CODE_ROOT / "research_stages/stage6b_wepl_calibration/qc/three_scene_metrics.csv"
    )
    by_dataset = {row["dataset"]: row for row in stage6b}
    stage7c = json.loads(
        (CODE_ROOT / "research_stages/stage7c_fluence_sensitivity/qc/stage7c_decision.json")
        .read_text(encoding="utf-8")
    )
    stage8 = json.loads(
        (REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8_decision.json")
        .read_text(encoding="utf-8")
    )
    stage8b = json.loads(
        (CODE_ROOT / "research_stages/stage8b_low_fluence_adaptation/qc/stage8b_decision.json")
        .read_text(encoding="utf-8")
    )
    stage8c = json.loads(
        (REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/fixed015_test_decision.json")
        .read_text(encoding="utf-8")
    )
    return {
        "s1_water_mean": float(by_dataset["s1"]["water_mean_calibrated"]),
        "s1_al_error": float(by_dataset["s1"]["aluminium_mlic_error_percent"]),
        "s4_large_mape": float(by_dataset["s4"]["mlic_large_insert_mape_calibrated_percent"]),
        "s4_all_mape": float(by_dataset["s4"]["mlic_all_non_air_mape_calibrated_percent"]),
        "s5_fmtf50": float(by_dataset["s5"]["fmtf50_calibrated_lp_per_mm"]),
        "s5_fmtf10": float(by_dataset["s5"]["fmtf10_calibrated_lp_per_mm"]),
        "minimum_fluence": float(
            stage7c["recommendations"]["combined_0p2mm_1pct"]["minimum_fluence_per_mm2_projection"]
        ),
        "stage8b_rmse_improvement_percent": 100.0 * float(
            stage8b["optimization"]["rmse_improvement_fraction"]
        ),
        "stage8b_water_std_improvement_percent": 100.0 * float(
            stage8b["optimization"]["water_std_improvement_fraction"]
        ),
        "stage8_test_wepl": float(stage8["test"]["wepl_rmse_mm"]),
        "stage8_water_bias_percent": 100.0 * float(stage8["image"]["water_bias"]),
        "stage8_large_mape": float(stage8["image"]["large_material_mape_percent"]),
        "stage8c_test_wepl": float(stage8c["test"]["wepl_rmse_mm"]),
        "stage8c_large_mape": float(stage8c["image"]["large_material_mape_percent"]),
    }


def project_status_dashboard(metrics: dict[str, float]) -> None:
    cards = [
        ("二维RSP准确度", f"S4大材料柱\nMAPE {metrics['s4_large_mape']:.3f}%", "#dcfce7", "#15803d"),
        ("二维空间分辨率", f"S5 fMTF10\n{metrics['s5_fmtf10']:.3f} lp/mm", "#dbeafe", "#1d4ed8"),
        (
            "D1低通量适配",
            f"总通量10%\nRMSE改善 {metrics['stage8b_rmse_improvement_percent']:.2f}%",
            "#fef3c7", "#b45309",
        ),
        ("三维体素基线", f"Stage 8C通过\n材料MAPE {metrics['stage8c_large_mape']:.3f}%", "#e0f2fe", "#0369a1"),
    ]
    fig, ax = plt.subplots(figsize=(13, 3.7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.7)
    ax.axis("off")
    for index, (title, value, face, edge) in enumerate(cards):
        x = 0.25 + index * 3.2
        patch = FancyBboxPatch(
            (x, 0.55), 2.75, 2.25, boxstyle="round,pad=0.06",
            facecolor=face, edgecolor=edge, linewidth=1.7,
        )
        ax.add_patch(patch)
        ax.text(x + 1.375, 2.33, title, ha="center", fontsize=11, weight="bold", color=edge)
        ax.text(x + 1.375, 1.42, value, ha="center", va="center", fontsize=12, color="#0f172a")
    ax.text(6.5, 3.35, "pCT项目结题状态（2026-08-09）", ha="center", fontsize=17, weight="bold")
    ax.text(
        6.5, 0.18,
        "二维算法与Stage 8C三维体素性能基线均已冻结；Stage 9可进入同场景3D Gaussian可行性验证。",
        ha="center", fontsize=9.5, color="#475569",
    )
    save(fig, "project_status_dashboard.png")


def expanded_scenario_overview() -> None:
    rows = [
        ("S1", "25根5 mm铝柱", "720 × 45万", "小目标恢复与径向一致性", "完成"),
        ("S2/S3", "均匀水，真空/空气", "720 × 10万", "边界与空气对照", "完成"),
        ("S4", "多材料、三个半径", "720 × 10万", "材料定量与部分容积", "完成"),
        ("S5", "线对与多方向斜边", "720 × 10万", "空间分辨率", "完成"),
        ("S6", "水/铝/空气薄板", "52个工况", "能量与WEPL一致性", "完成"),
        ("真实轨迹pilot", "异质模体与逐步轨迹", "72 × 5000", "路径模型理论上限", "完成"),
        ("虚拟MLIC", "深度剂量射程移动", "24组+高统计", "独立材料RSP真值", "完成"),
        ("独立水板", "30–230 MeV，4种厚度", "84个工况", "独立WEPL标定", "完成"),
        ("D1", "空气+四层硅", "720 × 45万", "探测器、噪声与通量", "低通量适配完成"),
        ("紧凑三维", "有限圆柱与5个材料球", "360 × 200万", "出平面三维重建", "Stage 8C通过"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["实验", "场景", "采样", "主要目的", "状态"],
        colWidths=[0.12, 0.26, 0.15, 0.31, 0.13],
        cellLoc="left", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#94a3b8")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold", color="#0f172a")
        elif column == 4:
            value = rows[row - 1][4]
            if value == "完成":
                cell.set_facecolor("#dcfce7")
                cell.set_text_props(weight="bold", color="#166534")
            elif value in {"低通量适配完成", "Stage 8C通过"}:
                cell.set_facecolor("#fef3c7")
                cell.set_text_props(weight="bold", color="#92400e")
            else:
                cell.set_facecolor("#fee2e2")
                cell.set_text_props(weight="bold", color="#b91c1c")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("仿真与独立标定实验矩阵", fontsize=17, weight="bold", pad=18)
    save(fig, "scenario_overview.png")


def full_stage_timeline() -> None:
    stages = [
        ("0", "冻结基线", "通过"), ("1", "能量口径", "通过"), ("2", "诊断模体", "通过"),
        ("3", "过滤与权重", "保留基线"), ("4", "迭代优化", "晋升"),
        ("5", "非均匀MLP", "保留基线"), ("6", "高级先验", "保留基线"),
        ("6A", "MLIC真值", "通过"), ("6B", "WEPL标定", "晋升"),
        ("7", "探测器效应", "通过"), ("7B", "噪声鲁棒", "保留基线"),
        ("7C", "通量敏感", "通过"), ("8", "三维体素", "性能未通过"),
        ("8A", "能量模型评估", "搁置"), ("8B", "低通量适配", "晋升"),
        ("8C", "三维诊断复算", "通过"),
    ]
    colors = {"通过": "#2563eb", "晋升": "#15803d", "保留基线": "#d97706", "性能未通过": "#dc2626", "进行中": "#7c3aed", "暂缓": "#64748b", "搁置": "#64748b"}
    fig, ax = plt.subplots(figsize=(15, 4.6))
    x = np.arange(len(stages))
    ax.plot(x, np.zeros_like(x), color="#cbd5e1", linewidth=3, zorder=0)
    for i, (number, title, status) in enumerate(stages):
        color = colors[status]
        ax.scatter(i, 0, s=430, color=color, edgecolor="white", linewidth=1.5, zorder=2)
        ax.text(i, 0, number, ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(i, 0.36 if i % 2 == 0 else -0.38, title, ha="center", va="bottom" if i % 2 == 0 else "top", rotation=35, fontsize=8.5)
        ax.text(i, -0.77, status, ha="center", fontsize=7.5, color=color, weight="bold", rotation=25)
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(-1.15, 1.05)
    ax.axis("off")
    ax.set_title("Stage 0–8B研究决策与当前进度", fontsize=17, weight="bold")
    save(fig, "stage_decisions.png")


def project_pipeline() -> None:
    blocks = [
        ("OpenGATE ROOT", "入口/出口相空间"), ("质子配对", "主质子Run/Event"),
        ("局部3σ", "能损与散射"), ("WEPL", "G4水射程标定"),
        ("Schulte MLP", "体内路径估计"), ("DDB-FDK", "解析基线/初值"),
        ("OS-SART", "GPU列表模式"), ("统一评价", "RSP、WEPL与MTF"),
    ]
    fig, ax = plt.subplots(figsize=(15, 3.6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    for index, (title, detail) in enumerate(blocks):
        x = 0.15 + index * 1.95
        face = "#eff6ff" if index < 5 else ("#fef3c7" if index == 5 else "#dcfce7")
        patch = FancyBboxPatch((x, 1.0), 1.55, 1.25, boxstyle="round,pad=0.04", facecolor=face, edgecolor="#475569", linewidth=1.0)
        ax.add_patch(patch)
        ax.text(x + 0.775, 1.78, title, ha="center", fontsize=9.5, weight="bold")
        ax.text(x + 0.775, 1.28, detail, ha="center", fontsize=8, color="#475569")
        if index < len(blocks) - 1:
            ax.annotate("", xy=(x + 1.9, 1.62), xytext=(x + 1.58, 1.62), arrowprops={"arrowstyle": "->", "color": "#64748b", "lw": 1.4})
    ax.text(8, 3.12, "pCT数据处理与重建总流程", ha="center", fontsize=17, weight="bold")
    save(fig, "project_pipeline.png")


def decision_funnel() -> None:
    candidates = [
        ("稳健过滤", "保留局部3σ"), ("WEPL数据权重", "保留等权"),
        ("Huber数据损失", "保留二次损失"), ("非均匀MLP", "保留水介质MLP"),
        ("TGV/自适应TV", "保留Huber-TV"), ("36个子集", "保留18个子集"),
        ("G4水WEPL标定", "正式晋升"),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.7))
    ax.axis("off")
    for index, (candidate, decision) in enumerate(candidates):
        y = len(candidates) - index
        width = 10.8 - index * 1.15
        x = (12 - width) / 2
        promoted = decision == "正式晋升"
        patch = FancyBboxPatch((x, y - 0.55), width, 0.72, boxstyle="round,pad=0.02", facecolor="#dcfce7" if promoted else "#f1f5f9", edgecolor="#15803d" if promoted else "#94a3b8")
        ax.add_patch(patch)
        ax.text(x + 0.18, y - 0.19, candidate, va="center", fontsize=10, weight="bold")
        ax.text(x + width - 0.18, y - 0.19, decision, ha="right", va="center", fontsize=9.5, color="#15803d" if promoted else "#475569")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, len(candidates) + 1.1)
    ax.set_title("候选方法的筛选与晋升结果", fontsize=17, weight="bold")
    save(fig, "decision_funnel.png")


def corrected_benchmark_context(metrics: dict[str, float]) -> None:
    names = ["本项目理想二维\n大材料柱", "Phase-II\n真实原型", "ProtonVDA\n真实原型", "2024 pCT\n塑料模体"]
    mape = [metrics["s4_large_mape"], 1.14, 0.81, 0.28]
    resolution = [metrics["s5_fmtf10"], 0.61, 0.46, 0.54]
    colors = ["#0f766e", "#2563eb", "#7c3aed", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(names, mape, color=colors)
    axes[0].set_ylabel("文献报告的RSP MAPE（%）")
    axes[0].set_title("RSP准确度（越低越好）")
    axes[1].bar(names, resolution, color=colors)
    axes[1].set_ylabel("文献报告的空间频率（lp/mm）")
    axes[1].set_title("空间分辨率（越高越好）")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("仅作量级定位：维度、探测器、剂量和ROI定义并不相同", fontsize=13.5, weight="bold")
    save(fig, "benchmark_context.png")


def stage7b_candidate_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT
        / "research_stages/stage7b_noise_robustness/qc/screen_candidate_metrics.csv"
    )
    source_seed = [
        row for row in rows
        if row["seed"] == "20260713" and row["mode"] in {"source", "candidate"}
    ]
    order = [
        "equal_quadratic", "analytic_invvar", "empirical_invvar",
        "huber_z1p5", "huber_z2p5", "empirical_huber_z2p5",
    ]
    by_name = {row["candidate"]: row for row in source_seed}
    selected = [by_name[name] for name in order if name in by_name]
    label_map = {
        "equal_quadratic": "等权二次",
        "analytic_invvar": "解析逆方差",
        "empirical_invvar": "经验逆方差",
        "huber_z1p5": "Huber 1.5",
        "huber_z2p5": "Huber 2.5",
        "empirical_huber_z2p5": "经验权重+Huber",
    }
    labels = [label_map.get(row["candidate"], row["candidate"]) for row in selected]
    values = [float(row["validation_ideal_rmse_mm"]) for row in selected]
    colors = ["#2563eb" if i == 0 else "#94a3b8" for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(values[0], color="#2563eb", linestyle="--", linewidth=1.2,
               label="等权基线")
    ax.set_ylabel("验证集理想WEPL RMSE（mm）")
    ax.set_title("Stage 7B：组合噪声下的候选数据项")
    ax.tick_params(axis="x", rotation=18, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006,
                f"{value:.3f}", ha="center", fontsize=8.5)
    fig.text(0.5, 0.01, "所有候选均高于等权基线，因此没有晋升。",
             ha="center", color="#475569")
    save(fig, "stage7b_candidate_validation.png")


def stage0_baseline_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT / "evaluation/baselines/results0716/checkpoint_metrics.csv"
    )
    selected = {
        row["checkpoint"]: row
        for row in rows
        if row["checkpoint"] in {"analytic_nohann", "iterative_epoch_03"}
    }
    labels = ["no-Hann DDB-FDK", "OS-SART第3轮"]
    checkpoints = ["analytic_nohann", "iterative_epoch_03"]
    metrics = [
        ("water_std_rsp", "水区RSP标准差"),
        ("phantom_rsp_rmse", "模体RSP RMSE"),
        ("validation_wepl_rmse_mm", "验证WEPL RMSE（mm）"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), constrained_layout=True)
    colors = ["#94a3b8", "#2563eb"]
    for axis, (field, title) in zip(axes, metrics):
        values = [float(selected[name][field]) for name in checkpoints]
        bars = axis.bar(labels, values, color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15, labelsize=8.5)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.015,
                f"{value:.4f}",
                ha="center",
                fontsize=8.5,
            )
    fig.suptitle("Stage 0：冻结解析与迭代历史基线", fontsize=14, weight="bold")
    save(fig, "stage0_frozen_baseline.png")


def stage6a_mlic_reference_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT
        / "research_stages/stage6a_mlic_reference/qc/mlic_reference_200mev.csv"
    )
    order = ["Water", "Lung", "A150_Tissue_Plastic", "SpineBone", "Aluminium"]
    labels = {
        "Water": "水",
        "Lung": "肺",
        "A150_Tissue_Plastic": "A150组织塑料",
        "SpineBone": "脊骨",
        "Aluminium": "铝",
    }
    by_material = {row["material"]: row for row in rows}
    values = [float(by_material[name]["mlic_rsp_200mev"]) for name in order]
    errors = [float(by_material[name]["bootstrap_sd"]) for name in order]
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    bars = ax.bar(
        [labels[name] for name in order],
        values,
        yerr=errors,
        capsize=4,
        color=["#2563eb", "#38bdf8", "#16a34a", "#d97706", "#64748b"],
    )
    ax.set_ylabel("200 MeV MLIC-RSP")
    ax.set_title("Stage 6A：虚拟MLIC独立材料真值")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035,
            f"{value:.6f}",
            ha="center",
            fontsize=8.5,
        )
    fig.text(
        0.5,
        0.01,
        "误差条为10个高统计重复得到的bootstrap标准差。",
        ha="center",
        color="#475569",
        fontsize=9,
    )
    save(fig, "stage6a_mlic_reference.png")


def stage1_water_consistency_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT
        / "research_stages/stage1_material_calibration/qc/results0717_s6_material_energy_scan"
        / "water_lut_consistency.csv"
    )
    thicknesses = sorted({float(row["thickness_mm"]) for row in rows})
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for thickness in thicknesses:
        subset = sorted(
            [row for row in rows if float(row["thickness_mm"]) == thickness],
            key=lambda row: float(row["energy_mev"]),
        )
        ax.plot(
            [float(row["energy_mev"]) for row in subset],
            [float(row["core_mean_rsp_bias_percent"]) for row in subset],
            marker="o", label=f"水厚 {thickness:g} mm",
        )
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_xlabel("入射能量（MeV）")
    ax.set_ylabel("BB78核心WEPL相对水厚偏差（%）")
    ax.set_title("Stage 1：BB78水射程与Geant4输运存在系统口径差异")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8.5)
    save(fig, "stage1_water_consistency.png")


def stage2_diagnostic_chart() -> None:
    image_rows = _csv_rows(
        CODE_ROOT / "research_stages/stage2_diagnostic_phantoms/qc/image_metrics.csv"
    )
    selected = {
        row["dataset"]: row for row in image_rows
        if row["method"] == "analytic" and row["variant"] == "selected"
        and row["dataset"] in {"vacuum", "air"}
    }
    line_pairs = _csv_rows(
        CODE_ROOT / "research_stages/stage2_diagnostic_phantoms/qc/line_pair_metrics.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    labels = ["真空均匀水", "空气均匀水"]
    boundary = [float(selected[key]["boundary_inner_rmse_vs_effective_rsp"]) for key in ("vacuum", "air")]
    outside = [float(selected[key]["outside_100_105_rmse_rsp"]) for key in ("vacuum", "air")]
    x = np.arange(2)
    width = 0.36
    axes[0].bar(x - width / 2, boundary, width, label="边界内侧RMSE", color="#2563eb")
    axes[0].bar(x + width / 2, outside, width, label="100–105 mm外侧RMSE", color="#d97706")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("RSP RMSE")
    axes[0].set_title("Air并非外围伪影的唯一来源")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(axis="y", alpha=0.25)
    lp = sorted(line_pairs, key=lambda row: float(row["spatial_frequency_lp_per_mm"]))
    axes[1].plot(
        [float(row["spatial_frequency_lp_per_mm"]) for row in lp],
        [float(row["modulation"]) for row in lp],
        marker="o", color="#16a34a", linewidth=2,
    )
    axes[1].set_xlabel("线对空间频率（lp/mm）")
    axes[1].set_ylabel("调制度")
    axes[1].set_title("S5线对调制度随频率下降")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Stage 2：边界、材料和空间分辨率诊断", fontsize=14, weight="bold")
    save(fig, "stage2_diagnostic_metrics.png")


def stage3_candidate_chart() -> None:
    filters = _csv_rows(
        CODE_ROOT / "research_stages/stage3_robust_weighting/qc/filter_validation_wepl.csv"
    )
    weights = _csv_rows(
        CODE_ROOT / "research_stages/stage3_robust_weighting/qc/weight_validation_wepl.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    candidate_names = {
        "baseline_3sigma": "局部3σ", "median_mad": "中位数/MAD",
        "robust_mahalanobis": "稳健马氏",
        "equal": "等权", "inverse_variance": "逆方差",
        "robust_confidence": "稳健置信", "combined": "组合权重",
    }
    for axis, rows, title in [
        (axes[0], filters, "过滤候选"), (axes[1], weights, "数据权重候选")
    ]:
        s2 = [row for row in rows if row["dataset"] == "s2"]
        labels = [candidate_names.get(row["candidate"], row["candidate"]) for row in s2]
        absolute = [float(row["rmse_mm"]) for row in s2]
        values = [100.0 * (value / absolute[0] - 1.0) for value in absolute]
        colors = ["#2563eb" if i == 0 else ("#dc2626" if value > 0 else "#16a34a")
                  for i, value in enumerate(values)]
        axis.bar(labels, values, color=colors)
        axis.axhline(0, color="#475569", linewidth=1)
        axis.set_ylabel("相对基线的验证WEPL RMSE变化（%）")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18, labelsize=8.5)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 3：稳健过滤与加权均未稳定超过基线", fontsize=14, weight="bold")
    save(fig, "stage3_candidate_comparison.png")


def stage4_convergence_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT / "research_stages/stage4_iterative_optimization/qc/regularization_final.csv"
    )
    variant = "r0p25_d0p2_quadratic_b0p0125_fixed_s18"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    colors = {"s2": "#2563eb", "s4": "#d97706", "s5": "#16a34a"}
    for dataset in ("s2", "s4", "s5"):
        subset = sorted(
            [row for row in rows if row["dataset"] == dataset and row["variant"] == variant],
            key=lambda row: int(row["epoch"]),
        )
        if not subset:
            continue
        axes[0].plot(
            [int(row["epoch"]) for row in subset],
            [float(row["validation_rmse_mm"]) for row in subset],
            marker="o", label=dataset.upper(), color=colors[dataset],
        )
    s2 = sorted(
        [row for row in rows if row["dataset"] == "s2" and row["variant"] == variant],
        key=lambda row: int(row["epoch"]),
    )
    axes[1].plot(
        [int(row["epoch"]) for row in s2],
        [float(row["water_core_std_rsp"]) for row in s2],
        marker="o", color="#2563eb", linewidth=2,
    )
    axes[0].set_xlabel("迭代轮数")
    axes[0].set_ylabel("验证WEPL RMSE（mm）")
    axes[0].set_title("数据一致性逐轮趋稳")
    axes[0].legend()
    axes[1].set_xlabel("迭代轮数")
    axes[1].set_ylabel("S2水区RSP标准差")
    axes[1].set_title("Huber-TV持续降低水区噪声")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Stage 4：冻结迭代调度的收敛证据", fontsize=14, weight="bold")
    save(fig, "stage4_convergence.png")


def stage5_path_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT / "research_stages/stage5_inhomogeneous_mlp/qc/path_metrics.csv"
    )
    selected = [row for row in rows if row["partition"] == "test"]
    groups = ["all", "heterogeneous"]
    labels = ["全部测试路径", "强异质路径"]
    by_group = {row["group"]: row for row in selected}
    water = [float(by_group[group]["water_rmse_mm"]) for group in groups]
    inhom = [float(by_group[group]["inhomogeneous_rmse_mm"]) for group in groups]
    x = np.arange(len(groups))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, water, width, label="Schulte水MLP", color="#2563eb")
    ax.bar(x + width / 2, inhom, width, label="真值材料非均匀MLP", color="#16a34a")
    ax.set_xticks(x, labels)
    ax.set_ylabel("相对Geant4真实轨迹RMSE（mm）")
    ax.set_title("Stage 5：非均匀MLP的路径收益非常有限")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, "stage5_path_comparison.png")


def stage6b_calibration_chart() -> None:
    rows = _csv_rows(
        CODE_ROOT / "research_stages/stage6b_wepl_calibration/qc/three_scene_metrics.csv"
    )
    by_dataset = {row["dataset"]: row for row in rows}
    s1 = by_dataset["s1"]
    s4 = by_dataset["s4"]
    s5 = by_dataset["s5"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), constrained_layout=True)
    labels = ["S1水偏差", "S4大材料柱MAPE"]
    before = [abs(float(s1["water_mean_old"]) - 1.0) * 100,
              float(s4["mlic_large_insert_mape_old_percent"])]
    after = [abs(float(s1["water_mean_calibrated"]) - 1.0) * 100,
             float(s4["mlic_large_insert_mape_calibrated_percent"])]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, before, width, label="标定前", color="#d97706")
    axes[0].bar(x + width / 2, after, width, label="标定后", color="#2563eb")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("绝对误差（%）")
    axes[0].set_title("RSP定量误差显著降低")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    mtf = [float(s5["fmtf10_old_lp_per_mm"]), float(s5["fmtf10_calibrated_lp_per_mm"])]
    axes[1].bar(["标定前", "标定后"], mtf, color=["#d97706", "#2563eb"])
    axes[1].set_ylabel("S5 fMTF10（lp/mm）")
    axes[1].set_title("空间分辨率没有被牺牲")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Stage 6B：独立水板WEPL标定的实际收益", fontsize=14, weight="bold")
    save(fig, "stage6b_calibration_effect.png")


def stage7c_charts() -> None:
    rows = _csv_rows(
        CODE_ROOT
        / "research_stages/stage7c_fluence_sensitivity/qc/qualification_metrics.csv"
    )
    rows = [row for row in rows if row["seed"] == "20260730"]
    conditions = ["ideal_reference", "continuous_hits", "combined_0p2mm_1pct"]
    names = {
        "ideal_reference": "理想参考面",
        "continuous_hits": "连续硅hit",
        "combined_0p2mm_1pct": "组合噪声",
    }
    colors = {
        "ideal_reference": "#2563eb",
        "continuous_hits": "#16a34a",
        "combined_0p2mm_1pct": "#dc2626",
    }
    angular_rows = _csv_rows(
        CODE_ROOT
        / "research_stages/stage8b_low_fluence_adaptation/qc/"
        "angular_fluence_360x20/comparison_metrics.csv"
    )
    angular = next(
        row for row in angular_rows
        if row["configuration"] == "360_angles_x_20pct"
    )
    for metric, ylabel, title, filename in [
        ("water_std", "水区RSP标准差", "水区噪声随总扫描通量变化", "stage7c_water_noise.png"),
        ("phantom_rmse_vs_rsp_truth", "模体RSP RMSE", "模体误差随总扫描通量变化", "stage7c_rmse.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        for condition in conditions:
            subset = sorted(
                [row for row in rows if row["condition"] == condition],
                key=lambda row: float(row["fluence_per_mm2_projection"]),
            )
            x = [100.0 * float(row["fraction"]) for row in subset]
            y = [float(row[metric]) for row in subset]
            ax.plot(x, y, marker="o", linewidth=2, color=colors[condition],
                    label=names[condition])
        angular_x = 100.0 * float(angular["nominal_total_fraction"])
        angular_y = float(angular[metric])
        ax.scatter(
            [angular_x], [angular_y], marker="*", s=190,
            color="#7c3aed", edgecolor="white", linewidth=0.8, zorder=5,
            label="360角度×20%（等总通量）",
        )
        ax.annotate(
            "360×20%", (angular_x, angular_y), xytext=(7, 7),
            textcoords="offset points", fontsize=9, color="#6d28d9",
        )
        ax.axvline(25, color="#d97706", linestyle="--", linewidth=1.3,
                   label="冻结算法下限：25%")
        ax.set_xlabel("相对720角度全通量的总扫描通量（%）")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
        save(fig, filename)

    multi = _csv_rows(
        CODE_ROOT
        / "research_stages/stage7c_fluence_sensitivity/qc/multiseed_metrics.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    for axis, fraction in zip(axes, [0.25, 0.10]):
        subset = [row for row in multi if abs(float(row["fraction"]) - fraction) < 1e-9]
        seeds = [row["seed"][-2:] for row in subset]
        values = [float(row["water_std"]) for row in subset]
        passed = [row["passed"].lower() == "true" for row in subset]
        axis.bar(seeds, values, color=["#16a34a" if ok else "#dc2626" for ok in passed])
        axis.set_xlabel("随机种子末两位")
        axis.set_ylabel("水区RSP标准差")
        axis.set_title(f"组合噪声：{fraction:.0%}通量")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("低通量多随机种子复核：绿色通过，红色失败", fontsize=14, weight="bold")
    save(fig, "stage7c_multiseed.png")


def stage8b_final_charts() -> None:
    """Render the locked-test gain and the final negative weighting result."""
    qc = CODE_ROOT / "research_stages/stage8b_low_fluence_adaptation/qc"
    test_rows = _csv_rows(qc / "optimization_test.csv")
    baseline = next(
        row for row in test_rows
        if row["candidate"] == "stage7c_baseline" and int(row["epoch"]) == 5
    )
    optimized = next(
        row for row in test_rows
        if row["candidate"] == "lowpass_0p5_upsampled" and int(row["epoch"]) == 2
    )
    panels = [
        ("水区RSP标准差", "water_std", "", "lower"),
        ("模体RSP RMSE", "phantom_rmse_vs_rsp_truth", "", "lower"),
        ("铝柱—水CNR", "insert_cnr_median", "", "higher"),
        ("10%–90%边缘宽度", "aluminium_edge_10_90_median_mm", "mm", "lower"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    for axis, (title, key, unit, direction) in zip(axes.flat, panels):
        values = [float(baseline[key]), float(optimized[key])]
        bars = axis.bar(
            ["冻结高通量参数", "低通量适配参数"], values,
            color=["#94a3b8", "#2563eb"], edgecolor="#334155", linewidth=0.7,
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(0, max(values) * 1.22)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, value,
                f"{value:.4f}" if value < 1 else f"{value:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        change = values[1] / values[0] - 1.0
        wording = "降低" if change < 0 else "增加"
        axis.text(
            0.5, 0.93,
            f"适配后{wording}{abs(change):.2%}（{'越低越好' if direction == 'lower' else '越高越好'}）",
            transform=axis.transAxes, ha="center", va="top", fontsize=9,
            color="#1d4ed8",
        )
    fig.suptitle(
        "Stage 8B锁定测试：360角度 × 每角度20%通量",
        fontsize=15, weight="bold",
    )
    fig.text(
        0.5, -0.01,
        "对照为原冻结参数第5轮；适配方案为0.5 mm低通初值上采样、衰减0.1、β=0.05并停止于第2轮。",
        ha="center", fontsize=9, color="#475569",
    )
    save(fig, "stage8b_locked_test.png")

    noise_rows = _csv_rows(qc / "noise_source_metrics.csv")
    main_rows = {
        row["condition"]: row for row in noise_rows
        if int(row["acquisition_seed"]) == 20260730
    }
    conditions = [
        "continuous_hits", "position_0p2mm", "energy_1pct_only",
        "combined_0p2mm_1pct",
    ]
    labels = ["连续硅hit", "仅位置误差", "仅能量噪声", "位置+能量"]
    weight_rows = _csv_rows(qc / "energy_weight_candidates.csv")
    valid_weights = [
        row for row in weight_rows if row.get("phantom_rmse_vs_rsp_truth", "")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    x = np.arange(len(conditions))
    width = 0.36
    water_std = [float(main_rows[key]["water_std"]) for key in conditions]
    rsp_rmse = [
        float(main_rows[key]["phantom_rmse_vs_rsp_truth"]) for key in conditions
    ]
    axes[0].bar(
        x - width / 2, water_std, width, label="水区标准差",
        color="#93c5fd", edgecolor="#1d4ed8",
    )
    axes[0].bar(
        x + width / 2, rsp_rmse, width, label="模体RSP RMSE",
        color="#fdba74", edgecolor="#c2410c",
    )
    axes[0].set_xticks(x, labels, rotation=16)
    axes[0].set_title("噪声源分解（主随机种子）")
    axes[0].set_ylabel("RSP指标")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend()

    weight_labels = [
        "等权" if row["candidate"] == "equal"
        else f"逆方差 γ={float(row['gamma']):g}"
        for row in valid_weights
    ]
    weight_rmse = [
        float(row["phantom_rmse_vs_rsp_truth"]) for row in valid_weights
    ]
    bars = axes[1].bar(
        weight_labels, weight_rmse,
        color=["#2563eb", "#93c5fd", "#bfdbfe"],
        edgecolor="#334155", linewidth=0.7,
    )
    axes[1].set_title("能量加权候选")
    axes[1].set_ylabel("模体RSP RMSE")
    axes[1].set_ylim(0, max(weight_rmse) * 1.22)
    axes[1].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, weight_rmse):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.5f}",
            ha="center", va="bottom", fontsize=9,
        )
    rejected = next(
        (row for row in weight_rows if row.get("status") == "REJECTED"), None
    )
    if rejected is not None:
        axes[1].text(
            0.98, 0.94,
            "γ=1.0：有效样本比例0.767 < 0.80，淘汰",
            transform=axes[1].transAxes, ha="right", va="top",
            fontsize=9, color="#b45309",
        )
    fig.suptitle(
        "Stage 8B低通量噪声分解与能量加权",
        fontsize=15, weight="bold",
    )
    save(fig, "stage8b_noise_weighting.png")


def copy_frozen_assets() -> dict[str, str]:
    sources = {
        "stage7_difference_vs_truth.png": CODE_ROOT / "research_stages/stage7_detector_effects/qc/assets/stage7_difference_vs_mlic_truth.png",
        "stage7b_candidate_validation.png": CODE_ROOT / "research_stages/stage7b_noise_robustness/qc/assets/candidate_validation.png",
        "stage7b_noise_sources.png": CODE_ROOT / "research_stages/stage7b_noise_robustness/qc/assets/noise_source_separation.png",
        "stage7c_water_noise.png": CODE_ROOT / "research_stages/stage7c_fluence_sensitivity/qc/assets/water_noise_vs_fluence.png",
        "stage7c_rmse.png": CODE_ROOT / "research_stages/stage7c_fluence_sensitivity/qc/assets/rmse_vs_fluence.png",
        "stage7c_multiseed.png": CODE_ROOT / "research_stages/stage7c_fluence_sensitivity/qc/assets/combined_multiseed_uncertainty.png",
        "stage8b_angular_images.png": CODE_ROOT / "research_stages/stage8b_low_fluence_adaptation/qc/angular_fluence_360x20/assets/image_and_difference_comparison.png",
        "stage8b_angular_epochs.png": CODE_ROOT / "research_stages/stage8b_low_fluence_adaptation/qc/angular_fluence_360x20/assets/epoch_metric_comparison.png",
        "stage8_slices.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/assets/truth_reconstruction_error.png",
        "stage8_orthogonal.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/assets/orthogonal_slices.png",
        "stage8_convergence.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/assets/epoch_convergence.png",
        "stage8_material_metrics.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/assets/material_and_edge_metrics.png",
        "stage8c_coverage.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/coverage_slices.png",
        "stage8c_rotation.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/rotation_residuals.png",
        "stage8c_convergence_candidates.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/convergence_candidates.png",
        "stage8c_full_convergence.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/full_fixed015_convergence.png",
        "stage8c_reconstruction_comparison.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/stage8_stage8c_comparison.png",
        "stage8c_material_comparison.png": REPOSITORY_ROOT / "pct3d_reconstruction/qc/results0718_compact_3d_pilot/stage8c/assets/material_recovery_comparison.png",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen QC assets:\n" + "\n".join(missing))
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        shutil.copy2(source, ASSET_ROOT / name)
    return {name: str(source.relative_to(REPOSITORY_ROOT)) for name, source in sources.items()}


def heavy_mhd_inputs() -> list[Path]:
    paths = [
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s2_water_vacuum_pilot/stage4/variants/r0p25_d0p2_quadratic_b0p0125_fixed_s18/recon/epoch_05.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s2_water_vacuum_pilot/stage6/variants/directional_tv_b0p0125_m0p5/recon/epoch_05.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s5_resolution_air_pilot/stage4/variants/r0p25_d0p2_quadratic_b0p0125_fixed_s18/recon/epoch_05.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s5_resolution_air_pilot/stage6/variants/directional_tv_b0p0125_m0p5/recon/epoch_05.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0716/analytic/truth/truth_rsp_200mev.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s1_aluminium_air_full/stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot/analytic/truth/truth_rsp_200mev.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot/stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s5_resolution_air_pilot/analytic/truth/truth_rsp_200mev.mhd",
        REPOSITORY_ROOT / "data/reconstruction_data/results0717_s5_resolution_air_pilot/stage6b_calibrated/iterative/recon/recon_iterative_gpu.mhd",
    ]
    return paths


def write_manifest(metrics: dict[str, float], copied: dict[str, str], refreshed: bool) -> None:
    entries = []
    for path in sorted(ASSET_ROOT.glob("*.png")):
        entries.append({
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "source": copied.get(path.name, "generated from code-side QC"),
        })
    payload = {
        "document": "pct2d_reconstruction/project_overview.md",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_safe_default": True,
        "mhd_panels_refreshed": refreshed,
        "key_metrics": metrics,
        "assets": entries,
    }
    (ASSET_ROOT / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-mhd-panels", action="store_true",
        help="regenerate classic/stage-6 MHD panels; requires restored archived data",
    )
    args = parser.parse_args()
    if args.refresh_mhd_panels:
        missing = [str(path) for path in heavy_mhd_inputs() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Cannot refresh MHD panels. Restore these inputs first:\n" + "\n".join(missing)
            )
    metrics = load_key_metrics()
    copied = copy_frozen_assets()
    project_status_dashboard(metrics)
    expanded_scenario_overview()
    full_stage_timeline()
    project_pipeline()
    decision_funnel()
    corrected_benchmark_context(metrics)
    best_pipeline()
    stage0_baseline_chart()
    stage1_water_consistency_chart()
    stage2_diagnostic_chart()
    stage3_candidate_chart()
    stage4_convergence_chart()
    stage5_path_chart()
    stage6a_mlic_reference_chart()
    stage6b_calibration_chart()
    stage7b_candidate_chart()
    stage7c_charts()
    stage8b_final_charts()
    for regenerated in (
        "stage7b_candidate_validation.png",
        "stage7c_water_noise.png",
        "stage7c_rmse.png",
        "stage7c_multiseed.png",
    ):
        copied.pop(regenerated, None)
    if args.refresh_mhd_panels:
        stage6_tradeoff()
        classic_scenario_results()
    else:
        for frozen in ("stage6_tradeoff.png", "classic_scenario_results.png"):
            if not (ASSET_ROOT / frozen).exists():
                raise FileNotFoundError(
                    f"Frozen panel missing: {ASSET_ROOT / frozen}. Restore it or use --refresh-mhd-panels."
                )
        copied["stage6_tradeoff.png"] = "frozen tracked panel; source MHD is archived"
        copied["classic_scenario_results.png"] = "frozen tracked panel; source MHD is archived"
    write_manifest(metrics, copied, args.refresh_mhd_panels)
    print(ASSET_ROOT)


if __name__ == "__main__":
    main()
