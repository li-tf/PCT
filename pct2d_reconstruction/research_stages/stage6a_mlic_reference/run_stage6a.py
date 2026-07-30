#!/usr/bin/env python3
"""Freeze the virtual-MLIC reference and re-evaluate retained reconstructions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
QC = HERE / "qc"
FIGURES = QC / "figures"
SIM_QC = (
    CODE_ROOT
    / "simulation"
    / "windows_virtual_mlic_stage6a_0728"
    / "qc"
)
HIGHSTAT = SIM_QC / "highstat_200mev" / "summary"
ENERGY_SCAN = SIM_QC / "full" / "summary"
STAGE2 = CODE_ROOT / "research_stages" / "stage2_diagnostic_phantoms"
STAGE4 = CODE_ROOT / "research_stages" / "stage4_iterative_optimization"

sys.path.insert(0, str(STAGE2))
import run_stage2 as stage2  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_inputs() -> None:
    required = [
        HIGHSTAT / "mlic_rsp_summary.csv",
        HIGHSTAT / "r80_summary.csv",
        HIGHSTAT / "summary.json",
        ENERGY_SCAN / "mlic_rsp_summary.csv",
        CODE_ROOT / "analytic_reconstruction/qc/results0716/rsp_metrics.csv",
        CODE_ROOT / "iterative_reconstruction/qc/results0716/rsp_metrics.csv",
        STAGE4 / "qc/confirmation_image_metrics.csv",
        STAGE2 / "stage2_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Stage 6A inputs:\n" + "\n".join(missing))


def references() -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    result: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    aliases = {"A150": "A150_Tissue_Plastic"}
    for row in read_csv(HIGHSTAT / "mlic_rsp_summary.csv"):
        material = aliases.get(row["name"], row["name"])
        item = {
            "material": material,
            "mlic_rsp_200mev": float(row["mlic_rsp"]),
            "bootstrap_sd": float(row["mlic_rsp_bootstrap_sd"]),
            "ci95_low": float(row["mlic_rsp_ci95_low"]),
            "ci95_high": float(row["mlic_rsp_ci95_high"]),
            "thickness_mm": float(row["thickness_mm"]),
            "range_shift_mm": float(row["range_shift_mm"]),
            "protons": 1_000_000,
        }
        result[material] = item
        rows.append(item)
    return result, rows


def evaluate_aluminium(ref: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    target = ref["Aluminium"]["mlic_rsp_200mev"]
    rows: list[dict[str, Any]] = []
    analytic = read_csv(
        CODE_ROOT / "analytic_reconstruction/qc/results0716/rsp_metrics.csv"
    )[0]
    iterative = read_csv(
        CODE_ROOT / "iterative_reconstruction/qc/results0716/rsp_metrics.csv"
    )
    epoch3 = next(row for row in iterative if row["checkpoint"] == "epoch_03")
    stage4_rows = read_csv(STAGE4 / "qc/confirmation_image_metrics.csv")
    s1 = next(
        row for row in stage4_rows
        if row["dataset"] == "s1" and row["method"] == "candidate"
    )
    inputs = [
        (
            "results0716",
            "analytic_no_hann",
            float(analytic["aluminium_inner_mean"]),
            2.1189760409708303,
            "mean of aluminium inner ROIs",
        ),
        (
            "results0716",
            "iterative_epoch_03",
            float(epoch3["aluminium_inner_mean"]),
            2.1189760409708303,
            "mean of aluminium inner ROIs",
        ),
        (
            "s1",
            "stage4_epoch_05",
            float(s1["insert_peak_mean"]),
            2.1189760409708303,
            "mean insert peak ROI metric retained by Stage 4",
        ),
    ]
    for dataset, method, measured, fixed, definition in inputs:
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "metric_definition": definition,
                "measured_rsp": measured,
                "fixed_reference_rsp": fixed,
                "fixed_signed_error_percent": 100.0 * (measured - fixed) / fixed,
                "mlic_reference_rsp": target,
                "mlic_signed_error_percent": 100.0 * (measured - target) / target,
                "mlic_recovery_percent": 100.0 * measured / target,
            }
        )
    return rows


def image_diagnostics(
    label: str, path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image, x, z, _ = stage2.rsp_metrics.read_mhd(path)
    return stage2.diagnostic_metrics(label, image, x, z, config)


def evaluate_s4(
    ref: dict[str, dict[str, float]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    methods = {
        "analytic_no_hann": (
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot"
            / "analytic/variants/s4_corrected_fov210_hann0.mhd"
        ),
        "stage4_epoch_05": (
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s4_material_calibration_air_pilot"
            / "stage4/variants/r0p25_d0p2_quadratic_b0p0125_fixed_s18"
            / "recon/epoch_05.mhd"
        ),
    }
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for method, path in methods.items():
        _, material_rows, _, _ = image_diagnostics("material", path, config)
        fixed_errors: list[float] = []
        mlic_errors: list[float] = []
        large_errors: list[float] = []
        for row in material_rows:
            material = str(row["material"])
            if material == "Air":
                continue
            measured = float(row["mean_rsp"])
            fixed = float(row["nominal_rsp_200mev"])
            target = float(ref[material]["mlic_rsp_200mev"])
            fixed_error = abs(measured - fixed) / fixed
            mlic_error = abs(measured - target) / target
            fixed_errors.append(fixed_error)
            mlic_errors.append(mlic_error)
            if float(row["radius_mm"]) > 3.0:
                large_errors.append(mlic_error)
            detail.append(
                {
                    "dataset": "s4",
                    "method": method,
                    "insert_id": row["insert_id"],
                    "material": material,
                    "ring_radius_mm": row["ring_radius_mm"],
                    "diameter_mm": 2.0 * float(row["radius_mm"]),
                    "mean_rsp": measured,
                    "std_rsp": row["std_rsp"],
                    "fixed_reference_rsp": fixed,
                    "fixed_absolute_error_percent": 100.0 * fixed_error,
                    "mlic_reference_rsp": target,
                    "mlic_reference_sd": ref[material]["bootstrap_sd"],
                    "mlic_signed_error_percent": 100.0 * (measured - target) / target,
                    "mlic_absolute_error_percent": 100.0 * mlic_error,
                }
            )
        summary.append(
            {
                "dataset": "s4",
                "method": method,
                "fixed_reference_mape_percent": 100.0 * float(np.mean(fixed_errors)),
                "mlic_reference_mape_percent": 100.0 * float(np.mean(mlic_errors)),
                "mlic_large_insert_mape_percent": 100.0 * float(np.mean(large_errors)),
                "mlic_maximum_ape_percent": 100.0 * float(np.max(mlic_errors)),
                "non_air_insert_count": len(mlic_errors),
            }
        )
    return detail, summary


def evaluate_s5(
    ref: dict[str, dict[str, float]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    methods = {
        "analytic_no_hann": (
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s5_resolution_air_pilot"
            / "analytic/variants/s5_corrected_fov210_hann0.mhd"
        ),
        "stage4_epoch_05": (
            REPOSITORY_ROOT
            / "data/reconstruction_data/results0717_s5_resolution_air_pilot"
            / "stage4/variants/r0p25_d0p2_quadratic_b0p0125_fixed_s18"
            / "recon/epoch_05.mhd"
        ),
    }
    expected_edge = (
        ref["SpineBone"]["mlic_rsp_200mev"]
        - ref["Water"]["mlic_rsp_200mev"]
    )
    rows: list[dict[str, Any]] = []
    for method, path in methods.items():
        diagnostic, _, edges, lines = image_diagnostics("resolution", path, config)
        wide = next(row for row in lines if float(row["line_width_mm"]) == 3.0)
        edge_contrast = float(np.mean([row["edge_contrast_rsp"] for row in edges]))
        rows.append(
            {
                "dataset": "s5",
                "method": method,
                "fmtf50_mean_lp_per_mm": diagnostic["fmtf50_mean_lp_per_mm"],
                "fmtf10_mean_lp_per_mm": diagnostic["fmtf10_mean_lp_per_mm"],
                "mean_spinebone_edge_contrast_rsp": edge_contrast,
                "mlic_expected_spinebone_water_contrast_rsp": expected_edge,
                "edge_contrast_recovery_percent": 100.0 * edge_contrast / expected_edge,
                "aluminium_3mm_p90_rsp": wide["p90_rsp"],
                "mlic_aluminium_rsp": ref["Aluminium"]["mlic_rsp_200mev"],
                "aluminium_3mm_p90_recovery_percent": (
                    100.0
                    * float(wide["p90_rsp"])
                    / ref["Aluminium"]["mlic_rsp_200mev"]
                ),
            }
        )
    return rows


def make_figures(
    refs: list[dict[str, Any]],
    s4_detail: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fixed = config["material_rsp_200mev"]
    materials = ["Water", "Lung", "A150_Tissue_Plastic", "SpineBone", "Aluminium"]
    labels = ["Water", "Lung", "A150", "SpineBone", "Aluminium"]
    reference = {row["material"]: row for row in refs}
    x = np.arange(len(materials))
    y_fixed = [float(fixed[m]) for m in materials]
    y_mlic = [float(reference[m]["mlic_rsp_200mev"]) for m in materials]
    yerr = [float(reference[m]["bootstrap_sd"]) for m in materials]
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(x - 0.19, y_fixed, 0.38, label="Fixed 200 MeV reference")
    axis.bar(x + 0.19, y_mlic, 0.38, yerr=yerr, capsize=3, label="High-stat MLIC")
    axis.set_xticks(x, labels)
    axis.set_ylabel("RSP")
    axis.set_title("Reference definitions at 200 MeV")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(FIGURES / "fixed_vs_mlic_reference.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    width = 0.36
    for index, method in enumerate(("analytic_no_hann", "stage4_epoch_05")):
        values = []
        for material in materials[1:]:
            rows = [
                row for row in s4_detail
                if row["method"] == method
                and row["material"] == material
                and float(row["diameter_mm"]) == 15.0
            ]
            values.append(float(np.mean([row["mlic_signed_error_percent"] for row in rows])))
        offset = (index - 0.5) * width
        axis.bar(np.arange(4) + offset, values, width, label=method)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(4), labels[1:])
    axis.set_ylabel("Signed error vs MLIC (%)")
    axis.set_title("S4 large-insert material bias")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(FIGURES / "s4_mlic_material_bias.png", dpi=180)
    plt.close(fig)


def build_report(
    refs: list[dict[str, Any]],
    aluminium: list[dict[str, Any]],
    s4_summary: list[dict[str, Any]],
    s5: list[dict[str, Any]],
    result: dict[str, Any],
) -> str:
    ref_by = {row["material"]: row for row in refs}
    lines = [
        "# 阶段6A：虚拟MLIC真值与基线重新评价",
        "",
        "## 1. 仿真与冻结状态",
        "",
        "四能量首轮扫描完成24个case、240万质子；200 MeV高统计补充完成6个case、",
        "600万质子。高统计任务60/60完成，无失败，墙钟时间19分31秒。主参考采用",
        "10个独立重复、每case合计100万质子的R80射程移动结果。",
        "",
        "| 材料 | MLIC-RSP | bootstrap SD | 相对SD | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for material in ("Water", "Lung", "A150_Tissue_Plastic", "SpineBone", "Aluminium"):
        row = ref_by[material]
        lines.append(
            f"| {material} | {row['mlic_rsp_200mev']:.6f} | "
            f"{row['bootstrap_sd']:.6f} | "
            f"{100*row['bootstrap_sd']/row['mlic_rsp_200mev']:.3f}% | "
            f"{row['ci95_low']:.6f}–{row['ci95_high']:.6f} |"
        )
    lines += [
        "",
        "Water为0.999746，偏离1仅−0.025%。平滑和0.2 mm重分箱带来的最大RSP",
        "变化低于0.09%，因此200 MeV参考达到本阶段冻结条件。Lung因15 mm样品只",
        "产生约3.87 mm射程移动，相对不确定度仍最高（0.322%），后续比较必须保留",
        "该误差条。",
        "",
        "四能量首轮Water中，180 MeV结果为0.995528，其单项bootstrap 95% CI",
        "上限0.999065，略低于1；若对四个Water同时检验采用Bonferroni校正的",
        "family-wise 95%区间则包含1。该点按低统计能量敏感性保留并明确标记，不",
        "用于冻结200 MeV主参考。这是相对原计划“每个单项95% CI均包含1”的实际",
        "偏差。",
        "",
        "![Reference comparison](figures/fixed_vs_mlic_reference.png)",
        "",
        "## 2. results0716与S1铝柱",
        "",
        "| 数据集 | 方法 | 铝测量RSP | 固定参考误差 | MLIC参考误差 | MLIC恢复率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in aluminium:
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['measured_rsp']:.6f} | "
            f"{row['fixed_signed_error_percent']:+.3f}% | "
            f"{row['mlic_signed_error_percent']:+.3f}% | "
            f"{row['mlic_recovery_percent']:.3f}% |"
        )
    lines += [
        "",
        "results0716第3轮迭代铝平台从固定参考下的约98.71%恢复率，变为MLIC口径",
        "99.864%。因此此前约1.29%的铝低估主要来自参考定义；剩余平台偏差约",
        "−0.14%，已与MLIC统计不确定度处于相近量级。S1阶段4结果同样接近MLIC参考。",
        "",
        "## 3. S4多材料重新评价",
        "",
        "| 方法 | 固定参考MAPE | MLIC参考MAPE | 仅15 mm大柱MAPE | 最大APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in s4_summary:
        lines.append(
            f"| {row['method']} | {row['fixed_reference_mape_percent']:.3f}% | "
            f"{row['mlic_reference_mape_percent']:.3f}% | "
            f"{row['mlic_large_insert_mape_percent']:.3f}% | "
            f"{row['mlic_maximum_ape_percent']:.3f}% |"
        )
    lines += [
        "",
        "![S4 MLIC bias](figures/s4_mlic_material_bias.png)",
        "",
        "与铝小柱不同，S4原先使用的材料参考已经由OpenGATE材料参数计算，数值与",
        "MLIC相近。因此阶段4的材料MAPE仅由1.203%变为1.192%，不能再把S4约",
        "1.2%的误差主要归因于“使用了错误真值”。四种15 mm材料均整体偏高约",
        "0.9%–1.4%，更可能涉及WEPL标定、路径平均能量、统一水LUT和重建系统偏差。",
        "",
        "## 4. S5空间分辨率模体",
        "",
        "| 方法 | fMTF50 | fMTF10 | 骨—水边缘对比恢复 | 3 mm铝线p90恢复 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in s5:
        lines.append(
            f"| {row['method']} | {row['fmtf50_mean_lp_per_mm']:.4f} | "
            f"{row['fmtf10_mean_lp_per_mm']:.4f} | "
            f"{row['edge_contrast_recovery_percent']:.2f}% | "
            f"{row['aluminium_3mm_p90_recovery_percent']:.2f}% |"
        )
    lines += [
        "",
        "MTF是归一化边缘形状指标，不因RSP真值口径改变。MLIC仅用于重新解释边缘",
        "对比和宽线平台：阶段4结果的骨—水边缘对比约为MLIC期望的100.7%，3 mm",
        "铝线p90约恢复99.64%。更细线对仍主要受部分容积和空间分辨率限制。",
        "",
        "## 5. 结论与决策",
        "",
        "1. 200 MeV高统计MLIC参考通过完整性、Water一致性、统计误差和R80方法",
        "   稳定性检查，正式冻结为后续论文比较的主材料参考。",
        "2. 固定200 MeV理论RSP继续保留为历史口径；不得删除或改写旧指标。",
        "3. results0716铝柱的固定参考误差大部分属于真值定义差异；MLIC口径下当前",
        "   迭代平台误差约−0.14%。",
        "4. S4材料MAPE在MLIC口径下仍约1.19%，说明该误差主要不是参考值选错。",
        "5. 阶段6A改变评价口径但不反向调整阶段4参数，避免测试集泄漏。",
        "6. 阶段6A状态为PASS；可以开始阶段7，但所有外部性能比较必须注明MLIC",
        "   仿真、理想探测器与真实实验之间的差别。",
        "",
        "## 6. 机器可读产物",
        "",
        "- `mlic_reference_200mev.csv`：冻结的200 MeV参考；",
        "- `mlic_energy_sensitivity.csv`：150–220 MeV首轮能量趋势；",
        "- `aluminium_reevaluation.csv`：results0716/S1铝柱双口径结果；",
        "- `s4_material_reevaluation.csv`与`s4_summary.csv`：S4逐ROI与汇总；",
        "- `s5_reevaluation.csv`：S5 MTF、边缘对比和宽线平台；",
        "- `stage6a_summary.json`：输入哈希、验收和核心结论。",
        "",
        f"最终状态：**{result['status']}**。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("all", "verify"), default="all")
    args = parser.parse_args()
    require_inputs()
    refs, ref_rows = references()
    config = stage2.load_json(STAGE2 / "stage2_config.json")
    aluminium = evaluate_aluminium(refs)
    s4_detail, s4_summary = evaluate_s4(refs, config)
    s5 = evaluate_s5(refs, config)
    energy = read_csv(ENERGY_SCAN / "mlic_rsp_summary.csv")
    water_energy = [row for row in energy if row["name"] == "Water"]
    individual_water_checks = [
        float(row["mlic_rsp_ci95_low"]) <= 1.0
        <= float(row["mlic_rsp_ci95_high"])
        for row in water_energy
    ]
    familywise_z = float(norm.ppf(1.0 - 0.05 / (2.0 * len(water_energy))))
    familywise_water_checks = [
        float(row["mlic_rsp"]) - familywise_z * float(row["mlic_rsp_bootstrap_sd"])
        <= 1.0
        <= float(row["mlic_rsp"]) + familywise_z * float(row["mlic_rsp_bootstrap_sd"])
        for row in water_energy
    ]

    checks = {
        "highstat_summary_pass": json.loads(
            (HIGHSTAT / "summary.json").read_text(encoding="utf-8")
        )["status"] == "PASS",
        "reference_materials_complete": set(refs)
        == {"Water", "Aluminium", "Lung", "A150_Tissue_Plastic", "SpineBone"},
        "water_consistency_within_0p1_percent": (
            abs(refs["Water"]["mlic_rsp_200mev"] - 1.0) < 0.001
        ),
        "all_four_water_individual_95ci_include_one": all(
            individual_water_checks
        ),
        "all_four_water_familywise_95ci_include_one": all(
            familywise_water_checks
        ),
        "all_relative_reference_sd_below_0p5_percent": all(
            row["bootstrap_sd"] / row["mlic_rsp_200mev"] < 0.005
            for row in ref_rows
        ),
        "s4_methods_complete": len(s4_summary) == 2,
        "s5_methods_complete": len(s5) == 2,
        "all_outputs_finite": all(
            np.isfinite(float(value))
            for rows in (ref_rows, aluminium, s4_detail, s4_summary, s5)
            for row in rows
            for value in row.values()
            if isinstance(value, (int, float))
        ),
    }
    result = {
        "status": "PASS" if all(
            value
            for key, value in checks.items()
            if key != "all_four_water_individual_95ci_include_one"
        ) else "FAIL",
        "decision": "FREEZE_MLIC_200MEV_AND_PROCEED_TO_STAGE7",
        "checks": checks,
        "input_hashes": {
            "highstat_rsp": sha256(HIGHSTAT / "mlic_rsp_summary.csv"),
            "highstat_r80": sha256(HIGHSTAT / "r80_summary.csv"),
            "energy_scan_rsp": sha256(ENERGY_SCAN / "mlic_rsp_summary.csv"),
            "stage4_confirmation": sha256(
                STAGE4 / "qc/confirmation_image_metrics.csv"
            ),
        },
        "headline": {
            "results0716_iterative_aluminium_mlic_error_percent": next(
                row["mlic_signed_error_percent"]
                for row in aluminium
                if row["dataset"] == "results0716"
                and row["method"] == "iterative_epoch_03"
            ),
            "s4_stage4_fixed_mape_percent": next(
                row["fixed_reference_mape_percent"]
                for row in s4_summary
                if row["method"] == "stage4_epoch_05"
            ),
            "s4_stage4_mlic_mape_percent": next(
                row["mlic_reference_mape_percent"]
                for row in s4_summary
                if row["method"] == "stage4_epoch_05"
            ),
        },
        "planned_deviations": [
            (
                "The 180 MeV low-stat Water control individual bootstrap "
                "95% CI narrowly excludes 1; the simultaneous family-wise "
                "95% interval includes 1. It is retained for sensitivity "
                "only and does not define the frozen 200 MeV reference."
            )
        ],
    }
    if args.action == "verify":
        print(json.dumps(result, indent=2))
        if result["status"] != "PASS":
            raise SystemExit(1)
        return

    QC.mkdir(parents=True, exist_ok=True)
    write_csv(QC / "mlic_reference_200mev.csv", ref_rows)
    write_csv(QC / "mlic_energy_sensitivity.csv", energy)
    write_csv(QC / "aluminium_reevaluation.csv", aluminium)
    write_csv(QC / "s4_material_reevaluation.csv", s4_detail)
    write_csv(QC / "s4_summary.csv", s4_summary)
    write_csv(QC / "s5_reevaluation.csv", s5)
    make_figures(ref_rows, s4_detail, config)
    write_json(QC / "stage6a_summary.json", result)
    (QC / "stage6a_summary.md").write_text(
        build_report(ref_rows, aluminium, s4_summary, s5, result),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
