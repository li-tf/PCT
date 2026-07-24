#!/usr/bin/env python3
"""Calculate Stage 6 image, insert, localization and edge metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig-test0713-analytic")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = ["ddb_nohann", "ddb_hann1"]


def header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_mhd(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    h = header(path)
    size = [int(v) for v in h["DimSize"].split()]
    spacing = [float(v) for v in h["ElementSpacing"].split()]
    origin = [float(v) for v in h.get("Offset", h.get("Origin", "0 0 0")).split()]
    dtype = {"MET_FLOAT": "<f4", "MET_DOUBLE": "<f8"}[h["ElementType"]]
    raw = path.parent / h["ElementDataFile"]
    array = np.fromfile(raw, dtype=dtype).reshape(size[2], size[1], size[0])
    if size[1] != 1:
        raise ValueError(f"expected one slice, got {size} in {path}")
    x = origin[0] + spacing[0] * np.arange(size[0])
    z = origin[2] + spacing[2] * np.arange(size[2])
    meta = {"size": size, "spacing": spacing, "origin": origin, "path": str(path.resolve())}
    return np.asarray(array[:, 0, :], dtype=np.float32), x, z, meta


def bilinear(image: np.ndarray, x: np.ndarray, z: np.ndarray, origin: tuple[float, float], spacing: tuple[float, float]) -> np.ndarray:
    fx = (x - origin[0]) / spacing[0]
    fz = (z - origin[1]) / spacing[1]
    ix = np.floor(fx).astype(int)
    iz = np.floor(fz).astype(int)
    ix = np.clip(ix, 0, image.shape[1] - 2)
    iz = np.clip(iz, 0, image.shape[0] - 2)
    dx, dz = fx - ix, fz - iz
    return (
        image[iz, ix] * (1 - dx) * (1 - dz)
        + image[iz, ix + 1] * dx * (1 - dz)
        + image[iz + 1, ix] * (1 - dx) * dz
        + image[iz + 1, ix + 1] * dx * dz
    )


def falling_crossing(position: np.ndarray, values: np.ndarray, level: float) -> float:
    indices = np.flatnonzero(
        (position[:-1] >= 1.5)
        & (position[:-1] <= 3.5)
        & (values[:-1] >= level)
        & (values[1:] < level)
    )
    if not len(indices):
        return float("nan")
    i = int(indices[np.argmin(np.abs(position[indices] - 2.5))])
    if values[i] == values[i + 1]:
        return float(position[i])
    fraction = (level - values[i]) / (values[i + 1] - values[i])
    return float(position[i] + fraction * (position[i + 1] - position[i]))


def aluminium_edge_widths(
    image: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    centers: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Measure Rit et al.'s 10%-90% edge width on each aluminium insert.

    A 4 mm edge profile is formed by averaging 360 equally spaced radial
    profiles from the insert centre.  The coordinate spans 0.5--4.5 mm from
    the centre, placing the nominal 2.5 mm-radius boundary at its midpoint.
    """

    radial_position = np.arange(0.5, 4.5001, 0.025)
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    cosines = np.cos(angles)[:, None]
    sines = np.sin(angles)[:, None]
    rows: list[dict[str, float]] = []
    for center in centers:
        cx, cz = float(center["x"]), float(center["z"])
        profiles = bilinear(
            image,
            cx + cosines * radial_position[None, :],
            cz + sines * radial_position[None, :],
            (float(x[0]), float(z[0])),
            (float(x[1] - x[0]), float(z[1] - z[0])),
        )
        average_profile = profiles.mean(axis=0)
        inner = float(
            np.median(
                average_profile[
                    (radial_position >= 0.5) & (radial_position <= 1.5)
                ]
            )
        )
        outer = float(
            np.median(
                average_profile[
                    (radial_position >= 3.5) & (radial_position <= 4.5)
                ]
            )
        )
        contrast = inner - outer
        normalized = (average_profile - outer) / contrast
        r90 = falling_crossing(radial_position, normalized, 0.9)
        r10 = falling_crossing(radial_position, normalized, 0.1)
        width = r10 - r90
        rows.append(
            {
                "insert_id": int(center["id"]),
                "distance_from_isocenter_mm": math.hypot(cx, cz),
                "profile_count": 360,
                "profile_length_mm": 4.0,
                "inner_value": inner,
                "outer_value": outer,
                "contrast": contrast,
                "r90_mm": r90,
                "r10_mm": r10,
                "width_10_90_mm": width,
                "valid": bool(math.isfinite(width) and 0.0 < width < 4.0),
            }
        )
    return rows


def metrics_for(image: np.ndarray, red: np.ndarray, rsp: np.ndarray, x: np.ndarray, z: np.ndarray, centers: list[dict[str, float]]) -> tuple[dict[str, float], list[dict[str, float]]]:
    xx, zz = np.meshgrid(x, z)
    rr = np.hypot(xx, zz)
    exclusion = np.zeros_like(image, dtype=bool)
    for center in centers:
        exclusion |= (xx - center["x"]) ** 2 + (zz - center["z"]) ** 2 <= 5.0**2
    water_mask = (rr <= 95.0) & ~exclusion
    outside_mask = rr > 102.0
    phantom_mask = red > 1e-6
    aluminium_mask = red > 1.0 + 1e-6
    water = image[water_mask]
    inserts: list[dict[str, float]] = []
    for center in centers:
        cx, cz = float(center["x"]), float(center["z"])
        d2 = (xx - cx) ** 2 + (zz - cz) ** 2
        roi = d2 <= 2.0**2
        peak_roi = d2 <= 4.0**2
        shell = (d2 >= 4.0**2) & (d2 <= 8.0**2) & water_mask
        local = image[shell]
        local_mean = float(local.mean()) if local.size else float(water.mean())
        local_std = float(local.std()) if local.size else float(water.std())
        peak_index = int(np.argmax(np.where(peak_roi, image, -np.inf)))
        iz, ix = np.unravel_index(peak_index, image.shape)
        roi_mean = float(image[roi].mean())
        peak = float(image[iz, ix])
        positive = np.maximum(image[peak_roi] - local_mean, 0.0).astype(np.float64)
        roi_x = xx[peak_roi]
        roi_z = zz[peak_roi]
        if positive.sum() > 0:
            centroid_x = float(np.sum(positive * roi_x) / positive.sum())
            centroid_z = float(np.sum(positive * roi_z) / positive.sum())
            centroid_offset = math.hypot(centroid_x - cx, centroid_z - cz)
        else:
            centroid_x = centroid_z = centroid_offset = float("nan")
        inserts.append(
            {
                "insert_id": int(center["id"]),
                "truth_x_mm": cx,
                "truth_z_mm": cz,
                "roi_mean_r2": roi_mean,
                "roi_recovery_vs_red": (roi_mean - local_mean) / (2.343247781381188 - 1.0),
                "roi_recovery_vs_rsp": (roi_mean - local_mean) / (2.1189760409708303 - 1.0),
                "peak_r4": peak,
                "local_water_mean": local_mean,
                "local_water_std": local_std,
                "contrast": peak - local_mean,
                "cnr": (peak - local_mean) / local_std if local_std else float("nan"),
                "roi_cnr": (roi_mean - local_mean) / local_std if local_std else float("nan"),
                "contrast_centroid_offset_mm": centroid_offset,
                "contrast_centroid_x_mm": centroid_x,
                "contrast_centroid_z_mm": centroid_z,
                "peak_offset_mm": math.hypot(float(x[ix]) - cx, float(z[iz]) - cz),
                "peak_x_mm": float(x[ix]),
                "peak_z_mm": float(z[iz]),
            }
        )
    def rmse(diff: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(diff, dtype=np.float64))))
    peaks = np.array([v["peak_r4"] for v in inserts])
    contrasts = np.array([v["contrast"] for v in inserts])
    cnrs = np.array([v["cnr"] for v in inserts])
    roi_cnrs = np.array([v["roi_cnr"] for v in inserts])
    roi_red_recovery = np.array([v["roi_recovery_vs_red"] for v in inserts])
    roi_rsp_recovery = np.array([v["roi_recovery_vs_rsp"] for v in inserts])
    centroid_offsets = np.array([v["contrast_centroid_offset_mm"] for v in inserts])
    offsets = np.array([v["peak_offset_mm"] for v in inserts])
    summary = {
        "finite_fraction": float(np.isfinite(image).mean()),
        "image_min": float(image.min()),
        "image_max": float(image.max()),
        "water_mean": float(water.mean()),
        "water_std": float(water.std()),
        "water_rmse_vs_1": rmse(water - 1.0),
        "water_mae_vs_1": float(np.mean(np.abs(water - 1.0))),
        "outside_mean": float(image[outside_mask].mean()),
        "outside_std": float(image[outside_mask].std()),
        "outside_rmse_vs_0": rmse(image[outside_mask]),
        "phantom_rmse_vs_red_truth": rmse(image[phantom_mask] - red[phantom_mask]),
        "phantom_mae_vs_red_truth": float(np.mean(np.abs(image[phantom_mask] - red[phantom_mask]))),
        "phantom_rmse_vs_rsp_truth": rmse(image[phantom_mask] - rsp[phantom_mask]),
        "phantom_mae_vs_rsp_truth": float(np.mean(np.abs(image[phantom_mask] - rsp[phantom_mask]))),
        "aluminium_voxel_rmse_vs_red_truth": rmse(image[aluminium_mask] - red[aluminium_mask]),
        "aluminium_voxel_rmse_vs_rsp_truth": rmse(image[aluminium_mask] - rsp[aluminium_mask]),
        "aluminium_excess_red_recovery": float((image[aluminium_mask] - 1.0).sum(dtype=np.float64) / (red[aluminium_mask] - 1.0).sum(dtype=np.float64)),
        "aluminium_excess_rsp_recovery": float((image[aluminium_mask] - 1.0).sum(dtype=np.float64) / (rsp[aluminium_mask] - 1.0).sum(dtype=np.float64)),
        "insert_peak_mean": float(peaks.mean()),
        "insert_contrast_mean": float(contrasts.mean()),
        "insert_cnr_median": float(np.nanmedian(cnrs)),
        "insert_roi_cnr_median": float(np.nanmedian(roi_cnrs)),
        "insert_roi_red_recovery_mean": float(np.nanmean(roi_red_recovery)),
        "insert_roi_rsp_recovery_mean": float(np.nanmean(roi_rsp_recovery)),
        "insert_contrast_centroid_offset_mean_mm": float(np.nanmean(centroid_offsets)),
        "insert_contrast_centroid_offset_max_mm": float(np.nanmax(centroid_offsets)),
        "insert_peak_offset_mean_mm": float(offsets.mean()),
        "insert_peak_offset_max_mm": float(offsets.max()),
    }
    return summary, inserts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon-dir", type=Path, default=HERE / "recon")
    parser.add_argument("--truth-dir", type=Path, default=HERE / "truth")
    parser.add_argument("--qc-dir", type=Path, default=HERE / "qc")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument(
        "--flip-z-centers",
        action="store_true",
        help="Reflect insert ROI centers across z=0 for source-side/RTK audits.",
    )
    args = parser.parse_args()
    cases = args.cases
    qc = args.qc_dir
    qc.mkdir(parents=True, exist_ok=True)
    definition = json.loads((HERE.parent / "simulation" / "truth_geometry_definition.json").read_text(encoding="utf-8"))
    centers = definition["geometry"]["insert_centers_xz_mm"]
    if args.flip_z_centers:
        centers = [{**item, "z": -float(item["z"])} for item in centers]
    red, truth_x, truth_z, truth_meta = read_mhd(args.truth_dir / "truth_red.mhd")
    rsp, rsp_x, rsp_z, _ = read_mhd(args.truth_dir / "truth_rsp_200mev.mhd")
    if not (np.array_equal(truth_x, rsp_x) and np.array_equal(truth_z, rsp_z)):
        raise ValueError("RED and RSP truth grids differ")

    summaries: dict[str, dict[str, float]] = {}
    all_insert_rows: list[dict[str, object]] = []
    all_edge_rows: list[dict[str, object]] = []
    images: dict[str, np.ndarray] = {}
    grids: dict[str, dict[str, object]] = {}
    for case in cases:
        image, x, z, meta = read_mhd(args.recon_dir / f"recon_{case}.mhd")
        if not (np.array_equal(x, truth_x) and np.array_equal(z, truth_z)):
            raise ValueError(f"truth and {case} grids differ")
        summary, inserts = metrics_for(image, red, rsp, x, z, centers)
        edge = aluminium_edge_widths(image, x, z, centers)
        # Match Fig. 6 of Rit et al.: use the 24 non-central inserts.
        paper_edge = [
            row
            for row in edge
            if row["distance_from_isocenter_mm"] > 0.0 and row["valid"]
        ]
        widths = np.array([row["width_10_90_mm"] for row in paper_edge])
        inner_values = np.array([row["inner_value"] for row in edge])
        outer_values = np.array([row["outer_value"] for row in edge])
        summary.update(
            {
                "aluminium_inner_mean": float(inner_values.mean()),
                "aluminium_inner_std": float(inner_values.std()),
                "aluminium_outer_mean": float(outer_values.mean()),
                "aluminium_inner_ratio_to_red_truth": float(
                    inner_values.mean() / 2.343247781381188
                ),
                "aluminium_inner_ratio_to_rsp_200mev_truth": float(
                    inner_values.mean() / 2.1189760409708303
                ),
                "aluminium_edge_insert_count": int(len(widths)),
                "aluminium_edge_10_90_min_mm": float(widths.min()),
                "aluminium_edge_10_90_median_mm": float(np.median(widths)),
                "aluminium_edge_10_90_mean_mm": float(widths.mean()),
                "aluminium_edge_10_90_max_mm": float(widths.max()),
                "aluminium_edge_10_90_p05_mm": float(np.percentile(widths, 5)),
                "aluminium_edge_10_90_p95_mm": float(np.percentile(widths, 95)),
            }
        )
        summaries[case] = summary
        all_insert_rows.extend({"case": case, **row} for row in inserts)
        all_edge_rows.extend({"case": case, **row} for row in edge)
        images[case] = image
        grids[case] = meta

    metric_rows = [{"case": case, **values} for case, values in summaries.items()]
    write_csv(qc / "reconstruction_metrics.csv", metric_rows)
    write_csv(qc / "insert_metrics.csv", all_insert_rows)
    write_csv(qc / "aluminium_edge_resolution.csv", all_edge_rows)
    radial_rows: list[dict[str, object]] = []
    radial_groups = [(0, 6, "0-25"), (7, 12, "29-49"), (13, 18, "53-73"), (19, 24, "77-97")]
    for case in cases:
        case_rows = [row for row in all_insert_rows if row["case"] == case]
        for first_id, last_id, label in radial_groups:
            group = [row for row in case_rows if first_id <= int(row["insert_id"]) <= last_id]
            radial_rows.append(
                {
                    "case": case,
                    "radius_range_mm": label,
                    "insert_count": len(group),
                    "mean_roi_red_recovery": float(np.mean([row["roi_recovery_vs_red"] for row in group])),
                    "mean_roi_rsp_recovery": float(np.mean([row["roi_recovery_vs_rsp"] for row in group])),
                    "mean_roi_cnr": float(np.mean([row["roi_cnr"] for row in group])),
                    "mean_contrast_centroid_offset_mm": float(np.nanmean([row["contrast_centroid_offset_mm"] for row in group])),
                }
            )
    write_csv(qc / "radial_recovery.csv", radial_rows)

    result = {
        "truth_grid": truth_meta,
        "reconstruction_grids": grids,
        "flip_z_centers": args.flip_z_centers,
        "summaries": summaries,
        "radial_recovery": radial_rows,
        "paper_reported_spatial_resolution_range_mm": [0.7, 1.6],
        "notes": [
            "RED is the primary material truth; 200 MeV reference RSP is auxiliary.",
            "Aluminium edge resolution follows Rit et al.: 360 equally spaced radial profiles are averaged into one 4 mm profile per insert.",
            "The reported range uses the 24 non-central inserts, matching Fig. 6 of Rit et al.",
        ],
    }
    (qc / "stage6_metrics_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    extent = [float(truth_x[0]), float(truth_x[-1]), float(truth_z[0]), float(truth_z[-1])]
    panels = [("RED truth", red), ("RSP truth", rsp)] + [(case, images[case]) for case in cases]
    ncols = 2
    nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.25 * nrows), constrained_layout=True, squeeze=False)
    for ax, (title, data) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="lower", extent=extent, vmin=0, vmax=2.35, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.savefig(qc / "reconstruction_overview.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"summaries": summaries}, indent=2))


if __name__ == "__main__":
    raise SystemExit("internal metrics module; use run_analytic_reconstruction.py")
