"""Voxel-truth and quantitative metrics for the compact 3-D phantom."""

from __future__ import annotations

from typing import Any

import numpy as np

from physics3d import coordinates, support_mask


def _crossing(coordinate: np.ndarray, normalized: np.ndarray, level: float) -> float:
    for index in range(len(normalized) - 1):
        first, second = normalized[index], normalized[index + 1]
        if (first - level) * (second - level) <= 0 and first != second:
            fraction = (level - first) / (second - first)
            return float(coordinate[index] + fraction * (coordinate[index + 1] - coordinate[index]))
    return float("nan")


def sphere_edge_widths(
    image: np.ndarray, config: dict, simulation: dict, mlic: dict[str, float]
) -> list[dict[str, Any]]:
    x, y, z = coordinates(config)
    axes = {"x": (x, 2), "y": (y, 1), "z": (z, 0)}
    rows = []
    air_value = float(config["air_wepl_slope_mm_per_mm"])
    for item in simulation["spheres"]:
        center = np.asarray(item["scanner_center_mm"], dtype=float)
        radius = float(item["diameter_mm"]) / 2
        target = air_value if item["material"] == "Air" else float(mlic[item["material"]])
        for axis_name, (coord, dimension) in axes.items():
            indexes = [int(np.argmin(np.abs(x - center[0]))), int(np.argmin(np.abs(y - center[1]))), int(np.argmin(np.abs(z - center[2])))]
            if dimension == 2:
                profile = image[indexes[2], indexes[1], :]
                center_value = center[0]
            elif dimension == 1:
                profile = image[indexes[2], :, indexes[0]]
                center_value = center[1]
            else:
                profile = image[:, indexes[1], indexes[0]]
                center_value = center[2]
            widths = []
            for sign in (-1, 1):
                lo, hi = sorted((center_value + sign * (radius - 2.0), center_value + sign * (radius + 2.0)))
                mask = (coord >= lo) & (coord <= hi)
                c, p = coord[mask], profile[mask]
                if sign < 0:
                    c, p = c[::-1], p[::-1]
                normalized = (p - float(mlic["Water"])) / (target - float(mlic["Water"]))
                at90, at10 = _crossing(c, normalized, 0.9), _crossing(c, normalized, 0.1)
                widths.append(abs(at10 - at90) if np.isfinite(at90 + at10) else np.nan)
            rows.append(
                {
                    "sphere": item["name"],
                    "material": item["material"],
                    "axis": axis_name,
                    "edge_width_10_90_mm": float(np.nanmean(widths)) if np.isfinite(widths).any() else float("nan"),
                }
            )
    return rows


def image_metrics(
    image: np.ndarray,
    truth: np.ndarray,
    config: dict,
    simulation: dict,
    mlic: dict[str, float],
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    image = np.asarray(image, dtype=np.float32)
    if image.shape != truth.shape or not np.isfinite(image).all():
        raise ValueError("invalid reconstruction volume")
    x, y, z = coordinates(config)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    support = support_mask(config)
    water = (
        (xx * xx + zz * zz <= (float(config["phantom_radius_mm"]) - 3.0) ** 2)
        & (np.abs(yy) <= float(config["phantom_half_length_y_mm"]) - 3.0)
    )
    for item in simulation["spheres"]:
        center = np.asarray(item["scanner_center_mm"], dtype=float)
        radius = float(item["diameter_mm"]) / 2 + 2.0
        water &= (
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
            > radius**2
        )
    water_values = image[water]
    material_rows: list[dict[str, Any]] = []
    absolute_nonair = []
    absolute_large = []
    for item in simulation["spheres"]:
        center = np.asarray(item["scanner_center_mm"], dtype=float)
        radius = max(float(item["diameter_mm"]) / 2 - 1.0, 0.5)
        roi = (
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
            <= radius**2
        )
        values = image[roi]
        reference = (
            float(config["air_wepl_slope_mm_per_mm"])
            if item["material"] == "Air"
            else float(mlic[item["material"]])
        )
        error_percent = 100.0 * (float(np.mean(values)) - reference) / reference
        if item["material"] != "Air":
            absolute_nonair.append(abs(error_percent))
            if item["material"] != "Aluminium":
                absolute_large.append(abs(error_percent))
        contrast = abs(float(np.mean(values)) - float(np.mean(water_values)))
        neighborhood = (
            (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
            <= (float(item["diameter_mm"]) / 2 + 2.0) ** 2
        )
        midpoint = 0.5 * (reference + float(mlic["Water"]))
        segmented = neighborhood & ((image < midpoint) if reference < float(mlic["Water"]) else (image > midpoint))
        if np.any(segmented):
            centroid = np.array(
                [np.mean(xx[segmented]), np.mean(yy[segmented]), np.mean(zz[segmented])]
            )
            centroid_error = float(np.linalg.norm(centroid - center))
        else:
            centroid_error = float("nan")
        voxel_volume = float(np.prod(config["grid"]["spacing_xyz_mm"]))
        material_rows.append(
            {
                "sphere": item["name"],
                "material": item["material"],
                "samples": int(np.count_nonzero(roi)),
                "mean_rsp": float(np.mean(values)),
                "std_rsp": float(np.std(values)),
                "reference_rsp": reference,
                "error_percent": error_percent,
                "absolute_rsp_error": abs(float(np.mean(values)) - reference),
                "cnr": contrast / max(float(np.std(water_values)), 1e-12),
                "centroid_error_mm": centroid_error,
                "equivalent_volume_mm3": float(np.count_nonzero(segmented) * voxel_volume),
                "reference_volume_mm3": float(4.0 * np.pi * (float(item["diameter_mm"]) / 2) ** 3 / 3.0),
            }
        )
    edge_rows = sphere_edge_widths(image, config, simulation, mlic)
    finite_edges = [row["edge_width_10_90_mm"] for row in edge_rows if np.isfinite(row["edge_width_10_90_mm"])]
    edge_by_axis = {
        axis: [
            row["edge_width_10_90_mm"]
            for row in edge_rows
            if row["axis"] == axis and np.isfinite(row["edge_width_10_90_mm"])
        ]
        for axis in ("x", "y", "z")
    }
    edge_axis_mean = {
        axis: float(np.mean(values)) if values else float("nan")
        for axis, values in edge_by_axis.items()
    }
    finite_axis_means = [
        value for value in edge_axis_mean.values() if np.isfinite(value)
    ]
    metrics = {
        "water_mean_rsp": float(np.mean(water_values)),
        "water_bias": float(np.mean(water_values) - float(mlic["Water"])),
        "water_std_rsp": float(np.std(water_values)),
        "phantom_rmse": float(np.sqrt(np.mean((image[support] - truth[support]) ** 2))),
        "nonair_material_mape_percent": float(np.mean(absolute_nonair)),
        "nonair_material_max_error_percent": float(np.max(absolute_nonair)),
        "large_material_mape_percent": float(np.mean(absolute_large)),
        "edge_width_mean_mm": float(np.mean(finite_edges)),
        "edge_width_x_mm": edge_axis_mean["x"],
        "edge_width_y_mm": edge_axis_mean["y"],
        "edge_width_z_mm": edge_axis_mean["z"],
        "edge_width_anisotropy_ratio": (
            float(max(finite_axis_means) / min(finite_axis_means))
            if finite_axis_means and min(finite_axis_means) > 0
            else float("nan")
        ),
        "outside_nonzero": int(np.count_nonzero(image[~support])),
        "nonfinite": int(np.count_nonzero(~np.isfinite(image))),
    }
    return metrics, material_rows, edge_rows
