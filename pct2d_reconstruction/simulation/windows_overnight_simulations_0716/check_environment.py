#!/usr/bin/env python3
"""Check the native-Windows environment and all overnight configurations."""

from __future__ import annotations

import json
import math
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy
import opengate
import scipy
import uproot

from run_ct_angle import load_config, validate_config
from run_material_case import enumerate_cases, load_config as load_material_config


HERE = Path(__file__).resolve().parent


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def rectangle(center, size, angle_deg):
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    hx, hy = float(size[0]) / 2.0, float(size[1]) / 2.0
    corners = []
    for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
        corners.append(numpy.asarray([center[0] + c * dx - s * dy,
                                      center[1] + s * dx + c * dy]))
    return corners


def rectangles_overlap(first, second, tolerance=1e-8):
    for polygon in (first, second):
        for index in range(4):
            edge = polygon[(index + 1) % 4] - polygon[index]
            axis = numpy.asarray([-edge[1], edge[0]])
            axis /= numpy.linalg.norm(axis)
            a = [float(numpy.dot(point, axis)) for point in first]
            b = [float(numpy.dot(point, axis)) for point in second]
            if max(a) <= min(b) + tolerance or max(b) <= min(a) + tolerance:
                return False
    return True


def validate_layout(config: dict) -> dict:
    kind = config["phantom_kind"]
    cylinders = []
    rectangles = []
    if kind == "aluminium_spiral":
        for index, radius in enumerate(config["insert_radii_mm"]):
            angle = math.radians(float(config["insert_angle_step_deg"]) * index)
            cylinders.append((f"spiral_{index}", numpy.asarray([radius * math.cos(angle), radius * math.sin(angle)]),
                              float(config["insert_diameter_mm"]) / 2.0))
    elif kind == "material_calibration":
        materials = config["calibration_materials"]
        for ring_index, ring in enumerate(config["calibration_rings"]):
            for material_index, material in enumerate(materials):
                angle = math.radians(float(ring["angle_offset_deg"]) + material_index * 360.0 / len(materials))
                center = numpy.asarray([float(ring["radius_mm"]) * math.cos(angle),
                                        float(ring["radius_mm"]) * math.sin(angle)])
                cylinders.append((f"ring{ring_index}_{material}", center, float(ring["diameter_mm"]) / 2.0))
        cylinders.append(("small_aluminium", numpy.asarray([0.0, 0.0]),
                          float(config["small_aluminium_diameter_mm"]) / 2.0))
    elif kind == "resolution":
        for group_index, group in enumerate(config["line_pair_groups"]):
            width = float(group["line_width_mm"]); count = int(group["bar_count"])
            total = (2 * count - 1) * width
            angle = math.radians(float(group.get("rotation_deg", 0.0)))
            for bar_index in range(count):
                offset = -0.5 * total + 0.5 * width + 2.0 * width * bar_index
                center = [float(group["center_mm"][0]) + math.cos(angle) * offset,
                          float(group["center_mm"][1]) + math.sin(angle) * offset]
                rectangles.append((f"line{group_index}_{bar_index}", rectangle(
                    center, [width, float(group["bar_length_mm"])], group.get("rotation_deg", 0.0))))
        for index, target in enumerate(config["edge_targets"]):
            rectangles.append((f"edge_{index}", rectangle(
                target["center_mm"], target["size_xy_mm"], target["rotation_deg"])))
    for name, center, radius in cylinders:
        if numpy.linalg.norm(center) + radius > float(config["phantom_radius_mm"]) + 1e-8:
            raise ValueError(f"{name} extends outside phantom")
    for index, (name, center, radius) in enumerate(cylinders):
        for other_name, other_center, other_radius in cylinders[index + 1:]:
            if numpy.linalg.norm(center - other_center) < radius + other_radius - 1e-8:
                raise ValueError(f"cylinders overlap: {name}, {other_name}")
    for name, polygon in rectangles:
        if max(float(numpy.linalg.norm(point)) for point in polygon) > float(config["phantom_radius_mm"]) + 1e-8:
            raise ValueError(f"{name} extends outside phantom")
    for index, (name, polygon) in enumerate(rectangles):
        for other_name, other_polygon in rectangles[index + 1:]:
            if rectangles_overlap(polygon, other_polygon):
                raise ValueError(f"rectangles overlap: {name}, {other_name}")
    return {"cylinders": len(cylinders), "rectangles": len(rectangles), "overlaps": 0}


def main() -> None:
    errors = []
    opengate_version = version("opengate")
    core_version = version("opengate-core")
    if opengate_version != "10.1.0":
        errors.append(f"expected opengate 10.1.0, found {opengate_version}")
    if core_version != "10.1.0":
        errors.append(f"expected opengate-core 10.1.0, found {core_version}")
    material_db = opengate.utility.get_contrib_path() / "GateMaterials.db"
    material_text = material_db.read_text(encoding="utf-8", errors="replace")
    required_materials = ["Vacuum", "Air", "Water", "Lung", "A150_Tissue_Plastic", "SpineBone", "Aluminium"]
    missing_materials = [name for name in required_materials if f"{name}:" not in material_text]
    if missing_materials:
        errors.append(f"materials missing from GateMaterials.db: {missing_materials}")
    scenarios = []
    for path in sorted((HERE / "scenarios").glob("s*.json")):
        config = load_config(path)
        validate_config(config, 0, int(config["protons_per_projection"]))
        layout = validate_layout(config)
        scenarios.append({
            "scenario_id": config["scenario_id"], "output_name": config["output_name"],
            "projections": config["projections"],
            "protons_per_projection": config["protons_per_projection"],
            "world_material": config["world_material"], "phantom_kind": config["phantom_kind"],
            "layout": layout,
        })
    material_config = load_material_config(HERE / "material_scan_config.json")
    material_cases = enumerate_cases(material_config)
    result = {
        "status": "PASS" if not errors else "FAIL", "errors": errors,
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "opengate": opengate_version, "opengate_core": core_version,
        "numpy": numpy.__version__, "scipy": scipy.__version__, "uproot": uproot.__version__,
        "material_database": str(material_db), "required_materials": required_materials,
        "ct_scenarios": scenarios, "material_scan_case_count": len(material_cases),
    }
    qc = HERE / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "environment_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
