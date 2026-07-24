#!/usr/bin/env python3
"""Run the retained no-Hann DDB-FDK reconstruction and integrated RSP QC."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPOSITORY = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT))

from common import load_experiment, path_for  # noqa: E402
from analytic_reconstruction import rsp_metrics, truth_maps  # noqa: E402


def command_path(name: str) -> Path:
    path = REPOSITORY / ".venv-gate" / "bin" / name
    if not path.is_file():
        raise FileNotFoundError(f"required executable not found: {path}")
    return path


def run(command: list[str]) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}:\n"
            f"{shlex.join(command)}\n{result.stdout}"
        )
    return elapsed, result.stdout


def generate_truth(
    definition_path: Path,
    reference: Path,
    output: Path,
    supersampling: int = 8,
) -> dict:
    header = truth_maps.read_header(reference)
    size = [int(value) for value in header["DimSize"].split()]
    spacing = [float(value) for value in header["ElementSpacing"].split()]
    origin_text = header.get("Offset", header.get("Origin"))
    if origin_text is None or size[1] != 1:
        raise RuntimeError(f"unexpected analytic reference grid: {reference}")
    origin = [float(value) for value in origin_text.split()]
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    geometry = definition["geometry"]
    materials = definition["materials"]
    radius = float(geometry["phantom_radius_mm"])
    insert_radius = float(geometry["insert_radius_mm"])
    centers = geometry["insert_centers_xz_mm"]

    x = origin[0] + np.arange(size[0], dtype=np.float64) * spacing[0]
    z = origin[2] + np.arange(size[2], dtype=np.float64) * spacing[2]
    water = np.zeros((size[2], size[0]), dtype=np.float32)
    aluminium = np.zeros_like(water)
    offsets = (np.arange(supersampling, dtype=np.float64) + 0.5) / supersampling - 0.5
    for oz in offsets:
        zz2 = (z + oz * spacing[2])[:, None] ** 2
        for ox in offsets:
            water += (x + ox * spacing[0])[None, :] ** 2 + zz2 <= radius**2
    water /= float(supersampling**2)
    for item in centers:
        cx, cz = float(item["x"]), float(item["z"])
        margin = insert_radius + max(spacing[0], spacing[2])
        ix = np.flatnonzero(np.abs(x - cx) <= margin)
        iz = np.flatnonzero(np.abs(z - cz) <= margin)
        local = np.zeros((len(iz), len(ix)), dtype=np.float32)
        for oz in offsets:
            zz2 = (z[iz] + oz * spacing[2] - cz)[:, None] ** 2
            for ox in offsets:
                local += (
                    (x[ix] + ox * spacing[0] - cx)[None, :] ** 2 + zz2
                    <= insert_radius**2
                )
        local /= float(supersampling**2)
        aluminium[np.ix_(iz, ix)] += local
        water[np.ix_(iz, ix)] -= local
    np.clip(water, 0.0, 1.0, out=water)
    np.clip(aluminium, 0.0, 1.0, out=aluminium)
    red = water * float(materials["Water"]["red"]) + aluminium * float(materials["Aluminium"]["red"])
    rsp = water * float(materials["Water"]["rsp_200mev"]) + aluminium * float(materials["Aluminium"]["rsp_200mev"])
    output.mkdir(parents=True, exist_ok=True)
    truth_maps.write_mhd(output / "truth_red.mhd", red[:, None, :], spacing, origin)
    truth_maps.write_mhd(output / "truth_rsp_200mev.mhd", rsp[:, None, :], spacing, origin)
    metadata = {
        "definition": str(definition_path),
        "reference": str(reference),
        "size": size,
        "spacing_mm": spacing,
        "origin_mm": origin,
        "supersampling_per_axis": supersampling,
        "finite": bool(np.isfinite(red).all() and np.isfinite(rsp).all()),
        "rsp_200mev_min_max": [float(rsp.min()), float(rsp.max())],
    }
    return metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="0716")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    projections = path_for(config, "preprocessing_data") / "projections_ddb"
    reconstruction = path_for(config, "reconstruction_data") / "analytic"
    recon = reconstruction / "recon"
    truth = reconstruction / "truth"
    qc = HERE / "qc" / f"results{args.experiment}"
    qc.mkdir(parents=True, exist_ok=True)
    recon.mkdir(parents=True, exist_ok=True)
    truth.mkdir(parents=True, exist_ok=True)
    output = recon / "recon_ddb_nohann.mhd"
    existing = [output, output.with_suffix(".raw"), truth / "truth_rsp_200mev.mhd"]
    if any(path.exists() for path in existing) and not args.force:
        raise FileExistsError(f"analytic output exists under {reconstruction}; use --force")
    if args.force:
        for directory in (recon, truth):
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()
    missing = [projections / f"proj{run_id:04d}.mhd" for run_id in range(720) if not (projections / f"proj{run_id:04d}.mhd").is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} DDB projections; first: {missing[0]}")

    analytic_config = config["analytic"]
    acquisition = config["acquisition"]
    geometry = reconstruction / "geometry.xml"
    geometry_command = [
        str(command_path("rtksimulatedgeometry")),
        "--nproj", str(acquisition["projections"]),
        "--first_angle", f"{acquisition['first_angle_deg']:g}",
        "--arc", f"{acquisition['arc_deg']:g}",
        "--sid", f"{acquisition['source_to_isocenter_mm']:g}",
        "--sdd", f"{acquisition['source_to_detector_mm']:g}",
        "--output", str(geometry),
    ]
    started = datetime.now()
    geometry_seconds, geometry_log = run(geometry_command)
    fdk_command = [
        str(command_path("pctfdk")), "--lowmem",
        "--geometry", str(geometry),
        "--path", str(projections),
        "--regexp", r"proj....\.mhd",
        "--output", str(output),
        "--dimension", str(analytic_config["grid_size"]), "1", str(analytic_config["grid_size"]),
        "--spacing", f"{analytic_config['grid_spacing_mm']:g}", "1", f"{analytic_config['grid_spacing_mm']:g}",
        "--hann", "0", "--verbose",
    ]
    print("Starting no-Hann DDB-FDK", flush=True)
    fdk_seconds, fdk_log = run(fdk_command)
    definition = path_for(config, "simulation_code") / "truth_geometry_definition.json"
    truth_metadata = generate_truth(definition, output, truth)

    image, x, z, grid = rsp_metrics.read_mhd(output)
    red, red_x, red_z, _ = rsp_metrics.read_mhd(truth / "truth_red.mhd")
    rsp, rsp_x, rsp_z, _ = rsp_metrics.read_mhd(truth / "truth_rsp_200mev.mhd")
    if not (np.array_equal(x, red_x) and np.array_equal(z, red_z) and np.array_equal(x, rsp_x) and np.array_equal(z, rsp_z)):
        raise RuntimeError("truth and reconstruction grids differ")
    definition_data = json.loads(definition.read_text(encoding="utf-8"))
    centers = definition_data["geometry"]["insert_centers_xz_mm"]
    metrics, insert_rows = rsp_metrics.metrics_for(image, red, rsp, x, z, centers)
    edge_rows = rsp_metrics.aluminium_edge_widths(image, x, z, centers)
    valid_widths = np.array([
        row["width_10_90_mm"]
        for row in edge_rows
        if row["distance_from_isocenter_mm"] > 0 and row["valid"]
    ])
    inner = np.array([row["inner_value"] for row in edge_rows])
    metrics.update({
        "aluminium_inner_mean": float(inner.mean()),
        "aluminium_platform_rsp_recovery": float(inner.mean() / 2.1189760409708303),
        "aluminium_edge_10_90_median_mm": float(np.median(valid_widths)),
        "aluminium_edge_10_90_min_mm": float(valid_widths.min()),
        "aluminium_edge_10_90_max_mm": float(valid_widths.max()),
    })
    write_csv(qc / "rsp_metrics.csv", [{"case": "ddb_nohann", **metrics}])
    write_csv(qc / "insert_metrics.csv", insert_rows)
    write_csv(qc / "edge_metrics.csv", edge_rows)
    finite = bool(np.isfinite(image).all())
    summary = {
        "status": "PASS" if finite and truth_metadata["finite"] and len(valid_widths) == 24 else "FAIL",
        "experiment": args.experiment,
        "started": started.isoformat(timespec="seconds"),
        "stopped": datetime.now().isoformat(timespec="seconds"),
        "method": "DDB-FDK no-Hann",
        "geometry_seconds": geometry_seconds,
        "fdk_seconds": fdk_seconds,
        "grid": grid,
        "finite": finite,
        "truth": truth_metadata,
        "rsp_metrics": metrics,
        "commands": {"geometry": geometry_command, "fdk": fdk_command},
        "outputs": {"reconstruction": str(output), "truth_rsp": str(truth / "truth_rsp_200mev.mhd")},
    }
    (qc / "analytic_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (qc / "analytic.log").write_text(
        f"geometry command: {shlex.join(geometry_command)}\n{geometry_log}\n"
        f"fdk command: {shlex.join(fdk_command)}\n{fdk_log}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "rsp_metrics": metrics}, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
