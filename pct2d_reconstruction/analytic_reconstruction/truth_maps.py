#!/usr/bin/env python3
"""Voxelize RED and 200 MeV RSP truth on the retained FDK grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "simulation" / "truth_geometry_definition.json"
REFERENCE = HERE / "recon" / "recon_ddb_nohann.mhd"


def read_header(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_mhd(path: Path, data: np.ndarray, spacing: list[float], origin: list[float]) -> None:
    raw = path.with_suffix(".raw")
    np.asarray(data, dtype="<f4").tofile(raw)
    size = [data.shape[2], data.shape[1], data.shape[0]]
    path.write_text(
        "ObjectType = Image\n"
        "NDims = 3\n"
        "BinaryData = True\n"
        "BinaryDataByteOrderMSB = False\n"
        "CompressedData = False\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        f"Offset = {' '.join(f'{v:.12g}' for v in origin)}\n"
        "CenterOfRotation = 0 0 0\n"
        "AnatomicalOrientation = RAI\n"
        f"ElementSpacing = {' '.join(f'{v:.12g}' for v in spacing)}\n"
        f"DimSize = {' '.join(map(str, size))}\n"
        "ElementType = MET_FLOAT\n"
        f"ElementDataFile = {raw.name}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "truth")
    parser.add_argument("--supersampling", type=int, default=8)
    parser.add_argument(
        "--write-fractions",
        action="store_true",
        help="Also retain water/aluminium fraction images for diagnostic use.",
    )
    parser.add_argument(
        "--flip-z-centers",
        action="store_true",
        help="Reflect insert centers across z=0 for source-side/RTK geometry audits.",
    )
    args = parser.parse_args()
    if args.supersampling < 1:
        parser.error("--supersampling must be positive")

    header = read_header(args.reference)
    size = [int(v) for v in header["DimSize"].split()]
    spacing = [float(v) for v in header["ElementSpacing"].split()]
    origin_text = header.get("Offset", header.get("Origin"))
    if origin_text is None:
        raise RuntimeError("reference MHD has neither Offset nor Origin")
    origin = [float(v) for v in origin_text.split()]
    if size[1] != 1:
        raise RuntimeError(f"expected a single-slice reference, got {size}")

    definition = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    geometry = definition["geometry"]
    materials = definition["materials"]
    radius = float(geometry["phantom_radius_mm"])
    insert_radius = float(geometry["insert_radius_mm"])
    centers = geometry["insert_centers_xz_mm"]
    if args.flip_z_centers:
        centers = [
            {**item, "z": -float(item["z"])}
            for item in centers
        ]

    x = origin[0] + np.arange(size[0], dtype=np.float64) * spacing[0]
    z = origin[2] + np.arange(size[2], dtype=np.float64) * spacing[2]
    water_fraction = np.zeros((size[2], size[0]), dtype=np.float32)
    aluminium_fraction = np.zeros_like(water_fraction)
    offsets = (np.arange(args.supersampling, dtype=np.float64) + 0.5) / args.supersampling - 0.5

    # Supersample only the central x-z plane; each sample receives an exact material label.
    for oz in offsets:
        zz2 = (z + oz * spacing[2])[:, None] ** 2
        for ox in offsets:
            water_fraction += (
                (x + ox * spacing[0])[None, :] ** 2 + zz2 <= radius * radius
            )
    water_fraction /= float(args.supersampling**2)

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
                    <= insert_radius * insert_radius
                )
        local /= float(args.supersampling**2)
        aluminium_fraction[np.ix_(iz, ix)] += local
        water_fraction[np.ix_(iz, ix)] -= local

    np.clip(water_fraction, 0.0, 1.0, out=water_fraction)
    np.clip(aluminium_fraction, 0.0, 1.0, out=aluminium_fraction)
    red = (
        water_fraction * float(materials["Water"]["red"])
        + aluminium_fraction * float(materials["Aluminium"]["red"])
    )
    rsp = (
        water_fraction * float(materials["Water"]["rsp_200mev"])
        + aluminium_fraction * float(materials["Aluminium"]["rsp_200mev"])
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "truth_red": red,
        "truth_rsp_200mev": rsp,
    }
    if args.write_fractions:
        arrays.update(
            {
                "truth_water_fraction": water_fraction,
                "truth_aluminium_fraction": aluminium_fraction,
            }
        )
    for name, array in arrays.items():
        write_mhd(output / f"{name}.mhd", array[:, None, :], spacing, origin)

    metadata = {
        "reference": str(args.reference.resolve()),
        "definition": str(GEOMETRY.resolve()),
        "size": size,
        "spacing_mm": spacing,
        "origin_mm": origin,
        "supersampling_per_axis": args.supersampling,
        "subsamples_per_voxel": args.supersampling**2,
        "flip_z_centers": args.flip_z_centers,
        "water_fraction_sum": float(water_fraction.sum(dtype=np.float64)),
        "aluminium_fraction_sum": float(aluminium_fraction.sum(dtype=np.float64)),
        "red_min_max": [float(red.min()), float(red.max())],
        "rsp_200mev_min_max": [float(rsp.min()), float(rsp.max())],
        "finite": bool(all(np.isfinite(value).all() for value in arrays.values())),
        "outputs": {name: str((output / f"{name}.mhd").resolve()) for name in arrays},
    }
    (output / "truth_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    raise SystemExit("internal truth module; use run_analytic_reconstruction.py")
