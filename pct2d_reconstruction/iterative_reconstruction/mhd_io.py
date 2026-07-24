"""Minimal MetaImage I/O helpers for the local iterative experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DTYPES = {
    "MET_FLOAT": np.dtype("<f4"),
    "MET_DOUBLE": np.dtype("<f8"),
    "MET_UINT": np.dtype("<u4"),
}


def read_header(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_pairs(path: Path) -> np.memmap:
    header = read_header(path)
    if header.get("DimSize", "").split()[0] != "5":
        raise ValueError(f"unexpected pair layout in {path}")
    rows = int(header["DimSize"].split()[1])
    channels = int(header.get("ElementNumberOfChannels", "1"))
    raw = path.parent / header["ElementDataFile"]
    return np.memmap(raw, dtype=DTYPES[header["ElementType"]], mode="r", shape=(rows, 5, channels))


def read_image_2d(path: Path) -> tuple[np.ndarray, list[float], list[float]]:
    header = read_header(path)
    size = [int(value) for value in header["DimSize"].split()]
    spacing = [float(value) for value in header["ElementSpacing"].split()]
    origin_text = header.get("Offset", header.get("Origin"))
    if origin_text is None:
        raise ValueError(f"missing origin in {path}")
    origin = [float(value) for value in origin_text.split()]
    raw = path.parent / header["ElementDataFile"]
    data = np.memmap(raw, dtype=DTYPES[header["ElementType"]], mode="r", shape=tuple(size[::-1]))
    if len(size) != 3 or size[1] != 1:
        raise ValueError(f"expected x-z single-slice image, got {size}")
    return np.asarray(data[:, 0, :]), spacing, origin


def write_image_2d(path: Path, image: np.ndarray, spacing_mm: float, origin_mm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw")
    np.asarray(image[:, None, :], dtype="<f4").tofile(raw)
    nz, nx = image.shape
    path.write_text(
        "ObjectType = Image\n"
        "NDims = 3\n"
        "BinaryData = True\n"
        "BinaryDataByteOrderMSB = False\n"
        "CompressedData = False\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        f"Offset = {origin_mm:.12g} 0 {origin_mm:.12g}\n"
        "CenterOfRotation = 0 0 0\n"
        "AnatomicalOrientation = RAI\n"
        f"ElementSpacing = {spacing_mm:.12g} 1 {spacing_mm:.12g}\n"
        f"DimSize = {nx} 1 {nz}\n"
        "ElementType = MET_FLOAT\n"
        f"ElementDataFile = {raw.name}\n",
        encoding="utf-8",
    )


def resample_to_grid(
    image: np.ndarray,
    source_spacing: list[float],
    source_origin: list[float],
    size: int,
    spacing_mm: float,
) -> tuple[np.ndarray, float]:
    from scipy.ndimage import map_coordinates

    origin = -0.5 * (size - 1) * spacing_mm
    coordinates = origin + np.arange(size, dtype=np.float64) * spacing_mm
    ix = (coordinates - source_origin[0]) / source_spacing[0]
    iz = (coordinates - source_origin[2]) / source_spacing[2]
    zz, xx = np.meshgrid(iz, ix, indexing="ij")
    sampled = map_coordinates(np.asarray(image), [zz, xx], order=1, mode="constant", cval=0.0)
    return sampled.astype(np.float32), origin
