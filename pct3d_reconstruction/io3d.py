"""Small, explicit binary formats used by Stage 8."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_pairs(path: Path, pairs: np.ndarray) -> None:
    pairs = np.asarray(pairs, dtype="<f4")
    if pairs.ndim != 3 or pairs.shape[1:] != (5, 3):
        raise ValueError("pairs must have shape (N,5,3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw")
    raw_temporary = raw.with_suffix(raw.suffix + ".tmp")
    header_temporary = path.with_suffix(path.suffix + ".tmp")
    pairs.tofile(raw_temporary)
    raw_temporary.replace(raw)
    header_temporary.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 2",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 1",
                "Offset = 0 0",
                "CenterOfRotation = 0 0",
                "ElementSpacing = 1 1",
                f"DimSize = 5 {len(pairs)}",
                "AnatomicalOrientation = ??",
                "ElementNumberOfChannels = 3",
                "ElementType = MET_FLOAT",
                f"ElementDataFile = {raw.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    header_temporary.replace(path)


def _header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_pairs(path: Path, mmap: bool = False) -> np.ndarray:
    head = _header(path)
    size = [int(value) for value in head["DimSize"].split()]
    if size[0] != 5 or head.get("ElementNumberOfChannels") != "3":
        raise ValueError(f"unexpected pairs header: {path}")
    raw = path.parent / head["ElementDataFile"]
    data = np.memmap(raw, "<f4", mode="r") if mmap else np.fromfile(raw, "<f4")
    result = data.reshape(size[1], 5, 3)
    return result if mmap else np.asarray(result)


def write_volume(
    path: Path,
    volume_zyx: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    origin_xyz: tuple[float, float, float],
) -> None:
    volume = np.asarray(volume_zyx, dtype="<f4")
    if volume.ndim != 3:
        raise ValueError("volume must have shape (z,y,x)")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.with_suffix(".raw")
    raw_temporary = raw.with_suffix(raw.suffix + ".tmp")
    header_temporary = path.with_suffix(path.suffix + ".tmp")
    volume.tofile(raw_temporary)
    raw_temporary.replace(raw)
    nz, ny, nx = volume.shape
    header_temporary.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "NDims = 3",
                "BinaryData = True",
                "BinaryDataByteOrderMSB = False",
                "CompressedData = False",
                "TransformMatrix = 1 0 0 0 1 0 0 0 1",
                f"Offset = {origin_xyz[0]:.12g} {origin_xyz[1]:.12g} {origin_xyz[2]:.12g}",
                "CenterOfRotation = 0 0 0",
                f"ElementSpacing = {spacing_xyz[0]:.12g} {spacing_xyz[1]:.12g} {spacing_xyz[2]:.12g}",
                f"DimSize = {nx} {ny} {nz}",
                "AnatomicalOrientation = RAI",
                "ElementType = MET_FLOAT",
                f"ElementDataFile = {raw.name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    header_temporary.replace(path)


def read_volume(path: Path, mmap: bool = False) -> tuple[np.ndarray, list[float], list[float]]:
    head = _header(path)
    nx, ny, nz = (int(value) for value in head["DimSize"].split())
    raw = path.parent / head["ElementDataFile"]
    data = np.memmap(raw, "<f4", mode="r") if mmap else np.fromfile(raw, "<f4")
    volume = data.reshape(nz, ny, nx)
    return volume, [float(x) for x in head["ElementSpacing"].split()], [
        float(x) for x in head["Offset"].split()
    ]


def write_partition(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.uint8)
    if np.any(values > 2):
        raise ValueError("partition codes must be 0, 1, or 2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            validation=np.packbits(values == 1, bitorder="little"),
            test=np.packbits(values == 2, bitorder="little"),
            length=np.int64(len(values)),
        )
    temporary.replace(path)


def read_partition(path: Path) -> np.ndarray:
    with np.load(path) as data:
        length = int(data["length"])
        result = np.zeros(length, dtype=np.uint8)
        result[np.unpackbits(data["validation"], bitorder="little")[:length].astype(bool)] = 1
        result[np.unpackbits(data["test"], bitorder="little")[:length].astype(bool)] = 2
        return result


def batch_slices(length: int, size: int):
    for begin in range(0, length, size):
        yield slice(begin, min(begin + size, length))


def pair_batch(pairs: np.ndarray, selection: np.ndarray | slice) -> dict[str, Any]:
    p = np.asarray(pairs[selection])
    return {
        "position_in": p[:, 0],
        "position_out": p[:, 1],
        "direction_in": p[:, 2],
        "direction_out": p[:, 3],
        "wepl_mm": p[:, 4, 1],
    }
