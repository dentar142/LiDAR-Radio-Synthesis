"""Streaming OBJ statistics and orthographic height evidence."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


Bounds3D = tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class ObjStats:
    vertex_count: int
    triangle_count: int
    bounds: Bounds3D


@dataclass(frozen=True)
class HeightRaster:
    dtm: np.ndarray
    representative: np.ndarray
    dsm: np.ndarray
    bounds: Bounds3D
    resolution_m: float
    z_bin_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.dsm.shape


def iter_vertices(path: Path) -> Iterator[tuple[float, float, float]]:
    """Yield OBJ vertices without retaining the source mesh in memory."""

    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not (line.startswith(b"v ") or line.startswith(b"v\t")):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"invalid OBJ vertex at {path}:{line_number}")
            yield float(fields[1]), float(fields[2]), float(fields[3])


def scan_obj(path: Path) -> ObjStats:
    """Count vertices/triangles and find bounds in a streaming pass."""

    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertices = triangles = 0
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith(b"v ") or line.startswith(b"v\t"):
                fields = line.split()
                if len(fields) < 4:
                    raise ValueError(f"invalid OBJ vertex at {path}:{line_number}")
                value = (float(fields[1]), float(fields[2]), float(fields[3]))
                vertices += 1
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], value[axis])
                    maximum[axis] = max(maximum[axis], value[axis])
            elif line.startswith(b"f ") or line.startswith(b"f\t"):
                triangles += max(0, len(line.split()) - 3)
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {path}")
    return ObjStats(vertices, triangles, (tuple(minimum), tuple(maximum)))  # type: ignore[arg-type]


def rasterize_height(
    path: Path,
    resolution_m: float,
    z_bin_size_m: float,
    stats: ObjStats | None = None,
) -> HeightRaster:
    """Build DTM, lower-median representative height and DSM arrays."""

    if resolution_m <= 0 or z_bin_size_m <= 0:
        raise ValueError("raster resolution and Z-bin size must be positive")
    source_stats = stats or scan_obj(path)
    (min_x, min_y, _), (max_x, max_y, _) = source_stats.bounds
    columns = max(1, math.ceil((max_x - min_x) / resolution_m))
    rows = max(1, math.ceil((max_y - min_y) / resolution_m))
    dtm = np.full((rows, columns), np.nan, dtype=np.float32)
    dsm = np.full((rows, columns), np.nan, dtype=np.float32)
    cell_ids = array("I")
    z_bins = array("f")
    for x, y, z in iter_vertices(path):
        column = min(columns - 1, int((x - min_x) // resolution_m))
        row = min(rows - 1, int((max_y - y) // resolution_m))
        if np.isnan(dtm[row, column]) or z < dtm[row, column]:
            dtm[row, column] = z
        if np.isnan(dsm[row, column]) or z > dsm[row, column]:
            dsm[row, column] = z
        cell_ids.append(row * columns + column)
        z_bins.append(math.floor(z / z_bin_size_m) * z_bin_size_m)
    ids = np.frombuffer(cell_ids, dtype=np.uint32)
    bins = np.frombuffer(z_bins, dtype=np.float32)
    order = np.lexsort((bins, ids))
    sorted_ids = ids[order]
    sorted_bins = bins[order]
    unique_ids, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    representative = np.full((rows, columns), np.nan, dtype=np.float32)
    representative.flat[unique_ids] = sorted_bins[starts + (counts - 1) // 2]
    return HeightRaster(dtm, representative, dsm, source_stats.bounds, resolution_m, z_bin_size_m)


def save_height_raster(raster: HeightRaster, stats: ObjStats, source: Path, output: Path) -> dict[str, object]:
    """Persist arrays, previews and their coordinate contract."""

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "source_dtm.npy", raster.dtm, allow_pickle=False)
    np.save(output / "source_dsm.npy", raster.dsm, allow_pickle=False)
    finite = np.isfinite(raster.representative)
    preview = np.zeros(raster.shape, dtype=np.uint8)
    if np.any(finite):
        low, high = np.percentile(raster.representative[finite], [1.0, 99.0])
        high = max(float(high), float(low) + 1.0)
        preview[finite] = np.rint(
            np.clip((raster.representative[finite] - low) / (high - low), 0.0, 1.0) * 254.0 + 1.0
        ).astype(np.uint8)
    else:
        low = high = None
    cv2.imwrite(str(output / "source_height.png"), preview)
    gradient_y, gradient_x = np.gradient(preview.astype(np.float32))
    magnitude = np.hypot(gradient_x, gradient_y)
    positive = magnitude[magnitude > 0]
    threshold = float(np.percentile(positive, 75.0)) if positive.size else math.inf
    edges = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    cv2.imwrite(str(output / "source_edges.png"), edges)
    metadata = {
        "schema": "hkustgz.ortho_height.v1",
        "source_obj": str(source.resolve()),
        "vertex_count": stats.vertex_count,
        "triangle_count": stats.triangle_count,
        "bounds_m": [list(stats.bounds[0]), list(stats.bounds[1])],
        "grid": {
            "shape_rows_columns": list(raster.shape),
            "resolution_m": raster.resolution_m,
            "z_bin_size_m": raster.z_bin_size_m,
            "row_zero": "maximum_y",
            "column_zero": "minimum_x",
            "occupied_cells": int(np.count_nonzero(finite)),
        },
        "display_stretch_z_m": [None if low is None else float(low), None if high is None else float(high)],
    }
    (output / "source_ortho_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def grid_to_world(
    columns_rows: np.ndarray, bounds: Bounds3D, resolution_m: float
) -> np.ndarray:
    """Convert floating-point ``(column,row)`` grid coordinates to world XY."""

    (min_x, _, _), (_, max_y, _) = bounds
    result = np.empty_like(columns_rows, dtype=np.float64)
    result[:, 0] = min_x + (columns_rows[:, 0] + 0.5) * resolution_m
    result[:, 1] = max_y - (columns_rows[:, 1] + 0.5) * resolution_m
    return result
