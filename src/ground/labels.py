"""Deterministic ground-surface candidate labeling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

class GroundLabel(str, Enum):
    """Ground classes kept separate from building materials."""
    TERRAIN = "terrain"
    ROAD = "road"
    WALKWAY = "walkway"
    WATER = "water"
    VEGETATION = "vegetation"
    UNKNOWN = "unknown"


def label_ground(
    face_ids: Sequence[int],
    semantic_categories: Sequence[str] | None = None,
    normal_z: Sequence[float] | None = None,
    centroid_z: Sequence[float] | None = None,
    low_surface_threshold_m: float | None = None,
) -> Sequence[GroundLabel]:
    """Assign conservative ground labels from semantics and geometry.

    Geometry-only horizontal low surfaces become generic terrain, never a
    specific visible material. Specific road/water/vegetation labels require a
    semantic category supplied by another evidence source.
    """

    count = len(face_ids)
    for values, name in (
        (semantic_categories, "semantic_categories"),
        (normal_z, "normal_z"),
        (centroid_z, "centroid_z"),
    ):
        if values is not None and len(values) != count:
            raise ValueError(f"{name} length must match face_ids")
    if low_surface_threshold_m is None and centroid_z:
        ordered = sorted(float(value) for value in centroid_z)
        low_surface_threshold_m = ordered[max(0, min(len(ordered) - 1, len(ordered) // 5))] + 1.0

    labels: list[GroundLabel] = []
    for index in range(count):
        category = semantic_categories[index].strip().lower() if semantic_categories else ""
        if category in {"road", "asphalt"}:
            label = GroundLabel.ROAD
        elif category in {"walkway", "stone", "stone_paving", "paving"}:
            label = GroundLabel.WALKWAY
        elif category == "water":
            label = GroundLabel.WATER
        elif category in {"vegetation", "grass"}:
            label = GroundLabel.VEGETATION
        elif category in {"ground", "terrain", "soil"}:
            label = GroundLabel.TERRAIN
        elif (
            normal_z is not None
            and centroid_z is not None
            and low_surface_threshold_m is not None
            and abs(normal_z[index]) >= 0.85
            and centroid_z[index] <= low_surface_threshold_m
        ):
            label = GroundLabel.TERRAIN
        else:
            label = GroundLabel.UNKNOWN
        labels.append(label)
    return tuple(labels)
