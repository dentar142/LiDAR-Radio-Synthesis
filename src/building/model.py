"""Stable building grouping from source object names and face heights."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

@dataclass(frozen=True)
class BuildingObject:
    """Stable building identifier and geometry references."""
    object_id: str
    face_ids: Tuple[int, ...]
    effective_height_m: float | None = None
    status: str = "AUTO_GROUPED_REVIEW_REQUIRED"


def group_buildings(
    face_ids: Sequence[int],
    object_ids: Sequence[str] | None = None,
    face_z_ranges: Sequence[tuple[float, float]] | None = None,
) -> Sequence[BuildingObject]:
    """Group faces by stable source object name, falling back to one scene group.

    Object names are provenance, not proof that an object is a real building. The
    caller must keep the returned review-required status unless an independent
    footprint or annotation confirms the grouping.
    """

    if object_ids is not None and len(object_ids) != len(face_ids):
        raise ValueError("object_ids length must match face_ids")
    if face_z_ranges is not None and len(face_z_ranges) != len(face_ids):
        raise ValueError("face_z_ranges length must match face_ids")

    grouped: dict[str, list[tuple[int, tuple[float, float] | None]]] = defaultdict(list)
    for offset, face_id in enumerate(face_ids):
        source_name = object_ids[offset].strip() if object_ids else "scene"
        source_name = source_name or "scene"
        z_range = face_z_ranges[offset] if face_z_ranges else None
        grouped[source_name].append((face_id, z_range))

    results: list[BuildingObject] = []
    used_ids: dict[str, int] = defaultdict(int)
    for source_name in sorted(grouped):
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_name).strip("_")
        safe = safe or "scene"
        used_ids[safe] += 1
        object_id = safe if used_ids[safe] == 1 else f"{safe}_{used_ids[safe]}"
        entries = grouped[source_name]
        ranges = [entry[1] for entry in entries if entry[1] is not None]
        height = None
        if ranges:
            height = max(value[1] for value in ranges) - min(value[0] for value in ranges)
        results.append(
            BuildingObject(
                object_id=object_id,
                face_ids=tuple(entry[0] for entry in entries),
                effective_height_m=height,
            )
        )
    return tuple(results)
