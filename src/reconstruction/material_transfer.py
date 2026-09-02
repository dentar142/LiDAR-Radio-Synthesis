"""Transfer original textured-LiDAR semantic evidence to clean surfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .mesh import CleanMesh


VOXEL_DTYPE = np.dtype([
    ("ix", "<i4"), ("iy", "<i4"), ("iz", "<i4"),
    ("label", "u1"), ("confidence", "u1"), ("count", "<u2"),
])

SOURCE_CLASSES = {
    0: "unknown_surface",
    1: "water",
    2: "grass",
    3: "soil",
    4: "vegetation",
    5: "asphalt",
    6: "roof_surface",
    7: "concrete_exterior_wall",
    8: "glass_facade",
    9: "other_building_surface",
}


@dataclass(frozen=True)
class SurfaceMaterial:
    surface_id: str
    surface_type: str
    label: str
    obj_material: str
    confidence: float
    source_hit_rate: float
    top_vote_share: float
    evidence_samples: int
    status: str
    provenance: str
    review_reasons: tuple[str, ...]


def load_semantic_voxels(paths: Sequence[Path]) -> tuple[dict[tuple[int, int, int], tuple[int, int, int]], dict[str, object]]:
    """Load one or more compact original-mesh semantic voxel indexes."""

    aggregate: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    counts: dict[str, int] = {}
    for path in paths:
        records = np.fromfile(path, dtype=VOXEL_DTYPE)
        counts[str(path)] = int(len(records))
        for row in records:
            key = int(row["ix"]), int(row["iy"]), int(row["iz"])
            aggregate[key].append((int(row["label"]), int(row["confidence"]), int(row["count"])))
    lookup: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for key, values in aggregate.items():
        scores: dict[int, float] = defaultdict(float)
        total_counts: dict[int, int] = defaultdict(int)
        confidences: dict[int, float] = defaultdict(float)
        for label, confidence, count in values:
            scores[label] += max(1, count) * max(1, confidence)
            total_counts[label] += count
            confidences[label] += confidence * max(1, count)
        label = max(scores, key=lambda candidate: (scores[candidate], -candidate))
        count = total_counts[label]
        confidence = round(confidences[label] / max(1, count))
        lookup[key] = label, confidence, min(65535, count)
    return lookup, {"source_files": counts, "combined_voxels": len(lookup)}


def transfer_surface_materials(
    mesh: CleanMesh,
    voxel_lookup: Mapping[tuple[int, int, int], tuple[int, int, int]],
    config: Mapping[str, object],
) -> tuple[list[SurfaceMaterial], list[str], dict[str, object]]:
    """Aggregate source evidence per connected clean-model surface."""

    cell_m = float(config.get("cell_m", 1.0))
    radius = int(config.get("neighbor_cells", 2))
    maximum_samples = int(config.get("maximum_samples_per_surface", 400))
    minimum_hit_rate = float(config.get("minimum_hit_rate", 0.35))
    minimum_confidence = float(config.get("minimum_confidence", 0.45))
    minimum_share = float(config.get("minimum_top_vote_share", 0.60))
    grouped: dict[str, list[int]] = defaultdict(list)
    for face_id, surface_id in enumerate(mesh.face_surface_ids):
        grouped[surface_id].append(face_id)
    rows: list[SurfaceMaterial] = []
    material_by_surface: dict[str, str] = {}
    for surface_id in sorted(grouped):
        face_ids = grouped[surface_id]
        surface_type = mesh.face_surface_types[face_ids[0]]
        samples = _surface_samples(mesh, face_ids, maximum_samples)
        votes: Counter[int] = Counter()
        confidence_sum: dict[int, float] = defaultdict(float)
        accepted_counts: Counter[int] = Counter()
        hits = 0
        for point in samples:
            hit = _nearest_voxel(point, voxel_lookup, cell_m, radius)
            if hit is None:
                continue
            label, confidence, count, distance = hit
            hits += 1
            if not _label_allowed(surface_type, label):
                continue
            weight = max(1, min(count, 100)) * (confidence / 255.0) / (1.0 + distance)
            votes[label] += weight
            confidence_sum[label] += confidence / 255.0
            accepted_counts[label] += 1
        reasons: list[str] = []
        hit_rate = hits / len(samples) if samples else 0.0
        if votes:
            ranked = votes.most_common()
            label_id, top_score = ranked[0]
            vote_total = sum(votes.values())
            share = float(top_score / vote_total) if vote_total else 0.0
            mean_source_confidence = confidence_sum[label_id] / max(1, accepted_counts[label_id])
            confidence = hit_rate * share * mean_source_confidence
            label = SOURCE_CLASSES[label_id]
            provenance = "original_textured_mesh_segformer_v3_voxel_projection"
        else:
            label = _fallback_label(surface_type, config)
            share = 0.0
            confidence = 0.25 if surface_type in {"roof", "facade"} else 0.10
            provenance = "geometry_fallback_no_compatible_source_votes"
            reasons.append("NO_COMPATIBLE_SOURCE_SEMANTIC_VOTES")
        if hit_rate < minimum_hit_rate:
            reasons.append("SOURCE_ASSOCIATION_COVERAGE_LOW")
        if share < minimum_share:
            reasons.append("SOURCE_LABEL_PURITY_LOW")
        if confidence < minimum_confidence:
            reasons.append("COMBINED_CONFIDENCE_LOW")
        status = "AUTO_ACCEPTED_VISUAL_PRIOR" if not reasons else "REVIEW_REQUIRED"
        obj_material = _obj_material(label, surface_type)
        row = SurfaceMaterial(
            surface_id, surface_type, label, obj_material, round(confidence, 6),
            round(hit_rate, 6), round(share, 6), len(samples), status, provenance,
            tuple(dict.fromkeys(reasons)),
        )
        rows.append(row)
        material_by_surface[surface_id] = obj_material
    face_materials = [material_by_surface[surface_id] for surface_id in mesh.face_surface_ids]
    accepted = sum(row.status.startswith("AUTO_ACCEPTED") for row in rows)
    audit = {
        "status": "PASS_AUTOMATIC_CANDIDATES" if rows else "NO_GO_NO_SURFACES",
        "surface_count": len(rows),
        "auto_accepted_visual_prior": accepted,
        "review_required": len(rows) - accepted,
        "mean_source_hit_rate": round(float(np.mean([row.source_hit_rate for row in rows])), 4) if rows else 0.0,
        "label_counts": dict(Counter(row.label for row in rows)),
        "claim_boundary": "visual/geometry material candidates only; no measured EM parameters or construction truth",
    }
    return rows, face_materials, audit


def write_surface_materials(rows: Sequence[SurfaceMaterial], path: Path, source_audit: Mapping[str, object]) -> Path:
    payload = {
        "schema": "hkustgz.surface_material_transfer.v1",
        "source_voxel_index": dict(source_audit),
        "claim_boundary": "automatic candidates are not field-calibrated electromagnetic materials",
        "surfaces": [row.__dict__ for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _surface_samples(mesh: CleanMesh, face_ids: Sequence[int], maximum: int) -> list[tuple[float, float, float]]:
    candidates: list[tuple[float, float, float]] = []
    for face_id in face_ids:
        points = [mesh.vertices[index] for index in mesh.faces[face_id]]
        candidates.extend(points)
        candidates.append(tuple(sum(point[axis] for point in points) / 3.0 for axis in range(3)))  # type: ignore[arg-type]
        candidates.extend(tuple((points[a][axis] + points[b][axis]) / 2.0 for axis in range(3)) for a, b in ((0, 1), (1, 2), (2, 0)))
    unique = list(dict.fromkeys(candidates))
    if len(unique) <= maximum:
        return unique
    return [unique[min(len(unique) - 1, int(index * len(unique) / maximum))] for index in range(maximum)]


def _nearest_voxel(
    point: tuple[float, float, float],
    lookup: Mapping[tuple[int, int, int], tuple[int, int, int]],
    cell_m: float,
    radius: int,
) -> tuple[int, int, int, float] | None:
    base = tuple(math.floor(value / cell_m) for value in point)
    best: tuple[float, int, int, int] | None = None
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                value = lookup.get((base[0] + dx, base[1] + dy, base[2] + dz))
                if value is None:
                    continue
                label, confidence, count = value
                distance = math.sqrt(dx * dx + dy * dy + dz * dz) * cell_m
                candidate = distance, -count, -confidence, label
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return None
    distance, negative_count, negative_confidence, label = best
    return label, -negative_confidence, -negative_count, distance


def _label_allowed(surface_type: str, label: int) -> bool:
    if surface_type == "facade":
        return label in {7, 8, 9}
    if surface_type == "roof":
        return label in {6, 7, 9}
    return label in {7, 9}


def _fallback_label(surface_type: str, config: Mapping[str, object]) -> str:
    fallbacks = config.get("fallback_labels", {})
    if isinstance(fallbacks, Mapping) and surface_type in fallbacks:
        return str(fallbacks[surface_type])
    return {"roof": "roof_surface", "facade": "other_building_surface", "base": "structural_base"}.get(surface_type, "unknown_surface")


def _obj_material(label: str, surface_type: str) -> str:
    mapping = {
        "glass_facade": "AUTO_GLASS_VISUAL_PRIOR",
        "concrete_exterior_wall": "AUTO_CONCRETE_VISUAL_PRIOR",
        "other_building_surface": "AUTO_OTHER_BUILDING_VISUAL_PRIOR",
        "roof_surface": "AUTO_ROOF_VISUAL_PRIOR",
        "structural_base": "AUTO_STRUCTURAL_BASE",
    }
    return mapping.get(label, f"AUTO_{surface_type.upper()}_REVIEW_REQUIRED")
