"""End-to-end conservative automation for one bounded scene."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import yaml

from .building import BuildingObject, group_buildings
from .export import SceneManifest, export_manifest
from .geometry import GeometryInput, GeometryModel, export_normalized_obj, reconstruct_geometry
from .georef import CoordinateAudit, audit_coordinates
from .ground import GroundLabel, label_ground
from .materials import MaterialCandidate, map_materials
from .semantic import (
    SegmentationLevel,
    SemanticEvidence,
    SemanticFusionResult,
    SemanticSource,
    fuse_semantics,
)


STAGES = ("geometry", "building", "semantic", "materials", "ground", "georef", "export")


@dataclass(frozen=True)
class PipelineResult:
    """Machine-readable completion result."""

    run_directory: Path
    status: str
    completed_stages: tuple[str, ...]
    outputs: Mapping[str, str]
    review_reasons: tuple[str, ...]


def load_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate a YAML configuration."""

    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    if not isinstance(payload.get("scene"), dict) or not payload["scene"].get("id"):
        raise ValueError("configuration must define scene.id")
    if not isinstance(payload.get("sources"), dict) or not payload["sources"].get("geometry"):
        raise ValueError("configuration must define sources.geometry")
    return payload


def run_pipeline(
    config_path: Path,
    output_directory: Path | None = None,
    stop_after: str = "export",
) -> PipelineResult:
    """Run all prerequisites through ``stop_after`` and export audit artifacts."""

    if stop_after not in STAGES:
        raise ValueError(f"unknown stage: {stop_after}")
    config_path = config_path.resolve()
    config = load_config(config_path)
    scene_id = str(config["scene"]["id"])
    geometry_path = _resolve_path(config_path, str(config["sources"]["geometry"]))
    annotation_path = _optional_path(config_path, config["sources"].get("annotations"))
    fingerprint = _fingerprint(config_path, geometry_path, annotation_path)
    if output_directory is None:
        output_root = _resolve_path(config_path, str(config.get("output", {}).get("root", "runs")))
        run_directory = output_root / scene_id / fingerprint[:12]
    else:
        run_directory = output_directory.resolve()
    run_directory.mkdir(parents=True, exist_ok=True)

    target_index = STAGES.index(stop_after)
    selected = STAGES[: target_index + 1]
    completed: list[str] = []
    outputs: dict[str, str] = {}
    review_reasons: list[str] = []
    timings: dict[str, float] = {}

    started = perf_counter()
    geometry = _timed(
        timings,
        "geometry",
        lambda: reconstruct_geometry(
            GeometryInput(geometry_path, str(config.get("coordinate_reference", {}).get("unit", "m")))
        ),
    )
    normalized_obj = export_normalized_obj(geometry, run_directory / "scene_normalized.obj")
    outputs["normalized_geometry"] = str(normalized_obj)
    _write_json(run_directory / "geometry.json", _geometry_payload(geometry))
    outputs["geometry_audit"] = str(run_directory / "geometry.json")
    completed.append("geometry")
    if target_index == 0:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    building_face_ids = [
        face_id
        for face_id, (object_name, material_name) in enumerate(zip(geometry.face_objects, geometry.face_materials))
        if not _is_ground_name(object_name) and not _is_ground_name(material_name)
    ]
    if not building_face_ids:
        building_face_ids = list(range(geometry.face_count))
        review_reasons.append("no building-specific object groups found; grouped the full scene")
    objects = [geometry.face_objects[face_id] for face_id in building_face_ids]
    z_ranges = [
        (
            min(geometry.vertices[index][2] for index in geometry.faces[face_id]),
            max(geometry.vertices[index][2] for index in geometry.faces[face_id]),
        )
        for face_id in building_face_ids
    ]
    buildings = _timed(
        timings,
        "building",
        lambda: group_buildings(building_face_ids, objects, z_ranges),
    )
    _write_json(run_directory / "buildings.json", {"buildings": _jsonable(buildings)})
    outputs["buildings"] = str(run_directory / "buildings.json")
    if any(item.status.endswith("REVIEW_REQUIRED") for item in buildings):
        review_reasons.append("source object groups are automatic building candidates")
    completed.append("building")
    if target_index == 1:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    annotations = _load_annotations(annotation_path, geometry.face_count)
    evidence = _build_evidence(geometry, annotations)
    semantics = _timed(timings, "semantic", lambda: fuse_semantics(evidence))
    semantic_by_face = _semantic_by_face(geometry.face_count, semantics)
    _write_json(
        run_directory / "semantics.json",
        {
            "status": "PASS" if semantics.passed else "REVIEW_REQUIRED",
            "regions": _jsonable(semantics.regions),
            "conflicts": _jsonable(semantics.conflicts),
        },
    )
    _write_face_csv(run_directory / "face_semantics.csv", semantic_by_face, "category")
    outputs["semantics"] = str(run_directory / "semantics.json")
    outputs["face_semantics"] = str(run_directory / "face_semantics.csv")
    if semantics.conflicts:
        review_reasons.append(f"{len(semantics.conflicts)} semantic entities have unresolved evidence")
    completed.append("semantic")
    if target_index == 2:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    overrides = config.get("material_parameter_overrides")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("material_parameter_overrides must be a mapping")
    material_candidates = _timed(
        timings,
        "materials",
        lambda: map_materials(semantic_by_face, overrides),
    )
    _write_json(
        run_directory / "materials.json",
        {
            "warning": "visual and engineering candidates are not measured EM material truth",
            "face_material_candidates": _jsonable(material_candidates),
        },
    )
    outputs["materials"] = str(run_directory / "materials.json")
    if any(candidate.provenance.startswith("UNCALIBRATED") for candidate in material_candidates):
        review_reasons.append("EM parameters remain uncalibrated visual/engineering priors")
    completed.append("materials")
    if target_index == 3:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    ground_labels = _timed(
        timings,
        "ground",
        lambda: label_ground(
            list(range(geometry.face_count)),
            semantic_by_face,
            [normal[2] for normal in geometry.face_normals],
            [centroid[2] for centroid in geometry.face_centroids],
        ),
    )
    ground_values = [label.value for label in ground_labels]
    _write_face_csv(run_directory / "face_ground_labels.csv", ground_values, "ground_label")
    _write_json(
        run_directory / "ground.json",
        {
            "counts": _counts(ground_values),
            "unknown_faces": ground_values.count(GroundLabel.UNKNOWN.value),
        },
    )
    outputs["ground"] = str(run_directory / "ground.json")
    outputs["face_ground_labels"] = str(run_directory / "face_ground_labels.csv")
    completed.append("ground")
    if target_index == 4:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    coordinate_metadata = {key: str(value) for key, value in config.get("coordinate_reference", {}).items()}
    coordinate_metadata["unit"] = "m"
    coordinate_audit = _timed(timings, "georef", lambda: audit_coordinates(coordinate_metadata))
    _write_json(run_directory / "coordinate_audit.json", _jsonable(coordinate_audit))
    outputs["coordinate_audit"] = str(run_directory / "coordinate_audit.json")
    if not coordinate_audit.passed or coordinate_audit.notes:
        review_reasons.extend(coordinate_audit.notes)
    completed.append("georef")
    if target_index == 5:
        return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)

    manifest = SceneManifest(
        scene_id=scene_id,
        inputs={
            "config": str(config_path),
            "geometry": str(geometry_path),
            "geometry_sha256": geometry.metadata["source_sha256"],
            **({"annotations": str(annotation_path)} if annotation_path else {}),
        },
        outputs=outputs,
        version=fingerprint,
    )
    export_manifest(manifest, run_directory / "scene_manifest.json")
    outputs["manifest"] = str(run_directory / "scene_manifest.json")
    timings["export"] = round(perf_counter() - started - sum(timings.values()), 6)
    completed.append("export")
    return _finish(run_directory, scene_id, fingerprint, completed, outputs, review_reasons, timings, started)


def _finish(
    run_directory: Path,
    scene_id: str,
    fingerprint: str,
    completed: Sequence[str],
    outputs: Mapping[str, str],
    review_reasons: Sequence[str],
    timings: Mapping[str, float],
    started: float,
) -> PipelineResult:
    unique_reasons = tuple(dict.fromkeys(reason for reason in review_reasons if reason))
    status = "PASS_AUTOMATED_PRIOR" if not unique_reasons else "REVIEW_REQUIRED"
    report = {
        "schema": "hkustgz.material_mapping.pipeline_report.v1",
        "scene_id": scene_id,
        "fingerprint": fingerprint,
        "status": status,
        "completed_stages": list(completed),
        "review_reasons": list(unique_reasons),
        "timings_seconds": dict(timings),
        "total_seconds": round(perf_counter() - started, 6),
        "outputs": dict(outputs),
        "claim_boundary": "automatic candidates only; not survey or measured EM truth",
    }
    _write_json(run_directory / "pipeline_report.json", report)
    final_outputs = dict(outputs)
    final_outputs["pipeline_report"] = str(run_directory / "pipeline_report.json")
    return PipelineResult(run_directory, status, tuple(completed), final_outputs, unique_reasons)


def _build_evidence(geometry: GeometryModel, annotations: Mapping[int, tuple[str, float]]) -> list[SemanticEvidence]:
    evidence: list[SemanticEvidence] = []
    height = geometry.bounds_max[2] - geometry.bounds_min[2]
    low_threshold = geometry.bounds_min[2] + max(1.0, height * 0.05)
    for face_id in range(geometry.face_count):
        entity_id = f"face:{face_id}"
        material_category = _category_from_name(geometry.face_materials[face_id])
        if material_category:
            evidence.append(
                SemanticEvidence(
                    entity_id, SegmentationLevel.FACE, material_category,
                    SemanticSource.EXISTING_MESH_SEMANTIC, 0.95, 0.95,
                )
            )
        normal_z = abs(geometry.face_normals[face_id][2])
        z = geometry.face_centroids[face_id][2]
        if normal_z >= 0.85:
            category = "ground" if z <= low_threshold else "roof"
            confidence = 0.75 if category == "ground" else 0.60
        elif normal_z <= 0.35:
            category = "wall"
            confidence = 0.65
        else:
            category = "other_surface"
            confidence = 0.40
        evidence.append(
            SemanticEvidence(
                entity_id, SegmentationLevel.FACE, category,
                SemanticSource.GEOMETRY, confidence, 0.70,
            )
        )
        if face_id in annotations:
            category, confidence = annotations[face_id]
            evidence.append(
                SemanticEvidence(
                    entity_id, SegmentationLevel.FACE, category,
                    SemanticSource.ANNOTATION, confidence, 1.0,
                )
            )
    return evidence


def _semantic_by_face(face_count: int, result: SemanticFusionResult) -> list[str]:
    labels = ["unknown"] * face_count
    for region in result.regions:
        for face_id in region.face_ids:
            if 0 <= face_id < face_count:
                labels[face_id] = region.category
    return labels


def _load_annotations(path: Path | None, face_count: int) -> dict[int, tuple[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_labels = payload.get("face_labels", {}) if isinstance(payload, dict) else {}
    if not isinstance(raw_labels, dict):
        raise ValueError("annotations.face_labels must be a mapping")
    result: dict[int, tuple[str, float]] = {}
    for raw_face_id, raw_value in raw_labels.items():
        face_id = int(raw_face_id)
        if face_id < 0 or face_id >= face_count:
            raise ValueError(f"annotation face id out of range: {face_id}")
        if isinstance(raw_value, str):
            result[face_id] = (raw_value, 1.0)
        elif isinstance(raw_value, dict) and raw_value.get("category"):
            result[face_id] = (str(raw_value["category"]), float(raw_value.get("confidence", 1.0)))
        else:
            raise ValueError(f"invalid face annotation for {face_id}")
    return result


def _category_from_name(name: str) -> str | None:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    keywords = (
        ("glass", "glass"), ("curtain", "glass"), ("water", "water"),
        ("asphalt", "asphalt"), ("road", "asphalt"), ("vegetation", "vegetation"),
        ("grass", "vegetation"), ("limestone", "limestone"), ("stone", "stone"),
        ("concrete", "concrete"), ("metal", "metal_roof"), ("roof", "roof"),
        ("soil", "soil"), ("terrain", "ground"), ("ground", "ground"),
    )
    return next((category for keyword, category in keywords if keyword in normalized), None)


def _is_ground_name(name: str) -> bool:
    normalized = name.lower()
    return any(token in normalized for token in ("ground", "terrain", "road", "asphalt", "water", "vegetation"))


def _resolve_path(config_path: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = (config_path.parent / path, config_path.parent.parent / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[-1].resolve())


def _optional_path(config_path: Path, raw: object) -> Path | None:
    if raw in (None, ""):
        return None
    return _resolve_path(config_path, str(raw))


def _fingerprint(config_path: Path, geometry_path: Path, annotation_path: Path | None) -> str:
    digest = sha256()
    for path in (config_path, geometry_path, annotation_path):
        if path is not None and path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _timed(timings: dict[str, float], name: str, function: Any) -> Any:
    started = perf_counter()
    result = function()
    timings[name] = round(perf_counter() - started, 6)
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_face_csv(path: Path, values: Sequence[str], column: str) -> None:
    lines = [f"face_id,{column}\n"]
    lines.extend(f"{face_id},{value}\n" for face_id, value in enumerate(values))
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _geometry_payload(model: GeometryModel) -> dict[str, object]:
    return {
        "status": "PASS",
        "vertex_count": model.vertex_count,
        "face_count": model.face_count,
        "bounds_m": {"min": model.bounds_min, "max": model.bounds_max},
        "metadata": dict(model.metadata),
        "warnings": list(model.warnings),
        "object_count": len(set(model.face_objects)),
        "material_slot_count": len(set(model.face_materials)),
    }


def _counts(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))
