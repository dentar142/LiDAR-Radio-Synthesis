"""End-to-end automatic plan + LiDAR reconstruction orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import cv2
import numpy as np
import yaml

from .footprints import extract_footprints
from .heights import fit_building_parts
from .material_transfer import load_semantic_voxels, transfer_surface_materials, write_surface_materials
from .mesh import build_clean_mesh, write_materialized_obj
from .ortho import HeightRaster, ObjStats, rasterize_height, save_height_raster, scan_obj
from .registration import register_plan, transform_points


@dataclass(frozen=True)
class ReconstructionResult:
    run_directory: Path
    operational_status: str
    scientific_status: str
    outputs: Mapping[str, str]
    gates: Mapping[str, bool]


def run_reconstruction_pipeline(config_path: Path, output_directory: Path | None = None) -> ReconstructionResult:
    """Run automatic geometry reconstruction and original-mesh material transfer."""

    started = perf_counter()
    config_path = config_path.resolve()
    config = _load_config(config_path)
    source = _path(config_path, config["sources"]["lidar_obj"])
    plan = _path(config_path, config["sources"]["plan_image"])
    voxel_paths = [_path(config_path, value) for value in config["sources"].get("semantic_voxels", [])]
    scene_id = str(config["scene"]["id"])
    code_paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    fingerprint = _fingerprint(config_path, source, plan, *voxel_paths, *code_paths)
    run_directory = (
        output_directory.resolve()
        if output_directory is not None
        else _path(config_path, config.get("output", {}).get("root", "runs")) / scene_id / fingerprint[:12]
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    outputs: dict[str, str] = {}

    stats = _timed(timings, "scan_lidar_obj", lambda: scan_obj(source))
    raster_directory = run_directory / "ortho"
    raster = _load_or_build_raster(source, stats, raster_directory, config.get("raster", {}), timings)
    outputs["ortho_metadata"] = str(raster_directory / "source_ortho_metadata.json")

    footprints, footprint_audit = _timed(
        timings, "extract_plan_footprints", lambda: extract_footprints(plan, config.get("plan", {}))
    )
    _write_json(run_directory / "footprint_audit.json", footprint_audit)
    outputs["footprint_audit"] = str(run_directory / "footprint_audit.json")

    edge_image = cv2.imread(str(raster_directory / "source_edges.png"), cv2.IMREAD_GRAYSCALE)
    height_image = cv2.imread(str(raster_directory / "source_height.png"), cv2.IMREAD_GRAYSCALE)
    if edge_image is None or height_image is None:
        raise RuntimeError("orthographic preview generation failed")
    registration_config = dict(config.get("registration", {}))
    registration_config["raster_resolution_m"] = raster.resolution_m
    transform, registration_audit = _timed(
        timings, "register_plan_to_lidar", lambda: register_plan(
            footprints, edge_image, registration_config, height_image
        )
    )
    _write_json(run_directory / "registration_audit.json", registration_audit)
    _write_registration_overlay(footprints, transform, height_image, run_directory / "registration_overlay.png")
    outputs["registration_audit"] = str(run_directory / "registration_audit.json")
    outputs["registration_overlay"] = str(run_directory / "registration_overlay.png")

    registration_passed = str(registration_audit["status"]).startswith("PASS")
    if not registration_passed and bool(config.get("gates", {}).get("stop_on_registration_failure", True)):
        return _finish(
            run_directory, scene_id, fingerprint, outputs, timings, started,
            {"footprints": True, "registration": False, "height": False, "geometry": False, "materials": False},
            "NO_GO", "REVIEW_REQUIRED", config,
        )

    parts, height_records, height_audit = _timed(
        timings, "fit_building_heights", lambda: fit_building_parts(
            footprints, transform, raster, config.get("height_fit", {})
        )
    )
    _write_json(run_directory / "building_height_records.json", height_records)
    _write_json(run_directory / "height_fit_audit.json", height_audit)
    outputs["height_records"] = str(run_directory / "building_height_records.json")
    outputs["height_audit"] = str(run_directory / "height_fit_audit.json")

    mesh, geometry_audit = _timed(timings, "build_clean_geometry", lambda: build_clean_mesh(parts))
    _write_json(run_directory / "geometry_audit.json", geometry_audit)
    outputs["geometry_audit"] = str(run_directory / "geometry_audit.json")

    if voxel_paths:
        voxel_lookup, voxel_audit = _timed(timings, "load_source_semantics", lambda: load_semantic_voxels(voxel_paths))
        surface_rows, face_materials, material_audit = _timed(
            timings, "transfer_surface_materials", lambda: transfer_surface_materials(
                mesh, voxel_lookup, config.get("material_transfer", {})
            )
        )
        del voxel_lookup
    else:
        voxel_audit = {"source_files": {}, "combined_voxels": 0}
        surface_rows, face_materials, material_audit = transfer_surface_materials(
            mesh, {}, config.get("material_transfer", {})
        )
    write_surface_materials(surface_rows, run_directory / "surface_materials.json", voxel_audit)
    _write_json(run_directory / "material_transfer_audit.json", material_audit)
    outputs["surface_materials"] = str(run_directory / "surface_materials.json")
    outputs["material_audit"] = str(run_directory / "material_transfer_audit.json")

    obj, mtl = _timed(
        timings, "export_obj", lambda: write_materialized_obj(mesh, face_materials, run_directory / "scene_auto_clean.obj")
    )
    outputs["scene_obj"] = str(obj)
    outputs["scene_mtl"] = str(mtl)

    gates = _evaluate_gates(footprint_audit, registration_audit, height_audit, geometry_audit, material_audit, config)
    operational_status = "PASS" if all(gates.values()) else "NO_GO"
    return _finish(
        run_directory, scene_id, fingerprint, outputs, timings, started, gates,
        operational_status, "REVIEW_REQUIRED", config,
    )


def _load_or_build_raster(
    source: Path, stats: ObjStats, output: Path, config: Mapping[str, object], timings: dict[str, float]
) -> HeightRaster:
    metadata_path = output / "source_ortho_metadata.json"
    dsm_path = output / "source_dsm.npy"
    dtm_path = output / "source_dtm.npy"
    representative_path = output / "source_representative.npy"
    if metadata_path.exists() and dsm_path.exists() and dtm_path.exists() and representative_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_obj") == str(source.resolve()):
            timings["rasterize_lidar"] = 0.0
            bounds = tuple(tuple(float(value) for value in corner) for corner in metadata["bounds_m"])
            return HeightRaster(
                np.load(dtm_path, allow_pickle=False), np.load(representative_path, allow_pickle=False),
                np.load(dsm_path, allow_pickle=False), bounds,  # type: ignore[arg-type]
                float(metadata["grid"]["resolution_m"]), float(metadata["grid"]["z_bin_size_m"]),
            )
    resolution = float(config.get("resolution_m", 1.0))
    z_bin = float(config.get("z_bin_size_m", 1.0))
    raster = _timed(timings, "rasterize_lidar", lambda: rasterize_height(source, resolution, z_bin, stats))
    save_height_raster(raster, stats, source, output)
    np.save(representative_path, raster.representative, allow_pickle=False)
    return raster


def _evaluate_gates(
    footprint: Mapping[str, object], registration: Mapping[str, object], height: Mapping[str, object],
    geometry: Mapping[str, object], materials: Mapping[str, object], config: Mapping[str, object],
) -> dict[str, bool]:
    gate_config = config.get("gates", {})
    expected = gate_config.get("expected_building_count") if isinstance(gate_config, Mapping) else None
    footprint_gate = int(footprint["building_count"]) == int(expected) if expected is not None else int(footprint["building_count"]) > 0
    direct_ratio = int(height["buildings_with_direct_height_fit"]) / max(1, int(height["buildings_total"]))
    material_hit = float(materials.get("mean_source_hit_rate", 0.0))
    return {
        "footprints": footprint_gate,
        "registration": str(registration["status"]).startswith("PASS"),
        "height": direct_ratio >= float(gate_config.get("minimum_direct_height_ratio", 0.75)),
        "geometry": str(geometry["status"]) == "PASS",
        "materials": material_hit >= float(gate_config.get("minimum_mean_material_hit_rate", 0.20)),
    }


def _finish(
    run_directory: Path, scene_id: str, fingerprint: str, outputs: dict[str, str],
    timings: Mapping[str, float], started: float, gates: Mapping[str, bool],
    operational_status: str, scientific_status: str, config: Mapping[str, object],
) -> ReconstructionResult:
    report = {
        "schema": "hkustgz.auto_plan_lidar_material_pipeline.v1",
        "scene_id": scene_id,
        "fingerprint": fingerprint,
        "operational_status": operational_status,
        "scientific_status": scientific_status,
        "fully_automatic": True,
        "manual_annotations_used": False,
        "gates": dict(gates),
        "timings_seconds": dict(timings),
        "total_seconds": round(perf_counter() - started, 6),
        "outputs": dict(outputs),
        "coordinate_reference": config.get("coordinate_reference", {}),
        "claim_boundaries": [
            "plan-to-LiDAR registration has no independent GCP unless supplied externally",
            "heights are LiDAR-derived automatic priors, not survey/BIM truth",
            "surface materials are visual semantic candidates, not construction records or measured EM parameters",
            "all low-coverage or conflicting surfaces remain REVIEW_REQUIRED",
        ],
    }
    report_path = run_directory / "pipeline_report.json"
    _write_json(report_path, report)
    outputs["pipeline_report"] = str(report_path)
    return ReconstructionResult(run_directory, operational_status, scientific_status, dict(outputs), dict(gates))


def _write_registration_overlay(footprints: Any, transform: Any, height_image: np.ndarray, path: Path) -> None:
    canvas = cv2.cvtColor(height_image, cv2.COLOR_GRAY2BGR)
    for footprint in footprints:
        outer = np.rint(transform_points(footprint.outer_local_m.astype(np.float32), transform, height_image.shape)).astype(np.int32)
        cv2.polylines(canvas, [outer], True, (40, 64, 255), 2, cv2.LINE_AA)
        centre = tuple(np.rint(np.mean(outer, axis=0)).astype(int))
        cv2.putText(canvas, footprint.building_id.split("_")[-1], centre, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (82, 229, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    if not isinstance(payload.get("scene"), dict) or not payload["scene"].get("id"):
        raise ValueError("configuration must define scene.id")
    if not isinstance(payload.get("sources"), dict):
        raise ValueError("configuration must define sources")
    for field in ("lidar_obj", "plan_image"):
        if not payload["sources"].get(field):
            raise ValueError(f"configuration must define sources.{field}")
    return payload


def _path(config_path: Path, raw: object) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _fingerprint(*paths: Path) -> str:
    digest = sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        if stat.st_size <= 64 * 1024 * 1024:
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _timed(timings: dict[str, float], name: str, function: Any) -> Any:
    started = perf_counter()
    value = function()
    timings[name] = round(perf_counter() - started, 6)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
