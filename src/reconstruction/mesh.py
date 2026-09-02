"""Watertight prism generation from regularized building parts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from shapely import constrained_delaunay_triangles
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from .heights import BuildingPart


Vector3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class CleanMesh:
    vertices: tuple[Vector3, ...]
    faces: tuple[Triangle, ...]
    face_surface_ids: tuple[str, ...]
    face_surface_types: tuple[str, ...]
    face_part_ids: tuple[str, ...]


def build_clean_mesh(parts: Sequence[BuildingPart]) -> tuple[CleanMesh, dict[str, object]]:
    """Extrude every part as a closed solid while retaining surface provenance."""

    vertices: list[Vector3] = []
    faces: list[Triangle] = []
    surface_ids: list[str] = []
    surface_types: list[str] = []
    part_ids: list[str] = []
    for part in parts:
        polygon = orient(part.polygon_world_xy, 1.0)
        triangulation = constrained_delaunay_triangles(polygon)
        top_triangles = [item for item in triangulation.geoms if item.area > 1e-9]
        for triangle_index, triangle in enumerate(top_triangles):
            coordinates = list(triangle.exterior.coords)[:3]
            if _signed_area_2d(coordinates) < 0:
                coordinates.reverse()
            top_ids = _append_vertices(vertices, [(x, y, part.roof_z_m) for x, y in coordinates])
            bottom_ids = _append_vertices(vertices, [(x, y, part.ground_z_m) for x, y in coordinates])
            _append_face(faces, surface_ids, surface_types, part_ids, tuple(top_ids), f"{part.part_id}:roof", "roof", part.part_id)
            _append_face(faces, surface_ids, surface_types, part_ids, tuple(reversed(bottom_ids)), f"{part.part_id}:base", "base", part.part_id)
        for ring_index, ring in enumerate((polygon.exterior, *polygon.interiors)):
            coordinates = list(ring.coords)
            for edge_index, (a, b) in enumerate(zip(coordinates, coordinates[1:])):
                ids = _append_vertices(vertices, [
                    (a[0], a[1], part.ground_z_m), (b[0], b[1], part.ground_z_m),
                    (b[0], b[1], part.roof_z_m), (a[0], a[1], part.roof_z_m),
                ])
                surface_id = f"{part.part_id}:facade:{ring_index}:{edge_index}"
                _append_face(faces, surface_ids, surface_types, part_ids, (ids[0], ids[1], ids[2]), surface_id, "facade", part.part_id)
                _append_face(faces, surface_ids, surface_types, part_ids, (ids[0], ids[2], ids[3]), surface_id, "facade", part.part_id)
    mesh = CleanMesh(tuple(vertices), tuple(faces), tuple(surface_ids), tuple(surface_types), tuple(part_ids))
    non_manifold, cross_part_coincident = _edge_audit(mesh)
    audit = {
        "status": "PASS" if non_manifold == 0 else "NO_GO_NON_MANIFOLD",
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "surface_components": len(set(mesh.face_surface_ids)),
        "parts": len(set(mesh.face_part_ids)),
        "non_manifold_coordinate_edges": non_manifold,
        "cross_part_coincident_coordinate_edges": cross_part_coincident,
        "edge_contract": "watertightness is evaluated per closed part; cross-part touching edges are reported separately",
    }
    return mesh, audit


def write_materialized_obj(
    mesh: CleanMesh,
    face_materials: Sequence[str],
    destination: Path,
    material_colours: Mapping[str, tuple[float, float, float]] | None = None,
) -> tuple[Path, Path]:
    """Write a semantically grouped OBJ/MTL pair."""

    if len(face_materials) != len(mesh.faces):
        raise ValueError("face material count does not match mesh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = destination.with_suffix(".mtl")
    lines = [f"mtllib {mtl_path.name}\n", "o AUTO_PLAN_LIDAR_CLEAN_MODEL\n"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in mesh.vertices)
    previous_material = previous_part = None
    for face, material, part_id in zip(mesh.faces, face_materials, mesh.face_part_ids, strict=True):
        if part_id != previous_part:
            lines.append(f"g {part_id}\n")
            previous_part = part_id
        if material != previous_material:
            lines.append(f"usemtl {material}\n")
            previous_material = material
        lines.append("f " + " ".join(str(index + 1) for index in face) + "\n")
    destination.write_text("".join(lines), encoding="utf-8", newline="\n")
    palette = material_colours or {}
    materials = sorted(set(face_materials))
    mtl_lines: list[str] = []
    for material in materials:
        colour = palette.get(material, _default_colour(material))
        mtl_lines.extend([
            f"newmtl {material}\n", f"Kd {colour[0]:.4f} {colour[1]:.4f} {colour[2]:.4f}\n",
            "Ka 0.0500 0.0500 0.0500\n", "Ks 0.1000 0.1000 0.1000\n", "Ns 30.0\n", "d 1.0\n\n",
        ])
    mtl_path.write_text("".join(mtl_lines), encoding="utf-8", newline="\n")
    return destination, mtl_path


def _append_vertices(target: list[Vector3], values: Sequence[Vector3]) -> list[int]:
    start = len(target)
    target.extend(values)
    return list(range(start, start + len(values)))


def _append_face(
    faces: list[Triangle], surface_ids: list[str], surface_types: list[str], part_ids: list[str],
    face: Triangle, surface_id: str, surface_type: str, part_id: str,
) -> None:
    faces.append(face); surface_ids.append(surface_id); surface_types.append(surface_type); part_ids.append(part_id)


def _signed_area_2d(points: Sequence[tuple[float, float]]) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, (*points[1:], points[0]))) / 2.0


def _coordinate_key(value: Vector3) -> tuple[int, int, int]:
    return tuple(round(item * 1_000_000) for item in value)  # type: ignore[return-value]


def _edge_audit(mesh: CleanMesh) -> tuple[int, int]:
    per_part_counts: dict[tuple[str, tuple[tuple[int, int, int], tuple[int, int, int]]], int] = {}
    edge_parts: dict[tuple[tuple[int, int, int], tuple[int, int, int]], set[str]] = {}
    for face, part_id in zip(mesh.faces, mesh.face_part_ids, strict=True):
        points = [_coordinate_key(mesh.vertices[index]) for index in face]
        for a, b in zip(points, (*points[1:], points[0])):
            edge = tuple(sorted((a, b)))
            key = part_id, edge
            per_part_counts[key] = per_part_counts.get(key, 0) + 1
            edge_parts.setdefault(edge, set()).add(part_id)
    non_manifold = sum(count != 2 for count in per_part_counts.values())
    cross_part_coincident = sum(len(parts) > 1 for parts in edge_parts.values())
    return non_manifold, cross_part_coincident


def _default_colour(material: str) -> tuple[float, float, float]:
    normalized = material.lower()
    if "glass" in normalized:
        return 0.20, 0.55, 0.75
    if "concrete" in normalized or "wall" in normalized:
        return 0.72, 0.72, 0.68
    if "roof" in normalized:
        return 0.45, 0.48, 0.52
    if "unknown" in normalized:
        return 0.80, 0.25, 0.35
    return 0.62, 0.62, 0.62
