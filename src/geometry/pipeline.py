"""Deterministic geometry loading, validation, and normalized OBJ export."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from pathlib import Path
from typing import Mapping


Vector3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class GeometryInput:
    """Source geometry and declared unit metadata."""

    path: Path
    unit: str


@dataclass(frozen=True)
class GeometryModel:
    """Validated, triangulated geometry with face-level provenance."""

    vertex_count: int
    face_count: int
    metadata: Mapping[str, str]
    vertices: tuple[Vector3, ...] = ()
    faces: tuple[Triangle, ...] = ()
    face_objects: tuple[str, ...] = ()
    face_materials: tuple[str, ...] = ()
    face_normals: tuple[Vector3, ...] = ()
    face_centroids: tuple[Vector3, ...] = ()
    bounds_min: Vector3 = (0.0, 0.0, 0.0)
    bounds_max: Vector3 = (0.0, 0.0, 0.0)
    warnings: tuple[str, ...] = ()


def _unit_scale(unit: str) -> float:
    normalized = unit.strip().lower()
    scales = {"m": 1.0, "meter": 1.0, "metre": 1.0, "cm": 0.01, "mm": 0.001}
    if normalized not in scales:
        raise ValueError(f"unsupported geometry unit: {unit!r}; expected m, cm, or mm")
    return scales[normalized]


def _parse_obj(
    path: Path, scale: float
) -> tuple[list[Vector3], list[Triangle], list[str], list[str], list[str]]:
    vertices: list[Vector3] = []
    faces: list[Triangle] = []
    objects: list[str] = []
    materials: list[str] = []
    warnings: list[str] = []
    current_object = "scene"
    current_material = "unassigned"

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            keyword = parts[0].lower()
            if keyword == "v" and len(parts) >= 4:
                vertex = tuple(float(value) * scale for value in parts[1:4])
                if not all(isfinite(value) for value in vertex):
                    raise ValueError(f"non-finite vertex at {path}:{line_number}")
                vertices.append(vertex)  # type: ignore[arg-type]
            elif keyword in {"o", "g"} and len(parts) >= 2:
                current_object = " ".join(parts[1:]).strip() or "scene"
            elif keyword == "usemtl" and len(parts) >= 2:
                current_material = " ".join(parts[1:]).strip() or "unassigned"
            elif keyword == "f" and len(parts) >= 4:
                polygon: list[int] = []
                for token in parts[1:]:
                    raw_index = token.split("/", 1)[0]
                    if not raw_index:
                        raise ValueError(f"missing OBJ vertex index at {path}:{line_number}")
                    index = int(raw_index)
                    index = len(vertices) + index if index < 0 else index - 1
                    if index < 0 or index >= len(vertices):
                        raise ValueError(f"OBJ face index out of range at {path}:{line_number}")
                    polygon.append(index)
                for offset in range(1, len(polygon) - 1):
                    faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
                    objects.append(current_object)
                    materials.append(current_material)
                if len(polygon) > 3:
                    warnings.append(f"triangulated polygon with {len(polygon)} vertices at line {line_number}")
    return vertices, faces, objects, materials, warnings


def _parse_ascii_ply(
    path: Path, scale: float
) -> tuple[list[Vector3], list[Triangle], list[str], list[str], list[str]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError(f"not a PLY file: {path}")
        vertex_count = face_count = None
        fmt = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            stripped = line.strip()
            if stripped.startswith("format "):
                fmt = stripped.split()[1]
            elif stripped.startswith("element vertex "):
                vertex_count = int(stripped.split()[2])
            elif stripped.startswith("element face "):
                face_count = int(stripped.split()[2])
            elif stripped == "end_header":
                break
        if fmt != "ascii":
            raise ValueError("automatic baseline supports ASCII PLY only; convert binary PLY to OBJ or ASCII PLY")
        if vertex_count is None or face_count is None:
            raise ValueError("PLY header must declare vertex and face counts")

        vertices: list[Vector3] = []
        for _ in range(vertex_count):
            values = handle.readline().split()
            if len(values) < 3:
                raise ValueError(f"truncated PLY vertex data: {path}")
            vertex = tuple(float(value) * scale for value in values[:3])
            if not all(isfinite(value) for value in vertex):
                raise ValueError(f"non-finite PLY vertex: {path}")
            vertices.append(vertex)  # type: ignore[arg-type]

        faces: list[Triangle] = []
        warnings: list[str] = []
        for face_index in range(face_count):
            values = handle.readline().split()
            if not values:
                raise ValueError(f"truncated PLY face data: {path}")
            count = int(values[0])
            polygon = [int(value) for value in values[1 : count + 1]]
            if len(polygon) != count or any(index < 0 or index >= len(vertices) for index in polygon):
                raise ValueError(f"invalid PLY face {face_index}: {path}")
            for offset in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[offset], polygon[offset + 1]))
            if count > 3:
                warnings.append(f"triangulated PLY polygon {face_index} with {count} vertices")
        return vertices, faces, ["scene"] * len(faces), ["unassigned"] * len(faces), warnings


def _triangle_geometry(vertices: list[Vector3], face: Triangle) -> tuple[Vector3, Vector3, float]:
    a, b, c = (vertices[index] for index in face)
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    norm = sqrt(sum(value * value for value in cross))
    normal = (0.0, 0.0, 0.0) if norm == 0.0 else tuple(value / norm for value in cross)
    centroid = tuple((a[i] + b[i] + c[i]) / 3.0 for i in range(3))
    return normal, centroid, norm * 0.5  # type: ignore[return-value]


def reconstruct_geometry(source: GeometryInput) -> GeometryModel:
    """Load, triangulate, normalize to metres, and reject degenerate geometry."""

    path = source.path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"geometry source not found: {path}")
    scale = _unit_scale(source.unit)
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices, faces, objects, materials, warnings = _parse_obj(path, scale)
    elif suffix == ".ply":
        vertices, faces, objects, materials, warnings = _parse_ascii_ply(path, scale)
    else:
        raise ValueError(f"unsupported geometry format {suffix!r}; automatic baseline supports OBJ and ASCII PLY")
    if not vertices or not faces:
        raise ValueError(f"geometry is empty: {path}")

    valid_faces: list[Triangle] = []
    valid_objects: list[str] = []
    valid_materials: list[str] = []
    normals: list[Vector3] = []
    centroids: list[Vector3] = []
    dropped = 0
    for face, object_name, material_name in zip(faces, objects, materials, strict=True):
        normal, centroid, area = _triangle_geometry(vertices, face)
        if area <= 1e-12:
            dropped += 1
            continue
        valid_faces.append(face)
        valid_objects.append(object_name)
        valid_materials.append(material_name)
        normals.append(normal)
        centroids.append(centroid)
    if not valid_faces:
        raise ValueError(f"all geometry faces are degenerate: {path}")
    if dropped:
        warnings.append(f"dropped {dropped} degenerate triangles")

    bounds_min = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    bounds_max = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    metadata = {
        "source": str(path),
        "source_sha256": sha256(path.read_bytes()).hexdigest(),
        "source_unit": source.unit,
        "normalized_unit": "m",
        "format": suffix.lstrip("."),
        "dropped_degenerate_faces": str(dropped),
    }
    return GeometryModel(
        vertex_count=len(vertices),
        face_count=len(valid_faces),
        metadata=metadata,
        vertices=tuple(vertices),
        faces=tuple(valid_faces),
        face_objects=tuple(valid_objects),
        face_materials=tuple(valid_materials),
        face_normals=tuple(normals),
        face_centroids=tuple(centroids),
        bounds_min=bounds_min,  # type: ignore[arg-type]
        bounds_max=bounds_max,  # type: ignore[arg-type]
        warnings=tuple(warnings),
    )


def export_normalized_obj(model: GeometryModel, destination: Path) -> Path:
    """Write a deterministic triangulated OBJ without modifying the source."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# normalized geometry generated by HKUSTGZ material mapping\n"]
    for x, y, z in model.vertices:
        lines.append(f"v {x:.9g} {y:.9g} {z:.9g}\n")
    previous_object = previous_material = None
    for face, object_name, material_name in zip(
        model.faces, model.face_objects, model.face_materials, strict=True
    ):
        if object_name != previous_object:
            lines.append(f"o {object_name}\n")
            previous_object = object_name
        if material_name != previous_material:
            lines.append(f"usemtl {material_name}\n")
            previous_material = material_name
        a, b, c = (index + 1 for index in face)
        lines.append(f"f {a} {b} {c}\n")
    destination.write_text("".join(lines), encoding="utf-8", newline="\n")
    return destination
