"""Interfaces for scene geometry reconstruction."""
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

@dataclass(frozen=True)
class GeometryInput:
    """Source geometry and declared unit metadata."""
    path: Path
    unit: str

@dataclass(frozen=True)
class GeometryModel:
    """Indexed geometry placeholder."""
    vertex_count: int
    face_count: int
    metadata: Mapping[str, str]

def reconstruct_geometry(source: GeometryInput) -> GeometryModel:
    """Reconstruct and index geometry; algorithm intentionally omitted."""
    raise NotImplementedError
