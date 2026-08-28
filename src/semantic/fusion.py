"""Interfaces for provenance-aware semantic fusion."""
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

class SemanticSource(str, Enum):
    """Allowed semantic evidence sources."""
    PHOTO = "photo"
    ANNOTATION = "annotation"
    GEOMETRY = "geometry"

@dataclass(frozen=True)
class SemanticLabel:
    """Face-level label with source and confidence."""
    face_id: int
    category: str
    source: SemanticSource
    confidence: float

def fuse_semantics(labels: Sequence[SemanticLabel]) -> Sequence[SemanticLabel]:
    """Resolve labels by explicit provenance rules; implementation omitted."""
    raise NotImplementedError
