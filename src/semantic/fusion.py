"""Contracts for adaptive, provenance-aware semantic segmentation."""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class SemanticSource(str, Enum):
    """Evidence sources whose reliability can be adapted per region."""

    PHOTO = "photo"
    TEXTURE = "texture"
    ANNOTATION = "annotation"
    GEOMETRY = "geometry"
    EXISTING_MESH_SEMANTIC = "existing_mesh_semantic"


class SegmentationLevel(str, Enum):
    """Hierarchy at which a semantic decision is made."""

    SCENE = "scene"
    BUILDING = "building"
    PART = "part"
    SURFACE_COMPONENT = "surface_component"
    FACE = "face"


class ReconstructionPolicy(str, Enum):
    """Geometry action selected from semantic and geometric evidence."""

    REGULARIZE_PLANE = "regularize_plane"
    PRESERVE_DETAIL = "preserve_detail"
    MERGE_COMPONENT = "merge_component"
    KEEP_BOUNDARY = "keep_boundary"
    MANUAL_REVIEW = "manual_review"


@dataclass(frozen=True)
class SemanticEvidence:
    """One observation with region-specific reliability."""

    entity_id: str
    level: SegmentationLevel
    category: str
    source: SemanticSource
    confidence: float
    reliability: float


@dataclass(frozen=True)
class SemanticConflict:
    """Unresolved disagreement retained for review instead of hidden fallback."""

    entity_id: str
    categories: tuple[str, ...]
    sources: tuple[SemanticSource, ...]
    reason: str


@dataclass(frozen=True)
class SemanticRegion:
    """Adaptive region passed to geometry reconstruction."""

    region_id: str
    level: SegmentationLevel
    category: str
    face_ids: tuple[int, ...]
    confidence: float
    policy: ReconstructionPolicy
    evidence_sources: tuple[SemanticSource, ...]


@dataclass(frozen=True)
class SemanticFusionResult:
    """Regions, conflicts and gate state produced by adaptive fusion."""

    regions: tuple[SemanticRegion, ...]
    conflicts: tuple[SemanticConflict, ...]
    passed: bool


def fuse_semantics(evidence: Sequence[SemanticEvidence]) -> SemanticFusionResult:
    """Adapt evidence weights and segmentation granularity; algorithm omitted."""

    raise NotImplementedError
