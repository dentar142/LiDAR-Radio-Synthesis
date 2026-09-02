"""Adaptive, provenance-aware semantic fusion with conservative gates."""

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
    """Fuse evidence per entity while retaining conflicts for manual review.

    The baseline is deterministic and intentionally conservative. It never
    converts a visual category into physical EM truth and it does not hide
    close-score disagreements behind a fallback class.
    """

    grouped: dict[str, list[SemanticEvidence]] = {}
    for item in evidence:
        if not 0.0 <= item.confidence <= 1.0 or not 0.0 <= item.reliability <= 1.0:
            raise ValueError("semantic confidence and reliability must be within [0, 1]")
        grouped.setdefault(item.entity_id, []).append(item)

    regions: list[SemanticRegion] = []
    conflicts: list[SemanticConflict] = []
    for entity_id in sorted(grouped):
        items = grouped[entity_id]
        scores: dict[str, float] = {}
        for item in items:
            scores[item.category] = scores.get(item.category, 0.0) + item.confidence * item.reliability
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        best_category, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total = sum(scores.values()) or 1.0
        normalized = best_score / total
        margin = (best_score - second_score) / total
        sources = tuple(sorted({item.source for item in items}, key=lambda value: value.value))
        level = max(items, key=lambda item: item.level.value).level

        if best_score < 0.35 or normalized < 0.55 or margin < 0.10:
            conflicts.append(
                SemanticConflict(
                    entity_id=entity_id,
                    categories=tuple(category for category, _ in ranked),
                    sources=sources,
                    reason="insufficient weighted support or category margin",
                )
            )
            category = "unknown"
            policy = ReconstructionPolicy.MANUAL_REVIEW
        else:
            category = best_category
            policy = _policy_for_category(category)

        face_ids: tuple[int, ...] = ()
        if entity_id.startswith("face:"):
            try:
                face_ids = (int(entity_id.split(":", 1)[1]),)
            except ValueError:
                face_ids = ()
        regions.append(
            SemanticRegion(
                region_id=entity_id,
                level=level,
                category=category,
                face_ids=face_ids,
                confidence=round(normalized, 6),
                policy=policy,
                evidence_sources=sources,
            )
        )
    return SemanticFusionResult(tuple(regions), tuple(conflicts), not conflicts)


def _policy_for_category(category: str) -> ReconstructionPolicy:
    normalized = category.lower()
    if normalized in {"glass", "glass_facade", "curtain_wall"}:
        return ReconstructionPolicy.MERGE_COMPONENT
    if normalized in {"wall", "concrete", "limestone", "stone", "roof", "metal_roof"}:
        return ReconstructionPolicy.REGULARIZE_PLANE
    if normalized in {"edge", "frame", "window_frame", "detail"}:
        return ReconstructionPolicy.PRESERVE_DETAIL
    if normalized in {"ground", "road", "asphalt", "water", "vegetation", "walkway"}:
        return ReconstructionPolicy.KEEP_BOUNDARY
    return ReconstructionPolicy.MANUAL_REVIEW
