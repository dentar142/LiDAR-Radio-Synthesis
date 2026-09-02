"""Conservative semantic-to-material candidate mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

@dataclass(frozen=True)
class MaterialCandidate:
    """Candidate material with bounded effective EM parameters."""
    label: str
    relative_permittivity: tuple[float, float] | None
    conductivity_s_m: tuple[float, float] | None
    confidence: float
    provenance: str


_VISUAL_TO_ENGINEERING = {
    "wall": "concrete",
    "concrete": "concrete",
    "limestone": "limestone",
    "stone": "stone_paving",
    "stone_paving": "stone_paving",
    "walkway": "stone_paving",
    "glass": "glass",
    "glass_facade": "glass",
    "curtain_wall": "glass",
    "roof": "metal_roof",
    "metal": "metal_roof",
    "metal_roof": "metal_roof",
    "road": "asphalt",
    "asphalt": "asphalt",
    "water": "water",
    "vegetation": "vegetation",
    "grass": "vegetation",
    "soil": "terrain",
    "ground": "terrain",
    "terrain": "terrain",
}


def map_materials(
    categories: Sequence[str],
    parameter_overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> Sequence[MaterialCandidate]:
    """Map semantic labels to engineering candidates without inventing EM truth.

    Relative permittivity and conductivity stay ``None`` unless the caller
    explicitly supplies bounded values in ``parameter_overrides``.
    """

    overrides = parameter_overrides or {}
    candidates: list[MaterialCandidate] = []
    for category in categories:
        normalized = category.strip().lower()
        label = _VISUAL_TO_ENGINEERING.get(normalized, "unknown")
        override = overrides.get(label, {})
        eps = _bounded_pair(override.get("relative_permittivity"), "relative_permittivity")
        conductivity = _bounded_pair(override.get("conductivity_s_m"), "conductivity_s_m")
        calibrated = eps is not None or conductivity is not None
        candidates.append(
            MaterialCandidate(
                label=label,
                relative_permittivity=eps,
                conductivity_s_m=conductivity,
                confidence=0.0 if label == "unknown" else 0.65,
                provenance=(
                    "CONFIGURED_EFFECTIVE_PARAMETER_RANGE_NOT_PHYSICAL_TRUTH"
                    if calibrated
                    else "UNCALIBRATED_VISUAL_ENGINEERING_PRIOR"
                ),
            )
        )
    return tuple(candidates)


def _bounded_pair(value: object, field: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} override must be a two-value range")
    low, high = float(value[0]), float(value[1])
    if low < 0.0 or high < low:
        raise ValueError(f"invalid {field} range: {value!r}")
    return (low, high)
