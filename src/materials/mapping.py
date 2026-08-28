"""Interfaces for semantic-to-material mapping."""
from dataclasses import dataclass
from typing import Sequence

@dataclass(frozen=True)
class MaterialCandidate:
    """Candidate material with bounded effective EM parameters."""
    label: str
    relative_permittivity: tuple[float, float] | None
    conductivity_s_m: tuple[float, float] | None
    confidence: float
    provenance: str

def map_materials(categories: Sequence[str]) -> Sequence[MaterialCandidate]:
    """Map semantic categories to candidates; calibration logic omitted."""
    raise NotImplementedError
