"""Building grouping interfaces."""
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

@dataclass(frozen=True)
class BuildingObject:
    """Stable building identifier and geometry references."""
    object_id: str
    face_ids: Tuple[int, ...]
    effective_height_m: float | None = None

def group_buildings(face_ids: Sequence[int]) -> Sequence[BuildingObject]:
    """Group faces into building objects; grouping logic is a placeholder."""
    raise NotImplementedError
