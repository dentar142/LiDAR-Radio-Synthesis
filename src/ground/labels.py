"""Interfaces for ground-surface labeling."""
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

class GroundLabel(str, Enum):
    """Ground classes kept separate from building materials."""
    TERRAIN = "terrain"
    ROAD = "road"
    WALKWAY = "walkway"
    UNKNOWN = "unknown"

def label_ground(face_ids: Sequence[int]) -> Sequence[GroundLabel]:
    """Assign ground labels; classifier intentionally omitted."""
    raise NotImplementedError
