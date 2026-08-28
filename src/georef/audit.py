"""Interfaces for coordinate and elevation audits."""
from dataclasses import dataclass
from typing import Mapping

@dataclass(frozen=True)
class CoordinateAudit:
    """Declared CRS, axis order, units and audit status."""
    crs: str
    axis_order: str
    unit: str
    vertical_datum: str | None
    passed: bool
    notes: tuple[str, ...] = ()

def audit_coordinates(metadata: Mapping[str, str]) -> CoordinateAudit:
    """Audit coordinate metadata and conversion chain; implementation omitted."""
    raise NotImplementedError
