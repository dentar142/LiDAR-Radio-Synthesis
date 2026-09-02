"""Coordinate and elevation contract auditing."""

from __future__ import annotations

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
    """Audit required coordinate fields without claiming survey validation."""

    crs = str(metadata.get("crs", "")).strip()
    axis_order = str(metadata.get("axis_order", "")).strip().upper()
    unit = str(metadata.get("unit", "")).strip().lower()
    vertical_datum = str(metadata.get("vertical_datum", "")).strip() or None
    notes: list[str] = []
    passed = True
    if not crs:
        notes.append("missing CRS declaration")
        passed = False
    if axis_order not in {"ENU", "EUN", "XYZ", "YUP", "ZUP"}:
        notes.append(f"unsupported or missing axis_order: {axis_order or '<empty>'}")
        passed = False
    if unit not in {"m", "meter", "metre"}:
        notes.append(f"coordinate unit must be metres after normalization, got {unit or '<empty>'}")
        passed = False
    origin_status = str(metadata.get("origin_status", "")).strip().lower()
    if not origin_status:
        notes.append("origin status is undeclared; absolute placement requires review")
    elif "survey" not in origin_status or any(token in origin_status for token in ("not_", "candidate", "review")):
        notes.append("origin is a candidate or non-survey reference; keep REVIEW_REQUIRED")
    if crs.lower() in {"local_enu", "enu"}:
        required = ("origin_latitude_deg", "origin_longitude_deg", "origin_altitude_m")
        missing = [field for field in required if metadata.get(field) in (None, "")]
        if missing:
            notes.append("local ENU origin is incomplete: " + ", ".join(missing))
            passed = False
    return CoordinateAudit(
        crs=crs,
        axis_order=axis_order,
        unit=unit,
        vertical_datum=vertical_datum,
        passed=passed,
        notes=tuple(notes),
    )
