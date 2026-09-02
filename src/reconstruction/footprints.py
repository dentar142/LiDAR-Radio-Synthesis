"""Engineering/master-plan building footprint extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient


@dataclass(frozen=True)
class Footprint:
    building_id: str
    outer_local_m: np.ndarray
    holes_local_m: tuple[np.ndarray, ...]
    area_m2: float
    source_component: int


def extract_footprints(plan_image: Path, config: Mapping[str, object]) -> tuple[list[Footprint], dict[str, object]]:
    """Extract building fills using a configured RGB rule and plan scale."""

    image = cv2.imread(str(plan_image), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read plan image: {plan_image}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)
    lower = np.asarray(config.get("rgb_lower", [160, 120, 70]), dtype=np.int16)
    upper = np.asarray(config.get("rgb_upper", [245, 220, 190]), dtype=np.int16)
    mask = np.all((rgb > lower) & (rgb < upper), axis=2)
    rules = config.get("channel_difference_min", {"r_minus_g": 10, "g_minus_b": 10})
    if isinstance(rules, Mapping):
        mask &= rgb[..., 0] - rgb[..., 1] > int(rules.get("r_minus_g", -256))
        mask &= rgb[..., 1] - rgb[..., 2] > int(rules.get("g_minus_b", -256))
    crop = tuple(int(value) for value in config.get("crop_px", [0, 0, image.shape[1], image.shape[0]]))
    if len(crop) != 4:
        raise ValueError("plan.crop_px must contain [left, top, right, bottom]")
    left, top, right, bottom = crop
    spatial = np.zeros(mask.shape, dtype=np.uint8)
    spatial[top:bottom, left:right] = mask[top:bottom, left:right]
    minimum_pixels = int(config.get("minimum_component_pixels", 1000))
    metres_per_pixel = float(config.get("metres_per_pixel", 1.0))
    simplify_m = float(config.get("simplify_m", 1.5))
    minimum_area_m2 = float(config.get("minimum_area_m2", 40.0))
    minimum_hole_area_m2 = float(config.get("minimum_hole_area_m2", 100.0))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(spatial)
    raw: list[tuple[np.ndarray, tuple[np.ndarray, ...], float, int]] = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < minimum_pixels:
            continue
        component = (labels == label).astype(np.uint8)
        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        for contour_index, contour in enumerate(contours):
            if hierarchy[0][contour_index][3] != -1 or len(contour) < 4:
                continue
            outer_px = contour[:, 0, :].astype(np.float64)
            outer = np.column_stack((outer_px[:, 0], -outer_px[:, 1])) * metres_per_pixel
            holes: list[np.ndarray] = []
            child = hierarchy[0][contour_index][2]
            while child != -1:
                if len(contours[child]) >= 4:
                    hole_px = contours[child][:, 0, :].astype(np.float64)
                    holes.append(np.column_stack((hole_px[:, 0], -hole_px[:, 1])) * metres_per_pixel)
                child = hierarchy[0][child][0]
            repaired = Polygon(outer.tolist(), [hole.tolist() for hole in holes]).buffer(0)
            parts = [repaired] if isinstance(repaired, Polygon) else list(repaired.geoms) if isinstance(repaired, MultiPolygon) else []
            for part in parts:
                kept_holes = [
                    list(ring.coords)
                    for ring in part.interiors
                    if Polygon(ring).area >= minimum_hole_area_m2
                ]
                smooth = orient(
                    Polygon(part.exterior.coords, kept_holes).simplify(simplify_m, preserve_topology=True),
                    1.0,
                )
                if smooth.is_valid and smooth.area >= minimum_area_m2:
                    raw.append((
                        np.asarray(smooth.exterior.coords[:-1], dtype=np.float64),
                        tuple(np.asarray(ring.coords[:-1], dtype=np.float64) for ring in smooth.interiors),
                        float(smooth.area),
                        label,
                    ))
    if not raw:
        raise RuntimeError("plan extraction produced no building footprints")
    all_points = np.vstack([item[0] for item in raw])
    centre = (all_points.min(axis=0) + all_points.max(axis=0)) / 2.0
    centered = [(outer - centre, tuple(hole - centre for hole in holes), area, label) for outer, holes, area, label in raw]
    centered.sort(key=lambda item: (-float(np.mean(item[0][:, 1])), float(np.mean(item[0][:, 0]))))
    footprints = [
        Footprint(f"PLAN_BUILDING_{index:02d}", outer, holes, area, label)
        for index, (outer, holes, area, label) in enumerate(centered, start=1)
    ]
    audit = {
        "status": "PASS_AUTOMATIC_EXTRACTION",
        "plan_image": str(plan_image.resolve()),
        "building_count": len(footprints),
        "mask_pixels": int(spatial.sum()),
        "metres_per_pixel": metres_per_pixel,
        "crop_px": list(crop),
        "centre_removed_m": centre.tolist(),
        "total_footprint_area_m2": round(sum(item.area_m2 for item in footprints), 3),
    }
    return footprints, audit


def _signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]) / 2.0)


def boundary_samples(footprints: Sequence[Footprint], spacing_m: float) -> np.ndarray:
    """Sample all plan polygon boundaries at approximately fixed spacing."""

    samples: list[tuple[float, float]] = []
    for footprint in footprints:
        for ring in (footprint.outer_local_m, *footprint.holes_local_m):
            for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
                length = float(np.linalg.norm(b - a))
                count = max(1, int(np.ceil(length / spacing_m)))
                for index in range(count):
                    point = a + (b - a) * (index / count)
                    samples.append((float(point[0]), float(point[1])))
    return np.asarray(samples, dtype=np.float32)
