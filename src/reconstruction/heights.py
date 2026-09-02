"""Automatic local-ground, roof-level and building-part inference."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .footprints import Footprint
from .ortho import HeightRaster, grid_to_world
from .registration import PlanTransform, footprint_mask, transform_points


@dataclass(frozen=True)
class BuildingPart:
    part_id: str
    building_id: str
    polygon_world_xy: Polygon
    ground_z_m: float
    roof_z_m: float
    height_m: float
    source: str
    review_reasons: tuple[str, ...]


def fit_building_parts(
    footprints: Sequence[Footprint],
    transform: PlanTransform,
    raster: HeightRaster,
    config: Mapping[str, object],
) -> tuple[list[BuildingPart], list[dict[str, object]], dict[str, object]]:
    """Fit local ground, discover stable roof modes and split connected levels."""

    records: list[dict[str, object]] = []
    interim: list[tuple[Footprint, np.ndarray, np.ndarray, list[float], float | None, list[str]]] = []
    fitted_heights: list[float] = []
    ground_ring_m = float(config.get("ground_ring_m", 15.0))
    interior_erode_m = float(config.get("interior_erode_m", 3.0))
    minimum_coverage = float(config.get("minimum_roof_coverage", 0.30))
    for footprint in footprints:
        mask = footprint_mask(footprint, transform, raster.shape)
        erode_cells = max(1, round(interior_erode_m / raster.resolution_m))
        ring_cells = max(erode_cells + 1, round(ground_ring_m / raster.resolution_m))
        interior = cv2.erode(mask, _ellipse_kernel(erode_cells))
        if not np.any(interior):
            interior = mask
        outer = cv2.dilate(mask, _ellipse_kernel(ring_cells))
        ground_selection = (outer > 0) & (mask == 0) & np.isfinite(raster.dsm)
        roof_selection = (interior > 0) & np.isfinite(raster.dsm)
        ground_values = raster.dsm[ground_selection]
        raw_roof = raster.dsm[roof_selection]
        ground = float(np.percentile(ground_values, 20.0)) if ground_values.size else None
        relative_grid = (
            raster.dsm - ground
            if ground is not None
            else np.full(raster.shape, np.nan, dtype=np.float32)
        )
        relative = relative_grid[roof_selection]
        plausible = relative[(relative >= float(config.get("minimum_height_m", 3.0))) &
                             (relative <= float(config.get("maximum_height_m", 80.0)))]
        footprint_cells = int(np.count_nonzero(interior))
        coverage = float(plausible.size / footprint_cells) if footprint_cells else 0.0
        peaks = _stable_peaks(relative, config)
        fitted = float(np.percentile(plausible, 65.0)) if plausible.size else None
        spread = (
            float(np.percentile(plausible, 90.0) - np.percentile(plausible, 10.0))
            if plausible.size else None
        )
        reasons: list[str] = []
        if ground_values.size < int(config.get("minimum_ground_cells", 50)):
            reasons.append("LOCAL_GROUND_COVERAGE_LOW")
        if coverage < minimum_coverage:
            reasons.append("ROOF_COVERAGE_LOW")
        if fitted is None:
            reasons.append("NO_PLAUSIBLE_ROOF_HEIGHT")
        if spread is not None and (spread > float(config.get("multilevel_spread_m", 8.0)) or len(peaks) > 1):
            reasons.append("MULTI_LEVEL_ROOF")
        if fitted is not None:
            fitted_heights.append(fitted)
        interim.append((footprint, mask, relative_grid, peaks, ground, reasons))
        records.append({
            "building_id": footprint.building_id,
            "area_m2": round(footprint.area_m2, 3),
            "local_ground_z_m": None if ground is None else round(ground, 3),
            "fitted_height_m": None if fitted is None else round(fitted, 3),
            "height_spread_p90_p10_m": None if spread is None else round(spread, 3),
            "coverage_ratio": round(coverage, 4),
            "roof_sample_cells": int(plausible.size),
            "ground_sample_cells": int(ground_values.size),
            "candidate_roof_levels_m": [round(value, 2) for value in peaks],
            "status": "AUTO_FIT_REVIEW_REQUIRED" if reasons else "AUTO_FIT_CANDIDATE",
            "review_reasons": reasons,
        })
    campus_median = float(np.median(fitted_heights)) if fitted_heights else float(config.get("fallback_height_m", 16.0))
    parts: list[BuildingPart] = []
    for item_index, (footprint, mask, relative, peaks, ground, reasons) in enumerate(interim):
        record = records[item_index]
        fitted = record["fitted_height_m"]
        source = "lidar_local_ground_and_roof_modes"
        if ground is None:
            finite_ground = raster.dtm[np.isfinite(raster.dtm)]
            ground = float(np.percentile(finite_ground, 10.0)) if finite_ground.size else 0.0
            reasons.append("CAMPUS_GROUND_FALLBACK")
        levels = peaks[:] if peaks else ([float(fitted)] if fitted is not None else [])
        if not levels:
            levels = [_area_or_campus_fallback(footprint.area_m2, campus_median)]
            source = "automatic_area_class_fallback"
            reasons.append("HEIGHT_FALLBACK_NOT_LIDAR_FIT")
        label_grid = _label_levels(mask, relative, levels, footprint.area_m2, raster.resolution_m, config)
        building_parts: list[BuildingPart] = []
        for level_index, level in enumerate(levels):
            level_mask = (label_grid == level_index).astype(np.uint8)
            polygons = _merge_touching_polygons(_mask_polygons_world(level_mask, raster, config))
            for component_index, polygon in enumerate(polygons, start=1):
                part = BuildingPart(
                    part_id=f"{footprint.building_id}_L{level_index + 1:02d}_P{component_index:02d}",
                    building_id=footprint.building_id,
                    polygon_world_xy=polygon,
                    ground_z_m=float(ground),
                    roof_z_m=float(ground + level),
                    height_m=float(level),
                    source=source,
                    review_reasons=tuple(dict.fromkeys(reasons)),
                )
                parts.append(part)
                building_parts.append(part)
        if not building_parts:
            world = _footprint_world_polygon(footprint, transform, raster)
            level = float(levels[-1])
            fallback = BuildingPart(
                f"{footprint.building_id}_FALLBACK", footprint.building_id, world,
                float(ground), float(ground + level), level,
                "automatic_whole_footprint_fallback", tuple(dict.fromkeys([*reasons, "NO_VALID_LEVEL_REGIONS"])),
            )
            parts.append(fallback)
            building_parts.append(fallback)
        record["part_count"] = len(building_parts)
        record["generated_levels_m"] = [round(value, 2) for value in levels]
    audit = {
        "status": "PASS_AUTOMATIC_HEIGHT_PRIOR",
        "buildings_total": len(footprints),
        "buildings_with_direct_height_fit": sum(row["fitted_height_m"] is not None for row in records),
        "building_part_count": len(parts),
        "campus_median_fitted_height_m": round(campus_median, 3),
        "fallback_buildings": sum(any(reason.startswith("HEIGHT_FALLBACK") for reason in row["review_reasons"]) for row in records),
        "claim_boundary": "automatic LiDAR height prior; no independent survey or BIM height control",
    }
    return parts, records, audit


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _stable_peaks(relative: np.ndarray, config: Mapping[str, object]) -> list[float]:
    values = relative[(relative >= float(config.get("minimum_height_m", 3.0))) &
                      (relative <= float(config.get("maximum_height_m", 80.0)))]
    if values.size < 20:
        return []
    bins = np.arange(2.5, 81.5, 1.0)
    counts, edges = np.histogram(values, bins=bins)
    smooth = np.convolve(counts.astype(np.float64), np.asarray([0.25, 0.5, 0.25]), mode="same")
    support_minimum = max(20.0, values.size * float(config.get("peak_minimum_fraction", 0.025)))
    candidates: list[tuple[float, float]] = []
    for index in range(1, len(smooth) - 1):
        if smooth[index] >= support_minimum and smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1]:
            candidates.append((float(smooth[index]), float((edges[index] + edges[index + 1]) / 2.0)))
    selected: list[tuple[float, float]] = []
    separation = float(config.get("peak_separation_m", 3.0))
    for support, height in sorted(candidates, reverse=True):
        if all(abs(height - existing[1]) >= separation for existing in selected):
            selected.append((support, height))
    return [height for _, height in sorted(selected, key=lambda pair: pair[1])[: int(config.get("maximum_levels", 5))]]


def _label_levels(
    footprint: np.ndarray,
    relative: np.ndarray,
    levels: Sequence[float],
    area_m2: float,
    resolution_m: float,
    config: Mapping[str, object],
) -> np.ndarray:
    labels = np.full(footprint.shape, -1, dtype=np.int16)
    plausible = (footprint > 0) & np.isfinite(relative)
    if np.any(plausible):
        differences = np.stack([np.abs(relative - level) for level in levels])
        labels[plausible] = np.argmin(differences[:, plausible], axis=0).astype(np.int16)
    minimum_area = max(float(config.get("minimum_part_area_m2", 35.0)), area_m2 * 0.005)
    minimum_cells = max(1, round(minimum_area / (resolution_m**2)))
    for level_index in range(len(levels)):
        count, components, stats, _ = cv2.connectedComponentsWithStats((labels == level_index).astype(np.uint8))
        for component in range(1, count):
            if int(stats[component, cv2.CC_STAT_AREA]) < minimum_cells:
                labels[components == component] = -1
    available = [index for index in range(len(levels)) if np.any(labels == index)]
    if not available:
        labels[footprint > 0] = len(levels) - 1
        return labels
    distances = [cv2.distanceTransform((labels != index).astype(np.uint8), cv2.DIST_L2, 5) for index in available]
    unknown = (footprint > 0) & (labels < 0)
    nearest = np.argmin(np.stack(distances), axis=0)
    for position, level_index in enumerate(available):
        labels[unknown & (nearest == position)] = level_index
    return labels


def _mask_polygons_world(mask: np.ndarray, raster: HeightRaster, config: Mapping[str, object]) -> list[Polygon]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    minimum_area = float(config.get("minimum_part_area_m2", 35.0))
    epsilon = float(config.get("part_simplify_m", 1.0)) / raster.resolution_m
    polygons: list[Polygon] = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] != -1 or len(contour) < 3:
            continue
        outer_approximation = cv2.approxPolyDP(contour, epsilon, True)
        if len(outer_approximation) < 3:
            continue
        outer_pixels = outer_approximation[:, 0, :].astype(np.float64)
        outer = grid_to_world(outer_pixels, raster.bounds, raster.resolution_m)
        holes: list[list[list[float]]] = []
        child = hierarchy[0][index][2]
        while child != -1:
            if len(contours[child]) >= 3:
                hole_approximation = cv2.approxPolyDP(contours[child], epsilon, True)
                if len(hole_approximation) >= 3:
                    hole_pixels = hole_approximation[:, 0, :].astype(np.float64)
                    holes.append(grid_to_world(hole_pixels, raster.bounds, raster.resolution_m).tolist())
            child = hierarchy[0][child][0]
        repaired = Polygon(outer.tolist(), holes).buffer(0)
        candidates = [repaired] if isinstance(repaired, Polygon) else list(repaired.geoms) if isinstance(repaired, MultiPolygon) else []
        for candidate in candidates:
            smooth = orient(candidate.simplify(float(config.get("part_simplify_m", 1.0)), preserve_topology=True), 1.0)
            if smooth.is_valid and smooth.area >= minimum_area:
                polygons.append(smooth)
    return polygons


def _merge_touching_polygons(polygons: Sequence[Polygon]) -> list[Polygon]:
    """Dissolve same-height components that share a boundary after simplification."""

    if not polygons:
        return []
    merged = unary_union(polygons).buffer(0)
    if isinstance(merged, Polygon):
        return [orient(merged, 1.0)]
    if isinstance(merged, MultiPolygon):
        return [orient(part, 1.0) for part in merged.geoms]
    return []


def _footprint_world_polygon(footprint: Footprint, transform: PlanTransform, raster: HeightRaster) -> Polygon:
    outer_grid = transform_points(footprint.outer_local_m.astype(np.float32), transform, raster.shape)
    outer = grid_to_world(outer_grid, raster.bounds, raster.resolution_m)
    holes = [
        grid_to_world(transform_points(hole.astype(np.float32), transform, raster.shape), raster.bounds, raster.resolution_m).tolist()
        for hole in footprint.holes_local_m
    ]
    return orient(Polygon(outer.tolist(), holes).buffer(0), 1.0)


def _area_or_campus_fallback(area_m2: float, campus_median: float) -> float:
    area_guess = 24.0 if area_m2 >= 8000 else 20.0 if area_m2 >= 3000 else 16.0 if area_m2 >= 1000 else 10.0
    return float(np.median([area_guess, campus_median]))
