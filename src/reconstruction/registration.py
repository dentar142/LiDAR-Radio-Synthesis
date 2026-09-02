"""Cross-modal master-plan to LiDAR height-edge registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .footprints import Footprint, boundary_samples


@dataclass(frozen=True)
class PlanTransform:
    rotation_deg: float
    scale_px_per_m: float
    translation_col_px: float
    translation_row_px: float


def transform_points(points: np.ndarray, transform: PlanTransform, shape: tuple[int, int]) -> np.ndarray:
    angle = math.radians(transform.rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x = transform.scale_px_per_m * (cosine * points[:, 0] - sine * points[:, 1])
    y = transform.scale_px_per_m * (sine * points[:, 0] + cosine * points[:, 1])
    rows, columns = shape
    return np.column_stack(
        ((columns - 1) / 2 + transform.translation_col_px + x,
         (rows - 1) / 2 + transform.translation_row_px - y)
    )


def footprint_mask(footprint: Footprint, transform: PlanTransform, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    outer = np.rint(transform_points(footprint.outer_local_m.astype(np.float32), transform, shape)).astype(np.int32)
    cv2.fillPoly(mask, [outer], 1)
    for hole in footprint.holes_local_m:
        points = np.rint(transform_points(hole.astype(np.float32), transform, shape)).astype(np.int32)
        cv2.fillPoly(mask, [points], 0)
    return mask


def register_plan(
    footprints: Sequence[Footprint],
    edge_image: np.ndarray,
    config: Mapping[str, object],
    height_image: np.ndarray | None = None,
) -> tuple[PlanTransform, dict[str, object]]:
    """Deterministically minimize boundary-to-height-edge chamfer distance."""

    binary_edges = cv2.dilate((edge_image > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(1 - binary_edges, cv2.DIST_L2, 5)
    points = boundary_samples(footprints, float(config.get("boundary_spacing_m", 4.0)))
    nominal_scale = 1.0 / float(config.get("raster_resolution_m", 1.0))
    rotation_range = float(config.get("rotation_range_deg", 8.0))
    scale_fraction = float(config.get("scale_fraction", 0.10))
    translation_range = float(config.get("translation_range_px", 80.0))
    stages = [
        (
            np.arange(-rotation_range, rotation_range + 1e-6, 2.0),
            nominal_scale * np.arange(1.0 - scale_fraction, 1.0 + scale_fraction + 1e-6, 0.025),
            np.arange(-translation_range, translation_range + 1e-6, 10.0),
        ),
        (np.arange(-1.5, 1.501, 0.5), np.arange(-0.02, 0.0201, 0.005), np.arange(-8.0, 8.1, 2.0)),
    ]
    best = (math.inf, PlanTransform(0.0, nominal_scale, 0.0, 0.0), 0.0, math.inf)
    for stage_index, (rotations, scales, translations) in enumerate(stages):
        current = best[1]
        for rotation in rotations:
            candidate_rotation = float(rotation if stage_index == 0 else current.rotation_deg + rotation)
            for scale in scales:
                candidate_scale = float(scale if stage_index == 0 else current.scale_px_per_m + scale)
                for dx in translations:
                    for dy in translations:
                        transform = PlanTransform(
                            candidate_rotation,
                            candidate_scale,
                            float(dx if stage_index == 0 else current.translation_col_px + dx),
                            float(dy if stage_index == 0 else current.translation_row_px + dy),
                        )
                        score, coverage, p80 = _alignment_score(points, transform, distance)
                        if score < best[0]:
                            best = score, transform, coverage, p80
    _, transform, coverage, p80 = best
    silhouette_threshold = int(config.get("height_threshold_uint8", 60))
    silhouette_source = edge_image if height_image is None else height_image
    silhouette = _silhouette_metrics(footprints, transform, silhouette_source, silhouette_threshold)
    initial = _silhouette_metrics(
        footprints, PlanTransform(0.0, nominal_scale, 0.0, 0.0), silhouette_source, silhouette_threshold
    )
    minimum_iou = float(config.get("minimum_silhouette_iou", 0.45))
    audit = {
        "status": "PASS_AUTOMATIC_ASSOCIATION" if silhouette["iou"] >= minimum_iou else "NO_GO_REGISTRATION",
        "method": "plan boundary to LiDAR representative-height edge chamfer search",
        "transform_plan_local_m_to_source_grid": transform.__dict__,
        "edge_score": round(float(best[0]), 4),
        "boundary_inside_grid_ratio": round(float(coverage), 4),
        "boundary_distance_p80_px": round(float(p80), 4),
        "silhouette_support": silhouette,
        "initial_silhouette_iou": initial["iou"],
        "independent_control_points": 0,
        "claim_boundary": "automatic cross-modal association; not survey-grade georeferencing",
    }
    return transform, audit


def _alignment_score(points: np.ndarray, transform: PlanTransform, distance: np.ndarray) -> tuple[float, float, float]:
    mapped = transform_points(points, transform, distance.shape)
    columns = np.rint(mapped[:, 0]).astype(np.int32)
    rows = np.rint(mapped[:, 1]).astype(np.int32)
    inside = (columns >= 0) & (columns < distance.shape[1]) & (rows >= 0) & (rows < distance.shape[0])
    coverage = float(np.mean(inside))
    if coverage < 0.85:
        return 1e6, coverage, 1e6
    values = distance[rows[inside], columns[inside]]
    median = float(np.median(values))
    p80 = float(np.percentile(values, 80))
    return median + 0.35 * p80 + 20.0 * (1.0 - coverage), coverage, p80


def _silhouette_metrics(
    footprints: Sequence[Footprint], transform: PlanTransform, height_image: np.ndarray, threshold: int
) -> dict[str, float | int]:
    plan = np.zeros(height_image.shape, dtype=np.uint8)
    for footprint in footprints:
        plan |= footprint_mask(footprint, transform, height_image.shape)
    candidate = height_image >= threshold
    overlap = int(np.count_nonzero((plan > 0) & candidate))
    plan_area = int(np.count_nonzero(plan))
    candidate_area = int(np.count_nonzero(candidate))
    union = plan_area + candidate_area - overlap
    return {
        "display_height_threshold_uint8": threshold,
        "iou": round(overlap / union, 4) if union else 0.0,
        "plan_supported_ratio": round(overlap / plan_area, 4) if plan_area else 0.0,
        "candidate_inside_plan_ratio": round(overlap / candidate_area, 4) if candidate_area else 0.0,
        "plan_cells": plan_area,
        "candidate_cells": candidate_area,
        "overlap_cells": overlap,
    }
