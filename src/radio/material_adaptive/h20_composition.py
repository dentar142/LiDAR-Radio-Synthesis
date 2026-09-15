#!/usr/bin/env python3
"""Train-only H20 TWC proxy corrections and fixed prediction compositions.

Both corrections are engineering proxies inspired by the environment-semantic
structure/residual idea in Liu and Chen (IEEE TWC 2023).  ``CLASS_TWC`` is a
four-term distance/LoS-class correction and ``GRID_TWC`` is a 100 m horizontal
line-integral attenuation field.  Neither is the complete paper algorithm, a
wall-count estimator, or a material-identification model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .semantic_impact import GridSpec, fit_semantic_influence, segment_grid_matrix


GRID_CELL_M = 100.0
CLASS_RIDGE = 5.0
GRID_RIDGE = 5.0
GRID_MIN_CELL_DB = -20.0
CLAIM_BOUNDARY = (
    "engineering TWC-inspired proxy; not the complete original TWC method, "
    "wall count, physical material recovery, or site-measured attenuation"
)


def _xyz_tx(xyz, tx, *, minimum_rows: int = 1) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_rows:
        raise ValueError(f"xyz must be [N,3] with at least {minimum_rows} rows")
    transmitters = np.asarray(tx, dtype=float)
    if transmitters.shape == (3,):
        transmitters = np.broadcast_to(transmitters, points.shape).copy()
    elif transmitters.shape != points.shape:
        raise ValueError("tx must be [3] or aligned [N,3]")
    if not np.isfinite(points).all() or not np.isfinite(transmitters).all():
        raise ValueError("xyz and tx must be finite")
    return points, transmitters


def _finite_vector(values, rows: int, name: str, *, boolean: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != rows:
        raise ValueError(f"{name} must be an aligned vector")
    if boolean:
        if array.dtype.kind != "b":
            if not np.isin(array, (0, 1, False, True)).all():
                raise ValueError(f"{name} must contain only booleans")
        return array.astype(bool)
    array = array.astype(float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _distance_features(xyz: np.ndarray, tx: np.ndarray, nlos: np.ndarray, center: float) -> np.ndarray:
    distance = np.linalg.norm(xyz - tx, axis=1)
    log_distance = np.log10(np.maximum(distance, 1.0)) - float(center)
    category = nlos.astype(float)
    return np.column_stack((np.ones(len(xyz)), log_distance, category, category * log_distance))


@dataclass(frozen=True)
class ClassTWCModel:
    log_distance_center: float
    coefficients: np.ndarray
    ridge: float = CLASS_RIDGE
    proxy_kind: str = "CLASS_TWC"
    claim_boundary: str = CLAIM_BOUNDARY

    def predict(self, xyz, tx, nlos) -> np.ndarray:
        points, transmitters = _xyz_tx(xyz, tx)
        category = _finite_vector(nlos, len(points), "nlos", boolean=True)
        coefficients = np.asarray(self.coefficients, dtype=float)
        if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
            raise RuntimeError("invalid fitted CLASS_TWC coefficients")
        prediction = _distance_features(
            points, transmitters, category, self.log_distance_center
        ) @ coefficients
        if prediction.shape != (len(points),) or not np.isfinite(prediction).all():
            raise RuntimeError("nonfinite CLASS_TWC prediction")
        return prediction

    def metadata(self) -> dict:
        return {
            "proxy_kind": self.proxy_kind,
            "feature_order": ["intercept", "centered_log10_distance", "nlos",
                              "nlos_x_centered_log10_distance"],
            "log_distance_center": float(self.log_distance_center),
            "coefficients": np.asarray(self.coefficients, float).tolist(),
            "ridge": float(self.ridge),
            "ridge_excludes_intercept": True,
            "claim_boundary": self.claim_boundary,
        }


def fit_class(xyz, tx, nlos, residual) -> ClassTWCModel:
    """Fit CLASS_TWC from training rows only; prediction never accepts labels."""
    points, transmitters = _xyz_tx(xyz, tx, minimum_rows=4)
    category = _finite_vector(nlos, len(points), "nlos", boolean=True)
    target = _finite_vector(residual, len(points), "residual")
    if np.unique(category).size != 2:
        raise ValueError("CLASS_TWC requires both LoS and NLoS training support")
    raw_log_distance = np.log10(np.maximum(np.linalg.norm(points - transmitters, axis=1), 1.0))
    if np.unique(raw_log_distance).size < 2:
        raise ValueError("CLASS_TWC requires at least two training distances")
    center = float(raw_log_distance.mean())
    design = _distance_features(points, transmitters, category, center)
    penalty = np.diag([0.0, CLASS_RIDGE, CLASS_RIDGE, CLASS_RIDGE])
    system = design.T @ design + penalty
    right = design.T @ target
    try:
        coefficients = np.linalg.solve(system, right)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("CLASS_TWC ridge solve failed") from exc
    if coefficients.shape != (4,) or not np.isfinite(coefficients).all():
        raise RuntimeError("CLASS_TWC fit produced invalid coefficients")
    return ClassTWCModel(center, np.asarray(coefficients, float))


def fixed_grid_bounds(xyz, tx) -> tuple[float, float, float, float]:
    """Build label-free 100 m-aligned bounds from all campaign positions and Tx."""
    points, transmitters = _xyz_tx(xyz, tx)
    all_xy = np.vstack((points[:, :2], transmitters[:, :2]))
    low = np.floor(all_xy.min(axis=0) / GRID_CELL_M) * GRID_CELL_M
    high = np.ceil(all_xy.max(axis=0) / GRID_CELL_M) * GRID_CELL_M
    high = np.maximum(high, low + GRID_CELL_M)
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def _grid_from_bounds(bounds) -> GridSpec:
    values = np.asarray(bounds, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("bounds must be finite (x_min, y_min, x_max, y_max)")
    x_min, y_min, x_max, y_max = values.tolist()
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounds must have positive width and height")
    scaled = values / GRID_CELL_M
    if not np.allclose(scaled, np.rint(scaled), rtol=0.0, atol=1e-10):
        raise ValueError("bounds must be aligned to the fixed 100 m grid")
    nx = int(round((x_max - x_min) / GRID_CELL_M))
    ny = int(round((y_max - y_min) / GRID_CELL_M))
    if nx < 1 or ny < 1:
        raise ValueError("bounds produce an empty grid")
    return GridSpec(x_min, y_min, GRID_CELL_M, nx, ny)


def _inside_grid(points: np.ndarray, transmitters: np.ndarray, grid: GridSpec) -> bool:
    xy = np.vstack((points[:, :2], transmitters[:, :2]))
    tolerance = 1e-9
    return bool(
        (xy[:, 0] >= grid.x_min - tolerance).all()
        and (xy[:, 0] <= grid.x_max + tolerance).all()
        and (xy[:, 1] >= grid.y_min - tolerance).all()
        and (xy[:, 1] <= grid.y_max + tolerance).all()
    )


def _path_grid(points: np.ndarray, transmitters: np.ndarray, grid: GridSpec) -> np.ndarray:
    if not _inside_grid(points, transmitters, grid):
        raise ValueError("positions or transmitters fall outside frozen grid bounds")
    output = np.zeros((len(points), grid.n_cells), dtype=float)
    # The reusable primitive has one Tx per invocation; group identical Tx rows
    # so selected-source callers can supply an aligned [N,3] assignment.
    unique_tx, inverse = np.unique(transmitters[:, :2], axis=0, return_inverse=True)
    for index, transmitter_xy in enumerate(unique_tx):
        rows = np.flatnonzero(inverse == index)
        output[rows] = segment_grid_matrix(transmitter_xy, points[rows, :2], grid)
    expected = np.linalg.norm(points[:, :2] - transmitters[:, :2], axis=1) / GRID_CELL_M
    if not np.allclose(output.sum(axis=1), expected, rtol=1e-10, atol=1e-10):
        raise RuntimeError("GRID_TWC horizontal path coverage is not conserved")
    if not np.isfinite(output).all() or (output < 0).any():
        raise RuntimeError("invalid GRID_TWC line-integral matrix")
    return output


@dataclass(frozen=True)
class GridTWCModel:
    grid: GridSpec
    intercept_db: float
    cell_attenuation_db: np.ndarray
    supported_cells: np.ndarray
    ridge: float = GRID_RIDGE
    min_cell_db: float = GRID_MIN_CELL_DB
    proxy_kind: str = "GRID_TWC"
    claim_boundary: str = CLAIM_BOUNDARY

    def predict(self, xyz, tx) -> np.ndarray:
        points, transmitters = _xyz_tx(xyz, tx)
        coefficients = np.asarray(self.cell_attenuation_db, dtype=float)
        supported = np.asarray(self.supported_cells, dtype=bool)
        if (coefficients.shape != (self.grid.n_cells,) or supported.shape != coefficients.shape
                or not np.isfinite(coefficients).all()):
            raise RuntimeError("invalid fitted GRID_TWC state")
        if not np.all(coefficients[~supported] == 0.0):
            raise RuntimeError("unsupported GRID_TWC cells must have zero coefficient")
        prediction = float(self.intercept_db) + _path_grid(points, transmitters, self.grid) @ coefficients
        if prediction.shape != (len(points),) or not np.isfinite(prediction).all():
            raise RuntimeError("nonfinite GRID_TWC prediction")
        return prediction

    def metadata(self) -> dict:
        coefficients = np.asarray(self.cell_attenuation_db, float)
        supported = np.asarray(self.supported_cells, bool)
        return {
            "proxy_kind": self.proxy_kind,
            "grid_cell_m": GRID_CELL_M,
            "bounds": [self.grid.x_min, self.grid.y_min, self.grid.x_max, self.grid.y_max],
            "grid_shape": [self.grid.ny, self.grid.nx],
            "intercept_db": float(self.intercept_db),
            "cell_attenuation_db": coefficients.tolist(),
            "supported_cells": supported.tolist(),
            "supported_cell_n": int(supported.sum()),
            "unsupported_cells_fixed_zero": True,
            "ridge": float(self.ridge),
            "coefficient_bounds_db_per_cell_width": [float(self.min_cell_db), 0.0],
            "claim_boundary": self.claim_boundary,
        }


def fit_grid(xyz, tx, residual, *, bounds) -> GridTWCModel:
    """Fit a frozen-domain GRID_TWC correction using training labels only."""
    points, transmitters = _xyz_tx(xyz, tx, minimum_rows=3)
    target = _finite_vector(residual, len(points), "residual")
    grid = _grid_from_bounds(bounds)
    path_grid = _path_grid(points, transmitters, grid)
    supported = np.any(path_grid > 0.0, axis=0)
    if not supported.any():
        raise ValueError("GRID_TWC has no traversed training cells")
    compact = path_grid[:, supported]
    if np.any(compact.sum(axis=0) <= 0.0):
        raise RuntimeError("GRID_TWC support bookkeeping failed")
    fitted = fit_semantic_influence(
        compact, target, ridge=GRID_RIDGE, min_cell_db=GRID_MIN_CELL_DB
    )
    compact_coefficients = np.asarray(fitted.cell_attenuation_db, dtype=float)
    if (not fitted.optimizer_success or not np.isfinite(fitted.optimizer_cost)
            or not np.isfinite(fitted.intercept_db)
            or compact_coefficients.shape != (int(supported.sum()),)
            or not np.isfinite(compact_coefficients).all()
            or (compact_coefficients < GRID_MIN_CELL_DB - 1e-8).any()
            or (compact_coefficients > 1e-8).any()):
        raise RuntimeError("GRID_TWC bounded optimizer failed validation")
    coefficients = np.zeros(grid.n_cells, dtype=float)
    coefficients[supported] = np.clip(compact_coefficients, GRID_MIN_CELL_DB, 0.0)
    model = GridTWCModel(
        grid=grid,
        intercept_db=float(fitted.intercept_db),
        cell_attenuation_db=coefficients,
        supported_cells=supported,
    )
    if not np.isfinite(model.predict(points, transmitters)).all():
        raise RuntimeError("GRID_TWC training prediction validation failed")
    return model


def serial_prediction(baseline, first_correction, second_correction) -> np.ndarray:
    """Apply two residual corrections once each to one shared baseline."""
    arrays = [np.asarray(value, dtype=float) for value in
              (baseline, first_correction, second_correction)]
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("serial inputs must be aligned vectors")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("serial inputs must be finite")
    return arrays[0] + arrays[1] + arrays[2]


def fixed_prediction_blends(baseline, material_prediction, twc_prediction) -> dict[str, np.ndarray]:
    """Return only predeclared absolute-prediction blends; no weights are fit."""
    arrays = [np.asarray(value, dtype=float) for value in
              (baseline, material_prediction, twc_prediction)]
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("blend inputs must be aligned vectors")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("blend inputs must be finite")
    base, material, twc = arrays
    return {
        "PARALLEL_MATERIAL_TWC_50_50": 0.5 * material + 0.5 * twc,
        "PARALLEL_BASE_MATERIAL_TWC_THIRDS": (base + material + twc) / 3.0,
    }
