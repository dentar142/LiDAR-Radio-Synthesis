"""TWC-2023-inspired low-resolution radio-semantic influence field.

The coefficients are effective attenuation terms learned from RSS residuals.
They are not building heights, visual labels, or electromagnetic materials.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    y_min: float
    cell_size_m: float
    nx: int
    ny: int

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def x_max(self) -> float:
        return self.x_min + self.nx * self.cell_size_m

    @property
    def y_max(self) -> float:
        return self.y_min + self.ny * self.cell_size_m

    def cell_center(self, cell_id: int) -> tuple[float, float]:
        iy, ix = divmod(int(cell_id), self.nx)
        return (
            self.x_min + (ix + 0.5) * self.cell_size_m,
            self.y_min + (iy + 0.5) * self.cell_size_m,
        )


def make_grid(tx_xy: np.ndarray, rx_xy: np.ndarray, cell_size_m: float) -> GridSpec:
    tx = np.asarray(tx_xy, float).reshape(2)
    rx = np.asarray(rx_xy, float)
    if rx.ndim != 2 or rx.shape[1] != 2 or cell_size_m <= 0:
        raise ValueError("rx_xy must be [N,2] and cell_size_m must be positive")
    all_xy = np.vstack([tx[None, :], rx])
    low = np.floor(all_xy.min(axis=0) / cell_size_m) * cell_size_m
    high = np.ceil(all_xy.max(axis=0) / cell_size_m) * cell_size_m
    high = np.maximum(high, low + cell_size_m)
    nx, ny = np.maximum(np.rint((high - low) / cell_size_m).astype(int), 1)
    return GridSpec(float(low[0]), float(low[1]), float(cell_size_m), int(nx), int(ny))


def segment_grid_matrix(tx_xy: np.ndarray, rx_xy: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Return exact horizontal path length in each cell, normalized by cell size."""
    tx = np.asarray(tx_xy, float).reshape(2)
    rx = np.asarray(rx_xy, float)
    output = np.zeros((len(rx), grid.n_cells), dtype=float)
    x_edges = grid.x_min + np.arange(1, grid.nx) * grid.cell_size_m
    y_edges = grid.y_min + np.arange(1, grid.ny) * grid.cell_size_m
    for row, endpoint in enumerate(rx):
        delta = endpoint - tx
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            continue
        cuts = [0.0, 1.0]
        if abs(delta[0]) > 1e-12:
            cuts.extend(((x_edges - tx[0]) / delta[0]).tolist())
        if abs(delta[1]) > 1e-12:
            cuts.extend(((y_edges - tx[1]) / delta[1]).tolist())
        cuts = np.asarray([t for t in cuts if -1e-12 <= t <= 1.0 + 1e-12], float)
        cuts = np.unique(np.clip(cuts, 0.0, 1.0))
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left <= 1e-12:
                continue
            midpoint = tx + 0.5 * (left + right) * delta
            ix = int(np.floor((midpoint[0] - grid.x_min) / grid.cell_size_m))
            iy = int(np.floor((midpoint[1] - grid.y_min) / grid.cell_size_m))
            ix = min(max(ix, 0), grid.nx - 1)
            iy = min(max(iy, 0), grid.ny - 1)
            output[row, iy * grid.nx + ix] += distance * (right - left) / grid.cell_size_m
    return output


@dataclass
class SemanticInfluenceModel:
    intercept_db: float
    cell_attenuation_db: np.ndarray
    ridge: float
    optimizer_success: bool
    optimizer_cost: float

    def predict_correction(self, x: np.ndarray) -> np.ndarray:
        return self.intercept_db + np.asarray(x, float) @ self.cell_attenuation_db


def fit_semantic_influence(
    x: np.ndarray,
    residual_db: np.ndarray,
    *,
    ridge: float = 5.0,
    min_cell_db: float = -20.0,
) -> SemanticInfluenceModel:
    """Fit bias plus non-positive, ridge-regularized cell attenuation."""
    x = np.asarray(x, float)
    y = np.asarray(residual_db, float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(y) < 3:
        raise ValueError("aligned X/y with at least three observations required")
    if ridge < 0 or min_cell_db >= 0:
        raise ValueError("ridge must be non-negative and min_cell_db negative")
    design = np.column_stack([np.ones(len(x)), x])
    if ridge > 0:
        regularizer = np.zeros((x.shape[1], x.shape[1] + 1), dtype=float)
        regularizer[:, 1:] = np.sqrt(ridge) * np.eye(x.shape[1])
        design = np.vstack([design, regularizer])
        y = np.r_[y, np.zeros(x.shape[1])]
    lower = np.r_[-np.inf, np.full(x.shape[1], min_cell_db)]
    upper = np.r_[np.inf, np.zeros(x.shape[1])]
    result = lsq_linear(design, y, bounds=(lower, upper), method="trf", lsmr_tol="auto")
    return SemanticInfluenceModel(
        intercept_db=float(result.x[0]),
        cell_attenuation_db=np.asarray(result.x[1:], float),
        ridge=float(ridge),
        optimizer_success=bool(result.success),
        optimizer_cost=float(result.cost),
    )


def fit_rbf_residual(
    train_xy: np.ndarray,
    residual_db: np.ndarray,
    query_xy: np.ndarray,
    *,
    length_m: float = 45.0,
    ridge: float = 1.0,
) -> np.ndarray:
    """Same local residual model as the existing U2 engineering baseline."""
    train_xy = np.asarray(train_xy, float)
    residual = np.asarray(residual_db, float).reshape(-1)
    query_xy = np.asarray(query_xy, float)
    if len(train_xy) == 0:
        return np.zeros(len(query_xy), dtype=float)
    unique_xy, inverse = np.unique(train_xy, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=residual)
    counts = np.bincount(inverse)
    targets = sums / np.maximum(counts, 1)
    kernel_train = np.exp(-(cdist(unique_xy, unique_xy) ** 2) / (2.0 * length_m**2))
    kernel_query = np.exp(-(cdist(query_xy, unique_xy) ** 2) / (2.0 * length_m**2))
    weights = np.linalg.solve(kernel_train + ridge * np.eye(len(unique_xy)), targets)
    return kernel_query @ weights


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, float) - np.asarray(observed, float)
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "mae_db": float(np.mean(absolute)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "p90_abs_db": float(np.percentile(absolute, 90)),
        "bias_db": float(np.mean(error)),
    }
