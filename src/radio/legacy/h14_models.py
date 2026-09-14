"""Bounded H14 estimator pool for spatial radio-map experiments.

These are ordinary statistical estimators used for a controlled candidate
screen.  In particular, the kernel-ridge candidates are not Gaussian-process
models and none of the candidates is a reimplementation of a cited paper.
The module is deliberately data-agnostic: callers must supply train/query
arrays explicitly and no experiment data are loaded here.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


_CHUNK = 2048
_MAX_NEIGHBORS = 128


def candidate_specs() -> list[dict]:
    """Return the fixed, deterministic H14 candidate catalogue."""
    specs: list[dict] = []
    for bandwidth in (10.0, 25.0, 50.0, 100.0):
        specs.append({
            "id": f"gaussian_nw_bw{int(bandwidth)}",
            "family": "gaussian_nw",
            "bandwidth_m": bandwidth,
            "neighbors": _MAX_NEIGHBORS,
        })
    for k in (16, 32, 64):
        specs.append({"id": f"adaptive_gaussian_k{k}", "family": "adaptive_gaussian", "k": k})
    for k, power in product((8, 32, 64), (1.0, 2.0)):
        specs.append({
            "id": f"idw_k{k}_p{int(power)}",
            "family": "idw",
            "k": k,
            "power": power,
        })
    for length_m, alpha in product((25.0, 75.0, 150.0), (0.1, 1.0)):
        specs.append({
            "id": f"matern32_krr_l{int(length_m)}_a{alpha:g}",
            "family": "matern32_krr",
            "length_m": length_m,
            "alpha": alpha,
        })
    for lx, ly in ((25.0, 100.0), (100.0, 25.0)):
        specs.append({
            "id": f"anisotropic_matern32_krr_l{int(lx)}x{int(ly)}_a0.3",
            "family": "anisotropic_matern32_krr",
            "length_xy_m": (lx, ly),
            "alpha": 0.3,
        })
    for leaf in (4, 16):
        specs.append({
            "id": f"extra_trees_xy_leaf{leaf}",
            "family": "extra_trees_xy",
            "min_samples_leaf": leaf,
            "n_estimators": 96,
        })
    for features in ("xy", "xy_aux"):
        for loss in ("squared_error", "absolute_error"):
            specs.append({
                "id": f"hist_gb_{features}_{loss}",
                "family": "hist_gradient_boosting",
                "features": features,
                "loss": loss,
                "max_iter": 120,
                "max_leaf_nodes": 15,
                "l2_regularization": 10.0,
            })
    for alpha in (0.3, 1.0):
        specs.append({
            "id": f"physics_matern32_krr_a{alpha:g}",
            "family": "physics_matern32_krr",
            "xy_scale_m": 75.0,
            "alpha": alpha,
        })
    return specs


def _as_xy(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    return array.copy()


def _as_aux(values, rows: int, name: str) -> np.ndarray:
    if values is None:
        return np.empty((rows, 0), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or len(array) != rows:
        raise ValueError(f"{name} must have shape ({rows}, D)")
    return array.copy()


def _clean_inputs(train_xy, train_y, query_xy, train_aux, query_aux):
    tx = _as_xy(train_xy, "train_xy")
    qx = _as_xy(query_xy, "query_xy")
    y = np.asarray(train_y, dtype=float).reshape(-1).copy()
    if len(y) != len(tx):
        raise ValueError("train_y length must match train_xy")
    ta = _as_aux(train_aux, len(tx), "train_aux")
    qa = _as_aux(query_aux, len(qx), "query_aux")
    if ta.shape[1] != qa.shape[1]:
        raise ValueError("train_aux and query_aux must have the same column count")

    valid = np.isfinite(y) & np.isfinite(tx).all(axis=1)
    tx, y, ta = tx[valid], y[valid], ta[valid]
    if len(tx):
        xy_fill = np.median(tx, axis=0)
        bad = ~np.isfinite(qx)
        qx[bad] = np.broadcast_to(xy_fill, qx.shape)[bad]
    else:
        qx[~np.isfinite(qx)] = 0.0
    return tx, y, qx, ta, qa


def _robust_aux(train_aux: np.ndarray, query_aux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-only robust scaling with finite imputation and bounded leverage."""
    if train_aux.shape[1] == 0:
        return train_aux.copy(), query_aux.copy()
    finite = np.isfinite(train_aux)
    center = np.zeros(train_aux.shape[1], dtype=float)
    for column in range(train_aux.shape[1]):
        observed = train_aux[finite[:, column], column]
        center[column] = float(np.median(observed)) if len(observed) else 0.0
    train = np.where(finite, train_aux, center)
    query = np.where(np.isfinite(query_aux), query_aux, center)
    mad = np.median(np.abs(train - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.std(train, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(fallback > 1e-8, fallback, 1.0))
    return np.clip((train - center) / scale, -5.0, 5.0), np.clip((query - center) / scale, -5.0, 5.0)


def _neighbors(train_xy: np.ndarray, query_xy: np.ndarray, k: int):
    width = min(max(1, int(k)), len(train_xy))
    distance, index = cKDTree(train_xy).query(query_xy, k=width, workers=1)
    if width == 1:
        distance, index = distance[:, None], index[:, None]
    return np.asarray(distance, float), np.asarray(index, int)


def _local_predict(family: str, spec: dict, train_xy, train_y, query_xy) -> np.ndarray:
    k = int(spec.get("neighbors", spec.get("k", _MAX_NEIGHBORS)))
    output = np.empty(len(query_xy), dtype=float)
    for start in range(0, len(query_xy), _CHUNK):
        stop = min(len(query_xy), start + _CHUNK)
        distance, index = _neighbors(train_xy, query_xy[start:stop], k)
        values = train_y[index]
        if family == "gaussian_nw":
            weight = np.exp(-0.5 * np.square(distance / float(spec["bandwidth_m"])))
        elif family == "adaptive_gaussian":
            bandwidth = np.maximum(distance[:, -1:], 1e-6)
            weight = np.exp(-0.5 * np.square(distance / bandwidth))
        else:
            weight = 1.0 / np.maximum(distance, 1e-3) ** float(spec["power"])
        total = weight.sum(axis=1)
        prediction = np.sum(weight * values, axis=1) / np.maximum(total, 1e-12)
        prediction[total <= 1e-12] = values[total <= 1e-12, 0]
        output[start:stop] = prediction
    return output


def _matern32(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    squared = np.maximum(
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2.0 * left @ right.T,
        0.0,
    )
    radius = np.sqrt(squared)
    root3_radius = np.sqrt(3.0) * radius
    return (1.0 + root3_radius) * np.exp(-root3_radius)


def _krr_predict(train_features, train_y, query_features, alpha: float) -> np.ndarray:
    center = float(np.mean(train_y))
    kernel = _matern32(train_features, train_features)
    diagonal = float(alpha) + 1e-10
    kernel.flat[:: len(kernel) + 1] += diagonal
    factor = cho_factor(kernel, lower=True, check_finite=False)
    coefficient = cho_solve(factor, train_y - center, check_finite=False)
    output = np.empty(len(query_features), dtype=float)
    for start in range(0, len(query_features), _CHUNK):
        stop = min(len(query_features), start + _CHUNK)
        output[start:stop] = center + _matern32(query_features[start:stop], train_features) @ coefficient
    return output


def _tree_features(train_xy, query_xy, train_aux, query_aux, include_aux: bool):
    if not include_aux:
        return train_xy, query_xy
    scaled_train, scaled_query = _robust_aux(train_aux, query_aux)
    return np.column_stack((train_xy / 75.0, scaled_train)), np.column_stack((query_xy / 75.0, scaled_query))


def predict_candidate(
    spec: dict,
    train_xy,
    train_y,
    query_xy,
    train_aux=None,
    query_aux=None,
    seed: int = 0,
) -> np.ndarray:
    """Fit one candidate on explicit training arrays and predict query rows."""
    tx, y, qx, ta, qa = _clean_inputs(train_xy, train_y, query_xy, train_aux, query_aux)
    if len(qx) == 0:
        return np.empty(0, dtype=float)
    if len(y) == 0:
        return np.zeros(len(qx), dtype=float)
    if len(y) == 1 or np.ptp(y) <= 1e-12:
        return np.full(len(qx), float(np.mean(y)), dtype=float)

    family = str(spec.get("family", ""))
    if family in {"gaussian_nw", "adaptive_gaussian", "idw"}:
        output = _local_predict(family, spec, tx, y, qx)
    elif family in {"matern32_krr", "anisotropic_matern32_krr", "physics_matern32_krr"}:
        if family == "matern32_krr":
            scale = np.full(2, float(spec["length_m"]))
            train_features, query_features = tx / scale, qx / scale
        elif family == "anisotropic_matern32_krr":
            scale = np.asarray(spec["length_xy_m"], dtype=float)
            train_features, query_features = tx / scale, qx / scale
        else:
            scaled_train_aux, scaled_query_aux = _robust_aux(ta, qa)
            spatial_scale = float(spec.get("xy_scale_m", 75.0))
            train_features = np.column_stack((tx / spatial_scale, scaled_train_aux))
            query_features = np.column_stack((qx / spatial_scale, scaled_query_aux))
        output = _krr_predict(train_features, y, query_features, float(spec["alpha"]))
    elif family == "extra_trees_xy":
        model = ExtraTreesRegressor(
            n_estimators=int(spec["n_estimators"]),
            min_samples_leaf=min(int(spec["min_samples_leaf"]), max(1, len(y) // 2)),
            random_state=int(seed),
            n_jobs=1,
        )
        model.fit(tx, y)
        output = model.predict(qx)
    elif family == "hist_gradient_boosting":
        include_aux = spec.get("features") == "xy_aux"
        train_features, query_features = _tree_features(tx, qx, ta, qa, include_aux)
        model = HistGradientBoostingRegressor(
            loss=str(spec["loss"]),
            max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]),
            l2_regularization=float(spec["l2_regularization"]),
            early_stopping=False,
            random_state=int(seed),
        )
        model.fit(train_features, y)
        output = model.predict(query_features)
    else:
        raise KeyError(f"unknown H14 candidate family: {family!r}")

    output = np.asarray(output, dtype=float).reshape(len(qx))
    fallback = float(np.mean(y))
    return np.where(np.isfinite(output), output, fallback)
