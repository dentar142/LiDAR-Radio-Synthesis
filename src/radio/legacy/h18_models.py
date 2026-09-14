#!/usr/bin/env python3
"""Fixed H18 primary model pool; accepts only explicit fit and query indices."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .h14_models import predict_candidate
from .h15_models import _exact_gp
from .run_h11_sparse_learning_curves import family_indices, fit_physical_sparse


EXPERTS = (
    "NW50",
    "GP_XY_M32",
    "RT_PRIOR",
    "RT_GP",
    "GEOMETRY_KRR",
    "TREND_ONLY",
    "TREND_GP",
    "RT_TREND_ONLY",
    "RT_TREND_GP",
    "RT_TREND_SHRUNK_GP",
)
NW50 = {"family": "gaussian_nw", "bandwidth_m": 50.0, "neighbors": 128}
GEOMETRY_KRR = {"family": "physics_matern32_krr", "xy_scale_m": 75.0, "alpha": 1.0}
TREND_RIDGE = 10.0
RT_GP_SHRINK_M = 50.0


def affine_trend(train_xy, target, query_xy):
    """Fit the frozen H18 ridge plane and return train/query trends plus metadata."""
    train_xy = np.asarray(train_xy, dtype=float)
    query_xy = np.asarray(query_xy, dtype=float)
    target = np.asarray(target, dtype=float).reshape(-1)
    if train_xy.ndim != 2 or train_xy.shape[1] != 2 or query_xy.ndim != 2 or query_xy.shape[1] != 2:
        raise ValueError("affine_trend requires two-dimensional coordinates")
    if len(train_xy) != len(target) or not len(target):
        raise ValueError("affine_trend requires one nonempty target per training coordinate")
    if not (np.isfinite(train_xy).all() and np.isfinite(query_xy).all() and np.isfinite(target).all()):
        raise ValueError("affine_trend inputs must be finite")
    center = train_xy.mean(axis=0)
    scale = train_xy.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    train_z = (train_xy - center) / scale
    query_z = (query_xy - center) / scale
    design = np.column_stack((np.ones(len(train_z)), train_z))
    query_design = np.column_stack((np.ones(len(query_z)), query_z))
    penalty = np.diag([0.0, TREND_RIDGE, TREND_RIDGE])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    fitted = design @ coefficients
    queried = query_design @ coefficients
    metadata = {
        "kind": "two_stage_ridge_affine_trend",
        "coordinate_center_m": center.tolist(),
        "coordinate_scale_m": scale.tolist(),
        "slope_ridge": TREND_RIDGE,
        "intercept_penalized": False,
        "coefficients": coefficients.tolist(),
    }
    return fitted, queried, metadata


def _gp(train_xy, target, query_xy, seed, device, quick):
    prediction, variance, metadata = _exact_gp(train_xy, target, query_xy, seed, device, quick)
    if prediction.shape != (len(query_xy),) or not np.isfinite(prediction).all():
        raise RuntimeError("H18 GP returned incomplete predictions")
    return prediction, {
        "fit": metadata,
        "posterior_variance_summary": {
            "min": float(variance.min()) if len(variance) else None,
            "mean": float(variance.mean()) if len(variance) else None,
            "max": float(variance.max()) if len(variance) else None,
            "calibrated_interval_claim": False,
        },
    }


def fit_pool(data, tr, qu, seed, device="cpu", quick=False):
    """Fit the ten frozen H18 experts and predict only ``qu`` rows."""
    tr = np.asarray(tr, dtype=int).reshape(-1)
    qu = np.asarray(qu, dtype=int).reshape(-1)
    if not len(tr) or len(np.unique(tr)) != len(tr) or len(np.unique(qu)) != len(qu):
        raise ValueError("fit/query indices must be unique and training must be nonempty")
    if np.intersect1d(tr, qu).size:
        raise ValueError("fit and query indices overlap")

    # Physical helpers receive a copy in which every non-fit target is erased.
    frame = data.points.copy()
    observed = np.full(len(frame), np.nan, dtype=float)
    observed[tr] = frame.observed_dbm.to_numpy(float)[tr]
    if not np.isfinite(observed[tr]).all():
        raise RuntimeError("training targets missing")
    frame["observed_dbm"] = observed
    fit_data = replace(data, points=frame)

    xy = frame[["x", "y"]].to_numpy(float)
    train_xy, query_xy = xy[tr], xy[qu]
    target = observed[tr]
    union = np.r_[tr, qu]
    train_n = len(tr)
    nw_union = predict_candidate(NW50, train_xy, target, xy[union], seed=seed)
    nw_train, nw_query = nw_union[:train_n], nw_union[train_n:]
    predictions = {"NW50": nw_query}
    info = {
        "NW50": {"family": "gaussian_nw", "bandwidth_m": 50.0, "neighbors": 128}
    }

    gp_xy, gp_xy_info = _gp(train_xy, target, query_xy, seed, device, quick)
    predictions["GP_XY_M32"] = gp_xy
    info["GP_XY_M32"] = gp_xy_info

    fit_mask = np.zeros(len(frame), dtype=bool)
    fit_mask[tr] = True
    physical, physical_info = fit_physical_sparse(
        fit_data, observed, fit_mask, "BASE", twc=False, class_ridge=5.0
    )
    mean_union = nw_union.copy()
    if physical is not None:
        finite = np.isfinite(physical[union])
        mean_union[finite] = physical[union][finite]
    mean_train, mean_query = mean_union[:train_n], mean_union[train_n:]
    if not np.isfinite(mean_union).all():
        raise RuntimeError("RT/NW shared mean is nonfinite")
    predictions["RT_PRIOR"] = mean_query

    base_indices = family_indices(fit_data.configs, "BASE")
    if not base_indices:
        raise RuntimeError("neutral BASE RT configuration missing")
    raw = np.asarray(fit_data.gains[base_indices[0]], dtype=float)
    query_has_path = np.isfinite(raw[qu])
    effective_has_path = np.zeros(len(union), dtype=bool) if physical is None else np.isfinite(physical[union])
    info["RT_PRIOR"] = {
        "physical_fit": physical_info,
        "mean_contract": "finite_fitted_BASE_RT_else_same_fit_NW50",
        "all_training_records_used_by_residual_models": True,
        "train_records": train_n,
        "train_effective_rt_n": int(effective_has_path[:train_n].sum()),
        "train_nw_fallback_n": int((~effective_has_path[:train_n]).sum()),
        "query_records": len(qu),
        "query_raw_path_n": int(query_has_path.sum()),
        "query_effective_rt_n": int(effective_has_path[train_n:].sum()),
        "query_nw_fallback_n": int((~effective_has_path[train_n:]).sum()),
    }

    rt_residual = target - mean_train
    rt_gp_residual, rt_gp_info = _gp(train_xy, rt_residual, query_xy, seed, device, quick)
    predictions["RT_GP"] = mean_query + rt_gp_residual
    info["RT_GP"] = {"target": "y_minus_shared_rt_nw_mean", "gp": rt_gp_info}

    xyz = frame[["x", "y", "z"]].to_numpy(float)
    aux = np.column_stack((
        np.where(np.isfinite(raw), raw, 0.0),
        np.isfinite(raw),
        np.asarray(fit_data.los, dtype=bool),
        np.log10(np.maximum(1.0, np.linalg.norm(xyz - fit_data.tx, axis=1))),
    ))
    predictions["GEOMETRY_KRR"] = predict_candidate(
        GEOMETRY_KRR, train_xy, target, query_xy, aux[tr], aux[qu], seed
    )
    info["GEOMETRY_KRR"] = {
        "family": "physics_matern32_krr", "xy_scale_m": 75.0, "alpha": 1.0,
        "auxiliary_features": ["raw_base_rt", "raw_path", "los", "log10_tx_distance"],
        "auxiliary_scaling": "training_robust",
    }

    trend_train, trend_query, trend_info = affine_trend(train_xy, target, query_xy)
    predictions["TREND_ONLY"] = trend_query
    trend_gp_residual, trend_gp_info = _gp(
        train_xy, target - trend_train, query_xy, seed, device, quick
    )
    predictions["TREND_GP"] = trend_query + trend_gp_residual
    info["TREND_ONLY"] = trend_info
    info["TREND_GP"] = {"trend": trend_info, "gp_target": "y_minus_trend", "gp": trend_gp_info}

    rt_trend_train, rt_trend_query, rt_trend_info = affine_trend(
        train_xy, rt_residual, query_xy
    )
    rt_trend_base = mean_query + rt_trend_query
    predictions["RT_TREND_ONLY"] = rt_trend_base
    rt_trend_gp_residual, rt_trend_gp_info = _gp(
        train_xy, rt_residual - rt_trend_train, query_xy, seed, device, quick
    )
    predictions["RT_TREND_GP"] = rt_trend_base + rt_trend_gp_residual
    nearest_fit_m = cKDTree(train_xy).query(query_xy, workers=1)[0]
    shrink = np.exp(-nearest_fit_m / RT_GP_SHRINK_M)
    predictions["RT_TREND_SHRUNK_GP"] = rt_trend_base + shrink * rt_trend_gp_residual
    info["RT_TREND_ONLY"] = {
        "shared_rt_nw_mean": True, "residual_trend": rt_trend_info,
    }
    info["RT_TREND_GP"] = {
        "shared_rt_nw_mean": True, "residual_trend": rt_trend_info,
        "gp_target": "y_minus_shared_mean_minus_residual_trend", "gp": rt_trend_gp_info,
    }
    info["RT_TREND_SHRUNK_GP"] = {
        "shared_gp_with": "RT_TREND_GP", "shrink_only_gp_residual": True,
        "shrink_length_m": RT_GP_SHRINK_M,
        "nearest_fit_m": {
            "min": float(nearest_fit_m.min()) if len(nearest_fit_m) else None,
            "median": float(np.median(nearest_fit_m)) if len(nearest_fit_m) else None,
            "max": float(nearest_fit_m.max()) if len(nearest_fit_m) else None,
        },
    }

    if tuple(predictions) != EXPERTS:
        raise RuntimeError("H18 expert order changed")
    matrix = np.column_stack([predictions[name] for name in EXPERTS])
    if matrix.shape != (len(qu), len(EXPERTS)) or not np.isfinite(matrix).all():
        raise RuntimeError("H18 expert matrix is incomplete")
    return matrix, info, query_has_path
