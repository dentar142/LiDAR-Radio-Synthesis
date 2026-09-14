#!/usr/bin/env python3
"""H11 spatially balanced sparse-sample learning curves on V7-r4 caches.

Sampling is signal-blind and nested. Every method shares the same train/test
split for a track, seed, and budget. Test targets enter only final scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import resource
except ImportError:  # Windows-only local contract tests
    resource = None

from .run_h8_combined_screen import (
    IDW_K,
    IDW_P,
    TWC_CLASS_PENALTIES_M,
    TWC_RIDGES,
    apply_class_correction,
    augmented_xy,
    family_indices,
    fit_class_correction,
    idw,
    load_band,
)
from .run_h8_geospatial_models import GAUSSIAN_BANDWIDTH_M, gaussian_predict
from .run_h10_physics_gaussian_hybrid import RESIDUAL_SHRINKS


BANDS = ("n41", "n79")
TRACK_COUNTS = {
    "full": {"n41": 22141, "n79": 16841},
    "ew_core": {"n41": 5502, "n79": 1534},
}
TRACK_BUDGETS = {
    "full": (30, 100, 300, 1000, 3000, 10000, 20000),
    "ew_core": (30, 100, 300, 1000, 3000),
}
METHODS = (
    "GAUSSIAN",
    "U2",
    "WEDT_P",
    "ONETWIN",
    "TWC_U2",
    "TWC_WEDT_P",
    "TWC_ONETWIN",
    "TWC_U2_GAUSSIAN_RESIDUAL",
    "TWC_WEDT_P_GAUSSIAN_RESIDUAL",
    "TWC_ONETWIN_GAUSSIAN_RESIDUAL",
)
CELL_SIZE_M = 10.0
TOP_FARTHEST_FRACTION = 0.10
MIN_PHYSICAL_TRAIN = 4
INNER_FOLDS = 3
STRICT_MIN_RANK_N = 500
DEFAULT_GAUSSIAN_BW = 50.0
DEFAULT_LOCAL = (16, 2.0)
DEFAULT_TWC = (5.0, 0.0, 16, 2.0)
DEFAULT_RESIDUAL = (5.0, 50.0, 1.0)


@dataclass
class BandData:
    band: str
    points: pd.DataFrame
    configs: list[dict]
    gains: np.ndarray
    los: np.ndarray
    tx: np.ndarray
    source_row_index: np.ndarray


def query_neighbors(train_features, query_features, max_k):
    """Bound each worker to one CPU thread so process parallelism stays controlled."""
    train_features = np.asarray(train_features, float)
    query_features = np.asarray(query_features, float)
    if not len(train_features):
        raise ValueError("empty neighbor training set")
    k = min(int(max_k), len(train_features))
    distance, index = cKDTree(train_features).query(query_features, k=k, workers=1)
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    return np.asarray(distance, float), np.asarray(index, int)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def derive_seed(sample_seed: int, *labels: object) -> int:
    payload = "|".join([str(int(sample_seed)), *(str(value) for value in labels)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def cell_keys(xy: np.ndarray, cell_size_m: float = CELL_SIZE_M) -> np.ndarray:
    return np.floor(np.asarray(xy, float) / float(cell_size_m)).astype(np.int64)


def allocate_budget(total: int, counts: dict[str, int]) -> dict[str, int]:
    if total < 1 or total > sum(counts.values()):
        raise ValueError(f"invalid total budget {total}")
    raw = {band: total * counts[band] / sum(counts.values()) for band in BANDS}
    allocation = {band: int(math.floor(raw[band])) for band in BANDS}
    remainder = total - sum(allocation.values())
    order = sorted(BANDS, key=lambda band: (-(raw[band] - allocation[band]), band))
    for band in order[:remainder]:
        allocation[band] += 1
    if total >= 8:
        for band in BANDS:
            if allocation[band] >= 4:
                continue
            need = 4 - allocation[band]
            donor = next(other for other in BANDS if other != band)
            take = min(need, max(0, allocation[donor] - 4))
            allocation[donor] -= take
            allocation[band] += take
            if take != need:
                raise RuntimeError(f"cannot enforce band minimum for budget={total}")
    if sum(allocation.values()) != total:
        raise RuntimeError("band allocation total mismatch")
    if any(allocation[band] > counts[band] for band in BANDS):
        raise RuntimeError("band allocation exceeds available points")
    return allocation


def randomized_spatial_order(
    xy: np.ndarray,
    max_n: int,
    seed: int,
    *,
    cell_size_m: float = CELL_SIZE_M,
    top_fraction: float = TOP_FARTHEST_FRACTION,
) -> np.ndarray:
    """Generate a nested, signal-blind order with exact cell-first maximin."""
    xy = np.asarray(xy, float)
    if max_n < 1 or max_n > len(xy):
        raise ValueError("max_n outside point range")
    keys = cell_keys(xy, cell_size_m)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    cell_count = len(unique_keys)
    rng = np.random.default_rng(np.uint32(seed))

    point_orders: list[np.ndarray] = []
    centroids = np.empty((cell_count, 2), float)
    point_counts = np.empty(cell_count, dtype=int)
    for cell in range(cell_count):
        indices = np.flatnonzero(inverse == cell)
        point_orders.append(rng.permutation(indices))
        point_counts[cell] = len(indices)
        centroids[cell] = np.mean(xy[indices], axis=0)

    output: list[int] = []
    cursor = np.zeros(cell_count, dtype=int)
    chosen = np.zeros(cell_count, dtype=bool)
    min_distance_sq = np.full(cell_count, np.inf, float)
    selected_cell = int(rng.integers(0, cell_count))
    target_first_pass = min(max_n, cell_count)

    while len(output) < target_first_pass:
        chosen[selected_cell] = True
        output.append(int(point_orders[selected_cell][0]))
        cursor[selected_cell] = 1
        delta = centroids - centroids[selected_cell]
        min_distance_sq = np.minimum(min_distance_sq, np.sum(delta * delta, axis=1))
        min_distance_sq[chosen] = -np.inf
        if len(output) >= target_first_pass:
            break
        remaining = np.flatnonzero(~chosen)
        top_n = max(1, int(math.ceil(float(top_fraction) * len(remaining))))
        if top_n >= len(remaining):
            candidates = remaining
        else:
            local = np.argpartition(min_distance_sq[remaining], -top_n)[-top_n:]
            candidates = remaining[local]
        selected_cell = int(rng.choice(candidates))

    while len(output) < max_n:
        available = np.flatnonzero(cursor < point_counts)
        if not len(available):
            break
        for cell in rng.permutation(available):
            if len(output) >= max_n:
                break
            cell = int(cell)
            output.append(int(point_orders[cell][cursor[cell]]))
            cursor[cell] += 1

    result = np.asarray(output, dtype=np.int64)
    if len(result) != max_n or len(np.unique(result)) != max_n:
        raise RuntimeError("spatial order contract failed")
    return result


def spatial_cv_labels(xy: np.ndarray, train_mask: np.ndarray, seed: int) -> np.ndarray:
    keys = cell_keys(xy)
    train_cells = np.unique(keys[train_mask], axis=0)
    rng = np.random.default_rng(np.uint32(seed))
    shuffled = train_cells[rng.permutation(len(train_cells))]
    mapping = {tuple(key): int(index % INNER_FOLDS) for index, key in enumerate(shuffled)}
    labels = np.full(len(xy), -1, dtype=int)
    for index in np.flatnonzero(train_mask):
        labels[index] = mapping[tuple(keys[index])]
    return labels


def inner_splits(train_mask, cv_labels, *, min_fit: int, min_valid: int):
    for fold in range(INNER_FOLDS):
        valid = train_mask & (cv_labels == fold)
        fit = train_mask & (cv_labels >= 0) & (cv_labels != fold)
        if fit.sum() >= min_fit and valid.sum() >= min_valid:
            yield fold, fit, valid


def gaussian_for_masks(xy, observed, fit, query, bandwidth_m):
    distance, index = query_neighbors(xy[fit], xy[query], max_k=128)
    return gaussian_predict(observed[fit], distance, index, float(bandwidth_m))


def choose_gaussian_bandwidth(xy, observed, train_mask, cv_labels):
    scores = {float(bw): [0.0, 0.0, 0] for bw in GAUSSIAN_BANDWIDTH_M}
    for _, fit, valid in inner_splits(train_mask, cv_labels, min_fit=2, min_valid=1):
        distance, index = query_neighbors(xy[fit], xy[valid], max_k=128)
        for bw in GAUSSIAN_BANDWIDTH_M:
            prediction = gaussian_predict(observed[fit], distance, index, float(bw))
            update_score(scores, float(bw), observed[valid], prediction)
    selected, info = rank_scores(scores, DEFAULT_GAUSSIAN_BW)
    return float(selected), info


def fit_physical_sparse(data, observed, fit_mask, family, *, twc, class_ridge):
    xyz = data.points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - data.tx[None, :], axis=1)
    nlos = ~np.asarray(data.los, bool)
    best = None
    for config_index in family_indices(data.configs, family):
        finite = fit_mask & np.isfinite(data.gains[config_index])
        if finite.sum() < MIN_PHYSICAL_TRAIN:
            continue
        if twc:
            center, coefficients = fit_class_correction(
                distance[finite],
                nlos[finite],
                observed[finite] - data.gains[config_index, finite],
                float(class_ridge),
            )
            prediction = data.gains[config_index] + apply_class_correction(
                distance, nlos, center, coefficients
            )
            fitted = {
                "center": float(center),
                "coefficients": np.asarray(coefficients, float).tolist(),
            }
        else:
            bias = float(np.median(observed[finite] - data.gains[config_index, finite]))
            prediction = data.gains[config_index] + bias
            fitted = {"bias": bias}
        error = prediction[finite] - observed[finite]
        candidate = (
            float(np.sqrt(np.mean(error**2))),
            str(data.configs[config_index]["id"]),
            np.asarray(prediction, float),
            int(finite.sum()),
            fitted,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return None, {
            "physical_fit_fallback": True,
            "fallback_reason": f"fewer_than_{MIN_PHYSICAL_TRAIN}_finite_train_paths",
        }
    return best[2], {
        "physical_fit_fallback": False,
        "config_id": best[1],
        "fit_rmse_db": best[0],
        "finite_train_paths": best[3],
        **best[4],
    }


def predict_expert_from_physical(
    data,
    observed,
    fit,
    query,
    gaussian_fallback,
    physical,
    *,
    local_k,
    local_power,
    class_penalty_m,
):
    output = np.asarray(gaussian_fallback, float).copy()
    if physical is None:
        return output, {"test_no_path_n": int(query.sum()), "physical_fit_fallback": True}
    xy = data.points[["x", "y"]].to_numpy(float)
    nlos = ~np.asarray(data.los, bool)
    finite_fit = fit & np.isfinite(physical)
    query_physical = np.asarray(physical[query], float)
    has_path = np.isfinite(query_physical)
    if finite_fit.any() and has_path.any():
        distance, index = query_neighbors(
            augmented_xy(xy[finite_fit], nlos[finite_fit], class_penalty_m),
            augmented_xy(xy[query][has_path], nlos[query][has_path], class_penalty_m),
            max_k=local_k,
        )
        residual = observed[finite_fit] - physical[finite_fit]
        output[has_path] = query_physical[has_path] + idw(
            residual, distance, index, int(local_k), float(local_power)
        )
    return output, {
        "test_no_path_n": int((~has_path).sum()),
        "physical_fit_fallback": False,
    }


def update_score(score, key, observed, prediction):
    error = np.asarray(prediction, float) - np.asarray(observed, float)
    if not np.isfinite(error).all():
        return
    score[key][0] += float(np.sum(error**2))
    score[key][1] += float(np.sum(np.abs(error)))
    score[key][2] += int(len(error))


def rank_scores(score, default):
    ranked = [
        (math.sqrt(sse / count), absolute / count, json.dumps(key), key)
        for key, (sse, absolute, count) in score.items()
        if count > 0
    ]
    if not ranked:
        return default, {"selection": "protocol_default", "cv_n": 0}
    rmse, mae, _, key = min(ranked)
    return key, {
        "selection": "train_spatial_cv",
        "cv_n": int(score[key][2]),
        "inner_rmse_db": float(rmse),
        "inner_mae_db": float(mae),
    }


def choose_standalone_params(data, observed, train_mask, cv_labels, family, gaussian_bw):
    keys = [(int(k), float(power)) for k in IDW_K for power in IDW_P]
    scores = {key: [0.0, 0.0, 0] for key in keys}
    fallback_folds = 0
    xy = data.points[["x", "y"]].to_numpy(float)
    for _, fit, valid in inner_splits(train_mask, cv_labels, min_fit=2, min_valid=1):
        gaussian = gaussian_for_masks(xy, observed, fit, valid, gaussian_bw)
        physical, _ = fit_physical_sparse(
            data, observed, fit, family, twc=False, class_ridge=5.0
        )
        fallback_folds += int(physical is None)
        for key in keys:
            prediction, _ = predict_expert_from_physical(
                data,
                observed,
                fit,
                valid,
                gaussian,
                physical,
                local_k=key[0],
                local_power=key[1],
                class_penalty_m=0.0,
            )
            update_score(scores, key, observed[valid], prediction)
    selected, info = rank_scores(scores, DEFAULT_LOCAL)
    info["inner_physical_fallback_folds"] = int(fallback_folds)
    return selected, info


def choose_twc_params_sparse(data, observed, train_mask, cv_labels, family, gaussian_bw):
    keys = [
        (float(ridge), float(penalty), int(k), float(power))
        for ridge in TWC_RIDGES
        for penalty in TWC_CLASS_PENALTIES_M
        for k in IDW_K
        for power in IDW_P
    ]
    scores = {key: [0.0, 0.0, 0] for key in keys}
    fallback_fits = 0
    xy = data.points[["x", "y"]].to_numpy(float)
    nlos = ~np.asarray(data.los, bool)
    for _, fit, valid in inner_splits(train_mask, cv_labels, min_fit=2, min_valid=1):
        gaussian = gaussian_for_masks(xy, observed, fit, valid, gaussian_bw)
        for ridge in TWC_RIDGES:
            physical, _ = fit_physical_sparse(
                data, observed, fit, family, twc=True, class_ridge=float(ridge)
            )
            if physical is None:
                fallback_fits += 1
                for penalty in TWC_CLASS_PENALTIES_M:
                    for k in IDW_K:
                        for power in IDW_P:
                            update_score(
                                scores,
                                (float(ridge), float(penalty), int(k), float(power)),
                                observed[valid],
                                gaussian,
                            )
                continue
            finite_fit = fit & np.isfinite(physical)
            query_physical = physical[valid]
            has_path = np.isfinite(query_physical)
            residual = observed[finite_fit] - physical[finite_fit]
            for penalty in TWC_CLASS_PENALTIES_M:
                if finite_fit.any() and has_path.any():
                    distance, index = query_neighbors(
                        augmented_xy(xy[finite_fit], nlos[finite_fit], float(penalty)),
                        augmented_xy(xy[valid][has_path], nlos[valid][has_path], float(penalty)),
                        max_k=max(IDW_K),
                    )
                for k in IDW_K:
                    for power in IDW_P:
                        prediction = gaussian.copy()
                        if finite_fit.any() and has_path.any():
                            prediction[has_path] = query_physical[has_path] + idw(
                                residual, distance, index, int(k), float(power)
                            )
                        update_score(
                            scores,
                            (float(ridge), float(penalty), int(k), float(power)),
                            observed[valid],
                            prediction,
                        )
    selected, info = rank_scores(scores, DEFAULT_TWC)
    info["inner_physical_fallback_fits"] = int(fallback_fits)
    return selected, info


def choose_residual_params(data, observed, train_mask, cv_labels, family):
    keys = [
        (float(ridge), float(bandwidth), float(shrink))
        for ridge in TWC_RIDGES
        for bandwidth in GAUSSIAN_BANDWIDTH_M
        for shrink in RESIDUAL_SHRINKS
    ]
    scores = {key: [0.0, 0.0, 0] for key in keys}
    fallback_fits = 0
    xy = data.points[["x", "y"]].to_numpy(float)
    for _, fit, valid in inner_splits(train_mask, cv_labels, min_fit=2, min_valid=1):
        gaussian_cache = {
            float(bw): gaussian_for_masks(xy, observed, fit, valid, float(bw))
            for bw in GAUSSIAN_BANDWIDTH_M
        }
        for ridge in TWC_RIDGES:
            physical, _ = fit_physical_sparse(
                data, observed, fit, family, twc=True, class_ridge=float(ridge)
            )
            if physical is None:
                fallback_fits += 1
                for bandwidth in GAUSSIAN_BANDWIDTH_M:
                    for shrink in RESIDUAL_SHRINKS:
                        update_score(
                            scores,
                            (float(ridge), float(bandwidth), float(shrink)),
                            observed[valid],
                            gaussian_cache[float(bandwidth)],
                        )
                continue
            finite_fit = fit & np.isfinite(physical)
            query_physical = physical[valid]
            has_path = np.isfinite(query_physical)
            corrections = {}
            if finite_fit.any() and has_path.any():
                distance, index = query_neighbors(
                    xy[finite_fit], xy[valid][has_path], max_k=128
                )
                residual = observed[finite_fit] - physical[finite_fit]
                corrections = {
                    float(bw): gaussian_predict(residual, distance, index, float(bw))
                    for bw in GAUSSIAN_BANDWIDTH_M
                }
            for bandwidth in GAUSSIAN_BANDWIDTH_M:
                bandwidth = float(bandwidth)
                for shrink in RESIDUAL_SHRINKS:
                    prediction = gaussian_cache[bandwidth].copy()
                    if bandwidth in corrections:
                        prediction[has_path] = (
                            query_physical[has_path] + float(shrink) * corrections[bandwidth]
                        )
                    update_score(
                        scores,
                        (float(ridge), bandwidth, float(shrink)),
                        observed[valid],
                        prediction,
                    )
    selected, info = rank_scores(scores, DEFAULT_RESIDUAL)
    info["inner_physical_fallback_fits"] = int(fallback_fits)
    return selected, info


def final_standalone(data, observed, train, test, gaussian_test, family, params):
    physical, physical_info = fit_physical_sparse(
        data, observed, train, family, twc=False, class_ridge=5.0
    )
    prediction, prediction_info = predict_expert_from_physical(
        data,
        observed,
        train,
        test,
        gaussian_test,
        physical,
        local_k=params[0],
        local_power=params[1],
        class_penalty_m=0.0,
    )
    return prediction, {**physical_info, **prediction_info}


def final_twc(data, observed, train, test, gaussian_test, family, params):
    ridge, penalty, k, power = params
    physical, physical_info = fit_physical_sparse(
        data, observed, train, family, twc=True, class_ridge=ridge
    )
    prediction, prediction_info = predict_expert_from_physical(
        data,
        observed,
        train,
        test,
        gaussian_test,
        physical,
        local_k=k,
        local_power=power,
        class_penalty_m=penalty,
    )
    return prediction, {**physical_info, **prediction_info}


def final_residual(data, observed, train, test, family, params):
    ridge, bandwidth, shrink = params
    xy = data.points[["x", "y"]].to_numpy(float)
    gaussian = gaussian_for_masks(xy, observed, train, test, bandwidth)
    physical, physical_info = fit_physical_sparse(
        data, observed, train, family, twc=True, class_ridge=ridge
    )
    prediction = gaussian.copy()
    test_no_path_n = int(test.sum())
    if physical is not None:
        finite_fit = train & np.isfinite(physical)
        query_physical = physical[test]
        has_path = np.isfinite(query_physical)
        test_no_path_n = int((~has_path).sum())
        if finite_fit.any() and has_path.any():
            distance, index = query_neighbors(xy[finite_fit], xy[test][has_path], max_k=128)
            residual = observed[finite_fit] - physical[finite_fit]
            correction = gaussian_predict(residual, distance, index, bandwidth)
            prediction[has_path] = query_physical[has_path] + shrink * correction
    return prediction, {**physical_info, "test_no_path_n": test_no_path_n}


def peak_rss_mb():
    if resource is None:
        return float("nan")
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)


def fit_predict_methods(data, train, test, sample_seed, budget):
    observed = data.points["observed_dbm"].to_numpy(float)
    xy = data.points[["x", "y"]].to_numpy(float)
    cv_labels = spatial_cv_labels(
        xy, train, derive_seed(sample_seed, data.band, budget, "cv")
    )
    predictions = {}
    parameters = {}

    started = time.perf_counter()
    gaussian_bw, selection_info = choose_gaussian_bandwidth(
        xy, observed, train, cv_labels
    )
    gaussian_test = gaussian_for_masks(xy, observed, train, test, gaussian_bw)
    predictions["GAUSSIAN"] = gaussian_test
    parameters["GAUSSIAN"] = {
        "bandwidth_m": gaussian_bw,
        **selection_info,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
        "physical_fallback_n": 0,
    }

    family_names = {"U2": "BASE", "WEDT_P": "S4W", "ONETWIN": "S5"}
    for method, family in family_names.items():
        started = time.perf_counter()
        selected, selection_info = choose_standalone_params(
            data, observed, train, cv_labels, family, gaussian_bw
        )
        prediction, fit_info = final_standalone(
            data, observed, train, test, gaussian_test, family, selected
        )
        predictions[method] = prediction
        parameters[method] = {
            "family": family,
            "local_k": selected[0],
            "local_power": selected[1],
            **selection_info,
            **fit_info,
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb(),
            "physical_fallback_n": int(fit_info.get("test_no_path_n", 0)),
        }

    for method, family in family_names.items():
        name = f"TWC_{method}"
        started = time.perf_counter()
        selected, selection_info = choose_twc_params_sparse(
            data, observed, train, cv_labels, family, gaussian_bw
        )
        prediction, fit_info = final_twc(
            data, observed, train, test, gaussian_test, family, selected
        )
        predictions[name] = prediction
        parameters[name] = {
            "family": family,
            "class_ridge": selected[0],
            "class_penalty_m": selected[1],
            "local_k": selected[2],
            "local_power": selected[3],
            **selection_info,
            **fit_info,
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb(),
            "physical_fallback_n": int(fit_info.get("test_no_path_n", 0)),
        }

    for method, family in family_names.items():
        name = f"TWC_{method}_GAUSSIAN_RESIDUAL"
        started = time.perf_counter()
        selected, selection_info = choose_residual_params(
            data, observed, train, cv_labels, family
        )
        prediction, fit_info = final_residual(
            data, observed, train, test, family, selected
        )
        predictions[name] = prediction
        parameters[name] = {
            "family": family,
            "class_ridge": selected[0],
            "bandwidth_m": selected[1],
            "shrink": selected[2],
            **selection_info,
            **fit_info,
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mb": peak_rss_mb(),
            "physical_fallback_n": int(fit_info.get("test_no_path_n", 0)),
        }

    if tuple(predictions) != METHODS:
        raise RuntimeError(f"method order mismatch: {tuple(predictions)}")
    for method, prediction in predictions.items():
        if len(prediction) != int(test.sum()) or not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite or wrong-size prediction for {method}")
    return predictions, parameters


def strict_test_mask(xy, train, test):
    keys = cell_keys(xy)
    train_cells = {tuple(value) for value in keys[train]}
    strict = test.copy()
    for index in np.flatnonzero(test):
        if tuple(keys[index]) in train_cells:
            strict[index] = False
    return strict


def error_metrics(observed, predicted, nearest_distance):
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    nearest_distance = np.asarray(nearest_distance, float)
    valid = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(nearest_distance)
    error = predicted[valid] - observed[valid]
    absolute = np.abs(error)
    distance = nearest_distance[valid]
    if not len(error):
        return {
            "test_n": 0,
            "rmse_db": np.nan,
            "mae_db": np.nan,
            "median_abs_db": np.nan,
            "p90_abs_db": np.nan,
            "p95_abs_db": np.nan,
            "p99_abs_db": np.nan,
            "max_abs_db": np.nan,
            "gt8_n": 0,
            "gt10_n": 0,
            "gt15_n": 0,
            "gt8_rate": np.nan,
            "gt10_rate": np.nan,
            "gt15_rate": np.nan,
            "nearest_mean_m": np.nan,
            "nearest_median_m": np.nan,
            "nearest_p90_m": np.nan,
        }
    return {
        "test_n": int(len(error)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(absolute)),
        "median_abs_db": float(np.median(absolute)),
        "p90_abs_db": float(np.percentile(absolute, 90)),
        "p95_abs_db": float(np.percentile(absolute, 95)),
        "p99_abs_db": float(np.percentile(absolute, 99)),
        "max_abs_db": float(np.max(absolute)),
        "gt8_n": int(np.sum(absolute > 8.0)),
        "gt10_n": int(np.sum(absolute > 10.0)),
        "gt15_n": int(np.sum(absolute > 15.0)),
        "gt8_rate": float(np.mean(absolute > 8.0)),
        "gt10_rate": float(np.mean(absolute > 10.0)),
        "gt15_rate": float(np.mean(absolute > 15.0)),
        "nearest_mean_m": float(np.mean(distance)),
        "nearest_median_m": float(np.median(distance)),
        "nearest_p90_m": float(np.percentile(distance, 90)),
    }


def load_track(data_root, rt_root, track, core_csv):
    if track == "ew_core":
        if core_csv is None:
            raise ValueError("--core-csv required for ew_core")
        core = pd.read_csv(core_csv, low_memory=False)
        if len(core) != sum(TRACK_COUNTS["ew_core"].values()):
            raise RuntimeError("EW_CORE row-count contract failed")
        core_ids = set(core["point_id"].astype(str))
    else:
        core_ids = None

    output = {}
    for band in BANDS:
        points, configs, gains, los, tx = load_band(data_root, rt_root, band)
        source_row_index = np.arange(len(points), dtype=np.int64)
        if core_ids is not None:
            keep = points["point_id"].astype(str).isin(core_ids).to_numpy()
            points = points.loc[keep].reset_index(drop=True)
            gains = np.asarray(gains)[:, keep]
            los = np.asarray(los)[keep]
            source_row_index = source_row_index[keep]
        expected = TRACK_COUNTS[track][band]
        if len(points) != expected or points["point_id"].astype(str).nunique() != expected:
            raise RuntimeError(f"{track}/{band} point contract failed")
        output[band] = BandData(
            band=band,
            points=points,
            configs=configs,
            gains=np.asarray(gains, float),
            los=np.asarray(los, bool),
            tx=np.asarray(tx, float),
            source_row_index=source_row_index,
        )
    return output


def save_sampling_manifest(path, track, seed_id, sample_seed, orders, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(
            stream,
            track=np.asarray(track),
            seed_id=np.asarray(seed_id, dtype=np.int32),
            sample_seed=np.asarray(sample_seed, dtype=np.uint32),
            n41_track_row_index=orders["n41"].astype(np.int32),
            n79_track_row_index=orders["n79"].astype(np.int32),
            n41_source_row_index=data["n41"].source_row_index[orders["n41"]].astype(np.int32),
            n79_source_row_index=data["n79"].source_row_index[orders["n79"]].astype(np.int32),
        )
    temp.replace(path)


def run_budget(*, track, seed_id, sample_seed, budget, allocation, data, orders):
    band_results = {}
    all_parameters = {}
    for band in BANDS:
        band_data = data[band]
        train = np.zeros(len(band_data.points), dtype=bool)
        train[orders[band][: allocation[band]]] = True
        test = ~train
        xy = band_data.points[["x", "y"]].to_numpy(float)
        predictions, parameters = fit_predict_methods(
            band_data, train, test, sample_seed, budget
        )
        nearest_query, _ = cKDTree(xy[train]).query(xy[test], k=1, workers=1)
        nearest = np.full(len(xy), np.nan, float)
        nearest[test] = np.asarray(nearest_query, float)
        strict = strict_test_mask(xy, train, test)
        full_predictions = {}
        for method in METHODS:
            vector = np.full(len(xy), np.nan, float)
            vector[test] = predictions[method]
            full_predictions[method] = vector
        band_results[band] = {
            "observed": band_data.points["observed_dbm"].to_numpy(float),
            "train": train,
            "test": test,
            "strict": strict,
            "nearest": nearest,
            "predictions": full_predictions,
        }
        all_parameters[band] = {
            "train_n": int(train.sum()),
            "test_n": int(test.sum()),
            "strict_test_n": int(strict.sum()),
            "methods": parameters,
        }

    rows = []
    for scope in (*BANDS, "all"):
        scope_bands = (scope,) if scope in BANDS else BANDS
        for mode in ("ordinary_complement", "strict_new_10m_cell"):
            masks = [
                band_results[band]["test"]
                if mode == "ordinary_complement"
                else band_results[band]["strict"]
                for band in scope_bands
            ]
            expected_test_n = int(sum(mask.sum() for mask in masks))
            train_n = int(sum(band_results[band]["train"].sum() for band in scope_bands))
            for method in METHODS:
                observed = np.concatenate(
                    [band_results[band]["observed"][mask] for band, mask in zip(scope_bands, masks)]
                )
                predicted = np.concatenate(
                    [band_results[band]["predictions"][method][mask] for band, mask in zip(scope_bands, masks)]
                )
                nearest = np.concatenate(
                    [band_results[band]["nearest"][mask] for band, mask in zip(scope_bands, masks)]
                )
                metric = error_metrics(observed, predicted, nearest)
                runtime = float(
                    sum(all_parameters[band]["methods"][method]["runtime_seconds"] for band in scope_bands)
                )
                fallback_n = int(
                    sum(all_parameters[band]["methods"][method].get("physical_fallback_n", 0) for band in scope_bands)
                )
                rows.append(
                    {
                        "track": track,
                        "seed_id": int(seed_id),
                        "sample_seed": int(sample_seed),
                        "budget": int(budget),
                        "scope": scope,
                        "evaluation_mode": mode,
                        "method": method,
                        "train_n": train_n,
                        "expected_test_n": expected_test_n,
                        "rank_eligible": bool(
                            mode == "ordinary_complement"
                            or (scope == "all" and expected_test_n >= STRICT_MIN_RANK_N)
                        ),
                        "runtime_seconds": runtime,
                        "peak_rss_mb": peak_rss_mb(),
                        "physical_fallback_n": fallback_n,
                        **metric,
                    }
                )
    return pd.DataFrame(rows), all_parameters


def parse_budgets(raw, track):
    allowed = TRACK_BUDGETS[track]
    if not raw:
        return allowed
    selected = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not selected or any(value not in allowed for value in selected):
        raise ValueError(f"budgets must be subset of {allowed}")
    return tuple(value for value in allowed if value in set(selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--rt-root", required=True, type=Path)
    parser.add_argument("--core-csv", type=Path)
    parser.add_argument("--seed-csv", required=True, type=Path)
    parser.add_argument("--track", choices=tuple(TRACK_COUNTS), required=True)
    parser.add_argument("--seed-id", required=True, type=int)
    parser.add_argument("--budgets")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    seeds = pd.read_csv(args.seed_csv)
    if len(seeds) != 128 or seeds["sample_seed"].nunique() != 128:
        raise RuntimeError("seed manifest contract failed")
    if not ((seeds["sample_seed"] >= 0) & (seeds["sample_seed"] <= 2**32 - 1)).all():
        raise RuntimeError("seed outside uint32")
    selected_seed = seeds.loc[seeds["seed_id"].astype(int).eq(args.seed_id)]
    if len(selected_seed) != 1:
        raise RuntimeError(f"unknown seed_id {args.seed_id}")
    sample_seed = int(selected_seed.iloc[0]["sample_seed"])
    budgets = parse_budgets(args.budgets, args.track)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_dir = args.output_dir / "jobs" / args.track / f"seed_{args.seed_id:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    state_path = seed_dir / "seed_state.json"
    atomic_json(
        state_path,
        {
            "status": "running",
            "track": args.track,
            "seed_id": args.seed_id,
            "sample_seed": sample_seed,
            "budgets": budgets,
            "started_at": utc_now(),
            "pid": os.getpid(),
        },
    )

    try:
        data = load_track(args.data_root, args.rt_root, args.track, args.core_csv)
        max_allocation = allocate_budget(max(budgets), TRACK_COUNTS[args.track])
        orders = {
            band: randomized_spatial_order(
                data[band].points[["x", "y"]].to_numpy(float),
                max_allocation[band],
                derive_seed(sample_seed, args.track, band, "sample_order"),
            )
            for band in BANDS
        }
        manifest = args.output_dir / "manifests" / args.track / f"seed_{args.seed_id:03d}.npz"
        save_sampling_manifest(manifest, args.track, args.seed_id, sample_seed, orders, data)

        completed = []
        for budget in budgets:
            stem = seed_dir / f"budget_{budget:06d}"
            metrics_path = stem.with_name(stem.name + "_metrics.csv")
            params_path = stem.with_name(stem.name + "_parameters.json")
            done_path = stem.with_name(stem.name + "_state.json")
            if metrics_path.exists() and params_path.exists() and done_path.exists():
                prior = json.loads(done_path.read_text(encoding="utf-8"))
                if prior.get("status") == "complete":
                    completed.append(budget)
                    print(f"SKIP track={args.track} seed={args.seed_id:03d} budget={budget}", flush=True)
                    continue
            allocation = allocate_budget(budget, TRACK_COUNTS[args.track])
            started = time.perf_counter()
            atomic_json(
                done_path,
                {
                    "status": "running",
                    "track": args.track,
                    "seed_id": args.seed_id,
                    "budget": budget,
                    "allocation": allocation,
                    "started_at": utc_now(),
                },
            )
            metrics_frame, parameters = run_budget(
                track=args.track,
                seed_id=args.seed_id,
                sample_seed=sample_seed,
                budget=budget,
                allocation=allocation,
                data=data,
                orders=orders,
            )
            expected_rows = len(METHODS) * 2 * 3
            if len(metrics_frame) != expected_rows:
                raise RuntimeError(f"metric row contract failed {len(metrics_frame)} != {expected_rows}")
            ordinary = metrics_frame[metrics_frame["evaluation_mode"].eq("ordinary_complement")]
            if not np.isfinite(ordinary["rmse_db"]).all():
                raise RuntimeError("ordinary metric contains non-finite RMSE")
            atomic_csv(metrics_path, metrics_frame)
            atomic_json(
                params_path,
                {
                    "track": args.track,
                    "seed_id": args.seed_id,
                    "sample_seed": sample_seed,
                    "budget": budget,
                    "allocation": allocation,
                    "cell_size_m": CELL_SIZE_M,
                    "top_farthest_fraction": TOP_FARTHEST_FRACTION,
                    "minimum_physical_train": MIN_PHYSICAL_TRAIN,
                    "parameters": parameters,
                },
            )
            elapsed = time.perf_counter() - started
            atomic_json(
                done_path,
                {
                    "status": "complete",
                    "track": args.track,
                    "seed_id": args.seed_id,
                    "budget": budget,
                    "allocation": allocation,
                    "metric_rows": len(metrics_frame),
                    "elapsed_seconds": elapsed,
                    "completed_at": utc_now(),
                },
            )
            completed.append(budget)
            print(
                f"PASS track={args.track} seed={args.seed_id:03d} budget={budget} elapsed={elapsed:.3f}s",
                flush=True,
            )
        atomic_json(
            state_path,
            {
                "status": "complete",
                "track": args.track,
                "seed_id": args.seed_id,
                "sample_seed": sample_seed,
                "budgets": budgets,
                "completed_budgets": completed,
                "completed_at": utc_now(),
                "peak_rss_mb": peak_rss_mb(),
            },
        )
    except Exception as error:
        atomic_json(
            state_path,
            {
                "status": "failed",
                "track": args.track,
                "seed_id": args.seed_id,
                "sample_seed": sample_seed,
                "budgets": budgets,
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
