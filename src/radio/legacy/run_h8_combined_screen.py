#!/usr/bin/env python3
"""Leakage-safe H8 fixed-source combination screen on all H6 raw points.

This is the first H8 inner-loop experiment. It reuses the V7-r4 RT cache,
adds the TWC-style LoS/NLoS structural calibration, and evaluates constrained
expert fusion plus two explicitly separated selective tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize
from scipy.spatial import cKDTree


METHOD_FAMILY = {"U2": "BASE", "WEDT_P": "S4W", "ONETWIN": "S5"}
FIXED_TX = {
    "n41": np.asarray([134.42377217610678, 45.10205841064453, 37.0], float),
    "n79": np.asarray([-136.84656524658203, 188.49261474609375, 17.0], float),
}
QUANTILES = (10, 25, 50, 75, 90)
IDW_K = (8, 16, 32, 64)
IDW_P = (1.0, 2.0)
TWC_RIDGES = (1.0, 5.0, 20.0)
TWC_CLASS_PENALTIES_M = (0.0, 40.0, 100.0)


def stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(observed: np.ndarray, predicted: np.ndarray, total_n: int | None = None) -> dict:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    error = predicted[valid] - observed[valid]
    absolute = np.abs(error)
    denominator = len(observed) if total_n is None else int(total_n)
    return {
        "total_n": int(denominator),
        "predicted_n": int(valid.sum()),
        "coverage": float(valid.sum() / max(1, denominator)),
        "rmse_db": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
        "mae_db": float(np.mean(absolute)) if len(error) else np.nan,
        "p90_abs_db": float(np.percentile(absolute, 90)) if len(error) else np.nan,
        "bias_db": float(np.mean(error)) if len(error) else np.nan,
    }


def inner_group_map(groups: np.ndarray, mask: np.ndarray, folds: int = 3) -> dict[str, int]:
    ordered = sorted(set(groups[mask]), key=stable_rank)
    return {str(group): index % folds for index, group in enumerate(ordered)}


def query_neighbors(train_features: np.ndarray, query_features: np.ndarray, max_k: int):
    if len(train_features) == 0:
        raise ValueError("empty neighbor training set")
    k = min(int(max_k), len(train_features))
    distance, index = cKDTree(np.asarray(train_features, float)).query(
        np.asarray(query_features, float), k=k, workers=-1
    )
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    return np.asarray(distance, float), np.asarray(index, int)


def idw(values: np.ndarray, distance: np.ndarray, index: np.ndarray, k: int, power: float) -> np.ndarray:
    width = min(int(k), index.shape[1])
    d = distance[:, :width]
    v = np.asarray(values, float)[index[:, :width]]
    zero = d <= 1e-9
    output = np.empty(len(d), dtype=float)
    exact = zero.any(axis=1)
    if exact.any():
        count = zero[exact].sum(axis=1)
        output[exact] = np.sum(np.where(zero[exact], v[exact], 0.0), axis=1) / count
    if (~exact).any():
        weight = 1.0 / np.maximum(d[~exact], 1e-6) ** float(power)
        output[~exact] = np.sum(weight * v[~exact], axis=1) / np.sum(weight, axis=1)
    return output


def augmented_xy(xy: np.ndarray, nlos: np.ndarray, class_penalty_m: float) -> np.ndarray:
    return np.column_stack([np.asarray(xy, float), np.asarray(nlos, float) * class_penalty_m])


def fit_class_correction(distance_m, nlos, residual_db, ridge: float):
    log_distance = np.log10(np.maximum(np.asarray(distance_m, float), 1.0))
    center = float(log_distance.mean())
    q = log_distance - center
    nlos = np.asarray(nlos, float)
    design = np.column_stack([np.ones(len(q)), q, nlos, q * nlos])
    penalty = np.diag([0.0, ridge, ridge, ridge])
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(residual_db, float))
    return center, coef


def apply_class_correction(distance_m, nlos, center, coefficients):
    q = np.log10(np.maximum(np.asarray(distance_m, float), 1.0)) - float(center)
    nlos = np.asarray(nlos, float)
    design = np.column_stack([np.ones(len(q)), q, nlos, q * nlos])
    return design @ np.asarray(coefficients, float)


def load_band(data_root: Path, rt_root: Path, band: str):
    points = pd.read_csv(data_root / "cache" / band / "points.csv")
    points["date"] = pd.to_numeric(points["date"], errors="coerce").astype("Int64").astype(str).str.zfill(4)
    mapping = np.load(data_root / "cache" / band / "point_to_unique.npy").astype(int)
    meta = json.loads((rt_root / "cache" / band / "configs.json").read_text(encoding="utf-8"))
    configs = meta["configs"]
    unique_gain = [np.load(rt_root / "cache" / band / f"{config['id']}.npy") for config in configs]
    gains = np.asarray([gain[mapping] for gain in unique_gain], float)
    los_path = rt_root / "cache" / band / "los_unique.npy"
    if not los_path.exists():
        raise FileNotFoundError(f"missing H8 direct-path cache: {los_path}")
    los = np.load(los_path).astype(bool)[mapping]
    tx = np.asarray(meta.get("tx_position", FIXED_TX[band]), float)
    return points, configs, gains, los, tx


def family_indices(configs: list[dict], family: str) -> list[int]:
    if family == "BASE":
        return [index for index, config in enumerate(configs) if config["id"] == "AUTO_BASE"]
    return [index for index, config in enumerate(configs) if config["family"] == family]


def fit_physical(
    configs: list[dict],
    gains: np.ndarray,
    observed: np.ndarray,
    distance: np.ndarray,
    nlos: np.ndarray,
    fit_mask: np.ndarray,
    family: str,
    *,
    twc: bool,
    class_ridge: float,
) -> tuple[np.ndarray, dict]:
    best = None
    for index in family_indices(configs, family):
        finite = fit_mask & np.isfinite(gains[index])
        if finite.sum() < 100:
            continue
        if twc:
            center, coef = fit_class_correction(
                distance[finite], nlos[finite], observed[finite] - gains[index, finite], class_ridge
            )
            correction = apply_class_correction(distance, nlos, center, coef)
            prediction = gains[index] + correction
            params = {"center": center, "coefficients": coef.tolist()}
        else:
            bias = float(np.median(observed[finite] - gains[index, finite]))
            prediction = gains[index] + bias
            params = {"bias": bias}
        error = prediction[finite] - observed[finite]
        candidate = (float(np.sqrt(np.mean(error**2))), configs[index]["id"], index, prediction, params)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"no usable physical config for {family}")
    return np.asarray(best[3], float), {
        "config_id": str(best[1]),
        "fit_rmse_db": float(best[0]),
        **best[4],
    }


def signal_idw_predict(
    xy: np.ndarray,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    query_mask: np.ndarray,
    k: int,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance, index = query_neighbors(xy[fit_mask], xy[query_mask], max_k=k)
    return idw(observed[fit_mask], distance, index, k, power), distance[:, 0]


def choose_signal_params(xy, observed, groups, outer_train) -> tuple[int, float, float]:
    fold_map = inner_group_map(groups, outer_train)
    aggregate = {(k, p): [0.0, 0] for k in IDW_K for p in IDW_P}
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        distance, index = query_neighbors(xy[fit], xy[valid], max_k=max(IDW_K))
        for k in IDW_K:
            for power in IDW_P:
                pred = idw(observed[fit], distance, index, k, power)
                aggregate[(k, power)][0] += float(np.sum((pred - observed[valid]) ** 2))
                aggregate[(k, power)][1] += int(valid.sum())
    ranked = [
        (math.sqrt(sse / count), k, power)
        for (k, power), (sse, count) in aggregate.items() if count > 0
    ]
    if not ranked:
        return 16, 2.0, np.nan
    rmse, k, power = min(ranked)
    return int(k), float(power), float(rmse)


def expert_predict(
    points: pd.DataFrame,
    configs: list[dict],
    gains: np.ndarray,
    los: np.ndarray,
    tx: np.ndarray,
    observed: np.ndarray,
    fit_mask: np.ndarray,
    query_mask: np.ndarray,
    family: str,
    *,
    twc: bool,
    class_ridge: float,
    local_k: int,
    local_power: float,
    class_penalty_m: float,
    signal_k: int,
    signal_power: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    xy = points[["x", "y"]].to_numpy(float)
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - tx[None, :], axis=1)
    nlos = ~np.asarray(los, bool)
    physical, params = fit_physical(
        configs, gains, observed, distance, nlos, fit_mask, family,
        twc=twc, class_ridge=class_ridge,
    )
    finite_fit = fit_mask & np.isfinite(physical)
    query_index = np.flatnonzero(query_mask)
    prediction = np.full(len(query_index), np.nan, float)
    query_has_path = np.isfinite(physical[query_mask])
    if finite_fit.any() and query_has_path.any():
        train_feature = augmented_xy(xy[finite_fit], nlos[finite_fit], class_penalty_m)
        query_feature = augmented_xy(xy[query_mask][query_has_path], nlos[query_mask][query_has_path], class_penalty_m)
        distance_local, index_local = query_neighbors(train_feature, query_feature, max_k=local_k)
        residual = observed[finite_fit] - physical[finite_fit]
        correction = idw(residual, distance_local, index_local, local_k, local_power)
        prediction[query_has_path] = physical[query_mask][query_has_path] + correction
    if (~query_has_path).any():
        fill_mask = np.zeros(len(points), dtype=bool)
        fill_mask[query_index[~query_has_path]] = True
        fill, _ = signal_idw_predict(xy, observed, fit_mask, fill_mask, signal_k, signal_power)
        prediction[~query_has_path] = fill
    params.update({
        "twc": bool(twc), "class_ridge": float(class_ridge),
        "local_k": int(local_k), "local_power": float(local_power),
        "class_penalty_m": float(class_penalty_m),
        "signal_k": int(signal_k), "signal_power": float(signal_power),
    })
    return prediction, physical[query_mask], params


def choose_twc_params(
    points, configs, gains, los, tx, observed, groups, outer_train, signal_params
) -> dict:
    xy = points[["x", "y"]].to_numpy(float)
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - tx[None, :], axis=1)
    nlos = ~np.asarray(los, bool)
    fold_map = inner_group_map(groups, outer_train)
    scores = {
        (ridge, penalty, k, power): [0.0, 0]
        for ridge in TWC_RIDGES
        for penalty in TWC_CLASS_PENALTIES_M
        for k in IDW_K
        for power in IDW_P
    }
    signal_k, signal_power = signal_params
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        signal_fill, _ = signal_idw_predict(xy, observed, fit, valid, signal_k, signal_power)
        for ridge in TWC_RIDGES:
            physical, _ = fit_physical(
                configs, gains, observed, distance, nlos, fit, "BASE", twc=True, class_ridge=ridge
            )
            finite_fit = fit & np.isfinite(physical)
            has_path = np.isfinite(physical[valid])
            for penalty in TWC_CLASS_PENALTIES_M:
                distance_local = index_local = None
                if finite_fit.any() and has_path.any():
                    distance_local, index_local = query_neighbors(
                        augmented_xy(xy[finite_fit], nlos[finite_fit], penalty),
                        augmented_xy(xy[valid][has_path], nlos[valid][has_path], penalty),
                        max_k=max(IDW_K),
                    )
                residual = observed[finite_fit] - physical[finite_fit]
                for k in IDW_K:
                    for power in IDW_P:
                        pred = signal_fill.copy()
                        if has_path.any():
                            pred[has_path] = physical[valid][has_path] + idw(
                                residual, distance_local, index_local, k, power
                            )
                        key = (ridge, penalty, k, power)
                        scores[key][0] += float(np.sum((pred - observed[valid]) ** 2))
                        scores[key][1] += int(valid.sum())
    ranked = [
        (math.sqrt(sse / count), ridge, penalty, k, power)
        for (ridge, penalty, k, power), (sse, count) in scores.items() if count > 0
    ]
    if not ranked:
        return {"inner_rmse_db": np.nan, "class_ridge": 5.0, "class_penalty_m": 0.0, "local_k": 16, "local_power": 2.0}
    rmse, ridge, penalty, k, power = min(ranked)
    return {
        "inner_rmse_db": float(rmse), "class_ridge": float(ridge),
        "class_penalty_m": float(penalty), "local_k": int(k), "local_power": float(power),
    }


def crossfit_expert(
    points, configs, gains, los, tx, observed, groups, outer_train, family, params, *, twc
) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(len(points), np.nan, float)
    physical = np.full(len(points), np.nan, float)
    fold_map = inner_group_map(groups, outer_train)
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        pred, phys, _ = expert_predict(
            points, configs, gains, los, tx, observed, fit, valid, family,
            twc=twc,
            class_ridge=float(params.get("class_ridge", 5.0)),
            local_k=int(params["local_k"]),
            local_power=float(params["local_power"]),
            class_penalty_m=float(params.get("class_penalty_m", 0.0)),
            signal_k=int(params["signal_k"]),
            signal_power=float(params["signal_power"]),
        )
        output[valid] = pred
        physical[valid] = phys
    return output, physical


def crossfit_signal(points, observed, groups, outer_train, k, power) -> np.ndarray:
    xy = points[["x", "y"]].to_numpy(float)
    output = np.full(len(points), np.nan, float)
    fold_map = inner_group_map(groups, outer_train)
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        output[valid], _ = signal_idw_predict(xy, observed, fit, valid, k, power)
    return output


def fit_stack_weights(predictions: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, float, float]:
    predictions = np.asarray(predictions, float)
    observed = np.asarray(observed, float)
    valid = np.isfinite(observed) & np.isfinite(predictions).all(axis=1)
    x = predictions[valid]
    y = observed[valid]
    if len(y) < 20:
        weight = np.full(predictions.shape[1], 1.0 / predictions.shape[1])
        return weight, 0.0, np.nan

    def objective(weight):
        blended = x @ weight
        bias = np.median(y - blended)
        return float(np.mean((blended + bias - y) ** 2))

    initial = np.full(x.shape[1], 1.0 / x.shape[1])
    result = minimize(
        objective, initial, method="SLSQP",
        bounds=[(0.0, 1.0)] * x.shape[1],
        constraints=[{"type": "eq", "fun": lambda weight: np.sum(weight) - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weight = np.asarray(result.x if result.success else initial, float)
    weight = np.maximum(weight, 0.0)
    weight /= np.sum(weight)
    bias = float(np.median(y - x @ weight))
    return weight, bias, float(math.sqrt(objective(weight)))


def choose_stack_local(xy, residual, groups, outer_train) -> tuple[int, float, float]:
    fold_map = inner_group_map(groups, outer_train)
    scores = {(k, p): [0.0, 0] for k in IDW_K for p in IDW_P}
    for fold in range(3):
        fit = outer_train & np.isfinite(residual) & np.array([fold_map.get(str(g), -1) != fold for g in groups])
        valid = outer_train & np.isfinite(residual) & np.array([fold_map.get(str(g), -1) == fold for g in groups])
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        distance, index = query_neighbors(xy[fit], xy[valid], max_k=max(IDW_K))
        for k in IDW_K:
            for power in IDW_P:
                pred = idw(residual[fit], distance, index, k, power)
                scores[(k, power)][0] += float(np.sum((pred - residual[valid]) ** 2))
                scores[(k, power)][1] += int(valid.sum())
    ranked = [
        (math.sqrt(sse / count), k, power)
        for (k, power), (sse, count) in scores.items() if count > 0
    ]
    if not ranked:
        return 16, 2.0, np.nan
    rmse, k, power = min(ranked)
    return int(k), float(power), float(rmse)


def apply_stack_local(xy, residual, outer_train, test, k, power) -> np.ndarray:
    fit = outer_train & np.isfinite(residual)
    distance, index = query_neighbors(xy[fit], xy[test], max_k=k)
    return idw(residual[fit], distance, index, k, power)


def neighbor_error_feature(xy, train_mask, query_mask, train_error, k: int = 16):
    fit = train_mask & np.isfinite(train_error)
    distance, index = query_neighbors(xy[fit], xy[query_mask], max_k=k)
    predicted = idw(np.abs(train_error[fit]), distance, index, k, 1.0)
    return predicted, distance[:, 0]


def fit_confidence_model(train_features: np.ndarray, target_abs_error: np.ndarray):
    feature = np.asarray(train_features, float)
    target = np.asarray(target_abs_error, float)
    scale = np.maximum(np.nanmedian(feature, axis=0), 1e-3)
    design = np.column_stack([np.ones(len(feature)), feature / scale])
    ridge = 1.0
    regularizer = np.zeros((feature.shape[1], feature.shape[1] + 1), float)
    regularizer[:, 1:] = math.sqrt(ridge) * np.eye(feature.shape[1])
    augmented_x = np.vstack([design, regularizer])
    augmented_y = np.r_[target, np.zeros(feature.shape[1])]
    lower = np.r_[-np.inf, np.zeros(feature.shape[1])]
    upper = np.full(feature.shape[1] + 1, np.inf)
    result = lsq_linear(augmented_x, augmented_y, bounds=(lower, upper))
    return scale, result.x


def confidence_scores(
    xy, outer_train, test, observed, oof_pred, test_pred,
    disagreement_oof, disagreement_test, path_missing_oof, path_missing_test,
):
    oof_error = oof_pred - observed
    neighbor_train, nearest_train = neighbor_error_feature(xy, outer_train, outer_train, oof_error)
    # Avoid a point using its own OOF error as an exact-distance confidence feature.
    exact = nearest_train <= 1e-9
    if exact.any():
        distance, index = query_neighbors(xy[outer_train], xy[outer_train][exact], max_k=17)
        values = np.abs(oof_error[outer_train])
        neighbor_train[exact] = idw(values, distance[:, 1:], index[:, 1:], 16, 1.0)
        nearest_train[exact] = distance[:, 1]
    neighbor_test, nearest_test = neighbor_error_feature(xy, outer_train, test, oof_error)
    feature_train = np.column_stack([
        neighbor_train,
        disagreement_oof[outer_train],
        nearest_train,
        np.asarray(path_missing_oof[outer_train], float),
    ])
    feature_test = np.column_stack([
        neighbor_test,
        disagreement_test,
        nearest_test,
        np.asarray(path_missing_test, float),
    ])
    valid = np.isfinite(feature_train).all(axis=1) & np.isfinite(oof_error[outer_train])
    scale, coef = fit_confidence_model(feature_train[valid], np.abs(oof_error[outer_train][valid]))
    train_score = np.column_stack([np.ones(valid.sum()), feature_train[valid] / scale]) @ coef
    test_score = np.column_stack([np.ones(len(feature_test)), feature_test / scale]) @ coef
    thresholds = {q: float(np.quantile(train_score, q / 100.0)) for q in QUANTILES}
    return np.maximum(test_score, 0.0), thresholds, {"scale": scale.tolist(), "coefficients": coef.tolist()}


def source_scores(observed, outer_train, test, physical_oof, physical_test):
    train_residual = observed[outer_train] - physical_oof[outer_train]
    finite = np.isfinite(train_residual)
    center = float(np.median(train_residual[finite]))
    mad = float(np.median(np.abs(train_residual[finite] - center)))
    scale = max(1.4826 * mad, 0.5)
    train_score = np.full(len(train_residual), 1e6, float)
    train_score[finite] = np.abs(train_residual[finite] - center) / scale
    test_residual = observed[test] - physical_test
    test_score = np.full(len(test_residual), 1e6, float)
    finite_test = np.isfinite(test_residual)
    test_score[finite_test] = np.abs(test_residual[finite_test] - center) / scale
    thresholds = {q: float(np.quantile(train_score[np.isfinite(train_score) & (train_score < 1e5)], q / 100.0)) for q in QUANTILES}
    return test_score, thresholds, {"center_db": center, "robust_scale_db": scale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--rt-root", required=True, type=Path)
    parser.add_argument("--h7-score-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_frame = pd.read_csv(args.h7_score_dir / "trajectory_fold_assignment.csv")
    fold_map = dict(zip(fold_frame["trajectory_group"].astype(str), fold_frame["outer_fold"].astype(int)))
    h7_long = pd.read_csv(args.h7_score_dir / "point_predictions.csv")
    h7_long = h7_long[h7_long["variant"].eq("IDW_FILL")]
    h7_wide = h7_long.pivot(index="point_id", columns="method", values="predicted_dbm")
    h7_choices = pd.read_csv(args.h7_score_dir / "method_choices.csv")

    all_output = []
    parameter_rows = []
    method_names = None
    for band in ("n41", "n79"):
        points, configs, gains, los, tx = load_band(args.data_root, args.rt_root, band)
        observed = pd.to_numeric(points["observed_dbm"], errors="coerce").to_numpy(float)
        xy = points[["x", "y"]].to_numpy(float)
        groups = points["trajectory_group"].astype(str).to_numpy()
        outer_fold = np.array([fold_map[str(group)] for group in groups], int)
        band_h7 = h7_wide.reindex(points["point_id"].astype(str)).copy()
        if band_h7.isna().any().any():
            raise RuntimeError(f"H7 standalone predictions incomplete for {band}")

        for fold in range(5):
            outer_train = outer_fold != fold
            test = outer_fold == fold
            if outer_train.sum() < 100 or test.sum() < 20:
                raise RuntimeError(f"insufficient H8 fold {band}/{fold}")
            signal_k, signal_power, signal_inner_rmse = choose_signal_params(
                xy, observed, groups, outer_train
            )
            signal_test, _ = signal_idw_predict(
                xy, observed, outer_train, test, signal_k, signal_power
            )
            signal_oof = crossfit_signal(
                points, observed, groups, outer_train, signal_k, signal_power
            )
            twc_param = choose_twc_params(
                points, configs, gains, los, tx, observed, groups, outer_train,
                (signal_k, signal_power),
            )
            twc_param.update({"signal_k": signal_k, "signal_power": signal_power})

            test_pred: dict[str, np.ndarray] = {
                "TRAIN_MEAN": np.full(test.sum(), float(np.mean(observed[outer_train]))),
                "SIGNAL_IDW": signal_test,
            }
            oof_pred: dict[str, np.ndarray] = {
                "TRAIN_MEAN": np.full(len(points), np.nan, float),
                "SIGNAL_IDW": signal_oof,
            }
            fold_inner = inner_group_map(groups, outer_train)
            for inner in range(3):
                valid = outer_train & np.array([fold_inner.get(str(g), -1) == inner for g in groups])
                fit = outer_train & ~valid
                oof_pred["TRAIN_MEAN"][valid] = float(np.mean(observed[fit]))

            # Common RT baseline with no local residual.
            xyz = points[["x", "y", "z"]].to_numpy(float)
            distance = np.linalg.norm(xyz - tx[None, :], axis=1)
            nlos = ~los
            rt_physical, rt_params = fit_physical(
                configs, gains, observed, distance, nlos, outer_train, "BASE", twc=False, class_ridge=5.0
            )
            test_pred["RT_BIAS"] = np.where(np.isfinite(rt_physical[test]), rt_physical[test], signal_test)
            rt_oof = np.full(len(points), np.nan, float)
            for inner in range(3):
                valid = outer_train & np.array([fold_inner.get(str(g), -1) == inner for g in groups])
                fit = outer_train & ~valid
                phys, _ = fit_physical(
                    configs, gains, observed, distance, nlos, fit, "BASE", twc=False, class_ridge=5.0
                )
                inner_signal, _ = signal_idw_predict(xy, observed, fit, valid, signal_k, signal_power)
                rt_oof[valid] = np.where(np.isfinite(phys[valid]), phys[valid], inner_signal)
            oof_pred["RT_BIAS"] = rt_oof

            standalone_oof, twc_oof = {}, {}
            standalone_test, twc_test = {}, {}
            twc_physical_oof, twc_physical_test = {}, {}
            for method, family in METHOD_FAMILY.items():
                choice = h7_choices[
                    h7_choices["band"].eq(band)
                    & h7_choices["outer_fold"].eq(fold)
                    & h7_choices["method"].eq(method)
                ].iloc[0]
                standalone_params = {
                    "local_k": int(choice["idw_k"]),
                    "local_power": float(choice["idw_power"]),
                    "class_penalty_m": 0.0,
                    "class_ridge": 5.0,
                    "signal_k": signal_k,
                    "signal_power": signal_power,
                }
                standalone_test[method] = band_h7.loc[test, method].to_numpy(float)
                standalone_oof[method], _ = crossfit_expert(
                    points, configs, gains, los, tx, observed, groups, outer_train,
                    family, standalone_params, twc=False,
                )
                twc_name = f"TWC_{method}"
                twc_test[twc_name], twc_physical_test[method], twc_fit_params = expert_predict(
                    points, configs, gains, los, tx, observed, outer_train, test, family,
                    twc=True,
                    class_ridge=twc_param["class_ridge"],
                    local_k=twc_param["local_k"],
                    local_power=twc_param["local_power"],
                    class_penalty_m=twc_param["class_penalty_m"],
                    signal_k=signal_k,
                    signal_power=signal_power,
                )
                twc_oof[twc_name], twc_physical_oof[method] = crossfit_expert(
                    points, configs, gains, los, tx, observed, groups, outer_train,
                    family, twc_param, twc=True,
                )
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": twc_name,
                    **twc_fit_params,
                })

            test_pred.update(standalone_test)
            test_pred.update(twc_test)
            oof_pred.update(standalone_oof)
            oof_pred.update(twc_oof)

            standalone_matrix_test = np.column_stack([standalone_test[m] for m in METHOD_FAMILY])
            standalone_matrix_oof = np.column_stack([standalone_oof[m] for m in METHOD_FAMILY])
            twc_keys = [f"TWC_{m}" for m in METHOD_FAMILY]
            twc_matrix_test = np.column_stack([twc_test[m] for m in twc_keys])
            twc_matrix_oof = np.column_stack([twc_oof[m] for m in twc_keys])

            test_pred["MEAN3"] = np.mean(standalone_matrix_test, axis=1)
            test_pred["MEDIAN3"] = np.median(standalone_matrix_test, axis=1)
            oof_pred["MEAN3"] = np.full(len(points), np.nan, float)
            oof_pred["MEDIAN3"] = np.full(len(points), np.nan, float)
            oof_pred["MEAN3"][outer_train] = np.mean(standalone_matrix_oof[outer_train], axis=1)
            oof_pred["MEDIAN3"][outer_train] = np.median(standalone_matrix_oof[outer_train], axis=1)

            test_pred["TWC_MEAN3"] = np.mean(twc_matrix_test, axis=1)
            test_pred["TWC_MEDIAN3"] = np.median(twc_matrix_test, axis=1)
            oof_pred["TWC_MEAN3"] = np.full(len(points), np.nan, float)
            oof_pred["TWC_MEDIAN3"] = np.full(len(points), np.nan, float)
            oof_pred["TWC_MEAN3"][outer_train] = np.mean(twc_matrix_oof[outer_train], axis=1)
            oof_pred["TWC_MEDIAN3"][outer_train] = np.median(twc_matrix_oof[outer_train], axis=1)

            weight, stack_bias, stack_inner_rmse = fit_stack_weights(
                standalone_matrix_oof[outer_train], observed[outer_train]
            )
            test_pred["STACK3"] = standalone_matrix_test @ weight + stack_bias
            oof_pred["STACK3"] = np.full(len(points), np.nan, float)
            oof_pred["STACK3"][outer_train] = standalone_matrix_oof[outer_train] @ weight + stack_bias

            twc_weight, twc_stack_bias, twc_stack_inner_rmse = fit_stack_weights(
                twc_matrix_oof[outer_train], observed[outer_train]
            )
            test_pred["TWC_STACK3"] = twc_matrix_test @ twc_weight + twc_stack_bias
            oof_pred["TWC_STACK3"] = np.full(len(points), np.nan, float)
            oof_pred["TWC_STACK3"][outer_train] = twc_matrix_oof[outer_train] @ twc_weight + twc_stack_bias

            all_keys = ["SIGNAL_IDW", *METHOD_FAMILY.keys(), *twc_keys]
            all_test_matrix = np.column_stack([test_pred[key] for key in all_keys])
            all_oof_matrix = np.column_stack([oof_pred[key] for key in all_keys])
            all_weight, all_bias, all_inner_rmse = fit_stack_weights(
                all_oof_matrix[outer_train], observed[outer_train]
            )
            test_pred["ALL_STACK7"] = all_test_matrix @ all_weight + all_bias
            oof_pred["ALL_STACK7"] = np.full(len(points), np.nan, float)
            oof_pred["ALL_STACK7"][outer_train] = all_oof_matrix[outer_train] @ all_weight + all_bias

            for base_name in ("TWC_STACK3", "ALL_STACK7"):
                residual = np.full(len(points), np.nan, float)
                residual[outer_train] = observed[outer_train] - oof_pred[base_name][outer_train]
                local_k, local_power, local_inner_rmse = choose_stack_local(
                    xy, residual, groups, outer_train
                )
                local_name = base_name.replace("7", "_LOCAL").replace("3", "3_LOCAL")
                test_pred[local_name] = test_pred[base_name] + apply_stack_local(
                    xy, residual, outer_train, test, local_k, local_power
                )
                # OOF local correction is group-held-out again within the outer train.
                local_oof = np.full(len(points), np.nan, float)
                for inner in range(3):
                    valid = outer_train & np.array([fold_inner.get(str(g), -1) == inner for g in groups])
                    fit = outer_train & ~valid & np.isfinite(residual)
                    distance_local, index_local = query_neighbors(xy[fit], xy[valid], max_k=local_k)
                    local_oof[valid] = oof_pred[base_name][valid] + idw(
                        residual[fit], distance_local, index_local, local_k, local_power
                    )
                oof_pred[local_name] = local_oof
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": local_name,
                    "local_k": local_k, "local_power": local_power,
                    "inner_residual_rmse_db": local_inner_rmse,
                })

            candidate_for_auto = [
                "U2", "WEDT_P", "ONETWIN", *twc_keys,
                "STACK3", "TWC_STACK3", "ALL_STACK7", "TWC_STACK3_LOCAL", "ALL_STACK_LOCAL",
            ]
            selection = []
            for key in candidate_for_auto:
                row = metrics(observed[outer_train], oof_pred[key][outer_train])
                selection.append((row["rmse_db"], row["mae_db"], key))
            _, _, selected = min(selection)
            test_pred["HYBRID_AUTOSELECT"] = test_pred[selected].copy()
            oof_pred["HYBRID_AUTOSELECT"] = oof_pred[selected].copy()

            parameter_rows.extend([
                {
                    "band": band, "outer_fold": fold, "method": "COMMON",
                    "signal_k": signal_k, "signal_power": signal_power,
                    "signal_inner_rmse_db": signal_inner_rmse,
                    **twc_param,
                },
                {
                    "band": band, "outer_fold": fold, "method": "STACK3",
                    "weights": json.dumps(dict(zip(METHOD_FAMILY, weight)), sort_keys=True),
                    "bias_db": stack_bias, "inner_rmse_db": stack_inner_rmse,
                },
                {
                    "band": band, "outer_fold": fold, "method": "TWC_STACK3",
                    "weights": json.dumps(dict(zip(twc_keys, twc_weight)), sort_keys=True),
                    "bias_db": twc_stack_bias, "inner_rmse_db": twc_stack_inner_rmse,
                },
                {
                    "band": band, "outer_fold": fold, "method": "ALL_STACK7",
                    "weights": json.dumps(dict(zip(all_keys, all_weight)), sort_keys=True),
                    "bias_db": all_bias, "inner_rmse_db": all_inner_rmse,
                },
                {
                    "band": band, "outer_fold": fold, "method": "HYBRID_AUTOSELECT",
                    "selected_method": selected,
                },
            ])

            disagreement_oof = np.nanstd(twc_matrix_oof, axis=1)
            disagreement_test = np.std(twc_matrix_test, axis=1)
            path_matrix_oof = np.column_stack([np.isfinite(twc_physical_oof[m]) for m in METHOD_FAMILY])
            path_matrix_test = np.column_stack([np.isfinite(twc_physical_test[m]) for m in METHOD_FAMILY])
            path_missing_oof = 3 - path_matrix_oof.sum(axis=1)
            path_missing_test = 3 - path_matrix_test.sum(axis=1)
            source_score, source_threshold, source_param = source_scores(
                observed, outer_train, test,
                twc_physical_oof["WEDT_P"], twc_physical_test["WEDT_P"],
            )

            test_indices = np.flatnonzero(test)
            fold_output = points.loc[test, [
                "point_id", "band", "date", "trajectory_group", "segment_group", "x", "y", "z", "observed_dbm"
            ]].copy().reset_index(drop=True)
            fold_output["outer_fold"] = fold
            fold_output["source_score"] = source_score
            for q, threshold in source_threshold.items():
                fold_output[f"retain_source_q{q}"] = source_score <= threshold
                fold_output[f"source_threshold_q{q}"] = threshold

            for method, prediction in test_pred.items():
                fold_output[f"pred_{method}"] = prediction
                confidence, threshold, confidence_param = confidence_scores(
                    xy, outer_train, test, observed,
                    oof_pred[method], prediction,
                    disagreement_oof, disagreement_test,
                    path_missing_oof, path_missing_test,
                )
                fold_output[f"confidence_{method}"] = confidence
                for q, value in threshold.items():
                    fold_output[f"retain_target_{method}_q{q}"] = confidence <= value
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": method,
                    "confidence_model": json.dumps(confidence_param, sort_keys=True),
                    "source_model": json.dumps(source_param, sort_keys=True),
                })
            fold_output["hybrid_selected_method"] = selected
            all_output.append(fold_output)
            method_names = sorted(test_pred)

    prediction_frame = pd.concat(all_output, ignore_index=True)
    if len(prediction_frame) != 38982 or prediction_frame["point_id"].nunique() != 38982:
        raise RuntimeError("H8 point contract failed")
    prediction_frame.to_csv(args.output_dir / "point_predictions_wide.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)

    summary_rows = []
    for method in method_names:
        pred = pd.to_numeric(prediction_frame[f"pred_{method}"], errors="coerce").to_numpy(float)
        obs = pd.to_numeric(prediction_frame["observed_dbm"], errors="coerce").to_numpy(float)
        summary_rows.append({
            "track": "full_coverage", "gate": "FULL", "method": method,
            "retained_n": len(prediction_frame), "retention": 1.0,
            **metrics(obs, pred, len(prediction_frame)),
        })
        for q in QUANTILES:
            target_mask = prediction_frame[f"retain_target_{method}_q{q}"].astype(bool).to_numpy()
            summary_rows.append({
                "track": "target_blind_selective_prediction", "gate": f"OOF_CONF_Q{q}", "method": method,
                "retained_n": int(target_mask.sum()), "retention": float(target_mask.mean()),
                **metrics(obs[target_mask], pred[target_mask], int(target_mask.sum())),
            })
            source_mask = prediction_frame[f"retain_source_q{q}"].astype(bool).to_numpy()
            summary_rows.append({
                "track": "measurement_assisted_source_curation", "gate": f"SOURCE_Q{q}", "method": method,
                "retained_n": int(source_mask.sum()), "retention": float(source_mask.mean()),
                **metrics(obs[source_mask], pred[source_mask], int(source_mask.sum())),
            })
    summary = pd.DataFrame(summary_rows).sort_values(["track", "gate", "rmse_db", "mae_db"])
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    champions = {}
    for track in summary["track"].unique():
        rows = summary[summary["track"].eq(track)].copy()
        if track == "full_coverage":
            chosen = rows.sort_values(["rmse_db", "mae_db"]).iloc[0]
        else:
            eligible = rows[rows["retention"].ge(0.10)]
            chosen = eligible.sort_values(["rmse_db", "mae_db", "retention"], ascending=[True, True, False]).iloc[0]
        champions[track] = chosen.to_dict()
    write_json(args.output_dir / "result.json", {
        "status": "COMPLETED",
        "experiment": "H8_fixed_source_combined_screen",
        "statistical_points": int(len(prediction_frame)),
        "champions": champions,
        "claim_boundaries": [
            "fixed-source screen uses n41/E and n79/W propagation-equivalent candidates",
            "target-blind gates do not read test RSRP",
            "source curation reads observed RSRP and is not unknown-position prediction",
            "WEDT-p is RSRP-only, not full CSI/PDP WEDT",
        ],
    })
    print(json.dumps(json_safe({"status": "COMPLETED", "champions": champions}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
