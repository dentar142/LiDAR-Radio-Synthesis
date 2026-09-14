#!/usr/bin/env python3
"""H8.2 date-conditioned and local geospatial models on raw NR7 points."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .run_h8_combined_screen import (
    METHOD_FAMILY,
    choose_signal_params,
    crossfit_signal,
    idw,
    inner_group_map,
    json_safe,
    load_band,
    metrics,
    query_neighbors,
    signal_idw_predict,
    write_json,
)
from .run_h8_dual_source_router import exact_rank_mask, tune_and_predict_field


RETENTIONS = (10, 25, 50, 75, 90)
DATE_SHRINKAGE = (20.0, 100.0, 500.0)
LOCAL_PARAMS = tuple(itertools.product((16, 32, 64), (1.0, 2.0), (1.0, 10.0)))
GAUSSIAN_BANDWIDTH_M = (10.0, 25.0, 50.0, 100.0)
ROBUST_K = (8, 16, 32, 64)
PHYSICAL_LOCAL_PARAMS = tuple(itertools.product((32, 64), (1.0,), (1.0, 10.0)))


def date_offsets(values, dates, fit_mask, shrinkage):
    values = np.asarray(values, float)
    dates = np.asarray(dates, str)
    fit_mask = np.asarray(fit_mask, bool) & np.isfinite(values)
    global_center = float(np.median(values[fit_mask]))
    offsets = {}
    for date in sorted(set(dates[fit_mask])):
        mask = fit_mask & (dates == date)
        count = int(mask.sum())
        raw = float(np.median(values[mask]) - global_center)
        offsets[date] = float(count / (count + float(shrinkage)) * raw)
    return offsets


def offset_vector(dates, offsets):
    return np.asarray([float(offsets.get(str(date), 0.0)) for date in dates], float)


def idw_predict_indices(xy, values, fit, query, k, power):
    distance, index = query_neighbors(xy[fit], xy[query], max_k=k)
    return idw(values[fit], distance, index, k, power)


def date_only_idw(xy, values, dates, fit, query, k, power):
    output = np.full(int(query.sum()), np.nan, float)
    query_indices = np.flatnonzero(query)
    for date in sorted(set(dates[query])):
        local_query = query_indices[dates[query_indices] == date]
        local_fit = fit & (dates == date) & np.isfinite(values)
        if local_fit.sum() < 2:
            local_fit = fit & np.isfinite(values)
        query_mask = np.zeros(len(values), dtype=bool)
        query_mask[local_query] = True
        output[dates[query_indices] == date] = idw_predict_indices(
            xy, values, local_fit, query_mask, k, power
        )
    return output


def date_center_idw(xy, values, dates, fit, query, k, power, shrinkage):
    offsets = date_offsets(values, dates, fit, shrinkage)
    normalized = np.asarray(values, float) - offset_vector(dates, offsets)
    return idw_predict_indices(xy, normalized, fit & np.isfinite(normalized), query, k, power) + offset_vector(
        dates[query], offsets
    )


def neighbor_cache(xy, fit, query, max_k):
    distance, index = query_neighbors(xy[fit], xy[query], max_k=max_k)
    return distance, index, np.flatnonzero(fit)


def weighted_median_predict(values, distance, index, k, power):
    width = min(int(k), index.shape[1])
    d = distance[:, :width]
    v = np.asarray(values, float)[index[:, :width]]
    weight = 1.0 / np.maximum(d, 1e-3) ** float(power)
    order = np.argsort(v, axis=1)
    sorted_v = np.take_along_axis(v, order, axis=1)
    sorted_w = np.take_along_axis(weight, order, axis=1)
    cumulative = np.cumsum(sorted_w, axis=1)
    cutoff = cumulative[:, -1:] * 0.5
    position = np.argmax(cumulative >= cutoff, axis=1)
    return sorted_v[np.arange(len(sorted_v)), position]


def gaussian_predict(values, distance, index, bandwidth):
    v = np.asarray(values, float)[index]
    weight = np.exp(-0.5 * (distance / float(bandwidth)) ** 2)
    total = weight.sum(axis=1)
    output = np.sum(weight * v, axis=1) / np.maximum(total, 1e-12)
    output[total <= 1e-12] = v[total <= 1e-12, 0]
    return output


def local_ridge_predict(
    train_xy, train_features, train_values, query_xy, query_features,
    *, k, power, ridge, batch_size=2048,
):
    train_xy = np.asarray(train_xy, float)
    train_features = np.asarray(train_features, float)
    train_values = np.asarray(train_values, float)
    query_xy = np.asarray(query_xy, float)
    query_features = np.asarray(query_features, float)
    scale = np.maximum(np.std(train_features, axis=0), 1e-3)
    train_scaled = train_features / scale
    query_scaled = query_features / scale
    tree = cKDTree(train_xy)
    output = np.empty(len(query_xy), float)
    width = min(int(k), len(train_xy))
    for start in range(0, len(query_xy), batch_size):
        stop = min(len(query_xy), start + batch_size)
        distance, index = tree.query(query_xy[start:stop], k=width, workers=-1)
        if width == 1:
            distance = distance[:, None]
            index = index[:, None]
        difference = train_scaled[index] - query_scaled[start:stop, None, :]
        design = np.concatenate([np.ones((*difference.shape[:2], 1)), difference], axis=2)
        weight = 1.0 / np.maximum(distance, 1e-3) ** float(power)
        matrix = np.einsum("bki,bk,bkj->bij", design, weight, design)
        right = np.einsum("bki,bk,bk->bi", design, weight, train_values[index])
        regularizer = np.zeros_like(matrix)
        regularizer[:, 0, 0] = 1e-8
        diagonal = np.arange(1, matrix.shape[1])
        regularizer[:, diagonal, diagonal] = float(ridge)
        matrix += regularizer
        try:
            coefficient = np.linalg.solve(matrix, right[..., None])[..., 0]
        except np.linalg.LinAlgError:
            coefficient = np.linalg.lstsq(matrix, right[..., None], rcond=None)[0][..., 0]
        output[start:stop] = coefficient[:, 0]
    return output


def physical_features(points, configs, gains, los, tx):
    family_columns = []
    for family in ("BASE", "S4W", "S5"):
        indices = [i for i, config in enumerate(configs) if config["family"] == family or (family == "BASE" and config["id"] == "AUTO_BASE")]
        family_columns.append(np.nanmedian(gains[indices], axis=0))
    matrix = np.column_stack(family_columns)
    missing = (~np.isfinite(matrix)).sum(axis=1).astype(float)
    for column in range(matrix.shape[1]):
        finite = np.isfinite(matrix[:, column])
        fill = float(np.median(matrix[finite, column])) if finite.any() else -150.0
        matrix[~finite, column] = fill
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - np.asarray(tx, float)[None, :], axis=1)
    return np.column_stack([
        points[["x", "y"]].to_numpy(float), matrix,
        np.asarray(los, float), np.log10(np.maximum(distance, 1.0)), missing,
    ])


def predict_candidate(name, params, xy, features, values, dates, fit, query):
    k = int(params.get("k", 16))
    power = float(params.get("power", 2.0))
    if name == "POOLED_IDW":
        return idw_predict_indices(xy, values, fit, query, k, power)
    if name == "DATE_ONLY_IDW":
        return date_only_idw(xy, values, dates, fit, query, k, power)
    if name == "DATE_CENTER_IDW":
        return date_center_idw(xy, values, dates, fit, query, k, power, params["shrinkage"])

    working_values = np.asarray(values, float)
    query_offsets = np.zeros(int(query.sum()), float)
    if name.startswith("DATE_CENTER_"):
        offsets = date_offsets(values, dates, fit, params["shrinkage"])
        working_values = working_values - offset_vector(dates, offsets)
        query_offsets = offset_vector(dates[query], offsets)

    distance, index, fit_indices = neighbor_cache(xy, fit, query, max_k=128)
    train_values = working_values[fit_indices]
    if name in ("GAUSSIAN_KERNEL", "DATE_CENTER_GAUSSIAN"):
        return gaussian_predict(train_values, distance, index, params["bandwidth_m"]) + query_offsets
    if name == "ROBUST_LOCAL_MEDIAN":
        return weighted_median_predict(train_values, distance, index, k, power)
    if name.endswith("LOCAL_LINEAR"):
        prediction = local_ridge_predict(
            xy[fit], xy[fit], working_values[fit], xy[query], xy[query],
            k=k, power=power, ridge=params["ridge"],
        )
        return prediction + query_offsets
    if name == "PHYS_ASSISTED_LOCAL":
        return local_ridge_predict(
            xy[fit], features[fit], values[fit], xy[query], features[query],
            k=k, power=power, ridge=params["ridge"],
        )
    raise KeyError(name)


def crossfit_candidate(name, params, xy, features, values, dates, groups, outer_train):
    output = np.full(len(values), np.nan, float)
    fold_map = inner_group_map(groups, outer_train)
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
        output[valid] = predict_candidate(name, params, xy, features, values, dates, fit, valid)
    return output


def choose_params(name, grid, xy, features, values, dates, groups, outer_train):
    fold_map = inner_group_map(groups, outer_train)
    score = []
    for params in grid:
        sse = 0.0
        absolute = 0.0
        count = 0
        for fold in range(3):
            fit = outer_train & np.array([fold_map.get(str(group), -1) != fold for group in groups])
            valid = outer_train & np.array([fold_map.get(str(group), -1) == fold for group in groups])
            prediction = predict_candidate(name, params, xy, features, values, dates, fit, valid)
            error = prediction - values[valid]
            sse += float(np.sum(error**2))
            absolute += float(np.sum(np.abs(error)))
            count += int(valid.sum())
        score.append((math.sqrt(sse / count), absolute / count, json.dumps(params, sort_keys=True), params))
    return min(score)[3], min(score)[0], min(score)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--rt-root", required=True, type=Path)
    parser.add_argument("--h7-score-dir", required=True, type=Path)
    parser.add_argument("--fixed-h8-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_frame = pd.read_csv(args.h7_score_dir / "trajectory_fold_assignment.csv")
    outer_assignment = dict(zip(fold_frame["trajectory_group"].astype(str), fold_frame["outer_fold"].astype(int)))
    fixed_h8 = pd.read_csv(args.fixed_h8_predictions)
    fixed_h8["point_id"] = fixed_h8["point_id"].astype(str)
    fixed_h8 = fixed_h8.set_index("point_id")

    outputs = []
    parameter_rows = []
    risk_methods = set()
    for band in ("n41", "n79"):
        points, configs, gains, los, tx = load_band(args.data_root, args.rt_root, band)
        point_ids = points["point_id"].astype(str)
        band_h8 = fixed_h8.reindex(point_ids)
        values = pd.to_numeric(points["observed_dbm"], errors="coerce").to_numpy(float)
        xy = points[["x", "y"]].to_numpy(float)
        dates = points["date"].astype(str).to_numpy()
        groups = points["trajectory_group"].astype(str).to_numpy()
        outer_fold = np.asarray([outer_assignment[str(group)] for group in groups], int)
        features = physical_features(points, configs, gains, los, tx)

        for fold in range(5):
            outer_train = outer_fold != fold
            test = outer_fold == fold
            signal_k, signal_power, signal_inner = choose_signal_params(xy, values, groups, outer_train)
            pooled_params = {"k": signal_k, "power": signal_power}
            date_grid = [{**pooled_params, "shrinkage": shrinkage} for shrinkage in DATE_SHRINKAGE]
            date_params, date_inner_rmse, date_inner_mae = choose_params(
                "DATE_CENTER_IDW", date_grid, xy, features, values, dates, groups, outer_train
            )
            local_grid = [{"k": k, "power": power, "ridge": ridge} for k, power, ridge in LOCAL_PARAMS]
            local_params, local_inner_rmse, local_inner_mae = choose_params(
                "LOCAL_LINEAR", local_grid, xy, features, values, dates, groups, outer_train
            )
            gaussian_grid = [{"bandwidth_m": bandwidth} for bandwidth in GAUSSIAN_BANDWIDTH_M]
            gaussian_params, gaussian_inner_rmse, gaussian_inner_mae = choose_params(
                "GAUSSIAN_KERNEL", gaussian_grid, xy, features, values, dates, groups, outer_train
            )
            robust_grid = [{"k": k, "power": 1.0} for k in ROBUST_K]
            robust_params, robust_inner_rmse, robust_inner_mae = choose_params(
                "ROBUST_LOCAL_MEDIAN", robust_grid, xy, features, values, dates, groups, outer_train
            )
            physical_grid = [
                {"k": k, "power": power, "ridge": ridge}
                for k, power, ridge in PHYSICAL_LOCAL_PARAMS
            ]
            physical_params, physical_inner_rmse, physical_inner_mae = choose_params(
                "PHYS_ASSISTED_LOCAL", physical_grid, xy, features, values, dates, groups, outer_train
            )

            specifications = {
                "POOLED_IDW": pooled_params,
                "DATE_ONLY_IDW": pooled_params,
                "DATE_CENTER_IDW": date_params,
                "LOCAL_LINEAR": local_params,
                "DATE_CENTER_LOCAL_LINEAR": {**local_params, "shrinkage": date_params["shrinkage"]},
                "GAUSSIAN_KERNEL": gaussian_params,
                "DATE_CENTER_GAUSSIAN": {**gaussian_params, "shrinkage": date_params["shrinkage"]},
                "ROBUST_LOCAL_MEDIAN": robust_params,
                "PHYS_ASSISTED_LOCAL": physical_params,
            }
            pred_test = {}
            pred_oof = {}
            for name, params in specifications.items():
                if name == "POOLED_IDW":
                    pred_test[name] = band_h8["pred_SIGNAL_IDW"].to_numpy(float)[test]
                    pred_oof[name] = crossfit_signal(
                        points, values, groups, outer_train, signal_k, signal_power
                    )
                else:
                    pred_test[name] = predict_candidate(
                        name, params, xy, features, values, dates, outer_train, test
                    )
                    pred_oof[name] = crossfit_candidate(
                        name, params, xy, features, values, dates, groups, outer_train
                    )

            selection = []
            for name in specifications:
                row = metrics(values[outer_train], pred_oof[name][outer_train])
                selection.append((row["rmse_db"], row["mae_db"], name))
            _, _, selected = min(selection)
            pred_test["AUTO_GEOSPATIAL"] = pred_test[selected].copy()
            pred_oof["AUTO_GEOSPATIAL"] = pred_oof[selected].copy()

            non_idw = min(item for item in selection if item[2] != "POOLED_IDW")[2]
            alpha_scores = []
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                blended = alpha * pred_oof["POOLED_IDW"][outer_train] + (1.0 - alpha) * pred_oof[non_idw][outer_train]
                row = metrics(values[outer_train], blended)
                alpha_scores.append((row["rmse_db"], row["mae_db"], alpha))
            _, _, alpha = min(alpha_scores)
            pred_test["SHRUNK_GEO_BLEND"] = alpha * pred_test["POOLED_IDW"] + (1.0 - alpha) * pred_test[non_idw]
            pred_oof["SHRUNK_GEO_BLEND"] = alpha * pred_oof["POOLED_IDW"] + (1.0 - alpha) * pred_oof[non_idw]

            risk_test = {}
            for name in pred_test:
                absolute_error = np.full(len(points), np.nan, float)
                absolute_error[outer_train] = np.abs(pred_oof[name][outer_train] - values[outer_train])
                _, risk, risk_param = tune_and_predict_field(
                    xy, absolute_error, groups, outer_train, test
                )
                risk_test[name] = np.maximum(risk, 0.1)
                risk_methods.add(name)
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": f"RISK_{name}",
                    "parameters": json.dumps(json_safe(risk_param), sort_keys=True),
                })

            fold_output = points.loc[test, [
                "point_id", "band", "date", "trajectory_group", "segment_group", "x", "y", "z", "observed_dbm"
            ]].copy().reset_index(drop=True)
            fold_output["outer_fold"] = fold
            for name, prediction in pred_test.items():
                fold_output[f"pred_{name}"] = prediction
                fold_output[f"risk_{name}"] = risk_test[name]
                for percent in RETENTIONS:
                    fold_output[f"retain_target_{name}_q{percent}"] = exact_rank_mask(risk_test[name], percent)
            outputs.append(fold_output)
            parameter_rows.append({
                "band": band, "outer_fold": fold, "method": "COMMON_GEO",
                "signal_inner_rmse": signal_inner,
                "date_inner_rmse": date_inner_rmse, "date_inner_mae": date_inner_mae,
                "local_inner_rmse": local_inner_rmse, "local_inner_mae": local_inner_mae,
                "gaussian_inner_rmse": gaussian_inner_rmse, "gaussian_inner_mae": gaussian_inner_mae,
                "robust_inner_rmse": robust_inner_rmse, "robust_inner_mae": robust_inner_mae,
                "physical_inner_rmse": physical_inner_rmse, "physical_inner_mae": physical_inner_mae,
                "specifications": json.dumps(json_safe(specifications), sort_keys=True),
                "auto_selected": selected, "blend_non_idw": non_idw, "blend_alpha_idw": alpha,
            })

    prediction = pd.concat(outputs, ignore_index=True)
    if len(prediction) != 38982 or prediction["point_id"].astype(str).nunique() != 38982:
        raise RuntimeError("H8.2 point contract failed")
    pred_columns = sorted(column for column in prediction if column.startswith("pred_"))
    for column in pred_columns:
        if not np.isfinite(pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)).all():
            raise RuntimeError(f"non-finite H8.2 prediction: {column}")
    prediction.to_csv(args.output_dir / "point_predictions_wide.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)

    observed = pd.to_numeric(prediction["observed_dbm"], errors="coerce").to_numpy(float)
    campaign_methods = {
        "DATE_ONLY_IDW", "DATE_CENTER_IDW", "DATE_CENTER_LOCAL_LINEAR", "DATE_CENTER_GAUSSIAN"
    }
    static_methods = {
        "POOLED_IDW", "LOCAL_LINEAR", "GAUSSIAN_KERNEL", "ROBUST_LOCAL_MEDIAN", "PHYS_ASSISTED_LOCAL"
    }
    rows = []
    for column in pred_columns:
        name = column[5:]
        predicted = pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)
        scope = (
            "campaign_conditioned" if name in campaign_methods
            else "static_map" if name in static_methods
            else "nested_mixed"
        )
        rows.append({
            "track": "full_coverage", "scope": scope, "gate": "FULL", "method": name,
            "retained_n": len(prediction), "retention": 1.0,
            **metrics(observed, predicted, len(prediction)),
        })
        for percent in RETENTIONS:
            mask = prediction[f"retain_target_{name}_q{percent}"].astype(bool).to_numpy()
            rows.append({
                "track": "target_blind_selective_prediction", "scope": scope,
                "gate": f"RISK_Q{percent}", "method": name,
                "retained_n": int(mask.sum()), "retention": float(mask.mean()),
                **metrics(observed[mask], predicted[mask], int(mask.sum())),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    full = summary[summary["track"].eq("full_coverage")].sort_values(["rmse_db", "mae_db"])
    target_champions = {}
    for percent in RETENTIONS:
        target = summary[
            summary["track"].eq("target_blind_selective_prediction")
            & summary["gate"].eq(f"RISK_Q{percent}")
        ].sort_values(["rmse_db", "mae_db"]).iloc[0]
        target_champions[str(percent)] = json_safe(target.to_dict())
    result = {
        "status": "COMPLETED", "experiment": "H8_2_geospatial_models",
        "statistical_points": int(len(prediction)),
        "champions": {
            "full_overall": json_safe(full.iloc[0].to_dict()),
            "full_static_map": json_safe(full[full["scope"].eq("static_map")].iloc[0].to_dict()),
            "full_campaign_conditioned": json_safe(full[full["scope"].eq("campaign_conditioned")].iloc[0].to_dict()),
            "target_blind_by_fixed_retention": target_champions,
        },
        "claim_boundaries": [
            "date-conditioned models require campaign/date metadata at prediction time",
            "static-map and campaign-conditioned scores are reported separately",
            "target-blind ranking does not read test RSRP",
            "interpolated values are never scoring truth",
        ],
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
