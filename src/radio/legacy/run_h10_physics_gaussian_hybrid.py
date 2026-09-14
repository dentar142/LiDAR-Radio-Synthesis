#!/usr/bin/env python3
"""H10 leakage-safe TWC-WEDT-p and Gaussian residual hybrids.

The outer unit is an anonymous trajectory. All material, TWC, kernel, blend,
distance-gate, and risk-gate choices are made from outer-training data only.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .run_h8_combined_screen import (
    fit_physical,
    inner_group_map,
    json_safe,
    load_band,
    metrics,
    query_neighbors,
    write_json,
)
from .run_h8_dual_source_router import tune_and_predict_field
from .run_h8_geospatial_models import (
    GAUSSIAN_BANDWIDTH_M,
    choose_params,
    crossfit_candidate,
    gaussian_predict,
    predict_candidate,
)


BANDS = ("n41", "n79")
OUTER_FOLDS = tuple(range(5))
CLASS_RIDGES = (1.0, 5.0, 20.0)
RESIDUAL_SHRINKS = (0.5, 0.75, 1.0)
BLEND_GAUSSIAN_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
DISTANCE_SCALES_M = (10.0, 25.0, 50.0, 100.0, 200.0)
RISK_POWERS = (1.0, 2.0)
METHODS = (
    "H8_GAUSSIAN",
    "V7R4_RT_BIAS",
    "V7R4_TWC",
    "V7R4_TWC_GAUSSIAN_RESIDUAL",
    "WEDT_P_TWC",
    "WEDT_P_TWC_GAUSSIAN_RESIDUAL",
    "GLOBAL_BLEND",
    "DISTANCE_GATE",
    "RISK_GATE",
    "H10_AUTO",
)


def split_masks(groups: np.ndarray, outer_train: np.ndarray):
    fold_map = inner_group_map(groups, outer_train)
    for fold in range(3):
        fit = outer_train & np.asarray([fold_map.get(str(group), -1) != fold for group in groups])
        valid = outer_train & np.asarray([fold_map.get(str(group), -1) == fold for group in groups])
        if fit.sum() >= 100 and valid.sum() >= 20:
            yield fold, fit, valid


def gaussian_for_split(
    xy: np.ndarray,
    values: np.ndarray,
    fit: np.ndarray,
    query: np.ndarray,
    bandwidth_m: float,
) -> np.ndarray:
    distance, index = query_neighbors(xy[fit], xy[query], max_k=128)
    return gaussian_predict(values[fit], distance, index, bandwidth_m)


def nearest_distance_for_split(xy: np.ndarray, fit: np.ndarray, query: np.ndarray) -> np.ndarray:
    distance, _ = cKDTree(xy[fit]).query(xy[query], k=1, workers=-1)
    return np.asarray(distance, float)


def distance_gate_weight(nearest_distance_m: np.ndarray, scale_m: float) -> np.ndarray:
    """Weight on Gaussian: local observations dominate close to training support."""
    distance = np.maximum(np.asarray(nearest_distance_m, float), 0.0)
    return np.exp(-0.5 * (distance / float(scale_m)) ** 2)


def inverse_risk_weight(gaussian_risk: np.ndarray, physical_risk: np.ndarray, power: float) -> np.ndarray:
    """Weight on Gaussian under inverse predicted-risk blending."""
    risk_g = np.maximum(np.asarray(gaussian_risk, float), 0.1) ** float(power)
    risk_p = np.maximum(np.asarray(physical_risk, float), 0.1) ** float(power)
    return risk_p / np.maximum(risk_g + risk_p, 1e-12)


def convex_mix(gaussian: np.ndarray, physical: np.ndarray, gaussian_weight) -> np.ndarray:
    weight = np.clip(np.asarray(gaussian_weight, float), 0.0, 1.0)
    return weight * np.asarray(gaussian, float) + (1.0 - weight) * np.asarray(physical, float)


def fill_no_path(physical: np.ndarray, gaussian: np.ndarray) -> tuple[np.ndarray, int]:
    output = np.asarray(physical, float).copy()
    missing = ~np.isfinite(output)
    output[missing] = np.asarray(gaussian, float)[missing]
    return output, int(missing.sum())


def fit_physical_all(
    points: pd.DataFrame,
    configs: list[dict],
    gains: np.ndarray,
    los: np.ndarray,
    tx: np.ndarray,
    observed: np.ndarray,
    fit: np.ndarray,
    ridge: float,
    *,
    family: str,
    twc: bool,
) -> tuple[np.ndarray, dict]:
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - np.asarray(tx, float)[None, :], axis=1)
    return fit_physical(
        configs,
        gains,
        observed,
        distance,
        ~np.asarray(los, bool),
        fit,
        family,
        twc=bool(twc),
        class_ridge=float(ridge),
    )


def residual_hybrid_for_split(
    xy: np.ndarray,
    observed: np.ndarray,
    physical_all: np.ndarray,
    fit: np.ndarray,
    query: np.ndarray,
    gaussian_fallback: np.ndarray,
    bandwidth_m: float,
    shrink: float,
) -> tuple[np.ndarray, int]:
    query_physical = np.asarray(physical_all[query], float)
    output = np.asarray(gaussian_fallback, float).copy()
    has_path = np.isfinite(query_physical)
    finite_fit = fit & np.isfinite(physical_all)
    if has_path.any() and finite_fit.sum() >= 2:
        distance, index = query_neighbors(xy[finite_fit], xy[query][has_path], max_k=128)
        residual = observed[finite_fit] - physical_all[finite_fit]
        correction = gaussian_predict(residual, distance, index, bandwidth_m)
        output[has_path] = query_physical[has_path] + float(shrink) * correction
    return output, int((~has_path).sum())


def score_prediction(observed: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    row = metrics(observed[mask], prediction[mask])
    return float(row["rmse_db"]), float(row["mae_db"])


def select_best(scored: list[tuple[float, float, str, object]]):
    if not scored:
        raise RuntimeError("empty model selection set")
    return min(scored, key=lambda row: (row[0], row[1], row[2]))


def crossfit_raw_physical(
    points,
    configs,
    gains,
    los,
    tx,
    observed,
    groups,
    outer_train,
    xy,
    gaussian_bandwidth,
    ridge,
    *,
    family,
    twc,
):
    output = np.full(len(observed), np.nan, float)
    fallback_count = 0
    config_ids = []
    for _, fit, valid in split_masks(groups, outer_train):
        gaussian = gaussian_for_split(xy, observed, fit, valid, gaussian_bandwidth)
        physical, params = fit_physical_all(
            points, configs, gains, los, tx, observed, fit, ridge,
            family=family, twc=twc,
        )
        output[valid], count = fill_no_path(physical[valid], gaussian)
        fallback_count += count
        config_ids.append(params["config_id"])
    return output, fallback_count, config_ids


def crossfit_residual_hybrid(
    points,
    configs,
    gains,
    los,
    tx,
    observed,
    groups,
    outer_train,
    xy,
    ridge,
    bandwidth_m,
    shrink,
    *,
    family="S4W",
):
    output = np.full(len(observed), np.nan, float)
    fallback_count = 0
    config_ids = []
    for _, fit, valid in split_masks(groups, outer_train):
        gaussian = gaussian_for_split(xy, observed, fit, valid, bandwidth_m)
        physical, params = fit_physical_all(
            points, configs, gains, los, tx, observed, fit, ridge,
            family=family, twc=True,
        )
        output[valid], count = residual_hybrid_for_split(
            xy,
            observed,
            physical,
            fit,
            valid,
            gaussian,
            bandwidth_m,
            shrink,
        )
        fallback_count += count
        config_ids.append(params["config_id"])
    return output, fallback_count, config_ids


def crossfit_residual_grid(
    points,
    configs,
    gains,
    los,
    tx,
    observed,
    groups,
    outer_train,
    xy,
    *,
    family,
):
    """Evaluate the full residual grid while fitting physics once per ridge/split."""
    keys = [
        (float(ridge), float(bandwidth), float(shrink))
        for ridge in CLASS_RIDGES
        for bandwidth in GAUSSIAN_BANDWIDTH_M
        for shrink in RESIDUAL_SHRINKS
    ]
    outputs = {key: np.full(len(observed), np.nan, float) for key in keys}
    fallback_counts = {key: 0 for key in keys}
    config_ids = {key: [] for key in keys}
    for _, fit, valid in split_masks(groups, outer_train):
        gaussian = {
            float(bandwidth): gaussian_for_split(xy, observed, fit, valid, float(bandwidth))
            for bandwidth in GAUSSIAN_BANDWIDTH_M
        }
        for ridge in CLASS_RIDGES:
            physical, params = fit_physical_all(
                points, configs, gains, los, tx, observed, fit, float(ridge),
                family=family, twc=True,
            )
            query_physical = np.asarray(physical[valid], float)
            has_path = np.isfinite(query_physical)
            finite_fit = fit & np.isfinite(physical)
            corrections = {}
            if has_path.any() and finite_fit.sum() >= 2:
                distance, index = query_neighbors(
                    xy[finite_fit], xy[valid][has_path], max_k=128
                )
                residual = observed[finite_fit] - physical[finite_fit]
                corrections = {
                    float(bandwidth): gaussian_predict(
                        residual, distance, index, float(bandwidth)
                    )
                    for bandwidth in GAUSSIAN_BANDWIDTH_M
                }
            for bandwidth in GAUSSIAN_BANDWIDTH_M:
                bandwidth = float(bandwidth)
                for shrink in RESIDUAL_SHRINKS:
                    key = (float(ridge), bandwidth, float(shrink))
                    prediction = gaussian[bandwidth].copy()
                    if has_path.any() and bandwidth in corrections:
                        prediction[has_path] = (
                            query_physical[has_path]
                            + float(shrink) * corrections[bandwidth]
                        )
                    outputs[key][valid] = prediction
                    fallback_counts[key] += int((~has_path).sum())
                    config_ids[key].append(params["config_id"])
    return outputs, fallback_counts, config_ids


def crossfit_nearest_distance(xy, groups, outer_train):
    output = np.full(len(xy), np.nan, float)
    for _, fit, valid in split_masks(groups, outer_train):
        output[valid] = nearest_distance_for_split(xy, fit, valid)
    return output


def fit_outer_models(
    *,
    points,
    configs,
    gains,
    los,
    tx,
    observed,
    groups,
    outer_train,
    test,
    h8_reference,
):
    xy = points[["x", "y"]].to_numpy(float)
    dates = points["date"].astype(str).to_numpy()
    dummy_features = np.empty((len(points), 0), float)

    gaussian_grid = [{"bandwidth_m": float(bandwidth)} for bandwidth in GAUSSIAN_BANDWIDTH_M]
    gaussian_params, gaussian_inner_rmse, gaussian_inner_mae = choose_params(
        "GAUSSIAN_KERNEL",
        gaussian_grid,
        xy,
        dummy_features,
        observed,
        dates,
        groups,
        outer_train,
    )
    gaussian_bw = float(gaussian_params["bandwidth_m"])
    gaussian_oof = crossfit_candidate(
        "GAUSSIAN_KERNEL",
        gaussian_params,
        xy,
        dummy_features,
        observed,
        dates,
        groups,
        outer_train,
    )
    gaussian_test = predict_candidate(
        "GAUSSIAN_KERNEL",
        gaussian_params,
        xy,
        dummy_features,
        observed,
        dates,
        outer_train,
        test,
    )
    reference_delta = float(np.max(np.abs(gaussian_test - np.asarray(h8_reference, float))))

    def select_raw_branch(family, twc, ridges):
        candidates = []
        cache = {}
        for ridge in ridges:
            prediction, fallback_count, config_ids = crossfit_raw_physical(
                points,
                configs,
                gains,
                los,
                tx,
                observed,
                groups,
                outer_train,
                xy,
                gaussian_bw,
                float(ridge),
                family=family,
                twc=twc,
            )
            rmse, mae = score_prediction(observed, prediction, outer_train)
            cache[float(ridge)] = (prediction, fallback_count, config_ids)
            candidates.append((rmse, mae, f"ridge={ridge:g}", float(ridge)))
        inner_rmse, inner_mae, _, ridge = select_best(candidates)
        oof_prediction, oof_fallback, inner_configs = cache[ridge]
        physical, fit_params = fit_physical_all(
            points, configs, gains, los, tx, observed, outer_train, ridge,
            family=family, twc=twc,
        )
        test_prediction, test_fallback = fill_no_path(physical[test], gaussian_test)
        return oof_prediction, test_prediction, {
            "class_ridge": ridge,
            "inner_rmse_db": inner_rmse,
            "inner_mae_db": inner_mae,
            "outer_config_id": fit_params["config_id"],
            "inner_config_ids": inner_configs,
            "oof_fallback_n": oof_fallback,
            "test_fallback_n": test_fallback,
        }

    v7_bias_oof, v7_bias_test, v7_bias_info = select_raw_branch(
        "BASE", False, (0.0,)
    )
    v7_twc_oof, v7_twc_test, v7_twc_info = select_raw_branch(
        "BASE", True, CLASS_RIDGES
    )
    wedt_twc_oof, wedt_twc_test, wedt_twc_info = select_raw_branch(
        "S4W", True, CLASS_RIDGES
    )

    def select_residual_branch(family):
        candidates = []
        outputs, fallbacks, configs_by_key = crossfit_residual_grid(
            points,
            configs,
            gains,
            los,
            tx,
            observed,
            groups,
            outer_train,
            xy,
            family=family,
        )
        for key, prediction in outputs.items():
            rmse, mae = score_prediction(observed, prediction, outer_train)
            candidates.append((rmse, mae, json.dumps(key), key))
        inner_rmse, inner_mae, _, key = select_best(candidates)
        ridge, bandwidth, shrink = key
        oof_prediction = outputs[key]
        physical, fit_params = fit_physical_all(
            points, configs, gains, los, tx, observed, outer_train, ridge,
            family=family, twc=True,
        )
        test_prediction, test_fallback = residual_hybrid_for_split(
            xy,
            observed,
            physical,
            outer_train,
            test,
            gaussian_test,
            bandwidth,
            shrink,
        )
        return oof_prediction, test_prediction, {
            "class_ridge": ridge,
            "bandwidth_m": bandwidth,
            "shrink": shrink,
            "inner_rmse_db": inner_rmse,
            "inner_mae_db": inner_mae,
            "outer_config_id": fit_params["config_id"],
            "inner_config_ids": configs_by_key[key],
            "oof_fallback_n": fallbacks[key],
            "test_fallback_n": test_fallback,
        }

    v7_residual_oof, v7_residual_test, v7_residual_info = select_residual_branch("BASE")
    wedt_residual_oof, wedt_residual_test, wedt_residual_info = select_residual_branch("S4W")
    residual_choices = [
        (*score_prediction(observed, v7_residual_oof, outer_train), "V7R4_TWC_GAUSSIAN_RESIDUAL"),
        (*score_prediction(observed, wedt_residual_oof, outer_train), "WEDT_P_TWC_GAUSSIAN_RESIDUAL"),
    ]
    _, _, preferred_residual_name = min(
        residual_choices, key=lambda row: (row[0], row[1], row[2])
    )
    if preferred_residual_name == "V7R4_TWC_GAUSSIAN_RESIDUAL":
        residual_oof, residual_test = v7_residual_oof, v7_residual_test
    else:
        residual_oof, residual_test = wedt_residual_oof, wedt_residual_test

    blend_candidates = []
    for weight in BLEND_GAUSSIAN_WEIGHTS:
        prediction = convex_mix(gaussian_oof[outer_train], residual_oof[outer_train], weight)
        row = metrics(observed[outer_train], prediction)
        blend_candidates.append((row["rmse_db"], row["mae_db"], f"w={weight:g}", float(weight)))
    blend_rmse, blend_mae, _, blend_weight = select_best(blend_candidates)
    blend_oof = np.full(len(observed), np.nan, float)
    blend_oof[outer_train] = convex_mix(
        gaussian_oof[outer_train], residual_oof[outer_train], blend_weight
    )
    blend_test = convex_mix(gaussian_test, residual_test, blend_weight)

    nearest_oof = crossfit_nearest_distance(xy, groups, outer_train)
    nearest_test = nearest_distance_for_split(xy, outer_train, test)
    distance_candidates = []
    for scale in DISTANCE_SCALES_M:
        weight = distance_gate_weight(nearest_oof[outer_train], scale)
        prediction = convex_mix(gaussian_oof[outer_train], residual_oof[outer_train], weight)
        row = metrics(observed[outer_train], prediction)
        distance_candidates.append((row["rmse_db"], row["mae_db"], f"d={scale:g}", float(scale)))
    distance_rmse, distance_mae, _, distance_scale = select_best(distance_candidates)
    distance_oof = np.full(len(observed), np.nan, float)
    distance_oof[outer_train] = convex_mix(
        gaussian_oof[outer_train],
        residual_oof[outer_train],
        distance_gate_weight(nearest_oof[outer_train], distance_scale),
    )
    distance_test = convex_mix(
        gaussian_test,
        residual_test,
        distance_gate_weight(nearest_test, distance_scale),
    )

    gaussian_error = np.full(len(observed), np.nan, float)
    residual_error = np.full(len(observed), np.nan, float)
    gaussian_error[outer_train] = np.abs(gaussian_oof[outer_train] - observed[outer_train])
    residual_error[outer_train] = np.abs(residual_oof[outer_train] - observed[outer_train])
    gaussian_risk_oof, gaussian_risk_test, gaussian_risk_params = tune_and_predict_field(
        xy, gaussian_error, groups, outer_train, test, clip=(0.1, 30.0)
    )
    residual_risk_oof, residual_risk_test, residual_risk_params = tune_and_predict_field(
        xy, residual_error, groups, outer_train, test, clip=(0.1, 30.0)
    )
    risk_candidates = []
    for power in RISK_POWERS:
        weight = inverse_risk_weight(
            gaussian_risk_oof[outer_train], residual_risk_oof[outer_train], power
        )
        prediction = convex_mix(gaussian_oof[outer_train], residual_oof[outer_train], weight)
        row = metrics(observed[outer_train], prediction)
        risk_candidates.append((row["rmse_db"], row["mae_db"], f"p={power:g}", float(power)))
    risk_rmse, risk_mae, _, risk_power = select_best(risk_candidates)
    risk_oof = np.full(len(observed), np.nan, float)
    risk_oof[outer_train] = convex_mix(
        gaussian_oof[outer_train],
        residual_oof[outer_train],
        inverse_risk_weight(
            gaussian_risk_oof[outer_train], residual_risk_oof[outer_train], risk_power
        ),
    )
    risk_test = convex_mix(
        gaussian_test,
        residual_test,
        inverse_risk_weight(gaussian_risk_test, residual_risk_test, risk_power),
    )

    oof = {
        "H8_GAUSSIAN": gaussian_oof,
        "V7R4_RT_BIAS": v7_bias_oof,
        "V7R4_TWC": v7_twc_oof,
        "V7R4_TWC_GAUSSIAN_RESIDUAL": v7_residual_oof,
        "WEDT_P_TWC": wedt_twc_oof,
        "WEDT_P_TWC_GAUSSIAN_RESIDUAL": wedt_residual_oof,
        "GLOBAL_BLEND": blend_oof,
        "DISTANCE_GATE": distance_oof,
        "RISK_GATE": risk_oof,
    }
    test_predictions = {
        "H8_GAUSSIAN": gaussian_test,
        "V7R4_RT_BIAS": v7_bias_test,
        "V7R4_TWC": v7_twc_test,
        "V7R4_TWC_GAUSSIAN_RESIDUAL": v7_residual_test,
        "WEDT_P_TWC": wedt_twc_test,
        "WEDT_P_TWC_GAUSSIAN_RESIDUAL": wedt_residual_test,
        "GLOBAL_BLEND": blend_test,
        "DISTANCE_GATE": distance_test,
        "RISK_GATE": risk_test,
    }
    auto_candidates = []
    for name, prediction in oof.items():
        rmse, mae = score_prediction(observed, prediction, outer_train)
        auto_candidates.append((rmse, mae, name, name))
    auto_rmse, auto_mae, _, auto_method = select_best(auto_candidates)
    oof["H10_AUTO"] = oof[auto_method].copy()
    test_predictions["H10_AUTO"] = test_predictions[auto_method].copy()

    parameters = {
        "gaussian": {
            "bandwidth_m": gaussian_bw,
            "inner_rmse_db": gaussian_inner_rmse,
            "inner_mae_db": gaussian_inner_mae,
            "reference_max_abs_delta_db": reference_delta,
        },
        "v7r4_rt_bias": v7_bias_info,
        "v7r4_twc": v7_twc_info,
        "v7r4_twc_gaussian_residual": v7_residual_info,
        "wedt_p_twc": wedt_twc_info,
        "wedt_p_twc_gaussian_residual": wedt_residual_info,
        "preferred_physical_residual_branch": preferred_residual_name,
        "global_blend": {
            "gaussian_weight": blend_weight,
            "inner_rmse_db": blend_rmse,
            "inner_mae_db": blend_mae,
        },
        "distance_gate": {
            "scale_m": distance_scale,
            "inner_rmse_db": distance_rmse,
            "inner_mae_db": distance_mae,
        },
        "risk_gate": {
            "power": risk_power,
            "inner_rmse_db": risk_rmse,
            "inner_mae_db": risk_mae,
            "gaussian_risk": gaussian_risk_params,
            "physical_risk": residual_risk_params,
        },
        "auto": {
            "selected_method": auto_method,
            "inner_rmse_db": auto_rmse,
            "inner_mae_db": auto_mae,
        },
    }
    return test_predictions, parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--rt-root", required=True, type=Path)
    parser.add_argument("--fold-csv", required=True, type=Path)
    parser.add_argument("--h8-geospatial", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_frame = pd.read_csv(args.fold_csv)
    outer_assignment = dict(
        zip(fold_frame["trajectory_group"].astype(str), fold_frame["outer_fold"].astype(int))
    )
    h8 = pd.read_csv(args.h8_geospatial)
    h8["point_id"] = h8["point_id"].astype(str)
    if "pred_GAUSSIAN_KERNEL" not in h8.columns:
        raise KeyError("H8 geospatial predictions lack pred_GAUSSIAN_KERNEL")
    h8 = h8.set_index("point_id")

    output_frames = []
    parameter_rows = []
    reference_max_delta = 0.0
    for band in BANDS:
        points, configs, gains, los, tx = load_band(args.data_root, args.rt_root, band)
        point_ids = points["point_id"].astype(str)
        observed = pd.to_numeric(points["observed_dbm"], errors="coerce").to_numpy(float)
        groups = points["trajectory_group"].astype(str).to_numpy()
        outer_fold = np.asarray([outer_assignment[str(group)] for group in groups], int)
        band_h8 = h8.reindex(point_ids)
        if band_h8["pred_GAUSSIAN_KERNEL"].isna().any():
            raise RuntimeError(f"missing H8 reference rows for {band}")

        band_output = points.copy()
        band_output["outer_fold"] = outer_fold
        for method in METHODS:
            band_output[f"pred_{method}"] = np.nan

        for fold in OUTER_FOLDS:
            print(f"H10 {band} outer_fold={fold}", flush=True)
            outer_train = outer_fold != fold
            test = outer_fold == fold
            predictions, parameters = fit_outer_models(
                points=points,
                configs=configs,
                gains=gains,
                los=los,
                tx=tx,
                observed=observed,
                groups=groups,
                outer_train=outer_train,
                test=test,
                h8_reference=band_h8["pred_GAUSSIAN_KERNEL"].to_numpy(float)[test],
            )
            for method, prediction in predictions.items():
                band_output.loc[test, f"pred_{method}"] = prediction
            reference_max_delta = max(
                reference_max_delta,
                float(parameters["gaussian"]["reference_max_abs_delta_db"]),
            )
            parameter_rows.append(
                {
                    "band": band,
                    "outer_fold": fold,
                    "parameters": json.dumps(json_safe(parameters), ensure_ascii=False, sort_keys=True),
                    "auto_selected": parameters["auto"]["selected_method"],
                }
            )
        output_frames.append(band_output)

    output = pd.concat(output_frames, ignore_index=True)
    if len(output) != 38_982 or output["point_id"].astype(str).nunique() != 38_982:
        raise RuntimeError("H10 point contract failed")
    if reference_max_delta > 1e-8:
        raise RuntimeError(f"H8 Gaussian reproduction failed: {reference_max_delta:.12g} dB")
    for method in METHODS:
        if not np.isfinite(output[f"pred_{method}"].to_numpy(float)).all():
            raise RuntimeError(f"non-finite full-coverage prediction: {method}")

    metric_rows = []
    for scope, subset in [("all", output), *( (band, output[output["band"] == band]) for band in BANDS )]:
        observed = subset["observed_dbm"].to_numpy(float)
        for method in METHODS:
            row = metrics(observed, subset[f"pred_{method}"].to_numpy(float), total_n=len(subset))
            metric_rows.append({"scope": scope, "method": method, **row})
    summary = pd.DataFrame(metric_rows).sort_values(["scope", "rmse_db", "mae_db", "method"])
    parameters = pd.DataFrame(parameter_rows)
    output.to_csv(args.output_dir / "point_predictions_wide.csv", index=False)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)
    parameters.to_csv(args.output_dir / "parameters.csv", index=False)

    all_rows = summary[summary["scope"] == "all"].set_index("method")
    champion = str(all_rows["rmse_db"].idxmin())
    baseline_rmse = float(all_rows.loc["H8_GAUSSIAN", "rmse_db"])
    champion_rmse = float(all_rows.loc[champion, "rmse_db"])
    champion_mae = float(all_rows.loc[champion, "mae_db"])
    baseline_mae = float(all_rows.loc["H8_GAUSSIAN", "mae_db"])
    successful = (
        baseline_rmse - champion_rmse >= 0.05 - 1e-12
        and champion_mae <= baseline_mae + 1e-12
    )
    result = {
        "status": "completed",
        "point_rows": int(len(output)),
        "trajectory_groups": int(output["trajectory_group"].astype(str).nunique()),
        "outer_folds": 5,
        "h8_reproduction_max_abs_delta_db": reference_max_delta,
        "champion": champion,
        "baseline_RMSE_dB": baseline_rmse,
        "baseline_MAE_dB": baseline_mae,
        "champion_RMSE_dB": champion_rmse,
        "champion_MAE_dB": champion_mae,
        "RMSE_improvement_dB": baseline_rmse - champion_rmse,
        "MAE_improvement_dB": baseline_mae - champion_mae,
        "success_threshold_met": bool(successful),
        "full_coverage_all_methods": True,
        "CSI_or_PDP_available": False,
        "interpretation": "WEDT-p narrowband RSRP physics prior; not full WEDT or unique field EM inversion",
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
