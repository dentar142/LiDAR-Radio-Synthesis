#!/usr/bin/env python3
"""H8.1 leakage-safe E/W dual-source routing on the H6 raw-point contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .run_h8_combined_screen import (
    IDW_K,
    IDW_P,
    METHOD_FAMILY,
    choose_signal_params,
    choose_twc_params,
    crossfit_expert,
    crossfit_signal,
    expert_predict,
    fit_physical,
    idw,
    inner_group_map,
    json_safe,
    load_band,
    metrics,
    query_neighbors,
    signal_idw_predict,
    write_json,
)


RETENTIONS = (10, 25, 50, 75, 90)
SOURCE_PENALTIES_M = (0.0, 60.0, 120.0)


def exact_rank_mask(scores: np.ndarray, percent: int) -> np.ndarray:
    """Retain an exact percentage of finite query scores without using targets."""
    value = np.asarray(scores, float)
    finite_index = np.flatnonzero(np.isfinite(value))
    output = np.zeros(len(value), dtype=bool)
    if len(finite_index) == 0:
        return output
    count = len(finite_index) if percent >= 100 else max(1, int(round(len(finite_index) * percent / 100.0)))
    order = np.lexsort((finite_index, value[finite_index]))
    output[finite_index[order[:count]]] = True
    return output


def blend_pair(fixed: np.ndarray, alternate: np.ndarray, alternate_probability: np.ndarray) -> np.ndarray:
    fixed = np.asarray(fixed, float)
    alternate = np.asarray(alternate, float)
    probability = np.clip(np.asarray(alternate_probability, float), 0.0, 1.0)
    output = np.full(len(fixed), np.nan, float)
    both = np.isfinite(fixed) & np.isfinite(alternate)
    output[both] = (1.0 - probability[both]) * fixed[both] + probability[both] * alternate[both]
    fixed_only = np.isfinite(fixed) & ~np.isfinite(alternate)
    alternate_only = ~np.isfinite(fixed) & np.isfinite(alternate)
    output[fixed_only] = fixed[fixed_only]
    output[alternate_only] = alternate[alternate_only]
    return output


def field_features(xy: np.ndarray, auxiliary: np.ndarray | None, penalty: float) -> np.ndarray:
    if auxiliary is None or penalty <= 0:
        return np.asarray(xy, float)
    return np.column_stack([np.asarray(xy, float), np.asarray(auxiliary, float) * float(penalty)])


def tune_and_predict_field(
    xy: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    test: np.ndarray,
    *,
    auxiliary: np.ndarray | None = None,
    penalties: tuple[float, ...] = (0.0,),
    clip: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select an IDW field by group-held-out training error and crossfit it."""
    values = np.asarray(values, float)
    fold_map = inner_group_map(groups, outer_train)
    score = {(penalty, k, power): [0.0, 0] for penalty in penalties for k in IDW_K for power in IDW_P}
    for fold in range(3):
        fit = outer_train & np.isfinite(values) & np.array([fold_map.get(str(g), -1) != fold for g in groups])
        valid = outer_train & np.isfinite(values) & np.array([fold_map.get(str(g), -1) == fold for g in groups])
        if fit.sum() < 50 or valid.sum() < 10:
            continue
        for penalty in penalties:
            feature = field_features(xy, auxiliary, penalty)
            distance, index = query_neighbors(feature[fit], feature[valid], max_k=max(IDW_K))
            for k in IDW_K:
                for power in IDW_P:
                    predicted = idw(values[fit], distance, index, k, power)
                    score[(penalty, k, power)][0] += float(np.sum((predicted - values[valid]) ** 2))
                    score[(penalty, k, power)][1] += int(valid.sum())
    ranked = [
        (math.sqrt(sse / count), penalty, k, power)
        for (penalty, k, power), (sse, count) in score.items() if count > 0
    ]
    if ranked:
        inner_rmse, penalty, k, power = min(ranked)
    else:
        inner_rmse, penalty, k, power = np.nan, penalties[0], 16, 1.0

    feature = field_features(xy, auxiliary, penalty)
    oof = np.full(len(values), np.nan, float)
    for fold in range(3):
        fit = outer_train & np.isfinite(values) & np.array([fold_map.get(str(g), -1) != fold for g in groups])
        valid = outer_train & np.array([fold_map.get(str(g), -1) == fold for g in groups])
        if fit.sum() < 2 or valid.sum() == 0:
            continue
        distance, index = query_neighbors(feature[fit], feature[valid], max_k=k)
        oof[valid] = idw(values[fit], distance, index, k, power)

    fit = outer_train & np.isfinite(values)
    if fit.sum() >= 2 and test.sum() > 0:
        distance, index = query_neighbors(feature[fit], feature[test], max_k=k)
        predicted_test = idw(values[fit], distance, index, k, power)
    else:
        predicted_test = np.full(test.sum(), float(np.nanmedian(values[fit])) if fit.any() else 0.0)
    fallback = float(np.nanmedian(values[fit])) if fit.any() else 0.0
    oof[outer_train & ~np.isfinite(oof)] = fallback
    predicted_test[~np.isfinite(predicted_test)] = fallback
    if clip is not None:
        oof = np.clip(oof, clip[0], clip[1])
        predicted_test = np.clip(predicted_test, clip[0], clip[1])
    return oof, predicted_test, {
        "penalty_m": float(penalty), "k": int(k), "power": float(power),
        "inner_rmse": float(inner_rmse),
    }


def crossfit_physical(
    points: pd.DataFrame,
    configs: list[dict],
    gains: np.ndarray,
    los: np.ndarray,
    tx: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    family: str,
    class_ridge: float,
) -> np.ndarray:
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - tx[None, :], axis=1)
    nlos = ~np.asarray(los, bool)
    output = np.full(len(points), np.nan, float)
    fold_map = inner_group_map(groups, outer_train)
    for fold in range(3):
        fit = outer_train & np.array([fold_map.get(str(g), -1) != fold for g in groups])
        valid = outer_train & np.array([fold_map.get(str(g), -1) == fold for g in groups])
        physical, _ = fit_physical(
            configs, gains, observed, distance, nlos, fit, family,
            twc=True, class_ridge=class_ridge,
        )
        output[valid] = physical[valid]
    return output


def fit_full_physical(
    points: pd.DataFrame,
    configs: list[dict],
    gains: np.ndarray,
    los: np.ndarray,
    tx: np.ndarray,
    observed: np.ndarray,
    outer_train: np.ndarray,
    family: str,
    class_ridge: float,
) -> tuple[np.ndarray, dict]:
    xyz = points[["x", "y", "z"]].to_numpy(float)
    distance = np.linalg.norm(xyz - tx[None, :], axis=1)
    return fit_physical(
        configs, gains, observed, distance, ~np.asarray(los, bool), outer_train, family,
        twc=True, class_ridge=class_ridge,
    )


def pseudo_source_labels(observed, fixed_physical, alternate_physical, outer_train):
    fixed_error = np.abs(np.asarray(observed, float) - np.asarray(fixed_physical, float))
    alternate_error = np.abs(np.asarray(observed, float) - np.asarray(alternate_physical, float))
    labels = np.full(len(observed), np.nan, float)
    both = outer_train & np.isfinite(fixed_error) & np.isfinite(alternate_error)
    labels[both] = (alternate_error[both] < fixed_error[both]).astype(float)
    labels[outer_train & np.isfinite(fixed_error) & ~np.isfinite(alternate_error)] = 0.0
    labels[outer_train & ~np.isfinite(fixed_error) & np.isfinite(alternate_error)] = 1.0
    return labels


def source_curation_score(
    observed: np.ndarray,
    outer_train: np.ndarray,
    test: np.ndarray,
    fixed_oof: np.ndarray,
    alternate_oof: np.ndarray,
    fixed_test: np.ndarray,
    alternate_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scores = []
    parameters = []
    for train_physical, test_physical in ((fixed_oof, fixed_test), (alternate_oof, alternate_test)):
        residual = observed[outer_train] - train_physical[outer_train]
        finite = np.isfinite(residual)
        center = float(np.median(residual[finite]))
        mad = float(np.median(np.abs(residual[finite] - center)))
        scale = max(1.4826 * mad, 0.5)
        test_residual = observed[test] - test_physical
        score = np.full(test.sum(), 1e6, float)
        valid = np.isfinite(test_residual)
        score[valid] = np.abs(test_residual[valid] - center) / scale
        scores.append(score)
        parameters.append({"center_db": center, "robust_scale_db": scale})
    matrix = np.column_stack(scores)
    assignment = np.argmin(matrix, axis=1)
    return np.min(matrix, axis=1), assignment, {"fixed": parameters[0], "alternate": parameters[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--fixed-rt-root", required=True, type=Path)
    parser.add_argument("--alternate-rt-root", required=True, type=Path)
    parser.add_argument("--h7-score-dir", required=True, type=Path)
    parser.add_argument("--fixed-h8-predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_frame = pd.read_csv(args.h7_score_dir / "trajectory_fold_assignment.csv")
    outer_assignment = dict(zip(fold_frame["trajectory_group"].astype(str), fold_frame["outer_fold"].astype(int)))
    h7_choices = pd.read_csv(args.h7_score_dir / "method_choices.csv")
    fixed_h8 = pd.read_csv(args.fixed_h8_predictions)
    fixed_h8["point_id"] = fixed_h8["point_id"].astype(str)
    fixed_h8 = fixed_h8.set_index("point_id")

    all_output = []
    parameter_rows = []
    target_methods: set[str] = set()
    for band in ("n41", "n79"):
        points, fixed_configs, fixed_gains, fixed_los, fixed_tx = load_band(
            args.data_root, args.fixed_rt_root, band
        )
        alt_points, alt_configs, alt_gains, alt_los, alt_tx = load_band(
            args.data_root, args.alternate_rt_root, band
        )
        if not points["point_id"].astype(str).equals(alt_points["point_id"].astype(str)):
            raise RuntimeError(f"fixed/alternate point order mismatch for {band}")
        if [item["id"] for item in fixed_configs] != [item["id"] for item in alt_configs]:
            raise RuntimeError(f"fixed/alternate config mismatch for {band}")

        point_ids = points["point_id"].astype(str)
        band_h8 = fixed_h8.reindex(point_ids)
        if band_h8.isna().all(axis=1).any():
            raise RuntimeError(f"missing H8 predictions for {band}")
        observed = pd.to_numeric(points["observed_dbm"], errors="coerce").to_numpy(float)
        xy = points[["x", "y"]].to_numpy(float)
        groups = points["trajectory_group"].astype(str).to_numpy()
        outer_fold = np.asarray([outer_assignment[str(group)] for group in groups], int)

        for fold in range(5):
            outer_train = outer_fold != fold
            test = outer_fold == fold
            signal_k, signal_power, signal_inner = choose_signal_params(xy, observed, groups, outer_train)
            signal_oof = crossfit_signal(points, observed, groups, outer_train, signal_k, signal_power)
            signal_test = band_h8.loc[:, "pred_SIGNAL_IDW"].to_numpy(float)[test]

            fixed_twc = choose_twc_params(
                points, fixed_configs, fixed_gains, fixed_los, fixed_tx,
                observed, groups, outer_train, (signal_k, signal_power),
            )
            fixed_twc.update({"signal_k": signal_k, "signal_power": signal_power})
            alt_twc = choose_twc_params(
                points, alt_configs, alt_gains, alt_los, alt_tx,
                observed, groups, outer_train, (signal_k, signal_power),
            )
            alt_twc.update({"signal_k": signal_k, "signal_power": signal_power})

            pred_test: dict[str, np.ndarray] = {}
            pred_oof: dict[str, np.ndarray] = {}
            for column in band_h8.columns:
                if column.startswith("pred_"):
                    pred_test[column[5:]] = band_h8[column].to_numpy(float)[test]
            pred_oof["SIGNAL_IDW"] = signal_oof
            pred_test["SIGNAL_IDW"] = signal_test

            fixed_twc_oof = {}
            for method, family in METHOD_FAMILY.items():
                choice = h7_choices[
                    h7_choices["band"].eq(band)
                    & h7_choices["outer_fold"].eq(fold)
                    & h7_choices["method"].eq(method)
                ].iloc[0]
                standalone_params = {
                    "local_k": int(choice["idw_k"]), "local_power": float(choice["idw_power"]),
                    "class_penalty_m": 0.0, "class_ridge": 5.0,
                    "signal_k": signal_k, "signal_power": signal_power,
                }
                pred_oof[method], _ = crossfit_expert(
                    points, fixed_configs, fixed_gains, fixed_los, fixed_tx,
                    observed, groups, outer_train, family, standalone_params, twc=False,
                )
                twc_name = f"TWC_{method}"
                fixed_twc_oof[method], _ = crossfit_expert(
                    points, fixed_configs, fixed_gains, fixed_los, fixed_tx,
                    observed, groups, outer_train, family, fixed_twc, twc=True,
                )
                pred_oof[twc_name] = fixed_twc_oof[method]

            pred_oof["TWC_MEDIAN3"] = np.full(len(points), np.nan, float)
            pred_oof["TWC_MEDIAN3"][outer_train] = np.median(
                np.column_stack([fixed_twc_oof[m][outer_train] for m in METHOD_FAMILY]), axis=1
            )

            dual_oof = {}
            dual_test = {}
            source_parts = {}
            for method, family in METHOD_FAMILY.items():
                fixed_phys_oof = crossfit_physical(
                    points, fixed_configs, fixed_gains, fixed_los, fixed_tx,
                    observed, groups, outer_train, family, fixed_twc["class_ridge"],
                )
                alt_phys_oof = crossfit_physical(
                    points, alt_configs, alt_gains, alt_los, alt_tx,
                    observed, groups, outer_train, family, alt_twc["class_ridge"],
                )
                fixed_phys, fixed_param = fit_full_physical(
                    points, fixed_configs, fixed_gains, fixed_los, fixed_tx,
                    observed, outer_train, family, fixed_twc["class_ridge"],
                )
                alt_phys, alt_param = fit_full_physical(
                    points, alt_configs, alt_gains, alt_los, alt_tx,
                    observed, outer_train, family, alt_twc["class_ridge"],
                )
                labels = pseudo_source_labels(observed, fixed_phys_oof, alt_phys_oof, outer_train)
                gate_oof, gate_test, gate_param = tune_and_predict_field(
                    xy, labels, groups, outer_train, test, clip=(0.0, 1.0)
                )
                blended_oof = blend_pair(fixed_phys_oof, alt_phys_oof, gate_oof)
                blended_test = blend_pair(fixed_phys[test], alt_phys[test], gate_test)
                residual = np.full(len(points), np.nan, float)
                residual[outer_train] = observed[outer_train] - blended_oof[outer_train]
                auxiliary = np.full(len(points), np.nan, float)
                auxiliary[outer_train] = gate_oof[outer_train]
                auxiliary[test] = gate_test
                correction_oof, correction_test, residual_param = tune_and_predict_field(
                    xy, residual, groups, outer_train, test,
                    auxiliary=auxiliary, penalties=SOURCE_PENALTIES_M,
                )
                name = f"DUAL_TWC_{method}_LOCAL"
                dual_oof[name] = blended_oof + correction_oof
                dual_test[name] = blended_test + correction_test
                dual_oof[name][outer_train & ~np.isfinite(dual_oof[name])] = signal_oof[
                    outer_train & ~np.isfinite(dual_oof[name])
                ]
                missing_test = ~np.isfinite(dual_test[name])
                dual_test[name][missing_test] = signal_test[missing_test]
                source_parts[method] = {
                    "fixed_oof": fixed_phys_oof, "alternate_oof": alt_phys_oof,
                    "fixed_test": fixed_phys[test], "alternate_test": alt_phys[test],
                    "gate_oof": gate_oof, "gate_test": gate_test,
                }
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": name,
                    "fixed_tx": json.dumps(fixed_tx.tolist()), "alternate_tx": json.dumps(alt_tx.tolist()),
                    "fixed_physical": json.dumps(json_safe(fixed_param), sort_keys=True),
                    "alternate_physical": json.dumps(json_safe(alt_param), sort_keys=True),
                    "gate": json.dumps(json_safe(gate_param), sort_keys=True),
                    "local_residual": json.dumps(json_safe(residual_param), sort_keys=True),
                    "alternate_label_fraction": float(np.nanmean(labels[outer_train])),
                })

            pred_test.update(dual_test)
            pred_oof.update(dual_oof)
            dual_names = list(dual_test)
            pred_test["DUAL_TWC_MEDIAN3_LOCAL"] = np.median(
                np.column_stack([dual_test[name] for name in dual_names]), axis=1
            )
            pred_oof["DUAL_TWC_MEDIAN3_LOCAL"] = np.full(len(points), np.nan, float)
            pred_oof["DUAL_TWC_MEDIAN3_LOCAL"][outer_train] = np.median(
                np.column_stack([dual_oof[name][outer_train] for name in dual_names]), axis=1
            )

            router_candidates = [
                "SIGNAL_IDW", "TWC_U2", "TWC_WEDT_P", "TWC_ONETWIN", "TWC_MEDIAN3",
                *dual_names, "DUAL_TWC_MEDIAN3_LOCAL",
            ]
            risk_test = {}
            risk_oof = {}
            for method in router_candidates:
                absolute_oof_error = np.full(len(points), np.nan, float)
                absolute_oof_error[outer_train] = np.abs(pred_oof[method][outer_train] - observed[outer_train])
                risk_oof[method], risk_test[method], risk_param = tune_and_predict_field(
                    xy, absolute_oof_error, groups, outer_train, test
                )
                risk_oof[method] = np.maximum(risk_oof[method], 0.1)
                risk_test[method] = np.maximum(risk_test[method], 0.1)
                parameter_rows.append({
                    "band": band, "outer_fold": fold, "method": f"RISK_{method}",
                    "risk_model": json.dumps(json_safe(risk_param), sort_keys=True),
                })

            candidate_test_matrix = np.column_stack([pred_test[name] for name in router_candidates])
            candidate_oof_matrix = np.column_stack([pred_oof[name] for name in router_candidates])
            risk_test_matrix = np.column_stack([risk_test[name] for name in router_candidates])
            risk_oof_matrix = np.column_stack([risk_oof[name] for name in router_candidates])
            hard_choice_test = np.argmin(risk_test_matrix, axis=1)
            hard_choice_oof = np.argmin(risk_oof_matrix, axis=1)
            pred_test["POINTWISE_RISK_ROUTER"] = candidate_test_matrix[
                np.arange(test.sum()), hard_choice_test
            ]
            routed_oof = np.full(len(points), np.nan, float)
            train_index = np.flatnonzero(outer_train)
            routed_oof[outer_train] = candidate_oof_matrix[train_index, hard_choice_oof[outer_train]]
            pred_oof["POINTWISE_RISK_ROUTER"] = routed_oof
            risk_test["POINTWISE_RISK_ROUTER"] = np.min(risk_test_matrix, axis=1)

            test_weight = 1.0 / np.maximum(risk_test_matrix, 0.25) ** 2
            test_weight /= test_weight.sum(axis=1, keepdims=True)
            pred_test["SOFT_RISK_BLEND"] = np.sum(test_weight * candidate_test_matrix, axis=1)
            oof_weight = 1.0 / np.maximum(risk_oof_matrix[outer_train], 0.25) ** 2
            oof_weight /= oof_weight.sum(axis=1, keepdims=True)
            soft_oof = np.full(len(points), np.nan, float)
            soft_oof[outer_train] = np.sum(oof_weight * candidate_oof_matrix[outer_train], axis=1)
            pred_oof["SOFT_RISK_BLEND"] = soft_oof
            risk_test["SOFT_RISK_BLEND"] = np.sum(test_weight * risk_test_matrix, axis=1)

            safe_candidates = ["SIGNAL_IDW", "DUAL_TWC_WEDT_P_LOCAL", "DUAL_TWC_MEDIAN3_LOCAL"]
            safe_test_matrix = np.column_stack([pred_test[name] for name in safe_candidates])
            safe_oof_matrix = np.column_stack([pred_oof[name] for name in safe_candidates])
            safe_risk_test = np.column_stack([risk_test[name] for name in safe_candidates])
            safe_risk_oof = np.column_stack([risk_oof[name] for name in safe_candidates])
            safe_test_choice = np.argmin(safe_risk_test, axis=1)
            safe_oof_choice = np.argmin(safe_risk_oof, axis=1)
            pred_test["IDW_DUAL_SAFE_ROUTER"] = safe_test_matrix[
                np.arange(test.sum()), safe_test_choice
            ]
            safe_router_oof = np.full(len(points), np.nan, float)
            safe_router_oof[outer_train] = safe_oof_matrix[train_index, safe_oof_choice[outer_train]]
            pred_oof["IDW_DUAL_SAFE_ROUTER"] = safe_router_oof
            risk_test["IDW_DUAL_SAFE_ROUTER"] = np.min(safe_risk_test, axis=1)

            source_score, source_assignment, source_param = source_curation_score(
                observed, outer_train, test,
                source_parts["WEDT_P"]["fixed_oof"], source_parts["WEDT_P"]["alternate_oof"],
                source_parts["WEDT_P"]["fixed_test"], source_parts["WEDT_P"]["alternate_test"],
            )

            fold_output = points.loc[test, [
                "point_id", "band", "date", "trajectory_group", "segment_group", "x", "y", "z", "observed_dbm"
            ]].copy().reset_index(drop=True)
            fold_output["outer_fold"] = fold
            fold_output["source_score_dual"] = source_score
            fold_output["source_assignment_dual"] = source_assignment
            for percent in RETENTIONS:
                fold_output[f"retain_source_q{percent}"] = exact_rank_mask(source_score, percent)
            for method, prediction in pred_test.items():
                fold_output[f"pred_{method}"] = prediction
            for method, risk in risk_test.items():
                fold_output[f"risk_{method}"] = risk
                for percent in RETENTIONS:
                    fold_output[f"retain_target_{method}_q{percent}"] = exact_rank_mask(risk, percent)
                target_methods.add(method)
            parameter_rows.append({
                "band": band, "outer_fold": fold, "method": "COMMON_DUAL",
                "signal_k": signal_k, "signal_power": signal_power, "signal_inner_rmse": signal_inner,
                "fixed_twc": json.dumps(json_safe(fixed_twc), sort_keys=True),
                "alternate_twc": json.dumps(json_safe(alt_twc), sort_keys=True),
                "source_curation": json.dumps(json_safe(source_param), sort_keys=True),
            })
            all_output.append(fold_output)

    prediction = pd.concat(all_output, ignore_index=True)
    if len(prediction) != 38982 or prediction["point_id"].astype(str).nunique() != 38982:
        raise RuntimeError("H8.1 point contract failed")
    pred_columns = sorted(column for column in prediction if column.startswith("pred_"))
    for column in pred_columns:
        if not np.isfinite(pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)).all():
            raise RuntimeError(f"non-finite full prediction: {column}")
    prediction.to_csv(args.output_dir / "point_predictions_wide.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)

    observed = pd.to_numeric(prediction["observed_dbm"], errors="coerce").to_numpy(float)
    summary_rows = []
    for column in pred_columns:
        method = column[5:]
        predicted = pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)
        summary_rows.append({
            "track": "full_coverage", "gate": "FULL", "method": method,
            "retained_n": len(prediction), "retention": 1.0,
            **metrics(observed, predicted, len(prediction)),
        })
        if method in target_methods:
            for percent in RETENTIONS:
                mask = prediction[f"retain_target_{method}_q{percent}"].astype(bool).to_numpy()
                summary_rows.append({
                    "track": "target_blind_selective_prediction", "gate": f"RISK_Q{percent}",
                    "method": method, "retained_n": int(mask.sum()),
                    "retention": float(mask.mean()), **metrics(observed[mask], predicted[mask], int(mask.sum())),
                })
        for percent in RETENTIONS:
            mask = prediction[f"retain_source_q{percent}"].astype(bool).to_numpy()
            summary_rows.append({
                "track": "measurement_assisted_source_curation", "gate": f"DUAL_SOURCE_Q{percent}",
                "method": method, "retained_n": int(mask.sum()),
                "retention": float(mask.mean()), **metrics(observed[mask], predicted[mask], int(mask.sum())),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    full = summary[summary["track"].eq("full_coverage")].sort_values(["rmse_db", "mae_db"]).iloc[0]
    target_champions = {}
    source_champions = {}
    for percent in RETENTIONS:
        target = summary[
            summary["track"].eq("target_blind_selective_prediction")
            & summary["gate"].eq(f"RISK_Q{percent}")
        ].sort_values(["rmse_db", "mae_db"]).iloc[0]
        source = summary[
            summary["track"].eq("measurement_assisted_source_curation")
            & summary["gate"].eq(f"DUAL_SOURCE_Q{percent}")
        ].sort_values(["rmse_db", "mae_db"]).iloc[0]
        target_champions[str(percent)] = json_safe(target.to_dict())
        source_champions[str(percent)] = json_safe(source.to_dict())
    result = {
        "status": "COMPLETED", "experiment": "H8_1_dual_source_router",
        "statistical_points": int(len(prediction)),
        "champions": {
            "full_coverage": json_safe(full.to_dict()),
            "target_blind_by_fixed_retention": target_champions,
            "measurement_assisted_by_fixed_retention": source_champions,
        },
        "claim_boundaries": [
            "E/W are propagation-equivalent candidates, not surveyed transmitters",
            "target-blind ranking does not read test RSRP",
            "source curation reads test RSRP and is not unknown-position prediction",
            "WEDT-p is RSRP-only because CSI/PDP is unavailable",
        ],
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
