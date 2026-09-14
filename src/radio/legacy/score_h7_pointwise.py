#!/usr/bin/env python3
"""Score H7 on every original point with strictly train-only interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


METHOD_FAMILY = {"U2": "BASE", "WEDT_P": "S4W", "ONETWIN": "S5"}
IDW_CANDIDATES = tuple((k, p) for k in (8, 16, 32, 64) for p in (1.0, 2.0))


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


def stable_rank(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def balanced_group_folds(points: pd.DataFrame, n_folds: int) -> dict[str, int]:
    counts = points.groupby(["trajectory_group", "band"]).size().unstack(fill_value=0)
    for band in ("n41", "n79"):
        if band not in counts.columns:
            counts[band] = 0
    counts = counts[["n41", "n79"]]
    order = sorted(
        counts.index,
        key=lambda group: (-int(counts.loc[group].sum()), stable_rank(str(group))),
    )
    vectors = counts.loc[order].to_numpy(float)
    target = counts.sum(axis=0).to_numpy(float) / n_folds
    minimum = np.maximum(counts.sum(axis=0).to_numpy(float) * 0.02, 100.0)
    rng = np.random.default_rng(240903)
    best = None
    for _ in range(200000):
        candidate = rng.integers(0, n_folds, size=len(order), dtype=np.int16)
        if len(np.unique(candidate)) != n_folds:
            continue
        loads = np.zeros((n_folds, 2), dtype=float)
        np.add.at(loads, candidate, vectors)
        shortfall = np.maximum(minimum[None, :] - loads, 0.0) / minimum[None, :]
        normalized = loads / np.maximum(target[None, :], 1.0)
        score = (
            float(np.sum(shortfall**2)),
            float(np.sum((normalized - 1.0) ** 2)),
            float(np.max(normalized)),
            tuple(int(value) for value in candidate),
        )
        if best is None or score < best[0]:
            best = (score, candidate.copy())
    if best is None or best[0][0] > 0.0:
        raise RuntimeError("cannot construct band-balanced trajectory folds")
    return {str(group): int(fold) for group, fold in zip(order, best[1])}


def query_neighbors(train_xy: np.ndarray, query_xy: np.ndarray, max_k: int):
    if len(train_xy) == 0:
        raise ValueError("empty IDW training set")
    k = min(max_k, len(train_xy))
    distance, index = cKDTree(train_xy).query(query_xy, k=k, workers=-1)
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    return np.asarray(distance, float), np.asarray(index, int)


def idw_from_neighbors(
    values: np.ndarray, distance: np.ndarray, index: np.ndarray, k: int, power: float
) -> np.ndarray:
    width = min(k, index.shape[1])
    d = distance[:, :width]
    v = values[index[:, :width]]
    zero = d <= 1e-9
    zero_count = zero.sum(axis=1)
    output = np.empty(len(d), dtype=float)
    exact = zero_count > 0
    if exact.any():
        output[exact] = np.sum(np.where(zero[exact], v[exact], 0.0), axis=1) / zero_count[exact]
    if (~exact).any():
        weights = 1.0 / np.maximum(d[~exact], 1e-6) ** power
        output[~exact] = np.sum(weights * v[~exact], axis=1) / np.sum(weights, axis=1)
    return output


def regression_metrics(observed: np.ndarray, predicted: np.ndarray, total_n: int) -> dict:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    error = predicted[valid] - observed[valid]
    absolute = np.abs(error)
    return {
        "total_n": int(total_n),
        "predicted_n": int(valid.sum()),
        "coverage": float(valid.sum() / max(1, total_n)),
        "rmse_db": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
        "mae_db": float(np.mean(absolute)) if len(error) else np.nan,
        "p90_abs_db": float(np.percentile(absolute, 90)) if len(error) else np.nan,
        "bias_db": float(np.mean(error)) if len(error) else np.nan,
    }


def select_material(
    family: str,
    configs: list[dict],
    gains: np.ndarray,
    observed: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[int, float, float]:
    candidates = [i for i, config in enumerate(configs) if config["family"] == family]
    if family == "BASE":
        candidates = [i for i, config in enumerate(configs) if config["id"] == "AUTO_BASE"]
    best = None
    for index in candidates:
        finite = train_mask & np.isfinite(gains[index])
        if finite.sum() < 100:
            continue
        bias = float(np.median(observed[finite] - gains[index, finite]))
        error = gains[index, finite] + bias - observed[finite]
        rmse = float(np.sqrt(np.mean(error**2)))
        candidate = (rmse, configs[index]["id"], index, bias)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError(f"no usable training candidate for {family}")
    return int(best[2]), float(best[3]), float(best[0])


def choose_idw(
    xy: np.ndarray,
    observed: np.ndarray,
    base_prediction: np.ndarray,
    trajectory: np.ndarray,
    outer_train: np.ndarray,
) -> tuple[int, float, float]:
    groups = sorted(set(trajectory[outer_train]), key=stable_rank)
    inner_fold = {group: index % 3 for index, group in enumerate(groups)}
    scores: dict[tuple[int, float], list[float]] = {candidate: [] for candidate in IDW_CANDIDATES}
    for fold in range(3):
        fit = outer_train & np.array([inner_fold.get(group, -1) != fold for group in trajectory])
        valid = outer_train & np.array([inner_fold.get(group, -1) == fold for group in trajectory])
        fit &= np.isfinite(base_prediction)
        valid &= np.isfinite(base_prediction)
        if fit.sum() < 100 or valid.sum() < 20:
            continue
        distance, index = query_neighbors(xy[fit], xy[valid], max_k=64)
        residual = observed[fit] - base_prediction[fit]
        for k, power in IDW_CANDIDATES:
            correction = idw_from_neighbors(residual, distance, index, k, power)
            error = base_prediction[valid] + correction - observed[valid]
            scores[(k, power)].append(float(np.sqrt(np.mean(error**2))))
    ranked = [
        (float(np.mean(values)), k, power)
        for (k, power), values in scores.items() if values
    ]
    if not ranked:
        return 16, 2.0, np.nan
    inner_rmse, k, power = min(ranked)
    return int(k), float(power), float(inner_rmse)


def load_band(data_root: Path, rt_root: Path, band: str):
    points = pd.read_csv(data_root / "cache" / band / "points.csv")
    points["date"] = pd.to_numeric(points["date"], errors="coerce").astype("Int64").astype(str).str.zfill(4)
    mapping = np.load(data_root / "cache" / band / "point_to_unique.npy").astype(int)
    configs = json.loads((rt_root / "cache" / band / "configs.json").read_text(encoding="utf-8"))["configs"]
    unique_gains = []
    for config in configs:
        path = rt_root / "cache" / band / f"{config['id']}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        unique_gains.append(np.load(path))
    gains = np.asarray([gain[mapping] for gain in unique_gains], dtype=float)
    return points, configs, gains


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--rt-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_points = pd.read_csv(args.data_root / "points_all.csv", usecols=["trajectory_group", "band"])
    fold_map = balanced_group_folds(all_points, args.folds)
    pd.DataFrame(
        [{"trajectory_group": group, "outer_fold": fold} for group, fold in sorted(fold_map.items())]
    ).to_csv(args.output_dir / "trajectory_fold_assignment.csv", index=False)

    prediction_rows, fold_rows, choice_rows = [], [], []
    for band in ("n41", "n79"):
        points, configs, gains = load_band(args.data_root, args.rt_root, band)
        observed = pd.to_numeric(points["observed_dbm"], errors="coerce").to_numpy(float)
        xy = points[["x", "y"]].to_numpy(float)
        trajectory = points["trajectory_group"].astype(str).to_numpy()
        outer_folds = np.array([fold_map[group] for group in trajectory], dtype=int)
        for fold in range(args.folds):
            train = outer_folds != fold
            test = outer_folds == fold
            if train.sum() < 100 or test.sum() < 20:
                raise RuntimeError(f"insufficient fold {band}/{fold}: train={train.sum()} test={test.sum()}")
            train_mean = float(np.mean(observed[train]))
            for method, family in METHOD_FAMILY.items():
                config_index, bias, train_rmse = select_material(
                    family, configs, gains, observed, train
                )
                config_id = configs[config_index]["id"]
                base = gains[config_index] + bias
                k, power, inner_rmse = choose_idw(
                    xy, observed, base, trajectory, train
                )
                finite_train = train & np.isfinite(base)
                residual_distance, residual_index = query_neighbors(
                    xy[finite_train], xy[test], max_k=max(k, 1)
                )
                residual = observed[finite_train] - base[finite_train]
                residual_correction = idw_from_neighbors(
                    residual, residual_distance, residual_index, k, power
                )
                signal_distance, signal_index = query_neighbors(
                    xy[train], xy[test], max_k=max(k, 1)
                )
                signal_fill = idw_from_neighbors(
                    observed[train], signal_distance, signal_index, k, power
                )
                test_base = base[test]
                variants = {
                    "BASE": test_base.copy(),
                    "BASE_MEAN_FILL": np.where(np.isfinite(test_base), test_base, train_mean),
                    "IDW": np.where(
                        np.isfinite(test_base), test_base + residual_correction, np.nan
                    ),
                    "IDW_FILL": np.where(
                        np.isfinite(test_base), test_base + residual_correction, signal_fill
                    ),
                }
                choice_rows.append({
                    "band": band, "outer_fold": fold, "method": method,
                    "config_id": config_id, "train_rmse_db": train_rmse,
                    "bias_db": bias, "idw_k": k, "idw_power": power,
                    "inner_rmse_db": inner_rmse, "train_n": int(train.sum()),
                    "test_n": int(test.sum()),
                })
                test_indices = np.flatnonzero(test)
                for variant, predicted in variants.items():
                    metric = regression_metrics(observed[test], predicted, int(test.sum()))
                    fold_rows.append({
                        "band": band, "outer_fold": fold, "method": method,
                        "variant": variant, "config_id": config_id,
                        "idw_k": k, "idw_power": power, **metric,
                    })
                    for local, point_index in enumerate(test_indices):
                        prediction_rows.append({
                            "point_id": points.iloc[point_index]["point_id"],
                            "band": band, "date": points.iloc[point_index]["date"],
                            "trajectory_group": trajectory[point_index],
                            "outer_fold": fold, "method": method, "variant": variant,
                            "config_id": config_id,
                            "observed_dbm": float(observed[point_index]),
                            "predicted_dbm": float(predicted[local]) if np.isfinite(predicted[local]) else np.nan,
                            "path_available": bool(np.isfinite(test_base[local])),
                        })

    predictions = pd.DataFrame(prediction_rows)
    predictions["error_db"] = predictions["predicted_dbm"] - predictions["observed_dbm"]
    folds = pd.DataFrame(fold_rows)
    choices = pd.DataFrame(choice_rows)
    predictions.to_csv(args.output_dir / "point_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    choices.to_csv(args.output_dir / "method_choices.csv", index=False)

    summary_rows = []
    for (method, variant), frame in predictions.groupby(["method", "variant"], sort=True):
        summary_rows.append({"method": method, "variant": variant, **regression_metrics(
            frame["observed_dbm"].to_numpy(float), frame["predicted_dbm"].to_numpy(float), len(frame)
        )})
    summary = pd.DataFrame(summary_rows).sort_values(["coverage", "rmse_db"], ascending=[False, True])
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False)

    diagnostics = []
    for keys, frame in predictions.groupby(["method", "variant", "band", "date"], sort=True):
        diagnostics.append({
            "method": keys[0], "variant": keys[1], "band": keys[2], "date": keys[3],
            **regression_metrics(
                frame["observed_dbm"].to_numpy(float),
                frame["predicted_dbm"].to_numpy(float), len(frame)
            ),
        })
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "band_date_diagnostics.csv", index=False)

    eligible = summary.loc[summary["coverage"] >= 0.95]
    if len(eligible):
        champion = eligible.sort_values(["rmse_db", "mae_db"]).iloc[0].to_dict()
        rule = "lowest pooled RMSE among coverage >= 95%"
    else:
        champion = summary.sort_values(["coverage", "rmse_db"], ascending=[False, True]).iloc[0].to_dict()
        rule = "highest coverage then lowest pooled RMSE"
    result = {
        "status": "PASS", "raw_scoring_points": int(len(pd.read_csv(args.data_root / "points_all.csv"))),
        "outer_folds": int(args.folds), "date_gate": False,
        "spatial_aggregation": False, "train_only_interpolation": True,
        "interpolated_points_are_scoring_truth": False,
        "direct_test_error_filtering": False, "champion_rule": rule,
        "champion": champion,
        "claim_boundary": "pointwise pooled evaluation; E/W sources are propagation-equivalent; WEDT-p is RSRP-only",
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(json_safe(result), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
