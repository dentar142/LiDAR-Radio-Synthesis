#!/usr/bin/env python3
"""H13 retrospective, protocol-locked statistical revalidation.

This CPU-only driver has three independent entry points:

* ``h7-factorial`` reconstructs the complete residual-IDW x no-path-fill
  2x2 table from frozen legacy H7 out-of-fold predictions.  It does not refit
  H7 and is retained only as an audit of the invalidated legacy RT cache.
* ``factorial-refit`` refits the three H7 proxy families from a corrected
  complex-CIR cache under band-local 50 m/30 m buffered spatial folds, then
  constructs the same four factorial arms on every held-out point.
* ``spatial-revalidation`` evaluates one frequency band and one repeat with
  50 m coordinate groups, a 30 m Euclidean exclusion buffer, nested
  signal-blind budgets, and train-only model routing.

The driver never removes an outer-test point because its prediction error is
large.  Structural failures and insufficient buffered training support are
written explicitly instead of silently shrinking a budget or test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .run_h8_combined_screen import load_band
from .score_h7_pointwise import (
    METHOD_FAMILY as H7_METHOD_FAMILY,
    choose_idw as choose_h7_idw,
    idw_from_neighbors as h7_idw_from_neighbors,
    query_neighbors as h7_query_neighbors,
    select_material as select_h7_material,
)
from .run_h11_sparse_learning_curves import (
    BandData,
    TRACK_COUNTS,
    atomic_csv,
    atomic_json,
    choose_gaussian_bandwidth,
    choose_standalone_params,
    choose_twc_params_sparse,
    derive_seed,
    final_standalone,
    final_twc,
    gaussian_for_masks,
    inner_splits,
    peak_rss_mb,
    randomized_spatial_order,
    spatial_cv_labels,
)
from .trace_h13_corrected_search_cache import array_sha256 as corrected_array_sha256


BANDS = ("n41", "n79")
DEFAULT_BUDGETS = (30, 100, 300, 1000)
DEFAULT_OUTER_FOLDS = 5
DEFAULT_OUTER_CELL_M = 50.0
DEFAULT_BUFFER_M = 30.0
DEFAULT_BOOTSTRAP_DRAWS = 10_000
CORRECTED_CACHE_SCHEMA = "h13-corrected-search-cache-v1"
CORRECTED_POWER_ESTIMATOR = (
    "incoherent sum_path(|complex CIR amplitude|^2); real^2+imag^2"
)

H7_METHODS = ("U2", "WEDT_P", "ONETWIN")
H7_SOURCE_VARIANTS = ("BASE", "BASE_MEAN_FILL", "IDW", "IDW_FILL")
H7_ARMS = (
    "R0_MEAN",
    "R0_SIGNAL_IDW",
    "R1_MEAN",
    "R1_SIGNAL_IDW",
)

CANDIDATE_METHODS = (
    "GAUSSIAN",
    "U2",
    "WEDT_P",
    "ONETWIN",
    "TWC_U2",
    "TWC_WEDT_P",
    "TWC_ONETWIN",
)
ROUTER_METHOD = "TRAIN_ONLY_ROUTER"
EVALUATED_METHODS = (*CANDIDATE_METHODS, ROUTER_METHOD)
FAMILY_NAMES = {"U2": "BASE", "WEDT_P": "S4W", "ONETWIN": "S5"}


@dataclass
class BufferedOuterSplit:
    fold: int
    train_pool: np.ndarray
    test: np.ndarray
    buffer_excluded: np.ndarray
    nearest_pool_to_test_m: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_payload(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def validate_corrected_cache_manifest(manifest: dict, band: str) -> None:
    """Reject legacy or non-final RT caches before any statistical refit."""

    if manifest.get("schema") != CORRECTED_CACHE_SCHEMA:
        raise RuntimeError(
            f"factorial-refit requires {CORRECTED_CACHE_SCHEMA}, got {manifest.get('schema')!r}"
        )
    if manifest.get("band") != band:
        raise RuntimeError(f"corrected RT cache band mismatch: {manifest.get('band')!r} != {band!r}")
    if manifest.get("mode") != "full" or manifest.get("eligible_for_final") is not True:
        raise RuntimeError("benchmark or non-final corrected RT cache is not eligible")
    if manifest.get("power_estimator") != CORRECTED_POWER_ESTIMATOR:
        raise RuntimeError("RT cache does not use full complex-CIR incoherent path power")
    contract_hash = manifest.get("contract_hash")
    if not isinstance(contract_hash, str) or len(contract_hash) != 64:
        raise RuntimeError("corrected RT manifest lacks a valid contract hash")
    body = {key: value for key, value in manifest.items() if key != "contract_hash"}
    if sha256_payload(body) != contract_hash:
        raise RuntimeError("corrected RT manifest contract hash mismatch")


def corrected_job_contract_hash(manifest_contract_hash: str, config_id: str) -> str:
    """Reproduce the producer's per-artifact contract hash exactly."""

    return sha256_payload(
        {
            "manifest_contract_hash": str(manifest_contract_hash),
            "config_id": str(config_id),
        }
    )


def validate_corrected_cache_status(
    status: dict,
    *,
    manifest_contract_hash: str,
    band: str,
    config_id: str,
    receiver_count: int,
) -> None:
    """Validate both the global manifest binding and producer job binding."""

    expected_job_hash = corrected_job_contract_hash(manifest_contract_hash, config_id)
    if (
        status.get("status") != "PASS"
        or status.get("eligible_for_final") is not True
        or status.get("band") != band
        or status.get("config_id") != config_id
        or status.get("manifest_contract_hash") != manifest_contract_hash
        or status.get("contract_hash") != expected_job_hash
        or int(status.get("receiver_count", -1)) != int(receiver_count)
    ):
        raise RuntimeError(f"invalid corrected RT status contract for {config_id}")


def parse_budgets(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return DEFAULT_BUDGETS
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("budgets must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("budgets must be unique and strictly increasing")
    return values


def normalize_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"{name} has invalid boolean values: {unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def complete_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted shapes differ")
    if not len(observed):
        raise ValueError("empty scoring support")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("common-support scoring requires finite observed and predicted values")
    error = predicted - observed
    absolute = np.abs(error)
    return {
        "n": int(len(error)),
        "rmse_db": float(np.sqrt(np.mean(error**2))),
        "mae_db": float(np.mean(absolute)),
        "median_abs_db": float(np.median(absolute)),
        "p90_abs_db": float(np.percentile(absolute, 90)),
        "p95_abs_db": float(np.percentile(absolute, 95)),
        "p99_abs_db": float(np.percentile(absolute, 99)),
        "max_abs_db": float(np.max(absolute)),
        "gt10_n": int(np.sum(absolute > 10.0)),
        "gt15_n": int(np.sum(absolute > 15.0)),
        "gt10_rate": float(np.mean(absolute > 10.0)),
        "gt15_rate": float(np.mean(absolute > 15.0)),
        "bias_db": float(np.mean(error)),
    }


def construct_h7_factorial(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recombine frozen H7 predictions into all four factorial arms."""

    required = {
        "point_id",
        "band",
        "date",
        "trajectory_group",
        "outer_fold",
        "method",
        "variant",
        "config_id",
        "observed_dbm",
        "predicted_dbm",
        "path_available",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise KeyError(f"H7 prediction columns missing: {missing}")

    frame = predictions.loc[
        predictions["method"].isin(H7_METHODS)
        & predictions["variant"].isin(H7_SOURCE_VARIANTS)
    ].copy()
    frame["point_id"] = frame["point_id"].astype(str)
    frame["band"] = frame["band"].astype(str)
    frame["method"] = frame["method"].astype(str)
    frame["path_available"] = normalize_bool(frame["path_available"], "path_available")
    frame["observed_dbm"] = pd.to_numeric(frame["observed_dbm"], errors="raise")
    frame["predicted_dbm"] = pd.to_numeric(frame["predicted_dbm"], errors="coerce")
    if set(frame["method"].unique()) != set(H7_METHODS):
        raise ValueError("H7 source does not contain all three proxy methods")

    unit = ["point_id", "band", "method"]
    duplicate = frame.duplicated([*unit, "variant"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, [*unit, "variant"]].head(5).to_dict("records")
        raise ValueError(f"duplicate H7 point/method/variant rows: {sample}")

    variant_counts = frame.groupby(unit, sort=False)["variant"].nunique()
    if len(variant_counts) == 0 or not variant_counts.eq(len(H7_SOURCE_VARIANTS)).all():
        raise ValueError("each H7 point/method must contain all four source variants")

    invariant_columns = [
        "date",
        "trajectory_group",
        "outer_fold",
        "config_id",
        "observed_dbm",
        "path_available",
    ]
    for column in invariant_columns:
        inconsistent = frame.groupby(unit, sort=False)[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            raise ValueError(f"H7 invariant differs across variants: {column}")

    metadata = frame.groupby(unit, sort=False)[invariant_columns].first().reset_index()
    wide = frame.pivot(index=unit, columns="variant", values="predicted_dbm").reset_index()
    merged = metadata.merge(wide, on=unit, how="inner", validate="one_to_one")
    path = merged["path_available"].to_numpy(bool)

    base = merged["BASE"].to_numpy(float)
    base_mean = merged["BASE_MEAN_FILL"].to_numpy(float)
    residual = merged["IDW"].to_numpy(float)
    residual_signal = merged["IDW_FILL"].to_numpy(float)
    if not np.array_equal(np.isfinite(base), path):
        raise ValueError("BASE finite support disagrees with path_available")
    if not np.array_equal(np.isfinite(residual), path):
        raise ValueError("IDW finite support disagrees with path_available")
    if not np.isfinite(base_mean).all() or not np.isfinite(residual_signal).all():
        raise ValueError("H7 filled source variants must be finite on every point")

    arm_values = {
        "R0_MEAN": base_mean,
        "R0_SIGNAL_IDW": np.where(path, base, residual_signal),
        "R1_MEAN": np.where(path, residual, base_mean),
        "R1_SIGNAL_IDW": residual_signal,
    }
    factors = {
        "R0_MEAN": (False, "mean"),
        "R0_SIGNAL_IDW": (False, "signal_idw"),
        "R1_MEAN": (True, "mean"),
        "R1_SIGNAL_IDW": (True, "signal_idw"),
    }
    aligned_metadata = merged[[*unit, *invariant_columns]].copy()
    rows = []
    for arm in H7_ARMS:
        local = aligned_metadata.copy()
        local["arm"] = arm
        local["residual_idw"] = factors[arm][0]
        local["no_path_fill"] = factors[arm][1]
        local["predicted_dbm"] = np.asarray(arm_values[arm], float)
        local["error_db"] = local["predicted_dbm"] - local["observed_dbm"]
        rows.append(local)
    output = pd.concat(rows, ignore_index=True)
    if not np.isfinite(output[["observed_dbm", "predicted_dbm", "error_db"]].to_numpy(float)).all():
        raise RuntimeError("constructed H7 factorial contains non-finite scores")
    return output.sort_values(["band", "method", "point_id", "arm"]).reset_index(drop=True)


def h7_factorial_metrics(factorial: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("all", *BANDS):
        scoped = factorial if scope == "all" else factorial[factorial["band"].eq(scope)]
        for support, subset in (
            ("all_points", scoped),
            ("path_available", scoped[scoped["path_available"]]),
            ("no_path", scoped[~scoped["path_available"]]),
        ):
            if subset.empty:
                continue
            for (method, arm), group in subset.groupby(["method", "arm"], sort=True):
                rows.append(
                    {
                        "scope": scope,
                        "support": support,
                        "method": method,
                        "arm": arm,
                        "residual_idw": bool(group["residual_idw"].iloc[0]),
                        "no_path_fill": str(group["no_path_fill"].iloc[0]),
                        **complete_metrics(
                            group["observed_dbm"].to_numpy(float),
                            group["predicted_dbm"].to_numpy(float),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def factorial_contrasts(metric_by_arm: np.ndarray) -> dict[str, np.ndarray]:
    """Return positive-gain contrasts; final axis follows ``H7_ARMS``."""

    metric_by_arm = np.asarray(metric_by_arm, float)
    if metric_by_arm.shape[-1] != 4:
        raise ValueError("factorial contrast input must have four arms")
    r0_mean, r0_signal, r1_mean, r1_signal = np.moveaxis(metric_by_arm, -1, 0)
    residual_mean = r0_mean - r1_mean
    residual_signal = r0_signal - r1_signal
    fill_r0 = r0_mean - r0_signal
    fill_r1 = r1_mean - r1_signal
    return {
        "residual_idw_gain_mean_fill": residual_mean,
        "residual_idw_gain_signal_fill": residual_signal,
        "residual_idw_main_gain": 0.5 * (residual_mean + residual_signal),
        "signal_fill_gain_residual_off": fill_r0,
        "signal_fill_gain_residual_on": fill_r1,
        "signal_fill_main_gain": 0.5 * (fill_r0 + fill_r1),
        "interaction_residual_gain_signal_minus_mean": residual_signal - residual_mean,
    }


def h7_cluster_bootstrap_contrasts(
    factorial: pd.DataFrame,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = 20260905,
    interval_scope: str = "conditional fixed-OOF trajectory cluster bootstrap",
) -> pd.DataFrame:
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    rows = []
    for scope in ("all", *BANDS):
        scoped = factorial if scope == "all" else factorial[factorial["band"].eq(scope)]
        if scoped.empty:
            continue
        for method in H7_METHODS:
            subset = scoped[scoped["method"].eq(method)].copy()
            subset["trajectory_group"] = subset["trajectory_group"].astype(str)
            subset["sq_error"] = subset["error_db"] ** 2
            subset["abs_error"] = subset["error_db"].abs()
            aggregated = (
                subset.groupby(["trajectory_group", "arm"], sort=True)
                .agg(n=("error_db", "size"), sse=("sq_error", "sum"), sae=("abs_error", "sum"))
                .reset_index()
            )
            clusters = sorted(aggregated["trajectory_group"].astype(str).unique())
            if len(clusters) < 2:
                raise ValueError(f"too few trajectory clusters for {scope}/{method}")
            arrays = {}
            for field in ("n", "sse", "sae"):
                pivot = aggregated.pivot(index="trajectory_group", columns="arm", values=field)
                pivot = pivot.reindex(index=clusters, columns=H7_ARMS)
                if pivot.isna().any().any():
                    raise ValueError(f"incomplete factorial cluster table for {scope}/{method}/{field}")
                arrays[field] = pivot.to_numpy(float)

            rng = np.random.default_rng(np.uint32(derive_seed(seed, scope, method, "h7_cluster")))
            multiplicity = rng.multinomial(
                len(clusters),
                np.repeat(1.0 / len(clusters), len(clusters)),
                size=int(draws),
            )
            boot_n = multiplicity @ arrays["n"]
            boot_rmse = np.sqrt((multiplicity @ arrays["sse"]) / boot_n)
            boot_mae = (multiplicity @ arrays["sae"]) / boot_n
            point_n = arrays["n"].sum(axis=0)
            point_metrics = {
                "rmse_db": np.sqrt(arrays["sse"].sum(axis=0) / point_n),
                "mae_db": arrays["sae"].sum(axis=0) / point_n,
            }
            boot_metrics = {"rmse_db": boot_rmse, "mae_db": boot_mae}
            for metric in ("rmse_db", "mae_db"):
                point = factorial_contrasts(point_metrics[metric])
                bootstrap = factorial_contrasts(boot_metrics[metric])
                for contrast, estimate in point.items():
                    low, high = np.quantile(bootstrap[contrast], [0.025, 0.975])
                    rows.append(
                        {
                            "scope": scope,
                            "method": method,
                            "metric": metric,
                            "contrast": contrast,
                            "gain_db": float(estimate),
                            "ci_low_db": float(low),
                            "ci_high_db": float(high),
                            "trajectory_clusters": int(len(clusters)),
                            "bootstrap_draws": int(draws),
                            "interval_scope": str(interval_scope),
                        }
                    )
    return pd.DataFrame(rows)


def run_h7_factorial(args: argparse.Namespace) -> dict:
    source = Path(args.predictions)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source, low_memory=False)
    factorial = construct_h7_factorial(frame)
    metrics = h7_factorial_metrics(factorial)
    contrasts = h7_cluster_bootstrap_contrasts(
        factorial,
        draws=int(args.bootstrap_draws),
        seed=int(args.bootstrap_seed),
    )
    atomic_csv(output / "h7_factorial_predictions.csv", factorial)
    atomic_csv(output / "h7_factorial_metrics.csv", metrics)
    atomic_csv(output / "h7_factorial_contrasts.csv", contrasts)
    contract = {
        "status": "complete",
        "analysis": "legacy H7 frozen-prediction 2x2 factorial recombination audit",
        "retrospective_protocol_locked": True,
        "refit_performed": False,
        "legacy_rt_cache_invalidated": True,
        "eligible_for_new_manuscript_claims": False,
        "causal_scope": "fixed OOF predictor recombination; not a retrained factorial experiment",
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "source_rows": int(len(frame)),
        "factorial_rows": int(len(factorial)),
        "point_method_units": int(len(factorial) // len(H7_ARMS)),
        "methods": H7_METHODS,
        "arms": H7_ARMS,
        "bootstrap_unit": "trajectory_group",
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_seed": int(args.bootstrap_seed),
        "test_error_filtering": False,
    }
    atomic_json(output / "h7_factorial_contract.json", contract)
    return contract


def spatial_cell_keys(xy: np.ndarray, cell_size_m: float) -> np.ndarray:
    xy = np.asarray(xy, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("xy must be a finite n-by-2 array")
    if not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    return np.floor(xy / float(cell_size_m)).astype(np.int64)


def balanced_spatial_fold_labels(
    xy: np.ndarray,
    *,
    n_folds: int = DEFAULT_OUTER_FOLDS,
    cell_size_m: float = DEFAULT_OUTER_CELL_M,
    seed: int = 20260905,
) -> np.ndarray:
    """Assign complete coordinate cells to folds without reading signal values."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    keys = spatial_cell_keys(xy, cell_size_m)
    unique_keys, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    if len(unique_keys) < n_folds:
        raise ValueError("fewer spatial cells than outer folds")
    rng = np.random.default_rng(np.uint32(seed))
    jitter = rng.random(len(unique_keys))
    order = sorted(
        range(len(unique_keys)),
        key=lambda index: (
            -int(counts[index]),
            float(jitter[index]),
            int(unique_keys[index, 0]),
            int(unique_keys[index, 1]),
        ),
    )
    loads = np.zeros(n_folds, dtype=np.int64)
    cell_fold = np.full(len(unique_keys), -1, dtype=np.int16)
    for index in order:
        minimum = int(loads.min())
        candidates = np.flatnonzero(loads == minimum)
        fold = int(rng.choice(candidates))
        cell_fold[index] = fold
        loads[fold] += int(counts[index])
    labels = cell_fold[inverse].astype(int)
    if set(np.unique(labels)) != set(range(n_folds)):
        raise RuntimeError("outer spatial fold assignment is incomplete")
    for key in unique_keys:
        local = labels[np.all(keys == key, axis=1)]
        if len(np.unique(local)) != 1:
            raise RuntimeError("one spatial cell was split across outer folds")
    return labels


def buffered_outer_splits(
    xy: np.ndarray,
    labels: np.ndarray,
    *,
    buffer_m: float = DEFAULT_BUFFER_M,
) -> list[BufferedOuterSplit]:
    xy = np.asarray(xy, float)
    labels = np.asarray(labels, int)
    if len(xy) != len(labels):
        raise ValueError("xy and fold labels differ in length")
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m must be non-negative")
    folds = sorted(np.unique(labels).tolist())
    test_membership = np.zeros(len(xy), dtype=int)
    output = []
    for fold in folds:
        test = labels == fold
        test_membership += test.astype(int)
        candidate = ~test
        distance = np.full(len(xy), np.nan, dtype=float)
        candidate_index = np.flatnonzero(candidate)
        nearest, _ = cKDTree(xy[test]).query(xy[candidate], k=1, workers=1)
        distance[candidate_index] = np.asarray(nearest, float)
        train_pool = candidate & (distance >= float(buffer_m))
        excluded = candidate & ~train_pool
        if not test.any() or not train_pool.any():
            raise RuntimeError(f"empty buffered outer split {fold}")
        minimum = float(np.nanmin(distance[train_pool]))
        if minimum + 1e-10 < float(buffer_m):
            raise RuntimeError(f"buffer contract failed in fold {fold}: {minimum}")
        output.append(
            BufferedOuterSplit(
                fold=int(fold),
                train_pool=train_pool,
                test=test,
                buffer_excluded=excluded,
                nearest_pool_to_test_m=distance,
            )
        )
    if not np.all(test_membership == 1):
        raise RuntimeError("outer-test membership must be exactly one fold per point")
    return output


def load_h13_band(data_root: Path, rt_root: Path, band: str) -> BandData:
    points, configs, gains, los, tx = load_band(data_root, rt_root, band)
    expected = int(TRACK_COUNTS["full"][band])
    if len(points) != expected or points["point_id"].astype(str).nunique() != expected:
        raise RuntimeError(f"H13 {band} point-count contract failed: {len(points)} != {expected}")
    required = ["point_id", "x", "y", "z", "observed_dbm", "trajectory_group", "date"]
    missing = sorted(set(required) - set(points.columns))
    if missing:
        raise KeyError(f"H13 {band} point columns missing: {missing}")
    numeric = points[["x", "y", "z", "observed_dbm"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError(f"H13 {band} contains non-finite coordinates or targets")
    points = points.copy()
    points[["x", "y", "z", "observed_dbm"]] = numeric
    return BandData(
        band=band,
        points=points,
        configs=configs,
        gains=np.asarray(gains, float),
        los=np.asarray(los, bool),
        tx=np.asarray(tx, float),
        source_row_index=np.arange(len(points), dtype=np.int64),
    )


def load_corrected_h13_band(
    data_root: Path,
    rt_root: Path,
    band: str,
) -> tuple[BandData, dict]:
    """Load one final corrected complex-CIR cache and bind every source by hash."""

    data_root = Path(data_root)
    rt_root = Path(rt_root)
    band_dir = rt_root / "cache" / band
    manifest_path = band_dir / "manifest.json"
    configs_path = band_dir / "configs.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_corrected_cache_manifest(manifest, band)
    configs_payload = json.loads(configs_path.read_text(encoding="utf-8"))
    if configs_payload.get("contract_hash") != manifest["contract_hash"]:
        raise RuntimeError("configs.json is not bound to the corrected RT manifest")
    configs = configs_payload.get("configs")
    if not isinstance(configs, list) or len(configs) != 13:
        raise RuntimeError("corrected RT cache must contain exactly 13 search configurations")
    config_ids = [str(config.get("id")) for config in configs]
    if len(set(config_ids)) != len(config_ids):
        raise RuntimeError("corrected RT configuration IDs are not unique")
    if sha256_payload(configs) != manifest.get("configs_sha256"):
        raise RuntimeError("corrected RT configuration list hash mismatch")

    points_path = data_root / "cache" / band / "points.csv"
    mapping_path = data_root / "cache" / band / "point_to_unique.npy"
    geometry_path = data_root / "cache" / band / "unique_xyz.npy"
    if sha256_file(geometry_path) != manifest.get("geometry_file_sha256"):
        raise RuntimeError("receiver geometry file hash differs from corrected RT manifest")
    unique_xyz = np.load(geometry_path, allow_pickle=False)
    receiver_count = int(manifest.get("receiver_count", -1))
    if unique_xyz.shape != (receiver_count, 3):
        raise RuntimeError(
            f"corrected RT receiver shape mismatch: {unique_xyz.shape} != {(receiver_count, 3)}"
        )
    if corrected_array_sha256(np.asarray(unique_xyz, dtype=float)) != manifest.get(
        "geometry_array_sha256"
    ):
        raise RuntimeError("receiver geometry array digest mismatch")

    mapping = np.load(mapping_path, allow_pickle=False).astype(np.int64)
    if mapping.ndim != 1 or len(mapping) == 0 or mapping.min() < 0 or mapping.max() >= receiver_count:
        raise RuntimeError("point-to-unique mapping is outside corrected receiver geometry")

    source_files: dict[str, dict[str, object]] = {}

    def record(name: str, path: Path) -> None:
        source_files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }

    for name, path in (
        ("points", points_path),
        ("point_to_unique", mapping_path),
        ("unique_xyz", geometry_path),
        ("rt_manifest", manifest_path),
        ("rt_configs", configs_path),
    ):
        record(name, path)

    artifacts = [("LOS", "los_unique", np.bool_)] + [
        (config_id, config_id, np.floating) for config_id in config_ids
    ]
    for config_id, stem, expected_kind in artifacts:
        array_path = band_dir / f"{stem}.npy"
        status_path = band_dir / "status" / f"{stem}.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        validate_corrected_cache_status(
            status,
            manifest_contract_hash=manifest["contract_hash"],
            band=band,
            config_id=config_id,
            receiver_count=receiver_count,
        )
        values = np.load(array_path, allow_pickle=False)
        if values.shape != (receiver_count,):
            raise RuntimeError(f"wrong corrected RT array shape for {config_id}: {values.shape}")
        if expected_kind is np.bool_:
            if not np.issubdtype(values.dtype, np.bool_):
                raise RuntimeError("corrected LOS cache is not boolean")
        elif not np.issubdtype(values.dtype, np.floating) or np.isinf(values).any():
            raise RuntimeError(f"corrected RT gain array is invalid for {config_id}")
        if corrected_array_sha256(values) != status.get("output_sha256"):
            raise RuntimeError(f"corrected RT array digest mismatch for {config_id}")
        record(f"rt_{stem}", array_path)
        record(f"rt_{stem}_status", status_path)

    data = load_h13_band(data_root, rt_root, band)
    if len(mapping) != len(data.points):
        raise RuntimeError("point-to-unique mapping length differs from point table")
    if data.gains.shape != (len(configs), len(data.points)) or data.los.shape != (len(data.points),):
        raise RuntimeError("expanded corrected RT cache shape differs from band point table")
    if [str(config.get("id")) for config in data.configs] != config_ids:
        raise RuntimeError("loaded configuration order differs from corrected cache contract")

    source_contract = {
        "schema": "h13-factorial-refit-source-v1",
        "band": band,
        "corrected_rt_contract_hash": manifest["contract_hash"],
        "corrected_cache_schema": manifest["schema"],
        "power_estimator": manifest["power_estimator"],
        "source_files": source_files,
    }
    source_contract["source_contract_sha256"] = sha256_payload(source_contract)
    return data, source_contract


def refit_h7_proxy_fold(
    data: BandData,
    train: np.ndarray,
    test: np.ndarray,
    *,
    outer_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit H7's three proxy families using labels from one outer train only."""

    train = np.asarray(train, bool)
    test = np.asarray(test, bool)
    if train.shape != (len(data.points),) or test.shape != (len(data.points),):
        raise ValueError("refit train/test masks differ from the band point table")
    if np.any(train & test) or train.sum() < 100 or test.sum() < 1:
        raise ValueError("invalid buffered outer train/test masks for H7 refit")
    observed = data.points["observed_dbm"].to_numpy(float)
    if not np.isfinite(observed).all():
        raise ValueError("H7 refit requires finite observed_dbm")
    xy = data.points[["x", "y"]].to_numpy(float)
    trajectory = data.points["trajectory_group"].astype(str).to_numpy()
    # Deliberately erase every non-training target before any estimator sees
    # the array. Test targets are restored only while writing score records.
    train_observed = np.full(len(observed), np.nan, dtype=float)
    train_observed[train] = observed[train]
    train_mean = float(np.mean(train_observed[train]))
    test_indices = np.flatnonzero(test)
    nearest_train, _ = cKDTree(xy[train]).query(xy[test], k=1, workers=1)
    nearest_train = np.asarray(nearest_train, float)

    prediction_frames: list[pd.DataFrame] = []
    choices: list[dict] = []
    base_metadata = data.points.iloc[test_indices][
        ["point_id", "date", "trajectory_group", "x", "y", "z"]
    ].copy()
    base_metadata.insert(0, "source_row_index", data.source_row_index[test_indices])
    base_metadata.insert(2, "band", data.band)
    base_metadata["outer_fold"] = int(outer_fold)
    base_metadata["nearest_train_m"] = nearest_train
    base_metadata["train_n"] = int(train.sum())

    for method in H7_METHODS:
        family = H7_METHOD_FAMILY[method]
        config_index, bias, train_rmse = select_h7_material(
            family,
            data.configs,
            data.gains,
            train_observed,
            train,
        )
        config_id = str(data.configs[config_index]["id"])
        base = np.asarray(data.gains[config_index], float) + float(bias)
        k, power, inner_rmse = choose_h7_idw(
            xy,
            train_observed,
            base,
            trajectory,
            train,
        )
        finite_train = train & np.isfinite(base)
        if finite_train.sum() < 1:
            raise RuntimeError(f"no finite corrected RT training paths for {data.band}/{method}")
        residual_distance, residual_index = h7_query_neighbors(
            xy[finite_train], xy[test], max_k=max(k, 1)
        )
        residual = train_observed[finite_train] - base[finite_train]
        residual_correction = h7_idw_from_neighbors(
            residual,
            residual_distance,
            residual_index,
            k,
            power,
        )
        signal_distance, signal_index = h7_query_neighbors(
            xy[train], xy[test], max_k=max(k, 1)
        )
        signal_fill = h7_idw_from_neighbors(
            train_observed[train],
            signal_distance,
            signal_index,
            k,
            power,
        )
        test_base = base[test]
        path_available = np.isfinite(test_base)
        corrected = test_base + residual_correction
        variants = {
            "BASE": test_base.copy(),
            "BASE_MEAN_FILL": np.where(path_available, test_base, train_mean),
            "IDW": np.where(path_available, corrected, np.nan),
            "IDW_FILL": np.where(path_available, corrected, signal_fill),
        }
        if not np.array_equal(np.isfinite(variants["BASE"]), path_available):
            raise RuntimeError("BASE support differs from corrected RT path availability")
        if not np.array_equal(np.isfinite(variants["IDW"]), path_available):
            raise RuntimeError("IDW support differs from corrected RT path availability")
        if not np.isfinite(variants["BASE_MEAN_FILL"]).all() or not np.isfinite(
            variants["IDW_FILL"]
        ).all():
            raise RuntimeError("H7 refit no-path fill did not preserve full outer-test support")

        choices.append(
            {
                "band": data.band,
                "outer_fold": int(outer_fold),
                "method": method,
                "family": family,
                "config_id": config_id,
                "bias_db": float(bias),
                "config_train_rmse_db": float(train_rmse),
                "idw_k": int(k),
                "idw_power": float(power),
                "idw_inner_rmse_db": float(inner_rmse) if np.isfinite(inner_rmse) else np.nan,
                "train_n": int(train.sum()),
                "train_path_n": int(finite_train.sum()),
                "test_n": int(test.sum()),
                "test_path_n": int(path_available.sum()),
                "test_no_path_n": int((~path_available).sum()),
                "nearest_train_test_min_m": float(nearest_train.min()),
                "nearest_train_test_median_m": float(np.median(nearest_train)),
            }
        )
        for variant in H7_SOURCE_VARIANTS:
            local = base_metadata.copy()
            local["method"] = method
            local["variant"] = variant
            local["config_id"] = config_id
            # Test targets enter only the immutable reporting rows, after every
            # prediction and hyperparameter choice above has been completed.
            local["observed_dbm"] = observed[test]
            local["predicted_dbm"] = np.asarray(variants[variant], float)
            local["path_available"] = path_available
            local["error_db"] = local["predicted_dbm"] - local["observed_dbm"]
            prediction_frames.append(local)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    legacy_columns = [
        "point_id",
        "band",
        "date",
        "trajectory_group",
        "outer_fold",
        "method",
        "variant",
        "config_id",
        "observed_dbm",
        "predicted_dbm",
        "path_available",
        "error_db",
    ]
    predictions = predictions[
        [*legacy_columns, *(column for column in predictions if column not in legacy_columns)]
    ]
    return predictions, pd.DataFrame(choices)


def run_factorial_refit(args: argparse.Namespace) -> dict:
    """Refit H7 proxies on corrected RT and score the full band-local 2x2 design."""

    if args.band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    atomic_json(
        state_path,
        {
            "status": "running",
            "analysis": "corrected complex-CIR H7 2x2 factorial refit",
            "band": args.band,
            "pid": os.getpid(),
            "started_at_epoch": time.time(),
        },
    )
    try:
        data, source_contract = load_corrected_h13_band(
            Path(args.data_root), Path(args.rt_root), args.band
        )
        xy = data.points[["x", "y"]].to_numpy(float)
        outer_seed = derive_seed(int(args.outer_seed), args.band, "h13_factorial_refit_outer")
        labels = balanced_spatial_fold_labels(
            xy,
            n_folds=int(args.outer_folds),
            cell_size_m=float(args.outer_cell_m),
            seed=outer_seed,
        )
        splits = buffered_outer_splits(xy, labels, buffer_m=float(args.buffer_m))
        outer_manifest = make_outer_manifest(data, labels, float(args.outer_cell_m))
        outer_manifest_path = output / "outer_fold_manifest.csv"
        atomic_csv(outer_manifest_path, outer_manifest)

        raw_frames: list[pd.DataFrame] = []
        choice_frames: list[pd.DataFrame] = []
        for split in splits:
            predictions, choices = refit_h7_proxy_fold(
                data,
                split.train_pool,
                split.test,
                outer_fold=split.fold,
            )
            choices["buffer_excluded_n"] = int(split.buffer_excluded.sum())
            choices["outer_cell_m"] = float(args.outer_cell_m)
            choices["buffer_m"] = float(args.buffer_m)
            raw_frames.append(predictions)
            choice_frames.append(choices)

        raw = pd.concat(raw_frames, ignore_index=True)
        choices = pd.concat(choice_frames, ignore_index=True)
        expected_rows = len(data.points) * len(H7_METHODS) * len(H7_SOURCE_VARIANTS)
        if len(raw) != expected_rows:
            raise RuntimeError(f"H7 refit row count mismatch: {len(raw)} != {expected_rows}")
        point_counts = raw.groupby("point_id", sort=False).size()
        expected_per_point = len(H7_METHODS) * len(H7_SOURCE_VARIANTS)
        if len(point_counts) != len(data.points) or not point_counts.eq(expected_per_point).all():
            raise RuntimeError("not every band point appears exactly once per method/source variant")
        if raw.groupby("point_id", sort=False)["outer_fold"].nunique().max() != 1:
            raise RuntimeError("one point was scored in more than one outer fold")

        factorial = construct_h7_factorial(raw)
        if len(factorial) != expected_rows or not np.isfinite(
            factorial[["observed_dbm", "predicted_dbm", "error_db"]].to_numpy(float)
        ).all():
            raise RuntimeError("factorial-refit did not preserve full finite scoring support")
        metrics = h7_factorial_metrics(factorial)
        metrics = metrics[metrics["scope"].eq(args.band)].reset_index(drop=True)
        contrasts = h7_cluster_bootstrap_contrasts(
            factorial,
            draws=int(args.bootstrap_draws),
            seed=int(args.bootstrap_seed),
            interval_scope="conditional corrected-refit OOF trajectory cluster bootstrap",
        )
        contrasts = contrasts[contrasts["scope"].eq(args.band)].reset_index(drop=True)

        raw_path = output / "point_predictions.csv"
        factorial_path = output / "h7_factorial_predictions.csv"
        metrics_path = output / "h7_factorial_metrics.csv"
        contrasts_path = output / "h7_factorial_contrasts.csv"
        choices_path = output / "h7_refit_fold_choices.csv"
        source_path = output / "source_manifest.json"
        atomic_csv(raw_path, raw)
        atomic_csv(factorial_path, factorial)
        atomic_csv(metrics_path, metrics)
        atomic_csv(contrasts_path, contrasts)
        atomic_csv(choices_path, choices)
        atomic_json(source_path, source_contract)
        output_hashes = {
            path.name: sha256_file(path)
            for path in (
                outer_manifest_path,
                raw_path,
                factorial_path,
                metrics_path,
                contrasts_path,
                choices_path,
                source_path,
            )
        }
        result = {
            "status": "complete",
            "analysis": "corrected complex-CIR band-local H7 2x2 factorial refit",
            "retrospective_protocol_locked": True,
            "refit_performed": True,
            "legacy_h7_cache_used": False,
            "band": args.band,
            "frequency_pooling": False,
            "pooled_champion_generated": False,
            "point_count": int(len(data.points)),
            "raw_prediction_rows": int(len(raw)),
            "factorial_rows": int(len(factorial)),
            "methods": H7_METHODS,
            "source_variants": H7_SOURCE_VARIANTS,
            "arms": H7_ARMS,
            "outer_folds": int(args.outer_folds),
            "outer_cell_m": float(args.outer_cell_m),
            "buffer_m": float(args.buffer_m),
            "outer_seed_uint32": int(outer_seed),
            "outer_training_budget": "all points remaining after the fixed Euclidean buffer",
            "outer_test_error_filtering": False,
            "outer_test_point_membership": "exactly once per band point",
            "test_targets_used_for_fit_or_selection": False,
            "bootstrap_unit": "trajectory_group",
            "bootstrap_draws": int(args.bootstrap_draws),
            "bootstrap_seed": int(args.bootstrap_seed),
            "corrected_rt_contract_hash": source_contract["corrected_rt_contract_hash"],
            "source_contract_sha256": source_contract["source_contract_sha256"],
            "output_sha256": output_hashes,
            "claim_boundary": (
                "single-campus retrospective band-local OOF refit; H7 proxy methods, "
                "not full original-paper reproductions or independent campuses"
            ),
        }
        atomic_json(output / "h7_factorial_contract.json", result)
        atomic_json(state_path, result)
        return result
    except Exception as error:
        atomic_json(
            state_path,
            {
                "status": "failed",
                "analysis": "corrected complex-CIR H7 2x2 factorial refit",
                "band": args.band,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def fit_candidate_methods(
    data: BandData,
    train: np.ndarray,
    query: np.ndarray,
    *,
    sample_seed: int,
    budget_label: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Reuse H11 estimators but omit its three residual-hybrid candidates."""

    observed = data.points["observed_dbm"].to_numpy(float)
    xy = data.points[["x", "y"]].to_numpy(float)
    train = np.asarray(train, bool)
    query = np.asarray(query, bool)
    if train.sum() < 4 or query.sum() < 1 or np.any(train & query):
        raise ValueError("invalid train/query masks for candidate fitting")
    cv_labels = spatial_cv_labels(
        xy,
        train,
        derive_seed(sample_seed, data.band, budget_label, "h13_parameter_cv"),
    )
    predictions: dict[str, np.ndarray] = {}
    parameters: dict[str, dict] = {}

    started = time.perf_counter()
    gaussian_bw, selection = choose_gaussian_bandwidth(xy, observed, train, cv_labels)
    gaussian_query = gaussian_for_masks(xy, observed, train, query, gaussian_bw)
    predictions["GAUSSIAN"] = gaussian_query
    parameters["GAUSSIAN"] = {
        "bandwidth_m": gaussian_bw,
        **selection,
        "runtime_seconds": time.perf_counter() - started,
        "physical_fallback_n": 0,
    }

    for method, family in FAMILY_NAMES.items():
        started = time.perf_counter()
        selected, selection = choose_standalone_params(
            data, observed, train, cv_labels, family, gaussian_bw
        )
        prediction, fitted = final_standalone(
            data, observed, train, query, gaussian_query, family, selected
        )
        predictions[method] = prediction
        parameters[method] = {
            "family": family,
            "local_k": selected[0],
            "local_power": selected[1],
            **selection,
            **fitted,
            "runtime_seconds": time.perf_counter() - started,
            "physical_fallback_n": int(fitted.get("test_no_path_n", 0)),
        }

    for method, family in FAMILY_NAMES.items():
        name = f"TWC_{method}"
        started = time.perf_counter()
        selected, selection = choose_twc_params_sparse(
            data, observed, train, cv_labels, family, gaussian_bw
        )
        prediction, fitted = final_twc(
            data, observed, train, query, gaussian_query, family, selected
        )
        predictions[name] = prediction
        parameters[name] = {
            "family": family,
            "class_ridge": selected[0],
            "class_penalty_m": selected[1],
            "local_k": selected[2],
            "local_power": selected[3],
            **selection,
            **fitted,
            "runtime_seconds": time.perf_counter() - started,
            "physical_fallback_n": int(fitted.get("test_no_path_n", 0)),
        }

    if tuple(predictions) != CANDIDATE_METHODS:
        raise RuntimeError(f"candidate method order mismatch: {tuple(predictions)}")
    expected = int(query.sum())
    for method, prediction in predictions.items():
        prediction = np.asarray(prediction, float)
        if len(prediction) != expected or not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite or wrong-size H13 prediction for {method}")
    return predictions, parameters


def select_method_from_scores(scores: dict[str, tuple[float, float, int]]) -> tuple[str, dict]:
    ranked = []
    for method in CANDIDATE_METHODS:
        sse, sae, count = scores.get(method, (0.0, 0.0, 0))
        if count > 0:
            ranked.append(
                (
                    math.sqrt(float(sse) / int(count)),
                    float(sae) / int(count),
                    CANDIDATE_METHODS.index(method),
                    method,
                )
            )
    if not ranked:
        raise RuntimeError("train-only router has no finite inner-CV scores")
    rmse, mae, _, selected = min(ranked)
    details = {
        method: {
            "inner_sse": float(scores[method][0]),
            "inner_sae": float(scores[method][1]),
            "inner_n": int(scores[method][2]),
            "inner_rmse_db": float(math.sqrt(scores[method][0] / scores[method][2]))
            if scores[method][2]
            else None,
            "inner_mae_db": float(scores[method][1] / scores[method][2])
            if scores[method][2]
            else None,
        }
        for method in CANDIDATE_METHODS
    }
    return selected, {
        "selected_method": selected,
        "selected_inner_rmse_db": float(rmse),
        "selected_inner_mae_db": float(mae),
        "candidate_scores": details,
        "tie_break": "RMSE, then MAE, then preregistered candidate order",
    }


def choose_train_only_router(
    data: BandData,
    train: np.ndarray,
    *,
    sample_seed: int,
    budget: int,
) -> tuple[str, dict]:
    """Nested selection: outer-test coordinates and targets never score a route."""

    xy = data.points[["x", "y"]].to_numpy(float)
    observed = data.points["observed_dbm"].to_numpy(float)
    cv_labels = spatial_cv_labels(
        xy,
        train,
        derive_seed(sample_seed, data.band, budget, "h13_router_cv"),
    )
    scores = {method: [0.0, 0.0, 0] for method in CANDIDATE_METHODS}
    folds = []
    for inner_fold, fit, valid in inner_splits(train, cv_labels, min_fit=8, min_valid=2):
        predictions, _ = fit_candidate_methods(
            data,
            fit,
            valid,
            sample_seed=derive_seed(sample_seed, "h13_router_fit", inner_fold),
            budget_label=int(fit.sum()),
        )
        for method in CANDIDATE_METHODS:
            error = np.asarray(predictions[method], float) - observed[valid]
            if not np.isfinite(error).all():
                raise RuntimeError(f"non-finite router validation error for {method}")
            scores[method][0] += float(np.sum(error**2))
            scores[method][1] += float(np.sum(np.abs(error)))
            scores[method][2] += int(len(error))
        folds.append(
            {
                "inner_fold": int(inner_fold),
                "fit_n": int(fit.sum()),
                "valid_n": int(valid.sum()),
            }
        )
    if len(folds) < 2:
        raise RuntimeError(f"train-only router has fewer than two usable inner folds: {folds}")
    selected, details = select_method_from_scores(
        {method: tuple(value) for method, value in scores.items()}
    )
    details.update(
        {
            "selection_scope": "training-only nested spatial CV",
            "outer_test_used_for_selection": False,
            "inner_folds": folds,
        }
    )
    return selected, details


def make_outer_manifest(data: BandData, labels: np.ndarray, cell_size_m: float) -> pd.DataFrame:
    keys = spatial_cell_keys(data.points[["x", "y"]].to_numpy(float), cell_size_m)
    return pd.DataFrame(
        {
            "source_row_index": data.source_row_index,
            "point_id": data.points["point_id"].astype(str),
            "band": data.band,
            "outer_cell_gx": keys[:, 0],
            "outer_cell_gy": keys[:, 1],
            "outer_fold": labels,
        }
    )


def run_spatial_revalidation(args: argparse.Namespace) -> dict:
    if args.band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}")
    if not (0 <= int(args.sample_seed) <= 2**32 - 1):
        raise ValueError("sample_seed must be uint32")
    if int(args.repeat_id) < 0:
        raise ValueError("repeat_id must be non-negative")
    budgets = parse_budgets(args.budgets)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "state.json"
    atomic_json(
        state_path,
        {
            "status": "running",
            "band": args.band,
            "repeat_id": int(args.repeat_id),
            "sample_seed": int(args.sample_seed),
            "pid": os.getpid(),
            "started_at_epoch": time.time(),
        },
    )
    try:
        data, source_contract = load_corrected_h13_band(
            Path(args.data_root), Path(args.rt_root), args.band
        )
        xy = data.points[["x", "y"]].to_numpy(float)
        labels = balanced_spatial_fold_labels(
            xy,
            n_folds=int(args.outer_folds),
            cell_size_m=float(args.outer_cell_m),
            seed=derive_seed(int(args.outer_seed), args.band, "h13_outer_cells"),
        )
        splits = buffered_outer_splits(xy, labels, buffer_m=float(args.buffer_m))
        outer_manifest_path = output / "outer_fold_manifest.csv"
        atomic_csv(outer_manifest_path, make_outer_manifest(data, labels, float(args.outer_cell_m)))

        status_rows: list[dict] = []
        metric_rows: list[dict] = []
        prediction_frames: list[pd.DataFrame] = []
        parameter_payload: dict[str, dict] = {}
        observed = data.points["observed_dbm"].to_numpy(float)
        all_complete = True

        for split in splits:
            pool_indices = np.flatnonzero(split.train_pool)
            feasible = [budget for budget in budgets if budget <= len(pool_indices)]
            if feasible:
                relative_order = randomized_spatial_order(
                    xy[pool_indices],
                    max(feasible),
                    derive_seed(
                        int(args.sample_seed),
                        args.band,
                        int(args.repeat_id),
                        split.fold,
                        "h13_sample_order",
                    ),
                )
                nested_order = pool_indices[relative_order]
            else:
                nested_order = np.empty(0, dtype=np.int64)

            test_indices = np.flatnonzero(split.test)
            for budget in budgets:
                common = {
                    "band": args.band,
                    "repeat_id": int(args.repeat_id),
                    "sample_seed": int(args.sample_seed),
                    "outer_fold": int(split.fold),
                    "budget": int(budget),
                    "outer_test_n": int(split.test.sum()),
                    "buffered_train_pool_n": int(split.train_pool.sum()),
                    "buffer_excluded_n": int(split.buffer_excluded.sum()),
                    "outer_cell_m": float(args.outer_cell_m),
                    "buffer_m": float(args.buffer_m),
                }
                if budget > len(pool_indices):
                    all_complete = False
                    status_rows.append(
                        {
                            **common,
                            "status": "INSUFFICIENT_BUFFERED_TRAINING_SUPPORT",
                            "reason": f"requested={budget}; available={len(pool_indices)}; budget_not_reduced",
                        }
                    )
                    continue

                train = np.zeros(len(xy), dtype=bool)
                train[nested_order[:budget]] = True
                if int(train.sum()) != budget:
                    raise RuntimeError("nested budget did not select the exact requested sample count")
                nearest_test, _ = cKDTree(xy[train]).query(xy[split.test], k=1, workers=1)
                nearest_test = np.asarray(nearest_test, float)
                if float(nearest_test.min()) + 1e-10 < float(args.buffer_m):
                    raise RuntimeError("sampled training set violates outer Euclidean buffer")

                started = time.perf_counter()
                try:
                    selected_method, router = choose_train_only_router(
                        data,
                        train,
                        sample_seed=derive_seed(
                            int(args.sample_seed), args.band, split.fold, budget, "h13_router"
                        ),
                        budget=int(budget),
                    )
                    predictions, fitted = fit_candidate_methods(
                        data,
                        train,
                        split.test,
                        sample_seed=derive_seed(
                            int(args.sample_seed), args.band, split.fold, budget, "h13_final"
                        ),
                        budget_label=int(budget),
                    )
                    predictions[ROUTER_METHOD] = np.asarray(predictions[selected_method], float).copy()
                    if tuple(predictions) != EVALUATED_METHODS:
                        raise RuntimeError("H13 evaluated method order mismatch")
                    if any(
                        len(prediction) != len(test_indices)
                        or not np.isfinite(np.asarray(prediction, float)).all()
                        for prediction in predictions.values()
                    ):
                        raise RuntimeError("outer predictions do not share complete finite support")

                    for method in EVALUATED_METHODS:
                        metrics = complete_metrics(observed[split.test], predictions[method])
                        method_parameters = fitted[selected_method] if method == ROUTER_METHOD else fitted[method]
                        metric_rows.append(
                            {
                                **common,
                                "method": method,
                                "router_selected_method": selected_method if method == ROUTER_METHOD else "",
                                "train_n": int(train.sum()),
                                "nearest_train_test_min_m": float(nearest_test.min()),
                                "nearest_train_test_median_m": float(np.median(nearest_test)),
                                "nearest_train_test_p90_m": float(np.percentile(nearest_test, 90)),
                                "physical_fallback_n": int(
                                    method_parameters.get("physical_fallback_n", 0)
                                ),
                                **metrics,
                            }
                        )

                    point_frame = data.points.iloc[test_indices][
                        ["point_id", "date", "trajectory_group", "x", "y", "z", "observed_dbm"]
                    ].copy()
                    point_frame.insert(0, "source_row_index", data.source_row_index[test_indices])
                    point_frame.insert(1, "band", args.band)
                    point_frame["repeat_id"] = int(args.repeat_id)
                    point_frame["sample_seed"] = int(args.sample_seed)
                    point_frame["outer_fold"] = int(split.fold)
                    point_frame["budget"] = int(budget)
                    point_frame["nearest_train_m"] = nearest_test
                    point_frame["router_selected_method"] = selected_method
                    for method in EVALUATED_METHODS:
                        point_frame[f"pred_{method}"] = predictions[method]
                    prediction_frames.append(point_frame)
                    parameter_payload[f"fold_{split.fold}/budget_{budget}"] = {
                        "router": router,
                        "final_methods": fitted,
                        "sampled_source_row_index": data.source_row_index[train].astype(int).tolist(),
                    }
                    status_rows.append(
                        {
                            **common,
                            "status": "COMPLETE",
                            "reason": "",
                            "train_n": int(train.sum()),
                            "router_selected_method": selected_method,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                except Exception as error:  # condition failure is an explicit data product
                    all_complete = False
                    status_rows.append(
                        {
                            **common,
                            "status": "FAILED",
                            "reason": f"{type(error).__name__}: {error}",
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )

        status_frame = pd.DataFrame(status_rows)
        metric_frame = pd.DataFrame(metric_rows)
        prediction_frame = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        condition_path = output / "condition_status.csv"
        metrics_path = output / "metrics.csv"
        predictions_path = output / "point_predictions_wide.csv"
        parameters_path = output / "parameters.json"
        source_path = output / "source_manifest.json"
        atomic_csv(condition_path, status_frame)
        atomic_csv(metrics_path, metric_frame)
        atomic_csv(predictions_path, prediction_frame)
        atomic_json(
            parameters_path,
            {
                "band": args.band,
                "repeat_id": int(args.repeat_id),
                "sample_seed": int(args.sample_seed),
                "outer_seed": int(args.outer_seed),
                "parameters": parameter_payload,
            },
        )
        atomic_json(source_path, source_contract)
        output_hashes = {
            path.name: sha256_file(path)
            for path in (
                outer_manifest_path,
                condition_path,
                metrics_path,
                predictions_path,
                parameters_path,
                source_path,
            )
        }
        completed_conditions = int(status_frame["status"].eq("COMPLETE").sum())
        expected_conditions = int(len(splits) * len(budgets))
        state = {
            "status": "complete" if all_complete and completed_conditions == expected_conditions else "partial",
            "analysis": "corrected complex-CIR band-local buffered spatial revalidation",
            "retrospective_protocol_locked": True,
            "band": args.band,
            "repeat_id": int(args.repeat_id),
            "sample_seed": int(args.sample_seed),
            "budgets": budgets,
            "outer_folds": int(args.outer_folds),
            "outer_cell_m": float(args.outer_cell_m),
            "buffer_m": float(args.buffer_m),
            "completed_conditions": completed_conditions,
            "expected_conditions": expected_conditions,
            "metric_rows": int(len(metric_frame)),
            "prediction_rows": int(len(prediction_frame)),
            "evaluated_methods": EVALUATED_METHODS,
            "frequency_pooling": False,
            "outer_test_error_filtering": False,
            "outer_test_used_for_model_or_route_selection": False,
            "common_outer_test_support_per_condition": True,
            "corrected_cache_schema": source_contract["corrected_cache_schema"],
            "corrected_rt_contract_hash": source_contract["corrected_rt_contract_hash"],
            "source_contract_sha256": source_contract["source_contract_sha256"],
            "power_estimator": source_contract["power_estimator"],
            "output_sha256": output_hashes,
            "peak_rss_mb": peak_rss_mb(),
        }
        atomic_json(state_path, state)
        return state
    except Exception as error:
        atomic_json(
            state_path,
            {
                "status": "failed",
                "band": args.band,
                "repeat_id": int(args.repeat_id),
                "sample_seed": int(args.sample_seed),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    factorial = subparsers.add_parser("h7-factorial")
    factorial.add_argument("--predictions", required=True, type=Path)
    factorial.add_argument("--output-dir", required=True, type=Path)
    factorial.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    factorial.add_argument("--bootstrap-seed", type=int, default=20260905)

    refit = subparsers.add_parser("factorial-refit")
    refit.add_argument("--data-root", required=True, type=Path)
    refit.add_argument("--rt-root", required=True, type=Path)
    refit.add_argument("--band", choices=BANDS, required=True)
    refit.add_argument("--output-dir", required=True, type=Path)
    refit.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    refit.add_argument("--outer-cell-m", type=float, default=DEFAULT_OUTER_CELL_M)
    refit.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    refit.add_argument("--outer-seed", type=int, default=20260905)
    refit.add_argument("--bootstrap-draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    refit.add_argument("--bootstrap-seed", type=int, default=20260905)

    spatial = subparsers.add_parser("spatial-revalidation")
    spatial.add_argument("--data-root", required=True, type=Path)
    spatial.add_argument("--rt-root", required=True, type=Path)
    spatial.add_argument("--band", choices=BANDS, required=True)
    spatial.add_argument("--repeat-id", required=True, type=int)
    spatial.add_argument("--sample-seed", required=True, type=int)
    spatial.add_argument("--output-dir", required=True, type=Path)
    spatial.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    spatial.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    spatial.add_argument("--outer-cell-m", type=float, default=DEFAULT_OUTER_CELL_M)
    spatial.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    spatial.add_argument("--outer-seed", type=int, default=20260905)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "h7-factorial":
        result = run_h7_factorial(args)
    elif args.command == "factorial-refit":
        result = run_factorial_refit(args)
    elif args.command == "spatial-revalidation":
        result = run_spatial_revalidation(args)
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if result.get("status") != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
