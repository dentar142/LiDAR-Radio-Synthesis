"""Label-free finite-candidate spatial matching and H18 OOF routing.

This is not the published kNNDM algorithm. Matching is transductive to one
complete, known, unlabeled query batch; freeze it before batched inference.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from .h17_local_models import local_risk_weights
from .run_h11_sparse_learning_curves import derive_seed
from .run_h13_statistical_revalidation import balanced_spatial_fold_labels


QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SCHEMES = ("FIXED", "MATCHED")
META_BASE = ("SELECT_MAE", "SELECT_RMSE", "LOCAL_RISK_MAE", "LOCAL_RISK_RMSE")


def _xy(values, name):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite N by 2 coordinate array")
    return array


def _candidate_splits(xy, mode, seed):
    if mode not in ("infill", "spatial30", "spatial60"):
        raise ValueError("unknown spatial mode")
    buffer = 0.0 if mode == "infill" else float(mode.removeprefix("spatial"))
    cell = 1.0 if mode == "infill" else 50.0
    cells, inverse = np.unique(np.floor(xy / cell).astype(np.int64), axis=0, return_inverse=True)
    insufficient_cells = len(cells) < 3
    # An impossible three-fold partition is a documented geometric fallback,
    # not a numerical failure. Empty folds below make it explicitly infeasible.
    labels = inverse if insufficient_cells else balanced_spatial_fold_labels(
        xy, n_folds=3, cell_size_m=cell, seed=seed)
    position_groups = np.unique(np.floor(xy).astype(np.int64), axis=0, return_inverse=True)[1]
    partitions, audit, distances = [], [], []
    for fold in range(3):
        valid = np.flatnonzero(labels == fold)
        fit = np.empty(0, dtype=int)
        if len(valid):
            nearest_valid = cKDTree(xy[valid]).query(xy, workers=1)[0]
            fit = np.flatnonzero((labels != fold) & (nearest_valid >= buffer))
        if np.intersect1d(position_groups[fit], position_groups[valid]).size:
            raise RuntimeError("inner position group leakage")
        minimum = None
        if len(fit) and len(valid):
            nearest_fit = cKDTree(xy[fit]).query(xy[valid], workers=1)[0]
            minimum = float(nearest_fit.min())
            distances.append(nearest_fit)
            if minimum + 1e-8 < buffer:
                raise RuntimeError("inner buffer violation")
        partitions.append((fold, fit, valid))
        audit.append({
            "fold": fold, "fit_n": len(fit), "valid_n": len(valid),
            "fit_indices": fit.tolist(), "validation_indices": valid.tolist(),
            "buffer_m": buffer, "minimum_validation_to_fit_m": minimum,
            "diagnostic_only_insufficient_spatial_cells": insufficient_cells,
        })
    all_valid = np.concatenate([valid for _, _, valid in partitions])
    if not np.array_equal(np.sort(all_valid), np.arange(len(xy))):
        raise RuntimeError("each training row must be validated exactly once")
    feasible = all(item["fit_n"] >= 8 and item["valid_n"] >= 3 for item in audit)
    joined = np.concatenate(distances) if feasible else None
    if feasible and len(joined) != len(xy):
        raise RuntimeError("incomplete nearest-distance sample")
    return partitions, audit, feasible, joined


def _distance_summary(distances):
    values = np.asarray(distances, dtype=np.float64)
    return {
        "count": len(values),
        "quantiles_m": np.quantile(values, QUANTILES, method="linear").tolist(),
        "log1p_quantiles": np.quantile(np.log1p(values), QUANTILES, method="linear").tolist(),
        "sorted_nearest_m": np.sort(values).tolist(),
    }


def validation_designs(train_xy, query_xy, mode, seed):
    """Freeze FIXED/MATCHED partitions from coordinates, never labels/scores.

    Returned partition indices are local to train_xy. Every candidate includes
    its raw distance ECDF sample and deterministic construction metadata.
    """
    train = _xy(train_xy, "train_xy")
    query = _xy(query_xy, "query_xy")
    target_distance = cKDTree(train).query(query, workers=1)[0]
    target_summary = _distance_summary(target_distance)
    candidates, partitions_by_id = [], {}
    for candidate_id in range(8):
        candidate_seed = derive_seed(seed, "H18-inner", candidate_id)
        partitions, folds, feasible, distances = _candidate_splits(train, mode, candidate_seed)
        partitions_by_id[candidate_id] = partitions
        inner_summary = _distance_summary(distances) if feasible else None
        mismatch = None
        wasserstein = None
        if feasible:
            mismatch = float(np.mean(np.abs(
                np.asarray(inner_summary["log1p_quantiles"]) -
                np.asarray(target_summary["log1p_quantiles"])
            )))
            wasserstein = float(wasserstein_distance(distances, target_distance))
        candidates.append({
            "candidate_id": candidate_id, "seed": int(candidate_seed),
            "feasible": feasible, "folds": folds, "inner_distance": inner_summary,
            "matching_loss": mismatch, "raw_wasserstein_m": wasserstein,
        })
    eligible = [c for c in candidates if c["feasible"]]
    selected = min(eligible, key=lambda c: (c["matching_loss"], c["candidate_id"])) if eligible else None
    result = {}
    for scheme, candidate in (("FIXED", candidates[0]), ("MATCHED", selected)):
        feasible = candidate is not None and candidate["feasible"]
        chosen_id = candidate["candidate_id"] if candidate is not None else None
        result[scheme] = {
            "partitions": partitions_by_id[chosen_id] if feasible else [],
            "feasible": bool(feasible),
            "audit": {
                "scheme": scheme,
                "status": "FEASIBLE" if feasible else "NOT_FITTED_INNER_INFEASIBLE",
                "selected_candidate_id": chosen_id,
                "selected_matching_loss": candidate["matching_loss"] if feasible else None,
                "quantile_probabilities": list(QUANTILES), "quantile_method": "linear",
                "matching_objective": "mean_absolute_log1p_quantile_difference",
                "tie_break": "smallest_candidate_id",
                "query_distance": target_summary,
                "candidates": candidates,
                "query_labels_used": False,
                "partition_uses_query_coordinates": scheme == "MATCHED",
                "scope": "fixed_known_query_batch_transductive_design" if scheme == "MATCHED"
                         else "training_coordinates_only_partition",
            },
        }
    return result


def combine(oof, y, meta_features, query_features, query_matrix):
    """Four fixed OOF-trained outputs, using the H17 local-risk mechanism."""
    from .h18_models import EXPERTS

    oof = np.asarray(oof, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    if (y.ndim != 1 or len(y) == 0 or oof.ndim != 2 or
            oof.shape != (len(y), len(EXPERTS)) or query_matrix.ndim != 2 or
            query_matrix.shape[1] != len(EXPERTS) or
            not all(np.isfinite(a).all() for a in (oof, y, query_matrix))):
        raise ValueError("complete finite H18 expert matrices required")
    error = oof - y[:, None]
    predictions, details, weight_outputs = {}, {}, {}
    for metric in ("MAE", "RMSE"):
        loss = np.mean(np.abs(error), axis=0) if metric == "MAE" else np.sqrt(np.mean(error ** 2, axis=0))
        selected = int(np.argmin(loss))
        name = "SELECT_" + metric
        predictions[name] = query_matrix[:, selected].copy()
        one_hot = np.zeros_like(query_matrix)
        one_hot[:, selected] = 1.0
        weight_outputs[name] = one_hot
        details[name] = {"selected_method": EXPERTS[selected], "pooled_inner_loss": loss.tolist(),
                         "status": "FITTED", "tie_break": "fixed_expert_order"}
        weights, metadata = local_risk_weights(meta_features, error, query_features,
                                              metric=metric, k=64, shrinkage=32)
        if (not np.isfinite(weights).all() or np.any(weights < 0) or
                not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)):
            raise RuntimeError("invalid local-risk simplex")
        name = "LOCAL_RISK_" + metric
        predictions[name] = np.sum(weights * query_matrix, axis=1)
        details[name] = dict(metadata, status="FITTED")
        weight_outputs[name] = weights
    return predictions, details, weight_outputs
