"""Strict, ID-aligned input adapter for the public radio entry point."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def load_dataset(directory):
    from .legacy.run_h11_sparse_learning_curves import BandData
    root = Path(directory)
    p = pd.read_csv(root / "points.csv", dtype={"point_id": str})
    required = {"point_id", "x", "y", "z", "band", "role", "observed_dbm"}
    if required - set(p):
        raise ValueError(f"Missing columns: {sorted(required - set(p))}")
    if p.point_id.isna().any() or p.point_id.duplicated().any():
        raise ValueError("point_id must be nonempty and unique")
    if not np.isfinite(p[["x", "y", "z"]].to_numpy(float)).all():
        raise ValueError("Coordinates must be finite local metric coordinates")
    if p.band.nunique() != 1 or p.band.iloc[0] not in ("n41", "n79"):
        raise ValueError("Use one n41 or n79 band per dataset")
    if not set(p.role) <= {"train", "query"}:
        raise ValueError("role must be train or query")
    tr = np.flatnonzero(p.role.eq("train"))
    qu = np.flatnonzero(p.role.eq("query"))
    if len(tr) < 24 or not len(qu):
        raise ValueError("At least 24 training rows and one query row required")
    if not np.isfinite(p.observed_dbm.to_numpy(float)[tr]).all():
        raise ValueError("Training labels must be finite")
    if not p.observed_dbm.iloc[qu].isna().all():
        raise ValueError("Remove query labels from points.csv; score separately")
    groups = np.floor(p[["x", "y"]].to_numpy(float)).astype(np.int64)
    if set(map(tuple, groups[tr])) & set(map(tuple, groups[qu])):
        raise ValueError("Train/query positions share a 1 m spatial group")
    with np.load(root / "rt.npz", allow_pickle=False) as rt:
        if not np.array_equal(rt["point_id"].astype(str), p.point_id.to_numpy()):
            raise ValueError("RT point_id order differs from points.csv")
        gains, los, tx = (rt[k].copy() for k in ("gains", "los", "tx"))
    configs = json.loads((root / "configs.json").read_text(encoding="utf-8"))
    n = len(p)
    if gains.shape != (len(configs), n) or not len(configs):
        raise ValueError("gains must have shape [configuration, point]")
    if los.shape != (n,) or tx.shape != (3,) or not np.isfinite(tx).all():
        raise ValueError("los must be [point], tx must be a finite [3] vector")
    if not np.isin(los, [0, 1]).all() or np.isinf(gains).any():
        raise ValueError("LOS must be boolean; missing paths use NaN, not infinity")
    if configs[0].get("family") != "BASE" or configs[0].get("id") != "AUTO_BASE":
        raise ValueError("First configuration must be the aligned AUTO_BASE RT prior")
    return BandData(p.band.iloc[0], p, configs, gains, los, tx, np.arange(n)), tr, qu


def score_predictions(predictions, truth):
    p = pd.read_csv(predictions, dtype={"point_id": str})
    t = pd.read_csv(truth, dtype={"point_id": str})
    if p.point_id.duplicated().any() or t.point_id.duplicated().any():
        raise ValueError("Duplicate scoring IDs")
    if set(p.point_id) != set(t.point_id):
        raise ValueError("Truth IDs must match query prediction IDs exactly")
    joined = p.merge(t[["point_id", "observed_dbm"]], on="point_id", validate="one_to_one")
    y = joined.observed_dbm.to_numpy(float)
    result = []
    for name in p.columns.drop("point_id"):
        error = joined[name].to_numpy(float) - y
        if not np.isfinite(error).all():
            raise ValueError("Scoring requires finite predictions and truth")
        result.append({"method": name, "n": len(y),
                       "MAE_dB": float(np.abs(error).mean()),
                       "RMSE_dB": float(np.sqrt(np.mean(error ** 2)))})
    return pd.DataFrame(result)
