"""Query-batch selection using the integrated H18 models and validation design."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_dataset


def predict(directory, output, *, mode="infill", seed=241301, quick=False, device="cpu"):
    from .legacy import h18_models as models
    from .legacy.aligned_factorial import supported_calibration
    from .legacy.h18_validation import validation_designs, combine
    from .legacy.h17_local_models import risk_features
    data, tr, qu = load_dataset(directory)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Prediction output must be empty")
    xy = data.points[["x", "y"]].to_numpy(float)
    designs = validation_designs(xy[tr], xy[qu], mode, seed)
    design = designs["MATCHED"]
    if not design["feasible"]:
        raise ValueError("No feasible matched three-fold validation for this support")
    oof = np.full((len(tr), len(models.EXPERTS)), np.nan)
    features = np.full((len(tr), 4), np.nan)
    original = models.fit_physical_sparse
    # This is the calibration used by the aligned campaign. Restore on all exits.
    models.fit_physical_sparse = supported_calibration
    try:
        for fold, fit, valid in design["partitions"]:
            matrix, _, path = models.fit_pool(
                data, tr[fit], tr[valid], seed + fold, device=device, quick=quick)
            oof[valid] = matrix
            features[valid] = risk_features(xy[tr[fit]], xy[tr[valid]], matrix, path)
        matrix, expert_info, path = models.fit_pool(
            data, tr, qu, seed, device=device, quick=quick)
    finally:
        models.fit_physical_sparse = original
    query_features = risk_features(xy[tr], xy[qu], matrix, path)
    selected, details, weights = combine(
        oof, data.points.observed_dbm.to_numpy(float)[tr], features, query_features, matrix)
    frame = pd.DataFrame(matrix, columns=models.EXPERTS)
    for name, values in selected.items():
        frame["MATCHED_" + name] = values
    frame.insert(0, "point_id", data.points.point_id.to_numpy()[qu])
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(oof, columns=models.EXPERTS).assign(
        point_id=data.points.point_id.to_numpy()[tr]).to_csv(out / "oof.csv", index=False)
    np.savez_compressed(out / "weights.npz", **weights)
    audit = {"mode": mode, "seed": seed, "quick": quick, "device": device,
             "query_labels_used": False, "training_n": len(tr), "query_n": len(qu),
             "calibration": "aligned_supported_power", "validation": design["audit"],
             "selection": details, "experts": expert_info}
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return frame
