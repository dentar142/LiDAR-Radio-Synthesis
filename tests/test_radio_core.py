from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.radio.legacy import aligned_factorial
from src.radio.legacy import h18_models
from src.radio.legacy import h18_validation
from src.radio.legacy import run_h18_distance_trend
from src.radio.legacy._origin import ORIGIN_SHA256
from src.radio.legacy.run_h13_statistical_revalidation import sha256_file
from src.radio.legacy.run_h11_sparse_learning_curves import BandData


def _band_data(rows: int = 48) -> BandData:
    x = np.arange(rows, dtype=float)
    points = pd.DataFrame(
        {
            "point_id": [f"p{i}" for i in range(rows)],
            "x": x,
            "y": np.mod(x * 7.0, 31.0),
            "z": np.full(rows, 1.5),
            "observed_dbm": -82.0 + 0.15 * x,
        }
    )
    gains = np.asarray([-101.0 + 0.25 * x])
    return BandData(
        band="n41",
        points=points,
        configs=[{"id": "AUTO_BASE", "family": "BASE", "groups": {}}],
        gains=gains,
        los=np.mod(np.arange(rows), 2) == 0,
        tx=np.asarray([0.0, 0.0, 20.0]),
        source_row_index=np.arange(rows),
    )


def test_frozen_runtime_provenance_declares_all_18_files() -> None:
    provenance = json.loads(Path("docs/radio-source-provenance.json").read_text(encoding="utf-8"))
    assert tuple(provenance["origin_runtime_files"]) == run_h18_distance_trend.RUNTIME_FILES
    assert provenance["origin_runtime_files"] == ORIGIN_SHA256
    current = run_h18_distance_trend.runtime_hashes()
    source = Path(run_h18_distance_trend.__file__).parent
    assert current == {name: sha256_file(source / name) for name in run_h18_distance_trend.RUNTIME_FILES}
    assert current != ORIGIN_SHA256
    assert len(provenance["aligned_factorial_sha256"]) == 64
    assert provenance["credentials_or_data_copied"] is False


def test_ten_expert_pool_erases_nonfit_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _band_data(24)
    train = np.arange(16)
    query = np.arange(16, 24)
    isolation = {}

    def fake_candidate(_spec, _train_xy, train_y, query_xy, *_args, **_kwargs):
        return np.full(len(query_xy), float(np.mean(train_y)))

    def fake_gp(_train_xy, target, query_xy, *_args, **_kwargs):
        return np.full(len(query_xy), float(np.mean(target))), {"kind": "test"}

    def fake_physical(fit_data, observed, fit_mask, *_args, **_kwargs):
        isolation["nonfit_nan"] = bool(np.isnan(observed[~fit_mask]).all())
        isolation["frame_nonfit_nan"] = bool(fit_data.points.observed_dbm[~fit_mask].isna().all())
        baseline = float(np.mean(observed[fit_mask]))
        return np.full(len(observed), baseline), {"kind": "test"}

    monkeypatch.setattr(h18_models, "predict_candidate", fake_candidate)
    monkeypatch.setattr(h18_models, "_gp", fake_gp)
    monkeypatch.setattr(h18_models, "fit_physical_sparse", fake_physical)

    matrix, info, has_path = h18_models.fit_pool(data, train, query, seed=7, device="cpu", quick=True)

    assert matrix.shape == (len(query), 10)
    assert np.isfinite(matrix).all()
    assert tuple(info) == h18_models.EXPERTS
    assert has_path.shape == (len(query),)
    assert isolation == {"nonfit_nan": True, "frame_nonfit_nan": True}


def test_calibration_rejects_query_labels_and_returns_finite_predictions() -> None:
    data = _band_data()
    fit_mask = np.zeros(len(data.points), dtype=bool)
    fit_mask[:40] = True
    observed = np.full(len(data.points), np.nan)
    observed[fit_mask] = data.points.observed_dbm.to_numpy()[fit_mask]

    prediction, details = aligned_factorial.supported_calibration(
        data, observed, fit_mask, "BASE"
    )
    assert prediction.shape == (len(data.points),)
    assert np.isfinite(prediction).all()
    assert details["prediction_clipping"] is False

    leaked = observed.copy()
    leaked[~fit_mask] = -50.0
    with pytest.raises(RuntimeError, match="target isolation"):
        aligned_factorial.supported_calibration(data, leaked, fit_mask, "BASE")


def test_query_matched_selection_uses_oof_errors() -> None:
    rows = 36
    y = np.linspace(-90.0, -70.0, rows)
    oof = np.column_stack([y + (index + 1.0) for index in range(10)])
    oof[:, 3] = y
    query_matrix = np.tile(np.arange(10, dtype=float), (7, 1))
    meta = np.column_stack((np.linspace(0.0, 1.0, rows), np.ones(rows)))
    query_meta = np.column_stack((np.linspace(0.0, 1.0, 7), np.ones(7)))

    predictions, details, weights = h18_validation.combine(
        oof, y, meta, query_meta, query_matrix
    )

    assert details["SELECT_MAE"]["selected_method"] == h18_models.EXPERTS[3]
    assert np.array_equal(predictions["SELECT_MAE"], query_matrix[:, 3])
    assert np.allclose(weights["LOCAL_RISK_MAE"].sum(axis=1), 1.0)


def test_aligned_device_auto_is_portable() -> None:
    assert aligned_factorial._resolve_device("cpu") == "cpu"
    assert aligned_factorial._resolve_device("auto") in {"cpu", "cuda"}
