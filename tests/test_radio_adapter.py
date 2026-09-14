import numpy as np
import pandas as pd
import pytest
from src.radio.demo import create_fixture
from src.radio.data import load_dataset, score_predictions


def test_fixture_and_hidden_query_labels(tmp_path):
    create_fixture(tmp_path)
    data, train, query = load_dataset(tmp_path)
    assert (len(train), len(query)) == (60, 24)
    assert data.points.observed_dbm.iloc[query].isna().all()
    p = pd.read_csv(tmp_path / "points.csv")
    p.loc[p.role == "query", "observed_dbm"] = -80
    p.to_csv(tmp_path / "points.csv", index=False)
    with pytest.raises(ValueError, match="query labels"):
        load_dataset(tmp_path)


def test_rt_alignment_is_strict(tmp_path):
    create_fixture(tmp_path)
    with np.load(tmp_path / "rt.npz") as rt:
        values = {k: rt[k] for k in rt.files}
    values["point_id"] = values["point_id"][::-1]
    np.savez_compressed(tmp_path / "rt.npz", **values)
    with pytest.raises(ValueError, match="point_id order"):
        load_dataset(tmp_path)


def test_scoring_matches_ids_not_row_order(tmp_path):
    pd.DataFrame({"point_id": ["a", "b"], "model": [1., 3.]}).to_csv(tmp_path / "p.csv", index=False)
    pd.DataFrame({"point_id": ["b", "a"], "observed_dbm": [4., 2.]}).to_csv(tmp_path / "t.csv", index=False)
    result = score_predictions(tmp_path / "p.csv", tmp_path / "t.csv")
    assert result.MAE_dB.iloc[0] == result.RMSE_dB.iloc[0] == 1.
