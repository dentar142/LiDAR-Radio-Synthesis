import numpy as np
import pytest

from src.radio.material_adaptive.material_expert_replacement import (
    OLD_EXPERTS,
    compose,
    replacement_from_oof,
    validate_input,
)


def _data():
    xyz = np.array([[0., 0., 1.], [10., 0., 1.], [0., 10., 1.],
                    [10., 10., 1.], [20., 0., 1.], [0., 20., 1.]])
    qxyz = np.array([[5., 5., 1.], [15., 5., 1.]])
    return {"fit_xyz": xyz, "fit_y": np.array([-80., -81., -82., -83., -84., -85.]),
            "fit_id": np.array([f"f{i}" for i in range(6)]),
            "query_xyz": qxyz, "query_id": np.array(["q0", "q1"]),
            "tx": np.array([0., 0., 20.])}


def test_compose_cpu_path():
    d = _data()
    pred, meta = compose(d, np.array([-80., -81., -82., -83., -84., -85.]),
                         np.array([np.nan, -84.]), u2=True)
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
    assert meta["fit_path_n"] == 6


def test_validation_rejects_query_labels():
    d = _data()
    d["query_y"] = np.array([-80., -81.])
    with pytest.raises(ValueError, match="fit labels and unlabeled"):
        validate_input(d)


def test_oof_replacement_ranking():
    truth = np.zeros(4)
    pred = np.zeros((4, 10))
    pred[:, 0] = 3.0
    pred[:, 1] = 2.0
    result = replacement_from_oof(pred, truth)
    assert result["removed"] == [OLD_EXPERTS[0], OLD_EXPERTS[1]]
    assert len(result["pool"]) == 10
    assert result["pool"][-2:] == ["FIELD_RT_VOM", "CLASS_RT_VOM_U2"]
