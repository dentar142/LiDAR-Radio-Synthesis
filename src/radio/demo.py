"""Deterministic synthetic fixture. Not report data or a physical RT simulation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def create_fixture(output, seed=42):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Fixture output must be empty")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 300, (84, 2))
    xyz = np.column_stack((xy, np.full(len(xy), 1.5)))
    tx = np.array([150., 150., 30.])
    gains = -30 - 20 * np.log10(np.linalg.norm(xyz - tx, axis=1))
    truth = gains + 4 * np.sin(xy[:, 0] / 70) + rng.normal(0, 1, len(xy))
    ids = np.array([f"synthetic-{i:04d}" for i in range(len(xy))])
    p = pd.DataFrame(xyz, columns=["x", "y", "z"])
    p.insert(0, "point_id", ids)
    p["band"] = "n41"
    p["role"] = ["train"] * 60 + ["query"] * 24
    p["observed_dbm"] = truth
    p.loc[p.role.eq("query"), "observed_dbm"] = np.nan
    p.to_csv(out / "points.csv", index=False)
    pd.DataFrame({"point_id": ids[60:], "observed_dbm": truth[60:]}).to_csv(
        out / "truth.csv", index=False)
    np.savez_compressed(out / "rt.npz", point_id=ids, gains=gains[None],
                        los=xy[:, 0] > 100, tx=tx)
    (out / "configs.json").write_text(json.dumps(
        [{"id": "AUTO_BASE", "family": "BASE", "groups": {}}]), encoding="utf-8")
    (out / "README.txt").write_text(
        "SYNTHETIC SOFTWARE TEST ONLY. Analytic mock path gains, not ray-traced data.\n",
        encoding="utf-8")
