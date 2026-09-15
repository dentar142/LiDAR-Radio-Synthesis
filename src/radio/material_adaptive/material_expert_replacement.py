"""Two material-updating candidates; does not mutate the frozen H18 pool.

Worker inputs contain fit labels and unlabeled query coordinates only.
VOM denotes the existing line-integral attenuation proxy, not a full VOM solver.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.spatial import cKDTree

NEW_EXPERTS = ("FIELD_RT_VOM", "CLASS_RT_VOM_U2")
OLD_EXPERTS = ("NW50", "GP_XY_M32", "RT_PRIOR", "RT_GP", "GEOMETRY_KRR",
               "TREND_ONLY", "TREND_GP", "RT_TREND_ONLY", "RT_TREND_GP",
               "RT_TREND_SHRUNK_GP")
INPUT_KEYS = {"fit_xyz", "fit_y", "fit_id", "query_xyz", "query_id", "tx"}


def validate_input(data):
    if set(data) != INPUT_KEYS:
        raise ValueError("worker accepts fit labels and unlabeled queries only")
    n, q = len(data["fit_xyz"]), len(data["query_xyz"])
    if n < 3 or q < 1:
        raise ValueError("insufficient fit/query rows")
    for key, shape in (("fit_xyz", (n, 3)), ("query_xyz", (q, 3)),
                       ("fit_y", (n,)), ("tx", (3,))):
        value = np.asarray(data[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid " + key)
    for key, length in (("fit_id", n), ("query_id", q)):
        value = np.asarray(data[key]).astype(str)
        if value.shape != (length,) or len(np.unique(value)) != length:
            raise ValueError("duplicate/misaligned IDs")
    if np.intersect1d(data["fit_id"], data["query_id"]).size:
        raise ValueError("fit/query ID overlap")
    if cKDTree(data["fit_xyz"]).query(data["query_xyz"])[0].min() < 1e-8:
        raise ValueError("fit/query coordinate overlap")


def replacement_from_oof(prediction, truth, names=OLD_EXPERTS):
    """Call separately for each outer-training task, never on outer test rows.

    Mean OOF absolute error is primary; RMSE breaks ties, then expert name.
    Selection may not be interpreted as an unbiased OOF estimate after selection.
    """
    p, y = np.asarray(prediction, float), np.asarray(truth, float)
    if tuple(names) != OLD_EXPERTS or p.shape != (len(y), 10) or y.ndim != 1:
        raise ValueError("expected complete ten-expert training OOF matrix")
    if not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("incomplete OOF support")
    error = p - y[:, None]
    mae, rmse = np.mean(abs(error), axis=0), np.sqrt(np.mean(error**2, axis=0))
    order = sorted(range(10), key=lambda i: (-mae[i], -rmse[i], names[i]))
    removed = tuple(names[i] for i in order[:2])
    return {"removed": list(removed),
            "pool": [x for x in names if x not in removed] + list(NEW_EXPERTS),
            "ranking": [{"expert": names[i], "oof_mae": float(mae[i]),
                         "oof_rmse": float(rmse[i])} for i in order]}


def compose(data, material_train, material_query, *, u2):
    from .h20_common import nw50, physical_mean
    from .h20_composition import fit_grid, fixed_grid_bounds
    xyz, qxyz, y, tx = (data[k] for k in ("fit_xyz", "query_xyz", "fit_y", "tx"))
    path, qpath = np.isfinite(material_train), np.isfinite(material_query)
    if path.sum() < 3:
        raise RuntimeError("fewer than three material-supported training points")
    fallback = nw50(xyz[:, :2], y, np.vstack((xyz[:, :2], qxyz[:, :2])))
    fitted, predicted, beta = physical_mean(
        y, material_train, material_query, fallback[:len(y)], fallback[len(y):])
    bounds = fixed_grid_bounds(np.vstack((xyz, qxyz)), tx)
    vom = fit_grid(xyz[path], tx, (y-fitted)[path], bounds=bounds)
    fitted = fitted + np.where(path, vom.predict(xyz, tx), 0.)
    predicted = predicted + np.where(qpath, vom.predict(qxyz, tx), 0.)
    if u2 and qpath.any():
        # Fixed, predeclared local residual rule; no tuning on query outcomes.
        k = min(8, int(path.sum()))
        distance, index = cKDTree(xyz[path, :2]).query(qxyz[qpath, :2], k=k)
        distance, index = distance.reshape(-1, k), index.reshape(-1, k)
        weight = 1. / np.maximum(distance, 1e-6)**2
        weight /= weight.sum(axis=1, keepdims=True)
        predicted[qpath] += np.sum(weight * (y-fitted)[path][index], axis=1)
    if not np.isfinite(predicted).all():
        raise RuntimeError("nonfinite composed prediction")
    return predicted, {"beta_db": beta, "vom": vom.metadata(),
                       "fit_path_n": int(path.sum()), "query_path_n": int(qpath.sum()),
                       "query_n": len(qxyz), "fallback": "same-fit NW50",
                       "u2_residual": "fixed k=8 inverse-distance-squared" if u2 else None}


def run(args):
    from .h20_common import write_json, sha
    if args.output.exists():
        raise FileExistsError("use a fresh output directory; no silent overwrite/resume")
    with np.load(args.input, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_input(data)
    if not 0 <= args.seed <= np.iinfo(np.uint32).max or args.steps < 1 or args.samples < 1:
        raise ValueError("invalid seed or training budget")
    args.output.mkdir(parents=True)
    if args.expert == "FIELD_RT_VOM":
        from .h16_material import fit_material
        result = fit_material(args.scene, args.priors, args.band, data["tx"],
            data["fit_xyz"], data["fit_y"], data["query_xyz"], seed=args.seed,
            steps=args.steps, samples=args.samples, output_dir=args.output / "material")
        branch = result
    else:
        from .h20_banked_fit import fit_material_branches_banked
        zero = np.zeros(len(data["fit_y"]), np.float32)
        result = fit_material_branches_banked(args.scene, args.priors, args.band, data["tx"],
            data["fit_xyz"], data["fit_y"], data["query_xyz"], train_row_ids=data["fit_id"],
            target_offsets={"TWC_CLASS": zero, "TWC_GRID": zero}, seed=args.seed,
            steps=args.steps, samples=args.samples, output_dir=args.output / "material",
            deduplicate_equal_targets=True)
        branch = result["branches"]["RAW"]
    prediction, metadata = compose(data, branch["material_train"], branch["material_query"],
                                   u2=args.expert == "CLASS_RT_VOM_U2")
    np.savez_compressed(args.output / "prediction.npz", query_id=data["query_id"], prediction=prediction)
    write_json(args.output / "status.json", {"status": "COMPLETE", "expert": args.expert,
        "band": args.band, "steps": args.steps, "samples": args.samples,
        "input_sha256": sha(args.input), "scene_sha256": sha(args.scene),
        "priors_sha256": sha(args.priors), "worker_sha256": sha(__file__),
        "material": result["metadata"], "composition": metadata,
        "query_labels_used": False, "accuracy_evaluated": False})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--priors", type=Path, required=True)
    p.add_argument("--band", choices=("n41", "n79"), required=True)
    p.add_argument("--expert", choices=NEW_EXPERTS, required=True)
    p.add_argument("--seed", type=int, default=130913)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--samples", type=int, default=50000)
    run(p.parse_args())
