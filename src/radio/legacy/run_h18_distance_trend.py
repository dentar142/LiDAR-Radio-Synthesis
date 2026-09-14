#!/usr/bin/env python3
"""H18 distance-matched validation and low-order trend campaign runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import run_h15_physical_neural as io
from .h17_local_models import risk_features
from .run_h11_sparse_learning_curves import atomic_csv, atomic_json, derive_seed
from .run_h13_statistical_revalidation import balanced_spatial_fold_labels, sha256_file


BANDS = ("n41", "n79")
MODES = ("infill", "spatial30", "spatial60")
BUDGETS = (30, 100, 300, 1000, 2500)
OUTER_FOLDS = (3, 4)
REPEATS = range(4)
EXPERTS = ("NW50", "GP_XY_M32", "RT_PRIOR", "RT_GP", "GEOMETRY_KRR", "TREND_ONLY",
           "TREND_GP", "RT_TREND_ONLY", "RT_TREND_GP", "RT_TREND_SHRUNK_GP")
SCHEMES = ("FIXED", "MATCHED")
META_BASE = ("SELECT_MAE", "SELECT_RMSE", "LOCAL_RISK_MAE", "LOCAL_RISK_RMSE")
META = tuple(f"{scheme}_{name}" for scheme in SCHEMES for name in META_BASE)
METHODS = EXPERTS + META
OUTER_SEEDS = {"n41": 180041, "n79": 180079}
RUNTIME_FILES = (
    "run_h18_distance_trend.py", "h18_validation.py", "h18_models.py",
    "h17_local_models.py", "h15_models.py", "h14_models.py",
    "run_h15_physical_neural.py", "run_h11_sparse_learning_curves.py",
    "run_h13_statistical_revalidation.py", "run_h8_combined_screen.py",
    "run_h8_geospatial_models.py", "run_h10_physics_gaussian_hybrid.py",
    "run_h13_semantic_ablation.py", "run_h8_dual_source_router.py",
    "run_h13_queue.py",
    "score_h7_pointwise.py", "trace_h7_pointwise_rt.py",
    "trace_h13_corrected_search_cache.py",
)
MECHANISMS = (
    ("TREND_GP", "GP_XY_M32", "TREND_GP_minus_GP_XY_M32"),
    ("RT_TREND_GP", "RT_GP", "RT_TREND_GP_minus_RT_GP"),
    ("RT_TREND_SHRUNK_GP", "RT_TREND_GP", "RT_TREND_SHRUNK_GP_minus_RT_TREND_GP"),
)


def _validation_api():
    from .h18_validation import combine, validation_designs
    return validation_designs, combine


def _model_api():
    from . import h18_models
    if tuple(h18_models.EXPERTS) != EXPERTS:
        raise RuntimeError("h18_models expert order differs from the frozen H18 protocol")
    return h18_models.fit_pool


def runtime_hashes() -> dict[str, str]:
    source = Path(__file__).parent
    missing = [name for name in RUNTIME_FILES if not (source / name).is_file()]
    if missing:
        raise RuntimeError(f"missing H18 runtime dependencies: {missing}")
    return {name: sha256_file(source / name) for name in RUNTIME_FILES}


def _acquire_process_lock(path: Path):
    """Acquire the original non-blocking run lock on POSIX and Windows."""
    stream = path.open("a+b")
    if os.name == "nt":
        import msvcrt

        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            stream.close()
            raise RuntimeError(f"run is already locked: {path}") from None
    else:
        import fcntl

        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.close()
            raise RuntimeError(f"run is already locked: {path}") from None
    return stream


def _run_id(outer_fold: int, band: str, mode: str, budget: int, repeat: int) -> str:
    return f"f{outer_fold}_{band}_{mode}_b{budget}_r{repeat:02d}"


def _expected_ids() -> set[str]:
    return {_run_id(fold, band, mode, budget, repeat) for fold in OUTER_FOLDS
            for band in BANDS for mode in MODES for budget in BUDGETS for repeat in REPEATS}


def child_root(root: Path, outer_fold: int) -> Path:
    return root / f"fold{outer_fold}"


def validate_manifest(manifest: dict, *, require_all_eligible: bool = False) -> None:
    records = list(manifest.get("runs", ())) + list(manifest.get("ineligible", ()))
    ids = [str(record["id"]) for record in records]
    if manifest.get("schema") != "h18-v1" or len(ids) != len(set(ids)) or set(ids) != _expected_ids():
        raise RuntimeError("incomplete or changed H18 240-run matrix")
    if tuple(manifest.get("methods", ())) != METHODS or manifest.get("outer_folds") != list(OUTER_FOLDS):
        raise RuntimeError("changed H18 method or outer-fold contract")
    if int(manifest.get("repeats", -1)) != 4 or tuple(manifest.get("budgets", ())) != BUDGETS:
        raise RuntimeError("changed H18 repeat or budget contract")
    if not isinstance(manifest.get("runtime_sha256"), dict):
        raise RuntimeError("missing H18 explicit runtime hashes")
    if len(str(manifest.get("protocol_sha256", ""))) != 64:
        raise RuntimeError("missing H18 frozen protocol hash")
    child_hashes = manifest.get("child_manifest_sha256", {})
    if set(child_hashes) != {"fold3", "fold4"} or not all(len(str(value)) == 64 for value in child_hashes.values()):
        raise RuntimeError("missing H18 child-manifest hashes")
    if require_all_eligible and manifest.get("ineligible"):
        raise RuntimeError("INELIGIBLE_BUDGET present; full H18 plan is blocked")


def validate_child_manifests(root: Path, manifest: dict) -> None:
    for outer_fold in OUTER_FOLDS:
        path = child_root(root, outer_fold) / "manifest.json"
        expected = manifest["child_manifest_sha256"][f"fold{outer_fold}"]
        if sha256_file(path) != expected:
            raise RuntimeError("H18 child manifest changed")


def make_outer_splits(xy: np.ndarray, band: str, outer_fold: int) -> dict:
    groups = io.position_groups(xy)
    result = {}
    for mode in MODES:
        cell = 1.0 if mode == "infill" else 50.0
        labels = balanced_spatial_fold_labels(
            xy, n_folds=5, cell_size_m=cell, seed=OUTER_SEEDS[band])
        test = labels == outer_fold
        distance = cKDTree(xy[test]).query(xy, workers=1)[0]
        buffer = 0.0 if mode == "infill" else float(mode.replace("spatial", ""))
        pool = (~test) & (distance >= buffer)
        if np.intersect1d(groups[test], groups[pool]).size:
            raise RuntimeError("H18 position group crosses outer train/test")
        result[mode] = (np.flatnonzero(pool), np.flatnonzero(test), buffer)
    if not np.array_equal(result["spatial30"][1], result["spatial60"][1]):
        raise RuntimeError("H18 spatial30/spatial60 outer support differs")
    return result


def _h18_prepare_seed(outer_fold: int, _base: int, band: str, mode: str, repeat: int,
                      *tail: object) -> int:
    if tail == ("H15-order",):
        return derive_seed(20260905, "H18", band, outer_fold, mode, repeat, "order")
    if len(tail) == 2 and tail[1] == "H15-fit":
        return derive_seed(20260905, "H18", band, outer_fold, mode, repeat, int(tail[0]), "fit")
    raise RuntimeError(f"unexpected inherited prepare seed request: {tail}")


def _rewrite_child(child: Path, outer_fold: int) -> dict:
    manifest_path = child / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for collection in ("runs", "ineligible"):
        for spec in manifest[collection]:
            old_id = str(spec["id"])
            new_id = f"f{outer_fold}_{old_id}"
            spec["id"] = new_id
            spec["outer_fold"] = outer_fold
            if collection == "runs":
                old_dir = child / "inputs" / "runs" / old_id
                new_dir = child / "inputs" / "runs" / new_id
                old_dir.rename(new_dir)
                spec_path = new_dir / "spec.json"
                disk_spec = json.loads(spec_path.read_text(encoding="utf-8"))
                disk_spec["id"] = new_id
                disk_spec["outer_fold"] = outer_fold
                atomic_json(spec_path, disk_spec)
    manifest.update(schema="h18-child-v1", outer_fold=outer_fold, repeats=4,
                    methods=list(METHODS), budgets=list(BUDGETS))
    atomic_json(manifest_path, manifest)
    return manifest


def prepare(args) -> None:
    root = args.output_dir
    if root.exists():
        raise RuntimeError("H18 prepare requires a fresh campaign output directory")
    root.mkdir(parents=True)
    child_manifests = {}
    for outer_fold in OUTER_FOLDS:
        old = (io.make_splits, io.REPEATS, io.derive_seed, io.BUDGETS)
        try:
            io.make_splits = lambda xy, band, fold=outer_fold: make_outer_splits(xy, band, fold)
            io.REPEATS = REPEATS
            io.BUDGETS = BUDGETS
            io.derive_seed = lambda base, band, mode, repeat, *tail, fold=outer_fold: (
                _h18_prepare_seed(fold, base, band, mode, repeat, *tail))
            local_args = argparse.Namespace(**vars(args))
            local_args.output_dir = child_root(root, outer_fold)
            io.prepare(local_args)
        finally:
            io.make_splits, io.REPEATS, io.derive_seed, io.BUDGETS = old
        child_manifests[outer_fold] = _rewrite_child(child_root(root, outer_fold), outer_fold)

    runs = [spec for fold in OUTER_FOLDS for spec in child_manifests[fold]["runs"]]
    ineligible = [spec for fold in OUTER_FOLDS for spec in child_manifests[fold]["ineligible"]]
    protocol_hash = sha256_file(args.protocol)
    manifest = {
        "schema": "h18-v1", "status": "PREPARED", "bands": list(BANDS),
        "modes": list(MODES), "budgets": list(BUDGETS), "repeats": 4,
        "outer_folds": list(OUTER_FOLDS), "methods": list(METHODS),
        "experts": list(EXPERTS), "meta": list(META), "runs": runs,
        "ineligible": ineligible, "protocol_sha256": protocol_hash,
        "runtime_sha256": runtime_hashes(),
        "child_manifest_sha256": {f"fold{fold}": sha256_file(child_root(root, fold) / "manifest.json")
                                  for fold in OUTER_FOLDS},
        "scope": "exploratory_previously_exposed_single_campus_two_outer_folds",
    }
    atomic_json(root / "manifest.json", manifest)
    validate_manifest(manifest)
    validation_designs, _ = _validation_api()
    preflight = []
    for spec in runs:
        fold = int(spec["outer_fold"])
        data, frozen, train, query = io.load_fit_input(child_root(root, fold), spec["id"])
        designs = validation_designs(data.points[["x", "y"]].to_numpy(float)[train],
                                     data.points[["x", "y"]].to_numpy(float)[query],
                                     spec["mode"], int(spec["seed"]))
        preflight.append({"id": spec["id"], "schemes": {
            scheme: {"feasible": bool(designs[scheme]["feasible"]),
                     "audit": designs[scheme]["audit"]} for scheme in SCHEMES}})
    atomic_json(root / "preflight.json", {"schema": "h18-preflight-v1", "runs": preflight})
    print(json.dumps({"prepared_runs": len(runs), "ineligible": len(ineligible)}), flush=True)


def _same_partitions(left: list, right: list) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if int(a[0]) != int(b[0]) or not np.array_equal(a[1], b[1]) or not np.array_equal(a[2], b[2]):
            return False
    return True


def check_complete(directory: Path, contract: dict) -> bool:
    contract_path = directory / "run_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise RuntimeError("changed H18 run contract")
    status_path = directory / "status.json"
    if not status_path.exists():
        return False
    validate_completed_run(directory, contract)
    return True


def validate_completed_run(directory: Path, expected_contract: dict) -> dict:
    """Validate the complete, hashed H18 artifact contract before reuse or scoring."""
    status_path = directory / "status.json"
    if not status_path.is_file():
        raise RuntimeError("completed H18 status missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE" or status.get("contract") != expected_contract:
        raise RuntimeError("changed completed H18 status/spec/quick contract")
    if status.get("methods") != list(METHODS):
        raise RuntimeError("completed H18 method contract changed")
    eligibility = status.get("scheme_eligible")
    if not isinstance(eligibility, dict) or set(eligibility) != set(SCHEMES):
        raise RuntimeError("completed H18 scheme eligibility keys changed")
    if any(type(eligibility[scheme]) is not bool for scheme in SCHEMES):
        raise RuntimeError("completed H18 scheme eligibility must be boolean")
    artifacts = status.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("completed H18 artifact hash map missing")
    required = {"predictions.csv.gz", "parameters.json", "run_contract.json",
                "validation_fixed.json", "validation_matched.json"}
    for scheme in SCHEMES:
        prefix = scheme.lower()
        audit_name = f"validation_{prefix}.json"
        audit_path = directory / audit_name
        if audit_name not in artifacts or not audit_path.is_file():
            raise RuntimeError(f"completed H18 {scheme} validation audit missing")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_status = "FEASIBLE" if eligibility[scheme] else "NOT_FITTED_INNER_INFEASIBLE"
        if audit.get("scheme") != scheme or audit.get("status") != expected_status:
            raise RuntimeError(f"completed H18 {scheme} audit/feasibility mismatch")
        if "feasible" in audit and type(audit["feasible"]) is not bool:
            raise RuntimeError(f"completed H18 {scheme} audit feasible field is not boolean")
        if "feasible" in audit and audit["feasible"] != eligibility[scheme]:
            raise RuntimeError(f"completed H18 {scheme} audit feasible field disagrees")
        if eligibility[scheme]:
            required.update({f"{prefix}_inner_oof.csv", f"{prefix}_inner_risk.npz",
                             f"{prefix}_routing_weights.npz"})
            required.update(f"{prefix}_inner_{fold}_ids.csv" for fold in range(3))
    missing_hashes = required - set(artifacts)
    if missing_hashes:
        raise RuntimeError(f"completed H18 required artifact hashes missing: {sorted(missing_hashes)}")
    for name, digest in artifacts.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"completed H18 artifact changed or missing: {name}")
    run_contract = json.loads((directory / "run_contract.json").read_text(encoding="utf-8"))
    if run_contract != expected_contract:
        raise RuntimeError("completed H18 run_contract.json differs from status contract")
    return status


def _progress(output: Path, started: float, phase: str, **fields: object) -> None:
    """Write label-free progress suitable for ETA polling and stalled-run diagnosis."""
    record = {"phase": phase, "elapsed_seconds": time.time() - started, **fields}
    atomic_json(output / "progress.json", record)
    print(json.dumps(record, separators=(",", ":")), flush=True)


def run(args) -> None:
    start = time.time()
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    validate_child_manifests(args.root, manifest)
    if manifest["runtime_sha256"] != runtime_hashes():
        raise RuntimeError("H18 runtime changed; use a fresh release and result root")
    spec = next((item for item in manifest["runs"] if item["id"] == args.key), None)
    if spec is None:
        raise RuntimeError("unknown or ineligible H18 run id")
    fold = int(spec["outer_fold"])
    data, child_spec, train, query = io.load_fit_input(child_root(args.root, fold), args.key)
    if child_spec != spec:
        raise RuntimeError("campaign/child run spec mismatch")
    output = args.output_dir or args.root / "runs" / args.key
    output.mkdir(parents=True, exist_ok=True)
    lock = _acquire_process_lock(output / "run.lock")
    contract = {"spec": spec, "quick": bool(args.quick),
                "manifest_sha256": sha256_file(args.root / "manifest.json")}
    if check_complete(output, contract):
        print(json.dumps({"id": args.key, "status": "VERIFIED_COMPLETE_SKIP"}))
        return
    atomic_json(output / "run_contract.json", contract)
    xy = data.points[["x", "y"]].to_numpy(float)
    y = data.points.observed_dbm.to_numpy(float)
    validation_designs, combine = _validation_api()
    fit_pool = _model_api()
    _progress(output, start, "VALIDATION_DESIGN", id=args.key)
    designs = validation_designs(xy[train], xy[query], spec["mode"], int(spec["seed"]))
    for scheme in SCHEMES:
        atomic_json(output / f"validation_{scheme.lower()}.json", designs[scheme]["audit"])
    oof_cache = {}
    scheme_results = {}
    for scheme in SCHEMES:
        design = designs[scheme]
        predictions = {}
        details = {"status": "NOT_FITTED_INNER_INFEASIBLE", "fallback": "NW50"}
        weights = {}
        if design["feasible"]:
            partitions = design["partitions"]
            reused = next((name for name, value in oof_cache.items()
                           if _same_partitions(partitions, value["partitions"])), None)
            if reused is None:
                oof = np.full((len(train), len(EXPERTS)), np.nan)
                meta_features = np.full((len(train), 4), np.nan)
                fold_ids = np.full(len(train), -1, dtype=int)
                inner_info = []
                for fold_id, fit_local, valid_local in partitions:
                    seed = derive_seed(int(spec["seed"]), "H18-inner-fit", int(fold_id))
                    _progress(output, start, "INNER_FIT_START", id=args.key,
                              scheme=scheme, fold=int(fold_id))
                    matrix, info, has_path = fit_pool(data, train[fit_local], train[valid_local], seed,
                                                      args.device, args.quick)
                    if matrix.shape != (len(valid_local), len(EXPERTS)) or not np.isfinite(matrix).all():
                        raise RuntimeError("invalid H18 inner expert matrix")
                    oof[valid_local] = matrix
                    fold_ids[valid_local] = int(fold_id)
                    meta_features[valid_local] = risk_features(xy[train[fit_local]], xy[train[valid_local]],
                                                               matrix, has_path)
                    atomic_csv(output / f"{scheme.lower()}_inner_{fold_id}_ids.csv",
                               pd.DataFrame({"point_id": data.points.point_id.to_numpy()[train],
                                             "fit": np.isin(np.arange(len(train)), fit_local),
                                             "valid": np.isin(np.arange(len(train)), valid_local)}))
                    inner_info.append(info)
                    _progress(output, start, "INNER_FIT_COMPLETE", id=args.key,
                              scheme=scheme, fold=int(fold_id))
                if (fold_ids < 0).any() or not np.isfinite(oof).all() or not np.isfinite(meta_features).all():
                    raise RuntimeError("incomplete H18 OOF predictions")
                cached = {"partitions": partitions, "oof": oof, "features": meta_features,
                          "folds": fold_ids, "inner": inner_info}
                oof_cache[scheme] = cached
                atomic_csv(output / f"{scheme.lower()}_inner_oof.csv",
                           pd.DataFrame(oof, columns=EXPERTS).assign(
                               point_id=data.points.point_id.to_numpy()[train],
                               observed_dbm=y[train], fold=fold_ids))
                np.savez_compressed(output / f"{scheme.lower()}_inner_risk.npz",
                                    features=meta_features, errors=oof - y[train, None], folds=fold_ids)
            else:
                cached = oof_cache[reused]
                oof_cache[scheme] = cached
                atomic_json(output / f"{scheme.lower()}_oof_reuse.json",
                            {"identical_to": reused, "exact_partition_equality": True})
                atomic_csv(output / f"{scheme.lower()}_inner_oof.csv",
                           pd.DataFrame(cached["oof"], columns=EXPERTS).assign(
                               point_id=data.points.point_id.to_numpy()[train],
                               observed_dbm=y[train], fold=cached["folds"]))
                np.savez_compressed(output / f"{scheme.lower()}_inner_risk.npz",
                                    features=cached["features"], errors=cached["oof"] - y[train, None],
                                    folds=cached["folds"])
                for fold_id, fit_local, valid_local in partitions:
                    atomic_csv(output / f"{scheme.lower()}_inner_{fold_id}_ids.csv",
                               pd.DataFrame({"point_id": data.points.point_id.to_numpy()[train],
                                             "fit": np.isin(np.arange(len(train)), fit_local),
                                             "valid": np.isin(np.arange(len(train)), valid_local)}))
            scheme_results[scheme] = {"eligible": True, "cached": cached, "details": details,
                                      "predictions": predictions, "weights": weights}
        else:
            scheme_results[scheme] = {"eligible": False, "details": details,
                                      "predictions": predictions, "weights": weights}

    _progress(output, start, "FINAL_FIT", id=args.key)
    expert_matrix, final_info, query_path = fit_pool(data, train, query, int(spec["seed"]),
                                                     args.device, args.quick)
    if expert_matrix.shape != (len(query), len(EXPERTS)) or not np.isfinite(expert_matrix).all():
        raise RuntimeError("invalid H18 final expert matrix")
    all_predictions = {name: expert_matrix[:, index] for index, name in enumerate(EXPERTS)}
    for scheme in SCHEMES:
        result = scheme_results[scheme]
        _progress(output, start, "ROUTING", id=args.key, scheme=scheme,
                  feasible=bool(result["eligible"]))
        if result["eligible"]:
            cached = result["cached"]
            query_features = risk_features(xy[train], xy[query], expert_matrix, query_path)
            extra, details, weights = combine(cached["oof"], y[train], cached["features"],
                                              query_features, expert_matrix)
            if set(extra) != set(META_BASE):
                raise RuntimeError("H18 combine output contract changed")
            result.update(predictions=extra, details=details, weights=weights)
            np.savez_compressed(output / f"{scheme.lower()}_routing_weights.npz", **weights)
        else:
            result["predictions"] = {name: expert_matrix[:, 0].copy() for name in META_BASE}
        all_predictions.update({f"{scheme}_{name}": values for name, values in result["predictions"].items()})
    if set(all_predictions) != set(METHODS):
        raise RuntimeError("H18 18-output contract changed")
    frame = data.points.iloc[query][["point_id", "x", "y", "date", "trajectory_group"]].reset_index(drop=True)
    frame["has_path"] = query_path
    frame["nearest_train_m"] = cKDTree(xy[train]).query(xy[query], workers=1)[0]
    for name in METHODS:
        values = np.asarray(all_predictions[name], float)
        if values.shape != (len(query),) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid H18 prediction {name}")
        frame[name] = values
    frame.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    atomic_json(output / "parameters.json", {"final": final_info,
                "schemes": {scheme: {"eligible": value["eligible"], "details": value["details"],
                                      "inner_models": value.get("cached", {}).get("inner", [])}
                            for scheme, value in scheme_results.items()}})
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name not in ("status.json", "run.lock", "progress.json")}
    status = {"status": "COMPLETE", "contract": contract, "methods": list(METHODS),
              "scheme_eligible": {scheme: bool(scheme_results[scheme]["eligible"]) for scheme in SCHEMES},
              "artifacts": artifacts, "elapsed_seconds": time.time() - start}
    atomic_json(output / "status.json", status)
    validate_completed_run(output, contract)
    _progress(output, start, "COMPLETE", id=args.key)
    print(json.dumps({"id": args.key, "status": "COMPLETE", "elapsed_seconds": status["elapsed_seconds"]}),
          flush=True)


def is_smoke_spec(spec: dict) -> bool:
    return (int(spec["repeat"]) == 0 and spec["mode"] in ("infill", "spatial60")
            and int(spec["budget"]) in (30, 2500))


def verify_smoke(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, require_all_eligible=True)
    selected = {spec["id"] for spec in manifest["runs"] if is_smoke_spec(spec)}
    expected = {_run_id(fold, band, mode, budget, 0) for fold in OUTER_FOLDS for band in BANDS
                for mode in ("infill", "spatial60") for budget in (30, 2500)}
    if selected != expected:
        raise RuntimeError("missing required sixteen H18 smoke jobs")
    for spec in manifest["runs"]:
        if not is_smoke_spec(spec):
            continue
        path = root / "smoke" / spec["id"]
        expected_contract = {"spec": spec, "quick": False,
                             "manifest_sha256": sha256_file(root / "manifest.json")}
        validate_completed_run(path, expected_contract)


def plan(args) -> None:
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, require_all_eligible=True)
    validate_child_manifests(args.root, manifest)
    if manifest["runtime_sha256"] != runtime_hashes():
        raise RuntimeError("H18 runtime changed; refusing to emit a mixed-release plan")
    specs = [spec for spec in manifest["runs"] if not args.smoke or is_smoke_spec(spec)]
    jobs = []
    for spec in sorted(specs, key=lambda item: (-int(item["budget"]), int(item["repeat"]),
                                                int(item["outer_fold"]), item["band"], item["mode"])):
        command = [str(args.python), "-m", "src.radio.legacy.run_h18_distance_trend", "run",
                   "--root", str(args.root), "--key", spec["id"], "--device", "cuda"]
        if args.smoke:
            command += ["--output-dir", str(args.root / "smoke" / spec["id"])]
        jobs.append({"id": f"fit_{spec['id']}", "command": command, "cwd": str(args.release),
                     "kind": "gpu", "min_free_mib": 12000, "timeout_seconds": 7200,
                     "depends": [], "eta_group": f"b{spec['budget']}"})
    if not args.smoke:
        verify_smoke(args.root)
        jobs.append({"id": "score_h18", "command": [str(args.python), "-m",
                     "src.radio.legacy.run_h18_distance_trend", "score", "--root", str(args.root)],
                     "cwd": str(args.release), "kind": "cpu", "timeout_seconds": 7200,
                     "depends": [job["id"] for job in jobs], "eta_group": "scoring"})
    atomic_json(args.output_dir, {"schema": "h18-queue-v1", "cpu_workers": 1,
                "gpu_candidates": [1, 2, 3], "jobs": jobs,
                "manifest_sha256": sha256_file(args.root / "manifest.json"), "smoke": bool(args.smoke)})
    print(json.dumps({"jobs": len(jobs)}))


def _metric_row(error: np.ndarray) -> dict:
    absolute = np.abs(error)
    return {"mae_db": float(absolute.mean()), "rmse_db": float(np.sqrt(np.mean(error ** 2))),
            "medae_db": float(np.median(absolute)), "p90_db": float(np.quantile(absolute, .9)),
            "bias_db": float(error.mean())}


def _paired(metrics: pd.DataFrame, comparisons: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
    keys = ["band", "outer_fold", "mode", "budget", "repeat"]
    fields = ["mae_db", "rmse_db", "medae_db", "p90_db", "bias_db"]
    pieces = []
    for left, right, name in comparisons:
        a = metrics[metrics.method == left][keys + fields]
        b = metrics[metrics.method == right][keys + fields]
        joined = a.merge(b, on=keys, suffixes=("_left", "_right"), validate="one_to_one")
        result = joined[keys].copy(); result["comparison"] = name
        for field in fields:
            result[f"delta_{field}"] = joined[f"{field}_left"] - joined[f"{field}_right"]
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def score(args) -> None:
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest, require_all_eligible=True)
    validate_child_manifests(args.root, manifest)
    if manifest["runtime_sha256"] != runtime_hashes():
        raise RuntimeError("H18 runtime changed; refusing mixed-release scoring")
    frozen = []; support = {}
    # Hard gate: finish every contract/hash/support check before opening any scorer truth.
    for spec in manifest["runs"]:
        path = args.root / "runs" / spec["id"]
        contract = {"spec": spec, "quick": False,
                    "manifest_sha256": sha256_file(args.root / "manifest.json")}
        status = validate_completed_run(path, contract)
        frame = pd.read_csv(path / "predictions.csv.gz")
        ids = frame.point_id.to_numpy()
        key = (int(spec["outer_fold"]), spec["band"], spec["mode"])
        if not frame.point_id.is_unique or len(ids) != int(spec["test_n"]):
            raise RuntimeError("H18 test denominator changed")
        if not np.isfinite(frame[list(METHODS)].to_numpy(float)).all():
            raise RuntimeError("H18 nonfinite frozen prediction")
        if key in support and not np.array_equal(ids, support[key]):
            raise RuntimeError("H18 shared test support changed")
        child = child_root(args.root, int(spec["outer_fold"]))
        expected = np.load(child / "inputs" / spec["band"] / f"test_{spec['mode']}.npy", allow_pickle=False)
        features = pd.read_csv(child / "inputs" / spec["band"] / "features.csv")
        if not np.array_equal(ids, features.point_id.to_numpy()[expected]):
            raise RuntimeError("H18 test IDs changed")
        support[key] = ids
        design_audits = {scheme: json.loads(
            (path / f"validation_{scheme.lower()}.json").read_text(encoding="utf-8"))
                         for scheme in SCHEMES}
        frozen.append((spec, status, frame, design_audits))

    truths = {}
    for outer_fold in OUTER_FOLDS:
        child = child_root(args.root, outer_fold)
        child_manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
        for band in BANDS:
            truth_path = child / "scorer_truth" / f"{band}.csv"
            if sha256_file(truth_path) != child_manifest["sources"][band]["scorer_truth_sha256"]:
                raise RuntimeError("H18 scorer truth changed")
            truths[(outer_fold, band)] = pd.read_csv(truth_path).set_index("point_id").observed_dbm
    rows = []
    design_rows = []
    for spec, status, frame, design_audits in frozen:
        observed = truths[(int(spec["outer_fold"]), spec["band"])].loc[frame.point_id].to_numpy(float)
        for scheme in SCHEMES:
            audit = design_audits[scheme]
            design_rows.append({key: spec[key] for key in ("band", "outer_fold", "mode", "budget", "repeat")} | {
                "scheme": scheme, "feasible": bool(status["scheme_eligible"][scheme]),
                "selected_candidate_id": audit.get("selected_candidate_id"),
                "selected_matching_loss": audit.get("selected_matching_loss"),
                "raw_wasserstein_m": next((candidate.get("raw_wasserstein_m")
                    for candidate in audit.get("candidates", [])
                    if candidate.get("candidate_id") == audit.get("selected_candidate_id")), None)})
        for method in METHODS:
            scheme = method.split("_", 1)[0] if method.startswith(SCHEMES) else None
            row = {key: spec[key] for key in ("band", "outer_fold", "mode", "budget", "repeat")}
            row.update(method=method, n=len(observed),
                       scheme_eligible=True if scheme is None else bool(status["scheme_eligible"][scheme]),
                       elapsed_seconds=float(status["elapsed_seconds"]))
            row.update(_metric_row(frame[method].to_numpy(float) - observed))
            rows.append(row)
    metrics = pd.DataFrame(rows)
    output = args.root / "report"; output.mkdir(exist_ok=True)
    atomic_csv(output / "repeat_metrics.csv", metrics)
    atomic_csv(output / "validation_design_summary.csv", pd.DataFrame(design_rows))
    summary = metrics.groupby(["band", "outer_fold", "mode", "budget", "method"]).agg(
        repeats=("repeat", "nunique"), n=("n", "first"),
        mae_mean=("mae_db", "mean"), mae_sd=("mae_db", "std"),
        rmse_mean=("rmse_db", "mean"), rmse_sd=("rmse_db", "std"),
        medae_mean=("medae_db", "mean"), p90_mean=("p90_db", "mean"),
        bias_mean=("bias_db", "mean")).reset_index()
    atomic_csv(output / "summary.csv", summary)
    mechanisms = _paired(metrics, MECHANISMS)
    atomic_csv(output / "paired_mechanisms.csv", mechanisms)
    design_pairs = tuple((f"MATCHED_{name}", f"FIXED_{name}", f"{name}_MATCHED_minus_FIXED")
                         for name in META_BASE)
    common_keys = metrics[(metrics.method == "FIXED_SELECT_MAE") & metrics.scheme_eligible][
        ["band", "outer_fold", "mode", "budget", "repeat"]].merge(
            metrics[(metrics.method == "MATCHED_SELECT_MAE") & metrics.scheme_eligible][
                ["band", "outer_fold", "mode", "budget", "repeat"]],
            on=["band", "outer_fold", "mode", "budget", "repeat"])
    common = metrics.merge(common_keys, on=["band", "outer_fold", "mode", "budget", "repeat"])
    atomic_csv(output / "paired_matched_vs_fixed_common_feasible.csv", _paired(common, design_pairs))
    scheme_rows = metrics[metrics.method.isin(("FIXED_SELECT_MAE", "MATCHED_SELECT_MAE"))].copy()
    scheme_rows["scheme"] = scheme_rows.method.str.split("_", n=1).str[0]
    coverage = scheme_rows.groupby(["band", "outer_fold", "mode", "budget", "scheme"]).agg(
        total_repeats=("repeat", "nunique"), feasible_repeats=("scheme_eligible", "sum")).reset_index()
    coverage["fallback_repeats"] = coverage.total_repeats - coverage.feasible_repeats
    atomic_csv(output / "scheme_coverage.csv", coverage)
    budget = summary.groupby(["band", "budget", "method"]).agg(
        conditions=("mode", "size"), macro_mae=("mae_mean", "mean"),
        worst_mae=("mae_mean", "max"), macro_rmse=("rmse_mean", "mean"),
        worst_rmse=("rmse_mean", "max")).reset_index()
    atomic_csv(output / "budget_multicondition_summary.csv", budget)
    target = metrics.groupby(["band", "outer_fold", "mode", "budget", "method"]).agg(
        mean_mae_db=("mae_db", "mean"), worst_repeat_mae_db=("mae_db", "max"),
        hit_fraction=("mae_db", lambda values: float((values <= 4.0).mean()))).reset_index()
    target["mean_gap_to_4db"] = target.mean_mae_db - 4.0
    target["worst_repeat_gap_to_4db"] = target.worst_repeat_mae_db - 4.0
    atomic_csv(output / "target4db.csv", target)
    minimum_rows = []
    for keys, group in target.groupby(["band", "outer_fold", "mode", "method"]):
        hits = group[group.mean_mae_db <= 4.0].sort_values("budget")
        minimum_rows.append(dict(zip(("band", "outer_fold", "mode", "method"), keys)) | {
            "minimum_observed_budget_mean_le_4db": None if hits.empty else int(hits.iloc[0].budget),
            "best_observed_mean_mae_db": float(group.mean_mae_db.min()),
            "best_observed_gap_to_4db": float(group.mean_gap_to_4db.min())})
    atomic_csv(output / "minimum_budget_overview.csv", pd.DataFrame(minimum_rows))
    atomic_json(output / "audit.json", {"status": "PASS", "schema": "h18-score-v1",
                "runs": len(frozen), "methods": len(METHODS), "metric_rows": len(metrics),
                "truth_access": "after_all_240_prediction_contracts_frozen",
                "scope": manifest["scope"]})
    print(json.dumps({"status": "PASS", "runs": len(frozen), "metric_rows": len(metrics)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "plan", "score"))
    for flag in ("root", "output-dir", "data-root", "rt-root", "release", "python", "protocol"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--key")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
