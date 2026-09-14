#!/usr/bin/env python3
"""Fresh OSM-aligned semantic-vs-uniform H18 factorial campaign.

The 240 frozen H18 split specifications are reused.  RT is recomputed once for
each (band, material arm) over the complete aligned receiver geometry and is
then consumed by every matching split.  Every inner fold fits its own
training-only supported-power calibration through the patched H18 model API.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType

import numpy as np
from scipy.optimize import least_squares


ARMS = ("uniform_nonconductor_concrete", "semantic_8class")
BANDS = ("n41", "n79")
EXPERTS = ("NW50", "GP_XY_M32", "RT_PRIOR", "RT_GP", "GEOMETRY_KRR",
           "TREND_ONLY", "TREND_GP", "RT_TREND_ONLY", "RT_TREND_GP",
           "RT_TREND_SHRUNK_GP")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_NAME = "src.radio.legacy.aligned_factorial"


def _module_command(python: Path, command: str) -> list[str]:
    return [str(python), "-m", MODULE_NAME, command]


def _lock_stream(stream) -> None:
    """Hold an exclusive lock for the lifetime of ``stream`` on any platform."""
    if os.name == "nt":
        import msvcrt

        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write("0")
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(stream, fcntl.LOCK_EX)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     delete=False, suffix=".tmp") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def inverse_alignment(old_xyz: np.ndarray, transform: dict) -> np.ndarray:
    old_xyz = np.asarray(old_xyz, float)
    enu = np.column_stack((-old_xyz[:, 0], 100.0 - old_xyz[:, 1]))
    best = transform["best_candidate"]
    if int(best["mirror"]) != 1:
        raise RuntimeError("selected OSM transform is not the proper transform")
    angle = np.deg2rad(float(best["angle_deg"]))
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
    model_xy = (enu - [best["tx_m"], best["ty_m"]]) @ rotation / float(best["scale"])
    return np.column_stack((model_xy, old_xyz[:, 2]))


def prepare(args) -> None:
    import pandas as pd
    root = args.job_root
    if root.exists():
        raise RuntimeError("prepare requires a fresh job root")
    (root / "prepared").mkdir(parents=True)
    (root / "state").mkdir()
    transform = json.loads(args.transform_json.read_text(encoding="utf-8"))
    if transform.get("status") != "APPROXIMATE_VISUAL_ALIGNMENT__NOT_SURVEYED_GEOREFERENCE":
        raise RuntimeError("unexpected alignment status")
    source_manifest = json.loads((args.results_root / "manifest.json").read_text())
    specs = source_manifest["runs"]
    if len(specs) != 240 or len({item["id"] for item in specs}) != 240:
        raise RuntimeError("frozen H18 240-run contract changed")
    source_hashes = {}
    for band in BANDS:
        frames = []
        for fold in (3, 4):
            path = args.results_root / f"fold{fold}" / "inputs" / band / "features.csv"
            frame = pd.read_csv(path)
            frames.append(frame)
            source_hashes[f"fold{fold}_{band}_features"] = sha256_file(path)
        columns = ["point_id", "x", "y", "z", "date", "trajectory_group", "position_group"]
        if not frames[0][columns].equals(frames[1][columns]):
            raise RuntimeError(f"fold geometry differs for {band}")
        aligned = inverse_alignment(frames[0][["x", "y", "z"]].to_numpy(float), transform)
        unique_xyz, mapping = np.unique(aligned, axis=0, return_inverse=True)
        np.savez_compressed(root / "prepared" / f"{band}.npz", band=np.asarray(band),
                            unique_xyz=unique_xyz, point_to_unique=mapping,
                            aligned_xyz=aligned,
                            point_id=np.asarray(frames[0].point_id.astype(str).tolist(), dtype="U"))
    protocol = {
        "schema": "h18-osm-aligned-material-power-v2-factorial-v1",
        "status": "PREPARED",
        "claim_boundary": "approximate OSM shape registration; not surveyed georeferencing",
        "matrix": {"bands": list(BANDS), "spatial_modes": ["infill", "spatial30", "spatial60"],
                   "budgets": [30, 100, 300, 1000, 2500], "outer_folds": [3, 4],
                   "repeats": 4, "tasks": 240, "material_arms_per_task": list(ARMS)},
        "rt_contract": "fresh solve per band and material arm; identical aligned geometry and ray seed",
        "fit_contract": "each arm and each inner fold independently fits supported power_v2 using fit labels only",
        "fallback_contract": "missing/unsupported RT uses same-fold NW50; no rows are filtered",
        "forbidden": ["query/test labels during fitting", "large-error filtering", "no-path filtering"],
        "source_results_manifest_sha256": sha256_file(args.results_root / "manifest.json"),
        "transform_sha256": sha256_file(args.transform_json),
        "scene_sha256": sha256_file(args.scene_xml),
        "source_hashes": source_hashes,
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
        "created_unix": time.time(),
    }
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "state" / "state.json", {"status": "PREPARED", "completed_tasks": 0,
                                                   "failed_tasks": 0, "created_unix": time.time()})
    print(json.dumps({"status": "PREPARED", "tasks": 240, "arms": 2}))


def supported_calibration(data, observed, fit_mask, family, **_kwargs):
    """Fit state-specific monotone power mapping with training-only fallback."""
    fit = np.flatnonzero(np.asarray(fit_mask, bool))
    observed = np.asarray(observed, float)
    y = observed[fit]
    if not len(fit) or not np.isfinite(y).all() or np.isfinite(observed[~np.asarray(fit_mask, bool)]).any():
        raise RuntimeError("power_v2 target isolation contract failed")
    from . import h18_models as models
    raw = np.asarray(data.gains[0], float)
    los = np.asarray(data.los, bool)
    xy = data.points[["x", "y"]].to_numpy(float)
    nw = models.predict_candidate(models.NW50, xy[fit], y, xy, seed=241301)
    prediction = nw.copy()
    parameters = {}
    for state in (False, True):
        valid = np.isfinite(raw[fit]) & (los[fit] == state)
        ids, target = fit[valid], y[valid]
        if len(ids) < 20:
            parameters[str(state)] = {"n": int(len(ids)), "status": "NW50_FALLBACK_INSUFFICIENT_SUPPORT"}
            continue
        x = raw[ids]
        center, intercept = float(np.median(x)), float(np.median(target))
        result = least_squares(lambda p: p[0] * (x - center) + p[1] - target,
                               [0.5, intercept], bounds=([0.0, -np.inf], [1.0, np.inf]),
                               loss="soft_l1", f_scale=5.0, max_nfev=1000)
        if not result.success:
            raise RuntimeError("power_v2 calibration optimization failed")
        slope, bias = result.x
        eligible = np.isfinite(raw) & (los == state)
        values = raw[eligible]
        outside = np.maximum(np.maximum(x.min() - values, values - x.max()), 0.0)
        taper = max(float(np.subtract(*np.percentile(x, [75, 25]))), 1.0)
        weight = np.exp(-np.square(outside / taper))
        calibrated = slope * (values - center) + bias
        prediction[eligible] = weight * calibrated + (1.0 - weight) * nw[eligible]
        parameters[str(state)] = {"n": int(len(ids)), "status": "FIT", "slope": float(slope),
                                  "center_gain_db": center, "center_signal_dbm": float(bias),
                                  "training_gain_range_db": [float(x.min()), float(x.max())],
                                  "support_taper_db": taper}
    if not np.isfinite(prediction).all():
        raise RuntimeError("power_v2 produced nonfinite predictions")
    return prediction, {"calibration": "training-only state-specific robust monotone affine power_v2",
                        "fallback": "same-fit NW50 for no path, insufficient state, or tapered extrapolation",
                        "minimum_state_training_n": 20, "state_models": parameters,
                        "prediction_clipping": False}


def trace(args) -> None:
    root = args.job_root
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        # The frozen RT functions do not execute pandas code; the Sionna env is
        # intentionally minimal, while the source module imports pandas only
        # for preparation/scoring helpers and deferred annotations.
        sys.modules["pandas"] = ModuleType("pandas")
    from . import run_h13_semantic_ablation as semantic
    arm = next(item for item in semantic.build_material_arms() if item["id"] == args.arm)
    with np.load(root / "prepared" / f"{args.band}.npz", allow_pickle=False) as blob:
        xyz = np.asarray(blob["unique_xyz"], float)
    out_dir = root / "rt" / args.arm / args.band
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(xyz), args.chunk_n):
        stop = min(start + args.chunk_n, len(xyz))
        output = out_dir / f"chunk_{start:05d}_{stop:05d}.npz"
        status = output.with_suffix(".json")
        contract = {"band": args.band, "arm": args.arm, "start": start, "stop": stop,
                    "seed": args.seed, "samples_per_source": args.samples_per_source,
                    "max_depth": args.max_depth, "scene_sha256": sha256_file(args.scene_xml),
                    "prepared_sha256": sha256_file(root / "prepared" / f"{args.band}.npz")}
        if output.exists() and status.exists():
            existing = json.loads(status.read_text())
            if existing.get("status") == "COMPLETE" and existing.get("contract") == contract and \
                    existing.get("output_sha256") == sha256_file(output):
                continue
            raise RuntimeError(f"invalid partial RT artifact: {output}")
        before = time.time()
        gain, los = semantic.trace_fresh_source(args.scene_xml, arm, args.band, xyz[start:stop],
                                                semantic.FIXED_TX_BY_BAND[args.band],
                                                samples_per_source=args.samples_per_source,
                                                max_depth=args.max_depth, seed=args.seed,
                                                chunk_size=args.solve_chunk_n)
        if gain.shape != (stop - start,) or los.shape != gain.shape or los.dtype != np.bool_:
            raise RuntimeError("invalid RT result shape/type")
        np.savez_compressed(output, start=start, stop=stop, xyz=xyz[start:stop],
                            raw_base_gain_db=gain, los=los)
        atomic_json(status, {"status": "COMPLETE", "contract": contract,
                             "elapsed_seconds": time.time() - before,
                             "path_fraction": float(np.isfinite(gain).mean()),
                             "los_fraction": float(los.mean()), "output_sha256": sha256_file(output)})
        print(json.dumps({"arm": args.arm, "band": args.band, "start": start, "stop": stop,
                          "elapsed_seconds": time.time() - before}), flush=True)


def smoke_trace(args) -> None:
    """Trace the same 64 receivers under both material arms before full launch."""
    try:
        import pandas  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["pandas"] = ModuleType("pandas")
    from . import run_h13_semantic_ablation as semantic
    with np.load(args.job_root / "prepared" / "n41.npz", allow_pickle=False) as blob:
        xyz = np.asarray(blob["unique_xyz"], float)[:64]
    output = args.job_root / "smoke"
    output.mkdir(exist_ok=True)
    rows = []
    for arm_id in ARMS:
        arm = next(item for item in semantic.build_material_arms() if item["id"] == arm_id)
        before = time.time()
        gain, los = semantic.trace_fresh_source(
            args.scene_xml, arm, "n41", xyz, semantic.FIXED_TX_BY_BAND["n41"],
            samples_per_source=50000, max_depth=3, seed=241301, chunk_size=64)
        if gain.shape != (64,) or los.shape != (64,) or los.dtype != np.bool_:
            raise RuntimeError("smoke RT output contract failed")
        path = output / f"{arm_id}_n41_64.npz"
        np.savez_compressed(path, xyz=xyz, raw_base_gain_db=gain, los=los)
        rows.append({"arm": arm_id, "n": 64, "finite_n": int(np.isfinite(gain).sum()),
                     "los_n": int(los.sum()), "elapsed_seconds": time.time() - before,
                     "output_sha256": sha256_file(path)})
    atomic_json(output / "status.json", {"status": "PASS", "same_geometry": True,
                "same_seed": 241301, "rows": rows})
    print(json.dumps({"status": "PASS", "rows": rows}))


def merge_rt(root: Path, arm: str, band: str, expected_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cursor, gains, los = 0, [], []
    for path in sorted((root / "rt" / arm / band).glob("chunk_*.npz")):
        status = json.loads(path.with_suffix(".json").read_text())
        if status.get("status") != "COMPLETE" or status.get("output_sha256") != sha256_file(path):
            raise RuntimeError(f"invalid RT chunk status: {path}")
        with np.load(path, allow_pickle=False) as blob:
            start, stop = int(blob["start"]), int(blob["stop"])
            if start != cursor:
                raise RuntimeError("noncontiguous RT chunks")
            np.testing.assert_array_equal(blob["xyz"], expected_xyz[start:stop])
            gains.append(np.asarray(blob["raw_base_gain_db"], float))
            los.append(np.asarray(blob["los"], bool))
            cursor = stop
    if cursor != len(expected_xyz):
        raise RuntimeError(f"incomplete RT for {arm}/{band}: {cursor}/{len(expected_xyz)}")
    return np.concatenate(gains), np.concatenate(los)


def aligned_input(args, original_loader, key: str, arm: str):
    import pandas as pd  # noqa: F401 - required by the frozen loader
    data, spec, train, query = original_loader(args.results_root / f"fold{spec_fold(key)}", key)
    band = spec["band"]
    with np.load(args.job_root / "prepared" / f"{band}.npz", allow_pickle=False) as blob:
        point_ids = np.asarray(blob["point_id"]).astype(str)
        aligned_xyz = np.asarray(blob["aligned_xyz"], float)
        unique_xyz = np.asarray(blob["unique_xyz"], float)
        mapping = np.asarray(blob["point_to_unique"], int)
    if not np.array_equal(data.points.point_id.astype(str).to_numpy(), point_ids):
        raise RuntimeError("aligned point ID order changed")
    gain, los = merge_rt(args.job_root, arm, band, unique_xyz)
    points = data.points.copy()
    points.loc[:, ["x", "y", "z"]] = aligned_xyz
    transformed = replace(data, points=points,
                          configs=[{"id": "AUTO_BASE", "family": "BASE", "groups": {}}],
                          gains=np.asarray([gain[mapping]], float), los=los[mapping],
                          source_row_index=np.arange(len(points)))
    return transformed, spec, train, query


def spec_fold(key: str) -> int:
    return int(key.split("_", 1)[0][1:])


def run_case(args) -> None:
    from . import h18_models
    from . import run_h18_distance_trend as runner
    original_loader = runner.io.load_fit_input
    h18_models.fit_physical_sparse = supported_calibration
    for arm in ARMS:
        output = args.job_root / "runs" / args.key / arm
        arm_contract = {"schema": "aligned-material-arm-v1", "key": args.key, "arm": arm,
                        "protocol_sha256": sha256_file(args.job_root / "protocol.json"),
                        "power_v2_source_sha256": sha256_file(Path(__file__).resolve())}
        contract_path = output / "arm_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != arm_contract:
            raise RuntimeError("arm contract changed")
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, arm_contract)
        runner.io.load_fit_input = lambda _root, key, chosen=arm: aligned_input(
            args, original_loader, key, chosen)
        namespace = argparse.Namespace(root=args.results_root, key=args.key, output_dir=output,
                                      device=_resolve_device(args.device), quick=False)
        runner.run(namespace)
    atomic_json(args.job_root / "runs" / args.key / "status.json",
                {"status": "COMPLETE", "key": args.key, "arms": list(ARMS),
                 "completed_unix": time.time()})


def gpu_snapshot(gpu: int) -> dict:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    values = next([int(x.strip()) for x in row.split(",")] for row in rows
                  if int(row.split(",")[0]) == gpu)
    return {"gpu": values[0], "free_mib": values[1], "total_mib": values[2], "utilization": values[3]}


def run_locked(root: Path, gpu: int, command: list[str], log: Path, min_free: int,
               timeout: int, retries: int = 2) -> None:
    lock_path = root / "state" / f"gpu{gpu}.lock"
    with lock_path.open("a+") as lock:
        _lock_stream(lock)
        while gpu_snapshot(gpu)["free_mib"] < min_free:
            time.sleep(30)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        for attempt in range(retries + 1):
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "START", "attempt": attempt,
                                         "gpu": gpu, "command": command, "time": time.time()}) + "\n")
                stream.flush()
                result = subprocess.run(command, env=environment, stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=timeout,
                                        cwd=PROJECT_ROOT)
            if result.returncode == 0:
                return
            if attempt == retries:
                raise RuntimeError(f"command failed after bounded retries: {command}")


def controller(args) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    root = args.job_root
    (root / "logs").mkdir(exist_ok=True)
    state_path = root / "state" / "state.json"
    state = json.loads(state_path.read_text())
    state.update(status="RUNNING_RT", controller_pid=os.getpid(), started_unix=time.time())
    atomic_json(state_path, state)
    trace_jobs = [(arm, band) for arm in ARMS for band in BANDS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for gpu, (arm, band) in enumerate(trace_jobs):
            command = _module_command(args.sionna_python, "trace") + [
                       "--job-root", str(root),
                       "--scene-xml", str(args.scene_xml), "--arm", arm, "--band", band]
            futures[pool.submit(run_locked, root, gpu, command,
                                root / "logs" / f"trace_{arm}_{band}.log", 4000, 7200)] = (arm, band)
        for future in as_completed(futures):
            future.result()
    state.update(status="RUNNING_240_TASKS", rt_completed_unix=time.time())
    atomic_json(state_path, state)
    manifest = json.loads((args.results_root / "manifest.json").read_text())
    smoke_spec = min(manifest["runs"], key=lambda item: (int(item["budget"]), item["id"]))
    smoke_command = _module_command(args.fit_python, "run-case") + [
                     "--job-root", str(root), "--results-root", str(args.results_root),
                     "--key", smoke_spec["id"], "--device", args.device]
    run_locked(root, 0, smoke_command, root / "logs" / "model_smoke.log", 12000, 10800)
    state.update(model_smoke_status="PASS", model_smoke_key=smoke_spec["id"],
                 model_smoke_completed_unix=time.time())
    atomic_json(state_path, state)
    specs = sorted(manifest["runs"], key=lambda item: (-int(item["budget"]), int(item["repeat"]), item["id"]))
    completed = {item["id"] for item in specs
                 if (root / "runs" / item["id"] / "status.json").exists()}
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {}
        for index, spec in enumerate(item for item in specs if item["id"] not in completed):
            gpu = index % 4
            command = _module_command(args.fit_python, "run-case") + [
                       "--job-root", str(root), "--results-root", str(args.results_root),
                       "--key", spec["id"], "--device", args.device]
            pending[pool.submit(run_locked, root, gpu, command,
                                root / "logs" / f"task_{spec['id']}.log", 12000, 10800)] = spec["id"]
        failures = []
        for future in as_completed(pending):
            key = pending[future]
            try:
                future.result(); completed.add(key)
            except Exception as error:
                failures.append({"key": key, "error": repr(error)})
            elapsed = time.time() - started
            done = len(completed)
            remaining = 240 - done
            state.update(completed_tasks=done, failed_tasks=len(failures), failures=failures,
                         elapsed_fit_seconds=elapsed,
                         eta_seconds=(elapsed / max(1, done) * remaining), updated_unix=time.time())
            atomic_json(state_path, state)
    if failures:
        state.update(status="FAILED", failures=failures, completed_tasks=len(completed))
        atomic_json(state_path, state)
        raise RuntimeError(f"{len(failures)} tasks failed")
    command = _module_command(args.fit_python, "score") + [
               "--job-root", str(root), "--results-root", str(args.results_root),
               ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    state.update(status="COMPLETE", completed_tasks=240, failed_tasks=0,
                 eta_seconds=0, completed_unix=time.time())
    atomic_json(state_path, state)


def metrics(error: np.ndarray) -> dict:
    absolute = np.abs(error)
    return {"n": int(len(error)), "mae_db": float(absolute.mean()),
            "rmse_db": float(np.sqrt(np.mean(np.square(error)))),
            "medae_db": float(np.median(absolute)), "p90_db": float(np.quantile(absolute, .9)),
            "bias_db": float(error.mean())}


def score(args) -> None:
    import pandas as pd
    from . import run_h18_distance_trend as runner
    manifest = json.loads((args.results_root / "manifest.json").read_text())
    frozen = []
    for spec in manifest["runs"]:
        for arm in ARMS:
            output = args.job_root / "runs" / spec["id"] / arm
            contract = {"spec": spec, "quick": False,
                        "manifest_sha256": sha256_file(args.results_root / "manifest.json")}
            runner.validate_completed_run(output, contract)
            frame = pd.read_csv(output / "predictions.csv.gz")
            if len(frame) != int(spec["test_n"]) or not frame.point_id.is_unique:
                raise RuntimeError("test denominator changed")
            if not np.isfinite(frame[list(runner.METHODS)].to_numpy(float)).all():
                raise RuntimeError("nonfinite prediction")
            frozen.append((spec, arm, frame))
    truths = {}
    for fold in (3, 4):
        child_manifest = json.loads((args.results_root / f"fold{fold}" / "manifest.json").read_text())
        for band in BANDS:
            path = args.results_root / f"fold{fold}" / "scorer_truth" / f"{band}.csv"
            if sha256_file(path) != child_manifest["sources"][band]["scorer_truth_sha256"]:
                raise RuntimeError("scorer truth changed")
            truths[(fold, band)] = pd.read_csv(path).set_index("point_id").observed_dbm
    rows = []
    for spec, arm, frame in frozen:
        truth = truths[(int(spec["outer_fold"]), spec["band"])].loc[frame.point_id].to_numpy(float)
        for method in runner.METHODS:
            row = {key: spec[key] for key in ("id", "band", "outer_fold", "mode", "budget", "repeat")}
            row.update(arm=arm, method=method)
            row.update(metrics(frame[method].to_numpy(float) - truth))
            rows.append(row)
    table = pd.DataFrame(rows)
    report = args.job_root / "comparison"; report.mkdir(exist_ok=True)
    table.to_csv(report / "repeat_metrics.csv.gz", index=False, compression="gzip")
    keys = ["id", "band", "outer_fold", "mode", "budget", "repeat", "method"]
    sem = table[table.arm == "semantic_8class"].drop(columns="arm")
    uni = table[table.arm == "uniform_nonconductor_concrete"].drop(columns="arm")
    paired = sem.merge(uni, on=keys, suffixes=("_semantic", "_uniform"), validate="one_to_one")
    for field in ("mae_db", "rmse_db", "medae_db", "p90_db", "bias_db"):
        paired[f"semantic_minus_uniform_{field}"] = paired[f"{field}_semantic"] - paired[f"{field}_uniform"]
    paired.to_csv(report / "paired_semantic_vs_uniform.csv.gz", index=False, compression="gzip")
    atomic_json(report / "manifest.json", {"schema": "aligned-material-power-v2-comparison-v1",
                "status": "COMPLETE", "task_n": 240, "arm_run_n": 480,
                "rows_scored_without_filtering": int(sum(item[2].shape[0] for item in frozen)),
                "methods": list(runner.METHODS), "artifacts": {
                    "repeat_metrics.csv.gz": sha256_file(report / "repeat_metrics.csv.gz"),
                    "paired_semantic_vs_uniform.csv.gz": sha256_file(report / "paired_semantic_vs_uniform.csv.gz")}})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--job-root", required=True, type=Path)
    prepare_parser = sub.add_parser("prepare", parents=[common])
    prepare_parser.add_argument("--results-root", required=True, type=Path)
    prepare_parser.add_argument("--transform-json", required=True, type=Path)
    prepare_parser.add_argument("--scene-xml", required=True, type=Path)
    trace_parser = sub.add_parser("trace", parents=[common])
    trace_parser.add_argument("--scene-xml", required=True, type=Path)
    trace_parser.add_argument("--arm", required=True, choices=ARMS)
    trace_parser.add_argument("--band", required=True, choices=BANDS)
    trace_parser.add_argument("--chunk-n", type=int, default=1024)
    trace_parser.add_argument("--solve-chunk-n", type=int, default=128)
    trace_parser.add_argument("--samples-per-source", type=int, default=50000)
    trace_parser.add_argument("--max-depth", type=int, default=3)
    trace_parser.add_argument("--seed", type=int, default=241301)
    smoke_parser = sub.add_parser("smoke-trace", parents=[common])
    smoke_parser.add_argument("--scene-xml", required=True, type=Path)
    run_parser = sub.add_parser("run-case", parents=[common])
    run_parser.add_argument("--results-root", required=True, type=Path)
    run_parser.add_argument("--key", required=True)
    run_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    controller_parser = sub.add_parser("controller", parents=[common])
    controller_parser.add_argument("--results-root", required=True, type=Path)
    controller_parser.add_argument("--scene-xml", required=True, type=Path)
    controller_parser.add_argument("--sionna-python", required=True, type=Path)
    controller_parser.add_argument("--fit-python", required=True, type=Path)
    controller_parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    score_parser = sub.add_parser("score", parents=[common])
    score_parser.add_argument("--results-root", required=True, type=Path)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"prepare": prepare, "trace": trace, "smoke-trace": smoke_trace, "run-case": run_case,
     "controller": controller, "score": score}[args.command](args)
