#!/usr/bin/env python3
"""H13 fixed-geometry semantic-material ablation.

The runner deliberately separates three operations:

* ``plan`` freezes the semantic, uniform, and shuffled material arms;
* ``trace`` performs a fresh Sionna RT solve for one arm and one band;
* ``score-band`` applies the same train-only calibration to every arm and
  reports own-path, common-path, and predeclared-fallback metrics.

Existing H7 gain arrays are not reusable for shuffled arms: they contain only
the final field/power after materials were assigned before path solving.  With
refraction enabled, neither the final gains nor their finite-path mask are a
material-independent candidate-path cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("n41", "n79")
FREQUENCY_MHZ = {"n41": 2524.95, "n79": 4827.36}
TX_SITES = {
    "E": np.asarray([134.42377217610678, 45.10205841064453, 37.0], float),
    "W": np.asarray([-136.84656524658203, 188.49261474609375, 17.0], float),
}
FIXED_TX_BY_BAND = {"n41": TX_SITES["E"], "n79": TX_SITES["W"]}
SEMANTIC_PRIORS = {
    "HKUSTGZ_metal_roof": "metal",
    "HKUSTGZ_limestone_solid": "marble",
    "HKUSTGZ_glass_solid": "glass",
    "HKUSTGZ_building_base": "concrete",
    "HKUSTGZ_ground_asphalt": "concrete",
    "HKUSTGZ_ground_water": "wet_ground",
    "HKUSTGZ_ground_vegetation": "medium_dry_ground",
    "HKUSTGZ_ground_stone": "marble",
}
CONDUCTOR_OBJECTS = frozenset({"HKUSTGZ_metal_roof"})
NONCONDUCTOR_OBJECTS = tuple(
    name for name in SEMANTIC_PRIORS if name not in CONDUCTOR_OBJECTS
)
DEFAULT_SHUFFLE_SEEDS = (13001, 13002, 13003)
MECHANISMS = {
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": True,
    "refraction": True,
    "diffraction": False,
    "edge_diffraction": False,
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(json.dumps(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def point_id_sha256(point_ids) -> str:
    digest = hashlib.sha256()
    for value in point_ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_u32(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def validate_uint32(value: int) -> int:
    value = int(value)
    if not 0 <= value <= np.iinfo(np.uint32).max:
        raise ValueError(f"seed outside uint32: {value}")
    return value


def _shuffled_assignment(seed: int) -> tuple[dict[str, str], dict[str, str]]:
    """Return a deterministic non-trivial permutation of nonconductor priors."""
    seed = validate_uint32(seed)
    labels = np.asarray([SEMANTIC_PRIORS[name] for name in NONCONDUCTOR_OBJECTS], object)
    rng = np.random.default_rng(seed)
    for _ in range(128):
        order = rng.permutation(len(NONCONDUCTOR_OBJECTS))
        shuffled = labels[order]
        if np.any(shuffled != labels):
            break
    else:
        raise RuntimeError(f"could not construct non-trivial shuffle for seed {seed}")
    assignments = {"HKUSTGZ_metal_roof": "metal"}
    source_objects = {"HKUSTGZ_metal_roof": "HKUSTGZ_metal_roof"}
    for target_index, target in enumerate(NONCONDUCTOR_OBJECTS):
        source = NONCONDUCTOR_OBJECTS[int(order[target_index])]
        assignments[target] = SEMANTIC_PRIORS[source]
        source_objects[target] = source
    return assignments, source_objects


def build_material_arms(shuffle_seeds=DEFAULT_SHUFFLE_SEEDS) -> list[dict]:
    seeds = tuple(validate_uint32(seed) for seed in shuffle_seeds)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("at least three distinct uint32 shuffle seeds are required")
    semantic = {
        "id": "semantic_8class",
        "kind": "semantic",
        "assignments": dict(SEMANTIC_PRIORS),
        "claim_boundary": "engineering ITU priors/proxies, not measured site EM parameters",
    }
    uniform_assignments = {
        name: ("metal" if name in CONDUCTOR_OBJECTS else "concrete")
        for name in SEMANTIC_PRIORS
    }
    uniform = {
        "id": "uniform_nonconductor_concrete",
        "kind": "uniform",
        "assignments": uniform_assignments,
        "metal_policy": "preserve HKUSTGZ_metal_roof as metal",
        "reason": "remove nonconductor semantic distinctions without deleting conductor physics",
    }
    arms = [uniform, semantic]
    fingerprints = set()
    for index, seed in enumerate(seeds, start=1):
        assignments, source_objects = _shuffled_assignment(seed)
        fingerprint = tuple(assignments[name] for name in SEMANTIC_PRIORS)
        if fingerprint in fingerprints:
            raise ValueError("shuffle seeds produced duplicate material assignments")
        fingerprints.add(fingerprint)
        arms.append({
            "id": f"shuffle_{index:02d}_seed{seed}",
            "kind": "label_shuffle",
            "seed": seed,
            "assignments": assignments,
            "source_object_by_target": source_objects,
            "metal_policy": "metal object fixed; labels permuted only among seven nonconductors",
        })
    return arms


def validate_prior_file(payload: dict) -> None:
    actual = {row["object"]: row["itu_type"] for row in payload.get("objects", [])}
    if actual != SEMANTIC_PRIORS:
        raise ValueError(f"prior file does not match frozen V7-r4 mapping: {actual}")
    if payload.get("status") != "PASS":
        raise ValueError("V7-r4 prior audit is not PASS")


def legacy_cache_reuse_assessment() -> dict:
    return {
        "status": "NO_GO_FRESH_RT_REQUIRED",
        "strict_reuse_allowed": False,
        "exact_semantic_reproduction_only": True,
        "reasons": [
            "legacy cache stores only final per-receiver gain, not path geometry or object interactions",
            "materials are assigned before PathSolver is called",
            "refraction is enabled, so material changes can alter valid path fields and finite-path support",
            "shuffled and uniform arms therefore require fresh matched Sionna solves",
        ],
    }


def load_arm(path: Path, arm_id: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [arm for arm in payload["arms"] if arm["id"] == arm_id]
    if len(matches) != 1:
        raise ValueError(f"material arm not found or duplicated: {arm_id}")
    return matches[0]


def load_receiver_contract(data_root: Path, band: str):
    cache = data_root / "cache" / band
    xyz = np.load(cache / "unique_xyz.npy").astype(float)
    mapping = np.load(cache / "point_to_unique.npy").astype(int)
    points = pd.read_csv(cache / "points.csv")
    required = {"point_id", "x", "y", "z"}
    if not required.issubset(points.columns):
        raise KeyError(f"points.csv missing columns: {sorted(required - set(points.columns))}")
    if len(mapping) != len(points) or mapping.min(initial=0) < 0 or mapping.max(initial=-1) >= len(xyz):
        raise ValueError("point_to_unique mapping is not aligned with points.csv/unique_xyz.npy")
    mapped = xyz[mapping]
    if not np.allclose(mapped, points[["x", "y", "z"]].to_numpy(float), equal_nan=False):
        raise ValueError("unique receiver mapping does not reproduce points.csv coordinates")
    return points, xyz, mapping


def geometry_stratified_sample(points: pd.DataFrame, count: int, seed: int, block_m: float) -> pd.DataFrame:
    """Signal-blind spatial round-robin sample from geometry columns only."""
    required = {"point_id", "x", "y", "z"}
    if not required.issubset(points.columns):
        raise KeyError(f"geometry pool missing columns: {sorted(required - set(points.columns))}")
    if count < 1 or count > len(points):
        raise ValueError(f"invalid sample count {count} for pool of {len(points)}")
    output = points.copy()
    output["spatial_block"] = (
        np.floor(output["x"].to_numpy(float) / float(block_m)).astype(int).astype(str)
        + ":"
        + np.floor(output["y"].to_numpy(float) / float(block_m)).astype(int).astype(str)
    )
    output["_point_rank"] = [stable_u32(f"H13/sample/{seed}/{value}") for value in output["point_id"]]
    output["_block_rank"] = [stable_u32(f"H13/block/{seed}/{value}") for value in output["spatial_block"]]
    output = output.sort_values(["spatial_block", "_point_rank", "point_id"], kind="stable")
    output["_within_block"] = output.groupby("spatial_block", sort=False).cumcount()
    output = output.sort_values(
        ["_within_block", "_block_rank", "_point_rank", "point_id"], kind="stable"
    ).head(int(count)).copy()
    return output.drop(columns=["_point_rank", "_block_rank", "_within_block"]).reset_index(drop=True)


def assign_balanced_spatial_folds(blocks, folds: int, seed: int) -> dict[str, int]:
    counts = pd.Series(np.asarray(blocks).astype(str)).value_counts().to_dict()
    ordered = sorted(
        counts,
        key=lambda block: (-counts[block], stable_u32(f"H13/fold/{seed}/{block}"), block),
    )
    load = [0] * int(folds)
    assignment = {}
    for block in ordered:
        fold = min(range(folds), key=lambda value: (load[value], value))
        assignment[block] = fold
        load[fold] += int(counts[block])
    return assignment


def prepare_band_data(
    source_points: Path,
    output_root: Path,
    band: str,
    *,
    count: int,
    seed: int,
    block_m: float,
    folds: int,
) -> dict:
    header = pd.read_csv(source_points, nrows=0).columns.tolist()
    geometry_columns = [
        name for name in (
            "point_id", "band", "band_task", "date", "trajectory_group", "segment_group",
            "x", "y", "z",
        ) if name in header
    ]
    required_geometry = {"point_id", "x", "y", "z"}
    if not required_geometry.issubset(geometry_columns):
        raise KeyError(f"source points lack geometry columns: {sorted(required_geometry - set(geometry_columns))}")
    geometry_pool = pd.read_csv(source_points, usecols=geometry_columns)
    selected = geometry_stratified_sample(geometry_pool, count, seed, block_m)
    fold_map = assign_balanced_spatial_folds(selected["spatial_block"], folds, seed)
    selected["outer_fold"] = selected["spatial_block"].map(fold_map).astype(int)
    selected["band"] = band

    # Labels are read only after the geometry-only IDs and folds are frozen.
    label_pool = pd.read_csv(source_points, usecols=["point_id", "observed_dbm"])
    label_pool["point_id"] = label_pool["point_id"].astype(str)
    if label_pool["point_id"].duplicated().any():
        raise ValueError(f"duplicate point_id in {source_points}")
    selected["point_id"] = selected["point_id"].astype(str)
    labels = selected[["point_id"]].merge(label_pool, on="point_id", how="left", validate="one_to_one")
    if not np.isfinite(pd.to_numeric(labels["observed_dbm"], errors="coerce")).all():
        raise ValueError("selected label file contains non-finite observed_dbm")

    cache = output_root / "geometry" / "cache" / band
    label_dir = output_root / "labels"
    cache.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(cache / "points.csv", index=False)
    xyz = selected[["x", "y", "z"]].to_numpy(float)
    unique_xyz, mapping = np.unique(xyz, axis=0, return_inverse=True)
    np.save(cache / "unique_xyz.npy", unique_xyz)
    np.save(cache / "point_to_unique.npy", mapping)
    labels.to_csv(label_dir / f"{band}.csv", index=False)
    payload = {
        "status": "PREPARED",
        "band": band,
        "source_points_sha256": sha256_file(source_points),
        "selected_rows": int(len(selected)),
        "unique_receivers": int(len(unique_xyz)),
        "sample_seed": validate_uint32(seed),
        "block_m": float(block_m),
        "folds": int(folds),
        "fold_counts": selected["outer_fold"].value_counts().sort_index().to_dict(),
        "point_id_sha256": point_id_sha256(selected["point_id"]),
        "geometry_xyz_sha256": array_sha256(xyz),
        "signal_fields_read_during_selection": [],
        "labels_read_after_selection": ["observed_dbm"],
    }
    write_json(output_root / f"prepare_{band}.json", payload)
    return payload


def assign_arm_materials(scene, arm: dict) -> None:
    from sionna.rt import ITURadioMaterial

    for object_name, itu_type in arm["assignments"].items():
        if object_name not in scene.objects:
            raise KeyError(f"scene lacks frozen V7-r4 object {object_name}")
        scene.objects[object_name].radio_material = ITURadioMaterial(
            name=f"h13_{arm['id']}_{object_name}",
            itu_type=str(itu_type),
            thickness=0.1,
        )
        scene.objects[object_name].radio_material.scattering_coefficient = 0.20
        scene.objects[object_name].radio_material.xpd_coefficient = 0.15


def as_complex_array(amplitudes) -> np.ndarray:
    """Normalize Sionna complex output without dropping the imaginary part."""
    if isinstance(amplitudes, (tuple, list)) and len(amplitudes) == 2:
        real = np.asarray(amplitudes[0])
        imag = np.asarray(amplitudes[1])
        if real.shape != imag.shape:
            raise ValueError("real/imag CIR components have different shapes")
        return real.astype(float) + 1j * imag.astype(float)
    array = np.asarray(amplitudes)
    if not np.iscomplexobj(array):
        return array.astype(float).astype(complex)
    return array.astype(complex, copy=False)


def incoherent_cir_power_db(amplitudes, receiver_count: int) -> np.ndarray:
    """Sum complex per-path CIR power, preserving the receiver dimension."""
    array = as_complex_array(amplitudes)
    if array.ndim < 1 or array.shape[0] < int(receiver_count):
        raise ValueError("CIR amplitude array lacks the expected receiver dimension")
    axes = tuple(range(1, array.ndim))
    power = np.sum(np.abs(array) ** 2, axis=axes) if axes else np.abs(array) ** 2
    power = np.asarray(power, float).reshape(-1)[: int(receiver_count)]
    output = np.full(int(receiver_count), np.nan, float)
    valid = np.isfinite(power) & (power > 0.0)
    output[valid] = 10.0 * np.log10(power[valid])
    return output


def _solve_scene_for_source(
    scene,
    solver,
    xyz: np.ndarray,
    tx_position: np.ndarray,
    tx_name: str,
    *,
    samples_per_source: int,
    max_depth: int,
    seed: int,
    chunk_size: int,
    direct_only: bool,
) -> np.ndarray:
    import drjit as dr
    from sionna.rt import Receiver, Transmitter

    scene.add(Transmitter(name=tx_name, position=tx_position.tolist(), power_dbm=30.0))
    output = np.full(len(xyz), np.nan, float)
    for start in range(0, len(xyz), chunk_size):
        stop = min(start + chunk_size, len(xyz))
        receiver_names = []
        for index, point in enumerate(xyz[start:stop], start):
            name = f"h13_rx_{index}"
            receiver_names.append(name)
            scene.add(Receiver(name=name, position=point.tolist()))
        paths = solver(
            scene=scene,
            max_depth=0 if direct_only else int(max_depth),
            samples_per_src=1 if direct_only else int(samples_per_source),
            los=True,
            specular_reflection=False if direct_only else MECHANISMS["specular_reflection"],
            diffuse_reflection=False if direct_only else MECHANISMS["diffuse_reflection"],
            refraction=False if direct_only else MECHANISMS["refraction"],
            diffraction=False,
            edge_diffraction=False,
            seed=seed,
        )
        amplitudes, _ = paths.cir(out_type="numpy")
        output[start:stop] = incoherent_cir_power_db(amplitudes, stop - start)
        for name in receiver_names:
            scene.remove(name)
        dr.sync_thread()
    scene.remove(tx_name)
    return output


def trace_fresh_source(
    scene_xml: Path,
    arm: dict,
    band: str,
    xyz: np.ndarray,
    tx_position: np.ndarray,
    *,
    samples_per_source: int,
    max_depth: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PathSolver, PlanarArray, load_scene

    scene = load_scene(str(scene_xml.resolve()))
    scene.frequency = FREQUENCY_MHZ[band] * 1e6
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.rx_array = scene.tx_array
    assign_arm_materials(scene, arm)
    solver = PathSolver()
    tx_position = np.asarray(tx_position, float)
    gain = _solve_scene_for_source(
        scene, solver, xyz, tx_position, "h13_tx",
        samples_per_source=samples_per_source, max_depth=max_depth, seed=seed,
        chunk_size=chunk_size, direct_only=False,
    )
    direct = np.isfinite(_solve_scene_for_source(
        scene, solver, xyz, tx_position, "h13_los_tx",
        samples_per_source=1, max_depth=0, seed=seed,
        chunk_size=chunk_size, direct_only=True,
    ))
    del solver, scene
    gc.collect()
    dr.sync_thread()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    return np.asarray(gain, float), np.asarray(direct, bool)


def trace_manifest_contract(
    *,
    band: str,
    arm: dict,
    scene_xml: Path,
    points: pd.DataFrame,
    xyz: np.ndarray,
    mapping: np.ndarray,
    tx_position: np.ndarray,
    seed: int,
    samples_per_source: int,
    max_depth: int,
    chunk_size: int,
) -> dict:
    return {
        "status": "COMPLETED",
        "band": band,
        "frequency_mhz": FREQUENCY_MHZ[band],
        "arm_id": arm["id"],
        "arm_kind": arm["kind"],
        "assignments": arm["assignments"],
        "scene_xml_sha256": sha256_file(scene_xml),
        "point_id_sha256": point_id_sha256(points["point_id"].astype(str)),
        "receiver_xyz_sha256": array_sha256(xyz),
        "point_to_unique_sha256": array_sha256(mapping),
        "evaluation_rows": int(len(points)),
        "unique_receivers": int(len(xyz)),
        "tx_position_m": np.asarray(tx_position, float).tolist(),
        "source_hypothesis": "n41 uses E rooftop; n79 uses W rooftop, matching the frozen H7/H8 condition",
        "tx_power_dbm_declared": 30.0,
        "tx_power_note": "H13 fixes declared transmitter power at 30 dBm; train-only bias absorbs unknown EIRP offset",
        "ray_seed": validate_uint32(seed),
        "samples_per_source": int(samples_per_source),
        "max_depth": int(max_depth),
        "chunk_size": int(chunk_size),
        "mechanisms": dict(MECHANISMS),
        "field_observable": "complex CIR coefficients",
        "power_aggregation": "incoherent sum of abs(a_path)^2 over all non-receiver CIR axes",
        "common_scattering_condition": {
            "scattering_coefficient": 0.20,
            "xpd_coefficient": 0.15,
            "applied_identically_to_all_material_arms": True,
            "matches_original_v7r4_scene_default": False,
        },
        "cross_experiment_delta_policy": "do not subtract H7/H12 metrics; H13 uses complex-CIR incoherent path power, a new common scattering condition, and a different RT/sample contract",
        "cache_policy": legacy_cache_reuse_assessment(),
    }


def validate_matched_manifests(manifests: list[dict]) -> None:
    if len(manifests) < 5:
        raise ValueError("expected uniform, semantic, and at least three shuffled manifests")
    invariant_keys = (
        "band", "frequency_mhz", "scene_xml_sha256", "point_id_sha256",
        "receiver_xyz_sha256", "point_to_unique_sha256", "evaluation_rows",
        "unique_receivers", "tx_position_m", "tx_power_dbm_declared", "ray_seed",
        "samples_per_source", "max_depth", "chunk_size", "mechanisms",
        "field_observable", "power_aggregation", "common_scattering_condition",
        "cross_experiment_delta_policy",
    )
    reference = manifests[0]
    for manifest in manifests:
        if manifest.get("status") != "COMPLETED":
            raise ValueError(f"incomplete RT arm: {manifest.get('arm_id')}")
        for key in invariant_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"unmatched RT contract field {key}: {manifest.get('arm_id')}")
    arm_ids = [manifest["arm_id"] for manifest in manifests]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("duplicate material arm manifests")
    kinds = [manifest.get("arm_kind") for manifest in manifests]
    if kinds.count("uniform") != 1 or kinds.count("semantic") != 1 or kinds.count("label_shuffle") < 3:
        raise ValueError("material arms do not satisfy the H13 ablation contract")


def common_path_mask(gains_by_arm: dict[str, np.ndarray]) -> np.ndarray:
    if not gains_by_arm:
        raise ValueError("no gain arrays")
    if any(np.asarray(value).ndim != 1 for value in gains_by_arm.values()):
        raise ValueError("common-path inputs must be one gain per evaluation row")
    lengths = {len(np.asarray(value)) for value in gains_by_arm.values()}
    if len(lengths) != 1:
        raise ValueError("gain arrays have different lengths")
    return np.logical_and.reduce([
        np.isfinite(np.asarray(value, float)) for value in gains_by_arm.values()
    ])


def apply_fixed_fallback(prediction: np.ndarray, fallback: np.ndarray) -> tuple[np.ndarray, int]:
    prediction = np.asarray(prediction, float)
    fallback = np.asarray(fallback, float)
    if prediction.shape != fallback.shape:
        raise ValueError("prediction/fallback shape mismatch")
    if not np.isfinite(fallback).all():
        raise ValueError("predeclared fallback must cover every evaluation row")
    output = prediction.copy()
    missing = ~np.isfinite(output)
    output[missing] = fallback[missing]
    return output, int(missing.sum())


def metrics(observed: np.ndarray, predicted: np.ndarray, mask=None) -> dict:
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    if observed.shape != predicted.shape:
        raise ValueError("observed/predicted shape mismatch")
    include = np.ones(len(observed), bool) if mask is None else np.asarray(mask, bool)
    valid = include & np.isfinite(observed) & np.isfinite(predicted)
    error = predicted[valid] - observed[valid]
    return {
        "eligible_n": int(include.sum()),
        "predicted_n": int(valid.sum()),
        "coverage": float(valid.sum() / max(1, include.sum())),
        "rmse_db": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
        "mae_db": float(np.mean(np.abs(error))) if len(error) else np.nan,
        "median_abs_db": float(np.median(np.abs(error))) if len(error) else np.nan,
        "bias_db": float(np.mean(error)) if len(error) else np.nan,
    }


def fit_twc(distance_m, nlos, residual_db, ridge: float):
    log_distance = np.log10(np.maximum(np.asarray(distance_m, float), 1.0))
    center = float(log_distance.mean())
    q = log_distance - center
    nlos = np.asarray(nlos, float)
    design = np.column_stack([np.ones(len(q)), q, nlos, q * nlos])
    penalty = np.diag([0.0, ridge, ridge, ridge])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ residual_db)
    return center, coefficients


def apply_twc(distance_m, nlos, center: float, coefficients: np.ndarray) -> np.ndarray:
    q = np.log10(np.maximum(np.asarray(distance_m, float), 1.0)) - center
    nlos = np.asarray(nlos, float)
    design = np.column_stack([np.ones(len(q)), q, nlos, q * nlos])
    return design @ np.asarray(coefficients, float)


def idw_residual(train_xy, train_values, query_xy, *, k: int, power: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    k = min(int(k), len(train_xy))
    if k < 1:
        raise ValueError("empty IDW training set")
    distance, index = cKDTree(np.asarray(train_xy, float)).query(
        np.asarray(query_xy, float), k=k, workers=-1
    )
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    values = np.asarray(train_values, float)[index]
    zero = distance <= 1e-9
    output = np.empty(len(query_xy), float)
    exact = zero.any(axis=1)
    if exact.any():
        output[exact] = np.sum(np.where(zero[exact], values[exact], 0.0), axis=1) / zero[exact].sum(axis=1)
    if (~exact).any():
        weights = 1.0 / np.maximum(distance[~exact], 1e-6) ** float(power)
        output[~exact] = np.sum(weights * values[~exact], axis=1) / weights.sum(axis=1)
    return output


def buffered_train_mask(frame: pd.DataFrame, held_out: int, buffer_m: float) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    fold = frame["outer_fold"].to_numpy(int)
    test = fold == int(held_out)
    candidate = ~test
    if not test.any():
        return candidate, test
    xy = frame[["x", "y"]].to_numpy(float)
    nearest, _ = cKDTree(xy[test]).query(xy[candidate], k=1, workers=-1)
    train = candidate.copy()
    train[candidate] = np.asarray(nearest, float) >= float(buffer_m)
    return train, test


def gaussian_predict_fixed(train_xy, train_values, query_xy, *, bandwidth_m: float, k: int) -> np.ndarray:
    from scipy.spatial import cKDTree

    width = min(int(k), len(train_xy))
    if width < 1:
        raise ValueError("empty Gaussian training set")
    distance, index = cKDTree(np.asarray(train_xy, float)).query(
        np.asarray(query_xy, float), k=width, workers=-1
    )
    if width == 1:
        distance = distance[:, None]
        index = index[:, None]
    weights = np.exp(-0.5 * (distance / float(bandwidth_m)) ** 2)
    weight_sum = weights.sum(axis=1)
    prediction = np.sum(weights * np.asarray(train_values, float)[index], axis=1) / np.maximum(weight_sum, 1e-12)
    zero_weight = weight_sum <= 1e-12
    if zero_weight.any():
        prediction[zero_weight] = np.asarray(train_values, float)[index[zero_weight, 0]]
    return prediction


def crossfit_gaussian_fallback(
    frame: pd.DataFrame,
    *,
    buffer_m: float,
    bandwidth_m: float,
    k: int,
    min_train: int,
) -> tuple[np.ndarray, list[dict]]:
    observed = frame["observed_dbm"].to_numpy(float)
    xy = frame[["x", "y"]].to_numpy(float)
    output = np.full(len(frame), np.nan, float)
    audit = []
    for held_out in sorted(frame["outer_fold"].astype(int).unique()):
        train, test = buffered_train_mask(frame, held_out, buffer_m)
        if train.sum() < int(min_train):
            raise RuntimeError(f"fold {held_out} has only {train.sum()} buffered Gaussian training rows")
        output[test] = gaussian_predict_fixed(
            xy[train], observed[train], xy[test], bandwidth_m=bandwidth_m, k=k
        )
        audit.append({
            "outer_fold": int(held_out), "test_n": int(test.sum()),
            "train_after_buffer_n": int(train.sum()),
            "buffer_excluded_n": int((~test).sum() - train.sum()),
        })
    if not np.isfinite(output).all():
        raise RuntimeError("training-only Gaussian fallback lacks full held-out coverage")
    return output, audit


def crossfit_calibrate(
    frame: pd.DataFrame,
    raw_gain: np.ndarray,
    distance_m: np.ndarray,
    nlos: np.ndarray,
    *,
    postprocessor: str,
    twc_ridge: float,
    idw_k: int,
    idw_power: float,
    min_train_paths: int,
    buffer_m: float,
) -> np.ndarray:
    required = {"observed_dbm", "outer_fold", "x", "y"}
    if not required.issubset(frame.columns):
        raise KeyError(f"score frame missing columns: {sorted(required - set(frame.columns))}")
    observed = frame["observed_dbm"].to_numpy(float)
    fold = frame["outer_fold"].to_numpy(int)
    xy = frame[["x", "y"]].to_numpy(float)
    distance = np.asarray(distance_m, float)
    nlos = np.asarray(nlos, float)
    raw_gain = np.asarray(raw_gain, float)
    if len(raw_gain) != len(frame):
        raise ValueError("raw gain is not aligned with score frame")
    prediction = np.full(len(frame), np.nan, float)
    for held_out in sorted(set(fold.tolist())):
        train, test = buffered_train_mask(frame, held_out, buffer_m)
        train_path = train & np.isfinite(raw_gain)
        test_path = test & np.isfinite(raw_gain)
        if train_path.sum() < int(min_train_paths):
            raise RuntimeError(f"fold {held_out} has only {train_path.sum()} training paths")
        if postprocessor == "bias":
            bias = float(np.median(observed[train_path] - raw_gain[train_path]))
            prediction[test_path] = raw_gain[test_path] + bias
            continue
        center, coefficients = fit_twc(
            distance[train_path], nlos[train_path],
            observed[train_path] - raw_gain[train_path], float(twc_ridge),
        )
        corrected = raw_gain + apply_twc(distance, nlos, center, coefficients)
        prediction[test_path] = corrected[test_path]
        if postprocessor == "twc_idw" and test_path.any():
            residual = observed[train_path] - corrected[train_path]
            prediction[test_path] += idw_residual(
                xy[train_path], residual, xy[test_path], k=idw_k, power=idw_power
            )
    return prediction


def score_arms(
    frame: pd.DataFrame,
    gains_by_arm: dict[str, np.ndarray],
    los_by_arm: dict[str, np.ndarray],
    tx_position: np.ndarray,
    *,
    postprocessor: str,
    twc_ridge: float = 5.0,
    idw_k: int = 32,
    idw_power: float = 2.0,
    min_train_paths: int = 100,
    buffer_m: float = 30.0,
    gaussian_bandwidth_m: float = 50.0,
    gaussian_k: int = 128,
):
    if postprocessor not in {"bias", "twc", "twc_idw"}:
        raise ValueError(f"unknown postprocessor: {postprocessor}")
    observed = frame["observed_dbm"].to_numpy(float)
    xyz = frame[["x", "y", "z"]].to_numpy(float)
    tx_position = np.asarray(tx_position, float)
    if tx_position.shape != (3,):
        raise ValueError("H13 requires one frozen Tx coordinate per band")
    distance = np.linalg.norm(xyz - tx_position[None, :], axis=1)
    common = common_path_mask(gains_by_arm)
    fallback, fold_audit = crossfit_gaussian_fallback(
        frame, buffer_m=buffer_m, bandwidth_m=gaussian_bandwidth_m,
        k=gaussian_k, min_train=min_train_paths,
    )
    predictions = {}
    rows = []
    for arm_id, raw_gain in gains_by_arm.items():
        nlos = ~np.asarray(los_by_arm[arm_id], bool)
        prediction = crossfit_calibrate(
            frame, raw_gain, distance, nlos,
            postprocessor=postprocessor, twc_ridge=twc_ridge,
            idw_k=idw_k, idw_power=idw_power, min_train_paths=min_train_paths,
            buffer_m=buffer_m,
        )
        predictions[arm_id] = prediction
        own = metrics(observed, prediction)
        rows.append({"arm_id": arm_id, "scope": "own_path", **own, "fallback_n": 0})
        shared = metrics(observed, prediction, common)
        rows.append({"arm_id": arm_id, "scope": "common_path", **shared, "fallback_n": 0})
        filled, fallback_n = apply_fixed_fallback(prediction, fallback)
        full = metrics(observed, filled)
        rows.append({"arm_id": arm_id, "scope": "predeclared_fallback", **full, "fallback_n": fallback_n})
    return pd.DataFrame(rows), predictions, common, fallback, fold_audit


def command_plan(args) -> None:
    priors = json.loads(args.priors_json.read_text(encoding="utf-8"))
    validate_prior_file(priors)
    arms = build_material_arms(args.shuffle_seed)
    payload = {
        "status": "PLANNED_NOT_TRACED",
        "scene_xml_sha256_from_prior_audit": priors["scene_xml_sha256"],
        "arms": arms,
        "cache_reuse": legacy_cache_reuse_assessment(),
        "comparison_rule": "same scene, source, band, receivers, ray seed, mechanisms and postprocessor",
    }
    write_json(args.output_dir / "material_arms.json", payload)
    write_json(args.output_dir / "cache_reuse_assessment.json", payload["cache_reuse"])


def command_prepare(args) -> None:
    if (
        args.count_per_band != 2048
        or args.sample_seed != 241300
        or not np.isclose(args.block_m, 50.0)
        or args.folds != 5
    ):
        raise ValueError("H13 integration is frozen at 2048/band, seed 241300, 50 m blocks, and five folds")
    seed = validate_uint32(args.sample_seed)
    summaries = []
    for band in BANDS:
        summaries.append(prepare_band_data(
            args.h6_data_root / "cache" / band / "points.csv",
            args.output_dir,
            band,
            count=args.count_per_band,
            seed=seed,
            block_m=args.block_m,
            folds=args.folds,
        ))
    write_json(args.output_dir / "prepare_result.json", {
        "status": "PREPARED",
        "bands": summaries,
        "total_selected_rows": int(sum(row["selected_rows"] for row in summaries)),
        "selection_contract": "geometry-only before labels are read",
    })


def command_trace(args) -> None:
    seed = validate_uint32(args.ray_seed)
    if args.tx_power_dbm != 30.0:
        raise ValueError("H13 protocol is fixed at a declared 30 dBm; refusing a misleading override")
    arm = load_arm(args.arms_json, args.arm_id)
    points, xyz, mapping = load_receiver_contract(args.data_root, args.band)
    if args.limit_unique_receivers:
        raise ValueError(
            "prefix receiver limits are forbidden; prepare a geometry-only fixed sample as a separate data root"
        )
    tx_position = np.asarray(
        args.tx_position if args.tx_position is not None else FIXED_TX_BY_BAND[args.band],
        float,
    )
    if tx_position.shape != (3,):
        raise ValueError("H13 requires exactly one frozen --tx-position for each band")
    if not np.allclose(tx_position, FIXED_TX_BY_BAND[args.band], rtol=0.0, atol=1e-9):
        raise ValueError("H13 Tx must match the frozen H7/H8 n41/E or n79/W coordinate")
    if args.samples_per_source != 50000 or args.max_depth != 3 or seed != 241301:
        raise ValueError("H13 RT is frozen at ray_seed=241301, samples_per_source=50000, max_depth=3")
    gain, direct_los = trace_fresh_source(
        args.scene_xml, arm, args.band, xyz, tx_position,
        samples_per_source=args.samples_per_source, max_depth=args.max_depth,
        seed=seed, chunk_size=args.chunk_size,
    )
    output = args.output_dir / args.band / args.arm_id
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "gain_unique.npy", gain)
    np.save(output / "los_unique.npy", direct_los)
    manifest = trace_manifest_contract(
        band=args.band, arm=arm, scene_xml=args.scene_xml, points=points,
        xyz=xyz, mapping=mapping, tx_position=tx_position,
        seed=seed, samples_per_source=args.samples_per_source,
        max_depth=args.max_depth, chunk_size=args.chunk_size,
    )
    manifest["unique_path_n"] = int(np.isfinite(gain).sum())
    manifest["unique_path_coverage"] = float(np.isfinite(gain).mean())
    for package in ("sionna", "mitsuba", "drjit"):
        try:
            manifest[f"{package}_version"] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            manifest[f"{package}_version"] = "unknown"
    write_json(output / "manifest.json", manifest)


def command_score_band(args) -> None:
    if (
        args.expected_rows != 2048
        or not np.isclose(args.buffer_m, 30.0)
        or not np.isclose(args.twc_ridge, 5.0)
        or args.idw_k != 32
        or not np.isclose(args.idw_power, 2.0)
        or not np.isclose(args.gaussian_bandwidth_m, 50.0)
        or args.gaussian_k != 128
    ):
        raise ValueError("H13 scoring parameters differ from the frozen bounded protocol")
    manifests = []
    gains = {}
    los = {}
    for manifest_path in sorted(args.trace_root.glob(f"{args.band}/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        gains[manifest["arm_id"]] = np.load(manifest_path.parent / "gain_unique.npy")
        los[manifest["arm_id"]] = np.load(manifest_path.parent / "los_unique.npy").astype(bool)
    validate_matched_manifests(manifests)
    points, _, mapping = load_receiver_contract(args.data_root, args.band)
    if len(points) != args.expected_rows:
        raise RuntimeError(f"expected {args.expected_rows} rows for {args.band}, found {len(points)}")
    if point_id_sha256(points["point_id"].astype(str)) != manifests[0]["point_id_sha256"]:
        raise RuntimeError("score points do not match trace manifest")
    observation = pd.read_csv(args.labels_csv)
    observation["point_id"] = observation["point_id"].astype(str)
    points["point_id"] = points["point_id"].astype(str)
    if observation["point_id"].duplicated().any():
        raise ValueError("duplicate observation point_id")
    frame = points.merge(observation, on="point_id", how="left", validate="one_to_one", suffixes=("", "_obs"))
    required = ("observed_dbm", "outer_fold")
    if frame[list(required)].isna().any().any():
        raise RuntimeError("missing observation/fold/geometry fields after point_id join")
    if set(frame["outer_fold"].astype(int)) != set(range(5)):
        raise RuntimeError("H13 score requires all five geometry-only outer folds")
    point_gains = {arm_id: np.asarray(value, float)[mapping] for arm_id, value in gains.items()}
    point_los = {arm_id: np.asarray(value, bool)[mapping] for arm_id, value in los.items()}
    reference_los = next(iter(point_los.values()))
    if any(not np.array_equal(value, reference_los) for value in point_los.values()):
        raise RuntimeError("direct-path geometry differs across material arms")
    tx_position = np.asarray(manifests[0]["tx_position_m"], float)
    processors = ("bias", "twc", "twc_idw") if args.postprocessor == "all" else (args.postprocessor,)
    summaries = []
    predictions_by_processor = {}
    common = None
    fallback = None
    fold_audit = None
    for processor in processors:
        local_summary, predictions, local_common, local_fallback, local_audit = score_arms(
            frame, point_gains, point_los, tx_position, postprocessor=processor,
            twc_ridge=args.twc_ridge, idw_k=args.idw_k,
            idw_power=args.idw_power, min_train_paths=args.min_train_paths,
            buffer_m=args.buffer_m, gaussian_bandwidth_m=args.gaussian_bandwidth_m,
            gaussian_k=args.gaussian_k,
        )
        local_summary.insert(1, "postprocessor", processor)
        summaries.append(local_summary)
        predictions_by_processor[processor] = predictions
        if common is not None and not np.array_equal(common, local_common):
            raise RuntimeError("common-path mask changed across postprocessors")
        if fallback is not None and not np.allclose(fallback, local_fallback):
            raise RuntimeError("Gaussian fallback changed across postprocessors")
        common, fallback, fold_audit = local_common, local_fallback, local_audit
    summary = pd.concat(summaries, ignore_index=True)
    output = args.output_dir / args.band
    output.mkdir(parents=True, exist_ok=True)
    prediction_frame = frame[["point_id", "outer_fold", "observed_dbm", "x", "y", "z"]].copy()
    prediction_frame["common_path_all_arms"] = common
    prediction_frame["pred_training_only_gaussian_fallback"] = fallback
    for processor, predictions in predictions_by_processor.items():
        for arm_id, prediction in predictions.items():
            prediction_frame[f"pred_{processor}_{arm_id}"] = prediction
    for arm_id in point_gains:
        prediction_frame[f"has_path_{arm_id}"] = np.isfinite(point_gains[arm_id])
    prediction_frame.to_csv(output / "point_predictions.csv", index=False)
    summary.to_csv(output / "summary_metrics.csv", index=False)
    write_json(output / "result.json", {
        "status": "COMPLETED",
        "band": args.band,
        "rows": len(frame),
        "arms": sorted(point_gains),
        "common_path_n": int(common.sum()),
        "postprocessors": list(processors),
        "postprocessor_parameters_predeclared": {
            "twc_ridge": args.twc_ridge,
            "idw_k": args.idw_k,
            "idw_power": args.idw_power,
            "spatial_buffer_m": args.buffer_m,
            "gaussian_bandwidth_m": args.gaussian_bandwidth_m,
            "gaussian_k": args.gaussian_k,
        },
        "fold_buffer_audit": fold_audit,
        "source_hypothesis": "n41/E and n79/W are separate frozen band conditions, not simultaneous transmitters",
        "selection_rule": "none; every input row is scored, with a frozen fallback for no-path rows",
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--priors-json", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--shuffle-seed", action="append", type=int, default=[])
    plan.set_defaults(func=command_plan)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--h6-data-root", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--count-per-band", type=int, default=2048)
    prepare.add_argument("--sample-seed", type=int, default=241300)
    prepare.add_argument("--block-m", type=float, default=50.0)
    prepare.add_argument("--folds", type=int, default=5)
    prepare.set_defaults(func=command_prepare)

    trace = subparsers.add_parser("trace")
    trace.add_argument("--scene-xml", required=True, type=Path)
    trace.add_argument("--arms-json", required=True, type=Path)
    trace.add_argument("--arm-id", required=True)
    trace.add_argument("--data-root", required=True, type=Path)
    trace.add_argument("--output-dir", required=True, type=Path)
    trace.add_argument("--band", required=True, choices=BANDS)
    trace.add_argument(
        "--tx-position", nargs=3, type=float,
        default=None,
        help="optional override; defaults to n41/E or n79/W exactly as frozen in H7/H8",
    )
    trace.add_argument("--tx-power-dbm", type=float, default=30.0)
    trace.add_argument("--samples-per-source", type=int, default=50000)
    trace.add_argument("--max-depth", type=int, default=3)
    trace.add_argument("--ray-seed", type=int, default=241301)
    trace.add_argument("--chunk-size", type=int, default=128)
    trace.add_argument("--limit-unique-receivers", type=int, default=0)
    trace.set_defaults(func=command_trace)

    score = subparsers.add_parser("score-band")
    score.add_argument("--trace-root", required=True, type=Path)
    score.add_argument("--data-root", required=True, type=Path)
    score.add_argument("--labels-csv", required=True, type=Path)
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--band", required=True, choices=BANDS)
    score.add_argument("--expected-rows", required=True, type=int)
    score.add_argument("--postprocessor", choices=("all", "bias", "twc", "twc_idw"), default="all")
    score.add_argument("--twc-ridge", type=float, default=5.0)
    score.add_argument("--idw-k", type=int, default=32)
    score.add_argument("--idw-power", type=float, default=2.0)
    score.add_argument("--min-train-paths", type=int, default=100)
    score.add_argument("--buffer-m", type=float, default=30.0)
    score.add_argument("--gaussian-bandwidth-m", type=float, default=50.0)
    score.add_argument("--gaussian-k", type=int, default=128)
    score.set_defaults(func=command_score_band)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan" and not args.shuffle_seed:
        args.shuffle_seed = list(DEFAULT_SHUFFLE_SEEDS)
    args.func(args)


if __name__ == "__main__":
    main()
