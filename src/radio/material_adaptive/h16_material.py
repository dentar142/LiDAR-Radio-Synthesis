#!/usr/bin/env python3
"""H16 train-only differentiable neural-material arm for frozen V7-r4 geometry.

The public :func:`fit_material` function deliberately accepts query coordinates
but no query labels.  Candidate paths are traced once for the concatenated
train/query receiver set, detached, and reused for the prior and learned field.
Reported gains are raw channel gains (not power-offset-adjusted predictions).

The learned values are effective, RSRP-conditioned material fields.  They are
not an identifiable recovery of site-specific dielectric constants.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "wedt_full_v1" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

FREQUENCY_HZ = {"n41": 2524.95e6, "n79": 4827.36e6}
BOUNDS_MIN = np.asarray([-504.0, -634.0, 0.0], np.float32)
BOUNDS_MAX = np.asarray([596.0, 586.0, 53.0], np.float32)
SCHEMA = "h16-real-sionna-neural-material-v1"
PRIOR_REGULARIZATION = 1e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _xyz(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1:] != (3,) or (not allow_empty and len(array) == 0):
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate_inputs(scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
                     seed, steps, samples, width, bands):
    scene_xml, priors_json = Path(scene_xml), Path(priors_json)
    if not scene_xml.is_file() or not priors_json.is_file():
        raise FileNotFoundError("scene_xml and priors_json must be existing files")
    band = str(band).lower()
    if band not in FREQUENCY_HZ:
        raise ValueError(f"band must be one of {sorted(FREQUENCY_HZ)}")
    tx = np.asarray(tx, np.float32)
    if tx.shape != (3,) or not np.isfinite(tx).all():
        raise ValueError("tx must contain three finite coordinates")
    train_xyz = _xyz(train_xyz, "train_xyz")
    query_xyz = _xyz(query_xyz, "query_xyz", allow_empty=True)
    train_y = np.asarray(train_y, np.float32).reshape(-1)
    if len(train_y) != len(train_xyz):
        raise ValueError("train_y length must equal train_xyz length")
    if not np.isfinite(train_y).any():
        raise ValueError("train_y must contain at least one finite label")
    if not 0 <= int(seed) <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be uint32")
    if int(steps) < 1 or int(samples) < 1 or int(width) < 1 or int(bands) < 1:
        raise ValueError("steps, samples, width, and bands must be positive")
    return scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz


def _load_priors(path: Path, band: str, W):
    # Reuse the vetted H12 parser, but only the engineering-prior artifact is
    # consumed: no H12 weights, BCM, pseudo-observations, or fitted state.
    from run_h12_material_smoke import load_semantic_priors

    priors, payload = load_semantic_priors(path, band, W)
    expected = payload.get("scene_xml_sha256")
    return priors, payload, expected


def _build_prior_scene(scene_xml, positions, tx, band, priors):
    from run_h12_material_smoke import attach_static_truth, build_scene

    scene = build_scene(scene_xml, positions, tx, band)
    prior_values = {key: (value.eps_r, value.sigma) for key, value in priors.items()}
    attach_static_truth(scene, priors, prior_values, f"h16_prior_{band}")
    return scene


def _trace_kwargs(seed: int, samples: int) -> dict:
    return {
        "max_depth": 3,
        "max_num_paths_per_src": int(samples),
        "samples_per_src": int(samples),
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        # The installed stack cannot provide the required differentiable
        # diffraction path.  Keep this fixed, explicit, and auditable.
        "diffraction": False,
        "edge_diffraction": False,
        "seed": int(np.uint32(seed)),
    }


def incoherent_path_log_gain(paths, W):
    """Return differentiable 10log10(sum_path(|a|^2)) per receiver.

    This intentionally does not use CFR(0): paths are summed in power rather
    than coherently in complex amplitude.  The companion NumPy power is used
    solely to freeze no-path support; gradients flow through ``gain_db``.
    """
    real, imag = paths.a
    power = W.dr.square(real) + W.dr.square(imag)
    num_rx = int(power.shape[0])
    num_paths = int(power.shape[-1])
    power = W.dr.reshape(W.mi.TensorXf, power, [num_rx, -1, num_paths])
    power = W.dr.sum(W.dr.sum(power, axis=2), axis=1)
    gain_db = 10.0 * W.dr.log(W.dr.maximum(power, 1e-30)) / np.log(10.0)
    return gain_db, np.asarray(power, dtype=float).reshape(-1)


def _attach_fields(scene, priors, W, seed: int, width: int, bands: int, lr: float):
    from run_h12_material_smoke import add_and_assign

    fields = W.SemanticFieldSet(
        BOUNDS_MIN, BOUNDS_MAX, priors, bands=int(bands), width=int(width),
        rsrp_mode="fixed", eps_residual_fraction=0.35, sigma_log_residual=1.0,
    )
    arrays = fields.initial_parameter_arrays(int(seed))
    # Exact engineering-prior start: zero every output residual while retaining
    # random hidden features so the first backward pass can move eps/sigma rows.
    for key in tuple(arrays):
        if key.endswith(".w2") or key.endswith(".b2"):
            arrays[key] = np.zeros_like(arrays[key], dtype=np.float32)
    optimizer = W.mi.ad.Adam(lr=float(lr), params={k: W.mi.Float(v) for k, v in arrays.items()})
    for semantic_id, prior in priors.items():
        material = W.SemanticSpatialRadioMaterial(
            fields, semantic_id, optimizer, name=f"h16_field_{band_safe(semantic_id)}",
            relative_permittivity=prior.eps_r, conductivity=prior.sigma,
            scattering_coefficient=prior.scattering, xpd_coefficient=prior.xpd,
        )
        add_and_assign(scene, semantic_id, material)
    return fields, optimizer, arrays


def band_safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))


def _optimizer_arrays(optimizer) -> dict[str, np.ndarray]:
    return {str(key): np.asarray(optimizer[key], np.float32).copy() for key in optimizer.keys()}


def _select(value, indices: np.ndarray, W):
    return W.dr.gather(W.mi.Float, W.dr.ravel(value), W.mi.UInt(indices.astype(np.uint32)))


def _regularization(optimizer, initial, W):
    terms = []
    for key in optimizer.keys():
        delta = optimizer[key] - W.mi.Float(initial[str(key)])
        terms.append(W.dr.mean(W.dr.square(delta)))
    value = terms[0]
    for term in terms[1:]:
        value = value + term
    return PRIOR_REGULARIZATION * value / len(terms)


def _gradient_audit(optimizer, W) -> tuple[float, bool]:
    maximum = 0.0
    finite = True
    for key in optimizer.keys():
        gradient = np.asarray(W.dr.grad(optimizer[key]), float).reshape(-1)
        if gradient.size:
            finite &= bool(np.isfinite(gradient).all())
            if np.isfinite(gradient).any():
                maximum = max(maximum, float(np.max(np.abs(gradient[np.isfinite(gradient)]))))
    return maximum, finite


def _material_table(fields, params, priors, W, points: np.ndarray) -> list[dict]:
    rows = []
    p = W.mi.Point3f(points[:, 0], points[:, 1], points[:, 2])
    for semantic_id in sorted(priors):
        values = fields.field_for(semantic_id)(p, params)
        arrays = [np.asarray(value, float).reshape(-1) for value in values]
        for index, xyz in enumerate(points):
            sampled = [value[index] if len(value) > 1 else value[0] for value in arrays]
            rows.append({
                "semantic_id": semantic_id, "x": float(xyz[0]), "y": float(xyz[1]),
                "z": float(xyz[2]), "eps_r": float(sampled[0]),
                "sigma_s_per_m": float(sampled[1]),
                "scattering_fixed": float(sampled[2]),
                "xpd_fixed": float(sampled[3]),
            })
    return rows


def _sampling_points() -> np.ndarray:
    fractions = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], np.float32)[:, None]
    return BOUNDS_MIN + fractions * (BOUNDS_MAX - BOUNDS_MIN)


def _coverage(mask: np.ndarray) -> dict:
    return {"count": int(mask.sum()), "total": int(len(mask)),
            "fraction": float(mask.mean()) if len(mask) else 0.0}


def fit_material(scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz, *,
                 seed, steps=60, samples=50000, width=8, bands=2,
                 output_dir=None) -> dict:
    """Fit eps/sigma residual fields using finite fixed-path training rows only."""
    started = time.time()
    (scene_xml, priors_json, band, tx, train_xyz, train_y,
     query_xyz) = _validate_inputs(scene_xml, priors_json, band, tx, train_xyz,
                                   train_y, query_xyz, seed, steps, samples, width, bands)
    import wedt_sionna2 as W

    priors, prior_payload, expected_scene_hash = _load_priors(priors_json, band, W)
    scene_hash = _sha256(scene_xml)
    if expected_scene_hash and scene_hash != expected_scene_hash:
        raise RuntimeError("scene hash differs from the frozen V7-r4 prior contract")
    all_xyz = np.vstack((train_xyz, query_xyz))
    scene = _build_prior_scene(scene_xml, all_xyz, tx, band, priors)
    trace_config = _trace_kwargs(seed, samples)
    with W.dr.suspend_grad():
        cache = W.trace_geometry(scene, **trace_config)
        baseline_tensor, baseline_power = incoherent_path_log_gain(cache.compute_fields(), W)
        baseline = np.asarray(baseline_tensor, float).reshape(-1)
        repeat_tensor, repeat_power = incoherent_path_log_gain(cache.compute_fields(), W)
        repeat = np.asarray(repeat_tensor, float).reshape(-1)
    fixed_mask = np.isfinite(baseline) & np.isfinite(baseline_power) & (baseline_power > 0.0)
    repeat_mask = np.isfinite(repeat) & np.isfinite(repeat_power) & (repeat_power > 0.0)
    baseline = np.where(fixed_mask, baseline, np.nan)
    repeat = np.where(repeat_mask, repeat, np.nan)
    if not np.array_equal(fixed_mask, repeat_mask) or not np.allclose(
            baseline[fixed_mask], repeat[fixed_mask], atol=1e-5, rtol=1e-6):
        raise RuntimeError("baseline repeat stability audit failed on detached geometry")

    n_train = len(train_xyz)
    train_mask = fixed_mask[:n_train] & np.isfinite(train_y)
    query_mask = fixed_mask[n_train:].copy()
    if not train_mask.any():
        raise RuntimeError("no finite fixed-path training rows")
    indices = np.flatnonzero(train_mask).astype(np.uint32)
    target = W.mi.Float(train_y[train_mask])
    fields, optimizer, initial = _attach_fields(scene, priors, W, int(seed), width, bands, 1e-3)
    initial_state = _optimizer_arrays(optimizer)
    initial_field, initial_power = incoherent_path_log_gain(cache.compute_fields(), W)
    initial_np = np.asarray(initial_field, float).reshape(-1)
    if not np.allclose(initial_np[fixed_mask], baseline[fixed_mask], atol=1e-5, rtol=1e-6):
        raise RuntimeError("zero-residual neural field does not exactly reproduce the prior")

    grid = _sampling_points()
    table_before = _material_table(fields, optimizer, priors, W, grid)
    history = []
    max_gradient = 0.0
    gradients_finite = True
    for step in range(int(steps)):
        all_gain, _ = incoherent_path_log_gain(cache.compute_fields(), W)
        prediction = _select(all_gain, indices, W)
        beta = W.dr.mean(target - prediction)
        data_loss = W.dr.mean(W.dr.square(prediction + beta - target))
        prior_loss = _regularization(optimizer, initial, W)
        loss = data_loss + prior_loss
        loss_value = float(np.asarray(loss).reshape(-1)[0])
        data_value = float(np.asarray(data_loss).reshape(-1)[0])
        beta_value = float(np.asarray(beta).reshape(-1)[0])
        if not np.isfinite([loss_value, data_value, beta_value]).all():
            raise RuntimeError(f"non-finite material objective at step {step + 1}")
        W.dr.backward(loss)
        grad_value, grad_finite = _gradient_audit(optimizer, W)
        max_gradient = max(max_gradient, grad_value)
        gradients_finite &= grad_finite
        optimizer.step()
        W.dr.eval()
        history.append({"step": step + 1, "loss": loss_value,
                        "train_mse_db2": data_value, "beta_db": beta_value})
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == int(steps):
            print(json.dumps({"material_step": step + 1, "steps": int(steps),
                "train_mse_db2": data_value, "max_gradient": grad_value,
                "elapsed_seconds": time.time() - started}), flush=True)

    final_tensor, final_power = incoherent_path_log_gain(cache.compute_fields(), W)
    final = np.asarray(final_tensor, float).reshape(-1)
    final_mask = np.isfinite(final) & np.isfinite(final_power) & (final_power > 0.0)
    if not np.array_equal(final_mask[:n_train], fixed_mask[:n_train]) or not np.array_equal(
            final_mask[n_train:], query_mask):
        raise RuntimeError("fixed train/query path masks changed during material optimization")
    final = np.where(fixed_mask, final, np.nan)
    final_train_pred = final[:n_train][train_mask]
    beta_final = float(np.mean(train_y[train_mask] - final_train_pred))
    final_train_mse = float(np.mean((final_train_pred + beta_final - train_y[train_mask]) ** 2))
    baseline_beta = float(np.mean(train_y[train_mask] - baseline[:n_train][train_mask]))
    baseline_mse = float(np.mean((baseline[:n_train][train_mask] + baseline_beta - train_y[train_mask]) ** 2))
    final_state = _optimizer_arrays(optimizer)
    max_change = max(float(np.max(np.abs(final_state[k] - initial_state[k]))) for k in final_state)
    if not gradients_finite or not np.isfinite(max_gradient) or max_gradient <= 0.0:
        raise RuntimeError("material technical audit found no positive finite gradient")
    if not np.isfinite(max_change) or max_change <= 0.0:
        raise RuntimeError("material technical audit found no positive finite parameter change")
    table_after = _material_table(fields, optimizer, priors, W, grid)
    field_change = max(
        max(abs(after["eps_r"] - before["eps_r"]),
            abs(after["sigma_s_per_m"] - before["sigma_s_per_m"]))
        for before, after in zip(table_before, table_after)
    )
    fixed_nuisance = all(
        after["scattering_fixed"] == before["scattering_fixed"]
        and after["xpd_fixed"] == before["xpd_fixed"]
        for before, after in zip(table_before, table_after)
    )
    if not np.isfinite(field_change) or field_change <= 0.0:
        raise RuntimeError("material parameters moved but sampled eps/sigma field did not change")
    if not fixed_nuisance:
        raise RuntimeError("fixed scattering/XPD nuisance fields changed")

    params_file = None
    params_hash = None
    metadata_file = None
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        params_path = destination / "material_params.npz"
        np.savez_compressed(params_path, **final_state)
        params_file = str(params_path.resolve())
        params_hash = _sha256(params_path)

    result = {
        "baseline_train": baseline[:n_train].copy(),
        "baseline_query": baseline[n_train:].copy(),
        "material_train": final[:n_train].copy(),
        "material_query": final[n_train:].copy(),
        "coverage": {"train": _coverage(fixed_mask[:n_train]), "query": _coverage(query_mask)},
        "coverage_train": float(fixed_mask[:n_train].mean()),
        "coverage_query": float(query_mask.mean()) if len(query_mask) else 0.0,
        "history": history,
        "metadata": {
            "schema": SCHEMA, "band": band, "frequency_hz": FREQUENCY_HZ[band],
            "seed_uint32": int(np.uint32(seed)), "steps": int(steps), "samples": int(samples),
            "width": int(width), "bands": int(bands), "optimizer_lr": 1e-3,
            "prior_regularization": PRIOR_REGULARIZATION,
            "scene_xml_sha256": scene_hash, "priors_json_sha256": _sha256(priors_json),
            "source_sha256": _sha256(Path(__file__)),
            "prior_artifact_status": prior_payload.get("status"), "trace": trace_config,
            "power_estimator": "incoherent_sum_path(real_squared_plus_imag_squared)",
            "geometry_contract": "one detached combined train/query V7-r4 cache reused before and after updates",
            "supervision_contract": "finite fixed-path train rows only; no query labels accepted",
            "train_mask": fixed_mask[:n_train].tolist(), "query_mask": query_mask.tolist(),
            "baseline_beta_db": baseline_beta, "material_beta_db": beta_final,
            "baseline_train_mse_db2": baseline_mse, "material_train_mse_db2": final_train_mse,
            "training_objective_outcome": (
                "improved" if final_train_mse < baseline_mse else "negative_or_null"
            ),
            "claim_boundary": "effective train-conditioned field; not true EM material recovery",
            "technical_audit": {"max_abs_gradient": max_gradient,
                                "max_abs_parameter_change": max_change,
                                "max_sampled_eps_sigma_change": field_change,
                                "baseline_repeat_stable": True,
                                "fixed_train_query_masks": True,
                                "scattering_xpd_fixed": fixed_nuisance,
                                "finite_nonzero_gradients_and_changes": True},
            "material_sampling_grid": {"points": grid.tolist(), "prior": table_before,
                                       "learned": table_after},
            "material_params_file": params_file,
            "material_params_sha256": params_hash,
            "elapsed_seconds": time.time() - started,
        },
        "material_params_file": params_file,
    }
    if output_dir is not None:
        metadata_path = Path(output_dir) / "material_metadata.json"
        _write_json(metadata_path, result["metadata"])
        metadata_file = str(metadata_path.resolve())
        result["metadata_file"] = metadata_file
    return result


def synthetic_positive_control(scene_xml, priors_json, band, tx, xyz, *, seed,
                               steps=12, samples=50000, width=8, bands=2) -> dict:
    """Same-geometry real-Sionna positive control with known admissible changes.

    A fixed cache is evaluated under alternating admissible eps/sigma interval
    endpoints to make synthetic labels, then a neural field is optimized from
    the engineering prior on that exact detached cache.  No real or query truth
    is accepted.
    """
    xyz = _xyz(xyz, "xyz")
    import wedt_sionna2 as W
    from run_h12_material_smoke import attach_static_truth, build_truth_values

    priors, _, _ = _load_priors(Path(priors_json), str(band).lower(), W)
    scene = _build_prior_scene(Path(scene_xml), xyz, np.asarray(tx, np.float32), str(band).lower(), priors)
    template = W.SemanticFieldSet(BOUNDS_MIN, BOUNDS_MAX, priors, bands=bands, width=width,
                                  rsrp_mode="fixed", eps_residual_fraction=0.35,
                                  sigma_log_residual=1.0)
    truth = build_truth_values(template, priors)
    with W.dr.suspend_grad():
        cache = W.trace_geometry(scene, **_trace_kwargs(seed, samples))
        cache_identity = id(cache)
        prior_db, prior_power = incoherent_path_log_gain(cache.compute_fields(), W)
    attach_static_truth(scene, priors, truth, f"h16_positive_{band}_{seed}")
    truth_db, truth_power = incoherent_path_log_gain(cache.compute_fields(), W)
    prior_np, truth_np = np.asarray(prior_db, float).reshape(-1), np.asarray(truth_db, float).reshape(-1)
    valid = (np.asarray(prior_power) > 0) & (np.asarray(truth_power) > 0)
    if not valid.any() or not np.any(np.abs(truth_np[valid] - prior_np[valid]) > 1e-6):
        raise RuntimeError("synthetic material perturbation has no finite measurable effect")

    # Replace truth materials with exact-prior residual fields without tracing
    # again. Prior, truth, optimization, and final evaluation consequently use
    # one identical candidate/image-method buffer.
    _, optimizer, initial = _attach_fields(
        scene, priors, W, int(seed), int(width), int(bands), 1e-3
    )
    initial_state = _optimizer_arrays(optimizer)
    initial_db, initial_power = incoherent_path_log_gain(cache.compute_fields(), W)
    initial_np = np.asarray(initial_db, float).reshape(-1)
    initial_mask = np.isfinite(initial_np) & (
        np.asarray(initial_power, float).reshape(-1) > 0
    )
    if id(cache) != cache_identity:
        raise RuntimeError("positive-control cache identity changed before fitting")
    if not np.array_equal(initial_mask, valid) or not np.allclose(
            initial_np[valid], prior_np[valid], atol=1e-5, rtol=1e-6):
        raise RuntimeError("positive control does not reproduce the prior on its frozen cache")

    indices = np.flatnonzero(valid).astype(np.uint32)
    target = W.mi.Float(np.asarray(truth_np[valid], np.float32))
    initial_beta = float(np.mean(truth_np[valid] - initial_np[valid]))
    initial_mse = float(np.mean(
        (initial_np[valid] + initial_beta - truth_np[valid]) ** 2
    ))
    max_gradient = 0.0
    gradients_finite = True
    history = []
    for step in range(int(steps)):
        prediction_all, _ = incoherent_path_log_gain(cache.compute_fields(), W)
        prediction = _select(prediction_all, indices, W)
        beta = W.dr.mean(target - prediction)
        data_loss = W.dr.mean(W.dr.square(prediction + beta - target))
        loss = data_loss + _regularization(optimizer, initial, W)
        data_value = float(np.asarray(data_loss).reshape(-1)[0])
        beta_value = float(np.asarray(beta).reshape(-1)[0])
        if not np.isfinite([data_value, beta_value]).all():
            raise RuntimeError("non-finite positive-control objective")
        W.dr.backward(loss)
        grad_value, grad_finite = _gradient_audit(optimizer, W)
        max_gradient = max(max_gradient, grad_value)
        gradients_finite &= grad_finite
        optimizer.step()
        W.dr.eval()
        history.append({"step": step + 1, "train_mse_db2": data_value,
                        "beta_db": beta_value})
        if step == 0 or (step + 1) % 5 == 0 or step + 1 == int(steps):
            print(json.dumps({"positive_control_step": step + 1, "steps": int(steps),
                "train_mse_db2": data_value, "max_gradient": grad_value}), flush=True)

    final_db, final_power = incoherent_path_log_gain(cache.compute_fields(), W)
    final_np = np.asarray(final_db, float).reshape(-1)
    final_mask = np.isfinite(final_np) & (
        np.asarray(final_power, float).reshape(-1) > 0
    )
    if id(cache) != cache_identity or not np.array_equal(final_mask, valid):
        raise RuntimeError("positive-control cache identity/support changed during fitting")
    final_beta = float(np.mean(truth_np[valid] - final_np[valid]))
    final_mse = float(np.mean(
        (final_np[valid] + final_beta - truth_np[valid]) ** 2
    ))
    final_state = _optimizer_arrays(optimizer)
    max_change = max(float(np.max(np.abs(final_state[key] - initial_state[key])))
                     for key in final_state)
    decreased = final_mse < initial_mse
    passed = (decreased and gradients_finite and max_gradient > 0.0
              and np.isfinite(max_change) and max_change > 0.0)
    return {"status": "PASS" if passed else "FAIL_TECHNICAL_CONTROL",
            "finite_count": int(valid.sum()),
            "known_effect_mean_abs_db": float(np.mean(np.abs(truth_np[valid] - prior_np[valid]))),
            "initial_loss_db2": initial_mse, "final_loss_db2": final_mse,
            "positive_gradient": bool(gradients_finite and max_gradient > 0.0),
            "max_abs_gradient": max_gradient,
            "positive_parameter_change": bool(np.isfinite(max_change) and max_change > 0.0),
            "max_abs_parameter_change": max_change,
            "loss_decreased": bool(decreased),
            "same_geometry": True, "same_cache_identity": id(cache) == cache_identity,
            "cache_identity": cache_identity, "history": history,
            "claim_boundary": "synthetic optimizer control only; no real query truth"}


__all__ = ["fit_material", "incoherent_path_log_gain", "synthetic_positive_control"]
