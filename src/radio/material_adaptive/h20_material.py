#!/usr/bin/env python3
"""H20 train-only shared-class material calibration on frozen Sionna paths.

The six frozen ITU classes each receive one bounded permittivity residual and
one bounded log-conductivity residual.  These are effective RSRP-conditioned
parameters, not identifiable site-material measurements.  Query coordinates
are accepted only for post-fit inference; query labels are not part of the API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "wedt_full_v1" / "src"
for candidate in (SRC, VENDOR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

FREQUENCY_HZ = {"n41": 2524.95e6, "n79": 4827.36e6}
EXPECTED_CLASSES = (
    "concrete", "glass", "marble", "medium_dry_ground", "metal", "wet_ground"
)
OFFSET_BRANCHES = ("TWC_CLASS", "TWC_GRID")
RAW_BRANCH = "RAW"
SCHEMA = "h20-shared-itu-class-material-v1"
EPS_RESIDUAL_FRACTION = 0.35
SIGMA_LOG_RESIDUAL = 1.0
PRIOR_REGULARIZATION = 1e-5
OPTIMIZER_LR = 0.03
QUERY_BATCH_SIZE = 2048


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value, dtype=None) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _payload_sha256(payload: Mapping) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(text))


def _xyz(value, name: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value, np.float32)
    if array.ndim != 2 or array.shape[1:] != (3,) or (not allow_empty and not len(array)):
        raise ValueError(f"{name} must have shape (n, 3)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _validate_inputs(scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
                     target_offsets, seed, steps, samples):
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
    if len(train_y) != len(train_xyz) or not np.isfinite(train_y).any():
        raise ValueError("train_y must align with train_xyz and contain a finite label")
    if not isinstance(target_offsets, Mapping) and not callable(target_offsets):
        raise ValueError("target_offsets must be a mapping or baseline callback")
    offsets = (target_offsets if callable(target_offsets)
               else _validate_offsets(target_offsets, len(train_xyz)))
    if not 0 <= int(seed) <= np.iinfo(np.uint32).max:
        raise ValueError("seed must be uint32")
    if int(steps) < 1 or int(samples) < 1:
        raise ValueError("steps and samples must be positive")
    return (scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
            offsets)


def _validate_offsets(offsets, train_n: int) -> dict[str, np.ndarray]:
    if not isinstance(offsets, Mapping) or set(offsets) != set(OFFSET_BRANCHES):
        raise ValueError(f"target_offsets must have exactly {OFFSET_BRANCHES}")
    validated = {}
    for name in OFFSET_BRANCHES:
        value = np.asarray(offsets[name], np.float32).reshape(-1)
        if len(value) != int(train_n) or not np.isfinite(value).all():
            raise ValueError(f"target offset {name} must be finite and aligned with train_xyz")
        validated[name] = value
    return validated


def _load_priors(path: Path, band: str, W):
    from run_h12_material_smoke import load_semantic_priors

    priors, payload = load_semantic_priors(path, band, W)
    return priors, payload, payload.get("scene_xml_sha256")


def _class_mapping(priors, payload) -> dict[str, str]:
    rows = payload.get("objects")
    if not isinstance(rows, list):
        raise RuntimeError("prior artifact lacks object class records")
    mapping = {}
    for row in rows:
        semantic_id, class_name = str(row.get("object", "")), str(row.get("itu_type", ""))
        if semantic_id in priors:
            if not class_name:
                raise RuntimeError(f"missing itu_type for {semantic_id}")
            mapping[semantic_id] = class_name
    if set(mapping) != set(priors):
        raise RuntimeError("prior object/class mapping does not cover all semantic priors")
    if tuple(sorted(set(mapping.values()))) != EXPECTED_CLASSES:
        raise RuntimeError("frozen prior must contain exactly the six H20 ITU classes")
    return mapping


def _build_prior_scene(scene_xml, positions, tx, band, priors):
    from run_h12_material_smoke import attach_static_truth, build_scene

    scene = build_scene(scene_xml, positions, tx, band)
    values = {key: (prior.eps_r, prior.sigma) for key, prior in priors.items()}
    attach_static_truth(scene, priors, values, f"h20_prior_{band}")
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
        "diffraction": False,
        "edge_diffraction": False,
        "seed": int(np.uint32(seed)),
    }


def _derive_seed(seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"H20/{int(seed)}/{tag}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def incoherent_path_log_gain(paths, W):
    real, imag = paths.a
    power = W.dr.square(real) + W.dr.square(imag)
    num_rx, num_paths = int(power.shape[0]), int(power.shape[-1])
    if num_paths == 0 or num_rx == 0:
        # DrJit's reduction over an empty path axis loses the receiver shape.
        # Zero received power is an invalid path, not a synthetic prediction:
        # _finite_gain converts these rows to NaN and retains their row count.
        power = W.dr.zeros(W.mi.TensorXf, shape=(num_rx,))
        gain = W.dr.full(W.mi.TensorXf, -300.0, shape=(num_rx,))
        return gain, np.zeros(num_rx, dtype=float)
    else:
        power = W.dr.reshape(W.mi.TensorXf, power, [num_rx, -1, num_paths])
        power = W.dr.sum(W.dr.sum(power, axis=2), axis=1)
    gain = 10.0 * W.dr.log(W.dr.maximum(power, 1e-30)) / np.log(10.0)
    return gain, np.asarray(power, float).reshape(-1)


def _output_intervals(prior) -> dict[str, tuple[float, float]]:
    radius = EPS_RESIDUAL_FRACTION * (prior.eps_bounds[1] - prior.eps_bounds[0])
    eps = (max(prior.eps_bounds[0], prior.eps_r - radius),
           min(prior.eps_bounds[1], prior.eps_r + radius))
    sigma = (max(prior.sigma_bounds[0], prior.sigma * np.exp(-SIGMA_LOG_RESIDUAL)),
             min(prior.sigma_bounds[1], prior.sigma * np.exp(SIGMA_LOG_RESIDUAL)))
    return {"eps_r": tuple(map(float, eps)), "sigma": tuple(map(float, sigma))}


class _ClassScalarField:
    """Object-specific prior center driven by one shared class residual pair."""

    def __init__(self, prior, class_name: str, W):
        self.prior, self.class_name, self.W = prior, class_name, W
        self.eps_key = f"class.{_safe(class_name)}.eps_raw"
        self.sigma_key = f"class.{_safe(class_name)}.log_sigma_raw"

    @staticmethod
    def _bounded(raw, center: float, interval, W):
        low, high = map(float, interval)
        # NumPy scalars take ownership of mixed arithmetic and coerce DrJit AD
        # values to ndarrays.  Keep every constant on the Python-scalar side so
        # the DrJit overload remains active and the material graph stays live.
        center = float(center)
        value = W.dr.tanh(raw)
        return center + W.dr.select(value >= 0.0, value * (high - center), value * (center - low))

    def __call__(self, _point, params):
        W, prior = self.W, self.prior
        intervals = _output_intervals(prior)
        eps = self._bounded(params[self.eps_key], float(prior.eps_r), intervals["eps_r"], W)
        log_center = float(np.log(float(prior.sigma)))
        delta_log_sigma = self._bounded(
            params[self.sigma_key], 0.0,
            tuple(float(value) - log_center for value in np.log(intervals["sigma"])), W
        )
        # exp(log(sigma0)) can differ from sigma0 by several FP32 ULPs.
        # Express the identical bounded map as a relative factor so raw zero
        # reproduces the static prior exactly, including near critical angles.
        sigma = W.mi.Float(prior.sigma) * W.dr.exp(delta_log_sigma)
        return eps, sigma, W.mi.Float(prior.scattering), W.mi.Float(prior.xpd)


class _ClassFieldSet:
    def __init__(self, priors, class_by_object, W):
        self.fields = {
            semantic_id: _ClassScalarField(prior, class_by_object[semantic_id], W)
            for semantic_id, prior in priors.items()
        }

    def field_for(self, semantic_id):
        return self.fields[semantic_id]


def _initial_arrays() -> dict[str, np.ndarray]:
    arrays = {}
    for class_name in EXPECTED_CLASSES:
        prefix = f"class.{_safe(class_name)}"
        arrays[prefix + ".eps_raw"] = np.zeros(1, np.float32)
        arrays[prefix + ".log_sigma_raw"] = np.zeros(1, np.float32)
    return arrays


def _attach_materials(scene, priors, class_by_object, W, params, tag: str):
    from run_h12_material_smoke import add_and_assign

    fields = _ClassFieldSet(priors, class_by_object, W)
    for semantic_id, prior in priors.items():
        material = W.SemanticSpatialRadioMaterial(
            fields, semantic_id, params, name=f"h20_{_safe(tag)}_{_safe(semantic_id)}",
            relative_permittivity=prior.eps_r, conductivity=prior.sigma,
            scattering_coefficient=prior.scattering, xpd_coefficient=prior.xpd,
        )
        add_and_assign(scene, semantic_id, material)
    return fields


def _fresh_optimizer(scene, priors, class_by_object, W, tag: str):
    arrays = _initial_arrays()
    optimizer = W.mi.ad.Adam(
        lr=OPTIMIZER_LR, params={key: W.mi.Float(value) for key, value in arrays.items()}
    )
    fields = _attach_materials(scene, priors, class_by_object, W, optimizer, tag)
    return fields, optimizer, arrays


def _optimizer_arrays(optimizer) -> dict[str, np.ndarray]:
    return {str(key): np.asarray(optimizer[key], np.float32).copy() for key in optimizer.keys()}


def _select(value, indices: np.ndarray, W):
    return W.dr.gather(W.mi.Float, W.dr.ravel(value), W.mi.UInt(indices.astype(np.uint32)))


def _regularization(optimizer, W):
    terms = [W.dr.mean(W.dr.square(optimizer[key])) for key in optimizer.keys()]
    value = terms[0]
    for term in terms[1:]:
        value = value + term
    return PRIOR_REGULARIZATION * value / len(terms)


def _gradient_audit(optimizer, W) -> tuple[float, bool]:
    maximum, finite = 0.0, True
    for key in optimizer.keys():
        gradient = np.asarray(W.dr.grad(optimizer[key]), float).reshape(-1)
        finite &= bool(np.isfinite(gradient).all())
        if gradient.size:
            maximum = max(maximum, float(np.max(np.abs(gradient))))
    return maximum, finite


def _finite_gain(gain, power) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(gain, float).reshape(-1)
    power = np.asarray(power, float).reshape(-1)
    if value.shape != power.shape:
        raise RuntimeError("receiver gain/power length mismatch")
    mask = np.isfinite(value) & np.isfinite(power) & (power > 0.0)
    return np.where(mask, value, np.nan), mask


def _coverage(mask: np.ndarray) -> dict:
    return {"count": int(mask.sum()), "total": int(len(mask)),
            "fraction": float(mask.mean()) if len(mask) else 0.0}


def _class_table(fields, params, priors, class_by_object, W) -> list[dict]:
    rows = []
    point = W.mi.Point3f(0.0, 0.0, 0.0)
    for semantic_id in sorted(priors):
        values = fields.field_for(semantic_id)(point, params)
        rows.append({
            "semantic_id": semantic_id,
            "itu_type": class_by_object[semantic_id],
            "eps_r": float(np.asarray(values[0]).reshape(-1)[0]),
            "sigma_s_per_m": float(np.asarray(values[1]).reshape(-1)[0]),
            "scattering_fixed": float(priors[semantic_id].scattering),
            "xpd_fixed": float(priors[semantic_id].xpd),
        })
    return rows


def _fit_branch(cache, baseline, fixed_mask, train_y, offset, priors,
                class_by_object, W, name: str, steps: int):
    optimizer_started = time.time()
    print(json.dumps({"event": "h20_branch_start", "branch": name,
                      "steps": int(steps), "fit_path_n": int(fixed_mask.sum())}), flush=True)
    fields, optimizer, initial = _fresh_optimizer(
        cache.scene, priors, class_by_object, W, f"fit_{name}"
    )
    initial_gain, initial_power = incoherent_path_log_gain(cache.compute_fields(), W)
    initial_np, initial_mask = _finite_gain(initial_gain, initial_power)
    if not np.array_equal(initial_mask, fixed_mask) or not np.allclose(
            initial_np[fixed_mask], baseline[fixed_mask], atol=1e-5, rtol=1e-6):
        raise RuntimeError(f"{name}: zero class residual does not reproduce prior")
    fit_mask = fixed_mask & np.isfinite(train_y)
    indices = np.flatnonzero(fit_mask).astype(np.uint32)
    adjusted = np.asarray(train_y - offset, np.float32)
    target = W.mi.Float(adjusted[fit_mask])
    history, max_gradient, gradients_finite = [], 0.0, True
    for step in range(int(steps)):
        all_gain, _ = incoherent_path_log_gain(cache.compute_fields(), W)
        prediction = _select(all_gain, indices, W)
        beta = W.dr.mean(target - prediction)
        data_loss = W.dr.mean(W.dr.square(prediction + beta - target))
        loss = data_loss + _regularization(optimizer, W)
        values = [float(np.asarray(value).reshape(-1)[0]) for value in (loss, data_loss, beta)]
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name}: non-finite objective at step {step + 1}")
        W.dr.backward(loss)
        gradient, finite = _gradient_audit(optimizer, W)
        max_gradient, gradients_finite = max(max_gradient, gradient), gradients_finite and finite
        optimizer.step()
        W.dr.eval()
        history.append({"step": step + 1, "loss": values[0],
                        "train_mse_db2": values[1], "beta_db": values[2]})
        if (step + 1) % 20 == 0 or step + 1 == int(steps):
            print(json.dumps({"event": "h20_branch_progress", "branch": name,
                              "step": step + 1, "steps": int(steps),
                              "train_mse_db2": values[1],
                              "optimizer_elapsed_seconds": time.time() - optimizer_started}),
                  flush=True)
    final_gain, final_power = incoherent_path_log_gain(cache.compute_fields(), W)
    final, final_mask = _finite_gain(final_gain, final_power)
    if not np.array_equal(final_mask, fixed_mask):
        raise RuntimeError(f"{name}: fixed training path mask changed")
    state = _optimizer_arrays(optimizer)
    max_change = max(float(np.max(np.abs(state[key] - initial[key]))) for key in state)
    if not gradients_finite or max_gradient <= 0.0 or max_change <= 0.0:
        raise RuntimeError(f"{name}: no positive finite gradient/parameter change")
    beta_final = float(np.mean(adjusted[fit_mask] - final[fit_mask]))
    baseline_beta = float(np.mean(adjusted[fit_mask] - baseline[fit_mask]))
    baseline_mse = float(np.mean((baseline[fit_mask] + baseline_beta - adjusted[fit_mask]) ** 2))
    final_mse = float(np.mean((final[fit_mask] + beta_final - adjusted[fit_mask]) ** 2))
    return {
        "material_train": final,
        "params": state,
        "history": history,
        "metadata": {
            "target_offset_name": name,
            "target_definition": "train_y_minus_target_offset",
            "baseline_beta_db": baseline_beta,
            "material_beta_db": beta_final,
            "baseline_train_mse_db2": baseline_mse,
            "material_train_mse_db2": final_mse,
            "training_objective_outcome": "improved" if final_mse < baseline_mse else "negative_or_null",
            "max_abs_gradient": max_gradient,
            "max_abs_parameter_change": max_change,
            "optimizer_elapsed_seconds": time.time() - optimizer_started,
            "class_materials": _class_table(fields, optimizer, priors, class_by_object, W),
        },
    }


def _assert_equal_offset_branches(branch_offsets, fitted) -> list[list[str]]:
    """Prove that equal targets yield equal states on the shared cache."""
    groups = []
    names = list(branch_offsets)
    for index, left in enumerate(names):
        group = [left]
        for right in names[index + 1:]:
            if not np.array_equal(branch_offsets[left], branch_offsets[right]):
                continue
            if not np.allclose(fitted[left]["material_train"], fitted[right]["material_train"],
                               equal_nan=True, atol=1e-6, rtol=1e-7):
                raise RuntimeError(f"equal-offset branch equivalence failed: {left}/{right}")
            for key in fitted[left]["params"]:
                if not np.array_equal(fitted[left]["params"][key], fitted[right]["params"][key]):
                    raise RuntimeError(f"equal-offset parameter equivalence failed: {left}/{right}")
            group.append(right)
        if len(group) > 1 and not any(set(group) <= set(existing) for existing in groups):
            groups.append(group)
    return groups


def _query_batches(scene_xml, priors, band, tx, query_xyz, class_by_object,
                   states, W, seed, samples):
    query_started = time.time()
    batch_count = ((len(query_xyz) + QUERY_BATCH_SIZE - 1) // QUERY_BATCH_SIZE
                   if len(query_xyz) else 0)
    print(json.dumps({"event": "h20_query_start", "query_n": len(query_xyz),
                      "query_batch_count": batch_count,
                      "query_batch_size": QUERY_BATCH_SIZE}), flush=True)
    baseline_parts = []
    material_parts = {name: [] for name in states}
    mask_parts = []
    for start in range(0, len(query_xyz), QUERY_BATCH_SIZE):
        stop = min(start + QUERY_BATCH_SIZE, len(query_xyz))
        scene = _build_prior_scene(scene_xml, query_xyz[start:stop], tx, band, priors)
        config = _trace_kwargs(_derive_seed(seed, f"query/{start}/{stop}"), samples)
        with W.dr.suspend_grad():
            cache = W.trace_geometry(scene, **config)
            base_gain, base_power = incoherent_path_log_gain(cache.compute_fields(), W)
            baseline, fixed_mask = _finite_gain(base_gain, base_power)
            repeat_gain, repeat_power = incoherent_path_log_gain(cache.compute_fields(), W)
            repeat, repeat_mask = _finite_gain(repeat_gain, repeat_power)
        if not np.array_equal(fixed_mask, repeat_mask) or not np.allclose(
                baseline[fixed_mask], repeat[fixed_mask], atol=1e-5, rtol=1e-6):
            raise RuntimeError("query baseline repeat stability failed")
        baseline_parts.append(baseline)
        mask_parts.append(fixed_mask)
        for name, state in states.items():
            params = {key: W.mi.Float(value) for key, value in state.items()}
            _attach_materials(scene, priors, class_by_object, W, params, f"query_{name}_{start}")
            gain, power = incoherent_path_log_gain(cache.compute_fields(), W)
            material, material_mask = _finite_gain(gain, power)
            if not np.array_equal(material_mask, fixed_mask):
                raise RuntimeError(f"{name}: fixed query path mask changed")
            material_parts[name].append(material)
    baseline = np.concatenate(baseline_parts) if baseline_parts else np.empty(0, float)
    mask = np.concatenate(mask_parts) if mask_parts else np.empty(0, bool)
    material = {
        name: np.concatenate(parts) if parts else np.empty(0, float)
        for name, parts in material_parts.items()
    }
    elapsed = time.time() - query_started
    print(json.dumps({"event": "h20_query_complete", "query_n": len(query_xyz),
                      "query_batch_count": batch_count,
                      "query_elapsed_seconds": elapsed}), flush=True)
    return baseline, material, mask, elapsed


def validate_metadata(metadata: Mapping) -> None:
    """CPU-only fail-closed validation for persisted fitter metadata."""
    if metadata.get("schema") != SCHEMA:
        raise ValueError("wrong H20 material schema")
    if tuple(metadata.get("itu_classes", ())) != EXPECTED_CLASSES:
        raise ValueError("wrong H20 ITU class order")
    if metadata.get("trainable_parameter_count") != 12:
        raise ValueError("H20 requires exactly twelve trainable scalars per band")
    if tuple(metadata.get("branches", ())) != (RAW_BRANCH, *OFFSET_BRANCHES):
        raise ValueError("wrong H20 branch contract")
    if metadata.get("query_labels_used") is not False:
        raise ValueError("query label isolation is not explicit")
    if metadata.get("training_receiver_strategy") not in {
            "one_full_train_only_cache", "one_scene_chunked_host_geometry_bank"}:
        raise ValueError("unexpected receiver training strategy")
    if metadata.get("optimizer_lr") != OPTIMIZER_LR:
        raise ValueError("wrong frozen H20 optimizer learning rate")
    contract = dict(metadata.get("construction_contract", {}))
    expected = contract.pop("sha256", None)
    if not expected or _payload_sha256(contract) != expected:
        raise ValueError("invalid H20 construction-contract hash")
    if "no serialized cache hash exists" not in contract.get("geometry_provenance", ""):
        raise ValueError("missing in-memory geometry provenance boundary")
    timing = [metadata.get("train_trace_elapsed_seconds"),
              metadata.get("query_inference_elapsed_seconds")]
    branch_timing = metadata.get("branch_optimizer_elapsed_seconds", {})
    if set(branch_timing) != {RAW_BRANCH, *OFFSET_BRANCHES}:
        raise ValueError("wrong per-branch optimizer timing contract")
    timing.extend(branch_timing.values())
    if not np.isfinite(timing).all() or any(float(value) < 0.0 for value in timing):
        raise ValueError("invalid H20 phase timing")


def fit_material_branches(scene_xml, priors_json, band, tx, train_xyz, train_y,
                          query_xyz, *, target_offsets, seed, steps=80,
                          samples=50000, output_dir=None) -> dict:
    """Fit RAW, TWC_CLASS, and TWC_GRID material branches from training labels."""
    started = time.time()
    (scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
     offsets) = _validate_inputs(scene_xml, priors_json, band, tx, train_xyz,
                                 train_y, query_xyz, target_offsets, seed, steps, samples)
    import wedt_sionna2 as W

    priors, prior_payload, expected_scene_hash = _load_priors(priors_json, band, W)
    scene_hash = _sha256(scene_xml)
    if expected_scene_hash and expected_scene_hash != scene_hash:
        raise RuntimeError("scene hash differs from frozen prior contract")
    class_by_object = _class_mapping(priors, prior_payload)
    scene = _build_prior_scene(scene_xml, train_xyz, tx, band, priors)
    trace = _trace_kwargs(seed, samples)
    trace_started = time.time()
    print(json.dumps({"event": "h20_trace_start", "band": band,
                      "train_n": len(train_xyz), "query_n": len(query_xyz),
                      "samples_per_src": int(samples)}), flush=True)
    with W.dr.suspend_grad():
        cache = W.trace_geometry(scene, **trace)
        cache_identity = id(cache)
        baseline_gain, baseline_power = incoherent_path_log_gain(cache.compute_fields(), W)
        baseline, train_path_mask = _finite_gain(baseline_gain, baseline_power)
        repeat_gain, repeat_power = incoherent_path_log_gain(cache.compute_fields(), W)
        repeat, repeat_mask = _finite_gain(repeat_gain, repeat_power)
    if not np.array_equal(train_path_mask, repeat_mask) or not np.allclose(
            baseline[train_path_mask], repeat[train_path_mask], atol=1e-5, rtol=1e-6):
        raise RuntimeError("training baseline repeat stability failed")
    trace_elapsed_seconds = time.time() - trace_started
    print(json.dumps({"event": "h20_trace_complete", "band": band,
                      "train_n": len(train_xyz),
                      "train_path_n": int(train_path_mask.sum()),
                      "trace_elapsed_seconds": trace_elapsed_seconds}), flush=True)
    if callable(offsets):
        # The callback receives only the newly traced raw training baseline.
        # It may close over training-side coordinates/labels to build TWC
        # corrections, but query labels are structurally absent from this API.
        offsets = _validate_offsets(offsets(baseline.copy()), len(train_xyz))
    fit_mask = train_path_mask & np.isfinite(train_y)
    if not fit_mask.any():
        raise RuntimeError("no finite fixed-path training rows")

    branch_offsets = {RAW_BRANCH: np.zeros(len(train_xyz), np.float32), **offsets}
    fitted = {}
    for name, offset in branch_offsets.items():
        if id(cache) != cache_identity:
            raise RuntimeError("training cache identity changed across branches")
        fitted[name] = _fit_branch(
            cache, baseline, train_path_mask, train_y, offset, priors,
            class_by_object, W, name, int(steps)
        )
    equivalent_groups = _assert_equal_offset_branches(branch_offsets, fitted)
    states = {name: result["params"] for name, result in fitted.items()}
    baseline_query, material_query, query_path_mask, query_elapsed_seconds = _query_batches(
        scene_xml, priors, band, tx, query_xyz, class_by_object, states,
        W, int(seed), int(samples)
    )

    output = Path(output_dir) if output_dir is not None else None
    parameter_files = {}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for name, state in states.items():
            path = output / f"material_params_{name.lower()}.npz"
            np.savez_compressed(path, **state)
            parameter_files[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    branches = {}
    for name, result in fitted.items():
        branches[name] = {
            "baseline_train": baseline.copy(),
            "baseline_query": baseline_query.copy(),
            "material_train": result["material_train"].copy(),
            "material_query": material_query[name].copy(),
            "history": result["history"],
            "metadata": result["metadata"],
        }
    construction_contract = {
        "band": band,
        "frequency_hz": FREQUENCY_HZ[band],
        "scene_xml_sha256": scene_hash,
        "priors_json_sha256": _sha256(priors_json),
        "tx_float32_sha256": _array_sha256(tx, np.float32),
        "train_xyz_float32_sha256": _array_sha256(train_xyz, np.float32),
        "query_xyz_float32_sha256": _array_sha256(query_xyz, np.float32),
        "seed_uint32": int(np.uint32(seed)),
        "trace": trace,
        "baseline_power_float64_sha256": _array_sha256(baseline_power, np.float64),
        "train_path_mask_bool_sha256": _array_sha256(train_path_mask, np.bool_),
        "geometry_provenance": (
            "detached in-memory candidate/image-method geometry returned by "
            "wedt_sionna2.trace_geometry; no serialized cache hash exists"
        ),
    }
    construction_contract["sha256"] = _payload_sha256(construction_contract)
    metadata = {
        "schema": SCHEMA,
        "band": band,
        "frequency_hz": FREQUENCY_HZ[band],
        "seed_uint32": int(np.uint32(seed)),
        "steps": int(steps),
        "samples": int(samples),
        "branches": [RAW_BRANCH, *OFFSET_BRANCHES],
        "itu_classes": list(EXPECTED_CLASSES),
        "class_by_object": class_by_object,
        "trainable_parameter_count": len(_initial_arrays()),
        "parameterization": "six shared ITU-class bounded eps/log-sigma residual pairs",
        "eps_residual_fraction": EPS_RESIDUAL_FRACTION,
        "sigma_log_residual": SIGMA_LOG_RESIDUAL,
        "scattering_xpd_fixed": True,
        "optimizer_lr": OPTIMIZER_LR,
        "prior_regularization": PRIOR_REGULARIZATION,
        "scene_xml_sha256": scene_hash,
        "priors_json_sha256": _sha256(priors_json),
        "construction_contract": construction_contract,
        "source_sha256": _sha256(Path(__file__)),
        "prior_artifact_status": prior_payload.get("status"),
        "trace": trace,
        "power_estimator": "incoherent_sum_path(real_squared_plus_imag_squared)",
        "query_labels_used": False,
        "supervision_contract": "finite fixed-path training rows only; query API has no labels",
        "training_receiver_strategy": "one_full_train_only_cache",
        "query_receiver_strategy": f"frozen-parameter batches of at most {QUERY_BATCH_SIZE}",
        "minibatch_contract": (
            "not activated; any future train batching requires two-pass global beta, "
            "count-weighted gradient accumulation, and one optimizer update per epoch"
        ),
        "same_training_cache_all_branches": id(cache) == cache_identity,
        "equal_offset_branch_groups_verified": equivalent_groups,
        "train_trace_elapsed_seconds": trace_elapsed_seconds,
        "query_inference_elapsed_seconds": query_elapsed_seconds,
        "branch_optimizer_elapsed_seconds": {
            name: result["metadata"]["optimizer_elapsed_seconds"]
            for name, result in fitted.items()
        },
        "train_path_mask": train_path_mask.tolist(),
        "fit_mask": fit_mask.tolist(),
        "query_path_mask": query_path_mask.tolist(),
        "coverage": {"train": _coverage(train_path_mask), "query": _coverage(query_path_mask)},
        "branch_metadata": {name: result["metadata"] for name, result in fitted.items()},
        "parameter_files": parameter_files,
        "claim_boundary": "effective train-conditioned shared-class correction; not true EM material recovery",
        "scalability_boundary": "full approximately-15000-receiver training cache requires a server GPU memory probe",
        "elapsed_seconds": time.time() - started,
    }
    validate_metadata(metadata)
    result = {
        "branches": branches,
        "coverage": metadata["coverage"],
        "masks": {"train_path": train_path_mask.copy(), "train_fit": fit_mask.copy(),
                  "query_path": query_path_mask.copy()},
        "metadata": metadata,
    }
    if output is not None:
        metadata_path = output / "material_metadata.json"
        _write_json(metadata_path, metadata)
        result["metadata_file"] = str(metadata_path.resolve())
    return result


def synthetic_same_cache_positive_control(scene_xml, priors_json, band, tx, xyz, *,
                                          seed, steps=30, samples=50000) -> dict:
    """Real-GPU same-cache control with synthetic shared-class material truth."""
    xyz = _xyz(xyz, "xyz")
    import wedt_sionna2 as W

    priors, payload, _ = _load_priors(Path(priors_json), str(band).lower(), W)
    mapping = _class_mapping(priors, payload)
    scene = _build_prior_scene(Path(scene_xml), xyz, np.asarray(tx, np.float32),
                               str(band).lower(), priors)
    cache = W.trace_geometry(scene, **_trace_kwargs(seed, samples))
    cache_identity = id(cache)
    with W.dr.suspend_grad():
        prior_gain, prior_power = incoherent_path_log_gain(cache.compute_fields(), W)
    truth_state = _initial_arrays()
    for index, class_name in enumerate(EXPECTED_CLASSES):
        sign = 0.8 if index % 2 == 0 else -0.8
        truth_state[f"class.{_safe(class_name)}.eps_raw"][:] = sign
        truth_state[f"class.{_safe(class_name)}.log_sigma_raw"][:] = -sign
    truth_params = {key: W.mi.Float(value) for key, value in truth_state.items()}
    _attach_materials(scene, priors, mapping, W, truth_params, "positive_truth")
    truth_gain, truth_power = incoherent_path_log_gain(cache.compute_fields(), W)
    prior, prior_mask = _finite_gain(prior_gain, prior_power)
    truth, truth_mask = _finite_gain(truth_gain, truth_power)
    if not np.array_equal(prior_mask, truth_mask):
        raise RuntimeError("synthetic material perturbation changed the fixed path mask")
    valid = prior_mask
    if not valid.any() or not np.any(np.abs(truth[valid] - prior[valid]) > 1e-6):
        raise RuntimeError("synthetic shared-class perturbation has no measurable effect")
    fitted = _fit_branch(cache, prior, valid, truth, np.zeros(len(xyz), np.float32),
                         priors, mapping, W, "POSITIVE", int(steps))
    initial_beta = float(np.mean(truth[valid] - prior[valid]))
    initial_mse = float(np.mean((prior[valid] + initial_beta - truth[valid]) ** 2))
    final = fitted["material_train"]
    final_beta = float(np.mean(truth[valid] - final[valid]))
    final_mse = float(np.mean((final[valid] + final_beta - truth[valid]) ** 2))
    passed = id(cache) == cache_identity and final_mse < initial_mse
    return {
        "status": "PASS" if passed else "FAIL_TECHNICAL_CONTROL",
        "same_cache_identity": id(cache) == cache_identity,
        "cache_identity": cache_identity,
        "finite_count": int(valid.sum()),
        "known_effect_mean_abs_db": float(np.mean(np.abs(truth[valid] - prior[valid]))),
        "initial_loss_db2": initial_mse,
        "final_loss_db2": final_mse,
        "loss_decreased": bool(final_mse < initial_mse),
        "positive_gradient": fitted["metadata"]["max_abs_gradient"] > 0.0,
        "positive_parameter_change": fitted["metadata"]["max_abs_parameter_change"] > 0.0,
        "parameter_count": len(fitted["params"]),
        "claim_boundary": "synthetic optimizer control only; no real or query truth",
    }


__all__ = [
    "fit_material_branches", "incoherent_path_log_gain",
    "synthetic_same_cache_positive_control", "validate_metadata",
]
