#!/usr/bin/env python3
"""Full-P H20 material fitting over one in-process host geometry bank.

This module is the scalable counterpart of :mod:`h20_material`.  It keeps one
parsed scene/BVH, stores immutable proposal geometry in host memory, and
hydrates at most one receiver chunk at a time.  Every epoch uses one global
power intercept, a globally normalized data loss, accumulated chunk gradients,
one regularizer evaluation, and one Adam update.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from . import h20_geometry_bank as geometry_bank
from . import h20_material as material


TRAINING_STRATEGY = "one_scene_chunked_host_geometry_bank"
PATH_CAPACITY = 500_000
RECEIVER_CAPACITY = 256


class SceneReceiverBank:
    """Own the exact active receiver list in one already parsed scene."""

    capacity = RECEIVER_CAPACITY
    _prefix = "h12_material_rx"

    def __init__(self, scene, initial_count: int):
        from sionna.rt import Receiver

        self.scene = scene
        self.Receiver = Receiver
        self._count = int(initial_count)
        if not 1 <= self._count <= self.capacity:
            raise ValueError("initial receiver count must be in [1, 256]")
        if len(scene.receivers) != self._count:
            raise RuntimeError("initial scene receiver list is not owned by the bank")

    def set_chunk(self, xyz: np.ndarray) -> None:
        points = np.asarray(xyz, np.float32)
        if (points.ndim != 2 or points.shape[1] != 3
                or not 1 <= len(points) <= self.capacity
                or not np.isfinite(points).all()):
            raise ValueError("receiver chunk must contain 1..256 finite xyz rows")
        receivers = self.scene.receivers
        shared = min(self._count, len(points))
        for index in range(shared):
            receivers[f"{self._prefix}{index:05d}"].position = points[index].tolist()
        for index in range(self._count - 1, len(points) - 1, -1):
            self.scene.remove(f"{self._prefix}{index:05d}")
        for index in range(self._count, len(points)):
            self.scene.add(self.Receiver(
                name=f"{self._prefix}{index:05d}", position=points[index].tolist()
            ))
        self._count = len(points)
        if len(self.scene.receivers) != self._count:
            raise RuntimeError("scene receiver list does not match active bank chunk")


def _validate_row_ids(row_ids: Sequence[str], train_n: int) -> tuple[str, ...]:
    values = tuple(str(value) for value in row_ids)
    if len(values) != int(train_n) or len(set(values)) != len(values):
        raise ValueError("train_row_ids must be unique and align with train_xyz")
    return values


def _global_beta(prediction: np.ndarray, target: np.ndarray,
                 fit_mask: np.ndarray) -> float:
    """One detached global intercept; its omitted derivative cancels exactly."""
    if not np.asarray(fit_mask, bool).any():
        raise RuntimeError("no finite fixed-path training rows")
    return float(np.mean(np.asarray(target)[fit_mask] - np.asarray(prediction)[fit_mask]))


def _global_centered_mse(prediction: np.ndarray, target: np.ndarray,
                         fit_mask: np.ndarray, beta: float) -> float:
    residual = (np.asarray(prediction)[fit_mask] + float(beta)
                - np.asarray(target)[fit_mask])
    return float(np.mean(residual ** 2))


def _streamed_centered_mse(prediction: np.ndarray, target: np.ndarray,
                           fit_mask: np.ndarray, beta: float, blocks) -> float:
    """Production FULL-P normalization expressed as deterministic block sums."""
    fit_n = int(np.asarray(fit_mask, bool).sum())
    if fit_n == 0:
        raise RuntimeError("no finite fixed-path training rows")
    total = 0.0
    for block in blocks:
        local = np.asarray(fit_mask, bool)[block.start:block.stop]
        residual = (np.asarray(prediction)[block.start:block.stop][local] + float(beta)
                    - np.asarray(target)[block.start:block.stop][local])
        total += float(np.sum(residual ** 2))
    return total / fit_n


def _clear_optimizer_gradients(optimizer, W) -> None:
    for key in optimizer.keys():
        W.dr.clear_grad(optimizer[key])


def _read_optimizer_gradients(optimizer, W) -> tuple[dict[str, np.ndarray], float, bool]:
    values, maximum, finite = {}, 0.0, True
    for key in optimizer.keys():
        gradient = np.asarray(W.dr.grad(optimizer[key]), np.float32).reshape(-1).copy()
        values[str(key)] = gradient
        finite &= bool(np.isfinite(gradient).all())
        if gradient.size:
            maximum = max(maximum, float(np.max(np.abs(gradient))))
    return values, maximum, finite


def _add_gradients(total: dict[str, np.ndarray], values: Mapping[str, np.ndarray]) -> None:
    for key, value in values.items():
        total[key] += np.asarray(value, np.float32)


def _combine_epoch_gradients(chunk_gradients, regularizer_gradients):
    """Combine every data chunk and the single epoch regularizer exactly once."""
    combined = {
        str(key): np.zeros_like(np.asarray(value, np.float32))
        for key, value in regularizer_gradients.items()
    }
    for gradients in chunk_gradients:
        _add_gradients(combined, gradients)
    _add_gradients(combined, regularizer_gradients)
    return combined


def _install_gradients(optimizer, gradients: Mapping[str, np.ndarray], W) -> None:
    for key in optimizer.keys():
        W.dr.set_grad(optimizer[key], W.mi.Float(gradients[str(key)]))


def _apply_streamed_epoch_update(optimizer, chunk_gradients, W):
    """Add one regularizer, audit the applied FP32 sum, then take one Adam step."""
    _clear_optimizer_gradients(optimizer, W)
    regularizer = material._regularization(optimizer, W)
    regularizer_value = float(np.asarray(regularizer).reshape(-1)[0])
    W.dr.backward(regularizer)
    regularizer_gradients, _, regularizer_finite = _read_optimizer_gradients(
        optimizer, W
    )
    combined = _combine_epoch_gradients(chunk_gradients, regularizer_gradients)
    combined_finite = all(np.isfinite(value).all() for value in combined.values())
    combined_max = max(
        (float(np.max(np.abs(value))) for value in combined.values() if value.size),
        default=0.0,
    )
    if not regularizer_finite or not combined_finite or not np.isfinite(regularizer_value):
        raise RuntimeError("non-finite regularizer or accumulated FP32 gradient")
    _clear_optimizer_gradients(optimizer, W)
    _install_gradients(optimizer, combined, W)
    optimizer.step()
    W.dr.eval()
    return regularizer_value, combined, combined_max


def _evaluate_bank(bank, W, *, expected_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gains, powers, masks = [], [], []
    for ordinal, block in enumerate(bank.blocks):
        with bank.hydrate(ordinal) as cache:
            gain, power = material.incoherent_path_log_gain(cache.compute_fields(), W)
            gain = np.asarray(gain, float).reshape(-1)
            power = np.asarray(power, float).reshape(-1)
        expected = block.stop - block.start
        if len(gain) != expected or len(power) != expected:
            raise RuntimeError("hydrated block returned wrong receiver count")
        gains.append(gain)
        powers.append(power)
        masks.append(np.isfinite(gain) & np.isfinite(power) & (power > 0.0))
    combined_gain = np.concatenate(gains)
    combined_power = np.concatenate(powers)
    combined_mask = np.concatenate(masks)
    if len(combined_gain) != int(expected_n):
        raise RuntimeError("geometry bank evaluation omitted training rows")
    return np.where(combined_mask, combined_gain, np.nan), combined_power, combined_mask


def _query_batches(scene_xml, priors, band, tx, query_xyz, class_by_object,
                   states, W, seed, samples):
    """Evaluate Q without labels on immutable proposals of at most 256 receivers."""
    started = time.time()
    if not len(query_xyz):
        empty = np.empty(0, float)
        return empty, {name: empty.copy() for name in states}, np.empty(0, bool), 0.0
    initial_n = min(RECEIVER_CAPACITY, len(query_xyz))
    scene = material._build_prior_scene(
        scene_xml, query_xyz[:initial_n], tx, band, priors
    )
    controller = SceneReceiverBank(scene, initial_n)
    baseline_parts, mask_parts = [], []
    material_parts = {name: [] for name in states}

    def evaluate_range(start: int, stop: int, chunk_size: int) -> None:
        cursor = start
        while cursor < stop:
            end = min(cursor + chunk_size, stop)
            controller.set_chunk(query_xyz[cursor:end])
            zero_params = {
                key: W.mi.Float(value) for key, value in material._initial_arrays().items()
            }
            material._attach_materials(
                scene, priors, class_by_object, W, zero_params,
                f"banked_query_prior_{cursor}_{end}"
            )
            config = material._trace_kwargs(
                material._derive_seed(seed, f"query/{cursor}/{end}"), samples
            )
            config["max_num_paths_per_src"] = PATH_CAPACITY
            with W.dr.suspend_grad():
                cache = W.trace_geometry(scene, **config)
                observed = geometry_bank._stored_path_count(cache.paths_buffer)
            if observed >= PATH_CAPACITY:
                cache = None
                if chunk_size <= geometry_bank.MIN_RECEIVERS:
                    raise RuntimeError(
                        "RESOURCE_NO_GO: query path capacity hit at 64 receivers"
                    )
                evaluate_range(
                    cursor, end,
                    128 if chunk_size > 128 else geometry_bank.MIN_RECEIVERS
                )
                cursor = end
                continue
            with W.dr.suspend_grad():
                base_gain, base_power = material.incoherent_path_log_gain(
                    cache.compute_fields(), W
                )
                baseline, fixed_mask = material._finite_gain(base_gain, base_power)
                repeat_gain, repeat_power = material.incoherent_path_log_gain(
                    cache.compute_fields(), W
                )
                repeat, repeat_mask = material._finite_gain(repeat_gain, repeat_power)
            if (not np.array_equal(fixed_mask, repeat_mask)
                    or not np.allclose(baseline[fixed_mask], repeat[fixed_mask],
                                       atol=1e-5, rtol=1e-6)):
                raise RuntimeError("query baseline repeat stability failed")
            baseline_parts.append(baseline)
            mask_parts.append(fixed_mask)
            for name, state in states.items():
                params = {key: W.mi.Float(value) for key, value in state.items()}
                material._attach_materials(
                    scene, priors, class_by_object, W, params,
                    f"banked_query_{name}_{cursor}_{end}"
                )
                gain, power = material.incoherent_path_log_gain(cache.compute_fields(), W)
                prediction, prediction_mask = material._finite_gain(gain, power)
                if not np.array_equal(prediction_mask, fixed_mask):
                    raise RuntimeError(f"{name}: fixed query path mask changed")
                material_parts[name].append(prediction)
            cache = None
            cursor = end

    evaluate_range(0, len(query_xyz), RECEIVER_CAPACITY)
    baseline = np.concatenate(baseline_parts)
    mask = np.concatenate(mask_parts)
    predictions = {name: np.concatenate(parts) for name, parts in material_parts.items()}
    if len(baseline) != len(query_xyz):
        raise RuntimeError("query chunking omitted or duplicated rows")
    return baseline, predictions, mask, time.time() - started


def _fit_branch(bank, baseline, fixed_mask, train_y, offset, priors,
                class_by_object, W, name: str, steps: int):
    """Two-pass global objective with host-accumulated chunk gradients."""
    started = time.time()
    fields, optimizer, initial = material._fresh_optimizer(
        bank.scene, priors, class_by_object, W, f"banked_fit_{name}"
    )
    initial_gain, _, initial_mask = _evaluate_bank(bank, W, expected_n=len(train_y))
    if (not np.array_equal(initial_mask, fixed_mask)
            or not np.allclose(initial_gain[fixed_mask], baseline[fixed_mask],
                               atol=1e-5, rtol=1e-6)):
        raise RuntimeError(f"{name}: zero class residual does not reproduce bank prior")

    adjusted = np.asarray(train_y - offset, np.float32)
    fit_mask = fixed_mask & np.isfinite(adjusted)
    fit_n = int(fit_mask.sum())
    if fit_n == 0:
        raise RuntimeError(f"{name}: no finite fixed-path training rows")
    history, max_gradient, gradients_finite = [], 0.0, True
    parameter_names = tuple(str(key) for key in optimizer.keys())

    for step in range(int(steps)):
        # Pass one computes the single FULL-P intercept and objective value.
        with W.dr.suspend_grad():
            prediction, _, current_mask = _evaluate_bank(
                bank, W, expected_n=len(train_y)
            )
        if not np.array_equal(current_mask, fixed_mask):
            raise RuntimeError(f"{name}: fixed training path mask changed")
        beta = _global_beta(prediction, adjusted, fit_mask)
        data_mse = _streamed_centered_mse(
            prediction, adjusted, fit_mask, beta, bank.blocks
        )

        chunk_gradients = []
        # Pass two accumulates sum-of-squares/N gradients.  No per-block beta,
        # normalization, regularizer, or optimizer update is allowed here.
        for ordinal, block in enumerate(bank.blocks):
            local_mask = fit_mask[block.start:block.stop]
            if not local_mask.any():
                continue
            local_indices = np.flatnonzero(local_mask).astype(np.uint32)
            local_target = W.mi.Float(adjusted[block.start:block.stop][local_mask])
            _clear_optimizer_gradients(optimizer, W)
            with bank.hydrate(ordinal) as cache:
                gain, _ = material.incoherent_path_log_gain(cache.compute_fields(), W)
                selected = material._select(gain, local_indices, W)
                residual = selected + float(beta) - local_target
                chunk_loss = W.dr.sum(W.dr.square(residual)) / fit_n
                W.dr.backward(chunk_loss)
                gradients, chunk_max, finite = _read_optimizer_gradients(optimizer, W)
            chunk_gradients.append(gradients)
            max_gradient = max(max_gradient, chunk_max)
            gradients_finite &= finite

        # The prior penalty and optimizer update occur exactly once per epoch.
        regularizer_value, _, applied_max = _apply_streamed_epoch_update(
            optimizer, chunk_gradients, W
        )
        max_gradient = max(max_gradient, applied_max)
        if not gradients_finite or not np.isfinite(data_mse + regularizer_value):
            raise RuntimeError(f"{name}: non-finite streamed objective/gradient")
        history.append({"step": step + 1, "loss": data_mse + regularizer_value,
                        "train_mse_db2": data_mse, "beta_db": beta})
        if (step + 1) % 20 == 0 or step + 1 == int(steps):
            print(json.dumps({
                "event": "h20_banked_branch_progress", "branch": name,
                "step": step + 1, "steps": int(steps),
                "train_mse_db2": data_mse,
                "optimizer_elapsed_seconds": time.time() - started,
            }), flush=True)

    final, _, final_mask = _evaluate_bank(bank, W, expected_n=len(train_y))
    if not np.array_equal(final_mask, fixed_mask):
        raise RuntimeError(f"{name}: final fixed training path mask changed")
    state = material._optimizer_arrays(optimizer)
    max_change = max(float(np.max(np.abs(state[key] - initial[key]))) for key in state)
    if max_gradient <= 0.0 or max_change <= 0.0:
        raise RuntimeError(f"{name}: no positive finite gradient/parameter change")
    beta_final = _global_beta(final, adjusted, fit_mask)
    baseline_beta = _global_beta(baseline, adjusted, fit_mask)
    baseline_mse = _global_centered_mse(baseline, adjusted, fit_mask, baseline_beta)
    final_mse = _global_centered_mse(final, adjusted, fit_mask, beta_final)
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
            "training_objective_outcome": (
                "improved" if final_mse < baseline_mse else "negative_or_null"
            ),
            "max_abs_gradient": max_gradient,
            "max_abs_parameter_change": max_change,
            "optimizer_elapsed_seconds": time.time() - started,
            "class_materials": material._class_table(
                fields, optimizer, priors, class_by_object, W
            ),
            "global_fit_n": fit_n,
            "global_beta_once_per_epoch": True,
            "global_mse_normalization": fit_n,
            "optimizer_updates": int(steps),
            "regularizer_evaluations": int(steps),
        },
    }


def _fit_or_reuse_branches(branch_offsets, fit_one, *, deduplicate: bool):
    """Fit each target, or deep-copy the first exactly equal fitted target."""
    fitted, reused_from = {}, {}
    for name, offset in branch_offsets.items():
        source = next((existing for existing, existing_offset in branch_offsets.items()
                       if existing in fitted
                       and np.array_equal(offset, existing_offset)), None)
        if bool(deduplicate) and source is not None:
            fitted[name] = copy.deepcopy(fitted[source])
            fitted[name]["metadata"]["target_offset_name"] = name
            fitted[name]["metadata"]["optimizer_reused_from"] = source
            fitted[name]["metadata"]["optimizer_elapsed_seconds"] = 0.0
            reused_from[name] = source
        else:
            fitted[name] = fit_one(name, offset)
    return fitted, reused_from


def fit_material_branches_banked(scene_xml, priors_json, band, tx, train_xyz,
                                 train_y, query_xyz, *, train_row_ids,
                                 target_offsets, seed, steps=80, samples=50000,
                                 output_dir=None,
                                 deduplicate_equal_targets=False) -> dict:
    """Fit the three frozen H20 arms from full legal P without reading Q labels."""
    started = time.time()
    (scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
     offsets) = material._validate_inputs(
        scene_xml, priors_json, band, tx, train_xyz, train_y, query_xyz,
        target_offsets, seed, steps, samples
    )
    row_ids = _validate_row_ids(train_row_ids, len(train_xyz))
    import wedt_sionna2 as W

    priors, prior_payload, expected_scene_hash = material._load_priors(priors_json, band, W)
    scene_hash = material._sha256(scene_xml)
    if expected_scene_hash and expected_scene_hash != scene_hash:
        raise RuntimeError("scene hash differs from frozen prior contract")
    mapping = material._class_mapping(priors, prior_payload)
    first_n = min(RECEIVER_CAPACITY, len(train_xyz))
    scene = material._build_prior_scene(scene_xml, train_xyz[:first_n], tx, band, priors)
    controller = SceneReceiverBank(scene, first_n)
    trace = material._trace_kwargs(int(seed), int(samples))
    trace["max_num_paths_per_src"] = PATH_CAPACITY
    trace_started = time.time()
    bank = geometry_bank.build_geometry_bank(
        scene, controller, row_ids, train_xyz, W=W, trace_kwargs=trace,
        round_trip_verifier=__import__(
            "probe_h20_geometry_bank", fromlist=["BankVerifier"]
        ).BankVerifier(np.asarray(row_ids), material, W)
    )
    trace_elapsed = time.time() - trace_started
    bank_manifest_before = bank.private_manifest()
    bank_hash_before = material._payload_sha256(bank_manifest_before)

    with W.dr.suspend_grad():
        baseline, baseline_power, train_path_mask = _evaluate_bank(
            bank, W, expected_n=len(train_xyz)
        )
        repeat, _, repeat_mask = _evaluate_bank(bank, W, expected_n=len(train_xyz))
    if (not np.array_equal(train_path_mask, repeat_mask)
            or not np.allclose(baseline[train_path_mask], repeat[train_path_mask],
                               atol=1e-5, rtol=1e-6)):
        raise RuntimeError("banked training baseline repeat stability failed")
    if callable(offsets):
        offsets = material._validate_offsets(offsets(baseline.copy()), len(train_xyz))
    fit_mask = train_path_mask & np.isfinite(train_y)
    if not fit_mask.any():
        raise RuntimeError("no finite fixed-path training rows")

    branch_offsets = {
        material.RAW_BRANCH: np.zeros(len(train_xyz), np.float32), **offsets
    }
    fitted, reused_from = _fit_or_reuse_branches(
        branch_offsets,
        lambda name, offset: _fit_branch(
            bank, baseline, train_path_mask, train_y, offset,
            priors, mapping, W, name, int(steps)
        ),
        deduplicate=bool(deduplicate_equal_targets),
    )
    equivalent = material._assert_equal_offset_branches(branch_offsets, fitted)
    states = {name: result["params"] for name, result in fitted.items()}
    baseline_query, material_query, query_mask, query_elapsed = _query_batches(
        scene_xml, priors, band, tx, query_xyz, mapping, states, W,
        int(seed), int(samples)
    )
    bank_hash_after = material._payload_sha256(bank.private_manifest())
    if bank_hash_after != bank_hash_before:
        raise RuntimeError("immutable host geometry bank changed during fitting")

    output = Path(output_dir) if output_dir is not None else None
    parameter_files = {}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for name, state in states.items():
            path = output / f"material_params_{name.lower()}.npz"
            np.savez_compressed(path, **state)
            parameter_files[name] = {
                "path": str(path.resolve()), "sha256": material._sha256(path)
            }

    construction = {
        "band": band,
        "frequency_hz": material.FREQUENCY_HZ[band],
        "scene_xml_sha256": scene_hash,
        "priors_json_sha256": material._sha256(priors_json),
        "tx_float32_sha256": material._array_sha256(tx, np.float32),
        "train_xyz_float32_sha256": material._array_sha256(train_xyz, np.float32),
        "train_row_ids_sha256": material._payload_sha256(list(row_ids)),
        "query_xyz_float32_sha256": material._array_sha256(query_xyz, np.float32),
        "seed_uint32": int(np.uint32(seed)),
        "trace": trace,
        "baseline_power_float64_sha256": material._array_sha256(
            baseline_power, np.float64
        ),
        "train_path_mask_bool_sha256": material._array_sha256(
            train_path_mask, np.bool_
        ),
        "host_geometry_bank_manifest_sha256": bank_hash_before,
        "geometry_bank_source_sha256": material._sha256(
            Path(geometry_bank.__file__).resolve()
        ),
        "material_source_sha256": material._sha256(Path(material.__file__).resolve()),
        "adapter_source_sha256": material._sha256(Path(W.__file__).resolve()),
        "geometry_provenance": (
            "one parsed scene with immutable process-local host geometry chunks; "
            "no serialized cache hash exists and no cross-process resume is claimed"
        ),
    }
    construction["sha256"] = material._payload_sha256(construction)
    metadata = {
        "schema": material.SCHEMA,
        "band": band,
        "frequency_hz": material.FREQUENCY_HZ[band],
        "seed_uint32": int(np.uint32(seed)),
        "steps": int(steps),
        "samples": int(samples),
        "branches": [material.RAW_BRANCH, *material.OFFSET_BRANCHES],
        "itu_classes": list(material.EXPECTED_CLASSES),
        "class_by_object": mapping,
        "trainable_parameter_count": len(material._initial_arrays()),
        "parameterization": "six shared ITU-class bounded eps/log-sigma residual pairs",
        "eps_residual_fraction": material.EPS_RESIDUAL_FRACTION,
        "sigma_log_residual": material.SIGMA_LOG_RESIDUAL,
        "scattering_xpd_fixed": True,
        "optimizer_lr": material.OPTIMIZER_LR,
        "prior_regularization": material.PRIOR_REGULARIZATION,
        "scene_xml_sha256": scene_hash,
        "priors_json_sha256": material._sha256(priors_json),
        "construction_contract": construction,
        "source_sha256": material._sha256(Path(__file__)),
        "prior_artifact_status": prior_payload.get("status"),
        "trace": trace,
        "power_estimator": "incoherent_sum_path(real_squared_plus_imag_squared)",
        "query_labels_used": False,
        "supervision_contract": "finite full-P fixed-path rows only; query API has no labels",
        "training_receiver_strategy": TRAINING_STRATEGY,
        "query_receiver_strategy": (
            "one separate parsed query scene; immutable proposal batches of at most 256; "
            "500000 shared path cap with deterministic 128/64 split"
        ),
        "minibatch_contract": (
            "two-pass global beta; FULL-P count-normalized streamed gradients; "
            "one Adam update and one regularizer evaluation per epoch"
        ),
        "same_training_cache_all_branches": True,
        "deduplicate_equal_targets": bool(deduplicate_equal_targets),
        "equal_target_optimizer_reuse": reused_from,
        "equal_offset_branch_groups_verified": equivalent,
        "train_trace_elapsed_seconds": trace_elapsed,
        "query_inference_elapsed_seconds": query_elapsed,
        "branch_optimizer_elapsed_seconds": {
            name: result["metadata"]["optimizer_elapsed_seconds"]
            for name, result in fitted.items()
        },
        "train_path_mask": train_path_mask.tolist(),
        "fit_mask": fit_mask.tolist(),
        "query_path_mask": query_mask.tolist(),
        "coverage": {"train": material._coverage(train_path_mask),
                     "query": material._coverage(query_mask)},
        "branch_metadata": {name: result["metadata"] for name, result in fitted.items()},
        "parameter_files": parameter_files,
        "geometry_bank": bank_manifest_before,
        "geometry_bank_immutable": True,
        "parameter_selection": "all 12 frozen before labels; no Q-based selection",
        "claim_boundary": (
            "effective train-conditioned shared-class correction; not true EM material recovery"
        ),
        "scalability_boundary": "process-local host bank must be rebuilt after process exit",
        "elapsed_seconds": time.time() - started,
    }
    validate_banked_metadata(metadata)
    branches = {
        name: {
            "baseline_train": baseline.copy(),
            "baseline_query": baseline_query.copy(),
            "material_train": result["material_train"].copy(),
            "material_query": material_query[name].copy(),
            "params": {
                key: value.copy() for key, value in result["params"].items()
            },
            "history": result["history"],
            "metadata": result["metadata"],
        }
        for name, result in fitted.items()
    }
    result = {
        "branches": branches,
        "coverage": metadata["coverage"],
        "masks": {"train_path": train_path_mask.copy(),
                  "train_fit": fit_mask.copy(), "query_path": query_mask.copy()},
        "metadata": metadata,
    }
    if output is not None:
        metadata_path = output / "material_metadata.json"
        material._write_json(metadata_path, metadata)
        result["metadata_file"] = str(metadata_path.resolve())
    return result


def validate_banked_metadata(metadata: Mapping) -> None:
    """Fail closed so a legacy all-at-once result cannot pass the banked gate."""
    material.validate_metadata(metadata)
    if metadata.get("training_receiver_strategy") != TRAINING_STRATEGY:
        raise ValueError("result is not a chunked host-geometry-bank fit")
    if metadata.get("geometry_bank_immutable") is not True:
        raise ValueError("host geometry immutability is not explicit")
    contract = metadata.get("construction_contract", {})
    for key in ("host_geometry_bank_manifest_sha256", "geometry_bank_source_sha256",
                "material_source_sha256", "adapter_source_sha256"):
        value = contract.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing banked construction hash: {key}")


__all__ = [
    "fit_material_branches_banked", "validate_banked_metadata", "SceneReceiverBank"
]
