"""Bounded, leakage-safe neural and exact-GP models for the H15 screen.

The conditional diffusion model in this module is a small scalar residual
model written for this experiment.  It is not an implementation or claimed
reproduction of RadioDiff (or of any other image diffusion publication).
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import numpy as np


_WAVELENGTHS_M = (25.0, 75.0, 200.0)
_JITTER = 1e-8
_UINT32_MODULUS = 2 ** 32
# Required by deterministic CUDA GEMM.  H15 deliberately imports torch lazily,
# so this executes before this module can import/initialize torch's CUDA path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised by deployment
        raise RuntimeError("H15 models require PyTorch") from exc
    return torch, nn


def _seed32(seed: int, offset: int = 0) -> int:
    """Return a stable uint32 seed for every derived model/RNG stream."""
    return (int(seed) + int(offset)) % _UINT32_MODULUS


def _array2(values, rows: int | None, name: str, columns: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or (rows is not None and len(array) != rows):
        expected = "N" if rows is None else str(rows)
        raise ValueError(f"{name} must have shape ({expected}, D)")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array.copy()


def _clean_inputs(train_xy, train_y, query_xy, train_aux, query_aux, physical_means):
    tx = _array2(train_xy, None, "train_xy", 2)
    qx = _array2(query_xy, None, "query_xy", 2)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1).copy()
    if len(y) != len(tx) or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("train_y must be finite, aligned, and contain at least two rows")
    if train_aux is None:
        ta = np.empty((len(tx), 0), dtype=np.float64)
    else:
        ta = _array2(train_aux, len(tx), "train_aux")
    if query_aux is None:
        qa = np.empty((len(qx), 0), dtype=np.float64)
    else:
        qa = _array2(query_aux, len(qx), "query_aux")
    if ta.shape[1] != qa.shape[1]:
        raise ValueError("train_aux and query_aux must have the same number of columns")

    means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in ("RT", "MAT_TWC"):
        if key not in physical_means or len(physical_means[key]) != 2:
            raise ValueError(f"physical_means[{key!r}] must be a (train, query) pair")
        train_mean = np.asarray(physical_means[key][0], dtype=np.float64).reshape(-1).copy()
        query_mean = np.asarray(physical_means[key][1], dtype=np.float64).reshape(-1).copy()
        if len(train_mean) != len(tx) or len(query_mean) != len(qx):
            raise ValueError(f"physical_means[{key!r}] has inconsistent row counts")
        if not np.isfinite(train_mean).all() or not np.isfinite(query_mean).all():
            raise ValueError(f"physical_means[{key!r}] must be finite")
        means[key] = (train_mean, query_mean)
    return tx, y, qx, ta, qa, means


def _standardize_aux(train: np.ndarray, query: np.ndarray):
    if train.shape[1] == 0:
        return train.copy(), query.copy(), [], []
    center = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    return (
        np.clip((train - center) / scale, -10.0, 10.0),
        np.clip((query - center) / scale, -10.0, 10.0),
        center.tolist(),
        scale.tolist(),
    )


def _condition_features(train_xy, query_xy, train_aux, query_aux):
    train_scaled_aux, query_scaled_aux, center, scale = _standardize_aux(train_aux, query_aux)

    def build(xy, aux):
        parts = [xy / 100.0]
        for wavelength in _WAVELENGTHS_M:
            angle = (2.0 * np.pi / wavelength) * xy
            parts.extend((np.sin(angle), np.cos(angle)))
        parts.append(aux)
        return np.column_stack(parts).astype(np.float32, copy=False)

    return build(train_xy, train_scaled_aux), build(query_xy, query_scaled_aux), {
        "xy_scale_m": 100.0,
        "fourier_wavelengths_m": list(_WAVELENGTHS_M),
        "aux_center": center,
        "aux_scale": scale,
        "aux_clip": [-10.0, 10.0],
    }


def _target_scale(values: np.ndarray):
    center = float(values.mean())
    scale = float(values.std())
    if scale < 1e-10:
        scale = 1.0
    return (values - center) / scale, center, scale


def _seed_torch(torch, seed: int, device: str):
    seed = _seed32(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _bounded_log(raw, low: float, high: float):
    return raw.sigmoid() * (math.log(high) - math.log(low)) + math.log(low)


def _raw_for_initial(torch, initial: float, low: float, high: float, device):
    fraction = (math.log(initial) - math.log(low)) / (math.log(high) - math.log(low))
    return torch.tensor(math.log(fraction / (1.0 - fraction)), dtype=torch.float64, device=device)


def _matern32(torch, left, right, length, signal_variance):
    delta = (left[:, None, :] - right[None, :, :]) / length[None, None, :]
    # A tiny positive floor avoids the undefined derivative of sqrt at the
    # many exact-zero diagonal distances without acting as covariance jitter.
    radius = torch.sqrt(torch.clamp(torch.sum(delta * delta, dim=-1), min=1e-24))
    z = math.sqrt(3.0) * radius
    return signal_variance * (1.0 + z) * torch.exp(-z)


def _exact_gp(train_xy, train_y, query_xy, seed: int, device: str, quick: bool):
    """Fit a float64 exact ARD Matérn-3/2 GP and return latent moments."""
    torch, _ = _torch()
    _seed_torch(torch, seed, device)
    dev = torch.device(device)
    x = torch.as_tensor(train_xy, dtype=torch.float64, device=dev)
    q = torch.as_tensor(query_xy, dtype=torch.float64, device=dev)
    standardized, y_center, y_scale = _target_scale(train_y)
    y = torch.as_tensor(standardized, dtype=torch.float64, device=dev)
    eye = torch.eye(len(x), dtype=torch.float64, device=dev)
    steps = 3 if quick else 60
    # Normal mode is the initial fit plus two fixed restarts (three runs total).
    initials = [(50.0, 50.0, 1.0, 0.05)] if quick else [
        (50.0, 50.0, 1.0, 0.05),
        (20.0, 100.0, 0.5, 0.2),
        (150.0, 35.0, 2.0, 0.01),
    ]
    runs: list[dict[str, Any]] = []
    best = None
    for run_index, (lx0, ly0, signal0, noise0) in enumerate(initials):
        raw_length = torch.nn.Parameter(torch.stack([
            _raw_for_initial(torch, lx0, 1.0, 2000.0, dev),
            _raw_for_initial(torch, ly0, 1.0, 2000.0, dev),
        ]))
        raw_signal = torch.nn.Parameter(_raw_for_initial(torch, signal0, 0.01, 100.0, dev))
        raw_noise = torch.nn.Parameter(_raw_for_initial(torch, noise0, 0.0001, 10.0, dev))
        optimizer = torch.optim.Adam([raw_length, raw_signal, raw_noise], lr=0.05)
        losses = []
        run_best = None
        for _step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            length = torch.exp(_bounded_log(raw_length, 1.0, 2000.0))
            signal = torch.exp(_bounded_log(raw_signal, 0.01, 100.0))
            noise = torch.exp(_bounded_log(raw_noise, 0.0001, 10.0))
            kernel = _matern32(torch, x, x, length, signal) + (noise + _JITTER) * eye
            try:
                chol = torch.linalg.cholesky(kernel)
            except RuntimeError as exc:
                raise RuntimeError(f"exact GP Cholesky failed in restart {run_index}") from exc
            alpha = torch.cholesky_solve(y[:, None], chol)[:, 0]
            nll = 0.5 * torch.dot(y, alpha) + torch.log(torch.diagonal(chol)).sum()
            nll = nll + 0.5 * len(x) * math.log(2.0 * math.pi)
            if not bool(torch.isfinite(nll)):
                raise RuntimeError(f"exact GP produced non-finite NLL in restart {run_index}")
            loss_value = float(nll.detach().cpu())
            if run_best is None or loss_value < run_best[0]:
                run_best = (loss_value, {
                    "length_m": [float(v) for v in length.detach().cpu()],
                    "signal_variance": float(signal.detach().cpu()),
                    "noise_variance": float(noise.detach().cpu()),
                })
            nll.backward()
            optimizer.step()
            losses.append(loss_value)
        assert run_best is not None
        params = run_best[1]
        run = {"restart": run_index, "steps": steps, "initial_nll": losses[0],
               "final_nll": losses[-1], "minimum_step_nll": run_best[0], **params}
        runs.append(run)
        if best is None or run_best[0] < best[0]:
            best = (run_best[0], params, run_index)

    assert best is not None
    params = best[1]
    length = torch.as_tensor(params["length_m"], dtype=torch.float64, device=dev)
    signal = torch.tensor(params["signal_variance"], dtype=torch.float64, device=dev)
    noise = torch.tensor(params["noise_variance"], dtype=torch.float64, device=dev)
    kernel = _matern32(torch, x, x, length, signal) + (noise + _JITTER) * eye
    try:
        chol = torch.linalg.cholesky(kernel)
    except RuntimeError as exc:
        raise RuntimeError("exact GP Cholesky failed for selected fit") from exc
    alpha = torch.cholesky_solve(y[:, None], chol)
    mean_parts = []
    variance_parts = []
    for start in range(0, len(q), 512):
        cross = _matern32(torch, q[start:start + 512], x, length, signal)
        mean = (cross @ alpha)[:, 0]
        solved = torch.linalg.solve_triangular(chol, cross.T, upper=False)
        variance = torch.clamp(signal - torch.sum(solved * solved, dim=0), min=0.0)
        mean_parts.append(mean.detach().cpu().numpy())
        variance_parts.append(variance.detach().cpu().numpy())
    mean_np = (np.concatenate(mean_parts) if mean_parts else np.empty(0)) * y_scale + y_center
    variance_np = (np.concatenate(variance_parts) if variance_parts else np.empty(0)) * (y_scale ** 2)
    metadata = {
        "kind": "exact_gp", "dtype": "float64", "kernel": "Matern32_ARD_2D",
        "jitter": _JITTER, "optimizer": "Adam", "optimizer_steps_per_restart": steps,
        "restart_count": len(initials), "selected_restart": int(best[2]),
        "target_center": y_center, "target_scale": y_scale, "convergence": runs,
        "hyperparameter_bounds": {"length_m": [1.0, 2000.0],
                                  "signal_variance": [0.01, 100.0],
                                  "noise_variance": [0.0001, 10.0]},
        "posterior_variance": {"min": float(variance_np.min()) if len(variance_np) else None,
                               "mean": float(variance_np.mean()) if len(variance_np) else None,
                               "max": float(variance_np.max()) if len(variance_np) else None,
                               "includes_observation_noise": False},
    }
    del x, q, y, eye, kernel, chol, alpha
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return mean_np, variance_np, metadata


def _make_mlp(nn, input_dim: int):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))


def _fit_mlp(train_features, target, query_features, seed, device, quick, loss_name):
    torch, nn = _torch()
    _seed_torch(torch, seed, device)
    dev = torch.device(device)
    x = torch.as_tensor(train_features, dtype=torch.float32, device=dev)
    q = torch.as_tensor(query_features, dtype=torch.float32, device=dev)
    scaled_y, center, scale = _target_scale(target)
    y = torch.as_tensor(scaled_y, dtype=torch.float32, device=dev)
    model = _make_mlp(nn, x.shape[1]).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.MSELoss() if loss_name == "mse" else nn.HuberLoss(delta=1.0)
    updates = 3 if quick else 1000
    generator = torch.Generator(device="cpu").manual_seed(_seed32(seed, 991))
    losses = []
    model.train()
    for _ in range(updates):
        index = torch.randint(len(x), (min(128, len(x)),), generator=generator).to(dev)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x[index])[:, 0], y[index])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        prediction = model(q)[:, 0].cpu().numpy().astype(np.float64) * scale + center
    metadata = {"kind": "residual_mlp", "architecture": "2x64_SiLU", "loss": loss_name,
                "optimizer": "AdamW", "updates": updates, "batch_size": 128,
                "learning_rate": 0.001, "weight_decay": 0.001,
                "target_center": center, "target_scale": scale,
                "train_loss": {"initial": losses[0], "final": losses[-1], "minimum": min(losses)}}
    del model, optimizer, x, q, y
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, metadata


def _cosine_schedule(torch, steps: int, device):
    grid = torch.linspace(0, steps, steps + 1, dtype=torch.float32, device=device)
    cumulative = torch.cos(((grid / steps + 0.008) / 1.008) * math.pi / 2.0) ** 2
    cumulative = cumulative / cumulative[0]
    betas = torch.clamp(1.0 - cumulative[1:] / cumulative[:-1], 1e-4, 0.999)
    return betas, torch.cumprod(1.0 - betas, dim=0)


def _time_embedding(torch, timestep, total: int):
    phase = timestep.float()[:, None] / max(total - 1, 1)
    frequencies = torch.tensor([1.0, 2.0, 4.0, 8.0], device=timestep.device)[None, :]
    angle = 2.0 * math.pi * phase * frequencies
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=1)


def _fit_ddpm(train_features, target, query_features, seed, device, quick):
    torch, nn = _torch()
    _seed_torch(torch, seed, device)
    dev = torch.device(device)
    x = torch.as_tensor(train_features, dtype=torch.float32, device=dev)
    q = torch.as_tensor(query_features, dtype=torch.float32, device=dev)
    scaled_y, center, scale = _target_scale(target)
    y = torch.as_tensor(scaled_y, dtype=torch.float32, device=dev)
    total = 32
    _, alpha_bar = _cosine_schedule(torch, total, dev)
    model = _make_mlp(nn, x.shape[1] + 1 + 8).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    updates = 5 if quick else 2000
    generator = torch.Generator(device="cpu").manual_seed(_seed32(seed, 1777))
    losses = []
    model.train()
    for _ in range(updates):
        size = min(128, len(x))
        index = torch.randint(len(x), (size,), generator=generator).to(dev)
        timestep = torch.randint(total, (size,), generator=generator).to(dev)
        noise = torch.randn(size, generator=generator).to(dev)
        abar = alpha_bar[timestep]
        noisy = abar.sqrt() * y[index] + (1.0 - abar).sqrt() * noise
        target_v = abar.sqrt() * noise - (1.0 - abar).sqrt() * y[index]
        inputs = torch.cat((x[index], noisy[:, None], _time_embedding(torch, timestep, total)), dim=1)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs)[:, 0] - target_v) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    # The same fixed antithetic 16-noise design is used for every query row.
    # Consequently results do not depend on query order or caller chunking.
    sample_generator = torch.Generator(device="cpu").manual_seed(_seed32(seed, 2777))
    half = torch.randn(8, generator=sample_generator)
    initial = torch.cat((half, -half)).to(dev)
    draws = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(q), 2048):
            condition = q[start:start + 2048]
            rows = len(condition)
            state = initial[None, :].expand(rows, -1).clone()
            repeated_condition = condition[:, None, :].expand(-1, 16, -1)
            for step in range(total - 1, -1, -1):
                timestep = torch.full((rows * 16,), step, dtype=torch.long, device=dev)
                network_input = torch.cat((
                    repeated_condition.reshape(rows * 16, -1), state.reshape(-1, 1),
                    _time_embedding(torch, timestep, total)), dim=1)
                velocity = model(network_input)[:, 0].reshape(rows, 16)
                abar = alpha_bar[step]
                # v-parameterization avoids division by sqrt(alpha_bar) at
                # the very small endpoint of the cosine schedule.
                x0 = torch.sqrt(abar) * state - torch.sqrt(1.0 - abar) * velocity
                epsilon = torch.sqrt(1.0 - abar) * state + torch.sqrt(abar) * velocity
                if step > 0:  # deterministic DDIM, eta=0
                    previous = alpha_bar[step - 1]
                    state = torch.sqrt(previous) * x0 + torch.sqrt(1.0 - previous) * epsilon
                else:
                    state = x0
            draws.append(state.cpu().numpy().astype(np.float64) * scale + center)
    draw_array = np.concatenate(draws, axis=0) if draws else np.empty((0, 16), dtype=np.float64)
    metadata = {"kind": "conditional_scalar_DDPM", "claimed_literature_reproduction": False,
                "schedule": "cosine", "diffusion_steps": total, "training_updates": updates,
                "batch_size": 128, "objective": "v_prediction_MSE",
                "sampler": "DDIM", "ddim_steps": 32, "ddim_eta": 0.0,
                "draws": 16, "draw_design": "fixed_antithetic_per_query",
                "target_center": center, "target_scale": scale,
                "train_noise_loss": {"initial": losses[0], "final": losses[-1], "minimum": min(losses)},
                "draw_std_summary": {"mean": float(draw_array.std(axis=1).mean()) if len(draw_array) else None,
                                     "max": float(draw_array.std(axis=1).max()) if len(draw_array) else None,
                                     "is_calibrated_confidence": False}}
    del model, optimizer, x, q, y, alpha_bar
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return draw_array.mean(axis=1), np.median(draw_array, axis=1), metadata


def predict_models(train_xy, train_y, query_xy, train_aux, query_aux,
                   physical_means, seed, device="cpu", quick=False):
    """Fit the fixed H15 pool and return query predictions plus JSON metadata."""
    tx, y, qx, ta, qa, means = _clean_inputs(
        train_xy, train_y, query_xy, train_aux, query_aux, physical_means)
    started = time.perf_counter()
    predictions: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}

    for offset, (name, target, baseline) in enumerate((
        ("GP_XY_M32", y, None),
        ("RT_GP", y - means["RT"][0], means["RT"][1]),
        ("MAT_TWC_GP", y - means["MAT_TWC"][0], means["MAT_TWC"][1]),
    )):
        model_started = time.perf_counter()
        estimate, variance, info = _exact_gp(tx, target, qx, _seed32(seed, offset), device, bool(quick))
        predictions[name] = estimate if baseline is None else baseline + estimate
        info["elapsed_seconds"] = float(time.perf_counter() - model_started)
        info["physical_prior"] = None if baseline is None else ("RT" if name == "RT_GP" else "MAT_TWC")
        model_metadata[name] = info

    train_features, query_features, feature_metadata = _condition_features(tx, qx, ta, qa)
    for offset, (name, prior_key, loss_name) in enumerate((
        ("RT_MLP_MSE", "RT", "mse"),
        ("RT_MLP_HUBER", "RT", "huber"),
        ("MAT_TWC_MLP_MSE", "MAT_TWC", "mse"),
    ), start=20):
        model_started = time.perf_counter()
        train_prior, query_prior = means[prior_key]
        residual, info = _fit_mlp(train_features, y - train_prior, query_features,
                                  _seed32(seed, offset), device, bool(quick), loss_name)
        predictions[name] = query_prior + residual
        info["physical_prior"] = prior_key
        info["features"] = feature_metadata
        info["elapsed_seconds"] = float(time.perf_counter() - model_started)
        model_metadata[name] = info

    rt_train, rt_query = means["RT"]
    model_started = time.perf_counter()
    ddpm_mean, ddpm_median, info = _fit_ddpm(
        train_features, y - rt_train, query_features, _seed32(seed, 40), device, bool(quick))
    predictions["RT_CDDPM"] = rt_query + ddpm_mean
    predictions["RT_CDDPM_MEDIAN"] = rt_query + ddpm_median
    info["physical_prior"] = "RT"
    info["features"] = feature_metadata
    info["elapsed_seconds"] = float(time.perf_counter() - model_started)
    model_metadata["RT_CDDPM"] = info
    metadata = {"seed": _seed32(seed), "device": str(device), "quick": bool(quick),
                "elapsed_seconds": float(time.perf_counter() - started),
                "models": model_metadata,
                "leakage_contract": "train scaling and fitting only; query targets unavailable"}
    return predictions, metadata
