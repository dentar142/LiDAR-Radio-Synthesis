#!/usr/bin/env python3
"""Sharded Sionna tracing for H7 full pointwise evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


V7R4_SEARCH_OBJECTS = {
    "HKUSTGZ_limestone_solid": "facade",
    "HKUSTGZ_building_base": "facade",
    "HKUSTGZ_glass_solid": "glass",
}

TX_SITES = {
    "E": np.asarray([134.42377217610678, 45.10205841064453, 37.0], dtype=float),
    "W": np.asarray([-136.84656524658203, 188.49261474609375, 17.0], dtype=float),
}


def v7r4_semantic_preserving_configs() -> list[dict]:
    configs: list[dict] = [{"id": "AUTO_BASE", "family": "BASE", "groups": {}}]
    facade = {
        "F_L": {"eps": 3.0, "sigma": 0.015},
        "F_M": {"eps": 6.0, "sigma": 0.050},
        "F_H": {"eps": 10.0, "sigma": 0.150},
    }
    glass = {
        "G_L": {"eps": 4.0, "sigma": 0.005},
        "G_M": {"eps": 7.0, "sigma": 0.020},
        "G_H": {"eps": 10.0, "sigma": 0.080},
    }
    for f_key, f_value in facade.items():
        for g_key, g_value in glass.items():
            configs.append({
                "id": f"S4W_{f_key}_{g_key}_V7R4FIX",
                "family": "S4W",
                "groups": {"facade": f_value, "glass": g_value},
            })
    for facade_label in ("concrete", "marble", "brick"):
        configs.append({
            "id": f"S5_{facade_label}_glass_V7R4FIX",
            "family": "S5",
            "groups": {"facade": facade_label, "glass": "glass"},
        })
    return configs


def load_legacy(path: Path):
    spec = importlib.util.spec_from_file_location("ew_macro_legacy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import legacy runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def trace_one(module, args, config: dict, xyz: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.time()
    gain = module.trace_gain(
        args.scene_xml,
        args.band,
        xyz,
        config,
        args.samples_per_source,
        args.max_depth,
        args.seed,
        args.chunk_size,
    )
    return gain, time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-runner", required=True, type=Path)
    parser.add_argument("--scene-xml", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--band", required=True, choices=("n41", "n79"))
    parser.add_argument("--mode", required=True, choices=("benchmark", "trace", "los"))
    parser.add_argument("--config-start", type=int, default=0)
    parser.add_argument("--config-stop", type=int, default=0)
    parser.add_argument("--profile", choices=("legacy37", "v7r4_semantic_frozen"), default="legacy37")
    parser.add_argument("--tx-site", choices=("AUTO", "E", "W"), default="AUTO")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--samples-per-source", type=int, default=200000)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=240901)
    parser.add_argument("--limit-points", type=int, default=0)
    args = parser.parse_args()

    module = load_legacy(args.legacy_runner)
    if args.tx_site != "AUTO":
        module.TRANSMITTERS[args.band] = TX_SITES[args.tx_site].copy()
    if args.profile == "v7r4_semantic_frozen":
        module.GROUP_BY_OBJECT = V7R4_SEARCH_OBJECTS
        configs = v7r4_semantic_preserving_configs()
    else:
        configs = module.candidate_configs()
    xyz = np.load(args.data_root / "cache" / args.band / "unique_xyz.npy")
    if args.limit_points > 0:
        xyz = xyz[: args.limit_points]
    band_dir = args.output_root / "cache" / args.band
    band_dir.mkdir(parents=True, exist_ok=True)
    write_json(band_dir / "configs.json", {
        "profile": args.profile,
        "tx_site": args.tx_site,
        "tx_position": module.TRANSMITTERS[args.band].tolist(),
        "configs": configs,
        "search_objects": module.GROUP_BY_OBJECT,
        "v7r4_frozen_objects": [
            "HKUSTGZ_metal_roof", "HKUSTGZ_ground_water", "HKUSTGZ_ground_stone",
            "HKUSTGZ_ground_vegetation", "HKUSTGZ_ground_asphalt",
        ] if args.profile == "v7r4_semantic_frozen" else [],
    })

    if args.mode == "benchmark":
        config = configs[0]
        gain, elapsed = trace_one(module, args, config, xyz)
        result = {
            "status": "PASS",
            "band": args.band,
            "mode": "benchmark",
            "receivers": int(len(xyz)),
            "chunk_size": int(args.chunk_size),
            "elapsed_seconds": elapsed,
            "receivers_per_second": float(len(xyz) / elapsed),
            "coverage": float(np.isfinite(gain).mean()),
        }
        write_json(args.output_root / f"benchmark_{args.band}_c{args.chunk_size}.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    if args.mode == "los":
        los_args = argparse.Namespace(**vars(args))
        los_args.samples_per_source = 1
        los_args.max_depth = 0
        gain, elapsed = trace_one(module, los_args, configs[0], xyz)
        los = np.isfinite(gain)
        np.save(band_dir / "los_unique.npy", los)
        result = {
            "status": "PASS", "band": args.band, "mode": "los",
            "receivers": int(len(xyz)), "los_rows": int(los.sum()),
            "los_fraction": float(los.mean()), "elapsed_seconds": elapsed,
        }
        write_json(args.output_root / f"los_status_{args.band}.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    start = max(0, args.config_start)
    stop = len(configs) if args.config_stop <= 0 else min(len(configs), args.config_stop)
    if not start < stop:
        raise ValueError(f"empty config shard [{start}, {stop})")
    status_path = args.output_root / f"trace_status_{args.band}_{start:02d}_{stop:02d}.json"
    shard_started = time.time()
    completed = 0
    for index in range(start, stop):
        config = configs[index]
        output_path = band_dir / f"{config['id']}.npy"
        if output_path.exists() and np.load(output_path, mmap_mode="r").shape == (len(xyz),):
            completed += 1
            print(f"[{args.band}] resume {index + 1}/{len(configs)} {config['id']}", flush=True)
            continue
        gain, elapsed = trace_one(module, args, config, xyz)
        np.save(output_path, gain)
        completed += 1
        payload = {
            "status": "RUNNING", "band": args.band,
            "config_start": start, "config_stop": stop,
            "completed_in_shard": completed, "total_in_shard": stop - start,
            "last_config": config["id"], "last_elapsed_seconds": elapsed,
            "total_elapsed_seconds": time.time() - shard_started,
            "path_coverage": float(np.isfinite(gain).mean()),
            "receivers": int(len(xyz)), "chunk_size": int(args.chunk_size),
            "profile": args.profile,
        }
        write_json(status_path, payload)
        print(
            f"[{args.band}] traced {index + 1}/{len(configs)} {config['id']} "
            f"in {elapsed:.1f}s coverage={payload['path_coverage']:.3f}", flush=True,
        )
    payload = {
        "status": "COMPLETED", "band": args.band,
        "config_start": start, "config_stop": stop,
        "completed_in_shard": completed, "total_in_shard": stop - start,
        "elapsed_seconds": time.time() - shard_started,
        "receivers": int(len(xyz)), "chunk_size": int(args.chunk_size),
        "profile": args.profile,
    }
    write_json(status_path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
