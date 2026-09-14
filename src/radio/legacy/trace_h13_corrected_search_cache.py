#!/usr/bin/env python3
"""Build the corrected, full-geometry H13 Sionna search cache.

This runner intentionally reads receiver geometry only.  It never reads signal
labels, point-to-unique mappings, or any train/validation/test targets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .run_h13_semantic_ablation import (
    FIXED_TX_BY_BAND,
    FREQUENCY_MHZ,
    incoherent_cir_power_db,
)
from .trace_h7_pointwise_rt import (
    V7R4_SEARCH_OBJECTS,
    v7r4_semantic_preserving_configs,
)


BANDS = ("n41", "n79")
SAMPLES_PER_SOURCE = 50_000
MAX_DEPTH = 3
CHUNK_SIZE = 128
RAY_SEED = 241_301
MECHANISMS = {
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": True,
    "refraction": True,
    "diffraction": False,
    "edge_diffraction": False,
}
FROZEN_OBJECTS = [
    "HKUSTGZ_metal_roof",
    "HKUSTGZ_ground_water",
    "HKUSTGZ_ground_stone",
    "HKUSTGZ_ground_vegetation",
    "HKUSTGZ_ground_asphalt",
]


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    descriptor = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    return sha256_bytes(descriptor + value.tobytes())


def referenced_scene_asset_hashes(scene_xml: Path) -> dict[str, str]:
    """Hash file-valued XML attributes so mesh changes invalidate the cache."""
    assets: dict[str, str] = {}
    root = ET.parse(scene_xml).getroot()
    for element in root.iter():
        for value in element.attrib.values():
            candidate = (scene_xml.parent / value).resolve()
            if candidate.is_file() and candidate != scene_xml.resolve():
                try:
                    name = candidate.relative_to(scene_xml.parent.resolve()).as_posix()
                except ValueError:
                    name = candidate.name
                assets[name] = sha256_file(candidate)
    return dict(sorted(assets.items()))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def load_full_geometry(data_root: Path, band: str) -> tuple[np.ndarray, Path]:
    path = data_root / "cache" / band / "unique_xyz.npy"
    xyz = np.load(path, allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"invalid receiver geometry shape: {xyz.shape}")
    if not np.isfinite(xyz).all():
        raise ValueError("receiver geometry contains non-finite coordinates")
    return np.asarray(xyz, dtype=float), path


def output_band_dir(output_root: Path, band: str, benchmark_count: int) -> Path:
    base = output_root if benchmark_count == 0 else output_root / f"freshbenchmark_{benchmark_count:06d}"
    return base / "cache" / band


def material_specs(config: dict) -> dict[str, dict]:
    """Return only material changes; omitted scene objects remain untouched."""
    family = config["family"]
    if family == "BASE":
        return {}
    specs: dict[str, dict] = {}
    for object_name, group in V7R4_SEARCH_OBJECTS.items():
        value = config["groups"][group]
        if family == "S4W":
            specs[object_name] = {
                "kind": "continuous",
                "relative_permittivity": float(value["eps"]),
                "conductivity": float(value["sigma"]),
            }
        elif family == "S5":
            specs[object_name] = {"kind": "itu", "itu_type": str(value)}
        else:
            raise ValueError(f"unknown config family: {family}")
    return specs


def apply_common_scattering(scene, config: dict) -> None:
    """Normalize scattering for every arm without changing frozen eps/sigma."""
    del config  # The condition is deliberately identical for all 13 arms.
    all_objects = list(V7R4_SEARCH_OBJECTS) + FROZEN_OBJECTS
    for object_name in all_objects:
        if object_name not in scene.objects:
            raise KeyError(f"scene lacks V7-r4 semantic object: {object_name}")
        material = scene.objects[object_name].radio_material
        material.scattering_coefficient = 0.20
        material.xpd_coefficient = 0.15


def apply_config(scene, config: dict) -> None:
    apply_common_scattering(scene, config)
    from sionna.rt import ITURadioMaterial, RadioMaterial

    for object_name, spec in material_specs(config).items():
        name = f"h13_corrected_{config['id']}_{object_name}"
        if spec["kind"] == "continuous":
            material = RadioMaterial(
                name=name,
                relative_permittivity=spec["relative_permittivity"],
                conductivity=spec["conductivity"],
                scattering_coefficient=0.20,
                xpd_coefficient=0.15,
            )
        else:
            material = ITURadioMaterial(name=name, itu_type=spec["itu_type"], thickness=0.1)
            material.scattering_coefficient = 0.20
            material.xpd_coefficient = 0.15
        scene.objects[object_name].radio_material = material


def trace_config(scene_xml: Path, band: str, xyz: np.ndarray, config: dict | None) -> np.ndarray:
    import drjit as dr
    import mitsuba as mi

    if mi.variant() is None:
        mi.set_variant("cuda_ad_mono_polarized")
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    direct_only = config is None
    scene = load_scene(str(scene_xml.resolve()))
    scene.frequency = FREQUENCY_MHZ[band] * 1e6
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    if config is not None:
        apply_config(scene, config)
    scene.add(Transmitter(name="h13_corrected_tx", position=FIXED_TX_BY_BAND[band].tolist(), power_dbm=30.0))
    solver = PathSolver()
    result = np.full(len(xyz), np.nan, dtype=float)
    for start in range(0, len(xyz), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(xyz))
        names = []
        for index, point in enumerate(xyz[start:stop], start):
            name = f"h13_corrected_rx_{index}"
            names.append(name)
            scene.add(Receiver(name=name, position=point.tolist()))
        paths = solver(
            scene=scene,
            max_depth=0 if direct_only else MAX_DEPTH,
            samples_per_src=1 if direct_only else SAMPLES_PER_SOURCE,
            los=True,
            specular_reflection=False if direct_only else MECHANISMS["specular_reflection"],
            diffuse_reflection=False if direct_only else MECHANISMS["diffuse_reflection"],
            refraction=False if direct_only else MECHANISMS["refraction"],
            diffraction=False,
            edge_diffraction=False,
            seed=RAY_SEED,
        )
        amplitudes, _ = paths.cir(out_type="numpy")
        result[start:stop] = incoherent_cir_power_db(amplitudes, stop - start)
        for name in names:
            scene.remove(name)
        dr.sync_thread()
    del paths, solver, scene
    gc.collect()
    dr.sync_thread()
    if hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()
    return np.isfinite(result) if direct_only else result


def build_contract(scene_xml: Path, geometry_path: Path, xyz: np.ndarray, band: str,
                   benchmark_count: int, configs: list[dict],
                   source_full_receiver_count: int | None = None) -> dict:
    payload = {
        "schema": "h13-corrected-search-cache-v1",
        "band": band,
        "mode": "full" if benchmark_count == 0 else "fresh_benchmark_nonfinal",
        "eligible_for_final": benchmark_count == 0,
        "scene_xml_sha256": sha256_file(scene_xml),
        "scene_referenced_asset_sha256": referenced_scene_asset_hashes(scene_xml),
        "geometry_file_sha256": sha256_file(geometry_path),
        "geometry_array_sha256": array_sha256(xyz),
        "receiver_count": int(len(xyz)),
        "source_full_receiver_count": int(
            len(xyz) if source_full_receiver_count is None else source_full_receiver_count
        ),
        "configs_sha256": sha256_bytes(canonical_json(configs)),
        "search_objects": V7R4_SEARCH_OBJECTS,
        "frozen_scene_objects": FROZEN_OBJECTS,
        "common_scattering_all_semantic_objects": {
            "objects": list(V7R4_SEARCH_OBJECTS) + FROZEN_OBJECTS,
            "scattering_coefficient": 0.20,
            "xpd_coefficient": 0.15,
            "scope": "identical for BASE, S4W, and S5; frozen objects retain other EM parameters",
        },
        "frequency_mhz": FREQUENCY_MHZ[band],
        "equivalent_tx_assumption": "n41=E-rooftop; n79=W-rooftop; fixed comparison input, not physical source truth",
        "tx_position_sha256": array_sha256(np.asarray(FIXED_TX_BY_BAND[band], dtype=float)),
        "mechanisms": MECHANISMS,
        "samples_per_source": SAMPLES_PER_SOURCE,
        "max_depth": MAX_DEPTH,
        "chunk_size": CHUNK_SIZE,
        "ray_seed_uint32": RAY_SEED,
        "power_estimator": "incoherent sum_path(|complex CIR amplitude|^2); real^2+imag^2",
        "explicitly_not": "coherent CFR power; real-component-only CIR power",
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "complex_power_helper_sha256": sha256_file(
            Path(__file__).with_name("run_h13_semantic_ablation.py").resolve()
        ),
        "config_helper_sha256": sha256_file(
            Path(__file__).with_name("trace_h7_pointwise_rt.py").resolve()
        ),
    }
    payload["contract_hash"] = sha256_bytes(canonical_json(payload))
    return payload


def valid_resume(output: Path, status: Path, contract_hash: str, receiver_count: int) -> bool:
    if not output.exists() and not status.exists():
        return False
    if not output.exists() or not status.exists():
        raise RuntimeError(f"partial cache artifact requires cleanup: {output}")
    meta = json.loads(status.read_text(encoding="utf-8"))
    array = np.load(output, allow_pickle=False)
    if meta.get("contract_hash") != contract_hash or array.shape != (receiver_count,):
        raise RuntimeError(f"cache artifact contract mismatch: {output}")
    if meta.get("output_sha256") != array_sha256(array):
        raise RuntimeError(f"cache artifact digest mismatch: {output}")
    return True


def run(args: argparse.Namespace) -> dict:
    configs = v7r4_semantic_preserving_configs()
    by_id = {config["id"]: config for config in configs}
    if len(configs) != 13 or len(by_id) != 13:
        raise RuntimeError("expected exactly 13 distinct V7-r4 search configurations")
    if args.config_id != "LOS" and args.config_id not in by_id:
        raise ValueError(f"unknown config id: {args.config_id}")
    xyz_full, geometry_path = load_full_geometry(args.data_root, args.band)
    if args.benchmark_count:
        if args.benchmark_count <= 0 or args.benchmark_count >= len(xyz_full):
            raise ValueError("benchmark-count must be positive and smaller than the full geometry")
        xyz = xyz_full[: args.benchmark_count].copy()
    else:
        xyz = xyz_full
    band_dir = output_band_dir(args.output_root, args.band, args.benchmark_count)
    band_dir.mkdir(parents=True, exist_ok=True)
    contract = build_contract(
        args.scene_xml, geometry_path, xyz, args.band, args.benchmark_count, configs,
        source_full_receiver_count=len(xyz_full),
    )
    configs_payload = {
        "profile": "v7r4_semantic_frozen_h13_corrected",
        "tx_site": "E" if args.band == "n41" else "W",
        "tx_position": np.asarray(FIXED_TX_BY_BAND[args.band], float).tolist(),
        "configs": configs,
        "search_objects": V7R4_SEARCH_OBJECTS,
        "v7r4_frozen_objects": FROZEN_OBJECTS,
        "contract_hash": contract["contract_hash"],
    }
    configs_path = band_dir / "configs.json"
    if configs_path.exists():
        existing = json.loads(configs_path.read_text(encoding="utf-8"))
        if existing != configs_payload:
            raise RuntimeError(f"configs.json contract mismatch: {configs_path}")
    else:
        atomic_json(configs_path, configs_payload)
    atomic_json(band_dir / "manifest.json", contract)

    stem = "los_unique" if args.config_id == "LOS" else args.config_id
    output = band_dir / f"{stem}.npy"
    status = band_dir / "status" / f"{stem}.json"
    job_contract_hash = sha256_bytes(canonical_json({
        "manifest_contract_hash": contract["contract_hash"],
        "config_id": args.config_id,
    }))
    if valid_resume(output, status, job_contract_hash, len(xyz)):
        return {"status": "RESUMED", "band": args.band, "config_id": args.config_id,
                "receiver_count": len(xyz), "contract_hash": job_contract_hash,
                "manifest_contract_hash": contract["contract_hash"]}
    started = time.time()
    try:
        values = trace_config(args.scene_xml, args.band, xyz, None if args.config_id == "LOS" else by_id[args.config_id])
        values = np.asarray(values, dtype=bool if args.config_id == "LOS" else float)
        if values.shape != (len(xyz),):
            raise RuntimeError(f"trace returned wrong shape: {values.shape}")
        atomic_npy(output, values)
        payload = {
            "status": "PASS",
            "band": args.band,
            "config_id": args.config_id,
            "receiver_count": int(len(xyz)),
            "finite_count": int(values.sum()) if values.dtype == bool else int(np.isfinite(values).sum()),
            "elapsed_seconds": time.time() - started,
            "contract_hash": job_contract_hash,
            "manifest_contract_hash": contract["contract_hash"],
            "output_sha256": array_sha256(values),
            "eligible_for_final": args.benchmark_count == 0,
        }
        atomic_json(status, payload)
        return payload
    except Exception as exc:
        atomic_json(status.with_suffix(".failed.json"), {
            "status": "FAIL", "band": args.band, "config_id": args.config_id,
            "contract_hash": job_contract_hash,
            "manifest_contract_hash": contract["contract_hash"],
            "error_type": type(exc).__name__,
            "error": str(exc), "elapsed_seconds": time.time() - started,
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--band", required=True, choices=BANDS)
    parser.add_argument("--config-id", required=True, help="one of 13 config IDs, or LOS")
    parser.add_argument("--benchmark-count", type=int, default=0,
                        help="non-final prefix benchmark written below freshbenchmark_<N>")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True), flush=True)
