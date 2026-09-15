"""Bounded, in-process host bank for frozen H20 path geometry.

The bank deliberately does not know how a Sionna ``Scene`` stores receivers.
Its ``receiver_bank`` collaborator must own one pre-created receiver bank and
implement ``set_chunk(xyz)`` without reloading the scene or rebuilding its BVH.
Geometry contains live scene shape identifiers and is therefore valid only in
the process that built it.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import os
import time
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np


MAX_RECEIVERS = 256
PATH_CAPACITY = 500_000
MIN_RECEIVERS = 64
MAX_HOST_BYTES = 8 * 1024**3

_CAPTURE_FIELDS = (
    "src_positions", "tgt_positions", "src_orientations", "tgt_orientations",
    "rel_ant_positions_tx", "rel_ant_positions_rx", "tx_velocities", "rx_velocities",
)


class ReceiverBank(Protocol):
    """Owner of a fixed-capacity receiver collection in one parsed scene."""

    capacity: int

    def set_chunk(self, xyz: np.ndarray) -> None:
        """Expose exactly ``len(xyz)`` receivers, in the supplied order."""


class RoundTripVerifier(Protocol):
    """Optional server gate that never needs two resident cache masters."""

    def capture(self, cache: Any, block: "HostGeometryBlock") -> Any:
        """Return a CPU-only reference before the original cache is released."""

    def compare(self, reference: Any, hydrated_cache: Any,
                block: "HostGeometryBlock") -> Mapping[str, Any]:
        """Compare the reference with a separately hydrated cache."""


def _sha_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _row_hash(row_ids: Sequence[str]) -> str:
    return _sha_payload([str(value) for value in row_ids])


def available_memory_bytes() -> int:
    """Return currently available host memory without adding a dependency."""
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="ascii") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                values[key] = int(raw.strip().split()[0]) * 1024
        if values.get("MemAvailable", 0) > 0:
            return values["MemAvailable"]
    except (OSError, ValueError, IndexError):
        pass
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, TypeError):
            pass
    raise RuntimeError("cannot determine available host memory")


@dataclass(frozen=True)
class HostValue:
    """Host copy plus its exact in-process DrJit/Mitsuba constructor."""

    constructor: type | None
    type_name: str
    value: Any
    nbytes: int

    @classmethod
    def capture(cls, value: Any) -> "HostValue":
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return cls(None, type(value).__name__, value, 0)
        array = np.array(np.asarray(value), copy=True)
        array.setflags(write=False)
        constructor = type(value)
        name = f"{constructor.__module__}.{constructor.__qualname__}"
        return cls(constructor, name, array, int(array.nbytes))

    def hydrate(self) -> Any:
        if self.constructor is None:
            return self.value
        return self.constructor(np.array(self.value, copy=True))

    def digest(self) -> str:
        digest = hashlib.sha256(self.type_name.encode("utf-8"))
        if isinstance(self.value, np.ndarray):
            digest.update(self.value.dtype.str.encode("ascii"))
            digest.update(repr(self.value.shape).encode("ascii"))
            digest.update(self.value.tobytes(order="C"))
        else:
            digest.update(repr(self.value).encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class HostGeometryBlock:
    start: int
    stop: int
    row_ids: tuple[str, ...]
    xyz: np.ndarray
    geometry: Mapping[str, HostValue]
    captured: Mapping[str, HostValue]
    buffer_constructor: type
    samples_per_src: int
    diffraction: bool
    nbytes: int
    trace_seconds: float
    verification: Mapping[str, Any] | None = None

    @property
    def row_sha256(self) -> str:
        return _row_hash(self.row_ids)

    @property
    def geometry_sha256(self) -> str:
        return _sha_payload({key: value.digest() for key, value in sorted(self.geometry.items())})


def _validate_inputs(row_ids, xyz, receiver_bank, trace_kwargs) -> tuple[tuple[str, ...], np.ndarray, dict]:
    ids = tuple(str(value) for value in row_ids)
    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(ids):
        raise ValueError("row_ids and finite xyz[N,3] must align")
    if not ids or not np.isfinite(points).all() or len(set(ids)) != len(ids):
        raise ValueError("row_ids must be unique and xyz must be finite")
    capacity = getattr(receiver_bank, "capacity", None)
    if not isinstance(capacity, int) or capacity != MAX_RECEIVERS:
        raise ValueError("receiver bank capacity must be exactly 256")
    config = dict(trace_kwargs)
    if int(config.get("max_num_paths_per_src", -1)) != PATH_CAPACITY:
        raise ValueError("H20 geometry bank requires path capacity 500000")
    if not isinstance(config.get("seed"), (int, np.integer)):
        raise ValueError("trace seed is required")
    if not 0 <= int(config["seed"]) <= np.iinfo(np.uint32).max:
        raise ValueError("trace seed must be uint32")
    if config.get("diffraction") is not False or config.get("edge_diffraction") is not False:
        raise ValueError("host geometry bank currently supports frozen non-diffraction H20 only")
    return ids, points, config


def _capture_cache(cache, W, start: int, stop: int, ids, points, elapsed: float) -> HostGeometryBlock:
    master = cache.paths_buffer
    geometry = {}
    for name in W._PATH_BUFFER_GEOMETRY_FIELDS:
        value = getattr(master, name)
        if name == "_diffracting_wedges" and value is None:
            geometry[name] = HostValue.capture(None)
        else:
            geometry[name] = HostValue.capture(value)
    captured = {name: HostValue.capture(getattr(cache, name)) for name in _CAPTURE_FIELDS}
    point_copy = np.array(points, dtype=np.float32, copy=True)
    point_copy.setflags(write=False)
    total = int(point_copy.nbytes)
    total += sum(value.nbytes for value in geometry.values())
    total += sum(value.nbytes for value in captured.values())
    return HostGeometryBlock(
        start=start, stop=stop, row_ids=tuple(ids), xyz=point_copy,
        geometry=MappingProxyType(geometry), captured=MappingProxyType(captured),
        buffer_constructor=type(master),
        samples_per_src=int(cache.samples_per_src), diffraction=bool(cache.diffraction),
        nbytes=total, trace_seconds=float(elapsed),
    )


def _stored_path_count(master: Any) -> int:
    """Use the official counter when exposed; otherwise fail over to buffer size."""
    if hasattr(master, "paths_counter"):
        counter = np.asarray(master.paths_counter)
        if counter.size:
            return int(counter.max())
    return int(master.buffer_size)


class GeometryBank:
    """CPU-hosted path geometry with an exclusive one-master GPU lease."""

    def __init__(self, scene, receiver_bank: ReceiverBank, W, blocks, *,
                 support_seed: int, host_budget_bytes: int, cap_hits):
        self.scene = scene
        self.receiver_bank = receiver_bank
        self.W = W
        self.blocks = tuple(blocks)
        self.support_seed = int(support_seed)
        self.host_budget_bytes = int(host_budget_bytes)
        self.cap_hits = tuple(cap_hits)
        self._leased = False

    @property
    def host_bytes(self) -> int:
        return sum(block.nbytes for block in self.blocks)

    @contextmanager
    def hydrate(self, block_index: int) -> Iterator[Any]:
        """Hydrate exactly one immutable cache master for field replay."""
        if self._leased:
            raise RuntimeError("only one GPU cache master may be hydrated at a time")
        try:
            block = self.blocks[int(block_index)]
        except (IndexError, ValueError, TypeError) as exc:
            raise IndexError("invalid geometry block index") from exc
        self._leased = True
        cache = None
        try:
            self.receiver_bank.set_chunk(np.array(block.xyz, copy=True))
            geometry = block.geometry
            buffer = block.buffer_constructor(
                int(geometry["_buffer_size"].value),
                int(geometry["_max_depth"].value),
                block.diffraction,
            )
            for name, value in geometry.items():
                setattr(buffer, name, value.hydrate())
            solver = self.W.PathSolver()
            solver.loop_mode = "evaluated"
            values = {name: block.captured[name].hydrate() for name in _CAPTURE_FIELDS}
            cache = self.W.CachedGeometry(
                self.scene, solver, buffer,
                values["src_positions"], values["tgt_positions"],
                values["src_orientations"], values["tgt_orientations"],
                values["rel_ant_positions_tx"], values["rel_ant_positions_rx"],
                values["tx_velocities"], values["rx_velocities"],
                block.samples_per_src, block.diffraction,
            )
            yield cache
        finally:
            if cache is not None:
                # Make retaining the yielded wrapper unable to retain a GPU
                # master beyond the exclusive lease.
                cache.paths_buffer = None
            cache = None
            self._leased = False

    def private_manifest(self) -> dict:
        return {
            "schema": "h20-in-process-host-geometry-bank-v1",
            "same_process_only": True,
            "shape_references_cross_process_safe": False,
            "support_seed": self.support_seed,
            "host_bytes": self.host_bytes,
            "host_budget_bytes": self.host_budget_bytes,
            "cap_hits": list(self.cap_hits),
            "blocks": [
                {
                    "ordinal": index, "start": block.start, "stop": block.stop,
                    "rows": block.stop - block.start, "row_sha256": block.row_sha256,
                    "geometry_sha256": block.geometry_sha256, "bytes": block.nbytes,
                    "trace_seconds": block.trace_seconds,
                    "round_trip_verification": (dict(block.verification)
                                                if block.verification is not None else None),
                    "constructor_types": {
                        key: value.type_name for key, value in sorted(block.geometry.items())
                    },
                }
                for index, block in enumerate(self.blocks)
            ],
        }


def build_geometry_bank(scene, receiver_bank: ReceiverBank, row_ids, xyz, *, W,
                        trace_kwargs: Mapping[str, Any],
                        available_host_bytes: int | None = None,
                        round_trip_verifier: RoundTripVerifier | None = None) -> GeometryBank:
    """Trace static-prior geometry once per deterministic receiver chunk.

    A capacity hit discards that attempt and deterministically retries its same
    rows at 128, then 64 receivers. A hit at 64 is a hard resource failure.
    Every attempt uses the identical frozen support seed.
    """
    ids, points, config = _validate_inputs(row_ids, xyz, receiver_bank, trace_kwargs)
    available = available_memory_bytes() if available_host_bytes is None else int(available_host_bytes)
    if available <= 0:
        raise RuntimeError("available host memory must be positive")
    host_budget = min(available // 2, MAX_HOST_BYTES)
    blocks, cap_hits = [], []
    host_bytes = 0

    def trace_range(start: int, stop: int, target_size: int) -> None:
        nonlocal host_bytes
        cursor = start
        while cursor < stop:
            end = min(cursor + target_size, stop)
            chunk = points[cursor:end]
            receiver_bank.set_chunk(np.array(chunk, copy=True))
            started = time.monotonic()
            cache = W.trace_geometry(scene, **config)
            elapsed = time.monotonic() - started
            observed = _stored_path_count(cache.paths_buffer)
            cap_hit = observed >= PATH_CAPACITY
            if cap_hit:
                cap_hits.append({"start": cursor, "stop": end, "attempt_rows": end - cursor,
                                 "observed_paths": observed})
                cache = None
                if target_size <= MIN_RECEIVERS:
                    raise RuntimeError("RESOURCE_NO_GO: path capacity hit at 64 receivers")
                trace_range(cursor, end, 128 if target_size > 128 else MIN_RECEIVERS)
            else:
                block = _capture_cache(cache, W, cursor, end, ids[cursor:end], chunk, elapsed)
                if host_bytes + block.nbytes > host_budget:
                    raise RuntimeError("RESOURCE_NO_GO: strict host geometry budget exceeded")
                reference = (round_trip_verifier.capture(cache, block)
                             if round_trip_verifier is not None else None)
                # The trace cache is released before its host reconstruction is
                # created, so verification cannot double the resident masters.
                cache = None
                if round_trip_verifier is not None:
                    temporary = GeometryBank(
                        scene, receiver_bank, W, [block], support_seed=int(config["seed"]),
                        host_budget_bytes=host_budget, cap_hits=[],
                    )
                    with temporary.hydrate(0) as hydrated:
                        verification = dict(round_trip_verifier.compare(reference, hydrated, block))
                    if verification.get("passed") is not True:
                        raise RuntimeError("geometry bank round-trip verification failed: "
                                           + json.dumps(verification, sort_keys=True))
                    block = replace(block, verification=MappingProxyType(verification))
                blocks.append(block)
                host_bytes += block.nbytes
                print(json.dumps({"stage": "host_block_verified", "start": cursor,
                                  "stop": end, "total_rows": len(points),
                                  "host_bytes": host_bytes, "path_count": observed}), flush=True)
            cursor = end

    trace_range(0, len(points), min(MAX_RECEIVERS, receiver_bank.capacity))
    if tuple(value for block in blocks for value in block.row_ids) != ids:
        raise RuntimeError("geometry bank changed original receiver row order")
    return GeometryBank(scene, receiver_bank, W, blocks, support_seed=int(config["seed"]),
                        host_budget_bytes=host_budget, cap_hits=cap_hits)
