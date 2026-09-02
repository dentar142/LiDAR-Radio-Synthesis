"""Atomic, reproducible scene manifest export."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

@dataclass(frozen=True)
class SceneManifest:
    """Versioned list of inputs, outputs and audit references."""
    scene_id: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    version: str

def export_manifest(manifest: SceneManifest, destination: Path) -> Path:
    """Serialize a manifest atomically without modifying any source asset."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "hkustgz.material_mapping.scene_manifest.v1",
        "scene_id": manifest.scene_id,
        "version": manifest.version,
        "inputs": dict(sorted(manifest.inputs.items())),
        "outputs": dict(sorted(manifest.outputs.items())),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
