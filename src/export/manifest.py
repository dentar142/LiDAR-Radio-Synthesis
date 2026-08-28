"""Interfaces for reproducible scene export and archival."""
from dataclasses import dataclass
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
    """Serialize a manifest without overwriting source data; implementation omitted."""
    raise NotImplementedError
