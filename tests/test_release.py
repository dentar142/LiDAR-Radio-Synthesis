"""Tests for the source-only release builder."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import zipfile

import pytest

from tools.build_release import _safe_resolve, build_release


def test_release_contains_source_and_excludes_private_assets() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "repo"
        (root / "src").mkdir(parents=True)
        (root / "config").mkdir()
        (root / "data").mkdir()
        (root / "film").mkdir()
        (root / "src" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "config" / "example.yaml").write_text("input: /absolute/path/to/model.obj\n", encoding="utf-8")
        (root / "data" / "measurements.csv").write_text("private,data\n", encoding="utf-8")
        (root / "film" / "capture.mp4").write_bytes(b"private-video")
        (root / "README.md").write_text("Public source release.\n", encoding="utf-8")
        (root / "LICENSE").write_text("Test license.\n", encoding="utf-8")

        archive, manifest = build_release(root, root / "release")
        with zipfile.ZipFile(archive) as bundle:
            names = set(bundle.namelist())
        assert "src/module.py" in names
        assert "config/example.yaml" in names
        assert not any(name.startswith(("data/", "film/", "runs/")) for name in names)
        assert not any(name.endswith((".csv", ".mp4", ".obj")) for name in names)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["file_count"] == len(names)
        assert len(payload["sha256"]) == 64
        assert set(payload["file_sha256"]) == names
        assert all(len(value) == 64 for value in payload["file_sha256"].values())


def test_release_rejects_path_escape() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        with pytest.raises(ValueError, match="unsafe release path"):
            _safe_resolve(root, Path("../secret.txt"))
        with pytest.raises(ValueError, match="unsafe release name"):
            build_release(root, root / "release", "../../outside")


def test_release_rejects_high_risk_text() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "src").mkdir(parents=True)
        credential_fixture = "api_" + "key = 'abcdefghijklmnop'\n"
        (root / "src" / "bad.py").write_text(credential_fixture, encoding="utf-8")
        with pytest.raises(ValueError, match="credential assignment"):
            build_release(root, root / "release")
