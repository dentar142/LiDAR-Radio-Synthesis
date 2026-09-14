"""Build a source-only release archive after privacy and path-safety checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile


ROOT_FILES = (
    ".gitignore",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
)
ROOT_DIRS = (".github/workflows", "config", "docs", "src", "tests", "tools")
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "build", "dist", "release", "runs", "data", "film"}
EXCLUDED_SUFFIXES = {
    ".3ds", ".7z", ".avi", ".b3dm", ".bin", ".ckpt", ".csv", ".feather",
    ".gif", ".glb", ".gltf", ".jpeg", ".jpg", ".las", ".laz", ".mov", ".mp4",
    ".npy", ".npz", ".obj", ".onnx", ".osgb", ".parquet", ".pem", ".ply",
    ".png", ".pt", ".pth", ".tif", ".tiff", ".zip",
}
TEXT_SUFFIXES = {"", ".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SENSITIVE_PATTERNS = (
    ("Windows absolute path", re.compile(r"(?i)(?<![A-Z0-9_])[A-Z]:[\\/]")),
    ("home absolute path", re.compile(r"(?m)(?<![\w.])/(?:home|Users)/[A-Za-z0-9_.-]+/")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("credential assignment", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)\s*[:=]\s*['\"]?(?!placeholder|example|changeme)[A-Za-z0-9_./+=-]{12,}")),
)


def _safe_resolve(root: Path, relative: Path) -> Path:
    """Resolve a release candidate and reject absolute or escaping paths."""
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe release path: {relative}")
    root_resolved = root.resolve()
    candidate = (root_resolved / relative).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"release path escapes repository: {relative}")
    return candidate


def _candidate_paths(root: Path) -> list[Path]:
    candidates = [Path(name) for name in ROOT_FILES]
    for dirname in ROOT_DIRS:
        directory = _safe_resolve(root, Path(dirname))
        if directory.exists():
            candidates.extend(path.relative_to(root) for path in directory.rglob("*") if path.is_file())
    return sorted(set(candidates), key=lambda item: item.as_posix())


def collect_release_files(root: Path) -> list[Path]:
    """Return the checked, source-only release inventory."""
    root = root.resolve()
    included: list[Path] = []
    violations: list[str] = []
    for relative in _candidate_paths(root):
        path = _safe_resolve(root, relative)
        if not path.exists():
            continue
        posix = PurePosixPath(relative.as_posix())
        if any(part in EXCLUDED_PARTS for part in posix.parts) or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if (root / relative).is_symlink():
            violations.append(f"symlink not allowed: {relative.as_posix()}")
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            violations.append(f"binary content not allowed: {relative.as_posix()}")
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = data.decode("utf-8", errors="replace")
            for label, pattern in SENSITIVE_PATTERNS:
                if pattern.search(text):
                    violations.append(f"{label}: {relative.as_posix()}")
        included.append(relative)
    if violations:
        raise ValueError("release privacy check failed:\n" + "\n".join(sorted(violations)))
    return included


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(f"unsafe release name: {name}")
    return name


def build_release(root: Path, output_dir: Path, name: str = "pi-razer-source") -> tuple[Path, Path]:
    """Create a deterministic zip and JSON checksum manifest."""
    root = root.resolve()
    name = _validate_name(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"{name}.zip"
    inventory = collect_release_files(root)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for relative in inventory:
            data = _safe_resolve(root, relative).read_bytes()
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, data)
    manifest = output_dir / f"{name}.sha256.json"
    payload = {
        "archive": archive.name,
        "sha256": sha256(archive),
        "file_count": len(inventory),
        "files": [path.as_posix() for path in inventory],
        "file_sha256": {
            path.as_posix(): sha256(_safe_resolve(root, path)) for path in inventory
        },
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return archive, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--name", default="pi-razer-source")
    args = parser.parse_args()
    archive, manifest = build_release(args.root, args.output_dir, args.name)
    print(f"archive={archive}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
