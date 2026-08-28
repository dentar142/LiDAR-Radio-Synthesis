"""CLI scaffold for inspecting stage configuration."""
from __future__ import annotations
import argparse
from pathlib import Path

STAGES = ("geometry", "building", "semantic", "materials", "ground", "georef", "export")

def parse_args() -> argparse.Namespace:
    """Parse the stage and configuration path."""
    parser = argparse.ArgumentParser(description="HKUSTGZ material mapping stage scaffold")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    return parser.parse_args()

def main() -> int:
    """Validate paths and print the selected placeholder stages."""
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    selected = STAGES if args.stage == "all" else (args.stage,)
    print(f"config={args.config}")
    print("stages=" + ",".join(selected))
    print("scaffold only: no algorithms executed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
