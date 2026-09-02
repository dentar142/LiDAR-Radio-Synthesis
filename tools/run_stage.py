"""Run the generic audit pipeline or automatic plan + LiDAR reconstruction."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src import run_pipeline, run_reconstruction_pipeline

STAGES = ("geometry", "building", "semantic", "materials", "ground", "georef", "export")

def parse_args() -> argparse.Namespace:
    """Parse the stage and configuration path."""
    parser = argparse.ArgumentParser(description="HKUSTGZ clean-map automatic pipeline")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()

def main() -> int:
    """Dispatch to the configured pipeline mode and print the result."""
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"config not found: {args.config}")
    payload = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    mode = payload.get("mode", "existing_geometry_audit")
    if mode == "plan_lidar_reconstruction":
        if args.stage != "all":
            raise SystemExit("plan_lidar_reconstruction runs as one gated end-to-end transaction")
        result = run_reconstruction_pipeline(args.config, args.output)
        print(f"operational_status={result.operational_status}")
        print(f"scientific_status={result.scientific_status}")
        print(f"run_directory={result.run_directory}")
        print("gates=" + ",".join(f"{key}:{'PASS' if value else 'FAIL'}" for key, value in result.gates.items()))
        return 0 if result.operational_status == "PASS" else 4
    result = run_pipeline(args.config, args.output, "export" if args.stage == "all" else args.stage)
    print(f"status={result.status}")
    print(f"run_directory={result.run_directory}")
    return 0 if result.status.startswith("PASS") else 4

if __name__ == "__main__":
    raise SystemExit(main())
