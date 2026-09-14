"""Portable public entry point. Run --help for commands."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "predict"):
        p = commands.add_parser(name)
        p.add_argument("--input", type=Path, required=True)
        if name == "predict":
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--mode", choices=("infill", "spatial30", "spatial60"), default="infill")
            p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            p.add_argument("--seed", type=int, default=241301)
            p.add_argument("--quick", action="store_true", help="Reduced GP fitting for software checks")
    p = commands.add_parser("demo")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("score")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "inspect":
        from src.radio.data import load_dataset
        data, tr, qu = load_dataset(a.input)
        print(f"Valid {data.band}: {len(tr)} train, {len(qu)} query, {len(data.configs)} RT configurations")
    elif a.command == "predict":
        from src.radio.workflow import predict
        predict(a.input, a.output, mode=a.mode, seed=a.seed, quick=a.quick, device=a.device)
    elif a.command == "score":
        from src.radio.data import score_predictions
        a.output.parent.mkdir(parents=True, exist_ok=True)
        score_predictions(a.predictions, a.truth).to_csv(a.output, index=False)
    else:
        from src.radio.demo import create_fixture
        from src.radio.workflow import predict
        from src.radio.data import score_predictions
        import torch
        torch.set_num_threads(1)
        create_fixture(a.output / "input")
        predict(a.output / "input", a.output / "predictions", quick=True)
        score_predictions(a.output / "predictions" / "predictions.csv",
                          a.output / "input" / "truth.csv").to_csv(
                              a.output / "synthetic-test-metrics.csv", index=False)
        print(f"Synthetic software smoke test completed: {a.output}")


if __name__ == "__main__":
    main()
