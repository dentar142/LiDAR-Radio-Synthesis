"""Prepare lightweight presentation textures without duplicating source models."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS)


def grade(image: Image.Image, blue: float = 0.18) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(1.18)
    image = ImageEnhance.Color(image).enhance(0.62)
    overlay = Image.new("RGB", image.size, (8, 24, 52))
    return Image.blend(image, overlay, blue)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-plan", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--problem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with Image.open(args.master_plan) as source:
        crop = source.crop((4700, 1600, 14500, 9000))
        plan = grade(fit(crop, (1920, 1080)), 0.14)
        plan.filter(ImageFilter.UnsharpMask(1.5, 120, 2)).save(args.output / "plan_1080.png")

    with Image.open(args.registration) as source:
        registration = grade(fit(source, (1920, 1080)), 0.12)
        registration.save(args.output / "registration_1080.png")

    with Image.open(args.problem) as source:
        problem = grade(fit(source, (1920, 1080)), 0.22)
        problem.save(args.output / "problem_1080.png")


if __name__ == "__main__":
    main()
