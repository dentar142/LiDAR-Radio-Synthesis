"""Generate a restrained procedural stereo bed for the research film."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import struct
import wave


SCENE_CUES = (0.0, 8.0, 27.0, 47.0, 67.0, 97.0, 122.0, 140.0)
CHORDS = (
    (73.42, 110.00, 146.83),
    (65.41, 98.00, 130.81),
    (82.41, 123.47, 164.81),
    (55.00, 82.41, 110.00),
)


def envelope(value: float, attack: float, release: float) -> float:
    if value < 0.0 or value > release:
        return 0.0
    if value < attack:
        return value / attack
    return max(0.0, 1.0 - (value - attack) / max(0.001, release - attack))


def render(path: Path, duration: float, sample_rate: int = 48_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(142)
    phases = [0.0, 0.0, 0.0]
    noise_state = 0.0
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        block = bytearray()
        for index in range(int(duration * sample_rate)):
            t = index / sample_rate
            chord = CHORDS[int(t // 16.0) % len(CHORDS)]
            pad = 0.0
            for tone_index, frequency in enumerate(chord):
                phases[tone_index] += 2.0 * math.pi * frequency / sample_rate
                pad += math.sin(phases[tone_index]) + 0.22 * math.sin(phases[tone_index] * 0.5)
            pad *= 0.055 * (0.72 + 0.28 * math.sin(2.0 * math.pi * t / 12.0))

            pulse_phase = t % 2.0
            pulse = 0.045 * math.sin(2.0 * math.pi * 220.0 * t) * math.exp(-5.0 * pulse_phase)
            chime = 0.0
            for cue in SCENE_CUES:
                local = t - cue
                chime += 0.11 * envelope(local, 0.04, 1.8) * math.sin(
                    2.0 * math.pi * (440.0 + 90.0 * local) * local
                )

            raw_noise = random.uniform(-1.0, 1.0)
            noise_state = noise_state * 0.985 + raw_noise * 0.015
            whoosh = 0.0
            for cue in SCENE_CUES[1:]:
                local = t - (cue - 0.8)
                whoosh += 0.045 * envelope(local, 0.55, 1.25) * noise_state

            left = max(-1.0, min(1.0, pad + pulse + chime + whoosh))
            right = max(-1.0, min(1.0, pad + pulse * 0.92 + chime * 0.88 - whoosh))
            block.extend(struct.pack("<hh", int(left * 32767), int(right * 32767)))
            if len(block) >= 48_000:
                output.writeframesraw(block)
                block.clear()
        if block:
            output.writeframesraw(block)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--duration", type=float, default=155.0)
    args = parser.parse_args()
    render(args.output, args.duration)


if __name__ == "__main__":
    main()
