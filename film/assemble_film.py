"""Generate narration, subtitles, music and assemble the final film on Linux."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def restyle_ass(path: Path) -> None:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    style = (
        "Style: Default,Noto Sans CJK SC,12,&H00FFFFFF,&H000000FF,&H6807101F,&H00000000,"
        "-1,0,0,0,100,100,0,0,3,0,0,2,18,18,20,1"
    )
    rewritten: list[str] = []
    for line in lines:
        if line.startswith("Style: Default,"):
            rewritten.append(style)
        elif line.startswith("Dialogue:"):
            fields = line.split(",", 9)
            fields[9] = wrap_ass_text(fields[9])
            rewritten.append(",".join(fields))
        else:
            rewritten.append(line)
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def wrap_ass_text(text: str, maximum_units: float = 24.0) -> str:
    """Insert ASS line breaks for Chinese text, which libass cannot wrap at spaces."""

    text = text.replace("\\N", " ").strip()
    lines: list[str] = []
    current: list[str] = []
    units = 0.0
    punctuation = set("，。；：、！？,.!?;:")
    for character in text:
        weight = 0.5 if character.isascii() else 1.0
        if current and units + weight > maximum_units:
            split = next((index for index in range(len(current) - 1, max(-1, len(current) - 7), -1)
                          if current[index] in punctuation), -1)
            if split >= 0:
                lines.append("".join(current[: split + 1]).strip())
                current = current[split + 1 :]
                units = sum(0.5 if value.isascii() else 1.0 for value in current)
            else:
                lines.append("".join(current).strip())
                current = []
                units = 0.0
        current.append(character)
        units += weight
    if current:
        lines.append("".join(current).strip())
    return "\\N".join(line for line in lines if line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--narration-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    args = parser.parse_args()
    work = args.output.parent
    work.mkdir(parents=True, exist_ok=True)
    narration = work / "narration_zh.mp3"
    subtitles = work / "narration_zh.vtt"
    subtitles_ass = work / "narration_zh.ass"
    music = work / "music_bed.wav"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.deps) + os.pathsep + environment.get("PYTHONPATH", "")

    print("+ generate neural narration", flush=True)
    subprocess.run([
        sys.executable, "-m", "edge_tts", "--voice", "zh-CN-YunxiNeural", "--rate=+25%",
        "--file", str(args.narration_text), "--write-media", str(narration),
        "--write-subtitles", str(subtitles),
    ], check=True, env=environment)
    run([sys.executable, str(Path(__file__).with_name("make_audio_bed.py")), str(music), "--duration", "155"])
    run(["ffmpeg", "-y", "-i", str(subtitles), str(subtitles_ass)])
    restyle_ass(subtitles_ass)
    run([
        "ffmpeg", "-y", "-i", str(args.visual), "-i", str(narration), "-i", str(music),
        "-filter_complex",
        f"[0:v]ass={subtitles_ass}:fontsdir=/usr/share/fonts/opentype/noto[v];"
        "[1:a]volume=1.0,pan=stereo|c0=c0|c1=c0[voice];[2:a]volume=0.16[music];"
        "[voice][music]amix=inputs=2:duration=longest:dropout_transition=2[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-t", "155", "-movflags", "+faststart",
        str(args.output),
    ])


if __name__ == "__main__":
    main()
