#!/usr/bin/env python3
"""Stitch a title card, the annotated pedestrian clip, and an end card.

This is a convenience for the proposal/demo video, not part of the mandatory
PoC. The annotated clip is produced by:

    namma-drishti run -i data/samples/pedestrians_15s.mp4 \
        --save-video -o results/demo_pedestrians
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from namma_drishti.visualize import make_title_card

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "demo_pedestrians"
OUT = ROOT / "results" / "demo.mp4"


def build(title: Path, end: Path) -> int:
    filter_graph = (
        "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0c1016,setsar=1,fps=10[a];"
        "[1:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0c1016,setsar=1,fps=10[b];"
        "[2:v]scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0c1016,setsar=1,fps=10[c];"
        "[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-loop",
        "1",
        "-t",
        "2.5",
        "-i",
        str(title),
        "-i",
        str(RUN / "annotated.mp4"),
        "-loop",
        "1",
        "-t",
        "3.5",
        "-i",
        str(end),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "22",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(OUT),
    ]
    print("Building", OUT)
    subprocess.run(command, check=True)
    print(f"Wrote {OUT} ({OUT.stat().st_size / (1024 * 1024):.2f} MB)")
    return 0


def main() -> int:
    if not shutil.which("ffmpeg"):
        print("ffmpeg is required to build the demo video.")
        return 1
    if not (RUN / "annotated.mp4").exists():
        print(f"Missing {RUN / 'annotated.mp4'}. Run the pedestrian demo first.")
        return 1

    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    counts = summary["counts"]
    person = counts["person"]
    vehicles = sum(
        counts[name]["unique_tracked"] or 0
        for name in ("bicycle", "car", "motorcycle", "bus", "truck")
    )
    frames = summary["run"]["frames_processed"]
    model_ms = summary["latency"]["model_inference"]["mean_ms"]

    with tempfile.TemporaryDirectory() as tmp:
        title = Path(tmp) / "demo_title.png"
        end = Path(tmp) / "demo_end.png"
        make_title_card(
            title,
            "Namma Drishti",
            "Offline edge counting for safer streets",
            "ExecuTorch  ·  YOLO11n INT8  ·  target: Raspberry Pi 5  ·  dev preview",
        )
        make_title_card(
            end,
            "15 seconds. Zero cloud.",
            f"{frames} frames  ·  {person['unique_tracked']} tracked people (indicative)  ·  "
            f"{vehicles} tracked vehicles",
            f"Model {model_ms:.1f} ms/frame on this dev host  ·  Pi 5 reference 134 ms / 7.5 FPS",
        )
        return build(title, end)


if __name__ == "__main__":
    raise SystemExit(main())
