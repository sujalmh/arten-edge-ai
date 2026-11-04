#!/usr/bin/env python3
"""Optional reproducibility path: export YOLO11n to ExecuTorch yourself.

The hackathon build uses Arm's pre-optimized INT8 .pte because it is smaller
and ~2.2x faster on a Raspberry Pi 5 than the FP32 export. This script exists
so judges can verify the model lineage and so the team has a fallback if the
prebuilt model is unavailable.

Requirements (on a development host, not the Pi):
    pip install ultralytics executorch torch

Usage:
    python scripts/export_model.py --weights yolo11n.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo11n.pt", help="Ultralytics weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="models/yolo11n_fp32_xnnpack.pte")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install the export extras first: pip install ultralytics executorch torch")
        return 1

    print(f"Exporting {args.weights} at {args.imgsz}px to ExecuTorch...")
    model = YOLO(args.weights)
    exported = model.export(format="executorch", imgsz=args.imgsz)

    # Ultralytics writes a directory containing model.pte + metadata.yaml.
    exported_dir = Path(str(exported).replace(".pte", ""))
    pte_candidates = list(exported_dir.glob("*.pte")) if exported_dir.is_dir() else [Path(str(exported))]
    if not pte_candidates:
        raise FileNotFoundError(f"could not find a .pte in {exported}")
    source = pte_candidates[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(f"Wrote {output} ({output.stat().st_size / (1024 * 1024):.2f} MB)")
    print(
        "This is the FP32 export. The Arm INT8 model is the primary deployment "
        "artifact and is downloaded by scripts/download_model.sh."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
