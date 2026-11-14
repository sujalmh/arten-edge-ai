"""Command-line interface for Arten."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from .constants import (
    COCO_CLASSES,
    DEFAULT_CLASSES,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    PI5_REFERENCE,
)
from .metrics import environment_info, write_json
from .model import ExecuTorchDetector, ModelLoadError
from .pipeline import benchmark_model, run_input


def _parse_classes(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in COCO_CLASSES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown COCO categories: {', '.join(unknown)}. "
            f"Available: {', '.join(COCO_CLASSES)}"
        )
    return names


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"run_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arten",
        description=(
            "Offline edge object detection and counting with ExecuTorch. "
            "Process a video or images, get per-frame counts, annotated "
            "evidence, and latency numbers."
        ),
    )
    parser.add_argument("--version", action="version", version="arten 1.0.0")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a video or images")
    run.add_argument("--input", "-i", required=True, help="video, image, or image directory")
    run.add_argument("--output", "-o", default=None, help="output directory")
    run.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="path to .pte")
    run.add_argument(
        "--classes",
        type=_parse_classes,
        default=DEFAULT_CLASSES,
        help="comma-separated COCO categories (default: person,bicycle,car,motorcycle,bus,truck)",
    )
    run.add_argument("--conf", type=float, default=0.2, help="confidence threshold (0.15 for very dense scenes)")
    run.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    run.add_argument(
        "--tile",
        type=int,
        default=0,
        help="split each frame into an NxN grid of 640 crops (0/1 off; 2 = 2x2, about 4x cost)",
    )
    run.add_argument("--pose-model", default=None, help="optional Lite-HRNet18 .pte for posture")
    run.add_argument("--pose-every", type=int, default=10, help="run pose every N frames")
    run.add_argument("--pose-max-persons", type=int, default=5, help="max person crops per pose frame")
    run.add_argument("--seg-model", default=None, help="optional DDRNet23 .pte for road/sidewalk surface")
    run.add_argument("--seg-every", type=int, default=10, help="run segmentation every N frames")
    run.add_argument("--no-track", action="store_true", help="disable the indicative tracker")
    run.add_argument("--no-attributes", action="store_true", help="disable motion attributes (direction, relative speed, dwell)")
    run.add_argument("--save-video", action="store_true", help="write annotated.mp4")
    run.add_argument(
        "--annotate-every",
        type=int,
        default=10,
        help="save an annotated JPEG every N frames (0 disables JPEGs)",
    )
    run.add_argument("--max-frames", type=int, default=None, help="process at most N frames/images")
    run.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    run.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    run.add_argument(
        "--zone",
        default=None,
        help="optional JSON polygon profile for a localized people/vehicle co-presence audit",
    )
    run.add_argument(
        "--estimate-vehicle-colors",
        action="store_true",
        help="add heuristic HSV vehicle-color estimates (not a learned model)",
    )
    run.add_argument("--quiet", "-q", action="store_true", help="suppress progress and the final summary")

    bench = sub.add_parser("benchmark", help="measure model-only latency")
    bench.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    bench.add_argument("--input", default=str(DEFAULT_DATA_DIR / "bengaluru_traffic.jpg"))
    bench.add_argument("--runs", type=int, default=50)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--threads", type=int, default=None)
    bench.add_argument("--output", default=None, help="optional JSON output path")
    bench.add_argument("--json", action="store_true", help="print the full JSON result")

    info = sub.add_parser("info", help="show model and runtime information")
    info.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    info.add_argument("--json", action="store_true", help="print machine-readable JSON")

    doctor = sub.add_parser("doctor", help="check the environment and sample assets")
    doctor.add_argument("--model", default=str(DEFAULT_MODEL_PATH))

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    output = Path(args.output).expanduser().resolve() if args.output else _default_output()
    run_input(
        input_path=args.input,
        output_dir=output,
        model_path=args.model,
        classes=args.classes,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        track=not args.no_track,
        attributes=not args.no_attributes,
        save_video=args.save_video,
        annotate_every=args.annotate_every,
        max_frames=args.max_frames,
        stride=args.stride,
        threads=args.threads,
        quiet=args.quiet,
        zone_config=args.zone,
        estimate_vehicle_colors=args.estimate_vehicle_colors,
        tile_grid=args.tile,
        pose_model=args.pose_model,
        pose_every=args.pose_every,
        pose_max_persons=args.pose_max_persons,
        seg_model=args.seg_model,
        seg_every=args.seg_every,
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    result = benchmark_model(
        model_path=args.model,
        input_path=args.input,
        runs=args.runs,
        warmup=args.warmup,
        threads=args.threads,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        write_json(args.output, result)
        print(f"Wrote {args.output}")
    stats = result["latency"]
    reference = result.get("reference") or PI5_REFERENCE
    print("\nModel-only latency")
    print(f"  model      {result['model'].get('name', Path(args.model).name)}")
    print(f"  device     {result['environment'].get('machine')} · {result['environment'].get('platform')}")
    print(f"  runs       {result['runs']} (warmup {result['warmup']})")
    print(f"  mean       {stats['mean_ms']:.2f} ms")
    print(f"  p50        {stats['p50_ms']:.2f} ms")
    print(f"  p95        {stats['p95_ms']:.2f} ms")
    print(f"  throughput {stats['fps_from_mean']:.2f} FPS")
    print(f"  peak RSS   {result.get('peak_rss_mb')} MB (process-wide)")
    print("\nReference (Arm model card, Pi 5): "
          f"{reference['p50_latency_ms']:.2f} ms p50 · {reference['fps']:.2f} FPS")
    if args.json:
        print(payload)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    detector = ExecuTorchDetector(args.model, warmup=0)
    info = detector.info()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    model = info["model"]
    env = info.get("environment", info)
    print(f"\n{model['name']}\n")
    print(f"  path        {model['path']}")
    print(f"  size        {model['size_mb']:.2f} MB")
    print(f"  sha256      {model['sha256']}")
    print(f"  checksum    {'OK' if model['sha256_match'] else 'unknown' if model['sha256_match'] is None else 'MISMATCH'}")
    print(f"  format      {model['format']}")
    print(f"  backend     {model['backend']}")
    print(f"  input       {model['input_shape']} · range {model['input_range']}")
    print(f"  classes     {model['classes']} COCO categories")
    print(f"  runtime     ExecuTorch {model['executorch']} · torch {model['torch']}")
    print(f"  device      {env.get('machine')} · {env.get('platform')}")
    print(f"  load        {model['load_ms']:.2f} ms")
    print(f"  source      {model.get('source') or 'unknown (custom model)'}")
    print()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print("\nArten — environment check\n")
    print(f"  machine      {platform.machine()}")
    print(f"  platform     {platform.platform()}")
    print(f"  python       {platform.python_version()}")

    for module in ("torch", "executorch", "cv2", "numpy", "PIL"):
        try:
            imported = __import__(module)
            if module == "executorch":
                from importlib.metadata import version as package_version

                version = package_version("executorch")
            else:
                version = getattr(imported, "__version__", "ok")
            print(f"  {module:<12} {version}")
        except Exception as exc:
            ok = False
            print(f"  {module:<12} MISSING ({exc})")

    model_path = Path(args.model)
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"  model        {model_path.name} · {size:.2f} MB")
    else:
        ok = False
        print(f"  model        MISSING: {model_path}")

    sample = DEFAULT_DATA_DIR / "pedestrians_15s.mp4"
    print(f"  sample       {'OK' if sample.exists() else 'MISSING'} ({sample})")

    if not ok:
        print("\nFix the items above, then run `bash scripts/setup_pi.sh` if needed.\n")
        return 1
    print("\nAll checks passed. Try:")
    print(f"  arten run -i {sample} --save-video\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "benchmark":
            return cmd_benchmark(args)
        if args.command == "info":
            return cmd_info(args)
        if args.command == "doctor":
            return cmd_doctor(args)
    except (ModelLoadError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nError: {exc}\n", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
