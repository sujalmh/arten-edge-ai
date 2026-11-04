"""Human-readable terminal summary and a self-contained Markdown run report."""

from __future__ import annotations

from pathlib import Path

from .constants import PI5_REFERENCE


def _table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    out = [line("-")]
    out.append(
        "|"
        + "|".join(f" {h:<{w}} " for h, w in zip(headers, widths))
        + "|"
    )
    out.append(line("="))
    for row in rows:
        out.append(
            "|"
            + "|".join(f" {str(v):<{w}} " for v, w in zip(row, widths))
            + "|"
        )
    out.append(line("-"))
    return "\n".join(out)


def format_summary(summary: dict) -> str:
    run = summary["run"]
    model = summary["model"]
    env = summary["environment"]
    latency = summary["latency"]
    throughput = summary["throughput"]
    counts = summary["counts"]
    exposure = summary["exposure"]

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("  NAMMA DRISHTI  ·  offline edge detection & counting")
    lines.append("=" * 78)
    lines.append(f"  Input        {Path(run['input']).name}")
    lines.append(
        f"  Frames       {run['frames_processed']}"
        + (f" / {run['source_frames']}" if run.get("source_frames") else "")
        + (f"  (stride {run['stride']})" if run.get("stride", 1) > 1 else "")
    )
    if run.get("source_fps"):
        lines.append(f"  Source       {run['source_fps']:.2f} FPS · {run['source_size'][0]}×{run['source_size'][1]}")
    lines.append(f"  Device       {env.get('machine')} · {env.get('platform')}")
    lines.append(
        f"  Runtime      ExecuTorch {model.get('executorch')} · torch {model.get('torch')}"
    )
    lines.append(
        f"  Model        {model['name']} · {model['size_mb']:.2f} MB · "
        f"{'checksum OK' if model.get('sha256_match') else 'checksum differs'}"
    )
    lines.append("")

    lines.append("  COUNTS (per processed frame)")
    rows = []
    for name, stats in counts.items():
        rows.append(
            [
                name,
                str(stats["peak"]),
                f"{stats['mean']:.2f}",
                str(stats["p95"]),
                str(stats["unique_tracked"]) if stats["unique_tracked"] is not None else "—",
            ]
        )
    lines.append(
        _table(
            ["category", "peak/frame", "mean/frame", "p95", "unique*"],
            rows,
            [14, 10, 10, 5, 8],
        )
    )
    lines.append("  * unique is indicative, from a lightweight IoU/velocity tracker.")
    lines.append("")

    lines.append("  LATENCY / THROUGHPUT")
    rows = []
    for label, key in (
        ("model infer", "model_inference"),
        ("preprocess", "preprocess"),
        ("decode + NMS", "postprocess"),
        ("frame read", "read"),
        ("annotate/save", "annotate"),
        ("pipeline total", "pipeline_total"),
    ):
        stats = latency.get(key, {})
        if not stats:
            continue
        rows.append(
            [
                label,
                f"{stats['mean_ms']:.2f}",
                f"{stats['p50_ms']:.2f}",
                f"{stats['p95_ms']:.2f}",
                f"{stats['fps_from_mean']:.2f}",
            ]
        )
    lines.append(
        _table(
            ["stage", "mean ms", "p50 ms", "p95 ms", "FPS"],
            rows,
            [14, 9, 8, 8, 7],
        )
    )
    lines.append(
        f"  Model-only {throughput['model_fps_from_mean']:.2f} FPS · "
        f"pipeline {throughput['pipeline_fps_from_mean']:.2f} FPS · "
        f"wall {run['wall_time_s']:.2f} s"
    )
    lines.append("")

    lines.append("  EXPOSURE (clip-level, descriptive)")
    lines.append(
        f"  person-seconds {exposure['person_seconds']:.2f} · "
        f"vehicle-seconds {exposure['vehicle_seconds']:.2f} · "
        f"frames with both {exposure['frames_with_person_and_vehicle']} "
        f"({exposure['share_frames_with_person_and_vehicle'] * 100:.1f}%)"
    )
    lines.append("")

    lines.append("  REFERENCE (not measured here)")
    lines.append(
        f"  Arm model card, {PI5_REFERENCE['device']}: "
        f"{PI5_REFERENCE['p50_latency_ms']:.2f} ms p50 · "
        f"{PI5_REFERENCE['fps']:.2f} FPS · "
        f"{PI5_REFERENCE['peak_memory_mb']:.2f} MB peak"
    )
    lines.append(
        "  Cite this only as a reference; replace it with your own Pi run at the event."
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def write_markdown_report(summary: dict, path: str | Path) -> Path:
    run = summary["run"]
    model = summary["model"]
    env = summary["environment"]
    latency = summary["latency"]
    throughput = summary["throughput"]
    counts = summary["counts"]
    exposure = summary["exposure"]
    path = Path(path)

    lines: list[str] = []
    lines.append("# Namma Drishti — run report\n")
    lines.append(
        "Offline pedestrian and vehicle counting with ExecuTorch on a "
        "Raspberry Pi. This report was generated from the run's `summary.json`.\n"
    )

    lines.append("## Run\n")
    lines.append(f"- **Input:** `{run['input']}`")
    lines.append(f"- **Frames processed:** {run['frames_processed']}")
    if run.get("source_fps"):
        lines.append(
            f"- **Source:** {run['source_fps']:.2f} FPS, "
            f"{run['source_size'][0]}×{run['source_size'][1]}"
        )
    lines.append(f"- **Categories:** {', '.join(run['classes'])}")
    lines.append(
        f"- **Thresholds:** confidence {run['confidence_threshold']}, "
        f"IoU {run['iou_threshold']}"
    )
    lines.append(f"- **Tracking:** {'on (indicative)' if run['tracking'] else 'off'}")
    lines.append(f"- **Device:** {env.get('machine')} · {env.get('platform')}")
    lines.append(f"- **Peak process RSS:** {summary.get('peak_rss_mb')} MB")
    lines.append("")

    lines.append("## Model\n")
    lines.append(f"- **Name:** {model['name']}")
    lines.append(f"- **Format:** {model['format']}, {model['size_mb']:.2f} MB")
    lines.append(f"- **Backend:** {model['backend']}")
    lines.append(f"- **Quantization:** {model['quantization']}")
    lines.append(f"- **SHA-256:** `{model['sha256']}`")
    lines.append(f"- **Source:** {model['source']}")
    lines.append(
        f"- **Reference accuracy (COCO val2017):** mAP@0.5:0.95 "
        f"{PI5_REFERENCE['map_50_95']:.2f}%, mAP@0.5 {PI5_REFERENCE['map_50']:.2f}%"
    )
    lines.append("")

    lines.append("## Counts per frame\n")
    lines.append("| Category | Peak/frame | Mean/frame | p95 | Total detections | Unique (indicative) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, stats in counts.items():
        unique = stats["unique_tracked"]
        lines.append(
            f"| {name} | {stats['peak']} | {stats['mean']:.2f} | {stats['p95']} | "
            f"{stats['total_detections']} | {unique if unique is not None else '—'} |"
        )
    lines.append("")
    lines.append(
        "> Counts are per processed frame. `Unique` comes from a lightweight "
        "IoU/velocity tracker and is indicative, not a production MOT count."
    )
    lines.append("")

    lines.append("## Latency and throughput\n")
    lines.append("| Stage | Mean (ms) | p50 (ms) | p95 (ms) | FPS from mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, key in (
        ("Model inference", "model_inference"),
        ("Preprocess", "preprocess"),
        ("Decode + NMS", "postprocess"),
        ("Frame read", "read"),
        ("Annotate/save", "annotate"),
        ("Pipeline total", "pipeline_total"),
    ):
        stats = latency.get(key, {})
        if not stats:
            continue
        lines.append(
            f"| {label} | {stats['mean_ms']:.2f} | {stats['p50_ms']:.2f} | "
            f"{stats['p95_ms']:.2f} | {stats['fps_from_mean']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"- **Model-only:** {throughput['model_fps_from_mean']:.2f} FPS\n"
        f"- **Pipeline:** {throughput['pipeline_fps_from_mean']:.2f} FPS\n"
        f"- **Wall time:** {run['wall_time_s']:.2f} s"
    )
    lines.append("")

    lines.append("## Exposure signals\n")
    lines.append(
        f"- **Observed person-seconds:** {exposure['person_seconds']:.2f}\n"
        f"- **Observed vehicle-seconds:** {exposure['vehicle_seconds']:.2f}\n"
        f"- **Frames with both people and vehicles:** "
        f"{exposure['frames_with_person_and_vehicle']} "
        f"({exposure['share_frames_with_person_and_vehicle'] * 100:.1f}%)"
    )
    lines.append("")
    lines.append(
        "> These are descriptive measures for the supplied clip, not "
        "extrapolations to a full day. A 15-second clip at school dismissal "
        "is a sample, not a traffic census."
    )
    lines.append("")

    lines.append("## Evidence\n")
    if summary["outputs"].get("contact_sheet"):
        lines.append(f"![Contact sheet]({Path(summary['outputs']['contact_sheet']).name})\n")
    annotated_dir = summary["outputs"].get("annotated_dir")
    if annotated_dir:
        samples = sorted(Path(annotated_dir).glob("*.jpg"))[:4]
        for sample in samples:
            lines.append(f"![{sample.name}](annotated/{sample.name})\n")
    lines.append(
        f"- `counts.csv` — per-frame category counts and stage timings\n"
        f"- `detections.jsonl` — boxes, scores and optional track IDs\n"
        f"- `summary.json` — machine-readable summary\n"
        f"- `annotated/` — sample annotated frames\n"
    )
    if summary["outputs"].get("annotated_video"):
        lines.append(f"- `annotated.mp4` — annotated clip\n")
    lines.append("")

    lines.append("## Reference benchmark (not this run)\n")
    lines.append(
        f"Arm's model card reports **{PI5_REFERENCE['p50_latency_ms']:.2f} ms "
        f"p50 / {PI5_REFERENCE['fps']:.2f} FPS** on a Raspberry Pi 5 "
        f"({PI5_REFERENCE['runtime']}), with "
        f"**{PI5_REFERENCE['peak_memory_mb']:.2f} MB** peak memory. "
        f"Those numbers are a reference for the prebuilt model; the table above "
        f"is the only latency measured in this run."
    )
    lines.append("")

    lines.append("## Notes and limitations\n")
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
