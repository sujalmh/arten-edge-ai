"""Render an AskResult as a self-contained HTML page or Markdown."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .execute import AskResult


def render_html(result: AskResult, out_path: str | Path, *, db_path: str | Path) -> Path:
    out_path = Path(out_path).expanduser().resolve()
    run_dir = Path(db_path).expanduser().resolve().parent
    body: list[str] = []

    body.append(f"<h1>{html.escape(result.question)}</h1>")
    if result.unsupported:
        body.append(f"<div class='note'>{html.escape(result.unsupported)}</div>")
    else:
        body.append(
            "<p class='meta'>parser: "
            + html.escape(result.parser)
            + (f" · {html.escape(result.model)}" if result.model else "")
            + f" · {result.elapsed_ms:.0f} ms</p>"
        )
        if result.notes:
            body.append("<p class='meta'>" + html.escape("; ".join(result.notes)) + "</p>")
        if result.plan is not None:
            body.append("<h2>Plan</h2>")
            body.append(
                "<pre>" + html.escape(json.dumps(result.plan.to_dict(), indent=2, ensure_ascii=False)) + "</pre>"
            )

        for query_result in result.results:
            body.append(f"<h2>{html.escape(query_result.label)}</h2>")
            body.append("<pre>" + html.escape(query_result.sql) + "</pre>")
            body.append("<p class='meta'>params: " + html.escape(repr(query_result.params)) + "</p>")
            body.extend(_html_table(query_result.columns, query_result.rows))
            body.extend(_html_images(query_result.columns, query_result.rows, run_dir))

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(result.question)}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1d1d1f; }}
  h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
  h2 {{ font-size: 1rem; margin-top: 1.75rem; text-transform: uppercase;
        letter-spacing: .04em; color: #555; }}
  .meta {{ color: #666; font-size: .9rem; margin: .25rem 0; }}
  .note {{ background: #fff4e5; border-left: 4px solid #f0a020; padding: .75rem 1rem; }}
  pre {{ background: #f5f5f7; padding: .75rem 1rem; overflow-x: auto;
         border-radius: 6px; font-size: .85rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .35rem .5rem; border-bottom: 1px solid #e5e5e7; }}
  th {{ background: #fafafa; }}
  .thumbs {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: .5rem; margin-top: .75rem; }}
  .thumbs figure {{ margin: 0; }}
  .thumbs img {{ width: 100%; border-radius: 4px; display: block; }}
  .thumbs figcaption {{ font-size: .75rem; color: #666; }}
</style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")
    return out_path


def render_markdown(result: AskResult, *, db_path: str | Path) -> str:
    run_dir = Path(db_path).expanduser().resolve().parent
    lines = [f"# {result.question}", ""]
    if result.unsupported:
        lines.append(f"> {result.unsupported}")
        return "\n".join(lines) + "\n"

    lines.append(f"- parser: `{result.parser}`" + (f" (`{result.model}`)" if result.model else ""))
    lines.append(f"- total: {result.elapsed_ms:.0f} ms")
    if result.notes:
        lines.append(f"- notes: {'; '.join(result.notes)}")
    if result.plan is not None:
        lines.extend(["", "## Plan", "", "```json", json.dumps(result.plan.to_dict(), indent=2), "```"])

    for query_result in result.results:
        lines.extend(["", f"## {query_result.label}", "", "```sql", query_result.sql, "```", ""])
        if not query_result.rows:
            lines.append("_No rows._")
            continue
        lines.append("| " + " | ".join(query_result.columns) + " |")
        lines.append("|" + "|".join("---" for _ in query_result.columns) + "|")
        for row in query_result.rows[:20]:
            lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
        images = _image_paths(query_result.columns, query_result.rows, run_dir)
        if images:
            lines.extend(["", "![evidence](" + str(images[0]) + ")"])
    return "\n".join(lines) + "\n"


def _html_table(columns: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return ["<p class='meta'>No rows.</p>"]
    out = ["<table><thead><tr>"]
    out.extend(f"<th>{html.escape(str(c))}</th>" for c in columns)
    out.append("</tr></thead><tbody>")
    for row in rows[:30]:
        out.append("<tr>")
        out.extend(f"<td>{html.escape(_fmt(v))}</td>" for v in row)
        out.append("</tr>")
    out.append("</tbody></table>")
    if len(rows) > 30:
        out.append(f"<p class='meta'>… {len(rows) - 30} more row(s)</p>")
    return out


def _html_images(columns: list[str], rows: list[list[Any]], run_dir: Path) -> list[str]:
    paths = _image_paths(columns, rows, run_dir)
    if not paths:
        return []
    out = ["<div class='thumbs'>"]
    for path in paths:
        rel = path.name
        out.append(f"<figure><img src='annotated/{html.escape(rel)}' alt='{html.escape(rel)}'>"
                   f"<figcaption>{html.escape(rel)}</figcaption></figure>")
    out.append("</div>")
    return out


def _image_paths(columns: list[str], rows: list[list[Any]], run_dir: Path) -> list[Path]:
    if "frame_idx" not in columns:
        return []
    index = columns.index("frame_idx")
    annotated = run_dir / "annotated"
    if not annotated.exists():
        return []
    out: list[Path] = []
    for row in rows[:12]:
        try:
            frame = int(row[index])
        except (TypeError, ValueError):
            continue
        path = annotated / f"frame_{frame:05d}.jpg"
        if not path.exists():
            path = _nearest(annotated, frame)
        if path and path not in out:
            out.append(path)
    return out


def _nearest(annotated: Path, frame: int) -> Path | None:
    candidates = sorted(annotated.glob("frame_*.jpg"))
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda p: abs(int(p.stem.split("_")[1]) - frame),
    )
    return best


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)
