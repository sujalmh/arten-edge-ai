"""Execute a validated query plan against an evidence index."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compile import CompiledQuery, compile_plan
from .schema import PlanError, QueryPlan
from .semantics import UnsupportedQuery


@dataclass
class QueryResult:
    label: str
    columns: list[str]
    rows: list[list[Any]]
    sql: str
    params: list[Any]
    elapsed_ms: float


@dataclass
class AskResult:
    question: str
    parser: str
    model: str | None
    plan: QueryPlan | None
    results: list[QueryResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unsupported: str | None = None
    elapsed_ms: float = 0.0


def load_meta(db_path: str | Path) -> dict[str, Any]:
    """Load index metadata from the sidecar file, falling back to the table."""
    db_path = Path(db_path).expanduser().resolve()
    for sidecar in (
        db_path.with_name(db_path.name + ".meta.json"),
        db_path.with_name("index_meta.json"),  # legacy name
    ):
        if sidecar.exists():
            return json.loads(sidecar.read_text(encoding="utf-8"))

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
    finally:
        conn.close()
    meta: dict[str, Any] = {}
    for key, value in rows:
        try:
            meta[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            meta[key] = value
    return meta


def execute_plan(
    db_path: str | Path,
    plan: QueryPlan,
    meta: dict[str, Any],
) -> list[QueryResult]:
    compiled = compile_plan(plan, meta)
    db_path = Path(db_path).expanduser().resolve()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out: list[QueryResult] = []
    try:
        for index, query in enumerate(compiled):
            started = time.perf_counter()
            cursor = conn.execute(query.sql, query.params)
            rows = cursor.fetchall()
            elapsed = (time.perf_counter() - started) * 1000.0
            columns = [d[0] for d in cursor.description] if cursor.description else []
            label = query.description
            if plan.intent == "compare":
                label = "first half" if index == 0 else "second half"
            out.append(
                QueryResult(
                    label=label,
                    columns=columns,
                    rows=[list(row) for row in rows],
                    sql=query.sql,
                    params=list(query.params),
                    elapsed_ms=round(elapsed, 3),
                )
            )
    finally:
        conn.close()
    return out


def ask(
    db_path: str | Path,
    question: str,
    *,
    parser: str = "auto",
    model_path: str | Path | None = None,
    llama_bin: str | None = None,
    grammar_path: str | Path | None = None,
    timeout_s: float = 90.0,
    limit: int | None = None,
) -> AskResult:
    """Parse a question, execute it, and return an auditable result."""
    started = time.perf_counter()
    meta = load_meta(db_path)

    try:
        plan, parser_name, model_name, notes = _parse(
            question,
            meta,
            parser=parser,
            model_path=model_path,
            llama_bin=llama_bin,
            grammar_path=grammar_path,
            timeout_s=timeout_s,
        )
    except UnsupportedQuery as exc:
        return AskResult(
            question=question,
            parser=parser,
            model=None,
            plan=None,
            unsupported=str(exc),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except PlanError as exc:
        return AskResult(
            question=question,
            parser=parser,
            model=None,
            plan=None,
            unsupported=f"Could not turn that into a query: {exc}",
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    if limit is not None:
        plan.limit = max(1, min(50, int(limit)))
    results = execute_plan(db_path, plan, meta)
    return AskResult(
        question=question,
        parser=parser_name,
        model=model_name,
        plan=plan,
        results=results,
        notes=notes,
        elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


def _parse(
    question: str,
    meta: dict[str, Any],
    *,
    parser: str,
    model_path: str | Path | None,
    llama_bin: str | None,
    grammar_path: str | Path | None,
    timeout_s: float,
) -> tuple[QueryPlan, str, str | None, list[str]]:
    from .parser_rules import parse as parse_rules

    if parser == "rules":
        plan, notes = parse_rules(question, meta)
        return plan, "rules", None, notes

    if parser == "llm":
        from .parser_llm import LLMParser

        llm = LLMParser(
            model_path=model_path,
            llama_bin=llama_bin,
            grammar_path=grammar_path,
            timeout_s=timeout_s,
        )
        plan, notes = llm.parse(question, meta)
        return plan, "llm", llm.model_name, notes

    if parser == "auto":
        # Rules first: fast, deterministic, and correct for the common questions.
        # The local model is a fallback for phrasing the rules cannot map.
        try:
            plan, notes = parse_rules(question, meta)
            return plan, "rules", None, notes
        except UnsupportedQuery as exc:
            if "could not map" not in str(exc):
                raise
            original = exc

        try:
            from .parser_llm import LLMParser

            llm = LLMParser(
                model_path=model_path,
                llama_bin=llama_bin,
                grammar_path=grammar_path,
                timeout_s=timeout_s,
            )
            if not llm.available():
                raise original
            plan, notes = llm.parse(question, meta)
            return plan, "llm", llm.model_name, notes
        except UnsupportedQuery:
            raise
        except Exception:
            raise original

    raise PlanError(f"unknown parser: {parser!r}")


def format_answer(result: AskResult, *, max_rows: int = 12, explain: bool = False) -> str:
    lines: list[str] = [f"Question: {result.question}"]
    if result.unsupported:
        lines.append(f"Answer:   not supported — {result.unsupported}")
        lines.append(f"Time:     {result.elapsed_ms:.1f} ms")
        return "\n".join(lines)

    assert result.plan is not None
    lines.append(f"Parser:   {result.parser}" + (f" ({result.model})" if result.model else ""))
    if explain:
        lines.append(f"Plan:     {json.dumps(result.plan.to_dict(), ensure_ascii=False)}")
    if result.notes:
        lines.append("Notes:    " + "; ".join(result.notes))

    for query_result in result.results:
        lines.append(f"SQL:      {query_result.sql}")
        lines.append(f"Params:   {query_result.params}")
        if not query_result.rows:
            lines.append("Result:   no rows")
            continue
        lines.append(f"Result:   {query_result.label}")
        lines.extend(_format_table(query_result.columns, query_result.rows, max_rows=max_rows))
        lines.append(f"Query:    {query_result.elapsed_ms:.1f} ms")
    lines.append(f"Total:    {result.elapsed_ms:.1f} ms")
    return "\n".join(lines)


def _format_table(columns: list[str], rows: list[list[Any]], *, max_rows: int) -> list[str]:
    display_rows = rows[:max_rows]
    string_rows = [[_fmt(v) for v in row] for row in display_rows]
    widths = [len(c) for c in columns]
    for row in string_rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (w + 2) for w in widths) + "+"

    out = [line("-")]
    out.append("|" + "|".join(f" {c:<{w}} " for c, w in zip(columns, widths)) + "|")
    out.append(line("="))
    for row in string_rows:
        out.append("|" + "|".join(f" {v:<{w}} " for v, w in zip(row, widths)) + "|")
    out.append(line("-"))
    if len(rows) > max_rows:
        out.append(f"  … {len(rows) - max_rows} more row(s)")
    return out


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)
