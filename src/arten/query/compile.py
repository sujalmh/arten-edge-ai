"""Compile a validated QueryPlan to parameterised SQLite SQL.

No value from the plan is interpolated into the SQL string. Identifiers that
must be interpolated (GROUP BY / ORDER BY) come from closed whitelists. This is
the boundary that lets a language model propose a plan without ever executing
arbitrary SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import PlanError, QueryPlan
from .semantics import UnsupportedQuery


@dataclass
class CompiledQuery:
    sql: str
    params: list[Any]
    description: str


def compile_plan(plan: QueryPlan, meta: dict[str, Any]) -> list[CompiledQuery]:
    """Return one query, or two for a ``compare`` intent."""
    if plan.intent == "compare":
        return _compile_compare(plan, meta)
    if plan.intent == "count":
        return [_compile_count(plan, meta)]
    if plan.intent == "list_frames":
        return [_compile_list_frames(plan, meta)]
    if plan.intent == "list_tracks":
        return [_compile_list_tracks(plan, meta)]
    if plan.intent == "aggregate":
        return [_compile_aggregate(plan, meta)]
    if plan.intent == "explain":
        return [CompiledQuery("SELECT 1 AS ok", [], "explain")]
    raise PlanError(f"unsupported intent: {plan.intent}")


# ---------------------------------------------------------------------- #
# Intent compilers
# ---------------------------------------------------------------------- #
def _compile_count(plan: QueryPlan, meta: dict[str, Any]) -> CompiledQuery:
    where, params = _where(plan, meta)

    if plan.metric == "unique_tracks":
        sql = f"SELECT COUNT(DISTINCT d.track_id) AS count FROM detections d {where} AND d.track_id IS NOT NULL"
        return CompiledQuery(sql, params, "unique tracks")
    if plan.metric == "frames":
        sql = f"SELECT COUNT(DISTINCT d.frame_idx) AS count FROM detections d {where}"
        return CompiledQuery(sql, params, "distinct frames")
    if plan.metric == "peak":
        sql = (
            "SELECT d.frame_idx, d.timestamp_s, COUNT(*) AS count "
            f"FROM detections d {where} "
            "GROUP BY d.frame_idx, d.timestamp_s ORDER BY count DESC LIMIT 1"
        )
        return CompiledQuery(sql, params, "peak per-frame count")
    if plan.metric == "mean_dwell":
        where_t, params_t = _where_tracks(plan, meta)
        sql = f"SELECT AVG(t.dwell_s) AS mean FROM tracks t {where_t}"
        return CompiledQuery(sql, params_t, "mean dwell")
    if plan.metric == "mean_speed":
        where_t, params_t = _where_tracks(plan, meta)
        sql = f"SELECT AVG(t.speed_px_s) AS mean FROM tracks t {where_t}"
        return CompiledQuery(sql, params_t, "mean speed")
    if plan.metric == "mean_score":
        sql = f"SELECT AVG(d.score) AS mean FROM detections d {where}"
        return CompiledQuery(sql, params, "mean confidence")

    sql = f"SELECT COUNT(*) AS count FROM detections d {where}"
    return CompiledQuery(sql, params, "detections")


def _compile_list_frames(plan: QueryPlan, meta: dict[str, Any]) -> CompiledQuery:
    where, params = _where(plan, meta)
    having = ""
    if plan.min_count is not None:
        having += " AND COUNT(*) >= ?"
        params.append(plan.min_count)
    if plan.max_count is not None:
        having += " AND COUNT(*) <= ?"
        params.append(plan.max_count)

    order = "count DESC"
    if plan.order_by == "timestamp_s":
        order = "d.timestamp_s " + ("ASC" if plan.order_dir == "asc" else "DESC")

    sql = (
        "SELECT d.frame_idx, d.timestamp_s, COUNT(*) AS count "
        f"FROM detections d {where} "
        "GROUP BY d.frame_idx, d.timestamp_s "
        f"HAVING 1=1 {having} "
        f"ORDER BY {order} LIMIT ?"
    )
    params.append(plan.limit)
    return CompiledQuery(sql, params, "frames matching the filter")


def _compile_list_tracks(plan: QueryPlan, meta: dict[str, Any]) -> CompiledQuery:
    where, params = _where_tracks(plan, meta)
    order_field = plan.order_by or "speed_px_s"
    if order_field not in {"speed_px_s", "dwell_s", "score", "count"}:
        order_field = "speed_px_s"
    order = {
        "speed_px_s": "t.speed_px_s",
        "dwell_s": "t.dwell_s",
        "score": "t.mean_score",
        "count": "t.frames",
    }[order_field]
    direction = "ASC" if plan.order_dir == "asc" else "DESC"

    sql = (
        "SELECT t.track_id, t.class, t.direction, t.speed_px_s, t.dwell_s, "
        "t.frames, t.colour, t.crossed, t.mean_score "
        f"FROM tracks t {where} "
        f"ORDER BY {order} {direction} LIMIT ?"
    )
    params.append(plan.limit)
    return CompiledQuery(sql, params, "tracks matching the filter")


def _compile_aggregate(plan: QueryPlan, meta: dict[str, Any]) -> CompiledQuery:
    where, params = _where(plan, meta)
    fields = plan.group_by or ["class"]

    select_parts: list[str] = []
    group_parts: list[str] = []
    for field in fields:
        if field == "minute":
            expr = "CAST(d.timestamp_s / 60 AS INTEGER)"
            alias = "minute"
        elif field == "class":
            expr = "d.class"
            alias = "class"
        elif field == "direction":
            expr = "COALESCE(d.direction, 'unknown')"
            alias = "direction"
        elif field == "colour":
            expr = "COALESCE(d.colour, 'unknown')"
            alias = "colour"
        elif field == "half":
            expr = (
                "CASE WHEN d.timestamp_s < ("
                "SELECT (MIN(timestamp_s) + MAX(timestamp_s)) / 2.0 FROM frames"
                ") THEN 'first' ELSE 'second' END"
            )
            alias = "half"
        else:  # pragma: no cover - guarded by schema validation
            raise PlanError(f"unsupported group_by field: {field}")
        select_parts.append(f"{expr} AS {alias}")
        group_parts.append(expr)

    sql = (
        "SELECT " + ", ".join(select_parts) + ", COUNT(*) AS count "
        f"FROM detections d {where} "
        "GROUP BY " + ", ".join(group_parts) + " "
        "ORDER BY count DESC LIMIT ?"
    )
    params.append(plan.limit)
    return CompiledQuery(sql, params, "aggregate by " + ", ".join(fields))


def _compile_compare(plan: QueryPlan, meta: dict[str, Any]) -> list[CompiledQuery]:
    """Split the run into two halves and run the same count twice."""
    start, end = _time_range(meta)
    mid = (start + end) / 2.0

    first = QueryPlan.from_dict(plan.to_dict())
    first.intent = "count"
    first.time_start_s = start
    first.time_end_s = mid

    second = QueryPlan.from_dict(plan.to_dict())
    second.intent = "count"
    second.time_start_s = mid
    second.time_end_s = end

    return [
        _compile_count(first, meta),
        _compile_count(second, meta),
    ]


# ---------------------------------------------------------------------- #
# WHERE builders
# ---------------------------------------------------------------------- #
def _where(plan: QueryPlan, meta: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = ["1=1"]
    params: list[Any] = []

    if plan.classes:
        clauses.append(f"d.class IN ({_placeholders(len(plan.classes))})")
        params.extend(plan.classes)
    if plan.time_start_s is not None:
        clauses.append("d.timestamp_s >= ?")
        params.append(plan.time_start_s)
    if plan.time_end_s is not None:
        clauses.append("d.timestamp_s <= ?")
        params.append(plan.time_end_s)
    if plan.frame_start is not None:
        clauses.append("d.frame_idx >= ?")
        params.append(plan.frame_start)
    if plan.frame_end is not None:
        clauses.append("d.frame_idx <= ?")
        params.append(plan.frame_end)
    if plan.confidence_min is not None:
        clauses.append("d.score >= ?")
        params.append(plan.confidence_min)
    if plan.track_ids:
        clauses.append(f"d.track_id IN ({_placeholders(len(plan.track_ids))})")
        params.extend(plan.track_ids)
    if plan.direction is not None:
        _require(meta, "has_attributes", "direction filtering")
        clauses.append("d.direction = ?")
        params.append(plan.direction)
    if plan.speed_px_s_min is not None:
        _require(meta, "has_attributes", "speed filtering")
        clauses.append("d.speed_px_s >= ?")
        params.append(plan.speed_px_s_min)
    if plan.dwell_s_min is not None:
        _require(meta, "has_attributes", "dwell filtering")
        clauses.append("d.dwell_s >= ?")
        params.append(plan.dwell_s_min)
    if plan.colour:
        _require(meta, "has_colour", "colour filtering")
        clauses.append(f"d.colour IN ({_placeholders(len(plan.colour))})")
        params.extend(plan.colour)
    if plan.zone is not None:
        raise UnsupportedQuery(
            "Zone filtering is not populated in this index. Re-run with --zone and "
            "rebuild the index after zone support is added."
        )
    if plan.crossed:
        clauses.append("d.track_id IN (SELECT track_id FROM tracks WHERE crossed = 1)")
    if plan.ood:
        clauses.append("EXISTS (SELECT 1 FROM frames f WHERE f.frame_idx = d.frame_idx AND f.ood = 1)")

    if plan.co_presence_classes:
        clauses.append(
            "EXISTS (SELECT 1 FROM detections d2 "
            "WHERE d2.frame_idx = d.frame_idx "
            f"AND d2.class IN ({_placeholders(len(plan.co_presence_classes))}) "
            "AND d2.class != d.class "
            "AND ABS(d2.timestamp_s - d.timestamp_s) <= ?)"
        )
        params.extend(plan.co_presence_classes)
        params.append(plan.co_presence_within_s or 0.0)

    return "WHERE " + " AND ".join(clauses), params


def _where_tracks(plan: QueryPlan, meta: dict[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = ["1=1"]
    params: list[Any] = []

    if plan.classes:
        clauses.append(f"t.class IN ({_placeholders(len(plan.classes))})")
        params.extend(plan.classes)
    if plan.track_ids:
        clauses.append(f"t.track_id IN ({_placeholders(len(plan.track_ids))})")
        params.extend(plan.track_ids)
    if plan.direction is not None:
        _require(meta, "has_attributes", "direction filtering")
        clauses.append("t.direction = ?")
        params.append(plan.direction)
    if plan.speed_px_s_min is not None:
        _require(meta, "has_attributes", "speed filtering")
        clauses.append("t.speed_px_s >= ?")
        params.append(plan.speed_px_s_min)
    if plan.dwell_s_min is not None:
        _require(meta, "has_attributes", "dwell filtering")
        clauses.append("t.dwell_s >= ?")
        params.append(plan.dwell_s_min)
    if plan.colour:
        _require(meta, "has_colour", "colour filtering")
        clauses.append(f"t.colour IN ({_placeholders(len(plan.colour))})")
        params.extend(plan.colour)
    if plan.crossed:
        clauses.append("t.crossed = 1")
    if plan.time_start_s is not None or plan.time_end_s is not None:
        # A track is in range if it overlaps the window at all.
        if plan.time_start_s is not None:
            clauses.append("t.last_frame >= ?")
            params.append(int(plan.time_start_s))
        if plan.time_end_s is not None:
            clauses.append("t.first_frame <= ?")
            params.append(int(plan.time_end_s))

    return "WHERE " + " AND ".join(clauses), params


def _require(meta: dict[str, Any], key: str, label: str) -> None:
    if not meta.get(key):
        raise UnsupportedQuery(f"This index does not contain {label}.")


def _time_range(meta: dict[str, Any]) -> tuple[float, float]:
    bounds = meta.get("time_range_s") or [0.0, 0.0]
    return float(bounds[0]), float(bounds[1])


def _placeholders(n: int) -> str:
    return ",".join("?" for _ in range(n))
