"""Tests for the natural-language investigation layer.

These tests do not need ExecuTorch, OpenCV or a language model. The optional
LLM test is skipped unless llama.cpp and the GGUF model are present.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from arten.query.compile import compile_plan
from arten.query.execute import ask, execute_plan, load_meta
from arten.query.index import build_index
from arten.query.parser_rules import parse
from arten.query.schema import PlanError, QueryPlan
from arten.query.semantics import UnsupportedQuery

FIXTURE = Path(__file__).parent / "fixtures" / "run_mini"
REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "models" / "llm" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    return build_index(FIXTURE, tmp_path / "evidence.db")


@pytest.fixture()
def meta(db: Path) -> dict:
    return load_meta(db)


def test_index_row_counts(db: Path, meta: dict) -> None:
    assert meta["rows"] == {"frames": 5, "detections": 14, "tracks": 4}
    assert "person" in meta["classes"]
    assert meta["has_colour"] is True
    assert meta["has_attributes"] is True
    assert meta["has_zone"] is False


def test_index_derives_crossed_tracks(db: Path) -> None:
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        rows = dict(conn.execute("SELECT track_id, crossed FROM tracks").fetchall())
    finally:
        conn.close()
    assert rows[1] == 1  # right-moving person crosses the frame
    assert rows[4] == 1  # left-moving person crosses the frame
    assert rows[2] == 0  # stationary car
    assert rows[3] == 0  # single-frame bus


def test_rules_count_people(meta: dict) -> None:
    plan, _ = parse("how many people?", meta)
    assert plan.metric == "unique_tracks"
    assert plan.classes == ["person"]


def test_rules_count_detections(meta: dict) -> None:
    plan, _ = parse("how many person detections?", meta)
    assert plan.metric == "detections"
    assert plan.classes == ["person"]


def test_rules_peak(meta: dict) -> None:
    plan, _ = parse("what was the peak person count?", meta)
    assert plan.metric == "peak"


def test_rules_list_frames(meta: dict) -> None:
    plan, _ = parse("show frames with more than 1 people", meta)
    assert plan.intent == "list_frames"
    assert plan.min_count == 2


def test_rules_colour(meta: dict) -> None:
    plan, _ = parse("which cars were blue?", meta)
    assert plan.intent == "list_tracks"
    assert plan.colour == ["blue"]
    assert plan.classes == ["car"]


def test_rules_crossed(meta: dict) -> None:
    plan, _ = parse("how many people crossed?", meta)
    assert plan.metric == "unique_tracks"
    assert plan.crossed is True


def test_rules_unsupported_term(meta: dict) -> None:
    with pytest.raises(UnsupportedQuery, match="Auto-rickshaw"):
        parse("how many auto-rickshaws?", meta)


def test_rules_refuses_sql(meta: dict) -> None:
    with pytest.raises(UnsupportedQuery, match="cannot run SQL"):
        parse("DROP TABLE detections", meta)


def test_rules_co_presence(meta: dict) -> None:
    plan, _ = parse("was there a person and a bus within 2 seconds?", meta)
    assert plan.intent == "count"
    assert plan.metric == "frames"
    assert set(plan.co_presence_classes) == {"person", "bus"}


def test_rules_co_presence_frames(meta: dict) -> None:
    plan, _ = parse("show me frames where a person and a bus were within 2 seconds", meta)
    assert plan.intent == "list_frames"
    assert set(plan.co_presence_classes) == {"person", "bus"}


def test_execute_count(db: Path, meta: dict) -> None:
    plan, _ = parse("how many people?", meta)
    results = execute_plan(db, plan, meta)
    assert results[0].rows == [[2]]


def test_execute_detections(db: Path, meta: dict) -> None:
    plan, _ = parse("how many person detections?", meta)
    results = execute_plan(db, plan, meta)
    assert results[0].rows == [[8]]


def test_execute_peak(db: Path, meta: dict) -> None:
    plan, _ = parse("what was the peak person count?", meta)
    results = execute_plan(db, plan, meta)
    assert results[0].rows[0][2] == 2


def test_execute_list_frames(db: Path, meta: dict) -> None:
    plan, _ = parse("show frames with more than 1 people", meta)
    results = execute_plan(db, plan, meta)
    assert [row[0] for row in results[0].rows] == [2, 3, 4]


def test_execute_colour(db: Path, meta: dict) -> None:
    plan, _ = parse("which cars were blue?", meta)
    results = execute_plan(db, plan, meta)
    assert results[0].rows[0][0] == 2


def test_execute_co_presence(db: Path, meta: dict) -> None:
    plan, _ = parse("was there a person and a bus within 2 seconds?", meta)
    results = execute_plan(db, plan, meta)
    assert results[0].rows == [[1]]


def test_ask_unsupported(db: Path) -> None:
    result = ask(db, "how many auto-rickshaws?", parser="rules")
    assert result.unsupported
    assert "Auto-rickshaw" in result.unsupported


def test_compile_is_parameterised(meta: dict) -> None:
    plan, _ = parse("how many people?", meta)
    compiled = compile_plan(plan, meta)[0]
    assert "?" in compiled.sql
    assert "person" not in compiled.sql
    assert "person" in compiled.params


def test_schema_rejects_unknown_intent() -> None:
    with pytest.raises(PlanError):
        QueryPlan(intent="delete_everything")


def test_schema_rejects_bad_limit() -> None:
    with pytest.raises(PlanError):
        QueryPlan(intent="count", limit=999)


def test_schema_normalises_scalar_colour() -> None:
    plan = QueryPlan(intent="list_tracks", colour="Blue")
    assert plan.colour == ["blue"]


def test_schema_rejects_unknown_colour() -> None:
    with pytest.raises(PlanError):
        QueryPlan(intent="list_tracks", colour=["chartreuse"])


@pytest.mark.skipif(
    shutil.which("llama-cli") is None or not MODEL.exists(),
    reason="llama.cpp or the GGUF model is not installed",
)
def test_llm_classifier(meta: dict) -> None:
    from arten.query.parser_llm import LLMParser

    parser = LLMParser()
    assert parser.available()
    plan, notes = parser.parse("how many people?", meta)
    assert plan.metric == "unique_tracks"
    assert any("llm label" in note for note in notes)
