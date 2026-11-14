"""Optional local-LLM parser.

Measured on Qwen2.5-0.5B: free-form JSON generation is unreliable at this size
(the model tends to copy the schema or emit the wrong keys). A closed-label
classification with a GBNF grammar is reliable and much faster, so the model
chooses *what kind of question* this is, while the deterministic slot extractor
chooses classes, times and numbers.

The output is still a validated QueryPlan; the model never writes SQL.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .parser_rules import extract_slots
from .schema import QueryPlan
from .semantics import UnsupportedQuery, resolve_time_word

LABELS = (
    "count",
    "unique_count",
    "list_frames",
    "fastest",
    "direction_count",
    "dwell",
    "colour",
    "co_presence",
    "compare",
    "peak",
    "unsupported",
)

EXAMPLES = (
    ("how many people?", "count"),
    ("how many unique people?", "unique_count"),
    ("show frames with more than 5 people", "list_frames"),
    ("which track moved fastest", "fastest"),
    ("how many tracks moved left", "direction_count"),
    ("how long did people stay", "dwell"),
    ("which cars were red", "colour"),
    ("was there a bus and a person within 2 seconds", "co_presence"),
    ("compare the first half and the second half", "compare"),
    ("what was the peak person count", "peak"),
    ("how many auto-rickshaws", "unsupported"),
)


class LLMParseError(ValueError):
    """Raised when the local model cannot produce a usable label."""


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parents[3]


class LLMParser:
    """Classify a question with a local GGUF model, then build a QueryPlan."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        llama_bin: str | Path | None = None,
        grammar_path: str | Path | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        repo_root = _repo_root()
        self.model_path = Path(
            model_path or repo_root / "models/llm/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ).expanduser()
        self.grammar_path = Path(
            grammar_path or repo_root / "grammars/query_labels.gbnf"
        ).expanduser()
        self.llama_bin = self._resolve_bin(llama_bin, repo_root)
        self.timeout_s = timeout_s
        self.last_latency_ms: float | None = None

    @property
    def model_name(self) -> str:
        return self.model_path.name

    def available(self) -> bool:
        return self.model_path.exists() and self.llama_bin is not None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse(self, question: str, meta: dict[str, Any]) -> tuple[QueryPlan, list[str]]:
        slots = extract_slots(question, meta)

        # Deterministic guards: when the question contains an unambiguous cue,
        # trust the cue rather than the classifier. This is what keeps a small
        # model from overriding "unique", "crossed", "peak" or "more than N".
        label = None
        if slots["compare"]:
            label = "compare"
        elif slots["list_frames"]:
            label = "list_frames"
        elif slots["co_presence"]:
            label = "co_presence"
        elif slots["crossed"]:
            label = "unique_count"
        elif slots["unique"]:
            label = "unique_count"
        elif slots["peak"]:
            label = "peak"

        source = "cue"
        if label is None:
            label = self.classify(question)
            source = "llm"

        if label == "unsupported":
            raise UnsupportedQuery(
                "The local model classified this as outside the index. "
                "Try a question about counts, tracks, directions, times, colours or co-presence."
            )

        plan = self._plan_for_label(label, slots, meta)
        notes = [
            f"llm label={label} ({source})",
            f"model={self.model_name}",
            *slots["notes"],
        ]
        if self.last_latency_ms is not None:
            notes.append(f"llm={self.last_latency_ms:.0f} ms")
        return plan, notes

    def classify(self, question: str) -> str:
        prompt = self._classifier_prompt(question)
        started = time.perf_counter()
        output = self._run(prompt, max_tokens=8)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        for line in output.splitlines():
            candidate = line.strip().strip(":")
            if candidate in LABELS:
                return candidate

        raise LLMParseError(
            "The local model did not return a recognised label. "
            "Try `--parser rules` or rephrase the question."
        )

    # ------------------------------------------------------------------ #
    # Plan construction
    # ------------------------------------------------------------------ #
    def _plan_for_label(self, label: str, slots: dict[str, Any], meta: dict[str, Any]) -> QueryPlan:
        classes = slots["classes"]
        time_start = slots["time_window"][0] if slots["time_window"] else None
        time_end = slots["time_window"][1] if slots["time_window"] else None

        if label == "count":
            return QueryPlan(
                intent="count",
                metric="detections" if slots.get("instances") else "unique_tracks",
                classes=classes,
                time_start_s=time_start,
                time_end_s=time_end,
                output=["count"],
            )
        if label == "unique_count":
            return QueryPlan(
                intent="count",
                metric="unique_tracks",
                classes=classes,
                crossed=True if slots["crossed"] else None,
                time_start_s=time_start,
                time_end_s=time_end,
                output=["count"],
            )
        if label == "peak":
            return QueryPlan(
                intent="count",
                metric="peak",
                classes=classes,
                time_start_s=time_start,
                time_end_s=time_end,
                output=["count"],
            )
        if label == "list_frames":
            co = slots["co_presence"] or {}
            return QueryPlan(
                intent="list_frames",
                classes=classes,
                direction=slots["direction"],
                colour=slots["colour"],
                crossed=True if slots["crossed"] else None,
                min_count=slots["min_count"],
                max_count=slots["max_count"],
                co_presence_classes=co.get("classes") or [],
                co_presence_within_s=co.get("within_s"),
                time_start_s=time_start,
                time_end_s=time_end,
                order_by="count",
                limit=12,
                output=["frames", "contact_sheet"],
            )
        if label == "fastest":
            return QueryPlan(
                intent="list_tracks",
                classes=classes,
                direction=slots["direction"],
                colour=slots["colour"],
                order_by="speed_px_s",
                order_dir="desc",
                limit=5,
                output=["rows"],
            )
        if label == "direction_count":
            if not slots["direction"]:
                raise LLMParseError("The question looks directional but names no direction.")
            return QueryPlan(
                intent="count",
                metric="unique_tracks",
                classes=classes,
                direction=slots["direction"],
                output=["count"],
            )
        if label == "dwell":
            return QueryPlan(
                intent="count",
                metric="mean_dwell",
                classes=classes,
                output=["count"],
            )
        if label == "colour":
            if not slots["colour"]:
                raise LLMParseError("The question mentions colour but names none.")
            return QueryPlan(
                intent="list_tracks",
                classes=classes,
                colour=slots["colour"],
                order_by="count",
                limit=12,
                output=["rows"],
            )
        if label == "co_presence":
            co = slots["co_presence"]
            if not co:
                raise LLMParseError(
                    "A co-presence question needs two classes and a time window."
                )
            return QueryPlan(
                intent="count",
                metric="frames",
                classes=co["classes"],
                co_presence_classes=co["classes"],
                co_presence_within_s=co["within_s"],
                output=["count"],
            )
        if label == "compare":
            parts = slots["compare"]
            if not parts:
                raise LLMParseError("A comparison needs two time phrases.")
            first = resolve_time_word(
                parts[0], duration_s=slots["duration_s"], start_time_iso=slots["start_time"]
            )
            second = resolve_time_word(
                parts[1], duration_s=slots["duration_s"], start_time_iso=slots["start_time"]
            )
            return QueryPlan(
                intent="compare",
                classes=classes,
                time_start_s=first[0],
                time_end_s=second[1],
                group_by=["half"],
                output=["count"],
            )

        raise LLMParseError(f"Unsupported label: {label}")

    # ------------------------------------------------------------------ #
    # Model invocation
    # ------------------------------------------------------------------ #
    def _classifier_prompt(self, question: str) -> str:
        examples = "\n".join(f"{q} -> {label}" for q, label in EXAMPLES)
        return (
            "Label the question with exactly one label from: "
            + ", ".join(LABELS)
            + ".\nExamples:\n"
            + examples
            + f"\nQuestion: {question.strip()}\nLabel:"
        )

    def _run(self, prompt: str, max_tokens: int) -> str:
        if self.llama_bin is None:
            raise LLMParseError("llama-cli not found; install llama.cpp or use --parser rules.")
        if not self.model_path.exists():
            raise LLMParseError(f"model not found: {self.model_path}")

        base = [
            str(self.llama_bin),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(max_tokens),
            "--log-disable",
            "-st",
            "--temp",
            "0",
        ]
        if self.grammar_path.exists():
            base.extend(["--grammar-file", str(self.grammar_path)])

        # --no-jinja keeps this a raw completion task. Older builds lack it.
        for extra in (["--no-jinja"], []):
            cmd = base + extra
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise LLMParseError(f"local model timed out after {self.timeout_s:.0f}s") from exc
            if completed.returncode == 0 and completed.stdout.strip():
                return completed.stdout
        stderr = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "unknown error"
        raise LLMParseError(f"llama-cli failed: {stderr}")

    @staticmethod
    def _resolve_bin(llama_bin: str | Path | None, repo_root: Path) -> Path | None:
        if llama_bin:
            return Path(llama_bin).expanduser()
        found = shutil.which("llama-cli")
        if found:
            return Path(found)
        local = repo_root / "vendor/llama.cpp/build/bin/llama-cli"
        return local if local.exists() else None
