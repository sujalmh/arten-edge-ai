"""Rule-based question parser.

This is the always-available fallback. It is deliberately narrow: ten intents,
a synonym table, and an explicit refusal for anything outside the index. The
language model is an optional upgrade, not a dependency.
"""

from __future__ import annotations

import re
from typing import Any

from .schema import QueryPlan
from .semantics import (
    CLASS_SYNONYMS,
    COLOUR_SYNONYMS,
    DIRECTION_SYNONYMS,
    GROUP_SYNONYMS,
    UNSUPPORTED_TERMS,
    UnsupportedQuery,
    expand_class_words,
    resolve_time_word,
)

_SQL_WORDS = re.compile(r"\b(drop|delete|insert|update|alter|attach|detach|pragma|vacuum)\b|;", re.I)
_NUMBER = r"(\d+(?:\.\d+)?)"


def parse(question: str, meta: dict[str, Any]) -> tuple[QueryPlan, list[str]]:
    q = " " + re.sub(r"\s+", " ", question.strip().lower()) + " "
    available = {c.lower() for c in (meta.get("classes") or [])}
    duration = meta.get("duration_s")
    start_time = meta.get("start_time")
    notes: list[str] = []

    if _SQL_WORDS.search(q):
        raise UnsupportedQuery(
            "I only answer questions about the recorded detections. I cannot run SQL, "
            "and the language model never writes SQL."
        )

    for term, reason in UNSUPPORTED_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", q):
            raise UnsupportedQuery(reason)

    classes, group, class_notes = _find_classes(q, available)
    notes.extend(class_notes)
    direction = _find_direction(q)
    colour = _find_colour(q)

    time_window = _find_time(q, duration=duration, start_time=start_time)
    if time_window:
        notes.append(f"time -> {time_window[0]:.1f}–{time_window[1]:.1f} s")

    # --- explicit refusals that need context --------------------------- #
    if " zone " in q or " in the zone " in q:
        if not meta.get("has_zone"):
            raise UnsupportedQuery(
                "This index has no zone. Re-run Arten with --zone, then rebuild the index."
            )
        raise UnsupportedQuery("Zone filtering is not populated in this index yet.")

    # --- list frames (before co-presence: "show frames where ...") ------ #
    if re.search(r"\b(show|list|which|find)\b.*\b(frames?|images?)\b", q):
        min_count, max_count = _find_count_bounds(q)
        co = re.search(
            rf"\bwithin\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b",
            q,
        )
        co_classes: list[str] = []
        within: float | None = None
        if co:
            candidate = _find_classes(q, available)[0]
            if len(candidate) >= 2:
                co_classes = candidate
                within = float(co.group(1))
        plan = QueryPlan(
            intent="list_frames",
            classes=classes,
            direction=direction,
            colour=colour,
            crossed=True if re.search(r"\bcross(ed|ing)?\b", q) else None,
            min_count=min_count,
            max_count=max_count,
            co_presence_classes=co_classes,
            co_presence_within_s=within,
            time_start_s=time_window[0] if time_window else None,
            time_end_s=time_window[1] if time_window else None,
            order_by="count",
            limit=12,
            output=["frames", "contact_sheet"],
        )
        notes.append(
            "frames"
            + (f" with >= {min_count}" if min_count is not None else "")
            + (f" and <= {max_count}" if max_count is not None else "")
            + (f" co-present {', '.join(co_classes)} within {within:g} s" if co_classes else "")
        )
        return plan, notes

    # --- co-presence ---------------------------------------------------- #
    co = re.search(
        rf"\bwithin\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b",
        q,
    )
    if co:
        within = float(co.group(1))
        found = _find_classes(q, available)[0]
        if len(found) < 2:
            raise UnsupportedQuery(
                "A co-presence question needs two classes, for example "
                "'was there a bus and a person within 2 seconds?'"
            )
        plan = QueryPlan(
            intent="count",
            metric="frames",
            classes=found,
            co_presence_classes=found,
            co_presence_within_s=within,
            output=["count"],
        )
        notes.append(f"co-presence within {within:g} s -> {', '.join(found)}")
        return plan, notes

    # --- compare -------------------------------------------------------- #
    if q.strip().startswith("compare") or " compare " in q:
        first, second = _split_compare(q)
        a = resolve_time_word(first, duration_s=duration, start_time_iso=start_time)
        b = resolve_time_word(second, duration_s=duration, start_time_iso=start_time)
        plan = QueryPlan(
            intent="compare",
            metric="detections",
            classes=classes,
            time_start_s=a[0],
            time_end_s=a[1],
            group_by=["half"],
            output=["count"],
        )
        notes.append(f"compare {first} {a} with {second} {b}")
        return plan, notes

    # --- fastest / slowest track --------------------------------------- #
    if re.search(r"\b(fastest|slowest|quickest)\b", q):
        descending = "slowest" not in q and "quickest" not in q
        plan = QueryPlan(
            intent="list_tracks",
            classes=classes,
            direction=direction,
            colour=colour,
            order_by="speed_px_s",
            order_dir="desc" if descending else "asc",
            limit=5,
            output=["rows"],
        )
        notes.append("rank tracks by speed")
        return plan, notes

    # --- mean dwell / speed / confidence -------------------------------- #
    if re.search(r"\b(how long|stay|dwell|wait)\b", q):
        plan = QueryPlan(intent="count", metric="mean_dwell", classes=classes, output=["count"])
        notes.append("mean dwell over tracks")
        return plan, notes
    if re.search(r"\baverage\s+(speed|velocity)\b", q) or "how fast" in q:
        plan = QueryPlan(intent="count", metric="mean_speed", classes=classes, output=["count"])
        notes.append("mean speed over tracks")
        return plan, notes
    if re.search(r"\baverage\s+(confidence|score)\b", q):
        plan = QueryPlan(intent="count", metric="mean_score", classes=classes, output=["count"])
        notes.append("mean detection confidence")
        return plan, notes

    # --- direction count ------------------------------------------------ #
    if direction and re.search(r"\b(moved|move|moving|went|going)\b", q):
        plan = QueryPlan(
            intent="count",
            metric="unique_tracks",
            classes=classes,
            direction=direction,
            output=["count"],
        )
        notes.append(f"direction -> {direction}")
        return plan, notes

    # --- colour listing ------------------------------------------------- #
    if colour and re.search(r"\b(which|show|list|what)\b", q):
        plan = QueryPlan(
            intent="list_tracks",
            classes=classes,
            colour=colour,
            order_by="count",
            limit=12,
            output=["rows"],
        )
        notes.append(f"colour -> {colour}")
        return plan, notes

    # --- crossed -------------------------------------------------------- #
    if re.search(r"\bcross(ed|ing)?\b", q):
        plan = QueryPlan(
            intent="count",
            metric="unique_tracks",
            classes=classes or ["person"],
            crossed=True,
            time_start_s=time_window[0] if time_window else None,
            time_end_s=time_window[1] if time_window else None,
            output=["count"],
        )
        notes.append(
            "crossed = track with >=3 frames and net displacement >= 25% of the frame"
        )
        return plan, notes

    # --- peak ----------------------------------------------------------- #
    if re.search(r"\bpeak\b|\bmost\b.*\bat (once|a time)\b|\bmaximum\b", q):
        plan = QueryPlan(intent="count", metric="peak", classes=classes, output=["count"])
        notes.append("peak per-frame count")
        return plan, notes

    # --- generic count -------------------------------------------------- #
    if re.search(r"\b(how many|count|number of|total)\b", q):
        unique = bool(re.search(r"\b(unique|distinct)\b", q))
        instances = bool(re.search(r"\b(detections?|instances?|boxes|per frame)\b", q))
        metric = "unique_tracks" if unique or not instances else "detections"
        plan = QueryPlan(
            intent="count",
            metric=metric,
            classes=classes,
            direction=direction,
            colour=colour,
            time_start_s=time_window[0] if time_window else None,
            time_end_s=time_window[1] if time_window else None,
            output=["count"],
        )
        notes.append("unique tracks" if metric == "unique_tracks" else "detections")
        return plan, notes

    # --- aggregate by class --------------------------------------------- #
    if re.search(r"\b(breakdown|by class|per class|distribution)\b", q):
        plan = QueryPlan(
            intent="aggregate",
            classes=classes,
            group_by=["class"],
            limit=20,
            output=["rows"],
        )
        notes.append("aggregate by class")
        return plan, notes

    raise UnsupportedQuery(
        "I could not map that question to the index. Try: 'how many people?', "
        "'show frames with more than 5 people', 'which track moved fastest?', "
        "'how many people crossed?', or 'was there a bus and a person within 2 seconds?'"
    )


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _find_classes(q: str, available: set[str]) -> tuple[list[str], str | None, list[str]]:
    words: list[str] = []
    seen: set[str] = set()
    for phrase in sorted(
        list(CLASS_SYNONYMS) + list(GROUP_SYNONYMS),
        key=len,
        reverse=True,
    ):
        if phrase in seen:
            continue
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            words.append(phrase)
            seen.add(phrase)
    if not words:
        return [], None, []
    resolution = expand_class_words(words, available)
    return resolution.classes, resolution.group, resolution.notes


def _find_direction(q: str) -> str | None:
    for phrase in sorted(DIRECTION_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return DIRECTION_SYNONYMS[phrase]
    return None


def _find_colour(q: str) -> str | None:
    for phrase in sorted(COLOUR_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return COLOUR_SYNONYMS[phrase]
    return None


def _find_time(
    q: str,
    *,
    duration: float | None,
    start_time: str | None,
) -> tuple[float, float] | None:
    match = re.search(rf"\bafter\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b", q)
    if match:
        start = float(match.group(1))
        return (start, float(duration) if duration else start)

    match = re.search(rf"\bbefore\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b", q)
    if match:
        return (0.0, float(match.group(1)))

    match = re.search(
        rf"\bbetween\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)?\s+and\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b",
        q,
    )
    if match:
        return (float(match.group(1)), float(match.group(2)))

    for phrase in ("first half", "second half", "beginning", "middle", "end"):
        if phrase in q:
            return resolve_time_word(phrase, duration_s=duration, start_time_iso=start_time)

    for phrase in ("morning", "afternoon", "evening", "night", "rush hour", "midday", "noon"):
        if re.search(rf"\b{re.escape(phrase)}\b", q):
            return resolve_time_word(phrase, duration_s=duration, start_time_iso=start_time)

    return None


def _find_count_bounds(q: str) -> tuple[int | None, int | None]:
    match = re.search(rf"\bmore than\s+{_NUMBER}\b", q)
    if match:
        return (int(float(match.group(1))) + 1, None)
    match = re.search(rf"\bat least\s+{_NUMBER}\b", q)
    if match:
        return (int(float(match.group(1))), None)
    match = re.search(rf"\b(fewer|less) than\s+{_NUMBER}\b", q)
    if match:
        return (None, int(float(match.group(2))) - 1)
    match = re.search(rf"\bat most\s+{_NUMBER}\b", q)
    if match:
        return (None, int(float(match.group(1))))
    return (None, None)


def extract_slots(question: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Extract classes, time, numbers and relations without choosing an intent.

    The LLM classifier chooses the label; these slots are filled deterministically
    so the model cannot invent a class or a time window.
    """
    q = " " + re.sub(r"\s+", " ", question.strip().lower()) + " "
    available = {c.lower() for c in (meta.get("classes") or [])}
    duration = meta.get("duration_s")
    start_time = meta.get("start_time")

    if _SQL_WORDS.search(q):
        raise UnsupportedQuery(
            "I only answer questions about the recorded detections. I cannot run SQL."
        )
    for term, reason in UNSUPPORTED_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", q):
            raise UnsupportedQuery(reason)

    classes, group, notes = _find_classes(q, available)
    direction = _find_direction(q)
    colour = _find_colour(q)
    time_window = _find_time(q, duration=duration, start_time=start_time)
    min_count, max_count = _find_count_bounds(q)

    co_presence = None
    co = re.search(
        rf"\bwithin\s+{_NUMBER}\s*(?:s|sec|secs|second|seconds)\b",
        q,
    )
    if co and len(classes) >= 2:
        co_presence = {"classes": classes, "within_s": float(co.group(1))}

    compare = None
    if q.strip().startswith("compare") or " compare " in q:
        compare = _split_compare(q)

    return {
        "classes": classes,
        "group": group,
        "direction": direction,
        "colour": colour,
        "time_window": time_window,
        "min_count": min_count,
        "max_count": max_count,
        "co_presence": co_presence,
        "compare": compare,
        "crossed": bool(re.search(r"\bcross(ed|ing)?\b", q)),
        "unique": bool(re.search(r"\b(unique|distinct)\b", q)),
        "peak": bool(re.search(r"\bpeak\b|\bmaximum\b|\bmost\b.*?\bat (once|a time)\b", q)),
        "instances": bool(re.search(r"\b(detections?|instances?|boxes|per frame)\b", q)),
        "list_frames": bool(re.search(r"\b(show|list|which|find)\b.*\b(frames?|images?)\b", q)),
        "notes": notes,
        "duration_s": duration,
        "start_time": start_time,
    }


def _split_compare(q: str) -> tuple[str, str]:
    text = re.sub(r"^.*?\bcompare\b", "", q).strip(" ?.")
    text = re.sub(r"\bwith\b", "and", text)
    parts = re.split(r"\band\b", text, maxsplit=1)
    if len(parts) != 2:
        raise UnsupportedQuery(
            "A comparison needs two time phrases, for example "
            "'compare the first half and the second half'."
        )
    return parts[0].strip(), parts[1].strip()
