"""Closed query-plan schema for natural-language investigation.

The language model never emits SQL. It emits a :class:`QueryPlan`, which is
validated here and compiled to parameterised SQL elsewhere. Keeping the schema
closed is what makes the feature auditable and safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

INTENTS = {
    "count",
    "list_frames",
    "list_tracks",
    "aggregate",
    "compare",
    "explain",
}

METRICS = {
    "detections",
    "unique_tracks",
    "frames",
    "peak",
    "mean_dwell",
    "mean_speed",
    "mean_score",
}

GROUPS = {"vehicle", "two_wheeler", "vulnerable", "person"}

DIRECTIONS = {"left", "right", "up", "down", "stationary"}

GROUP_BY_FIELDS = {"minute", "class", "direction", "colour", "half"}

ORDER_FIELDS = {"count", "speed_px_s", "dwell_s", "timestamp_s", "score"}

OUTPUT_FIELDS = {"count", "rows", "frames", "contact_sheet"}

COLOURS = {
    "white",
    "black",
    "gray",
    "grey",
    "silver",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "brown",
}


class PlanError(ValueError):
    """Raised when a query plan is malformed or unsupported."""


@dataclass
class QueryPlan:
    """A validated, closed-schema query over one evidence index."""

    version: int = SCHEMA_VERSION
    intent: str = "count"
    metric: str = "detections"

    classes: list[str] = field(default_factory=list)
    group: str | None = None

    time_start_s: float | None = None
    time_end_s: float | None = None
    frame_start: int | None = None
    frame_end: int | None = None

    confidence_min: float | None = None
    zone: str | None = None
    direction: str | None = None
    speed_px_s_min: float | None = None
    dwell_s_min: float | None = None
    colour: list[str] = field(default_factory=list)
    track_ids: list[int] = field(default_factory=list)
    ood: bool | None = None
    crossed: bool | None = None

    min_count: int | None = None
    max_count: int | None = None

    co_presence_classes: list[str] = field(default_factory=list)
    co_presence_within_s: float | None = None

    group_by: list[str] = field(default_factory=list)
    order_by: str | None = None
    order_dir: str = "desc"
    limit: int = 12
    output: list[str] = field(default_factory=lambda: ["count"])

    def __post_init__(self) -> None:
        # Accept scalars and None in convenience constructors, and normalise case.
        self.classes = _as_str_list(self.classes)
        self.colour = _as_str_list(self.colour)
        self.group_by = _as_str_list(self.group_by)
        self.output = _as_str_list(self.output) or ["count"]
        self.track_ids = _as_int_list(self.track_ids)
        self.co_presence_classes = _as_str_list(self.co_presence_classes)

        self.classes = [str(c).lower() for c in self.classes]
        self.colour = [str(c).lower() for c in self.colour]
        self.co_presence_classes = [str(c).lower() for c in self.co_presence_classes]
        if self.direction is not None:
            self.direction = str(self.direction).lower()
        if self.group is not None:
            self.group = str(self.group).lower()
        self.validate()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryPlan":
        if not isinstance(data, dict):
            raise PlanError("plan must be a JSON object")

        if data.get("unsupported"):
            raise PlanError(str(data["unsupported"]))

        filters = data.get("filters") or {}
        relations = data.get("relations") or {}
        entities = data.get("entities") or {}

        if not isinstance(filters, dict):
            raise PlanError("filters must be an object")
        if not isinstance(relations, dict):
            raise PlanError("relations must be an object")
        if not isinstance(entities, dict):
            raise PlanError("entities must be an object")

        time_filter = filters.get("time") or {}
        frame_filter = filters.get("frame") or {}
        track_filter = filters.get("track") or {}
        co = relations.get("co_presence") or {}

        plan = cls(
            version=int(data.get("version", SCHEMA_VERSION)),
            intent=str(data.get("intent", "count")),
            metric=str(data.get("metric", "detections")),
            classes=_as_str_list(entities.get("classes")),
            group=_none_or_str(entities.get("group")),
            time_start_s=_none_or_float(time_filter.get("start_s")),
            time_end_s=_none_or_float(time_filter.get("end_s")),
            frame_start=_none_or_int(frame_filter.get("start")),
            frame_end=_none_or_int(frame_filter.get("end")),
            confidence_min=_none_or_float(filters.get("confidence_min")),
            zone=_none_or_str(filters.get("zone")),
            direction=_none_or_str(track_filter.get("direction")),
            speed_px_s_min=_none_or_float(track_filter.get("speed_px_s_min")),
            dwell_s_min=_none_or_float(track_filter.get("dwell_s_min")),
            colour=[c.lower() for c in _as_str_list(track_filter.get("colour"))],
            track_ids=_as_int_list(track_filter.get("track_ids")),
            ood=_none_or_bool(filters.get("ood")),
            crossed=_none_or_bool(track_filter.get("crossed")),
            min_count=_none_or_int(filters.get("min_count")),
            max_count=_none_or_int(filters.get("max_count")),
            co_presence_classes=_as_str_list(co.get("classes")),
            co_presence_within_s=_none_or_float(co.get("within_s")),
            group_by=_as_str_list(data.get("group_by")),
            order_by=_none_or_str(data.get("order_by")),
            order_dir=str(data.get("order_dir", "desc")).lower(),
            limit=int(data.get("limit", 12)),
            output=_as_str_list(data.get("output")) or ["count"],
        )
        if isinstance(data.get("order_by"), dict):
            order = data["order_by"]
            plan.order_by = _none_or_str(order.get("field"))
            plan.order_dir = str(order.get("dir", "desc")).lower()
        plan.validate()
        return plan

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intent": self.intent,
            "metric": self.metric,
            "entities": {"classes": self.classes, "group": self.group},
            "filters": {
                "time": {"start_s": self.time_start_s, "end_s": self.time_end_s},
                "frame": {"start": self.frame_start, "end": self.frame_end},
                "confidence_min": self.confidence_min,
                "zone": self.zone,
                "min_count": self.min_count,
                "max_count": self.max_count,
                "ood": self.ood,
                "track": {
                    "direction": self.direction,
                    "speed_px_s_min": self.speed_px_s_min,
                    "dwell_s_min": self.dwell_s_min,
                    "colour": self.colour,
                    "track_ids": self.track_ids,
                    "crossed": self.crossed,
                },
            },
            "relations": {
                "co_presence": {
                    "classes": self.co_presence_classes,
                    "within_s": self.co_presence_within_s,
                }
            },
            "group_by": self.group_by,
            "order_by": self.order_by,
            "order_dir": self.order_dir,
            "limit": self.limit,
            "output": self.output,
        }

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise PlanError(f"unsupported schema version: {self.version}")

        if self.intent not in INTENTS:
            raise PlanError(f"unknown intent: {self.intent!r}")
        if self.metric not in METRICS:
            raise PlanError(f"unknown metric: {self.metric!r}")
        if self.group is not None and self.group not in GROUPS:
            raise PlanError(f"unknown group: {self.group!r}")
        if self.direction is not None and self.direction not in DIRECTIONS:
            raise PlanError(f"unknown direction: {self.direction!r}")
        if self.order_by is not None and self.order_by not in ORDER_FIELDS:
            raise PlanError(f"unknown order field: {self.order_by!r}")
        if self.order_dir not in {"asc", "desc"}:
            raise PlanError("order_dir must be 'asc' or 'desc'")

        bad_groups = [g for g in self.group_by if g not in GROUP_BY_FIELDS]
        if bad_groups:
            raise PlanError(f"unknown group_by field(s): {', '.join(bad_groups)}")
        bad_outputs = [o for o in self.output if o not in OUTPUT_FIELDS]
        if bad_outputs:
            raise PlanError(f"unknown output field(s): {', '.join(bad_outputs)}")

        if not 1 <= self.limit <= 50:
            raise PlanError("limit must be between 1 and 50")

        if self.confidence_min is not None and not 0.0 <= self.confidence_min <= 1.0:
            raise PlanError("confidence_min must be between 0 and 1")
        if self.speed_px_s_min is not None and self.speed_px_s_min < 0:
            raise PlanError("speed_px_s_min must be >= 0")
        if self.dwell_s_min is not None and self.dwell_s_min < 0:
            raise PlanError("dwell_s_min must be >= 0")
        if self.min_count is not None and self.min_count < 0:
            raise PlanError("min_count must be >= 0")
        if self.max_count is not None and self.max_count < 0:
            raise PlanError("max_count must be >= 0")
        if (
            self.min_count is not None
            and self.max_count is not None
            and self.min_count > self.max_count
        ):
            raise PlanError("min_count must not exceed max_count")

        if (
            self.time_start_s is not None
            and self.time_end_s is not None
            and self.time_start_s >= self.time_end_s
        ):
            raise PlanError("time start must be before time end")
        if (
            self.frame_start is not None
            and self.frame_end is not None
            and self.frame_start >= self.frame_end
        ):
            raise PlanError("frame start must be before frame end")

        if self.co_presence_classes and self.co_presence_within_s is None:
            self.co_presence_within_s = 0.0
        if self.co_presence_within_s is not None and not 0.0 <= self.co_presence_within_s <= 60.0:
            raise PlanError("co_presence.within_s must be between 0 and 60")

        for colour in self.colour:
            if colour not in COLOURS:
                raise PlanError(f"unknown colour: {colour!r}")


# ---------------------------------------------------------------------- #
# Coercion helpers
# ---------------------------------------------------------------------- #
def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    raise PlanError("expected a string or list of strings")


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise PlanError("expected an integer or list of integers")


def _none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _none_or_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _none_or_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _none_or_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
