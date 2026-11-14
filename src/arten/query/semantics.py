"""Semantic layer for natural-language queries.

Natural language is ambiguous; this module makes the resolution explicit and
auditable. Every mapping here is printed by ``arten ask --explain`` so the user
can see exactly how "people", "morning" or "crossed" was interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import PlanError

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck", "bicycle")
TWO_WHEELER_CLASSES = ("motorcycle", "bicycle")
VULNERABLE_CLASSES = ("person", "bicycle", "motorcycle")

CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "vehicle": VEHICLE_CLASSES,
    "two_wheeler": TWO_WHEELER_CLASSES,
    "vulnerable": VULNERABLE_CLASSES,
    "person": ("person",),
}

CLASS_SYNONYMS: dict[str, str] = {
    "people": "person",
    "person": "person",
    "persons": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "walker": "person",
    "walkers": "person",
    "human": "person",
    "humans": "person",
    "car": "car",
    "cars": "car",
    "automobile": "car",
    "automobiles": "car",
    "sedan": "car",
    "sedans": "car",
    "suv": "car",
    "suvs": "car",
    "taxi": "car",
    "taxis": "car",
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "bike": "bicycle",
    "bikes": "bicycle",
    "cyclist": "bicycle",
    "cyclists": "bicycle",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "motorbike": "motorcycle",
    "motorbikes": "motorcycle",
    "scooter": "motorcycle",
    "scooters": "motorcycle",
    "two-wheeler": "motorcycle",
    "two-wheelers": "motorcycle",
    "bus": "bus",
    "buses": "bus",
    "coach": "bus",
    "coaches": "bus",
    "truck": "truck",
    "trucks": "truck",
    "lorry": "truck",
    "lorries": "truck",
    "van": "truck",
    "vans": "truck",
    "hgv": "truck",
    "hgvs": "truck",
}

GROUP_SYNONYMS: dict[str, str] = {
    "vehicle": "vehicle",
    "vehicles": "vehicle",
    "traffic": "vehicle",
    "motor traffic": "vehicle",
    "two-wheeler": "two_wheeler",
    "two-wheelers": "two_wheeler",
    "two wheeler": "two_wheeler",
    "two wheelers": "two_wheeler",
    "bike traffic": "two_wheeler",
    "vulnerable": "vulnerable",
    "vulnerable road user": "vulnerable",
    "vulnerable road users": "vulnerable",
    "vru": "vulnerable",
    "vrus": "vulnerable",
    "person": "person",
    "people": "person",
    "pedestrians": "person",
}

DIRECTION_SYNONYMS: dict[str, str] = {
    "left": "left",
    "leftward": "left",
    "right": "right",
    "rightward": "right",
    "up": "up",
    "upward": "up",
    "down": "down",
    "downward": "down",
    "stationary": "stationary",
    "still": "stationary",
    "stopped": "stationary",
}

COLOUR_SYNONYMS: dict[str, str] = {
    "white": "white",
    "black": "black",
    "gray": "gray",
    "grey": "gray",
    "silver": "silver",
    "red": "red",
    "blue": "blue",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
}

# Terms that are not in a COCO-trained index. The system should say so instead
# of returning a misleading zero.
UNSUPPORTED_TERMS: dict[str, str] = {
    "auto": "Auto-rickshaw is not a COCO class, so this index cannot count it. "
    "A YOLOE text prompt or an IDD fine-tune would be needed.",
    "auto-rickshaw": "Auto-rickshaw is not a COCO class, so this index cannot count it. "
    "A YOLOE text prompt or an IDD fine-tune would be needed.",
    "autorickshaw": "Auto-rickshaw is not a COCO class, so this index cannot count it. "
    "A YOLOE text prompt or an IDD fine-tune would be needed.",
    "tuk-tuk": "Auto-rickshaw is not a COCO class, so this index cannot count it.",
    "hand cart": "Hand carts are not a COCO class; this index cannot count them.",
    "handcart": "Hand carts are not a COCO class; this index cannot count them.",
    "cattle": "Cattle are not a COCO class; this index cannot count them.",
    "cow": "Cattle are not a COCO class; this index cannot count them.",
    "dog": "Dogs are not in this restricted class set; this index cannot count them.",
    "near miss": "Near-miss counts are not implemented. The system can report a "
    "conflict-exposure index, not a validated near-miss event count.",
    "near-miss": "Near-miss counts are not implemented. The system can report a "
    "conflict-exposure index, not a validated near-miss event count.",
    "red bag": "Bag attributes are not in the index; a VLM crop description or an "
    "open-vocabulary detector would be needed.",
    "carrying": "Carried-object attributes are not in the index; a VLM crop "
    "description or an open-vocabulary detector would be needed.",
    "running": "Gait is not in the index; a VLM crop description would be needed.",
    "wheelchair": "Wheelchairs are not a COCO class; an open-vocabulary detector would be needed.",
    "pram": "Prams are not a COCO class; an open-vocabulary detector would be needed.",
    "stroller": "Strollers are not a COCO class; an open-vocabulary detector would be needed.",
}

TIME_WORDS: dict[str, tuple[int, int]] = {
    # 24-hour clock, local time
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 6),
    "midday": (11, 14),
    "noon": (11, 14),
    "rush hour": (8, 10),
}

RELATIVE_WORDS = {
    "first half": (0.0, 0.5),
    "second half": (0.5, 1.0),
    "beginning": (0.0, 0.33),
    "middle": (0.33, 0.66),
    "end": (0.66, 1.0),
}

METRIC_WORDS = {
    "unique": "unique_tracks",
    "distinct": "unique_tracks",
    "peak": "peak",
    "maximum": "peak",
    "highest": "peak",
    "average": "mean",
    "mean": "mean",
    "typical": "mean",
}


class UnsupportedQuery(PlanError):
    """Raised when the question needs data the index does not contain."""


@dataclass
class Resolution:
    classes: list[str]
    group: str | None
    notes: list[str]


def expand_class_words(words: list[str], available: set[str]) -> Resolution:
    """Resolve class words and group words to concrete class names."""
    classes: list[str] = []
    groups: list[str] = []
    notes: list[str] = []
    available = {c.lower() for c in available}

    for raw in words:
        word = raw.strip().lower()
        if not word:
            continue
        if word in UNSUPPORTED_TERMS:
            raise UnsupportedQuery(UNSUPPORTED_TERMS[word])
        if word in GROUP_SYNONYMS:
            group = GROUP_SYNONYMS[word]
            groups.append(group)
            for cls in CLASS_GROUPS[group]:
                if cls in available and cls not in classes:
                    classes.append(cls)
            notes.append(f"'{word}' -> group '{group}' -> {', '.join(CLASS_GROUPS[group])}")
            continue
        cls = CLASS_SYNONYMS.get(word, word)
        if cls in available:
            if cls not in classes:
                classes.append(cls)
            if cls != word:
                notes.append(f"'{word}' -> '{cls}'")
        else:
            notes.append(f"'{word}' is not in this index (available: {', '.join(sorted(available))})")

    if not classes and groups:
        for group in groups:
            for cls in CLASS_GROUPS[group]:
                if cls in available and cls not in classes:
                    classes.append(cls)

    group = groups[0] if len(groups) == 1 else None
    return Resolution(classes=classes, group=group, notes=notes)


def resolve_direction(word: str) -> str | None:
    return DIRECTION_SYNONYMS.get(word.strip().lower())


def resolve_colour(word: str) -> str | None:
    return COLOUR_SYNONYMS.get(word.strip().lower())


def resolve_time_word(
    word: str,
    *,
    duration_s: float | None,
    start_time_iso: str | None,
) -> tuple[float, float]:
    """Resolve a time phrase to offsets from the start of the run.

    Time-of-day words require the index to know the wall-clock start time.
    Without it, the system refuses rather than guessing.
    """
    key = word.strip().lower()

    if key in RELATIVE_WORDS:
        if duration_s is None:
            raise UnsupportedQuery("This index has no duration, so relative time phrases are unavailable.")
        lo, hi = RELATIVE_WORDS[key]
        return (round(lo * duration_s, 3), round(hi * duration_s, 3))

    if key in TIME_WORDS:
        if not start_time_iso:
            raise UnsupportedQuery(
                "This index has no wall-clock start time, so 'morning'/'afternoon' "
                "cannot be resolved. Re-index with --start-time, or use a relative "
                "phrase such as 'first half'."
            )
        from datetime import datetime, timedelta

        start = datetime.fromisoformat(start_time_iso)
        begin_hour, end_hour = TIME_WORDS[key]
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        begin = day + timedelta(hours=begin_hour)
        end = day + timedelta(hours=end_hour)
        if end <= begin:
            end += timedelta(days=1)
        begin_s = max(0.0, (begin - start).total_seconds())
        end_s = max(0.0, (end - start).total_seconds())
        if end_s <= begin_s:
            raise UnsupportedQuery(f"'{word}' does not overlap this run.")
        return (round(begin_s, 3), round(end_s, 3))

    raise UnsupportedQuery(f"Unknown time phrase: {word!r}")


def metric_from_words(words: set[str]) -> str:
    if "unique" in words or "distinct" in words:
        return "unique_tracks"
    if "peak" in words or "maximum" in words or "highest" in words:
        return "peak"
    if "average" in words or "mean" in words or "typical" in words:
        return "mean"
    return "detections"


def describe_classes(classes: list[str], group: str | None) -> str:
    if group:
        return f"{group} ({', '.join(classes)})"
    return ", ".join(classes) if classes else "all classes"
