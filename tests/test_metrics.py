from namma_drishti.metrics import (
    frames_with_both,
    person_seconds,
    summarise_latencies,
)


def test_latency_summary():
    stats = summarise_latencies([10, 20, 30, 40])
    assert stats["count"] == 4
    assert stats["mean_ms"] == 25.0
    assert stats["p50_ms"] == 25.0
    assert stats["fps_from_mean"] == 40.0


def test_latency_zero_mean_is_safe():
    stats = summarise_latencies([0, 0])
    assert stats["fps_from_mean"] == 0.0


def test_person_seconds_accounts_for_stride():
    # 4 processed frames, 10 source FPS, stride 2 => 0.2 s per frame.
    assert person_seconds([2, 2, 2, 2], source_fps=10, stride=2) == 1.6


def test_frames_with_both():
    result = frames_with_both([1, 1, 0, 2], [0, 2, 3, 1])
    assert result["frames"] == 2
    assert result["share"] == 0.5
