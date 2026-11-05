from arten.postprocess import Detection
from arten.tracker import IoUTracker


def det(name, box, score=0.9, class_id=0):
    return Detection(class_id=class_id, class_name=name, score=score, box=tuple(box))


def test_same_box_is_one_unique_track():
    tracker = IoUTracker(min_hits=1)
    tracker.update([det("person", (10, 10, 50, 100))], 0)
    tracker.update([det("person", (10, 10, 50, 100))], 1)
    assert tracker.unique_counts == {"person": 1}


def test_moving_box_stays_one_track():
    tracker = IoUTracker(min_hits=1)
    tracker.update([det("person", (10, 10, 50, 100))], 0)
    tracker.update([det("person", (14, 12, 54, 102))], 1)
    tracker.update([det("person", (18, 14, 58, 104))], 2)
    assert tracker.unique_counts == {"person": 1}


def test_two_people_are_two_tracks():
    tracker = IoUTracker(min_hits=1)
    tracker.update(
        [det("person", (10, 10, 50, 100)), det("person", (200, 10, 240, 100))], 0
    )
    tracker.update(
        [det("person", (11, 10, 51, 100)), det("person", (201, 10, 241, 100))], 1
    )
    assert tracker.unique_counts == {"person": 2}


def test_two_hits_required_by_default():
    tracker = IoUTracker(min_hits=2)
    tracker.update([det("person", (10, 10, 50, 100))], 0)
    assert tracker.unique_counts == {}
    tracker.update([det("person", (10, 10, 50, 100))], 1)
    assert tracker.unique_counts == {"person": 1}


def test_track_id_is_attached():
    tracker = IoUTracker(min_hits=1)
    result = tracker.update([det("person", (10, 10, 50, 100))], 0)
    assert result[0].track_id == 1


def test_different_class_does_not_merge():
    tracker = IoUTracker(min_hits=1)
    tracker.update([det("person", (10, 10, 50, 100), class_id=0)], 0)
    tracker.update([det("car", (10, 10, 50, 100), class_id=2)], 1)
    assert tracker.unique_counts == {"person": 1, "car": 1}
