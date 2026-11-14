import numpy as np

from arten.postprocess import Detection
from arten.visualize import _class_color, annotate_frame


def test_detection_box_is_drawn_over_counts_panel():
    # 854x480 puts the counts panel at x=12..262, y=12..181.
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    det = Detection(
        class_id=2,
        class_name="car",
        score=0.9,
        box=(60.0, 60.0, 160.0, 140.0),
    )
    out = annotate_frame(frame, [det], {"car": 1}, frame_idx=0, total_frames=1)

    expected = _class_color("car")
    # The top edge of the box lies inside the panel. It must keep the class
    # colour instead of being covered by the HUD.
    edge = out[60, 60:161]
    matches = np.all(edge == np.array(expected, dtype=np.uint8), axis=1)
    assert matches.any(), "box outline was painted under the counts panel"

    # The panel itself is still present in an empty corner of the HUD.
    assert out[20, 250].sum() > 0, "counts panel background disappeared"
