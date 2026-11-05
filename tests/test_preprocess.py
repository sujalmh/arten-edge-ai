import numpy as np

from arten.preprocess import letterbox, unletterbox_boxes


def test_letterbox_shape_range_and_padding():
    image = np.full((200, 400, 3), 127, dtype=np.uint8)  # landscape
    chw, meta = letterbox(image, size=(640, 640), pad_value=114)

    assert chw.shape == (3, 640, 640)
    assert chw.dtype == np.float32
    assert 0.0 <= chw.min() <= chw.max() <= 1.0
    assert meta.scale == 1.6
    assert meta.pad_x == 0
    assert meta.pad_y == 160
    assert meta.orig_w == 400
    assert meta.orig_h == 200

    # Top padding is the 114-grey value (in RGB, all channels equal).
    assert np.allclose(chw[:, 0, 0], 114 / 255.0, atol=1e-6)
    # The content area is the original grey.
    assert np.allclose(chw[:, 320, 320], 127 / 255.0, atol=1e-6)


def test_letterbox_portrait_pads_horizontally():
    image = np.zeros((400, 200, 3), dtype=np.uint8)
    chw, meta = letterbox(image, size=(640, 640), pad_value=114)
    assert meta.scale == 1.6
    assert meta.pad_x == 160
    assert meta.pad_y == 0
    assert np.allclose(chw[:, 320, 0], 114 / 255.0, atol=1e-6)


def test_unletterbox_boxes_round_trip():
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    _, meta = letterbox(image, size=(640, 640))
    # Full content rectangle in letterboxed coordinates.
    boxes = np.array([[0.0, 160.0, 640.0, 480.0]], dtype=np.float32)
    mapped = unletterbox_boxes(boxes, meta)
    assert np.allclose(mapped, [[0.0, 0.0, 400.0, 200.0]], atol=1e-3)


def test_unletterbox_clips_to_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    _, meta = letterbox(image, size=(640, 640))
    boxes = np.array([[-50.0, -50.0, 10000.0, 10000.0]], dtype=np.float32)
    mapped = unletterbox_boxes(boxes, meta)
    assert mapped[0, 0] >= 0 and mapped[0, 1] >= 0
    assert mapped[0, 2] <= 100 and mapped[0, 3] <= 100
