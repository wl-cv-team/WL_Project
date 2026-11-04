import numpy as np
import pytest

from wl_snatch.contact_sheet import draw_yolo_boxes, label_image, tile_images


def test_draw_yolo_boxes_changes_pixels_on_a_copy():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out = draw_yolo_boxes(img, [(0, 0.5, 0.5, 0.5, 0.5)])
    assert img.sum() == 0
    assert out[25, 100].tolist() == [0, 255, 0]  # top edge: y = 0.25 * 100, x = center


def test_tile_images_shape_and_partial_last_row():
    imgs = [np.full((50, 50, 3), 255, dtype=np.uint8)] * 5
    sheet = tile_images(imgs, columns=2, cell=(20, 10))
    assert sheet.shape == (30, 40, 3)
    assert sheet[25, 30].sum() == 0  # empty cell of the last row
    assert sheet[0, 0].tolist() == [255, 255, 255]


def test_tile_images_rejects_bad_columns():
    with pytest.raises(ValueError):
        tile_images([], columns=0)


def test_label_image_returns_copy():
    img = np.zeros((40, 100, 3), dtype=np.uint8)
    out = label_image(img, "abc")
    assert img.sum() == 0 and out.sum() > 0
