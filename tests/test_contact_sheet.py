import numpy as np
import pytest

from wl_snatch.contact_sheet import letterbox, render_cell, tile_images


def test_letterbox_keeps_aspect_ratio_and_pads():
    img = np.full((200, 100, 3), 255, dtype=np.uint8)  # height 200, width 100
    cell, scale, ox, oy = letterbox(img, (100, 100))
    assert cell.shape == (100, 100, 3)
    assert (scale, ox, oy) == (0.5, 25, 0)
    assert cell[50, 50].tolist() == [255, 255, 255]
    assert cell[50, 10].tolist() == [0, 0, 0]  # left pad
    assert cell[50, 90].tolist() == [0, 0, 0]  # right pad
    assert cell[:, 25:75].min() == 255  # image content is not stretched


def test_render_cell_box_lands_on_expected_pixels():
    img = np.full((200, 100, 3), 255, dtype=np.uint8)
    out = render_cell(img, [(0, 0.5, 0.5, 0.4, 0.5)], (100, 100), "")
    # content spans x 25..75; box x: 25 + 0.3*50 = 40 .. 60, y: 25 .. 75
    assert out[50, 40].tolist() == [0, 255, 0]
    assert out[50, 60].tolist() == [0, 255, 0]
    assert out[25, 50].tolist() == [0, 255, 0]
    assert out[75, 50].tolist() == [0, 255, 0]
    assert out[50, 50].tolist() == [255, 255, 255]  # inside the box, not filled
    assert out[50, 10].tolist() == [0, 0, 0]  # padding untouched
    assert img.min() == 255  # input not modified


def test_render_cell_line_width_scales_with_cell():
    img = np.full((1280, 720, 3), 255, dtype=np.uint8)
    out = render_cell(img, [(0, 0.5, 0.5, 0.4, 0.2)], (270, 480), "")
    row = out[240]
    green = np.where((row == [0, 255, 0]).all(axis=1))[0]
    assert len(green) >= 3  # two edges, each 2 px wide


def test_render_cell_label_fits_in_cell_with_long_stem():
    img = np.zeros((1280, 720, 3), dtype=np.uint8)
    stem = "c02_sn_f_01_frame_0109"
    out = render_cell(img, [], (270, 480), stem)
    assert out.shape == (480, 270, 3)
    assert out[:30].sum() > 0  # text drawn at the top
    assert out[:, 262:].sum() == 0  # nothing at the right edge: text not cut off


def test_tile_images_shape_and_partial_last_row():
    imgs = [np.full((10, 20, 3), 255, dtype=np.uint8)] * 5
    sheet = tile_images(imgs, columns=2, cell=(20, 10))
    assert sheet.shape == (30, 40, 3)
    assert sheet[25, 30].sum() == 0  # empty cell of the last row
    assert sheet[0, 0].tolist() == [255, 255, 255]


def test_tile_images_rejects_bad_columns_and_wrong_cell_size():
    with pytest.raises(ValueError):
        tile_images([], columns=0)
    with pytest.raises(ValueError):
        tile_images([np.zeros((5, 5, 3), dtype=np.uint8)], columns=1, cell=(20, 10))
