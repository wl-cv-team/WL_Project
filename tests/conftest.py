from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

# BGR colors; one dominant channel per frame (0 = blue, 1 = green, 2 = red)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
DOMINANT = [0, 1, 2, 0, 1]


def write_video(path: Path, colors=COLORS, size=(64, 48), fps=10.0) -> Path:
    """Synthetic mp4v video: one solid color per frame, size is (width, height)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened()
    for color in colors:
        writer.write(np.full((size[1], size[0], 3), color, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def make_video() -> Callable[..., Path]:
    return write_video
