

from typing import Tuple
import cv2
import numpy as np
import torch
import torch.backends
import torch.backends.mps
from numpy.typing import NDArray
from deep_sort_reid.types.coords import CoordinatesXYXY


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    return device


def draw_bounding_box(frame: NDArray, coords: CoordinatesXYXY, color: Tuple, track_id: int):
    x1, y1, x2, y2 = int(coords.start_x), int(
        coords.start_y), int(coords.end_x), int(coords.end_y)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    label_height = 40
    cv2.rectangle(frame, (x1, y1 - label_height),
                  (x1 + 120, y1), color, -1)  # Filled rectangle

    text_color = (255, 255, 255)  # White text for good contrast
    cv2.putText(frame, f'ID: {track_id}', (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
    shadow_offset = 7
    shadow = np.zeros_like(frame)
    cv2.rectangle(shadow, (x1 + shadow_offset, y1 + shadow_offset),
                  (x2 + shadow_offset, y2 + shadow_offset), (50, 50, 50), 4)
