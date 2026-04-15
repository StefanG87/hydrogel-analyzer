# logic/scale_bar.py

"""
logic/scale_bar.py

Provides automatic scale bar detection and conversion utilities.

This module is responsible for:

- detecting a horizontal scale bar in microscopy images
- estimating its pixel length
- converting known physical length to µm/px calibration
- returning an optional debug image for visual verification

This module must remain independent of GUI logic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from skimage import color, filters, img_as_ubyte, measure, morphology
from skimage.draw import rectangle_perimeter


def calculate_um_per_px(scale_bar_px: float, known_length_mm: float) -> float:
    """
    Convert a detected scale bar length into µm per pixel.

    Args:
        scale_bar_px:
            Detected scale bar length in pixels.

        known_length_mm:
            Known real-world scale bar length in millimeters.

    Returns:
        float:
            Micrometers per pixel.
    """
    if scale_bar_px <= 0:
        raise ValueError("Scale bar pixel length must be greater than zero.")

    if known_length_mm <= 0:
        raise ValueError("Known scale bar length must be greater than zero.")

    known_length_um = float(known_length_mm) * 1000.0
    return known_length_um / float(scale_bar_px)


def find_scale_bar_px(image: np.ndarray):
    """
    Detect scale bar length in pixels using longest horizontal bright line detection.
    Handles rounded scale bar ends robustly.
    """

    if image is None:
        return 0.0, False, None

    debug_image = image.copy()

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32)
        if gray.max() > 1.0:
            gray = gray / 255.0

    height, width = gray.shape

    # ROI bottom-right region
    roi_x_min = int(width * 0.50)
    roi_x_max = int(width * 0.95)
    roi_y_min = int(height * 0.92)
    roi_y_max = height

    roi = gray[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

    threshold = filters.threshold_otsu(roi)
    binary = roi > threshold

    binary = morphology.remove_small_objects(binary, min_size=30)

    max_length = 0
    best_row = None
    best_start = None
    best_end = None

    # horizontal line scan
    for row_idx in range(binary.shape[0]):

        row = binary[row_idx]

        start = None

        for col_idx, value in enumerate(row):

            if value and start is None:
                start = col_idx

            elif not value and start is not None:

                length = col_idx - start

                if length > max_length:
                    max_length = length
                    best_row = row_idx
                    best_start = start
                    best_end = col_idx

                start = None

        if start is not None:

            length = len(row) - start

            if length > max_length:
                max_length = length
                best_row = row_idx
                best_start = start
                best_end = len(row)

    if best_row is None:
        return 0.0, False, debug_image

    # transform back to image coords
    y = best_row + roi_y_min
    x1 = best_start + roi_x_min
    x2 = best_end + roi_x_min

    # debug overlay (yellow line)
    if debug_image.ndim == 2:
        debug_image = np.stack([debug_image]*3, axis=-1)

    debug_image = img_as_ubyte(debug_image)

    debug_image[y, x1:x2] = [255, 255, 0]

    scale_bar_px = float(x2 - x1)

    print("Detected scale bar px:", scale_bar_px)

    return scale_bar_px, True, debug_image
  
