# tools/image_preprocessing.py

"""
tools/image_preprocessing.py

Provides the preprocessing pipeline for contour analysis.

This module is responsible for:

- grayscale conversion
- linear contrast and brightness adjustment
- optional CLAHE enhancement
- optional Gaussian blur
- optional median blur
- optional inversion
- optional binarization

The preprocessing pipeline returns a normalized float32 image suitable
for contour detection.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert the input image to grayscale float format in range [0, 1].

    Args:
        image: Input image.

    Returns:
        np.ndarray: Grayscale float image.
    """
    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
        if gray.max() > 1.0:
            gray /= 255.0
        return np.clip(gray, 0.0, 1.0)

    image_float = image.astype(np.float32, copy=False)
    if image.shape[2] == 4:
        gray = cv2.cvtColor(image_float, cv2.COLOR_RGBA2GRAY)
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
    else:
        gray = np.mean(image_float, axis=2)

    if gray.max() > 1.0:
        gray /= 255.0

    return np.clip(gray, 0.0, 1.0).astype(np.float32)


def apply_linear_adjustment(
    image: np.ndarray,
    contrast_factor: float = 100.0,
    brightness_offset: float = 0.0,
) -> np.ndarray:
    """
    Apply linear contrast and brightness adjustment.

    Args:
        image: Grayscale float image in range [0, 1].
        contrast_factor: Percentage scale, 100 means unchanged.
        brightness_offset: Additive offset in 8-bit intensity space.

    Returns:
        np.ndarray: Adjusted image.
    """
    contrast = float(contrast_factor) / 100.0
    brightness = float(brightness_offset) / 255.0
    adjusted = image.astype(np.float32, copy=False) * contrast + brightness
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE using OpenCV.

    Args:
        image: Grayscale float image in range [0, 1].
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile size.

    Returns:
        np.ndarray: CLAHE-enhanced image.
    """
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced = clahe.apply(image_uint8).astype(np.float32) / 255.0
    return np.clip(enhanced, 0.0, 1.0)


def apply_median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median blur using OpenCV.

    Args:
        image: Grayscale float image in range [0, 1].
        kernel_size: Median filter kernel size. Must be odd.

    Returns:
        np.ndarray: Filtered image.
    """
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    filtered = cv2.medianBlur(image_uint8, kernel_size).astype(np.float32) / 255.0
    return np.clip(filtered, 0.0, 1.0)


def preprocess_image(image: np.ndarray, settings: Dict[str, Any]) -> np.ndarray:
    """
    Run the full preprocessing pipeline.

    Processing order:
        1. grayscale conversion
        2. linear contrast / brightness adjustment
        3. optional CLAHE
        4. optional Gaussian blur
        5. optional median blur
        6. optional inversion
        7. optional binarization

    Args:
        image: Raw input image.
        settings: Preprocessing-related settings dictionary.

    Returns:
        np.ndarray: Preprocessed float32 image.
    """
    gray = ensure_grayscale(image)

    result = apply_linear_adjustment(
        gray,
        contrast_factor=settings.get("contrast_factor", 100),
        brightness_offset=settings.get("brightness_offset", 0),
    )

    if settings.get("clahe", False):
        result = apply_clahe(result)

    sigma = float(settings.get("blur_sigma", 0.0))
    if sigma > 0.0:
        result = cv2.GaussianBlur(
            result,
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        ).astype(np.float32)

    if settings.get("median_blur", False):
        result = apply_median_blur(result, kernel_size=5)

    if settings.get("invert", False):
        result = (1.0 - result).astype(np.float32)

    if settings.get("binarize", False):
        threshold = threshold_otsu(result)
        result = (result > threshold).astype(np.float32)

    return np.clip(result, 0.0, 1.0).astype(np.float32)
