# logic/baseline.py

"""
logic/baseline.py

Provides baseline-related helper functions for the Hydrogel Analyzer project.

This module is responsible for:

- validating baseline point pairs
- normalizing baseline input
- creating a baseline function y = f(x)
- computing baseline midpoint
- computing baseline direction vectors
- computing the baseline angle
- generating a default baseline for a given image size

This module must remain purely mathematical and must not contain UI code.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------

Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Baseline = Tuple[Point, Point]


# ---------------------------------------------------------------------
# Validation and normalization
# ---------------------------------------------------------------------

def normalize_baseline(baseline: object) -> Optional[Baseline]:
    """
    Validate and normalize a baseline definition.

    Expected input format:
        [(x1, y1), (x2, y2)]
    or
        ((x1, y1), (x2, y2))

    Returns:
        tuple[Point, Point] | None:
            Normalized baseline as integer point pairs, or None if invalid.
    """
    if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
        return None

    p1, p2 = baseline

    if not (
        isinstance(p1, (list, tuple)) and len(p1) == 2 and
        isinstance(p2, (list, tuple)) and len(p2) == 2
    ):
        return None

    try:
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0]), int(p2[1])
    except (TypeError, ValueError):
        return None

    if x1 == x2 and y1 == y2:
        return None

    return (x1, y1), (x2, y2)


def is_valid_baseline(baseline: object) -> bool:
    """
    Return whether the provided baseline is valid.

    Args:
        baseline: Candidate baseline object.

    Returns:
        bool: True if baseline can be normalized successfully.
    """
    return normalize_baseline(baseline) is not None


# ---------------------------------------------------------------------
# Baseline construction helpers
# ---------------------------------------------------------------------

def create_default_baseline(image_width: int, image_height: int) -> Baseline:
    """
    Create a centered horizontal default baseline for a given image size.

    The baseline spans from 25% to 75% of the image width and is placed
    at 50% of the image height.

    Args:
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Baseline: Default baseline as two integer points.
    """
    x1 = int(image_width * 0.25)
    x2 = int(image_width * 0.75)
    y = int(image_height * 0.50)
    return (x1, y), (x2, y)


def create_baseline_func(baseline: object) -> Callable[[float], float]:
    """
    Create a callable baseline function y = f(x).

    For a valid non-vertical baseline, this returns the line equation.
    For a valid vertical baseline, the returned function yields the average y.
    For invalid input, the returned function yields 0.0.

    Args:
        baseline: Baseline object in standard two-point format.

    Returns:
        Callable[[float], float]: Function that maps x to y.
    """
    normalized = normalize_baseline(baseline)
    if normalized is None:
        return lambda x: 0.0

    (x1, y1), (x2, y2) = normalized

    if x2 == x1:
        y_mean = (y1 + y2) / 2.0
        return lambda x: y_mean

    slope = (y2 - y1) / (x2 - x1)
    return lambda x: slope * (x - x1) + y1


# ---------------------------------------------------------------------
# Geometric properties
# ---------------------------------------------------------------------

def get_baseline_midpoint(baseline: object) -> Optional[FloatPoint]:
    """
    Compute the midpoint of a baseline.

    Args:
        baseline: Baseline object.

    Returns:
        tuple[float, float] | None:
            Baseline midpoint, or None if invalid.
    """
    normalized = normalize_baseline(baseline)
    if normalized is None:
        return None

    (x1, y1), (x2, y2) = normalized
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_baseline_vector(baseline: object) -> Optional[np.ndarray]:
    """
    Compute the baseline direction vector.

    Args:
        baseline: Baseline object.

    Returns:
        np.ndarray | None:
            Vector from point 1 to point 2 as float array, or None if invalid.
    """
    normalized = normalize_baseline(baseline)
    if normalized is None:
        return None

    (x1, y1), (x2, y2) = normalized
    return np.array([x2 - x1, y2 - y1], dtype=float)


def get_baseline_unit_vector(baseline: object) -> Optional[np.ndarray]:
    """
    Compute the normalized baseline direction vector.

    Args:
        baseline: Baseline object.

    Returns:
        np.ndarray | None:
            Unit direction vector, or None if invalid.
    """
    vector = get_baseline_vector(baseline)
    if vector is None:
        return None

    length = np.linalg.norm(vector)
    if length == 0:
        return None

    return vector / length


def get_baseline_normal_vector(baseline: object) -> Optional[np.ndarray]:
    """
    Compute a unit normal vector orthogonal to the baseline.

    The normal is obtained by rotating the unit baseline vector by +90°:
        (x, y) -> (-y, x)

    Args:
        baseline: Baseline object.

    Returns:
        np.ndarray | None:
            Unit normal vector, or None if invalid.
    """
    tangent = get_baseline_unit_vector(baseline)
    if tangent is None:
        return None

    return np.array([-tangent[1], tangent[0]], dtype=float)


def compute_angle_deg(baseline: object) -> float:
    """
    Compute the baseline angle in degrees relative to the x-axis.

    Args:
        baseline: Baseline object.

    Returns:
        float:
            Angle in degrees. Returns 0.0 for invalid baselines.
    """
    vector = get_baseline_vector(baseline)
    if vector is None:
        return 0.0

    dx, dy = vector
    return math.degrees(math.atan2(dy, dx))


def project_point_to_baseline(point: tuple[float, float], baseline: object) -> Optional[np.ndarray]:
    """
    Project a point orthogonally onto the infinite baseline.

    Args:
        point: Point to project as (x, y).
        baseline: Baseline object.

    Returns:
        np.ndarray | None:
            Projected point on the baseline, or None if baseline is invalid.
    """
    normalized = normalize_baseline(baseline)
    if normalized is None:
        return None

    unit = get_baseline_unit_vector(normalized)
    if unit is None:
        return None

    (x1, y1), _ = normalized
    origin = np.array([x1, y1], dtype=float)
    p = np.array(point, dtype=float)

    projection_length = np.dot(p - origin, unit)
    return origin + projection_length * unit


def signed_distance_to_baseline(point: tuple[float, float], baseline: object) -> Optional[float]:
    """
    Compute the signed orthogonal distance from a point to the baseline.

    The sign depends on the baseline normal direction.

    Args:
        point: Point as (x, y).
        baseline: Baseline object.

    Returns:
        float | None:
            Signed orthogonal distance, or None if baseline is invalid.
    """
    normalized = normalize_baseline(baseline)
    if normalized is None:
        return None

    normal = get_baseline_normal_vector(normalized)
    if normal is None:
        return None

    (x1, y1), _ = normalized
    origin = np.array([x1, y1], dtype=float)
    p = np.array(point, dtype=float)

    return float(np.dot(p - origin, normal))