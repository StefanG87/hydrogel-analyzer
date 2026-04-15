# logic/geometry.py

"""
logic/geometry.py

Provides geometry extraction functions for the Hydrogel Analyzer project.

This module is responsible for:

- measuring contour width relative to the baseline
- measuring contour height relative to the baseline
- computing optional um values
- preparing geometric reference points for later overlay rendering

This module must remain purely mathematical and must not contain UI code
or plotting code.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

from core.models import GeometryResult
from logic.baseline import (
    get_baseline_normal_vector,
    get_baseline_unit_vector,
    normalize_baseline,
    signed_distance_to_baseline,
)

PointLike = Tuple[float, float]


def _to_point_matrix(points: Iterable[PointLike]) -> np.ndarray:
    """
    Convert a point iterable into an Nx2 float matrix.

    Args:
        points: Iterable of (x, y) coordinates.

    Returns:
        np.ndarray: Point matrix with shape (N, 2).
    """
    rows = []
    for point in points:
        try:
            if len(point) != 2:
                continue
        except TypeError:
            continue
        rows.append((float(point[0]), float(point[1])))

    if not rows:
        return np.empty((0, 2), dtype=float)

    return np.asarray(rows, dtype=float)


def extract_geometry_from_contour(
    contour_points: Iterable[PointLike],
    baseline: object,
    um_per_px: Optional[float] = None,
) -> GeometryResult:
    """
    Extract geometric measurements from contour points relative to a baseline.

    Width is computed along the baseline direction.
    Height is computed orthogonal to the baseline.

    Args:
        contour_points: Contour support points as (x, y) pairs.
        baseline: Baseline definition.
        um_per_px: Optional scale factor for um conversion.

    Returns:
        GeometryResult: Structured geometry output.
    """
    normalized_baseline = normalize_baseline(baseline)
    points = _to_point_matrix(contour_points)

    if normalized_baseline is None or points.size == 0:
        return GeometryResult()

    tangent = get_baseline_unit_vector(normalized_baseline)
    normal = get_baseline_normal_vector(normalized_baseline)

    if tangent is None or normal is None:
        return GeometryResult()

    origin = np.asarray(normalized_baseline[0], dtype=float)
    relative = points - origin

    parallel_positions = relative @ tangent
    signed_heights = relative @ normal

    width_px = float(np.ptp(parallel_positions))
    height_px = float(np.ptp(signed_heights))

    idx_min_parallel = int(np.argmin(parallel_positions))
    idx_max_parallel = int(np.argmax(parallel_positions))
    idx_max_abs_height = int(np.argmax(np.abs(signed_heights)))

    width_point_a = points[idx_min_parallel]
    width_point_b = points[idx_max_parallel]

    width_proj_a = origin + parallel_positions[idx_min_parallel] * tangent
    width_proj_b = origin + parallel_positions[idx_max_parallel] * tangent

    height_point = points[idx_max_abs_height]
    height_proj = origin + parallel_positions[idx_max_abs_height] * tangent

    reference_vectors = {
        "width_point_a": width_point_a,
        "width_point_b": width_point_b,
        "width_projection_a": width_proj_a,
        "width_projection_b": width_proj_b,
        "height_point": height_point,
        "height_projection": height_proj,
        "tangent_unit": tangent,
        "normal_unit": normal,
    }

    return GeometryResult(
        height_px=height_px,
        width_px=width_px,
        height_um=(height_px * um_per_px) if um_per_px is not None else None,
        width_um=(width_px * um_per_px) if um_per_px is not None else None,
        reference_vectors=reference_vectors,
    )


def compute_width_from_contour(
    contour_points: Iterable[PointLike],
    baseline: object,
) -> Optional[float]:
    """
    Compute only the width of a contour relative to the baseline.

    Args:
        contour_points: Contour support points.
        baseline: Baseline definition.

    Returns:
        float | None: Width in pixels, or None if not computable.
    """
    result = extract_geometry_from_contour(contour_points, baseline)
    return result.width_px


def compute_height_from_contour(
    contour_points: Iterable[PointLike],
    baseline: object,
) -> Optional[float]:
    """
    Compute only the height of a contour relative to the baseline.

    Args:
        contour_points: Contour support points.
        baseline: Baseline definition.

    Returns:
        float | None: Height in pixels, or None if not computable.
    """
    result = extract_geometry_from_contour(contour_points, baseline)
    return result.height_px


def get_topmost_contour_point(
    contour_points: Iterable[PointLike],
) -> Optional[Tuple[float, float]]:
    """
    Return the contour point with the smallest y coordinate.

    Args:
        contour_points: Contour support points.

    Returns:
        tuple[float, float] | None: Topmost point, or None if unavailable.
    """
    points = list(contour_points)
    if not points:
        return None
    return min(points, key=lambda p: p[1])


def get_maximum_signed_height_point(
    contour_points: Iterable[PointLike],
    baseline: object,
) -> Optional[Tuple[float, float]]:
    """
    Return the contour point with maximum absolute signed distance to baseline.

    Args:
        contour_points: Contour support points.
        baseline: Baseline definition.

    Returns:
        tuple[float, float] | None: Selected point, or None if unavailable.
    """
    points = list(contour_points)
    if not points or normalize_baseline(baseline) is None:
        return None

    distances = []
    for point in points:
        distance = signed_distance_to_baseline(point, baseline)
        if distance is None:
            continue
        distances.append((abs(distance), point))

    if not distances:
        return None

    return max(distances, key=lambda item: item[0])[1]
