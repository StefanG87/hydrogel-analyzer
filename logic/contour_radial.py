# logic/contour_radial.py

"""
logic/contour_radial.py

Provides the radial ray-casting contour detection engine.

This module is responsible for:

- radial sampling from a specified origin
- directional edge detection along each ray
- threshold handling
- candidate filtering
- contour point cleanup
- spline fitting with robust fallback

This module must not contain UI code.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
from scipy.interpolate import splprep, splev

from core.models import ContourResult

Point = Tuple[int, int]
RayTuple = Tuple[Point, Point, bool]


def _angle_from_origin(origin: Tuple[float, float], point: Tuple[float, float]) -> float:
    """Return the polar angle of a point relative to the origin."""
    ox, oy = origin
    px, py = point
    return math.atan2(py - oy, px - ox)


def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Return Euclidean distance between two points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _sort_points_by_angle(
    points: Iterable[Tuple[float, float]],
    origin: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Sort points by polar angle around the given origin."""
    return sorted(points, key=lambda p: _angle_from_origin(origin, p))


def _deduplicate_points(
    points: Iterable[Tuple[float, float]],
    min_distance: float = 0.75,
) -> List[Tuple[float, float]]:
    """
    Remove near-duplicate points while preserving order.

    Args:
        points: Input points.
        min_distance: Minimum distance required between accepted neighbors.

    Returns:
        list[tuple[float, float]]: Filtered points.
    """
    filtered: List[Tuple[float, float]] = []
    last: Optional[Tuple[float, float]] = None

    for point in points:
        if last is None or _euclidean_distance(last, point) > min_distance:
            filtered.append(point)
            last = point

    return filtered


def _filter_large_neighbor_jumps(
    points: List[Tuple[float, float]],
    rel_jump_threshold: float = 0.6,
) -> List[Tuple[float, float]]:
    """
    Remove points behind excessive arc-length jumps.

    Args:
        points: Ordered contour points.
        rel_jump_threshold: Relative threshold compared to median spacing.

    Returns:
        list[tuple[float, float]]: Filtered points.
    """
    if len(points) < 6:
        return points[:]

    gaps = [_euclidean_distance(points[i], points[i + 1]) for i in range(len(points) - 1)]
    median_gap = float(np.median(gaps)) if gaps else 0.0

    if median_gap <= 0.0:
        return points[:]

    keep = [True] * len(points)
    for i, gap in enumerate(gaps):
        if gap > (1.0 + rel_jump_threshold) * median_gap:
            keep[i + 1] = False

    return [point for point, keep_flag in zip(points, keep) if keep_flag]


def _clip_to_baseline(
    points: Iterable[Tuple[float, float]],
    baseline_func: Optional[Callable[[float], float]],
    tolerance: float = 0.5,
) -> List[Tuple[float, float]]:
    """
    Keep only points at or above the baseline.

    In image coordinates, smaller y means visually higher.

    Args:
        points: Candidate contour points.
        baseline_func: Baseline function y = f(x).
        tolerance: Numerical tolerance.

    Returns:
        list[tuple[float, float]]: Clipped point set.
    """
    if baseline_func is None:
        return list(points)

    clipped: List[Tuple[float, float]] = []
    for x, y in points:
        baseline_y = float(baseline_func(x))
        if y <= baseline_y + tolerance:
            clipped.append((x, y))
    return clipped


def _sample_polyline(
    points: List[Tuple[float, float]],
    samples: int = 500,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Sample a piecewise-linear polyline through the given points.

    This mode performs no smoothing and therefore cannot overshoot
    beyond the provided supporting points.

    Args:
        points: Ordered contour points.
        samples: Number of output samples.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            Sampled x and y arrays.
    """
    if len(points) < 2:
        return None, None

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    if distances.size == 0:
        return x.copy(), y.copy()

    cumulative = np.concatenate(([0.0], np.cumsum(distances)))
    total_length = cumulative[-1]

    if total_length <= 0.0:
        return x.copy(), y.copy()

    targets = np.linspace(0.0, total_length, int(samples))
    poly_x = np.interp(targets, cumulative, x)
    poly_y = np.interp(targets, cumulative, y)

    return poly_x, poly_y


def _fit_spline_or_polyline(
    points: List[Tuple[float, float]],
    smoothness: float = 5.0,
    samples: int = 500,
    curve_mode: str = "polyline",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build a display curve through the contour points.

    Supported modes:
    - "polyline": piecewise-linear interpolation without smoothing
    - "spline": smoothed spline fit with polyline fallback on failure

    Args:
        points: Ordered contour points.
        smoothness: Spline smoothing factor (used only in spline mode).
        samples: Number of output samples.
        curve_mode: Display curve mode.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]:
            Curve x and y arrays.
    """
    if len(points) < 2:
        return None, None

    mode = str(curve_mode).strip().lower()

    if mode == "polyline":
        return _sample_polyline(points, samples=samples)

    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    spline_order = int(min(3, max(1, len(points) - 1)))
    spline_smoothness = float(smoothness) * max(1.0, len(points) / 40.0)

    try:
        tck, _ = splprep([x, y], k=spline_order, s=spline_smoothness)
        u = np.linspace(0.0, 1.0, int(samples))
        spline_x, spline_y = splev(u, tck)
        return np.asarray(spline_x), np.asarray(spline_y)
    except Exception:
        return _sample_polyline(points, samples=samples)


def _extract_directional_magnitudes(
    deltas: np.ndarray,
    edge_direction: str,
) -> np.ndarray:
    """Extract edge-direction-aware positive magnitudes from signed deltas."""
    if edge_direction == "dark_to_bright":
        return deltas[deltas > 0].astype(np.float32, copy=False)

    if edge_direction == "bright_to_dark":
        return (-deltas[deltas < 0]).astype(np.float32, copy=False)

    return np.abs(deltas).astype(np.float32, copy=False)


def _compute_otsu_threshold(values: np.ndarray) -> float:
    """
    Compute an Otsu threshold for 1D edge magnitudes.

    The incoming values are expected in intensity-jump units and are clipped
    into the 8-bit range because ray deltas are derived from uint8 images.
    """
    if values.size == 0:
        return 0.0

    values_uint8 = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    histogram = np.bincount(values_uint8, minlength=256).astype(np.float64)

    nonzero_bins = np.flatnonzero(histogram)
    if nonzero_bins.size <= 1:
        return float(nonzero_bins[0]) if nonzero_bins.size == 1 else 0.0

    bin_indices = np.arange(256, dtype=np.float64)
    total = float(histogram.sum())
    cumulative_weight = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * bin_indices)
    denominator = cumulative_weight * (total - cumulative_weight)

    between_class_variance = np.zeros_like(histogram)
    valid = denominator > 0.0
    if np.any(valid):
        numerator = (cumulative_mean[-1] * cumulative_weight[valid] - cumulative_mean[valid]) ** 2
        between_class_variance[valid] = numerator / denominator[valid]

    return float(np.argmax(between_class_variance))


def _compute_threshold(
    deltas: np.ndarray,
    threshold: float,
    threshold_mode: str,
    edge_direction: str,
) -> float:
    """
    Compute the effective threshold for a ray profile.

    Supported modes:
    - manual: fixed user threshold
    - auto_percentile: 90th percentile of directional edge magnitudes
    - mean_std: max(user threshold, mean + std of directional magnitudes)
    - otsu: max(user threshold, Otsu threshold of directional magnitudes)
    """
    mode = str(threshold_mode).strip().lower()
    manual_threshold = max(float(threshold), 0.0)

    if mode == "manual":
        return manual_threshold

    magnitudes = _extract_directional_magnitudes(deltas, edge_direction)
    if magnitudes.size == 0:
        return max(manual_threshold, 20.0)

    if mode == "auto_percentile":
        return float(np.percentile(magnitudes, 90))

    if mode == "mean_std":
        derived_threshold = float(np.mean(magnitudes) + np.std(magnitudes))
        if not np.isfinite(derived_threshold):
            return manual_threshold
        return max(manual_threshold, derived_threshold)

    if mode == "otsu":
        derived_threshold = _compute_otsu_threshold(magnitudes)
        if not np.isfinite(derived_threshold):
            return manual_threshold
        return max(manual_threshold, derived_threshold)

    return manual_threshold


def detect_contour_radial(
    image: np.ndarray,
    origin_x: float,
    origin_y: float,
    baseline_func: Optional[Callable[[float], float]] = None,
    threshold: float = 30,
    threshold_mode: str = "auto_percentile",
    min_delta: float = 5,
    num_rays: int = 60,
    max_ray_length_px: int = 600,
    angle_span_deg: float = 180,
    smoothness: float = 5,
    max_dev_pct: float = 30,
    edge_direction: str = "dark_to_bright",
    curve_mode: str = "polyline",
) -> ContourResult:
    """
    Perform radial ray-casting contour detection.

    Args:
        image: Preprocessed grayscale image.
        origin_x: Ray origin x coordinate.
        origin_y: Ray origin y coordinate.
        baseline_func: Optional baseline function.
        threshold: Threshold value.
        threshold_mode: Threshold mode.
        min_delta: Minimum valid jump.
        num_rays: Number of rays.
        angle_span_deg: Angular range.
        smoothness: Spline smoothness.
        max_dev_pct: Maximum neighbor deviation in percent.
        edge_direction: Edge polarity mode.

    Returns:
        ContourResult: Structured contour output.
    """
    if image.ndim == 3:
        image = image[..., 0]

    height, width = image.shape[:2]

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            working = (image * 255).astype(np.uint8)
        else:
            working = np.clip(image, 0, 255).astype(np.uint8)
    else:
        working = image

    cx = float(origin_x)
    cy = float(origin_y)

    angle_center_deg = 90.0
    angle_start = np.deg2rad(angle_center_deg - angle_span_deg / 2.0)
    angle_end = np.deg2rad(angle_center_deg + angle_span_deg / 2.0)
    angles = np.linspace(angle_start, angle_end, int(num_rays))
    radii = np.arange(1, max(2, int(max_ray_length_px)), dtype=np.float32)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    rays: List[RayTuple] = []
    raw_points: List[Tuple[Optional[Point], float]] = []

    for cos_angle, sin_angle in zip(cos_angles, sin_angles):
        x_line = (cx + radii * cos_angle).astype(np.int32)
        y_line = (cy - radii * sin_angle).astype(np.int32)

        valid_mask = (
            (0 <= x_line)
            & (x_line < width)
            & (0 <= y_line)
            & (y_line < height)
        )

        if baseline_func is not None:
            baseline_y = np.asarray(baseline_func(x_line), dtype=np.float32)
            valid_mask &= y_line <= baseline_y

        invalid_indices = np.flatnonzero(~valid_mask)
        valid_count = int(invalid_indices[0]) if invalid_indices.size else int(x_line.size)

        if valid_count < 2:
            rays.append(((0, 0), (0, 0), False))
            raw_points.append((None, 0.0))
            continue

        x_valid = x_line[:valid_count]
        y_valid = y_line[:valid_count]
        profile = working[y_valid, x_valid].astype(np.int16, copy=False)
        deltas = np.diff(profile)

        effective_threshold = _compute_threshold(
            deltas=deltas,
            threshold=threshold,
            threshold_mode=threshold_mode,
            edge_direction=edge_direction,
        )

        if edge_direction == "dark_to_bright":
            candidate_indices = np.flatnonzero(
                (deltas > effective_threshold) & (deltas > float(min_delta))
            )
        elif edge_direction == "bright_to_dark":
            candidate_indices = np.flatnonzero(
                (-deltas > efffective_threshold) & (-deltas > float(min_delta))
            )
        else:
            abs_deltas = np.abs(deltas)
            candidate_indices = np.flatnonzero(
                (abs_deltas > efffective_threshold) & (abs_deltas > float(min_delta))
            )

        if candidate_indices.size:
            hit_index = int(candidate_indices[-1] + 1)
            best_hit = (int(x_valid[hit_index]), int(y_valid[hit_index]))
            best_distance = float(radii[hit_index])
            rays.append(((int(x_valid[0]), int(y_valid[0])), best_hit, True))
            raw_points.append((best_hit, best_distance))
        else:
            rays.append(
                (
                    (int(x_valid[0]), int(y_valid[0])),
                    (int(x_valid[-1]), int(y_valid[-1])),
                    False,
                )
            )
            raw_points.append((None, 0.0))

    accepted_points: List[Tuple[float, float]] = []

    for i, (point, distance) in enumerate(raw_points):
        if point is None:
            continue

        neighbor_distances: List[float] = []

        for j in range(i - 1, -1, -1):
            if raw_points[j][0] is not None:
                neighbor_distances.append(raw_points[j][1])
                break

        for j in range(i + 1, len(raw_points)):
            if raw_points[j][0] is not None:
                neighbor_distances.append(raw_points[j][1])
                break

        if not neighbor_distances:
            accepted_points.append((float(point[0]), float(point[1])))
            continue

        average_distance = float(np.mean(neighbor_distances))
        if average_distance <= 0.0:
            accepted_points.append((float(point[0]), float(point[1])))
            continue

        deviation_pct = abs(distance - average_distance) / average_distance * 100.0
        if deviation_pct <= max_dev_pct:
            accepted_points.append((float(point[0]), float(point[1])))
        else:
            start, end, _ = rays[i]
            rays[i] = (start, end, False)

    cleaned_points = _sort_points_by_angle(accepted_points, origin=(cx, cy))
    cleaned_points = _deduplicate_points(cleaned_points, min_distance=0.75)
    cleaned_points = _filter_large_neighbor_jumps(cleaned_points, rel_jump_threshold=0.6)
    cleaned_points = _clip_to_baseline(cleaned_points, baseline_func, tolerance=0.5)

    spline_x, spline_y = _fit_spline_or_polyline(
        cleaned_points,
        smoothness=float(smoothness),
        samples=500,
        curve_mode=curve_mode,
    )

    supporting_points: List[Point] = [
        (int(round(x)), int(round(y)))
        for x, y in cleaned_points
    ]

    return ContourResult(
        supporting_points=supporting_points,
        rays=rays,
        spline_x=spline_x,
        spline_y=spline_y,
    )
