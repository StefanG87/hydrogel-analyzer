# logic/plotting.py

"""
logic/plotting.py

Provides overlay drawing functions for Hydrogel Analyzer results.

This module is responsible for drawing:

- the underlying image
- contour support points
- rays
- spline curve
- baseline
- origin marker
- geometry annotations

This module must only draw data that is already computed.
It must not perform contour analysis or geometry calculations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from matplotlib.axes import Axes

from core.models import AnalysisResult, GeometryResult


def _format_length(px_value: Optional[float], um_value: Optional[float]) -> str:
    """
    Format a geometry value for display.

    Args:
        px_value: Pixel measurement.
        um_value: Optional µm measurement.

    Returns:
        str: Formatted label.
    """
    if um_value is not None:
        return f"{um_value:.1f} µm"
    if px_value is not None:
        return f"{px_value:.1f} px"
    return "-"


def draw_analysis_result(
    ax: Axes,
    image: np.ndarray,
    result: AnalysisResult,
    settings: Dict[str, Any],
) -> None:
    """
    Draw a full analysis overlay onto a Matplotlib axis.

    Args:
        ax: Target axis.
        image: Image to display.
        result: Structured analysis result.
        settings: Display settings dictionary.
    """
    ax.clear()
    ax.imshow(image, cmap="gray", origin="upper")
    ax.set_axis_off()

    if settings.get("show_rays", False):
        _draw_rays(ax, result)

    if settings.get("show_supporting_points", False):
        _draw_supporting_points(ax, result)

    if settings.get("draw_spline", True):
        _draw_spline(ax, result)

    if settings.get("draw_baseline", True):
        _draw_baseline(ax, result)

    if settings.get("show_origin", False):
        _draw_origin(ax, result)

    if settings.get("show_geometry", True):
        _draw_geometry(ax, result)



def _draw_supporting_points(ax: Axes, result: AnalysisResult) -> None:
    """Draw accepted supporting points from the radial ray hits."""
    if not result.contour.supporting_points:
        return

    points = np.array(result.contour.supporting_points, dtype=float)
    ax.plot(points[:, 0], points[:, 1], "go", markersize=2)


def _draw_rays(ax: Axes, result: AnalysisResult) -> None:
    """Draw radial rays."""
    for start, end, valid in result.contour.rays:
        color = "w" if valid else "r"
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=0.5)


def _draw_spline(ax: Axes, result: AnalysisResult) -> None:
    """Draw fitted contour spline."""
    if result.contour.spline_x is None or result.contour.spline_y is None:
        return

    ax.plot(
        result.contour.spline_x,
        result.contour.spline_y,
        "r-",
        linewidth=1.5,
    )


def _draw_baseline(ax: Axes, result: AnalysisResult) -> None:
    """Draw the infinite baseline across the image width."""
    if result.baseline is None:
        return

    (x1, y1), (x2, y2) = result.baseline
    if x2 == x1:
        ax.axvline(x=x1, color="b", linewidth=1.0)
        return

    image_width = ax.images[0].get_array().shape[1]
    x_values = np.array([0, image_width], dtype=float)
    slope = (y2 - y1) / (x2 - x1)
    y_values = slope * (x_values - x1) + y1
    ax.plot(x_values, y_values, "b-", linewidth=1.0)


def _draw_origin(ax: Axes, result: AnalysisResult) -> None:
    """Draw the analysis origin marker."""
    if result.origin is None:
        return

    ox, oy = result.origin
    ax.plot(ox, oy, marker="x", markersize=8, color="yellow", markeredgewidth=1.5)


def _draw_geometry(ax: Axes, result: AnalysisResult) -> None:
    """
    Draw geometry arrows and labels.

    Width is shown as an offset arrow parallel to the baseline.
    Height is shown as an offset arrow parallel to the measured height vector,
    shifted to the right along the baseline direction so the gel remains visible.
    """
    geometry = result.geometry
    if geometry is None or geometry.reference_vectors is None:
        return

    refs = geometry.reference_vectors

    width_projection_a = refs.get("width_projection_a")
    width_projection_b = refs.get("width_projection_b")
    width_point_a = refs.get("width_point_a")
    width_point_b = refs.get("width_point_b")

    height_projection = refs.get("height_projection")
    height_point = refs.get("height_point")

    tangent_unit = refs.get("tangent_unit")
    normal_unit = refs.get("normal_unit")

    # --------------------------------------------------------------
    # Width: offset away from gel along baseline normal
    # --------------------------------------------------------------
    if (
        width_projection_a is not None
        and width_projection_b is not None
        and normal_unit is not None
    ):
        width_offset_px = 60.0
        wp_a = np.array(width_projection_a, dtype=float) + np.array(normal_unit, dtype=float) * width_offset_px
        wp_b = np.array(width_projection_b, dtype=float) + np.array(normal_unit, dtype=float) * width_offset_px

        ax.annotate(
            "",
            xy=wp_a,
            xytext=wp_b,
            arrowprops=dict(arrowstyle="<->", color="white"),
        )

        if width_point_a is not None:
            ax.plot(
                [width_point_a[0], wp_a[0]],
                [width_point_a[1], wp_a[1]],
                linestyle="dashed",
                color="white",
                linewidth=0.8,
            )

        if width_point_b is not None:
            ax.plot(
                [width_point_b[0], wp_b[0]],
                [width_point_b[1], wp_b[1]],
                linestyle="dashed",
                color="white",
                linewidth=0.8,
            )

        center = (wp_a + wp_b) / 2.0
        width_label = _format_length(geometry.width_px, geometry.width_um)
        ax.text(
            center[0],
            center[1] + 10,
            width_label,
            color="white",
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.2"),
        )

    # --------------------------------------------------------------
    # Height:
    # Use the real measured height vector from:
    #   height_projection -> height_point
    #
    # Display logic:
    # - bottom helper line stays on the baseline
    # - top helper line goes through the real maximum point and is
    #   parallel to the baseline
    # - the displayed height arrow is a translated copy of the real
    #   measured height vector, shifted to the right along the baseline
    # --------------------------------------------------------------
    if (
        height_projection is not None
        and height_point is not None
        and width_projection_b is not None
        and tangent_unit is not None
        and geometry.height_px is not None
    ):
        right_offset_px = 50.0

        tangent = np.array(tangent_unit, dtype=float)

        height_projection = np.array(height_projection, dtype=float)
        height_point = np.array(height_point, dtype=float)
        width_projection_b = np.array(width_projection_b, dtype=float)

        # Real measured height vector: orthogonal from baseline to maximum
        measured_height_vector = height_point - height_projection

        # Place displayed height arrow on the baseline, shifted to the right
        # relative to the right width reference point.
        base_point = width_projection_b + tangent * right_offset_px

        # Translate the real measured height vector to the display position
        top_point_display = base_point + measured_height_vector

        # Bottom helper line: on the baseline
        ax.plot(
            [height_projection[0], base_point[0]],
            [height_projection[1], base_point[1]],
            linestyle="dashed",
            color="white",
            linewidth=0.8,
        )

        # Top helper line: through the real maximum point, parallel to baseline
        ax.plot(
            [height_point[0], top_point_display[0]],
            [height_point[1], top_point_display[1]],
            linestyle="dashed",
            color="white",
            linewidth=1.0,
        )

        # Height arrow: translated copy of the actual measured height
        ax.annotate(
            "",
            xy=base_point,
            xytext=top_point_display,
            arrowprops=dict(arrowstyle="<->", color="white"),
        )

        # Label
        midpoint = (base_point + top_point_display) / 2.0
        height_label = _format_length(geometry.height_px, geometry.height_um)
        ax.text(
            midpoint[0] + 6,
            midpoint[1],
            height_label,
            color="white",
            ha="left",
            va="center",
            fontsize=9,
            bbox=dict(
                facecolor="black",
                alpha=0.5,
                edgecolor="none",
                boxstyle="round,pad=0.2",
            ),
        )