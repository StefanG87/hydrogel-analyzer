# core/settings_defaults.py

"""
core/settings_defaults.py

Defines the default settings schema for the Hydrogel Analyzer project.

This module is intentionally data-centric and must not contain any GUI logic.

Responsibilities:
- provide one centralized DEFAULT_SETTINGS dictionary
- define optional allowed value constants for constrained settings
- provide a helper for creating a fresh settings copy

Design rules:
- keep the settings structure flat and dictionary-based
- preserve full compatibility with PreviewWindow, BatchWorker,
  SettingsManager, ContourAnalyzer, and UI dialogs
- avoid runtime state or singleton behavior in this module
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Final


# ---------------------------------------------------------------------
# Allowed constrained values
# ---------------------------------------------------------------------

THRESHOLD_MODES: Final[tuple[str, ...]] = (
    "manual",
    "auto_percentile",
    "mean_std",
    "otsu",
)

EDGE_DIRECTIONS: Final[tuple[str, ...]] = (
    "dark_to_bright",
    "bright_to_dark",
    "both",
)

CURVE_MODES: Final[tuple[str, ...]] = (
    "polyline",
    "spline",
)

# ---------------------------------------------------------------------
# Central default settings
# ---------------------------------------------------------------------

DEFAULT_SETTINGS: Final[Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # Detection settings
    # -----------------------------------------------------------------
    "threshold": 30,
    "threshold_mode": "auto_percentile",
    "min_delta": 5,
    "num_rays": 60,
    "max_ray_length_px": 600,
    "angle_span_deg": 180,
    "smoothness": 5,
    "max_dev_pct": 30,
    "edge_direction": "dark_to_bright",
    "curve_mode": "polyline",

    # -----------------------------------------------------------------
    # Baseline and origin settings
    # -----------------------------------------------------------------
    # Baseline format:
    # [(x1, y1), (x2, y2)] or None
    "baseline": None,

    # Origin logic:
    # auto_origin=True  -> use baseline midpoint
    # manual_origin=True -> apply origin_dx/origin_dy relative to midpoint
    # auto_origin=False -> use explicit origin_x/origin_y
    "auto_origin": True,
    "manual_origin": False,

    # Explicit absolute origin (used when auto_origin is False)
    "origin_x": 0,
    "origin_y": 0,

    # Relative offset from baseline midpoint (used when auto_origin and manual_origin are True)
    "origin_dx": 0,
    "origin_dy": 0,

    # -----------------------------------------------------------------
    # Preprocessing settings
    # -----------------------------------------------------------------
    "preprocess_enabled": True,
    "show_preprocessing": False,
    "contrast_factor": 100,
    "brightness_offset": 0,
    "blur_sigma": 0.0,
    "median_blur": False,
    "clahe": False,
    "invert": False,
    "binarize": False,

    # -----------------------------------------------------------------
    # Overlay / display settings
    # -----------------------------------------------------------------
    # These are used both by preview rendering and optional video export.
    "show_rays": False,
    "show_supporting_points": False,
    "show_geometry": False,
    "draw_baseline": True,
    "draw_spline": True,
    "show_origin": False,
    "show_timestamp": True,

    # -----------------------------------------------------------------
    # ROI settings
    # -----------------------------------------------------------------
    # Currently not actively consumed by the main analysis pipeline,
    # but retained for compatibility and future extension.
    "use_roi": False,
    "roi_width": 500,
    "roi_height": 300,

    # -----------------------------------------------------------------
    # Calibration / unit conversion
    # -----------------------------------------------------------------
    "um_per_px": None,
    
    # -----------------------------------------------------------------
    # Batch outlier settings
    # -----------------------------------------------------------------
    "batch_outlier_filter_enabled": True,
    "batch_max_height_jump_pct": 30.0,
    "batch_max_width_jump_pct": 30.0,
    "batch_min_supporting_points_ratio": 0.5,
}


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def create_default_settings() -> Dict[str, Any]:
    """
    Return a fresh deep copy of the centralized default settings.

    Returns:
        dict: Independent settings dictionary initialized from DEFAULT_SETTINGS.
    """
    return deepcopy(DEFAULT_SETTINGS)