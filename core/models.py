# core/models.py

"""
core/models.py

Defines the shared data structures used across the Hydrogel Analyzer project.

These dataclasses form the contract between:

- analyzer
- contour detection
- geometry extraction
- plotting
- preview window
- batch processing
- export pipeline

All modules must use these structures instead of loose dictionaries.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# ---------------------------------------------------------
# Basic type aliases
# ---------------------------------------------------------

Point = Tuple[int, int]
FloatPoint = Tuple[float, float]
Ray = Tuple[Point, Point, bool]


# ---------------------------------------------------------
# Geometry Result
# ---------------------------------------------------------

@dataclass
class GeometryResult:
    """
    Stores extracted geometric measurements relative to baseline.
    """

    height_px: Optional[float] = None
    width_px: Optional[float] = None

    height_um: Optional[float] = None
    width_um: Optional[float] = None

    reference_vectors: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------
# Contour Result
# ---------------------------------------------------------

@dataclass
class ContourResult:
    """
    Stores radial contour detection output.

    supporting_points:
        Accepted ray-hit points after filtering and cleanup.
        These are the discrete support points used for spline fitting.

    rays:
        Ray casting visualization data:
        ((x_start, y_start), (x_hit, y_hit), success_flag)

    spline_x / spline_y:
        Optional smoothed spline approximation through the supporting points.
    """

    supporting_points: List[Point] = field(default_factory=list)
    rays: List[Ray] = field(default_factory=list)

    spline_x: Optional[np.ndarray] = None
    spline_y: Optional[np.ndarray] = None


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

@dataclass
class AnalysisMetadata:
    """
    Stores diagnostic metadata useful for debugging,
    reproducibility and export pipelines.
    """

    image_path: Optional[str] = None

    threshold_used: Optional[float] = None
    threshold_mode: Optional[str] = None

    num_rays: Optional[int] = None
    angle_span_deg: Optional[float] = None

    processing_time_ms: Optional[float] = None


# ---------------------------------------------------------
# Main Analysis Result
# ---------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    Complete output container returned by analyzer.process_image().

    contour:
        Structured radial detection output containing supporting points,
        rays, and spline approximation.

    This structure is consumed by:

    - preview window
    - plotting module
    - batch processor
    - CSV exporter
    - video renderer
    """

    contour: ContourResult
    geometry: Optional[GeometryResult] = None
    baseline: Optional[Tuple[Point, Point]] = None
    origin: Optional[Point] = None
    metadata: Optional[AnalysisMetadata] = None
    debug: Dict[str, Any] = field(default_factory=dict)