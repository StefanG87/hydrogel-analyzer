# logic/analyzer.py

"""
logic/analyzer.py

Provides the main analysis pipeline for the Hydrogel Analyzer project.

This module is responsible for orchestrating the full image analysis workflow:

1. validate or create baseline
2. compute effective origin
3. preprocess the image
4. run radial contour detection
5. extract geometry
6. assemble a structured AnalysisResult

This module must be the single entry point for contour analysis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from core.models import AnalysisMetadata, AnalysisResult
from core.settings_manager import SettingsManager
from logic.baseline import create_baseline_func, create_default_baseline, normalize_baseline
from logic.contour_radial import detect_contour_radial
from logic.geometry import extract_geometry_from_contour
from tools.image_preprocessing import preprocess_image


class ContourAnalyzer:
    """
    Central analysis orchestrator.

    All UI and batch modules should call this class instead of directly
    calling contour detection or geometry functions.
    """

    @staticmethod
    def process_image(
        image: np.ndarray,
        settings: Dict[str, Any],
        image_path: Optional[str] = None,
        include_debug_images: bool = False,
    ) -> AnalysisResult:
        """
        Run the complete contour analysis pipeline on one image.

        Args:
            image: Raw input image.
            settings: Settings dictionary.
            image_path: Optional image path for metadata.
            include_debug_images: Whether to keep processed debug images.

        Returns:
            AnalysisResult: Full structured analysis output.
        """
        start_time = time.perf_counter()

        working_settings = dict(settings)

        baseline = normalize_baseline(working_settings.get("baseline"))
        if baseline is None:
            height, width = image.shape[:2]
            baseline = create_default_baseline(width, height)
            working_settings["baseline"] = baseline

        origin_x, origin_y = SettingsManager.get_effective_origin(working_settings)

        if working_settings.get("preprocess_enabled", True):
            processed_image = preprocess_image(image, working_settings)
        else:
            processed_image = preprocess_image(image, {"contrast_factor": 100, "brightness_offset": 0})

        baseline_func = create_baseline_func(baseline)

        contour_result = detect_contour_radial(
            image=processed_image,
            origin_x=origin_x,
            origin_y=origin_y,
            baseline_func=baseline_func,
            threshold=working_settings.get("threshold", 30),
            threshold_mode=working_settings.get("threshold_mode", "auto_percentile"),
            min_delta=working_settings.get("min_delta", 5),
            num_rays=working_settings.get("num_rays", 60),
            max_ray_length_px=working_settings.get("max_ray_length_px", 600),
            angle_span_deg=working_settings.get("angle_span_deg", 180),
            smoothness=working_settings.get("smoothness", 5),
            max_dev_pct=working_settings.get("max_dev_pct", 30),
            edge_direction=working_settings.get("edge_direction", "dark_to_bright"),
            curve_mode=working_settings.get("curve_mode", "polyline"),
        )

        geometry_result = extract_geometry_from_contour(
            contour_points=contour_result.supporting_points,
            baseline=baseline,
            um_per_px=working_settings.get("um_per_px"),
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = AnalysisMetadata(
            image_path=image_path,
            threshold_used=float(working_settings.get("threshold", 30)),
            threshold_mode=working_settings.get("threshold_mode", "auto_percentile"),
            num_rays=int(working_settings.get("num_rays", 60)),
            angle_span_deg=float(working_settings.get("angle_span_deg", 180)),
            processing_time_ms=elapsed_ms,
        )

        return AnalysisResult(
            contour=contour_result,
            geometry=geometry_result,
            baseline=baseline,
            origin=(int(round(origin_x)), int(round(origin_y))),
            metadata=metadata,
            debug={"processed_image": processed_image} if include_debug_images else {},
        )