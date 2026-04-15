"""
tools/batch_worker.py

Provides threaded batch processing for Hydrogel Analyzer.

Two-phase execution model:

Phase 1:
    - run contour analysis
    - build CSV table
    - optionally keep AnalysisResult objects for video export

Phase 2:
    - compute optional plot configuration
    - render annotated video using the stored results
"""

from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal
from skimage import io

from core.models import AnalysisResult
from core.settings_manager import SettingsManager
from logic.analyzer import ContourAnalyzer
from tools.video_renderer import VideoRenderer


def _build_result_row_from_analysis(
    index: int,
    image_path: str,
    result: AnalysisResult,
    timestamp_s: Optional[float],
) -> Dict[str, Any]:
    """Build one CSV/export row from an analysis result."""
    geometry = result.geometry
    metadata = result.metadata

    return {
        "Index": index,
        "Filename": os.path.basename(image_path),
        "Timestamp [s]": timestamp_s,
        "Height [px]": geometry.height_px if geometry else None,
        "Width [px]": geometry.width_px if geometry else None,
        "Height [um]": geometry.height_um if geometry else None,
        "Width [um]": geometry.width_um if geometry else None,
        "Processing Time [ms]": metadata.processing_time_ms if metadata else None,
        "Supporting Points": len(result.contour.supporting_points) if result.contour else 0,
        "Valid Frame": True,
        "Outlier Reason": "",
    }


def _analyze_image_for_batch(
    index: int,
    image_path: str,
    settings: Dict[str, Any],
    timestamp_s: Optional[float],
    include_result: bool,
) -> Tuple[int, Dict[str, Any], Optional[AnalysisResult]]:
    """
    Analyze one batch image in a worker process.

    Keeping this as a module-level function makes it picklable for
    ProcessPoolExecutor on Windows.
    """
    image = io.imread(image_path)
    result = ContourAnalyzer.process_image(
        image=image,
        settings=settings,
        image_path=image_path,
        include_debug_images=False,
    )
    row = _build_result_row_from_analysis(
        index=index,
        image_path=image_path,
        result=result,
        timestamp_s=timestamp_s,
    )
    return index, row, result if include_result else None


class BatchWorker(QThread):
    """
    Run contour analysis for many images in a background thread.

    Signals:
        progress_update(current, total)
        finished(csv_path, video_path)
        aborted()
    """

    progress_update = pyqtSignal(int, int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(str, str)
    aborted = pyqtSignal()

    def __init__(
        self,
        image_paths: List[str],
        settings: Dict[str, Any],
        csv_path: str,
        delta_t: Optional[float] = None,
        create_video: bool = False,
        video_path: Optional[str] = None,
        fps: float = 24.0,
        video_settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.image_paths = list(image_paths)
        self.settings = dict(settings)
        self.csv_path = str(csv_path)
        self.delta_t = delta_t
        self.create_video = bool(create_video)
        self.video_path = str(video_path) if video_path else ""
        self.fps = float(fps)
        self.video_settings = dict(video_settings) if video_settings else {}

        self._cancel_requested = False

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def request_cancel(self) -> None:
        """Request graceful cancellation of the batch process."""
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute batch analysis."""
        if not self.image_paths:
            self.aborted.emit()
            return

        analysis_results: List[AnalysisResult] = []

        try:
            self.status_update.emit("Starting batch analysis...")
            analysis_phase = self._run_analysis_phase()
            if analysis_phase is None:
                return

            results_table, analysis_results = analysis_phase
            exported_video_path = ""

            if results_table:
                self._mark_temporal_outliers(results_table)

                settings_path = self._write_settings_json()
                self.status_update.emit("Writing CSV export...")
                self._write_csv(results_table, settings_path)

            if self.create_video and analysis_results:
                self.status_update.emit("Preparing video rendering...")
                plot_config = self._build_plot_config(results_table)

                if not self.video_path:
                    base = str(Path(self.csv_path).with_suffix(""))
                    self.video_path = base + ".mp4"

                first_image = io.imread(self.image_paths[0])
                height, width = first_image.shape[:2]

                renderer = VideoRenderer(
                    output_path=self.video_path,
                    frame_size=(width, height),
                    fps=self.fps,
                )

                total = len(self.image_paths)
                for index, image_path in enumerate(self.image_paths):
                    self.status_update.emit(
                        f"Rendering video {index + 1}/{total} frames..."
                    )

                    if self._cancel_requested:
                        renderer.finish()
                        self.aborted.emit()
                        return

                    image = io.imread(image_path)
                    renderer.add_frame(
                        image=image,
                        result=analysis_results[index],
                        settings=self.video_settings,
                        timestamp_s=self._compute_timestamp(index),
                        plot_config=plot_config,
                        frame_index=index,
                    )

                self.status_update.emit("Finalizing video...")
                renderer.finish()
                exported_video_path = self.video_path

            self.finished.emit(
                self.csv_path if results_table else "",
                exported_video_path if self.create_video else "",
            )

        except Exception:
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_timestamp(self, index: int) -> Optional[float]:
        if self.delta_t is None:
            return None
        return float(index) * float(self.delta_t)

    # ------------------------------------------------------------------

    def _run_analysis_phase(
        self,
    ) -> Optional[Tuple[List[Dict[str, Any]], List[AnalysisResult]]]:
        """Run the analysis phase either serially or across multiple cores."""
        timestamps = [self._compute_timestamp(i) for i in range(len(self.image_paths))]
        worker_count = self._determine_worker_count()
        if worker_count <= 1:
            return self._run_analysis_serial(timestamps)
        return self._run_analysis_parallel(worker_count, timestamps)

    # ------------------------------------------------------------------

    def _determine_worker_count(self) -> int:
        """
        Choose a sensible number of analysis workers.

        The GUI stays responsive by leaving one logical core free when
        possible, while still using multiple processes for CPU-bound work.
        """
        total = len(self.image_paths)
        available = os.cpu_count() or 1
        if total < 2 or available < 2:
            return 1
        return min(total, max(1, available - 1))

    # ------------------------------------------------------------------

    def _run_analysis_serial(
        self,
        timestamps: List[Optional[float]],
    ) -> Optional[Tuple[List[Dict[str, Any]], List[AnalysisResult]]]:
        """Run analysis in the current process."""
        total = len(self.image_paths)
        results_table: List[Dict[str, Any]] = []
        analysis_results: List[AnalysisResult] = []

        for index, image_path in enumerate(self.image_paths):
            if self._cancel_requested:
                self.aborted.emit()
                return None

            _, row, result = _analyze_image_for_batch(
                index=index,
                image_path=image_path,
                settings=self.settings,
                timestamp_s=timestamps[index],
                include_result=self.create_video,
            )

            results_table.append(row)
            if result is not None:
                analysis_results.append(result)

            self.progress_update.emit(index + 1, total)
            self.status_update.emit(f"Analyzing {index + 1}/{total} images...")

        return results_table, analysis_results

    # ------------------------------------------------------------------

    def _run_analysis_parallel(
        self,
        worker_count: int,
        timestamps: List[Optional[float]],
    ) -> Optional[Tuple[List[Dict[str, Any]], List[AnalysisResult]]]:
        """Run analysis across multiple worker processes."""
        total = len(self.image_paths)
        ordered_rows: List[Optional[Dict[str, Any]]] = [None] * total
        ordered_results: List[Optional[AnalysisResult]] = [None] * total if self.create_video else []
        cancelled = False

        executor = ProcessPoolExecutor(max_workers=worker_count)
        try:
            pending = {
                executor.submit(
                    _analyze_image_for_batch,
                    index,
                    image_path,
                    self.settings,
                    timestamps[index],
                    self.create_video,
                ): index
                for index, image_path in enumerate(self.image_paths)
            }

            completed = 0
            while pending:
                if self._cancel_requested:
                    cancelled = True
                    break

                done, _ = wait(
                    list(pending.keys()),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue

                for future in done:
                    pending.pop(future, None)
                    index, row, result = future.result()
                    ordered_rows[index] = row

                    if self.create_video and result is not None:
                        ordered_results[index] = result

                    completed += 1
                    self.progress_update.emit(completed, total)
                    self.status_update.emit(
                        f"Analyzing {completed}/{total} images using {worker_count} workers..."
                    )
        finally:
            executor.shutdown(wait=not cancelled, cancel_futures=cancelled)

        if cancelled:
            self.aborted.emit()
            return None

        results_table = [row for row in ordered_rows if row is not None]
        analysis_results = [result for result in ordered_results if result is not None]
        return results_table, analysis_results

    # ------------------------------------------------------------------

    def _build_plot_config(self, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Build the plot configuration for video rendering.

        This method interprets video_settings and derives:
        - whether plotting is enabled
        - x-axis limits from total batch duration
        - y-axis limits from either:
            * user-defined settings
            * batch-derived valid measurements
        """
        if not self.video_settings.get("show_time_plot", False):
            return None

        plot_height = bool(self.video_settings.get("plot_height", True))
        plot_width = bool(self.video_settings.get("plot_width", True))

        if not plot_height and not plot_width:
            return None

        units = str(self.video_settings.get("plot_units", "px")).lower()
        if units not in {"px", "um"}:
            units = "px"

        axis_mode = str(self.video_settings.get("plot_axis_mode", "separate")).lower()
        if axis_mode not in {"shared", "separate"}:
            axis_mode = "separate"

        scaling_mode = str(self.video_settings.get("plot_scaling_mode", "batch_derived")).lower()
        if scaling_mode not in {"batch_derived", "user_defined"}:
            scaling_mode = "batch_derived"

        if rows:
            if self.delta_t is not None:
                x_max = float(max(0, len(rows) - 1)) * float(self.delta_t)
            else:
                x_max = float(max(0, len(rows) - 1))
        else:
            x_max = 0.0

        config: Dict[str, Any] = {
            "enabled": True,
            "units": units,
            "plot_height": plot_height,
            "plot_width": plot_width,
            "axis_mode": axis_mode,
            "scaling_mode": scaling_mode,
            "x_min": 0.0,
            "x_max": x_max,
            "x_label": "Time [s]" if self.delta_t is not None else "Frame",
        }

        if scaling_mode == "user_defined":
            self._apply_user_defined_plot_limits(config)
        else:
            self._apply_batch_derived_plot_limits(config, rows)

        return config

    # ------------------------------------------------------------------

    def _apply_user_defined_plot_limits(self, config: Dict[str, Any]) -> None:
        """Apply user-defined y-axis limits from video_settings."""
        axis_mode = config["axis_mode"]

        if axis_mode == "shared":
            ymin = float(self.video_settings.get("plot_ymin", 0.0))
            ymax = float(self.video_settings.get("plot_ymax", 1.0))
            if ymax <= ymin:
                ymax = ymin + 1.0
            config["shared_ylim"] = (ymin, ymax)
            return

        height_ymin = float(self.video_settings.get("plot_height_ymin", 0.0))
        height_ymax = float(self.video_settings.get("plot_height_ymax", 1.0))
        width_ymin = float(self.video_settings.get("plot_width_ymin", 0.0))
        width_ymax = float(self.video_settings.get("plot_width_ymax", 1.0))

        if height_ymax <= height_ymin:
            height_ymax = height_ymin + 1.0
        if width_ymax <= width_ymin:
            width_ymax = width_ymin + 1.0

        config["height_ylim"] = (height_ymin, height_ymax)
        config["width_ylim"] = (width_ymin, width_ymax)

    # ------------------------------------------------------------------

    def _apply_batch_derived_plot_limits(
        self,
        config: Dict[str, Any],
        rows: List[Dict[str, Any]],
    ) -> None:
        """Derive stable y-axis limits from valid batch results."""
        units = config["units"]
        axis_mode = config["axis_mode"]

        height_key = "Height [um]" if units == "um" else "Height [px]"
        width_key = "Width [um]" if units == "um" else "Width [px]"

        valid_rows = [row for row in rows if row.get("Valid Frame", True)]

        height_values = self._extract_numeric_values(valid_rows, height_key) if config["plot_height"] else []
        width_values = self._extract_numeric_values(valid_rows, width_key) if config["plot_width"] else []

        if axis_mode == "shared":
            all_values = height_values + width_values
            config["shared_ylim"] = self._limits_from_values(all_values)
            return

        config["height_ylim"] = self._limits_from_values(height_values)
        config["width_ylim"] = self._limits_from_values(width_values)

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numeric_values(rows: List[Dict[str, Any]], key: str) -> List[float]:
        """Extract finite numeric values from result rows."""
        values: List[float] = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if pd.isna(value_f):
                continue
            values.append(value_f)
        return values

    # ------------------------------------------------------------------

    @staticmethod
    def _limits_from_values(values: List[float]) -> tuple[float, float]:
        """
        Build stable y-limits from a list of values.

        Strategy:
        - if all values are positive, anchor at 0
        - add 10% headroom
        - if empty, return a safe fallback
        """
        if not values:
            return (0.0, 1.0)

        vmin = min(values)
        vmax = max(values)

        if vmax == vmin:
            if vmax == 0.0:
                return (0.0, 1.0)
            margin = abs(vmax) * 0.1
            if margin <= 0.0:
                margin = 1.0
            return (vmin - margin, vmax + margin)

        if vmin >= 0.0:
            ymin = 0.0
            ymax = vmax * 1.1
            if ymax <= ymin:
                ymax = ymin + 1.0
            return (ymin, ymax)

        span = vmax - vmin
        margin = span * 0.1
        return (vmin - margin, vmax + margin)

    # ------------------------------------------------------------------

    def _mark_temporal_outliers(
        self,
        rows: List[Dict[str, Any]],
    ) -> None:
        if not self.settings.get("batch_outlier_filter_enabled", False):
            return

        max_h_jump = float(self.settings.get("batch_max_height_jump_pct", 30.0))
        max_w_jump = float(self.settings.get("batch_max_width_jump_pct", 30.0))
        min_sp_ratio = float(self.settings.get("batch_min_supporting_points_ratio", 0.5))

        last_valid = None

        for row in rows:
            row["Valid Frame"] = True
            row["Outlier Reason"] = ""

            if last_valid is None:
                last_valid = row
                continue

            reasons = []

            h_prev = last_valid.get("Height [px]")
            h_cur = row.get("Height [px]")

            if h_prev not in (None, 0) and h_cur is not None:
                jump = abs(h_cur - h_prev) / abs(h_prev) * 100.0
                row["Height Jump [%]"] = jump
                if jump > max_h_jump:
                    reasons.append(f"height jump {jump:.1f}%")

            w_prev = last_valid.get("Width [px]")
            w_cur = row.get("Width [px]")

            if w_prev not in (None, 0) and w_cur is not None:
                jump = abs(w_cur - w_prev) / abs(w_prev) * 100.0
                row["Width Jump [%]"] = jump
                if jump > max_w_jump:
                    reasons.append(f"width jump {jump:.1f}%")

            sp_prev = last_valid.get("Supporting Points")
            sp_cur = row.get("Supporting Points")

            if sp_prev not in (None, 0) and sp_cur is not None:
                ratio = float(sp_cur) / float(sp_prev)
                row["Supporting Points Ratio"] = ratio
                if ratio < min_sp_ratio:
                    reasons.append(f"supporting points ratio {ratio:.2f}")

            if reasons:
                row["Valid Frame"] = False
                row["Outlier Reason"] = " | ".join(reasons)
            else:
                last_valid = row

    # ------------------------------------------------------------------

    def _write_settings_json(self) -> str:
        settings_path = str(
            Path(self.csv_path).with_name(
                Path(self.csv_path).stem + "_settings.json"
            )
        )

        Path(settings_path).parent.mkdir(parents=True, exist_ok=True)

        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=4)

        return settings_path

    # ------------------------------------------------------------------

    def _build_metadata_rows(
        self,
        settings_path: str,
    ) -> List[List[Any]]:
        baseline = self.settings.get("baseline")

        effective_origin_x, effective_origin_y = SettingsManager.get_effective_origin(self.settings)

        return [
            ["Exported", datetime.now().isoformat(timespec="seconds")],
            ["Source Folder", str(Path(self.image_paths[0]).parent)],
            ["Settings File", settings_path],
            ["Baseline", str(baseline)],
            ["Origin X", int(round(effective_origin_x))],
            ["Origin Y", int(round(effective_origin_y))],
            ["Threshold Mode", self.settings.get("threshold_mode")],
            ["Threshold", self.settings.get("threshold")],
            ["Min Delta", self.settings.get("min_delta")],
            ["Number of Rays", self.settings.get("num_rays")],
            ["Angle Span [deg]", self.settings.get("angle_span_deg")],
            ["Smoothness", self.settings.get("smoothness")],
            ["Max Deviation [%]", self.settings.get("max_dev_pct")],
            ["Edge Direction", self.settings.get("edge_direction")],
            ["Auto Origin", self.settings.get("auto_origin")],
            ["Manual Origin", self.settings.get("manual_origin")],
            ["Origin dX", self.settings.get("origin_dx")],
            ["Origin dY", self.settings.get("origin_dy")],
            ["um_per_px", self.settings.get("um_per_px")],
            ["Delta t [s]", self.delta_t],
        ]

    # ------------------------------------------------------------------

    def _write_csv(
        self,
        rows: List[Dict[str, Any]],
        settings_path: str,
    ) -> None:
        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)

        dataframe = pd.DataFrame(rows)
        metadata_rows = self._build_metadata_rows(settings_path)

        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            for key, value in metadata_rows:
                f.write(f"{key};{value}\n")

            f.write("\n")

            dataframe.to_csv(
                f,
                sep=";",
                decimal=",",
                index=False,
                float_format="%.4f",
            )
