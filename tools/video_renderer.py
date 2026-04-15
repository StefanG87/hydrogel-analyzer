# tools/video_renderer.py

"""
tools/video_renderer.py

Provides video rendering for Hydrogel Analyzer batch processing.

This module is responsible for:

- rendering analysis overlays into video frames
- reusing the plotting pipeline for consistent visualization
- optionally embedding a live time plot
- writing MP4 video output with OpenCV

This module must not perform contour analysis itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from core.models import AnalysisResult
from logic.plotting import draw_analysis_result


class VideoRenderer:
    """
    Render annotated analysis frames into a video file.

    The renderer uses the same plotting pipeline as the interactive GUI
    so that preview, main window, and video export stay visually consistent.
    """

    def __init__(
        self,
        output_path: str,
        frame_size: Tuple[int, int],
        fps: float = 24.0,
    ):
        self.output_path = str(output_path)
        self.frame_width = int(frame_size[0])
        self.frame_height = int(frame_size[1])
        self.fps = float(fps)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for: {self.output_path}")

        self.fig = Figure(
            figsize=(self.frame_width / 100.0, self.frame_height / 100.0),
            dpi=100,
        )
        self.canvas = FigureCanvasAgg(self.fig)

        self.ax_image = None
        self.ax_plot = None
        self.ax_plot_right = None
        self._plot_layout_enabled: Optional[bool] = None
        self._aligned_plot_image_shape: Optional[Tuple[int, int]] = None
        self._history_x: List[float] = []
        self._history_height: List[float] = []
        self._history_width: List[float] = []

    # ------------------------------------------------------------------
    # Public rendering API
    # ------------------------------------------------------------------

    def add_frame(
        self,
        image: np.ndarray,
        result: AnalysisResult,
        settings: Dict[str, Any],
        timestamp_s: Optional[float] = None,
        plot_config: Optional[Dict[str, Any]] = None,
        frame_index: int = 0,
    ) -> None:
        """
        Render one annotated video frame.

        Processing order:
        1. prepare the figure layout for the current frame
        2. draw the image overlay
        3. optionally draw the time plot
        4. align the plot width to the visible image width
        5. rasterize the figure and write it to the video
        """
        plot_enabled = bool(plot_config and plot_config.get("enabled", False))

        self._prepare_axes(plot_enabled=plot_enabled)

        draw_analysis_result(
            ax=self.ax_image,
            image=image,
            result=result,
            settings=settings,
        )

        if settings.get("show_timestamp", True) and timestamp_s is not None:
            self.ax_image.text(
                10,
                25,
                f"t = {timestamp_s:.2f} s",
                color="white",
                fontsize=10,
                backgroundcolor="black",
            )

        if plot_enabled and self.ax_plot is not None:
            self._draw_time_plot(
                result=result,
                plot_config=plot_config,
                timestamp_s=timestamp_s,
                frame_index=frame_index,
            )
            self._match_plot_width_to_image_axis()

        self.canvas.draw()

        width, height = self.canvas.get_width_height()
        buffer = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
        frame_rgba = buffer.reshape((height, width, 4))
        frame_bgr = frame_rgba[:, :, 2::-1].copy()

        if frame_bgr.shape[1] != self.frame_width or frame_bgr.shape[0] != self.frame_height:
            frame_bgr = cv2.resize(frame_bgr, (self.frame_width, self.frame_height))

        self.writer.write(frame_bgr)

    def finish(self) -> None:
        """Release writer and figure resources."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if hasattr(self, "fig") and self.fig is not None:
            self.fig.clear()
            self.fig = None
            self.canvas = None
            self.ax_image = None
            self.ax_plot = None
            self.ax_plot_right = None
            self._aligned_plot_image_shape = None
            self._history_x = []
            self._history_height = []
            self._history_width = []

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _prepare_axes(self, plot_enabled: bool) -> None:
        """
        Prepare the figure axes for the current frame layout.

        Axes are only rebuilt when the high-level layout changes. This avoids
        repeated subplot construction during long video renders.
        """
        if (
            self._plot_layout_enabled == plot_enabled
            and self.ax_image is not None
            and (not plot_enabled or self.ax_plot is not None)
        ):
            return

        self.fig.clear()
        self.ax_plot_right = None
        self._aligned_plot_image_shape = None

        if plot_enabled:
            gs = self.fig.add_gridspec(
                nrows=2,
                ncols=1,
                height_ratios=[4.0, 1.6],
                hspace=0.18,
            )
            self.ax_image = self.fig.add_subplot(gs[0, 0])
            self.ax_plot = self.fig.add_subplot(gs[1, 0])
        else:
            self.ax_image = self.fig.add_subplot(111)
            self.ax_plot = None

        self.ax_image.set_axis_off()
        self._plot_layout_enabled = plot_enabled

    def _match_plot_width_to_image_axis(self) -> None:
        """
        Align the plot width to the actually visible image width.

        The alignment only needs to be recomputed when the image shape or
        overall subplot layout changes.
        """
        if self.ax_image is None or self.ax_plot is None:
            return

        if not self.ax_image.images:
            return

        image = self.ax_image.images[0].get_array()
        img_h, img_w = image.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return

        current_shape = (int(img_h), int(img_w))
        if self._aligned_plot_image_shape == current_shape:
            return

        image_pos = self.ax_image.get_position()
        plot_pos = self.ax_plot.get_position()

        fig_w_in, fig_h_in = self.fig.get_size_inches()
        axis_w_in = image_pos.width * fig_w_in
        axis_h_in = image_pos.height * fig_h_in

        if axis_w_in <= 0 or axis_h_in <= 0:
            return

        image_aspect = float(img_w) / float(img_h)
        axis_aspect = axis_w_in / axis_h_in

        if axis_aspect >= image_aspect:
            visible_h = image_pos.height
            visible_w = visible_h * (fig_h_in / fig_w_in) * image_aspect
            visible_x0 = image_pos.x0 + (image_pos.width - visible_w) / 2.0
        else:
            visible_w = image_pos.width
            visible_x0 = image_pos.x0

        new_position = [visible_x0, plot_pos.y0, visible_w, plot_pos.height]
        self.ax_plot.set_position(new_position)

        if self.ax_plot_right is not None:
            self.ax_plot_right.set_position(new_position)

        self._aligned_plot_image_shape = current_shape

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_time_plot(
        self,
        result: AnalysisResult,
        plot_config: Dict[str, Any],
        timestamp_s: Optional[float],
        frame_index: int,
    ) -> None:
        """
        Draw the live time plot below the image.

        The plot uses fixed axis limits from BatchWorker and only draws the
        trace up to the current frame index.
        """
        ax = self.ax_plot
        if ax is None:
            return

        ax.clear()
        ax.set_facecolor("white")
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.tick_params(axis="y", right=False, labelright=False)
        ax.spines["right"].set_visible(False)

        axis_mode = str(plot_config.get("axis_mode", "separate")).lower()
        units = str(plot_config.get("units", "px"))
        plot_height = bool(plot_config.get("plot_height", True))
        plot_width = bool(plot_config.get("plot_width", True))

        self._append_history(
            x_value=float(timestamp_s) if timestamp_s is not None else float(frame_index),
            result=result,
            units=units,
        )

        x_hist = self._history_x[: frame_index + 1]
        height_hist = self._history_height[: frame_index + 1] if plot_height else []
        width_hist = self._history_width[: frame_index + 1] if plot_width else []

        ax.set_xlim(
            float(plot_config.get("x_min", 0.0)),
            float(plot_config.get("x_max", 1.0)),
        )
        ax.set_xlabel(str(plot_config.get("x_label", "Time [s]")))
        ax.grid(True, alpha=0.25)

        if axis_mode == "shared":
            if self.ax_plot_right is not None:
                self.ax_plot_right.set_visible(False)

            shared_ylim = plot_config.get("shared_ylim", (0.0, 1.0))
            ax.set_ylim(shared_ylim)

            ylabel = "Value"
            if plot_height and not plot_width:
                ylabel = f"Height [{units}]"
            elif plot_width and not plot_height:
                ylabel = f"Width [{units}]"
            else:
                ylabel = f"Height / Width [{units}]"

            ax.set_ylabel(ylabel)
            ax.tick_params(axis="y", left=True, labelleft=True)
            ax.spines["left"].set_visible(True)

            if plot_height and height_hist:
                ax.plot(
                    x_hist,
                    height_hist,
                    linewidth=1.8,
                    color="tab:blue",
                    label=f"Height [{units}]",
                )
                ax.scatter([x_hist[-1]], [height_hist[-1]], s=18, color="tab:blue")

            if plot_width and width_hist:
                ax.plot(
                    x_hist,
                    width_hist,
                    linewidth=1.8,
                    color="tab:orange",
                    label=f"Width [{units}]",
                )
                ax.scatter([x_hist[-1]], [width_hist[-1]], s=18, color="tab:orange")

            ax.legend(loc="upper left", fontsize=8)

        else:
            if plot_width:
                if self.ax_plot_right is None:
                    self.ax_plot_right = ax.twinx()
                else:
                    self.ax_plot_right.clear()
                    self.ax_plot_right.set_visible(True)

                self.ax_plot_right.yaxis.set_label_position("right")
                self.ax_plot_right.yaxis.tick_right()
                self.ax_plot_right.spines["left"].set_visible(False)
                self.ax_plot_right.spines["right"].set_visible(True)
                self.ax_plot_right.tick_params(axis="y", left=False, labelleft=False)
            elif self.ax_plot_right is not None:
                self.ax_plot_right.set_visible(False)

            if plot_height:
                height_ylim = plot_config.get("height_ylim", (0.0, 1.0))
                ax.set_ylim(height_ylim)
                ax.set_ylabel(f"Height [{units}]")
                ax.yaxis.label.set_color("tab:blue")
                ax.tick_params(axis="y", left=True, labelleft=True, colors="tab:blue")
                ax.spines["left"].set_visible(True)

                if height_hist:
                    ax.plot(x_hist, height_hist, linewidth=1.8, color="tab:blue")
                    ax.scatter([x_hist[-1]], [height_hist[-1]], s=18, color="tab:blue")
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

            if plot_width and self.ax_plot_right is not None:
                width_ylim = plot_config.get("width_ylim", (0.0, 1.0))
                self.ax_plot_right.set_ylim(width_ylim)
                self.ax_plot_right.set_ylabel(f"Width [{units}]")
                self.ax_plot_right.yaxis.label.set_color("tab:orange")
                self.ax_plot_right.tick_params(axis="y", right=True, labelright=True, colors="tab:orange")

                if width_hist:
                    self.ax_plot_right.plot(
                        x_hist,
                        width_hist,
                        linewidth=1.8,
                        color="tab:orange",
                    )
                    self.ax_plot_right.scatter(
                        [x_hist[-1]],
                        [width_hist[-1]],
                        s=18,
                        color="tab:orange",
                    )
            elif self.ax_plot_right is not None:
                self.ax_plot_right.set_ylabel("")

            if self.ax_plot_right is not None:
                self.ax_plot_right.set_xlim(
                    float(plot_config.get("x_min", 0.0)),
                    float(plot_config.get("x_max", 1.0)),
                )

        current_x = x_hist[-1] if x_hist else (timestamp_s if timestamp_s is not None else float(frame_index))
        ax.axvline(current_x, linewidth=1.0)

    def _append_history(
        self,
        x_value: float,
        result: AnalysisResult,
        units: str,
    ) -> None:
        """Append current frame values to persistent plot history."""
        self._history_x.append(float(x_value))
        self._history_height.append(self._extract_metric_value(result, "height", units))
        self._history_width.append(self._extract_metric_value(result, "width", units))

    @staticmethod
    def _extract_metric_value(
        result: AnalysisResult,
        metric: str,
        units: str,
    ) -> float:
        """Extract one scalar metric from AnalysisResult."""
        geometry = result.geometry
        if geometry is None:
            return np.nan

        units = str(units).lower()

        if metric == "height":
            value = geometry.height_um if units == "um" else geometry.height_px
        else:
            value = geometry.width_um if units == "um" else geometry.width_px

        if value is None:
            return np.nan

        return float(value)
