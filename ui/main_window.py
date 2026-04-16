# ui/main_window.py

"""
ui/main_window.py

Provides the main application window for the Hydrogel Analyzer project.

This window is responsible for:

- loading image folders
- browsing through images
- opening the live contour settings panel
- running the current analysis on the active image
- showing measurement summaries
- computing scale calibration
- starting and cancelling batch processing
- optionally exporting annotated videos

This file intentionally avoids implementing contour or geometry logic.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import io

from core.settings_manager import SettingsManager
from logic.analyzer import ContourAnalyzer
from logic.plotting import draw_analysis_result
from logic.scale_bar import calculate_um_per_px, find_scale_bar_px
from tools.batch_worker import BatchWorker
from tools.zoom_handler import ZoomPanHandler
from ui.contour_settings_panel import ContourSettingsPanel


class MainWindow(QWidget):
    """
    Main application window for interactive single-image analysis.

    The window uses the global SettingsManager as the authoritative settings source.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hydrogel Analyzer – Clean Rebuild")

        self.settings_manager = SettingsManager()
        self.analysis_settings: Dict[str, Any] = self.settings_manager.get_all()

        self.image_paths: List[str] = []
        self.current_index: int = 0
        self.image = None
        self.analysis_result = None

        self.delta_t_seconds: Optional[int] = None
        self.um_per_px: Optional[float] = self.analysis_settings.get("um_per_px")

        self.batch_thread: Optional[BatchWorker] = None
        self._reset_zoom_on_next_draw: bool = True

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the complete main window layout."""
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()

        self.zoom_handler = ZoomPanHandler(self.ax, self.canvas)

        self.toolbar = QToolBar()

        self.open_button = QPushButton("Open Folder")
        self.settings_button = QPushButton("Contour Settings")
        self.batch_button = QPushButton("Run Batch")
        self.cancel_batch_button = QPushButton("Cancel Batch")
        self.cancel_batch_button.setEnabled(False)

        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")

        self.umpx_label = QLabel("µm/px: -")
        self.progress_label = QLabel("")
        self.height_label = QLabel("Height: -")
        self.width_label = QLabel("Width: -")

        self.mm_input = QLineEdit()
        self.mm_input.setPlaceholderText("Scale bar length (mm)")
        self.calc_scale_button = QPushButton("Calc µm/px")

        self.show_spline_checkbox = QCheckBox("Show Spline")
        self.show_spline_checkbox.setChecked(True)

        self.show_baseline_checkbox = QCheckBox("Show Baseline")
        self.show_baseline_checkbox.setChecked(True)

        self.show_geometry_checkbox = QCheckBox("Show Geometry")
        self.show_geometry_checkbox.setChecked(True)

        self.show_origin_checkbox = QCheckBox("Show Origin")
        self.show_origin_checkbox.setChecked(False)

        top_controls = QHBoxLayout()
        top_controls.addWidget(self.open_button)
        top_controls.addWidget(self.settings_button)
        top_controls.addWidget(self.batch_button)
        top_controls.addWidget(self.cancel_batch_button)
        top_controls.addStretch()

        nav_controls = QHBoxLayout()
        nav_controls.addWidget(self.prev_button)
        nav_controls.addWidget(self.next_button)
        nav_controls.addStretch()

        scale_controls = QHBoxLayout()
        scale_controls.addWidget(self.mm_input)
        scale_controls.addWidget(self.calc_scale_button)

        display_controls = QHBoxLayout()
        display_controls.addWidget(self.umpx_label)
        display_controls.addWidget(self.show_spline_checkbox)
        display_controls.addWidget(self.show_baseline_checkbox)
        display_controls.addWidget(self.show_geometry_checkbox)
        display_controls.addWidget(self.show_origin_checkbox)
        display_controls.addStretch()

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addLayout(top_controls)
        layout.addWidget(self.canvas)
        layout.addWidget(self.progress_label)
        layout.addLayout(nav_controls)
        layout.addLayout(scale_controls)
        layout.addLayout(display_controls)
        layout.addWidget(self.height_label)
        layout.addWidget(self.width_label)

        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect all UI signals."""
        self.open_button.clicked.connect(self.load_folder)
        self.settings_button.clicked.connect(self.open_settings_panel)
        self.batch_button.clicked.connect(self.run_batch_analysis)
        self.cancel_batch_button.clicked.connect(self.cancel_batch_analysis)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        self.calc_scale_button.clicked.connect(self.calculate_scale)

        self.show_spline_checkbox.stateChanged.connect(self.update_display_flags)
        self.show_baseline_checkbox.stateChanged.connect(self.update_display_flags)
        self.show_geometry_checkbox.stateChanged.connect(self.update_display_flags)
        self.show_origin_checkbox.stateChanged.connect(self.update_display_flags)

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    def load_folder(self) -> None:
        """Open a folder chooser and load all supported images."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        self.image_paths = sorted(
            [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ],
            key=os.path.getmtime,
        )

        if not self.image_paths:
            QMessageBox.warning(
                self,
                "No Images",
                "No supported images were found in the selected folder.",
            )
            self.image = None
            self.analysis_result = None
            self.update_display()
            return

        self.current_index = 0
        self._parse_delta_t()
        self.load_current_image()

    def _parse_delta_t(self) -> None:
        """Parse an optional time step from the first filename."""
        self.delta_t_seconds = None
        if not self.image_paths:
            return

        first_name = os.path.basename(self.image_paths[0])
        match = re.search(r"TL_(\d+)s", first_name)
        if match:
            self.delta_t_seconds = int(match.group(1))

    def load_current_image(self) -> None:
        """Load the current image and immediately run the analysis."""
        if not self.image_paths:
            self.image = None
            self.analysis_result = None
            self.update_display()
            return

        self.image = io.imread(self.image_paths[self.current_index])
        self._reset_zoom_on_next_draw = True
        self.run_analysis()

    def show_previous_image(self) -> None:
        """Show the previous image."""
        if not self.image_paths:
            return

        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_current_image()

    def show_next_image(self) -> None:
        """Show the next image."""
        if not self.image_paths:
            return

        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_current_image()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def run_analysis(self) -> None:
        """Run the analyzer on the current image."""
        if self.image is None:
            self.analysis_result = None
            self.update_display()
            return

        self.analysis_settings = self.settings_manager.get_all()

        if self.um_per_px is not None:
            self.analysis_settings["um_per_px"] = self.um_per_px

        self.analysis_result = ContourAnalyzer.process_image(
            image=self.image,
            settings=self.analysis_settings,
            image_path=self.image_paths[self.current_index] if self.image_paths else None,
            include_debug_images=False,
        )

        self.update_display()

    def update_display(self) -> None:
        """Redraw the current analysis result in the main view."""
        if self.image is None or self.analysis_result is None:
            self.canvas.draw_idle()
            self.height_label.setText("Height: -")
            self.width_label.setText("Width: -")
            return

        display_settings = self._build_display_settings()

        draw_analysis_result(
            ax=self.ax,
            image=self.image,
            result=self.analysis_result,
            settings=display_settings,
        )

        self.canvas.draw()

        if self._reset_zoom_on_next_draw:
            self.zoom_handler.update_default_view_from_current_axis()
            self.zoom_handler.reset_zoom()
            self._reset_zoom_on_next_draw = False
        else:
            self.zoom_handler.restore_zoom()
            self.canvas.draw_idle()

        self._update_measurement_labels()

    def _build_display_settings(self) -> Dict[str, Any]:
        """
        Build the display settings used only for the main window overlay.

        The main window intentionally keeps the display simpler than the preview.
        """
        display_settings = dict(self.analysis_settings)

        display_settings["show_rays"] = False
        display_settings["show_supporting_points"] = False
        display_settings["draw_spline"] = self.show_spline_checkbox.isChecked()
        display_settings["draw_baseline"] = self.show_baseline_checkbox.isChecked()
        display_settings["show_geometry"] = self.show_geometry_checkbox.isChecked()
        display_settings["show_origin"] = self.show_origin_checkbox.isChecked()

        return display_settings

    def _update_measurement_labels(self) -> None:
        """Update the height and width labels."""
        geometry = self.analysis_result.geometry if self.analysis_result is not None else None

        if geometry is None:
            self.height_label.setText("Height: -")
            self.width_label.setText("Width: -")
            return

        if not self.show_geometry_checkbox.isChecked():
            self.height_label.setText("")
            self.width_label.setText("")
            return

        if geometry.height_um is not None and geometry.width_um is not None:
            self.height_label.setText(f"Height: {geometry.height_um:.1f} µm")
            self.width_label.setText(f"Width: {geometry.width_um:.1f} µm")
        elif geometry.height_px is not None and geometry.width_px is not None:
            self.height_label.setText(f"Height: {geometry.height_px:.1f} px")
            self.width_label.setText(f"Width: {geometry.width_px:.1f} px")
        else:
            self.height_label.setText("Height: -")
            self.width_label.setText("Width: -")

    def update_display_flags(self) -> None:
        """Refresh the current overlay after display toggle changes."""
        self.update_display()

    # ------------------------------------------------------------------
    # Settings panel
    # ------------------------------------------------------------------

    def open_settings_panel(self) -> None:
        """Open the combined settings and preview dialog."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load a folder first.")
            return

        panel = ContourSettingsPanel(self, image_paths=self.image_paths)
        if panel.exec_():
            self.analysis_settings = self.settings_manager.get_all()
            self.um_per_px = self.analysis_settings.get("um_per_px")
            self.run_analysis()

    # ------------------------------------------------------------------
    # Scale calibration
    # ------------------------------------------------------------------

    def calculate_scale(self) -> None:
        """Detect a scale bar and compute µm per pixel."""
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            known_length_mm = float(self.mm_input.text())
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please enter a valid scale bar length in mm.",
            )
            return

        try:
            scale_bar_px, success, _ = find_scale_bar_px(self.image)
            if not success or scale_bar_px <= 0:
                QMessageBox.warning(
                    self,
                    "Detection Failed",
                    "Could not detect a scale bar.",
                )
                return

            self.um_per_px = calculate_um_per_px(scale_bar_px, known_length_mm)
            self.settings_manager.set("um_per_px", self.um_per_px)
            self.analysis_settings["um_per_px"] = self.um_per_px
            self.umpx_label.setText(f"µm/px: {self.um_per_px:.4f}")

            self.run_analysis()

        except Exception as exc:
            QMessageBox.critical(self, "Scale Error", str(exc))

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_batch_analysis(self) -> None:
        """Start batch processing in a background thread."""
        if not self.image_paths:
            QMessageBox.warning(self, "No Images", "Please load a folder first.")
            return

        current_settings = self.settings_manager.get_all()
        if self.um_per_px is not None:
            current_settings["um_per_px"] = self.um_per_px

        default_csv_path = str(
            Path(self.image_paths[0]).with_name(
                Path(self.image_paths[0]).parent.name + "_analysis.csv"
            )
        )

        csv_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Batch CSV",
            default_csv_path,
            "CSV Files (*.csv)",
        )
        if not csv_path:
            return

        create_video = False
        video_path = ""
        video_settings: Dict[str, Any] = {}

        reply = QMessageBox.question(
            self,
            "Create Video",
            "Do you want to create an annotated video as well?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            create_video = True

            selected_video_settings = self._get_video_annotation_settings()
            if selected_video_settings is None:
                return

            video_settings = selected_video_settings

            suggested_video_path = str(Path(csv_path).with_suffix(".mp4"))
            selected_video_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Batch Video",
                suggested_video_path,
                "MP4 Video (*.mp4)",
            )
            if not selected_video_path:
                return

            video_path = selected_video_path

        self._set_batch_ui_running(True)
        self.progress_label.setText("Starting batch analysis...")

        self.batch_thread = BatchWorker(
            image_paths=self.image_paths,
            settings=current_settings,
            csv_path=csv_path,
            delta_t=self.delta_t_seconds,
            create_video=create_video,
            video_path=video_path if create_video else None,
            fps=24.0,
            video_settings=video_settings if create_video else None,
        )

        self.batch_thread.progress_update.connect(self.update_batch_progress)
        self.batch_thread.status_update.connect(self.update_batch_status)
        self.batch_thread.finished.connect(self.batch_analysis_done)
        self.batch_thread.aborted.connect(self.batch_analysis_aborted)
        self.batch_thread.failed.connect(self.batch_analysis_failed)
        self.batch_thread.start()

    # ------------------------------------------------------------------
    # Video annotation dialogs
    # ------------------------------------------------------------------

    def _get_video_annotation_settings(self) -> Optional[Dict[str, Any]]:
        """
        Show a dialog that lets the user choose which annotations
        should appear in the exported video.

        Returns:
            dict | None:
                Selected video display settings, or None if cancelled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Video Annotations")
        dialog.setMinimumWidth(320)

        show_spline_cb = QCheckBox("Show Spline")
        show_spline_cb.setChecked(True)

        show_baseline_cb = QCheckBox("Show Baseline")
        show_baseline_cb.setChecked(True)

        show_geometry_cb = QCheckBox("Show Geometry")
        show_geometry_cb.setChecked(True)

        show_origin_cb = QCheckBox("Show Origin")
        show_origin_cb.setChecked(False)

        show_rays_cb = QCheckBox("Show Rays")
        show_rays_cb.setChecked(False)

        show_supporting_points_cb = QCheckBox("Show Supporting Points")
        show_supporting_points_cb.setChecked(False)

        show_timestamp_cb = QCheckBox("Show Timestamp")
        show_timestamp_cb.setChecked(True)

        show_time_plot_cb = QCheckBox("Plot Diagram")
        show_time_plot_cb.setChecked(False)

        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(show_spline_cb)
        layout.addWidget(show_baseline_cb)
        layout.addWidget(show_geometry_cb)
        layout.addWidget(show_origin_cb)
        layout.addWidget(show_rays_cb)
        layout.addWidget(show_supporting_points_cb)
        layout.addWidget(show_timestamp_cb)
        layout.addWidget(show_time_plot_cb)
        layout.addSpacing(10)
        layout.addLayout(button_row)

        dialog.setLayout(layout)

        if dialog.exec_() != QDialog.Accepted:
            return None

        settings: Dict[str, Any] = {
            "show_rays": show_rays_cb.isChecked(),
            "show_supporting_points": show_supporting_points_cb.isChecked(),
            "draw_spline": show_spline_cb.isChecked(),
            "draw_baseline": show_baseline_cb.isChecked(),
            "show_geometry": show_geometry_cb.isChecked(),
            "show_origin": show_origin_cb.isChecked(),
            "show_timestamp": show_timestamp_cb.isChecked(),
            "show_time_plot": False,
        }

        if show_time_plot_cb.isChecked():
            plot_settings = self._get_video_plot_settings()
            if plot_settings is None:
                return None
            settings.update(plot_settings)

        return settings

    def _get_video_plot_settings(self) -> Optional[Dict[str, Any]]:
        """
        Show a dialog for configuring the optional time plot
        that can be embedded into the exported video.

        Returns:
            dict | None:
                Plot settings dictionary, or None if cancelled.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Video Plot Settings")
        dialog.setMinimumWidth(360)

        plot_height_cb = QCheckBox("Plot Height")
        plot_height_cb.setChecked(True)

        plot_width_cb = QCheckBox("Plot Width")
        plot_width_cb.setChecked(True)

        units_combo = QComboBox()
        units_combo.addItems(["px", "um"])
        units_combo.setCurrentText("um")

        scaling_combo = QComboBox()
        scaling_combo.addItems(["batch_derived", "user_defined"])
        scaling_combo.setCurrentText("batch_derived")

        axis_mode_combo = QComboBox()
        axis_mode_combo.addItems(["separate", "shared"])
        axis_mode_combo.setCurrentText("separate")

        shared_ymin_spin = self._create_plot_spinbox(0.0)
        shared_ymax_spin = self._create_plot_spinbox(1000.0)

        height_ymin_spin = self._create_plot_spinbox(0.0)
        height_ymax_spin = self._create_plot_spinbox(500.0)

        width_ymin_spin = self._create_plot_spinbox(0.0)
        width_ymax_spin = self._create_plot_spinbox(1500.0)

        def update_visibility() -> None:
            user_defined = scaling_combo.currentText() == "user_defined"
            shared_axis = axis_mode_combo.currentText() == "shared"

            shared_ymin_spin.setEnabled(user_defined and shared_axis)
            shared_ymax_spin.setEnabled(user_defined and shared_axis)

            height_ymin_spin.setEnabled(user_defined and not shared_axis)
            height_ymax_spin.setEnabled(user_defined and not shared_axis)
            width_ymin_spin.setEnabled(user_defined and not shared_axis)
            width_ymax_spin.setEnabled(user_defined and not shared_axis)

        scaling_combo.currentTextChanged.connect(lambda _: update_visibility())
        axis_mode_combo.currentTextChanged.connect(lambda _: update_visibility())
        update_visibility()

        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        layout = QVBoxLayout()
        layout.addWidget(plot_height_cb)
        layout.addWidget(plot_width_cb)

        layout.addWidget(QLabel("Units"))
        layout.addWidget(units_combo)

        layout.addWidget(QLabel("Scaling Mode"))
        layout.addWidget(scaling_combo)

        layout.addWidget(QLabel("Axis Mode"))
        layout.addWidget(axis_mode_combo)

        layout.addWidget(QLabel("Shared Y Min"))
        layout.addWidget(shared_ymin_spin)
        layout.addWidget(QLabel("Shared Y Max"))
        layout.addWidget(shared_ymax_spin)

        layout.addWidget(QLabel("Height Y Min"))
        layout.addWidget(height_ymin_spin)
        layout.addWidget(QLabel("Height Y Max"))
        layout.addWidget(height_ymax_spin)

        layout.addWidget(QLabel("Width Y Min"))
        layout.addWidget(width_ymin_spin)
        layout.addWidget(QLabel("Width Y Max"))
        layout.addWidget(width_ymax_spin)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(ok_button)
        button_row.addWidget(cancel_button)

        layout.addSpacing(10)
        layout.addLayout(button_row)
        dialog.setLayout(layout)

        if dialog.exec_() != QDialog.Accepted:
            return None

        if not plot_height_cb.isChecked() and not plot_width_cb.isChecked():
            QMessageBox.warning(
                self,
                "Invalid Plot Selection",
                "Please enable at least one of: Plot Height or Plot Width.",
            )
            return None

        settings: Dict[str, Any] = {
            "show_time_plot": True,
            "plot_height": plot_height_cb.isChecked(),
            "plot_width": plot_width_cb.isChecked(),
            "plot_units": units_combo.currentText(),
            "plot_scaling_mode": scaling_combo.currentText(),
            "plot_axis_mode": axis_mode_combo.currentText(),
        }

        if scaling_combo.currentText() == "user_defined":
            if axis_mode_combo.currentText() == "shared":
                settings["plot_ymin"] = float(shared_ymin_spin.value())
                settings["plot_ymax"] = float(shared_ymax_spin.value())
            else:
                settings["plot_height_ymin"] = float(height_ymin_spin.value())
                settings["plot_height_ymax"] = float(height_ymax_spin.value())
                settings["plot_width_ymin"] = float(width_ymin_spin.value())
                settings["plot_width_ymax"] = float(width_ymax_spin.value())

        return settings

    @staticmethod
    def _create_plot_spinbox(default_value: float) -> QDoubleSpinBox:
        """Create a consistently configured floating-point spinbox for plot limits."""
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-1_000_000.0, 1_000_000.0)
        spin.setValue(default_value)
        return spin

    # ------------------------------------------------------------------
    # Batch callbacks
    # ------------------------------------------------------------------

    def cancel_batch_analysis(self) -> None:
        """Request graceful cancellation of the running batch process."""
        if self.batch_thread is None:
            return

        if self.batch_thread.isRunning():
            self.batch_thread.request_cancel()
            self.progress_label.setText("Cancel requested...")
            self.cancel_batch_button.setEnabled(False)

    def update_batch_progress(self, current: int, total: int) -> None:
        """Update the batch progress label."""
        self.progress_label.setText(f"Processing {current}/{total} images...")

    def batch_analysis_done(self, csv_path: str, video_path: str) -> None:
        """Handle successful batch completion."""
        self._set_batch_ui_running(False)
        self.batch_thread = None

        if video_path:
            self.progress_label.setText(
                f"Batch complete. CSV saved to: {csv_path} | Video saved to: {video_path}"
            )
        else:
            self.progress_label.setText(f"Batch complete. CSV saved to: {csv_path}")

        QMessageBox.information(
            self,
            "Batch Finished",
            "Batch analysis completed successfully.",
        )

    def batch_analysis_aborted(self) -> None:
        """Handle batch cancellation."""
        self._set_batch_ui_running(False)
        self.batch_thread = None
        self.progress_label.setText("Batch analysis was cancelled.")

    def batch_analysis_failed(self, message: str) -> None:
        """Handle an unexpected batch failure without leaving the UI disabled."""
        self._set_batch_ui_running(False)
        self.batch_thread = None
        self.progress_label.setText(f"Batch analysis failed: {message}")

        QMessageBox.critical(
            self,
            "Batch Failed",
            f"Batch analysis failed:\n\n{message}",
        )

    def _set_batch_ui_running(self, is_running: bool) -> None:
        """
        Enable or disable relevant UI controls during batch processing.
        """
        self.open_button.setEnabled(not is_running)
        self.settings_button.setEnabled(not is_running)
        self.batch_button.setEnabled(not is_running)

        self.prev_button.setEnabled(not is_running)
        self.next_button.setEnabled(not is_running)
        self.calc_scale_button.setEnabled(not is_running)

        self.cancel_batch_button.setEnabled(is_running)

    def update_batch_status(self, message: str) -> None:
        """Update the batch status label with a descriptive message."""
        self.progress_label.setText(message)
