# ui/contour_settings_dialog.py

"""
ui/contour_settings_dialog.py

Provides the parameter editing dialog for contour analysis settings.

This dialog is responsible for:

- exposing structured analysis parameters to the user
- updating a shared settings dictionary
- emitting change signals for live preview refresh
- keeping UI widgets synchronized with the current settings

This dialog does not run analysis itself.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ContourSettingsDialog(QDialog):
    """
    Settings editor operating directly on a shared settings dictionary.

    The dialog emits settings_changed whenever a parameter is updated,
    allowing the preview window to trigger live analysis refresh.
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent=None, settings: Dict[str, Any] | None = None):
        super().__init__(parent)

        if settings is None:
            raise ValueError("ContourSettingsDialog requires a shared settings dictionary.")

        self.setWindowTitle("Contour Settings")
        self.setMinimumWidth(420)

        self.settings = settings
        self.controls: Dict[str, Tuple[QWidget, QWidget | None]] = {}

        self._build_ui()
        self.update_ui_from_settings()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the complete dialog layout."""
        layout = QVBoxLayout()

        layout.addWidget(self._create_detection_group())
        layout.addWidget(self._create_origin_group())
        layout.addWidget(self._create_preprocessing_group())
        layout.addWidget(self._create_display_group())
        layout.addStretch()

        self.setLayout(layout)

    def _create_detection_group(self) -> QGroupBox:
        """Create the contour detection parameter group."""
        group = QGroupBox("Detection")
        form = QFormLayout()

        self._add_int_slider_row(form, "threshold", "Threshold", 0, 255)
        self._add_int_slider_row(form, "min_delta", "Min Delta", 0, 100)
        self._add_int_slider_row(form, "num_rays", "Number of Rays", 10, 360)
        self._add_int_slider_row(form, "angle_span_deg", "Angle Span (°)", 10, 360)
        self._add_int_slider_row(form, "smoothness", "Smoothness", 0, 50)
        self._add_int_slider_row(form, "max_dev_pct", "Max Deviation (%)", 0, 100)

        threshold_mode = QComboBox()
        threshold_mode.addItems(["manual", "auto_percentile", "mean_std", "otsu"])
        threshold_mode.currentTextChanged.connect(
            lambda value: self._set_setting("threshold_mode", value)
        )
        form.addRow(QLabel("Threshold Mode"), threshold_mode)
        self.controls["threshold_mode"] = (threshold_mode, None)

        edge_direction = QComboBox()
        edge_direction.addItems(["dark_to_bright", "bright_to_dark", "both"])
        edge_direction.currentTextChanged.connect(
            lambda value: self._set_setting("edge_direction", value)
        )
        form.addRow(QLabel("Edge Direction"), edge_direction)
        self.controls["edge_direction"] = (edge_direction, None)

        curve_mode = QComboBox()
        curve_mode.addItems(["polyline", "spline"])
        curve_mode.currentTextChanged.connect(
            lambda value: self._set_setting("curve_mode", value)
        )
        form.addRow(QLabel("Curve Mode"), curve_mode)
        self.controls["curve_mode"] = (curve_mode, None)

        group.setLayout(form)
        return group

    def _create_origin_group(self) -> QGroupBox:
        """Create the origin control group."""
        group = QGroupBox("Origin")
        form = QFormLayout()

        auto_origin_cb = QCheckBox("Use Baseline Midpoint")
        auto_origin_cb.stateChanged.connect(
            lambda state: self._set_setting("auto_origin", bool(state))
        )
        form.addRow(auto_origin_cb)
        self.controls["auto_origin"] = (auto_origin_cb, None)

        manual_origin_cb = QCheckBox("Enable Relative Offset")
        manual_origin_cb.stateChanged.connect(self._on_manual_origin_changed)
        form.addRow(manual_origin_cb)
        self.controls["manual_origin"] = (manual_origin_cb, None)

        self._add_int_slider_row(form, "origin_dx", "Offset X", -1000, 1000)
        self._add_int_slider_row(form, "origin_dy", "Offset Y", -1000, 1000)

        group.setLayout(form)
        return group

    def _create_preprocessing_group(self) -> QGroupBox:
        """Create the preprocessing parameter group."""
        group = QGroupBox("Preprocessing")
        form = QFormLayout()

        preprocess_cb = QCheckBox("Enable Preprocessing")
        preprocess_cb.stateChanged.connect(
            lambda state: self._set_setting("preprocess_enabled", bool(state))
        )
        form.addRow(preprocess_cb)
        self.controls["preprocess_enabled"] = (preprocess_cb, None)

        show_pre_cb = QCheckBox("Show Preprocessed Image")
        show_pre_cb.stateChanged.connect(
            lambda state: self._set_setting("show_preprocessing", bool(state))
        )
        form.addRow(show_pre_cb)
        self.controls["show_preprocessing"] = (show_pre_cb, None)

        clahe_cb = QCheckBox("Apply CLAHE")
        clahe_cb.stateChanged.connect(lambda state: self._set_setting("clahe", bool(state)))
        form.addRow(clahe_cb)
        self.controls["clahe"] = (clahe_cb, None)

        median_cb = QCheckBox("Apply Median Blur")
        median_cb.stateChanged.connect(
            lambda state: self._set_setting("median_blur", bool(state))
        )
        form.addRow(median_cb)
        self.controls["median_blur"] = (median_cb, None)

        invert_cb = QCheckBox("Invert Image")
        invert_cb.stateChanged.connect(lambda state: self._set_setting("invert", bool(state)))
        form.addRow(invert_cb)
        self.controls["invert"] = (invert_cb, None)

        binarize_cb = QCheckBox("Binarize Image")
        binarize_cb.stateChanged.connect(
            lambda state: self._set_setting("binarize", bool(state))
        )
        form.addRow(binarize_cb)
        self.controls["binarize"] = (binarize_cb, None)

        self._add_int_slider_row(form, "contrast_factor", "Contrast (%)", 10, 300)
        self._add_int_slider_row(form, "brightness_offset", "Brightness Offset", -100, 100)
        self._add_float_slider_row(form, "blur_sigma", "Gaussian Sigma", 0.0, 20.0, decimals=1)

        group.setLayout(form)
        return group

    def _create_display_group(self) -> QGroupBox:
        """Create the display control group."""
        group = QGroupBox("Display")
        form = QFormLayout()

        for key, label in [
            ("show_rays", "Show Rays"),
            ("show_supporting_points", "Show Supporting Points"),
            ("draw_spline", "Draw Spline"),
            ("draw_baseline", "Draw Baseline"),
            ("show_origin", "Show Origin"),
            ("show_geometry", "Show Height / Width"),
        ]:
            checkbox = QCheckBox(label)
            checkbox.stateChanged.connect(
                lambda state, setting_key=key: self._set_setting(setting_key, bool(state))
            )
            form.addRow(checkbox)
            self.controls[key] = (checkbox, None)

        group.setLayout(form)
        return group

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    def _add_int_slider_row(
        self,
        form: QFormLayout,
        key: str,
        label_text: str,
        minimum: int,
        maximum: int,
    ) -> None:
        """Add a linked integer slider and spinbox row."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)

        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)

        slider.valueChanged.connect(spinbox.setValue)
        spinbox.valueChanged.connect(slider.setValue)
        spinbox.valueChanged.connect(lambda value, setting_key=key: self._set_setting(setting_key, value))

        row.addWidget(slider)
        row.addWidget(spinbox)

        form.addRow(QLabel(label_text), container)
        self.controls[key] = (slider, spinbox)

    def _add_float_slider_row(
        self,
        form: QFormLayout,
        key: str,
        label_text: str,
        minimum: float,
        maximum: float,
        decimals: int = 1,
    ) -> None:
        """
        Add a float spinbox row.

        A slider is intentionally not used here because float sliders in Qt
        require manual scaling and add unnecessary complexity.
        """
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(0.1)
        spinbox.valueChanged.connect(lambda value, setting_key=key: self._set_setting(setting_key, float(value)))

        form.addRow(QLabel(label_text), spinbox)
        self.controls[key] = (spinbox, None)

    # ------------------------------------------------------------------
    # Internal state updates
    # ------------------------------------------------------------------

    def _set_setting(self, key: str, value: Any) -> None:
        """Store a setting update and emit the live update signal."""
        self.settings[key] = value
        self.settings_changed.emit()
        self._update_origin_control_state()

    def _on_manual_origin_changed(self, state: int) -> None:
        """Handle changes to the manual origin toggle."""
        self._set_setting("manual_origin", bool(state))

    def _update_origin_control_state(self) -> None:
        """Enable or disable origin offset controls based on current settings."""
        manual_enabled = bool(self.settings.get("manual_origin", False))
        for key in ("origin_dx", "origin_dy"):
            widget, linked = self.controls[key]
            widget.setEnabled(manual_enabled)
            if linked is not None:
                linked.setEnabled(manual_enabled)

    # ------------------------------------------------------------------
    # Public synchronization API
    # ------------------------------------------------------------------

    def update_ui_from_settings(self) -> None:
        """Synchronize all widgets from the current settings dictionary."""
        for key, (widget, linked) in self.controls.items():
            value = self.settings.get(key)

            if isinstance(widget, QSlider) and linked is not None:
                widget.blockSignals(True)
                linked.blockSignals(True)
                widget.setValue(int(value))
                linked.setValue(int(value))
                widget.blockSignals(False)
                linked.blockSignals(False)

            elif isinstance(widget, QSpinBox):
                widget.blockSignals(True)
                widget.setValue(int(value))
                widget.blockSignals(False)

            elif isinstance(widget, QDoubleSpinBox):
                widget.blockSignals(True)
                widget.setValue(float(value))
                widget.blockSignals(False)

            elif isinstance(widget, QComboBox):
                widget.blockSignals(True)
                index = widget.findText(str(value))
                if index >= 0:
                    widget.setCurrentIndex(index)
                widget.blockSignals(False)

            elif isinstance(widget, QCheckBox):
                widget.blockSignals(True)
                widget.setChecked(bool(value))
                widget.blockSignals(False)

        self._update_origin_control_state()