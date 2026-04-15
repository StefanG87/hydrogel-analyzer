# ui/contour_settings_panel.py

"""
ui/contour_settings_panel.py

Provides the combined settings + preview dialog.

This panel combines:

- the contour settings editor on the left
- the interactive preview window on the right
- shared apply / cancel logic
- optional save / load of settings JSON files

The panel owns one shared mutable settings dictionary that is used by both
the settings editor and the preview.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
)

from core.settings_manager import SettingsManager
from ui.contour_settings_dialog import ContourSettingsDialog
from ui.preview_window import PreviewWindow


class ContourSettingsPanel(QDialog):
    """
    Combined live settings and preview dialog.

    The panel edits a working copy of the current settings. Only pressing
    "Apply and Close" writes the final settings back to the SettingsManager.
    """

    def __init__(self, parent=None, image_paths: Optional[List[str]] = None):
        super().__init__(parent)

        self.setWindowTitle("Contour Settings")
        self.setMinimumSize(1300, 750)

        self.settings_manager = SettingsManager()
        self.working_settings = copy.deepcopy(self.settings_manager.get_all())

        image_folder = None
        if image_paths:
            import os
            image_folder = os.path.dirname(image_paths[0])

        self.settings_dialog = ContourSettingsDialog(self, settings=self.working_settings)
        self.preview_window = PreviewWindow(self, image_folder=image_folder, settings=self.working_settings)

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the panel layout."""
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.settings_dialog)
        splitter.addWidget(self.preview_window)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self.reset_button = QPushButton("Reset")
        self.save_button = QPushButton("Save Settings")
        self.load_button = QPushButton("Load Settings")
        self.apply_button = QPushButton("Apply and Close")
        self.cancel_button = QPushButton("Cancel")

        button_row = QHBoxLayout()
        button_row.addWidget(self.reset_button)
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.load_button)
        button_row.addStretch()
        button_row.addWidget(self.apply_button)
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        layout.addLayout(button_row)
        self.setLayout(layout)

    def _connect_signals(self) -> None:
        """Connect all panel-level signals."""
        self.settings_dialog.settings_changed.connect(self.preview_window.refresh_analysis)

        self.reset_button.clicked.connect(self.reset_settings)
        self.save_button.clicked.connect(self.save_settings)
        self.load_button.clicked.connect(self.load_settings)
        self.apply_button.clicked.connect(self.apply_and_close)
        self.cancel_button.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def reset_settings(self) -> None:
        """Reset working settings to application defaults."""
        self.working_settings.clear()
        self.working_settings.update(self.settings_manager.default_settings())
        self.settings_dialog.update_ui_from_settings()
        self.preview_window.ensure_valid_baseline()
        self.preview_window.refresh_analysis()

    def save_settings(self) -> None:
        """Save the current working settings to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return

        temp_manager = SettingsManager()
        original = temp_manager.get_all()
        try:
            temp_manager.reset()
            temp_manager.update(self.working_settings)
            temp_manager.save_to_json(path)
        finally:
            temp_manager.reset()
            temp_manager.update(original)

    def load_settings(self) -> None:
        """Load settings from a JSON file into the working settings."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Settings",
            "",
            "JSON Files (*.json)",
        )
        if not path:
            return

        temp_manager = SettingsManager()
        original = temp_manager.get_all()

        try:
            temp_manager.reset()
            temp_manager.load_from_json(path)
            loaded = temp_manager.get_all()
        finally:
            temp_manager.reset()
            temp_manager.update(original)

        self.working_settings.clear()
        self.working_settings.update(loaded)
        self.settings_dialog.update_ui_from_settings()
        self.preview_window.ensure_valid_baseline()
        self.preview_window.refresh_analysis()

    def apply_and_close(self) -> None:
        """Commit the working settings to the shared SettingsManager."""
        self.preview_window.update_baseline_from_handles()
        self.settings_manager.update(self.working_settings)
        self.accept()