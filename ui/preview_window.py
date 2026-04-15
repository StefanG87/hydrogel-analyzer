# ui/preview_window.py

"""
ui/preview_window.py

Provides the interactive preview window for contour analysis.

This window is responsible for:

- displaying the current image
- showing and updating the baseline interactively
- triggering live analysis updates
- drawing overlays through the plotting module
- preserving zoom state through the zoom handler

This window does not implement contour mathematics itself.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import copy

from PyQt5.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from skimage import io

from logic.analyzer import ContourAnalyzer
from logic.baseline import create_default_baseline, normalize_baseline
from logic.plotting import draw_analysis_result
from tools.image_preprocessing import preprocess_image
from tools.zoom_handler import ZoomPanHandler


class PreviewWindow(QDialog):
    """
    Interactive analysis preview for one image sequence folder.

    The preview uses a shared mutable settings dictionary so that all parameter
    changes immediately affect the analysis without extra synchronization layers.
    """

    def __init__(
        self,
        parent=None,
        image_folder: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)

        if settings is None:
            raise ValueError("PreviewWindow requires a shared settings dictionary.")

        self.setWindowTitle("Contour Preview")
        self.setMinimumSize(900, 650)

        self.settings = settings
        self.image_folder = image_folder
        self.image_paths: List[str] = self._collect_images(image_folder)
        self.current_index = 0

        self.imaga = None
        self.analysis_result = None

        self.handle_radius = 10
        self.handle1: Optional[Circle] = None
        self.handle2: Optional[Circle] = None
        self.dragging_handle: Optional[Circle] = None

        self._build_ui()
        self._connect_events()

        if self.image_paths:
            self.load_image()
            self.ensure_valid_baseline()
            self.run_analysis()
            self.update_display()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the preview canvas and navigation controls."""
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_axis_off()

        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.reset_zoom_button = QPushButton("Reset View")

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.reset_zoom_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(nav_layout)
        self.setLayout(layout)

        self.zoom_handler = ZoomPanHandler(self.ax, self.canvas)

    def _connect_events(self) -> None:
        """Connect UI and Matplotlib events."""
        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.reset_zoom_button.clicked.connect(self.zoom_handler.reset_zoom)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_images(folder: Optional[str]) -> List[str]:
        """Collect image paths from a folder."""
        if not folder or not os.path.isdir(folder):
            return []

        paths = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        return sorted(paths, key=os.path.getmtime)

    def load_image(self) -> None:
        """Load the currently selected image."""
        if not self.image_paths:
            self.image = None
            return

        self.image = io.imread(self.image_paths[self.current_index])

    def show_previous_image(self) -> None:
        """Switch to the previous image."""
        if not self.image_paths:
            return

        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self.load_image()
        self.ensure_valid_baseline()
        self.run_analysis()
        self.update_display()

    def show_next_image(self) -> None:
        """Switch to the next image."""
        if not self.image_paths:
            return

        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self.load_image()
        self.ensure_valid_baseline()
        self.run_analysis()
        self.update_display()

    # ------------------------------------------------------------------
    # Baseline handling
    # ------------------------------------------------------------------

    def ensure_valid_baseline(self) -> None:
        """Ensure that a valid baseline exists for the current image."""
        baseline = normalize_baseline(self.settings.get("baseline"))

        if baseline is None and self.image is not None:
            height, width = self.image.shape[:2]
            baseline = create_default_baseline(width, height)
            self.settings["baseline"] = baseline

        if baseline is None:
            return

        (x1, y1), (x2, y2) = baseline
        self.handle1 = Circle((x1, y1), radius=self.handle_radius, color="red", picker=True)
        self.handle2 = Circle((x2, y2), radius=self.handle_radius, color="blue", picker=True)

    def update_baseline_from_handles(self) -> None:
        """Write the current handle positions back into the shared settings."""
        if self.handle1 is None or self.handle2 is None:
            return

        baseline = [
            (int(round(self.handle1.center[0])), int(round(self.handle1.center[1]))),
            (int(round(self.handle2.center[0])), int(round(self.handle2.center[1]))),
        ]
        self.settings["baseline"] = baseline

    def _draw_handles(self) -> None:
        """Draw interactive baseline handles on top of the overlay."""
        if self.handle1 is None or self.handle2 is None:
            return

        self.ax.add_patch(Circle(self.handle1.center, radius=self.handle_radius, color="red"))
        self.ax.add_patch(Circle(self.handle2.center, radius=self.handle_radius, color="blue"))

    # ------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------