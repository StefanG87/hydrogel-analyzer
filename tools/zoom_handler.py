# tools/zoom_handler.py

"""
tools/zoom_handler.py

Provides interactive zoom and pan handling for a Matplotlib canvas embedded in Qt.

This class is responsible for:

- mouse wheel zooming
- right mouse button panning
- restoring the previous zoom state
- resetting the view to the full image extent

This module must not contain analysis logic or UI-specific business logic.
"""

from __future__ import annotations

from typing import Optional, Tuple

from matplotlib.axes import Axes


class ZoomPanHandler:
    """
    Handle zoom and pan interaction on a Matplotlib axis.

    Interaction model:
    - mouse wheel: zoom in / out around cursor position
    - right mouse button drag: pan
    - reset_zoom(): restore full image view
    - restore_zoom(): restore last stored user zoom if available
    """

    def __init__(self, ax: Axes, canvas, zoom_base: float = 1.2):
        """
        Initialize the zoom and pan handler.

        Args:
            ax: Matplotlib axis to control.
            canvas: Matplotlib canvas.
            zoom_base: Zoom factor per wheel step.
        """
        self.ax = ax
        self.canvas = canvas
        self.zoom_base = float(zoom_base)

        self.default_xlim: Optional[Tuple[float, float]] = None
        self.default_ylim: Optional[Tuple[float, float]] = None

        self.zoom_xlim: Optional[Tuple[float, float]] = None
        self.zoom_ylim: Optional[Tuple[float, float]] = None

        self._pan_start_data: Optional[Tuple[float, float]] = None
        self._pan_start_xlim: Optional[Tuple[float, float]] = None
        self._pan_start_ylim: Optional[Tuple[float, float]] = None

        self._connect_events()

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------

    def _connect_events(self) -> None:
        """Connect all Matplotlib interaction events."""
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("draw_event", self.on_draw)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def on_draw(self, event=None) -> None:
        """
        Store the current axis state.

        The first valid draw is also used as the default full-view state.
        """
        current_xlim = tuple(self.ax.get_xlim())
        current_ylim = tuple(self.ax.get_ylim())

        if self.default_xlim is None or self.default_ylim is None:
            self.default_xlim = current_xlim
            self.default_ylim = current_ylim

        self.zoom_xlim = current_xlim
        self.zoom_ylim = current_ylim

    def restore_zoom(self) -> None:
        """
        Restore the last stored zoom state.

        If no zoom state is available yet, restore the default full view.
        """
        if self.zoom_xlim is not None and self.zoom_ylim is not None:
            self.ax.set_xlim(self.zoom_xlim)
            self.ax.set_ylim(self.zoom_ylim)
        elif self.default_xlim is not None and self.default_ylim is not None:
            self.ax.set_xlim(self.default_xlim)
            self.ax.set_ylim(self.default_ylim)

    def reset_zoom(self) -> None:
        """
        Reset the axis to the default full-image view.
        """
        if self.default_xlim is None or self.default_ylim is None:
            return

        self.ax.set_xlim(self.default_xlim)
        self.ax.set_ylim(self.default_ylim)

        self.zoom_xlim = self.default_xlim
        self.zoom_ylim = self.default_ylim

        self.canvas.draw_idle()

    def update_default_view_from_current_axis(self) -> None:
        """
        Explicitly overwrite the default full-view state with the current axis limits.

        This is useful after loading a new image and drawing it for the first time.
        """
        self.default_xlim = tuple(self.ax.get_xlim())
        self.default_ylim = tuple(self.ax.get_ylim())

        if self.zoom_xlim is None or self.zoom_ylim is None:
            self.zoom_xlim = self.default_xlim
            self.zoom_ylim = self.default_ylim

    # ------------------------------------------------------------------
    # Zoom interaction
    # ------------------------------------------------------------------

    def on_scroll(self, event) -> None:
        """
        Zoom around the cursor position.

        Args:
            event: Matplotlib scroll event.
        """
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        xdata = float(event.xdata)
        ydata = float(event.ydata)

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        if event.button == "up":
            scale_factor = 1.0 / self.zoom_base
        elif event.button == "down":
            scale_factor = self.zoom_base
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        rel_x = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]) if cur_xlim[1] != cur_xlim[0] else 0.5
        rel_y = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]) if cur_ylim[1] != cur_ylim[0] else 0.5

        new_xlim = (
            xdata - rel_x * new_width,
            xdata + (1.0 - rel_x) * new_width,
        )
        new_ylim = (
            ydata - rel_y * new_height,
            ydata + (1.0 - rel_y) * new_height,
        )

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        self.zoom_xlim = tuple(new_xlim)
        self.zoom_ylim = tuple(new_ylim)

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Pan interaction
    # ------------------------------------------------------------------

    def on_button_press(self, event) -> None:
        """
        Start panning on right mouse button press.

        Args:
            event: Matplotlib mouse press event.
        """
        if event.inaxes != self.ax:
            return

        if event.button != 3:
            return

        if event.xdata is None or event.ydata is None:
            return

        self._pan_start_data = (float(event.xdata), float(event.ydata))
        self._pan_start_xlim = tuple(self.ax.get_xlim())
        self._pan_start_ylim = tuple(self.ax.get_ylim())

    def on_mouse_move(self, event) -> None:
        """
        Pan the current view while dragging with the right mouse button.

        Args:
            event: Matplotlib mouse move event.
        """
        if self._pan_start_data is None:
            return

        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        if self._pan_start_xlim is None or self._pan_start_ylim is None:
            return

        start_x, start_y = self._pan_start_data
        current_x = float(event.xdata)
        current_y = float(event.ydata)

        dx = current_x - start_x
        dy = current_y - start_y

        new_xlim = (
            self._pan_start_xlim[0] - dx,
            self._pan_start_xlim[1] - dx,
        )
        new_ylim = (
            self._pan_start_ylim[0] - dy,
            self._pan_start_ylim[1] - dy,
        )

        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)

        self.zoom_xlim = tuple(new_xlim)
        self.zoom_ylim = tuple(new_ylim)

        self.canvas.draw_idle()

    def on_button_release(self, event) -> None:
        """
        Finish panning.

        Args:
            event: Matplotlib mouse release event.
        """
        self._pan_start_data = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None