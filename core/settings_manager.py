# core/settings_manager.py

"""
core/settings_manager.py

Centralized settings access layer for Hydrogel Analyzer.

Responsibilities:

- manage runtime settings dictionary
- provide validated defaults from settings_defaults.py
- support JSON import/export
- compute derived values (effective origin)
- keep compatibility with GUI + batch mode + analyzer pipeline

Design requirement:

DEFAULT_SETTINGS from settings_defaults.py is the single
source of truth for all initial values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.settings_defaults import (
    DEFAULT_SETTINGS,
    create_default_settings,
)


class SettingsManager:
    """
    Central manager for application-wide analysis settings.

    Singleton-like behavior:
    multiple instances share the same internal settings dictionary.
    """

    _shared_settings: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        if SettingsManager._shared_settings is None:
            SettingsManager._shared_settings = create_default_settings()

        self._settings = SettingsManager._shared_settings

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def default_settings() -> Dict[str, Any]:
        """
        Return a fresh default settings dictionary.
        """
        return create_default_settings()

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key in DEFAULT_SETTINGS:
            self._settings[key] = value

    def update(self, new_settings: Dict[str, Any]) -> None:
        """
        Update settings using only known schema keys.
        """
        for key, value in new_settings.items():
            if key in DEFAULT_SETTINGS:
                self._settings[key] = value

    def get_all(self) -> Dict[str, Any]:
        """
        Return a copy of current settings.
        """
        return dict(self._settings)

    def reset(self) -> None:
        """
        Reset settings to DEFAULT_SETTINGS.
        """
        self._settings.clear()
        self._settings.update(create_default_settings())

    # ------------------------------------------------------------------
    # JSON import/export
    # ------------------------------------------------------------------

    def save_to_json(self, path: str | Path) -> None:
        """
        Save settings to JSON.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, indent=4)

    def load_from_json(self, path: str | Path) -> None:
        """
        Load settings from JSON file.

        Unknown keys are ignored.
        """
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        for key, value in loaded.items():
            if key in DEFAULT_SETTINGS:
                self._settings[key] = value

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------

    @staticmethod
    def get_effective_origin(settings: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute the effective origin from settings dictionary.

        Logic:

        auto_origin=True:
            use baseline midpoint

        manual_origin=True:
            apply origin_dx / origin_dy relative to midpoint

        otherwise:
            use origin_x / origin_y
        """

        baseline = settings.get("baseline")
        auto_origin = bool(settings.get("auto_origin", True))
        manual_origin = bool(settings.get("manual_origin", False))

        valid_baseline = None

        if isinstance(baseline, (list, tuple)) and len(baseline) == 2:
            p1, p2 = baseline

            if (
                isinstance(p1, (list, tuple)) and len(p1) == 2 and
                isinstance(p2, (list, tuple)) and len(p2) == 2
            ):
                try:
                    x1, y1 = int(p1[0]), int(p1[1])
                    x2, y2 = int(p2[0]), int(p2[1])

                    if not (x1 == x2 and y1 == y2):
                        valid_baseline = [(x1, y1), (x2, y2)]

                except (TypeError, ValueError):
                    pass

        if auto_origin and valid_baseline is not None:

            (x1, y1), (x2, y2) = valid_baseline

            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0

            if manual_origin:
                dx = float(settings.get("origin_dx", 0))
                dy = float(settings.get("origin_dy", 0))

                # positive dy moves origin upward
                return mid_x + dx, mid_y - dy

            return mid_x, mid_y

        return (
            float(settings.get("origin_x", 0)),
            float(settings.get("origin_y", 0)),
        )