# main.py

"""
main.py

Application entry point for the Hydrogel Analyzer project.

This module is responsible for:

- creating the Qt application instance
- creating and showing the main window
- starting the Qt event loop

No analysis logic belongs in this file.
"""

from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    """
    Create and run the Hydrogel Analyzer application.

    Returns:
        int: Qt application exit code.
    """
    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(1200, 900)
    window.show()

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())