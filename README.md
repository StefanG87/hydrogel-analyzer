# Hydrogel Analyzer

Hydrogel Analyzer is a modular PyQt5-based desktop application for contour detection and geometry analysis in hydrogel image series. The current implementation combines interactive baseline editing, radial contour detection, baseline-aware height and width extraction, batch CSV export, automatic scale calibration, and optional annotated video rendering.

## Overview

The application is organized around a small set of focused modules:

- `main.py` starts `QApplication`, creates `MainWindow`, and enters the Qt event loop.
- `ui/` contains the desktop GUI, including the main window, the combined settings and preview panel, and the live preview window.
- `logic/` contains baseline utilities, radial contour detection, geometry extraction, plotting, and scale-bar calibration.
- `tools/` contains preprocessing, threaded batch execution, video rendering, and zoom/pan helpers.
- `core/` contains shared dataclasses and the centralized settings layer.

The current architecture reference is [docs/architecture_current.md](docs/architecture_current.md). The exported files in `Dokumente/` are older snapshots and should be treated as archived background material rather than the current source of truth.

## Features

- Interactive desktop GUI for image-folder based analysis
- Live contour preview with draggable baseline handles
- Radial ray-casting contour detection with configurable threshold modes
- Baseline-aware height and width extraction
- Optional unit conversion through `um_per_px`
- Automatic scale-bar calibration from the current image
- Shared settings workflow between preview and final application state
- Batch CSV export with metadata header and sibling settings JSON export
- Temporal outlier marking for batch results
- Optional annotated MP4 rendering
- Optional embedded time-plot rendering in exported videos

## Project Structure

```text
hydrogel analyzer - codex/
├── core/
│   ├── models.py
│   ├── settings_defaults.py
│   └── settings_manager.py
├── logic/
│   ├── analyzer.py
│   ├── baseline.py
│   ├── contour_radial.py
│   ├── geometry.py
│   ├── plotting.py
│   └── scale_bar.py
├── tools/
│   ├── batch_worker.py
│   ├── image_preprocessing.py
│   ├── video_renderer.py
│   └── zoom_handler.py
├── ui/
│   ├── contour_settings_dialog.py
│   ├── contour_settings_panel.py
│   ├── main_window.py
│   └── preview_window.py
├── Dokumente/
│   └── archived architecture exports
├── docs/
│   ├── architecture_current.md
│   └── github_metadata.md
└── main.py
```

## Installation

Python 3.10+ is recommended because the codebase uses modern type syntax such as `str | Path`.

The repository currently does not ship a pinned dependency file, so install the runtime dependencies manually:

```bash
pip install PyQt5 matplotlib numpy pandas scipy scikit-image opencv-python
```

Depending on your platform and environment, you may also want a virtual environment before installation.

## Running the Application

Start the GUI from the project root:

```bash
python main.py
```

Typical workflow:

1. Open a folder that contains image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`).
2. Review the current image in the main window.
3. Open `Contour Settings` to edit detection, preprocessing, origin, and overlay parameters.
4. Use the live preview to adjust the baseline and inspect the current result.
5. Apply the working settings back to the main application with `Apply and Close`.
6. Run batch export when the current configuration is stable.

## Analysis Pipeline

The runtime analysis path is centered in `logic/analyzer.py`:

1. Normalize or create a baseline.
2. Compute the effective origin through `SettingsManager.get_effective_origin()`.
3. Preprocess the image through `tools/image_preprocessing.py`.
4. Run radial contour detection through `logic/contour_radial.py`.
5. Extract baseline-relative geometry through `logic/geometry.py`.
6. Return a structured `AnalysisResult`.

The returned result is not flat. It is composed of:

- `ContourResult`
- `GeometryResult`
- `AnalysisMetadata`
- optional debug payloads

This structure is reused by preview rendering, the main window, batch export, and video rendering.

## Batch Export

Batch processing runs in `tools/batch_worker.py` as a two-phase workflow:

1. Analysis phase:
   - process images serially or through `ProcessPoolExecutor`
   - collect structured `AnalysisResult` objects
   - build export rows for CSV output
2. Export/render phase:
   - mark temporal outliers
   - write a sibling settings JSON file
   - write the CSV export
   - optionally render an MP4 using the stored in-memory results

Current batch behavior:

- CSV export uses `;` as separator and `,` as decimal marker.
- A metadata header block is written before the tabular data.
- A sibling `*_settings.json` file is exported alongside the CSV.
- Optional video export reuses `logic.plotting.draw_analysis_result()` so overlays stay visually aligned with the GUI.
- If filenames match `TL_<n>s`, the batch path derives a time step from the filename for timestamps and video plots.

## Known Limitations

- Settings are still represented as raw dictionaries across many modules.
- No packaged installer or pinned dependency file is included yet.
- Vertical baseline handling is supported by baseline helpers in principle, but broader end-to-end behavior should still be validated.
- `SettingsManager.get_effective_origin()` still duplicates baseline validation logic that partly overlaps with `logic/baseline.py`.
- The repository still contains archived documentation under `Dokumente/`; the current source-of-truth architecture document now lives in `docs/`.

## Roadmap

- Unify baseline and origin validation into one shared utility path.
- Tighten settings schema handling without breaking the current UI workflow.
- Clarify vertical-baseline behavior across UI, geometry, and overlay rendering.
- Add a pinned dependency file and example project assets.
- Improve GitHub-facing metadata, release tagging, and public documentation hygiene.
