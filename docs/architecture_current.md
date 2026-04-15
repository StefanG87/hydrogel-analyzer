# Hydrogel Analyzer: Current Architecture

## Status

This document describes the current implementation in the `hydrogel-analyzer` source tree. It supersedes the older exported architecture documents stored in `Dokumente/`.

The goal here is not to propose a redesign. It is to document the code as it exists today.

## Project Structure

### Entry Point

- `main.py`
  - creates `QApplication`
  - creates `ui.main_window.MainWindow`
  - starts the Qt event loop

`main.py` does not directly create `SettingsManager`.

### Core Layer

- `core/models.py`
  - defines the shared analysis dataclasses
  - current contract: `AnalysisResult -> ContourResult / GeometryResult / AnalysisMetadata`
- `core/settings_defaults.py`
  - defines the flat default settings schema
  - exposes `create_default_settings()`
- `core/settings_manager.py`
  - owns the shared runtime settings dictionary
  - provides JSON import/export
  - computes effective origin from baseline and origin settings

`SettingsManager` is singleton-like by sharing one internal dictionary across instances.

### Logic Layer

- `logic/analyzer.py`
  - central orchestration entry point for one-image analysis
- `logic/baseline.py`
  - baseline validation, normalization, direction vectors, midpoint, projection, signed distances
- `logic/contour_radial.py`
  - radial ray casting, directional thresholding, point cleanup, spline/polyline generation
- `logic/geometry.py`
  - baseline-relative width and height extraction
- `logic/plotting.py`
  - pure rendering of already computed analysis results
- `logic/scale_bar.py`
  - scale bar detection and `um_per_px` conversion

### Tools Layer

- `tools/image_preprocessing.py`
  - grayscale conversion and preprocessing pipeline
- `tools/batch_worker.py`
  - threaded batch execution
  - optional multiprocessing analysis
  - CSV export
  - settings JSON export
  - temporal outlier marking
  - optional video rendering
- `tools/video_renderer.py`
  - MP4 rendering from in-memory `AnalysisResult` objects
  - optional embedded time plot
- `tools/zoom_handler.py`
  - Matplotlib zoom and pan behavior shared by GUI views

### UI Layer

- `ui/main_window.py`
  - main desktop window
  - folder loading
  - single-image analysis display
  - scale calibration
  - batch launch and cancel flow
- `ui/contour_settings_panel.py`
  - combined settings and preview container
  - owns a working-copy settings dictionary
- `ui/contour_settings_dialog.py`
  - settings editor widgets
  - live updates to the working settings dictionary
- `ui/preview_window.py`
  - interactive preview
  - baseline handle dragging
  - live re-analysis and overlay refresh

## Layer Responsibilities

### MainWindow

`MainWindow` is the application shell. It does not implement contour math. It:

- reads current settings from `SettingsManager`
- runs `ContourAnalyzer.process_image()` for the active image
- displays the result through `draw_analysis_result()`
- owns the batch worker lifecycle

### Settings and Preview Flow

The settings path is intentionally split into two states:

- authoritative application state in `SettingsManager`
- temporary working state inside `ContourSettingsPanel`

`ContourSettingsPanel` deep-copies the current manager state into `working_settings`. That dictionary is then shared by:

- `ContourSettingsDialog`
- `PreviewWindow`

This enables live preview updates while keeping the main application state unchanged until the user presses `Apply and Close`.

### Analysis Flow

`ContourAnalyzer.process_image()` is the single analysis entry point used by both GUI and batch workflows. It is responsible for:

1. baseline normalization or default creation
2. effective origin computation
3. preprocessing
4. radial contour detection
5. geometry extraction
6. metadata assembly
7. `AnalysisResult` creation

### Batch Flow

`BatchWorker` implements a two-phase pipeline:

1. analysis phase
   - serial or multi-process execution
   - one `AnalysisResult` per image
   - one export row per image
2. export/render phase
   - temporal outlier marking
   - settings JSON export
   - CSV export
   - optional annotated MP4 rendering

Video rendering is part of the batch workflow. It does not rerun analysis. It consumes the stored in-memory `AnalysisResult` objects from phase 1.

## Runtime Flows

### Application Startup

1. `main.py` creates `QApplication`.
2. `MainWindow` creates its UI and a `SettingsManager` instance.
3. `MainWindow` pulls a copy of the shared settings with `get_all()`.

### Main Window Analysis

1. User loads an image folder.
2. `MainWindow` stores sorted image paths and optional `delta_t_seconds`.
3. The current image is read with `skimage.io.imread`.
4. `ContourAnalyzer.process_image()` returns a structured result.
5. `logic.plotting.draw_analysis_result()` draws the overlay.

### Settings Panel and Preview

1. `MainWindow.open_settings_panel()` creates `ContourSettingsPanel`.
2. The panel builds `working_settings` from a deep copy of the manager state.
3. `ContourSettingsDialog` writes widget changes directly into `working_settings`.
4. `settings_changed` triggers `PreviewWindow.refresh_analysis()`.
5. `PreviewWindow` reruns the analyzer with the current working settings.
6. `Apply and Close` commits the final working settings to `SettingsManager`.

### Batch Export and Video

1. `MainWindow.run_batch_analysis()` collects export paths and optional video settings.
2. `BatchWorker` runs in a background `QThread`.
3. Batch analysis runs serially or through `ProcessPoolExecutor`.
4. Result rows are marked for temporal outliers after raw analysis completes.
5. CSV and settings JSON are written.
6. If enabled, `VideoRenderer` renders frames using the stored analysis results and shared plotting path.

## Data Contracts

### Settings Dictionary

The current settings model is a flat dictionary defined in `core/settings_defaults.py`.

Major groups:

- detection settings
- baseline and origin settings
- preprocessing settings
- overlay/display settings
- ROI compatibility settings
- calibration settings
- batch outlier settings

This model is practical and easy to serialize, but still loosely typed.

### AnalysisResult

The main result contract is structured, not flat:

```text
AnalysisResult
├── contour: ContourResult
│   ├── supporting_points
│   ├── rays
│   ├── spline_x
│   └── spline_y
├── geometry: GeometryResult | None
│   ├── height_px / width_px
│   ├── height_um / width_um
│   └── reference_vectors
├── baseline
├── origin
├── metadata: AnalysisMetadata | None
└── debug
```

This structure is consumed by:

- `ui/main_window.py`
- `ui/preview_window.py`
- `logic/plotting.py`
- `tools/batch_worker.py`
- `tools/video_renderer.py`

## Current Analysis Pipeline

### Baseline and Origin

- baseline comes from settings if valid
- otherwise a centered default baseline is created
- origin is computed by `SettingsManager.get_effective_origin()`
  - baseline midpoint if `auto_origin=True`
  - optional relative offsets if `manual_origin=True`
  - explicit absolute coordinates otherwise

### Preprocessing

`tools/image_preprocessing.py` currently supports:

- grayscale conversion
- linear contrast and brightness adjustment
- optional CLAHE
- optional Gaussian blur
- optional median blur
- optional inversion
- optional Otsu binarization

### Contour Detection

`logic/contour_radial.py`:

- casts rays from the effective origin
- clips them against image bounds and optional baseline
- computes directional edge deltas
- supports threshold modes:
  - `manual`
  - `auto_percentile`
  - `mean_std`
  - `otsu`
- filters out large neighbor-distance deviations
- sorts and cleans accepted supporting points
- outputs either a polyline or a smoothed spline for display

### Geometry Extraction

`logic/geometry.py` measures:

- width parallel to the baseline direction
- height orthogonal to the baseline direction

It also builds `reference_vectors` for rendering width and height annotations without recomputing geometry inside the plotting layer.

## Preview and Overlay Behavior

`PreviewWindow` has two important runtime behaviors:

- the baseline can be dragged interactively through Matplotlib handle circles
- during handle dragging, geometry annotation is temporarily suppressed until analysis is recomputed on mouse release

It also validates whether an existing baseline still fits the current image size and regenerates a default one if needed.

## Batch Export Details

`BatchWorker` writes:

- a CSV export
- a sibling `*_settings.json`

The CSV includes:

- metadata header rows
- one data row per image
- optional timestamps
- pixel and optional micron measurements
- supporting-point counts
- outlier flags and outlier reasons

Temporal outlier marking currently checks:

- height jump percentage
- width jump percentage
- supporting-points ratio

## Video Rendering Details

`VideoRenderer`:

- receives raw image frames plus `AnalysisResult`
- calls `draw_analysis_result()` for each frame
- optionally draws timestamp text
- optionally draws an embedded time plot
- supports shared or separate y-axis plot modes

The current time-plot path uses fixed plot limits derived in `BatchWorker`, not per-frame auto-scaling.

## Settings Model and State Boundaries

### Shared Runtime State

`SettingsManager` keeps one shared internal dictionary:

- all instances point to the same underlying state
- `get_all()` returns a shallow copy
- `update()` only accepts keys known to `DEFAULT_SETTINGS`

### Working Copy State

`ContourSettingsPanel` intentionally breaks away from the shared state while open:

- deep copy current settings
- modify working copy live
- preview against working copy
- only commit on `Apply and Close`

This is the current and intentional source of truth for the preview/settings workflow.

## Deviations from the Older Architecture Document

The older architecture export should be considered outdated in the following areas:

1. `main.py` only starts `QApplication` and `MainWindow`.
   It does not directly create `SettingsManager`.
2. `SettingsManager` is shared-state based, not a purely stateless helper.
3. Analysis output is structured through dataclasses, not a flat result object.
4. Video rendering is part of the batch pipeline and consumes stored analysis results.
5. The settings/preview workflow uses a working copy that is only committed on `Apply and Close`.
6. The current batch flow includes optional multiprocessing, temporal outlier marking, settings JSON export, CSV export, and optional MP4 rendering.

## Focused Refactor Notes

These are the most grounded next-step targets implied by the current code:

### High Priority

1. Unify baseline and origin validation paths.
   `SettingsManager.get_effective_origin()` still duplicates part of baseline validation logic from `logic/baseline.py`.
2. Tighten settings schema handling.
   The current flat dictionary is still practical, but it remains fragile as the project grows.
3. Clarify vertical-baseline behavior end-to-end.
   Helper functions support vertical baselines in principle, but UI and rendering behavior still need explicit validation.
4. Improve release and documentation hygiene before larger geometry refactors.

### Medium Priority

1. Rename `Dokumente/` to `docs/` once the repository layout is confirmed and updated.
   No readable code references to `Dokumente/` were found during this documentation pass.
2. Separate export formatting more clearly from batch execution if the export path grows further.
3. Add example settings files, screenshots, and a pinned dependency file.
4. Remove minor debug residue such as the scale-bar detection `print()` in `logic/scale_bar.py`.
