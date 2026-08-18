# Architecture

## Current Entry Points

- `Stackelberg_Main.py`: compatibility CLI. It now launches the package
  single-window Pygame/OpenGL app.
- `phd_3d_animator/app.py`: single-window Pygame application shell and event
  loop.
- `phd_3d_animator/data.py`: typed `TrackSurface`, `VehicleTrajectory`, and
  `RaceData` objects.
- `Stackleberg_3DAnimator.py`: standalone Pygame/OpenGL renderer.
- `Stackelberg_HUD.py`: MATLAB data loading, telemetry calculation, Matplotlib
  HUD widgets, and precomputed 2D animation data. This is now legacy/standalone
  dashboard code rather than the default app shell.
- `VideoWriter.py`: headless/export path. It duplicates substantial logic from
  the integrated runner, but its duplicate data-processing call and duplicate
  exception handler have been removed.

## Current Data Flow

```mermaid
flowchart LR
    A["Leader/Follower .mat files"] --> B["Stackelberg_HUD.DataProcessor"]
    A --> C["Stackleberg_3DAnimator.Vehicle3DAnimatorGL"]
    D["Track .mat file"] --> C
    B --> E["PrecomputedData"]
    E --> F["Matplotlib HUD"]
    C --> G["Pygame/OpenGL 3D window"]
    F --> H["FuncAnimation update loop"]
    H --> G
```

The new package data path loads the MATLAB files into typed objects:

```mermaid
flowchart LR
    A["Leader/Follower .mat files"] --> B["RaceData"]
    D["Track .mat file"] --> B
    B --> C["Vehicle3DAnimatorGL"]
    C --> E["Single Pygame/OpenGL window"]
```

The legacy HUD and renderer classes still keep some older loading code for
compatibility, but new code should prefer `RaceData`.

## Legacy Fake Embedding

The old integrated dashboard path did not truly embed Pygame. The centre of the
Matplotlib dashboard was only a placeholder axes:

- `Stackelberg_Main.py` defines `animation_pos` and `animation_size` for a
  Matplotlib axes.
- `_setup_animation_area()` draws a black panel and text there.
- `_init_pygame()` then creates an entirely separate Pygame window using
  `pygame.display.set_mode(..., pygame.NOFRAME)`.
- The code now tries to set `SDL_VIDEO_WINDOW_POS` from the realised pixel
  rectangle of the Matplotlib axes. If that fails, it falls back to the old
  layout ratios.

So the Pygame scene was not a child widget of the HUD. It was a second top-level
window that happened to be borderless and positioned over the centre panel.

The default CLI no longer uses that architecture. It now opens one Pygame window
and lets the renderer own all OpenGL drawing.

This is fragile because the location depends on:

- Matplotlib backend behaviour.
- Window-manager decorations and maximisation.
- Wayland versus X11.
- Display scaling.
- Multiple monitors.
- Whether the Matplotlib window has actually been realised before Pygame is
  positioned.
- DPI differences between Matplotlib figure units and physical screen pixels.

## Broad Architecture Changes Needed

### 1. Split Model, View, And Application State

Create a small package structure instead of three large scripts:

```text
phd_3d_animator/
  data/
    loaders.py
    trajectory.py
    track.py
  math/
    frames.py
    tyre_forces.py
    monge_surface.py
  render/
    opengl_scene.py
    hud.py
  app/
    main_window.py
    playback.py
```

The data layer should load each `.mat` file once and produce explicit objects:

- `TrackSurface`: centreline, width, Monge coefficients, coordinate transforms,
  surface normal, and tangent/lateral basis.
- `VehicleTrajectory`: state arrays in physical units.
- `RaceData`: leader/follower trajectories resampled to one playback timeline.

### 2. Use One Windowing Framework

The first clean fix is now implemented: stop mixing a Matplotlib top-level
window with a separate Pygame top-level window.

Further recommended route if a rich research GUI is needed:

- Use PySide6/PyQt6 as the application shell.
- Put the 3D scene in a `QOpenGLWidget`.
- Put HUD plots in Qt widgets, a Matplotlib `FigureCanvasQTAgg`, or pyqtgraph.
- Drive everything from one timer and one playback state.

Implemented route:

- Make Pygame the only window.
- Render the 3D scene in that single window.
- Next: render the HUD in Pygame/OpenGL, possibly with ImGui or a lightweight
  UI overlay.

The Qt route is still better for research tooling because it gives native docking,
menus, file dialogs, resizing, and robust widget layout. The Pygame-only route
is simpler if the goal is just a robust animation/video viewer.

### 3. Isolate Coordinate Frames

Previously the 3D renderer flipped `y` and `z` internally:

```python
yMesh = -yMesh
zMesh = -zMesh * self.z_scale
```

That display choice now lives in explicit helpers:

```python
model_to_display_points(...)
display_to_model_points(...)
display_pose_from_model_pose(...)
```

Every renderer, camera, trail, and car transform should continue to consume
display-space data produced by the same frame module. No renderer should
silently reintroduce axis flips.

### 4. Use A Surface-Oriented Vehicle Pose

The car pose should be built from a proper orthonormal frame:

- longitudinal axis from the vehicle heading projected onto the Monge tangent
  plane;
- lateral axis on the surface;
- normal axis from the Monge surface normal;
- optional body pitch/roll only if the dynamic model provides it or if it can be
  consistently inferred.

The 3D code now builds a pose matrix from the vehicle heading projected onto the
Monge tangent plane and the local surface normal. It is a better visual pose
than yaw-plus-bank, but it is still not a full reconstruction of every body-rate
quantity in the MATLAB dynamics.

### 5. Add A Testable Validation Layer

The first headless regression tests now compare Python-derived values against
MATLAB-derived formulas and fixture data:

- state unscaling;
- `s,n -> x,y,z` conversion;
- surface normal and bank/camber angle;
- slip angles;
- tyre forces;
- resampling onto playback time;
- coordinate-frame conversion into display space;
- renderer/HUD common-timeline coordinate agreement;
- orthonormal display pose matrices.

These tests should run headlessly and should not require a display server.

## Short-Term Fix If Keeping The Current Architecture

The current code implements the first part of this fallback:

- Create the Matplotlib window and force a draw.
- Query the actual pixel bounds of `animation_ax` with
  `animation_ax.get_window_extent()`.
- Convert canvas coordinates to physical screen coordinates for Tk/Qt-like
  backends when possible.
- Position the Pygame window from that rectangle.

Still needed:

- Reposition/resize Pygame when the Matplotlib window moves or resizes.
- Make detached-window mode explicit when reliable placement is not possible.

This would reduce the annoyance, but it will remain fragile compared with a
single real GUI window.
