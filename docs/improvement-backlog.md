# Improvement Backlog

## P0: Fix Correctness Before Polishing

1. Done: fix yaw-rate unscaling in `Stackelberg_HUD.py`.
   - Implemented in `animator_math.unscale_vehicle_states()`.
   - Expected from MATLAB post-processing: multiply by `timescale`.

2. Done: fix wheelbase scaling in slip-angle calculations.
   - Implemented in `animator_math.physical_wheelbase()`.
   - Expected: use `a / lengthscale` and `b / lengthscale` when combining with
     physical `u`, `v`, and `omega_Bz`.

3. Partly done: build regression tests against MATLAB reference formulas.
   - Use the bundled `LeaderFixed.mat`, `FollowerFixed.mat`, and
     `NASCAR_Track_Monge_v3.mat`.
   - Current tests cover state unscaling, road-coordinate conversion, slip
     angles, display-frame conversion, Monge height, HUD/renderer common
     timeline agreement, and pose orthonormality.
   - Still needed: tyre-force regression against MATLAB reference arrays.

4. Done: isolate model coordinates from display coordinates.
   - Documented and tested the current `y` and `z` sign flips.
   - Ensure camera vectors, heading angles, trails, cars, and mesh all use the
     same display transform.

## P1: Fix The HUD/3D Window Architecture

1. Done for the default path: choose one application framework.
   - Recommended: PySide6/PyQt6 with a `QOpenGLWidget` and either Qt-native HUD
     widgets, pyqtgraph, or embedded Matplotlib.
   - Implemented: PySide6 HUD shell with a central `QOpenGLWidget`.

2. Done for the default path: remove the fake embedding.
   - `Stackelberg_Main.py` now launches the Qt HUD by default.
   - `python -m phd_3d_animator` uses the same Qt entry point.
   - Pygame-only remains available through `--pygame-only`.
   - The old fake-embedded Matplotlib/Pygame class remains as legacy code.
   - Still needed: decide whether to delete the legacy class or move it behind
     an explicit legacy entry point.

3. Partly done: centralise playback state.
   - One clock.
   - One current frame.
   - One play/pause state.
   - One input routing layer.
   - The Qt default path now has one `QTimer` and one renderer frame index, but
     remaining legacy classes still own separate playback code.

4. Done for the Qt default path: make resize behaviour deliberate.
   - The 3D viewport should resize with the app window.
   - The camera projection should update from the actual viewport aspect ratio.

## P2: Rebuild The 3D Vehicle Pose

1. Done: compute `s,n -> x,y,z` directly from the Monge coefficients instead of
   nearest-cell XY lookup.
2. Done: compute the Monge surface basis and normal at each vehicle position.
3. Done: build an orthonormal vehicle frame from heading and the surface normal.
4. Done: use a rotation matrix instead of sequential global OpenGL Euler
   rotations.
5. Still needed: add visual debugging overlays:
   - surface normal;
   - vehicle longitudinal axis;
   - lateral axis;
   - centreline tangent;
   - track boundaries.

## P3: Clean Up Data And Code Structure

1. Partly done: convert scripts into an importable Python package.
   - Added `phd_3d_animator/` with app, data, maths, geometry, and rendering
     modules.
   - Still needed: move remaining legacy classes fully into the package.
2. Done: add `requirements.txt`.
3. Partly done: remove duplicate data loading from `Stackelberg_Main.py`,
   `Stackleberg_3DAnimator.py`, and `VideoWriter.py`.
   - Added typed `RaceData`, `TrackSurface`, and `VehicleTrajectory`.
   - Still needed: make the legacy HUD and renderer consume `RaceData`
     directly.
4. Replace broad wildcard OpenGL imports with a renderer module boundary.
   - Done for `Stackelberg_Main.py` and `VideoWriter.py`; OpenGL calls are now
     delegated to the renderer.
5. Done: fix stale `TelemetryDashboard.__init__()` constructor path.
6. Done: fix duplicate/stale patterns in `VideoWriter.py`.
   - Removed the duplicate `process_data()` call.
   - Removed duplicate exception handlers.
   - Made OpenCV an optional import with a clear export-time error.
7. Partly done: add CLI argument parsing.
   - `Stackelberg_HUD.py` now accepts leader/follower/track paths.
   - Still needed:
   - detached versus embedded mode;
   - playback speed;
   - camera mode;
   - export options.

## P4: Improve User Experience

1. Start in a stable default view with visible cars and track.
2. Add a timeline slider and frame counter.
3. Add camera presets with visible buttons, not only keyboard controls.
4. Add file-open flow and clear errors when `.mat` fields are missing.
5. Partly done: add diagnostics showing:
   - loaded files;
   - frame count;
   - time range;
   - coordinate frame;
   - detected display backend;
   - OpenGL renderer string.
   - Added toggleable body-axis and surface-normal visual overlays with `V`.
6. Support screenshots and video export from the same rendering path.

## P5: Nice-To-Have Visual Improvements

1. Replace rectangular blocks with a simple car mesh or clearer vehicle glyph.
2. Add wheels and steering angle visualisation in 3D.
3. Colour track elevation/camber with a legend.
4. Add toggles for leader/follower trails, normals, and boundaries.
5. Add camera smoothing that respects speed and curvature.
