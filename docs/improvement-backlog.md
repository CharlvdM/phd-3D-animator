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

1. Choose one application framework.
   - Recommended: PySide6/PyQt6 with a `QOpenGLWidget` and either Qt-native HUD
     widgets, pyqtgraph, or embedded Matplotlib.
   - Acceptable simpler alternative: Pygame-only app with an OpenGL HUD overlay.

2. Partly done: reduce the fake embedding fragility.
   - The current central HUD panel still does not contain a real widget.
   - The Pygame window is now positioned from the actual placeholder axes when
     backend geometry is available, with a fallback to the old ratios.
   - True embedding still requires one GUI framework.

3. Centralise playback state.
   - One clock.
   - One current frame.
   - One play/pause state.
   - One input routing layer.

4. Make resize behaviour deliberate.
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

1. Convert scripts into an importable Python package.
2. Done: add `requirements.txt`.
3. Remove duplicate data loading from `Stackelberg_Main.py`,
   `Stackleberg_3DAnimator.py`, and `VideoWriter.py`.
4. Replace broad wildcard OpenGL imports with a renderer module boundary.
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
5. Add a diagnostics panel showing:
   - loaded files;
   - frame count;
   - time range;
   - coordinate frame;
   - detected display backend;
   - OpenGL renderer string.
6. Support screenshots and video export from the same rendering path.

## P5: Nice-To-Have Visual Improvements

1. Replace rectangular blocks with a simple car mesh or clearer vehicle glyph.
2. Add wheels and steering angle visualisation in 3D.
3. Colour track elevation/camber with a legend.
4. Add toggles for leader/follower trails, normals, and boundaries.
5. Add camera smoothing that respects speed and curvature.
