# 3D Animator Technical Notes

This folder tracks the current technical state of the Python 3D animator for
Stackelberg racing visualisation.

## Documents

- [Architecture](architecture.md): current structure, the new Qt/OpenGL widget
  shell, and the remaining architecture changes needed.
- [Mathematical Model Audit](mathematical-model-audit.md): comparison between
  the Python visualiser and the PhD/MATLAB Monge-track vehicle model.
- [Improvement Backlog](improvement-backlog.md): prioritised fixes and
  refactors.

## Scope

The project is a visualiser, not the optimisation model itself. It reads MATLAB
`.mat` outputs from the racing game/OCP code and reconstructs positions,
telemetry, tyre-force plots, and a Pygame/OpenGL 3D scene.

The most important current conclusions are:

- The default app now uses a single PySide6 window with a `QOpenGLWidget`, so it
  no longer depends on fake Matplotlib/Pygame embedding.
- Coordinate conversion is centralised in `animator_math.py`, and new code can
  use typed `RaceData`, `TrackSurface`, and `VehicleTrajectory` objects.
- The 2D path mapping `x = xc - n sin(psi)`, `y = yc + n cos(psi)` matches the
  MATLAB Monge coordinate mapping.
- The 3D renderer deliberately flips `y` and `z` for display through explicit,
  tested model/display frame helpers.
- The yaw-rate and wheelbase scaling errors in the HUD telemetry have been
  fixed and covered by regression tests.
- The renderer includes toggleable visual diagnostics for body axes and sparse
  surface normals.
