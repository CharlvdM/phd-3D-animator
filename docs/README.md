# 3D Animator Technical Notes

This folder tracks the current technical state of the Python 3D animator for
Stackelberg racing visualisation.

## Documents

- [Architecture](architecture.md): current structure, why the Pygame window does
  not sit reliably inside the HUD, and the broad architecture changes needed.
- [Mathematical Model Audit](mathematical-model-audit.md): comparison between
  the Python visualiser and the PhD/MATLAB Monge-track vehicle model.
- [Improvement Backlog](improvement-backlog.md): prioritised fixes and
  refactors.

## Scope

The project is a visualiser, not the optimisation model itself. It reads MATLAB
`.mat` outputs from the racing game/OCP code and reconstructs positions,
telemetry, tyre-force plots, and a Pygame/OpenGL 3D scene.

The most important current conclusions are:

- The central Pygame view is still not truly embedded in the HUD. It is a
  separate borderless top-level window, now positioned from the actual
  Matplotlib placeholder axes when the backend exposes that geometry.
- Coordinate conversion is now centralised in `animator_math.py`, although data
  loading is still duplicated between the HUD and the 3D renderer.
- The 2D path mapping `x = xc - n sin(psi)`, `y = yc + n cos(psi)` matches the
  MATLAB Monge coordinate mapping.
- The 3D renderer deliberately flips `y` and `z` for display through explicit,
  tested model/display frame helpers.
- The yaw-rate and wheelbase scaling errors in the HUD telemetry have been
  fixed and covered by regression tests.
