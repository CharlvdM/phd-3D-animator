# PhD 3D Animator

Python visualisation tooling for Stackelberg racing `.mat` outputs. The app
loads leader/follower trajectory files and a Monge-track surface file, then
renders a 3D OpenGL race view with animated telemetry panels.

The default interface is a PySide6/Qt HUD with a real `QOpenGLWidget` viewport
in the centre. Pygame-only and legacy Matplotlib/Pygame modes are still
available for debugging.

## What It Shows

- 3D track surface, leader/follower vehicle blocks, trails, and camera presets.
- Animated track overview with leader/follower positions.
- Steering wheel, speed, gap, position, G-G diagram, tyre-force plots, and
  acceleration/input bars.
- Optional 3D diagnostics for body axes and surface normals.

## Repository Layout

- `Stackelberg_Main.py` - main compatibility CLI; launches the Qt HUD by
  default.
- `phd_3d_animator/qt_app.py` - PySide6 HUD shell and embedded OpenGL viewport.
- `phd_3d_animator/app.py` - Pygame-only fallback shell.
- `Stackleberg_3DAnimator.py` - OpenGL renderer used by Qt, Pygame, and export
  paths.
- `Stackelberg_HUD.py` - telemetry processing and Matplotlib HUD helpers.
- `animator_math.py` - shared coordinate, scaling, Monge surface, tyre, and
  pose math.
- `docs/` - architecture notes, mathematical audit, improvement backlog, and
  technical notes.
- `tests/` - regression tests for math, geometry, renderer scaling, and CLI
  defaults.

## Data Files

Typical inputs are:

- `LeaderFixed.mat` - leader trajectory data.
- `FollowerFixed.mat` - follower trajectory data.
- `NASCAR_Track_Monge_v3.mat` - default track surface/mesh data.

The track file is optional on the CLI because `NASCAR_Track_Monge_v3.mat` is
used by default. Do not literally pass `path/to/track.mat`; that is only a
placeholder.

## Setup

Use the project virtual environment directly:

```bash
cd /home/charl/Dev/PhD/phd-3D-animator
.venv/bin/python -m pip install -r requirements.txt
```

If the virtual environment needs to be recreated:

```bash
cd /home/charl/Dev/PhD/phd-3D-animator
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Pop!_OS/Ubuntu, `python` may not exist as a command. Prefer
`.venv/bin/python` from this project.

## Running

Default Qt HUD:

```bash
cd /home/charl/Dev/PhD/phd-3D-animator
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat
```

Equivalent package entry point:

```bash
.venv/bin/python -m phd_3d_animator LeaderFixed.mat FollowerFixed.mat
```

Explicit track file:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat NASCAR_Track_Monge_v3.mat
```

Pygame-only fallback:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --pygame-only
```

Legacy Matplotlib HUD with borderless Pygame overlay:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --legacy-hud
```

Useful options:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --camera overview
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --diagnostics
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --car-scale 1.0
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --fps 20
```

`--car-scale` is visual only. It changes the rendered block size, not the
vehicle dynamics, slip angles, wheelbase, or telemetry calculations.

## Controls

- Space: play/pause.
- `ESC`: close the app.
- `1` to `5`: camera presets.
- `V`: toggle body-axis and surface-normal diagnostics.
- Mouse drag: rotate in free camera mode.
- Mouse wheel: zoom in free camera mode.
- `W`, `A`, `S`, `D`, `Q`, `E`: pan in free camera mode.

## Testing

Run the regression tests with:

```bash
cd /home/charl/Dev/PhD/phd-3D-animator
MPLCONFIGDIR=/tmp/mpl-3d-animator .venv/bin/python -m unittest discover -s tests
```

Compile-check the main modules:

```bash
MPLCONFIGDIR=/tmp/mpl-3d-animator .venv/bin/python -m py_compile Stackelberg_Main.py Stackleberg_3DAnimator.py VideoWriter.py phd_3d_animator/*.py tests/test_math_consistency.py
```

## Linux Display Notes

The default Qt path uses Qt's OpenGL context. The Pygame-only and legacy HUD
paths scope SDL/X11 and PyOpenGL/GLX setup to the Pygame code so those settings
do not interfere with Qt.

If a display backend problem appears, try software OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat
```

## Further Documentation

Start with [docs/technical-notes.md](docs/technical-notes.md), then see:

- [docs/architecture.md](docs/architecture.md)
- [docs/mathematical-model-audit.md](docs/mathematical-model-audit.md)
- [docs/improvement-backlog.md](docs/improvement-backlog.md)

## Known Issues

- Running with system Python often fails because dependencies are only installed
  in the project `.venv`.
- The legacy HUD mode still uses a borderless Pygame window positioned over a
  Matplotlib dashboard. Use the default Qt HUD for a real embedded OpenGL
  viewport.
- `VideoWriter.py` imports without OpenCV, but actual video export requires
  `opencv-python`.
