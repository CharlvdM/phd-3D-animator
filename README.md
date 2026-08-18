# PhD 3D Animator

Python/OpenGL visualisation for Stackelberg racing data. The main entry point is
`Stackelberg_Main.py`, which opens a Matplotlib HUD dashboard and a Pygame
OpenGL 3D animation window.

## Project Files

- `Stackelberg_Main.py` - integrated HUD and 3D animation runner.
- `Stackelberg_HUD.py` - data processing and dashboard/HUD helpers.
- `Stackleberg_3DAnimator.py` - Pygame/OpenGL 3D renderer.
- `docs/` - technical notes, architecture audit, mathematical audit, and
  improvement backlog.
- `LeaderFixed.mat` - leader trajectory data.
- `FollowerFixed.mat` - follower trajectory data.
- `NASCAR_Track_Monge_v3.mat` - default track mesh/data file.

## Technical Notes

Start with [`docs/README.md`](docs/README.md). The current headline issue is
that the 3D scene is not truly embedded in the HUD; it is a separate borderless
Pygame window now positioned from the Matplotlib placeholder axes when possible.
The docs also record the mathematical fixes made for yaw-rate scaling,
wheelbase scaling, display-frame transforms, and Monge surface pose handling.

## Python Environment

Use the project virtual environment directly:

```bash
cd ~/Dev/PhD/phd-3D-animator
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat
```

On Pop!_OS/Ubuntu, `python` may not exist as a command. If the virtual
environment is activated, `python3` can work, but the most reliable command is
the direct `.venv/bin/python` command above.

To activate the environment manually:

```bash
cd ~/Dev/PhD/phd-3D-animator
source .venv/bin/activate
which python3
```

`which python3` should point inside this project:

```text
/home/charl/Dev/PhD/phd-3D-animator/.venv/bin/python3
```

If it points to `/usr/bin/python3`, use `.venv/bin/python` directly.

## Dependencies

The project currently depends on:

- `numpy`
- `scipy`
- `matplotlib`
- `pygame`
- `PyOpenGL`
- `PyOpenGL_accelerate` if available

If the virtual environment needs to be recreated:

```bash
cd ~/Dev/PhD/phd-3D-animator
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

## Running

Default run, using `NASCAR_Track_Monge_v3.mat` automatically:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat
```

Explicit track file:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat NASCAR_Track_Monge_v3.mat
```

Do not literally pass `path/to/track.mat`; that was only a placeholder.

The code defaults Linux runs to SDL/X11 and PyOpenGL/GLX before importing
OpenGL. If a future display backend problem appears, test software OpenGL:

```bash
LIBGL_ALWAYS_SOFTWARE=1 .venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat
```

## Controls

- Space: play/pause.
- `F`: toggle Matplotlib fullscreen.
- `ESC` or `Q`: close the Pygame animation window.
- `1` to `5`: camera modes in the Pygame window.
- Mouse drag: rotate in free camera mode.
- Mouse wheel: zoom.
- `W`, `A`, `S`, `D`, `Q`, `E`: pan in free camera mode.

## Current Issues

These are the current known issues observed on Pop!_OS:

- Running with the system Python fails with `ModuleNotFoundError: No module named 'pygame'`. Use `.venv/bin/python`.
- The 3D view is still a borderless Pygame window over the HUD, not a real child
  widget.
- The Pygame window position is now derived from the realised HUD axes where the
  active Matplotlib backend exposes window geometry, but this remains less
  robust than a single Qt/Pygame application shell.
- `VideoWriter.py` imports without OpenCV, but video export requires
  `opencv-python`.

## Next Debugging Target

The next cleanup target is replacing the mixed Matplotlib/Pygame top-level
window arrangement with one real application framework, ideally PySide6/PyQt6
with a `QOpenGLWidget` or a Pygame-only UI shell.
