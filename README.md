# PhD 3D Animator

Python/OpenGL visualisation for Stackelberg racing data. The main entry point is
`Stackelberg_Main.py`, which launches the HUD-backed runner. The package entry
point remains available for a Pygame-only view.

## Project Files

- `Stackelberg_Main.py` - HUD-backed integrated runner.
- `phd_3d_animator/` - package modules for data, maths, geometry, rendering,
  and the Pygame app.
- `Stackelberg_HUD.py` - data processing and dashboard/HUD helpers.
- `Stackleberg_3DAnimator.py` - Pygame/OpenGL 3D renderer.
- `docs/` - technical notes, architecture audit, mathematical audit, and
  improvement backlog.
- `LeaderFixed.mat` - leader trajectory data.
- `FollowerFixed.mat` - follower trajectory data.
- `NASCAR_Track_Monge_v3.mat` - default track mesh/data file.

## Technical Notes

Start with [`docs/README.md`](docs/README.md). The default path keeps the
Matplotlib HUD visible and positions the Pygame/OpenGL renderer over the central
viewport. The package entry point is a Pygame-only alternative.

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

Pygame-only package entry point:

```bash
.venv/bin/python -m phd_3d_animator LeaderFixed.mat FollowerFixed.mat
```

Pygame-only through the main script:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --pygame-only
```

The 3D cars are deliberately scaled up slightly for visibility. The default
visual scale is `1.5`, and it does not affect the vehicle dynamics, slip angles, or
wheelbase calculations:

```bash
.venv/bin/python Stackelberg_Main.py LeaderFixed.mat FollowerFixed.mat --car-scale 2
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
- `ESC` or `Q`: close the app.
- `1` to `5`: camera modes in the Pygame window.
- `V`: toggle body-axis and surface-normal diagnostics.
- Mouse drag: rotate in free camera mode.
- Mouse wheel: zoom.
- `W`, `A`, `S`, `D`, `Q`, `E`: pan in free camera mode.

## Current Issues

These are the current known issues observed on Pop!_OS:

- Running with the system Python fails with `ModuleNotFoundError: No module named 'pygame'`. Use `.venv/bin/python`.
- The HUD-backed mode still relies on a borderless Pygame window positioned over
  the central Matplotlib viewport. That keeps the HUD visible, but true widget
  embedding would still require a Qt-style application rewrite.
- `VideoWriter.py` imports without OpenCV, but video export requires
  `opencv-python`.

## Next Debugging Target

The next cleanup target is building a richer HUD overlay inside the single
Pygame window, then retiring or moving the legacy Matplotlib dashboard.
