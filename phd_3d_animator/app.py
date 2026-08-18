"""Application entry points.

The full single-window app is introduced in a later refactor step. This shim
keeps `python -m phd_3d_animator` explicit instead of silently choosing one of
the legacy scripts.
"""


def main(argv=None):
    raise SystemExit(
        "The package entry point is installed. Use Stackelberg_Main.py until "
        "the single-window app refactor is enabled."
    )
