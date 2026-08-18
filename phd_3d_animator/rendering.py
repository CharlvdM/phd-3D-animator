"""Rendering package boundary.

The concrete OpenGL renderer still lives in the legacy
`Stackleberg_3DAnimator` module. New application code should import it through
this module so the implementation can be swapped without leaking OpenGL calls
through the rest of the codebase.
"""

from Stackleberg_3DAnimator import Vehicle3DAnimatorGL

__all__ = ["Vehicle3DAnimatorGL"]
