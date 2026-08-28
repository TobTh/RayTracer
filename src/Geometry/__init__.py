"""Geometric helpers for the ray tracer.

The rotation matrices follow the row-vector convention used throughout the
project: they are applied from the right, as ``points @ rotation``, so each row
of a matrix holds one rotated basis vector. Re-exported here so callers can use
``from Geometry import rotx`` without reaching into the submodule.
"""

from .Rotations import rotx, roty, rotz

__all__ = ["rotx", "roty", "rotz"]
