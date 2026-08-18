"""Geometry builders for OpenGL render paths."""

from __future__ import annotations

import numpy as np


def car_prism_geometry(a_m, b_m, width_m=1.8, height_m=0.6):
    """Return vertices and normals for a simple rectangular vehicle prism."""
    half_width = width_m / 2.0
    x_front = a_m
    x_rear = -b_m
    y_left = -half_width
    y_right = half_width
    z_bottom = 0.0
    z_top = height_m

    faces = [
        ((0.0, 0.0, -1.0), [(x_rear, y_left, z_bottom), (x_front, y_left, z_bottom), (x_front, y_right, z_bottom), (x_rear, y_right, z_bottom)]),
        ((0.0, 0.0, 1.0), [(x_rear, y_left, z_top), (x_rear, y_right, z_top), (x_front, y_right, z_top), (x_front, y_left, z_top)]),
        ((1.0, 0.0, 0.0), [(x_front, y_left, z_bottom), (x_front, y_left, z_top), (x_front, y_right, z_top), (x_front, y_right, z_bottom)]),
        ((-1.0, 0.0, 0.0), [(x_rear, y_left, z_bottom), (x_rear, y_right, z_bottom), (x_rear, y_right, z_top), (x_rear, y_left, z_top)]),
        ((0.0, -1.0, 0.0), [(x_rear, y_left, z_bottom), (x_rear, y_left, z_top), (x_front, y_left, z_top), (x_front, y_left, z_bottom)]),
        ((0.0, 1.0, 0.0), [(x_rear, y_right, z_bottom), (x_front, y_right, z_bottom), (x_front, y_right, z_top), (x_rear, y_right, z_top)]),
    ]

    vertices = []
    normals = []
    for normal, face_vertices in faces:
        vertices.extend(face_vertices)
        normals.extend([normal] * 4)

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32)


def trail_geometry(x, y, z, z_offset=0.1):
    """Return line-strip vertices for a vehicle trail."""
    return np.column_stack([x, y, np.asarray(z) + z_offset]).astype(np.float32)


def pose_axis_segments(pose, axis_scale=4.0):
    """Return line-segment vertices for x/y/z axes of one display-space pose."""
    pose = np.asarray(pose, dtype=float)
    origin = pose[:3, 3]
    axes = pose[:3, :3].T
    segments = []
    for axis in axes:
        segments.extend([origin, origin + axis * axis_scale])
    return np.array(segments, dtype=np.float32)


def normal_segments_from_poses(poses, stride=80, normal_scale=6.0):
    """Return sparse surface-normal line segments from display-space poses."""
    segments = []
    for pose in poses[::stride]:
        origin = pose[:3, 3]
        normal = pose[:3, 2]
        segments.extend([origin, origin + normal * normal_scale])
    return np.array(segments, dtype=np.float32)
