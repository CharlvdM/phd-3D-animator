"""Shared maths helpers for the Stackelberg race visualiser."""

from __future__ import annotations

import numpy as np


DISPLAY_AXIS_SIGN = np.array([1.0, -1.0, -1.0])


def unscale_vehicle_states(states, lengthscale, velscale, timescale):
    """Return optimiser states in physical plotting units.

    MATLAB uses the state scale
    [lengthscale, 1, velscale, 1/timescale, velscale, 1, 1, 1, timescale].
    GPOPS stores scaled values, so physical yaw rate is multiplied by
    timescale, not divided by it.
    """
    states = np.asarray(states)
    return {
        "n": states[:, 0] / lengthscale,
        "xi": states[:, 1],
        "v": states[:, 2] / velscale,
        "omega_Bz": states[:, 3] * timescale,
        "u": states[:, 4] / velscale,
        "delta": states[:, 5],
        "k_f": states[:, 6],
        "k_r": states[:, 7],
        "t": states[:, 8] / timescale,
    }


def physical_wheelbase(auxdata):
    """Return front and rear CG distances in metres."""
    lengthscale = auxdata.lengthscale
    return auxdata.a / lengthscale, auxdata.b / lengthscale


def road_xy(xc, yc, psi, n):
    """Map Monge road coordinates to model-space x/y coordinates."""
    return xc - n * np.sin(psi), yc + n * np.cos(psi)


def monge_height(z0, z1, z2, z3, n):
    """Evaluate z(s,n) = z0 + z1*n + z2*n^2 + z3*n^3."""
    return z0 + z1 * n + z2 * n**2 + z3 * n**3


def monge_dz_dn(z1, z2, z3, n):
    """Evaluate the lateral derivative of the Monge surface height."""
    return z1 + 2.0 * z2 * n + 3.0 * z3 * n**2


def monge_dz_ds(dz0, dz1, dz2, dz3, n):
    """Evaluate the longitudinal derivative of the Monge surface height."""
    return dz0 + dz1 * n + dz2 * n**2 + dz3 * n**3


def model_to_display_points(x, y, z, z_scale=1.0):
    """Convert model-space coordinates to the OpenGL display frame."""
    return np.asarray(x), -np.asarray(y), -np.asarray(z) * z_scale


def display_to_model_points(x, y, z, z_scale=1.0):
    """Convert OpenGL display-frame coordinates back to model space."""
    return np.asarray(x), -np.asarray(y), -np.asarray(z) / z_scale


def display_rotation_from_model(rotation):
    """Apply the display-frame axis flip to a model-space rotation matrix."""
    return DISPLAY_AXIS_SIGN[:, np.newaxis] * np.asarray(rotation)


def normalise(vec, fallback=None):
    """Return a unit vector, using fallback if the norm is too small."""
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm > 1.0e-12:
        return vec / norm
    if fallback is None:
        return np.zeros_like(vec)
    return np.asarray(fallback, dtype=float)


def surface_basis(psi, dpsi_ds, dz_ds, dz_dn, n):
    """Return model-space surface tangent, lateral and normal unit vectors."""
    dx_ds = np.array([
        np.cos(psi) * (1.0 - n * dpsi_ds),
        np.sin(psi) * (1.0 - n * dpsi_ds),
        dz_ds,
    ], dtype=float)
    dx_dn = np.array([
        -np.sin(psi),
        np.cos(psi),
        dz_dn,
    ], dtype=float)
    tangent = normalise(dx_ds, fallback=[1.0, 0.0, 0.0])
    lateral = normalise(dx_dn, fallback=[0.0, 1.0, 0.0])
    normal = normalise(np.cross(dx_ds, dx_dn), fallback=[0.0, 0.0, 1.0])
    if normal[2] < 0:
        normal = -normal
    return tangent, lateral, normal


def vehicle_pose_matrix(x, y, z, heading, psi, dpsi_ds, dz_ds, dz_dn, n):
    """Build a model-space vehicle pose aligned with the Monge surface."""
    _, _, normal = surface_basis(psi, dpsi_ds, dz_ds, dz_dn, n)
    heading_xy = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=float)
    body_x = heading_xy - np.dot(heading_xy, normal) * normal
    body_x = normalise(body_x, fallback=[np.cos(psi), np.sin(psi), 0.0])
    body_y = normalise(np.cross(normal, body_x), fallback=[-np.sin(heading), np.cos(heading), 0.0])
    body_x = normalise(np.cross(body_y, normal), fallback=body_x)
    rotation = np.column_stack([body_x, body_y, normal])

    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose


def display_pose_from_model_pose(model_pose, z_scale=1.0):
    """Convert a model-space 4x4 pose matrix to OpenGL display space."""
    pose = np.asarray(model_pose, dtype=float).copy()
    pose[:3, :3] = display_rotation_from_model(pose[:3, :3])
    pose[:3, 3] = np.array(model_to_display_points(*pose[:3, 3], z_scale=z_scale))
    return pose

