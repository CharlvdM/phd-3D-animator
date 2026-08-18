"""Shared mathematical helpers for track and vehicle geometry.

This module keeps the new package import path stable while the legacy scripts
still import `animator_math` directly.
"""

from animator_math import (
    DISPLAY_AXIS_SIGN,
    display_pose_from_model_pose,
    display_rotation_from_model,
    display_to_model_points,
    model_to_display_points,
    monge_dz_dn,
    monge_dz_ds,
    monge_height,
    normalise,
    physical_wheelbase,
    road_xy,
    surface_basis,
    unscale_vehicle_states,
    vehicle_pose_matrix,
)

__all__ = [
    "DISPLAY_AXIS_SIGN",
    "display_pose_from_model_pose",
    "display_rotation_from_model",
    "display_to_model_points",
    "model_to_display_points",
    "monge_dz_dn",
    "monge_dz_ds",
    "monge_height",
    "normalise",
    "physical_wheelbase",
    "road_xy",
    "surface_basis",
    "unscale_vehicle_states",
    "vehicle_pose_matrix",
]
