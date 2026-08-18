"""Typed data model for Stackelberg race visualisation inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import loadmat

from .maths import (
    display_pose_from_model_pose,
    model_to_display_points,
    monge_dz_dn,
    monge_dz_ds,
    monge_height,
    physical_wheelbase,
    road_xy,
    unscale_vehicle_states,
    vehicle_pose_matrix,
)


@dataclass(frozen=True)
class VehicleDimensions:
    """Physical vehicle dimensions used by the visualiser."""

    a_m: float
    b_m: float
    width_m: float = 1.8
    height_m: float = 0.6


@dataclass(frozen=True)
class VehicleTrajectory:
    """One vehicle trajectory in physical plotting units."""

    s: np.ndarray
    n: np.ndarray
    xi: np.ndarray
    v: np.ndarray
    omega_bz: np.ndarray
    u: np.ndarray
    delta: np.ndarray
    k_f: np.ndarray
    k_r: np.ndarray
    t: np.ndarray
    controls: np.ndarray

    @classmethod
    def from_phase(cls, phase, lengthscale, velscale, timescale):
        states = unscale_vehicle_states(phase.state, lengthscale, velscale, timescale)
        return cls(
            s=phase.time / lengthscale,
            n=states["n"],
            xi=states["xi"],
            v=states["v"],
            omega_bz=states["omega_Bz"],
            u=states["u"],
            delta=states["delta"],
            k_f=states["k_f"],
            k_r=states["k_r"],
            t=states["t"],
            controls=phase.control,
        )

    def heading(self, track: "TrackSurface"):
        psi = np.interp(self.s, track.s_centre, track.psi_centre)
        return np.unwrap(psi + self.xi)


@dataclass(frozen=True)
class TrackSurface:
    """Monge-patch road surface and centreline data."""

    s: np.ndarray
    xc: np.ndarray
    yc: np.ndarray
    psi: np.ndarray
    rw: np.ndarray | float
    z0: np.ndarray
    z1: np.ndarray
    z2: np.ndarray
    z3: np.ndarray
    dpsi_ds: np.ndarray
    dz0_ds: np.ndarray
    dz1_ds: np.ndarray
    dz2_ds: np.ndarray
    dz3_ds: np.ndarray
    s_centre: np.ndarray
    xc_centre: np.ndarray
    yc_centre: np.ndarray
    psi_centre: np.ndarray
    rw_centre: np.ndarray | float

    @classmethod
    def from_files(cls, track_file, auxdata):
        data = loadmat(track_file, squeeze_me=True)
        rw = data["rw"]
        if isinstance(rw, np.ndarray):
            rw = float(rw) if rw.size == 1 else rw
        else:
            rw = float(rw)

        return cls(
            s=np.asarray(data["s"], dtype=float),
            xc=np.asarray(data["xc"], dtype=float),
            yc=np.asarray(data["yc"], dtype=float),
            psi=np.asarray(data["psi"], dtype=float),
            rw=rw,
            z0=np.asarray(data["z0"], dtype=float),
            z1=np.asarray(data["z1"], dtype=float),
            z2=np.asarray(data["z2"], dtype=float),
            z3=np.asarray(data["z3"], dtype=float),
            dpsi_ds=np.asarray(data["dpsi"], dtype=float),
            dz0_ds=np.asarray(data["dz0"], dtype=float),
            dz1_ds=np.asarray(data["dz1"], dtype=float),
            dz2_ds=np.asarray(data["dz2"], dtype=float),
            dz3_ds=np.asarray(data["dz3"], dtype=float),
            s_centre=np.asarray(auxdata.track.s / auxdata.lengthscale, dtype=float),
            xc_centre=np.asarray(auxdata.track.xc, dtype=float),
            yc_centre=np.asarray(auxdata.track.yc, dtype=float),
            psi_centre=np.asarray(auxdata.track.psi, dtype=float),
            rw_centre=auxdata.track.rw / auxdata.lengthscale,
        )

    def centreline_at(self, s):
        return (
            np.interp(s, self.s_centre, self.xc_centre),
            np.interp(s, self.s_centre, self.yc_centre),
            np.interp(s, self.s_centre, self.psi_centre),
        )

    def model_xy(self, s, n):
        xc, yc, psi = self.centreline_at(s)
        return road_xy(xc, yc, psi, n)

    def surface_values(self, s, n):
        z0 = np.interp(s, self.s, self.z0)
        z1 = np.interp(s, self.s, self.z1)
        z2 = np.interp(s, self.s, self.z2)
        z3 = np.interp(s, self.s, self.z3)
        dz0 = np.interp(s, self.s, self.dz0_ds)
        dz1 = np.interp(s, self.s, self.dz1_ds)
        dz2 = np.interp(s, self.s, self.dz2_ds)
        dz3 = np.interp(s, self.s, self.dz3_ds)
        dpsi = np.interp(s, self.s, self.dpsi_ds)
        return (
            monge_height(z0, z1, z2, z3, n),
            monge_dz_ds(dz0, dz1, dz2, dz3, n),
            monge_dz_dn(z1, z2, z3, n),
            dpsi,
        )

    def mesh(self, nlat=13, z_scale=1.0):
        if isinstance(self.rw, np.ndarray) and self.rw.size > 1:
            n_spaced = (np.linspace(-0.5, 0.5, nlat)[:, np.newaxis] * self.rw).T
        else:
            n_spaced = np.tile(np.linspace(-0.5, 0.5, nlat) * float(self.rw), (len(self.s), 1))

        z_model = monge_height(
            self.z0[:, np.newaxis],
            self.z1[:, np.newaxis],
            self.z2[:, np.newaxis],
            self.z3[:, np.newaxis],
            n_spaced,
        )
        x_model, y_model = road_xy(self.xc[:, np.newaxis], self.yc[:, np.newaxis], self.psi[:, np.newaxis], n_spaced)
        x_display, y_display, z_display = model_to_display_points(x_model, y_model, z_model, z_scale=z_scale)
        return x_model, y_model, z_model, x_display, y_display, z_display, n_spaced

    def display_poses(self, s_values, n_values, headings, x_model, y_model, z_model, z_scale=1.0):
        poses = []
        for s, n, heading, x, y, z in zip(s_values, n_values, headings, x_model, y_model, z_model):
            psi = np.interp(s, self.s, self.psi)
            _, dz_ds, dz_dn, dpsi_ds = self.surface_values(s, n)
            model_pose = vehicle_pose_matrix(x, y, z, heading, psi, dpsi_ds, dz_ds, dz_dn, n)
            poses.append(display_pose_from_model_pose(model_pose, z_scale=z_scale))
        return np.array(poses, dtype=float)


@dataclass(frozen=True)
class PlaybackVehicle:
    """Vehicle arrays resampled onto the common playback timeline."""

    s: np.ndarray
    n: np.ndarray
    heading: np.ndarray
    x_model: np.ndarray
    y_model: np.ndarray
    z_model: np.ndarray
    x_display: np.ndarray
    y_display: np.ndarray
    z_display: np.ndarray
    pose_display: np.ndarray
    banking_angle: np.ndarray


@dataclass(frozen=True)
class RaceData:
    """Loaded and resampled leader/follower race data."""

    leader: VehicleTrajectory
    follower: VehicleTrajectory
    track: TrackSurface
    dimensions: VehicleDimensions
    t: np.ndarray
    leader_playback: PlaybackVehicle
    follower_playback: PlaybackVehicle
    z_scale: float = 1.0

    @classmethod
    def from_files(cls, leader_file, follower_file, track_file, z_scale=1.0):
        leader_mat = loadmat(leader_file, squeeze_me=True, struct_as_record=False)
        follower_mat = loadmat(follower_file, squeeze_me=True, struct_as_record=False)
        auxdata = follower_mat["output"].result.setup.auxdata
        lengthscale = auxdata.lengthscale
        timescale = auxdata.timescale
        velscale = lengthscale / timescale

        leader = VehicleTrajectory.from_phase(
            leader_mat["output"].result.interpsolution.phase,
            lengthscale,
            velscale,
            timescale,
        )
        follower = VehicleTrajectory.from_phase(
            follower_mat["output"].result.interpsolution.phase,
            lengthscale,
            velscale,
            timescale,
        )
        track = TrackSurface.from_files(track_file, auxdata)
        dimensions = VehicleDimensions(*physical_wheelbase(auxdata))
        t = np.linspace(0, min(follower.t[-1], leader.t[-1]), len(follower.t))

        follower_playback = cls._playback_for(follower, t, track, z_scale)
        leader_playback = cls._playback_for(leader, t, track, z_scale)
        return cls(leader, follower, track, dimensions, t, leader_playback, follower_playback, z_scale)

    @staticmethod
    def _playback_for(vehicle, t, track, z_scale):
        s = np.interp(t, vehicle.t, vehicle.s)
        n = np.interp(t, vehicle.t, vehicle.n)
        x_model, y_model = track.model_xy(s, n)
        z_model, _, dz_dn, _ = track.surface_values(s, n)
        heading = np.interp(t, vehicle.t, vehicle.heading(track))
        x_display, y_display, z_display = model_to_display_points(x_model, y_model, z_model, z_scale=z_scale)
        pose_display = track.display_poses(s, n, heading, x_model, y_model, z_model, z_scale=z_scale)
        return PlaybackVehicle(
            s=s,
            n=n,
            heading=heading,
            x_model=x_model,
            y_model=y_model,
            z_model=z_model,
            x_display=x_display,
            y_display=y_display,
            z_display=z_display,
            pose_display=pose_display,
            banking_angle=np.arctan(dz_dn * z_scale),
        )

    @property
    def frame_count(self):
        return len(self.t)
