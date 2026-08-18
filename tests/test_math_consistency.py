import unittest

import numpy as np
from scipy.io import loadmat

from animator_math import (
    display_to_model_points,
    model_to_display_points,
    monge_height,
    physical_wheelbase,
    road_xy,
    unscale_vehicle_states,
)
from Stackelberg_HUD import DataProcessor
from Stackleberg_3DAnimator import Vehicle3DAnimatorGL
from phd_3d_animator.data import RaceData
from phd_3d_animator.geometry import car_prism_geometry, trail_geometry
from phd_3d_animator.geometry import normal_segments_from_poses, pose_axis_segments


LEADER = "LeaderFixed.mat"
FOLLOWER = "FollowerFixed.mat"
TRACK = "NASCAR_Track_Monge_v3.mat"


class MathConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.follower = loadmat(FOLLOWER, squeeze_me=True, struct_as_record=False)
        cls.aux = cls.follower["output"].result.setup.auxdata
        cls.states = cls.follower["output"].result.interpsolution.phase.state

    def test_state_unscaling_matches_matlab_convention(self):
        states = unscale_vehicle_states(
            self.states,
            self.aux.lengthscale,
            self.aux.lengthscale / self.aux.timescale,
            self.aux.timescale,
        )

        np.testing.assert_allclose(states["omega_Bz"], self.states[:, 3] * self.aux.timescale)
        np.testing.assert_allclose(states["t"], self.states[:, 8] / self.aux.timescale)

    def test_physical_wheelbase_matches_setup_values(self):
        a_m, b_m = physical_wheelbase(self.aux)

        self.assertAlmostEqual(a_m, 1.32, places=10)
        self.assertAlmostEqual(b_m, 1.47, places=10)

    def test_road_xy_matches_monge_mapping(self):
        n = self.states[:20, 0] / self.aux.lengthscale
        s = self.follower["output"].result.interpsolution.phase.time[:20] / self.aux.lengthscale
        s_track = self.aux.track.s / self.aux.lengthscale
        xc = np.interp(s, s_track, self.aux.track.xc)
        yc = np.interp(s, s_track, self.aux.track.yc)
        psi = np.interp(s, s_track, self.aux.track.psi)

        x, y = road_xy(xc, yc, psi, n)

        np.testing.assert_allclose(x, xc - n * np.sin(psi))
        np.testing.assert_allclose(y, yc + n * np.cos(psi))

    def test_hud_slip_angles_use_physical_scaling(self):
        dp = DataProcessor(LEADER, FOLLOWER, TRACK)

        ref_alp_f = np.arctan2(dp.vF + dp.omega_BzF * dp.a_m, dp.uF) - dp.deltaF
        ref_alp_r = np.arctan2(dp.vF - dp.omega_BzF * dp.b_m, dp.uF)

        np.testing.assert_allclose(dp.alp_fF, ref_alp_f)
        np.testing.assert_allclose(dp.alp_rF, ref_alp_r)

    def test_renderer_display_frame_is_explicitly_invertible(self):
        x = np.array([1.0, -2.0])
        y = np.array([3.0, -4.0])
        z = np.array([-5.0, 6.0])

        xd, yd, zd = model_to_display_points(x, y, z)
        xm, ym, zm = display_to_model_points(xd, yd, zd)

        np.testing.assert_allclose(xm, x)
        np.testing.assert_allclose(ym, y)
        np.testing.assert_allclose(zm, z)

    def test_renderer_model_coordinates_match_hud_on_common_timeline(self):
        dp = DataProcessor(LEADER, FOLLOWER, TRACK)
        animator = Vehicle3DAnimatorGL(LEADER, FOLLOWER, TRACK)

        np.testing.assert_allclose(animator.xF_model, dp.xF_interp, atol=1.0e-10)
        np.testing.assert_allclose(animator.yF_model, dp.yF_interp, atol=1.0e-10)
        np.testing.assert_allclose(animator.xL_model, dp.xL_interp, atol=1.0e-10)
        np.testing.assert_allclose(animator.yL_model, dp.yL_interp, atol=1.0e-10)
        np.testing.assert_allclose(animator.xF, dp.xF_interp, atol=1.0e-10)
        np.testing.assert_allclose(animator.yF, -dp.yF_interp, atol=1.0e-10)

    def test_renderer_height_uses_monge_polynomial(self):
        animator = Vehicle3DAnimatorGL(LEADER, FOLLOWER, TRACK)
        z0 = np.interp(animator.sF_i, animator.s, animator.z0)
        z1 = np.interp(animator.sF_i, animator.s, animator.z1)
        z2 = np.interp(animator.sF_i, animator.s, animator.z2)
        z3 = np.interp(animator.sF_i, animator.s, animator.z3)

        np.testing.assert_allclose(
            animator.zF_model,
            monge_height(z0, z1, z2, z3, animator.nF_i),
            atol=1.0e-12,
        )

    def test_pose_matrices_are_orthonormal(self):
        animator = Vehicle3DAnimatorGL(LEADER, FOLLOWER, TRACK)

        for pose in (animator.poseF[0], animator.poseF[len(animator.poseF) // 2], animator.poseF[-1]):
            rotation = pose[:3, :3]
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-10)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)

    def test_typed_race_data_matches_existing_renderer_arrays(self):
        race = RaceData.from_files(LEADER, FOLLOWER, TRACK)
        animator = Vehicle3DAnimatorGL(LEADER, FOLLOWER, TRACK)

        np.testing.assert_allclose(race.t, animator.t)
        np.testing.assert_allclose(race.follower_playback.x_model, animator.xF_model)
        np.testing.assert_allclose(race.follower_playback.y_model, animator.yF_model)
        np.testing.assert_allclose(race.follower_playback.z_model, animator.zF_model)
        np.testing.assert_allclose(race.follower_playback.pose_display, animator.poseF)
        np.testing.assert_allclose(race.leader_playback.x_display, animator.xL)
        np.testing.assert_allclose(race.leader_playback.y_display, animator.yL)
        np.testing.assert_allclose(race.leader_playback.z_display, animator.zL)

    def test_geometry_builders_return_vertex_array_inputs(self):
        vertices, normals = car_prism_geometry(1.32, 1.47)

        self.assertEqual(vertices.shape, (24, 3))
        self.assertEqual(normals.shape, (24, 3))
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(normals.dtype, np.float32)

        trail = trail_geometry(np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0]))
        np.testing.assert_allclose(trail, np.array([[1.0, 3.0, 5.1], [2.0, 4.0, 6.1]], dtype=np.float32))

        pose = np.eye(4)
        axes = pose_axis_segments(pose, axis_scale=2.0)
        self.assertEqual(axes.shape, (6, 3))
        np.testing.assert_allclose(axes[1], [2.0, 0.0, 0.0])
        normals = normal_segments_from_poses(np.array([pose]), stride=1, normal_scale=3.0)
        np.testing.assert_allclose(normals, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
