import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from scipy.io import loadmat
from animator_math import (
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
from phd_3d_animator.geometry import (
    car_prism_geometry,
    normal_segments_from_poses,
    pose_axis_segments,
    trail_geometry,
)

class Vehicle3DAnimatorGL:
    def __init__(self, leader_file, follower_file, track_file, car_visual_scale=1.5):
        print("Loading vehicle data...")
        
        # Load vehicle simulation data
        self.leader = loadmat(leader_file, squeeze_me=True, struct_as_record=False)
        self.follower = loadmat(follower_file, squeeze_me=True, struct_as_record=False)
        
        # Extract scaling parameters
        auxdata = self.follower["output"].result.setup.auxdata
        self.lengthscale = auxdata.lengthscale
        self.massscale = auxdata.massscale
        self.timescale = auxdata.timescale
        self.velscale = self.lengthscale / self.timescale
        
        # Vehicle dimensions
        self.a = auxdata.a
        self.b = auxdata.b
        self.a_m, self.b_m = physical_wheelbase(auxdata)
        self.car_visual_scale = car_visual_scale
        self.rwTrack = auxdata.track.rw / self.lengthscale

        # Track centerline
        self.xc = auxdata.track.xc
        self.yc = auxdata.track.yc
        self.psiTrack = auxdata.track.psi
        self.sTrack = auxdata.track.s / self.lengthscale

        # Add z-scaling factor
        self.z_scale = 1.0  # Adjust this to make elevation more visible
        
        # Load 3D track mesh
        print("Loading 3D track...")
        data = loadmat(track_file, squeeze_me=True)
        self.s = data['s']
        self.track_xc = data['xc']
        self.track_yc = data['yc']
        self.psi = data['psi']
        
        rw_data = data['rw']
        if isinstance(rw_data, np.ndarray):
            self.rw = float(rw_data) if rw_data.size == 1 else rw_data
        else:
            self.rw = float(rw_data)
        
        self.z0 = data['z0']
        self.z1 = data['z1']
        self.z2 = data['z2']
        self.z3 = data['z3']
        self.dpsi_ds = data['dpsi']
        self.dz0_ds = data['dz0']
        self.dz1_ds = data['dz1']
        self.dz2_ds = data['dz2']
        self.dz3_ds = data['dz3']
        
        # Compute track mesh
        self.xMesh, self.yMesh, self.zMesh, self.n_spaced = self._compute_track_mesh(nlat=13)
        print(f"Track mesh: {self.xMesh.shape}")
        
        # Process vehicle trajectories
        self._process_vehicle_data()
        self._prepare_common_timeline()
        
 #     Camera parameters
        self.camera_mode = 'follow'  # Start in follow mode
        self.camera_distance = 800
        self.camera_angle_h = 45
        self.camera_angle_v = 30
        self.camera_target = np.array([
            np.mean(self.xMesh),
            np.mean(self.yMesh),
            np.mean(self.zMesh)
        ])

        # Animation state
        self.mouse_down = False
        self.last_mouse_pos = (0, 0)
        self.playing = True
        self.current_frame = 0
        self.trail_length = 100

        # Add animation speed control
        self.animation_speed = 1.0  # 1.0 = normal speed
        self.frame_accumulator = 0.0
        self.show_diagnostics = False
        self.diagnostic_stride = 80
        self.axis_scale = 4.0
        self.normal_scale = 6.0

        # VBO placeholders
        self.track_vbo = None
        self.boundary_vbos = None
        self._rebuild_car_geometry()
        self.follower_normal_segments = normal_segments_from_poses(
            self.poseF,
            stride=self.diagnostic_stride,
            normal_scale=self.normal_scale,
        )
        self.leader_normal_segments = normal_segments_from_poses(
            self.poseL,
            stride=self.diagnostic_stride,
            normal_scale=self.normal_scale,
        )

    def set_car_visual_scale(self, visual_scale):
        """Update the visual-only car block scale and rebuild its mesh."""
        self.car_visual_scale = float(visual_scale)
        self._rebuild_car_geometry()

    def _rebuild_car_geometry(self):
        self.car_vertices, self.car_normals = car_prism_geometry(
            self.a_m,
            self.b_m,
            visual_scale=self.car_visual_scale,
        )
        
    def _compute_track_mesh(self, nlat=13):
        """Compute 3D track mesh"""
        if isinstance(self.rw, np.ndarray) and self.rw.size > 1:
            n_spaced_ratio = np.linspace(-0.5, 0.5, nlat)
            n_spaced = n_spaced_ratio[:, np.newaxis] * self.rw
            n_spaced = n_spaced.T
        else:
            rw_scalar = float(self.rw) if isinstance(self.rw, np.ndarray) else self.rw
            n_spaced = np.linspace(-0.5, 0.5, nlat) * rw_scalar
            n_spaced = np.tile(n_spaced, (len(self.s), 1))
        
        zMesh_model = np.zeros((len(self.s), nlat))
        for k in range(nlat):
            nk = n_spaced[:, k]
            zMesh_model[:, k] = monge_height(self.z0, self.z1, self.z2, self.z3, nk)

        xMesh_model, yMesh_model = road_xy(
            self.track_xc[:, np.newaxis],
            self.track_yc[:, np.newaxis],
            self.psi[:, np.newaxis],
            n_spaced,
        )
        xMesh, yMesh, zMesh = model_to_display_points(
            xMesh_model, yMesh_model, zMesh_model, z_scale=self.z_scale
        )
        self.xMesh_model = xMesh_model
        self.yMesh_model = yMesh_model
        self.zMesh_model = zMesh_model

        return xMesh, yMesh, zMesh, n_spaced
    
    def _process_vehicle_data(self):
        """Extract vehicle states from simulation data"""
        # Leader states
        self.sL = self.leader["output"].result.interpsolution.phase.time / self.lengthscale
        statesL = self.leader["output"].result.interpsolution.phase.state
        leader = unscale_vehicle_states(statesL, self.lengthscale, self.velscale, self.timescale)
        self.nL = leader["n"]
        self.xiL = leader["xi"]
        self.tL = leader["t"]
        
        # Follower states
        self.sF = self.follower["output"].result.interpsolution.phase.time / self.lengthscale
        statesF = self.follower["output"].result.interpsolution.phase.state
        follower = unscale_vehicle_states(statesF, self.lengthscale, self.velscale, self.timescale)
        self.nF = follower["n"]
        self.xiF = follower["xi"]
        self.tF = follower["t"]
        
        print(f"Leader: {len(self.sL)} points, Follower: {len(self.sF)} points")
    
    def get_surface_values_at_sn(self, s, n):
        """Evaluate Monge height and derivatives at track coordinates."""
        z0 = np.interp(s, self.s, self.z0)
        z1 = np.interp(s, self.s, self.z1)
        z2 = np.interp(s, self.s, self.z2)
        z3 = np.interp(s, self.s, self.z3)
        dz0 = np.interp(s, self.s, self.dz0_ds)
        dz1 = np.interp(s, self.s, self.dz1_ds)
        dz2 = np.interp(s, self.s, self.dz2_ds)
        dz3 = np.interp(s, self.s, self.dz3_ds)
        dpsi_ds = np.interp(s, self.s, self.dpsi_ds)
        z = monge_height(z0, z1, z2, z3, n)
        dz_dn = monge_dz_dn(z1, z2, z3, n)
        dz_ds = monge_dz_ds(dz0, dz1, dz2, dz3, n)
        return z, dz_ds, dz_dn, dpsi_ds
    
    def get_banking_angle_at_sn(self, s, n):
        """Get banking angle at track coordinates (s,n)"""
        _, _, dz_dn, _ = self.get_surface_values_at_sn(s, n)
        return np.arctan(dz_dn * self.z_scale)

    def _display_poses(self, s_values, n_values, headings, x_model, y_model, z_model):
        poses = []
        for s, n, heading, x, y, z in zip(s_values, n_values, headings, x_model, y_model, z_model):
            psi = np.interp(s, self.s, self.psi)
            _, dz_ds, dz_dn, dpsi_ds = self.get_surface_values_at_sn(s, n)
            pose = vehicle_pose_matrix(x, y, z, heading, psi, dpsi_ds, dz_ds, dz_dn, n)
            poses.append(display_pose_from_model_pose(pose, z_scale=self.z_scale))
        return np.array(poses, dtype=float)
    
    def _prepare_common_timeline(self):
        """Interpolate vehicle data to common timeline"""
        # Resample to common timeline
        self.tNum = len(self.tF)
        self.t = np.linspace(0, min(self.tF[-1], self.tL[-1]), self.tNum)

        self.sF_i = np.interp(self.t, self.tF, self.sF)
        self.nF_i = np.interp(self.t, self.tF, self.nF)
        self.sL_i = np.interp(self.t, self.tL, self.sL)
        self.nL_i = np.interp(self.t, self.tL, self.nL)

        psiF = np.interp(self.sF_i, self.sTrack, self.psiTrack)
        psiL = np.interp(self.sL_i, self.sTrack, self.psiTrack)
        xcF = np.interp(self.sF_i, self.sTrack, self.xc)
        ycF = np.interp(self.sF_i, self.sTrack, self.yc)
        xcL = np.interp(self.sL_i, self.sTrack, self.xc)
        ycL = np.interp(self.sL_i, self.sTrack, self.yc)

        self.xF_model, self.yF_model = road_xy(xcF, ycF, psiF, self.nF_i)
        self.xL_model, self.yL_model = road_xy(xcL, ycL, psiL, self.nL_i)
        self.zF_model = self.get_surface_values_at_sn(self.sF_i, self.nF_i)[0]
        self.zL_model = self.get_surface_values_at_sn(self.sL_i, self.nL_i)[0]

        headingF_raw = np.unwrap(np.interp(self.sF, self.sTrack, self.psiTrack) + self.xiF)
        headingL_raw = np.unwrap(np.interp(self.sL, self.sTrack, self.psiTrack) + self.xiL)
        self.carAngleF = np.interp(self.t, self.tF, headingF_raw)
        self.carAngleL = np.interp(self.t, self.tL, headingL_raw)

        self.xF, self.yF, self.zF = model_to_display_points(
            self.xF_model, self.yF_model, self.zF_model, z_scale=self.z_scale
        )
        self.xL, self.yL, self.zL = model_to_display_points(
            self.xL_model, self.yL_model, self.zL_model, z_scale=self.z_scale
        )

        self.poseF = self._display_poses(
            self.sF_i, self.nF_i, self.carAngleF, self.xF_model, self.yF_model, self.zF_model
        )
        self.poseL = self._display_poses(
            self.sL_i, self.nL_i, self.carAngleL, self.xL_model, self.yL_model, self.zL_model
        )

        self.banking_angleF = self.get_banking_angle_at_sn(self.sF_i, self.nF_i)
        self.banking_angleL = self.get_banking_angle_at_sn(self.sL_i, self.nL_i)

        print(f"Animation timeline: {self.tNum} frames, {self.t[-1]:.2f}s")
        print(f"Follower model Z range: {np.min(self.zF_model):.2f} to {np.max(self.zF_model):.2f}")
        print(f"Leader model Z range: {np.min(self.zL_model):.2f} to {np.max(self.zL_model):.2f}")
    
    def init_pygame_opengl(self, width=1600, height=1200):
        """Initialize Pygame with OpenGL"""
        pygame.init()
        display = (width, height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Vehicle Animation - Mouse: Rotate | Scroll: Zoom | Space: Play/Pause")

        self.configure_opengl(width, height)
        print("OpenGL initialized!")

    def configure_opengl(self, width, height):
        """Configure OpenGL state for the active Pygame context."""
        glViewport(0, 0, width, height)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.4, 1))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
        
        glShadeModel(GL_SMOOTH)
        glEnable(GL_NORMALIZE)
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width/height, 1.0, 10000.0)
        glMatrixMode(GL_MODELVIEW)
        
        glClearColor(1.0, 1.0, 1.0, 1)  # White background
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        if not hasattr(self, "track_vertex_vbo") or self.track_vertex_vbo is None:
            self.build_track_vbo()
        if not hasattr(self, "inner_boundary_vbo") or self.inner_boundary_vbo is None:
            self.build_boundary_vbos()

    def render_frame(self, frame_idx=None, handle_input=False):
        """Render one frame into the current OpenGL context."""
        if frame_idx is not None:
            self.current_frame = int(np.clip(frame_idx, 0, self.tNum - 1))
        if handle_input:
            self.handle_input()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.update_camera()
        self.render_track_surface()
        self.render_track_edges()

        trail_start = max(0, self.current_frame - self.trail_length)
        xF_trail = self.xF[trail_start:self.current_frame+1]
        yF_trail = self.yF[trail_start:self.current_frame+1]
        zF_trail = self.zF[trail_start:self.current_frame+1]
        xL_trail = self.xL[trail_start:self.current_frame+1]
        yL_trail = self.yL[trail_start:self.current_frame+1]
        zL_trail = self.zL[trail_start:self.current_frame+1]

        self.render_trail(xF_trail, yF_trail, zF_trail, (1, 0, 0))
        self.render_trail(xL_trail, yL_trail, zL_trail, (0, 0, 1))
        if self.show_diagnostics:
            self.render_diagnostics()

        self.render_car(
            self.xF[self.current_frame],
            self.yF[self.current_frame],
            self.zF[self.current_frame],
            self.carAngleF[self.current_frame],
            self.banking_angleF[self.current_frame],
            (0.8, 0.1, 0.1),
            self.poseF[self.current_frame],
        )
        self.render_car(
            self.xL[self.current_frame],
            self.yL[self.current_frame],
            self.zL[self.current_frame],
            self.carAngleL[self.current_frame],
            self.banking_angleL[self.current_frame],
            (0.1, 0.1, 0.8),
            self.poseL[self.current_frame],
        )

    def read_frame_bgr(self, width, height):
        """Read the current OpenGL framebuffer as a BGR uint8 image."""
        glFlush()
        glFinish()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE)
        if data is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    
    def compute_normal(self, v0, v1, v2):
        """Compute surface normal"""
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal)
        if length > 1e-10:
            normal = normal / length
        return normal
    
    def build_track_vbo(self):
        """Build VBO for track surface"""
        print("Building track VBO...")
        vertices = []
        normals = []
        colors = []

        # Calculate min/max Z for color mapping
        z_min = np.min(self.zMesh)
        z_max = np.max(self.zMesh)
        z_range = z_max - z_min

        for i in range(self.xMesh.shape[0] - 1):
            for j in range(self.yMesh.shape[1] - 1):
                v0 = np.array([self.xMesh[i, j], self.yMesh[i, j], self.zMesh[i, j]], dtype=np.float32)
                v1 = np.array([self.xMesh[i+1, j], self.yMesh[i+1, j], self.zMesh[i+1, j]], dtype=np.float32)
                v2 = np.array([self.xMesh[i+1, j+1], self.yMesh[i+1, j+1], self.zMesh[i+1, j+1]], dtype=np.float32)
                v3 = np.array([self.xMesh[i, j+1], self.yMesh[i, j+1], self.zMesh[i, j+1]], dtype=np.float32)

                n1 = self.compute_normal(v0, v1, v2)
                n2 = self.compute_normal(v0, v2, v3)
                normal = ((n1 + n2) / 2).astype(np.float32)
                length = np.linalg.norm(normal)
                if length > 1e-10:
                    normal = normal / length

                # Calculate average height for this quad
                avg_z = (v0[2] + v1[2] + v2[2] + v3[2]) / 4

                # Map height to grey scale (light grey=low, dark grey=high)
                if z_range > 0:
                    height_ratio = (avg_z - z_min) / z_range
                else:
                    height_ratio = 0.5

                # Create grey scale gradient: light grey to dark grey
                # Lower elevation = lighter grey (higher values: 0.8 -> 0.2)
                # Higher elevation = darker grey (lower values: 0.8 -> 0.2)
                grey_value = 0.8 - (height_ratio * 0.6)  # Range: 0.8 (light) to 0.2 (dark)

                color = np.array([grey_value, grey_value, grey_value, 0.9], dtype=np.float32)

                vertices.extend([v0, v1, v2, v0, v2, v3])
                normals.extend([normal] * 6)
                colors.extend([color] * 6)

        self.track_vertex_vbo = vbo.VBO(np.array(vertices, dtype=np.float32))
        self.track_normal_vbo = vbo.VBO(np.array(normals, dtype=np.float32))
        self.track_color_vbo = vbo.VBO(np.array(colors, dtype=np.float32))
        self.track_vertex_count = len(vertices)
        print(f"Track VBO ready: {self.track_vertex_count} vertices")
        print(f"Height range: {z_min:.2f} to {z_max:.2f} (range: {z_range:.2f})")

    def build_boundary_vbos(self):
        """Build VBOs for track boundaries"""
        inner_verts = [[self.xMesh[i, 0], self.yMesh[i, 0], self.zMesh[i, 0]] 
                       for i in range(self.xMesh.shape[0])]
        outer_verts = [[self.xMesh[i, -1], self.yMesh[i, -1], self.zMesh[i, -1]] 
                       for i in range(self.xMesh.shape[0])]
        start_verts = [[self.xMesh[0, j], self.yMesh[0, j], self.zMesh[0, j] + 0.1] 
                       for j in range(self.yMesh.shape[1])]
        
        self.inner_boundary_vbo = vbo.VBO(np.array(inner_verts, dtype=np.float32))
        self.outer_boundary_vbo = vbo.VBO(np.array(outer_verts, dtype=np.float32))
        self.start_line_vbo = vbo.VBO(np.array(start_verts, dtype=np.float32))
        
        self.inner_boundary_count = len(inner_verts)
        self.outer_boundary_count = len(outer_verts)
        self.start_line_count = len(start_verts)
    
    def render_track_surface(self):
        """Render track using VBOs with height-based colors"""
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        self.track_vertex_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        
        self.track_normal_vbo.bind()
        glNormalPointer(GL_FLOAT, 0, None)
        
        self.track_color_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)
        
        glDrawArrays(GL_TRIANGLES, 0, self.track_vertex_count)
        
        self.track_vertex_vbo.unbind()
        self.track_normal_vbo.unbind()
        self.track_color_vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
    
    def render_track_edges(self):
        """Render track boundaries"""
        glDisable(GL_LIGHTING)
        glEnableClientState(GL_VERTEX_ARRAY)
        
        glColor3f(0.2, 0.2, 0.2) 
        glLineWidth(4)
        self.inner_boundary_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINE_STRIP, 0, self.inner_boundary_count)
        
        self.outer_boundary_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINE_STRIP, 0, self.outer_boundary_count)
        
        glColor3f(1, 0, 0)
        glLineWidth(6)
        self.start_line_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_LINE_STRIP, 0, self.start_line_count)
        
        self.start_line_vbo.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)
        glEnable(GL_LIGHTING)

    def render_line_segments(self, vertices, color, line_width=2):
        """Render independent diagnostic line segments."""
        if len(vertices) == 0:
            return
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glLineWidth(line_width)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glDrawArrays(GL_LINES, 0, len(vertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        glLineWidth(1)
        glEnable(GL_LIGHTING)

    def render_pose_axes(self, pose, line_width=3):
        """Render local x/y/z axes for a display-space pose."""
        axes = pose_axis_segments(pose, axis_scale=self.axis_scale)
        self.render_line_segments(axes[0:2], (1.0, 0.0, 0.0), line_width=line_width)
        self.render_line_segments(axes[2:4], (0.0, 0.65, 0.0), line_width=line_width)
        self.render_line_segments(axes[4:6], (0.0, 0.25, 1.0), line_width=line_width)

    def render_diagnostics(self):
        """Render body-axis and surface-normal diagnostic overlays."""
        self.render_pose_axes(self.poseF[self.current_frame])
        self.render_pose_axes(self.poseL[self.current_frame])
        self.render_line_segments(self.follower_normal_segments, (1.0, 0.45, 0.45), line_width=1)
        self.render_line_segments(self.leader_normal_segments, (0.45, 0.45, 1.0), line_width=1)
    
    def render_car(self, x, y, z, angle, banking_angle, color, pose_matrix=None):
        """Render a car in display space."""
        glPushMatrix()

        # Lift car slightly above surface to prevent z-fighting
        z_offset = 0.05
        if pose_matrix is not None:
            pose = np.array(pose_matrix, dtype=np.float32, copy=True)
            pose[:3, 3] -= pose[:3, 2] * z_offset
            glMultMatrixf(pose.T)
        else:
            glTranslatef(x, y, z + z_offset)
            glRotatef(np.degrees(-angle), 0, 0, 1)
            glRotatef(np.degrees(banking_angle), 1, 0, 0)

        # Set color
        glColor3f(*color)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.car_vertices)
        glNormalPointer(GL_FLOAT, 0, self.car_normals)
        glDrawArrays(GL_QUADS, 0, len(self.car_vertices))
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glPopMatrix()
    
    def render_trail(self, x_trail, y_trail, z_trail, color):
        """Render vehicle trail"""
        if len(x_trail) < 2:
            return
        
        z_offset = 0.3  # Lift trail above track to prevent clipping
            
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glLineWidth(2)
        vertices = trail_geometry(x_trail, y_trail, z_trail, z_offset=z_offset)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)
        glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
        glDisableClientState(GL_VERTEX_ARRAY)
        
        glEnable(GL_LIGHTING)
    
    def update_camera(self):
        """Update camera view based on mode"""
        glLoadIdentity()

        xF = self.xF[self.current_frame]
        yF = self.yF[self.current_frame]
        zF = self.zF[self.current_frame]
        angleF = self.carAngleF[self.current_frame]

        if self.camera_mode == 'follow':
            # Follow camera
            offset_back = 30
            offset_up = 12

            cam_x = xF - offset_back * np.cos(angleF)
            cam_y = yF + offset_back * np.sin(angleF)
            cam_z = zF + offset_up

            look_ahead = 8
            look_x = xF + look_ahead * np.cos(angleF)
            look_y = yF - look_ahead * np.sin(angleF)
            look_z = zF + 2

            gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 0, 1)

        elif self.camera_mode == 'top_down':
            # Top-down camera following the follower
            height = 50  # Height above track
            cam_x = xF
            cam_y = yF
            cam_z = zF + height

            # Look straight down at the car
            look_x = xF
            look_y = yF
            look_z = zF

            gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 
                      np.cos(angleF), -np.sin(angleF), 0)

        elif self.camera_mode == 'rear_view':
            # Rear-view camera
            offset_front = 15  # Position in front of car
            offset_up = 8

            cam_x = xF + offset_front * np.cos(angleF)
            cam_y = yF - offset_front * np.sin(angleF)
            cam_z = zF + offset_up

            # Look behind the car
            look_behind = 20
            look_x = xF - look_behind * np.cos(angleF)
            look_y = yF + look_behind * np.sin(angleF)
            look_z = zF

            gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 0, 1)

        elif self.camera_mode == 'overview':
            # Overview camera - far above showing entire track
            track_center_x = np.mean(self.xMesh)
            track_center_y = np.mean(self.yMesh)
            track_center_z = np.mean(self.zMesh)

            # Calculate bounds to frame the entire track
            track_width = np.max(self.xMesh) - np.min(self.xMesh)
            track_height = np.max(self.yMesh) - np.min(self.yMesh)
            view_distance = max(track_width, track_height) * 1.5

            cam_x = track_center_x
            cam_y = track_center_y
            cam_z = track_center_z + view_distance

            # Look straight down at track center
            look_x = track_center_x
            look_y = track_center_y
            look_z = track_center_z

            gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, look_z, 0, 1, 0)

        else:  # free camera mode
            # Free camera mode (WASD Movement, Mouse scroll to zoom)
            angle_h_rad = np.radians(self.camera_angle_h)
            angle_v_rad = np.radians(self.camera_angle_v)

            cam_x = self.camera_target[0] + self.camera_distance * np.cos(angle_v_rad) * np.sin(angle_h_rad)
            cam_y = self.camera_target[1] + self.camera_distance * np.cos(angle_v_rad) * np.cos(angle_h_rad)
            cam_z = self.camera_target[2] + self.camera_distance * np.sin(angle_v_rad)

            gluLookAt(cam_x, cam_y, cam_z,
                      self.camera_target[0], self.camera_target[1], self.camera_target[2],
                      0, 0, 1)
    
    def handle_input(self):
        """Handle keyboard input"""
        keys = pygame.key.get_pressed()
        
        # Only allow panning in free camera mode
        if self.camera_mode == 'free':
            pan_speed = 15
            if keys[K_w]:
                self.camera_target[1] += pan_speed
            if keys[K_s]:
                self.camera_target[1] -= pan_speed
            if keys[K_a]:
                self.camera_target[0] -= pan_speed
            if keys[K_d]:
                self.camera_target[0] += pan_speed
            if keys[K_q]:
                self.camera_target[2] += pan_speed
            if keys[K_e]:
                self.camera_target[2] -= pan_speed
            if keys[K_r]:
                self.camera_distance = 800
                self.camera_angle_h = 45
                self.camera_angle_v = 30

    
    def run(self):
        """Main animation loop"""
        self.init_pygame_opengl()
        
        clock = pygame.time.Clock()
        running = True
        
        print("\n=== CONTROLS ===")
        print("Space: Play/Pause")
        print("F: Follow camera (behind follower)")
        print("C: Free camera")
        print("Mouse Drag (free mode): Rotate")
        print("Scroll (free mode): Zoom")
        print("WASD/QE (free mode): Pan")
        print("R (free mode): Reset camera")
        print("V: Toggle diagnostics")
        print("ESC: Exit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                        print(f"{'Playing' if self.playing else 'Paused'}")
                    elif event.key == pygame.K_v:
                        self.show_diagnostics = not self.show_diagnostics
                        print(f"Diagnostics {'on' if self.show_diagnostics else 'off'}")
                    elif event.key == pygame.K_1:
                        self.camera_mode = 'follow'
                        print("Follow camera mode")
                    elif event.key == pygame.K_3:
                        self.camera_mode = 'top_down'
                        print("Top-down camera mode")
                    elif event.key == pygame.K_2:
                        self.camera_mode = 'rear_view'
                        print("Rear-view camera mode")
                    elif event.key == pygame.K_4:
                        self.camera_mode = 'overview'
                        print("Overview camera mode")
                    elif event.key == pygame.K_5:
                        self.camera_mode = 'free'
                        print("Free camera mode")
                    elif event.key == pygame.K_UP and not self.playing:
                        # Advance 1 frame when paused
                        self.current_frame = min(self.current_frame + 1, self.tNum - 1)
                        print(f"Frame: {self.current_frame}/{self.tNum-1}")
                    elif event.key == pygame.K_DOWN and not self.playing:
                        # Rewind 1 frame when paused
                        self.current_frame = max(self.current_frame - 1, 0)
                        print(f"Frame: {self.current_frame}/{self.tNum-1}")
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mouse_down = True
                        self.last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4 and self.camera_mode == 'free':
                        self.camera_distance *= 0.9
                        self.camera_distance = max(50, self.camera_distance)
                    elif event.button == 5 and self.camera_mode == 'free':
                        self.camera_distance *= 1.1
                        self.camera_distance = min(5000, self.camera_distance)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.mouse_down = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.mouse_down and self.camera_mode == 'free':
                        current_pos = pygame.mouse.get_pos()
                        dx = current_pos[0] - self.last_mouse_pos[0]
                        dy = current_pos[1] - self.last_mouse_pos[1]
                        
                        self.camera_angle_h += dx * 0.3
                        self.camera_angle_v += dy * 0.3
                        self.camera_angle_v = np.clip(self.camera_angle_v, -89, 89)
                        
                        self.last_mouse_pos = current_pos
            
            self.handle_input()
            
            # Advance animation
            if self.playing:
                self.current_frame = (self.current_frame + 1) % self.tNum
            
            # Get current vehicle states
            xF = self.xF[self.current_frame]
            yF = self.yF[self.current_frame]
            zF = self.zF[self.current_frame]
            angleF = self.carAngleF[self.current_frame]
            bankingF = self.banking_angleF[self.current_frame]
            poseF = self.poseF[self.current_frame]
            
            xL = self.xL[self.current_frame]
            yL = self.yL[self.current_frame]
            zL = self.zL[self.current_frame]
            angleL = self.carAngleL[self.current_frame]
            bankingL = self.banking_angleL[self.current_frame]
            poseL = self.poseL[self.current_frame]
            
            # Trail data
            trail_start = max(0, self.current_frame - self.trail_length)
            xF_trail = self.xF[trail_start:self.current_frame+1]
            yF_trail = self.yF[trail_start:self.current_frame+1]
            zF_trail = self.zF[trail_start:self.current_frame+1]
            
            xL_trail = self.xL[trail_start:self.current_frame+1]
            yL_trail = self.yL[trail_start:self.current_frame+1]
            zL_trail = self.zL[trail_start:self.current_frame+1]
            
            # Render scene
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.update_camera()
            
            # Draw track
            self.render_track_surface()
            self.render_track_edges()
            
            # Draw trails
            self.render_trail(xF_trail, yF_trail, zF_trail, (1, 0, 0))  # Red for follower
            self.render_trail(xL_trail, yL_trail, zL_trail, (0, 0, 1))  # Blue for leader
            
            # Draw cars with banking
            self.render_car(xF, yF, zF, angleF, bankingF, (0.8, 0.1, 0.1), poseF)  # Follower red
            self.render_car(xL, yL, zL, angleL, bankingL, (0.1, 0.1, 0.8), poseL)  # Leader blue
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS for smooth animation
        
        pygame.quit()


def main():
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python script.py <leader.mat> <follower.mat> [track.mat]")
        print("Example: python script.py Leader.mat SimResult.mat NASCAR_Track_Monge_v3.mat")
        sys.exit(1)
    
    leader_file = sys.argv[1]
    follower_file = sys.argv[2]
    track_file = sys.argv[3] if len(sys.argv) > 3 else "NASCAR_Track_Monge_v3.mat"
    
    try:
        animator = Vehicle3DAnimatorGL(leader_file, follower_file, track_file)
        animator.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
