import os
import sys

# Keep SDL/PyOpenGL on the same Linux display stack. This avoids PyOpenGL
# looking for an EGL context while Pygame has created an X11/GLX context.
if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

class Vehicle3DAnimatorGL:
    def __init__(self, leader_file, follower_file, track_file):
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
        self.rwTrack = auxdata.track.rw / self.lengthscale
        
        # Track centerline
        self.xc = auxdata.track.xc
        self.yc = auxdata.track.yc * -1
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

        # VBO placeholders
        self.track_vbo = None
        self.boundary_vbos = None
        
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
        
        zMesh = np.zeros((len(self.s), nlat))
        for k in range(nlat):
            nk = n_spaced[:, k]
            zMesh[:, k] = self.z0 + self.z1 * nk + self.z2 * nk**2 + self.z3 * nk**3
        
        xMesh = self.track_xc[:, np.newaxis] - n_spaced * np.sin(self.psi[:, np.newaxis])
        yMesh = self.track_yc[:, np.newaxis] + n_spaced * np.cos(self.psi[:, np.newaxis])
        
        yMesh = -yMesh  # SAE coordinate transform
        zMesh = -zMesh * self.z_scale  # SAE coordinate transform

        return xMesh, yMesh, zMesh, n_spaced
    
    def _process_vehicle_data(self):
        """Extract vehicle states from simulation data"""
        # Leader states
        self.sL = self.leader["output"].result.interpsolution.phase.time / self.lengthscale
        statesL = self.leader["output"].result.interpsolution.phase.state
        
        self.nL = statesL[:, 0] / self.lengthscale
        self.xiL = statesL[:, 1]
        self.vL = statesL[:, 2] / self.velscale
        self.tL = statesL[:, 8] / self.timescale
        
        # Follower states
        self.sF = self.follower["output"].result.interpsolution.phase.time / self.lengthscale
        statesF = self.follower["output"].result.interpsolution.phase.state
        
        self.nF = statesF[:, 0] / self.lengthscale
        self.xiF = statesF[:, 1]
        self.vF = statesF[:, 2] / self.velscale
        self.tF = statesF[:, 8] / self.timescale
        
        print(f"Leader: {len(self.sL)} points, Follower: {len(self.sF)} points")
    
    def get_elevation_at_xy(self, x, y):
        """Get track elevation at world coordinates"""
        x = np.asarray(x)
        y = np.asarray(y)
        
        def interpolate_point(xi, yi):
            '''bilinear interpolation'''
            distances = (self.xMesh - xi)**2 + (self.yMesh - yi)**2
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            i_closest, j_closest = min_idx
            
            search_radius = 3
            i_min = max(0, i_closest - search_radius)
            i_max = min(self.xMesh.shape[0] - 1, i_closest + search_radius)
            j_min = max(0, j_closest - search_radius)
            j_max = min(self.yMesh.shape[1] - 1, j_closest + search_radius)
            
            best_zi = None
            min_extrapolation = float('inf')
            
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    x00, y00, z00 = self.xMesh[i, j], self.yMesh[i, j], self.zMesh[i, j]
                    x01, y01, z01 = self.xMesh[i, j+1], self.yMesh[i, j+1], self.zMesh[i, j+1]
                    x10, y10, z10 = self.xMesh[i+1, j], self.yMesh[i+1, j], self.zMesh[i+1, j]
                    x11, y11, z11 = self.xMesh[i+1, j+1], self.yMesh[i+1, j+1], self.zMesh[i+1, j+1]
                    
                    u = ((xi - x00) * (x10 - x00) + (yi - y00) * (y10 - y00)) / \
                        ((x10 - x00)**2 + (y10 - y00)**2 + 1e-10)
                    v = ((xi - x00) * (x01 - x00) + (yi - y00) * (y01 - y00)) / \
                        ((x01 - x00)**2 + (y01 - y00)**2 + 1e-10)
                    
                    u_clamped = np.clip(u, 0, 1)
                    v_clamped = np.clip(v, 0, 1)
                    extrapolation = abs(u - u_clamped) + abs(v - v_clamped)
                    
                    zi = (1 - u_clamped) * (1 - v_clamped) * z00 + \
                         u_clamped * (1 - v_clamped) * z10 + \
                         (1 - u_clamped) * v_clamped * z01 + \
                         u_clamped * v_clamped * z11
                    
                    if extrapolation < min_extrapolation:
                        min_extrapolation = extrapolation
                        best_zi = zi
            
            if best_zi is not None:
                return best_zi
            return self.zMesh[i_closest, j_closest]
        
        if x.ndim == 0:
            return interpolate_point(float(x), float(y))
        else:
            z = np.zeros_like(x, dtype=float)
            for i, (xi, yi) in enumerate(zip(x, y)):
                z[i] = interpolate_point(float(xi), float(yi))
            return z
    
    def get_banking_angle_at_sn(self, s, n):
        """Get banking angle at track coordinates (s,n)"""
        s = np.asarray(s)
        n = np.asarray(n)

        if s.ndim == 0:
            z1 = np.interp(s, self.s, self.z1)
            z2 = np.interp(s, self.s, self.z2)
            z3 = np.interp(s, self.s, self.z3)
            # Apply z-scaling to the slope coefficients
            z1_scaled = z1 * self.z_scale
            z2_scaled = z2 * self.z_scale
            z3_scaled = z3 * self.z_scale
            return np.arctan(z1_scaled + 2*z2_scaled*n + 3*z3_scaled*n**2)
        else:
            banking_rad = np.zeros_like(s)
            for i, (si, ni) in enumerate(zip(s, n)):
                z1 = np.interp(si, self.s, self.z1)
                z2 = np.interp(si, self.s, self.z2)
                z3 = np.interp(si, self.s, self.z3)
                # Apply z-scaling to the slope coefficients
                z1_scaled = z1 * self.z_scale
                z2_scaled = z2 * self.z_scale
                z3_scaled = z3 * self.z_scale
                banking_rad[i] = np.arctan(z1_scaled + 2*z2_scaled*ni + 3*z3_scaled*ni**2)
            return banking_rad
    
    def _prepare_common_timeline(self):
        """Interpolate vehicle data to common timeline"""
        # Follower trajectory
        xcF = np.interp(self.sF, self.sTrack, self.xc)
        ycF = np.interp(self.sF, self.sTrack, self.yc)
        psiF = np.interp(self.sF, self.sTrack, self.psiTrack)
        
        self.xF_orig = xcF - self.nF * np.sin(psiF)
        self.yF_orig = ycF + self.nF * np.cos(psiF)
        self.zF_orig = self.get_elevation_at_xy(self.xF_orig, self.yF_orig)
        self.carAngleF_orig = np.unwrap(psiF + self.xiF)
        
        # Leader trajectory
        xcL = np.interp(self.sL, self.sTrack, self.xc)
        ycL = np.interp(self.sL, self.sTrack, self.yc)
        psiL = np.interp(self.sL, self.sTrack, self.psiTrack)
        
        self.xL_orig = xcL - self.nL * np.sin(psiL)
        self.yL_orig = ycL + self.nL * np.cos(psiL)
        self.zL_orig = self.get_elevation_at_xy(self.xL_orig, self.yL_orig)
        self.carAngleL_orig = np.unwrap(psiL + self.xiL)
        
        # Banking angles
        self.banking_angleF_orig = np.array([
            self.get_banking_angle_at_sn(s, n) for s, n in zip(self.sF, self.nF)
        ])
        self.banking_angleL_orig = np.array([
            self.get_banking_angle_at_sn(s, n) for s, n in zip(self.sL, self.nL) 
        ])
        
        # Resample to common timeline
        self.tNum = len(self.tF)
        self.t = np.linspace(0, min(self.tF[-1], self.tL[-1]), self.tNum)
        
        self.xF = np.interp(self.t, self.tF, self.xF_orig)
        self.yF = np.interp(self.t, self.tF, self.yF_orig)
        self.zF = np.interp(self.t, self.tF, self.zF_orig)
        self.carAngleF = np.interp(self.t, self.tF, self.carAngleF_orig)
        self.banking_angleF = np.interp(self.t, self.tF, self.banking_angleF_orig)
        
        self.xL = np.interp(self.t, self.tL, self.xL_orig)
        self.yL = np.interp(self.t, self.tL, self.yL_orig)
        self.zL = np.interp(self.t, self.tL, self.zL_orig)
        self.carAngleL = np.interp(self.t, self.tL, self.carAngleL_orig)
        self.banking_angleL = np.interp(self.t, self.tL, self.banking_angleL_orig)
        
        # Smoothing filter for z movement
        smoothing_window = 5
        self.zF = uniform_filter1d(self.zF, size=smoothing_window, mode='nearest')
        self.zL = uniform_filter1d(self.zL, size=smoothing_window, mode='nearest')
        
        # SAE coordinate system (invert z)
        #self.zF = self.zF * -1
        #self.zL = self.zL * -1 
        
        print(f"Animation timeline: {self.tNum} frames, {self.t[-1]:.2f}s")
        print(f"Follower Z range: {np.min(self.zF):.2f} to {np.max(self.zF):.2f}")
        print(f"Leader Z range: {np.min(self.zL):.2f} to {np.max(self.zL):.2f}")
    
    def init_pygame_opengl(self, width=1600, height=1200):
        """Initialize Pygame with OpenGL"""
        pygame.init()
        display = (width, height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Vehicle Animation - Mouse: Rotate | Scroll: Zoom | Space: Play/Pause")
        
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
        gluPerspective(45, width/height, 1.0, 10000.0)
        glMatrixMode(GL_MODELVIEW)
        
        glClearColor(1.0, 1.0, 1.0, 1)  # White background
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.build_track_vbo()
        self.build_boundary_vbos()
        print("OpenGL initialized!")
    
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
    
    def render_car(self, x, y, z, angle, banking_angle, color):
        """Render a car with banking"""
        glPushMatrix()
        
        # Lift car slightly above surface to prevent z-fighting
        z_offset = 0.0
        
        # Translate to position
        glTranslatef(x, y, z + z_offset)
        
        # Rotate for heading
        glRotatef(np.degrees(-angle), 0, 0, 1)
        
        # Rotate for banking 
        glRotatef(np.degrees(banking_angle), 1, 0, 0)
        
        # Car dimensions
        car_length = (self.a + self.b) / self.lengthscale
        car_width = 1.8
        car_height = 0.6
        
        # Set color
        glColor3f(*color)
        
        # Draw car body 
        glBegin(GL_QUADS)
        
        #Initilise Vertices
        # Bottom
        glNormal3f(0, 0, -1)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, 0)
        glVertex3f(self.a/self.lengthscale, -car_width/2, 0)
        glVertex3f(self.a/self.lengthscale, car_width/2, 0)
        glVertex3f(-self.b/self.lengthscale, car_width/2, 0)
        
        # Top
        glNormal3f(0, 0, 1)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, car_height)
        glVertex3f(-self.b/self.lengthscale, car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, -car_width/2, car_height)
        
        # Front
        glNormal3f(1, 0, 0)
        glVertex3f(self.a/self.lengthscale, -car_width/2, 0)
        glVertex3f(self.a/self.lengthscale, -car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, car_width/2, 0)
        
        # Rear
        glNormal3f(-1, 0, 0)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, 0)
        glVertex3f(-self.b/self.lengthscale, car_width/2, 0)
        glVertex3f(-self.b/self.lengthscale, car_width/2, car_height)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, car_height)
        
        # Left
        glNormal3f(0, -1, 0)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, 0)
        glVertex3f(-self.b/self.lengthscale, -car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, -car_width/2, car_height)
        glVertex3f(self.a/self.lengthscale, -car_width/2, 0)
        
        # Right
        glNormal3f(0, 1, 0)
        glVertex3f(-self.b/self.lengthscale, car_width/2, 0)
        glVertex3f(self.a/self.lengthscale, car_width/2, 0)
        glVertex3f(self.a/self.lengthscale, car_width/2, car_height)
        glVertex3f(-self.b/self.lengthscale, car_width/2, car_height)
        
        glEnd()
        
        glPopMatrix()
    
    def render_trail(self, x_trail, y_trail, z_trail, color):
        """Render vehicle trail"""
        if len(x_trail) < 2:
            return
        
        z_offset = 0.3  # Lift trail above track to prevent clipping
            
        glDisable(GL_LIGHTING)
        glColor3f(*color)
        glLineWidth(2)
        
        glBegin(GL_LINE_STRIP)
        for i in range(len(x_trail)):
            glVertex3f(x_trail[i], y_trail[i], z_trail[i] + z_offset)
        glEnd()
        
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
            
            xL = self.xL[self.current_frame]
            yL = self.yL[self.current_frame]
            zL = self.zL[self.current_frame]
            angleL = self.carAngleL[self.current_frame]
            bankingL = self.banking_angleL[self.current_frame]
            
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
            self.render_car(xF, yF, zF, angleF, bankingF, (0.8, 0.1, 0.1))  # Follower red
            self.render_car(xL, yL, zL, angleL, bankingL, (0.1, 0.1, 0.8))  # Leader blue
            
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
