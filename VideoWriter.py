# VideoWriter.py
import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import threading
import time
import sys
import os
from Stackelberg_HUD import DataProcessor, PrecomputedData, TelemetryDashboard

try:
    import cv2
except ImportError:
    cv2 = None

class HeadlessAnimationRecorder:
    def __init__(self, leader_file, follower_file, track_file):
        self.leader_file = leader_file
        self.follower_file = follower_file
        self.track_file = track_file
        
        # Animation state
        self.current_frame = 0
        self.total_frames = 0
        
        # Video recording state
        self.recording = True  # Always recording in headless mode
        self.video_frames = []
        self.output_file = "stackelberg_race_headless.mp4"
        self.fps = 30
        
        # Initialize data processor
        print("Loading data...")
        self.data_processor = DataProcessor(leader_file, follower_file, track_file)
        self.total_frames = len(self.data_processor.t)
        
        # Precomputed data
        self.precomputed_data = PrecomputedData(self.data_processor)
        self.precomputed_data.precompute_all()
        
        # HUD components
        self.fig = None
        self.mini_carF = None
        self.mini_carL = None
        self.spokes_lines = []
        self.center_mark = None
        self.steer_text = None
        self.gg_leader_dot = None
        self.gg_follower_dot = None
        self.dash_vals = {}
        self.bar_throttle = None
        self.bar_ax = None
        self.bar_ay = None
        self.front_dot_F = None
        self.front_dot_L = None
        self.rear_dot_F = None
        self.rear_dot_L = None
        
        # Pygame/OpenGL state
        self.pygame_initialized = False
        self.pygame_animator = None

    def setup_headless_hud(self):
        """Setup the HUD components without displaying the window"""
        print("Setting up HUD...")

        # Use Agg backend for headless plotting
        plt.switch_backend('Agg')

        # Create figure with the same dimensions as before
        self.fig = plt.figure(figsize=(19.2, 10.8), facecolor='#1a1a1a')

        # Setup all HUD components (same as before but headless)
        self._setup_minimap()
        self._setup_steering_wheel()
        self._setup_gg_diagram()
        self._setup_dashboard()
        self._setup_telemetry_bars()
        self._setup_friction_circles()

        # Setup the animation area for Pygame overlay
        self._setup_animation_area()

        # Tight layout for consistent rendering
        #plt.tight_layout()

        # Force initial render to set up the canvas properly
        self.fig.canvas.draw()
        print("HUD setup complete")
    
    def _setup_animation_area(self):
        """Setup the animation display area for Pygame overlay"""
        self.animation_ax = self.fig.add_axes([0.35, 0.05, 0.40, 0.90])  # Same position as before
        self.animation_ax.set_facecolor('#000000')  # Black background for 3D view
        self.animation_ax.set_xticks([])
        self.animation_ax.set_yticks([])
        self.animation_ax.set_title('3D Vehicle Animation', 
                                   fontsize=16, color='white', fontweight='bold', pad=20)

        # Draw the border
        for spine in self.animation_ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

    def setup_headless_pygame(self):
        """Setup Pygame/OpenGL in headless mode"""
        try:
            from Stackleberg_3DAnimator import Vehicle3DAnimatorGL

            print("Setting up OpenGL...")
            self.pygame_animator = Vehicle3DAnimatorGL(
                self.leader_file, 
                self.follower_file, 
                self.track_file
            )

            # Initialize Pygame with proper display size matching the animation area
            pygame.init()

            hud_width_pixels = 1920  # Full HD width
            hud_height_pixels = 1080  # Full HD height

            # Calculate the animation area size in pixels
            area_width = int(0.43 * hud_width_pixels)   # 688 pixels
            area_height = int(0.90 * hud_height_pixels) # 900 pixels

            # Use the calculated size for Pygame
            display_size = (area_width, area_height)  # 688x900

            # Create offscreen buffer
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL | pygame.HIDDEN)

            self.pygame_animator.configure_opengl(*display_size)

            # Set camera to follower perspective
            self.pygame_animator.camera_mode = 'follow'

            self.pygame_initialized = True
            print("OpenGL setup complete")

        except Exception as e:
            print(f"Headless Pygame initialization error: {e}")
            import traceback
            traceback.print_exc()

    # KEEP ALL YOUR EXISTING HUD SETUP METHODS (they work the same headless)
    def _setup_minimap(self):
        """Setup minimap component (same as before)"""
        ax_minimap = self.fig.add_axes([0.005, 0.50, 0.15, 0.4])
        ax_minimap.set_title('Track Overview', fontsize=14, color='white', fontweight='bold', pad=12)
        ax_minimap.set_facecolor('#0a0a0a')
        ax_minimap.set_aspect('equal')
        ax_minimap.grid(True, alpha=0.15, color='#333333')
        ax_minimap.set_xticks([])
        ax_minimap.set_yticks([])
        for spine in ax_minimap.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

        # Full track on minimap
        ax_minimap.plot(self.data_processor.xc, self.data_processor.yc, color='#666666', linestyle=':', linewidth=2.0, alpha=0.5)
        ax_minimap.plot(self.data_processor.x_inner, self.data_processor.y_inner, color='#00ff00', linewidth=2.5, alpha=0.6)
        ax_minimap.plot(self.data_processor.x_outer, self.data_processor.y_outer, color='#00ff00', linewidth=2.5, alpha=0.6)

        xmin = min(np.min(self.data_processor.x_inner), np.min(self.data_processor.x_outer)) - 10
        xmax = max(np.max(self.data_processor.x_inner), np.max(self.data_processor.x_outer)) + 10
        ymin = min(np.min(self.data_processor.y_inner), np.min(self.data_processor.y_outer)) - 10
        ymax = max(np.max(self.data_processor.y_inner), np.max(self.data_processor.y_outer)) + 10
        ax_minimap.set_xlim(xmin, xmax)
        ax_minimap.set_ylim(ymin, ymax)
        ax_minimap.invert_yaxis()

        # Car dots on minimap
        self.mini_carF, = ax_minimap.plot([], [], 'o', color='#ff3333', markersize=16, markeredgecolor='white', markeredgewidth=3)
        self.mini_carL, = ax_minimap.plot([], [], 'o', color='#3366ff', markersize=16, markeredgecolor='white', markeredgewidth=3)


    def _setup_steering_wheel(self):
        """Setup steering wheel component with larger fonts"""
        ax_wheel = self.fig.add_axes([0.785, 0.75, 0.17, 0.21])
        ax_wheel.set_title('Steering (Follower)', fontsize=14, color='#ff3333', fontweight='bold', pad=12)
        ax_wheel.set_facecolor('#0a0a0a')
        ax_wheel.set_aspect('equal')
        ax_wheel.set_xlim(-1.15, 1.15)
        ax_wheel.set_ylim(-1.15, 1.15)
        ax_wheel.set_xticks([])
        ax_wheel.set_yticks([])
        for spine in ax_wheel.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

        # Professional 3D-style steering wheel
        import matplotlib.patches as patches
        wheel_outer = patches.Circle((0, 0), 1.0, fill=False, edgecolor='#cccccc', linewidth=6)
        wheel_inner = patches.Circle((0, 0), 0.88, fill=False, edgecolor='#999999', linewidth=4)
        # FIX: Remove color parameter to avoid warning
        wheel_hub = patches.Circle((0, 0), 0.18, facecolor='#333333', edgecolor='#666666', linewidth=3)
        ax_wheel.add_patch(wheel_outer)
        ax_wheel.add_patch(wheel_inner)
        ax_wheel.add_patch(wheel_hub)

        # Three spokes
        base_spokes = np.deg2rad([90, 210, 330])
        spoke_len = 0.82
        for ang in base_spokes:
            x = np.array([0.0, spoke_len*np.cos(ang)])
            y = np.array([0.0, spoke_len*np.sin(ang)])
            ln, = ax_wheel.plot(x, y, color='#aaaaaa', linewidth=7, solid_capstyle='round')
            self.spokes_lines.append(ln)

        # Center indicator mark
        self.center_mark = ax_wheel.plot([0, 0], [0.22, 0.35], color='#ff0000', linewidth=6, solid_capstyle='round')[0]

        # Steering angle text
        self.steer_text = ax_wheel.text(0, -0.65, '', fontsize=16, color='#ffffff', ha='center', 
                                fontweight='bold', family='monospace',
                                bbox=dict(boxstyle='round,pad=0.8', facecolor='#2a2a2a', edgecolor='#555555', linewidth=3))
    
    def _setup_gg_diagram(self):
        """Setup G-G diagram component - UPDATED FOR G-FORCES AND 4 DOTS"""
        ax_gg = self.fig.add_axes([0.05, 0.12, 0.21, 0.26])
        ax_gg.set_facecolor('#0a0a0a')

        # Combine all data and convert to g-forces
        G_CONVERSION = 1.0 / 9.8  # Convert m/s² to g
        
        all_ax = np.concatenate([self.data_processor.ax_F_interp, self.data_processor.ax_L_interp, 
                                self.data_processor.ax_F_direct_interp, self.data_processor.ax_L_direct_interp]) * G_CONVERSION
        all_ay = np.concatenate([self.data_processor.ay_F_interp, self.data_processor.ay_L_interp,
                                self.data_processor.ay_F_direct_interp, self.data_processor.ay_L_direct_interp]) * G_CONVERSION

        # Use rolling window to find sustained maximum (not transient spikes)
        window_size = 100
        def rolling_max(data, window):
            return np.array([np.max(np.abs(data[i:i+window])) for i in range(0, len(data)-window+1, window)])

        if len(all_ax) > window_size:
            AX_MAX = np.percentile(rolling_max(all_ax, window_size), 90) + 1e-6
            AY_MAX = np.percentile(rolling_max(all_ay, window_size), 90) + 1e-6
        else:
            AX_MAX = np.percentile(np.abs(all_ax), 95) + 1e-6
            AY_MAX = np.percentile(np.abs(all_ay), 95) + 1e-6

        # Set limits in g-forces
        AX_LIM = max(0.5, min(2.5, 1.2 * AX_MAX))  # Reasonable g-force limits
        AY_LIM = max(0.5, min(2.5, 1.2 * AY_MAX))

        ax_gg.set_xlim(-AY_LIM*1.1, AY_LIM*1.1)
        ax_gg.set_ylim(-AX_LIM*1.1, AX_LIM*1.1)

        # Update labels for g-forces
        ax_gg.set_xlabel('← Lateral → (g)', fontsize=12, color='white', fontweight='bold')
        ax_gg.set_ylabel('← Longitudinal (g) →  ', fontsize=12, color='white', fontweight='bold')
        ax_gg.set_title('G-G Diagram', fontsize=14, color='white', fontweight='bold', pad=12)
        ax_gg.grid(True, alpha=0.2, color='#333333')
        ax_gg.axhline(0, color='#666666', linewidth=2)
        ax_gg.axvline(0, color='#666666', linewidth=2)
        ax_gg.tick_params(colors='white', labelsize=11)
        for spine in ax_gg.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

        # Current position dots - 4 DOTS LIKE ANIMATOR
        # Solid for direct data, hollow for numerical
        self.gg_leader_direct, = ax_gg.plot([], [], 'o', color='#3366ff', markersize=10, markeredgecolor='white', markeredgewidth=2)
        self.gg_follower_direct, = ax_gg.plot([], [], 'o', color='#ff3333', markersize=10, markeredgecolor='white', markeredgewidth=2)

        ax_gg.legend([self.gg_leader_direct, self.gg_follower_direct], 
                     ['Leader', 'Follower'], 
                     loc='upper right', fontsize=9, facecolor='#2a2a2a', edgecolor='#555555', labelcolor='white')
    
    def _setup_dashboard(self):
        """Setup main dashboard display with larger fonts"""
        ax_dash = self.fig.add_axes([0.785, 0.34, 0.21, 0.37])
        ax_dash.set_facecolor('#0a0a0a')
        ax_dash.set_xticks([])
        ax_dash.set_yticks([])
        for spine in ax_dash.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

        # Dashboard text elements with larger fonts
        self.dash_vals = {
            "time": ax_dash.text(0.50, 0.85, "", transform=ax_dash.transAxes, fontsize=22, 
                                 color='#00ff00', ha='center', fontweight='bold', family='monospace'),
            "spdF": ax_dash.text(0.50, 0.72, "", transform=ax_dash.transAxes, fontsize=18, 
                                 color='#ff3333', ha='center', fontweight='bold', family='monospace'),
            "spdL": ax_dash.text(0.50, 0.60, "", transform=ax_dash.transAxes, fontsize=18, 
                                 color='#3366ff', ha='center', fontweight='bold', family='monospace'),
            "gapS": ax_dash.text(0.50, 0.44, "", transform=ax_dash.transAxes, fontsize=16, 
                                 color='#ffff00', ha='center', fontweight='bold', family='monospace'),
            "gapD": ax_dash.text(0.50, 0.32, "", transform=ax_dash.transAxes, fontsize=14, 
                                 color='#aaaaaa', ha='center', fontweight='normal', family='monospace'),
            "p1": ax_dash.text(0.50, 0.12, "", transform=ax_dash.transAxes, fontsize=16, 
                               color='#ffffff', ha='center', fontweight='bold'),
        }

        # Labels with larger fonts
        ax_dash.text(0.27, 0.86, "TIME", transform=ax_dash.transAxes, fontsize=12, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.05, 0.72, "F:", transform=ax_dash.transAxes, fontsize=15, 
                     color='#ff3333', ha='left', fontweight='bold')
        ax_dash.text(0.05, 0.60, "L:", transform=ax_dash.transAxes, fontsize=15, 
                     color='#3366ff', ha='left', fontweight='bold')
        ax_dash.text(0.50, 0.51, "GAP (Arc Length)", transform=ax_dash.transAxes, fontsize=12, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.50, 0.22, "POSITION", transform=ax_dash.transAxes, fontsize=12, 
                     color='#888888', ha='center', fontweight='bold')
    
    def _setup_telemetry_bars(self):
        """Setup telemetry bars with larger fonts"""
        ax_bars = self.fig.add_axes([0.785, 0.06, 0.21, 0.24])
        ax_bars.set_title('Accel & Input (Follower: Red, Leader: Blue)', fontsize=12, color='white', fontweight='bold', pad=12)
        ax_bars.set_facecolor('#0a0a0a')
        ax_bars.set_xlim(0, 1)
        ax_bars.set_ylim(-1.0, 1.0)
        ax_bars.set_xticks([])
        ax_bars.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax_bars.set_yticklabels(['-100%', '-50%', '0', '+50%', '+100%'], fontsize=10, color='white')
        ax_bars.axhline(0, color='#666666', linewidth=3)
        ax_bars.grid(True, axis='y', alpha=0.2, color='#333333')
        for spine in ax_bars.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(3)

        # Bar positions
        bar_x = [0.2, 0.50, 0.80]
        bar_width = 0.15
        bar_half = bar_width / 2
        bar_offset = 0.04

        # Create dual bars
        def make_dual_bar(x_pos, label_text):
            # Follower (left side, red)
            f_pos = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#ff3333',
                                edgecolor='white', linewidth=2)[0]
            f_neg = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#990000',
                                edgecolor='white', linewidth=2)[0]
            
            # Leader (right side, blue)
            l_pos = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#3366ff',
                                edgecolor='white', linewidth=2)[0]
            l_neg = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#000099',
                                edgecolor='white', linewidth=2)[0]
            
            # Label
            ax_bars.text(x_pos, -1.15, label_text, ha='center', va='top', fontsize=11, 
                         color='white', fontweight='bold')
            
            return (f_pos, f_neg, l_pos, l_neg)

        # Create all bars
        self.bar_throttle = make_dual_bar(bar_x[0], 'Throttle/\nBrake')
        self.bar_ax = make_dual_bar(bar_x[1], 'Accel X')
        self.bar_ay = make_dual_bar(bar_x[2], 'Accel Y')
    
    def _setup_friction_circles(self):
        """Setup friction circles (same as animator)"""
        ax_fc_front = self.fig.add_axes([0.15, 0.75, 0.20, 0.20])
        ax_fc_rear = self.fig.add_axes([0.15, 0.45, 0.20, 0.20])

        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_facecolor('#0a0a0a')
            ax_fc.grid(True, alpha=0.3, color='#333')
            ax_fc.set_aspect('equal', adjustable='box')
            ax_fc.tick_params(colors='white', labelsize=11)
            for sp in ax_fc.spines.values():
                sp.set_edgecolor('#555')
                sp.set_linewidth(3)

        # Apply scaling to get forces in kN (divide by 1000)
        scale_factor = 1.0 / self.data_processor.forcescale / 1000.0  # Convert to kN

        # Calculate limits in kN
        FCX = float(np.nanmax(np.abs([
            self.data_processor.Fxmax_f_F * scale_factor, 
            self.data_processor.Fxmax_f_L * scale_factor,
            self.data_processor.Fxmax_r_F * scale_factor, 
            self.data_processor.Fxmax_r_L * scale_factor
        ])) + 1e-6)

        FCY = float(np.nanmax(np.abs([
            self.data_processor.Fymax_f_F * scale_factor,
            self.data_processor.Fymax_f_L * scale_factor, 
            self.data_processor.Fymax_r_F * scale_factor,
            self.data_processor.Fymax_r_L * scale_factor
        ])) + 1e-6)

        # Set limits to nice round numbers
        FCX_lim = max(5, np.ceil(FCX))
        FCY_lim = max(5, np.ceil(FCY))

        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_xlim(-1.1*FCX_lim, 1.1*FCX_lim)
            ax_fc.set_ylim(-1.1*FCY_lim, 1.1*FCY_lim)

        # Store the scale factor for use in updates
        self.friction_scale_factor = scale_factor

        # Add multiplier indicator to titles
        ax_fc_front.set_title(f'Front Axle', color='white', fontsize=16, fontweight='bold')
        ax_fc_front.set_xlabel('← Lateral (KN) →', color='white', fontsize=12)
        ax_fc_front.set_ylabel('← Longitudinal (KN) →', color='white', fontsize=12)

        ax_fc_rear.set_title(f'Rear Axle', color='white', fontsize=16, fontweight='bold')
        ax_fc_rear.set_xlabel('← Lateral (KN) →', color='white', fontsize=12)
        ax_fc_rear.set_ylabel('← Longitudinal (KN) →', color='white', fontsize=12)

        # Artists to update each frame
        self.front_dot_F, = ax_fc_front.plot([], [], 'o', color='#ff3333', mec='white', mew=2, ms=10, label='Follower')
        self.front_dot_L, = ax_fc_front.plot([], [], 'o', color='#3366ff', mec='white', mew=2, ms=10, label='Leader')
        ax_fc_front.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white', fontsize=11)

        self.rear_dot_F, = ax_fc_rear.plot([], [], 'o', color='#ff3333', mec='white', mew=2, ms=10, label='Follower')
        self.rear_dot_L, = ax_fc_rear.plot([], [], 'o', color='#3366ff', mec='white', mew=2, ms=10, label='Leader')
        ax_fc_rear.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white', fontsize=11)

    def _update_hud_components(self, data_idx):
        """Update all HUD components - UPDATED FOR GG DIAGRAM CHANGES"""
        # Update minimap
        xF, yF, xL, yL = self.precomputed_data.minimap_data[data_idx]
        self.mini_carF.set_data([xF], [yF])
        self.mini_carL.set_data([xL], [yL])

        # Update steering wheel
        spoke_data, center_mark_data, steer_deg = self.precomputed_data.steering_data[data_idx]
        for ln, (x_data, y_data) in zip(self.spokes_lines, spoke_data):
            ln.set_data(x_data, y_data)
        self.center_mark.set_data(*center_mark_data)
        self.steer_text.set_text(f'{steer_deg:+.1f}°')

        # Update GG diagram - CONVERT TO G-FORCES AND HANDLE 4 DOTS
        G_CONVERSION = 1.0 / 9.8
        
        ay_L_dir, ax_L_dir, ay_F_dir, ax_F_dir = self.precomputed_data.gg_data[data_idx]
        
        # Convert all to g-forces
        ay_L_dir *= G_CONVERSION
        ax_L_dir *= G_CONVERSION
        ay_F_dir *= G_CONVERSION
        ax_F_dir *= G_CONVERSION

        # Direct data (solid circles)
        self.gg_leader_direct.set_data([ay_L_dir], [ax_L_dir])
        self.gg_follower_direct.set_data([ay_F_dir], [ax_F_dir])

        # Update dashboard
        time_val, spdF_val, spdL_val, gap_s_val, gap_xy_val = self.precomputed_data.dashboard_data[data_idx]
        self.dash_vals["time"].set_text(f"{time_val:05.2f}s")
        self.dash_vals["spdF"].set_text(f"{spdF_val:6.2f} m/s")
        self.dash_vals["spdL"].set_text(f"{spdL_val:6.2f} m/s")
        
        gap_color = '#00ff00' if gap_s_val > 0 else '#ff3333'
        self.dash_vals["gapS"].set_text(f"{gap_s_val:+.2f} m")
        self.dash_vals["gapS"].set_color(gap_color)

        if gap_s_val > 0:
            self.dash_vals["p1"].set_text("P1: Follower  P2: Leader")
            self.dash_vals["p1"].set_color('#00ff00')
        else:
            self.dash_vals["p1"].set_text("P1: Leader  P2: Follower")
            self.dash_vals["p1"].set_color('#ffaa00')

        # Update bars
        norm_ax_F, norm_ax_L, norm_ay_F, norm_ay_L = self.precomputed_data.bar_data[data_idx]
        self._update_dual_bar_fast(self.bar_throttle, norm_ax_F, norm_ax_L)
        self._update_dual_bar_fast(self.bar_ax, norm_ax_F, norm_ax_L)
        self._update_dual_bar_fast(self.bar_ay, norm_ay_F, norm_ay_L)

        # Update friction circles
        Ffy_F, Ffx_F, Ffy_L, Ffx_L, Fry_F, Frx_F, Fry_L, Frx_L = self.precomputed_data.friction_data[data_idx]

        # Apply scaling to convert to physical units (Newtons)
        scale_factor = getattr(self, 'friction_scale_factor', 1.0)

        self.front_dot_F.set_data([Ffy_F * scale_factor], [Ffx_F * scale_factor])
        self.front_dot_L.set_data([Ffy_L * scale_factor], [Ffx_L * scale_factor])
        self.rear_dot_F.set_data([Fry_F * scale_factor], [Frx_F * scale_factor])
        self.rear_dot_L.set_data([Fry_L * scale_factor], [Frx_L * scale_factor])
    
    def _update_dual_bar_fast(self, bars, f_val, l_val):
        """Update dual bar display (same as before)"""
        f_pos, f_neg, l_pos, l_neg = bars
        
        # Follower
        if f_val >= 0:
            f_pos.set_height(f_val)
            f_pos.set_y(0)
            f_neg.set_height(0)
        else:
            f_neg.set_height(-f_val)
            f_neg.set_y(f_val)
            f_pos.set_height(0)
        
        # Leader
        if l_val >= 0:
            l_pos.set_height(l_val)
            l_pos.set_y(0)
            l_neg.set_height(0)
        else:
            l_neg.set_height(-l_val)
            l_neg.set_y(l_val)
            l_pos.set_height(0)

    def _render_pygame_frame_headless(self, frame_idx):
        """Render a single Pygame frame in headless mode"""
        if not self.pygame_initialized:
            return np.zeros((972, 825, 3), dtype=np.uint8)  # Match the display_size

        try:
            # Set current frame
            self.pygame_animator.current_frame = frame_idx

            self.pygame_animator.camera_mode = 'follow'
            self.pygame_animator.render_frame(frame_idx)

            width, height = 825, 972
            return np.flipud(self.pygame_animator.read_frame_bgr(width, height))

        except Exception as e:
            print(f"Pygame rendering error at frame {frame_idx}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((972, 825, 3), dtype=np.uint8)

    def _capture_hud_frame(self, frame_idx):
        """Capture HUD frame for given frame index"""
        # Update HUD components
        self._update_hud_components(frame_idx)
        
        # Render to buffer - FIXED METHOD FOR AGG BACKEND
        self.fig.canvas.draw()
        
        # Get the buffer as numpy array - FIXED FOR AGG BACKEND
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        
        # Convert RGBA to BGRA for transparency
        buf_bgra = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGRA)
        
        return buf_bgra

    def _overlay_pygame_on_hud(self, hud_frame, pygame_frame):
        """Overlay Pygame 3D animation onto the HUD at the correct position"""
        # Convert HUD to BGRA for alpha blending if needed
        hud_bgra = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2BGRA)

        # Define the position where Pygame should be placed (same as animation area in HUD)
        # Based on the axes position: [0.32, 0.05, 0.43, 0.90]
        hud_height, hud_width = hud_bgra.shape[:2]

        # Calculate pixel coordinates for the animation area
        x_start = int(0.35 * hud_width)    # 32% from left
        y_start = int(0.05 * hud_height)   # 5% from bottom  
        area_width = int(0.40 * hud_width)  # 43% width
        area_height = int(0.90 * hud_height) # 90% height

        # Resize Pygame frame to fit the animation area
        pygame_resized = cv2.resize(pygame_frame, (area_width, area_height))

        # Overlay Pygame frame onto HUD at the calculated position
        result = hud_bgra.copy()
        result[y_start:y_start+area_height, x_start:x_start+area_width, :3] = pygame_resized

        # Convert back to BGR for video output
        result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        return result_bgr

    def record_animation(self, output_file=None, fps=30):
        """Main recording function - processes all frames without display"""
        if cv2 is None:
            raise RuntimeError(
                "Video export requires OpenCV. Install opencv-python in the project venv."
            )

        if output_file:
            self.output_file = output_file
        self.fps = fps

        print(f"Starting recording...")
        print(f"Total frames to process: {self.total_frames}")
        print(f"Output: {self.output_file}")
        print(f"FPS: {fps}")

        # Setup components
        self.setup_headless_hud()
        self.setup_headless_pygame()

        print("\n === Video Writing Started ===")

        start_time = time.time()
        
        # Process all frames
        for frame_idx in range(self.total_frames):
            # Progress bar
            progress = (frame_idx + 1) / self.total_frames
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f'\rProgress: |{bar}| {frame_idx+1}/{self.total_frames} ({progress:.1%})', end='', flush=True)

            # Capture HUD frame first (base layer)
            hud_frame = self._capture_hud_frame(frame_idx)

            # Capture Pygame frame (3D animation)
            pygame_frame = self._render_pygame_frame_headless(frame_idx)

            # Overlay Pygame onto HUD at the correct position
            combined_frame = self._overlay_pygame_on_hud(hud_frame, pygame_frame)

            # Store frame
            self.video_frames.append(combined_frame)

        # Save video
        self._save_video()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Recording completed in {total_time:.2f} seconds")
        print(f"Average FPS: {self.total_frames/total_time:.2f}")

    def _save_video(self):
        """Save captured frames as MP4"""
        if not self.video_frames:
            print("No frames to save")
            return
            
        print(f"Saving video with {len(self.video_frames)} frames...")
        
        # Get dimensions from first frame
        height, width = self.video_frames[0].shape[:2]
        
        # Try different codecs - suppress OpenH264 warnings
        import warnings
        warnings.filterwarnings("ignore")
        
        # Preferred codecs in order
        codecs = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # XVID
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion-JPEG
        ]
        
        video_writer = None
        
        for codec_name, fourcc in codecs:
            try:
                video_writer = cv2.VideoWriter(self.output_file, fourcc, self.fps, (width, height))
                if video_writer.isOpened():
                    print(f"Using codec: {codec_name}")
                    break
                else:
                    video_writer = None
            except Exception as e:
                print(f"Codec {codec_name} failed: {e}")
                continue
        
        if video_writer is None or not video_writer.isOpened():
            print("Error: Could not create video writer with any codec")
            return
            
        for frame in self.video_frames:
            video_writer.write(frame)
            
        video_writer.release()
        print(f"Video successfully saved as: {self.output_file}")
        print(f"Final video: {len(self.video_frames)} frames, {width}x{height}")

    def cleanup(self):
        """Cleanup resources"""
        if self.pygame_initialized:
            pygame.quit()
        plt.close('all')


def main():
    """Entry point for headless recording"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python integrated_animation_recorder.py <leader.mat> <follower.mat> [track.mat]")
        print("Optional: --output <filename.mp4> --fps <frames_per_second>")
        sys.exit(1)
    
    leader_file = sys.argv[1]
    follower_file = sys.argv[2]
    track_file = sys.argv[3] if len(sys.argv) > 3 else "NASCAR_Track_Monge_v3.mat"
    
    # Parse optional arguments
    output_file = "stackelberg_race.mp4"
    fps = 30
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--fps" and i + 1 < len(sys.argv):
            fps = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    try:
        recorder = HeadlessAnimationRecorder(leader_file, follower_file, track_file)
        recorder.record_animation(output_file=output_file, fps=fps)
        recorder.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

'''
# Basic usage
python VideoWriter.py Leader.mat SimResult.mat

# With custom output and FPS
python VideoWriter.py Leader.mat SimResult.mat --output my_race.mp4 --fps 60
'''
