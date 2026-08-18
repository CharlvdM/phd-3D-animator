import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
import pygame
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
import threading
import time
from Stackelberg_HUD import DataProcessor, PrecomputedData, TelemetryDashboard

class IntegratedAnimationFixed:
    def __init__(self, leader_file, follower_file, track_file, car_visual_scale=1.5):
        self.leader_file = leader_file
        self.follower_file = follower_file
        self.track_file = track_file
        self.car_visual_scale = car_visual_scale
        
        # Shared state
        self.current_frame = 0
        self.playing = True
        self.frame_lock = threading.Lock()
        
        # Pygame state
        self.pygame_initialized = False
        self.pygame_animator = None
        
        # Screen dimensions (will be detected automatically)
        self.screen_width = None
        self.screen_height = None
        self.layout_ratios = {}  # Store layout ratios
        self.scale_factor = 1.0  # Global scale factor
        
        # Initialize data processor
        print("Loading data...")
        self.data_processor = DataProcessor(leader_file, follower_file, track_file)

        # Precomputed data
        self.precomputed_data = PrecomputedData(self.data_processor)
        self.precomputed_data.precompute_all()
        
        # HUD components
        self.fig = None
        self.animation_ax = None
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
        
    def detect_screen_resolution(self):
        """Detect the screen resolution dynamically"""
        try:
            # Method 1: Use pygame to detect display info
            pygame.init()
            info = pygame.display.Info()
            self.screen_width = info.current_w
            self.screen_height = info.current_h
            pygame.quit()
        except:
            try:
                # Method 2: Use tkinter (usually available)
                import tkinter as tk
                root = tk.Tk()
                self.screen_width = root.winfo_screenwidth()
                self.screen_height = root.winfo_screenheight()
                root.destroy()
            except:
                # Method 3: Default fallback
                self.screen_width = 1920
                self.screen_height = 1080
                print("Could not detect screen resolution, using default 1920x1080")
        
        print(f"Detected screen resolution: {self.screen_width} x {self.screen_height}")
        
    def calculate_layout_ratios(self):
        """Calculate layout ratios based on screen resolution"""
        # Base ratios from your original layout (for 1920x1080)
        base_width = 1920
        base_height = 1080
        
        # Scale factors
        width_scale = self.screen_width / base_width
        height_scale = self.screen_height / base_height
        
        # Use the smaller scale to maintain proportions
        self.scale_factor = min(width_scale, height_scale) * 0.95  # 95% to leave some margin
        
        # Calculate figure size maintaining 16:10 ratio
        fig_width = 16 * self.scale_factor
        fig_height = 10 * self.scale_factor
        
        # Store layout positions as ratios of figure size
        self.layout_ratios = {
            # Minimap position and size
            'minimap_pos': [0.005, 0.50],
            'minimap_size': [0.15, 0.40],
            
            # Steering wheel position and size
            'steering_pos': [0.785, 0.75],
            'steering_size': [0.17, 0.21],
            
            # GG diagram position and size
            'gg_pos': [0.05, 0.12],
            'gg_size': [0.21, 0.26],
            
            # Dashboard position and size
            'dashboard_pos': [0.785, 0.34],
            'dashboard_size': [0.21, 0.37],
            
            # Telemetry bars position and size
            'telemetry_pos': [0.785, 0.06],
            'telemetry_size': [0.21, 0.24],
            
            # Friction circles positions and sizes
            'friction_front_pos': [0.15, 0.75],
            'friction_front_size': [0.20, 0.20],
            'friction_rear_pos': [0.15, 0.45],
            'friction_rear_size': [0.20, 0.20],
            
            # Animation area position and size
            'animation_pos': [0.32, 0.05],
            'animation_size': [0.43, 0.90],
            
            # Figure size
            'fig_size': [fig_width, fig_height],
            
            # Pygame window size (relative to screen)
            'pygame_size': [int(820 * self.scale_factor), int(875 * self.scale_factor)],
            'pygame_pos': [int(self.screen_width * 0.33), int(self.screen_height * 0.1)]  
        }
        
        print(f"Layout calculated with scale factor: {self.scale_factor:.2f}")
        
    def setup_dashboard(self):
        """Setup the complete dashboard in fullscreen"""
        print("Setting up fullscreen dashboard...")
        
        # Detect screen resolution and calculate layout
        self.detect_screen_resolution()
        self.calculate_layout_ratios()
        
        # Create the main figure with calculated size
        self.fig = plt.figure(figsize=self.layout_ratios['fig_size'], facecolor='#1a1a1a')
        
        # Setup all HUD components FIRST
        self._setup_minimap()
        self._setup_steering_wheel()
        self._setup_gg_diagram()
        self._setup_dashboard()
        self._setup_telemetry_bars()
        self._setup_friction_circles()
        
        # Setup the main animation area
        self._setup_animation_area()
        
        # Connect events
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        
        # Show matplotlib window FIRST
        plt.tight_layout()
        
        # Set fullscreen after creating the figure
        self._set_fullscreen()
        self.fig.canvas.draw()
        plt.pause(0.2)

        # Initialize Pygame AFTER matplotlib window is fully shown and rendered
        self._init_pygame()

    def _set_fullscreen(self):
        """Set matplotlib window to fullscreen"""
        try:
            manager = plt.get_current_fig_manager()
            # Try different methods for fullscreen across different backends
            if hasattr(manager, 'window'):
                if sys.platform.startswith("linux") and hasattr(manager.window, 'attributes'):
                    manager.window.attributes('-zoomed', True)
                elif hasattr(manager.window, 'showMaximized'):
                    manager.window.showMaximized()
                elif hasattr(manager.window, 'state'):
                    manager.window.state('zoomed')  # Windows
                elif hasattr(manager.window, 'attributes'):
                    manager.window.attributes('-zoomed', True)  # Linux
                elif hasattr(manager.window, 'full_screen_toggle'):
                    manager.window.full_screen_toggle()
            elif hasattr(manager, 'frame'):
                manager.frame.Maximize(True)  # wx backend
            elif hasattr(manager, 'full_screen_toggle'):
                manager.full_screen_toggle()
            print("Matplotlib window set to fullscreen")
        except Exception as e:
            print(f"Fullscreen not available: {e}")

    def _setup_animation_area(self):
        """Setup the animation display area"""
        pos = self.layout_ratios['animation_pos']
        size = self.layout_ratios['animation_size']
        
        self.animation_ax = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        self.animation_ax.set_facecolor('#000000')
        self.animation_ax.set_xticks([])
        self.animation_ax.set_yticks([])
        
        # Calculate font sizes based on screen resolution
        title_font_size = int(16 * self.scale_factor)
        text_font_size = int(14 * self.scale_factor)
        
        self.animation_ax.set_title('3D Vehicle Animation', 
                                   fontsize=title_font_size, color='white', fontweight='bold', pad=20)
        
        # Add status text with dynamically sized font
        self.animation_ax.text(0.5, 0.5, '3D Animation Running in Pygame Window\n\n'
                              'Controls:\n'
                              '• Space: Play/Pause\n'
                              '• 1-5: Camera Modes\n'
                              '• Mouse: Rotate (Free Camera)\n'
                              '• Scroll: Zoom\n'
                              '• WASD/QE: Pan (Free Camera)', 
                              ha='center', va='center', color='white', fontsize=text_font_size,
                              transform=self.animation_ax.transAxes)
        
        # Draw the border
        for spine in self.animation_ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

    def _animation_window_geometry(self):
        """Return screen position and size for the 3D viewport placeholder."""
        try:
            self.fig.canvas.draw()
            bbox = self.animation_ax.get_window_extent()
            canvas_width, canvas_height = self.fig.canvas.get_width_height()
            x0, y0, width, height = bbox.bounds

            manager = plt.get_current_fig_manager()
            root_x = 0
            root_y = 0
            if hasattr(manager, 'window'):
                window = manager.window
                if hasattr(window, 'winfo_rootx') and hasattr(window, 'winfo_rooty'):
                    root_x = window.winfo_rootx()
                    root_y = window.winfo_rooty()
                elif hasattr(window, 'geometry'):
                    geom = window.geometry()
                    if hasattr(geom, 'x') and hasattr(geom, 'y'):
                        root_x = geom.x()
                        root_y = geom.y()

            screen_x = int(root_x + x0)
            screen_y = int(root_y + canvas_height - y0 - height)
            viewport_size = (max(200, int(width)), max(200, int(height)))
            return (screen_x, screen_y), viewport_size
        except Exception as e:
            print(f"Could not derive animation viewport geometry: {e}")
            return tuple(self.layout_ratios['pygame_pos']), tuple(self.layout_ratios['pygame_size'])

    def _init_pygame(self):
        """Initialize Pygame with dynamic positioning"""
        try:
            if sys.platform.startswith("linux"):
                os.environ.setdefault("SDL_VIDEODRIVER", "x11")
                os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

            from Stackleberg_3DAnimator import Vehicle3DAnimatorGL
            
            print("Initializing Pygame (borderless window)...")
            self.pygame_animator = Vehicle3DAnimatorGL(
                self.leader_file, 
                self.follower_file, 
                self.track_file,
                car_visual_scale=self.car_visual_scale,
            )

            pygame_pos, pygame_size = self._animation_window_geometry()

            # Set window position
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{pygame_pos[0]},{pygame_pos[1]}"

            # Initialize Pygame with NOFRAME flag for borderless window
            pygame.init()

            # Create borderless window with calculated size
            self.pygame_surface = pygame.display.set_mode(pygame_size,
                                                         pygame.DOUBLEBUF | pygame.OPENGL | pygame.NOFRAME)
            
            # Set window title
            pygame.display.set_caption("3D Vehicle Animation")

            # Platform-specific window management
            self._setup_window_always_on_top()
            
            self.pygame_animator.configure_opengl(*pygame_size)
            
            self.pygame_initialized = True
            print(f"Pygame initialized successfully at position {pygame_pos} with size {pygame_size}")
            
        except Exception as e:
            print(f"Pygame initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_window_always_on_top(self):
        """Setup window to be always on top (platform-specific)"""
        try:
            import platform
            system = platform.system()
            
            if system == "Windows":
                import ctypes
                try:
                    import win32gui
                    import win32con
                    hwnd = pygame.display.get_wm_info()['window']
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    print("Pygame window set to always-on-top (Windows)")
                except ImportError:
                    # Fallback using ctypes only
                    hwnd = pygame.display.get_wm_info()['window']
                    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
                    print("Pygame window set to always-on-top (Windows fallback)")
            
            elif system == "Linux":
                # Linux method using xprop (requires xdotool)
                try:
                    import subprocess
                    hwnd = pygame.display.get_wm_info()['window']
                    subprocess.run(['xprop', '-id', str(hwnd), '-f', '_NET_WM_STATE', '32a', 
                                  '-set', '_NET_WM_STATE', '_NET_WM_STATE_ABOVE'])
                    print("Pygame window set to always-on-top (Linux)")
                except:
                    print("Could not set always-on-top on Linux (xprop required)")
            
            elif system == "Darwin":  # macOS
                print("Always-on-top not implemented for macOS")
            
        except Exception as e:
            print(f"Could not set always-on-top: {e}")

    def _setup_minimap(self):
        """Setup minimap component"""
        pos = self.layout_ratios['minimap_pos']
        size = self.layout_ratios['minimap_size']
        
        ax_minimap = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        
        # Calculate font sizes based on screen resolution
        title_font_size = int(14 * self.scale_factor)
        
        ax_minimap.set_title('Track Overview', fontsize=title_font_size, color='white', fontweight='bold', pad=12)
        ax_minimap.set_facecolor('#0a0a0a')
        ax_minimap.set_aspect('equal')
        ax_minimap.grid(True, alpha=0.15, color='#333333')
        ax_minimap.set_xticks([])
        ax_minimap.set_yticks([])
        for spine in ax_minimap.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

        # Full track on minimap
        ax_minimap.plot(self.data_processor.xc, self.data_processor.yc, color='#666666', linestyle=':', 
                       linewidth=2.0 * self.scale_factor, alpha=0.5)
        ax_minimap.plot(self.data_processor.x_inner, self.data_processor.y_inner, color='#00ff00', 
                       linewidth=2.5 * self.scale_factor, alpha=0.6)
        ax_minimap.plot(self.data_processor.x_outer, self.data_processor.y_outer, color='#00ff00', 
                       linewidth=2.5 * self.scale_factor, alpha=0.6)

        xmin = min(np.min(self.data_processor.x_inner), np.min(self.data_processor.x_outer)) - 10
        xmax = max(np.max(self.data_processor.x_inner), np.max(self.data_processor.x_outer)) + 10
        ymin = min(np.min(self.data_processor.y_inner), np.min(self.data_processor.y_outer)) - 10
        ymax = max(np.max(self.data_processor.y_inner), np.max(self.data_processor.y_outer)) + 10
        ax_minimap.set_xlim(xmin, xmax)
        ax_minimap.set_ylim(ymin, ymax)
        ax_minimap.invert_yaxis()

        # Calculate marker size based on screen resolution
        marker_size = max(8, int(16 * self.scale_factor))

        # Car dots on minimap
        self.mini_carF, = ax_minimap.plot([], [], 'o', color='#ff3333', markersize=marker_size, 
                                         markeredgecolor='white', markeredgewidth=3 * self.scale_factor)
        self.mini_carL, = ax_minimap.plot([], [], 'o', color='#3366ff', markersize=marker_size, 
                                         markeredgecolor='white', markeredgewidth=3 * self.scale_factor)
    
    def _setup_steering_wheel(self):
        """Setup steering wheel component"""
        pos = self.layout_ratios['steering_pos']
        size = self.layout_ratios['steering_size']
        
        ax_wheel = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        
        # Calculate font sizes based on screen resolution
        title_font_size = int(14 * self.scale_factor)
        text_font_size = int(16 * self.scale_factor)
        
        ax_wheel.set_title('Steering (Follower)', fontsize=title_font_size, color='#ff3333', fontweight='bold', pad=12)
        ax_wheel.set_facecolor('#0a0a0a')
        ax_wheel.set_aspect('equal')
        ax_wheel.set_xlim(-1.15, 1.15)
        ax_wheel.set_ylim(-1.15, 1.15)
        ax_wheel.set_xticks([])
        ax_wheel.set_yticks([])
        for spine in ax_wheel.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

        # Calculate line widths based on scale
        outer_linewidth = max(4, int(6 * self.scale_factor))
        inner_linewidth = max(3, int(4 * self.scale_factor))
        hub_linewidth = max(2, int(3 * self.scale_factor))
        spoke_linewidth = max(5, int(7 * self.scale_factor))
        center_linewidth = max(4, int(6 * self.scale_factor))

        # steering wheel
        import matplotlib.patches as patches
        wheel_outer = patches.Circle((0, 0), 1.0, fill=False, edgecolor='#cccccc', linewidth=outer_linewidth)
        wheel_inner = patches.Circle((0, 0), 0.88, fill=False, edgecolor='#999999', linewidth=inner_linewidth)
        wheel_hub = patches.Circle((0, 0), 0.18, color='#333333', edgecolor='#666666', linewidth=hub_linewidth)
        ax_wheel.add_patch(wheel_outer)
        ax_wheel.add_patch(wheel_inner)
        ax_wheel.add_patch(wheel_hub)

        # Three spokes
        base_spokes = np.deg2rad([90, 210, 330])
        spoke_len = 0.82
        for ang in base_spokes:
            x = np.array([0.0, spoke_len*np.cos(ang)])
            y = np.array([0.0, spoke_len*np.sin(ang)])
            ln, = ax_wheel.plot(x, y, color='#aaaaaa', linewidth=spoke_linewidth, solid_capstyle='round')
            self.spokes_lines.append(ln)

        # Center indicator mark
        self.center_mark = ax_wheel.plot([0, 0], [0.22, 0.35], color='#ff0000', 
                                       linewidth=center_linewidth, solid_capstyle='round')[0]

        # Steering angle text
        self.steer_text = ax_wheel.text(0, -0.65, '', fontsize=text_font_size, color='#ffffff', 
                                      ha='center', fontweight='bold', family='monospace',
                                      bbox=dict(boxstyle='round,pad=0.8', facecolor='#2a2a2a', 
                                              edgecolor='#555555', linewidth=3 * self.scale_factor))
    
    def _setup_gg_diagram(self):
        """Setup G-G diagram component"""
        pos = self.layout_ratios['gg_pos']
        size = self.layout_ratios['gg_size']
        
        ax_gg = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        ax_gg.set_facecolor('#0a0a0a')

        # Calculate font sizes based on screen resolution
        title_font_size = int(14 * self.scale_factor)
        label_font_size = int(12 * self.scale_factor)
        tick_font_size = int(11 * self.scale_factor)
        legend_font_size = max(8, int(9 * self.scale_factor))

        # Combine all data
        all_ax = np.concatenate([self.data_processor.ax_F_interp, self.data_processor.ax_L_interp, 
                                self.data_processor.ax_F_direct_interp, self.data_processor.ax_L_direct_interp])
        all_ay = np.concatenate([self.data_processor.ay_F_interp, self.data_processor.ay_L_interp,
                                self.data_processor.ay_F_direct_interp, self.data_processor.ay_L_direct_interp])

        # Use rolling window to find sustained maximum
        window_size = 100  # Adjust based on frame rate
        def rolling_max(data, window):
            return np.array([np.max(np.abs(data[i:i+window])) for i in range(0, len(data)-window+1, window)])

        if len(all_ax) > window_size:
            AX_MAX = np.percentile(rolling_max(all_ax, window_size), 90) + 1e-6
            AY_MAX = np.percentile(rolling_max(all_ay, window_size), 90) + 1e-6
        else:
            AX_MAX = np.percentile(np.abs(all_ax), 95) + 1e-6
            AY_MAX = np.percentile(np.abs(all_ay), 95) + 1e-6

        # Set limits in g-forces
        AX_LIM = max(0.5, min(2.5, 1.2 * AX_MAX)) 
        AY_LIM = max(0.5, min(2.5, 1.2 * AY_MAX))

        ax_gg.set_xlim(-AY_LIM*1.1, AY_LIM*1.1)
        ax_gg.set_ylim(-AX_LIM*1.1, AX_LIM*1.1)

        ax_gg.set_xlabel('← Lateral (g) → ', fontsize=label_font_size, color='white', fontweight='bold')
        ax_gg.set_ylabel('← Longitudinal (g) →  ', fontsize=label_font_size, color='white', fontweight='bold')
        ax_gg.set_title('G-G Diagram', fontsize=title_font_size, color='white', fontweight='bold', pad=12)
        ax_gg.grid(True, alpha=0.2, color='#333333')
        ax_gg.axhline(0, color='#666666', linewidth=2 * self.scale_factor)
        ax_gg.axvline(0, color='#666666', linewidth=2 * self.scale_factor)
        ax_gg.tick_params(colors='white', labelsize=tick_font_size)
        for spine in ax_gg.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

        # Calculate marker size based on screen resolution
        marker_size = max(8, int(10 * self.scale_factor))

        # Current position dots
        self.gg_leader_direct, = ax_gg.plot([], [], 'o', color='#3366ff', markersize=marker_size, 
                                           markeredgecolor='white', markeredgewidth=2 * self.scale_factor)
        self.gg_follower_direct, = ax_gg.plot([], [], 'o', color='#ff3333', markersize=marker_size, 
                                             markeredgecolor='white', markeredgewidth=2 * self.scale_factor)

        ax_gg.legend([self.gg_leader_direct, self.gg_follower_direct], 
                     ['Leader', 'Follower'], 
                     loc='upper right', fontsize=legend_font_size, facecolor='#2a2a2a', 
                     edgecolor='#555555', labelcolor='white')

    def _setup_dashboard(self):
        """Setup main dashboard display"""
        pos = self.layout_ratios['dashboard_pos']
        size = self.layout_ratios['dashboard_size']
        
        ax_dash = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        ax_dash.set_facecolor('#0a0a0a')
        ax_dash.set_xticks([])
        ax_dash.set_yticks([])
        for spine in ax_dash.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

        # Calculate font sizes based on screen resolution
        time_font_size = int(22 * self.scale_factor)
        speed_font_size = int(18 * self.scale_factor)
        gap_font_size = int(16 * self.scale_factor)
        small_gap_font_size = int(14 * self.scale_factor)
        position_font_size = int(16 * self.scale_factor)
        label_font_size = int(12 * self.scale_factor)

        # Dashboard text elements with scaled fonts
        self.dash_vals = {
            "time": ax_dash.text(0.50, 0.85, "", transform=ax_dash.transAxes, fontsize=time_font_size, 
                                 color='#00ff00', ha='center', fontweight='bold', family='monospace'),
            "spdF": ax_dash.text(0.50, 0.72, "", transform=ax_dash.transAxes, fontsize=speed_font_size, 
                                 color='#ff3333', ha='center', fontweight='bold', family='monospace'),
            "spdL": ax_dash.text(0.50, 0.60, "", transform=ax_dash.transAxes, fontsize=speed_font_size, 
                                 color='#3366ff', ha='center', fontweight='bold', family='monospace'),
            "gapS": ax_dash.text(0.50, 0.44, "", transform=ax_dash.transAxes, fontsize=gap_font_size, 
                                 color='#ffff00', ha='center', fontweight='bold', family='monospace'),
            "gapD": ax_dash.text(0.50, 0.32, "", transform=ax_dash.transAxes, fontsize=small_gap_font_size, 
                                 color='#aaaaaa', ha='center', fontweight='normal', family='monospace'),
            "p1": ax_dash.text(0.50, 0.12, "", transform=ax_dash.transAxes, fontsize=position_font_size, 
                               color='#ffffff', ha='center', fontweight='bold'),
        }

        # Labels with scaled fonts
        ax_dash.text(0.27, 0.86, "TIME", transform=ax_dash.transAxes, fontsize=label_font_size, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.05, 0.72, "F:", transform=ax_dash.transAxes, fontsize=int(15 * self.scale_factor), 
                     color='#ff3333', ha='left', fontweight='bold')
        ax_dash.text(0.05, 0.60, "L:", transform=ax_dash.transAxes, fontsize=int(15 * self.scale_factor), 
                     color='#3366ff', ha='left', fontweight='bold')
        ax_dash.text(0.50, 0.51, "GAP (Arc Length)", transform=ax_dash.transAxes, fontsize=label_font_size, 
                     color='#888888', ha='center', fontweight='bold')
        ax_dash.text(0.50, 0.22, "POSITION", transform=ax_dash.transAxes, fontsize=label_font_size, 
                     color='#888888', ha='center', fontweight='bold')

    def _setup_telemetry_bars(self):
        """Setup telemetry bars"""
        pos = self.layout_ratios['telemetry_pos']
        size = self.layout_ratios['telemetry_size']
        
        ax_bars = self.fig.add_axes([pos[0], pos[1], size[0], size[1]])
        
        # Calculate font sizes based on screen resolution
        title_font_size = int(12 * self.scale_factor)
        tick_font_size = int(10 * self.scale_factor)
        label_font_size = int(11 * self.scale_factor)
        
        ax_bars.set_title('Accel & Input (Follower: Red, Leader: Blue)', 
                         fontsize=title_font_size, color='white', fontweight='bold', pad=12)
        ax_bars.set_facecolor('#0a0a0a')
        ax_bars.set_xlim(0, 1)
        ax_bars.set_ylim(-1.0, 1.0)
        ax_bars.set_xticks([])
        ax_bars.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax_bars.set_yticklabels(['-100%', '-50%', '0', '+50%', '+100%'], fontsize=tick_font_size, color='white')
        ax_bars.axhline(0, color='#666666', linewidth=3 * self.scale_factor)
        ax_bars.grid(True, axis='y', alpha=0.2, color='#333333')
        for spine in ax_bars.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(int(3 * self.scale_factor))

        # Bar positions
        bar_x = [0.2, 0.50, 0.80]
        bar_width = 0.15 * self.scale_factor
        bar_half = bar_width / 2
        bar_offset = 0.04 * self.scale_factor

        # Create dual bars
        def make_dual_bar(x_pos, label_text):
            # Follower (left side, red)
            f_pos = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#ff3333',
                                edgecolor='white', linewidth=2 * self.scale_factor)[0]
            f_neg = ax_bars.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color='#990000',
                                edgecolor='white', linewidth=2 * self.scale_factor)[0]
            
            # Leader (right side, blue)
            l_pos = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#3366ff',
                                edgecolor='white', linewidth=2 * self.scale_factor)[0]
            l_neg = ax_bars.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color='#000099',
                                edgecolor='white', linewidth=2 * self.scale_factor)[0]
            
            # Label
            ax_bars.text(x_pos, -1.15, label_text, ha='center', va='top', fontsize=label_font_size, 
                         color='white', fontweight='bold')
            
            return (f_pos, f_neg, l_pos, l_neg)

        # Create all bars
        self.bar_throttle = make_dual_bar(bar_x[0], 'Throttle/\nBrake')
        self.bar_ax = make_dual_bar(bar_x[1], 'Accel X')
        self.bar_ay = make_dual_bar(bar_x[2], 'Accel Y')
    
    def _setup_friction_circles(self):
        """Setup friction circles with scaled axes"""
        pos_front = self.layout_ratios['friction_front_pos']
        size_front = self.layout_ratios['friction_front_size']
        pos_rear = self.layout_ratios['friction_rear_pos']
        size_rear = self.layout_ratios['friction_rear_size']
        
        ax_fc_front = self.fig.add_axes([pos_front[0], pos_front[1], size_front[0], size_front[1]])
        ax_fc_rear = self.fig.add_axes([pos_rear[0], pos_rear[1], size_rear[0], size_rear[1]])

        # Calculate font sizes based on screen resolution
        title_font_size = int(16 * self.scale_factor)
        label_font_size = int(12 * self.scale_factor)
        tick_font_size = int(11 * self.scale_factor)
        legend_font_size = max(10, int(11 * self.scale_factor))

        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_facecolor('#0a0a0a')
            ax_fc.grid(True, alpha=0.3, color='#333')
            ax_fc.set_aspect('equal', adjustable='box')
            ax_fc.tick_params(colors='white', labelsize=tick_font_size)
            for sp in ax_fc.spines.values():
                sp.set_edgecolor('#555')
                sp.set_linewidth(int(3 * self.scale_factor))

        scale_factor = 1.0 / self.data_processor.forcescale / 1000.0  # Convert to kN

        # Calculate limits
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

        # Set limits
        FCX_lim = max(5, np.ceil(FCX))
        FCY_lim = max(5, np.ceil(FCY))

        for ax_fc in (ax_fc_front, ax_fc_rear):
            ax_fc.set_xlim(-1.1*FCX_lim, 1.1*FCX_lim)
            ax_fc.set_ylim(-1.1*FCY_lim, 1.1*FCY_lim)

        # Store the scale factor for use in updates
        self.friction_scale_factor = scale_factor

        # Add titles and labels with scaled fonts
        ax_fc_front.set_title(f'Front Axle', color='white', fontsize=title_font_size, fontweight='bold')
        ax_fc_front.set_xlabel('← Lateral (KN) →', color='white', fontsize=label_font_size)
        ax_fc_front.set_ylabel('← Longitudinal (KN) →', color='white', fontsize=label_font_size)

        ax_fc_rear.set_title(f'Rear Axle', color='white', fontsize=title_font_size, fontweight='bold')
        ax_fc_rear.set_xlabel('← Lateral (KN) →', color='white', fontsize=label_font_size)
        ax_fc_rear.set_ylabel('← Longitudinal (KN) →', color='white', fontsize=label_font_size)

        # Calculate marker size based on screen resolution
        marker_size = max(8, int(10 * self.scale_factor))

        # Artists to update each frame
        self.front_dot_F, = ax_fc_front.plot([], [], 'o', color='#ff3333', mec='white', 
                                            mew=2 * self.scale_factor, ms=marker_size, label='Follower')
        self.front_dot_L, = ax_fc_front.plot([], [], 'o', color='#3366ff', mec='white', 
                                            mew=2 * self.scale_factor, ms=marker_size, label='Leader')
        ax_fc_front.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white', fontsize=legend_font_size)

        self.rear_dot_F, = ax_fc_rear.plot([], [], 'o', color='#ff3333', mec='white', 
                                          mew=2 * self.scale_factor, ms=marker_size, label='Follower')
        self.rear_dot_L, = ax_fc_rear.plot([], [], 'o', color='#3366ff', mec='white', 
                                          mew=2 * self.scale_factor, ms=marker_size, label='Leader')
        ax_fc_rear.legend(facecolor='#2a2a2a', edgecolor='#555', labelcolor='white', fontsize=legend_font_size)

    def _on_key_press(self, event):
        """Handle key press events in matplotlib window"""
        if event.key == ' ':
            with self.frame_lock:
                self.playing = not self.playing
                print(f"{'Playing' if self.playing else 'Paused'}")
        elif event.key == 'up' and not self.playing:
            with self.frame_lock:
                self.current_frame = min(self.current_frame + 1, len(self.data_processor.t) - 1)
        elif event.key == 'down' and not self.playing:
            with self.frame_lock:
                self.current_frame = max(self.current_frame - 1, 0)
        elif event.key == 'f':
            # Toggle fullscreen with 'f' key
            self._set_fullscreen()
    
    def _on_close(self, event):
        """Handle window close event"""
        if self.pygame_initialized:
            pygame.quit()
    
    def _update_hud_components(self, data_idx):
        """Update all HUD components"""
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

        G_CONVERSION = 1.0 / 9.8

        # Update GG diagram
        ay_L_dir, ax_L_dir, ay_F_dir, ax_F_dir = self.precomputed_data.gg_data[data_idx]

        # Convert all to g-forces
        ay_L_dir *= G_CONVERSION
        ax_L_dir *= G_CONVERSION
        ay_F_dir *= G_CONVERSION
        ax_F_dir *= G_CONVERSION
    
        # Direct data
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

        # Apply scaling to convert to Newtons
        scale_factor = getattr(self, 'friction_scale_factor', 1.0)

        self.front_dot_F.set_data([Ffy_F * scale_factor], [Ffx_F * scale_factor])
        self.front_dot_L.set_data([Ffy_L * scale_factor], [Ffx_L * scale_factor])
        self.rear_dot_F.set_data([Fry_F * scale_factor], [Frx_F * scale_factor])
        self.rear_dot_L.set_data([Fry_L * scale_factor], [Frx_L * scale_factor])
    
    def _update_dual_bar_fast(self, bars, f_val, l_val):
        """Update dual bar display"""
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
    
    def _update_pygame_frame(self):
        """Update and render a single Pygame frame"""
        if not self.pygame_initialized or not self.pygame_animator:
            return
        
        try:
            # Set current frame
            self.pygame_animator.current_frame = self.current_frame
            
            # Handle Pygame events - add custom close for borderless window
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        # ESC to close borderless window
                        return False
                    elif event.key == pygame.K_SPACE:
                        with self.frame_lock:
                            self.playing = not self.playing
                    elif event.key == pygame.K_UP and not self.playing:
                        with self.frame_lock:
                            self.current_frame = min(self.current_frame + 1, len(self.data_processor.t) - 1)
                    elif event.key == pygame.K_DOWN and not self.playing:
                        with self.frame_lock:
                            self.current_frame = max(self.current_frame - 1, 0)
                    elif event.key == pygame.K_1:
                        self.pygame_animator.camera_mode = 'follow'
                    elif event.key == pygame.K_2:
                        self.pygame_animator.camera_mode = 'rear_view'
                    elif event.key == pygame.K_3:
                        self.pygame_animator.camera_mode = 'top_down'
                    elif event.key == pygame.K_4:
                        self.pygame_animator.camera_mode = 'overview'
                    elif event.key == pygame.K_5:
                        self.pygame_animator.camera_mode = 'free'
                    elif event.key == pygame.K_q:
                        # Q to close borderless window
                        return False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.pygame_animator.mouse_down = True
                        self.pygame_animator.last_mouse_pos = pygame.mouse.get_pos()
                    elif event.button == 4 and self.pygame_animator.camera_mode == 'free':
                        self.pygame_animator.camera_distance *= 0.9
                    elif event.button == 5 and self.pygame_animator.camera_mode == 'free':
                        self.pygame_animator.camera_distance *= 1.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.pygame_animator.mouse_down = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.pygame_animator.mouse_down and self.pygame_animator.camera_mode == 'free':
                        current_pos = pygame.mouse.get_pos()
                        dx = current_pos[0] - self.pygame_animator.last_mouse_pos[0]
                        dy = current_pos[1] - self.pygame_animator.last_mouse_pos[1]
                        
                        self.pygame_animator.camera_angle_h += dx * 0.3
                        self.pygame_animator.camera_angle_v += dy * 0.3
                        self.pygame_animator.camera_angle_v = np.clip(self.pygame_animator.camera_angle_v, -89, 89)
                        
                        self.pygame_animator.last_mouse_pos = current_pos
            
            self.pygame_animator.render_frame(self.current_frame, handle_input=True)
            
            pygame.display.flip()
            
            return True
            
        except Exception as e:
            print(f"Pygame rendering error: {e}")
            return True

    def animate(self):
        """Run the integrated animation"""
        print("Starting integrated animation...")
        print("Two windows will open:")
        print("1. Matplotlib HUD Dashboard (FULLSCREEN)")
        print("2. Pygame 3D Animation (borderless window)")
        print("Controls:")
        print("• Space: Play/Pause (works in both windows)")
        print("• F: Toggle fullscreen (Matplotlib window)")
        print("• ESC or Q: Close Pygame borderless window")
        print("• 1-5: Camera Modes (Pygame window only)")
        print("• Mouse: Rotate (Free Camera mode)")
        print("• Scroll: Zoom")
        print("• WASD/QE: Pan (Free Camera mode)")
        
        def update(frame):
            """Update function for animation"""
            with self.frame_lock:
                if self.playing:
                    self.current_frame = (self.current_frame + 1) % len(self.data_processor.t)
                # Note: frame stepping is handled by key events above
                data_idx = self.current_frame

            # Update Pygame frame
            if self.pygame_initialized:
                pygame_result = self._update_pygame_frame()
                if pygame_result is False:
                    # Pygame window was closed
                    plt.close()
                    return []

            # Update HUD components
            self._update_hud_components(data_idx)

            # Return all artists that need updating
            all_artists = [
                self.mini_carF, self.mini_carL, *self.spokes_lines, self.center_mark, self.steer_text,
                self.gg_leader_direct, self.gg_follower_direct,
                *self.dash_vals.values(), *self.bar_throttle, *self.bar_ax, *self.bar_ay,
                self.front_dot_F, self.front_dot_L, self.rear_dot_F, self.rear_dot_L
            ]

            return all_artists
        
        # Start the animation
        ani = FuncAnimation(
            self.fig, 
            update,
            frames=len(self.data_processor.t),
            interval=13,  # ~30 FPS
            blit=True,
            repeat=True,
            cache_frame_data=False
        )
        
        plt.show()
        
        # Cleanup
        if self.pygame_initialized:
            pygame.quit()
        
        return ani


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the Qt Stackelberg animator HUD.")
    parser.add_argument("leader_file", help="Leader trajectory .mat file")
    parser.add_argument("follower_file", help="Follower trajectory .mat file")
    parser.add_argument(
        "track_file",
        nargs="?",
        default="NASCAR_Track_Monge_v3.mat",
        help="Track .mat file, default: NASCAR_Track_Monge_v3.mat",
    )
    parser.add_argument(
        "--pygame-only",
        action="store_true",
        help="Run the single-window Pygame renderer without the Matplotlib HUD.",
    )
    parser.add_argument(
        "--legacy-hud",
        action="store_true",
        help="Run the old Matplotlib HUD with a borderless Pygame overlay window.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        choices=["follow", "rear_view", "top_down", "overview", "free"],
        default="follow",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--car-scale",
        type=float,
        default=None,
        help="Visual-only scale for the car blocks; does not affect vehicle maths.",
    )
    args = parser.parse_args()

    if args.pygame_only:
        from phd_3d_animator.app import SingleWindowRaceApp

        SingleWindowRaceApp(
            args.leader_file,
            args.follower_file,
            args.track_file,
            car_scale=args.car_scale if args.car_scale is not None else 1.5,
            fps=args.fps,
            camera_mode=args.camera,
            diagnostics=args.diagnostics,
        ).run()
        return

    if not args.legacy_hud:
        from phd_3d_animator.qt_app import QtRaceWindow, configure_surface_format
        from PySide6.QtWidgets import QApplication

        configure_surface_format()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        window = QtRaceWindow(
            args.leader_file,
            args.follower_file,
            args.track_file,
            fps=args.fps,
            camera_mode=args.camera,
            diagnostics=args.diagnostics,
            car_scale=args.car_scale if args.car_scale is not None else 1.5,
        )
        window.resize(1800, 1000)
        window.show()
        sys.exit(app.exec())

    try:
        integrated_anim = IntegratedAnimationFixed(
            args.leader_file,
            args.follower_file,
            args.track_file,
            car_visual_scale=args.car_scale if args.car_scale is not None else 1.5,
        )
        integrated_anim.setup_dashboard()
        integrated_anim.animate()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Usage:
# python Stackelberg_Main.py LeaderData.mat FollowerData.mat TrackData.mat
# eg:
# python Stackelberg_Main.py Leader.mat SimResult.mat NASCAR_Track_Monge_v3.mat

