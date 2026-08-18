"""Qt application shell with a real OpenGL widget in the HUD centre."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from matplotlib import patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from Stackelberg_HUD import DataProcessor, PrecomputedData

from .rendering import Vehicle3DAnimatorGL


class RaceOpenGLWidget(QOpenGLWidget):
    """Qt-owned OpenGL viewport that delegates drawing to the renderer."""

    def __init__(self, animator, parent=None):
        super().__init__(parent)
        self.animator = animator
        self.setMinimumSize(760, 560)
        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        self.makeCurrent()
        self.animator.configure_opengl(max(1, self.width()), max(1, self.height()))

    def resizeGL(self, width, height):
        self.makeCurrent()
        self.animator.configure_opengl(max(1, width), max(1, height))

    def paintGL(self):
        self.makeCurrent()
        self.animator.render_frame(self.animator.current_frame, handle_input=False)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.animator.mouse_down = True
            self.animator.last_mouse_pos = (event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.animator.mouse_down = False

    def mouseMoveEvent(self, event):
        if not self.animator.mouse_down or self.animator.camera_mode != "free":
            return

        current_pos = (event.position().x(), event.position().y())
        dx = current_pos[0] - self.animator.last_mouse_pos[0]
        dy = current_pos[1] - self.animator.last_mouse_pos[1]
        self.animator.camera_angle_h += dx * 0.3
        self.animator.camera_angle_v = np.clip(self.animator.camera_angle_v + dy * 0.3, -89, 89)
        self.animator.last_mouse_pos = current_pos

    def wheelEvent(self, event):
        if self.animator.camera_mode != "free":
            return
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        self.animator.camera_distance = np.clip(self.animator.camera_distance * zoom_factor, 50, 5000)


class QtRaceWindow(QMainWindow):
    """Main Qt HUD window with a central QOpenGLWidget viewport."""

    def __init__(
        self,
        leader_file,
        follower_file,
        track_file,
        fps=30,
        camera_mode="follow",
        diagnostics=False,
        car_scale=1.5,
    ):
        super().__init__()
        self.setWindowTitle("Stackelberg 3D Animator")
        self.fps = fps
        self.data = DataProcessor(leader_file, follower_file, track_file)
        self.precomputed_data = PrecomputedData(self.data)
        self.precomputed_data.precompute_all()
        self.animator = Vehicle3DAnimatorGL(
            leader_file,
            follower_file,
            track_file,
            car_visual_scale=car_scale,
        )
        self.animator.camera_mode = camera_mode
        self.animator.show_diagnostics = diagnostics
        self.labels = {}
        self.camera_buttons = {}
        self.spokes_lines = []
        self.dash_vals = {}

        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(1, int(1000 / self.fps)))

    def _build_ui(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        left_panel = self._build_left_panel()
        self.viewport = RaceOpenGLWidget(self.animator)
        right_panel = self._build_right_panel()

        layout.addWidget(left_panel, 0)
        layout.addWidget(self.viewport, 1)
        layout.addWidget(right_panel, 0)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QWidget { background: #171717; color: #f2f2f2; font-size: 14px; }
            QGroupBox { border: 2px solid #555; margin-top: 12px; padding: 10px; font-weight: 700; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel.metric { font-family: monospace; font-size: 20px; font-weight: 700; }
            QPushButton { background: #2d2d2d; border: 1px solid #666; padding: 6px 8px; }
            QPushButton:checked { background: #204d7a; border-color: #5aa6ff; }
            """
        )

    def _build_left_panel(self):
        panel = QFrame()
        panel.setFixedWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.left_canvas = self._build_left_hud_canvas()
        layout.addWidget(self.left_canvas, 1)
        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setFixedWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.right_canvas = self._build_right_hud_canvas()
        layout.addWidget(self.right_canvas, 1)
        camera_group = QGroupBox("Camera")
        camera_layout = QGridLayout(camera_group)
        for idx, (label, mode) in enumerate([
            ("1 Follow", "follow"),
            ("2 Rear", "rear_view"),
            ("3 Top", "top_down"),
            ("4 Overview", "overview"),
            ("5 Free", "free"),
        ]):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=mode: self._set_camera(value))
            self.camera_buttons[mode] = button
            camera_layout.addWidget(button, idx // 2, idx % 2)

        self.play_button = QPushButton("Pause")
        self.play_button.clicked.connect(self._toggle_playing)
        camera_layout.addWidget(self.play_button, 3, 0)

        self.diag_button = QPushButton("Diagnostics")
        self.diag_button.setCheckable(True)
        self.diag_button.setChecked(self.animator.show_diagnostics)
        self.diag_button.clicked.connect(self._toggle_diagnostics)
        camera_layout.addWidget(self.diag_button, 3, 1)

        layout.addWidget(camera_group)
        layout.addWidget(self._build_controls_group())
        self._sync_camera_buttons()
        return panel

    def _hud_figure(self, width, height):
        fig = Figure(figsize=(width, height), facecolor="#171717")
        fig.subplots_adjust(0, 0, 1, 1)
        canvas = FigureCanvas(fig)
        canvas.setStyleSheet("background: #171717;")
        return fig, canvas

    def _style_axes(self, ax, title=None):
        ax.set_facecolor("#0a0a0a")
        ax.grid(True, alpha=0.18, color="#333333")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(1.5)
        if title:
            ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)

    def _build_left_hud_canvas(self):
        fig, canvas = self._hud_figure(3.6, 9.8)
        self._setup_minimap(fig.add_axes([0.08, 0.64, 0.84, 0.32]))
        self._setup_friction_axes(
            fig.add_axes([0.13, 0.43, 0.74, 0.16]),
            fig.add_axes([0.13, 0.25, 0.74, 0.16]),
        )
        self._setup_gg_diagram(fig.add_axes([0.14, 0.065, 0.74, 0.145]))
        return canvas

    def _build_right_hud_canvas(self):
        fig, canvas = self._hud_figure(3.8, 7.0)
        self._setup_steering_wheel(fig.add_axes([0.22, 0.72, 0.56, 0.24]))
        self._setup_dashboard_axes(fig.add_axes([0.06, 0.38, 0.88, 0.28]))
        self._setup_telemetry_bars(fig.add_axes([0.14, 0.08, 0.78, 0.22]))
        return canvas

    def _setup_minimap(self, ax):
        self._style_axes(ax, "Track Overview")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(self.data.xc, self.data.yc, color="#666666", linestyle=":", linewidth=1.0, alpha=0.5)
        ax.plot(self.data.x_inner, self.data.y_inner, color="#00ff00", linewidth=1.4, alpha=0.6)
        ax.plot(self.data.x_outer, self.data.y_outer, color="#00ff00", linewidth=1.4, alpha=0.6)
        xmin = min(np.min(self.data.x_inner), np.min(self.data.x_outer)) - 10
        xmax = max(np.max(self.data.x_inner), np.max(self.data.x_outer)) + 10
        ymin = min(np.min(self.data.y_inner), np.min(self.data.y_outer)) - 10
        ymax = max(np.max(self.data.y_inner), np.max(self.data.y_outer)) + 10
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.invert_yaxis()
        self.mini_carF, = ax.plot([], [], "o", color="#ff3333", markersize=10, markeredgecolor="white", markeredgewidth=1.8)
        self.mini_carL, = ax.plot([], [], "o", color="#3366ff", markersize=10, markeredgecolor="white", markeredgewidth=1.8)

    def _setup_steering_wheel(self, ax):
        self._style_axes(ax, "Steering (Follower)")
        ax.title.set_color("#ff3333")
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_patch(patches.Circle((0, 0), 1.0, fill=False, edgecolor="#cccccc", linewidth=4))
        ax.add_patch(patches.Circle((0, 0), 0.88, fill=False, edgecolor="#999999", linewidth=2))
        ax.add_patch(patches.Circle((0, 0), 0.18, facecolor="#333333", edgecolor="#666666", linewidth=2))
        for ang in np.deg2rad([90, 210, 330]):
            x = np.array([0.0, 0.82 * np.cos(ang)])
            y = np.array([0.0, 0.82 * np.sin(ang)])
            line, = ax.plot(x, y, color="#aaaaaa", linewidth=5, solid_capstyle="round")
            self.spokes_lines.append(line)
        self.center_mark = ax.plot([0, 0], [0.22, 0.35], color="#ff0000", linewidth=4, solid_capstyle="round")[0]
        self.steer_text = ax.text(
            0,
            -0.65,
            "",
            fontsize=12,
            color="#ffffff",
            ha="center",
            fontweight="bold",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#2a2a2a", edgecolor="#555555", linewidth=2),
        )

    def _setup_gg_diagram(self, ax):
        self._style_axes(ax, "G-G Diagram")
        ax.set_xlim(-self.data.AY_LIM * 1.1, self.data.AY_LIM * 1.1)
        ax.set_ylim(-self.data.AX_LIM * 1.1, self.data.AX_LIM * 1.1)
        ax.set_xlabel("Lateral (m/s^2)", fontsize=8, color="white", fontweight="bold")
        ax.set_ylabel("Longitudinal", fontsize=8, color="white", fontweight="bold")
        ax.axhline(0, color="#666666", linewidth=1.2)
        ax.axvline(0, color="#666666", linewidth=1.2)
        ax.plot(self.data.ay_L, self.data.ax_L, color="#3366ff", alpha=0.15, linewidth=1.0)
        ax.plot(self.data.ay_F, self.data.ax_F, color="#ff3333", alpha=0.15, linewidth=1.0)
        self.gg_leader_dot, = ax.plot([], [], "o", color="#3366ff", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
        self.gg_follower_dot, = ax.plot([], [], "o", color="#ff3333", markersize=8, markeredgecolor="white", markeredgewidth=1.5)

    def _setup_dashboard_axes(self, ax):
        ax.set_facecolor("#0a0a0a")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")
            spine.set_linewidth(1.5)
        self.dash_vals = {
            "time": ax.text(0.50, 0.86, "", transform=ax.transAxes, fontsize=18, color="#00ff00", ha="center", fontweight="bold", family="monospace"),
            "spdF": ax.text(0.50, 0.70, "", transform=ax.transAxes, fontsize=13, color="#ff3333", ha="center", fontweight="bold", family="monospace"),
            "spdL": ax.text(0.50, 0.58, "", transform=ax.transAxes, fontsize=13, color="#3366ff", ha="center", fontweight="bold", family="monospace"),
            "gapS": ax.text(0.50, 0.39, "", transform=ax.transAxes, fontsize=12, color="#ffff00", ha="center", fontweight="bold", family="monospace"),
            "gapD": ax.text(0.50, 0.30, "", transform=ax.transAxes, fontsize=10, color="#aaaaaa", ha="center", family="monospace"),
            "p1": ax.text(0.50, 0.10, "", transform=ax.transAxes, fontsize=12, color="#ffffff", ha="center", fontweight="bold"),
        }
        ax.text(0.28, 0.87, "TIME", transform=ax.transAxes, fontsize=9, color="#888888", ha="center", fontweight="bold")
        ax.text(0.08, 0.70, "F:", transform=ax.transAxes, fontsize=12, color="#ff3333", ha="left", fontweight="bold")
        ax.text(0.08, 0.58, "L:", transform=ax.transAxes, fontsize=12, color="#3366ff", ha="left", fontweight="bold")
        ax.text(0.50, 0.47, "GAP (Arc Length)", transform=ax.transAxes, fontsize=9, color="#888888", ha="center", fontweight="bold")
        ax.text(0.50, 0.20, "POSITION", transform=ax.transAxes, fontsize=9, color="#888888", ha="center", fontweight="bold")

    def _setup_telemetry_bars(self, ax):
        self._style_axes(ax, "Accel & Input (Follower: Red, Leader: Blue)")
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
        ax.set_yticklabels(["-100%", "-50%", "0", "+50%", "+100%"], fontsize=8, color="white")
        ax.axhline(0, color="#666666", linewidth=2)
        ax.grid(True, axis="y", alpha=0.2, color="#333333")

        def make_dual_bar(x_pos, label_text):
            bar_offset = 0.04
            bar_half = 0.075
            f_pos = ax.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color="#ff3333", edgecolor="white", linewidth=1)[0]
            f_neg = ax.bar(x_pos - bar_offset, 0, bar_half, bottom=0, color="#990000", edgecolor="white", linewidth=1)[0]
            l_pos = ax.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color="#3366ff", edgecolor="white", linewidth=1)[0]
            l_neg = ax.bar(x_pos + bar_offset, 0, bar_half, bottom=0, color="#000099", edgecolor="white", linewidth=1)[0]
            ax.text(x_pos, -1.18, label_text, ha="center", va="top", fontsize=8, color="white", fontweight="bold")
            return (f_pos, f_neg, l_pos, l_neg)

        self.bar_throttle = make_dual_bar(0.2, "Throttle/\nBrake")
        self.bar_ax = make_dual_bar(0.5, "Accel X")
        self.bar_ay = make_dual_bar(0.8, "Accel Y")

    def _setup_friction_axes(self, ax_front, ax_rear):
        for ax, title in ((ax_front, "Front Axle"), (ax_rear, "Rear Axle")):
            self._style_axes(ax, title)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("Lateral (N)", fontsize=8, color="white", fontweight="bold")
            ax.set_ylabel("Longitudinal", fontsize=8, color="white", fontweight="bold")
        self.front_dot_F, = ax_front.plot([], [], "o", color="#ff3333", mec="white", mew=1.3, ms=7, label="Follower")
        self.front_dot_L, = ax_front.plot([], [], "o", color="#3366ff", mec="white", mew=1.3, ms=7, label="Leader")
        self.rear_dot_F, = ax_rear.plot([], [], "o", color="#ff3333", mec="white", mew=1.3, ms=7, label="Follower")
        self.rear_dot_L, = ax_rear.plot([], [], "o", color="#3366ff", mec="white", mew=1.3, ms=7, label="Leader")
        fcx = float(np.nanmax(np.abs([self.data.Fxmax_f_F, self.data.Fxmax_f_L, self.data.Fxmax_r_F, self.data.Fxmax_r_L])) + 1e-6)
        fcy = float(np.nanmax(np.abs([self.data.Fymax_f_F, self.data.Fymax_f_L, self.data.Fymax_r_F, self.data.Fymax_r_L])) + 1e-6)
        for ax in (ax_front, ax_rear):
            ax.set_xlim(-1.1 * fcx, 1.1 * fcx)
            ax.set_ylim(-1.1 * fcy, 1.1 * fcy)

    def _group(self, title, rows):
        group = QGroupBox(title)
        layout = QGridLayout(group)
        for row, (label_text, key) in enumerate(rows):
            label = QLabel(label_text)
            value = QLabel("")
            value.setProperty("class", "metric")
            self.labels[key] = value
            layout.addWidget(label, row, 0)
            layout.addWidget(value, row, 1)
        return group

    def _build_controls_group(self):
        group = QGroupBox("Controls")
        layout = QGridLayout(group)
        label = QLabel("Keys")
        value = QLabel("Space play/pause\n1-5 cameras\nV diagnostics\nFree: mouse + WASD/QE/R")
        value.setStyleSheet("font-family: monospace; font-size: 13px; font-weight: 700;")
        layout.addWidget(label, 0, 0)
        layout.addWidget(value, 0, 1)
        return group

    def _tick(self):
        if self.animator.playing:
            self.animator.current_frame = (self.animator.current_frame + 1) % self.animator.tNum
        self._update_hud()
        self.viewport.update()

    def _update_hud(self):
        i = min(self.animator.current_frame, len(self.data.t) - 1)
        self._update_matplotlib_hud(i)

    def _update_matplotlib_hud(self, i):
        xF, yF, xL, yL = self.precomputed_data.minimap_data[i]
        self.mini_carF.set_data([xF], [yF])
        self.mini_carL.set_data([xL], [yL])

        spoke_data, center_mark_data, steer_deg = self.precomputed_data.steering_data[i]
        for line, (x_data, y_data) in zip(self.spokes_lines, spoke_data):
            line.set_data(x_data, y_data)
        self.center_mark.set_data(*center_mark_data)
        self.steer_text.set_text(f"{steer_deg:+.1f} deg")

        ay_L, ax_L, ay_F, ax_F = self.precomputed_data.gg_data[i]
        self.gg_leader_dot.set_data([ay_L], [ax_L])
        self.gg_follower_dot.set_data([ay_F], [ax_F])

        time_val, spdF_val, spdL_val, gap_s_val, gap_xy_val = self.precomputed_data.dashboard_data[i]
        self.dash_vals["time"].set_text(f"{time_val:05.2f}s")
        self.dash_vals["spdF"].set_text(f"{spdF_val:6.2f} m/s")
        self.dash_vals["spdL"].set_text(f"{spdL_val:6.2f} m/s")
        self.dash_vals["gapS"].set_text(f"{gap_s_val:+.2f} m")
        self.dash_vals["gapS"].set_color("#00ff00" if gap_s_val > 0 else "#ff3333")
        self.dash_vals["gapD"].set_text(f"(Euclidean: {gap_xy_val:.2f}m)")
        if gap_s_val > 0:
            self.dash_vals["p1"].set_text("P1: Follower  P2: Leader")
            self.dash_vals["p1"].set_color("#00ff00")
        else:
            self.dash_vals["p1"].set_text("P1: Leader  P2: Follower")
            self.dash_vals["p1"].set_color("#ffaa00")

        norm_ax_F, norm_ax_L, norm_ay_F, norm_ay_L = self.precomputed_data.bar_data[i]
        self._update_dual_bar(self.bar_throttle, norm_ax_F, norm_ax_L)
        self._update_dual_bar(self.bar_ax, norm_ax_F, norm_ax_L)
        self._update_dual_bar(self.bar_ay, norm_ay_F, norm_ay_L)

        Ffy_F, Ffx_F, Ffy_L, Ffx_L, Fry_F, Frx_F, Fry_L, Frx_L = self.precomputed_data.friction_data[i]
        self.front_dot_F.set_data([Ffy_F], [Ffx_F])
        self.front_dot_L.set_data([Ffy_L], [Ffx_L])
        self.rear_dot_F.set_data([Fry_F], [Frx_F])
        self.rear_dot_L.set_data([Fry_L], [Frx_L])
        self.left_canvas.draw_idle()
        self.right_canvas.draw_idle()

    def _update_dual_bar(self, bars, f_val, l_val):
        f_pos, f_neg, l_pos, l_neg = bars
        if f_val >= 0:
            f_pos.set_height(f_val)
            f_pos.set_y(0)
            f_neg.set_height(0)
        else:
            f_neg.set_height(-f_val)
            f_neg.set_y(f_val)
            f_pos.set_height(0)
        if l_val >= 0:
            l_pos.set_height(l_val)
            l_pos.set_y(0)
            l_neg.set_height(0)
        else:
            l_neg.set_height(-l_val)
            l_neg.set_y(l_val)
            l_pos.set_height(0)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()
        elif key == Qt.Key_Space:
            self._toggle_playing()
        elif key == Qt.Key_V:
            self._toggle_diagnostics()
        elif key == Qt.Key_1:
            self._set_camera("follow")
        elif key == Qt.Key_2:
            self._set_camera("rear_view")
        elif key == Qt.Key_3:
            self._set_camera("top_down")
        elif key == Qt.Key_4:
            self._set_camera("overview")
        elif key == Qt.Key_5:
            self._set_camera("free")
        elif key == Qt.Key_Right and not self.animator.playing:
            self.animator.current_frame = min(self.animator.current_frame + 1, self.animator.tNum - 1)
        elif key == Qt.Key_Left and not self.animator.playing:
            self.animator.current_frame = max(self.animator.current_frame - 1, 0)
        elif self.animator.camera_mode == "free":
            self._handle_free_camera_key(key)

    def _handle_free_camera_key(self, key):
        pan_speed = 15
        if key == Qt.Key_W:
            self.animator.camera_target[1] += pan_speed
        elif key == Qt.Key_S:
            self.animator.camera_target[1] -= pan_speed
        elif key == Qt.Key_A:
            self.animator.camera_target[0] -= pan_speed
        elif key == Qt.Key_D:
            self.animator.camera_target[0] += pan_speed
        elif key == Qt.Key_E:
            self.animator.camera_target[2] += pan_speed
        elif key == Qt.Key_Q:
            self.animator.camera_target[2] -= pan_speed
        elif key == Qt.Key_R:
            self.animator.camera_distance = 800
            self.animator.camera_angle_h = 45
            self.animator.camera_angle_v = 30

    def _toggle_playing(self):
        self.animator.playing = not self.animator.playing
        self.play_button.setText("Pause" if self.animator.playing else "Play")

    def _toggle_diagnostics(self):
        self.animator.show_diagnostics = not self.animator.show_diagnostics
        self.diag_button.setChecked(self.animator.show_diagnostics)

    def _set_camera(self, mode):
        self.animator.camera_mode = mode
        self._sync_camera_buttons()

    def _sync_camera_buttons(self):
        for mode, button in self.camera_buttons.items():
            button.setChecked(mode == self.animator.camera_mode)


def configure_surface_format():
    fmt = QSurfaceFormat()
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)


def build_parser():
    parser = argparse.ArgumentParser(description="Run the Qt Stackelberg animator HUD.")
    parser.add_argument("leader_file", help="Leader trajectory .mat file")
    parser.add_argument("follower_file", help="Follower trajectory .mat file")
    parser.add_argument(
        "track_file",
        nargs="?",
        default="NASCAR_Track_Monge_v3.mat",
        help="Track .mat file, default: NASCAR_Track_Monge_v3.mat",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--camera",
        choices=["follow", "rear_view", "top_down", "overview", "free"],
        default="follow",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--car-scale", type=float, default=1.5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    configure_surface_format()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QtRaceWindow(
        args.leader_file,
        args.follower_file,
        args.track_file,
        fps=args.fps,
        camera_mode=args.camera,
        diagnostics=args.diagnostics,
        car_scale=args.car_scale,
    )
    window.resize(1800, 1000)
    window.show()
    return app.exec()
