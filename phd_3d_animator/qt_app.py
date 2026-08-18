"""Qt application shell with a real OpenGL widget in the HUD centre."""

from __future__ import annotations

import argparse
import sys

import numpy as np
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

from Stackelberg_HUD import DataProcessor

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
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._group("Track", [("Frame", "frame"), ("Time", "time"), ("Gap s", "gap_s")]))
        layout.addWidget(self._group("Follower", [("Speed", "speed_f"), ("Steer", "steer_f"), ("Accel X", "ax_f"), ("Accel Y", "ay_f")]))
        layout.addWidget(self._group("Leader", [("Speed", "speed_l"), ("Steer", "steer_l"), ("Accel X", "ax_l"), ("Accel Y", "ay_l")]))
        layout.addStretch(1)
        return panel

    def _build_right_panel(self):
        panel = QFrame()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)

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
        layout.addWidget(self._group("Position", [("P1", "position"), ("Euclidean gap", "gap_xy")]))
        layout.addWidget(self._group("Controls", [("Keys", "keys")]))
        layout.addStretch(1)
        self._sync_camera_buttons()
        self.labels["keys"].setText("Space play/pause\n1-5 cameras\nV diagnostics\nFree: mouse + WASD/QE/R")
        return panel

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

    def _tick(self):
        if self.animator.playing:
            self.animator.current_frame = (self.animator.current_frame + 1) % self.animator.tNum
        self._update_hud()
        self.viewport.update()

    def _update_hud(self):
        i = min(self.animator.current_frame, len(self.data.t) - 1)
        self.labels["frame"].setText(f"{i}/{len(self.data.t) - 1}")
        self.labels["time"].setText(f"{self.data.t[i]:.2f} s")
        self.labels["gap_s"].setText(f"{self.data.gap_s[i]:+.2f} m")
        self.labels["gap_xy"].setText(f"{self.data.gap_xy[i]:.2f} m")
        self.labels["speed_f"].setText(f"{self.data.spdF[i]:.2f} m/s")
        self.labels["speed_l"].setText(f"{self.data.spdL[i]:.2f} m/s")
        self.labels["steer_f"].setText(f"{np.degrees(self.data.deltaF_i[i]):+.1f} deg")
        self.labels["steer_l"].setText(f"{np.degrees(self.data.deltaL_i[i]):+.1f} deg")
        self.labels["ax_f"].setText(f"{self.data.ax_F_interp[i]:+.2f} m/s^2")
        self.labels["ay_f"].setText(f"{self.data.ay_F_interp[i]:+.2f} m/s^2")
        self.labels["ax_l"].setText(f"{self.data.ax_L_interp[i]:+.2f} m/s^2")
        self.labels["ay_l"].setText(f"{self.data.ay_L_interp[i]:+.2f} m/s^2")
        self.labels["position"].setText("Follower" if self.data.gap_s[i] > 0 else "Leader")

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
