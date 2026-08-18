"""Single-window Pygame application for the Stackelberg 3D animator."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE

if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

from .rendering import Vehicle3DAnimatorGL


class SingleWindowRaceApp:
    """Own the Pygame window and event loop for one race visualisation."""

    def __init__(
        self,
        leader_file,
        follower_file,
        track_file="NASCAR_Track_Monge_v3.mat",
        size=(1600, 1000),
        fps=30,
        camera_mode="follow",
        diagnostics=False,
        car_scale=1.5,
    ):
        self.size = size
        self.fps = fps
        self.animator = Vehicle3DAnimatorGL(
            leader_file,
            follower_file,
            track_file,
            car_visual_scale=car_scale,
        )
        self.animator.camera_mode = camera_mode
        self.animator.show_diagnostics = diagnostics
        self.running = True

    def run(self):
        pygame.init()
        pygame.display.set_mode(self.size, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Stackelberg 3D Animator")
        self.animator.configure_opengl(*self.size)

        clock = pygame.time.Clock()
        self._print_controls()

        while self.running:
            for event in pygame.event.get():
                self._handle_event(event)

            if self.animator.playing:
                self.animator.current_frame = (self.animator.current_frame + 1) % self.animator.tNum

            self.animator.render_frame(handle_input=True)
            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def _handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return

        if event.type == pygame.VIDEORESIZE:
            self.size = (max(320, event.w), max(240, event.h))
            pygame.display.set_mode(self.size, DOUBLEBUF | OPENGL | RESIZABLE)
            self.animator.configure_opengl(*self.size)
            return

        if event.type == pygame.KEYDOWN:
            self._handle_key(event.key)
            return

        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_button(event.button)
            return

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.animator.mouse_down = False
            return

        if event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion()
            return

        if event.type == pygame.MOUSEWHEEL and self.animator.camera_mode == "free":
            zoom_factor = 0.9 if event.y > 0 else 1.1
            self.animator.camera_distance = np.clip(self.animator.camera_distance * zoom_factor, 50, 5000)

    def _handle_key(self, key):
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.running = False
        elif key == pygame.K_SPACE:
            self.animator.playing = not self.animator.playing
        elif key == pygame.K_v:
            self.animator.show_diagnostics = not self.animator.show_diagnostics
        elif key == pygame.K_1:
            self.animator.camera_mode = "follow"
        elif key == pygame.K_2:
            self.animator.camera_mode = "rear_view"
        elif key == pygame.K_3:
            self.animator.camera_mode = "top_down"
        elif key == pygame.K_4:
            self.animator.camera_mode = "overview"
        elif key == pygame.K_5:
            self.animator.camera_mode = "free"
        elif key == pygame.K_RIGHT and not self.animator.playing:
            self.animator.current_frame = min(self.animator.current_frame + 1, self.animator.tNum - 1)
        elif key == pygame.K_LEFT and not self.animator.playing:
            self.animator.current_frame = max(self.animator.current_frame - 1, 0)

    def _handle_mouse_button(self, button):
        if button == 1:
            self.animator.mouse_down = True
            self.animator.last_mouse_pos = pygame.mouse.get_pos()
        elif button == 4 and self.animator.camera_mode == "free":
            self.animator.camera_distance = max(50, self.animator.camera_distance * 0.9)
        elif button == 5 and self.animator.camera_mode == "free":
            self.animator.camera_distance = min(5000, self.animator.camera_distance * 1.1)

    def _handle_mouse_motion(self):
        if not self.animator.mouse_down or self.animator.camera_mode != "free":
            return

        current_pos = pygame.mouse.get_pos()
        dx = current_pos[0] - self.animator.last_mouse_pos[0]
        dy = current_pos[1] - self.animator.last_mouse_pos[1]
        self.animator.camera_angle_h += dx * 0.3
        self.animator.camera_angle_v = np.clip(self.animator.camera_angle_v + dy * 0.3, -89, 89)
        self.animator.last_mouse_pos = current_pos

    @staticmethod
    def _print_controls():
        print("\n=== CONTROLS ===")
        print("Space: play/pause")
        print("1-5: camera modes")
        print("Left/Right: step when paused")
        print("Mouse drag/WASD/QE/R: free-camera controls")
        print("V: diagnostics")
        print("ESC/Q: exit")


def build_parser():
    parser = argparse.ArgumentParser(description="Run the single-window Stackelberg 3D animator.")
    parser.add_argument("leader_file", help="Leader trajectory .mat file")
    parser.add_argument("follower_file", help="Follower trajectory .mat file")
    parser.add_argument(
        "track_file",
        nargs="?",
        default="NASCAR_Track_Monge_v3.mat",
        help="Track .mat file, default: NASCAR_Track_Monge_v3.mat",
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
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
        default=1.5,
        help="Visual-only scale for the car blocks; does not affect vehicle maths.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    app = SingleWindowRaceApp(
        args.leader_file,
        args.follower_file,
        args.track_file,
        size=(args.width, args.height),
        fps=args.fps,
        camera_mode=args.camera,
        diagnostics=args.diagnostics,
        car_scale=args.car_scale,
    )
    app.run()
