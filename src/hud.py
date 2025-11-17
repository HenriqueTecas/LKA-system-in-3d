"""
HUD Module - Heads-Up Display
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from .config import *


class HUD:
    """Heads-up display for 3D view"""
    def __init__(self):
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        self.fps_history = []
        self.fps_update_counter = 0

    def render(self, surface, car, camera, lka, current_fps, mpc=None):
        """Render HUD overlays"""
        # FPS counter
        self._draw_fps(surface, current_fps)

        # LKA status (both Pure Pursuit and MPC)
        self._draw_lka_status(surface, lka, mpc)

        # Speed and steering info
        self._draw_telemetry(surface, car)

        # Lane detection status
        self._draw_lane_status(surface, camera)

    def _draw_fps(self, surface, fps):
        """Draw FPS counter with smooth averaging"""
        # Add to history
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:  # Keep last 30 frames
            self.fps_history.pop(0)

        # Calculate average FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps

        # Color based on performance
        if avg_fps >= 55:
            color = GREEN
        elif avg_fps >= 40:
            color = YELLOW
        else:
            color = RED

        text = self.font.render(f"FPS: {avg_fps:.1f}", True, color)
        # Position below minimap (minimap is 500px + 10 margin + 10 padding = 520)
        rect = text.get_rect(topright=(WIDTH - 10, 530))

        # Background
        bg_rect = rect.inflate(10, 5)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)

        surface.blit(text, rect)

    def _draw_lka_status(self, surface, lka, mpc=None):
        """Draw LKA status indicator"""
        if lka.active:
            status_text = "LKA: Pure Pursuit"
            color = GREEN
        elif mpc and mpc.active:
            status_text = "LKA: MPC"
            color = (0, 200, 255)  # Cyan
        else:
            status_text = "LKA: OFF"
            color = RED

        text = self.font_large.render(status_text, True, color)
        rect = text.get_rect(center=(WIDTH // 2, 30))

        # Background
        bg_rect = rect.inflate(20, 10)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)

        surface.blit(text, rect)

    def _draw_telemetry(self, surface, car):
        """Draw speed and steering information"""
        # Convert m/s to km/h
        speed_kmh = abs(car.velocity) * 3.6
        texts = [
            f"Speed: {speed_kmh:.1f} km/h ({abs(car.velocity):.1f} m/s)",
            f"Steering: {np.degrees(car.steering_angle):.1f}°",
            f"Throttle: {car.throttle_state:.2f} | Brake: {car.brake_state:.2f}",
        ]

        y = HEIGHT - 100
        for text in texts:
            rendered = self.font.render(text, True, WHITE)
            # Background
            rect = rendered.get_rect(topleft=(10, y))
            bg_rect = rect.inflate(10, 5)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
            surface.blit(s, bg_rect.topleft)

            surface.blit(rendered, rect)
            y += 30

    def _draw_lane_status(self, surface, camera):
        """Draw lane detection status"""
        texts = [
            f"Lane: {camera.current_lane}",
            f"Left: {'OK' if camera.left_lane_detected else 'NO'}",
            f"Right: {'OK' if camera.right_lane_detected else 'NO'}",
        ]

        y = 80
        for text in texts:
            color = GREEN if ('OK' in text or 'LEFT' in text or 'RIGHT' in text) else WHITE
            if 'NO' in text:
                color = RED

            rendered = self.font.render(text, True, color)
            rect = rendered.get_rect(topleft=(10, y))
            bg_rect = rect.inflate(10, 5)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
            surface.blit(s, bg_rect.topleft)

            surface.blit(rendered, rect)
            y += 30


