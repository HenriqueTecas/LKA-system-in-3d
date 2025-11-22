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

    def render(self, surface, car, camera, current_fps, camera_view_mode="chase", hybrid_controller=None, hybrid_warnings=None):
        """Render HUD overlays"""
        # FPS counter
        self._draw_fps(surface, current_fps)

        # Hybrid Controller status (new 3-mode system)
        if hybrid_controller:
            self._draw_hybrid_status(surface, hybrid_controller, hybrid_warnings)

        # Speed and steering info
        self._draw_telemetry(surface, car)

        # Lane detection status
        self._draw_lane_status(surface, camera)
        
        # Camera view mode
        self._draw_camera_mode(surface, camera_view_mode)

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

    def _draw_hybrid_status(self, surface, hybrid, warnings):
        """Draw Hybrid Controller status indicator with mode and warnings"""
        mode_name = hybrid.get_mode_name()
        
        # Color based on mode
        if mode_name == "MANUAL":
            color = (150, 150, 150)  # Gray
            icon = "⚙"
        elif mode_name == "WARNING":
            color = YELLOW
            icon = "⚠"
        elif mode_name == "ASSIST":
            color = GREEN
            icon = "🤖"
        else:
            color = WHITE
            icon = "?"
        
        # Main status
        status_text = f"{icon} MODE: {mode_name} (1/2/3)"
        text = self.font_large.render(status_text, True, color)
        rect = text.get_rect(center=(WIDTH // 2, 30))

        # Background
        bg_rect = rect.inflate(20, 10)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)
        surface.blit(text, rect)
        
        # Show warnings if in WARNING or ASSIST mode
        if warnings and (mode_name == "WARNING" or mode_name == "ASSIST"):
            warning_y = 70
            
            if warnings.get('lane_departure'):
                warning_text = self.font.render("⚠ LANE DEPARTURE", True, RED)
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)
                warning_y += 35
            
            if warnings.get('speed_too_high'):
                warning_text = self.font.render("⚠ SLOW DOWN FOR CURVE", True, (255, 150, 0))
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)
                warning_y += 35
            
            if warnings.get('time_to_crossing'):
                warning_text = self.font.render("⚠ LANE CROSSING IMMINENT", True, RED)
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)
        
        # Show intervention strength if in ASSIST mode
        if mode_name == "ASSIST" and hasattr(hybrid, 'intervention_strength'):
            intervention = hybrid.intervention_strength * 100
            if intervention > 0.1:
                interv_text = f"Assist: {intervention:.0f}%"
                color_interv = (int(255 * intervention / 100), int(255 * (1 - intervention / 100)), 0)
                text_interv = self.font.render(interv_text, True, color_interv)
                rect_interv = text_interv.get_rect(center=(WIDTH // 2, HEIGHT - 40))
                
                bg_rect = rect_interv.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(text_interv, rect_interv)

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
    
    def _draw_camera_mode(self, surface, mode):
        """Draw camera view mode indicator"""
        if mode == "realistic":
            text_str = "VIEW: Lane Camera (C)"
            color = (255, 200, 0)  # Orange
        else:
            text_str = "VIEW: Chase Cam (C)"
            color = (150, 150, 150)  # Gray
        
        text = self.font.render(text_str, True, color)
        rect = text.get_rect(topright=(WIDTH - 10, 10))
        
        # Background
        bg_rect = rect.inflate(10, 5)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)
        
        surface.blit(text, rect)

