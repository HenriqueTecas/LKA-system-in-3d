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
        self.font_small = pygame.font.Font(None, 22)
        self.font_large = pygame.font.Font(None, 36)
        self.fps_history = []
        self.fps_update_counter = 0
        self.lka_logger = None  # Will be set externally if logging enabled

    def render(self, surface, car, camera, current_fps, camera_view_mode="chase", lka_controller=None, lka_warnings=None):
        """Render HUD overlays"""
        # FPS counter
        self._draw_fps(surface, current_fps)

        # LKA Controller status (3-mode system)
        if lka_controller:
            self._draw_lka_status(surface, lka_controller, lka_warnings or {})

        # Enhanced LKA metrics panel (right side)
        if lka_controller and lka_controller.mode != lka_controller.MODE_MANUAL:
            self._draw_lka_metrics_panel(surface, car, lka_controller)

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

    def _draw_lka_status(self, surface, lka, warnings):
        """Draw LKA Controller status indicator with mode and warnings"""
        mode_name = lka.get_mode_name()

        # Color based on mode
        if mode_name == "MANUAL":
            color = (150, 150, 150)  # Gray
            prefix = "[MANUAL]"
        elif mode_name == "WARNING":
            color = YELLOW
            prefix = "[WARN]"
        elif mode_name == "ASSIST":
            color = GREEN
            prefix = "[ASSIST]"
        else:
            color = WHITE
            prefix = "[MODE]"

        status_text = f"{prefix} LKA (1/2/3)"
        text = self.font_large.render(status_text, True, color)
        rect = text.get_rect(center=(WIDTH // 2, 30))

        bg_rect = rect.inflate(20, 10)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)
        surface.blit(text, rect)

        # Show warnings if in WARNING or ASSIST mode
        if warnings and (mode_name == "WARNING" or mode_name == "ASSIST"):
            warning_y = 70

            if warnings.get('lane_departure'):
                warning_text = self.font.render("Going out of lane!", True, RED)
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)
                warning_y += 35

            if warnings.get('speed_too_high'):
                warning_text = self.font.render("Speed too high for curve", True, (255, 150, 0))
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)
                warning_y += 35

            if warnings.get('time_to_crossing'):
                warning_text = self.font.render("Lane crossing imminent", True, RED)
                rect = warning_text.get_rect(center=(WIDTH // 2, warning_y))
                bg_rect = rect.inflate(15, 8)
                s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
                pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
                surface.blit(s, bg_rect.topleft)
                surface.blit(warning_text, rect)

        # Intervention strength if in ASSIST mode
        if mode_name == "ASSIST" and hasattr(lka, 'intervention_strength'):
            intervention = lka.intervention_strength
            if intervention >= 0.5:
                interv_text = "ASSIST ACTIVE"
                color_interv = (255, 180, 0)
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
            f"Limits: steer {'SAT' if getattr(car, 'steering_saturated', False) else 'OK'} | vel {'SAT' if getattr(car, 'velocity_saturated', False) else 'OK'}",
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

        bg_rect = rect.inflate(10, 5)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 180), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)

        surface.blit(text, rect)

    def _draw_lka_metrics_panel(self, surface, car, lka_controller):
        """Draw comprehensive LKA performance metrics panel"""
        # Get metrics from logger if available, otherwise compute directly
        if self.lka_logger:
            metrics = self.lka_logger.get_current_metrics()
        else:
            metrics = self._compute_metrics_direct(car, lka_controller)
        
        # Panel dimensions
        panel_width = 320
        panel_height = 360
        panel_x = WIDTH - panel_width - 10
        panel_y = 570  # Below FPS counter
        
        # Draw panel background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (0, 0, 0, 180), (0, 0, panel_width, panel_height), border_radius=8)
        pygame.draw.rect(panel_surface, (100, 100, 100, 255), (0, 0, panel_width, panel_height), 2, border_radius=8)
        surface.blit(panel_surface, (panel_x, panel_y))
        
        # Title
        title = self.font_large.render("LKA METRICS", True, (200, 200, 255))
        surface.blit(title, (panel_x + 10, panel_y + 5))
        
        y_offset = panel_y + 45
        
        # 1. Lateral Error (color-coded)
        lateral_error = metrics.get('lateral_error', 0.0)
        if abs(lateral_error) < 0.5:
            color = GREEN
        elif abs(lateral_error) < 1.0:
            color = YELLOW
        else:
            color = RED
        
        label = self.font_small.render("Lateral Error:", True, WHITE)
        value = self.font.render(f"{lateral_error:+.3f} m", True, color)
        surface.blit(label, (panel_x + 10, y_offset))
        surface.blit(value, (panel_x + 160, y_offset))
        y_offset += 30
        
        # 2. Heading Error (degrees)
        heading_error = metrics.get('heading_error', 0.0)
        label = self.font_small.render("Heading Error:", True, WHITE)
        value = self.font.render(f"{heading_error:+.1f}°", True, WHITE)
        surface.blit(label, (panel_x + 10, y_offset))
        surface.blit(value, (panel_x + 160, y_offset))
        y_offset += 30
        
        # 3. Speed Ratio (v/v_safe as percentage)
        speed_ratio = metrics.get('speed_ratio', 0.0)
        if speed_ratio < 80:
            ratio_color = GREEN
        elif speed_ratio < 100:
            ratio_color = YELLOW
        else:
            ratio_color = RED
        
        label = self.font_small.render("Speed Ratio:", True, WHITE)
        value = self.font.render(f"{speed_ratio:.0f}%", True, ratio_color)
        surface.blit(label, (panel_x + 10, y_offset))
        surface.blit(value, (panel_x + 160, y_offset))
        y_offset += 30
        
        # Draw small bar graph for speed ratio
        bar_width = 200
        bar_height = 12
        bar_x = panel_x + 10
        bar_y = y_offset
        
        # Background bar
        pygame.draw.rect(surface, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        # Filled bar
        filled_width = int(min(speed_ratio / 100.0, 1.2) * bar_width)
        pygame.draw.rect(surface, ratio_color, (bar_x, bar_y, filled_width, bar_height))
        # Border
        pygame.draw.rect(surface, WHITE, (bar_x, bar_y, bar_width, bar_height), 1)
        y_offset += 25
        
        # 4. Curve Radius
        curve_radius = metrics.get('curve_radius', float('inf'))
        label = self.font_small.render("Curve Radius:", True, WHITE)
        if curve_radius > 500:
            radius_text = "STRAIGHT"
            radius_color = GREEN
        else:
            radius_text = f"{curve_radius:.0f} m"
            radius_color = YELLOW if curve_radius < 200 else WHITE
        value = self.font.render(radius_text, True, radius_color)
        surface.blit(label, (panel_x + 10, y_offset))
        surface.blit(value, (panel_x + 160, y_offset))
        y_offset += 30
        
        # 5. Lookahead Distance
        lookahead = metrics.get('lookahead', 0.0)
        label = self.font_small.render("Lookahead:", True, WHITE)
        value = self.font.render(f"{lookahead:.1f} m", True, (150, 255, 150))
        surface.blit(label, (panel_x + 10, y_offset))
        surface.blit(value, (panel_x + 160, y_offset))
        y_offset += 30
        
        # 6. Steering Angle (dial/gauge)
        steering_angle = metrics.get('steering_angle', 0.0)
        label = self.font_small.render("Steering Angle:", True, WHITE)
        surface.blit(label, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Draw steering gauge
        gauge_center_x = panel_x + panel_width // 2
        gauge_center_y = y_offset + 40
        gauge_radius = 35
        
        self._draw_steering_gauge(surface, gauge_center_x, gauge_center_y, gauge_radius, steering_angle)
        y_offset += 90
        
        # 7. Intervention State
        intervening = metrics.get('intervening', False)
        if intervening:
            state_text = "ASSISTING"
            state_color = (255, 180, 0)
        else:
            state_text = "CENTERED"
            state_color = GREEN
        
        state_label = self.font.render(state_text, True, state_color)
        state_rect = state_label.get_rect(center=(panel_x + panel_width // 2, y_offset))
        
        # Background for state
        bg_rect = state_rect.inflate(20, 10)
        state_bg = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(state_bg, (0, 0, 0, 200), (0, 0, bg_rect.width, bg_rect.height), border_radius=5)
        pygame.draw.rect(state_bg, state_color, (0, 0, bg_rect.width, bg_rect.height), 2, border_radius=5)
        surface.blit(state_bg, bg_rect.topleft)
        surface.blit(state_label, state_rect)
    
    def _draw_steering_gauge(self, surface, center_x, center_y, radius, angle_deg):
        """Draw circular steering angle gauge"""
        # Background circle
        pygame.draw.circle(surface, (40, 40, 40), (center_x, center_y), radius)
        pygame.draw.circle(surface, (100, 100, 100), (center_x, center_y), radius, 2)
        
        # Tick marks at -35, 0, +35 degrees
        for tick_angle in [-35, 0, 35]:
            tick_rad = np.radians(tick_angle - 90)  # Rotate to top = 0
            tick_start_r = radius - 8
            tick_end_r = radius - 2
            
            x1 = center_x + tick_start_r * np.cos(tick_rad)
            y1 = center_y + tick_start_r * np.sin(tick_rad)
            x2 = center_x + tick_end_r * np.cos(tick_rad)
            y2 = center_y + tick_end_r * np.sin(tick_rad)
            
            tick_color = RED if abs(tick_angle) == 35 else WHITE
            pygame.draw.line(surface, tick_color, (x1, y1), (x2, y2), 2)
        
        # Center dot
        pygame.draw.circle(surface, WHITE, (center_x, center_y), 3)
        
        # Steering angle needle
        # Clamp to -35 to +35 degrees
        clamped_angle = np.clip(angle_deg, -35, 35)
        needle_rad = np.radians(clamped_angle - 90)  # Rotate to top = 0
        needle_length = radius - 10
        
        needle_x = center_x + needle_length * np.cos(needle_rad)
        needle_y = center_y + needle_length * np.sin(needle_rad)
        
        # Color based on saturation
        if abs(angle_deg) >= 34:
            needle_color = RED
        elif abs(angle_deg) >= 25:
            needle_color = YELLOW
        else:
            needle_color = GREEN
        
        pygame.draw.line(surface, needle_color, (center_x, center_y), (needle_x, needle_y), 3)
        pygame.draw.circle(surface, needle_color, (int(needle_x), int(needle_y)), 4)
        
        # Angle value below gauge
        angle_text = self.font_small.render(f"{angle_deg:+.1f}°", True, needle_color)
        angle_rect = angle_text.get_rect(center=(center_x, center_y + radius + 15))
        surface.blit(angle_text, angle_rect)
    
    def _compute_metrics_direct(self, car, controller):
        """Compute metrics directly when logger not available"""
        lateral_error = controller._estimate_lane_offset() if hasattr(controller, '_estimate_lane_offset') else 0.0
        
        heading_error = 0.0
        if controller.target_direction is not None:
            heading_error = (controller.target_direction - controller.car.theta + np.pi) % (2 * np.pi) - np.pi
            heading_error = np.degrees(heading_error)
        
        speed = abs(car.velocity)
        safe_speed = controller.safe_speed if controller.safe_speed else speed
        speed_ratio = (speed / safe_speed * 100) if safe_speed > 0 else 0.0
        
        curve_radius = controller._last_curve_radius if controller._last_curve_radius else float('inf')
        
        lookahead = np.clip(speed * controller.lookahead_speed_scale, 
                           controller.lookahead_min, 
                           controller.lookahead_max)
        
        steering_angle = np.degrees(car.steering_angle)
        
        return {
            'lateral_error': lateral_error,
            'heading_error': heading_error,
            'speed': speed,
            'safe_speed': safe_speed,
            'speed_ratio': speed_ratio,
            'curve_radius': curve_radius,
            'lookahead': lookahead,
            'steering_angle': steering_angle,
            'intervening': controller.intervening,
        }

