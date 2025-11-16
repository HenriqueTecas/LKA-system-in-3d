"""
Minimap Module - 2D Top-Down View
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from .config import *


class Minimap:
    """2D minimap renderer - reuses original drawing code"""
    def __init__(self, size, track):
        self.size = size
        self.track = track
        self.surface = pygame.Surface((size, size))

        # Calculate track bounding box for proper scaling
        self._calculate_track_bounds()

    def _calculate_track_bounds(self):
        """Calculate bounding box of entire track"""
        # Get all track points including boundaries
        all_points = list(self.track.centerline)
        outer = self.track._offset_line(self.track.centerline, self.track.track_width / 2)
        inner = self.track._offset_line(self.track.centerline, -self.track.track_width / 2)
        all_points.extend(outer)
        all_points.extend(inner)

        # Find min/max coordinates
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)

        # Calculate scale to fit in minimap with margin
        margin = 20  # pixels
        track_width = self.max_x - self.min_x
        track_height = self.max_y - self.min_y

        # Scale to fit within minimap size minus margins
        scale_x = (self.size - 2 * margin) / track_width
        scale_y = (self.size - 2 * margin) / track_height

        # Use the smaller scale to maintain aspect ratio
        self.scale = min(scale_x, scale_y)
        self.margin = margin

        print("="*60)
        print("MINIMAP INITIALIZATION")
        print(f"Minimap size: {self.size}x{self.size}")
        print(f"Track bounds: X=[{self.min_x:.1f}, {self.max_x:.1f}] Y=[{self.min_y:.1f}, {self.max_y:.1f}]")
        print(f"Track dimensions: {track_width:.1f} x {track_height:.1f}")
        print(f"Scale: {self.scale:.4f} (scale_x={scale_x:.4f}, scale_y={scale_y:.4f})")
        print(f"Expected minimap bounds: ({self.margin}, {self.margin}) to ({int(track_width*self.scale + self.margin)}, {int(track_height*self.scale + self.margin)})")
        print("="*60)

    def _world_to_minimap(self, x, y):
        """Convert world coordinates to minimap coordinates"""
        # Translate to origin, scale, then translate to minimap with margin
        map_x = (x - self.min_x) * self.scale + self.margin
        map_y = (y - self.min_y) * self.scale + self.margin
        return int(map_x), int(map_y)

    def render(self, car, camera, lka):
        """Render minimap with original 2D view"""
        # Fill with semi-transparent dark background
        self.surface.fill((20, 20, 20))  # Very dark gray background

        # Draw border around minimap
        pygame.draw.rect(self.surface, (100, 100, 100), (0, 0, self.size, self.size), 3)
        pygame.draw.rect(self.surface, (200, 200, 200), (2, 2, self.size-4, self.size-4), 1)

        # DEBUG: Draw grid to show we're using full minimap space
        grid_color = (40, 40, 40)
        for i in range(0, self.size, 50):
            pygame.draw.line(self.surface, grid_color, (i, 0), (i, self.size), 1)
            pygame.draw.line(self.surface, grid_color, (0, i), (self.size, i), 1)

        # DEBUG: Draw expected bounds rectangle (should be near edges)
        min_scaled = self._world_to_minimap(self.min_x, self.min_y)
        max_scaled = self._world_to_minimap(self.max_x, self.max_y)
        pygame.draw.rect(self.surface, (255, 0, 0),
                        (min_scaled[0], min_scaled[1],
                         max_scaled[0] - min_scaled[0],
                         max_scaled[1] - min_scaled[1]), 2)

        # DEBUG: Label the bounds
        font = pygame.font.Font(None, 16)
        bounds_text = font.render(f"Bounds: {min_scaled} to {max_scaled}", True, (255, 0, 0))
        self.surface.blit(bounds_text, (5, self.size - 20))

        # Draw track
        self._draw_track_2d()

        # Draw camera FOV and detections
        self._draw_camera_view_2d(camera)

        # Draw ALL LKA lane center points (small yellow dots)
        if lka.active and hasattr(lka, 'lane_center_points') and lka.lane_center_points:
            for cx, cy, dist in lka.lane_center_points:
                center_scaled = self._world_to_minimap(cx, cy)
                pygame.draw.circle(self.surface, (255, 255, 100), center_scaled, 3)

        # Draw LKA selected lookahead point (larger, brighter)
        if lka.active and hasattr(lka, 'lookahead_point'):
            lx, ly = lka.lookahead_point
            car_scaled = self._world_to_minimap(car.x, car.y)
            lookahead_scaled = self._world_to_minimap(lx, ly)
            pygame.draw.line(self.surface, YELLOW, car_scaled, lookahead_scaled, 2)
            pygame.draw.circle(self.surface, YELLOW, lookahead_scaled, 7)  # Larger for selected point

        # Draw car (simple representation)
        self._draw_car_2d(car)

        # DEBUG: Draw scale info
        font = pygame.font.Font(None, 20)
        scale_text = font.render(f"Scale: {self.scale:.3f}", True, (255, 255, 0))
        self.surface.blit(scale_text, (5, 5))

        return self.surface

    def _draw_track_2d(self):
        """Draw track in minimap with proper scaling"""
        outer = self.track._offset_line(self.track.centerline, self.track.track_width / 2)
        inner = self.track._offset_line(self.track.centerline, -self.track.track_width / 2)

        # Convert to minimap coordinates
        outer_scaled = [self._world_to_minimap(x, y) for x, y in outer]
        inner_scaled = [self._world_to_minimap(x, y) for x, y in inner]

        if len(outer_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, outer_scaled, 2)
        if len(inner_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, inner_scaled, 2)

        # Draw centerline dashed
        centerline_scaled = [self._world_to_minimap(x, y) for x, y in self.track.centerline]
        for i in range(0, len(centerline_scaled) - 1, 2):
            p1 = centerline_scaled[i]
            p2 = centerline_scaled[i + 1]
            pygame.draw.line(self.surface, GRAY, p1, p2, 1)

    def _draw_camera_view_2d(self, camera):
        """Draw camera FOV and detected lanes"""
        camera_x, camera_y = camera.get_camera_position()

        # Draw FOV cone
        fov_points = [(camera_x, camera_y)]
        left_angle = camera.car.theta - camera.field_of_view / 2
        fov_points.append((
            camera_x + camera.max_range * np.cos(left_angle),
            camera_y + camera.max_range * np.sin(left_angle)
        ))
        right_angle = camera.car.theta + camera.field_of_view / 2
        fov_points.append((
            camera_x + camera.max_range * np.cos(right_angle),
            camera_y + camera.max_range * np.sin(right_angle)
        ))

        # Convert FOV points to minimap coordinates
        fov_points_scaled = [self._world_to_minimap(x, y) for x, y in fov_points]

        # Draw semi-transparent FOV
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.polygon(s, (0, 255, 0, 30), fov_points_scaled)
        self.surface.blit(s, (0, 0))

        # Draw FOV edges
        camera_pos_scaled = fov_points_scaled[0]
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[1], 1)
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[2], 1)

        # Draw detected lane points
        left_lane, right_lane, center_lane = camera.detect_lanes(camera.car.track)

        # Get wheel positions
        (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y) = camera.car.get_front_wheel_positions()
        left_wheel_scaled = self._world_to_minimap(left_wheel_x, left_wheel_y)
        right_wheel_scaled = self._world_to_minimap(right_wheel_x, right_wheel_y)

        # Draw left lane points with vectors from LEFT wheel
        for px, py, _ in left_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (255, 0, 0), (px_scaled, py_scaled), 3)
            # Vector from left wheel to left lane point
            pygame.draw.line(self.surface, (255, 128, 0), left_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw right lane points with vectors from RIGHT wheel
        for px, py, _ in right_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (0, 128, 255), (px_scaled, py_scaled), 3)
            # Vector from right wheel to right lane point
            pygame.draw.line(self.surface, (0, 200, 200), right_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw center lane points
        for px, py, _ in center_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (0, 0, 200), (px_scaled, py_scaled), 2)

        # Draw camera position
        pygame.draw.circle(self.surface, GREEN, camera_pos_scaled, 5)

        # Draw wheel positions
        pygame.draw.circle(self.surface, (255, 100, 0), left_wheel_scaled, 5)  # Orange
        pygame.draw.circle(self.surface, (0, 150, 255), right_wheel_scaled, 5)  # Cyan

    def _draw_car_2d(self, car):
        """Draw car in minimap"""
        # Draw main axis
        rear_x = car.x - (car.length/2) * np.cos(car.theta)
        rear_y = car.y - (car.length/2) * np.sin(car.theta)
        front_x = car.x + (car.length/2) * np.cos(car.theta)
        front_y = car.y + (car.length/2) * np.sin(car.theta)

        # Convert to minimap coordinates
        rear_scaled = self._world_to_minimap(rear_x, rear_y)
        front_scaled = self._world_to_minimap(front_x, front_y)
        center_scaled = self._world_to_minimap(car.x, car.y)

        pygame.draw.line(self.surface, WHITE, rear_scaled, front_scaled, 2)

        # Draw direction indicator
        pygame.draw.circle(self.surface, BLUE, front_scaled, 4)
        pygame.draw.circle(self.surface, YELLOW, center_scaled, 3)


