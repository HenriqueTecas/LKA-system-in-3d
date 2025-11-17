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
        # Multiply by 8 for zoomed-in view centered on car
        self.scale = min(scale_x, scale_y) * 8.0
        self.margin = margin
        
        # Store track dimensions for dynamic centering
        self.track_center_x = (self.min_x + self.max_x) / 2
        self.track_center_y = (self.min_y + self.max_y) / 2

    def _world_to_minimap(self, x, y, car):
        """Convert world coordinates to minimap coordinates, centered on car"""
        # Center the view on the car position
        car_x = car.get_x_pixels()
        car_y = car.get_y_pixels()
        
        # Translate to car-centered coordinates, scale, then center in minimap
        map_x = (x - car_x) * self.scale + self.size / 2
        map_y = (y - car_y) * self.scale + self.size / 2
        return int(map_x), int(map_y)

    def render(self, car, camera, lka, mpc=None):
        """Render minimap with original 2D view"""
        # Fill with semi-transparent dark background
        self.surface.fill((20, 20, 20))  # Very dark gray background

        # Draw border around minimap
        pygame.draw.rect(self.surface, (100, 100, 100), (0, 0, self.size, self.size), 3)
        pygame.draw.rect(self.surface, (200, 200, 200), (2, 2, self.size-4, self.size-4), 1)

        # Draw track
        self._draw_track_2d(car)

        # Draw camera FOV and detections
        self._draw_camera_view_2d(camera, car)

        # Draw ALL LKA lane center points (small yellow dots)
        if lka.active and hasattr(lka, 'lane_center_points') and lka.lane_center_points:
            for cx, cy, dist in lka.lane_center_points:
                # Convert meters to pixels for minimap
                px = cx * car.pixels_per_meter
                py = cy * car.pixels_per_meter
                center_scaled = self._world_to_minimap(px, py, car)
                pygame.draw.circle(self.surface, (255, 255, 100), center_scaled, 3)

        # Draw LKA selected lookahead point (larger, brighter)
        if lka.active and hasattr(lka, 'lookahead_point') and lka.lookahead_point is not None:
            lx, ly = lka.lookahead_point
            car_scaled = self._world_to_minimap(car.get_x_pixels(), car.get_y_pixels(), car)
            # Convert lookahead from meters to pixels
            lx_px = lx * car.pixels_per_meter
            ly_px = ly * car.pixels_per_meter
            lookahead_scaled = self._world_to_minimap(lx_px, ly_px, car)
            pygame.draw.line(self.surface, YELLOW, car_scaled, lookahead_scaled, 2)
            pygame.draw.circle(self.surface, YELLOW, lookahead_scaled, 7)  # Larger for selected point

        # Draw MPC predicted trajectory (silver/gray dots)
        if mpc and mpc.active and hasattr(mpc, 'predicted_trajectory') and mpc.predicted_trajectory:
            for mx, my in mpc.predicted_trajectory:
                # Convert meters to pixels for minimap
                px = mx * car.pixels_per_meter
                py = my * car.pixels_per_meter
                traj_scaled = self._world_to_minimap(px, py, car)
                pygame.draw.circle(self.surface, (192, 192, 192), traj_scaled, 4)  # Silver

        # Draw car (simple representation)
        self._draw_car_2d(car)

        return self.surface

    def _draw_track_2d(self, car):
        """Draw track in minimap with proper scaling"""
        outer = self.track._offset_line(self.track.centerline, self.track.track_width / 2)
        inner = self.track._offset_line(self.track.centerline, -self.track.track_width / 2)

        # Convert to minimap coordinates
        outer_scaled = [self._world_to_minimap(x, y, car) for x, y in outer]
        inner_scaled = [self._world_to_minimap(x, y, car) for x, y in inner]

        if len(outer_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, outer_scaled, 2)
        if len(inner_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, inner_scaled, 2)

        # Draw centerline dashed
        centerline_scaled = [self._world_to_minimap(x, y, car) for x, y in self.track.centerline]
        for i in range(0, len(centerline_scaled) - 1, 2):
            p1 = centerline_scaled[i]
            p2 = centerline_scaled[i + 1]
            pygame.draw.line(self.surface, GRAY, p1, p2, 1)

    def _draw_camera_view_2d(self, camera, car):
        """Draw camera FOV and detected lanes"""
        # Get camera position in meters, convert to pixels
        # Realistic camera returns (x, y, z), simple camera returns (x, y)
        cam_pos = camera.get_camera_position()
        if len(cam_pos) == 3:
            camera_x_m, camera_y_m, _ = cam_pos  # Unpack 3D position, ignore z
        else:
            camera_x_m, camera_y_m = cam_pos  # Unpack 2D position
        camera_x = camera_x_m * camera.car.pixels_per_meter
        camera_y = camera_y_m * camera.car.pixels_per_meter

        # Draw FOV cone (convert max_range from meters to pixels)
        max_range_px = camera.max_range * camera.car.pixels_per_meter
        fov_points = [(camera_x, camera_y)]
        left_angle = camera.car.theta - camera.field_of_view / 2
        fov_points.append((
            camera_x + max_range_px * np.cos(left_angle),
            camera_y + max_range_px * np.sin(left_angle)
        ))
        right_angle = camera.car.theta + camera.field_of_view / 2
        fov_points.append((
            camera_x + max_range_px * np.cos(right_angle),
            camera_y + max_range_px * np.sin(right_angle)
        ))

        # Convert FOV points to minimap coordinates
        fov_points_scaled = [self._world_to_minimap(x, y, car) for x, y in fov_points]

        # Draw semi-transparent FOV
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.polygon(s, (0, 255, 0, 30), fov_points_scaled)
        self.surface.blit(s, (0, 0))

        # Draw FOV edges
        camera_pos_scaled = fov_points_scaled[0]
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[1], 2)
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[2], 2)

        # Draw detected lane points - ONLY FOR CURRENT LANE
        left_lane, right_lane, center_lane = camera.detect_lanes(camera.car.track)

        # Determine which lane we're in and which boundaries to display
        current_lane = camera.current_lane

        if current_lane == "LEFT":
            # In left lane: show left outer boundary and center line
            lane_left_boundary = left_lane
            lane_right_boundary = center_lane
        elif current_lane == "RIGHT":
            # In right lane: show center line and right outer boundary
            lane_left_boundary = center_lane
            lane_right_boundary = right_lane
        else:
            # Unknown - show all (fallback)
            lane_left_boundary = left_lane
            lane_right_boundary = right_lane

        # Get wheel positions
        (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y) = camera.car.get_front_wheel_positions()
        left_wheel_scaled = self._world_to_minimap(left_wheel_x, left_wheel_y, car)
        right_wheel_scaled = self._world_to_minimap(right_wheel_x, right_wheel_y, car)

        # Draw left boundary of current lane with vectors from LEFT wheel
        for point in lane_left_boundary:
            # Handle both 3-tuple and 4-tuple formats
            mx, my = point[0], point[1]
            # Convert from meters to pixels
            px = mx * camera.car.pixels_per_meter
            py = my * camera.car.pixels_per_meter
            px_scaled, py_scaled = self._world_to_minimap(px, py, car)
            pygame.draw.circle(self.surface, (255, 0, 0), (px_scaled, py_scaled), 4)
            # Vector from left wheel to left lane point
            pygame.draw.line(self.surface, (255, 128, 0), left_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw right boundary of current lane with vectors from RIGHT wheel
        for point in lane_right_boundary:
            # Handle both 3-tuple and 4-tuple formats
            mx, my = point[0], point[1]
            # Convert from meters to pixels
            px = mx * camera.car.pixels_per_meter
            py = my * camera.car.pixels_per_meter
            px_scaled, py_scaled = self._world_to_minimap(px, py, car)
            pygame.draw.circle(self.surface, (0, 128, 255), (px_scaled, py_scaled), 4)
            # Vector from right wheel to right lane point
            pygame.draw.line(self.surface, (0, 200, 200), right_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw camera position
        pygame.draw.circle(self.surface, GREEN, camera_pos_scaled, 6)

        # Draw wheel positions
        pygame.draw.circle(self.surface, (255, 100, 0), left_wheel_scaled, 6)  # Orange
        pygame.draw.circle(self.surface, (0, 150, 255), right_wheel_scaled, 6)  # Cyan

    def _draw_car_2d(self, car):
        """Draw car representation on minimap"""
        # Car position scaled to minimap (convert meters to pixels first)
        car_x_px = car.get_x_pixels()
        car_y_px = car.get_y_pixels()
        
        # Calculate car corners for rectangle representation
        length_px = car.length / 2  # Half length in pixels (already in pixels)
        width_px = car.width / 2    # Half width in pixels
        
        # Get car corners in world space (pixels)
        cos_theta = np.cos(car.theta)
        sin_theta = np.sin(car.theta)
        
        # Calculate 4 corners
        corners_world = [
            (car_x_px + length_px * cos_theta - width_px * sin_theta,
             car_y_px + length_px * sin_theta + width_px * cos_theta),
            (car_x_px + length_px * cos_theta + width_px * sin_theta,
             car_y_px + length_px * sin_theta - width_px * cos_theta),
            (car_x_px - length_px * cos_theta + width_px * sin_theta,
             car_y_px - length_px * sin_theta - width_px * cos_theta),
            (car_x_px - length_px * cos_theta - width_px * sin_theta,
             car_y_px - length_px * sin_theta + width_px * cos_theta),
        ]
        
        # Convert to minimap coordinates
        corners_minimap = [self._world_to_minimap(x, y, car) for x, y in corners_world]
        
        # Draw filled rectangle for car body
        pygame.draw.polygon(self.surface, BLUE, corners_minimap)
        pygame.draw.polygon(self.surface, WHITE, corners_minimap, 2)  # White outline
        
        # Draw front indicator (small yellow circle at front)
        front_x = car_x_px + length_px * cos_theta
        front_y = car_y_px + length_px * sin_theta
        front_minimap = self._world_to_minimap(front_x, front_y, car)
        pygame.draw.circle(self.surface, YELLOW, front_minimap, 5)


