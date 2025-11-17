"""
Camera Sensor Module - Lane Detection
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


class CameraSensor:
    """Camera sensor for lane detection - identical logic to original"""
    def __init__(self, car):
        self.car = car
        self.field_of_view = np.radians(100)
        self.max_range = 300
        self.min_range = 20
        self.image_width = 1280
        self.image_height = 720
        self.mount_offset = self.car.length * 0.10
        self.detection_confidence = 0.95
        self.lane_sample_points = 10

        # NEW: Evenly-spaced sampling configuration
        self.sample_interval = 5  # pixels between detection points
        self.use_uniform_sampling = True  # Use evenly-spaced points instead of all visible points

        self.left_lane_detected = False
        self.right_lane_detected = False
        self.left_lane_position = None
        self.right_lane_position = None
        self.lane_center_offset = 0.0
        self.lane_heading_error = 0.0
        self.current_lane = "UNKNOWN"

    def get_camera_position(self):
        """Get camera world position"""
        camera_x = self.car.x + self.mount_offset * np.cos(self.car.theta)
        camera_y = self.car.y + self.mount_offset * np.sin(self.car.theta)
        return camera_x, camera_y

    def detect_lanes(self, track):
        """Detect lane lines - same logic as original"""
        camera_x, camera_y = self.get_camera_position()
        camera_angle = self.car.theta

        # Get track boundaries
        left_outer_boundary = track._offset_line(track.centerline, -track.lane_width)
        center_boundary = track.centerline
        right_outer_boundary = track._offset_line(track.centerline, track.lane_width)

        # Detect boundaries
        left_outer_points = self._detect_lane_boundary(
            left_outer_boundary, camera_x, camera_y, camera_angle
        )
        center_points = self._detect_lane_boundary(
            center_boundary, camera_x, camera_y, camera_angle
        )
        right_outer_points = self._detect_lane_boundary(
            right_outer_boundary, camera_x, camera_y, camera_angle
        )

        # Determine current lane
        car_lateral_offset = self._get_lateral_offset_from_track_center(track)

        if car_lateral_offset < 0:
            left_lane_points = left_outer_points
            right_lane_points = center_points
            current_lane = "LEFT"
        else:
            left_lane_points = center_points
            right_lane_points = right_outer_points
            current_lane = "RIGHT"

        self.left_lane_detected = len(left_lane_points) > 0
        self.right_lane_detected = len(right_lane_points) > 0
        self.current_lane = current_lane

        if self.left_lane_detected and len(left_lane_points) > 0:
            self.left_lane_position = self._calculate_lane_position(left_lane_points[0])

        if self.right_lane_detected and len(right_lane_points) > 0:
            self.right_lane_position = self._calculate_lane_position(right_lane_points[0])

        self._calculate_lane_tracking_errors(left_lane_points, right_lane_points)

        return left_lane_points, right_lane_points, center_points

    def _detect_lane_boundary(self, boundary_points, camera_x, camera_y, camera_angle):
        """Detect visible lane boundary points with uniform sampling"""
        if self.use_uniform_sampling:
            return self._detect_lane_boundary_uniform(boundary_points, camera_x, camera_y, camera_angle)
        else:
            # Original method: detect all visible points
            return self._detect_lane_boundary_all(boundary_points, camera_x, camera_y, camera_angle)

    def _detect_lane_boundary_all(self, boundary_points, camera_x, camera_y, camera_angle):
        """Original method: Detect ALL visible lane boundary points"""
        visible_points = []

        for point in boundary_points:
            px, py = point
            dx = px - camera_x
            dy = py - camera_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < self.min_range or distance > self.max_range:
                continue

            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - camera_angle
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            if abs(angle_diff) < self.field_of_view / 2:
                visible_points.append((px, py, angle_diff))

        return visible_points

    def _detect_lane_boundary_uniform(self, boundary_points, camera_x, camera_y, camera_angle):
        """NEW: Sample lane boundary points at uniform intervals along the visible boundary"""
        visible_points = []

        # First pass: collect all visible points with their cumulative distance
        points_with_distance = []
        cumulative_distance = 0.0

        for i, point in enumerate(boundary_points):
            px, py = point
            dx = px - camera_x
            dy = py - camera_y
            distance_from_camera = np.sqrt(dx**2 + dy**2)

            # Check if point is in camera range
            if distance_from_camera < self.min_range or distance_from_camera > self.max_range:
                continue

            # Check if point is in field of view
            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - camera_angle
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            if abs(angle_diff) > self.field_of_view / 2:
                continue

            # Calculate cumulative distance along the boundary
            if i > 0 and len(points_with_distance) > 0:
                prev_px, prev_py = points_with_distance[-1][0], points_with_distance[-1][1]
                segment_length = np.sqrt((px - prev_px)**2 + (py - prev_py)**2)
                cumulative_distance += segment_length

            points_with_distance.append((px, py, angle_diff, cumulative_distance, distance_from_camera))

        if len(points_with_distance) == 0:
            return []

        # Second pass: sample at uniform intervals
        total_length = points_with_distance[-1][3]  # Last cumulative distance
        num_samples = max(1, int(total_length / self.sample_interval))

        for i in range(num_samples + 1):
            target_distance = i * self.sample_interval

            # Find the closest point to this target distance
            best_point = min(points_with_distance,
                           key=lambda p: abs(p[3] - target_distance))

            px, py, angle_diff, cum_dist, dist_from_cam = best_point

            # Avoid duplicates
            if not any(abs(vp[0] - px) < 5 and abs(vp[1] - py) < 5 for vp in visible_points):
                visible_points.append((px, py, angle_diff))

        return visible_points

    def _calculate_lane_position(self, point_data):
        """Calculate lane position (angle only)"""
        px, py, angle = point_data
        return angle

    def _get_lateral_offset_from_track_center(self, track):
        """Calculate lateral offset from track centerline"""
        min_dist = float('inf')
        closest_idx = 0

        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((self.car.x - cx)**2 + (self.car.y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]

        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)

        to_car_x = self.car.x - p_curr[0]
        to_car_y = self.car.y - p_curr[1]

        perp_angle = track_angle + np.pi / 2
        lateral_offset = (to_car_x * np.cos(perp_angle) +
                         to_car_y * np.sin(perp_angle))

        return lateral_offset

    def _calculate_lane_tracking_errors(self, left_points, right_points):
        """Calculate lateral offset and heading error"""
        if not left_points or not right_points:
            return

        left_closest = min(left_points, key=lambda p: abs(p[2]))
        right_closest = min(right_points, key=lambda p: abs(p[2]))

        left_angle = left_closest[2]
        right_angle = right_closest[2]

        self.lane_center_offset = (right_angle + left_angle) / 2
        self.lane_heading_error = self.lane_center_offset


