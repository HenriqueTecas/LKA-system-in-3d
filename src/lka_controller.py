"""
LKA Controller Module - Pure Pursuit Algorithm
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from .config import LKA_LOOKAHEAD_SMOOTHING_ALPHA, LKA_LOOKAHEAD_SNAP_THRESHOLD


class PurePursuitLKA:
    """Pure Pursuit Lane Keeping Assist using SI units (meters)"""
    def __init__(self, car, camera):
        self.car = car
        self.camera = camera
        self.active = False
        self.was_manually_overridden = False

        self.steering_gain = 1.0

        # Store all lane center points for visualization
        self.lane_center_points = []  # All computed lane center points (meters)
        self.lookahead_point = None  # Selected lookahead point (meters)

        # Smoothing for lookahead point to reduce jitter from noisy detections
        self.smoothing_alpha = float(LKA_LOOKAHEAD_SMOOTHING_ALPHA)
        self._smoothed_lookahead = None  # (x, y, dist) in meters

    def toggle(self):
        """Toggle LKA on/off"""
        self.active = not self.active
        self.was_manually_overridden = False
        return self.active

    def deactivate(self):
        """Deactivate LKA"""
        if self.active:
            self.active = False
            self.was_manually_overridden = True

    def calculate_steering(self, track):
        """Pure Pursuit algorithm with enhanced lane center point generation"""
        if not self.active:
            self.lane_center_points = []
            return None

        # Use camera's last measurement (already detected in main loop)
        left_lane, right_lane, center_lane = self.camera.last_measurement

        # Determine which lane we're in and which boundaries to use
        current_lane = self.camera.current_lane

        if current_lane == "LEFT":
            # In left lane: use left outer boundary and center line
            lane_left_boundary = left_lane
            lane_right_boundary = center_lane
        elif current_lane == "RIGHT":
            # In right lane: use center line and right outer boundary
            lane_left_boundary = center_lane
            lane_right_boundary = right_lane
        else:
            # Unknown lane - fallback (should rarely happen)
            lane_left_boundary = left_lane
            lane_right_boundary = right_lane

        # Check if we have both boundaries for the current lane
        if len(lane_left_boundary) == 0 or len(lane_right_boundary) == 0:
            self.lane_center_points = []
            return None

        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        # Generate lane center points using camera's distance-weighted detections
        lane_center_points = self._generate_dense_lane_centers(lane_left_boundary, lane_right_boundary, car_x, car_y)

        # Store for visualization
        self.lane_center_points = lane_center_points

        if len(lane_center_points) == 0:
            return None

        # Adaptive lookahead based on lateral error
        # Calculate how far we are from the closest lane center point (lateral error)
        closest_point = min(lane_center_points, key=lambda p: p[2])
        closest_dist = closest_point[2]
        
        # Calculate lateral error (perpendicular distance from car to closest point)
        closest_x, closest_y = closest_point[0], closest_point[1]
        dx = closest_x - car_x
        dy = closest_y - car_y
        
        # Lateral error (perpendicular to car heading)
        lateral_error = abs(-dx * np.sin(car_theta) + dy * np.cos(car_theta))
        
        # Adaptive lookahead distance based on lateral error
        # Large error → medium lookahead (5-7m) for correction
        # Small error → long lookahead (8-12m) for stability
        min_lookahead = 5.0
        max_lookahead = 14.0
        
        # Normalize lateral error (assume lane width is ~5m, so 1m error = 20%)
        error_normalized = np.clip(lateral_error / 2.0, 0.0, 1.0)
        
        # Inverse relationship: high error → low lookahead
        target_lookahead = max_lookahead - (max_lookahead - min_lookahead) * error_normalized
        
        # Find point closest to target lookahead
        best_point = min(lane_center_points, key=lambda p: abs(p[2] - target_lookahead))

        lookahead_x, lookahead_y, actual_distance = best_point

        # Smooth the selected lookahead point (EMA) to reduce twitching
        if self._smoothed_lookahead is None:
            self._smoothed_lookahead = (lookahead_x, lookahead_y, actual_distance)
        else:
            prev_x, prev_y, prev_dist = self._smoothed_lookahead
            # Snap immediately if change is large (prevents lag on big maneuvers)
            dx_snap = lookahead_x - prev_x
            dy_snap = lookahead_y - prev_y
            snap_dist = np.hypot(dx_snap, dy_snap)
            if snap_dist >= float(LKA_LOOKAHEAD_SNAP_THRESHOLD):
                sm_x, sm_y, sm_d = lookahead_x, lookahead_y, actual_distance
            else:
                alpha = self.smoothing_alpha
                sm_x = alpha * lookahead_x + (1 - alpha) * prev_x
                sm_y = alpha * lookahead_y + (1 - alpha) * prev_y
                sm_d = alpha * actual_distance + (1 - alpha) * prev_dist

            self._smoothed_lookahead = (sm_x, sm_y, sm_d)

        lookahead_x, lookahead_y, actual_distance = self._smoothed_lookahead

        dx = lookahead_x - car_x
        dy = lookahead_y - car_y
        angle_to_point = np.arctan2(dy, dx)

        alpha = angle_to_point - car_theta
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

        wheelbase = self.car.wheelbase

        if actual_distance < 1.0:
            return 0.0

        steering_angle = np.arctan2(2 * wheelbase * np.sin(alpha), actual_distance)
        steering_angle *= self.steering_gain
        steering_angle = np.clip(steering_angle,
                                -self.car.max_steering_angle,
                                self.car.max_steering_angle)

        self.lookahead_point = (lookahead_x, lookahead_y)
        self.lookahead_distance = actual_distance

        return steering_angle

    def _generate_dense_lane_centers(self, left_lane, right_lane, car_x, car_y):
        """
        Generate lane center points by pairing left and right boundaries.
        Camera already provides distance and angle, so we just compute centers.
        
        Lane points format: (x, y, angle, confidence) or (x, y, angle)
        Returns: [(center_x, center_y, distance_from_car), ...]
        """
        lane_center_points = []
        
        # Simple aligned pairing (both boundaries should have similar point counts)
        n = min(len(left_lane), len(right_lane))
        
        for i in range(n):
            left_x, left_y = left_lane[i][0], left_lane[i][1]
            right_x, right_y = right_lane[i][0], right_lane[i][1]
            
            # Lane center is midpoint
            center_x = (left_x + right_x) / 2.0
            center_y = (left_y + right_y) / 2.0
            
            # Use camera's distance calculation (already in meters)
            dx = center_x - car_x
            dy = center_y - car_y
            distance = np.sqrt(dx**2 + dy**2)
            
            lane_center_points.append((center_x, center_y, distance))
        
        return lane_center_points



