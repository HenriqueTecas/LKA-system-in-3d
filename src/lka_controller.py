"""
LKA Controller Module - Pure Pursuit Algorithm
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


class PurePursuitLKA:
    """Pure Pursuit Lane Keeping Assist - identical logic to original"""
    def __init__(self, car, camera):
        self.car = car
        self.camera = camera
        self.active = False
        self.was_manually_overridden = False

        self.base_lookahead_distance = 80.0
        self.lookahead_gain = 0.5
        self.min_lookahead = 40.0
        self.max_lookahead = 150.0
        self.steering_gain = 1.2

        # NEW: Store all lane center points for visualization
        self.lane_center_points = []  # All computed lane center points
        self.lookahead_point = None  # Selected lookahead point

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

        left_lane, right_lane, center_lane = self.camera.detect_lanes(track)

        if not (self.camera.left_lane_detected and self.camera.right_lane_detected):
            self.lane_center_points = []
            return None

        speed = abs(self.car.velocity)
        lookahead_distance = self.base_lookahead_distance + self.lookahead_gain * speed
        lookahead_distance = np.clip(lookahead_distance, self.min_lookahead, self.max_lookahead)

        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        # IMPROVED: Generate MORE lane center points with better interpolation
        lane_center_points = self._generate_dense_lane_centers(left_lane, right_lane, car_x, car_y)

        # Store for visualization
        self.lane_center_points = lane_center_points

        if len(lane_center_points) == 0:
            return None

        best_point = min(lane_center_points,
                        key=lambda p: abs(p[2] - lookahead_distance))

        lookahead_x, lookahead_y, actual_distance = best_point

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
        """Generate dense lane center points by interpolating between boundaries"""
        lane_center_points = []

        # Method 1: Pair closest left-right points (original)
        for left_point in left_lane:
            left_x, left_y, left_ang = left_point
            min_dist = float('inf')
            closest_right = None

            for right_point in right_lane:
                right_x, right_y, right_ang = right_point
                dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_right = right_point

            if closest_right:
                right_x, right_y, right_ang = closest_right
                center_x = (left_x + right_x) / 2
                center_y = (left_y + right_y) / 2
                dx = center_x - car_x
                dy = center_y - car_y
                distance = np.sqrt(dx**2 + dy**2)
                lane_center_points.append((center_x, center_y, distance))

        # Method 2: Also pair from right side (creates more points)
        for right_point in right_lane:
            right_x, right_y, right_ang = right_point
            min_dist = float('inf')
            closest_left = None

            for left_point in left_lane:
                left_x, left_y, left_ang = left_point
                dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_left = left_point

            if closest_left:
                left_x, left_y, left_ang = closest_left
                center_x = (left_x + right_x) / 2
                center_y = (left_y + right_y) / 2
                dx = center_x - car_x
                dy = center_y - car_y
                distance = np.sqrt(dx**2 + dy**2)

                # Avoid duplicates (within 5 pixels)
                is_duplicate = False
                for existing_cx, existing_cy, _ in lane_center_points:
                    if abs(center_x - existing_cx) < 5 and abs(center_y - existing_cy) < 5:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    lane_center_points.append((center_x, center_y, distance))

        # Method 3: Interpolate additional points along the path
        if len(lane_center_points) >= 2:
            # Sort by distance from car
            sorted_centers = sorted(lane_center_points, key=lambda p: p[2])
            interpolated = []

            for i in range(len(sorted_centers) - 1):
                cx1, cy1, d1 = sorted_centers[i]
                cx2, cy2, d2 = sorted_centers[i + 1]

                # Add midpoint if points are far apart
                dist_between = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                if dist_between > 40:  # If more than 40 pixels apart
                    mid_x = (cx1 + cx2) / 2
                    mid_y = (cy1 + cy2) / 2
                    mid_dist = np.sqrt((mid_x - car_x)**2 + (mid_y - car_y)**2)
                    interpolated.append((mid_x, mid_y, mid_dist))

            lane_center_points.extend(interpolated)

        return lane_center_points


