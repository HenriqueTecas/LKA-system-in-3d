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

        # Predictive path extension parameters
        self.enable_prediction = True  # Enable curvature-based path prediction
        self.prediction_horizon = 150.0  # How far to predict ahead (pixels)
        self.prediction_step = 10.0  # Spacing between predicted points
        self.curvature_sample_points = 8  # Number of points to use for curvature estimation

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
            # Unknown lane - fall back to original behavior
            lane_left_boundary = left_lane
            lane_right_boundary = right_lane

        # Check if we have both boundaries for the current lane
        if len(lane_left_boundary) == 0 or len(lane_right_boundary) == 0:
            self.lane_center_points = []
            return None

        speed = abs(self.car.velocity)
        lookahead_distance = self.base_lookahead_distance + self.lookahead_gain * speed
        lookahead_distance = np.clip(lookahead_distance, self.min_lookahead, self.max_lookahead)

        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        # IMPROVED: Generate lane center points for CURRENT LANE ONLY
        lane_center_points = self._generate_dense_lane_centers(lane_left_boundary, lane_right_boundary, car_x, car_y)

        # NEW: Apply predictive path extension if enabled
        if self.enable_prediction and len(lane_center_points) >= 3:
            lane_center_points = self._extend_path_with_prediction(lane_center_points, car_x, car_y, car_theta)

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

    def _world_to_ego(self, world_points, car_x, car_y, car_theta):
        """Transform world coordinates to ego (car-centric) frame"""
        ego_points = []
        cos_th = np.cos(-car_theta)
        sin_th = np.sin(-car_theta)

        for wx, wy, dist in world_points:
            dx = wx - car_x
            dy = wy - car_y

            x_ego = dx * cos_th - dy * sin_th
            y_ego = dx * sin_th + dy * cos_th

            ego_points.append((x_ego, y_ego))

        return ego_points

    def _ego_to_world(self, ego_points, car_x, car_y, car_theta):
        """Transform ego coordinates back to world frame"""
        world_points = []
        cos_th = np.cos(car_theta)
        sin_th = np.sin(car_theta)

        for x_ego, y_ego in ego_points:
            # Rotate to world frame
            dx = x_ego * cos_th - y_ego * sin_th
            dy = x_ego * sin_th + y_ego * cos_th

            # Translate to world position
            wx = car_x + dx
            wy = car_y + dy

            world_points.append((wx, wy))

        return world_points

    def _fit_lane_polynomial(self, ego_points):
        """
        Fit polynomial to lane: y(x) = ax² + bx + c
        Returns coefficients (a, b, c) or None if fit fails
        """
        if len(ego_points) < 3:
            return None

        # Filter points ahead of car (x_ego > 0)
        forward_points = [(x, y) for x, y in ego_points if x > 0]

        if len(forward_points) < 3:
            return None

        # Take closest N points for fitting
        forward_points = sorted(forward_points, key=lambda p: p[0])[:self.curvature_sample_points]

        # Extract x and y arrays
        x_arr = np.array([p[0] for p in forward_points])
        y_arr = np.array([p[1] for p in forward_points])

        try:
            # Fit quadratic polynomial: y = ax² + bx + c
            coeffs = np.polyfit(x_arr, y_arr, 2)
            return coeffs  # [a, b, c]
        except:
            return None

    def _extend_path_with_prediction(self, lane_center_points, car_x, car_y, car_theta):
        """
        Predictive path extension using polynomial continuation.
        This is realistic: uses visible lane geometry to predict where it goes.
        
        FIXED: Now properly continues the polynomial instead of mixing with arc formulas.
        """
        if len(lane_center_points) < 3:
            return lane_center_points

        # Step 1: Transform to ego frame
        ego_points = self._world_to_ego(lane_center_points, car_x, car_y, car_theta)

        # Step 2: Fit polynomial to visible points
        coeffs = self._fit_lane_polynomial(ego_points)
        
        if coeffs is None:
            return lane_center_points  # Can't fit, return original points
        
        a, b, c = coeffs  # y(x) = ax² + bx + c

        # Step 3: Find furthest visible point in ego frame
        forward_ego = [(x, y) for x, y in ego_points if x > 0]
        if len(forward_ego) == 0:
            return lane_center_points

        furthest_x = max([x for x, y in forward_ego])

        # Step 4: Generate predicted points by evaluating the polynomial forward
        predicted_ego = []
        
        # Start from just beyond the furthest visible point
        x_start = furthest_x + self.prediction_step
        x_end = furthest_x + self.prediction_horizon
        
        x_values = np.arange(x_start, x_end, self.prediction_step)
        
        for x_pred in x_values:
            # Evaluate polynomial at this x
            y_pred = a * x_pred**2 + b * x_pred + c
            predicted_ego.append((x_pred, y_pred))

        # Step 5: Transform predicted points back to world frame
        predicted_world = self._ego_to_world(predicted_ego, car_x, car_y, car_theta)

        # Step 6: Combine original + predicted points
        extended_points = list(lane_center_points)  # Keep originals

        for wx, wy in predicted_world:
            dx = wx - car_x
            dy = wy - car_y
            dist = np.sqrt(dx**2 + dy**2)
            extended_points.append((wx, wy, dist))

        return extended_points



