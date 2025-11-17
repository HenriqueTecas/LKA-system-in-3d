"""
MPC Controller Module - Model Predictive Control for Lane Keeping
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np


class MPCLaneKeeping:
    """
    Model Predictive Control Lane Keeping Assist

    Uses forward simulation with the Ackermann kinematic model to predict
    future trajectories for different steering commands, then selects the
    command that minimizes a cost function.

    Cost function penalizes:
    - Lateral deviation from lane center
    - Leaving the road (off-track)
    - Crossing into opposite lane
    """

    def __init__(self, car, camera):
        self.car = car
        self.camera = camera
        self.active = False
        self.was_manually_overridden = False

        # MPC parameters - TUNED for smoother behavior
        self.prediction_horizon = 20  # Look further ahead for smoother planning
        self.dt = 0.15  # Larger time steps = less reactive
        self.num_candidates = 9  # Fewer candidates = faster, less twitchy

        # Cost function weights - SOFTENED for less aggressive behavior
        self.weight_lateral_deviation = 20.0  # Reduced from 100 - less aggressive corrections
        self.weight_off_track = 10000.0  # Keep strong - safety critical
        self.weight_wrong_lane = 5000.0  # Keep strong - safety critical
        self.weight_steering_effort = 10.0  # Increased from 1 - prefer less steering
        self.weight_steering_change = 50.0  # NEW - penalize rapid steering changes
        self.weight_boundary_proximity = 200.0  # NEW - heavily penalize getting close to boundaries

        # Steering command candidates (fraction of max steering) - MORE CONSERVATIVE
        self.steering_fractions = np.linspace(-0.6, 0.6, self.num_candidates)  # Limited to ±60% instead of ±100%

        # Store predicted trajectory for visualization
        self.predicted_trajectory = []  # List of (x, y) positions
        self.selected_steering = 0.0
        self.best_cost = float('inf')
        self.previous_steering = 0.0  # Track previous steering for smoothness

    def toggle(self):
        """Toggle MPC LKA on/off"""
        self.active = not self.active
        self.was_manually_overridden = False
        return self.active

    def deactivate(self):
        """Deactivate MPC LKA"""
        if self.active:
            self.active = False
            self.was_manually_overridden = True

    def calculate_steering(self, track):
        """
        MPC control loop:
        1. Generate candidate steering commands
        2. Simulate forward for each candidate
        3. Compute cost for each trajectory
        4. Select command with minimum cost
        """
        if not self.active:
            self.predicted_trajectory = []
            return None

        # Get current lane detection
        left_lane, right_lane, center_lane = self.camera.detect_lanes(track)

        # Determine which lane we're in
        current_lane = self.camera.current_lane

        if current_lane == "LEFT":
            lane_left_boundary = left_lane
            lane_right_boundary = center_lane
        elif current_lane == "RIGHT":
            lane_left_boundary = center_lane
            lane_right_boundary = right_lane
        else:
            # Unknown lane - can't compute costs properly
            self.predicted_trajectory = []
            return None

        # Need at least some lane points for cost calculation
        if len(lane_left_boundary) == 0 or len(lane_right_boundary) == 0:
            self.predicted_trajectory = []
            return None

        # Compute lane center points for cost function
        lane_center_points = self._compute_lane_centers(
            lane_left_boundary, lane_right_boundary
        )

        if len(lane_center_points) == 0:
            self.predicted_trajectory = []
            return None

        # Current state
        current_x = self.car.x
        current_y = self.car.y
        current_theta = self.car.theta
        current_velocity = self.car.velocity

        # Evaluate all candidate steering commands
        best_steering = 0.0
        best_cost = float('inf')
        best_trajectory = []

        for steering_fraction in self.steering_fractions:
            # Convert fraction to actual steering angle
            steering_cmd = steering_fraction * self.car.max_steering_angle

            # Simulate forward with this steering command
            trajectory, cost = self._simulate_and_cost(
                current_x, current_y, current_theta, current_velocity,
                steering_cmd, lane_center_points, track, current_lane
            )

            # Track best command
            if cost < best_cost:
                best_cost = cost
                best_steering = steering_cmd
                best_trajectory = trajectory

        # Store for visualization
        self.predicted_trajectory = best_trajectory
        self.selected_steering = best_steering
        self.best_cost = best_cost

        # Update previous steering for next iteration (smoother transitions)
        self.previous_steering = best_steering

        return best_steering

    def _compute_lane_centers(self, left_boundary, right_boundary):
        """Compute lane center points from left and right boundaries"""
        lane_centers = []

        # Pair closest points
        for left_x, left_y, _ in left_boundary:
            min_dist = float('inf')
            closest_right = None

            for right_x, right_y, _ in right_boundary:
                dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_right = (right_x, right_y)

            if closest_right:
                center_x = (left_x + closest_right[0]) / 2
                center_y = (left_y + closest_right[1]) / 2
                lane_centers.append((center_x, center_y))

        return lane_centers

    def _simulate_and_cost(self, start_x, start_y, start_theta, start_velocity,
                          steering_cmd, lane_centers, track, current_lane):
        """
        Simulate vehicle forward with constant steering command
        and compute trajectory cost
        """
        trajectory = []
        total_cost = 0.0

        # Simulate state
        x = start_x
        y = start_y
        theta = start_theta
        velocity = start_velocity

        wheelbase = self.car.wheelbase

        for step in range(self.prediction_horizon):
            # Store position
            trajectory.append((x, y))

            # Ackermann kinematic model update
            if abs(velocity) > 0.1:
                omega = velocity * np.tan(steering_cmd) / wheelbase
                x += velocity * np.cos(theta) * self.dt
                y += velocity * np.sin(theta) * self.dt
                theta += omega * self.dt
                theta = np.arctan2(np.sin(theta), np.cos(theta))

            # Compute cost at this position
            step_cost = self._compute_position_cost(
                x, y, lane_centers, track, current_lane
            )

            total_cost += step_cost

            # Early termination if cost is already too high
            if total_cost > 1e6:
                break

        # Add steering effort cost (penalize large steering angles)
        total_cost += self.weight_steering_effort * abs(steering_cmd)

        # Add steering change cost (penalize rapid changes from previous steering)
        steering_change = abs(steering_cmd - self.previous_steering)
        total_cost += self.weight_steering_change * steering_change

        return trajectory, total_cost

    def _compute_position_cost(self, x, y, lane_centers, track, current_lane):
        """
        Compute cost for a given position:
        - Lateral deviation from lane center
        - Penalty for being off-track
        - Penalty for being in wrong lane
        """
        cost = 0.0

        # 1. Lateral deviation from lane center
        if len(lane_centers) > 0:
            # Find closest lane center point
            distances = [np.sqrt((x - cx)**2 + (y - cy)**2)
                        for cx, cy in lane_centers]
            min_dist = min(distances)
            closest_idx = distances.index(min_dist)
            closest_center = lane_centers[closest_idx]

            # Lateral deviation (linear cost for smoother behavior near center)
            lateral_dev = np.sqrt((x - closest_center[0])**2 +
                                 (y - closest_center[1])**2)
            cost += self.weight_lateral_deviation * lateral_dev

        # 2. Check if position is on track
        on_track, lateral_offset = self._is_on_track(x, y, track)

        if not on_track:
            # Strongly penalize being off the road
            cost += self.weight_off_track
        else:
            # 3. Boundary proximity cost - exponentially increase as we approach edges
            # Track has total width = 2 * lane_width, so half_width = track_width / 2
            half_width = track.track_width / 2.0
            distance_to_boundary = half_width - abs(lateral_offset)

            # Exponential cost that increases rapidly as we approach the boundary
            # When distance_to_boundary is small, cost is high
            if distance_to_boundary > 0:
                # Normalize: 0 = at boundary (edge), 1 = at centerline
                normalized_distance = distance_to_boundary / half_width
                # Exponential penalty: cost increases as normalized_distance decreases
                # e^(-3*x) gives strong penalty when x is small (near boundary)
                proximity_cost = np.exp(-3.0 * normalized_distance)
                cost += self.weight_boundary_proximity * proximity_cost

            # 4. Check if in wrong lane
            in_wrong_lane = self._is_in_wrong_lane(lateral_offset, current_lane, track.lane_width)
            if in_wrong_lane:
                cost += self.weight_wrong_lane

        return cost

    def _is_on_track(self, x, y, track):
        """
        Check if position (x, y) is on the track
        Returns: (on_track: bool, lateral_offset: float)
        """
        # Find closest centerline point
        min_dist = float('inf')
        closest_idx = 0

        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Compute lateral offset from centerline
        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]

        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)

        to_point_x = x - p_curr[0]
        to_point_y = y - p_curr[1]

        perp_angle = track_angle + np.pi / 2
        lateral_offset = (to_point_x * np.cos(perp_angle) +
                         to_point_y * np.sin(perp_angle))

        # Check if within track bounds
        # Track width is total width (2 * lane_width), so edges are at ±track_width/2
        half_width = track.track_width / 2.0
        on_track = abs(lateral_offset) <= half_width

        return on_track, lateral_offset

    def _is_in_wrong_lane(self, lateral_offset, current_lane, lane_width):
        """
        Check if lateral offset corresponds to being in the wrong lane
        
        Track geometry:
        - Track centerline is at lateral_offset = 0
        - Left lane center is at -lane_width/2
        - Right lane center is at +lane_width/2
        - Lane boundary (center line) is at lateral_offset = 0
        """
        half_lane = lane_width / 2.0
        
        if current_lane == "LEFT":
            # Left lane center is at -half_lane
            # Wrong if we cross centerline into right lane (positive offset > half_lane)
            return lateral_offset > half_lane
        elif current_lane == "RIGHT":
            # Right lane center is at +half_lane
            # Wrong if we cross centerline into left lane (negative offset < -half_lane)
            return lateral_offset < -half_lane
        else:
            return False
