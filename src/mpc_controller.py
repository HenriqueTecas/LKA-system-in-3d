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

    Uses forward simulation with physics-based Ackermann model to predict
    future trajectories for different steering commands, then selects the
    command that minimizes a cost function.

    Cost function penalizes:
    - Lateral deviation from lane center
    - Leaving the road (off-track)
    - Crossing into opposite lane
    
    All units are SI (meters, m/s, radians).
    """

    def __init__(self, car, camera):
        self.car = car
        self.camera = camera
        self.active = False
        self.was_manually_overridden = False

        # MPC parameters - BALANCED for curves and performance
        self.prediction_horizon = 15  # Good lookahead with reasonable performance
        self.dt = 0.18  # Balance between accuracy and speed
        self.num_candidates = 7  # Keep reduced for performance

        # Cost function weights - TUNED for curve handling
        self.weight_lateral_deviation = 50.0  # Increased to prioritize lane center tracking
        self.weight_off_track = 10000.0  # Keep strong - safety critical
        self.weight_wrong_lane = 5000.0  # Keep strong - safety critical
        self.weight_steering_effort = 5.0  # Reduced to allow sharper turns in curves
        self.weight_steering_change = 30.0  # Reduced for more responsive steering
        self.weight_boundary_proximity = 100.0  # Reduced - was causing late reactions

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

        # Use camera's last measurement (already detected in main loop)
        left_lane, center_lane, right_lane = self.camera.last_measurement

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
)            # Track best command
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
        """Compute lane center points from left and right boundaries (optimized)"""
        lane_centers = []

        # Simple index-based pairing (assumes points are roughly aligned)
        # Much faster than closest-point search
        n = min(len(left_boundary), len(right_boundary))
        
        for i in range(n):
            left_x, left_y = left_boundary[i][0], left_boundary[i][1]
            right_x, right_y = right_boundary[i][0], right_boundary[i][1]
            
            center_x = (left_x + right_x) / 2.0
            center_y = (left_y + right_y) / 2.0
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

            # Compute cost at this position (first 3 steps always, then every other)
            # This ensures accurate near-term prediction while saving computation far ahead
            if step < 3 or step % 2 == 0:
                step_cost = self._compute_position_cost(
                    x, y, lane_centers, track, current_lane
                )
                total_cost += step_cost

            # Early termination if cost is already too high
            if total_cost > 6e5:  # Balanced threshold
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

        # 1. Lateral deviation from lane center (vectorized for speed)
        if len(lane_centers) > 0:
            # Find closest lane center point using numpy
            centers_array = np.array(lane_centers)
            distances_sq = (centers_array[:, 0] - x)**2 + (centers_array[:, 1] - y)**2
            closest_idx = np.argmin(distances_sq)
            
            # Lateral deviation (linear cost for smoother behavior near center)
            lateral_dev = np.sqrt(distances_sq[closest_idx])
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

        perp_angle = track_angle + np.pi / 2
        lateral_offset_px = (to_point_x * np.cos(perp_angle) +
                            to_point_y * np.sin(perp_angle))
        
        # Convert to meters
        lateral_offset = lateral_offset_px / self.car.pixels_per_meter

        # Check if within track bounds (track width in pixels, convert to meters)
        # Track width is total width (2 * lane_width), so edges are at ±track_width/2
        half_width = (track.track_width / 2.0) / self.car.pixels_per_meter
        on_track = abs(lateral_offset) <= half_width

        return on_track, lateral_offset

    def _is_in_wrong_lane(self, lateral_offset, current_lane, lane_width):
        """
        Check if lateral offset places vehicle in wrong lane
        
        Args:
            lateral_offset: perpendicular distance from centerline in meters
            current_lane: "LEFT" or "RIGHT"
            lane_width: width of one lane in meters
        
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
