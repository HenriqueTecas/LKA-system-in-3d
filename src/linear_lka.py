"""
Linear LKA Controller

Implements lane-keeping assist using linear state-feedback control from robotics course slides.
Uses SENSOR DATA ONLY for all state estimation (position, heading, velocity):
- GNSS: Position (x, y) with realistic noise
- IMU: Yaw rate and heading (theta) estimation  
- Wheel encoders: Velocity with realistic noise
- Camera: Lane detection only (not used for state estimation)

NEVER uses raw car.x/y/theta/velocity - those are ground truth for simulation only.
All control calculations use noisy sensor estimates to match real-world conditions.
"""

from __future__ import annotations

import numpy as np


class LinearLKAController:
    MODE_MANUAL = 0
    MODE_WARNING = 1
    MODE_ASSIST = 2

    # Provide defaults at class level so attributes exist even if initialization
    # is bypassed or a stale instance is deserialized without running __init__.
    _last_curve_radius: float | None = None
    safe_speed: float | None = None

    def __init__(self, car, camera, sensors=None, track=None):
        self.car = car
        self.camera = camera
        self.sensors = sensors
        self.track = track  # Track object for geometry-based speed calculation

        # Modes
        self.mode = self.MODE_MANUAL
        self.intervening = False

        # Geometry / lanes
        self.lane_width = 4.0
        self.min_points_for_direction = 4

        # Linear control gains from slides (Linear Control III)
        self.zeta = 0.8 # Damping ratio for pole placement
        self.omega_n = 1.2 # Natural frequency
        
        # Speed-adaptive lookahead (inspired by MPC horizon concept)
        # Lookahead = base_time * velocity, clamped to [min, max]
        self.lookahead_time_straight = 1.0  # seconds for straights
        self.lookahead_time_curve = 0.3  # seconds for curves (much tighter)
        self.lookahead_min = 2.0  # minimum lookahead distance (m)
        self.lookahead_max_straight = 25.0  # maximum lookahead on straights (m)
        self.lookahead_max_curve = 6.0  # maximum lookahead in curves (reduced)
        
        # Engagement thresholds
        self.assist_offset_deadband = 1.0
        self.assist_heading_deadband = 0.3
        self.warning_lateral_offset = 0.8

        # Speed limits for curve safety
        # Typical car comfort: 0.3-0.4g lateral
        # We use conservative 0.25g for safety margin
        self.lateral_accel_limit = 0.25 * 9.81  # m/s² - comfortable lateral acceleration
        self.safe_speed_scale = 0.90  # Apply 10% safety margin to calculated speed
        self.fallback_curve_radius = 200.0  # Default radius if detection fails (gentle curve)
        self.curve_threshold = 500.0  # Radius below which we consider it a curve (meters)
        
        # Adaptive curve exit logic (instead of hardcoded frames)
        self.curve_exit_time_min = 1.5  # Minimum time to confirm straight (seconds)
        self.curve_exit_time_max = 3.5  # Maximum time to confirm straight (seconds)
        self.curve_exit_radius_factor = 1.5  # Must be 1.5x threshold to start counting

        # (Adaptive lookahead only; no persistent straight-locking flag)

        # State
        self.last_omega = 0.0  # Last angular velocity command
        self.center_line_points: list[tuple[float, float]] = []
        self.overspeed_state = False  # Hysteresis state for speed warning
        self.in_curve_steering = False  # Immediate curve state for steering/lookahead
        self.in_curve_speed = False  # Hysteresis curve state for speed control
        self.curve_min_safe_speed = float('inf')  # Minimum safe speed for current curve
        self.curve_min_radius = float('inf')  # Tightest radius encountered in curve
        self.straight_start_time = None  # Timestamp when straight section started
        self.last_update_time = 0.0  # Track time for adaptive exit
        self.target_point: tuple[float, float] | None = None
        self.target_direction: float | None = None
        self.safe_speed: float | None = None
        self._last_curve_radius: float | None = None

        self.warnings = {
            "lane_departure": False,
            "speed_too_high": False,
            "assist_on": False,
            "assist_intervening": False,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_mode(self, mode):
        if mode in (self.MODE_MANUAL, self.MODE_WARNING, self.MODE_ASSIST):
            self.mode = mode
            self.intervening = False
            self.warnings = {k: False for k in self.warnings}
            return True
        return False

    def get_mode_name(self):
        return ["MANUAL", "WARNING", "ASSIST"][self.mode]

    @property
    def active(self):
        return self.mode != self.MODE_MANUAL

    def deactivate(self):
        self.set_mode(self.MODE_MANUAL)

    def calculate_control(self, track=None):  # track kept for interface compatibility
        if self.mode == self.MODE_MANUAL:
            self.intervening = False
            return None, None, None, {}, False

        self._state = self._get_vehicle_state()
        
        # Track time for adaptive curve exit
        import time
        current_time = time.time()
        if self.last_update_time == 0.0:
            self.last_update_time = current_time

        left_lane, center_lane, right_lane = self.camera.last_measurement
        current_lane = self.camera.current_lane

        active_lane = self._select_active_lane(current_lane, left_lane, center_lane, right_lane)
        if active_lane is None:
            self.intervening = False
            return None, None, None, {}, False

        if active_lane == "LEFT":
            lane_left = left_lane if len(left_lane) > 0 else center_lane
            lane_right = center_lane if len(center_lane) > 0 else right_lane
        else:
            lane_left = center_lane if len(center_lane) > 0 else left_lane
            lane_right = right_lane if len(right_lane) > 0 else center_lane

        # Update curve state FIRST (before steering calculation needs it)
        self._update_curve_state(lane_left, lane_right)

        speed = abs(self._state.get("velocity", 0.0))
        steering_cmd = self._calculate_steering_direction(lane_left, lane_right, speed)
        speed_cmd = self._calculate_speed_control(lane_left, lane_right)
        lane_offset = self._estimate_lane_offset()
        heading_err = self._heading_error()

        self._update_warnings(steering_cmd, speed_cmd, lane_offset)

        centered = abs(lane_offset) < self.assist_offset_deadband and abs(heading_err) < self.assist_heading_deadband
        lane_warning = self.warnings.get("lane_departure")

        throttle_cmd = None
        brake_cmd = None

        # Speed control policy:
        # - MODE_WARNING: only raise warnings, never override pedals.
        # - MODE_ASSIST: always override speed when overspeed, independent of lane intervention
        if self.mode == self.MODE_WARNING:
            self.intervening = False
            steering_cmd = None
        else:
            # Lane-based steering intervention
            self.intervening = lane_warning and not centered
            if not self.intervening:
                steering_cmd = None

        # Speed intervention (separate from steering intervention)
        if self.mode == self.MODE_ASSIST and speed_cmd and speed_cmd.get("over_speed"):
            throttle_cmd = 0.0
            brake_cmd = speed_cmd.get("brake", 0.0)

        self.warnings["assist_on"] = self.mode == self.MODE_ASSIST
        self.warnings["assist_intervening"] = self.intervening

        return steering_cmd, throttle_cmd, brake_cmd, self.warnings, self.intervening

    # ------------------------------------------------------------------
    # Steering - Linear Feedback Control from Slides
    # ------------------------------------------------------------------
    def _calculate_steering_direction(self, lane_left, lane_right, current_speed):
        """Linear state-feedback control from slides (LTA control II)."""
        car_x, car_y, car_theta = self._state["x"], self._state["y"], self._state["theta"]

        has_left = len(lane_left) >= self.min_points_for_direction
        has_right = len(lane_right) >= self.min_points_for_direction
        if not has_left and not has_right:
            return 0.0

        # Build centerline
        if has_left and has_right:
            n = min(len(lane_left), len(lane_right))
            center_line = [
                ((lane_left[i][0] + lane_right[i][0]) / 2.0,
                 (lane_left[i][1] + lane_right[i][1]) / 2.0)
                for i in range(n)
            ]
        elif has_left:
            center_line = self._create_virtual_center(lane_left, offset_right=True)
        else:
            center_line = self._create_virtual_center(lane_right, offset_right=False)

        if len(center_line) < 2:
            return 0.0

        self.center_line_points = center_line

        # Speed-adaptive lookahead (inspired by MPC predictive horizon)
        # Lookahead = time_horizon * velocity, clamped to safe bounds
        # This naturally reduces lookahead in tight curves (where speed is low)
        if self.in_curve_steering:
            lookahead_time = self.lookahead_time_curve
            lookahead_max = self.lookahead_max_curve
        else:
            lookahead_time = self.lookahead_time_straight
            lookahead_max = self.lookahead_max_straight
        
        # Calculate adaptive lookahead: L = t * v, clamped
        current_lookahead = lookahead_time * max(abs(current_speed), 1.0)
        current_lookahead = np.clip(current_lookahead, self.lookahead_min, lookahead_max)

        # (Adaptive lookahead remains velocity-dependent and clamped)

        # Find goal point using lookahead (from slides: LTA control II)
        # (x_g, y_g) = point on centerline furthest from lane boundaries
        # (x_la, y_la) = (x_g, y_g) + v_la in direction of centerline
        
        # For simplicity: use lookahead distance along centerline
        goal_point = self._find_lookahead_point(center_line, car_x, car_y, current_lookahead)
        if goal_point is None:
            return 0.0
        
        x_la, y_la = goal_point
        self.target_point = goal_point  # For visualization
        
        # Compute error in robot frame (from slides: Linear Model II)
        # e_x = x_ref - x (forward error)
        # e_y = y_ref - y (lateral error) 
        # e_theta = theta_ref - theta (heading error)
        
        # Goal orientation: tangent to centerline at goal point
        theta_goal = self._compute_goal_heading(center_line, goal_point)
        self.target_direction = theta_goal  # For visualization
        
        # Errors in world frame
        w_e_x = x_la - car_x
        w_e_y = y_la - car_y
        w_e_theta = theta_goal - car_theta
        w_e_theta = (w_e_theta + np.pi) % (2 * np.pi) - np.pi
        
        # Transform to robot frame (from slides: Linear Model II)
        # B_e = R(theta)^T * W_e
        cos_th = np.cos(car_theta)
        sin_th = np.sin(car_theta)
        
        e_x = cos_th * w_e_x + sin_th * w_e_y
        e_y = -sin_th * w_e_x + cos_th * w_e_y
        e_theta = w_e_theta
        
        # Reference velocity: use current speed (assuming constant speed along trajectory)
        v_ref = max(abs(current_speed), 1.0)  # Avoid division by zero
        omega_ref = 0.0  # Assume straight reference (can be improved)
        
        # Linear feedback control (from slides: Linear Control III)
        # u = [-K1*e_x, -K2*sgn(v_ref)*e_y - K3*e_theta]
        # For lane keeping, we focus on lateral error (e_y) and heading error (e_theta)
        K1 = 2 * self.zeta * self.omega_n
        K2 = (self.omega_n ** 2 - omega_ref ** 2) / abs(v_ref)
        K3 = 2 * self.zeta * self.omega_n
        
        # Compute angular velocity control
        # Positive e_y means target is to the right -> need positive omega (turn right)
        # Positive e_theta means we need to turn to align -> positive omega
        omega_control = K2 * e_y + K3 * e_theta
        
        # Convert angular velocity to steering angle (Ackermann)
        # omega = v * tan(delta) / L
        # delta = atan(omega * L / v)
        if abs(current_speed) > 0.5:
            steering_angle = np.arctan(omega_control * self.car.wheelbase / current_speed)
        else:
            # At low speeds, use direct proportional control
            steering_angle = omega_control * 0.5
        
        # Clamp and smooth
        steering_angle = np.clip(steering_angle, -self.car.max_steering_angle, self.car.max_steering_angle)
        
        # Simple rate limiting
        max_delta = 0.05  # rad/step
        delta = steering_angle - self.last_omega
        if abs(delta) > max_delta:
            steering_angle = self.last_omega + np.sign(delta) * max_delta
        
        self.last_omega = steering_angle
        return steering_angle
    
    def _find_lookahead_point(self, centerline, car_x, car_y, lookahead_dist):
        """Find point on centerline at lookahead distance ahead of car."""
        if len(centerline) < 2:
            return None
        
        # Find nearest point on centerline that is ahead of the car
        min_dist = float('inf')
        nearest_idx = 0
        for i, (px, py) in enumerate(centerline):
            dist = (px - car_x) ** 2 + (py - car_y) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Make sure we start from a point ahead or use first available
        start_idx = max(0, nearest_idx)
        
        # Walk forward along centerline from start point
        accumulated = 0.0
        for i in range(start_idx, len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i + 1]
            segment_len = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            
            if accumulated + segment_len >= lookahead_dist:
                # Interpolate within this segment
                remaining = lookahead_dist - accumulated
                t = remaining / segment_len if segment_len > 1e-6 else 0.0
                return (
                    p1[0] + t * (p2[0] - p1[0]),
                    p1[1] + t * (p2[1] - p1[1])
                )
            accumulated += segment_len
        
        # If we ran out of centerline, return the last point
        return centerline[-1]
    
    def _compute_goal_heading(self, centerline, goal_point):
        """
        Compute goal heading as the direction from car to lookahead point.
        This is always correct regardless of travel direction along centerline.
        """
        if goal_point is None:
            return 0.0
        
        car_x = self._state.get('x', self.car.x)
        car_y = self._state.get('y', self.car.y)
        
        gx, gy = goal_point
        
        # Direction from car to goal point
        # This automatically handles any travel direction!
        theta_goal = np.arctan2(gy - car_y, gx - car_x)
        
        return theta_goal

    # ------------------------------------------------------------------
    # Curve state detection (used by both steering and speed control)
    # ------------------------------------------------------------------
    def _update_curve_state(self, lane_left, lane_right):
        """
        Update curve state with UNIFIED adaptive time-based exit logic.
        Both steering (lookahead) and speed control use the same hysteresis:
          * Tighter curves require longer confirmation time before exiting
          * Time = f(min_radius) ensures safety and stability
        """
        # Calculate curve radius from visible lane geometry
        curve_radius = self._predict_curve_radius_from_lane(lane_left, lane_right)
        
        # If lane detection fails (None), use fallback
        if curve_radius is None:
            curve_radius = self.fallback_curve_radius
        
        self._last_curve_radius = curve_radius
        
        # UNIFIED: Adaptive time-based curve exit for both steering and speed
        # Tighter curves need longer confirmation before exiting
        if curve_radius < self.curve_threshold:
            # Curve detected (R < 500m)
            self.straight_start_time = None  # Reset exit timer
            
            if not self.in_curve_steering:
                # ENTERING CURVE (both steering and speed)
                self.in_curve_steering = True
                self.in_curve_speed = True
                self.curve_min_radius = curve_radius
            else:
                # Already in curve - track tightest radius
                if curve_radius < self.curve_min_radius:
                    self.curve_min_radius = curve_radius
        else:
            # Straight section detected (R >= 500m)
            if self.in_curve_steering or self.in_curve_speed:
                # Require radius to be significantly higher than threshold
                # This prevents premature exit on radius fluctuations
                exit_threshold = self.curve_threshold * self.curve_exit_radius_factor
                
                if curve_radius >= exit_threshold:
                    # Start counting exit time if not already started
                    if self.straight_start_time is None:
                        import time
                        self.straight_start_time = time.time()
                    
                    # Calculate required exit time based on how tight the curve was
                    # Tighter curves (smaller min_radius) need longer confirmation
                    # Linear interpolation: tight curves (80m) = 4s, gentle curves (300m) = 2s
                    if self.curve_min_radius < 80:
                        required_exit_time = self.curve_exit_time_max
                    elif self.curve_min_radius > 300:
                        required_exit_time = self.curve_exit_time_min
                    else:
                        # Linear interpolation between min and max
                        t = (self.curve_min_radius - 80) / (300 - 80)
                        required_exit_time = self.curve_exit_time_max - t * (self.curve_exit_time_max - self.curve_exit_time_min)
                    
                    # Check if enough time has passed
                    import time
                    elapsed_time = time.time() - self.straight_start_time
                    
                    if elapsed_time >= required_exit_time:
                        # EXITING CURVE - confirmed straight for required duration
                        # Both steering and speed exit together
                        self.in_curve_steering = False
                        self.in_curve_speed = False
                        self.straight_start_time = None
                        self.curve_min_radius = float('inf')
                        # Mark that we've just exited a curve so steering lookahead
                        # can be locked to the straight maximum for improved preview.
                        self.just_exited_curve = True
                else:
                    # Radius dropped below exit threshold - reset timer
                    self.straight_start_time = None
            else:
                # Not in curve - ensure timer is reset
                self.straight_start_time = None

            # If we detect a curve again, clear the just_exited flag so lookahead
            # returns to normal curve behaviour.
            if curve_radius < self.curve_threshold:
                self.just_exited_curve = False

    # ------------------------------------------------------------------
    # Speed control (lane geometry based)
    # ------------------------------------------------------------------
    def _calculate_speed_control(self, lane_left, lane_right):
        """
        Calculate safe speed based on LANE GEOMETRY ahead (not car steering).
        Curve state already updated by _update_curve_state().
        Formula: v_safe = sqrt(a_lat * R) * safety_factor
        
        ONLY applies speed limiting when in a curve - straights are unrestricted.
        """
        current_speed = abs(self._state.get("velocity", 0.0))
        
        # Only apply speed control when in curve
        # On straights, don't interfere with driver's speed choice
        if not self.in_curve_speed:
            # Not in curve - no speed limiting
            self.safe_speed = float('inf')
            self.overspeed_state = False
            return {
                "safe_speed": float('inf'),
                "over_speed": False,
                "brake": 0.0,
            }
        
        # IN CURVE: Calculate safe speed based on geometry
        # Use already-calculated radius from _update_curve_state
        curve_radius = self._last_curve_radius
        
        # Calculate instantaneous safe speed for current radius: v = sqrt(a_lat * R)
        instantaneous_safe_speed = np.sqrt(self.lateral_accel_limit * curve_radius) * self.safe_speed_scale
        instantaneous_safe_speed = min(instantaneous_safe_speed, self.car.max_velocity)
        
        # In curve - track minimum safe speed
        if self.in_curve_speed:
            if not hasattr(self, 'curve_min_safe_speed') or self.curve_min_safe_speed == float('inf'):
                # First frame in curve
                self.curve_min_safe_speed = instantaneous_safe_speed
            elif instantaneous_safe_speed < self.curve_min_safe_speed:
                # Tighter section detected
                self.curve_min_safe_speed = instantaneous_safe_speed
            
            # Use minimum safe speed for this curve
            safe_speed = self.curve_min_safe_speed
        
        self.safe_speed = safe_speed
        
        # Determine if we're overspeeding (with hysteresis to prevent oscillation)
        if not self.overspeed_state:
            # Not currently warning - trigger when exceeding safe speed
            if current_speed > safe_speed:
                self.overspeed_state = True
        else:
            # Currently warning - clear only when speed drops 10% below safe speed
            if current_speed <= safe_speed * 0.90:
                self.overspeed_state = False
        
        over_speed = self.overspeed_state
        
        # Calculate brake command if overspeeding
        brake_cmd = 0.0
        if over_speed:
            # Progressive braking based on speed excess
            speed_excess = current_speed - safe_speed
            speed_excess_ratio = speed_excess / max(safe_speed, 1e-3)
            
            # Scale brake from 30% to 100%
            brake_cmd = np.clip(0.3 + speed_excess_ratio * 3.0, 0.3, 1.0)

        return {
            "safe_speed": safe_speed,
            "over_speed": over_speed,
            "brake": brake_cmd,
        }

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------
    def _update_warnings(self, steering_command, speed_command, lane_offset=None):
        if lane_offset is None:
            lane_offset = self._estimate_lane_offset()

        # Use overspeed flag from speed_command (includes hysteresis)
        speed_too_high = bool(speed_command and speed_command.get("over_speed", False))

        lane_departure = abs(lane_offset) > self.warning_lateral_offset if lane_offset is not None else False

        self.warnings["speed_too_high"] = speed_too_high
        self.warnings["lane_departure"] = lane_departure

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_vehicle_state(self):
        """Get vehicle state exclusively from sensors (GNSS, IMU, wheel encoders)"""
        if self.sensors is not None:
            state = self.sensors.get_state_estimate()
            if state is not None:
                # Sensors should provide all necessary state - no fallbacks to raw car data
                return state
        
        # If no sensors available, return empty state (controller will use defaults)
        # NEVER use self.car.x/y/theta/velocity directly - those are ground truth, not sensor data
        return {
            "x": 0.0,
            "y": 0.0,
            "theta": 0.0,
            "velocity": 0.0,
        }

    def _heading_error(self):
        if self.target_direction is None:
            return 0.0
        return (self.target_direction - self._state.get("theta", 0.0) + np.pi) % (2 * np.pi) - np.pi

    def _estimate_lane_offset(self):
        if not self.center_line_points:
            return 0.0
        car_x, car_y, car_theta = self._state["x"], self._state["y"], self._state["theta"]
        perp_x, perp_y = -np.sin(car_theta), np.cos(car_theta)
        nearest = min(self.center_line_points, key=lambda p: (p[0]-car_x) ** 2 + (p[1]-car_y) ** 2)
        dx, dy = nearest[0] - car_x, nearest[1] - car_y
        return dx * perp_x + dy * perp_y

    def _curve_radius_from_centerline(self):
        if len(self.center_line_points) < 3:
            return None
        p1, p2, p3 = self.center_line_points[0], self.center_line_points[len(self.center_line_points)//2], self.center_line_points[-1]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        a = x1 - x2
        b = y1 - y2
        c = x1 - x3
        d = y1 - y3
        e = a * (x1 + x2) + b * (y1 + y2)
        f = c * (x1 + x3) + d * (y1 + y3)
        g = 2 * (a * (y3 - y2) - b * (x3 - x2))
        if abs(g) < 1e-6:
            return float("inf")
        cx = (d * e - b * f) / g
        cy = (a * f - c * e) / g
        return max(np.hypot(x1 - cx, y1 - cy), 1.0)

    def _predict_curve_radius_from_lane(self, lane_left, lane_right):
        """
        Calculate curve radius from LANE GEOMETRY ahead.
        Uses polynomial fitting and curvature formula.
        """
        # Get center line points from lane boundaries
        n = min(len(lane_left), len(lane_right))
        if n < 5:
            return float('inf')  # Not enough points
        
        # Build centerline from lane boundaries
        center_points = []
        for i in range(n):
            cx = (lane_left[i][0] + lane_right[i][0]) / 2.0
            cy = (lane_left[i][1] + lane_right[i][1]) / 2.0
            center_points.append((cx, cy))
        
        if len(center_points) < 5:
            return float('inf')
        
        # Extract x, y arrays
        points_x = np.array([p[0] for p in center_points])
        points_y = np.array([p[1] for p in center_points])
        
        # Check if it's essentially a straight line using R² test
        dx = points_x[-1] - points_x[0]
        dy = points_y[-1] - points_y[0]
        
        try:
            if abs(dx) > abs(dy):
                # Fit y = mx + b (line)
                coeffs = np.polyfit(points_x, points_y, 1)
                y_fit = np.polyval(coeffs, points_x)
                ss_res = np.sum((points_y - y_fit) ** 2)
                ss_tot = np.sum((points_y - np.mean(points_y)) ** 2)
            else:
                # Fit x = my + b (line)
                coeffs = np.polyfit(points_y, points_x, 1)
                x_fit = np.polyval(coeffs, points_y)
                ss_res = np.sum((points_x - x_fit) ** 2)
                ss_tot = np.sum((points_x - np.mean(points_x)) ** 2)
            
            if ss_tot > 1e-10:
                r_squared = 1 - (ss_res / ss_tot)
                # If R² > 0.995, it's a straight line
                if r_squared > 0.995:
                    return float('inf')
        except:
            pass
        
        # Calculate curvature using 2nd order polynomial fit
        # Use middle portion of detected lane (most reliable data)
        try:
            if abs(dx) > abs(dy):
                # Fit y = a*x² + b*x + c
                coeffs = np.polyfit(points_x, points_y, 2)
                a, b, c = coeffs
                
                # Evaluate at middle point
                mid_x = np.mean(points_x)
                # dy/dx = 2*a*x + b
                dydx = 2 * a * mid_x + b
                # d²y/dx² = 2*a
                d2ydx2 = 2 * a
                
                # Curvature: κ = |d²y/dx²| / (1 + (dy/dx)²)^(3/2)
                curvature = abs(d2ydx2) / ((1 + dydx**2) ** 1.5)
            else:
                # Fit x = a*y² + b*y + c
                coeffs = np.polyfit(points_y, points_x, 2)
                a, b, c = coeffs
                
                # Evaluate at middle point
                mid_y = np.mean(points_y)
                # dx/dy = 2*a*y + b
                dxdy = 2 * a * mid_y + b
                # d²x/dy² = 2*a
                d2xdy2 = 2 * a
                
                # Curvature: κ = |d²x/dy²| / (1 + (dx/dy)²)^(3/2)
                curvature = abs(d2xdy2) / ((1 + dxdy**2) ** 1.5)
            
            # Radius = 1 / curvature
            if curvature > 1e-6:
                radius = 1.0 / curvature
            else:
                radius = float('inf')
            
            return max(radius, 1.0)  # Minimum 1m radius
            
        except Exception as e:
            # Fallback to 3-point circle if polynomial fails
            if len(center_points) >= 3:
                idx1 = len(center_points) // 4
                idx2 = len(center_points) // 2
                idx3 = (3 * len(center_points)) // 4
                
                p1 = center_points[idx1]
                p2 = center_points[idx2]
                p3 = center_points[idx3]
                
                return self._circle_radius_from_3_points(p1, p2, p3)
            
            return float('inf')
    
    def _circle_radius_from_3_points(self, p1, p2, p3):
        """Calculate radius of circle passing through 3 points"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        a = x1 - x2
        b = y1 - y2
        c = x1 - x3
        d = y1 - y3
        e = a * (x1 + x2) + b * (y1 + y2)
        f = c * (x1 + x3) + d * (y1 + y3)
        g = 2 * (a * (y3 - y2) - b * (x3 - x2))
        
        if abs(g) < 1e-6:
            return float('inf')  # Collinear points (straight line)
        
        cx = (d * e - b * f) / g
        cy = (a * f - c * e) / g
        radius = np.sqrt((x1 - cx)**2 + (y1 - cy)**2)
        
        return max(radius, 1.0)  # Minimum 1m radius

    def _create_virtual_center(self, lane_boundary, offset_right):
        if len(lane_boundary) < 3:
            return []
        xs = np.array([p[0] for p in lane_boundary])
        ys = np.array([p[1] for p in lane_boundary])
        try:
            a, b, c = np.polyfit(xs, ys, 2)
        except Exception:
            shift = self.lane_width / 2.0 * (1 if offset_right else -1)
            return [(x, y + shift) for x, y in lane_boundary]

        offset_dist = self.lane_width / 2.0
        virtual_center = []
        for x_val in xs:
            y_poly = a * x_val ** 2 + b * x_val + c
            tangent_slope = 2 * a * x_val + b
            tangent_angle = np.arctan(tangent_slope)
            offset_angle = tangent_angle + (np.pi / 2 if offset_right else -np.pi / 2)
            vx = x_val + offset_dist * np.cos(offset_angle)
            vy = y_poly + offset_dist * np.sin(offset_angle)
            virtual_center.append((vx, vy))
        return virtual_center

    def _select_active_lane(self, current_lane, left_lane, center_lane, right_lane):
        # Prefer the lane the camera thinks we're in; otherwise choose side with more points.
        if current_lane in ("LEFT", "CENTER", "RIGHT"):
            return current_lane
        counts = {
            "LEFT": len(left_lane),
            "CENTER": len(center_lane),
            "RIGHT": len(right_lane),
        }
        best = max(counts, key=counts.get)
        return best if counts[best] >= self.min_points_for_direction else None


__all__ = ["LinearLKAController"]
