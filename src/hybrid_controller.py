"""
Hybrid Controller Module - Direction-based LKA + Predictive Speed Control
Part of the 3D Robotics Lab simulation.

Combines:
- Direction-following LKA (tangent-based, single-line capable)
- MPC-style speed prediction (curve detection, anticipatory braking)
- 3 control modes: Manual, Warning, Assist
"""

import numpy as np


class HybridLaneController:
    """
    Hybrid Lane Keeping and Speed Control System

    Features:
    - Direction-based steering (follows lane tangent)
    - Single-line operation (works with only one boundary visible)
    - Predictive curve detection (looks ahead ~2.7 seconds)
    - Comfortable anticipatory braking (0.3g limit)
    - 3 modes: Manual / Warning / Assist
    """

    # Control Modes
    MODE_MANUAL = 0      # No assistance (full manual control)
    MODE_WARNING = 1     # Monitoring only (warnings, no control)
    MODE_ASSIST = 2      # Active assistance (blended control)

    def __init__(self, car, camera):
        self.car = car
        self.camera = camera

        # Current mode and intervention state
        self.mode = self.MODE_MANUAL
        self.intervening = False  # True when assist temporarily takes over

        # Debug flag
        self.debug = False  # Set to True to enable debug output

        # ================================================================
        # DIRECTION FOLLOWING PARAMETERS (LKA)
        # ================================================================
        self.lane_width = 4.0  # meters (match narrower lane width from reference)
        self.min_points_for_direction = 4  # minimum points to trust direction

        # Rolling MEDIAN smoothing (BFMC professional approach - uses median not average)
        self.steering_history = []  # Store last N steering angles
        self.rolling_median_window = 3  # Increased to 3 for better high-speed stability
        self.last_steering = 0.0  # Fallback when lanes lost (BFMC approach)
        self.last_heading_error = 0.0  # Cache heading error vs lane tangent for warnings
        self.last_lateral_error = 0.0  # Cache lateral error for warnings
        self.steering_lpf_alpha = 0.10  # low-pass filter for smoother outputs

        # Adaptive lane width for sharp turns (BFMC technique)
        self.base_lane_offset = self.lane_width / 2.0  # Base offset for virtual center
        self.sharp_turn_threshold = 0.0001  # Curvature threshold for sharp turn detection
        self.sharp_turn_multiplier = 1.3  # Widen virtual lane by 30% in sharp turns

        # Image-height-equivalent reference (BFMC uses image height in pixels ~720)
        # Balanced at 14m for good curve entry without over-aggressiveness
        self.image_height_equivalent = 14.0  # meters - balanced for stability

        # Heading error correction (fixes post-curve oscillation)
        self.heading_correction_weight = 0.4  # Match controller2 feel

        # ================================================================
        # SPEED PREDICTION PARAMETERS (MPC-based)
        # ================================================================
        self.prediction_horizon = 25  # steps (increased to see further ahead)
        self.prediction_dt = 0.18  # seconds per step
        self.lateral_accel_limit = 0.3 * 9.81  # 0.3g safe limit

        # Speed control
        self.comfort_margin = 0.82  # target ~82% of max safe speed (stricter)
        self.brake_threshold = 0.85  # start braking earlier to be conservative
        self.max_comfort_decel = 1.0 * 9.81  # 1.0g maximum braking
        self.accel_threshold = 1.0  # accelerate only if below threshold ratio

        # Curve radius estimation (running average)
        self.radius_history = []
        self.radius_history_size = 3

        # Track minimum safe speed in current curve
        self.in_curve = False
        self.curve_min_safe_speed = float('inf')
        self.curve_entry_threshold = 500  # meters - radius below this means we're in a curve

        # Cached lane/speed state for robustness
        self.last_known_lane = None
        self._last_safe_speed = None
        self._last_curve_radius = None

        # ================================================================
        # INTERVENTION PARAMETERS (Mode 3: Assist)
        # ================================================================
        # Lateral intervention zones
        self.no_intervention_zone = 0.5  # meters from center
        self.gentle_intervention_zone = 1.5  # meters from center
        # Beyond gentle zone = strong intervention
        self.intervention_release_speed_margin = 1.05  # exit assist when back under ~105% of safe speed

        # ================================================================
        # WARNING SYSTEM (Mode 2: Warning)
        # ================================================================
        self.warnings = {
            'lane_departure': False,
            'speed_too_high': False,
            'time_to_crossing': False
        }
        self.warning_thresholds = {
            'lateral_offset_warn': 1.2,  # meters (looser to avoid false warnings)
            'time_to_crossing_warn': 1.0,  # seconds
            'speed_margin_warn': 1.05  # 5% over safe speed (stricter)
        }

        # ================================================================
        # VISUALIZATION DATA
        # ================================================================
        self.predicted_path = []
        self.target_direction = None
        self.target_point = None
        self.center_line_points = []  # Center points used for direction following
        self.current_curve_radius = float('inf')
        self.safe_speed = None
        self.intervention_strength = 0.0
        self.heading_divergence_threshold = 0.2  # rad; require stronger divergence
        self.heading_divergence_offset_gate = 0.4  # m; ignore tiny offsets for divergence checks
        self.heading_divergence_heading_gate = 0.05  # rad; ignore tiny heading noise
        self.lane_offset_intervention = 0.7  # meters; stricter lane drift trigger
        self.speed_overshoot_margin = 1.03  # 3% over safe speed triggers assist
        self.release_stable_frames = 5  # consecutive frames required before releasing assist
        self._stable_release_counter = 0
        self.hazard_stable_frames = 2  # consecutive hazard frames to engage assist
        self._hazard_counter = 0
        self.heading_target_window_enter = 0.14  # rad (~8 deg) vs next target point for engage
        self.heading_target_window_release = 0.08  # rad (~4.5 deg) for release
        self.lane_departure_brake_cap = 0.5  # cap lane-departure braking (avoid full stop)
        self.curvature_sticky_threshold = 0.0012  # keep assist engaged briefly in tighter curves
        self.curve_hold_frames = 8
        self._curve_hold_counter = 0
        self._steering_curvature_proxy = 0.0

    def set_mode(self, mode):
        """Set control mode (0=Manual, 1=Warning, 2=Assist)"""
        if mode in [self.MODE_MANUAL, self.MODE_WARNING, self.MODE_ASSIST]:
            self.mode = mode
            self.warnings = {k: False for k in self.warnings}  # Clear warnings
            print(f"Hybrid Controller Mode: {['MANUAL', 'WARNING', 'ASSIST'][mode]}")
            return True
        return False

    def get_mode_name(self):
        """Get current mode name"""
        return ['MANUAL', 'WARNING', 'ASSIST'][self.mode]

    @property
    def active(self):
        """Return True if controller is active (not in manual mode)"""
        return self.mode != self.MODE_MANUAL

    def deactivate(self):
        """Deactivate controller (set to manual mode)"""
        self.set_mode(self.MODE_MANUAL)

    def calculate_control(self, track):
        """
        Main control loop
        Returns: (steering_angle, throttle, brake, warnings, intervening)
        """
        if self.mode == self.MODE_MANUAL:
            # No assistance
            self.intervening = False
            return None, None, None, {}, False

        # Get lane detection from camera
        left_lane, center_lane, right_lane = self.camera.last_measurement
        current_lane = self.camera.current_lane

        active_lane = self._select_active_lane(current_lane, left_lane, center_lane, right_lane)
        if active_lane in ("LEFT", "RIGHT"):
            self.last_known_lane = active_lane
        else:
            self.intervening = False
            return None, None, None, {}, False

        # Determine lane boundaries based on which lane we're in (fallback to nearest available edge)
        if active_lane == "LEFT":
            lane_left = left_lane if len(left_lane) > 0 else center_lane
            lane_right = center_lane if len(center_lane) > 0 else right_lane
        else:
            lane_left = center_lane if len(center_lane) > 0 else left_lane
            lane_right = right_lane if len(right_lane) > 0 else center_lane

        # ============================================================
        # STEERING CONTROL (Direction Following)
        # ============================================================
        current_speed = abs(self.car.velocity)
        steering_command = self._calculate_steering_direction(
            lane_left, lane_right, current_speed
        )

        # ============================================================
        # SPEED CONTROL (Curve Prediction)
        # ============================================================
        speed_command = self._calculate_speed_control(
            lane_left, lane_right
        )
        lane_offset = self._estimate_lane_offset()

        # ============================================================
        # MODE-SPECIFIC BEHAVIOR
        # ============================================================
        if self.mode == self.MODE_WARNING:
            # Warning mode: monitor, surface lane drift + overspeed warnings
            self._update_warnings(steering_command, speed_command, lane_offset=lane_offset, speed_only=False)
            self.intervening = False
            return None, None, None, self.warnings, False

        # Assist mode: provide steering and throttle/brake when needed
        self._update_warnings(steering_command, speed_command, lane_offset=lane_offset)
        self._update_intervention_state(lane_offset, speed_command)

        # Base commands from speed planner (default to coasting if absent)
        throttle_cmd = 0.0
        brake_cmd = 0.0
        if speed_command:
            throttle_cmd = speed_command.get('throttle', 0.0)
            brake_cmd = speed_command.get('brake', 0.0)

        # If we are intervening, enforce braking when drifting or overspeeding
        if self.intervening:
            lateral_speed_est = 0.0
            if steering_command is not None:
                lateral_speed_est = abs(self.car.velocity) * abs(np.tan(steering_command))

            lane_brake = 0.0
            if self.warnings.get('lane_departure'):
                lane_brake = np.clip(
                    (abs(lane_offset) / self.lane_width) + (lateral_speed_est / 8.0),
                    0.0,
                    self.lane_departure_brake_cap,
                )

            overspeed_brake = 0.0
            if speed_command and speed_command.get('action') == 'brake':
                overspeed_brake = speed_command.get('brake', 0.0) or 0.0
            elif self.safe_speed and self.safe_speed > 0:
                current_speed = abs(self.car.velocity)
                if current_speed > self.safe_speed * self.speed_overshoot_margin:
                    overspeed_brake = np.clip(
                        (current_speed - self.safe_speed) / (self.safe_speed + 1e-3),
                        0.0,
                        1.0,
                    )

            brake_cmd = max(brake_cmd or 0.0, lane_brake, overspeed_brake)
            if brake_cmd > 0:
                throttle_cmd = 0.0

        # In assist mode the system always commands steering/throttle/brake; human input is ignored.
        return steering_command, throttle_cmd, brake_cmd, self.warnings, self.intervening

    # =======================================================================
    # INTERNAL METHODS - STEERING
    # =======================================================================
    def _calculate_steering_direction(self, lane_left, lane_right, current_speed):
        """Compute steering toward a smoothed, densified centerline with adaptive lookahead."""
        car_x, car_y, car_theta = self.car.x, self.car.y, self.car.theta

        has_left = len(lane_left) >= self.min_points_for_direction
        has_right = len(lane_right) >= self.min_points_for_direction

        if not has_left and not has_right:
            return self.last_steering

        # Build center line
        if has_left and has_right:
            n = min(len(lane_left), len(lane_right))
            center_avg = [
                ((lane_left[i][0] + lane_right[i][0]) / 2.0,
                 (lane_left[i][1] + lane_right[i][1]) / 2.0)
                for i in range(n)
            ]
            if len(center_avg) >= 3:
                try:
                    points_x = np.array([p[0] for p in center_avg])
                    points_y = np.array([p[1] for p in center_avg])
                    a, b, c = np.polyfit(points_x, points_y, 2)
                    curvature = abs(a)
                    base_lookahead = 2
                    if curvature > 0.0002:
                        lookahead_points = 0
                    elif curvature > 0.0001:
                        lookahead_points = 1
                    else:
                        lookahead_points = base_lookahead
                        if current_speed > 15.0:
                            lookahead_points = 3

                    self.center_line_points = []
                    for i, x_val in enumerate(points_x):
                        if lookahead_points > 0:
                            lookahead_idx = min(i + lookahead_points, len(points_x) - 1)
                            x_lookahead = points_x[lookahead_idx]
                            y_poly = a * x_lookahead**2 + b * x_lookahead + c
                            tangent_slope = 2 * a * x_lookahead + b
                            y_current = y_poly - (x_lookahead - x_val) * tangent_slope
                            self.center_line_points.append((x_val, y_current))
                        else:
                            self.center_line_points.append(center_avg[i])
                except Exception:
                    self.center_line_points = center_avg
            else:
                self.center_line_points = center_avg
        elif has_left:
            self.center_line_points = self._create_virtual_center(
                lane_left, car_x, car_y, car_theta, offset_right=True, current_speed=current_speed
            )
        else:
            self.center_line_points = self._create_virtual_center(
                lane_right, car_x, car_y, car_theta, offset_right=False, current_speed=current_speed
            )

        if len(self.center_line_points) < 2:
            return self.last_steering

        forward_x = np.cos(car_theta)
        forward_y = np.sin(car_theta)
        perp_x = -np.sin(car_theta)
        perp_y = np.cos(car_theta)

        # Densify and smooth center line to avoid jagged steering, more points in curves
        dense_points = []
        for i in range(len(self.center_line_points) - 1):
            p1 = self.center_line_points[i]
            p2 = self.center_line_points[i + 1]
            dense_points.append(p1)
            seg_dx = p2[0] - p1[0]
            seg_dy = p2[1] - p1[1]
            seg_len = np.hypot(seg_dx, seg_dy)
            if seg_len > 0.1:
                # insert midpoints proportional to segment length (more in longer/curvier parts)
                steps = int(max(1, min(4, seg_len / 2.0)))
                for s in range(1, steps):
                    t = s / steps
                    dense_points.append((p1[0] + seg_dx * t, p1[1] + seg_dy * t))
        dense_points.append(self.center_line_points[-1])

        # Smooth with small moving average window
        smoothed_points = []
        window = 3
        for i in range(len(dense_points)):
            start = max(0, i - 1)
            end = min(len(dense_points), i + window)
            xs = [dense_points[j][0] for j in range(start, end)]
            ys = [dense_points[j][1] for j in range(start, end)]
            smoothed_points.append((np.mean(xs), np.mean(ys)))

        # Dynamic lookahead based on speed and curvature proxy
        curvature_proxy = 0.0
        if len(smoothed_points) >= 3:
            p1 = smoothed_points[0]
            p2 = smoothed_points[len(smoothed_points) // 2]
            p3 = smoothed_points[-1]
            area = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            base = np.hypot(p3[0]-p1[0], p3[1]-p1[1]) + 1e-3
            curvature_proxy = area / (base**3)

        lookahead = np.clip(current_speed * 0.5, 5.0, 22.0)
        if curvature_proxy > 0.002:
            lookahead = max(4.0, lookahead * 0.30)
        elif curvature_proxy > 0.001:
            lookahead = max(5.0, lookahead * 0.5)

        # Track curvature proxy for assist stickiness
        self._steering_curvature_proxy = curvature_proxy

        # Select target point along arc length starting from the nearest forward point
        forward_dists = []
        for point in smoothed_points:
            dxp = point[0] - car_x
            dyp = point[1] - car_y
            forward_dists.append(dxp * forward_x + dyp * forward_y)

        forward_indices = [i for i, fd in enumerate(forward_dists) if fd > 0.1]
        if forward_indices:
            start_idx = min(forward_indices, key=lambda i: forward_dists[i])
            target_point = smoothed_points[start_idx]
            remaining = lookahead
            for j in range(start_idx, len(smoothed_points) - 1):
                p_curr = smoothed_points[j]
                p_next = smoothed_points[j + 1]
                seg_dx = p_next[0] - p_curr[0]
                seg_dy = p_next[1] - p_curr[1]
                seg_len = np.hypot(seg_dx, seg_dy)
                if seg_len < 1e-3:
                    continue
                if remaining <= seg_len:
                    ratio = remaining / seg_len
                    target_point = (
                        p_curr[0] + seg_dx * ratio,
                        p_curr[1] + seg_dy * ratio,
                    )
                    break
                remaining -= seg_len
            else:
                target_point = smoothed_points[-1]
        else:
            target_point = smoothed_points[-1]

        dx = target_point[0] - car_x
        dy = target_point[1] - car_y
        lateral_error = dx * perp_x + dy * perp_y
        heading_to_target = np.arctan2(dy, dx) - car_theta
        while heading_to_target > np.pi:
            heading_to_target -= 2 * np.pi
        while heading_to_target < -np.pi:
            heading_to_target += 2 * np.pi

        self.target_point = target_point
        self.target_direction = np.arctan2(
            self.target_point[1] - car_y,
            self.target_point[0] - car_x
        )

        combined_error = lateral_error + heading_to_target * self.image_height_equivalent * self.heading_correction_weight
        self.last_heading_error = heading_to_target
        self.last_lateral_error = combined_error

        raw_steering_degrees = 90.0 - np.degrees(np.arctan2(self.image_height_equivalent, combined_error))
        steering_angle = np.clip(
            np.radians(raw_steering_degrees),
            -self.car.max_steering_angle,
            self.car.max_steering_angle,
        )

        self.steering_history.insert(0, steering_angle)
        if len(self.steering_history) > self.rolling_median_window:
            self.steering_history.pop()
        steering_smoothed = np.median(self.steering_history)

        # Low-pass blend with previous steering for extra smoothness
        steering_filtered = (
            self.steering_lpf_alpha * steering_smoothed
            + (1.0 - self.steering_lpf_alpha) * self.last_steering
        )

        self.last_steering = steering_filtered
        return steering_filtered

    # =======================================================================
    # INTERNAL METHODS - SPEED
    # =======================================================================
    def _calculate_speed_control(self, lane_left, lane_right):
        """Predict curvature ahead and set safe speed. Returns dict with action/brake/throttle."""
        curve_radius = self._predict_curve_radius_from_lane(lane_left, lane_right)
        if curve_radius is None or not np.isfinite(curve_radius):
            return self._conservative_speed_command_from_cache()

        self.radius_history.append(curve_radius)
        if len(self.radius_history) > self.radius_history_size:
            self.radius_history.pop(0)
        radius_smoothed = np.mean(self.radius_history)
        self.current_curve_radius = radius_smoothed

        if not np.isfinite(radius_smoothed) or radius_smoothed <= 0:
            return self._conservative_speed_command_from_cache()

        safe_speed = np.sqrt(self.lateral_accel_limit * max(radius_smoothed, 1e-3))
        if not np.isfinite(safe_speed) or safe_speed <= 0:
            return self._conservative_speed_command_from_cache()
        safe_speed = min(safe_speed, self.car.max_velocity)

        if radius_smoothed < self.curve_entry_threshold:
            if not self.in_curve:
                self.in_curve = True
                self.curve_min_safe_speed = safe_speed
            else:
                self.curve_min_safe_speed = min(self.curve_min_safe_speed, safe_speed)
            safe_speed = self.curve_min_safe_speed
        else:
            self.in_curve = False
            self.curve_min_safe_speed = float('inf')

        self.safe_speed = safe_speed
        self._last_safe_speed = safe_speed
        self._last_curve_radius = radius_smoothed

        current_speed = abs(self.car.velocity)

        if radius_smoothed < self.curve_entry_threshold and current_speed > safe_speed * self.brake_threshold:
            speed_error = current_speed - safe_speed * self.comfort_margin
            brake_cmd = np.clip(speed_error / (safe_speed + 1e-3), 0.0, 1.0)
            brake_cmd = min(brake_cmd, 1.0)
            return {
                'action': 'brake',
                'brake': brake_cmd,
                'throttle': 0.0,
                'curve_radius': radius_smoothed,
                'safe_speed': safe_speed
            }
        elif current_speed < safe_speed * self.comfort_margin * self.accel_threshold:
            throttle_cmd = np.clip((safe_speed - current_speed) / safe_speed, 0.0, 1.0)
            return {
                'action': 'accelerate',
                'throttle': throttle_cmd,
                'brake': 0.0,
                'curve_radius': radius_smoothed,
                'safe_speed': safe_speed
            }

        return {
            'action': 'hold',
            'throttle': 0.0,
            'brake': 0.0,
            'curve_radius': radius_smoothed,
            'safe_speed': safe_speed
        }

    def _conservative_speed_command_from_cache(self):
        """Fallback speed command when lane samples are sparse; prefer slowing/coasting."""
        if self._last_safe_speed is None:
            return None
        safe_speed = self._last_safe_speed
        current_speed = abs(self.car.velocity)
        curve_radius = self._last_curve_radius if self._last_curve_radius is not None else float('inf')

        if current_speed > safe_speed * self.brake_threshold:
            speed_error = current_speed - safe_speed * self.comfort_margin
            brake_cmd = np.clip(speed_error / (safe_speed + 1e-3), 0.0, 1.0)
            return {
                'action': 'brake',
                'brake': brake_cmd,
                'throttle': 0.0,
                'curve_radius': curve_radius,
                'safe_speed': safe_speed
            }

        return {
            'action': 'hold',
            'throttle': 0.0,
            'brake': 0.0,
            'curve_radius': curve_radius,
            'safe_speed': safe_speed
        }

    def _estimate_lane_offset(self):
        """Estimate signed lateral offset using the nearest forward center point."""
        if len(self.center_line_points) == 0:
            return 0.0

        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        best = None
        best_dist = float('inf')
        for cx, cy in self.center_line_points:
            dx = cx - car_x
            dy = cy - car_y
            forward = dx * np.cos(car_theta) + dy * np.sin(car_theta)
            if forward < 0.0:
                continue
            if forward < best_dist:
                best_dist = forward
                best = (dx, dy)

        if best is None:
            dx = self.center_line_points[0][0] - car_x
            dy = self.center_line_points[0][1] - car_y
        else:
            dx, dy = best

        local_y = dx * (-np.sin(car_theta)) + dy * np.cos(car_theta)
        return local_y

    # =======================================================================
    # INTERNAL METHODS - WARNINGS
    # =======================================================================
    def _update_warnings(self, steering_command, speed_command, lane_offset=None, speed_only=False):
        """Update warning flags based on current state"""
        if lane_offset is None:
            lane_offset = self._estimate_lane_offset()

        heading_error = getattr(self, 'last_heading_error', 0.0)
        heading_diverging = (
            abs(lane_offset) > self.heading_divergence_offset_gate
            and abs(heading_error) > self.heading_divergence_heading_gate
            and (lane_offset * heading_error) > self.heading_divergence_threshold
        )

        speed_too_high = False
        time_to_crossing = False

        safe_speed = None
        if speed_command and speed_command.get('safe_speed'):
            safe_speed = speed_command['safe_speed']
        elif self.safe_speed:
            safe_speed = self.safe_speed

        if safe_speed:
            current_speed = abs(self.car.velocity)
            if safe_speed > 0 and current_speed > safe_speed * self.warning_thresholds['speed_margin_warn']:
                speed_too_high = True

        lane_departure = False
        if not speed_only:
            # Time to lane crossing (approx)
            lateral_speed = 0.0  # No direct lateral speed; approximate with steering-induced lateral rate
            if abs(self.car.velocity) > 0.1 and steering_command is not None:
                lateral_speed = abs(self.car.velocity) * np.tan(steering_command)
                if lateral_speed > 0.01:
                    time_to_crossing_est = abs(lane_offset) / lateral_speed
                    if time_to_crossing_est < self.warning_thresholds['time_to_crossing_warn']:
                        time_to_crossing = True
            lane_departure = (
                abs(lane_offset) > self.warning_thresholds['lateral_offset_warn']
                or heading_diverging
                or time_to_crossing
            )

        self.warnings['lane_departure'] = lane_departure
        self.warnings['speed_too_high'] = speed_too_high
        self.warnings['time_to_crossing'] = time_to_crossing

        # Intervention strength for HUD (0-1)
        if self.mode == self.MODE_ASSIST:
            self.intervention_strength = 1.0 if self.intervening else 0.0
        else:
            self.intervention_strength = 0.0

    def _update_intervention_state(self, lane_offset, speed_command):
        """Determine whether assist should take over and when to release."""
        if self.mode != self.MODE_ASSIST:
            self.intervening = False
            return

        safe_speed = None
        if speed_command and speed_command.get('safe_speed'):
            safe_speed = speed_command['safe_speed']
        elif self.safe_speed:
            safe_speed = self.safe_speed

        current_speed = abs(self.car.velocity)
        overspeed = False
        if safe_speed is not None and safe_speed > 0:
            overspeed = current_speed > safe_speed * self.speed_overshoot_margin

        heading_error = getattr(self, 'last_heading_error', 0.0)

        # Prefer heading to the next follow point (target point) for engagement logic
        heading_to_target = heading_error
        if self.target_direction is not None:
            car_heading = self.car.theta
            heading_to_target = self.target_direction - car_heading
            while heading_to_target > np.pi:
                heading_to_target -= 2 * np.pi
            while heading_to_target < -np.pi:
                heading_to_target += 2 * np.pi

        heading_diverging = (
            abs(lane_offset) > self.heading_divergence_offset_gate
            and abs(heading_to_target) > self.heading_divergence_heading_gate
            and (lane_offset * heading_to_target) > self.heading_divergence_threshold
        )

        brake_requested = speed_command and speed_command.get('action') == 'brake'

        sticky_curve = getattr(self, "_steering_curvature_proxy", 0.0) > self.curvature_sticky_threshold

        hazard = (
            abs(lane_offset) > self.lane_offset_intervention
            or abs(heading_to_target) > self.heading_target_window_enter
            or heading_diverging
            or overspeed
            or brake_requested
            or self.warnings.get('lane_departure')
        )

        if hazard:
            self._hazard_counter += 1
            if sticky_curve:
                self._curve_hold_counter = max(self._curve_hold_counter, self.curve_hold_frames)
            if self._hazard_counter >= self.hazard_stable_frames:
                self.intervening = True
                self.intervention_strength = 1.0
                if sticky_curve:
                    self._curve_hold_counter = max(self._curve_hold_counter, self.curve_hold_frames)
            self._stable_release_counter = 0
            return
        else:
            self._hazard_counter = 0

        if sticky_curve and self.intervening:
            self._curve_hold_counter = max(self._curve_hold_counter, self.curve_hold_frames)
        elif self._curve_hold_counter > 0:
            self._curve_hold_counter -= 1

        if self.intervening:
            speed_ok = True
            if safe_speed is not None and safe_speed > 0:
                speed_ok = current_speed < safe_speed * self.intervention_release_speed_margin

            heading_aligned = abs(heading_to_target) < self.heading_target_window_release
            centered = abs(lane_offset) < self.no_intervention_zone

            lane_confident = len(self.center_line_points) >= self.min_points_for_direction

            release_ready = centered and heading_aligned and speed_ok and lane_confident and not overspeed and self._curve_hold_counter == 0

            if release_ready:
                self._stable_release_counter += 1
            else:
                self._stable_release_counter = 0

            if self._stable_release_counter >= self.release_stable_frames:
                self.intervening = False
                self.intervention_strength = 0.0
                self._stable_release_counter = 0

    def _select_active_lane(self, current_lane, left_lane, center_lane, right_lane):
        """Choose which lane to operate in, falling back to last known and available edges."""
        if current_lane in ("LEFT", "RIGHT"):
            return current_lane
        if self.last_known_lane in ("LEFT", "RIGHT"):
            return self.last_known_lane
        if len(left_lane) > 0 and len(center_lane) > 0:
            return "LEFT"
        if len(center_lane) > 0 and len(right_lane) > 0:
            return "RIGHT"
        return None

    def _predict_curve_radius_from_lane(self, lane_left, lane_right):
        """Estimate upcoming curve radius directly from lane geometry."""
        n = min(len(lane_left), len(lane_right))
        if n < 5:
            return float('inf')

        center_points = [
            ((lane_left[i][0] + lane_right[i][0]) / 2.0,
             (lane_left[i][1] + lane_right[i][1]) / 2.0)
            for i in range(n)
        ]
        if len(center_points) < 5:
            return float('inf')

        points_x = np.array([p[0] for p in center_points])
        points_y = np.array([p[1] for p in center_points])

        dx_span = points_x[-1] - points_x[0]
        dy_span = points_y[-1] - points_y[0]

        try:
            if abs(dx_span) > abs(dy_span):
                coeffs = np.polyfit(points_x, points_y, 2)
                a, b, c = coeffs
                mid_x = np.mean(points_x)
                dydx = 2 * a * mid_x + b
                d2ydx2 = 2 * a
                curvature = abs(d2ydx2) / ((1 + dydx ** 2) ** 1.5)
            else:
                coeffs = np.polyfit(points_y, points_x, 2)
                a, b, c = coeffs
                mid_y = np.mean(points_y)
                dxdy = 2 * a * mid_y + b
                d2xdy2 = 2 * a
                curvature = abs(d2xdy2) / ((1 + dxdy ** 2) ** 1.5)

            if curvature > 1e-6:
                return 1.0 / curvature
            return float('inf')
        except Exception:
            if len(center_points) < 3:
                return float('inf')
            idx1 = max(1, len(center_points) // 4)
            idx2 = max(2, len(center_points) // 2)
            idx3 = max(3, (3 * len(center_points)) // 4)
            idx3 = min(idx3, len(center_points) - 1)
            idx2 = min(idx2, idx3 - 1)
            idx1 = min(idx1, idx2 - 1)
            return self._circle_radius_from_3_points(
                center_points[idx1], center_points[idx2], center_points[idx3]
            )

    def _circle_radius_from_3_points(self, p1, p2, p3):
        """Radius of circle passing through three points (fallback for curvature)."""
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
            return float('inf')

        cx = (d * e - b * f) / g
        cy = (a * f - c * e) / g

        radius = np.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
        return max(radius, 1.0)

    def _create_virtual_center(self, lane_boundary, car_x, car_y, car_theta, offset_right, current_speed):
        """Controller2 virtual center with predictive polynomial offset."""
        if len(lane_boundary) < 3:
            return []

        points_x = np.array([p[0] for p in lane_boundary])
        points_y = np.array([p[1] for p in lane_boundary])

        try:
            a, b, c = np.polyfit(points_x, points_y, 2)
        except Exception:
            return [(p[0] + self.base_lane_offset * (-1 if offset_right else 1), p[1]) for p in lane_boundary]

        curvature = abs(a)
        offset_dist = self.base_lane_offset
        if curvature > self.sharp_turn_threshold:
            offset_dist = self.base_lane_offset * self.sharp_turn_multiplier

        base_lookahead = 2
        if curvature > 0.0002:
            lookahead_points = 0
        elif curvature > 0.0001:
            lookahead_points = 1
        else:
            lookahead_points = base_lookahead
            if current_speed > 15.0:
                lookahead_points = 3

        virtual_center = []
        for i, x_val in enumerate(points_x):
            if lookahead_points > 0:
                lookahead_idx = min(i + lookahead_points, len(points_x) - 1)
                x_lookahead = points_x[lookahead_idx]
                y_poly = a * x_lookahead**2 + b * x_lookahead + c
                tangent_slope = 2 * a * x_lookahead + b
                tangent_angle = np.arctan(tangent_slope)
            else:
                x_lookahead = x_val
                y_poly = a * x_val**2 + b * x_val + c
                tangent_slope = 2 * a * x_val + b
                tangent_angle = np.arctan(tangent_slope)

            offset_angle = tangent_angle + (np.pi / 2 if offset_right else -np.pi / 2)
            vx = x_val + offset_dist * np.cos(offset_angle)
            if lookahead_points > 0:
                vy = y_poly - (x_lookahead - x_val) * np.tan(tangent_angle) + offset_dist * np.sin(offset_angle)
            else:
                vy = y_poly + offset_dist * np.sin(offset_angle)
            virtual_center.append((vx, vy))

        return virtual_center
