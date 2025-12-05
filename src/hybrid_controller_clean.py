"""
Minimal Hybrid Lane Controller (cleaned)

Goals:
- Keep only the essentials for lane-following, speed control, warnings, and assist.
- Use the existing car/camera interfaces (camera.last_measurement, camera.current_lane, car.x/y/theta/velocity, car.max_steering_angle, car.max_velocity).
- Be readable and lightly commented without extra legacy switches.
"""

import numpy as np


class HybridLaneController:
    # Control modes
    MODE_MANUAL = 0
    MODE_WARNING = 1
    MODE_ASSIST = 2

    def __init__(self, car, camera):
        self.car = car
        self.camera = camera

        # Mode state
        self.mode = self.MODE_MANUAL
        self.intervening = False

        # Lane/steering params
        self.lane_width = 4.0
        self.min_points_for_direction = 4
        self.image_height_equivalent = 14.0
        self.heading_correction_weight = 0.25  # slightly stronger heading weight for stability
        self.rolling_median_window = 6  # more history to damp oscillations
        self.steering_lpf_alpha = 0.1  # stronger low-pass smoothing

        # Assist engagement deadbands (only help when drifting or pointing out)
        # GREEN ZONE SIZE: Increase these values to make the green zone bigger
        self.assist_offset_deadband = 1.2  # meters - lateral tolerance before assist engages
        self.assist_heading_deadband = 0.8  # radians - heading tolerance before assist engages
        self.steering_history = []
        self.last_steering = 0.0

        # Lookahead behavior
        self.lookahead_speed_scale = 0.5  # meters per m/s
        self.lookahead_min = 5.0
        self.lookahead_max = 22.0
        self.curve_lookahead_high = 0.30
        self.curve_lookahead_mid = 0.50
        self.curve_proxy_high = 0.002
        self.curve_proxy_mid = 0.001

        # Speed planning
        self.safe_speed_scale = 0.8  # global multiplier to make safe speed more conservative
        self.lateral_accel_limit = 0.2 * 9.81  # lower lateral g limit to slow down in curves
        self.curve_entry_threshold = 650.0  # higher threshold: treat gentler curves as curves
        self.comfort_margin = 0.8  # tighter comfort factor -> stronger brake target
        self.brake_threshold = 0.8  # trigger braking sooner relative to safe speed
        self.accel_threshold = 1.0
        self.fallback_curve_radius = 120.0  # conservative radius when curvature estimate is missing

        # Warning / intervention thresholds
        self.warning_lateral_offset = 1.2
        self.warning_speed_margin = 1.05
        self.intervention_lane_offset = 0.5  # less strict lateral trigger
        self.heading_target_window_enter = 0.18  # less strict heading trigger
        self.heading_target_window_release = 0.09
        self.speed_overshoot_margin = 1.08  # allow small overspeed without engage
        self.release_stable_frames = 5
        self.hazard_stable_frames = 3  # require more stable hazard frames

        # Internal caches
        self.center_line_points = []
        self.target_point = None
        self.target_direction = None
        self.safe_speed = None
        self._last_safe_speed = None
        self._last_curve_radius = None
        self._hazard_counter = 0
        self._stable_release_counter = 0
        self.warnings = {
            "lane_departure": False,
            "speed_too_high": False,
            "time_to_crossing": False,
            "assist_on": False,
            "assist_intervening": False,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_mode(self, mode):
        if mode in (self.MODE_MANUAL, self.MODE_WARNING, self.MODE_ASSIST):
            self.mode = mode
            self.warnings = {k: False for k in self.warnings}
            self.intervening = False
            print(f"Hybrid Controller Mode: {['MANUAL','WARNING','ASSIST'][mode]}")
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

        speed = abs(self.car.velocity)
        steering_cmd = self._calculate_steering_direction(lane_left, lane_right, speed)
        speed_cmd = self._calculate_speed_control(lane_left, lane_right)
        lane_offset = self._estimate_lane_offset()
        heading_error = 0.0
        if self.target_direction is not None:
            heading_error = (self.target_direction - self.car.theta + np.pi) % (2 * np.pi) - np.pi

        # Always compute speed warnings first
        self._update_warnings(steering_cmd, speed_cmd, lane_offset, speed_only=True)

        # Deadband check: centered and pointed straight (applies to both modes)
        centered = abs(lane_offset) < self.assist_offset_deadband and abs(heading_error) < self.assist_heading_deadband
        
        if self.mode == self.MODE_WARNING:
            if not centered:
                # Only show lane/time warnings when outside green zone
                self._update_warnings(steering_cmd, speed_cmd, lane_offset, speed_only=False)
            else:
                # In green zone: keep speed warnings, suppress lane/time
                self.warnings["lane_departure"] = False
                self.warnings["time_to_crossing"] = False
            self.intervening = False
            return None, None, None, self.warnings, False

        # ASSIST mode from here
        # Check if speed warning is active BEFORE checking centered zone
        if self.warnings.get("speed_too_high"):
            # Speed too high overrides green zone - must brake immediately
            centered = False  # Force out of green zone when speeding
        
        if centered:
            self.intervening = False
            # Keep speed warnings visible; suppress lane/time warnings only
            self.warnings["lane_departure"] = False
            self.warnings["time_to_crossing"] = False
            self.warnings["assist_on"] = False
            self.warnings["assist_intervening"] = False
            return None, None, None, self.warnings, False

        # Outside green zone: compute all warnings and intervention
        self._update_warnings(steering_cmd, speed_cmd, lane_offset, speed_only=False)
        self._update_intervention_state(lane_offset, speed_cmd)

        # CRITICAL: In assist mode, ANY warning means the computer takes full control
        # Check if any warning is active (speed, lane departure, or time to crossing)
        any_warning_active = (
            self.warnings.get("speed_too_high") 
            or self.warnings.get("lane_departure")
            or self.warnings.get("time_to_crossing")
        )

        # If any warning is active, force intervention and take control
        if any_warning_active:
            self.intervening = True

        # Initialize commands
        throttle_cmd = None
        brake_cmd = None

        # Apply control when intervening
        if self.intervening:
            # Speed-based braking: if speed warning OR brake action requested
            if self.warnings.get("speed_too_high") or (speed_cmd and speed_cmd.get("action") == "brake"):
                if self.safe_speed and self.safe_speed > 0:
                    # Use warning margin for braking calculation
                    target_speed = self.safe_speed * self.warning_speed_margin
                    if speed > target_speed:
                        overspeed_brake = (speed - self.safe_speed * self.comfort_margin) / (self.safe_speed + 1e-3)
                        brake_cmd = max(brake_cmd or 0.0, np.clip(overspeed_brake, 0.0, 1.0))
                
                # Also apply speed command brake if available
                if speed_cmd and speed_cmd.get("action") == "brake":
                    brake_cmd = max(brake_cmd or 0.0, speed_cmd.get("brake", 0.0))
                
                # Only cut throttle when speed warning is active (not for lane departure)
                throttle_cmd = 0.0
            elif self.warnings.get("lane_departure") or self.warnings.get("time_to_crossing"):
                # For lane departure: only apply steering correction, let driver control speed
                throttle_cmd = None
                brake_cmd = None
        else:
            # Not intervening: allow acceleration if speed planner suggests it
            if speed_cmd and speed_cmd.get("action") == "accelerate":
                throttle_cmd = speed_cmd.get("throttle", 0.0)
            else:
                throttle_cmd = None
            brake_cmd = None

        # HUD flags
        self.warnings["assist_on"] = self.mode == self.MODE_ASSIST
        self.warnings["assist_intervening"] = self.intervening

        return steering_cmd, throttle_cmd, brake_cmd, self.warnings, self.intervening

    # ------------------------------------------------------------------
    # Steering
    # ------------------------------------------------------------------
    def _calculate_steering_direction(self, lane_left, lane_right, current_speed):
        # Robust, minimal pure-pursuit steering (mirrors lanekeeping style)
        car_x, car_y, car_theta = self.car.x, self.car.y, self.car.theta

        has_left = len(lane_left) >= self.min_points_for_direction
        has_right = len(lane_right) >= self.min_points_for_direction
        if not has_left and not has_right:
            return self.last_steering

        # Build centerline safely
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
            return self.last_steering

        # Simple smoothing (3-point average) to avoid sharp kinks
        smoothed = []
        for i in range(len(center_line)):
            start = max(0, i - 1)
            end = min(len(center_line), i + 2)
            xs = [center_line[j][0] for j in range(start, end)]
            ys = [center_line[j][1] for j in range(start, end)]
            smoothed.append((np.mean(xs), np.mean(ys)))

        # Keep latest centerline for offset and speed curvature fallback
        self.center_line_points = smoothed

        # Adaptive lookahead by curvature proxy (triangle area over baseline^3)
        curvature_proxy = 0.0
        if len(smoothed) >= 3:
            p1, p2, p3 = smoothed[0], smoothed[len(smoothed)//2], smoothed[-1]
            area = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
            base = np.hypot(p3[0]-p1[0], p3[1]-p1[1]) + 1e-3
            curvature_proxy = area / (base ** 3)

        lookahead = np.clip(current_speed * self.lookahead_speed_scale, self.lookahead_min, self.lookahead_max)
        if curvature_proxy > self.curve_proxy_high:
            lookahead = max(4.0, lookahead * self.curve_lookahead_high)
        elif curvature_proxy > self.curve_proxy_mid:
            lookahead = max(5.0, lookahead * self.curve_lookahead_mid)

        # Select a target along arc length starting from the nearest forward point
        forward_x, forward_y = np.cos(car_theta), np.sin(car_theta)
        perp_x, perp_y = -np.sin(car_theta), np.cos(car_theta)

        forward_dists = []
        for pt in smoothed:
            dx, dy = pt[0] - car_x, pt[1] - car_y
            forward_dists.append(dx * forward_x + dy * forward_y)

        forward_indices = [i for i, fd in enumerate(forward_dists) if fd > 0.1]
        if not forward_indices:
            return self.last_steering

        start_idx = min(forward_indices, key=lambda i: forward_dists[i])
        target = smoothed[start_idx]
        remaining = lookahead
        for j in range(start_idx, len(smoothed) - 1):
            p_curr, p_next = smoothed[j], smoothed[j + 1]
            seg_dx, seg_dy = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            seg_len = np.hypot(seg_dx, seg_dy)
            if seg_len < 1e-3:
                continue
            if remaining <= seg_len:
                t = remaining / seg_len
                target = (p_curr[0] + seg_dx * t, p_curr[1] + seg_dy * t)
                break
            remaining -= seg_len
        else:
            target = smoothed[-1]

        dx, dy = target[0] - car_x, target[1] - car_y
        lateral_error = dx * perp_x + dy * perp_y
        heading_to_target = np.arctan2(dy, dx) - car_theta
        heading_to_target = (heading_to_target + np.pi) % (2 * np.pi) - np.pi

        self.target_point = target
        self.target_direction = np.arctan2(target[1] - car_y, target[0] - car_x)

        combined_error = lateral_error + heading_to_target * self.image_height_equivalent * self.heading_correction_weight
        raw_steering_deg = 90.0 - np.degrees(np.arctan2(self.image_height_equivalent, combined_error))
        steering_angle = np.clip(np.radians(raw_steering_deg), -self.car.max_steering_angle, self.car.max_steering_angle)

        # Median + low-pass
        self.steering_history.insert(0, steering_angle)
        if len(self.steering_history) > self.rolling_median_window:
            self.steering_history.pop()
        steering_med = np.median(self.steering_history)
        steering_filtered = self.steering_lpf_alpha * steering_med + (1 - self.steering_lpf_alpha) * self.last_steering

        self.last_steering = steering_filtered
        return steering_filtered

    # ------------------------------------------------------------------
    # Speed control
    # ------------------------------------------------------------------
    def _calculate_speed_control(self, lane_left, lane_right):
        curve_radius = self._predict_curve_radius_from_lane(lane_left, lane_right)
        if curve_radius is None or not np.isfinite(curve_radius):
            curve_radius = self._curve_radius_from_centerline()
        if curve_radius is None or not np.isfinite(curve_radius):
            curve_radius = self.fallback_curve_radius

        self._last_curve_radius = curve_radius
        safe_speed = np.sqrt(self.lateral_accel_limit * max(curve_radius, 1e-3))
        safe_speed = min(safe_speed * self.safe_speed_scale, self.car.max_velocity)
        self.safe_speed = safe_speed
        self._last_safe_speed = safe_speed

        current_speed = abs(self.car.velocity)
        if curve_radius < self.curve_entry_threshold and current_speed > safe_speed * self.brake_threshold:
            brake_cmd = np.clip((current_speed - safe_speed * self.comfort_margin) / (safe_speed + 1e-3), 0.0, 1.0)
            return {"action": "brake", "brake": brake_cmd, "throttle": 0.0, "safe_speed": safe_speed}

        if current_speed < safe_speed * self.comfort_margin * self.accel_threshold:
            throttle_cmd = np.clip((safe_speed - current_speed) / safe_speed, 0.0, 1.0)
            return {"action": "accelerate", "throttle": throttle_cmd, "brake": 0.0, "safe_speed": safe_speed}

        return {"action": "hold", "throttle": 0.0, "brake": 0.0, "safe_speed": safe_speed}

    def _conservative_speed_command_from_cache(self):
        if self._last_safe_speed is None:
            return None
        safe_speed = self._last_safe_speed
        current_speed = abs(self.car.velocity)
        if current_speed > safe_speed * self.brake_threshold:
            brake_cmd = np.clip((current_speed - safe_speed * self.comfort_margin) / (safe_speed + 1e-3), 0.0, 1.0)
            return {"action": "brake", "brake": brake_cmd, "throttle": 0.0, "safe_speed": safe_speed}
        return {"action": "hold", "throttle": 0.0, "brake": 0.0, "safe_speed": safe_speed}

    # ------------------------------------------------------------------
    # Warnings / intervention
    # ------------------------------------------------------------------
    def _update_warnings(self, steering_command, speed_command, lane_offset=None, speed_only=False):
        if lane_offset is None:
            lane_offset = self._estimate_lane_offset()

        safe_speed = speed_command.get("safe_speed") if speed_command else self.safe_speed
        speed_too_high = False
        if safe_speed and safe_speed > 0:
            if abs(self.car.velocity) > safe_speed * self.warning_speed_margin:
                speed_too_high = True

        lane_departure = False
        time_to_crossing = False
        if not speed_only and steering_command is not None:
            lateral_speed = abs(self.car.velocity) * abs(np.tan(steering_command))
            if lateral_speed > 1e-3:
                ttc = abs(lane_offset) / lateral_speed
                time_to_crossing = ttc < 1.0
            lane_departure = abs(lane_offset) > self.warning_lateral_offset or time_to_crossing

        self.warnings["speed_too_high"] = speed_too_high
        self.warnings["lane_departure"] = lane_departure
        self.warnings["time_to_crossing"] = time_to_crossing

    def _update_intervention_state(self, lane_offset, speed_command):
        safe_speed = speed_command.get("safe_speed") if speed_command else self.safe_speed
        current_speed = abs(self.car.velocity)
        overspeed = safe_speed and safe_speed > 0 and current_speed > safe_speed * self.speed_overshoot_margin

        if speed_command and speed_command.get("action") == "brake":
            # Immediate engage for braking recommendation from speed planner
            self.intervening = True
            self._hazard_counter = 0
            self._stable_release_counter = 0
            return

        heading_to_target = 0.0
        if self.target_direction is not None:
            heading_to_target = self.target_direction - self.car.theta
            heading_to_target = (heading_to_target + np.pi) % (2 * np.pi) - np.pi

        hazard = (
            abs(lane_offset) > self.intervention_lane_offset
            or abs(heading_to_target) > self.heading_target_window_enter
            or overspeed
            or (speed_command and speed_command.get("action") == "brake")
            or self.warnings.get("speed_too_high")
            or self.warnings.get("lane_departure")
        )

        if hazard:
            self._hazard_counter += 1
            if self._hazard_counter >= self.hazard_stable_frames:
                self.intervening = True
                self._stable_release_counter = 0
            return

        self._hazard_counter = 0
        if self.intervening:
            heading_ok = abs(heading_to_target) < self.heading_target_window_release
            centered = abs(lane_offset) < self.intervention_lane_offset / 2
            speed_ok = not overspeed
            if heading_ok and centered and speed_ok:
                self._stable_release_counter += 1
                if self._stable_release_counter >= self.release_stable_frames:
                    self.intervening = False
                    self._stable_release_counter = 0
            else:
                self._stable_release_counter = 0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _estimate_lane_offset(self):
        if not self.center_line_points:
            return 0.0
        car_x, car_y, car_theta = self.car.x, self.car.y, self.car.theta
        best = None
        best_dist = float('inf')
        for cx, cy in self.center_line_points:
            dx, dy = cx - car_x, cy - car_y
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
        return dx * (-np.sin(car_theta)) + dy * np.cos(car_theta)

    def _select_active_lane(self, current_lane, left_lane, center_lane, right_lane):
        if current_lane in ("LEFT", "RIGHT"):
            return current_lane
        if len(left_lane) > 0 and len(center_lane) > 0:
            return "LEFT"
        if len(center_lane) > 0 and len(right_lane) > 0:
            return "RIGHT"
        return None

    def _predict_curve_radius_from_lane(self, lane_left, lane_right):
        n = min(len(lane_left), len(lane_right))
        if n >= 5:
            center_points = [
                ((lane_left[i][0] + lane_right[i][0]) / 2.0,
                 (lane_left[i][1] + lane_right[i][1]) / 2.0)
                for i in range(n)
            ]
            radius = self._curve_radius_from_points(center_points)
            if np.isfinite(radius):
                return radius

        # Fallback: use stored centerline from steering if lane boundaries are incomplete
        return self._curve_radius_from_centerline()

    def _curve_radius_from_centerline(self):
        if not self.center_line_points or len(self.center_line_points) < 5:
            return float('inf')
        return self._curve_radius_from_points(self.center_line_points)

    def _curve_radius_from_points(self, points):
        if len(points) < 5:
            return float('inf')
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        dx_span = xs[-1] - xs[0]
        dy_span = ys[-1] - ys[0]
        try:
            if abs(dx_span) > abs(dy_span):
                a, b, c = np.polyfit(xs, ys, 2)
                mid_x = np.mean(xs)
                dydx = 2 * a * mid_x + b
                d2ydx2 = 2 * a
                curvature = abs(d2ydx2) / ((1 + dydx ** 2) ** 1.5)
            else:
                a, b, c = np.polyfit(ys, xs, 2)
                mid_y = np.mean(ys)
                dxdy = 2 * a * mid_y + b
                d2xdy2 = 2 * a
                curvature = abs(d2xdy2) / ((1 + dxdy ** 2) ** 1.5)
            return float('inf') if curvature <= 1e-6 else 1.0 / curvature
        except Exception:
            if len(points) < 3:
                return float('inf')
            return self._circle_radius_from_3_points(
                points[1], points[len(points)//2], points[-2]
            )

    def _circle_radius_from_3_points(self, p1, p2, p3):
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
        return max(np.hypot(x1 - cx, y1 - cy), 1.0)

    def _create_virtual_center(self, lane_boundary, offset_right):
        # Shift boundary inward by half lane width to synthesize center when only one edge is visible.
        if len(lane_boundary) < 3:
            return []
        xs = np.array([p[0] for p in lane_boundary])
        ys = np.array([p[1] for p in lane_boundary])
        try:
            a, b, c = np.polyfit(xs, ys, 2)
        except Exception:
            # Fallback: constant offset
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
