"""
Hybrid Controller Module - Direction-based LKA + Predictive Speed Control
Part of the 3D Robotics Lab simulation.

Combines:
- Direction-following LKA (tangent-based, single-line capable)
- MPC-based speed prediction (curve detection, anticipatory braking)
- 3 control modes: Manual, Warning, Assist
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.stats import norm as scipy_norm


class HybridLaneController:
    """
    Hybrid Lane Keeping and Speed Control System
    
    Features:
    - Direction-based steering (follows lane tangent)
    - Single-line operation (works with only one boundary visible)
    - Predictive curve detection (looks ahead 2.7 seconds)
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
        
        # Current mode
        self.mode = self.MODE_MANUAL
        
        # Debug flag
        self.debug = True  # Set to False to disable debug output
        
        # ================================================================
        # DIRECTION FOLLOWING PARAMETERS (LKA)
        # ================================================================
        self.lane_width = 4.0  # meters (actual lane width)
        self.min_points_for_direction = 4  # minimum points to trust direction
        
        # Speed-adaptive lookahead (prevents short lookahead oscillation)
        self.min_lookahead_distance = 15.0  # meters - minimum at low speeds
        self.max_lookahead_distance = 30.0  # meters - maximum at high speeds
        self.lookahead_speed_factor = 0.5  # seconds - lookahead = speed * factor (time-based)
        
        # Rolling MEDIAN smoothing (BFMC professional approach - uses median not average)
        self.steering_history = []  # Store last N steering angles
        self.rolling_median_window = 3  # Increased to 3 for better high-speed stability
        self.last_steering = 0.0  # Fallback when lanes lost (BFMC approach)
        
        # Adaptive lane width for sharp turns (BFMC technique)
        self.base_lane_offset = self.lane_width / 2.0  # Base offset for virtual center
        self.sharp_turn_threshold = 0.0001  # Curvature threshold for sharp turn detection
        self.sharp_turn_multiplier = 1.3  # Widen virtual lane by 30% in sharp turns
        
        # Image-height-equivalent reference (BFMC uses image height in pixels ~720)
        # Balanced at 14m for good curve entry without over-aggressiveness
        self.image_height_equivalent = 14.0  # meters - balanced for stability
        
        # Heading error correction (fixes post-curve oscillation)
        self.heading_correction_weight = 0.4  # Weight for heading error term
        
        # ================================================================
        # SPEED PREDICTION PARAMETERS (MPC-based)
        # ================================================================
        self.prediction_horizon = 25  # steps (increased to see further ahead)
        self.prediction_dt = 0.18  # seconds per step
        self.lateral_accel_limit = 0.3 * 9.81  # 0.5g safe limit (increased from 0.3g to prevent slip)
        
        # Speed control
        self.comfort_margin = 0.85  # target 85% of max safe speed (more conservative for safety)
        self.brake_threshold = 0.85  # start braking at 85% of safe speed (much earlier!)
        self.max_comfort_decel = 1.0 * 9.81  # 1.0g maximum braking (full brake capability)
        self.accel_threshold = 1  # accelerate only if below 75% of target (wait longer before accelerating)
        
        # Curve radius estimation (running average)
        self.radius_history = []
        self.radius_history_size = 3
        
        # Track minimum safe speed in current curve
        self.in_curve = False
        self.curve_min_safe_speed = float('inf')
        self.curve_entry_threshold = 500  # meters - radius below this means we're in a curve
        
        # ================================================================
        # INTERVENTION PARAMETERS (Mode 3: Assist)
        # ================================================================
        # Lateral intervention zones
        self.no_intervention_zone = 0.5  # meters from center
        self.gentle_intervention_zone = 1.5  # meters from center
        # Beyond gentle zone = strong intervention
        
        # ================================================================
        # WARNING SYSTEM (Mode 2: Warning)
        # ================================================================
        self.warnings = {
            'lane_departure': False,
            'speed_too_high': False,
            'time_to_crossing': False
        }
        self.warning_thresholds = {
            'lateral_offset_warn': 1.0,  # meters
            'time_to_crossing_warn': 1.0,  # seconds
            'speed_margin_warn': 1.15  # 15% over safe speed
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
        Returns: (steering_angle, throttle, brake, warnings)
        """
        if self.mode == self.MODE_MANUAL:
            # No assistance
            return None, None, None, {}
        
        # Get lane detection from camera
        left_lane, center_lane, right_lane = self.camera.last_measurement
        current_lane = self.camera.current_lane
        
        # Determine lane boundaries based on which lane we're in
        if current_lane == "LEFT":
            lane_left = left_lane
            lane_right = center_lane
        elif current_lane == "RIGHT":
            lane_left = center_lane
            lane_right = right_lane
        else:
            # Unknown lane
            return None, None, None, {}
        
        # ============================================================
        # STEERING CONTROL (Direction Following)
        # ============================================================
        current_speed = abs(self.car.velocity)
        steering_command = self._calculate_steering_direction(
            lane_left, lane_right, track, current_speed
        )
        
        # ============================================================
        # SPEED CONTROL (Curve Prediction)
        # ============================================================
        speed_command = self._calculate_speed_control(
            lane_left, lane_right, track
        )
        
        if self.debug and speed_command:
            # print(f"[HYBRID CALC] Speed command: {speed_command['action']}, Brake={speed_command.get('brake', 0):.2f}, Throttle={speed_command.get('throttle', 0):.2f}, Radius={speed_command.get('curve_radius', 0):.1f}m")
            pass
        
        # ============================================================
        # MODE-SPECIFIC BEHAVIOR
        # ============================================================
        if self.mode == self.MODE_WARNING:
            # Warning mode: monitor only, no control
            self._update_warnings(steering_command, speed_command)
            return None, None, None, self.warnings
        
        elif self.mode == self.MODE_ASSIST:
            # Assist mode: blend with manual control
            steering, throttle, brake = self._apply_blended_control(
                steering_command, speed_command
            )
            if self.debug:
                steering_str = f"{steering:.3f}" if steering is not None else "None"
                throttle_str = f"{throttle:.3f}" if throttle is not None else "None"
                brake_str = f"{brake:.3f}" if brake is not None else "None"
                # print(f"[HYBRID OUTPUT] Returning: Steering={steering_str}, Throttle={throttle_str}, Brake={brake_str}")
            self._update_warnings(steering_command, speed_command)
            return steering, throttle, brake, self.warnings
        
        return None, None, None, {}
    
    def _calculate_steering_direction(self, lane_left, lane_right, track, current_speed):
        """
        Calculate steering using weighted error approach (inspired by professional LKA systems)
        Uses weighted average of lateral errors along the path
        """
        car_x, car_y, car_theta = self.car.x, self.car.y, self.car.theta
        
        # Check what we have visible
        has_left = len(lane_left) >= self.min_points_for_direction
        has_right = len(lane_right) >= self.min_points_for_direction
        
        if self.debug:
            print(f"\n[LKA DEBUG] === Boundary Detection ===")
            print(f"[LKA DEBUG] Left boundary points: {len(lane_left)} (has_left: {has_left})")
            print(f"[LKA DEBUG] Right boundary points: {len(lane_right)} (has_right: {has_right})")
        
        if not has_left and not has_right:
            if self.debug:
                print(f"[LKA DEBUG] No boundaries detected! Using last steering: {np.degrees(self.last_steering):.2f}°")
            return self.last_steering  # BFMC: Keep last good steering when lanes lost
        
        # Build center line from available boundaries
        # Camera already provides densely interpolated boundaries (2.0m spacing)
        if has_left and has_right:
            # Both boundaries: create PREDICTIVE center line using polynomial
            n = min(len(lane_left), len(lane_right))
            
            # Simple average as baseline
            center_avg = [
                ((lane_left[i][0] + lane_right[i][0]) / 2.0,
                 (lane_left[i][1] + lane_right[i][1]) / 2.0)
                for i in range(n)
            ]
            
            # Fit polynomial to averaged center for predictive path
            if len(center_avg) >= 3:
                try:
                    points_x = np.array([p[0] for p in center_avg])
                    points_y = np.array([p[1] for p in center_avg])
                    coeffs = np.polyfit(points_x, points_y, 2)
                    a, b, c = coeffs
                    
                    # ADAPTIVE PREDICTIVE LOOKAHEAD based on curvature and speed
                    curvature = abs(a)
                    
                    # Base lookahead: 2 points (4m)
                    # Sharp curves: reduce to 0-1 points to avoid cutting corners
                    # High speed: increase to 3 points for stability
                    base_lookahead = 2
                    
                    # Reduce lookahead in sharp curves (high curvature)
                    if curvature > 0.0002:  # Very sharp curve
                        lookahead_points = 0  # No prediction, follow exactly
                    elif curvature > 0.0001:  # Sharp curve
                        lookahead_points = 1  # Minimal prediction
                    else:  # Gentle curve or straight
                        lookahead_points = base_lookahead
                        # Increase at high speed for stability
                        if current_speed > 15.0:  # > 54 km/h
                            lookahead_points = 3
                    
                    # Create predictive center with adaptive lookahead
                    self.center_line_points = []
                    for i, x_val in enumerate(points_x):
                        if lookahead_points > 0:
                            lookahead_idx = min(i + lookahead_points, len(points_x) - 1)
                            x_lookahead = points_x[lookahead_idx]
                            
                            # Polynomial at lookahead
                            y_poly = a * x_lookahead**2 + b * x_lookahead + c
                            tangent_slope = 2 * a * x_lookahead + b
                            
                            # Project back to current position with lookahead geometry
                            y_current = y_poly - (x_lookahead - x_val) * tangent_slope
                            self.center_line_points.append((x_val, y_current))
                        else:
                            # No prediction in very sharp curves - use exact center
                            self.center_line_points.append(center_avg[i])
                    
                    if self.debug:
                        print(f"[LKA DEBUG] BOTH boundaries: curvature={curvature:.6f}, lookahead={lookahead_points} points, speed={current_speed*3.6:.1f}km/h")
                except:
                    # Fallback to simple average if polynomial fails
                    self.center_line_points = center_avg
                    if self.debug:
                        print(f"[LKA DEBUG] Using BOTH boundaries (simple average, polynomial failed) - {n} points")
            else:
                self.center_line_points = center_avg
                if self.debug:
                    print(f"[LKA DEBUG] Using BOTH boundaries (simple average, too few points) - {n} points")
        elif has_left:
            # Only left boundary: create virtual center
            self.center_line_points = self._create_virtual_center(
                lane_left, car_x, car_y, car_theta, offset_right=True, current_speed=current_speed
            )
            if self.debug:
                print(f"[LKA DEBUG] Using LEFT boundary only - {len(lane_left)} points (camera provides 2.0m spacing)")
        else:
            # Only right boundary: create virtual center
            self.center_line_points = self._create_virtual_center(
                lane_right, car_x, car_y, car_theta, offset_right=False, current_speed=current_speed
            )
            if self.debug:
                print(f"[LKA DEBUG] Using RIGHT boundary only - {len(lane_right)} points (camera provides 2.0m spacing)")
        
        if self.debug and len(self.center_line_points) > 0:
            first_pt = self.center_line_points[0]
            last_pt = self.center_line_points[-1]
            print(f"[LKA DEBUG] Final center line: {len(self.center_line_points)} points (2.0m spacing)")
            print(f"[LKA DEBUG] Center line - first: ({first_pt[0]:.2f}, {first_pt[1]:.2f}), last: ({last_pt[0]:.2f}, {last_pt[1]:.2f})")
        
        if len(self.center_line_points) < 2:
            if self.debug:
                print(f"[LKA DEBUG] Not enough center points: {len(self.center_line_points)}")
            return None
        
        # Calculate weighted lateral error (professional approach adapted for 3D)
        # Key insight: Weight by FORWARD DISTANCE, not total distance
        # Closer points (in front of car) get higher weight
        lateral_errors = []
        forward_distances = []
        
        # Calculate car's forward and perpendicular directions
        forward_x = np.cos(car_theta)  # forward direction
        forward_y = np.sin(car_theta)
        perp_x = -np.sin(car_theta)    # perpendicular to heading (positive = left)
        perp_y = np.cos(car_theta)
        
        if self.debug:
            print(f"\n[LKA DEBUG] ========== STEERING CALCULATION ==========")
            print(f"[LKA DEBUG] Car position: ({car_x:.2f}, {car_y:.2f}), heading: {np.degrees(car_theta):.1f}°")
            print(f"[LKA DEBUG] Forward vector: ({forward_x:.3f}, {forward_y:.3f})")
            print(f"[LKA DEBUG] Perpendicular vector: ({perp_x:.3f}, {perp_y:.3f})")
            print(f"[LKA DEBUG] Total center line points: {len(self.center_line_points)}")
        
        points_behind = 0
        for i, (px, py) in enumerate(self.center_line_points):
            # Vector from car to path point
            dx = px - car_x
            dy = py - car_y
            
            # Project onto car's forward axis (how far ahead this point is)
            forward_dist = dx * forward_x + dy * forward_y
            
            # Only consider points ahead of the car
            if forward_dist < 0.1:  # Skip points behind or at car
                points_behind += 1
                continue
            
            # Project onto car's perpendicular axis (lateral error)
            # NOTE: We want NEGATIVE when path is to the right (we should steer right)
            # and POSITIVE when path is to the left (we should steer left)
            # The perpendicular vector points LEFT, so dx*perp_x + dy*perp_y gives:
            # POSITIVE when point is to the LEFT, NEGATIVE when point is to the RIGHT
            lateral_error = dx * perp_x + dy * perp_y
            
            if self.debug and i < 3:  # Show first 3 points
                print(f"[LKA DEBUG]   Point {i}: ({px:.2f}, {py:.2f}), forward: {forward_dist:.2f}m, lateral: {lateral_error:.2f}m")
            
            lateral_errors.append(lateral_error)
            forward_distances.append(forward_dist)
        
        if self.debug and points_behind > 0:
            print(f"[LKA DEBUG] Points behind car (skipped): {points_behind}")
            print(f"[LKA DEBUG] Points ahead of car: {len(lateral_errors)}")
        
        if len(lateral_errors) == 0:
            if self.debug:
                print(f"[LKA DEBUG] No points ahead of car! Using last steering: {np.degrees(self.last_steering):.2f}°")
            return self.last_steering  # BFMC: Keep last good steering
        
        # Weight by inverse forward distance (professional approach)
        # Closer points (smaller forward distance) get higher weight
        # Using survival function approach from professional code
        forward_distances = np.array(forward_distances)
        lateral_errors = np.array(lateral_errors)
        
        # SPEED-ADAPTIVE LOOKAHEAD: Prevents oscillation from too-short lookahead at high speeds
        # At low speeds: shorter lookahead for tight maneuvers
        # At high speeds: longer lookahead for smooth, stable control
        current_speed = abs(self.car.velocity)
        speed_based_lookahead = current_speed * self.lookahead_speed_factor
        max_lookahead = np.clip(
            speed_based_lookahead,
            self.min_lookahead_distance,
            self.max_lookahead_distance
        )
        close_enough_mask = forward_distances <= max_lookahead
        
        if np.sum(close_enough_mask) > 0:
            # Filter to only use close points
            forward_distances = forward_distances[close_enough_mask]
            lateral_errors = lateral_errors[close_enough_mask]
        
        if self.debug:
            print(f"[LKA DEBUG] Speed: {current_speed*3.6:.1f} km/h, Adaptive lookahead: {max_lookahead:.2f}m")
            print(f"[LKA DEBUG] Forward distances - min: {forward_distances.min():.2f}m, max: {forward_distances.max():.2f}m, avg: {forward_distances.mean():.2f}m")
            print(f"[LKA DEBUG] Lateral errors - min: {lateral_errors.min():.2f}m, max: {lateral_errors.max():.2f}m, avg: {lateral_errors.mean():.2f}m")
            print(f"[LKA DEBUG] Using {len(forward_distances)} points within {max_lookahead:.2f}m lookahead")
        
        # Normalize forward distances to [0, 1] range
        max_dist = np.max(forward_distances)
        min_dist = np.min(forward_distances)
        if max_dist - min_dist > 0.1:
            norm_dists = (forward_distances - min_dist) / (max_dist - min_dist)
        else:
            norm_dists = np.ones_like(forward_distances) * 0.5
        
        # Apply survival function weighting (BFMC weights by SLICE INDEX not distance)
        # Balanced weighting for 14m image height
        mu = 0.3  # Slight focus on near-mid range
        sigma = 0.35  # Moderate transition
        
        cdf = scipy_norm.cdf(norm_dists, mu, sigma)
        sf = 1 - cdf  # Survival function
        sf_normalized = (sf - sf.min()) / (sf.max() - sf.min() + 1e-6)
        weights = sf_normalized + 0.1  # Add small constant to avoid zero weights
        
        # Calculate weighted average error (lateral)
        weighted_error = np.average(lateral_errors, weights=weights)
        
        # HEADING ERROR CORRECTION: Calculate path tangent to fix post-curve oscillation
        # This helps the car align its heading with the path, not just position
        heading_error = 0.0
        heading_error_contribution = 0.0
        if len(self.center_line_points) >= 3:
            # Use points at 1/3 distance for tangent calculation (stable lookahead)
            tangent_idx = min(len(forward_distances) // 3, len(self.center_line_points) - 2)
            if tangent_idx >= 1:
                # Get world coordinates of tangent points
                p1 = self.center_line_points[tangent_idx - 1]
                p2 = self.center_line_points[tangent_idx + 1]
                
                # Path tangent direction in world frame
                path_dx = p2[0] - p1[0]
                path_dy = p2[1] - p1[1]
                path_heading = np.arctan2(path_dy, path_dx)
                
                # Car heading in world frame
                car_heading = car_theta
                
                # Heading error (normalized to [-pi, pi])
                heading_error = path_heading - car_heading
                while heading_error > np.pi:
                    heading_error -= 2 * np.pi
                while heading_error < -np.pi:
                    heading_error += 2 * np.pi
                
                # Convert heading error to lateral error equivalent
                # Small heading errors become position errors further ahead
                heading_error_contribution = heading_error * self.image_height_equivalent * self.heading_correction_weight
                
                if self.debug:
                    print(f"[LKA DEBUG] Path heading: {np.degrees(path_heading):.1f}°, Car heading: {np.degrees(car_heading):.1f}°")
                    print(f"[LKA DEBUG] Heading error: {np.degrees(heading_error):.2f}°, contribution: {heading_error_contribution:.3f}m")
        
        # Combine lateral error with heading correction
        combined_error = weighted_error + heading_error_contribution
        
        # BFMC Professional steering formula: angle = 90 - atan2(image_height, error)
        # CRITICAL: BFMC uses IMAGE HEIGHT (~720 pixels) not variable lookahead
        # We use meters equivalent of typical image height viewing distance
        avg_forward_dist = np.average(forward_distances, weights=weights)
        
        # Use combined error (lateral + heading correction)
        raw_steering_degrees = 90.0 - np.degrees(np.arctan2(self.image_height_equivalent, combined_error))
        raw_steering = np.radians(raw_steering_degrees)
        
        if self.debug:
            print(f"[LKA DEBUG] Lateral error: {weighted_error:.3f}m, Combined error: {combined_error:.3f}m")
            print(f"[LKA DEBUG] Avg forward distance: {avg_forward_dist:.2f}m")
            print(f"[LKA DEBUG] Image height equiv: {self.image_height_equivalent:.1f}m")
            print(f"[LKA DEBUG] Raw steering angle: {np.degrees(raw_steering):.2f}° (BFMC formula)")
            print(f"[LKA DEBUG] Weight distribution - min: {weights.min():.3f}, max: {weights.max():.3f}")
        
        # Store target point for visualization (use FIXED LOOKAHEAD, not closest!)
        if len(self.center_line_points) > 0:
            # Use a fixed lookahead distance to avoid oscillation
            # Find the point closest to the lookahead distance (e.g., 5-6 meters ahead)
            target_lookahead = 6.0  # meters - stable lookahead distance
            
            best_point = self.center_line_points[0]
            best_diff = float('inf')
            
            for point in self.center_line_points:
                dx = point[0] - car_x
                dy = point[1] - car_y
                forward_dist = dx * forward_x + dy * forward_y
                
                # Only consider points ahead
                if forward_dist > 0.1:
                    # Find point closest to target lookahead distance
                    diff = abs(forward_dist - target_lookahead)
                    if diff < best_diff:
                        best_diff = diff
                        best_point = point
            
            self.target_point = best_point
            self.target_direction = np.arctan2(
                self.target_point[1] - car_y,
                self.target_point[0] - car_x
            )
        
        # SIMPLIFIED FILTERING (BFMC approach - no excessive filtering)
        # Stage 1: Clip raw steering to valid range
        steering_angle = np.clip(
            raw_steering,
            -self.car.max_steering_angle,
            self.car.max_steering_angle
        )
        
        # Stage 2: ROLLING MEDIAN (BFMC uses MEDIAN not average for robustness)
        self.steering_history.insert(0, steering_angle)
        if len(self.steering_history) > self.rolling_median_window:
            self.steering_history.pop()
        
        final_steering = np.median(self.steering_history)
        
        if self.debug:
            print(f"[LKA DEBUG] Clipped steering: {np.degrees(steering_angle):.2f}°")
            print(f"[LKA DEBUG] Rolling median ({len(self.steering_history)} frames): {np.degrees(final_steering):.2f}°")
            print(f"[LKA DEBUG] ==========================================\n")
        
        self.last_steering = final_steering  # Store for fallback (BFMC approach)
        return final_steering
    
    def _create_virtual_center(self, lane_boundary, car_x, car_y, car_theta, offset_right, current_speed):
        """Create virtual center line using BFMC's predictive approach
        Instead of offsetting each point perpendicular, fit polynomial and sample from it
        This creates a PREDICTIVE path that enters curves earlier
        Adaptive lookahead based on curvature and speed
        """
        if len(lane_boundary) < 3:
            return []  # Need at least 3 points to fit
        
        # Extract x,y coordinates
        points_x = np.array([p[0] for p in lane_boundary])
        points_y = np.array([p[1] for p in lane_boundary])
        
        # Fit 2nd order polynomial to the boundary
        try:
            coeffs = np.polyfit(points_x, points_y, 2)
            a, b, c = coeffs
        except:
            # Fallback: simple offset if fit fails
            return [(p[0] + self.base_lane_offset * (-1 if offset_right else 1), p[1]) 
                    for p in lane_boundary]
        
        # Detect sharp turn and adapt offset (BFMC technique)
        curvature = abs(a)
        offset_dist = self.base_lane_offset
        
        if curvature > self.sharp_turn_threshold:
            offset_dist = self.base_lane_offset * self.sharp_turn_multiplier
            if self.debug:
                print(f"[LKA DEBUG] Sharp turn in boundary! Curvature: {curvature:.6f}, widened offset: {offset_dist:.2f}m")
        
        # ADAPTIVE PREDICTIVE LOOKAHEAD (same logic as both-boundaries case)
        base_lookahead = 2
        if curvature > 0.0002:  # Very sharp curve
            lookahead_points = 0
        elif curvature > 0.0001:  # Sharp curve
            lookahead_points = 1
        else:
            lookahead_points = base_lookahead
            if current_speed > 15.0:  # High speed
                lookahead_points = 3
        
        # Create virtual center by sampling from polynomial with offset
        virtual_center = []
        for i, x_val in enumerate(points_x):
            if lookahead_points > 0:
                # Use lookahead x-coordinate for polynomial evaluation
                lookahead_idx = min(i + lookahead_points, len(points_x) - 1)
                x_lookahead = points_x[lookahead_idx]
                
                # Calculate y from polynomial at lookahead position
                y_poly = a * x_lookahead**2 + b * x_lookahead + c
                
                # Tangent at lookahead position
                tangent_slope = 2 * a * x_lookahead + b
                tangent_angle = np.arctan(tangent_slope)
            else:
                # No lookahead - use current position
                x_lookahead = x_val
                y_poly = a * x_val**2 + b * x_val + c
                tangent_slope = 2 * a * x_val + b
                tangent_angle = np.arctan(tangent_slope)
            
            # Perpendicular offset
            offset_angle = tangent_angle + (np.pi/2 if offset_right else -np.pi/2)
            
            # Apply offset
            vx = x_val + offset_dist * np.cos(offset_angle)
            if lookahead_points > 0:
                vy = y_poly - (x_lookahead - x_val) * np.tan(tangent_angle) + offset_dist * np.sin(offset_angle)
            else:
                vy = y_poly + offset_dist * np.sin(offset_angle)
            virtual_center.append((vx, vy))
        
        return virtual_center
    
    def _calculate_speed_control(self, lane_left, lane_right, track):
        """
        Predict curve ahead and calculate required speed adjustment
        Uses LANE GEOMETRY to detect curves (not car steering!)
        """
        # Predict future trajectory using LANE PATH
        curve_radius = self._predict_curve_radius_from_lane(lane_left, lane_right)
        
        # Store for visualization
        self.current_curve_radius = curve_radius
        
        # Calculate maximum safe speed for this curve
        # Formula: v_max = sqrt(a_lat_max * R)
        if curve_radius < self.curve_entry_threshold:  # We're in a curve
            max_safe_speed = np.sqrt(self.lateral_accel_limit * curve_radius)
            
            # Track if we just entered a curve
            if not self.in_curve:
                self.in_curve = True
                self.curve_min_safe_speed = max_safe_speed
                print(f"\033[91m{'='*80}\033[0m")
                print(f"\033[91m🔴 ENTERING CURVE! Radius: {curve_radius:.1f}m | Safe Speed: {max_safe_speed*3.6:.1f} km/h\033[0m")
                print(f"\033[91m{'='*80}\033[0m")
            else:
                # Already in curve - track minimum safe speed
                if max_safe_speed < self.curve_min_safe_speed:
                    self.curve_min_safe_speed = max_safe_speed
                    print(f"\033[93m⚠️ TIGHTER CURVE! New min speed: {max_safe_speed*3.6:.1f} km/h (radius: {curve_radius:.1f}m)\033[0m")
                
                # Use the minimum safe speed encountered in this curve
                max_safe_speed = self.curve_min_safe_speed
        else:
            # Exiting curve or on straight
            if self.in_curve:
                print(f"\033[92m✓ EXITING CURVE - Back to straight\033[0m")
                self.in_curve = False
                self.curve_min_safe_speed = float('inf')
            
            max_safe_speed = self.car.max_velocity  # No speed limit for straight
        
        self.safe_speed = max_safe_speed
        
        current_speed = abs(self.car.velocity)
        target_speed = max_safe_speed * self.comfort_margin
        
        # DEBUG - Show speed calculations
        print(f"[SPEED] InCurve: {self.in_curve}, Radius: {curve_radius:.1f}m, Current: {current_speed*3.6:.1f} km/h, Safe: {max_safe_speed*3.6:.1f} km/h, Target: {target_speed*3.6:.1f} km/h")
        
        # CRITICAL: Brake if significantly above target, but add hysteresis to avoid constant braking
        # Brake threshold: 105% of target (allows small overshoot before braking)
        brake_threshold = target_speed * 1.05
        
        if current_speed > brake_threshold:
            # Speed is too high - MUST brake NOW
            prediction_distance = current_speed * (self.prediction_horizon * self.prediction_dt)
            
            if prediction_distance > 0.1:
                # Required deceleration: v^2 = v0^2 + 2ad
                required_decel = (current_speed**2 - target_speed**2) / (2 * prediction_distance)
                brake_decel = min(required_decel, self.max_comfort_decel)
                
                # Convert to brake pedal (0-1), using full 1.0g braking capability
                brake_amount = brake_decel / (1.0 * 9.81)
                
                # Increase brake force if speed is dangerously high
                if current_speed > max_safe_speed:
                    # Already over safe limit - HARD BRAKE!
                    brake_amount = max(brake_amount, 1.0)  # Full emergency brake
                
                brake_amount = np.clip(brake_amount, 0.3, 1.0)  # Allow up to 100% brake
                
                return {
                    'action': 'brake',
                    'brake': brake_amount,
                    'target_speed': target_speed,
                    'curve_radius': curve_radius
                }
            else:
                # Emergency brake if prediction fails
                return {
                    'action': 'brake',
                    'brake': 1.0,  # Full brake in emergency
                    'target_speed': target_speed,
                    'curve_radius': curve_radius
                }
        
        elif current_speed < target_speed * self.accel_threshold:
            # Only accelerate if NOT in a curve
            if self.in_curve:
                # In curve - coast, don't accelerate
                return {
                    'action': 'maintain',
                    'target_speed': target_speed,
                    'curve_radius': curve_radius
                }
            
            # Safe to accelerate - on straight section
            speed_ratio = current_speed / max_safe_speed if max_safe_speed > 0 else 0
            
            if curve_radius > 500:  # Straight or very gentle curve
                throttle_amount = 1.0  # Full throttle
            elif curve_radius > 100:
                throttle_amount = 0.7  # Moderate throttle for gentle curves
            else:
                throttle_amount = 0.4  # Gentle throttle for tight curves
            
            return {
                'action': 'accelerate',
                'throttle': throttle_amount,
                'target_speed': target_speed,
                'curve_radius': curve_radius
            }
        
        # Speed is between accel threshold and target - coast (no brake, no throttle)
        # Let natural drag slow the car slightly without fighting it
        return {
            'action': 'maintain',
            'target_speed': target_speed,
            'curve_radius': curve_radius
        }
    
    def _predict_curve_radius_from_lane(self, lane_left, lane_right):
        """
        Calculate curve radius from the LANE GEOMETRY ahead
        This detects upcoming curves regardless of current steering
        
        Uses curvature formula: κ = |dx*d2y - dy*d2x| / (dx^2 + dy^2)^(3/2)
        Radius = 1/κ
        """
        # Get center line points
        n = min(len(lane_left), len(lane_right))
        if n < 5:
            # print(f"[CURVE DEBUG] Not enough lane points: {n}")
            return float('inf')  # Not enough points
        
        center_points = []
        for i in range(n):  # Use ALL available points
            cx = (lane_left[i][0] + lane_right[i][0]) / 2.0
            cy = (lane_left[i][1] + lane_right[i][1]) / 2.0
            center_points.append((cx, cy))
        
        if len(center_points) < 5:
            return float('inf')
        
        # Check if points form a straight line using R² coefficient
        # If R² > 0.995, it's essentially a straight line
        points_x = np.array([p[0] for p in center_points])
        points_y = np.array([p[1] for p in center_points])
        
        # Calculate R² for linear fit
        if len(points_x) >= 3:
            # Fit linear regression
            dx = points_x[-1] - points_x[0]
            dy = points_y[-1] - points_y[0]
            
            if abs(dx) > abs(dy):
                # Fit y = mx + b
                coeffs = np.polyfit(points_x, points_y, 1)
                y_fit = np.polyval(coeffs, points_x)
                ss_res = np.sum((points_y - y_fit) ** 2)
                ss_tot = np.sum((points_y - np.mean(points_y)) ** 2)
            else:
                # Fit x = my + b
                coeffs = np.polyfit(points_y, points_x, 1)
                x_fit = np.polyval(coeffs, points_y)
                ss_res = np.sum((points_x - x_fit) ** 2)
                ss_tot = np.sum((points_x - np.mean(points_x)) ** 2)
            
            if ss_tot > 1e-10:  # Avoid division by zero
                r_squared = 1 - (ss_res / ss_tot)
                
                # If R² > 0.995, treat as straight line
                if r_squared > 0.995:
                    return float('inf')
        
        # Calculate curvature using derivative method (more accurate for actual road curves)
        # Use points in the middle portion of the detected lane (where data is most reliable)
        start_idx = min(2, len(center_points) - 3)
        end_idx = max(start_idx + 3, len(center_points) - 1)
        
        # Fit a polynomial to the lane center
        points_x = [p[0] for p in center_points[start_idx:end_idx]]
        points_y = [p[1] for p in center_points[start_idx:end_idx]]
        
        if len(points_x) < 3:
            return float('inf')
        
        # Fit 2nd order polynomial: y = a*x^2 + b*x + c
        # Or if vertical, fit x = a*y^2 + b*y + c
        dx = points_x[-1] - points_x[0]
        dy = points_y[-1] - points_y[0]
        
        try:
            if abs(dx) > abs(dy):
                # Fit y as function of x
                coeffs = np.polyfit(points_x, points_y, 2)
                a, b, c = coeffs
                
                # Evaluate at middle point
                mid_x = np.mean(points_x)
                # First derivative: dy/dx = 2*a*x + b
                dydx = 2 * a * mid_x + b
                # Second derivative: d2y/dx2 = 2*a
                d2ydx2 = 2 * a
                
                # Curvature: κ = |d2y/dx2| / (1 + (dy/dx)^2)^(3/2)
                curvature = abs(d2ydx2) / ((1 + dydx**2) ** 1.5)
            else:
                # Fit x as function of y
                coeffs = np.polyfit(points_y, points_x, 2)
                a, b, c = coeffs
                
                # Evaluate at middle point
                mid_y = np.mean(points_y)
                # First derivative: dx/dy = 2*a*y + b
                dxdy = 2 * a * mid_y + b
                # Second derivative: d2x/dy2 = 2*a
                d2xdy2 = 2 * a
                
                # Curvature: κ = |d2x/dy2| / (1 + (dx/dy)^2)^(3/2)
                curvature = abs(d2xdy2) / ((1 + dxdy**2) ** 1.5)
            
            # Radius is inverse of curvature
            if curvature > 1e-6:
                radius = 1.0 / curvature
            else:
                radius = float('inf')
                
        except Exception as e:
            # Fallback to 3-point circle fit if polynomial fails
            idx1 = max(1, len(center_points) // 4)
            idx2 = max(2, len(center_points) // 2)
            idx3 = max(3, (3 * len(center_points)) // 4)
            
            if idx3 >= len(center_points):
                idx3 = len(center_points) - 1
            if idx2 >= idx3:
                idx2 = idx3 - 1
            if idx1 >= idx2:
                idx1 = idx2 - 1
            
            p1 = center_points[idx1]
            p2 = center_points[idx2]
            p3 = center_points[idx3]
            
            radius = self._circle_radius_from_3_points(p1, p2, p3)
        
        # Update running average
        self.radius_history.append(radius)
        if len(self.radius_history) > self.radius_history_size:
            self.radius_history.pop(0)
        
        avg_radius = np.mean(self.radius_history)
        return avg_radius
    
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
    
    def _apply_blended_control(self, steering_command, speed_command):
        """
        Blend AI control with manual input based on intervention strength
        (Mode 3: Assist)
        """
        # Calculate lateral offset from lane center to determine intervention strength
        lateral_offset = self._calculate_lateral_offset()
        intervention = self._calculate_intervention_strength(lateral_offset)
        
        self.intervention_strength = intervention
        
        # Blend steering (manual input comes from car's current state)
        if steering_command is not None:
            # For now, full AI steering when assisting
            # TODO: blend with actual manual input when we track it
            final_steering = steering_command
        else:
            final_steering = None
        
        # Blend speed control (always apply full speed commands for proper acceleration/braking)
        # CRITICAL: Never apply throttle and brake simultaneously!
        if speed_command:
            if speed_command['action'] == 'brake':
                # Apply full brake as calculated, NO throttle
                final_brake = speed_command['brake']
                final_throttle = 0.0
                print(f"[BRAKE ACTION] Brake: {final_brake:.2f}, Throttle: {final_throttle:.2f}")
            elif speed_command['action'] == 'accelerate':
                # Apply full throttle as calculated, NO brake
                final_throttle = speed_command['throttle']
                final_brake = 0.0
                print(f"[ACCEL ACTION] Throttle: {final_throttle:.2f}, Brake: {final_brake:.2f}")
            elif speed_command['action'] == 'maintain':
                # Maintain mode: coast - no throttle, no brake (let natural drag work)
                final_throttle = 0.0
                final_brake = 0.0
                print(f"[MAINTAIN ACTION] COAST - Brake: {final_brake:.2f}, Throttle: {final_throttle:.2f}")
            else:
                final_throttle = 0.0
                final_brake = 0.0
        else:
            final_throttle = 0.0
            final_brake = 0.0
        
        return final_steering, final_throttle, final_brake
    
    def _calculate_lateral_offset(self):
        """Calculate lateral distance from lane center"""
        if self.target_point is None:
            return 0.0
        
        car_x, car_y = self.car.x, self.car.y
        target_x, target_y = self.target_point
        
        # Distance to target point
        offset = np.hypot(target_x - car_x, target_y - car_y)
        
        return offset
    
    def _calculate_intervention_strength(self, lateral_offset):
        """
        Calculate intervention strength (0.0 to 1.0) based on lateral offset
        
        Zones:
        - < 0.5m: No intervention (0.0)
        - 0.5m - 1.5m: Gentle intervention (0.2 - 0.5)
        - > 1.5m: Strong intervention (0.5 - 1.0)
        """
        if lateral_offset < self.no_intervention_zone:
            return 0.0
        
        elif lateral_offset < self.gentle_intervention_zone:
            # Gentle zone: linear interpolation 0.2 to 0.5
            normalized = (lateral_offset - self.no_intervention_zone) / \
                        (self.gentle_intervention_zone - self.no_intervention_zone)
            return 0.2 + 0.3 * normalized
        
        else:
            # Strong zone: linear interpolation 0.5 to 1.0
            normalized = min((lateral_offset - self.gentle_intervention_zone) / 1.0, 1.0)
            return 0.5 + 0.5 * normalized
    
    def _update_warnings(self, steering_command, speed_command):
        """Update warning states for Mode 2 (Warning)"""
        # Clear warnings
        self.warnings = {k: False for k in self.warnings}
        
        # Lane departure warning
        lateral_offset = self._calculate_lateral_offset()
        if lateral_offset > self.warning_thresholds['lateral_offset_warn']:
            self.warnings['lane_departure'] = True
        
        # Speed too high for curve
        if speed_command and self.safe_speed:
            current_speed = abs(self.car.velocity)
            if current_speed > self.safe_speed * self.warning_thresholds['speed_margin_warn']:
                self.warnings['speed_too_high'] = True
        
        # Time to lane crossing
        # Simple estimate: lateral_offset / lateral_velocity
        # For now, triggered if offset is high
        if lateral_offset > 1.2:
            self.warnings['time_to_crossing'] = True
