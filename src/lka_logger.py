"""
LKA Performance Logger and Visualization Module

Logs performance metrics for empirical validation of Linear State-Feedback Control:
- Lateral error (e_y) and heading error (e_theta) time series
- Speed profiles vs safe speed (curve-based)
- Steering response and angular velocity commands
- Trajectory data with curve detection
- Control gains (K2, K3) variation
- Error distributions and phase portraits
- Curve state transitions and adaptive lookahead
- Control effort statistics
"""

import numpy as np
import time
import json
import os
from collections import deque


def _to_serializable(obj):
    """Recursively convert numpy types to native Python for JSON serialization (NumPy >=2)."""
    import numpy as _np

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return [_to_serializable(o) for o in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


class LKAPerformanceLogger:
    """Logs LKA system performance metrics for analysis and visualization"""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with specified directory"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Session info
        self.session_start = time.time()
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        
        # Real-time metrics (last N samples for moving statistics)
        self.window_size = 300  # 10 seconds at 30 Hz
        self.lateral_errors = deque(maxlen=self.window_size)  # e_y in robot frame
        self.heading_errors = deque(maxlen=self.window_size)  # e_theta in robot frame
        self.speeds = deque(maxlen=self.window_size)
        self.safe_speeds = deque(maxlen=self.window_size)
        self.curve_radii = deque(maxlen=self.window_size)
        self.lookahead_distances = deque(maxlen=self.window_size)  # Adaptive lookahead
        self.steering_angles = deque(maxlen=self.window_size)
        self.angular_velocities = deque(maxlen=self.window_size)  # ω commands
        self.intervention_states = deque(maxlen=self.window_size)
        self.curve_states = deque(maxlen=self.window_size)  # in_curve boolean
        self.control_gains_K2 = deque(maxlen=self.window_size)  # K2 gain values
        self.control_gains_K3 = deque(maxlen=self.window_size)  # K3 gain values
        
        # Full session logs (for post-processing)
        self.time_series = []
        self.trajectories = []
        
        # Statistics
        self.total_frames = 0
        self.intervention_count = 0
        self.last_intervention_state = False
        self.intervention_durations = []
        self.current_intervention_start = None
        
        # Performance metrics
        self.speed_violations = 0
        self.lane_departures = 0
        
    def log_frame(self, timestamp, car, lka_controller, warnings):
        """Log data for a single frame"""
        self.total_frames += 1
        
        # Extract metrics
        lateral_error = self._compute_lateral_error(lka_controller)  # e_y
        heading_error = self._compute_heading_error(lka_controller)  # e_theta
        speed = abs(car.velocity)
        safe_speed = lka_controller.safe_speed if lka_controller.safe_speed else speed
        curve_radius = getattr(lka_controller, "_last_curve_radius", None)
        curve_radius = curve_radius if curve_radius else float('inf')
        
        # Adaptive lookahead based on curve state (steering uses immediate detection)
        in_curve_steering = getattr(lka_controller, "in_curve_steering", False)
        in_curve_speed = getattr(lka_controller, "in_curve_speed", False)
        
        # Calculate speed-adaptive lookahead
        if in_curve_steering:
            lookahead_time = getattr(lka_controller, 'lookahead_time_curve', 0.5)
            lookahead_max = getattr(lka_controller, 'lookahead_max_curve', 8.0)
        else:
            lookahead_time = getattr(lka_controller, 'lookahead_time_straight', 1.0)
            lookahead_max = getattr(lka_controller, 'lookahead_max_straight', 20.0)
        
        lookahead_min = getattr(lka_controller, 'lookahead_min', 3.0)
        lookahead = lookahead_time * max(abs(speed), 1.0)
        lookahead = np.clip(lookahead, lookahead_min, lookahead_max)
        
        steering_angle = np.degrees(car.steering_angle)
        
        # Calculate angular velocity from steering (ω = v·tan(δ)/L)
        if abs(speed) > 0.1:
            angular_velocity = speed * np.tan(car.steering_angle) / car.wheelbase
        else:
            angular_velocity = 0.0
        
        # Calculate control gains (K2, K3) from controller parameters
        zeta = getattr(lka_controller, "zeta", 0.8)
        omega_n = getattr(lka_controller, "omega_n", 1.2)
        omega_ref = 0.0  # Assuming straight reference
        v_ref = max(abs(speed), 1.0)
        K2 = (omega_n ** 2 - omega_ref ** 2) / abs(v_ref)
        K3 = 2 * zeta * omega_n
        
        intervening = lka_controller.intervening
        
        # Update real-time buffers
        self.lateral_errors.append(lateral_error)
        self.heading_errors.append(heading_error)
        self.speeds.append(speed)
        self.safe_speeds.append(safe_speed)
        self.curve_radii.append(curve_radius)
        self.lookahead_distances.append(lookahead)
        self.steering_angles.append(steering_angle)
        self.angular_velocities.append(angular_velocity)
        self.intervention_states.append(1 if intervening else 0)
        self.curve_states.append(1 if in_curve_speed else 0)  # Use speed curve state (with hysteresis)
        self.control_gains_K2.append(K2)
        self.control_gains_K3.append(K3)
        
        # Track interventions
        if intervening and not self.last_intervention_state:
            self.intervention_count += 1
            self.current_intervention_start = timestamp
        elif not intervening and self.last_intervention_state:
            if self.current_intervention_start is not None:
                duration = timestamp - self.current_intervention_start
                self.intervention_durations.append(duration)
                self.current_intervention_start = None
        self.last_intervention_state = intervening
        
        # Track violations
        if warnings.get('speed_too_high'):
            self.speed_violations += 1
        if warnings.get('lane_departure'):
            self.lane_departures += 1
        
        # Store full time series (sample at 5 Hz to reduce memory)
        if self.total_frames % 6 == 0:
            self.time_series.append({
                'timestamp': timestamp,
                'lateral_error': lateral_error,
                'heading_error': heading_error,
                'speed': speed,
                'safe_speed': safe_speed,
                'curve_radius': curve_radius,
                'lookahead': lookahead,
                'steering': steering_angle,
                'angular_velocity': angular_velocity,
                'intervening': intervening,
                'in_curve': in_curve_speed,  # Speed curve state (with hysteresis)
                'in_curve_steering': in_curve_steering,  # Immediate steering state
                'K2': K2,
                'K3': K3,
                'car_x': car.x,
                'car_y': car.y,
                'car_theta': car.theta,
            })
        
        # Store trajectory points (lower sample rate)
        if self.total_frames % 10 == 0:
            self.trajectories.append({
                'x': car.x,
                'y': car.y,
                'speed': speed,
                'error': lateral_error,
            })
    
    def _compute_lateral_error(self, controller):
        """Estimate lateral error from controller state"""
        return controller._estimate_lane_offset() if hasattr(controller, '_estimate_lane_offset') else 0.0
    
    def _compute_heading_error(self, controller):
        """Estimate heading error from controller state (e_theta in robot frame)"""
        if controller.target_direction is not None and hasattr(controller, '_state'):
            # Use sensor-based heading from controller state
            sensor_theta = controller._state.get('theta', controller.car.theta)
            heading_error = (controller.target_direction - sensor_theta + np.pi) % (2 * np.pi) - np.pi
            return np.degrees(heading_error)
        return 0.0
    
    def _estimate_lookahead(self, controller, speed):
        """Estimate current lookahead distance (speed-adaptive with curve state)"""
        in_curve_steering = getattr(controller, 'in_curve_steering', False)
        
        # Get lookahead parameters
        if in_curve_steering:
            lookahead_time = getattr(controller, 'lookahead_time_curve', 0.5)
            lookahead_max = getattr(controller, 'lookahead_max_curve', 8.0)
        else:
            lookahead_time = getattr(controller, 'lookahead_time_straight', 1.0)
            lookahead_max = getattr(controller, 'lookahead_max_straight', 20.0)
        
        lookahead_min = getattr(controller, 'lookahead_min', 3.0)
        
        # Calculate adaptive lookahead: L = t * v, clamped
        lookahead = lookahead_time * max(abs(speed), 1.0)
        lookahead = np.clip(lookahead, lookahead_min, lookahead_max)
        
        return lookahead
    
    def get_current_metrics(self):
        """Get current real-time metrics for HUD display"""
        if not self.lateral_errors:
            return {}
        
        return {
            'lateral_error': self.lateral_errors[-1],
            'lateral_error_std': np.std(self.lateral_errors) if len(self.lateral_errors) > 1 else 0.0,
            'heading_error': self.heading_errors[-1],
            'speed': self.speeds[-1],
            'safe_speed': self.safe_speeds[-1],
            'speed_ratio': (self.speeds[-1] / self.safe_speeds[-1] * 100) if self.safe_speeds[-1] > 0 else 0.0,
            'curve_radius': self.curve_radii[-1],
            'lookahead': self.lookahead_distances[-1],
            'steering_angle': self.steering_angles[-1],
            'intervening': bool(self.intervention_states[-1]),
        }
    
    def get_statistics(self):
        """Compute session statistics. Always return all keys, even if no data."""
        now = time.time()
        duration = now - self.session_start if hasattr(self, 'session_start') else 0.0
        if not self.lateral_errors:
            # Return all keys with default values
            return {
                'lateral_error_mean': 0.0,
                'lateral_error_std': 0.0,
                'lateral_error_max': 0.0,
                'speed_compliance_rate': 0.0,
                'intervention_count': 0,
                'intervention_frequency': 0.0,
                'mean_intervention_duration': 0.0,
                'speed_violations': 0,
                'lane_departures': 0,
                'steering_mean': 0.0,
                'steering_std': 0.0,
                'total_frames': 0,
                'session_duration': duration,
            }
        lateral_arr = np.array(self.lateral_errors)
        speed_arr = np.array(self.speeds)
        safe_speed_arr = np.array(self.safe_speeds)
        steering_arr = np.array(self.steering_angles)
        # Speed compliance
        compliant = np.sum(speed_arr <= safe_speed_arr)
        compliance_rate = (compliant / len(speed_arr) * 100) if len(speed_arr) > 0 else 0.0
        # Intervention stats
        mean_intervention_duration = np.mean(self.intervention_durations) if self.intervention_durations else 0.0
        intervention_freq = (self.intervention_count / duration) * 60 if duration > 0 else 0.0
        return {
            'lateral_error_mean': float(np.mean(lateral_arr)),
            'lateral_error_std': float(np.std(lateral_arr)),
            'lateral_error_max': float(np.max(np.abs(lateral_arr))),
            'speed_compliance_rate': float(compliance_rate),
            'intervention_count': self.intervention_count,
            'intervention_frequency': float(intervention_freq),
            'mean_intervention_duration': float(mean_intervention_duration),
            'speed_violations': self.speed_violations,
            'lane_departures': self.lane_departures,
            'steering_mean': float(np.mean(steering_arr)),
            'steering_std': float(np.std(steering_arr)),
            'total_frames': self.total_frames,
            'session_duration': duration,
        }
    
    def save_session(self):
        """Save session data to disk for post-processing"""
        session_file = os.path.join(self.log_dir, f"lka_session_{self.session_id}.json")
        
        data = {
            'session_id': self.session_id,
            'statistics': self.get_statistics(),
            'time_series': self.time_series,
            'trajectories': self.trajectories,
        }

        # Convert numpy scalar types (including np.bool_) to native Python
        data = _to_serializable(data)
        
        with open(session_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Session data saved to: {session_file}")
        return session_file
    
    def print_summary(self):
        """Print session summary to console"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("LKA PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {stats['session_duration']:.1f} s")
        print(f"Total Frames: {stats['total_frames']}")
        print()
        print("LATERAL TRACKING:")
        print(f"  Mean Error: {stats['lateral_error_mean']:+.3f} m")
        print(f"  Std Dev: {stats['lateral_error_std']:.3f} m")
        print(f"  Max Error: {stats['lateral_error_max']:.3f} m")
        print()
        print("SPEED CONTROL:")
        print(f"  Compliance Rate: {stats['speed_compliance_rate']:.1f}%")
        print(f"  Speed Violations: {stats['speed_violations']}")
        print()
        print("INTERVENTIONS:")
        print(f"  Total Count: {stats['intervention_count']}")
        print(f"  Frequency: {stats['intervention_frequency']:.2f} /min")
        print(f"  Mean Duration: {stats['mean_intervention_duration']:.3f} s")
        print()
        print("STEERING:")
        print(f"  Mean Angle: {stats['steering_mean']:+.1f}°")
        print(f"  Std Dev: {stats['steering_std']:.1f}°")
        print("="*60 + "\n")


class LKAVisualizationGenerator:
    """Generate visualization plots from logged data"""
    
    def __init__(self, session_file):
        """Load session data from file"""
        with open(session_file, 'r') as f:
            data = json.load(f)
        
        self.session_id = data['session_id']
        self.statistics = data['statistics']
        self.time_series = data['time_series']
        self.trajectories = data['trajectories']
    
    def generate_all_plots(self, output_dir="plots"):
        """Generate all visualization plots for Linear LKA Control"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating visualization plots...")
        
        # Extract time series data
        timestamps = [d['timestamp'] for d in self.time_series]
        lateral_errors = [d['lateral_error'] for d in self.time_series]
        heading_errors = [d['heading_error'] for d in self.time_series]
        speeds = [d['speed'] for d in self.time_series]
        safe_speeds = [d['safe_speed'] for d in self.time_series]
        steering_angles = [d['steering'] for d in self.time_series]
        angular_velocities = [d.get('angular_velocity', 0) for d in self.time_series]
        intervening = [d['intervening'] for d in self.time_series]
        in_curve = [d.get('in_curve', False) for d in self.time_series]
        curve_radii = [d['curve_radius'] for d in self.time_series]
        lookaheads = [d['lookahead'] for d in self.time_series]
        K2_values = [d.get('K2', 0) for d in self.time_series]
        K3_values = [d.get('K3', 0) for d in self.time_series]
        
        # Normalize timestamps to start at 0
        t0 = timestamps[0]
        time_sec = [(t - t0) for t in timestamps]
        
        # 1. Lateral & Heading Errors (dual plot)
        self._plot_error_time_series(time_sec, lateral_errors, heading_errors, intervening, in_curve, output_dir)
        
        # 2. Speed Profile vs Safe Speed with Curve Zones
        self._plot_speed_profile(time_sec, speeds, safe_speeds, in_curve, curve_radii, output_dir)
        
        # 3. Steering & Angular Velocity Response
        self._plot_steering_response(time_sec, steering_angles, angular_velocities, lateral_errors, output_dir)
        
        # 4. Control Gains Variation (K2, K3)
        self._plot_control_gains(time_sec, K2_values, K3_values, speeds, output_dir)
        
        # 5. Adaptive Lookahead vs Curve State
        self._plot_adaptive_lookahead(time_sec, lookaheads, in_curve, curve_radii, output_dir)
        
        # 6. Trajectory Overlay with Curve Zones
        self._plot_trajectory_overlay(output_dir)
        
        # 7. Error Distribution Histogram
        self._plot_error_distribution(lateral_errors, heading_errors, output_dir)
        
        # 8. Phase Portrait (e_y vs e_theta)
        self._plot_phase_portrait(lateral_errors, heading_errors, output_dir)
        
        # 9. Control Effort Distribution
        self._plot_control_distribution(steering_angles, output_dir)
        
        print(f"All plots saved to: {output_dir}/")
    
    def _plot_error_time_series(self, time_sec, lateral_errors, heading_errors, intervening, in_curve, output_dir):
        """Plot lateral and heading errors over time with intervention and curve zones"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Shade curve zones
        in_curve_zone = False
        start_t = None
        for i, (t, curve) in enumerate(zip(time_sec, in_curve)):
            if curve and not in_curve_zone:
                start_t = t
                in_curve_zone = True
            elif not curve and in_curve_zone:
                ax1.axvspan(start_t, t, alpha=0.15, color='yellow', label='Curve' if start_t == time_sec[0] else '')
                ax2.axvspan(start_t, t, alpha=0.15, color='yellow')
                in_curve_zone = False
        
        # Shade intervention zones
        in_intervention = False
        start_t = None
        for i, (t, interv) in enumerate(zip(time_sec, intervening)):
            if interv and not in_intervention:
                start_t = t
                in_intervention = True
            elif not interv and in_intervention:
                ax1.axvspan(start_t, t, alpha=0.2, color='orange', label='Intervention' if i < 10 else '')
                ax2.axvspan(start_t, t, alpha=0.2, color='orange')
                in_intervention = False
        
        # Plot lateral error (e_y)
        ax1.plot(time_sec, lateral_errors, 'b-', linewidth=1.5, label='Lateral Error (e_y)')
        ax1.axhline(0.8, color='orange', linestyle='--', linewidth=1, label='Intervention Threshold')
        ax1.axhline(-0.8, color='orange', linestyle='--', linewidth=1)
        ax1.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Lateral Error e_y (m)')
        ax1.set_title('Linear LKA Control: Error Tracking Performance')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot heading error (e_theta)
        ax2.plot(time_sec, heading_errors, 'r-', linewidth=1.5, label='Heading Error (e_θ)')
        ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Heading Error e_θ (°)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_time_series.png'), dpi=300)
        plt.close()
    
    def _plot_speed_profile(self, time_sec, speeds, safe_speeds, in_curve, curve_radii, output_dir):
        """Plot actual speed vs safe speed with curve zones"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Shade curve zones
        in_curve_zone = False
        start_t = None
        for i, (t, curve) in enumerate(zip(time_sec, in_curve)):
            if curve and not in_curve_zone:
                start_t = t
                in_curve_zone = True
            elif not curve and in_curve_zone:
                ax1.axvspan(start_t, t, alpha=0.15, color='yellow', label='Curve Zone' if start_t == time_sec[0] else '')
                ax2.axvspan(start_t, t, alpha=0.15, color='yellow')
                in_curve_zone = False
        
        # Convert to km/h
        speeds_kmh = [s * 3.6 for s in speeds]
        safe_kmh = [s * 3.6 for s in safe_speeds]
        
        # Speed plot
        ax1.plot(time_sec, speeds_kmh, 'b-', linewidth=1.5, label='Actual Speed')
        ax1.plot(time_sec, safe_kmh, 'r--', linewidth=1.5, label='Safe Speed')
        ax1.fill_between(time_sec, speeds_kmh, safe_kmh, 
                        where=np.array(speeds) > np.array(safe_speeds),
                        alpha=0.3, color='red', label='Overspeed')
        ax1.set_ylabel('Speed (km/h)')
        ax1.set_title('Speed Control Performance with Curve Detection')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Curve radius plot
        # Clip infinite values for visualization
        radii_clipped = [min(r, 1000) for r in curve_radii]
        ax2.plot(time_sec, radii_clipped, 'g-', linewidth=1.5, label='Curve Radius')
        ax2.axhline(500, color='orange', linestyle='--', linewidth=1, label='Curve Threshold (500m)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Radius (m)')
        ax2.set_ylim(0, 1000)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_profile.png'), dpi=300)
        plt.close()
    
    def _plot_steering_response(self, time_sec, steering, angular_velocities, errors, output_dir):
        """Triple-axis plot: steering angle, angular velocity, and lateral error"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Steering angle
        ax1.plot(time_sec, steering, 'b-', linewidth=1.5, label='Steering Angle δ')
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax1.set_ylabel('Steering Angle δ (°)')
        ax1.set_title('Control Commands and Response')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Angular velocity (ω command)
        ax2.plot(time_sec, angular_velocities, 'g-', linewidth=1.5, label='Angular Velocity ω')
        ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax2.set_ylabel('Angular Velocity ω (rad/s)')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Lateral error (response)
        ax3.plot(time_sec, errors, 'r-', linewidth=1.5, label='Lateral Error e_y')
        ax3.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Lateral Error e_y (m)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'steering_response.png'), dpi=300)
        plt.close()
    
    def _plot_trajectory_overlay(self, output_dir):
        """2D top-down trajectory with color-coded speed"""
        import matplotlib.pyplot as plt
        
        if not self.trajectories:
            return
        
        xs = [t['x'] for t in self.trajectories]
        ys = [t['y'] for t in self.trajectories]
        speeds = [t['speed'] * 3.6 for t in self.trajectories]  # km/h
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        scatter = ax.scatter(xs, ys, c=speeds, cmap='jet', s=5, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Speed (km/h)')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Vehicle Trajectory (Color-coded by Speed)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trajectory_overlay.png'), dpi=300)
        plt.close()
    
    def _plot_error_distribution(self, lateral_errors, heading_errors, output_dir):
        """Histogram of lateral and heading errors with Gaussian fit"""
        import matplotlib.pyplot as plt
        from scipy import stats
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Lateral error distribution
        lat_arr = np.array(lateral_errors)
        mu_lat = np.mean(lat_arr)
        sigma_lat = np.std(lat_arr)
        
        ax1.hist(lat_arr, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        x_lat = np.linspace(lat_arr.min(), lat_arr.max(), 100)
        gaussian_lat = stats.norm.pdf(x_lat, mu_lat, sigma_lat)
        ax1.plot(x_lat, gaussian_lat, 'r-', linewidth=2, 
                label=f'Gaussian fit\nμ={mu_lat:.3f}m, σ={sigma_lat:.3f}m')
        ax1.set_xlabel('Lateral Error e_y (m)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Lateral Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Heading error distribution
        head_arr = np.array(heading_errors)
        mu_head = np.mean(head_arr)
        sigma_head = np.std(head_arr)
        
        ax2.hist(head_arr, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
        x_head = np.linspace(head_arr.min(), head_arr.max(), 100)
        gaussian_head = stats.norm.pdf(x_head, mu_head, sigma_head)
        ax2.plot(x_head, gaussian_head, 'b-', linewidth=2,
                label=f'Gaussian fit\nμ={mu_head:.2f}°, σ={sigma_head:.2f}°')
        ax2.set_xlabel('Heading Error e_θ (°)')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Heading Error Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_phase_portrait(self, lateral_errors, heading_errors, output_dir):
        """Phase portrait: e_y vs e_theta (state space)"""
        import matplotlib.pyplot as plt
        
        lat_arr = np.array(lateral_errors)
        head_arr = np.array(heading_errors)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Color by time (blue -> red)
        colors = np.linspace(0, 1, len(lat_arr))
        scatter = ax.scatter(lat_arr, head_arr, c=colors, cmap='coolwarm', 
                            s=10, alpha=0.6, label='Trajectory')
        
        # Highlight origin (stable point)
        ax.plot(0, 0, 'go', markersize=12, label='Target (0, 0)', zorder=5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        # Add colorbar for time
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Normalized Time')
        
        ax.set_xlabel('Lateral Error e_y (m)')
        ax.set_ylabel('Heading Error e_θ (°)')
        ax.set_title('State Space: e_y vs e_θ')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'phase_portrait.png'), dpi=300)
        plt.close()
    
    def _plot_control_distribution(self, steering_angles, output_dir):
        """Histogram of steering angles"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        steering_arr = np.array(steering_angles)
        
        ax.hist(steering_arr, bins=40, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Center')
        ax.axvline(np.mean(steering_arr), color='blue', linestyle='--', linewidth=2, 
                  label=f'Mean: {np.mean(steering_arr):.1f}°')
        
        ax.set_xlabel('Steering Angle (°)')
        ax.set_ylabel('Frequency')
        ax.set_title('Control Effort Distribution (Steering Angles)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'control_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_control_gains(self, time_sec, K2_values, K3_values, speeds, output_dir):
        """Plot control gains variation over time"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # K2 gain (lateral error feedback)
        ax1.plot(time_sec, K2_values, 'b-', linewidth=1.5, label='K2 (lateral gain)')
        ax1.set_ylabel('K2')
        ax1.set_title('Linear Control Gains Variation')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # K3 gain (heading error feedback)
        ax2.plot(time_sec, K3_values, 'r-', linewidth=1.5, label='K3 (heading gain)')
        ax2.set_ylabel('K3')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Speed (for context - K2 varies with speed)
        speeds_kmh = [s * 3.6 for s in speeds]
        ax3.plot(time_sec, speeds_kmh, 'g-', linewidth=1.5, label='Speed')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Speed (km/h)')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'control_gains.png'), dpi=300)
        plt.close()
    
    def _plot_adaptive_lookahead(self, time_sec, lookaheads, in_curve, curve_radii, output_dir):
        """Plot adaptive lookahead distance vs curve state"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Shade curve zones
        in_curve_zone = False
        start_t = None
        for i, (t, curve) in enumerate(zip(time_sec, in_curve)):
            if curve and not in_curve_zone:
                start_t = t
                in_curve_zone = True
            elif not curve and in_curve_zone:
                ax1.axvspan(start_t, t, alpha=0.15, color='yellow', label='Curve' if start_t == time_sec[0] else '')
                ax2.axvspan(start_t, t, alpha=0.15, color='yellow')
                in_curve_zone = False
        
        # Lookahead distance
        ax1.plot(time_sec, lookaheads, 'b-', linewidth=2, label='Lookahead Distance')
        ax1.axhline(12.0, color='green', linestyle='--', linewidth=1, label='Straight (12m)')
        ax1.axhline(5.0, color='orange', linestyle='--', linewidth=1, label='Curve (5m)')
        ax1.set_ylabel('Lookahead Distance (m)')
        ax1.set_title('Adaptive Lookahead Control')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 15)
        
        # Curve radius (clipped for visualization)
        radii_clipped = [min(r, 1000) for r in curve_radii]
        ax2.plot(time_sec, radii_clipped, 'r-', linewidth=1.5, label='Curve Radius')
        ax2.axhline(500, color='orange', linestyle='--', linewidth=1, label='Curve Threshold')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Radius (m)')
        ax2.set_ylim(0, 1000)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adaptive_lookahead.png'), dpi=300)
        plt.close()
