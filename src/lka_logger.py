"""
LKA Performance Logger and Visualization Module

Logs performance metrics for empirical validation:
- Lateral error time series
- Speed profiles vs safe speed
- Steering response
- Trajectory data
- Error distributions
- Phase portraits
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
        self.lateral_errors = deque(maxlen=self.window_size)
        self.heading_errors = deque(maxlen=self.window_size)
        self.speeds = deque(maxlen=self.window_size)
        self.safe_speeds = deque(maxlen=self.window_size)
        self.curve_radii = deque(maxlen=self.window_size)
        self.lookahead_distances = deque(maxlen=self.window_size)
        self.steering_angles = deque(maxlen=self.window_size)
        self.intervention_states = deque(maxlen=self.window_size)
        
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
        
    def log_frame(self, timestamp, car, hybrid_controller, warnings):
        """Log data for a single frame"""
        self.total_frames += 1
        
        # Extract metrics
        lateral_error = self._compute_lateral_error(hybrid_controller)
        heading_error = self._compute_heading_error(hybrid_controller)
        speed = abs(car.velocity)
        safe_speed = hybrid_controller.safe_speed if hybrid_controller.safe_speed else speed
        curve_radius = getattr(hybrid_controller, "_last_curve_radius", None)
        curve_radius = curve_radius if curve_radius else float('inf')
        lookahead = self._estimate_lookahead(hybrid_controller, speed)
        steering_angle = np.degrees(car.steering_angle)
        intervening = hybrid_controller.intervening
        
        # Update real-time buffers
        self.lateral_errors.append(lateral_error)
        self.heading_errors.append(heading_error)
        self.speeds.append(speed)
        self.safe_speeds.append(safe_speed)
        self.curve_radii.append(curve_radius)
        self.lookahead_distances.append(lookahead)
        self.steering_angles.append(steering_angle)
        self.intervention_states.append(1 if intervening else 0)
        
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
                'intervening': intervening,
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
        """Estimate heading error from controller state"""
        if controller.target_direction is not None:
            heading_error = (controller.target_direction - controller.car.theta + np.pi) % (2 * np.pi) - np.pi
            return np.degrees(heading_error)
        return 0.0
    
    def _estimate_lookahead(self, controller, speed):
        """Estimate current lookahead distance"""
        # Updated for linear control - uses fixed lookahead_distance
        return getattr(controller, 'lookahead_distance', 15.0)
    
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
        """Compute session statistics"""
        if not self.lateral_errors:
            return {}
        
        lateral_arr = np.array(self.lateral_errors)
        speed_arr = np.array(self.speeds)
        safe_speed_arr = np.array(self.safe_speeds)
        steering_arr = np.array(self.steering_angles)
        
        # Speed compliance
        compliant = np.sum(speed_arr <= safe_speed_arr)
        compliance_rate = (compliant / len(speed_arr) * 100) if len(speed_arr) > 0 else 0.0
        
        # Intervention stats
        mean_intervention_duration = np.mean(self.intervention_durations) if self.intervention_durations else 0.0
        intervention_freq = (self.intervention_count / (time.time() - self.session_start)) * 60  # per minute
        
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
            'session_duration': time.time() - self.session_start,
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
        """Generate all visualization plots"""
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
        intervening = [d['intervening'] for d in self.time_series]
        
        # Normalize timestamps to start at 0
        t0 = timestamps[0]
        time_sec = [(t - t0) for t in timestamps]
        
        # 1. Lateral Error Time Series
        self._plot_lateral_error_series(time_sec, lateral_errors, intervening, output_dir)
        
        # 2. Speed Profile vs Safe Speed
        self._plot_speed_profile(time_sec, speeds, safe_speeds, output_dir)
        
        # 3. Steering Response Plot
        self._plot_steering_response(time_sec, steering_angles, lateral_errors, output_dir)
        
        # 4. Trajectory Overlay
        self._plot_trajectory_overlay(output_dir)
        
        # 5. Error Distribution Histogram
        self._plot_error_distribution(lateral_errors, output_dir)
        
        # 6. Phase Portrait
        self._plot_phase_portrait(lateral_errors, output_dir)
        
        # 7. Control Effort Distribution
        self._plot_control_distribution(steering_angles, output_dir)
        
        print(f"All plots saved to: {output_dir}/")
    
    def _plot_lateral_error_series(self, time_sec, errors, intervening, output_dir):
        """Plot lateral error over time with intervention zones"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Shade intervention zones
        in_intervention = False
        start_t = None
        for i, (t, interv) in enumerate(zip(time_sec, intervening)):
            if interv and not in_intervention:
                start_t = t
                in_intervention = True
            elif not interv and in_intervention:
                ax.axvspan(start_t, t, alpha=0.2, color='orange', label='Intervention' if start_t == time_sec[0] else '')
                in_intervention = False
        
        # Plot error
        ax.plot(time_sec, errors, 'b-', linewidth=1.5, label='Lateral Error')
        ax.axhline(0.5, color='yellow', linestyle='--', linewidth=1, label='Intervention Threshold')
        ax.axhline(-0.5, color='yellow', linestyle='--', linewidth=1)
        ax.axhline(1.2, color='red', linestyle='--', linewidth=1, label='Warning Threshold')
        ax.axhline(-1.2, color='red', linestyle='--', linewidth=1)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Lateral Error (m)')
        ax.set_title('Lateral Error Time Series with Intervention Zones')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lateral_error_series.png'), dpi=300)
        plt.close()
    
    def _plot_speed_profile(self, time_sec, speeds, safe_speeds, output_dir):
        """Plot actual speed vs safe speed"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Convert to km/h
        speeds_kmh = [s * 3.6 for s in speeds]
        safe_kmh = [s * 3.6 for s in safe_speeds]
        
        ax.plot(time_sec, speeds_kmh, 'b-', linewidth=1.5, label='Actual Speed')
        ax.plot(time_sec, safe_kmh, 'r--', linewidth=1.5, label='Safe Speed')
        ax.fill_between(time_sec, speeds_kmh, safe_kmh, 
                        where=np.array(speeds) > np.array(safe_speeds),
                        alpha=0.3, color='red', label='Overspeed')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (km/h)')
        ax.set_title('Speed Profile vs Safe Speed')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_profile.png'), dpi=300)
        plt.close()
    
    def _plot_steering_response(self, time_sec, steering, errors, output_dir):
        """Dual-axis plot: steering angle and lateral error"""
        import matplotlib.pyplot as plt
        
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        color1 = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Steering Angle (°)', color=color1)
        ax1.plot(time_sec, steering, color=color1, linewidth=1.5, label='Steering')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Lateral Error (m)', color=color2)
        ax2.plot(time_sec, errors, color=color2, linewidth=1.5, alpha=0.7, label='Error')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Steering Response vs Lateral Error')
        fig.tight_layout()
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
    
    def _plot_error_distribution(self, errors, output_dir):
        """Histogram of lateral errors with Gaussian fit"""
        import matplotlib.pyplot as plt
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        errors_arr = np.array(errors)
        mu = np.mean(errors_arr)
        sigma = np.std(errors_arr)
        
        # Histogram
        n, bins, patches = ax.hist(errors_arr, bins=50, density=True, 
                                    alpha=0.7, color='blue', edgecolor='black')
        
        # Gaussian fit
        x = np.linspace(errors_arr.min(), errors_arr.max(), 100)
        gaussian = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}m, σ={sigma:.3f}m')
        
        ax.set_xlabel('Lateral Error (m)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Lateral Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=300)
        plt.close()
    
    def _plot_phase_portrait(self, errors, output_dir):
        """Phase portrait: error vs error rate"""
        import matplotlib.pyplot as plt
        
        errors_arr = np.array(errors)
        # Approximate derivative (error rate)
        dt = 0.2  # sampling interval (5 Hz)
        error_rate = np.gradient(errors_arr, dt)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Color by time (blue -> red)
        colors = np.linspace(0, 1, len(errors_arr))
        scatter = ax.scatter(errors_arr, error_rate, c=colors, cmap='coolwarm', 
                            s=10, alpha=0.6)
        
        # Highlight origin
        ax.plot(0, 0, 'go', markersize=10, label='Stable Point')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        
        ax.set_xlabel('Lateral Error (m)')
        ax.set_ylabel('Error Rate (m/s)')
        ax.set_title('Phase Portrait (e_lat vs de_lat/dt)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
