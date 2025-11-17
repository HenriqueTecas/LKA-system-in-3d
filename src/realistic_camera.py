"""
Realistic Camera Sensor Module - Lane Detection with Pinhole Model and Homography
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
from .config import CAMERA_FRAME_RATE, CAMERA_LATENCY_MS, CAMERA_NOISE_STD, PIXELS_PER_METER


class RealisticCameraSensor:
    """
    Realistic camera sensor with pinhole model and homography-based
    inverse perspective mapping (IPM) for lane detection.
    
    Uses proper camera geometry with:
    - Intrinsic matrix (focal length, principal point)
    - Extrinsic matrix (camera pose: position, pitch, yaw)
    - Homography for ground plane projection
    - Realistic timing (frame rate, latency)
    - Spatial noise model
    """
    
    def __init__(self, car):
        self.car = car
        
        # === Physical Camera Parameters ===
        self.camera_height = 1.3  # meters above ground (BMW E36 windshield height)
        self.pitch_angle = np.radians(10)  # 10 downward tilt (looking at road)
        self.roll_angle = 0.0  # No roll (camera level)
        self.mount_offset_forward = 0.8  # 0.8m ahead of CG (at windshield)
        self.mount_offset_lateral = 0.0  # Centered laterally
        
        # === Image Sensor Parameters ===
        self.image_width = 1280
        self.image_height = 720
        self.horizontal_fov = np.radians(90)  # 90 horizontal FOV (typical dashcam)
        
        # === Intrinsic Camera Matrix ===
        # Calculate focal length from FOV: f = (width/2) / tan(FOV/2)
        self.fx = (self.image_width / 2) / np.tan(self.horizontal_fov / 2)
        self.fy = self.fx  # Square pixels assumed
        self.cx = self.image_width / 2  # Principal point at image center
        self.cy = self.image_height / 2
        
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.K_inv = np.linalg.inv(self.K)
        
        # === Detection Parameters (Based on Real LKA Systems) ===
        # Real automotive LKA cameras have distance-dependent reliability:
        # 0-30m: ~100% confidence (optimal range)
        # 30-40m: 100% → 80% confidence (good range)
        # 40-50m: 80% → 30% confidence (degraded range)
        # 50m+: <30% confidence (unreliable)
        self.min_detection_distance = 1.0  # meters (minimum distance)
        self.optimal_detection_distance = 30.0  # meters (100% confidence)
        self.good_detection_distance = 40.0  # meters (80% confidence)
        self.max_detection_distance = 50.0  # meters (30% confidence)
        self.sample_interval = 1.0  # Sample every 1.0 meters along lane (optimized for performance)
        self.interpolation_interval = 2.0  # Interpolation density (optimized for performance)
        self.detection_max_range = 50.0  # Maximum detection range
        self.detection_min_range = 1.0  # Minimum detection range
        self.pixels_per_meter = float(PIXELS_PER_METER)
        self.use_uniform_sampling = True  # Use optimized uniform sampling
        
        # === Timing and Noise ===
        self.frame_rate = CAMERA_FRAME_RATE  # Hz
        self.frame_interval = 1.0 / float(self.frame_rate) if self.frame_rate > 0 else 0.0
        self.latency = float(CAMERA_LATENCY_MS) / 1000.0  # Convert ms to seconds
        self.noise_std = float(CAMERA_NOISE_STD)  # meters (5cm spatial noise)
        
        self.last_capture_time = 0.0
        self._buffer = []  # Frame buffer for latency simulation
        self.last_measurement = ([], [], [])
        self.rng = np.random.default_rng()
        
        # === Lane Detection State ===
        self.current_lane = "UNKNOWN"
        self.left_lane_detected = False
        self.right_lane_detected = False
        self.left_lane_position = None
        self.right_lane_position = None
        self.lane_center_offset = 0.0
        self.lane_heading_error = 0.0
        
        # Lane change hysteresis to prevent oscillation
        self._lane_change_hysteresis = 2.5  # meters (half lane width - must cross to other lane to switch)
        self._previous_lane = None  # None = first detection, then "LEFT" or "RIGHT"
        
        # Store homography matrices (updated each frame)
        self.H = None
        self.H_inv = None
    
    def get_detection_confidence(self, distance):
        """
        Calculate detection confidence based on distance.
        Models real automotive LKA camera performance.
        
        Args:
            distance: Distance from camera in meters
            
        Returns:
            confidence: 0.0 to 1.0 (probability of correct detection)
        """
        if distance < self.min_detection_distance:
            return 0.0  # Too close, can't see lane markings
        elif distance <= self.optimal_detection_distance:
            # 0-30m: 100% confidence (optimal range)
            return 1.0
        elif distance <= self.good_detection_distance:
            # 30-40m: linear degradation from 100% to 80%
            t = (distance - self.optimal_detection_distance) / \
                (self.good_detection_distance - self.optimal_detection_distance)
            return 1.0 - 0.2 * t
        elif distance <= self.max_detection_distance:
            # 40-50m: linear degradation from 80% to 30%
            t = (distance - self.good_detection_distance) / \
                (self.max_detection_distance - self.good_detection_distance)
            return 0.8 - 0.5 * t
        else:
            # Beyond 50m: poor confidence, exponential decay
            excess = distance - self.max_detection_distance
            return max(0.0, 0.3 * np.exp(-excess / 10.0))
    
    def get_camera_position(self):
        """Get camera position in world coordinates (meters)"""
        cam_x = self.car.x + self.mount_offset_forward * np.cos(self.car.theta)
        cam_y = self.car.y + self.mount_offset_forward * np.sin(self.car.theta)
        cam_z = self.camera_height
        return cam_x, cam_y, cam_z
    
    def compute_homography(self):
        """
        Compute homography matrix from ground plane (Z=0) to image plane.
        
        H maps world coordinates (X_w, Y_w, 1) to image coordinates (u, v, 1).
        H_inv does the reverse: image pixels to world meters.
        
        This is the core of Inverse Perspective Mapping (IPM).
        """
        # Get camera position
        cam_x, cam_y, cam_z = self.get_camera_position()
        
        # === Build Rotation Matrix ===
        # Pitch: rotation around X-axis (camera looking down at road)
        pitch = self.pitch_angle
        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        # Yaw: rotation around Z-axis (car heading direction)
        yaw = self.car.theta
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
    def compute_homography(self):
        """
        Compute homography matrix from ground plane (Z=0) to image plane.
        
        Uses standard camera projection: p_img = K @ [R|t] @ p_world
        For ground plane, this simplifies to a homography.
        """
        # Get camera position
        cam_x, cam_y, cam_z = self.get_camera_position()
        
        # Camera coordinate system (standard computer vision):
        # X_cam = right, Y_cam = down, Z_cam = forward (into scene)
        # World coordinate system (our 2D top-down):  
        # X_world = right, Y_world = forward, Z_world = up
        
        # === Build Rotation from World to Camera ===
        # Step 1: Rotate by vehicle heading (yaw around Z-axis)
        yaw = -self.car.theta  # Negative because we're going world->camera
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Step 2: Rotate by camera pitch (tilt down around X-axis)
        pitch = -self.pitch_angle  # Negative for world->camera
        cp, sp = np.cos(pitch), np.sin(pitch)
        
        # Combined rotation (yaw first in world, then pitch in camera frame)
        # This aligns: world Z->cam Z, then tilts camera down
        R = np.array([
            [cy, -sy * cp, sy * sp],
            [sy, cy * cp, -cy * sp],
            [0, sp, cp]
        ])
        
        # === Build Translation ===
        # Transform camera position from world to camera coordinates
        t = -R @ np.array([[cam_x], [cam_y], [cam_z]])
        
        # === Homography for Ground Plane (Z_world = 0) ===
        # Points on ground: [X_world, Y_world, 0]
        # After transformation: X_cam = r11*X + r12*Y + t1
        #                       Y_cam = r21*X + r22*Y + t2
        #                       Z_cam = r31*X + r32*Y + t3
        # Projection: u = fx * X_cam/Z_cam + cx
        #             v = fy * Y_cam/Z_cam + cy
        # Homography combines these: p_img ~ H @ [X,Y,1]
        
        # H = K @ [r1 r2 t] where r1, r2 are first two columns
        H = self.K @ np.hstack([R[:, 0:1], R[:, 1:2], t])
        
        # Normalize
        if abs(H[2, 2]) > 1e-6:
            H = H / H[2, 2]
        
        self.H = H
        try:
            self.H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.H_inv = None
        
        return H, self.H_inv
        
        return H, self.H_inv
    
    def image_to_world(self, u, v):
        """
        Convert image pixel (u, v) to world coordinates (X_w, Y_w) on road plane.
        
        Args:
            u, v: Pixel coordinates in image
            
        Returns:
            X_w, Y_w: World coordinates in meters (or None, None if invalid)
        """
        # Ensure homography is current
        if self.H_inv is None:
            self.compute_homography()
        
        # Homogeneous image coordinates
        p_img = np.array([u, v, 1.0])
        
        # Apply inverse homography
        p_world_h = self.H_inv @ p_img
        
        # Normalize to get Cartesian coordinates
        if abs(p_world_h[2]) < 1e-6:
            return None, None  # Point at infinity (horizon line)
        
        X_w = p_world_h[0] / p_world_h[2]
        Y_w = p_world_h[1] / p_world_h[2]
        
        return X_w, Y_w
    
    def world_to_image(self, X_w, Y_w):
        """
        Convert world coordinates (X_w, Y_w) to image pixel (u, v).
        
        Args:
            X_w, Y_w: World coordinates in meters
            
        Returns:
            u, v: Pixel coordinates
            visible: True if point is visible in image
        """
        # Compute homography
        if self.H is None:
            self.compute_homography()
        
        # Homogeneous world coordinates (Z=0 for ground)
        p_world = np.array([X_w, Y_w, 1.0])
        
        # Apply homography
        p_img_h = self.H @ p_world
        
        # Normalize to get pixel coordinates
        if abs(p_img_h[2]) < 1e-6:
            return None, None, False
        
        u = p_img_h[0] / p_img_h[2]
        v = p_img_h[1] / p_img_h[2]
        
        # Check visibility: within image bounds and below horizon
        visible = (0 <= u < self.image_width and 
                  0 <= v < self.image_height and
                  v > self.cy)  # Only points below horizon line
        
        return u, v, visible
    
    def detect_lanes(self, track):
        """
        Main detection method with timing simulation.
        Returns lane points in world coordinates (meters).
        
        Simulates:
        - Frame rate (30 Hz capture)
        - Latency (100ms delay)
        - Spatial noise (5cm std)
        """
        now = time.perf_counter()
        
        # Capture new frame if interval elapsed
        if (self.last_capture_time == 0.0 or 
            now - self.last_capture_time >= self.frame_interval):
            
            # Perform detection using homography
            left, center, right = self._perform_detection_realistic(track)
            
            # Add noise
            left_noisy = self._add_noise(left)
            center_noisy = self._add_noise(center)
            right_noisy = self._add_noise(right)
            
            # Buffer with latency
            release_time = now + self.latency
            self._buffer.append({
                'release': release_time,
                'data': (left_noisy, center_noisy, right_noisy)
            })
            self.last_capture_time = now
        
        # Get most recent ready frame
        ready = [b for b in self._buffer if b['release'] <= now]
        if ready:
            item = ready[-1]
            self._buffer = [b for b in self._buffer if b['release'] > item['release']]
            
            left_pts, center_pts, right_pts = item['data']
            
            # Determine current lane based on distance to lane boundaries
            cam_x, cam_y, _ = self.get_camera_position()
            
            # Calculate distance to each lane's center
            # Left lane center is between left boundary and centerline
            # Right lane center is between centerline and right boundary
            
            if len(left_pts) > 0 and len(center_pts) > 0 and len(right_pts) > 0:
                # Get closest points on each boundary THAT ARE NEAR THE CAR'S LATERAL POSITION
                # We want points that are ahead but close to the car, not far ahead
                # Filter points within reasonable forward distance (e.g., 10m ahead)
                max_forward_dist = 10.0  # meters
                
                def get_laterally_close_point(boundary_pts):
                    """Get point closest laterally (perpendicular to car direction)"""
                    nearby = [p for p in boundary_pts if np.hypot(p[0]-cam_x, p[1]-cam_y) < max_forward_dist]
                    if not nearby:
                        nearby = boundary_pts  # fallback
                    # Return closest by distance
                    return min(nearby, key=lambda p: np.hypot(p[0]-cam_x, p[1]-cam_y))
                
                left_close = get_laterally_close_point(left_pts)
                center_close = get_laterally_close_point(center_pts)
                right_close = get_laterally_close_point(right_pts)
                
                # Calculate lane centers
                left_lane_center_x = (left_close[0] + center_close[0]) / 2
                left_lane_center_y = (left_close[1] + center_close[1]) / 2
                right_lane_center_x = (center_close[0] + right_close[0]) / 2
                right_lane_center_y = (center_close[1] + right_close[1]) / 2
                
                # Distance to each lane center
                dist_to_left_lane = np.hypot(cam_x - left_lane_center_x, cam_y - left_lane_center_y)
                dist_to_right_lane = np.hypot(cam_x - right_lane_center_x, cam_y - right_lane_center_y)
                
                # First detection: pick closest lane
                if self._previous_lane is None:
                    if dist_to_left_lane < dist_to_right_lane:
                        self.current_lane = "LEFT"
                    else:
                        self.current_lane = "RIGHT"
                # Apply hysteresis: only switch if other lane is significantly closer
                elif self._previous_lane == "LEFT":
                    if dist_to_right_lane < (dist_to_left_lane - self._lane_change_hysteresis):
                        self.current_lane = "RIGHT"
                    else:
                        self.current_lane = "LEFT"
                elif self._previous_lane == "RIGHT":
                    if dist_to_left_lane < (dist_to_right_lane - self._lane_change_hysteresis):
                        self.current_lane = "LEFT"
                    else:
                        self.current_lane = "RIGHT"
                
                self._previous_lane = self.current_lane
            else:
                # Fallback if boundaries missing
                self.current_lane = "UNKNOWN"
            
            if self.current_lane == "LEFT":
                self.left_lane_detected = len(left_pts) > 0
                self.right_lane_detected = len(center_pts) > 0
            else:
                self.left_lane_detected = len(center_pts) > 0
                self.right_lane_detected = len(right_pts) > 0
            
            # Always return all 3 boundaries for controllers to use
            self.last_measurement = (left_pts, center_pts, right_pts)
            
            # Calculate errors based on current lane
            if self.current_lane == "LEFT":
                self._calculate_lane_tracking_errors(left_pts, center_pts)
            else:
                self._calculate_lane_tracking_errors(center_pts, right_pts)
            
            return left_pts, center_pts, right_pts
        
        return self.last_measurement
    
    def _perform_detection_realistic(self, track):
        """
        Detect lane boundaries using perspective projection and homography.
        
        This is where the realistic camera model comes into play:
        1. Get lane boundaries in world coordinates
        2. Project each point to image using homography
        3. Filter visible points (within FOV, below horizon)
        4. Sample uniformly along visible sections
        """
        # Get camera position
        cam_x, cam_y, cam_z = self.get_camera_position()
        camera_yaw = self.car.theta
        
        # Get lane boundaries in world pixels from track
        # NOTE: Track boundaries appear to be swapped - negative offset gives RIGHT, positive gives LEFT
        right_boundary_px = track._offset_line(track.centerline, -track.lane_width)  # SWAPPED!
        center_line_px = track.centerline
        left_boundary_px = track._offset_line(track.centerline, track.lane_width)  # SWAPPED!
        
        # Convert from pixels to meters
        left_boundary = [(x/self.pixels_per_meter, y/self.pixels_per_meter) 
                        for x, y in left_boundary_px]
        center_line = [(x/self.pixels_per_meter, y/self.pixels_per_meter) 
                      for x, y in center_line_px]
        right_boundary = [(x/self.pixels_per_meter, y/self.pixels_per_meter) 
                         for x, y in right_boundary_px]
        
        # Detect each boundary using optimized sampling
        left_detected = self._detect_boundary(left_boundary, cam_x, cam_y, cam_z, camera_yaw)
        center_detected = self._detect_boundary(center_line, cam_x, cam_y, cam_z, camera_yaw)
        right_detected = self._detect_boundary(right_boundary, cam_x, cam_y, cam_z, camera_yaw)
        
        return left_detected, center_detected, right_detected
    
    def _detect_boundary(self, boundary_points, camera_x, camera_y, camera_z, camera_yaw):
        """
        Detect visible lane boundary points using optimized sampling.
        Returns list of (x, y, angle, confidence) tuples with distance-based confidence.
        """
        if self.use_uniform_sampling:
            return self._detect_boundary_uniform(boundary_points, camera_x, camera_y, camera_yaw)
        else:
            return self._detect_boundary_all(boundary_points, camera_x, camera_y, camera_yaw)
    
    def _detect_boundary_all(self, boundary_points, camera_x, camera_y, camera_yaw):
        """
        Original method: detect all visible points with confidence.
        Returns list of (x, y, angle, confidence) tuples.
        """
    
    def _interpolate_boundary(self, boundary_points, interpolation_interval=2.0):
        """Interpolate sparse boundary points to create dense sampling.
        
        Args:
            boundary_points: List of (x, y) tuples in meters (sparse, from track)
            interpolation_interval: Distance between interpolated points in meters
            
        Returns:
            List of (x, y) tuples with dense sampling
        """
        if len(boundary_points) < 2:
            return list(boundary_points)
        
        dense_points = []
        
        for i in range(len(boundary_points) - 1):
            p1 = np.array(boundary_points[i])
            p2 = np.array(boundary_points[i + 1])
            
            segment_vec = p2 - p1
            segment_length = np.linalg.norm(segment_vec)
            
            if segment_length < 0.01:  # Skip tiny segments
                continue
            
            # Number of interpolated points for this segment
            num_points = max(1, int(segment_length / interpolation_interval))
            
            # Add interpolated points
            for j in range(num_points):
                t = j / num_points
                interp_point = p1 + t * segment_vec
                dense_points.append((float(interp_point[0]), float(interp_point[1])))
        
        # Add final point
        dense_points.append((float(boundary_points[-1][0]), float(boundary_points[-1][1])))
        
        return dense_points
    
    def _detect_boundary_uniform(self, boundary_points, camera_x, camera_y, camera_yaw):
        """Optimized detection with interpolation and uniform sampling.
        Returns list of (x, y, angle, confidence) tuples.
        """
        # STEP 1: Filter to nearby track points
        max_process_range = self.detection_max_range * 5.0  # 250m search radius
        nearby_points = []
        nearby_indices = []
        
        for i, (px, py) in enumerate(boundary_points):
            dist = np.sqrt((px - camera_x)**2 + (py - camera_y)**2)
            if dist < max_process_range:
                nearby_points.append((px, py))
                nearby_indices.append(i)
        
        # Expand to include adjacent segments
        if len(nearby_indices) > 0:
            min_idx = max(0, min(nearby_indices) - 2)
            max_idx = min(len(boundary_points), max(nearby_indices) + 3)
            nearby_points = boundary_points[min_idx:max_idx]
        
        if len(nearby_points) < 2:
            return []
        
        # STEP 2: Interpolate nearby points
        dense_points = self._interpolate_boundary(nearby_points, interpolation_interval=self.interpolation_interval)
        
        # STEP 3: Filter by range and calculate angles/confidence
        visible_points = []
        
        for px, py in dense_points:
            dx = px - camera_x
            dy = py - camera_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Check range
            if distance < self.detection_min_range or distance > self.detection_max_range:
                continue
            
            # Calculate angle relative to camera
            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - camera_yaw
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            
            # Simple FOV check (90 degrees = ±45 degrees)
            fov_half = np.radians(45)
            if abs(angle_diff) > fov_half:
                continue
            
            # Calculate confidence based on distance
            confidence = self.get_detection_confidence(distance)
            
            visible_points.append((px, py, angle_diff, confidence))
        
        if len(visible_points) == 0:
            return []
        
        # STEP 4: Uniform sampling
        points_with_distance = []
        cumulative_distance = 0.0
        
        for i, (px, py, ang, conf) in enumerate(visible_points):
            if i > 0:
                prev_px, prev_py, _, _ = visible_points[i-1]
                segment_length = np.sqrt((px - prev_px)**2 + (py - prev_py)**2)
                cumulative_distance += segment_length
            
            points_with_distance.append((px, py, ang, conf, cumulative_distance))
        
        # Sample at uniform intervals
        total_length = points_with_distance[-1][4]
        num_samples = max(1, int(total_length / self.sample_interval))
        
        sampled_points = []
        for i in range(num_samples + 1):
            target_distance = i * self.sample_interval
            
            # Find closest point
            best_point = min(points_with_distance, key=lambda p: abs(p[4] - target_distance))
            px, py, ang, conf, _ = best_point
            
            # Avoid duplicates
            if not any(abs(sp[0] - px) < 0.05 and abs(sp[1] - py) < 0.05 for sp in sampled_points):
                sampled_points.append((px, py, ang, conf))
        
        return sampled_points
    
    def _add_noise(self, points):
        """Apply spatial noise to detected points, scaled by confidence"""
        if not points:
            return []
        
        noisy = []
        for point_data in points:
            if len(point_data) == 4:
                px, py, ang, conf = point_data
            else:
                # Fallback for 3-tuple
                px, py, ang = point_data
                conf = 1.0
            
            # Scale noise inversely with confidence (high conf = low noise)
            noise_scale = 1.0 - 0.5 * conf  # 100% conf → 0.5x noise, 0% conf → 1.0x noise
            nx = px + float(self.rng.normal(0.0, self.noise_std * noise_scale))
            ny = py + float(self.rng.normal(0.0, self.noise_std * noise_scale))
            nang = ang + float(self.rng.normal(0.0, max(1e-3, self.noise_std * 0.01 * noise_scale)))
            noisy.append((nx, ny, nang, conf))
        return noisy
    
    def _get_lateral_offset(self, track, x, y):
        """Calculate lateral offset from track centerline (for lane determination)"""
        # Convert position to pixels for track comparison
        x_px = x * self.pixels_per_meter
        y_px = y * self.pixels_per_meter
        
        # Find closest centerline point
        min_dist = float('inf')
        closest_idx = 0
        
        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((x_px - cx)**2 + (y_px - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Compute perpendicular offset
        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]
        
        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)
        
        to_point_x = x_px - p_curr[0]
        to_point_y = y_px - p_curr[1]
        
        perp_angle = track_angle + np.pi / 2
        lateral_offset_px = (to_point_x * np.cos(perp_angle) + 
                           to_point_y * np.sin(perp_angle))
        
        return lateral_offset_px / self.pixels_per_meter
    
    def _calculate_lane_tracking_errors(self, left_points, right_points):
        """Calculate lateral offset and heading error from lane center"""
        if not left_points or not right_points:
            return
        
        # Find closest points (smallest angle) - handle 4-tuple format
        left_closest = min(left_points, key=lambda p: abs(p[2]))
        right_closest = min(right_points, key=lambda p: abs(p[2]))
        
        left_angle = left_closest[2]
        right_angle = right_closest[2]
        
        # Lane center is midpoint between left and right
        self.lane_center_offset = (right_angle + left_angle) / 2
        self.lane_heading_error = self.lane_center_offset
        
        # Store lane positions
        self.left_lane_position = left_angle
        self.right_lane_position = right_angle
    
    def get_field_of_view(self):
        """Return horizontal FOV for visualization"""
        return self.horizontal_fov
    
    @property
    def field_of_view(self):
        """Compatibility property for existing code"""
        return self.horizontal_fov
    
    @property
    def max_range(self):
        """Compatibility property for existing code"""
        return self.max_detection_distance
