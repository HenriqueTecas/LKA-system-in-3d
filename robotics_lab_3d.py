"""
Robotics Lab Assignment 1 - 3D OpenGL Version
Task 1: Lane Tracing Assist (LTA) Simulation - First-Person View
Task 2b: Camera Sensor Model for Lane Detection
Task 2c: Pure Pursuit Lane Keeping Assist (LKA) Controller

3D rendering with hood camera view and minimap.
Preserves all Ackermann kinematics and Pure Pursuit logic from original.

Controls:
- W: Accelerate
- S: Brake/Reverse
- A: Steer left (manual - deactivates LKA)
- D: Steer right (manual - deactivates LKA)
- F: Toggle LKA on/off
- ESC: Exit
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os

# Set environment variables for better OpenGL compatibility
os.environ['SDL_VIDEO_X11_FORCE_EGL'] = '0'  # Disable EGL, use GLX instead
os.environ['PYOPENGL_PLATFORM'] = 'glx'  # Force GLX platform

# Check if display is available
if 'DISPLAY' not in os.environ:
    print("ERROR: No display found!")
    print("Solutions:")
    print("1. If running over SSH: use 'ssh -X' for X11 forwarding")
    print("2. If local: ensure you're running in a graphical environment")
    print("3. For headless: install and use xvfb-run: 'xvfb-run python robotics_lab_3d.py'")
    sys.exit(1)

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH = 1600
HEIGHT = 900
MINIMAP_SIZE = 500  # Size of minimap in top-right corner (increased for better visibility)

# Try to create OpenGL context with fallback options
screen = None
error_messages = []

# Try different display modes in order of preference
display_configs = [
    (DOUBLEBUF | OPENGL, "Double-buffered OpenGL"),
    (OPENGL, "Single-buffered OpenGL"),
    (DOUBLEBUF | OPENGL | HWSURFACE, "Hardware-accelerated OpenGL"),
]

for flags, description in display_configs:
    try:
        print(f"Trying {description}...")
        screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
        pygame.display.set_caption("Lab 1 - 3D Car Simulation with Hood View")
        print(f"✓ Successfully initialized with {description}")
        break
    except pygame.error as e:
        error_messages.append(f"  {description}: {e}")
        continue

if screen is None:
    print("\nERROR: Could not initialize OpenGL display!")
    print("\nTried the following configurations:")
    for msg in error_messages:
        print(msg)
    print("\nPossible solutions:")
    print("1. Check OpenGL drivers: 'glxinfo | grep OpenGL'")
    print("2. Install mesa-utils: 'sudo apt-get install mesa-utils'")
    print("3. Install required GL libraries: 'sudo apt-get install libgl1-mesa-glx libglu1-mesa'")
    print("4. For virtual display: 'sudo apt-get install xvfb && xvfb-run -s \"-screen 0 1920x1080x24\" python robotics_lab_3d.py'")
    print("5. Try software rendering: 'export LIBGL_ALWAYS_SOFTWARE=1'")
    sys.exit(1)

# Colors (for minimap and UI)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60


class Car:
    """Car with Ackermann steering kinematics - identical to original"""
    def __init__(self, x, y, theta):
        # Position and orientation
        self.x = x
        self.y = y
        self.theta = theta  # heading angle (radians)

        # Car dimensions
        self.length = 40  # car length (pixels)
        self.width = 20   # car width (pixels)
        self.wheelbase = 30  # L in the kinematic model

        # Kinematic state
        self.velocity = 0.0  # V - linear velocity
        self.steering_angle = 0.0  # φ (phi) - steering angle (radians)

        # Control parameters
        self.max_velocity = 120.0
        self.max_steering_angle = np.radians(35)
        self.acceleration = 50.0
        self.deceleration = 100.0
        self.steering_rate = np.radians(60)
        self.friction = 30.0

        # 3D rendering properties
        self.height = 15  # car height for 3D
        self.hood_height = 10  # camera mount height

    def update(self, dt, keys, lka_steering=None, lka_controller=None):
        """Update car state based on Ackermann steering model"""
        # Handle acceleration
        if keys[pygame.K_w]:
            self.velocity += self.acceleration * dt
        elif keys[pygame.K_s]:
            self.velocity -= self.deceleration * dt
        else:
            # Apply friction
            if self.velocity > 0:
                self.velocity -= self.friction * dt
                if self.velocity < 0:
                    self.velocity = 0
            elif self.velocity < 0:
                self.velocity += self.friction * dt
                if self.velocity > 0:
                    self.velocity = 0

        # Limit velocity
        self.velocity = np.clip(self.velocity, -self.max_velocity * 0.5, self.max_velocity)

        # Handle steering
        manual_steering = keys[pygame.K_a] or keys[pygame.K_d]

        if manual_steering:
            if lka_controller and lka_controller.active:
                lka_controller.deactivate()

            if keys[pygame.K_a]:
                self.steering_angle += self.steering_rate * dt  # Turn LEFT (SWAPPED for 3D view)
            elif keys[pygame.K_d]:
                self.steering_angle -= self.steering_rate * dt  # Turn RIGHT (SWAPPED for 3D view)
        elif lka_steering is not None:
            self.steering_angle = lka_steering
        else:
            if abs(self.steering_angle) > 0.01:
                self.steering_angle *= 0.9
            else:
                self.steering_angle = 0

        # Limit steering angle
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # Ackermann steering kinematics
        if abs(self.velocity) > 0.1:
            omega = self.velocity * np.tan(self.steering_angle) / self.wheelbase

            # Store previous position for collision handling
            prev_x, prev_y = self.x, self.y

            # Update position and orientation
            self.x += self.velocity * np.cos(self.theta) * dt
            self.y += self.velocity * np.sin(self.theta) * dt
            self.theta += omega * dt
            self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

            # Check for collision and revert if off-track (handled in main loop)
            self.prev_x = prev_x
            self.prev_y = prev_y

    def get_front_axle_position(self):
        """Return front axle center position"""
        front_axle_x = self.x + (self.length/2 - 5) * np.cos(self.theta)
        front_axle_y = self.y + (self.length/2 - 5) * np.sin(self.theta)
        return front_axle_x, front_axle_y

    def get_front_wheel_positions(self):
        """Return left and right front wheel centers"""
        front_axle_x, front_axle_y = self.get_front_axle_position()
        wheel_angle = self.theta + np.pi/2
        wheel_half_width = self.width / 2

        left_wheel_x = front_axle_x + wheel_half_width * np.cos(wheel_angle)
        left_wheel_y = front_axle_y + wheel_half_width * np.sin(wheel_angle)
        right_wheel_x = front_axle_x - wheel_half_width * np.cos(wheel_angle)
        right_wheel_y = front_axle_y - wheel_half_width * np.sin(wheel_angle)

        return (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y)

    def get_hood_camera_position(self):
        """Get position and orientation for hood camera"""
        # Camera at front of car, slightly above hood
        cam_x = self.x + (self.length/2 - 10) * np.cos(self.theta)
        cam_y = self.y + (self.length/2 - 10) * np.sin(self.theta)
        cam_z = self.hood_height

        # Look-at point ahead of car
        look_distance = 50
        look_x = self.x + look_distance * np.cos(self.theta)
        look_y = self.y + look_distance * np.sin(self.theta)
        look_z = self.hood_height

        return (cam_x, cam_y, cam_z), (look_x, look_y, look_z)

    def draw_3d(self):
        """Draw car in 3D"""
        glPushMatrix()

        # Transform to car position and orientation
        glTranslatef(self.x, self.y, self.height/2)
        glRotatef(np.degrees(self.theta), 0, 0, 1)

        # Draw car body (simple box)
        glColor3f(0.2, 0.5, 0.8)  # Blue car
        self._draw_box(self.length, self.width, self.height)

        # Draw hood (front part, slightly higher)
        glPushMatrix()
        glTranslatef(self.length/4, 0, self.height/3)
        glColor3f(0.3, 0.6, 0.9)
        self._draw_box(self.length/2, self.width*0.8, self.height/3)
        glPopMatrix()

        # Draw wheels
        self._draw_wheels()

        glPopMatrix()

    def _draw_box(self, length, width, height):
        """Draw a simple box centered at origin"""
        l, w, h = length/2, width/2, height/2

        glBegin(GL_QUADS)

        # Front face
        glVertex3f(l, -w, -h)
        glVertex3f(l, w, -h)
        glVertex3f(l, w, h)
        glVertex3f(l, -w, h)

        # Back face
        glVertex3f(-l, -w, -h)
        glVertex3f(-l, -w, h)
        glVertex3f(-l, w, h)
        glVertex3f(-l, w, -h)

        # Top face
        glVertex3f(-l, -w, h)
        glVertex3f(l, -w, h)
        glVertex3f(l, w, h)
        glVertex3f(-l, w, h)

        # Bottom face
        glVertex3f(-l, -w, -h)
        glVertex3f(-l, w, -h)
        glVertex3f(l, w, -h)
        glVertex3f(l, -w, -h)

        # Right face
        glVertex3f(-l, w, -h)
        glVertex3f(-l, w, h)
        glVertex3f(l, w, h)
        glVertex3f(l, w, -h)

        # Left face
        glVertex3f(-l, -w, -h)
        glVertex3f(l, -w, -h)
        glVertex3f(l, -w, h)
        glVertex3f(-l, -w, h)

        glEnd()

    def _draw_wheels(self):
        """Draw car wheels"""
        wheel_radius = 4
        wheel_width = 3

        glColor3f(0.1, 0.1, 0.1)  # Dark wheels

        # Wheel positions relative to car center
        wheel_positions = [
            (self.length/2 - 5, self.width/2, 0),      # Front left
            (self.length/2 - 5, -self.width/2, 0),     # Front right
            (-self.length/2 + 5, self.width/2, 0),     # Rear left
            (-self.length/2 + 5, -self.width/2, 0),    # Rear right
        ]

        for i, (wx, wy, wz) in enumerate(wheel_positions):
            glPushMatrix()
            glTranslatef(wx, wy, wz)

            # Front wheels have steering angle
            if i < 2:
                glRotatef(np.degrees(self.steering_angle), 0, 0, 1)

            # Draw wheel as cylinder
            glRotatef(90, 0, 1, 0)
            self._draw_cylinder(wheel_radius, wheel_width, 8)

            glPopMatrix()

    def _draw_cylinder(self, radius, height, slices):
        """Draw a simple cylinder"""
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, -height/2)
            glVertex3f(x, y, height/2)
        glEnd()

    def is_on_track(self, track):
        """Check if car is within track boundaries"""
        # Find closest point on centerline
        min_dist = float('inf')
        closest_idx = 0

        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((self.x - cx)**2 + (self.y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Get track direction at closest point
        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]

        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)

        # Calculate perpendicular distance from track center
        to_car_x = self.x - p_curr[0]
        to_car_y = self.y - p_curr[1]

        perp_angle = track_angle + np.pi / 2
        lateral_distance = abs(to_car_x * np.cos(perp_angle) + to_car_y * np.sin(perp_angle))

        # Check if within track width (MORE FORGIVING - added extra margin)
        # Allow car to go slightly beyond visual track edge before collision
        max_distance = track.track_width / 2 + self.width  # Extra margin added
        return lateral_distance <= max_distance

    def handle_collision(self):
        """Handle collision by reverting to previous position and stopping"""
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            self.x = self.prev_x
            self.y = self.prev_y
            self.velocity = 0  # Stop the car


class CameraSensor:
    """Camera sensor for lane detection - identical logic to original"""
    def __init__(self, car):
        self.car = car
        self.field_of_view = np.radians(80)
        self.max_range = 300
        self.min_range = 20
        self.image_width = 1280
        self.image_height = 720
        self.mount_offset = self.car.length * 0.10
        self.detection_confidence = 0.95
        self.lane_sample_points = 10

        self.left_lane_detected = False
        self.right_lane_detected = False
        self.left_lane_position = None
        self.right_lane_position = None
        self.lane_center_offset = 0.0
        self.lane_heading_error = 0.0
        self.current_lane = "UNKNOWN"

    def get_camera_position(self):
        """Get camera world position"""
        camera_x = self.car.x + self.mount_offset * np.cos(self.car.theta)
        camera_y = self.car.y + self.mount_offset * np.sin(self.car.theta)
        return camera_x, camera_y

    def detect_lanes(self, track):
        """Detect lane lines - same logic as original"""
        camera_x, camera_y = self.get_camera_position()
        camera_angle = self.car.theta

        # Get track boundaries
        left_outer_boundary = track._offset_line(track.centerline, -track.lane_width)
        center_boundary = track.centerline
        right_outer_boundary = track._offset_line(track.centerline, track.lane_width)

        # Detect boundaries
        left_outer_points = self._detect_lane_boundary(
            left_outer_boundary, camera_x, camera_y, camera_angle
        )
        center_points = self._detect_lane_boundary(
            center_boundary, camera_x, camera_y, camera_angle
        )
        right_outer_points = self._detect_lane_boundary(
            right_outer_boundary, camera_x, camera_y, camera_angle
        )

        # Determine current lane
        car_lateral_offset = self._get_lateral_offset_from_track_center(track)

        if car_lateral_offset < 0:
            left_lane_points = left_outer_points
            right_lane_points = center_points
            current_lane = "LEFT"
        else:
            left_lane_points = center_points
            right_lane_points = right_outer_points
            current_lane = "RIGHT"

        self.left_lane_detected = len(left_lane_points) > 0
        self.right_lane_detected = len(right_lane_points) > 0
        self.current_lane = current_lane

        if self.left_lane_detected and len(left_lane_points) > 0:
            self.left_lane_position = self._calculate_lane_position(left_lane_points[0])

        if self.right_lane_detected and len(right_lane_points) > 0:
            self.right_lane_position = self._calculate_lane_position(right_lane_points[0])

        self._calculate_lane_tracking_errors(left_lane_points, right_lane_points)

        return left_lane_points, right_lane_points, center_points

    def _detect_lane_boundary(self, boundary_points, camera_x, camera_y, camera_angle):
        """Detect visible lane boundary points"""
        visible_points = []

        for point in boundary_points:
            px, py = point
            dx = px - camera_x
            dy = py - camera_y
            distance = np.sqrt(dx**2 + dy**2)

            if distance < self.min_range or distance > self.max_range:
                continue

            point_angle = np.arctan2(dy, dx)
            angle_diff = point_angle - camera_angle
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            if abs(angle_diff) < self.field_of_view / 2:
                visible_points.append((px, py, angle_diff))

        return visible_points

    def _calculate_lane_position(self, point_data):
        """Calculate lane position (angle only)"""
        px, py, angle = point_data
        return angle

    def _get_lateral_offset_from_track_center(self, track):
        """Calculate lateral offset from track centerline"""
        min_dist = float('inf')
        closest_idx = 0

        for i, (cx, cy) in enumerate(track.centerline):
            dist = np.sqrt((self.car.x - cx)**2 + (self.car.y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        p_curr = track.centerline[closest_idx]
        p_next = track.centerline[(closest_idx + 1) % len(track.centerline)]

        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        track_angle = np.arctan2(dy, dx)

        to_car_x = self.car.x - p_curr[0]
        to_car_y = self.car.y - p_curr[1]

        perp_angle = track_angle + np.pi / 2
        lateral_offset = (to_car_x * np.cos(perp_angle) +
                         to_car_y * np.sin(perp_angle))

        return lateral_offset

    def _calculate_lane_tracking_errors(self, left_points, right_points):
        """Calculate lateral offset and heading error"""
        if not left_points or not right_points:
            return

        left_closest = min(left_points, key=lambda p: abs(p[2]))
        right_closest = min(right_points, key=lambda p: abs(p[2]))

        left_angle = left_closest[2]
        right_angle = right_closest[2]

        self.lane_center_offset = (right_angle + left_angle) / 2
        self.lane_heading_error = self.lane_center_offset


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
        """Pure Pursuit algorithm"""
        if not self.active:
            return None

        left_lane, right_lane, center_lane = self.camera.detect_lanes(track)

        if not (self.camera.left_lane_detected and self.camera.right_lane_detected):
            return None

        speed = abs(self.car.velocity)
        lookahead_distance = self.base_lookahead_distance + self.lookahead_gain * speed
        lookahead_distance = np.clip(lookahead_distance, self.min_lookahead, self.max_lookahead)

        car_x = self.car.x
        car_y = self.car.y
        car_theta = self.car.theta

        # Calculate lane center points
        lane_center_points = []
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


class SaoPauloTrack:
    """São Paulo F1 Circuit - identical to original"""
    def __init__(self, offset_x=100, offset_y=100):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.lane_width = 50
        self.track_width = 2 * self.lane_width

        scale = 1.0
        self.centerline = [
            (800, 600), (750, 500), (650, 400), (550, 350), (450, 330),
            (350, 300), (250, 250), (200, 180), (180, 120), (200, 60),
            (300, 30), (500, 30), (700, 30), (900, 30), (1100, 50),
            (1150, 100), (1180, 200), (1180, 300), (1180, 400), (1150, 500),
            (1100, 550), (1000, 600), (900, 600), (800, 600),
        ]

        self.centerline = [(x * scale + offset_x, y * scale + offset_y)
                          for x, y in self.centerline]

    def _offset_line(self, points, offset):
        """Offset a line perpendicular to its direction"""
        offset_points = []

        for i in range(len(points)):
            p_prev = points[i - 1] if i > 0 else points[-1]
            p_curr = points[i]
            p_next = points[(i + 1) % len(points)]

            dx1 = p_curr[0] - p_prev[0]
            dy1 = p_curr[1] - p_prev[1]
            len1 = np.sqrt(dx1**2 + dy1**2) or 1

            dx2 = p_next[0] - p_curr[0]
            dy2 = p_next[1] - p_curr[1]
            len2 = np.sqrt(dx2**2 + dy2**2) or 1

            perp_x = -(dy1/len1 + dy2/len2) / 2
            perp_y = (dx1/len1 + dx2/len2) / 2
            perp_len = np.sqrt(perp_x**2 + perp_y**2) or 1

            offset_x = p_curr[0] + (perp_x / perp_len) * offset
            offset_y = p_curr[1] + (perp_y / perp_len) * offset

            offset_points.append((offset_x, offset_y))

        return offset_points

    def get_start_position(self, lane_number=1):
        """Get starting position"""
        start_point = self.centerline[0]
        next_point = self.centerline[1]

        dx = next_point[0] - start_point[0]
        dy = next_point[1] - start_point[1]
        theta = np.arctan2(dy, dx)

        perp_angle = theta + np.pi / 2
        if lane_number == 1:
            offset = -self.lane_width / 2
        else:
            offset = self.lane_width / 2

        x = start_point[0] + offset * np.cos(perp_angle)
        y = start_point[1] + offset * np.sin(perp_angle)

        return x, y, theta

    def draw_3d(self):
        """Draw track in 3D"""
        # Draw road surface
        self._draw_road_surface()

        # Draw lane markings
        self._draw_lane_markings()

        # Draw surrounding terrain
        self._draw_terrain()

        # Draw visual features (checkpoints, arrows, sectors)
        self._draw_track_features()

        # Draw scenery elements (trees, signs, buildings)
        self._draw_scenery()

    def _draw_road_surface(self):
        """Draw flat road surface with subtle texture pattern"""
        # Draw road as triangulated strips with alternating shades for depth
        outer_points = self._offset_line(self.centerline, self.track_width / 2)
        inner_points = self._offset_line(self.centerline, -self.track_width / 2)

        glBegin(GL_TRIANGLE_STRIP)
        for i in range(len(outer_points)):
            # Alternate between two subtle shades of gray for texture
            if i % 3 == 0:
                glColor3f(0.28, 0.28, 0.28)  # Slightly darker
            elif i % 3 == 1:
                glColor3f(0.30, 0.30, 0.30)  # Base gray
            else:
                glColor3f(0.32, 0.32, 0.32)  # Slightly lighter

            ox, oy = outer_points[i]
            ix, iy = inner_points[i]
            glVertex3f(ox, oy, 0)
            glVertex3f(ix, iy, 0)
        # Close the loop
        ox, oy = outer_points[0]
        ix, iy = inner_points[0]
        glVertex3f(ox, oy, 0)
        glVertex3f(ix, iy, 0)
        glEnd()

    def _draw_lane_markings(self):
        """Draw lane markings on road"""
        glLineWidth(3)

        # Outer boundaries (solid white)
        glColor3f(1.0, 1.0, 1.0)
        outer_points = self._offset_line(self.centerline, self.track_width / 2)
        inner_points = self._offset_line(self.centerline, -self.track_width / 2)

        glBegin(GL_LINE_STRIP)
        for x, y in outer_points:
            glVertex3f(x, y, 0.1)
        glVertex3f(outer_points[0][0], outer_points[0][1], 0.1)
        glEnd()

        glBegin(GL_LINE_STRIP)
        for x, y in inner_points:
            glVertex3f(x, y, 0.1)
        glVertex3f(inner_points[0][0], inner_points[0][1], 0.1)
        glEnd()

        # Center line (dashed yellow)
        glColor3f(1.0, 1.0, 0.0)
        dash_length = 20
        gap_length = 15

        total_length = 0
        for i in range(len(self.centerline)):
            p1 = self.centerline[i]
            p2 = self.centerline[(i + 1) % len(self.centerline)]
            seg_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

            if seg_length > 0:
                dx = (p2[0] - p1[0]) / seg_length
                dy = (p2[1] - p1[1]) / seg_length

                seg_pos = 0
                while seg_pos < seg_length:
                    pattern_pos = (total_length + seg_pos) % (dash_length + gap_length)

                    if pattern_pos < dash_length:
                        dash_start = seg_pos
                        dash_end = min(seg_pos + (dash_length - pattern_pos), seg_length)

                        x1 = p1[0] + dx * dash_start
                        y1 = p1[1] + dy * dash_start
                        x2 = p1[0] + dx * dash_end
                        y2 = p1[1] + dy * dash_end

                        glBegin(GL_LINES)
                        glVertex3f(x1, y1, 0.1)
                        glVertex3f(x2, y2, 0.1)
                        glEnd()

                        seg_pos = dash_end
                    else:
                        seg_pos += (dash_length + gap_length - pattern_pos)

                total_length += seg_length

    def _draw_terrain(self):
        """Draw elevated terrain around track with textured pattern"""
        # Create terrain boundary (offset further from track)
        terrain_offset = 200
        outer_terrain = self._offset_line(self.centerline, self.track_width / 2 + terrain_offset)
        inner_terrain = self._offset_line(self.centerline, -self.track_width / 2 - terrain_offset)
        outer_track = self._offset_line(self.centerline, self.track_width / 2)
        inner_track = self._offset_line(self.centerline, -self.track_width / 2)

        terrain_height = 30

        # Draw outer terrain wall with alternating stripe pattern
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(len(outer_terrain)):
            # Alternate between two shades of green for stripe pattern
            if i % 2 == 0:
                glColor3f(0.2, 0.5, 0.2)  # Darker green
            else:
                glColor3f(0.25, 0.55, 0.25)  # Lighter green

            tx, ty = outer_terrain[i]
            rx, ry = outer_track[i]
            glVertex3f(rx, ry, 0)
            glVertex3f(tx, ty, terrain_height)
        # Close loop
        tx, ty = outer_terrain[0]
        rx, ry = outer_track[0]
        glVertex3f(rx, ry, 0)
        glVertex3f(tx, ty, terrain_height)
        glEnd()

        # Draw inner terrain wall with alternating stripe pattern
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(len(inner_terrain)):
            # Alternate between two shades of green for stripe pattern
            if i % 2 == 0:
                glColor3f(0.2, 0.5, 0.2)  # Darker green
            else:
                glColor3f(0.25, 0.55, 0.25)  # Lighter green

            tx, ty = inner_terrain[i]
            rx, ry = inner_track[i]
            glVertex3f(rx, ry, 0)
            glVertex3f(tx, ty, terrain_height)
        # Close loop
        tx, ty = inner_terrain[0]
        rx, ry = inner_track[0]
        glVertex3f(rx, ry, 0)
        glVertex3f(tx, ty, terrain_height)
        glEnd()

        # Draw terrain top surface with grid pattern
        glBegin(GL_TRIANGLE_STRIP)
        for i in range(len(outer_terrain)):
            # Create checkerboard pattern on top surface
            if (i // 2) % 2 == 0:
                glColor3f(0.15, 0.4, 0.15)  # Darker grass
            else:
                glColor3f(0.18, 0.45, 0.18)  # Lighter grass

            tx, ty = outer_terrain[i]
            glVertex3f(tx, ty, terrain_height)
            glVertex3f(tx, ty, terrain_height + 10)
        tx, ty = outer_terrain[0]
        glVertex3f(tx, ty, terrain_height)
        glVertex3f(tx, ty, terrain_height + 10)
        glEnd()

    def _draw_track_features(self):
        """Draw visual features like checkpoints, sectors, and direction arrows"""
        glDisable(GL_LIGHTING)

        # Define checkpoint/sector positions (every N points along the track)
        checkpoint_interval = 6  # Every 6 points
        arrow_interval = 3  # More frequent arrows for direction indication

        for i in range(0, len(self.centerline), checkpoint_interval):
            px, py = self.centerline[i]
            next_idx = (i + 1) % len(self.centerline)
            next_px, next_py = self.centerline[next_idx]

            # Calculate track direction
            dx = next_px - px
            dy = next_py - py
            track_angle = np.arctan2(dy, dx)
            perp_angle = track_angle + np.pi / 2

            # Draw checkpoint markers (tall colored poles at track sides)
            marker_height = 25
            marker_offset = self.track_width / 2 + 5

            # Left marker (cyan)
            left_x = px + marker_offset * np.cos(perp_angle)
            left_y = py + marker_offset * np.sin(perp_angle)
            self._draw_checkpoint_marker(left_x, left_y, marker_height, (0.0, 0.8, 1.0))

            # Right marker (cyan)
            right_x = px - marker_offset * np.cos(perp_angle)
            right_y = py - marker_offset * np.sin(perp_angle)
            self._draw_checkpoint_marker(right_x, right_y, marker_height, (0.0, 0.8, 1.0))

            # Draw sector number in the air
            sector_num = i // checkpoint_interval + 1
            mid_x = px
            mid_y = py
            self._draw_sector_number(mid_x, mid_y, 20, sector_num)

        # Draw direction arrows on track surface
        for i in range(0, len(self.centerline), arrow_interval):
            px, py = self.centerline[i]
            next_idx = (i + 1) % len(self.centerline)
            next_px, next_py = self.centerline[next_idx]

            # Calculate track direction
            dx = next_px - px
            dy = next_py - py
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length

                # Draw arrow
                self._draw_direction_arrow(px, py, dx, dy)

        # Draw start/finish line markers (special color)
        px, py = self.centerline[0]
        next_px, next_py = self.centerline[1]
        dx = next_px - px
        dy = next_py - py
        track_angle = np.arctan2(dy, dx)
        perp_angle = track_angle + np.pi / 2

        marker_offset = self.track_width / 2 + 5
        marker_height = 35  # Taller for start/finish

        # Start/finish markers (red and white pattern)
        left_x = px + marker_offset * np.cos(perp_angle)
        left_y = py + marker_offset * np.sin(perp_angle)
        self._draw_checkpoint_marker(left_x, left_y, marker_height, (1.0, 0.0, 0.0))

        right_x = px - marker_offset * np.cos(perp_angle)
        right_y = py - marker_offset * np.sin(perp_angle)
        self._draw_checkpoint_marker(right_x, right_y, marker_height, (1.0, 1.0, 1.0))

        glEnable(GL_LIGHTING)

    def _draw_checkpoint_marker(self, x, y, height, color):
        """Draw a checkpoint marker pole"""
        glColor3f(*color)

        # Draw vertical pole
        glLineWidth(4)
        glBegin(GL_LINES)
        glVertex3f(x, y, 0)
        glVertex3f(x, y, height)
        glEnd()

        # Draw sphere at top
        glPushMatrix()
        glTranslatef(x, y, height)
        quadric = gluNewQuadric()
        gluSphere(quadric, 3, 8, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def _draw_sector_number(self, x, y, height, number):
        """Draw floating sector number (simplified as a marker)"""
        # Draw as colored floating sphere
        color = ((number * 0.3) % 1.0, (number * 0.5) % 1.0, (number * 0.7) % 1.0)
        glColor3f(*color)

        glPushMatrix()
        glTranslatef(x, y, height)
        quadric = gluNewQuadric()
        gluSphere(quadric, 5, 8, 8)
        gluDeleteQuadric(quadric)
        glPopMatrix()

    def _draw_direction_arrow(self, x, y, dx, dy):
        """Draw a direction arrow on the track surface"""
        glColor3f(1.0, 1.0, 0.0)  # Yellow arrows
        glLineWidth(3)

        arrow_length = 15
        arrow_width = 8

        # Arrow shaft
        end_x = x + dx * arrow_length
        end_y = y + dy * arrow_length

        glBegin(GL_LINES)
        glVertex3f(x, y, 0.2)
        glVertex3f(end_x, end_y, 0.2)
        glEnd()

        # Arrowhead (two lines forming V)
        head_angle = np.arctan2(dy, dx)
        left_angle = head_angle + 2.5
        right_angle = head_angle - 2.5

        left_x = end_x - arrow_width * np.cos(left_angle)
        left_y = end_y - arrow_width * np.sin(left_angle)
        right_x = end_x - arrow_width * np.cos(right_angle)
        right_y = end_y - arrow_width * np.sin(right_angle)

        glBegin(GL_LINES)
        glVertex3f(end_x, end_y, 0.2)
        glVertex3f(left_x, left_y, 0.2)
        glVertex3f(end_x, end_y, 0.2)
        glVertex3f(right_x, right_y, 0.2)
        glEnd()

    def _draw_scenery(self):
        """Draw trees, signs, and buildings for spatial awareness (OPTIMIZED)"""
        glDisable(GL_LIGHTING)

        # REDUCED scenery for performance - only draw every other frame worth
        tree_interval = 8  # Trees every 8 points (reduced from 4)
        sign_positions = [0, 10, 20]  # Fewer signs (reduced from 5)

        # Draw trees on outer edge of track
        for i in range(0, len(self.centerline), tree_interval):
            px, py = self.centerline[i]
            next_idx = (i + 1) % len(self.centerline)
            next_px, next_py = self.centerline[next_idx]

            # Calculate track direction
            dx = next_px - px
            dy = next_py - py
            track_angle = np.arctan2(dy, dx)
            perp_angle = track_angle + np.pi / 2

            # Alternate trees on left and right
            side_offset = self.track_width / 2 + 30
            if i % 2 == 0:
                # Tree on left
                tree_x = px + side_offset * np.cos(perp_angle)
                tree_y = py + side_offset * np.sin(perp_angle)
                self._draw_tree(tree_x, tree_y)
            else:
                # Tree on right
                tree_x = px - side_offset * np.cos(perp_angle)
                tree_y = py - side_offset * np.sin(perp_angle)
                self._draw_tree(tree_x, tree_y)

        # Draw distance signs at key corners
        for sign_idx in sign_positions:
            if sign_idx < len(self.centerline):
                px, py = self.centerline[sign_idx]
                next_idx = (sign_idx + 1) % len(self.centerline)
                next_px, next_py = self.centerline[next_idx]

                dx = next_px - px
                dy = next_py - py
                track_angle = np.arctan2(dy, dx)
                perp_angle = track_angle + np.pi / 2

                # Sign on right side
                sign_offset = self.track_width / 2 + 15
                sign_x = px - sign_offset * np.cos(perp_angle)
                sign_y = py - sign_offset * np.sin(perp_angle)
                self._draw_distance_sign(sign_x, sign_y, sign_idx * 100)  # Distance markers

        # Draw buildings at specific corners for landmarks
        building_positions = [8, 18]  # Reduced to 2 buildings for performance
        for building_idx in building_positions:
            if building_idx < len(self.centerline):
                px, py = self.centerline[building_idx]
                next_idx = (building_idx + 1) % len(self.centerline)
                next_px, next_py = self.centerline[next_idx]

                dx = next_px - px
                dy = next_py - py
                track_angle = np.arctan2(dy, dx)
                perp_angle = track_angle + np.pi / 2

                # Building on outer edge
                building_offset = self.track_width / 2 + 60
                building_x = px + building_offset * np.cos(perp_angle)
                building_y = py + building_offset * np.sin(perp_angle)
                self._draw_building(building_x, building_y, building_idx)

        glEnable(GL_LIGHTING)

    def _draw_tree(self, x, y):
        """Draw a simple tree (trunk + foliage) - OPTIMIZED"""
        glPushMatrix()
        glTranslatef(x, y, 0)

        # Draw trunk (simplified - single line instead of loop)
        glColor3f(0.4, 0.2, 0.1)
        trunk_height = 15
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, trunk_height)
        glEnd()

        # Tree foliage (green sphere) - reduced detail
        glColor3f(0.1, 0.5, 0.1)
        glTranslatef(0, 0, trunk_height)
        quadric = gluNewQuadric()
        gluSphere(quadric, 8, 4, 4)  # Reduced from 6,6 to 4,4
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def _draw_distance_sign(self, x, y, distance):
        """Draw a distance/corner marker sign"""
        glColor3f(1.0, 0.5, 0.0)  # Orange sign

        # Sign post
        glLineWidth(3)
        glBegin(GL_LINES)
        glVertex3f(x, y, 0)
        glVertex3f(x, y, 15)
        glEnd()

        # Sign board (rectangle)
        glPushMatrix()
        glTranslatef(x, y, 12)

        # Draw sign as colored box
        sign_width = 6
        sign_height = 4
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-sign_width/2, 0, -sign_height/2)
        glVertex3f(sign_width/2, 0, -sign_height/2)
        glVertex3f(sign_width/2, 0, sign_height/2)
        glVertex3f(-sign_width/2, 0, sign_height/2)
        glEnd()

        # Draw distance marker sphere on top
        glTranslatef(0, 0, sign_height/2 + 2)
        color_intensity = (distance % 500) / 500.0
        glColor3f(1.0, color_intensity, 0.0)
        quadric = gluNewQuadric()
        gluSphere(quadric, 2, 6, 6)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def _draw_building(self, x, y, building_type):
        """Draw a building/grandstand as a landmark"""
        # Different colored buildings for variety
        colors = [
            (0.7, 0.7, 0.8),  # Light gray
            (0.8, 0.6, 0.4),  # Brown
            (0.6, 0.6, 0.7),  # Blue-gray
            (0.7, 0.5, 0.5),  # Red-gray
        ]
        color = colors[building_type % len(colors)]
        glColor3f(*color)

        building_width = 20
        building_depth = 15
        building_height = 25 + (building_type * 5)  # Varying heights

        glPushMatrix()
        glTranslatef(x, y, building_height/2)

        # Draw building as box
        w, d, h = building_width/2, building_depth/2, building_height/2
        glBegin(GL_QUADS)

        # Front face
        glVertex3f(-w, d, -h)
        glVertex3f(w, d, -h)
        glVertex3f(w, d, h)
        glVertex3f(-w, d, h)

        # Back face
        glVertex3f(-w, -d, -h)
        glVertex3f(-w, -d, h)
        glVertex3f(w, -d, h)
        glVertex3f(w, -d, -h)

        # Top face
        glColor3f(color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        glVertex3f(-w, -d, h)
        glVertex3f(w, -d, h)
        glVertex3f(w, d, h)
        glVertex3f(-w, d, h)

        # Left face
        glColor3f(*color)
        glVertex3f(-w, -d, -h)
        glVertex3f(-w, d, -h)
        glVertex3f(-w, d, h)
        glVertex3f(-w, -d, h)

        # Right face
        glVertex3f(w, -d, -h)
        glVertex3f(w, -d, h)
        glVertex3f(w, d, h)
        glVertex3f(w, d, -h)

        glEnd()

        # Add windows (small bright squares)
        glColor3f(1.0, 1.0, 0.8)
        window_rows = 3
        window_cols = 4
        for row in range(window_rows):
            for col in range(window_cols):
                wx = -w + (col + 0.5) * building_width / window_cols - building_width/2
                wz = -h + (row + 0.5) * building_height / window_rows

                glBegin(GL_QUADS)
                glVertex3f(wx, d + 0.1, wz)
                glVertex3f(wx + 2, d + 0.1, wz)
                glVertex3f(wx + 2, d + 0.1, wz + 2)
                glVertex3f(wx, d + 0.1, wz + 2)
                glEnd()

        glPopMatrix()


class Renderer3D:
    """3D OpenGL renderer"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.setup_opengl()

    def setup_opengl(self):
        """Initialize OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Lighting setup
        glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 1000, 0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

        glClearColor(0.6, 0.8, 1.0, 1.0)  # Sky blue background

    def setup_3d_view(self, car):
        """Setup 3D perspective for main view"""
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 1.0, 5000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Hood camera position
        cam_pos, look_pos = car.get_hood_camera_position()
        gluLookAt(
            cam_pos[0], cam_pos[1], cam_pos[2],  # Camera position
            look_pos[0], look_pos[1], look_pos[2],  # Look-at point
            0, 0, 1  # Up vector
        )

    def draw_lane_markers_3d(self, camera, track):
        """Draw 3D markers for detected lane points"""
        left_lane, right_lane, center_lane = camera.detect_lanes(track)

        glDisable(GL_LIGHTING)

        # Draw left lane markers (red)
        glColor3f(1.0, 0.0, 0.0)
        for px, py, angle in left_lane:
            self._draw_marker(px, py, 5.0, 8.0)

        # Draw right lane markers (cyan)
        glColor3f(0.0, 0.8, 1.0)
        for px, py, angle in right_lane:
            self._draw_marker(px, py, 5.0, 8.0)

        # Draw center lane markers (yellow)
        glColor3f(1.0, 1.0, 0.0)
        for px, py, angle in center_lane:
            self._draw_marker(px, py, 3.0, 6.0)

        glEnable(GL_LIGHTING)

    def _draw_marker(self, x, y, radius, height):
        """Draw a cylindrical marker at position"""
        glPushMatrix()
        glTranslatef(x, y, 0)

        # Draw vertical line
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, height)
        glEnd()

        # Draw sphere at top
        glTranslatef(0, 0, height)
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 8, 8)
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def draw_lookahead_point_3d(self, lka):
        """Draw LKA lookahead point in 3D"""
        if lka.active and hasattr(lka, 'lookahead_point'):
            lx, ly = lka.lookahead_point

            glDisable(GL_LIGHTING)

            # Draw vertical marker
            glColor3f(1.0, 1.0, 0.0)
            glLineWidth(3)
            glBegin(GL_LINES)
            glVertex3f(lx, ly, 0)
            glVertex3f(lx, ly, 30)
            glEnd()

            # Draw sphere at top
            glPushMatrix()
            glTranslatef(lx, ly, 30)
            quadric = gluNewQuadric()
            gluSphere(quadric, 8, 12, 12)
            gluDeleteQuadric(quadric)
            glPopMatrix()

            glEnable(GL_LIGHTING)


class Minimap:
    """2D minimap renderer - reuses original drawing code"""
    def __init__(self, size, track):
        self.size = size
        self.track = track
        self.surface = pygame.Surface((size, size))

        # Calculate track bounding box for proper scaling
        self._calculate_track_bounds()

    def _calculate_track_bounds(self):
        """Calculate bounding box of entire track"""
        # Get all track points including boundaries
        all_points = list(self.track.centerline)
        outer = self.track._offset_line(self.track.centerline, self.track.track_width / 2)
        inner = self.track._offset_line(self.track.centerline, -self.track.track_width / 2)
        all_points.extend(outer)
        all_points.extend(inner)

        # Find min/max coordinates
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]

        self.min_x = min(xs)
        self.max_x = max(xs)
        self.min_y = min(ys)
        self.max_y = max(ys)

        # Calculate scale to fit in minimap with margin
        margin = 20  # pixels
        track_width = self.max_x - self.min_x
        track_height = self.max_y - self.min_y

        # Scale to fit within minimap size minus margins
        scale_x = (self.size - 2 * margin) / track_width
        scale_y = (self.size - 2 * margin) / track_height

        # Use the smaller scale to maintain aspect ratio
        self.scale = min(scale_x, scale_y)
        self.margin = margin

        print("="*60)
        print("MINIMAP INITIALIZATION")
        print(f"Minimap size: {self.size}x{self.size}")
        print(f"Track bounds: X=[{self.min_x:.1f}, {self.max_x:.1f}] Y=[{self.min_y:.1f}, {self.max_y:.1f}]")
        print(f"Track dimensions: {track_width:.1f} x {track_height:.1f}")
        print(f"Scale: {self.scale:.4f} (scale_x={scale_x:.4f}, scale_y={scale_y:.4f})")
        print(f"Expected minimap bounds: ({self.margin}, {self.margin}) to ({int(track_width*self.scale + self.margin)}, {int(track_height*self.scale + self.margin)})")
        print("="*60)

    def _world_to_minimap(self, x, y):
        """Convert world coordinates to minimap coordinates"""
        # Translate to origin, scale, then translate to minimap with margin
        map_x = (x - self.min_x) * self.scale + self.margin
        map_y = (y - self.min_y) * self.scale + self.margin
        return int(map_x), int(map_y)

    def render(self, car, camera, lka):
        """Render minimap with original 2D view"""
        # Fill with semi-transparent dark background
        self.surface.fill((20, 20, 20))  # Very dark gray background

        # Draw border around minimap
        pygame.draw.rect(self.surface, (100, 100, 100), (0, 0, self.size, self.size), 3)
        pygame.draw.rect(self.surface, (200, 200, 200), (2, 2, self.size-4, self.size-4), 1)

        # DEBUG: Draw grid to show we're using full minimap space
        grid_color = (40, 40, 40)
        for i in range(0, self.size, 50):
            pygame.draw.line(self.surface, grid_color, (i, 0), (i, self.size), 1)
            pygame.draw.line(self.surface, grid_color, (0, i), (self.size, i), 1)

        # DEBUG: Draw expected bounds rectangle (should be near edges)
        min_scaled = self._world_to_minimap(self.min_x, self.min_y)
        max_scaled = self._world_to_minimap(self.max_x, self.max_y)
        pygame.draw.rect(self.surface, (255, 0, 0),
                        (min_scaled[0], min_scaled[1],
                         max_scaled[0] - min_scaled[0],
                         max_scaled[1] - min_scaled[1]), 2)

        # DEBUG: Label the bounds
        font = pygame.font.Font(None, 16)
        bounds_text = font.render(f"Bounds: {min_scaled} to {max_scaled}", True, (255, 0, 0))
        self.surface.blit(bounds_text, (5, self.size - 20))

        # Draw track
        self._draw_track_2d()

        # Draw camera FOV and detections
        self._draw_camera_view_2d(camera)

        # Draw LKA lookahead
        if lka.active and hasattr(lka, 'lookahead_point'):
            lx, ly = lka.lookahead_point
            car_scaled = self._world_to_minimap(car.x, car.y)
            lookahead_scaled = self._world_to_minimap(lx, ly)
            pygame.draw.line(self.surface, YELLOW, car_scaled, lookahead_scaled, 2)
            pygame.draw.circle(self.surface, YELLOW, lookahead_scaled, 6)

        # Draw car (simple representation)
        self._draw_car_2d(car)

        # DEBUG: Draw scale info
        font = pygame.font.Font(None, 20)
        scale_text = font.render(f"Scale: {self.scale:.3f}", True, (255, 255, 0))
        self.surface.blit(scale_text, (5, 5))

        return self.surface

    def _draw_track_2d(self):
        """Draw track in minimap with proper scaling"""
        outer = self.track._offset_line(self.track.centerline, self.track.track_width / 2)
        inner = self.track._offset_line(self.track.centerline, -self.track.track_width / 2)

        # Convert to minimap coordinates
        outer_scaled = [self._world_to_minimap(x, y) for x, y in outer]
        inner_scaled = [self._world_to_minimap(x, y) for x, y in inner]

        if len(outer_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, outer_scaled, 2)
        if len(inner_scaled) > 2:
            pygame.draw.lines(self.surface, WHITE, True, inner_scaled, 2)

        # Draw centerline dashed
        centerline_scaled = [self._world_to_minimap(x, y) for x, y in self.track.centerline]
        for i in range(0, len(centerline_scaled) - 1, 2):
            p1 = centerline_scaled[i]
            p2 = centerline_scaled[i + 1]
            pygame.draw.line(self.surface, GRAY, p1, p2, 1)

    def _draw_camera_view_2d(self, camera):
        """Draw camera FOV and detected lanes"""
        camera_x, camera_y = camera.get_camera_position()

        # Draw FOV cone
        fov_points = [(camera_x, camera_y)]
        left_angle = camera.car.theta - camera.field_of_view / 2
        fov_points.append((
            camera_x + camera.max_range * np.cos(left_angle),
            camera_y + camera.max_range * np.sin(left_angle)
        ))
        right_angle = camera.car.theta + camera.field_of_view / 2
        fov_points.append((
            camera_x + camera.max_range * np.cos(right_angle),
            camera_y + camera.max_range * np.sin(right_angle)
        ))

        # Convert FOV points to minimap coordinates
        fov_points_scaled = [self._world_to_minimap(x, y) for x, y in fov_points]

        # Draw semi-transparent FOV
        s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.polygon(s, (0, 255, 0, 30), fov_points_scaled)
        self.surface.blit(s, (0, 0))

        # Draw FOV edges
        camera_pos_scaled = fov_points_scaled[0]
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[1], 1)
        pygame.draw.line(self.surface, GREEN, camera_pos_scaled, fov_points_scaled[2], 1)

        # Draw detected lane points
        left_lane, right_lane, center_lane = camera.detect_lanes(camera.car.track)

        # Get wheel positions
        (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y) = camera.car.get_front_wheel_positions()
        left_wheel_scaled = self._world_to_minimap(left_wheel_x, left_wheel_y)
        right_wheel_scaled = self._world_to_minimap(right_wheel_x, right_wheel_y)

        # Draw left lane points with vectors from LEFT wheel
        for px, py, _ in left_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (255, 0, 0), (px_scaled, py_scaled), 3)
            # Vector from left wheel to left lane point
            pygame.draw.line(self.surface, (255, 128, 0), left_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw right lane points with vectors from RIGHT wheel
        for px, py, _ in right_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (0, 128, 255), (px_scaled, py_scaled), 3)
            # Vector from right wheel to right lane point
            pygame.draw.line(self.surface, (0, 200, 200), right_wheel_scaled, (px_scaled, py_scaled), 1)

        # Draw center lane points
        for px, py, _ in center_lane:
            px_scaled, py_scaled = self._world_to_minimap(px, py)
            pygame.draw.circle(self.surface, (0, 0, 200), (px_scaled, py_scaled), 2)

        # Draw camera position
        pygame.draw.circle(self.surface, GREEN, camera_pos_scaled, 5)

        # Draw wheel positions
        pygame.draw.circle(self.surface, (255, 100, 0), left_wheel_scaled, 5)  # Orange
        pygame.draw.circle(self.surface, (0, 150, 255), right_wheel_scaled, 5)  # Cyan

    def _draw_car_2d(self, car):
        """Draw car in minimap"""
        # Draw main axis
        rear_x = car.x - (car.length/2) * np.cos(car.theta)
        rear_y = car.y - (car.length/2) * np.sin(car.theta)
        front_x = car.x + (car.length/2) * np.cos(car.theta)
        front_y = car.y + (car.length/2) * np.sin(car.theta)

        # Convert to minimap coordinates
        rear_scaled = self._world_to_minimap(rear_x, rear_y)
        front_scaled = self._world_to_minimap(front_x, front_y)
        center_scaled = self._world_to_minimap(car.x, car.y)

        pygame.draw.line(self.surface, WHITE, rear_scaled, front_scaled, 2)

        # Draw direction indicator
        pygame.draw.circle(self.surface, BLUE, front_scaled, 4)
        pygame.draw.circle(self.surface, YELLOW, center_scaled, 3)


class HUD:
    """Heads-up display for 3D view"""
    def __init__(self):
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)

    def render(self, surface, car, camera, lka):
        """Render HUD overlays"""
        # LKA status
        self._draw_lka_status(surface, lka)

        # Speed and steering info
        self._draw_telemetry(surface, car)

        # Lane detection status
        self._draw_lane_status(surface, camera)

    def _draw_lka_status(self, surface, lka):
        """Draw LKA status indicator"""
        if lka.active:
            status_text = "LKA: ACTIVE"
            color = GREEN
        else:
            status_text = "LKA: OFF"
            color = RED

        text = self.font_large.render(status_text, True, color)
        rect = text.get_rect(center=(WIDTH // 2, 30))

        # Background
        bg_rect = rect.inflate(20, 10)
        s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
        surface.blit(s, bg_rect.topleft)

        surface.blit(text, rect)

    def _draw_telemetry(self, surface, car):
        """Draw speed and steering information"""
        texts = [
            f"Speed: {abs(car.velocity):.1f} px/s",
            f"Steering: {np.degrees(car.steering_angle):.1f}°",
        ]

        y = HEIGHT - 100
        for text in texts:
            rendered = self.font.render(text, True, WHITE)
            # Background
            rect = rendered.get_rect(topleft=(10, y))
            bg_rect = rect.inflate(10, 5)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
            surface.blit(s, bg_rect.topleft)

            surface.blit(rendered, rect)
            y += 30

    def _draw_lane_status(self, surface, camera):
        """Draw lane detection status"""
        texts = [
            f"Lane: {camera.current_lane}",
            f"Left: {'OK' if camera.left_lane_detected else 'NO'}",
            f"Right: {'OK' if camera.right_lane_detected else 'NO'}",
        ]

        y = 80
        for text in texts:
            color = GREEN if ('OK' in text or 'LEFT' in text or 'RIGHT' in text) else WHITE
            if 'NO' in text:
                color = RED

            rendered = self.font.render(text, True, color)
            rect = rendered.get_rect(topleft=(10, y))
            bg_rect = rect.inflate(10, 5)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
            surface.blit(s, bg_rect.topleft)

            surface.blit(rendered, rect)
            y += 30


def main():
    """Main simulation loop"""
    # Create track
    track = SaoPauloTrack(offset_x=50, offset_y=50)

    # Create car
    start_x, start_y, start_theta = track.get_start_position(lane_number=1)
    car = Car(start_x, start_y, start_theta)
    car.track = track  # Store reference for camera

    # Create camera sensor
    camera = CameraSensor(car)

    # Create LKA controller
    lka = PurePursuitLKA(car, camera)

    # Create renderer
    renderer = Renderer3D(WIDTH, HEIGHT)

    # Create minimap
    minimap = Minimap(MINIMAP_SIZE, track)

    # Create HUD
    hud = HUD()

    # Main loop
    running = True
    dt = 1.0 / FPS

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    lka.toggle()

        # Get keyboard state
        keys = pygame.key.get_pressed()

        # Calculate LKA steering
        lka_steering = lka.calculate_steering(track) if lka.active else None

        # Update car
        car.update(dt, keys, lka_steering, lka)

        # Check collision with track boundaries
        if not car.is_on_track(track):
            car.handle_collision()

        # === 3D RENDERING ===
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup 3D view
        renderer.setup_3d_view(car)

        # Draw track
        track.draw_3d()

        # Draw lane markers
        renderer.draw_lane_markers_3d(camera, track)

        # Draw LKA lookahead point
        renderer.draw_lookahead_point_3d(lka)

        # Draw car (disabled in first-person, but could draw for debugging)
        # car.draw_3d()

        # === 2D OVERLAY RENDERING ===
        # Switch to 2D orthographic projection for HUD and minimap
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)

        # Create pygame surface for 2D overlay
        overlay_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay_surface.fill((0, 0, 0, 0))

        # Render HUD
        hud.render(overlay_surface, car, camera, lka)

        # Render minimap
        minimap_surface = minimap.render(car, camera, lka)
        minimap_pos = (WIDTH - MINIMAP_SIZE - 10, 10)

        # Draw minimap background
        bg_rect = pygame.Rect(minimap_pos[0] - 5, minimap_pos[1] - 5,
                             MINIMAP_SIZE + 10, MINIMAP_SIZE + 10)
        pygame.draw.rect(overlay_surface, (0, 0, 0, 200), bg_rect)
        pygame.draw.rect(overlay_surface, WHITE, bg_rect, 2)

        overlay_surface.blit(minimap_surface, minimap_pos)

        # Draw controls hint
        hint_font = pygame.font.Font(None, 20)
        hint_texts = [
            "W/S: Accel/Brake | A/D: Steer | F: Toggle LKA | ESC: Exit"
        ]
        y = HEIGHT - 30
        for hint in hint_texts:
            text = hint_font.render(hint, True, WHITE)
            rect = text.get_rect(center=(WIDTH // 2, y))
            bg_rect = rect.inflate(10, 5)
            s = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(s, (0, 0, 0, 150), (0, 0, bg_rect.width, bg_rect.height))
            overlay_surface.blit(s, bg_rect.topleft)
            overlay_surface.blit(text, rect)
            y += 25

        # Convert pygame surface to OpenGL texture and render as textured quad
        # This is more reliable than glDrawPixels
        texture_data = pygame.image.tostring(overlay_surface, "RGBA", False)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Create and bind texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw textured quad covering the entire screen
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WIDTH, HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)

        # Clean up texture
        glDeleteTextures([texture_id])
        glDisable(GL_BLEND)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)

        # Restore projection matrices
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        # Update display
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
