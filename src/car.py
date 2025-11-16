"""
Car Module - Ackermann Steering Kinematics
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

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

        # Wheel rotation tracking for animation
        self.wheel_rotation = 0.0  # Current wheel rotation angle in radians
        self.wheel_radius = 4  # Wheel radius for rotation calculation

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

            # Update wheel rotation based on distance traveled
            # Negate to fix backwards rotation
            distance_traveled = self.velocity * dt
            self.wheel_rotation -= distance_traveled / self.wheel_radius
            self.wheel_rotation = self.wheel_rotation % (2 * np.pi)  # Keep in [0, 2π]

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
        """Get position and orientation for center-of-car first-person camera"""
        # Camera at center of car, at eye level
        cam_x = self.x
        cam_y = self.y
        cam_z = self.hood_height

        # Look-at point ahead of car
        look_distance = 100
        look_x = self.x + look_distance * np.cos(self.theta)
        look_y = self.y + look_distance * np.sin(self.theta)
        look_z = self.hood_height - 2  # Look slightly down

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

    def draw_wheels_only_3d(self):
        """Draw ONLY the wheels in 3D for first-person view (car body invisible)"""
        glPushMatrix()

        # Transform to car position and orientation
        glTranslatef(self.x, self.y, 0)
        glRotatef(np.degrees(self.theta), 0, 0, 1)

        # Draw wheels with enhanced details
        self._draw_wheels_enhanced()

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

    def _draw_wheels_enhanced(self):
        """Draw car wheels with tire treads and rotation"""
        wheel_radius = 4
        wheel_width = 3

        # Wheel positions relative to car center
        # Raised Z position to put wheels above ground
        wheel_positions = [
            (self.length/2 - 5, self.width/2, wheel_radius),      # Front left
            (self.length/2 - 5, -self.width/2, wheel_radius),     # Front right
            (-self.length/2 + 5, self.width/2, wheel_radius),     # Rear left
            (-self.length/2 + 5, -self.width/2, wheel_radius),    # Rear right
        ]

        for i, (wx, wy, wz) in enumerate(wheel_positions):
            glPushMatrix()
            glTranslatef(wx, wy, wz)

            # Front wheels have steering angle
            if i < 2:
                glRotatef(np.degrees(self.steering_angle), 0, 0, 1)

            # Orient wheel to point forward (rotate 90 degrees around X-axis)
            glRotatef(90, 1, 0, 0)

            # Apply wheel rotation (rolling animation around Z-axis after rotation)
            glRotatef(np.degrees(self.wheel_rotation), 0, 0, 1)

            # Draw tire (black rubber)
            glColor3f(0.1, 0.1, 0.1)
            self._draw_tire_with_treads(wheel_radius, wheel_width, 12)  # Reduced slices from 16 to 12

            # Draw rim (metallic gray) - OPTIMIZED: reduced from 8 to 6 slices
            glColor3f(0.6, 0.6, 0.65)
            self._draw_cylinder(wheel_radius * 0.6, wheel_width * 0.8, 6)

            glPopMatrix()

    def _draw_tire_with_treads(self, radius, width, slices):
        """Draw a tire with tread pattern - OPTIMIZED"""
        # Draw main tire body
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, -width/2)
            glVertex3f(x, y, width/2)
        glEnd()

        # Draw tire caps (sides) - OPTIMIZED: reduced slices
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, width/2)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, width/2)
        glEnd()

        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, -width/2)
        for i in range(slices + 1):
            angle = 2 * np.pi * i / slices
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            glVertex3f(x, y, -width/2)
        glEnd()

        # Draw tread grooves - OPTIMIZED: reduced from 12 to 8 treads
        glColor3f(0.05, 0.05, 0.05)
        glLineWidth(2)
        num_treads = 8  # Reduced from 12
        for i in range(num_treads):
            angle = 2 * np.pi * i / num_treads

            # Calculate positions for the chevron pattern
            x_center = radius * np.cos(angle)
            y_center = radius * np.sin(angle)

            # Next position (for the arrow point direction)
            angle_offset = np.pi / (num_treads * 2)

            # Create V-shape chevron pointing in rolling direction
            x_left = radius * np.cos(angle - angle_offset)
            y_left = radius * np.sin(angle - angle_offset)

            x_right = radius * np.cos(angle + angle_offset)
            y_right = radius * np.sin(angle + angle_offset)

            # Draw left arm (from edge to center)
            glBegin(GL_LINES)
            glVertex3f(x_left, y_left, -width/2 * 0.7)
            glVertex3f(x_center, y_center, 0)
            glEnd()

            # Draw right arm (from center to edge)
            glBegin(GL_LINES)
            glVertex3f(x_center, y_center, 0)
            glVertex3f(x_right, y_right, width/2 * 0.7)
            glEnd()
        glLineWidth(1)

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


