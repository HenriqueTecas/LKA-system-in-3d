"""
Car Module - Ackermann Steering Kinematics
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from .config import (
    PIXELS_PER_METER, VEHICLE_MASS, VEHICLE_INERTIA_Z, VEHICLE_WHEELBASE,
    VEHICLE_CG_TO_FRONT, VEHICLE_CG_TO_REAR, TIRE_CORNERING_STIFFNESS_FRONT,
    TIRE_CORNERING_STIFFNESS_REAR, AERO_CD, AERO_AREA, AIR_DENSITY,
    AERO_CL, AERO_DOWNFORCE_AREA,
    ROLLING_RESISTANCE_COEFF, GRAVITY, MAX_DRIVE_FORCE, MAX_BRAKE_FORCE,
    THROTTLE_TAU, BRAKE_TAU, STEERING_TAU, MAX_STEERING_ANGLE, MAX_STEERING_RATE,
    MAX_VELOCITY
)

class Car:
    """Car with Ackermann kinematics and realistic longitudinal/lateral dynamics.
    
    Features:
    - Ackermann steering kinematics
    - Realistic forces: drive, brake, drag, rolling resistance, engine braking
    - Downforce-dependent tire grip
    - Speed-dependent handling (tire slip at high speeds)
    - Cornering resistance
    
    Internal state uses SI units (meters, m/s, radians).
    Rendering positions are converted to/from pixels.
    """
    
    def __init__(self, x_pixels, y_pixels, theta):
        """Initialize car at given position and orientation."""
        # ====================================================================
        # Unit Conversion
        # ====================================================================
        self.pixels_per_meter = float(PIXELS_PER_METER)
        
        # ====================================================================
        # Position and Orientation (SI units)
        # ====================================================================
        self.x = x_pixels / self.pixels_per_meter  # meters
        self.y = y_pixels / self.pixels_per_meter  # meters
        self.theta = theta  # radians
        
        # ====================================================================
        # Vehicle Geometry (SI units)
        # ====================================================================
        self.mass = float(VEHICLE_MASS)  # kg
        self.inertia_z = float(VEHICLE_INERTIA_Z)  # kg·m²
        self.wheelbase = float(VEHICLE_WHEELBASE)  # m
        self.lf = float(VEHICLE_CG_TO_FRONT)  # m (CG to front axle)
        self.lr = float(VEHICLE_CG_TO_REAR)  # m (CG to rear axle)
        
        # ====================================================================
        # Tire Parameters
        # ====================================================================
        self.c_f = float(TIRE_CORNERING_STIFFNESS_FRONT)  # N/rad
        self.c_r = float(TIRE_CORNERING_STIFFNESS_REAR)  # N/rad
        self.wheel_radius = 0.35  # meters (typical tire radius)
        
        # ====================================================================
        # Forces and Resistances
        # ====================================================================
        self.g = float(GRAVITY)  # m/s²
        
        # Aerodynamics
        self.rho = float(AIR_DENSITY)  # kg/m³
        self.c_d = float(AERO_CD)  # Drag coefficient
        self.area = float(AERO_AREA)  # m² (frontal area)
        self.c_l = float(AERO_CL)  # Downforce coefficient
        self.downforce_area = float(AERO_DOWNFORCE_AREA)  # m² (top area)
        
        # Rolling resistance
        self.c_rr = float(ROLLING_RESISTANCE_COEFF)
        
        # Drive/brake limits
        self.max_drive_force = float(MAX_DRIVE_FORCE)  # N
        self.max_brake_force = float(MAX_BRAKE_FORCE)  # N
        
        # ====================================================================
        # Vehicle Dynamic State (SI units)
        # ====================================================================
        self.velocity = 0.0  # m/s (longitudinal velocity)
        self.steering_angle = 0.0  # radians
        
        # ====================================================================
        # Performance Limits
        # ====================================================================
        self.max_velocity = float(MAX_VELOCITY)  # m/s
        self.max_steering_angle = float(MAX_STEERING_ANGLE)  # rad
        self.max_steering_rate = float(MAX_STEERING_RATE)  # rad/s
        
        # ====================================================================
        # Actuator States and Time Constants
        # ====================================================================
        self.throttle_state = 0.0  # [-1, 1]
        self.brake_state = 0.0  # [0, 1]
        self.throttle_tau = float(THROTTLE_TAU)  # seconds
        self.brake_tau = float(BRAKE_TAU)  # seconds
        self.steering_tau = float(STEERING_TAU)  # seconds
        
        # ====================================================================
        # Rendering Dimensions (pixels)
        # ====================================================================
        # BMW E36 (1998) dimensions
        self.length = 4.433 * self.pixels_per_meter  # 4433mm
        self.width = 1.71 * self.pixels_per_meter    # 1710mm
        self.height = 1.39 * self.pixels_per_meter   # 1390mm
        self.hood_height = 1.0 * self.pixels_per_meter  # estimated
        
        # Wheel rotation for animation
        self.wheel_rotation = 0.0  # radians

    # ========================================================================
    # MAIN UPDATE LOOP
    # ========================================================================
    
    def update(self, dt, keys, lka_steering=None, lka_controller=None):
        """Update car state based on user input and LKA control."""
        
        # ====================================================================
        # LONGITUDINAL CONTROL - User Input
        # ====================================================================
        desired_throttle = 0.0
        desired_brake = 0.0

        if keys[pygame.K_w] and not keys[pygame.K_s]:
            desired_throttle = 1.0
            desired_brake = 0.0
        elif keys[pygame.K_s] and not keys[pygame.K_w]:
            # If moving forward, treat S as brake; if near zero, allow reverse
            if self.velocity > 0.5:
                desired_throttle = 0.0
                desired_brake = 1.0
            else:
                desired_throttle = -1.0
                desired_brake = 0.0
        else:
            desired_throttle = 0.0
            desired_brake = 0.0

        # ====================================================================
        # ACTUATOR DYNAMICS - First-order response
        # ====================================================================
        if self.throttle_tau > 0:
            self.throttle_state += (desired_throttle - self.throttle_state) * (dt / self.throttle_tau)
        else:
            self.throttle_state = desired_throttle

        if self.brake_tau > 0:
            self.brake_state += (desired_brake - self.brake_state) * (dt / self.brake_tau)
        else:
            self.brake_state = desired_brake

        # ====================================================================
        # LONGITUDINAL FORCES
        # ====================================================================
        vel_sign = np.sign(self.velocity) if abs(self.velocity) > 0.01 else 0.0
        
        # Drive force
        f_drive = self.throttle_state * self.max_drive_force  # N
        
        # Brake force
        f_brake = self.brake_state * self.max_brake_force * (-vel_sign)  # N
        
        # Engine braking (when coasting)
        if abs(self.throttle_state) < 0.05:
            base_engine_braking = 300.0  # N
            speed_factor = abs(self.velocity) * 30.0  # N·s/m
            f_engine_braking = -(base_engine_braking + speed_factor) * vel_sign
        else:
            f_engine_braking = 0.0
        
        # Aerodynamic drag: F = 0.5 * ρ * Cd * A * v²
        f_drag = -0.5 * self.rho * self.c_d * self.area * self.velocity * abs(self.velocity)  # N
        
        # Rolling resistance: F = Crr * m * g
        f_rolling = -self.c_rr * self.mass * self.g * vel_sign  # N
        
        # Cornering drag (energy lost to tire slip during steering)
        steering_drag_coeff = 0.15
        f_cornering_drag = -steering_drag_coeff * self.mass * abs(self.steering_angle) * self.velocity * abs(self.velocity)  # N
        
        # ====================================================================
        # VELOCITY INTEGRATION
        # ====================================================================
        # Net force and acceleration
        f_net = f_drive + f_brake + f_engine_braking + f_drag + f_rolling + f_cornering_drag
        acceleration = f_net / self.mass  # m/s²
        
        # Update velocity
        self.velocity += acceleration * dt
        
        # Snap to zero if nearly stopped with no input
        if abs(self.velocity) < 0.01 and desired_throttle == 0 and desired_brake == 0:
            self.velocity = 0.0
        
        # Apply velocity limit
        self.velocity = np.clip(self.velocity, -self.max_velocity * 0.5, self.max_velocity)

        # ====================================================================
        # LATERAL CONTROL - Steering
        # ====================================================================
        manual_steering = keys[pygame.K_a] or keys[pygame.K_d]

        if manual_steering:
            # Manual steering - deactivate LKA if active
            if lka_controller and lka_controller.active:
                lka_controller.deactivate()

            if keys[pygame.K_a]:
                self.steering_angle += self.max_steering_rate * dt  # LEFT
            elif keys[pygame.K_d]:
                self.steering_angle -= self.max_steering_rate * dt  # RIGHT
                
        elif lka_steering is not None:
            # LKA steering - first-order actuator with rate limit
            steering_error = lka_steering - self.steering_angle
            max_change = self.max_steering_rate * dt

            if self.steering_tau is not None and self.steering_tau > 0:
                delta = (steering_error) * (dt / self.steering_tau)
            else:
                delta = np.sign(steering_error) * max_change

            # Enforce rate limit
            if abs(delta) > max_change:
                delta = np.sign(delta) * max_change

            # Snap to target if error is very small
            if abs(steering_error) <= 1e-5:
                self.steering_angle = lka_steering
            else:
                self.steering_angle += delta
        else:
            # Return to center when no input
            if abs(self.steering_angle) > 0.01:
                self.steering_angle *= 0.9
            else:
                self.steering_angle = 0

        # Apply steering limit
        self.steering_angle = np.clip(self.steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # ====================================================================
        # VEHICLE KINEMATICS - Ackermann Steering with Tire Dynamics
        # ====================================================================
        if abs(self.velocity) > 0.01:
            # ----------------------------------------------------------------
            # Downforce Calculation (increases tire grip)
            # ----------------------------------------------------------------
            # F_downforce = 0.5 * ρ * CL * A * v²
            downforce = 0.5 * self.rho * self.c_l * self.downforce_area * self.velocity * abs(self.velocity)
            
            # Total normal force = weight + downforce
            weight = self.mass * self.g
            total_normal_force = weight + downforce
            
            # Maximum lateral grip with downforce
            base_friction_coeff = 0.8  # Road tire friction
            max_lateral_force = base_friction_coeff * total_normal_force
            max_lateral_accel_with_downforce = max_lateral_force / self.mass
            
            # ----------------------------------------------------------------
            # Speed-Dependent Handling
            # ----------------------------------------------------------------
            velocity_threshold = 5.0  # m/s
            
            if abs(self.velocity) < velocity_threshold:
                # Low speed: Simple Ackermann kinematics
                omega = self.velocity * np.tan(self.steering_angle) / self.wheelbase
            else:
                # High speed: Include tire slip and grip limits
                beta = np.arctan2(self.lr * np.tan(self.steering_angle), self.wheelbase)
                
                if abs(self.steering_angle) > 0.001:
                    turn_radius = abs(self.wheelbase / np.tan(self.steering_angle))
                    lateral_accel = (self.velocity ** 2) / turn_radius
                    
                    # Check if exceeding grip limit
                    if lateral_accel > max_lateral_accel_with_downforce:
                        # Understeer/slip - reduce effective steering
                        grip_factor = max_lateral_accel_with_downforce / lateral_accel
                        effective_steering = self.steering_angle * grip_factor
                    else:
                        effective_steering = self.steering_angle
                    
                    omega = self.velocity * np.tan(effective_steering) / self.wheelbase
                else:
                    omega = 0.0
            
            # ----------------------------------------------------------------
            # Position and Orientation Integration
            # ----------------------------------------------------------------
            prev_x, prev_y = self.x, self.y

            self.x += self.velocity * np.cos(self.theta) * dt
            self.y += self.velocity * np.sin(self.theta) * dt
            self.theta += omega * dt
            self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

            # Update wheel rotation for animation
            distance_traveled = self.velocity * dt
            self.wheel_rotation -= distance_traveled / self.wheel_radius
            self.wheel_rotation = self.wheel_rotation % (2 * np.pi)

            # Store previous position for collision detection
            self.prev_x = prev_x
            self.prev_y = prev_y
    
    # ========================================================================
    # COORDINATE CONVERSION
    # ========================================================================
    
    def get_x_pixels(self):
        """Get x position in pixels (for rendering)."""
        return self.x * self.pixels_per_meter
    
    # ============================================================
    # POSITION AND GEOMETRY HELPERS
    # ============================================================
    
    def get_y_pixels(self):
        """Get y position in pixels (for rendering)"""
        return self.y * self.pixels_per_meter

    def get_front_axle_position(self):
        """Return front axle center position (meters)"""
        front_axle_x = self.x + self.lf * np.cos(self.theta)
        front_axle_y = self.y + self.lf * np.sin(self.theta)
        return front_axle_x, front_axle_y

    def get_front_wheel_positions(self):
        """Return left and right front wheel centers (meters)"""
        front_axle_x, front_axle_y = self.get_front_axle_position()
        wheel_angle = self.theta + np.pi/2
        track_half_width = 1.5 / 2.0  # meters (typical track width / 2)

        left_wheel_x = front_axle_x + track_half_width * np.cos(wheel_angle)
        left_wheel_y = front_axle_y + track_half_width * np.sin(wheel_angle)
        right_wheel_x = front_axle_x - track_half_width * np.cos(wheel_angle)
        right_wheel_y = front_axle_y - track_half_width * np.sin(wheel_angle)

        return (left_wheel_x, left_wheel_y), (right_wheel_x, right_wheel_y)

    def get_hood_camera_position(self):
        """Get position and orientation for center-of-car first-person camera (pixels for rendering)"""
        # Camera at center of car, at eye level (pixels)
        cam_x = self.get_x_pixels()
        cam_y = self.get_y_pixels()
        cam_z = self.hood_height

        # Look-at point ahead of car (pixels)
        look_distance_m = 5.0  # meters
        look_x = (self.x + look_distance_m * np.cos(self.theta)) * self.pixels_per_meter
        look_y = (self.y + look_distance_m * np.sin(self.theta)) * self.pixels_per_meter
        look_z = self.hood_height - 2  # Look slightly down

        return (cam_x, cam_y, cam_z), (look_x, look_y, look_z)

    # ============================================================
    # 3D RENDERING
    # ============================================================

    def draw_3d(self):
        """Draw car in 3D (uses pixel coordinates for rendering)"""
        glPushMatrix()

        # Transform to car position and orientation (convert to pixels)
        x_px = self.get_x_pixels()
        y_px = self.get_y_pixels()
        glTranslatef(x_px, y_px, self.height/2)
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

        # Transform to car position and orientation (convert to pixels)
        x_px = self.get_x_pixels()
        y_px = self.get_y_pixels()
        glTranslatef(x_px, y_px, 0)
        glRotatef(np.degrees(self.theta), 0, 0, 1)

        # Draw wheels with enhanced details
        self._draw_wheels_enhanced()

        glPopMatrix()

    # ============================================================
    # 3D RENDERING PRIMITIVES
    # ============================================================

    def draw_3d(self):
        """Draw car in 3D (uses pixel coordinates for rendering)"""
        glPushMatrix()

        # Transform to car position and orientation (convert to pixels)
        x_px = self.get_x_pixels()
        y_px = self.get_y_pixels()
        glTranslatef(x_px, y_px, self.height/2)
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

        # Transform to car position and orientation (convert to pixels)
        x_px = self.get_x_pixels()
        y_px = self.get_y_pixels()
        glTranslatef(x_px, y_px, 0)
        glRotatef(np.degrees(self.theta), 0, 0, 1)

        # Draw wheels with enhanced details
        self._draw_wheels_enhanced()

        glPopMatrix()    # ============================================================
    # 3D RENDERING PRIMITIVES
    # ============================================================

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

    # ============================================================
    # WHEEL RENDERING
    # ============================================================

    # ============================================================
    # WHEEL RENDERING
    # ============================================================

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

    # ============================================================
    # COLLISION DETECTION
    # ============================================================

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


