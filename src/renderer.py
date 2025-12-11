"""
Renderer Module - 3D OpenGL Scene Management
Part of the 3D Robotics Lab simulation.
"""
from OpenGL.GL import *
import numpy as np
from .config import PIXELS_PER_METER
from .glu_fallback import perspective, look_at, draw_sphere


class Renderer3D:
    """3D OpenGL renderer"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels_per_meter = float(PIXELS_PER_METER)
        self.camera_view_mode = "chase"  # "chase" or "realistic"
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

    def setup_3d_view(self, car, camera_sensor=None):
        """Setup 3D perspective for main view (supports lane-camera POV toggle)"""
        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        if self.camera_view_mode == "realistic" and camera_sensor is not None:
            fov_deg = np.degrees(camera_sensor.horizontal_fov)
            perspective(fov_deg, self.width / self.height, 0.1, 1000.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            cam_x, cam_y, cam_z = camera_sensor.get_camera_position()
            cam_pos_pixels = (cam_x * self.pixels_per_meter,
                              cam_y * self.pixels_per_meter,
                              cam_z * self.pixels_per_meter)

            look_distance = 20.0  # meters
            pitch = camera_sensor.pitch_angle
            look_x = cam_x + look_distance * np.cos(car.theta)
            look_y = cam_y + look_distance * np.sin(car.theta)
            look_z = cam_z - look_distance * np.tan(pitch)
            look_pos_pixels = (look_x * self.pixels_per_meter,
                               look_y * self.pixels_per_meter,
                               look_z * self.pixels_per_meter)

            look_at(
                (cam_pos_pixels[0], cam_pos_pixels[1], cam_pos_pixels[2]),
                (look_pos_pixels[0], look_pos_pixels[1], look_pos_pixels[2]),
                (0, 0, 1),
            )
        else:
            perspective(60, self.width / self.height, 1.0, 5000.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            # Hood camera position
            cam_pos, look_pos = car.get_hood_camera_position()
            look_at(
                (cam_pos[0], cam_pos[1], cam_pos[2]),
                (look_pos[0], look_pos[1], look_pos[2]),
                (0, 0, 1),
            )

    def draw_lane_markers_3d(self, camera, track):
        """Draw 3D markers for detected lane points - ONLY FOR CURRENT LANE"""
        # PERFORMANCE FIX: Use cached measurement instead of re-detecting
        left_lane, center_lane, right_lane = camera.last_measurement

        # Determine which lane we're in and which boundaries to display
        current_lane = camera.current_lane

        if current_lane == "LEFT":
            # In left lane: show left outer boundary and center line
            lane_left_boundary = left_lane
            lane_right_boundary = center_lane
        elif current_lane == "RIGHT":
            # In right lane: show center line and right outer boundary
            lane_left_boundary = center_lane
            lane_right_boundary = right_lane
        else:
            # Unknown - show all (fallback)
            lane_left_boundary = left_lane
            lane_right_boundary = right_lane

        glDisable(GL_LIGHTING)

        # Draw left boundary of current lane (red)
        glColor3f(1.0, 0.0, 0.0)
        for point in lane_left_boundary:
            # Handle both 3-tuple (x,y,angle) and 4-tuple (x,y,angle,confidence)
            mx, my = point[0], point[1]
            px = mx * self.pixels_per_meter
            py = my * self.pixels_per_meter
            self._draw_marker(px, py, 5.0, 8.0)

        # Draw right boundary of current lane (cyan)
        glColor3f(0.0, 0.8, 1.0)
        for point in lane_right_boundary:
            mx, my = point[0], point[1]
            px = mx * self.pixels_per_meter
            py = my * self.pixels_per_meter
            self._draw_marker(px, py, 5.0, 8.0)

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

        # Draw sphere at top - OPTIMIZED: reduced from 8,8 to 6,6
        glTranslatef(0, 0, height)
        draw_sphere(radius, 6, 6)

        glPopMatrix()

    # Removed unused LKA/MPC draw helpers (lookahead points and MPC trajectory)

    def draw_hybrid_target_3d(self, hybrid, car):
        """Draw Hybrid Controller's target point and direction vector (yellow)"""
        if hybrid.mode == hybrid.MODE_MANUAL:
            return

        glDisable(GL_LIGHTING)

        # Draw center line points (small yellow spheres) when both boundaries visible
        if hasattr(hybrid, 'center_line_points') and len(hybrid.center_line_points) > 0:
            glColor3f(1.0, 1.0, 0.3)  # Light yellow
            for cx, cy in hybrid.center_line_points:
                px = cx * self.pixels_per_meter
                py = cy * self.pixels_per_meter

                glLineWidth(2)
                glBegin(GL_LINES)
                glVertex3f(px, py, 0)
                glVertex3f(px, py, 15)
                glEnd()

                glPushMatrix()
                glTranslatef(px, py, 15)
                draw_sphere(4, 4, 4)
                glPopMatrix()

        # Draw target point if available (BRIGHT YELLOW SPHERE)
        if hybrid.target_point is not None:
            tx, ty = hybrid.target_point
            px = tx * self.pixels_per_meter
            py = ty * self.pixels_per_meter

            glColor3f(1.0, 1.0, 0.0)
            glLineWidth(5)
            glBegin(GL_LINES)
            glVertex3f(px, py, 0)
            glVertex3f(px, py, 40)
            glEnd()

            glPushMatrix()
            glTranslatef(px, py, 40)
            draw_sphere(10, 8, 8)
            glPopMatrix()

        # Draw direction vector from car to target (YELLOW ARROW)
        if hybrid.target_direction is not None:
            car_x = car.x * self.pixels_per_meter
            car_y = car.y * self.pixels_per_meter

            vector_length = 15.0 * self.pixels_per_meter
            end_x = car_x + vector_length * np.cos(hybrid.target_direction)
            end_y = car_y + vector_length * np.sin(hybrid.target_direction)

            glColor3f(1.0, 1.0, 0.0)
            glLineWidth(4)
            glBegin(GL_LINES)
            glVertex3f(car_x, car_y, 25)
            glVertex3f(end_x, end_y, 25)
            glEnd()

            arrow_size = 20
            arrow_angle = 0.4
            left_x = end_x - arrow_size * np.cos(hybrid.target_direction - arrow_angle)
            left_y = end_y - arrow_size * np.sin(hybrid.target_direction - arrow_angle)
            right_x = end_x - arrow_size * np.cos(hybrid.target_direction + arrow_angle)
            right_y = end_y - arrow_size * np.sin(hybrid.target_direction + arrow_angle)

            glBegin(GL_TRIANGLES)
            glVertex3f(end_x, end_y, 25)
            glVertex3f(left_x, left_y, 25)
            glVertex3f(right_x, right_y, 25)
            glEnd()

        glEnable(GL_LIGHTING)
