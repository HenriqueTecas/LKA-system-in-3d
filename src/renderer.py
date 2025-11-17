"""
Renderer Module - 3D OpenGL Scene Management
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from .config import PIXELS_PER_METER


class Renderer3D:
    """3D OpenGL renderer"""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels_per_meter = float(PIXELS_PER_METER)
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
        """Draw 3D markers for detected lane points - ONLY FOR CURRENT LANE"""
        left_lane, right_lane, center_lane = camera.detect_lanes(track)

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
        quadric = gluNewQuadric()
        gluSphere(quadric, radius, 6, 6)  # Reduced detail for performance
        gluDeleteQuadric(quadric)

        glPopMatrix()

    def draw_lookahead_point_3d(self, lka):
        """Draw ALL LKA lane center points and highlight the selected lookahead point"""
        if not lka.active:
            return

        glDisable(GL_LIGHTING)

        # Draw ALL lane center points (smaller, semi-transparent yellow)
        if hasattr(lka, 'lane_center_points') and lka.lane_center_points:
            glColor3f(1.0, 1.0, 0.5)  # Light yellow
            for cx, cy, dist in lka.lane_center_points:
                # Convert meters to pixels
                px = cx * self.pixels_per_meter
                py = cy * self.pixels_per_meter
                # Draw small vertical marker
                glLineWidth(2)
                glBegin(GL_LINES)
                glVertex3f(px, py, 0)
                glVertex3f(px, py, 15)
                glEnd()

                # Draw small sphere at top
                glPushMatrix()
                glTranslatef(px, py, 15)
                quadric = gluNewQuadric()
                gluSphere(quadric, 4, 4, 4)  # Small sphere
                gluDeleteQuadric(quadric)
                glPopMatrix()

        # Draw the SELECTED lookahead point (larger, bright yellow)
        if hasattr(lka, 'lookahead_point') and lka.lookahead_point:
            lx, ly = lka.lookahead_point
            # Convert meters to pixels
            px = lx * self.pixels_per_meter
            py = ly * self.pixels_per_meter

            # Draw vertical marker
            glColor3f(1.0, 1.0, 0.0)  # Bright yellow
            glLineWidth(4)
            glBegin(GL_LINES)
            glVertex3f(px, py, 0)
            glVertex3f(px, py, 35)
            glEnd()

            # Draw large sphere at top (this is the actual target)
            glPushMatrix()
            glTranslatef(px, py, 35)
            quadric = gluNewQuadric()
            gluSphere(quadric, 8, 6, 6)  # Large sphere for selected point
            gluDeleteQuadric(quadric)
            glPopMatrix()

        glEnable(GL_LIGHTING)

    def draw_mpc_trajectory_3d(self, mpc):
        """Draw MPC predicted trajectory as silver markers"""
        if not mpc.active:
            return

        if not hasattr(mpc, 'predicted_trajectory') or not mpc.predicted_trajectory:
            return

        glDisable(GL_LIGHTING)

        # Draw MPC trajectory points (silver/gray)
        glColor3f(0.75, 0.75, 0.75)  # Silver color

        for mx, my in mpc.predicted_trajectory:
            # Convert meters to pixels
            px = mx * self.pixels_per_meter
            py = my * self.pixels_per_meter
            # Draw vertical marker
            glLineWidth(2)
            glBegin(GL_LINES)
            glVertex3f(px, py, 0)
            glVertex3f(px, py, 20)
            glEnd()

            # Draw sphere at top
            glPushMatrix()
            glTranslatef(px, py, 20)
            quadric = gluNewQuadric()
            gluSphere(quadric, 5, 6, 6)  # Medium-sized sphere
            gluDeleteQuadric(quadric)
            glPopMatrix()

        glEnable(GL_LIGHTING)



