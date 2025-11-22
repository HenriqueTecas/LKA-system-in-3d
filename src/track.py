"""
Track Module - São Paulo F1 Circuit
Part of the 3D Robotics Lab simulation.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy import interpolate


class SaoPauloTrack:
    """São Paulo F1 Circuit (Interlagos) - Scaled to real dimensions
    
    Real track: 4.309 km lap length, ~10m width
    At 12 pixels/meter: needs ~19.4x scale factor from base coordinates
    """
    def __init__(self, offset_x=500, offset_y=500):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.lane_width = 48  # 4 meters per lane at 12 pixels/meter
        self.track_width = 2 * self.lane_width  # 10 meters total (120 pixels)

        # Scale factor to achieve 4309m lap length
        # Original track in base coords: 2663.2 pixels
        # Target: 4309m × 12 px/m = 51,708 pixels
        # Scale: 51,708 / 2663.2 ≈ 19.4
        scale = 19.4
        
        # Base centerline coordinates (original track shape - control points)
        control_points = [
            (800, 600), (750, 500), (650, 400), (550, 350), (450, 330),
            (350, 300), (250, 250), (200, 180), (180, 120), (200, 60),
            (300, 30), (500, 30), (700, 30), (900, 30), (1100, 50),
            (1150, 100), (1180, 200), (1180, 300), (1180, 400), (1150, 500),
            (1100, 550), (1000, 600), (900, 600), (800, 600),
        ]

        # Scale control points
        control_points = [(x * scale + offset_x, y * scale + offset_y)
                         for x, y in control_points]
        
        # Use control points directly as centerline
        self.centerline = control_points


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

        # Draw sphere at top - OPTIMIZED: reduced from 8,8 to 6,6
        glPushMatrix()
        glTranslatef(x, y, height)
        quadric = gluNewQuadric()
        gluSphere(quadric, 3, 6, 6)  # Reduced detail for performance
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
        gluSphere(quadric, 5, 6, 6)  # Reduced detail for performance
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
        gluSphere(quadric, 2, 4, 4)  # OPTIMIZED: reduced from 6,6 to 4,4
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


