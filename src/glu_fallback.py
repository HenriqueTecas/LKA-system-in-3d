"""
GLU Fallback helpers.

Some environments (CI/headless containers) do not ship libGLU, causing
PyOpenGL to expose GLU functions that fail at call time. These helpers
try GLU first and fall back to equivalent OpenGL implementations so the
simulation can start without native GLU.
"""

import math
import numpy as np
from OpenGL.GL import (
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUAD_STRIP,
    glBegin,
    glEnd,
    glFrustum,
    glLoadIdentity,
    glMatrixMode,
    glMultMatrixf,
    glNormal3f,
    glVertex3f,
)

try:
    from OpenGL.GLU import gluPerspective as _gluPerspective
    from OpenGL.GLU import gluLookAt as _gluLookAt
    from OpenGL.GLU import gluNewQuadric, gluDeleteQuadric, gluSphere
except Exception:  # GLU missing or unavailable
    _gluPerspective = None
    _gluLookAt = None
    gluNewQuadric = None
    gluDeleteQuadric = None
    gluSphere = None


def _can_use(func) -> bool:
    """Return True when a GLU function pointer exists and is callable."""
    if func is None:
        return False
    try:
        return bool(func)
    except Exception:
        return False


def perspective(fov_y: float, aspect: float, z_near: float, z_far: float):
    """gluPerspective replacement."""
    if _can_use(_gluPerspective):
        try:
            return _gluPerspective(fov_y, aspect, z_near, z_far)
        except Exception:
            pass

    top = z_near * math.tan(math.radians(fov_y) / 2.0)
    bottom = -top
    right = top * aspect
    left = -right

    glFrustum(left, right, bottom, top, z_near, z_far)


def look_at(eye, center, up):
    """gluLookAt replacement."""
    if _can_use(_gluLookAt):
        try:
            return _gluLookAt(
                eye[0], eye[1], eye[2],
                center[0], center[1], center[2],
                up[0], up[1], up[2],
            )
        except Exception:
            pass

    eye = np.array(eye, dtype=np.float32)
    center = np.array(center, dtype=np.float32)
    up = np.array(up, dtype=np.float32)

    f = center - eye
    f_norm = np.linalg.norm(f) or 1.0
    f /= f_norm

    up_norm = np.linalg.norm(up) or 1.0
    up /= up_norm

    s = np.cross(f, up)
    s_norm = np.linalg.norm(s) or 1.0
    s /= s_norm

    u = np.cross(s, f)

    m = np.array([
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye), 1.0],
    ], dtype=np.float32)

    glMultMatrixf(m)


def draw_sphere(radius: float, slices: int = 6, stacks: int = 6):
    """Draw a sphere without relying on GLU."""
    if _can_use(gluNewQuadric) and _can_use(gluSphere) and _can_use(gluDeleteQuadric):
        try:
            quadric = gluNewQuadric()
            gluSphere(quadric, radius, max(3, slices), max(2, stacks))
            gluDeleteQuadric(quadric)
            return
        except Exception:
            pass

    slices = max(3, slices)
    stacks = max(2, stacks)

    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)

        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * (j % slices) / slices
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x, y, z0 / radius if radius else 0.0)
            glVertex3f(x * zr0, y * zr0, z0)

            glNormal3f(x, y, z1 / radius if radius else 0.0)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()
