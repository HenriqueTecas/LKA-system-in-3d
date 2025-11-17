"""
Main simulation module - Entry point for 3D Robotics Lab.

This module initializes the display, creates all components,
and runs the main simulation loop.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os
import platform
import time

# Import configuration
from .config import *

# Import all components
from .car import Car
#from .camera_sensor import CameraSensor
from .realistic_camera import RealisticCameraSensor
from .lka_controller import PurePursuitLKA
from .mpc_controller import MPCLaneKeeping
from .track import SaoPauloTrack
from .renderer import Renderer3D
from .minimap import Minimap
from .hud import HUD


def init_display():
    """Initialize Pygame and OpenGL display."""
    # Set environment variables for better OpenGL compatibility
    os.environ['SDL_VIDEO_X11_FORCE_EGL'] = '0'  # Disable EGL, use GLX instead
    os.environ['PYOPENGL_PLATFORM'] = 'glx'  # Force GLX platform

    # Check if display is available on Unix-like systems
    # On Windows, SDL/pygame does not rely on the DISPLAY env var so skip this check there.
    if platform.system() != 'Windows' and 'DISPLAY' not in os.environ:
        print("ERROR: No display found!")
        print("Solutions:")
        print("1. If running over SSH: use 'ssh -X' for X11 forwarding")
        print("2. If local: ensure you're running in a graphical environment")
        print("3. For headless: install and use xvfb-run")
        sys.exit(1)

    # Initialize Pygame
    pygame.init()

    # Try to create OpenGL context with fallback options
    screen = None
    error_messages = []

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
            print(f"[OK] Successfully initialized with {description}")
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
        print("3. Install required GL libraries")
        print("4. Try software rendering: 'export LIBGL_ALWAYS_SOFTWARE=1'")
        sys.exit(1)

    return screen


def main():
    """Main simulation loop."""
    # Initialize display
    screen = init_display()
    clock = pygame.time.Clock()

    # Create track
    track = SaoPauloTrack(offset_x=50, offset_y=50)

    # Create car
    start_x, start_y, start_theta = track.get_start_position(lane_number=1)  # Start in lane 1 (negative offset = LEFT of centerline)
    car = Car(start_x, start_y, start_theta)
    car.track = track  # Store reference for camera

    # Create camera sensor - USING SIMPLE CAMERA (realistic has issues)
    # Simple FOV-based camera works reliably
    # Create camera sensor with distance-based confidence
    camera = RealisticCameraSensor(car)
    # To use simple camera: camera = CameraSensor(car)

    # Create LKA controllers
    lka = PurePursuitLKA(car, camera)
    mpc = MPCLaneKeeping(car, camera)

    # Create renderer
    renderer = Renderer3D(WIDTH, HEIGHT)

    # Create minimap
    minimap = Minimap(MINIMAP_SIZE, track)

    # Create HUD
    hud = HUD()

    # Pre-allocate texture for overlay (performance optimization)
    overlay_texture_id = glGenTextures(1)

    # Main loop -> fixed-timestep physics + rendering
    running = True
    physics_dt = PHYSICS_DT
    accumulator = 0.0
    prev_time = time.perf_counter()

    while running:
        # Time management
        current_time = time.perf_counter()
        frame_time = current_time - prev_time
        prev_time = current_time
        # Clamp very large frame times (e.g., when debugger paused)
        if frame_time > 0.25:
            frame_time = 0.25
        accumulator += frame_time

        # Handle events once per frame (input / toggles)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # Toggle Pure Pursuit LKA
                    if mpc.active:
                        mpc.deactivate()
                    lka.toggle()
                elif event.key == pygame.K_g:
                    # Toggle MPC LKA
                    if lka.active:
                        lka.deactivate()
                    mpc.toggle()

        # Capture current key state (will be used during physics steps)
        keys = pygame.key.get_pressed()

        # Run physics updates at fixed timestep. Controllers and vehicle state
        # are advanced in these steps so their timing is deterministic.
        while accumulator >= physics_dt:
            # Detect lanes once per physics step (shared by controllers and visualization)
            camera.detect_lanes(track)
            
            # Calculate LKA steering (Pure Pursuit or MPC) at physics rate
            lka_steering = None
            if lka.active:
                lka_steering = lka.calculate_steering(track)
            elif mpc.active:
                lka_steering = mpc.calculate_steering(track)

            # Update car (pass both LKA controllers)
            active_lka = lka if lka.active else (mpc if mpc.active else None)
            car.update(physics_dt, keys, lka_steering, active_lka)

            # Collision detection disabled for open-world driving
            # if not car.is_on_track(track):
            #     car.handle_collision()

            accumulator -= physics_dt

        # === 3D RENDERING ===
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup 3D view
        renderer.setup_3d_view(car)

        # Draw track
        track.draw_3d()

        # Draw lane markers
        renderer.draw_lane_markers_3d(camera, track)

        # Draw Pure Pursuit lookahead points (yellow)
        renderer.draw_lookahead_point_3d(lka)

        # Draw MPC predicted trajectory (silver)
        renderer.draw_mpc_trajectory_3d(mpc)

        # Draw ONLY wheels (car body invisible for first-person view)
        car.draw_wheels_only_3d()

        # === 2D OVERLAY RENDERING ===
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

        # Render HUD with FPS (pass both controllers)
        current_fps = clock.get_fps()
        hud.render(overlay_surface, car, camera, lka, current_fps, mpc)

        # Render minimap (pass both controllers)
        minimap_surface = minimap.render(car, camera, lka, mpc)
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
            "W/S: Accel/Brake | A/D: Steer | F: Pure Pursuit | G: MPC | ESC: Exit"
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

        # Convert pygame surface to OpenGL texture
        texture_data = pygame.image.tostring(overlay_surface, "RGBA", False)

        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Bind and update existing texture
        glBindTexture(GL_TEXTURE_2D, overlay_texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Draw textured quad
        glEnable(GL_TEXTURE_2D)
        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, 0)
        glTexCoord2f(1, 0); glVertex2f(WIDTH, 0)
        glTexCoord2f(1, 1); glVertex2f(WIDTH, HEIGHT)
        glTexCoord2f(0, 1); glVertex2f(0, HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)
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

    # Cleanup
    glDeleteTextures([overlay_texture_id])
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
