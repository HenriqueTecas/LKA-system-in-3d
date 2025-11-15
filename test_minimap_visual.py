#!/usr/bin/env python3
"""
Visual test for minimap - generates an image showing what it should look like
"""
import pygame
import numpy as np

# Initialize pygame
pygame.init()

class SaoPauloTrack:
    """Simplified track for testing"""
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

    def _offset_line(self, line, offset):
        """Offset a line perpendicular to its direction"""
        offset_line = []
        for i in range(len(line)):
            p1 = line[i]
            p2 = line[(i + 1) % len(line)]

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)

            if length > 0:
                nx = -dy / length
                ny = dx / length

                offset_x = p1[0] + nx * offset
                offset_y = p1[1] + ny * offset
                offset_line.append((offset_x, offset_y))
            else:
                offset_line.append(p1)

        return offset_line


def create_test_minimap(size=500):
    """Create a test minimap surface"""
    surface = pygame.Surface((size, size))
    track = SaoPauloTrack(offset_x=50, offset_y=50)

    # Calculate bounds
    all_points = list(track.centerline)
    outer = track._offset_line(track.centerline, track.track_width / 2)
    inner = track._offset_line(track.centerline, -track.track_width / 2)
    all_points.extend(outer)
    all_points.extend(inner)

    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Calculate scale
    margin = 20
    track_width = max_x - min_x
    track_height = max_y - min_y
    scale_x = (size - 2 * margin) / track_width
    scale_y = (size - 2 * margin) / track_height
    scale = min(scale_x, scale_y)

    def world_to_minimap(x, y):
        map_x = (x - min_x) * scale + margin
        map_y = (y - min_y) * scale + margin
        return int(map_x), int(map_y)

    # Fill background
    surface.fill((20, 20, 20))

    # Draw border
    pygame.draw.rect(surface, (100, 100, 100), (0, 0, size, size), 3)
    pygame.draw.rect(surface, (200, 200, 200), (2, 2, size-4, size-4), 1)

    # Draw grid
    grid_color = (40, 40, 40)
    for i in range(0, size, 50):
        pygame.draw.line(surface, grid_color, (i, 0), (i, size), 1)
        pygame.draw.line(surface, grid_color, (0, i), (size, i), 1)

    # Draw bounds rectangle
    min_scaled = world_to_minimap(min_x, min_y)
    max_scaled = world_to_minimap(max_x, max_y)
    pygame.draw.rect(surface, (255, 0, 0),
                    (min_scaled[0], min_scaled[1],
                     max_scaled[0] - min_scaled[0],
                     max_scaled[1] - min_scaled[1]), 2)

    # Draw track
    outer_scaled = [world_to_minimap(x, y) for x, y in outer]
    inner_scaled = [world_to_minimap(x, y) for x, y in inner]

    if len(outer_scaled) > 2:
        pygame.draw.lines(surface, (255, 255, 255), True, outer_scaled, 2)
    if len(inner_scaled) > 2:
        pygame.draw.lines(surface, (255, 255, 255), True, inner_scaled, 2)

    # Draw centerline
    centerline_scaled = [world_to_minimap(x, y) for x, y in track.centerline]
    for i in range(0, len(centerline_scaled) - 1, 2):
        p1 = centerline_scaled[i]
        p2 = centerline_scaled[i + 1]
        pygame.draw.line(surface, (128, 128, 128), p1, p2, 1)

    # Add labels
    font = pygame.font.Font(None, 20)
    scale_text = font.render(f"Scale: {scale:.3f}", True, (255, 255, 0))
    surface.blit(scale_text, (5, 5))

    bounds_text = font.render(f"Bounds: {min_scaled} to {max_scaled}", True, (255, 0, 0))
    surface.blit(bounds_text, (5, size - 20))

    return surface


if __name__ == "__main__":
    # Create test minimap
    minimap = create_test_minimap(500)

    # Save to file
    pygame.image.save(minimap, "test_minimap_output.png")
    print("Created test_minimap_output.png")
    print("This shows what the minimap SHOULD look like:")
    print("- White track should fit within red rectangle")
    print("- Red rectangle should be near the edges with ~20px margin")
    print("- Grid shows 50px spacing")
    print("- Scale should be ~0.419")
