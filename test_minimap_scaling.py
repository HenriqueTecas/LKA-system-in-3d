#!/usr/bin/env python3
"""
Test script to verify minimap scaling calculations
"""
import numpy as np

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


def test_minimap_scaling():
    """Test the minimap scaling logic"""
    print("Testing Minimap Scaling")
    print("=" * 70)

    # Create track with same parameters as real code
    track = SaoPauloTrack(offset_x=50, offset_y=50)

    # Get all track points
    all_points = list(track.centerline)
    outer = track._offset_line(track.centerline, track.track_width / 2)
    inner = track._offset_line(track.centerline, -track.track_width / 2)
    all_points.extend(outer)
    all_points.extend(inner)

    # Find bounds
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    print(f"Track centerline points: {len(track.centerline)}")
    print(f"First centerline point: {track.centerline[0]}")
    print(f"Last centerline point: {track.centerline[-1]}")
    print()

    print(f"Track bounds:")
    print(f"  X: [{min_x:.1f}, {max_x:.1f}] (width: {max_x - min_x:.1f})")
    print(f"  Y: [{min_y:.1f}, {max_y:.1f}] (height: {max_y - min_y:.1f})")
    print()

    # Calculate scale for different minimap sizes
    for minimap_size in [400, 500]:
        margin = 20
        track_width = max_x - min_x
        track_height = max_y - min_y

        scale_x = (minimap_size - 2 * margin) / track_width
        scale_y = (minimap_size - 2 * margin) / track_height
        scale = min(scale_x, scale_y)

        print(f"Minimap size: {minimap_size}x{minimap_size}")
        print(f"  Available space: {minimap_size - 2*margin}x{minimap_size - 2*margin}")
        print(f"  scale_x: {scale_x:.4f}")
        print(f"  scale_y: {scale_y:.4f}")
        print(f"  Final scale: {scale:.4f}")

        # Test transformation
        def world_to_minimap(x, y):
            map_x = (x - min_x) * scale + margin
            map_y = (y - min_y) * scale + margin
            return int(map_x), int(map_y)

        # Transform corners
        min_scaled = world_to_minimap(min_x, min_y)
        max_scaled = world_to_minimap(max_x, max_y)

        print(f"  World bounds ({min_x:.0f},{min_y:.0f}) to ({max_x:.0f},{max_y:.0f})")
        print(f"  Minimap bounds {min_scaled} to {max_scaled}")
        print(f"  Minimap coverage: {max_scaled[0] - min_scaled[0]}x{max_scaled[1] - min_scaled[1]} pixels")
        print()

        # Test a few track points
        print(f"  Sample transformations:")
        for i in [0, len(track.centerline)//2, -1]:
            world_pt = track.centerline[i]
            minimap_pt = world_to_minimap(world_pt[0], world_pt[1])
            print(f"    Point {i}: world={world_pt} -> minimap={minimap_pt}")
        print()


if __name__ == "__main__":
    test_minimap_scaling()
