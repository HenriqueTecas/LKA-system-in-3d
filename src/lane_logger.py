"""
Lane detection logger and time-series plotter.

Records lane detections over time and renders a small history chart on the HUD.
Also streams detections to a CSV file for offline inspection.
"""

import csv
import time
from collections import deque
from pathlib import Path
import math

import pygame


class LaneDetectionLogger:
    """Log lane detections and render a short history plot."""

    def __init__(
        self,
        window_seconds=12.0,
        max_samples=2000,
        csv_dir="logs",
    ):
        self.window_seconds = float(window_seconds)
        self.samples = deque(maxlen=max_samples)
        self.start_time = time.perf_counter()
        self.font = pygame.font.Font(None, 18)
        self.small_font = pygame.font.Font(None, 16)

        # CSV logging setup
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        logs_dir = Path(csv_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = logs_dir / f"lane_detections_{timestamp}.csv"
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "time_s",
                "lane",
                "left_offset_m",
                "right_offset_m",
                "center_offset_m",
                "left_conf",
                "right_conf",
                "left_points",
                "center_points",
                "right_points",
            ]
        )
        self._flush_every = 60
        self._flush_counter = 0
        self._last_draw_time = 0.0

    def close(self):
        """Close CSV file safely."""
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def log_detection(self, car, camera, left_pts, center_pts, right_pts):
        """
        Capture a detection snapshot.

        Offsets are expressed in the vehicle frame (positive = left of car center).
        """
        now = time.perf_counter() - self.start_time
        lane = camera.current_lane

        # Choose which boundaries define the current lane
        if lane == "LEFT":
            lane_left_pts, lane_right_pts = left_pts, center_pts
        elif lane == "RIGHT":
            lane_left_pts, lane_right_pts = center_pts, right_pts
        else:
            lane_left_pts, lane_right_pts = left_pts, right_pts

        lane_left_offset, lane_left_conf = self._closest_offset(car, lane_left_pts)
        lane_right_offset, lane_right_conf = self._closest_offset(car, lane_right_pts)
        center_offset = None

        if lane_left_offset is not None and lane_right_offset is not None:
            center_offset = (lane_left_offset + lane_right_offset) / 2.0

        sample = {
            "t": now,
            "lane": lane,
            "lane_left_offset": lane_left_offset,
            "lane_right_offset": lane_right_offset,
            "center_offset": center_offset,
            "lane_left_conf": lane_left_conf,
            "lane_right_conf": lane_right_conf,
            "counts": (
                len(left_pts) if left_pts else 0,
                len(center_pts) if center_pts else 0,
                len(right_pts) if right_pts else 0,
            ),
        }

        self.samples.append(sample)
        self._write_csv_row(sample)

    def _write_csv_row(self, sample):
        """Persist the most recent detection to CSV."""
        if not self._csv_writer:
            return

        self._csv_writer.writerow(
            [
                f"{sample['t']:.3f}",
                sample["lane"],
                "" if sample["lane_left_offset"] is None else f"{sample['lane_left_offset']:.4f}",
                "" if sample["lane_right_offset"] is None else f"{sample['lane_right_offset']:.4f}",
                "" if sample["center_offset"] is None else f"{sample['center_offset']:.4f}",
                f"{sample['lane_left_conf']:.3f}",
                f"{sample['lane_right_conf']:.3f}",
                sample["counts"][0],
                sample["counts"][1],
                sample["counts"][2],
            ]
        )
        self._flush_counter += 1
        if self._flush_counter >= self._flush_every:
            self._csv_file.flush()
            self._flush_counter = 0

    def _closest_offset(self, car, boundary_points):
        """Return lateral offset (m) of the closest boundary point to the car frame."""
        if not boundary_points:
            return None, 0.0

        cos_t = math.cos(car.theta)
        sin_t = math.sin(car.theta)

        best = None

        for pt in boundary_points:
            px, py = pt[0], pt[1]
            dx = px - car.x
            dy = py - car.y

            # Car-centric frame: x forward, y left
            x_local = dx * cos_t + dy * sin_t
            y_local = -dx * sin_t + dy * cos_t

            # Favor points in front of the car but fall back to any point
            ahead_bonus = 0.0 if x_local >= -0.5 else 1.0
            cost = ahead_bonus + abs(y_local)

            if best is None or cost < best[0]:
                conf = pt[3] if len(pt) > 3 else 1.0
                best = (cost, y_local, conf)

        if best is None:
            return None, 0.0

        return best[1], best[2]

    def draw_plot(self, surface, position=(20, 560), size=(520, 200)):
        """Render a time-series plot of lane offsets on the given surface."""
        if not self.samples:
            return

        plot_rect = pygame.Rect(position, size)
        chart_rect = plot_rect.inflate(-30, -40)

        # Background and border
        pygame.draw.rect(surface, (0, 0, 0, 180), plot_rect)
        pygame.draw.rect(surface, (200, 200, 200), plot_rect, 1)

        # Filter samples inside the time window
        now = time.perf_counter() - self.start_time
        window_start = now - self.window_seconds
        window_samples = [s for s in self.samples if s["t"] >= window_start]
        if len(window_samples) < 2:
            return

        # Determine vertical scaling
        values = [
            v
            for s in window_samples
            for v in (s["lane_left_offset"], s["lane_right_offset"], s["center_offset"])
            if v is not None
        ]
        max_mag = max(abs(v) for v in values) if values else 1.0
        scale = max(1.5, min(8.0, max_mag * 1.4))

        def _to_screen(sample, key):
            val = sample.get(key)
            if val is None:
                return None
            x_norm = (sample["t"] - window_start) / self.window_seconds
            x = chart_rect.left + x_norm * chart_rect.width
            y = chart_rect.centery - (val / scale) * (chart_rect.height / 2)
            return x, y

        # Axes
        pygame.draw.line(surface, (120, 120, 120), (chart_rect.left, chart_rect.centery), (chart_rect.right, chart_rect.centery), 1)
        pygame.draw.line(surface, (120, 120, 120), (chart_rect.left, chart_rect.top), (chart_rect.left, chart_rect.bottom), 1)

        # Series
        self._draw_series(surface, window_samples, _to_screen, "lane_left_offset", (255, 80, 80))
        self._draw_series(surface, window_samples, _to_screen, "lane_right_offset", (0, 180, 255))
        self._draw_series(surface, window_samples, _to_screen, "center_offset", (255, 230, 80))

        # Labels
        title = self.font.render(f"Lane detections (last {int(self.window_seconds)}s)", True, (255, 255, 255))
        surface.blit(title, (plot_rect.left + 10, plot_rect.top + 8))

        scale_text = self.small_font.render(f"+/- {scale:.1f} m", True, (200, 200, 200))
        surface.blit(scale_text, (plot_rect.right - scale_text.get_width() - 10, plot_rect.top + 8))

        legend_items = [
            ("Left boundary", (255, 80, 80)),
            ("Right boundary", (0, 180, 255)),
            ("Lane center", (255, 230, 80)),
        ]
        legend_y = plot_rect.bottom - 18
        legend_x = plot_rect.left + 10
        for text, color in legend_items:
            pygame.draw.line(surface, color, (legend_x, legend_y), (legend_x + 16, legend_y), 2)
            label = self.small_font.render(text, True, (230, 230, 230))
            surface.blit(label, (legend_x + 20, legend_y - 8))
            legend_x += label.get_width() + 50

        # CSV filename for traceability
        csv_label = self.small_font.render(f"CSV: {self.csv_path}", True, (180, 180, 180))
        surface.blit(csv_label, (plot_rect.left + 10, plot_rect.top + 26))

    def _draw_series(self, surface, samples, to_screen, key, color):
        """Draw a polyline for a given series key."""
        points = []
        for s in samples:
            pt = to_screen(s, key)
            if pt is not None:
                points.append(pt)

        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)
