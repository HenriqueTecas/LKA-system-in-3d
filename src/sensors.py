"""
Simulated ego sensors (IMU, GNSS, wheel encoder) with rate, latency, and noise.
These provide realistic, noisy measurements instead of perfect ground truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _wrap_angle(rad: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(rad), np.cos(rad))


@dataclass
class TimedMeasurement:
    timestamp: float
    data: dict


class _LatencyBuffer:
    """Small helper to model sensor latency via a FIFO of time-tagged samples."""

    def __init__(self, latency_s: float):
        self.latency_s = max(0.0, float(latency_s))
        self._buffer: list[tuple[float, dict]] = []
        self.latest: Optional[TimedMeasurement] = None

    def push(self, now: float, payload: dict):
        release_time = now + self.latency_s
        self._buffer.append((release_time, payload))

    def pop_ready(self, now: float) -> Optional[TimedMeasurement]:
        if not self._buffer:
            return None
        ready = [item for item in self._buffer if item[0] <= now]
        if not ready:
            return None
        release_time, payload = ready[-1]
        self._buffer = [item for item in self._buffer if item[0] > release_time]
        self.latest = TimedMeasurement(timestamp=release_time, data=payload)
        return self.latest


class IMUSensor:
    """Simulated IMU providing yaw rate and body-frame acceleration."""

    def __init__(
        self,
        car,
        sample_rate_hz: float = 100.0,
        gyro_noise_std: float = 0.0015,  # rad/s (mid-grade MEMS noise)
        accel_noise_std: float = 0.05,   # m/s^2 (lightly noisy accel)
        bias_walk_std: float = 5e-5,     # modest bias drift
        latency_s: float = 0.01,
        rng: Optional[np.random.Generator] = None,
    ):
        self.car = car
        self.sample_rate = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        self.gyro_noise_std = gyro_noise_std
        self.accel_noise_std = accel_noise_std
        self.bias_walk_std = bias_walk_std
        self.buffer = _LatencyBuffer(latency_s)
        self.rng = rng or np.random.default_rng()
        self._accumulator = 0.0
        self._last_theta = self.car.theta
        self._last_vx = self.car.velocity * np.cos(self.car.theta)
        self._last_vy = self.car.velocity * np.sin(self.car.theta)
        self._gyro_bias = 0.0
        self._accel_bias = np.zeros(2)

    def update(self, dt: float, now: float):
        """Integrate time and emit a sample when the rate budget allows."""
        if self.sample_interval == 0:
            return
        self._accumulator += dt
        if self._accumulator < self.sample_interval:
            return
        self._accumulator -= self.sample_interval

        theta = self.car.theta
        vx = self.car.velocity * np.cos(theta)
        vy = self.car.velocity * np.sin(theta)

        dt_safe = max(dt, 1e-3)
        yaw_rate_true = _wrap_angle(theta - self._last_theta) / dt_safe

        ax_world = (vx - self._last_vx) / dt_safe
        ay_world = (vy - self._last_vy) / dt_safe

        # Rotate world acceleration into body frame (gravity removed for simplicity)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        ax_body = ax_world * cos_t + ay_world * sin_t
        ay_body = -ax_world * sin_t + ay_world * cos_t

        # Bias random walk
        self._gyro_bias += float(self.rng.normal(0.0, self.bias_walk_std))
        self._accel_bias += self.rng.normal(0.0, self.bias_walk_std, size=2)

        yaw_rate_meas = yaw_rate_true + self._gyro_bias + float(self.rng.normal(0.0, self.gyro_noise_std))
        accel_meas = np.array([ax_body, ay_body]) + self._accel_bias + self.rng.normal(0.0, self.accel_noise_std, size=2)

        self.buffer.push(now, {
            "yaw_rate": yaw_rate_meas,
            "ax_body": accel_meas[0],
            "ay_body": accel_meas[1],
        })

        self._last_theta = theta
        self._last_vx = vx
        self._last_vy = vy

    def read(self, now: Optional[float] = None) -> Optional[TimedMeasurement]:
        now = time.perf_counter() if now is None else now
        meas = self.buffer.pop_ready(now)
        return meas or self.buffer.latest


class GNSSSensor:
    """Simulated GNSS providing position and heading (dual-antenna style)."""

    def __init__(
        self,
        car,
        sample_rate_hz: float = 10.0,
        pos_noise_std: float = 0.2,       # meters (between tight RTK and consumer GNSS)
        heading_noise_std: float = 0.003, # radians (~0.17 deg)
        latency_s: float = 0.05,
        dropout_prob: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.car = car
        self.sample_rate = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        self.pos_noise_std = pos_noise_std
        self.heading_noise_std = heading_noise_std
        self.dropout_prob = dropout_prob
        self.buffer = _LatencyBuffer(latency_s)
        self._accumulator = 0.0
        self.rng = rng or np.random.default_rng()

    def update(self, dt: float, now: float):
        if self.sample_interval == 0:
            return
        self._accumulator += dt
        if self._accumulator < self.sample_interval:
            return
        self._accumulator -= self.sample_interval

        if self.dropout_prob > 0.0 and self.rng.random() < self.dropout_prob:
            return

        noisy_x = self.car.x + float(self.rng.normal(0.0, self.pos_noise_std))
        noisy_y = self.car.y + float(self.rng.normal(0.0, self.pos_noise_std))
        noisy_heading = _wrap_angle(self.car.theta + float(self.rng.normal(0.0, self.heading_noise_std)))

        self.buffer.push(now, {
            "x": noisy_x,
            "y": noisy_y,
            "theta": noisy_heading,
        })

    def read(self, now: Optional[float] = None) -> Optional[TimedMeasurement]:
        now = time.perf_counter() if now is None else now
        meas = self.buffer.pop_ready(now)
        return meas or self.buffer.latest


class WheelEncoderSensor:
    """Simulated wheel encoder giving longitudinal speed magnitude."""

    def __init__(
        self,
        car,
        sample_rate_hz: float = 50.0,
        speed_noise_std: float = 0.02,  # m/s (mildly noisy wheel speed)
        latency_s: float = 0.02,
        rng: Optional[np.random.Generator] = None,
    ):
        self.car = car
        self.sample_rate = float(sample_rate_hz)
        self.sample_interval = 1.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        self.speed_noise_std = speed_noise_std
        self.buffer = _LatencyBuffer(latency_s)
        self._accumulator = 0.0
        self.rng = rng or np.random.default_rng()

    def update(self, dt: float, now: float):
        if self.sample_interval == 0:
            return
        self._accumulator += dt
        if self._accumulator < self.sample_interval:
            return
        self._accumulator -= self.sample_interval

        noisy_speed = abs(self.car.velocity) + float(self.rng.normal(0.0, self.speed_noise_std))
        self.buffer.push(now, {"speed": max(0.0, noisy_speed)})

    def read(self, now: Optional[float] = None) -> Optional[TimedMeasurement]:
        now = time.perf_counter() if now is None else now
        meas = self.buffer.pop_ready(now)
        return meas or self.buffer.latest


class SensorSuite:
    """Convenience wrapper exposing a fused, noisy ego state estimate."""

    def __init__(self, car):
        rng = np.random.default_rng()
        self.imu = IMUSensor(car, rng=rng)
        self.gnss = GNSSSensor(car, rng=rng)
        self.wheel = WheelEncoderSensor(car, rng=rng)
        self._last_state: Optional[dict] = None
        self._filt_vel = 0.0
        self._filt_theta = car.theta
        self._vel_alpha = 0.2  # low-pass for speed
        self._theta_alpha = 0.2  # low-pass for heading

    def update(self, dt: float, now: Optional[float] = None):
        now = time.perf_counter() if now is None else now
        self.imu.update(dt, now)
        self.gnss.update(dt, now)
        self.wheel.update(dt, now)

    def get_state_estimate(self, now: Optional[float] = None) -> Optional[dict]:
        now = time.perf_counter() if now is None else now
        gnss_meas = self.gnss.read(now)
        wheel_meas = self.wheel.read(now)
        imu_meas = self.imu.read(now)

        if gnss_meas is None and wheel_meas is None and imu_meas is None:
            return self._last_state

        state = self._last_state.copy() if self._last_state else {}

        if gnss_meas is not None:
            state.update(gnss_meas.data)
            state["timestamp"] = gnss_meas.timestamp

        if wheel_meas is not None:
            state["velocity"] = wheel_meas.data.get("speed", state.get("velocity", 0.0))
            state["timestamp"] = max(state.get("timestamp", 0.0), wheel_meas.timestamp)

        if "velocity" not in state and imu_meas is not None:
            # Approximate speed from IMU lateral acceleration and yaw rate when wheel data missing
            yaw_rate = imu_meas.data.get("yaw_rate", 0.0)
            ax_body = imu_meas.data.get("ax_body", 0.0)
            ay_body = imu_meas.data.get("ay_body", 0.0)
            state["velocity"] = float(np.hypot(ax_body, ay_body) / max(abs(yaw_rate), 1e-3)) if abs(yaw_rate) > 1e-3 else 0.0

        # Always expose yaw rate when available
        if imu_meas is not None:
            state["yaw_rate"] = imu_meas.data.get("yaw_rate", state.get("yaw_rate", 0.0))
            state["timestamp"] = max(state.get("timestamp", 0.0), imu_meas.timestamp)

        # If heading missing, integrate IMU yaw rate over last known heading
        if "theta" not in state and imu_meas is not None:
            last_theta = self._last_state.get("theta", 0.0) if self._last_state else 0.0
            state["theta"] = _wrap_angle(last_theta + imu_meas.data.get("yaw_rate", 0.0) * self.imu.sample_interval)
            state["timestamp"] = max(state.get("timestamp", 0.0), imu_meas.timestamp)

        # Apply light low-pass filtering to stabilize control inputs
        raw_vel = state.get("velocity", 0.0)
        self._filt_vel = (1 - self._vel_alpha) * self._filt_vel + self._vel_alpha * raw_vel
        state["velocity"] = self._filt_vel

        raw_theta = state.get("theta", self._filt_theta)
        # Handle wrap-around smoothly
        dtheta = _wrap_angle(raw_theta - self._filt_theta)
        self._filt_theta = _wrap_angle(self._filt_theta + self._theta_alpha * dtheta)
        state["theta"] = self._filt_theta

        self._last_state = state
        return state