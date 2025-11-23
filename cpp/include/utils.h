#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace lka {

// Fast math utilities with SIMD hints
namespace math {

// Normalize angle to [-π, π]
inline double normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

// Fast distance computation (avoid sqrt when possible)
inline double distance_squared(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return dx * dx + dy * dy;
}

inline double distance(double x1, double y1, double x2, double y2) {
    return std::sqrt(distance_squared(x1, y1, x2, y2));
}

// Vector magnitude
inline double magnitude(const Eigen::Vector2d& v) {
    return v.norm();
}

inline double magnitude(const Eigen::Vector3d& v) {
    return v.norm();
}

// Clamp value
template<typename T>
inline T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(value, max_val));
}

// Linear interpolation
template<typename T>
inline T lerp(T a, T b, double t) {
    return a + t * (b - a);
}

// Compute median (for rolling median filter)
double median(std::vector<double>& values);

} // namespace math

// Geometry utilities
namespace geometry {

// Point-to-line distance (2D)
double point_to_line_distance(double px, double py,
                              double x1, double y1,
                              double x2, double y2);

// Project point onto line segment
std::pair<double, double> project_point_to_segment(
    double px, double py,
    double x1, double y1,
    double x2, double y2);

// Compute curvature from 3 points
double compute_curvature(double x1, double y1,
                        double x2, double y2,
                        double x3, double y3);

// Find closest point on polyline
struct ClosestPointResult {
    double x;
    double y;
    double distance;
    int segment_index;
};

ClosestPointResult find_closest_point(
    double px, double py,
    const std::vector<double>& polyline_x,
    const std::vector<double>& polyline_y);

} // namespace geometry

// Performance utilities
namespace perf {

// Simple timer for benchmarking
class Timer {
public:
    Timer();
    void start();
    double elapsed_ms() const;
    double elapsed_us() const;

private:
    long long start_time_;
};

} // namespace perf

} // namespace lka
