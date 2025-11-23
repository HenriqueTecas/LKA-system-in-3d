#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace lka {
namespace math {

double median(std::vector<double>& values) {
    if (values.empty()) return 0.0;

    size_t n = values.size();
    size_t mid = n / 2;

    // Partial sort to find median
    std::nth_element(values.begin(), values.begin() + mid, values.end());

    if (n % 2 == 1) {
        return values[mid];
    } else {
        // Even number of elements: average of two middle values
        double mid1 = values[mid];
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        double mid2 = values[mid - 1];
        return (mid1 + mid2) / 2.0;
    }
}

} // namespace math

namespace geometry {

double point_to_line_distance(double px, double py,
                              double x1, double y1,
                              double x2, double y2) {
    // Distance from point (px, py) to line segment (x1, y1) - (x2, y2)

    double dx = x2 - x1;
    double dy = y2 - y1;
    double len_sq = dx * dx + dy * dy;

    if (len_sq < 1e-10) {
        // Degenerate line segment
        return math::distance(px, py, x1, y1);
    }

    // Project point onto line
    double t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    t = math::clamp(t, 0.0, 1.0); // Clamp to segment

    double proj_x = x1 + t * dx;
    double proj_y = y1 + t * dy;

    return math::distance(px, py, proj_x, proj_y);
}

std::pair<double, double> project_point_to_segment(
    double px, double py,
    double x1, double y1,
    double x2, double y2) {

    double dx = x2 - x1;
    double dy = y2 - y1;
    double len_sq = dx * dx + dy * dy;

    if (len_sq < 1e-10) {
        return {x1, y1};
    }

    double t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    t = math::clamp(t, 0.0, 1.0);

    return {x1 + t * dx, y1 + t * dy};
}

double compute_curvature(double x1, double y1,
                        double x2, double y2,
                        double x3, double y3) {
    // Compute curvature from 3 points using Menger curvature
    // K = 4*Area / (a*b*c) where Area is triangle area, a,b,c are side lengths

    // Compute side lengths
    double a = math::distance(x1, y1, x2, y2);
    double b = math::distance(x2, y2, x3, y3);
    double c = math::distance(x3, y3, x1, y1);

    if (a < 1e-6 || b < 1e-6 || c < 1e-6) {
        return 0.0;
    }

    // Compute triangle area using cross product
    double area = 0.5 * std::abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));

    // Compute curvature
    double curvature = 4.0 * area / (a * b * c);

    return curvature;
}

ClosestPointResult find_closest_point(
    double px, double py,
    const std::vector<double>& polyline_x,
    const std::vector<double>& polyline_y) {

    ClosestPointResult result;
    result.distance = std::numeric_limits<double>::max();
    result.segment_index = -1;

    if (polyline_x.size() != polyline_y.size() || polyline_x.size() < 2) {
        return result;
    }

    for (size_t i = 0; i < polyline_x.size() - 1; ++i) {
        double x1 = polyline_x[i];
        double y1 = polyline_y[i];
        double x2 = polyline_x[i + 1];
        double y2 = polyline_y[i + 1];

        auto [proj_x, proj_y] = project_point_to_segment(px, py, x1, y1, x2, y2);
        double dist = math::distance(px, py, proj_x, proj_y);

        if (dist < result.distance) {
            result.distance = dist;
            result.x = proj_x;
            result.y = proj_y;
            result.segment_index = i;
        }
    }

    return result;
}

} // namespace geometry

namespace perf {

Timer::Timer() : start_time_(0) {
    start();
}

void Timer::start() {
    auto now = std::chrono::high_resolution_clock::now();
    start_time_ = std::chrono::time_point_cast<std::chrono::microseconds>(now)
                      .time_since_epoch()
                      .count();
}

double Timer::elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    long long now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now)
                           .time_since_epoch()
                           .count();
    return (now_us - start_time_) / 1000.0;
}

double Timer::elapsed_us() const {
    auto now = std::chrono::high_resolution_clock::now();
    long long now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now)
                           .time_since_epoch()
                           .count();
    return static_cast<double>(now_us - start_time_);
}

} // namespace perf

} // namespace lka
