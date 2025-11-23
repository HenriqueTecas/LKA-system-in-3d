#pragma once

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include "lane_detection.h"
#include "types.h"

namespace lka {

enum class ControlMode {
    MANUAL = 0,
    WARNING = 1,
    ASSIST = 2
};

struct HybridConfig {
    double base_lookahead;
    double min_lookahead;
    double max_lookahead;
    double curvature_scale;
    double lateral_error_threshold;
    double heading_error_threshold;
    double time_to_boundary_threshold;
    int smoothing_window;

    HybridConfig()
        : base_lookahead(15.0),
          min_lookahead(8.0),
          max_lookahead(25.0),
          curvature_scale(10.0),
          lateral_error_threshold(0.5),
          heading_error_threshold(0.15),
          time_to_boundary_threshold(2.0),
          smoothing_window(3) {}
};

class HybridController {
public:
    explicit HybridController(const HybridConfig& config = HybridConfig());

    // Calculate control output with mode selection
    struct ControlOutput {
        double steering;
        ControlMode mode;
        double lateral_error;
        double heading_error;
        double time_to_boundary;
        double curvature;
        double lookahead_distance;
    };

    ControlOutput calculate_control(const VehicleState& vehicle_state,
                                    const LaneBoundary& left_boundary,
                                    const LaneBoundary& right_boundary,
                                    double driver_steering,
                                    bool driver_override);

    // Fast polynomial fitting with Eigen (2nd order)
    Eigen::Vector3d fit_polynomial(const std::vector<Point3D>& points) const;

    // Evaluate polynomial and its derivative
    double eval_polynomial(const Eigen::Vector3d& coeffs, double x) const;
    double eval_polynomial_derivative(const Eigen::Vector3d& coeffs, double x) const;

    // Get visualization data
    Point2D get_lookahead_point() const { return lookahead_point_; }

private:
    HybridConfig config_;
    Point2D lookahead_point_;
    std::deque<double> steering_history_;

    // Determine control mode based on errors
    ControlMode determine_mode(double lateral_error,
                               double heading_error,
                               double time_to_boundary,
                               bool driver_override) const;

    // Compute adaptive lookahead based on curvature and speed
    double compute_adaptive_lookahead(double curvature, double speed) const;

    // Smooth steering with rolling median
    double smooth_steering(double raw_steering);

    // Compute time to boundary crossing
    double compute_time_to_boundary(const VehicleState& vehicle_state,
                                    const LaneBoundary& left_boundary,
                                    const LaneBoundary& right_boundary) const;
};

} // namespace lka
