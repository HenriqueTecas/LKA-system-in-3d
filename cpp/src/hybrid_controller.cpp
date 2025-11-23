#include "hybrid_controller.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

namespace lka {

HybridController::HybridController(const HybridConfig& config)
    : config_(config) {
}

Eigen::Vector3d HybridController::fit_polynomial(const std::vector<Point3D>& points) const {
    if (points.size() < 3) {
        return Eigen::Vector3d::Zero();
    }

    // Fit 2nd order polynomial: y = a*x^2 + b*x + c
    // Using normal equations: A^T*A*coeffs = A^T*b

    int n = points.size();
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd b(n);

    // Build design matrix and observation vector
    for (int i = 0; i < n; ++i) {
        double x = points[i].x;
        double y = points[i].y;
        A(i, 0) = x * x;
        A(i, 1) = x;
        A(i, 2) = 1.0;
        b(i) = y;
    }

    // Solve least squares: (A^T*A)*coeffs = A^T*b
    Eigen::Matrix3d ATA = A.transpose() * A;
    Eigen::Vector3d ATb = A.transpose() * b;

    // Use LDLT decomposition for speed (symmetric positive definite)
    Eigen::Vector3d coeffs = ATA.ldlt().solve(ATb);

    return coeffs;
}

double HybridController::eval_polynomial(const Eigen::Vector3d& coeffs, double x) const {
    return coeffs(0) * x * x + coeffs(1) * x + coeffs(2);
}

double HybridController::eval_polynomial_derivative(const Eigen::Vector3d& coeffs, double x) const {
    return 2.0 * coeffs(0) * x + coeffs(1);
}

double HybridController::compute_adaptive_lookahead(double curvature, double speed) const {
    // Adaptive lookahead based on curvature and speed
    double curvature_factor = config_.curvature_scale * std::abs(curvature);
    double lookahead = config_.base_lookahead - curvature_factor;

    // Clamp to min/max
    lookahead = math::clamp(lookahead, config_.min_lookahead, config_.max_lookahead);

    // Adjust for speed (higher speed = longer lookahead)
    lookahead *= (1.0 + 0.01 * speed);

    return lookahead;
}

double HybridController::smooth_steering(double raw_steering) {
    // Add to history
    steering_history_.push_back(raw_steering);

    // Keep only recent history
    if (steering_history_.size() > config_.smoothing_window) {
        steering_history_.pop_front();
    }

    // Compute rolling median
    std::vector<double> values(steering_history_.begin(), steering_history_.end());
    return math::median(values);
}

ControlMode HybridController::determine_mode(double lateral_error,
                                             double heading_error,
                                             double time_to_boundary,
                                             bool driver_override) const {
    // Manual mode if driver is actively steering
    if (driver_override) {
        return ControlMode::MANUAL;
    }

    // Check if warnings or intervention needed
    bool lateral_warning = std::abs(lateral_error) > config_.lateral_error_threshold;
    bool heading_warning = std::abs(heading_error) > config_.heading_error_threshold;
    bool time_warning = time_to_boundary < config_.time_to_boundary_threshold;

    // Assist mode if any critical condition
    if (time_warning || (lateral_warning && heading_warning)) {
        return ControlMode::ASSIST;
    }

    // Warning mode if approaching limits
    if (lateral_warning || heading_warning) {
        return ControlMode::WARNING;
    }

    // Default to manual
    return ControlMode::MANUAL;
}

double HybridController::compute_time_to_boundary(const VehicleState& vehicle_state,
                                                  const LaneBoundary& left_boundary,
                                                  const LaneBoundary& right_boundary) const {
    // Simple time-to-boundary estimation
    // Find closest boundary points
    double min_dist = std::numeric_limits<double>::max();

    for (const auto& pt : left_boundary.points) {
        double dist = std::abs(vehicle_state.y - pt.y);
        if (dist < min_dist) min_dist = dist;
    }

    for (const auto& pt : right_boundary.points) {
        double dist = std::abs(vehicle_state.y - pt.y);
        if (dist < min_dist) min_dist = dist;
    }

    // Estimate time assuming constant lateral velocity
    // This is simplified - real implementation would consider yaw rate
    double lateral_velocity = vehicle_state.speed * std::sin(vehicle_state.yaw);
    if (std::abs(lateral_velocity) < 0.1) {
        return std::numeric_limits<double>::max();
    }

    return min_dist / std::abs(lateral_velocity);
}

HybridController::ControlOutput HybridController::calculate_control(
    const VehicleState& vehicle_state,
    const LaneBoundary& left_boundary,
    const LaneBoundary& right_boundary,
    double driver_steering,
    bool driver_override) {

    ControlOutput output;

    // Fit polynomials to lane boundaries
    Eigen::Vector3d left_coeffs = fit_polynomial(left_boundary.points);
    Eigen::Vector3d right_coeffs = fit_polynomial(right_boundary.points);

    // Compute lane center polynomial
    Eigen::Vector3d center_coeffs = (left_coeffs + right_coeffs) / 2.0;

    // Compute current lateral error (y position vs lane center)
    double lane_center_y = eval_polynomial(center_coeffs, vehicle_state.x);
    output.lateral_error = vehicle_state.y - lane_center_y;

    // Compute lane heading and heading error
    double lane_heading = std::atan(eval_polynomial_derivative(center_coeffs, vehicle_state.x));
    output.heading_error = math::normalize_angle(vehicle_state.yaw - lane_heading);

    // Compute curvature (second derivative of polynomial)
    output.curvature = 2.0 * center_coeffs(0); // Second derivative of ax^2+bx+c is 2a

    // Compute time to boundary
    output.time_to_boundary = compute_time_to_boundary(vehicle_state, left_boundary, right_boundary);

    // Determine control mode
    output.mode = determine_mode(output.lateral_error, output.heading_error,
                                 output.time_to_boundary, driver_override);

    // Compute adaptive lookahead distance
    output.lookahead_distance = compute_adaptive_lookahead(output.curvature, vehicle_state.speed);

    // Find lookahead point on lane center
    double lookahead_x = vehicle_state.x + output.lookahead_distance * std::cos(vehicle_state.yaw);
    double lookahead_y = eval_polynomial(center_coeffs, lookahead_x);
    lookahead_point_ = Point2D(lookahead_x, lookahead_y);

    // Pure pursuit control law
    double alpha = std::atan2(lookahead_y - vehicle_state.y, lookahead_x - vehicle_state.x) - vehicle_state.yaw;
    double steering_angle = std::atan(2.0 * 2.7 * std::sin(alpha) / output.lookahead_distance);

    // Apply mode-specific blending
    double final_steering = 0.0;
    switch (output.mode) {
        case ControlMode::MANUAL:
            final_steering = driver_steering;
            break;

        case ControlMode::WARNING:
            // Gentle steering assist (30% LKA, 70% driver)
            final_steering = 0.3 * steering_angle + 0.7 * driver_steering;
            break;

        case ControlMode::ASSIST:
            // Strong steering assist (80% LKA, 20% driver)
            final_steering = 0.8 * steering_angle + 0.2 * driver_steering;
            break;
    }

    // Apply smoothing
    output.steering = smooth_steering(final_steering);

    return output;
}

} // namespace lka
