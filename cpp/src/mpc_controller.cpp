#include "mpc_controller.h"
#include "utils.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lka {

MPCController::MPCController(const MPCConfig& config)
    : config_(config) {
}

VehicleState MPCController::forward_simulate(const VehicleState& state,
                                             double steering_angle,
                                             double dt) const {
    VehicleState next = state;

    // Ackermann steering model (kinematic bicycle model)
    // Clamp steering angle
    double steering = math::clamp(steering_angle,
                                  -config_.max_steering_angle,
                                  config_.max_steering_angle);

    // Update yaw rate
    double yaw_rate = (state.speed / config_.wheelbase) * std::tan(steering);

    // Update position and yaw
    next.yaw = state.yaw + yaw_rate * dt;
    next.yaw = math::normalize_angle(next.yaw);

    next.x = state.x + state.speed * std::cos(next.yaw) * dt;
    next.y = state.y + state.speed * std::sin(next.yaw) * dt;

    // Speed remains constant in this simplified model
    next.speed = state.speed;

    return next;
}

void MPCController::get_lane_reference(double x, double y,
                                       const LaneBoundary& left_boundary,
                                       const LaneBoundary& right_boundary,
                                       double& lane_center_y,
                                       double& lane_heading) const {
    // Find closest points on both boundaries
    double min_dist_left = std::numeric_limits<double>::max();
    double min_dist_right = std::numeric_limits<double>::max();
    size_t idx_left = 0;
    size_t idx_right = 0;

    // Find closest left boundary point
    for (size_t i = 0; i < left_boundary.points.size(); ++i) {
        double dist = math::distance_squared(x, y,
                                             left_boundary.points[i].x,
                                             left_boundary.points[i].y);
        if (dist < min_dist_left) {
            min_dist_left = dist;
            idx_left = i;
        }
    }

    // Find closest right boundary point
    for (size_t i = 0; i < right_boundary.points.size(); ++i) {
        double dist = math::distance_squared(x, y,
                                             right_boundary.points[i].x,
                                             right_boundary.points[i].y);
        if (dist < min_dist_right) {
            min_dist_right = dist;
            idx_right = i;
        }
    }

    // Compute lane center
    if (left_boundary.points.empty() || right_boundary.points.empty()) {
        lane_center_y = y;
        lane_heading = 0;
        return;
    }

    const Point3D& left_pt = left_boundary.points[idx_left];
    const Point3D& right_pt = right_boundary.points[idx_right];

    lane_center_y = (left_pt.y + right_pt.y) / 2.0;

    // Estimate lane heading from adjacent points
    if (idx_left < left_boundary.points.size() - 1 && idx_right < right_boundary.points.size() - 1) {
        const Point3D& left_next = left_boundary.points[idx_left + 1];
        const Point3D& right_next = right_boundary.points[idx_right + 1];

        double heading_left = std::atan2(left_next.y - left_pt.y, left_next.x - left_pt.x);
        double heading_right = std::atan2(right_next.y - right_pt.y, right_next.x - right_pt.x);

        lane_heading = (heading_left + heading_right) / 2.0;
    } else {
        lane_heading = 0;
    }
}

double MPCController::distance_to_boundaries(double x, double y,
                                             const LaneBoundary& left_boundary,
                                             const LaneBoundary& right_boundary) const {
    double min_dist_left = std::numeric_limits<double>::max();
    double min_dist_right = std::numeric_limits<double>::max();

    // Find minimum distance to left boundary
    for (const auto& pt : left_boundary.points) {
        double dist = std::abs(y - pt.y); // Simplified lateral distance
        if (dist < min_dist_left) {
            min_dist_left = dist;
        }
    }

    // Find minimum distance to right boundary
    for (const auto& pt : right_boundary.points) {
        double dist = std::abs(y - pt.y); // Simplified lateral distance
        if (dist < min_dist_right) {
            min_dist_right = dist;
        }
    }

    return std::min(min_dist_left, min_dist_right);
}

double MPCController::compute_trajectory_cost(const VehicleState& initial_state,
                                              double steering_angle,
                                              const LaneBoundary& left_boundary,
                                              const LaneBoundary& right_boundary,
                                              double current_steering) const {
    double total_cost = 0.0;
    VehicleState state = initial_state;

    // Simulate forward for prediction horizon
    for (int step = 0; step < config_.prediction_horizon; ++step) {
        // Forward simulate one step
        state = forward_simulate(state, steering_angle, config_.dt);

        // Get lane reference at this position
        double lane_center_y, lane_heading;
        get_lane_reference(state.x, state.y, left_boundary, right_boundary,
                          lane_center_y, lane_heading);

        // Lane center tracking cost
        double lateral_error = state.y - lane_center_y;
        total_cost += config_.weight_lane_center * lateral_error * lateral_error;

        // Heading alignment cost
        double heading_error = math::normalize_angle(state.yaw - lane_heading);
        total_cost += config_.weight_heading * heading_error * heading_error;

        // Boundary proximity penalty (higher cost near boundaries)
        double boundary_dist = distance_to_boundaries(state.x, state.y,
                                                      left_boundary, right_boundary);
        if (boundary_dist < 1.0) {
            total_cost += 100.0 * (1.0 - boundary_dist) * (1.0 - boundary_dist);
        }
    }

    // Steering effort cost
    total_cost += config_.weight_steering_effort * steering_angle * steering_angle;

    // Steering smoothness cost (penalize large changes from current steering)
    double steering_change = steering_angle - current_steering;
    total_cost += config_.weight_steering_smoothness * steering_change * steering_change;

    return total_cost;
}

double MPCController::calculate_steering(const VehicleState& state,
                                         const LaneBoundary& left_boundary,
                                         const LaneBoundary& right_boundary,
                                         double current_steering) {
    // Generate steering candidates
    std::vector<double> candidates(config_.num_candidates);
    for (int i = 0; i < config_.num_candidates; ++i) {
        double t = static_cast<double>(i) / (config_.num_candidates - 1);
        candidates[i] = -config_.max_steering_angle +
                       t * 2.0 * config_.max_steering_angle;
    }

    // Evaluate all candidates in parallel
    std::vector<double> costs(config_.num_candidates);
    std::vector<std::vector<Point2D>> trajectories(config_.num_candidates);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < config_.num_candidates; ++i) {
        costs[i] = compute_trajectory_cost(state, candidates[i],
                                           left_boundary, right_boundary,
                                           current_steering);

        // Store trajectory for visualization
        VehicleState sim_state = state;
        trajectories[i].reserve(config_.prediction_horizon);
        for (int step = 0; step < config_.prediction_horizon; ++step) {
            sim_state = forward_simulate(sim_state, candidates[i], config_.dt);
            trajectories[i].emplace_back(sim_state.x, sim_state.y);
        }
    }

    // Find best candidate
    int best_idx = 0;
    double best_cost = costs[0];
    for (int i = 1; i < config_.num_candidates; ++i) {
        if (costs[i] < best_cost) {
            best_cost = costs[i];
            best_idx = i;
        }
    }

    // Store predicted trajectory for visualization
    predicted_trajectory_ = trajectories[best_idx];

    return candidates[best_idx];
}

} // namespace lka
