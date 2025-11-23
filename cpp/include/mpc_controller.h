#pragma once

#include <Eigen/Dense>
#include <vector>
#include "lane_detection.h"
#include "types.h"

namespace lka {

struct MPCConfig {
    int prediction_horizon;
    int num_candidates;
    double dt;
    double wheelbase;
    double max_steering_angle;
    double weight_lane_center;
    double weight_heading;
    double weight_steering_effort;
    double weight_steering_smoothness;

    MPCConfig()
        : prediction_horizon(15),
          num_candidates(7),
          dt(0.1),
          wheelbase(2.7),
          max_steering_angle(0.52),
          weight_lane_center(10.0),
          weight_heading(5.0),
          weight_steering_effort(0.5),
          weight_steering_smoothness(2.0) {}
};

class MPCController {
public:
    explicit MPCController(const MPCConfig& config = MPCConfig());

    // Calculate optimal steering (parallelized candidate evaluation)
    double calculate_steering(const VehicleState& state,
                             const LaneBoundary& left_boundary,
                             const LaneBoundary& right_boundary,
                             double current_steering);

    // Get predicted trajectory for visualization
    std::vector<Point2D> get_predicted_trajectory() const {
        return predicted_trajectory_;
    }

    // Update configuration
    void set_config(const MPCConfig& config) { config_ = config; }
    const MPCConfig& get_config() const { return config_; }

private:
    MPCConfig config_;
    std::vector<Point2D> predicted_trajectory_;

    // Forward simulate vehicle dynamics (Ackermann model)
    VehicleState forward_simulate(const VehicleState& state,
                                  double steering_angle,
                                  double dt) const;

    // Compute cost for a trajectory (parallelized)
    double compute_trajectory_cost(const VehicleState& initial_state,
                                   double steering_angle,
                                   const LaneBoundary& left_boundary,
                                   const LaneBoundary& right_boundary,
                                   double current_steering) const;

    // Fast interpolation for lane center and heading
    void get_lane_reference(double x, double y,
                           const LaneBoundary& left_boundary,
                           const LaneBoundary& right_boundary,
                           double& lane_center_y,
                           double& lane_heading) const;

    // Distance from point to lane boundaries
    double distance_to_boundaries(double x, double y,
                                  const LaneBoundary& left_boundary,
                                  const LaneBoundary& right_boundary) const;
};

} // namespace lka
