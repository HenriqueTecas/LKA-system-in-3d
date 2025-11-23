#pragma once

#include <Eigen/Dense>
#include <array>

namespace lka {

struct WheelState {
    double angular_velocity;  // rad/s
    double torque;            // N⋅m
    double slip_ratio;

    WheelState() : angular_velocity(0), torque(0), slip_ratio(0) {}
};

struct VehiclePhysicsState {
    Eigen::Vector2d position;
    Eigen::Vector2d velocity;
    double yaw;
    double yaw_rate;
    std::array<WheelState, 4> wheels; // FL, FR, RL, RR

    VehiclePhysicsState()
        : position(Eigen::Vector2d::Zero()),
          velocity(Eigen::Vector2d::Zero()),
          yaw(0), yaw_rate(0) {}
};

struct VehicleParams {
    double mass;              // kg
    double wheelbase;         // m
    double track_width;       // m
    double moment_of_inertia; // kg⋅m²
    double wheel_radius;      // m
    double max_steering_angle;// rad
    double engine_max_torque; // N⋅m
    double brake_max_torque;  // N⋅m
    double tire_friction;     // coefficient
    double rolling_resistance;// coefficient
    double drag_coefficient;  // Cd
    double frontal_area;      // m²

    VehicleParams()
        : mass(1500.0),
          wheelbase(2.7),
          track_width(1.5),
          moment_of_inertia(2500.0),
          wheel_radius(0.34),
          max_steering_angle(0.52),
          engine_max_torque(250.0),
          brake_max_torque(3000.0),
          tire_friction(0.9),
          rolling_resistance(0.015),
          drag_coefficient(0.3),
          frontal_area(2.2) {}
};

class VehiclePhysics {
public:
    explicit VehiclePhysics(const VehicleParams& params = VehicleParams());

    // Update physics simulation with SIMD optimizations
    void update(VehiclePhysicsState& state,
                double steering_angle,
                double throttle,
                double brake,
                double dt);

    // Compute tire forces (Pacejka magic formula simplified)
    Eigen::Vector2d compute_tire_force(const WheelState& wheel,
                                       double normal_force,
                                       double slip_angle) const;

    // Ackermann steering geometry
    std::pair<double, double> compute_ackermann_angles(double steering_input) const;

    // Get vehicle speed
    double get_speed(const VehiclePhysicsState& state) const;

private:
    VehicleParams params_;

    // Update wheel dynamics
    void update_wheels(VehiclePhysicsState& state,
                      double throttle,
                      double brake,
                      double dt);

    // Compute aerodynamic drag
    Eigen::Vector2d compute_drag_force(const Eigen::Vector2d& velocity) const;

    // Compute rolling resistance
    Eigen::Vector2d compute_rolling_resistance(const VehiclePhysicsState& state) const;
};

} // namespace lka
