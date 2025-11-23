#include "physics.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

namespace lka {

VehiclePhysics::VehiclePhysics(const VehicleParams& params)
    : params_(params) {
}

double VehiclePhysics::get_speed(const VehiclePhysicsState& state) const {
    return state.velocity.norm();
}

std::pair<double, double> VehiclePhysics::compute_ackermann_angles(double steering_input) const {
    // Ackermann steering geometry
    // Inner wheel has larger angle than outer wheel

    if (std::abs(steering_input) < 1e-6) {
        return {0.0, 0.0};
    }

    double turn_radius = params_.wheelbase / std::tan(steering_input);

    // Inner wheel (larger angle)
    double inner_radius = turn_radius - params_.track_width / 2.0;
    double angle_inner = std::atan(params_.wheelbase / inner_radius);

    // Outer wheel (smaller angle)
    double outer_radius = turn_radius + params_.track_width / 2.0;
    double angle_outer = std::atan(params_.wheelbase / outer_radius);

    return {angle_inner, angle_outer};
}

Eigen::Vector2d VehiclePhysics::compute_tire_force(const WheelState& wheel,
                                                   double normal_force,
                                                   double slip_angle) const {
    // Simplified Pacejka tire model
    // F = D * sin(C * arctan(B * slip_angle))

    constexpr double B = 10.0;  // Stiffness factor
    constexpr double C = 1.9;   // Shape factor
    double D = params_.tire_friction * normal_force;  // Peak force

    double lateral_force = D * std::sin(C * std::atan(B * slip_angle));

    // Longitudinal force from wheel torque
    double longitudinal_force = wheel.torque / params_.wheel_radius;

    // Combine forces (simplified)
    double total_force = std::sqrt(lateral_force * lateral_force +
                                   longitudinal_force * longitudinal_force);

    // Limit by friction circle
    double max_force = params_.tire_friction * normal_force;
    if (total_force > max_force) {
        double scale = max_force / total_force;
        lateral_force *= scale;
        longitudinal_force *= scale;
    }

    return Eigen::Vector2d(longitudinal_force, lateral_force);
}

Eigen::Vector2d VehiclePhysics::compute_drag_force(const Eigen::Vector2d& velocity) const {
    // Aerodynamic drag: F = 0.5 * rho * Cd * A * v^2
    constexpr double air_density = 1.225; // kg/m³

    double speed = velocity.norm();
    if (speed < 0.1) return Eigen::Vector2d::Zero();

    double drag_magnitude = 0.5 * air_density * params_.drag_coefficient *
                           params_.frontal_area * speed * speed;

    // Drag opposes velocity
    Eigen::Vector2d drag_direction = -velocity.normalized();
    return drag_magnitude * drag_direction;
}

Eigen::Vector2d VehiclePhysics::compute_rolling_resistance(const VehiclePhysicsState& state) const {
    // Rolling resistance: F = Crr * m * g
    constexpr double gravity = 9.81; // m/s²

    double resistance_magnitude = params_.rolling_resistance * params_.mass * gravity;

    // Resistance opposes velocity
    if (state.velocity.norm() < 0.1) return Eigen::Vector2d::Zero();

    Eigen::Vector2d resistance_direction = -state.velocity.normalized();
    return resistance_magnitude * resistance_direction;
}

void VehiclePhysics::update_wheels(VehiclePhysicsState& state,
                                  double throttle,
                                  double brake,
                                  double dt) {
    // Distribute torque to wheels
    double drive_torque = throttle * params_.engine_max_torque / 2.0; // Rear wheel drive
    double brake_torque = brake * params_.brake_max_torque / 4.0;     // All wheels

    // Update rear wheels (driven)
    for (int i = 2; i < 4; ++i) {
        WheelState& wheel = state.wheels[i];
        wheel.torque = drive_torque - brake_torque;

        // Update angular velocity
        double angular_accel = wheel.torque / (0.5 * params_.wheel_radius * params_.wheel_radius);
        wheel.angular_velocity += angular_accel * dt;

        // Compute slip ratio
        double wheel_speed = wheel.angular_velocity * params_.wheel_radius;
        double vehicle_speed = state.velocity.norm();
        if (vehicle_speed > 0.1) {
            wheel.slip_ratio = (wheel_speed - vehicle_speed) / vehicle_speed;
        } else {
            wheel.slip_ratio = 0.0;
        }
    }

    // Update front wheels (non-driven, just braking)
    for (int i = 0; i < 2; ++i) {
        WheelState& wheel = state.wheels[i];
        wheel.torque = -brake_torque;

        double angular_accel = wheel.torque / (0.5 * params_.wheel_radius * params_.wheel_radius);
        wheel.angular_velocity += angular_accel * dt;
        wheel.slip_ratio = 0.0; // Simplified
    }
}

void VehiclePhysics::update(VehiclePhysicsState& state,
                           double steering_angle,
                           double throttle,
                           double brake,
                           double dt) {
    // Clamp inputs
    steering_angle = math::clamp(steering_angle, -params_.max_steering_angle, params_.max_steering_angle);
    throttle = math::clamp(throttle, 0.0, 1.0);
    brake = math::clamp(brake, 0.0, 1.0);

    // Update wheels
    update_wheels(state, throttle, brake, dt);

    // Compute Ackermann angles
    auto [angle_inner, angle_outer] = compute_ackermann_angles(steering_angle);

    // Simplified force computation (bicycle model)
    double speed = state.velocity.norm();

    // Compute lateral force from steering
    double lateral_force = 0.0;
    if (speed > 0.1) {
        // Simplified: lateral force proportional to steering and speed
        lateral_force = params_.mass * speed * speed * std::tan(steering_angle) / params_.wheelbase;
    }

    // Compute longitudinal force from wheels
    double longitudinal_force = 0.0;
    for (int i = 2; i < 4; ++i) { // Rear wheels
        longitudinal_force += state.wheels[i].torque / params_.wheel_radius;
    }

    // Add drag and rolling resistance
    Eigen::Vector2d drag = compute_drag_force(state.velocity);
    Eigen::Vector2d rolling_resist = compute_rolling_resistance(state);

    // Total force in vehicle frame
    Eigen::Vector2d force_vehicle(longitudinal_force, lateral_force);

    // Transform to world frame
    double cos_yaw = std::cos(state.yaw);
    double sin_yaw = std::sin(state.yaw);
    Eigen::Matrix2d rotation;
    rotation << cos_yaw, -sin_yaw,
                sin_yaw, cos_yaw;

    Eigen::Vector2d force_world = rotation * force_vehicle + drag + rolling_resist;

    // Integrate acceleration
    Eigen::Vector2d acceleration = force_world / params_.mass;
    state.velocity += acceleration * dt;

    // Integrate position
    state.position += state.velocity * dt;

    // Update yaw rate and yaw (from bicycle model)
    if (speed > 0.1) {
        state.yaw_rate = speed * std::tan(steering_angle) / params_.wheelbase;
        state.yaw += state.yaw_rate * dt;
        state.yaw = math::normalize_angle(state.yaw);
    }
}

} // namespace lka
