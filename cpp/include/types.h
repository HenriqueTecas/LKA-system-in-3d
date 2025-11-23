#pragma once

namespace lka {

struct VehicleState {
    double x;
    double y;
    double yaw;
    double speed;

    VehicleState() : x(0), y(0), yaw(0), speed(0) {}
    VehicleState(double x_, double y_, double yaw_, double speed_)
        : x(x_), y(y_), yaw(yaw_), speed(speed_) {}
};

} // namespace lka
