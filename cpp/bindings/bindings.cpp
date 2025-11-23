#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include "lane_detection.h"
#include "mpc_controller.h"
#include "hybrid_controller.h"
#include "physics.h"
#include "utils.h"

namespace py = pybind11;
using namespace lka;

PYBIND11_MODULE(lka_cpp_accelerators, m) {
    m.doc() = "High-performance C++ accelerators for LKA system";

    // ========== Data structures ==========

    py::class_<Point2D>(m, "Point2D")
        .def(py::init<>())
        .def(py::init<double, double>())
        .def_readwrite("x", &Point2D::x)
        .def_readwrite("y", &Point2D::y)
        .def("__repr__", [](const Point2D& p) {
            return "Point2D(x=" + std::to_string(p.x) + ", y=" + std::to_string(p.y) + ")";
        });

    py::class_<Point3D>(m, "Point3D")
        .def(py::init<>())
        .def(py::init<double, double, double>())
        .def_readwrite("x", &Point3D::x)
        .def_readwrite("y", &Point3D::y)
        .def_readwrite("z", &Point3D::z)
        .def("__repr__", [](const Point3D& p) {
            return "Point3D(x=" + std::to_string(p.x) +
                   ", y=" + std::to_string(p.y) +
                   ", z=" + std::to_string(p.z) + ")";
        });

    py::class_<LaneBoundary>(m, "LaneBoundary")
        .def(py::init<>())
        .def_readwrite("points", &LaneBoundary::points)
        .def_readwrite("distances", &LaneBoundary::distances);

    // ========== Lane Detection ==========

    py::class_<LaneDetector>(m, "LaneDetector")
        .def(py::init<int, int, double, double, double>(),
             py::arg("image_width"),
             py::arg("image_height"),
             py::arg("camera_height"),
             py::arg("pitch_angle"),
             py::arg("fov_horizontal"))
        .def("detect_lanes", &LaneDetector::detect_lanes,
             py::arg("left_boundary"),
             py::arg("right_boundary"),
             py::arg("camera_pos"),
             py::arg("camera_forward"),
             py::arg("camera_right"),
             py::arg("camera_up"),
             py::arg("max_distance") = 50.0,
             py::arg("sample_interval") = 6.0,
             "Detect visible lane boundaries from camera view (parallelized)")
        .def("compute_distances_to_camera", &LaneDetector::compute_distances_to_camera,
             py::arg("points"),
             py::arg("camera_pos"),
             "Fast distance computation with SIMD")
        .def("project_point", &LaneDetector::project_point,
             py::arg("world_point"),
             py::arg("camera_pos"),
             py::arg("image_point"),
             "Project 3D point to 2D image coordinates");

    // ========== MPC Controller ==========

    py::class_<MPCConfig>(m, "MPCConfig")
        .def(py::init<>())
        .def_readwrite("prediction_horizon", &MPCConfig::prediction_horizon)
        .def_readwrite("num_candidates", &MPCConfig::num_candidates)
        .def_readwrite("dt", &MPCConfig::dt)
        .def_readwrite("wheelbase", &MPCConfig::wheelbase)
        .def_readwrite("max_steering_angle", &MPCConfig::max_steering_angle)
        .def_readwrite("weight_lane_center", &MPCConfig::weight_lane_center)
        .def_readwrite("weight_heading", &MPCConfig::weight_heading)
        .def_readwrite("weight_steering_effort", &MPCConfig::weight_steering_effort)
        .def_readwrite("weight_steering_smoothness", &MPCConfig::weight_steering_smoothness);

    py::class_<VehicleState>(m, "VehicleState")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(),
             py::arg("x"), py::arg("y"), py::arg("yaw"), py::arg("speed"))
        .def_readwrite("x", &VehicleState::x)
        .def_readwrite("y", &VehicleState::y)
        .def_readwrite("yaw", &VehicleState::yaw)
        .def_readwrite("speed", &VehicleState::speed);

    py::class_<MPCController>(m, "MPCController")
        .def(py::init<const MPCConfig&>(), py::arg("config") = MPCConfig())
        .def("calculate_steering", &MPCController::calculate_steering,
             py::arg("state"),
             py::arg("left_boundary"),
             py::arg("right_boundary"),
             py::arg("current_steering"),
             "Calculate optimal steering angle (parallelized)")
        .def("get_predicted_trajectory", &MPCController::get_predicted_trajectory,
             "Get predicted trajectory for visualization")
        .def("set_config", &MPCController::set_config, py::arg("config"))
        .def("get_config", &MPCController::get_config);

    // ========== Hybrid Controller ==========

    py::enum_<ControlMode>(m, "ControlMode")
        .value("MANUAL", ControlMode::MANUAL)
        .value("WARNING", ControlMode::WARNING)
        .value("ASSIST", ControlMode::ASSIST)
        .export_values();

    py::class_<HybridConfig>(m, "HybridConfig")
        .def(py::init<>())
        .def_readwrite("base_lookahead", &HybridConfig::base_lookahead)
        .def_readwrite("min_lookahead", &HybridConfig::min_lookahead)
        .def_readwrite("max_lookahead", &HybridConfig::max_lookahead)
        .def_readwrite("curvature_scale", &HybridConfig::curvature_scale)
        .def_readwrite("lateral_error_threshold", &HybridConfig::lateral_error_threshold)
        .def_readwrite("heading_error_threshold", &HybridConfig::heading_error_threshold)
        .def_readwrite("time_to_boundary_threshold", &HybridConfig::time_to_boundary_threshold)
        .def_readwrite("smoothing_window", &HybridConfig::smoothing_window);

    py::class_<HybridController::ControlOutput>(m, "ControlOutput")
        .def(py::init<>())
        .def_readwrite("steering", &HybridController::ControlOutput::steering)
        .def_readwrite("mode", &HybridController::ControlOutput::mode)
        .def_readwrite("lateral_error", &HybridController::ControlOutput::lateral_error)
        .def_readwrite("heading_error", &HybridController::ControlOutput::heading_error)
        .def_readwrite("time_to_boundary", &HybridController::ControlOutput::time_to_boundary)
        .def_readwrite("curvature", &HybridController::ControlOutput::curvature)
        .def_readwrite("lookahead_distance", &HybridController::ControlOutput::lookahead_distance);

    py::class_<HybridController>(m, "HybridController")
        .def(py::init<const HybridConfig&>(), py::arg("config") = HybridConfig())
        .def("calculate_control", &HybridController::calculate_control,
             py::arg("vehicle_state"),
             py::arg("left_boundary"),
             py::arg("right_boundary"),
             py::arg("driver_steering"),
             py::arg("driver_override"),
             "Calculate control output with mode selection")
        .def("fit_polynomial", &HybridController::fit_polynomial,
             py::arg("points"),
             "Fast polynomial fitting with Eigen (2nd order)")
        .def("get_lookahead_point", &HybridController::get_lookahead_point,
             "Get current lookahead point for visualization");

    // ========== Vehicle Physics ==========

    py::class_<WheelState>(m, "WheelState")
        .def(py::init<>())
        .def_readwrite("angular_velocity", &WheelState::angular_velocity)
        .def_readwrite("torque", &WheelState::torque)
        .def_readwrite("slip_ratio", &WheelState::slip_ratio);

    py::class_<VehiclePhysicsState>(m, "VehiclePhysicsState")
        .def(py::init<>())
        .def_readwrite("position", &VehiclePhysicsState::position)
        .def_readwrite("velocity", &VehiclePhysicsState::velocity)
        .def_readwrite("yaw", &VehiclePhysicsState::yaw)
        .def_readwrite("yaw_rate", &VehiclePhysicsState::yaw_rate)
        .def_readwrite("wheels", &VehiclePhysicsState::wheels);

    py::class_<VehicleParams>(m, "VehicleParams")
        .def(py::init<>())
        .def_readwrite("mass", &VehicleParams::mass)
        .def_readwrite("wheelbase", &VehicleParams::wheelbase)
        .def_readwrite("track_width", &VehicleParams::track_width)
        .def_readwrite("moment_of_inertia", &VehicleParams::moment_of_inertia)
        .def_readwrite("wheel_radius", &VehicleParams::wheel_radius)
        .def_readwrite("max_steering_angle", &VehicleParams::max_steering_angle)
        .def_readwrite("engine_max_torque", &VehicleParams::engine_max_torque)
        .def_readwrite("brake_max_torque", &VehicleParams::brake_max_torque)
        .def_readwrite("tire_friction", &VehicleParams::tire_friction)
        .def_readwrite("rolling_resistance", &VehicleParams::rolling_resistance)
        .def_readwrite("drag_coefficient", &VehicleParams::drag_coefficient)
        .def_readwrite("frontal_area", &VehicleParams::frontal_area);

    py::class_<VehiclePhysics>(m, "VehiclePhysics")
        .def(py::init<const VehicleParams&>(), py::arg("params") = VehicleParams())
        .def("update", &VehiclePhysics::update,
             py::arg("state"),
             py::arg("steering_angle"),
             py::arg("throttle"),
             py::arg("brake"),
             py::arg("dt"),
             "Update physics simulation with SIMD optimizations")
        .def("get_speed", &VehiclePhysics::get_speed,
             py::arg("state"),
             "Get vehicle speed from state")
        .def("compute_ackermann_angles", &VehiclePhysics::compute_ackermann_angles,
             py::arg("steering_input"),
             "Compute Ackermann steering geometry");

    // ========== Utility Functions ==========

    // Math utilities
    m.def("normalize_angle", &math::normalize_angle, py::arg("angle"),
          "Normalize angle to [-π, π]");
    m.def("distance", py::overload_cast<double, double, double, double>(&math::distance),
          py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
          "Fast distance computation");
    m.def("distance_squared", &math::distance_squared,
          py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"),
          "Fast squared distance (avoids sqrt)");
    m.def("clamp", &math::clamp<double>,
          py::arg("value"), py::arg("min_val"), py::arg("max_val"),
          "Clamp value to range");
    m.def("lerp", &math::lerp<double>,
          py::arg("a"), py::arg("b"), py::arg("t"),
          "Linear interpolation");
    m.def("median", &math::median, py::arg("values"),
          "Compute median of values");

    // Geometry utilities
    m.def("point_to_line_distance", &geometry::point_to_line_distance,
          py::arg("px"), py::arg("py"),
          py::arg("x1"), py::arg("y1"),
          py::arg("x2"), py::arg("y2"),
          "Distance from point to line segment");
    m.def("compute_curvature", &geometry::compute_curvature,
          py::arg("x1"), py::arg("y1"),
          py::arg("x2"), py::arg("y2"),
          py::arg("x3"), py::arg("y3"),
          "Compute curvature from 3 points");

    // Performance timer
    py::class_<perf::Timer>(m, "Timer")
        .def(py::init<>())
        .def("start", &perf::Timer::start, "Start/restart timer")
        .def("elapsed_ms", &perf::Timer::elapsed_ms, "Get elapsed time in milliseconds")
        .def("elapsed_us", &perf::Timer::elapsed_us, "Get elapsed time in microseconds");
}
