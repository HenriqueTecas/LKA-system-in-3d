#pragma once

#include <Eigen/Dense>
#include <vector>
#include <tuple>

namespace lka {

struct Point2D {
    double x;
    double y;

    Point2D() : x(0), y(0) {}
    Point2D(double x_, double y_) : x(x_), y(y_) {}
};

struct Point3D {
    double x;
    double y;
    double z;

    Point3D() : x(0), y(0), z(0) {}
    Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

struct LaneBoundary {
    std::vector<Point3D> points;
    std::vector<double> distances;
};

class LaneDetector {
public:
    LaneDetector(int image_width, int image_height, double camera_height,
                 double pitch_angle, double fov_horizontal);

    // Compute homography matrix from camera parameters
    void compute_homography(const Eigen::Vector3d& camera_pos,
                           const Eigen::Vector3d& camera_forward,
                           const Eigen::Vector3d& camera_right,
                           const Eigen::Vector3d& camera_up);

    // Detect lanes from track boundaries (parallelized)
    std::tuple<LaneBoundary, LaneBoundary> detect_lanes(
        const std::vector<Point3D>& left_boundary,
        const std::vector<Point3D>& right_boundary,
        const Eigen::Vector3d& camera_pos,
        const Eigen::Vector3d& camera_forward,
        const Eigen::Vector3d& camera_right,
        const Eigen::Vector3d& camera_up,
        double max_distance = 50.0,
        double sample_interval = 6.0);

    // Fast distance computation with SIMD
    std::vector<double> compute_distances_to_camera(
        const std::vector<Point3D>& points,
        const Eigen::Vector3d& camera_pos);

    // Project 3D point to 2D image coordinates
    bool project_point(const Point3D& world_point,
                      const Eigen::Vector3d& camera_pos,
                      Point2D& image_point) const;

private:
    int image_width_;
    int image_height_;
    double camera_height_;
    double pitch_angle_;
    double fov_horizontal_;
    double focal_length_;

    Eigen::Matrix3d homography_;
    Eigen::Matrix3d K_; // Camera intrinsic matrix

    // Interpolate boundary at regular intervals (parallelized)
    std::vector<Point3D> interpolate_boundary(
        const std::vector<Point3D>& boundary,
        double interval);

    // Check if point is in front of camera
    bool is_in_front_of_camera(const Point3D& point,
                               const Eigen::Vector3d& camera_pos,
                               const Eigen::Vector3d& camera_forward) const;
};

} // namespace lka
