#include "lane_detection.h"
#include "utils.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lka {

LaneDetector::LaneDetector(int image_width, int image_height, double camera_height,
                           double pitch_angle, double fov_horizontal)
    : image_width_(image_width),
      image_height_(image_height),
      camera_height_(camera_height),
      pitch_angle_(pitch_angle),
      fov_horizontal_(fov_horizontal) {

    // Compute focal length from FOV
    focal_length_ = (image_width / 2.0) / std::tan(fov_horizontal / 2.0);

    // Build camera intrinsic matrix
    K_ << focal_length_, 0, image_width / 2.0,
          0, focal_length_, image_height / 2.0,
          0, 0, 1;
}

void LaneDetector::compute_homography(const Eigen::Vector3d& camera_pos,
                                     const Eigen::Vector3d& camera_forward,
                                     const Eigen::Vector3d& camera_right,
                                     const Eigen::Vector3d& camera_up) {
    // Build rotation matrix (world to camera)
    Eigen::Matrix3d R;
    R.row(0) = camera_right.normalized();
    R.row(1) = camera_up.normalized();
    R.row(2) = camera_forward.normalized();

    // Build translation vector
    Eigen::Vector3d t = -R * camera_pos;

    // Homography for ground plane (z=0)
    // H = K * [r1 r2 t], where r1, r2 are first two columns of R
    Eigen::Matrix3d H_temp;
    H_temp.col(0) = K_ * R.col(0);
    H_temp.col(1) = K_ * R.col(1);
    H_temp.col(2) = K_ * t;

    homography_ = H_temp;
}

bool LaneDetector::project_point(const Point3D& world_point,
                                 const Eigen::Vector3d& camera_pos,
                                 Point2D& image_point) const {
    // Transform to homogeneous coordinates
    Eigen::Vector3d world(world_point.x, world_point.y, 1.0);

    // Apply homography
    Eigen::Vector3d image_homo = homography_ * world;

    // Check if point is behind camera (negative z in homogeneous coords)
    if (image_homo(2) <= 0) {
        return false;
    }

    // Convert to image coordinates
    image_point.x = image_homo(0) / image_homo(2);
    image_point.y = image_homo(1) / image_homo(2);

    // Check if within image bounds
    return (image_point.x >= 0 && image_point.x < image_width_ &&
            image_point.y >= 0 && image_point.y < image_height_);
}

bool LaneDetector::is_in_front_of_camera(const Point3D& point,
                                         const Eigen::Vector3d& camera_pos,
                                         const Eigen::Vector3d& camera_forward) const {
    Eigen::Vector3d to_point(point.x - camera_pos.x(),
                            point.y - camera_pos.y(),
                            point.z - camera_pos.z());
    return to_point.dot(camera_forward) > 0;
}

std::vector<Point3D> LaneDetector::interpolate_boundary(
    const std::vector<Point3D>& boundary,
    double interval) {

    if (boundary.size() < 2) {
        return boundary;
    }

    std::vector<Point3D> interpolated;
    interpolated.reserve(boundary.size() * 4); // Pre-allocate

    for (size_t i = 0; i < boundary.size() - 1; ++i) {
        const Point3D& p1 = boundary[i];
        const Point3D& p2 = boundary[i + 1];

        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        double dz = p2.z - p1.z;
        double segment_length = std::sqrt(dx * dx + dy * dy + dz * dz);

        int num_points = static_cast<int>(segment_length / interval);
        if (num_points < 1) num_points = 1;

        // Add interpolated points (parallelization overhead not worth it for small segments)
        for (int j = 0; j <= num_points; ++j) {
            double t = static_cast<double>(j) / num_points;
            Point3D p(
                p1.x + t * dx,
                p1.y + t * dy,
                p1.z + t * dz
            );
            interpolated.push_back(p);
        }
    }

    // Add last point
    interpolated.push_back(boundary.back());

    return interpolated;
}

std::vector<double> LaneDetector::compute_distances_to_camera(
    const std::vector<Point3D>& points,
    const Eigen::Vector3d& camera_pos) {

    std::vector<double> distances(points.size());

    // Parallelize distance computation with OpenMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < points.size(); ++i) {
        double dx = points[i].x - camera_pos.x();
        double dy = points[i].y - camera_pos.y();
        double dz = points[i].z - camera_pos.z();
        distances[i] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    return distances;
}

std::tuple<LaneBoundary, LaneBoundary> LaneDetector::detect_lanes(
    const std::vector<Point3D>& left_boundary,
    const std::vector<Point3D>& right_boundary,
    const Eigen::Vector3d& camera_pos,
    const Eigen::Vector3d& camera_forward,
    const Eigen::Vector3d& camera_right,
    const Eigen::Vector3d& camera_up,
    double max_distance,
    double sample_interval) {

    // Update homography
    compute_homography(camera_pos, camera_forward, camera_right, camera_up);

    LaneBoundary left_detected, right_detected;

    // Process both boundaries in parallel using OpenMP sections
    #pragma omp parallel sections
    {
        // Process left boundary
        #pragma omp section
        {
            // Interpolate boundary
            std::vector<Point3D> left_interp = interpolate_boundary(left_boundary, sample_interval);

            // Filter points: in front of camera and within max distance
            std::vector<Point3D> left_filtered;
            left_filtered.reserve(left_interp.size());

            for (const auto& point : left_interp) {
                if (!is_in_front_of_camera(point, camera_pos, camera_forward)) {
                    continue;
                }

                double dx = point.x - camera_pos.x();
                double dy = point.y - camera_pos.y();
                double dz = point.z - camera_pos.z();
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist <= max_distance) {
                    left_filtered.push_back(point);
                }
            }

            // Project to image and keep visible points
            for (const auto& point : left_filtered) {
                Point2D img_point;
                if (project_point(point, camera_pos, img_point)) {
                    left_detected.points.push_back(point);
                }
            }

            // Compute distances for detected points
            left_detected.distances = compute_distances_to_camera(
                left_detected.points, camera_pos);
        }

        // Process right boundary (identical logic)
        #pragma omp section
        {
            // Interpolate boundary
            std::vector<Point3D> right_interp = interpolate_boundary(right_boundary, sample_interval);

            // Filter points: in front of camera and within max distance
            std::vector<Point3D> right_filtered;
            right_filtered.reserve(right_interp.size());

            for (const auto& point : right_interp) {
                if (!is_in_front_of_camera(point, camera_pos, camera_forward)) {
                    continue;
                }

                double dx = point.x - camera_pos.x();
                double dy = point.y - camera_pos.y();
                double dz = point.z - camera_pos.z();
                double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (dist <= max_distance) {
                    right_filtered.push_back(point);
                }
            }

            // Project to image and keep visible points
            for (const auto& point : right_filtered) {
                Point2D img_point;
                if (project_point(point, camera_pos, img_point)) {
                    right_detected.points.push_back(point);
                }
            }

            // Compute distances for detected points
            right_detected.distances = compute_distances_to_camera(
                right_detected.points, camera_pos);
        }
    }

    return std::make_tuple(left_detected, right_detected);
}

} // namespace lka
