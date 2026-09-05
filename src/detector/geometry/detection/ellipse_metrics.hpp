#pragma once

#include <opencv2/core.hpp>

#include <cmath>
#include <limits>

namespace score_processing
{
    // Returns 0 at the ellipse center, 1 on its boundary, and values greater
    // than 1 outside. Invalid ellipses return NaN so callers cannot
    // accidentally classify a point as inside them.
    inline double normalizedEllipseRadius(
        const cv::Point2f &point,
        const cv::RotatedRect &ellipse)
    {
        const double radius_x = static_cast<double>(ellipse.size.width) / 2.0;
        const double radius_y = static_cast<double>(ellipse.size.height) / 2.0;
        if (!std::isfinite(radius_x) || !std::isfinite(radius_y) ||
            radius_x <= 0.0 || radius_y <= 0.0)
        {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const cv::Point2f relative = point - ellipse.center;
        const double angle_radians =
            -static_cast<double>(ellipse.angle) * CV_PI / 180.0;
        const double cos_angle = std::cos(angle_radians);
        const double sin_angle = std::sin(angle_radians);
        const double rotated_x =
            static_cast<double>(relative.x) * cos_angle -
            static_cast<double>(relative.y) * sin_angle;
        const double rotated_y =
            static_cast<double>(relative.x) * sin_angle +
            static_cast<double>(relative.y) * cos_angle;

        return std::hypot(rotated_x / radius_x, rotated_y / radius_y);
    }

    // Negative values are inside the ellipse, zero is the boundary, and
    // positive values are outside. The unit is normalized ellipse radius.
    inline double signedNormalizedEllipseMargin(
        const cv::Point2f &point,
        const cv::RotatedRect &ellipse)
    {
        return normalizedEllipseRadius(point, ellipse) - 1.0;
    }
} // namespace score_processing
