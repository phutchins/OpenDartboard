#pragma once

#include <opencv2/opencv.hpp>

#include <cmath>
#include <limits>
#include <vector>

namespace dart_geometry
{
    struct BoardwardTipSelection
    {
        bool valid = false;
        cv::Point2f point{0.0f, 0.0f};
        float extensionPixels = 0.0f;
    };

    // The largest changed shape is normally the dart's flight. Its boardward
    // hull extreme is the point, while the opposite extreme can be a flight or
    // even a contour clipped by the camera frame.
    inline BoardwardTipSelection selectBoardwardHullPoint(
        const std::vector<cv::Point> &hull,
        const cv::Point2f &shapeCenter,
        const cv::Point2f &boardCenter,
        float minimumExtensionPixels = 10.0f)
    {
        BoardwardTipSelection result;
        cv::Point2f boardward = boardCenter - shapeCenter;
        const float boardwardLength = cv::norm(boardward);
        if (hull.empty() || !std::isfinite(boardwardLength) || boardwardLength <= 0.0f)
            return result;
        boardward /= boardwardLength;

        float bestProjection = -std::numeric_limits<float>::infinity();
        for (const auto &candidate : hull)
        {
            const cv::Point2f offset = cv::Point2f(candidate) - shapeCenter;
            const float projection = offset.dot(boardward);
            if (projection > bestProjection)
            {
                bestProjection = projection;
                result.point = cv::Point2f(candidate);
                result.extensionPixels = cv::norm(offset);
            }
        }

        result.valid = std::isfinite(bestProjection) &&
                       result.extensionPixels >= minimumExtensionPixels;
        return result;
    }

    inline bool isPlausibleBoardPoint(
        const cv::Point2f &point,
        const cv::Point2f &boardCenter,
        const cv::RotatedRect &outerDoubleEllipse,
        float maximumNormalizedRadius = 1.45f)
    {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            outerDoubleEllipse.size.width <= 0.0f || outerDoubleEllipse.size.height <= 0.0f)
            return false;

        const float angle = -outerDoubleEllipse.angle * static_cast<float>(CV_PI) / 180.0f;
        const float cosAngle = std::cos(angle);
        const float sinAngle = std::sin(angle);
        const cv::Point2f relative = point - boardCenter;
        const cv::Point2f rotated(
            relative.x * cosAngle - relative.y * sinAngle,
            relative.x * sinAngle + relative.y * cosAngle);
        const float radiusX = outerDoubleEllipse.size.width * 0.5f;
        const float radiusY = outerDoubleEllipse.size.height * 0.5f;
        const float normalizedRadius = std::sqrt(
            (rotated.x * rotated.x) / (radiusX * radiusX) +
            (rotated.y * rotated.y) / (radiusY * radiusY));
        return std::isfinite(normalizedRadius) && normalizedRadius <= maximumNormalizedRadius;
    }
}
