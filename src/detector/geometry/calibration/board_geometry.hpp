#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace board_geometry
{
    inline cv::Point2f normalizeVector(const cv::Point2f &vector, const cv::RotatedRect &ellipse)
    {
        const float radiusX = ellipse.size.width * 0.5f;
        const float radiusY = ellipse.size.height * 0.5f;
        if (radiusX <= 0.0f || radiusY <= 0.0f)
            return cv::Point2f(0.0f, 0.0f);

        const float angle = -ellipse.angle * static_cast<float>(CV_PI) / 180.0f;
        const float cosAngle = std::cos(angle);
        const float sinAngle = std::sin(angle);
        const cv::Point2f rotated(
            vector.x * cosAngle - vector.y * sinAngle,
            vector.x * sinAngle + vector.y * cosAngle);
        return cv::Point2f(rotated.x / radiusX, rotated.y / radiusY);
    }

    inline cv::Point2f denormalizeVector(const cv::Point2f &vector, const cv::RotatedRect &ellipse)
    {
        const float angle = ellipse.angle * static_cast<float>(CV_PI) / 180.0f;
        const float cosAngle = std::cos(angle);
        const float sinAngle = std::sin(angle);
        const cv::Point2f scaled(
            vector.x * ellipse.size.width * 0.5f,
            vector.y * ellipse.size.height * 0.5f);
        return cv::Point2f(
            scaled.x * cosAngle - scaled.y * sinAngle,
            scaled.x * sinAngle + scaled.y * cosAngle);
    }

    inline float angleDegrees(const cv::Point2f &vector)
    {
        float angle = std::atan2(vector.y, vector.x) * 180.0f / static_cast<float>(CV_PI);
        if (angle < 0.0f)
            angle += 360.0f;
        return angle;
    }

    inline float normalizedAngleDegrees(
        const cv::Point2f &point,
        const cv::Point2f &center,
        const cv::RotatedRect &ellipse)
    {
        return angleDegrees(normalizeVector(point - center, ellipse));
    }

    inline float circularAngleDifference(float first, float second)
    {
        float difference = std::fabs(first - second);
        return std::min(difference, 360.0f - difference);
    }

    inline std::vector<std::vector<cv::Point2f>> groupByNormalizedAngle(
        const std::vector<cv::Point2f> &points,
        const cv::Point2f &center,
        const cv::RotatedRect &ellipse,
        float toleranceDegrees)
    {
        struct AngularPoint
        {
            cv::Point2f point;
            float angle;
        };

        std::vector<AngularPoint> sortedPoints;
        sortedPoints.reserve(points.size());
        for (const auto &point : points)
        {
            const cv::Point2f normalized = normalizeVector(point - center, ellipse);
            if (!std::isfinite(normalized.x) || !std::isfinite(normalized.y) || cv::norm(normalized) <= 0.0f)
                continue;
            sortedPoints.push_back({point, angleDegrees(normalized)});
        }

        std::sort(sortedPoints.begin(), sortedPoints.end(), [](const AngularPoint &left, const AngularPoint &right) {
            return left.angle < right.angle;
        });

        std::vector<std::vector<cv::Point2f>> groups;
        std::vector<float> groupAnchorAngles;
        for (const auto &candidate : sortedPoints)
        {
            if (groups.empty() ||
                circularAngleDifference(candidate.angle, groupAnchorAngles.back()) > toleranceDegrees)
            {
                groups.push_back({candidate.point});
                groupAnchorAngles.push_back(candidate.angle);
            }
            else
            {
                groups.back().push_back(candidate.point);
            }
        }

        // The first and last groups can be the same direction split across 0/360 degrees.
        if (groups.size() > 1 &&
            circularAngleDifference(groupAnchorAngles.front(), groupAnchorAngles.back()) <= toleranceDegrees)
        {
            groups.front().insert(groups.front().end(), groups.back().begin(), groups.back().end());
            groups.pop_back();
        }

        return groups;
    }

    struct WireSpacingDiagnostics
    {
        bool valid = false;
        size_t count = 0;
        float minGapDegrees = 0.0f;
        float maxGapDegrees = 0.0f;
        float rmsGapErrorDegrees = 0.0f;
    };

    inline WireSpacingDiagnostics validateWireSpacing(
        const std::vector<cv::Point2f> &wires,
        const cv::Point2f &center,
        const cv::RotatedRect &ellipse,
        size_t expectedCount = 20,
        float minGapDegrees = 8.0f,
        float maxGapDegrees = 28.0f,
        float maxRmsGapErrorDegrees = 6.0f)
    {
        WireSpacingDiagnostics result;
        result.count = wires.size();
        if (wires.size() != expectedCount || expectedCount < 2)
            return result;

        std::vector<float> angles;
        angles.reserve(wires.size());
        for (const auto &wire : wires)
        {
            const cv::Point2f normalized = normalizeVector(wire - center, ellipse);
            if (!std::isfinite(normalized.x) || !std::isfinite(normalized.y) || cv::norm(normalized) <= 0.0f)
                return result;
            angles.push_back(angleDegrees(normalized));
        }
        std::sort(angles.begin(), angles.end());

        const float expectedGap = 360.0f / static_cast<float>(expectedCount);
        result.minGapDegrees = std::numeric_limits<float>::max();
        float squaredError = 0.0f;
        for (size_t index = 0; index < angles.size(); ++index)
        {
            float gap = angles[(index + 1) % angles.size()] - angles[index];
            if (index + 1 == angles.size())
                gap += 360.0f;
            result.minGapDegrees = std::min(result.minGapDegrees, gap);
            result.maxGapDegrees = std::max(result.maxGapDegrees, gap);
            const float error = gap - expectedGap;
            squaredError += error * error;
        }
        result.rmsGapErrorDegrees = std::sqrt(squaredError / static_cast<float>(angles.size()));
        result.valid = result.minGapDegrees >= minGapDegrees &&
                       result.maxGapDegrees <= maxGapDegrees &&
                       result.rmsGapErrorDegrees <= maxRmsGapErrorDegrees;
        return result;
    }

    enum class ClipLayout
    {
        UNKNOWN,
        LEFT_BIASED,
        BALANCED,
        RIGHT_BIASED
    };

    struct ClipLayoutDiagnostics
    {
        ClipLayout layout = ClipLayout::UNKNOWN;
        size_t validCount = 0;
        int leftCount = 0;
        int rightCount = 0;
        float horizontalBalance = 0.0f;
    };

    inline ClipLayoutDiagnostics analyzeClipLayout(
        const std::vector<cv::Point2f> &clipPoints,
        const cv::Point2f &center,
        float balancedThreshold = 0.20f,
        float sideBiasThreshold = 0.35f)
    {
        ClipLayoutDiagnostics result;
        float signedHorizontal = 0.0f;
        float absoluteHorizontal = 0.0f;
        for (const auto &point : clipPoints)
        {
            cv::Point2f direction = point - center;
            const float length = cv::norm(direction);
            if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || length <= 0.0f)
                continue;
            direction /= length;
            signedHorizontal += direction.x;
            absoluteHorizontal += std::fabs(direction.x);
            if (direction.x < -0.05f)
                result.leftCount++;
            else if (direction.x > 0.05f)
                result.rightCount++;
            result.validCount++;
        }

        if (result.validCount < 2 || result.validCount > 4 || absoluteHorizontal <= 0.0f)
            return result;

        result.horizontalBalance = signedHorizontal / absoluteHorizontal;
        if (result.leftCount > 0 && result.rightCount > 0 &&
            std::fabs(result.horizontalBalance) <= balancedThreshold)
        {
            result.layout = ClipLayout::BALANCED;
        }
        else if (result.horizontalBalance <= -sideBiasThreshold)
        {
            result.layout = ClipLayout::LEFT_BIASED;
        }
        else if (result.horizontalBalance >= sideBiasThreshold)
        {
            result.layout = ClipLayout::RIGHT_BIASED;
        }
        return result;
    }
} // namespace board_geometry
