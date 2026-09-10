#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
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

    // A thresholded frame can contain several sizeable, disconnected changes:
    // the dart's flight/shaft as well as reflections, wires, or an old dart
    // settling. Combining every contour into one convex hull lets an unrelated
    // speck become the selected tip. Keep the primary contour and only add
    // fragments that continue from it toward the board.
    inline std::vector<cv::Point> collectBoardwardDartPoints(
        const std::vector<std::vector<cv::Point>> &pieces,
        const cv::Point2f &boardCenter,
        float minimumCloserPixels = 2.0f,
        float baseAlignmentTolerancePixels = 32.0f,
        float alignmentToleranceSlope = 0.65f)
    {
        std::vector<cv::Point> selectedPoints;
        if (pieces.empty())
            return selectedPoints;

        const cv::Moments primaryMoments = cv::moments(pieces.front());
        if (primaryMoments.m00 == 0.0)
            return selectedPoints;

        const cv::Point2f primaryCenter(
            static_cast<float>(primaryMoments.m10 / primaryMoments.m00),
            static_cast<float>(primaryMoments.m01 / primaryMoments.m00));
        cv::Point2f boardward = boardCenter - primaryCenter;
        const float primaryBoardDistance = cv::norm(boardward);
        if (!std::isfinite(primaryBoardDistance) || primaryBoardDistance <= 0.0f)
            return selectedPoints;
        boardward /= primaryBoardDistance;

        selectedPoints.insert(
            selectedPoints.end(), pieces.front().begin(), pieces.front().end());
        for (size_t index = 1; index < pieces.size(); ++index)
        {
            const cv::Moments candidateMoments = cv::moments(pieces[index]);
            if (candidateMoments.m00 == 0.0)
                continue;
            const cv::Point2f candidateCenter(
                static_cast<float>(candidateMoments.m10 / candidateMoments.m00),
                static_cast<float>(candidateMoments.m01 / candidateMoments.m00));
            const float candidateBoardDistance = cv::norm(boardCenter - candidateCenter);
            if (!std::isfinite(candidateBoardDistance) ||
                candidateBoardDistance > primaryBoardDistance - minimumCloserPixels)
                continue;

            const cv::Point2f offset = candidateCenter - primaryCenter;
            const float forward = offset.dot(boardward);
            const float perpendicular = std::fabs(
                offset.x * boardward.y - offset.y * boardward.x);
            const float allowedPerpendicular = std::max(
                baseAlignmentTolerancePixels,
                std::max(0.0f, forward) * alignmentToleranceSlope);
            if (forward < -minimumCloserPixels || perpendicular > allowedPerpendicular)
                continue;

            selectedPoints.insert(
                selectedPoints.end(), pieces[index].begin(), pieces[index].end());
        }
        return selectedPoints;
    }

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

    // State changes need evidence that a dart actually remained on or directly
    // around the board. A miss near the edge can present a full dart silhouette
    // to only one camera, so allow the other cameras' smaller persistent masks
    // to corroborate it without requiring them to produce a second tip.
    inline bool hasSufficientDartEvidence(
        size_t camerasMovingUp,
        size_t camerasWithTips,
        size_t camerasSupportingPersistentChange,
        bool orientedTipInsideScoringArea)
    {
        if (camerasMovingUp == 0 || camerasWithTips == 0)
            return false;
        if (orientedTipInsideScoringArea)
            return true;
        return camerasMovingUp >= 2 || camerasSupportingPersistentChange >= 2;
    }

    // A miss has no ring/wedge score. It is nevertheless real when two cameras
    // locate tips, or when one tip is corroborated by persistent image changes
    // from at least one more camera.
    inline bool hasSufficientMissEvidence(
        size_t camerasWithTips,
        size_t camerasSupportingPersistentChange)
    {
        return camerasWithTips >= 2 ||
               (camerasWithTips >= 1 && camerasSupportingPersistentChange >= 2);
    }
}
