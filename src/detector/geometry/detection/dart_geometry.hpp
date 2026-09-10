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

    inline cv::Point2f contourCenter(const std::vector<cv::Point> &contour)
    {
        const cv::Moments moments = cv::moments(contour);
        if (moments.m00 == 0.0)
            return cv::Point2f(0.0f, 0.0f);
        return cv::Point2f(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00));
    }

    inline cv::Point2f principalAxis(const std::vector<cv::Point> &points)
    {
        if (points.size() < 2)
            return cv::Point2f(0.0f, 0.0f);

        cv::Point2f mean(0.0f, 0.0f);
        for (const auto &point : points)
            mean += cv::Point2f(point);
        mean *= 1.0f / static_cast<float>(points.size());

        double xx = 0.0;
        double xy = 0.0;
        double yy = 0.0;
        for (const auto &point : points)
        {
            const cv::Point2f offset = cv::Point2f(point) - mean;
            xx += offset.x * offset.x;
            xy += offset.x * offset.y;
            yy += offset.y * offset.y;
        }
        const float angle = 0.5f * std::atan2(
            static_cast<float>(2.0 * xy),
            static_cast<float>(xx - yy));
        return cv::Point2f(std::cos(angle), std::sin(angle));
    }

    // A thresholded frame can contain several sizeable, disconnected changes:
    // the dart's flight/shaft as well as reflections, wires, or an old dart
    // settling. Combining every contour into one convex hull lets an unrelated
    // speck become the selected tip. Keep the primary contour and only add
    // fragments that continue its shaft. The visible shaft is not guaranteed
    // to point toward the bull: a dart in the upper board can enter farther
    // from the bull than its flight appears in a low camera. Accept a strongly
    // elongated, collinear fragment on either side of the primary contour,
    // while retaining the older boardward test for compact connected pieces.
    inline std::vector<cv::Point> collectBoardwardDartPoints(
        const std::vector<std::vector<cv::Point>> &pieces,
        const cv::Point2f &boardCenter,
        float minimumCloserPixels = 2.0f,
        float baseAlignmentTolerancePixels = 32.0f,
        float alignmentToleranceSlope = 0.65f,
        float maximumAlignedFragmentDistancePixels = 240.0f,
        float minimumElongation = 2.2f)
    {
        std::vector<cv::Point> selectedPoints;
        if (pieces.empty())
            return selectedPoints;

        const cv::Point2f primaryCenter = contourCenter(pieces.front());
        if (primaryCenter == cv::Point2f(0.0f, 0.0f))
            return selectedPoints;
        cv::Point2f boardward = boardCenter - primaryCenter;
        const float primaryBoardDistance = cv::norm(boardward);
        if (!std::isfinite(primaryBoardDistance) || primaryBoardDistance <= 0.0f)
            return selectedPoints;
        boardward /= primaryBoardDistance;

        selectedPoints.insert(
            selectedPoints.end(), pieces.front().begin(), pieces.front().end());
        for (size_t index = 1; index < pieces.size(); ++index)
        {
            const cv::Point2f candidateCenter = contourCenter(pieces[index]);
            if (candidateCenter == cv::Point2f(0.0f, 0.0f))
                continue;
            const float candidateBoardDistance = cv::norm(boardCenter - candidateCenter);
            if (!std::isfinite(candidateBoardDistance))
                continue;

            const cv::Point2f offset = candidateCenter - primaryCenter;
            const float forward = offset.dot(boardward);
            const float perpendicular = std::fabs(
                offset.x * boardward.y - offset.y * boardward.x);
            const float allowedPerpendicular = std::max(
                baseAlignmentTolerancePixels,
                std::max(0.0f, forward) * alignmentToleranceSlope);
            const bool continuesBoardward =
                candidateBoardDistance <= primaryBoardDistance - minimumCloserPixels &&
                forward >= -minimumCloserPixels &&
                perpendicular <= allowedPerpendicular;

            const cv::RotatedRect candidateBox = cv::minAreaRect(pieces[index]);
            const float candidateLong = std::max(
                candidateBox.size.width, candidateBox.size.height);
            const float candidateShort = std::max(
                1.0f, std::min(candidateBox.size.width, candidateBox.size.height));
            const float elongation = candidateLong / candidateShort;
            const cv::Point2f candidateAxis = principalAxis(pieces[index]);
            const float candidateDistance = cv::norm(offset);
            const float axisPerpendicular = std::fabs(
                offset.x * candidateAxis.y - offset.y * candidateAxis.x);
            const bool continuesAlignedShaft =
                elongation >= minimumElongation &&
                candidateDistance <= maximumAlignedFragmentDistancePixels &&
                axisPerpendicular <= std::max(
                    baseAlignmentTolerancePixels, candidateShort * 1.5f);

            if (!continuesBoardward && !continuesAlignedShaft)
                continue;

            selectedPoints.insert(
                selectedPoints.end(), pieces[index].begin(), pieces[index].end());
        }
        return selectedPoints;
    }

    // Choose the narrow end of the complete flight/shaft silhouette. The dart
    // point is the tapered end, whereas flights occupy a much wider cross
    // section. If both ends have similar width, use the endpoint closer to the
    // bull as a conservative fallback.
    inline BoardwardTipSelection selectTaperedDartEndpoint(
        const std::vector<cv::Point> &points,
        const cv::Point2f &shapeCenter,
        const cv::Point2f &boardCenter,
        float minimumExtensionPixels = 10.0f,
        float endFraction = 0.22f,
        float decisiveWidthRatio = 0.78f)
    {
        BoardwardTipSelection result;
        const cv::Point2f axis = principalAxis(points);
        if (points.empty() || cv::norm(axis) <= 0.0f)
            return result;
        const cv::Point2f perpendicular(-axis.y, axis.x);

        float minimumProjection = std::numeric_limits<float>::infinity();
        float maximumProjection = -std::numeric_limits<float>::infinity();
        cv::Point2f minimumPoint;
        cv::Point2f maximumPoint;
        for (const auto &candidate : points)
        {
            const float projection = cv::Point2f(candidate).dot(axis);
            if (projection < minimumProjection)
            {
                minimumProjection = projection;
                minimumPoint = cv::Point2f(candidate);
            }
            if (projection > maximumProjection)
            {
                maximumProjection = projection;
                maximumPoint = cv::Point2f(candidate);
            }
        }

        const float span = maximumProjection - minimumProjection;
        if (!std::isfinite(span) || span <= 0.0f)
            return result;
        const float cap = std::max(3.0f, span * endFraction);
        float minimumSideLow = std::numeric_limits<float>::infinity();
        float minimumSideHigh = -std::numeric_limits<float>::infinity();
        float maximumSideLow = std::numeric_limits<float>::infinity();
        float maximumSideHigh = -std::numeric_limits<float>::infinity();
        for (const auto &candidate : points)
        {
            const cv::Point2f point(candidate);
            const float projection = point.dot(axis);
            const float cross = point.dot(perpendicular);
            if (projection <= minimumProjection + cap)
            {
                minimumSideLow = std::min(minimumSideLow, cross);
                minimumSideHigh = std::max(minimumSideHigh, cross);
            }
            if (projection >= maximumProjection - cap)
            {
                maximumSideLow = std::min(maximumSideLow, cross);
                maximumSideHigh = std::max(maximumSideHigh, cross);
            }
        }

        const float minimumWidth = minimumSideHigh - minimumSideLow;
        const float maximumWidth = maximumSideHigh - maximumSideLow;
        if (minimumWidth <= maximumWidth * decisiveWidthRatio)
            result.point = minimumPoint;
        else if (maximumWidth <= minimumWidth * decisiveWidthRatio)
            result.point = maximumPoint;
        else
            result.point = cv::norm(minimumPoint - boardCenter) <=
                                   cv::norm(maximumPoint - boardCenter)
                               ? minimumPoint
                               : maximumPoint;

        result.extensionPixels = cv::norm(result.point - shapeCenter);
        result.valid = std::isfinite(result.extensionPixels) &&
                       result.extensionPixels >= minimumExtensionPixels;
        return result;
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
