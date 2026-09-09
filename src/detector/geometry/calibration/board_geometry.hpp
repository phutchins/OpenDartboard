#pragma once

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace board_geometry
{
    // One projective mapping is the source of truth for every scoring ring,
    // wire, overlay, and normalized dart position. Board coordinates are in
    // millimetres with the bull at (0, 0), +x right, and +y down.
    struct PlanarBoardTransform
    {
        std::array<float, 9> boardToImage{{1.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f}};
        std::array<float, 9> imageToBoard{{1.0f, 0.0f, 0.0f,
                                           0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f}};
        bool valid = false;
        bool manual = false;
        float ringResidualMeanPixels = std::numeric_limits<float>::infinity();
        float ringResidualP90Pixels = std::numeric_limits<float>::infinity();
        int residualSamples = 0;
    };

    inline bool projectPoint(
        const std::array<float, 9> &matrix,
        const cv::Point2f &input,
        cv::Point2f &output)
    {
        const float denominator =
            matrix[6] * input.x + matrix[7] * input.y + matrix[8];
        if (!std::isfinite(denominator) || std::fabs(denominator) < 1.0e-6f)
            return false;
        output.x = (matrix[0] * input.x + matrix[1] * input.y + matrix[2]) / denominator;
        output.y = (matrix[3] * input.x + matrix[4] * input.y + matrix[5]) / denominator;
        return std::isfinite(output.x) && std::isfinite(output.y);
    }

    inline bool boardToImage(
        const PlanarBoardTransform &transform,
        const cv::Point2f &boardPoint,
        cv::Point2f &imagePoint)
    {
        return transform.valid && projectPoint(transform.boardToImage, boardPoint, imagePoint);
    }

    inline bool imageToBoard(
        const PlanarBoardTransform &transform,
        const cv::Point2f &imagePoint,
        cv::Point2f &boardPoint)
    {
        return transform.valid && projectPoint(transform.imageToBoard, imagePoint, boardPoint);
    }

    inline PlanarBoardTransform estimatePlanarBoardTransform(
        const std::vector<cv::Point2f> &boardPoints,
        const std::vector<cv::Point2f> &imagePoints,
        bool manual = false)
    {
        PlanarBoardTransform result;
        result.manual = manual;
        if (boardPoints.size() < 4 || boardPoints.size() != imagePoints.size())
            return result;

        const cv::Mat homography = cv::findHomography(boardPoints, imagePoints, 0);
        if (homography.empty() || homography.rows != 3 || homography.cols != 3)
            return result;
        cv::Mat homography64;
        homography.convertTo(homography64, CV_64F);
        const cv::Mat inverse = homography64.inv(cv::DECOMP_SVD);
        if (inverse.empty())
            return result;

        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 3; ++column)
            {
                const int index = row * 3 + column;
                result.boardToImage[index] = static_cast<float>(homography64.at<double>(row, column));
                result.imageToBoard[index] = static_cast<float>(inverse.at<double>(row, column));
                if (!std::isfinite(result.boardToImage[index]) ||
                    !std::isfinite(result.imageToBoard[index]))
                    return PlanarBoardTransform();
            }
        }

        cv::Point2f projectedCenter;
        result.valid = projectPoint(result.boardToImage, cv::Point2f(0.0f, 0.0f), projectedCenter);
        return result;
    }

    inline std::vector<cv::Point2f> projectedRingPoints(
        const PlanarBoardTransform &transform,
        float radius,
        int sampleCount = 180)
    {
        std::vector<cv::Point2f> points;
        if (!transform.valid || radius <= 0.0f || sampleCount < 8)
            return points;
        points.reserve(sampleCount);
        for (int index = 0; index < sampleCount; ++index)
        {
            const float angle = static_cast<float>(2.0 * CV_PI * index / sampleCount);
            cv::Point2f projected;
            if (boardToImage(
                    transform,
                    cv::Point2f(radius * std::cos(angle), radius * std::sin(angle)),
                    projected))
                points.push_back(projected);
        }
        return points;
    }

    inline cv::RotatedRect fitProjectedRingEllipse(
        const PlanarBoardTransform &transform,
        float radius)
    {
        const auto points = projectedRingPoints(transform, radius, 180);
        return points.size() >= 5 ? cv::fitEllipse(points) : cv::RotatedRect();
    }

    inline bool rayEllipseIntersectionDistance(
        const cv::Point2f &origin,
        const cv::Point2f &direction,
        const cv::RotatedRect &ellipse,
        float &distance)
    {
        const float major = ellipse.size.width * 0.5f;
        const float minor = ellipse.size.height * 0.5f;
        const float directionLength = cv::norm(direction);
        if (major <= 0.0f || minor <= 0.0f || directionLength <= 0.0f)
            return false;

        const float rotation = -ellipse.angle * static_cast<float>(CV_PI) / 180.0f;
        const float cosine = std::cos(rotation);
        const float sine = std::sin(rotation);
        const cv::Point2f normalizedDirection = direction / directionLength;
        const cv::Point2f relative = origin - ellipse.center;
        const cv::Point2f rotatedOrigin(
            relative.x * cosine - relative.y * sine,
            relative.x * sine + relative.y * cosine);
        const cv::Point2f rotatedDirection(
            normalizedDirection.x * cosine - normalizedDirection.y * sine,
            normalizedDirection.x * sine + normalizedDirection.y * cosine);
        const float a =
            rotatedDirection.x * rotatedDirection.x / (major * major) +
            rotatedDirection.y * rotatedDirection.y / (minor * minor);
        const float b = 2.0f * (
            rotatedOrigin.x * rotatedDirection.x / (major * major) +
            rotatedOrigin.y * rotatedDirection.y / (minor * minor));
        const float c =
            rotatedOrigin.x * rotatedOrigin.x / (major * major) +
            rotatedOrigin.y * rotatedOrigin.y / (minor * minor) - 1.0f;
        const float discriminant = b * b - 4.0f * a * c;
        if (a <= 0.0f || discriminant < 0.0f)
            return false;
        const float root = std::sqrt(discriminant);
        const float first = (-b - root) / (2.0f * a);
        const float second = (-b + root) / (2.0f * a);
        distance = std::max(first, second);
        if (distance <= 0.0f)
            distance = std::min(first, second);
        return std::isfinite(distance) && distance > 0.0f;
    }

    struct RingPairDiagnostics
    {
        bool valid = false;
        float areaRatio = 0.0f;
        float expectedAreaRatio = 0.0f;
        float centerOffsetRatio = 0.0f;
        float aspectRatioDifference = 0.0f;
        float majorAxisAngleDifferenceDegrees = 0.0f;
        const char *reason = "invalid_ellipse";
    };

    inline RingPairDiagnostics validateProjectedRingPair(
        const cv::RotatedRect &inner,
        const cv::RotatedRect &outer,
        float expectedInnerRadius,
        float expectedOuterRadius,
        float minAreaRatio,
        float maxAreaRatio,
        float maxCenterOffsetRatio,
        float maxAspectRatioDifference,
        float maxAngleDifferenceDegrees);

    struct RawRingRefinement
    {
        bool valid = false;
        cv::RotatedRect inner;
        cv::RotatedRect outer;
        int innerPointCount = 0;
        int outerPointCount = 0;
        RingPairDiagnostics diagnostics;
    };

    inline bool maskOccupied(const cv::Mat &mask, const cv::Point2f &point)
    {
        const int x = cvRound(point.x);
        const int y = cvRound(point.y);
        if (x < 1 || y < 1 || x >= mask.cols - 1 || y >= mask.rows - 1)
            return false;
        int occupied = 0;
        for (int row = y - 1; row <= y + 1; ++row)
            for (int column = x - 1; column <= x + 1; ++column)
                occupied += mask.at<uchar>(row, column) > 0 ? 1 : 0;
        return occupied >= 3;
    }

    inline std::vector<cv::Point2f> refineRingBoundaryFromRawMask(
        const cv::Mat &rawMask,
        const cv::Point2f &boardCenter,
        const cv::RotatedRect &seed,
        bool enteringColor,
        float searchWindowPixels = 36.0f,
        float angleStepDegrees = 2.0f)
    {
        std::vector<cv::Point2f> points;
        if (rawMask.empty() || rawMask.type() != CV_8UC1 ||
            seed.size.width <= 0.0f || seed.size.height <= 0.0f)
            return points;

        for (float angleDegrees = 0.0f; angleDegrees < 360.0f; angleDegrees += angleStepDegrees)
        {
            const float angle = angleDegrees * static_cast<float>(CV_PI) / 180.0f;
            const cv::Point2f direction(std::cos(angle), std::sin(angle));
            float seedDistance = 0.0f;
            if (!rayEllipseIntersectionDistance(boardCenter, direction, seed, seedDistance))
                continue;

            const int firstDistance = std::max(2, cvFloor(seedDistance - searchWindowPixels));
            const int lastDistance = cvCeil(seedDistance + searchWindowPixels);
            float bestDistance = -1.0f;
            float bestError = std::numeric_limits<float>::infinity();
            for (int distance = firstDistance; distance <= lastDistance; ++distance)
            {
                const bool before = maskOccupied(rawMask, boardCenter + direction * (distance - 2.0f));
                const bool after = maskOccupied(rawMask, boardCenter + direction * (distance + 2.0f));
                const bool transition = enteringColor ? (!before && after) : (before && !after);
                if (!transition)
                    continue;
                const float error = std::fabs(static_cast<float>(distance) - seedDistance);
                if (error < bestError)
                {
                    bestError = error;
                    bestDistance = static_cast<float>(distance);
                }
            }
            if (bestDistance > 0.0f)
                points.push_back(boardCenter + direction * bestDistance);
        }
        return points;
    }

    inline RawRingRefinement refineRingPairFromRawMask(
        const cv::Mat &rawMask,
        const cv::Point2f &boardCenter,
        const cv::RotatedRect &innerSeed,
        const cv::RotatedRect &outerSeed,
        float physicalInnerRadius,
        float physicalOuterRadius,
        float searchWindowPixels = 36.0f,
        int minimumPoints = 60)
    {
        RawRingRefinement result;
        const auto innerPoints = refineRingBoundaryFromRawMask(
            rawMask, boardCenter, innerSeed, true, searchWindowPixels);
        const auto outerPoints = refineRingBoundaryFromRawMask(
            rawMask, boardCenter, outerSeed, false, searchWindowPixels);
        result.innerPointCount = static_cast<int>(innerPoints.size());
        result.outerPointCount = static_cast<int>(outerPoints.size());
        if (result.innerPointCount < minimumPoints || result.outerPointCount < minimumPoints)
            return result;
        result.inner = cv::fitEllipse(innerPoints);
        result.outer = cv::fitEllipse(outerPoints);
        result.diagnostics = validateProjectedRingPair(
            result.inner,
            result.outer,
            physicalInnerRadius,
            physicalOuterRadius,
            0.72f,
            0.96f,
            0.22f,
            0.14f,
            14.0f);
        result.valid = result.diagnostics.valid;
        return result;
    }

    struct RingEdgeResidual
    {
        bool valid = false;
        float meanPixels = std::numeric_limits<float>::infinity();
        float p90Pixels = std::numeric_limits<float>::infinity();
        int samples = 0;
    };

    inline RingEdgeResidual measureRingEdgeResidual(
        const cv::Mat &rawMask,
        const PlanarBoardTransform &transform,
        float maximumMeanPixels = 5.0f,
        float maximumP90Pixels = 10.0f)
    {
        RingEdgeResidual result;
        if (rawMask.empty() || rawMask.type() != CV_8UC1 || !transform.valid)
            return result;

        cv::Mat edgeMask;
        cv::morphologyEx(
            rawMask,
            edgeMask,
            cv::MORPH_GRADIENT,
            cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        cv::Mat inverseEdges;
        cv::threshold(edgeMask, inverseEdges, 0, 255, cv::THRESH_BINARY_INV);
        cv::Mat distances;
        cv::distanceTransform(inverseEdges, distances, cv::DIST_L2, 3);

        std::vector<float> samples;
        for (const float radius : {99.0f, 107.0f, 162.0f, 170.0f})
        {
            for (const auto &point : projectedRingPoints(transform, radius, 180))
            {
                const int x = cvRound(point.x);
                const int y = cvRound(point.y);
                if (x < 0 || y < 0 || x >= distances.cols || y >= distances.rows)
                    continue;
                const float distance = distances.at<float>(y, x);
                if (std::isfinite(distance))
                    samples.push_back(distance);
            }
        }
        result.samples = static_cast<int>(samples.size());
        if (samples.size() < 500)
            return result;
        std::sort(samples.begin(), samples.end());
        double total = 0.0;
        for (float sample : samples)
            total += sample;
        result.meanPixels = static_cast<float>(total / samples.size());
        result.p90Pixels = samples[static_cast<size_t>(0.90 * (samples.size() - 1))];
        result.valid = result.meanPixels <= maximumMeanPixels &&
                       result.p90Pixels <= maximumP90Pixels;
        return result;
    }

    enum class BullCenterSource
    {
        NONE,
        INNER_BULL,
        OUTER_BULL
    };

    struct BullCenterRefinement
    {
        bool accepted = false;
        cv::Point2f center{0.0f, 0.0f};
        BullCenterSource source = BullCenterSource::NONE;
        float displacementPixels = 0.0f;
        float maxDisplacementPixels = 0.0f;
        float innerOuterDistancePixels = 0.0f;
        float maxInnerOuterDistancePixels = 0.0f;
        const char *reason = "no_usable_bull_ellipse";
    };

    inline const char *bullCenterSourceToString(BullCenterSource source)
    {
        switch (source)
        {
        case BullCenterSource::INNER_BULL:
            return "inner_bull";
        case BullCenterSource::OUTER_BULL:
            return "outer_bull";
        default:
            return "none";
        }
    }

    inline bool hasUsableEllipseCenter(
        const cv::RotatedRect &ellipse,
        const cv::Size &frameSize)
    {
        return frameSize.width > 0 && frameSize.height > 0 &&
               std::isfinite(ellipse.center.x) && std::isfinite(ellipse.center.y) &&
               std::isfinite(ellipse.size.width) && std::isfinite(ellipse.size.height) &&
               ellipse.size.width > 0.0f && ellipse.size.height > 0.0f &&
               ellipse.center.x >= 0.0f && ellipse.center.x < frameSize.width &&
               ellipse.center.y >= 0.0f && ellipse.center.y < frameSize.height;
    }

    inline float ellipseArea(const cv::RotatedRect &ellipse)
    {
        return ellipse.size.width * ellipse.size.height;
    }

    inline float ellipseAspectRatio(const cv::RotatedRect &ellipse)
    {
        const float major = std::max(ellipse.size.width, ellipse.size.height);
        const float minor = std::min(ellipse.size.width, ellipse.size.height);
        return major > 0.0f ? minor / major : 0.0f;
    }

    inline float ellipseMajorAxisAngle(const cv::RotatedRect &ellipse)
    {
        float angle = ellipse.angle + (ellipse.size.width < ellipse.size.height ? 90.0f : 0.0f);
        while (angle < 0.0f)
            angle += 180.0f;
        while (angle >= 180.0f)
            angle -= 180.0f;
        return angle;
    }

    inline float undirectedAngleDifference(float first, float second)
    {
        const float difference = std::fabs(first - second);
        return std::min(difference, 180.0f - difference);
    }

    // Neighboring circles on a real board remain a close, nested pair after
    // perspective projection. These checks reject plausible-looking contours
    // that actually span felt outside the physical ring wires.
    inline RingPairDiagnostics validateProjectedRingPair(
        const cv::RotatedRect &inner,
        const cv::RotatedRect &outer,
        float expectedInnerRadius,
        float expectedOuterRadius,
        float minAreaRatio = 0.80f,
        float maxAreaRatio = 0.94f,
        float maxCenterOffsetRatio = 0.18f,
        float maxAspectRatioDifference = 0.12f,
        float maxAngleDifferenceDegrees = 12.0f)
    {
        RingPairDiagnostics result;
        const float innerArea = ellipseArea(inner);
        const float outerArea = ellipseArea(outer);
        const float outerMinorRadius = std::min(outer.size.width, outer.size.height) * 0.5f;
        if (!std::isfinite(innerArea) || !std::isfinite(outerArea) ||
            !std::isfinite(expectedInnerRadius) || !std::isfinite(expectedOuterRadius) ||
            innerArea <= 0.0f || outerArea <= 0.0f || outerMinorRadius <= 0.0f ||
            expectedInnerRadius <= 0.0f || expectedOuterRadius <= expectedInnerRadius)
            return result;

        result.areaRatio = innerArea / outerArea;
        result.expectedAreaRatio =
            (expectedInnerRadius * expectedInnerRadius) /
            (expectedOuterRadius * expectedOuterRadius);
        result.centerOffsetRatio = cv::norm(inner.center - outer.center) / outerMinorRadius;
        result.aspectRatioDifference =
            std::fabs(ellipseAspectRatio(inner) - ellipseAspectRatio(outer));
        result.majorAxisAngleDifferenceDegrees = undirectedAngleDifference(
            ellipseMajorAxisAngle(inner),
            ellipseMajorAxisAngle(outer));

        const float innerMajor = std::max(inner.size.width, inner.size.height);
        const float innerMinor = std::min(inner.size.width, inner.size.height);
        const float outerMajor = std::max(outer.size.width, outer.size.height);
        const float outerMinor = std::min(outer.size.width, outer.size.height);
        if (innerMajor >= outerMajor || innerMinor >= outerMinor)
        {
            result.reason = "not_nested";
            return result;
        }
        if (result.areaRatio < minAreaRatio || result.areaRatio > maxAreaRatio)
        {
            result.reason = "implausible_ring_width";
            return result;
        }
        if (result.centerOffsetRatio > maxCenterOffsetRatio)
        {
            result.reason = "centers_disagree";
            return result;
        }
        if (result.aspectRatioDifference > maxAspectRatioDifference)
        {
            result.reason = "perspective_shape_disagrees";
            return result;
        }
        if (result.majorAxisAngleDifferenceDegrees > maxAngleDifferenceDegrees)
        {
            result.reason = "axes_disagree";
            return result;
        }

        result.valid = true;
        result.reason = "valid";
        return result;
    }

    struct RingPairCorrection
    {
        bool valid = false;
        cv::RotatedRect inner;
        cv::RotatedRect outer;
        RingPairDiagnostics sourceDiagnostics;
        RingPairDiagnostics correctedDiagnostics;
        float centerSeparationScale = 0.0f;
    };

    // Color-mask morphology is intentionally generous so the twenty red and
    // green segments form one connected ring. That makes the fitted band wider
    // than the wire-to-wire scoring band. Preserve its measured centerline and
    // perspective, then collapse the two boundaries to the official radii.
    inline RingPairCorrection correctProjectedRingPair(
        const cv::RotatedRect &detectedInner,
        const cv::RotatedRect &detectedOuter,
        float physicalInnerRadius,
        float physicalOuterRadius)
    {
        RingPairCorrection result;
        result.sourceDiagnostics = validateProjectedRingPair(
            detectedInner,
            detectedOuter,
            physicalInnerRadius,
            physicalOuterRadius,
            0.60f,
            0.96f,
            0.30f,
            0.16f,
            15.0f);
        if (!result.sourceDiagnostics.valid)
            return result;

        const float physicalMidRadius =
            (physicalInnerRadius + physicalOuterRadius) * 0.5f;
        const float targetHalfWidthRatio =
            (physicalOuterRadius - physicalInnerRadius) /
            (physicalOuterRadius + physicalInnerRadius);
        const float detectedRadiusRatio = std::sqrt(result.sourceDiagnostics.areaRatio);
        const float detectedHalfWidthRatio =
            (1.0f - detectedRadiusRatio) / (1.0f + detectedRadiusRatio);
        if (!std::isfinite(physicalMidRadius) || physicalMidRadius <= 0.0f ||
            !std::isfinite(detectedHalfWidthRatio) || detectedHalfWidthRatio <= 0.0f)
            return result;

        result.centerSeparationScale = std::min(
            1.0f,
            targetHalfWidthRatio / detectedHalfWidthRatio);
        const cv::Point2f middleCenter =
            (detectedOuter.center + detectedInner.center) * 0.5f;
        const cv::Point2f halfCenterSeparation =
            (detectedOuter.center - detectedInner.center) * 0.5f *
            result.centerSeparationScale;

        const float middleMajor =
            (std::max(detectedOuter.size.width, detectedOuter.size.height) +
             std::max(detectedInner.size.width, detectedInner.size.height)) *
            0.5f;
        const float middleMinor =
            (std::min(detectedOuter.size.width, detectedOuter.size.height) +
             std::min(detectedInner.size.width, detectedInner.size.height)) *
            0.5f;
        const float majorAxisAngle = ellipseMajorAxisAngle(detectedOuter);

        result.outer = cv::RotatedRect(
            middleCenter + halfCenterSeparation,
            cv::Size2f(
                middleMajor * physicalOuterRadius / physicalMidRadius,
                middleMinor * physicalOuterRadius / physicalMidRadius),
            majorAxisAngle);
        result.inner = cv::RotatedRect(
            middleCenter - halfCenterSeparation,
            cv::Size2f(
                middleMajor * physicalInnerRadius / physicalMidRadius,
                middleMinor * physicalInnerRadius / physicalMidRadius),
            majorAxisAngle);
        result.correctedDiagnostics = validateProjectedRingPair(
            result.inner,
            result.outer,
            physicalInnerRadius,
            physicalOuterRadius);
        result.valid = result.correctedDiagnostics.valid;
        return result;
    }

    struct ProjectedBoardModel
    {
        bool valid = false;
        cv::Point2f center{0.0f, 0.0f};
        cv::Point2f centerShiftPerSquaredRadius{0.0f, 0.0f};
        float referenceRadius = 0.0f;
        float referenceMajorDiameter = 0.0f;
        float referenceMinorDiameter = 0.0f;
        float referenceAngleDegrees = 0.0f;
        const char *reason = "invalid_ring_geometry";
    };

    // Under a projective camera view, centers of concentric circle images move
    // approximately linearly with squared physical radius. Two independently
    // fitted scoring rings therefore provide a stable estimate of the true
    // board center even when a small bull color contour is fragmented.
    inline ProjectedBoardModel estimateProjectedBoardModel(
        const cv::RotatedRect &innerDouble,
        const cv::RotatedRect &outerDouble,
        const cv::RotatedRect &innerTriple,
        const cv::RotatedRect &outerTriple,
        const cv::Size &frameSize,
        float innerDoubleRadius = 162.0f,
        float outerDoubleRadius = 170.0f,
        float innerTripleRadius = 99.0f,
        float outerTripleRadius = 107.0f)
    {
        ProjectedBoardModel result;
        const float doubleRadius = (innerDoubleRadius + outerDoubleRadius) * 0.5f;
        const float tripleRadius = (innerTripleRadius + outerTripleRadius) * 0.5f;
        const float denominator = doubleRadius * doubleRadius - tripleRadius * tripleRadius;
        const cv::Point2f doubleCenter =
            (innerDouble.center + outerDouble.center) * 0.5f;
        const cv::Point2f tripleCenter =
            (innerTriple.center + outerTriple.center) * 0.5f;
        const float referenceMinorRadius =
            std::min(outerDouble.size.width, outerDouble.size.height) * 0.5f;
        if (frameSize.width <= 0 || frameSize.height <= 0 ||
            ellipseArea(innerDouble) <= 0.0f || ellipseArea(outerDouble) <= 0.0f ||
            ellipseArea(innerTriple) <= 0.0f || ellipseArea(outerTriple) <= 0.0f ||
            !std::isfinite(denominator) || denominator <= 0.0f ||
            !std::isfinite(referenceMinorRadius) || referenceMinorRadius <= 0.0f)
            return result;

        result.centerShiftPerSquaredRadius =
            (doubleCenter - tripleCenter) / denominator;
        result.center =
            tripleCenter - result.centerShiftPerSquaredRadius * (tripleRadius * tripleRadius);
        result.referenceRadius = tripleRadius;
        result.referenceMajorDiameter =
            (std::max(innerTriple.size.width, innerTriple.size.height) +
             std::max(outerTriple.size.width, outerTriple.size.height)) *
            0.5f;
        result.referenceMinorDiameter =
            (std::min(innerTriple.size.width, innerTriple.size.height) +
             std::min(outerTriple.size.width, outerTriple.size.height)) *
            0.5f;
        result.referenceAngleDegrees = ellipseMajorAxisAngle(outerTriple);

        if (!std::isfinite(result.center.x) || !std::isfinite(result.center.y) ||
            result.center.x < 0.0f || result.center.x >= frameSize.width ||
            result.center.y < 0.0f || result.center.y >= frameSize.height)
        {
            result.reason = "center_outside_frame";
            return result;
        }
        if (cv::norm(result.center - tripleCenter) > referenceMinorRadius * 0.50f)
        {
            result.reason = "extrapolation_exceeds_bound";
            return result;
        }

        result.valid = true;
        result.reason = "valid";
        return result;
    }

    inline cv::RotatedRect projectRingFromBoardModel(
        const ProjectedBoardModel &model,
        float physicalRadius)
    {
        if (!model.valid || !std::isfinite(physicalRadius) || physicalRadius <= 0.0f ||
            model.referenceRadius <= 0.0f)
            return cv::RotatedRect();

        const float scale = physicalRadius / model.referenceRadius;
        return cv::RotatedRect(
            model.center +
                model.centerShiftPerSquaredRadius * (physicalRadius * physicalRadius),
            cv::Size2f(
                model.referenceMajorDiameter * scale,
                model.referenceMinorDiameter * scale),
            model.referenceAngleDegrees);
    }

    struct BullPairDiagnostics
    {
        bool valid = false;
        float areaRatio = 0.0f;
        float centerDistancePixels = 0.0f;
        float maxCenterDistancePixels = 0.0f;
        const char *reason = "invalid_ellipse";
    };

    inline BullPairDiagnostics validateBullPair(
        const cv::RotatedRect &innerBull,
        const cv::RotatedRect &outerBull,
        const cv::RotatedRect &outerDouble,
        float maxCenterDistanceRatio = 0.10f)
    {
        BullPairDiagnostics result;
        const float innerArea = ellipseArea(innerBull);
        const float outerArea = ellipseArea(outerBull);
        const float referenceMinorRadius =
            std::min(outerDouble.size.width, outerDouble.size.height) * 0.5f;
        if (!std::isfinite(innerArea) || !std::isfinite(outerArea) ||
            !std::isfinite(referenceMinorRadius) || innerArea <= 0.0f ||
            outerArea <= 0.0f || referenceMinorRadius <= 0.0f)
            return result;

        result.areaRatio = innerArea / outerArea;
        result.centerDistancePixels = cv::norm(innerBull.center - outerBull.center);
        result.maxCenterDistancePixels = referenceMinorRadius * maxCenterDistanceRatio;
        if (result.areaRatio <= 0.03f || result.areaRatio >= 0.65f)
        {
            result.reason = "implausible_bull_size";
            return result;
        }
        if (result.centerDistancePixels > result.maxCenterDistancePixels)
        {
            result.reason = "centers_disagree";
            return result;
        }

        result.valid = true;
        result.reason = "valid";
        return result;
    }

    // Refine a coarse color-contour center from the fitted bull ellipses. The
    // outer-double minor radius provides a scale-independent displacement
    // bound. Inner bull is preferred only when its independently fitted outer
    // bull is concentric; a clipped/noisy inner-bull mask can otherwise select
    // a nearby artifact.
    inline BullCenterRefinement selectBullCenterRefinement(
        const cv::Point2f &coarseCenter,
        const cv::RotatedRect &innerBull,
        const cv::RotatedRect &outerBull,
        const cv::RotatedRect &outerDouble,
        const cv::Size &frameSize,
        float maxDisplacementRatio = 0.40f,
        float maxInnerOuterDistanceRatio = 0.10f)
    {
        BullCenterRefinement result;
        result.center = coarseCenter;

        const float minorRadius =
            std::min(outerDouble.size.width, outerDouble.size.height) * 0.5f;
        if (!std::isfinite(coarseCenter.x) || !std::isfinite(coarseCenter.y) ||
            !std::isfinite(minorRadius) || minorRadius <= 0.0f ||
            !std::isfinite(maxDisplacementRatio) || maxDisplacementRatio <= 0.0f ||
            !std::isfinite(maxInnerOuterDistanceRatio) || maxInnerOuterDistanceRatio <= 0.0f)
        {
            result.reason = "invalid_reference_geometry";
            return result;
        }

        result.maxDisplacementPixels = minorRadius * maxDisplacementRatio;
        result.maxInnerOuterDistancePixels = minorRadius * maxInnerOuterDistanceRatio;

        const bool innerUsable = hasUsableEllipseCenter(innerBull, frameSize);
        const bool outerUsable = hasUsableEllipseCenter(outerBull, frameSize);
        cv::Point2f candidate;

        if (innerUsable && outerUsable)
        {
            result.innerOuterDistancePixels = cv::norm(innerBull.center - outerBull.center);
            if (result.innerOuterDistancePixels <= result.maxInnerOuterDistancePixels)
            {
                candidate = innerBull.center;
                result.source = BullCenterSource::INNER_BULL;
                result.reason = "inner_outer_agree";
            }
            else
            {
                candidate = outerBull.center;
                result.source = BullCenterSource::OUTER_BULL;
                result.reason = "inner_outer_disagree";
            }
        }
        else if (innerUsable)
        {
            candidate = innerBull.center;
            result.source = BullCenterSource::INNER_BULL;
            result.reason = "outer_unavailable";
        }
        else if (outerUsable)
        {
            candidate = outerBull.center;
            result.source = BullCenterSource::OUTER_BULL;
            result.reason = "inner_unavailable";
        }
        else
        {
            return result;
        }

        result.center = candidate;
        result.displacementPixels = cv::norm(candidate - coarseCenter);
        if (!std::isfinite(result.displacementPixels) ||
            result.displacementPixels > result.maxDisplacementPixels)
        {
            result.reason = "displacement_exceeds_bound";
            return result;
        }

        result.accepted = true;
        return result;
    }

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

    inline float distanceToBoundaryLinePixels(
        const cv::Point2f &point,
        const cv::Point2f &center,
        const cv::Point2f &boundaryPoint)
    {
        const cv::Point2f boundary = boundaryPoint - center;
        const float boundaryLength = cv::norm(boundary);
        if (!std::isfinite(boundaryLength) || boundaryLength <= 0.0f)
            return std::numeric_limits<float>::infinity();

        const cv::Point2f relative = point - center;
        return std::fabs(relative.x * boundary.y - relative.y * boundary.x) / boundaryLength;
    }

    template <typename WireContainer>
    inline float nearestWireDistancePixels(
        const cv::Point2f &point,
        const cv::Point2f &center,
        const WireContainer &wireEndpoints)
    {
        float nearest = std::numeric_limits<float>::infinity();
        for (const auto &wire : wireEndpoints)
            nearest = std::min(nearest, distanceToBoundaryLinePixels(point, center, wire));
        return nearest;
    }

    // A single camera cannot reliably choose a side of a spider wire when the
    // detected tip is only a pixel or two from the boundary. Keep broadcasting
    // the best score candidate, but make the uncertainty explicit to clients.
    inline float singleCameraBoundaryConfidence(float distancePixels, float baseConfidence = 0.7f)
    {
        if (!std::isfinite(distancePixels))
            return baseConfidence;
        if (distancePixels <= 1.0f)
            return std::min(baseConfidence, 0.2f);
        if (distancePixels <= 2.0f)
            return std::min(baseConfidence, 0.35f);
        if (distancePixels <= 4.0f)
            return std::min(baseConfidence, 0.5f);
        return baseConfidence;
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
        float maxGapDegrees = 30.0f,
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

    struct WireLatticeRecovery
    {
        bool recovered = false;
        std::vector<cv::Point2f> wires;
        size_t supportedSlots = 0;
        size_t inferredSlots = 0;
        float phaseDegrees = 0.0f;
        float rmsResidualDegrees = 0.0f;
    };

    // Recover a complete dartboard lattice when otherwise-good detectors split
    // one or more physical wires into nearby angular groups. This never invents
    // a missing wire by default: every one of the 20 lattice slots must have an
    // observed candidate within the residual bound. A caller may explicitly
    // lower minSupportedSlots for a ring-only calibration; inferred slots are
    // reported so that orientation scoring can remain disabled.
    inline WireLatticeRecovery recoverWireLattice(
        const std::vector<cv::Point2f> &candidates,
        const cv::Point2f &center,
        const cv::RotatedRect &ellipse,
        size_t expectedCount = 20,
        size_t maxExtraCandidates = 4,
        float maxResidualDegrees = 6.0f,
        size_t minSupportedSlots = 20,
        float maxRmsResidualDegrees = 4.5f)
    {
        WireLatticeRecovery result;
        if (expectedCount < 2 || candidates.size() <= expectedCount ||
            candidates.size() > expectedCount + maxExtraCandidates ||
            minSupportedSlots > expectedCount || minSupportedSlots < 2 ||
            !std::isfinite(maxResidualDegrees) || maxResidualDegrees <= 0.0f ||
            !std::isfinite(maxRmsResidualDegrees) || maxRmsResidualDegrees <= 0.0f)
        {
            return result;
        }

        struct AngularCandidate
        {
            cv::Point2f point;
            float angle;
        };

        std::vector<AngularCandidate> angularCandidates;
        angularCandidates.reserve(candidates.size());
        for (const auto &candidate : candidates)
        {
            const cv::Point2f normalized = normalizeVector(candidate - center, ellipse);
            if (!std::isfinite(normalized.x) || !std::isfinite(normalized.y) || cv::norm(normalized) <= 0.0f)
                continue;
            angularCandidates.push_back({candidate, angleDegrees(normalized)});
        }
        if (angularCandidates.size() != candidates.size())
            return result;

        const float stepDegrees = 360.0f / static_cast<float>(expectedCount);
        constexpr float phaseIncrementDegrees = 0.1f;
        size_t bestCoverage = 0;
        float bestSquaredResidual = std::numeric_limits<float>::infinity();
        float bestPhase = 0.0f;
        std::vector<int> bestCandidateBySlot;

        for (float phase = 0.0f; phase < stepDegrees; phase += phaseIncrementDegrees)
        {
            std::vector<float> residualBySlot(expectedCount, std::numeric_limits<float>::infinity());
            std::vector<int> candidateBySlot(expectedCount, -1);

            for (size_t candidateIndex = 0; candidateIndex < angularCandidates.size(); ++candidateIndex)
            {
                const float relative = (angularCandidates[candidateIndex].angle - phase) / stepDegrees;
                int slot = static_cast<int>(std::lround(relative)) % static_cast<int>(expectedCount);
                if (slot < 0)
                    slot += static_cast<int>(expectedCount);
                float targetAngle = phase + static_cast<float>(slot) * stepDegrees;
                if (targetAngle >= 360.0f)
                    targetAngle -= 360.0f;
                const float residual = circularAngleDifference(angularCandidates[candidateIndex].angle, targetAngle);
                if (residual <= maxResidualDegrees && residual < residualBySlot[slot])
                {
                    residualBySlot[slot] = residual;
                    candidateBySlot[slot] = static_cast<int>(candidateIndex);
                }
            }

            size_t coverage = 0;
            float squaredResidual = 0.0f;
            for (size_t slot = 0; slot < expectedCount; ++slot)
            {
                if (candidateBySlot[slot] < 0)
                    continue;
                coverage++;
                squaredResidual += residualBySlot[slot] * residualBySlot[slot];
            }

            if (coverage > bestCoverage ||
                (coverage == bestCoverage && squaredResidual < bestSquaredResidual))
            {
                bestCoverage = coverage;
                bestSquaredResidual = squaredResidual;
                bestPhase = phase;
                bestCandidateBySlot = candidateBySlot;
            }
        }

        result.supportedSlots = bestCoverage;
        result.phaseDegrees = bestPhase;
        if (bestCoverage > 0 && std::isfinite(bestSquaredResidual))
            result.rmsResidualDegrees = std::sqrt(bestSquaredResidual / static_cast<float>(bestCoverage));
        result.inferredSlots = expectedCount - std::min(bestCoverage, expectedCount);
        if (bestCoverage < minSupportedSlots ||
            result.rmsResidualDegrees > maxRmsResidualDegrees ||
            bestCandidateBySlot.size() != expectedCount)
            return result;

        result.wires.reserve(expectedCount);
        for (size_t slot = 0; slot < expectedCount; ++slot)
        {
            const int candidateIndex = bestCandidateBySlot[slot];
            if (candidateIndex < 0)
            {
                const float angleRadians =
                    (bestPhase + static_cast<float>(slot) * stepDegrees) *
                    static_cast<float>(CV_PI) / 180.0f;
                result.wires.push_back(center + denormalizeVector(
                                                    cv::Point2f(
                                                        std::cos(angleRadians),
                                                        std::sin(angleRadians)),
                                                    ellipse));
                continue;
            }
            result.wires.push_back(angularCandidates[static_cast<size_t>(candidateIndex)].point);
        }

        const auto spacing = validateWireSpacing(result.wires, center, ellipse, expectedCount);
        result.recovered = spacing.valid;
        if (!result.recovered)
            result.wires.clear();
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

    struct SectorDirectionMatch
    {
        int wireIndex = -1;
        float widthDegrees = 0.0f;
        float leftMarginDegrees = 0.0f;
        float rightMarginDegrees = 0.0f;
        bool valid = false;
    };

    // Match a known board-space direction (for example the upright 20 sector)
    // to the observed wire lattice. Camera mounting clips are useful metadata,
    // but they must not be required to recover the board's absolute rotation.
    template <typename WireContainer>
    inline SectorDirectionMatch findSectorForImageDirection(
        const cv::Point2f &imageDirection,
        const WireContainer &wireEndpoints,
        const cv::Point2f &center,
        const cv::RotatedRect &outerDoubleEllipse)
    {
        SectorDirectionMatch result;
        if (wireEndpoints.size() != 20 || cv::norm(imageDirection) <= 0.0f)
            return result;

        const float targetAngle = angleDegrees(
            normalizeVector(imageDirection, outerDoubleEllipse));
        for (size_t index = 0; index < wireEndpoints.size(); ++index)
        {
            float firstAngle = normalizedAngleDegrees(
                wireEndpoints[index], center, outerDoubleEllipse);
            float secondAngle = normalizedAngleDegrees(
                wireEndpoints[(index + 1) % wireEndpoints.size()],
                center,
                outerDoubleEllipse);
            if (secondAngle <= firstAngle)
                secondAngle += 360.0f;

            float adjustedTarget = targetAngle;
            if (adjustedTarget < firstAngle)
                adjustedTarget += 360.0f;
            if (adjustedTarget < firstAngle || adjustedTarget > secondAngle)
                continue;

            result.wireIndex = static_cast<int>(index);
            result.widthDegrees = secondAngle - firstAngle;
            result.leftMarginDegrees = adjustedTarget - firstAngle;
            result.rightMarginDegrees = secondAngle - adjustedTarget;
            result.valid = result.widthDegrees >= 8.0f && result.widthDegrees <= 28.0f &&
                           result.leftMarginDegrees >= 4.0f && result.rightMarginDegrees >= 4.0f &&
                           std::fabs(result.leftMarginDegrees - result.rightMarginDegrees) <= 5.0f;
            return result;
        }
        return result;
    }
} // namespace board_geometry
