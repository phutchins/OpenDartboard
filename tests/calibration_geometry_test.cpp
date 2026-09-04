#include "detector/geometry/calibration/board_geometry.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace
{
    bool require(bool condition, const char *message)
    {
        if (!condition)
            std::cerr << "FAILED: " << message << std::endl;
        return condition;
    }

    cv::Point2f pointAtNormalizedAngle(
        float angleDegrees,
        const cv::Point2f &center,
        const cv::RotatedRect &ellipse)
    {
        const float angle = angleDegrees * static_cast<float>(CV_PI) / 180.0f;
        return center + board_geometry::denormalizeVector(
                            cv::Point2f(std::cos(angle), std::sin(angle)), ellipse);
    }
}

int main()
{
    bool passed = true;
    const cv::Point2f center(640.0f, 360.0f);
    const cv::RotatedRect obliqueEllipse(center, cv::Size2f(720.0f, 210.0f), 27.0f);

    std::vector<cv::Point2f> wires;
    std::vector<cv::Point2f> duplicatedCandidates;
    for (int index = 0; index < 20; ++index)
    {
        const float angle = static_cast<float>(index) * 18.0f + 4.0f;
        wires.push_back(pointAtNormalizedAngle(angle, center, obliqueEllipse));
        duplicatedCandidates.push_back(pointAtNormalizedAngle(angle - 2.0f, center, obliqueEllipse));
        duplicatedCandidates.push_back(pointAtNormalizedAngle(angle + 2.0f, center, obliqueEllipse));
    }

    const auto spacing = board_geometry::validateWireSpacing(wires, center, obliqueEllipse);
    passed &= require(spacing.valid, "regular wires must validate after ellipse normalization");
    passed &= require(std::fabs(spacing.minGapDegrees - 18.0f) < 0.01f,
                      "normalized minimum gap must be 18 degrees");
    passed &= require(std::fabs(spacing.maxGapDegrees - 18.0f) < 0.01f,
                      "normalized maximum gap must be 18 degrees");

    const auto groups = board_geometry::groupByNormalizedAngle(
        duplicatedCandidates, center, obliqueEllipse, 8.0f);
    passed &= require(groups.size() == 20,
                      "two detections of each wire must form 20 normalized groups");

    std::vector<cv::Point2f> malformedWires = wires;
    malformedWires[1] = malformedWires[0];
    passed &= require(!board_geometry::validateWireSpacing(
                           malformedWires, center, obliqueEllipse)
                           .valid,
                      "duplicate and missing boundaries must fail validation");

    const auto balanced = board_geometry::analyzeClipLayout(
        {center + cv::Point2f(-250.0f, -160.0f), center + cv::Point2f(250.0f, -160.0f)},
        center);
    passed &= require(balanced.layout == board_geometry::ClipLayout::BALANCED,
                      "symmetric clip candidates must identify a centered view");

    const auto leftBiased = board_geometry::analyzeClipLayout(
        {center + cv::Point2f(-250.0f, -160.0f),
         center + cv::Point2f(-150.0f, -210.0f),
         center + cv::Point2f(100.0f, -210.0f)},
        center);
    passed &= require(leftBiased.layout == board_geometry::ClipLayout::LEFT_BIASED,
                      "a clear left clip bias must not be called centered");

    const auto ambiguous = board_geometry::analyzeClipLayout(
        {center + cv::Point2f(-100.0f, -200.0f), center + cv::Point2f(220.0f, -200.0f)},
        center);
    passed &= require(ambiguous.layout == board_geometry::ClipLayout::UNKNOWN,
                      "an intermediate clip balance must remain unknown");

    return passed ? 0 : 1;
}
