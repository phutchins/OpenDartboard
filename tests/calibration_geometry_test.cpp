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

    const cv::Size frameSize(1280, 720);
    const cv::RotatedRect outerDouble(center, cv::Size2f(600.0f, 300.0f), 0.0f);

    const auto agreeingBulls = board_geometry::selectBullCenterRefinement(
        cv::Point2f(623.0f, 422.0f),
        cv::RotatedRect(cv::Point2f(624.0f, 447.0f), cv::Size2f(30.0f, 16.0f), 0.0f),
        cv::RotatedRect(cv::Point2f(624.5f, 447.5f), cv::Size2f(62.0f, 32.0f), 0.0f),
        outerDouble,
        frameSize);
    passed &= require(agreeingBulls.accepted,
                      "concentric fitted bull ellipses must refine the coarse center");
    passed &= require(agreeingBulls.source == board_geometry::BullCenterSource::INNER_BULL,
                      "inner bull must be preferred when fitted bull centers agree");

    const auto disagreeingBulls = board_geometry::selectBullCenterRefinement(
        cv::Point2f(598.0f, 408.0f),
        cv::RotatedRect(cv::Point2f(563.0f, 425.0f), cv::Size2f(30.0f, 27.0f), 0.0f),
        cv::RotatedRect(cv::Point2f(604.0f, 452.0f), cv::Size2f(68.0f, 33.0f), 0.0f),
        outerDouble,
        frameSize);
    passed &= require(disagreeingBulls.accepted,
                      "bounded outer bull must recover from a false inner-bull component");
    passed &= require(disagreeingBulls.source == board_geometry::BullCenterSource::OUTER_BULL,
                      "outer bull must win when fitted bull centers disagree");

    const auto outerOnly = board_geometry::selectBullCenterRefinement(
        cv::Point2f(616.0f, 404.0f),
        cv::RotatedRect(),
        cv::RotatedRect(cv::Point2f(622.0f, 460.0f), cv::Size2f(64.0f, 35.0f), 0.0f),
        outerDouble,
        frameSize);
    passed &= require(outerOnly.accepted &&
                          outerOnly.source == board_geometry::BullCenterSource::OUTER_BULL,
                      "outer bull must be a bounded fallback when inner bull is unavailable");

    const auto excessiveShift = board_geometry::selectBullCenterRefinement(
        cv::Point2f(620.0f, 360.0f),
        cv::RotatedRect(),
        cv::RotatedRect(cv::Point2f(620.0f, 425.0f), cv::Size2f(64.0f, 35.0f), 0.0f),
        outerDouble,
        frameSize);
    passed &= require(!excessiveShift.accepted,
                      "bull refinement beyond 40 percent of minor radius must be rejected");

    return passed ? 0 : 1;
}
