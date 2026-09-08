#include "detector/geometry/calibration/board_geometry.hpp"

#include <cmath>
#include <iostream>
#include <string>
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

    std::vector<cv::Point2f> splitWireCandidates = wires;
    splitWireCandidates[7] = pointAtNormalizedAngle(4.0f + 7.0f * 18.0f - 5.0f, center, obliqueEllipse);
    splitWireCandidates.push_back(pointAtNormalizedAngle(4.0f + 7.0f * 18.0f + 5.0f, center, obliqueEllipse));
    const auto recoveredSplitWire = board_geometry::recoverWireLattice(
        splitWireCandidates, center, obliqueEllipse);
    passed &= require(recoveredSplitWire.recovered && recoveredSplitWire.wires.size() == 20,
                      "a split boundary with all 20 observed slots must recover");
    passed &= require(board_geometry::validateWireSpacing(
                           recoveredSplitWire.wires, center, obliqueEllipse)
                           .valid,
                      "recovered boundary candidates must pass independent spacing validation");

    std::vector<cv::Point2f> missingSlotCandidates = wires;
    missingSlotCandidates.erase(missingSlotCandidates.begin() + 7);
    missingSlotCandidates.push_back(pointAtNormalizedAngle(4.0f + 2.0f * 18.0f + 4.5f, center, obliqueEllipse));
    missingSlotCandidates.push_back(pointAtNormalizedAngle(4.0f + 12.0f * 18.0f - 4.5f, center, obliqueEllipse));
    passed &= require(!board_geometry::recoverWireLattice(
                           missingSlotCandidates, center, obliqueEllipse)
                           .recovered,
                      "lattice recovery must reject a genuinely unobserved wire slot");

    const auto inferredMissingSlot = board_geometry::recoverWireLattice(
        missingSlotCandidates,
        center,
        obliqueEllipse,
        20,
        4,
        6.0f,
        17,
        4.5f);
    passed &= require(inferredMissingSlot.recovered && inferredMissingSlot.inferredSlots == 1,
                      "ring-only recovery may explicitly infer one unsupported lattice slot");
    passed &= require(board_geometry::validateWireSpacing(
                           inferredMissingSlot.wires, center, obliqueEllipse)
                           .valid,
                      "ring-only inferred lattice must still pass spacing validation");

    std::vector<cv::Point2f> insufficientCandidates(
        wires.begin(), wires.begin() + 16);
    insufficientCandidates.insert(
        insufficientCandidates.end(),
        duplicatedCandidates.begin(),
        duplicatedCandidates.begin() + 6);
    passed &= require(!board_geometry::recoverWireLattice(
                           insufficientCandidates,
                           center,
                           obliqueEllipse,
                           20,
                           4,
                           6.0f,
                           17,
                           4.5f)
                           .recovered,
                      "ring-only recovery must reject fewer than 17 supported slots");

    std::vector<cv::Point2f> cameraTwoLikeWires;
    float accumulatedAngle = 0.0f;
    cameraTwoLikeWires.push_back(pointAtNormalizedAngle(accumulatedAngle, center, obliqueEllipse));
    accumulatedAngle += 28.2f;
    for (int index = 1; index < 20; ++index)
    {
        cameraTwoLikeWires.push_back(pointAtNormalizedAngle(accumulatedAngle, center, obliqueEllipse));
        accumulatedAngle += (360.0f - 28.2f) / 19.0f;
    }
    passed &= require(board_geometry::validateWireSpacing(
                           cameraTwoLikeWires, center, obliqueEllipse)
                           .valid,
                      "a noisy 28.2 degree contour gap must remain recoverable");

    passed &= require(std::fabs(board_geometry::nearestWireDistancePixels(
                                    center + cv::Point2f(100.0f, 1.0f),
                                    center,
                                    std::vector<cv::Point2f>{
                                        center + cv::Point2f(200.0f, 0.0f)}) -
                                1.0f) < 0.001f,
                      "wire proximity must be measured perpendicular to the boundary");
    passed &= require(board_geometry::singleCameraBoundaryConfidence(0.5f) == 0.2f,
                      "sub-pixel wire decisions must be low confidence");
    passed &= require(board_geometry::singleCameraBoundaryConfidence(5.0f) == 0.7f,
                      "well-separated single-camera scores must retain base confidence");

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

    const cv::RotatedRect uprightEllipse(center, cv::Size2f(600.0f, 300.0f), 0.0f);
    std::vector<cv::Point2f> uprightWires;
    for (int index = 0; index < 20; ++index)
    {
        // Boundaries at -99 and -81 degrees put image north in the center of
        // the upright 20 sector.
        uprightWires.push_back(pointAtNormalizedAngle(
            -99.0f + static_cast<float>(index) * 18.0f,
            center,
            uprightEllipse));
    }
    const auto uprightTwenty = board_geometry::findSectorForImageDirection(
        cv::Point2f(0.0f, -1.0f), uprightWires, center, uprightEllipse);
    passed &= require(uprightTwenty.valid && uprightTwenty.wireIndex == 0,
                      "upright 20 must be recoverable without camera clip metadata");

    std::vector<cv::Point2f> offCenterWires = uprightWires;
    offCenterWires[0] = pointAtNormalizedAngle(-93.0f, center, uprightEllipse);
    const auto offCenterTwenty = board_geometry::findSectorForImageDirection(
        cv::Point2f(0.0f, -1.0f), offCenterWires, center, uprightEllipse);
    passed &= require(!offCenterTwenty.valid,
                      "a direction too close to a scoring wire must not establish orientation");

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

    const cv::RotatedRect plausibleOuterTriple(
        cv::Point2f(620.0f, 430.0f), cv::Size2f(410.0f, 176.0f), 88.0f);
    const cv::RotatedRect plausibleInnerTriple(
        cv::Point2f(620.5f, 432.0f), cv::Size2f(379.0f, 163.0f), 88.5f);
    const auto plausibleTriplePair = board_geometry::validateProjectedRingPair(
        plausibleInnerTriple, plausibleOuterTriple, 99.0f, 107.0f);
    passed &= require(plausibleTriplePair.valid,
                      "a nested projected triple ring near the standard width must validate");

    const cv::RotatedRect dilationInflatedInnerTriple(
        cv::Point2f(620.5f, 432.0f), cv::Size2f(350.0f, 154.0f), 88.5f);
    const auto dilationInflatedTriplePair = board_geometry::validateProjectedRingPair(
        dilationInflatedInnerTriple, plausibleOuterTriple, 99.0f, 107.0f);
    passed &= require(!dilationInflatedTriplePair.valid &&
                          std::string(dilationInflatedTriplePair.reason) == "implausible_ring_width",
                      "a morphology-inflated triple band must not be reported ready");
    const auto correctedInflatedTriplePair = board_geometry::correctProjectedRingPair(
        dilationInflatedInnerTriple, plausibleOuterTriple, 99.0f, 107.0f);
    passed &= require(correctedInflatedTriplePair.valid,
                      "a connected morphology band must collapse to physical wire spacing");
    passed &= require(
        std::fabs(
            correctedInflatedTriplePair.correctedDiagnostics.areaRatio -
            correctedInflatedTriplePair.correctedDiagnostics.expectedAreaRatio) <
            0.001f,
        "corrected triple geometry must use the official inner/outer radius ratio");
    passed &= require(
        board_geometry::ellipseArea(correctedInflatedTriplePair.outer) <
            board_geometry::ellipseArea(plausibleOuterTriple) &&
            board_geometry::ellipseArea(correctedInflatedTriplePair.inner) >
                board_geometry::ellipseArea(dilationInflatedInnerTriple),
        "triple correction must move both inflated mask edges toward the centerline");

    const cv::RotatedRect displacedInnerTriple(
        cv::Point2f(620.0f, 475.0f), cv::Size2f(379.0f, 163.0f), 88.5f);
    passed &= require(!board_geometry::validateProjectedRingPair(
                           displacedInnerTriple,
                           plausibleOuterTriple,
                           99.0f,
                           107.0f)
                           .valid,
                      "triple boundaries with implausibly separated centers must fail");

    const cv::Point2f projectedBullCenter(622.0f, 448.0f);
    const cv::Point2f perspectiveShift(0.0001f, -0.0020f);
    const auto centerAtRadius = [&](float radius) {
        return projectedBullCenter + perspectiveShift * (radius * radius);
    };
    const auto projectedBoard = board_geometry::estimateProjectedBoardModel(
        cv::RotatedRect(centerAtRadius(162.0f), cv::Size2f(630.0f, 278.0f), 88.0f),
        cv::RotatedRect(centerAtRadius(170.0f), cv::Size2f(660.0f, 292.0f), 88.0f),
        cv::RotatedRect(centerAtRadius(99.0f), cv::Size2f(385.0f, 170.0f), 88.0f),
        cv::RotatedRect(centerAtRadius(107.0f), cv::Size2f(416.0f, 184.0f), 88.0f),
        frameSize);
    passed &= require(projectedBoard.valid,
                      "double and triple ring centers must define a projected board model");
    passed &= require(cv::norm(projectedBoard.center - projectedBullCenter) < 0.2f,
                      "projected ring centers must recover the physical bull center");
    const auto modeledInnerBull = board_geometry::projectRingFromBoardModel(
        projectedBoard, 6.35f);
    const auto modeledOuterBull = board_geometry::projectRingFromBoardModel(
        projectedBoard, 15.9f);
    passed &= require(board_geometry::validateBullPair(
                           modeledInnerBull,
                           modeledOuterBull,
                           outerDouble)
                           .valid,
                      "modeled bull rings must form a valid concentric pair");

    const auto plausibleBullPair = board_geometry::validateBullPair(
        cv::RotatedRect(cv::Point2f(622.0f, 448.0f), cv::Size2f(28.0f, 13.0f), 88.0f),
        cv::RotatedRect(cv::Point2f(622.5f, 448.5f), cv::Size2f(65.0f, 31.0f), 88.0f),
        outerDouble);
    passed &= require(plausibleBullPair.valid,
                      "concentric inner and outer bull contours must validate");

    const auto falseBullPair = board_geometry::validateBullPair(
        cv::RotatedRect(cv::Point2f(565.0f, 420.0f), cv::Size2f(28.0f, 13.0f), 88.0f),
        cv::RotatedRect(cv::Point2f(622.5f, 448.5f), cv::Size2f(65.0f, 31.0f), 88.0f),
        outerDouble);
    passed &= require(!falseBullPair.valid,
                      "a stray bull contour must not make calibration ready");

    return passed ? 0 : 1;
}
