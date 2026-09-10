#include "detector/geometry/detection/dart_geometry.hpp"

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
}

int main()
{
    bool passed = true;
    const cv::Point2f flightCenter(100.0f, 650.0f);
    const cv::Point2f boardCenter(130.0f, 360.0f);
    const std::vector<cv::Point> hull = {
        cv::Point(70, 710),
        cv::Point(130, 710),
        cv::Point(125, 620),
        cv::Point(112, 300),
        cv::Point(90, 620)};

    const auto selection = dart_geometry::selectBoardwardHullPoint(
        hull, flightCenter, boardCenter);
    passed &= require(selection.valid, "boardward dart endpoint must be selected");
    passed &= require(selection.point == cv::Point2f(112.0f, 300.0f),
                      "boardward selection must prefer the embedded tip over the frame-edge flight");

    const cv::RotatedRect boardEllipse(
        boardCenter, cv::Size2f(400.0f, 600.0f), 0.0f);
    passed &= require(dart_geometry::isPlausibleBoardPoint(
                           selection.point, boardCenter, boardEllipse),
                      "a detected tip on the board must be plausible");
    passed &= require(!dart_geometry::isPlausibleBoardPoint(
                           cv::Point2f(100.0f, 900.0f), boardCenter, boardEllipse),
                      "a frame-edge flight beyond the surround must be rejected");

    const std::vector<cv::Point> primaryDart = {
        cv::Point(455, 520), cv::Point(485, 520),
        cv::Point(485, 615), cv::Point(455, 615)};
    const std::vector<cv::Point> unrelatedReflection = {
        cv::Point(920, 500), cv::Point(955, 500),
        cv::Point(955, 535), cv::Point(920, 535)};
    const auto isolatedPoints = dart_geometry::collectBoardwardDartPoints(
        {primaryDart, unrelatedReflection}, cv::Point2f(601.0f, 456.0f));
    passed &= require(isolatedPoints.size() == primaryDart.size(),
                      "unrelated mask islands must not be combined with the dart hull");

    const std::vector<cv::Point> flight = {
        cv::Point(550, 500), cv::Point(610, 500),
        cv::Point(610, 625), cv::Point(550, 625)};
    const std::vector<cv::Point> alignedShaft = {
        cv::Point(578, 315), cv::Point(598, 315),
        cv::Point(598, 455), cv::Point(578, 455)};
    const auto fragmentedDartPoints = dart_geometry::collectBoardwardDartPoints(
        {flight, alignedShaft}, cv::Point2f(623.0f, 447.0f));
    passed &= require(fragmentedDartPoints.size() == flight.size() + alignedShaft.size(),
                      "an aligned shaft fragment closer to the board must remain part of the dart");
    std::vector<cv::Point> fragmentedHull;
    cv::convexHull(fragmentedDartPoints, fragmentedHull);
    const auto fragmentedSelection = dart_geometry::selectBoardwardHullPoint(
        fragmentedHull, cv::Point2f(580.0f, 562.5f), cv::Point2f(623.0f, 447.0f));
    passed &= require(fragmentedSelection.valid && fragmentedSelection.point.y == 315.0f,
                      "the aligned boardward fragment must supply the detected tip");

    const std::vector<cv::Point> shaftAwayFromBull = {
        cv::Point(646, 309), cv::Point(668, 309),
        cv::Point(668, 426), cv::Point(646, 426)};
    const std::vector<cv::Point> flightNearBull = {
        cv::Point(597, 431), cv::Point(708, 431),
        cv::Point(708, 545), cv::Point(597, 545)};
    const auto awayFacingDartPoints = dart_geometry::collectBoardwardDartPoints(
        {flightNearBull, shaftAwayFromBull}, cv::Point2f(623.0f, 447.0f));
    passed &= require(
        awayFacingDartPoints.size() == flightNearBull.size() + shaftAwayFromBull.size(),
        "an aligned shaft farther from the bull must remain part of the dart");
    const auto taperedSelection = dart_geometry::selectTaperedDartEndpoint(
        awayFacingDartPoints, cv::Point2f(654.0f, 489.0f), cv::Point2f(623.0f, 447.0f));
    passed &= require(
        taperedSelection.valid && taperedSelection.point.y == 309.0f,
        "the narrow shaft end must beat the wide flight even when it points away from the bull");

    const auto reflectionStillExcluded = dart_geometry::collectBoardwardDartPoints(
        {flightNearBull, unrelatedReflection}, cv::Point2f(623.0f, 447.0f));
    passed &= require(
        reflectionStillExcluded.size() == flightNearBull.size(),
        "a compact off-axis reflection must stay excluded by bidirectional shaft matching");

    passed &= require(!dart_geometry::hasSufficientDartEvidence(1, 1, 1, false),
                      "one uncorroborated edge tip must not consume a dart");
    passed &= require(dart_geometry::hasSufficientDartEvidence(2, 2, 2, false),
                      "two independent camera tips must confirm a dart or miss");
    passed &= require(dart_geometry::hasSufficientDartEvidence(1, 1, 1, true),
                      "one oriented in-board tip may confirm a numbered dart");
    passed &= require(dart_geometry::hasSufficientDartEvidence(1, 1, 2, false),
                      "one edge-on tip plus a second persistent view must confirm a miss");
    passed &= require(!dart_geometry::hasSufficientMissEvidence(1, 1),
                      "one uncorroborated camera must not create a phantom miss");
    passed &= require(dart_geometry::hasSufficientMissEvidence(2, 2),
                      "two cameras may confirm a real miss");
    passed &= require(dart_geometry::hasSufficientMissEvidence(1, 2),
                      "one tip plus two persistent camera changes may confirm a real miss");

    return passed ? 0 : 1;
}
