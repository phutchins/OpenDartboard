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

    passed &= require(!dart_geometry::hasSufficientDartEvidence(1, false),
                      "one unverified edge tip must not consume a dart");
    passed &= require(dart_geometry::hasSufficientDartEvidence(2, false),
                      "two independent camera tips must confirm a dart or miss");
    passed &= require(dart_geometry::hasSufficientDartEvidence(1, true),
                      "one oriented in-board tip may confirm a numbered dart");
    passed &= require(!dart_geometry::hasSufficientMissEvidence(1),
                      "one camera must not create a phantom miss");
    passed &= require(dart_geometry::hasSufficientMissEvidence(2),
                      "two cameras may confirm a real miss");

    return passed ? 0 : 1;
}
