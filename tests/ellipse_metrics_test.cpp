#include "detector/geometry/detection/ellipse_metrics.hpp"

#include <cmath>
#include <iostream>

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
    const cv::Point2f center(100.0f, 60.0f);
    const cv::RotatedRect ellipse(center, cv::Size2f(40.0f, 20.0f), 30.0f);
    const double angle = 30.0 * CV_PI / 180.0;
    const cv::Point2f majorAxisUnit(
        static_cast<float>(std::cos(angle)),
        static_cast<float>(std::sin(angle)));

    passed &= require(
        std::fabs(score_processing::normalizedEllipseRadius(center, ellipse)) < 1e-6,
        "ellipse center must have normalized radius zero");
    passed &= require(
        std::fabs(score_processing::signedNormalizedEllipseMargin(center, ellipse) + 1.0) < 1e-6,
        "ellipse center must have signed margin negative one");

    const cv::Point2f boundary = center + majorAxisUnit * 20.0f;
    passed &= require(
        std::fabs(score_processing::normalizedEllipseRadius(boundary, ellipse) - 1.0) < 1e-5,
        "point on rotated major-axis boundary must have normalized radius one");
    passed &= require(
        std::fabs(score_processing::signedNormalizedEllipseMargin(boundary, ellipse)) < 1e-5,
        "point on ellipse boundary must have signed margin zero");

    const cv::Point2f outside = center + majorAxisUnit * 30.0f;
    passed &= require(
        std::fabs(score_processing::normalizedEllipseRadius(outside, ellipse) - 1.5) < 1e-5,
        "outside point must report its normalized radial distance");
    passed &= require(
        score_processing::signedNormalizedEllipseMargin(outside, ellipse) > 0.0,
        "outside point must have a positive signed margin");

    const cv::RotatedRect invalid(center, cv::Size2f(0.0f, 20.0f), 0.0f);
    passed &= require(
        std::isnan(score_processing::normalizedEllipseRadius(center, invalid)),
        "invalid ellipse must return NaN");

    return passed ? 0 : 1;
}
