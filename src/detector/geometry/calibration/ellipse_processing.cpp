#include "ellipse_processing.hpp"
#include "board_geometry.hpp"
#include "utils.hpp"
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

namespace ellipse_processing
{
    // Rolling baseline ring width validation
    vector<Point> performDoubleRayTrace(const Mat &preprocessedMask, const Point &bullCenter, const EllipseParams &params,
                                        vector<Point> &allInnerPoints, vector<Point> &allOuterPoints, vector<bool> &rayValidFlags)
    {
        vector<double> ringWidths;
        allInnerPoints.clear();
        allOuterPoints.clear();
        rayValidFlags.clear();

        log_debug("Starting double ray trace from (" + log_string(bullCenter.x) + "," + log_string(bullCenter.y) + ")");

        // PHASE 1: Cast all rays and measure inner/outer boundaries + ring widths
        for (double angle = 0; angle < 360; angle += params.angleStepDegrees)
        {
            double radians = angle * CV_PI / 180.0;
            Point2f direction(cos(radians), sin(radians));

            // Scan every colored run and retain the outermost complete one.
            // The mask contains both scoring rings; choosing the first white
            // pixel incorrectly fits the triple ring as doubles whenever the
            // outer ring is not the largest connected component.
            Point innerBoundary = bullCenter;
            Point outerBoundary = bullCenter;
            bool foundInner = false;
            bool foundOuter = false;
            bool inWhiteRun = false;
            Point runInner = bullCenter;
            Point lastWhite = bullCenter;
            int consecutiveBlackCount = 0;
            const int minConsecutiveBlack = 5;

            const auto acceptRun = [&]()
            {
                if (inWhiteRun && lastWhite != runInner)
                {
                    innerBoundary = runInner;
                    outerBoundary = lastWhite;
                    foundInner = true;
                    foundOuter = true;
                }
            };

            for (int distance = params.innerRayStartDistance; distance < params.maxRayDistance; distance++)
            {
                Point checkPoint = bullCenter + Point(direction.x * distance, direction.y * distance);
                if (checkPoint.x < 0 || checkPoint.x >= preprocessedMask.cols ||
                    checkPoint.y < 0 || checkPoint.y >= preprocessedMask.rows)
                {
                    acceptRun();
                    break;
                }

                if (preprocessedMask.at<uchar>(checkPoint) > 0)
                {
                    if (!inWhiteRun)
                    {
                        runInner = checkPoint;
                        inWhiteRun = true;
                    }
                    lastWhite = checkPoint;
                    consecutiveBlackCount = 0;
                }
                else if (inWhiteRun)
                {
                    consecutiveBlackCount++;
                    if (consecutiveBlackCount >= minConsecutiveBlack)
                    {
                        acceptRun();
                        inWhiteRun = false;
                        consecutiveBlackCount = 0;
                    }
                }
            }
            acceptRun();

            // Calculate ring width
            bool validRay = foundInner && foundOuter;
            double ringWidth = validRay ? norm(outerBoundary - innerBoundary) : -1;

            // Store all data
            allInnerPoints.push_back(innerBoundary);
            allOuterPoints.push_back(outerBoundary);
            rayValidFlags.push_back(validRay);

            if (validRay)
            {
                ringWidths.push_back(ringWidth);
            }
            else
            {
                ringWidths.push_back(-1);
            }
        }

        log_debug("Double ray trace completed. Found " + log_string(count(rayValidFlags.begin(), rayValidFlags.end(), true)) + " valid rays");

        // PHASE 2: Rolling baseline validation
        vector<Point> validatedOuterPoints;
        vector<bool> finalValidFlags(rayValidFlags.size(), false);

        if (count(rayValidFlags.begin(), rayValidFlags.end(), true) >= params.minValidRingMeasurements)
        {

            // Create rolling baseline of ring widths
            vector<double> rollingBaseline;
            vector<double> validRingWidths;

            // Collect all valid ring widths
            for (size_t i = 0; i < rayValidFlags.size(); i++)
            {
                if (rayValidFlags[i] && ringWidths[i] > 0)
                {
                    validRingWidths.push_back(ringWidths[i]);
                }
            }

            if (validRingWidths.size() < params.minValidRingMeasurements)
            {
                log_debug("Not enough valid ring widths for rolling baseline");
                return vector<Point>();
            }

            // Calculate global statistics for outlier detection
            sort(validRingWidths.begin(), validRingWidths.end());
            double globalMedian = validRingWidths[validRingWidths.size() / 2];

            // Calculate standard deviation
            double sum = 0;
            for (double width : validRingWidths)
            {
                sum += (width - globalMedian) * (width - globalMedian);
            }
            double stdDev = sqrt(sum / validRingWidths.size());

            log_debug("Global ring width stats - median: " + log_string(globalMedian) + ", stdDev: " + log_string(stdDev));

            // Rolling validation - go around the circle
            double rollingExpected = globalMedian; // Start with global median
            int validCount = 0;

            for (size_t i = 0; i < rayValidFlags.size(); i++)
            {
                if (!rayValidFlags[i] || ringWidths[i] <= 0)
                    continue;

                double currentWidth = ringWidths[i];
                double angle = i * params.angleStepDegrees;

                // Check against rolling expected value
                double deviation = abs(currentWidth - rollingExpected);
                double stdDeviation = abs(currentWidth - globalMedian) / stdDev;

                bool passesJumpTest = (deviation <= params.maxRingWidthJump);
                bool passesOutlierTest = (stdDeviation <= params.ringWidthOutlierThreshold);

                // temporary removed to not clutter the output
                // cout << "DEBUG: Ray " << angle << "° - width=" << currentWidth
                //      << ", expected=" << rollingExpected << ", deviation=" << deviation
                //      << ", stdDev=" << stdDeviation << " - ";

                if (passesJumpTest && passesOutlierTest)
                {
                    // ACCEPT this ray
                    validatedOuterPoints.push_back(allOuterPoints[i]);
                    finalValidFlags[i] = true;
                    validCount++;

                    // Update rolling expected (smooth transition)
                    double alpha = 0.3; // Smoothing factor
                    rollingExpected = alpha * currentWidth + (1.0 - alpha) * rollingExpected;

                    // temporary removed to not clutter the output
                    // cout << "ACCEPTED (newExpected=" << rollingExpected << ")" << endl;
                }
                else
                {
                    // temporary removed to not clutter the output
                    // cout << "REJECTED (jump=" << !passesJumpTest
                    //      << ", outlier=" << !passesOutlierTest << ")" << endl;
                }
            }

            // Update rayValidFlags to reflect final validation
            rayValidFlags = finalValidFlags;

            log_debug("Rolling baseline validation result: " + log_string(validatedOuterPoints.size()) + " validated rays from " + log_string(allOuterPoints.size()) + " total rays");

            return validatedOuterPoints;
        }

        log_debug("Insufficient ring width measurements, using all valid rays");
        vector<Point> allValidOuterPoints;
        for (size_t i = 0; i < rayValidFlags.size(); i++)
        {
            if (rayValidFlags[i])
            {
                allValidOuterPoints.push_back(allOuterPoints[i]);
            }
        }
        return allValidOuterPoints;
    }

    // Main function uses MaskBundle and proper terminology
    EllipseBoundaryData processEllipse(
        const Mat &originalFrame,
        const mask_processing::MaskBundle &masks,
        const Point &bullCenter,
        const Point &frameCenter,
        int camera_idx,
        bool debug_mode,
        const EllipseParams &params)
    {
        log_debug("Ellipse processing camera " + log_string(camera_idx) + " starting...");
        EllipseBoundaryData result;

        // SECTION 1: RAY TRACING for DOUBLES (most accurate method)
        if (!masks.doublesMask.empty())
        {
            log_debug("Processing doubles ring with ray tracing...");

            // No preprocessing needed - mask is already clean!
            int whitePixels = countNonZero(masks.doublesMask);
            log_debug("Doubles mask has " + log_string(whitePixels) + " white pixels");

            vector<Point> allInnerPoints, allOuterPoints;
            vector<bool> rayValidFlags;
            vector<Point> finalBoundaryPoints;

            if (whitePixels >= params.minWhitePixelsThreshold)
            {
                // Perform ray tracing on clean doubles mask
                finalBoundaryPoints = performDoubleRayTrace(masks.doublesMask, bullCenter, params,
                                                            allInnerPoints, allOuterPoints, rayValidFlags);

                if (finalBoundaryPoints.size() >= params.minValidRays)
                {
                    try
                    {
                        // Fit ellipse to validated outer boundary points
                        result.outerDoubleEllipse = fitEllipse(finalBoundaryPoints);
                        result.validOuterPoints = finalBoundaryPoints.size();
                        log_debug("SUCCESS - Fitted outer double ellipse from " + log_string(result.validOuterPoints) + " boundary points");

                        // Fit ellipse to inner boundary points (validated ones only)
                        vector<Point> validatedInnerPoints;
                        set<pair<int, int>> acceptedOuterPoints;
                        for (const Point &pt : finalBoundaryPoints)
                        {
                            acceptedOuterPoints.insert({pt.x, pt.y});
                        }

                        for (size_t i = 0; i < allOuterPoints.size(); i++)
                        {
                            if (acceptedOuterPoints.count({allOuterPoints[i].x, allOuterPoints[i].y}) > 0 &&
                                i < allInnerPoints.size())
                            {
                                validatedInnerPoints.push_back(allInnerPoints[i]);
                            }
                        }

                        if (validatedInnerPoints.size() >= 5)
                        {
                            result.innerDoubleEllipse = fitEllipse(validatedInnerPoints);
                            result.validInnerPoints = validatedInnerPoints.size();
                            log_debug("SUCCESS - Fitted inner double ellipse from " + log_string(result.validInnerPoints) + " inner points");
                        }
                        else
                        {
                            // Fallback: scale outer ellipse for inner boundary
                            result.innerDoubleEllipse = result.outerDoubleEllipse;
                            result.innerDoubleEllipse.size.width *= 0.92;
                            result.innerDoubleEllipse.size.height *= 0.92;
                            result.validInnerPoints = 0;
                            log_debug("FALLBACK - Calculated inner double ellipse from outer");
                        }

                        result.hasValidDoubles = true;
                    }
                    catch (const cv::Exception &e)
                    {
                        log_debug("Double ellipse fitting failed: " + string(e.what()));
                        result.hasValidDoubles = false;
                    }
                }
                else
                {
                    log_debug("Not enough boundary points for doubles: " + log_string(finalBoundaryPoints.size()));
                    result.hasValidDoubles = false;
                }
            }
        }

        // SECTION 2: CONTOUR FITTING for TRIPLES (efficient method)
        if (!masks.triplesMask.empty())
        {
            log_debug("Processing triples ring with contour fitting...");

            // Find ALL contours
            vector<vector<Point>> allTriplesContours;
            findContours(masks.triplesMask, allTriplesContours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            log_debug("Found " + log_string(allTriplesContours.size()) + " total contours");

            // Sort by area (largest first)
            sort(allTriplesContours.begin(), allTriplesContours.end(),
                 [](const vector<Point> &a, const vector<Point> &b)
                 {
                     return contourArea(a) > contourArea(b);
                 });

            // Debug contour areas
            for (size_t i = 0; i < allTriplesContours.size(); i++)
            {
                log_debug("Contour " + log_string(i) + " area: " + log_string(contourArea(allTriplesContours[i])));
            }

            try
            {
                // Evaluate contour pairs instead of assuming that the two
                // largest contours are physical wire boundaries. A valid pair
                // must have the width, nesting, center, and perspective shape
                // of the standard 99/107 mm triple ring.
                double bestQuality = numeric_limits<double>::infinity();
                board_geometry::RingPairDiagnostics bestDiagnostics;
                for (size_t outerIndex = 0; outerIndex < allTriplesContours.size(); ++outerIndex)
                {
                    if (allTriplesContours[outerIndex].size() < 5 ||
                        contourArea(allTriplesContours[outerIndex]) < params.minContourArea)
                        continue;

                    const RotatedRect outerCandidate = fitEllipse(allTriplesContours[outerIndex]);
                    const float relativeToOuterDouble =
                        board_geometry::ellipseArea(outerCandidate) /
                        board_geometry::ellipseArea(result.outerDoubleEllipse);
                    // The triple outer radius is 107/170 of the double outer
                    // radius. Broad bounds allow perspective while excluding
                    // the doubles contours that share this all-rings mask.
                    if (!isfinite(relativeToOuterDouble) ||
                        relativeToOuterDouble < 0.20f ||
                        relativeToOuterDouble > 0.65f)
                        continue;

                    for (size_t innerIndex = outerIndex + 1; innerIndex < allTriplesContours.size(); ++innerIndex)
                    {
                        if (allTriplesContours[innerIndex].size() < 5 ||
                            contourArea(allTriplesContours[innerIndex]) < params.minContourArea)
                            continue;

                        const RotatedRect innerCandidate = fitEllipse(allTriplesContours[innerIndex]);
                        const auto correction = board_geometry::correctProjectedRingPair(
                            innerCandidate,
                            outerCandidate,
                            99.0f,
                            107.0f);
                        const auto &diagnostics = correction.sourceDiagnostics;
                        log_debug(
                            "TRIPLE_RING_CANDIDATE outer=" + log_string(outerIndex) +
                            " inner=" + log_string(innerIndex) +
                            " valid=" + (correction.valid ? string("true") : string("false")) +
                            " detected_area_ratio=" + log_string(diagnostics.areaRatio) +
                            " outer_to_double_area_ratio=" + log_string(relativeToOuterDouble) +
                            " expected_area_ratio=" + log_string(diagnostics.expectedAreaRatio) +
                            " center_offset_ratio=" + log_string(diagnostics.centerOffsetRatio) +
                            " aspect_difference=" + log_string(diagnostics.aspectRatioDifference) +
                            " axis_difference_degrees=" + log_string(diagnostics.majorAxisAngleDifferenceDegrees) +
                            " reason=" + diagnostics.reason);
                        if (!correction.valid)
                            continue;

                        const double quality =
                            abs(diagnostics.areaRatio - diagnostics.expectedAreaRatio) +
                            diagnostics.centerOffsetRatio * 0.25 +
                            diagnostics.aspectRatioDifference * 0.25 +
                            diagnostics.majorAxisAngleDifferenceDegrees / 360.0;
                        if (quality < bestQuality)
                        {
                            bestQuality = quality;
                            bestDiagnostics = correction.correctedDiagnostics;
                            result.outerTripleEllipse = correction.outer;
                            result.innerTripleEllipse = correction.inner;
                        }
                    }
                }

                result.hasValidTriples = isfinite(bestQuality);
                if (result.hasValidTriples)
                {
                    log_info(
                        "TRIPLE_RING_QUALITY camera=" + log_string(camera_idx) +
                        " status=VALID area_ratio=" + log_string(bestDiagnostics.areaRatio) +
                        " expected_area_ratio=" + log_string(bestDiagnostics.expectedAreaRatio) +
                        " center_offset_ratio=" + log_string(bestDiagnostics.centerOffsetRatio));
                }
                else
                {
                    result.outerTripleEllipse = RotatedRect();
                    result.innerTripleEllipse = RotatedRect();
                    log_warning(
                        "TRIPLE_RING_QUALITY camera=" + log_string(camera_idx) +
                        " status=INVALID reason=no_plausible_contour_pair");
                }
            }
            catch (const cv::Exception &e)
            {
                log_debug("FAIL - Triple ellipse fitting failed: " + string(e.what()));
                result.hasValidTriples = false;
            }
        }

        // SECTION 3: CONTOUR FITTING for BULL RINGS (simple method)
        if (!masks.outerBullMask.empty())
        {
            log_debug("Processing outer bull (25-point) with contour fitting...");

            vector<vector<Point>> outerBullContours;
            findContours(masks.outerBullMask, outerBullContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!outerBullContours.empty())
            {
                try
                {
                    // Sort by area and fit to largest contour
                    sort(outerBullContours.begin(), outerBullContours.end(),
                         [](const vector<Point> &a, const vector<Point> &b)
                         {
                             return contourArea(a) > contourArea(b);
                         });

                    if (outerBullContours[0].size() >= 5)
                    {
                        result.outerBullEllipse = fitEllipse(outerBullContours[0]);
                        log_debug("SUCCESS - Fitted outer bull ellipse from contour");
                    }
                }
                catch (const cv::Exception &e)
                {
                    log_debug("Outer bull ellipse fitting failed: " + string(e.what()));
                }
            }
        }

        // SECTION 3.1: CONTOUR FITTING for INNER BULL RING (50-point)
        if (!masks.bullMask.empty())
        {
            log_debug("Processing bullseye (50-point) with contour fitting...");

            vector<vector<Point>> bullContours;
            findContours(masks.bullMask, bullContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            if (!bullContours.empty())
            {
                try
                {
                    // Sort by area and fit to largest contour
                    sort(bullContours.begin(), bullContours.end(),
                         [](const vector<Point> &a, const vector<Point> &b)
                         {
                             return contourArea(a) > contourArea(b);
                         });

                    if (bullContours[0].size() >= 5)
                    {
                        result.innerBullEllipse = fitEllipse(bullContours[0]);
                        log_debug("SUCCESS - Fitted bullseye ellipse from contour");
                    }
                }
                catch (const cv::Exception &e)
                {
                    log_debug("Bullseye ellipse fitting failed: " + string(e.what()));
                }
            }
        }

        // Prefer a board-center and bull model derived from the two complete
        // scoring rings. Small bull color contours are frequently fragmented
        // or confused with nearby red lettering in oblique camera views.
        const auto rawBullDiagnostics = board_geometry::validateBullPair(
            result.innerBullEllipse,
            result.outerBullEllipse,
            result.outerDoubleEllipse);
        const auto projectedBoard = board_geometry::estimateProjectedBoardModel(
            result.innerDoubleEllipse,
            result.outerDoubleEllipse,
            result.innerTripleEllipse,
            result.outerTripleEllipse,
            originalFrame.size());
        if (projectedBoard.valid)
        {
            result.innerBullEllipse = board_geometry::projectRingFromBoardModel(
                projectedBoard, 6.35f);
            result.outerBullEllipse = board_geometry::projectRingFromBoardModel(
                projectedBoard, 15.9f);
            log_info(
                "BULL_RING_MODEL camera=" + log_string(camera_idx) +
                " status=APPLIED center_x=" + log_string(projectedBoard.center.x) +
                " center_y=" + log_string(projectedBoard.center.y) +
                " raw_bull_status=" + (rawBullDiagnostics.valid ? string("VALID") : string("INVALID")));
        }
        else
        {
            log_warning(
                "BULL_RING_MODEL camera=" + log_string(camera_idx) +
                " status=UNAVAILABLE reason=" + projectedBoard.reason);
        }

        const auto bullDiagnostics = board_geometry::validateBullPair(
            result.innerBullEllipse,
            result.outerBullEllipse,
            result.outerDoubleEllipse);
        result.hasValidBulls = bullDiagnostics.valid;
        if (result.hasValidBulls)
        {
            log_info(
                "BULL_RING_QUALITY camera=" + log_string(camera_idx) +
                " status=VALID area_ratio=" + log_string(bullDiagnostics.areaRatio) +
                " center_distance_px=" + log_string(bullDiagnostics.centerDistancePixels));
        }
        else
        {
            log_warning(
                "BULL_RING_QUALITY camera=" + log_string(camera_idx) +
                " status=INVALID area_ratio=" + log_string(bullDiagnostics.areaRatio) +
                " center_distance_px=" + log_string(bullDiagnostics.centerDistancePixels) +
                " max_center_distance_px=" + log_string(bullDiagnostics.maxCenterDistancePixels) +
                " reason=" + bullDiagnostics.reason);
        }

        // SECTION 4: PERSPECTIVE ANALYSIS (using doubles as reference)
        if (result.hasValidDoubles)
        {
            Point ellipseCenter(result.outerDoubleEllipse.center);
            result.offsetX = ellipseCenter.x - bullCenter.x;
            result.offsetY = ellipseCenter.y - bullCenter.y;
            result.offsetMagnitude = norm(Point2f(result.offsetX, result.offsetY));
            result.offsetAngle = atan2(result.offsetY, result.offsetX) * 180.0 / CV_PI;

            log_debug("Perspective offset - X=" + log_string(result.offsetX) + ", Y=" + log_string(result.offsetY) + ", magnitude=" + log_string(result.offsetMagnitude) + "px");
        }

        // Set hasDetectedEllipses flag
        result.hasDetectedEllipses = (result.hasValidDoubles && result.hasValidTriples && result.hasValidBulls);

        // SECTION 5: DEBUG VISUALIZATION
        if (debug_mode)
        {
            Mat ellipseVis = originalFrame.clone();

            // Draw all detected ellipses with different colors
            if (result.hasValidDoubles)
            {
                ellipse(ellipseVis, result.outerDoubleEllipse, Scalar(255, 0, 255), 3);
                ellipse(ellipseVis, result.innerDoubleEllipse, Scalar(255, 255, 0), 2);
            }

            if (result.hasValidTriples)
            {
                ellipse(ellipseVis, result.outerTripleEllipse, Scalar(255, 0, 255), 2);
                ellipse(ellipseVis, result.innerTripleEllipse, Scalar(255, 255, 0), 2);
            }

            if (result.hasValidBulls)
            {
                ellipse(ellipseVis, result.outerBullEllipse, Scalar(255, 0, 255), 2);
                ellipse(ellipseVis, result.innerBullEllipse, Scalar(255, 255, 0), 2);
            }

            // // Draw bull center
            // circle(ellipseVis, bullCenter, 8, Scalar(0, 0, 0), -1);
            // circle(ellipseVis, bullCenter, 10, Scalar(255, 255, 255), 2);

            system("mkdir -p debug_frames/ellipse_processing");
            imwrite("debug_frames/ellipse_processing/ellipse_result_" + to_string(camera_idx) + ".jpg", ellipseVis);

            log_debug("Saved ellipse visualization with all ring types");
        }

        return result;
    }

} // namespace ellipse_processing
