#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "orientation_processing.hpp"
#include "board_geometry.hpp"
#include "perspective_processing.hpp"
#include "geometry_calibration.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

namespace orientation_processing
{
    // Create perspective-aware number region mask
    static Mat createNumberRegionMask(const Mat &frame, const DartboardCalibration &calib, bool enableDebug, const OrientationParams &params)
    {
        log_debug("1Creating number region mask for camera " + log_string(calib.camera_index));

        // Get the outer doubles ellipse (our reference)
        RotatedRect outerDoubles = calib.ellipses.outerDoubleEllipse;

        // Create dartboard boundary - scale up from outer doubles
        RotatedRect dartboardBoundary = outerDoubles;
        dartboardBoundary.size.width *= params.boundryScaleFactor;
        dartboardBoundary.size.height *= params.boundryScaleFactor;

        // Apply perspective correction using the existing offset data
        Point2f perspectiveOffset(calib.ellipses.offsetX, calib.ellipses.offsetY);
        dartboardBoundary.center = outerDoubles.center + perspectiveOffset * (params.boundryScaleFactor - 1) * 2;

        log_debug("Perspective offset: (" + log_string(perspectiveOffset.x) + "," + log_string(perspectiveOffset.y) + ")");
        log_debug("Dartboard outer boundary center: (" + log_string(dartboardBoundary.center.x) + "," + log_string(dartboardBoundary.center.y) + ")");

        RotatedRect dartboardInnerBoundary = outerDoubles;
        dartboardInnerBoundary.size.width *= params.innerBoundryScaleFactor;
        dartboardInnerBoundary.size.height *= params.innerBoundryScaleFactor;

        // Apply perspective correction using the existing offset data
        Point2f perspectiveOffsetInner(calib.ellipses.offsetX, calib.ellipses.offsetY);
        dartboardInnerBoundary.center = outerDoubles.center + perspectiveOffsetInner * (params.innerBoundryScaleFactor - 1) * 2;

        log_debug("Perspective offset: (" + log_string(perspectiveOffsetInner.x) + "," + log_string(perspectiveOffsetInner.y) + ")");
        log_debug("Dartboard inner boundary center: (" + log_string(dartboardInnerBoundary.center.x) + "," + log_string(dartboardInnerBoundary.center.y) + ")");

        // Create the two masks
        Mat outerMask = Mat::zeros(frame.size(), CV_8UC1);
        Mat innerMask = Mat::zeros(frame.size(), CV_8UC1);

        // Draw the ellipses
        ellipse(outerMask, dartboardBoundary, Scalar(255), -1);      // Outer boundary (white)
        ellipse(innerMask, dartboardInnerBoundary, Scalar(255), -1); // Inner boundary (white)

        // Create ring mask (outer - inner = ring between them)
        Mat numberRegionMask = outerMask - innerMask;

        log_debug("Number region mask created successfully");
        log_debug("Preprocessing text region for camera " + log_string(calib.camera_index));

        // Extract the number region from frame using the mask
        Mat numberRegion = Mat::zeros(frame.size(), CV_8UC3);
        frame.copyTo(numberRegion, numberRegionMask);

        // Convert to grayscale
        Mat grayRegion;
        cvtColor(numberRegion, grayRegion, COLOR_BGR2GRAY);

        // Remove black pixels AND dark grays - numbers are on white/colored background
        Mat brightMask;
        threshold(grayRegion, brightMask, params.brightnessThreshold, 255, THRESH_BINARY); // Remove dark areas

        // Apply bright mask to keep only bright areas where numbers can be
        Mat brightRegion;
        grayRegion.copyTo(brightRegion, brightMask);

        // Enhance contrast for better text detection
        Mat enhancedGray;
        Ptr<cv::CLAHE> claheProcessor = cv::createCLAHE(3.0, Size(8, 8));
        claheProcessor->apply(brightRegion, enhancedGray);

        // Binary threshold to get white text on black background
        Mat binaryTextMask;
        threshold(enhancedGray, binaryTextMask, 0, 255, THRESH_BINARY + THRESH_OTSU);

        log_debug("Text preprocessing completed");

        /// TODO so spider check and then get star camera index

        // Debug output
        if (enableDebug)
        {
            imwrite("debug_frames/orientation_processing/number_region_" + to_string(calib.camera_index) + ".jpg", numberRegion);
            imwrite("debug_frames/orientation_processing/bright_region_" + to_string(calib.camera_index) + ".jpg", brightRegion);
            imwrite("debug_frames/orientation_processing/binary_text_" + to_string(calib.camera_index) + ".jpg", binaryTextMask);
        }

        return binaryTextMask; // Return perspective-corrected masks
    }

    // Find the 4 clip wires by extending wires outward and checking for dartboard collision
    static vector<pair<Point2f, Point2f>> findClipWires(const Mat &frame, const Mat &mask, const DartboardCalibration &calib, bool enableDebug, const OrientationParams &params)
    {
        log_debug("Finding clip wires for camera " + log_string(calib.camera_index));

        vector<pair<Point2f, Point2f>> clipWires;

        // Get dartboard center from ellipse
        Point2f center = calib.bullCenter;

        // Loop through all detected wire endpoints
        for (const auto &wireEndpoint : calib.wires.wireEndpoints)
        {
            // Calculate direction vector from center to wire endpoint
            Point2f direction = wireEndpoint - center;
            float length = norm(direction);

            if (length > 0)
            {
                // Normalize direction
                direction = direction / length;

                // Extend wire outward by 1000+ pixels from the endpoint
                Point2f extendedEnd = wireEndpoint + direction * params.wireExtensionDistance;

                // Check if extended line hits dartboard (sample points along extension)
                bool hitsEmptySpace = true;
                int sampleCount = params.spiderSampleCount; // Sample 50 points along the extension

                for (int i = 1; i <= sampleCount; i++)
                {
                    float t = (float)i / sampleCount;
                    Point2f samplePoint = wireEndpoint + direction * (params.wireExtensionDistance * t);

                    // Check bounds
                    if (samplePoint.x >= 0 && samplePoint.x < mask.cols &&
                        samplePoint.y >= 0 && samplePoint.y < mask.rows)
                    {
                        // If we hit white (dartboard), this is not a clip wire
                        if (mask.at<uchar>(samplePoint) > 128)
                        {
                            hitsEmptySpace = false;
                            break;
                        }
                    }
                }

                // If wire extension stays in empty space, it's a clip wire
                if (hitsEmptySpace)
                {
                    clipWires.push_back(make_pair(wireEndpoint, extendedEnd));
                    log_debug("Found clip wire at (" + log_string(wireEndpoint.x) + "," + log_string(wireEndpoint.y) + ")");
                }
            }
        }

        if (enableDebug)
        {
            Mat clipWireDebug = frame.clone();
            for (const auto &wire : clipWires)
            {
                line(clipWireDebug, wire.first, wire.second, Scalar(0, 255, 0), 2);
                circle(clipWireDebug, wire.first, 5, Scalar(255, 0, 0), -1);
            }
            imwrite("debug_frames/orientation_processing/clip_wires_" + to_string(calib.camera_index) + ".jpg", clipWireDebug);
        }

        if (clipWires.empty())
        {
            LOG_ERROR("No clip wires found for camera " + log_string(calib.camera_index));
        }
        else
        {
            log_debug("Found " + log_string(clipWires.size()) + " clip wires");
        }
        return clipWires;
    }

    // Determine optional camera-layout metadata from real clip candidates and
    // establish the absolute wedge map independently from the wire lattice.
    static OrientationData determineCameraPosition(const vector<pair<Point2f, Point2f>> &clipWires, const DartboardCalibration &calib)
    {
        log_debug("=== DETERMINING CAMERA POSITION FOR CAMERA " + log_string(calib.camera_index) + " ===");

        OrientationData result;
        result.camera_index = calib.camera_index;

        const Point2f center(calib.bullCenter);
        vector<Point2f> clipPoints;
        clipPoints.reserve(clipWires.size());
        for (const auto &wire : clipWires)
            clipPoints.push_back(wire.first);

        const auto clipLayout = board_geometry::analyzeClipLayout(clipPoints, center);
        result.avgClipWireCrossProduct = -clipLayout.horizontalBalance;
        log_debug("ORIENTATION_CANDIDATES camera=" + to_string(calib.camera_index) +
                  " raw=" + to_string(clipWires.size()) +
                  " valid=" + to_string(clipLayout.validCount) +
                  " left=" + to_string(clipLayout.leftCount) +
                  " right=" + to_string(clipLayout.rightCount) +
                  " horizontal_balance=" + to_string(clipLayout.horizontalBalance));

        // A regulation board is installed with 20 at the top. Resolve that
        // absolute direction from the independently validated wire lattice for
        // every camera. Clip visibility may identify a camera's mounting role,
        // but lighting around the number ring makes that signal intermittent.
        const auto northSector = board_geometry::findSectorForImageDirection(
            Point2f(0.0f, -1.0f),
            calib.wires.wireEndpoints,
            center,
            calib.ellipses.outerDoubleEllipse);
        log_debug("ORIENTATION_NORTH_SECTOR camera=" + to_string(calib.camera_index) +
                  " wire=" + to_string(northSector.wireIndex) +
                  " width_deg=" + to_string(northSector.widthDegrees) +
                  " left_margin_deg=" + to_string(northSector.leftMarginDegrees) +
                  " right_margin_deg=" + to_string(northSector.rightMarginDegrees) +
                  " valid=" + (northSector.valid ? "true" : "false"));
        if (northSector.valid)
        {
            result.wedge20WireIndex = northSector.wireIndex;
            result.wedgeNumber = 20;
            result.orientation = Point2f(0.0f, -1.0f);
        }

        // Find the actual closest wire to image south. The old implementation
        // selected the smallest positive angle, which was not necessarily the
        // closest wire on the other side of the 0/360 boundary.
        const float southAngle = board_geometry::angleDegrees(
            board_geometry::normalizeVector(Point2f(0.0f, 1.0f), calib.ellipses.outerDoubleEllipse));
        float closestSouthDifference = 360.0f;
        for (size_t index = 0; index < calib.wires.wireEndpoints.size(); ++index)
        {
            const float wireAngle = board_geometry::normalizedAngleDegrees(
                calib.wires.wireEndpoints[index], center, calib.ellipses.outerDoubleEllipse);
            const float difference = board_geometry::circularAngleDifference(wireAngle, southAngle);
            if (difference < closestSouthDifference)
            {
                closestSouthDifference = difference;
                result.southWireIndex = static_cast<int>(index);
                result.angleOffsetFromSouth = difference;
            }
        }

        switch (clipLayout.layout)
        {
        case board_geometry::ClipLayout::BALANCED:
            result.cameraPosition = CameraPosition::MIDDLE;
            result.isStarCamera = true;
            break;
        case board_geometry::ClipLayout::LEFT_BIASED:
            result.cameraPosition = CameraPosition::BOTTOM;
            break;
        case board_geometry::ClipLayout::RIGHT_BIASED:
            result.cameraPosition = CameraPosition::TOP;
            break;
        case board_geometry::ClipLayout::UNKNOWN:
            break;
        }

        if (result.wedge20WireIndex < 0)
        {
            log_warning("ORIENTATION_STATUS camera=" + to_string(calib.camera_index) +
                        " status=DEGRADED role=" + cameraPositionToString(result.cameraPosition) +
                        " reason=upright_wedge20_not_validated");
        }

        // Debug output
        log_debug("Camera " + log_string(calib.camera_index) + " Position: " + cameraPositionToString(result.cameraPosition));
        log_debug("Camera " + log_string(calib.camera_index) + " Wedge Number: " + log_string(result.wedgeNumber));
        log_debug("Camera " + log_string(calib.camera_index) + " South Wire Index: " + log_string(result.southWireIndex));
        log_debug("Camera " + log_string(calib.camera_index) + " Wedge 20 Wire Index: " + log_string(result.wedge20WireIndex));
        log_debug("Camera " + log_string(calib.camera_index) + " Angle Offset: " + log_string(result.angleOffsetFromSouth) + "°");
        log_debug("=== END CAMERA POSITION DETERMINATION ===");

        return result;
    }

    OrientationData processOrientation(
        const Mat &frame,
        const Mat &colorMask,
        const DartboardCalibration &calib,
        bool enableDebug,
        const OrientationParams &params)
    {
        log_debug("STARTING ORIENTATION DETECTION for camera " + log_string(calib.camera_index));

        // Safety checks
        if (frame.empty())
        {
            log_error("Invalid input data for orientation detection");
            OrientationData result;
            result.camera_index = calib.camera_index;
            return result;
        }

        log_debug("Available wire count: " + log_string(calib.wires.wireEndpoints.size()));

        // Create debug visualization framework
        Mat debugFrame;
        if (enableDebug)
        {
            debugFrame = frame.clone();
            system("mkdir -p debug_frames/orientation_processing");
        }

        // STEP 1: Create the number region mask and apply preprocessing
        Mat binaryTextMask = createNumberRegionMask(frame, calib, enableDebug, params);

        // STEP 2: Find clip wires using the dartboard mask
        vector<pair<Point2f, Point2f>> clipWires = findClipWires(frame, binaryTextMask, calib, enableDebug, params);

        // STEP 3: Determine comprehensive orientation data
        OrientationData result = determineCameraPosition(clipWires, calib);

        // STEP 4: Comprehensive debug visualization
        if (enableDebug)
        {
            Mat orientationDebug = frame.clone();
            Point2f center = calib.bullCenter;

            // Draw clip wires in green
            for (const auto &wire : clipWires)
            {
                line(orientationDebug, wire.first, wire.second, Scalar(0, 255, 0), 2);
                circle(orientationDebug, wire.first, 5, Scalar(0, 255, 0), -1);
            }

            // Draw south wire in blue (if found)
            if (result.southWireIndex >= 0 && result.southWireIndex < calib.wires.wireEndpoints.size())
            {
                Point2f southWire = calib.wires.wireEndpoints[result.southWireIndex];
                line(orientationDebug, center, southWire, Scalar(255, 0, 0), 3);
                circle(orientationDebug, southWire, 8, Scalar(255, 0, 0), -1);
                putText(orientationDebug, "SOUTH", southWire + Point2f(10, 10),
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
            }

            // Draw wedge 20 wire (for all cameras)
            if (result.wedge20WireIndex >= 0 && result.wedge20WireIndex < calib.wires.wireEndpoints.size())
            {
                Point2f wedge20Wire = calib.wires.wireEndpoints[result.wedge20WireIndex];
                line(orientationDebug, center, wedge20Wire, Scalar(0, 0, 255), 4);
                circle(orientationDebug, wedge20Wire, 10, Scalar(0, 0, 255), -1);
                putText(orientationDebug, "WEDGE 20", wedge20Wire + Point2f(10, -10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }

            // Draw south reference line
            Point2f southLineEnd = center + Point2f(0, 100);
            line(orientationDebug, center, southLineEnd, Scalar(128, 128, 128), 2);
            putText(orientationDebug, "SOUTH REF", southLineEnd + Point2f(5, 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(128, 128, 128), 1);

            // Add comprehensive camera status text
            string statusText = "CAM " + to_string(calib.camera_index) + ": " +
                                (result.isStarCamera ? "STAR" : "NON-STAR") + " | " +
                                cameraPositionToString(result.cameraPosition) + " | WEDGE " + to_string(result.wedgeNumber) +
                                " | S:" + to_string(result.southWireIndex) + " | W20:" + to_string(result.wedge20WireIndex);
            putText(orientationDebug, statusText, Point(20, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);

            imwrite("debug_frames/orientation_processing/orientation_result_" + to_string(calib.camera_index) + ".jpg", orientationDebug);
        }

        return result;
    }

} // namespace orientation_processing
