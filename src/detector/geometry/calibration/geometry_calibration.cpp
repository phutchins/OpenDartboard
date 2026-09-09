#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <nlohmann/json.hpp>

#include "utils.hpp"
#include "geometry_calibration.hpp"
#include "color_processing.hpp"
#include "roi_processing.hpp"
#include "mask_processing.hpp"
#include "contour_processing.hpp"
#include "bull_processing.hpp"
#include "ellipse_processing.hpp"
#include "wire_processing.hpp"
#include "board_geometry.hpp"
#include "orientation_processing.hpp"
#include "dartboard_visualization.hpp"
#include "perspective_processing.hpp"

using namespace cv;
using namespace std;
using json = nlohmann::json;

namespace geometry_calibration
{
    namespace
    {
        constexpr float outerDoubleRadius = 170.0f;

        Point2f canonicalPoint(float radius, float angleDegrees)
        {
            const float angle = angleDegrees * static_cast<float>(CV_PI) / 180.0f;
            return Point2f(radius * cos(angle), radius * sin(angle));
        }

        Point2f ellipseIntersection(
            const Point2f &center,
            const Point2f &toward,
            const RotatedRect &ellipse)
        {
            Point2f direction = toward - center;
            const float length = norm(direction);
            float distance = 0.0f;
            if (length <= 0.0f ||
                !board_geometry::rayEllipseIntersectionDistance(center, direction, ellipse, distance))
                return Point2f(-1.0f, -1.0f);
            return center + direction / length * distance;
        }

        void applyCanonicalGeometry(
            DartboardCalibration &calibration,
            const board_geometry::PlanarBoardTransform &transform,
            int anchorWireIndex)
        {
            calibration.boardTransform = transform;
            if (!transform.valid)
                return;

            Point2f imageCenter;
            if (!board_geometry::boardToImage(transform, Point2f(0.0f, 0.0f), imageCenter))
                return;
            calibration.bullCenter = Point(cvRound(imageCenter.x), cvRound(imageCenter.y));
            calibration.ellipses.innerBullEllipse = board_geometry::fitProjectedRingEllipse(transform, 6.35f);
            calibration.ellipses.outerBullEllipse = board_geometry::fitProjectedRingEllipse(transform, 15.9f);
            calibration.ellipses.innerTripleEllipse = board_geometry::fitProjectedRingEllipse(transform, 99.0f);
            calibration.ellipses.outerTripleEllipse = board_geometry::fitProjectedRingEllipse(transform, 107.0f);
            calibration.ellipses.innerDoubleEllipse = board_geometry::fitProjectedRingEllipse(transform, 162.0f);
            calibration.ellipses.outerDoubleEllipse = board_geometry::fitProjectedRingEllipse(transform, 170.0f);
            calibration.ellipses.hasValidBulls = true;
            calibration.ellipses.hasValidTriples = true;
            calibration.ellipses.hasValidDoubles = true;
            calibration.ellipses.hasDetectedEllipses = true;

            calibration.wires.isValid = true;
            calibration.wires.hasInferredEndpoints = false;
            for (int wireIndex = 0; wireIndex < 20; ++wireIndex)
            {
                const int canonicalIndex = (wireIndex - anchorWireIndex + 20) % 20;
                Point2f endpoint;
                board_geometry::boardToImage(
                    transform,
                    canonicalPoint(outerDoubleRadius, -99.0f + canonicalIndex * 18.0f),
                    endpoint);
                calibration.wires.wireEndpoints[wireIndex] = endpoint;
            }
        }

        board_geometry::PlanarBoardTransform buildAutomaticTransform(
            const DartboardCalibration &calibration,
            const Mat &rawMask)
        {
            board_geometry::PlanarBoardTransform result;
            if (!calibration.ellipses.hasValidDoubles ||
                !calibration.ellipses.hasValidTriples ||
                !calibration.wires.isValid)
                return result;

            const int anchorWireIndex = hasValidOrientation(calibration)
                                            ? calibration.orientation.wedge20WireIndex
                                            : 0;
            const Point2f center(calibration.bullCenter);
            const array<pair<float, const RotatedRect *>, 4> rings{{
                {99.0f, &calibration.ellipses.innerTripleEllipse},
                {107.0f, &calibration.ellipses.outerTripleEllipse},
                {162.0f, &calibration.ellipses.innerDoubleEllipse},
                {170.0f, &calibration.ellipses.outerDoubleEllipse},
            }};
            vector<Point2f> boardPoints{Point2f(0.0f, 0.0f)};
            vector<Point2f> imagePoints{center};
            for (int wireIndex = 0; wireIndex < 20; ++wireIndex)
            {
                const int canonicalIndex = (wireIndex - anchorWireIndex + 20) % 20;
                const float angleDegrees = -99.0f + canonicalIndex * 18.0f;
                for (const auto &ring : rings)
                {
                    const Point2f imagePoint = ellipseIntersection(
                        center,
                        calibration.wires.wireEndpoints[wireIndex],
                        *ring.second);
                    if (imagePoint.x < 0.0f || imagePoint.y < 0.0f)
                        continue;
                    boardPoints.push_back(canonicalPoint(ring.first, angleDegrees));
                    imagePoints.push_back(imagePoint);
                }
            }

            result = board_geometry::estimatePlanarBoardTransform(boardPoints, imagePoints, false);
            const auto residual = board_geometry::measureRingEdgeResidual(rawMask, result);
            result.ringResidualMeanPixels = residual.meanPixels;
            result.ringResidualP90Pixels = residual.p90Pixels;
            result.residualSamples = residual.samples;
            result.valid = result.valid && residual.valid;
            return result;
        }

        bool readLandmark(
            const json &camera,
            const char *name,
            const Size &frameSize,
            Point2f &point)
        {
            if (!camera.contains(name) || !camera[name].is_object())
                return false;
            const auto &value = camera[name];
            if (!value.contains("x") || !value.contains("y") ||
                !value["x"].is_number() || !value["y"].is_number())
                return false;
            point.x = value["x"].get<float>();
            point.y = value["y"].get<float>();
            return isfinite(point.x) && isfinite(point.y) &&
                   point.x >= 0.0f && point.y >= 0.0f &&
                   point.x < frameSize.width && point.y < frameSize.height;
        }

        bool applyManualOverride(
            DartboardCalibration &calibration,
            const Mat &rawMask,
            const Size &frameSize)
        {
            ifstream file("calibration_overrides.json");
            if (!file)
                return false;
            json document;
            try
            {
                file >> document;
            }
            catch (const exception &error)
            {
                log_warning("CALIBRATION_OVERRIDE status=INVALID reason=json_parse_error detail=" + string(error.what()));
                return false;
            }

            const string cameraKey = to_string(calibration.camera_index);
            if (!document.contains("cameras") || !document["cameras"].is_object() ||
                !document["cameras"].contains(cameraKey))
                return false;
            const auto &camera = document["cameras"][cameraKey];
            Point2f center, north, east, south, west;
            if (!readLandmark(camera, "center", frameSize, center) ||
                !readLandmark(camera, "north", frameSize, north) ||
                !readLandmark(camera, "east", frameSize, east) ||
                !readLandmark(camera, "south", frameSize, south) ||
                !readLandmark(camera, "west", frameSize, west))
            {
                log_warning("CALIBRATION_OVERRIDE camera=" + to_string(calibration.camera_index) +
                            " status=INVALID reason=missing_or_out_of_frame_landmark");
                return false;
            }
            if (norm(north - center) < 50.0f || norm(east - center) < 50.0f ||
                norm(south - center) < 50.0f || norm(west - center) < 50.0f)
            {
                log_warning("CALIBRATION_OVERRIDE camera=" + to_string(calibration.camera_index) +
                            " status=INVALID reason=landmarks_too_close");
                return false;
            }

            const vector<Point2f> boardPoints{
                Point2f(0.0f, 0.0f), Point2f(0.0f, -outerDoubleRadius),
                Point2f(outerDoubleRadius, 0.0f), Point2f(0.0f, outerDoubleRadius),
                Point2f(-outerDoubleRadius, 0.0f)};
            const vector<Point2f> imagePoints{center, north, east, south, west};
            auto transform = board_geometry::estimatePlanarBoardTransform(boardPoints, imagePoints, true);
            if (!transform.valid)
            {
                log_warning("CALIBRATION_OVERRIDE camera=" + to_string(calibration.camera_index) +
                            " status=INVALID reason=homography_failed");
                return false;
            }
            const auto residual = board_geometry::measureRingEdgeResidual(rawMask, transform, 12.0f, 20.0f);
            transform.ringResidualMeanPixels = residual.meanPixels;
            transform.ringResidualP90Pixels = residual.p90Pixels;
            transform.residualSamples = residual.samples;

            calibration.orientation.camera_index = calibration.camera_index;
            calibration.orientation.wedge20WireIndex = 0;
            calibration.orientation.southWireIndex = 10;
            calibration.orientation.wedgeNumber = 20;
            calibration.orientation.orientation = Point2f(0.0f, -1.0f);
            applyCanonicalGeometry(calibration, transform, 0);
            log_info("CALIBRATION_OVERRIDE camera=" + to_string(calibration.camera_index) +
                     " status=APPLIED residual_mean_px=" + to_string(transform.ringResidualMeanPixels) +
                     " residual_p90_px=" + to_string(transform.ringResidualP90Pixels));
            return true;
        }

        string buildCalibrationModelDetails(const DartboardCalibration &calibration)
        {
            const auto &transform = calibration.boardTransform;
            Point2f center, north, east, south, west;
            bool hasLandmarks = transform.residualSamples > 0 &&
                                board_geometry::projectPoint(transform.boardToImage, Point2f(0.0f, 0.0f), center) &&
                                board_geometry::projectPoint(transform.boardToImage, Point2f(0.0f, -outerDoubleRadius), north) &&
                                board_geometry::projectPoint(transform.boardToImage, Point2f(outerDoubleRadius, 0.0f), east) &&
                                board_geometry::projectPoint(transform.boardToImage, Point2f(0.0f, outerDoubleRadius), south) &&
                                board_geometry::projectPoint(transform.boardToImage, Point2f(-outerDoubleRadius, 0.0f), west);

            // A camera can identify the outer double and all numbered wires
            // while still failing an inner-ring check. Export useful manual
            // editor seeds from that independently detected geometry instead
            // of forcing the user to begin from a generic rectangle.
            if (!hasLandmarks && calibration.ellipses.hasValidDoubles &&
                calibration.wires.isValid && hasValidOrientation(calibration))
            {
                center = Point2f(calibration.bullCenter);
                const int anchor = calibration.orientation.wedge20WireIndex;
                const auto sectorCenter = [&](int sectorOffset, Point2f &point) {
                    const int first = (anchor + sectorOffset) % 20;
                    const int second = (first + 1) % 20;
                    const Point2f toward =
                        (calibration.wires.wireEndpoints[first] +
                         calibration.wires.wireEndpoints[second]) *
                        0.5f;
                    point = ellipseIntersection(
                        center, toward, calibration.ellipses.outerDoubleEllipse);
                    return point.x >= 0.0f && point.y >= 0.0f;
                };
                hasLandmarks = sectorCenter(0, north) &&
                               sectorCenter(5, east) &&
                               sectorCenter(10, south) &&
                               sectorCenter(15, west);
            }
            return "CALIBRATION_MODEL camera=" + to_string(calibration.camera_index) +
                   " source=" + (transform.manual ? string("MANUAL") : string("AUTO")) +
                   " status=" + (transform.valid ? string("VALID") : string("INVALID")) +
                   " residual_mean_px=" + to_string(transform.ringResidualMeanPixels) +
                   " residual_p90_px=" + to_string(transform.ringResidualP90Pixels) +
                   " residual_samples=" + to_string(transform.residualSamples) +
                   " center_x=" + to_string(hasLandmarks ? center.x : -1.0f) +
                   " center_y=" + to_string(hasLandmarks ? center.y : -1.0f) +
                   " north_x=" + to_string(hasLandmarks ? north.x : -1.0f) +
                   " north_y=" + to_string(hasLandmarks ? north.y : -1.0f) +
                   " east_x=" + to_string(hasLandmarks ? east.x : -1.0f) +
                   " east_y=" + to_string(hasLandmarks ? east.y : -1.0f) +
                   " south_x=" + to_string(hasLandmarks ? south.x : -1.0f) +
                   " south_y=" + to_string(hasLandmarks ? south.y : -1.0f) +
                   " west_x=" + to_string(hasLandmarks ? west.x : -1.0f) +
                   " west_y=" + to_string(hasLandmarks ? west.y : -1.0f);
        }
    }

    string calibrationModelDetails(const DartboardCalibration &calibration)
    {
        return buildCalibrationModelDetails(calibration);
    }

    bool hasValidGeometry(const DartboardCalibration &calibration)
    {
        return calibration.camera_index >= 0 &&
               calibration.boardTransform.valid &&
               calibration.ellipses.hasValidDoubles &&
               calibration.wires.isValid;
    }

    bool hasCompleteRingGeometry(const DartboardCalibration &calibration)
    {
        return hasValidGeometry(calibration) &&
               calibration.ellipses.hasValidTriples &&
               calibration.ellipses.hasValidBulls;
    }

    bool hasValidOrientation(const DartboardCalibration &calibration)
    {
        const auto &orientation = calibration.orientation;
        const int wire_count = static_cast<int>(calibration.wires.wireEndpoints.size());

        return !calibration.wires.hasInferredEndpoints &&
               orientation.southWireIndex >= 0 && orientation.southWireIndex < wire_count &&
               orientation.wedge20WireIndex >= 0 && orientation.wedge20WireIndex < wire_count &&
               orientation.wedgeNumber > 0;
    }

    CalibrationStatus getCalibrationStatus(const DartboardCalibration &calibration)
    {
        if (!hasValidGeometry(calibration))
            return CalibrationStatus::INVALID;
        if (!hasCompleteRingGeometry(calibration) || !hasValidOrientation(calibration))
            return CalibrationStatus::DEGRADED;
        return CalibrationStatus::READY;
    }

    const char *calibrationStatusToString(CalibrationStatus status)
    {
        switch (status)
        {
        case CalibrationStatus::READY:
            return "READY";
        case CalibrationStatus::DEGRADED:
            return "DEGRADED";
        default:
            return "INVALID";
        }
    }

    // This function orchestrates the entire calibration pipeline for one camera
    DartboardCalibration calibrateSingleCamera(const Mat &frame, int cameraIdx, bool debugMode)
    {
        log_info("Calibrating camera " + log_string(cameraIdx + 1));

        if (frame.empty())
        {
            log_error("Empty frame from camera " + log_string(cameraIdx + 1));
            return DartboardCalibration(); // Return empty calibration
        }

        DartboardCalibration calibration;
        calibration.camera_index = cameraIdx;
        calibration.capture_width = frame.cols;
        calibration.capture_height = frame.rows;
        calibration.timestamp = static_cast<uint64_t>(time(nullptr)); // Current Unix timestamp

        Mat orginalFrame = frame.clone();
        Point frameCenter = math::calculateFrameCenter(frame);
        calibration.frameCenter = frameCenter;

        // [===STEP 1:===] Create ROI using the clean ROI processing module
        roi_processing::ROIParams roiParams;
        Mat roiFrame = roi_processing::processROI(orginalFrame, debugMode, cameraIdx, roiParams);

        // [===STEP 2:===] Detect red-green colors using the CLEAN color detection module
        color_processing::ColorParams colorParams;
        Mat redGreenFrame = color_processing::processColors(roiFrame, cameraIdx, debugMode, colorParams);

        // [===STEP 3:===] contour DETECTION using the new contour processing module
        // Note: This step is currently commented out as it is not used in the new pipeline
        // Uncomment if contour processing is needed in the future, for now leave it here for reference
        // contour_processing::ContourParams contourParams;
        // vector<vector<Point>> contours = contour_processing::processContours(redGreenFrame, orginalFrame, cameraIdx, debugMode, contourParams);

        // [===STEP 4:===] BULL dectection using the new bull processing module
        bull_processing::BullParams bullParams;
        Point bullCenter = bull_processing::processBull(redGreenFrame, frameCenter, cameraIdx, debugMode, bullParams);
        calibration.bullCenter = bullCenter;

        // [===STEP 5:===] Create binary mask for contour processing
        mask_processing::MaskParams maskParams;
        mask_processing::MaskBundle masks = mask_processing::processMask(redGreenFrame, bullCenter, cameraIdx, debugMode, maskParams);

        // [===STEP 6:===] ELLIPSE DETECTION for dartboard shape
        ellipse_processing::EllipseParams ellipseParams;
        ellipse_processing::EllipseBoundaryData ellipseData = ellipse_processing::processEllipse(orginalFrame, masks, bullCenter, frameCenter, cameraIdx, debugMode, ellipseParams);
        calibration.ellipses = ellipseData;

        // The initial color-contour pass can prefer a large perspective ring
        // over the bull. Refine it from the independently fitted bull ellipses
        // before wire and orientation processing, but only within a bounded,
        // board-scale displacement.
        const auto bullRefinement = board_geometry::selectBullCenterRefinement(
            Point2f(calibration.bullCenter),
            calibration.ellipses.innerBullEllipse,
            calibration.ellipses.outerBullEllipse,
            calibration.ellipses.outerDoubleEllipse,
            orginalFrame.size());
        const string refinementDetails =
            "BULL_CENTER_REFINEMENT camera=" + to_string(cameraIdx) +
            " status=" + (bullRefinement.accepted ? "ACCEPTED" : "REJECTED") +
            " source=" + board_geometry::bullCenterSourceToString(bullRefinement.source) +
            " coarse_x=" + to_string(calibration.bullCenter.x) +
            " coarse_y=" + to_string(calibration.bullCenter.y) +
            " candidate_x=" + to_string(bullRefinement.center.x) +
            " candidate_y=" + to_string(bullRefinement.center.y) +
            " displacement_px=" + to_string(bullRefinement.displacementPixels) +
            " max_displacement_px=" + to_string(bullRefinement.maxDisplacementPixels) +
            " inner_outer_distance_px=" + to_string(bullRefinement.innerOuterDistancePixels) +
            " max_inner_outer_distance_px=" + to_string(bullRefinement.maxInnerOuterDistancePixels) +
            " reason=" + bullRefinement.reason;

        if (bullRefinement.accepted)
        {
            calibration.bullCenter = Point(
                cvRound(bullRefinement.center.x),
                cvRound(bullRefinement.center.y));

            // Keep perspective diagnostics consistent with the center used by
            // wire and orientation processing.
            if (calibration.ellipses.hasValidDoubles)
            {
                calibration.ellipses.offsetX =
                    calibration.ellipses.outerDoubleEllipse.center.x - calibration.bullCenter.x;
                calibration.ellipses.offsetY =
                    calibration.ellipses.outerDoubleEllipse.center.y - calibration.bullCenter.y;
                calibration.ellipses.offsetMagnitude = norm(Point2f(
                    calibration.ellipses.offsetX,
                    calibration.ellipses.offsetY));
                calibration.ellipses.offsetAngle = atan2(
                    calibration.ellipses.offsetY,
                    calibration.ellipses.offsetX) * 180.0 / CV_PI;
            }
            log_info(refinementDetails);
        }
        else
        {
            log_warning(refinementDetails);
        }

        // [===STEP 8:===] Extract actual wire positions for segment alignment
        wire_processing::WireDetectionConfig wireConfig;
        // When useHoughLinesDetection = false, it will use ensemble method

        wire_processing::WireData wireData = wire_processing::processWires(orginalFrame, redGreenFrame, calibration, debugMode, wireConfig);
        calibration.wires = wireData;

        if (calibration.wires.isValid)
        {
            //[===STEP 8.5:===] PERSPECTIVE CORRECTION - Apply perspective correction to the mask
            perspective_processing::DartboardSpec perspectiveSpec;
            Mat rectifiedImage = perspective_processing::processPerspective(orginalFrame, calibration, debugMode, perspectiveSpec);

            // [===STEP 9:===] ORIENTATION DETECTION - Find where "20" segment is located
            orientation_processing::OrientationParams orientationParams;
            orientation_processing::OrientationData orientationData = orientation_processing::processOrientation(orginalFrame, redGreenFrame, calibration, debugMode, orientationParams);
            calibration.orientation = orientationData;
        }
        else
        {
            log_warning("Skipping perspective and orientation processing for camera " + to_string(cameraIdx) +
                        " because wire detection is invalid");
        }

        const int anchorWireIndex = hasValidOrientation(calibration)
                                        ? calibration.orientation.wedge20WireIndex
                                        : 0;
        const auto automaticTransform = buildAutomaticTransform(calibration, masks.fullMask);
        applyCanonicalGeometry(calibration, automaticTransform, anchorWireIndex);
        applyManualOverride(calibration, masks.fullMask, orginalFrame.size());
        if (calibration.boardTransform.valid)
            log_info(calibrationModelDetails(calibration));
        else
            log_warning(calibrationModelDetails(calibration));

        return calibration;
    }

    // Multi-camera calibration orchestration
    vector<DartboardCalibration> calibrateMultipleCameras(const vector<Mat> &frames, bool debugMode, int targetWidth, int targetHeight)
    {
        vector<DartboardCalibration> calibrations;

        log_debug("DARTBOARD CALIBRATION STARTED");

        if (frames.empty())
        {
            log_error("No frames provided for calibration");
            return calibrations;
        }

        // Check if all frames have the same size and calibrate
        for (size_t cam_idx = 0; cam_idx < frames.size(); cam_idx++)
        {
            if (frames[cam_idx].empty())
            {
                log_warning("Empty frame from camera " + log_string(cam_idx + 1));
                continue;
            }

            //  Just call the static function
            DartboardCalibration calibration = calibrateSingleCamera(frames[cam_idx], cam_idx, debugMode);
            calibrations.push_back(calibration);

            const CalibrationStatus status = getCalibrationStatus(calibration);
            const bool geometry_valid = hasValidGeometry(calibration);
            const bool orientation_valid = hasValidOrientation(calibration);
            const string status_details =
                "CALIBRATION_STATUS camera=" + to_string(cam_idx) +
                " status=" + calibrationStatusToString(status) +
                " geometry=" + (geometry_valid ? "valid" : "invalid") +
                " orientation=" + (orientation_valid ? "valid" : "invalid") +
                " camera_position=" + orientation_processing::cameraPositionToString(calibration.orientation.cameraPosition) +
                " wedge20_wire=" + to_string(calibration.orientation.wedge20WireIndex) +
                " south_wire=" + to_string(calibration.orientation.southWireIndex) +
                " wires_valid=" + (calibration.wires.isValid ? "true" : "false") +
                " doubles_valid=" + (calibration.ellipses.hasValidDoubles ? "true" : "false") +
                " triples_valid=" + (calibration.ellipses.hasValidTriples ? "true" : "false") +
                " bulls_valid=" + (calibration.ellipses.hasValidBulls ? "true" : "false");

            if (debugMode)
                log_debug(status_details);
            if (calibration.boardTransform.valid)
                log_info(calibrationModelDetails(calibration));
            else
                log_warning(calibrationModelDetails(calibration));

            if (status == CalibrationStatus::DEGRADED)
            {
                string reasons;
                if (!hasCompleteRingGeometry(calibration))
                    reasons = "ring_geometry_incomplete";
                if (!orientation_valid)
                    reasons += (reasons.empty() ? "" : ",") + string("orientation_invalid");
                log_warning("CALIBRATION_DEGRADED camera=" + to_string(cam_idx) + " reason=" + reasons +
                            (orientation_valid ? "" : "; wedge scores are disabled for this camera"));
            }
            else if (status == CalibrationStatus::INVALID)
                log_error("CALIBRATION_INVALID camera=" + to_string(cam_idx) + " reason=geometry_invalid");

            if (debugMode)
            {
                // Create a debug visualization showing the calibration
                cv::Mat visFrame = dartboard_visualization::drawCalibrationOverlay(frames[cam_idx], calibration, true);
                system("mkdir -p debug_frames/geometry_calibration");
                imwrite("debug_frames/geometry_calibration/calibration_camera_" + to_string(cam_idx) + ".jpg", visFrame);
            }
        }

        // once all cameras are calibrated, we need to find the star camera
        // and then use it to determine the perspective correction for the other 2 cameras

        if (debugMode && !calibrations.empty())
        {
            log_debug("Creating combined calibration visualization");
            Mat combinedViz = debug::createCombinedCalibrationVisualization(frames.size());

            if (!combinedViz.empty())
            {
                system("mkdir -p debug_frames");
                string filename = "debug_frames/calibration_summary_all_cameras.jpg";
                imwrite(filename, combinedViz);
                log_debug("Saved combined calibration visualization: " + filename);
            }
        }

        log_debug("DARTBOARD CALIBRATION COMPLETED");
        return calibrations;
    }

} // namespace geometry_calibration
