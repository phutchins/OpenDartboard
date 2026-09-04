#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "utils.hpp"
#include "geometry_calibration.hpp"
#include "color_processing.hpp"
#include "roi_processing.hpp"
#include "mask_processing.hpp"
#include "contour_processing.hpp"
#include "bull_processing.hpp"
#include "ellipse_processing.hpp"
#include "wire_processing.hpp"
#include "orientation_processing.hpp"
#include "dartboard_visualization.hpp"
#include "perspective_processing.hpp"

using namespace cv;
using namespace std;

namespace geometry_calibration
{
    bool hasValidGeometry(const DartboardCalibration &calibration)
    {
        return calibration.camera_index >= 0 &&
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

        return orientation.cameraPosition != orientation_processing::CameraPosition::UNKNOWN &&
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
