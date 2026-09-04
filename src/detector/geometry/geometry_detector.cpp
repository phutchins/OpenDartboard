#include <iostream>
#include <algorithm>

#include "geometry_detector.hpp"
#include "calibration/geometry_calibration.hpp"
#include "detection/dart_processing.hpp"
#include "detection/score_processing.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;

// Constructor
GeometryDetector::GeometryDetector(bool debug_mode, int target_width, int target_height, int target_fps,
                                   const motion_processing::MotionParams &motion_params)
    : initialized(false), calibrated(false), debug_mode(debug_mode), target_width(target_width), target_height(target_height), target_fps(target_fps), motion_params(motion_params)
{
}

// Main process method - simplified to basic structure
DetectorResult GeometryDetector::process(const vector<Mat> &frames)
{
#ifdef DEBUG_VIA_VIDEO_INPUT
    if (!frames.empty())
    {
        Mat combined_raw = debug::createCombinedFrame(frames, "RAW");
        raw_streamer->push(combined_raw);
    }
#endif

    DetectorResult result;

    if (!calibrated || frames.empty())
    {
        return result;
    }

    // Process motion session - all motion logic is now handled in motion_processing
    motion_processing::MotionResult motion_result = motion_processing::processMotion(frames, background_frames, debug_mode, motion_params);

    // Process dart state detection
    dart_processing::DartStateResult dart_result = dart_processing::processDartState(frames, background_frames, motion_result.motion_finished, debug_mode);

    // Process scoring using the new scoring system
    score_processing::ScoreResult score_result = score_processing::processScore(background_frames, dart_result, calibrations, debug_mode);

    // Only return result if scoring system says it's valid (state changed)
    if (score_result.valid)
    {
        result.dart_detected = true;
        result.score = score_result.score;
        result.position = score_result.pixel_position;
        result.board_position = score_result.dartboard_position;
        result.has_board_position = score_result.has_dartboard_position;
        result.confidence = score_result.confidence;
        result.camera_index = score_result.camera_index;
    }

    return result;
}

// Main initialization method
bool GeometryDetector::initialize(vector<VideoCapture> &cameras)
{
#ifdef DEBUG_VIA_VIDEO_INPUT
    // Initialize multiple debug streamers
    raw_streamer = make_unique<streamer>(8081, cameras[0].get(cv::CAP_PROP_FPS));
    cv::Mat startup_img_raw(target_height, target_width, CV_8UC3, cv::Scalar::all(0));
    cv::putText(startup_img_raw, "Raw Cameras", {50, 100}, cv::FONT_HERSHEY_SIMPLEX, 1.2, {0, 255, 0}, 2);
#endif

    // Try to load cached calibration first
    // calibrations = cache::geometry::load();
    if (!calibrations.empty())
    {
        // Already calibrated, just set initialized
        log_info("Loaded cached calibration with " + to_string(calibrations.size()) + " cameras");

        // Load background frames
        background_frames = cache::geometry::loadBackgroundFrames();

        // set initialized and calibrated
        initialized = true;
        calibrated = true;
        return true;
    }

    log_info("Capturing frames for calibration...");
    vector<Mat> initial_frames = camera::captureAndAverageFrames(cameras, 30); // Capture 75 frames for averaging

    if (!initial_frames.empty())
    {
        log_info("Performing immediate calibration...");

        calibrations = geometry_calibration::calibrateMultipleCameras(
            initial_frames,
            debug_mode,
            target_width,
            target_height);

        int ready_calibrations = 0;
        int degraded_calibrations = 0;
        int invalid_calibrations = 0;
        for (const auto &calibration : calibrations)
        {
            switch (geometry_calibration::getCalibrationStatus(calibration))
            {
            case CalibrationStatus::READY:
                ready_calibrations++;
                break;
            case CalibrationStatus::DEGRADED:
                degraded_calibrations++;
                break;
            case CalibrationStatus::INVALID:
                invalid_calibrations++;
                break;
            }
        }

        calibrated = (ready_calibrations + degraded_calibrations) > 0;

        if (calibrated)
        {
            // Save frames as background (for dart detection)
            background_frames.clear();
            for (const auto &frame : initial_frames)
                background_frames.push_back(frame.clone());

            if (ready_calibrations == static_cast<int>(calibrations.size()))
            {
                log_info("CALIBRATION_SUMMARY status=READY ready=" + to_string(ready_calibrations) +
                         " degraded=0 invalid=0");
            }
            else
            {
                log_warning("CALIBRATION_SUMMARY status=DEGRADED ready=" + to_string(ready_calibrations) +
                            " degraded=" + to_string(degraded_calibrations) +
                            " invalid=" + to_string(invalid_calibrations) +
                            "; geometric detection may continue but reliable wedge scoring requires READY orientation");
            }

            // Save calibration for future use
            if (cache::geometry::save(calibrations))
            {
                log_debug("Saved calibration");
            }

            // Save background frames for dart detection
            if (cache::geometry::saveBackgroundFrames(background_frames))
            {
                log_debug("Saved background frames");
            }

            initialized = true;
            calibrated = true;
        }
        else
        {
            log_error("CALIBRATION_SUMMARY status=FAILED ready=0 degraded=0 invalid=" + to_string(invalid_calibrations));
            initialized = false;
            calibrated = false;
        }
    }
    else
    {
        log_error("No initial frames captured for calibration");
        initialized = false;
        calibrated = false;
    }

    return initialized;
}
