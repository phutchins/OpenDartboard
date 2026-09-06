#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "geometry_detector.hpp"
#include "calibration/geometry_calibration.hpp"
#include "detection/dart_processing.hpp"
#include "detection/score_processing.hpp"
#include "utils.hpp"

using namespace cv;
using namespace std;
using json = nlohmann::json;

namespace
{
    uint64_t diagnostic_event_sequence = 0;

    json pointJson(const Point2f &point)
    {
        return {{"x", point.x}, {"y", point.y}};
    }

    string makeDiagnosticEventId(uint64_t timestamp)
    {
        return "event-" + to_string(timestamp) + "-" + to_string(++diagnostic_event_sequence);
    }

    void saveDiagnosticImage(
        const filesystem::path &event_directory,
        const string &filename,
        const Mat &image,
        json &files)
    {
        if (image.empty())
            return;

        const auto path = event_directory / filename;
        if (imwrite(path.string(), image))
            files.push_back(filename);
    }

    const score_processing::CameraScoreDiagnostic *findScoreDiagnostic(
        const score_processing::ScoreResult &score_result,
        int camera_index)
    {
        for (const auto &diagnostic : score_result.camera_diagnostics)
        {
            if (diagnostic.camera_index == camera_index)
                return &diagnostic;
        }
        return nullptr;
    }

    void captureDiagnosticEvent(
        const string &event_id,
        uint64_t timestamp,
        const vector<Mat> &current_frames,
        const vector<Mat> &background_frames,
        const dart_processing::DartStateResult &dart_result,
        const score_processing::ScoreResult &score_result,
        const vector<DartboardCalibration> &calibrations)
    {
        try
        {
            const filesystem::path event_directory = filesystem::path("debug_frames") / "events" / event_id;
            filesystem::create_directories(event_directory);

            json manifest;
            manifest["schema_version"] = 1;
            manifest["event_id"] = event_id;
            manifest["captured_at_epoch_ms"] = timestamp;
            manifest["ground_truth"] = nullptr;
            manifest["state"] = {
                {"previous", dart_processing::getDartBoardStateName(dart_result.previous_state)},
                {"current", dart_processing::getDartBoardStateName(dart_result.current_state)}};
            manifest["result"] = {
                {"score", score_result.score},
                {"confidence", score_result.confidence},
                {"selected_camera", score_result.camera_index},
                {"pixel_position", pointJson(score_result.pixel_position)},
                {"has_board_position", score_result.has_dartboard_position}};
            if (score_result.has_dartboard_position)
                manifest["result"]["board_position"] = pointJson(score_result.dartboard_position);

            manifest["cameras"] = json::array();
            const size_t camera_count = std::max({
                current_frames.size(),
                background_frames.size(),
                dart_result.camera_results.size(),
                calibrations.size()});

            for (size_t i = 0; i < camera_count; ++i)
            {
                json camera;
                camera["camera_index"] = i;
                camera["files"] = json::array();

                if (i < dart_result.camera_results.size())
                {
                    const auto &detection = dart_result.camera_results[i];
                    camera["state"] = dart_processing::getDartBoardStateName(detection.detected_state);
                    camera["changed_pixels"] = detection.total_changed_pixels;
                    camera["change_ratio_percent"] = detection.change_ratio;
                    camera["tip_found"] = detection.tip_found;
                    camera["tip"] = pointJson(detection.tip_position);
                    camera["shape_center"] = pointJson(detection.center_position);
                }

                if (i < calibrations.size())
                {
                    const auto &calibration = calibrations[i];
                    camera["calibration"] = {
                        {"status", geometry_calibration::calibrationStatusToString(
                                       geometry_calibration::getCalibrationStatus(calibration))},
                        {"geometry_valid", geometry_calibration::hasValidGeometry(calibration)},
                        {"orientation_valid", geometry_calibration::hasValidOrientation(calibration)},
                        {"camera_position", orientation_processing::cameraPositionToString(
                                                calibration.orientation.cameraPosition)},
                        {"bull_center", pointJson(Point2f(calibration.bullCenter))}};
                }

                const auto *score_diagnostic = findScoreDiagnostic(score_result, static_cast<int>(i));
                if (score_diagnostic != nullptr)
                {
                    camera["classification"] = {
                        {"ring", score_diagnostic->ring},
                        {"score", score_diagnostic->score}};
                    if (isfinite(score_diagnostic->nearest_wire_distance))
                        camera["classification"]["nearest_wire_distance_px"] = score_diagnostic->nearest_wire_distance;
                    else
                        camera["classification"]["nearest_wire_distance_px"] = nullptr;
                }

                if (i < background_frames.size())
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_background.jpg", background_frames[i], camera["files"]);
                if (i < current_frames.size())
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_current.jpg", current_frames[i], camera["files"]);
                if (i < dart_result.averaged_frames.size())
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_averaged.jpg", dart_result.averaged_frames[i], camera["files"]);
                if (i < dart_result.board_state_masks.size())
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_board_state_mask.jpg", dart_result.board_state_masks[i], camera["files"]);
                if (i < dart_result.new_dart_masks.size())
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_new_dart_mask.jpg", dart_result.new_dart_masks[i], camera["files"]);

                if (i < background_frames.size() && i < dart_result.camera_results.size())
                {
                    Mat overlay = background_frames[i].clone();
                    const auto &detection = dart_result.camera_results[i];
                    if (detection.tip_found)
                    {
                        line(overlay, detection.center_position, detection.tip_position, Scalar(0, 215, 255), 2);
                        circle(overlay, detection.center_position, 6, Scalar(0, 215, 255), 2);
                        circle(overlay, detection.tip_position, 8, Scalar(0, 0, 255), -1);
                    }
                    const string camera_score = score_diagnostic == nullptr
                                                    ? "NO TIP"
                                                    : score_diagnostic->ring + " / " + score_diagnostic->score;
                    putText(overlay, camera_score, Point(20, 35), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
                    saveDiagnosticImage(event_directory, "camera_" + to_string(i) + "_overlay.jpg", overlay, camera["files"]);
                }

                manifest["cameras"].push_back(camera);
            }

            ofstream manifest_file(event_directory / "manifest.json");
            manifest_file << manifest.dump(2) << endl;
            log_info("DIAGNOSTIC_EVENT id=" + event_id +
                     " score=" + score_result.score +
                     " path=" + event_directory.string());
        }
        catch (const exception &error)
        {
            log_error("Failed to capture diagnostic event " + event_id + ": " + error.what());
        }
    }
}

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
    dart_processing::DartStateResult dart_result = dart_processing::processDartState(
        frames, background_frames, calibrations, motion_result.motion_finished, debug_mode);

    // Process scoring using the new scoring system
    score_processing::ScoreResult score_result = score_processing::processScore(background_frames, dart_result, calibrations, debug_mode);

    // Only return result if scoring system says it's valid (state changed)
    if (score_result.valid)
    {
        const uint64_t timestamp = chrono::duration_cast<chrono::milliseconds>(
                                       chrono::system_clock::now().time_since_epoch())
                                       .count();
        result.timestamp = timestamp;
        result.event_id = makeDiagnosticEventId(timestamp);
        result.dart_detected = true;
        result.score = score_result.score;
        result.position = score_result.pixel_position;
        result.board_position = score_result.dartboard_position;
        result.has_board_position = score_result.has_dartboard_position;
        result.confidence = score_result.confidence;
        result.camera_index = score_result.camera_index;

        if (debug_mode)
        {
            captureDiagnosticEvent(
                result.event_id,
                timestamp,
                frames,
                background_frames,
                dart_result,
                score_result,
                calibrations);
        }
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
                            "; motion uses all camera frames and scoring uses only cameras with READY orientation");
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
