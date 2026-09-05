#include "motion_processing.hpp"
#include "utils.hpp"
#include "utils/streamer.hpp"
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

namespace motion_processing
{
    // Static storage for previous frames (maintains state between calls)
    static vector<Mat> previous_frames;
    static bool initialized = false;

    static unique_ptr<streamer> motion_streamer;
    static unique_ptr<streamer> motion2_streamer;

    // Event-based dart detection state
    static DartEventState current_state = DartEventState::IDLE;
    static chrono::steady_clock::time_point event_start_time;
    static chrono::steady_clock::time_point cooldown_start_time;
    static vector<bool> cameras_spiked;
    static vector<double> peak_motion_ratios;
    static vector<double> intensity_history;
    static int stable_frame_count = 0;

    // Debug-only pre-trigger telemetry accumulates peaks between rate-limited
    // log entries so a brief dart-sized pulse is not lost before the next log.
    static bool pretrigger_window_initialized = false;
    static chrono::steady_clock::time_point pretrigger_window_start;
    static vector<double> pretrigger_peak_ratios;
    static double pretrigger_peak_average = 0.0;
    static bool pretrigger_activity_seen = false;

    static string formatCameraRatios(const vector<MotionData> &motion_data)
    {
        ostringstream output;
        output << fixed << setprecision(6) << "[";
        for (size_t i = 0; i < motion_data.size(); i++)
        {
            if (i > 0)
                output << ",";
            output << "cam" << i << "=" << motion_data[i].motion_ratio;
        }
        output << "]";
        return output.str();
    }

    static string formatPeakRatios()
    {
        ostringstream output;
        output << fixed << setprecision(6) << "[";
        for (size_t i = 0; i < peak_motion_ratios.size(); i++)
        {
            if (i > 0)
                output << ",";
            output << "cam" << i << "=" << peak_motion_ratios[i];
        }
        output << "]";
        return output.str();
    }

    static string formatRatioValues(const vector<double> &ratios)
    {
        ostringstream output;
        output << fixed << setprecision(6) << "[";
        for (size_t i = 0; i < ratios.size(); ++i)
        {
            if (i > 0)
                output << ",";
            output << "cam" << i << "=" << ratios[i];
        }
        output << "]";
        return output.str();
    }

    static void resetPretriggerWindow(size_t cameraCount, chrono::steady_clock::time_point now)
    {
        pretrigger_window_initialized = true;
        pretrigger_window_start = now;
        pretrigger_peak_ratios.assign(cameraCount, 0.0);
        pretrigger_peak_average = 0.0;
        pretrigger_activity_seen = false;
    }

    static void recordPretriggerTelemetry(
        const vector<MotionData> &motionData,
        double currentAverage,
        chrono::steady_clock::time_point now,
        const MotionParams &params)
    {
        if (!pretrigger_window_initialized || pretrigger_peak_ratios.size() != motionData.size())
            resetPretriggerWindow(motionData.size(), now);

        double currentMax = 0.0;
        for (size_t index = 0; index < motionData.size(); ++index)
        {
            pretrigger_peak_ratios[index] = max(pretrigger_peak_ratios[index], motionData[index].motion_ratio);
            currentMax = max(currentMax, motionData[index].motion_ratio);
        }
        pretrigger_peak_average = max(pretrigger_peak_average, currentAverage);
        if (currentMax >= params.pretrigger_activity_ratio)
            pretrigger_activity_seen = true;

        const auto elapsedMs = chrono::duration_cast<chrono::milliseconds>(now - pretrigger_window_start).count();
        if (elapsedMs < max(1, params.pretrigger_log_interval_ms))
            return;

        if (pretrigger_activity_seen)
        {
            const double peakMax = pretrigger_peak_ratios.empty()
                                       ? 0.0
                                       : *max_element(pretrigger_peak_ratios.begin(), pretrigger_peak_ratios.end());
            log_debug("MOTION_PRETRIGGER state=IDLE peak_ratios=" + formatRatioValues(pretrigger_peak_ratios) +
                      " peak_average=" + to_string(pretrigger_peak_average) +
                      " peak_max=" + to_string(peakMax) +
                      " spike_threshold=" + to_string(params.spike_threshold) +
                      " per_camera_threshold=" + to_string(params.threshold_ratio) +
                      " activity_floor=" + to_string(params.pretrigger_activity_ratio) +
                      " window_ms=" + to_string(elapsedMs));
        }
        resetPretriggerWindow(motionData.size(), now);
    }

    vector<MotionData> detectMotion(const vector<Mat> &current_frames, const vector<Mat> &background_frames, bool debug_mode, const MotionParams &params)
    {
        // Initialize previous frames on first run
        if (!initialized || previous_frames.size() != current_frames.size())
        {
            previous_frames.clear();

            // check if we have any frames to initialize
            if (current_frames.size() != 3)
            {
                log_warning("Motion processing initialized with " + to_string(current_frames.size()) + " cameras, but expected 3 cameras. Skipping initialization.");
                return vector<MotionData>(current_frames.size());
            }

            for (const auto &frame : current_frames)
            {
                previous_frames.push_back(frame.clone());
            }

            initialized = true;
            log_debug("Motion processing initialized with " + to_string(current_frames.size()) + " cameras");

            if (debug_mode)
            {
                motion_streamer = make_unique<streamer>(8082, 15);
                motion2_streamer = make_unique<streamer>(8083, 15);
            }

            // Return no motion on first frame
            return vector<MotionData>(current_frames.size());
        }

        // Detect current motion using our processing - now with actual data
        vector<MotionData> motion_data(current_frames.size());
        vector<Mat> motion_viz_frames;
        vector<Mat> motion_viz_frames2;

        // Motion detection for each camera
        for (size_t i = 0; i < current_frames.size() && i < previous_frames.size(); i++)
        {
            if (current_frames[i].empty() || previous_frames[i].empty() || background_frames[i].empty())
                continue;

            // Convert to grayscale
            Mat prev_gray, curr_gray;
            cvtColor(previous_frames[i], prev_gray, COLOR_BGR2GRAY);
            cvtColor(current_frames[i], curr_gray, COLOR_BGR2GRAY);

            // clean up frames so there no noise
            GaussianBlur(prev_gray, prev_gray, Size(params.blur_kernel_size, params.blur_kernel_size), params.blur_sigma_x, params.blur_sigma_y, BORDER_DEFAULT);
            GaussianBlur(curr_gray, curr_gray, Size(params.blur_kernel_size, params.blur_kernel_size), params.blur_sigma_x, params.blur_sigma_y, BORDER_DEFAULT);

            // Calculate absolute difference
            Mat diff;
            absdiff(prev_gray, curr_gray, diff);

            // Apply threshold
            Mat thresh;
            threshold(diff, thresh, params.binary_threshold, 255, THRESH_BINARY);

            // Apply morphological operations to reduce noise
            Mat kernel = getStructuringElement(params.morph_type, Size(params.morph_kernel_size, params.morph_kernel_size));
            morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

            // Debug: Save motion detection images
            if (debug_mode)
            {
                system("mkdir -p debug_frames/motion_processing");
                imwrite("debug_frames/motion_processing/diff_cam_" + to_string(i) + ".jpg", diff);
                imwrite("debug_frames/motion_processing/thresh_cam_" + to_string(i) + ".jpg", thresh);

                // this just an example of how to visualize motion
                // this should do MORE;

                motion_viz_frames.push_back(diff);
                motion_viz_frames2.push_back(thresh);
            }

            // Count motion pixels and calculate ratio - CAPTURE THE REAL DATA!
            int motion_pixels = countNonZero(thresh);
            int total_pixels = thresh.rows * thresh.cols;
            double motion_ratio = (double)motion_pixels / total_pixels;

            // Store all the motion data
            motion_data[i].motion_pixels = motion_pixels;
            motion_data[i].total_pixels = total_pixels;
            motion_data[i].motion_ratio = motion_ratio;
            motion_data[i].motion_detected = (motion_ratio > params.threshold_ratio);
        }

        // Update previous frames for next iteration
        for (size_t i = 0; i < current_frames.size(); i++)
        {
            previous_frames[i] = current_frames[i].clone();
        }

        if (debug_mode)
        {
            Mat combined_motion = debug::createCombinedFrame(motion_viz_frames, "diff");
            motion_streamer->push(combined_motion);
            Mat combined_motion2 = debug::createCombinedFrame(motion_viz_frames2, "thresh");
            motion2_streamer->push(combined_motion2);
        }

        return motion_data;
    }

    MotionResult processMotion(const vector<Mat> &current_frames, const vector<Mat> &background_frames, bool debug_mode, const MotionParams &params)
    {
        MotionResult result;

        // Get motion data from all cameras
        vector<MotionData> motion_data = detectMotion(current_frames, background_frames, debug_mode, params);
        auto now = chrono::steady_clock::now();

        if (motion_data.empty())
            return result;

        // Calculate overall motion intensity (average across all cameras)
        double total_intensity = 0.0;
        int cameras_with_motion = 0;

        for (const auto &data : motion_data)
        {
            total_intensity += data.motion_ratio;
            if (data.motion_detected)
                cameras_with_motion++;
        }

        double current_intensity = total_intensity / motion_data.size();

        if (debug_mode && current_state == DartEventState::IDLE &&
            current_intensity <= params.spike_threshold)
        {
            recordPretriggerTelemetry(motion_data, current_intensity, now, params);
        }

        // Initialize cameras_spiked vector if needed
        if (cameras_spiked.size() != motion_data.size())
        {
            cameras_spiked.resize(motion_data.size(), false);
            peak_motion_ratios.resize(motion_data.size(), 0.0);
        }

        if (current_state != DartEventState::IDLE)
        {
            for (size_t i = 0; i < motion_data.size(); i++)
                peak_motion_ratios[i] = max(peak_motion_ratios[i], motion_data[i].motion_ratio);
        }

        // Calculate detection duration if we're in an active state
        if (current_state != DartEventState::IDLE && current_state != DartEventState::COOLDOWN)
        {
            result.detection_duration_ms = chrono::duration_cast<chrono::milliseconds>(now - event_start_time).count();
        }

        // State machine for dart event detection
        switch (current_state)
        {
        case DartEventState::IDLE:
        {
            // Look for motion spike that could indicate dart hit
            if (current_intensity > params.spike_threshold)
            {
                current_state = DartEventState::SPIKE_DETECTED;
                event_start_time = now;
                fill(cameras_spiked.begin(), cameras_spiked.end(), false);
                for (size_t i = 0; i < motion_data.size(); i++)
                    peak_motion_ratios[i] = motion_data[i].motion_ratio;
                intensity_history.clear();
                stable_frame_count = 0;
                resetPretriggerWindow(motion_data.size(), now);

                // Mark cameras that are spiking
                for (size_t i = 0; i < motion_data.size(); i++)
                {
                    if (motion_data[i].motion_ratio > params.spike_threshold)
                    {
                        cameras_spiked[i] = true;
                    }
                }

                if (debug_mode)
                {
                    log_debug("MOTION_EVENT_START average=" + to_string(current_intensity) +
                              " ratios=" + formatCameraRatios(motion_data) +
                              " spike_threshold=" + to_string(params.spike_threshold));
                }
            }
            break;
        }

        case DartEventState::SPIKE_DETECTED:
        {
            // Continue tracking which cameras spike during the event window
            for (size_t i = 0; i < motion_data.size(); i++)
            {
                if (motion_data[i].motion_ratio > params.spike_threshold)
                {
                    cameras_spiked[i] = true;
                }
            }

            int cameras_that_spiked = count(cameras_spiked.begin(), cameras_spiked.end(), true);
            auto event_duration = chrono::duration_cast<chrono::milliseconds>(now - event_start_time).count();
            const int spike_window_duration_ms = spikeWindowDurationMs(params);

            // Check if we have enough camera participation and motion is settling
            if (cameras_that_spiked >= params.min_cameras_for_event &&
                current_intensity <= params.low_threshold)
            {
                current_state = DartEventState::STABILIZING;
                stable_frame_count = 1;
                if (debug_mode)
                {
                    log_debug("MOTION_EVENT_STABILIZING cameras=" + to_string(cameras_that_spiked) +
                              "/" + to_string(params.min_cameras_for_event) +
                              " average=" + to_string(current_intensity) +
                              " peaks=" + formatPeakRatios());
                }
            }
            // Timeout if event takes too long or insufficient participation
            else if (event_duration > params.max_event_duration_ms ||
                     (spike_window_duration_ms > 0 && event_duration > spike_window_duration_ms &&
                      cameras_that_spiked < params.min_cameras_for_event))
            {
                current_state = DartEventState::IDLE;
                log_warning("DART EVENT: Event timeout or insufficient cameras (" + to_string(cameras_that_spiked) + "/" + to_string(params.min_cameras_for_event) + ") after " + to_string(event_duration) + "ms");
                if (debug_mode)
                {
                    log_debug("MOTION_EVENT_REJECTED reason=timeout_or_insufficient_cameras cameras=" +
                              to_string(cameras_that_spiked) + "/" + to_string(params.min_cameras_for_event) +
                              " average=" + to_string(current_intensity) +
                              " spike_window_ms=" + to_string(spike_window_duration_ms) +
                              " peaks=" + formatPeakRatios());
                }
            }
            break;
        }

        case DartEventState::STABILIZING:
        {
            // Track motion intensity to confirm stability
            if (current_intensity <= params.low_threshold)
            {
                stable_frame_count++;
                if (stable_frame_count >= params.stability_frames)
                {
                    // Motion event finished!
                    current_state = DartEventState::END;
                    result.motion_finished = true;
                    int cameras_that_spiked = count(cameras_spiked.begin(), cameras_spiked.end(), true);
                    if (debug_mode)
                    {
                        log_debug("MOTION_EVENT_CONFIRMED cameras=" + to_string(cameras_that_spiked) +
                                  "/" + to_string(params.min_cameras_for_event) +
                                  " duration_ms=" + to_string(result.detection_duration_ms) +
                                  " peaks=" + formatPeakRatios());
                    }
                }
            }
            else
            {
                // Motion increased - could be dart removal or false positive
                if (current_intensity > params.spike_threshold)
                {
                    // Big spike during stabilization - probably dart removal, reset
                    current_state = DartEventState::IDLE;
                    if (debug_mode)
                    {
                        log_debug("MOTION_EVENT_REJECTED reason=motion_during_stabilization average=" +
                                  to_string(current_intensity) + " peaks=" + formatPeakRatios());
                    }
                }
                else
                {
                    // Small increase, reset stability counter but keep trying
                    stable_frame_count = 0;
                }
                break;
            }
        }

        case DartEventState::END:
        {
            // Transition to cooldown immediately
            current_state = DartEventState::COOLDOWN;
            cooldown_start_time = now;
            result.motion_finished = true;

            break;
        }

        case DartEventState::COOLDOWN:
        {
            auto cooldown_elapsed = chrono::duration_cast<chrono::milliseconds>(now - cooldown_start_time).count();
            if (cooldown_elapsed >= params.cooldown_period_ms)
            {
                current_state = DartEventState::IDLE;
            }
            else if (debug_mode && current_intensity > params.spike_threshold)
            {
                log_debug("COOLDOWN: Motion during cooldown period - intensity: " + to_string(current_intensity) + ", remaining: " + to_string(params.cooldown_period_ms - cooldown_elapsed) + "ms");
            }
            break;
        }
        }

        result.current_state = current_state;
        return result;
    }

} // namespace motion_processing
