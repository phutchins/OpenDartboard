#include "detector/geometry/detection/motion_processing.hpp"

#include <iostream>
#include <limits>

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
    motion_processing::MotionParams params;

    // Lock down existing production detection defaults while adding telemetry.
    passed &= require(params.spike_threshold == 0.08, "default spike threshold changed");
    passed &= require(params.low_threshold == 0.001, "default low threshold changed");
    passed &= require(params.min_cameras_for_event == 2, "default minimum camera count changed");
    passed &= require(params.pretrigger_activity_ratio == 0.005,
                      "default pre-trigger activity ratio changed");
    passed &= require(motion_processing::isValidPretriggerActivityRatio(0.0),
                      "zero must be a valid pre-trigger activity ratio");
    passed &= require(motion_processing::isValidPretriggerActivityRatio(1.0),
                      "one must be a valid pre-trigger activity ratio");
    passed &= require(!motion_processing::isValidPretriggerActivityRatio(-0.001),
                      "negative pre-trigger activity ratios must be rejected");
    passed &= require(!motion_processing::isValidPretriggerActivityRatio(1.001),
                      "pre-trigger activity ratios above one must be rejected");
    passed &= require(!motion_processing::isValidPretriggerActivityRatio(
                          std::numeric_limits<double>::quiet_NaN()),
                      "non-finite pre-trigger activity ratios must be rejected");
    passed &= require(motion_processing::isValidTimingConfiguration(5, 250),
                      "low-latency timing configuration must validate");
    passed &= require(!motion_processing::isValidTimingConfiguration(0, 250),
                      "zero stability frames must be rejected");
    passed &= require(!motion_processing::isValidTimingConfiguration(5, -1),
                      "negative cooldown must be rejected");

    params.spike_window_frames = 10;
    params.processing_fps = 15.0;
    passed &= require(motion_processing::spikeWindowDurationMs(params) == 667,
                      "10 frames at 15 FPS must allow 667 ms");

    params.processing_fps = 20.0;
    passed &= require(motion_processing::spikeWindowDurationMs(params) == 500,
                      "10 frames at 20 FPS must allow 500 ms");

    params.processing_fps = 30.0;
    passed &= require(motion_processing::spikeWindowDurationMs(params) == 334,
                      "fractional frame windows must round up");

    params.processing_fps = 0.0;
    passed &= require(motion_processing::spikeWindowDurationMs(params) == 0,
                      "invalid FPS must disable the participation timeout");

    params.processing_fps = 15.0;
    params.spike_window_frames = 0;
    passed &= require(motion_processing::spikeWindowDurationMs(params) == 0,
                      "zero frame window must disable the participation timeout");

    return passed ? 0 : 1;
}
