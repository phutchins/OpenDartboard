#include "detector/geometry/detection/motion_processing.hpp"

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
    motion_processing::MotionParams params;

    // Lock down existing production detection defaults while adding telemetry.
    passed &= require(params.spike_threshold == 0.08, "default spike threshold changed");
    passed &= require(params.low_threshold == 0.001, "default low threshold changed");
    passed &= require(params.min_cameras_for_event == 2, "default minimum camera count changed");

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
