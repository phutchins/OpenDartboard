#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Result structure with all detection data
struct DetectorResult
{
    bool dart_detected = false;
    string score = "";
    Point2f position{-1, -1}; // Dart position for overlay/preview
    Point2f board_position{0, 0}; // Normalized dartboard position (outer double ring ~= unit ellipse)
    bool has_board_position = false;
    float confidence = 0.0f;
    int camera_index = -1;
    uint64_t timestamp = 0;
    string event_id = ""; // Stable identifier for diagnostic capture and correction labels
    string previous_board_state = "";
    string current_board_state = "";

    // Metadata for debugging/analysis
    bool motion_detected = false;
    int processing_time_ms = 0;

    // Easy boolean check
    operator bool() const { return dart_detected; }
};

// Abstract interface for any dart detection method
class DetectorInterface
{
public:
    virtual ~DetectorInterface() = default;

    // Initialize the detector
    virtual bool initialize(vector<VideoCapture> &cameras) = 0;

    // Whether the detector is ready
    virtual bool isInitialized() const = 0;

    // Process frames and return detection results
    virtual DetectorResult process(const vector<Mat> &frames) = 0;
};
