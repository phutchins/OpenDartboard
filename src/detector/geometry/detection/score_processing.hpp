#pragma once

#include <opencv2/opencv.hpp>
#include <limits>
#include "dart_processing.hpp"
#include "../calibration/geometry_calibration.hpp"

using namespace cv;
using namespace std;

namespace score_processing
{
    struct CameraScoreDiagnostic
    {
        int camera_index = -1;
        bool tip_found = false;
        string ring = "MISS";
        string score = "MISS";
        float nearest_wire_distance = numeric_limits<float>::infinity();
        Point2f scoring_position = Point2f(-1, -1);
        Point2f board_position_mm = Point2f(-1, -1);
        bool board_entry_extrapolated = false;
        float visible_tip_radius_mm = numeric_limits<float>::quiet_NaN();
        float scoring_radius_mm = numeric_limits<float>::quiet_NaN();
        float tip_extension_mm = 0.0f;
    };

    // Score result for a single dart
    struct ScoreResult
    {
        string score = "MISS";                        // Dart score (S20, D5, T17, BULL, etc.)
        Point2f dartboard_position = Point2f(-1, -1); // Position on dartboard coordinate system
        bool has_dartboard_position = false;          // Whether normalized board coordinates are available
        Point2f pixel_position = Point2f(-1, -1);     // Original pixel position
        Point2f center_position = Point2f(-1, -1);    // Dart center position
        float confidence = 0.0f;                      // Scoring confidence
        int camera_index = -1;                        // Which camera detected this
        bool valid = false;                           // Is this a valid score result
        vector<CameraScoreDiagnostic> camera_diagnostics;
    };

    // Convert a camera pixel to a canonical board-centered coordinate system.
    // The outer double ellipse is approximately unit radius and wedge 20 is up.
    bool normalizeDartboardPosition(
        const Point2f &pixel,
        const DartboardCalibration &calibration,
        Point2f &normalized_position);

    // Process dart scoring from tip detection results
    ScoreResult processScore(
        const vector<Mat> &background_frames,
        const dart_processing::DartStateResult &dart_result,
        const vector<DartboardCalibration> &calib,
        bool debug_mode = false);

} // namespace score_processing
