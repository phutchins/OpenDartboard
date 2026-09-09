#include "score_processing.hpp"
#include "dart_geometry.hpp"
#include "ellipse_metrics.hpp"
#include "score_consensus.hpp"
#include "utils.hpp"
#include "utils/streamer.hpp"
#include "../calibration/geometry_calibration.hpp"
#include "../calibration/board_geometry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace cv;
using namespace std;

namespace score_processing
{

    static bool initialized = false;
    static unique_ptr<streamer> point_on_screen_streamer;

    static string formatEllipseTelemetry(
        const string &name,
        const Point2f &point,
        const RotatedRect &ellipse)
    {
        ostringstream output;
        output << fixed << setprecision(6)
               << name << "_center_x=" << ellipse.center.x << " "
               << name << "_center_y=" << ellipse.center.y << " "
               << name << "_width=" << ellipse.size.width << " "
               << name << "_height=" << ellipse.size.height << " "
               << name << "_angle_deg=" << ellipse.angle << " "
               << name << "_radius=" << normalizedEllipseRadius(point, ellipse) << " "
               << name << "_margin=" << signedNormalizedEllipseMargin(point, ellipse);
        return output.str();
    }

    static void logRingClassificationTelemetry(
        size_t camera_index,
        const Point2f &point,
        const DartboardCalibration &calibration)
    {
        ostringstream prefix;
        prefix << fixed << setprecision(3)
               << "RING_CLASSIFICATION camera=" << camera_index
               << " point_x=" << point.x
               << " point_y=" << point.y << " ";

        log_debug(prefix.str() +
                  formatEllipseTelemetry("inner_triple", point, calibration.ellipses.innerTripleEllipse) + " " +
                  formatEllipseTelemetry("outer_triple", point, calibration.ellipses.outerTripleEllipse) + " " +
                  formatEllipseTelemetry("inner_double", point, calibration.ellipses.innerDoubleEllipse) + " " +
                  formatEllipseTelemetry("outer_double", point, calibration.ellipses.outerDoubleEllipse));
    }

    bool normalizeDartboardPosition(
        const Point2f &pixel,
        const DartboardCalibration &calib,
        Point2f &normalized_position)
    {
        if (!geometry_calibration::hasValidGeometry(calib) || !geometry_calibration::hasValidOrientation(calib))
            return false;
        Point2f boardPosition;
        if (!board_geometry::imageToBoard(calib.boardTransform, pixel, boardPosition))
            return false;
        normalized_position = boardPosition / 170.0f;
        return std::isfinite(normalized_position.x) && std::isfinite(normalized_position.y);
    }

    // Helper: Check if point is inside ellipse (pure math)
    bool isPointInEllipse(Point2f point, const RotatedRect &ellipse)
    {
        if (!std::isfinite(ellipse.center.x) || !std::isfinite(ellipse.center.y) ||
            !std::isfinite(ellipse.size.width) || !std::isfinite(ellipse.size.height) ||
            ellipse.size.width <= 0.0f || ellipse.size.height <= 0.0f)
            return false;

        Point2f center = ellipse.center;
        Point2f relative = point - center;

        // Rotate point to ellipse coordinate system
        float angle_rad = -ellipse.angle * CV_PI / 180.0f;
        float cos_a = cos(angle_rad);
        float sin_a = sin(angle_rad);

        Point2f rotated(
            relative.x * cos_a - relative.y * sin_a,
            relative.x * sin_a + relative.y * cos_a);

        // Ellipse equation: (x/a)² + (y/b)² <= 1
        float a = ellipse.size.width / 2.0f;
        float b = ellipse.size.height / 2.0f;

        return (rotated.x * rotated.x) / (a * a) + (rotated.y * rotated.y) / (b * b) <= 1.0f;
    }

    static Ring classifyRingAtPoint(Point2f pixel, const DartboardCalibration &calib)
    {
        if (!geometry_calibration::hasValidGeometry(calib))
        {
            log_debug("SCORE: Invalid calibration data");
            return Ring::MISS;
        }

        Point2f boardPosition;
        if (!board_geometry::imageToBoard(calib.boardTransform, pixel, boardPosition))
            return Ring::MISS;
        const float radius = norm(boardPosition);

        if (radius <= 6.35f)
        {
            log_debug("SCORE: Point in INNER BULL");
            return Ring::INNER_BULL;
        }

        if (radius <= 15.9f)
        {
            log_debug("SCORE: Point in OUTER BULL");
            return Ring::OUTER_BULL;
        }

        log_debug("SCORE: Canonical radius = " + log_string(radius) + "mm");
        if (radius >= 162.0f && radius <= 170.0f)
        {
            log_debug("SCORE: Ring type = DOUBLE");
            return Ring::DOUBLE;
        }
        if (radius >= 99.0f && radius <= 107.0f)
        {
            log_debug("SCORE: Ring type = TRIPLE");
            return Ring::TRIPLE;
        }
        if (radius <= 170.0f)
        {
            log_debug("SCORE: Ring type = SINGLE");
            return Ring::SINGLE;
        }

        log_debug("SCORE: Point outside dartboard");
        return Ring::MISS;
    }

    static int getWedgeAtPoint(Point2f pixel, const DartboardCalibration &calib)
    {
        if (!geometry_calibration::hasValidOrientation(calib))
        {
            log_debug("SCORE: Invalid orientation data; refusing to guess a wedge");
            return -1;
        }

        Point2f boardPosition;
        if (!board_geometry::imageToBoard(calib.boardTransform, pixel, boardPosition))
            return -1;
        float angleDegrees = atan2(boardPosition.y, boardPosition.x) * 180.0f / static_cast<float>(CV_PI);
        float fromFirstBoundary = angleDegrees - (-99.0f);
        while (fromFirstBoundary < 0.0f)
            fromFirstBoundary += 360.0f;
        while (fromFirstBoundary >= 360.0f)
            fromFirstBoundary -= 360.0f;
        const int wedgeIndex = min(19, static_cast<int>(floor(fromFirstBoundary / 18.0f)));
        const array<int, 20> dartboardNumbers{{20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
                                                3, 19, 7, 16, 8, 11, 14, 9, 12, 5}};
        log_debug("SCORE: Canonical angle=" + log_string(angleDegrees) +
                  " wedge=" + log_string(dartboardNumbers[wedgeIndex]));
        return dartboardNumbers[wedgeIndex];
    }

    static string getScoreForRingAtPoint(
        Ring ring,
        Point2f pixel,
        const DartboardCalibration &calib)
    {
        if (ring == Ring::INNER_BULL)
            return "BULL";
        if (ring == Ring::OUTER_BULL)
            return "OUTER";
        if (!ringRequiresWedge(ring))
            return "MISS";

        const int wedge = getWedgeAtPoint(pixel, calib);
        if (wedge < 0)
            return "MISS";
        return string(ringToString(ring)) + to_string(wedge);
    }

    ScoreResult processScore(const vector<Mat> &background_frames, const dart_processing::DartStateResult &dart_result, const vector<DartboardCalibration> &calibrations, bool debug_mode)
    {

        if (!initialized)
        {
            log_debug("SCORE: Initializing score processing");
            point_on_screen_streamer = make_unique<streamer>(8088, 1);
            initialized = true;
        }

        ScoreResult result;

        if (dart_result.previous_state == dart_result.current_state)
        {
            // No state change, return invalid result
            result.valid = false;
            return result;
        }

        // State changed - now process the scoring
        switch (dart_result.current_state)
        {
        case dart_processing::DartBoardState::CLEAN:
            result.score = "END";
            result.confidence = 1.0f;
            result.camera_index = -1;
            result.valid = true;
            break;

        case dart_processing::DartBoardState::DART_1:
        case dart_processing::DartBoardState::DART_2:
        case dart_processing::DartBoardState::DART_3:
            // Collect scores from all cameras with detected tips
            struct CameraScoreCandidate
            {
                string score;
                int camera_index;
                float nearest_wire_distance;
            };
            vector<CameraScoreCandidate> camera_scores;
            vector<Ring> ring_observations;
            vector<Mat> points_on_screen;            // For debug images

            const size_t camera_count = min({dart_result.camera_results.size(), calibrations.size(), background_frames.size()});
            for (size_t i = 0; i < camera_count; i++)
            {
                log_debug("-------");
                if (debug_mode && dart_result.camera_results[i].tip_found)
                {
                    logRingClassificationTelemetry(
                        i,
                        dart_result.camera_results[i].tip_position,
                        calibrations[i]);
                }
                Ring ring = Ring::MISS;
                if (dart_result.camera_results[i].tip_found &&
                    geometry_calibration::hasCompleteRingGeometry(calibrations[i]))
                {
                    ring = classifyRingAtPoint(
                        dart_result.camera_results[i].tip_position,
                        calibrations[i]);
                    if (ring != Ring::MISS)
                        ring_observations.push_back(ring);
                }
                else if (dart_result.camera_results[i].tip_found && debug_mode)
                {
                    log_warning(
                        "RING_OBSERVATION_REJECTED camera=" + to_string(i) +
                        " reason=incomplete_ring_geometry");
                }
                string score_test = getScoreForRingAtPoint(
                    ring,
                    dart_result.camera_results[i].tip_position,
                    calibrations[i]);
                CameraScoreDiagnostic camera_diagnostic;
                camera_diagnostic.camera_index = static_cast<int>(i);
                camera_diagnostic.tip_found = dart_result.camera_results[i].tip_found;
                camera_diagnostic.ring = ringToString(ring);
                camera_diagnostic.score = score_test;
                log_debug("-------");

                // print image
                if (debug_mode)
                {
                    const string display_score =
                        score_test != "MISS" || !ringRequiresWedge(ring)
                            ? score_test
                            : string(ringToString(ring)) + "?";
                    log_warning("Camera " + to_string(i) +
                                " ring: " + ringToString(ring) +
                                " score: " + score_test);

                    // just draw the point on the screen
                    Mat some_mat = background_frames[i].clone();
                    circle(some_mat, dart_result.camera_results[i].tip_position, 5, Scalar(0, 255, 0), -1);
                    putText(some_mat, display_score, dart_result.camera_results[i].tip_position + Point2f(10, 10),
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                    system("mkdir -p debug_frames/score_processing");
                    imwrite("debug_frames/score_processing/point_on_screen" + to_string(i) + ".jpg", some_mat);

                    points_on_screen.push_back(some_mat);
                }

                if (dart_result.camera_results[i].tip_found && score_test != "MISS")
                {
                    float nearest_wire_distance = numeric_limits<float>::infinity();
                    if (score_test != "BULL" && score_test != "OUTER")
                    {
                        nearest_wire_distance = board_geometry::nearestWireDistancePixels(
                            dart_result.camera_results[i].tip_position,
                            Point2f(calibrations[i].bullCenter),
                            calibrations[i].wires.wireEndpoints);
                    }
                    camera_scores.push_back({score_test, static_cast<int>(i), nearest_wire_distance});
                    camera_diagnostic.nearest_wire_distance = nearest_wire_distance;
                    log_debug("SCORE_BOUNDARY_PROXIMITY camera=" + to_string(i) +
                              " distance_px=" + to_string(nearest_wire_distance));
                }
                result.camera_diagnostics.push_back(camera_diagnostic);
            }

            if (debug_mode)
            {
                try
                {
                    point_on_screen_streamer->push(debug::createCombinedFrame(points_on_screen, "points_on_screen"));
                }
                catch (const std::exception &e)
                {
                    log_error("Failed to push points_on_screen frame: " + string(e.what()));
                }
            }

            if (!camera_scores.empty())
            {
                // Consensus scoring logic
                string final_score;
                int best_camera = -1;
                float selected_confidence = 0.0f;

                // Count occurrences of each score
                map<string, vector<int>> score_cameras;
                for (const auto &candidate : camera_scores)
                {
                    score_cameras[candidate.score].push_back(candidate.camera_index);
                }

                // Look for consensus (2+ cameras agreeing)
                string consensus_score;
                int max_consensus = 0;
                for (const auto &[score, cameras] : score_cameras)
                {
                    if (cameras.size() >= 2 && cameras.size() > max_consensus)
                    {
                        consensus_score = score;
                        max_consensus = cameras.size();
                    }
                }

                if (!consensus_score.empty())
                {
                    // Use consensus score, pick first camera from the group
                    final_score = consensus_score;
                    best_camera = score_cameras[consensus_score][0];
                    selected_confidence = 0.9f;
                    log_info("Consensus score: " + final_score + " from " + to_string(max_consensus) + " cameras");
                }
                else
                {
                    const auto ring_consensus = selectRingConsensus(ring_observations);
                    const string ring_adjusted_score = ring_consensus.valid
                                                           ? applyRingConsensus(
                                                                 ring_consensus.ring,
                                                                 camera_scores[0].score)
                                                           : "MISS";

                    best_camera = camera_scores[0].camera_index;
                    if (ring_adjusted_score != "MISS")
                    {
                        final_score = ring_adjusted_score;
                        selected_confidence = board_geometry::singleCameraBoundaryConfidence(
                            camera_scores[0].nearest_wire_distance,
                            0.85f);
                        log_info("Ring consensus score: " + final_score +
                                 " from " + to_string(ring_consensus.votes) +
                                 " geometry cameras; wedge from camera " +
                                 to_string(best_camera));
                        log_debug("SCORE_CONFIDENCE mode=RING_CONSENSUS camera=" +
                                  to_string(best_camera) +
                                  " confidence=" + to_string(selected_confidence) +
                                  " ring=" + ringToString(ring_consensus.ring) +
                                  " ring_votes=" + to_string(ring_consensus.votes) +
                                  " nearest_wedge_wire_distance_px=" +
                                  to_string(camera_scores[0].nearest_wire_distance));
                    }
                    else
                    {
                        final_score = camera_scores[0].score;
                        selected_confidence = board_geometry::singleCameraBoundaryConfidence(
                            camera_scores[0].nearest_wire_distance);
                        log_info("No consensus, using single camera score: " + final_score + " from camera " + to_string(best_camera));
                        log_debug("SCORE_CONFIDENCE mode=SINGLE_CAMERA camera=" + to_string(best_camera) +
                                  " confidence=" + to_string(selected_confidence) +
                                  " nearest_wire_distance_px=" + to_string(camera_scores[0].nearest_wire_distance) +
                                  " reason=no_second_ready_camera_agreement");
                    }
                }

                result.score = final_score;
                result.pixel_position = dart_result.camera_results[best_camera].tip_position;
                result.center_position = dart_result.camera_results[best_camera].center_position;
                result.has_dartboard_position = normalizeDartboardPosition(result.pixel_position, calibrations[best_camera], result.dartboard_position);
                if (!result.has_dartboard_position)
                    result.dartboard_position = Point2f(-1, -1);
                result.confidence = selected_confidence;
                result.camera_index = best_camera;
                result.valid = true;
            }
            else
            {
                const size_t cameras_with_tips = count_if(
                    dart_result.camera_results.begin(),
                    dart_result.camera_results.end(),
                    [](const dart_processing::CameraDetectionResult &camera) {
                        return camera.tip_found;
                    });
                const size_t cameras_supporting_persistent_change = count_if(
                    dart_result.camera_results.begin(),
                    dart_result.camera_results.end(),
                    [](const dart_processing::CameraDetectionResult &camera) {
                        return camera.supports_persistent_change;
                    });
                if (dart_geometry::hasSufficientMissEvidence(
                        cameras_with_tips,
                        cameras_supporting_persistent_change))
                {
                    result.score = "MISS";
                    result.confidence = 0.5f;
                    result.camera_index = -1;
                    result.valid = true;

                    // Keep a directional position for misses when the
                    // orientation-ready camera saw the dart. The client can
                    // then place a zero-score hit near the correct numbered
                    // sector instead of dropping it at an anonymous location.
                    for (size_t i = 0; i < camera_count; ++i)
                    {
                        if (!dart_result.camera_results[i].tip_found ||
                            !geometry_calibration::hasValidOrientation(calibrations[i]))
                            continue;
                        Point2f normalized_position;
                        if (!normalizeDartboardPosition(
                                dart_result.camera_results[i].tip_position,
                                calibrations[i],
                                normalized_position))
                            continue;
                        result.pixel_position = dart_result.camera_results[i].tip_position;
                        result.center_position = dart_result.camera_results[i].center_position;
                        result.dartboard_position = normalized_position;
                        result.has_dartboard_position = true;
                        result.camera_index = static_cast<int>(i);
                        break;
                    }
                    if (debug_mode)
                        log_warning("State changed with corroborated tip evidence but no in-board score; reporting MISS");
                }
                else if (debug_mode)
                {
                    log_warning("SCORE_EVENT_REJECTED reason=insufficient_miss_evidence cameras_with_tips=" +
                                to_string(cameras_with_tips) +
                                " cameras_supporting_persistent_change=" +
                                to_string(cameras_supporting_persistent_change));
                }
            }
            break;
        }

        return result;
    }

} // namespace score_processing
