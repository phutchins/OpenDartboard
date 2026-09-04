#include "scorer/scorer.hpp"
#include "utils/args.hpp"
#include "utils/debug.hpp"
#include "utils/signals.hpp"
#include "utils/logging.hpp"
#include "utils/autocam.hpp"
#include "detector/geometry/detection/motion_processing.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>

using namespace std;

// version string for the application
const string version = APP_VERSION;

int main(int argc, char **argv)
{
  // Check for help or version flags first
  if (hasFlag(argc, argv, "--version"))
    debug::printVersionAndExit(version);
  if (hasFlag(argc, argv, "--help"))
    debug::printHelpAndExit();

  // Parse command line arguments with defaults
  string model_path = getArg(argc, argv, "--model", "/usr/local/share/opendartboard/models/dart.param");
  bool useAuto = hasFlag(argc, argv, "--autocams");
  int width = getArg(argc, argv, "--width", 1280);
  int height = getArg(argc, argv, "--height", 720);
  int fps = getArg(argc, argv, "--fps", 15);
  bool debug_mode = hasFlag(argc, argv, "--debug") || hasFlag(argc, argv, "-d");
  bool quite_mode = hasFlag(argc, argv, "--quiet") || hasFlag(argc, argv, "-q");

  motion_processing::MotionParams motion_params;
  motion_params.spike_threshold = getArg(argc, argv, "--motion-spike-threshold", motion_params.spike_threshold);
  motion_params.low_threshold = getArg(argc, argv, "--motion-low-threshold", motion_params.low_threshold);
  motion_params.min_cameras_for_event = getArg(argc, argv, "--motion-min-cameras", motion_params.min_cameras_for_event);

  if (!std::isfinite(motion_params.spike_threshold) || motion_params.spike_threshold <= 0.0 || motion_params.spike_threshold > 1.0 ||
      !std::isfinite(motion_params.low_threshold) || motion_params.low_threshold < 0.0 || motion_params.low_threshold >= motion_params.spike_threshold ||
      motion_params.min_cameras_for_event < 1)
  {
    cerr << "Invalid motion configuration: require 0 <= low threshold < spike threshold <= 1 and at least one camera" << endl;
    return 2;
  }

  // Replace boolean flag with detector type string
  string detector_type = getArg(argc, argv, "--detector", "geometry");

  // set log level based on debug mode
  if (debug_mode)
  {
    logging::setLogLevel(logging::LogLevel::DEBUG); // Show everything
    log_info("Debug mode enabled - showing all log messages");
  }
  else if (quite_mode)
  {
    logging::setLogLevel(logging::LogLevel::ERROR); // Only errors
    log_info("Quiet mode enabled - showing only error messages");
  }

  // Print startup and configuration information
  debug::printStartup("OpenDartboard", version);

  // setup cams
  vector<string> cams;
  if (useAuto)
  {
    cams = autocam::detectAndLock(/*max*/ 3, width, height, fps);
  }
  else
  {
    cams = getArgVector(argc, argv, "--cams", "/dev/video0,/dev/video1,/dev/video2");
  }

  debug::printConfig(width, height, fps, model_path, cams);

  if (motion_params.min_cameras_for_event > static_cast<int>(cams.size()))
  {
    cerr << "Invalid motion configuration: --motion-min-cameras (" << motion_params.min_cameras_for_event
         << ") exceeds configured camera count (" << cams.size() << ")" << endl;
    return 2;
  }

  log_info("MOTION_CONFIG spike_threshold=" + to_string(motion_params.spike_threshold) +
           " low_threshold=" + to_string(motion_params.low_threshold) +
           " min_cameras=" + to_string(motion_params.min_cameras_for_event));

  // Initialise the scorer with debug mode if requested
  Scorer scorer(model_path, width, height, fps, cams, debug_mode, detector_type, motion_params);

  // Register signal handlers with a lambda to stop the scorer
  signals::setupSignalHandlers([&scorer]()
                               { scorer.stop(); });

  // Start the scorer processing in background thread
  scorer.run();

  // Best practice: wait for the scorer thread to finish
  return 0;
}
