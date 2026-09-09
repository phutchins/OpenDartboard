#include "detector/geometry/detection/score_consensus.hpp"

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
    using score_processing::Ring;
    bool passed = true;

    const auto tripleConsensus = score_processing::selectRingConsensus(
        {Ring::TRIPLE, Ring::SINGLE, Ring::TRIPLE});
    passed &= require(tripleConsensus.valid &&
                          tripleConsensus.ring == Ring::TRIPLE &&
                          tripleConsensus.votes == 2,
                      "two geometry cameras must outvote one incorrect ring");
    passed &= require(
        score_processing::applyRingConsensus(tripleConsensus.ring, "S20") == "T20",
        "ring consensus must preserve the orientation-ready wedge number");

    const auto singleConsensus = score_processing::selectRingConsensus(
        {Ring::SINGLE, Ring::SINGLE, Ring::SINGLE});
    passed &= require(singleConsensus.valid && singleConsensus.votes == 3,
                      "unanimous single-ring observations must form consensus");
    passed &= require(
        score_processing::applyRingConsensus(singleConsensus.ring, "T18") == "S18",
        "ring consensus must be able to correct an incorrect multiplier");

    const auto tie = score_processing::selectRingConsensus(
        {Ring::SINGLE, Ring::TRIPLE});
    passed &= require(!tie.valid,
                      "a one-to-one ring disagreement must not be treated as consensus");
    passed &= require(
        score_processing::applyRingConsensus(Ring::TRIPLE, "MISS") == "MISS",
        "ring consensus cannot invent a wedge without an oriented score");

    passed &= require(
        score_processing::selectMostSeparatedCandidate({2.80f, 0.44f}) == 0,
        "wedge tie-break must retain the camera farther from its wire");
    passed &= require(
        score_processing::selectMostSeparatedCandidate({1.08f, 4.21f}) == 1,
        "wedge tie-break must not depend on camera order");
    passed &= require(
        score_processing::selectMostSeparatedCandidate(
            {std::numeric_limits<float>::quiet_NaN(), 3.0f}) == 1,
        "invalid boundary distances must not beat measured candidates");

    passed &= require(
        !score_processing::hasOutsideBoardConsensus(1),
        "one outside-board observation must not override valid camera scores");
    passed &= require(
        score_processing::hasOutsideBoardConsensus(2),
        "two outside-board observations must override one stale in-board contour");

    return passed ? 0 : 1;
}
