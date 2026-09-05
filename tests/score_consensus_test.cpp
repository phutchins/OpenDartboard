#include "detector/geometry/detection/score_consensus.hpp"

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

    return passed ? 0 : 1;
}
