#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace score_processing
{
    enum class Ring
    {
        MISS = 0,
        SINGLE,
        DOUBLE,
        TRIPLE,
        OUTER_BULL,
        INNER_BULL
    };

    inline const char *ringToString(Ring ring)
    {
        switch (ring)
        {
        case Ring::SINGLE:
            return "S";
        case Ring::DOUBLE:
            return "D";
        case Ring::TRIPLE:
            return "T";
        case Ring::OUTER_BULL:
            return "OUTER";
        case Ring::INNER_BULL:
            return "BULL";
        case Ring::MISS:
        default:
            return "MISS";
        }
    }

    inline bool ringRequiresWedge(Ring ring)
    {
        return ring == Ring::SINGLE || ring == Ring::DOUBLE || ring == Ring::TRIPLE;
    }

    struct RingConsensus
    {
        Ring ring = Ring::MISS;
        int votes = 0;
        bool valid = false;
    };

    inline RingConsensus selectRingConsensus(
        const std::vector<Ring> &observations,
        int minimumVotes = 2)
    {
        std::array<int, 6> counts{};
        for (const Ring ring : observations)
        {
            if (ring != Ring::MISS)
                counts[static_cast<size_t>(ring)]++;
        }

        RingConsensus result;
        bool tied = false;
        for (size_t index = 1; index < counts.size(); ++index)
        {
            if (counts[index] > result.votes)
            {
                result.ring = static_cast<Ring>(index);
                result.votes = counts[index];
                tied = false;
            }
            else if (counts[index] > 0 && counts[index] == result.votes)
            {
                tied = true;
            }
        }

        result.valid = result.votes >= minimumVotes && !tied;
        if (!result.valid)
            result.ring = Ring::MISS;
        return result;
    }

    // Keep the wedge number supplied by an orientation-ready camera while
    // replacing only its ring prefix with the multi-camera ring consensus.
    inline std::string applyRingConsensus(
        Ring ring,
        const std::string &orientedScore)
    {
        if (ring == Ring::INNER_BULL)
            return "BULL";
        if (ring == Ring::OUTER_BULL)
            return "OUTER";
        if (!ringRequiresWedge(ring) || orientedScore.size() < 2 ||
            (orientedScore[0] != 'S' && orientedScore[0] != 'D' && orientedScore[0] != 'T'))
            return "MISS";

        return std::string(ringToString(ring)) + orientedScore.substr(1);
    }

    // When orientation-ready cameras disagree on adjacent wedges, prefer the
    // observation whose detected tip is farther from its nearest numbered
    // wire. Camera ordering is not a measure of confidence.
    inline size_t selectMostSeparatedCandidate(
        const std::vector<float> &nearestWireDistances)
    {
        size_t bestIndex = 0;
        float bestDistance = -std::numeric_limits<float>::infinity();
        for (size_t index = 0; index < nearestWireDistances.size(); ++index)
        {
            const float distance = nearestWireDistances[index];
            if (std::isnan(distance))
                continue;
            if (distance > bestDistance)
            {
                bestDistance = distance;
                bestIndex = index;
            }
        }
        return bestIndex;
    }
} // namespace score_processing
