#include "Animation/RootMotionUnroll.h"

#include <algorithm>
#include <cmath>

namespace RootMotionUnroll {

LoopedTime wrap(double seconds, float duration_seconds, bool loop) {
    LoopedTime out;
    if (!(duration_seconds > 0.0f) || !std::isfinite(seconds)) {
        return out;
    }
    const double duration = static_cast<double>(duration_seconds);
    if (!loop) {
        out.seconds = static_cast<float>(std::clamp(seconds, 0.0, duration));
        return out;
    }
    const double cycles = std::floor(seconds / duration);
    double inside = seconds - cycles * duration;
    // floor() of an exact multiple can leave inside == duration after
    // rounding; that instant belongs to the next cycle's start.
    int64_t whole = static_cast<int64_t>(cycles);
    if (inside >= duration) {
        inside -= duration;
        ++whole;
    }
    out.seconds = static_cast<float>(std::max(0.0, inside));
    out.cycles = whole;
    return out;
}

bool cycleTravel(const AnimationData& clip, const std::string& bone, Vec3& travel) {
    const auto it = clip.positionKeys.find(bone);
    if (it == clip.positionKeys.end() || it->second.size() < 2) {
        return false;
    }
    // The keys span one cycle; the samplers' wrap interpolation between the
    // last and first key is the snap this travel undoes.
    travel = it->second.back().value - it->second.front().value;
    return std::isfinite(travel.x) && std::isfinite(travel.y) && std::isfinite(travel.z);
}

} // namespace RootMotionUnroll
