#pragma once

#include "Animation/AnimationData.h"

#include <cstdint>
#include <string>

// Timeline-deterministic root motion.
//
// A looping locomotion clip carries its forward travel in the root bone's
// translation keys and snaps back by one cycle's travel at every wrap. Root
// motion here undoes exactly that snap: the root bone keeps its in-cycle
// motion (height included) and gains completed_cycles * cycle_travel. Both
// terms are functions of the clip time, so scrubbing, rewinding and replaying
// land on the same pose, and nothing is written into the object's transform.
namespace RootMotionUnroll {

struct LoopedTime {
    float seconds = 0.0f;   // time inside the clip
    int64_t cycles = 0;     // completed loops before it; always 0 when not looping
};

// Splits an absolute clip time into (cycles, time inside the clip). A
// non-looping clip is clamped to [0, duration].
LoopedTime wrap(double seconds, float duration_seconds, bool loop);

// Travel of `bone` over one full cycle, in the bone's parent space (the space
// its translation keys live in): last position key minus first. False when
// the clip has fewer than two position keys for that bone.
bool cycleTravel(const AnimationData& clip, const std::string& bone, Vec3& travel);

inline Vec3 unrolled(const Vec3& local_translation, const Vec3& travel, int64_t cycles) {
    return local_translation + travel * static_cast<float>(cycles);
}

} // namespace RootMotionUnroll
