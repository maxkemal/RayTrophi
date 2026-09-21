#pragma once

#include "Animation/AnimationData.h"

#include <string>

namespace RigAuthoring {

enum class BoneCurveChannel {
    Position,
    Rotation
};

// Edits one canonical vector/quaternion key. Times are seconds; AnimationData
// remains tick-based internally. Null values preserve the stored value and move
// only time. The operation stages the whole clip and publishes nothing on error.
bool editBoneCurveKey(AnimationData& clip,
                      const std::string& bone,
                      BoneCurveChannel channel,
                      double sourceSeconds,
                      double targetSeconds,
                      const Vec3* position,
                      const Quaternion* rotation,
                      std::string& error);

} // namespace RigAuthoring
