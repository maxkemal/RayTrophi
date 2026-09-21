#include "Animation/RigBoneCurves.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

namespace RigAuthoring {
namespace {

constexpr double kKeyTimeTolerance = 1e-8;

template <typename Keys>
auto exactKey(Keys& keys, double time) {
    auto key = std::lower_bound(keys.begin(), keys.end(), time,
                                [](const auto& candidate, double value) {
                                    return candidate.time < value;
                                });
    if (key != keys.end() && std::fabs(key->time - time) < kKeyTimeTolerance)
        return key;
    if (key != keys.begin()) {
        auto previous = std::prev(key);
        if (std::fabs(previous->time - time) < kKeyTimeTolerance)
            return previous;
    }
    return keys.end();
}

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z);
}

bool finite(const Quaternion& value) {
    return std::isfinite(value.w) && std::isfinite(value.x) &&
           std::isfinite(value.y) && std::isfinite(value.z);
}

void stabilizeQuaternionSigns(RayTrophi::QuatKeys& keys) {
    for (size_t index = 1; index < keys.size(); ++index) {
        const Quaternion& previous = keys[index - 1].value;
        Quaternion& current = keys[index].value;
        const float dot = previous.w * current.w + previous.x * current.x +
                          previous.y * current.y + previous.z * current.z;
        if (dot < 0.0f) {
            current.w = -current.w;
            current.x = -current.x;
            current.y = -current.y;
            current.z = -current.z;
        }
    }
}

} // namespace

bool editBoneCurveKey(AnimationData& clip,
                      const std::string& bone,
                      BoneCurveChannel channel,
                      double sourceSeconds,
                      double targetSeconds,
                      const Vec3* position,
                      const Quaternion* rotation,
                      std::string& error) {
    error.clear();
    if (!clip.rigAuthoring) {
        error = "rig_pose_clip_not_editable";
        return false;
    }
    if (bone.empty()) {
        error = "unknown_bone";
        return false;
    }
    if (!std::isfinite(sourceSeconds) || !std::isfinite(targetSeconds) ||
        sourceSeconds < 0.0 || targetSeconds < 0.0 ||
        !std::isfinite(clip.duration) || clip.duration <= 0.0 ||
        !std::isfinite(clip.ticksPerSecond) || clip.ticksPerSecond <= 0.0) {
        error = "invalid_clip_timing";
        return false;
    }
    const double sourceTime = sourceSeconds * clip.ticksPerSecond;
    const double targetTime = targetSeconds * clip.ticksPerSecond;
    if (!std::isfinite(sourceTime) || !std::isfinite(targetTime) ||
        sourceTime > 1000000.0 || targetTime > 1000000.0) {
        error = "rig_pose_time_limit";
        return false;
    }

    AnimationData staged = clip;
    bool changed = false;
    if (channel == BoneCurveChannel::Position) {
        auto found = staged.positionKeys.find(bone);
        if (found == staged.positionKeys.end()) {
            error = "rig_curve_key_not_found";
            return false;
        }
        auto& keys = found->second;
        auto source = exactKey(keys, sourceTime);
        if (source == keys.end()) {
            error = "rig_curve_key_not_found";
            return false;
        }
        if (position && !finite(*position)) {
            error = "rig_curve_invalid_value";
            return false;
        }
        if (std::fabs(sourceTime - targetTime) >= kKeyTimeTolerance) {
            if (exactKey(keys, targetTime) != keys.end()) {
                error = "rig_curve_key_conflict";
                return false;
            }
            source->time = targetTime;
            changed = true;
        }
        if (position && (source->value.x != position->x ||
                         source->value.y != position->y ||
                         source->value.z != position->z)) {
            source->value = *position;
            changed = true;
        }
        std::sort(keys.begin(), keys.end(),
                  [](const auto& left, const auto& right) {
                      return left.time < right.time;
                  });
    } else {
        auto found = staged.rotationKeys.find(bone);
        if (found == staged.rotationKeys.end()) {
            error = "rig_curve_key_not_found";
            return false;
        }
        auto& keys = found->second;
        auto source = exactKey(keys, sourceTime);
        if (source == keys.end()) {
            error = "rig_curve_key_not_found";
            return false;
        }
        Quaternion value = rotation ? *rotation : source->value;
        const float norm = value.w * value.w + value.x * value.x +
                           value.y * value.y + value.z * value.z;
        if (!finite(value) || !std::isfinite(norm) || norm < 1e-12f) {
            error = "rig_curve_invalid_value";
            return false;
        }
        value.normalize();
        if (std::fabs(sourceTime - targetTime) >= kKeyTimeTolerance) {
            if (exactKey(keys, targetTime) != keys.end()) {
                error = "rig_curve_key_conflict";
                return false;
            }
            source->time = targetTime;
            changed = true;
        }
        if (rotation && (source->value.w != value.w || source->value.x != value.x ||
                         source->value.y != value.y || source->value.z != value.z)) {
            source->value = value;
            changed = true;
        }
        std::sort(keys.begin(), keys.end(),
                  [](const auto& left, const auto& right) {
                      return left.time < right.time;
                  });
        stabilizeQuaternionSigns(keys);
    }

    if (!changed) {
        error = "rig_edit_no_change";
        return false;
    }
    staged.duration = std::max(staged.duration, targetTime + 1.0);
    staged.endFrame = std::max(staged.endFrame,
                               static_cast<int>(std::ceil(targetTime)));
    clip = std::move(staged);
    return true;
}

} // namespace RigAuthoring
