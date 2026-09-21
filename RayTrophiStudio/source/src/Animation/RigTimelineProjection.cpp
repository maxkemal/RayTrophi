#include "Animation/RigTimelineProjection.h"

#include "Animation/RigSelection.h"
#include "KeyframeSystem.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace RigAuthoring {
namespace {

void hashBytes(std::uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for(size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
}

void hashString(std::uint64_t& hash, const std::string& value) {
    hashBytes(hash, value.data(), value.size());
}

Vec3 quaternionEulerDegrees(const Quaternion& value) {
    const float sinRoll = 2.0f * (value.w * value.x + value.y * value.z);
    const float cosRoll = 1.0f - 2.0f * (value.x * value.x + value.y * value.y);
    const float sinPitch = 2.0f * (value.w * value.y - value.z * value.x);
    const float sinYaw = 2.0f * (value.w * value.z + value.x * value.y);
    const float cosYaw = 1.0f - 2.0f * (value.y * value.y + value.z * value.z);
    constexpr float radiansToDegrees = 57.29577951308232f;
    return Vec3(std::atan2(sinRoll, cosRoll) * radiansToDegrees,
                (std::abs(sinPitch) >= 1.0f
                     ? std::copysign(1.5707963267948966f, sinPitch)
                     : std::asin(sinPitch)) * radiansToDegrees,
                std::atan2(sinYaw, cosYaw) * radiansToDegrees);
}

} // namespace

bool synchronizeRigTimelineProjection(SceneData& scene,
                                      std::set<std::string>& projectedTracks,
                                      std::uint64_t& projectionSignature,
                                      std::string& preferredTrack) {
    preferredTrack.clear();
    const auto& pose = scene.rigView.pose;
    const auto removeProjection = [&]() {
        for(const auto& name : projectedTracks) {
            scene.timeline.tracks.erase(name);
        }
        projectedTracks.clear();
        projectionSignature = 0;
    };
    if(!pose.active || pose.character.empty()) {
        if(projectedTracks.empty()) {
            return false;
        }
        removeProjection();
        return true;
    }

    const auto selectedClip = pose.clips.find(pose.character);
    if(selectedClip == pose.clips.end()) {
        if(projectedTracks.empty()) {
            return false;
        }
        removeProjection();
        return true;
    }
    std::shared_ptr<AnimationData> clip;
    for(const auto& candidate : scene.animationDataList) {
        if(candidate && candidate->rigAuthoring && candidate->modelName == pose.character &&
           candidate->name == selectedClip->second) {
            clip = candidate;
            break;
        }
    }
    if(!clip) {
        if(projectedTracks.empty()) {
            return false;
        }
        removeProjection();
        return true;
    }

    auto bones = selectedBones(scene);
    std::sort(bones.begin(), bones.end());
    std::uint64_t signature = 1469598103934665603ull;
    hashString(signature, pose.character);
    hashString(signature, clip->name);
    const auto identity = reinterpret_cast<std::uintptr_t>(clip.get());
    hashBytes(signature, &identity, sizeof(identity));
    hashBytes(signature, &pose.fps, sizeof(pose.fps));
    for(const auto& bone : bones) {
        hashString(signature, bone);
    }
    if(signature == projectionSignature) {
        preferredTrack = scene.rigView.bone;
        return false;
    }

    removeProjection();
    // Older one-way imports and reopened projects may already contain a generic
    // track for every bone. Skeleton names are reserved channel identities, so
    // clear the active character's stale projections before adding the selection.
    for(const auto& model : scene.importedModelContexts) {
        for(const auto& node : model.nodeHierarchy.nodes) {
            scene.timeline.tracks.erase(node.uniqueName);
        }
    }
    const double fps = std::isfinite(pose.fps) && pose.fps > 0.0f ? pose.fps : 24.0;
    const double ticks = clip->ticksPerSecond > 0.0 ? clip->ticksPerSecond : fps;
    for(const auto& bone : bones) {
        ObjectAnimationTrack track;
        track.object_name = bone;
        const auto position = clip->positionKeys.find(bone);
        if(position != clip->positionKeys.end()) {
            for(const auto& key : position->second) {
                const int frame = static_cast<int>(std::llround(key.time / ticks * fps));
                Keyframe value(frame);
                value.has_transform = true;
                value.transform.has_position = true;
                value.transform.has_pos_x = true;
                value.transform.has_pos_y = true;
                value.transform.has_pos_z = true;
                value.transform.has_rotation = false;
                value.transform.has_scale = false;
                value.transform.position = key.value;
                track.addKeyframe(value);
            }
        }
        const auto rotation = clip->rotationKeys.find(bone);
        if(rotation != clip->rotationKeys.end()) {
            for(const auto& key : rotation->second) {
                const int frame = static_cast<int>(std::llround(key.time / ticks * fps));
                auto* existing = track.getKeyframeAt(frame);
                if(existing) {
                    existing->has_transform = true;
                    existing->transform.has_rotation = true;
                    existing->transform.has_rot_x = true;
                    existing->transform.has_rot_y = true;
                    existing->transform.has_rot_z = true;
                    existing->transform.rotation = quaternionEulerDegrees(key.value);
                } else {
                    Keyframe value(frame);
                    value.has_transform = true;
                    value.transform.has_position = false;
                    value.transform.has_rotation = true;
                    value.transform.has_rot_x = true;
                    value.transform.has_rot_y = true;
                    value.transform.has_rot_z = true;
                    value.transform.has_scale = false;
                    value.transform.rotation = quaternionEulerDegrees(key.value);
                    track.addKeyframe(value);
                }
            }
        }
        scene.timeline.tracks[bone] = std::move(track);
        projectedTracks.insert(bone);
    }
    projectionSignature = signature;
    preferredTrack = scene.rigView.bone;
    return true;
}

bool previewRigTimelineCurveDrag(ObjectAnimationTrack& track,
                                 int channel,
                                 int currentFrame,
                                 int targetFrame,
                                 float positionValue) {
    if (channel < 0 || channel > 5)
        return false;
    const bool positionChannel = channel < 3;
    const auto hasFamily = [&](const Keyframe& keyframe) {
        return keyframe.has_transform &&
               (positionChannel ? keyframe.transform.has_position
                                : keyframe.transform.has_rotation);
    };
    auto current = std::find_if(
        track.keyframes.begin(), track.keyframes.end(),
        [&](const Keyframe& keyframe) {
            return keyframe.frame == currentFrame && hasFamily(keyframe);
        });
    if (current == track.keyframes.end())
        return false;
    if (targetFrame != currentFrame &&
        std::any_of(
            track.keyframes.begin(), track.keyframes.end(),
            [&](const Keyframe& keyframe) {
                return keyframe.frame == targetFrame && hasFamily(keyframe);
            }))
        return false;

    Keyframe moving = *current;
    if (positionChannel)
        moving.transform.setChannelValue(channel, positionValue);
    if (targetFrame == currentFrame) {
        *current = std::move(moving);
        return true;
    }

    if (positionChannel) {
        moving.transform.has_rotation = false;
        moving.transform.has_rot_x = false;
        moving.transform.has_rot_y = false;
        moving.transform.has_rot_z = false;
        current->transform.has_position = false;
        current->transform.has_pos_x = false;
        current->transform.has_pos_y = false;
        current->transform.has_pos_z = false;
    } else {
        moving.transform.has_position = false;
        moving.transform.has_pos_x = false;
        moving.transform.has_pos_y = false;
        moving.transform.has_pos_z = false;
        current->transform.has_rotation = false;
        current->transform.has_rot_x = false;
        current->transform.has_rot_y = false;
        current->transform.has_rot_z = false;
    }
    const bool keepCurrent = current->transform.has_position ||
                             current->transform.has_rotation;
    if (keepCurrent)
        ++current;
    else
        current = track.keyframes.erase(current);
    moving.frame = targetFrame;
    track.keyframes.insert(current, std::move(moving));
    std::stable_sort(
        track.keyframes.begin(), track.keyframes.end(),
        [](const Keyframe& left, const Keyframe& right) {
            return left.frame < right.frame;
        });
    return true;
}

bool readRigTimelineCurvePreview(const ObjectAnimationTrack& track,
                                 int channel,
                                 int frame,
                                 Vec3& position) {
    if (channel < 0 || channel > 5)
        return false;
    const bool positionChannel = channel < 3;
    const auto keyframe = std::find_if(
        track.keyframes.cbegin(), track.keyframes.cend(),
        [&](const Keyframe& candidate) {
            if (candidate.frame != frame || !candidate.has_transform)
                return false;
            return positionChannel ? candidate.transform.has_position
                                   : candidate.transform.has_rotation;
        });
    if (keyframe == track.keyframes.cend())
        return false;
    if (positionChannel)
        position = keyframe->transform.position;
    return true;
}

} // namespace RigAuthoring
