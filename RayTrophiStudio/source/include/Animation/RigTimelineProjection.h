#pragma once

#include <cstdint>
#include <set>
#include <string>

struct SceneData;
struct ObjectAnimationTrack;
class Vec3;

namespace RigAuthoring {

// Projects only the active authored clip's selected bones into legacy Timeline
// tracks. AnimationData remains authoritative; these tracks are disposable UI data.
bool synchronizeRigTimelineProjection(SceneData& scene,
                                      std::set<std::string>& projectedTracks,
                                      std::uint64_t& projectionSignature,
                                      std::string& preferredTrack);

// Updates only the disposable Graph Editor projection while a rig key is dragged.
// Position (channels 0..2) and rotation (3..5) are kept as separate atomic families.
// Returns false when the selected family is missing or the destination is occupied.
bool previewRigTimelineCurveDrag(ObjectAnimationTrack& track,
                                 int channel,
                                 int currentFrame,
                                 int targetFrame,
                                 float positionValue);

// Reads the family currently displayed by a live drag preview. Position is filled
// only for channels 0..2; rotation callers use the boolean presence result.
bool readRigTimelineCurvePreview(const ObjectAnimationTrack& track,
                                 int channel,
                                 int frame,
                                 Vec3& position);

} // namespace RigAuthoring
