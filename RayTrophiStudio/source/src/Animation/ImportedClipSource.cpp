#include "Animation/ImportedClipSource.h"

namespace RayTrophi {

ImportedClipSource::ImportedClipSource(const std::string& clipId, const std::string& clipName, float durationSeconds) {
    m_metadata.id = clipId;
    m_metadata.name = clipName;
    m_metadata.type = AnimationSourceType::ImportedClip;
    m_metadata.durationSeconds = durationSeconds;
    m_metadata.frameRate = 30.0f;
    m_metadata.isLoopable = true;
    m_metadata.hasRootMotion = false;
}

bool ImportedClipSource::evaluate(float timeSeconds, uint32_t boneMask, void* outPoseData) {
    (void)timeSeconds;
    (void)boneMask;
    (void)outPoseData;
    // Evaluation delegates to Ozz / ClipBinding pipeline
    return true;
}

} // namespace RayTrophi
