#pragma once
#include "Animation/AnimationSource.h"

namespace RayTrophi {

class ImportedClipSource : public IAnimationSource {
public:
    ImportedClipSource(const std::string& clipId, const std::string& clipName, float durationSeconds);
    virtual ~ImportedClipSource() = default;

    const AnimationSourceMetadata& getMetadata() const override { return m_metadata; }
    bool evaluate(float timeSeconds, uint32_t boneMask, void* outPoseData) override;

    void setRootMotion(bool enabled) { m_metadata.hasRootMotion = enabled; }
    void setLoopable(bool loopable) { m_metadata.isLoopable = loopable; }

private:
    AnimationSourceMetadata m_metadata;
};

} // namespace RayTrophi
