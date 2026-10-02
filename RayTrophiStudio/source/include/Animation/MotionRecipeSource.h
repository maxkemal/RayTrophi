#pragma once
#include "Animation/AnimationSource.h"
#include <unordered_map>

namespace RayTrophi {

class MotionRecipeSource : public IAnimationSource {
public:
    MotionRecipeSource(const std::string& recipeId, const std::string& recipeName);
    virtual ~MotionRecipeSource() = default;

    const AnimationSourceMetadata& getMetadata() const override { return m_metadata; }
    bool evaluate(float timeSeconds, uint32_t boneMask, void* outPoseData) override;

    void setModifier(const std::string& modifierName, float weight);
    float getModifier(const std::string& modifierName) const;
    const std::unordered_map<std::string, float>& getModifiers() const { return m_modifiers; }

    void setSpeed(float speed) { m_speed = speed; }
    void setStride(float stride) { m_stride = stride; }

private:
    AnimationSourceMetadata m_metadata;
    std::unordered_map<std::string, float> m_modifiers;
    float m_speed = 1.0f;
    float m_stride = 1.0f;
};

} // namespace RayTrophi
