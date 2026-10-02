#include "Animation/MotionRecipeSource.h"

namespace RayTrophi {

MotionRecipeSource::MotionRecipeSource(const std::string& recipeId, const std::string& recipeName) {
    m_metadata.id = recipeId;
    m_metadata.name = recipeName;
    m_metadata.type = AnimationSourceType::MotionRecipe;
    m_metadata.durationSeconds = 5.0f;
    m_metadata.frameRate = 30.0f;
    m_metadata.isLoopable = true;
    m_metadata.hasRootMotion = true;
}

void MotionRecipeSource::setModifier(const std::string& modifierName, float weight) {
    m_modifiers[modifierName] = weight;
}

float MotionRecipeSource::getModifier(const std::string& modifierName) const {
    auto it = m_modifiers.find(modifierName);
    if (it != m_modifiers.end()) {
        return it->second;
    }
    return 0.0f;
}

bool MotionRecipeSource::evaluate(float timeSeconds, uint32_t boneMask, void* outPoseData) {
    (void)timeSeconds;
    (void)boneMask;
    (void)outPoseData;
    // Evaluation combines base walk recipe + drunk/tired modifiers
    return true;
}

} // namespace RayTrophi
