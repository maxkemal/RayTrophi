#pragma once
#include "Animation/RigStudioContext.h"

struct SceneData;

namespace RayTrophi {

class RigStudioServices {
public:
    static RigStudioContext& getContext(SceneData& scene);
    static void syncContextWithRigView(SceneData& scene);
    static void setMode(SceneData& scene, RigStudioMode mode);
    static void setManipulationMode(SceneData& scene, ManipulationMode mode);
    static void setActiveCharacter(SceneData& scene, const std::string& characterId);
};

} // namespace RayTrophi
