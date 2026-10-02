#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace RayTrophi {

enum class RigStudioMode {
    Rest,
    Pose,
    Animate,
    Skin
};

enum class ManipulationMode {
    FK,
    IK,
    Aim,
    SplineIK
};

enum class LocalWorldMode {
    Local,
    World,
    Component
};

struct RigStudioContext {
    std::string characterId;
    std::string rigId;
    std::string activeAnimGraphId;

    RigStudioMode mode = RigStudioMode::Animate;
    ManipulationMode manipulationMode = ManipulationMode::IK;
    LocalWorldMode transformSpace = LocalWorldMode::Local;

    // Selection State
    int activeTab = 0;
    int requestedTab = -1;
    std::vector<std::string> selectedBoneIds;
    std::string activeBoneId;
    std::string activeControllerId;
    std::string activeMotionBlockId;
    std::string activeNodeId;

    // Timeline & Frame State
    int32_t currentFrame = 0;
    float currentTimeSeconds = 0.0f;
    uint32_t boneMask = 0xFFFFFFFF; // Full body by default

    // Workflow Toggles
    bool windowOpen = true;
    bool autoKey = false;
    bool mirrorMode = false;
    bool footLockEnabled = true;

    // Viewport & Overlay Flags
    bool showBones = true;
    bool showIKTargets = true;
    bool showEnvelopes = false;
    bool showWeightHeatmap = false;
    bool showValidationErrors = true;

    void reset() {
        characterId.clear();
        rigId.clear();
        activeAnimGraphId.clear();
        selectedBoneIds.clear();
        activeBoneId.clear();
        activeControllerId.clear();
        activeMotionBlockId.clear();
        activeNodeId.clear();
        currentFrame = 0;
        currentTimeSeconds = 0.0f;
        boneMask = 0xFFFFFFFF;
    }
};

} // namespace RayTrophi
