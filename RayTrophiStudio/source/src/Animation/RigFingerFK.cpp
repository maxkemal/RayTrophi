#include "Animation/RigFingerFK.h"
#include <array>
#include <cmath>
#include <utility>

namespace RigAuthoring {
namespace {

constexpr float Pi = 3.14159265359f;

const RigChain* findChain(const RigAnatomy& anatomy, const std::string& name) {
    for (const auto& chain : anatomy.chains) {
        if (chain.name == name)
            return &chain;
    }
    return nullptr;
}

int findNode(const RayTrophi::NodeHierarchy& hierarchy, const std::string& name) {
    for (size_t index = 0; index < hierarchy.nodes.size(); ++index) {
        if (hierarchy.nodes[index].uniqueName == name)
            return static_cast<int>(index);
    }
    return -1;
}

bool validAmount(float value) {
    return std::isfinite(value) && value >= -1.f && value <= 1.f;
}

bool applySide(const RigAnatomy& anatomy, const std::string& side, float curl,
               float spread, float thumb, RayTrophi::NodeHierarchy& output,
               std::vector<std::string>& affectedBones, std::string& error) {
    static constexpr std::array<const char*, 5> FingerNames = {
        "thumb", "index", "middle", "ring", "pinky"};
    static constexpr std::array<float, 5> SpreadWeights = {0.f, 1.f, .25f, -.35f, -1.f};
    static constexpr std::array<float, 3> CurlWeights = {.72f, .88f, 1.f};
    const float sideSign = side == "left" ? 1.f : -1.f;

    for (size_t fingerIndex = 0; fingerIndex < FingerNames.size(); ++fingerIndex) {
        const bool isThumb = fingerIndex == 0;
        const auto* chain = findChain(anatomy, side + "_" + FingerNames[fingerIndex]);
        if (!chain || chain->bones.size() != 5) {
            error = "rig_finger_controls_unavailable";
            return false;
        }
        for (size_t joint = 1; joint <= 3; ++joint) {
            const int nodeIndex = findNode(output, chain->bones[joint]);
            if (nodeIndex < 0) {
                error = "rig_finger_controls_unavailable";
                return false;
            }
            const int parent = findNode(output, chain->bones[joint - 1]);
            if (output.nodes[static_cast<size_t>(nodeIndex)].parent != parent) {
                error = "rig_finger_controls_unavailable";
                return false;
            }

            const float curlDegrees = isThumb ? 58.f : 82.f;
            float curlAngle = -sideSign * curl * CurlWeights[joint - 1] * curlDegrees;
            float spreadAngle = 0.f;
            if (joint == 1) {
                spreadAngle = -sideSign * spread * SpreadWeights[fingerIndex] * 18.f;
            }
            if (isThumb) {
                curlAngle += -sideSign * thumb * CurlWeights[joint - 1] * 52.f;
                spreadAngle += -sideSign * thumb * (joint == 1 ? 30.f : 8.f);
            }
            if (std::fabs(curlAngle) < 1e-6f && std::fabs(spreadAngle) < 1e-6f)
                continue;

            auto& local = output.nodes[static_cast<size_t>(nodeIndex)].localBind;
            local = local * Matrix4x4::rotationY(spreadAngle * Pi / 180.f) *
                    Matrix4x4::rotationZ(curlAngle * Pi / 180.f);
            affectedBones.push_back(chain->bones[joint]);
        }
    }
    return true;
}

} // namespace

bool poseDetailedHumanoidFingers(const RayTrophi::NodeHierarchy& input,
                                 const RigAnatomy& anatomy, const std::string& side,
                                 float curl, float spread, float thumb,
                                 RayTrophi::NodeHierarchy& output,
                                 std::vector<std::string>& affectedBones,
                                 std::string& error) {
    error.clear();
    affectedBones.clear();
    if (side != "left" && side != "right" && side != "both") {
        error = "rig_finger_invalid_side";
        return false;
    }
    if (!validAmount(curl) || !validAmount(spread) || !validAmount(thumb)) {
        error = "rig_finger_invalid_amount";
        return false;
    }
    if (std::fabs(curl) < 1e-6f && std::fabs(spread) < 1e-6f &&
        std::fabs(thumb) < 1e-6f) {
        error = "rig_edit_no_change";
        return false;
    }

    RayTrophi::NodeHierarchy staged = input;
    if ((side == "left" || side == "both") &&
        !applySide(anatomy, "left", curl, spread, thumb, staged, affectedBones, error)) {
        affectedBones.clear();
        return false;
    }
    if ((side == "right" || side == "both") &&
        !applySide(anatomy, "right", curl, spread, thumb, staged, affectedBones, error)) {
        affectedBones.clear();
        return false;
    }
    output = std::move(staged);
    return true;
}

} // namespace RigAuthoring
