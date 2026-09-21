#include "Animation/RigDrivenControls.h"
#include "Quaternion.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace RigAuthoring {
namespace {

constexpr float Pi = 3.14159265359f;

int findNode(const RayTrophi::NodeHierarchy& hierarchy, const std::string& name) {
    for (size_t index = 0; index < hierarchy.nodes.size(); ++index) {
        if (hierarchy.nodes[index].uniqueName == name)
            return static_cast<int>(index);
    }
    return -1;
}

Matrix4x4 axisRotation(const Vec3& axis, float radians) {
    const auto normalized = axis.normalize();
    const float sine = std::sin(radians * .5f);
    Quaternion rotation(std::cos(radians * .5f), normalized.x * sine,
                        normalized.y * sine, normalized.z * sine);
    rotation.normalize();
    return rotation.toMatrix();
}

} // namespace

bool evaluateRigDrivenControls(const RayTrophi::NodeHierarchy& input,
                               const std::vector<RigDrivenControl>& definitions,
                               const std::map<std::string, float>& values,
                               RayTrophi::NodeHierarchy& output,
                               std::vector<std::string>& affectedBones,
                               std::string& error) {
    error.clear();
    affectedBones.clear();
    if (values.empty()) {
        error = "rig_control_values_empty";
        return false;
    }
    std::unordered_map<std::string, const RigDrivenControl*> controls;
    for (const auto& control : definitions)
        controls.emplace(control.id, &control);

    for (const auto& value : values) {
        const auto found = controls.find(value.first);
        if (found == controls.end()) {
            error = "rig_control_unknown";
            return false;
        }
        const auto& control = *found->second;
        if (!std::isfinite(value.second) || value.second < control.minimum ||
            value.second > control.maximum) {
            error = "rig_control_value_out_of_range";
            return false;
        }
    }

    RayTrophi::NodeHierarchy staged = input;
    std::unordered_set<std::string> affected;
    for (const auto& control : definitions) {
        const auto value = values.find(control.id);
        if (value == values.end() ||
            std::fabs(value->second - control.defaultValue) < 1e-6f) {
            continue;
        }
        for (const auto& driver : control.drivers) {
            const int nodeIndex = findNode(staged, driver.bone);
            if (nodeIndex < 0) {
                error = "rig_control_unknown_bone";
                return false;
            }
            const float normalized = value->second - control.defaultValue;
            auto& local = staged.nodes[static_cast<size_t>(nodeIndex)].localBind;
            local = local * axisRotation(driver.axis, driver.degrees * normalized * Pi / 180.f);
            affected.insert(driver.bone);
        }
    }
    if (affected.empty()) {
        error = "rig_edit_no_change";
        return false;
    }
    affectedBones.assign(affected.begin(), affected.end());
    std::sort(affectedBones.begin(), affectedBones.end());
    output = std::move(staged);
    return true;
}

nlohmann::json serializeRigDrivenControls(const std::vector<RigDrivenControl>& controls) {
    auto result = nlohmann::json::array();
    for (const auto& control : controls) {
        auto drivers = nlohmann::json::array();
        for (const auto& driver : control.drivers) {
            drivers.push_back({{"bone", driver.bone},
                               {"axis", {driver.axis.x, driver.axis.y, driver.axis.z}},
                               {"degrees", driver.degrees}});
        }
        result.push_back({{"id", control.id},
                          {"label", control.label},
                          {"group", control.group},
                          {"anchor", control.anchor},
                          {"side", control.side},
                          {"shape", control.shape},
                          {"minimum", control.minimum},
                          {"maximum", control.maximum},
                          {"default", control.defaultValue},
                          {"drivers", std::move(drivers)}});
    }
    return result;
}

} // namespace RigAuthoring
