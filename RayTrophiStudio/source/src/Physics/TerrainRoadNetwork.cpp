#include "TerrainRoadNetwork.h"

#include <algorithm>

namespace TerrainNodesV2 {

RoadCarveSettings RoadAssignment::effectiveCarve() const {
    if (hasOverride) return overrideCarve;
    RoadProfile profile;
    if (findRoadProfile(profileId, profile)) return profile.carve;
    // Unknown profile: the assignment is invalid and the caller is expected to
    // have refused it at write time. Returning defaults here would let a broken
    // assignment carve something reasonable-looking, so the network solver
    // checks the profile again and reports instead of relying on this.
    return RoadCarveSettings{};
}

RoadNetworkRegistry& RoadNetworkRegistry::getInstance() {
    static RoadNetworkRegistry instance;
    return instance;
}

RoadAssignment* RoadNetworkRegistry::mutableFind(const std::string& splineObject) {
    const auto found = std::find_if(assignments_.begin(), assignments_.end(),
        [&splineObject](const RoadAssignment& a) { return a.splineObject == splineObject; });
    return found == assignments_.end() ? nullptr : &*found;
}

const RoadAssignment* RoadNetworkRegistry::find(const std::string& splineObject) const {
    const auto found = std::find_if(assignments_.begin(), assignments_.end(),
        [&splineObject](const RoadAssignment& a) { return a.splineObject == splineObject; });
    return found == assignments_.end() ? nullptr : &*found;
}

bool RoadNetworkRegistry::assignProfile(const std::string& splineObject,
                                        const std::string& profileId,
                                        std::string* error) {
    if (splineObject.empty()) {
        if (error) *error = "spline object name is required";
        return false;
    }
    RoadProfile profile;
    if (!findRoadProfile(profileId, profile)) {
        if (error) *error = "unknown road profile '" + profileId +
                            "' (expected footpath|dirt_road|main_road)";
        return false;
    }
    if (RoadAssignment* existing = mutableFind(splineObject)) {
        existing->profileId = profileId;
    } else {
        RoadAssignment assignment;
        assignment.splineObject = splineObject;
        assignment.profileId = profileId;
        assignments_.push_back(assignment);
    }
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::setCrossingMode(const std::string& splineObject,
                                          RoadCrossingMode mode, std::string* error) {
    RoadAssignment* assignment = mutableFind(splineObject);
    if (!assignment) {
        if (error) *error = "no road assignment for '" + splineObject + "'";
        return false;
    }
    assignment->crossingMode = mode;
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::setEnabled(const std::string& splineObject, bool enabled,
                                     std::string* error) {
    RoadAssignment* assignment = mutableFind(splineObject);
    if (!assignment) {
        if (error) *error = "no road assignment for '" + splineObject + "'";
        return false;
    }
    assignment->enabled = enabled;
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::setCarveOverride(const std::string& splineObject,
                                           const RoadCarveSettings& settings,
                                           std::string* error) {
    RoadAssignment* assignment = mutableFind(splineObject);
    if (!assignment) {
        if (error) *error = "no road assignment for '" + splineObject + "'";
        return false;
    }
    assignment->hasOverride = true;
    assignment->overrideCarve = settings;
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::clearCarveOverride(const std::string& splineObject,
                                             std::string* error) {
    RoadAssignment* assignment = mutableFind(splineObject);
    if (!assignment) {
        if (error) *error = "no road assignment for '" + splineObject + "'";
        return false;
    }
    assignment->hasOverride = false;
    assignment->overrideCarve = RoadCarveSettings{};
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::setMeshObject(const std::string& splineObject,
                                        const std::string& meshObject,
                                        std::string* error) {
    RoadAssignment* assignment = mutableFind(splineObject);
    if (!assignment) {
        if (error) *error = "no road assignment for '" + splineObject + "'";
        return false;
    }
    assignment->meshObject = meshObject;
    ++generation_;
    return true;
}

bool RoadNetworkRegistry::clearProfile(const std::string& splineObject) {
    const auto found = std::remove_if(assignments_.begin(), assignments_.end(),
        [&splineObject](const RoadAssignment& a) { return a.splineObject == splineObject; });
    if (found == assignments_.end()) return false;
    assignments_.erase(found, assignments_.end());
    ++generation_;
    return true;
}

std::vector<std::string> RoadNetworkRegistry::danglingAssignments(
    const std::function<bool(const std::string&)>& splineExists) const {
    std::vector<std::string> dangling;
    if (!splineExists) return dangling;
    for (const auto& assignment : assignments_) {
        if (!splineExists(assignment.splineObject)) dangling.push_back(assignment.splineObject);
    }
    return dangling;
}

void RoadNetworkRegistry::clear() {
    assignments_.clear();
    ++generation_;
}

nlohmann::json RoadNetworkRegistry::serialize() const {
    nlohmann::json items = nlohmann::json::array();
    for (const auto& assignment : assignments_) {
        nlohmann::json item;
        item["splineObject"] = assignment.splineObject;
        item["profileId"] = assignment.profileId;
        item["crossingMode"] = roadCrossingModeName(assignment.crossingMode);
        item["enabled"] = assignment.enabled;
        if (!assignment.meshObject.empty()) item["meshObject"] = assignment.meshObject;
        if (assignment.hasOverride) {
            item["overrideCarve"] = roadCarveSettingsToJson(assignment.overrideCarve);
        }
        items.push_back(std::move(item));
    }
    nlohmann::json root;
    root["version"] = 1;
    root["assignments"] = std::move(items);
    return root;
}

void RoadNetworkRegistry::deserialize(const nlohmann::json& value) {
    assignments_.clear();
    ++generation_;
    if (!value.is_object()) return;
    const auto items = value.find("assignments");
    if (items == value.end() || !items->is_array()) return;
    for (const auto& item : *items) {
        if (!item.is_object()) continue;
        RoadAssignment assignment;
        assignment.splineObject = item.value("splineObject", std::string());
        if (assignment.splineObject.empty()) continue;
        assignment.profileId = item.value("profileId", std::string("dirt_road"));
        // An unknown profile in a saved file is kept as-is rather than silently
        // rewritten: the solver reports it by name, which tells the author what
        // actually went missing. Rewriting it to a default would carve something
        // plausible and erase the evidence.
        RoadCrossingMode mode = RoadCrossingMode::Auto;
        parseRoadCrossingMode(item.value("crossingMode", std::string("auto")), mode);
        assignment.crossingMode = mode;
        assignment.enabled = item.value("enabled", true);
        assignment.meshObject = item.value("meshObject", std::string());
        const auto overrideValue = item.find("overrideCarve");
        if (overrideValue != item.end() && overrideValue->is_object()) {
            assignment.hasOverride = true;
            assignment.overrideCarve =
                roadCarveSettingsFromJson(*overrideValue, RoadCarveSettings{});
        }
        assignments_.push_back(std::move(assignment));
    }
}

} // namespace TerrainNodesV2
