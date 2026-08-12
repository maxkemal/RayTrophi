#include "ThermalFractureSerialization.h"

#include <algorithm>
#include <string>

namespace RayTrophiSim {

void serializeThermalFracture(const RigidBodyObject& body,
                              nlohmann::json& body_json) {
    if (!body.getBreakable()) return;
    body_json["fracture"] = {
        {"breakable", true},
        {"break_velocity", body.getBreakVelocity()},
        {"group", body.getFractureGroup()},
        {"integrity_weakening", body.getIntegrityWeakening()},
        {"integrity_exponent", body.getIntegrityExponent()},
        {"minimum_threshold_scale", body.getMinimumThresholdScale()}
    };
}

void deserializeThermalFracture(const nlohmann::json& body_json,
                                RigidBodyObject& body) {
    if (!body_json.contains("fracture") ||
        !body_json["fracture"].is_object()) return;
    const auto& fracture = body_json["fracture"];
    body.setBreakable(fracture.value("breakable", false));
    // ★ The key changed with the units (N.s -> m/s). Projects written before the
    // change carry `break_impulse` and are NOT read here on purpose: their number
    // was an impulse tuned against 1 kg shards, so re-reading it as a velocity
    // would be right only by coincidence. Falling back to the default is the
    // honest answer, and 5 m/s is the same default those projects had.
    body.setBreakVelocity(std::max(fracture.value("break_velocity", 5.0f), 0.001f));
    body.setFractureGroup(fracture.value("group", std::string{}));
    body.setIntegrityWeakening(fracture.value("integrity_weakening", true));
    body.setIntegrityExponent(std::max(
        fracture.value("integrity_exponent", 1.5f), 0.01f));
    body.setMinimumThresholdScale(std::clamp(
        fracture.value("minimum_threshold_scale", 0.15f), 0.0f, 1.0f));
}

} // namespace RayTrophiSim
