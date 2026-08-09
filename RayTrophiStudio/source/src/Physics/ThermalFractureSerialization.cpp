#include "ThermalFractureSerialization.h"

#include <algorithm>
#include <string>

namespace RayTrophiSim {

void serializeThermalFracture(const RigidBodyObject& body,
                              nlohmann::json& body_json) {
    if (!body.getBreakable()) return;
    body_json["fracture"] = {
        {"breakable", true},
        {"break_impulse", body.getBreakImpulse()},
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
    body.setBreakImpulse(std::max(fracture.value("break_impulse", 5.0f), 0.001f));
    body.setFractureGroup(fracture.value("group", std::string{}));
    body.setIntegrityWeakening(fracture.value("integrity_weakening", true));
    body.setIntegrityExponent(std::max(
        fracture.value("integrity_exponent", 1.5f), 0.01f));
    body.setMinimumThresholdScale(std::clamp(
        fracture.value("minimum_threshold_scale", 0.15f), 0.0f, 1.0f));
}

} // namespace RayTrophiSim
