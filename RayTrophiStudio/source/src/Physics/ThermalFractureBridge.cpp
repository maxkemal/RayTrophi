#include "scene_data.h"
#include "ThermalFractureBridge.h"

#include <algorithm>
#include <cmath>

float RayTrophiSim::effectiveFractureThreshold(
    const RigidBodyObject& body,
    const MaterialIntegritySummary& summary) {
    float threshold = body.getBreakImpulse();
    if (!body.getIntegrityWeakening() || !summary.valid) return threshold;
    const float structural = std::clamp(
        summary.mean_integrity * 0.65f +
        summary.remaining_support_ratio * 0.35f, 0.0f, 1.0f);
    const float scale = std::max(
        std::clamp(body.getMinimumThresholdScale(), 0.0f, 1.0f),
        std::pow(structural, std::max(body.getIntegrityExponent(), 0.01f)));
    return threshold * scale;
}

RayTrophiSim::MaterialIntegritySummary SceneData::fractureIntegritySummary(
    const std::string& group) const {
    for (const auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& fields = system.runtime->materialStateFields();
        const auto it = fields.find(group);
        if (it != fields.end())
            return RayTrophiSim::MaterialStateFieldSystem::summarizeIntegrity(it->second);
    }
    return {};
}

void SceneData::processFractureImpacts() {
    if (!rigid_body_system || !rigid_body_system->contactEventsEnabled()) return;

    // Sparse bridge: request a host summary snapshot every eight physics calls,
    // never hand the full MSF field to Jolt.
    if ((fracture_summary_tick_++ & 7u) == 0u) {
        for (auto& system : particle_systems)
            if (system.runtime && system.runtime->hasMaterialStateFields())
                system.runtime->requestMaterialStateFieldReadback();
    }
    const auto& events = rigid_body_system->contactEvents();
    if (events.empty()) return;

    auto findBreakable = [&](const std::string& source) -> RayTrophiSim::RigidBodyObject* {
        if (source.empty()) return nullptr;
        for (auto& body : rigid_bodies)
            if (body.getBreakable() && !body.broken && body.source_name == source)
                return &body;
        return nullptr;
    };

    for (const auto& event : events) {
        auto* hit = findBreakable(event.source_a);
        Vec3 direction = event.normal;
        if (!hit) {
            hit = findBreakable(event.source_b);
            direction = event.normal * -1.0f;
        }
        if (!hit) continue;

        applyFractureImpulse(hit->getFractureGroup(), event.point,
                             direction, event.impulse);
    }
}

bool SceneData::applyFractureImpulse(const std::string& group,
                                     const Vec3& point,
                                     const Vec3& direction,
                                     float impulse) {
    RayTrophiSim::RigidBodyObject* authored = nullptr;
    for (auto& body : rigid_bodies) {
        if (body.getBreakable() && !body.broken &&
            body.getFractureGroup() == group) {
            authored = &body;
            break;
        }
    }
    if (!authored) return false;
    const auto summary = fractureIntegritySummary(group);
    const float threshold = RayTrophiSim::effectiveFractureThreshold(
        *authored, summary);
    if (std::max(impulse, 0.0f) < threshold) return false;

    Vec3 safe_direction = direction;
    const float length = safe_direction.length();
    if (length > 1e-5f) safe_direction = safe_direction * (1.0f / length);
    else safe_direction = Vec3(0.0f, 1.0f, 0.0f);
    breakFractureGroup(group, point, safe_direction,
                       std::max(2.0f, impulse * 0.5f));
    return true;
}
