#include "scene_data.h"

#include <algorithm>
#include <unordered_set>

void SceneData::queueStructuralImpulse(
    RayTrophiSim::StructuralImpulseEvent event) {
    event.sequence = ++structural_impulse_sequence_;
    event.radius = std::max(event.radius, 0.001f);
    event.peak_pressure_kpa = std::max(event.peak_pressure_kpa, 0.0f);
    event.duration_seconds = std::max(event.duration_seconds, 0.0f);
    event.coupling = std::max(event.coupling, 0.0f);
    structural_impulse_events_.push_back(event);
    ++structural_impulse_stats_.queued;
}

void SceneData::processStructuralImpulseEvents() {
    if (structural_impulse_events_.empty()) return;
    std::vector<RayTrophiSim::StructuralImpulseEvent> events;
    events.swap(structural_impulse_events_);

    for (const auto& event : events) {
        ++structural_impulse_stats_.consumed;
        structural_impulse_stats_.last_peak_pressure_kpa = event.peak_pressure_kpa;
        structural_impulse_stats_.last_max_impulse = 0.0f;
        std::unordered_set<std::string> visited;
        for (const auto& body : rigid_bodies) {
            if (!body.getBreakable() || body.broken) continue;
            const std::string group = body.getFractureGroup();
            if (group.empty() || !visited.insert(group).second) continue;
            Vec3 sum(0.0f);
            int count = 0;
            for (const auto& shard : rigid_bodies) {
                if (!shard.getBreakable() || shard.broken ||
                    shard.getFractureGroup() != group) continue;
                sum += nodeWorldCenter(shard.source_name);
                ++count;
            }
            if (count == 0) continue;
            const Vec3 center = sum * (1.0f / static_cast<float>(count));
            Vec3 radial = center - event.center;
            const float distance = radial.length();
            if (distance > event.radius) continue;
            const float falloff = 1.0f - distance / event.radius;
            const float impulse = event.peak_pressure_kpa *
                event.duration_seconds * event.coupling * falloff;
            if (impulse <= 0.0f) continue;
            ++structural_impulse_stats_.affected_groups;
            structural_impulse_stats_.last_max_impulse = std::max(
                structural_impulse_stats_.last_max_impulse, impulse);
            const Vec3 direction = distance > 1e-5f
                ? radial * (1.0f / distance) : Vec3(0.0f, 1.0f, 0.0f);
            if (applyFractureImpulse(group, event.center, direction, impulse))
                ++structural_impulse_stats_.fractured_groups;
        }
    }
}
