#include "scene_data.h"
#include "ThermalFractureBridge.h"

#include <algorithm>
#include <cmath>

float RayTrophiSim::effectiveFractureThreshold(
    const RigidBodyObject& body,
    const MaterialIntegritySummary& summary,
    float group_mass_kg) {
    // Velocity -> impulse. A group whose mass never got computed falls back to
    // 1 kg rather than to zero: zero would make it break on the first breath of
    // pressure, and "I could not weigh it" must not read as "it is weightless".
    const float mass = group_mass_kg > 0.0f ? group_mass_kg : 1.0f;
    float threshold = body.getBreakVelocity() * mass;
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
    // ★★ A CLUSTER IS NOT AN OBJECT, AND THIS LOOKUP USED TO ASSUME IT WAS.
    //
    // MSF fields are keyed by the SOURCE OBJECT. While every object had exactly
    // one fracture group the two names coincided and `fields.find(group)` worked
    // by accident. Splitting an object into structural clusters broke that
    // silently: only the cluster that inherited the object's own name found a
    // field, and every other cluster fell through to the default summary —
    // integrity 1.0, full strength, no thermal weakening, forever. Five clusters
    // out of six were quietly exempt from the entire feature, and the numbers
    // looked plausible enough (1.0 is what a pristine object reports) that
    // nothing flagged it.
    const auto sit = fracture_group_source_.find(group);
    const std::string& source = sit != fracture_group_source_.end() ? sit->second : group;

    // The cluster's own region, so it is judged on the damage IT carries rather
    // than on the object average.
    //
    // ★ The pad used to be 25% of the box plus 2 cm, sized for a box built from
    // shard CENTRES — which sits strictly INSIDE the shard and therefore missed
    // the surface elements entirely. The box is now the shard's real vertex
    // AABB, so the surface is already inside it and that pad became pure
    // overlap: neighbouring clusters read each other's damage and converged on
    // one shared average. What is left is slack for source elements whose
    // centroid falls just outside a shard that cuts across them.
    Vec3 bounds_min(0.0f), bounds_max(0.0f);
    const bool bounded = fractureGroupBounds(group, bounds_min, bounds_max);
    if (bounded) {
        const Vec3 pad = (bounds_max - bounds_min) * 0.05f + Vec3(0.002f, 0.002f, 0.002f);
        bounds_min -= pad;
        bounds_max += pad;
    }

    for (const auto& system : particle_systems) {
        if (!system.runtime) continue;
        const auto& fields = system.runtime->materialStateFields();
        const auto it = fields.find(source);
        if (it == fields.end()) continue;
        if (bounded) {
            const auto local =
                RayTrophiSim::MaterialStateFieldSystem::summarizeIntegrityInBounds(
                    it->second, bounds_min, bounds_max);
            // No element landed in the region (a cluster of a mesh whose UV
            // unwrap leaves that area uncovered). Falling back to the whole
            // object is wrong-but-conservative; reporting a confident 1.0 would
            // be wrong-and-dangerous. The fallback marks itself via
            // `regional == false` — silently substituting the object average is
            // how every cluster came to report the same number unnoticed.
            if (local.valid) return local;
        }
        return RayTrophiSim::MaterialStateFieldSystem::summarizeIntegrity(it->second);
    }
    return {};
}

void SceneData::accumulateFractureGroupBounds(
    const std::unordered_map<std::string, std::string>& node_to_group,
    std::unordered_map<std::string, RayTrophiSim::FractureGroupBounds>& out) const {
    if (node_to_group.empty()) return;
    for (const auto& object : world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(object);
        if (tri) {
            const auto it = node_to_group.find(tri->getNodeName());
            if (it == node_to_group.end()) continue;
            RayTrophiSim::FractureGroupBounds& bounds = out[it->second];
            for (int i = 0; i < 3; ++i) bounds.add(tri->getVertexPosition(i));
            continue;
        }
        // Flat (direct SoA) meshes carry no per-face facades, so a flat-migrated
        // shard is invisible to the Triangle branch above. Without this it would
        // be silently treated as a zero-area group sitting at the origin.
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || !mesh->geometry) continue;
        const auto it = node_to_group.find(mesh->nodeName);
        if (it == node_to_group.end()) continue;
        const DNA::GeometryDetail* geometry = mesh->geometry.get();
        const Vec3* positions = geometry->get_attribute_data<Vec3>("P");
        if (!positions) continue;
        RayTrophiSim::FractureGroupBounds& bounds = out[it->second];
        const std::size_t vertex_count = geometry->get_vertex_count();
        for (std::size_t v = 0; v < vertex_count; ++v) bounds.add(positions[v]);
    }
}

// Closed-mesh volume of one node, by the divergence theorem over its triangles.
//
// ★★ WHY THIS EXISTS AT ALL: the volume-derived mass is computed inside
// ensureBodyCreated, which runs when the SIMULATION creates the body. Until
// somebody presses play, `rb.mass` is still the authored 1.0 — so the panel, the
// scripting API and the break threshold all saw a 40-shard object weighing
// exactly 40 kg, one kilogram per piece, regardless of size. That is the same
// reading the "every shard weighs 1 kg" bug produced, and it is indistinguishable
// from it without looking at the numbers. Measured here so the mass is truthful
// in the scene data itself, at authoring time.
//
// Flat SoA first, facade second: geometry in this project is flat, and a scan
// that only knows facades silently answers zero.
float SceneData::nodeMeshVolume(const std::string& node) const {
    if (node.empty()) return 0.0f;
    double volume6 = 0.0;
    for (const auto& object : world.objects) {
        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
            if (mesh->nodeName != node || !mesh->geometry) continue;
            const DNA::GeometryDetail* geometry = mesh->geometry.get();
            const Vec3* positions = geometry->get_attribute_data<Vec3>("P");
            if (!positions) continue;
            // ★ The SoA is UNINDEXED: GeometryDetail stores per-corner vertices
            // and carries no index buffer, so triangles are consecutive triples.
            // (Welding is a separate cache built on demand, not part of the
            // geometry.) Anything that assumes an index buffer here does not
            // compile, which is the good outcome — it fails loudly.
            const std::size_t vertex_count = geometry->get_vertex_count();
            for (std::size_t v = 0; v + 2 < vertex_count; v += 3)
                volume6 += positions[v].dot(positions[v + 1].cross(positions[v + 2]));
            continue;
        }
        if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            if (tri->getNodeName() != node) continue;
            volume6 += tri->getVertexPosition(0).dot(
                tri->getVertexPosition(1).cross(tri->getVertexPosition(2)));
        }
    }
    // Absolute value: winding decides the sign, and a mesh with mixed winding
    // would otherwise report a negative (or cancelling) volume. Not a silent
    // clamp — a genuinely empty node returns 0 and the caller decides.
    return static_cast<float>(std::fabs(volume6) / 6.0);
}

bool SceneData::fractureGroupBounds(const std::string& group,
                                    Vec3& out_min, Vec3& out_max) const {
    // ★ Shard GEOMETRY, not shard centres. The old version accumulated
    // nodeWorldCenter(shard), which makes a one-shard cluster a single point:
    // extent (0,0,0) -> projected area 0 -> impulse 0 -> that cluster could
    // never be broken by pressure, and multi-shard clusters lost half a shard
    // of reach on every face.
    std::unordered_map<std::string, std::string> node_to_group;
    for (const auto& rb : rigid_bodies) {
        if (!rb.getBreakable() || rb.getFractureGroup() != group ||
            rb.source_name.empty()) continue;
        node_to_group.emplace(rb.source_name, group);
    }
    std::unordered_map<std::string, RayTrophiSim::FractureGroupBounds> bounds;
    accumulateFractureGroupBounds(node_to_group, bounds);

    const auto it = bounds.find(group);
    // A body whose node carries no geometry simply contributes nothing above, so
    // a leaked entry can no longer drag the region onto the world origin.
    if (it == bounds.end() || !it->second.any) return false;
    out_min = it->second.min;
    out_max = it->second.max;
    return true;
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
    const float mass = fractureGroupMass(group);
    const float threshold = RayTrophiSim::effectiveFractureThreshold(
        *authored, summary, mass);
    if (std::max(impulse, 0.0f) < threshold) return false;

    Vec3 safe_direction = direction;
    const float length = safe_direction.length();
    if (length > 1e-5f) safe_direction = safe_direction * (1.0f / length);
    else safe_direction = Vec3(0.0f, 1.0f, 0.0f);
    // ★★★ THIS is why a blast blew the structure to pieces. `strength` is
    // consumed as a LAUNCH VELOCITY (breakFractureGroup writes it straight into
    // pending_launch_velocity), and `impulse * 0.5` fed it newton-seconds. A
    // 400 N.s blast therefore launched every shard at 200 m/s — not a strong
    // blast, a unit error, and one that reads on screen as "the tuning is too
    // high" rather than as a bug. No amount of turning the pressure scale down
    // would have fixed the shape of it.
    //
    // Impulse / mass is the velocity that impulse actually imparts, so the
    // pieces now leave at the speed the physics says they should, and a heavy
    // group visibly resists what throws a light one.
    const float scatter_speed = std::max(1.0f, impulse * 0.5f /
                                         (mass > 0.0f ? mass : 1.0f));
    breakFractureGroup(group, point, safe_direction, scatter_speed);
    return true;
}

// Summed mass of every body in the group, in kilograms.
//
// ★ Read from the AUTHORED bodies rather than from Jolt, so it answers the same
// number whether or not the simulation is running — the threshold is queried by
// panels and scripts while stopped, and a threshold that changed when you
// pressed play would be unexplainable.
float SceneData::fractureGroupMass(const std::string& group) const {
    float total = 0.0f;
    for (const auto& body : rigid_bodies) {
        if (body.getFractureGroup() != group) continue;
        total += std::max(body.mass, 0.0f);
    }
    return total;
}
