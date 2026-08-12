#include "RtApiInternal.h"
#include "ThermalFractureBridge.h"

#include <algorithm>
#include <map>

namespace rtapi {

// ── Shard GENERATION over the API ───────────────────────────────────────────
//
// ★ Until this existed, the destruction pipeline had a manual step in the
// middle: a script could weaken an object, and a script could make its shards
// breakable, but the cut itself was reachable only by pressing a button. Every
// automated test therefore stopped halfway and handed the rest to a human, and
// on a one-person project that is the same as not being tested.
//
// The generator lives in SceneUI because it owns the parked originals and the
// shard bookkeeping. Reimplementing it here would fork that state — two paths
// that park, name and erase shards differently is precisely the class of bug
// this codebase keeps paying for. So this drives the same call the panel does.
Result fractureObject(const std::string& node, int site_count, uint32_t seed,
                      int pattern, int cluster_count, bool exact_surface,
                      float preview_gap, FractureResultInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (node.empty()) return Result::fail("object name cannot be empty");
    if (!objectExists(node)) return Result::fail("object not found: " + node);
    if (pattern < 0 || pattern > 2)
        return Result::fail("pattern must be 0 (uniform), 1 (impact) or 2 (thermal)");

    SceneData::FractureRecipe settings;
    settings.site_count    = std::max(1, site_count);
    settings.seed          = seed ? seed : 1u;
    settings.pattern       = pattern;
    settings.cluster_count = std::max(1, cluster_count);
    settings.exact_surface = exact_surface;
    settings.preview_gap   = std::clamp(preview_gap, 0.0f, 0.9f);
    // sites left EMPTY on purpose: this is a fresh cut, not a replay, so the
    // thermal pattern still gets to read the damage field.

    ui.fractureSelectedMesh(*g_ctx, node, settings.site_count, settings.seed,
                            settings.pattern, &settings);

    out_info = FractureResultInfo{};
    out_info.object = node;
    const auto sit = ui.fracture_shard_nodes_.find(node);
    if (sit == ui.fracture_shard_nodes_.end() || sit->second.empty()) {
        // ★ Report the failure rather than an empty success. The generator
        // refuses degenerate input (an open or flat mesh has no 3D hull) and
        // restores the original; a caller that saw shard_count 0 with ok=true
        // would go on to build a fracture group out of nothing.
        return Result::fail("fracture produced no shards for '" + node +
                            "' - the mesh is not solid (flat, open or degenerate)");
    }
    out_info.shard_objects = sit->second;
    const auto cit = ui.fracture_shard_clusters_.find(node);
    if (cit != ui.fracture_shard_clusters_.end() &&
        cit->second.size() == sit->second.size()) {
        out_info.shard_clusters = cit->second;
        for (int c : cit->second)
            out_info.cluster_count = std::max(out_info.cluster_count, c + 1);
    } else {
        out_info.cluster_count = 1;
    }
    const auto rit = g_ctx->scene.fracture_recipes.find(node);
    if (rit != g_ctx->scene.fracture_recipes.end())
        out_info.site_count = static_cast<int>(rit->second.sites.size());
    return Result::success();
}

Result unfractureObject(const std::string& node) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (node.empty()) return Result::fail("object name cannot be empty");
    if (!ui.isMeshFractured(node))
        return Result::fail("object is not fractured: " + node);
    ui.unfractureMesh(*g_ctx, node);
    return Result::success();
}

// The group names "Make Breakable" would produce for this object, so a script
// can register the same clusters the panel would without re-deriving the naming
// convention (cluster 0 keeps the object's own name; the rest get a suffix).
Result fractureClusterGroups(const std::string& node,
                             std::vector<std::string>& out_groups,
                             std::vector<std::vector<std::string>>& out_members) {
    if (!g_ctx) return notBound();
    out_groups.clear();
    out_members.clear();
    const auto sit = ui.fracture_shard_nodes_.find(node);
    if (sit == ui.fracture_shard_nodes_.end() || sit->second.empty())
        return Result::fail("object is not fractured: " + node);
    const auto cit = ui.fracture_shard_clusters_.find(node);
    const bool clustered = cit != ui.fracture_shard_clusters_.end() &&
                           cit->second.size() == sit->second.size();
    std::map<int, std::vector<std::string>> by_cluster;
    for (std::size_t i = 0; i < sit->second.size(); ++i)
        by_cluster[clustered ? cit->second[i] : 0].push_back(sit->second[i]);
    for (const auto& entry : by_cluster) {
        out_groups.push_back(entry.first == 0
            ? node : node + "__cluster_" + std::to_string(entry.first));
        out_members.push_back(entry.second);
    }
    return Result::success();
}

Result getPhysicsFractureGroup(const std::string& group,
                               FractureGroupInfo& out_info) {
    if (!g_ctx) return notBound();
    out_info = FractureGroupInfo{};
    out_info.group = group;
    const RayTrophiSim::RigidBodyObject* authored = nullptr;
    for (const auto& rb : g_ctx->scene.rigid_bodies) {
        if (!rb.getBreakable() || rb.getFractureGroup() != group) continue;
        if (!authored) authored = &rb;
        ++out_info.shard_count;
        if (rb.broken) ++out_info.broken_count;
    }
    if (!authored) return Result::fail("fracture group not found: " + group);

    // Exactly the AABB the pressure bridge projects — same accumulator, same
    // shard vertices — so the reported geometry and the geometry the impulse was
    // computed from cannot disagree.
    Vec3 bounds_min(0.0f), bounds_max(0.0f);
    if (g_ctx->scene.fractureGroupBounds(group, bounds_min, bounds_max)) {
        out_info.world_center = (bounds_min + bounds_max) * 0.5f;
        out_info.world_extent = Vec3(std::max(bounds_max.x - bounds_min.x, 0.0f),
                                     std::max(bounds_max.y - bounds_min.y, 0.0f),
                                     std::max(bounds_max.z - bounds_min.z, 0.0f));
    }
    // ★ Both are reported, because they answer different questions and a test
    // that only saw one could not tell a strong group from a heavy one:
    //   break_velocity — what the artist authored, in m/s, mass-free
    //   break_impulse  — what an incoming impulse is actually compared against
    out_info.group_mass_kg = g_ctx->scene.fractureGroupMass(group);
    out_info.base_break_velocity = authored->getBreakVelocity();
    out_info.base_break_impulse = RayTrophiSim::effectiveFractureThreshold(
        *authored, RayTrophiSim::MaterialIntegritySummary(),
        out_info.group_mass_kg);
    out_info.integrity_weakening = authored->getIntegrityWeakening();
    out_info.integrity_exponent = authored->getIntegrityExponent();
    out_info.minimum_threshold_scale = authored->getMinimumThresholdScale();

    const auto summary = g_ctx->scene.fractureIntegritySummary(group);
    if (summary.valid) {
        out_info.integrity_regional = summary.regional;
        out_info.integrity_sampled_elements =
            static_cast<int>(summary.sampled_elements);
        out_info.mean_integrity = summary.mean_integrity;
        out_info.minimum_integrity = summary.minimum_integrity;
        out_info.remaining_support_ratio = summary.remaining_support_ratio;
        if (out_info.integrity_weakening) {
            out_info.effective_break_impulse =
                RayTrophiSim::effectiveFractureThreshold(*authored, summary,
                                                         out_info.group_mass_kg);
            return Result::success();
        }
    }
    out_info.effective_break_impulse = out_info.base_break_impulse;
    return Result::success();
}

Result makePhysicsFractureGroup(const std::string& group,
                                const std::vector<std::string>& shard_objects,
                                float break_velocity,
                                bool integrity_weakening,
                                float integrity_exponent,
                                float minimum_threshold_scale,
                                FractureGroupInfo& out_info,
                                const std::string& source_object) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (group.empty()) return Result::fail("fracture group name cannot be empty");
    if (shard_objects.empty()) return Result::fail("fracture group needs at least one shard object");
    for (const auto& object : shard_objects)
        if (!objectExists(object)) return Result::fail("fracture shard object not found: " + object);
    g_ctx->scene.makeFractureGroupBreakable(
        group, shard_objects, std::max(break_velocity, 0.001f),
        integrity_weakening, std::max(integrity_exponent, 0.01f),
        std::clamp(minimum_threshold_scale, 0.0f, 1.0f), source_object);
    return getPhysicsFractureGroup(group, out_info);
}

Result breakPhysicsFractureGroup(const std::string& group, float strength) {
    if (!g_ctx) return notBound();
    FractureGroupInfo info;
    Result found = getPhysicsFractureGroup(group, info);
    if (!found.ok) return found;
    g_ctx->scene.breakFractureGroupNow(group, std::max(strength, 0.0f));
    return Result::success();
}

Result applyPhysicsFractureImpulse(const std::string& group, Vec3 point,
                                   Vec3 direction, float impulse,
                                   bool& out_triggered) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    FractureGroupInfo info;
    Result found = getPhysicsFractureGroup(group, info);
    if (!found.ok) return found;
    out_triggered = g_ctx->scene.applyFractureImpulse(
        group, point, direction, std::max(impulse, 0.0f));
    return Result::success();
}

Result emitGasPressurePulse(const std::string& domain, Vec3 center,
                            float radius, float peak_pressure_kpa,
                            float duration_seconds, float coupling,
                            uint64_t& out_sequence) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    FluidDomainInfo info;
    Result found = getFluidDomain(domain, info);
    if (!found.ok) return found;
    if (info.type != "gas") return Result::fail("pressure pulse requires a gas domain");
    if (radius <= 0.0f || peak_pressure_kpa < 0.0f ||
        duration_seconds < 0.0f || coupling < 0.0f)
        return Result::fail("pressure pulse values must be non-negative and radius must be positive");
    if (center.x < info.domain_min.x || center.y < info.domain_min.y ||
        center.z < info.domain_min.z || center.x > info.domain_max.x ||
        center.y > info.domain_max.y || center.z > info.domain_max.z)
        return Result::fail("pressure pulse center is outside the gas domain");
    auto& runtime = g_ctx->scene.ensureParticleSimulationSystem();
    runtime.synchronizeGridDomainsNow();
    if (!runtime.injectGasPressurePulse(info.name, center, radius,
                                        peak_pressure_kpa))
        return Result::fail("gas pressure pulse could not reach a live domain grid");
    RayTrophiSim::StructuralImpulseEvent event;
    event.domain = info.name;
    event.center = center;
    event.radius = radius;
    event.peak_pressure_kpa = peak_pressure_kpa;
    event.duration_seconds = duration_seconds;
    event.coupling = coupling;
    g_ctx->scene.queueStructuralImpulse(event);
    out_sequence = g_ctx->scene.structuralImpulseStats().queued;
    return Result::success();
}

Result getStructuralImpulseInfo(StructuralImpulseInfo& out_info) {
    if (!g_ctx) return notBound();
    const auto& stats = g_ctx->scene.structuralImpulseStats();
    out_info.queued = stats.queued;
    out_info.consumed = stats.consumed;
    out_info.affected_groups = stats.affected_groups;
    out_info.fractured_groups = stats.fractured_groups;
    out_info.last_peak_pressure_kpa = stats.last_peak_pressure_kpa;
    out_info.last_max_impulse = stats.last_max_impulse;
    out_info.last_projected_area_m2 = stats.last_projected_area_m2;
    return Result::success();
}

} // namespace rtapi
