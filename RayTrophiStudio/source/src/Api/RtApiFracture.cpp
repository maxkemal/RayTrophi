#include "RtApiInternal.h"
#include "ThermalFractureBridge.h"

#include <algorithm>

namespace rtapi {

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
    out_info.base_break_impulse = authored->getBreakImpulse();
    out_info.integrity_weakening = authored->getIntegrityWeakening();
    out_info.integrity_exponent = authored->getIntegrityExponent();
    out_info.minimum_threshold_scale = authored->getMinimumThresholdScale();

    const auto summary = g_ctx->scene.fractureIntegritySummary(group);
    if (summary.valid) {
        out_info.mean_integrity = summary.mean_integrity;
        out_info.minimum_integrity = summary.minimum_integrity;
        out_info.remaining_support_ratio = summary.remaining_support_ratio;
        if (out_info.integrity_weakening) {
            out_info.effective_break_impulse =
                RayTrophiSim::effectiveFractureThreshold(*authored, summary);
            return Result::success();
        }
    }
    out_info.effective_break_impulse = out_info.base_break_impulse;
    return Result::success();
}

Result makePhysicsFractureGroup(const std::string& group,
                                const std::vector<std::string>& shard_objects,
                                float break_impulse,
                                bool integrity_weakening,
                                float integrity_exponent,
                                float minimum_threshold_scale,
                                FractureGroupInfo& out_info) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (group.empty()) return Result::fail("fracture group name cannot be empty");
    if (shard_objects.empty()) return Result::fail("fracture group needs at least one shard object");
    for (const auto& object : shard_objects)
        if (!objectExists(object)) return Result::fail("fracture shard object not found: " + object);
    g_ctx->scene.makeFractureGroupBreakable(
        group, shard_objects, std::max(break_impulse, 0.001f),
        integrity_weakening, std::max(integrity_exponent, 0.01f),
        std::clamp(minimum_threshold_scale, 0.0f, 1.0f));
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
    return Result::success();
}

} // namespace rtapi
