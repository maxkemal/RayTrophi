#include "ParticleSimulation.h"
#include "Fluid/FluidRenderProxy.h"
#include "InstanceGroup.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {

namespace {

uint32_t proxyHash(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

float proxyUnitFloat(uint32_t value) {
    return static_cast<float>(proxyHash(value) & 0xffffu) / 65535.0f;
}

Vec3 proxyPairDirection(uint32_t parent_index, uint32_t pair_index) {
    const uint32_t seed = proxyHash(
        parent_index ^ (pair_index * 0x85ebca6bu) ^ 0x9e3779b9u);
    Vec3 direction(
        proxyUnitFloat(seed ^ 0x68bc21ebu) * 2.0f - 1.0f,
        proxyUnitFloat(seed ^ 0x02e5be93u) * 2.0f - 1.0f,
        proxyUnitFloat(seed ^ 0x967a889bu) * 2.0f - 1.0f);
    const float length_squared = direction.dot(direction);
    if (!(length_squared > 1e-6f)) {
        return Vec3(1.0f, 0.0f, 0.0f);
    }
    return direction / std::sqrt(length_squared);
}

} // namespace

FluidRenderProxyLayout resolveFluidRenderProxyLayout(
    bool virtual_grains_enabled,
    uint32_t requested_children) {
    FluidRenderProxyLayout layout;
    if (!virtual_grains_enabled) {
        return layout;
    }

    layout.children_per_parent = std::clamp(
        requested_children, 1u, kFluidRenderProxyMaxChildren);
    layout.child_radius_scale = 1.0f /
        std::cbrt(static_cast<float>(layout.children_per_parent));
    layout.spread_radius_scale = 1.0f - layout.child_radius_scale;
    return layout;
}

uint32_t limitFluidRenderProxyChildren(uint32_t requested_children,
                                       uint64_t parent_count,
                                       uint64_t max_visual_spheres) {
    const uint32_t requested = std::clamp(
        requested_children, 1u, kFluidRenderProxyMaxChildren);
    if (parent_count == 0 || max_visual_spheres == 0) {
        return 1;
    }
    const uint64_t safe_budget = std::min(
        max_visual_spheres,
        kFluidRenderProxyMaxVisualSphereBudget);
    const uint64_t available = std::max<uint64_t>(1, safe_budget / parent_count);
    return static_cast<uint32_t>(std::min<uint64_t>(requested, available));
}

uint64_t fluidRenderProxyCarrierCapacity(uint64_t primary_capacity,
                                         uint64_t secondary_capacity) {
    const uint64_t primary = std::min(
        primary_capacity,
        kFluidRenderProxyMaxVisualSphereBudget);
    return primary + std::min(
        secondary_capacity,
        kFluidRenderProxyMaxVisualSphereBudget - primary);
}

float fluidRenderProxyObjectRandom(int group_id,
                                   uint32_t parent_index,
                                   uint32_t child_index) {
    uint32_t key = static_cast<uint32_t>(std::max(group_id, 0));
    key = proxyHash(key ^ 0x9e3779b9u);
    key = proxyHash(key ^ (parent_index * 0x85ebca6bu));
    key = proxyHash(key ^ (child_index * 0xc2b2ae35u));
    return static_cast<float>(key >> 8u) * (1.0f / 16777216.0f);
}

bool resolveFluidRenderProxyTransform(const InstanceGroup& group,
                                      uint32_t parent_index,
                                      uint32_t child_index,
                                      InstanceTransform& transform) {
    if (parent_index >= group.instances.size()) {
        return false;
    }

    const InstanceTransform& parent = group.instances[parent_index];
    transform = parent;
    const uint32_t children = group.point_sphere_mode
        ? std::max<uint32_t>(1u, group.point_sphere_visual_children)
        : 1u;
    if (child_index >= children || children <= 1u) {
        return child_index == 0u;
    }

    const float parent_diameter = std::max({
        std::abs(parent.scale.x),
        std::abs(parent.scale.y),
        std::abs(parent.scale.z)});
    if (!(parent_diameter > 0.0f)) {
        transform.scale = Vec3(0.0f);
        return true;
    }

    Vec3 candidates[kFluidRenderProxyNeighborCandidates];
    uint32_t candidate_count = 0;
    constexpr uint32_t kNeighborSteps = kFluidRenderProxyNeighborCandidates / 2u;
    for (uint32_t slot = 0; slot < kNeighborSteps; ++slot) {
        const uint32_t step = 1u << slot;
        for (uint32_t side = 0; side < 2u; ++side) {
            if ((side == 0u && parent_index < step) ||
                (side == 1u && parent_index + step >= group.instances.size())) {
                continue;
            }
            const uint32_t candidate_index = side == 0u
                ? parent_index - step
                : parent_index + step;
            const InstanceTransform& candidate = group.instances[candidate_index];
            const float candidate_diameter = std::max({
                std::abs(candidate.scale.x),
                std::abs(candidate.scale.y),
                std::abs(candidate.scale.z)});
            if (!(candidate_diameter > 0.0f) ||
                !std::isfinite(candidate.position.x) ||
                !std::isfinite(candidate.position.y) ||
                !std::isfinite(candidate.position.z)) {
                continue;
            }
            candidates[candidate_count++] = candidate.position;
        }
    }

    const float parent_radius = parent_diameter * 0.5f;
    const float base_radius = parent_radius /
        std::cbrt(static_cast<float>(children));
    const float spread_radius = group.point_sphere_visual_spread_radius > 0.0f
        ? group.point_sphere_visual_spread_radius
        : std::max(0.0f, parent_radius - base_radius);
    const Vec3 offset = fluidRenderProxyNeighborOffset(
        parent_index,
        child_index,
        children,
        parent.position,
        candidates,
        candidate_count,
        spread_radius * kFluidRenderProxyNeighborSupportScale,
        spread_radius);
    const float radius = parent_radius * fluidRenderProxyChildRadiusScale(
        parent_index,
        child_index,
        children,
        group.point_sphere_visual_size_variation);
    transform.position = parent.position + offset;
    transform.scale = Vec3(radius * 2.0f);
    return true;
}

Vec3 fluidRenderProxyOffset(uint32_t parent_index,
                            uint32_t child_index,
                            uint32_t children_per_parent,
                            float spread_radius) {
    if (children_per_parent <= 1 || !(spread_radius > 0.0f)) {
        return Vec3(0.0f);
    }
    const uint32_t child = child_index % children_per_parent;
    const uint32_t pair = child / 2u;
    const float radial_scale = 0.72f + 0.28f * proxyUnitFloat(
        parent_index ^ (pair * 0xc2b2ae35u) ^ 0x27d4eb2fu);
    const float sign = (child & 1u) != 0u ? 1.0f : -1.0f;
    Vec3 direction = proxyPairDirection(parent_index, pair) * sign;
    if ((children_per_parent & 1u) != 0u &&
        child == children_per_parent - 1u) {
        direction = proxyPairDirection(parent_index, pair + 1u);
    }
    return direction *
        (spread_radius * radial_scale);
}

Vec3 fluidRenderProxyNeighborOffset(uint32_t parent_index,
                                    uint32_t child_index,
                                    uint32_t children_per_parent,
                                    const Vec3& parent_position,
                                    const Vec3* candidates,
                                    uint32_t candidate_count,
                                    float support_radius,
                                    float fallback_spread_radius) {
    if (children_per_parent <= 1 || !candidates || candidate_count == 0 ||
        !(support_radius > 0.0f)) {
        return fluidRenderProxyOffset(
            parent_index,
            child_index,
            children_per_parent,
            fallback_spread_radius);
    }

    const uint32_t child = child_index % children_per_parent;
    const uint32_t pair = child / 2u;
    const float sign = (child & 1u) != 0u ? 1.0f : -1.0f;
    Vec3 direction = proxyPairDirection(parent_index, pair) * sign;
    if ((children_per_parent & 1u) != 0u &&
        child == children_per_parent - 1u) {
        direction = proxyPairDirection(parent_index, pair + 1u);
    }

    Vec3 weighted_delta(0.0f);
    float weight_sum = 0.0f;
    for (uint32_t i = 0; i < candidate_count; ++i) {
        const Vec3 delta = candidates[i] - parent_position;
        const float distance_squared = delta.dot(delta);
        if (!(distance_squared > 1e-8f) ||
            distance_squared >= support_radius * support_radius) {
            continue;
        }
        const float distance = std::sqrt(distance_squared);
        const Vec3 neighbor_direction = delta / distance;
        const float alignment = std::max(0.0f, direction.dot(neighbor_direction));
        const float kernel = 1.0f - distance / support_radius;
        const float weight = kernel * kernel * (0.15f + 0.85f * alignment);
        weighted_delta = weighted_delta + delta * weight;
        weight_sum += weight;
    }
    if (!(weight_sum > 1e-6f)) {
        return fluidRenderProxyOffset(
            parent_index,
            child_index,
            children_per_parent,
            fallback_spread_radius);
    }

    Vec3 offset = weighted_delta * (0.52f / weight_sum);
    const float length = offset.length();
    const float max_length = fallback_spread_radius * 1.25f;
    if (length > max_length && max_length > 0.0f) {
        offset = offset * (max_length / length);
    }
    return offset;
}

float fluidRenderProxyChildRadiusScale(uint32_t parent_index,
                                       uint32_t child_index,
                                       uint32_t children_per_parent,
                                       float size_variation) {
    const uint32_t count = std::clamp(
        children_per_parent, 1u, kFluidRenderProxyMaxChildren);
    const float base_scale = 1.0f / std::cbrt(static_cast<float>(count));
    const float variation = std::clamp(
        size_variation, 0.0f, kFluidRenderProxyMaxSizeVariation);
    if (count <= 1u || !(variation > 0.0f)) {
        return base_scale;
    }
    const uint32_t child = child_index % count;
    if ((count & 1u) != 0u && child == count - 1u) {
        return base_scale;
    }
    const uint32_t pair = child / 2u;
    const float signed_random = proxyUnitFloat(
        parent_index ^ (pair * 0x165667b1u) ^ 0xd3a2646cu) * 2.0f - 1.0f;
    const float delta = variation * signed_random;
    const float smaller = 1.0f - delta;
    const float larger = 1.0f + delta;
    const float pair_normalization = std::cbrt(
        2.0f / (smaller * smaller * smaller + larger * larger * larger));
    const float raw_scale = (child & 1u) != 0u ? larger : smaller;
    return base_scale * pair_normalization * raw_scale;
}

} // namespace RayTrophiSim::Fluid

namespace RayTrophiSim {

bool ParticleSimulationSystem::fluidResidentPositionBuffer(
    std::size_t domain_index,
    const SimulationComputeContext& compute,
    FluidResidentPositionBuffer& out) const {
    out = {};
    if (domain_index >= grid_domain_states_.size() ||
        domain_index >= grid_domain_compute_buffers_.size() ||
        compute.backendType() != ComputeBackendType::VulkanCompute) {
        return false;
    }

    const SimulationGridDomainState& state = grid_domain_states_[domain_index];
    const bool matter = state.type == SimulationDomainType::Matter;
    if (matter && domain_index >= matter_liquid_compute_buffers_.size()) {
        return false;
    }
    const SimulationGridDomainComputeBuffers& buffers = matter
        ? matter_liquid_compute_buffers_[domain_index]
        : grid_domain_compute_buffers_[domain_index];
    const std::size_t count = state.particles.size();
    if (!state.valid || !simulationDomainHasLiquid(state.type) || count == 0 ||
        count > std::numeric_limits<uint32_t>::max() ||
        buffers.backend != ComputeBackendType::VulkanCompute ||
        !buffers.fluid_positions.valid() ||
        buffers.fluid_uploaded_particle_count < count) {
        return false;
    }

    out.device = compute.nativeDevice();
    out.positions = compute.nativeBufferPtr(buffers.fluid_positions);
    out.particle_count = static_cast<uint32_t>(count);
    out.state_version = state.version;
    return out.device != nullptr && out.positions != nullptr;
}

} // namespace RayTrophiSim
