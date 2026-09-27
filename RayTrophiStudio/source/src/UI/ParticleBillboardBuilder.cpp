#include "UI/ParticleBillboardBuilder.h"

#include "ParticleSimulation.h"
#include "scene_data.h"
#include "InstanceManager.h"
#include "globals.h"
#include "Viewport/SphereImpostorAvailability.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ParticleBillboardBuilder {
namespace {

constexpr std::size_t kMaxBillboards = 60000;  // safety cap across all systems

void appendRow(std::vector<float>& lut, const float* row) {
    lut.insert(lut.end(), row, row + RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
}

uint32_t rowCount(const std::vector<float>& lut) {
    return static_cast<uint32_t>(lut.size() /
                                 RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
}

void pushQuad(std::vector<ParticleBillboardVertex>& out, float x, float y, float z,
              float age, uint32_t lut_row, float size_scale) {
    static constexpr float kCorners[6][2] = {
        {-1.f, -1.f}, {1.f, -1.f}, {1.f, 1.f},
        {-1.f, -1.f}, {1.f, 1.f}, {-1.f, 1.f},
    };
    for (const auto& c : kCorners) {
        ParticleBillboardVertex v{};
        v.center[0] = x;
        v.center[1] = y;
        v.center[2] = z;
        v.corner[0] = c[0];
        v.corner[1] = c[1];
        v.age = age;
        v.lut_row = static_cast<float>(lut_row);
        v.size_scale = size_scale;
        out.push_back(v);
    }
}

// A device-resident system on the viewport's own VkDevice is drawn from the
// simulation buffers (vertex pulling). Its profile rows go into the LUT like
// any other system's; the id -> row map goes into row_lookup.
bool appendPulledSystem(const RayTrophiSim::ParticleSimulationSystem& runtime,
                        void* viewport_device, ParticleBillboardUpload& out) {
    RayTrophiSim::ParticleResidentDrawBuffers device;
    if (!viewport_device || !runtime.residentDrawBuffers(device) ||
        device.device != viewport_device || device.capacity == 0) {
        return false;
    }
    ParticlePulledDraw draw;
    for (uint32_t i = 0; i < kPulledStreamCount; ++i) {
        draw.buffers[i] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device.buffers[i]));
    }
    draw.particle_count = device.capacity;
    draw.state_version = device.state_version;
    uint32_t max_id = 0;
    for (const auto& profile : runtime.appearanceProfiles()) {
        max_id = std::max(max_id, profile.id);
    }
    draw.lookup_offset = static_cast<uint32_t>(out.row_lookup.size());
    draw.lookup_count = max_id + 1;
    out.row_lookup.resize(out.row_lookup.size() + draw.lookup_count, 0u);  // 0 = fallback row
    for (const auto& profile : runtime.appearanceProfiles()) {
        const bool alpha = profile.blend == RayTrophiSim::ParticleAppearanceBlend::Alpha;
        draw.has_alpha = draw.has_alpha || alpha;
        out.row_lookup[draw.lookup_offset + profile.id] =
            rowCount(out.lut) | (alpha ? 0x80000000u : 0u);
        appendRow(out.lut, runtime.appearanceLutRow(profile.id));
    }
    out.pulled.push_back(draw);
    return true;
}

void appendParticleSystems(const SceneData& scene, void* viewport_device,
                           ParticleBillboardUpload& out, std::size_t& drawn) {
    using RayTrophiSim::ParticleAppearanceBlend;
    struct RowInfo {
        uint32_t row = 0;
        bool alpha = false;
    };
    std::unordered_map<uint32_t, RowInfo> rows;

    for (const auto& system : scene.particle_systems) {
        if (!system.visible || !system.runtime || system.render.emitter_only) {
            continue;
        }
        auto& runtime = *system.runtime;
        if (appendPulledSystem(runtime, viewport_device, out)) {
            continue;
        }
        // Not pullable: either the host already holds the state, or the
        // simulation runs on ANOTHER VkDevice and its buffers cannot be bound
        // here. The second is a counted snapshot (particle.stats), the same
        // consequence live gas has on a foreign device.
        if (runtime.kinematicResidency() != RayTrophiSim::ParticleKinematicResidency::Host) {
            runtime.syncHostState("foreign_device");
        }
        rows.clear();
        for (const auto& profile : runtime.appearanceProfiles()) {
            RowInfo info;
            info.row = rowCount(out.lut);
            info.alpha = profile.blend == ParticleAppearanceBlend::Alpha;
            appendRow(out.lut, runtime.appearanceLutRow(profile.id));
            rows[profile.id] = info;
        }

        const auto& buf = runtime.buffers();
        const std::size_t cap = buf.alive.size();
        for (std::size_t i = 0; i < cap && drawn < kMaxBillboards; ++i) {
            if (buf.alive[i] == 0u) continue;
            const float x = buf.position_x[i];
            const float y = buf.position_y[i];
            const float z = buf.position_z[i];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

            const float life = buf.lifetime_seconds[i];
            const float age = life > 1e-6f
                ? std::clamp(buf.age_seconds[i] / life, 0.0f, 1.0f) : 0.0f;
            const uint32_t id = i < buf.appearance_profile.size()
                ? buf.appearance_profile[i] : 0u;
            const float scale = i < buf.size_scale.size() ? buf.size_scale[i] : 1.0f;

            // Unknown / zero id -> row 0, the fallback (additive).
            RowInfo info;
            if (auto it = rows.find(id); it != rows.end()) {
                info = it->second;
            }
            pushQuad(info.alpha ? out.alpha : out.additive, x, y, z, age, info.row, scale);
            ++drawn;
        }
        if (drawn >= kMaxBillboards) break;
    }
}

} // namespace

void build(const SceneData& scene, void* viewport_device, ParticleBillboardUpload& out) {
    out.spheres.clear();
    const bool spherePath = viewport_device && g_sphere_impostor_ready.load() &&
        g_solid_viewport_active && !g_material_preview_viewport_active;
    for (auto& group : InstanceManager::getInstance().getGroups()) {
        const bool active = spherePath && group.raster_sphere_candidate;
        if (group.raster_sphere_active != active) {
            group.raster_sphere_active = active;
            // Same geometry, different raster representation: the generation
            // gate would drop a bare rebuild request and leave the pool out.
            g_viewport_raster_cache_invalid = true;
            g_viewport_raster_rebuild_pending = true;
        }
        if (!active) {
            continue;
        }
        for (const auto& instance : group.instances) {
            const auto& p = instance.position;
            const float radius = instance.scale.x * 0.5f;
            if (radius > 0.0f && std::isfinite(radius) && std::isfinite(p.x) &&
                std::isfinite(p.y) && std::isfinite(p.z)) {
                out.spheres.push_back({{p.x, p.y, p.z, radius}});
            }
        }
    }
    out.additive.clear();
    out.alpha.clear();
    out.lut.clear();
    out.pulled.clear();
    out.row_lookup.assign(1, 0u);  // never empty: keeps the binding valid
    // Row 0: the fallback appearance, for particles whose profile id resolves
    // to nothing. Always present so the LUT binding is never empty.
    std::vector<float> fallback;
    RayTrophiSim::bakeParticleAppearanceLut(RayTrophiSim::fallbackParticleAppearance(),
                                            fallback);
    appendRow(out.lut, fallback.data());

    std::size_t drawn = 0;
    appendParticleSystems(scene, viewport_device, out, drawn);
    // Fluid domains in Particles mode are NOT drawn here: the render bridge
    // already draws them as sphere instances in raster and RT. A second copy
    // as billboards read the live positions while the instances lagged a
    // step, so during playback a blue disc poked out in front of every sphere.
}

} // namespace ParticleBillboardBuilder
