#include "UI/ParticleBillboardBuilder.h"

#include "ParticleSimulation.h"
#include "scene_data.h"

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

void appendParticleSystems(const SceneData& scene, ParticleBillboardUpload& out,
                           std::size_t& drawn) {
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
        const auto& runtime = *system.runtime;
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

// Fluid domains in Particles render mode: one constant LUT row per domain
// (domain colour, opacity 0.92, width 2 * radius), alpha blended.
void appendGridDomainParticles(const SceneData& scene, ParticleBillboardUpload& out,
                               std::size_t& drawn) {
    for (const auto& system : scene.particle_systems) {
        if (!system.visible || !system.enabled || !system.runtime) continue;
        const auto& domains = system.runtime->gridDomains();
        const auto& states = system.runtime->gridDomainStates();
        for (std::size_t d = 0; d < domains.size() && d < states.size(); ++d) {
            const auto& desc = domains[d];
            const auto& state = states[d];
            if (!desc.enabled || desc.type != RayTrophiSim::SimulationDomainType::Fluid ||
                desc.fluid_render_mode != RayTrophiSim::Fluid::FluidRenderMode::Particles ||
                !state.valid || state.particles.position.empty()) {
                continue;
            }
            const float radius = std::max(1.0e-4f,
                std::max(state.voxel_size, 1.0e-4f) *
                desc.fluid_particle_radius_factor *
                desc.fluid_particle_size_multiplier);

            RayTrophiSim::ParticleAppearanceProfile look;
            look.name = "Fluid Particles";
            look.blend = RayTrophiSim::ParticleAppearanceBlend::Alpha;
            look.color_ramp = {{0.0f, desc.fluid_particle_color}};
            look.opacity_curve = {{0.0f, 0.92f}};
            look.size_curve = {{0.0f, 2.0f * radius}};
            look.emission_curve = {{0.0f, 1.0f}};
            std::vector<float> row;
            RayTrophiSim::bakeParticleAppearanceLut(look, row);
            const uint32_t lut_row = rowCount(out.lut);
            appendRow(out.lut, row.data());

            for (const Vec3& center : state.particles.position) {
                if (drawn >= kMaxBillboards) return;
                if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
                    !std::isfinite(center.z)) continue;
                pushQuad(out.alpha, center.x, center.y, center.z, 0.0f, lut_row, 1.0f);
                ++drawn;
            }
        }
    }
}

} // namespace

void build(const SceneData& scene, ParticleBillboardUpload& out) {
    out.additive.clear();
    out.alpha.clear();
    out.lut.clear();
    // Row 0: the fallback appearance, for particles whose profile id resolves
    // to nothing. Always present so the LUT binding is never empty.
    std::vector<float> fallback;
    RayTrophiSim::bakeParticleAppearanceLut(RayTrophiSim::fallbackParticleAppearance(),
                                            fallback);
    appendRow(out.lut, fallback.data());

    std::size_t drawn = 0;
    appendParticleSystems(scene, out, drawn);
    appendGridDomainParticles(scene, out, drawn);
}

} // namespace ParticleBillboardBuilder
