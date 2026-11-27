#include "UI/ParticleBillboardBuilder.h"

#include "Fluid/FluidParticleVisualRadius.h"
#include "Fluid/FluidRenderProxy.h"
#include "Fluid/FluidViewResolver.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/SourceSplatRenderPlan.h"
#include "InstanceManager.h"
#include "Light.h"
#include "ParticleSimulation.h"
#include "Viewport/SphereImpostorAvailability.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <string>
#include <string_view>
#include <unordered_set>

namespace ParticleBillboardBuilder {
namespace {

constexpr std::size_t kMaxBillboards = 60000; // safety cap across all systems
void appendRow(std::vector<float> &lut, const float *row) {
  lut.insert(lut.end(), row,
             row + RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
}

uint32_t rowCount(const std::vector<float> &lut) {
  return static_cast<uint32_t>(
      lut.size() / RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
}

// ── Lit drops (2026-11-18) ──────────────────────────────────────────────────
// A system drawn as non-emissive spheres (ParticleRenderSettings: shape Sphere,
// emissive false - the same setting the RT bridge reads) is shaded in the
// billboard shaders as a lit liquid bead: the vertex stage weighs the scene's
// lights at the particle (the RT falloff: colour * intensity / d^2), the two
// strongest light it directly (wrap diffuse + GGX at render.roughness), the
// rest tint its ambient; Fresnel sky reflection on top. Emissive systems
// (sparks, confetti, the fallback) stay flat as before.
//
// The billboards are drawn in the display-referred overlay pass, after the
// tonemap, so absolute HDR units cannot be matched: brightness is RELATIVE,
// the particle's irradiance over the irradiance at a reference point (below
// the light centroid), square-rooted and clamped.
//
// The inputs ride in the BUILDER'S COPY of the row, in the sample slots [1].zw
// that the LUT layout leaves zero (the CPU sampler, the wet paint colour
// lookup and the vertex size/emission read only [1].xy):
//   sample 0: (roughness, 1 = lit)   sample 1: (light count, reference E)
//   light i (i < 8): sample 2+3i: (pos.x, pos.y)  3+3i: (pos.z, R)  4+3i: (G,
//   B)
// with RGB = colour * intensity (directional: a point 1e4 m away, x 1e8).
// No push constant, pipeline or header changes.
constexpr int kLitMaxLights = 8;

struct LitShading {
  bool lit = false;
  float roughness = 0.3f;
};

LitShading litShadingFor(const SceneData::ParticleRenderSettings &render) {
  LitShading out;
  out.lit = render.shape == SceneData::ParticleRenderShape::Sphere &&
            !render.emissive;
  out.roughness = std::clamp(render.roughness, 0.03f, 1.0f);
  return out;
}

struct LitLight {
    Vec3 position;
    Vec3 radiance; // colour * intensity (RT units, / d^2 at the particle)

    bool operator==(const LitLight& other) const {
        return position.x == other.position.x &&
               position.y == other.position.y &&
               position.z == other.position.z &&
               radiance.x == other.radiance.x &&
               radiance.y == other.radiance.y &&
               radiance.z == other.radiance.z;
    }
};
struct LitLights {
    std::vector<LitLight> lights;
    float reference_e = 1.0f;

    bool operator==(const LitLights& other) const {
        return reference_e == other.reference_e && lights == other.lights;
    }
};

float luminance(const Vec3 &c) {
  return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

// Point and spot lights as points (spot cones are ignored: a drop is small and
// rarely outside a room light's cone); a directional light as a point 1e4 m
// away with its radiance scaled by d^2, so one falloff serves every kind.
// Area lights as a point at their centre. Strongest kLitMaxLights by radiance.
LitLights litLightsFor(const SceneData &scene) {
  LitLights out;
  for (const auto &light : scene.lights) {
    if (!light || !light->visible)
      continue;
    const Vec3 rad = light->color * light->intensity;
    if (!(luminance(rad) > 0.0f))
      continue;
    LitLight l;
    if (light->type() == LightType::Directional) {
      const Vec3 d = -light->direction;
      if (!(d.length() > 1e-6f))
        continue;
      l.position = d.normalize() * 1.0e4f;
      l.radiance = rad * 1.0e8f;
    } else {
      l.position = light->position;
      l.radiance = rad;
    }
    out.lights.push_back(l);
  }
  if (out.lights.empty()) {
    // No light at all: a soft key from above-front (the raster preview's).
    out.lights.push_back(
        {Vec3(0.45f, 0.8f, 0.35f).normalize() * 1.0e4f, Vec3(1.0e8f)});
  }
  std::sort(out.lights.begin(), out.lights.end(),
            [](const LitLight &a, const LitLight &b) {
              const bool fa = a.position.length() > 1.0e3f,
                         fb = b.position.length() > 1.0e3f;
              const float ea = luminance(a.radiance) * (fa ? 1.0e-8f : 1.0f);
              const float eb = luminance(b.radiance) * (fb ? 1.0e-8f : 1.0f);
              return ea > eb;
            });
  if (out.lights.size() > static_cast<std::size_t>(kLitMaxLights)) {
    out.lights.resize(kLitMaxLights);
  }
  // Reference point: 1.5 m under the centroid of the near lights (room
  // lights hang at the ceiling; drops fly below them).
  Vec3 centroid(0.0f);
  int near_count = 0;
  for (const auto &l : out.lights) {
    if (l.position.length() < 1.0e3f) {
      centroid = centroid + l.position;
      ++near_count;
    }
  }
  centroid = near_count > 0
                 ? centroid * (1.0f / near_count) - Vec3(0.0f, 1.5f, 0.0f)
                 : Vec3(0.0f, 1.0f, 0.0f);
  float e = 0.0f;
  for (const auto &l : out.lights) {
    const Vec3 d = l.position - centroid;
    e += luminance(l.radiance) /
         std::max(d.x * d.x + d.y * d.y + d.z * d.z, 0.04f);
  }
  out.reference_e = e > 0.0f ? e : 1.0f;
  return out;
}

float flatSpriteStyle(std::string_view system,
                      const RayTrophiSim::ParticleAppearanceProfile &profile) {
  if (system == "BalloonConfetti" && profile.name.starts_with("Confetti ")) {
    return 3.0f;
  }
  if (profile.blend != RayTrophiSim::ParticleAppearanceBlend::Additive) {
    return 0.0f;
  }
  const std::string_view name = profile.name;
  if (system == "BalloonSparkles") {
    if (name.starts_with("Sparkle ")) {
      return 2.0f;
    }
    if (name.starts_with("Drop Spark ") || name.starts_with("Confetti Mote ")) {
      return 1.0f;
    }
  }
  if (system == "BalloonSparkleTrails" &&
      (name.starts_with("Sparkle Quench ") ||
       name.starts_with("Sparkle Glitter "))) {
    return 1.0f;
  }
  return 0.0f;
}

void markParticleRow(std::vector<float> &lut, uint32_t row,
                     const LitShading &shading, const LitLights &lights,
                     std::string_view system,
                     const RayTrophiSim::ParticleAppearanceProfile &profile) {
  float *r = lut.data() + static_cast<std::size_t>(row) *
                              RayTrophiSim::kParticleAppearanceLutFloatsPerRow;
  if (!shading.lit) {
    // The roughness slot is unused by flat sprites. Carry the sprite
    // style through the existing vertex output, with no layout changes.
    r[6] = flatSpriteStyle(system, profile);
    return;
  }
  constexpr int kSample = RayTrophiSim::kParticleAppearanceLutFloatsPerSample;
  auto slot = [&](int sample, float z, float w) {
    r[sample * kSample + 6] = z;
    r[sample * kSample + 7] = w;
  };
  const int count = static_cast<int>(lights.lights.size());
  slot(0, shading.roughness, 1.0f);
  slot(1, static_cast<float>(count), lights.reference_e);
  for (int i = 0; i < count; ++i) {
    const LitLight &l = lights.lights[static_cast<std::size_t>(i)];
    slot(2 + 3 * i, l.position.x, l.position.y);
    slot(3 + 3 * i, l.position.z, l.radiance.x);
    slot(4 + 3 * i, l.radiance.y, l.radiance.z);
  }
}

void pushQuad(std::vector<ParticleBillboardVertex>& out, float x, float y,
              float z, float age, uint32_t lut_row, float size_scale) {
    static constexpr float kCorners[6][2] = {
        {-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f},
        {-1.0f, -1.0f}, {1.0f, 1.0f},  {-1.0f, 1.0f},
    };
    const float row = static_cast<float>(lut_row);
    const std::size_t base = out.size();
    out.resize(base + 6);
    for (std::size_t k = 0; k < 6; ++k) {
        ParticleBillboardVertex& v = out[base + k];
        v.center[0] = x;
        v.center[1] = y;
        v.center[2] = z;
        v.corner[0] = kCorners[k][0];
        v.corner[1] = kCorners[k][1];
        v.age = age;
        v.lut_row = row;
        v.size_scale = size_scale;
    }
}

const float* fallbackLutRow() {
    // Row 0: the fallback appearance, for particles whose profile id resolves
    // to nothing. Always present so the LUT binding is never empty.
    static const std::vector<float> kFallbackRow = [] {
        std::vector<float> baked;
        RayTrophiSim::bakeParticleAppearanceLut(
            RayTrophiSim::fallbackParticleAppearance(), baked);
        return baked;
    }();
    return kFallbackRow.data();
}

struct RowInfo {
    uint32_t row = 0;
    bool alpha = false;
};

struct SystemLutKey {
    const void* runtime_ptr = nullptr;
    std::string name;
    uint64_t appearance_revision = 0;
    SceneData::ParticleRenderShape shape = SceneData::ParticleRenderShape::Sphere;
    bool emissive = false;
    float roughness = 0.0f;
    bool pulled = false;
};

struct CachedSystemMeta {
    bool pulled = false;
    uint32_t lookup_offset = 0;
    uint32_t lookup_count = 0;
    bool has_alpha = false;
    // Reusable flat lookup indexed by profile.id (default RowInfo{0, false}).
    std::vector<RowInfo> host_rows;
};

struct FrameSystemProbe {
    std::size_t scene_index = 0;
    bool pulled = false;
    RayTrophiSim::ParticleResidentDrawBuffers device{};
};

struct LutBuildCache {
    bool valid = false;
    std::vector<SystemLutKey> keys;
    LitLights lights;
    std::vector<float> lut;
    std::vector<uint32_t> row_lookup;
    std::vector<CachedSystemMeta> systems;

    const ParticleBillboardUpload* last_out = nullptr;
    const float* last_lut_ptr = nullptr;
    const uint32_t* last_row_lookup_ptr = nullptr;

    std::vector<FrameSystemProbe> frame_probes;
};

bool keyMatches(const SystemLutKey& key,
                const SceneData::ParticleSystemObject& system,
                const RayTrophiSim::ParticleSimulationSystem& runtime,
                bool pulled) {
    return key.runtime_ptr == system.runtime.get() &&
           key.appearance_revision == runtime.appearanceRevision() &&
           key.shape == system.render.shape &&
           key.emissive == system.render.emissive &&
           key.roughness == system.render.roughness &&
           key.pulled == pulled &&
           key.name == system.name;
}

SystemLutKey makeSystemKey(const SceneData::ParticleSystemObject& system,
                           const RayTrophiSim::ParticleSimulationSystem& runtime,
                           bool pulled) {
    SystemLutKey key;
    key.runtime_ptr = system.runtime.get();
    key.name = system.name;
    key.appearance_revision = runtime.appearanceRevision();
    key.shape = system.render.shape;
    key.emissive = system.render.emissive;
    key.roughness = system.render.roughness;
    key.pulled = pulled;
    return key;
}

void rebuildLutCache(const SceneData& scene, const LitLights& lights,
                     LutBuildCache& cache) {
    using RayTrophiSim::ParticleAppearanceBlend;

    cache.keys.clear();
    cache.systems.clear();
    cache.lut.clear();
    cache.lights = lights;
    appendRow(cache.lut, fallbackLutRow());
    cache.row_lookup.assign(1, 0u); // never empty: keeps the binding valid
    cache.keys.reserve(cache.frame_probes.size());
    cache.systems.reserve(cache.frame_probes.size());

    for (const FrameSystemProbe& probe : cache.frame_probes) {
        const auto& system = scene.particle_systems[probe.scene_index];
        const auto& runtime = *system.runtime;
        const LitShading shading = litShadingFor(system.render);
        cache.keys.push_back(makeSystemKey(system, runtime, probe.pulled));

        CachedSystemMeta meta;
        meta.pulled = probe.pulled;
        const auto& profiles = runtime.appearanceProfiles();
        uint32_t max_id = 0;
        for (const auto& profile : profiles) {
            max_id = std::max(max_id, profile.id);
        }

        if (probe.pulled) {
            meta.lookup_offset = static_cast<uint32_t>(cache.row_lookup.size());
            meta.lookup_count = max_id + 1u;
            cache.row_lookup.resize(cache.row_lookup.size() + meta.lookup_count, 0u);
            for (const auto& profile : profiles) {
                const bool alpha = profile.blend == ParticleAppearanceBlend::Alpha;
                meta.has_alpha = meta.has_alpha || alpha;
                const uint32_t row = rowCount(cache.lut);
                cache.row_lookup[meta.lookup_offset + profile.id] =
                    row | (alpha ? 0x80000000u : 0u);
                appendRow(cache.lut, runtime.appearanceLutRow(profile.id));
                markParticleRow(cache.lut, row, shading, lights, system.name, profile);
            }
        } else {
            // ★ Rows BEFORE the alive check (2026-11-24): a dead system keeps its
            // LUT rows so starting/ending effect systems never triggers a drain.
            meta.host_rows.assign(static_cast<std::size_t>(max_id) + 1u, RowInfo{});
            for (const auto& profile : profiles) {
                RowInfo info;
                info.row = rowCount(cache.lut);
                info.alpha = profile.blend == ParticleAppearanceBlend::Alpha;
                appendRow(cache.lut, runtime.appearanceLutRow(profile.id));
                markParticleRow(cache.lut, info.row, shading, lights, system.name,
                                profile);
                meta.host_rows[profile.id] = info;
            }
        }
        cache.systems.push_back(std::move(meta));
    }
    cache.valid = true;
}

void appendParticleSystems(const SceneData& scene, void* viewport_device,
                           ParticleBillboardUpload& out, std::size_t& drawn) {
    static LutBuildCache cache;
    const LitLights lights = litLightsFor(scene);

    cache.frame_probes.clear();
    bool keys_match = cache.valid && (cache.lights == lights);
    std::size_t key_idx = 0;
    for (std::size_t idx = 0; idx < scene.particle_systems.size(); ++idx) {
        const auto& system = scene.particle_systems[idx];
        if (!system.visible || !system.runtime || system.render.emitter_only) {
            continue;
        }
        const auto& runtime = *system.runtime;
        FrameSystemProbe probe;
        probe.scene_index = idx;
        probe.pulled = viewport_device &&
                       runtime.residentDrawBuffers(probe.device) &&
                       probe.device.device == viewport_device &&
                       probe.device.capacity != 0;
        if (keys_match) {
            if (key_idx >= cache.keys.size() ||
                !keyMatches(cache.keys[key_idx], system, runtime, probe.pulled)) {
                keys_match = false;
            }
        }
        ++key_idx;
        cache.frame_probes.push_back(probe);
    }
    if (key_idx != cache.keys.size()) {
        keys_match = false;
    }

    if (!keys_match) {
        rebuildLutCache(scene, lights, cache);
        out.lut = cache.lut;
        out.row_lookup = cache.row_lookup;
        cache.last_out = &out;
        cache.last_lut_ptr = out.lut.data();
        cache.last_row_lookup_ptr = out.row_lookup.data();
    } else {
        const bool out_in_sync =
            (&out == cache.last_out) &&
            out.lut.size() == cache.lut.size() &&
            out.row_lookup.size() == cache.row_lookup.size() &&
            out.lut.data() == cache.last_lut_ptr &&
            out.row_lookup.data() == cache.last_row_lookup_ptr;
        if (!out_in_sync) {
            out.lut = cache.lut;
            out.row_lookup = cache.row_lookup;
            cache.last_out = &out;
            cache.last_lut_ptr = out.lut.data();
            cache.last_row_lookup_ptr = out.row_lookup.data();
        }
    }

    for (std::size_t idx = 0; idx < cache.frame_probes.size(); ++idx) {
        if (drawn >= kMaxBillboards) {
            break;
        }
        const FrameSystemProbe& probe = cache.frame_probes[idx];
        const CachedSystemMeta& meta = cache.systems[idx];
        auto& runtime = *scene.particle_systems[probe.scene_index].runtime;

        if (meta.pulled) {
            ParticlePulledDraw draw;
            for (uint32_t i = 0; i < kPulledStreamCount; ++i) {
                draw.buffers[i] = static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(probe.device.buffers[i]));
            }
            // Births are host-owned. A zero host count means there is nothing to
            // draw, while the stable descriptor slot and LUT rows still serve respawn.
            const bool has_live_particles = runtime.aliveCount() != 0;
            draw.slot_capacity = probe.device.capacity;
            draw.particle_count = has_live_particles ? probe.device.capacity : 0;
            draw.state_version = has_live_particles ? probe.device.state_version : 0;
            draw.lookup_offset = meta.lookup_offset;
            draw.lookup_count = meta.lookup_count;
            draw.has_alpha = meta.has_alpha;
            out.pulled.push_back(draw);
            continue;
        }

        // Not pullable: either the host already holds the state, or the
        // simulation runs on ANOTHER VkDevice and its buffers cannot be bound
        // here. The second is a counted snapshot (particle.stats), the same
        // consequence live gas has on a foreign device.
        if (runtime.kinematicResidency() !=
            RayTrophiSim::ParticleKinematicResidency::Host) {
            runtime.syncHostState("foreign_device");
        }
        std::size_t remaining = runtime.aliveCount();
        if (remaining == 0) {
            continue;
        }

        const auto& buf = runtime.buffers();
        const std::size_t cap = buf.alive.size();
        bool have_cached_row = false;
        uint32_t cached_profile_id = 0;
        RowInfo cached_info;
        for (std::size_t i = 0; i < cap && drawn < kMaxBillboards; ++i) {
            if (buf.alive[i] == 0u) {
                continue;
            }
            const bool last_alive = (--remaining == 0);
            const float x = buf.position_x[i];
            const float y = buf.position_y[i];
            const float z = buf.position_z[i];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                if (last_alive) {
                    break;
                }
                continue;
            }

            const float life = buf.lifetime_seconds[i];
            const float age = life > 1e-6f
                                  ? std::clamp(buf.age_seconds[i] / life, 0.0f, 1.0f)
                                  : 0.0f;
            const uint32_t id =
                i < buf.appearance_profile.size() ? buf.appearance_profile[i] : 0u;
            const float scale = i < buf.size_scale.size() ? buf.size_scale[i] : 1.0f;

            // Unknown / zero id -> row 0, the fallback (additive).
            if (!have_cached_row || id != cached_profile_id) {
                cached_profile_id = id;
                cached_info = id < meta.host_rows.size()
                                  ? meta.host_rows[id]
                                  : RowInfo{};
                have_cached_row = true;
            }
            pushQuad(cached_info.alpha ? out.alpha : out.additive, x, y, z, age,
                     cached_info.row, scale);
            ++drawn;
            if (last_alive) {
                break;
            }
        }
    }
}

void appendFluidSphereProxies(const SceneData &scene, void *viewport_device,
                              ParticleBillboardUpload &out,
                              std::unordered_set<int> &pulled_groups) {
  if (!viewport_device)
    return;
  const RayTrophiSim::SimulationComputeContext &compute =
      scene.simulation_world.compute();
  for (const auto &system : scene.particle_systems) {
    if (!system.visible || !system.enabled || !system.runtime)
      continue;
    const auto &domains = system.runtime->gridDomains();
    const auto &states = system.runtime->gridDomainStates();
    const std::size_t count = std::min(domains.size(), states.size());
    for (std::size_t d = 0; d < count; ++d) {
      const auto &desc = domains[d];
      const auto &state = states[d];
      // Grain domains stay on the CPU bridge: it sizes every carrier by
      // its own owner (granular -> 2 * grain radius, liquid -> voxel
      // radius), while this proxy has one radius per domain, and the
      // resident position stream holds only the liquid lane's subset.
      if (!state.valid || desc.fluid_visual_samples.enabled ||
          desc.fluid_params.grain.enabled ||
          desc.fluid_grain_mesh_near_enabled ||
          RayTrophiSim::Fluid::hasSourceSplatMeshes(*system.runtime, d) ||
          desc.fluid_render_mode !=
              RayTrophiSim::Fluid::FluidRenderMode::Particles ||
          desc.fluid_particle_geometry_mode != 0 ||
          d >= system.domain_particle_render_group_ids.size()) {
        continue;
      }
      const int group_id = system.domain_particle_render_group_ids[d];
      if (group_id < 0)
        continue;

      const auto plan = RayTrophiSim::Fluid::resolveFluidViews(
          desc, RayTrophiSim::Fluid::cachedDistinctViewKeys(state.particles,
                                                            state.version));
      // A single packed position stream can be pulled only when every
      // live parcel uses the splat representation. Mixed substance/label
      // routes retain the exact CPU-filtered bridge.
      if (!plan.allLiveIn(RayTrophiSim::Fluid::FluidView::Splat) ||
          desc.fluid_params.pore_exchange.wet_appearance_enabled) {
        continue;
      }

      RayTrophiSim::FluidResidentPositionBuffer resident;
      if (!system.runtime->fluidResidentPositionBuffer(d, compute, resident) ||
          resident.device != viewport_device || resident.particle_count == 0) {
        continue;
      }

      const uint32_t requested_children =
          desc.fluid_params.granular_enabled &&
                  desc.fluid_granular_physical_carriers
              ? 1u
              : static_cast<uint32_t>(std::clamp(
                    desc.fluid_particle_visual_children, 1,
                    static_cast<int>(
                        RayTrophiSim::Fluid::kFluidRenderProxyMaxChildren)));
      auto layout = RayTrophiSim::Fluid::resolveFluidRenderProxyLayout(
          true, requested_children);
      layout = RayTrophiSim::Fluid::resolveFluidRenderProxyLayout(
          true, RayTrophiSim::Fluid::limitFluidRenderProxyChildren(
                    layout.children_per_parent,
                    RayTrophiSim::Fluid::fluidRenderProxyCarrierCapacity(
                        desc.fluid_max_particles,
                        desc.fluid_foam_params.enabled
                            ? desc.fluid_foam_params.max_foam
                            : 0u),
                    desc.fluid_particle_visual_budget));
      const auto visual = RayTrophiSim::Fluid::resolveFluidParticleVisualRadius(
          desc.fluid_params.granular_enabled, true,
          desc.fluid_params.particles_per_cell,
          desc.fluid_particle_radius_factor,
          desc.fluid_particle_size_multiplier);
      const float parent_radius =
          std::max(1e-4f, RayTrophiSim::Fluid::liquidGrid(state).voxel_size *
                              visual.effective_voxels);

      FluidSphereProxyDraw draw;
      draw.position_buffer = static_cast<uint64_t>(
          reinterpret_cast<uintptr_t>(resident.positions));
      draw.parent_count = resident.particle_count;
      draw.children_per_parent = layout.children_per_parent;
      draw.child_radius = parent_radius * layout.child_radius_scale;
      draw.spread_radius = std::max(
          0.0f,
          RayTrophiSim::Fluid::liquidGrid(state).voxel_size *
                  RayTrophiSim::Fluid::kFluidRenderProxyFillSupportVoxels -
              draw.child_radius);
      draw.size_variation = desc.fluid_particle_visual_size_variation;
      draw.state_version = resident.state_version;
      out.fluid_sphere_proxies.push_back(draw);
      pulled_groups.insert(group_id);
    }
  }
}

bool appendGroupSphereProxies(const InstanceGroup &group,
                              ParticleBillboardUpload &out) {
  const std::size_t start = out.spheres.size();
  const uint32_t children =
      group.point_sphere_mode
          ? std::max<uint32_t>(1u, group.point_sphere_visual_children)
          : 1u;
  const uint64_t additional =
      static_cast<uint64_t>(group.instances.size()) * children;
  if (additional > out.spheres.max_size() - out.spheres.size()) {
    return false;
  }
  try {
    out.spheres.reserve(out.spheres.size() +
                        static_cast<std::size_t>(additional));
    for (std::size_t parent = 0; parent < group.instances.size(); ++parent) {
      const auto &instance = group.instances[parent];
      const auto &position = instance.position;
      const float parent_radius = instance.scale.x * 0.5f;
      if (!(parent_radius > 0.0f) || !std::isfinite(parent_radius) ||
          !std::isfinite(position.x) || !std::isfinite(position.y) ||
          !std::isfinite(position.z)) {
        continue;
      }

      Vec3 candidates[RayTrophiSim::Fluid::kFluidRenderProxyNeighborCandidates];
      uint32_t candidate_count = 0;
      constexpr uint32_t kNeighborSteps =
          RayTrophiSim::Fluid::kFluidRenderProxyNeighborCandidates / 2u;
      for (uint32_t slot = 0; slot < kNeighborSteps; ++slot) {
        const uint32_t step = 1u << slot;
        for (uint32_t side = 0; side < 2u; ++side) {
          if ((side == 0u && parent < step) ||
              (side == 1u && parent + step >= group.instances.size())) {
            continue;
          }
          const std::size_t candidate_index =
              side == 0u ? parent - step : parent + step;
          const auto &candidate = group.instances[candidate_index];
          if (!(candidate.scale.x > 0.0f) ||
              !std::isfinite(candidate.position.x) ||
              !std::isfinite(candidate.position.y) ||
              !std::isfinite(candidate.position.z)) {
            continue;
          }
          candidates[candidate_count++] = candidate.position;
        }
      }

      for (uint32_t child = 0; child < children; ++child) {
        const Vec3 offset = RayTrophiSim::Fluid::fluidRenderProxyNeighborOffset(
            static_cast<uint32_t>(parent), child, children, position,
            candidates, candidate_count,
            group.point_sphere_visual_spread_radius *
                RayTrophiSim::Fluid::kFluidRenderProxyNeighborSupportScale,
            group.point_sphere_visual_spread_radius);
        const Vec3 center =
            RayTrophiSim::Fluid::fluidRenderProxyIsolated(instance.scale)
                ? position
                : position + offset;
        const float radius =
            parent_radius *
            RayTrophiSim::Fluid::fluidRenderProxyChildRadiusScale(
                static_cast<uint32_t>(parent), child, children,
                group.point_sphere_visual_size_variation) *
            RayTrophiSim::Fluid::fluidRenderProxyIsolatedRadiusFactor(
                child, children,
                RayTrophiSim::Fluid::fluidRenderProxyIsolated(instance.scale));
        out.spheres.push_back({{center.x, center.y, center.z, radius}});
      }
    }
  } catch (const std::bad_alloc &) {
    out.spheres.resize(start);
    return false;
  }
  return true;
}

} // namespace

void build(const SceneData &scene, void *viewport_device,
           ParticleBillboardUpload &out) {
  out.spheres.clear();
  out.fluid_sphere_proxies.clear();
  const bool spherePath = viewport_device && g_sphere_impostor_ready.load() &&
                          g_solid_viewport_active &&
                          !g_material_preview_viewport_active;
  std::unordered_set<int> pulled_fluid_groups;
  if (spherePath && g_fluid_sphere_proxy_ready.load()) {
    appendFluidSphereProxies(scene, viewport_device, out, pulled_fluid_groups);
  }
  for (auto &group : InstanceManager::getInstance().getGroups()) {
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
    if (pulled_fluid_groups.find(group.id) != pulled_fluid_groups.end()) {
      continue;
    }
    appendGroupSphereProxies(group, out);
  }
  out.additive.clear();
  out.alpha.clear();
  out.pulled.clear();

  std::size_t drawn = 0;
  appendParticleSystems(scene, viewport_device, out, drawn);
  // Fluid domains in Particles mode are NOT drawn here: the render bridge
  // already draws them as sphere instances in raster and RT. A second copy
  // as billboards read the live positions while the instances lagged a
  // step, so during playback a blue disc poked out in front of every sphere.
}

} // namespace ParticleBillboardBuilder
