#pragma once

// Particle appearance profiles (particle roadmap Phase 1.5 Batch A).
//
// A profile says how a particle LOOKS over its life: colour, opacity, size and
// emission as functions of normalized age (0 = birth, 1 = death), plus how it
// blends. Emitters reference a profile by id; particles carry the id they were
// born with. Curves are baked into a fixed-size LUT, and every consumer — the
// raster billboard shader, the RT instance bridge, the debug overlay and the
// particle -> gas deposit weight — reads that same LUT, so no two paths can
// disagree about what a particle looks like at a given age.
//
// Before this, each particle carried start/end endpoints captured at spawn and
// the CPU step lerped colour/size/opacity into the SoA every frame. The legacy
// fields survive only as loader input (see migrateLegacyEmitterAppearance).

#include "Vec3.h"
#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {

// How the raster billboard pass composites a profile. This is the profile's
// first consumer: a campfire needs additive flame and alpha smoke in ONE
// system, which the old per-system blend mode could not express.
enum class ParticleAppearanceBlend : uint8_t {
    Additive = 0,
    Alpha = 1,
};

struct ParticleCurveKey {
    float t = 0.0f;      // normalized age, 0..1
    float value = 0.0f;
};

struct ParticleColorStop {
    float t = 0.0f;      // normalized age, 0..1
    Vec3 color = Vec3(1.0f, 1.0f, 1.0f);
};

struct ParticleAppearanceProfile {
    uint32_t id = 0;     // per system, monotonic, never reused; 0 = none
    std::string name = "Appearance";
    ParticleAppearanceBlend blend = ParticleAppearanceBlend::Additive;
    // Piecewise linear over normalized age. Keys are kept sorted by t; values
    // outside the first/last key clamp to that key. An empty curve means the
    // documented neutral value (white, opacity 1, size 0.05 m, emission 1).
    std::vector<ParticleColorStop> color_ramp;
    std::vector<ParticleCurveKey> opacity_curve;   // 0..1
    std::vector<ParticleCurveKey> size_curve;      // full billboard width, metres
    std::vector<ParticleCurveKey> emission_curve;  // multiplier on colour, >= 0
};

// One evaluated point of a profile.
struct ParticleAppearanceSample {
    Vec3 color = Vec3(1.0f, 1.0f, 1.0f);
    float opacity = 1.0f;
    float size = 0.05f;
    float emission = 1.0f;
};

// LUT layout, shared with shaders/particle_viewport.vert (keep in sync):
// a row is kParticleAppearanceLutSamples samples; a sample is two vec4,
//   [0] = (r, g, b, opacity)   [1] = (size, emission, 0, 0)
// sample s sits at normalized age s / (samples - 1); readers interpolate
// linearly between neighbouring samples.
inline constexpr int kParticleAppearanceLutSamples = 64;
inline constexpr int kParticleAppearanceLutFloatsPerSample = 8;
inline constexpr int kParticleAppearanceLutFloatsPerRow =
    kParticleAppearanceLutSamples * kParticleAppearanceLutFloatsPerSample;
inline constexpr std::size_t kParticleAppearanceMaxKeys = 16;

// Exact evaluation of the authored curves (used only to bake).
ParticleAppearanceSample evaluateParticleAppearanceCurves(
    const ParticleAppearanceProfile& profile, float t);

// Bakes `profile` into one LUT row (kParticleAppearanceLutFloatsPerRow floats).
void bakeParticleAppearanceLut(const ParticleAppearanceProfile& profile,
                               std::vector<float>& row);

// Samples a baked row exactly the way the shader does. `row` must hold
// kParticleAppearanceLutFloatsPerRow floats.
ParticleAppearanceSample sampleParticleAppearanceLut(const float* row, float t);

// Sorts keys, then validates ranges. Returns an empty string when valid,
// otherwise the reason (surfaced unchanged by IPC and Python).
std::string normalizeParticleAppearanceProfile(ParticleAppearanceProfile& profile);

// Appearance of a particle whose profile id resolves to nothing (scripted
// spawns with id 0, or a profile removed while particles were alive): white,
// opacity 1 -> 0, unit size curve (so the particle's own size_scale is its
// width in metres), emission 1.
const ParticleAppearanceProfile& fallbackParticleAppearance();

// Two-key profile equivalent to the legacy start/end emitter fields. Used by
// the load-time migration, the presets and built-in spawners.
ParticleAppearanceProfile makeTwoKeyParticleAppearance(
    const std::string& name, ParticleAppearanceBlend blend,
    float start_size, float end_size,
    float start_opacity, float end_opacity,
    const Vec3& start_color, const Vec3& end_color,
    float emission = 1.0f);

// Blend names used by IPC / Python / the panel: "additive" | "alpha".
const char* particleAppearanceBlendName(ParticleAppearanceBlend blend);
bool parseParticleAppearanceBlend(const std::string& name, ParticleAppearanceBlend& out);

// ── Serialization ───────────────────────────────────────────────────────────
nlohmann::json serializeParticleAppearanceProfile(const ParticleAppearanceProfile& profile);
// Reads a profile written by serializeParticleAppearanceProfile. Returns false
// when the object has no id.
bool deserializeParticleAppearanceProfile(const nlohmann::json& j,
                                          ParticleAppearanceProfile& out);

class ParticleSimulationSystem;
struct ParticleEmitterDesc;

// System-level (de)serialization, used by every project/scene writer so the
// formats cannot drift. Keys: "appearance_profiles", "next_appearance_profile_id".
void serializeParticleAppearanceProfiles(const ParticleSimulationSystem& system,
                                         nlohmann::json& system_json);
// Must run BEFORE the system's emitters are added, so their ids resolve.
void deserializeParticleAppearanceProfiles(const nlohmann::json& system_json,
                                           ParticleSimulationSystem& system);
// Emitter side: writes "appearance_profile_id".
void serializeEmitterAppearance(const ParticleEmitterDesc& emitter, nlohmann::json& emitter_json);
// Reads "appearance_profile_id". An emitter saved before profiles existed has
// no such key: its legacy start/end fields become a two-key profile on
// `system` (load-time migration; the legacy keys are read nowhere else and
// never written again). Idempotent: a saved migrated project has the key.
void readEmitterAppearance(const nlohmann::json& emitter_json, int legacy_blend,
                           ParticleSimulationSystem& system, ParticleEmitterDesc& emitter);

// Builds the two-key profile an old emitter JSON describes. Reads ONLY the
// legacy keys start_size/end_size/start_opacity/end_opacity/start_color/
// end_color (missing keys take the old defaults); `legacy_blend` is the old
// per-system blend_mode (0 = additive, 1 = alpha).
ParticleAppearanceProfile migrateLegacyEmitterAppearance(
    const nlohmann::json& emitter_json, int legacy_blend,
    const std::string& emitter_name);

} // namespace RayTrophiSim
