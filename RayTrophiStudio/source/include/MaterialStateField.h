/*
* ─────────────────────────────────────────────────────────────────────────────
* File:          MaterialStateField.h
* Project:       RayTrophi Studio
* Description:   Material State Field (MSF) — per-object, persistent thermal /
*                combustion surface state.
*
* See docs/NODE_SIMULATION_ARCHITECTURE_PLAN.md (Bölüm A) for the design.
*
* WHY THIS EXISTS
* ---------------
* The existing pyrolysis model (shaders/sim_gas_collider_source.comp) keeps its
* surface state in `surface_state[]`, indexed by the GAS DOMAIN'S VOXEL. That is
* right for a gas SOURCE but wrong for material damage:
*   - state dies when the object leaves the domain,
*   - it is voxel-resolution, so it cannot drive shading or geometry,
*   - and the shader deliberately writes -1 to the remaining-material plane when
*     contact is lost, i.e. it ERASES char so an animated collider restarts fresh.
*
* MSF moves that state onto the object and keeps it. Phase 1 only INTEGRATES the
* state (gather); it does not feed the gas back. The voxel path keeps running
* untouched, so the two can be compared side by side before anything is removed.
* ─────────────────────────────────────────────────────────────────────────────
*/
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "SimulationCompute.h"
#include "SurfaceMeshCache.h"
#include "SurfaceField.h"

namespace RayTrophiSim {

// ─────────────────────────────────────────────────────────────────────────────
// UNITS (Phase 2)
//
// Substances are authored in KELVIN, always. The solver's temperature field is
// normalized (0 = ambient, `max_temperature` = the hottest the domain allows),
// and three incompatible conventions already exist in this codebase:
//   - World.h AtmosphereParams.temperature — Celsius, but a RENDER parameter
//     (atmospheric scattering). Deliberately NOT used as a thermal authority:
//     making it one means editing the sky sets the scene on fire.
//   - GasSimulator.h — Kelvin (293/6000), but that path is dead.
//   - RtApi.h fire_max_temperature = 10.0f — normalized, and the live path.
//
// So Kelvin is the authoring unit and the conversion happens in EXACTLY one
// place: MaterialTemperatureScale below. Nothing downstream of the push-constant
// fill ever sees Kelvin.
//
// Phase 4 promoted the AUTHORING of these two numbers to WorldThermalState
// below; this struct stays as the derived conversion object, so there is still
// exactly one Kelvin -> normalized code path.
// ─────────────────────────────────────────────────────────────────────────────
struct MaterialTemperatureScale {
    float ambient_kelvin = 293.0f;    // normalized 0 maps to this
    // Kelvin per normalized unit. The default is chosen so the legacy authored
    // defaults keep their meaning: wood's real ignition point (573 K) lands on
    // 0.8 normalized, which is exactly ParticleColliderDesc's old default. Iron's
    // melting point (1811 K) lands on 4.34, comfortably inside a default domain's
    // max_temperature of 10. Existing scenes therefore do not change behaviour.
    float kelvin_per_unit = 350.0f;

    float toNormalized(float kelvin) const {
        if (!(kelvin_per_unit > 0.0f)) return 0.0f;
        return (kelvin - ambient_kelvin) / kelvin_per_unit;
    }
    float toKelvin(float normalized) const {
        return ambient_kelvin + normalized * kelvin_per_unit;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// WORLD THERMAL STATE (Phase 4)
//
// The world carries BOUNDARY CONDITIONS, not simulation state. Its whole point
// is that an object which is inside no domain at all still has a defined
// temperature: a burning log carried out of the smoke domain must cool toward
// the room by radiation/convection, not freeze mid-burn and not be reset.
//
// Layering (see the plan doc, A.6):
//     World  ->  Domain override  ->  Thermal field  ->  object MSF
//
// ★ `kelvin_per_unit` deliberately has NO per-domain override. It is the
// normalized<->Kelvin CALIBRATION, and the mask quantizes surface temperature in
// absolute Kelvin (Phase 3d). If two domains disagreed about it, the same object
// would report a different temperature — and glow differently — depending on
// which box it happened to be standing in. Ambient and oxygen are boundary
// conditions and DO override per domain; the unit mapping is not one.
//
// ★ SCOPE LIMIT, stated rather than hidden: this is stored on the particle
// simulation runtime, which is where the substance/scale authority already
// lives. A scene with two particle systems can therefore author two "worlds".
// Promoting it to SceneData is a follow-up; the node layer (Phase 9) is where a
// single World node makes that unambiguous.
// ─────────────────────────────────────────────────────────────────────────────
struct WorldThermalState {
    // Everything the room is at, unless a domain says otherwise. Normalized 0 is
    // DEFINED as this temperature, which is why an untouched scene relaxes to 0.
    float ambient_kelvin = 293.0f;
    // Kelvin per normalized solver unit. Calibration — see the note above.
    float kelvin_per_unit = 350.0f;
    // Scales every substance's passive cooling toward ambient. 1 = the substance
    // constant as authored; higher = a draughty room, 0 = a perfect thermos.
    float convection_coefficient = 1.0f;
    // 0..1. Scales pyrolysis burn rate; 0 smothers combustion entirely. This is
    // the cheap version of "seal the box and the fire goes out" — the real oxygen
    // transport model is not in scope.
    float oxygen_availability = 1.0f;

    MaterialTemperatureScale scale() const {
        MaterialTemperatureScale s;
        s.ambient_kelvin = ambient_kelvin;
        s.kelvin_per_unit = kelvin_per_unit;
        return s;
    }
};

// Physical description of what an object is made of. All temperatures in Kelvin.
//
// Phase 2 wires the thermal/combustion half (used by sim_msf_gather). The melt /
// ash / optical fields are authored now but consumed later — Phase 3 reads
// char_color and molten emission, Phase 6 reads melt_kelvin and melt_viscosity,
// Phase 7 reads ash_yield. They live here so the presets are written once with
// real values instead of being retrofitted per phase.
struct SubstanceProfile {
    std::string name;

    // ── Thermal ──────────────────────────────────────────────────────────────
    float density            = 1000.0f;  // kg/m^3
    float specific_heat      = 1000.0f;  // J/(kg·K)
    float conductivity       = 1.0f;     // W/(m·K)
    float emissivity         = 0.9f;     // 0..1

    // Thermal inertia: the surface follows nearby gas temperature with
    // 1-exp(-response*dt), so a one-frame spike cannot ignite it. Matches the
    // hard-coded 3.0 / 0.08 in sim_gas_collider_source.comp.
    float thermal_response   = 3.0f;     // 1/s
    float cooling_rate       = 0.08f;    // normalized units/s, passive loss

    // ── Moisture (Phase 5) ───────────────────────────────────────────────────
    // How readily the surface takes on water from a liquid domain, per second
    // toward saturation. 0 = water runs straight off (metal, stone, glass);
    // high = a sponge (cloth, paper). This is what makes "hose down the wooden
    // crate" and "hose down the iron beam" behave differently.
    float absorbency         = 0.0f;
    // Passive drying toward zero, per second, at the surface's own temperature.
    // Scaled up sharply once the surface is hot — a wet plank next to a fire
    // dries in seconds, one in a cold room takes far longer.
    float dry_rate           = 0.03f;

    // ── Combustion ───────────────────────────────────────────────────────────
    bool  combustible        = false;
    float ignition_kelvin    = 573.0f;   // pyrolysis onset
    float fuel_capacity      = 4.0f;     // combustible mass per surface element
    float burn_rate          = 0.75f;    // mass consumed per second
    float char_rate          = 1.0f;     // char produced per unit burned mass
    float ash_yield          = 0.15f;    // fraction of mass left as ash (Phase 7)
    // What the released vapour does to the gas. The old voxel path hard-coded
    // smoke at 0.30 and heat at the ignition threshold for every material; making
    // them per-substance is the point of having substances at all — smouldering
    // damp wood should smoke heavily and heat little, paper the opposite.
    float smoke_yield        = 0.30f;    // gas density per unit burned mass
    float heat_release       = 0.8f;     // gas temperature (normalized) per unit mass
    float flame_level        = 0.25f;    // visible flame level while burning, 0..1

    // ── Phase change ─────────────────────────────────────────────────────────
    bool  meltable           = false;
    float melt_kelvin        = 1811.0f;
    float boiling_kelvin     = 3134.0f;
    float latent_heat_fusion = 2.7e5f;   // J/kg
    float melt_viscosity     = 0.5f;     // 0 = runs like water, 1 = barely flows

    // ── Optical (Phase 3) ────────────────────────────────────────────────────
    float char_color[3]      = {0.05f, 0.04f, 0.035f};
    float molten_emission    = 1.0f;     // blackbody emission multiplier
};

// ─────────────────────────────────────────────────────────────────────────────
// Render-bridge telemetry.
//
// The path from "MSF has char" to "the surface looks burnt" runs through four
// separate systems, and a break in any of them looks identical on screen:
// nothing happens. These counters make the break point observable instead of a
// guess — read the first one that is zero.
// ─────────────────────────────────────────────────────────────────────────────
struct MaterialStateFieldBridgeStats {
    uint32_t fields_seen = 0;        // MSF fields the bridge was handed
    uint32_t masks_ready = 0;        // ...whose host mask is populated (readback done)
    uint32_t textures_live = 0;      // ...uploaded to a VkImage
    uint32_t indices_resolved = 0;   // ...with a valid bindless index
    uint32_t instances_bound = 0;    // TLAS instances that matched a field BY NAME
    uint32_t instances_seen = 0;     // TLAS instances the resolver was asked about
    // How many times a non-empty instance list was walked this frame. 0 here with
    // fields > 0 means the backend never reached the resolve pass at all, which is
    // a different fault from "walked it but no name matched".
    uint32_t instance_lists_seen = 0;
    // ★ Times the pass ran but the backend had NO TLAS instances at all. Without
    // this row, "the bridge was never called" and "it was called against an
    // unbuilt RT scene" are the same 0 — the exact ambiguity that sent the Phase
    // 3b diagnosis chasing a name mismatch for an afternoon, one level down.
    uint32_t instance_lists_empty = 0;
    // Instances whose source object could not be named at all (null/unknown
    // source pointer). Distinguishes "no name to match" from "name did not match".
    uint32_t instances_unnamed = 0;
    uint32_t last_index = 0;
    // Name of an instance the resolver had no field for. Almost always the real
    // culprit: MSF is keyed by the collider's source_name while instances are
    // keyed by node name, and the two can legitimately differ.
    char unmatched_example[64] = {0};
};
MaterialStateFieldBridgeStats& materialStateFieldBridgeStats();

// Built-in substance library. Lookup is by name; an unknown name yields the
// "Custom" profile so a project authored against a newer build degrades to
// editable free values instead of failing to load.
const std::vector<SubstanceProfile>& substanceLibrary();
const SubstanceProfile& findSubstance(const std::string& name);

// Per-object deviation from the library profile.
//
// A substance library gives physically-right defaults, but an artist legitimately
// wants "this paper is thinner, it catches sooner" without editing what Paper
// means everywhere. The global Kelvin scale cannot express that — it is a domain
// calibration, so nudging it moves EVERY material at once.
//
// This is a delta on the one conversion path, not a second path: overrides are
// authored in Kelvin like everything else and are applied inside fromProfile().
struct SubstanceOverride {
    bool  override_ignition = false;
    float ignition_kelvin   = 573.0f;   // used only when override_ignition
    float burn_rate_scale     = 1.0f;   // 1 = as the substance says
    float fuel_capacity_scale = 1.0f;
    bool isDefault() const {
        return !override_ignition && burn_rate_scale == 1.0f &&
               fuel_capacity_scale == 1.0f;
    }
};

// What sim_msf_gather actually needs, in the solver's normalized units. Derived
// from a profile + overrides + scale; this is the single conversion point.
struct MaterialSubstance {
    float ignition_temperature = 0.8f;
    float fuel_capacity        = 4.0f;
    float burn_rate            = 0.75f;
    float thermal_response     = 3.0f;
    float cooling_rate         = 0.08f;
    float char_rate            = 1.0f;
    // ── Phase 5 ──────────────────────────────────────────────────────────────
    float absorbency           = 0.0f;
    float dry_rate             = 0.03f;
    // Water's boiling point in the solver's normalized units. Derived here, in
    // the one conversion point, rather than hard-coded in a shader: it is what
    // pins a wet surface's temperature until the water is gone, and a shader
    // constant would silently mean a different temperature at every calibration.
    float boil_normalized      = 0.23f;
    // ── Phase 6b ─────────────────────────────────────────────────────────────
    // The substance's melting point in solver units. A NON-meltable substance
    // gets a sentinel far above any reachable temperature rather than a separate
    // bool: one comparison in the shader instead of a branch, and no way for a
    // "meltable" flag and a melt point to disagree.
    float melt_normalized      = 1.0e9f;
    // How fast the solid->liquid fraction advances per unit of temperature
    // overshoot per second. Derived from latent_heat_fusion — a high latent heat
    // means more energy has to go in before the same fraction melts.
    float melt_rate            = 0.0f;
    // Total transferable surface mass per square metre. Combustibles use their
    // authored fuel capacity; meltable non-combustibles use a documented 1 mm
    // surface-shell mass (density * 0.001 m).
    float mass_capacity        = 0.0f;

    static MaterialSubstance fromProfile(const SubstanceProfile& profile,
                                         const MaterialTemperatureScale& scale,
                                         const SubstanceOverride& overrides = {});
};

// Water boils at 373.15 K. A physical constant, so it lives next to the other
// absolute thresholds (Draper, melting points) rather than inside a shader.
constexpr float kWaterBoilingKelvin = 373.15f;

struct MaterialStateFieldStats {
    bool     stepped          = false;
    uint32_t field_count      = 0;   // objects carrying an MSF this step
    uint32_t element_count    = 0;   // total surface elements integrated
    uint32_t burning_count    = 0;   // elements above ignition with fuel left
    float    max_temperature  = 0.0f;
    float    mean_temperature = 0.0f;
    float    mean_char        = 0.0f;
    // max_char next to mean_char is the cheap proof that char became SPATIAL:
    // with per-triangle sampling the two tracked each other, with a texel mask a
    // localized burn drives max toward 1 while mean stays low.
    float    max_char         = 0.0f;
    float    fuel_remaining   = 0.0f;
    float    dispatch_ms      = 0.0f;
    float    readback_ms      = 0.0f;  // 0 unless a readback was requested
    // ── Phase 4: the ambient pass, billed separately ─────────────────────────
    // It runs once per frame per field, OUTSIDE the per-domain loop, so its cost
    // does not scale with domain count the way the gather does.
    bool     ambient_stepped  = false;
    float    ambient_ms       = 0.0f;
    uint32_t thermal_sources  = 0;  // Thermal force fields fed to the ambient pass
    uint32_t ambient_zones    = 0;  // domains overriding ambient inside their box
    // ── Phase 5: moisture ────────────────────────────────────────────────────
    float    mean_moisture    = 0.0f;
    float    max_moisture     = 0.0f;
    uint32_t wet_elements     = 0;  // above the "suppresses burning" threshold
    uint32_t wetting_domains  = 0;  // liquid domains that wetted anything this step
    float    wetting_ms       = 0.0f;
    // ── Phase 6b: melting ────────────────────────────────────────────────────
    float    mean_melt        = 0.0f;
    float    max_melt         = 0.0f;
    uint32_t molten_elements  = 0;  // fully molten (melt >= 1)
    uint32_t melting_elements = 0;  // partially molten, absorbing latent heat
    // ── Phase 6a: UV lookup coverage ─────────────────────────────────────────
    // How much of the mask a UV query can actually land on. This is the number
    // that says whether geometry displacement will FIND its melt value, and it is
    // reported separately from the melt values themselves because "nothing melted"
    // and "melted but unreachable through UV" are different faults.
    uint32_t lookup_texels_covered = 0;
    uint32_t lookup_texels_total   = 0;
    uint32_t lookup_fields_no_uv   = 0;  // fell back to centroids: cannot displace
};

// Sparse CPU-side bridge consumed by fracture/Jolt. Derived from the latest
// requested MSF readback; Jolt never sees or samples the full surface field.
struct MaterialIntegritySummary {
    bool valid = false;
    float mean_integrity = 1.0f;
    float minimum_integrity = 1.0f;
    float remaining_support_ratio = 1.0f;
    Vec3 weakest_world_position = Vec3(0.0f, 0.0f, 0.0f);
    float total_mass_loss = 0.0f;
    uint64_t content_generation = 0;
};

struct MaterialMassBudgetSummary {
    bool valid = false;
    float initial_mass = 0.0f;
    float solid_mass = 0.0f;
    float pyrolyzed_mass = 0.0f;
    float molten_reservoir_mass = 0.0f;
    float transferred_mass = 0.0f;
    float conservation_error = 0.0f;
};

// One object's surface state.
//
// SAMPLING (Phase 3a): elements are TEXELS of the object's UV space, not
// triangles.
//
// Phase 1 sampled triangle centroids, which was right for thermal state but
// useless for shading: a default cube has 12 triangles, so a burn mark could
// only ever be 12 flat patches. Promoting the sample set to texels makes `char`
// spatially resolved for free — the gather/scatter shaders did not change at
// all, they were always "one invocation per element".
//
// A mesh with no usable UVs falls back to triangle centroids. That looks blocky
// but keeps working, which beats silently producing nothing.
//
// ★ RESOLUTION INDEPENDENCE: fuel and burn rate are per unit AREA, not per
// element. Without that, raising the mask resolution would multiply the total
// combustible mass in the scene and the same object would burn longer and vent
// more smoke purely because its mask got bigger.
//
// Phase 6 (melting) moves geometry and will still need a vertex mapping; texels
// cannot displace a surface. That promotion is called out in the plan doc.
class MaterialStateField {
public:
    std::string object_key;               // collider source_name
    // What this surface is made of, carried here so the render bridge can read
    // char_color / molten_emission without re-deriving the collider it came from.
    // ★ It is also the AUTHORITY for the step: the substance is resolved per
    // field from this name, not handed down once for the whole domain. A domain
    // holds objects of different materials, and a single shared profile made the
    // second collider burn as if it were made of the first one's substance.
    std::string substance_name;
    // Per-object delta on that profile, carried for the same reason.
    SubstanceOverride overrides;
    uint64_t    topology_generation = 0;  // from the mesh resolver; rebuild on change

    // Host mirror of the device state. Authoritative for cache/debug; the device
    // buffer is authoritative during a step (see the class comment in the .cpp).
    //
    // state layout, 8 floats (2 vec4) per element:
    //   [0..3] temperature, fuel_remaining, char, moisture
    //   [4..7] released_this_step, melt, mass_loss, transferred_mass
    static constexpr std::size_t kStateStride = 8u;

    std::vector<float> centers;  // 4 floats per element: xyz world, w = area
    std::vector<float> state;    // kStateStride floats per element

    // UV-space mask. `mask_resolution` is 0 when this field fell back to
    // triangle-centroid sampling (no usable UVs); `texel_index` then stays empty
    // and there is no mask to composite.
    // The mask is what the renderer consumes. One RGBA texture keeps every
    // derived visual channel on the same UV/sample/revision path:
    // R=char, G=temperature, B=mass-loss fraction, A=integrity.
    // B/A are complements today, but keeping the explicit integrity channel
    // leaves room for pressure/contact damage to diverge from chemistry later.
    static constexpr std::size_t kMaskChannels = 4;
    // ★ G is quantized against an ABSOLUTE Kelvin range, not the domain's
    // max_temperature. Incandescence is physics: iron glows at ~800 K whatever
    // the solver's ceiling happens to be. Quantizing against the ceiling made the
    // same hot object glow or not purely because a domain setting moved, and with
    // a default ceiling of 10 normalized a 717 K surface landed at 0.12 — below
    // the shader's glow threshold, so nothing ever lit up.
    // 3000 K spans past steel's boiling point; 8 bits gives ~12 K steps.
    // closesthit.rchit MUST use the same constant.
    static constexpr float kMaskKelvinRange = 3000.0f;
    int mask_resolution = 0;
    std::vector<uint32_t> texel_index;  // element -> texel, size == elementCount()
    // mask_resolution^2 * kMaskChannels, unorm8. Filled on readback.
    std::vector<uint8_t> char_mask;
    // Bumped whenever char_mask content changes, so the render bridge can skip
    // re-uploading a texture that did not move.
    uint64_t mask_revision = 0;

    // ── Phase 6a: the texel -> melt lookup that geometry displacement reads ────
    //
    // ★ There is no spatial vertex weld here, and that is the whole point. The
    // question "how much has the surface under this vertex melted?" does not need
    // vertex IDENTITY, only a melt VALUE, and a vertex already carries a UV. So
    // the chain is vertex -> UV -> texel -> melt: exactly the chain the renderer
    // walks when it samples the mask (closesthit.rchit, rawUV). Going through UV
    // is therefore not merely safe, it is REQUIRED — any other mapping could let
    // the displaced geometry and the glow/char shading disagree about where the
    // surface melted.
    //
    // ★ KNOWN AND ACCEPTED: if the UV layout has mirrored or overlapping islands,
    // two different surface locations share a texel and melt together. That is not
    // a new defect — it is the behaviour char has had since Phase 3a, and the
    // renderer already shades both sides identically. Geometry matching shading is
    // the self-consistent answer; "fixing" it here would desynchronise the two.
    //
    // Both arrays are res*res, rebuilt beside char_mask on readback. NOT cached to
    // disk: derived from `state`, like char_mask (see MaterialStateFieldSnapshot).
    std::vector<float>   melt_texel;    // 0 where uncovered
    std::vector<float>   local_mass_texel; // solid + untransferred molten fraction
    std::vector<uint8_t> melt_covered;  // 1 where at least one element wrote it

    ComputeBufferHandle gpu_centers;
    ComputeBufferHandle gpu_state;

    // Grow-only device buffers: the allocation may be larger than the current
    // element count after a mesh shrinks. Every upload/dispatch MUST be sized by
    // elementCount(), never by the buffer's byte size — uploading .size() worth
    // of a stale-larger buffer is a known silent-fallback bug in this codebase.
    std::size_t elementCount() const { return centers.size() / 4u; }

    bool centers_dirty = true;  // world positions changed (moving/animated object)
    bool state_dirty = false;   // host reset must replace the device-authoritative state
};

// ─────────────────────────────────────────────────────────────────────────────
// Cache snapshot (Phase 4b)
//
// MSF is per-OBJECT and lives on the simulation runtime, while the frame cache
// stores per-DOMAIN grid state. The two do not nest, and Phase 4 settled which
// way that asymmetry resolves: an object outside every domain is still
// simulated, so MSF must NOT be moved under a domain. It is cached alongside,
// the same way rigid/soft/particle snapshots already are.
//
// ★ Named channels, not the raw kStateStride block. The stride carries scratch
// (`released_this_step`, zeroed every gather) and a reserved slot; pinning the
// on-disk format to it would invalidate every cache the day a scratch slot is
// added. These six are the state that actually has to survive.
//
// Not stored, because it is DERIVED and cheaper to rebuild than to keep correct:
//   - centers / texel_index — rebuilt by syncField from the live mesh,
//   - char_mask — regenerated from `charred`/`temperature` by scatterCharMask.
// Storing them would also make the file wrong the moment the mesh moved.
struct MaterialStateFieldSnapshot {
    std::string object_key;
    int      mask_resolution = 0;
    uint32_t element_count = 0;
    std::vector<float> temperature;
    std::vector<float> fuel;
    std::vector<float> charred;
    std::vector<float> moisture;   // Phase 5
    std::vector<float> melt;       // Phase 6
    std::vector<float> mass_loss;  // Phase 7
    std::vector<float> transferred_mass; // Phase 6 APIC debit ledger

    bool valid() const {
        const std::size_t n = element_count;
        return n > 0 && temperature.size() == n && fuel.size() == n &&
               charred.size() == n && moisture.size() == n &&
               melt.size() == n && mass_loss.size() == n &&
               transferred_mass.size() == n;
    }
};

// Owns every object's MSF and runs the per-step integration.
class MaterialStateFieldSystem {
public:
    // Creates or refreshes the field for `object_key` from a world-space triangle
    // soup. A changed `generation` rebuilds the field (state is reset, because the
    // element set no longer corresponds); an unchanged generation only refreshes
    // the centroid positions so a moving object keeps its accumulated char.
    bool syncField(const std::string& object_key,
                   const std::vector<SurfaceMeshTriangle>& triangles,
                   uint64_t generation,
                   int mask_resolution,
                   const std::string& substance_name,
                   const SubstanceOverride& overrides,
                   SimulationComputeContext& compute);

    // The gas fields MSF reads from and deposits into, plus the fixed-point
    // accumulators the scatter pass sums through. The accumulators are owned by
    // the caller (they are per-domain, cell-sized) and are cleared by the resolve
    // pass itself, so they need no separate zero-fill.
    struct GasBinding {
        ComputeBufferHandle density;
        ComputeBufferHandle temperature;
        ComputeBufferHandle fuel;
        ComputeBufferHandle flame;
        ComputeBufferHandle solid_mask;
        ComputeBufferHandle accum_fuel;
        ComputeBufferHandle accum_density;
        ComputeBufferHandle accum_heat;
        ComputeBufferHandle accum_flame;
        bool valid() const;
    };

    // Full three-pass step: gather (integrate state, record released mass) ->
    // scatter (atomically deposit into the target cells) -> resolve (fold the
    // accumulators into the gas fields and clear them).
    //
    // MSF is the sole owner of pyrolysis since Phase 2b; the voxel surface_state
    // path it used to shadow has been removed.
    //
    // ★ No profile/substance parameter: each field resolves its OWN substance
    // from the name it carries. Passing one down applied the first collider's
    // material to every object in the domain — a wooden crate next to an iron
    // beam burned both as wood (or neither, if iron happened to be resolved
    // first, since the scatter dispatch is skipped for non-combustibles).
    //
    // ★ Since Phase 4 the gather does NOT cool. Passive relaxation toward ambient
    // belongs to stepAmbient() below, which runs once per frame; leaving it here
    // meant every extra gas domain cooled the object a second time, and a domain
    // the object was not even inside cooled it just as hard.
    //
    // `oxygen_availability` is the world value, overridden by the domain when it
    // asks to. It scales pyrolysis burn rate only — it can smother a fire but it
    // can never light one.
    bool step(SimulationComputeContext& compute,
              const GasBinding& gas,
              int nx, int ny, int nz,
              std::size_t cell_count,
              const Vec3& grid_origin,
              float voxel_size,
              float dt,
              float max_temperature,
              const MaterialTemperatureScale& scale,
              float oxygen_availability);

    // ── Phase 4: ambient / boundary-condition pass ───────────────────────────
    //
    // A local heat source, in world space and in Kelvin. Built by the caller from
    // Thermal force fields — MaterialStateField deliberately does not know what a
    // Physics::ForceField is, so the layering stays one-directional.
    //
    // 32 bytes, two vec4 in std430. Keep in lockstep with sim_msf_ambient.comp.
    struct ThermalSource {
        float position[3] = {0.0f, 0.0f, 0.0f};
        float inner_radius = 0.0f;
        float falloff_radius = 5.0f;
        float delta_kelvin = 0.0f;   // added on top of ambient at full strength
        int   falloff_type = 2;      // Physics::FalloffType
        int   infinite = 0;          // 1 = no falloff, affects everywhere
    };

    // A domain's ambient override, as a world-space AABB. Element-granular on
    // purpose: one object can straddle a domain boundary, and half of it really
    // is in the hot box.
    struct AmbientZone {
        float bounds_min[3] = {0.0f, 0.0f, 0.0f};
        float ambient_kelvin = 293.0f;
        float bounds_max[3] = {0.0f, 0.0f, 0.0f};
        float _reserved = 0.0f;
    };

    // Relaxes every element toward its local ambient temperature. Runs once per
    // frame, before any domain gather, and does NOT need a gas domain to exist —
    // that is the whole point: an object outside every domain still has a defined
    // temperature and cools like a real object instead of being frozen or reset.
    bool stepAmbient(SimulationComputeContext& compute,
                     const WorldThermalState& world,
                     const std::vector<ThermalSource>& sources,
                     const std::vector<AmbientZone>& zones,
                     float dt,
                     float max_temperature);

    // ── Phase 5: liquid contact -> moisture ──────────────────────────────────
    //
    // Raises `moisture` where the surface sits inside a liquid domain's occupied
    // cells. Called once per FLUID domain, from that domain's branch, because the
    // occupancy grid it reads is per-domain and only valid after that domain has
    // splatted its density.
    //
    // ★ Per-domain is correct HERE, unlike the cooling that Phase 4 had to move
    // out of the gather: wetting is a monotonic SOURCE, so two overlapping water
    // tanks wetting the same plank is physically fine. A relaxation applied twice
    // is not. The distinction is the rule, not the special case.
    //
    // Moisture is only ever REMOVED by the ambient pass. One process, one owner —
    // splitting evaporation between here and the gather is exactly how the
    // double-cooling bug happened.
    bool stepWetting(SimulationComputeContext& compute,
                     ComputeBufferHandle liquid_occupancy,
                     int nx, int ny, int nz,
                     const Vec3& grid_origin,
                     float voxel_size,
                     float dt);

    // Stats need the device state back. That costs a stall, so it only happens
    // when something actually wants to look (stats panel / debug view) — never
    // per frame by default.
    void requestReadback() { readback_requested_ = true; }
    bool readbackPending() const { return readback_requested_; }
    // Called once per frame AFTER every domain has stepped. It used to live at
    // the end of step(), which meant a two-domain scene stalled the pipeline
    // twice and rebuilt every char mask twice for one frame of state — and in a
    // scene with no gas domain at all it never ran, so the panel read empty.
    void flushReadback(SimulationComputeContext& compute);

    const MaterialStateFieldStats& stats() const { return stats_; }

    // Drops accumulated state but keeps the fields (used on timeline reset to
    // frame 0 — MSF is runtime state and must be re-simulatable from scratch).
    void resetState();
    // Same, for one object. Burn damage is per-object and permanent by design, so
    // "undo the damage on THIS thing" needs to exist without resetting the scene.
    // Returns false when the object carries no field.
    bool clearField(const std::string& object_key);

    // Forgets fields whose object was not synced this step, so a deleted or
    // disabled collider does not keep a device allocation alive forever.
    void beginSyncPass();
    void endSyncPass(SimulationComputeContext& compute);

    void release(SimulationComputeContext& compute);

    // ── Frame cache (Phase 4b) ───────────────────────────────────────────────
    // Capture forces a host refresh: the device buffer is authoritative during a
    // step, so without it the snapshot would silently record whatever the last
    // requested readback happened to leave behind.
    std::vector<MaterialStateFieldSnapshot> captureSnapshot(
        SimulationComputeContext& compute);

    // Restore is DEFERRED by design. A scrub lands on a frame before the sim has
    // rebuilt its fields, so the element set that the snapshot belongs to may not
    // exist yet. Snapshots whose field is already present and element-compatible
    // are applied at once (the common case: scrubbing a running sim); the rest
    // are parked and claimed by syncField when it rebuilds that object.
    //
    // An element-count mismatch DROPS the snapshot rather than remapping it: the
    // element<->state correspondence is exactly what a changed mask resolution or
    // retopology destroys, and smearing burn marks onto unrelated surface is
    // worse than losing them.
    void restoreSnapshot(const std::vector<MaterialStateFieldSnapshot>& snapshot,
                         SimulationComputeContext& compute);

    const std::unordered_map<std::string, MaterialStateField>& fields() const {
        return fields_;
    }

    // ── Phase 6a: melt lookup for geometry displacement ──────────────────────
    //
    // Returns false when this object cannot be displaced at all — no field, or a
    // field that fell back to centroid sampling because the mesh has no usable UV
    // layout. A caller must treat false as "leave the geometry alone", never as
    // melt = 0: silently not melting is a visible, explicable outcome, whereas
    // silently displacing by a made-up value is not.
    //
    // `u`/`v` are the mesh's OWN, unflipped UVs — the same space buildTexelSamples
    // rasterized. Do NOT pass the V-flipped Vulkan sampling UV; that mirrors the
    // lookup vertically and lands on the far side of every UV island.
    static bool sampleMeltAtUV(const MaterialStateField& field,
                               float u, float v, float& out_melt);
    static bool sampleLocalMassAtUV(const MaterialStateField& field,
                                    float u, float v, float& out_fraction);

    // Convenience for one-off queries. Per-vertex loops should look the field up
    // ONCE with findField() and call sampleMeltAtUV directly — this does a hash
    // lookup per call.
    bool sampleMelt(const std::string& object_key, float u, float v, float& out_melt) const;

    const MaterialStateField* findField(const std::string& object_key) const {
        auto it = fields_.find(object_key);
        return it == fields_.end() ? nullptr : &it->second;
    }

    static MaterialIntegritySummary summarizeIntegrity(
        const MaterialStateField& field);
    static MaterialMassBudgetSummary summarizeMassBudget(
        const MaterialStateField& field);
    bool consumeMoltenMass(const std::string& object_key, float requested_mass,
                           SimulationComputeContext& compute,
                           float& out_consumed_mass);

private:
    bool ensureBuffers(SimulationComputeContext& compute, MaterialStateField& field);
    // Grow-only device mirrors of the ambient pass inputs. Both are ALWAYS
    // allocated with at least one element even when the scene has none: a
    // zero-sized storage buffer is not a legal descriptor, and the real count
    // travels in the push constant instead.
    bool ensureAmbientInputs(SimulationComputeContext& compute,
                             const std::vector<ThermalSource>& sources,
                             const std::vector<AmbientZone>& zones);
    static void clearField(MaterialStateField& field);
    static void scatterCharMask(MaterialStateField& field,
                                const MaterialTemperatureScale& scale);
    void readback(SimulationComputeContext& compute);
    // Pulls the device state into the host mirror for ONE field. Split out of
    // readback() so a cache capture can refresh the mirror without also
    // rebuilding masks and overwriting the frame's stats.
    bool refreshHostState(SimulationComputeContext& compute,
                          MaterialStateField& field);
    // Applies a snapshot to a field whose element set already matches. Uploads
    // and rebuilds the mask, because a scrub may not be followed by a step and
    // the renderer reads the mask, not the state.
    bool applySnapshot(const MaterialStateFieldSnapshot& snap,
                       MaterialStateField& field,
                       SimulationComputeContext& compute);

    std::unordered_map<std::string, MaterialStateField> fields_;
    // Snapshots waiting for their field to be rebuilt (see restoreSnapshot).
    // Cleared as they are claimed; a snapshot for an object that never comes back
    // simply expires with the next restore.
    std::unordered_map<std::string, MaterialStateFieldSnapshot> pending_restore_;
    ComputeBufferHandle gpu_thermal_sources_;
    ComputeBufferHandle gpu_ambient_zones_;
    std::vector<std::string> synced_this_pass_;
    MaterialStateFieldStats stats_;
    bool readback_requested_ = false;
    // The Kelvin mapping in force at the last step, carried to the readback so the
    // mask is quantized in absolute units rather than guessed at render time.
    MaterialTemperatureScale readback_scale_;
};

} // namespace RayTrophiSim
