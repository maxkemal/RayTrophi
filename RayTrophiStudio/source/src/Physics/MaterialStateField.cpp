/*
* ─────────────────────────────────────────────────────────────────────────────
* File:          MaterialStateField.cpp
* Project:       RayTrophi Studio
* Description:   Material State Field — host side. See MaterialStateField.h and
*                docs/NODE_SIMULATION_ARCHITECTURE_PLAN.md (Bölüm A, Faz 1).
*
* OWNERSHIP MODEL
* ---------------
* The device buffer is authoritative while the simulation runs; the host mirror
* is only refreshed when a readback is explicitly requested (stats/debug) or when
* the field is rebuilt. There is deliberately NO per-frame readback — that was
* the whole point of putting MSF on the GPU, and a silent readback would be
* indistinguishable from a slow GPU in a profile.
* ─────────────────────────────────────────────────────────────────────────────
*/
#include "MaterialStateField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace RayTrophiSim {

namespace {

using Clock = std::chrono::steady_clock;

float elapsedMs(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

// Push constant for sim_msf_gather. Must match the shader's PC block exactly.
struct MsfGatherConstants {
    int   dim[4] = {};          // nx, ny, nz, element_count
    float origin_voxel[4] = {}; // xyz origin, w voxel size
    float material0[4] = {};    // ignition_T, fuel_capacity, burn_rate, thermal_response
    float material1[4] = {};    // (reserved, was cooling_rate), char_rate, dt, max_temperature
    float material2[4] = {};    // boil_normalized, melt_normalized, melt_rate, unused
};
static_assert(sizeof(MsfGatherConstants) == 80,
              "sim_msf_gather push-constant ABI changed");

struct MsfScatterConstants {
    int   dim[4] = {};
    float origin_voxel[4] = {};
    float yield[4] = {};  // smoke_per_mass, heat_per_mass, flame_level, fp_scale
};
static_assert(sizeof(MsfScatterConstants) == 48,
              "sim_msf_scatter push-constant ABI changed");

struct MsfResolveConstants {
    int   dim[4] = {};
    float params[4] = {};  // fp_scale, max_temperature, unused, unused
};
static_assert(sizeof(MsfResolveConstants) == 32,
              "sim_msf_resolve push-constant ABI changed");

// Phase 4 ambient pass. Must match sim_msf_ambient.comp's PC block exactly.
struct MsfAmbientConstants {
    int   counts[4] = {};   // element_count, source_count, zone_count, unused
    float params[4] = {};   // dt, relax_rate, world_ambient_K, inv_kelvin_per_unit
    float params2[4] = {};  // max_temperature, dry_rate, boil_normalized, unused
};
static_assert(sizeof(MsfAmbientConstants) == 48,
              "sim_msf_ambient push-constant ABI changed");

// Phase 5 wetting pass. Must match sim_msf_wet.comp's PC block exactly.
struct MsfWetConstants {
    int   dim[4] = {};          // fluid nx, ny, nz, element_count
    float origin_voxel[4] = {}; // fluid grid origin xyz, w voxel size
    float params[4] = {};       // absorbency, dt, liquid_threshold, unused
};
static_assert(sizeof(MsfWetConstants) == 48,
              "sim_msf_wet push-constant ABI changed");

// Liquid occupancy above which a cell counts as "there is water here". The fluid
// domain's post-step density splat is normalized around 1.0 for a full cell, so
// a fraction of that catches the free-surface band without treating a stray
// splash particle's smear as a soaking.
constexpr float kLiquidOccupancyThreshold = 0.35f;
// Moisture above which pyrolysis is considered suppressed, for the stats row.
// The shader gates continuously; this is only what "wet elements" counts.
constexpr float kWetElementThreshold = 0.05f;

// Fixed-point scale for the scatter accumulators.
//
// ★ This was 1<<16, sized when an element was a TRIANGLE. Phase 3a promoted
// elements to TEXELS, which cut per-element released mass by ~3 orders of
// magnitude, and the scale was not revisited. Combined with the truncating
// uint() cast in the scatter shader that quietly ate the deposit: at a 6336-texel
// mask the smoke deposit lost 14%, and at a 512 mask fuel/smoke/heat ALL floored
// to zero — the object charred but vented nothing, so no flame was fed.
//
// It also broke Phase 3a's own resolution-independence guarantee: the area
// scaling keeps total mass constant, but per-element truncation scales with
// element count, so a finer mask smoked less.
//
// 1<<22 keeps the smallest realistic deposit around ~1 unit even at a 1024 mask,
// and the headroom is ample: a cell receives at most (elements over that cell) *
// (burn_rate * area * dt), which for any plausible object is well under 1e3 units
// of mass — ~4e9 of uint32 range against ~1e10 needed only if an entire large
// object's per-step burn landed in ONE cell, which the nearest-open-cell search
// cannot produce.
constexpr float kFixedPointScale = 4194304.0f;  // 1<<22

// One MSF sample: a point on the surface with the world area it represents, and
// which texel of the UV mask it drives (kNoTexel when sampling fell back to
// triangle centroids).
constexpr uint32_t kNoTexel = 0xFFFFFFFFu;

struct SurfaceSample {
    Vec3 position;
    float area = 0.0f;
    uint32_t texel = kNoTexel;
};

float triangleWorldArea(const SurfaceMeshTriangle& t) {
    // The cached `area` is authoritative when present; recompute otherwise so a
    // resolver that leaves it at zero cannot silently zero out the fuel budget.
    if (t.area > 0.0f) return t.area;
    return 0.5f * Vec3::cross(t.p1 - t.p0, t.p2 - t.p0).length();
}

bool triangleHasUsableUV(const SurfaceMeshTriangle& t) {
    // A degenerate UV triangle carries no area to rasterize into. Meshes that
    // were never unwrapped typically have all-zero UVs, which lands here.
    const float du1 = t.uv1.x - t.uv0.x, dv1 = t.uv1.y - t.uv0.y;
    const float du2 = t.uv2.x - t.uv0.x, dv2 = t.uv2.y - t.uv0.y;
    return std::fabs(du1 * dv2 - du2 * dv1) > 1e-12f;
}

// Rasterizes the mesh into UV space, one sample per covered texel. Each sample
// carries the world position that texel maps to, so the thermal gather reads the
// gas at the right place instead of at a triangle's average.
//
// Returns false when the mesh has no usable UV layout; the caller then falls back
// to centroid sampling rather than producing an empty field.
bool buildTexelSamples(const std::vector<SurfaceMeshTriangle>& triangles,
                       int resolution,
                       std::vector<SurfaceSample>& out) {
    if (resolution <= 0) return false;

    std::size_t uv_triangles = 0;
    for (const SurfaceMeshTriangle& t : triangles) {
        if (triangleHasUsableUV(t)) ++uv_triangles;
    }
    // A handful of stray UV'd triangles on an otherwise unwrapped-free mesh would
    // give a mask with a few islands and nothing else — worse than the honest
    // blocky fallback.
    if (uv_triangles * 2 < triangles.size()) return false;

    out.clear();
    const float res_f = static_cast<float>(resolution);
    std::vector<SurfaceSample> per_triangle;

    for (const SurfaceMeshTriangle& t : triangles) {
        if (!triangleHasUsableUV(t)) continue;
        per_triangle.clear();

        const float min_u = std::min({t.uv0.x, t.uv1.x, t.uv2.x});
        const float max_u = std::max({t.uv0.x, t.uv1.x, t.uv2.x});
        const float min_v = std::min({t.uv0.y, t.uv1.y, t.uv2.y});
        const float max_v = std::max({t.uv0.y, t.uv1.y, t.uv2.y});

        int x0 = static_cast<int>(std::floor(min_u * res_f));
        int x1 = static_cast<int>(std::ceil(max_u * res_f));
        int y0 = static_cast<int>(std::floor(min_v * res_f));
        int y1 = static_cast<int>(std::ceil(max_v * res_f));
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(resolution - 1, x1); y1 = std::min(resolution - 1, y1);

        const float du1 = t.uv1.x - t.uv0.x, dv1 = t.uv1.y - t.uv0.y;
        const float du2 = t.uv2.x - t.uv0.x, dv2 = t.uv2.y - t.uv0.y;
        const float det = du1 * dv2 - du2 * dv1;
        if (std::fabs(det) < 1e-12f) continue;
        const float inv_det = 1.0f / det;

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / res_f;
                const float v = (static_cast<float>(y) + 0.5f) / res_f;
                const float pu = u - t.uv0.x, pv = v - t.uv0.y;
                const float b1 = (pu * dv2 - du2 * pv) * inv_det;
                const float b2 = (du1 * pv - pu * dv1) * inv_det;
                const float b0 = 1.0f - b1 - b2;
                // A small negative tolerance keeps texels whose centre falls just
                // outside a shared edge; without it adjacent triangles both miss
                // the seam texel and the mask gets a one-texel crack along it.
                constexpr float kEdgeTolerance = -1e-3f;
                if (b0 < kEdgeTolerance || b1 < kEdgeTolerance || b2 < kEdgeTolerance) continue;

                SurfaceSample s;
                s.position = t.p0 * b0 + t.p1 * b1 + t.p2 * b2;
                s.texel = static_cast<uint32_t>(y) * static_cast<uint32_t>(resolution) +
                          static_cast<uint32_t>(x);
                per_triangle.push_back(s);
            }
        }

        if (per_triangle.empty()) {
            // Triangle smaller than one texel: keep its centroid so a thin strip
            // of geometry still burns instead of dropping out of the field.
            SurfaceSample s;
            s.position = (t.p0 + t.p1 + t.p2) * (1.0f / 3.0f);
            s.area = triangleWorldArea(t);
            const float cu = (t.uv0.x + t.uv1.x + t.uv2.x) / 3.0f;
            const float cv = (t.uv0.y + t.uv1.y + t.uv2.y) / 3.0f;
            const int cx = std::clamp(static_cast<int>(cu * res_f), 0, resolution - 1);
            const int cy = std::clamp(static_cast<int>(cv * res_f), 0, resolution - 1);
            s.texel = static_cast<uint32_t>(cy) * static_cast<uint32_t>(resolution) +
                      static_cast<uint32_t>(cx);
            out.push_back(s);
            continue;
        }

        // Split the triangle's real world area across the texels covering it, so
        // the total combustible mass is a property of the SURFACE and not of the
        // mask resolution.
        const float share = triangleWorldArea(t) / static_cast<float>(per_triangle.size());
        for (SurfaceSample& s : per_triangle) {
            s.area = share;
            out.push_back(s);
        }
    }
    return !out.empty();
}

void buildCentroidSamples(const std::vector<SurfaceMeshTriangle>& triangles,
                          std::vector<SurfaceSample>& out) {
    out.clear();
    out.reserve(triangles.size());
    for (const SurfaceMeshTriangle& t : triangles) {
        SurfaceSample s;
        s.position = (t.p0 + t.p1 + t.p2) * (1.0f / 3.0f);
        s.area = triangleWorldArea(t);
        out.push_back(s);
    }
}

} // namespace

bool MaterialStateFieldSystem::GasBinding::valid() const {
    return density.valid() && temperature.valid() && fuel.valid() &&
           flame.valid() && solid_mask.valid() && accum_fuel.valid() &&
           accum_density.valid() && accum_heat.valid() && accum_flame.valid();
}

// ─────────────────────────────────────────────────────────────────────────────
// Substance library
//
// Values are real physical constants where one exists. Where a number is a
// simulation tuning knob rather than a measurable property (fuel_capacity,
// burn_rate, char_rate, melt_viscosity) it is chosen relative to the others so
// the materials rank sensibly against each other: paper burns fast and leaves
// little, oak burns slowly and chars heavily, steel does not burn at all.
// ─────────────────────────────────────────────────────────────────────────────
MaterialStateFieldBridgeStats& materialStateFieldBridgeStats() {
    static MaterialStateFieldBridgeStats stats;
    return stats;
}

const std::vector<SubstanceProfile>& substanceLibrary() {
    static const std::vector<SubstanceProfile> library = [] {
        std::vector<SubstanceProfile> out;

        auto add = [&out](SubstanceProfile p) { out.push_back(std::move(p)); };

        // Wood is first, so it is the default: it is the substance that exercises
        // every channel (heats, ignites, chars, leaves ash) and therefore the one
        // a new collider should start as.
        {
            SubstanceProfile p;
            p.name = "Wood (Oak)";
            // Phase 5 moisture: porous end grain drinks water and holds it.
            p.absorbency = 0.55f; p.dry_rate = 0.025f;
            p.density = 750.0f; p.specific_heat = 1700.0f; p.conductivity = 0.17f;
            p.emissivity = 0.9f;
            // Low conductivity: the surface heats and ignites long before the
            // bulk does, which is why wood chars rather than melting through.
            p.thermal_response = 2.0f; p.cooling_rate = 0.06f;
            p.combustible = true;
            p.ignition_kelvin = 573.0f;
            p.fuel_capacity = 6.0f; p.burn_rate = 0.6f;
            p.char_rate = 1.0f; p.ash_yield = 0.12f;
            p.char_color[0] = 0.045f; p.char_color[1] = 0.035f; p.char_color[2] = 0.03f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Iron";
            // Phase 5 moisture: water beads off and flashes away.
            p.absorbency = 0.0f;  p.dry_rate = 0.30f;
            p.density = 7874.0f; p.specific_heat = 449.0f; p.conductivity = 80.4f;
            p.emissivity = 0.35f;
            // Metal follows the gas quickly (high conductivity) and sheds heat
            // quickly too — that is what makes glowing steel cool visibly.
            p.thermal_response = 6.0f; p.cooling_rate = 0.10f;
            p.combustible = false;
            p.meltable = true;
            p.melt_kelvin = 1811.0f; p.boiling_kelvin = 3134.0f;
            p.latent_heat_fusion = 2.72e5f; p.melt_viscosity = 0.35f;
            p.molten_emission = 1.4f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Steel";
            // Phase 5 moisture: same as iron.
            p.absorbency = 0.0f;  p.dry_rate = 0.30f;
            p.density = 7850.0f; p.specific_heat = 490.0f; p.conductivity = 45.0f;
            p.emissivity = 0.4f;
            p.thermal_response = 5.5f; p.cooling_rate = 0.10f;
            p.meltable = true;
            p.melt_kelvin = 1698.0f; p.boiling_kelvin = 3134.0f;
            p.latent_heat_fusion = 2.6e5f; p.melt_viscosity = 0.4f;
            p.molten_emission = 1.4f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Copper";
            // Phase 5 moisture: same, and sheds heat fastest.
            p.absorbency = 0.0f;  p.dry_rate = 0.35f;
            p.density = 8960.0f; p.specific_heat = 385.0f; p.conductivity = 401.0f;
            p.emissivity = 0.15f;
            p.thermal_response = 8.0f; p.cooling_rate = 0.14f;
            p.meltable = true;
            p.melt_kelvin = 1358.0f; p.boiling_kelvin = 2835.0f;
            p.latent_heat_fusion = 2.05e5f; p.melt_viscosity = 0.25f;
            p.molten_emission = 1.3f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Paper";
            // Phase 5 moisture: soaks instantly, dries fast (thin).
            p.absorbency = 0.95f; p.dry_rate = 0.12f;
            p.density = 800.0f; p.specific_heat = 1340.0f; p.conductivity = 0.05f;
            p.thermal_response = 4.0f; p.cooling_rate = 0.10f;
            p.combustible = true;
            p.ignition_kelvin = 506.0f;   // ~233 C
            p.fuel_capacity = 0.8f; p.burn_rate = 2.5f;
            p.char_rate = 1.6f; p.ash_yield = 0.05f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Cloth";
            // Phase 5 moisture: the sponge of the library.
            p.absorbency = 1.0f;  p.dry_rate = 0.08f;
            p.density = 400.0f; p.specific_heat = 1300.0f; p.conductivity = 0.06f;
            p.thermal_response = 3.5f; p.cooling_rate = 0.09f;
            p.combustible = true;
            p.ignition_kelvin = 573.0f;
            p.fuel_capacity = 1.5f; p.burn_rate = 1.4f;
            p.char_rate = 1.3f; p.ash_yield = 0.08f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Plastic (PE)";
            // Phase 5 moisture: non-porous.
            p.absorbency = 0.05f; p.dry_rate = 0.25f;
            p.density = 950.0f; p.specific_heat = 1900.0f; p.conductivity = 0.4f;
            p.thermal_response = 3.0f; p.cooling_rate = 0.07f;
            p.combustible = true;
            p.ignition_kelvin = 622.0f;
            p.fuel_capacity = 3.0f; p.burn_rate = 1.0f;
            p.char_rate = 0.7f; p.ash_yield = 0.03f;
            // Plastic both melts and burns — the melt point is far below ignition,
            // so it slumps before it catches.
            p.meltable = true;
            p.melt_kelvin = 403.0f; p.boiling_kelvin = 673.0f;
            p.latent_heat_fusion = 2.0e5f; p.melt_viscosity = 0.75f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Wax";
            // Phase 5 moisture: hydrophobic by definition.
            p.absorbency = 0.0f;  p.dry_rate = 0.30f;
            p.density = 900.0f; p.specific_heat = 2100.0f; p.conductivity = 0.25f;
            p.thermal_response = 3.5f; p.cooling_rate = 0.08f;
            p.combustible = true;
            p.ignition_kelvin = 523.0f;
            p.fuel_capacity = 2.5f; p.burn_rate = 0.8f;
            p.char_rate = 0.2f; p.ash_yield = 0.01f;
            p.meltable = true;
            p.melt_kelvin = 330.0f; p.boiling_kelvin = 643.0f;
            p.latent_heat_fusion = 2.1e5f; p.melt_viscosity = 0.15f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Ice";
            // Phase 5 moisture: meltwater is not "moisture"; Phase 6 owns it.
            p.absorbency = 0.0f;  p.dry_rate = 0.01f;
            p.density = 917.0f; p.specific_heat = 2100.0f; p.conductivity = 2.2f;
            p.emissivity = 0.97f;
            p.thermal_response = 4.5f; p.cooling_rate = 0.05f;
            p.meltable = true;
            p.melt_kelvin = 273.15f; p.boiling_kelvin = 373.15f;
            p.latent_heat_fusion = 3.34e5f; p.melt_viscosity = 0.02f;
            p.molten_emission = 0.0f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Stone";
            // Phase 5 moisture: porous but slow both ways.
            p.absorbency = 0.25f; p.dry_rate = 0.04f;
            p.density = 2600.0f; p.specific_heat = 840.0f; p.conductivity = 2.5f;
            p.emissivity = 0.85f;
            p.thermal_response = 1.5f; p.cooling_rate = 0.04f;
            p.meltable = true;
            p.melt_kelvin = 1473.0f; p.boiling_kelvin = 3200.0f;
            p.latent_heat_fusion = 4.0e5f; p.melt_viscosity = 0.9f;
            p.molten_emission = 1.2f;
            add(p);
        }
        {
            SubstanceProfile p;
            p.name = "Flesh";
            // Phase 5 moisture: mostly water already.
            p.absorbency = 0.60f; p.dry_rate = 0.05f;
            p.density = 1050.0f; p.specific_heat = 3500.0f; p.conductivity = 0.5f;
            p.thermal_response = 2.5f; p.cooling_rate = 0.07f;
            p.combustible = true;
            p.ignition_kelvin = 673.0f;
            p.fuel_capacity = 4.0f; p.burn_rate = 0.5f;
            p.char_rate = 1.2f; p.ash_yield = 0.2f;
            add(p);
        }
        return out;
    }();
    return library;
}

const SubstanceProfile& findSubstance(const std::string& name) {
    const auto& library = substanceLibrary();
    for (const SubstanceProfile& p : library) {
        if (p.name == name) return p;
    }
    // Unknown name (project authored against a newer build, or an empty string
    // from a collider that predates substances): fall back to the default rather
    // than refusing to load.
    return library.front();
}

MaterialSubstance MaterialSubstance::fromProfile(const SubstanceProfile& profile,
                                                 const MaterialTemperatureScale& scale,
                                                 const SubstanceOverride& overrides) {
    MaterialSubstance out;
    out.thermal_response = std::max(0.0f, profile.thermal_response);
    out.cooling_rate = std::max(0.0f, profile.cooling_rate);
    out.absorbency = std::max(0.0f, profile.absorbency);
    out.dry_rate = std::max(0.0f, profile.dry_rate);
    // The single Kelvin -> normalized conversion point, as for every other
    // threshold. A shader-side constant would mean a different temperature at
    // every domain calibration — the Phase 3d incandescence bug, repeated.
    out.boil_normalized = scale.toNormalized(kWaterBoilingKelvin);

    // ── Phase 6b: melting ────────────────────────────────────────────────────
    // Sentinel rather than a bool for a non-meltable substance: the shader then
    // has one comparison and no way for a flag and a threshold to disagree.
    if (profile.meltable && profile.melt_kelvin > 0.0f) {
        out.melt_normalized = scale.toNormalized(profile.melt_kelvin);
        // Latent heat maps to a RATE, not to an energy budget. A real enthalpy
        // model needs mass and specific heat per element, which this layer does
        // not carry; what it does carry is the physically meaningful ORDERING —
        // ice (3.34e5) is stubborn, wax (2.1e5) gives way easily — and the
        // reference is iron's, so iron melts at 1.0 per normalized degree per
        // second. Stated as a tuning mapping because that is what it is.
        constexpr float kLatentReference = 2.72e5f;  // iron
        out.melt_rate = kLatentReference / std::max(1.0f, profile.latent_heat_fusion);
    } else {
        out.melt_normalized = 1.0e9f;
        out.melt_rate = 0.0f;
    }
    if (profile.combustible) {
        // Overrides are applied in Kelvin, BEFORE the conversion, so there stays
        // exactly one Kelvin -> normalized conversion point for the whole system.
        const float ignition_kelvin = overrides.override_ignition
            ? overrides.ignition_kelvin : profile.ignition_kelvin;
        out.ignition_temperature = std::max(0.0f, scale.toNormalized(ignition_kelvin));
        out.fuel_capacity =
            std::max(0.0f, profile.fuel_capacity * std::max(0.0f, overrides.fuel_capacity_scale));
        out.burn_rate =
            std::max(0.0f, profile.burn_rate * std::max(0.0f, overrides.burn_rate_scale));
        out.char_rate = std::max(0.0f, profile.char_rate);
    } else {
        // Non-combustible: no fuel at all, and combustion overrides are ignored
        // outright — scaling a burn rate must never make iron burn. It still
        // heats up (the temperature channel keeps integrating); it just never chars.
        out.ignition_temperature = 0.0f;
        out.fuel_capacity = 0.0f;
        out.burn_rate = 0.0f;
        out.char_rate = 0.0f;
    }
    return out;
}

bool MaterialStateFieldSystem::syncField(const std::string& object_key,
                                         const std::vector<SurfaceMeshTriangle>& triangles,
                                         uint64_t generation,
                                         int mask_resolution,
                                         const std::string& substance_name,
                                         const SubstanceOverride& overrides,
                                         SimulationComputeContext& compute) {
    if (object_key.empty() || triangles.empty()) return false;

    synced_this_pass_.push_back(object_key);
    MaterialStateField& field = fields_[object_key];

    std::vector<SurfaceSample> samples;
    int resolution = mask_resolution;
    if (!buildTexelSamples(triangles, resolution, samples)) {
        // No usable UV layout: blocky per-triangle sampling, and no mask.
        resolution = 0;
        buildCentroidSamples(triangles, samples);
    }
    if (samples.empty()) return false;
    const std::size_t element_count = samples.size();

    // A changed topology generation invalidates the element↔state correspondence.
    // Carrying char across a retopology would smear burn marks onto unrelated
    // parts of the surface, so the state is reset rather than remapped. A changed
    // mask resolution changes the element set the same way.
    // ★ Deliberately NOT keyed on `generation`. The surface cache re-versions
    // whenever an object is re-posed, and treating that as a rebuild wiped the
    // burn marks the moment the object was moved — the exact failure MSF exists
    // to prevent (the old voxel path lost char on contact loss).
    //
    // What actually invalidates the element<->state correspondence is the element
    // SET changing: a different count, or a different mask resolution. A re-pose
    // keeps both, so the state carries and only the world positions are refreshed.
    const bool rebuild = field.object_key != object_key ||
                         field.mask_resolution != resolution ||
                         field.elementCount() != element_count;

    field.object_key = object_key;
    field.substance_name = substance_name;
    field.overrides = overrides;
    field.topology_generation = generation;
    field.mask_resolution = resolution;
    field.centers.resize(element_count * 4u);
    field.texel_index.resize(resolution > 0 ? element_count : 0u);
    // ★ Only on rebuild. syncField runs EVERY frame, so wiping the mask here
    // unconditionally blanked the burn marks between readbacks — the mark
    // appeared and cleared instead of accumulating. Char is a permanent deposit;
    // nothing outside a rebuild may clear it.
    const std::size_t mask_bytes =
        resolution > 0 ? static_cast<std::size_t>(resolution) * resolution *
                         MaterialStateField::kMaskChannels : 0u;
    if (rebuild || field.char_mask.size() != mask_bytes) {
        field.char_mask.assign(mask_bytes, uint8_t{0});
    }

    for (std::size_t i = 0; i < element_count; ++i) {
        const SurfaceSample& s = samples[i];
        field.centers[i * 4u + 0u] = s.position.x;
        field.centers[i * 4u + 1u] = s.position.y;
        field.centers[i * 4u + 2u] = s.position.z;
        field.centers[i * 4u + 3u] = s.area;
        if (resolution > 0) field.texel_index[i] = s.texel;
    }
    field.centers_dirty = true;

    if (rebuild) {
        field.state.assign(element_count * MaterialStateField::kStateStride, 0.0f);
        // Fuel is seeded per element from the substance at first step (below);
        // seeding here would need the substance, which belongs to the caller's
        // per-domain context. -1 marks "not initialized yet".
        for (std::size_t i = 0; i < element_count; ++i) {
            field.state[i * MaterialStateField::kStateStride + 1u] = -1.0f;
        }
    }

    if (!ensureBuffers(compute, field)) return false;

    // Always size uploads by the CURRENT element count. The device buffers are
    // grow-only, so after a mesh shrinks the allocation is larger than the live
    // data — uploading the allocation's worth is the known silent-fallback bug.
    const std::size_t center_bytes = field.centers.size() * sizeof(float);
    if (!compute.uploadBuffer(field.gpu_centers, field.centers.data(), center_bytes)) {
        return false;
    }
    field.centers_dirty = false;

    if (rebuild) {
        if (!compute.uploadBuffer(field.gpu_state, field.state.data(),
                                  field.state.size() * sizeof(float))) {
            return false;
        }
    }

    // ── Claim a parked cache snapshot ────────────────────────────────────────
    // This is the ONLY place that knows the live element set, which is why the
    // restore is deferred to here rather than applied when the frame was
    // installed: a scrub lands before the sim has rebuilt its fields.
    //
    // Done AFTER the rebuild upload above, so the snapshot wins over the freshly
    // zeroed state rather than being overwritten by it.
    if (!pending_restore_.empty()) {
        auto pending = pending_restore_.find(object_key);
        if (pending != pending_restore_.end()) {
            // applySnapshot rejects an element-count / mask-resolution mismatch,
            // and the entry is dropped either way: a snapshot that no longer
            // corresponds to this surface must not be retried every frame, and
            // remapping it would smear burn marks onto unrelated geometry.
            applySnapshot(pending->second, field, compute);
            pending_restore_.erase(pending);
        }
    }
    return true;
}

bool MaterialStateFieldSystem::ensureBuffers(SimulationComputeContext& compute,
                                             MaterialStateField& field) {
    const std::size_t center_bytes = field.centers.size() * sizeof(float);
    const std::size_t state_bytes = field.state.size() * sizeof(float);
    if (center_bytes == 0 || state_bytes == 0) return false;

    const ComputeBufferUsage center_usage = ComputeBufferUsage::Storage |
                                            ComputeBufferUsage::Upload |
                                            ComputeBufferUsage::ReadOnly;
    const ComputeBufferUsage state_usage = ComputeBufferUsage::Storage |
                                           ComputeBufferUsage::Upload |
                                           ComputeBufferUsage::Download |
                                           ComputeBufferUsage::ReadWrite;

    auto ensure = [&](ComputeBufferHandle& handle,
                      const char* name,
                      std::size_t bytes,
                      ComputeBufferUsage usage) -> bool {
        if (handle.valid() && compute.getBufferSize(handle) >= bytes) return true;
        if (handle.valid()) compute.destroyBuffer(handle);
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = bytes;
        desc.usage = usage;
        handle = compute.createBuffer(desc);
        return handle.valid();
    };

    return ensure(field.gpu_centers, "MsfCenters", center_bytes, center_usage) &&
           ensure(field.gpu_state, "MsfState", state_bytes, state_usage);
}

bool MaterialStateFieldSystem::step(SimulationComputeContext& compute,
                                    const GasBinding& gas,
                                    int nx, int ny, int nz,
                                    std::size_t cell_count,
                                    const Vec3& grid_origin,
                                    float voxel_size,
                                    float dt,
                                    float max_temperature,
                                    const MaterialTemperatureScale& scale,
                                    float oxygen_availability) {
    // ★ Nothing is RESET here. step() runs once per gas domain, so assigning
    // would make the last domain the only one reported: its dispatch time would
    // hide the first domain's, and `stepped` would flip back to false whenever
    // the last domain happened to have no eligible field. stepAmbient() zeroes
    // the whole struct once per frame; these rows accumulate on top of that, and
    // flushReadback() fills the readback rows afterwards.
    if (fields_.empty()) return true;
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        return false;
    }
    if (!gas.valid() || cell_count == 0) return false;
    if (nx <= 0 || ny <= 0 || nz <= 0) return false;
    if (!(voxel_size > 0.0f) || !(dt > 0.0f) || !std::isfinite(dt)) return false;

    const auto dispatch_start = Clock::now();
    bool all_ok = true;
    bool any_scatter = false;
    uint32_t stepped_fields = 0u;

    for (auto& entry : fields_) {
        MaterialStateField& field = entry.second;
        const std::size_t element_count = field.elementCount();
        if (element_count == 0 || !field.gpu_centers.valid() || !field.gpu_state.valid()) {
            continue;
        }

        // ★ Resolve this object's OWN substance. Doing it per field is the whole
        // point: a domain holds objects of different materials, and the profile
        // drives ignition point, fuel, burn rate AND whether the scatter pass
        // runs at all.
        const SubstanceProfile& profile = findSubstance(field.substance_name);
        MaterialSubstance substance =
            MaterialSubstance::fromProfile(profile, scale, field.overrides);
        // Oxygen throttles pyrolysis only. Applied AFTER fromProfile so a
        // non-combustible substance (whose burn_rate is already 0) cannot be
        // talked into burning by a boundary condition, and so the substance layer
        // stays a pure function of the material.
        substance.burn_rate *= std::clamp(oxygen_availability, 0.0f, 1.0f);

        // First step after a rebuild: seed the combustible mass. Done here rather
        // than in syncField because the substance is per-domain context.
        constexpr std::size_t kStride = MaterialStateField::kStateStride;
        bool needs_seed = false;
        for (std::size_t i = 0; i < element_count && !needs_seed; ++i) {
            if (field.state[i * kStride + 1u] < 0.0f) needs_seed = true;
        }
        if (needs_seed) {
            for (std::size_t i = 0; i < element_count; ++i) {
                if (field.state[i * kStride + 1u] < 0.0f) {
                    // ★ Per unit AREA, not per element. Seeding a flat capacity
                    // per element would make a higher mask resolution multiply
                    // the object's combustible mass — the same cube would burn
                    // longer and smoke more purely because its mask got bigger.
                    const float area = field.centers[i * 4u + 3u];
                    field.state[i * kStride + 1u] =
                        std::max(0.0f, substance.fuel_capacity) * std::max(0.0f, area);
                }
            }
            if (!compute.uploadBuffer(field.gpu_state, field.state.data(),
                                      element_count * kStride * sizeof(float))) {
                all_ok = false;
                continue;
            }
        }

        // Positions follow the object every step: an animated or simulated
        // collider must carry its accumulated char with it, which is exactly what
        // the voxel path cannot do.
        if (field.centers_dirty) {
            if (!compute.uploadBuffer(field.gpu_centers, field.centers.data(),
                                      element_count * 4u * sizeof(float))) {
                all_ok = false;
                continue;
            }
            field.centers_dirty = false;
        }

        MsfGatherConstants pc;
        pc.dim[0] = nx; pc.dim[1] = ny; pc.dim[2] = nz;
        pc.dim[3] = static_cast<int>(element_count);
        pc.origin_voxel[0] = grid_origin.x;
        pc.origin_voxel[1] = grid_origin.y;
        pc.origin_voxel[2] = grid_origin.z;
        pc.origin_voxel[3] = voxel_size;
        pc.material0[0] = std::max(0.0f, substance.ignition_temperature);
        pc.material0[1] = std::max(0.0f, substance.fuel_capacity);
        pc.material0[2] = std::max(0.0f, substance.burn_rate);
        pc.material0[3] = std::max(0.0f, substance.thermal_response);
        // ★ Slot 0 used to be the passive cooling rate. Cooling moved to the
        // ambient pass in Phase 4 — leaving it here made every additional gas
        // domain cool the object again, including domains it was nowhere near.
        // The slot stays (the 64-byte ABI is shared with the .spv) and reads 0.
        pc.material1[0] = 0.0f;
        pc.material1[1] = std::max(0.0f, substance.char_rate);
        // Phase 5: a wet surface cannot climb past water's boiling point until
        // the water is gone, and cannot pyrolyse while wet. Derived Kelvin, not a
        // shader constant — see MaterialSubstance::boil_normalized.
        pc.material2[0] = substance.boil_normalized;
        // Phase 6b. A non-meltable substance carries a sentinel melt point far
        // above anything reachable, so the shader needs no separate flag.
        pc.material2[1] = substance.melt_normalized;
        pc.material2[2] = substance.melt_rate;
        pc.material1[2] = dt;
        pc.material1[3] = std::max(1.0f, max_temperature);

        const uint32_t element_groups =
            (static_cast<uint32_t>(element_count) + 255u) / 256u;

        ComputeBufferHandle gather_bufs[4] = {
            field.gpu_centers, field.gpu_state, gas.temperature, gas.solid_mask
        };
        ComputeDispatch gather;
        gather.kernel = "sim_msf_gather";
        gather.buffers = gather_bufs;
        gather.buffer_count = 4;
        gather.constants = &pc;
        gather.constants_size = sizeof(pc);
        gather.groups.groups_x = element_groups;
        if (!compute.dispatch(gather)) { all_ok = false; continue; }

        // Scatter: deposit the released vapour into the accumulators. Only
        // combustible substances can have released anything, so skip the whole
        // dispatch for iron/stone rather than launching a no-op over every
        // element of a dense mesh.
        if (profile.combustible) {
            MsfScatterConstants sc;
            sc.dim[0] = nx; sc.dim[1] = ny; sc.dim[2] = nz;
            sc.dim[3] = static_cast<int>(element_count);
            sc.origin_voxel[0] = grid_origin.x;
            sc.origin_voxel[1] = grid_origin.y;
            sc.origin_voxel[2] = grid_origin.z;
            sc.origin_voxel[3] = voxel_size;
            sc.yield[0] = std::max(0.0f, profile.smoke_yield);
            sc.yield[1] = std::max(0.0f, profile.heat_release);
            sc.yield[2] = std::clamp(profile.flame_level, 0.0f, 1.0f);
            sc.yield[3] = kFixedPointScale;

            ComputeBufferHandle scatter_bufs[7] = {
                field.gpu_centers, field.gpu_state, gas.solid_mask,
                gas.accum_fuel, gas.accum_density, gas.accum_heat, gas.accum_flame
            };
            ComputeDispatch scatter;
            scatter.kernel = "sim_msf_scatter";
            scatter.buffers = scatter_bufs;
            scatter.buffer_count = 7;
            scatter.constants = &sc;
            scatter.constants_size = sizeof(sc);
            scatter.groups.groups_x = element_groups;
            if (!compute.dispatch(scatter)) { all_ok = false; continue; }
            any_scatter = true;
        }

        stepped_fields += 1u;
    }

    // Resolve once for the whole domain, after every field has scattered — the
    // accumulators are per-cell and shared, so folding them in per field would
    // apply one field's deposit several times.
    if (any_scatter) {
        MsfResolveConstants rc;
        rc.dim[0] = nx; rc.dim[1] = ny; rc.dim[2] = nz;
        rc.dim[3] = static_cast<int>(cell_count);
        rc.params[0] = kFixedPointScale;
        rc.params[1] = std::max(1.0f, max_temperature);

        ComputeBufferHandle resolve_bufs[8] = {
            gas.density, gas.temperature, gas.fuel, gas.flame,
            gas.accum_fuel, gas.accum_density, gas.accum_heat, gas.accum_flame
        };
        ComputeDispatch resolve;
        resolve.kernel = "sim_msf_resolve";
        resolve.buffers = resolve_bufs;
        resolve.buffer_count = 8;
        resolve.constants = &rc;
        resolve.constants_size = sizeof(rc);
        // Sized by the live cell count, never by the buffer's byte size: the
        // accumulators are grow-only and a stale-larger allocation would make
        // this walk cells that are not part of the current grid.
        resolve.groups.groups_x = (static_cast<uint32_t>(cell_count) + 255u) / 256u;
        if (!compute.dispatch(resolve)) all_ok = false;
    }

    stats_.stepped = stats_.stepped || (stepped_fields > 0u);
    stats_.dispatch_ms += elapsedMs(dispatch_start, Clock::now());

    // The mask quantizes temperature in ABSOLUTE Kelvin, so the mapping has to
    // travel with the readback rather than being guessed at render time.
    readback_scale_ = scale;
    // ★ The readback is NOT done here any more. step() is called once per gas
    // domain, so a two-domain scene stalled the pipeline twice per frame and
    // rebuilt every char mask twice for one frame's worth of state. The caller
    // now calls flushReadback() once, after every domain has stepped.
    return all_ok;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ambient / boundary-condition pass (Phase 4)
//
// This is the layer that makes an object outside every domain a real object: it
// relaxes toward the room, warms up next to a Thermal field, and keeps its char.
// It runs once per frame regardless of how many gas domains exist — which is
// exactly what the old in-gather cooling could not do.
// ─────────────────────────────────────────────────────────────────────────────
bool MaterialStateFieldSystem::ensureAmbientInputs(
        SimulationComputeContext& compute,
        const std::vector<ThermalSource>& sources,
        const std::vector<AmbientZone>& zones) {
    const ComputeBufferUsage usage = ComputeBufferUsage::Storage |
                                     ComputeBufferUsage::Upload |
                                     ComputeBufferUsage::ReadOnly;

    auto ensure = [&](ComputeBufferHandle& handle,
                      const char* name,
                      std::size_t bytes) -> bool {
        if (handle.valid() && compute.getBufferSize(handle) >= bytes) return true;
        if (handle.valid()) compute.destroyBuffer(handle);
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = bytes;
        desc.usage = usage;
        handle = compute.createBuffer(desc);
        return handle.valid();
    };

    // Always at least one element: a zero-sized storage buffer is not a legal
    // descriptor, and the real counts travel in the push constant. The dummy is
    // never read because the shader loops to `count`.
    const std::size_t source_bytes =
        std::max<std::size_t>(1u, sources.size()) * sizeof(ThermalSource);
    const std::size_t zone_bytes =
        std::max<std::size_t>(1u, zones.size()) * sizeof(AmbientZone);
    if (!ensure(gpu_thermal_sources_, "MsfThermalSources", source_bytes)) return false;
    if (!ensure(gpu_ambient_zones_, "MsfAmbientZones", zone_bytes)) return false;

    // ★ Uploaded by live element count, never by the allocation's size: these are
    // grow-only, and after a field is deleted the buffer stays large.
    if (!sources.empty() &&
        !compute.uploadBuffer(gpu_thermal_sources_, sources.data(),
                              sources.size() * sizeof(ThermalSource))) {
        return false;
    }
    if (!zones.empty() &&
        !compute.uploadBuffer(gpu_ambient_zones_, zones.data(),
                              zones.size() * sizeof(AmbientZone))) {
        return false;
    }
    return true;
}

bool MaterialStateFieldSystem::stepAmbient(SimulationComputeContext& compute,
                                           const WorldThermalState& world,
                                           const std::vector<ThermalSource>& sources,
                                           const std::vector<AmbientZone>& zones,
                                           float dt,
                                           float max_temperature) {
    // This pass owns the whole stats struct for the frame: it is the first MSF
    // work to run, and it is the only pass guaranteed to run at all.
    stats_ = MaterialStateFieldStats{};
    stats_.thermal_sources = static_cast<uint32_t>(sources.size());
    stats_.ambient_zones = static_cast<uint32_t>(zones.size());
    for (const auto& entry : fields_) {
        const std::size_t n = entry.second.elementCount();
        if (n == 0u) continue;
        stats_.field_count += 1u;
        stats_.element_count += static_cast<uint32_t>(n);
    }
    // The Kelvin mapping is the world's, always — a readback can happen in a
    // frame where no domain stepped, and the mask must still quantize correctly.
    readback_scale_ = world.scale();

    if (fields_.empty()) return true;
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        return false;
    }
    if (!(dt > 0.0f) || !std::isfinite(dt)) return false;
    if (!ensureAmbientInputs(compute, sources, zones)) return false;

    const auto start = Clock::now();
    const float kelvin_per_unit = std::max(1e-3f, world.kelvin_per_unit);
    const float convection = std::max(0.0f, world.convection_coefficient);
    bool all_ok = true;

    for (auto& entry : fields_) {
        MaterialStateField& field = entry.second;
        const std::size_t element_count = field.elementCount();
        if (element_count == 0 || !field.gpu_centers.valid() || !field.gpu_state.valid()) {
            continue;
        }
        // Positions must be current here too: this pass is the only one that runs
        // when there is no gas domain, so without this an object moved into a
        // Thermal field would be sampled at where it used to be.
        if (field.centers_dirty) {
            if (!compute.uploadBuffer(field.gpu_centers, field.centers.data(),
                                      element_count * 4u * sizeof(float))) {
                all_ok = false;
                continue;
            }
            field.centers_dirty = false;
        }

        // Passive cooling is a material property, so it is resolved per field for
        // the same reason the substance is (Phase 3e).
        const SubstanceProfile& profile = findSubstance(field.substance_name);
        const MaterialTemperatureScale scale = world.scale();

        MsfAmbientConstants pc;
        pc.counts[0] = static_cast<int>(element_count);
        pc.counts[1] = static_cast<int>(sources.size());
        pc.counts[2] = static_cast<int>(zones.size());
        pc.params[0] = dt;
        pc.params[1] = std::max(0.0f, profile.cooling_rate) * convection;
        pc.params[2] = world.ambient_kelvin;
        pc.params[3] = 1.0f / kelvin_per_unit;
        pc.params2[0] = std::max(1.0f, max_temperature);
        // ★ Drying lives HERE and nowhere else. Moisture has exactly one sink,
        // for the same reason cooling now has exactly one: a process split across
        // the per-frame pass and the per-domain gather runs once per gas domain
        // and silently scales with domain count.
        pc.params2[1] = std::max(0.0f, profile.dry_rate);
        pc.params2[2] = scale.toNormalized(kWaterBoilingKelvin);

        ComputeBufferHandle bufs[4] = {
            field.gpu_centers, field.gpu_state,
            gpu_thermal_sources_, gpu_ambient_zones_
        };
        ComputeDispatch ambient;
        ambient.kernel = "sim_msf_ambient";
        ambient.buffers = bufs;
        ambient.buffer_count = 4;
        ambient.constants = &pc;
        ambient.constants_size = sizeof(pc);
        ambient.groups.groups_x = (static_cast<uint32_t>(element_count) + 255u) / 256u;
        if (!compute.dispatch(ambient)) { all_ok = false; continue; }
        stats_.ambient_stepped = true;
    }

    stats_.ambient_ms = elapsedMs(start, Clock::now());
    return all_ok;
}

bool MaterialStateFieldSystem::stepWetting(SimulationComputeContext& compute,
                                           ComputeBufferHandle liquid_occupancy,
                                           int nx, int ny, int nz,
                                           const Vec3& grid_origin,
                                           float voxel_size,
                                           float dt) {
    if (fields_.empty() || !liquid_occupancy.valid()) return false;
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        return false;
    }
    if (nx <= 0 || ny <= 0 || nz <= 0) return false;
    if (!(voxel_size > 0.0f) || !(dt > 0.0f) || !std::isfinite(dt)) return false;

    const auto start = Clock::now();
    bool any = false;

    for (auto& entry : fields_) {
        MaterialStateField& field = entry.second;
        const std::size_t element_count = field.elementCount();
        if (element_count == 0 || !field.gpu_centers.valid() || !field.gpu_state.valid()) {
            continue;
        }
        // Absorbency is a material property, resolved per field like every other
        // one since Phase 3e. A non-absorbent substance is skipped outright
        // rather than dispatched with a zero rate: water running off an iron beam
        // should cost nothing, not a no-op pass over every texel of it.
        const SubstanceProfile& profile = findSubstance(field.substance_name);
        if (!(profile.absorbency > 0.0f)) continue;

        // Positions must be current: the wetting test is purely spatial, so a
        // stale centre samples the water the object used to be standing in.
        if (field.centers_dirty) {
            if (!compute.uploadBuffer(field.gpu_centers, field.centers.data(),
                                      element_count * 4u * sizeof(float))) {
                continue;
            }
            field.centers_dirty = false;
        }

        MsfWetConstants pc;
        pc.dim[0] = nx; pc.dim[1] = ny; pc.dim[2] = nz;
        pc.dim[3] = static_cast<int>(element_count);
        pc.origin_voxel[0] = grid_origin.x;
        pc.origin_voxel[1] = grid_origin.y;
        pc.origin_voxel[2] = grid_origin.z;
        pc.origin_voxel[3] = voxel_size;
        pc.params[0] = profile.absorbency;
        pc.params[1] = dt;
        pc.params[2] = kLiquidOccupancyThreshold;

        ComputeBufferHandle bufs[3] = {
            field.gpu_centers, field.gpu_state, liquid_occupancy
        };
        ComputeDispatch wet;
        wet.kernel = "sim_msf_wet";
        wet.buffers = bufs;
        wet.buffer_count = 3;
        wet.constants = &pc;
        wet.constants_size = sizeof(pc);
        wet.groups.groups_x = (static_cast<uint32_t>(element_count) + 255u) / 256u;
        if (!compute.dispatch(wet)) continue;
        any = true;
    }

    if (any) stats_.wetting_domains += 1u;
    // Accumulates: this runs once per LIQUID domain, so assigning would report
    // only the last tank's cost.
    stats_.wetting_ms += elapsedMs(start, Clock::now());
    return any;
}

void MaterialStateFieldSystem::flushReadback(SimulationComputeContext& compute) {
    if (readback_requested_) readback(compute);
}

void MaterialStateFieldSystem::scatterCharMask(MaterialStateField& field,
                                               const MaterialTemperatureScale& scale) {
    const int res = field.mask_resolution;
    if (res <= 0 || field.texel_index.empty()) return;
    const std::size_t texels = static_cast<std::size_t>(res) * static_cast<std::size_t>(res);
    const std::size_t bytes = texels * MaterialStateField::kMaskChannels;
    if (field.char_mask.size() != bytes) field.char_mask.assign(bytes, 0u);
    std::fill(field.char_mask.begin(), field.char_mask.end(), uint8_t{0});

    // ── Phase 6a: the melt lookup is rebuilt in LOCKSTEP with the mask ────────
    // Same pass, same texels, same dilation. If these two ever drifted apart the
    // geometry would melt somewhere the shading did not, which is precisely the
    // failure the UV routing exists to make impossible.
    if (field.melt_texel.size() != texels) field.melt_texel.assign(texels, 0.0f);
    if (field.melt_covered.size() != texels) field.melt_covered.assign(texels, 0u);
    std::fill(field.melt_texel.begin(), field.melt_texel.end(), 0.0f);
    std::fill(field.melt_covered.begin(), field.melt_covered.end(), uint8_t{0});

    constexpr std::size_t kStride = MaterialStateField::kStateStride;
    const std::size_t element_count =
        std::min(field.elementCount(), field.texel_index.size());
    // ABSOLUTE Kelvin, not a fraction of the domain ceiling. Incandescence is a
    // property of the temperature, not of a solver setting: quantizing against
    // max_temperature made the same hot surface glow or not depending on a domain
    // slider, and with the default ceiling a 717 K surface quantized to 0.12 —
    // under the shader's glow threshold, so nothing ever lit up.
    const float inv_kelvin_range = 1.0f / MaterialStateField::kMaskKelvinRange;

    auto quantize = [](float v) -> uint8_t {
        return static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
    };

    for (std::size_t i = 0; i < element_count; ++i) {
        const uint32_t texel = field.texel_index[i];
        if (texel >= texels) continue;
        const uint8_t charred = quantize(field.state[i * kStride + 2u]);
        const float kelvin = scale.toKelvin(field.state[i * kStride + 0u]);
        const uint8_t heat = quantize(kelvin * inv_kelvin_range);
        const std::size_t o = static_cast<std::size_t>(texel) * MaterialStateField::kMaskChannels;
        // max, not overwrite: two triangles can map to the same texel across a UV
        // seam, and the burnt/hotter one must win rather than whichever came last.
        field.char_mask[o + 0u] = std::max(field.char_mask[o + 0u], charred);
        field.char_mask[o + 1u] = std::max(field.char_mask[o + 1u], heat);
        // Max for the same reason as above: across a UV seam the MORE melted
        // element must win, so a vertex on the seam sinks with its neighbours
        // instead of staying pinned to whichever element was rasterized last.
        field.melt_texel[texel] = std::max(field.melt_texel[texel],
                                           field.state[i * kStride + 5u]);
        field.melt_covered[texel] = 1u;
    }

    // One-texel dilation. Texels straddling a UV island border get no sample, so
    // without this a burn mark shows a hairline crack along every seam once the
    // mask is filtered at render time.
    // ★ The melt lookup is dilated in the SAME loop, not a separate one. A seam
    // texel that the mask fills but the lookup does not would leave a vertex
    // reading melt = 0 on a surface the renderer is already showing as molten.
    std::vector<uint8_t> dilated = field.char_mask;
    std::vector<float>   dilated_melt = field.melt_texel;
    std::vector<uint8_t> dilated_cover = field.melt_covered;
    for (int y = 0; y < res; ++y) {
        for (int x = 0; x < res; ++x) {
            const std::size_t t = static_cast<std::size_t>(y) * res + x;
            const std::size_t o = t * MaterialStateField::kMaskChannels;
            const bool mask_empty = (field.char_mask[o] == 0u && field.char_mask[o + 1u] == 0u);
            const bool cover_empty = (field.melt_covered[t] == 0u);
            if (!mask_empty && !cover_empty) continue;
            uint8_t best_char = 0u, best_heat = 0u, best_cover = 0u;
            float best_melt = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= res || ny >= res) continue;
                    const std::size_t nt = static_cast<std::size_t>(ny) * res + nx;
                    const std::size_t n = nt * MaterialStateField::kMaskChannels;
                    best_char = std::max(best_char, field.char_mask[n]);
                    best_heat = std::max(best_heat, field.char_mask[n + 1u]);
                    if (field.melt_covered[nt] != 0u) {
                        best_cover = 1u;
                        best_melt = std::max(best_melt, field.melt_texel[nt]);
                    }
                }
            }
            if (mask_empty) {
                dilated[o] = best_char;
                dilated[o + 1u] = best_heat;
            }
            if (cover_empty) {
                dilated_melt[t] = best_melt;
                dilated_cover[t] = best_cover;
            }
        }
    }
    field.char_mask.swap(dilated);
    field.melt_texel.swap(dilated_melt);
    field.melt_covered.swap(dilated_cover);
    field.mask_revision += 1u;
}

bool MaterialStateFieldSystem::sampleMeltAtUV(const MaterialStateField& field,
                                              float u, float v, float& out_melt) {
    const int res = field.mask_resolution;
    // res == 0 is the centroid fallback: no UV layout, so no lookup is possible.
    // Reported as "cannot displace", never as melt = 0.
    if (res <= 0 || field.melt_texel.empty()) return false;

    // Wrap rather than clamp. UVs outside [0,1) are a legitimate tiling layout,
    // and clamping would pile every out-of-range vertex onto the border texels —
    // a stripe of geometry melting along the UV edge for no physical reason.
    auto wrap01 = [](float t) {
        t = t - std::floor(t);
        return (t < 0.0f || t >= 1.0f) ? 0.0f : t;
    };
    const int x = std::clamp(static_cast<int>(wrap01(u) * static_cast<float>(res)), 0, res - 1);
    const int y = std::clamp(static_cast<int>(wrap01(v) * static_cast<float>(res)), 0, res - 1);
    const std::size_t t = static_cast<std::size_t>(y) * static_cast<std::size_t>(res) + x;
    if (t >= field.melt_covered.size() || field.melt_covered[t] == 0u) {
        // Inside the mask but on no island: this vertex's UV lands in empty UV
        // space. The one-texel dilation above already covers seams, so reaching
        // here means genuinely unmapped surface. Do not displace it.
        return false;
    }
    out_melt = field.melt_texel[t];
    return true;
}

bool MaterialStateFieldSystem::sampleMelt(const std::string& object_key,
                                          float u, float v, float& out_melt) const {
    const MaterialStateField* field = findField(object_key);
    return field ? sampleMeltAtUV(*field, u, v, out_melt) : false;
}

bool MaterialStateFieldSystem::refreshHostState(SimulationComputeContext& compute,
                                                MaterialStateField& field) {
    const std::size_t element_count = field.elementCount();
    constexpr std::size_t kStride = MaterialStateField::kStateStride;
    if (element_count == 0 || !field.gpu_state.valid()) return false;
    // ★ Sized by the live element count, never by the buffer's byte size: the
    // device buffers are grow-only, so after a mesh shrinks the allocation is
    // larger than the live data.
    return compute.downloadBuffer(field.gpu_state, field.state.data(),
                                  element_count * kStride * sizeof(float));
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame cache (Phase 4b)
// ─────────────────────────────────────────────────────────────────────────────
std::vector<MaterialStateFieldSnapshot> MaterialStateFieldSystem::captureSnapshot(
        SimulationComputeContext& compute) {
    std::vector<MaterialStateFieldSnapshot> out;
    if (fields_.empty()) return out;
    out.reserve(fields_.size());

    // The device buffer is authoritative while the sim runs, so the host mirror
    // has to be pulled here. Without it the snapshot would record whatever the
    // last panel-requested readback left behind — i.e. correct only while the
    // stats panel happened to be open, which is the worst kind of bug.
    const bool can_read = compute.backendType() == ComputeBackendType::VulkanCompute &&
                          compute.supportsDispatch();
    if (can_read) compute.synchronize();

    constexpr std::size_t kStride = MaterialStateField::kStateStride;
    for (auto& entry : fields_) {
        MaterialStateField& field = entry.second;
        const std::size_t n = field.elementCount();
        if (n == 0) continue;
        if (can_read) refreshHostState(compute, field);
        if (field.state.size() < n * kStride) continue;

        MaterialStateFieldSnapshot snap;
        snap.object_key = entry.first;
        snap.mask_resolution = field.mask_resolution;
        snap.element_count = static_cast<uint32_t>(n);
        snap.temperature.resize(n); snap.fuel.resize(n);
        snap.charred.resize(n);     snap.moisture.resize(n);
        snap.melt.resize(n);        snap.mass_loss.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const float* s = &field.state[i * kStride];
            snap.temperature[i] = s[0];
            snap.fuel[i]        = s[1];
            snap.charred[i]     = s[2];
            snap.moisture[i]    = s[3];
            // s[4] is released_this_step — transient scratch, deliberately not
            // stored: the gather zeroes it every step and scatter consumes it
            // within the same frame.
            snap.melt[i]        = s[5];
            snap.mass_loss[i]   = s[6];
        }
        out.push_back(std::move(snap));
    }
    return out;
}

bool MaterialStateFieldSystem::applySnapshot(const MaterialStateFieldSnapshot& snap,
                                             MaterialStateField& field,
                                             SimulationComputeContext& compute) {
    const std::size_t n = field.elementCount();
    if (!snap.valid() || n != static_cast<std::size_t>(snap.element_count)) return false;
    if (field.mask_resolution != snap.mask_resolution) return false;

    constexpr std::size_t kStride = MaterialStateField::kStateStride;
    if (field.state.size() < n * kStride) field.state.assign(n * kStride, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        float* s = &field.state[i * kStride];
        s[0] = snap.temperature[i];
        s[1] = snap.fuel[i];
        s[2] = snap.charred[i];
        s[3] = snap.moisture[i];
        s[4] = 0.0f;                  // released_this_step: per-step scratch
        s[5] = snap.melt[i];
        s[6] = snap.mass_loss[i];
    }
    if (field.gpu_state.valid()) {
        compute.uploadBuffer(field.gpu_state, field.state.data(),
                             n * kStride * sizeof(float));
    }
    // ★ Rebuild the mask here, not at the next readback. A scrub is often NOT
    // followed by a step (the timeline can land paused), and the renderer reads
    // the MASK, not the state — without this the object would keep showing the
    // damage it had before the scrub. Same trap Clear Damage hit in Phase 3c.
    scatterCharMask(field, readback_scale_);
    return true;
}

void MaterialStateFieldSystem::restoreSnapshot(
        const std::vector<MaterialStateFieldSnapshot>& snapshot,
        SimulationComputeContext& compute) {
    pending_restore_.clear();
    for (const MaterialStateFieldSnapshot& snap : snapshot) {
        if (!snap.valid() || snap.object_key.empty()) continue;
        auto it = fields_.find(snap.object_key);
        if (it != fields_.end() && applySnapshot(snap, it->second, compute)) {
            continue;  // field already existed and matched — done
        }
        // Field not built yet (or the element set has moved on): park it for
        // syncField, which is the only place that knows the live element set.
        pending_restore_[snap.object_key] = snap;
    }
}

void MaterialStateFieldSystem::readback(SimulationComputeContext& compute) {
    const auto start = Clock::now();
    compute.synchronize();

    double temperature_sum = 0.0;
    double char_sum = 0.0;
    double fuel_sum = 0.0;
    double moisture_sum = 0.0;
    double melt_sum = 0.0;
    uint32_t counted = 0u;

    for (auto& entry : fields_) {
        MaterialStateField& field = entry.second;
        const std::size_t element_count = field.elementCount();
        constexpr std::size_t kStride = MaterialStateField::kStateStride;
        if (element_count == 0) continue;
        if (!refreshHostState(compute, field)) continue;
        for (std::size_t i = 0; i < element_count; ++i) {
            const float t = field.state[i * kStride + 0u];
            const float fuel = field.state[i * kStride + 1u];
            const float charred = field.state[i * kStride + 2u];
            const float wet = field.state[i * kStride + 3u];
            const float molten = field.state[i * kStride + 5u];
            melt_sum += molten;
            stats_.max_melt = std::max(stats_.max_melt, molten);
            if (molten >= 1.0f) stats_.molten_elements += 1u;
            else if (molten > 0.0f) stats_.melting_elements += 1u;
            temperature_sum += t;
            char_sum += charred;
            fuel_sum += std::max(0.0f, fuel);
            moisture_sum += wet;
            stats_.max_temperature = std::max(stats_.max_temperature, t);
            stats_.max_char = std::max(stats_.max_char, charred);
            stats_.max_moisture = std::max(stats_.max_moisture, wet);
            if (wet > kWetElementThreshold) stats_.wet_elements += 1u;
            // ★ "Burning" now also requires the surface to be dry. A wet element
            // with leftover char would otherwise be counted as still burning
            // while the shader has it firmly suppressed — the panel must not
            // disagree with the solver about whether something is on fire.
            if (fuel > 0.0f && charred > 0.0f && wet <= kWetElementThreshold) {
                stats_.burning_count += 1u;
            }
            ++counted;
        }
        scatterCharMask(field, readback_scale_);

        // ── Phase 6a: can geometry displacement actually reach this melt? ─────
        // Counted after the scatter so it measures the DILATED lookup, i.e. what
        // a vertex query will really see. "No UV" is a separate counter from "low
        // coverage": the first means displacement is impossible for this object,
        // the second means the unwrap leaves holes — different fixes.
        if (field.mask_resolution <= 0) {
            stats_.lookup_fields_no_uv += 1u;
        } else {
            stats_.lookup_texels_total +=
                static_cast<uint32_t>(field.melt_covered.size());
            uint32_t covered = 0u;
            for (uint8_t c : field.melt_covered) covered += (c != 0u) ? 1u : 0u;
            stats_.lookup_texels_covered += covered;
        }
    }

    if (counted > 0u) {
        stats_.mean_temperature = static_cast<float>(temperature_sum / counted);
        stats_.mean_char = static_cast<float>(char_sum / counted);
        stats_.mean_moisture = static_cast<float>(moisture_sum / counted);
        stats_.mean_melt = static_cast<float>(melt_sum / counted);
    }
    stats_.fuel_remaining = static_cast<float>(fuel_sum);
    stats_.readback_ms = elapsedMs(start, Clock::now());
    readback_requested_ = false;
}

void MaterialStateFieldSystem::clearField(MaterialStateField& field) {
    std::fill(field.state.begin(), field.state.end(), 0.0f);
    for (std::size_t i = 0; i < field.elementCount(); ++i) {
        // re-seed fuel on the next step
        field.state[i * MaterialStateField::kStateStride + 1u] = -1.0f;
    }
    field.centers_dirty = true;

    // The mask is what the RENDERER consumes, and it is only rebuilt by
    // scatterCharMask on a readback. Zeroing the state alone would leave the
    // object visibly burnt until the next readback happened to land — clearing
    // has to be immediate to read as "clear". Bump the revision or the bridge
    // skips the re-upload as unchanged.
    if (!field.char_mask.empty()) {
        std::fill(field.char_mask.begin(), field.char_mask.end(), uint8_t{0});
        field.mask_revision += 1u;
    }
    // ★ The melt lookup is cleared with the same urgency and for the same reason:
    // it is read by geometry, and leaving it hot would keep the surface displaced
    // after a Clear Damage until some later readback happened to land. `covered`
    // is NOT cleared — it describes the UV layout, not the damage.
    std::fill(field.melt_texel.begin(), field.melt_texel.end(), 0.0f);
}

bool MaterialStateFieldSystem::clearField(const std::string& object_key) {
    auto it = fields_.find(object_key);
    if (it == fields_.end()) return false;
    clearField(it->second);
    return true;
}

void MaterialStateFieldSystem::resetState() {
    for (auto& entry : fields_) {
        clearField(entry.second);
    }
    // A parked snapshot outliving a reset would re-apply the damage the reset
    // just wiped, the moment the field was next rebuilt.
    pending_restore_.clear();
    stats_ = MaterialStateFieldStats{};
}

void MaterialStateFieldSystem::beginSyncPass() {
    synced_this_pass_.clear();
}

void MaterialStateFieldSystem::endSyncPass(SimulationComputeContext& compute) {
    if (fields_.empty()) return;
    for (auto it = fields_.begin(); it != fields_.end();) {
        const bool kept = std::find(synced_this_pass_.begin(),
                                    synced_this_pass_.end(),
                                    it->first) != synced_this_pass_.end();
        if (kept) { ++it; continue; }
        if (it->second.gpu_centers.valid()) compute.destroyBuffer(it->second.gpu_centers);
        if (it->second.gpu_state.valid()) compute.destroyBuffer(it->second.gpu_state);
        it = fields_.erase(it);
    }
}

void MaterialStateFieldSystem::release(SimulationComputeContext& compute) {
    for (auto& entry : fields_) {
        if (entry.second.gpu_centers.valid()) compute.destroyBuffer(entry.second.gpu_centers);
        if (entry.second.gpu_state.valid()) compute.destroyBuffer(entry.second.gpu_state);
    }
    if (gpu_thermal_sources_.valid()) {
        compute.destroyBuffer(gpu_thermal_sources_);
        gpu_thermal_sources_ = {};
    }
    if (gpu_ambient_zones_.valid()) {
        compute.destroyBuffer(gpu_ambient_zones_);
        gpu_ambient_zones_ = {};
    }
    fields_.clear();
    pending_restore_.clear();
    synced_this_pass_.clear();
    stats_ = MaterialStateFieldStats{};
}

} // namespace RayTrophiSim
