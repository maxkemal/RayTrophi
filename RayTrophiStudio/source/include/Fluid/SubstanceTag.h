/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          SubstanceTag.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * The identity a parcel of liquid carries: which SUBSTANCE it is.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace RayTrophiSim {
namespace Fluid {

// ═══════════════════════════════════════════════════════════════════════════
// SUBSTANCE IDENTITY BELONGS TO THE MATERIAL, NOT TO THE EMITTER.
// ═══════════════════════════════════════════════════════════════════════════
// ★★★ The discriminating question is what happens at a surface point where two
// streams have mixed: there is no "which emitter" there. An emitter is a place
// where liquid is BORN, not what the liquid IS. Two emitters pouring the same
// chocolate must merge seamlessly; two pouring different chocolates must mix —
// and only a per-parcel identity that survives advection can express both.
//
// So the emitter names a substance, the particle carries the tag, and look and
// physics are looked up from the substance. The tag rides the particle through
// advection, compaction and reseed inheritance, which is what makes a mixture a
// real field rather than a rendering trick.
inline uint32_t substanceTag(const std::string& text) {
    if (text.empty()) return 0u;
    uint32_t hash = 2166136261u;                       // FNV-1a
    for (unsigned char c : text) hash = (hash ^ c) * 16777619u;
    // ★ 0 is RESERVED for "untagged", so a name that happens to hash to it must
    // not silently become untagged liquid. Astronomically unlikely and trivially
    // cheap to exclude — and the failure it prevents is one substance in a scene
    // behaving like it has no identity, which nobody would trace back to a hash.
    return hash == 0u ? 1u : hash;
}

// "This parcel was never assigned a substance." A legitimate state: plain water
// from a domain fill, liquid from a scripted spawn, everything authored before
// substances existed.
//
// ★ NOT a substance in its own right. Consumers must treat it as "use the
// domain's single material" rather than looking it up in the table and finding
// nothing — the difference is a fallback that renders versus a surface that
// silently loses its material.
constexpr uint32_t kSubstanceUntagged = 0u;

// How many distinct substances one domain may bind materials for.
//
// ★ A cap rather than an open table, because the composition gather accumulates
// a weight PER DISTINCT SUBSTANCE in every cell. Unbounded, that is a per-cell
// search whose cost stays invisible until a scene has dozens of substances and
// then shows up as "the surface rebuild got slow" with no obvious cause. Eight
// is far past any real recipe and keeps the per-cell accumulator on the stack.
constexpr std::size_t kMaxFluidSubstanceMaterials = 8;

// Per-substance render routing. Inherit keeps old scenes on the domain-wide
// visualization mode; the explicit values allow one APIC domain to feed both
// the reconstructed level set and discrete splat spheres.
enum class SubstanceRepresentation : uint8_t {
    Inherit = 0,
    Splat   = 1,
    SurfaceSDF = 2
};

// ═══════════════════════════════════════════════════════════════════════════
// PHASE — a state of MATTER, and a separate axis from representation.
// ═══════════════════════════════════════════════════════════════════════════
// Solid parcels are rasterized into the grid's solid mask every step, so the
// existing no-slip boundary makes the liquid flow AROUND them and cling to
// them. The solver does not care where a solid cell came from, which is why
// this costs one producer and no new boundary condition.
//
// ★★★ NOT THE SAME KNOB AS SubstanceRepresentation, even though "splat" and
// "chunk" describe the same picture. Representation answers HOW IT IS DRAWN;
// phase answers WHAT IT IS. Fusing them would make the flow change when
// somebody switched a render mode — the exact failure the substance viscosity
// gather is documented to avoid (see buildSubstanceViscosityField). A solid
// can be drawn as splat spheres or reconstructed into the isosurface, and a
// liquid can be drawn either way too.
//
// ★★ What this deliberately is NOT: rigid dynamics. A solid parcel is an
// OBSTACLE with mass and velocity, not a body with orientation or cohesion —
// a pile spreads under load because nothing holds the parcels together. The
// value here is that the phase TRANSITION is continuous (melting already
// exists in both directions as a material state), not that it competes with
// Jolt. Dragged, cohesive clusters belong to Jolt; writing a second rigid
// solver inside the fluid step is how you get two answers to one question.
enum class SubstancePhase : uint8_t {
    Liquid = 0,
    Solid  = 1
};

} // namespace Fluid
} // namespace RayTrophiSim
