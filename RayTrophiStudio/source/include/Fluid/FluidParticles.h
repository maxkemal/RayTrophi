/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidParticles.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * APIC (Affine Particle-In-Cell) particle storage for liquid simulation.
 *
 * Layout is SoA so future Vulkan compute upload is a flat memcpy. Particles
 * carry an affine velocity matrix C (Jiang et al. 2015) instead of plain
 * velocity, which preserves angular momentum during P2G/G2P without the
 * dissipation of PIC or the noise of FLIP.
 *
 * This is the source of truth for liquid mass. The MAC grid (FluidGrid) is
 * a transient scratchpad used per step for pressure projection. The narrow
 * band level set (FluidLevelSet, Phase 2) is derived from these particles
 * for rendering — not advected.
 */

#pragma once

#include "../Vec3.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>

namespace RayTrophiSim {
namespace Fluid {

// Affine velocity matrix (3x3) stored as three column vectors. Used by APIC
// to weight particle-grid transfers; degenerates to plain PIC when zero.
struct AffineC {
    Vec3 col0;
    Vec3 col1;
    Vec3 col2;

    AffineC() : col0(0,0,0), col1(0,0,0), col2(0,0,0) {}
};

class FluidParticles {
public:
    std::vector<Vec3>     position;   // world space
    std::vector<Vec3>     velocity;   // world space, m/s
    std::vector<AffineC>  affine;     // APIC velocity gradient
    std::vector<uint32_t> flags;      // reserved (bit 0 = sleeping, etc.)
    // Remaining material mass relative to the authored particle mass.  This
    // is deliberately kept separate from position/velocity so combustion can
    // evaporate or burn a liquid without changing APIC momentum until the
    // hysteresis-based lifecycle pass safely compacts particles.
    std::vector<float>    mass_fraction;
    std::vector<float>    temperature;
    std::vector<float>    combustible_fraction;
    std::vector<uint32_t> substance_tag;
    // Granular-only state. Liquid particles keep identity deformation and zero
    // hardening; the Vulkan constitutive pass consumes these flat SoA arrays.
    std::vector<Vec3> granular_deformation_col0;
    std::vector<Vec3> granular_deformation_col1;
    std::vector<Vec3> granular_deformation_col2;
    std::vector<float> granular_plastic_volume;
    std::vector<float> granular_hardening;
    std::vector<uint32_t> granular_material_flags;
    // Symmetric Cauchy stress: diagonal (xx,yy,zz) and shear (xy,xz,yz).
    // Stored explicitly for the rate-form Drucker-Prager MPM update.
    std::vector<Vec3> granular_stress_diag;
    std::vector<Vec3> granular_stress_shear;
    std::vector<float> granular_yield_value;
    std::vector<float> granular_plastic_increment;
    std::vector<float> granular_damage;
    // Maximum irreversible Rankine bond-opening strain. Kept separate from
    // frictional plastic strain: dry granular rearrangement may harden/flow
    // without falsely consuming the cohesive fracture budget. Deliberately
    // not integrated over solver substeps.
    std::vector<float> granular_fracture_history;
    // ── Material coordinate (UVW) ────────────────────────────────────────────
    // Where this parcel of liquid WAS BORN, in world units. This is a texture
    // anchor, and it is Lagrangian: it rides the particle and is never
    // integrated, so it cannot drift and needs no re-normalisation.
    //
    // ★ Why birth POSITION and not a normalised [0,1] domain coordinate: at the
    // first frame of a resting body uvw == position exactly, so the rendered
    // result is IDENTICAL to the world-anchored tri-planar projection it
    // replaces, and the material's uv_scale keeps meaning "world units per
    // tile". Nothing to re-author, and no scene changes appearance on the frame
    // this shipped. What changes is only what should: once the liquid MOVES,
    // the texture goes with it instead of the body sliding through a stationary
    // projection.
    //
    // ★ It also fixes a carried container for free: a glass moved across the
    // scene carries its contents' uvw, because the coordinate is attached to the
    // material and not to the world or to the domain.
    //
    // ★★ THE LIMIT THIS SOLVES, and it is inherent to material coordinates
    // rather than to any implementation of them: a Lagrangian map STRETCHES
    // with the flow, without bound. A long pour or a melting body smears the
    // texture along the stretch direction, and after enough deformation the
    // mapping is chaotic. No amount of resolution fixes it — the map itself is
    // the thing degrading.
    //
    // TWO GENERATIONS, RESET ON STAGGERED PHASES. Each is periodically reset to
    // the identity (uvw = position); the phases are half a period apart, and the
    // renderer blends them with triangular weights. No generation is ever older
    // than one period, so accumulated distortion is bounded by construction.
    //
    // ★★★ THE WEIGHT IS ZERO EXACTLY WHEN A GENERATION IS RESET. That is what
    // makes the reset invisible: the discontinuity is multiplied by zero. A
    // scheme that reset at full weight would pop once per period, which is worse
    // than the stretching it cures because it is rhythmic and draws the eye.
    //
    // ★★ Why this is affordable, and why it is NOT "re-parametrise the surface
    // every frame": there is no surface parametrisation here at all. The
    // coordinate is volumetric and belongs to the MATERIAL, so bounding its
    // distortion costs one extra vector and a periodic assignment — no UV
    // layout, no temporal-coherence problem from a re-cut atlas.
    //
    // ★ And the blend is only well-posed because these are DISPLACEMENTS
    // downstream (see buildMaterialCoordinateGrid): blending two absolute
    // coordinates would average two positions and flatten the gradient, which is
    // the same failure the extrapolation sweep once had. A freshly reset
    // generation has displacement exactly zero, i.e. perfect identity quality,
    // and it earns weight as it deforms.
    std::vector<Vec3>     uvw;     // generation A
    std::vector<Vec3>     uvw_b;   // generation B, reset half a period later

    // Steps since the epoch, advanced by the solver. Drives both resets.
    // ★ Counts SOLVER STEPS, not timeline frames: the coordinate degrades with
    // simulated deformation, and that is what steps measure. Tying it to the
    // playhead would make the refresh rate depend on the frame rate.
    uint32_t              uvw_step = 0;
    // Steps between resets of the SAME generation. Larger = the texture is
    // carried further before being refreshed (more stretch, less blending);
    // smaller = crisper but more of the "two patterns crossfading" look.
    int                   uvw_refresh_period = 240;

    // Triangular hat over one period, peaking mid-life. The two generations are
    // half a period out of phase, so these ALWAYS sum to exactly 1 — the
    // coordinate is a weighted average, never brightened or dimmed by the
    // schedule.
    void materialCoordWeights(float& w_a, float& w_b) const {
        const int period = uvw_refresh_period > 1 ? uvw_refresh_period : 1;
        const int half = period / 2;
        const float inv = 1.0f / static_cast<float>(period);
        const float age_a = static_cast<float>(uvw_step % period) * inv;
        const float age_b = static_cast<float>((uvw_step + half) % period) * inv;
        w_a = 1.0f - std::fabs(2.0f * age_a - 1.0f);
        w_b = 1.0f - std::fabs(2.0f * age_b - 1.0f);
        // Guard the odd-period case, where the two hats can miss unity by a
        // fraction of a step. Normalising is cheaper than constraining the
        // period to be even, and it keeps the invariant true by construction
        // rather than by arithmetic that happens to work out.
        const float sum = w_a + w_b;
        if (sum > 1e-6f) { w_a /= sum; w_b /= sum; }
        else             { w_a = 1.0f; w_b = 0.0f; }
    }

    // Advance the schedule and reset whichever generation is due. Called once
    // per solver step, AFTER advection — a generation reset to the identity must
    // describe where the liquid is now, not where it was when the step began.
    void advanceMaterialCoordinates() {
        ++uvw_step;
        const int period = uvw_refresh_period > 1 ? uvw_refresh_period : 1;
        const int half = period / 2;
        const std::size_t n = position.size();
        if (uvw_b.size() != n) uvw_b.assign(uvw.begin(), uvw.end());
        if (uvw_step % period == 0u) {
            for (std::size_t i = 0; i < n && i < uvw.size(); ++i) uvw[i] = position[i];
        }
        if (half > 0 && (uvw_step + static_cast<uint32_t>(half)) % period == 0u) {
            for (std::size_t i = 0; i < n && i < uvw_b.size(); ++i) uvw_b[i] = position[i];
        }
    }

    void clear() {
        position.clear();
        velocity.clear();
        affine.clear();
        flags.clear();
        mass_fraction.clear();
        temperature.clear();
        combustible_fraction.clear();
        substance_tag.clear();
        granular_deformation_col0.clear(); granular_deformation_col1.clear();
        granular_deformation_col2.clear(); granular_plastic_volume.clear();
        granular_hardening.clear(); granular_material_flags.clear();
        granular_stress_diag.clear(); granular_stress_shear.clear();
        granular_yield_value.clear(); granular_plastic_increment.clear();
        granular_damage.clear(); granular_fracture_history.clear();
        uvw.clear();
        uvw_b.clear();
        uvw_step = 0;
    }

    size_t size() const { return position.size(); }
    bool   empty() const { return position.empty(); }

    void ensureGranularStateSize() {
        const size_t n = position.size();
        granular_deformation_col0.resize(n, Vec3(1,0,0));
        granular_deformation_col1.resize(n, Vec3(0,1,0));
        granular_deformation_col2.resize(n, Vec3(0,0,1));
        granular_plastic_volume.resize(n, 1.0f);
        granular_hardening.resize(n, 0.0f);
        granular_material_flags.resize(n, 0u);
        granular_stress_diag.resize(n, Vec3(0,0,0));
        granular_stress_shear.resize(n, Vec3(0,0,0));
        granular_yield_value.resize(n, 0.0f);
        granular_plastic_increment.resize(n, 0.0f);
        granular_damage.resize(n, 0.0f);
        granular_fracture_history.resize(n, 0.0f);
    }

    void reserve(size_t n) {
        position.reserve(n);
        velocity.reserve(n);
        affine.reserve(n);
        flags.reserve(n);
        mass_fraction.reserve(n);
        temperature.reserve(n);
        combustible_fraction.reserve(n);
        substance_tag.reserve(n);
        granular_deformation_col0.reserve(n); granular_deformation_col1.reserve(n);
        granular_deformation_col2.reserve(n); granular_plastic_volume.reserve(n);
        granular_hardening.reserve(n); granular_material_flags.reserve(n);
        granular_stress_diag.reserve(n); granular_stress_shear.reserve(n);
        granular_yield_value.reserve(n); granular_plastic_increment.reserve(n);
        granular_damage.reserve(n); granular_fracture_history.reserve(n);
        uvw.reserve(n);
        uvw_b.reserve(n);
    }

    // `birth_uvw` defaults to the birth position, which is the correct seed for
    // every genuinely NEW parcel of liquid — an emitter, an initial fill, a
    // molten mass transfer. Only particles that are a CONTINUATION of existing
    // liquid pass it explicitly (the reseed top-up), and they must, because a
    // reseeded particle is not new material: it is a resampling of material that
    // already has a coordinate. Defaulting there would stamp a fresh coordinate
    // into the middle of an existing body and tear the texture exactly where
    // reseeding is busiest — in splashes.
    // ★ `birth_uvw_b` is the SECOND generation's seed and defaults to the first.
    // A reseed continuation must pass both: the two generations were reset at
    // different times, so a donor's A and B coordinates genuinely differ, and
    // copying A into both would silently collapse the pair into one generation
    // for every reseeded particle — i.e. the stretch cure would quietly stop
    // working exactly in splashes, where reseeding is busiest and stretch is
    // worst.
    void emit(const Vec3& p, const Vec3& v, float temp = 0.0f,
              float combustible = 0.0f, uint32_t material = 0u,
              const Vec3* birth_uvw = nullptr,
              const Vec3* birth_uvw_b = nullptr) {
        position.push_back(p);
        velocity.push_back(v);
        affine.emplace_back();
        flags.push_back(0u);
        mass_fraction.push_back(1.0f);
        temperature.push_back(temp);
        combustible_fraction.push_back(combustible);
        substance_tag.push_back(material);
        granular_deformation_col0.emplace_back(1,0,0);
        granular_deformation_col1.emplace_back(0,1,0);
        granular_deformation_col2.emplace_back(0,0,1);
        granular_plastic_volume.push_back(1.0f);
        granular_hardening.push_back(0.0f);
        granular_material_flags.push_back(0u);
        granular_stress_diag.emplace_back(0,0,0);
        granular_stress_shear.emplace_back(0,0,0);
        granular_yield_value.push_back(0.0f);
        granular_plastic_increment.push_back(0.0f);
        granular_damage.push_back(0.0f);
        granular_fracture_history.push_back(0.0f);
        // Genuinely new material is the identity in both generations, which is
        // what the null defaults give.
        uvw.push_back(birth_uvw ? *birth_uvw : p);
        uvw_b.push_back(birth_uvw_b ? *birth_uvw_b
                                    : (birth_uvw ? *birth_uvw : p));
    }

    // Remove particle i in O(1) via swap-with-back. Order is not preserved.
    void removeSwap(size_t i) {
        size_t last = position.size() - 1;
        if (i != last) {
            position[i] = position[last];
            velocity[i] = velocity[last];
            affine[i]   = affine[last];
            flags[i]    = flags[last];
            mass_fraction[i] = mass_fraction[last];
            temperature[i] = temperature[last];
            combustible_fraction[i] = combustible_fraction[last];
            substance_tag[i] = substance_tag[last];
            granular_deformation_col0[i] = granular_deformation_col0[last];
            granular_deformation_col1[i] = granular_deformation_col1[last];
            granular_deformation_col2[i] = granular_deformation_col2[last];
            granular_plastic_volume[i] = granular_plastic_volume[last];
            granular_hardening[i] = granular_hardening[last];
            granular_material_flags[i] = granular_material_flags[last];
            granular_stress_diag[i] = granular_stress_diag[last];
            granular_stress_shear[i] = granular_stress_shear[last];
            granular_yield_value[i] = granular_yield_value[last];
            granular_plastic_increment[i] = granular_plastic_increment[last];
            granular_damage[i] = granular_damage[last];
            granular_fracture_history[i] = granular_fracture_history[last];
            uvw[i] = uvw[last];
            if (i < uvw_b.size() && last < uvw_b.size()) uvw_b[i] = uvw_b[last];
        }
        position.pop_back();
        velocity.pop_back();
        affine.pop_back();
        flags.pop_back();
        mass_fraction.pop_back();
        temperature.pop_back();
        combustible_fraction.pop_back();
        substance_tag.pop_back();
        granular_deformation_col0.pop_back(); granular_deformation_col1.pop_back();
        granular_deformation_col2.pop_back(); granular_plastic_volume.pop_back();
        granular_hardening.pop_back(); granular_material_flags.pop_back();
        granular_stress_diag.pop_back(); granular_stress_shear.pop_back();
        granular_yield_value.pop_back(); granular_plastic_increment.pop_back();
        granular_damage.pop_back();
        granular_fracture_history.pop_back();
        uvw.pop_back();
        if (!uvw_b.empty()) uvw_b.pop_back();
    }
};

} // namespace Fluid
} // namespace RayTrophiSim
