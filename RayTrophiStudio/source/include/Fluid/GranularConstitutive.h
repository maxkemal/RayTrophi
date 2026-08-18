#pragma once

#include "../Vec3.h"
#include "FluidParticles.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

// =============================================================================
// Granular (Drucker-Prager + Rankine damage) constitutive core — CPU reference.
//
// ★★★ THIS FILE IS THE REFERENCE, sim_fluid_granular_stress_update.comp IS THE
// PORT. Both must answer identically for the same inputs; where they differ,
// the shader is wrong. Keep the two edited in lockstep — every clamp, every
// epsilon, every flag bit. The push-constant block over there and
// StressUpdateParams here are deliberately field-for-field the same.
//
// What used to live in this file was a scalar `evaluate()` that nothing ever
// called: a yield-surface sketch with no deformation gradient, no return map
// and no damage. It has been replaced rather than deleted, because a CPU
// reference is exactly what the roadmap's Faz 0/2/3 acceptance gates need and
// what its absence made impossible to run.
//
// ★ Before this existed the CPU branch was worse than missing. A granular
// domain with the GPU backend unavailable fell through to the incompressible
// LIQUID solver: same particles, same panel, silently a different material.
// A resting sand pile became a boiling puddle and nothing reported a fallback.
// =============================================================================

namespace RayTrophiSim::Fluid::Granular {

// ── Material constants shared by BOTH backends and by the step planner ───────
// They live here, with the constitutive law, because that is what they describe.
// GranularGpuDispatch.h includes this header and uses the same values, so the
// CPU reference and the Vulkan push constants cannot drift apart.

// Bulk density the granular stress divergence is expressed against. The stress
// P2G divides by this same number on both paths, so the two must move together.
constexpr float kGranularDensity = 1600.0f;

// Largest elastic strain a particle's deformation gradient may STORE. Strain a
// material cannot elastically hold is permanent -- it is compaction, and it
// lives in the particle positions, which already carry the plastic motion.
//
// ★ Without this cap the back-projection at the end of the stress update asks
// the gradient to store stress/E, which at low stiffness exceeds 1: I + eps goes
// negative-determinant, the next step declares the particle invalid, and its
// stress is dumped to zero in one step. Measured on a 1 kPa pile: 22 such
// discharges in 240 steps, seen as small bursts escaping the surface.
//
// 0.5 keeps det(I + eps) >= 0.125 for any strain shape (Frobenius bounds every
// eigenvalue), which is what makes the downstream determinant floor unreachable
// by that route.
constexpr float kGranularMaxStoredStrain = 0.5f;

// TARGET |dt*C| used to SIZE the subcycle (elasticStepInfo). (I + dt*C) is a
// first-order exponential map; past roughly a quarter it stops being a rotation
// plus small strain and starts manufacturing strain.
constexpr float kGranularMaxStrainIncrement = 0.20f;

// HARD CEILING the constitutive update clamps to when the subcycle could not be
// granted. Deliberately above the sizing target: the planner aims for 0.20 and
// this only catches what it could not cover (an impact landing inside a frame
// the host measurement could not see). Frobenius <= 0.25 keeps every eigenvalue
// of I + dC at modulus >= 0.75, so the trial gradient can never invert.
constexpr float kGranularStrainIncrementCeiling = 0.25f;

// Stiffness needed per unit of carried load, expressed as the reciprocal of the
// largest elastic strain the corotational predictor is still meaningful at.
// 10 means "the bottom layer may compress by 10 percent" -- already the outer
// edge of small-strain, deliberately generous so the shipped Sand preset
// (2e5 Pa) clears it with margin on a metre-deep pile and only genuinely
// unphysical stiffnesses trip the gate.
//
// ★ The threshold is a STRAIN, not a stiffness. Naming it in strain is what
// keeps it voxel-size and frame-rate independent; a Pa figure here would have to
// be retuned for every scene and would quietly stop meaning anything.
constexpr float kGranularStiffnessLoadRatio = 10.0f;

// ── Thermal / burn softening of the skeleton ─────────────────────────────────
// One scalar per particle, 1 = cold and intact, 0 = fully softened. It is
// derived on the host (where `temperature` and `mass_fraction` live) and handed
// to both backends as data, so the kernels do not need a thermal model.
//
// ★★ COHESION AND STIFFNESS DO NOT FALL ON THE SAME CURVE, and that difference
// is what makes a melt read as a melt. Bond strength is squared, so cohesion and
// the tensile cut-off collapse well before the modulus does: the block stops
// holding its shape first and only then goes soft. Falling together instead
// produces a body that sags uniformly while still resisting being pulled apart,
// which reads as rubber, not as melting.
//
// ★ tensile_cutoff MUST follow cohesion. They are the same bonds; softening one
// and not the other yields a melting block that still refuses to tear.
struct SofteningParams {
    float softening_temperature = 0.0f; // K; 0 disables the whole path
    float softening_range = 40.0f;      // K, transition width
    float residual_strength = 0.05f;    // floor once fully softened
    // ★★★ Bond strength is NOT a copy of the stiffness curve, and assuming it
    // was is why burning plastic crumbled instead of melting.
    //
    // Stiffness falls monotonically with temperature — that part was right. Bond
    // strength does not: a thermoplastic goes TACKY on the way to molten. Around
    // the glass transition the grains stop being rigid and start sticking, the
    // pile fuses into one body, and only above the melt point does the yield
    // stress vanish and viscosity take over. So cohesion traces a HUMP, not a
    // ramp.
    //
    // ★ The two failures look almost identical and that is the trap: both make
    // the pile lose its shape. The discriminator is whether it stays ONE BODY.
    // Bonds that only fall give grains that separate and scatter — the block
    // CRUMBLES. Bonds that rise first give a body that slumps and fuses — which
    // is what melting actually looks like.
    //
    // 1.0 = no hump (pure ramp, the old behaviour). Thermoplastics want 2-4.
    float tack_peak = 1.0f;
};

// Cohesion/tensile multiplier. `molten` is 0 cold .. 1 fully molten.
//
// mix(1, residual, molten) is the ramp; 4*m*(1-m) is a parabola peaking at 1.0
// halfway through the transition, so `tack_peak` is read directly as "how many
// times the authored cohesion at peak tackiness".
inline float bondScaleFromMolten(float molten, const SofteningParams& sp) {
    const float m = std::clamp(molten, 0.0f, 1.0f);
    const float res = std::clamp(sp.residual_strength, 0.0f, 1.0f);
    const float ramp = 1.0f + (res - 1.0f) * m;
    const float hump = 4.0f * m * (1.0f - m);
    const float peak = std::max(sp.tack_peak, 0.0f);
    return std::max(ramp + (peak - 1.0f) * hump, 0.0f);
}

// smoothstep-based transition; both endpoints have zero derivative, so a
// particle drifting across the threshold does not produce a stiffness step.
inline float softeningFactor(float temperature, float mass_fraction,
                             const SofteningParams& sp) {
    if (!(sp.softening_temperature > 0.0f)) return 1.0f;
    const float mass = std::isfinite(mass_fraction)
        ? std::clamp(mass_fraction, 0.0f, 1.0f) : 1.0f;
    float thermal = 1.0f;
    if (std::isfinite(temperature)) {
        const float range = std::max(sp.softening_range, 1.0f);
        const float x = std::clamp(
            (temperature - (sp.softening_temperature - 0.5f * range)) / range,
            0.0f, 1.0f);
        thermal = 1.0f - (x * x * (3.0f - 2.0f * x));
    }
    // ★ Remaining mass is the second input and it costs no extra dial: a
    // charring foam loses its skeleton roughly with the mass it has burned off.
    // The burn path already maintains mass_fraction, so this half was free.
    const float intact = std::clamp(thermal * mass, 0.0f, 1.0f);
    return std::clamp(sp.residual_strength +
                      (1.0f - sp.residual_strength) * intact, 0.0f, 1.0f);
}

// Refresh every particle's softening scalar. Called ONCE PER FRAME by the
// simulation tick, before the granular state is uploaded, so both backends read
// the same field: the device consumes it as a storage buffer and the CPU
// reference indexes the same array.
//
// ★ Once per frame, not once per elastic substep. Temperature does not change
// inside the subcycle, and re-deriving there would make the melt rate depend on
// the substep count -- the same substep-dependence trap the Rankine bond
// history had to be rewritten to avoid.
// ★★ BOTH scalars are derived here and uploaded as plain per-particle arrays.
// The shader multiplies; it does not re-evaluate the curves. A second copy of a
// transition curve in GLSL is a copy that drifts silently the first time either
// side is tuned, and the drift shows up as "the CPU and GPU melt differently"
// with nothing to point at.
inline void updateSoftening(FluidParticles& particles, const SofteningParams& sp) {
    const std::size_t n = particles.size();
    if (particles.granular_softening.size() != n) return;
    const bool has_bond = particles.granular_bond_scale.size() == n;
    const bool has_temperature = particles.temperature.size() == n;
    const bool has_mass = particles.mass_fraction.size() == n;
    if (!(sp.softening_temperature > 0.0f)) {
        // Disabled must be a BIT-EXACT no-op: 1.0 in both arrays reproduces the
        // pre-softening numbers exactly. Verified against the standalone run.
        std::fill(particles.granular_softening.begin(),
                  particles.granular_softening.end(), 1.0f);
        if (has_bond)
            std::fill(particles.granular_bond_scale.begin(),
                      particles.granular_bond_scale.end(), 1.0f);
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const float temp = has_temperature ? particles.temperature[i] : 0.0f;
        const float mass = has_mass ? particles.mass_fraction[i] : 1.0f;
        const float soft = softeningFactor(temp, mass, sp);
        particles.granular_softening[i] = soft;
        if (!has_bond) continue;
        // `molten` is derived from the SAME softening scalar, not re-derived
        // from temperature, so the stiffness and bond curves can never disagree
        // about where the transition is. residual_strength is the floor of
        // softeningFactor, so undo it to recover the 0..1 progress.
        const float res = std::clamp(sp.residual_strength, 0.0f, 1.0f);
        const float span = std::max(1.0f - res, 1e-4f);
        const float molten = std::clamp(1.0f - (soft - res) / span, 0.0f, 1.0f);
        // Mass loss still removes bonds outright: a grain that has burned away
        // cannot be tacky. This keeps char (mass gone) distinct from melt
        // (mass intact, bonds tacky) without a second dial.
        particles.granular_bond_scale[i] =
            bondScaleFromMolten(molten, sp) * std::clamp(mass, 0.0f, 1.0f);
    }
}

// Authored material, shared with the GPU dispatch path.
struct Parameters {
    float friction_angle_radians = 0.61086524f; // 35 degrees
    float cohesion = 0.0f;
    float dilatancy = 0.0f;
    float hardening = 0.0f;
    float tensile_cutoff = 0.0f;
    float detach_pressure = 1.0e-4f;
};

// Field-for-field mirror of the shader's push constant block. Angles arrive
// already converted to tangents so both paths clamp the angle identically.
struct StressUpdateParams {
    float dt = 0.0f;
    float young_modulus = 2.0e5f;
    float poisson_ratio = 0.25f;
    float friction_tangent = 0.7002075f;
    float cohesion = 0.0f;
    float dilatancy_tangent = 0.0874887f;
    float tensile_cutoff = 0.0f;
    float fracture_strain = 0.04f;
    float damage_rate = 6.0f;
    float healing_rate = 0.0f;
    bool  rebonding = false;
    float hardening_coefficient = 0.0f;
    float max_stored_strain = kGranularMaxStoredStrain;
};

// Flag bits. Shared vocabulary with the shader and with the stats/IPC readout.
enum : uint32_t {
    kFlagYielded          = 1u,
    kFlagDetached         = 2u,
    kFlagInvalid          = 4u,
    kFlagSleeping         = 8u,   // written by the settle stage
    kFlagStrainLimited    = 16u,  // dt*C had to be clamped: step was coarse
    kFlagCompactionCapped = 32u,  // stored strain capped: compaction went plastic
};

// ── Column-major 3x3, matching GLSL mat3 exactly ─────────────────────────────
// GLSL's mat3(c0, c1, c2) builds COLUMNS and A[i] is column i, so the port is
// only readable if the reference uses the same convention. Do not "tidy" this
// into row-major; the transposes in the polar decomposition would silently
// invert and the stress would come out rotated.
struct Mat3 {
    Vec3 c[3];
    Mat3() : c{ Vec3(0,0,0), Vec3(0,0,0), Vec3(0,0,0) } {}
    Mat3(const Vec3& c0, const Vec3& c1, const Vec3& c2) : c{ c0, c1, c2 } {}
    static Mat3 identity() { return Mat3(Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)); }
    // A[i][j] in GLSL terms: column i, row j.
    float at(int i, int j) const {
        const Vec3& v = c[i];
        return j == 0 ? v.x : (j == 1 ? v.y : v.z);
    }
};

inline Mat3 operator*(const Mat3& a, float s) {
    return Mat3(a.c[0] * s, a.c[1] * s, a.c[2] * s);
}
inline Mat3 operator+(const Mat3& a, const Mat3& b) {
    return Mat3(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
}
inline Mat3 operator-(const Mat3& a, const Mat3& b) {
    return Mat3(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
}
inline Vec3 operator*(const Mat3& a, const Vec3& v) {
    return a.c[0] * v.x + a.c[1] * v.y + a.c[2] * v.z;
}
inline Mat3 operator*(const Mat3& a, const Mat3& b) {
    return Mat3(a * b.c[0], a * b.c[1], a * b.c[2]);
}
inline Mat3 transpose(const Mat3& a) {
    return Mat3(Vec3(a.at(0,0), a.at(1,0), a.at(2,0)),
                Vec3(a.at(0,1), a.at(1,1), a.at(2,1)),
                Vec3(a.at(0,2), a.at(1,2), a.at(2,2)));
}
inline float determinant(const Mat3& a) {
    return a.c[0].dot(a.c[1].cross(a.c[2]));
}
inline Mat3 inverse(const Mat3& a) {
    const Vec3 r0 = a.c[1].cross(a.c[2]);
    const Vec3 r1 = a.c[2].cross(a.c[0]);
    const Vec3 r2 = a.c[0].cross(a.c[1]);
    const float det = a.c[0].dot(r0);
    if (!(std::fabs(det) > 1.0e-20f)) return Mat3::identity();
    const float inv = 1.0f / det;
    // rows of the adjugate become columns after the implicit transpose
    return Mat3(Vec3(r0.x, r1.x, r2.x) * inv,
                Vec3(r0.y, r1.y, r2.y) * inv,
                Vec3(r0.z, r1.z, r2.z) * inv);
}
// GLSL outerProduct(u, v): column j is u * v[j].
inline Mat3 outerProduct(const Vec3& u, const Vec3& v) {
    return Mat3(u * v.x, u * v.y, u * v.z);
}
inline float trace(const Mat3& a) { return a.at(0,0) + a.at(1,1) + a.at(2,2); }
inline bool isFiniteMat(const Mat3& a) {
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(a.c[i].x) || !std::isfinite(a.c[i].y) ||
            !std::isfinite(a.c[i].z)) return false;
    }
    return true;
}
inline float frobenius(const Mat3& a) {
    return std::sqrt(a.c[0].length_squared() + a.c[1].length_squared() +
                     a.c[2].length_squared());
}

// Rankine damage is driven by the largest principal (tensile) stress, not by
// hydrostatic mean stress and not by ordinary frictional plastic flow.
// Analytic eigenvalue for a real symmetric 3x3 (stable diagonal fast path).
inline float maxPrincipalStress(const Mat3& A) {
    const float off2 = A.at(0,1) * A.at(0,1) + A.at(0,2) * A.at(0,2) +
                       A.at(1,2) * A.at(1,2);
    if (off2 < 1.0e-16f)
        return std::max(A.at(0,0), std::max(A.at(1,1), A.at(2,2)));
    const float center = trace(A) / 3.0f;
    const float d0 = A.at(0,0) - center, d1 = A.at(1,1) - center, d2 = A.at(2,2) - center;
    const float spread2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f * off2;
    const float radius = std::sqrt(std::max(spread2 / 6.0f, 1.0e-20f));
    const Mat3 B = (A - Mat3::identity() * center) * (1.0f / radius);
    const float r = std::clamp(determinant(B) * 0.5f, -1.0f, 1.0f);
    const float phi = std::acos(r) / 3.0f;
    return center + 2.0f * radius * std::cos(phi);
}

inline Vec3 principalDirection(const Mat3& A, float eigenvalue) {
    const Mat3 M = A - Mat3::identity() * eigenvalue;
    const Vec3 row0(M.at(0,0), M.at(1,0), M.at(2,0));
    const Vec3 row1(M.at(0,1), M.at(1,1), M.at(2,1));
    const Vec3 row2(M.at(0,2), M.at(1,2), M.at(2,2));
    const Vec3 c01 = row0.cross(row1), c02 = row0.cross(row2), c12 = row1.cross(row2);
    const float n01 = c01.length_squared(), n02 = c02.length_squared(),
                n12 = c12.length_squared();
    Vec3 v = (n01 >= n02 && n01 >= n12) ? c01 : ((n02 >= n12) ? c02 : c12);
    float n2 = v.length_squared();
    if (n2 < 1.0e-16f) {
        v = (A.at(0,0) >= A.at(1,1) && A.at(0,0) >= A.at(2,2)) ? Vec3(1,0,0)
          : ((A.at(1,1) >= A.at(2,2)) ? Vec3(0,1,0) : Vec3(0,0,1));
        n2 = 1.0f;
    }
    return v * (1.0f / std::sqrt(n2));
}

// Rotation from the deformation gradient. Newton polar iteration makes the
// elastic predictor objective under large rigid rotations, unlike directly
// integrating Cauchy stress from sym(C).
inline Mat3 polarRotation(const Mat3& F) {
    Mat3 R = F;
    for (int iteration = 0; iteration < 4; ++iteration) {
        const float det_r = determinant(R);
        if (!(std::fabs(det_r) > 1.0e-8f) || !std::isfinite(det_r))
            return Mat3::identity();
        R = (R + transpose(inverse(R))) * 0.5f;
    }
    return R;
}

// ── Per-particle constitutive update ─────────────────────────────────────────
// Mirrors sim_fluid_granular_stress_update.comp line for line. `affine` is the
// APIC velocity gradient C; everything else is read-modify-write particle state.
inline void stressUpdateParticle(const AffineC& affine,
                                 const StressUpdateParams& pc,
                                 float softening,
                                 float bond_scale_in,
                                 Vec3& deformation_col0,
                                 Vec3& deformation_col1,
                                 Vec3& deformation_col2,
                                 Vec3& stress_diag,
                                 Vec3& stress_shear,
                                 float& plastic_volume,
                                 float& damage_state,
                                 float& hardening_state,
                                 float& fracture_history,
                                 float& out_yield_value,
                                 float& out_plastic_increment,
                                 uint32_t& out_flags) {
    const Mat3 C(affine.col0, affine.col1, affine.col2);
    const float nu = std::clamp(pc.poisson_ratio, -0.49f, 0.49f);
    // Stiffness falls linearly with softening; bond strength follows its OWN
    // curve (see SofteningParams -- it humps through the tacky window rather
    // than tracking stiffness down). Both arrive precomputed per particle.
    // young_modulus itself is the AUTHORED value and stays untouched — only
    // these local copies are scaled.
    const float soft = std::isfinite(softening)
        ? std::clamp(softening, 0.0f, 1.0f) : 1.0f;
    const float bond_scale = std::isfinite(bond_scale_in)
        ? std::max(bond_scale_in, 0.0f) : 1.0f;
    const float young = std::max(pc.young_modulus * soft, 1.0f);
    const float cohesion = std::max(pc.cohesion, 0.0f) * bond_scale;
    const float tensile_cutoff = std::max(pc.tensile_cutoff, 0.0f) * bond_scale;
    const float mu = young / (2.0f * (1.0f + nu));
    const float lambda = young * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
    Mat3 F(deformation_col0, deformation_col1, deformation_col2);

    // (I + dt*C) is a first-order exponential map with its own CFL, one that
    // contains no Young modulus. Frobenius <= 0.25 keeps every eigenvalue of
    // I + dC at modulus >= 0.75, so the trial gradient cannot invert in a
    // single step whatever C arrives. A clamped step is still an UNDER-RESOLVED
    // step, so it raises kFlagStrainLimited rather than passing silently.
    Mat3 dC = C * pc.dt;
    const float strain_increment = frobenius(dC);
    bool strain_limited = false;
    if (!(strain_increment < 1.0e30f)) { dC = Mat3(); strain_limited = true; }
    else if (strain_increment > kGranularStrainIncrementCeiling) {
        dC = dC * (kGranularStrainIncrementCeiling / strain_increment);
        strain_limited = true;
    }

    Mat3 F_trial = (Mat3::identity() + dC) * F;
    bool bad_kinematics = !isFiniteMat(F_trial);
    if (bad_kinematics) F_trial = isFiniteMat(F) ? F : Mat3::identity();

    const Mat3 R = polarRotation(F_trial);
    const Mat3 stretch = transpose(R) * F_trial;
    const Mat3 elastic_strain =
        (stretch + transpose(stretch)) * 0.5f - Mat3::identity();
    Mat3 local_stress = elastic_strain * (2.0f * mu) +
                        Mat3::identity() * (lambda * trace(elastic_strain));
    Mat3 S = R * local_stress * transpose(R);

    float mean = trace(S) / 3.0f;
    float pressure = -mean;
    Mat3 dev = S - Mat3::identity() * mean;
    auto deviatoricNorm = [](const Mat3& m) {
        return std::sqrt(std::max(1.5f * (m.at(0,0) * m.at(0,0) + m.at(1,1) * m.at(1,1) +
                                          m.at(2,2) * m.at(2,2) +
                                          2.0f * (m.at(0,1) * m.at(0,1) +
                                                  m.at(0,2) * m.at(0,2) +
                                                  m.at(1,2) * m.at(1,2))), 0.0f));
    };
    float q = deviatoricNorm(dev);

    float d = std::clamp(damage_state, 0.0f, 1.0f);
    const bool has_bonds = cohesion > 0.0f || tensile_cutoff > 0.0f;
    float accumulated_plastic = std::max(hardening_state, 0.0f);
    float bond_opening = std::max(fracture_history, 0.0f);
    const float hardening_scale = std::exp(std::clamp(
        pc.hardening_coefficient * accumulated_plastic, 0.0f, 4.0f));
    float effective_cohesion = (1.0f - d) * cohesion;
    float strength = std::max((effective_cohesion +
        std::max(pressure, 0.0f) * pc.friction_tangent) * hardening_scale, 0.0f);
    const float tensile = std::max(maxPrincipalStress(S), 0.0f);
    float effective_tensile = (1.0f - d) * tensile_cutoff;
    const bool tensile_failure = tensile > effective_tensile + 1.0e-6f;
    bool detached = false;
    bool yielded = q > strength + 1.0e-6f;
    const float trial_yield_overstress = std::max(q - strength, 0.0f);
    float plastic_increment = 0.0f;

    if (yielded) {
        const float scale = strength / std::max(q, 1.0e-6f);
        S = Mat3::identity() * mean + dev * scale;
        const float dp = (q - strength) / young;
        plastic_increment = dp;
        accumulated_plastic += dp;
        plastic_volume = std::clamp(
            plastic_volume * std::exp(-pc.dilatancy_tangent * dp), 0.25f, 4.0f);
    }
    if (tensile_failure) {
        if (has_bonds) {
            const float tensile_overstress_strain =
                std::max(tensile - effective_tensile, 0.0f) /
                young;
            // Rankine history is a maximum equivalent opening, not a time
            // integral. Adding the same elastic overstress every substep would
            // make the result depend on substep count and eventually fracture a
            // body under harmless APIC/free-surface noise.
            bond_opening = std::max(bond_opening, tensile_overstress_strain);
        } else {
            detached = true;
        }
    }
    if (has_bonds) {
        const float damage_driver = std::max(bond_opening - pc.fracture_strain, 0.0f);
        const float target_damage = 1.0f - std::exp(-pc.damage_rate * damage_driver);
        d = std::max(d, target_damage);
    }
    const float undamaged_strength = std::max((cohesion +
        std::max(pressure, 0.0f) * pc.friction_tangent) * hardening_scale, 0.0f);
    if (pc.rebonding && pressure > std::max(cohesion, 0.1f) &&
        q <= undamaged_strength + 1.0e-6f) {
        d -= pc.healing_rate * pc.dt;
        if (pc.damage_rate > 1.0e-6f) {
            const float compatible_history = pc.fracture_strain -
                std::log(std::max(1.0f - std::clamp(d, 0.0f, 0.999999f), 1.0e-6f)) /
                pc.damage_rate;
            bond_opening = std::min(bond_opening, std::max(compatible_history, 0.0f));
        }
    }
    d = std::clamp(d, 0.0f, 1.0f);
    effective_tensile = (1.0f - d) * tensile_cutoff;
    const float max_principal = maxPrincipalStress(S);
    if (max_principal > effective_tensile) {
        const Vec3 direction = principalDirection(S, max_principal);
        S = S - outerProduct(direction, direction) * (max_principal - effective_tensile);
    }
    mean = trace(S) / 3.0f;
    pressure = -mean;
    dev = S - Mat3::identity() * mean;
    q = deviatoricNorm(dev);
    effective_cohesion = (1.0f - d) * cohesion;
    strength = std::max((effective_cohesion +
        std::max(pressure, 0.0f) * pc.friction_tangent) * hardening_scale, 0.0f);
    if (q > strength + 1.0e-6f)
        S = Mat3::identity() * mean + dev * (strength / std::max(q, 1.0e-6f));
    detached = detached || (has_bonds && d >= 0.999f);

    // Return the projected stress to a small elastic stretch and keep the
    // rotation. Particle positions carry the large plastic motion; F stores only
    // the recoverable part, preventing stress drift over many rotations.
    local_stress = transpose(R) * S * R;
    const float local_mean = trace(local_stress) / 3.0f;
    const Mat3 local_dev = local_stress - Mat3::identity() * local_mean;
    const float bulk = std::max(lambda + 2.0f * mu / 3.0f, 1.0e-6f);
    Mat3 projected_strain = local_dev * (1.0f / std::max(2.0f * mu, 1.0e-6f)) +
                            Mat3::identity() * (local_mean / (3.0f * bulk));

    // ★★★★★ A DEFORMATION GRADIENT CAN ONLY CARRY RECOVERABLE STRAIN. The
    // strain a stress implies is stress/E, so as E falls this back-projection
    // asks the gradient to store more and more. Measured: a 1 kPa pile carries
    // 26 kPa of its own overburden with K = 667 Pa, wanting a volumetric strain
    // near -39. I + eps then has a wildly negative determinant, the next step
    // called the particle invalid, and its entire stress was dumped to zero in
    // one step -- a step-function discharge, seen as small bursts escaping the
    // pile (22 of them in a 240-step run).
    //
    // Strain past what the material can elastically hold is permanent: it is
    // compaction, and it belongs in the particle positions, which already carry
    // the plastic motion. So cap what gets STORED and let the excess be plastic.
    // S was already return-mapped and is written out untouched, so nothing is
    // discarded and there is nothing left to discharge.
    const float stored_strain = frobenius(projected_strain);
    bool compaction_capped = false;
    if (!(stored_strain < 1.0e30f)) { projected_strain = Mat3(); compaction_capped = true; }
    else if (stored_strain > pc.max_stored_strain) {
        projected_strain = projected_strain * (pc.max_stored_strain / stored_strain);
        compaction_capped = true;
    }

    Mat3 F_projected = R * (Mat3::identity() + projected_strain);
    const bool bad = bad_kinematics || !isFiniteMat(S) || !isFiniteMat(F_projected) ||
                     determinant(F_projected) <= 1.0e-6f ||
                     !std::isfinite(d) || !std::isfinite(accumulated_plastic) ||
                     !std::isfinite(bond_opening);
    if (bad) {
        S = Mat3(); F_projected = Mat3::identity(); d = 1.0f;
        accumulated_plastic = 0.0f; bond_opening = 0.0f; plastic_increment = 0.0f;
        detached = true; yielded = false; compaction_capped = false;
    }

    stress_diag  = Vec3(S.at(0,0), S.at(1,1), S.at(2,2));
    stress_shear = Vec3(S.at(0,1), S.at(0,2), S.at(1,2));
    out_yield_value = bad ? 0.0f : trial_yield_overstress;
    out_plastic_increment = plastic_increment;
    damage_state = d;
    hardening_state = std::clamp(accumulated_plastic, 0.0f, 1000.0f);
    fracture_history = std::clamp(bond_opening, 0.0f, 1000.0f);
    deformation_col0 = F_projected.c[0];
    deformation_col1 = F_projected.c[1];
    deformation_col2 = F_projected.c[2];
    out_flags = (yielded ? kFlagYielded : 0u) | (detached ? kFlagDetached : 0u) |
                (bad ? kFlagInvalid : 0u) |
                (strain_limited ? kFlagStrainLimited : 0u) |
                (compaction_capped ? kFlagCompactionCapped : 0u);
}

// Contact damping and sleeping, mirroring sim_fluid_granular_settle.comp.
struct SettleParams {
    float dt = 0.0f;
    float contact_damping = 8.0f;
    float sleep_speed = 0.03f;
    float pressure_threshold = 0.1f;
};

inline void settleParticle(const Vec3& stress_diag, const SettleParams& sp,
                           Vec3& velocity, uint32_t& flags) {
    const float pressure =
        std::max(-(stress_diag.x + stress_diag.y + stress_diag.z) / 3.0f, 0.0f);
    if (pressure <= sp.pressure_threshold) return;
    velocity = velocity * std::exp(-std::max(sp.contact_damping, 0.0f) *
                                    std::max(sp.dt, 0.0f));
    if (velocity.length() < std::max(sp.sleep_speed, 0.0f)) {
        velocity = Vec3(0, 0, 0);
        flags |= kFlagSleeping;
    } else {
        flags &= ~kFlagSleeping;
    }
}

} // namespace RayTrophiSim::Fluid::Granular
