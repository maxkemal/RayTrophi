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

    void clear() {
        position.clear();
        velocity.clear();
        affine.clear();
        flags.clear();
        mass_fraction.clear();
        temperature.clear();
        combustible_fraction.clear();
        substance_tag.clear();
    }

    size_t size() const { return position.size(); }
    bool   empty() const { return position.empty(); }

    void reserve(size_t n) {
        position.reserve(n);
        velocity.reserve(n);
        affine.reserve(n);
        flags.reserve(n);
        mass_fraction.reserve(n);
        temperature.reserve(n);
        combustible_fraction.reserve(n);
        substance_tag.reserve(n);
    }

    void emit(const Vec3& p, const Vec3& v, float temp = 0.0f,
              float combustible = 0.0f, uint32_t material = 0u) {
        position.push_back(p);
        velocity.push_back(v);
        affine.emplace_back();
        flags.push_back(0u);
        mass_fraction.push_back(1.0f);
        temperature.push_back(temp);
        combustible_fraction.push_back(combustible);
        substance_tag.push_back(material);
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
        }
        position.pop_back();
        velocity.pop_back();
        affine.pop_back();
        flags.pop_back();
        mass_fraction.pop_back();
        temperature.pop_back();
        combustible_fraction.pop_back();
        substance_tag.pop_back();
    }
};

} // namespace Fluid
} // namespace RayTrophiSim
