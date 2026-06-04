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

    void clear() {
        position.clear();
        velocity.clear();
        affine.clear();
        flags.clear();
    }

    size_t size() const { return position.size(); }
    bool   empty() const { return position.empty(); }

    void reserve(size_t n) {
        position.reserve(n);
        velocity.reserve(n);
        affine.reserve(n);
        flags.reserve(n);
    }

    void emit(const Vec3& p, const Vec3& v) {
        position.push_back(p);
        velocity.push_back(v);
        affine.emplace_back();
        flags.push_back(0u);
    }

    // Remove particle i in O(1) via swap-with-back. Order is not preserved.
    void removeSwap(size_t i) {
        size_t last = position.size() - 1;
        if (i != last) {
            position[i] = position[last];
            velocity[i] = velocity[last];
            affine[i]   = affine[last];
            flags[i]    = flags[last];
        }
        position.pop_back();
        velocity.pop_back();
        affine.pop_back();
        flags.pop_back();
    }
};

} // namespace Fluid
} // namespace RayTrophiSim
