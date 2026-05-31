/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidFoam.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Physically-based whitewater (spray / foam / bubbles) for the APIC liquid.
 *
 * Generation follows Ihmsen et al. 2012 "Unified Spray, Foam and Bubbles for
 * Particle-Based Fluids": each liquid particle accumulates three potentials —
 *   - trapped air  (high relative velocity between converging neighbours),
 *   - wave crest   (convex surface moving along its outward normal),
 *   - kinetic energy (fast particles),
 * each clamped to [0,1], and the product drives how many foam particles spawn
 * per step. Foam particles are then classified every step by how submerged they
 * are (fluid-neighbour count) and advected accordingly:
 *   - SPRAY  (few neighbours): ballistic — gravity + air drag.
 *   - FOAM   (mid):            advected WITH the fluid velocity field.
 *   - BUBBLE (many):           buoyant — rises against gravity, dragged by flow.
 * Foam has a finite lifetime (dissolves); out-of-domain particles are culled.
 *
 * Foam is a render/secondary layer: it is NOT fed back into the pressure solve,
 * so it never affects liquid mass or stability. Render side attaches a separate
 * white scattering material (see the render bridge).
 */

#pragma once

#include "../Vec3.h"
#include "../FluidGrid.h"
#include "FluidParticles.h"

#include <vector>
#include <cstddef>
#include <cstdint>

namespace RayTrophiSim {
namespace Fluid {

enum class FoamType : uint8_t { Spray = 0, Foam = 1, Bubble = 2 };

// How the whitewater is rendered.
//   Surface (default): the foam particles are fused into a metaball/Zhu-Bridson
//     SDF and meshed (marching cubes) into ONE deforming surface that carries an
//     assigned scene material — like the liquid SurfaceSDF, but as real geometry
//     so it composites correctly on every backend with no volume-integrator
//     coincidence issues. This is the production path.
//   Spheres: each foam particle is an instanced sphere (debug / close-up / low
//     counts). O(N) TLAS instances, so it crawls at the millions whitewater
//     reaches — not the default.
// (The old Volume mode — splatting foam into a white scattering NanoVDB — was
//  removed: it was unstable when coincident with the fluid surface volume.)
enum class FoamRenderMode : uint8_t { Surface = 0, Spheres = 1 };

// SoA foam storage (mirrors FluidParticles' layout philosophy).
struct FoamParticles {
    std::vector<Vec3>    position;
    std::vector<Vec3>    velocity;
    std::vector<float>   lifetime;   // remaining seconds (<= 0 → cull)
    std::vector<uint8_t> type;       // FoamType, set each step by classification

    void   clear() { position.clear(); velocity.clear(); lifetime.clear(); type.clear(); }
    size_t size()  const { return position.size(); }
    bool   empty() const { return position.empty(); }
    void   reserve(size_t n) { position.reserve(n); velocity.reserve(n); lifetime.reserve(n); type.reserve(n); }

    void emit(const Vec3& p, const Vec3& v, float life, FoamType t) {
        position.push_back(p);
        velocity.push_back(v);
        lifetime.push_back(life);
        type.push_back(static_cast<uint8_t>(t));
    }
    void removeSwap(size_t i) {
        size_t last = position.size() - 1;
        if (i != last) {
            position[i] = position[last];
            velocity[i] = velocity[last];
            lifetime[i] = lifetime[last];
            type[i]     = type[last];
        }
        position.pop_back(); velocity.pop_back(); lifetime.pop_back(); type.pop_back();
    }
};

struct FoamParams {
    bool  enabled = false;            // off by default (extra cost; opt-in)

    // Generation strength: foam particles per liquid particle per second at full
    // potential. trapped-air drives splashes/impacts, wave-crest drives breaking
    // crests / spray off the lip.
    float trapped_air_rate = 60.0f;
    float wave_crest_rate  = 60.0f;

    // Ihmsen criterion clamp ranges [tau_min, tau_max] → normalised to [0,1].
    float ta_min = 4.0f,  ta_max = 24.0f;   // trapped-air (relative-speed sum)
    float wc_min = 0.2f,  wc_max = 0.9f;    // wave-crest (normalised convexity)
    float ke_min = 1.0f,  ke_max = 60.0f;   // kinetic energy (0.5*|v|^2)

    // Crest detection: a particle is a wave crest only when it moves along its
    // outward surface normal by at least this cosine (1 = exactly outward).
    float crest_cos = 0.6f;

    // Neighbour radius (sim voxels) for generation potentials AND foam
    // classification (counting submersion). ~2.
    float neighbor_radius_voxels = 2.0f;

    // Classification thresholds on the fluid-neighbour count of a foam particle.
    int   spray_max_neighbors  = 4;    // <= this  → SPRAY (airborne)
    int   bubble_min_neighbors = 18;   // >= this  → BUBBLE (submerged)

    // Dynamics.
    float lifetime    = 2.5f;   // base foam lifetime (s); randomised +-25%
    float buoyancy    = 2.0f;   // bubble rise (× -gravity)
    float fluid_drag  = 8.0f;   // foam/bubble coupling rate to fluid velocity (1/s)
    float spray_drag  = 0.3f;   // air drag on spray (1/s)
    float spawn_jitter_voxels = 0.4f;  // spawn scatter radius (sim voxels)

    std::size_t max_foam = 100000;     // hard cap on live foam particles

    // ── Render ───────────────────────────────────────────────────────────────
    FoamRenderMode render_mode = FoamRenderMode::Surface; // metaball surface

    // Scene material assigned to the foam geometry (both Surface and Spheres).
    // -1 → a built-in default white foam material is created/used. Otherwise a
    // MaterialManager id the user picked in the foam panel.
    int   foam_material_id = -1;

    // Spheres mode: sphere radius (sim voxels).
    float render_radius_voxels = 0.3f;

    // ── Surface (metaball) mode ───────────────────────────────────────────────
    // Foam SDF reconstruction (Zhu-Bridson). Radii are in SIM voxels so the
    // surface shape is grid-resolution invariant.
    float surface_kernel_radius_voxels   = 2.0f;  // blob kernel radius
    float surface_particle_radius_voxels = 0.6f;  // per-particle bulge
    float surface_band_voxels            = 3.0f;  // narrow-band extent
    int   surface_smoothing_iterations   = 2;     // Laplacian smoothing sweeps
    int   surface_resolution_multiplier  = 1;     // SDF fineness vs sim grid (1..4)

    // Deprecated (old Volume mode); kept so existing saves deserialize cleanly.
    float volume_density = 1.0f;
};

struct FoamStats {
    std::size_t spawned   = 0;
    std::size_t alive     = 0;
    std::size_t spray     = 0;
    std::size_t foam      = 0;
    std::size_t bubble    = 0;
    float       gen_ms    = 0.0f;
    float       advect_ms = 0.0f;
};

// Advance whitewater one step: generate from the liquid particles' Ihmsen
// potentials, then classify + advect + age existing foam, culling the dead.
// `grid` supplies the velocity field for foam advection; `gravity` and `dt`
// drive spray/bubble dynamics. `seed` varies the per-step spawn jitter.
void stepFoam(const FluidParticles& fluid,
              const FluidSim::FluidGrid& grid,
              FoamParticles& foam,
              const FoamParams& params,
              const Vec3& gravity,
              float dt,
              uint32_t seed,
              FoamStats* stats = nullptr);

// Trilinear-splat the foam particles into a per-cell density field (resized to
// grid.getCellCount()) for the Volume render mode. Each particle deposits
// `density_per_particle` spread over its 8 surrounding cell centres.
void splatFoamDensity(const FoamParticles& foam,
                      const FluidSim::FluidGrid& grid,
                      std::vector<float>& density_out,
                      float density_per_particle);

// Same splat at an EXPLICIT resolution / extent. Used when the fluid surface
// volume is uploaded at a refined resolution (surface_resolution_multiplier),
// so the foam channel must match the SDF proxy's dims exactly. `origin`/`voxel`
// describe the same world extent as the sim grid, only (possibly) finer.
void splatFoamDensity(const FoamParticles& foam,
                      int nx, int ny, int nz, float voxel, const Vec3& origin,
                      std::vector<float>& density_out,
                      float density_per_particle);

} // namespace Fluid
} // namespace RayTrophiSim
