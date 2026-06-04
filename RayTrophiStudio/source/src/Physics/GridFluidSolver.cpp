/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GridFluidSolver.cpp
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*/
#include "GridFluidSolver.h"

#include "SimulationWorld.h" // SimulationForceFieldSnapshot, SimulationSystemKind
#include "CurlNoise.h"       // Physics::Noise::curlFBM_animated

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef OPENVDB_ENABLED
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridSampler.h>
#endif

namespace RayTrophiSim {
namespace GridFluid {
namespace {

using FluidSim::FluidGrid;

inline Vec3 upFromGravity(const Vec3& g) {
    const float len = std::sqrt(g.x * g.x + g.y * g.y + g.z * g.z);
    if (len < 1e-6f) {
        return Vec3(0.0f, 1.0f, 0.0f);
    }
    return Vec3(-g.x / len, -g.y / len, -g.z / len);
}

// World-space positions of the staggered MAC samples.
inline Vec3 faceXPos(const FluidGrid& g, int i, int j, int k) {
    return g.origin + Vec3(static_cast<float>(i) * g.voxel_size,
                           (static_cast<float>(j) + 0.5f) * g.voxel_size,
                           (static_cast<float>(k) + 0.5f) * g.voxel_size);
}
inline Vec3 faceYPos(const FluidGrid& g, int i, int j, int k) {
    return g.origin + Vec3((static_cast<float>(i) + 0.5f) * g.voxel_size,
                           static_cast<float>(j) * g.voxel_size,
                           (static_cast<float>(k) + 0.5f) * g.voxel_size);
}
inline Vec3 faceZPos(const FluidGrid& g, int i, int j, int k) {
    return g.origin + Vec3((static_cast<float>(i) + 0.5f) * g.voxel_size,
                           (static_cast<float>(j) + 0.5f) * g.voxel_size,
                           static_cast<float>(k) * g.voxel_size);
}

// Cell-centered velocity reconstructed from the surrounding MAC faces.
inline Vec3 cellVelocity(const FluidGrid& g, int i, int j, int k) {
    i = std::clamp(i, 0, g.nx - 1);
    j = std::clamp(j, 0, g.ny - 1);
    k = std::clamp(k, 0, g.nz - 1);
    const float u = 0.5f * (g.velXAt(i, j, k) + g.velXAt(i + 1, j, k));
    const float v = 0.5f * (g.velYAt(i, j, k) + g.velYAt(i, j + 1, k));
    const float w = 0.5f * (g.velZAt(i, j, k) + g.velZAt(i, j, k + 1));
    return Vec3(u, v, w);
}

// ─────────────────────────────────────────────────────────────────────────
// Advection (semi-Lagrangian)
// ─────────────────────────────────────────────────────────────────────────
void advectVelocity(FluidGrid& grid, float dt) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    std::vector<float> nvx(grid.vel_x.size(), 0.0f);
    std::vector<float> nvy(grid.vel_y.size(), 0.0f);
    std::vector<float> nvz(grid.vel_z.size(), 0.0f);

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const Vec3 p = faceXPos(grid, i, j, k);
                const Vec3 back = p - grid.sampleVelocity(p) * dt;
                nvx[grid.velXIndex(i, j, k)] = grid.sampleVelocity(back).x;
            }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Vec3 p = faceYPos(grid, i, j, k);
                const Vec3 back = p - grid.sampleVelocity(p) * dt;
                nvy[grid.velYIndex(i, j, k)] = grid.sampleVelocity(back).y;
            }
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Vec3 p = faceZPos(grid, i, j, k);
                const Vec3 back = p - grid.sampleVelocity(p) * dt;
                nvz[grid.velZIndex(i, j, k)] = grid.sampleVelocity(back).z;
            }

    grid.vel_x.swap(nvx);
    grid.vel_y.swap(nvy);
    grid.vel_z.swap(nvz);
}

void advectScalar(FluidGrid& grid, std::vector<float>& field, float dt) {
    if (field.empty()) {
        return;
    }
    std::vector<float> next(field.size(), 0.0f);
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Vec3 p = grid.gridToWorld(i, j, k);
                const Vec3 back = p - grid.sampleVelocity(p) * dt;
                next[grid.cellIndex(i, j, k)] = grid.sampleCellCentered(field, back);
            }
    field.swap(next);
}

// ─────────────────────────────────────────────────────────────────────────
// Forces
// ─────────────────────────────────────────────────────────────────────────
void addToFaces(FluidGrid& grid, int i, int j, int k, const Vec3& f) {
    grid.velXAt(i, j, k)     += f.x * 0.5f;
    grid.velXAt(i + 1, j, k) += f.x * 0.5f;
    grid.velYAt(i, j, k)     += f.y * 0.5f;
    grid.velYAt(i, j + 1, k) += f.y * 0.5f;
    grid.velZAt(i, j, k)     += f.z * 0.5f;
    grid.velZAt(i, j, k + 1) += f.z * 0.5f;
}

void addBuoyancy(FluidGrid& grid, const SolverParams& params, float dt) {
    const Vec3 up = upFromGravity(params.gravity);
    const bool has_density = params.channel_density;
    const bool has_temp = params.channel_temperature;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = grid.cellIndex(i, j, k);
                const float dens = has_density ? grid.density[c] : 0.0f;
                const float temp = has_temp ? grid.temperature[c] : 0.0f;
                const float b = (params.buoyancy_heat * (temp - params.ambient_temperature) +
                                 params.buoyancy_density * dens) * dt;
                if (b != 0.0f) {
                    addToFaces(grid, i, j, k, up * b);
                }
            }
}

void addForceFields(FluidGrid& grid,
                    float dt,
                    const SimulationForceFieldSnapshot* forces,
                    float time) {
    if (!forces || forces->empty()) {
        return;
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const Vec3 wp = grid.gridToWorld(i, j, k);
                const Vec3 vel = cellVelocity(grid, i, j, k);
                const Vec3 accel = forces->evaluateAt(wp, time, vel, SimulationSystemKind::Gas);
                addToFaces(grid, i, j, k, accel * dt);
            }
}

void vorticityConfinement(FluidGrid& grid, const SolverParams& params, float dt) {
    if (params.vorticity <= 0.0f || grid.nx < 3 || grid.ny < 3 || grid.nz < 3) {
        return;
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h = grid.voxel_size;
    const float inv2h = h > 1e-6f ? 1.0f / (2.0f * h) : 0.0f;
    const std::size_t cells = grid.getCellCount();

    std::vector<Vec3> omega(cells, Vec3(0.0f));
    std::vector<float> mag(cells, 0.0f);

    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i) {
                const Vec3 vxp = cellVelocity(grid, i + 1, j, k);
                const Vec3 vxm = cellVelocity(grid, i - 1, j, k);
                const Vec3 vyp = cellVelocity(grid, i, j + 1, k);
                const Vec3 vym = cellVelocity(grid, i, j - 1, k);
                const Vec3 vzp = cellVelocity(grid, i, j, k + 1);
                const Vec3 vzm = cellVelocity(grid, i, j, k - 1);

                const float wx = (vyp.z - vym.z) * inv2h - (vzp.y - vzm.y) * inv2h;
                const float wy = (vzp.x - vzm.x) * inv2h - (vxp.z - vxm.z) * inv2h;
                const float wz = (vxp.y - vxm.y) * inv2h - (vyp.x - vym.x) * inv2h;
                const std::size_t c = grid.cellIndex(i, j, k);
                omega[c] = Vec3(wx, wy, wz);
                mag[c] = std::sqrt(wx * wx + wy * wy + wz * wz);
            }

    const float eps = params.vorticity * h;
    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i) {
                const float gx = (mag[grid.cellIndex(i + 1, j, k)] - mag[grid.cellIndex(i - 1, j, k)]) * inv2h;
                const float gy = (mag[grid.cellIndex(i, j + 1, k)] - mag[grid.cellIndex(i, j - 1, k)]) * inv2h;
                const float gz = (mag[grid.cellIndex(i, j, k + 1)] - mag[grid.cellIndex(i, j, k - 1)]) * inv2h;
                const float glen = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-12f;
                const float nx_ = gx / glen, ny_ = gy / glen, nz_ = gz / glen;

                const Vec3 w = omega[grid.cellIndex(i, j, k)];
                // force = eps * (N x omega)
                const Vec3 force(
                    eps * (ny_ * w.z - nz_ * w.y),
                    eps * (nz_ * w.x - nx_ * w.z),
                    eps * (nx_ * w.y - ny_ * w.x));
                addToFaces(grid, i, j, k, force * dt);
            }
}

// Divergence-free curl-noise turbulence. Adds animated FBM swirl modulated by
// local activity (density / heat / density-edge) so still air stays still.
// Ported from the legacy GasSimulator; uses the same Physics::Noise field but
// reads parameters from SolverParams and deposits onto MAC faces.
void curlNoiseTurbulence(FluidGrid& grid, const SolverParams& params, float dt, float time_seconds) {
    if (params.turbulence_strength < 1e-3f || grid.nx < 3 || grid.ny < 3 || grid.nz < 3) {
        return;
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h = grid.voxel_size;
    const float inv2h = h > 1e-6f ? 1.0f / (2.0f * h) : 0.0f;
    const float strength = params.turbulence_strength;
    const float freq = params.turbulence_scale;
    const int octaves = std::clamp(params.turbulence_octaves, 1, 8);
    const float lacunarity = params.turbulence_lacunarity;
    const float persistence = params.turbulence_persistence;
    const float anim_speed = params.turbulence_speed;
    const int seed = params.turbulence_seed;
    const bool has_density = params.channel_density && !grid.density.empty();
    const bool has_temp = params.channel_temperature && !grid.temperature.empty();
    const float heat_range = std::max(params.ignition_temperature - params.ambient_temperature, 1.0f);

    #pragma omp parallel for collapse(2)
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                const std::size_t c = grid.cellIndex(i, j, k);
                const float density = has_density ? grid.density[c] : 0.0f;
                const float temperature = has_temp ? grid.temperature[c] : 0.0f;
                const float heat_norm = std::clamp((temperature - params.ambient_temperature) / heat_range, 0.0f, 3.0f);
                const float flame_norm = std::clamp(grid.interaction[c] * 0.05f, 0.0f, 2.0f);

                float edge_norm = 0.0f;
                if (has_density) {
                    const float gx = (grid.densityAt(i + 1, j, k) - grid.densityAt(i - 1, j, k)) * inv2h;
                    const float gy = (grid.densityAt(i, j + 1, k) - grid.densityAt(i, j - 1, k)) * inv2h;
                    const float gz = (grid.densityAt(i, j, k + 1) - grid.densityAt(i, j, k - 1)) * inv2h;
                    edge_norm = std::clamp(std::sqrt(gx * gx + gy * gy + gz * gz) * 0.08f, 0.0f, 2.0f);
                }
                const float activity = std::max({ density, heat_norm * 0.35f, flame_norm * 0.5f, edge_norm * 0.25f });
                if (activity < 0.01f) {
                    continue;
                }

                // Push more breakup into hot cores and density edges, not just bulk smoke.
                const float local_strength = strength * std::clamp(
                    0.18f + 0.45f * std::sqrt(std::max(density, 0.0f)) +
                    0.40f * std::min(heat_norm, 1.5f) +
                    0.30f * std::min(flame_norm, 1.5f) +
                    0.35f * std::min(edge_norm, 1.0f),
                    0.0f, 2.5f);

                const Vec3 wp = grid.gridToWorld(i, j, k);
                const Vec3 curl = Physics::Noise::curlFBM_animated(
                    wp, time_seconds, octaves, freq, lacunarity, persistence, anim_speed, seed);

                addToFaces(grid, i, j, k, curl * (local_strength * dt));
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Dissipation / clamping
// ─────────────────────────────────────────────────────────────────────────
void dissipate(std::vector<float>& field, float rate, float dt) {
    if (rate <= 0.0f || field.empty()) {
        return;
    }
    const float factor = std::exp(-rate * dt);
    for (float& v : field) {
        v *= factor;
    }
}

void clampVelocity(FluidGrid& grid, float max_velocity) {
    if (max_velocity <= 0.0f) {
        return;
    }
    const float m = max_velocity;
    for (float& v : grid.vel_x) v = std::clamp(v, -m, m);
    for (float& v : grid.vel_y) v = std::clamp(v, -m, m);
    for (float& v : grid.vel_z) v = std::clamp(v, -m, m);
}

// ─────────────────────────────────────────────────────────────────────────
// Combustion (opt-in fire). Bounded one-way burn; no neighbor-flame spread —
// fire moves only because advection transports hot temperature into fuel.
// ─────────────────────────────────────────────────────────────────────────
void processCombustion(FluidGrid& grid, const SolverParams& params, float dt) {
    if (!params.fire_enabled || !params.channel_fuel || grid.fuel.empty() || grid.temperature.empty()) {
        return;
    }
    const std::size_t n = grid.getCellCount();
    const float flame_decay = std::exp(-std::max(0.0f, params.flame_dissipation) * dt);
    const float inv_dt = dt > 1e-5f ? 1.0f / dt : 0.0f;

    for (std::size_t c = 0; c < n; ++c) {
        // Flame field always relaxes; re-lit below where fuel actually burns.
        grid.interaction[c] *= flame_decay;

        const float f = grid.fuel[c];
        const float t = grid.temperature[c];
        if (f <= 1e-5f || t < params.ignition_temperature) {
            continue;
        }

        // Consume a bounded fraction of the available fuel this step.
        const float burned = std::min(f, std::max(0.0f, params.burn_rate) * f * dt);
        if (burned <= 0.0f) {
            continue;
        }
        grid.fuel[c] = f - burned;
        grid.temperature[c] = std::min(t + params.heat_release * burned, params.max_temperature);
        if (!grid.density.empty()) {
            grid.density[c] += params.smoke_generation * burned;
        }
        grid.interaction[c] = std::min(1.0f, grid.interaction[c] + burned * inv_dt);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Boundary conditions
// ─────────────────────────────────────────────────────────────────────────
void setWallBcs(FluidGrid& grid, Boundary boundary) {
    if (boundary != Boundary::Closed) {
        return; // Open allows outflow; Periodic handled in the pressure solve.
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            grid.velXAt(0, j, k) = 0.0f;
            grid.velXAt(nx, j, k) = 0.0f;
        }
    for (int k = 0; k < nz; ++k)
        for (int i = 0; i < nx; ++i) {
            grid.velYAt(i, 0, k) = 0.0f;
            grid.velYAt(i, ny, k) = 0.0f;
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            grid.velZAt(i, j, 0) = 0.0f;
            grid.velZAt(i, j, nz) = 0.0f;
        }
}

// ─────────────────────────────────────────────────────────────────────────
// Solid (collider) coupling
// ─────────────────────────────────────────────────────────────────────────
// grid.solid[] is stamped by the collider voxelizer before the step. These two
// helpers make the gas solver honour those solids the same way the APIC liquid
// path does: MAC faces touching a solid cell carry the solid's velocity (no
// penetration; a MOVING collider hands its momentum to the gas), and scalar
// content inside solids is wiped so smoke never accumulates inside a collider.
// All guarded so a domain with no colliders (grid.solid all-zero) pays nothing
// beyond a single empty-check.
inline bool gridHasSolid(const FluidGrid& grid) {
    return grid.solid.size() == static_cast<std::size_t>(grid.getCellCount());
}

void enforceSolidBoundaries(FluidGrid& grid) {
    if (!gridHasSolid(grid)) return;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const bool has_vel = grid.solid_vel.size() == grid.solid.size();
    auto solidVel = [&](int i, int j, int k) -> Vec3 {
        return has_vel ? grid.solid_vel[grid.cellIndex(i, j, k)] : Vec3(0.0f, 0.0f, 0.0f);
    };
    // X faces: face (i,j,k) borders cells (i-1) and (i).
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const bool sL = (i > 0)  && grid.isSolid(i - 1, j, k);
                const bool sR = (i < nx) && grid.isSolid(i, j, k);
                if (!sL && !sR) continue;
                float v = 0.0f;
                if (sL && sR)      v = 0.5f * (solidVel(i - 1, j, k).x + solidVel(i, j, k).x);
                else if (sL)       v = solidVel(i - 1, j, k).x;
                else               v = solidVel(i, j, k).x;
                grid.velXAt(i, j, k) = v;
            }
    // Y faces.
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const bool sL = (j > 0)  && grid.isSolid(i, j - 1, k);
                const bool sR = (j < ny) && grid.isSolid(i, j, k);
                if (!sL && !sR) continue;
                float v = 0.0f;
                if (sL && sR)      v = 0.5f * (solidVel(i, j - 1, k).y + solidVel(i, j, k).y);
                else if (sL)       v = solidVel(i, j - 1, k).y;
                else               v = solidVel(i, j, k).y;
                grid.velYAt(i, j, k) = v;
            }
    // Z faces.
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const bool sL = (k > 0)  && grid.isSolid(i, j, k - 1);
                const bool sR = (k < nz) && grid.isSolid(i, j, k);
                if (!sL && !sR) continue;
                float v = 0.0f;
                if (sL && sR)      v = 0.5f * (solidVel(i, j, k - 1).z + solidVel(i, j, k).z);
                else if (sL)       v = solidVel(i, j, k - 1).z;
                else               v = solidVel(i, j, k).z;
                grid.velZAt(i, j, k) = v;
            }
}

void clearSolidScalars(FluidGrid& grid, const SolverParams& params) {
    if (!gridHasSolid(grid)) return;
    const std::size_t cells = static_cast<std::size_t>(grid.getCellCount());
    const bool clr_d = params.channel_density     && grid.density.size()     == cells;
    const bool clr_t = params.channel_temperature && grid.temperature.size() == cells;
    const bool clr_f = params.channel_fuel        && grid.fuel.size()        == cells;
    if (!clr_d && !clr_t && !clr_f) return;
    for (std::size_t c = 0; c < cells; ++c) {
        if (!grid.solid[c]) continue;
        if (clr_d) grid.density[c]     = 0.0f;
        if (clr_t) grid.temperature[c] = 0.0f;
        if (clr_f) grid.fuel[c]        = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Pressure projection (make velocity divergence-free)
// ─────────────────────────────────────────────────────────────────────────
void project(FluidGrid& grid, const SolverParams& params, float dt) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h = grid.voxel_size > 1e-6f ? grid.voxel_size : 1.0f;
    const float inv_h = 1.0f / h;

    std::vector<float>& div = grid.divergence;
    std::vector<float>& P = grid.pressure; // warm-started across frames
    if (div.size() != grid.getCellCount() || P.size() != grid.getCellCount()) {
        return;
    }

    // Solid (collider) mask. When present, solid cells act as internal Neumann
    // walls: their pressure is never solved, the Laplacian stencil mirrors the
    // centre cell across a solid face (zero flux), and the pressure gradient is
    // not subtracted on faces touching a solid (their velocity is already the
    // enforced solid velocity). With no colliders has_solid is false and every
    // branch below collapses to the original divergence-free interior solve.
    const bool has_solid = gridHasSolid(grid);
    auto isSolidCell = [&](int ii, int jj, int kk) -> bool {
        return has_solid && ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz &&
               grid.solid[grid.cellIndex(ii, jj, kk)] != 0u;
    };

    // Thermal expansion target: hot gas dilates, so the projection aims for a
    // POSITIVE divergence φ = expansion·max(0, T - ambient) at hot cells instead
    // of the usual div = 0. Subtracting φ from the measured divergence here makes
    // the solved field satisfy ∇·u = φ (gas pushed outward → fire roll / blast).
    const bool use_expansion =
        params.expansion > 0.0f && params.channel_temperature &&
        grid.temperature.size() == static_cast<std::size_t>(grid.getCellCount());
    const float ambient = params.ambient_temperature;
    const float expansion = params.expansion;
    const float max_phi = expansion * std::max(0.0f, params.max_temperature - ambient);

    // 1) Divergence of the current velocity field (skip solid cells).
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = grid.cellIndex(i, j, k);
                if (has_solid && grid.solid[c]) { div[c] = 0.0f; continue; }
                const float du = grid.velXAt(i + 1, j, k) - grid.velXAt(i, j, k);
                const float dv = grid.velYAt(i, j + 1, k) - grid.velYAt(i, j, k);
                const float dw = grid.velZAt(i, j, k + 1) - grid.velZAt(i, j, k);
                float d = (du + dv + dw) * inv_h;
                if (use_expansion) {
                    const float phi = std::min(max_phi, expansion * std::max(0.0f, grid.temperature[c] - ambient));
                    d -= phi; // target divergence φ instead of 0
                }
                div[c] = d;
            }

    const bool open = (params.boundary == Boundary::Open);
    const bool periodic = (params.boundary == Boundary::Periodic);
    const float h2 = h * h;
    const float inv_dt = dt > 1e-8f ? 1.0f / dt : 0.0f;

    // Pressure sample honoring the boundary mode (Dirichlet 0 / Neumann / wrap).
    auto getP = [&](int ii, int jj, int kk) -> float {
        if (periodic) {
            ii = (ii % nx + nx) % nx;
            jj = (jj % ny + ny) % ny;
            kk = (kk % nz + nz) % nz;
            return P[grid.cellIndex(ii, jj, kk)];
        }
        if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) {
            if (open) {
                return 0.0f; // Dirichlet p = 0 outside -> outflow
            }
            ii = std::clamp(ii, 0, nx - 1); // Neumann: mirror to the boundary cell
            jj = std::clamp(jj, 0, ny - 1);
            kk = std::clamp(kk, 0, nz - 1);
            return P[grid.cellIndex(ii, jj, kk)];
        }
        return P[grid.cellIndex(ii, jj, kk)];
    };

    // 2) Solve the Poisson system with SOR-relaxed Gauss-Seidel. A solid
    // neighbour mirrors the centre cell's pressure (Neumann: zero flux through
    // the solid face), matching how the closed-domain walls are treated, so the
    // fixed 6-point stencil is preserved and the no-collider path is unchanged.
    const int iterations = std::max(1, params.pressure_iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const std::size_t c = grid.cellIndex(i, j, k);
                    if (has_solid && grid.solid[c]) continue; // solid: pressure not solved
                    const float pc = P[c];
                    auto nb = [&](int ii, int jj, int kk) -> float {
                        return isSolidCell(ii, jj, kk) ? pc : getP(ii, jj, kk);
                    };
                    const float sum =
                        nb(i - 1, j, k) + nb(i + 1, j, k) +
                        nb(i, j - 1, k) + nb(i, j + 1, k) +
                        nb(i, j, k - 1) + nb(i, j, k + 1);
                    const float rhs = div[c] * h2 * inv_dt;
                    const float p_gs = (sum - rhs) / 6.0f;
                    P[c] += params.sor_omega * (p_gs - P[c]);
                }
    }

    // 3) Subtract the pressure gradient from velocity (boundary-aware). Faces
    // touching a solid cell are skipped — their velocity is the enforced solid
    // velocity and must not be perturbed by the projection.
    const float scale = dt * inv_h;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                if (isSolidCell(i - 1, j, k) || isSolidCell(i, j, k)) continue;
                grid.velXAt(i, j, k) -= scale * (getP(i, j, k) - getP(i - 1, j, k));
            }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (isSolidCell(i, j - 1, k) || isSolidCell(i, j, k)) continue;
                grid.velYAt(i, j, k) -= scale * (getP(i, j, k) - getP(i, j - 1, k));
            }
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (isSolidCell(i, j, k - 1) || isSolidCell(i, j, k)) continue;
                grid.velZAt(i, j, k) -= scale * (getP(i, j, k) - getP(i, j, k - 1));
            }
}

#ifdef OPENVDB_ENABLED
// Sparse, active-voxel-only pressure projection for the OpenVDB solver path.
// Mirrors project() exactly — forward-difference divergence, 7-point SOR
// Gauss-Seidel Poisson, backward-difference gradient subtraction, all at
// spacing h — but iterates ONLY the active velocity topology instead of the
// full nx*ny*nz grid. Inactive neighbours read as background (velocity 0 /
// pressure 0): an open (Dirichlet p=0) boundary at the active-region edge,
// the natural smoke/gas outflow case. THIS is the real sparsity win
// (O(active voxels), not O(cells)) and it replaces the old dense project()
// that stepSparseVDB used to run on the full grid after a dense write-back.
//
// Velocity is stored collocated at cell centres here (the convention the rest
// of stepSparseVDB uses), so the staggered project() math lands on the
// collocated samples — identical to what the old dense project() did to the
// written-back values, just restricted to the active set.
static void projectSparseVDB(const openvdb::FloatGrid::Ptr& velx,
                             const openvdb::FloatGrid::Ptr& vely,
                             const openvdb::FloatGrid::Ptr& velz,
                             const SolverParams& params, float h, float dt) {
    const float inv_h  = (h > 1e-6f) ? (1.0f / h) : 1.0f;
    const float h2     = h * h;
    const float inv_dt = (dt > 1e-8f) ? (1.0f / dt) : 0.0f;

    // Solve domain = union of the three velocity components' active voxels.
    openvdb::FloatGrid::Ptr pressure = openvdb::FloatGrid::create(0.0f);
    pressure->topologyUnion(*velx);
    pressure->topologyUnion(*vely);
    pressure->topologyUnion(*velz);
    if (pressure->activeVoxelCount() == 0) return;
    openvdb::FloatGrid::Ptr divg = pressure->deepCopy(); // same topology, zeros

    // 1) Divergence (forward differences), per active voxel.
    {
        openvdb::FloatGrid::ConstAccessor vxA = velx->getConstAccessor();
        openvdb::FloatGrid::ConstAccessor vyA = vely->getConstAccessor();
        openvdb::FloatGrid::ConstAccessor vzA = velz->getConstAccessor();
        for (openvdb::FloatGrid::ValueOnIter it = divg->beginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            const float du = vxA.getValue(c.offsetBy(1, 0, 0)) - vxA.getValue(c);
            const float dv = vyA.getValue(c.offsetBy(0, 1, 0)) - vyA.getValue(c);
            const float dw = vzA.getValue(c.offsetBy(0, 0, 1)) - vzA.getValue(c);
            it.setValue((du + dv + dw) * inv_h);
        }
    }

    // 2) SOR Gauss-Seidel Poisson on active voxels (Dirichlet p=0 outside).
    {
        const int   iterations = std::max(1, params.pressure_iterations);
        const float omega = params.sor_omega;
        openvdb::FloatGrid::Accessor      pA = pressure->getAccessor();
        openvdb::FloatGrid::ConstAccessor dA = divg->getConstAccessor();
        for (int iter = 0; iter < iterations; ++iter) {
            for (openvdb::FloatGrid::ValueOnIter it = pressure->beginValueOn(); it; ++it) {
                const openvdb::Coord c = it.getCoord();
                const float sum =
                    pA.getValue(c.offsetBy(-1, 0, 0)) + pA.getValue(c.offsetBy(1, 0, 0)) +
                    pA.getValue(c.offsetBy(0, -1, 0)) + pA.getValue(c.offsetBy(0, 1, 0)) +
                    pA.getValue(c.offsetBy(0, 0, -1)) + pA.getValue(c.offsetBy(0, 0, 1));
                const float rhs  = dA.getValue(c) * h2 * inv_dt;
                const float p_gs = (sum - rhs) / 6.0f;
                it.setValue(it.getValue() + omega * (p_gs - it.getValue()));
            }
        }
    }

    // 3) Subtract pressure gradient (backward differences) from velocity.
    {
        const float scale = dt * inv_h;
        openvdb::FloatGrid::ConstAccessor pA  = pressure->getConstAccessor();
        openvdb::FloatGrid::Accessor      vxA = velx->getAccessor();
        openvdb::FloatGrid::Accessor      vyA = vely->getAccessor();
        openvdb::FloatGrid::Accessor      vzA = velz->getAccessor();
        for (openvdb::FloatGrid::ValueOnCIter it = pressure->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            const float p  = it.getValue();
            const float gx = p - pA.getValue(c.offsetBy(-1, 0, 0));
            const float gy = p - pA.getValue(c.offsetBy(0, -1, 0));
            const float gz = p - pA.getValue(c.offsetBy(0, 0, -1));
            vxA.setValue(c, vxA.getValue(c) - scale * gx);
            vyA.setValue(c, vyA.getValue(c) - scale * gy);
            vzA.setValue(c, vzA.getValue(c) - scale * gz);
        }
    }
}
#endif

} // namespace

void step(FluidGrid& grid,
          const SolverParams& params,
          float dt,
          const SimulationForceFieldSnapshot* forces,
          float time_seconds) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return;
    }

    // 1) Transport.
    if (params.channel_velocity && !params.skip_velocity_advection) {
        advectVelocity(grid, dt);
    }
    if (params.channel_density && !params.skip_scalar_advection) {
        advectScalar(grid, grid.density, dt);
    }
    if (params.channel_temperature && !params.skip_scalar_advection) {
        advectScalar(grid, grid.temperature, dt);
    }
    if (params.channel_fuel && !params.skip_scalar_advection) {
        advectScalar(grid, grid.fuel, dt);
    }

    setWallBcs(grid, params.boundary);
    // Collider coupling: stop scalar content from advecting into solids, and pin
    // the velocity at solid faces (no-op when the domain has no colliders).
    clearSolidScalars(grid, params);
    if (params.channel_velocity) enforceSolidBoundaries(grid);

    // 2) Combustion: burn fuel -> heat + smoke (before buoyancy so released heat
    // lifts this step). Opt-in; no-op when fire is disabled.
    processCombustion(grid, params, dt);

    // 3) Body forces.
    if (params.channel_velocity) {
        addBuoyancy(grid, params, dt);
        addForceFields(grid, dt, forces, time_seconds);
        vorticityConfinement(grid, params, dt);
        curlNoiseTurbulence(grid, params, dt, time_seconds);
    }

    // 4) Dissipation.
    if (params.channel_density) {
        dissipate(grid.density, params.density_dissipation, dt);
    }
    if (params.channel_temperature) {
        dissipate(grid.temperature, params.temperature_dissipation, dt);
    }
    if (params.channel_fuel) {
        dissipate(grid.fuel, params.fuel_dissipation, dt);
    }

    // 5) Make the velocity field incompressible.
    if (params.channel_velocity) {
        if (!params.skip_velocity_dissipation_clamp) {
            dissipate(grid.vel_x, params.velocity_dissipation, dt);
            dissipate(grid.vel_y, params.velocity_dissipation, dt);
            dissipate(grid.vel_z, params.velocity_dissipation, dt);
            clampVelocity(grid, params.max_velocity);
        }
        // Re-pin solid faces after the body forces / dissipation perturbed them,
        // so the projection's divergence sees the true (possibly moving) solid
        // velocity and the result stays incompressible around the collider.
        enforceSolidBoundaries(grid);
        if (!params.skip_pressure_projection) {
            project(grid, params, dt);
        }
        setWallBcs(grid, params.boundary);
        enforceSolidBoundaries(grid);
    }
}

void stepSparseVDB(FluidGrid& grid,
                   const SolverParams& params,
                   float dt,
                   const SimulationForceFieldSnapshot* forces,
                   float time_seconds) {
#ifdef OPENVDB_ENABLED
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return;
    }

    // 1) First run combustion (processCombustion) on the dense grid to utilize all existing features.
    processCombustion(grid, params, dt);

    // 2) Create sparse OpenVDB grids
    openvdb::FloatGrid::Ptr density_vdb = openvdb::FloatGrid::create(0.0f);
    openvdb::FloatGrid::Ptr temp_vdb = openvdb::FloatGrid::create(params.ambient_temperature);
    openvdb::FloatGrid::Ptr fuel_vdb = openvdb::FloatGrid::create(0.0f);
    openvdb::FloatGrid::Ptr velx_vdb = openvdb::FloatGrid::create(0.0f);
    openvdb::FloatGrid::Ptr vely_vdb = openvdb::FloatGrid::create(0.0f);
    openvdb::FloatGrid::Ptr velz_vdb = openvdb::FloatGrid::create(0.0f);

    openvdb::FloatGrid::Accessor density_acc = density_vdb->getAccessor();
    openvdb::FloatGrid::Accessor temp_acc = temp_vdb->getAccessor();
    openvdb::FloatGrid::Accessor fuel_acc = fuel_vdb->getAccessor();
    openvdb::FloatGrid::Accessor velx_acc = velx_vdb->getAccessor();
    openvdb::FloatGrid::Accessor vely_acc = vely_vdb->getAccessor();
    openvdb::FloatGrid::Accessor velz_acc = velz_vdb->getAccessor();

    const bool has_density = params.channel_density && !grid.density.empty();
    const bool has_temp = params.channel_temperature && !grid.temperature.empty();
    const bool has_fuel = params.channel_fuel && !grid.fuel.empty();

    // Populate VDB grids from the dense grid
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                size_t idx = grid.cellIndex(i, j, k);
                float d = has_density ? grid.density[idx] : 0.0f;
                float t = has_temp ? grid.temperature[idx] : params.ambient_temperature;
                float f = has_fuel ? grid.fuel[idx] : 0.0f;
                float vx = grid.vel_x[grid.velXIndex(i, j, k)];
                float vy = grid.vel_y[grid.velYIndex(i, j, k)];
                float vz = grid.vel_z[grid.velZIndex(i, j, k)];

                openvdb::Coord coord(i, j, k);
                if (d > 0.0001f) density_acc.setValue(coord, d);
                if (std::abs(t - params.ambient_temperature) > 0.1f) temp_acc.setValue(coord, t);
                if (f > 0.0001f) fuel_acc.setValue(coord, f);
                if (std::abs(vx) > 0.0001f) velx_acc.setValue(coord, vx);
                if (std::abs(vy) > 0.0001f) vely_acc.setValue(coord, vy);
                if (std::abs(vz) > 0.0001f) velz_acc.setValue(coord, vz);
            }
        }
    }

    // 3) Apply Buoyancy and Gravity sparsely inside VDB grids
    const Vec3 up = upFromGravity(params.gravity);
    openvdb::CoordBBox bbox = density_vdb->evalActiveVoxelBoundingBox();
    if (!bbox.empty()) {
        for (int k = bbox.min().z(); k <= bbox.max().z(); ++k) {
            for (int j = bbox.min().y(); j <= bbox.max().y(); ++j) {
                for (int i = bbox.min().x(); i <= bbox.max().x(); ++i) {
                    openvdb::Coord coord(i, j, k);
                    float d = density_acc.getValue(coord);
                    float t = temp_acc.getValue(coord);
                    float temp_diff = t - params.ambient_temperature;
                    float buoyancy_force = params.buoyancy_heat * temp_diff + params.buoyancy_density * d;
                    
                    // Apply buoyancy and gravity to velocity Y
                    if (params.channel_velocity) {
                        float vx = velx_acc.getValue(coord) + (up.x * buoyancy_force + params.gravity.x) * dt;
                        float vy = vely_acc.getValue(coord) + (up.y * buoyancy_force + params.gravity.y) * dt;
                        float vz = velz_acc.getValue(coord) + (up.z * buoyancy_force + params.gravity.z) * dt;
                        velx_acc.setValue(coord, vx);
                        vely_acc.setValue(coord, vy);
                        velz_acc.setValue(coord, vz);
                    }
                }
            }
        }
    }

    // 4) Sparse Semi-Lagrangian Advection
    openvdb::FloatGrid::Ptr prev_density = density_vdb->deepCopy();
    openvdb::FloatGrid::Ptr prev_temp = temp_vdb->deepCopy();
    openvdb::FloatGrid::Ptr prev_fuel = fuel_vdb->deepCopy();
    openvdb::FloatGrid::Ptr prev_velx = velx_vdb->deepCopy();
    openvdb::FloatGrid::Ptr prev_vely = vely_vdb->deepCopy();
    openvdb::FloatGrid::Ptr prev_velz = velz_vdb->deepCopy();

    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_d(*prev_density);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_t(*prev_temp);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_f(*prev_fuel);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_vx(*prev_velx);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_vy(*prev_vely);
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler_vz(*prev_velz);

    density_vdb->clear();
    temp_vdb->clear();
    fuel_vdb->clear();
    velx_vdb->clear();
    vely_vdb->clear();
    velz_vdb->clear();

    openvdb::CoordBBox active_bbox = prev_density->evalActiveVoxelBoundingBox();
    if (!active_bbox.empty()) {
        active_bbox.expand(1);
        active_bbox.min().setX(std::max(0, active_bbox.min().x()));
        active_bbox.min().setY(std::max(0, active_bbox.min().y()));
        active_bbox.min().setZ(std::max(0, active_bbox.min().z()));
        active_bbox.max().setX(std::min(grid.nx - 1, active_bbox.max().x()));
        active_bbox.max().setY(std::min(grid.ny - 1, active_bbox.max().y()));
        active_bbox.max().setZ(std::min(grid.nz - 1, active_bbox.max().z()));

        for (int k = active_bbox.min().z(); k <= active_bbox.max().z(); ++k) {
            for (int j = active_bbox.min().y(); j <= active_bbox.max().y(); ++j) {
                for (int i = active_bbox.min().x(); i <= active_bbox.max().x(); ++i) {
                    openvdb::Coord coord(i, j, k);

                    float vx = sampler_vx.isSample(openvdb::Vec3R(i, j, k));
                    float vy = sampler_vy.isSample(openvdb::Vec3R(i, j, k));
                    float vz = sampler_vz.isSample(openvdb::Vec3R(i, j, k));

                    openvdb::Vec3R back_pos(
                        i - vx * (dt / grid.voxel_size),
                        j - vy * (dt / grid.voxel_size),
                        k - vz * (dt / grid.voxel_size)
                    );

                    back_pos.x() = std::clamp(back_pos.x(), 0.0, double(grid.nx - 1));
                    back_pos.y() = std::clamp(back_pos.y(), 0.0, double(grid.ny - 1));
                    back_pos.z() = std::clamp(back_pos.z(), 0.0, double(grid.nz - 1));

                    float advected_d = sampler_d.isSample(back_pos);
                    float advected_t = sampler_t.isSample(back_pos);
                    float advected_f = sampler_f.isSample(back_pos);
                    float advected_vx = sampler_vx.isSample(back_pos);
                    float advected_vy = sampler_vy.isSample(back_pos);
                    float advected_vz = sampler_vz.isSample(back_pos);

                    if (advected_d > 0.0001f) density_acc.setValue(coord, advected_d);
                    if (std::abs(advected_t - params.ambient_temperature) > 0.1f) temp_acc.setValue(coord, advected_t);
                    if (advected_f > 0.0001f) fuel_acc.setValue(coord, advected_f);
                    if (std::abs(advected_vx) > 0.0001f) velx_acc.setValue(coord, advected_vx);
                    if (std::abs(advected_vy) > 0.0001f) vely_acc.setValue(coord, advected_vy);
                    if (std::abs(advected_vz) > 0.0001f) velz_acc.setValue(coord, advected_vz);
                }
            }
        }
    }

    // 5) Apply physical dissipation
    float d_factor = std::exp(-params.density_dissipation * dt);
    float t_factor = std::exp(-params.temperature_dissipation * dt);
    float f_factor = std::exp(-params.fuel_dissipation * dt);
    float v_factor = std::exp(-params.velocity_dissipation * dt);

    for (openvdb::FloatGrid::ValueOnIter iter = density_vdb->beginValueOn(); iter; ++iter) {
        iter.setValue(*iter * d_factor);
    }
    for (openvdb::FloatGrid::ValueOnIter iter = temp_vdb->beginValueOn(); iter; ++iter) {
        float current_t = *iter;
        float new_t = params.ambient_temperature + (current_t - params.ambient_temperature) * t_factor;
        iter.setValue(new_t);
    }
    for (openvdb::FloatGrid::ValueOnIter iter = fuel_vdb->beginValueOn(); iter; ++iter) {
        iter.setValue(*iter * f_factor);
    }
    for (openvdb::FloatGrid::ValueOnIter iter = velx_vdb->beginValueOn(); iter; ++iter) {
        iter.setValue(*iter * v_factor);
    }
    for (openvdb::FloatGrid::ValueOnIter iter = vely_vdb->beginValueOn(); iter; ++iter) {
        iter.setValue(*iter * v_factor);
    }
    for (openvdb::FloatGrid::ValueOnIter iter = velz_vdb->beginValueOn(); iter; ++iter) {
        iter.setValue(*iter * v_factor);
    }

    // 6) Sparse pressure projection — make the velocity field divergence-free
    // on the ACTIVE topology only (O(active voxels), not O(nx*ny*nz)). Replaces
    // the old "dense write-back then dense project()" which negated all sparsity
    // (full-grid Poisson + full grid still allocated). See projectSparseVDB.
    if (params.channel_velocity && !params.skip_pressure_projection) {
        projectSparseVDB(velx_vdb, vely_vdb, velz_vdb, params, grid.voxel_size, dt);
    }

    // 7) Write the result to the dense FluidGrid the renderer consumes. The
    // dense arrays are only the rendering hand-off — the solve above stayed
    // sparse, so iterate ACTIVE voxels per channel rather than every cell.
    grid.clear();
    if (!grid.temperature.empty()) {
        std::fill(grid.temperature.begin(), grid.temperature.end(), params.ambient_temperature);
    }
    auto inBounds = [&](const openvdb::Coord& c) {
        return c.x() >= 0 && c.x() < grid.nx &&
               c.y() >= 0 && c.y() < grid.ny &&
               c.z() >= 0 && c.z() < grid.nz;
    };
    if (has_density) {
        for (auto it = density_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (!inBounds(c) || *it <= 0.0001f) continue;
            grid.density[grid.cellIndex(c.x(), c.y(), c.z())] = *it;
        }
    }
    if (has_temp) {
        for (auto it = temp_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (!inBounds(c) || *it <= 0.001f) continue;
            grid.temperature[grid.cellIndex(c.x(), c.y(), c.z())] =
                std::min(*it, params.max_temperature);
        }
    }
    if (has_fuel) {
        for (auto it = fuel_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (!inBounds(c) || *it <= 0.0001f) continue;
            grid.fuel[grid.cellIndex(c.x(), c.y(), c.z())] = *it;
        }
    }
    if (params.channel_velocity) {
        const float vmax = params.max_velocity;
        for (auto it = velx_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (inBounds(c)) grid.vel_x[grid.velXIndex(c.x(), c.y(), c.z())] = std::clamp(*it, -vmax, vmax);
        }
        for (auto it = vely_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (inBounds(c)) grid.vel_y[grid.velYIndex(c.x(), c.y(), c.z())] = std::clamp(*it, -vmax, vmax);
        }
        for (auto it = velz_vdb->cbeginValueOn(); it; ++it) {
            const openvdb::Coord c = it.getCoord();
            if (inBounds(c)) grid.vel_z[grid.velZIndex(c.x(), c.y(), c.z())] = std::clamp(*it, -vmax, vmax);
        }
        // Enforce wall BCs on the dense field (cheap; matches the dense path).
        setWallBcs(grid, params.boundary);
    }
#else
    // Fallback if OpenVDB not enabled
    step(grid, params, dt, forces, time_seconds);
#endif
}

void projectPressure(FluidGrid& grid, const SolverParams& params, float dt) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return;
    }
    project(grid, params, dt);
    setWallBcs(grid, params.boundary);
}

void advectVelocityField(FluidGrid& grid, const SolverParams& params, float dt) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f || !params.channel_velocity) {
        return;
    }
    advectVelocity(grid, dt);
}

} // namespace GridFluid
} // namespace RayTrophiSim
