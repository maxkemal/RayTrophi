#include "ParticleSimulation.h"

#include "GridFluidSolver.h"
#include "globals.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

#ifdef OPENVDB_ENABLED
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#endif

namespace RayTrophiSim {

namespace {

constexpr std::size_t kInvalidParticle = static_cast<std::size_t>(-1);
using SimulationClock = std::chrono::steady_clock;

float elapsedMilliseconds(SimulationClock::time_point start, SimulationClock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - start).count();
}

float safeInverseMass(float mass) {
    return mass > 1e-6f ? 1.0f / mass : 0.0f;
}

inline float lerpf(float a, float b, float t) {
    return a + (b - a) * t;
}

// Voxelize a collider list into the grid.solid[] mask. Called once per Fluid
// domain step BEFORE the APIC solver, so enforceSolidBoundaries and the free-
// surface pressure projection see the up-to-date solid set (movable colliders
// are supported — the mask is fully cleared and rebuilt every step).
//
// Each shape stamps cell centres inside its inflated volume (by `thickness`).
// We test cell-CENTRE rather than full overlap; for sub-voxel-thin colliders
// the user should bump `thickness` to ≥ voxel_size to guarantee a continuous
// solid band. AABB/OBB resolvers come from SceneData (object-bound colliders
// pull their bounds from the scene's mesh hierarchy).
Vec3 closestPointOnTriangle(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 ab = b - a;
    Vec3 ac = c - a;
    Vec3 ap = p - a;
    float d1 = Vec3::dot(ab, ap);
    float d2 = Vec3::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    Vec3 bp = p - b;
    float d3 = Vec3::dot(ab, bp);
    float d4 = Vec3::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }

    Vec3 cp = p - c;
    float d5 = Vec3::dot(ab, cp);
    float d6 = Vec3::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

inline void voxelizeCollidersIntoGrid(
    FluidSim::FluidGrid& grid,
    const std::vector<ParticleColliderDesc>& colliders,
    const std::function<bool(const ParticleColliderDesc&, Vec3&, Vec3&)>& bounds_resolver,
    const std::function<bool(const ParticleColliderDesc&, ParticleColliderOBB&)>& obb_resolver,
    const std::vector<Vec3>* collider_velocities = nullptr) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;

    // Lazily size solid_vel only when a collider is actually moving, so static /
    // no-collider domains never pay the per-cell Vec3 (keeps snapshots lean).
    bool any_moving = false;
    if (collider_velocities) {
        for (const auto& v : *collider_velocities) {
            if (v.x * v.x + v.y * v.y + v.z * v.z > 1e-12f) { any_moving = true; break; }
        }
    }
    if (any_moving && grid.solid_vel.size() != grid.solid.size()) {
        // Zero-fill new elements so cells outside the incrementally-cleared
        // footprint are never garbage (solid_vel is read only at solid cells,
        // but a wholesale GPU upload must still see clean zeros).
        grid.solid_vel.resize(grid.solid.size(), Vec3(0.0f, 0.0f, 0.0f));
    }
    const bool track_vel = (grid.solid_vel.size() == grid.solid.size());

    // ── Static-collider voxelization cache ───────────────────────────────────
    // Stamping a mesh collider is O(footprint cells × closest-point query) and
    // runs every step. A wide high-poly ground/beach collider has a footprint
    // spanning the whole domain in X/Z, so this dominates the frame and starves
    // a GPU solve. Hash the resolved collider set (transforms + per-collider
    // velocity + grid identity); if it matches the last real voxelization, skip
    // the entire clear/stamp — solid[]/solid_vel[] already hold the right mask.
    // Velocity is in the hash so the frame a moving collider STOPS still restamps
    // (vel→0) before the cache kicks in.
    auto hashU = [](uint64_t h, uint64_t v) -> uint64_t {
        return h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    };
    auto hashF = [&](uint64_t h, float f) -> uint64_t {
        uint32_t u = 0; std::memcpy(&u, &f, sizeof(u));
        return hashU(h, static_cast<uint64_t>(u));
    };
    uint64_t sig = 1469598103934665603ULL;
    sig = hashU(sig, static_cast<uint64_t>(nx));
    sig = hashU(sig, static_cast<uint64_t>(ny));
    sig = hashU(sig, static_cast<uint64_t>(nz));
    sig = hashF(sig, grid.voxel_size);
    sig = hashF(sig, grid.origin.x); sig = hashF(sig, grid.origin.y); sig = hashF(sig, grid.origin.z);
    for (std::size_t ci = 0; ci < colliders.size(); ++ci) {
        const auto& c = colliders[ci];
        sig = hashU(sig, c.enabled ? 1ull : 0ull);
        if (!c.enabled) continue;
        sig = hashU(sig, static_cast<uint64_t>(c.source_mode));
        sig = hashF(sig, c.thickness);
        if (collider_velocities && ci < collider_velocities->size()) {
            const Vec3& v = (*collider_velocities)[ci];
            sig = hashF(sig, v.x); sig = hashF(sig, v.y); sig = hashF(sig, v.z);
        }
        switch (c.source_mode) {
            case ParticleColliderSourceMode::PlaneY:
                sig = hashF(sig, c.plane_y);
                break;
            case ParticleColliderSourceMode::Sphere:
                sig = hashF(sig, c.sphere_center.x); sig = hashF(sig, c.sphere_center.y);
                sig = hashF(sig, c.sphere_center.z); sig = hashF(sig, c.sphere_radius);
                break;
            case ParticleColliderSourceMode::Capsule:
                sig = hashF(sig, c.capsule_start.x); sig = hashF(sig, c.capsule_start.y); sig = hashF(sig, c.capsule_start.z);
                sig = hashF(sig, c.capsule_end.x);   sig = hashF(sig, c.capsule_end.y);   sig = hashF(sig, c.capsule_end.z);
                sig = hashF(sig, c.capsule_radius);
                break;
            case ParticleColliderSourceMode::ObjectAABB: {
                Vec3 lo = c.bounds_min, hi = c.bounds_max;
                if (bounds_resolver && !c.source_name.empty()) {
                    Vec3 rlo, rhi;
                    if (bounds_resolver(c, rlo, rhi)) { lo = rlo; hi = rhi; }
                }
                sig = hashF(sig, lo.x); sig = hashF(sig, lo.y); sig = hashF(sig, lo.z);
                sig = hashF(sig, hi.x); sig = hashF(sig, hi.y); sig = hashF(sig, hi.z);
                break;
            }
            default: { // ObjectOBB / MeshSDF / ConvexDecomp / MeshBVH — transform-driven
                ParticleColliderOBB obb;
                if (obb_resolver && obb_resolver(c, obb)) {
                    for (int r = 0; r < 3; ++r)
                        for (int col = 0; col < 4; ++col)
                            sig = hashF(sig, obb.local_to_world.m[r][col]);
                    sig = hashF(sig, obb.local_bounds_min.x); sig = hashF(sig, obb.local_bounds_min.y); sig = hashF(sig, obb.local_bounds_min.z);
                    sig = hashF(sig, obb.local_bounds_max.x); sig = hashF(sig, obb.local_bounds_max.y); sig = hashF(sig, obb.local_bounds_max.z);
                } else {
                    sig = hashU(sig, 0xDEADBEEFull);
                }
                sig = hashF(sig, c.sdf_extents.x); sig = hashF(sig, c.sdf_extents.y); sig = hashF(sig, c.sdf_extents.z);
                break;
            }
        }
    }
    if (grid.collider_voxel_valid && grid.collider_voxel_sig == sig) {
        return; // collider set unchanged — keep last step's solid mask, skip restamp
    }
    // From here the mask is (re)built to match this signature.
    grid.collider_voxel_sig = sig;
    grid.collider_voxel_valid = true;

    // Dirty-region clear: wipe only LAST frame's collider footprint (grid.collider_cur)
    // instead of the whole grid. Clearing prev then stamping cur is correct —
    // vacated cells (prev∖cur) get zeroed, re-occupied cells get re-stamped to 1,
    // and cells outside prev∪cur keep their (already 0) value. A dimension change
    // or first call forces a full clear so no stale solids survive.
    const bool dim_changed = grid.collider_track_dim[0] != nx ||
                             grid.collider_track_dim[1] != ny ||
                             grid.collider_track_dim[2] != nz;
    if (dim_changed) {
        std::fill(grid.solid.begin(), grid.solid.end(), static_cast<uint8_t>(0));
        if (track_vel) std::fill(grid.solid_vel.begin(), grid.solid_vel.end(), Vec3(0.0f, 0.0f, 0.0f));
        grid.collider_track_dim[0] = nx; grid.collider_track_dim[1] = ny; grid.collider_track_dim[2] = nz;
        grid.collider_prev_lo[0] = 0; grid.collider_prev_lo[1] = 0; grid.collider_prev_lo[2] = 0;
        grid.collider_prev_hi[0] = -1; grid.collider_prev_hi[1] = -1; grid.collider_prev_hi[2] = -1;
    } else {
        const int clo0 = std::max(0, grid.collider_cur_lo[0]), chi0 = std::min(nx - 1, grid.collider_cur_hi[0]);
        const int clo1 = std::max(0, grid.collider_cur_lo[1]), chi1 = std::min(ny - 1, grid.collider_cur_hi[1]);
        const int clo2 = std::max(0, grid.collider_cur_lo[2]), chi2 = std::min(nz - 1, grid.collider_cur_hi[2]);
        for (int k = clo2; k <= chi2; ++k)
        for (int j = clo1; j <= chi1; ++j)
        for (int i = clo0; i <= chi0; ++i) {
            const std::size_t ci = grid.cellIndex(i, j, k);
            grid.solid[ci] = 0u;
            if (track_vel) grid.solid_vel[ci] = Vec3(0.0f, 0.0f, 0.0f);
        }
        // Last frame's footprint becomes prev (consumed by computeSolidFaceWeights
        // to reset face weights over the same region).
        grid.collider_prev_lo[0] = grid.collider_cur_lo[0]; grid.collider_prev_lo[1] = grid.collider_cur_lo[1]; grid.collider_prev_lo[2] = grid.collider_cur_lo[2];
        grid.collider_prev_hi[0] = grid.collider_cur_hi[0]; grid.collider_prev_hi[1] = grid.collider_cur_hi[1]; grid.collider_prev_hi[2] = grid.collider_cur_hi[2];
    }

    // This frame's footprint, accumulated by markCell during stamping.
    int cur_lo[3] = { nx, ny, nz };
    int cur_hi[3] = { -1, -1, -1 };
    auto storeCurFootprint = [&]() {
        grid.collider_cur_lo[0] = cur_lo[0]; grid.collider_cur_lo[1] = cur_lo[1]; grid.collider_cur_lo[2] = cur_lo[2];
        grid.collider_cur_hi[0] = cur_hi[0]; grid.collider_cur_hi[1] = cur_hi[1]; grid.collider_cur_hi[2] = cur_hi[2];
    };
    if (colliders.empty()) { storeCurFootprint(); return; }

    const float h = grid.voxel_size;
    const float invH = (h > 0.0f) ? 1.0f / h : 0.0f;

    // Linear velocity of the collider currently being stamped (set per collider
    // below). markCell records it into grid.solid_vel so the advect step can
    // hand a MOVING collider's momentum to the fluid it sweeps over / pushes.
    Vec3 active_collider_vel(0.0f, 0.0f, 0.0f);

    auto cellCenter = [&](int i, int j, int k) {
        return grid.origin + Vec3((i + 0.5f) * h, (j + 0.5f) * h, (k + 0.5f) * h);
    };
    // markCell only WRITES the solid mask — no shared footprint bookkeeping — so
    // the per-cell stamp loops below can run in parallel (each cell is visited
    // once per collider → distinct indices → no race). The dirty-region footprint
    // is grown per-collider from its cell range via expandFootprint (a superset of
    // the actually-marked cells; clearing a few extra zero cells next frame is
    // harmless).
    auto markCell = [&](int i, int j, int k) {
        if (i < 0 || i >= grid.nx || j < 0 || j >= grid.ny || k < 0 || k >= grid.nz) return;
        const std::size_t ci = grid.cellIndex(i, j, k);
        grid.solid[ci] = 1u;
        if (track_vel) grid.solid_vel[ci] = active_collider_vel;
    };
    auto expandFootprint = [&](int i0, int j0, int k0, int i1, int j1, int k1) {
        if (i1 < i0 || j1 < j0 || k1 < k0) return;
        if (i0 < cur_lo[0]) cur_lo[0] = i0; if (i1 > cur_hi[0]) cur_hi[0] = i1;
        if (j0 < cur_lo[1]) cur_lo[1] = j0; if (j1 > cur_hi[1]) cur_hi[1] = j1;
        if (k0 < cur_lo[2]) cur_lo[2] = k0; if (k1 > cur_hi[2]) cur_hi[2] = k1;
    };
    auto cellRange = [&](const Vec3& mn, const Vec3& mx,
                          int& i0, int& j0, int& k0,
                          int& i1, int& j1, int& k1) {
        const Vec3 lo = (mn - grid.origin) * invH;
        const Vec3 hi = (mx - grid.origin) * invH;
        i0 = std::clamp(static_cast<int>(std::floor(lo.x)) - 1, 0, grid.nx - 1);
        j0 = std::clamp(static_cast<int>(std::floor(lo.y)) - 1, 0, grid.ny - 1);
        k0 = std::clamp(static_cast<int>(std::floor(lo.z)) - 1, 0, grid.nz - 1);
        i1 = std::clamp(static_cast<int>(std::ceil(hi.x))  + 1, 0, grid.nx - 1);
        j1 = std::clamp(static_cast<int>(std::ceil(hi.y))  + 1, 0, grid.ny - 1);
        k1 = std::clamp(static_cast<int>(std::ceil(hi.z))  + 1, 0, grid.nz - 1);
    };

    for (std::size_t c_idx = 0; c_idx < colliders.size(); ++c_idx) {
        const auto& c = colliders[c_idx];
        if (!c.enabled) continue;
        const float thick = std::max(0.0f, c.thickness);
        active_collider_vel = (track_vel && collider_velocities && c_idx < collider_velocities->size())
                                  ? (*collider_velocities)[c_idx]
                                  : Vec3(0.0f, 0.0f, 0.0f);

        switch (c.source_mode) {
            case ParticleColliderSourceMode::PlaneY: {
                // Infinite ground plane; solid where cell-centre Y <= plane (+ thick band above).
                const float top = c.plane_y + thick;
                const Vec3 grid_min = grid.origin;
                const float grid_top = grid_min.y + grid.ny * h;
                if (top <= grid_min.y) break;
                const int k0 = 0, k1 = grid.nz - 1;
                const int j_max_raw = static_cast<int>(std::floor((top - grid_min.y) * invH));
                const int j1 = std::clamp(j_max_raw, 0, grid.ny - 1);
                (void)grid_top;
                expandFootprint(0, 0, k0, grid.nx - 1, j1, k1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = 0;  j <= j1; ++j)
                for (int i = 0;  i <  grid.nx; ++i) {
                    if (cellCenter(i, j, k).y < top) markCell(i, j, k);
                }
                break;
            }
            case ParticleColliderSourceMode::Sphere: {
                const float r = std::max(0.0f, c.sphere_radius) + thick;
                const Vec3 sc = c.sphere_center;
                int i0, j0, k0, i1, j1, k1;
                cellRange(sc - Vec3(r, r, r), sc + Vec3(r, r, r), i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);
                const float r2 = r * r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 d = cellCenter(i, j, k) - sc;
                    if (d.x*d.x + d.y*d.y + d.z*d.z < r2) markCell(i, j, k);
                }
                break;
            }
            case ParticleColliderSourceMode::Capsule: {
                const Vec3 a = c.capsule_start;
                const Vec3 b = c.capsule_end;
                const float r = std::max(0.0f, c.capsule_radius) + thick;
                const Vec3 ab = b - a;
                const float ab2 = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
                int i0, j0, k0, i1, j1, k1;
                cellRange(Vec3::min(a, b) - Vec3(r, r, r),
                          Vec3::max(a, b) + Vec3(r, r, r),
                          i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);
                const float r2 = r * r;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 cc = cellCenter(i, j, k);
                    const Vec3 ac = cc - a;
                    float t = 0.0f;
                    if (ab2 > 1e-12f) {
                        t = std::clamp((ac.x*ab.x + ac.y*ab.y + ac.z*ab.z) / ab2, 0.0f, 1.0f);
                    }
                    const Vec3 closest = a + ab * t;
                    const Vec3 d = cc - closest;
                    if (d.x*d.x + d.y*d.y + d.z*d.z < r2) markCell(i, j, k);
                }
                break;
            }
            case ParticleColliderSourceMode::ObjectAABB: {
                Vec3 lo = Vec3::min(c.bounds_min, c.bounds_max);
                Vec3 hi = Vec3::max(c.bounds_min, c.bounds_max);
                if (bounds_resolver && !c.source_name.empty()) {
                    Vec3 rlo, rhi;
                    if (bounds_resolver(c, rlo, rhi)) {
                        lo = Vec3::min(rlo, rhi);
                        hi = Vec3::max(rlo, rhi);
                    }
                }
                lo = lo - Vec3(thick, thick, thick);
                hi = hi + Vec3(thick, thick, thick);
                int i0, j0, k0, i1, j1, k1;
                cellRange(lo, hi, i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 cc = cellCenter(i, j, k);
                    if (cc.x >= lo.x && cc.x <= hi.x &&
                        cc.y >= lo.y && cc.y <= hi.y &&
                        cc.z >= lo.z && cc.z <= hi.z) markCell(i, j, k);
                }
                break;
            }
            case ParticleColliderSourceMode::ObjectOBB: {
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                const Vec3 local_mn = obb.local_bounds_min - Vec3(thick, thick, thick);
                const Vec3 local_mx = obb.local_bounds_max + Vec3(thick, thick, thick);
                // World-AABB of the OBB to limit cell range.
                Vec3 world_mn(1e30f, 1e30f, 1e30f), world_mx(-1e30f, -1e30f, -1e30f);
                for (int n = 0; n < 8; ++n) {
                    const Vec3 corner(
                        (n & 1) ? local_mx.x : local_mn.x,
                        (n & 2) ? local_mx.y : local_mn.y,
                        (n & 4) ? local_mx.z : local_mn.z);
                    const Vec3 w = obb.local_to_world * corner;
                    world_mn = Vec3::min(world_mn, w);
                    world_mx = Vec3::max(world_mx, w);
                }
                int i0, j0, k0, i1, j1, k1;
                cellRange(world_mn, world_mx, i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);
                const Matrix4x4 inv_xform = obb.local_to_world.inverse();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 cc = cellCenter(i, j, k);
                    const Vec3 lc = inv_xform * cc;
                    if (lc.x >= local_mn.x && lc.x <= local_mx.x &&
                        lc.y >= local_mn.y && lc.y <= local_mx.y &&
                        lc.z >= local_mn.z && lc.z <= local_mx.z) markCell(i, j, k);
                }
                break;
            }
            case ParticleColliderSourceMode::ObjectMeshSDF: {
                if (!c.sdf_grid_data || c.sdf_grid_data->empty()) break;
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;

                const float thick = std::max(0.0f, c.thickness);
                const Matrix4x4 inv_xform = obb.local_to_world.inverse();

                float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
                float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
                float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
                float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
                if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                const float local_thick = thick / avg_scale;

                Vec3 cooked_size = c.sdf_extents / 1.3f;
                Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
                Vec3 scale(
                    cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
                    cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
                    cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
                );
                scale.x = std::max(1e-4f, scale.x);
                scale.y = std::max(1e-4f, scale.y);
                scale.z = std::max(1e-4f, scale.z);

                const Vec3 local_mn = (c.sdf_origin * scale) - Vec3(thick);
                const Vec3 local_mx = ((c.sdf_origin + c.sdf_extents) * scale) + Vec3(thick);

                Vec3 world_mn(1e30f, 1e30f, 1e30f), world_mx(-1e30f, -1e30f, -1e30f);
                for (int n = 0; n < 8; ++n) {
                    const Vec3 corner(
                        (n & 1) ? local_mx.x : local_mn.x,
                        (n & 2) ? local_mx.y : local_mn.y,
                        (n & 4) ? local_mx.z : local_mn.z);
                    const Vec3 w = obb.local_to_world * corner;
                    world_mn = Vec3::min(world_mn, w);
                    world_mx = Vec3::max(world_mx, w);
                }

                int i0, j0, k0, i1, j1, k1;
                cellRange(world_mn, world_mx, i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);

                const Vec3 sdf_origin_scaled = c.sdf_origin * scale;
                const Vec3 sdf_extents_scaled = c.sdf_extents * scale;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 cc = cellCenter(i, j, k);
                    const Vec3 lc = inv_xform * cc;
                    const Vec3 local_pos = lc - sdf_origin_scaled;

                    float tx = sdf_extents_scaled.x > 1e-6f ? local_pos.x / sdf_extents_scaled.x : 0.0f;
                    float ty = sdf_extents_scaled.y > 1e-6f ? local_pos.y / sdf_extents_scaled.y : 0.0f;
                    float tz = sdf_extents_scaled.z > 1e-6f ? local_pos.z / sdf_extents_scaled.z : 0.0f;

                    if (tx >= -local_thick / std::max(1e-6f, sdf_extents_scaled.x) && tx <= 1.0f + local_thick / std::max(1e-6f, sdf_extents_scaled.x) &&
                        ty >= -local_thick / std::max(1e-6f, sdf_extents_scaled.y) && ty <= 1.0f + local_thick / std::max(1e-6f, sdf_extents_scaled.y) &&
                        tz >= -local_thick / std::max(1e-6f, sdf_extents_scaled.z) && tz <= 1.0f + local_thick / std::max(1e-6f, sdf_extents_scaled.z)) {

                        int idx_x = std::clamp(static_cast<int>(tx * (c.sdf_nx - 1)), 0, c.sdf_nx - 1);
                        int idx_y = std::clamp(static_cast<int>(ty * (c.sdf_ny - 1)), 0, c.sdf_ny - 1);
                        int idx_z = std::clamp(static_cast<int>(tz * (c.sdf_nz - 1)), 0, c.sdf_nz - 1);
                        std::size_t sdf_idx = static_cast<std::size_t>(idx_z * (c.sdf_nx * c.sdf_ny) + idx_y * c.sdf_nx + idx_x);
                        if (sdf_idx < c.sdf_grid_data->size()) {
                            float dist_cooked = (*c.sdf_grid_data)[sdf_idx];
                            float dist_local = dist_cooked * ((scale.x + scale.y + scale.z) / 3.0f);
                            if (dist_local <= local_thick) {
                                markCell(i, j, k);
                            }
                        }
                    }
                }
                break;
            }
            case ParticleColliderSourceMode::ObjectConvexDecomp: {
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                if (!c.octant_min_cache || c.octant_min_cache->empty()) break;
                const auto& oct_min = *c.octant_min_cache;
                const auto& oct_max = *c.octant_max_cache;
                const auto& oct_active = *c.octant_active_cache;

                const float thick = std::max(0.0f, c.thickness);
                const Matrix4x4 inv_xform = obb.local_to_world.inverse();

                float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
                float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
                float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
                float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
                if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                const float local_thick = thick / avg_scale;

                Vec3 cooked_size = c.sdf_extents / 1.3f;
                if (cooked_size.length_squared() < 1e-6f) {
                    cooked_size = obb.local_bounds_max - obb.local_bounds_min;
                }
                Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
                Vec3 scale(
                    cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
                    cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
                    cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
                );
                scale.x = std::max(1e-4f, scale.x);
                scale.y = std::max(1e-4f, scale.y);
                scale.z = std::max(1e-4f, scale.z);

                for (int o = 0; o < 8; ++o) {
                    if (!oct_active[o]) continue;

                    const Vec3 local_mn = (oct_min[o] * scale) - Vec3(local_thick);
                    const Vec3 local_mx = (oct_max[o] * scale) + Vec3(local_thick);

                    Vec3 world_mn(1e30f, 1e30f, 1e30f), world_mx(-1e30f, -1e30f, -1e30f);
                    for (int n = 0; n < 8; ++n) {
                        const Vec3 corner(
                            (n & 1) ? local_mx.x : local_mn.x,
                            (n & 2) ? local_mx.y : local_mn.y,
                            (n & 4) ? local_mx.z : local_mn.z);
                        const Vec3 w = obb.local_to_world * corner;
                        world_mn = Vec3::min(world_mn, w);
                        world_mx = Vec3::max(world_mx, w);
                    }

                    int i0, j0, k0, i1, j1, k1;
                    cellRange(world_mn, world_mx, i0, j0, k0, i1, j1, k1);
                    expandFootprint(i0, j0, k0, i1, j1, k1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                    for (int k = k0; k <= k1; ++k)
                    for (int j = j0; j <= j1; ++j)
                    for (int i = i0; i <= i1; ++i) {
                        const Vec3 cc = cellCenter(i, j, k);
                        const Vec3 lc = inv_xform * cc;
                        if (lc.x >= local_mn.x && lc.x <= local_mx.x &&
                            lc.y >= local_mn.y && lc.y <= local_mx.y &&
                            lc.z >= local_mn.z && lc.z <= local_mx.z) {
                            markCell(i, j, k);
                        }
                    }
                }
                break;
            }
            case ParticleColliderSourceMode::ObjectMeshBVH: {
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                const ColliderMeshBVH* bvh = c.mesh_bvh_cache.get();
                if (!bvh || bvh->empty()) break;

                const float thick = std::max(0.0f, c.thickness);
                const Matrix4x4 inv_xform = obb.local_to_world.inverse();

                float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
                float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
                float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
                float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
                if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                const float local_thick = thick / avg_scale;

                // Size-ratio scale (current object size vs the size the BVH
                // triangles were cached at). For BVH colliders sdf_extents is 0,
                // so this is ~1 — the query then runs in raw cached-local space.
                Vec3 cooked_size = c.sdf_extents / 1.3f;
                if (cooked_size.length_squared() < 1e-6f) {
                    cooked_size = obb.local_bounds_max - obb.local_bounds_min;
                }
                Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
                Vec3 scale(
                    cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
                    cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
                    cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
                );
                scale.x = std::max(1e-4f, scale.x);
                scale.y = std::max(1e-4f, scale.y);
                scale.z = std::max(1e-4f, scale.z);

                // The BVH stores raw cached-local triangles, so query a cell at
                // lc/scale and compare against the threshold mapped into that same
                // space (local_thick / mean_scale). Exact when scale==1 (the BVH
                // collider norm), a mild approximation under non-uniform scale —
                // the same class the triangle-scaling path used before.
                const float mean_scale = std::max(1e-6f, (scale.x + scale.y + scale.z) / 3.0f);
                const float thr = local_thick / mean_scale;
                const float thr2 = thr * thr;

                // Whole-mesh cell range from the scaled local bounds (+ thickness).
                const Vec3 local_mn = obb.local_bounds_min * scale - Vec3(local_thick);
                const Vec3 local_mx = obb.local_bounds_max * scale + Vec3(local_thick);
                Vec3 world_mn(1e30f, 1e30f, 1e30f), world_mx(-1e30f, -1e30f, -1e30f);
                for (int n = 0; n < 8; ++n) {
                    const Vec3 corner((n & 1) ? local_mx.x : local_mn.x,
                                      (n & 2) ? local_mx.y : local_mn.y,
                                      (n & 4) ? local_mx.z : local_mn.z);
                    const Vec3 w = obb.local_to_world * corner;
                    world_mn = Vec3::min(world_mn, w);
                    world_mx = Vec3::max(world_mx, w);
                }
                int i0, j0, k0, i1, j1, k1;
                cellRange(world_mn, world_mx, i0, j0, k0, i1, j1, k1);
                expandFootprint(i0, j0, k0, i1, j1, k1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
                for (int k = k0; k <= k1; ++k)
                for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const Vec3 cc = cellCenter(i, j, k);
                    const Vec3 lc = inv_xform * cc;
                    const Vec3 q(lc.x / scale.x, lc.y / scale.y, lc.z / scale.z);
                    Vec3 closest;
                    if (bvh->closestDistanceSquared(q, closest) <= thr2) {
                        markCell(i, j, k);
                    }
                }
                break;
            }
        }
    }

    storeCurFootprint();
}

// Fill the MAC-face fractional open-weights (FluidGrid::u/v/w_weight) used by the
// variational pressure projection. Two passes:
//   1. Binary: every face bordering a solid cell → fully blocked (0). Covers ALL
//      collider types (incl. mesh/SDF) at cell accuracy, matching the solid mask.
//   2. Analytic refine: for every collider with a sub-grid inside test —
//      sphere/capsule/AABB/OBB (closed form), ObjectMeshSDF (trilinear cooked
//      distance), and ObjectConvexDecomp (per-octant boxes) — super-sample each
//      face inside the collider's bbox and write the true open fraction (min with
//      the binary result so overlapping solids still win). → sub-grid-accurate
//      boundaries (no blocky leaking) for mesh colliders too. PlaneY (axis-aligned,
//      placed exactly by the binary pass) and ObjectMeshBVH (no acceleration
//      structure → too costly to super-sample) fall back to the binary pass.
inline void computeSolidFaceWeights(
    FluidSim::FluidGrid& grid,
    const std::vector<ParticleColliderDesc>& colliders,
    const std::function<bool(const ParticleColliderDesc&, Vec3&, Vec3&)>& bounds_resolver,
    const std::function<bool(const ParticleColliderDesc&, ParticleColliderOBB&)>& obb_resolver) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    if (nx <= 0 || ny <= 0 || nz <= 0) return;

    const std::size_t exp_u = static_cast<std::size_t>(nx + 1) * ny * nz;
    const std::size_t exp_v = static_cast<std::size_t>(nx) * (ny + 1) * nz;
    const std::size_t exp_w = static_cast<std::size_t>(nx) * ny * (nz + 1);
    const bool sized = grid.u_weight.size() == exp_u &&
                       grid.v_weight.size() == exp_v &&
                       grid.w_weight.size() == exp_w;

    // Reset the MAC-face weights back to fully-open (255) over a cell-AABB's
    // faces only. Faces a collider vacated (in prev) and faces outside the
    // refine footprint that stay open (min with 255 is a no-op) are handled by
    // resetting prev∪cur — everything else keeps last frame's value.
    auto resetWeightFaces = [&](const int* lo, const int* hi) {
        if (hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2]) return;
        const int ci0 = std::max(0, lo[0]), ci1 = std::min(nx - 1, hi[0]); // cell range
        const int cj0 = std::max(0, lo[1]), cj1 = std::min(ny - 1, hi[1]);
        const int ck0 = std::max(0, lo[2]), ck1 = std::min(nz - 1, hi[2]);
        if (ci1 < ci0 || cj1 < cj0 || ck1 < ck0) return;
        for (int k = ck0; k <= ck1; ++k)
            for (int j = cj0; j <= cj1; ++j)
                for (int i = ci0; i <= ci1 + 1; ++i) // X-faces span [ci0, ci1+1]
                    grid.u_weight[grid.velXIndex(i, j, k)] = 255u;
        for (int k = ck0; k <= ck1; ++k)
            for (int j = cj0; j <= cj1 + 1; ++j)       // Y-faces span [cj0, cj1+1]
                for (int i = ci0; i <= ci1; ++i)
                    grid.v_weight[grid.velYIndex(i, j, k)] = 255u;
        for (int k = ck0; k <= ck1 + 1; ++k)           // Z-faces span [ck0, ck1+1]
            for (int j = cj0; j <= cj1; ++j)
                for (int i = ci0; i <= ci1; ++i)
                    grid.w_weight[grid.velZIndex(i, j, k)] = 255u;
    };
    // prev ∪ cur collider footprint (cell AABB), the region to restore to open.
    // Expanded by a small margin: the analytic refine can set a boundary face
    // (whose cell centre is just outside the stamped solid) to <255 up to ~1 cell
    // beyond the footprint, so the reset must cover it. Over-resetting is harmless
    // (those faces are re-written below); under-resetting would leave stale values.
    auto footprintUnion = [&](int* outlo, int* outhi) -> bool {
        bool have = false;
        auto add = [&](const int* lo, const int* hi) {
            if (hi[0] < lo[0] || hi[1] < lo[1] || hi[2] < lo[2]) return;
            if (!have) { for (int a = 0; a < 3; ++a) { outlo[a] = lo[a]; outhi[a] = hi[a]; } have = true; }
            else { for (int a = 0; a < 3; ++a) { outlo[a] = std::min(outlo[a], lo[a]); outhi[a] = std::max(outhi[a], hi[a]); } }
        };
        add(grid.collider_prev_lo, grid.collider_prev_hi);
        add(grid.collider_cur_lo, grid.collider_cur_hi);
        if (have) { for (int a = 0; a < 3; ++a) { outlo[a] -= 2; outhi[a] += 2; } }
        return have;
    };

    auto fullOpenInit = [&]() {
        std::fill(grid.u_weight.begin(), grid.u_weight.end(), static_cast<uint8_t>(255));
        std::fill(grid.v_weight.begin(), grid.v_weight.end(), static_cast<uint8_t>(255));
        std::fill(grid.w_weight.begin(), grid.w_weight.end(), static_cast<uint8_t>(255));
    };

    // A full open-init is required when the arrays were just (re)allocated OR the
    // weights aren't known globally-valid (first variational frame / toggled off
    // mid-sim); otherwise the cheap incremental footprint reset suffices.
    const bool need_full = !sized || !grid.collider_weights_init;

    // Static-collider weight cache: when the collider set is unchanged since the
    // weights were last built (same voxelize signature) AND they are still valid,
    // the fractional face weights are identical — skip the whole reset + binary +
    // super-sample pass. This is the variational analogue of the solid-mask voxelize
    // cache, so a wide static collider (beach) pays the weight pass once, not every
    // step. (voxelizeCollidersIntoGrid ran just before and refreshed collider_voxel_sig.)
    if (!need_full && grid.collider_weights_sig == grid.collider_voxel_sig) {
        return;
    }

    // Colliders removed this frame: just leave everything open.
    if (colliders.empty()) {
        if (sized) {
            if (need_full) fullOpenInit();
            else { int rlo[3], rhi[3]; if (footprintUnion(rlo, rhi)) resetWeightFaces(rlo, rhi); }
            grid.collider_weights_init = true;
            grid.collider_weights_sig = grid.collider_voxel_sig;
        }
        return;
    }

    if (grid.u_weight.size() != exp_u) grid.u_weight.resize(exp_u);
    if (grid.v_weight.size() != exp_v) grid.v_weight.resize(exp_v);
    if (grid.w_weight.size() != exp_w) grid.w_weight.resize(exp_w);

    if (need_full) {
        fullOpenInit();
    } else {
        int rlo[3], rhi[3];
        if (footprintUnion(rlo, rhi)) resetWeightFaces(rlo, rhi);
    }
    grid.collider_weights_init = true;
    grid.collider_weights_sig = grid.collider_voxel_sig; // weights now match this collider set

    const float h = grid.voxel_size;
    const float invH = (h > 0.0f) ? 1.0f / h : 0.0f;

    // 1. Binary pass — faces of every solid cell are fully blocked. Only the
    //    current collider footprint can contain solids, so iterate that bbox
    //    instead of the whole grid.
    const int blo0 = std::max(0, grid.collider_cur_lo[0]), bhi0 = std::min(nx - 1, grid.collider_cur_hi[0]);
    const int blo1 = std::max(0, grid.collider_cur_lo[1]), bhi1 = std::min(ny - 1, grid.collider_cur_hi[1]);
    const int blo2 = std::max(0, grid.collider_cur_lo[2]), bhi2 = std::min(nz - 1, grid.collider_cur_hi[2]);
    for (int k = blo2; k <= bhi2; ++k)
    for (int j = blo1; j <= bhi1; ++j)
    for (int i = blo0; i <= bhi0; ++i) {
        if (!grid.solid[grid.cellIndex(i, j, k)]) continue;
        grid.u_weight[grid.velXIndex(i,     j, k)] = 0u;
        grid.u_weight[grid.velXIndex(i + 1, j, k)] = 0u;
        grid.v_weight[grid.velYIndex(i, j,     k)] = 0u;
        grid.v_weight[grid.velYIndex(i, j + 1, k)] = 0u;
        grid.w_weight[grid.velZIndex(i, j, k)]     = 0u;
        grid.w_weight[grid.velZIndex(i, j, k + 1)] = 0u;
    }

    // 2. Analytic refine. Resolve each closed-form collider's geometry into a
    //    point-inside test + world AABB (mirrors voxelizeCollidersIntoGrid so the
    //    weights agree with the solid mask).
    struct AnalyticSolid {
        enum Type { Sphere, Capsule, AABB, OBB, SDF } type;
        Vec3 sc; float sr = 0.0f;                 // sphere
        Vec3 ca, cb; float cr = 0.0f; float cab2 = 0.0f; // capsule
        Vec3 lo, hi;                              // aabb (also OBB local bounds)
        Matrix4x4 inv_xform;                      // OBB/SDF world->local
        Vec3 wmin, wmax;                          // world AABB
        // SDF (ObjectMeshSDF): trilinear cooked-distance test. Mirrors the
        // voxelizer + resolveSDFCollision criterion (dist_cooked*mean_scale <=
        // local_thick) so the fractional face weights agree with the solid mask.
        const std::vector<float>* sdf = nullptr;
        int snx = 0, sny = 0, snz = 0;
        Vec3 sdf_origin_scaled, sdf_extents_scaled;
        float sdf_dist_scale = 1.0f;              // cooked distance -> local space
        float sdf_thick = 0.0f;                   // local_thick threshold

        bool inside(const Vec3& p) const {
            switch (type) {
                case Sphere: { const Vec3 d = p - sc; return d.x*d.x + d.y*d.y + d.z*d.z < sr*sr; }
                case Capsule: {
                    const Vec3 ac = p - ca; float t = 0.0f;
                    if (cab2 > 1e-12f) {
                        const Vec3 ab = cb - ca;
                        t = std::clamp((ac.x*ab.x + ac.y*ab.y + ac.z*ab.z) / cab2, 0.0f, 1.0f);
                    }
                    const Vec3 cl = ca + (cb - ca) * t;
                    const Vec3 d = p - cl; return d.x*d.x + d.y*d.y + d.z*d.z < cr*cr;
                }
                case AABB: return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z;
                case OBB: { const Vec3 l = inv_xform * p; return l.x >= lo.x && l.x <= hi.x && l.y >= lo.y && l.y <= hi.y && l.z >= lo.z && l.z <= hi.z; }
                case SDF: {
                    if (!sdf || sdf->empty()) return false;
                    const Vec3 lp = (inv_xform * p) - sdf_origin_scaled;
                    const float tx = sdf_extents_scaled.x > 1e-6f ? lp.x / sdf_extents_scaled.x : 0.0f;
                    const float ty = sdf_extents_scaled.y > 1e-6f ? lp.y / sdf_extents_scaled.y : 0.0f;
                    const float tz = sdf_extents_scaled.z > 1e-6f ? lp.z / sdf_extents_scaled.z : 0.0f;
                    if (tx < 0.0f || tx > 1.0f || ty < 0.0f || ty > 1.0f || tz < 0.0f || tz > 1.0f) return false;
                    auto smp = [&](int x, int y, int z) -> float {
                        x = std::clamp(x, 0, snx - 1); y = std::clamp(y, 0, sny - 1); z = std::clamp(z, 0, snz - 1);
                        const std::size_t idx = static_cast<std::size_t>(z * (snx * sny) + y * snx + x);
                        return idx < sdf->size() ? (*sdf)[idx] : 0.0f;
                    };
                    const float fx = tx * (snx - 1), fy = ty * (sny - 1), fz = tz * (snz - 1);
                    const int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, snx - 1);
                    const int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, sny - 1);
                    const int iz = std::clamp(static_cast<int>(std::floor(fz)), 0, snz - 1);
                    const int ix1 = std::clamp(ix + 1, 0, snx - 1);
                    const int iy1 = std::clamp(iy + 1, 0, sny - 1);
                    const int iz1 = std::clamp(iz + 1, 0, snz - 1);
                    const float dx = fx - ix, dy = fy - iy, dz = fz - iz;
                    const float v00 = smp(ix, iy, iz)  * (1.0f - dx) + smp(ix1, iy, iz)  * dx;
                    const float v10 = smp(ix, iy1, iz) * (1.0f - dx) + smp(ix1, iy1, iz) * dx;
                    const float v01 = smp(ix, iy, iz1) * (1.0f - dx) + smp(ix1, iy, iz1) * dx;
                    const float v11 = smp(ix, iy1, iz1)* (1.0f - dx) + smp(ix1, iy1, iz1)* dx;
                    const float v0 = v00 * (1.0f - dy) + v10 * dy;
                    const float v1 = v01 * (1.0f - dy) + v11 * dy;
                    const float dist_cooked = v0 * (1.0f - dz) + v1 * dz;
                    return dist_cooked * sdf_dist_scale <= sdf_thick;
                }
            }
            return false;
        }
    };

    std::vector<AnalyticSolid> solids;
    solids.reserve(colliders.size());
    Vec3 union_min(1e30f, 1e30f, 1e30f), union_max(-1e30f, -1e30f, -1e30f);
    for (const auto& c : colliders) {
        if (!c.enabled) continue;
        const float thick = std::max(0.0f, c.thickness);
        AnalyticSolid s;
        bool ok = false;
        switch (c.source_mode) {
            case ParticleColliderSourceMode::Sphere: {
                s.type = AnalyticSolid::Sphere;
                s.sc = c.sphere_center; s.sr = std::max(0.0f, c.sphere_radius) + thick;
                s.wmin = s.sc - Vec3(s.sr); s.wmax = s.sc + Vec3(s.sr); ok = s.sr > 0.0f;
                break;
            }
            case ParticleColliderSourceMode::Capsule: {
                s.type = AnalyticSolid::Capsule;
                s.ca = c.capsule_start; s.cb = c.capsule_end;
                s.cr = std::max(0.0f, c.capsule_radius) + thick;
                const Vec3 ab = s.cb - s.ca; s.cab2 = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
                s.wmin = Vec3::min(s.ca, s.cb) - Vec3(s.cr);
                s.wmax = Vec3::max(s.ca, s.cb) + Vec3(s.cr); ok = s.cr > 0.0f;
                break;
            }
            case ParticleColliderSourceMode::ObjectAABB: {
                s.type = AnalyticSolid::AABB;
                Vec3 lo = c.bounds_min, hi = c.bounds_max;
                if (bounds_resolver && !c.source_name.empty()) bounds_resolver(c, lo, hi);
                s.lo = Vec3::min(lo, hi) - Vec3(thick);
                s.hi = Vec3::max(lo, hi) + Vec3(thick);
                s.wmin = s.lo; s.wmax = s.hi; ok = true;
                break;
            }
            case ParticleColliderSourceMode::ObjectOBB: {
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                s.type = AnalyticSolid::OBB;
                s.lo = obb.local_bounds_min - Vec3(thick);
                s.hi = obb.local_bounds_max + Vec3(thick);
                s.inv_xform = obb.local_to_world.inverse();
                for (int n = 0; n < 8; ++n) {
                    const Vec3 corner((n & 1) ? s.hi.x : s.lo.x,
                                      (n & 2) ? s.hi.y : s.lo.y,
                                      (n & 4) ? s.hi.z : s.lo.z);
                    const Vec3 wpt = obb.local_to_world * corner;
                    if (n == 0) { s.wmin = wpt; s.wmax = wpt; }
                    else { s.wmin = Vec3::min(s.wmin, wpt); s.wmax = Vec3::max(s.wmax, wpt); }
                }
                ok = true;
                break;
            }
            case ParticleColliderSourceMode::ObjectMeshSDF: {
                if (!c.sdf_grid_data || c.sdf_grid_data->empty()) break;
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                s.type = AnalyticSolid::SDF;
                s.inv_xform = obb.local_to_world.inverse();
                const float sx = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
                const float sy = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
                const float sz = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
                float avg_scale = (sx + sy + sz) / 3.0f; if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                // cooked_size = sdf_extents / 1.3 (the 0.15 pad per side baked at cook
                // time in rebuildSDFColliderAsync); keep in sync if that pad changes.
                const Vec3 cooked_size = c.sdf_extents / 1.3f;
                const Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
                Vec3 scl(cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
                         cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
                         cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f);
                scl.x = std::max(1e-4f, scl.x); scl.y = std::max(1e-4f, scl.y); scl.z = std::max(1e-4f, scl.z);
                s.sdf = c.sdf_grid_data.get();
                s.snx = c.sdf_nx; s.sny = c.sdf_ny; s.snz = c.sdf_nz;
                s.sdf_origin_scaled = c.sdf_origin * scl;
                s.sdf_extents_scaled = c.sdf_extents * scl;
                s.sdf_dist_scale = (scl.x + scl.y + scl.z) / 3.0f;
                s.sdf_thick = thick / avg_scale;
                const Vec3 local_mn = s.sdf_origin_scaled - Vec3(thick);
                const Vec3 local_mx = (c.sdf_origin + c.sdf_extents) * scl + Vec3(thick);
                for (int n = 0; n < 8; ++n) {
                    const Vec3 corner((n & 1) ? local_mx.x : local_mn.x,
                                      (n & 2) ? local_mx.y : local_mn.y,
                                      (n & 4) ? local_mx.z : local_mn.z);
                    const Vec3 wpt = obb.local_to_world * corner;
                    if (n == 0) { s.wmin = wpt; s.wmax = wpt; }
                    else { s.wmin = Vec3::min(s.wmin, wpt); s.wmax = Vec3::max(s.wmax, wpt); }
                }
                ok = true;
                break;
            }
            case ParticleColliderSourceMode::ObjectConvexDecomp: {
                // Each active octant is an axis-aligned box in object-local space →
                // reuse the OBB inside test (local bounds + world->local transform).
                // Pushes multiple solids, so it updates the union inline and skips
                // the single post-switch push.
                ParticleColliderOBB obb;
                if (!obb_resolver || !obb_resolver(c, obb)) break;
                if (!c.octant_min_cache || c.octant_min_cache->empty() ||
                    !c.octant_max_cache || !c.octant_active_cache) break;
                const auto& omin = *c.octant_min_cache;
                const auto& omax = *c.octant_max_cache;
                const auto& oact = *c.octant_active_cache;
                const Matrix4x4 invx = obb.local_to_world.inverse();
                const float sx = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
                const float sy = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
                const float sz = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
                float avg_scale = (sx + sy + sz) / 3.0f; if (avg_scale <= 1e-6f) avg_scale = 1.0f;
                const float local_thick = thick / avg_scale;
                Vec3 cooked_size = c.sdf_extents / 1.3f;
                if (cooked_size.length_squared() < 1e-6f) cooked_size = obb.local_bounds_max - obb.local_bounds_min;
                const Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
                Vec3 scl(cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
                         cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
                         cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f);
                scl.x = std::max(1e-4f, scl.x); scl.y = std::max(1e-4f, scl.y); scl.z = std::max(1e-4f, scl.z);
                const int oct_n = std::min<int>(8, static_cast<int>(omin.size()));
                for (int o = 0; o < oct_n; ++o) {
                    if (o >= static_cast<int>(oact.size()) || !oact[o]) continue;
                    if (o >= static_cast<int>(omax.size())) continue;
                    AnalyticSolid os;
                    os.type = AnalyticSolid::OBB;
                    os.lo = omin[o] * scl - Vec3(local_thick);
                    os.hi = omax[o] * scl + Vec3(local_thick);
                    os.inv_xform = invx;
                    for (int n = 0; n < 8; ++n) {
                        const Vec3 corner((n & 1) ? os.hi.x : os.lo.x,
                                          (n & 2) ? os.hi.y : os.lo.y,
                                          (n & 4) ? os.hi.z : os.lo.z);
                        const Vec3 wpt = obb.local_to_world * corner;
                        if (n == 0) { os.wmin = wpt; os.wmax = wpt; }
                        else { os.wmin = Vec3::min(os.wmin, wpt); os.wmax = Vec3::max(os.wmax, wpt); }
                    }
                    solids.push_back(os);
                    union_min = Vec3::min(union_min, os.wmin);
                    union_max = Vec3::max(union_max, os.wmax);
                }
                continue; // multi-push handled inline
            }
            default: break; // PlaneY + mesh BVH → binary pass only
        }
        if (!ok) continue;
        solids.push_back(s);
        union_min = Vec3::min(union_min, s.wmin);
        union_max = Vec3::max(union_max, s.wmax);
    }
    if (solids.empty()) return;

    // Cell range covering all analytic solids (one cell margin so boundary faces
    // are included). Faces outside this keep the binary result.
    auto clampi = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    const int ci0 = clampi(static_cast<int>(std::floor((union_min.x - grid.origin.x) * invH)) - 1, 0, nx);
    const int cj0 = clampi(static_cast<int>(std::floor((union_min.y - grid.origin.y) * invH)) - 1, 0, ny);
    const int ck0 = clampi(static_cast<int>(std::floor((union_min.z - grid.origin.z) * invH)) - 1, 0, nz);
    const int ci1 = clampi(static_cast<int>(std::ceil ((union_max.x - grid.origin.x) * invH)) + 1, 0, nx);
    const int cj1 = clampi(static_cast<int>(std::ceil ((union_max.y - grid.origin.y) * invH)) + 1, 0, ny);
    const int ck1 = clampi(static_cast<int>(std::ceil ((union_max.z - grid.origin.z) * invH)) + 1, 0, nz);

    constexpr int NS = 4;               // 4x4 sub-samples per face
    constexpr float invNS = 1.0f / NS;
    auto insideAny = [&](const Vec3& p) {
        for (const auto& s : solids) {
            if (p.x < s.wmin.x || p.x > s.wmax.x || p.y < s.wmin.y || p.y > s.wmax.y ||
                p.z < s.wmin.z || p.z > s.wmax.z) continue;
            if (s.inside(p)) return true;
        }
        return false;
    };
    auto openFrac = [&](const Vec3& corner, const Vec3& du, const Vec3& dv) {
        int open = 0;
        for (int a = 0; a < NS; ++a)
        for (int b = 0; b < NS; ++b) {
            const Vec3 p = corner + du * ((a + 0.5f) * invNS) + dv * ((b + 0.5f) * invNS);
            if (!insideAny(p)) ++open;
        }
        return static_cast<uint8_t>((open * 255) / (NS * NS));
    };
    const Vec3 ex(h, 0.0f, 0.0f), ey(0.0f, h, 0.0f), ez(0.0f, 0.0f, h);

    // X-faces (y-z spanning plane at x = origin + i*h).
    for (int k = ck0; k < ck1; ++k)
    for (int j = cj0; j < cj1; ++j)
    for (int i = ci0; i <= ci1; ++i) {
        if (i > nx || j >= ny || k >= nz) continue;
        const Vec3 corner = grid.origin + Vec3(i * h, j * h, k * h);
        const uint8_t wgt = openFrac(corner, ey, ez);
        uint8_t& cur = grid.u_weight[grid.velXIndex(i, j, k)];
        cur = std::min(cur, wgt);
    }
    // Y-faces.
    for (int k = ck0; k < ck1; ++k)
    for (int j = cj0; j <= cj1; ++j)
    for (int i = ci0; i < ci1; ++i) {
        if (i >= nx || j > ny || k >= nz) continue;
        const Vec3 corner = grid.origin + Vec3(i * h, j * h, k * h);
        const uint8_t wgt = openFrac(corner, ex, ez);
        uint8_t& cur = grid.v_weight[grid.velYIndex(i, j, k)];
        cur = std::min(cur, wgt);
    }
    // Z-faces.
    for (int k = ck0; k <= ck1; ++k)
    for (int j = cj0; j < cj1; ++j)
    for (int i = ci0; i < ci1; ++i) {
        if (i >= nx || j >= ny || k > nz) continue;
        const Vec3 corner = grid.origin + Vec3(i * h, j * h, k * h);
        const uint8_t wgt = openFrac(corner, ex, ey);
        uint8_t& cur = grid.w_weight[grid.velZIndex(i, j, k)];
        cur = std::min(cur, wgt);
    }
}

bool finiteParticleDesc(const ParticleSpawnDesc& desc) {
    return std::isfinite(desc.position.x) &&
           std::isfinite(desc.position.y) &&
           std::isfinite(desc.position.z) &&
           std::isfinite(desc.velocity.x) &&
           std::isfinite(desc.velocity.y) &&
           std::isfinite(desc.velocity.z) &&
           std::isfinite(desc.lifetime_seconds) &&
           std::isfinite(desc.mass);
}

struct GridProjectionGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int boundary = 0;
    float voxel_size = 1.0f;
    float dt = 0.0f;
    float sor_omega = 1.7f;
    int iterations = 40;
    int parity = 0;
    // MGPCG extras (must mirror the device GridProjectionConstants tail).
    float density_correction = 0.0f; // 0 = off (SOR path leaves it default)
    int   particles_per_cell = 0;
    // Variational solid coupling flag (must mirror device GridProjectionConstants).
    // 0 = binary solid faces (current GPU behaviour); 1 = fractional weights +
    // solid velocity (Stage 1 GPU variational, wired through the extra buffers).
    // Kept here so host/device struct layouts stay byte-identical (the dispatch
    // size check compares against sizeof(device GridProjectionConstants)).
    int   variational = 0;
    int   gfm_active = 0;
};

struct GridMGGpuConstants {
    int fine_nx = 0;
    int fine_ny = 0;
    int fine_nz = 0;
    int coarse_nx = 0;
    int coarse_ny = 0;
    int coarse_nz = 0;
    float omega = 0.8f;
    int parity = 0;
};

struct GridScalarAdvectionGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float voxel_size = 1.0f;
    float dt = 0.0f;
};

using GridVelocityAdvectionGpuConstants = GridScalarAdvectionGpuConstants;

struct GridVelocityDissipationGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float factor = 1.0f;
    float max_velocity = 0.0f;
};

struct FluidDensitySplatGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int particle_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
    float particle_density = 1.0f;
};

struct FluidParticleIntegrateGpuConstants {
    int particle_count = 0;
    float dt = 0.0f;
    float gravity_x = 0.0f;
    float gravity_y = 0.0f;
    float gravity_z = 0.0f;
    float container_velocity_x = 0.0f;
    float container_velocity_y = 0.0f;
    float container_velocity_z = 0.0f;
    float max_velocity = 0.0f;
};

struct FluidP2GGpuConstants {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int particle_count = 0;
    int component = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
};

// Must match FluidG2PConstants in SimulationComputeCuda.cu (field order + types).
struct FluidG2PGpuConstants {
    int   nx = 0;
    int   ny = 0;
    int   nz = 0;
    int   particle_count = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
    float flip_blend = 0.0f;
    float apic_blend = 1.0f;
    float internal_friction = 0.0f;
    float max_velocity = 50.0f;
    float dt = 0.0f;
    int   has_flip_snapshot = 0;
    int   use_solid_flip_limiter = 0;
    // APIC affine post-scale + clamp. WITHOUT these the reconstructed C matrix
    // grows unbounded across steps → P2G injects runaway velocity → fluid
    // explodes/sprays after a few seconds. Mirrors the CPU gridToParticle path
    // (affine_blend = apic_blend*affine_damping, then clampAffine(max_affine)).
    float affine_damping = 0.98f;
    float max_affine = 80.0f;
};

// Free-surface SOR reuses GridProjectionGpuConstants (same layout).
// kernel = "sim_fluid_free_surface_sor", buffers = [pressure, divergence, fluid_mask]

int boundaryToGpu(GridFluid::Boundary boundary) {
    switch (boundary) {
        case GridFluid::Boundary::Closed: return 1;
        case GridFluid::Boundary::Periodic: return 2;
        case GridFluid::Boundary::Open:
        default: return 0;
    }
}

void updateFluidDensityStats(SimulationGridDomainState& state) {
    state.active_density_cells = 0;
    state.max_density = 0.0f;
    for (float value : state.grid.density) {
        if (value > 1e-5f) {
            ++state.active_density_cells;
            state.max_density = std::max(state.max_density, value);
        }
    }
}

void splatFluidDensityCPU(SimulationGridDomainState& state, const Fluid::APICSolverParams& params) {
    auto& grid = state.grid;
    std::fill(grid.density.begin(), grid.density.end(), 0.0f);
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0.0f ||
        state.particles.empty() || grid.density.empty()) {
        updateFluidDensityStats(state);
        return;
    }

    const float inv_h = 1.0f / grid.voxel_size;
    const float particle_density =
        1.0f / static_cast<float>(std::max(1, params.particles_per_cell));

    for (const Vec3& p : state.particles.position) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }

        const Vec3 local = (p - grid.origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
        const int i0 = static_cast<int>(std::floor(local.x));
        const int j0 = static_cast<int>(std::floor(local.y));
        const int k0 = static_cast<int>(std::floor(local.z));
        const float fx = local.x - static_cast<float>(i0);
        const float fy = local.y - static_cast<float>(j0);
        const float fz = local.z - static_cast<float>(k0);

        for (int dz = 0; dz <= 1; ++dz) {
            const int k = k0 + dz;
            if (k < 0 || k >= grid.nz) continue;
            const float wz = dz ? fz : (1.0f - fz);
            for (int dy = 0; dy <= 1; ++dy) {
                const int j = j0 + dy;
                if (j < 0 || j >= grid.ny) continue;
                const float wy = dy ? fy : (1.0f - fy);
                for (int dx = 0; dx <= 1; ++dx) {
                    const int i = i0 + dx;
                    if (i < 0 || i >= grid.nx) continue;
                    const float wx = dx ? fx : (1.0f - fx);
                    grid.density[grid.cellIndex(i, j, k)] += particle_density * wx * wy * wz;
                }
            }
        }
    }

    updateFluidDensityStats(state);
}

bool ensureGpuFluidParticleBuffers(SimulationGridDomainState& state,
                                   SimulationComputeContext* compute,
                                   SimulationGridDomainComputeBuffers& gpu_buffers) {
    const std::size_t particle_count = state.particles.size();
    if (!compute || particle_count == 0) {
        return false;
    }

    const ComputeBufferUsage usage = ComputeBufferUsage::Storage |
                                     ComputeBufferUsage::Upload |
                                     ComputeBufferUsage::Download |
                                     ComputeBufferUsage::ReadWrite;
    const std::size_t position_bytes = particle_count * sizeof(Vec3);
    const std::size_t velocity_bytes = particle_count * sizeof(Vec3);
    const std::size_t affine_bytes = particle_count * sizeof(Fluid::AffineC);
    if (!gpu_buffers.fluid_positions.valid() ||
        gpu_buffers.fluid_positions.backend != compute->backendType() ||
        gpu_buffers.fluid_particle_capacity < particle_count) {
        if (gpu_buffers.fluid_positions.valid()) {
            compute->destroyBuffer(gpu_buffers.fluid_positions);
            gpu_buffers.fluid_positions = {};
        }
        if (gpu_buffers.fluid_velocities.valid()) {
            compute->destroyBuffer(gpu_buffers.fluid_velocities);
            gpu_buffers.fluid_velocities = {};
        }
        if (gpu_buffers.fluid_affine.valid()) {
            compute->destroyBuffer(gpu_buffers.fluid_affine);
            gpu_buffers.fluid_affine = {};
        }
        ComputeBufferDesc desc;
        desc.debug_name = "FluidParticlePositions";
        desc.size_bytes = position_bytes;
        desc.usage = usage;
        gpu_buffers.fluid_positions = compute->createBuffer(desc);
        desc.debug_name = "FluidParticleVelocities";
        desc.size_bytes = velocity_bytes;
        gpu_buffers.fluid_velocities = compute->createBuffer(desc);
        desc.debug_name = "FluidParticleAffine";
        desc.size_bytes = affine_bytes;
        gpu_buffers.fluid_affine = compute->createBuffer(desc);
        gpu_buffers.fluid_particle_capacity =
            (gpu_buffers.fluid_positions.valid() &&
             gpu_buffers.fluid_velocities.valid() &&
             gpu_buffers.fluid_affine.valid())
                ? particle_count
                : 0;
    }

    if (!gpu_buffers.fluid_positions.valid() ||
        !gpu_buffers.fluid_velocities.valid() ||
        !gpu_buffers.fluid_affine.valid()) {
        return false;
    }

    return compute->uploadBuffer(gpu_buffers.fluid_positions,
                                 state.particles.position.data(),
                                 position_bytes) &&
           compute->uploadBuffer(gpu_buffers.fluid_velocities,
                                 state.particles.velocity.data(),
                                 velocity_bytes) &&
           compute->uploadBuffer(gpu_buffers.fluid_affine,
                                 state.particles.affine.data(),
                                 affine_bytes);
}

bool runGpuFluidParticleIntegrateForces(SimulationGridDomainState& state,
                                        const Fluid::APICSolverParams& params,
                                        const Vec3& container_velocity_delta,
                                        float dt,
                                        SimulationComputeContext* compute,
                                        SimulationGridDomainComputeBuffers& gpu_buffers) {
    const std::size_t particle_count = state.particles.size();
    if (!compute || !compute->supportsDispatch() || particle_count == 0 || dt <= 0.0f) {
        return false;
    }
    if (!ensureGpuFluidParticleBuffers(state, compute, gpu_buffers)) {
        return false;
    }

    FluidParticleIntegrateGpuConstants constants;
    constants.particle_count = static_cast<int>(std::min<std::size_t>(
        particle_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    constants.dt = dt;
    constants.gravity_x = params.gravity.x;
    constants.gravity_y = params.gravity.y;
    constants.gravity_z = params.gravity.z;
    constants.container_velocity_x = container_velocity_delta.x;
    constants.container_velocity_y = container_velocity_delta.y;
    constants.container_velocity_z = container_velocity_delta.z;
    constants.max_velocity = params.max_velocity;

    ComputeBufferHandle buffers[1] = { gpu_buffers.fluid_velocities };
    ComputeDispatch cmd;
    cmd.kernel = "sim_fluid_particle_integrate_forces";
    cmd.buffers = buffers;
    cmd.buffer_count = 1;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    constexpr uint32_t threads = 256;
    cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + threads - 1u) / threads;
    bool ok = compute->dispatch(cmd);
    compute->synchronize();
    ok = ok && compute->downloadBuffer(gpu_buffers.fluid_velocities,
                                       state.particles.velocity.data(),
                                       particle_count * sizeof(Vec3));
    return ok;
}

// Build the GPU pressure-solve fluid mask. Tri-state float encoding consumed by
// the CG kernels:
//   mask < -0.5  → SOLID cell (Neumann wall; build_diag won't count it)
//   mask >  0.5  → FLUID cell, and the value IS the per-cell particle COUNT
//                  (used by residual_init's Bridson density correction)
//   else (==0)   → AIR (Dirichlet p=0 ghost)
// Mirrors the CPU PCG's CELL_SOLID/CELL_FLUID/CELL_AIR classification + count.
// `mask` is reused across calls (grown as needed) by the caller.
void buildFluidMaskFromParticles(const FluidSim::FluidGrid& grid,
                                 const Fluid::FluidParticles& particles,
                                 std::vector<float>& mask) {
    const std::size_t cells = grid.getCellCount();
    if (mask.size() < cells) mask.assign(cells, 0.0f);
    else std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(cells), 0.0f);
    // Stamp solids first (-1.0). Particles landing in solid cells are ignored.
    for (std::size_t c = 0; c < cells; ++c)
        if (grid.solid[c]) mask[c] = -1.0f;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float invH = grid.voxel_size > 1e-6f ? 1.0f / grid.voxel_size : 0.0f;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const Vec3 gp = (particles.position[p] - grid.origin) * invH;
        const int i = static_cast<int>(std::floor(gp.x));
        const int j = static_cast<int>(std::floor(gp.y));
        const int k = static_cast<int>(std::floor(gp.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        if (!grid.solid[c]) mask[c] += 1.0f; // accumulate count (air 0 → 1,2,…)
    }
}

void enforceGridSolidFaceBoundaries(FluidSim::FluidGrid& grid) {
    if (grid.solid.empty() || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        return;
    }
    if (!std::any_of(grid.solid.begin(), grid.solid.end(), [](uint8_t value) { return value != 0; })) {
        return;
    }

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                if ((i > 0 && grid.isSolid(i - 1, j, k)) ||
                    (i < grid.nx && grid.isSolid(i, j, k))) {
                    grid.vel_x[grid.velXIndex(i, j, k)] = 0.0f;
                }
            }
        }
    }

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                if ((j > 0 && grid.isSolid(i, j - 1, k)) ||
                    (j < grid.ny && grid.isSolid(i, j, k))) {
                    grid.vel_y[grid.velYIndex(i, j, k)] = 0.0f;
                }
            }
        }
    }

    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                if ((k > 0 && grid.isSolid(i, j, k - 1)) ||
                    (k < grid.nz && grid.isSolid(i, j, k))) {
                    grid.vel_z[grid.velZIndex(i, j, k)] = 0.0f;
                }
            }
        }
    }
}

bool runGpuFluidP2G(SimulationGridDomainState& state,
                    SimulationComputeContext* compute,
                    SimulationGridDomainComputeBuffers& gpu_buffers) {
    auto& grid = state.grid;
    const std::size_t particle_count = state.particles.size();
    if (!compute || !compute->supportsDispatch() || particle_count == 0 ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0.0f ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }
    if (!ensureGpuFluidParticleBuffers(state, compute, gpu_buffers)) {
        return false;
    }

    ComputeBufferHandle velocity_fields[3] = {
        gpu_buffers.vel_x,
        gpu_buffers.vel_y,
        gpu_buffers.vel_z
    };
    ComputeBufferHandle weight_fields[3] = {
        gpu_buffers.temperature,
        gpu_buffers.fuel,
        gpu_buffers.scratch_scalar
    };
    for (int comp = 0; comp < 3; ++comp) {
        if (!velocity_fields[comp].valid() || !weight_fields[comp].valid()) {
            return false;
        }
    }

    FluidP2GGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.particle_count = static_cast<int>(std::min<std::size_t>(
        particle_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    constants.origin_x = grid.origin.x;
    constants.origin_y = grid.origin.y;
    constants.origin_z = grid.origin.z;
    constants.voxel_size = grid.voxel_size;

    constexpr uint32_t threads = 256;
    bool ok = true;
    for (int comp = 0; comp < 3 && ok; ++comp) {
        constants.component = comp;
        const uint32_t field_count = static_cast<uint32_t>(
            comp == 0 ? grid.vel_x.size() : (comp == 1 ? grid.vel_y.size() : grid.vel_z.size()));

        ComputeBufferHandle clear_velocity[1] = { velocity_fields[comp] };
        ComputeDispatch cmd;
        cmd.kernel = "sim_fluid_clear_float";
        cmd.buffers = clear_velocity;
        cmd.buffer_count = 1;
        cmd.constants = &constants;
        cmd.constants_size = sizeof(constants);
        cmd.groups.groups_x = (field_count + threads - 1u) / threads;
        ok = compute->dispatch(cmd);

        ComputeBufferHandle clear_weight[1] = { weight_fields[comp] };
        cmd.buffers = clear_weight;
        ok = ok && compute->dispatch(cmd);

        ComputeBufferHandle scatter_buffers[5] = {
            gpu_buffers.fluid_positions,
            gpu_buffers.fluid_velocities,
            gpu_buffers.fluid_affine,
            velocity_fields[comp],
            weight_fields[comp]
        };
        cmd.kernel = "sim_fluid_p2g_scatter";
        cmd.buffers = scatter_buffers;
        cmd.buffer_count = 5;
        cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + threads - 1u) / threads;
        ok = ok && compute->dispatch(cmd);

        cmd.kernel = "sim_fluid_p2g_normalize";
        cmd.groups.groups_x = (field_count + threads - 1u) / threads;
        ok = ok && compute->dispatch(cmd);
    }
    compute->synchronize();

    ok = ok &&
         compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    return ok;
}

bool runGpuFluidDensitySplat(SimulationGridDomainState& state,
                             const Fluid::APICSolverParams& params,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers& gpu_buffers) {
    auto& grid = state.grid;
    const std::size_t particle_count = state.particles.size();
    if (!compute || !compute->supportsDispatch() || particle_count == 0 ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || grid.voxel_size <= 0.0f ||
        grid.density.empty()) {
        return false;
    }

    if (!ensureGpuFluidParticleBuffers(state, compute, gpu_buffers) ||
        !gpu_buffers.density.valid()) {
        return false;
    }

    FluidDensitySplatGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.particle_count = static_cast<int>(std::min<std::size_t>(
        particle_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    constants.origin_x = grid.origin.x;
    constants.origin_y = grid.origin.y;
    constants.origin_z = grid.origin.z;
    constants.voxel_size = grid.voxel_size;
    constants.particle_density =
        1.0f / static_cast<float>(std::max(1, params.particles_per_cell));

    constexpr uint32_t threads = 256;
    const uint32_t cell_count = static_cast<uint32_t>(std::min<std::size_t>(
        grid.getCellCount(), static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())));

    ComputeBufferHandle clear_buffers[1] = { gpu_buffers.density };
    ComputeDispatch cmd;
    cmd.kernel = "sim_fluid_density_clear";
    cmd.buffers = clear_buffers;
    cmd.buffer_count = 1;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    cmd.groups.groups_x = (cell_count + threads - 1u) / threads;
    bool ok = compute->dispatch(cmd);

    ComputeBufferHandle splat_buffers[2] = {
        gpu_buffers.fluid_positions,
        gpu_buffers.density
    };
    cmd.kernel = "sim_fluid_density_splat";
    cmd.buffers = splat_buffers;
    cmd.buffer_count = 2;
    cmd.groups.groups_x = (static_cast<uint32_t>(constants.particle_count) + threads - 1u) / threads;
    ok = ok && compute->dispatch(cmd);
    compute->synchronize();

    ok = ok && compute->downloadBuffer(gpu_buffers.density,
                                       grid.density.data(),
                                       grid.density.size() * sizeof(float));
    if (ok) {
        updateFluidDensityStats(state);
    }
    return ok;
}

// GPU free-surface pressure projection (SOR variant).
// Uploads the pre-computed fluid_mask (float, 0=air / 1=fluid), runs
// divergence + SOR iterations + gradient subtraction, then downloads
// updated vel_x/y/z. Leaves pressure on GPU (no download needed by caller).
bool runGpuFluidFreeSurfacePressure(SimulationGridDomainState& state,
                                    const Fluid::APICSolverParams& fluid_params,
                                    float dt,
                                    SimulationComputeContext* compute,
                                    SimulationGridDomainComputeBuffers& gpu_buffers,
                                    const std::vector<float>& fluid_mask_cpu) {
    auto& grid = state.grid;
    if (!compute || !compute->supportsDispatch() || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty() ||
        grid.pressure.size() != grid.getCellCount() ||
        grid.divergence.size() != grid.getCellCount()) {
        return false;
    }
    if (!gpu_buffers.vel_x.valid() || !gpu_buffers.vel_y.valid() ||
        !gpu_buffers.vel_z.valid() || !gpu_buffers.pressure.valid() ||
        !gpu_buffers.divergence.valid() || !gpu_buffers.fluid_mask.valid()) {
        return false;
    }

    const uint32_t threads = 256;
    const uint32_t cell_count  = static_cast<uint32_t>(grid.getCellCount());
    const uint32_t max_faces   = static_cast<uint32_t>(
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}));

    // Upload velocity (already reflects boundary + viscosity from CPU), mask, zero pressure.
    bool ok = compute->uploadBuffer(gpu_buffers.vel_x,    grid.vel_x.data(),    grid.vel_x.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_y,    grid.vel_y.data(),    grid.vel_y.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_z,    grid.vel_z.data(),    grid.vel_z.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.fluid_mask, fluid_mask_cpu.data(), fluid_mask_cpu.size() * sizeof(float));
    if (!ok) return false;

    // Zero pressure each step (no warm-start — free-surface topology changes).
    std::fill(grid.pressure.begin(), grid.pressure.end(), 0.0f);
    ok = compute->uploadBuffer(gpu_buffers.pressure, grid.pressure.data(), grid.pressure.size() * sizeof(float));
    if (!ok) return false;

    GridProjectionGpuConstants c;
    c.nx         = grid.nx;
    c.ny         = grid.ny;
    c.nz         = grid.nz;
    // 0=open (walls = p=0 outflow), 1=closed (solid walls), 2=periodic.
    // Mirror the domain wall mode so Open drains on the GPU like the CPU path.
    c.boundary   = (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Open)     ? 0
                 : (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic) ? 2
                 : 1;
    c.voxel_size = grid.voxel_size;
    c.dt         = dt;
    // SOR convergence needs omega ~1.7-1.8 and 3-4x more iterations than
    // PCG+MIC(0). The CPU PCG path ignores sor_omega, so its default (1.25)
    // is a placeholder — use 1.75 for the GPU SOR path.
    c.sor_omega  = 1.75f;
    c.iterations = std::max(1, fluid_params.pressure_iterations) * 3;

    ComputeBufferHandle proj_bufs[5] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.pressure, gpu_buffers.divergence
    };
    ComputeDispatch cmd;
    cmd.constants      = &c;
    cmd.constants_size = sizeof(c);

    const bool use_cuda_fluid_projection =
        compute->backendType() == ComputeBackendType::CUDA;
    ComputeBufferHandle fluid_divergence_bufs[5] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.fluid_mask, gpu_buffers.divergence
    };
    ComputeBufferHandle fluid_gradient_bufs[5] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.pressure, gpu_buffers.fluid_mask
    };

    // Compute divergence.
    cmd.kernel        = use_cuda_fluid_projection ? "sim_fluid_divergence" : "sim_grid_divergence";
    cmd.buffers       = use_cuda_fluid_projection ? fluid_divergence_bufs : proj_bufs;
    cmd.buffer_count  = 5;
    cmd.groups.groups_x = (cell_count + threads - 1u) / threads;
    ok = compute->dispatch(cmd);

    // Free-surface SOR: pressure, divergence, fluid_mask.
    ComputeBufferHandle sor_bufs[3] = {
        gpu_buffers.pressure, gpu_buffers.divergence, gpu_buffers.fluid_mask
    };
    cmd.kernel       = "sim_fluid_free_surface_sor";
    cmd.buffers      = sor_bufs;
    cmd.buffer_count = 3;
    cmd.groups.groups_x = (cell_count + threads - 1u) / threads;
    for (int iter = 0; ok && iter < c.iterations; ++iter) {
        c.parity = 0; ok = compute->dispatch(cmd);
        c.parity = 1; ok = ok && compute->dispatch(cmd);
    }

    // Subtract pressure gradient from velocity.
    cmd.kernel       = use_cuda_fluid_projection ? "sim_fluid_subtract_gradient" : "sim_grid_subtract_gradient";
    cmd.buffers      = use_cuda_fluid_projection ? fluid_gradient_bufs : proj_bufs;
    cmd.buffer_count = 5;
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    ok = ok && compute->dispatch(cmd);
    compute->synchronize();

    // Download updated velocities for CPU boundary re-enforcement and G2P.
    ok = ok &&
         compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    return ok;
}

// GPU free-surface pressure projection — MGPCG Layer A: Jacobi-preconditioned
// Conjugate Gradient. Solves the SAME free-surface Poisson system as
// runGpuFluidFreeSurfacePressure (SOR) and the CPU PCG+MIC(0), reusing the
// identical divergence + subtract-gradient kernels, so it can be dropped into
// the same pipeline slot. Returns false (clean fallback to SOR / CPU PCG) when
// the CG scratch buffers are not allocated. The dot products accumulate in
// double on-device (block partials) and are summed on the host — matching the
// CPU PCG's double dot products, which is essential for CG stability at scale.
//
// Algorithm (Shewchuk / Bridson ch.5, preconditioned):
//   r0 = b = -div*h*h/dt (fluid cells); p0 = 0; z = M^-1 r; s = z; sigma = r.z
//   loop: As = A s; alpha = sigma/(s.As); p += alpha s; r -= alpha As;
//         z = M^-1 r; sigma_new = r.z; beta = sigma_new/sigma; s = z + beta s;
//         sigma = sigma_new; until sigma_new <= tol*sigma0 or max_iter.
// The CG scalars alpha/beta are carried into the kernels via the (otherwise
// unused on this path) GridProjectionGpuConstants::sor_omega field.
bool runGpuFluidMGPCGPressure(SimulationGridDomainState& state,
                              const Fluid::APICSolverParams& fluid_params,
                              float dt,
                              SimulationComputeContext* compute,
                              SimulationGridDomainComputeBuffers& gpu_buffers,
                              const std::vector<float>& fluid_mask_cpu,
                              Fluid::APICSolverStats* mgpcg_stats = nullptr) {
    auto& grid = state.grid;
    if (!compute || !compute->supportsDispatch() || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty() ||
        grid.pressure.size() != grid.getCellCount() ||
        grid.divergence.size() != grid.getCellCount()) {
        return false;
    }
    if (!gpu_buffers.vel_x.valid() || !gpu_buffers.vel_y.valid() ||
        !gpu_buffers.vel_z.valid() || !gpu_buffers.pressure.valid() ||
        !gpu_buffers.divergence.valid() || !gpu_buffers.fluid_mask.valid()) {
        return false;
    }
    // CG scratch missing → signal fallback to the SOR / CPU PCG path.
    if (!gpu_buffers.cg_residual.valid() || !gpu_buffers.cg_z.valid() ||
        !gpu_buffers.cg_search.valid()   || !gpu_buffers.cg_As.valid() ||
        !gpu_buffers.cg_diag.valid()     || !gpu_buffers.cg_partials.valid()) {
        return false;
    }

    const uint32_t threads     = 256; // must match the device block size (256)
    const uint32_t cell_count  = static_cast<uint32_t>(grid.getCellCount());
    const uint32_t cell_groups = (cell_count + threads - 1u) / threads;
    const uint32_t max_faces   = static_cast<uint32_t>(
        std::max({grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size()}));

    // Upload velocity (already reflects boundary + viscosity from CPU) + mask.
    bool ok = compute->uploadBuffer(gpu_buffers.vel_x,      grid.vel_x.data(),      grid.vel_x.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_y,      grid.vel_y.data(),      grid.vel_y.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_z,      grid.vel_z.data(),      grid.vel_z.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.fluid_mask, fluid_mask_cpu.data(),  fluid_mask_cpu.size() * sizeof(float));
    if (!ok) return false;

    const bool is_variational = fluid_params.variational_solids &&
                                (grid.u_weight.size() == grid.vel_x.size()) &&
                                (grid.v_weight.size() == grid.vel_y.size()) &&
                                (grid.w_weight.size() == grid.vel_z.size()) &&
                                gpu_buffers.var_u_weight.valid() &&
                                gpu_buffers.var_v_weight.valid() &&
                                gpu_buffers.var_w_weight.valid() &&
                                gpu_buffers.var_svx.valid() &&
                                gpu_buffers.var_svy.valid() &&
                                gpu_buffers.var_svz.valid();

    const bool is_gfm = fluid_params.ghost_fluid_surface &&
                        (grid.fluid_phi.size() == cell_count) &&
                        gpu_buffers.var_fluid_phi.valid();

    if (is_variational) {
        // Convert and upload weights
        std::vector<float> uw_float(grid.u_weight.size());
        std::vector<float> vw_float(grid.v_weight.size());
        std::vector<float> ww_float(grid.w_weight.size());
        for (std::size_t i = 0; i < grid.u_weight.size(); ++i) uw_float[i] = FluidSim::FluidGrid::weightToFloat(grid.u_weight[i]);
        for (std::size_t i = 0; i < grid.v_weight.size(); ++i) vw_float[i] = FluidSim::FluidGrid::weightToFloat(grid.v_weight[i]);
        for (std::size_t i = 0; i < grid.w_weight.size(); ++i) ww_float[i] = FluidSim::FluidGrid::weightToFloat(grid.w_weight[i]);

        ok = ok && compute->uploadBuffer(gpu_buffers.var_u_weight, uw_float.data(), uw_float.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_v_weight, vw_float.data(), vw_float.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_w_weight, ww_float.data(), ww_float.size() * sizeof(float));

        // Deinterleave and upload solid velocities
        std::vector<float> svx(cell_count, 0.0f);
        std::vector<float> svy(cell_count, 0.0f);
        std::vector<float> svz(cell_count, 0.0f);
        if (grid.solid_vel.size() == cell_count) {
            for (std::size_t i = 0; i < cell_count; ++i) {
                svx[i] = grid.solid_vel[i].x;
                svy[i] = grid.solid_vel[i].y;
                svz[i] = grid.solid_vel[i].z;
            }
        }
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svx, svx.data(), svx.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svy, svy.data(), svy.size() * sizeof(float));
        ok = ok && compute->uploadBuffer(gpu_buffers.var_svz, svz.data(), svz.size() * sizeof(float));
    }

    if (is_gfm) {
        ok = ok && compute->uploadBuffer(gpu_buffers.var_fluid_phi, grid.fluid_phi.data(), grid.fluid_phi.size() * sizeof(float));
    }
    if (!ok) return false;

    GridProjectionGpuConstants c;
    c.nx         = grid.nx;
    c.ny         = grid.ny;
    c.nz         = grid.nz;
    // Mirror the domain wall mode onto the GPU MGPCG projection (0=open,
    // 1=closed, 2=periodic) so an Open domain treats its bounding walls as
    // p=0 outflow on the GPU exactly like the CPU free-surface solver. Without
    // this the GPU path always sealed the walls regardless of the UI setting.
    c.boundary   = (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Open)     ? 0
                 : (fluid_params.boundary == Fluid::APICSolverParams::BoundaryMode::Periodic) ? 2
                 : 1;
    c.voxel_size = grid.voxel_size;
    c.dt         = dt;
    c.sor_omega  = 0.0f; // reused to carry CG alpha/beta per dispatch
    c.iterations = 0;
    c.parity     = 0;
    // Bridson density-targeted projection (mask carries per-cell particle count).
    c.density_correction = fluid_params.density_correction;
    c.particles_per_cell = fluid_params.particles_per_cell;
    c.variational        = is_variational ? 1 : 0;
    c.gfm_active         = is_gfm ? 1 : 0;

    ComputeDispatch cmd;
    cmd.constants      = &c;
    cmd.constants_size = sizeof(c);

    auto dispatch1 = [&](const char* kernel, ComputeBufferHandle* bufs, int n, uint32_t groups) -> bool {
        cmd.kernel          = kernel;
        cmd.buffers         = bufs;
        cmd.buffer_count    = n;
        cmd.constants       = &c;
        cmd.constants_size  = sizeof(c);
        cmd.groups.groups_x = groups;
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };

    // divergence = (div u)/h  (positive; the residual_init kernel negates it).
    // CUDA has a fluid-mask-aware path that mirrors the CPU free-surface solver:
    // solid neighbours contribute zero flux, and later gradient subtraction
    // updates only fluid/air faces while zeroing true solid faces.
    const bool use_cuda_fluid_projection =
        compute->backendType() == ComputeBackendType::CUDA;
    ComputeBufferHandle proj_bufs[5] = {
        gpu_buffers.vel_x, gpu_buffers.vel_y, gpu_buffers.vel_z,
        gpu_buffers.pressure, gpu_buffers.divergence
    };
    ComputeBufferHandle fluid_divergence_bufs[11];
    fluid_divergence_bufs[0] = gpu_buffers.vel_x;
    fluid_divergence_bufs[1] = gpu_buffers.vel_y;
    fluid_divergence_bufs[2] = gpu_buffers.vel_z;
    fluid_divergence_bufs[3] = gpu_buffers.fluid_mask;
    fluid_divergence_bufs[4] = gpu_buffers.divergence;
    if (is_variational) {
        fluid_divergence_bufs[5] = gpu_buffers.var_u_weight;
        fluid_divergence_bufs[6] = gpu_buffers.var_v_weight;
        fluid_divergence_bufs[7] = gpu_buffers.var_w_weight;
        fluid_divergence_bufs[8] = gpu_buffers.var_svx;
        fluid_divergence_bufs[9] = gpu_buffers.var_svy;
        fluid_divergence_bufs[10] = gpu_buffers.var_svz;
    }
    const int div_buf_count = is_variational ? 11 : 5;

    ok = dispatch1(use_cuda_fluid_projection ? "sim_fluid_divergence" : "sim_grid_divergence",
                   use_cuda_fluid_projection ? fluid_divergence_bufs : proj_bufs,
                   use_cuda_fluid_projection ? div_buf_count : 5,
                   cell_groups);

    // diag = #in-bounds neighbours (fluid rows; 0 elsewhere).
    {
        ComputeBufferHandle b[6];
        b[0] = gpu_buffers.fluid_mask;
        b[1] = gpu_buffers.cg_diag;
        int diag_buf_count = 2;
        if (is_variational) {
            b[2] = gpu_buffers.var_u_weight;
            b[3] = gpu_buffers.var_v_weight;
            b[4] = gpu_buffers.var_w_weight;
            diag_buf_count = 5;
        }
        if (is_gfm) {
            b[diag_buf_count] = gpu_buffers.var_fluid_phi;
            diag_buf_count++;
        }
        ok = ok && dispatch1("sim_fluid_cg_build_diag", b, diag_buf_count, cell_groups);
    }
    // r = -div*h*h/dt at fluid cells; pressure reset to 0.
    { ComputeBufferHandle b[4] = { gpu_buffers.divergence, gpu_buffers.fluid_mask,
                                   gpu_buffers.cg_residual, gpu_buffers.pressure };
      ok = ok && dispatch1("sim_fluid_cg_residual_init", b, 4, cell_groups); }
    if (!ok) return false;

    const bool use_cuda_fused_reductions = compute->backendType() == ComputeBackendType::CUDA;

    // Reduction helpers: dispatch a reducing kernel, sync, download the per-block
    // double partials, sum on host. Function-static host buffer avoids per-call
    // heap churn (grows with the block count as needed).
    const uint32_t dot_blocks = cell_groups;
    static std::vector<double> cg_partials_host;
    if (cg_partials_host.size() < dot_blocks) cg_partials_host.assign(dot_blocks, 0.0);
    float dot_ms = 0.0f;
    int dot_count = 0;
    bool used_multigrid = false;
    auto finishReduction = [&](const auto& dot_begin, double& out) -> bool {
        compute->synchronize();
        if (!compute->downloadBuffer(gpu_buffers.cg_partials, cg_partials_host.data(),
                                     dot_blocks * sizeof(double))) return false;
        double sum = 0.0;
        for (uint32_t bi = 0; bi < dot_blocks; ++bi) sum += cg_partials_host[bi];
        out = sum;
        dot_ms += elapsedMilliseconds(dot_begin, SimulationClock::now());
        ++dot_count;
        return true;
    };
    auto dotProduct = [&](ComputeBufferHandle x, ComputeBufferHandle y, double& out) -> bool {
        const auto dot_begin = SimulationClock::now();
        ComputeBufferHandle b[3] = { x, y, gpu_buffers.cg_partials };
        if (!dispatch1("sim_fluid_cg_dot", b, 3, dot_blocks)) return false;
        return finishReduction(dot_begin, out);
    };

    auto mgLevelValid = [](const SimulationGridDomainMGLevelBuffers& level) -> bool {
        return level.nx > 0 && level.ny > 0 && level.nz > 0 &&
               level.mask.valid() && level.rhs.valid() &&
               level.z.valid() && level.diag.valid();
    };
    auto dispatchCellKernel = [&](const char* kernel,
                                  ComputeBufferHandle* bufs,
                                  int n,
                                  GridProjectionGpuConstants& pc) -> bool {
        cmd.kernel          = kernel;
        cmd.buffers         = bufs;
        cmd.buffer_count    = n;
        cmd.constants       = &pc;
        cmd.constants_size  = sizeof(pc);
        const uint32_t groups = (static_cast<uint32_t>(pc.nx) *
                                 static_cast<uint32_t>(pc.ny) *
                                 static_cast<uint32_t>(pc.nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto zeroCells = [&](ComputeBufferHandle values, int nx, int ny, int nz) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        ComputeBufferHandle b[1] = { values };
        return dispatchCellKernel("sim_fluid_mg_zero", b, 1, pc);
    };
    auto buildDiag = [&](ComputeBufferHandle mask, ComputeBufferHandle diag,
                         int nx, int ny, int nz) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        ComputeBufferHandle b[2] = { mask, diag };
        return dispatchCellKernel("sim_fluid_cg_build_diag", b, 2, pc);
    };
    auto smoothLevel = [&](ComputeBufferHandle rhs, ComputeBufferHandle mask,
                           ComputeBufferHandle diag, ComputeBufferHandle z,
                           int nx, int ny, int nz, int sweeps) -> bool {
        GridProjectionGpuConstants pc = c;
        pc.nx = nx;
        pc.ny = ny;
        pc.nz = nz;
        pc.sor_omega = 0.8f;
        ComputeBufferHandle b[4] = { rhs, mask, diag, z };
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            pc.parity = 0;
            if (!dispatchCellKernel("sim_fluid_mg_rbgs", b, 4, pc)) return false;
            pc.parity = 1;
            if (!dispatchCellKernel("sim_fluid_mg_rbgs", b, 4, pc)) return false;
        }
        return true;
    };
    auto restrictLevel = [&](ComputeBufferHandle fine_rhs, ComputeBufferHandle fine_mask,
                             int fine_nx, int fine_ny, int fine_nz,
                             SimulationGridDomainMGLevelBuffers& coarse) -> bool {
        GridMGGpuConstants mgc;
        mgc.fine_nx = fine_nx;
        mgc.fine_ny = fine_ny;
        mgc.fine_nz = fine_nz;
        mgc.coarse_nx = coarse.nx;
        mgc.coarse_ny = coarse.ny;
        mgc.coarse_nz = coarse.nz;
        ComputeBufferHandle b[4] = { fine_rhs, fine_mask, coarse.rhs, coarse.mask };
        cmd.kernel = "sim_fluid_mg_restrict";
        cmd.buffers = b;
        cmd.buffer_count = 4;
        cmd.constants = &mgc;
        cmd.constants_size = sizeof(mgc);
        const uint32_t groups =
            (static_cast<uint32_t>(coarse.nx) *
             static_cast<uint32_t>(coarse.ny) *
             static_cast<uint32_t>(coarse.nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto prolongateAdd = [&](const SimulationGridDomainMGLevelBuffers& coarse,
                             ComputeBufferHandle fine_mask,
                             ComputeBufferHandle fine_z,
                             int fine_nx, int fine_ny, int fine_nz) -> bool {
        GridMGGpuConstants mgc;
        mgc.fine_nx = fine_nx;
        mgc.fine_ny = fine_ny;
        mgc.fine_nz = fine_nz;
        mgc.coarse_nx = coarse.nx;
        mgc.coarse_ny = coarse.ny;
        mgc.coarse_nz = coarse.nz;
        ComputeBufferHandle b[3] = { coarse.z, fine_mask, fine_z };
        cmd.kernel = "sim_fluid_mg_prolongate_add";
        cmd.buffers = b;
        cmd.buffer_count = 3;
        cmd.constants = &mgc;
        cmd.constants_size = sizeof(mgc);
        const uint32_t groups =
            (static_cast<uint32_t>(fine_nx) *
             static_cast<uint32_t>(fine_ny) *
             static_cast<uint32_t>(fine_nz) + threads - 1u) / threads;
        cmd.groups.groups_x = std::max(1u, groups);
        cmd.groups.groups_y = 1;
        cmd.groups.groups_z = 1;
        return compute->dispatch(cmd);
    };
    auto mgPrecondition = [&]() -> bool {
        if (is_variational || !fluid_params.pressure_multigrid_preconditioner ||
            !use_cuda_fused_reductions ||
            gpu_buffers.mg_levels.empty()) {
            return false;
        }
        for (const auto& level : gpu_buffers.mg_levels) {
            if (!mgLevelValid(level)) return false;
        }

        ComputeBufferHandle fine_rhs = gpu_buffers.cg_residual;
        ComputeBufferHandle fine_mask = gpu_buffers.fluid_mask;
        int fine_nx = grid.nx;
        int fine_ny = grid.ny;
        int fine_nz = grid.nz;
        for (auto& level : gpu_buffers.mg_levels) {
            if (!restrictLevel(fine_rhs, fine_mask, fine_nx, fine_ny, fine_nz, level)) return false;
            if (!buildDiag(level.mask, level.diag, level.nx, level.ny, level.nz)) return false;
            if (!zeroCells(level.z, level.nx, level.ny, level.nz)) return false;
            fine_rhs = level.rhs;
            fine_mask = level.mask;
            fine_nx = level.nx;
            fine_ny = level.ny;
            fine_nz = level.nz;
        }

        auto& coarsest = gpu_buffers.mg_levels.back();
        if (!smoothLevel(coarsest.rhs, coarsest.mask, coarsest.diag, coarsest.z,
                         coarsest.nx, coarsest.ny, coarsest.nz, 12)) return false;

        for (int li = static_cast<int>(gpu_buffers.mg_levels.size()) - 2; li >= 0; --li) {
            auto& coarse = gpu_buffers.mg_levels[static_cast<std::size_t>(li + 1)];
            auto& fine = gpu_buffers.mg_levels[static_cast<std::size_t>(li)];
            if (!prolongateAdd(coarse, fine.mask, fine.z, fine.nx, fine.ny, fine.nz)) return false;
            if (!smoothLevel(fine.rhs, fine.mask, fine.diag, fine.z,
                             fine.nx, fine.ny, fine.nz, 2)) return false;
        }

        auto& first_coarse = gpu_buffers.mg_levels.front();
        if (!zeroCells(gpu_buffers.cg_z, grid.nx, grid.ny, grid.nz)) return false;
        if (!prolongateAdd(first_coarse, gpu_buffers.fluid_mask, gpu_buffers.cg_z,
                           grid.nx, grid.ny, grid.nz)) return false;
        if (!smoothLevel(gpu_buffers.cg_residual, gpu_buffers.fluid_mask, gpu_buffers.cg_diag,
                         gpu_buffers.cg_z, grid.nx, grid.ny, grid.nz, 2)) return false;
        return true;
    };
    auto jacobiAndDot = [&](double& out) -> bool {
        if (mgPrecondition()) {
            used_multigrid = true;
            return dotProduct(gpu_buffers.cg_residual, gpu_buffers.cg_z, out);
        }
        if (use_cuda_fused_reductions) {
            const auto dot_begin = SimulationClock::now();
            ComputeBufferHandle b[4] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag,
                                         gpu_buffers.cg_z, gpu_buffers.cg_partials };
            if (!dispatch1("sim_fluid_cg_jacobi_dot", b, 4, dot_blocks)) return false;
            return finishReduction(dot_begin, out);
        }
        { ComputeBufferHandle b[3] = { gpu_buffers.cg_residual, gpu_buffers.cg_diag, gpu_buffers.cg_z };
          if (!dispatch1("sim_fluid_cg_jacobi", b, 3, cell_groups)) return false; }
        return dotProduct(gpu_buffers.cg_residual, gpu_buffers.cg_z, out);
    };
    auto spmvAndDot = [&](double& out) -> bool {
        if (use_cuda_fused_reductions) {
            const auto dot_begin = SimulationClock::now();
            ComputeBufferHandle b[8] = { gpu_buffers.cg_search, gpu_buffers.fluid_mask,
                                         gpu_buffers.cg_diag, gpu_buffers.cg_As,
                                         gpu_buffers.cg_partials };
            int spmv_buf_count = 5;
            if (is_variational) {
                b[5] = gpu_buffers.var_u_weight;
                b[6] = gpu_buffers.var_v_weight;
                b[7] = gpu_buffers.var_w_weight;
                spmv_buf_count = 8;
            }
            if (!dispatch1("sim_fluid_cg_spmv_dot", b, spmv_buf_count, dot_blocks)) return false;
            return finishReduction(dot_begin, out);
        }
        {
            ComputeBufferHandle b[7] = { gpu_buffers.cg_search, gpu_buffers.fluid_mask,
                                         gpu_buffers.cg_diag, gpu_buffers.cg_As };
            int spmv_buf_count = 4;
            if (is_variational) {
                b[4] = gpu_buffers.var_u_weight;
                b[5] = gpu_buffers.var_v_weight;
                b[6] = gpu_buffers.var_w_weight;
                spmv_buf_count = 7;
            }
            if (!dispatch1("sim_fluid_cg_spmv", b, spmv_buf_count, cell_groups)) return false;
        }
        return dotProduct(gpu_buffers.cg_search, gpu_buffers.cg_As, out);
    };

    double sigma = 0.0;
    if (!jacobiAndDot(sigma)) return false;
    // s = z
    { ComputeBufferHandle b[2] = { gpu_buffers.cg_search, gpu_buffers.cg_z };
      if (!dispatch1("sim_fluid_cg_copy", b, 2, cell_groups)) return false; }

    const double sigma0   = sigma;
    const double rel_tol   = std::clamp(static_cast<double>(fluid_params.pressure_relative_residual),
                                        1.0e-8, 1.0e-2);
    const double tol       = rel_tol * rel_tol;                       // relative on r.z
    const int    max_iter  = std::max(1, fluid_params.pressure_iterations);
    int iterations_used = 0;

    if (sigma0 > 0.0) {
        for (int iter = 0; iter < max_iter; ++iter) {
            double sAs = 0.0;
            if (!spmvAndDot(sAs)) { ok = false; break; }
            if (std::abs(sAs) < 1e-30) break; // degenerate (no fluid rows)
            const float alpha = static_cast<float>(sigma / sAs);

            // p += alpha s
            c.sor_omega = alpha;
            { ComputeBufferHandle b[2] = { gpu_buffers.pressure, gpu_buffers.cg_search };
              if (!dispatch1("sim_fluid_cg_axpy", b, 2, cell_groups)) { ok = false; break; } }
            // r -= alpha As
            c.sor_omega = -alpha;
            { ComputeBufferHandle b[2] = { gpu_buffers.cg_residual, gpu_buffers.cg_As };
              if (!dispatch1("sim_fluid_cg_axpy", b, 2, cell_groups)) { ok = false; break; } }

            double sigma_new = 0.0;
            if (!jacobiAndDot(sigma_new)) { ok = false; break; }
            iterations_used = iter + 1;
            if (sigma_new <= tol * sigma0) { sigma = sigma_new; break; }

            const float beta = static_cast<float>(sigma_new / sigma);
            // s = z + beta s
            c.sor_omega = beta;
            { ComputeBufferHandle b[2] = { gpu_buffers.cg_search, gpu_buffers.cg_z };
              if (!dispatch1("sim_fluid_cg_zpby", b, 2, cell_groups)) { ok = false; break; } }
            sigma = sigma_new;
        }
    }
    if (!ok) return false;
    if (mgpcg_stats) {
        mgpcg_stats->pressure_cg_iterations = iterations_used;
        mgpcg_stats->pressure_cg_max_iterations = max_iter;
        mgpcg_stats->pressure_cg_dot_count = dot_count;
        mgpcg_stats->pressure_cg_dot_ms = dot_ms;
        mgpcg_stats->pressure_cg_multigrid = used_multigrid;
        mgpcg_stats->pressure_cg_final_relative_residual =
            sigma0 > 0.0 ? std::sqrt(std::max(0.0, sigma) / sigma0) : 0.0;
    }

    // Subtract pressure gradient from velocity.
    ComputeBufferHandle fluid_gradient_bufs[12];
    fluid_gradient_bufs[0] = gpu_buffers.vel_x;
    fluid_gradient_bufs[1] = gpu_buffers.vel_y;
    fluid_gradient_bufs[2] = gpu_buffers.vel_z;
    fluid_gradient_bufs[3] = gpu_buffers.pressure;
    fluid_gradient_bufs[4] = gpu_buffers.fluid_mask;
    if (is_variational) {
        fluid_gradient_bufs[5] = gpu_buffers.var_u_weight;
        fluid_gradient_bufs[6] = gpu_buffers.var_v_weight;
        fluid_gradient_bufs[7] = gpu_buffers.var_w_weight;
        fluid_gradient_bufs[8] = gpu_buffers.var_svx;
        fluid_gradient_bufs[9] = gpu_buffers.var_svy;
        fluid_gradient_bufs[10] = gpu_buffers.var_svz;
    }
    int grad_buf_count = is_variational ? 11 : 5;
    if (is_gfm) {
        fluid_gradient_bufs[grad_buf_count] = gpu_buffers.var_fluid_phi;
        grad_buf_count++;
    }

    cmd.kernel          = use_cuda_fluid_projection
        ? "sim_fluid_subtract_gradient"
        : "sim_grid_subtract_gradient";
    cmd.buffers         = use_cuda_fluid_projection ? fluid_gradient_bufs : proj_bufs;
    cmd.buffer_count    = use_cuda_fluid_projection ? grad_buf_count : 5;
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    cmd.groups.groups_y = 1;
    cmd.groups.groups_z = 1;
    ok = compute->dispatch(cmd);
    compute->synchronize();

    // Download updated velocities for CPU boundary re-enforcement and G2P.
    ok = ok &&
         compute->downloadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    return ok;
}

// GPU G2P gather.
// Expects vel_x/y/z already uploaded (post-pressure) and, for FLIP,
// scratch_vel_x/y/z uploaded as the pre-pressure snapshot.
// Downloads updated particle velocities + affine to CPU when done.
bool runGpuFluidG2P(SimulationGridDomainState& state,
                    const Fluid::APICSolverParams& fluid_params,
                    float dt,
                    SimulationComputeContext* compute,
                    SimulationGridDomainComputeBuffers& gpu_buffers,
                    bool has_flip_snapshot) {
    auto& grid      = state.grid;
    auto& particles = state.particles;
    const std::size_t n = particles.size();
    if (!compute || !compute->supportsDispatch() || n == 0 || dt <= 0.0f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        return false;
    }
    if (!gpu_buffers.fluid_positions.valid() ||
        !gpu_buffers.fluid_velocities.valid() ||
        !gpu_buffers.fluid_affine.valid()     ||
        !gpu_buffers.vel_x.valid()            ||
        !gpu_buffers.vel_y.valid()            ||
        !gpu_buffers.vel_z.valid()            ||
        !gpu_buffers.fluid_mask.valid()) {
        return false;
    }

    // Upload post-projection velocities (may already be up-to-date after
    // runGpuFluidFreeSurfacePressure, but we re-upload to be safe when the
    // caller has modified the CPU copy via boundary enforcement).
    bool ok = compute->uploadBuffer(gpu_buffers.vel_x, grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_y, grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
              compute->uploadBuffer(gpu_buffers.vel_z, grid.vel_z.data(), grid.vel_z.size() * sizeof(float));

    // FLIP snapshot: upload pre-projection velocities to scratch buffers.
    if (has_flip_snapshot &&
        gpu_buffers.scratch_vel_x.valid() &&
        gpu_buffers.scratch_vel_y.valid() &&
        gpu_buffers.scratch_vel_z.valid()) {
        // The caller has already stored the pre-projection copy in
        // gpu_buffers.scratch_vel_x/y/z via uploadBuffer before calling
        // runGpuFluidFreeSurfacePressure. No re-upload needed here.
    } else {
        has_flip_snapshot = false;
    }

    // Positions must be current (uploaded during forces/P2G stage).
    // Velocities: upload old particle velocities (needed for FLIP v_old term).
    ok = ok && compute->uploadBuffer(gpu_buffers.fluid_velocities,
                                     particles.velocity.data(),
                                     n * sizeof(Vec3));
    if (!ok) return false;

    FluidG2PGpuConstants c;
    c.nx                = grid.nx;
    c.ny                = grid.ny;
    c.nz                = grid.nz;
    c.particle_count    = static_cast<int>(std::min<std::size_t>(n, static_cast<std::size_t>(std::numeric_limits<int>::max())));
    c.origin_x          = grid.origin.x;
    c.origin_y          = grid.origin.y;
    c.origin_z          = grid.origin.z;
    c.voxel_size        = grid.voxel_size;
    c.flip_blend        = std::clamp(fluid_params.flip_blend, 0.0f, 1.0f);
    c.apic_blend        = std::clamp(fluid_params.apic_blend, 0.0f, 1.0f);
    c.internal_friction = fluid_params.internal_friction;
    c.max_velocity      = fluid_params.max_velocity;
    c.dt                = dt;
    c.has_flip_snapshot = has_flip_snapshot ? 1 : 0;
    c.use_solid_flip_limiter = compute->backendType() == ComputeBackendType::CUDA ? 1 : 0;
    c.affine_damping    = fluid_params.affine_damping;
    c.max_affine        = fluid_params.max_affine;

    constexpr uint32_t threads = 256;
    ComputeBufferHandle bufs[10] = {
        gpu_buffers.fluid_positions,
        gpu_buffers.fluid_velocities,
        gpu_buffers.fluid_affine,
        gpu_buffers.vel_x,
        gpu_buffers.vel_y,
        gpu_buffers.vel_z,
        gpu_buffers.scratch_vel_x,  // pre-projection snapshot (or unused)
        gpu_buffers.scratch_vel_y,
        gpu_buffers.scratch_vel_z,
        gpu_buffers.fluid_mask
    };
    ComputeDispatch cmd;
    cmd.kernel         = "sim_fluid_g2p";
    cmd.buffers        = bufs;
    cmd.buffer_count   = 10;
    cmd.constants      = &c;
    cmd.constants_size = sizeof(c);
    cmd.groups.groups_x = (static_cast<uint32_t>(c.particle_count) + threads - 1u) / threads;
    ok = compute->dispatch(cmd);
    compute->synchronize();

    ok = ok &&
         compute->downloadBuffer(gpu_buffers.fluid_velocities,
                                 particles.velocity.data(),
                                 n * sizeof(Vec3)) &&
         compute->downloadBuffer(gpu_buffers.fluid_affine,
                                 particles.affine.data(),
                                 n * sizeof(Fluid::AffineC));
    return ok;
}

bool runGpuPressureProjection(FluidSim::FluidGrid& grid,
                              const GridFluid::SolverParams& params,
                              float dt,
                              SimulationComputeContext* compute,
                              SimulationGridDomainComputeBuffers& gpu_buffers) {
    if (!compute || !compute->supportsDispatch() || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return false;
    }
    if (grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty() ||
        grid.pressure.size() != grid.getCellCount() ||
        grid.divergence.size() != grid.getCellCount()) {
        return false;
    }

    ComputeBufferHandle buffers[5];
    buffers[0] = gpu_buffers.vel_x;
    buffers[1] = gpu_buffers.vel_y;
    buffers[2] = gpu_buffers.vel_z;
    buffers[3] = gpu_buffers.pressure;
    buffers[4] = gpu_buffers.divergence;

    bool ok = true;
    for (const ComputeBufferHandle& h : buffers) {
        ok = ok && h.valid();
    }
    ok = ok &&
         compute->uploadBuffer(buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[3], grid.pressure.data(), grid.pressure.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[4], grid.divergence.data(), grid.divergence.size() * sizeof(float));
    if (!ok) {
        return false;
    }

    GridProjectionGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.boundary = boundaryToGpu(params.boundary);
    constants.voxel_size = grid.voxel_size;
    constants.dt = dt;
    constants.sor_omega = params.sor_omega;
    constants.iterations = std::max(1, params.pressure_iterations);

    const uint32_t threads = 256;
    const uint32_t cell_count = static_cast<uint32_t>(grid.getCellCount());
    const uint32_t max_faces = static_cast<uint32_t>(std::max({ grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size() }));
    ComputeDispatch cmd;
    cmd.buffers = buffers;
    cmd.buffer_count = 5;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);

    cmd.kernel = "sim_grid_divergence";
    cmd.groups.groups_x = (cell_count + threads - 1u) / threads;
    ok = compute->dispatch(cmd);

    cmd.kernel = "sim_grid_sor";
    cmd.groups.groups_x = (cell_count + threads - 1u) / threads;
    for (int iter = 0; ok && iter < constants.iterations; ++iter) {
        constants.parity = 0;
        ok = compute->dispatch(cmd);
        constants.parity = 1;
        ok = ok && compute->dispatch(cmd);
    }

    cmd.kernel = "sim_grid_subtract_gradient";
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    ok = ok && compute->dispatch(cmd);
    compute->synchronize();

    ok = ok &&
         compute->downloadBuffer(buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[3], grid.pressure.data(), grid.pressure.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[4], grid.divergence.data(), grid.divergence.size() * sizeof(float));

    return ok;
}

bool runGpuVelocityAdvection(FluidSim::FluidGrid& grid,
                             float dt,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers& gpu_buffers) {
    if (!compute || !compute->supportsDispatch() || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return false;
    }
    if (grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }

    ComputeBufferHandle buffers[6];
    buffers[0] = gpu_buffers.vel_x;
    buffers[1] = gpu_buffers.vel_y;
    buffers[2] = gpu_buffers.vel_z;
    buffers[3] = gpu_buffers.scratch_vel_x;
    buffers[4] = gpu_buffers.scratch_vel_y;
    buffers[5] = gpu_buffers.scratch_vel_z;

    bool ok = true;
    for (const ComputeBufferHandle& h : buffers) {
        ok = ok && h.valid();
    }
    ok = ok &&
         compute->uploadBuffer(buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->uploadBuffer(buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    if (!ok) {
        return false;
    }

    GridVelocityAdvectionGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.voxel_size = grid.voxel_size;
    constants.dt = dt;

    const uint32_t threads = 256;
    const uint32_t max_faces = static_cast<uint32_t>(std::max({ grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size() }));
    ComputeDispatch cmd;
    cmd.kernel = "sim_grid_advect_velocity";
    cmd.buffers = buffers;
    cmd.buffer_count = 6;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    ok = compute->dispatch(cmd);
    compute->synchronize();

    ok = ok &&
         compute->downloadBuffer(buffers[3], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[4], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[5], grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    return ok;
}

void applyCpuVelocityDissipationClamp(FluidSim::FluidGrid& grid,
                                      const GridFluid::SolverParams& params,
                                      float dt) {
    const float factor = params.velocity_dissipation > 0.0f
        ? std::exp(-params.velocity_dissipation * dt)
        : 1.0f;
    const float max_velocity = params.max_velocity;
    auto apply = [&](std::vector<float>& values) {
        for (float& value : values) {
            value *= factor;
            if (max_velocity > 0.0f) {
                value = std::clamp(value, -max_velocity, max_velocity);
            }
        }
    };
    apply(grid.vel_x);
    apply(grid.vel_y);
    apply(grid.vel_z);
}

bool runGpuVelocityDissipationClamp(FluidSim::FluidGrid& grid,
                                    const GridFluid::SolverParams& params,
                                    float dt,
                                    SimulationComputeContext* compute,
                                    SimulationGridDomainComputeBuffers& gpu_buffers) {
    if (!compute || !compute->supportsDispatch() || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return false;
    }
    if (grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }

    ComputeBufferHandle buffers[3];
    buffers[0] = gpu_buffers.vel_x;
    buffers[1] = gpu_buffers.vel_y;
    buffers[2] = gpu_buffers.vel_z;

    bool ok = buffers[0].valid() && buffers[1].valid() && buffers[2].valid() &&
              compute->uploadBuffer(buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
              compute->uploadBuffer(buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
              compute->uploadBuffer(buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    if (!ok) {
        return false;
    }

    GridVelocityDissipationGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.factor = params.velocity_dissipation > 0.0f
        ? std::exp(-params.velocity_dissipation * dt)
        : 1.0f;
    constants.max_velocity = params.max_velocity;

    const uint32_t threads = 256;
    const uint32_t max_faces = static_cast<uint32_t>(std::max({ grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size() }));
    ComputeDispatch cmd;
    cmd.kernel = "sim_grid_velocity_dissipate_clamp";
    cmd.buffers = buffers;
    cmd.buffer_count = 3;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    cmd.groups.groups_x = (max_faces + threads - 1u) / threads;
    ok = compute->dispatch(cmd);
    compute->synchronize();

    ok = ok &&
         compute->downloadBuffer(buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
         compute->downloadBuffer(buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    return ok;
}

bool runGpuScalarAdvection(FluidSim::FluidGrid& grid,
                           const GridFluid::SolverParams& params,
                           float dt,
                           SimulationComputeContext* compute,
                           SimulationGridDomainComputeBuffers& gpu_buffers) {
    if (!compute || !compute->supportsDispatch() || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || dt <= 0.0f) {
        return false;
    }
    if (grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty()) {
        return false;
    }

    ComputeBufferHandle vel_buffers[3];
    vel_buffers[0] = gpu_buffers.vel_x;
    vel_buffers[1] = gpu_buffers.vel_y;
    vel_buffers[2] = gpu_buffers.vel_z;

    bool ok = vel_buffers[0].valid() && vel_buffers[1].valid() && vel_buffers[2].valid() &&
              compute->uploadBuffer(vel_buffers[0], grid.vel_x.data(), grid.vel_x.size() * sizeof(float)) &&
              compute->uploadBuffer(vel_buffers[1], grid.vel_y.data(), grid.vel_y.size() * sizeof(float)) &&
              compute->uploadBuffer(vel_buffers[2], grid.vel_z.data(), grid.vel_z.size() * sizeof(float));
    if (!ok) {
        return false;
    }

    GridScalarAdvectionGpuConstants constants;
    constants.nx = grid.nx;
    constants.ny = grid.ny;
    constants.nz = grid.nz;
    constants.voxel_size = grid.voxel_size;
    constants.dt = dt;

    auto advectField = [&](const std::vector<float>& field,
                           std::vector<float>& output,
                           ComputeBufferHandle source_buffer) {
        if (field.empty() || field.size() != grid.getCellCount()) {
            return false;
        }
        output.assign(field.size(), 0.0f);

        ComputeBufferHandle buffers[5];
        buffers[0] = vel_buffers[0];
        buffers[1] = vel_buffers[1];
        buffers[2] = vel_buffers[2];
        buffers[3] = source_buffer;
        buffers[4] = gpu_buffers.scratch_scalar;

        bool field_ok = buffers[3].valid() && buffers[4].valid() &&
                        compute->uploadBuffer(buffers[3], field.data(), field.size() * sizeof(float));
        if (field_ok) {
            ComputeDispatch cmd;
            cmd.kernel = "sim_grid_advect_scalar";
            cmd.buffers = buffers;
            cmd.buffer_count = 5;
            cmd.constants = &constants;
            cmd.constants_size = sizeof(constants);
            const uint32_t cell_count = static_cast<uint32_t>(grid.getCellCount());
            cmd.groups.groups_x = (cell_count + 255u) / 256u;
            field_ok = compute->dispatch(cmd);
            compute->synchronize();
        }
        field_ok = field_ok &&
                   compute->downloadBuffer(buffers[4], output.data(), output.size() * sizeof(float));
        return field_ok;
    };

    std::vector<float> next_density;
    std::vector<float> next_temperature;
    std::vector<float> next_fuel;
    if (ok && params.channel_density) {
        ok = advectField(grid.density, next_density, gpu_buffers.density);
    }
    if (ok && params.channel_temperature) {
        ok = advectField(grid.temperature, next_temperature, gpu_buffers.temperature);
    }
    if (ok && params.channel_fuel) {
        ok = advectField(grid.fuel, next_fuel, gpu_buffers.fuel);
    }

    if (ok) {
        if (params.channel_density) {
            grid.density.swap(next_density);
        }
        if (params.channel_temperature) {
            grid.temperature.swap(next_temperature);
        }
        if (params.channel_fuel) {
            grid.fuel.swap(next_fuel);
        }
    }

    return ok;
}

uint32_t hashUInt(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float hashUnitFloat(uint32_t value) {
    return static_cast<float>(hashUInt(value) & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

uint64_t hashGridCell(int x, int y, int z) {
    const uint64_t ux = static_cast<uint64_t>(static_cast<int64_t>(x) + 0x1fffffLL) & 0x1fffffULL;
    const uint64_t uy = static_cast<uint64_t>(static_cast<int64_t>(y) + 0x1fffffLL) & 0x1fffffULL;
    const uint64_t uz = static_cast<uint64_t>(static_cast<int64_t>(z) + 0x1fffffLL) & 0x1fffffULL;
    return (ux << 42u) ^ (uy << 21u) ^ uz;
}

bool hasGridChannel(uint32_t channels, SimulationGridDomainChannelFlags channel) {
    return (channels & static_cast<uint32_t>(channel)) != 0u;
}

std::size_t gridCellCount(int x, int y, int z) {
    if (x <= 0 || y <= 0 || z <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(x) *
           static_cast<std::size_t>(y) *
           static_cast<std::size_t>(z);
}

void clampGridResolutionToCellBudget(int& x, int& y, int& z, std::size_t max_cells) {
    if (max_cells == 0) {
        x = y = z = 8;
        return;
    }
    x = std::max(8, x);
    y = std::max(8, y);
    z = std::max(8, z);
    while (gridCellCount(x, y, z) > max_cells && (x > 8 || y > 8 || z > 8)) {
        if (x >= y && x >= z && x > 8) {
            x = std::max(8, static_cast<int>(std::floor(static_cast<float>(x) * 0.85f)));
        } else if (y >= x && y >= z && y > 8) {
            y = std::max(8, static_cast<int>(std::floor(static_cast<float>(y) * 0.85f)));
        } else if (z > 8) {
            z = std::max(8, static_cast<int>(std::floor(static_cast<float>(z) * 0.85f)));
        } else {
            break;
        }
    }
}

Vec3 finiteDirectionOrUp(const Vec3& direction) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) || !std::isfinite(direction.z)) {
        return Vec3(0.0f, 1.0f, 0.0f);
    }
    const float length = direction.length();
    return length > 1e-5f ? direction * (1.0f / length) : Vec3(0.0f, 1.0f, 0.0f);
}

Vec3 emitterVelocity(const ParticleEmitterDesc& emitter, const Vec3& direction, uint32_t serial) {
    const Vec3 forward = finiteDirectionOrUp(direction);
    Vec3 tangent = Vec3::cross(forward, Vec3(0.0f, 1.0f, 0.0f));
    if (tangent.length() < 1e-5f) {
        tangent = Vec3::cross(forward, Vec3(1.0f, 0.0f, 0.0f));
    }
    tangent = finiteDirectionOrUp(tangent);
    const Vec3 bitangent = finiteDirectionOrUp(Vec3::cross(forward, tangent));

    const float angle = 6.28318530718f * hashUnitFloat(emitter.seed ^ serial ^ 0x9e3779b9u);
    const float radius = std::sqrt(hashUnitFloat(emitter.seed + serial * 1664525u + 1013904223u));
    const float spread = std::max(0.0f, emitter.spread);
    const Vec3 lateral = (tangent * std::cos(angle) + bitangent * std::sin(angle)) * (radius * spread);
    return finiteDirectionOrUp(forward + lateral) * std::max(0.0f, emitter.speed);
}

bool sampleAABBSurface(const Vec3& bounds_min,
                       const Vec3& bounds_max,
                       uint32_t seed,
                       Vec3& out_position,
                       Vec3& out_normal) {
    const Vec3 min_bound = Vec3::min(bounds_min, bounds_max);
    const Vec3 max_bound = Vec3::max(bounds_min, bounds_max);
    const Vec3 extent = max_bound - min_bound;
    const int positive_axes = (extent.x > 1e-5f ? 1 : 0) +
                              (extent.y > 1e-5f ? 1 : 0) +
                              (extent.z > 1e-5f ? 1 : 0);
    if (positive_axes < 2) {
        return false;
    }

    const float area_x = extent.y * extent.z;
    const float area_y = extent.x * extent.z;
    const float area_z = extent.x * extent.y;
    const float total_area = 2.0f * (area_x + area_y + area_z);
    if (total_area <= 1e-6f) {
        return false;
    }

    float pick = hashUnitFloat(seed ^ 0xa2f3u) * total_area;
    const float u = hashUnitFloat(seed ^ 0xb5297a4du);
    const float v = hashUnitFloat(seed ^ 0x68e31da4u);

    auto takeFace = [&](float area) {
        if (pick <= area) {
            return true;
        }
        pick -= area;
        return false;
    };

    if (takeFace(area_x)) {
        out_position = Vec3(min_bound.x, min_bound.y + extent.y * u, min_bound.z + extent.z * v);
        out_normal = Vec3(-1.0f, 0.0f, 0.0f);
    } else if (takeFace(area_x)) {
        out_position = Vec3(max_bound.x, min_bound.y + extent.y * u, min_bound.z + extent.z * v);
        out_normal = Vec3(1.0f, 0.0f, 0.0f);
    } else if (takeFace(area_y)) {
        out_position = Vec3(min_bound.x + extent.x * u, min_bound.y, min_bound.z + extent.z * v);
        out_normal = Vec3(0.0f, -1.0f, 0.0f);
    } else if (takeFace(area_y)) {
        out_position = Vec3(min_bound.x + extent.x * u, max_bound.y, min_bound.z + extent.z * v);
        out_normal = Vec3(0.0f, 1.0f, 0.0f);
    } else if (takeFace(area_z)) {
        out_position = Vec3(min_bound.x + extent.x * u, min_bound.y + extent.y * v, min_bound.z);
        out_normal = Vec3(0.0f, 0.0f, -1.0f);
    } else {
        out_position = Vec3(min_bound.x + extent.x * u, min_bound.y + extent.y * v, max_bound.z);
        out_normal = Vec3(0.0f, 0.0f, 1.0f);
    }
    return true;
}

void resolveAABBCollision(const ParticleColliderDesc& collider, Vec3& position, Vec3& velocity) {
    Vec3 min_bound = Vec3::min(collider.bounds_min, collider.bounds_max) - collider.thickness;
    Vec3 max_bound = Vec3::max(collider.bounds_min, collider.bounds_max) + collider.thickness;
    if (position.x < min_bound.x || position.x > max_bound.x ||
        position.y < min_bound.y || position.y > max_bound.y ||
        position.z < min_bound.z || position.z > max_bound.z) {
        return;
    }

    const float distances[6] = {
        std::abs(position.x - min_bound.x),
        std::abs(max_bound.x - position.x),
        std::abs(position.y - min_bound.y),
        std::abs(max_bound.y - position.y),
        std::abs(position.z - min_bound.z),
        std::abs(max_bound.z - position.z)
    };

    int face = 0;
    for (int i = 1; i < 6; ++i) {
        if (distances[i] < distances[face]) {
            face = i;
        }
    }

    Vec3 normal(0.0f, 1.0f, 0.0f);
    switch (face) {
        case 0: position.x = min_bound.x; normal = Vec3(-1.0f, 0.0f, 0.0f); break;
        case 1: position.x = max_bound.x; normal = Vec3(1.0f, 0.0f, 0.0f); break;
        case 2: position.y = min_bound.y; normal = Vec3(0.0f, -1.0f, 0.0f); break;
        case 3: position.y = max_bound.y; normal = Vec3(0.0f, 1.0f, 0.0f); break;
        case 4: position.z = min_bound.z; normal = Vec3(0.0f, 0.0f, -1.0f); break;
        case 5: position.z = max_bound.z; normal = Vec3(0.0f, 0.0f, 1.0f); break;
        default: break;
    }

    const float normal_velocity = Vec3::dot(velocity, normal);
    if (normal_velocity < 0.0f) {
        const float restitution = std::clamp(collider.restitution, 0.0f, 1.0f);
        const float friction = std::clamp(collider.friction, 0.0f, 1.0f);
        const Vec3 normal_component = normal * normal_velocity;
        const Vec3 tangent_component = velocity - normal_component;
        velocity = tangent_component * (1.0f - friction) - normal_component * restitution;
    }
}

void resolveVelocityAgainstNormal(const ParticleColliderDesc& collider, const Vec3& normal, Vec3& velocity) {
    const float normal_velocity = Vec3::dot(velocity, normal);
    if (normal_velocity < 0.0f) {
        const float restitution = std::clamp(collider.restitution, 0.0f, 1.0f);
        const float friction = std::clamp(collider.friction, 0.0f, 1.0f);
        const Vec3 normal_component = normal * normal_velocity;
        const Vec3 tangent_component = velocity - normal_component;
        velocity = tangent_component * (1.0f - friction) - normal_component * restitution;
    }
}

bool resolveSweptAABBCollision(const ParticleColliderDesc& collider,
                               const Vec3& previous_position,
                               Vec3& position,
                               Vec3& velocity) {
    Vec3 min_bound = Vec3::min(collider.bounds_min, collider.bounds_max) - collider.thickness;
    Vec3 max_bound = Vec3::max(collider.bounds_min, collider.bounds_max) + collider.thickness;
    const float shell = std::max(collider.thickness, 0.001f);
    min_bound = min_bound - shell;
    max_bound = max_bound + shell;

    const Vec3 delta = position - previous_position;
    const float axes_prev[3] = { previous_position.x, previous_position.y, previous_position.z };
    const float axes_pos[3] = { position.x, position.y, position.z };
    const float axes_delta[3] = { delta.x, delta.y, delta.z };
    const float mins[3] = { min_bound.x, min_bound.y, min_bound.z };
    const float maxs[3] = { max_bound.x, max_bound.y, max_bound.z };

    float best_t = 2.0f;
    int best_axis = -1;
    float best_sign = 1.0f;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::abs(axes_delta[axis]) <= 1e-8f) {
            continue;
        }

        const float planes[2] = { mins[axis], maxs[axis] };
        for (int side = 0; side < 2; ++side) {
            const float t = (planes[side] - axes_prev[axis]) / axes_delta[axis];
            if (t < 0.0f || t > 1.0f || t >= best_t) {
                continue;
            }

            float p[3] = {
                previous_position.x + delta.x * t,
                previous_position.y + delta.y * t,
                previous_position.z + delta.z * t
            };
            bool inside_other_axes = true;
            for (int other = 0; other < 3; ++other) {
                if (other == axis) {
                    continue;
                }
                if (p[other] < mins[other] || p[other] > maxs[other]) {
                    inside_other_axes = false;
                    break;
                }
            }
            if (!inside_other_axes) {
                continue;
            }

            const bool entered_from_min = side == 0 && axes_prev[axis] < mins[axis] && axes_pos[axis] >= mins[axis];
            const bool entered_from_max = side == 1 && axes_prev[axis] > maxs[axis] && axes_pos[axis] <= maxs[axis];
            if (!entered_from_min && !entered_from_max) {
                continue;
            }

            best_t = t;
            best_axis = axis;
            best_sign = entered_from_min ? -1.0f : 1.0f;
        }
    }

    if (best_axis < 0) {
        return false;
    }

    const Vec3 hit_position = previous_position + delta * best_t;
    Vec3 normal(0.0f, 0.0f, 0.0f);
    if (best_axis == 0) normal = Vec3(best_sign, 0.0f, 0.0f);
    if (best_axis == 1) normal = Vec3(0.0f, best_sign, 0.0f);
    if (best_axis == 2) normal = Vec3(0.0f, 0.0f, best_sign);

    position = hit_position + normal * shell;
    const float normal_velocity = Vec3::dot(velocity, normal);
    if (normal_velocity < 0.0f) {
        const float restitution = std::clamp(collider.restitution, 0.0f, 1.0f);
        const float friction = std::clamp(collider.friction, 0.0f, 1.0f);
        const Vec3 normal_component = normal * normal_velocity;
        const Vec3 tangent_component = velocity - normal_component;
        velocity = tangent_component * (1.0f - friction) - normal_component * restitution;
    }
    return true;
}

void resolveSphereCollision(const ParticleColliderDesc& collider,
                            float particle_radius,
                            Vec3& position,
                            Vec3& velocity) {
    const float effective_radius = std::max(0.0f, collider.sphere_radius) +
                                   std::max(0.0f, collider.thickness) +
                                   std::max(0.0f, particle_radius);
    if (effective_radius <= 1e-6f) {
        return;
    }

    Vec3 delta = position - collider.sphere_center;
    float distance = delta.length();
    if (distance >= effective_radius) {
        return;
    }

    Vec3 normal(0.0f, 1.0f, 0.0f);
    if (distance > 1e-6f) {
        normal = delta * (1.0f / distance);
    } else if (velocity.length() > 1e-6f) {
        normal = finiteDirectionOrUp(velocity * -1.0f);
    }

    position = collider.sphere_center + normal * effective_radius;
    resolveVelocityAgainstNormal(collider, normal, velocity);
}

bool resolveSweptSphereCollision(const ParticleColliderDesc& collider,
                                 float particle_radius,
                                 const Vec3& previous_position,
                                 Vec3& position,
                                 Vec3& velocity) {
    const float effective_radius = std::max(0.0f, collider.sphere_radius) +
                                   std::max(0.0f, collider.thickness) +
                                   std::max(0.0f, particle_radius);
    if (effective_radius <= 1e-6f) {
        return false;
    }

    const Vec3 segment = position - previous_position;
    const Vec3 from_center = previous_position - collider.sphere_center;
    const float a = Vec3::dot(segment, segment);
    if (a <= 1e-8f) {
        return false;
    }

    const float b = 2.0f * Vec3::dot(from_center, segment);
    const float c = Vec3::dot(from_center, from_center) - effective_radius * effective_radius;
    if (c <= 0.0f) {
        return false;
    }

    const float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = std::sqrt(discriminant);
    const float inv_denominator = 1.0f / (2.0f * a);
    const float t0 = (-b - sqrt_discriminant) * inv_denominator;
    const float t1 = (-b + sqrt_discriminant) * inv_denominator;
    const float t = (t0 >= 0.0f && t0 <= 1.0f) ? t0 : t1;
    if (t < 0.0f || t > 1.0f) {
        return false;
    }

    const Vec3 hit_position = previous_position + segment * t;
    const Vec3 normal = finiteDirectionOrUp(hit_position - collider.sphere_center);
    position = collider.sphere_center + normal * effective_radius;
    resolveVelocityAgainstNormal(collider, normal, velocity);
    return true;
}

Vec3 closestPointOnSegment(const Vec3& point, const Vec3& a, const Vec3& b) {
    const Vec3 ab = b - a;
    const float denom = Vec3::dot(ab, ab);
    if (denom <= 1e-8f) {
        return a;
    }
    const float t = std::clamp(Vec3::dot(point - a, ab) / denom, 0.0f, 1.0f);
    return a + ab * t;
}

float segmentSegmentDistanceSquared(const Vec3& p0, const Vec3& p1, const Vec3& q0, const Vec3& q1) {
    const Vec3 u = p1 - p0;
    const Vec3 v = q1 - q0;
    const Vec3 w = p0 - q0;
    const float a = Vec3::dot(u, u);
    const float b = Vec3::dot(u, v);
    const float c = Vec3::dot(v, v);
    const float d = Vec3::dot(u, w);
    const float e = Vec3::dot(v, w);
    const float denom = a * c - b * b;

    float s = 0.0f;
    float t = 0.0f;
    if (denom > 1e-8f) {
        s = std::clamp((b * e - c * d) / denom, 0.0f, 1.0f);
    }

    const float t_nom = b * s + e;
    if (t_nom <= 0.0f) {
        t = 0.0f;
        s = a > 1e-8f ? std::clamp(-d / a, 0.0f, 1.0f) : 0.0f;
    } else if (t_nom >= c) {
        t = 1.0f;
        s = a > 1e-8f ? std::clamp((b - d) / a, 0.0f, 1.0f) : 0.0f;
    } else {
        t = c > 1e-8f ? t_nom / c : 0.0f;
    }

    const Vec3 cp = p0 + u * s;
    const Vec3 cq = q0 + v * t;
    return Vec3::dot(cp - cq, cp - cq);
}

void resolveCapsuleCollision(const ParticleColliderDesc& collider,
                             float particle_radius,
                             Vec3& position,
                             Vec3& velocity) {
    const float effective_radius = std::max(0.0f, collider.capsule_radius) +
                                   std::max(0.0f, collider.thickness) +
                                   std::max(0.0f, particle_radius);
    if (effective_radius <= 1e-6f) {
        return;
    }

    const Vec3 closest = closestPointOnSegment(position, collider.capsule_start, collider.capsule_end);
    Vec3 delta = position - closest;
    const float distance = delta.length();
    if (distance >= effective_radius) {
        return;
    }

    Vec3 normal(0.0f, 1.0f, 0.0f);
    if (distance > 1e-6f) {
        normal = delta * (1.0f / distance);
    } else if (velocity.length() > 1e-6f) {
        normal = finiteDirectionOrUp(velocity * -1.0f);
    }

    position = closest + normal * effective_radius;
    resolveVelocityAgainstNormal(collider, normal, velocity);
}

bool resolveSweptCapsuleCollision(const ParticleColliderDesc& collider,
                                  float particle_radius,
                                  const Vec3& previous_position,
                                  Vec3& position,
                                  Vec3& velocity) {
    const float effective_radius = std::max(0.0f, collider.capsule_radius) +
                                   std::max(0.0f, collider.thickness) +
                                   std::max(0.0f, particle_radius);
    if (effective_radius <= 1e-6f) {
        return false;
    }

    const float distance_sq = segmentSegmentDistanceSquared(previous_position,
                                                           position,
                                                           collider.capsule_start,
                                                           collider.capsule_end);
    if (distance_sq > effective_radius * effective_radius) {
        return false;
    }

    resolveCapsuleCollision(collider, particle_radius, position, velocity);
    return true;
}

void resolveOBBCollision(const ParticleColliderDesc& collider,
                         const ParticleColliderOBB& obb,
                         Vec3& position,
                         Vec3& velocity) {
    const Matrix4x4 world_to_local = obb.local_to_world.inverse();
    Vec3 local_position = world_to_local.transform_point(position);
    Vec3 local_velocity = world_to_local.transform_vector(velocity);

    ParticleColliderDesc local_collider = collider;
    local_collider.bounds_min = obb.local_bounds_min;
    local_collider.bounds_max = obb.local_bounds_max;
    resolveAABBCollision(local_collider, local_position, local_velocity);

    position = obb.local_to_world.transform_point(local_position);
    velocity = obb.local_to_world.transform_vector(local_velocity);
}

bool resolveSweptOBBCollision(const ParticleColliderDesc& collider,
                              const ParticleColliderOBB& obb,
                              const Vec3& previous_position,
                              Vec3& position,
                              Vec3& velocity) {
    const Matrix4x4 world_to_local = obb.local_to_world.inverse();
    const Vec3 local_previous = world_to_local.transform_point(previous_position);
    Vec3 local_position = world_to_local.transform_point(position);
    Vec3 local_velocity = world_to_local.transform_vector(velocity);

    ParticleColliderDesc local_collider = collider;
    local_collider.bounds_min = obb.local_bounds_min;
    local_collider.bounds_max = obb.local_bounds_max;
    if (!resolveSweptAABBCollision(local_collider, local_previous, local_position, local_velocity)) {
        return false;
    }

    position = obb.local_to_world.transform_point(local_position);
    velocity = obb.local_to_world.transform_vector(local_velocity);
    return true;
}


void resolveSDFCollision(const ParticleColliderDesc& collider,
                          const ParticleColliderOBB& obb,
                          float particle_radius,
                          Vec3& position,
                          Vec3& velocity) {
    if (!collider.sdf_grid_data || collider.sdf_grid_data->empty()) return;
    
    // Transform world position/velocity to local OBB space
    const Matrix4x4 world_to_local = obb.local_to_world.inverse();
    const Vec3 local_pos = world_to_local.transform_point(position);
    
    // Calculate scale factor relative to cooked size
    Vec3 cooked_size = collider.sdf_extents / 1.3f;
    Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
    Vec3 scale(
        cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
        cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
        cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
    );
    // Clamp scale to avoid division by zero or negative
    scale.x = std::max(1e-4f, scale.x);
    scale.y = std::max(1e-4f, scale.y);
    scale.z = std::max(1e-4f, scale.z);

    float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
    float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
    float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
    float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
    if (avg_scale <= 1e-6f) avg_scale = 1.0f;

    // Thickness in local space
    const float world_thick = std::max(0.0f, collider.thickness) + particle_radius;
    const float local_thick = world_thick / avg_scale;

    const Vec3 sdf_local_pos = local_pos - (collider.sdf_origin * scale);
    const Vec3 sdf_extents_scaled = collider.sdf_extents * scale;
    
    // Check bounding box in local space
    if (sdf_local_pos.x < -local_thick || sdf_local_pos.x > sdf_extents_scaled.x + local_thick ||
        sdf_local_pos.y < -local_thick || sdf_local_pos.y > sdf_extents_scaled.y + local_thick ||
        sdf_local_pos.z < -local_thick || sdf_local_pos.z > sdf_extents_scaled.z + local_thick) {
        return;
    }

    const float tx = sdf_extents_scaled.x > 1e-6f ? sdf_local_pos.x / sdf_extents_scaled.x : 0.0f;
    const float ty = sdf_extents_scaled.y > 1e-6f ? sdf_local_pos.y / sdf_extents_scaled.y : 0.0f;
    const float tz = sdf_extents_scaled.z > 1e-6f ? sdf_local_pos.z / sdf_extents_scaled.z : 0.0f;

    auto sampleSDF = [&](int x, int y, int z) -> float {
        int ix = std::clamp(x, 0, collider.sdf_nx - 1);
        int iy = std::clamp(y, 0, collider.sdf_ny - 1);
        int iz = std::clamp(z, 0, collider.sdf_nz - 1);
        std::size_t idx = static_cast<std::size_t>(iz * (collider.sdf_nx * collider.sdf_ny) + iy * collider.sdf_nx + ix);
        return idx < collider.sdf_grid_data->size() ? (*collider.sdf_grid_data)[idx] : 0.0f;
    };

    // Tri-linear interpolation in local space
    float fx = std::clamp(tx, 0.0f, 1.0f) * (collider.sdf_nx - 1);
    float fy = std::clamp(ty, 0.0f, 1.0f) * (collider.sdf_ny - 1);
    float fz = std::clamp(tz, 0.0f, 1.0f) * (collider.sdf_nz - 1);

    int ix = std::clamp(static_cast<int>(std::floor(fx)), 0, collider.sdf_nx - 1);
    int iy = std::clamp(static_cast<int>(std::floor(fy)), 0, collider.sdf_ny - 1);
    int iz = std::clamp(static_cast<int>(std::floor(fz)), 0, collider.sdf_nz - 1);

    int ix1 = std::clamp(ix + 1, 0, collider.sdf_nx - 1);
    int iy1 = std::clamp(iy + 1, 0, collider.sdf_ny - 1);
    int iz1 = std::clamp(iz + 1, 0, collider.sdf_nz - 1);

    float dx = fx - ix;
    float dy = fy - iy;
    float dz = fz - iz;

    float v000 = sampleSDF(ix, iy, iz);
    float v100 = sampleSDF(ix1, iy, iz);
    float v010 = sampleSDF(ix, iy1, iz);
    float v110 = sampleSDF(ix1, iy1, iz);
    float v001 = sampleSDF(ix, iy, iz1);
    float v101 = sampleSDF(ix1, iy, iz1);
    float v011 = sampleSDF(ix, iy1, iz1);
    float v111 = sampleSDF(ix1, iy1, iz1);

    float v00 = v000 * (1.0f - dx) + v100 * dx;
    float v10 = v010 * (1.0f - dx) + v110 * dx;
    float v01 = v001 * (1.0f - dx) + v101 * dx;
    float v11 = v011 * (1.0f - dx) + v111 * dx;

    float v0 = v00 * (1.0f - dy) + v10 * dy;
    float v1 = v01 * (1.0f - dy) + v11 * dy;

    float dist_cooked = v0 * (1.0f - dz) + v1 * dz;
    float dist_local = dist_cooked * ((scale.x + scale.y + scale.z) / 3.0f); // Scale the cooked SDF distance to current local space

    if (dist_local <= local_thick) {
        // Calculate gradient in local space using grid steps
        float step_x = sdf_extents_scaled.x / std::max(1.0f, static_cast<float>(collider.sdf_nx - 1));
        float step_y = sdf_extents_scaled.y / std::max(1.0f, static_cast<float>(collider.sdf_ny - 1));
        float step_z = sdf_extents_scaled.z / std::max(1.0f, static_cast<float>(collider.sdf_nz - 1));

        auto interpSDF = [&](const Vec3& lp) -> float {
            Vec3 slp = lp - (collider.sdf_origin * scale);
            float px = std::clamp(sdf_extents_scaled.x > 1e-6f ? slp.x / sdf_extents_scaled.x : 0.0f, 0.0f, 1.0f);
            float py = std::clamp(sdf_extents_scaled.y > 1e-6f ? slp.y / sdf_extents_scaled.y : 0.0f, 0.0f, 1.0f);
            float pz = std::clamp(sdf_extents_scaled.z > 1e-6f ? slp.z / sdf_extents_scaled.z : 0.0f, 0.0f, 1.0f);
            
            float gfx = px * (collider.sdf_nx - 1);
            float gfy = py * (collider.sdf_ny - 1);
            float gfz = pz * (collider.sdf_nz - 1);

            int gix = std::clamp(static_cast<int>(std::floor(gfx)), 0, collider.sdf_nx - 1);
            int giy = std::clamp(static_cast<int>(std::floor(gfy)), 0, collider.sdf_ny - 1);
            int giz = std::clamp(static_cast<int>(std::floor(gfz)), 0, collider.sdf_nz - 1);

            int gix1 = std::clamp(gix + 1, 0, collider.sdf_nx - 1);
            int giy1 = std::clamp(giy + 1, 0, collider.sdf_ny - 1);
            int giz1 = std::clamp(giz + 1, 0, collider.sdf_nz - 1);

            float gdx = gfx - gix;
            float gdy = gfy - giy;
            float gdz = gfz - giz;

            float gv000 = sampleSDF(gix, giy, giz);
            float gv100 = sampleSDF(gix1, giy, giz);
            float gv010 = sampleSDF(gix, giy1, giz);
            float gv110 = sampleSDF(gix1, giy1, giz);
            float gv001 = sampleSDF(gix, giy, giz1);
            float gv101 = sampleSDF(gix1, giy, giz1);
            float gv011 = sampleSDF(gix, giy1, giz1);
            float gv111 = sampleSDF(gix1, giy1, giz1);

            float gv00 = gv000 * (1.0f - gdx) + gv100 * gdx;
            float gv10 = gv010 * (1.0f - gdx) + gv110 * gdx;
            float gv01 = gv001 * (1.0f - gdx) + gv101 * gdx;
            float gv11 = gv011 * (1.0f - gdx) + gv111 * gdx;

            float gv0 = gv00 * (1.0f - gdy) + gv10 * gdy;
            float gv1 = gv01 * (1.0f - gdy) + gv11 * gdy;

            return (gv0 * (1.0f - gdz) + gv1 * gdz) * ((scale.x + scale.y + scale.z) / 3.0f);
        };

        float dist_x_plus  = interpSDF(local_pos + Vec3(step_x, 0.0f, 0.0f));
        float dist_x_minus = interpSDF(local_pos - Vec3(step_x, 0.0f, 0.0f));
        float dist_y_plus  = interpSDF(local_pos + Vec3(0.0f, step_y, 0.0f));
        float dist_y_minus = interpSDF(local_pos - Vec3(0.0f, step_y, 0.0f));
        float dist_z_plus  = interpSDF(local_pos + Vec3(0.0f, 0.0f, step_z));
        float dist_z_minus = interpSDF(local_pos - Vec3(0.0f, 0.0f, step_z));

        Vec3 local_normal(dist_x_plus - dist_x_minus, dist_y_plus - dist_y_minus, dist_z_plus - dist_z_minus);
        float normal_len = local_normal.length();
        if (normal_len > 1e-6f) {
            local_normal = local_normal * (1.0f / normal_len);
        } else {
            local_normal = Vec3(0.0f, 1.0f, 0.0f);
        }

        // Transform local normal to world space
        Vec3 world_normal = obb.local_to_world.transform_vector(local_normal);
        float world_normal_len = world_normal.length();
        if (world_normal_len > 1e-6f) {
            world_normal = world_normal * (1.0f / world_normal_len);
        } else {
            world_normal = Vec3(0.0f, 1.0f, 0.0f);
        }

        // World displacement
        float world_dist = dist_local * avg_scale;
        float penetration = world_thick - world_dist;
        position = position + world_normal * penetration;

        // Resolve velocity in world space
        resolveVelocityAgainstNormal(collider, world_normal, velocity);
    }
}

void resolveMeshBVHCollision(const ParticleColliderDesc& collider,
                             const ParticleColliderOBB& obb,
                             float particle_radius,
                             Vec3& position,
                             Vec3& velocity) {
    if (!collider.local_triangles_cache || collider.local_triangles_cache->empty()) return;
    const auto& triangles = *collider.local_triangles_cache;

    // Transform world position/velocity to local OBB space
    const Matrix4x4 world_to_local = obb.local_to_world.inverse();
    const Vec3 local_pos = world_to_local.transform_point(position);

    float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
    float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
    float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
    float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
    if (avg_scale <= 1e-6f) avg_scale = 1.0f;

    // Calculate scale factor relative to cooked size
    Vec3 cooked_size = collider.sdf_extents / 1.3f;
    if (cooked_size.length_squared() < 1e-6f) {
        cooked_size = obb.local_bounds_max - obb.local_bounds_min;
    }
    Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
    Vec3 scale(
        cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
        cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
        cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
    );
    scale.x = std::max(1e-4f, scale.x);
    scale.y = std::max(1e-4f, scale.y);
    scale.z = std::max(1e-4f, scale.z);

    const float world_thick = std::max(0.0f, collider.thickness) + particle_radius;
    const float local_thick = world_thick / avg_scale;

    // Fast OBB check in scaled local space
    if (local_pos.x < obb.local_bounds_min.x - local_thick || local_pos.x > obb.local_bounds_max.x + local_thick ||
        local_pos.y < obb.local_bounds_min.y - local_thick || local_pos.y > obb.local_bounds_max.y + local_thick ||
        local_pos.z < obb.local_bounds_min.z - local_thick || local_pos.z > obb.local_bounds_max.z + local_thick) {
        return;
    }

    // Find closest triangle
    float min_dist_sq = std::numeric_limits<float>::max();
    Vec3 best_closest(0.0f);
    Vec3 best_normal(0.0f, 1.0f, 0.0f);

    for (const auto& tri : triangles) {
        // Apply scaling to the local cached triangle vertices
        Vec3 lp0 = tri.p0 * scale;
        Vec3 lp1 = tri.p1 * scale;
        Vec3 lp2 = tri.p2 * scale;
        Vec3 lnormal = tri.normal;

        float tri_min_x = std::min({lp0.x, lp1.x, lp2.x});
        float tri_max_x = std::max({lp0.x, lp1.x, lp2.x});
        float tri_min_y = std::min({lp0.y, lp1.y, lp2.y});
        float tri_max_y = std::max({lp0.y, lp1.y, lp2.y});
        float tri_min_z = std::min({lp0.z, lp1.z, lp2.z});
        float tri_max_z = std::max({lp0.z, lp1.z, lp2.z});

        if (local_pos.x < tri_min_x - local_thick || local_pos.x > tri_max_x + local_thick ||
            local_pos.y < tri_min_y - local_thick || local_pos.y > tri_max_y + local_thick ||
            local_pos.z < tri_min_z - local_thick || local_pos.z > tri_max_z + local_thick) {
            continue;
        }

        Vec3 closest = closestPointOnTriangle(local_pos, lp0, lp1, lp2);
        float d_sq = (local_pos - closest).length_squared();
        if (d_sq < min_dist_sq) {
            min_dist_sq = d_sq;
            best_closest = closest;
            best_normal = lnormal;
        }
    }

    if (min_dist_sq > local_thick * local_thick) {
        return;
    }

    float dist_local = std::sqrt(min_dist_sq);
    Vec3 delta = local_pos - best_closest;
    if (Vec3::dot(delta, best_normal) < 0.0f) {
        dist_local = -dist_local; // Penetrated inside
    }

    // Resolve in world space
    Vec3 world_normal = obb.local_to_world.transform_vector(best_normal);
    const float wn_len = world_normal.length();
    if (wn_len > 1e-6f) world_normal = world_normal * (1.0f / wn_len);

    float penetration = world_thick - (dist_local * avg_scale);
    position = position + world_normal * penetration;

    resolveVelocityAgainstNormal(collider, world_normal, velocity);
}

void resolveConvexDecompCollision(const ParticleColliderDesc& collider,
                                  const ParticleColliderOBB& obb,
                                  float particle_radius,
                                  Vec3& position,
                                  Vec3& velocity) {
    if (!collider.octant_min_cache || collider.octant_min_cache->empty()) return;
    const auto& oct_min = *collider.octant_min_cache;
    const auto& oct_max = *collider.octant_max_cache;
    const auto& oct_active = *collider.octant_active_cache;

    // Transform world position/velocity to local OBB space
    const Matrix4x4 world_to_local = obb.local_to_world.inverse();
    const Vec3 local_pos = world_to_local.transform_point(position);

    float scale_x = Vec3(obb.local_to_world.m[0][0], obb.local_to_world.m[1][0], obb.local_to_world.m[2][0]).length();
    float scale_y = Vec3(obb.local_to_world.m[0][1], obb.local_to_world.m[1][1], obb.local_to_world.m[2][1]).length();
    float scale_z = Vec3(obb.local_to_world.m[0][2], obb.local_to_world.m[1][2], obb.local_to_world.m[2][2]).length();
    float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
    if (avg_scale <= 1e-6f) avg_scale = 1.0f;

    // Calculate scale factor relative to cooked size
    Vec3 cooked_size = collider.sdf_extents / 1.3f;
    if (cooked_size.length_squared() < 1e-6f) {
        cooked_size = obb.local_bounds_max - obb.local_bounds_min;
    }
    Vec3 current_size = obb.local_bounds_max - obb.local_bounds_min;
    Vec3 scale(
        cooked_size.x > 1e-6f ? current_size.x / cooked_size.x : 1.0f,
        cooked_size.y > 1e-6f ? current_size.y / cooked_size.y : 1.0f,
        cooked_size.z > 1e-6f ? current_size.z / cooked_size.z : 1.0f
    );
    scale.x = std::max(1e-4f, scale.x);
    scale.y = std::max(1e-4f, scale.y);
    scale.z = std::max(1e-4f, scale.z);

    const float world_thick = std::max(0.0f, collider.thickness) + particle_radius;
    const float local_thick = world_thick / avg_scale;

    // Fast OBB check
    if (local_pos.x < obb.local_bounds_min.x - local_thick || local_pos.x > obb.local_bounds_max.x + local_thick ||
        local_pos.y < obb.local_bounds_min.y - local_thick || local_pos.y > obb.local_bounds_max.y + local_thick ||
        local_pos.z < obb.local_bounds_min.z - local_thick || local_pos.z > obb.local_bounds_max.z + local_thick) {
        return;
    }

    // Resolve collision against the closest active octant bounding box
    float min_dist = 1e30f;
    int best_oct = -1;
    Vec3 best_normal(0.0f, 1.0f, 0.0f);

    for (int o = 0; o < 8; ++o) {
        if (!oct_active[o]) continue;

        // Apply scale to the sub-hulls/octants
        Vec3 omn = (oct_min[o] * scale) - Vec3(local_thick);
        Vec3 omx = (oct_max[o] * scale) + Vec3(local_thick);

        if (local_pos.x >= omn.x && local_pos.x <= omx.x &&
            local_pos.y >= omn.y && local_pos.y <= omx.y &&
            local_pos.z >= omn.z && local_pos.z <= omx.z) {
            
            // We are inside this sub-box!
            float dists[6] = {
                std::abs(local_pos.x - omn.x),
                std::abs(omx.x - local_pos.x),
                std::abs(local_pos.y - omn.y),
                std::abs(omx.y - local_pos.y),
                std::abs(local_pos.z - omn.z),
                std::abs(omx.z - local_pos.z)
            };

            int face = 0;
            for (int f = 1; f < 6; ++f) {
                if (dists[f] < dists[face]) face = f;
            }

            if (dists[face] < min_dist) {
                min_dist = dists[face];
                best_oct = o;
                switch (face) {
                    case 0: best_normal = Vec3(-1.0f, 0.0f, 0.0f); break;
                    case 1: best_normal = Vec3(1.0f, 0.0f, 0.0f); break;
                    case 2: best_normal = Vec3(0.0f, -1.0f, 0.0f); break;
                    case 3: best_normal = Vec3(0.0f, 1.0f, 0.0f); break;
                    case 4: best_normal = Vec3(0.0f, 0.0f, -1.0f); break;
                    case 5: best_normal = Vec3(0.0f, 0.0f, 1.0f); break;
                }
            }
        }
    }

    if (best_oct < 0) return; // No collision

    // Resolve in world space
    Vec3 world_normal = obb.local_to_world.transform_vector(best_normal);
    const float wn_len = world_normal.length();
    if (wn_len > 1e-6f) world_normal = world_normal * (1.0f / wn_len);

    position = position + world_normal * (min_dist * avg_scale);
    resolveVelocityAgainstNormal(collider, world_normal, velocity);
}

} // namespace

bool ParticleSimulationSystem::enabled() const {
    return enabled_ && (alive_count_ > 0 || hasActiveEmitters() || hasActiveGridSimulation());
}

bool ParticleSimulationSystem::hasActiveGridSimulation() const {
    // Flow-source / gas-only domains have no particles, but must still advance:
    // keep the system active while a domain has an enabled flow source feeding it,
    // or any residual density left to advect/dissipate. Fluid (APIC liquid)
    // domains stay active whenever they hold particles — the particles are the
    // source of truth and need every tick to fall under gravity, project, etc.
    if (grid_domains_.empty()) {
        return false;
    }
    for (const auto& source : flow_sources_) {
        if (source.enabled) {
            return true;
        }
    }
    for (const auto& state : grid_domain_states_) {
        if (!state.valid) continue;
        if (state.type == SimulationDomainType::Fluid && !state.particles.empty()) {
            return true;
        }
        if (state.type == SimulationDomainType::Gas && state.active_density_cells > 0) {
            return true;
        }
    }
    return false;
}

void ParticleSimulationSystem::setEnabled(bool enabled) {
    enabled_ = enabled;
}

void ParticleSimulationSystem::reserve(std::size_t capacity) {
    if (capacity > buffers_.alive.size()) {
        resizeStorage(capacity);
    }
}

void ParticleSimulationSystem::clear() {
    buffers_ = ParticleSoABuffers{};
    neighbor_grid_.clear();
    compute_buffers_.capacity = 0;
    compute_buffers_.source_version = 0;
    alive_count_ = 0;
    for (auto& source : flow_sources_) {
        source.fluid_emit_accumulator = 0.0f;
        source.total_emitted_particles = 0;
    }
    ++data_version_;
}

void ParticleSimulationSystem::releaseComputeResources(SimulationComputeContext& compute) {
    auto destroy = [&](ComputeBufferHandle& handle) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
            handle = {};
        }
    };

    destroy(compute_buffers_.position_x);
    destroy(compute_buffers_.position_y);
    destroy(compute_buffers_.position_z);
    destroy(compute_buffers_.velocity_x);
    destroy(compute_buffers_.velocity_y);
    destroy(compute_buffers_.velocity_z);
    destroy(compute_buffers_.age_seconds);
    destroy(compute_buffers_.lifetime_seconds);
    destroy(compute_buffers_.inverse_mass);
    destroy(compute_buffers_.alive);
    compute_buffers_.capacity = 0;
    compute_buffers_.source_version = 0;

    for (auto& buffers : grid_domain_compute_buffers_) {
        releaseGridDomainComputeBuffers(compute, buffers);
    }
    grid_domain_compute_buffers_.clear();
}

std::size_t ParticleSimulationSystem::spawn(const ParticleSpawnDesc& desc) {
    if (!finiteParticleDesc(desc) || desc.lifetime_seconds <= 0.0f) {
        return kInvalidParticle;
    }

    std::size_t index = findDeadSlot();
    if (index == kInvalidParticle) {
        index = buffers_.alive.size();
        resizeStorage(index + 1);
    }

    buffers_.position_x[index] = desc.position.x;
    buffers_.position_y[index] = desc.position.y;
    buffers_.position_z[index] = desc.position.z;
    buffers_.velocity_x[index] = desc.velocity.x;
    buffers_.velocity_y[index] = desc.velocity.y;
    buffers_.velocity_z[index] = desc.velocity.z;
    buffers_.age_seconds[index] = 0.0f;
    buffers_.lifetime_seconds[index] = desc.lifetime_seconds;
    buffers_.inverse_mass[index] = safeInverseMass(desc.mass);

    buffers_.start_size[index] = desc.start_size;
    buffers_.end_size[index] = desc.end_size;
    buffers_.start_opacity[index] = desc.start_opacity;
    buffers_.end_opacity[index] = desc.end_opacity;
    buffers_.start_color_r[index] = desc.start_color.x;
    buffers_.start_color_g[index] = desc.start_color.y;
    buffers_.start_color_b[index] = desc.start_color.z;
    buffers_.end_color_r[index] = desc.end_color.x;
    buffers_.end_color_g[index] = desc.end_color.y;
    buffers_.end_color_b[index] = desc.end_color.z;

    // Current = start at birth (age = 0).
    buffers_.size[index] = desc.start_size;
    buffers_.opacity[index] = desc.start_opacity;
    buffers_.color_r[index] = desc.start_color.x;
    buffers_.color_g[index] = desc.start_color.y;
    buffers_.color_b[index] = desc.start_color.z;
    buffers_.rotation[index] = desc.rotation;
    buffers_.angular_velocity[index] = desc.angular_velocity;

    if (buffers_.alive[index] == 0u) {
        ++alive_count_;
    }
    buffers_.alive[index] = 1u;
    ++data_version_;
    return index;
}

bool ParticleSimulationSystem::kill(std::size_t index) {
    if (index >= buffers_.alive.size() || buffers_.alive[index] == 0u) {
        return false;
    }

    buffers_.alive[index] = 0u;
    if (alive_count_ > 0) {
        --alive_count_;
    }
    ++data_version_;
    return true;
}

std::vector<ParticleEmitterDesc>& ParticleSimulationSystem::emitters() {
    return emitters_;
}

const std::vector<ParticleEmitterDesc>& ParticleSimulationSystem::emitters() const {
    return emitters_;
}

ParticleEmitterDesc& ParticleSimulationSystem::addEmitter(const ParticleEmitterDesc& desc) {
    emitters_.push_back(desc);
    auto& emitter = emitters_.back();
    if (emitter.seed == 0u) {
        emitter.seed = static_cast<uint32_t>(emitters_.size() * 97u + 1u);
    }
    return emitter;
}

bool ParticleSimulationSystem::removeEmitter(std::size_t index) {
    if (index >= emitters_.size()) {
        return false;
    }
    emitters_.erase(emitters_.begin() + static_cast<std::ptrdiff_t>(index));
    return true;
}

void ParticleSimulationSystem::clearEmitters() {
    emitters_.clear();
}

void ParticleSimulationSystem::setEmitterSourceResolver(
    std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> resolver) {
    emitter_source_resolver_ = std::move(resolver);
}

void ParticleSimulationSystem::setEmitterBoundsResolver(
    std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> resolver) {
    emitter_bounds_resolver_ = std::move(resolver);
}

void ParticleSimulationSystem::setEmitterSurfaceSampler(
    std::function<bool(const ParticleEmitterDesc&, uint32_t, ParticleSurfaceSample&)> sampler) {
    emitter_surface_sampler_ = std::move(sampler);
}

std::vector<ParticleColliderDesc>& ParticleSimulationSystem::colliders() {
    return colliders_;
}

const std::vector<ParticleColliderDesc>& ParticleSimulationSystem::colliders() const {
    return colliders_;
}

ParticleColliderDesc& ParticleSimulationSystem::addCollider(const ParticleColliderDesc& desc) {
    colliders_.push_back(desc);
    return colliders_.back();
}

bool ParticleSimulationSystem::removeCollider(std::size_t index) {
    if (index >= colliders_.size()) {
        return false;
    }
    colliders_.erase(colliders_.begin() + static_cast<std::ptrdiff_t>(index));
    return true;
}

void ParticleSimulationSystem::clearColliders() {
    colliders_.clear();
}

void ParticleSimulationSystem::setColliderBoundsResolver(
    std::function<bool(const ParticleColliderDesc&, Vec3&, Vec3&)> resolver) {
    collider_bounds_resolver_ = std::move(resolver);
}

void ParticleSimulationSystem::setColliderOBBResolver(
    std::function<bool(const ParticleColliderDesc&, ParticleColliderOBB&)> resolver) {
    collider_obb_resolver_ = std::move(resolver);
}

void ParticleSimulationSystem::setColliderMeshResolver(
    std::function<bool(const ParticleColliderDesc&, std::vector<SurfaceMeshTriangle>&, uint64_t&)> resolver) {
    collider_mesh_resolver_ = std::move(resolver);
}

std::vector<SimulationGridDomainDesc>& ParticleSimulationSystem::gridDomains() {
    return grid_domains_;
}

const std::vector<SimulationGridDomainDesc>& ParticleSimulationSystem::gridDomains() const {
    return grid_domains_;
}

const std::vector<SimulationGridDomainState>& ParticleSimulationSystem::gridDomainStates() const {
    return grid_domain_states_;
}

const SimulationGpuFoamRenderBuffer* ParticleSimulationSystem::gridDomainFoamRenderBuffer(
    std::size_t domain_index) const {
    if (domain_index >= grid_domain_compute_buffers_.size()) {
        return nullptr;
    }
    const auto& buffer = grid_domain_compute_buffers_[domain_index].foam_render;
    return buffer.valid() ? &buffer : nullptr;
}

bool ParticleSimulationSystem::exportGridDomainToVDB(std::size_t domain_index,
                                                     const std::string& filepath) const {
    if (domain_index >= grid_domain_states_.size()) {
        return false;
    }
    const SimulationGridDomainState& state = grid_domain_states_[domain_index];
    if (!state.valid || state.type != SimulationDomainType::Gas) {
        return false;
    }
    const FluidSim::FluidGrid& grid = state.grid;
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        return false;
    }

    const bool has_density = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density) &&
                             !grid.density.empty();
    const bool has_temp = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Temperature) &&
                          !grid.temperature.empty();
    const bool has_fuel = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Fuel) &&
                          !grid.fuel.empty();
    const bool has_flame = !grid.interaction.empty();

#ifdef OPENVDB_ENABLED
    openvdb::initialize();

    openvdb::FloatGrid::Ptr density_grid = openvdb::FloatGrid::create(0.0f);
    density_grid->setName("density");
    density_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    openvdb::FloatGrid::Accessor density_acc = density_grid->getAccessor();

    openvdb::FloatGrid::Ptr temp_grid = openvdb::FloatGrid::create(0.0f);
    temp_grid->setName("temperature");
    openvdb::FloatGrid::Accessor temp_acc = temp_grid->getAccessor();

    openvdb::FloatGrid::Ptr fuel_grid = openvdb::FloatGrid::create(0.0f);
    fuel_grid->setName("fuel");
    openvdb::FloatGrid::Accessor fuel_acc = fuel_grid->getAccessor();

    openvdb::FloatGrid::Ptr flame_grid = openvdb::FloatGrid::create(0.0f);
    flame_grid->setName("flame");
    openvdb::FloatGrid::Accessor flame_acc = flame_grid->getAccessor();

    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                const std::size_t idx = grid.cellIndex(i, j, k);
                const openvdb::Coord coord(i, j, k);
                if (has_density) {
                    const float d = grid.density[idx];
                    if (d > 1e-6f) density_acc.setValue(coord, d);
                }
                if (has_temp) {
                    const float t = grid.temperature[idx];
                    if (t > 1e-3f) temp_acc.setValue(coord, t);
                }
                if (has_fuel) {
                    const float f = grid.fuel[idx];
                    if (f > 1e-3f) fuel_acc.setValue(coord, f);
                }
                if (has_flame) {
                    const float fl = grid.interaction[idx];
                    if (fl > 1e-3f) flame_acc.setValue(coord, fl);
                }
            }
        }
    }

    openvdb::math::Transform::Ptr xform =
        openvdb::math::Transform::createLinearTransform(grid.voxel_size);
    xform->postTranslate(openvdb::Vec3d(grid.origin.x, grid.origin.y, grid.origin.z));
    density_grid->setTransform(xform);
    temp_grid->setTransform(xform);
    fuel_grid->setTransform(xform);
    flame_grid->setTransform(xform);

    openvdb::GridPtrVec grids;
    if (has_density) grids.push_back(density_grid);
    if (has_temp)    grids.push_back(temp_grid);
    if (has_fuel)    grids.push_back(fuel_grid);
    if (has_flame)   grids.push_back(flame_grid);
    if (grids.empty()) {
        return false;
    }

    try {
        openvdb::io::File file(filepath);
        file.write(grids);
        file.close();
        return true;
    } catch (...) {
        return false;
    }
#else
    // Raw binary fallback (matches the legacy GasSimulator "GASV" layout).
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }
    const int32_t magic = 0x47415356; // "GASV"
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&grid.nx), sizeof(grid.nx));
    file.write(reinterpret_cast<const char*>(&grid.ny), sizeof(grid.ny));
    file.write(reinterpret_cast<const char*>(&grid.nz), sizeof(grid.nz));
    file.write(reinterpret_cast<const char*>(&grid.voxel_size), sizeof(grid.voxel_size));
    if (has_density) {
        file.write(reinterpret_cast<const char*>(grid.density.data()),
                   static_cast<std::streamsize>(grid.density.size() * sizeof(float)));
    }
    if (has_temp) {
        file.write(reinterpret_cast<const char*>(grid.temperature.data()),
                   static_cast<std::streamsize>(grid.temperature.size() * sizeof(float)));
    }
    file.close();
    return file.good();
#endif
}

SimulationGridDomainDesc& ParticleSimulationSystem::addGridDomain(const SimulationGridDomainDesc& desc) {
    grid_domains_.push_back(desc);
    grid_domain_states_.emplace_back();
    return grid_domains_.back();
}

bool ParticleSimulationSystem::removeGridDomain(std::size_t index) {
    if (index >= grid_domains_.size()) {
        return false;
    }
    grid_domains_.erase(grid_domains_.begin() + static_cast<std::ptrdiff_t>(index));
    if (index < grid_domain_states_.size()) {
        grid_domain_states_.erase(grid_domain_states_.begin() + static_cast<std::ptrdiff_t>(index));
    }
    for (auto it = flow_sources_.begin(); it != flow_sources_.end();) {
        if (it->domain_index == static_cast<int>(index)) {
            it = flow_sources_.erase(it);
        } else {
            if (it->domain_index > static_cast<int>(index)) {
                --it->domain_index;
            }
            ++it;
        }
    }
    return true;
}

void ParticleSimulationSystem::clearGridDomains() {
    grid_domains_.clear();
    grid_domain_states_.clear();
    flow_sources_.clear();
}

void ParticleSimulationSystem::resetGridDomainStates() {
    grid_domain_states_.clear();
    grid_domain_states_.resize(grid_domains_.size());
}

void ParticleSimulationSystem::setGridDomainStates(const std::vector<SimulationGridDomainState>& states) {
    grid_domain_states_ = states;
}

void ParticleSimulationSystem::setGridDomainBoundsResolver(
    std::function<bool(const SimulationGridDomainDesc&, Vec3&, Vec3&)> resolver) {
    grid_domain_bounds_resolver_ = std::move(resolver);
}

std::vector<SimulationFlowSourceDesc>& ParticleSimulationSystem::flowSources() {
    return flow_sources_;
}

const std::vector<SimulationFlowSourceDesc>& ParticleSimulationSystem::flowSources() const {
    return flow_sources_;
}

SimulationFlowSourceDesc& ParticleSimulationSystem::addFlowSource(const SimulationFlowSourceDesc& desc) {
    flow_sources_.push_back(desc);
    return flow_sources_.back();
}

bool ParticleSimulationSystem::removeFlowSource(std::size_t index) {
    if (index >= flow_sources_.size()) {
        return false;
    }
    flow_sources_.erase(flow_sources_.begin() + static_cast<std::ptrdiff_t>(index));
    return true;
}

void ParticleSimulationSystem::clearFlowSources() {
    flow_sources_.clear();
}

void ParticleSimulationSystem::setFlowSourceBoundsResolver(
    std::function<bool(const SimulationFlowSourceDesc&, Vec3&, Vec3&)> resolver) {
    flow_source_bounds_resolver_ = std::move(resolver);
}

void ParticleSimulationSystem::setFlowSourceSurfaceSampler(
    std::function<bool(const SimulationFlowSourceDesc&, uint32_t, ParticleSurfaceSample&)> sampler) {
    flow_source_surface_sampler_ = std::move(sampler);
}

std::size_t ParticleSimulationSystem::capacity() const {
    return buffers_.alive.size();
}

std::size_t ParticleSimulationSystem::aliveCount() const {
    return alive_count_;
}

const ParticleSoABuffers& ParticleSimulationSystem::buffers() const {
    return buffers_;
}

const ParticleComputeBuffers& ParticleSimulationSystem::computeBuffers() const {
    return compute_buffers_;
}

const ParticleSimulationStats& ParticleSimulationSystem::stats() const {
    return stats_;
}

void ParticleSimulationSystem::setGravity(const Vec3& gravity) {
    gravity_ = gravity;
}

void ParticleSimulationSystem::setLinearDrag(float drag) {
    linear_drag_ = std::max(0.0f, drag);
}

void ParticleSimulationSystem::setCollisionPlane(float y, bool enabled, float restitution) {
    collision_plane_y_ = y;
    collision_plane_enabled_ = enabled;
    collision_restitution_ = std::clamp(restitution, 0.0f, 1.0f);
}

Vec3 ParticleSimulationSystem::gravity() const {
    return gravity_;
}

float ParticleSimulationSystem::linearDrag() const {
    return linear_drag_;
}

bool ParticleSimulationSystem::collisionPlaneEnabled() const {
    return collision_plane_enabled_;
}

float ParticleSimulationSystem::collisionPlaneY() const {
    return collision_plane_y_;
}

float ParticleSimulationSystem::collisionRestitution() const {
    return collision_restitution_;
}

ParticlePhysicsSettings& ParticleSimulationSystem::physicsSettings() {
    return physics_settings_;
}

const ParticlePhysicsSettings& ParticleSimulationSystem::physicsSettings() const {
    return physics_settings_;
}

void ParticleSimulationSystem::applyPhysicsModePreset(ParticlePhysicsMode mode) {
    physics_settings_.mode = mode;
    switch (mode) {
        case ParticlePhysicsMode::Spark:
            physics_settings_.particle_radius = 0.025f;
            physics_settings_.self_collision_enabled = false;
            physics_settings_.solver_iterations = 1;
            physics_settings_.viscosity = 0.0f;
            physics_settings_.cohesion = 0.0f;
            physics_settings_.pressure_stiffness = 0.0f;
            physics_settings_.rest_density = 1.0f;
            physics_settings_.buoyancy = 0.0f;
            physics_settings_.gravity_scale = 1.0f;
            physics_settings_.vorticity = 0.0f;
            break;
        case ParticlePhysicsMode::Granular:
            physics_settings_.particle_radius = 0.045f;
            physics_settings_.self_collision_enabled = true;
            physics_settings_.solver_iterations = 2;
            physics_settings_.viscosity = 0.15f;
            physics_settings_.cohesion = 0.05f;
            physics_settings_.pressure_stiffness = 0.0f;
            physics_settings_.rest_density = 1500.0f;
            physics_settings_.buoyancy = 0.0f;
            physics_settings_.gravity_scale = 1.0f;
            physics_settings_.vorticity = 0.0f;
            break;
        case ParticlePhysicsMode::Fluid:
            physics_settings_.particle_radius = 0.05f;
            physics_settings_.self_collision_enabled = true;
            physics_settings_.solver_iterations = 4;
            physics_settings_.viscosity = 0.35f;
            physics_settings_.cohesion = 0.25f;
            physics_settings_.pressure_stiffness = 0.65f;
            physics_settings_.rest_density = 1000.0f;
            physics_settings_.buoyancy = 0.0f;
            physics_settings_.gravity_scale = 1.0f;
            physics_settings_.vorticity = 0.0f;
            break;
        case ParticlePhysicsMode::Gas:
            physics_settings_.particle_radius = 0.08f;
            physics_settings_.self_collision_enabled = false;
            physics_settings_.solver_iterations = 1;
            physics_settings_.viscosity = 0.2f;
            physics_settings_.cohesion = 0.0f;
            physics_settings_.pressure_stiffness = 0.15f;
            physics_settings_.rest_density = 1.2f;
            physics_settings_.buoyancy = 1.2f;
            physics_settings_.gravity_scale = 0.1f;
            physics_settings_.vorticity = 0.2f;
            break;
    }
}

void ParticleSimulationSystem::applyQualityModePreset(ParticleQualityMode quality) {
    physics_settings_.quality = quality;
    switch (quality) {
        case ParticleQualityMode::Realtime:
            physics_settings_.max_neighbors_per_particle = 24;
            physics_settings_.solver_iterations = std::clamp(physics_settings_.solver_iterations, 1, 3);
            break;
        case ParticleQualityMode::Preview:
            physics_settings_.max_neighbors_per_particle = 48;
            physics_settings_.solver_iterations = std::clamp(physics_settings_.solver_iterations, 2, 6);
            break;
        case ParticleQualityMode::Offline:
            physics_settings_.max_neighbors_per_particle = 96;
            physics_settings_.solver_iterations = std::clamp(physics_settings_.solver_iterations, 4, 16);
            break;
    }
}

void ParticleSimulationSystem::synchronizeGridDomainsNow() {
    synchronizeGridDomains();
}

void ParticleSimulationSystem::synchronizeGridDomains() {
    if (grid_domain_states_.size() != grid_domains_.size()) {
        grid_domain_states_.resize(grid_domains_.size());
    }

    for (std::size_t i = 0; i < grid_domains_.size(); ++i) {
        auto& domain = grid_domains_[i];
        auto& state = grid_domain_states_[i];

        Vec3 bounds_min = domain.bounds_min;
        Vec3 bounds_max = domain.bounds_max;
        float current_padding = std::max(0.0f, domain.padding);

        if (domain.source_mode == SimulationGridDomainSourceMode::ObjectBounds && grid_domain_bounds_resolver_) {
            Vec3 resolved_min = bounds_min;
            Vec3 resolved_max = bounds_max;
            if (grid_domain_bounds_resolver_(domain, resolved_min, resolved_max)) {
                bounds_min = resolved_min;
                bounds_max = resolved_max;
                domain.bounds_min = resolved_min;
                domain.bounds_max = resolved_max;
            }
        } else if (domain.source_mode == SimulationGridDomainSourceMode::Adaptive) {
            current_padding = 0.0f; // Padding handled internally to preserve floor locking
            Vec3 min_p(1.0e10f, 1.0e10f, 1.0e10f);
            Vec3 max_p(-1.0e10f, -1.0e10f, -1.0e10f);
            bool has_particles = false;

            if (domain.type == SimulationDomainType::Fluid) {
                if (!state.particles.position.empty()) {
                    for (const auto& pos : state.particles.position) {
                        min_p = Vec3::min(min_p, pos);
                        max_p = Vec3::max(max_p, pos);
                    }
                    has_particles = true;
                }
            } else {
                const std::size_t num_particles = buffers_.alive.size();
                for (std::size_t p = 0; p < num_particles; ++p) {
                    if (buffers_.alive[p] != 0u) {
                        Vec3 pos(buffers_.position_x[p], buffers_.position_y[p], buffers_.position_z[p]);
                        min_p = Vec3::min(min_p, pos);
                        max_p = Vec3::max(max_p, pos);
                        has_particles = true;
                    }
                }
            }

            if (has_particles) {
                float pad_val = std::max(0.0f, domain.padding);
                if (pad_val < 1.0e-5f) {
                    pad_val = std::max(domain.voxel_size * 3.0f, 0.05f);
                }
                min_p = min_p - Vec3(pad_val, pad_val, pad_val);
                max_p = max_p + Vec3(pad_val, pad_val, pad_val);

                if (domain.adaptive_lock_floor) {
                    min_p.y = domain.adaptive_floor_y;
                }

                // Grid snapping to voxel size to completely eliminate sub-voxel jitter
                const float voxel_size = domain.voxel_size > 1.0e-6f ? domain.voxel_size : 0.1f;
                min_p.x = std::floor(min_p.x / voxel_size) * voxel_size;
                min_p.y = std::floor(min_p.y / voxel_size) * voxel_size;
                min_p.z = std::floor(min_p.z / voxel_size) * voxel_size;

                max_p.x = std::ceil(max_p.x / voxel_size) * voxel_size;
                max_p.y = std::ceil(max_p.y / voxel_size) * voxel_size;
                max_p.z = std::ceil(max_p.z / voxel_size) * voxel_size;

                bounds_min = min_p;
                bounds_max = max_p;
                domain.bounds_min = bounds_min;
                domain.bounds_max = bounds_max;
            }
        }

        const Vec3 mn = Vec3::min(bounds_min, bounds_max) - current_padding;
        const Vec3 mx = Vec3::max(bounds_min, bounds_max) + current_padding;
        const Vec3 extent = mx - mn;
        const float max_extent = std::max({ extent.x, extent.y, extent.z, 0.001f });

        const int max_auto_res = std::clamp(domain.max_auto_resolution, 32, 512);
        int res_x, res_y, res_z;
        float voxel_size;

        if (domain.preserve_voxel_size_on_resize && domain.voxel_size > 1e-6f) {
            voxel_size = domain.voxel_size;
            res_x = std::clamp(static_cast<int>(std::ceil(std::max(extent.x, 0.001f) / voxel_size)), 8, max_auto_res);
            res_y = std::clamp(static_cast<int>(std::ceil(std::max(extent.y, 0.001f) / voxel_size)), 8, max_auto_res);
            res_z = std::clamp(static_cast<int>(std::ceil(std::max(extent.z, 0.001f) / voxel_size)), 8, max_auto_res);
        } else {
            // Derive per-axis cell counts proportionally from physical extents at the
            // finest voxel_size the max_auto_res allows (max_extent / max_auto_res).
            // This prevents the classic "large flat domain" problem: when all three
            // axes start at max_auto_res and budget clamping reduces them equally, the
            // thin Y axis loses cells disproportionately because max_extent/max_auto_res
            // gives a coarse voxel for the short dimension.
            voxel_size = max_extent / static_cast<float>(max_auto_res);
            res_x = std::clamp(static_cast<int>(std::ceil(std::max(extent.x, 0.001f) / voxel_size)), 8, 1024);
            res_y = std::clamp(static_cast<int>(std::ceil(std::max(extent.y, 0.001f) / voxel_size)), 8, 1024);
            res_z = std::clamp(static_cast<int>(std::ceil(std::max(extent.z, 0.001f) / voxel_size)), 8, 1024);
            domain.voxel_size = voxel_size;
        }

        // max_auto_resolution is the single authority for grid density: the
        // per-axis clamps above already cap every axis at max_auto_res, so the
        // total can never exceed max_auto_res^3. Deriving the cell budget from
        // that same knob makes the UI honest — "Max Auto Resolution = 512" means
        // the solver may actually reach 512 per axis (voxel size permitting),
        // instead of a hidden fixed cell cap silently collapsing a 2 m cube to
        // ~80^3 (= 25 mm voxels). The y-aspect term keeps the extra headroom for
        // flat domains (thin Y) where allocated cells are dominated by empty
        // space above the slab; it only ever raises an already-non-binding
        // budget. A high absolute ceiling remains purely as an OOM guard — the
        // UI's live cell-count + memory preview is the real guardrail for the
        // user's explicit choice. (GPU backends can lift this further once the
        // MGPCG path is live-wired.)
        const std::size_t knob_budget =
            static_cast<std::size_t>(max_auto_res) *
            static_cast<std::size_t>(max_auto_res) *
            static_cast<std::size_t>(max_auto_res);
        constexpr std::size_t MAX_GRID_DOMAIN_CELLS_HARD_CAP = 134217728; // 512^3 OOM guard
        const float y_aspect = std::clamp(extent.y / max_extent, 0.01f, 1.0f);
        const std::size_t adaptive_budget = std::min(
            static_cast<std::size_t>(static_cast<double>(knob_budget) / static_cast<double>(y_aspect)),
            MAX_GRID_DOMAIN_CELLS_HARD_CAP);
        clampGridResolutionToCellBudget(res_x, res_y, res_z, adaptive_budget);

        // The grid is a single-voxel-size (cubic) MAC grid, so each axis spans
        // exactly res_i * voxel_size. To COVER the whole domain box on every
        // axis we must pick the COARSEST per-axis voxel (extent_i / res_i): that
        // guarantees res_i * voxel_size >= extent_i for all i. Taking the
        // minimum instead under-covers the longest axis whenever the per-axis
        // resolutions are not perfectly proportional — which happens routinely
        // because a thin axis gets clamped UP to the 8-cell floor (giving it a
        // finer voxel) or the budget clamp shrinks the largest axis. The classic
        // symptom is a tall object whose Y is silently cropped (e.g. a 10 m
        // column rendered as 8 m) — i.e. the domain looks flattened even though
        // the bounds carry the object's true width/height/depth. Using max keeps
        // the aspect ratio intact; the off-axis over-coverage is at most one
        // extra voxel of headroom (harmless padding at the walls).
        const float vs_x = extent.x / static_cast<float>(std::max(res_x, 1));
        const float vs_y = extent.y / static_cast<float>(std::max(res_y, 1));
        const float vs_z = extent.z / static_cast<float>(std::max(res_z, 1));
        voxel_size = std::max({ vs_x, vs_y, vs_z });
        if (voxel_size < 1e-6f) {
            voxel_size = max_extent / static_cast<float>(std::max({ res_x, res_y, res_z, 1 }));
        }
        domain.voxel_size = voxel_size;
        domain.resolution_x = res_x;
        domain.resolution_y = res_y;
        domain.resolution_z = res_z;
        domain.max_auto_resolution = max_auto_res;

        const uint32_t channels = domain.channels;
        const bool layout_changed =
            !state.valid ||
            state.resolution_x != res_x ||
            state.resolution_y != res_y ||
            state.resolution_z != res_z ||
            state.channels != channels;
        if (layout_changed) {
            // FluidGrid always allocates every channel (MAC staggered layout);
            // the channel mask now gates injection/use, not allocation.
            state.grid.sparse_mode_enabled = domain.use_sparse_tiles || (domain.backend == SimulationDomainBackend::CPU_SparseVDB);
            state.grid.resize(res_x, res_y, res_z, voxel_size, mn);
            ++state.version;
        } else {
            // Object-bound domains can translate/scale without a resolution
            // change; keep the field contents but follow the new origin.
            state.grid.origin = mn;
            state.grid.voxel_size = voxel_size;
        }

        state.type = domain.type;
        state.domain_motion_delta = Vec3(0.0f, 0.0f, 0.0f);

        // Fluid seed AABB follows the domain's translation but NOT its resize.
        // We compare the corner deltas: if both corners shifted by the same
        // vector the user translated the domain → glue the seed to it. If
        // only one corner moved the user is dragging an extent → keep the
        // seed in place (its absolute world coords stay valid).
        if (domain.type == SimulationDomainType::Fluid) {
            if (domain.fluid_seed_anchor_min.x > -9.99e9f) {
                const Vec3 delta_min = mn - domain.fluid_seed_anchor_min;
                const Vec3 delta_max = mx - domain.fluid_seed_anchor_max;
                const Vec3 diff = delta_max - delta_min;
                const float diff_mag = std::abs(diff.x) + std::abs(diff.y) + std::abs(diff.z);
                const float trans_mag = std::abs(delta_min.x) + std::abs(delta_min.y) + std::abs(delta_min.z);
                if (diff_mag < 1e-4f && trans_mag > 1e-6f) {
                    domain.fluid_seed_min = domain.fluid_seed_min + delta_min;
                    domain.fluid_seed_max = domain.fluid_seed_max + delta_min;
                    state.domain_motion_delta = delta_min;
                    for (Vec3& p : state.particles.position) {
                        p = p + delta_min;
                    }
                }
            } else {
                // First sync as a fluid domain. Drop a sensible default seed
                // AABB inside the domain (upper half, quarter-sized) so the
                // user sees the cyan box inside the bounds instead of stuck
                // at the legacy (0..1, 1..1.5, 0..1) world coords that may
                // be far outside an arbitrary domain center.
                const Vec3 domain_extent = mx - mn;
                const Vec3 seed_extent = Vec3(domain_extent.x * 0.45f,
                                              domain_extent.y * 0.35f,
                                              domain_extent.z * 0.45f);
                const Vec3 current_center = (mn + mx) * 0.5f;
                const Vec3 seed_center = Vec3(current_center.x,
                                              mn.y + domain_extent.y * 0.75f,
                                              current_center.z);
                domain.fluid_seed_min = seed_center - seed_extent * 0.5f;
                domain.fluid_seed_max = seed_center + seed_extent * 0.5f;
            }
            domain.fluid_seed_anchor_min = mn;
            domain.fluid_seed_anchor_max = mx;
        }

        state.bounds_min = mn;
        state.bounds_max = mx;
        state.resolution_x = res_x;
        state.resolution_y = res_y;
        state.resolution_z = res_z;
        state.voxel_size = voxel_size;
        state.channels = channels;
        state.valid = domain.enabled && res_x > 0 && res_y > 0 && res_z > 0;
        state.active_density_cells = 0;
        state.max_density = 0.0f;

        // Fluid-only: process a pending seed request from the UI. The legacy
        // FluidObject seeded directly from the panel; the unified path keeps
        // the seed deferred so it applies on the next sim tick with a freshly
        // resized grid.
        if (domain.type == SimulationDomainType::Fluid && domain.fluid_pending_seed && state.valid) {
            if (domain.fluid_replace_on_seed) {
                state.particles.clear();
                state.foam.clear();   // drop stale whitewater with the old liquid
            }
            // Resolve the effective seed AABB. FillLevel mode treats the domain
            // as a resting tank: fill the whole footprint from the floor up to
            // fluid_fill_level of the domain height (skips the long emission /
            // settling transient for standing water). SeedBox uses the explicit
            // user AABB. Particles are emitted at rest (v=0) by seedBox either
            // way, so the result starts in hydrostatic balance.
            const std::size_t seed_budget =
                state.particles.size() < domain.fluid_max_particles
                    ? domain.fluid_max_particles - state.particles.size()
                    : 0u;

            // ppc is a STABILITY constant, not a budget knob: it must stay > 1 so
            // the cells carry enough samples to build real internal pressure
            // (incompressibility). At 1 ppc the liquid is under-resolved and just
            // collapses. So ppc is fixed and, when the budget can't afford the
            // full target fill, the fill HEIGHT drops instead (complete layers
            // from the floor up) — fully-resolved, stable, pile-free.
            const int ppc = std::max(1, domain.fluid_seed_particles_per_cell);

            Vec3 seed_lo = domain.fluid_seed_min;
            Vec3 seed_hi = domain.fluid_seed_max;
            if (domain.fluid_seed_mode == FluidSeedMode::FillLevel) {
                computeFluidFillSeedAABB(state.bounds_min, state.bounds_max,
                                         state.grid.voxel_size,
                                         domain.fluid_fill_level,
                                         domain.fluid_fill_wall_margin,
                                         ppc, seed_budget,
                                         seed_lo, seed_hi);
            }

            // The seeded density IS the rest density: couple the solver's
            // density-correction target to the seeded ppc so a freshly filled
            // tank starts at exactly "target per cell" (over == 0). Without this,
            // a seed denser than the fixed default target makes every cell
            // over-populated and the density-targeted pressure projection
            // permanently expels particles upward (the tank "rises").
            domain.fluid_params.particles_per_cell = ppc;
            // Stable per-domain seed: index-based, so jitter patterns are
            // reproducible across runs and don't depend on heap addresses
            // (the desc vector can reallocate as domains are added).
            Fluid::seedBox(state.particles,
                           state.grid,
                           seed_lo,
                           seed_hi,
                           ppc,
                           /*seed=*/static_cast<uint32_t>(i + 1u) * 2654435761u,
                           seed_budget);
            domain.fluid_pending_seed = false;
        }
    }
}

void ParticleSimulationSystem::injectFlowSourcesIntoGridDomains(float dt, float time_seconds) {
    if (flow_sources_.empty() || grid_domain_states_.empty()) {
        return;
    }

    const float time_scale = std::max(0.0f, dt);
    for (auto& source : flow_sources_) {
        if (!source.enabled ||
            source.domain_index < 0 ||
            source.domain_index >= static_cast<int>(grid_domain_states_.size())) {
            continue;
        }

        // Time Limit check (Houdini/Blender flow emitter style)
        if (source.use_time_limit) {
            if (time_seconds < source.start_time || time_seconds > source.end_time) {
                continue;
            }
        }

        auto& state = grid_domain_states_[static_cast<std::size_t>(source.domain_index)];
        if (!state.valid) {
            continue;
        }
        // Fluid (APIC liquid) flow sources spawn particles instead of
        // injecting density. Spawn rate accumulator survives across steps so
        // fractional rate*dt counts emit correctly. Capped by the domain's
        // max_particles.
        if (state.type == SimulationDomainType::Fluid) {
            const auto& fluid_domain = grid_domains_[static_cast<std::size_t>(source.domain_index)];
            const float rate = std::max(0.0f, source.fluid_particles_per_second);
            source.fluid_emit_accumulator += rate * std::max(0.0f, dt);
            int emit_count = static_cast<int>(source.fluid_emit_accumulator);
            if (emit_count <= 0) continue;
            source.fluid_emit_accumulator -= static_cast<float>(emit_count);

            // Hysteresis gate: reseed trims over-populated cells every step,
            // creating a small capacity gap even at max_particles. Without a
            // dead-band, the emitter fills that gap each step producing a
            // visible "trickle" of particles at full capacity.
            // Only allow emission when at least 1% capacity is available so
            // normal filling (empty → full) is unaffected but steady-state
            // at-max oscillation is suppressed.
            const std::size_t max_p = fluid_domain.fluid_max_particles;
            const std::size_t cur_p = state.particles.size();
            const std::size_t dead_band = std::max<std::size_t>(1u, max_p / 100u);
            const std::size_t remaining =
                (cur_p + dead_band < max_p) ? (max_p - cur_p) : 0u;
            emit_count = std::min<int>(emit_count, static_cast<int>(remaining));
            if (emit_count <= 0) continue;

            // Particle budget limit check
            if (source.use_particle_limit) {
                int limit_rem = source.max_emitted_particles - source.total_emitted_particles;
                if (limit_rem <= 0) {
                    continue;
                }
                emit_count = std::min<int>(emit_count, limit_rem);
            }
            if (emit_count <= 0) continue;

            // Resolve spawn volume for ObjectBounds; Point uses source.position
            // + radius sphere; MeshSurface samples per-particle below.
            Vec3 bounds_min = source.position - Vec3(source.radius);
            Vec3 bounds_max = source.position + Vec3(source.radius);
            if (source.source_mode == SimulationFlowSourceMode::ObjectBounds && flow_source_bounds_resolver_) {
                Vec3 resolved_min, resolved_max;
                if (flow_source_bounds_resolver_(source, resolved_min, resolved_max)) {
                    bounds_min = Vec3::min(resolved_min, resolved_max);
                    bounds_max = Vec3::max(resolved_min, resolved_max);
                }
            }
            // Over-pack guard: a high particles/sec dumped into a small spawn
            // volume in ONE step stacks dozens of particles into a single cell.
            // The density-correction term then sees a huge overshoot and blasts
            // them outward laterally — the source "splatters" into a disc/plate.
            // Cap this step's emission at what the spawn volume can physically
            // hold at peak packing, and return the surplus to the accumulator so
            // it emits over the following steps instead of all at once. (Mesh
            // surface emission spreads over an area, so it is left uncapped.)
            if (source.source_mode != SimulationFlowSourceMode::MeshSurface) {
                const float h = std::max(1e-4f, state.grid.voxel_size);
                float spawn_volume;
                if (source.source_mode == SimulationFlowSourceMode::ObjectBounds) {
                    const Vec3 ext = bounds_max - bounds_min;
                    spawn_volume = std::max(0.0f, ext.x) * std::max(0.0f, ext.y) * std::max(0.0f, ext.z);
                } else {
                    const float r = std::max(1e-4f, source.radius);
                    spawn_volume = (4.0f / 3.0f) * 3.14159265358979f * r * r * r;
                }
                const double spawn_cells = std::max(1.0, static_cast<double>(spawn_volume) / (h * h * h));
                const int pack_ceiling = std::max({ fluid_domain.fluid_params.particles_per_cell,
                                                    fluid_domain.fluid_params.reseed_max_per_cell, 1 });
                const int cap = std::max(1, static_cast<int>(spawn_cells * static_cast<double>(pack_ceiling)));
                if (emit_count > cap) {
                    source.fluid_emit_accumulator += static_cast<float>(emit_count - cap);
                    emit_count = cap;
                }
            }

            // Per-source-per-particle hash seed so jitter is deterministic but
            // not synchronized across sources.
            const uint32_t source_seed_base =
                static_cast<uint32_t>(reinterpret_cast<std::uintptr_t>(&source) >> 4) * 2654435761u;

            state.particles.reserve(state.particles.size() + static_cast<std::size_t>(emit_count));
            for (int p = 0; p < emit_count; ++p) {
                const uint32_t s = source_seed_base ^ (static_cast<uint32_t>(p) * 2246822519u);
                const float u1 = hashUnitFloat(s);
                const float u2 = hashUnitFloat(s ^ 0xdeadbeefu);
                const float u3 = hashUnitFloat(s ^ 0x9e3779b9u);
                Vec3 spawn_pos;
                Vec3 spawn_normal(0.0f, 0.0f, 0.0f); // valid only for MeshSurface
                if (source.source_mode == SimulationFlowSourceMode::MeshSurface && flow_source_surface_sampler_) {
                    ParticleSurfaceSample sample;
                    if (flow_source_surface_sampler_(source, s, sample)) {
                        // Offset slightly along normal so particles spawn just
                        // off the surface, not embedded.
                        spawn_pos = sample.position + sample.normal * std::max(0.001f, source.radius * 0.25f);
                        spawn_normal = sample.normal;
                    } else {
                        spawn_pos = source.position;
                    }
                } else if (source.source_mode == SimulationFlowSourceMode::ObjectBounds) {
                    spawn_pos.x = bounds_min.x + u1 * (bounds_max.x - bounds_min.x);
                    spawn_pos.y = bounds_min.y + u2 * (bounds_max.y - bounds_min.y);
                    spawn_pos.z = bounds_min.z + u3 * (bounds_max.z - bounds_min.z);
                } else {
                    // Point: rejection sample inside unit sphere then scale.
                    Vec3 d(u1 * 2.0f - 1.0f, u2 * 2.0f - 1.0f, u3 * 2.0f - 1.0f);
                    const float r2 = d.x * d.x + d.y * d.y + d.z * d.z;
                    if (r2 > 1.0f) {
                        const float inv = 1.0f / std::sqrt(r2);
                        d = d * (inv * std::cbrt(hashUnitFloat(s ^ 0x68e31da4u)));
                    }
                    spawn_pos = source.position + d * source.radius;
                }
                // Break the laminar stream: an APIC liquid has nothing to
                // disperse a column of identical-velocity particles mid-air, so
                // without a per-particle perturbation the emitted mass falls as
                // a coherent sheet/plate. Add random jitter scaled by the
                // emission speed (0 spread => exact source.velocity, laminar).
                Vec3 emit_vel = source.velocity;
                // MeshSurface + emit-along-normal: redirect the emission speed
                // along the local surface normal so the liquid sprays off the
                // geometry instead of all moving in one global direction.
                if (source.fluid_emit_along_normal &&
                    source.source_mode == SimulationFlowSourceMode::MeshSurface) {
                    const float nlen = spawn_normal.length();
                    if (nlen > 1e-5f) {
                        emit_vel = spawn_normal * (source.velocity.length() / nlen);
                    }
                }
                if (source.fluid_velocity_spread > 0.0f) {
                    const float jitter_mag = source.fluid_velocity_spread * emit_vel.length();
                    if (jitter_mag > 1e-6f) {
                        const uint32_t vs = s ^ 0x1b56c4e9u;
                        const Vec3 jitter(
                            hashUnitFloat(vs)               * 2.0f - 1.0f,
                            hashUnitFloat(vs ^ 0x7feb352du) * 2.0f - 1.0f,
                            hashUnitFloat(vs ^ 0x846ca68bu) * 2.0f - 1.0f);
                        emit_vel = emit_vel + jitter * jitter_mag;
                    }
                }
                state.particles.emit(spawn_pos, emit_vel);
            }
            source.total_emitted_particles += emit_count;
            continue;
        }

        Vec3 source_center = source.position;
        float source_radius = std::max(0.001f, source.radius);
        if (source.source_mode == SimulationFlowSourceMode::ObjectBounds && flow_source_bounds_resolver_) {
            Vec3 source_min;
            Vec3 source_max;
            if (!flow_source_bounds_resolver_(source, source_min, source_max)) {
                continue;
            }
            const Vec3 mn = Vec3::min(source_min, source_max);
            const Vec3 mx = Vec3::max(source_min, source_max);
            source_center = (mn + mx) * 0.5f;
            source_radius = std::max(source_radius, (mx - mn).length() * 0.25f);
        }

        const Vec3 mn = state.bounds_min;
        const Vec3 mx = state.bounds_max;
        if (source_center.x + source_radius < mn.x || source_center.x - source_radius > mx.x ||
            source_center.y + source_radius < mn.y || source_center.y - source_radius > mx.y ||
            source_center.z + source_radius < mn.z || source_center.z - source_radius > mx.z) {
            continue;
        }

        FluidSim::FluidGrid& grid = state.grid;
        if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
            continue;
        }

        // Cell range overlapping the source sphere (grid space).
        float fi0, fj0, fk0, fi1, fj1, fk1;
        grid.worldToGrid(source_center - Vec3(source_radius), fi0, fj0, fk0);
        grid.worldToGrid(source_center + Vec3(source_radius), fi1, fj1, fk1);
        const int min_x = std::clamp(static_cast<int>(std::floor(fi0)), 0, grid.nx - 1);
        const int max_x = std::clamp(static_cast<int>(std::ceil(fi1)), 0, grid.nx - 1);
        const int min_y = std::clamp(static_cast<int>(std::floor(fj0)), 0, grid.ny - 1);
        const int max_y = std::clamp(static_cast<int>(std::ceil(fj1)), 0, grid.ny - 1);
        const int min_z = std::clamp(static_cast<int>(std::floor(fk0)), 0, grid.nz - 1);
        const int max_z = std::clamp(static_cast<int>(std::ceil(fk1)), 0, grid.nz - 1);

        const float inv_radius = 1.0f / source_radius;
        const float falloff = std::max(0.0f, source.falloff);
        const bool write_density = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density);
        const bool write_temperature = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Temperature);
        const bool write_fuel = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Fuel);
        const bool write_pressure = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Pressure);
        const bool write_velocity = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Velocity);
        const float density_amount = source.density * time_scale;
        const float temperature_amount = source.temperature * time_scale;
        const float fuel_amount = source.fuel * time_scale;
        const Vec3 velocity_amount = source.velocity * time_scale;

        for (int z = min_z; z <= max_z; ++z) {
            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    const Vec3 cell_center = grid.gridToWorld(x, y, z);
                    const float d = (cell_center - source_center).length() * inv_radius;
                    if (d > 1.0f) {
                        continue;
                    }
                    const float weight = falloff <= 0.0f ? 1.0f : std::pow(std::max(0.0f, 1.0f - d), falloff);
                    const std::size_t cell = grid.cellIndex(x, y, z);

                    if (write_density) {
                        grid.density[cell] += density_amount * weight;
                    }
                    if (write_temperature) {
                        grid.temperature[cell] += temperature_amount * weight;
                    }
                    if (write_fuel) {
                        grid.fuel[cell] += fuel_amount * weight;
                    }
                    if (write_pressure) {
                        grid.pressure[cell] += density_amount * 0.2f * weight;
                    }
                    if (write_velocity) {
                        // Splat the source velocity onto the cell's MAC faces.
                        grid.velXAt(x, y, z)     += velocity_amount.x * weight;
                        grid.velXAt(x + 1, y, z) += velocity_amount.x * weight;
                        grid.velYAt(x, y, z)     += velocity_amount.y * weight;
                        grid.velYAt(x, y + 1, z) += velocity_amount.y * weight;
                        grid.velZAt(x, y, z)     += velocity_amount.z * weight;
                        grid.velZAt(x, y, z + 1) += velocity_amount.z * weight;
                    }
                }
            }
        }
    }
}

void ParticleSimulationSystem::stepGridDomains(const SimulationContext& context) {
    if (grid_domains_.empty()) {
        if (context.compute && !grid_domain_compute_buffers_.empty()) {
            for (auto& buffers : grid_domain_compute_buffers_) {
                releaseGridDomainComputeBuffers(*context.compute, buffers);
            }
            grid_domain_compute_buffers_.clear();
        }
        return;
    }

    const float dt = context.dt;
    synchronizeGridDomains();
    if (context.compute) {
        while (grid_domain_compute_buffers_.size() > grid_domain_states_.size()) {
            releaseGridDomainComputeBuffers(*context.compute, grid_domain_compute_buffers_.back());
            grid_domain_compute_buffers_.pop_back();
        }
        grid_domain_compute_buffers_.resize(grid_domain_states_.size());
    }

    // Sources and particles deposit into the grid before it is advanced.
    injectFlowSourcesIntoGridDomains(dt, context.time_seconds);

    if (alive_count_ > 0) {
        const float radius = std::max(physics_settings_.particle_radius, 0.001f);
        const float density_amount = physics_settings_.mode == ParticlePhysicsMode::Gas ? 0.08f : 0.035f;
        const float temperature_amount = physics_settings_.mode == ParticlePhysicsMode::Gas ? std::max(0.0f, physics_settings_.buoyancy) * 0.05f : 0.0f;

        for (std::size_t particle = 0; particle < buffers_.alive.size(); ++particle) {
            if (buffers_.alive[particle] == 0u) {
                continue;
            }

            const Vec3 position(buffers_.position_x[particle], buffers_.position_y[particle], buffers_.position_z[particle]);
            const Vec3 velocity(buffers_.velocity_x[particle], buffers_.velocity_y[particle], buffers_.velocity_z[particle]);
            const float opacity = particle < buffers_.opacity.size() ? std::clamp(buffers_.opacity[particle], 0.0f, 1.0f) : 1.0f;

            for (auto& state : grid_domain_states_) {
                if (!state.valid) {
                    continue;
                }
                // Particle SoA → grid deposit is gas-only (writes into
                // density/temperature/velocity which Fluid uses as scratch).
                if (state.type == SimulationDomainType::Fluid) {
                    continue;
                }
                const Vec3 mn = state.bounds_min;
                const Vec3 mx = state.bounds_max;
                if (position.x < mn.x || position.x > mx.x ||
                    position.y < mn.y || position.y > mx.y ||
                    position.z < mn.z || position.z > mx.z) {
                    continue;
                }

                FluidSim::FluidGrid& grid = state.grid;
                float fi, fj, fk;
                grid.worldToGrid(position, fi, fj, fk);
                const int x = std::clamp(static_cast<int>(fi), 0, grid.nx - 1);
                const int y = std::clamp(static_cast<int>(fj), 0, grid.ny - 1);
                const int z = std::clamp(static_cast<int>(fk), 0, grid.nz - 1);
                const std::size_t cell = grid.cellIndex(x, y, z);

                if (hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density)) {
                    grid.density[cell] += density_amount * opacity * std::max(1.0f, radius * 16.0f);
                }
                if (hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Temperature)) {
                    grid.temperature[cell] += temperature_amount * opacity;
                }
                if (hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Pressure)) {
                    grid.pressure[cell] += density_amount * 0.25f * opacity;
                }
                if (hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Velocity)) {
                    // Carry the particle's motion onto the cell's MAC faces.
                    grid.velXAt(x, y, z)     += velocity.x * 0.5f;
                    grid.velXAt(x + 1, y, z) += velocity.x * 0.5f;
                    grid.velYAt(x, y, z)     += velocity.y * 0.5f;
                    grid.velYAt(x, y + 1, z) += velocity.y * 0.5f;
                    grid.velZAt(x, y, z)     += velocity.z * 0.5f;
                    grid.velZAt(x, y, z + 1) += velocity.z * 0.5f;
                }
            }
        }
    }

    // Advance each grid domain with the shared fluid solver (advect -> wall bcs
    // -> buoyancy -> force fields -> vorticity -> dissipation -> pressure
    // projection). Fire/combustion is intentionally not run here yet.
    GridFluid::SolverParams base_params;
    base_params.gravity = gravity_;
    base_params.buoyancy_heat = physics_settings_.buoyancy;
    base_params.buoyancy_density = physics_settings_.buoyancy * 0.15f;
    base_params.ambient_temperature = 0.0f;
    base_params.vorticity = std::max(0.0f, physics_settings_.vorticity);
    base_params.pressure_iterations = 40;
    base_params.sor_omega = 1.7f;
    base_params.density_dissipation = physics_settings_.mode == ParticlePhysicsMode::Gas
        ? std::max(0.0f, physics_settings_.viscosity) * 0.35f
        : 0.5f;
    base_params.temperature_dissipation = 0.5f;
    base_params.fuel_dissipation = 0.25f;
    base_params.velocity_dissipation = std::max(0.0f, physics_settings_.viscosity) * 0.1f;
    base_params.max_velocity = 1000.0f;

    // ── Moving-collider velocity (momentum transfer) ─────────────────────────
    // Resolve each collider's current world centre and difference it against the
    // previous step to get a linear velocity, stamped into grid.solid_vel by the
    // voxelizer below so the advect step hands a moving collider's momentum to
    // the fluid it sweeps/pushes. Computed ONCE per step (shared by every domain).
    {
        const std::size_t nc = colliders_.size();
        collider_velocities_.assign(nc, Vec3(0.0f, 0.0f, 0.0f));
        if (prev_collider_centers_.size() != nc) {
            prev_collider_centers_.assign(nc, Vec3(0.0f, 0.0f, 0.0f));
            prev_collider_center_valid_.assign(nc, 0u);
        }
        const float inv_dt = (dt > 1e-6f) ? (1.0f / dt) : 0.0f;
        for (std::size_t ci = 0; ci < nc; ++ci) {
            const auto& c = colliders_[ci];
            Vec3 center(0.0f, 0.0f, 0.0f);
            bool have_center = false;
            if (c.enabled) {
                switch (c.source_mode) {
                    case ParticleColliderSourceMode::Sphere: {
                        Vec3 mn, mx;
                        if (!c.source_name.empty() && collider_bounds_resolver_ &&
                            collider_bounds_resolver_(c, mn, mx)) {
                            center = (Vec3::min(mn, mx) + Vec3::max(mn, mx)) * 0.5f;
                        } else {
                            center = c.sphere_center;
                        }
                        have_center = true;
                        break;
                    }
                    case ParticleColliderSourceMode::Capsule: {
                        Vec3 mn, mx;
                        if (!c.source_name.empty() && collider_bounds_resolver_ &&
                            collider_bounds_resolver_(c, mn, mx)) {
                            center = (Vec3::min(mn, mx) + Vec3::max(mn, mx)) * 0.5f;
                        } else {
                            center = (c.capsule_start + c.capsule_end) * 0.5f;
                        }
                        have_center = true;
                        break;
                    }
                    case ParticleColliderSourceMode::ObjectAABB: {
                        Vec3 mn = c.bounds_min, mx = c.bounds_max;
                        if (collider_bounds_resolver_ && !c.source_name.empty())
                            collider_bounds_resolver_(c, mn, mx);
                        center = (Vec3::min(mn, mx) + Vec3::max(mn, mx)) * 0.5f;
                        have_center = true;
                        break;
                    }
                    case ParticleColliderSourceMode::ObjectOBB:
                    case ParticleColliderSourceMode::ObjectMeshSDF:
                    case ParticleColliderSourceMode::ObjectConvexDecomp:
                    case ParticleColliderSourceMode::ObjectMeshBVH: {
                        ParticleColliderOBB obb;
                        if (collider_obb_resolver_ && collider_obb_resolver_(c, obb)) {
                            const Vec3 lc = (obb.local_bounds_min + obb.local_bounds_max) * 0.5f;
                            center = obb.local_to_world * lc;
                            have_center = true;
                        }
                        break;
                    }
                    default: break; // PlaneY: treated as static
                }
            }
            if (have_center) {
                if (prev_collider_center_valid_[ci] && inv_dt > 0.0f) {
                    collider_velocities_[ci] = (center - prev_collider_centers_[ci]) * inv_dt;
                }
                prev_collider_centers_[ci] = center;
                prev_collider_center_valid_[ci] = 1u;
            } else {
                prev_collider_center_valid_[ci] = 0u;
            }
        }
    }

    for (std::size_t i = 0; i < grid_domain_states_.size(); ++i) {
        auto& state = grid_domain_states_[i];
        if (!state.valid) {
            continue;
        }

        // Fluid (APIC liquid) domains route to a different solver. The MAC
        // grid is used as scratch (vel_x/vel_y/vel_z + pressure/divergence);
        // density/temperature/fuel channels are untouched. Idle gate: skip
        // when there are no particles to advance.
        if (state.type == SimulationDomainType::Fluid) {
            if (state.particles.empty()) {
                state.fluid_stats = Fluid::APICSolverStats{};
                continue;
            }
            auto fluid_params = (i < grid_domains_.size())
                ? grid_domains_[i].fluid_params
                : Fluid::APICSolverParams{};
            // Mirror the domain wall mode onto the particle solver so "Open
            // (Outflow)" actually drains instead of clamping like a sealed box.
            if (i < grid_domains_.size()) {
                switch (grid_domains_[i].boundary_mode) {
                    case SimulationGridDomainBoundaryMode::Open:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Open;
                        break;
                    case SimulationGridDomainBoundaryMode::Periodic:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Periodic;
                        break;
                    case SimulationGridDomainBoundaryMode::Closed:
                    default:
                        fluid_params.boundary = Fluid::APICSolverParams::BoundaryMode::Closed;
                        break;
                }
            }
            const float motion_coupling = std::clamp(fluid_params.domain_motion_coupling, 0.0f, 1.0f);
            const float motion_mag =
                std::abs(state.domain_motion_delta.x) +
                std::abs(state.domain_motion_delta.y) +
                std::abs(state.domain_motion_delta.z);
            const Vec3 container_velocity_delta =
                (dt > 1e-6f && motion_coupling > 0.0f && motion_mag > 1e-7f)
                    ? state.domain_motion_delta * (motion_coupling / dt)
                    : Vec3(0.0f, 0.0f, 0.0f);
            bool gpu_integrated_forces = false;
            const bool fluid_gpu_requested = (i < grid_domains_.size()) &&
                (grid_domains_[i].backend == SimulationDomainBackend::GPU_CUDA ||
                 grid_domains_[i].backend == SimulationDomainBackend::GPU_Vulkan);
            const bool fluid_gpu_compute_available =
                context.compute && context.compute->supportsDispatch();
            const bool force_fields_require_cpu =
                context.force_snapshot && !context.force_snapshot->empty();
            if (fluid_gpu_requested &&
                !force_fields_require_cpu &&
                fluid_gpu_compute_available &&
                i < grid_domain_compute_buffers_.size()) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    gpu_integrated_forces = runGpuFluidParticleIntegrateForces(state,
                                                                               fluid_params,
                                                                               container_velocity_delta,
                                                                               dt,
                                                                               context.compute,
                                                                               gpu_buffers);
                }
            }
            if (!gpu_integrated_forces && motion_mag > 1e-7f) {
                const std::size_t particle_count = state.particles.velocity.size();
                for (std::size_t pi = 0; pi < particle_count; ++pi) {
                    state.particles.velocity[pi] = state.particles.velocity[pi] + container_velocity_delta;
                }
            }
            // Stamp the active collider set into grid.solid[] every step so
            // Fluid::step's pressure projection + enforceSolidBoundaries see
            // up-to-date boundaries (works for moving/scaled colliders too).
            voxelizeCollidersIntoGrid(state.grid,
                                       colliders_,
                                       collider_bounds_resolver_,
                                       collider_obb_resolver_,
                                       &collider_velocities_);
            // Variational solid coupling: fractional MAC-face open weights for
            // sub-grid-accurate boundaries + moving-collider splash. Cheap (only
            // the collider neighbourhood is super-sampled); skipped when the flag
            // is off so the binary path stays available as a fallback.
            if (fluid_params.variational_solids) {
                computeSolidFaceWeights(state.grid,
                                        colliders_,
                                        collider_bounds_resolver_,
                                        collider_obb_resolver_);
            } else {
                // Weights aren't maintained while variational is off; mark them
                // stale so the next on-frame does a full open-init (the colliders
                // may have moved far during the gap, beyond the incremental reset).
                state.grid.collider_weights_init = false;
            }
            auto step_params = fluid_params;
            step_params.max_particles = (i < grid_domains_.size()) ? grid_domains_[i].fluid_max_particles : 100000;

            // One-shot GPU MGPCG correctness self-test. Runs on a synthetic
            // isolated grid the first time any fluid domain is stepped — does
            // not perturb this domain's state. Always logs one line (the result
            // or a "no GPU dispatch" warning) so the outcome is visible without
            // needing an env var. Env var RAYTROPHI_MGPCG_SELFTEST can force it
            // too, but it is no longer required.
            // Validated PASS (GPU CG matches CPU PCG to float precision), so the
            // self-test no longer auto-runs. Re-enable on demand by setting the
            // env var RAYTROPHI_MGPCG_SELFTEST (runs once, logs the result).
            static bool s_mgpcg_selftest_done = false;
            if (!s_mgpcg_selftest_done && std::getenv("RAYTROPHI_MGPCG_SELFTEST") != nullptr) {
                s_mgpcg_selftest_done = true;
                SCENE_LOG_INFO(std::string("[MGPCG SelfTest] Running GPU MGPCG correctness self-test..."));
                validateGpuFluidMGPCG(context.compute);
            }

            // ── GPU P2G ──────────────────────────────────────────────────────
            float gpu_p2g_ms = 0.0f;
            if (gpu_integrated_forces) {
                step_params.external_forces_preintegrated = true;
                const auto gpu_p2g_begin = SimulationClock::now();
                if (context.compute && i < grid_domain_compute_buffers_.size()) {
                    auto& gpu_buffers = grid_domain_compute_buffers_[i];
                    if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                        step_params.p2g_precomputed = runGpuFluidP2G(state,
                                                                     context.compute,
                                                                     gpu_buffers);
                    }
                }
                if (step_params.p2g_precomputed)
                    gpu_p2g_ms = elapsedMilliseconds(gpu_p2g_begin, SimulationClock::now());
            }

            // ── GPU pressure (MGPCG) + GPU G2P ───────────────────────────────
            // The pressure projection runs on the GPU via a Jacobi-preconditioned
            // CG (runGpuFluidMGPCGPressure), validated against the CPU PCG+MIC(0)
            // to float precision. Falls back to the full CPU step if any GPU
            // stage fails.
            //
            // Pattern (all-GPU pressure+G2P when compute backend dispatches):
            //   Call 1 (stop_after_viscosity=true): forces(opt) + P2G(opt) +
            //     boundary + viscosity → returns; grid.vel = pre-pressure field.
            //   GPU: upload pre-pressure vel to scratch (FLIP snapshot) → MGPCG
            //     pressure projection → GPU G2P → download particle vel/affine.
            //   Call 2 (pressure_g2p_precomputed=true): air_drag + damping +
            //     advect + reseed only.
            //   Any GPU step failing falls back to the full CPU step below.
            float gpu_g2p_ms = 0.0f;
            float gpu_pressure_ms = 0.0f;
            bool  g2p_on_gpu = false;
            Fluid::APICSolverStats gpu_mgpcg_stats;

            const bool try_gpu_g2p =
                fluid_params.free_surface &&
                // Periodic boundaries need the wrap-coupled CPU PCG (the GPU MGPCG
                // still treats out-of-grid as solid, i.e. behaves as Closed); fall
                // back to CPU so Periodic is correct on every device.
                fluid_params.boundary != Fluid::APICSolverParams::BoundaryMode::Periodic &&
                context.compute &&
                context.compute->supportsDispatch() &&
                i < grid_domain_compute_buffers_.size();

            if (try_gpu_g2p) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    // Call 1: physics up to viscosity, STOP before pressure so the
                    // GPU MGPCG solver does the pressure projection. After this,
                    // grid.vel is the pre-pressure (FLIP snapshot) MAC field.
                    auto call1_params = step_params;
                    call1_params.stop_after_viscosity = true;
                    Fluid::APICSolverStats call1_stats;
                    Fluid::step(state.particles, state.grid, call1_params, dt,
                                gpu_integrated_forces ? nullptr : context.force_snapshot,
                                context.time_seconds, &call1_stats);

                    // The current MAC velocity IS the FLIP pre-pressure snapshot.
                    // Upload it to scratch BEFORE the pressure solve overwrites
                    // grid.vel with the projected (divergence-free) field.
                    const bool has_flip =
                        gpu_buffers.scratch_vel_x.valid() &&
                        gpu_buffers.scratch_vel_y.valid() &&
                        gpu_buffers.scratch_vel_z.valid() &&
                        fluid_params.flip_blend > 0.0f;
                    bool upload_ok = true;
                    if (has_flip) {
                        upload_ok =
                            context.compute->uploadBuffer(gpu_buffers.scratch_vel_x,
                                state.grid.vel_x.data(), state.grid.vel_x.size() * sizeof(float)) &&
                            context.compute->uploadBuffer(gpu_buffers.scratch_vel_y,
                                state.grid.vel_y.data(), state.grid.vel_y.size() * sizeof(float)) &&
                            context.compute->uploadBuffer(gpu_buffers.scratch_vel_z,
                                state.grid.vel_z.data(), state.grid.vel_z.size() * sizeof(float));
                    }

                    // GPU pressure projection (MGPCG). Build the fluid mask from
                    // particle positions, then solve. On success grid.vel holds
                    // the projected field and the GPU vel buffers match it.
                    bool pressure_on_gpu = false;
                    if (upload_ok) {
                        static std::vector<float> s_fluid_mask_gpu; // function-static scratch
                        buildFluidMaskFromParticles(state.grid, state.particles, s_fluid_mask_gpu);
                        const auto gpu_pressure_begin = SimulationClock::now();
                        pressure_on_gpu = runGpuFluidMGPCGPressure(state, fluid_params, dt,
                                                                   context.compute, gpu_buffers,
                                                                   s_fluid_mask_gpu,
                                                                   &gpu_mgpcg_stats);
                        if (pressure_on_gpu) {
                            // CPU PCG zeros solid-adjacent faces during the
                            // projection update. The GPU gradient kernel uses
                            // the fluid mask matrix, then we mirror that final
                            // no-flow clamp here before G2P samples velocities.
                            enforceGridSolidFaceBoundaries(state.grid);
                            gpu_pressure_ms = elapsedMilliseconds(gpu_pressure_begin, SimulationClock::now());
                        }
                    }

                    if (pressure_on_gpu) {
                        const auto gpu_g2p_begin = SimulationClock::now();
                        g2p_on_gpu = runGpuFluidG2P(state, fluid_params, dt,
                                                     context.compute, gpu_buffers, has_flip);
                        if (g2p_on_gpu) {
                            gpu_g2p_ms = elapsedMilliseconds(gpu_g2p_begin, SimulationClock::now());
                            // Patch stats from Call 1 into fluid_stats so the
                            // second call's reset doesn't clobber them.
                            state.fluid_stats = call1_stats;
                        }
                    }

                    if (!g2p_on_gpu) {
                        // GPU pressure or G2P failed — fall through to the full
                        // CPU step below (re-runs boundary+viscosity+pressure+G2P
                        // from the post-Call-1 grid; g2p_on_gpu stays false).
                    }
                }
            }

            if (g2p_on_gpu) {
                // Call 2: GPU G2P already done; run only tail stages.
                step_params.pressure_g2p_precomputed = true;
                Fluid::step(state.particles, state.grid, step_params, dt,
                            nullptr, context.time_seconds, &state.fluid_stats);
                state.fluid_stats.g2p_ms       = gpu_g2p_ms;
                state.fluid_stats.g2p_on_gpu   = true;
                state.fluid_stats.pressure_ms     = gpu_pressure_ms;
                state.fluid_stats.pressure_on_gpu = true;
                state.fluid_stats.pressure_cg_iterations = gpu_mgpcg_stats.pressure_cg_iterations;
                state.fluid_stats.pressure_cg_max_iterations = gpu_mgpcg_stats.pressure_cg_max_iterations;
                state.fluid_stats.pressure_cg_dot_count = gpu_mgpcg_stats.pressure_cg_dot_count;
                state.fluid_stats.pressure_cg_dot_ms = gpu_mgpcg_stats.pressure_cg_dot_ms;
                state.fluid_stats.pressure_cg_multigrid = gpu_mgpcg_stats.pressure_cg_multigrid;
                state.fluid_stats.pressure_cg_final_relative_residual =
                    gpu_mgpcg_stats.pressure_cg_final_relative_residual;
            } else {
                // Full CPU path (either GPU G2P not engaged or failed).
                Fluid::step(state.particles, state.grid, step_params, dt,
                            gpu_integrated_forces ? nullptr : context.force_snapshot,
                            context.time_seconds, &state.fluid_stats);
            }

            if (step_params.p2g_precomputed) {
                state.fluid_stats.p2g_ms     = gpu_p2g_ms;
                state.fluid_stats.p2g_on_gpu = true;
            }

            // ── Whitewater (spray/foam/bubbles) — Ihmsen 2012 ────────────────
            // Secondary render-only particles generated from the post-step liquid
            // (relative velocity / wave crest / kinetic energy). Never fed back
            // into the pressure solve, so it cannot affect liquid mass/stability.
            // stepFoam early-outs cheaply when fluid_foam_params.enabled is false.
            if (i < grid_domains_.size()) {
                const auto& fparams = grid_domains_[i].fluid_foam_params;
                const uint32_t foam_seed =
                    static_cast<uint32_t>(context.time_seconds * 600.0f) +
                    static_cast<uint32_t>(i) * 9176u + 1u;
                Fluid::stepFoam(state.particles, state.grid, state.foam, fparams,
                                fluid_params.gravity, dt, foam_seed, &state.foam_stats);
            }

            // ── GPU density splat ────────────────────────────────────────────
            const auto density_begin = SimulationClock::now();
            bool density_on_gpu = false;
            if (fluid_gpu_requested && context.compute &&
                context.compute->supportsDispatch() &&
                i < grid_domain_compute_buffers_.size()) {
                auto& gpu_buffers = grid_domain_compute_buffers_[i];
                if (ensureGridDomainComputeBuffers(*context.compute, gpu_buffers, state.grid)) {
                    density_on_gpu = runGpuFluidDensitySplat(state,
                                                             fluid_params,
                                                             context.compute,
                                                             gpu_buffers);
                    static bool logged_gpu_density_fallback = false;
                    if (!density_on_gpu && !logged_gpu_density_fallback) {
                        SCENE_LOG_WARN("[SimCompute] GPU APIC fluid density splat failed; falling back to CPU NanoVDB density bridge.");
                        logged_gpu_density_fallback = true;
                    }
                }
            }
            if (!density_on_gpu)
                splatFluidDensityCPU(state, fluid_params);

            const auto density_end = SimulationClock::now();
            state.fluid_stats.density_ms           = elapsedMilliseconds(density_begin, density_end);
            state.fluid_stats.density_on_gpu       = density_on_gpu;
            state.fluid_stats.active_fluid_cells   = state.active_density_cells;
            state.fluid_stats.forces_on_gpu        = gpu_integrated_forces;
            state.fluid_stats.gpu_requested        = fluid_gpu_requested;
            state.fluid_stats.gpu_compute_available = fluid_gpu_compute_available;
            state.fluid_stats.compute_device = fluid_gpu_compute_available && context.compute
                ? context.compute->backendName() : "CPU";
            state.fluid_stats.gpu_fallback =
                fluid_gpu_requested &&
                (!gpu_integrated_forces || !step_params.p2g_precomputed || !density_on_gpu);

            // GPU status string. When g2p_on_gpu is set the all-GPU path ran,
            // which now includes the MGPCG pressure projection (Jacobi-PCG).
            if (!fluid_gpu_requested) {
                state.fluid_stats.gpu_status = "CPU reference path";
            } else if (!fluid_gpu_compute_available) {
                state.fluid_stats.gpu_status = "GPU requested, but no simulation compute backend available; CPU fallback.";
            } else if (force_fields_require_cpu) {
                state.fluid_stats.gpu_status = "GPU partial: force fields are CPU-only; force/P2G on CPU.";
            } else if (!gpu_integrated_forces) {
                state.fluid_stats.gpu_status = "GPU requested on " + state.fluid_stats.compute_device + ", but force integration failed; CPU fallback.";
            } else if (!step_params.p2g_precomputed) {
                state.fluid_stats.gpu_status = "GPU requested on " + state.fluid_stats.compute_device + ", but P2G failed; CPU P2G fallback.";
            } else if (!g2p_on_gpu) {
                state.fluid_stats.gpu_status = "GPU partial on " + state.fluid_stats.compute_device + ": forces/P2G/density. Pressure+G2P CPU fallback (PCG).";
            } else if (!density_on_gpu) {
                state.fluid_stats.gpu_status = "GPU on " + state.fluid_stats.compute_device + ": forces/P2G/pressure(MGPCG)/G2P. Density CPU fallback.";
            } else {
                state.fluid_stats.gpu_status = "GPU on " + state.fluid_stats.compute_device + ": forces/P2G/pressure(MGPCG)/G2P/density. Advect+reseed CPU.";
            }
            continue;
        }

        // Skip idle domains (no content, no source, no particles) so empty
        // grids do not pay for a full projection every frame. (Coarse gate;
        // sparse-tile activity tracking is a later optimization.)
        bool has_source = false;
        for (const auto& source : flow_sources_) {
            if (source.enabled && source.domain_index == static_cast<int>(i)) {
                has_source = true;
                break;
            }
        }
        const bool has_content = state.active_density_cells > 0 || state.max_density > 1e-5f;
        if (!has_source && !has_content && alive_count_ == 0) {
            continue;
        }

        // Stamp the active collider set into grid.solid[] before the gas solver
        // runs, exactly like the Fluid path above, so GridFluid::step's solid
        // boundary enforcement + pressure projection treat colliders as walls
        // (moving colliders carry momentum via grid.solid_vel). voxelize is a
        // no-op-cheap dirty-region update when nothing moved / no colliders.
        voxelizeCollidersIntoGrid(state.grid,
                                  colliders_,
                                  collider_bounds_resolver_,
                                  collider_obb_resolver_,
                                  &collider_velocities_);
        const bool domain_has_solid =
            state.grid.solid.size() == static_cast<std::size_t>(state.grid.getCellCount()) &&
            std::any_of(state.grid.solid.begin(), state.grid.solid.end(),
                        [](uint8_t s) { return s != 0u; });

        GridFluid::SolverParams params = base_params;
        if (i < grid_domains_.size()) {
            const auto& domain = grid_domains_[i];
            switch (domain.boundary_mode) {
                case SimulationGridDomainBoundaryMode::Closed:
                    params.boundary = GridFluid::Boundary::Closed;
                    break;
                case SimulationGridDomainBoundaryMode::Periodic:
                    params.boundary = GridFluid::Boundary::Periodic;
                    break;
                case SimulationGridDomainBoundaryMode::Open:
                default:
                    params.boundary = GridFluid::Boundary::Open;
                    break;
            }
            // Per-domain combustion settings.
            params.fire_enabled = domain.fire_enabled;
            params.ignition_temperature = domain.ignition_temperature;
            params.burn_rate = domain.burn_rate;
            params.heat_release = domain.heat_release;
            params.smoke_generation = domain.smoke_generation;
            params.flame_dissipation = domain.flame_dissipation;
            params.max_temperature = domain.fire_max_temperature;
            params.expansion = domain.fire_expansion;
            // Per-domain procedural turbulence.
            params.turbulence_strength = domain.turbulence_strength;
            params.turbulence_scale = domain.turbulence_scale;
            params.turbulence_octaves = domain.turbulence_octaves;
            params.turbulence_lacunarity = domain.turbulence_lacunarity;
            params.turbulence_persistence = domain.turbulence_persistence;
            params.turbulence_speed = domain.turbulence_speed;
        }
        params.channel_density = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density);
        params.channel_temperature = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Temperature);
        params.channel_fuel = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Fuel);
        params.channel_velocity = hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Velocity);

        const bool is_gpu_backend = (i < grid_domains_.size())
            ? (grid_domains_[i].backend == SimulationDomainBackend::GPU_CUDA ||
               grid_domains_[i].backend == SimulationDomainBackend::GPU_Vulkan)
            : g_sim_use_gpu_solver;

        // The GPU grid pressure/advection kernels don't yet read grid.solid or
        // the thermal-expansion divergence target, so a gas domain that contains
        // a collider OR uses fire expansion falls back to the CPU solver (which
        // honours both). Plain solid-free incompressible GPU gas is unaffected.
        const bool use_gpu_grid_solver =
            is_gpu_backend &&
            params.channel_velocity &&
            !domain_has_solid &&
            !(params.expansion > 0.0f) &&
            context.compute &&
            context.compute->supportsDispatch();
        SimulationGridDomainComputeBuffers* gpu_buffers = nullptr;
        if (use_gpu_grid_solver && i < grid_domain_compute_buffers_.size()) {
            gpu_buffers = &grid_domain_compute_buffers_[i];
            if (!ensureGridDomainComputeBuffers(*context.compute, *gpu_buffers, state.grid)) {
                gpu_buffers = nullptr;
                static bool logged_gpu_buffer_fallback = false;
                if (!logged_gpu_buffer_fallback) {
                    SCENE_LOG_WARN("[SimCompute] GPU grid buffer allocation failed; falling back to CPU grid solver.");
                    logged_gpu_buffer_fallback = true;
                }
            }
        }
        const bool gpu_grid_ready = gpu_buffers != nullptr;
        bool gpu_velocity_advection_pre_run = false;
        bool gpu_scalar_advection_ok = false;
        if (gpu_grid_ready) {
            gpu_velocity_advection_pre_run = runGpuVelocityAdvection(state.grid, dt, context.compute, *gpu_buffers);
            if (!gpu_velocity_advection_pre_run) {
                static bool logged_gpu_velocity_fallback = false;
                if (!logged_gpu_velocity_fallback) {
                    SCENE_LOG_WARN("[SimCompute] GPU grid velocity advection failed; falling back to CPU grid solver.");
                    logged_gpu_velocity_fallback = true;
                }
            }
        }
        if (gpu_velocity_advection_pre_run && (params.channel_density || params.channel_temperature || params.channel_fuel)) {
            gpu_scalar_advection_ok = runGpuScalarAdvection(state.grid, params, dt, context.compute, *gpu_buffers);
            if (!gpu_scalar_advection_ok) {
                static bool logged_gpu_advection_fallback = false;
                if (!logged_gpu_advection_fallback) {
                    SCENE_LOG_WARN("[SimCompute] GPU grid scalar advection failed; falling back to CPU advection.");
                    logged_gpu_advection_fallback = true;
                }
            }
        }
        params.skip_velocity_advection = gpu_velocity_advection_pre_run;
        params.skip_scalar_advection = gpu_scalar_advection_ok;
        params.skip_velocity_dissipation_clamp = gpu_grid_ready;
        params.skip_pressure_projection = gpu_grid_ready;

        // stepSparseVDB doesn't honour grid.solid yet, so a sparse-backend domain
        // that contains a collider runs on the dense (solid-aware) step instead.
        const bool is_sparse_vdb = (i < grid_domains_.size()) &&
                                   (grid_domains_[i].backend == SimulationDomainBackend::CPU_SparseVDB) &&
                                   !domain_has_solid;
        if (is_sparse_vdb) {
            GridFluid::stepSparseVDB(state.grid, params, dt, context.force_snapshot, context.time_seconds);
        } else {
            GridFluid::step(state.grid, params, dt, context.force_snapshot, context.time_seconds);
        }
        if (gpu_grid_ready) {
            if (!runGpuVelocityDissipationClamp(state.grid, params, dt, context.compute, *gpu_buffers)) {
                static bool logged_gpu_velocity_post_fallback = false;
                if (!logged_gpu_velocity_post_fallback) {
                    SCENE_LOG_WARN("[SimCompute] GPU grid velocity dissipation/clamp failed; falling back to CPU.");
                    logged_gpu_velocity_post_fallback = true;
                }
                applyCpuVelocityDissipationClamp(state.grid, params, dt);
            }
            if (!runGpuPressureProjection(state.grid, params, dt, context.compute, *gpu_buffers)) {
                static bool logged_gpu_projection_fallback = false;
                if (!logged_gpu_projection_fallback) {
                    SCENE_LOG_WARN("[SimCompute] GPU grid pressure projection failed; falling back to CPU projection.");
                    logged_gpu_projection_fallback = true;
                }
                GridFluid::SolverParams fallback_params = params;
                fallback_params.skip_pressure_projection = false;
                GridFluid::projectPressure(state.grid, fallback_params, dt);
            }
        }
    }

    for (auto& state : grid_domain_states_) {
        if (!state.valid) {
            continue;
        }
        state.active_density_cells = 0;
        state.max_density = 0.0f;
        // Density-channel scan is a gas-only readback. Fluid domains don't
        // touch grid.density; skip the O(cells) walk to avoid burning cycles.
        if (state.type == SimulationDomainType::Gas &&
            hasGridChannel(state.channels, SimulationGridDomainChannelFlags::Density)) {
            for (float value : state.grid.density) {
                if (value > 1e-4f) {
                    ++state.active_density_cells;
                    state.max_density = std::max(state.max_density, value);
                }
            }
        }
        ++state.version;
    }
}

void ParticleSimulationSystem::step(const SimulationContext& context) {
    const auto total_start = SimulationClock::now();
    stats_ = ParticleSimulationStats{};
    stats_.capacity = buffers_.alive.size();
    stats_.emitter_count = emitters_.size();
    stats_.collider_count = colliders_.size();
    stats_.domain_count = grid_domains_.size();

    if (!enabled_ || context.dt <= 0.0f) {
        return;
    }

    const auto emit_start = SimulationClock::now();
    emitFromEmitters(context);
    const auto emit_end = SimulationClock::now();
    stats_.emit_ms = elapsedMilliseconds(emit_start, emit_end);

    if (alive_count_ == 0) {
        // Pure grid/fluid sims (no global particles) still need the mesh-collider
        // caches (triangles / convex octants / BVH) refreshed before voxelization —
        // the integrate path below builds them, but this early-out skips it. The
        // call is version-gated, so it's a no-op once the caches are warm.
        refreshResolvedColliders(std::max(0.0f, physics_settings_.particle_radius));
        const auto grid_start = SimulationClock::now();
        stepGridDomains(context);
        const auto grid_end = SimulationClock::now();
        stats_.grid_domain_ms = elapsedMilliseconds(grid_start, grid_end);
        stats_.alive_count = alive_count_;
        stats_.capacity = buffers_.alive.size();
        stats_.total_ms = elapsedMilliseconds(total_start, SimulationClock::now());
        return;
    }

    const float dt = context.dt;
    const float effective_drag = std::max(0.0f, linear_drag_ + physics_settings_.viscosity);
    const float drag_factor = effective_drag > 0.0f ? std::max(0.0f, 1.0f - effective_drag * dt) : 1.0f;

    const auto integrate_start = SimulationClock::now();
    refreshResolvedColliders(std::max(0.0f, physics_settings_.particle_radius));
    for (std::size_t i = 0; i < buffers_.alive.size(); ++i) {
        if (buffers_.alive[i] == 0u) {
            continue;
        }

        buffers_.age_seconds[i] += dt;
        if (buffers_.age_seconds[i] >= buffers_.lifetime_seconds[i]) {
            buffers_.alive[i] = 0u;
            if (alive_count_ > 0) {
                --alive_count_;
            }
            continue;
        }

        const Vec3 previous_position(buffers_.position_x[i], buffers_.position_y[i], buffers_.position_z[i]);
        Vec3 position = previous_position;
        Vec3 velocity(buffers_.velocity_x[i], buffers_.velocity_y[i], buffers_.velocity_z[i]);
        Vec3 acceleration = gravity_ * physics_settings_.gravity_scale;
        if (physics_settings_.mode == ParticlePhysicsMode::Gas && physics_settings_.buoyancy != 0.0f) {
            acceleration = acceleration + Vec3(0.0f, physics_settings_.buoyancy, 0.0f);
        }
        if (context.force_snapshot) {
            acceleration = acceleration + context.force_snapshot->evaluateAt(
                position,
                context.time_seconds,
                velocity,
                SimulationSystemKind::Particle);
        }

        velocity = (velocity + acceleration * dt) * drag_factor;
        position = position + velocity * dt;

        applyColliders(position, velocity, &previous_position);

        buffers_.position_x[i] = position.x;
        buffers_.position_y[i] = position.y;
        buffers_.position_z[i] = position.z;
        buffers_.velocity_x[i] = velocity.x;
        buffers_.velocity_y[i] = velocity.y;
        buffers_.velocity_z[i] = velocity.z;

        // Evaluate over-life visual attributes (linear start -> end across lifetime).
        const float lifetime = buffers_.lifetime_seconds[i];
        const float t = lifetime > 1e-6f
            ? std::clamp(buffers_.age_seconds[i] / lifetime, 0.0f, 1.0f)
            : 0.0f;
        buffers_.size[i] = lerpf(buffers_.start_size[i], buffers_.end_size[i], t);
        buffers_.opacity[i] = lerpf(buffers_.start_opacity[i], buffers_.end_opacity[i], t);
        buffers_.color_r[i] = lerpf(buffers_.start_color_r[i], buffers_.end_color_r[i], t);
        buffers_.color_g[i] = lerpf(buffers_.start_color_g[i], buffers_.end_color_g[i], t);
        buffers_.color_b[i] = lerpf(buffers_.start_color_b[i], buffers_.end_color_b[i], t);
        buffers_.rotation[i] += buffers_.angular_velocity[i] * dt;
    }
    const auto integrate_end = SimulationClock::now();
    stats_.integrate_ms = elapsedMilliseconds(integrate_start, integrate_end);

    const auto self_collision_start = SimulationClock::now();
    solveSelfCollisions(dt);
    const auto self_collision_end = SimulationClock::now();
    stats_.self_collision_ms = elapsedMilliseconds(self_collision_start, self_collision_end);

    const auto grid_start = SimulationClock::now();
    stepGridDomains(context);
    const auto grid_end = SimulationClock::now();
    stats_.grid_domain_ms = elapsedMilliseconds(grid_start, grid_end);

    ++data_version_;
    const auto upload_start = SimulationClock::now();
    uploadToCompute(context);
    const auto upload_end = SimulationClock::now();
    stats_.upload_ms = elapsedMilliseconds(upload_start, upload_end);
    stats_.alive_count = alive_count_;
    stats_.capacity = buffers_.alive.size();
    stats_.domain_count = grid_domains_.size();
    stats_.total_ms = elapsedMilliseconds(total_start, SimulationClock::now());
}

void ParticleSimulationSystem::refreshResolvedColliders(float particle_radius) {
    resolved_colliders_.clear();
    resolved_colliders_.reserve(colliders_.size());

    for (auto& collider : colliders_) {
        if (!collider.enabled) {
            continue;
        }

        ResolvedCollider resolved;
        resolved.desc = collider;

        if (collider.source_mode == ParticleColliderSourceMode::Sphere) {
            if (!collider.source_name.empty() && collider_bounds_resolver_) {
                Vec3 min_bound = collider.bounds_min;
                Vec3 max_bound = collider.bounds_max;
                if (collider_bounds_resolver_(collider, min_bound, max_bound)) {
                    const Vec3 mn = Vec3::min(min_bound, max_bound);
                    const Vec3 mx = Vec3::max(min_bound, max_bound);
                    resolved.desc.sphere_center = (mn + mx) * 0.5f;
                    resolved.desc.sphere_radius = std::max(0.001f, (mx - mn).length() * 0.5f);
                }
            }
        } else if (collider.source_mode == ParticleColliderSourceMode::Capsule) {
            if (!collider.source_name.empty() && collider_bounds_resolver_) {
                Vec3 min_bound = collider.bounds_min;
                Vec3 max_bound = collider.bounds_max;
                if (collider_bounds_resolver_(collider, min_bound, max_bound)) {
                    const Vec3 mn = Vec3::min(min_bound, max_bound);
                    const Vec3 mx = Vec3::max(min_bound, max_bound);
                    const Vec3 center = (mn + mx) * 0.5f;
                    const Vec3 extent = mx - mn;
                    const float min_side = std::min({ extent.x, extent.y, extent.z });
                    resolved.desc.capsule_radius = std::max(0.001f, min_side * 0.5f);
                    if (extent.x >= extent.y && extent.x >= extent.z) {
                        resolved.desc.capsule_start = Vec3(mn.x, center.y, center.z);
                        resolved.desc.capsule_end = Vec3(mx.x, center.y, center.z);
                    } else if (extent.y >= extent.x && extent.y >= extent.z) {
                        resolved.desc.capsule_start = Vec3(center.x, mn.y, center.z);
                        resolved.desc.capsule_end = Vec3(center.x, mx.y, center.z);
                    } else {
                        resolved.desc.capsule_start = Vec3(center.x, center.y, mn.z);
                        resolved.desc.capsule_end = Vec3(center.x, center.y, mx.z);
                    }
                }
            }
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectAABB) {
            resolved.desc.thickness += particle_radius;
            if (collider_bounds_resolver_) {
                Vec3 min_bound = collider.bounds_min;
                Vec3 max_bound = collider.bounds_max;
                if (collider_bounds_resolver_(collider, min_bound, max_bound)) {
                    resolved.desc.bounds_min = min_bound;
                    resolved.desc.bounds_max = max_bound;
                }
            }
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectOBB) {
            resolved.desc.thickness += particle_radius;
            resolved.has_obb = collider_obb_resolver_ && collider_obb_resolver_(collider, resolved.obb);
            if (!resolved.has_obb) {
                continue;
            }
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectMeshSDF) {
            resolved.desc.thickness += particle_radius;
            resolved.has_obb = collider_obb_resolver_ && collider_obb_resolver_(collider, resolved.obb);
            if (!resolved.has_obb) {
                continue;
            }
        } else if (collider.source_mode == ParticleColliderSourceMode::ObjectConvexDecomp ||
                   collider.source_mode == ParticleColliderSourceMode::ObjectMeshBVH) {
            resolved.desc.thickness += particle_radius;
            resolved.has_obb = collider_obb_resolver_ && collider_obb_resolver_(collider, resolved.obb);
            if (!resolved.has_obb) {
                continue;
            }

            uint64_t current_version = 0;
            std::vector<SurfaceMeshTriangle> tris;
            if (collider_mesh_resolver_ && collider_mesh_resolver_(collider, tris, current_version)) {
                if (collider.last_mesh_cache_version != current_version || !collider.local_triangles_cache) {
                    auto local_tris = std::make_shared<std::vector<SurfaceMeshTriangle>>();
                    const Matrix4x4 world_to_local = resolved.obb.local_to_world.inverse();
                    
                    float ratio = std::clamp(collider.decimation_ratio, 0.01f, 1.0f);
                    std::size_t stride = static_cast<std::size_t>(1.0f / ratio);
                    if (stride < 1) stride = 1;

                    for (std::size_t i = 0; i < tris.size(); i += stride) {
                        SurfaceMeshTriangle local_tri = tris[i];
                        local_tri.p0 = world_to_local.transform_point(local_tri.p0);
                        local_tri.p1 = world_to_local.transform_point(local_tri.p1);
                        local_tri.p2 = world_to_local.transform_point(local_tri.p2);
                        local_tri.normal = world_to_local.transform_vector(local_tri.normal);
                        const float ln_len = local_tri.normal.length();
                        if (ln_len > 1e-6f) local_tri.normal = local_tri.normal * (1.0f / ln_len);
                        local_tris->push_back(local_tri);
                    }

                    collider.local_triangles_cache = local_tris;

                    if (collider.source_mode == ParticleColliderSourceMode::ObjectConvexDecomp) {
                        Vec3 center = (resolved.obb.local_bounds_min + resolved.obb.local_bounds_max) * 0.5f;
                        auto oct_min = std::make_shared<std::vector<Vec3>>(8, Vec3(1e30f));
                        auto oct_max = std::make_shared<std::vector<Vec3>>(8, Vec3(-1e30f));
                        auto oct_active = std::make_shared<std::vector<bool>>(8, false);

                        for (const auto& tri : *local_tris) {
                            Vec3 tri_center = (tri.p0 + tri.p1 + tri.p2) * 0.3333f;
                            int oct_idx = 0;
                            if (tri_center.x > center.x) oct_idx |= 1;
                            if (tri_center.y > center.y) oct_idx |= 2;
                            if (tri_center.z > center.z) oct_idx |= 4;

                            (*oct_active)[oct_idx] = true;
                            (*oct_min)[oct_idx] = Vec3::min((*oct_min)[oct_idx], tri.p0);
                            (*oct_min)[oct_idx] = Vec3::min((*oct_min)[oct_idx], tri.p1);
                            (*oct_min)[oct_idx] = Vec3::min((*oct_min)[oct_idx], tri.p2);
                            (*oct_max)[oct_idx] = Vec3::max((*oct_max)[oct_idx], tri.p0);
                            (*oct_max)[oct_idx] = Vec3::max((*oct_max)[oct_idx], tri.p1);
                            (*oct_max)[oct_idx] = Vec3::max((*oct_max)[oct_idx], tri.p2);
                        }

                        collider.octant_min_cache = oct_min;
                        collider.octant_max_cache = oct_max;
                        collider.octant_active_cache = oct_active;
                    } else if (collider.source_mode == ParticleColliderSourceMode::ObjectMeshBVH) {
                        // Build the accelerated triangle BVH once (same version
                        // gate as the triangle cache) so per-step voxelization is
                        // a logarithmic nearest-point query, not a linear scan.
                        auto bvh = std::make_shared<ColliderMeshBVH>();
                        std::vector<ColliderMeshBVH::Triangle> bvh_tris;
                        bvh_tris.reserve(local_tris->size());
                        for (const auto& t : *local_tris) bvh_tris.push_back({ t.p0, t.p1, t.p2 });
                        bvh->build(std::move(bvh_tris));
                        collider.mesh_bvh_cache = bvh;
                    }

                    collider.last_mesh_cache_version = current_version;
                }

                resolved.desc.local_triangles_cache = collider.local_triangles_cache;
                resolved.desc.octant_min_cache = collider.octant_min_cache;
                resolved.desc.octant_max_cache = collider.octant_max_cache;
                resolved.desc.octant_active_cache = collider.octant_active_cache;
                resolved.desc.mesh_bvh_cache = collider.mesh_bvh_cache;
            }
        }

        resolved_colliders_.push_back(resolved);
    }
}

void ParticleSimulationSystem::applyColliders(Vec3& position, Vec3& velocity, const Vec3* previous_position) const {
    const float particle_radius = std::max(0.0f, physics_settings_.particle_radius);
    if (collision_plane_enabled_ &&
        ((previous_position && previous_position->y >= collision_plane_y_ + particle_radius && position.y < collision_plane_y_ + particle_radius) ||
         position.y < collision_plane_y_ + particle_radius)) {
        position.y = collision_plane_y_ + particle_radius;
        if (velocity.y < 0.0f) {
            velocity.y = -velocity.y * collision_restitution_;
        }
    }

    for (const auto& resolved_collider : resolved_colliders_) {
        const auto& collider = resolved_collider.desc;

        if (collider.source_mode == ParticleColliderSourceMode::PlaneY) {
            const float plane_y = collider.plane_y + particle_radius;
            if ((previous_position && previous_position->y >= plane_y && position.y < plane_y) ||
                position.y < plane_y) {
                position.y = plane_y;
                if (velocity.y < 0.0f) {
                    const float restitution = std::clamp(collider.restitution, 0.0f, 1.0f);
                    const float friction = std::clamp(collider.friction, 0.0f, 1.0f);
                    velocity.x *= 1.0f - friction;
                    velocity.z *= 1.0f - friction;
                    velocity.y = -velocity.y * restitution;
                }
            }
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::Sphere) {
            if (previous_position && !resolveSweptSphereCollision(collider, particle_radius, *previous_position, position, velocity)) {
                resolveSphereCollision(collider, particle_radius, position, velocity);
            } else if (!previous_position) {
                resolveSphereCollision(collider, particle_radius, position, velocity);
            }
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::Capsule) {
            if (previous_position && !resolveSweptCapsuleCollision(collider, particle_radius, *previous_position, position, velocity)) {
                resolveCapsuleCollision(collider, particle_radius, position, velocity);
            } else if (!previous_position) {
                resolveCapsuleCollision(collider, particle_radius, position, velocity);
            }
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::ObjectAABB) {
            if (previous_position && !resolveSweptAABBCollision(collider, *previous_position, position, velocity)) {
                resolveAABBCollision(collider, position, velocity);
            } else if (!previous_position) {
                resolveAABBCollision(collider, position, velocity);
            }
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::ObjectOBB) {
            if (previous_position && !resolveSweptOBBCollision(collider, resolved_collider.obb, *previous_position, position, velocity)) {
                resolveOBBCollision(collider, resolved_collider.obb, position, velocity);
            } else if (!previous_position) {
                resolveOBBCollision(collider, resolved_collider.obb, position, velocity);
            }
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::ObjectMeshSDF) {
            resolveSDFCollision(collider, resolved_collider.obb, particle_radius, position, velocity);
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::ObjectConvexDecomp) {
            resolveConvexDecompCollision(collider, resolved_collider.obb, particle_radius, position, velocity);
            continue;
        }

        if (collider.source_mode == ParticleColliderSourceMode::ObjectMeshBVH) {
            resolveMeshBVHCollision(collider, resolved_collider.obb, particle_radius, position, velocity);
            continue;
        }
    }
}

void ParticleSimulationSystem::buildNeighborGrid(float cell_size) {
    neighbor_grid_.clear();
    if (alive_count_ == 0 || cell_size <= 1e-6f) {
        return;
    }

    neighbor_grid_.reserve(alive_count_);
    const float inv_cell = 1.0f / cell_size;
    for (std::size_t i = 0; i < buffers_.alive.size(); ++i) {
        if (buffers_.alive[i] == 0u) {
            continue;
        }

        const int cx = static_cast<int>(std::floor(buffers_.position_x[i] * inv_cell));
        const int cy = static_cast<int>(std::floor(buffers_.position_y[i] * inv_cell));
        const int cz = static_cast<int>(std::floor(buffers_.position_z[i] * inv_cell));
        NeighborGridEntry entry;
        entry.key = hashGridCell(cx, cy, cz);
        entry.x = cx;
        entry.y = cy;
        entry.z = cz;
        entry.index = i;
        neighbor_grid_.push_back(entry);
    }

    std::sort(neighbor_grid_.begin(), neighbor_grid_.end(),
        [](const NeighborGridEntry& a, const NeighborGridEntry& b) {
            if (a.key != b.key) return a.key < b.key;
            return a.index < b.index;
        });
}

void ParticleSimulationSystem::solveSelfCollisions(float dt) {
    if (!physics_settings_.self_collision_enabled || alive_count_ < 2 || dt <= 0.0f) {
        return;
    }

    const float radius = std::max(physics_settings_.particle_radius, 0.001f);
    const float min_distance = radius * 2.0f;
    const float min_distance_sq = min_distance * min_distance;
    const float cell_size = min_distance;
    const int iterations = std::clamp(physics_settings_.solver_iterations, 1, 16);
    const int max_neighbors = std::clamp(physics_settings_.max_neighbors_per_particle, 1, 256);
    const float stiffness = physics_settings_.mode == ParticlePhysicsMode::Fluid
        ? std::clamp(0.35f + physics_settings_.pressure_stiffness * 0.35f, 0.1f, 1.0f)
        : 1.0f;
    const float damping = std::clamp(physics_settings_.viscosity, 0.0f, 1.0f);
    const float cohesion = std::clamp(physics_settings_.cohesion, 0.0f, 1.0f);

    auto lowerForKey = [&](uint64_t key) {
        return std::lower_bound(neighbor_grid_.begin(), neighbor_grid_.end(), key,
            [](const NeighborGridEntry& entry, uint64_t value) {
                return entry.key < value;
            });
    };

    for (int iteration = 0; iteration < iterations; ++iteration) {
        buildNeighborGrid(cell_size);
        if (neighbor_grid_.empty()) {
            return;
        }

        for (const auto& entry : neighbor_grid_) {
            const std::size_t i = entry.index;
            if (i >= buffers_.alive.size() || buffers_.alive[i] == 0u) {
                continue;
            }

            Vec3 pi(buffers_.position_x[i], buffers_.position_y[i], buffers_.position_z[i]);
            Vec3 vi(buffers_.velocity_x[i], buffers_.velocity_y[i], buffers_.velocity_z[i]);
            const float inv_mass_i = buffers_.inverse_mass[i];
            int visited_neighbors = 0;

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (visited_neighbors >= max_neighbors) {
                            break;
                        }
                        const uint64_t key = hashGridCell(entry.x + dx, entry.y + dy, entry.z + dz);
                        auto it = lowerForKey(key);
                        for (; it != neighbor_grid_.end() && it->key == key; ++it) {
                            if (visited_neighbors >= max_neighbors) {
                                break;
                            }
                            const std::size_t j = it->index;
                            if (j <= i || j >= buffers_.alive.size() || buffers_.alive[j] == 0u) {
                                continue;
                            }
                            ++visited_neighbors;

                            Vec3 pj(buffers_.position_x[j], buffers_.position_y[j], buffers_.position_z[j]);
                            Vec3 delta = pj - pi;
                            float dist_sq = delta.dot(delta);
                            if (dist_sq >= min_distance_sq) {
                                if (cohesion > 0.0f && dist_sq < min_distance_sq * 6.25f) {
                                    const float dist = std::sqrt(std::max(dist_sq, 1e-8f));
                                    const Vec3 dir = delta * (1.0f / dist);
                                    const Vec3 pull = dir * (cohesion * radius * 0.0025f);
                                    pi = pi + pull;
                                    pj = pj - pull;
                                    buffers_.position_x[j] = pj.x;
                                    buffers_.position_y[j] = pj.y;
                                    buffers_.position_z[j] = pj.z;
                                }
                                continue;
                            }

                            float dist = std::sqrt(std::max(dist_sq, 1e-8f));
                            Vec3 normal = dist > 1e-5f
                                ? delta * (1.0f / dist)
                                : finiteDirectionOrUp(Vec3(
                                    hashUnitFloat(static_cast<uint32_t>(i * 17u + j * 31u)) - 0.5f,
                                    hashUnitFloat(static_cast<uint32_t>(i * 29u + j * 13u)) - 0.5f,
                                    hashUnitFloat(static_cast<uint32_t>(i * 7u + j * 43u)) - 0.5f));

                            const float inv_mass_j = buffers_.inverse_mass[j];
                            const float weight_sum = inv_mass_i + inv_mass_j;
                            const float wi = weight_sum > 1e-8f ? inv_mass_i / weight_sum : 0.5f;
                            const float wj = weight_sum > 1e-8f ? inv_mass_j / weight_sum : 0.5f;
                            const float correction_amount = (min_distance - dist) * stiffness;
                            const Vec3 correction = normal * correction_amount;
                            pi = pi - correction * wi;
                            pj = pj + correction * wj;

                            Vec3 vj(buffers_.velocity_x[j], buffers_.velocity_y[j], buffers_.velocity_z[j]);
                            const float rel_normal_velocity = (vj - vi).dot(normal);
                            if (rel_normal_velocity < 0.0f) {
                                const float impulse_strength = -(1.0f - damping * 0.5f) * rel_normal_velocity;
                                vi = vi - normal * (impulse_strength * wi);
                                vj = vj + normal * (impulse_strength * wj);
                                buffers_.velocity_x[j] = vj.x;
                                buffers_.velocity_y[j] = vj.y;
                                buffers_.velocity_z[j] = vj.z;
                            }

                            buffers_.position_x[j] = pj.x;
                            buffers_.position_y[j] = pj.y;
                            buffers_.position_z[j] = pj.z;
                        }
                    }
                }
            }

            applyColliders(pi, vi);
            buffers_.position_x[i] = pi.x;
            buffers_.position_y[i] = pi.y;
            buffers_.position_z[i] = pi.z;
            buffers_.velocity_x[i] = vi.x;
            buffers_.velocity_y[i] = vi.y;
            buffers_.velocity_z[i] = vi.z;
        }
    }
}

bool ParticleSimulationSystem::hasActiveEmitters() const {
    for (const auto& emitter : emitters_) {
        if (emitter.enabled && (emitter.rate_per_second > 0.0f || emitter.burst_count > 0)) {
            return true;
        }
    }
    return false;
}

void ParticleSimulationSystem::emitFromEmitters(const SimulationContext& context) {
    if (emitters_.empty()) {
        return;
    }

    for (auto& emitter : emitters_) {
        if (!emitter.enabled) {
            continue;
        }

        Vec3 source_position = emitter.point + emitter.local_offset;
        Vec3 source_direction = emitter.direction;
        if (emitter_source_resolver_) {
            Vec3 resolved_position = source_position;
            Vec3 resolved_direction = source_direction;
            if (emitter_source_resolver_(emitter, resolved_position, resolved_direction)) {
                source_position = resolved_position;
                source_direction = resolved_direction;
            }
        }

        emitter.accumulator += std::max(0.0f, emitter.rate_per_second) * context.dt;
        int spawn_count = static_cast<int>(std::floor(emitter.accumulator));
        if (spawn_count > 0) {
            emitter.accumulator -= static_cast<float>(spawn_count);
        }
        if (emitter.burst_count > 0) {
            spawn_count += emitter.burst_count;
            emitter.burst_count = 0;
        }
        spawn_count = std::clamp(spawn_count, 0, 512);

        for (int i = 0; i < spawn_count; ++i) {
            const uint32_t serial = emitter_spawn_serial_++;
            Vec3 spawn_position = source_position;
            Vec3 spawn_direction = source_direction;

            if (emitter.spawn_mode == ParticleEmitterSpawnMode::ObjectAABBSurface && emitter_bounds_resolver_) {
                Vec3 min_bound;
                Vec3 max_bound;
                Vec3 surface_position;
                Vec3 surface_normal;
                if (emitter_bounds_resolver_(emitter, min_bound, max_bound) &&
                    sampleAABBSurface(min_bound, max_bound, serial ^ emitter.seed, surface_position, surface_normal)) {
                    spawn_position = surface_position + surface_normal * std::max(0.0f, emitter.surface_offset);
                    spawn_direction = surface_normal;
                }
            } else if (emitter.spawn_mode == ParticleEmitterSpawnMode::MeshSurface && emitter_surface_sampler_) {
                ParticleSurfaceSample sample;
                if (emitter_surface_sampler_(emitter, serial ^ emitter.seed, sample)) {
                    const Vec3 normal = finiteDirectionOrUp(sample.normal);
                    spawn_position = sample.position + normal * std::max(0.0f, emitter.surface_offset);
                    spawn_direction = normal;
                }
            } else {
                const float angle = 6.28318530718f * hashUnitFloat(serial ^ emitter.seed ^ 0x51ed270bu);
                const float radius = 0.04f * std::sqrt(hashUnitFloat(serial + emitter.seed * 33u));
                const Vec3 spawn_offset(std::cos(angle) * radius, 0.0f, std::sin(angle) * radius);
                spawn_position = source_position + spawn_offset;
            }

            ParticleSpawnDesc desc;
            desc.position = spawn_position;
            desc.velocity = emitterVelocity(emitter, spawn_direction, serial);
            desc.lifetime_seconds = emitter.lifetime_seconds;
            desc.mass = emitter.mass;

            // Visual attributes (with per-particle jitter where configured).
            const float size_rand = 1.0f + emitter.size_jitter *
                (hashUnitFloat((serial * 9781u + emitter.seed) ^ 0x2c1b3c6du) * 2.0f - 1.0f);
            desc.start_size = std::max(0.0f, emitter.start_size * size_rand);
            desc.end_size = std::max(0.0f, emitter.end_size * size_rand);
            desc.start_opacity = emitter.start_opacity;
            desc.end_opacity = emitter.end_opacity;
            desc.start_color = emitter.start_color;
            desc.end_color = emitter.end_color;
            desc.rotation = 6.28318530718f * hashUnitFloat(serial ^ (emitter.seed * 2654435761u) ^ 0x7a5b9c1fu);
            desc.angular_velocity = emitter.angular_velocity + emitter.angular_jitter *
                (hashUnitFloat((serial * 40503u + emitter.seed) ^ 0x68bc21f3u) * 2.0f - 1.0f);

            spawn(desc);
        }
    }
}

void ParticleSimulationSystem::resizeStorage(std::size_t capacity) {
    buffers_.position_x.resize(capacity, 0.0f);
    buffers_.position_y.resize(capacity, 0.0f);
    buffers_.position_z.resize(capacity, 0.0f);
    buffers_.velocity_x.resize(capacity, 0.0f);
    buffers_.velocity_y.resize(capacity, 0.0f);
    buffers_.velocity_z.resize(capacity, 0.0f);
    buffers_.age_seconds.resize(capacity, 0.0f);
    buffers_.lifetime_seconds.resize(capacity, 0.0f);
    buffers_.inverse_mass.resize(capacity, 0.0f);
    buffers_.alive.resize(capacity, 0u);

    buffers_.size.resize(capacity, 0.0f);
    buffers_.rotation.resize(capacity, 0.0f);
    buffers_.angular_velocity.resize(capacity, 0.0f);
    buffers_.color_r.resize(capacity, 1.0f);
    buffers_.color_g.resize(capacity, 1.0f);
    buffers_.color_b.resize(capacity, 1.0f);
    buffers_.opacity.resize(capacity, 0.0f);
    buffers_.start_size.resize(capacity, 0.0f);
    buffers_.end_size.resize(capacity, 0.0f);
    buffers_.start_opacity.resize(capacity, 0.0f);
    buffers_.end_opacity.resize(capacity, 0.0f);
    buffers_.start_color_r.resize(capacity, 1.0f);
    buffers_.start_color_g.resize(capacity, 1.0f);
    buffers_.start_color_b.resize(capacity, 1.0f);
    buffers_.end_color_r.resize(capacity, 1.0f);
    buffers_.end_color_g.resize(capacity, 1.0f);
    buffers_.end_color_b.resize(capacity, 1.0f);
    ++data_version_;
}

std::size_t ParticleSimulationSystem::findDeadSlot() const {
    for (std::size_t i = 0; i < buffers_.alive.size(); ++i) {
        if (buffers_.alive[i] == 0u) {
            return i;
        }
    }
    return kInvalidParticle;
}

void ParticleSimulationSystem::uploadToCompute(const SimulationContext& context) {
    if (!context.compute || buffers_.alive.empty() || compute_buffers_.source_version == data_version_) {
        return;
    }

    SimulationComputeContext& compute = *context.compute;
    const std::size_t float_bytes = buffers_.alive.size() * sizeof(float);
    const std::size_t alive_bytes = buffers_.alive.size() * sizeof(uint8_t);
    const ComputeBufferUsage float_usage = ComputeBufferUsage::Storage |
                                           ComputeBufferUsage::Upload |
                                           ComputeBufferUsage::ReadWrite;
    const ComputeBufferUsage alive_usage = ComputeBufferUsage::Storage |
                                           ComputeBufferUsage::Upload |
                                           ComputeBufferUsage::ReadOnly;

    ensureComputeBuffer(compute, compute_buffers_.position_x, "ParticlePositionX", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.position_y, "ParticlePositionY", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.position_z, "ParticlePositionZ", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.velocity_x, "ParticleVelocityX", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.velocity_y, "ParticleVelocityY", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.velocity_z, "ParticleVelocityZ", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.age_seconds, "ParticleAge", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.lifetime_seconds, "ParticleLifetime", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.inverse_mass, "ParticleInverseMass", float_bytes, float_usage);
    ensureComputeBuffer(compute, compute_buffers_.alive, "ParticleAlive", alive_bytes, alive_usage);

    compute.uploadBuffer(compute_buffers_.position_x, buffers_.position_x.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.position_y, buffers_.position_y.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.position_z, buffers_.position_z.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.velocity_x, buffers_.velocity_x.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.velocity_y, buffers_.velocity_y.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.velocity_z, buffers_.velocity_z.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.age_seconds, buffers_.age_seconds.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.lifetime_seconds, buffers_.lifetime_seconds.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.inverse_mass, buffers_.inverse_mass.data(), float_bytes);
    compute.uploadBuffer(compute_buffers_.alive, buffers_.alive.data(), alive_bytes);

    compute_buffers_.capacity = buffers_.alive.size();
    compute_buffers_.source_version = data_version_;
}

void ParticleSimulationSystem::ensureComputeBuffer(SimulationComputeContext& compute,
                                                   ComputeBufferHandle& handle,
                                                   const char* name,
                                                   std::size_t size_bytes,
                                                   ComputeBufferUsage usage) {
    if (size_bytes == 0) {
        return;
    }

    if (!handle.valid() || handle.backend != compute.backendType()) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
        }
        ComputeBufferDesc desc;
        desc.debug_name = name ? name : "ParticleBuffer";
        desc.size_bytes = size_bytes;
        desc.usage = usage;
        handle = compute.createBuffer(desc);
        return;
    }

    if (compute.getBufferSize(handle) != size_bytes) {
        if (!compute.resizeBuffer(handle, size_bytes)) {
            compute.destroyBuffer(handle);
            ComputeBufferDesc desc;
            desc.debug_name = name ? name : "ParticleBuffer";
            desc.size_bytes = size_bytes;
            desc.usage = usage;
            handle = compute.createBuffer(desc);
        }
    }
}

void ParticleSimulationSystem::releaseGridDomainComputeBuffers(SimulationComputeContext& compute,
                                                               SimulationGridDomainComputeBuffers& buffers) {
    compute.synchronize();

    auto destroy = [&](ComputeBufferHandle& handle) {
        if (handle.valid()) {
            compute.destroyBuffer(handle);
            handle = {};
        }
    };

    destroy(buffers.vel_x);
    destroy(buffers.vel_y);
    destroy(buffers.vel_z);
    destroy(buffers.density);
    destroy(buffers.temperature);
    destroy(buffers.fuel);
    destroy(buffers.pressure);
    destroy(buffers.divergence);
    destroy(buffers.scratch_vel_x);
    destroy(buffers.scratch_vel_y);
    destroy(buffers.scratch_vel_z);
    destroy(buffers.scratch_scalar);
    destroy(buffers.fluid_mask);
    destroy(buffers.cg_residual);
    destroy(buffers.cg_z);
    destroy(buffers.cg_search);
    destroy(buffers.cg_As);
    destroy(buffers.cg_diag);
    destroy(buffers.cg_partials);
    destroy(buffers.var_u_weight);
    destroy(buffers.var_v_weight);
    destroy(buffers.var_w_weight);
    destroy(buffers.var_svx);
    destroy(buffers.var_svy);
    destroy(buffers.var_svz);
    destroy(buffers.var_fluid_phi);
    for (auto& level : buffers.mg_levels) {
        destroy(level.mask);
        destroy(level.rhs);
        destroy(level.z);
        destroy(level.diag);
    }
    buffers.mg_levels.clear();
    destroy(buffers.fluid_positions);
    destroy(buffers.fluid_velocities);
    destroy(buffers.fluid_affine);
    destroy(buffers.foam_positions);
    destroy(buffers.foam_render.spheres);
    buffers.foam_render = {};

    buffers.resolution_x = 0;
    buffers.resolution_y = 0;
    buffers.resolution_z = 0;
    buffers.fluid_particle_capacity = 0;
    buffers.backend = ComputeBackendType::CPU;
}

bool ParticleSimulationSystem::ensureGridDomainComputeBuffers(SimulationComputeContext& compute,
                                                              SimulationGridDomainComputeBuffers& buffers,
                                                              const FluidSim::FluidGrid& grid) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        return false;
    }

    // Detect stale handles from a replaced backend of the same type (e.g.
    // CPU→GPU→CPU→GPU: CUDA backend is destroyed and recreated, but handle IDs
    // belong to the old instance and are unknown to the new one).
    // getBufferSize returns 0 for IDs not in the current backend's table.
    const bool stale_handles = buffers.vel_x.valid() &&
        compute.getBufferSize(buffers.vel_x) == 0;

    const bool needs_rebuild =
        stale_handles ||
        buffers.backend != compute.backendType() ||
        buffers.resolution_x != grid.nx ||
        buffers.resolution_y != grid.ny ||
        buffers.resolution_z != grid.nz;
    if (needs_rebuild) {
        // Handles from a destroyed backend must not be passed to destroyBuffer
        // on the new backend — just zero them out directly.
        if (stale_handles) {
            buffers = SimulationGridDomainComputeBuffers{};
        } else {
            releaseGridDomainComputeBuffers(compute, buffers);
        }
    }

    const std::size_t cell_count = grid.getCellCount();
    const std::size_t cell_bytes = cell_count * sizeof(float);
    if (grid.vel_x.empty() || grid.vel_y.empty() || grid.vel_z.empty() || cell_bytes == 0) {
        return false;
    }

    const ComputeBufferUsage usage = ComputeBufferUsage::Storage |
                                     ComputeBufferUsage::Upload |
                                     ComputeBufferUsage::Download |
                                     ComputeBufferUsage::ReadWrite;
    // The fluid APIC P2G scatter (runGpuFluidP2G) reuses temperature / fuel /
    // scratch_scalar as the per-component WEIGHT field (shader binding 4). Those
    // weight indices are FACE-centred — component 0 reaches (nx+1)*ny*nz,
    // component 1 nx*(ny+1)*nz, component 2 nx*ny*(nz+1) — all larger than the
    // cell count. Sizing them at cell_bytes overflowed the weight buffer
    // (atomicAdd / clear / normalize past the end), corrupting the velocity
    // field → fluid that never falls and just disperses. Size the three weight
    // aliases to the max face count so every component's scatter stays in bounds.
    // (For gas domains these stay cell-channels; the larger allocation is
    // harmless — gas code only ever touches the first cell_count entries.)
    const std::size_t max_face_count =
        std::max({ grid.vel_x.size(), grid.vel_y.size(), grid.vel_z.size() });
    const std::size_t weight_bytes = max_face_count * sizeof(float);
    ensureComputeBuffer(compute, buffers.vel_x, "GridDomainVelX", grid.vel_x.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.vel_y, "GridDomainVelY", grid.vel_y.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.vel_z, "GridDomainVelZ", grid.vel_z.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.density, "GridDomainDensity", cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.temperature, "GridDomainTemperature", weight_bytes, usage);
    ensureComputeBuffer(compute, buffers.fuel, "GridDomainFuel", weight_bytes, usage);
    ensureComputeBuffer(compute, buffers.pressure, "GridDomainPressure", cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.divergence, "GridDomainDivergence", cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.scratch_vel_x, "GridDomainScratchVelX", grid.vel_x.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.scratch_vel_y, "GridDomainScratchVelY", grid.vel_y.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.scratch_vel_z, "GridDomainScratchVelZ", grid.vel_z.size() * sizeof(float), usage);
    ensureComputeBuffer(compute, buffers.scratch_scalar, "GridDomainScratchScalar", weight_bytes, usage);
    ensureComputeBuffer(compute, buffers.fluid_mask,    "GridDomainFluidMask",    cell_bytes, usage);

    if (compute.backendType() == ComputeBackendType::CUDA) {
        ensureComputeBuffer(compute, buffers.var_u_weight, "GridDomainVarUWeight", grid.vel_x.size() * sizeof(float), usage);
        ensureComputeBuffer(compute, buffers.var_v_weight, "GridDomainVarVWeight", grid.vel_y.size() * sizeof(float), usage);
        ensureComputeBuffer(compute, buffers.var_w_weight, "GridDomainVarWWeight", grid.vel_z.size() * sizeof(float), usage);
        ensureComputeBuffer(compute, buffers.var_svx,      "GridDomainVarSvx",      cell_bytes, usage);
        ensureComputeBuffer(compute, buffers.var_svy,      "GridDomainVarSvy",      cell_bytes, usage);
        ensureComputeBuffer(compute, buffers.var_svz,      "GridDomainVarSvz",      cell_bytes, usage);
        ensureComputeBuffer(compute, buffers.var_fluid_phi,"GridDomainVarFluidPhi", cell_bytes, usage);
    }

    // GPU MGPCG (Layer A) scratch — cell-sized float vectors + per-block double
    // partial sums for the dot reductions. Additive: if any of these fail to
    // allocate the SOR path is unaffected (runGpuFluidMGPCGPressure re-validates
    // and the caller falls back to SOR / CPU PCG).
    ensureComputeBuffer(compute, buffers.cg_residual, "GridDomainCGResidual", cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.cg_z,        "GridDomainCGZ",        cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.cg_search,   "GridDomainCGSearch",   cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.cg_As,       "GridDomainCGAs",       cell_bytes, usage);
    ensureComputeBuffer(compute, buffers.cg_diag,     "GridDomainCGDiag",     cell_bytes, usage);
    const std::size_t cg_blocks = (cell_count + 255u) / 256u;
    ensureComputeBuffer(compute, buffers.cg_partials, "GridDomainCGPartials", cg_blocks * sizeof(double), usage);

    if (compute.backendType() == ComputeBackendType::CUDA) {
        std::vector<SimulationGridDomainMGLevelBuffers> wanted_levels;
        int lx = (grid.nx + 1) / 2;
        int ly = (grid.ny + 1) / 2;
        int lz = (grid.nz + 1) / 2;
        constexpr int kMaxMgLevels = 2; // Layer B bootstrap: two-level V-cycle, expand after profiling.
        for (int level = 0; level < kMaxMgLevels && lx >= 4 && ly >= 4 && lz >= 4; ++level) {
            SimulationGridDomainMGLevelBuffers desc;
            desc.nx = lx;
            desc.ny = ly;
            desc.nz = lz;
            wanted_levels.push_back(desc);
            if (lx <= 8 || ly <= 8 || lz <= 8) {
                break;
            }
            lx = (lx + 1) / 2;
            ly = (ly + 1) / 2;
            lz = (lz + 1) / 2;
        }

        if (buffers.mg_levels.size() != wanted_levels.size()) {
            for (auto& level : buffers.mg_levels) {
                if (level.mask.valid()) compute.destroyBuffer(level.mask);
                if (level.rhs.valid()) compute.destroyBuffer(level.rhs);
                if (level.z.valid()) compute.destroyBuffer(level.z);
                if (level.diag.valid()) compute.destroyBuffer(level.diag);
            }
            buffers.mg_levels = wanted_levels;
        }

        for (std::size_t li = 0; li < buffers.mg_levels.size(); ++li) {
            auto& level = buffers.mg_levels[li];
            level.nx = wanted_levels[li].nx;
            level.ny = wanted_levels[li].ny;
            level.nz = wanted_levels[li].nz;
            const std::size_t level_cells = static_cast<std::size_t>(level.nx) *
                                            static_cast<std::size_t>(level.ny) *
                                            static_cast<std::size_t>(level.nz);
            const std::size_t level_bytes = level_cells * sizeof(float);
            char name[96];
            std::snprintf(name, sizeof(name), "GridDomainMGMask%zu", li);
            ensureComputeBuffer(compute, level.mask, name, level_bytes, usage);
            std::snprintf(name, sizeof(name), "GridDomainMGRhs%zu", li);
            ensureComputeBuffer(compute, level.rhs, name, level_bytes, usage);
            std::snprintf(name, sizeof(name), "GridDomainMGZ%zu", li);
            ensureComputeBuffer(compute, level.z, name, level_bytes, usage);
            std::snprintf(name, sizeof(name), "GridDomainMGDiag%zu", li);
            ensureComputeBuffer(compute, level.diag, name, level_bytes, usage);
        }
    } else if (!buffers.mg_levels.empty()) {
        for (auto& level : buffers.mg_levels) {
            if (level.mask.valid()) compute.destroyBuffer(level.mask);
            if (level.rhs.valid()) compute.destroyBuffer(level.rhs);
            if (level.z.valid()) compute.destroyBuffer(level.z);
            if (level.diag.valid()) compute.destroyBuffer(level.diag);
        }
        buffers.mg_levels.clear();
    }

    bool ok =
        buffers.vel_x.valid() &&
        buffers.vel_y.valid() &&
        buffers.vel_z.valid() &&
        buffers.density.valid() &&
        buffers.temperature.valid() &&
        buffers.fuel.valid() &&
        buffers.pressure.valid() &&
        buffers.divergence.valid() &&
        buffers.scratch_vel_x.valid() &&
        buffers.scratch_vel_y.valid() &&
        buffers.scratch_vel_z.valid() &&
        buffers.scratch_scalar.valid() &&
        buffers.fluid_mask.valid();
    if (compute.backendType() == ComputeBackendType::CUDA) {
        ok = ok &&
             buffers.var_u_weight.valid() &&
             buffers.var_v_weight.valid() &&
             buffers.var_w_weight.valid() &&
             buffers.var_svx.valid() &&
             buffers.var_svy.valid() &&
             buffers.var_svz.valid();
    }
    if (!ok) {
        return false;
    }

    buffers.resolution_x = grid.nx;
    buffers.resolution_y = grid.ny;
    buffers.resolution_z = grid.nz;
    buffers.backend = compute.backendType();
    return true;
}

bool ParticleSimulationSystem::validateGpuFluidMGPCG(SimulationComputeContext* compute) {
    if (!compute || !compute->supportsDispatch()) {
        SCENE_LOG_WARN("[MGPCG SelfTest] No GPU compute dispatch available; skipped.");
        return false;
    }

    // Synthetic free-surface problem: a fluid block suspended in air inside a
    // closed domain, with an imposed smooth (divergent) staggered velocity.
    const int   N  = 16;
    const float h  = 0.1f;
    const float dt = 1.0f / 60.0f;

    SimulationGridDomainState st;
    st.type = SimulationDomainType::Fluid;
    st.grid.resize(N, N, N, h, Vec3(0.0f, 0.0f, 0.0f));
    auto& g = st.grid;

    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i <= N; ++i)
        g.vel_x[g.velXIndex(i, j, k)] = std::sin(0.4f * i + 0.2f * j) * 0.5f;
    for (int k = 0; k < N; ++k) for (int j = 0; j <= N; ++j) for (int i = 0; i < N; ++i)
        g.vel_y[g.velYIndex(i, j, k)] = std::cos(0.3f * j + 0.1f * k) * 0.5f;
    for (int k = 0; k <= N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
        g.vel_z[g.velZIndex(i, j, k)] = std::sin(0.25f * k - 0.15f * i) * 0.5f;

    const std::size_t cells = g.getCellCount();
    auto isFluidCell = [&](int i, int j, int k) {
        return i >= 4 && i < 12 && j >= 4 && j < 12 && k >= 4 && k < 12;
    };
    std::vector<float> mask(cells, 0.0f);
    int fluid_count = 0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i)
        if (isFluidCell(i, j, k)) { mask[g.cellIndex(i, j, k)] = 1.0f; ++fluid_count; }

    // Snapshot the PRE-projection velocity field. runGpuFluidMGPCGPressure
    // subtracts the pressure gradient and downloads the (now divergence-free)
    // velocities back into grid.vel_*, so the host reference RHS must be built
    // from this snapshot — not from grid.vel_* after the call.
    const std::vector<float> vx0 = g.vel_x;
    const std::vector<float> vy0 = g.vel_y;
    const std::vector<float> vz0 = g.vel_z;

    // ── GPU solve ──
    SimulationGridDomainComputeBuffers buffers;
    if (!ensureGridDomainComputeBuffers(*compute, buffers, g)) {
        SCENE_LOG_WARN("[MGPCG SelfTest] GPU buffer allocation failed; skipped.");
        return false;
    }
    Fluid::APICSolverParams params;
    params.free_surface = true;
    params.pressure_iterations = 400; // generous cap; CG converges well before this
    params.density_correction = 0.0f; // isolate the linear solve (no Bridson term)
    bool ok = runGpuFluidMGPCGPressure(st, params, dt, compute, buffers, mask);
    std::vector<float> p_gpu(cells, 0.0f);
    std::vector<float> div_gpu(cells, 0.0f), diag_gpu(cells, 0.0f), r_gpu(cells, 0.0f);
    ok = ok && compute->downloadBuffer(buffers.pressure,    p_gpu.data(),    cells * sizeof(float));
    // Diagnostic: snapshot intermediate GPU state to localise any divergence.
    compute->downloadBuffer(buffers.divergence,   div_gpu.data(),  cells * sizeof(float));
    compute->downloadBuffer(buffers.cg_diag,      diag_gpu.data(), cells * sizeof(float));
    compute->downloadBuffer(buffers.cg_residual,  r_gpu.data(),    cells * sizeof(float));
    releaseGridDomainComputeBuffers(*compute, buffers);
    if (!ok) {
        SCENE_LOG_WARN("[MGPCG SelfTest] GPU solve / pressure download failed.");
        return false;
    }

    // ── Host reference: identical matrix-free operator + plain double CG ──
    // (no interior solids ⇒ this operator is bit-for-bit the GPU's: diag =
    //  #in-bounds neighbours, off-diag -1 per in-bounds fluid neighbour,
    //  b = -div*h*h/dt with div = (du+dv+dw)/h.)
    std::vector<uint8_t> fluid(cells, 0u);
    for (std::size_t c = 0; c < cells; ++c) fluid[c] = mask[c] > 0.5f ? 1u : 0u;
    auto idx = [&](int i, int j, int k) {
        return static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * N +
               static_cast<std::size_t>(k) * N * N;
    };
    auto inb  = [&](int i, int j, int k) { return i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N; };
    auto diagOf = [&](int i, int j, int k) {
        int d = 0;
        if (i - 1 >= 0) ++d; if (i + 1 < N) ++d;
        if (j - 1 >= 0) ++d; if (j + 1 < N) ++d;
        if (k - 1 >= 0) ++d; if (k + 1 < N) ++d;
        return d;
    };

    const double invdt = 1.0 / dt;
    std::vector<double> b(cells, 0.0);
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        if (!fluid[idx(i, j, k)]) continue;
        const double du = vx0[g.velXIndex(i + 1, j, k)] - vx0[g.velXIndex(i, j, k)];
        const double dv = vy0[g.velYIndex(i, j + 1, k)] - vy0[g.velYIndex(i, j, k)];
        const double dw = vz0[g.velZIndex(i, j, k + 1)] - vz0[g.velZIndex(i, j, k)];
        const double div = (du + dv + dw) / static_cast<double>(h);
        b[idx(i, j, k)] = -div * h * h * invdt;
    }
    auto applyA = [&](const std::vector<double>& x, std::vector<double>& Ax) {
        for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
            const std::size_t c = idx(i, j, k);
            if (!fluid[c]) { Ax[c] = 0.0; continue; }
            double s = static_cast<double>(diagOf(i, j, k)) * x[c];
            if (inb(i - 1, j, k) && fluid[idx(i - 1, j, k)]) s -= x[idx(i - 1, j, k)];
            if (inb(i + 1, j, k) && fluid[idx(i + 1, j, k)]) s -= x[idx(i + 1, j, k)];
            if (inb(i, j - 1, k) && fluid[idx(i, j - 1, k)]) s -= x[idx(i, j - 1, k)];
            if (inb(i, j + 1, k) && fluid[idx(i, j + 1, k)]) s -= x[idx(i, j + 1, k)];
            if (inb(i, j, k - 1) && fluid[idx(i, j, k - 1)]) s -= x[idx(i, j, k - 1)];
            if (inb(i, j, k + 1) && fluid[idx(i, j, k + 1)]) s -= x[idx(i, j, k + 1)];
            Ax[c] = s;
        }
    };
    auto dot = [&](const std::vector<double>& a, const std::vector<double>& bb) {
        double s = 0.0; for (std::size_t c = 0; c < cells; ++c) s += a[c] * bb[c]; return s;
    };

    std::vector<double> p(cells, 0.0), r = b, s = b, As(cells, 0.0);
    double sig = dot(r, r); const double sig0 = sig;
    for (int it = 0; it < 2000 && sig > 1e-24 * sig0; ++it) {
        applyA(s, As);
        const double sAs = dot(s, As);
        if (std::abs(sAs) < 1e-300) break;
        const double a = sig / sAs;
        for (std::size_t c = 0; c < cells; ++c) { p[c] += a * s[c]; r[c] -= a * As[c]; }
        const double sig_new = dot(r, r);
        const double beta = sig_new / sig;
        for (std::size_t c = 0; c < cells; ++c) s[c] = r[c] + beta * s[c];
        sig = sig_new;
    }

    // ── Compare ──
    std::vector<double> pg(cells, 0.0), Apg(cells, 0.0), Ap(cells, 0.0);
    for (std::size_t c = 0; c < cells; ++c) pg[c] = static_cast<double>(p_gpu[c]);
    applyA(pg, Apg);
    applyA(p, Ap);
    double rb = 0.0, bb = 0.0, cpu_r2 = 0.0, diff2 = 0.0, maxdiff = 0.0;
    for (std::size_t c = 0; c < cells; ++c) {
        if (!fluid[c]) continue;
        const double rg = b[c] - Apg[c];
        const double rc = b[c] - Ap[c];
        const double d  = pg[c] - p[c];
        rb += rg * rg; bb += b[c] * b[c]; cpu_r2 += rc * rc;
        diff2 += d * d; maxdiff = std::max(maxdiff, std::abs(d));
    }
    const double rel_gpu = bb > 0.0 ? std::sqrt(rb / bb) : 0.0;
    const double rel_cpu = bb > 0.0 ? std::sqrt(cpu_r2 / bb) : 0.0;
    const double l2diff  = std::sqrt(diff2);

    // ── Diagnostics: localise where the GPU path diverges from the host. ──
    // 1. GPU divergence vs CPU div  → tests vel upload + grid_divergence kernel.
    // 2. GPU diag vs CPU diagOf      → tests fluid_cg_build_diag.
    // 3. final GPU residual norm     → did the GPU CG converge at all?
    double div_maxabs = 0.0, diag_maxabs = 0.0, rfin2 = 0.0;
    for (int k = 0; k < N; ++k) for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        const std::size_t c = idx(i, j, k);
        if (!fluid[c]) continue;
        const double du = vx0[g.velXIndex(i + 1, j, k)] - vx0[g.velXIndex(i, j, k)];
        const double dv = vy0[g.velYIndex(i, j + 1, k)] - vy0[g.velYIndex(i, j, k)];
        const double dw = vz0[g.velZIndex(i, j, k + 1)] - vz0[g.velZIndex(i, j, k)];
        const double div_cpu = (du + dv + dw) / static_cast<double>(h);
        div_maxabs  = std::max(div_maxabs,  std::abs(static_cast<double>(div_gpu[c])  - div_cpu));
        diag_maxabs = std::max(diag_maxabs, std::abs(static_cast<double>(diag_gpu[c]) - static_cast<double>(diagOf(i, j, k))));
        rfin2 += static_cast<double>(r_gpu[c]) * static_cast<double>(r_gpu[c]);
    }
    const double rfin = std::sqrt(rfin2);

    char dbg[256];
    std::snprintf(dbg, sizeof(dbg),
        "[MGPCG SelfTest] DIAG | div_maxdiff=%.3e | diag_maxdiff=%.3e | ||r_final||=%.3e (||b||=%.3e)",
        div_maxabs, diag_maxabs, rfin, std::sqrt(bb));
    SCENE_LOG_INFO(std::string(dbg));

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "[MGPCG SelfTest] N=%d fluid=%d | GPU rel.res=%.3e | CPU rel.res=%.3e | "
        "||p_gpu-p_cpu||2=%.3e maxabs=%.3e",
        N, fluid_count, rel_gpu, rel_cpu, l2diff, maxdiff);
    SCENE_LOG_INFO(std::string(buf));
    const bool pass = rel_gpu < 1e-3 && l2diff < 1e-2;
    SCENE_LOG_INFO(pass ? std::string("[MGPCG SelfTest] PASS — GPU CG matches CPU reference.")
                        : std::string("[MGPCG SelfTest] FAIL — residual/diff above tolerance."));
    return pass;
}

} // namespace RayTrophiSim
