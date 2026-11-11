// sim_matter_p2g_fused.glsl -- the MPM lane's particle-to-grid transfer in
// three passes instead of 3 x (clear, clear, scatter, copy, stress, normalize,
// normalize) = 21 dispatches per substep (2026-11-11). Snow + water: the MPM
// lane's P2G (sim_matter_p2g x3) + stress P2G (x3) were 190 of a frame's
// ~390 GPU ms, every particle read six times per substep.
//
//   pass 0, one lane per face slot: clear momentum, weight and the stress
//     accumulator of all three components;
//   pass 1, one lane per parcel of this lane (matter index list): for each
//     component the APIC momentum w m (v + C dx) and weight w m (the scatter
//     of sim_fluid_p2g_scatter MATTER_INDEXED), and the stress impulse
//     -dt V_dry sigma . grad w (sim_fluid_granular_stress_p2g MATTER_INDEXED,
//     same small-node cut). With FLIP the stress goes to its own accumulator
//     (the FLIP baseline buffer), else straight into the momentum;
//   pass 2, one lane per face slot: v = (momentum + stress) / w and, with
//     FLIP, the baseline momentum / w (the transferred velocity WITHOUT the
//     stress impulse, as the old copy-between-scatter-and-stress gave it).
// Same atomics per node as before (momentum, weight, stress); no face mass
// gradient (its only reader was removed with sim_matter_contact).
#extension GL_EXT_shader_atomic_float : require
#ifdef P2G_CLUSTERED
// Same-address contention (sim_fluid_p2g_scatter.glsl P2G_CLUSTERED): a
// cluster of eight lanes sharing a component's base node sums its
// contributions first and one lane issues each atomic. Dense resting snow is
// where the per-lane atomics serialise; the fused kernel shipped without it
// and gained only ~10 % over the per-component chain (2026-11-11, 62k snow).
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#endif
#include "sim_dispatch.glsl"
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int nx; int ny; int nz;
    int particle_count;
    int pass;
    uint flags;            // bit 0: FLIP baseline (stress kept apart)
    float origin_x; float origin_y; float origin_z;
    float voxel_size;
    float dt;
    uint matter_lane;
#ifdef P2G_BLOCK
    int bx; int by; int bz;   // 4^3-cell blocks per axis
    int block_pad;
#endif
} pc;

layout(set = 0, binding = 0) readonly buffer Positions  { float pos_data[]; };
layout(set = 0, binding = 1) readonly buffer Velocities { float vel_data[]; };
layout(set = 0, binding = 2) readonly buffer Affine     { float affine[]; };
layout(set = 0, binding = 3) buffer MX { float mom_x[]; };
layout(set = 0, binding = 4) buffer MY { float mom_y[]; };
layout(set = 0, binding = 5) buffer MZ { float mom_z[]; };
layout(set = 0, binding = 6) buffer WX { float wt_x[]; };
layout(set = 0, binding = 7) buffer WY { float wt_y[]; };
layout(set = 0, binding = 8) buffer WZ { float wt_z[]; };
layout(set = 0, binding = 9) buffer SX { float st_x[]; };   // stress accumulator -> FLIP baseline
layout(set = 0, binding = 10) buffer SY { float st_y[]; };
layout(set = 0, binding = 11) buffer SZ { float st_z[]; };
layout(set = 0, binding = 12) readonly buffer StressDiag { float sd[]; };
layout(set = 0, binding = 13) readonly buffer StressShear { float ss[]; };
layout(set = 0, binding = 14) readonly buffer MatterIndices { uint matter_index[]; };
layout(set = 0, binding = 15) readonly buffer MatterCounts { uint matter_count[]; };
layout(set = 0, binding = 16) readonly buffer MatterRestMass { float matter_rest[]; };
layout(set = 0, binding = 17) readonly buffer MatterFraction { float matter_fraction[]; };
layout(set = 0, binding = 18) readonly buffer MatterDryVolume { float matter_dry_volume[]; };
#ifdef P2G_BLOCK
// Shared-memory block P2G (docs/dev/OLCEK_MILYON_PARCACIK.md S5, Gao et al.
// 2018). The clustered path above needs eight lanes on one base node, which
// the staggered MAC stencil and one-parcel-per-octant seeding almost never
// give: at 1M snow parcels P2G was 74 % of the frame's GPU time on global
// atomics (~243 per parcel per substep). Here one workgroup takes the parcels
// of one 4^3-cell block (sim_matter_p2g_block_sort.comp permutation) and
// accumulates them in a shared tile; the tile goes to global memory once.
// Tile = block +-2 nodes per axis (stencil reach + one cell of drift since
// the sort); a contribution outside it falls back to the global atomic, so
// the result never depends on how far a parcel moved.
layout(set = 0, binding = 19) readonly buffer BlockMeta { uint block_meta[]; };
layout(set = 0, binding = 20) readonly buffer BlockItems { uint block_items[]; };
const int kBlock = 4;
const int kTile = 9;
const int kTileNodes = kTile * kTile * kTile;
shared float tile[3 * 3 * kTileNodes];   // [field: momentum, weight, stress][component][node]
ivec3 g_tile_lo;
#endif

const float kMinNodeWeight = 1.0e-4;  // sim_fluid_granular_stress_p2g.glsl

int faceCount(int c) {
    return c == 0 ? (pc.nx + 1) * pc.ny * pc.nz
         : c == 1 ? pc.nx * (pc.ny + 1) * pc.nz
                  : pc.nx * pc.ny * (pc.nz + 1);
}
int faceIndex(int c, int x, int y, int z) {
    return c == 0 ? x + y * (pc.nx + 1) + z * (pc.nx + 1) * pc.ny
         : c == 1 ? x + y * pc.nx + z * pc.nx * (pc.ny + 1)
                  : x + y * pc.nx + z * pc.nx * pc.ny;
}
uint maxFaces() {
    return uint(max(faceCount(0), max(faceCount(1), faceCount(2))));
}

void weights(float x, out int b, out float w[3], out float dw[3]) {
    b = int(floor(x - 0.5));
    float d = x - float(b + 1);
    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
    dw[0] = d - 0.5; dw[1] = -2.0 * d; dw[2] = d + 0.5;
}

void addMomentum(int c, int f, float v) {
    if (c == 0) atomicAdd(mom_x[f], v); else if (c == 1) atomicAdd(mom_y[f], v); else atomicAdd(mom_z[f], v);
}
void addWeight(int c, int f, float v) {
    if (c == 0) atomicAdd(wt_x[f], v); else if (c == 1) atomicAdd(wt_y[f], v); else atomicAdd(wt_z[f], v);
}
void addStress(int c, int f, float v) {
    if ((pc.flags & 1u) == 0u) { addMomentum(c, f, v); return; }
    if (c == 0) atomicAdd(st_x[f], v); else if (c == 1) atomicAdd(st_y[f], v); else atomicAdd(st_z[f], v);
}

void clearPass(uint lane) {
    for (int c = 0; c < 3; ++c) {
        if (lane >= uint(faceCount(c))) continue;
        if (c == 0) { mom_x[lane] = 0.0; wt_x[lane] = 0.0; if ((pc.flags & 1u) != 0u) st_x[lane] = 0.0; }
        else if (c == 1) { mom_y[lane] = 0.0; wt_y[lane] = 0.0; if ((pc.flags & 1u) != 0u) st_y[lane] = 0.0; }
        else { mom_z[lane] = 0.0; wt_z[lane] = 0.0; if ((pc.flags & 1u) != 0u) st_z[lane] = 0.0; }
    }
}

void scatterPass(uint lane) {
    if (lane >= matter_count[pc.matter_lane]) return;
    int id = int(matter_index[lane]);
    if (id >= pc.particle_count || pc.voxel_size <= 1e-6) return;
    vec3 p = vec3(pos_data[id * 3], pos_data[id * 3 + 1], pos_data[id * 3 + 2]);
    if (any(isnan(p))) return;
    vec3 v = vec3(vel_data[id * 3], vel_data[id * 3 + 1], vel_data[id * 3 + 2]);
    float h = pc.voxel_size, ih = 1.0 / h;
    float mass = matter_rest[id] * matter_fraction[id];
    float dry = matter_dry_volume[id];
    int ai = id * 9;
    vec3 base = (p - vec3(pc.origin_x, pc.origin_y, pc.origin_z)) * ih;
    for (int c = 0; c < 3; ++c) {
        vec3 g = base;
        if (c == 0) g.yz -= vec2(0.5); else if (c == 1) g.xz -= vec2(0.5); else g.xy -= vec2(0.5);
        // No `continue` here: a lane leaving the loop body early would split
        // the cluster across iterations. An out-of-grid lane takes a base no
        // node accepts and contributes nothing.
        bool inside = !(any(isinf(g)) || any(lessThan(g, vec3(-2.0))) ||
            g.x > float(pc.nx) + 2.0 || g.y > float(pc.ny) + 2.0 || g.z > float(pc.nz) + 2.0);
        if (!inside) g = vec3(-100.0);
        int bx, by, bz;
        float wx[3], wy[3], wz[3], dx[3], dy[3], dz[3];
        weights(g.x, bx, wx, dx);
        weights(g.y, by, wy, dy);
        weights(g.z, bz, wz, dz);
        int xmax = c == 0 ? pc.nx : pc.nx - 1;
        int ymax = c == 1 ? pc.ny : pc.ny - 1;
        int zmax = c == 2 ? pc.nz : pc.nz - 1;
        float vp = c == 0 ? v.x : (c == 1 ? v.y : v.z);
        vec3 C_row = c == 0 ? vec3(affine[ai + 0], affine[ai + 3], affine[ai + 6])
                   : c == 1 ? vec3(affine[ai + 1], affine[ai + 4], affine[ai + 7])
                            : vec3(affine[ai + 2], affine[ai + 5], affine[ai + 8]);
        vec3 row = c == 0 ? vec3(sd[id * 3], ss[id * 3], ss[id * 3 + 1])
                 : c == 1 ? vec3(ss[id * 3], sd[id * 3 + 1], ss[id * 3 + 2])
                          : vec3(ss[id * 3 + 1], ss[id * 3 + 2], sd[id * 3 + 2]);
#ifdef P2G_CLUSTERED
        const bool same_base =
            subgroupClusteredMin(bx, 8u) == subgroupClusteredMax(bx, 8u) &&
            subgroupClusteredMin(by, 8u) == subgroupClusteredMax(by, 8u) &&
            subgroupClusteredMin(bz, 8u) == subgroupClusteredMax(bz, 8u);
        if (same_base) {
            const bool leader =
                subgroupClusteredMin(gl_SubgroupInvocationID, 8u) == gl_SubgroupInvocationID;
            for (int k = 0; k < 3; ++k)
            for (int j = 0; j < 3; ++j)
            for (int i = 0; i < 3; ++i) {
                int x = bx + i, y = by + j, z = bz + k;
                // Same base: in or out of range for the whole cluster.
                if (x < 0 || x > xmax || y < 0 || y > ymax || z < 0 || z > zmax) continue;
                float w = wx[i] * wy[j] * wz[k];
                vec3 dxw = vec3(float(x) - g.x, float(y) - g.y, float(z) - g.z) * h;
                float stress = 0.0;
                if (w >= kMinNodeWeight) {
                    vec3 grad = vec3(dx[i] * wy[j] * wz[k], wx[i] * dy[j] * wz[k],
                                     wx[i] * wy[j] * dz[k]) * ih;
                    stress = -pc.dt * dry * dot(row, grad);
                }
                const float momentum = subgroupClusteredAdd(w * mass * (vp + dot(C_row, dxw)), 8u);
                const float weight = subgroupClusteredAdd(w * mass, 8u);
                const float impulse = subgroupClusteredAdd(stress, 8u);
                if (!leader) continue;
                int f = faceIndex(c, x, y, z);
                addMomentum(c, f, momentum);
                addWeight(c, f, weight);
                addStress(c, f, impulse);
            }
            continue;
        }
#endif
        if (!inside) continue;
        for (int k = 0; k < 3; ++k)
        for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            int x = bx + i, y = by + j, z = bz + k;
            if (x < 0 || x > xmax || y < 0 || y > ymax || z < 0 || z > zmax) continue;
            float w = wx[i] * wy[j] * wz[k];
            int f = faceIndex(c, x, y, z);
            vec3 dxw = vec3(float(x) - g.x, float(y) - g.y, float(z) - g.z) * h;
            addMomentum(c, f, w * mass * (vp + dot(C_row, dxw)));
            addWeight(c, f, w * mass);
            if (w >= kMinNodeWeight) {
                vec3 grad = vec3(dx[i] * wy[j] * wz[k], wx[i] * dy[j] * wz[k],
                                 wx[i] * wy[j] * dz[k]) * ih;
                addStress(c, f, -pc.dt * dry * dot(row, grad));
            }
        }
    }
}

#ifdef P2G_BLOCK
void blockAdd(int field, int c, int x, int y, int z, float v) {
    if (field == 2 && (pc.flags & 1u) == 0u) field = 0;  // stress into the momentum (addStress)
    ivec3 l = ivec3(x, y, z) - g_tile_lo;
    if (all(greaterThanEqual(l, ivec3(0))) && all(lessThan(l, ivec3(kTile)))) {
        atomicAdd(tile[(field * 3 + c) * kTileNodes + l.x + kTile * (l.y + kTile * l.z)], v);
        return;
    }
    int f = faceIndex(c, x, y, z);
    if (field == 0) addMomentum(c, f, v); else if (field == 1) addWeight(c, f, v); else addStress(c, f, v);
}

// The per-component loop of scatterPass (unclustered branch), into the tile.
void blockScatterParticle(int id) {
    if (id >= pc.particle_count || pc.voxel_size <= 1e-6) return;
    vec3 p = vec3(pos_data[id * 3], pos_data[id * 3 + 1], pos_data[id * 3 + 2]);
    if (any(isnan(p))) return;
    vec3 v = vec3(vel_data[id * 3], vel_data[id * 3 + 1], vel_data[id * 3 + 2]);
    float h = pc.voxel_size, ih = 1.0 / h;
    float mass = matter_rest[id] * matter_fraction[id];
    float dry = matter_dry_volume[id];
    int ai = id * 9;
    vec3 base = (p - vec3(pc.origin_x, pc.origin_y, pc.origin_z)) * ih;
    for (int c = 0; c < 3; ++c) {
        vec3 g = base;
        if (c == 0) g.yz -= vec2(0.5); else if (c == 1) g.xz -= vec2(0.5); else g.xy -= vec2(0.5);
        if (any(isinf(g)) || any(lessThan(g, vec3(-2.0))) ||
            g.x > float(pc.nx) + 2.0 || g.y > float(pc.ny) + 2.0 || g.z > float(pc.nz) + 2.0) continue;
        int bx, by, bz;
        float wx[3], wy[3], wz[3], dx[3], dy[3], dz[3];
        weights(g.x, bx, wx, dx);
        weights(g.y, by, wy, dy);
        weights(g.z, bz, wz, dz);
        int xmax = c == 0 ? pc.nx : pc.nx - 1;
        int ymax = c == 1 ? pc.ny : pc.ny - 1;
        int zmax = c == 2 ? pc.nz : pc.nz - 1;
        float vp = c == 0 ? v.x : (c == 1 ? v.y : v.z);
        vec3 C_row = c == 0 ? vec3(affine[ai + 0], affine[ai + 3], affine[ai + 6])
                   : c == 1 ? vec3(affine[ai + 1], affine[ai + 4], affine[ai + 7])
                            : vec3(affine[ai + 2], affine[ai + 5], affine[ai + 8]);
        vec3 row = c == 0 ? vec3(sd[id * 3], ss[id * 3], ss[id * 3 + 1])
                 : c == 1 ? vec3(ss[id * 3], sd[id * 3 + 1], ss[id * 3 + 2])
                          : vec3(ss[id * 3 + 1], ss[id * 3 + 2], sd[id * 3 + 2]);
        // Lane-rotated stencil order: neighbouring lanes usually share a base
        // node (seed order is spatially coherent), so with a common order one
        // instruction sends the whole warp to the SAME shared address and the
        // atomics serialise. Starting each lane at a different stencil node
        // spreads them; the sum is the same. (Cell clusters with node
        // ownership were tried 2026-11-11 and were SLOWER: 1M snow 1000 ->
        // 1589 ms; with a 256 chunk 1994 ms. Removed.)
        const int rotate = int(gl_LocalInvocationID.x % 27u);
        for (int step = 0; step < 27; ++step) {
            const int n = (step + rotate) % 27;
            const int i = n % 3, j = (n / 3) % 3, k = n / 9;
            int x = bx + i, y = by + j, z = bz + k;
            if (x < 0 || x > xmax || y < 0 || y > ymax || z < 0 || z > zmax) continue;
            float w = wx[i] * wy[j] * wz[k];
            vec3 dxw = vec3(float(x) - g.x, float(y) - g.y, float(z) - g.z) * h;
            blockAdd(0, c, x, y, z, w * mass * (vp + dot(C_row, dxw)));
            blockAdd(1, c, x, y, z, w * mass);
            if (w >= kMinNodeWeight) {
                vec3 grad = vec3(dx[i] * wy[j] * wz[k], wx[i] * dy[j] * wz[k],
                                 wx[i] * wy[j] * dz[k]) * ih;
                blockAdd(2, c, x, y, z, -pc.dt * dry * dot(row, grad));
            }
        }
    }
}

void blockPass() {
    uint nb = uint(pc.bx * pc.by * pc.bz);
    uint b = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    if (b >= nb) return;                       // uniform per workgroup
    uint start = block_meta[nb + b];
    uint end = block_meta[nb + b + 1u];
    if (start == end) return;                  // uniform per workgroup
    ivec3 bc = ivec3(int(b) % pc.bx, (int(b) / pc.bx) % pc.by, int(b) / (pc.bx * pc.by));
    g_tile_lo = bc * kBlock - ivec3(2);
    uint t = gl_LocalInvocationID.x;
    for (uint i = t; i < uint(3 * 3 * kTileNodes); i += 256u) tile[i] = 0.0;
    barrier();
    // Linear walk. A strided permutation (k = m * P mod n) was measured 9 %
    // SLOWER (1M snow 1011 -> 1101 ms, 2026-11-11): it scatters the global
    // parcel reads more than it saves on shared-address conflicts.
    for (uint k = start + t; k < end; k += 256u) blockScatterParticle(int(block_items[k]));
    barrier();
    for (uint i = t; i < uint(3 * 3 * kTileNodes); i += 256u) {
        float value = tile[i];
        if (value == 0.0) continue;
        int slot = int(i) / kTileNodes;
        int field = slot / 3, c = slot % 3;
        int local = int(i) % kTileNodes;
        ivec3 node = g_tile_lo + ivec3(local % kTile, (local / kTile) % kTile, local / (kTile * kTile));
        // Only in-range nodes were ever added (blockScatterParticle checks).
        int f = faceIndex(c, node.x, node.y, node.z);
        if (field == 0) addMomentum(c, f, value);
        else if (field == 1) addWeight(c, f, value);
        else if (c == 0) atomicAdd(st_x[f], value);
        else if (c == 1) atomicAdd(st_y[f], value);
        else atomicAdd(st_z[f], value);
    }
}
#endif

void normalizePass(uint lane) {
    bool flip = (pc.flags & 1u) != 0u;
    for (int c = 0; c < 3; ++c) {
        if (lane >= uint(faceCount(c))) continue;
        float w = c == 0 ? wt_x[lane] : (c == 1 ? wt_y[lane] : wt_z[lane]);
        float m = c == 0 ? mom_x[lane] : (c == 1 ? mom_y[lane] : mom_z[lane]);
        float s = flip ? (c == 0 ? st_x[lane] : (c == 1 ? st_y[lane] : st_z[lane])) : 0.0;
        bool has = w > 1e-8;
        float velocity = has ? (m + s) / w : 0.0;
        if (c == 0) mom_x[lane] = velocity; else if (c == 1) mom_y[lane] = velocity; else mom_z[lane] = velocity;
        if (flip) {
            float baseline = has ? m / w : 0.0;
            if (c == 0) st_x[lane] = baseline; else if (c == 1) st_y[lane] = baseline; else st_z[lane] = baseline;
        }
    }
}

void main() {
#ifdef P2G_BLOCK
    // The block variant runs the scatter only; clear/normalize stay on the
    // plain fused kernel (same buffers).
    blockPass();
    return;
#endif
    uint count = pc.pass == 1 ? uint(pc.particle_count) : maxFaces();
    uint lane = simLane256(count);
    if (pc.pass == 0) clearPass(lane);
    else if (pc.pass == 1) scatterPass(lane);
    else normalizePass(lane);
}
