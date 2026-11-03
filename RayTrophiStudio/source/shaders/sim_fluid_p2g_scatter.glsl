// sim_fluid_p2g_scatter.comp
// APIC particle-to-grid scatter. Vec3 data = 3 tightly-packed floats (stride 12).
//
// Vulkan is the primary GPU fluid path: it is the vendor-independent one, so it
// is what gets exercised day to day. CUDA is somewhat faster where it exists,
// but it is the alternate, not the reference.
//
// Accumulation is float atomics (GL_EXT_shader_atomic_float). An integer
// fixed-point variant was tried and produced no flow; don't re-attempt it
// without also revisiting the scale factor that made it collapse.
#extension GL_EXT_shader_atomic_float : require
#if defined(P2G_CLUSTERED) && !defined(RT_SPARSE_MAC)
// Contention. Seeding fills a cell's parcels consecutively, so the eight lanes
// of a cluster usually share one base node and their 27-node stencils are the
// same addresses: per-lane atomics serialised on them. Measured on resting
// snow (2026-11-04): P2G time hardly fell when 65 % of the parcels left the
// lane, the signature of same-address contention, not throughput. A cluster
// whose lanes share the base node sums its contributions first and one lane
// issues the atomic; any other cluster keeps the per-lane path. Loaded only
// where the device reports clustered subgroup ops (SimulationComputeVulkan).
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_clustered : require
#define P2G_CLUSTER_ACTIVE 1
#endif
#include "sim_dispatch.glsl"
#ifdef RT_SPARSE_MAC
#include "sim_sparse_mac_address.glsl"
layout(set = 0, binding = 5) readonly buffer TileMap { uint tile_map[]; };
layout(set = 0, binding = 6) readonly buffer TileList { uint tile_list[]; };
#endif
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int nx; int ny; int nz;
    int particle_count;
    int component;
    float origin_x; float origin_y; float origin_z;
    float voxel_size;
#ifdef MATTER_INDEXED
    uint matter_lane;
#endif
} pc;

layout(set = 0, binding = 0) readonly buffer Positions  { float pos_data[]; };
layout(set = 0, binding = 1) readonly buffer Velocities { float vel_data[]; };
layout(set = 0, binding = 2) readonly buffer Affine     { float affine[];   };
layout(set = 0, binding = 3) buffer VelField            { float vel_field[]; };
layout(set = 0, binding = 4) buffer WeightField         { float wt_field[];  };

void quadratic_weights(float fx, out int base, out float w[3]) {
    base = int(floor(fx - 0.5));
    float d = fx - float(base + 1);
    w[0] = 0.5 * (0.5 - d) * (0.5 - d);
    w[1] = 0.75 - d * d;
    w[2] = 0.5 * (0.5 + d) * (0.5 + d);
}


#ifdef MATTER_INDEXED
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 7) readonly buffer MatterIndices { uint matter_index[]; };
#else
layout(set = 0, binding = 5) readonly buffer MatterIndices { uint matter_index[]; };
#endif
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 8) readonly buffer MatterCounts { uint matter_count[]; };
#else
layout(set = 0, binding = 6) readonly buffer MatterCounts { uint matter_count[]; };
#endif
#endif
#ifdef MATTER_INDEXED
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 9) readonly buffer MatterRestMass { float matter_rest[]; };
#else
layout(set = 0, binding = 7) readonly buffer MatterRestMass { float matter_rest[]; };
#endif
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 10) readonly buffer MatterFraction { float matter_fraction[]; };
#else
layout(set = 0, binding = 8) readonly buffer MatterFraction { float matter_fraction[]; };
#endif
#endif
#ifdef MATTER_INDEXED
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 11) buffer MatterGradient { float matter_gradient[]; };
#else
layout(set = 0, binding = 9) buffer MatterGradient { float matter_gradient[]; };
#endif
#endif
void main() {
    uint lane = simLane256(uint(pc.particle_count));
    if (lane >= uint(pc.particle_count)) return;
    int id = int(lane);
#ifdef MATTER_INDEXED
    // Bit 31: no face mass gradient this dispatch (no liquid to contact).
    const uint matter_lane = pc.matter_lane & 0x7fffffffu;
    const bool write_gradient = (pc.matter_lane & 0x80000000u) == 0u;
    if (uint(id) >= matter_count[matter_lane]) return;
    id = int(matter_index[id]);
#endif
    if (id >= pc.particle_count || pc.voxel_size <= 1e-6) return;

    // Read position and velocity (3 floats each, stride 12)
    vec3 p = vec3(pos_data[id*3], pos_data[id*3+1], pos_data[id*3+2]);
    if (isnan(p.x) || isnan(p.y) || isnan(p.z)) return;
    vec3 v = vec3(vel_data[id*3], vel_data[id*3+1], vel_data[id*3+2]);

    float h    = pc.voxel_size;
    float invH = 1.0 / h;

    float gx = (p.x - pc.origin_x) * invH;
    float gy = (p.y - pc.origin_y) * invH;
    float gz = (p.z - pc.origin_z) * invH;
    if (pc.component == 0) { gy -= 0.5; gz -= 0.5; }
    else if (pc.component == 1) { gx -= 0.5; gz -= 0.5; }
    else                        { gx -= 0.5; gy -= 0.5; }

    if (isinf(gx) || isinf(gy) || isinf(gz) ||
        gx < -2.0 || gy < -2.0 || gz < -2.0 ||
        gx > float(pc.nx) + 2.0 || gy > float(pc.ny) + 2.0 ||
        gz > float(pc.nz) + 2.0) return;

    int bx, by, bz;
    float wx[3], wy[3], wz[3];
    quadratic_weights(gx, bx, wx);
    quadratic_weights(gy, by, wy);
    quadratic_weights(gz, bz, wz);

    int xmax = (pc.component == 0) ? pc.nx     : pc.nx - 1;
    int ymax = (pc.component == 1) ? pc.ny     : pc.ny - 1;
    int zmax = (pc.component == 2) ? pc.nz     : pc.nz - 1;

    float vp = (pc.component == 0) ? v.x : (pc.component == 1) ? v.y : v.z;

    // APIC C matrix row for this component (col0.comp, col1.comp, col2.comp)
    int ai = id * 9;
    vec3 C_row;
    if      (pc.component == 0) C_row = vec3(affine[ai+0], affine[ai+3], affine[ai+6]);
    else if (pc.component == 1) C_row = vec3(affine[ai+1], affine[ai+4], affine[ai+7]);
    else                        C_row = vec3(affine[ai+2], affine[ai+5], affine[ai+8]);

#ifdef P2G_CLUSTER_ACTIVE
    const bool same_base =
        subgroupClusteredMin(bx, 8u) == subgroupClusteredMax(bx, 8u) &&
        subgroupClusteredMin(by, 8u) == subgroupClusteredMax(by, 8u) &&
        subgroupClusteredMin(bz, 8u) == subgroupClusteredMax(bz, 8u);
    if (same_base) {
        const bool leader =
            subgroupClusteredMin(gl_SubgroupInvocationID, 8u) == gl_SubgroupInvocationID;
#ifdef MATTER_INDEXED
        const float cluster_mass = matter_rest[id] * matter_fraction[id];
#endif
        for (int dk = 0; dk < 3; ++dk)
        for (int dj = 0; dj < 3; ++dj)
        for (int di = 0; di < 3; ++di) {
            int gi = bx+di, gj = by+dj, gk = bz+dk;
            // Same base node: in or out of range for the whole cluster.
            if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;
            float w = wx[di] * wy[dj] * wz[dk];
#ifdef MATTER_INDEXED
            vec3 derivative = vec3(
                (di == 0 ? gx - float(bx + 1) - 0.5 : di == 1 ?
                    -2.0 * (gx - float(bx + 1)) : gx - float(bx + 1) + 0.5) * wy[dj] * wz[dk],
                (dj == 0 ? gy - float(by + 1) - 0.5 : dj == 1 ?
                    -2.0 * (gy - float(by + 1)) : gy - float(by + 1) + 0.5) * wx[di] * wz[dk],
                (dk == 0 ? gz - float(bz + 1) - 0.5 : dk == 1 ?
                    -2.0 * (gz - float(bz + 1)) : gz - float(bz + 1) + 0.5) * wx[di] * wy[dj]);
            w *= cluster_mass;
#endif
            vec3 dxw = vec3(float(gi)-gx, float(gj)-gy, float(gk)-gz) * h;
            const float momentum = subgroupClusteredAdd(w * (vp + dot(C_row, dxw)), 8u);
            const float weight = subgroupClusteredAdd(w, 8u);
#ifdef MATTER_INDEXED
            vec3 gradient = vec3(0.0);
            if (write_gradient) {
                gradient = vec3(subgroupClusteredAdd(cluster_mass * derivative.x * invH, 8u),
                                subgroupClusteredAdd(cluster_mass * derivative.y * invH, 8u),
                                subgroupClusteredAdd(cluster_mass * derivative.z * invH, 8u));
            }
#endif
            if (!leader) continue;
            int fi;
            if      (pc.component == 0) fi = gi + gj*(pc.nx+1) + gk*(pc.nx+1)*pc.ny;
            else if (pc.component == 1) fi = gi + gj*pc.nx     + gk*pc.nx*(pc.ny+1);
            else                        fi = gi + gj*pc.nx     + gk*pc.nx*pc.ny;
            atomicAdd(vel_field[fi], momentum);
            atomicAdd(wt_field[fi], weight);
#ifdef MATTER_INDEXED
            if (write_gradient) {
                atomicAdd(matter_gradient[fi * 3], gradient.x);
                atomicAdd(matter_gradient[fi * 3 + 1], gradient.y);
                atomicAdd(matter_gradient[fi * 3 + 2], gradient.z);
            }
#endif
        }
        return;
    }
#endif
    for (int dk = 0; dk < 3; ++dk)
    for (int dj = 0; dj < 3; ++dj)
    for (int di = 0; di < 3; ++di) {
        int gi = bx+di, gj = by+dj, gk = bz+dk;
        if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;

        float w = wx[di] * wy[dj] * wz[dk];
#ifdef MATTER_INDEXED
        float parcel_mass = matter_rest[id] * matter_fraction[id];
        vec3 derivative = vec3(
            (di == 0 ? gx - float(bx + 1) - 0.5 : di == 1 ?
                -2.0 * (gx - float(bx + 1)) : gx - float(bx + 1) + 0.5) * wy[dj] * wz[dk],
            (dj == 0 ? gy - float(by + 1) - 0.5 : dj == 1 ?
                -2.0 * (gy - float(by + 1)) : gy - float(by + 1) + 0.5) * wx[di] * wz[dk],
            (dk == 0 ? gz - float(bz + 1) - 0.5 : dk == 1 ?
                -2.0 * (gz - float(bz + 1)) : gz - float(bz + 1) + 0.5) * wx[di] * wy[dj]);
        w *= parcel_mass;
#endif
        vec3 dxw = vec3(float(gi)-gx, float(gj)-gy, float(gk)-gz) * h;
        float apic = dot(C_row, dxw);

        int fi;
        if      (pc.component == 0) fi = gi + gj*(pc.nx+1) + gk*(pc.nx+1)*pc.ny;
        else if (pc.component == 1) fi = gi + gj*pc.nx     + gk*pc.nx*(pc.ny+1);
        else                        fi = gi + gj*pc.nx     + gk*pc.nx*pc.ny;

        int dense_fi = fi;
#ifdef RT_SPARSE_MAC
        ivec3 face = ivec3(gi, gj, gk);
        ivec3 cells = ivec3(pc.nx, pc.ny, pc.nz);
        uint slot = tile_map[sparseMacTileKey(face, pc.component, cells)];
        if (slot == 0u) continue;
        fi = int((slot - 1u) * 576u + sparseMacLocal(face, pc.component, cells));
#endif
        atomicAdd(vel_field[fi], w * (vp + apic));
        atomicAdd(wt_field[fi],  w);
#ifdef MATTER_INDEXED
        if (write_gradient) {
            atomicAdd(matter_gradient[dense_fi * 3], parcel_mass * derivative.x * invH);
            atomicAdd(matter_gradient[dense_fi * 3 + 1], parcel_mass * derivative.y * invH);
            atomicAdd(matter_gradient[dense_fi * 3 + 2], parcel_mass * derivative.z * invH);
        }
#endif
    }
}
