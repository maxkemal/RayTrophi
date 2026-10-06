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
layout(set = 0, binding = 5) readonly buffer MatterIndices { uint matter_index[]; };
layout(set = 0, binding = 6) readonly buffer MatterCounts { uint matter_count[]; };
#endif
#ifdef MATTER_INDEXED
layout(set = 0, binding = 7) readonly buffer MatterRestMass { float matter_rest[]; };
layout(set = 0, binding = 8) readonly buffer MatterFraction { float matter_fraction[]; };
#endif
#ifdef MATTER_INDEXED
layout(set = 0, binding = 9) buffer MatterGradient { float matter_gradient[]; };
#endif
void main() {
    int id = int(gl_GlobalInvocationID.x);
#ifdef MATTER_INDEXED
    if (uint(id) >= matter_count[pc.matter_lane]) return;
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

        atomicAdd(vel_field[fi], w * (vp + apic));
        atomicAdd(wt_field[fi],  w);
#ifdef MATTER_INDEXED
        atomicAdd(matter_gradient[fi * 3], parcel_mass * derivative.x * invH);
        atomicAdd(matter_gradient[fi * 3 + 1], parcel_mass * derivative.y * invH);
        atomicAdd(matter_gradient[fi * 3 + 2], parcel_mass * derivative.z * invH);
#endif
    }
}
