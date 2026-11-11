#include "sim_dispatch.glsl"
layout(local_size_x = 256) in;
layout(set = 0, binding = 0) buffer V0X { float v0x[]; };
layout(set = 0, binding = 1) buffer V0Y { float v0y[]; };
layout(set = 0, binding = 2) buffer V0Z { float v0z[]; };
layout(set = 0, binding = 3) buffer V1X { float v1x[]; };
layout(set = 0, binding = 4) buffer V1Y { float v1y[]; };
layout(set = 0, binding = 5) buffer V1Z { float v1z[]; };
layout(set = 0, binding = 6) readonly buffer M0X { float m0x[]; };
layout(set = 0, binding = 7) readonly buffer M0Y { float m0y[]; };
layout(set = 0, binding = 8) readonly buffer M0Z { float m0z[]; };
layout(set = 0, binding = 9) readonly buffer M1X { float m1x[]; };
layout(set = 0, binding = 10) readonly buffer M1Y { float m1y[]; };
layout(set = 0, binding = 11) readonly buffer M1Z { float m1z[]; };
layout(set = 0, binding = 12) readonly buffer G0X { float g0x[]; };
layout(set = 0, binding = 13) readonly buffer G0Y { float g0y[]; };
layout(set = 0, binding = 14) readonly buffer G0Z { float g0z[]; };
layout(set = 0, binding = 15) readonly buffer G1X { float g1x[]; };
layout(set = 0, binding = 16) readonly buffer G1Y { float g1y[]; };
layout(set = 0, binding = 17) readonly buffer G1Z { float g1z[]; };
layout(set = 0, binding = 18) buffer Stats { uint pair_count[]; };
// sim_matter_liquid_drag.glsl -- liquid <-> MPM momentum exchange as a RATE
// (docs/dev/MATTER_SIVI_MPM_BASINC_BAGLASIMI.md, step 2). Replaces
// sim_matter_contact on the liquid's own clock: that kernel removed the whole
// closing velocity of every face pair on every call (plus Coulomb friction), a
// per-call event that stopped water at a snow bank and made the result depend
// on the step. The normal coupling is the porous projection now; what is left
// is the drag of liquid flowing through the solid's pores. Darcy with
// Kozeny-Carman permeability k = d^2 eps^3 / (180 (1 - eps)^2):
//   beta = mu eps^2 / k = 180 mu (1 - eps)^2 / (d^2 eps)       [kg/(m^3 s)]
// on the face volume V. The pair relaxes exactly (implicit, any dt):
//   r' = r exp(-beta V (1/m_l + 1/m_s) dt),  J = (r - r') / (1/m_l + 1/m_s)
// equal and opposite: momentum is conserved to rounding, energy only lost.
// Bindings are sim_matter_contact's (the gradient slots are unused).
layout(push_constant) uniform Params {
    int nx; int ny; int nz;
    float dt;
    float drag_prefactor;  // 180 mu / d^2
    float inv_full_mass;   // 1 / (rho_constituent h^3)
    float max_fraction;    // 1 - minimum voidage (the projection's cap)
    float face_volume;     // h^3
} pc;
#ifdef RT_SPARSE_MAC
// Liquid lane (0) velocity and mass on compact pages at 0..2 and 6..8; the
// granular lane stays dense (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
layout(set = 0, binding = 19) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 20) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

void relax(inout float vl, inout float vg, float ml, float mg) {
    if (!(ml > 1e-8) || !(mg > 1e-8)) return;
    float phi = min(mg * pc.inv_full_mass, pc.max_fraction);
    if (phi < 1e-4) return;
    float eps = 1.0 - phi;
    float beta = pc.drag_prefactor * phi * phi / max(eps, 1e-3);
    float inv_mass = 1.0 / ml + 1.0 / mg;
    float rate = beta * pc.face_volume * inv_mass;
    float r = vl - vg;
    float kept = exp(-rate * pc.dt);
    float J = r * (1.0 - kept) / inv_mass;
    if (isnan(J) || isinf(J)) return;
    vl -= J / ml;
    vg += J / mg;
}

void main() {
    uint count = uint((pc.nx + 1) * (pc.ny + 1) * (pc.nz + 1));
    uint id = simLane256(count);
    if (id >= count) return;
    int i = int(id) % (pc.nx + 1);
    int j = (int(id) / (pc.nx + 1)) % (pc.ny + 1);
    int k = int(id) / ((pc.nx + 1) * (pc.ny + 1));
    // Each lattice tuple owns exactly one X, Y and Z face.
    ivec3 face = ivec3(i + j * (pc.nx + 1) + k * (pc.nx + 1) * pc.ny,
        i + j * pc.nx + k * pc.nx * (pc.ny + 1),
        i + j * pc.nx + k * pc.nx * pc.ny);
    bvec3 valid = bvec3(j < pc.ny && k < pc.nz,
        i < pc.nx && k < pc.nz, i < pc.nx && j < pc.ny);
    bool any_pair = false;
    if (valid[0]) {
        uint a = macAddress(0, i, j, k);
        if (a != MAC_ABSENT) {
            float vl = v0x[a], vg = v1x[face[0]];
            relax(vl, vg, m0x[a], m1x[face[0]]);
            if (vl != v0x[a]) { v0x[a] = vl; v1x[face[0]] = vg; any_pair = true; }
        }
    }
    if (valid[1]) {
        uint a = macAddress(1, i, j, k);
        if (a != MAC_ABSENT) {
            float vl = v0y[a], vg = v1y[face[1]];
            relax(vl, vg, m0y[a], m1y[face[1]]);
            if (vl != v0y[a]) { v0y[a] = vl; v1y[face[1]] = vg; any_pair = true; }
        }
    }
    if (valid[2]) {
        uint a = macAddress(2, i, j, k);
        if (a != MAC_ABSENT) {
            float vl = v0z[a], vg = v1z[face[2]];
            relax(vl, vg, m0z[a], m1z[face[2]]);
            if (vl != v0z[a]) { v0z[a] = vl; v1z[face[2]] = vg; any_pair = true; }
        }
    }
    if (any_pair) atomicAdd(pair_count[0], 1u);
}