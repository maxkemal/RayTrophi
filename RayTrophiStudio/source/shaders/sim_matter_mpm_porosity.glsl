// sim_matter_mpm_porosity.glsl -- the MPM lane as a porous moving solid in the
// liquid projection, computed on the device (2026-11-11; docs/dev/
// MATTER_SIVI_MPM_BASINC_BAGLASIMI.md). Until then the host read the MPM face
// masses and velocities back (~9.5 MB a liquid step), built these fields and
// uploaded them: at 17 liquid steps a frame (a pouring emitter) that round trip
// was ~170 MB down, ~150 MB up and over half of a ~900 ms frame.
//
// Same rules as the host code it replaces (MatterMpmPorosity.cpp):
//   pass 0, one lane per CELL: solid fraction phi = mean of its six faces'
//     min(m_face / (rho_c h^3), cap); collider cells (mask -1) and cells with no
//     MPM mass get 0. A porous cell's solid velocity, per component the
//     mass-weighted mean of its two faces of that axis, goes to var_sv*;
//   pass 1, one lane per lattice tuple (one X, Y and Z face each): the face's
//     phi is the mean of its in-grid cells, beta = min(1, (1 - phi) + phi r)
//     with r the effective mobility, and the projection's solid weight
//     (already uploaded/packed this solve) is multiplied by beta.
//   pass 2, one lane per CELL, after the solve: var_sv* back to 0 where pass 0
//     wrote (fraction > 0). A non-collider cell's solid velocity is always 0
//     on the host (only collider cells carry one), so this is the exact
//     restore and the ~4.5 MB re-upload per liquid step is gone.
// The weights are restored by the next solve: compact pages are repacked from
// the device source every call; dense weights are re-uploaded.
#include "sim_dispatch.glsl"
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int nx; int ny; int nz;
    int pass;              // 0 cells, 1 faces, 2 restore var_sv*
    float inv_full_mass;   // 1 / (rho_constituent h^3)
    float cap;             // 1 - minimum voidage
    float mobility;        // effective r (MatterGpuStep)
    float pad;
} pc;

layout(set = 0, binding = 0) readonly buffer MX { float mass_x[]; };   // MPM lane, dense
layout(set = 0, binding = 1) readonly buffer MY { float mass_y[]; };
layout(set = 0, binding = 2) readonly buffer MZ { float mass_z[]; };
layout(set = 0, binding = 3) readonly buffer VX { float vel_x[]; };
layout(set = 0, binding = 4) readonly buffer VY { float vel_y[]; };
layout(set = 0, binding = 5) readonly buffer VZ { float vel_z[]; };
layout(set = 0, binding = 6) readonly buffer Mask { float fluid_mask[]; };
layout(set = 0, binding = 7) buffer Fraction { float fraction[]; };
layout(set = 0, binding = 8) buffer SVX { float svx[]; };
layout(set = 0, binding = 9) buffer SVY { float svy[]; };
layout(set = 0, binding = 10) buffer SVZ { float svz[]; };
layout(set = 0, binding = 11) buffer WX { float weight_x[]; };          // liquid lane solid weights
layout(set = 0, binding = 12) buffer WY { float weight_y[]; };
layout(set = 0, binding = 13) buffer WZ { float weight_z[]; };
#ifdef RT_SPARSE_MAC
layout(set = 0, binding = 14) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 15) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

int cellIndex(int i, int j, int k) { return i + pc.nx * (j + pc.ny * k); }
int faceX(int i, int j, int k) { return i + (pc.nx + 1) * (j + pc.ny * k); }
int faceY(int i, int j, int k) { return i + pc.nx * (j + (pc.ny + 1) * k); }
int faceZ(int i, int j, int k) { return i + pc.nx * (j + pc.ny * k); }

float faceFraction(float m) {
    return m > 0.0 ? min(m * pc.inv_full_mass, pc.cap) : 0.0;
}

float meanVelocity(float m0, float v0, float m1, float v1) {
    float m = m0 + m1;
    return m > 0.0 ? (m0 * v0 + m1 * v1) / m : 0.0;
}

void cellPass(uint lane) {
    int cells = pc.nx * pc.ny * pc.nz;
    if (lane >= uint(cells)) return;
    int c = int(lane);
    int i = c % pc.nx, j = (c / pc.nx) % pc.ny, k = c / (pc.nx * pc.ny);
    fraction[c] = 0.0;
    if (fluid_mask[c] < -0.5) return;  // collider cell keeps its wall velocity
    int x0 = faceX(i, j, k), x1 = faceX(i + 1, j, k);
    int y0 = faceY(i, j, k), y1 = faceY(i, j + 1, k);
    int z0 = faceZ(i, j, k), z1 = faceZ(i, j, k + 1);
    float mx0 = mass_x[x0], mx1 = mass_x[x1];
    float my0 = mass_y[y0], my1 = mass_y[y1];
    float mz0 = mass_z[z0], mz1 = mass_z[z1];
    if (!(mx0 + mx1 > 0.0 || my0 + my1 > 0.0 || mz0 + mz1 > 0.0)) return;
    float phi = (faceFraction(mx0) + faceFraction(mx1) + faceFraction(my0) +
                 faceFraction(my1) + faceFraction(mz0) + faceFraction(mz1)) / 6.0;
    if (!(phi > 1.0e-4)) return;
    fraction[c] = phi;
    svx[c] = meanVelocity(mx0, vel_x[x0], mx1, vel_x[x1]);
    svy[c] = meanVelocity(my0, vel_y[y0], my1, vel_y[y1]);
    svz[c] = meanVelocity(mz0, vel_z[z0], mz1, vel_z[z1]);
}

void restorePass(uint lane) {
    if (lane >= uint(pc.nx * pc.ny * pc.nz)) return;
    int c = int(lane);
    if (!(fraction[c] > 0.0)) return;
    svx[c] = 0.0;
    svy[c] = 0.0;
    svz[c] = 0.0;
}

float cellPhi(ivec3 c, out bool inside) {
    inside = all(greaterThanEqual(c, ivec3(0))) && c.x < pc.nx && c.y < pc.ny && c.z < pc.nz;
    return inside ? fraction[cellIndex(c.x, c.y, c.z)] : 0.0;
}

void scaleFace(int component, ivec3 face) {
    ivec3 lo = face;
    lo[component] -= 1;
    bool a, b;
    float pa = cellPhi(lo, a), pb = cellPhi(face, b);
    float phi = a && b ? 0.5 * (pa + pb) : (a ? pa : pb);
    if (!(phi > 1.0e-4)) return;
    float beta = min(1.0, (1.0 - phi) + phi * max(pc.mobility, 0.0));
    if (beta >= 0.9999) return;  // as mobile as the liquid itself
    uint address = macAddress(component, face.x, face.y, face.z);
    if (address == MAC_ABSENT) return;
    if (component == 0) weight_x[address] *= beta;
    else if (component == 1) weight_y[address] *= beta;
    else weight_z[address] *= beta;
}

void facePass(uint lane) {
    uint count = uint((pc.nx + 1) * (pc.ny + 1) * (pc.nz + 1));
    if (lane >= count) return;
    int i = int(lane) % (pc.nx + 1);
    int j = (int(lane) / (pc.nx + 1)) % (pc.ny + 1);
    int k = int(lane) / ((pc.nx + 1) * (pc.ny + 1));
    // Each lattice tuple owns exactly one X, Y and Z face.
    if (j < pc.ny && k < pc.nz) scaleFace(0, ivec3(i, j, k));
    if (i < pc.nx && k < pc.nz) scaleFace(1, ivec3(i, j, k));
    if (i < pc.nx && j < pc.ny) scaleFace(2, ivec3(i, j, k));
}

void main() {
    uint count = pc.pass == 1 ? uint((pc.nx + 1) * (pc.ny + 1) * (pc.nz + 1))
                              : uint(pc.nx * pc.ny * pc.nz);
    uint lane = simLane256(count);
    if (pc.pass == 0) cellPass(lane);
    else if (pc.pass == 1) facePass(lane);
    else restorePass(lane);
}
