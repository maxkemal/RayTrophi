// sim_fluid_subtract_gradient_var.comp
// Pressure-gradient subtraction, VARIATIONAL branch. 1:1 port of the CUDA
// fluid_subtract_gradient_kernel with weights present and gfm_active == 0.
//
// Two differences from the plain shader, both consequences of fractional faces:
//   - the "is this face closed?" test reads the WEIGHT array instead of a binary
//     solid test, so a face is only clamped when it is essentially fully blocked;
//   - a clamped face is set to the SOLID's velocity, not to zero. That is what
//     makes a moving collider drag fluid along instead of stopping it dead.
// GFM is deliberately not handled here: the host routes ghost-fluid domains to
// the CPU, and porting two matrix changes at once would make a regression
// impossible to attribute.
layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int   nx; int ny; int nz;
    int   boundary;
    float voxel_size;
    float dt;
    float sor_omega;
    int   iterations;
    int   parity;
    float density_correction;
    int   particles_per_cell;
    int   variational;
    int   gfm_active;
} pc;

layout(set = 0, binding = 0) buffer VelX { float vel_x[]; };
layout(set = 0, binding = 1) buffer VelY { float vel_y[]; };
layout(set = 0, binding = 2) buffer VelZ { float vel_z[]; };
layout(set = 0, binding = 3) readonly buffer Pressure { float pressure[]; };
layout(set = 0, binding = 4) readonly buffer Mask     { float fluid_mask[]; };
layout(set = 0, binding = 5) readonly buffer UW       { float uw[]; };
layout(set = 0, binding = 6) readonly buffer VW       { float vw[]; };
layout(set = 0, binding = 7) readonly buffer WW       { float ww[]; };
layout(set = 0, binding = 8) readonly buffer SVX      { float svx[]; };
layout(set = 0, binding = 9) readonly buffer SVY      { float svy[]; };
layout(set = 0, binding = 10) readonly buffer SVZ     { float svz[]; };
#ifdef RT_SPARSE_MAC
// Compact velocity pages are bound at 0..2 (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
layout(set = 0, binding = 11) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 12) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"
#include "sim_mac_solid_weight.glsl"

int cell_id(int i, int j, int k) { return i + j*pc.nx + k*pc.nx*pc.ny; }
int vx_idx(int i, int j, int k)  { return i + j*(pc.nx+1) + k*(pc.nx+1)*pc.ny; }
int vy_idx(int i, int j, int k)  { return i + j*pc.nx + k*pc.nx*(pc.ny+1); }
int vz_idx(int i, int j, int k)  { return i + j*pc.nx + k*pc.nx*pc.ny; }

float mask_at(int i, int j, int k) {
    if (i < 0 || i >= pc.nx || j < 0 || j >= pc.ny || k < 0 || k >= pc.nz)
        return (pc.boundary == 0) ? 0.0 : -1.0;
    return fluid_mask[cell_id(i, j, k)];
}
bool is_solid(int i, int j, int k) { return mask_at(i, j, k) < -0.5; }
bool is_fluid(int i, int j, int k) { return mask_at(i, j, k) >  0.5; }

float fw_x(int i, int j, int k) {
    if (i <= 0 || i >= pc.nx) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(0, i, j, k);
}
float fw_y(int i, int j, int k) {
    if (j <= 0 || j >= pc.ny) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(1, i, j, k);
}
float fw_z(int i, int j, int k) {
    if (k <= 0 || k >= pc.nz) return (pc.boundary == 0) ? 1.0 : 0.0;
    return macSolidWeight(2, i, j, k);
}

float sv_x(int i, int j, int k) {
    if (i - 1 >= 0 && is_solid(i - 1, j, k)) return svx[cell_id(i - 1, j, k)];
    if (i < pc.nx && is_solid(i, j, k))      return svx[cell_id(i, j, k)];
    return 0.0;
}
float sv_y(int i, int j, int k) {
    if (j - 1 >= 0 && is_solid(i, j - 1, k)) return svy[cell_id(i, j - 1, k)];
    if (j < pc.ny && is_solid(i, j, k))      return svy[cell_id(i, j, k)];
    return 0.0;
}
float sv_z(int i, int j, int k) {
    if (k - 1 >= 0 && is_solid(i, j, k - 1)) return svz[cell_id(i, j, k - 1)];
    if (k < pc.nz && is_solid(i, j, k))      return svz[cell_id(i, j, k)];
    return 0.0;
}

void main() {
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
    if (lane >= macLaneCount()) {
        return;
    }
    float h     = pc.voxel_size > 1e-6 ? pc.voxel_size : 1.0;
    float scale = pc.dt / h;
    ivec3 f;
    uint a;
    // One lane covers the same face slot of all three components. A face
    // closed to flow takes the wall velocity; otherwise subtract dt/h grad p
    // with air pressure 0.
    if (macLaneFace(lane, 0, f, a)) {
        if (fw_x(f.x, f.y, f.z) < 1e-6) {
            vel_x[a] = sv_x(f.x, f.y, f.z);
        } else {
            ivec3 lo = f - ivec3(1, 0, 0);
            bool lo_fluid = is_fluid(lo.x, lo.y, lo.z);
            bool hi_fluid = is_fluid(f.x, f.y, f.z);
            if (lo_fluid || hi_fluid) {
                float p_lo = lo_fluid ? pressure[cell_id(lo.x, lo.y, lo.z)] : 0.0;
                float p_hi = hi_fluid ? pressure[cell_id(f.x, f.y, f.z)] : 0.0;
                vel_x[a] -= scale * (p_hi - p_lo);
            }
        }
    }
    if (macLaneFace(lane, 1, f, a)) {
        if (fw_y(f.x, f.y, f.z) < 1e-6) {
            vel_y[a] = sv_y(f.x, f.y, f.z);
        } else {
            ivec3 lo = f - ivec3(0, 1, 0);
            bool lo_fluid = is_fluid(lo.x, lo.y, lo.z);
            bool hi_fluid = is_fluid(f.x, f.y, f.z);
            if (lo_fluid || hi_fluid) {
                float p_lo = lo_fluid ? pressure[cell_id(lo.x, lo.y, lo.z)] : 0.0;
                float p_hi = hi_fluid ? pressure[cell_id(f.x, f.y, f.z)] : 0.0;
                vel_y[a] -= scale * (p_hi - p_lo);
            }
        }
    }
    if (macLaneFace(lane, 2, f, a)) {
        if (fw_z(f.x, f.y, f.z) < 1e-6) {
            vel_z[a] = sv_z(f.x, f.y, f.z);
        } else {
            ivec3 lo = f - ivec3(0, 0, 1);
            bool lo_fluid = is_fluid(lo.x, lo.y, lo.z);
            bool hi_fluid = is_fluid(f.x, f.y, f.z);
            if (lo_fluid || hi_fluid) {
                float p_lo = lo_fluid ? pressure[cell_id(lo.x, lo.y, lo.z)] : 0.0;
                float p_hi = hi_fluid ? pressure[cell_id(f.x, f.y, f.z)] : 0.0;
                vel_z[a] -= scale * (p_hi - p_lo);
            }
        }
    }
}
