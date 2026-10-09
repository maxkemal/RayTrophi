// sim_fluid_subtract_gradient.comp
// Free-surface pressure-gradient subtraction. 1:1 port of the CUDA
// fluid_subtract_gradient_kernel (SimulationComputeCuda.cu), NON-variational,
// NON-GFM branch. One thread covers the same face id across all three MAC
// components (mirrors the CUDA kernel's three if-blocks).
// Face rules: solid face (either adjacent cell solid, or closed domain wall)
// => velocity forced to solid velocity (0, static walls). Otherwise subtract
// dt/h * (p_hi - p_lo) with air pressure = 0.
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
#ifdef RT_SPARSE_MAC
// Compact velocity pages are bound at 0..2 (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md).
layout(set = 0, binding = 5) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 6) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

int cell_id(int i, int j, int k) { return i + j*pc.nx + k*pc.nx*pc.ny; }

float mask_at(int i, int j, int k) {
    if (i < 0 || i >= pc.nx || j < 0 || j >= pc.ny || k < 0 || k >= pc.nz)
        return (pc.boundary == 0) ? 0.0 : -1.0;
    return fluid_mask[cell_id(i, j, k)];
}
bool is_solid(int i, int j, int k) { return mask_at(i, j, k) < -0.5; }
bool is_fluid(int i, int j, int k) { return mask_at(i, j, k) >  0.5; }

// Binary open weight (non-variational): domain-boundary face follows the wall
// mode (open=1 outflow, closed=0 wall); interior face closed if either side solid.
float open_weight_x(int i, int j, int k) {
    if (i <= 0 || i >= pc.nx) return (pc.boundary == 0) ? 1.0 : 0.0;
    return (is_solid(i - 1, j, k) || is_solid(i, j, k)) ? 0.0 : 1.0;
}
float open_weight_y(int i, int j, int k) {
    if (j <= 0 || j >= pc.ny) return (pc.boundary == 0) ? 1.0 : 0.0;
    return (is_solid(i, j - 1, k) || is_solid(i, j, k)) ? 0.0 : 1.0;
}
float open_weight_z(int i, int j, int k) {
    if (k <= 0 || k >= pc.nz) return (pc.boundary == 0) ? 1.0 : 0.0;
    return (is_solid(i, j, k - 1) || is_solid(i, j, k)) ? 0.0 : 1.0;
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
        if (open_weight_x(f.x, f.y, f.z) < 1e-6) {
            vel_x[a] = 0.0;
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
        if (open_weight_y(f.x, f.y, f.z) < 1e-6) {
            vel_y[a] = 0.0;
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
        if (open_weight_z(f.x, f.y, f.z) < 1e-6) {
            vel_z[a] = 0.0;
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
