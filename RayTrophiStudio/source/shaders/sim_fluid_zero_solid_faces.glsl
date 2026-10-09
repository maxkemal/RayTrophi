// sim_fluid_zero_solid_faces.comp
// Device twin of APICFluidSolver.cpp:enforceSolidBoundaries. Zeros every MAC
// face whose either neighbour cell is solid OR outside the domain, for all three
// components in one dispatch (thread t handles x-face t, y-face t and z-face t).
//
// Exists so the granular substep chain can keep the grid velocity on the device
// between P2G and G2P. The host version needed the field downloaded, clamped and
// uploaded again on EVERY elastic substep - on a 114^3 domain that was ~60 MB of
// host traffic per substep, x32 substeps, for a thousand particles.
//
// Solidity comes from fluid_mask (< -0.5 = solid), which
// buildFluidMaskFromParticles stamps from exactly the grid.solid[] the host
// version reads. Out-of-range neighbours count as solid, as isSolid() does.
layout(local_size_x = 256) in;

// Same 36-byte layout as FluidP2GGpuConstants; only nx/ny/nz are read.
layout(push_constant) uniform PC {
    int nx; int ny; int nz;
    int particle_count;
    int component;
    float origin_x; float origin_y; float origin_z;
    float voxel_size;
} pc;

layout(set = 0, binding = 0) buffer VelX { float vel_x[]; };
layout(set = 0, binding = 1) buffer VelY { float vel_y[]; };
layout(set = 0, binding = 2) buffer VelZ { float vel_z[]; };
layout(set = 0, binding = 3) readonly buffer Mask { float fluid_mask[]; };
#ifdef RT_SPARSE_MAC
// Compact pages are bound at 0..2; the lanes cover the resident tiles only.
layout(set = 0, binding = 4) readonly buffer MacTileMap { uint mac_tile_map[]; };
layout(set = 0, binding = 5) readonly buffer MacTileList { uint mac_tile_list[]; };
#endif
#include "sim_mac_lane.glsl"

bool isSolid(int i, int j, int k) {
    if (i < 0 || i >= pc.nx || j < 0 || j >= pc.ny || k < 0 || k >= pc.nz) {
#ifdef MATTER_BOUNDARY
        return pc.component != 0;
#else
        return true;
#endif
    }
    return fluid_mask[(k * pc.ny + j) * pc.nx + i] < -0.5;
}

void main() {
#ifdef RT_SPARSE_MAC
    uint lane = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * 256u;
#else
    uint lane = gl_GlobalInvocationID.x;
#endif
    ivec3 f;
    uint a;
    // x-faces: either neighbour along x solid.
    if (macLaneFace(lane, 0, f, a) &&
        (isSolid(f.x - 1, f.y, f.z) || isSolid(f.x, f.y, f.z))) {
        vel_x[a] = 0.0;
    }
    if (macLaneFace(lane, 1, f, a) &&
        (isSolid(f.x, f.y - 1, f.z) || isSolid(f.x, f.y, f.z))) {
        vel_y[a] = 0.0;
    }
    if (macLaneFace(lane, 2, f, a) &&
        (isSolid(f.x, f.y, f.z - 1) || isSolid(f.x, f.y, f.z))) {
        vel_z[a] = 0.0;
    }
}
