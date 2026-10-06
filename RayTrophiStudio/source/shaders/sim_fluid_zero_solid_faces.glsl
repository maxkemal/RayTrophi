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
    int t = int(gl_GlobalInvocationID.x);
    int nx = pc.nx, ny = pc.ny, nz = pc.nz;

    // x-faces: (nx+1) * ny * nz, index = i + (nx+1)*(j + ny*k)
    if (t < (nx + 1) * ny * nz) {
        int i = t % (nx + 1);
        int j = (t / (nx + 1)) % ny;
        int k = t / ((nx + 1) * ny);
        if (isSolid(i - 1, j, k) || isSolid(i, j, k)) vel_x[t] = 0.0;
    }
    // y-faces: nx * (ny+1) * nz, index = i + nx*(j + (ny+1)*k)
    if (t < nx * (ny + 1) * nz) {
        int i = t % nx;
        int j = (t / nx) % (ny + 1);
        int k = t / (nx * (ny + 1));
        if (isSolid(i, j - 1, k) || isSolid(i, j, k)) vel_y[t] = 0.0;
    }
    // z-faces: nx * ny * (nz+1), index = i + nx*(j + ny*k)
    if (t < nx * ny * (nz + 1)) {
        int i = t % nx;
        int j = (t / nx) % ny;
        int k = t / (nx * ny);
        if (isSolid(i, j, k - 1) || isSolid(i, j, k)) vel_z[t] = 0.0;
    }
}
