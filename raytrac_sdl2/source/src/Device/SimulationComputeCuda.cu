/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          SimulationComputeCuda.cu
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*/
//
// CUDA implementation of the backend-independent simulation compute API
// (SimulationCompute.h). Phase 2: device buffers + a dispatch registry with a
// validation kernel + a self-test. The grid fluid solver kernels are added in
// Phase 3 and dispatched through this same backend.
//
#include "SimulationCompute.h"

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>

namespace RayTrophiSim {

namespace {

// ── Phase 2 validation kernel ────────────────────────────────────────────────
struct ScaleConstants {
    float factor;
    int count;
};

struct GridProjectionConstants {
    int nx;
    int ny;
    int nz;
    int boundary; // 0=open, 1=closed, 2=periodic
    float voxel_size;
    float dt;
    float sor_omega;
    int iterations;
    int parity;
    // MGPCG extras (appended — by-value struct, older kernels ignore them).
    // fluid_mask semantics for the CG path: >0.5 = fluid cell (value = particle
    // count for density correction); 0 = air; <-0.5 = solid (Neumann wall).
    float density_correction; // Bridson density-targeted projection gain (0=off)
    int   particles_per_cell; // target count per cell for the over-pack term
};

struct GridScalarAdvectionConstants {
    int nx;
    int ny;
    int nz;
    float voxel_size;
    float dt;
};

struct GridVelocityDissipationConstants {
    int nx;
    int ny;
    int nz;
    float factor;
    float max_velocity;
};

struct FluidDensitySplatConstants {
    int nx;
    int ny;
    int nz;
    int particle_count;
    float origin_x;
    float origin_y;
    float origin_z;
    float voxel_size;
    float particle_density;
};

struct FluidParticleIntegrateConstants {
    int particle_count;
    float dt;
    float gravity_x;
    float gravity_y;
    float gravity_z;
    float container_velocity_x;
    float container_velocity_y;
    float container_velocity_z;
    float max_velocity;
};

struct FluidP2GConstants {
    int nx;
    int ny;
    int nz;
    int particle_count;
    int component;
    float origin_x;
    float origin_y;
    float origin_z;
    float voxel_size;
};

struct DeviceVec3 {
    float x;
    float y;
    float z;
};

struct DeviceAffineC {
    DeviceVec3 col0;
    DeviceVec3 col1;
    DeviceVec3 col2;
};

__global__ void sim_scale_kernel(float* data, float factor, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] *= factor;
    }
}

__global__ void fluid_clear_float_kernel(float* values, int count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < count) {
        values[id] = 0.0f;
    }
}

__global__ void fluid_particle_integrate_forces_kernel(DeviceVec3* velocities,
                                                       FluidParticleIntegrateConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= c.particle_count) return;

    DeviceVec3 v = velocities[id];
    v.x += c.gravity_x * c.dt + c.container_velocity_x;
    v.y += c.gravity_y * c.dt + c.container_velocity_y;
    v.z += c.gravity_z * c.dt + c.container_velocity_z;

    const float max_v = c.max_velocity;
    if (max_v > 1e-6f) {
        const float speed2 = v.x * v.x + v.y * v.y + v.z * v.z;
        if (speed2 > max_v * max_v) {
            const float scale = max_v * rsqrtf(speed2);
            v.x *= scale;
            v.y *= scale;
            v.z *= scale;
        }
    }

    velocities[id] = v;
}

__device__ __forceinline__ void quadratic_weights_cuda(float fx, int& base, float w[3]) {
    base = static_cast<int>(floorf(fx - 0.5f));
    const float d = fx - (static_cast<float>(base) + 1.0f);
    w[0] = 0.5f * (0.5f - d) * (0.5f - d);
    w[1] = 0.75f - d * d;
    w[2] = 0.5f * (0.5f + d) * (0.5f + d);
}

__global__ void fluid_p2g_scatter_kernel(const DeviceVec3* positions,
                                         const DeviceVec3* velocities,
                                         const DeviceAffineC* affine,
                                         float* velocity_field,
                                         float* weight_field,
                                         FluidP2GConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= c.particle_count || c.voxel_size <= 1e-6f) return;

    const DeviceVec3 p = positions[id];
    const DeviceVec3 v = velocities[id];
    const DeviceAffineC C = affine[id];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

    const float h = c.voxel_size;
    const float inv_h = 1.0f / h;
    float gx = (p.x - c.origin_x) * inv_h;
    float gy = (p.y - c.origin_y) * inv_h;
    float gz = (p.z - c.origin_z) * inv_h;
    if (c.component == 0) {
        gy -= 0.5f;
        gz -= 0.5f;
    } else if (c.component == 1) {
        gx -= 0.5f;
        gz -= 0.5f;
    } else {
        gx -= 0.5f;
        gy -= 0.5f;
    }

    int bx, by, bz;
    float wx[3], wy[3], wz[3];
    quadratic_weights_cuda(gx, bx, wx);
    quadratic_weights_cuda(gy, by, wy);
    quadratic_weights_cuda(gz, bz, wz);

    const int xmax = (c.component == 0) ? c.nx : c.nx - 1;
    const int ymax = (c.component == 1) ? c.ny : c.ny - 1;
    const int zmax = (c.component == 2) ? c.nz : c.nz - 1;
    const float vp = (c.component == 0) ? v.x : ((c.component == 1) ? v.y : v.z);

    for (int dk = 0; dk < 3; ++dk)
    for (int dj = 0; dj < 3; ++dj)
    for (int di = 0; di < 3; ++di) {
        const int gi = bx + di;
        const int gj = by + dj;
        const int gk = bz + dk;
        if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;

        const float w = wx[di] * wy[dj] * wz[dk];
        const float dx = (static_cast<float>(gi) - gx) * h;
        const float dy = (static_cast<float>(gj) - gy) * h;
        const float dz = (static_cast<float>(gk) - gz) * h;
        float apic = 0.0f;
        if (c.component == 0) {
            apic = C.col0.x * dx + C.col1.x * dy + C.col2.x * dz;
        } else if (c.component == 1) {
            apic = C.col0.y * dx + C.col1.y * dy + C.col2.y * dz;
        } else {
            apic = C.col0.z * dx + C.col1.z * dy + C.col2.z * dz;
        }

        int field_index = 0;
        if (c.component == 0) {
            field_index = gi + gj * (c.nx + 1) + gk * (c.nx + 1) * c.ny;
        } else if (c.component == 1) {
            field_index = gi + gj * c.nx + gk * c.nx * (c.ny + 1);
        } else {
            field_index = gi + gj * c.nx + gk * c.nx * c.ny;
        }
        atomicAdd(&velocity_field[field_index], w * (vp + apic));
        atomicAdd(&weight_field[field_index], w);
    }
}

__global__ void fluid_p2g_normalize_kernel(float* velocity_field,
                                           const float* weight_field,
                                           int count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count) return;
    const float w = weight_field[id];
    velocity_field[id] = (w > 1e-8f) ? (velocity_field[id] / w) : 0.0f;
}

__global__ void fluid_density_clear_kernel(float* density, int cell_count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < cell_count) {
        density[id] = 0.0f;
    }
}

__global__ void fluid_density_splat_kernel(const DeviceVec3* positions,
                                           float* density,
                                           FluidDensitySplatConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= c.particle_count || c.voxel_size <= 1e-6f) return;

    const DeviceVec3 p = positions[id];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

    const float inv_h = 1.0f / c.voxel_size;
    const float lx = (p.x - c.origin_x) * inv_h - 0.5f;
    const float ly = (p.y - c.origin_y) * inv_h - 0.5f;
    const float lz = (p.z - c.origin_z) * inv_h - 0.5f;
    const int i0 = static_cast<int>(floorf(lx));
    const int j0 = static_cast<int>(floorf(ly));
    const int k0 = static_cast<int>(floorf(lz));
    const float fx = lx - static_cast<float>(i0);
    const float fy = ly - static_cast<float>(j0);
    const float fz = lz - static_cast<float>(k0);

    for (int dz = 0; dz <= 1; ++dz) {
        const int k = k0 + dz;
        if (k < 0 || k >= c.nz) continue;
        const float wz = dz ? fz : (1.0f - fz);
        for (int dy = 0; dy <= 1; ++dy) {
            const int j = j0 + dy;
            if (j < 0 || j >= c.ny) continue;
            const float wy = dy ? fy : (1.0f - fy);
            for (int dx = 0; dx <= 1; ++dx) {
                const int i = i0 + dx;
                if (i < 0 || i >= c.nx) continue;
                const float wx = dx ? fx : (1.0f - fx);
                const int cell = i + j * c.nx + k * c.nx * c.ny;
                atomicAdd(&density[cell],
                          c.particle_density * wx * wy * wz);
            }
        }
    }
}

__device__ __forceinline__ int cell_index(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

__device__ __forceinline__ int vel_x_index(int i, int j, int k, int nx, int ny) {
    return i + j * (nx + 1) + k * (nx + 1) * ny;
}

__device__ __forceinline__ int vel_y_index(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * (ny + 1);
}

__device__ __forceinline__ int vel_z_index(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

__device__ __forceinline__ int wrap_index(int v, int n) {
    return (v % n + n) % n;
}

__device__ float sample_pressure(const float* pressure,
                                 int i,
                                 int j,
                                 int k,
                                 const GridProjectionConstants& c) {
    if (c.boundary == 2) {
        i = wrap_index(i, c.nx);
        j = wrap_index(j, c.ny);
        k = wrap_index(k, c.nz);
        return pressure[cell_index(i, j, k, c.nx, c.ny)];
    }

    if (i < 0 || i >= c.nx || j < 0 || j >= c.ny || k < 0 || k >= c.nz) {
        if (c.boundary == 0) {
            return 0.0f;
        }
        i = min(max(i, 0), c.nx - 1);
        j = min(max(j, 0), c.ny - 1);
        k = min(max(k, 0), c.nz - 1);
    }
    return pressure[cell_index(i, j, k, c.nx, c.ny)];
}

__device__ __forceinline__ float lerp_float(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ float trilinear_clamped(const float* field,
                                   int i0,
                                   int j0,
                                   int k0,
                                   float fx,
                                   float fy,
                                   float fz,
                                   int nx,
                                   int ny,
                                   int nz) {
    i0 = min(max(i0, 0), nx - 1);
    j0 = min(max(j0, 0), ny - 1);
    k0 = min(max(k0, 0), nz - 1);
    const int i1 = min(i0 + 1, nx - 1);
    const int j1 = min(j0 + 1, ny - 1);
    const int k1 = min(k0 + 1, nz - 1);

    const float c000 = field[cell_index(i0, j0, k0, nx, ny)];
    const float c100 = field[cell_index(i1, j0, k0, nx, ny)];
    const float c010 = field[cell_index(i0, j1, k0, nx, ny)];
    const float c110 = field[cell_index(i1, j1, k0, nx, ny)];
    const float c001 = field[cell_index(i0, j0, k1, nx, ny)];
    const float c101 = field[cell_index(i1, j0, k1, nx, ny)];
    const float c011 = field[cell_index(i0, j1, k1, nx, ny)];
    const float c111 = field[cell_index(i1, j1, k1, nx, ny)];
    const float c00 = lerp_float(c000, c100, fx);
    const float c10 = lerp_float(c010, c110, fx);
    const float c01 = lerp_float(c001, c101, fx);
    const float c11 = lerp_float(c011, c111, fx);
    return lerp_float(lerp_float(c00, c10, fy), lerp_float(c01, c11, fy), fz);
}

__device__ float sample_vel_x(const float* vel_x, float fi, float fj, float fk, int nx, int ny, int nz) {
    const int i0 = static_cast<int>(floorf(fi));
    const int j0 = static_cast<int>(floorf(fj));
    const int k0 = static_cast<int>(floorf(fk));
    return trilinear_clamped(vel_x, i0, j0, k0, fi - i0, fj - j0, fk - k0, nx + 1, ny, nz);
}

__device__ float sample_vel_y(const float* vel_y, float fi, float fj, float fk, int nx, int ny, int nz) {
    const int i0 = static_cast<int>(floorf(fi));
    const int j0 = static_cast<int>(floorf(fj));
    const int k0 = static_cast<int>(floorf(fk));
    return trilinear_clamped(vel_y, i0, j0, k0, fi - i0, fj - j0, fk - k0, nx, ny + 1, nz);
}

__device__ float sample_vel_z(const float* vel_z, float fi, float fj, float fk, int nx, int ny, int nz) {
    const int i0 = static_cast<int>(floorf(fi));
    const int j0 = static_cast<int>(floorf(fj));
    const int k0 = static_cast<int>(floorf(fk));
    return trilinear_clamped(vel_z, i0, j0, k0, fi - i0, fj - j0, fk - k0, nx, ny, nz + 1);
}

__device__ float sample_cell_centered_open(const float* field,
                                           float local_x,
                                           float local_y,
                                           float local_z,
                                           int nx,
                                           int ny,
                                           int nz) {
    const float x = local_x - 0.5f;
    const float y = local_y - 0.5f;
    const float z = local_z - 0.5f;
    const int i0 = static_cast<int>(floorf(x));
    const int j0 = static_cast<int>(floorf(y));
    const int k0 = static_cast<int>(floorf(z));
    if (i0 < 0 || i0 >= nx - 1 || j0 < 0 || j0 >= ny - 1 || k0 < 0 || k0 >= nz - 1) {
        return 0.0f;
    }
    return trilinear_clamped(field, i0, j0, k0, x - i0, y - j0, z - k0, nx, ny, nz);
}

__global__ void grid_divergence_kernel(float* vel_x,
                                       float* vel_y,
                                       float* vel_z,
                                       float* pressure,
                                       float* divergence,
                                       GridProjectionConstants c) {
    (void)pressure;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = c.nx * c.ny * c.nz;
    if (id >= cell_count) return;

    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    const float inv_h = c.voxel_size > 1e-6f ? 1.0f / c.voxel_size : 1.0f;
    const float du = vel_x[vel_x_index(i + 1, j, k, c.nx, c.ny)] -
                     vel_x[vel_x_index(i, j, k, c.nx, c.ny)];
    const float dv = vel_y[vel_y_index(i, j + 1, k, c.nx, c.ny)] -
                     vel_y[vel_y_index(i, j, k, c.nx, c.ny)];
    const float dw = vel_z[vel_z_index(i, j, k + 1, c.nx, c.ny)] -
                     vel_z[vel_z_index(i, j, k, c.nx, c.ny)];
    divergence[id] = (du + dv + dw) * inv_h;
}

__global__ void grid_advect_scalar_kernel(float* vel_x,
                                          float* vel_y,
                                          float* vel_z,
                                          float* scalar_in,
                                          float* scalar_out,
                                          GridScalarAdvectionConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = c.nx * c.ny * c.nz;
    if (id >= cell_count) return;

    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    const float lx = static_cast<float>(i) + 0.5f;
    const float ly = static_cast<float>(j) + 0.5f;
    const float lz = static_cast<float>(k) + 0.5f;

    const float vx = sample_vel_x(vel_x, lx, ly - 0.5f, lz - 0.5f, c.nx, c.ny, c.nz);
    const float vy = sample_vel_y(vel_y, lx - 0.5f, ly, lz - 0.5f, c.nx, c.ny, c.nz);
    const float vz = sample_vel_z(vel_z, lx - 0.5f, ly - 0.5f, lz, c.nx, c.ny, c.nz);
    const float inv_h = c.voxel_size > 1e-6f ? 1.0f / c.voxel_size : 1.0f;

    const float back_x = lx - vx * c.dt * inv_h;
    const float back_y = ly - vy * c.dt * inv_h;
    const float back_z = lz - vz * c.dt * inv_h;
    scalar_out[id] = sample_cell_centered_open(scalar_in, back_x, back_y, back_z, c.nx, c.ny, c.nz);
}

__global__ void grid_advect_velocity_kernel(float* vel_x,
                                            float* vel_y,
                                            float* vel_z,
                                            float* out_x,
                                            float* out_y,
                                            float* out_z,
                                            GridScalarAdvectionConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int vx_count = (c.nx + 1) * c.ny * c.nz;
    const int vy_count = c.nx * (c.ny + 1) * c.nz;
    const int vz_count = c.nx * c.ny * (c.nz + 1);
    const float inv_h = c.voxel_size > 1e-6f ? 1.0f / c.voxel_size : 1.0f;

    if (id < vx_count) {
        const int plane = (c.nx + 1) * c.ny;
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / (c.nx + 1);
        const int i = rem - j * (c.nx + 1);
        const float lx = static_cast<float>(i);
        const float ly = static_cast<float>(j) + 0.5f;
        const float lz = static_cast<float>(k) + 0.5f;
        const float vx = sample_vel_x(vel_x, lx, ly - 0.5f, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vy = sample_vel_y(vel_y, lx - 0.5f, ly, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vz = sample_vel_z(vel_z, lx - 0.5f, ly - 0.5f, lz, c.nx, c.ny, c.nz);
        const float back_x = lx - vx * c.dt * inv_h;
        const float back_y = ly - vy * c.dt * inv_h;
        const float back_z = lz - vz * c.dt * inv_h;
        out_x[id] = sample_vel_x(vel_x, back_x, back_y - 0.5f, back_z - 0.5f, c.nx, c.ny, c.nz);
    }
    if (id < vy_count) {
        const int plane = c.nx * (c.ny + 1);
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / c.nx;
        const int i = rem - j * c.nx;
        const float lx = static_cast<float>(i) + 0.5f;
        const float ly = static_cast<float>(j);
        const float lz = static_cast<float>(k) + 0.5f;
        const float vx = sample_vel_x(vel_x, lx, ly - 0.5f, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vy = sample_vel_y(vel_y, lx - 0.5f, ly, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vz = sample_vel_z(vel_z, lx - 0.5f, ly - 0.5f, lz, c.nx, c.ny, c.nz);
        const float back_x = lx - vx * c.dt * inv_h;
        const float back_y = ly - vy * c.dt * inv_h;
        const float back_z = lz - vz * c.dt * inv_h;
        out_y[id] = sample_vel_y(vel_y, back_x - 0.5f, back_y, back_z - 0.5f, c.nx, c.ny, c.nz);
    }
    if (id < vz_count) {
        const int plane = c.nx * c.ny;
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / c.nx;
        const int i = rem - j * c.nx;
        const float lx = static_cast<float>(i) + 0.5f;
        const float ly = static_cast<float>(j) + 0.5f;
        const float lz = static_cast<float>(k);
        const float vx = sample_vel_x(vel_x, lx, ly - 0.5f, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vy = sample_vel_y(vel_y, lx - 0.5f, ly, lz - 0.5f, c.nx, c.ny, c.nz);
        const float vz = sample_vel_z(vel_z, lx - 0.5f, ly - 0.5f, lz, c.nx, c.ny, c.nz);
        const float back_x = lx - vx * c.dt * inv_h;
        const float back_y = ly - vy * c.dt * inv_h;
        const float back_z = lz - vz * c.dt * inv_h;
        out_z[id] = sample_vel_z(vel_z, back_x - 0.5f, back_y - 0.5f, back_z, c.nx, c.ny, c.nz);
    }
}

__device__ __forceinline__ float dissipate_clamp_velocity_value(float value,
                                                                 GridVelocityDissipationConstants c) {
    value *= c.factor;
    if (c.max_velocity > 0.0f) {
        value = fminf(fmaxf(value, -c.max_velocity), c.max_velocity);
    }
    return value;
}

__global__ void grid_velocity_dissipate_clamp_kernel(float* vel_x,
                                                     float* vel_y,
                                                     float* vel_z,
                                                     GridVelocityDissipationConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int vx_count = (c.nx + 1) * c.ny * c.nz;
    const int vy_count = c.nx * (c.ny + 1) * c.nz;
    const int vz_count = c.nx * c.ny * (c.nz + 1);

    if (id < vx_count) {
        vel_x[id] = dissipate_clamp_velocity_value(vel_x[id], c);
    }
    if (id < vy_count) {
        vel_y[id] = dissipate_clamp_velocity_value(vel_y[id], c);
    }
    if (id < vz_count) {
        vel_z[id] = dissipate_clamp_velocity_value(vel_z[id], c);
    }
}

__global__ void grid_sor_kernel(float* vel_x,
                                float* vel_y,
                                float* vel_z,
                                float* pressure,
                                float* divergence,
                                GridProjectionConstants c) {
    (void)vel_x;
    (void)vel_y;
    (void)vel_z;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int cell_count = c.nx * c.ny * c.nz;
    if (id >= cell_count) return;

    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    if (((i + j + k) & 1) != c.parity) return;

    const float sum =
        sample_pressure(pressure, i - 1, j, k, c) +
        sample_pressure(pressure, i + 1, j, k, c) +
        sample_pressure(pressure, i, j - 1, k, c) +
        sample_pressure(pressure, i, j + 1, k, c) +
        sample_pressure(pressure, i, j, k - 1, c) +
        sample_pressure(pressure, i, j, k + 1, c);
    const float h = c.voxel_size > 1e-6f ? c.voxel_size : 1.0f;
    const float inv_dt = c.dt > 1e-8f ? 1.0f / c.dt : 0.0f;
    const float rhs = divergence[id] * h * h * inv_dt;
    const float p_gs = (sum - rhs) / 6.0f;
    pressure[id] += c.sor_omega * (p_gs - pressure[id]);
}

__global__ void grid_subtract_gradient_kernel(float* vel_x,
                                              float* vel_y,
                                              float* vel_z,
                                              float* pressure,
                                              float* divergence,
                                              GridProjectionConstants c) {
    (void)divergence;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int vx_count = (c.nx + 1) * c.ny * c.nz;
    const int vy_count = c.nx * (c.ny + 1) * c.nz;
    const int vz_count = c.nx * c.ny * (c.nz + 1);
    const float h = c.voxel_size > 1e-6f ? c.voxel_size : 1.0f;
    const float scale = c.dt / h;

    if (id < vx_count) {
        const int plane = (c.nx + 1) * c.ny;
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / (c.nx + 1);
        const int i = rem - j * (c.nx + 1);
        vel_x[id] -= scale * (sample_pressure(pressure, i, j, k, c) -
                              sample_pressure(pressure, i - 1, j, k, c));
    }
    if (id < vy_count) {
        const int plane = c.nx * (c.ny + 1);
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / c.nx;
        const int i = rem - j * c.nx;
        vel_y[id] -= scale * (sample_pressure(pressure, i, j, k, c) -
                              sample_pressure(pressure, i, j - 1, k, c));
    }
    if (id < vz_count) {
        const int plane = c.nx * c.ny;
        const int k = id / plane;
        const int rem = id - k * plane;
        const int j = rem / c.nx;
        const int i = rem - j * c.nx;
        vel_z[id] -= scale * (sample_pressure(pressure, i, j, k, c) -
                              sample_pressure(pressure, i, j, k - 1, c));
    }
}

// ── APIC G2P gather ──────────────────────────────────────────────────────────
// One thread per particle. Gathers velocity + APIC affine C from the MAC grid.
// Supports FLIP blend via optional pre-projection velocity snapshot (buffers 6-8).
struct FluidG2PConstants {
    int nx, ny, nz;
    int particle_count;
    float origin_x, origin_y, origin_z;
    float voxel_size;
    float flip_blend;
    float apic_blend;
    float internal_friction;
    float max_velocity;
    float dt;
    int has_flip_snapshot; // 1 = scratch_vel_x/y/z contain pre-projection velocities
    // Must mirror host FluidG2PGpuConstants (ParticleSimulation.cpp) field order
    // — cmd.constants is reinterpret_cast to this struct.
    float affine_damping;
    float max_affine;
};

__global__ void fluid_g2p_gather_kernel(const DeviceVec3* positions,
                                         DeviceVec3* velocities,
                                         DeviceAffineC* affine,
                                         const float* vel_x_post,
                                         const float* vel_y_post,
                                         const float* vel_z_post,
                                         const float* vel_x_pre,
                                         const float* vel_y_pre,
                                         const float* vel_z_pre,
                                         FluidG2PConstants c) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= c.particle_count || c.voxel_size <= 1e-6f) return;

    const DeviceVec3 p = positions[id];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) return;

    const float h    = c.voxel_size;
    const float invH = 1.0f / h;
    const float D_inv = 4.0f * invH * invH;

    DeviceVec3   v_new   = {0.0f, 0.0f, 0.0f};
    DeviceVec3   v_pre_g = {0.0f, 0.0f, 0.0f};
    DeviceAffineC C_new;
    C_new.col0 = {0.0f,0.0f,0.0f};
    C_new.col1 = {0.0f,0.0f,0.0f};
    C_new.col2 = {0.0f,0.0f,0.0f};

    for (int comp = 0; comp < 3; ++comp) {
        float gx = (p.x - c.origin_x) * invH;
        float gy = (p.y - c.origin_y) * invH;
        float gz = (p.z - c.origin_z) * invH;
        if (comp == 0)      { gy -= 0.5f; gz -= 0.5f; }
        else if (comp == 1) { gx -= 0.5f; gz -= 0.5f; }
        else                { gx -= 0.5f; gy -= 0.5f; }

        int bx, by, bz;
        float wx[3], wy[3], wz[3];
        quadratic_weights_cuda(gx, bx, wx);
        quadratic_weights_cuda(gy, by, wy);
        quadratic_weights_cuda(gz, bz, wz);

        const int xmax = (comp == 0) ? c.nx     : c.nx - 1;
        const int ymax = (comp == 1) ? c.ny     : c.ny - 1;
        const int zmax = (comp == 2) ? c.nz     : c.nz - 1;

        float v_acc = 0.0f, vp_acc = 0.0f;
        float cx_acc = 0.0f, cy_acc = 0.0f, cz_acc = 0.0f;

        for (int dk = 0; dk < 3; ++dk)
        for (int dj = 0; dj < 3; ++dj)
        for (int di = 0; di < 3; ++di) {
            const int gi = bx + di, gj = by + dj, gk = bz + dk;
            if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;

            const float w = wx[di] * wy[dj] * wz[dk];

            int face_idx;
            if (comp == 0)      face_idx = gi + gj*(c.nx+1) + gk*(c.nx+1)*c.ny;
            else if (comp == 1) face_idx = gi + gj*c.nx      + gk*c.nx*(c.ny+1);
            else                face_idx = gi + gj*c.nx      + gk*c.nx*c.ny;

            const float vn = (comp==0) ? vel_x_post[face_idx]
                           : (comp==1) ? vel_y_post[face_idx]
                                       : vel_z_post[face_idx];
            v_acc += w * vn;

            if (c.has_flip_snapshot) {
                const float vn_pre = (comp==0) ? vel_x_pre[face_idx]
                                   : (comp==1) ? vel_y_pre[face_idx]
                                               : vel_z_pre[face_idx];
                vp_acc += w * vn_pre;
            }

            const float dxw = (static_cast<float>(gi) - gx) * h;
            const float dyw = (static_cast<float>(gj) - gy) * h;
            const float dzw = (static_cast<float>(gk) - gz) * h;
            cx_acc += w * vn * dxw;
            cy_acc += w * vn * dyw;
            cz_acc += w * vn * dzw;
        }

        if (comp == 0) {
            v_new.x = v_acc; v_pre_g.x = vp_acc;
            C_new.col0.x = cx_acc * D_inv;
            C_new.col1.x = cy_acc * D_inv;
            C_new.col2.x = cz_acc * D_inv;
        } else if (comp == 1) {
            v_new.y = v_acc; v_pre_g.y = vp_acc;
            C_new.col0.y = cx_acc * D_inv;
            C_new.col1.y = cy_acc * D_inv;
            C_new.col2.y = cz_acc * D_inv;
        } else {
            v_new.z = v_acc; v_pre_g.z = vp_acc;
            C_new.col0.z = cx_acc * D_inv;
            C_new.col1.z = cy_acc * D_inv;
            C_new.col2.z = cz_acc * D_inv;
        }
    }

    // FLIP/PIC blend
    const float flip = c.has_flip_snapshot ? c.flip_blend : 0.0f;
    DeviceVec3 v_out = v_new;
    if (flip > 0.0f) {
        const DeviceVec3 v_old = velocities[id];
        v_out.x += flip * (v_old.x - v_pre_g.x);
        v_out.y += flip * (v_old.y - v_pre_g.y);
        v_out.z += flip * (v_old.z - v_pre_g.z);
    }

    // Internal friction: v *= exp(-friction * dt)
    if (c.internal_friction > 0.0f && c.dt > 0.0f) {
        const float decay = expf(-c.internal_friction * c.dt);
        v_out.x *= decay; v_out.y *= decay; v_out.z *= decay;
    }

    // Velocity clamp
    if (c.max_velocity > 1e-6f) {
        const float sp2 = v_out.x*v_out.x + v_out.y*v_out.y + v_out.z*v_out.z;
        if (sp2 > c.max_velocity * c.max_velocity) {
            const float s = c.max_velocity * rsqrtf(sp2);
            v_out.x *= s; v_out.y *= s; v_out.z *= s;
        }
    }

    // APIC affine blend + damping, then clamp — parity with the CPU
    // gridToParticle (affine_blend = apic_blend*affine_damping, clampAffine).
    // The clamp is ESSENTIAL: without it C grows unbounded across steps and the
    // next P2G's apic term blows the velocity field up (fluid sprays/vanishes).
    const float ab = c.apic_blend * c.affine_damping;
    C_new.col0.x *= ab; C_new.col0.y *= ab; C_new.col0.z *= ab;
    C_new.col1.x *= ab; C_new.col1.y *= ab; C_new.col1.z *= ab;
    C_new.col2.x *= ab; C_new.col2.y *= ab; C_new.col2.z *= ab;
    const float ma = fmaxf(0.0f, c.max_affine);
    C_new.col0.x = fminf(fmaxf(C_new.col0.x, -ma), ma); C_new.col0.y = fminf(fmaxf(C_new.col0.y, -ma), ma); C_new.col0.z = fminf(fmaxf(C_new.col0.z, -ma), ma);
    C_new.col1.x = fminf(fmaxf(C_new.col1.x, -ma), ma); C_new.col1.y = fminf(fmaxf(C_new.col1.y, -ma), ma); C_new.col1.z = fminf(fmaxf(C_new.col1.z, -ma), ma);
    C_new.col2.x = fminf(fmaxf(C_new.col2.x, -ma), ma); C_new.col2.y = fminf(fmaxf(C_new.col2.y, -ma), ma); C_new.col2.z = fminf(fmaxf(C_new.col2.z, -ma), ma);

    velocities[id] = v_out;
    affine[id]     = C_new;
}

// ── Free-surface SOR pressure projection ─────────────────────────────────────
// fluid_mask: float buffer, > 0.5 = fluid cell, <= 0.5 = air (p = 0 Dirichlet).
// Solid walls are implicit: out-of-bounds grid faces contribute 0 to diagonal
// (closed-wall assumption). For each fluid cell, diagonal = count of in-bounds
// (non-solid) neighbors (fluid + air); only fluid neighbors contribute to sum.
__global__ void fluid_free_surface_sor_kernel(float* pressure,
                                               const float* divergence,
                                               const float* fluid_mask,
                                               GridProjectionConstants c) {
    const int cell_count = c.nx * c.ny * c.nz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;

    if (fluid_mask[id] < 0.5f) {
        pressure[id] = 0.0f;
        return;
    }

    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    if (((i + j + k) & 1) != c.parity) return;

    int diagonal = 0;
    float sum = 0.0f;

#define FS_NEIGHBOR(ni, nj, nk) \
    if ((ni) >= 0 && (ni) < c.nx && (nj) >= 0 && (nj) < c.ny && (nk) >= 0 && (nk) < c.nz) { \
        ++diagonal; \
        const int _nid = (ni) + (nj)*c.nx + (nk)*c.nx*c.ny; \
        if (fluid_mask[_nid] > 0.5f) sum += pressure[_nid]; \
    }
    FS_NEIGHBOR(i-1, j,   k  )
    FS_NEIGHBOR(i+1, j,   k  )
    FS_NEIGHBOR(i,   j-1, k  )
    FS_NEIGHBOR(i,   j+1, k  )
    FS_NEIGHBOR(i,   j,   k-1)
    FS_NEIGHBOR(i,   j,   k+1)
#undef FS_NEIGHBOR

    if (diagonal == 0) return;
    const float h      = c.voxel_size > 1e-6f ? c.voxel_size : 1.0f;
    const float inv_dt = c.dt > 1e-8f ? 1.0f / c.dt : 0.0f;
    const float rhs    = divergence[id] * h * h * inv_dt;
    pressure[id] += c.sor_omega * ((sum - rhs) / static_cast<float>(diagonal) - pressure[id]);
}

// ── MGPCG pressure solve — Layer A: Jacobi-preconditioned CG ─────────────────
// Matrix convention IDENTICAL to fluid_free_surface_sor_kernel so the GPU CG
// solves the exact same free-surface Poisson system as the CPU PCG+MIC(0):
//   fluid_mask>0.5 = fluid ROW; out-of-bounds = solid (no diagonal contribution);
//   air neighbour raises the diagonal but is a Dirichlet p=0 ghost (off-diag 0).
//   A[c,c] = #in-bounds neighbours ; A[c,n] = -1 for in-bounds FLUID neighbour.
//   rhs[c] = divergence[c]*h*h/dt   (== div / pressure_scale, pressure_scale=dt/h^2)
// Non-fluid cells are not rows: their r/z/s/As stay 0 so they drop out of every
// dot product and the SpMV. The CG scalars (alpha/beta) are passed in via
// GridProjectionConstants::sor_omega (this path never runs SOR).

__device__ __forceinline__ int fs_inbounds_neighbor_count(
        int i, int j, int k, const GridProjectionConstants& c) {
    int d = 0;
    if (i - 1 >= 0)   ++d;
    if (i + 1 < c.nx) ++d;
    if (j - 1 >= 0)   ++d;
    if (j + 1 < c.ny) ++d;
    if (k - 1 >= 0)   ++d;
    if (k + 1 < c.nz) ++d;
    return d;
}

// diag[c] = #in-bounds NON-SOLID neighbours for fluid cells, 0 otherwise.
// A solid neighbour (mask < -0.5) is a Neumann wall: no flux, so it must NOT
// raise the diagonal (matches the CPU PCG's "#non-solid neighbours"). Air and
// fluid neighbours both raise it; only fluid neighbours get an off-diagonal
// (handled in spmv via mask > 0.5).
__global__ void fluid_cg_build_diag_kernel(const float* fluid_mask,
                                           float* diag,
                                           GridProjectionConstants c) {
    const int cell_count = c.nx * c.ny * c.nz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    if (fluid_mask[id] < 0.5f) { diag[id] = 0.0f; return; }
    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    int d = 0;
#define CG_DIAG_N(ni, nj, nk) \
    if ((ni) >= 0 && (ni) < c.nx && (nj) >= 0 && (nj) < c.ny && (nk) >= 0 && (nk) < c.nz) { \
        if (fluid_mask[(ni) + (nj) * c.nx + (nk) * c.nx * c.ny] > -0.5f) ++d; \
    }
    CG_DIAG_N(i - 1, j,     k    )
    CG_DIAG_N(i + 1, j,     k    )
    CG_DIAG_N(i,     j - 1, k    )
    CG_DIAG_N(i,     j + 1, k    )
    CG_DIAG_N(i,     j,     k - 1)
    CG_DIAG_N(i,     j,     k + 1)
#undef CG_DIAG_N
    diag[id] = static_cast<float>(d);
}

// r = b = -div*h*h/dt at fluid cells (0 elsewhere); pressure reset to 0.
// SIGN: the SOR fixed point p=(sum-rhs)/diag solves A p = -rhs with
// rhs=div*h*h/dt, i.e. A p = -div*h*h/dt. Since p starts at 0, r0=b must equal
// that RHS — hence the NEGATIVE divergence here. (grid_divergence_kernel emits
// +(div u)/h, so this matches the CPU PCG's b = -(h/dt)(div u) exactly.)
__global__ void fluid_cg_residual_init_kernel(const float* divergence,
                                              const float* fluid_mask,
                                              float* r,
                                              float* pressure,
                                              GridProjectionConstants c) {
    const int cell_count = c.nx * c.ny * c.nz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    pressure[id] = 0.0f;
    if (fluid_mask[id] < 0.5f) { r[id] = 0.0f; return; }
    const float h      = c.voxel_size > 1e-6f ? c.voxel_size : 1.0f;
    const float inv_dt = c.dt > 1e-8f ? 1.0f / c.dt : 0.0f;
    float rhs = -divergence[id] * h * h * inv_dt;
    // Bridson density-targeted projection: over-packed fluid cells (the mask
    // value carries the per-cell particle COUNT) get a synthetic positive RHS
    // so the solver raises pressure there and expels particles — without this
    // FLIP piles collapse on top of each other. Matches the CPU PCG term
    // (+density_correction*(h/dt)*over/target). target = particles_per_cell.
    if (c.density_correction > 0.0f && c.particles_per_cell > 0) {
        const float over = fluid_mask[id] - static_cast<float>(c.particles_per_cell);
        if (over > 0.0f)
            rhs += c.density_correction * (h * inv_dt) * over /
                   static_cast<float>(c.particles_per_cell);
    }
    r[id] = rhs;
}

// As = A*s  (As[c]=0 for non-fluid rows).
__global__ void fluid_cg_spmv_kernel(const float* s,
                                     const float* fluid_mask,
                                     const float* diag,
                                     float* As,
                                     GridProjectionConstants c) {
    const int cell_count = c.nx * c.ny * c.nz;
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    if (fluid_mask[id] < 0.5f) { As[id] = 0.0f; return; }
    const int i = id % c.nx;
    const int j = (id / c.nx) % c.ny;
    const int k = id / (c.nx * c.ny);
    float nsum = 0.0f;
#define CG_SPMV_N(ni, nj, nk) \
    if ((ni) >= 0 && (ni) < c.nx && (nj) >= 0 && (nj) < c.ny && (nk) >= 0 && (nk) < c.nz) { \
        const int _n = (ni) + (nj) * c.nx + (nk) * c.nx * c.ny; \
        if (fluid_mask[_n] > 0.5f) nsum += s[_n]; \
    }
    CG_SPMV_N(i - 1, j,     k    )
    CG_SPMV_N(i + 1, j,     k    )
    CG_SPMV_N(i,     j - 1, k    )
    CG_SPMV_N(i,     j + 1, k    )
    CG_SPMV_N(i,     j,     k - 1)
    CG_SPMV_N(i,     j,     k + 1)
#undef CG_SPMV_N
    As[id] = diag[id] * s[id] - nsum;
}

// z = M^{-1} r  (Jacobi: z = r/diag; non-fluid diag==0 -> z=0).
__global__ void fluid_cg_jacobi_precon_kernel(const float* r,
                                              const float* diag,
                                              float* z,
                                              int cell_count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    const float d = diag[id];
    z[id] = (d > 0.5f) ? (r[id] / d) : 0.0f;
}

__global__ void fluid_cg_copy_kernel(float* dst, const float* src, int cell_count) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    dst[id] = src[id];
}

// y += a*x
__global__ void fluid_cg_axpy_kernel(float* y, const float* x, int cell_count, float a) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    y[id] += a * x[id];
}

// s = z + b*s
__global__ void fluid_cg_zpby_kernel(float* s, const float* z, int cell_count, float b) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= cell_count) return;
    s[id] = z[id] + b * s[id];
}

// partials[blockIdx] = sum_block (x*y) in double; host sums the (few) block
// partials. Double accumulation mirrors the CPU PCG's double dot products —
// essential for CG stability at scale.
__global__ void fluid_cg_dot_kernel(const float* x, const float* y,
                                    double* partials, int cell_count) {
    extern __shared__ double sdata[];
    const int tid = threadIdx.x;
    const int id  = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (id < cell_count) ? (static_cast<double>(x[id]) * static_cast<double>(y[id])) : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

} // namespace

class CudaSimulationComputeBackend final : public ISimulationComputeBackend {
public:
    ~CudaSimulationComputeBackend() override {
        cudaDeviceSynchronize();
        cudaGetLastError();
        for (auto& kv : buffers_) {
            if (kv.second.ptr) cudaFree(kv.second.ptr);
        }
    }

    ComputeBackendType type() const override { return ComputeBackendType::CUDA; }
    const char* name() const override { return "CUDA Simulation Compute"; }

    ComputeBackendCaps caps() const override {
        ComputeBackendCaps c;
        c.available = true;
        c.supports_async = true;
        c.supports_shared_graphics_interop = true;
        c.max_storage_buffer_bytes = static_cast<std::size_t>(2) * 1024 * 1024 * 1024;
        c.max_threads_per_group = 1024;
        return c;
    }

    ComputeBufferHandle createBuffer(const ComputeBufferDesc& desc) override {
        if (desc.size_bytes == 0) return {};
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, desc.size_bytes) != cudaSuccess) {
            cudaGetLastError();
            return {};
        }
        cudaMemset(ptr, 0, desc.size_bytes);
        const uint64_t id = next_id_++;
        buffers_[id] = Buf{ ptr, desc.size_bytes };
        ComputeBufferHandle h;
        h.id = id;
        h.backend = type();
        return h;
    }

    bool destroyBuffer(ComputeBufferHandle handle) override {
        if (handle.backend != type() || !handle.valid()) return false;
        auto it = buffers_.find(handle.id);
        if (it == buffers_.end()) return false;
        cudaDeviceSynchronize();
        cudaGetLastError();
        if (it->second.ptr) cudaFree(it->second.ptr);
        buffers_.erase(it);
        return true;
    }

    bool resizeBuffer(ComputeBufferHandle handle, std::size_t size_bytes) override {
        if (handle.backend != type() || !handle.valid() || size_bytes == 0) return false;
        auto it = buffers_.find(handle.id);
        if (it == buffers_.end()) return false;
        if (it->second.size == size_bytes) return true;
        void* ptr = nullptr;
        if (cudaMalloc(&ptr, size_bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cudaMemset(ptr, 0, size_bytes);
        cudaDeviceSynchronize();
        cudaGetLastError();
        if (it->second.ptr) cudaFree(it->second.ptr);
        it->second.ptr = ptr;
        it->second.size = size_bytes;
        return true;
    }

    std::size_t getBufferSize(ComputeBufferHandle handle) const override {
        if (handle.backend != type() || !handle.valid()) return 0;
        auto it = buffers_.find(handle.id);
        return it == buffers_.end() ? 0 : it->second.size;
    }

    bool uploadBuffer(ComputeBufferHandle handle, const void* data,
                      std::size_t size_bytes, std::size_t dst_offset_bytes) override {
        if (handle.backend != type() || !handle.valid() || (!data && size_bytes > 0)) return false;
        auto it = buffers_.find(handle.id);
        if (it == buffers_.end() || dst_offset_bytes + size_bytes > it->second.size) return false;
        if (size_bytes == 0) return true;
        return cudaMemcpy(static_cast<char*>(it->second.ptr) + dst_offset_bytes, data,
                          size_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool downloadBuffer(ComputeBufferHandle handle, void* data,
                        std::size_t size_bytes, std::size_t src_offset_bytes) const override {
        if (handle.backend != type() || !handle.valid() || (!data && size_bytes > 0)) return false;
        auto it = buffers_.find(handle.id);
        if (it == buffers_.end() || src_offset_bytes + size_bytes > it->second.size) return false;
        if (size_bytes == 0) return true;
        return cudaMemcpy(data, static_cast<const char*>(it->second.ptr) + src_offset_bytes,
                          size_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void* nativeBufferPtr(ComputeBufferHandle handle) const override {
        if (handle.backend != type() || !handle.valid()) return nullptr;
        auto it = buffers_.find(handle.id);
        return it == buffers_.end() ? nullptr : it->second.ptr;
    }

    void synchronize() override { cudaDeviceSynchronize(); }
    bool supportsDispatch() const override { return true; }

    bool dispatch(const ComputeDispatch& cmd) override {
        if (!cmd.kernel) return false;
        const std::string kernel = cmd.kernel;

        if (kernel == "sim_scale") {
            if (cmd.buffer_count < 1 || !cmd.constants || cmd.constants_size < sizeof(ScaleConstants)) {
                return false;
            }
            float* ptr = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            if (!ptr) return false;
            const ScaleConstants* c = static_cast<const ScaleConstants*>(cmd.constants);
            const int threads = 256;
            const int blocks = (c->count + threads - 1) / threads;
            sim_scale_kernel<<<blocks, threads>>>(ptr, c->factor, c->count);
            return cudaGetLastError() == cudaSuccess;
        }

        // Unknown kernel — Phase 3 registers the grid fluid solver stages here.
        if (kernel == "sim_grid_advect_scalar") {
            if (cmd.buffer_count < 5 || !cmd.constants || cmd.constants_size < sizeof(GridScalarAdvectionConstants)) {
                return false;
            }
            float* vel_x = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            float* vel_y = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
            float* vel_z = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
            float* scalar_in = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
            float* scalar_out = static_cast<float*>(nativeBufferPtr(cmd.buffers[4]));
            if (!vel_x || !vel_y || !vel_z || !scalar_in || !scalar_out) {
                return false;
            }
            const GridScalarAdvectionConstants c = *static_cast<const GridScalarAdvectionConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            grid_advect_scalar_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, scalar_in, scalar_out, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_grid_advect_velocity") {
            if (cmd.buffer_count < 6 || !cmd.constants || cmd.constants_size < sizeof(GridScalarAdvectionConstants)) {
                return false;
            }
            float* vel_x = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            float* vel_y = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
            float* vel_z = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
            float* out_x = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
            float* out_y = static_cast<float*>(nativeBufferPtr(cmd.buffers[4]));
            float* out_z = static_cast<float*>(nativeBufferPtr(cmd.buffers[5]));
            if (!vel_x || !vel_y || !vel_z || !out_x || !out_y || !out_z) {
                return false;
            }
            const GridScalarAdvectionConstants c = *static_cast<const GridScalarAdvectionConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            grid_advect_velocity_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, out_x, out_y, out_z, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_grid_velocity_dissipate_clamp") {
            if (cmd.buffer_count < 3 || !cmd.constants || cmd.constants_size < sizeof(GridVelocityDissipationConstants)) {
                return false;
            }
            float* vel_x = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            float* vel_y = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
            float* vel_z = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
            if (!vel_x || !vel_y || !vel_z) {
                return false;
            }
            const GridVelocityDissipationConstants c = *static_cast<const GridVelocityDissipationConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            grid_velocity_dissipate_clamp_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_particle_integrate_forces") {
            if (cmd.buffer_count < 1 || !cmd.constants || cmd.constants_size < sizeof(FluidParticleIntegrateConstants)) {
                return false;
            }
            DeviceVec3* velocities = static_cast<DeviceVec3*>(nativeBufferPtr(cmd.buffers[0]));
            if (!velocities) {
                return false;
            }
            const FluidParticleIntegrateConstants c = *static_cast<const FluidParticleIntegrateConstants*>(cmd.constants);
            if (c.particle_count <= 0 || c.dt <= 0.0f) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            fluid_particle_integrate_forces_kernel<<<grid, block>>>(velocities, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_clear_float") {
            if (cmd.buffer_count < 1 || !cmd.constants || cmd.constants_size < sizeof(FluidP2GConstants)) {
                return false;
            }
            float* values = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            if (!values) {
                return false;
            }
            const FluidP2GConstants c = *static_cast<const FluidP2GConstants*>(cmd.constants);
            const int count = c.component == 0
                ? (c.nx + 1) * c.ny * c.nz
                : (c.component == 1 ? c.nx * (c.ny + 1) * c.nz : c.nx * c.ny * (c.nz + 1));
            if (count <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            fluid_clear_float_kernel<<<grid, block>>>(values, count);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_p2g_scatter" || kernel == "sim_fluid_p2g_normalize") {
            if (cmd.buffer_count < 5 || !cmd.constants || cmd.constants_size < sizeof(FluidP2GConstants)) {
                return false;
            }
            const DeviceVec3* positions = static_cast<const DeviceVec3*>(nativeBufferPtr(cmd.buffers[0]));
            const DeviceVec3* velocities = static_cast<const DeviceVec3*>(nativeBufferPtr(cmd.buffers[1]));
            const DeviceAffineC* affine = static_cast<const DeviceAffineC*>(nativeBufferPtr(cmd.buffers[2]));
            float* velocity_field = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
            float* weight_field = static_cast<float*>(nativeBufferPtr(cmd.buffers[4]));
            if (!positions || !velocities || !affine || !velocity_field || !weight_field) {
                return false;
            }
            const FluidP2GConstants c = *static_cast<const FluidP2GConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0 || c.particle_count <= 0 || c.component < 0 || c.component > 2) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            if (kernel == "sim_fluid_p2g_scatter") {
                fluid_p2g_scatter_kernel<<<grid, block>>>(positions, velocities, affine, velocity_field, weight_field, c);
            } else {
                const int count = c.component == 0
                    ? (c.nx + 1) * c.ny * c.nz
                    : (c.component == 1 ? c.nx * (c.ny + 1) * c.nz : c.nx * c.ny * (c.nz + 1));
                fluid_p2g_normalize_kernel<<<grid, block>>>(velocity_field, weight_field, count);
            }
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_density_clear") {
            if (cmd.buffer_count < 1 || !cmd.constants || cmd.constants_size < sizeof(FluidDensitySplatConstants)) {
                return false;
            }
            float* density = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            if (!density) {
                return false;
            }
            const FluidDensitySplatConstants c = *static_cast<const FluidDensitySplatConstants*>(cmd.constants);
            const int cell_count = c.nx * c.ny * c.nz;
            if (cell_count <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            fluid_density_clear_kernel<<<grid, block>>>(density, cell_count);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_density_splat") {
            if (cmd.buffer_count < 2 || !cmd.constants || cmd.constants_size < sizeof(FluidDensitySplatConstants)) {
                return false;
            }
            const DeviceVec3* positions = static_cast<const DeviceVec3*>(nativeBufferPtr(cmd.buffers[0]));
            float* density = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
            if (!positions || !density) {
                return false;
            }
            const FluidDensitySplatConstants c = *static_cast<const FluidDensitySplatConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0 || c.particle_count <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            fluid_density_splat_kernel<<<grid, block>>>(positions, density, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_grid_divergence" ||
            kernel == "sim_grid_sor" ||
            kernel == "sim_grid_subtract_gradient") {
            if (cmd.buffer_count < 5 || !cmd.constants || cmd.constants_size < sizeof(GridProjectionConstants)) {
                return false;
            }
            float* vel_x = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
            float* vel_y = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
            float* vel_z = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
            float* pressure = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
            float* divergence = static_cast<float*>(nativeBufferPtr(cmd.buffers[4]));
            if (!vel_x || !vel_y || !vel_z || !pressure || !divergence) {
                return false;
            }
            const GridProjectionConstants c = *static_cast<const GridProjectionConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) {
                return false;
            }
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);
            if (kernel == "sim_grid_divergence") {
                grid_divergence_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, pressure, divergence, c);
            } else if (kernel == "sim_grid_sor") {
                grid_sor_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, pressure, divergence, c);
            } else {
                grid_subtract_gradient_kernel<<<grid, block>>>(vel_x, vel_y, vel_z, pressure, divergence, c);
            }
            return cudaGetLastError() == cudaSuccess;
        }

        // ── MGPCG (Layer A: Jacobi-preconditioned CG) pressure kernels ────────
        if (kernel == "sim_fluid_cg_build_diag" ||
            kernel == "sim_fluid_cg_residual_init" ||
            kernel == "sim_fluid_cg_spmv" ||
            kernel == "sim_fluid_cg_jacobi" ||
            kernel == "sim_fluid_cg_copy" ||
            kernel == "sim_fluid_cg_axpy" ||
            kernel == "sim_fluid_cg_zpby" ||
            kernel == "sim_fluid_cg_dot") {
            if (!cmd.constants || cmd.constants_size < sizeof(GridProjectionConstants)) return false;
            const GridProjectionConstants c = *static_cast<const GridProjectionConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) return false;
            const int cell_count = c.nx * c.ny * c.nz;
            const float scalar   = c.sor_omega;   // CG carries alpha/beta here
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1,
                            cmd.groups.groups_y > 0 ? cmd.groups.groups_y : 1,
                            cmd.groups.groups_z > 0 ? cmd.groups.groups_z : 1);

            if (kernel == "sim_fluid_cg_build_diag") {
                if (cmd.buffer_count < 2) return false;          // [mask, diag]
                const float* mask = static_cast<const float*>(nativeBufferPtr(cmd.buffers[0]));
                float* diag       = static_cast<float*>(nativeBufferPtr(cmd.buffers[1]));
                if (!mask || !diag) return false;
                fluid_cg_build_diag_kernel<<<grid, block>>>(mask, diag, c);
            } else if (kernel == "sim_fluid_cg_residual_init") {
                if (cmd.buffer_count < 4) return false;          // [div, mask, r, pressure]
                const float* div  = static_cast<const float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* mask = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                float* r          = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
                float* pressure   = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
                if (!div || !mask || !r || !pressure) return false;
                fluid_cg_residual_init_kernel<<<grid, block>>>(div, mask, r, pressure, c);
            } else if (kernel == "sim_fluid_cg_spmv") {
                if (cmd.buffer_count < 4) return false;          // [s, mask, diag, As]
                const float* s    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* mask = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                const float* diag = static_cast<const float*>(nativeBufferPtr(cmd.buffers[2]));
                float* As         = static_cast<float*>(nativeBufferPtr(cmd.buffers[3]));
                if (!s || !mask || !diag || !As) return false;
                fluid_cg_spmv_kernel<<<grid, block>>>(s, mask, diag, As, c);
            } else if (kernel == "sim_fluid_cg_jacobi") {
                if (cmd.buffer_count < 3) return false;          // [r, diag, z]
                const float* r    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* diag = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                float* z          = static_cast<float*>(nativeBufferPtr(cmd.buffers[2]));
                if (!r || !diag || !z) return false;
                fluid_cg_jacobi_precon_kernel<<<grid, block>>>(r, diag, z, cell_count);
            } else if (kernel == "sim_fluid_cg_copy") {
                if (cmd.buffer_count < 2) return false;          // [dst, src]
                float* dst        = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* src  = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                if (!dst || !src) return false;
                fluid_cg_copy_kernel<<<grid, block>>>(dst, src, cell_count);
            } else if (kernel == "sim_fluid_cg_axpy") {
                if (cmd.buffer_count < 2) return false;          // [y, x]; scalar=a
                float* y          = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* x    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                if (!y || !x) return false;
                fluid_cg_axpy_kernel<<<grid, block>>>(y, x, cell_count, scalar);
            } else if (kernel == "sim_fluid_cg_zpby") {
                if (cmd.buffer_count < 2) return false;          // [s, z]; scalar=b
                float* s          = static_cast<float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* z    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                if (!s || !z) return false;
                fluid_cg_zpby_kernel<<<grid, block>>>(s, z, cell_count, scalar);
            } else { // sim_fluid_cg_dot
                if (cmd.buffer_count < 3) return false;          // [x, y, partials(double*)]
                const float* x    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[0]));
                const float* y    = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
                double* partials  = static_cast<double*>(nativeBufferPtr(cmd.buffers[2]));
                if (!x || !y || !partials) return false;
                fluid_cg_dot_kernel<<<grid, block, block.x * sizeof(double)>>>(x, y, partials, cell_count);
            }
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_g2p") {
            if (cmd.buffer_count < 9 || !cmd.constants || cmd.constants_size < sizeof(FluidG2PConstants)) {
                return false;
            }
            const DeviceVec3*  positions  = static_cast<const DeviceVec3*> (nativeBufferPtr(cmd.buffers[0]));
            DeviceVec3*        velocities = static_cast<DeviceVec3*>       (nativeBufferPtr(cmd.buffers[1]));
            DeviceAffineC*     aff        = static_cast<DeviceAffineC*>    (nativeBufferPtr(cmd.buffers[2]));
            const float*       vx_post    = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[3]));
            const float*       vy_post    = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[4]));
            const float*       vz_post    = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[5]));
            const float*       vx_pre     = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[6]));
            const float*       vy_pre     = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[7]));
            const float*       vz_pre     = static_cast<const float*>      (nativeBufferPtr(cmd.buffers[8]));
            if (!positions || !velocities || !aff || !vx_post || !vy_post || !vz_post) return false;
            const FluidG2PConstants c = *static_cast<const FluidG2PConstants*>(cmd.constants);
            if (c.particle_count <= 0 || c.nx <= 0 || c.ny <= 0 || c.nz <= 0) return false;
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1, 1, 1);
            fluid_g2p_gather_kernel<<<grid, block>>>(positions, velocities, aff,
                                                     vx_post, vy_post, vz_post,
                                                     vx_pre,  vy_pre,  vz_pre, c);
            return cudaGetLastError() == cudaSuccess;
        }

        if (kernel == "sim_fluid_free_surface_sor") {
            if (cmd.buffer_count < 3 || !cmd.constants || cmd.constants_size < sizeof(GridProjectionConstants)) {
                return false;
            }
            float*       pressure   = static_cast<float*>      (nativeBufferPtr(cmd.buffers[0]));
            const float* divergence = static_cast<const float*>(nativeBufferPtr(cmd.buffers[1]));
            const float* fluid_mask = static_cast<const float*>(nativeBufferPtr(cmd.buffers[2]));
            if (!pressure || !divergence || !fluid_mask) return false;
            const GridProjectionConstants c = *static_cast<const GridProjectionConstants*>(cmd.constants);
            if (c.nx <= 0 || c.ny <= 0 || c.nz <= 0) return false;
            const dim3 block(256);
            const dim3 grid(cmd.groups.groups_x > 0 ? cmd.groups.groups_x : 1, 1, 1);
            fluid_free_surface_sor_kernel<<<grid, block>>>(pressure, divergence, fluid_mask, c);
            return cudaGetLastError() == cudaSuccess;
        }

        return false;
    }

private:
    struct Buf {
        void* ptr = nullptr;
        std::size_t size = 0;
    };
    std::unordered_map<uint64_t, Buf> buffers_;
    uint64_t next_id_ = 1;
};

std::unique_ptr<ISimulationComputeBackend> createCudaSimulationComputeBackend() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        return nullptr;
    }
    return std::make_unique<CudaSimulationComputeBackend>();
}

bool selfTestCudaSimulationCompute() {
    auto backend = createCudaSimulationComputeBackend();
    if (!backend) {
        logSimulationComputeWarning("[SimCompute] CUDA backend self-test: no CUDA device");
        return false;
    }

    ComputeBufferDesc desc;
    desc.debug_name = "sim_compute_selftest";
    desc.size_bytes = sizeof(float) * 4;
    ComputeBufferHandle h = backend->createBuffer(desc);
    if (!h.valid()) {
        logSimulationComputeError("[SimCompute] CUDA backend self-test: buffer alloc FAILED");
        return false;
    }

    const float in[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    backend->uploadBuffer(h, in, sizeof(in), 0);

    ScaleConstants constants{ 2.0f, 4 };
    ComputeDispatch cmd;
    cmd.kernel = "sim_scale";
    cmd.buffers = &h;
    cmd.buffer_count = 1;
    cmd.constants = &constants;
    cmd.constants_size = sizeof(constants);
    cmd.groups.groups_x = 1;
    const bool dispatched = backend->dispatch(cmd);
    backend->synchronize();

    float out[4] = { 0, 0, 0, 0 };
    backend->downloadBuffer(h, out, sizeof(out), 0);
    backend->destroyBuffer(h);

    const bool pass = dispatched &&
                      out[0] == 2.0f && out[1] == 4.0f && out[2] == 6.0f && out[3] == 8.0f;
    const std::string message =
        std::string("[SimCompute] CUDA backend self-test: ") +
        (pass ? "OK" : "FAILED") +
        " (out = " + std::to_string(out[0]) + " " +
        std::to_string(out[1]) + " " +
        std::to_string(out[2]) + " " +
        std::to_string(out[3]) + ")";
    if (pass) {
        logSimulationComputeInfo(message);
    } else {
        logSimulationComputeError(message);
    }
    return pass;
}

} // namespace RayTrophiSim
