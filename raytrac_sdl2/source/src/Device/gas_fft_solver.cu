/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          gas_fft_solver.cu
* Author:        Kemal Demirtas
* Date:          January 2026
* Description:   CUDA implementation of FFT Pressure Solver & Advanced Emitters
* =========================================================================
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "../../include/gas_fft_solver.cuh"
#include <cstdio>

namespace FluidSim {

#define BLOCK_SIZE 8
#define PI 3.14159265359f
#define TWO_PI 6.28318530718f

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

__device__ inline int idx3d(int x, int y, int z, int nx, int ny, int nz) {
    x = max(0, min(x, nx - 1));
    y = max(0, min(y, ny - 1));
    z = max(0, min(z, nz - 1));
    return x + nx * (y + ny * z);
}

// Complex multiplication
__device__ inline cufftComplex complexMul(cufftComplex a, cufftComplex b) {
    cufftComplex result;
    result.x = a.x * b.x - a.y * b.y;
    result.y = a.x * b.y + a.y * b.x;
    return result;
}

// Complex division by real
__device__ inline cufftComplex complexDivReal(cufftComplex a, float b) {
    cufftComplex result;
    if (fabsf(b) > 1e-10f) {
        result.x = a.x / b;
        result.y = a.y / b;
    } else {
        result.x = 0.0f;
        result.y = 0.0f;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT PRESSURE SOLVER KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Compute Laplacian eigenvalues in frequency domain
 * 
 * For discrete Laplacian: eigenvalue = -2 * (cos(2πi/N) + cos(2πj/N) + cos(2πk/N) - 3)
 * Solving ∇²P = div becomes: P̂ = div̂ / eigenvalue
 */
__global__ void computeLaplacianEigenvalues_kernel(
    int nx, int ny, int nz,
    int complex_nx,  // (nx/2 + 1)
    float* eigenvalues
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= complex_nx || j >= ny || k >= nz) return;
    
    int idx = i + complex_nx * (j + ny * k);
    
    // Frequency indices
    float kx = (float)i;
    float ky = (j <= ny/2) ? (float)j : (float)(j - ny);
    float kz = (k <= nz/2) ? (float)k : (float)(k - nz);
    
    // Laplacian eigenvalue for spectral discretization
    // λ = -4π² (kx²/Lx² + ky²/Ly² + kz²/Lz²) for continuous
    // For discrete: λ = 2(cos(2πkx/nx) - 1) + 2(cos(2πky/ny) - 1) + 2(cos(2πkz/nz) - 1)
    float lambda = 2.0f * (cosf(TWO_PI * kx / (float)nx) - 1.0f) +
                   2.0f * (cosf(TWO_PI * ky / (float)ny) - 1.0f) +
                   2.0f * (cosf(TWO_PI * kz / (float)nz) - 1.0f);
    
    // DC component (k=0,0,0) has zero eigenvalue - set to 1 to avoid div by zero
    // Pressure is determined up to a constant anyway
    if (i == 0 && j == 0 && k == 0) {
        lambda = 1.0f;
    }
    
    eigenvalues[idx] = lambda;
}

/**
 * @brief Apply inverse Laplacian in frequency domain
 * Divides each frequency component by its eigenvalue
 */
__global__ void applyInverseLaplacian_kernel(
    int complex_nx, int ny, int nz,
    cufftComplex* spectrum,
    const float* eigenvalues,
    float normalization  // 1.0 / (nx * ny * nz)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= complex_nx || j >= ny || k >= nz) return;
    
    int idx = i + complex_nx * (j + ny * k);
    
    float lambda = eigenvalues[idx];
    
    // P̂ = div̂ / λ
    cufftComplex val = spectrum[idx];
    
    if (fabsf(lambda) > 1e-10f) {
        // Apply normalization from FFT (1/N³) and inverse Laplacian (1/λ)
        float scale = normalization / lambda;
        val.x *= scale;
        val.y *= scale;
    } else {
        val.x = 0.0f;
        val.y = 0.0f;
    }
    
    spectrum[idx] = val;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT PRESSURE SOLVER API
// ═══════════════════════════════════════════════════════════════════════════════

bool initFFTPressureSolver(FFTPressureSolver& solver, int nx, int ny, int nz) {
    solver.nx = nx;
    solver.ny = ny;
    solver.nz = nz;
    
    int complex_nx = nx / 2 + 1;
    solver.complex_size = complex_nx * ny * nz;
    
    // Allocate spectrum buffer
    cudaError_t err = cudaMalloc(&solver.d_spectrum, solver.complex_size * sizeof(cufftComplex));
    if (err != cudaSuccess) {
        printf("[FFT Solver] Failed to allocate spectrum buffer: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    // Allocate eigenvalues buffer
    err = cudaMalloc(&solver.d_eigenvalues, solver.complex_size * sizeof(float));
    if (err != cudaSuccess) {
        printf("[FFT Solver] Failed to allocate eigenvalues buffer: %s\n", cudaGetErrorString(err));
        cudaFree(solver.d_spectrum);
        return false;
    }
    
    // Create cuFFT plans
    cufftResult cufft_err;
    
    // R2C plan (Real to Complex)
    cufft_err = cufftPlan3d(&solver.plan_forward, nz, ny, nx, CUFFT_R2C);
    if (cufft_err != CUFFT_SUCCESS) {
        printf("[FFT Solver] Failed to create forward plan: %d\n", cufft_err);
        cudaFree(solver.d_spectrum);
        cudaFree(solver.d_eigenvalues);
        return false;
    }
    
    // C2R plan (Complex to Real)
    cufft_err = cufftPlan3d(&solver.plan_inverse, nz, ny, nx, CUFFT_C2R);
    if (cufft_err != CUFFT_SUCCESS) {
        printf("[FFT Solver] Failed to create inverse plan: %d\n", cufft_err);
        cufftDestroy(solver.plan_forward);
        cudaFree(solver.d_spectrum);
        cudaFree(solver.d_eigenvalues);
        return false;
    }
    
    // Precompute eigenvalues
    dim3 block(8, 8, 8);
    dim3 grid((complex_nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
    computeLaplacianEigenvalues_kernel<<<grid, block>>>(nx, ny, nz, complex_nx, solver.d_eigenvalues);
    cudaDeviceSynchronize();
    
    solver.initialized = true;
    printf("[FFT Solver] Initialized for %dx%dx%d grid (10-50x faster than iterative!)\n", nx, ny, nz);
    
    return true;
}

void cleanupFFTPressureSolver(FFTPressureSolver& solver) {
    if (!solver.initialized) return;
    
    cufftDestroy(solver.plan_forward);
    cufftDestroy(solver.plan_inverse);
    
    if (solver.d_spectrum) cudaFree(solver.d_spectrum);
    if (solver.d_eigenvalues) cudaFree(solver.d_eigenvalues);
    
    solver.d_spectrum = nullptr;
    solver.d_eigenvalues = nullptr;
    solver.initialized = false;
}

void solvePressureFFT(
    FFTPressureSolver& solver,
    float* d_divergence,
    float* d_pressure
) {
    if (!solver.initialized) {
        printf("[FFT Solver] Error: Solver not initialized!\n");
        return;
    }
    
    int complex_nx = solver.nx / 2 + 1;
    float normalization = 1.0f / (float)(solver.nx * solver.ny * solver.nz);
    
    // Step 1: Forward FFT (divergence -> frequency domain)
    cufftResult err = cufftExecR2C(solver.plan_forward, d_divergence, solver.d_spectrum);
    if (err != CUFFT_SUCCESS) {
        printf("[FFT Solver] Forward FFT failed: %d\n", err);
        return;
    }
    
    // Step 2: Apply inverse Laplacian in frequency domain
    dim3 block(8, 8, 8);
    dim3 grid((complex_nx + 7) / 8, (solver.ny + 7) / 8, (solver.nz + 7) / 8);
    applyInverseLaplacian_kernel<<<grid, block>>>(
        complex_nx, solver.ny, solver.nz,
        solver.d_spectrum, solver.d_eigenvalues, normalization
    );
    
    // Step 3: Inverse FFT (frequency domain -> pressure)
    err = cufftExecC2R(solver.plan_inverse, solver.d_spectrum, d_pressure);
    if (err != CUFFT_SUCCESS) {
        printf("[FFT Solver] Inverse FFT failed: %d\n", err);
        return;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER KERNELS
// ═══════════════════════════════════════════════════════════════════════════════

// Perlin noise for emitter modulation
__device__ float perlinNoise3D_emitter(float x, float y, float z, int seed) {
    // Simple hash-based noise for GPU
    auto hash = [](int x, int y, int z, int seed) -> float {
        unsigned int n = x + y * 57 + z * 131 + seed * 1373;
        n = (n << 13) ^ n;
        unsigned int m = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
        return (float)m / (float)0x7fffffff * 2.0f - 1.0f;
    };
    
    int ix = (int)floorf(x);
    int iy = (int)floorf(y);
    int iz = (int)floorf(z);
    
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;
    
    // Smoothstep
    float u = fx * fx * (3.0f - 2.0f * fx);
    float v = fy * fy * (3.0f - 2.0f * fy);
    float w = fz * fz * (3.0f - 2.0f * fz);
    
    // Trilinear interpolation
    float n000 = hash(ix, iy, iz, seed);
    float n100 = hash(ix+1, iy, iz, seed);
    float n010 = hash(ix, iy+1, iz, seed);
    float n110 = hash(ix+1, iy+1, iz, seed);
    float n001 = hash(ix, iy, iz+1, seed);
    float n101 = hash(ix+1, iy, iz+1, seed);
    float n011 = hash(ix, iy+1, iz+1, seed);
    float n111 = hash(ix+1, iy+1, iz+1, seed);
    
    float n00 = n000 * (1-u) + n100 * u;
    float n10 = n010 * (1-u) + n110 * u;
    float n01 = n001 * (1-u) + n101 * u;
    float n11 = n011 * (1-u) + n111 * u;
    
    float n0 = n00 * (1-v) + n10 * v;
    float n1 = n01 * (1-v) + n11 * v;
    
    return n0 * (1-w) + n1 * w;
}

// Transform world position to emitter local space
__device__ float3 worldToLocal_emitter(const GPUAdvancedEmitter& e, float3 world_pos) {
    float3 local;
    local.x = world_pos.x - e.pos_x;
    local.y = world_pos.y - e.pos_y;
    local.z = world_pos.z - e.pos_z;
    
    // Apply inverse scale
    if (fabsf(e.scale_x) > 0.001f) local.x /= e.scale_x;
    if (fabsf(e.scale_y) > 0.001f) local.y /= e.scale_y;
    if (fabsf(e.scale_z) > 0.001f) local.z /= e.scale_z;
    
    // Apply inverse rotation (simplified - ZYX order)
    if (fabsf(e.rot_z) > 0.001f) {
        float cz = cosf(-e.rot_z), sz = sinf(-e.rot_z);
        float nx = cz * local.x + sz * local.y;
        float ny = -sz * local.x + cz * local.y;
        local.x = nx; local.y = ny;
    }
    if (fabsf(e.rot_y) > 0.001f) {
        float cy = cosf(-e.rot_y), sy = sinf(-e.rot_y);
        float nx = cy * local.x - sy * local.z;
        float nz = sy * local.x + cy * local.z;
        local.x = nx; local.z = nz;
    }
    if (fabsf(e.rot_x) > 0.001f) {
        float cx = cosf(-e.rot_x), sx = sinf(-e.rot_x);
        float ny = cx * local.y + sx * local.z;
        float nz = -sx * local.y + cx * local.z;
        local.y = ny; local.z = nz;
    }
    
    return local;
}

// Check if local position is inside emitter shape
__device__ bool isInsideShape_emitter(const GPUAdvancedEmitter& e, float3 local, float& normalized_dist) {
    float dist_sq, radial_sq;
    
    switch (e.shape) {
        case 0: // Point
            dist_sq = local.x * local.x + local.y * local.y + local.z * local.z;
            normalized_dist = sqrtf(dist_sq) / fmaxf(e.radius, 0.01f);
            return dist_sq < 0.1f * 0.1f;
            
        case 1: // Sphere
            dist_sq = local.x * local.x + local.y * local.y + local.z * local.z;
            normalized_dist = sqrtf(dist_sq) / e.radius;
            return dist_sq <= e.radius * e.radius;
            
        case 2: // Box
            if (fabsf(local.x) > e.size_x || fabsf(local.y) > e.size_y || fabsf(local.z) > e.size_z)
                return false;
            normalized_dist = fmaxf(fmaxf(fabsf(local.x) / e.size_x, fabsf(local.y) / e.size_y), 
                                    fabsf(local.z) / e.size_z);
            return true;
            
        case 3: // Cylinder
            radial_sq = local.x * local.x + local.z * local.z;
            if (radial_sq > e.radius * e.radius || local.y < 0 || local.y > e.height)
                return false;
            normalized_dist = fmaxf(sqrtf(radial_sq) / e.radius, fabsf(local.y / e.height - 0.5f) * 2.0f);
            return true;
            
        case 4: // Cone
            if (local.y < 0 || local.y > e.height) return false;
            {
                float ratio = local.y / e.height;
                float allowed_r = tanf(e.cone_angle * 0.5f) * local.y;
                radial_sq = local.x * local.x + local.z * local.z;
                if (radial_sq > allowed_r * allowed_r) return false;
                normalized_dist = fmaxf(sqrtf(radial_sq) / fmaxf(allowed_r, 0.01f), ratio);
            }
            return true;
            
        case 5: // Disc
            if (fabsf(local.y) > 0.1f) return false;
            radial_sq = local.x * local.x + local.z * local.z;
            if (radial_sq > e.radius * e.radius || radial_sq < e.inner_radius * e.inner_radius)
                return false;
            normalized_dist = sqrtf(radial_sq) / e.radius;
            return true;
            
        default:
            return false;
    }
}

// Calculate falloff factor
__device__ float calculateFalloff_emitter(const GPUAdvancedEmitter& e, float normalized_dist) {
    if (normalized_dist <= e.falloff_start) return 1.0f;
    if (normalized_dist >= e.falloff_end) return 0.0f;
    
    float t = (normalized_dist - e.falloff_start) / (e.falloff_end - e.falloff_start);
    
    switch (e.falloff_type) {
        case 0: // None
            return 1.0f;
        case 1: // Linear
            return 1.0f - t;
        case 2: // Smooth (smoothstep)
            return 1.0f - t * t * (3.0f - 2.0f * t);
        case 3: // Gaussian
            return expf(-4.0f * t * t);
        default:
            return 1.0f - t;
    }
}

// Check if emitter is active at given frame
__device__ bool isActiveAtFrame_emitter(const GPUAdvancedEmitter& e, float frame) {
    if (!e.enabled) return false;
    if (frame < e.start_frame) return false;
    if (e.end_frame >= 0 && frame > e.end_frame) return false;
    
    if (e.emission_mode == 2) { // Pulse
        float elapsed = frame - e.start_frame;
        float cycle_pos = fmodf(elapsed, e.pulse_interval);
        return cycle_pos < e.pulse_duration;
    }
    
    return true;
}

// Generate random velocity within spray cone
__device__ float3 randomVelocity_emitter(const GPUAdvancedEmitter& e, float3 base_vel, int voxel_hash, float time) {
    if (e.spray_cone_angle < 0.001f && fabsf(e.speed_max - e.speed_min) < 0.001f) {
        return base_vel;
    }
    
    // Use voxel hash for consistent randomization
    float rand1 = fmodf((float)voxel_hash * 0.61803398875f + time * 0.1f, 1.0f);
    float rand2 = fmodf((float)voxel_hash * 0.41421356237f + time * 0.13f, 1.0f);
    float rand3 = fmodf((float)voxel_hash * 0.73205080757f + time * 0.17f, 1.0f);
    
    // Speed variation
    float speed = e.speed_min + (e.speed_max - e.speed_min) * rand1;
    
    // Cone deviation
    if (e.spray_cone_angle > 0.001f) {
        float cone_angle = e.spray_cone_angle * rand2;
        float azimuth = TWO_PI * rand3;
        
        // Create perpendicular basis
        float base_len = sqrtf(base_vel.x * base_vel.x + base_vel.y * base_vel.y + base_vel.z * base_vel.z);
        if (base_len < 0.001f) base_len = 1.0f;
        
        float3 dir = make_float3(base_vel.x / base_len, base_vel.y / base_len, base_vel.z / base_len);
        
        // Simple cone deviation
        float3 perp1, perp2;
        if (fabsf(dir.y) < 0.9f) {
            perp1 = make_float3(-dir.z, 0, dir.x);
        } else {
            perp1 = make_float3(0, -dir.z, dir.y);
        }
        float perp1_len = sqrtf(perp1.x * perp1.x + perp1.y * perp1.y + perp1.z * perp1.z);
        perp1.x /= perp1_len; perp1.y /= perp1_len; perp1.z /= perp1_len;
        
        perp2.x = dir.y * perp1.z - dir.z * perp1.y;
        perp2.y = dir.z * perp1.x - dir.x * perp1.z;
        perp2.z = dir.x * perp1.y - dir.y * perp1.x;
        
        float sin_cone = sinf(cone_angle);
        float cos_cone = cosf(cone_angle);
        float sin_az = sinf(azimuth);
        float cos_az = cosf(azimuth);
        
        float3 new_dir;
        new_dir.x = dir.x * cos_cone + (perp1.x * cos_az + perp2.x * sin_az) * sin_cone;
        new_dir.y = dir.y * cos_cone + (perp1.y * cos_az + perp2.y * sin_az) * sin_cone;
        new_dir.z = dir.z * cos_cone + (perp1.z * cos_az + perp2.z * sin_az) * sin_cone;
        
        return make_float3(new_dir.x * speed * base_len, new_dir.y * speed * base_len, new_dir.z * speed * base_len);
    }
    
    return make_float3(base_vel.x * speed, base_vel.y * speed, base_vel.z * speed);
}

// Main advanced emitter kernel
__global__ void advancedEmitter_kernel(
    int nx, int ny, int nz, float dt, float time, float current_frame,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    const GPUAdvancedEmitter* emitters, int emitter_count,
    const float* world_mat
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    int voxel_id = idx3d(i, j, k, nx, ny, nz);
    
    // Compute world position from grid position
    float u = (i + 0.5f) / (float)nx;
    float v = (j + 0.5f) / (float)ny;
    float w = (k + 0.5f) / (float)nz;
    
    // Transform to world space using column-major matrix
    float3 world_pos;
    world_pos.x = world_mat[0] * u + world_mat[4] * v + world_mat[8]  * w + world_mat[12];
    world_pos.y = world_mat[1] * u + world_mat[5] * v + world_mat[9]  * w + world_mat[13];
    world_pos.z = world_mat[2] * u + world_mat[6] * v + world_mat[10] * w + world_mat[14];
    
    // Hash for randomization
    int voxel_hash = i + j * 57 + k * 131;
    
    // Accumulate emission from all emitters
    float total_density = 0.0f;
    float total_temperature = 0.0f;
    float total_fuel = 0.0f;
    float3 total_velocity = make_float3(0, 0, 0);
    float velocity_weight = 0.0f;
    
    for (int e_idx = 0; e_idx < emitter_count; ++e_idx) {
        const GPUAdvancedEmitter& e = emitters[e_idx];
        
        if (!isActiveAtFrame_emitter(e, current_frame)) continue;
        
        float3 local = worldToLocal_emitter(e, world_pos);
        float normalized_dist;
        
        if (!isInsideShape_emitter(e, local, normalized_dist)) continue;
        
        // Calculate falloff
        float falloff = calculateFalloff_emitter(e, normalized_dist);
        
        // Apply noise modulation
        float noise_mult = 1.0f;
        if (e.noise_enabled) {
            float noise_val = perlinNoise3D_emitter(
                world_pos.x * e.noise_frequency + time * e.noise_speed,
                world_pos.y * e.noise_frequency,
                world_pos.z * e.noise_frequency,
                e.noise_seed
            );
            noise_mult = 1.0f + noise_val * e.noise_amplitude;
            noise_mult = fmaxf(0.0f, noise_mult);
        }
        
        float strength = falloff * noise_mult;
        
        // Accumulate density
        if (e.noise_modulate_density || !e.noise_enabled) {
            total_density += e.density_rate * strength;
        }
        
        // Temperature (set, not add)
        if (d_temp[voxel_id] < e.temperature * strength) {
            total_temperature = fmaxf(total_temperature, e.temperature * strength);
        }
        
        // Fuel
        total_fuel += e.fuel_rate * strength;
        
        // Velocity with spray
        float3 base_vel = make_float3(e.vel_x, e.vel_y, e.vel_z);
        float3 emit_vel = randomVelocity_emitter(e, base_vel, voxel_hash + e_idx * 1000, time);
        
        total_velocity.x += emit_vel.x * strength;
        total_velocity.y += emit_vel.y * strength;
        total_velocity.z += emit_vel.z * strength;
        velocity_weight += strength;
    }
    
    // Apply accumulated emission
    if (total_density > 0.0f) {
        d_rho[voxel_id] += total_density * dt;
        d_rho[voxel_id] = fminf(d_rho[voxel_id], 50.0f); // Clamp
    }
    
    if (total_temperature > d_temp[voxel_id]) {
        d_temp[voxel_id] = fminf(total_temperature, 3000.0f);
    }
    
    if (total_fuel > 0.0f) {
        d_fuel[voxel_id] += total_fuel * dt;
        d_fuel[voxel_id] = fminf(d_fuel[voxel_id], 10.0f);
    }
    
    if (velocity_weight > 0.0f) {
        d_vx[voxel_id] += total_velocity.x * dt;
        d_vy[voxel_id] += total_velocity.y * dt;
        d_vz[voxel_id] += total_velocity.z * dt;
        
        // Clamp velocity
        const float MAX_VEL = 500.0f;
        d_vx[voxel_id] = fmaxf(-MAX_VEL, fminf(MAX_VEL, d_vx[voxel_id]));
        d_vy[voxel_id] = fmaxf(-MAX_VEL, fminf(MAX_VEL, d_vy[voxel_id]));
        d_vz[voxel_id] = fmaxf(-MAX_VEL, fminf(MAX_VEL, d_vz[voxel_id]));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER API
// ═══════════════════════════════════════════════════════════════════════════════

void cuda_apply_advanced_emitters(
    int nx, int ny, int nz, float dt,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    const GPUAdvancedEmitter* d_emitters,
    int emitter_count,
    const float* world_mat,
    float current_frame,
    float time
) {
    if (emitter_count <= 0 || !d_emitters) return;
    
    // Upload world matrix
    float* d_world_mat;
    cudaMalloc(&d_world_mat, 16 * sizeof(float));
    cudaMemcpy(d_world_mat, world_mat, 16 * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, (nz + block.z - 1) / block.z);
    
    advancedEmitter_kernel<<<grid, block>>>(
        nx, ny, nz, dt, time, current_frame,
        d_rho, d_temp, d_fuel, d_vx, d_vy, d_vz,
        d_emitters, emitter_count, d_world_mat
    );
    
    cudaFree(d_world_mat);
}

GPUAdvancedEmitter* cuda_upload_advanced_emitters(const GPUAdvancedEmitter* h_emitters, int count) {
    if (count <= 0 || !h_emitters) return nullptr;
    
    GPUAdvancedEmitter* d_emitters = nullptr;
    cudaMalloc(&d_emitters, count * sizeof(GPUAdvancedEmitter));
    cudaMemcpy(d_emitters, h_emitters, count * sizeof(GPUAdvancedEmitter), cudaMemcpyHostToDevice);
    return d_emitters;
}

void cuda_free_advanced_emitters(GPUAdvancedEmitter* d_emitters) {
    if (d_emitters) cudaFree(d_emitters);
}

} // namespace FluidSim
