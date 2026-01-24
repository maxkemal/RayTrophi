#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/gas_kernels.cuh"
#include <algorithm>

namespace FluidSim {

#define BLOCK_SIZE 8

// Helper: Index 3D
__device__ inline int idx(int x, int y, int z, int nx, int ny, int nz) {
    // Clamp coordinates
    if (x < 0) x = 0; if (x >= nx) x = nx - 1;
    if (y < 0) y = 0; if (y >= ny) y = ny - 1;
    if (z < 0) z = 0; if (z >= nz) z = nz - 1;
    return x + nx * (y + ny * z);
}

// ----------------------------------------------------------------------------
// ADVECTION (Semi-Lagrangian)
// ----------------------------------------------------------------------------
__global__ void advect_kernel(int nx, int ny, int nz, float dt,
                              const float* src, float* dst,
                              const float* vx, const float* vy, const float* vz) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int cur_idx = idx(i, j, k, nx, ny, nz);

    // Backtrace
    float x_prev = (float)i - dt * vx[cur_idx];
    float y_prev = (float)j - dt * vy[cur_idx];
    float z_prev = (float)k - dt * vz[cur_idx];

    // Trilinear Interpolation
    int i0 = (int)floorf(x_prev);
    int j0 = (int)floorf(y_prev);
    int k0 = (int)floorf(z_prev);
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    int k1 = k0 + 1;

    float s1 = x_prev - i0; float s0 = 1.0f - s1;
    float t1 = y_prev - j0; float t0 = 1.0f - t1;
    float u1 = z_prev - k0; float u0 = 1.0f - u1;

    // Helper lambda/macro won't work easily here, direct sampling
    float v000 = src[idx(i0, j0, k0, nx, ny, nz)];
    float v100 = src[idx(i1, j0, k0, nx, ny, nz)];
    float v010 = src[idx(i0, j1, k0, nx, ny, nz)];
    float v110 = src[idx(i1, j1, k0, nx, ny, nz)];
    float v001 = src[idx(i0, j0, k1, nx, ny, nz)];
    float v101 = src[idx(i1, j0, k1, nx, ny, nz)];
    float v011 = src[idx(i0, j1, k1, nx, ny, nz)];
    float v111 = src[idx(i1, j1, k1, nx, ny, nz)];

    dst[cur_idx] = 
        u0 * (t0 * (s0 * v000 + s1 * v100) + t1 * (s0 * v010 + s1 * v110)) +
        u1 * (t0 * (s0 * v001 + s1 * v101) + t1 * (s0 * v011 + s1 * v111));
}

// ----------------------------------------------------------------------------
// FORCES (Buoyancy + Wind)
// ----------------------------------------------------------------------------
__global__ void apply_forces_kernel(int nx, int ny, int nz, float dt,
                                    float* vx, float* vy, float* vz,
                                    const float* rho, const float* temp,
                                    float alpha, float beta, float ambient_temp,
                                    float wind_x, float wind_y, float wind_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    // Buoyancy: F = -alpha * rho + beta * (T - Tamb)
    // Only affects Y usually
    float buoy = -alpha * rho[id] + beta * (temp[id] - ambient_temp);
    
    vy[id] += buoy * dt;
    vx[id] += wind_x * dt;
    vy[id] += wind_y * dt;
    vz[id] += wind_z * dt;
}

// ----------------------------------------------------------------------------
// COMBUSTION
// ----------------------------------------------------------------------------
__global__ void combustion_kernel(int nx, int ny, int nz, float dt,
                                  float* rho, float* temp, float* fuel,
                                  float* div, // Use divergence buffer to store expansion temporarily? No, better apply directly to div later.
                                  float ignition_temp, float burn_rate, float heat_release, 
                                  float smoke_gen, float expansion)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    float f = fuel[id];
    float t = temp[id];

    if (f > 0.0f && t > ignition_temp) {
        float burned = f * burn_rate * dt;
        if (burned > f) burned = f;

        fuel[id] -= burned;
        rho[id] += burned * smoke_gen;
        temp[id] += burned * heat_release;
        
        // Expansion is harder in simple grid, usually added as divergence source
        // For simplicity here, we might skip direct expansion or add to divergence buffer if passed
        // For now, let's keep it simple: just heat and smoke.
    }
}

// ----------------------------------------------------------------------------
// DISSIPATION
// ----------------------------------------------------------------------------
__global__ void dissipation_kernel(int nx, int ny, int nz, float dt,
                                   float* rho, float* temp, float* fuel, 
                                   float* vx, float* vy, float* vz,
                                   float k_rho, float k_temp, float k_fuel, float k_vel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    rho[id] *= k_rho;
    temp[id] *= k_temp;
    fuel[id] *= k_fuel;
    
    // Simple decay for now, technically exp(-k*dt) is better but 
    // assuming k is Passed as (1-decay*dt) from CPU
    vx[id] *= k_vel;
    vy[id] *= k_vel;
    vz[id] *= k_vel;
}

// ----------------------------------------------------------------------------
// PROJECT: DIVERGENCE
// ----------------------------------------------------------------------------
__global__ void divergence_kernel(int nx, int ny, int nz,
                                  const float* vx, const float* vy, const float* vz,
                                  float* div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    float x0 = vx[idx(i-1, j, k, nx, ny, nz)];
    float x1 = vx[idx(i+1, j, k, nx, ny, nz)];
    float y0 = vy[idx(i, j-1, k, nx, ny, nz)];
    float y1 = vy[idx(i, j+1, k, nx, ny, nz)];
    float z0 = vz[idx(i, j, k-1, nx, ny, nz)];
    float z1 = vz[idx(i, j, k+1, nx, ny, nz)];

    div[idx(i, j, k, nx, ny, nz)] = -0.5f * ((x1 - x0) + (y1 - y0) + (z1 - z0));
}

// ----------------------------------------------------------------------------
// PROJECT: PRESSURE SOLVE (JACOBI)
// ----------------------------------------------------------------------------
__global__ void pressure_jacobi_kernel(int nx, int ny, int nz,
                                       const float* p_in, float* p_out, const float* div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    float p000 = p_in[idx(i-1, j, k, nx, ny, nz)];
    float p100 = p_in[idx(i+1, j, k, nx, ny, nz)];
    float p010 = p_in[idx(i, j-1, k, nx, ny, nz)];
    float p001 = p_in[idx(i, j-1, k, nx, ny, nz)]; // Typo check: should be p110? No, checking logic
    // Actually full 6 neighbors
    float x0 = p_in[idx(i-1, j, k, nx, ny, nz)];
    float x1 = p_in[idx(i+1, j, k, nx, ny, nz)];
    float y0 = p_in[idx(i, j-1, k, nx, ny, nz)];
    float y1 = p_in[idx(i, j+1, k, nx, ny, nz)];
    float z0 = p_in[idx(i, j, k-1, nx, ny, nz)];
    float z1 = p_in[idx(i, j, k+1, nx, ny, nz)];

    p_out[id] = (div[id] + x0 + x1 + y0 + y1 + z0 + z1) / 6.0f;
}

// ----------------------------------------------------------------------------
// PROJECT: SUBTRACT GRADIENT
// ----------------------------------------------------------------------------
__global__ void subtract_gradient_kernel(int nx, int ny, int nz,
                                         float* vx, float* vy, float* vz,
                                         const float* p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    // Gradient P
    float dx = 0.5f * (p[idx(i+1, j, k, nx, ny, nz)] - p[idx(i-1, j, k, nx, ny, nz)]);
    float dy = 0.5f * (p[idx(i, j+1, k, nx, ny, nz)] - p[idx(i, j-1, k, nx, ny, nz)]);
    float dz = 0.5f * (p[idx(i, j, k+1, nx, ny, nz)] - p[idx(i, j, k-1, nx, ny, nz)]);

    vx[id] -= dx;
    vy[id] -= dy;
    vz[id] -= dz;
}

// ----------------------------------------------------------------------------
// EMITTER
// ----------------------------------------------------------------------------
__global__ void emitter_kernel(int nx, int ny, int nz, float dt,
                               float* rho, float* temp, float* fuel,
                               float* vx, float* vy, float* vz,
                               int shape, float cx, float cy, float cz, 
                               float sx, float sy, float sz,
                               float d_rate, float t_val, float f_rate,
                               float evx, float evy, float evz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    // Convert grid index to world-like space (assuming grid is 0..N)
    // Actually Emitters define position in World Space, but here we work in Grid Space usually.
    // The wrapper should convert world pos to grid coords.
    float x = (float)i;
    float y = (float)j;
    float z = (float)k;

    bool inside = false;

    if (shape == 0) { // Sphere
        float dist2 = (x - cx)*(x - cx) + (y - cy)*(y - cy) + (z - cz)*(z - cz);
        if (dist2 <= sx * sx) inside = true;
    } 
    else if (shape == 1) { // Box
        if (fabs(x - cx) <= sx && fabs(y - cy) <= sy && fabs(z - cz) <= sz) inside = true;
    }

    if (inside) {
        int id = idx(i, j, k, nx, ny, nz);
        rho[id] += d_rate * dt;
        temp[id] = t_val; // Set temp directly or add? Usually 'set' for fire source, 'add' for smoke.
                          // But simpler to just lerp properly. 
                          // For now, let's mix: 
        if (temp[id] < t_val) temp[id] += (t_val - temp[id]) * 0.1f; // Slowly heat up
        
        fuel[id] += f_rate * dt;
        
        vx[id] += evx * dt;
        vy[id] += evy * dt;
        vz[id] += evz * dt;
    }
}


// wrappers
void cuda_init_simulation(int nx, int ny, int nz, 
                          float** d_rho, float** d_temp, float** d_fuel,
                          float** d_vx, float** d_vy, float** d_vz,
                          float** d_p, float** d_div,
                          float** d_vort_x, float** d_vort_y, float** d_vort_z)
{
    size_t size = nx * ny * nz * sizeof(float);
    cudaMalloc(d_rho, size); cudaMemset(*d_rho, 0, size);
    cudaMalloc(d_temp, size); cudaMemset(*d_temp, 0, size);
    cudaMalloc(d_fuel, size); cudaMemset(*d_fuel, 0, size);
    cudaMalloc(d_vx, size); cudaMemset(*d_vx, 0, size);
    cudaMalloc(d_vy, size); cudaMemset(*d_vy, 0, size);
    cudaMalloc(d_vz, size); cudaMemset(*d_vz, 0, size);
    cudaMalloc(d_p, size); cudaMemset(*d_p, 0, size);
    cudaMalloc(d_div, size); cudaMemset(*d_div, 0, size);
    // Vorticity optional? Allocated anyway
    cudaMalloc(d_vort_x, size); 
    cudaMalloc(d_vort_y, size);
    cudaMalloc(d_vort_z, size);
    
    cudaDeviceSynchronize();
}

void cuda_free_simulation(float* d_rho, float* d_temp, float* d_fuel,
                          float* d_vx, float* d_vy, float* d_vz,
                          float* d_p, float* d_div,
                          float* d_vort_x, float* d_vort_y, float* d_vort_z)
{
    if(d_rho) cudaFree(d_rho);
    if(d_temp) cudaFree(d_temp);
    if(d_fuel) cudaFree(d_fuel);
    if(d_vx) cudaFree(d_vx);
    if(d_vy) cudaFree(d_vy);
    if(d_vz) cudaFree(d_vz);
    if(d_p) cudaFree(d_p);
    if(d_div) cudaFree(d_div);
    if(d_vort_x) cudaFree(d_vort_x);
    if(d_vort_y) cudaFree(d_vort_y);
    if(d_vort_z) cudaFree(d_vort_z);
}

void cuda_step_simulation(
    int nx, int ny, int nz, float dt,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_p, float* d_div,
    float* d_vort_x, float* d_vort_y, float* d_vort_z,
    // Settings
    float vorticity_str, float buoy_a, float buoy_b,
    float amb_temp, float k_rho, float k_vel,
    float k_temp, float k_fuel,
    float ign_temp, float burn_rate, float heat_rel, 
    float smoke_gen, float exp,
    int p_iters,
    float wx, float wy, float wz)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y, (nz + block.z - 1)/block.z);

    // 1. Advect Velocity
    // Need Swap Buffers... ah, we need double buffering for stable advection.
    // For simplicity in this restoration, let's allocate temp buffers here or pass them?
    // Passed pointers are "current". We need "next". 
    // Ideally user class manages double buffering.
    // Hack: Malloc temp here (slow) or use ping-pong logic in wrapper?
    // Let's implement simpler "backtrace and write" - requires a copy of old velocity.
    // Ok, let's allocate ONE temp buffer for advection swap.
    // Just advect in-place? No, unstable.
    
    // Proper way: input -> output.
    // Let's allocate temp buffers.
    size_t sz = nx*ny*nz*sizeof(float);
    float *tmp1, *tmp2, *tmp3;
    cudaMalloc(&tmp1, sz);
    cudaMalloc(&tmp2, sz);
    cudaMalloc(&tmp3, sz);
    
    // Advect V -> tmp
    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, tmp1, d_vx, d_vy, d_vz);
    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, tmp2, d_vx, d_vy, d_vz);
    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, tmp3, d_vx, d_vy, d_vz);
    
    // Copy back
    cudaMemcpy(d_vx, tmp1, sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vy, tmp2, sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_vz, tmp3, sz, cudaMemcpyDeviceToDevice);

    // Forces
    apply_forces_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_vy, d_vz, d_rho, d_temp, buoy_a, buoy_b, amb_temp, wx, wy, wz);

    // Combustion
    combustion_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_div, ign_temp, burn_rate, heat_rel, smoke_gen, exp);

    // Project (Pressure)
    divergence_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_div);
    
    // Clear pressure
    cudaMemset(d_p, 0, sz);
    
    // Jacobi
    for(int i=0; i<p_iters; ++i) {
        // Need double buffer for pressure usually, or use Red-Black.
        // Simple Jacobi with temp buffer
        pressure_jacobi_kernel<<<grid, block>>>(nx, ny, nz, d_p, tmp1, d_div);
        cudaMemcpy(d_p, tmp1, sz, cudaMemcpyDeviceToDevice); // Swap
    }
    
    subtract_gradient_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_p);

    // Advect Scalars (Density, Temp, Fuel)
    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, tmp1, d_vx, d_vy, d_vz);
    cudaMemcpy(d_rho, tmp1, sz, cudaMemcpyDeviceToDevice);

    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, tmp2, d_vx, d_vy, d_vz);
    cudaMemcpy(d_temp, tmp2, sz, cudaMemcpyDeviceToDevice);

    advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, tmp3, d_vx, d_vy, d_vz);
    cudaMemcpy(d_fuel, tmp3, sz, cudaMemcpyDeviceToDevice);

    // Dissipation
    // Assuming passed k factors are (1 - decay * dt)
    float factor_rho = 1.0f - k_rho * dt;
    float factor_temp = 1.0f - k_temp * dt;
    float factor_fuel = 1.0f - k_fuel * dt;
    float factor_vel = 1.0f - k_vel * dt;
    
    dissipation_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_vx, d_vy, d_vz, 
                                        factor_rho, factor_temp, factor_fuel, factor_vel);

    cudaFree(tmp1);
    cudaFree(tmp2);
    cudaFree(tmp3);
}

void cuda_apply_emitter(int nx, int ny, int nz, float dt,
                        float* d_rho, float* d_temp, float* d_fuel,
                        float* d_vx, float* d_vy, float* d_vz,
                        int shape, float cx, float cy, float cz,
                        float sx, float sy, float sz,
                        float density_rate, float temp, float fuel_rate,
                        float evx, float evy, float evz)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y, (nz + block.z - 1)/block.z);

    emitter_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_vx, d_vy, d_vz,
                                    shape, cx, cy, cz, sx, sy, sz,
                                    density_rate, temp, fuel_rate, evx, evy, evz);
}

void cuda_upload_data(int nx, int ny, int nz,
                     float* h_rho, float* h_temp, float* h_fuel,
                     float* h_vx, float* h_vy, float* h_vz,
                     float* d_rho, float* d_temp, float* d_fuel,
                     float* d_vx, float* d_vy, float* d_vz)
{
    size_t sz = nx*ny*nz*sizeof(float);
    if(h_rho && d_rho) cudaMemcpy(d_rho, h_rho, sz, cudaMemcpyHostToDevice);
    if(h_temp && d_temp) cudaMemcpy(d_temp, h_temp, sz, cudaMemcpyHostToDevice);
    if(h_fuel && d_fuel) cudaMemcpy(d_fuel, h_fuel, sz, cudaMemcpyHostToDevice);
    if(h_vx && d_vx) cudaMemcpy(d_vx, h_vx, sz, cudaMemcpyHostToDevice);
    if(h_vy && d_vy) cudaMemcpy(d_vy, h_vy, sz, cudaMemcpyHostToDevice);
    if(h_vz && d_vz) cudaMemcpy(d_vz, h_vz, sz, cudaMemcpyHostToDevice);
}

void cuda_download_data(int nx, int ny, int nz,
                       float* d_rho, float* d_temp, float* d_fuel,
                       float* d_vx, float* d_vy, float* d_vz,
                       float* h_rho, float* h_temp, float* h_fuel,
                       float* h_vx, float* h_vy, float* h_vz)
{
    size_t sz = nx*ny*nz*sizeof(float);
    if(h_rho && d_rho) cudaMemcpy(h_rho, d_rho, sz, cudaMemcpyDeviceToHost);
    if(h_temp && d_temp) cudaMemcpy(h_temp, d_temp, sz, cudaMemcpyDeviceToHost);
    if(h_fuel && d_fuel) cudaMemcpy(h_fuel, d_fuel, sz, cudaMemcpyDeviceToHost);
    if(h_vx && d_vx) cudaMemcpy(h_vx, d_vx, sz, cudaMemcpyDeviceToHost);
    if(h_vy && d_vy) cudaMemcpy(h_vy, d_vy, sz, cudaMemcpyDeviceToHost);
    if(h_vz && d_vz) cudaMemcpy(h_vz, d_vz, sz, cudaMemcpyDeviceToHost);
}

} // namespace FluidSim
