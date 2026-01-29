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

// Math Helpers
__device__ inline float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator*(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 floor(float3 p) { return make_float3(floorf(p.x), floorf(p.y), floorf(p.z)); }
__device__ inline float fract(float x) { return x - floorf(x); }
__device__ inline float3 fract(float3 p) { return make_float3(p.x - floorf(p.x), p.y - floorf(p.y), p.z - floorf(p.z)); }
__device__ inline float lerp(float a, float b, float t) { return a + t * (b - a); }

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

    float s1 = x_prev - (float)i0; float s0 = 1.0f - s1;
    float t1 = y_prev - (float)j0; float t0 = 1.0f - t1;
    float u1 = z_prev - (float)k0; float u0 = 1.0f - u1;

    float v000 = src[idx(i0, j0, k0, nx, ny, nz)];
    float v100 = src[idx(i1, j0, k0, nx, ny, nz)];
    float v010 = src[idx(i0, j1, k0, nx, ny, nz)];
    float v110 = src[idx(i1, j1, k0, nx, ny, nz)];
    float v001 = src[idx(i0, j0, k1, nx, ny, nz)];
    float v101 = src[idx(i1, j0, k1, nx, ny, nz)];
    float v011 = src[idx(i0, j1, k1, nx, ny, nz)];
    float v111 = src[idx(i1, j1, k1, nx, ny, nz)];

    float val = u0 * (t0 * (s0 * v000 + s1 * v100) + t1 * (s0 * v010 + s1 * v110)) +
                u1 * (t0 * (s0 * v001 + s1 * v101) + t1 * (s0 * v011 + s1 * v111));
    
    // Safety check: Ensure result is finite and non-negative (for most scalars)
    // For velocity (negative allowed), just ensure it's finite.
    if (!isfinite(val)) {
        val = 0.0f;
    }
    dst[cur_idx] = val;
}

// ----------------------------------------------------------------------------
// FORCES V2 (Buoyancy + Gravity + Wind) - All parameters from UI
// ----------------------------------------------------------------------------
__global__ void apply_forces_kernel_v2(int nx, int ny, int nz, float dt,
                                       float* vx, float* vy, float* vz,
                                       const float* rho, const float* temp,
                                       float buoy_density, float buoy_temp, float ambient_temp,
                                       float gravity_x, float gravity_y, float gravity_z,
                                       float wind_x, float wind_y, float wind_z,
                                       float max_velocity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    // Buoyancy: density sinks, hot air rises
    float buoy = buoy_density * rho[id] + buoy_temp * (temp[id] - ambient_temp);
    
    // Clamp velocity using UI parameter
    float next_vx = vx[id] + (gravity_x + wind_x) * dt;
    float next_vy = vy[id] + (gravity_y + wind_y + buoy) * dt;
    float next_vz = vz[id] + (gravity_z + wind_z) * dt;

    if (!isfinite(next_vx)) next_vx = 0.0f;
    if (!isfinite(next_vy)) next_vy = 0.0f;
    if (!isfinite(next_vz)) next_vz = 0.0f;

    vx[id] = fmaxf(-max_velocity, fminf(max_velocity, next_vx));
    vy[id] = fmaxf(-max_velocity, fminf(max_velocity, next_vy));
    vz[id] = fmaxf(-max_velocity, fminf(max_velocity, next_vz));
}

// ----------------------------------------------------------------------------
// FORCES (Buoyancy + Wind) with velocity clamping for stability
// Legacy kernel - kept for backwards compatibility
// ----------------------------------------------------------------------------
#define MAX_VELOCITY 500.0f  // Maximum velocity in grid units per second (increased for proper smoke rise)

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

    // Buoyancy: Match CPU formula exactly
    // CPU: buoyancy_force = buoyancy_temperature * (temp - ambient) + buoyancy_density * density
    // alpha = buoyancy_density (positive value pushes smoke UP when there's density)
    // beta = buoyancy_temperature (positive value pushes hot smoke UP)
    float buoy = alpha * rho[id] + beta * (temp[id] - ambient_temp);
    
    vy[id] += buoy * dt;
    vx[id] += wind_x * dt;
    vy[id] += wind_y * dt;
    vz[id] += wind_z * dt;
    
    // Clamp velocity for stability - ensure no NaNs
    vx[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, isfinite(vx[id]) ? vx[id] : 0.0f));
    vy[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, isfinite(vy[id]) ? vy[id] : 0.0f));
    vz[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, isfinite(vz[id]) ? vz[id] : 0.0f));
}

// ----------------------------------------------------------------------------
// COMBUSTION V2 with UI-configurable limits
// ----------------------------------------------------------------------------
__global__ void combustion_kernel_v2(int nx, int ny, int nz, float dt,
                                     float* rho, float* temp, float* fuel,
                                     float* div,
                                     float ignition_temp, float burn_rate, float heat_release, 
                                     float smoke_gen, float expansion,
                                     float max_density, float max_temperature)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    float f = fuel[id];
    float t = temp[id];
    float rho_val = rho[id];

    if (f > 0.0f && t > ignition_temp) {
        // Burn rate: throttled by density to prevent "density explosion"
        float density_throttle = fmaxf(0.05f, 1.0f - (rho_val / max_density));
        
        float burned = f * burn_rate * dt * density_throttle;
        if (burned > f) burned = f;
        
        // Soft throttle: strongly reduce burn rate as temperature approaches max
        float range = max_temperature - ignition_temp;
        float temp_factor = (max_temperature - t) / (range > 0 ? range : 1.0f);
        float temp_headroom = fmaxf(0.0f, fminf(1.0f, temp_factor));
        burned *= temp_headroom; 
        
        if (burned < 0.00001f) return; 

        fuel[id] -= burned;
        rho[id] += burned * smoke_gen;
        
        // Soft clamp for temperature: exponential approach towards max
        float heat_to_add = burned * heat_release;
        float new_temp = t + heat_to_add;
        float soft_start = max_temperature * 0.90f; // More linear range before softening
        
        if (new_temp > soft_start) {
            float excess = new_temp - soft_start;
            float scale = max_temperature - soft_start;
            new_temp = soft_start + scale * (1.0f - expf(-excess / scale));
        }
        temp[id] = fminf(new_temp, max_temperature);
        
        // Absolute clamp for density
        rho[id] = fminf(rho[id], max_density);
        
        // Expansion: Throttled much more heavily to prevent pressure solver explosion (NaNs)
        // High pressure divergence is the #1 cause of black-spot artifacts
        float expansion_throttle = density_throttle * temp_headroom;
        div[id] += fminf(burned * expansion * expansion_throttle, 100.0f); // Hard limit on divergence per cell
    }
}

// ----------------------------------------------------------------------------
// COMBUSTION with temperature clamping for stability
// Legacy kernel - kept for backwards compatibility
// ----------------------------------------------------------------------------
#define MAX_TEMPERATURE 6000.0f   // Support White-Hot (Blue-White 6000K+)
#define MAX_DENSITY 40.0f         
#define MAX_FUEL 10.0f            

__global__ void combustion_kernel(int nx, int ny, int nz, float dt,
                                  float* rho, float* temp, float* fuel,
                                  float* div,
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
        
        // Clamp values for stability
        rho[id] = fminf(rho[id], MAX_DENSITY);
        temp[id] = fminf(temp[id], MAX_TEMPERATURE);
        
        // Expansion added to divergence to affect pressure projection
        div[id] += burned * expansion;
    }
}

// ----------------------------------------------------------------------------
// DISSIPATION
// ----------------------------------------------------------------------------
__global__ void dissipation_kernel(int nx, int ny, int nz, float dt,
                                   float* rho, float* temp, float* fuel, 
                                   float* vx, float* vy, float* vz,
                                   float k_rho, float k_temp, float k_fuel, float k_vel,
                                   float ambient_temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    // Density: exponential decay with threshold
    rho[id] *= k_rho;
    if (rho[id] < 0.001f) rho[id] = 0.0f;
    
    // Temperature: decay towards ambient (like CPU)
    temp[id] = ambient_temp + (temp[id] - ambient_temp) * k_temp;
    
    // Fuel: exponential decay with threshold
    fuel[id] *= k_fuel;
    if (fuel[id] < 0.001f) fuel[id] = 0.0f;
    
    // Velocity: exponential decay
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

    // div[id] = 0.5 * (dvx/dx + dvy/dy + dvz/dz)
    // combustion might have already added source terms to div
    div[idx(i, j, k, nx, ny, nz)] += -0.5f * ((x1 - x0) + (y1 - y0) + (z1 - z0));
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

    float dx = 0.5f * (p[idx(i+1, j, k, nx, ny, nz)] - p[idx(i-1, j, k, nx, ny, nz)]);
    float dy = 0.5f * (p[idx(i, j+1, k, nx, ny, nz)] - p[idx(i, j-1, k, nx, ny, nz)]);
    float dz = 0.5f * (p[idx(i, j, k+1, nx, ny, nz)] - p[idx(i, j, k-1, nx, ny, nz)]);

    if (!isfinite(dx)) dx = 0.0f;
    if (!isfinite(dy)) dy = 0.0f;
    if (!isfinite(dz)) dz = 0.0f;

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

    float x = (float)i;
    float y = (float)j;
    float z = (float)k;

    bool inside = false;
    if (shape == 0) { // Sphere
        float dist2 = (x - cx)*(x - cx) + (y - cy)*(y - cy) + (z - cz)*(z - cz);
        if (dist2 <= sx * sx) inside = true;
    } 
    else if (shape == 1) { // Box
        if (fabsf(x - cx) <= sx && fabsf(y - cy) <= sy && fabsf(z - cz) <= sz) inside = true;
    }

    if (inside) {
        int id = idx(i, j, k, nx, ny, nz);
        
        // Density injection with clamp
        rho[id] += d_rate * dt;
        rho[id] = fminf(rho[id], MAX_DENSITY);
        
        // Temperature: set to target if below (match CPU behavior)
        // Clamp to prevent explosion in emitter zone
        if (temp[id] < t_val) {
            temp[id] = fminf(t_val, MAX_TEMPERATURE);
        }
        
        // Fuel injection - with improved capping
        fuel[id] += f_rate * dt;
        fuel[id] = fminf(fuel[id], MAX_FUEL);  // Use stricter defined cap
        
        // Velocity injection (accumulate like CPU)
        vx[id] += evx * dt;
        vy[id] += evy * dt;
        vz[id] += evz * dt;
        
        // Clamp velocity in emitter zone for stability
        vx[id] = fmaxf(-MAX_VELOCITY, fminf(vx[id], MAX_VELOCITY));
        vy[id] = fmaxf(-MAX_VELOCITY, fminf(vy[id], MAX_VELOCITY));
        vz[id] = fmaxf(-MAX_VELOCITY, fminf(vz[id], MAX_VELOCITY));
    }
}

// ----------------------------------------------------------------------------
// VORTICITY
// ----------------------------------------------------------------------------
__global__ void compute_vorticity_kernel(int nx, int ny, int nz,
                                       const float* vx, const float* vy, const float* vz,
                                       float* vort_x, float* vort_y, float* vort_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;

    float du_dy = 0.5f * (vx[idx(i, j+1, k, nx, ny, nz)] - vx[idx(i, j-1, k, nx, ny, nz)]);
    float du_dz = 0.5f * (vx[idx(i, j, k+1, nx, ny, nz)] - vx[idx(i, j, k-1, nx, ny, nz)]);
    float dv_dx = 0.5f * (vy[idx(i+1, j, k, nx, ny, nz)] - vy[idx(i-1, j, k, nx, ny, nz)]);
    float dv_dz = 0.5f * (vy[idx(i, j, k+1, nx, ny, nz)] - vy[idx(i, j, k-1, nx, ny, nz)]);
    float dw_dx = 0.5f * (vz[idx(i+1, j, k, nx, ny, nz)] - vz[idx(i-1, j, k, nx, ny, nz)]);
    float dw_dy = 0.5f * (vz[idx(i, j+1, k, nx, ny, nz)] - vz[idx(i, j-1, k, nx, ny, nz)]);

    int id = idx(i, j, k, nx, ny, nz);
    vort_x[id] = dw_dy - dv_dz;
    vort_y[id] = du_dz - dw_dx;
    vort_z[id] = dv_dx - du_dy;
}

__global__ void apply_vorticity_confinement_kernel(int nx, int ny, int nz, float dt,
                                                 float* vx, float* vy, float* vz,
                                                 const float* vort_x, const float* vort_y, const float* vort_z,
                                                 float strength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 2 || i >= nx - 2 || j < 2 || j >= ny - 2 || k < 2 || k >= nz - 2) return;

    auto get_mag = [&](int x, int y, int z) {
        int id = idx(x, y, z, nx, ny, nz);
        float vox = vort_x[id]; float voy = vort_y[id]; float voz = vort_z[id];
        return sqrtf(vox*vox + voy*voy + voz*voz);
    };

    float mag_px = get_mag(i+1, j, k); float mag_mx = get_mag(i-1, j, k);
    float mag_py = get_mag(i, j+1, k); float mag_my = get_mag(i, j-1, k);
    float mag_pz = get_mag(i, j, k+1); float mag_mz = get_mag(i, j, k-1);

    float dx = 0.5f * (mag_px - mag_mx);
    float dy = 0.5f * (mag_py - mag_my);
    float dz = 0.5f * (mag_pz - mag_mz);

    float len = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
    dx /= len; dy /= len; dz /= len;

    int id = idx(i, j, k, nx, ny, nz);
    // Confinement force = cross(grad, vorticity)
    float vox = vort_x[id]; float voy = vort_y[id]; float voz = vort_z[id];

    // MOD: Enhance vorticity at smoke/fire boundaries for more 'mushroom' detail
    float force_x = (dy * voz - dz * voy) * strength;
    float force_y = (dz * vox - dx * voz) * strength;
    float force_z = (dx * voy - dy * vox) * strength;

    vx[id] += force_x * dt;
    vy[id] += force_y * dt;
    vz[id] += force_z * dt;

    // Clamp velocity for stability
    vx[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vx[id]));
    vy[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vy[id]));
    vz[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vz[id]));
}

// ----------------------------------------------------------------------------
// PROJECT: PRESSURE SOLVE (RED-BLACK GAUSS-SEIDEL)
// ----------------------------------------------------------------------------
__global__ void pressure_rbgs_kernel(int nx, int ny, int nz,
                                     float* p, const float* div, int rb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;
    if ((i + j + k) % 2 != rb) return;

    int id = idx(i, j, k, nx, ny, nz);

    float p_l = p[idx(i-1, j, k, nx, ny, nz)];
    float p_r = p[idx(i+1, j, k, nx, ny, nz)];
    float p_b = p[idx(i, j-1, k, nx, ny, nz)];
    float p_t = p[idx(i, j+1, k, nx, ny, nz)];
    float p_back = p[idx(i, j, k-1, nx, ny, nz)];
    float p_front = p[idx(i, j, k+1, nx, ny, nz)];

    float p_gs = (div[id] + p_l + p_r + p_b + p_t + p_back + p_front) / 6.0f;
    if (!isfinite(p_gs)) p_gs = 0.0f;
    p[id] = p_gs;
}

// ----------------------------------------------------------------------------
// PROJECT: PRESSURE SOLVE (SOR - Successive Over-Relaxation)
// Faster convergence than Gauss-Seidel with optimal omega ~1.7
// ----------------------------------------------------------------------------
__global__ void pressure_sor_kernel(int nx, int ny, int nz,
                                    float* p, const float* div, int rb, float omega)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;
    if ((i + j + k) % 2 != rb) return;

    int id = idx(i, j, k, nx, ny, nz);

    float p_l = p[idx(i-1, j, k, nx, ny, nz)];
    float p_r = p[idx(i+1, j, k, nx, ny, nz)];
    float p_b = p[idx(i, j-1, k, nx, ny, nz)];
    float p_t = p[idx(i, j+1, k, nx, ny, nz)];
    float p_back = p[idx(i, j, k-1, nx, ny, nz)];
    float p_front = p[idx(i, j, k+1, nx, ny, nz)];

    // Gauss-Seidel update
    float p_gs_raw = (div[id] + p_l + p_r + p_b + p_t + p_back + p_front) / 6.0f;
    if (!isfinite(p_gs_raw)) p_gs_raw = 0.0f;
    
    // SOR: blend between old value and GS value using omega
    p[id] = (1.0f - omega) * p[id] + omega * p_gs_raw;
}

// ----------------------------------------------------------------------------
// INDUSTRY-STANDARD CURL NOISE (Divergence-Free Turbulence)
// Based on GPU Gems 3 and Houdini/EmberGen implementations
// ----------------------------------------------------------------------------

// Permutation table for gradient noise (first 256 entries, duplicated for wrap)
__constant__ int PERM[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,
    20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,
    230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
    169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,
    147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,
    44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
    112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,
    114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,
    20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,
    230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
    169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,38,
    147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,
    44,154,163,70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
    112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,93,222,
    114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

// 12 gradient vectors for 3D Perlin noise
__constant__ float GRAD3[12][3] = {
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

__device__ __forceinline__ float smootherstep(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float grad3d(int hash, float x, float y, float z) {
    int h = hash & 11;
    return x * GRAD3[h][0] + y * GRAD3[h][1] + z * GRAD3[h][2];
}

// 3D Perlin noise (for curl computation)
__device__ float perlin3d_gpu(float x, float y, float z) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    int Z = (int)floorf(z) & 255;
    
    x -= floorf(x);
    y -= floorf(y);
    z -= floorf(z);
    
    float u = smootherstep(x);
    float v = smootherstep(y);
    float w = smootherstep(z);
    
    int A  = PERM[X] + Y;
    int AA = PERM[A] + Z;
    int AB = PERM[A + 1] + Z;
    int B  = PERM[X + 1] + Y;
    int BA = PERM[B] + Z;
    int BB = PERM[B + 1] + Z;
    
    float g000 = grad3d(PERM[AA], x, y, z);
    float g100 = grad3d(PERM[BA], x - 1, y, z);
    float g010 = grad3d(PERM[AB], x, y - 1, z);
    float g110 = grad3d(PERM[BB], x - 1, y - 1, z);
    float g001 = grad3d(PERM[AA + 1], x, y, z - 1);
    float g101 = grad3d(PERM[BA + 1], x - 1, y, z - 1);
    float g011 = grad3d(PERM[AB + 1], x, y - 1, z - 1);
    float g111 = grad3d(PERM[BB + 1], x - 1, y - 1, z - 1);
    
    return lerp(
        lerp(lerp(g000, g100, u), lerp(g010, g110, u), v),
        lerp(lerp(g001, g101, u), lerp(g011, g111, u), v),
        w
    );
}

// FBM (Fractal Brownian Motion) for multi-scale detail
__device__ float fbm3d_gpu(float x, float y, float z, int octaves, float lacunarity, float persistence) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        value += perlin3d_gpu(x * frequency, y * frequency, z * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / max_value;
}

// Compute curl of noise field (divergence-free!)
__device__ float3 curl3d_gpu(float x, float y, float z, float freq, int octaves, float lacunarity, float persistence) {
    const float eps = 0.01f;
    
    // Sample noise potential at offset positions
    // Curl = ∇ × Ψ where Ψ is the potential field
    float n_y1 = fbm3d_gpu(x, y + eps, z, octaves, lacunarity, persistence);
    float n_y0 = fbm3d_gpu(x, y - eps, z, octaves, lacunarity, persistence);
    float n_z1 = fbm3d_gpu(x, y, z + eps, octaves, lacunarity, persistence);
    float n_z0 = fbm3d_gpu(x, y, z - eps, octaves, lacunarity, persistence);
    float n_x1 = fbm3d_gpu(x + eps, y, z, octaves, lacunarity, persistence);
    float n_x0 = fbm3d_gpu(x - eps, y, z, octaves, lacunarity, persistence);
    
    // For 3D curl, we use three potential functions offset by large values
    // to get uncorrelated components
    float p2_y1 = fbm3d_gpu(x + 123.456f, y + eps, z + 789.012f, octaves, lacunarity, persistence);
    float p2_y0 = fbm3d_gpu(x + 123.456f, y - eps, z + 789.012f, octaves, lacunarity, persistence);
    float p2_x1 = fbm3d_gpu(x + eps + 123.456f, y, z + 789.012f, octaves, lacunarity, persistence);
    float p2_x0 = fbm3d_gpu(x - eps + 123.456f, y, z + 789.012f, octaves, lacunarity, persistence);
    
    float p3_z1 = fbm3d_gpu(x + 456.789f, y + 234.567f, z + eps, octaves, lacunarity, persistence);
    float p3_z0 = fbm3d_gpu(x + 456.789f, y + 234.567f, z - eps, octaves, lacunarity, persistence);
    float p3_x1 = fbm3d_gpu(x + eps + 456.789f, y + 234.567f, z, octaves, lacunarity, persistence);
    float p3_x0 = fbm3d_gpu(x - eps + 456.789f, y + 234.567f, z, octaves, lacunarity, persistence);
    
    float inv_2eps = 1.0f / (2.0f * eps);
    
    // Curl components: ∂Pz/∂y - ∂Py/∂z, ∂Px/∂z - ∂Pz/∂x, ∂Py/∂x - ∂Px/∂y
    float curl_x = (p3_z1 - p3_z0) * inv_2eps - (n_z1 - n_z0) * inv_2eps;
    float curl_y = (n_x1 - n_x0) * inv_2eps - (p3_x1 - p3_x0) * inv_2eps;
    float curl_z = (p2_x1 - p2_x0) * inv_2eps - (p2_y1 - p2_y0) * inv_2eps;
    
    return make_float3(curl_x, curl_y, curl_z);
}

// Industry-standard curl noise turbulence kernel
__global__ void apply_curl_noise_kernel(int nx, int ny, int nz, float dt,
                                        float* vx, float* vy, float* vz,
                                        const float* density,
                                        float strength, float scale, float time,
                                        int octaves, float lacunarity, float persistence)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;

    int id = idx(i, j, k, nx, ny, nz);
    
    // MOD: Apply where there's density OR temperature (explosion core)
    float d = density[id];
    // Also consider temperature impact for turbulence in fire
    // (We reuse the temperature data or just look at density)
    if (d < 0.001f) return; 
    
    // Density-weighted strength with a minimum threshold to allow detail in thin smoke
    float local_strength = strength * fmaxf(d, 0.2f);
    
    // Multi-scale animated position
    float px = (float)i * scale + time * 0.4f;
    float py = (float)j * scale + time * 0.35f;
    float pz = (float)k * scale + time * 0.3f;
    
    // Get divergence-free curl noise
    float3 curl = curl3d_gpu(px, py, pz, scale, octaves, lacunarity, persistence);
    
    // Apply to velocity as a force
    vx[id] += curl.x * local_strength * dt;
    vy[id] += curl.y * local_strength * dt;
    vz[id] += curl.z * local_strength * dt;
    
    // Clamp velocity for stability
    vx[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vx[id]));
    vy[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vy[id]));
    vz[id] = fmaxf(-MAX_VELOCITY, fminf(MAX_VELOCITY, vz[id]));
}

// Legacy simple turbulence (kept for compatibility)
__device__ float hash31(float3 p) {
    p = fract(p * 0.1031f);
    float d = dot(p, make_float3(p.y, p.z, p.x) + make_float3(33.33f, 33.33f, 33.33f));
    p = p + make_float3(d, d, d);
    return fract((p.x + p.y) * p.z);
}

__device__ float noise3d(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    float3 f_sq = make_float3(f.x*f.x*(3.0f-2.0f*f.x), f.y*f.y*(3.0f-2.0f*f.y), f.z*f.z*(3.0f-2.0f*f.z));
    
    float v000 = hash31(i + make_float3(0,0,0));
    float v100 = hash31(i + make_float3(1,0,0));
    float v010 = hash31(i + make_float3(0,1,0));
    float v110 = hash31(i + make_float3(1,1,0));
    float v001 = hash31(i + make_float3(0,0,1));
    float v101 = hash31(i + make_float3(1,0,1));
    float v011 = hash31(i + make_float3(0,1,1));
    float v111 = hash31(i + make_float3(1,1,1));
    
    return lerp(lerp(lerp(v000, v100, f_sq.x), lerp(v010, v110, f_sq.x), f_sq.y),
                lerp(lerp(v001, v101, f_sq.x), lerp(v011, v111, f_sq.x), f_sq.y), f_sq.z);
}

__global__ void apply_turbulence_kernel(int nx, int ny, int nz, float dt,
                                      float* vx, float* vy, float* vz,
                                      float strength, float scale, float time)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    float3 pos = make_float3((float)i, (float)j, (float)k) * scale + make_float3(time, time, time);
    int id = idx(i, j, k, nx, ny, nz);
    
    vx[id] += (noise3d(pos) - 0.5f) * strength * dt;
    vy[id] += (noise3d(pos + make_float3(31, 17, 13)) - 0.5f) * strength * dt;
    vz[id] += (noise3d(pos + make_float3(11, 41, 7)) - 0.5f) * strength * dt;
}

// ----------------------------------------------------------------------------
// ADVECTION (MacCormack)
// ----------------------------------------------------------------------------
__global__ void maccormack_correction_kernel(int nx, int ny, int nz, float dt,
                                          const float* src, const float* fwd, const float* bwd,
                                          float* dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);
    
    float c_fwd = fwd[id];
    float c_src = src[id];
    float c_bwd = bwd[id];
    
    float corrected = c_fwd + 0.5f * (c_src - c_bwd);
    
    // MAC-CORMACK PROTECTION (Clamped MacCormack)
    // Standard advection overshoots/undershoots are the #1 source of simulation NaNs
    // We clamp the corrected value to be reasonably close to the forward sample
    // to prevent ringing artifacts (black cells).
    if (!isfinite(corrected)) {
        corrected = isfinite(c_fwd) ? c_fwd : 0.0f;
    }
    
    // Safety: ensure it doesn't go negative if the original was positive
    if (c_src >= 0.0f && corrected < 0.0f) corrected = 0.0f;
    
    dst[id] = corrected;
}

// ----------------------------------------------------------------------------
// NEW: STABILIZATION / SANITIZATION KERNEL
// ----------------------------------------------------------------------------
__device__ void sanitize_simulation_impl(int nx, int ny, int nz,
                                          float* rho, float* temp, float* fuel,
                                          float max_density, float max_temperature, float ambient_temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz) return;

    int id = idx(i, j, k, nx, ny, nz);

    // Sanitize Density
    float r = rho[id];
    if (!isfinite(r) || r < 0.0f) r = 0.0f;
    rho[id] = fminf(r, max_density);

    // Sanitize Temperature
    float t = temp[id];
    if (!isfinite(t)) t = ambient_temp;
    temp[id] = fmaxf(ambient_temp, fminf(t, max_temperature));

    // Sanitize Fuel
    float f = fuel[id];
    if (!isfinite(f) || f < 0.0f) f = 0.0f;
    fuel[id] = fminf(f, 100.0f);
}

__global__ void sanitize_simulation_kernel(int nx, int ny, int nz,
                                          float* rho, float* temp, float* fuel,
                                          float max_density, float max_temperature, float ambient_temp)
{
    sanitize_simulation_impl(nx, ny, nz, rho, temp, fuel, max_density, max_temperature, ambient_temp);
}

// ----------------------------------------------------------------------------
// WRAPPERS
// ----------------------------------------------------------------------------
void cuda_init_simulation(int nx, int ny, int nz, 
                          float** d_rho, float** d_temp, float** d_fuel,
                          float** d_vx, float** d_vy, float** d_vz,
                          float** d_p, float** d_div,
                          float** d_vort_x, float** d_vort_y, float** d_vort_z)
{
    size_t size = (size_t)nx * ny * nz * sizeof(float);
    cudaMalloc(d_rho, size); cudaMemset(*d_rho, 0, size);
    cudaMalloc(d_temp, size); cudaMemset(*d_temp, 0, size);
    cudaMalloc(d_fuel, size); cudaMemset(*d_fuel, 0, size);
    cudaMalloc(d_vx, size); cudaMemset(*d_vx, 0, size);
    cudaMalloc(d_vy, size); cudaMemset(*d_vy, 0, size);
    cudaMalloc(d_vz, size); cudaMemset(*d_vz, 0, size);
    cudaMalloc(d_p, size); cudaMemset(*d_p, 0, size);
    cudaMalloc(d_div, size); cudaMemset(*d_div, 0, size);
    cudaMalloc(d_vort_x, size); cudaMemset(*d_vort_x, 0, size);
    cudaMalloc(d_vort_y, size); cudaMemset(*d_vort_y, 0, size);
    cudaMalloc(d_vort_z, size); cudaMemset(*d_vort_z, 0, size);
    
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
    float* d_tmp1, float* d_tmp2, float* d_tmp3,
    float vorticity_str, float turbulence_str, float turbulence_scale,
    int advection_mode,
    float buoy_a, float buoy_b,
    float amb_temp, float k_rho, float k_vel,
    float k_temp, float k_fuel,
    float ign_temp, float burn_rate, float heat_rel, 
    float smoke_gen, float exp,
    int p_iters,
    float wx, float wy, float wz)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y, (nz + block.z - 1)/block.z);
    size_t sz = (size_t)nx*ny*nz*sizeof(float);

    // 0. Initial divergence source from zero
    cudaMemset(d_div, 0, sz);

    // === MATCH CPU ORDER EXACTLY ===
    
    // 1. Combustion FIRST (fuel + heat -> fire, generates temperature)
    combustion_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_div, ign_temp, burn_rate, heat_rel, smoke_gen, exp);

    // 2. Apply Forces (buoyancy needs temperature from combustion)
    apply_forces_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_vy, d_vz, d_rho, d_temp, buoy_a, buoy_b, amb_temp, wx, wy, wz);

    // 3. Vorticity Confinement
    if (vorticity_str > 0.001f) {
        compute_vorticity_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_vort_x, d_vort_y, d_vort_z);
        apply_vorticity_confinement_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_vy, d_vz, d_vort_x, d_vort_y, d_vort_z, vorticity_str);
    }

    // 4. Curl Noise Turbulence
    if (turbulence_str > 0.0f) {
        static float time = 0.0f; time += dt;
        apply_curl_noise_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_vy, d_vz, d_rho,
                                                  turbulence_str, turbulence_scale, time,
                                                  3, 2.0f, 0.5f);
    }

    // 5. Advect Velocity
    if (advection_mode == 1) { // MacCormack
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vx, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vy, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vz, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    } else {
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp2, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp3, d_vx, d_vy, d_vz);
        cudaMemcpy(d_vx, d_tmp1, sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vy, d_tmp2, sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vz, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    }

    // 6. Pressure Solve & Project
    divergence_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_div);
    for(int i=0; i<p_iters; ++i) {
        pressure_rbgs_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 0);
        pressure_rbgs_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 1);
    }
    subtract_gradient_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_p);

    if (advection_mode == 1) { // MacCormack
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_rho, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_temp, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_fuel, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    } else {
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_vx, d_vy, d_vz);
        cudaMemcpy(d_rho, d_tmp1, sz, cudaMemcpyDeviceToDevice);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp2, d_vx, d_vy, d_vz);
        cudaMemcpy(d_temp, d_tmp2, sz, cudaMemcpyDeviceToDevice);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp3, d_vx, d_vy, d_vz);
        cudaMemcpy(d_fuel, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    }

    // Use exponential decay (pow) to match CPU behavior exactly
    // CPU: factor = pow(dissipation, dt), dissipation is ~0.98-0.999
    float f_rho = powf(k_rho, dt);
    float f_temp = powf(k_temp, dt);
    float f_fuel = powf(k_fuel, dt);
    float f_vel = powf(k_vel, dt);
    dissipation_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_vx, d_vy, d_vz, f_rho, f_temp, f_fuel, f_vel, amb_temp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEW: cuda_step_simulation_v2 - Uses GPUSimulationParams for all settings
// ═══════════════════════════════════════════════════════════════════════════════
void cuda_step_simulation_v2(
    const GPUSimulationParams& p,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_p, float* d_div,
    float* d_vort_x, float* d_vort_y, float* d_vort_z,
    float* d_tmp1, float* d_tmp2, float* d_tmp3)
{
    const int nx = p.nx, ny = p.ny, nz = p.nz;
    const float dt = p.dt;
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y, (nz + block.z - 1)/block.z);
    size_t sz = (size_t)nx * ny * nz * sizeof(float);

    // 0. Clear divergence
    cudaMemset(d_div, 0, sz);

    // === SIMULATION PIPELINE ===
    
    // 1. Combustion (fuel + heat -> fire, generates temperature)
    combustion_kernel_v2<<<grid, block>>>(
        nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_div,
        p.ignition_temperature, p.burn_rate, p.heat_release,
        p.smoke_generation, p.expansion_strength,
        p.max_density, p.max_temperature
    );

    // 2. Apply Forces (buoyancy + gravity + wind)
    apply_forces_kernel_v2<<<grid, block>>>(
        nx, ny, nz, dt, d_vx, d_vy, d_vz, d_rho, d_temp,
        p.buoyancy_density, p.buoyancy_temperature, p.ambient_temperature,
        p.gravity_x, p.gravity_y, p.gravity_z,
        p.wind_x, p.wind_y, p.wind_z,
        p.max_velocity
    );

    // 3. Vorticity Confinement
    if (p.vorticity_strength > 0.001f) {
        compute_vorticity_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_vort_x, d_vort_y, d_vort_z);
        apply_vorticity_confinement_kernel<<<grid, block>>>(
            nx, ny, nz, dt, d_vx, d_vy, d_vz, d_vort_x, d_vort_y, d_vort_z, p.vorticity_strength
        );
    }

    // 4. Curl Noise Turbulence (using UI parameters for octaves/lacunarity/persistence)
    if (p.turbulence_strength > 0.0f) {
        apply_curl_noise_kernel<<<grid, block>>>(
            nx, ny, nz, dt, d_vx, d_vy, d_vz, d_rho,
            p.turbulence_strength, p.turbulence_scale, p.time,
            p.turbulence_octaves, p.turbulence_lacunarity, p.turbulence_persistence
        );
    }

    // 5. Advect Velocity
    if (p.advection_mode == 1) { // MacCormack
        // Forward advection
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vx, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vy, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_vz, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    } else { // Semi-Lagrangian
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vy, d_tmp2, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vz, d_tmp3, d_vx, d_vy, d_vz);
        cudaMemcpy(d_vx, d_tmp1, sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vy, d_tmp2, sz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vz, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    }

    // 6. Pressure Solve & Project
    divergence_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_div);
    
    // Choose solver based on mode
    // 0 = Gauss-Seidel, 1 = SOR, 2 = Multigrid (TODO), 3 = FFT
    if (p.pressure_solver_mode == 3) {
        // FFT Solver - handled separately via solvePressureFFT in GasSimulator::stepCUDA
        // For now, fall back to SOR if FFT solver not set up at higher level
        // The actual FFT solve is called by the host code before this pipeline
        for (int i = 0; i < p.pressure_iterations; ++i) {
            pressure_sor_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 0, 1.7f);
            pressure_sor_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 1, 1.7f);
        }
    } else if (p.pressure_solver_mode == 1) { // SOR
        for (int i = 0; i < p.pressure_iterations; ++i) {
            pressure_sor_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 0, p.sor_omega);
            pressure_sor_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 1, p.sor_omega);
        }
    } else { // Gauss-Seidel (default, and Multigrid fallback)
        for (int i = 0; i < p.pressure_iterations; ++i) {
            pressure_rbgs_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 0);
            pressure_rbgs_kernel<<<grid, block>>>(nx, ny, nz, d_p, d_div, 1);
        }
    }
    subtract_gradient_kernel<<<grid, block>>>(nx, ny, nz, d_vx, d_vy, d_vz, d_p);

    // 7. Advect Scalars
    if (p.advection_mode == 1) { // MacCormack
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_rho, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_temp, d_tmp3, sz, cudaMemcpyDeviceToDevice);

        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp1, d_vx, d_vy, d_vz);
        advect_kernel<<<grid, block>>>(nx, ny, nz, -dt, d_tmp1, d_tmp2, d_vx, d_vy, d_vz);
        maccormack_correction_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp1, d_tmp2, d_tmp3);
        cudaMemcpy(d_fuel, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    } else {
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_rho, d_tmp1, d_vx, d_vy, d_vz);
        cudaMemcpy(d_rho, d_tmp1, sz, cudaMemcpyDeviceToDevice);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_temp, d_tmp2, d_vx, d_vy, d_vz);
        cudaMemcpy(d_temp, d_tmp2, sz, cudaMemcpyDeviceToDevice);
        advect_kernel<<<grid, block>>>(nx, ny, nz, dt, d_fuel, d_tmp3, d_vx, d_vy, d_vz);
        cudaMemcpy(d_fuel, d_tmp3, sz, cudaMemcpyDeviceToDevice);
    }

    // 8. Dissipation (already computed as pow(rate, dt) in params)
    dissipation_kernel<<<grid, block>>>(
        nx, ny, nz, dt, d_rho, d_temp, d_fuel, d_vx, d_vy, d_vz,
        p.density_dissipation, p.temperature_dissipation, p.fuel_dissipation,
        p.velocity_dissipation, p.ambient_temperature
    );

    // 9. Sanitize / Stabilize (Final pass to scrub NaNs and enforce UI limits)
    sanitize_simulation_kernel<<<grid, block>>>(
        nx, ny, nz, d_rho, d_temp, d_fuel,
        p.max_density, p.max_temperature, p.ambient_temperature
    );
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
    size_t sz = (size_t)nx*ny*nz*sizeof(float);
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
    size_t sz = (size_t)nx*ny*nz*sizeof(float);
    if(h_rho && d_rho) cudaMemcpy(h_rho, d_rho, sz, cudaMemcpyDeviceToHost);
    if(h_temp && d_temp) cudaMemcpy(h_temp, d_temp, sz, cudaMemcpyDeviceToHost);
    if(h_fuel && d_fuel) cudaMemcpy(h_fuel, d_fuel, sz, cudaMemcpyDeviceToHost);
    if(h_vx && d_vx) cudaMemcpy(h_vx, d_vx, sz, cudaMemcpyDeviceToHost);
    if(h_vy && d_vy) cudaMemcpy(h_vy, d_vy, sz, cudaMemcpyDeviceToHost);
    if(h_vz && d_vz) cudaMemcpy(h_vz, d_vz, sz, cudaMemcpyDeviceToHost);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD GPU IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

// Device function: Transform point from normalized [0,1] to world space using column-major 4x4 matrix
__device__ inline float3 transform_point(const float* mat, float3 p) {
    // Column-major: mat[0-3] = col0, mat[4-7] = col1, etc.
    float3 result;
    result.x = mat[0] * p.x + mat[4] * p.y + mat[8]  * p.z + mat[12];
    result.y = mat[1] * p.x + mat[5] * p.y + mat[9]  * p.z + mat[13];
    result.z = mat[2] * p.x + mat[6] * p.y + mat[10] * p.z + mat[14];
    return result;
}

// Device function: Compute local position relative to force field
__device__ inline float3 world_to_local_ff(const GPUForceField& ff, float3 world_pos) {
    // Simple inverse: translate then rotate (ignore scale for now)
    float3 local = world_pos - make_float3(ff.pos_x, ff.pos_y, ff.pos_z);
    
    // Apply inverse rotation (transpose of rotation matrix for orthogonal)
    float cx = cosf(-ff.rot_x), sx = sinf(-ff.rot_x);
    float cy = cosf(-ff.rot_y), sy = sinf(-ff.rot_y);
    float cz = cosf(-ff.rot_z), sz = sinf(-ff.rot_z);
    
    // Inverse ZYX rotation
    float3 p1;
    p1.x = cz * local.x + sz * local.y;
    p1.y = -sz * local.x + cz * local.y;
    p1.z = local.z;
    
    float3 p2;
    p2.x = cy * p1.x - sy * p1.z;
    p2.y = p1.y;
    p2.z = sy * p1.x + cy * p1.z;
    
    float3 p3;
    p3.x = p2.x;
    p3.y = cx * p2.y + sx * p2.z;
    p3.z = -sx * p2.y + cx * p2.z;
    
    return p3;
}

// Device function: Check if point is inside force field influence zone
__device__ inline bool is_inside_ff(const GPUForceField& ff, float3 local) {
    if (ff.shape == 0) return true; // Infinite
    
    float dist = sqrtf(local.x * local.x + local.y * local.y + local.z * local.z);
    
    switch (ff.shape) {
        case 1: // Sphere
            return dist <= ff.falloff_radius;
        case 2: // Box
            return fabsf(local.x) <= ff.falloff_radius &&
                   fabsf(local.y) <= ff.falloff_radius &&
                   fabsf(local.z) <= ff.falloff_radius;
        case 3: { // Cylinder
            float radial = sqrtf(local.x * local.x + local.z * local.z);
            return radial <= ff.falloff_radius && fabsf(local.y) <= ff.falloff_radius;
        }
        case 4: { // Cone
            if (local.y < 0 || local.y > ff.falloff_radius) return false;
            float ratio = local.y / ff.falloff_radius;
            float allowed = ff.falloff_radius * ratio;
            float radial = sqrtf(local.x * local.x + local.z * local.z);
            return radial <= allowed;
        }
    }
    return true;
}

// Device function: Calculate falloff factor
__device__ inline float calculate_falloff_ff(const GPUForceField& ff, float dist) {
    if (dist <= ff.inner_radius) return 1.0f;
    if (dist >= ff.falloff_radius) return 0.0f;
    
    float t = (dist - ff.inner_radius) / (ff.falloff_radius - ff.inner_radius);
    
    switch (ff.falloff_type) {
        case 0: return 1.0f; // None
        case 1: return 1.0f - t; // Linear
        case 2: return 1.0f - t * t * (3.0f - 2.0f * t); // Smooth (smoothstep)
        case 3: return sqrtf(1.0f - t * t); // Sphere
        case 4: { // InverseSquare
            float r = ff.inner_radius + t * (ff.falloff_radius - ff.inner_radius);
            if (r < 0.01f) r = 0.01f;
            float ref = ff.inner_radius > 0.01f ? ff.inner_radius : 0.01f;
            return (ref * ref) / (r * r);
        }
        case 5: return expf(-3.0f * t); // Exponential
    }
    return 1.0f - t;
}

// Device function: Evaluate single force field at world position
__device__ inline float3 evaluate_force_field(const GPUForceField& ff, float3 world_pos, float time) {
    if (!ff.enabled) return make_float3(0, 0, 0);
    
    float3 local = world_to_local_ff(ff, world_pos);
    if (!is_inside_ff(ff, local)) return make_float3(0, 0, 0);
    
    float dist = sqrtf(local.x * local.x + local.y * local.y + local.z * local.z);
    float falloff = (ff.shape != 0) ? calculate_falloff_ff(ff, dist) : 1.0f;
    
    float3 force = make_float3(0, 0, 0);
    
    switch (ff.type) {
        case 0: // Wind
            force = make_float3(ff.dir_x, ff.dir_y, ff.dir_z) * ff.strength;
            break;
            
        case 1: // Gravity
            force = make_float3(ff.dir_x, ff.dir_y, ff.dir_z) * ff.strength;
            break;
            
        case 2: { // Attractor
            if (dist > 0.01f) {
                float3 dir = make_float3(-local.x, -local.y, -local.z) * (1.0f / dist);
                force = dir * ff.strength;
            }
            break;
        }
        
        case 3: { // Repeller
            if (dist > 0.01f) {
                float3 dir = make_float3(local.x, local.y, local.z) * (1.0f / dist);
                force = dir * ff.strength;
            }
            break;
        }
        
        case 4: { // Vortex
            // Cross product of axis and position gives tangent direction
            float3 axis = make_float3(ff.axis_x, ff.axis_y, ff.axis_z);
            float3 tangent;
            tangent.x = axis.y * local.z - axis.z * local.y;
            tangent.y = axis.z * local.x - axis.x * local.z;
            tangent.z = axis.x * local.y - axis.y * local.x;
            
            float tang_len = sqrtf(tangent.x * tangent.x + tangent.y * tangent.y + tangent.z * tangent.z);
            if (tang_len > 0.01f) {
                tangent = tangent * (1.0f / tang_len);
                force = tangent * ff.strength;
                
                // Add inward force for spiral
                if (dist > 0.01f && ff.inward_force != 0.0f) {
                    float3 inward = make_float3(-local.x, -local.y, -local.z) * (1.0f / dist);
                    force = force + inward * ff.inward_force;
                }
            }
            break;
        }
        
        case 5: { // Turbulence (simplified noise)
            // Use world position for noise sampling
            float freq = ff.noise_frequency;
            float amp = ff.noise_amplitude * ff.strength;
            float t = time * ff.noise_speed;
            
            // Simple pseudo-random based on position (not true Perlin, but fast)
            float nx = sinf(world_pos.x * freq + t) * cosf(world_pos.z * freq * 0.7f + t * 0.5f);
            float ny = sinf(world_pos.y * freq * 1.3f + t * 0.8f) * cosf(world_pos.x * freq * 0.9f + t);
            float nz = sinf(world_pos.z * freq * 1.1f + t * 1.2f) * cosf(world_pos.y * freq * 0.8f + t * 0.7f);
            
            force = make_float3(nx, ny, nz) * amp;
            break;
        }
        
        case 7: { // Drag
            // Drag is applied separately since it needs velocity
            // This case handled in kernel
            break;
        }
    }
    
    return force * falloff;
}

// Kernel: Apply force fields to velocity components
__global__ void force_field_kernel(
    int nx, int ny, int nz, float dt,
    float* vx, float* vy, float* vz,
    const float* world_mat,
    const GPUForceField* force_fields,
    int num_fields, float time)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    // Compute normalized position [0,1]
    float3 pos_norm = make_float3(
        (i + 0.5f) / (float)nx,
        (j + 0.5f) / (float)ny,
        (k + 0.5f) / (float)nz
    );
    
    // Transform to world space
    float3 world_pos = transform_point(world_mat, pos_norm);
    
    // Accumulate forces from all fields
    float3 total_force = make_float3(0, 0, 0);
    
    for (int f = 0; f < num_fields; ++f) {
        float3 force = evaluate_force_field(force_fields[f], world_pos, time);
        total_force = total_force + force;
    }
    
    // Apply to velocity
    int id = idx(i, j, k, nx, ny, nz);
    vx[id] += total_force.x * dt;
    vy[id] += total_force.y * dt;
    vz[id] += total_force.z * dt;
}

// Host wrapper functions
void cuda_apply_force_fields(
    int nx, int ny, int nz, float dt,
    float* d_vx, float* d_vy, float* d_vz,
    const float* world_mat,
    const GPUForceField* d_force_fields,
    int num_fields,
    float time)
{
    if (num_fields <= 0 || !d_force_fields) return;
    
    // Upload world matrix to GPU
    float* d_world_mat;
    cudaMalloc(&d_world_mat, 16 * sizeof(float));
    cudaMemcpy(d_world_mat, world_mat, 16 * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y, (nz + block.z - 1)/block.z);
    
    force_field_kernel<<<grid, block>>>(nx, ny, nz, dt, d_vx, d_vy, d_vz, 
                                        d_world_mat, d_force_fields, num_fields, time);
    
    // DEBUG: Check for kernel errors (remove after debugging)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA ERROR] force_field_kernel: %s\n", cudaGetErrorString(err));
    }
    
    cudaFree(d_world_mat);
}

GPUForceField* cuda_upload_force_fields(const GPUForceField* h_fields, int count) {
    if (count <= 0 || !h_fields) return nullptr;
    
    GPUForceField* d_fields = nullptr;
    cudaMalloc(&d_fields, count * sizeof(GPUForceField));
    cudaMemcpy(d_fields, h_fields, count * sizeof(GPUForceField), cudaMemcpyHostToDevice);
    return d_fields;
}

void cuda_free_force_fields(GPUForceField* d_fields) {
    if (d_fields) cudaFree(d_fields);
}

} // namespace FluidSim
