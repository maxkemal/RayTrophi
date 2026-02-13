#pragma once
// ═══════════════════════════════════════════════════════════════════════════════
// FFT OCEAN - Tessendorf Algorithm for Film-Quality Water
// ═══════════════════════════════════════════════════════════════════════════════
// Based on Jerry Tessendorf's "Simulating Ocean Water" (SIGGRAPH 2001)
// Uses CUDA FFT (cuFFT) for GPU-accelerated ocean simulation

#ifndef __CUDACC__
#include <cmath>
#include <complex>
#endif

#include <cuda_runtime.h>
#include <cuComplex.h>

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

#define FFT_GRAVITY 9.81f
#define FFT_PI 3.14159265359f
#define FFT_TWO_PI 6.28318530718f

// ═══════════════════════════════════════════════════════════════════════════════
// OCEAN PARAMETERS STRUCT
// ═══════════════════════════════════════════════════════════════════════════════

struct FFTOceanParams {
    // Grid settings
    int resolution = 256;          // FFT grid size (power of 2: 64, 128, 256, 512)
    float ocean_size = 100.0f;     // World space size covered by the grid
    
    // Wind settings (Phillips spectrum)
    float wind_speed = 10.0f;      // Wind speed in m/s
    float wind_direction = 0.0f;   // Wind direction in radians
    float wind_dependency = 0.07f; // Directional spreading (0 = omnidirectional, 1 = unidirectional)
    
    // Wave settings
    float amplitude = 0.0002f;     // Wave amplitude multiplier (Phillips A parameter)
    float choppiness = 1.0f;       // Horizontal displacement strength
    float small_wave_cutoff = 0.001f; // Suppress small wavelengths (L = V^2/g * cutoff)
    
    // Animation
    float time_scale = 1.0f;       // Animation speed multiplier
    
    // Derived values (computed on init)
    float patch_length = 100.0f;   // = ocean_size
    float dk = 0.0f;               // = 2*PI / patch_length
    
    // Helper to compute derived values
    __host__ void computeDerived() {
        patch_length = ocean_size;
        dk = FFT_TWO_PI / patch_length;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// FFT OCEAN STATE (GPU Resources)
// ═══════════════════════════════════════════════════════════════════════════════

struct FFTOceanState {
    // Spectrum textures (complex)
    cuFloatComplex* d_h0;          // Initial spectrum h̃₀(k) 
    cuFloatComplex* d_h0_conj;     // Conjugate h̃₀*(-k)
    cuFloatComplex* d_ht;          // Time-evolved spectrum h̃(k,t)
    cuFloatComplex* d_displacement_x; // Choppy X displacement spectrum
    cuFloatComplex* d_displacement_z; // Choppy Z displacement spectrum
    
    // Output textures (real)
    float* d_height;               // Height field h(x,t)
    float* d_displacement_x_real;  // X displacement
    float* d_displacement_z_real;  // Z displacement
    float2* d_normal;              // Normal (X, Z) packed for texture sampling
    
    // CUDA Arrays for proper 2D texturing (linear filtering, wrapping)
    cudaArray_t height_array = nullptr;
    cudaArray_t normal_array = nullptr;
    
    // cuFFT plans
    void* fft_plan;                // cufftHandle (opaque to avoid header dependency)
    
    // State tracking
    bool initialized = false;
    int current_resolution = 0;
    float last_update_time = -1.0f;
    FFTOceanParams cached_params;
    
    // CUDA texture objects for shader access
    cudaTextureObject_t tex_height = 0;
    cudaTextureObject_t tex_normal = 0;
    cudaTextureObject_t tex_displacement = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// PHILLIPS SPECTRUM
// ═══════════════════════════════════════════════════════════════════════════════
// The Phillips spectrum models the energy distribution of ocean waves based on 
// wind speed and direction. It's the standard for realistic ocean simulation.

__device__ __host__ inline float phillipsSpectrum(
    float kx, float ky,
    float wind_speed,
    float wind_dir_x, float wind_dir_y,
    float amplitude,
    float small_wave_cutoff,
    float wind_dependency
) {
    float k_len = sqrtf(kx * kx + ky * ky);
    if (k_len < 0.0001f) return 0.0f;
    
    // Largest possible wave from wind (L = V^2 / g)
    float L = wind_speed * wind_speed / FFT_GRAVITY;
    
    // Phillips spectrum formula
    float k2 = k_len * k_len;
    float k4 = k2 * k2;
    float L2 = L * L;
    
    // Exponential decay for waves larger than wind can support
    float phillips = amplitude * expf(-1.0f / (k2 * L2)) / k4;
    
    // Suppress small waves (numerical stability)
    float l2 = L2 * small_wave_cutoff * small_wave_cutoff;
    phillips *= expf(-k2 * l2);
    
    // Directional spreading: waves align with wind direction
    float kx_norm = kx / k_len;
    float ky_norm = ky / k_len;
    float k_dot_w = kx_norm * wind_dir_x + ky_norm * wind_dir_y;
    
    // Cosine-squared directional spreading with dependency control
    float directional = k_dot_w * k_dot_w;
    
    // Suppress waves moving against the wind
    if (k_dot_w < 0.0f) {
        directional *= (1.0f - wind_dependency);
    }
    
    phillips *= directional;
    
    return phillips;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISPERSION RELATION
// ═══════════════════════════════════════════════════════════════════════════════
// ω(k) = sqrt(g * |k|) for deep water waves

__device__ __host__ inline float dispersion(float k_len) {
    return sqrtf(FFT_GRAVITY * k_len);
}

// ═══════════════════════════════════════════════════════════════════════════════
// GAUSSIAN RANDOM (for spectrum initialization)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __host__ inline void boxMullerGaussian(
    float u1, float u2,
    float& gauss1, float& gauss2
) {
    float r = sqrtf(-2.0f * logf(fmaxf(u1, 1e-6f)));
    float theta = FFT_TWO_PI * u2;
    gauss1 = r * cosf(theta);
    gauss2 = r * sinf(theta);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PSEUDO-RANDOM HASH (for deterministic spectrum)
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __host__ inline float hashFloat(int x, int y, int seed) {
    unsigned int n = x + y * 57 + seed * 131;
    n = (n << 13) ^ n;
    unsigned int m = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
    return float(m) / float(0x7fffffff);
}

// ═══════════════════════════════════════════════════════════════════════════════
// API DECLARATIONS (implemented in fft_ocean.cu)
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __cplusplus
extern "C" {
#endif

    // Initialize FFT ocean system (allocate buffers, create cuFFT plans)
    // Returns 1 on success, 0 on failure
    int initFFTOcean(FFTOceanState* state, const FFTOceanParams* params);

    // Update ocean for current time (runs FFT, generates height/normal maps)
    void updateFFTOcean(FFTOceanState* state, const FFTOceanParams* params, float time);

    // Cleanup FFT ocean resources
    void cleanupFFTOcean(FFTOceanState* state);

    // Sample ocean height at world position (for CPU queries)
    float sampleOceanHeight(const FFTOceanState* state, const FFTOceanParams* params, float x, float z);

    // Get ocean normal at world position
    void sampleOceanNormal(const FFTOceanState* state, const FFTOceanParams* params, 
                           float x, float z, float* nx, float* ny, float* nz);

    // ═══════════════════════════════════════════════════════════════════════════════
    // GPU GEOMETRIC WAVES API
    // ═══════════════════════════════════════════════════════════════════════════════
    
    struct GeoWaveParams {
        float time;
        float wave_height;
        float wave_scale;
        float choppiness;
        int octaves;
        float persistence;
        float lacunarity;
        float swell_direction;
        float swell_amplitude;
        float alignment;
        float damping;
        float ridge_offset;
        int noise_type; // 0=Perlin, 1=FBM, 2=Ridge, 3=Voronoi, 4=Billow, 5=Gerstner, 6=Tessendorf
    };
    
    struct GPUGeoWaveState {
        void* d_vertices;
        void* d_original;
        int vertex_count;
        bool initialized;
    };
    
    // Initialize GPU geometric waves (upload original vertices)
    int initGPUGeometricWaves(GPUGeoWaveState* state, const float* h_original, int vertex_count);
    
    // Update geometric waves on GPU and download displaced vertices
    void updateGPUGeometricWaves(GPUGeoWaveState* state, const GeoWaveParams* params, float* h_output);
    
    // Cleanup GPU geometric waves resources
    void cleanupGPUGeometricWaves(GPUGeoWaveState* state);
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // FFT-DRIVEN MESH DISPLACEMENT API
    // ═══════════════════════════════════════════════════════════════════════════════
    // Uses the high-quality FFT ocean data to displace mesh vertices
    // Combines the best of both worlds: FFT quality + physical mesh
    
    struct FFTMeshParams {
        float ocean_size;           // World space size of ocean tile
        float height_scale;         // Height displacement multiplier
        float choppiness;           // Horizontal displacement multiplier
        float mesh_offset_x;        // Mesh origin X offset in world space
        float mesh_offset_z;        // Mesh origin Z offset in world space
    };
    
    // Displace mesh vertices using FFT height/displacement data (GPU accelerated)
    // Input: original vertex positions, FFT state with computed height/displacement
    // Output: displaced vertices with proper normals
    void displaceFFTMesh(
        const FFTOceanState* state,
        const FFTMeshParams* params,
        const float* h_original,    // Host: original vertex positions (x,y,z) x vertex_count
        float* h_output,            // Host: output displaced positions (x,y,z) x vertex_count
        float* h_normals,           // Host: output normals (nx,ny,nz) x vertex_count
        int vertex_count
    );
    
    // GPU-side displacement (for already GPU-resident vertex data)
    void displaceFFTMeshGPU(
        const FFTOceanState* state,
        const FFTMeshParams* params,
        const float3* d_original,   // Device: original positions
        float3* d_output,           // Device: displaced positions
        float3* d_normals,          // Device: output normals
        int vertex_count
    );
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // CPU BATCH SAMPLING API (for mesh displacement without per-vertex GPU calls)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    // Download entire FFT ocean data to CPU for fast batch processing
    // Returns true on success
    bool downloadFFTOceanData(
        const FFTOceanState* state,
        float* h_height,           // Output: height buffer (N*N floats)
        float* h_disp_x,           // Output: X displacement buffer (N*N floats)
        float* h_disp_z,           // Output: Z displacement buffer (N*N floats)
        float* h_normal_x,         // Output: normal X buffer (N*N floats)
        float* h_normal_z,         // Output: normal Z buffer (N*N floats)
        int* out_resolution        // Output: actual resolution N
    );

#ifdef __cplusplus
}
#endif

// ═══════════════════════════════════════════════════════════════════════════════
// GPU TEXTURE SAMPLING (for shaders)
// ═══════════════════════════════════════════════════════════════════════════════

#ifdef __CUDACC__
// Sample height from GPU texture
__device__ inline float sampleOceanHeightGPU(
    cudaTextureObject_t tex_height,
    float u, float v
) {
    return tex2D<float>(tex_height, u, v);
}

// Sample normal from GPU texture (packed as float2: nx, nz, ny computed)
__device__ inline float3 sampleOceanNormalGPU(
    cudaTextureObject_t tex_normal,
    float u, float v
) {
    float2 nxz = tex2D<float2>(tex_normal, u, v);
    float ny = sqrtf(fmaxf(0.0f, 1.0f - nxz.x * nxz.x - nxz.y * nxz.y));
    return make_float3(nxz.x, ny, nxz.y);
}

// Sample displacement from GPU texture (float3: dx, dy, dz)
__device__ inline float3 sampleOceanDisplacementGPU(
    cudaTextureObject_t tex_displacement,
    float u, float v
) {
    float4 d = tex2D<float4>(tex_displacement, u, v);
    return make_float3(d.x, d.y, d.z);
}

// World position to UV coordinates
__device__ inline float2 worldToOceanUV(float x, float z, float ocean_size) {
    float u = x / ocean_size;
    float v = z / ocean_size;
    // Wrap to [0, 1]
    u = u - floorf(u);
    v = v - floorf(v);
    return make_float2(u, v);
}
#endif // __CUDACC__
