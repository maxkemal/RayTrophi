// ═══════════════════════════════════════════════════════════════════════════════
// FFT OCEAN - CUDA Implementation
// ═══════════════════════════════════════════════════════════════════════════════
// Tessendorf's ocean simulation with cuFFT

#include <cstdio>
#include <vector>
#include "fft_ocean.cuh"
#include <cufft.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA ERROR CHECKING
// ═══════════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("[FFT Ocean] CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            printf("[FFT Ocean] cuFFT Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            return 0; \
        } \
    } while(0)

// Void-safe version (just logs, doesn't return)
#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("[FFT Ocean] CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL: Initialize H0 Spectrum
// ═══════════════════════════════════════════════════════════════════════════════
// Generates the initial spectrum h̃₀(k) using Phillips spectrum

__global__ void kernelInitializeSpectrum(
    cuFloatComplex* d_h0,
    cuFloatComplex* d_h0_conj,
    int N,
    float dk,
    float wind_speed,
    float wind_dir_x,
    float wind_dir_y,
    float amplitude,
    float small_wave_cutoff,
    float wind_dependency,
    int seed
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= N || y >= N) return;
    
    int idx = y * N + x;
    
    // Compute wave vector k
    // Shift so that (0,0) is at center: k = (x - N/2, y - N/2) * dk
    float kx = (float(x) - float(N) / 2.0f) * dk;
    float ky = (float(y) - float(N) / 2.0f) * dk;
    
    // Phillips spectrum value
    float P = phillipsSpectrum(kx, ky, wind_speed, wind_dir_x, wind_dir_y,
                               amplitude, small_wave_cutoff, wind_dependency);
    
    // Generate Gaussian random numbers
    float u1 = hashFloat(x, y, seed);
    float u2 = hashFloat(x, y, seed + 1);
    float gauss1, gauss2;
    boxMullerGaussian(u1, u2, gauss1, gauss2);
    
    // h̃₀(k) = (1/√2) * (ξr + i*ξi) * √P(k)
    float sqrtP = sqrtf(P);
    float inv_sqrt2 = 0.7071067811865476f;
    
    cuFloatComplex h0;
    h0.x = inv_sqrt2 * gauss1 * sqrtP;
    h0.y = inv_sqrt2 * gauss2 * sqrtP;
    
    d_h0[idx] = h0;
    
    // Store conjugate for -k (used in time evolution)
    // h̃₀*(-k) is accessed via (N-x, N-y)
    int conj_x = (N - x) % N;
    int conj_y = (N - y) % N;
    int conj_idx = conj_y * N + conj_x;
    
    cuFloatComplex h0_conj;
    h0_conj.x = h0.x;
    h0_conj.y = -h0.y;  // Complex conjugate
    
    d_h0_conj[conj_idx] = h0_conj;
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL: Time Evolution
// ═══════════════════════════════════════════════════════════════════════════════
// h̃(k,t) = h̃₀(k) * exp(iωt) + h̃₀*(-k) * exp(-iωt)

__global__ void kernelTimeEvolution(
    cuFloatComplex* d_ht,
    cuFloatComplex* d_displacement_x,
    cuFloatComplex* d_displacement_z,
    const cuFloatComplex* d_h0,
    const cuFloatComplex* d_h0_conj,
    int N,
    float dk,
    float time
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= N || y >= N) return;
    
    int idx = y * N + x;
    
    // Wave vector
    float kx = (float(x) - float(N) / 2.0f) * dk;
    float ky = (float(y) - float(N) / 2.0f) * dk;
    float k_len = sqrtf(kx * kx + ky * ky);
    
    // Dispersion relation: ω = √(g|k|)
    float omega = dispersion(k_len);
    
    // exp(iωt) = cos(ωt) + i*sin(ωt)
    float cos_wt = cosf(omega * time);
    float sin_wt = sinf(omega * time);
    
    // Get h0(k) and h0*(-k)
    cuFloatComplex h0 = d_h0[idx];
    
    // For h0*(-k), access via conjugate index
    int conj_x = (N - x) % N;
    int conj_y = (N - y) % N;
    int conj_idx = conj_y * N + conj_x;
    cuFloatComplex h0_conj = d_h0_conj[conj_idx];
    
    // h(k,t) = h0 * exp(iwt) + h0_conj(-k) * exp(-iwt)
    // exp(iwt) = (cos_wt, sin_wt)
    // exp(-iwt) = (cos_wt, -sin_wt)
    
    cuFloatComplex exp_iwt = make_cuFloatComplex(cos_wt, sin_wt);
    cuFloatComplex exp_neg_iwt = make_cuFloatComplex(cos_wt, -sin_wt);
    
    cuFloatComplex ht;
    ht.x = h0.x * exp_iwt.x - h0.y * exp_iwt.y + h0_conj.x * exp_neg_iwt.x - h0_conj.y * exp_neg_iwt.y;
    ht.y = h0.x * exp_iwt.y + h0.y * exp_iwt.x + h0_conj.x * exp_neg_iwt.y + h0_conj.y * exp_neg_iwt.x;
    
    d_ht[idx] = ht;
    
    // Choppy wave displacement: D(k) = -i * k/|k| * h(k,t)
    // Multiplying by -i: (a + bi) * (-i) = b - ai → (b, -a)
    if (k_len > 0.0001f) {
        float inv_k = 1.0f / k_len;
        
        // -i * kx/|k| * h
        cuFloatComplex dx;
        dx.x = (kx * inv_k) * ht.y;
        dx.y = -(kx * inv_k) * ht.x;
        d_displacement_x[idx] = dx;
        
        // -i * ky/|k| * h
        cuFloatComplex dz;
        dz.x = (ky * inv_k) * ht.y;
        dz.y = -(ky * inv_k) * ht.x;
        d_displacement_z[idx] = dz;
    } else {
        d_displacement_x[idx] = make_cuFloatComplex(0.0f, 0.0f);
        d_displacement_z[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// KERNEL: Extract Real Part & Compute Normals
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void kernelExtractAndNormalize(
    float* d_height,
    float* d_displacement_x_real,
    float* d_displacement_z_real,
    float2* d_normal,
    const cuFloatComplex* d_ht,
    const cuFloatComplex* d_dx,
    const cuFloatComplex* d_dz,
    int N,
    float choppiness,
    float dk
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= N || y >= N) return;
    
    int idx = y * N + x;
    
    // Extract real parts (after IFFT, imaginary should be ~0)
    // Apply sign correction for checkerboard pattern from FFT shift
    float sign = ((x + y) & 1) ? -1.0f : 1.0f;
    
    float height = d_ht[idx].x * sign;
    float disp_x = d_dx[idx].x * sign * choppiness;
    float disp_z = d_dz[idx].x * sign * choppiness;
    
    d_height[idx] = height;
    d_displacement_x_real[idx] = disp_x;
    d_displacement_z_real[idx] = disp_z;
    
    // Compute normals from height gradient
    // Using finite differences
    int left = ((x - 1 + N) % N) + y * N;
    int right = ((x + 1) % N) + y * N;
    int up = x + ((y + 1) % N) * N;
    int down = x + ((y - 1 + N) % N) * N;
    
    float h_left = d_ht[left].x * (((x - 1 + N + y) & 1) ? -1.0f : 1.0f);
    float h_right = d_ht[right].x * (((x + 1 + y) & 1) ? -1.0f : 1.0f);
    float h_up = d_ht[up].x * (((x + y + 1) & 1) ? -1.0f : 1.0f);
    float h_down = d_ht[down].x * (((x + y - 1 + N) & 1) ? -1.0f : 1.0f);
    
    // Gradient
    float dhdx = (h_right - h_left) * 0.5f * dk * float(N) / FFT_TWO_PI;
    float dhdz = (h_up - h_down) * 0.5f * dk * float(N) / FFT_TWO_PI;
    
    // Normal = normalize(-dhdx, 1, -dhdz)
    // We store just x and z, y is computed in shader
    d_normal[idx] = make_float2(-dhdx, -dhdz);
}

// ═══════════════════════════════════════════════════════════════════════════════
// API IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

extern "C" {

int initFFTOcean(FFTOceanState* state, const FFTOceanParams* params) {
    if (!state) return 0;
    // Cleanup if already initialized
    if (state->initialized) {
        cleanupFFTOcean(state);
    }
    
    int N = params->resolution;
    int size = N * N;
    
    printf("[FFT Ocean] Initializing with resolution %dx%d\n", N, N);
    
    // Allocate complex buffers
    CUDA_CHECK(cudaMalloc(&state->d_h0, size * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&state->d_h0_conj, size * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&state->d_ht, size * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&state->d_displacement_x, size * sizeof(cuFloatComplex)));
    CUDA_CHECK(cudaMalloc(&state->d_displacement_z, size * sizeof(cuFloatComplex)));
    
    // Allocate real buffers
    CUDA_CHECK(cudaMalloc(&state->d_height, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_displacement_x_real, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_displacement_z_real, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state->d_normal, size * sizeof(float2)));
    
    // Create cuFFT plan (2D complex-to-complex)
    cufftHandle* plan = new cufftHandle;
    CUFFT_CHECK(cufftPlan2d(plan, N, N, CUFFT_C2C));
    state->fft_plan = plan;
    
    // Initialize spectrum
    FFTOceanParams p = *params;
    p.computeDerived();
    
    float wind_dir_x = cosf(p.wind_direction);
    float wind_dir_y = sinf(p.wind_direction);
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    
    kernelInitializeSpectrum<<<grid, block>>>(
        state->d_h0, state->d_h0_conj, N, p.dk,
        p.wind_speed, wind_dir_x, wind_dir_y,
        p.amplitude, p.small_wave_cutoff, p.wind_dependency,
        12345  // seed
    );
    
    cudaDeviceSynchronize();
    
    // Create CUDA Arrays for proper 2D texturing with linear filtering
    // Height texture (float, 1 channel)
    cudaChannelFormatDesc heightDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&state->height_array, &heightDesc, N, N));
    
    // Normal texture (float2, 2 channels)
    cudaChannelFormatDesc normalDesc = cudaCreateChannelDesc<float2>();
    CUDA_CHECK(cudaMallocArray(&state->normal_array, &normalDesc, N, N));
    
    // Create Texture Objects from CUDA Arrays
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = state->height_array;
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;  // Bilinear filtering
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.normalizedCoords = 1; // Use [0,1] coordinates
    
    CUDA_CHECK(cudaCreateTextureObject(&state->tex_height, &resDesc, &texDesc, NULL));
    
    // Normal texture
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = state->normal_array;
    
    CUDA_CHECK(cudaCreateTextureObject(&state->tex_normal, &resDesc, &texDesc, NULL));
    
    state->initialized = true;
    state->current_resolution = N;
    state->last_update_time = -1.0f;
    state->cached_params = p;  // Use computed derived params for proper comparison later
    
    // Run initial FFT to populate textures (time=0)
    // This ensures textures have valid data immediately after init
    {
        dim3 block2(16, 16);
        dim3 grid2((N + block2.x - 1) / block2.x, (N + block2.y - 1) / block2.y);
        
        kernelTimeEvolution<<<grid2, block2>>>(
            state->d_ht, state->d_displacement_x, state->d_displacement_z,
            state->d_h0, state->d_h0_conj, N, p.dk, 0.0f
        );
        
        cufftExecC2C(*plan, state->d_ht, state->d_ht, CUFFT_INVERSE);
        cufftExecC2C(*plan, state->d_displacement_x, state->d_displacement_x, CUFFT_INVERSE);
        cufftExecC2C(*plan, state->d_displacement_z, state->d_displacement_z, CUFFT_INVERSE);
        
        kernelExtractAndNormalize<<<grid2, block2>>>(
            state->d_height, state->d_displacement_x_real, state->d_displacement_z_real,
            state->d_normal,
            state->d_ht, state->d_displacement_x, state->d_displacement_z,
            N, p.choppiness, p.dk
        );
        
        cudaDeviceSynchronize();
        
        // Copy to arrays
        CUDA_CHECK(cudaMemcpy2DToArray(
            state->height_array, 0, 0,
            state->d_height, N * sizeof(float),
            N * sizeof(float), N,
            cudaMemcpyDeviceToDevice
        ));
        
        CUDA_CHECK(cudaMemcpy2DToArray(
            state->normal_array, 0, 0,
            state->d_normal, N * sizeof(float2),
            N * sizeof(float2), N,
            cudaMemcpyDeviceToDevice
        ));
    }
    
    printf("[FFT Ocean] Initialization complete\n");
    return 1;
}

void updateFFTOcean(FFTOceanState* state, const FFTOceanParams* params, float time) {
    if (!state || !state->initialized) return;
    
    int N = params->resolution;
    if (N != state->current_resolution) {
        // Resolution changed, reinitialize
        initFFTOcean(state, params);
        return;
    }
    
    FFTOceanParams p = *params;
    p.computeDerived();
    
    // Check if we need to re-initialize spectrum (wind/amplitude/size changed)
    bool need_new_spectrum = false;
    if (p.ocean_size != state->cached_params.ocean_size ||
        p.wind_speed != state->cached_params.wind_speed ||
        p.wind_direction != state->cached_params.wind_direction ||
        p.wind_dependency != state->cached_params.wind_dependency ||
        p.amplitude != state->cached_params.amplitude ||
        p.small_wave_cutoff != state->cached_params.small_wave_cutoff) 
    {
        need_new_spectrum = true;
    }
    
    // Always update cached params so choppiness/time changes don't trigger constant GPU re-syncs!
    state->cached_params = p;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    if (need_new_spectrum) {
        float wind_dir_x = cosf(p.wind_direction);
        float wind_dir_y = sinf(p.wind_direction);
        
        kernelInitializeSpectrum<<<grid, block>>>(
            state->d_h0, state->d_h0_conj, N, p.dk,
            p.wind_speed, wind_dir_x, wind_dir_y,
            p.amplitude, p.small_wave_cutoff, p.wind_dependency,
            12345  // seed
        );
    }
    
    float scaled_time = time * p.time_scale;

    // Time evolution: compute h(k,t)
    kernelTimeEvolution<<<grid, block>>>(
        state->d_ht, state->d_displacement_x, state->d_displacement_z,
        state->d_h0, state->d_h0_conj, N, p.dk, scaled_time
    );
    
    // Perform inverse FFT on height
    cufftHandle* plan = (cufftHandle*)state->fft_plan;
    cufftExecC2C(*plan, state->d_ht, state->d_ht, CUFFT_INVERSE);
    cufftExecC2C(*plan, state->d_displacement_x, state->d_displacement_x, CUFFT_INVERSE);
    cufftExecC2C(*plan, state->d_displacement_z, state->d_displacement_z, CUFFT_INVERSE);
    
    // Extract real parts and compute normals
    kernelExtractAndNormalize<<<grid, block>>>(
        state->d_height, state->d_displacement_x_real, state->d_displacement_z_real,
        state->d_normal,
        state->d_ht, state->d_displacement_x, state->d_displacement_z,
        N, p.choppiness, p.dk
    );
    
    cudaDeviceSynchronize();
    
    // Copy computed data to CUDA Arrays for texture sampling
    // Height: N*N floats
    CUDA_CHECK_VOID(cudaMemcpy2DToArray(
        state->height_array, 0, 0,
        state->d_height, N * sizeof(float),
        N * sizeof(float), N,
        cudaMemcpyDeviceToDevice
    ));
    
    // Normal: N*N float2s
    CUDA_CHECK_VOID(cudaMemcpy2DToArray(
        state->normal_array, 0, 0,
        state->d_normal, N * sizeof(float2),
        N * sizeof(float2), N,
        cudaMemcpyDeviceToDevice
    ));
    
    state->last_update_time = time;
}

void cleanupFFTOcean(FFTOceanState* state) {
    if (!state || !state->initialized) return;
    
    printf("[FFT Ocean] Cleanup\n");
    
    // Destroy cuFFT plan
    if (state->fft_plan) {
        cufftDestroy(*(cufftHandle*)state->fft_plan);
        delete (cufftHandle*)state->fft_plan;
        state->fft_plan = nullptr;
    }
    
    // Free complex buffers
    if (state->d_h0) cudaFree(state->d_h0);
    if (state->d_h0_conj) cudaFree(state->d_h0_conj);
    if (state->d_ht) cudaFree(state->d_ht);
    if (state->d_displacement_x) cudaFree(state->d_displacement_x);
    if (state->d_displacement_z) cudaFree(state->d_displacement_z);
    
    // Free real buffers
    if (state->d_height) cudaFree(state->d_height);
    if (state->d_displacement_x_real) cudaFree(state->d_displacement_x_real);
    if (state->d_displacement_z_real) cudaFree(state->d_displacement_z_real);
    if (state->d_normal) cudaFree(state->d_normal);
    
    // Destroy textures (must be done before freeing arrays)
    if (state->tex_height) cudaDestroyTextureObject(state->tex_height);
    if (state->tex_normal) cudaDestroyTextureObject(state->tex_normal);
    if (state->tex_displacement) cudaDestroyTextureObject(state->tex_displacement);
    
    // Free CUDA Arrays
    if (state->height_array) cudaFreeArray(state->height_array);
    if (state->normal_array) cudaFreeArray(state->normal_array);
    
    *state = FFTOceanState();  // Reset to defaults
}

// CPU sampling (for queries outside GPU rendering)
float sampleOceanHeight(const FFTOceanState* state, const FFTOceanParams* params, float x, float z) {
    if (!state || !state->initialized) return 0.0f;
    
    int N = params->resolution;
    
    // World to grid coordinates
    float u = x / params->ocean_size;
    float v = z / params->ocean_size;
    
    // Wrap to [0, 1]
    u = u - floorf(u);
    v = v - floorf(v);
    
    // Grid indices
    int ix = int(u * N) % N;
    int iy = int(v * N) % N;
    int idx = iy * N + ix;
    
    // Copy single value from GPU
    float height;
    cudaMemcpy(&height, state->d_height + idx, sizeof(float), cudaMemcpyDeviceToHost);
    
    return height;
}

void sampleOceanNormal(const FFTOceanState* state, const FFTOceanParams* params,
                       float x, float z, float* nx, float* ny, float* nz) {
    if (!state || !state->initialized) {
        *nx = 0.0f; *ny = 1.0f; *nz = 0.0f;
        return;
    }
    
    int N = params->resolution;
    
    // World to grid coordinates
    float u = x / params->ocean_size;
    float v = z / params->ocean_size;
    u = u - floorf(u);
    v = v - floorf(v);
    
    int ix = int(u * N) % N;
    int iy = int(v * N) % N;
    int idx = iy * N + ix;
    
    // Copy from GPU
    float2 norm;
    cudaMemcpy(&norm, state->d_normal + idx, sizeof(float2), cudaMemcpyDeviceToHost);
    
    // Reconstruct Y and normalize
    float len_sq = norm.x * norm.x + norm.y * norm.y + 1.0f;
    float inv_len = 1.0f / sqrtf(len_sq);
    
    *nx = norm.x * inv_len;
    *ny = inv_len;
    *nz = norm.y * inv_len;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU GEOMETRIC WAVES - Mesh Displacement Kernel
// ═══════════════════════════════════════════════════════════════════════════════
// Displaces mesh vertices using Gerstner waves or procedural noise on GPU
// Note: GeoWaveParams struct is defined in fft_ocean.cuh

// Simple GPU Perlin noise hash
__device__ inline float gpuHash(int x, int y) {
    unsigned int n = x + y * 57;
    n = (n << 13) ^ n;
    unsigned int m = (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
    return float(m) / float(0x7fffffff);
}

// GPU Gradient noise 2D
__device__ inline float gpuGradientNoise(float x, float y) {
    int ix = (int)floorf(x);
    int iy = (int)floorf(y);
    float fx = x - ix;
    float fy = y - iy;
    
    // Smoothstep
    float u = fx * fx * (3.0f - 2.0f * fx);
    float v = fy * fy * (3.0f - 2.0f * fy);
    
    // Hash corners
    float n00 = gpuHash(ix, iy) * 2.0f - 1.0f;
    float n10 = gpuHash(ix + 1, iy) * 2.0f - 1.0f;
    float n01 = gpuHash(ix, iy + 1) * 2.0f - 1.0f;
    float n11 = gpuHash(ix + 1, iy + 1) * 2.0f - 1.0f;
    
    // Bilinear interpolation
    float nx0 = n00 * (1.0f - u) + n10 * u;
    float nx1 = n01 * (1.0f - u) + n11 * u;
    return nx0 * (1.0f - v) + nx1 * v;
}

// GPU FBM (Fractal Brownian Motion)
__device__ inline float gpuFBM(float x, float y, int octaves, float persistence, float lacunarity) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        value += amplitude * gpuGradientNoise(x * frequency, y * frequency);
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / maxValue;
}

// GPU Ridge Noise
__device__ inline float gpuRidgeNoise(float x, float y, int octaves, float persistence, float lacunarity, float offset) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float weight = 1.0f;
    float maxValue = 0.0f;
    
    for (int i = 0; i < octaves; ++i) {
        float n = gpuGradientNoise(x * frequency, y * frequency);
        n = offset - fabsf(n);
        n = n * n;
        n *= weight;
        weight = fminf(1.0f, fmaxf(0.0f, n * 2.0f));
        value += amplitude * n;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / maxValue;
}

// Gerstner Wave computation
__device__ inline float3 gpuGerstnerWave(
    float x, float z, float time,
    float wavelength, float amplitude, float steepness,
    float dirX, float dirZ
) {
    float k = 2.0f * 3.14159265f / wavelength;
    float speed = sqrtf(9.81f * k);
    float phase = k * (dirX * x + dirZ * z) - speed * time;
    
    float s = sinf(phase);
    float c = cosf(phase);
    
    float3 disp;
    disp.x = steepness * amplitude * dirX * c;
    disp.y = amplitude * s;
    disp.z = steepness * amplitude * dirZ * c;
    
    return disp;
}

// Main geometric wave displacement kernel
__global__ void kernelGeometricWaves(
    float3* d_vertices,           // Output: displaced vertices
    const float3* d_original,     // Input: original vertex positions
    int vertex_count,
    GeoWaveParams params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;
    
    float3 orig = d_original[idx];
    float x = orig.x;
    float z = orig.z;
    float base_y = orig.y;
    
    float3 displacement = make_float3(0, 0, 0);
    
    float scale = fmaxf(0.1f, params.wave_scale);
    float time = params.time;
    
    if (params.noise_type == 5 || params.noise_type == 6) {
        // Gerstner / Tessendorf waves
        int numWaves = 6;
        for (int i = 0; i < numWaves; ++i) {
            float freqMult = powf(params.lacunarity, (float)i);
            float ampMult = powf(params.persistence, (float)i);
            
            float wavelength = scale / freqMult;
            float amplitude = params.wave_height * ampMult * 0.25f;
            float steepness = fminf(1.0f, params.choppiness * 0.5f);
            
            // Direction spreading
            float dirSpread = (1.0f - params.alignment) * 3.14159265f * 0.5f;
            float dirOffset = ((float)i - 2.5f) / 2.5f * dirSpread;
            float dir = params.swell_direction + dirOffset;
            
            // Damping for perpendicular waves
            float angleDiff = fabsf(dirOffset);
            float dampFactor = 1.0f - params.damping * sinf(angleDiff);
            amplitude *= fmaxf(0.1f, dampFactor);
            
            float dirX = cosf(dir);
            float dirZ = sinf(dir);
            
            float3 wave = gpuGerstnerWave(x, z, time, wavelength, amplitude, steepness, dirX, dirZ);
            displacement.x += wave.x;
            displacement.y += wave.y;
            displacement.z += wave.z;
        }
        
        // Add swell
        if (params.swell_amplitude > 0.0f) {
            float swellDir = params.swell_direction + 3.14159265f * 0.25f;
            float3 swell = gpuGerstnerWave(x, z, time * 0.5f, scale * 3.0f, 
                                           params.wave_height * params.swell_amplitude, 
                                           0.3f, cosf(swellDir), sinf(swellDir));
            displacement.x += swell.x;
            displacement.y += swell.y;
            displacement.z += swell.z;
        }
    } else {
        // Procedural noise types
        float nx = x / scale + time * 0.1f;
        float nz = z / scale + time * 0.1f;
        float height = 0.0f;
        
        switch (params.noise_type) {
            case 0: // Perlin
                height = gpuGradientNoise(nx, nz);
                break;
            case 1: // FBM
                height = gpuFBM(nx, nz, params.octaves, params.persistence, params.lacunarity);
                break;
            case 2: // Ridge
                height = gpuRidgeNoise(nx, nz, params.octaves, params.persistence, 
                                       params.lacunarity, params.ridge_offset);
                break;
            case 3: // Voronoi (simplified)
                height = gpuFBM(nx * 2.0f, nz * 2.0f, params.octaves, params.persistence, params.lacunarity);
                height = fabsf(height);
                break;
            case 4: // Billow
                height = fabsf(gpuFBM(nx, nz, params.octaves, params.persistence, params.lacunarity)) * 2.0f - 1.0f;
                break;
        }
        
        displacement.y = height * params.wave_height;
    }
    
    // Write displaced vertex
    d_vertices[idx] = make_float3(
        x + displacement.x,
        base_y + displacement.y,
        z + displacement.z
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU GEOMETRIC WAVES API
// ═══════════════════════════════════════════════════════════════════════════════
// Note: GPUGeoWaveState struct is defined in fft_ocean.cuh

int initGPUGeometricWaves(GPUGeoWaveState* state, const float* h_original, int vertex_count) {
    if (!state) return 0;
    
    state->vertex_count = vertex_count;
    
    cudaError_t err;
    float3* d_orig_ptr = nullptr;
    float3* d_vert_ptr = nullptr;
    
    err = cudaMalloc(&d_orig_ptr, vertex_count * sizeof(float3));
    if (err != cudaSuccess) return 0;
    
    err = cudaMalloc(&d_vert_ptr, vertex_count * sizeof(float3));
    if (err != cudaSuccess) {
        cudaFree(d_orig_ptr);
        return 0;
    }
    
    err = cudaMemcpy(d_orig_ptr, h_original, vertex_count * sizeof(float3), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_orig_ptr);
        cudaFree(d_vert_ptr);
        return 0;
    }
    
    state->d_original = d_orig_ptr;
    state->d_vertices = d_vert_ptr;
    state->initialized = true;
    return 1;
}

void updateGPUGeometricWaves(GPUGeoWaveState* state, const GeoWaveParams* params, float* h_output) {
    if (!state || !state->initialized) return;
    
    float3* d_vertices = static_cast<float3*>(state->d_vertices);
    float3* d_original = static_cast<float3*>(state->d_original);
    
    int blockSize = 256;
    int gridSize = (state->vertex_count + blockSize - 1) / blockSize;
    
    kernelGeometricWaves<<<gridSize, blockSize>>>(
        d_vertices,
        d_original,
        state->vertex_count,
        *params
    );
    
    cudaDeviceSynchronize();
    
    // Copy back to host
    cudaMemcpy(h_output, d_vertices, state->vertex_count * sizeof(float3), cudaMemcpyDeviceToHost);
}

void cleanupGPUGeometricWaves(GPUGeoWaveState* state) {
    if (!state) return;
    
    if (state->d_vertices) {
        cudaFree(state->d_vertices);
        state->d_vertices = nullptr;
    }
    if (state->d_original) {
        cudaFree(state->d_original);
        state->d_original = nullptr;
    }
    state->initialized = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT-DRIVEN MESH DISPLACEMENT
// ═══════════════════════════════════════════════════════════════════════════════
// This kernel samples the FFT height/displacement textures and applies them to mesh vertices
// Result: Film-quality ocean waves on physical mesh for raytracing

__global__ void kernelFFTMeshDisplacement(
    float3* d_output,              // Output displaced positions
    float3* d_normals,             // Output normals
    const float3* d_original,      // Original positions
    const float* d_height,         // FFT height data
    const float* d_disp_x,         // FFT X displacement
    const float* d_disp_z,         // FFT Z displacement
    const float2* d_fft_normal,    // FFT normals (x,z packed)
    int vertex_count,
    int N,                         // FFT resolution
    float ocean_size,
    float height_scale,
    float choppiness,
    float offset_x,
    float offset_z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= vertex_count) return;
    
    float3 orig = d_original[idx];
    
    // World position to UV (0-1 range, wrapping)
    float u = (orig.x - offset_x) / ocean_size;
    float v = (orig.z - offset_z) / ocean_size;
    
    // Wrap to [0, 1)
    u = u - floorf(u);
    v = v - floorf(v);
    
    // UV to grid indices with bilinear interpolation
    float fx = u * (float)N;
    float fz = v * (float)N;
    
    int ix0 = (int)floorf(fx);
    int iz0 = (int)floorf(fz);
    int ix1 = (ix0 + 1) % N;
    int iz1 = (iz0 + 1) % N;
    ix0 = ix0 % N;
    iz0 = iz0 % N;
    
    float tx = fx - floorf(fx);
    float tz = fz - floorf(fz);
    
    // Bilinear sample height
    float h00 = d_height[iz0 * N + ix0];
    float h10 = d_height[iz0 * N + ix1];
    float h01 = d_height[iz1 * N + ix0];
    float h11 = d_height[iz1 * N + ix1];
    
    float h0 = h00 * (1.0f - tx) + h10 * tx;
    float h1 = h01 * (1.0f - tx) + h11 * tx;
    float height = h0 * (1.0f - tz) + h1 * tz;
    
    // Bilinear sample X displacement
    float dx00 = d_disp_x[iz0 * N + ix0];
    float dx10 = d_disp_x[iz0 * N + ix1];
    float dx01 = d_disp_x[iz1 * N + ix0];
    float dx11 = d_disp_x[iz1 * N + ix1];
    
    float dx0 = dx00 * (1.0f - tx) + dx10 * tx;
    float dx1 = dx01 * (1.0f - tx) + dx11 * tx;
    float disp_x = dx0 * (1.0f - tz) + dx1 * tz;
    
    // Bilinear sample Z displacement
    float dz00 = d_disp_z[iz0 * N + ix0];
    float dz10 = d_disp_z[iz0 * N + ix1];
    float dz01 = d_disp_z[iz1 * N + ix0];
    float dz11 = d_disp_z[iz1 * N + ix1];
    
    float dz0 = dz00 * (1.0f - tx) + dz10 * tx;
    float dz1 = dz01 * (1.0f - tx) + dz11 * tx;
    float disp_z = dz0 * (1.0f - tz) + dz1 * tz;
    
    // Bilinear sample normal
    float2 n00 = d_fft_normal[iz0 * N + ix0];
    float2 n10 = d_fft_normal[iz0 * N + ix1];
    float2 n01 = d_fft_normal[iz1 * N + ix0];
    float2 n11 = d_fft_normal[iz1 * N + ix1];
    
    float2 n0 = make_float2(n00.x * (1.0f - tx) + n10.x * tx, n00.y * (1.0f - tx) + n10.y * tx);
    float2 n1 = make_float2(n01.x * (1.0f - tx) + n11.x * tx, n01.y * (1.0f - tx) + n11.y * tx);
    float2 norm_xz = make_float2(n0.x * (1.0f - tz) + n1.x * tz, n0.y * (1.0f - tz) + n1.y * tz);
    
    // Apply displacements
    float3 displaced;
    displaced.x = orig.x + disp_x * choppiness;
    displaced.y = orig.y + height * height_scale;
    displaced.z = orig.z + disp_z * choppiness;
    
    d_output[idx] = displaced;
    
    // Compute normal (normalize the gradient)
    float nx = norm_xz.x;
    float nz = norm_xz.y;
    float ny = sqrtf(fmaxf(0.0f, 1.0f - nx * nx - nz * nz));
    
    // Renormalize
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len > 0.0001f) {
        nx /= len;
        ny /= len;
        nz /= len;
    } else {
        nx = 0.0f;
        ny = 1.0f;
        nz = 0.0f;
    }
    
    d_normals[idx] = make_float3(nx, ny, nz);
}

void displaceFFTMesh(
    const FFTOceanState* state,
    const FFTMeshParams* params,
    const float* h_original,
    float* h_output,
    float* h_normals,
    int vertex_count
) {
    if (!state || !state->initialized || vertex_count == 0) return;
    
    int N = state->current_resolution;
    
    // Allocate temporary GPU buffers
    float3* d_original = nullptr;
    float3* d_output = nullptr;
    float3* d_normals = nullptr;
    
    cudaMalloc(&d_original, vertex_count * sizeof(float3));
    cudaMalloc(&d_output, vertex_count * sizeof(float3));
    cudaMalloc(&d_normals, vertex_count * sizeof(float3));
    
    // Copy original to GPU
    cudaMemcpy(d_original, h_original, vertex_count * sizeof(float3), cudaMemcpyHostToDevice);
    
    // Run kernel
    int blockSize = 256;
    int gridSize = (vertex_count + blockSize - 1) / blockSize;
    
    kernelFFTMeshDisplacement<<<gridSize, blockSize>>>(
        d_output,
        d_normals,
        d_original,
        state->d_height,
        state->d_displacement_x_real,
        state->d_displacement_z_real,
        state->d_normal,
        vertex_count,
        N,
        params->ocean_size,
        params->height_scale,
        params->choppiness,
        params->mesh_offset_x,
        params->mesh_offset_z
    );
    
    cudaDeviceSynchronize();
    
    // Copy back results
    cudaMemcpy(h_output, d_output, vertex_count * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_normals, d_normals, vertex_count * sizeof(float3), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_original);
    cudaFree(d_output);
    cudaFree(d_normals);
}

void displaceFFTMeshGPU(
    const FFTOceanState* state,
    const FFTMeshParams* params,
    const float3* d_original,
    float3* d_output,
    float3* d_normals,
    int vertex_count
) {
    if (!state || !state->initialized || vertex_count == 0) return;
    
    int N = state->current_resolution;
    
    int blockSize = 256;
    int gridSize = (vertex_count + blockSize - 1) / blockSize;
    
    kernelFFTMeshDisplacement<<<gridSize, blockSize>>>(
        d_output,
        d_normals,
        d_original,
        state->d_height,
        state->d_displacement_x_real,
        state->d_displacement_z_real,
        state->d_normal,
        vertex_count,
        N,
        params->ocean_size,
        params->height_scale,
        params->choppiness,
        params->mesh_offset_x,
        params->mesh_offset_z
    );
    
    cudaDeviceSynchronize();
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOWNLOAD FFT OCEAN DATA TO CPU (for batch processing)
// ═══════════════════════════════════════════════════════════════════════════════
bool downloadFFTOceanData(
    const FFTOceanState* state,
    float* h_height,
    float* h_disp_x,
    float* h_disp_z,
    float* h_normal_x,
    float* h_normal_z,
    int* out_resolution
) {
    if (!state || !state->initialized) return false;
    
    int N = state->current_resolution;
    int N2 = N * N;
    
    *out_resolution = N;
    
    // Download height map
    if (h_height && state->d_height) {
        cudaMemcpy(h_height, state->d_height, N2 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Download X displacement
    if (h_disp_x && state->d_displacement_x_real) {
        cudaMemcpy(h_disp_x, state->d_displacement_x_real, N2 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Download Z displacement
    if (h_disp_z && state->d_displacement_z_real) {
        cudaMemcpy(h_disp_z, state->d_displacement_z_real, N2 * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    // Download normals (packed as float2)
    if ((h_normal_x || h_normal_z) && state->d_normal) {
        std::vector<float2> h_normals(N2);
        cudaMemcpy(h_normals.data(), state->d_normal, N2 * sizeof(float2), cudaMemcpyDeviceToHost);
        
        for (int i = 0; i < N2; ++i) {
            if (h_normal_x) h_normal_x[i] = h_normals[i].x;
            if (h_normal_z) h_normal_z[i] = h_normals[i].y;
        }
    }
    
    return true;
}

} // extern "C"
