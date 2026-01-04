// ═══════════════════════════════════════════════════════════════════════════════
// FFT OCEAN - CUDA Implementation
// ═══════════════════════════════════════════════════════════════════════════════
// Tessendorf's ocean simulation with cuFFT

#include <cstdio>
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
            return false; \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            printf("[FFT Ocean] cuFFT Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            return false; \
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
    
    // Create Texture Objects
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = state->d_height;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = size * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.normalizedCoords = 1; // Use [0,1] coordinates
    
    cudaCreateTextureObject(&state->tex_height, &resDesc, &texDesc, NULL);
    
    // Create Normal Texture (float2)
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = state->d_normal;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.sizeInBytes = size * sizeof(float2);
    
    cudaCreateTextureObject(&state->tex_normal, &resDesc, &texDesc, NULL);
    
    state->initialized = true;
    state->current_resolution = N;
    state->last_update_time = -1.0f;
    state->cached_params = *params;
    
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
        state->cached_params = p; // Update cache
    }
    
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
    
    // Destroy textures
    if (state->tex_height) cudaDestroyTextureObject(state->tex_height);
    if (state->tex_normal) cudaDestroyTextureObject(state->tex_normal);
    if (state->tex_displacement) cudaDestroyTextureObject(state->tex_displacement);
    
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

} // extern "C"
