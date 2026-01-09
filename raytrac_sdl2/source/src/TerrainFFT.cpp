/**
 * @file TerrainFFT.cpp
 * @brief Implementation of FFT-based terrain generation with CUDA fallback
 */

#include "TerrainFFT.h"
#include <algorithm>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace TerrainFFT {

// ============================================================================
// CUDA ENUM DEFINITIONS (to avoid cuda header dependency)
// ============================================================================

enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

// ============================================================================
// INTERNAL CUDA STATE
// ============================================================================

struct CUDAFFTState {
    // Device buffers
    void* d_spectrum = nullptr;     // Complex spectrum
    void* d_heightmap = nullptr;    // Real heightmap output
    void* fft_plan = nullptr;       // cuFFT plan handle
    
    int currentResolution = 0;
    bool initialized = false;
};

// ============================================================================
// TERRAIN FFT MANAGER IMPLEMENTATION
// ============================================================================

TerrainFFTManager::TerrainFFTManager() 
    : rng_(std::random_device{}()) 
{
    cudaState_ = std::make_unique<CUDAFFTState>();
    
    // Try to initialize CUDA (won't crash if unavailable)
    initCUDA();
}

TerrainFFTManager::~TerrainFFTManager() {
    cleanupCUDA();
}

bool TerrainFFTManager::initCUDA() {
    cudaAvailable_ = false;
    cudaStatusMessage_ = "Checking CUDA availability...";
    
#ifdef _WIN32
    // Try to load CUDA runtime DLL
    // Try multiple possible names
    const char* cudartNames[] = {
        "cudart64_12.dll",
        "cudart64_11.dll", 
        "cudart64_110.dll",
        "cudart64_102.dll",
        "cudart64_101.dll",
        "cudart64_100.dll",
        nullptr
    };
    
    for (int i = 0; cudartNames[i] != nullptr; i++) {
        cudartDLL_ = LoadLibraryA(cudartNames[i]);
        if (cudartDLL_) {
            cudaStatusMessage_ = std::string("Loaded ") + cudartNames[i];
            break;
        }
    }
    
    if (!cudartDLL_) {
        cudaStatusMessage_ = "CUDA runtime DLL not found - using CPU fallback";
        return false;
    }
    
    // Load essential functions
    pfnCudaGetDeviceCount_ = (PFN_CudaGetDeviceCount)GetProcAddress(cudartDLL_, "cudaGetDeviceCount");
    pfnCudaSetDevice_ = (PFN_CudaSetDevice)GetProcAddress(cudartDLL_, "cudaSetDevice");
    pfnCudaMalloc_ = (PFN_CudaMalloc)GetProcAddress(cudartDLL_, "cudaMalloc");
    pfnCudaFree_ = (PFN_CudaFree)GetProcAddress(cudartDLL_, "cudaFree");
    pfnCudaMemcpy_ = (PFN_CudaMemcpy)GetProcAddress(cudartDLL_, "cudaMemcpy");
    pfnCudaDeviceSynchronize_ = (PFN_CudaDeviceSynchronize)GetProcAddress(cudartDLL_, "cudaDeviceSynchronize");
    
    if (!pfnCudaGetDeviceCount_ || !pfnCudaMalloc_ || !pfnCudaFree_ || !pfnCudaMemcpy_) {
        cudaStatusMessage_ = "Failed to load CUDA functions - using CPU fallback";
        FreeLibrary(cudartDLL_);
        cudartDLL_ = nullptr;
        return false;
    }
    
    // Check if CUDA device is available
    int deviceCount = 0;
    int result = pfnCudaGetDeviceCount_(&deviceCount);
    
    if (result != 0 || deviceCount == 0) {
        cudaStatusMessage_ = "No CUDA devices found - using CPU fallback";
        FreeLibrary(cudartDLL_);
        cudartDLL_ = nullptr;
        return false;
    }
    
    // Try to load cuFFT DLL
    const char* cufftNames[] = {
        "cufft64_11.dll",
        "cufft64_10.dll",
        "cufft64_100.dll",
        nullptr
    };
    
    for (int i = 0; cufftNames[i] != nullptr; i++) {
        cufftDLL_ = LoadLibraryA(cufftNames[i]);
        if (cufftDLL_) break;
    }
    
    // cuFFT is optional - we can still use basic CUDA operations
    if (!cufftDLL_) {
        cudaStatusMessage_ = "cuFFT DLL not found - GPU available but FFT will use CPU";
    }
    
    cudaAvailable_ = true;
    cudaInitialized_ = true;
    cudaStatusMessage_ = "CUDA initialized successfully (" + std::to_string(deviceCount) + " device(s))";
    
#else
    // Linux/Mac - similar approach with dlopen
    cudartDLL_ = dlopen("libcudart.so", RTLD_LAZY);
    if (!cudartDLL_) {
        cudaStatusMessage_ = "CUDA runtime not found - using CPU fallback";
        return false;
    }
    
    // Load functions with dlsym
    pfnCudaGetDeviceCount_ = (PFN_CudaGetDeviceCount)dlsym(cudartDLL_, "cudaGetDeviceCount");
    
    int deviceCount = 0;
    if (!pfnCudaGetDeviceCount_ || pfnCudaGetDeviceCount_(&deviceCount) != 0 || deviceCount == 0) {
        cudaStatusMessage_ = "No CUDA devices available - using CPU fallback";
        dlclose(cudartDLL_);
        cudartDLL_ = nullptr;
        return false;
    }
    
    cudaAvailable_ = true;
    cudaInitialized_ = true;
    cudaStatusMessage_ = "CUDA initialized on Linux";
#endif
    
    return true;
}

void TerrainFFTManager::cleanupCUDA() {
    // Free CUDA resources
    if (cudaState_ && cudaState_->initialized && pfnCudaFree_) {
        if (cudaState_->d_spectrum) pfnCudaFree_(cudaState_->d_spectrum);
        if (cudaState_->d_heightmap) pfnCudaFree_(cudaState_->d_heightmap);
        cudaState_->initialized = false;
    }
    
    // Unload DLLs
#ifdef _WIN32
    if (cufftDLL_) {
        FreeLibrary(cufftDLL_);
        cufftDLL_ = nullptr;
    }
    if (cudartDLL_) {
        FreeLibrary(cudartDLL_);
        cudartDLL_ = nullptr;
    }
#else
    if (cufftDLL_) {
        dlclose(cufftDLL_);
        cufftDLL_ = nullptr;
    }
    if (cudartDLL_) {
        dlclose(cudartDLL_);
        cudartDLL_ = nullptr;
    }
#endif
    
    cudaAvailable_ = false;
    cudaInitialized_ = false;
}

FFTNoiseResult TerrainFFTManager::generateNoise(const FFTNoiseParams& params) {
    // Decide whether to use GPU or CPU
    bool useGPU = cudaAvailable_ && !forceCPU_ && cufftDLL_ != nullptr;
    
    if (useGPU) {
        FFTNoiseResult result = generateNoiseGPU(params);
        if (result.success) {
            return result;
        }
        // GPU failed, fall back to CPU
    }
    
    // Use CPU
    return generateNoiseCPU(params);
}

bool TerrainFFTManager::generateNoiseInPlace(const FFTNoiseParams& params, float* output) {
    if (!output) return false;
    
    FFTNoiseResult result = generateNoise(params);
    if (!result.success) return false;
    
    size_t size = (size_t)params.resolution * params.resolution;
    std::memcpy(output, result.heightmap.data(), size * sizeof(float));
    
    return true;
}

// ============================================================================
// GPU IMPLEMENTATION
// ============================================================================

FFTNoiseResult TerrainFFTManager::generateNoiseGPU(const FFTNoiseParams& params) {
    FFTNoiseResult result;
    result.usedGPU = true;
    
    // For now, fall back to CPU since full CUDA kernel implementation
    // would require .cu file compilation
    // This can be expanded when CUDA kernels are added
    
    result.success = false;
    result.errorMessage = "GPU FFT not yet implemented - using CPU fallback";
    
    // If GPU implementation is added later, it would:
    // 1. Allocate device buffers
    // 2. Generate spectrum on GPU
    // 3. Run cuFFT inverse transform
    // 4. Copy result back to host
    
    return result;
}

// ============================================================================
// CPU IMPLEMENTATION
// ============================================================================

FFTNoiseResult TerrainFFTManager::generateNoiseCPU(const FFTNoiseParams& params) {
    FFTNoiseResult result;
    result.usedGPU = false;
    
    int N = params.resolution;
    result.width = N;
    result.height = N;
    result.heightmap.resize(N * N, 0.0f);
    
    // Use CPU noise generation
    CPUNoise::generateHeightmap(result.heightmap.data(), N, N, params);
    
    result.success = true;
    return result;
}

// ============================================================================
// SPECTRUM UTILITIES
// ============================================================================

float TerrainFFTManager::phillipsSpectrum(float kx, float ky, float windSpeed,
                                          float windDirX, float windDirY, float amplitude) {
    float k_len = std::sqrt(kx * kx + ky * ky);
    if (k_len < 0.0001f) return 0.0f;
    
    const float g = 9.81f;
    float L = windSpeed * windSpeed / g;
    
    float k2 = k_len * k_len;
    float k4 = k2 * k2;
    float L2 = L * L;
    
    // Phillips spectrum
    float P = amplitude * std::exp(-1.0f / (k2 * L2)) / k4;
    
    // Directional factor
    float kx_norm = kx / k_len;
    float ky_norm = ky / k_len;
    float k_dot_w = kx_norm * windDirX + ky_norm * windDirY;
    P *= k_dot_w * k_dot_w;
    
    return P;
}

float TerrainFFTManager::powerSpectrum(float k, float exponent) {
    if (k < 0.0001f) return 0.0f;
    return 1.0f / std::pow(k, exponent);
}

// ============================================================================
// CPU NOISE NAMESPACE IMPLEMENTATION
// ============================================================================

namespace CPUNoise {

// Hash function for noise
inline uint32_t hash(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

inline float hashFloat(int x, int y, int seed) {
    uint32_t n = hash(x + hash(y) + hash(seed));
    return static_cast<float>(n) / static_cast<float>(0xFFFFFFFF);
}

// Smooth interpolation
inline float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Simple gradient noise
float gradientNoise(float x, float y, int seed) {
    int ix = static_cast<int>(std::floor(x));
    int iy = static_cast<int>(std::floor(y));
    float fx = x - ix;
    float fy = y - iy;
    
    // Get random values at corners
    float v00 = hashFloat(ix, iy, seed) * 2.0f - 1.0f;
    float v10 = hashFloat(ix + 1, iy, seed) * 2.0f - 1.0f;
    float v01 = hashFloat(ix, iy + 1, seed) * 2.0f - 1.0f;
    float v11 = hashFloat(ix + 1, iy + 1, seed) * 2.0f - 1.0f;
    
    // Smooth interpolation
    float sx = smoothstep(fx);
    float sy = smoothstep(fy);
    
    float n0 = lerp(v00, v10, sx);
    float n1 = lerp(v01, v11, sx);
    
    return lerp(n0, n1, sy);
}

float fbmNoise(float x, float y, int octaves, float persistence,
               float lacunarity, int seed) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    
    for (int i = 0; i < octaves; i++) {
        total += gradientNoise(x * frequency, y * frequency, seed + i) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return total / maxValue;
}

float ridgedNoise(float x, float y, int octaves, float persistence,
                  float lacunarity, float offset, float gain, int seed) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float weight = 1.0f;
    
    for (int i = 0; i < octaves; i++) {
        float n = gradientNoise(x * frequency, y * frequency, seed + i);
        
        // Make ridges
        n = std::abs(n);
        n = offset - n;
        n = n * n;
        
        // Weight successive octaves by previous
        n *= weight;
        weight = (std::max)(0.0f, (std::min)(1.0f, n * gain));
        
        total += n * amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return total;
}

float billowNoise(float x, float y, int octaves, float persistence,
                  float lacunarity, int seed) {
    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float maxValue = 0.0f;
    
    for (int i = 0; i < octaves; i++) {
        float n = gradientNoise(x * frequency, y * frequency, seed + i);
        n = std::abs(n) * 2.0f - 1.0f;  // Billow transformation
        
        total += n * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return total / maxValue;
}

void generateHeightmap(float* output, int width, int height,
                       const FFTNoiseParams& params) {
    if (!output) return;
    
    float invWidth = 1.0f / static_cast<float>(width);
    float invHeight = 1.0f / static_cast<float>(height);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float nx = static_cast<float>(x) * invWidth;
            float ny = static_cast<float>(y) * invHeight;
            
            // Scale to noise space
            float sx = nx * params.scale / params.frequency;
            float sy = ny * params.scale / params.frequency;
            
            float value = 0.0f;
            
            switch (params.pattern) {
                case NoisePattern::OceanSpectrum:
                case NoisePattern::Turbulence:
                    value = fbmNoise(sx, sy, params.octaves, params.persistence,
                                    params.lacunarity, params.seed);
                    break;
                    
                case NoisePattern::Ridged:
                    value = ridgedNoise(sx, sy, params.octaves, params.persistence,
                                       params.lacunarity, params.ridgeOffset,
                                       params.ridgeSharpness, params.seed);
                    break;
                    
                case NoisePattern::Billow:
                    value = billowNoise(sx, sy, params.octaves, params.persistence,
                                       params.lacunarity, params.seed);
                    break;
                    
                case NoisePattern::Plateau: {
                    // Use FBM but with plateau shaping
                    float n = fbmNoise(sx, sy, params.octaves, params.persistence,
                                      params.lacunarity, params.seed);
                    // Apply plateau curve: steep sides, flat top
                    float threshold = 0.3f;
                    if (n > threshold) {
                        value = threshold + (n - threshold) * 0.2f;
                    } else {
                        value = n;
                    }
                    break;
                }
                    
                case NoisePattern::Custom:
                default:
                    value = fbmNoise(sx, sy, params.octaves, params.persistence,
                                    params.lacunarity, params.seed);
                    break;
            }
            
            // Apply amplitude
            value *= params.amplitude;
            
            output[y * width + x] = value;
        }
    }
}

} // namespace CPUNoise

} // namespace TerrainFFT
