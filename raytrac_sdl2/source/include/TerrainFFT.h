#pragma once
/**
 * @file TerrainFFT.h
 * @brief FFT-based procedural terrain generation with CUDA acceleration
 * 
 * Features:
 * - Dynamic CUDA DLL loading (no crash if CUDA not available)
 * - CPU fallback for systems without CUDA
 * - Multiple noise patterns (ocean spectrum, turbulence, ridged, etc.)
 * - Thread-safe singleton design
 */

#include <vector>
#include <cmath>
#include <memory>
#include <string>
#include <functional>
#include <random>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#endif
#include <windows.h>
#endif

namespace TerrainFFT {

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr float PI = 3.14159265359f;
constexpr float TWO_PI = 6.28318530718f;

// ============================================================================
// FFT NOISE PARAMETERS
// ============================================================================

/**
 * @brief Types of procedural noise that can be generated via FFT
 */
enum class NoisePattern {
    OceanSpectrum,      // Phillips spectrum (realistic ocean-like terrain)
    Turbulence,         // Fractal turbulence (rough terrain)
    Ridged,             // Ridge/mountain chains
    Billow,             // Soft rolling hills
    Plateau,            // Flat-topped with steep edges
    Custom              // User-defined spectrum
};

/**
 * @brief Parameters for FFT-based noise generation
 */
struct FFTNoiseParams {
    // Grid settings
    int resolution = 256;           // FFT grid size (power of 2)
    float scale = 1.0f;             // World scale
    
    // Pattern settings
    NoisePattern pattern = NoisePattern::OceanSpectrum;
    
    // Common parameters
    float amplitude = 1.0f;         // Height multiplier
    float frequency = 0.01f;        // Base frequency
    int octaves = 6;                // Number of octaves for fractal patterns
    float persistence = 0.5f;       // Amplitude falloff per octave
    float lacunarity = 2.0f;        // Frequency multiplier per octave
    
    // Ocean spectrum specific
    float windSpeed = 15.0f;        // For Phillips spectrum
    float windDirection = 0.0f;     // Wind direction in radians
    float choppiness = 1.2f;        // Horizontal displacement
    
    // Ridge/mountain specific
    float ridgeOffset = 1.0f;       // For ridged noise
    float ridgeSharpness = 2.0f;
    
    // Seed for reproducibility
    int seed = 12345;
    
    // Helper
    void computeDerived() {
        // Nothing yet
    }
};

/**
 * @brief Result of FFT noise generation
 */
struct FFTNoiseResult {
    std::vector<float> heightmap;   // Generated heights
    int width = 0;
    int height = 0;
    bool success = false;
    bool usedGPU = false;           // True if GPU was used
    std::string errorMessage;
};

// ============================================================================
// CUDA DLL FUNCTION TYPES
// ============================================================================

// Forward declare internal CUDA state
struct CUDAFFTState;

// Function pointer types for dynamically loaded CUDA functions
typedef int  (*PFN_CudaGetDeviceCount)(int*);
typedef int  (*PFN_CudaSetDevice)(int);
typedef int  (*PFN_CudaMalloc)(void**, size_t);
typedef int  (*PFN_CudaFree)(void*);
typedef int  (*PFN_CudaMemcpy)(void*, const void*, size_t, int);
typedef int  (*PFN_CudaDeviceSynchronize)();

// ============================================================================
// TERRAIN FFT MANAGER (SINGLETON)
// ============================================================================

/**
 * @brief Manages FFT-based terrain generation with optional CUDA acceleration
 * 
 * Automatically detects CUDA availability at runtime.
 * Falls back to CPU-based generation if CUDA is unavailable.
 */
class TerrainFFTManager {
public:
    static TerrainFFTManager& getInstance() {
        static TerrainFFTManager instance;
        return instance;
    }
    
    // Deleted copy/move
    TerrainFFTManager(const TerrainFFTManager&) = delete;
    TerrainFFTManager& operator=(const TerrainFFTManager&) = delete;
    
    // ========================================================================
    // CUDA STATUS
    // ========================================================================
    
    /**
     * @brief Check if CUDA is available for GPU acceleration
     */
    bool isCUDAAvailable() const { return cudaAvailable_; }
    
    /**
     * @brief Get CUDA status message
     */
    const std::string& getCUDAStatus() const { return cudaStatusMessage_; }
    
    /**
     * @brief Force use of CPU even if CUDA is available
     */
    void setForceCPU(bool force) { forceCPU_ = force; }
    bool isForcingCPU() const { return forceCPU_; }
    
    // ========================================================================
    // NOISE GENERATION
    // ========================================================================
    
    /**
     * @brief Generate FFT-based noise heightmap
     * 
     * Uses GPU if available, falls back to CPU otherwise.
     * 
     * @param params Noise generation parameters
     * @return Result containing heightmap data
     */
    FFTNoiseResult generateNoise(const FFTNoiseParams& params);
    
    /**
     * @brief Generate noise directly into an existing buffer
     * 
     * @param params Noise parameters
     * @param output Pre-allocated output buffer (must be params.resolution^2)
     * @return true on success
     */
    bool generateNoiseInPlace(const FFTNoiseParams& params, float* output);
    
    // ========================================================================
    // SPECTRUM UTILITIES
    // ========================================================================
    
    /**
     * @brief Phillips spectrum value for ocean-like terrain
     */
    static float phillipsSpectrum(float kx, float ky, float windSpeed, 
                                  float windDirX, float windDirY, float amplitude);
    
    /**
     * @brief Power spectrum for fractal terrain
     */
    static float powerSpectrum(float k, float exponent = 2.0f);
    
private:
    TerrainFFTManager();
    ~TerrainFFTManager();
    
    // CUDA initialization
    bool initCUDA();
    void cleanupCUDA();
    
    // CPU fallback implementation
    FFTNoiseResult generateNoiseCPU(const FFTNoiseParams& params);
    
    // GPU implementation
    FFTNoiseResult generateNoiseGPU(const FFTNoiseParams& params);
    
    // State
    bool cudaAvailable_ = false;
    bool cudaInitialized_ = false;
    bool forceCPU_ = false;
    std::string cudaStatusMessage_ = "Not initialized";
    
    // Dynamic library handles
#ifdef _WIN32
    HMODULE cudartDLL_ = nullptr;
    HMODULE cufftDLL_ = nullptr;
#else
    void* cudartDLL_ = nullptr;
    void* cufftDLL_ = nullptr;
#endif
    
    // Function pointers
    PFN_CudaGetDeviceCount pfnCudaGetDeviceCount_ = nullptr;
    PFN_CudaSetDevice pfnCudaSetDevice_ = nullptr;
    PFN_CudaMalloc pfnCudaMalloc_ = nullptr;
    PFN_CudaFree pfnCudaFree_ = nullptr;
    PFN_CudaMemcpy pfnCudaMemcpy_ = nullptr;
    PFN_CudaDeviceSynchronize pfnCudaDeviceSynchronize_ = nullptr;
    
    // Internal CUDA state
    std::unique_ptr<CUDAFFTState> cudaState_;
    
    // Random generator for CPU fallback
    std::mt19937 rng_;
};

// ============================================================================
// CPU NOISE UTILITIES (Always available)
// ============================================================================

namespace CPUNoise {
    
    /**
     * @brief Simple gradient noise (Perlin-like)
     */
    float gradientNoise(float x, float y, int seed = 0);
    
    /**
     * @brief FBM (Fractal Brownian Motion) noise
     */
    float fbmNoise(float x, float y, int octaves, float persistence, 
                   float lacunarity, int seed = 0);
    
    /**
     * @brief Ridged multifractal noise (for mountains)
     */
    float ridgedNoise(float x, float y, int octaves, float persistence,
                      float lacunarity, float offset, float gain, int seed = 0);
    
    /**
     * @brief Billow noise (soft rolling hills)
     */
    float billowNoise(float x, float y, int octaves, float persistence,
                      float lacunarity, int seed = 0);
    
    /**
     * @brief Generate complete heightmap using CPU noise
     */
    void generateHeightmap(float* output, int width, int height,
                           const FFTNoiseParams& params);
    
} // namespace CPUNoise

} // namespace TerrainFFT
