/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          gas_fft_solver.cuh
* Author:        Kemal Demirtas
* Date:          January 2026
* Description:   Spectral Poisson Solver using cuFFT for Gas Simulation
*                10-50x faster than iterative solvers for large grids
* =========================================================================
*/
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

namespace FluidSim {

// ═══════════════════════════════════════════════════════════════════════════════
// FFT PRESSURE SOLVER STATE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief GPU resources for FFT-based Poisson pressure solver
 * 
 * Uses spectral method: P = FFT⁻¹( FFT(div) / eigenvalues )
 * This solves ∇²P = div exactly in O(N log N) time.
 * 
 * Much faster than iterative Gauss-Seidel/SOR for grids > 64³
 */
struct FFTPressureSolver {
    // cuFFT plans
    cufftHandle plan_forward;   // R2C: Real to Complex
    cufftHandle plan_inverse;   // C2R: Complex to Real
    
    // GPU buffers
    cufftComplex* d_spectrum;   // Complex spectrum buffer
    float* d_eigenvalues;       // Precomputed Laplacian eigenvalues
    
    // Grid dimensions
    int nx, ny, nz;
    int complex_size;           // (nx/2+1) * ny * nz
    
    // State
    bool initialized = false;
};

// ═══════════════════════════════════════════════════════════════════════════════
// GPU ADVANCED EMITTER STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief GPU-optimized advanced emitter for CUDA kernels
 * 
 * Features:
 * - Multiple shapes (Point, Sphere, Box, Cone, Cylinder, Disc)
 * - Smooth falloff (Linear, Smooth, Gaussian)
 * - Noise modulation for organic emission
 * - Velocity spray cone (randomized direction)
 * - Time-based emission profile
 */
struct GPUAdvancedEmitter {
    // ─────────────────────────────────────────────────────────────────────────
    // Shape & Transform
    // ─────────────────────────────────────────────────────────────────────────
    int shape;                  // 0=Point, 1=Sphere, 2=Box, 3=Cylinder, 4=Cone, 5=Disc
    int enabled;                // Boolean as int
    
    float pos_x, pos_y, pos_z;          // Position (world space)
    float rot_x, rot_y, rot_z;          // Rotation (radians)
    float scale_x, scale_y, scale_z;    // Scale
    
    // Shape dimensions
    float radius;               // For Sphere, Cylinder, Cone, Disc
    float size_x, size_y, size_z; // For Box (half-extents)
    float height;               // For Cylinder, Cone
    float inner_radius;         // For hollow shapes (Disc, Ring)
    float cone_angle;           // Opening angle for Cone (radians)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Emission Parameters
    // ─────────────────────────────────────────────────────────────────────────
    float density_rate;         // Density injection per second
    float temperature;          // Emission temperature (Kelvin)
    float fuel_rate;            // Fuel injection per second
    
    float vel_x, vel_y, vel_z;  // Base velocity direction
    float velocity_magnitude;   // Base speed
    
    // ─────────────────────────────────────────────────────────────────────────
    // Falloff
    // ─────────────────────────────────────────────────────────────────────────
    int falloff_type;           // 0=None, 1=Linear, 2=Smooth, 3=Gaussian
    float falloff_start;        // Where falloff begins (0-1 of radius)
    float falloff_end;          // Where emission ends (0-1 of radius)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Noise Modulation
    // ─────────────────────────────────────────────────────────────────────────
    int noise_enabled;          // Enable noise modulation
    float noise_frequency;      // Noise frequency
    float noise_amplitude;      // Noise strength (0-1)
    float noise_speed;          // Animation speed
    int noise_seed;             // Random seed for consistency
    
    int noise_modulate_density;     // Apply noise to density
    int noise_modulate_temperature; // Apply noise to temperature
    int noise_modulate_velocity;    // Apply noise to velocity
    
    // ─────────────────────────────────────────────────────────────────────────
    // Velocity Variance (Spray)
    // ─────────────────────────────────────────────────────────────────────────
    float spray_cone_angle;     // Spray cone half-angle (radians)
    float speed_min;            // Minimum speed multiplier
    float speed_max;            // Maximum speed multiplier
    
    // ─────────────────────────────────────────────────────────────────────────
    // Emission Profile (Time-based)
    // ─────────────────────────────────────────────────────────────────────────
    int emission_mode;          // 0=Continuous, 1=Burst, 2=Pulse
    float start_frame;          // When emission starts
    float end_frame;            // When emission ends (-1 = never)
    float pulse_interval;       // Frames between pulses
    float pulse_duration;       // Duration of each pulse
};

// Maximum emitters per simulation
#define MAX_GPU_ADVANCED_EMITTERS 32

// ═══════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Initialize FFT pressure solver
 * @return true on success
 */
bool initFFTPressureSolver(FFTPressureSolver& solver, int nx, int ny, int nz);

/**
 * @brief Cleanup FFT solver resources
 */
void cleanupFFTPressureSolver(FFTPressureSolver& solver);

/**
 * @brief Solve pressure using spectral method
 * Solves ∇²P = divergence in frequency domain
 * 
 * @param solver Initialized FFT solver
 * @param d_divergence Input divergence field
 * @param d_pressure Output pressure field (will be overwritten)
 */
void solvePressureFFT(
    FFTPressureSolver& solver,
    float* d_divergence,
    float* d_pressure
);

/**
 * @brief Apply advanced emitters to simulation grids
 * 
 * @param emitters Array of GPU emitter structures
 * @param emitter_count Number of active emitters
 * @param world_mat 4x4 column-major world matrix for gas volume
 * @param current_frame Current simulation frame (for time-based profiles)
 * @param time Accumulated simulation time (for noise animation)
 */
void cuda_apply_advanced_emitters(
    int nx, int ny, int nz, float dt,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    const GPUAdvancedEmitter* d_emitters,
    int emitter_count,
    const float* world_mat,
    float current_frame,
    float time
);

/**
 * @brief Upload advanced emitters to GPU
 * @return Device pointer (caller should NOT free, managed internally)
 */
GPUAdvancedEmitter* cuda_upload_advanced_emitters(const GPUAdvancedEmitter* h_emitters, int count);

/**
 * @brief Free GPU advanced emitter buffer
 */
void cuda_free_advanced_emitters(GPUAdvancedEmitter* d_emitters);

/**
 * @brief Convert Physics::AdvancedEmitter to GPUAdvancedEmitter
 * Helper function for CPU-side conversion before upload
 */
// Note: Implemented in GasSimulator.cpp to avoid Physics namespace dependency here

} // namespace FluidSim
