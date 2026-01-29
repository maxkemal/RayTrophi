#pragma once
#include <cuda_runtime.h>

namespace FluidSim {

// ═══════════════════════════════════════════════════════════════════════════════
// GPU SIMULATION PARAMETERS - All settings passed from UI to GPU
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Complete simulation parameters for GPU
 * Replaces hardcoded values with UI-driven settings
 */
struct GPUSimulationParams {
    // ─────────────────────────────────────────────────────────────────────────
    // Grid Info
    // ─────────────────────────────────────────────────────────────────────────
    int nx, ny, nz;
    float voxel_size;
    float inv_voxel_size;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Timestep
    // ─────────────────────────────────────────────────────────────────────────
    float dt;
    float time;                     // Accumulated simulation time
    
    // ─────────────────────────────────────────────────────────────────────────
    // Advection
    // ─────────────────────────────────────────────────────────────────────────
    int advection_mode;             // 0=SemiLagrangian, 1=MacCormack, 2=BFECC
    
    // ─────────────────────────────────────────────────────────────────────────
    // Forces & Buoyancy
    // ─────────────────────────────────────────────────────────────────────────
    float buoyancy_density;         // Alpha - density buoyancy coefficient
    float buoyancy_temperature;     // Beta - temperature buoyancy coefficient
    float ambient_temperature;
    
    float gravity_x, gravity_y, gravity_z;  // Gravity vector (world space)
    float wind_x, wind_y, wind_z;           // Wind vector (already in grid units)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Dissipation (decay factors per frame, already computed as pow(rate, dt))
    // ─────────────────────────────────────────────────────────────────────────
    float density_dissipation;
    float velocity_dissipation;
    float temperature_dissipation;
    float fuel_dissipation;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Combustion
    // ─────────────────────────────────────────────────────────────────────────
    float ignition_temperature;
    float burn_rate;
    float heat_release;
    float smoke_generation;
    float expansion_strength;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Vorticity & Turbulence
    // ─────────────────────────────────────────────────────────────────────────
    float vorticity_strength;
    float turbulence_strength;
    float turbulence_scale;
    int   turbulence_octaves;       // FBM octaves (1-8)
    float turbulence_lacunarity;    // Frequency multiplier per octave (default 2.0)
    float turbulence_persistence;   // Amplitude decay per octave (default 0.5)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Pressure Solver
    // ─────────────────────────────────────────────────────────────────────────
    int pressure_iterations;
    int pressure_solver_mode;       // 0=GaussSeidel, 1=SOR, 2=FFT
    float sor_omega;                // SOR relaxation factor (1.0-2.0, optimal ~1.7)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Stability Limits (UI configurable instead of hardcoded)
    // ─────────────────────────────────────────────────────────────────────────
    float max_velocity;             // Maximum velocity clamp (grid units/s)
    float max_temperature;          // Maximum temperature clamp (K)
    float max_density;              // Maximum density clamp
    
    // ─────────────────────────────────────────────────────────────────────────
    // Sparse Grid (future optimization)
    // ─────────────────────────────────────────────────────────────────────────
    int sparse_mode;                // 0=disabled, 1=enabled
    float sparse_threshold;         // Minimum density to consider active
};

// CUDA Kernels wrapper functions
void cuda_init_simulation(int nx, int ny, int nz, 
                         float** d_rho, float** d_temp, float** d_fuel,
                         float** d_vx, float** d_vy, float** d_vz,
                         float** d_p, float** d_div,
                         float** d_vort_x, float** d_vort_y, float** d_vort_z);

void cuda_free_simulation(float* d_rho, float* d_temp, float* d_fuel,
                         float* d_vx, float* d_vy, float* d_vz,
                         float* d_p, float* d_div,
                         float* d_vort_x, float* d_vort_y, float* d_vort_z);

void cuda_upload_data(int nx, int ny, int nz,
                     float* h_rho, float* h_temp, float* h_fuel,
                     float* h_vx, float* h_vy, float* h_vz,
                     float* d_rho, float* d_temp, float* d_fuel,
                     float* d_vx, float* d_vy, float* d_vz);

void cuda_download_data(int nx, int ny, int nz,
                       float* d_rho, float* d_temp, float* d_fuel,
                       float* d_vx, float* d_vy, float* d_vz,
                       float* h_rho, float* h_temp, float* h_fuel,
                       float* h_vx, float* h_vy, float* h_vz);

/**
 * @brief Main simulation step using unified parameters struct
 * All settings come from UI through GPUSimulationParams
 */
void cuda_step_simulation_v2(
    const GPUSimulationParams& params,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_p, float* d_div,
    float* d_vort_x, float* d_vort_y, float* d_vort_z,
    float* d_tmp1, float* d_tmp2, float* d_tmp3
);

// Legacy function (deprecated - use cuda_step_simulation_v2)
void cuda_step_simulation(
    int nx, int ny, int nz, float dt,
    float* d_rho, float* d_temp, float* d_fuel,
    float* d_vx, float* d_vy, float* d_vz,
    float* d_p, float* d_div,
    float* d_vort_x, float* d_vort_y, float* d_vort_z,
    // Temporary Buffers (Passed from class to avoid malloc)
    float* d_tmp1, float* d_tmp2, float* d_tmp3,
    // Settings
    float vorticity_str, float turbulence_str, float turbulence_scale,
    int advection_mode,
    float buoyancy_alpha, float buoyancy_beta,
    float ambient_temp, float density_dissipation, float vel_dissipation,
    float temp_dissipation, float fuel_dissipation,
    float ignition_temp, float burn_rate, float heat_release, 
    float smoke_gen, float expansion,
    int pressure_iters,
    // External forces
    float wind_x, float wind_y, float wind_z
);

void cuda_apply_emitter(int nx, int ny, int nz, float dt,
                        float* d_rho, float* d_temp, float* d_fuel,
                        float* d_vx, float* d_vy, float* d_vz,
                        int shape_type, float cx, float cy, float cz,
                        float sx, float sy, float sz,
                        float density_rate, float temp, float fuel_rate,
                        float emit_vx, float emit_vy, float emit_vz);

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD GPU SUPPORT
// ═══════════════════════════════════════════════════════════════════════════════

// Maximum force fields that can be uploaded to GPU
#define MAX_GPU_FORCE_FIELDS 16

/**
 * @brief GPU-friendly Force Field structure
 * All data needed for GPU evaluation packed into POD struct
 */
struct GPUForceField {
    // Type (matches ForceFieldType enum)
    int type;           // 0=Wind, 1=Gravity, 2=Attractor, 3=Repeller, 4=Vortex, 5=Turbulence, 6=CurlNoise, 7=Drag
    int shape;          // 0=Infinite, 1=Sphere, 2=Box, 3=Cylinder, 4=Cone
    int falloff_type;   // 0=None, 1=Linear, 2=Smooth, 3=Sphere, 4=InverseSquare, 5=Exponential
    int enabled;        // Boolean as int for GPU
    
    // Transform (world space)
    float pos_x, pos_y, pos_z;
    float rot_x, rot_y, rot_z;      // Euler angles in radians
    float scale_x, scale_y, scale_z;
    
    // Force parameters
    float strength;
    float dir_x, dir_y, dir_z;      // Direction for Wind/Gravity
    
    // Falloff
    float falloff_radius;
    float inner_radius;
    
    // Vortex-specific
    float axis_x, axis_y, axis_z;
    float inward_force;
    
    // Noise (for Turbulence)
    float noise_frequency;
    float noise_amplitude;
    float noise_speed;
    int noise_octaves;
    
    // Drag
    float drag_coefficient;
};

/**
 * @brief Apply external force fields to velocity grid
 * @param world_mat 4x4 column-major matrix for gas volume transform
 * @param force_fields Array of force field data
 * @param num_fields Number of active force fields
 */
void cuda_apply_force_fields(
    int nx, int ny, int nz, float dt,
    float* d_vx, float* d_vy, float* d_vz,
    const float* world_mat,             // 16 floats, column-major
    const GPUForceField* d_force_fields,
    int num_fields,
    float time
);

/**
 * @brief Upload force field data to GPU
 * @return Device pointer to force field array (caller should NOT free this)
 */
GPUForceField* cuda_upload_force_fields(const GPUForceField* h_fields, int count);

/**
 * @brief Free GPU force field buffer
 */
void cuda_free_force_fields(GPUForceField* d_fields);

} // namespace FluidSim
