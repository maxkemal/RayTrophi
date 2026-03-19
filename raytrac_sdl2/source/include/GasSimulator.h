/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GasSimulator.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file GasSimulator.h
 * @brief Gas/Smoke simulation engine with CPU and CUDA backends
 * 
 * Features:
 * - Semi-Lagrangian advection for stable large timesteps
 * - Jacobi/Red-Black Gauss-Seidel pressure solver
 * - Buoyancy for rising hot smoke
 * - Vorticity confinement for turbulent detail
 * - Sphere/Box emitters
 * - VDB export support
 * 
 * Based on Jos Stam's "Stable Fluids" (1999) with modern enhancements.
 */

#include "FluidGrid.h"
#include "Vec3.h"
#include "Matrix4x4.h"
#include "json.hpp"
#include "KeyframeSystem.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>

// Forward declare FFT solver (avoid CUDA header in pure C++ headers)
namespace FluidSim {
    struct FFTPressureSolver;
    struct GPUAdvancedEmitter;
}

// Forward declarations
namespace openvdb { template<typename T> class Grid; }
namespace Physics { class ForceFieldManager; }

namespace FluidSim {

// ═══════════════════════════════════════════════════════════════════════════════
// ENUMS AND SETTINGS
// ═══════════════════════════════════════════════════════════════════════════════

enum class SimulationMode {
    RealTime,   // Simulate each frame on-the-fly
    Baked       // Pre-compute and cache frames
};

enum class SolverBackend {
    CPU,        // Multi-threaded CPU solver
    CUDA        // GPU-accelerated solver
};

enum class EmitterShape {
    Sphere,
    Box,
    Point,
    Cylinder,
    Cone,
    Disc
};

enum class EmitterFalloffType {
    None,
    Linear,
    Smooth,
    Gaussian
};

enum class EmitterEmissionMode {
    Continuous,
    Burst,
    Pulse
};

enum class FuelPhase {
    Gas,
    Liquid,
    Solid
};

/**
 * @brief Emitter configuration for density/temperature injection
 */
struct Emitter {
    EmitterShape shape = EmitterShape::Sphere;
    Vec3 position = Vec3(0, 0, 0);
    Vec3 size = Vec3(1, 1, 1);      // Radius for sphere, half-extents for box
    float radius = 1.0f;            // For sphere
    float height = 1.0f;            // For cylinder / cone
    float inner_radius = 0.0f;      // For disc
    float cone_angle = 45.0f;       // Degrees
    
    float density_rate = 2.5f;       // Density injection per second
    float fuel_rate = 0.0f;          // Fuel injection per second
    float temperature = 340.0f;      // Temperature in Kelvin
    Vec3 velocity = Vec3(0, 1.1f, 0);  // Initial velocity of emitted smoke
    FuelPhase fuel_phase = FuelPhase::Gas;
    float phase_change_temperature = 420.0f; // Liquid boiling / solid pyrolysis threshold
    float fuel_release_rate = 1.0f;          // Multiplier for vaporization / pyrolysis response
    float flame_contact_sensitivity = 0.35f; // How strongly nearby flame unlocks fuel release

    EmitterFalloffType falloff_type = EmitterFalloffType::Smooth;
    float falloff_start = 0.7f;
    float falloff_end = 1.0f;

    bool noise_enabled = false;
    float noise_frequency = 0.65f;
    float noise_amplitude = 0.22f;
    float noise_speed = 0.25f;
    int noise_seed = 42;
    bool noise_modulate_density = true;
    bool noise_modulate_temperature = false;
    bool noise_modulate_velocity = false;

    float spray_cone_angle = 0.0f;  // Degrees
    float speed_min = 1.0f;
    float speed_max = 1.0f;

    EmitterEmissionMode emission_mode = EmitterEmissionMode::Continuous;
    float start_frame = 0.0f;
    float end_frame = -1.0f;
    float pulse_interval = 10.0f;
    float pulse_duration = 3.0f;
    
    bool enabled = true;
    std::string name = "Emitter";
    uint32_t uid = 0;               // Unique ID for animation tracks (NEW)
    
    // =========================================================================
    // KEYFRAME ANIMATION SUPPORT
    // =========================================================================
    std::map<int, ::EmitterKeyframe> keyframes;  // frame -> keyframe data
    
    /// @brief Get interpolated values at specific time/frame
    ::EmitterKeyframe getInterpolatedKeyframe(float current_frame) const;
    
    /// @brief Apply interpolated keyframe to this emitter
    void applyKeyframe(const ::EmitterKeyframe& kf);
    
    // Serialization
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

/**
 * @brief Complete simulation settings
 */
struct GasSimulationSettings {
    // ─────────────────────────────────────────────────────────────────────────
    // Grid settings
    // ─────────────────────────────────────────────────────────────────────────
    int resolution_x = 64;
    int resolution_y = 64;
    int resolution_z = 64;
    Vec3 grid_size = Vec3(5, 5, 5); // Total world size of the grid (Fixed Bounding Box)
    float voxel_size = 0.1f;        // Calculated automatically: grid_size / resolution
    bool preserve_voxel_size_on_resize = true;
    int max_auto_resolution = 256;
    Vec3 grid_offset = Vec3(0, 0, 0); // Grid origin in world space
    
    // ─────────────────────────────────────────────────────────────────────────
    // Simulation parameters
    // ─────────────────────────────────────────────────────────────────────────
    float timestep = 0.016f;         // Simulation timestep (default ~60 FPS)
    int substeps = 1;                // Substeps per frame for stability
    float time_scale = 0.45f;        // Simulation speed multiplier
    int pressure_iterations = 40;    // Jacobi iterations (more = more accurate)
    
    // CFL Adaptive Timestep (Industry Standard)
    bool adaptive_timestep = true;   // Enable automatic timestep adjustment
    float cfl_number = 0.4f;         // CFL condition (0.5-1.0, lower = more stable)
    float min_timestep = 0.001f;     // Minimum timestep (prevents simulation halt)
    float max_timestep = 0.05f;      // Maximum timestep (prevents too large steps)
    
    float density_dissipation = 0.994f;    // Density decay per second
    float velocity_dissipation = 0.985f;   // Velocity damping (lower = more drag)
    float temperature_dissipation = 0.985f; // Temperature cooling
    float fuel_dissipation = 0.985f;       // Fuel decay per second

    // ─────────────────────────────────────────────────────────────────────────
    // Combustion parameters
    // ─────────────────────────────────────────────────────────────────────────
    float ignition_temperature = 420.0f;   // Temp at which fuel starts burning
    float burn_rate = 0.55f;               // Speed of fuel consumption
    float heat_release = 18.0f;            // Temperature added per unit fuel burned
    float expansion_strength = 0.8f;       // Expansion (pressure) from fire
    float smoke_generation = 0.25f;        // Smoke generated per unit fuel burned
    float soot_generation = 0.04f;         // Density generated from burning
    
    // ─────────────────────────────────────────────────────────────────────────
    // Physical forces
    // ─────────────────────────────────────────────────────────────────────────
    float buoyancy_density = -0.18f;      // Density buoyancy (negative = sink)
    float buoyancy_temperature = 2.2f;    // Temperature buoyancy (hot rises)
    float ambient_temperature = 293.0f;   // Ambient temp in Kelvin (~20°C)
    
    float vorticity_strength = 0.32f;     // Vorticity confinement (detail)
    float turbulence_strength = 0.08f;    // Added turbulence noise strength
    float turbulence_scale = 1.2f;        // Turbulence noise scale
    int   turbulence_octaves = 3;         // FBM octaves (1-8, more = more detail)
    float turbulence_lacunarity = 2.0f;   // Frequency multiplier per octave
    float turbulence_persistence = 0.5f;  // Amplitude decay per octave
    
    enum class AdvectionMode {
        SemiLagrangian,
        MacCormack,
        BFECC
    };
    AdvectionMode advection_mode = AdvectionMode::MacCormack;
    
    // Pressure Solver Mode (Industry Standard)
    enum class PressureSolverMode {
        GaussSeidel,      // Basic Red-Black Gauss-Seidel
        SOR,              // Successive Over-Relaxation (2-3x faster convergence)
        Multigrid,        // Multigrid V-cycle (future implementation)
        FFT               // Spectral solver using cuFFT (10-50x faster!)
    };
    enum class BoundaryMode {
        Open,             // Best for plume / smoke escaping the domain
        Periodic          // Looping domain; matches the current FFT spectral solver
    };
    PressureSolverMode pressure_solver = PressureSolverMode::SOR;
    BoundaryMode boundary_mode = BoundaryMode::Open;
    float sor_omega = 1.7f;  // SOR relaxation factor (optimal ~1.7 for 3D Poisson)

    Vec3 gravity = Vec3(0, -9.81f, 0);
    Vec3 wind = Vec3(0, 0, 0);

    // Sparse Grid Settings (VDB-style optimization)
    bool sparse_mode = true;          // Enable sparse tile processing
    float sparse_threshold = 0.001f;  // Minimum density to consider tile active
    
    // ─────────────────────────────────────────────────────────────────────────
    // Stability Limits (GPU clamping)
    // ─────────────────────────────────────────────────────────────────────────
    float max_velocity = 300.0f;      // Maximum velocity (grid units/s)
    float max_temperature = 6000.0f;  // Maximum temperature (K) - Support white-hot
    float max_density = 40.0f;        // Maximum density
    
    // ─────────────────────────────────────────────────────────────────────────
    // Mode settings
    // ─────────────────────────────────────────────────────────────────────────
    SimulationMode mode = SimulationMode::RealTime;
    SolverBackend backend = SolverBackend::CUDA;  // Prioritize GPU if available
    
    // ─────────────────────────────────────────────────────────────────────────
    // Baking settings
    // ─────────────────────────────────────────────────────────────────────────
    std::string cache_directory = "";
    int bake_start_frame = 0;
    int bake_end_frame = 100;
    float bake_fps = 30.0f;
    
    // Serialization
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

/**
 * @brief Main gas simulation engine
 * 
 * Usage:
 * @code
 * GasSimulator sim;
 * sim.initialize(settings);
 * sim.addEmitter(emitter);
 * 
 * // Game loop
 * while (running) {
 *     sim.step(dt);
 *     // Use sim.getGrid() for rendering
 * }
 * @endcode
 */
class GasSimulator {
public:
    struct StateSnapshot {
        FluidGrid grid;
        FluidGrid grid_temp;
        std::vector<Vec3> persistent_vorticity;
        int current_frame = 0;
        float accumulated_time = 0.0f;
        bool valid = false;
    };

    GasSimulator();
    ~GasSimulator();
    
    // ═══════════════════════════════════════════════════════════════════════
    // LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Initialize with settings
    void initialize(const GasSimulationSettings& settings);
    
    /// @brief Advance simulation by dt seconds
    void step(float dt, const Matrix4x4& world_matrix = Matrix4x4::identity());
    
    /// @brief Reset simulation to initial state
    void reset();
    
    /// @brief Cleanup resources
    void shutdown();
    
    // ═══════════════════════════════════════════════════════════════════════
    // EMITTERS
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Add an emitter
    int addEmitter(const Emitter& emitter);
    
    /// @brief Remove emitter by index
    void removeEmitter(int index);
    
    /// @brief Get all emitters
    std::vector<Emitter>& getEmitters() { return emitters; }
    const std::vector<Emitter>& getEmitters() const { return emitters; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // GRID ACCESS
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Get the fluid grid (for rendering/export)
    FluidGrid& getGrid() { return grid; }
    const FluidGrid& getGrid() const { return grid; }
    
    /// @brief Get settings
    GasSimulationSettings& getSettings() { return settings; }
    const GasSimulationSettings& getSettings() const { return settings; }
    
    /// @brief Set external force field manager (from SceneData)
    void setExternalForceFieldManager(const Physics::ForceFieldManager* manager) {
        external_force_field_manager = manager;
    }
    
    /// @brief Get external force field manager
    const Physics::ForceFieldManager* getExternalForceFieldManager() const {
        return external_force_field_manager;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // SAMPLING (for rendering)
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Sample density at world position
    float sampleDensity(const Vec3& world_pos) const;
    
    /// @brief Sample temperature at world position  
    float sampleTemperature(const Vec3& world_pos) const;
    
    /// @brief Sample flame intensity (combustion/interaction) at world position
    float sampleFlameIntensity(const Vec3& world_pos) const;
    
    /// @brief Sample fuel at world position
    float sampleFuel(const Vec3& world_pos) const;
    
    /// @brief Sample velocity at world position
    Vec3 sampleVelocity(const Vec3& world_pos) const;
    
    // ═══════════════════════════════════════════════════════════════════════
    // BAKING
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Start baking simulation to cache
    void startBake(int start_frame, int end_frame, const std::string& cache_dir, const Matrix4x4& world_matrix = Matrix4x4::identity());
    
    /// @brief Cancel ongoing bake
    void cancelBake();
    
    /// @brief Is baking in progress?
    bool isBaking() const { return is_baking; }
    
    /// @brief Get bake progress (0.0 - 1.0)
    float getBakeProgress() const { return bake_progress; }

    /// @brief Get currently baking frame
    int getBakingFrame() const { return baking_frame; }
    
    /// @brief Load baked frame
    bool loadBakedFrame(int frame);
    

    /// @brief Export current frame to VDB file
    bool exportToVDB(const std::string& filepath) const;
    
    /// @brief Export frame sequence to VDB
    bool exportSequenceToVDB(const std::string& directory, 
                             const std::string& base_name,
                             int start_frame, int end_frame,
                             const Matrix4x4& world_matrix = Matrix4x4::identity());
    

    
    float getTotalDensity() const { return grid.getTotalDensity(); }
    float getMaxDensity() const { return grid.getMaxDensity(); }
    float getMaxVelocity() const { return grid.getMaxVelocity(); }
    int getActiveVoxelCount() const { return grid.getActiveVoxelCount(); }
    
    float getLastStepTime() const { return last_step_time_ms; }
    int getCurrentFrame() const { return current_frame; }
    uint32_t emitter_id_counter = 1; // Counter for unique emitter IDs (NEW)
    
  
    /// @brief Get GPU density data pointer (CUDA)
    void* getGPUDensityPtr() const { return d_density; }
    
    /// @brief Get GPU temperature data pointer (CUDA)
    void* getGPUTemperaturePtr() const { return d_temperature; }
    
    /// @brief Is GPU data valid/up-to-date?
    bool isGPUDataValid() const { return gpu_data_valid; }
    bool isUsingCUDA() const { return settings.backend == SolverBackend::CUDA && cuda_initialized; }
    bool isCUDAInitialized() const { return cuda_initialized; }
    bool canUseFFTPressureSolver() const;

    /// @brief Setup a preset (Fire, Smoke, Explosion)
    void applyPreset(const std::string& presetName);
    
    /// @brief Upload current grid to GPU
    void uploadToGPU();
    
    /// @brief Download from GPU to CPU grid
    void downloadFromGPU();
    StateSnapshot captureState() const;
    void restoreState(const StateSnapshot& snapshot);

private:
    // ═══════════════════════════════════════════════════════════════════════
    // SIMULATION STEPS (CPU)
    // ═══════════════════════════════════════════════════════════════════════
    
    void advectVelocity(float dt);
    void advectScalars(float dt);
    void applyForces(float dt, const Matrix4x4& world_matrix);
    void applyGravity(float dt);               // Gravity from settings.gravity
    void applyBuoyancy(float dt);
    void applyVorticity(float dt);
    void applyCurlNoiseTurbulence(float dt);   // Industry-standard curl noise turbulence
    void applyWind(float dt);
    void applyVelocityClamping();              // Stability limit: settings.max_velocity
    void applyExternalForceFields(float dt, const Matrix4x4& world_matrix); // Accept world transform
    void solvePressure();
    void project();
    void applyEmitters(float dt);
    void applyDissipation(float dt);
    void processCombustion(float dt); // Fuel + Heat -> Fire
    void enforceBoundaries();
    
    // CFL Adaptive Timestep (Industry Standard)
    float computeCFLTimestep(float requested_dt);  // Returns stable timestep based on max velocity
    float computeMaxVelocity();                    // Scans grid for maximum velocity magnitude
   
    
    void stepCUDA(float dt, const Matrix4x4& world_matrix);
    void initCUDA();
    void freeCUDA();
    void clearGPU();  // Clear GPU buffers without reallocating
   
    
    FluidGrid grid;
    FluidGrid grid_temp;  // Temporary buffer for advection
    GasSimulationSettings settings;
    std::vector<Emitter> emitters;
    
    // External force fields (scene-level)
    const Physics::ForceFieldManager* external_force_field_manager = nullptr;
    
    // State
    int current_frame = 0;
    float accumulated_time = 0.0f;
    float last_step_time_ms = 0.0f;
    int smoothed_adaptive_chunk_count = 1;
    bool initialized = false;
    
    // Baking
    std::atomic<bool> is_baking{false};
    std::atomic<float> bake_progress{0.0f};
    std::atomic<bool> cancel_bake{false};
    std::atomic<int> baking_frame{0};
    std::unique_ptr<std::thread> bake_thread;
    
    // GPU resources
    void* d_density = nullptr;
    void* d_temperature = nullptr;
    void* d_fuel = nullptr;
    void* d_vel_x = nullptr;
    void* d_vel_y = nullptr;
    void* d_vel_z = nullptr;
    void* d_pressure = nullptr;
    void* d_divergence = nullptr;
    void* d_vort_x = nullptr;
    void* d_vort_y = nullptr;
    void* d_vort_z = nullptr;
    void* d_tmp1 = nullptr;    // Temporary GPU buffer for advection/pressure
    void* d_tmp2 = nullptr;    // Temporary GPU buffer
    void* d_tmp3 = nullptr;    // Temporary GPU buffer
    bool gpu_data_valid = false;
    bool cuda_initialized = false;
    
    // GPU Force Field resources
    void* d_force_fields = nullptr;  // GPU force field buffer
    int gpu_force_field_count = 0;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FFT PRESSURE SOLVER (10-50x faster than iterative for large grids)
    // ═══════════════════════════════════════════════════════════════════════════
    FFTPressureSolver* fft_solver = nullptr;  // Spectral Poisson solver
    bool use_fft_solver = false;              // Auto-enabled when mode=FFT
    
    void initFFTSolver();
    void cleanupFFTSolver();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU ADVANCED EMITTERS
    // ═══════════════════════════════════════════════════════════════════════════
    GPUAdvancedEmitter* d_advanced_emitters = nullptr;
    int gpu_advanced_emitter_count = 0;
    
    void uploadAdvancedEmittersToGPU();
    void freeGPUAdvancedEmitters();
    
    // CPU persistent buffers to avoid allocations
    std::vector<Vec3> persistent_vorticity;
    
    // Helper: Upload force fields to GPU for current frame
    void uploadForceFieldsToGPU();
    void freeGPUForceFields();
    
};

} // namespace FluidSim

