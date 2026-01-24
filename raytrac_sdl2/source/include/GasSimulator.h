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

// Forward declarations
namespace openvdb { template<typename T> class Grid; }
namespace Physics { class ForceFieldManager; }

namespace FluidSim {

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// ENUMS AND SETTINGS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

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
    Point
};

/**
 * @brief Emitter configuration for density/temperature injection
 */
struct Emitter {
    EmitterShape shape = EmitterShape::Sphere;
    Vec3 position = Vec3(0, 0, 0);
    Vec3 size = Vec3(1, 1, 1);      // Radius for sphere, half-extents for box
    float radius = 1.0f;            // For sphere
    
    float density_rate = 10.0f;      // Density injection per second
    float fuel_rate = 0.0f;         // Fuel injection per second
    float temperature = 500.0f;      // Temperature in Kelvin
    Vec3 velocity = Vec3(0, 2, 0);  // Initial velocity of emitted smoke
    
    bool enabled = true;
    std::string name = "Emitter";
    uint32_t uid = 0;               // Unique ID for animation tracks (NEW)
    
    // ═════════════════════════════════════════════════════════════════════════
    // KEYFRAME ANIMATION SUPPORT
    // ═════════════════════════════════════════════════════════════════════════
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
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Grid settings
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    int resolution_x = 64;
    int resolution_y = 64;
    int resolution_z = 64;
    Vec3 grid_size = Vec3(5, 5, 5); // Total world size of the grid (Fixed Bounding Box)
    float voxel_size = 0.1f;        // Calculated automatically: grid_size / resolution
    Vec3 grid_offset = Vec3(0, 0, 0); // Grid origin in world space
    
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Simulation parameters
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    float timestep = 0.016f;         // Simulation timestep (default ~60 FPS)
    int substeps = 1;                // Substeps per frame for stability
    int pressure_iterations = 40;    // Jacobi iterations (more = more accurate)
    
    float density_dissipation = 0.995f;    // Density decay per second (0.99 = slow)
    float velocity_dissipation = 0.998f;   // Velocity damping
    float temperature_dissipation = 0.99f; // Temperature cooling
    float fuel_dissipation = 0.99f;        // Fuel decay per second

    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Combustion parameters
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    float ignition_temperature = 400.0f;   // Temp at which fuel starts burning
    float burn_rate = 1.0f;                // Speed of fuel consumption
    float heat_release = 100.0f;           // Temperature added per unit fuel burned
    float expansion_strength = 2.0f;       // Expansion (pressure) from fire
    float smoke_generation = 0.5f;         // Smoke generated per unit fuel burned
    float soot_generation = 0.1f;          // Density generated from burning
    
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Physical forces
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    float buoyancy_density = -0.5f;       // Density buoyancy (negative = sink)
    float buoyancy_temperature = 1.0f;    // Temperature buoyancy (hot rises)
    float ambient_temperature = 293.0f;   // Ambient temp in Kelvin (~20Â°C)
    
    float vorticity_strength = 0.5f;      // Vorticity confinement (detail)
    Vec3 gravity = Vec3(0, -9.81f, 0);
    Vec3 wind = Vec3(0, 0, 0);
    
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Mode settings
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    SimulationMode mode = SimulationMode::RealTime;
    SolverBackend backend = SolverBackend::CPU;
    
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // Baking settings
    // â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    GasSimulator();
    ~GasSimulator();
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // LIFECYCLE
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Initialize with settings
    void initialize(const GasSimulationSettings& settings);
    
    /// @brief Advance simulation by dt seconds
    void step(float dt, const Matrix4x4& world_matrix = Matrix4x4::identity());
    
    /// @brief Reset simulation to initial state
    void reset();
    
    /// @brief Cleanup resources
    void shutdown();
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // EMITTERS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Add an emitter
    int addEmitter(const Emitter& emitter);
    
    /// @brief Remove emitter by index
    void removeEmitter(int index);
    
    /// @brief Get all emitters
    std::vector<Emitter>& getEmitters() { return emitters; }
    const std::vector<Emitter>& getEmitters() const { return emitters; }
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // GRID ACCESS
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
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
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // SAMPLING (for rendering)
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    /// @brief Sample density at world position
    float sampleDensity(const Vec3& world_pos) const;
    
    /// @brief Sample temperature at world position  
    float sampleTemperature(const Vec3& world_pos) const;
    
    /// @brief Sample velocity at world position
    Vec3 sampleVelocity(const Vec3& world_pos) const;
    
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // BAKING
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
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

    /// @brief Setup a preset (Fire, Smoke, Explosion)
    void applyPreset(const std::string& presetName);
    
    /// @brief Upload current grid to GPU
    void uploadToGPU();
    
    /// @brief Download from GPU to CPU grid
    void downloadFromGPU();

private:
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // SIMULATION STEPS (CPU)
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    
    void advectVelocity(float dt);
    void advectScalars(float dt);
    void applyForces(float dt, const Matrix4x4& world_matrix);
    void applyBuoyancy(float dt);
    void applyVorticity(float dt);
    void applyWind(float dt);
    void applyExternalForceFields(float dt, const Matrix4x4& world_matrix); // UPDATED: Accept world transform
    void solvePressure();
    void project();
    void applyEmitters(float dt);
    void applyDissipation(float dt);
    void processCombustion(float dt); // NEW: Fuel + Heat -> Fire
    void enforceBoundaries();
   
    
    void stepCUDA(float dt, const Matrix4x4& world_matrix);
    void initCUDA();
    void freeCUDA();
   
    
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
    bool gpu_data_valid = false;
    bool cuda_initialized = false;
    
    // CPU persistent buffers to avoid allocations
    std::vector<Vec3> persistent_vorticity;
    
};

} // namespace FluidSim

