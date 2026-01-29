/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GasVolume.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file GasVolume.h
 * @brief Gas/Smoke volume scene object for RayTrophi
 * 
 * This class wraps GasSimulator and integrates it with the scene system,
 * providing transform, rendering, and serialization support.
 */

#include "GasSimulator.h"
#include "VolumeShader.h"
#include "Transform.h"
#include "Hittable.h"
#include "AABB.h"
#include "json.hpp"
#include <memory>
#include <string>

/**
 * @brief Gas/Smoke volume as a scene object
 * 
 * Integrates GasSimulator with the scene system, providing:
 * - Transform (position, rotation, scale)
 * - VolumeShader for rendering
 * - Timeline integration
 * - Serialization
 * 
 * @example
 * ```cpp
 * auto gas = std::make_shared<GasVolume>();
 * gas->setName("Campfire Smoke");
 * gas->setPosition(Vec3(0, 0, 0));
 * gas->getSettings().resolution_x = 64;
 * gas->initialize();
 * gas->addEmitter(emitter);
 * gas->play();
 * scene.gas_volumes.push_back(gas);
 * ```
 */
class GasVolume : public Hittable {
public:
    // ═══════════════════════════════════════════════════════════════════════
    // CONSTRUCTION
    // ═══════════════════════════════════════════════════════════════════════
    
    GasVolume();
    GasVolume(const std::string& name);
    virtual ~GasVolume();
    
    // ═══════════════════════════════════════════════════════════════════════
    // HITTABLE INTERFACE
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Ray-volume intersection (for CPU rendering)
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override;
    
    /// @brief Get bounding box
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;
    
    /// @brief Get object type
    std::string getObjectType() const { return "GasVolume"; }
    
    /// @brief Get object name
    std::string getName() const { return name; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // TRANSFORM
    // ═══════════════════════════════════════════════════════════════════════
    
    Vec3 getPosition() const { return position; }
    void setPosition(const Vec3& pos);
    
    Vec3 getRotation() const { return rotation; }
    void setRotation(const Vec3& rot);
    
    Vec3 getScale() const { return scale; }
    void setScale(const Vec3& s);
    
    bool isVisible() const { return visible; }
    void setVisible(bool v) { visible = v; }
    
    /// @brief Get transform handle for gizmo integration
    std::shared_ptr<Transform> getTransformHandle() const { return transform; }

    /// @brief Get the final transformation matrix
    Matrix4x4 getTransform() const { return transform ? transform->getFinal() : Matrix4x4::identity(); }
    
    /// @brief Apply transform to simulation grid
    void applyTransform();
    
    // ═══════════════════════════════════════════════════════════════════════
    // SIMULATION CONTROL
    // ═══════════════════════════════════════════════════════════════════════
    
    void freeGPUResources();

    /// @brief Initialize simulation with current settings
    void initialize();
    
    /// @brief Start/resume simulation
    void play() { is_playing = true; }
    
    /// @brief Pause simulation
    void pause() { is_playing = false; }
    
    /// @brief Stop and reset simulation
    void stop();
    
    /// @brief Step one frame
    void stepFrame(float dt, void* stream = nullptr);
    
    /// @brief Update simulation (called from main loop)
    void update(float dt, void* stream = nullptr);
    
    /// @brief Reset simulation to initial state
    void reset();
    
    /// @brief Is simulation playing?
    bool isPlaying() const { return is_playing; }
    
    /// @brief Is simulation initialized?
    bool isInitialized() const { return initialized; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // SETTINGS & SIMULATOR ACCESS
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Get simulation settings - returns SIMULATOR's settings directly
    /// This ensures UI changes are immediately reflected in simulation
    FluidSim::GasSimulationSettings& getSettings() { return simulator.getSettings(); }
    const FluidSim::GasSimulationSettings& getSettings() const { return simulator.getSettings(); }
    
    /// @brief Get simulator (for advanced access)
    FluidSim::GasSimulator& getSimulator() { return simulator; }
    const FluidSim::GasSimulator& getSimulator() const { return simulator; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // EMITTERS
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Add an emitter
    int addEmitter(const FluidSim::Emitter& emitter);
    
    /// @brief Remove emitter by index
    void removeEmitter(int index);
    
    /// @brief Get emitters
    std::vector<FluidSim::Emitter>& getEmitters() { return simulator.getEmitters(); }
    const std::vector<FluidSim::Emitter>& getEmitters() const { return simulator.getEmitters(); }
    
    /// @brief Get colliders
   
    // ═══════════════════════════════════════════════════════════════════════
    // SHADER
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Set volume shader for rendering
    void setShader(std::shared_ptr<VolumeShader> s) { shader = s; }
    
    /// @brief Get volume shader
    std::shared_ptr<VolumeShader> getShader() const { return shader; }
    
    /// @brief Get or create default shader
    std::shared_ptr<VolumeShader> getOrCreateShader();
    
    // ═══════════════════════════════════════════════════════════════════════
    // SAMPLING (for rendering)
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Sample density at world position
    float sampleDensity(const Vec3& world_pos) const;
    
    /// @brief Sample temperature at world position
    float sampleTemperature(const Vec3& world_pos) const;
    
    /// @brief Sample flame/fire intensity at world position (combustion interaction field)
    float sampleFlameIntensity(const Vec3& world_pos) const;
    
    /// @brief Sample fuel at world position
    float sampleFuel(const Vec3& world_pos) const;
    
    /// @brief Sample velocity at world position
    Vec3 sampleVelocity(const Vec3& world_pos) const;
    
    /// @brief Get world-space bounds
    void getWorldBounds(Vec3& min_out, Vec3& max_out) const;
    
    // ═══════════════════════════════════════════════════════════════════════
    // BAKING
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Start baking simulation
    void startBake(int start_frame, int end_frame);
    
    /// @brief Cancel baking
    void cancelBake() { simulator.cancelBake(); }
    
    /// @brief Is baking in progress?
    bool isBaking() const { return simulator.isBaking(); }
    
    /// @brief Get bake progress (0-1)
    float getBakeProgress() const { return simulator.getBakeProgress(); }
    
    // ═══════════════════════════════════════════════════════════════════════
    // EXPORT
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Export current frame to VDB
    bool exportToVDB(const std::string& filepath) const;
    
    /// @brief Export sequence to VDB
    bool exportSequence(const std::string& directory, int start_frame, int end_frame);
    
    // ═══════════════════════════════════════════════════════════════════════
    // TIMELINE INTEGRATION
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Link to main timeline
    void setLinkedToTimeline(bool linked) { linked_to_timeline = linked; }
    bool isLinkedToTimeline() const { return linked_to_timeline; }
    
    /// @brief Set frame offset from timeline
    void setFrameOffset(int offset) { frame_offset = offset; }
    int getFrameOffset() const { return frame_offset; }
    
    void setFrame(int frame);
    int getCurrentFrame() const { return simulator.getCurrentFrame(); }
    
    /// @brief Update frame from timeline (called by animation system)
    void updateFromTimeline(int timeline_frame, void* stream = nullptr);
    
  
    // ═══════════════════════════════════════════════════════════════════════
    // STATISTICS
    // ═══════════════════════════════════════════════════════════════════════
    
    float getTotalDensity() const { return simulator.getTotalDensity(); }
    float getMaxDensity() const { return simulator.getMaxDensity(); }
    float getMaxVelocity() const { return simulator.getMaxVelocity(); }
    int getActiveVoxelCount() const { return simulator.getActiveVoxelCount(); }
    float getLastStepTime() const { return simulator.getLastStepTime(); }
    
    // ═══════════════════════════════════════════════════════════════════════
    // SERIALIZATION
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Serialize to JSON
    nlohmann::json toJson() const;
    
    /// @brief Deserialize from JSON
    void fromJson(const nlohmann::json& j);
    
    // ═══════════════════════════════════════════════════════════════════════
    // GPU ACCESS
    // ═══════════════════════════════════════════════════════════════════════
    
    /// @brief Upload density to GPU for rendering
    void uploadToGPU(void* stream = nullptr);
    
    /// @brief Get GPU density pointer
    void* getGPUDensityPtr() const { return simulator.getGPUDensityPtr(); }
    
    /// @brief Get GPU temperature pointer
    void* getGPUTemperaturePtr() const { return simulator.getGPUTemperaturePtr(); }
    
    /// @brief Get density texture object (cudaTextureObject_t cast to ulong)
    unsigned long long getDensityTexture() const { return density_texture; }
    
    /// @brief Get temperature texture object
    unsigned long long getTemperatureTexture() const { return temperature_texture; }
    
    /// @brief Get velocity texture object
    unsigned long long getVelocityTexture() const { return velocity_texture; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // PUBLIC DATA
    // ═══════════════════════════════════════════════════════════════════════
    
    std::string name = "Gas Volume";
    int id = -1;                    // Unique ID in scene
    bool visible = true;            // Render visibility
    bool selected = false;          // Selection state
    // NOTE: Use getSettings() to access simulation settings - they are stored in simulator
    bool linked_to_timeline = true;
    
    enum class VolumeRenderPath {
        Legacy,     // Original real-time path
        VDBUnified, // High-quality VDB pipeline
    };
    
    VolumeRenderPath render_path = VolumeRenderPath::VDBUnified; // Default to high quality
    int live_vdb_id = -1;  // Handle to registered VDB volume
    private:
    // Transform
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0);
    Vec3 scale = Vec3(1, 1, 1);
    std::shared_ptr<Transform> transform;
    
    // Simulation
    FluidSim::GasSimulator simulator;
   
    bool initialized = false;
    bool is_playing = true;
    
    // Rendering
    std::shared_ptr<VolumeShader> shader;
    
    // GPU Resources
    unsigned long long density_texture = 0;     // cudaTextureObject_t (cast to ulong for header purity)
    void* density_array = nullptr;              // cudaArray_t (opaque pointer)
    unsigned long long temperature_texture = 0; 
    void* temperature_array = nullptr;
    unsigned long long velocity_texture = 0;
    void* velocity_array = nullptr;
    
    // Track current GPU texture resolution to detect changes
    int gpu_res_x = 0;
    int gpu_res_y = 0;
    int gpu_res_z = 0;
    
    // Timeline
  
    int frame_offset = 0;
    
    // Cached bounds
    mutable Vec3 cached_bounds_min;
    mutable Vec3 cached_bounds_max;
    mutable bool bounds_dirty = true;
    
    void updateBounds() const;
};

