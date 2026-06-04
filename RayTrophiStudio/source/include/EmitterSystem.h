/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          EmitterSystem.h
* Author:        Kemal Demirtaş
* Date:          January 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file EmitterSystem.h
 * @brief Advanced Emitter System for Gas/Particle Simulations
 * 
 * Industry-standard emitter features:
 * - Multiple shapes (Sphere, Box, Cylinder, Cone, Mesh, Curve, Image)
 * - Noise modulation for organic emission
 * - Falloff/attenuation curves
 * - Animation support (keyframes, expressions)
 * - Velocity variance and spray patterns
 * 
 * Based on: Houdini POP Source, Blender Mantaflow Inflow, Maya nParticles
 */

#include "Vec3.h"
#include "json.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <cmath>

// Forward declarations
class Triangle;
class Mesh;

namespace Physics {

// ═══════════════════════════════════════════════════════════════════════════════
// ENUMS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Shape types for emitters
 */
enum class EmitterShape {
    // Basic Shapes
    Point,             ///< Single point emission
    Sphere,            ///< Spherical volume
    Box,               ///< Box/cubic volume
    Cylinder,          ///< Cylindrical volume
    Cone,              ///< Conical volume (great for flames)
    Disc,              ///< Flat circular disc
    
    // Advanced Shapes
    Mesh,              ///< Emit from mesh surface/volume
    Curve,             ///< Emit along a curve/spline
    Image,             ///< Emit based on image mask intensity
    
    COUNT
};

/**
 * @brief Emission mode
 */
enum class EmissionMode {
    Continuous,        ///< Emit every frame
    Burst,             ///< Emit once at start
    Pulse,             ///< Periodic bursts
    Random             ///< Random emission timing
};

/**
 * @brief Velocity distribution type
 */
enum class VelocityMode {
    Constant,          ///< Fixed direction and speed
    Normal,            ///< Along surface normal (for mesh)
    Random,            ///< Random direction in cone
    Radial,            ///< Outward from center
    Tangent,           ///< Tangent to surface
    Custom             ///< User-defined via callback
};

/**
 * @brief Falloff type for soft emission boundaries
 */
enum class EmitterFalloffType {
    None,              ///< Hard edge
    Linear,            ///< Linear falloff
    Smooth,            ///< Smooth (smoothstep)
    Spherical,         ///< Spherical (sqrt based)
    Gaussian,          ///< Gaussian (bell curve)
    Custom             ///< Custom curve
};

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER PROFILE (Noise & Animation Settings)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Noise modulation settings for organic emission
 */
struct EmitterNoiseSettings {
    bool enabled = false;               ///< Enable noise modulation
    float frequency = 0.5f;             ///< Noise frequency
    float amplitude = 0.5f;             ///< Noise amplitude (0-1)
    int octaves = 2;                    ///< FBM octaves
    float lacunarity = 2.0f;            ///< Frequency multiplier
    float persistence = 0.5f;           ///< Amplitude multiplier
    int seed = 42;                      ///< Random seed
    float speed = 0.1f;                 ///< Animation speed
    
    // What to modulate
    bool modulate_density = true;       ///< Modulate emission density
    bool modulate_temperature = false;  ///< Modulate emission temperature
    bool modulate_velocity = false;     ///< Modulate emission velocity
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

/**
 * @brief Falloff settings for soft emission boundaries
 */
struct EmitterFalloff {
    EmitterFalloffType type = EmitterFalloffType::Smooth;
    float falloff_start = 0.8f;         ///< Where falloff begins (0-1 of radius)
    float falloff_end = 1.0f;           ///< Where emission ends (0-1 of radius)
    
    // Custom curve (if type == Custom)
    std::vector<float> custom_curve;    ///< Lookup table for custom falloff
    
    /**
     * @brief Calculate falloff factor
     * @param normalized_distance Distance from center / radius (0-1)
     * @return Falloff multiplier (0-1)
     */
    float calculate(float normalized_distance) const;
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

/**
 * @brief Velocity randomization settings
 */
struct VelocityVariance {
    float speed_min = 1.0f;             ///< Minimum speed multiplier
    float speed_max = 1.0f;             ///< Maximum speed multiplier
    float cone_angle = 0.0f;            ///< Spray cone angle (degrees)
    float spread = 0.0f;                ///< Random spread amount
    bool inherit_velocity = false;       ///< Inherit velocity from moving emitter
    float inherit_factor = 1.0f;        ///< How much velocity to inherit
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

/**
 * @brief Time-based emission profile
 */
struct EmissionProfile {
    EmissionMode mode = EmissionMode::Continuous;
    
    // Timing
    float start_frame = 0.0f;           ///< When emission starts
    float end_frame = -1.0f;            ///< When emission ends (-1 = never)
    float warmup_frames = 0.0f;         ///< Pre-simulation frames
    
    // Pulse/Burst settings
    float burst_count = 1.0f;           ///< Number of bursts (for Pulse mode)
    float pulse_interval = 10.0f;       ///< Frames between pulses
    float pulse_duration = 1.0f;        ///< Duration of each pulse
    
    // Rate curve (keyframe-like)
    bool use_rate_curve = false;        ///< Use rate curve over time
    std::vector<std::pair<float, float>> rate_curve; ///< (frame, rate_multiplier) pairs
    
    /**
     * @brief Get rate multiplier at given frame
     */
    float getRateAtFrame(float frame) const;
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
};

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Advanced Emitter with industry-standard features
 * 
 * @example
 * ```cpp
 * AdvancedEmitter emitter;
 * emitter.name = "Fire Source";
 * emitter.shape = EmitterShape::Cone;
 * emitter.position = Vec3(0, 0, 0);
 * emitter.radius = 0.5f;
 * emitter.height = 1.0f;
 * 
 * // Organic noise
 * emitter.noise.enabled = true;
 * emitter.noise.frequency = 0.3f;
 * emitter.noise.modulate_density = true;
 * 
 * // Velocity spray
 * emitter.velocity = Vec3(0, 3, 0);
 * emitter.velocity_variance.cone_angle = 15.0f;
 * emitter.velocity_variance.speed_min = 0.8f;
 * emitter.velocity_variance.speed_max = 1.2f;
 * 
 * // Falloff for soft edges
 * emitter.falloff.type = EmitterFalloffType::Smooth;
 * emitter.falloff.falloff_start = 0.6f;
 * 
 * gasVolume->addEmitter(emitter);
 * ```
 */
class AdvancedEmitter {
public:
    // ─────────────────────────────────────────────────────────────────────────
    // Identification
    // ─────────────────────────────────────────────────────────────────────────
    std::string name = "Emitter";
    int id = -1;
    bool enabled = true;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Shape Definition
    // ─────────────────────────────────────────────────────────────────────────
    EmitterShape shape = EmitterShape::Sphere;
    
    // Transform
    Vec3 position = Vec3(0, 0, 0);
    Vec3 rotation = Vec3(0, 0, 0);      ///< Euler angles (degrees)
    Vec3 scale = Vec3(1, 1, 1);
    
    // Shape dimensions
    float radius = 1.0f;                ///< For Sphere, Cylinder, Cone, Disc
    Vec3 size = Vec3(1, 1, 1);          ///< For Box (half-extents)
    float height = 1.0f;                ///< For Cylinder, Cone
    float inner_radius = 0.0f;          ///< For hollow shapes (Disc, Ring)
    float cone_angle = 45.0f;           ///< Opening angle for Cone (degrees)
    
    // Mesh emission (for EmitterShape::Mesh)
    std::shared_ptr<Mesh> source_mesh;
    bool emit_from_surface = true;      ///< Emit from mesh surface
    bool emit_from_volume = false;      ///< Emit from mesh interior
    bool use_vertex_colors = false;     ///< Scale emission by vertex colors
    
    // ─────────────────────────────────────────────────────────────────────────
    // Emission Parameters
    // ─────────────────────────────────────────────────────────────────────────
    float density_rate = 10.0f;         ///< Density injection per second
    float temperature = 500.0f;         ///< Emission temperature (Kelvin)
    float fuel = 0.0f;                  ///< Fuel amount (for combustion)
    Vec3 color = Vec3(1, 1, 1);         ///< Emission color/tint
    
    // ─────────────────────────────────────────────────────────────────────────
    // Velocity
    // ─────────────────────────────────────────────────────────────────────────
    VelocityMode velocity_mode = VelocityMode::Constant;
    Vec3 velocity = Vec3(0, 2, 0);      ///< Base velocity direction
    float velocity_magnitude = 2.0f;    ///< Base speed
    VelocityVariance velocity_variance;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Advanced Settings
    // ─────────────────────────────────────────────────────────────────────────
    EmitterNoiseSettings noise;         ///< Noise modulation
    EmitterFalloff falloff;             ///< Edge falloff
    EmissionProfile profile;            ///< Time-based emission
    
    // ─────────────────────────────────────────────────────────────────────────
    // Jitter/Randomization
    // ─────────────────────────────────────────────────────────────────────────
    float position_jitter = 0.0f;       ///< Random position offset
    int random_seed = 0;                ///< Consistent random seed
    
public:
    AdvancedEmitter() = default;
    AdvancedEmitter(const std::string& name_) : name(name_) {}
    virtual ~AdvancedEmitter() = default;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SAMPLING
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Check if a world position is inside the emitter volume
     * @param world_pos Position to check
     * @return true if inside, false otherwise
     */
    bool isInsideVolume(const Vec3& world_pos) const;
    
    /**
     * @brief Get emission strength at a position (includes falloff and noise)
     * @param world_pos Position to sample
     * @param time Current simulation time
     * @return Emission multiplier (0-1)
     */
    float getEmissionStrength(const Vec3& world_pos, float time) const;
    
    /**
     * @brief Get emission velocity at a position
     * @param world_pos Position
     * @param surface_normal Surface normal (for mesh emission)
     * @param time Current time
     * @return Velocity vector
     */
    Vec3 getEmissionVelocity(const Vec3& world_pos, const Vec3& surface_normal, float time) const;
    
    /**
     * @brief Sample a random point inside the emitter volume
     * @param random01 Random value [0,1] for position selection
     * @param random02 Second random value
     * @param random03 Third random value
     * @return Random position inside emitter
     */
    Vec3 sampleRandomPoint(float random01, float random02, float random03) const;
    
    /**
     * @brief Check if emitter is active at given frame
     */
    bool isActiveAtFrame(float frame) const;
    
    /**
     * @brief Get emission rate at given frame (considering profile)
     */
    float getEmissionRate(float frame) const;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // BOUNDING BOX
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Get axis-aligned bounding box
     */
    void getBounds(Vec3& min_out, Vec3& max_out) const;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SERIALIZATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UTILITY
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Get readable shape name
     */
    static const char* getShapeName(EmitterShape s);
    
    /**
     * @brief Get readable velocity mode name
     */
    static const char* getVelocityModeName(VelocityMode m);
    
    /**
     * @brief Get icon for UI
     */
    const char* getIconName() const;

private:
    // Transform helpers
    Vec3 worldToLocal(const Vec3& world_pos) const;
    Vec3 localToWorld(const Vec3& local_pos) const;
    Vec3 transformDirection(const Vec3& dir) const;
    
    // Shape-specific checks
    bool isInsideSphere(const Vec3& local_pos) const;
    bool isInsideBox(const Vec3& local_pos) const;
    bool isInsideCylinder(const Vec3& local_pos) const;
    bool isInsideCone(const Vec3& local_pos) const;
    bool isInsideDisc(const Vec3& local_pos) const;
    
    // Shape-specific sampling
    Vec3 sampleSphere(float r1, float r2, float r3) const;
    Vec3 sampleBox(float r1, float r2, float r3) const;
    Vec3 sampleCylinder(float r1, float r2, float r3) const;
    Vec3 sampleCone(float r1, float r2, float r3) const;
    Vec3 sampleDisc(float r1, float r2) const;
};

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER MANAGER / BATCH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Manages multiple emitters for a simulation volume
 */
class EmitterManager {
public:
    std::vector<std::shared_ptr<AdvancedEmitter>> emitters;
    
    /**
     * @brief Add an emitter
     * @return Assigned ID
     */
    int addEmitter(std::shared_ptr<AdvancedEmitter> emitter);
    
    /**
     * @brief Remove an emitter
     */
    bool removeEmitter(int id);
    bool removeEmitter(std::shared_ptr<AdvancedEmitter> emitter);
    
    /**
     * @brief Find emitter by name
     */
    std::shared_ptr<AdvancedEmitter> findByName(const std::string& name) const;
    
    /**
     * @brief Find emitter by ID
     */
    std::shared_ptr<AdvancedEmitter> findById(int id) const;
    
    /**
     * @brief Get combined emission strength at a position
     */
    float getCombinedEmission(const Vec3& world_pos, float time) const;
    
    /**
     * @brief Clear all emitters
     */
    void clear() { emitters.clear(); next_id = 0; }
    
    // Serialization
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
    
private:
    int next_id = 0;
};

} // namespace Physics
