/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          ForceField.h
* Author:        Kemal Demirtaş
* Date:          January 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file ForceField.h
 * @brief Universal Force Field system for physics simulations
 * 
 * Force Fields can affect:
 * - Gas/Smoke simulations
 * - Particle systems (future)
 * - Cloth simulations (future)
 * - Rigid body physics (future)
 * 
 * Inspired by industry standards: Houdini, Blender, Maya
 */

#include "Vec3.h"
#include "json.hpp"
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <cmath>

// Forward declarations
class Transform;

namespace Physics {

// ═══════════════════════════════════════════════════════════════════════════════
// ENUMS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Types of force fields available
 */
enum class ForceFieldType {
    // Directional Forces
    Wind,              ///< Uniform directional force (like global wind)
    Gravity,           ///< Directional gravity with optional falloff
    
    // Point-based Forces
    Attractor,         ///< Pulls particles/gas towards center
    Repeller,          ///< Pushes particles/gas away from center
    Vortex,            ///< Creates swirling motion around axis
    
    // Volume-based Forces
    Turbulence,        ///< Stochastic noise-based force
    CurlNoise,         ///< Divergence-free noise for fluid-like motion
    Drag,              ///< Velocity damping/air resistance
    
    // Special Forces
    Magnetic,          ///< Follows magnetic field lines
    DirectionalNoise,  ///< Noise applied along a direction
    
    COUNT
};

/**
 * @brief Falloff curve types for force attenuation
 */
enum class FalloffType {
    None,              ///< No falloff (constant strength)
    Linear,            ///< Linear decrease: 1 - r/R
    Smooth,            ///< Hermite interpolation (smoothstep)
    Sphere,            ///< Spherical falloff: sqrt(1 - (r/R)²)
    InverseSquare,     ///< Physical falloff: 1/r²
    Exponential,       ///< Exponential decay: e^(-r/R)
    Custom             ///< User-defined curve (future)
};

/**
 * @brief Shape of the force field influence zone
 */
enum class ForceFieldShape {
    Infinite,          ///< Global force, affects entire scene
    Sphere,            ///< Spherical influence zone
    Box,               ///< Box-shaped influence zone
    Cylinder,          ///< Cylindrical influence zone
    Cone               ///< Conical influence zone
};

// ═══════════════════════════════════════════════════════════════════════════════
// NOISE SETTINGS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Noise parameters for Turbulence/CurlNoise force fields
 */
struct NoiseSettings {
    int octaves = 4;                   ///< Number of noise layers (1-8)
    float frequency = 0.5f;            ///< Base frequency (scale in world units)
    float lacunarity = 2.0f;           ///< Frequency multiplier per octave
    float persistence = 0.5f;          ///< Amplitude multiplier per octave
    float amplitude = 1.0f;            ///< Overall noise strength
    int seed = 42;                     ///< Random seed for reproducibility
    float speed = 0.1f;                ///< Time-based animation speed
    
    // Serialization
    nlohmann::json toJson() const {
        nlohmann::json j;
        j["octaves"] = octaves;
        j["frequency"] = frequency;
        j["lacunarity"] = lacunarity;
        j["persistence"] = persistence;
        j["amplitude"] = amplitude;
        j["seed"] = seed;
        j["speed"] = speed;
        return j;
    }
    
    void fromJson(const nlohmann::json& j) {
        if (j.contains("octaves")) octaves = j["octaves"];
        if (j.contains("frequency")) frequency = j["frequency"];
        if (j.contains("lacunarity")) lacunarity = j["lacunarity"];
        if (j.contains("persistence")) persistence = j["persistence"];
        if (j.contains("amplitude")) amplitude = j["amplitude"];
        if (j.contains("seed")) seed = j["seed"];
        if (j.contains("speed")) speed = j["speed"];
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD CLASS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Universal Force Field that can affect various physics simulations
 * 
 * @example
 * ```cpp
 * // Create a vortex force field
 * auto vortex = std::make_shared<ForceField>("Tornado");
 * vortex->type = ForceFieldType::Vortex;
 * vortex->position = Vec3(0, 5, 0);
 * vortex->strength = 10.0f;
 * vortex->falloff_radius = 5.0f;
 * scene.addForceField(vortex);
 * 
 * // In simulation loop:
 * Vec3 force = scene.evaluateForceFieldsAt(particle_pos, time);
 * particle_velocity += force * dt;
 * ```
 */
class ForceField {
public:
    // ─────────────────────────────────────────────────────────────────────────
    // Identification
    // ─────────────────────────────────────────────────────────────────────────
    std::string name = "Force Field";
    int id = -1;                           ///< Unique ID assigned by SceneData
    bool enabled = true;                   ///< Is this force field active?
    bool visible = true;                   ///< Show in viewport (gizmo)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Type & Shape
    // ─────────────────────────────────────────────────────────────────────────
    ForceFieldType type = ForceFieldType::Wind;
    ForceFieldShape shape = ForceFieldShape::Sphere;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Transform
    // ─────────────────────────────────────────────────────────────────────────
    Vec3 position = Vec3(0, 0, 0);         ///< Center position in world space
    Vec3 rotation = Vec3(0, 0, 0);         ///< Euler rotation (degrees)
    Vec3 scale = Vec3(1, 1, 1);            ///< Non-uniform scale
    
    // ─────────────────────────────────────────────────────────────────────────
    // Force Parameters
    // ─────────────────────────────────────────────────────────────────────────
    float strength = 1.0f;                 ///< Force magnitude
    Vec3 direction = Vec3(0, 1, 0);        ///< Direction for Wind/Gravity types
    
    // ─────────────────────────────────────────────────────────────────────────
    // Falloff
    // ─────────────────────────────────────────────────────────────────────────
    FalloffType falloff_type = FalloffType::Smooth;
    float falloff_radius = 5.0f;           ///< Outer radius where force = 0
    float inner_radius = 0.0f;             ///< Inner radius with full strength
    
    // ─────────────────────────────────────────────────────────────────────────
    // Noise (for Turbulence/CurlNoise)
    // ─────────────────────────────────────────────────────────────────────────
    bool use_noise = false;                ///< Enable noise modulation
    NoiseSettings noise;
    
    // ─────────────────────────────────────────────────────────────────────────
    // Vortex-specific
    // ─────────────────────────────────────────────────────────────────────────
    Vec3 axis = Vec3(0, 1, 0);             ///< Rotation axis for Vortex
    float inward_force = 0.0f;             ///< Pull towards center (spiral)
    float upward_force = 0.0f;             ///< Lift along axis (tornado)
    
    // ─────────────────────────────────────────────────────────────────────────
    // Drag-specific
    // ─────────────────────────────────────────────────────────────────────────
    float linear_drag = 0.1f;              ///< Linear velocity damping: F = -drag * v
    float quadratic_drag = 0.0f;           ///< Quadratic damping: F = -drag * v²
    
    // ─────────────────────────────────────────────────────────────────────────
    // Time-based
    // ─────────────────────────────────────────────────────────────────────────
    float start_frame = 0.0f;              ///< Frame when field becomes active
    float end_frame = -1.0f;               ///< Frame when field stops (-1 = never)
    float phase = 0.0f;                    ///< Animation phase offset
    
    // ─────────────────────────────────────────────────────────────────────────
    // Affect Masks (which systems this field affects)
    // ─────────────────────────────────────────────────────────────────────────
    bool affects_gas = true;               ///< Affect gas/smoke simulations
    bool affects_particles = true;         ///< Affect particle systems
    bool affects_cloth = true;             ///< Affect cloth simulations
    bool affects_rigidbody = true;         ///< Affect rigid bodies
    
    // ─────────────────────────────────────────────────────────────────────────
    // Cached transform handle (for gizmo integration)
    // ─────────────────────────────────────────────────────────────────────────
    std::shared_ptr<Transform> transform_handle;
    
public:
    ForceField() = default;
    ForceField(const std::string& name_) : name(name_) {}
    virtual ~ForceField() = default;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EVALUATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Evaluate force at a world position
     * @param world_pos Position to evaluate force at
     * @param time Current simulation time
     * @param velocity Current velocity (for drag calculations)
     * @return Force vector at the given position
     */
    Vec3 evaluate(const Vec3& world_pos, float time, const Vec3& velocity = Vec3(0,0,0)) const;
    
    /**
     * @brief Check if a point is within the force field's influence zone
     * @param world_pos Position to check
     * @return true if the point is affected by this field
     */
    bool isInsideInfluenceZone(const Vec3& world_pos) const;
    
    /**
     * @brief Calculate falloff factor at a given distance
     * @param distance Distance from force field center
     * @return Falloff multiplier (0.0 - 1.0)
     */
    float calculateFalloff(float distance) const;
    
    /**
     * @brief Check if field is active at given frame
     */
    bool isActiveAtFrame(float frame) const {
        if (!enabled) return false;
        if (frame < start_frame) return false;
        if (end_frame >= 0 && frame > end_frame) return false;
        return true;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // SERIALIZATION
    // ═══════════════════════════════════════════════════════════════════════════
    
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // UTILITY
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @brief Get readable name for force field type
     */
    static const char* getTypeName(ForceFieldType t);
    
    /**
     * @brief Get readable name for falloff type
     */
    static const char* getFalloffName(FalloffType f);
    
    /**
     * @brief Get readable name for shape
     */
    static const char* getShapeName(ForceFieldShape s);
    
    /**
     * @brief Get icon name for UI display
     */
    const char* getIconName() const;

private:
    // Internal evaluation functions for each type
    Vec3 evaluateWind(const Vec3& local_pos, float time) const;
    Vec3 evaluateGravity(const Vec3& local_pos, float time) const;
    Vec3 evaluateAttractor(const Vec3& local_pos, float time) const;
    Vec3 evaluateRepeller(const Vec3& local_pos, float time) const;
    Vec3 evaluateVortex(const Vec3& local_pos, float time) const;
    Vec3 evaluateTurbulence(const Vec3& world_pos, float time) const;
    Vec3 evaluateCurlNoise(const Vec3& world_pos, float time) const;
    Vec3 evaluateDrag(const Vec3& local_pos, const Vec3& velocity, float time) const;
    Vec3 evaluateMagnetic(const Vec3& local_pos, float time) const;
    
    // Transform world position to local space
    Vec3 worldToLocal(const Vec3& world_pos) const;
};

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Manages all force fields in the scene and provides evaluation
 */
class ForceFieldManager {
public:
    std::vector<std::shared_ptr<ForceField>> force_fields;
    
    /**
     * @brief Add a force field to the manager
     * @return Assigned ID
     */
    int addForceField(std::shared_ptr<ForceField> field);
    
    /**
     * @brief Remove a force field
     */
    bool removeForceField(std::shared_ptr<ForceField> field);
    bool removeForceField(int id);
    
    /**
     * @brief Find force field by name
     */
    std::shared_ptr<ForceField> findByName(const std::string& name) const;
    
    /**
     * @brief Find force field by ID
     */
    std::shared_ptr<ForceField> findById(int id) const;
    
    /**
     * @brief Evaluate combined force from all active fields at a position (with system filter)
     * @param world_pos Position to evaluate
     * @param time Current time
     * @param velocity Current velocity (for drag)
     * @param is_gas, is_particle, is_cloth, is_rigidbody - filter flags
     * @return Combined force vector
     */
    Vec3 evaluateAtFiltered(const Vec3& world_pos, float time, 
                            const Vec3& velocity,
                            bool is_gas, bool is_particle,
                            bool is_cloth, bool is_rigidbody) const;
    
    /**
     * @brief Evaluate combined force (affects all systems, no filtering)
     */
    Vec3 evaluateAt(const Vec3& world_pos, float time, const Vec3& velocity) const;
    
    /**
     * @brief Evaluate combined force (affects all systems, no velocity - simplified)
     */
    Vec3 evaluateAt(const Vec3& world_pos, float time) const;
    
    /**
     * @brief Clear all force fields
     */
    void clear() { force_fields.clear(); next_id = 0; }
    
    /**
     * @brief Get count of active force fields
     */
    size_t getActiveCount() const;
    
    // Serialization
    nlohmann::json toJson() const;
    void fromJson(const nlohmann::json& j);
    
private:
    int next_id = 0;
};

} // namespace Physics
