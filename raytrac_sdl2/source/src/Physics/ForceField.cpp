/**
 * @file ForceField.cpp
 * @brief Implementation of Universal Force Field system
 */

#include "ForceField.h"
#include "CurlNoise.h"
#include <cmath>
#include <algorithm>
#include <Matrix4x4.h>

namespace Physics {

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD - EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 ForceField::worldToLocal(const Vec3& world_pos) const {
    // Construct full transform matrix
    Matrix4x4 mat = Matrix4x4::translation(position) * 
                    Matrix4x4::rotationX(rotation.x * 0.0174533f) * 
                    Matrix4x4::rotationY(rotation.y * 0.0174533f) * 
                    Matrix4x4::rotationZ(rotation.z * 0.0174533f) * 
                    Matrix4x4::scaling(scale);
                    
    // Inverse transform to get local position
    Matrix4x4 inv = mat.inverse();
    return inv.transform_point(world_pos);
}

bool ForceField::isInsideInfluenceZone(const Vec3& world_pos) const {
    if (shape == ForceFieldShape::Infinite) return true;
    
    Vec3 local = worldToLocal(world_pos);
    float dist = local.length();
    
    switch (shape) {
        case ForceFieldShape::Sphere:
            return dist <= falloff_radius;
            
        case ForceFieldShape::Box:
            return std::abs(local.x) <= falloff_radius &&
                   std::abs(local.y) <= falloff_radius &&
                   std::abs(local.z) <= falloff_radius;
            
        case ForceFieldShape::Cylinder: {
            float radial_dist = std::sqrt(local.x * local.x + local.z * local.z);
            return radial_dist <= falloff_radius && std::abs(local.y) <= falloff_radius;
        }
        
        case ForceFieldShape::Cone: {
            if (local.y < 0 || local.y > falloff_radius) return false;
            float ratio = local.y / falloff_radius;
            float allowed_radius = falloff_radius * ratio;
            float radial_dist = std::sqrt(local.x * local.x + local.z * local.z);
            return radial_dist <= allowed_radius;
        }
        
        default:
            return true;
    }
}

float ForceField::calculateFalloff(float distance) const {
    if (distance <= inner_radius) return 1.0f;
    if (distance >= falloff_radius) return 0.0f;
    
    float t = (distance - inner_radius) / (falloff_radius - inner_radius);
    
    switch (falloff_type) {
        case FalloffType::None:
            return 1.0f;
            
        case FalloffType::Linear:
            return 1.0f - t;
            
        case FalloffType::Smooth:
            return 1.0f - t * t * (3.0f - 2.0f * t); // smoothstep(1-t)
            
        case FalloffType::Sphere:
            return std::sqrt(1.0f - t * t);
            
        case FalloffType::InverseSquare: {
            float r = inner_radius + t * (falloff_radius - inner_radius);
            if (r < 0.01f) r = 0.01f;
            float ref = inner_radius > 0.01f ? inner_radius : 0.01f;
            return (ref * ref) / (r * r);
        }
        
        case FalloffType::Exponential:
            return std::exp(-3.0f * t); // e^-3 at edge ≈ 0.05
            
        default:
            return 1.0f - t;
    }
}

Vec3 ForceField::evaluate(const Vec3& world_pos, float time, const Vec3& velocity) const {
    if (!enabled) return Vec3(0, 0, 0);
    if (!isInsideInfluenceZone(world_pos)) return Vec3(0, 0, 0);
    
    Vec3 local_pos = worldToLocal(world_pos);
    Vec3 force(0, 0, 0);
    
    switch (type) {
        case ForceFieldType::Wind:
            force = evaluateWind(local_pos, time);
            break;
        case ForceFieldType::Gravity:
            force = evaluateGravity(local_pos, time);
            break;
        case ForceFieldType::Attractor:
            force = evaluateAttractor(local_pos, time);
            break;
        case ForceFieldType::Repeller:
            force = evaluateRepeller(local_pos, time);
            break;
        case ForceFieldType::Vortex:
            force = evaluateVortex(local_pos, time);
            break;
        case ForceFieldType::Turbulence:
            force = evaluateTurbulence(world_pos, time);
            break;
        case ForceFieldType::CurlNoise:
            force = evaluateCurlNoise(world_pos, time);
            break;
        case ForceFieldType::Drag:
            force = evaluateDrag(local_pos, velocity, time);
            break;
        case ForceFieldType::Magnetic:
            force = evaluateMagnetic(local_pos, time);
            break;
        default:
            break;
    }
    
    // Apply falloff
    float falloff = 1.0f;
    if (shape != ForceFieldShape::Infinite) {
        float dist = local_pos.length();
        falloff = calculateFalloff(dist);
    }
    
    Vec3 final_force = force * falloff;
    
    // Transform force back to world space (ignore translation for vector)
    Matrix4x4 rot_scale = Matrix4x4::rotationX(rotation.x * 0.0174533f) * 
                        Matrix4x4::rotationY(rotation.y * 0.0174533f) * 
                        Matrix4x4::rotationZ(rotation.z * 0.0174533f) * 
                        Matrix4x4::scaling(scale);
                        
    return rot_scale.transform_vector(final_force);
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD - TYPE EVALUATORS
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 ForceField::evaluateWind(const Vec3& local_pos, float time) const {
    Vec3 force = direction * strength;
    
    // Add noise modulation if enabled
    if (use_noise) {
        Vec3 world_pos = local_pos + position;
        float noise_val = Noise::fbm3D_animated(
            world_pos, time * noise.speed, 
            noise.octaves, noise.frequency,
            noise.lacunarity, noise.persistence, noise.seed
        );
        force = force * (1.0f + noise_val * noise.amplitude);
    }
    
    return force;
}

Vec3 ForceField::evaluateGravity(const Vec3& local_pos, float time) const {
    return direction * strength * -9.81f;
}

Vec3 ForceField::evaluateAttractor(const Vec3& local_pos, float time) const {
    float dist = local_pos.length();
    if (dist < 0.001f) return Vec3(0, 0, 0);
    
    Vec3 to_center = local_pos * (-1.0f / dist);  // Normalized, pointing inward
    float force_mag = strength;
    
    // Inverse square for more realistic attraction
    if (falloff_type == FalloffType::InverseSquare) {
        force_mag = strength / (dist * dist + 0.1f);
    }
    
    return to_center * force_mag;
}

Vec3 ForceField::evaluateRepeller(const Vec3& local_pos, float time) const {
    float dist = local_pos.length();
    if (dist < 0.001f) dist = 0.001f;
    
    Vec3 away = local_pos / dist;  // Normalized, pointing outward
    float force_mag = strength;
    
    // Inverse square for more realistic repulsion
    if (falloff_type == FalloffType::InverseSquare) {
        force_mag = strength / (dist * dist + 0.1f);
    }
    
    return away * force_mag;
}

Vec3 ForceField::evaluateVortex(const Vec3& local_pos, float time) const {
    // Project position onto the vortex plane (perpendicular to axis)
    Vec3 norm_axis = axis;
    float axis_len = norm_axis.length();
    if (axis_len > 0.001f) norm_axis = norm_axis / axis_len;
    else norm_axis = Vec3(0, 1, 0);
    
    // Component along axis
    float along_axis = local_pos.x * norm_axis.x + local_pos.y * norm_axis.y + local_pos.z * norm_axis.z;
    Vec3 along_axis_vec = norm_axis * along_axis;
    
    // Radial component (perpendicular to axis)
    Vec3 radial = local_pos - along_axis_vec;
    float radial_dist = radial.length();
    
    if (radial_dist < 0.001f) {
        // On the axis - only apply upward force
        return norm_axis * upward_force;
    }
    
    // Tangent direction (perpendicular to both axis and radial)
    Vec3 tangent(
        norm_axis.y * radial.z - norm_axis.z * radial.y,
        norm_axis.z * radial.x - norm_axis.x * radial.z,
        norm_axis.x * radial.y - norm_axis.y * radial.x
    );
    float tangent_len = tangent.length();
    if (tangent_len > 0.001f) tangent = tangent / tangent_len;
    
    // Main swirl force
    Vec3 force = tangent * strength;
    
    // Inward spiral
    Vec3 normalized_radial = radial / radial_dist;
    force = force - normalized_radial * inward_force;
    
    // Upward lift
    force = force + norm_axis * upward_force;
    
    return force;
}

Vec3 ForceField::evaluateTurbulence(const Vec3& world_pos, float time) const {
    Vec3 animated_pos = world_pos + Vec3(time * noise.speed, 0, 0);
    
    // Use 3 noise samples for x, y, z force components
    float fx = Noise::fbm3D(
        animated_pos * noise.frequency, 
        noise.octaves, 1.0f, noise.lacunarity, noise.persistence, noise.seed
    );
    float fy = Noise::fbm3D(
        animated_pos * noise.frequency, 
        noise.octaves, 1.0f, noise.lacunarity, noise.persistence, noise.seed + 100
    );
    float fz = Noise::fbm3D(
        animated_pos * noise.frequency, 
        noise.octaves, 1.0f, noise.lacunarity, noise.persistence, noise.seed + 200
    );
    
    return Vec3(fx, fy, fz) * strength * noise.amplitude;
}

Vec3 ForceField::evaluateCurlNoise(const Vec3& world_pos, float time) const {
    Vec3 curl = Noise::curlFBM_animated(
        world_pos, time,
        noise.octaves, noise.frequency,
        noise.lacunarity, noise.persistence,
        noise.speed, noise.seed
    );
    
    return curl * strength * noise.amplitude;
}

Vec3 ForceField::evaluateDrag(const Vec3& local_pos, const Vec3& velocity, float time) const {
    float speed = velocity.length();
    if (speed < 0.001f) return Vec3(0, 0, 0);
    
    Vec3 normalized_vel = velocity / speed;
    
    // F = -linear_drag * v - quadratic_drag * v²
    float drag_force = linear_drag * speed + quadratic_drag * speed * speed;
    
    return normalized_vel * (-drag_force * strength);
}

Vec3 ForceField::evaluateMagnetic(const Vec3& local_pos, float time) const {
    // Simplified magnetic field - force perpendicular to position
    float dist = local_pos.length();
    if (dist < 0.001f) return Vec3(0, 0, 0);
    
    Vec3 normalized = local_pos / dist;
    
    // Cross with direction to get perpendicular force
    Vec3 force(
        direction.y * normalized.z - direction.z * normalized.y,
        direction.z * normalized.x - direction.x * normalized.z,
        direction.x * normalized.y - direction.y * normalized.x
    );
    
    return force * strength;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD - SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json ForceField::toJson() const {
    nlohmann::json j;
    
    j["name"] = name;
    j["enabled"] = enabled;
    j["visible"] = visible;
    j["type"] = static_cast<int>(type);
    j["shape"] = static_cast<int>(shape);
    
    j["position"] = { position.x, position.y, position.z };
    j["rotation"] = { rotation.x, rotation.y, rotation.z };
    j["scale"] = { scale.x, scale.y, scale.z };
    
    j["strength"] = strength;
    j["direction"] = { direction.x, direction.y, direction.z };
    
    j["falloff_type"] = static_cast<int>(falloff_type);
    j["falloff_radius"] = falloff_radius;
    j["inner_radius"] = inner_radius;
    
    j["use_noise"] = use_noise;
    j["noise"] = noise.toJson();
    
    // Vortex
    j["axis"] = { axis.x, axis.y, axis.z };
    j["inward_force"] = inward_force;
    j["upward_force"] = upward_force;
    
    // Drag
    j["linear_drag"] = linear_drag;
    j["quadratic_drag"] = quadratic_drag;
    
    // Time
    j["start_frame"] = start_frame;
    j["end_frame"] = end_frame;
    j["phase"] = phase;
    
    // Affect masks
    j["affects_gas"] = affects_gas;
    j["affects_particles"] = affects_particles;
    j["affects_cloth"] = affects_cloth;
    j["affects_rigidbody"] = affects_rigidbody;
    
    return j;
}

void ForceField::fromJson(const nlohmann::json& j) {
    if (j.contains("name")) name = j["name"];
    if (j.contains("enabled")) enabled = j["enabled"];
    if (j.contains("visible")) visible = j["visible"];
    if (j.contains("type")) type = static_cast<ForceFieldType>(j["type"].get<int>());
    if (j.contains("shape")) shape = static_cast<ForceFieldShape>(j["shape"].get<int>());
    
    if (j.contains("position")) {
        auto p = j["position"];
        position = Vec3(p[0], p[1], p[2]);
    }
    if (j.contains("rotation")) {
        auto r = j["rotation"];
        rotation = Vec3(r[0], r[1], r[2]);
    }
    if (j.contains("scale")) {
        auto s = j["scale"];
        scale = Vec3(s[0], s[1], s[2]);
    }
    
    if (j.contains("strength")) strength = j["strength"];
    if (j.contains("direction")) {
        auto d = j["direction"];
        direction = Vec3(d[0], d[1], d[2]);
    }
    
    if (j.contains("falloff_type")) falloff_type = static_cast<FalloffType>(j["falloff_type"].get<int>());
    if (j.contains("falloff_radius")) falloff_radius = j["falloff_radius"];
    if (j.contains("inner_radius")) inner_radius = j["inner_radius"];
    
    if (j.contains("use_noise")) use_noise = j["use_noise"];
    if (j.contains("noise")) noise.fromJson(j["noise"]);
    
    if (j.contains("axis")) {
        auto a = j["axis"];
        axis = Vec3(a[0], a[1], a[2]);
    }
    if (j.contains("inward_force")) inward_force = j["inward_force"];
    if (j.contains("upward_force")) upward_force = j["upward_force"];
    
    if (j.contains("linear_drag")) linear_drag = j["linear_drag"];
    if (j.contains("quadratic_drag")) quadratic_drag = j["quadratic_drag"];
    
    if (j.contains("start_frame")) start_frame = j["start_frame"];
    if (j.contains("end_frame")) end_frame = j["end_frame"];
    if (j.contains("phase")) phase = j["phase"];
    
    if (j.contains("affects_gas")) affects_gas = j["affects_gas"];
    if (j.contains("affects_particles")) affects_particles = j["affects_particles"];
    if (j.contains("affects_cloth")) affects_cloth = j["affects_cloth"];
    if (j.contains("affects_rigidbody")) affects_rigidbody = j["affects_rigidbody"];
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD - UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

const char* ForceField::getTypeName(ForceFieldType t) {
    switch (t) {
        case ForceFieldType::Wind: return "Wind";
        case ForceFieldType::Gravity: return "Gravity";
        case ForceFieldType::Attractor: return "Attractor";
        case ForceFieldType::Repeller: return "Repeller";
        case ForceFieldType::Vortex: return "Vortex";
        case ForceFieldType::Turbulence: return "Turbulence";
        case ForceFieldType::CurlNoise: return "Curl Noise";
        case ForceFieldType::Drag: return "Drag";
        case ForceFieldType::Magnetic: return "Magnetic";
        case ForceFieldType::DirectionalNoise: return "Directional Noise";
        default: return "Unknown";
    }
}

const char* ForceField::getFalloffName(FalloffType f) {
    switch (f) {
        case FalloffType::None: return "None";
        case FalloffType::Linear: return "Linear";
        case FalloffType::Smooth: return "Smooth";
        case FalloffType::Sphere: return "Sphere";
        case FalloffType::InverseSquare: return "Inverse Square";
        case FalloffType::Exponential: return "Exponential";
        case FalloffType::Custom: return "Custom";
        default: return "Unknown";
    }
}

const char* ForceField::getShapeName(ForceFieldShape s) {
    switch (s) {
        case ForceFieldShape::Infinite: return "Infinite";
        case ForceFieldShape::Sphere: return "Sphere";
        case ForceFieldShape::Box: return "Box";
        case ForceFieldShape::Cylinder: return "Cylinder";
        case ForceFieldShape::Cone: return "Cone";
        default: return "Unknown";
    }
}

const char* ForceField::getIconName() const {
    switch (type) {
        case ForceFieldType::Wind: return "icon_wind";
        case ForceFieldType::Gravity: return "icon_gravity";
        case ForceFieldType::Attractor: return "icon_attractor";
        case ForceFieldType::Repeller: return "icon_repeller";
        case ForceFieldType::Vortex: return "icon_vortex";
        case ForceFieldType::Turbulence: return "icon_turbulence";
        case ForceFieldType::CurlNoise: return "icon_curl";
        case ForceFieldType::Drag: return "icon_drag";
        default: return "icon_force";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCE FIELD MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

int ForceFieldManager::addForceField(std::shared_ptr<ForceField> field) {
    if (!field) return -1;
    field->id = next_id++;
    force_fields.push_back(field);
    return field->id;
}

bool ForceFieldManager::removeForceField(std::shared_ptr<ForceField> field) {
    if (!field) return false;
    auto it = std::find(force_fields.begin(), force_fields.end(), field);
    if (it != force_fields.end()) {
        force_fields.erase(it);
        return true;
    }
    return false;
}

bool ForceFieldManager::removeForceField(int id) {
    auto it = std::find_if(force_fields.begin(), force_fields.end(),
        [id](const std::shared_ptr<ForceField>& f) { return f && f->id == id; });
    if (it != force_fields.end()) {
        force_fields.erase(it);
        return true;
    }
    return false;
}

std::shared_ptr<ForceField> ForceFieldManager::findByName(const std::string& name) const {
    for (const auto& f : force_fields) {
        if (f && f->name == name) return f;
    }
    return nullptr;
}

std::shared_ptr<ForceField> ForceFieldManager::findById(int id) const {
    for (const auto& f : force_fields) {
        if (f && f->id == id) return f;
    }
    return nullptr;
}

Vec3 ForceFieldManager::evaluateAtFiltered(const Vec3& world_pos, float time,
                                           const Vec3& velocity,
                                           bool is_gas, bool is_particle,
                                           bool is_cloth, bool is_rigidbody) const {
    Vec3 total_force(0, 0, 0);
    
    for (const auto& field : force_fields) {
        if (!field || !field->enabled) continue;
        
        // Check affect masks
        if (is_gas && !field->affects_gas) continue;
        if (is_particle && !field->affects_particles) continue;
        if (is_cloth && !field->affects_cloth) continue;
        if (is_rigidbody && !field->affects_rigidbody) continue;
        
        total_force = total_force + field->evaluate(world_pos, time, velocity);
    }
    
    return total_force;
}

Vec3 ForceFieldManager::evaluateAt(const Vec3& world_pos, float time, const Vec3& velocity) const {
    Vec3 total_force(0, 0, 0);
    
    for (const auto& field : force_fields) {
        if (!field || !field->enabled) continue;
        total_force = total_force + field->evaluate(world_pos, time, velocity);
    }
    
    return total_force;
}

Vec3 ForceFieldManager::evaluateAt(const Vec3& world_pos, float time) const {
    return evaluateAt(world_pos, time, Vec3(0, 0, 0));
}

size_t ForceFieldManager::getActiveCount() const {
    size_t count = 0;
    for (const auto& f : force_fields) {
        if (f && f->enabled) ++count;
    }
    return count;
}

nlohmann::json ForceFieldManager::toJson() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& f : force_fields) {
        if (f) arr.push_back(f->toJson());
    }
    return arr;
}

void ForceFieldManager::fromJson(const nlohmann::json& j) {
    force_fields.clear();
    next_id = 0;
    
    if (!j.is_array()) return;
    
    for (const auto& item : j) {
        auto field = std::make_shared<ForceField>();
        field->fromJson(item);
        addForceField(field);
    }
}

} // namespace Physics
