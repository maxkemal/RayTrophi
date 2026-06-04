/**
 * @file EmitterSystem.cpp
 * @brief Implementation of Advanced Emitter System
 */

#include "EmitterSystem.h"
#include "CurlNoise.h"
#include <cmath>
#include <algorithm>
#include <random>

namespace Physics {

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER NOISE SETTINGS - SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json EmitterNoiseSettings::toJson() const {
    nlohmann::json j;
    j["enabled"] = enabled;
    j["frequency"] = frequency;
    j["amplitude"] = amplitude;
    j["octaves"] = octaves;
    j["lacunarity"] = lacunarity;
    j["persistence"] = persistence;
    j["seed"] = seed;
    j["speed"] = speed;
    j["modulate_density"] = modulate_density;
    j["modulate_temperature"] = modulate_temperature;
    j["modulate_velocity"] = modulate_velocity;
    return j;
}

void EmitterNoiseSettings::fromJson(const nlohmann::json& j) {
    if (j.contains("enabled")) enabled = j["enabled"];
    if (j.contains("frequency")) frequency = j["frequency"];
    if (j.contains("amplitude")) amplitude = j["amplitude"];
    if (j.contains("octaves")) octaves = j["octaves"];
    if (j.contains("lacunarity")) lacunarity = j["lacunarity"];
    if (j.contains("persistence")) persistence = j["persistence"];
    if (j.contains("seed")) seed = j["seed"];
    if (j.contains("speed")) speed = j["speed"];
    if (j.contains("modulate_density")) modulate_density = j["modulate_density"];
    if (j.contains("modulate_temperature")) modulate_temperature = j["modulate_temperature"];
    if (j.contains("modulate_velocity")) modulate_velocity = j["modulate_velocity"];
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER FALLOFF
// ═══════════════════════════════════════════════════════════════════════════════

float EmitterFalloff::calculate(float normalized_distance) const {
    if (normalized_distance <= falloff_start) return 1.0f;
    if (normalized_distance >= falloff_end) return 0.0f;
    
    float t = (normalized_distance - falloff_start) / (falloff_end - falloff_start);
    
    switch (type) {
        case EmitterFalloffType::None:
            return 1.0f;
            
        case EmitterFalloffType::Linear:
            return 1.0f - t;
            
        case EmitterFalloffType::Smooth:
            return 1.0f - t * t * (3.0f - 2.0f * t);
            
        case EmitterFalloffType::Spherical:
            return std::sqrt(1.0f - t * t);
            
        case EmitterFalloffType::Gaussian:
            return std::exp(-4.0f * t * t);
            
        case EmitterFalloffType::Custom:
            if (custom_curve.empty()) return 1.0f - t;
            // Lookup in custom curve
            {
                float idx = t * (custom_curve.size() - 1);
                int i0 = (int)idx;
                int i1 = std::min(i0 + 1, (int)custom_curve.size() - 1);
                float frac = idx - i0;
                return custom_curve[i0] * (1.0f - frac) + custom_curve[i1] * frac;
            }
            
        default:
            return 1.0f - t;
    }
}

nlohmann::json EmitterFalloff::toJson() const {
    nlohmann::json j;
    j["type"] = static_cast<int>(type);
    j["falloff_start"] = falloff_start;
    j["falloff_end"] = falloff_end;
    if (!custom_curve.empty()) {
        j["custom_curve"] = custom_curve;
    }
    return j;
}

void EmitterFalloff::fromJson(const nlohmann::json& j) {
    if (j.contains("type")) type = static_cast<EmitterFalloffType>(j["type"].get<int>());
    if (j.contains("falloff_start")) falloff_start = j["falloff_start"];
    if (j.contains("falloff_end")) falloff_end = j["falloff_end"];
    if (j.contains("custom_curve")) {
        custom_curve = j["custom_curve"].get<std::vector<float>>();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VELOCITY VARIANCE - SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json VelocityVariance::toJson() const {
    nlohmann::json j;
    j["speed_min"] = speed_min;
    j["speed_max"] = speed_max;
    j["cone_angle"] = cone_angle;
    j["spread"] = spread;
    j["inherit_velocity"] = inherit_velocity;
    j["inherit_factor"] = inherit_factor;
    return j;
}

void VelocityVariance::fromJson(const nlohmann::json& j) {
    if (j.contains("speed_min")) speed_min = j["speed_min"];
    if (j.contains("speed_max")) speed_max = j["speed_max"];
    if (j.contains("cone_angle")) cone_angle = j["cone_angle"];
    if (j.contains("spread")) spread = j["spread"];
    if (j.contains("inherit_velocity")) inherit_velocity = j["inherit_velocity"];
    if (j.contains("inherit_factor")) inherit_factor = j["inherit_factor"];
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMISSION PROFILE
// ═══════════════════════════════════════════════════════════════════════════════

float EmissionProfile::getRateAtFrame(float frame) const {
    if (!use_rate_curve || rate_curve.empty()) return 1.0f;
    
    // Find surrounding keyframes
    if (frame <= rate_curve.front().first) return rate_curve.front().second;
    if (frame >= rate_curve.back().first) return rate_curve.back().second;
    
    for (size_t i = 1; i < rate_curve.size(); ++i) {
        if (frame <= rate_curve[i].first) {
            float t = (frame - rate_curve[i-1].first) / 
                      (rate_curve[i].first - rate_curve[i-1].first);
            return rate_curve[i-1].second * (1.0f - t) + rate_curve[i].second * t;
        }
    }
    
    return 1.0f;
}

nlohmann::json EmissionProfile::toJson() const {
    nlohmann::json j;
    j["mode"] = static_cast<int>(mode);
    j["start_frame"] = start_frame;
    j["end_frame"] = end_frame;
    j["warmup_frames"] = warmup_frames;
    j["burst_count"] = burst_count;
    j["pulse_interval"] = pulse_interval;
    j["pulse_duration"] = pulse_duration;
    j["use_rate_curve"] = use_rate_curve;
    
    if (!rate_curve.empty()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& pair : rate_curve) {
            arr.push_back({ pair.first, pair.second });
        }
        j["rate_curve"] = arr;
    }
    
    return j;
}

void EmissionProfile::fromJson(const nlohmann::json& j) {
    if (j.contains("mode")) mode = static_cast<EmissionMode>(j["mode"].get<int>());
    if (j.contains("start_frame")) start_frame = j["start_frame"];
    if (j.contains("end_frame")) end_frame = j["end_frame"];
    if (j.contains("warmup_frames")) warmup_frames = j["warmup_frames"];
    if (j.contains("burst_count")) burst_count = j["burst_count"];
    if (j.contains("pulse_interval")) pulse_interval = j["pulse_interval"];
    if (j.contains("pulse_duration")) pulse_duration = j["pulse_duration"];
    if (j.contains("use_rate_curve")) use_rate_curve = j["use_rate_curve"];
    
    if (j.contains("rate_curve")) {
        rate_curve.clear();
        for (const auto& item : j["rate_curve"]) {
            if (item.is_array() && item.size() >= 2) {
                rate_curve.push_back({ item[0].get<float>(), item[1].get<float>() });
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - TRANSFORM HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 AdvancedEmitter::worldToLocal(const Vec3& world_pos) const {
    Vec3 local = world_pos - position;
    
    // Apply inverse scale
    if (std::abs(scale.x) > 0.001f) local.x /= scale.x;
    if (std::abs(scale.y) > 0.001f) local.y /= scale.y;
    if (std::abs(scale.z) > 0.001f) local.z /= scale.z;
    
    // TODO: Apply inverse rotation
    
    return local;
}

Vec3 AdvancedEmitter::localToWorld(const Vec3& local_pos) const {
    Vec3 world = local_pos;
    
    // Apply scale
    world.x *= scale.x;
    world.y *= scale.y;
    world.z *= scale.z;
    
    // TODO: Apply rotation
    
    // Apply translation
    return world + position;
}

Vec3 AdvancedEmitter::transformDirection(const Vec3& dir) const {
    // TODO: Apply rotation to direction
    return dir;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - SHAPE CHECKS
// ═══════════════════════════════════════════════════════════════════════════════

bool AdvancedEmitter::isInsideSphere(const Vec3& local_pos) const {
    float dist_sq = local_pos.x * local_pos.x + 
                    local_pos.y * local_pos.y + 
                    local_pos.z * local_pos.z;
    return dist_sq <= radius * radius;
}

bool AdvancedEmitter::isInsideBox(const Vec3& local_pos) const {
    return std::abs(local_pos.x) <= size.x &&
           std::abs(local_pos.y) <= size.y &&
           std::abs(local_pos.z) <= size.z;
}

bool AdvancedEmitter::isInsideCylinder(const Vec3& local_pos) const {
    float radial_dist_sq = local_pos.x * local_pos.x + local_pos.z * local_pos.z;
    return radial_dist_sq <= radius * radius && 
           local_pos.y >= 0 && local_pos.y <= height;
}

bool AdvancedEmitter::isInsideCone(const Vec3& local_pos) const {
    if (local_pos.y < 0 || local_pos.y > height) return false;
    
    float ratio = local_pos.y / height;
    float cone_rad = std::tan(cone_angle * 0.5f * 3.14159f / 180.0f) * height;
    float allowed_radius = cone_rad * ratio;
    
    float radial_dist_sq = local_pos.x * local_pos.x + local_pos.z * local_pos.z;
    return radial_dist_sq <= allowed_radius * allowed_radius;
}

bool AdvancedEmitter::isInsideDisc(const Vec3& local_pos) const {
    if (std::abs(local_pos.y) > 0.1f) return false; // Thin disc
    
    float radial_dist_sq = local_pos.x * local_pos.x + local_pos.z * local_pos.z;
    return radial_dist_sq <= radius * radius && 
           radial_dist_sq >= inner_radius * inner_radius;
}

bool AdvancedEmitter::isInsideVolume(const Vec3& world_pos) const {
    Vec3 local = worldToLocal(world_pos);
    
    switch (shape) {
        case EmitterShape::Point:
            return (world_pos - position).length() < 0.001f;
        case EmitterShape::Sphere:
            return isInsideSphere(local);
        case EmitterShape::Box:
            return isInsideBox(local);
        case EmitterShape::Cylinder:
            return isInsideCylinder(local);
        case EmitterShape::Cone:
            return isInsideCone(local);
        case EmitterShape::Disc:
            return isInsideDisc(local);
        case EmitterShape::Mesh:
            // TODO: Implement mesh containment check
            return false;
        case EmitterShape::Curve:
            // TODO: Implement curve proximity check
            return false;
        case EmitterShape::Image:
            // TODO: Implement image mask check
            return false;
        default:
            return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 AdvancedEmitter::sampleSphere(float r1, float r2, float r3) const {
    // Uniform sampling inside sphere
    float theta = 2.0f * 3.14159f * r1;
    float phi = std::acos(2.0f * r2 - 1.0f);
    float r = radius * std::cbrt(r3); // Cube root for uniform volume distribution
    
    return Vec3(
        r * std::sin(phi) * std::cos(theta),
        r * std::sin(phi) * std::sin(theta),
        r * std::cos(phi)
    );
}

Vec3 AdvancedEmitter::sampleBox(float r1, float r2, float r3) const {
    return Vec3(
        (r1 * 2.0f - 1.0f) * size.x,
        (r2 * 2.0f - 1.0f) * size.y,
        (r3 * 2.0f - 1.0f) * size.z
    );
}

Vec3 AdvancedEmitter::sampleCylinder(float r1, float r2, float r3) const {
    float theta = 2.0f * 3.14159f * r1;
    float r = radius * std::sqrt(r2); // Square root for uniform disc distribution
    float h = r3 * height;
    
    return Vec3(r * std::cos(theta), h, r * std::sin(theta));
}

Vec3 AdvancedEmitter::sampleCone(float r1, float r2, float r3) const {
    float h = r3 * height;
    float cone_rad = std::tan(cone_angle * 0.5f * 3.14159f / 180.0f) * height;
    float max_r = cone_rad * (h / height);
    
    float theta = 2.0f * 3.14159f * r1;
    float r = max_r * std::sqrt(r2);
    
    return Vec3(r * std::cos(theta), h, r * std::sin(theta));
}

Vec3 AdvancedEmitter::sampleDisc(float r1, float r2) const {
    float theta = 2.0f * 3.14159f * r1;
    float r_range = radius - inner_radius;
    float r = inner_radius + r_range * std::sqrt(r2);
    
    return Vec3(r * std::cos(theta), 0, r * std::sin(theta));
}

Vec3 AdvancedEmitter::sampleRandomPoint(float r1, float r2, float r3) const {
    Vec3 local_point;
    
    switch (shape) {
        case EmitterShape::Point:
            local_point = Vec3(0, 0, 0);
            break;
        case EmitterShape::Sphere:
            local_point = sampleSphere(r1, r2, r3);
            break;
        case EmitterShape::Box:
            local_point = sampleBox(r1, r2, r3);
            break;
        case EmitterShape::Cylinder:
            local_point = sampleCylinder(r1, r2, r3);
            break;
        case EmitterShape::Cone:
            local_point = sampleCone(r1, r2, r3);
            break;
        case EmitterShape::Disc:
            local_point = sampleDisc(r1, r2);
            break;
        default:
            local_point = Vec3(0, 0, 0);
    }
    
    // Add position jitter
    if (position_jitter > 0) {
        local_point.x += (r1 - 0.5f) * position_jitter;
        local_point.y += (r2 - 0.5f) * position_jitter;
        local_point.z += (r3 - 0.5f) * position_jitter;
    }
    
    return localToWorld(local_point);
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - EMISSION STRENGTH
// ═══════════════════════════════════════════════════════════════════════════════

float AdvancedEmitter::getEmissionStrength(const Vec3& world_pos, float time) const {
    if (!enabled) return 0.0f;
    if (!isInsideVolume(world_pos)) return 0.0f;
    
    Vec3 local = worldToLocal(world_pos);
    float strength = 1.0f;
    
    // Calculate normalized distance from center
    float normalized_dist = 0.0f;
    switch (shape) {
        case EmitterShape::Sphere:
            normalized_dist = local.length() / radius;
            break;
        case EmitterShape::Box:
            normalized_dist = std::max({
                std::abs(local.x) / size.x,
                std::abs(local.y) / size.y,
                std::abs(local.z) / size.z
            });
            break;
        case EmitterShape::Cylinder: {
            float radial = std::sqrt(local.x * local.x + local.z * local.z) / radius;
            float axial = local.y / height;
            normalized_dist = std::max(radial, std::abs(axial - 0.5f) * 2.0f);
            break;
        }
        default:
            normalized_dist = local.length() / radius;
    }
    
    // Apply falloff
    strength *= falloff.calculate(normalized_dist);
    
    // Apply noise modulation
    if (noise.enabled && noise.modulate_density) {
        float noise_val = Noise::fbm3D_animated(
            world_pos, time * noise.speed,
            noise.octaves, noise.frequency,
            noise.lacunarity, noise.persistence, noise.seed
        );
        // Map noise from [-1, 1] to [1 - amplitude, 1 + amplitude]
        strength *= (1.0f + noise_val * noise.amplitude);
        strength = std::max(0.0f, strength);
    }
    
    return strength;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - VELOCITY
// ═══════════════════════════════════════════════════════════════════════════════

Vec3 AdvancedEmitter::getEmissionVelocity(const Vec3& world_pos, const Vec3& surface_normal, float time) const {
    Vec3 base_velocity = velocity;
    
    switch (velocity_mode) {
        case VelocityMode::Constant:
            base_velocity = velocity;
            break;
            
        case VelocityMode::Normal:
            base_velocity = surface_normal * velocity_magnitude;
            break;
            
        case VelocityMode::Radial: {
            Vec3 dir = world_pos - position;
            float len = dir.length();
            if (len > 0.001f) {
                base_velocity = (dir / len) * velocity_magnitude;
            }
            break;
        }
        
        case VelocityMode::Random:
        case VelocityMode::Tangent:
        case VelocityMode::Custom:
            base_velocity = velocity;
            break;
    }
    
    // Apply velocity variance
    if (velocity_variance.cone_angle > 0 || velocity_variance.speed_min != velocity_variance.speed_max) {
        // Simple random speed variation (could use hash for consistency)
        float speed_mult = velocity_variance.speed_min + 
                          (velocity_variance.speed_max - velocity_variance.speed_min) * 0.5f;
        base_velocity = base_velocity * speed_mult;
    }
    
    // Apply noise to velocity
    if (noise.enabled && noise.modulate_velocity) {
        Vec3 noise_offset = Noise::curl3D_animated(
            world_pos, time, 
            noise.frequency, noise.speed, noise.seed + 500
        );
        base_velocity = base_velocity + noise_offset * noise.amplitude;
    }
    
    return base_velocity;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - TIME
// ═══════════════════════════════════════════════════════════════════════════════

bool AdvancedEmitter::isActiveAtFrame(float frame) const {
    if (!enabled) return false;
    if (frame < profile.start_frame) return false;
    if (profile.end_frame >= 0 && frame > profile.end_frame) return false;
    
    if (profile.mode == EmissionMode::Pulse) {
        float elapsed = frame - profile.start_frame;
        float cycle_pos = std::fmod(elapsed, profile.pulse_interval);
        return cycle_pos < profile.pulse_duration;
    }
    
    return true;
}

float AdvancedEmitter::getEmissionRate(float frame) const {
    if (!isActiveAtFrame(frame)) return 0.0f;
    
    float rate = density_rate;
    
    // Apply profile curve
    rate *= profile.getRateAtFrame(frame);
    
    return rate;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - BOUNDS
// ═══════════════════════════════════════════════════════════════════════════════

void AdvancedEmitter::getBounds(Vec3& min_out, Vec3& max_out) const {
    Vec3 extent;
    
    switch (shape) {
        case EmitterShape::Sphere:
            extent = Vec3(radius, radius, radius);
            break;
        case EmitterShape::Box:
            extent = size;
            break;
        case EmitterShape::Cylinder:
            extent = Vec3(radius, height, radius);
            break;
        case EmitterShape::Cone:
            extent = Vec3(radius, height, radius);
            break;
        case EmitterShape::Disc:
            extent = Vec3(radius, 0.1f, radius);
            break;
        default:
            extent = Vec3(radius, radius, radius);
    }
    
    // Apply scale
    extent.x *= scale.x;
    extent.y *= scale.y;
    extent.z *= scale.z;
    
    min_out = position - extent;
    max_out = position + extent;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json AdvancedEmitter::toJson() const {
    nlohmann::json j;
    
    j["name"] = name;
    j["enabled"] = enabled;
    j["shape"] = static_cast<int>(shape);
    
    j["position"] = { position.x, position.y, position.z };
    j["rotation"] = { rotation.x, rotation.y, rotation.z };
    j["scale"] = { scale.x, scale.y, scale.z };
    
    j["radius"] = radius;
    j["size"] = { size.x, size.y, size.z };
    j["height"] = height;
    j["inner_radius"] = inner_radius;
    j["cone_angle"] = cone_angle;
    
    j["density_rate"] = density_rate;
    j["temperature"] = temperature;
    j["fuel"] = fuel;
    j["color"] = { color.x, color.y, color.z };
    
    j["velocity_mode"] = static_cast<int>(velocity_mode);
    j["velocity"] = { velocity.x, velocity.y, velocity.z };
    j["velocity_magnitude"] = velocity_magnitude;
    j["velocity_variance"] = velocity_variance.toJson();
    
    j["noise"] = noise.toJson();
    j["falloff"] = falloff.toJson();
    j["profile"] = profile.toJson();
    
    j["position_jitter"] = position_jitter;
    j["random_seed"] = random_seed;
    
    return j;
}

void AdvancedEmitter::fromJson(const nlohmann::json& j) {
    if (j.contains("name")) name = j["name"];
    if (j.contains("enabled")) enabled = j["enabled"];
    if (j.contains("shape")) shape = static_cast<EmitterShape>(j["shape"].get<int>());
    
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
    
    if (j.contains("radius")) radius = j["radius"];
    if (j.contains("size")) {
        auto s = j["size"];
        size = Vec3(s[0], s[1], s[2]);
    }
    if (j.contains("height")) height = j["height"];
    if (j.contains("inner_radius")) inner_radius = j["inner_radius"];
    if (j.contains("cone_angle")) cone_angle = j["cone_angle"];
    
    if (j.contains("density_rate")) density_rate = j["density_rate"];
    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("fuel")) fuel = j["fuel"];
    if (j.contains("color")) {
        auto c = j["color"];
        color = Vec3(c[0], c[1], c[2]);
    }
    
    if (j.contains("velocity_mode")) velocity_mode = static_cast<VelocityMode>(j["velocity_mode"].get<int>());
    if (j.contains("velocity")) {
        auto v = j["velocity"];
        velocity = Vec3(v[0], v[1], v[2]);
    }
    if (j.contains("velocity_magnitude")) velocity_magnitude = j["velocity_magnitude"];
    if (j.contains("velocity_variance")) velocity_variance.fromJson(j["velocity_variance"]);
    
    if (j.contains("noise")) noise.fromJson(j["noise"]);
    if (j.contains("falloff")) falloff.fromJson(j["falloff"]);
    if (j.contains("profile")) profile.fromJson(j["profile"]);
    
    if (j.contains("position_jitter")) position_jitter = j["position_jitter"];
    if (j.contains("random_seed")) random_seed = j["random_seed"];
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVANCED EMITTER - UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

const char* AdvancedEmitter::getShapeName(EmitterShape s) {
    switch (s) {
        case EmitterShape::Point: return "Point";
        case EmitterShape::Sphere: return "Sphere";
        case EmitterShape::Box: return "Box";
        case EmitterShape::Cylinder: return "Cylinder";
        case EmitterShape::Cone: return "Cone";
        case EmitterShape::Disc: return "Disc";
        case EmitterShape::Mesh: return "Mesh";
        case EmitterShape::Curve: return "Curve";
        case EmitterShape::Image: return "Image";
        default: return "Unknown";
    }
}

const char* AdvancedEmitter::getVelocityModeName(VelocityMode m) {
    switch (m) {
        case VelocityMode::Constant: return "Constant";
        case VelocityMode::Normal: return "Normal";
        case VelocityMode::Random: return "Random";
        case VelocityMode::Radial: return "Radial";
        case VelocityMode::Tangent: return "Tangent";
        case VelocityMode::Custom: return "Custom";
        default: return "Unknown";
    }
}

const char* AdvancedEmitter::getIconName() const {
    switch (shape) {
        case EmitterShape::Sphere: return "icon_sphere";
        case EmitterShape::Box: return "icon_box";
        case EmitterShape::Cylinder: return "icon_cylinder";
        case EmitterShape::Cone: return "icon_cone";
        default: return "icon_emitter";
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

int EmitterManager::addEmitter(std::shared_ptr<AdvancedEmitter> emitter) {
    if (!emitter) return -1;
    emitter->id = next_id++;
    emitters.push_back(emitter);
    return emitter->id;
}

bool EmitterManager::removeEmitter(int id) {
    auto it = std::find_if(emitters.begin(), emitters.end(),
        [id](const std::shared_ptr<AdvancedEmitter>& e) { return e && e->id == id; });
    if (it != emitters.end()) {
        emitters.erase(it);
        return true;
    }
    return false;
}

bool EmitterManager::removeEmitter(std::shared_ptr<AdvancedEmitter> emitter) {
    if (!emitter) return false;
    auto it = std::find(emitters.begin(), emitters.end(), emitter);
    if (it != emitters.end()) {
        emitters.erase(it);
        return true;
    }
    return false;
}

std::shared_ptr<AdvancedEmitter> EmitterManager::findByName(const std::string& name) const {
    for (const auto& e : emitters) {
        if (e && e->name == name) return e;
    }
    return nullptr;
}

std::shared_ptr<AdvancedEmitter> EmitterManager::findById(int id) const {
    for (const auto& e : emitters) {
        if (e && e->id == id) return e;
    }
    return nullptr;
}

float EmitterManager::getCombinedEmission(const Vec3& world_pos, float time) const {
    float total = 0.0f;
    for (const auto& e : emitters) {
        if (e && e->enabled) {
            total += e->getEmissionStrength(world_pos, time);
        }
    }
    return total;
}

nlohmann::json EmitterManager::toJson() const {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& e : emitters) {
        if (e) arr.push_back(e->toJson());
    }
    return arr;
}

void EmitterManager::fromJson(const nlohmann::json& j) {
    emitters.clear();
    next_id = 0;
    
    if (!j.is_array()) return;
    
    for (const auto& item : j) {
        auto emitter = std::make_shared<AdvancedEmitter>();
        emitter->fromJson(item);
        addEmitter(emitter);
    }
}

} // namespace Physics
