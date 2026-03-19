/**
 * @file GasSimulator.cpp
 * @brief CPU implementation of gas/smoke simulation
 */

#include "GasSimulator.h"
#include "ForceField.h"
#include "gas_kernels.cuh"
#include "gas_fft_solver.cuh"  // FFT Pressure Solver & Advanced Emitters
#include "CurlNoise.h"
#include "globals.h" // For SCENE_LOG_*
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <cstdio>
#include <string>
#include <omp.h> // Include OpenMP for parallelization🎬

#ifndef IMATH_DLL
#define IMATH_DLL
#endif
#ifndef OPENEXR_DLL
#define OPENEXR_DLL
#endif

#include <cuda_runtime.h> // Required for thread context if needed

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Interpolation.h>

namespace FluidSim {

namespace {

constexpr float kPi = 3.14159265359f;

float degreesToRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

int mapEmitterShapeToGPU(EmitterShape shape) {
    switch (shape) {
        case EmitterShape::Point: return 0;
        case EmitterShape::Sphere: return 1;
        case EmitterShape::Box: return 2;
        case EmitterShape::Cylinder: return 3;
        case EmitterShape::Cone: return 4;
        case EmitterShape::Disc: return 5;
        default: return 1;
    }
}

float hashNoise3D(int x, int y, int z, int seed) {
    unsigned int n = static_cast<unsigned int>(x + y * 57 + z * 131 + seed * 1373);
    n = (n << 13U) ^ n;
    unsigned int m = (n * (n * n * 15731U + 789221U) + 1376312589U) & 0x7fffffffU;
    return static_cast<float>(m) / static_cast<float>(0x7fffffffU) * 2.0f - 1.0f;
}

float sampleEmitterNoise(const Vec3& p, float frequency, int seed, float time, float speed) {
    const Vec3 q = p * frequency + Vec3(time * speed, 0.0f, 0.0f);
    const int ix = static_cast<int>(std::floor(q.x));
    const int iy = static_cast<int>(std::floor(q.y));
    const int iz = static_cast<int>(std::floor(q.z));
    return hashNoise3D(ix, iy, iz, seed);
}

float computeEmitterFuelPhaseStrength(const Emitter& emitter,
                                      float local_temperature,
                                      float ambient_temperature,
                                      float flame_activity) {
    const float release = std::max(0.0f, emitter.fuel_release_rate);
    const float contact = std::clamp(flame_activity * std::max(0.0f, emitter.flame_contact_sensitivity), 0.0f, 1.0f);
    const float phase_temp = std::max(emitter.phase_change_temperature, ambient_temperature + 1.0f);
    const float thermal_norm = std::clamp((local_temperature - ambient_temperature) / std::max(phase_temp - ambient_temperature, 1.0f), 0.0f, 2.5f);

    switch (emitter.fuel_phase) {
        case FuelPhase::Gas:
            return std::clamp(release, 0.0f, 4.0f);
        case FuelPhase::Liquid: {
            const float warm_vapor = std::clamp((thermal_norm - 0.35f) / 0.85f, 0.0f, 1.0f);
            const float release_factor = std::max(contact * 0.55f, warm_vapor);
            return std::clamp(release * (0.08f + 0.92f * release_factor), 0.0f, 4.0f);
        }
        case FuelPhase::Solid: {
            const float pyrolysis = std::clamp((thermal_norm - 0.8f) / 0.9f, 0.0f, 1.0f);
            const float release_factor = std::max(contact, pyrolysis);
            return std::clamp(release * (0.02f + 0.45f * release_factor), 0.0f, 4.0f);
        }
        default:
            return 1.0f;
    }
}

float computeEmitterHeatPhaseStrength(const Emitter& emitter,
                                      float local_temperature,
                                      float ambient_temperature,
                                      float flame_activity) {
    const float phase_temp = std::max(emitter.phase_change_temperature, ambient_temperature + 1.0f);
    const float thermal_norm = std::clamp((local_temperature - ambient_temperature) / std::max(phase_temp - ambient_temperature, 1.0f), 0.0f, 2.5f);

    switch (emitter.fuel_phase) {
        case FuelPhase::Gas:
            return 1.0f;
        case FuelPhase::Liquid: {
            const float warmup = std::clamp((thermal_norm - 0.25f) / 0.9f, 0.0f, 1.0f);
            return 0.18f + 0.82f * std::max(flame_activity * 0.5f, warmup);
        }
        case FuelPhase::Solid: {
            const float pyrolysis = std::clamp((thermal_norm - 0.75f) / 1.0f, 0.0f, 1.0f);
            return 0.08f + 0.52f * std::max(flame_activity, pyrolysis);
        }
        default:
            return 1.0f;
    }
}

bool isEmitterActiveAtFrame(const Emitter& emitter, float frame) {
    if (!emitter.enabled) return false;
    if (frame < emitter.start_frame) return false;
    if (emitter.end_frame >= 0.0f && frame > emitter.end_frame) return false;
    if (emitter.emission_mode == EmitterEmissionMode::Pulse) {
        const float elapsed = frame - emitter.start_frame;
        if (emitter.pulse_interval <= 0.0f) return true;
        const float cycle_pos = std::fmod(std::max(0.0f, elapsed), emitter.pulse_interval);
        return cycle_pos < emitter.pulse_duration;
    }
    if (emitter.emission_mode == EmitterEmissionMode::Burst) {
        return frame <= emitter.start_frame + 1.0f;
    }
    return true;
}

bool isInsideEmitter(const Emitter& emitter, const Vec3& cell_pos, float& normalized_dist) {
    const Vec3 local = cell_pos - emitter.position;
    switch (emitter.shape) {
        case EmitterShape::Sphere: {
            const float r = std::max(emitter.radius, 0.001f);
            normalized_dist = local.length() / r;
            return normalized_dist <= 1.0f;
        }
        case EmitterShape::Box: {
            if (std::abs(local.x) > emitter.size.x || std::abs(local.y) > emitter.size.y || std::abs(local.z) > emitter.size.z) {
                return false;
            }
            normalized_dist = std::max({std::abs(local.x) / std::max(emitter.size.x, 0.001f),
                                        std::abs(local.y) / std::max(emitter.size.y, 0.001f),
                                        std::abs(local.z) / std::max(emitter.size.z, 0.001f)});
            return true;
        }
        case EmitterShape::Point: {
            const float r = std::max(emitter.radius, 0.001f);
            normalized_dist = local.length() / r;
            return local.length() <= r;
        }
        case EmitterShape::Cylinder: {
            const float radial = std::sqrt(local.x * local.x + local.z * local.z);
            if (radial > emitter.radius || local.y < 0.0f || local.y > emitter.height) return false;
            normalized_dist = std::max(radial / std::max(emitter.radius, 0.001f),
                                       std::abs(local.y / std::max(emitter.height, 0.001f) - 0.5f) * 2.0f);
            return true;
        }
        case EmitterShape::Cone: {
            if (local.y < 0.0f || local.y > emitter.height) return false;
            const float height = std::max(emitter.height, 0.001f);
            const float base_radius = std::max(emitter.radius, 0.001f);
            const float vertical_ratio = std::clamp(local.y / height, 0.0f, 1.0f);
            const float angle_scale = std::clamp(emitter.cone_angle / 45.0f, 0.25f, 2.5f);
            const float allowed_r = std::max(0.001f, base_radius * (1.0f - vertical_ratio) * angle_scale);
            const float radial = std::sqrt(local.x * local.x + local.z * local.z);
            if (radial > allowed_r) return false;
            normalized_dist = std::max(radial / std::max(allowed_r, 0.001f), vertical_ratio);
            return true;
        }
        case EmitterShape::Disc: {
            if (std::abs(local.y) > 0.5f * std::max(emitter.size.y, 0.1f)) return false;
            const float radial = std::sqrt(local.x * local.x + local.z * local.z);
            if (radial > emitter.radius || radial < emitter.inner_radius) return false;
            normalized_dist = radial / std::max(emitter.radius, 0.001f);
            return true;
        }
        default:
            return false;
    }
}

float computeEmitterFalloff(const Emitter& emitter, float normalized_dist) {
    if (emitter.falloff_type == EmitterFalloffType::None) return 1.0f;
    if (normalized_dist <= emitter.falloff_start) return 1.0f;
    if (normalized_dist >= emitter.falloff_end) return 0.0f;
    const float denom = std::max(emitter.falloff_end - emitter.falloff_start, 0.0001f);
    const float t = (normalized_dist - emitter.falloff_start) / denom;
    switch (emitter.falloff_type) {
        case EmitterFalloffType::Linear: return 1.0f - t;
        case EmitterFalloffType::Smooth: return 1.0f - t * t * (3.0f - 2.0f * t);
        case EmitterFalloffType::Gaussian: return std::exp(-4.0f * t * t);
        default: return 1.0f;
    }
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json Emitter::toJson() const {
    nlohmann::json j;
    j["shape"] = static_cast<int>(shape);
    j["position"] = {position.x, position.y, position.z};
    j["size"] = {size.x, size.y, size.z};
    j["radius"] = radius;
    j["height"] = height;
    j["inner_radius"] = inner_radius;
    j["cone_angle"] = cone_angle;
    j["density_rate"] = density_rate;
    j["fuel_rate"] = fuel_rate;
    j["temperature"] = temperature;
    j["velocity"] = {velocity.x, velocity.y, velocity.z};
    j["fuel_phase"] = static_cast<int>(fuel_phase);
    j["phase_change_temperature"] = phase_change_temperature;
    j["fuel_release_rate"] = fuel_release_rate;
    j["flame_contact_sensitivity"] = flame_contact_sensitivity;
    j["falloff_type"] = static_cast<int>(falloff_type);
    j["falloff_start"] = falloff_start;
    j["falloff_end"] = falloff_end;
    j["noise_enabled"] = noise_enabled;
    j["noise_frequency"] = noise_frequency;
    j["noise_amplitude"] = noise_amplitude;
    j["noise_speed"] = noise_speed;
    j["noise_seed"] = noise_seed;
    j["noise_modulate_density"] = noise_modulate_density;
    j["noise_modulate_temperature"] = noise_modulate_temperature;
    j["noise_modulate_velocity"] = noise_modulate_velocity;
    j["spray_cone_angle"] = spray_cone_angle;
    j["speed_min"] = speed_min;
    j["speed_max"] = speed_max;
    j["emission_mode"] = static_cast<int>(emission_mode);
    j["start_frame"] = start_frame;
    j["end_frame"] = end_frame;
    j["pulse_interval"] = pulse_interval;
    j["pulse_duration"] = pulse_duration;
    j["name"] = name;
    j["uid"] = uid;
    
    nlohmann::json kfs_j = nlohmann::json::object();
    for (auto const& [frame, kf] : keyframes) {
        kfs_j[std::to_string(frame)] = kf;
    }
    j["keyframes"] = kfs_j;
    
    return j;
}

void Emitter::fromJson(const nlohmann::json& j) {
    if (j.contains("shape")) shape = static_cast<EmitterShape>(j["shape"].get<int>());
    if (j.contains("position")) {
        auto p = j["position"];
        position = Vec3(p[0], p[1], p[2]);
    }
    if (j.contains("size")) {
        auto s = j["size"];
        size = Vec3(s[0], s[1], s[2]);
    }
    if (j.contains("radius")) radius = j["radius"];
    if (j.contains("height")) height = j["height"];
    if (j.contains("inner_radius")) inner_radius = j["inner_radius"];
    if (j.contains("cone_angle")) cone_angle = j["cone_angle"];
    if (j.contains("density_rate")) density_rate = j["density_rate"];
    if (j.contains("fuel_rate")) fuel_rate = j["fuel_rate"];
    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("fuel_phase")) fuel_phase = static_cast<FuelPhase>(j["fuel_phase"].get<int>());
    if (j.contains("phase_change_temperature")) phase_change_temperature = j["phase_change_temperature"];
    if (j.contains("fuel_release_rate")) fuel_release_rate = j["fuel_release_rate"];
    if (j.contains("flame_contact_sensitivity")) flame_contact_sensitivity = j["flame_contact_sensitivity"];
    if (j.contains("velocity")) {
        auto v = j["velocity"];
        velocity = Vec3(v[0], v[1], v[2]);
    }
    if (j.contains("falloff_type")) falloff_type = static_cast<EmitterFalloffType>(j["falloff_type"].get<int>());
    if (j.contains("falloff_start")) falloff_start = j["falloff_start"];
    if (j.contains("falloff_end")) falloff_end = j["falloff_end"];
    if (j.contains("noise_enabled")) noise_enabled = j["noise_enabled"];
    if (j.contains("noise_frequency")) noise_frequency = j["noise_frequency"];
    if (j.contains("noise_amplitude")) noise_amplitude = j["noise_amplitude"];
    if (j.contains("noise_speed")) noise_speed = j["noise_speed"];
    if (j.contains("noise_seed")) noise_seed = j["noise_seed"];
    if (j.contains("noise_modulate_density")) noise_modulate_density = j["noise_modulate_density"];
    if (j.contains("noise_modulate_temperature")) noise_modulate_temperature = j["noise_modulate_temperature"];
    if (j.contains("noise_modulate_velocity")) noise_modulate_velocity = j["noise_modulate_velocity"];
    if (j.contains("spray_cone_angle")) spray_cone_angle = j["spray_cone_angle"];
    if (j.contains("speed_min")) speed_min = j["speed_min"];
    if (j.contains("speed_max")) speed_max = j["speed_max"];
    if (j.contains("emission_mode")) emission_mode = static_cast<EmitterEmissionMode>(j["emission_mode"].get<int>());
    if (j.contains("start_frame")) start_frame = j["start_frame"];
    if (j.contains("end_frame")) end_frame = j["end_frame"];
    if (j.contains("pulse_interval")) pulse_interval = j["pulse_interval"];
    if (j.contains("pulse_duration")) pulse_duration = j["pulse_duration"];
    if (j.contains("name")) name = j["name"];
    if (j.contains("uid")) uid = j["uid"];
    
    if (j.contains("keyframes")) {
        keyframes.clear();
        for (auto it = j["keyframes"].begin(); it != j["keyframes"].end(); ++it) {
            int frame = std::stoi(it.key());
            keyframes[frame] = it.value().get<::EmitterKeyframe>();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER KEYFRAME METHODS
// ═══════════════════════════════════════════════════════════════════════════════

::EmitterKeyframe Emitter::getInterpolatedKeyframe(float current_frame) const {
    if (keyframes.empty()) {
        ::EmitterKeyframe kf;
        kf.fuel_rate = fuel_rate; kf.has_fuel_rate = true;
        kf.density_rate = density_rate; kf.has_density_rate = true;
        kf.temperature = temperature; kf.has_temperature = true;
        kf.velocity = velocity; kf.has_velocity = true;
        kf.position = position; kf.has_position = true;
        kf.size = size; kf.has_size = true;
        kf.radius = radius; kf.has_radius = true;
        kf.enabled = enabled; kf.has_enabled = true;
        return kf;
    }
    
    auto it_after = keyframes.lower_bound((int)std::ceil(current_frame));
    
    if (it_after == keyframes.begin()) {
        return it_after->second;
    }
    
    if (it_after == keyframes.end()) {
        return keyframes.rbegin()->second;
    }
    
    auto it_before = std::prev(it_after);
    
    int frame_before = it_before->first;
    int frame_after = it_after->first;
    
    float t = (current_frame - frame_before) / (float)(frame_after - frame_before);
    t = std::max(0.0f, std::min(1.0f, t));
    
    return ::EmitterKeyframe::lerp(it_before->second, it_after->second, t);
}

void Emitter::applyKeyframe(const ::EmitterKeyframe& kf) {
    if (kf.has_fuel_rate) fuel_rate = kf.fuel_rate;
    if (kf.has_density_rate) density_rate = kf.density_rate;
    if (kf.has_temperature) temperature = kf.temperature;
    if (kf.has_velocity) velocity = kf.velocity;
    if (kf.has_position) position = kf.position;
    if (kf.has_size) size = kf.size;
    if (kf.has_radius) radius = kf.radius;
    if (kf.has_enabled) enabled = kf.enabled;
}


// ═══════════════════════════════════════════════════════════════════════════════
// SETTINGS SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json GasSimulationSettings::toJson() const {
    nlohmann::json j;
    j["resolution"] = {resolution_x, resolution_y, resolution_z};
    j["grid_size"] = {grid_size.x, grid_size.y, grid_size.z};
    j["voxel_size"] = voxel_size;
    j["preserve_voxel_size_on_resize"] = preserve_voxel_size_on_resize;
    j["max_auto_resolution"] = max_auto_resolution;
    j["grid_offset"] = {grid_offset.x, grid_offset.y, grid_offset.z};
    j["timestep"] = timestep;
    j["substeps"] = substeps;
    j["time_scale"] = time_scale;
    j["pressure_iterations"] = pressure_iterations;
    
    // CFL Adaptive Timestep
    j["adaptive_timestep"] = adaptive_timestep;
    j["cfl_number"] = cfl_number;
    j["min_timestep"] = min_timestep;
    j["max_timestep"] = max_timestep;
    
    j["density_dissipation"] = density_dissipation;
    j["velocity_dissipation"] = velocity_dissipation;
    j["temperature_dissipation"] = temperature_dissipation;
    j["fuel_dissipation"] = fuel_dissipation;
    
    j["ignition_temperature"] = ignition_temperature;
    j["burn_rate"] = burn_rate;
    j["heat_release"] = heat_release;
    j["expansion_strength"] = expansion_strength;
    j["smoke_generation"] = smoke_generation;
    j["soot_generation"] = soot_generation;
    j["buoyancy_density"] = buoyancy_density;
    j["buoyancy_temperature"] = buoyancy_temperature;
    j["ambient_temperature"] = ambient_temperature;
    j["vorticity_strength"] = vorticity_strength;
    j["turbulence_strength"] = turbulence_strength;
    j["turbulence_scale"] = turbulence_scale;
    j["turbulence_octaves"] = turbulence_octaves;
    j["turbulence_lacunarity"] = turbulence_lacunarity;
    j["turbulence_persistence"] = turbulence_persistence;
    j["advection_mode"] = static_cast<int>(advection_mode);
    j["pressure_solver"] = static_cast<int>(pressure_solver);
    j["boundary_mode"] = static_cast<int>(boundary_mode);
    j["sor_omega"] = sor_omega;
    j["gravity"] = {gravity.x, gravity.y, gravity.z};
    j["wind"] = {wind.x, wind.y, wind.z};
    j["sparse_mode"] = sparse_mode;
    j["sparse_threshold"] = sparse_threshold;
    j["max_velocity"] = max_velocity;
    j["max_temperature"] = max_temperature;
    j["max_density"] = max_density;
    j["mode"] = static_cast<int>(mode);
    j["backend"] = static_cast<int>(backend);
    j["cache_directory"] = cache_directory;
    j["bake_start_frame"] = bake_start_frame;
    j["bake_end_frame"] = bake_end_frame;
    j["bake_fps"] = bake_fps;
    return j;
}

void GasSimulationSettings::fromJson(const nlohmann::json& j) {
    if (j.contains("resolution")) {
        auto r = j["resolution"];
        resolution_x = r[0]; resolution_y = r[1]; resolution_z = r[2];
    }
    if (j.contains("grid_size")) {
        auto s = j["grid_size"];
        grid_size = Vec3(s[0], s[1], s[2]);
    }
    if (j.contains("voxel_size")) voxel_size = j["voxel_size"];
    if (j.contains("preserve_voxel_size_on_resize")) preserve_voxel_size_on_resize = j["preserve_voxel_size_on_resize"];
    if (j.contains("max_auto_resolution")) max_auto_resolution = j["max_auto_resolution"];
    if (j.contains("grid_offset")) {
        auto o = j["grid_offset"];
        grid_offset = Vec3(o[0], o[1], o[2]);
    }
    if (j.contains("timestep")) timestep = j["timestep"];
    if (j.contains("substeps")) substeps = j["substeps"];
    if (j.contains("time_scale")) time_scale = j["time_scale"];
    if (j.contains("pressure_iterations")) pressure_iterations = j["pressure_iterations"];
    
    // CFL Adaptive Timestep
    if (j.contains("adaptive_timestep")) adaptive_timestep = j["adaptive_timestep"];
    if (j.contains("cfl_number")) cfl_number = j["cfl_number"];
    if (j.contains("min_timestep")) min_timestep = j["min_timestep"];
    if (j.contains("max_timestep")) max_timestep = j["max_timestep"];
    
    if (j.contains("density_dissipation")) density_dissipation = j["density_dissipation"];
    if (j.contains("velocity_dissipation")) velocity_dissipation = j["velocity_dissipation"];
    if (j.contains("temperature_dissipation")) temperature_dissipation = j["temperature_dissipation"];
    if (j.contains("fuel_dissipation")) fuel_dissipation = j["fuel_dissipation"];

    if (j.contains("ignition_temperature")) ignition_temperature = j["ignition_temperature"];
    if (j.contains("burn_rate")) burn_rate = j["burn_rate"];
    if (j.contains("heat_release")) heat_release = j["heat_release"];
    if (j.contains("expansion_strength")) expansion_strength = j["expansion_strength"];
    if (j.contains("smoke_generation")) smoke_generation = j["smoke_generation"];
    if (j.contains("soot_generation")) soot_generation = j["soot_generation"];
    if (j.contains("buoyancy_density")) buoyancy_density = j["buoyancy_density"];
    if (j.contains("buoyancy_temperature")) buoyancy_temperature = j["buoyancy_temperature"];
    if (j.contains("ambient_temperature")) ambient_temperature = j["ambient_temperature"];
    if (j.contains("vorticity_strength")) vorticity_strength = j["vorticity_strength"];
    if (j.contains("turbulence_strength")) turbulence_strength = j["turbulence_strength"];
    if (j.contains("turbulence_scale")) turbulence_scale = j["turbulence_scale"];
    if (j.contains("turbulence_octaves")) turbulence_octaves = j["turbulence_octaves"];
    if (j.contains("turbulence_lacunarity")) turbulence_lacunarity = j["turbulence_lacunarity"];
    if (j.contains("turbulence_persistence")) turbulence_persistence = j["turbulence_persistence"];
    if (j.contains("advection_mode")) advection_mode = static_cast<AdvectionMode>(j["advection_mode"].get<int>());
    if (j.contains("pressure_solver")) pressure_solver = static_cast<PressureSolverMode>(j["pressure_solver"].get<int>());
    if (j.contains("boundary_mode")) boundary_mode = static_cast<BoundaryMode>(j["boundary_mode"].get<int>());
    if (j.contains("sor_omega")) sor_omega = j["sor_omega"];
    if (j.contains("gravity")) {
        auto g = j["gravity"];
        gravity = Vec3(g[0], g[1], g[2]);
    }
    if (j.contains("wind")) {
        auto w = j["wind"];
        wind = Vec3(w[0], w[1], w[2]);
    }
    if (j.contains("sparse_mode")) sparse_mode = j["sparse_mode"];
    if (j.contains("sparse_threshold")) sparse_threshold = j["sparse_threshold"];
    if (j.contains("max_velocity")) max_velocity = j["max_velocity"];
    if (j.contains("max_temperature")) max_temperature = j["max_temperature"];
    if (j.contains("max_density")) max_density = j["max_density"];
    if (j.contains("mode")) mode = static_cast<SimulationMode>(j["mode"].get<int>());
    if (j.contains("backend")) backend = static_cast<SolverBackend>(j["backend"].get<int>());
    if (j.contains("cache_directory")) cache_directory = j["cache_directory"];
    if (j.contains("bake_start_frame")) bake_start_frame = j["bake_start_frame"];
    if (j.contains("bake_end_frame")) bake_end_frame = j["bake_end_frame"];
    if (j.contains("bake_fps")) bake_fps = j["bake_fps"];
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTOR / DESTRUCTOR
// ═══════════════════════════════════════════════════════════════════════════════

GasSimulator::GasSimulator() {
#ifdef OPENVDB_ENABLED
    static bool vdb_init = false;
    if (!vdb_init) {
        openvdb::initialize();
        vdb_init = true;
    }
#endif
}

GasSimulator::~GasSimulator() {
    if (bake_thread && bake_thread->joinable()) {
        cancel_bake = true;
        bake_thread->join();
    }
    shutdown();
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIFECYCLE
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::initialize(const GasSimulationSettings& s) {
    settings = s;
    
    // Safety checks - clamp resolution to reasonable range (Increased for more detail)
    settings.resolution_x = std::clamp(settings.resolution_x, 8, 512);
    settings.resolution_y = std::clamp(settings.resolution_y, 8, 512);
    settings.resolution_z = std::clamp(settings.resolution_z, 8, 512);
    settings.max_auto_resolution = std::clamp(settings.max_auto_resolution, 32, 512);
    
    // Calculate voxel size based on grid dimensions and resolution
    // Use the maximum resolution to determine voxel size for uniform voxels
    float max_dim = std::max({settings.grid_size.x, settings.grid_size.y, settings.grid_size.z});
    int max_res = std::max({settings.resolution_x, settings.resolution_y, settings.resolution_z});
    
    // Determine the master voxel size based on the longest axis
    settings.voxel_size = max_dim / (float)max_res;
    if (settings.voxel_size < 0.0001f) settings.voxel_size = 0.1f;
    
    // DO NOT recalculate resolution - use what user specified (after clamping)
    // This prevents resolution explosion when grid_size and resolution don't match aspect ratio
    // The uniform voxel size is already determined by max_res, other axes will just have fewer voxels
    
    // Safety check for memory - cap at 256^3 for high detail (Increased from 128^3)
    long long total_cells = (long long)settings.resolution_x * settings.resolution_y * settings.resolution_z;
    const long long MAX_CELLS = 256LL * 256 * 256; // ~16.7 million cells max
    if (total_cells > MAX_CELLS) {
        float cap_factor = std::pow((float)MAX_CELLS / total_cells, 1.0f/3.0f);
        settings.resolution_x = std::max(8, (int)(settings.resolution_x * cap_factor));
        settings.resolution_y = std::max(8, (int)(settings.resolution_y * cap_factor));
        settings.resolution_z = std::max(8, (int)(settings.resolution_z * cap_factor));
        settings.voxel_size = settings.grid_size.x / (float)settings.resolution_x;
    }
    
  

    // Create grids with specified resolutions
    grid.resize(settings.resolution_x, settings.resolution_y, settings.resolution_z, settings.voxel_size, settings.grid_offset);
    grid_temp.resize(settings.resolution_x, settings.resolution_y, settings.resolution_z, settings.voxel_size, settings.grid_offset);
    
    // Clear all data (density, fuel, velocity, etc.)
    grid.clear();
    grid_temp.clear();
    
    // Initialize persistent buffers
    persistent_vorticity.assign(grid.getCellCount(), Vec3(0, 0, 0));
    
    // Initialize with ambient temperature
    std::fill(grid.temperature.begin(), grid.temperature.end(), settings.ambient_temperature);
    
    // CRITICAL FIX: Always free old CUDA resources before reallocating
    // This prevents buffer size mismatch when resolution changes
    if (cuda_initialized) {
        freeCUDA();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // BACKEND SELECTION - Respect settings while prioritizing CUDA if possible
    // ═══════════════════════════════════════════════════════════════════════════════
    bool cuda_available = false;
    // Respect central detection flag to avoid calling CUDA symbols when driver missing
    if (g_hasCUDA) {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.major >= 5) {
                cuda_available = true;
            }
        } else {
            cudaGetLastError(); // Clear any error flag
        }
    }
    
    // Choose which backend to actually use
    if (settings.backend == SolverBackend::CUDA) {
        if (cuda_available) {
            initCUDA();
            // If initCUDA failed (e.g. OOM), it would have logged error and left cuda_initialized=false
            if (!cuda_initialized) {
                SCENE_LOG_ERROR("[GasSimulator] CUDA backend requested but initialization failed. Falling back to CPU.");
                settings.backend = SolverBackend::CPU;
            }
        } else {
            SCENE_LOG_WARN("[GasSimulator] CUDA backend requested but no compatible GPU found. Falling back to CPU.");
            settings.backend = SolverBackend::CPU;
        }
    } else {
        // User explicitly chose CPU (useful for debugging/development)
        SCENE_LOG_INFO("[GasSimulator] Using CPU backend as requested.");
    }
    
    current_frame = 0;
    accumulated_time = 0.0f;
    smoothed_adaptive_chunk_count = 1;
    gpu_data_valid = false;
    initialized = true;
}

void GasSimulator::step(float dt, const Matrix4x4& world_matrix) {
    if (!initialized) return;
    
    // CRITICAL: Check for resolution mismatch - do NOT run if buffers don't match settings
    // This prevents buffer overflow when user changes resolution in UI without restarting
    if (settings.resolution_x != grid.nx || 
        settings.resolution_y != grid.ny || 
        settings.resolution_z != grid.nz) {
        // Resolution mismatch - skip step until user reinitializes
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Apply time scale for simulation speed control
    float scaled_dt = dt * settings.time_scale;
    
    // CFL Adaptive Timestep should preserve the requested physical time by subcycling,
    // not by silently slowing the simulation as resolution increases.
    float base_chunk_dt = scaled_dt;
    int adaptive_chunk_count = 1;
    if (settings.adaptive_timestep && scaled_dt > 0.0f) {
        const float cfl_dt = computeCFLTimestep(scaled_dt);
        if (cfl_dt > 0.0f) {
            adaptive_chunk_count = std::max(1, static_cast<int>(std::ceil(scaled_dt / cfl_dt)));
        }
    }

    const bool combustion_active = settings.burn_rate > 0.0f;
    const int max_adaptive_chunks = combustion_active ? 6 : 24;
    adaptive_chunk_count = std::clamp(adaptive_chunk_count, 1, max_adaptive_chunks);
    if (!settings.adaptive_timestep) {
        smoothed_adaptive_chunk_count = 1;
    } else {
        const int previous_smoothed = std::clamp(smoothed_adaptive_chunk_count, 1, max_adaptive_chunks);
        if (adaptive_chunk_count > previous_smoothed) {
            smoothed_adaptive_chunk_count = std::min(adaptive_chunk_count, previous_smoothed + 1);
        } else if (adaptive_chunk_count < previous_smoothed) {
            smoothed_adaptive_chunk_count = std::max(adaptive_chunk_count, previous_smoothed - 1);
        } else {
            smoothed_adaptive_chunk_count = adaptive_chunk_count;
        }
        adaptive_chunk_count = std::clamp(smoothed_adaptive_chunk_count, 1, max_adaptive_chunks);
    }
    base_chunk_dt = scaled_dt / static_cast<float>(adaptive_chunk_count);

    for (int chunk = 0; chunk < adaptive_chunk_count; ++chunk) {
        const float chunk_dt = base_chunk_dt;

        // Update active tiles for sparse processing (VDB-style optimization)
        grid.sparse_mode_enabled = settings.sparse_mode;
        grid.sparse_threshold = settings.sparse_threshold;
        if (settings.sparse_mode) {
            grid.updateActiveTiles(settings.ambient_temperature);
        }

        // Use CUDA if enabled and available
        if (settings.backend == SolverBackend::CUDA && cuda_initialized) {
            stepCUDA(chunk_dt, world_matrix);
        } else {
            // CPU solver (OpenMP disabled to prevent deadlock after resolution change)
            float substep_dt = chunk_dt / std::max(1, settings.substeps);

            for (int sub = 0; sub < settings.substeps; ++sub) {
                // 1. Apply emitters (inject density/temperature/fuel)
                applyEmitters(substep_dt);

                // 2. Process Combustion (Fuel + Heat -> Fire)
                processCombustion(substep_dt);

                // 3. Apply all forces (Internal Buoyancy/Vorticity + External Force Fields)
                applyForces(substep_dt, world_matrix);

                // 4. Advect velocity field
                advectVelocity(substep_dt);

                // 5. Solve pressure and project to divergence-free
                solvePressure();
                project();

                // 6. Advect scalars (density, temperature, fuel)
                advectScalars(substep_dt);

                // 7. Apply dissipation
                applyDissipation(substep_dt);

                // 8. Enforce boundaries
                enforceBoundaries();
            }
        }

    }
    
    auto end = std::chrono::high_resolution_clock::now();
    last_step_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    accumulated_time += scaled_dt;
    current_frame++;
    gpu_data_valid = false;
}

void GasSimulator::reset() {
    grid.clear();
    grid_temp.clear();
    std::fill(grid.temperature.begin(), grid.temperature.end(), settings.ambient_temperature);
    current_frame = 0;
    accumulated_time = 0.0f;
    smoothed_adaptive_chunk_count = 1;
    
    // Clear GPU buffers if CUDA is active
    if (cuda_initialized && settings.backend == SolverBackend::CUDA) {
        clearGPU();
    }
    gpu_data_valid = false;
}

void GasSimulator::shutdown() {
    if (cuda_initialized) {
        freeCUDA();
    }
    initialized = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTERS
// ═══════════════════════════════════════════════════════════════════════════════

int GasSimulator::addEmitter(const Emitter& emitter) {
    Emitter e = emitter;
    if (e.uid == 0) e.uid = emitter_id_counter++; // Assign new UID if not present
    emitters.push_back(e);
    return static_cast<int>(emitters.size()) - 1;
}

void GasSimulator::removeEmitter(int index) {
    if (index >= 0 && index < static_cast<int>(emitters.size())) {
        emitters.erase(emitters.begin() + index);
    }
}

void GasSimulator::applyEmitters(float dt) {
    for (int e_idx = 0; e_idx < (int)emitters.size(); ++e_idx) {
        const auto& emitter = emitters[e_idx];
        if (!isEmitterActiveAtFrame(emitter, static_cast<float>(current_frame))) continue;
        
        // Calculate grid-space bounding box
        Vec3 min_pos, max_pos;
        if (emitter.shape == EmitterShape::Sphere || emitter.shape == EmitterShape::Point) {
            float r = (emitter.shape == EmitterShape::Point) ? grid.voxel_size : emitter.radius;
            min_pos = emitter.position - Vec3(r, r, r);
            max_pos = emitter.position + Vec3(r, r, r);
        } else if (emitter.shape == EmitterShape::Cylinder || emitter.shape == EmitterShape::Cone) {
            float r = std::max(emitter.radius, 0.001f);
            min_pos = emitter.position + Vec3(-r, 0.0f, -r);
            max_pos = emitter.position + Vec3(r, emitter.height, r);
        } else if (emitter.shape == EmitterShape::Disc) {
            float r = std::max(emitter.radius, 0.001f);
            float h = std::max(emitter.size.y, grid.voxel_size);
            min_pos = emitter.position + Vec3(-r, -0.5f * h, -r);
            max_pos = emitter.position + Vec3(r, 0.5f * h, r);
        } else {
            min_pos = emitter.position - emitter.size;
            max_pos = emitter.position + emitter.size;
        }
        
        int i_start = std::max(0, (int)std::floor((min_pos.x - grid.origin.x) / grid.voxel_size));
        int j_start = std::max(0, (int)std::floor((min_pos.y - grid.origin.y) / grid.voxel_size));
        int k_start = std::max(0, (int)std::floor((min_pos.z - grid.origin.z) / grid.voxel_size));
        
        int i_end = std::min(grid.nx - 1, (int)std::ceil((max_pos.x - grid.origin.x) / grid.voxel_size));
        int j_end = std::min(grid.ny - 1, (int)std::ceil((max_pos.y - grid.origin.y) / grid.voxel_size));
        int k_end = std::min(grid.nz - 1, (int)std::ceil((max_pos.z - grid.origin.z) / grid.voxel_size));

        #pragma omp parallel for collapse(2)
        for (int k = k_start; k <= k_end; ++k) {
            for (int j = j_start; j <= j_end; ++j) {
                for (int i = i_start; i <= i_end; ++i) {
                    Vec3 cell_pos = grid.gridToWorld(i, j, k);
                    float normalized_dist = 0.0f;
                    const bool inside = isInsideEmitter(emitter, cell_pos, normalized_dist);
                    
                    if (inside) {
                        size_t idx = grid.cellIndex(i, j, k);
                        const float falloff_strength = computeEmitterFalloff(emitter, normalized_dist);
                        float density_strength = falloff_strength;
                        float temperature_strength = falloff_strength;
                        float velocity_strength = falloff_strength;
                        if (emitter.noise_enabled) {
                            const float noise = sampleEmitterNoise(cell_pos, emitter.noise_frequency, emitter.noise_seed, accumulated_time, emitter.noise_speed);
                            const float noise_mult = std::max(0.0f, 1.0f + noise * emitter.noise_amplitude);
                            if (emitter.noise_modulate_density) density_strength *= noise_mult;
                            if (emitter.noise_modulate_temperature) temperature_strength *= noise_mult;
                            if (emitter.noise_modulate_velocity) velocity_strength *= noise_mult;
                            if (!emitter.noise_modulate_density && !emitter.noise_modulate_temperature && !emitter.noise_modulate_velocity) {
                                density_strength *= noise_mult;
                                temperature_strength *= noise_mult;
                                velocity_strength *= noise_mult;
                            }
                        }
                        const float strength = std::max({ density_strength, temperature_strength, velocity_strength });
                        if (strength <= 0.0001f) continue;
                        
                        // Inject with limits to prevent overflow
                        grid.density[idx] += emitter.density_rate * dt * density_strength;
                        grid.density[idx] = std::min(grid.density[idx], settings.max_density);
                        
                        const float flame_activity = std::clamp(grid.interaction[idx] * 0.15f, 0.0f, 1.0f);
                        const float hotness = std::clamp((grid.temperature[idx] - settings.ambient_temperature) /
                            std::max(settings.ignition_temperature - settings.ambient_temperature, 1.0f), 0.0f, 3.0f);
                        const float oxygen_room = std::clamp(
                            1.0f - (grid.density[idx] / std::max(settings.max_density * 0.7f, 0.001f)) -
                            (grid.fuel[idx] / 14.0f) -
                            flame_activity * 0.35f,
                            0.0f, 1.0f);
                        const float fuel_room = std::clamp(1.0f - (grid.fuel[idx] / 12.0f), 0.0f, 1.0f);
                        const float phase_strength = computeEmitterFuelPhaseStrength(
                            emitter,
                            grid.temperature[idx],
                            settings.ambient_temperature,
                            flame_activity
                        );
                        const float moderated_fuel_strength =
                            density_strength * fuel_room * oxygen_room * phase_strength * (1.0f - 0.55f * flame_activity);
                        grid.fuel[idx] += emitter.fuel_rate * dt * moderated_fuel_strength;
                        // Fuel limit: prevent runaway accumulation pockets
                        grid.fuel[idx] = std::min(grid.fuel[idx], 12.0f);
                        
                        // Temperature: inject relative to ambient so emitter edges still stay physically warm.
                        const float heat_phase_strength = computeEmitterHeatPhaseStrength(
                            emitter,
                            grid.temperature[idx],
                            settings.ambient_temperature,
                            flame_activity
                        );
                        const float hot_headroom = std::clamp(1.0f - hotness * 0.28f, 0.15f, 1.0f);
                        const float usable_strength = std::clamp(std::sqrt(std::max(temperature_strength, 0.0f)), 0.0f, 1.0f) *
                            heat_phase_strength * oxygen_room * hot_headroom;
                        const float temp_delta = std::max(0.0f, emitter.temperature - settings.ambient_temperature);
                        float target_temp = settings.ambient_temperature + temp_delta * usable_strength;
                        target_temp = std::min(target_temp, settings.max_temperature * 0.92f);
                        if (grid.temperature[idx] < target_temp) {
                            const float dt_scale = std::clamp(dt / (1.0f / 60.0f), 0.05f, 1.0f);
                            const float heat_blend = std::clamp((0.10f + 0.38f * usable_strength) * dt_scale, 0.02f, 0.48f);
                            grid.temperature[idx] = grid.temperature[idx] * (1.0f - heat_blend) + target_temp * heat_blend;
                        }
                        
                        if (emitter.velocity.length() > 0.001f) {
                            if (i > 0) grid.vel_x[grid.velXIndex(i, j, k)] += emitter.velocity.x * dt * velocity_strength;
                            if (j > 0) grid.vel_y[grid.velYIndex(i, j, k)] += emitter.velocity.y * dt * velocity_strength;
                            if (k > 0) grid.vel_z[grid.velZIndex(i, j, k)] += emitter.velocity.z * dt * velocity_strength;
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORCES
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::applyForces(float dt, const Matrix4x4& world_matrix) {
    applyGravity(dt);
    applyBuoyancy(dt);
    applyVorticity(dt);
    applyCurlNoiseTurbulence(dt);  // Industry-standard curl noise
    applyWind(dt);
    applyExternalForceFields(dt, world_matrix);
    applyVelocityClamping(); // Stability limit from UI
}

void GasSimulator::applyGravity(float dt) {
    if (settings.gravity.length() < 0.001f) return;
    
    Vec3 grav_accel = settings.gravity * dt;
    
    // Apply gravity uniformly to all velocity components
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_x.size(); ++i) {
        grid.vel_x[i] += grav_accel.x;
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_y.size(); ++i) {
        grid.vel_y[i] += grav_accel.y;
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_z.size(); ++i) {
        grid.vel_z[i] += grav_accel.z;
    }
}

void GasSimulator::applyBuoyancy(float dt) {
    // Buoyancy: hot smoke rises, dense smoke sinks
    // FIX: Iterate over Y-faces (velocity components) directly to avoid data races
    // when multiple cells update the same face.
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 1; j < grid.ny; ++j) { // Faces between cells
            for (int i = 0; i < grid.nx; ++i) {
                // Average density and temperature from the two adjacent cells
                float d_avg = 0.5f * (grid.densityAt(i, j - 1, k) + grid.densityAt(i, j, k));
                float t_avg = 0.5f * (grid.temperatureAt(i, j - 1, k) + grid.temperatureAt(i, j, k));
                
                float temp_diff = t_avg - settings.ambient_temperature;
                float buoyancy_force = settings.buoyancy_temperature * temp_diff + 
                                      settings.buoyancy_density * d_avg;
                
                grid.vel_y[grid.velYIndex(i, j, k)] += buoyancy_force * dt;
            }
        }
    }
}

void GasSimulator::applyVorticity(float dt) {
    if (settings.vorticity_strength < 0.001f) return;
    
    // Compute vorticity (curl of velocity)
    if (persistent_vorticity.size() != grid.getCellCount()) {
        persistent_vorticity.assign(grid.getCellCount(), Vec3(0, 0, 0));
    }
    
    #pragma omp parallel for collapse(2)
    for (int k = 1; k < grid.nz - 1; ++k) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                // Finite difference approximation of curl
                float dwy_dz = (grid.velYAt(i, j, k + 1) - grid.velYAt(i, j, k - 1)) / (2.0f * grid.voxel_size);
                float dwz_dy = (grid.velZAt(i, j + 1, k) - grid.velZAt(i, j - 1, k)) / (2.0f * grid.voxel_size);
                float dwx_dz = (grid.velXAt(i, j, k + 1) - grid.velXAt(i, j, k - 1)) / (2.0f * grid.voxel_size);
                float dwz_dx = (grid.velZAt(i + 1, j, k) - grid.velZAt(i - 1, j, k)) / (2.0f * grid.voxel_size);
                float dwx_dy = (grid.velXAt(i, j + 1, k) - grid.velXAt(i, j - 1, k)) / (2.0f * grid.voxel_size);
                float dwy_dx = (grid.velYAt(i + 1, j, k) - grid.velYAt(i - 1, j, k)) / (2.0f * grid.voxel_size);
                
                size_t idx = grid.cellIndex(i, j, k);
                persistent_vorticity[idx] = Vec3(dwy_dz - dwz_dy, dwx_dz - dwz_dx, dwx_dy - dwy_dx);
            }
        }
    }
    
    // Compute gradient of vorticity magnitude and apply confinement force
    #pragma omp parallel for collapse(2)
    for (int k = 2; k < grid.nz - 2; ++k) {
        for (int j = 2; j < grid.ny - 2; ++j) {
            for (int i = 2; i < grid.nx - 2; ++i) {
                size_t idx = grid.cellIndex(i, j, k);
                
                float vor_mag_px = persistent_vorticity[grid.cellIndex(i + 1, j, k)].length();
                float vor_mag_mx = persistent_vorticity[grid.cellIndex(i - 1, j, k)].length();
                float vor_mag_py = persistent_vorticity[grid.cellIndex(i, j + 1, k)].length();
                float vor_mag_my = persistent_vorticity[grid.cellIndex(i, j - 1, k)].length();
                float vor_mag_pz = persistent_vorticity[grid.cellIndex(i, j, k + 1)].length();
                float vor_mag_mz = persistent_vorticity[grid.cellIndex(i, j, k - 1)].length();
                
                // Gradient of vorticity magnitude
                Vec3 grad_vor(
                    (vor_mag_px - vor_mag_mx) / (2.0f * grid.voxel_size),
                    (vor_mag_py - vor_mag_my) / (2.0f * grid.voxel_size),
                    (vor_mag_pz - vor_mag_mz) / (2.0f * grid.voxel_size)
                );
                
                float grad_len = grad_vor.length();
                if (grad_len > 1e-6f) {
                    grad_vor = grad_vor / grad_len;
                    
                    // Confinement force = cross(grad, vorticity)
                    Vec3 vor = persistent_vorticity[idx];
                    Vec3 force = Vec3(
                        grad_vor.y * vor.z - grad_vor.z * vor.y,
                        grad_vor.z * vor.x - grad_vor.x * vor.z,
                        grad_vor.x * vor.y - grad_vor.y * vor.x
                    ) * settings.vorticity_strength * grid.voxel_size;
                    
                    // Apply force to velocity
                    grid.vel_x[grid.velXIndex(i, j, k)] += force.x * dt;
                    grid.vel_y[grid.velYIndex(i, j, k)] += force.y * dt;
                    grid.vel_z[grid.velZIndex(i, j, k)] += force.z * dt;
                }
            }
        }
    }
}

void GasSimulator::applyWind(float dt) {
    if (settings.wind.length() < 0.001f) return;
    
    Vec3 wind_accel = settings.wind * dt;
    
    for (size_t i = 0; i < grid.vel_x.size(); ++i) {
        grid.vel_x[i] += wind_accel.x;
    }
    for (size_t i = 0; i < grid.vel_y.size(); ++i) {
        grid.vel_y[i] += wind_accel.y;
    }
    for (size_t i = 0; i < grid.vel_z.size(); ++i) {
        grid.vel_z[i] += wind_accel.z;
    }
}

void GasSimulator::applyVelocityClamping() {
    // Apply stability limits from UI settings
    const float max_vel = settings.max_velocity;
    
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_x.size(); ++i) {
        grid.vel_x[i] = std::max(-max_vel, std::min(max_vel, grid.vel_x[i]));
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_y.size(); ++i) {
        grid.vel_y[i] = std::max(-max_vel, std::min(max_vel, grid.vel_y[i]));
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)grid.vel_z.size(); ++i) {
        grid.vel_z[i] = std::max(-max_vel, std::min(max_vel, grid.vel_z[i]));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CURL NOISE TURBULENCE (Industry-Standard Divergence-Free Turbulence)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::applyCurlNoiseTurbulence(float dt) {
    if (settings.turbulence_strength < 0.001f) return;
    
    const float time = accumulated_time;
    const float freq = settings.turbulence_scale;
    const float strength = settings.turbulence_strength;
    
    // Curl noise parameters from UI settings
    const int octaves = settings.turbulence_octaves;
    const float lacunarity = settings.turbulence_lacunarity;
    const float persistence = settings.turbulence_persistence;
    const float animation_speed = 0.5f;       // How fast turbulence evolves
    
    // Apply curl noise to velocity field
    // This adds divergence-free swirling motion that looks natural
    #pragma omp parallel for collapse(2)
    for (int k = 1; k < grid.nz - 1; ++k) {
        for (int j = 1; j < grid.ny - 1; ++j) {
            for (int i = 1; i < grid.nx - 1; ++i) {
                // World-space position for consistent noise sampling
                Vec3 world_pos = grid.origin + Vec3(
                    (i + 0.5f) * grid.voxel_size,
                    (j + 0.5f) * grid.voxel_size,
                    (k + 0.5f) * grid.voxel_size
                );
                
                const float density = grid.densityAt(i, j, k);
                const float temperature = grid.temperatureAt(i, j, k);
                const float heat_norm = std::clamp((temperature - settings.ambient_temperature) /
                    std::max(settings.ignition_temperature - settings.ambient_temperature, 1.0f), 0.0f, 3.0f);
                const float flame_norm = std::clamp(grid.interaction[grid.cellIndex(i, j, k)] * 0.05f, 0.0f, 2.0f);

                const float density_grad_x = (grid.densityAt(i + 1, j, k) - grid.densityAt(i - 1, j, k)) / (2.0f * grid.voxel_size);
                const float density_grad_y = (grid.densityAt(i, j + 1, k) - grid.densityAt(i, j - 1, k)) / (2.0f * grid.voxel_size);
                const float density_grad_z = (grid.densityAt(i, j, k + 1) - grid.densityAt(i, j, k - 1)) / (2.0f * grid.voxel_size);
                const float edge_norm = std::clamp(std::sqrt(density_grad_x * density_grad_x + density_grad_y * density_grad_y + density_grad_z * density_grad_z) * 0.08f, 0.0f, 2.0f);
                const float activity = std::max({ density, heat_norm * 0.35f, flame_norm * 0.5f, edge_norm * 0.25f });
                if (activity < 0.01f) continue;
                
                // Push more breakup into hot cores and density edges instead of only dense bulk smoke.
                float local_strength = strength * std::clamp(
                    0.18f + 0.45f * std::sqrt(std::max(density, 0.0f)) +
                    0.40f * std::min(heat_norm, 1.5f) +
                    0.30f * std::min(flame_norm, 1.5f) +
                    0.35f * std::min(edge_norm, 1.0f),
                    0.0f, 2.5f);
                
                // Get animated curl noise (divergence-free!)
                Vec3 curl = Physics::Noise::curlFBM_animated(
                    world_pos, time,
                    octaves, freq, lacunarity, persistence,
                    animation_speed, 42  // seed
                );
                
                // Apply to velocity
                size_t idx_x = grid.velXIndex(i, j, k);
                size_t idx_y = grid.velYIndex(i, j, k);
                size_t idx_z = grid.velZIndex(i, j, k);
                
                grid.vel_x[idx_x] += curl.x * local_strength * dt;
                grid.vel_y[idx_y] += curl.y * local_strength * dt;
                grid.vel_z[idx_z] += curl.z * local_strength * dt;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CFL ADAPTIVE TIMESTEP (Industry Standard)
// ═══════════════════════════════════════════════════════════════════════════════

float GasSimulator::computeMaxVelocity() {
    // Scan grid for maximum velocity magnitude
    // NOTE: OpenMP removed due to potential deadlock on Windows after resolution change
    float max_vel_sq = 0.0f;
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                // Sample velocity at cell center
                float vx = (grid.vel_x[grid.velXIndex(i, j, k)] + grid.vel_x[grid.velXIndex(i + 1, j, k)]) * 0.5f;
                float vy = (grid.vel_y[grid.velYIndex(i, j, k)] + grid.vel_y[grid.velYIndex(i, j + 1, k)]) * 0.5f;
                float vz = (grid.vel_z[grid.velZIndex(i, j, k)] + grid.vel_z[grid.velZIndex(i, j, k + 1)]) * 0.5f;
                
                float vel_sq = vx * vx + vy * vy + vz * vz;
                if (vel_sq > max_vel_sq) max_vel_sq = vel_sq;
            }
        }
    }
    
    return std::sqrt(max_vel_sq);
}

float GasSimulator::computeCFLTimestep(float requested_dt) {
    // CFL Condition: dt <= CFL_number * voxel_size / max_velocity
    // This ensures fluid doesn't travel more than CFL_number cells per timestep
    
    float max_velocity = computeMaxVelocity();
    
    // Avoid division by zero - if no velocity, use requested dt
    if (max_velocity < 1e-6f) {
        return std::min(requested_dt, settings.max_timestep);
    }
    
    // Calculate CFL-limited timestep
    float cfl_dt = settings.cfl_number * settings.voxel_size / max_velocity;
    
    // Clamp to min/max bounds
    cfl_dt = std::clamp(cfl_dt, settings.min_timestep, settings.max_timestep);
    
    // Return the smaller of requested and CFL-limited
    float effective_dt = std::min(requested_dt, cfl_dt);
    
    // Log if we're limiting significantly (for debugging)
    if (cfl_dt < requested_dt * 0.5f) {
        // High velocity detected, timestep reduced significantly
        // Could add logging here if needed
    }
    
    return effective_dt;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXTERNAL FORCE FIELDS (Scene-level)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::applyExternalForceFields(float dt, const Matrix4x4& world_matrix) {
    if (!external_force_field_manager) return;
    size_t active_count = external_force_field_manager->getActiveCount();
    if (active_count == 0) return; // Early exit if no fields
    
    float time = accumulated_time;
    
    // For force vectors, we only need to apply inverse ROTATION, not scale!
    // Scale should not affect force magnitude - only direction matters.
    // Extract rotation from world_matrix and compute its inverse.
    // 
    // NOTE: Force is in world space (m/s^2), velocity in grid is also in m/s.
    // Since the grid is axis-aligned in local space but may be rotated in world,
    // we need to rotate force from world to local orientation.
    // But for now, gas volumes are typically not rotated, so we can apply force directly.
    // If rotation is needed, we would extract R from world_matrix and use R^-1.
    
    // SIMPLIFIED: Apply world force directly to grid velocity.
    // This works because grid velocity is in m/s (world units), not grid units.
    // The force field returns m/s^2, so force * dt gives m/s delta.
    
    // Apply force fields to each staggered face component
    // NOTE: OpenMP removed to prevent deadlock after resolution change on Windows
    
    // X-velocity components (nx+1, ny, nz)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                // NORMALIZED POSITION [0, 1]: Matrix handles scaling/translation
                Vec3 face_pos_norm(i / (float)grid.nx, (j + 0.5f) / (float)grid.ny, (k + 0.5f) / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                // Sample current velocity (already in m/s)
                Vec3 local_pos_physical = grid.origin + Vec3(i * grid.voxel_size, (j + 0.5f) * grid.voxel_size, (k + 0.5f) * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                
                // Evaluate force field at world position
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, local_vel, true, false, false, false);
                
                // Apply force directly - no scale transformation needed for force vectors
                grid.vel_x[grid.velXIndex(i, j, k)] += world_force.x * dt;
            }
        }
    }

    // Y-velocity components (nx, ny+1, nz)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 face_pos_norm((i + 0.5f) / (float)grid.nx, j / (float)grid.ny, (k + 0.5f) / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                Vec3 local_pos_physical = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, j * grid.voxel_size, (k + 0.5f) * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, local_vel, true, false, false, false);
                
                grid.vel_y[grid.velYIndex(i, j, k)] += world_force.y * dt;
            }
        }
    }

    // Z-velocity components (nx, ny, nz+1)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 face_pos_norm((i + 0.5f) / (float)grid.nx, (j + 0.5f) / (float)grid.ny, k / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                Vec3 local_pos_physical = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, (j + 0.5f) * grid.voxel_size, k * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, local_vel, true, false, false, false);
                
                grid.vel_z[grid.velZIndex(i, j, k)] += world_force.z * dt;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADVECTION (Semi-Lagrangian)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::advectVelocity(float dt) {
    const bool use_maccormack = settings.advection_mode == GasSimulationSettings::AdvectionMode::MacCormack;
    const bool use_bfecc = settings.advection_mode == GasSimulationSettings::AdvectionMode::BFECC;

    // Copy current velocity to temp
    grid_temp.vel_x = grid.vel_x;
    grid_temp.vel_y = grid.vel_y;
    grid_temp.vel_z = grid.vel_z;

    std::vector<float> vel_x_forward;
    std::vector<float> vel_y_forward;
    std::vector<float> vel_z_forward;
    std::vector<float> vel_x_backward;
    std::vector<float> vel_y_backward;
    std::vector<float> vel_z_backward;

    if (use_maccormack || use_bfecc) {
        vel_x_forward.resize(grid.vel_x.size(), 0.0f);
        vel_y_forward.resize(grid.vel_y.size(), 0.0f);
        vel_z_forward.resize(grid.vel_z.size(), 0.0f);
        vel_x_backward.resize(grid.vel_x.size(), 0.0f);
        vel_y_backward.resize(grid.vel_y.size(), 0.0f);
        vel_z_backward.resize(grid.vel_z.size(), 0.0f);
    }
    
    // Advect X-velocity
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                // Position of X-face
                Vec3 pos = grid.origin + Vec3(i * grid.voxel_size, 
                                              (j + 0.5f) * grid.voxel_size, 
                                              (k + 0.5f) * grid.voxel_size);
                
                // Trace backwards
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                
                const size_t idx = grid.velXIndex(i, j, k);
                const float forward_value = grid_temp.sampleVelocity(prev_pos).x;
                if (use_maccormack || use_bfecc) {
                    vel_x_forward[idx] = forward_value;
                } else {
                    grid.vel_x[idx] = forward_value;
                }
            }
        }
    }
    
    // Advect Y-velocity
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, 
                                              j * grid.voxel_size, 
                                              (k + 0.5f) * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                const size_t idx = grid.velYIndex(i, j, k);
                const float forward_value = grid_temp.sampleVelocity(prev_pos).y;
                if (use_maccormack || use_bfecc) {
                    vel_y_forward[idx] = forward_value;
                } else {
                    grid.vel_y[idx] = forward_value;
                }
            }
        }
    }
    
    // Advect Z-velocity
    #pragma omp parallel for collapse(2)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, 
                                              (j + 0.5f) * grid.voxel_size, 
                                              k * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                const size_t idx = grid.velZIndex(i, j, k);
                const float forward_value = grid_temp.sampleVelocity(prev_pos).z;
                if (use_maccormack || use_bfecc) {
                    vel_z_forward[idx] = forward_value;
                } else {
                    grid.vel_z[idx] = forward_value;
                }
            }
        }
    }

    if (!use_maccormack && !use_bfecc) {
        return;
    }

    grid_temp.vel_x = vel_x_forward;
    grid_temp.vel_y = vel_y_forward;
    grid_temp.vel_z = vel_z_forward;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3(i * grid.voxel_size,
                                              (j + 0.5f) * grid.voxel_size,
                                              (k + 0.5f) * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 forward_pos = pos + vel * dt;
                vel_x_backward[grid.velXIndex(i, j, k)] = grid_temp.sampleVelocity(forward_pos).x;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size,
                                              j * grid.voxel_size,
                                              (k + 0.5f) * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 forward_pos = pos + vel * dt;
                vel_y_backward[grid.velYIndex(i, j, k)] = grid_temp.sampleVelocity(forward_pos).y;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size,
                                              (j + 0.5f) * grid.voxel_size,
                                              k * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 forward_pos = pos + vel * dt;
                vel_z_backward[grid.velZIndex(i, j, k)] = grid_temp.sampleVelocity(forward_pos).z;
            }
        }
    }

    grid_temp.vel_x = grid.vel_x;
    grid_temp.vel_y = grid.vel_y;
    grid_temp.vel_z = grid.vel_z;

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(grid.vel_x.size()); ++idx) {
        float corrected = use_bfecc
            ? grid_temp.vel_x[idx] + 0.5f * (grid_temp.vel_x[idx] - vel_x_backward[idx])
            : vel_x_forward[idx] + 0.5f * (grid_temp.vel_x[idx] - vel_x_backward[idx]);
        if (!std::isfinite(corrected)) corrected = grid_temp.vel_x[idx];
        vel_x_forward[idx] = corrected;
    }
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(grid.vel_y.size()); ++idx) {
        float corrected = use_bfecc
            ? grid_temp.vel_y[idx] + 0.5f * (grid_temp.vel_y[idx] - vel_y_backward[idx])
            : vel_y_forward[idx] + 0.5f * (grid_temp.vel_y[idx] - vel_y_backward[idx]);
        if (!std::isfinite(corrected)) corrected = grid_temp.vel_y[idx];
        vel_y_forward[idx] = corrected;
    }
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(grid.vel_z.size()); ++idx) {
        float corrected = use_bfecc
            ? grid_temp.vel_z[idx] + 0.5f * (grid_temp.vel_z[idx] - vel_z_backward[idx])
            : vel_z_forward[idx] + 0.5f * (grid_temp.vel_z[idx] - vel_z_backward[idx]);
        if (!std::isfinite(corrected)) corrected = grid_temp.vel_z[idx];
        vel_z_forward[idx] = corrected;
    }

    if (use_maccormack) {
        grid.vel_x = vel_x_forward;
        grid.vel_y = vel_y_forward;
        grid.vel_z = vel_z_forward;
        return;
    }

    grid_temp.vel_x = vel_x_forward;
    grid_temp.vel_y = vel_y_forward;
    grid_temp.vel_z = vel_z_forward;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3(i * grid.voxel_size,
                                              (j + 0.5f) * grid.voxel_size,
                                              (k + 0.5f) * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                grid.vel_x[grid.velXIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).x;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size,
                                              j * grid.voxel_size,
                                              (k + 0.5f) * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                grid.vel_y[grid.velYIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).y;
            }
        }
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.origin + Vec3((i + 0.5f) * grid.voxel_size,
                                              (j + 0.5f) * grid.voxel_size,
                                              k * grid.voxel_size);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                grid.vel_z[grid.velZIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).z;
            }
        }
    }
}

void GasSimulator::advectScalars(float dt) {
    const bool use_maccormack = settings.advection_mode == GasSimulationSettings::AdvectionMode::MacCormack;
    const bool use_bfecc = settings.advection_mode == GasSimulationSettings::AdvectionMode::BFECC;
    const std::vector<float> preserved_interaction = grid.interaction;

    // Copy to temp
    grid_temp.density = grid.density;
    grid_temp.temperature = grid.temperature;
    grid_temp.fuel = grid.fuel;
    grid_temp.interaction = grid.interaction;

    std::vector<float> density_forward;
    std::vector<float> temperature_forward;
    std::vector<float> fuel_forward;
    std::vector<float> interaction_forward;
    std::vector<float> density_backward;
    std::vector<float> temperature_backward;
    std::vector<float> fuel_backward;
    std::vector<float> interaction_backward;

    if (use_maccormack || use_bfecc) {
        density_forward.resize(grid.density.size(), 0.0f);
        temperature_forward.resize(grid.temperature.size(), 0.0f);
        fuel_forward.resize(grid.fuel.size(), 0.0f);
        interaction_forward.resize(grid.interaction.size(), 0.0f);
        density_backward.resize(grid.density.size(), 0.0f);
        temperature_backward.resize(grid.temperature.size(), 0.0f);
        fuel_backward.resize(grid.fuel.size(), 0.0f);
        interaction_backward.resize(grid.interaction.size(), 0.0f);
    }
    
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.gridToWorld(i, j, k);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;
                
                size_t idx = grid.cellIndex(i, j, k);
                const float density_value = grid_temp.sampleDensity(prev_pos);
                const float temperature_value = grid_temp.sampleTemperature(prev_pos);
                const float fuel_value = grid_temp.sampleCellCentered(grid_temp.fuel, prev_pos);
                const float interaction_value = grid_temp.sampleCellCentered(grid_temp.interaction, prev_pos);

                if (use_maccormack || use_bfecc) {
                    density_forward[idx] = density_value;
                    temperature_forward[idx] = temperature_value;
                    fuel_forward[idx] = fuel_value;
                    interaction_forward[idx] = interaction_value;
                } else {
                    grid.density[idx] = density_value;
                    grid.temperature[idx] = temperature_value;
                    grid.fuel[idx] = fuel_value;
                    grid.interaction[idx] = interaction_value;
                }
            }
        }
    }

    if (!use_maccormack && !use_bfecc) {
        // Flame interaction is a local reaction memory, not a transported scalar.
        grid.interaction = preserved_interaction;
        return;
    }

    grid_temp.density = density_forward;
    grid_temp.temperature = temperature_forward;
    grid_temp.fuel = fuel_forward;
    grid_temp.interaction = interaction_forward;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.gridToWorld(i, j, k);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 forward_pos = pos + vel * dt;
                size_t idx = grid.cellIndex(i, j, k);
                density_backward[idx] = grid_temp.sampleDensity(forward_pos);
                temperature_backward[idx] = grid_temp.sampleTemperature(forward_pos);
                fuel_backward[idx] = grid_temp.sampleCellCentered(grid_temp.fuel, forward_pos);
                interaction_backward[idx] = grid_temp.sampleCellCentered(grid_temp.interaction, forward_pos);
            }
        }
    }

    grid_temp.density = grid.density;
    grid_temp.temperature = grid.temperature;
    grid_temp.fuel = grid.fuel;
    grid_temp.interaction = grid.interaction;

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(grid.density.size()); ++idx) {
        density_forward[idx] = std::max(0.0f, (use_bfecc ? grid_temp.density[idx] : density_forward[idx]) + 0.5f * (grid_temp.density[idx] - density_backward[idx]));
        temperature_forward[idx] = std::max(0.0f, (use_bfecc ? grid_temp.temperature[idx] : temperature_forward[idx]) + 0.5f * (grid_temp.temperature[idx] - temperature_backward[idx]));
        fuel_forward[idx] = std::max(0.0f, (use_bfecc ? grid_temp.fuel[idx] : fuel_forward[idx]) + 0.5f * (grid_temp.fuel[idx] - fuel_backward[idx]));
        interaction_forward[idx] = std::max(0.0f, (use_bfecc ? grid_temp.interaction[idx] : interaction_forward[idx]) + 0.5f * (grid_temp.interaction[idx] - interaction_backward[idx]));
    }

    if (use_maccormack) {
        grid.density = density_forward;
        grid.temperature = temperature_forward;
        grid.fuel = fuel_forward;
        grid.interaction = preserved_interaction;
        return;
    }

    grid_temp.density = density_forward;
    grid_temp.temperature = temperature_forward;
    grid_temp.fuel = fuel_forward;
    grid_temp.interaction = interaction_forward;

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 pos = grid.gridToWorld(i, j, k);
                Vec3 vel = grid.sampleVelocity(pos);
                Vec3 prev_pos = pos - vel * dt;

                size_t idx = grid.cellIndex(i, j, k);
                grid.density[idx] = grid_temp.sampleDensity(prev_pos);
                grid.temperature[idx] = grid_temp.sampleTemperature(prev_pos);
                grid.fuel[idx] = grid_temp.sampleCellCentered(grid_temp.fuel, prev_pos);
            }
        }
    }

    grid.interaction = preserved_interaction;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRESSURE SOLVER (with SOR support for faster convergence)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::solvePressure() {
    float scale = 1.0f / grid.voxel_size;
    
    // Pressure buffer is also used as a transient combustion expansion source.
    // Consume it here so pressure does not accumulate across frames and blow the sim apart.
    std::vector<float> expansion_source = grid.pressure;
    std::fill(grid.pressure.begin(), grid.pressure.end(), 0.0f);

    // Compute divergence of the velocity field
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float div = (grid.velXAt(i + 1, j, k) - grid.velXAt(i, j, k) +
                             grid.velYAt(i, j + 1, k) - grid.velYAt(i, j, k) +
                             grid.velZAt(i, j, k + 1) - grid.velZAt(i, j, k)) * scale;
                grid.divergence[grid.cellIndex(i, j, k)] = div + expansion_source[grid.cellIndex(i, j, k)];
            }
        }
    }
    
    float h2 = grid.voxel_size * grid.voxel_size;
    
    // SOR relaxation factor (omega)
    // omega = 1.0 -> Gauss-Seidel, omega > 1.0 -> SOR (faster convergence)
    // Optimal for 3D Poisson: omega ≈ 2 / (1 + sin(π/n)) ≈ 1.7-1.9
    float omega = (settings.pressure_solver == GasSimulationSettings::PressureSolverMode::SOR) 
                  ? settings.sor_omega : 1.0f;
    
    // Red-Black Gauss-Seidel / SOR for thread-safe parallel convergence
    for (int iter = 0; iter < settings.pressure_iterations; ++iter) {
        // Red cells update
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    if ((i + j + k) % 2 == 0) {
                        // OPEN BOUNDARY CONDITION: Surroundings outside return 0 pressure
                        float p_sum = (i + 1 < grid.nx ? grid.pressureAt(i + 1, j, k) : 0.0f) +
                                      (i - 1 >= 0 ? grid.pressureAt(i - 1, j, k) : 0.0f) +
                                      (j + 1 < grid.ny ? grid.pressureAt(i, j + 1, k) : 0.0f) +
                                      (j - 1 >= 0 ? grid.pressureAt(i, j - 1, k) : 0.0f) +
                                      (k + 1 < grid.nz ? grid.pressureAt(i, j, k + 1) : 0.0f) +
                                      (k - 1 >= 0 ? grid.pressureAt(i, j, k - 1) : 0.0f);
                        
                        float div = grid.divergence[grid.cellIndex(i, j, k)];
                        float p_new = (p_sum - div * h2) / 6.0f;
                        
                        // SOR: blend old and new values
                        float p_old = grid.pressure[grid.cellIndex(i, j, k)];
                        grid.pressure[grid.cellIndex(i, j, k)] = p_old + omega * (p_new - p_old);
                    }
                }
            }
        }
        
        // Black cells update
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    if ((i + j + k) % 2 == 1) {
                        float p_sum = (i + 1 < grid.nx ? grid.pressureAt(i + 1, j, k) : 0.0f) +
                                      (i - 1 >= 0 ? grid.pressureAt(i - 1, j, k) : 0.0f) +
                                      (j + 1 < grid.ny ? grid.pressureAt(i, j + 1, k) : 0.0f) +
                                      (j - 1 >= 0 ? grid.pressureAt(i, j - 1, k) : 0.0f) +
                                      (k + 1 < grid.nz ? grid.pressureAt(i, j, k + 1) : 0.0f) +
                                      (k - 1 >= 0 ? grid.pressureAt(i, j, k - 1) : 0.0f);
                        
                        float div = grid.divergence[grid.cellIndex(i, j, k)];
                        float p_new = (p_sum - div * h2) / 6.0f;
                        
                        // SOR: blend old and new values
                        float p_old = grid.pressure[grid.cellIndex(i, j, k)];
                        grid.pressure[grid.cellIndex(i, j, k)] = p_old + omega * (p_new - p_old);
                    }
                }
            }
        }
    }
}

void GasSimulator::project() {
    // Subtract pressure gradient from velocity
    float scale = 1.0f / grid.voxel_size;
    
    // OPEN BOUNDARIES (Outflow): Include boundaries in the update.
    // We assume pressure outside the domain is 0.0f.
    
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                float p_right = (i < grid.nx) ? grid.pressureAt(i, j, k) : 0.0f;
                float p_left  = (i > 0) ? grid.pressureAt(i - 1, j, k) : 0.0f;
                grid.vel_x[grid.velXIndex(i, j, k)] -= (p_right - p_left) * scale;
            }
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float p_top  = (j < grid.ny) ? grid.pressureAt(i, j, k) : 0.0f;
                float p_bot  = (j > 0) ? grid.pressureAt(i, j - 1, k) : 0.0f;
                grid.vel_y[grid.velYIndex(i, j, k)] -= (p_top - p_bot) * scale;
            }
        }
    }
    
    #pragma omp parallel for collapse(2)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float p_front = (k < grid.nz) ? grid.pressureAt(i, j, k) : 0.0f;
                float p_back  = (k > 0) ? grid.pressureAt(i, j, k - 1) : 0.0f;
                grid.vel_z[grid.velZIndex(i, j, k)] -= (p_front - p_back) * scale;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISSIPATION & BOUNDARIES
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::applyDissipation(float dt) {
    float density_factor = std::pow(settings.density_dissipation, dt);
    float temp_factor = std::pow(settings.temperature_dissipation, dt);
    float vel_factor = std::pow(settings.velocity_dissipation, dt);
    float fuel_factor = std::pow(settings.fuel_dissipation, dt); 
    
    // NOTE: OpenMP removed due to deadlock issues on Windows after resolution change
    for (int i = 0; i < (int)grid.density.size(); ++i) {
        grid.density[i] *= density_factor;
        if (grid.density[i] < 0.001f) grid.density[i] = 0.0f;
    }
    
    for (int i = 0; i < (int)grid.temperature.size(); ++i) {
        const float hotness = std::clamp((grid.temperature[i] - settings.ambient_temperature) /
            std::max(settings.ignition_temperature - settings.ambient_temperature, 1.0f), 0.0f, 3.0f);
        const float active_flame = std::clamp(grid.interaction[i] * 0.04f, 0.0f, 1.0f);
        const float hot_temp_cooling = 1.0f / (1.0f + std::max(0.0f, hotness - active_flame) * 1.8f * dt);
        grid.temperature[i] = settings.ambient_temperature +
            (grid.temperature[i] - settings.ambient_temperature) * temp_factor * hot_temp_cooling;
    }
    
    for (int i = 0; i < (int)grid.fuel.size(); ++i) {
        const float hotness = std::clamp((grid.temperature[i] - settings.ambient_temperature) /
            std::max(settings.ignition_temperature - settings.ambient_temperature, 1.0f), 0.0f, 3.0f);
        const float hot_fuel_decay = 1.0f / (1.0f + 1.4f * hotness * dt);
        grid.fuel[i] *= fuel_factor * hot_fuel_decay;
        if (grid.fuel[i] < 0.001f) grid.fuel[i] = 0.0f;
    }

    for (int i = 0; i < (int)grid.interaction.size(); ++i) {
        grid.interaction[i] *= 0.5f;
        if (grid.interaction[i] < 0.001f) grid.interaction[i] = 0.0f;
    }
    
    for (int i = 0; i < (int)grid.vel_x.size(); ++i) grid.vel_x[i] *= vel_factor;
    for (int i = 0; i < (int)grid.vel_y.size(); ++i) grid.vel_y[i] *= vel_factor;
    for (int i = 0; i < (int)grid.vel_z.size(); ++i) grid.vel_z[i] *= vel_factor;
}

void GasSimulator::processCombustion(float dt) {
    // Yanma işlemi: Yakıt + Yüksek Sıcaklık -> Alev + Isı + Duman
    if (settings.burn_rate <= 0.0f) {
        std::fill(grid.interaction.begin(), grid.interaction.end(), 0.0f);
        return;
    }

    const float max_temp = settings.max_temperature;
    const float max_dens = settings.max_density;
    
    long long cell_count = (long long)grid.getCellCount(); 
    
    auto neighbor_flame_activity = [&](long long idx) -> float {
        const int plane = grid.nx * grid.ny;
        const int k = static_cast<int>(idx / plane);
        const int rem = static_cast<int>(idx - static_cast<long long>(k) * plane);
        const int j = rem / grid.nx;
        const int i = rem - j * grid.nx;

        float max_neighbor = 0.0f;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) continue;
                    const int ni = i + dx;
                    const int nj = j + dy;
                    const int nk = k + dz;
                    if (ni < 0 || nj < 0 || nk < 0 || ni >= grid.nx || nj >= grid.ny || nk >= grid.nz) continue;
                    max_neighbor = std::max(max_neighbor, grid.interaction[grid.cellIndex(ni, nj, nk)]);
                }
            }
        }
        return std::clamp(max_neighbor * 0.10f, 0.0f, 1.0f);
    };

    #pragma omp parallel for
    for (long long i = 0; i < cell_count; ++i) {
        // Defensive check: reset NaNs if any
        if (!std::isfinite(grid.fuel[i])) grid.fuel[i] = 0.0f;
        if (!std::isfinite(grid.temperature[i])) grid.temperature[i] = settings.ambient_temperature;
        if (!std::isfinite(grid.density[i])) grid.density[i] = 0.0f;

        float f = grid.fuel[i];
        float t = grid.temperature[i];
        
        const float previous_interaction = grid.interaction[i];

        // Reset reaction each frame
        grid.interaction[i] = 0.0f;
        
        if (f > 0.001f && t > settings.ignition_temperature) {
            // Yanma gerçekleşir!
            float den = grid.density[i];
            const float ignition_band = std::max(60.0f, (max_temp - settings.ignition_temperature) * 0.08f);
            const float pilot_factor = std::clamp((t - settings.ignition_temperature) / std::max(ignition_band * 0.4f, 1.0f), 0.0f, 1.0f);
            const float autoignite_start = settings.ignition_temperature + std::max(520.0f, ignition_band * 5.0f);
            const float autoignite_factor = std::clamp((t - autoignite_start) / std::max(ignition_band * 5.5f, 220.0f), 0.0f, 1.0f);
            const float flame_memory = std::clamp(previous_interaction * 0.12f, 0.0f, 1.0f);
            const float neighbor_flame = neighbor_flame_activity(i);
            const float guided_flame = std::max(flame_memory, neighbor_flame);
            const float burn_gate = std::max(guided_flame, autoignite_factor * 0.2f);
            if (guided_flame < 0.04f && autoignite_factor < 0.35f) {
                continue;
            }
            const float thermal_saturation = std::clamp((t - settings.ignition_temperature) /
                std::max(max_temp - settings.ignition_temperature, 1.0f), 0.0f, 1.0f);
            const float oxygen_proxy = std::clamp(
                1.0f -
                (den / std::max(max_dens * 0.55f, 0.001f)) -
                (f / 8.0f) -
                thermal_saturation * 0.25f -
                flame_memory * 0.2f +
                neighbor_flame * 0.15f,
                0.0f, 1.0f);
            float density_throttle = std::clamp(1.0f - (den / std::max(max_dens * 0.65f, 0.001f)), 0.08f, 1.0f);
            
            float burned = f * settings.burn_rate * dt * density_throttle;
            if (burned > f) burned = f;
            
            // Soft throttle: strongly reduce burn rate as temperature approaches max
            float range = max_temp - settings.ignition_temperature;
            float temp_factor = (max_temp - t) / (range > 0 ? range : 1.0f);
            float temp_headroom = std::max(0.0f, std::min(1.0f, temp_factor));
            burned *= temp_headroom * pilot_factor * std::max(0.02f, burn_gate) * oxygen_proxy * oxygen_proxy * oxygen_proxy;
            burned = std::min(burned, 0.035f * dt + f * 0.045f);
            
            if (burned < 0.00001f) continue; 
            
            grid.fuel[i] -= burned;
            
            float heat_to_add = burned * settings.heat_release;
            float new_temp = t + heat_to_add;
            float soft_start = max_temp * 0.90f;
            
            if (new_temp > soft_start) {
                float excess = new_temp - soft_start;
                float scale = max_temp - soft_start;
                new_temp = soft_start + scale * (1.0f - std::exp(-excess / scale));
            }
            grid.temperature[i] = std::min(new_temp, max_temp);
            grid.density[i] = std::min(den + burned * settings.smoke_generation, max_dens);
            
            grid.interaction[i] = burned / dt;
            
            // Genişleme (patlama etkisi): Pressure solver patlamasın diye limitle
            if (settings.expansion_strength > 0) {
                float expansion_val = burned * settings.expansion_strength * density_throttle * temp_headroom;
                grid.pressure[i] += std::min(expansion_val, 100.0f); 
            }
        }
    }
}

void GasSimulator::enforceBoundaries() {
    // OPEN boundaries are best for rising plumes. Periodic keeps velocity continuous
    // across the domain and is the only mode compatible with the current spectral FFT solve.
    if (settings.boundary_mode == GasSimulationSettings::BoundaryMode::Periodic) {
        return;
    }

    // OPEN BOUNDARIES: We no longer zero the velocities at the domain borders.
    // This allows smoke to flow out freely instead of bouncing or pooling.
    // The pressure solver with Dirichlet p=0 handles the flux.
    /*
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            grid.vel_x[grid.velXIndex(0, j, k)] = 0.0f;
            grid.vel_x[grid.velXIndex(grid.nx, j, k)] = 0.0f;
        }
    }
    ...
    */
}

void GasSimulator::applyPreset(const std::string& name) {
    const float domain_width = std::max(grid.nx * grid.voxel_size, grid.voxel_size);
    const float domain_height = std::max(grid.ny * grid.voxel_size, grid.voxel_size);
    const float domain_depth = std::max(grid.nz * grid.voxel_size, grid.voxel_size);
    const float domain_xz = std::max(std::min(domain_width, domain_depth), grid.voxel_size);

    auto ensurePrimaryEmitter = [&]() -> Emitter& {
        if (emitters.empty()) {
            Emitter e;
            e.name = "Primary Emitter";
            e.shape = EmitterShape::Sphere;
            e.position = grid.origin + Vec3(domain_width * 0.5f,
                                            std::max(grid.voxel_size * 2.0f, domain_height * 0.08f),
                                            domain_depth * 0.5f);
            e.radius = std::max(grid.voxel_size * 3.0f, domain_xz * 0.08f);
            emitters.push_back(e);
        }
        return emitters[0];
    };

    auto resetEmitterAdvancedControls = [&](Emitter& e) {
        e.fuel_phase = FuelPhase::Gas;
        e.phase_change_temperature = 420.0f;
        e.fuel_release_rate = 1.0f;
        e.flame_contact_sensitivity = 0.35f;
        e.falloff_type = EmitterFalloffType::Smooth;
        e.falloff_start = 0.15f;
        e.falloff_end = 1.0f;
        e.noise_enabled = true;
        e.noise_modulate_density = true;
        e.noise_modulate_temperature = false;
        e.noise_modulate_velocity = true;
        e.noise_speed = 0.45f;
        e.spray_cone_angle = 0.0f;
        e.speed_min = 1.0f;
        e.speed_max = 1.0f;
        e.emission_mode = EmitterEmissionMode::Continuous;
        e.start_frame = 0.0f;
        e.end_frame = -1.0f;
        e.pulse_interval = 10.0f;
        e.pulse_duration = 3.0f;
        e.enabled = true;
    };

    if (name == "Fire") {
        // ═══════════════════════════════════════════════════════════════════
        // FIRE PRESET - Houdini/EmberGen style realistic fire
        // Key: Strong buoyancy, fast fuel burn, moderate vorticity
        // ═══════════════════════════════════════════════════════════════════
        settings.time_scale = 0.45f;              // Slower default so fire growth stays controllable
        settings.substeps = 2;                    // More substeps for stability
        
        // Dissipation - keep values high (close to 1.0) for persistence
        settings.density_dissipation = 0.995f;    // Smoke persists well
        settings.temperature_dissipation = 0.95f; // Heat cools faster to prevent runaway
        settings.velocity_dissipation = 0.98f;    // Velocity persists
        settings.fuel_dissipation = 0.92f;        // Fuel burns away faster
        
        // Buoyancy - STRONG upward force
        settings.buoyancy_density = -0.3f;        // Slight downward from density
        settings.buoyancy_temperature = 3.8f;     // Strong upward force from heat without instant runaway
        settings.ambient_temperature = 293.0f;    // Room temp (20°C)
        
        // Turbulence & Detail
        settings.vorticity_strength = 0.8f;       // Good flame detail
        settings.turbulence_strength = 0.3f;      // Add noise for flicker
        settings.turbulence_scale = 2.0f;
        
        // Combustion
        settings.ignition_temperature = 350.0f;   // Lower ignition for easy start
        settings.burn_rate = 1.35f;               // More controlled combustion
        settings.heat_release = 72.0f;            // Warm but less explosive default
        settings.expansion_strength = 2.0f;       // Fire expands outward without blowing up the whole domain
        settings.smoke_generation = 0.65f;        // Moderate smoke from fire
        settings.soot_generation = 0.1f;
        settings.boundary_mode = GasSimulationSettings::BoundaryMode::Open;
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack;
        settings.adaptive_timestep = true;
        settings.cfl_number = 0.45f;
        
        Emitter& e = ensurePrimaryEmitter();
        resetEmitterAdvancedControls(e);
        e.name = "Fire Emitter";
        e.shape = EmitterShape::Cone;
        e.position = grid.origin + Vec3(domain_width * 0.5f,
                                        std::max(grid.voxel_size * 1.2f, domain_height * 0.04f),
                                        domain_depth * 0.5f);
        e.radius = std::max(grid.voxel_size * 2.5f, domain_xz * 0.09f);
        e.height = std::max(grid.voxel_size * 7.0f, domain_height * 0.18f);
        e.cone_angle = 28.0f;
        e.fuel_rate = 16.0f;
        e.density_rate = 1.6f;
        e.temperature = 920.0f;
        e.fuel_phase = FuelPhase::Gas;
        e.velocity = Vec3(0, 3.4f, 0);
        e.falloff_start = 0.0f;
        e.falloff_end = 0.85f;
        e.noise_frequency = 1.35f;
        e.noise_amplitude = 0.55f;
        e.noise_speed = 1.25f;
        e.noise_modulate_temperature = true;
        e.noise_modulate_velocity = true;
        e.spray_cone_angle = 12.0f;
        e.speed_min = 0.9f;
        e.speed_max = 1.35f;
    }
    else if (name == "Smoke") {
        // ═══════════════════════════════════════════════════════════════════
        // SMOKE PRESET - Dense, slow-rising smoke plume
        // Key: Moderate buoyancy, no combustion, slow dissipation
        // ═══════════════════════════════════════════════════════════════════
        settings.time_scale = 0.55f;
        settings.substeps = 1;
        
        // Dissipation - smoke lingers (values close to 1.0)
        settings.density_dissipation = 0.995f;    // Smoke stays much longer
        settings.temperature_dissipation = 0.98f; // Temperature persists for buoyancy
        settings.velocity_dissipation = 0.98f;    // Velocity persists
        settings.fuel_dissipation = 0.99f;
        
        // Buoyancy - warm smoke rises
        settings.buoyancy_density = -0.3f;        // Slight downward from mass
        settings.buoyancy_temperature = 3.8f;     // Lift, but less rocket-like at low resolutions
        settings.ambient_temperature = 293.0f;
        
        // Turbulence - subtle detail
        settings.vorticity_strength = 0.6f;       // Some swirls
        settings.turbulence_strength = 0.15f;     // Minimal noise
        settings.turbulence_scale = 1.5f;
        
        // No combustion
        settings.burn_rate = 0.0f;
        settings.heat_release = 0.0f;
        settings.expansion_strength = 0.0f;
        settings.smoke_generation = 0.0f;
        settings.boundary_mode = GasSimulationSettings::BoundaryMode::Open;
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack;
        settings.adaptive_timestep = true;
        
        Emitter& e = ensurePrimaryEmitter();
        resetEmitterAdvancedControls(e);
        e.name = "Smoke Emitter";
        e.shape = EmitterShape::Cylinder;
        e.position = grid.origin + Vec3(domain_width * 0.5f,
                                        std::max(grid.voxel_size * 1.0f, domain_height * 0.035f),
                                        domain_depth * 0.5f);
        e.radius = std::max(grid.voxel_size * 3.5f, domain_xz * 0.12f);
        e.height = std::max(grid.voxel_size * 3.0f, domain_height * 0.10f);
        e.fuel_rate = 0.0f;
        e.density_rate = 18.0f;
        e.temperature = 375.0f;
        e.fuel_phase = FuelPhase::Gas;
        e.velocity = Vec3(0, 1.8f, 0);
        e.falloff_type = EmitterFalloffType::Gaussian;
        e.falloff_start = 0.0f;
        e.falloff_end = 1.0f;
        e.noise_frequency = 0.95f;
        e.noise_amplitude = 0.42f;
        e.noise_speed = 0.42f;
        e.noise_modulate_temperature = true;
        e.noise_modulate_velocity = true;
        e.spray_cone_angle = 8.0f;
        e.speed_min = 0.85f;
        e.speed_max = 1.15f;
    }
    else if (name == "Explosion") {
        // ═══════════════════════════════════════════════════════════════════
        // EXPLOSION PRESET - Initial burst with shockwave
        // Key: Massive initial energy, fast expansion, high vorticity
        // ═══════════════════════════════════════════════════════════════════
        settings.time_scale = 0.8f;               // Slightly slow-mo for drama
        settings.substeps = 3;                    // High substeps for stability
        
        // Dissipation
        settings.density_dissipation = 0.94f;     // Smoke fades over time
        settings.temperature_dissipation = 0.80f; // Fire cools quickly
        settings.velocity_dissipation = 0.88f;    // Blast wave dies down
        settings.fuel_dissipation = 0.75f;        // Fuel consumed rapidly
        
        // Buoyancy - initial outward, then thermal rise
        settings.buoyancy_density = -0.3f;
        settings.buoyancy_temperature = 4.0f;     // Strong thermal lift
        settings.ambient_temperature = 293.0f;
        
        // Turbulence - chaotic
        settings.vorticity_strength = 3.0f;       // Very turbulent
        settings.turbulence_strength = 0.8f;      // High noise
        settings.turbulence_scale = 3.0f;
        
        // Combustion - intense but controlled
        settings.ignition_temperature = 300.0f;
        settings.burn_rate = 8.0f;                
        settings.heat_release = 200.0f;           // High heat for explosion
        settings.expansion_strength = 50.0f;      
        settings.smoke_generation = 2.0f;         // Lots of smoke
        settings.soot_generation = 0.5f;
        
        // Stability limits for explosion - allow more headroom
        settings.max_temperature = 6000.0f;       // Full range for explosion
        settings.max_velocity = 400.0f;
        settings.max_density = 50.0f;
        settings.boundary_mode = GasSimulationSettings::BoundaryMode::Open;
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack; // High detail
        settings.adaptive_timestep = true;
        settings.cfl_number = 0.4f;               // More conservative for stability

        Emitter& e = ensurePrimaryEmitter();
        resetEmitterAdvancedControls(e);
        e.name = "Explosion Emitter";
        e.shape = EmitterShape::Sphere;
        e.position = grid.origin + Vec3(grid.nx * grid.voxel_size * 0.5f,
                                        grid.ny * grid.voxel_size * 0.22f,
                                        grid.nz * grid.voxel_size * 0.5f);
        e.radius = grid.voxel_size * std::max(3.0f, grid.nx * 0.06f);
        e.fuel_rate = 120.0f;
        e.density_rate = 10.0f;
        e.temperature = 1800.0f;
        e.fuel_phase = FuelPhase::Liquid;
        e.phase_change_temperature = 520.0f;
        e.fuel_release_rate = 1.35f;
        e.flame_contact_sensitivity = 0.55f;
        e.velocity = Vec3(0, 8.0f, 0);
        e.falloff_type = EmitterFalloffType::Gaussian;
        e.falloff_start = 0.0f;
        e.falloff_end = 0.75f;
        e.noise_frequency = 2.2f;
        e.noise_amplitude = 0.85f;
        e.noise_speed = 2.0f;
        e.noise_modulate_temperature = true;
        e.noise_modulate_velocity = true;
        e.spray_cone_angle = 35.0f;
        e.speed_min = 1.0f;
        e.speed_max = 1.8f;
        e.emission_mode = EmitterEmissionMode::Burst;
        e.start_frame = 0.0f;
        e.end_frame = 2.0f;
        
        // Clear grid and set initial 'fireball'
        grid.clear();
        std::fill(grid.temperature.begin(), grid.temperature.end(), settings.ambient_temperature);
        
        int cx = settings.resolution_x / 2;
        int cy = settings.resolution_y / 4;       // Start near bottom
        int cz = settings.resolution_z / 2;
        int radius = std::max(4, settings.resolution_x / 8);  // Bigger initial blast
        
        for (int k = -radius; k <= radius; ++k) {
            for (int j = -radius; j <= radius; ++j) {
                for (int i = -radius; i <= radius; ++i) {
                    int x = cx + i, y = cy + j, z = cz + k;
                    if (x >= 0 && x < grid.nx && y >= 0 && y < grid.ny && z >= 0 && z < grid.nz) {
                        float dist = sqrtf((float)(i*i + j*j + k*k));
                        if (dist <= radius) {
                            size_t idx = grid.cellIndex(x, y, z);
                            float factor = 1.0f - (dist / (float)radius);
                            factor = factor * factor;  // Quadratic falloff for sharper core
                            
                            grid.fuel[idx] = 150.0f * factor;
                            grid.temperature[idx] = settings.ambient_temperature + 4000.0f * factor;
                            grid.density[idx] = 15.0f * factor;
                            
                            // Initial outward velocity (explosion wave)
                            float vx = (float)i / (radius + 0.01f) * 20.0f * factor;
                            float vy = (float)j / (radius + 0.01f) * 20.0f * factor + 10.0f * factor;
                            float vz = (float)k / (radius + 0.01f) * 20.0f * factor;
                            
                            // Set velocities (approximate - staggered grid)
                            if (x < grid.nx) grid.vel_x[grid.velXIndex(x, y, z)] += vx;
                            if (y < grid.ny) grid.vel_y[grid.velYIndex(x, y, z)] += vy;
                            if (z < grid.nz) grid.vel_z[grid.velZIndex(x, y, z)] += vz;
                        }
                    }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

float GasSimulator::sampleDensity(const Vec3& world_pos) const {
    return grid.sampleDensity(world_pos);
}

float GasSimulator::sampleTemperature(const Vec3& world_pos) const {
    return grid.sampleTemperature(world_pos);
}

float GasSimulator::sampleFlameIntensity(const Vec3& world_pos) const {
    return grid.sampleInteraction(world_pos);
}

float GasSimulator::sampleFuel(const Vec3& world_pos) const {
    return grid.sampleFuel(world_pos);
}

Vec3 GasSimulator::sampleVelocity(const Vec3& world_pos) const {
    return grid.sampleVelocity(world_pos);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BAKING
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::startBake(int start_frame, int end_frame, const std::string& cache_dir, const Matrix4x4& world_matrix) {
    if (is_baking) return;
    
    is_baking = true;
    cancel_bake = false;
    bake_progress = 0.0f;
    
    // TODO: Run baking in separate thread
    // For now, synchronous
    reset();
    float dt = 1.0f / settings.bake_fps;
    
    for (int frame = start_frame; frame <= end_frame && !cancel_bake; ++frame) {
        step(dt, world_matrix);
        
        // Save frame to cache
        std::string filename = cache_dir + "/frame_" + std::to_string(frame) + ".gas";
        // TODO: Implement frame caching
        
        bake_progress = static_cast<float>(frame - start_frame) / (end_frame - start_frame);
    }
    
    is_baking = false;
}

void GasSimulator::cancelBake() {
    cancel_bake = true;
}

bool GasSimulator::loadBakedFrame(int frame) {
    // TODO: Implement frame loading from cache
    return false;
}

// Enable OpenVDB support for Gas Simulation exports
#ifndef OPENVDB_ENABLED
#define OPENVDB_ENABLED
#endif

bool GasSimulator::exportToVDB(const std::string& filepath) const {
#ifdef OPENVDB_ENABLED
    
    // Create density grid
    openvdb::FloatGrid::Ptr density_grid = openvdb::FloatGrid::create();
    density_grid->setName("density");
    density_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
    
    openvdb::FloatGrid::Accessor accessor = density_grid->getAccessor();
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float d = grid.density[grid.cellIndex(i, j, k)];
                if (d > 0.000001f) { // Lowered threshold to preserve faint smoke
                    accessor.setValue(openvdb::Coord(i, j, k), d);
                }
            }
        }
    }
    
    // Create temperature grid (store as delta above ambient for better range)
    openvdb::FloatGrid::Ptr temp_grid = openvdb::FloatGrid::create();
    temp_grid->setName("temperature");
    openvdb::FloatGrid::Accessor temp_accessor = temp_grid->getAccessor();
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float t = grid.temperature[grid.cellIndex(i, j, k)];
                // Write absolute temperature (Kelvin). 
                // Industry standard VDBs use Kelvin for blackbody temperature.
                if (t > 0.001f) {
                    temp_accessor.setValue(openvdb::Coord(i, j, k), t);
                }
            }
        }
    }
    
    // Create fuel grid
    openvdb::FloatGrid::Ptr fuel_grid = openvdb::FloatGrid::create();
    fuel_grid->setName("fuel");
    openvdb::FloatGrid::Accessor fuel_accessor = fuel_grid->getAccessor();
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float f = grid.fuel[grid.cellIndex(i, j, k)];
                if (f > 0.001f) {
                    fuel_accessor.setValue(openvdb::Coord(i, j, k), f);
                }
            }
        }
    }
    
    // Create flame/interaction grid (fire intensity)
    openvdb::FloatGrid::Ptr flame_grid = openvdb::FloatGrid::create();
    flame_grid->setName("flame");
    openvdb::FloatGrid::Accessor flame_accessor = flame_grid->getAccessor();
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float f = grid.interaction[grid.cellIndex(i, j, k)];
                if (f > 0.001f) {
                    flame_accessor.setValue(openvdb::Coord(i, j, k), f);
                }
            }
        }
    }
    
    // Set transforms (Proper world-space mapping)
    openvdb::math::Transform::Ptr vdb_transform = 
        openvdb::math::Transform::createLinearTransform(grid.voxel_size);
    vdb_transform->postTranslate(openvdb::Vec3d(grid.origin.x, grid.origin.y, grid.origin.z));
    
    density_grid->setTransform(vdb_transform);
    temp_grid->setTransform(vdb_transform);
    fuel_grid->setTransform(vdb_transform);
    flame_grid->setTransform(vdb_transform);
    
    // Write file
    openvdb::GridPtrVec grids;
    grids.push_back(density_grid);
    grids.push_back(temp_grid);
    grids.push_back(fuel_grid);
    grids.push_back(flame_grid);
    
    try {
        openvdb::io::File file(filepath);
        file.write(grids);
        file.close();
        return true;
    } catch (...) {
        return false;
    }
#else
    // OpenVDB not available - write raw binary format as fallback
    std::ofstream file(filepath, std::ios::binary);
    if (!file) return false;
    
    // Header
    int32_t magic = 0x47415356; // "GASV"
    file.write(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&grid.nx), sizeof(grid.nx));
    file.write(reinterpret_cast<const char*>(&grid.ny), sizeof(grid.ny));
    file.write(reinterpret_cast<const char*>(&grid.nz), sizeof(grid.nz));
    file.write(reinterpret_cast<const char*>(&grid.voxel_size), sizeof(grid.voxel_size));
    
    // Density data
    file.write(reinterpret_cast<const char*>(grid.density.data()), 
               grid.density.size() * sizeof(float));
    
    // Temperature data  
    file.write(reinterpret_cast<const char*>(grid.temperature.data()),
               grid.temperature.size() * sizeof(float));
    
    file.close();
    return true;
#endif
}

bool GasSimulator::exportSequenceToVDB(const std::string& directory,
                                       const std::string& base_name,
                                       int start_frame, int end_frame,
                                       const Matrix4x4& world_matrix) {
    if (is_baking) return false;
    
    is_baking = true;
    cancel_bake = false;
    bake_progress = 0.0f;
    baking_frame = start_frame;
    
    // Thread safety: copy parameters
    std::string dir = directory;
    std::string name = base_name;
    // Capture matrix by value
    Matrix4x4 bake_world_matrix = world_matrix;
    
    if (bake_thread && bake_thread->joinable()) bake_thread->join();
    
    bake_thread = std::make_unique<std::thread>([this, dir, name, start_frame, end_frame, bake_world_matrix]() {
        // If using CUDA, we should ensure this thread has access to the device
        if (settings.backend == SolverBackend::CUDA) {
            cudaSetDevice(0); 
        }
        
        reset(); // Start from fresh state
        float dt = 1.0f / 24.0f; 
        
        std::string clean_dir = dir;
        if (!clean_dir.empty() && (clean_dir.back() == '/' || clean_dir.back() == '\\')) {
            clean_dir.pop_back();
        }
        std::filesystem::create_directories(clean_dir);

        for (int frame = start_frame; frame <= end_frame; ++frame) {
            if (cancel_bake) break;
            baking_frame = frame;

            // IMPORTANT: Apply keyframes to emitters during bake!
            // This was the cause of 2KB empty VDBs.
            for (auto& e : emitters) {
                if (!e.keyframes.empty()) {
                    e.applyKeyframe(e.getInterpolatedKeyframe((float)frame));
                }
            }
            
            if (frame > start_frame) {
                step(dt, bake_world_matrix);
            }

            // CRITICAL: If using CUDA, download data to CPU before VDB export
            if (settings.backend == SolverBackend::CUDA) {
                downloadFromGPU();
            }
            
            char filename[256];
            sprintf_s(filename, "%s/%s_%04d.vdb", clean_dir.c_str(), name.c_str(), frame);
            exportToVDB(filename);
            
            bake_progress = static_cast<float>(frame - start_frame) / std::max(1, (end_frame - start_frame));
        }
        
        is_baking = false;
        bake_progress = 1.0f;
    });

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU (CUDA) - Placeholder implementations
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::initCUDA() {
    SCENE_LOG_INFO("[GasSimulator::initCUDA] START - Resolution: " +
        std::to_string(settings.resolution_x) + "x" +
        std::to_string(settings.resolution_y) + "x" +
        std::to_string(settings.resolution_z));

    if (!g_hasCUDA) {
        SCENE_LOG_WARN("[GasSimulator::initCUDA] CUDA unavailable. Keeping CPU backend.");
        cuda_initialized = false;
        return;
    }

    // Clear any previous sticky CUDA errors before allocation
    cudaGetLastError();

    if (cuda_initialized) {
        SCENE_LOG_INFO("[GasSimulator::initCUDA] Freeing old CUDA resources first...");
        freeCUDA();
    }

    FluidSim::cuda_init_simulation(
        settings.resolution_x, settings.resolution_y, settings.resolution_z,
        (float**)&d_density, (float**)&d_temperature, (float**)&d_fuel,
        (float**)&d_vel_x, (float**)&d_vel_y, (float**)&d_vel_z,
        (float**)&d_pressure, (float**)&d_divergence,
        (float**)&d_vort_x, (float**)&d_vort_y, (float**)&d_vort_z
    );

    // Allocate temporary buffers for advection
    size_t sz = (size_t)settings.resolution_x * settings.resolution_y * settings.resolution_z * sizeof(float);
    cudaMalloc(&d_tmp1, sz);
    cudaMalloc(&d_tmp2, sz);
    cudaMalloc(&d_tmp3, sz);

    cudaError_t allocErr = cudaGetLastError();
    if (allocErr != cudaSuccess) {
        SCENE_LOG_ERROR("[GasSimulator::initCUDA] CUDA allocation error: " + std::string(cudaGetErrorString(allocErr)));
        FluidSim::cuda_free_simulation(
            (float*)d_density, (float*)d_temperature, (float*)d_fuel,
            (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
            (float*)d_pressure, (float*)d_divergence,
            (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z
        );
        if (d_tmp1) cudaFree(d_tmp1);
        if (d_tmp2) cudaFree(d_tmp2);
        if (d_tmp3) cudaFree(d_tmp3);
        d_density = d_temperature = d_fuel = nullptr;
        d_vel_x = d_vel_y = d_vel_z = nullptr;
        d_pressure = d_divergence = nullptr;
        d_vort_x = d_vort_y = d_vort_z = nullptr;
        d_tmp1 = d_tmp2 = d_tmp3 = nullptr;
        cuda_initialized = false;
        gpu_data_valid = false;
        return;
    }

    // Initial upload of CPU state
    uploadToGPU();

    // Initialize FFT solver if pressure solver mode is FFT
    if (settings.pressure_solver == GasSimulationSettings::PressureSolverMode::FFT) {
        initFFTSolver();
    }

    cuda_initialized = true;
    gpu_data_valid = true;

    SCENE_LOG_INFO("[GasSimulator::initCUDA] COMPLETE - Buffer size: " + std::to_string(sz) + " bytes");
}
void GasSimulator::freeCUDA() {
    if (!cuda_initialized) return;
    if (!g_hasCUDA) {
        SCENE_LOG_WARN("[GasSimulator::freeCUDA] CUDA not present, skipping GPU free");
        cuda_initialized = false;
        gpu_data_valid = false;
        return;
    }
    
    SCENE_LOG_INFO("[GasSimulator::freeCUDA] START - Syncing device before free...");
    
    // Clear sticky errors before sync
    cudaGetLastError();
    
    // Check if CUDA is available before sync
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        // Wait for all async operations to complete before freeing
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            SCENE_LOG_ERROR("[GasSimulator::freeCUDA] cudaDeviceSynchronize failed: " + 
                            std::string(cudaGetErrorString(err)));
        }
        // Clear any errors from sync (may happen if previous operations failed)
        cudaGetLastError();
    } else {
        SCENE_LOG_WARN("[GasSimulator::freeCUDA] CUDA not available, skipping sync");
    }
    
    FluidSim::cuda_free_simulation(
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
        (float*)d_pressure, (float*)d_divergence,
        (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z
    );

    if (d_tmp1) cudaFree(d_tmp1);
    if (d_tmp2) cudaFree(d_tmp2);
    if (d_tmp3) cudaFree(d_tmp3);
    
    // Free force field GPU buffer
    freeGPUForceFields();
    
    // Free advanced emitters
    freeGPUAdvancedEmitters();
    
    // Cleanup FFT solver
    cleanupFFTSolver();
    
    d_density = d_temperature = d_fuel = nullptr;
    d_vel_x = d_vel_y = d_vel_z = nullptr;
    d_pressure = d_divergence = nullptr;
    d_vort_x = d_vort_y = d_vort_z = nullptr;
    d_tmp1 = d_tmp2 = d_tmp3 = nullptr;
    
    cuda_initialized = false;
    gpu_data_valid = false;
    
    SCENE_LOG_INFO("[GasSimulator::freeCUDA] COMPLETE");
}

void GasSimulator::clearGPU() {
    if (!cuda_initialized) return;
    
    // CRITICAL: Use grid dimensions (actual buffer size), not settings
    size_t sz = (size_t)grid.nx * grid.ny * grid.nz * sizeof(float);
    
    // Clear all GPU buffers to zero
    cudaMemset(d_density, 0, sz);
    cudaMemset(d_fuel, 0, sz);
    cudaMemset(d_vel_x, 0, sz);
    cudaMemset(d_vel_y, 0, sz);
    cudaMemset(d_vel_z, 0, sz);
    cudaMemset(d_pressure, 0, sz);
    cudaMemset(d_divergence, 0, sz);
    cudaMemset(d_vort_x, 0, sz);
    cudaMemset(d_vort_y, 0, sz);
    cudaMemset(d_vort_z, 0, sz);
    
    // Set temperature to ambient
    std::vector<float> ambient_temp_data((size_t)grid.nx * grid.ny * grid.nz, settings.ambient_temperature);
    cudaMemcpy(d_temperature, ambient_temp_data.data(), sz, cudaMemcpyHostToDevice);
    
    gpu_data_valid = true;
}

void GasSimulator::stepCUDA(float dt, const Matrix4x4& world_matrix) {
    if (!cuda_initialized) {
        SCENE_LOG_WARN("[GasSimulator::stepCUDA] Called but cuda_initialized=false!");
        return;
    }

    if (settings.pressure_solver == GasSimulationSettings::PressureSolverMode::FFT && canUseFFTPressureSolver()) {
        if (!use_fft_solver || !fft_solver || fft_solver->nx != grid.nx || fft_solver->ny != grid.ny || fft_solver->nz != grid.nz) {
            initFFTSolver();
        }
    } else if (use_fft_solver) {
        cleanupFFTSolver();
    }
    
    // SAFETY CHECK: Resolution mismatch should be handled in step(), but double-check here
    // This is a fallback safety check in case auto-reinit in step() fails
    if (settings.resolution_x != grid.nx || 
        settings.resolution_y != grid.ny || 
        settings.resolution_z != grid.nz) {
        SCENE_LOG_ERROR("[GasSimulator::stepCUDA] Resolution mismatch still present after step()! " 
                        "Settings: " + std::to_string(settings.resolution_x) + "x" + 
                        std::to_string(settings.resolution_y) + "x" + 
                        std::to_string(settings.resolution_z) + 
                        ", Grid: " + std::to_string(grid.nx) + "x" + 
                        std::to_string(grid.ny) + "x" + std::to_string(grid.nz) +
                        ". This should not happen - check step() auto-reinit logic!");
        return; // Do NOT run kernels with wrong resolution!
    }
    
    // Validate pointers before use
    if (!d_density || !d_temperature || !d_fuel || !d_vel_x || !d_vel_y || !d_vel_z ||
        !d_pressure || !d_divergence || !d_tmp1 || !d_tmp2 || !d_tmp3) {
        SCENE_LOG_ERROR("[GasSimulator::stepCUDA] NULL pointer detected! Aborting step.");
        return;
    }

    // NOTE: Removed pre-step cudaDeviceSynchronize - was causing UI freezes
    // Errors will be caught by individual kernel launches or at end of step
    cudaGetLastError(); // Just clear sticky error without sync

    uploadAdvancedEmittersToGPU();

    float world_mat[16];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            world_mat[c * 4 + r] = world_matrix.m[r][c];
        }
    }

    float substep_dt = dt / settings.substeps;
    float inv_voxel_size = 1.0f / (settings.voxel_size > 0.0001f ? settings.voxel_size : 0.1f);
    
    const float step_time_base = accumulated_time;
    for (int sub = 0; sub < settings.substeps; ++sub) {
        const float substep_time = step_time_base + static_cast<float>(sub) * substep_dt;
        // 1. Apply Emitters
        if (d_advanced_emitters && gpu_advanced_emitter_count > 0) {
            FluidSim::cuda_apply_advanced_emitters(
                grid.nx, grid.ny, grid.nz, substep_dt,
                (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                d_advanced_emitters,
                gpu_advanced_emitter_count,
                world_mat,
                static_cast<float>(current_frame) + static_cast<float>(sub) / std::max(1, settings.substeps),
                substep_time
            );
        }
        
        // 2. Apply External Force Fields (GPU)
        if (external_force_field_manager && external_force_field_manager->getActiveCount() > 0) {
            uploadForceFieldsToGPU();
            
            if (d_force_fields && gpu_force_field_count > 0) {
                // NOTE: Force field returns force in m/s², but GPU velocity is in grid units.
                // We need to scale the force by inv_voxel_size to convert to grid units.
                // This is done inside the kernel by passing inv_voxel_size as a parameter.
                // For now, we apply a larger dt to compensate (hacky but works)
                // TODO: Pass inv_voxel_size to kernel and scale force properly
                
                FluidSim::cuda_apply_force_fields(
                    grid.nx, grid.ny, grid.nz, substep_dt * inv_voxel_size,  // Scale dt to convert force to grid units
                    (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                    world_mat,
                    (FluidSim::GPUForceField*)d_force_fields,
                    gpu_force_field_count,
                    substep_time
                );
            }
        }
        
        // 3. Step Simulation using new unified params API
        // Build GPUSimulationParams from settings
        FluidSim::GPUSimulationParams params = {};
        
        // Grid
        params.nx = grid.nx;
        params.ny = grid.ny;
        params.nz = grid.nz;
        params.voxel_size = settings.voxel_size;
        params.inv_voxel_size = inv_voxel_size;
        
        // Timestep
        params.dt = substep_dt;
        params.time = substep_time;
        
        // Advection
        params.advection_mode = static_cast<int>(settings.advection_mode);
        
        // Forces (convert to grid units)
        params.buoyancy_density = settings.buoyancy_density * inv_voxel_size;
        params.buoyancy_temperature = settings.buoyancy_temperature * inv_voxel_size;
        params.ambient_temperature = settings.ambient_temperature;
        
        // Gravity (convert to grid units) - now actually used!
        Vec3 gravity_grid = settings.gravity * inv_voxel_size;
        params.gravity_x = gravity_grid.x;
        params.gravity_y = gravity_grid.y;
        params.gravity_z = gravity_grid.z;
        
        // Wind (convert to grid units)
        Vec3 wind_grid = settings.wind * inv_voxel_size;
        params.wind_x = wind_grid.x;
        params.wind_y = wind_grid.y;
        params.wind_z = wind_grid.z;
        
        // Dissipation (pre-compute pow for GPU)
        params.density_dissipation = std::pow(settings.density_dissipation, substep_dt);
        params.velocity_dissipation = std::pow(settings.velocity_dissipation, substep_dt);
        params.temperature_dissipation = std::pow(settings.temperature_dissipation, substep_dt);
        params.fuel_dissipation = std::pow(settings.fuel_dissipation, substep_dt);
        
        // Combustion
        params.ignition_temperature = settings.ignition_temperature;
        params.burn_rate = settings.burn_rate;
        params.heat_release = settings.heat_release;
        params.smoke_generation = settings.smoke_generation;
        params.expansion_strength = settings.expansion_strength;
        
        // Vorticity & Turbulence
        params.vorticity_strength = settings.vorticity_strength;
        params.turbulence_strength = settings.turbulence_strength;
        params.turbulence_scale = settings.turbulence_scale;
        params.turbulence_octaves = settings.turbulence_octaves;
        params.turbulence_lacunarity = settings.turbulence_lacunarity;
        params.turbulence_persistence = settings.turbulence_persistence;
        
        // Pressure Solver
        params.pressure_iterations = settings.pressure_iterations;
        params.pressure_solver_mode = static_cast<int>(settings.pressure_solver);
        params.sor_omega = settings.sor_omega;
        // Stability limits (now from UI!)
        params.max_velocity = settings.max_velocity;      
        params.max_temperature = settings.max_temperature;  
        params.max_density = settings.max_density;               
        // Sparse (future)
        params.sparse_mode = settings.sparse_mode ? 1 : 0;
        params.sparse_threshold = settings.sparse_threshold;
        
        const bool use_fft_projection =
            settings.pressure_solver == GasSimulationSettings::PressureSolverMode::FFT &&
            use_fft_solver && fft_solver != nullptr;

        if (use_fft_projection) {
            FluidSim::cuda_step_simulation_v2_fft(
                params,
                fft_solver,
                (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                (float*)d_pressure, (float*)d_divergence,
                (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z,
                (float*)d_tmp1, (float*)d_tmp2, (float*)d_tmp3
            );
        } else {
            FluidSim::cuda_step_simulation_v2(
                params,
                (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                (float*)d_pressure, (float*)d_divergence,
                (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z,
                (float*)d_tmp1, (float*)d_tmp2, (float*)d_tmp3
            );
        }
    }
    
    // NOTE: Removed post-step cudaDeviceSynchronize - was causing UI freezes
    // CUDA kernels run asynchronously, errors will be caught on next operation that requires sync
    
    // Mark valid so next upload won't be redundant (though step updates it anyway)
    gpu_data_valid = true;
}

void GasSimulator::uploadToGPU() {
    if (!cuda_initialized) return;
    
    // CRITICAL: Use grid dimensions (actual allocated size), not settings (user requested)
    FluidSim::cuda_upload_data(
        grid.nx, grid.ny, grid.nz,
        grid.density.data(), grid.temperature.data(), grid.fuel.data(),
        grid.vel_x.data(), grid.vel_y.data(), grid.vel_z.data(),
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z
    );
    
    gpu_data_valid = true;
}

void GasSimulator::downloadFromGPU() {
    if (!cuda_initialized) return;
    
    // CRITICAL: Use grid dimensions (actual allocated size), not settings (user requested)
    FluidSim::cuda_download_data(
        grid.nx, grid.ny, grid.nz,
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
        grid.density.data(), grid.temperature.data(), grid.fuel.data(),
        grid.vel_x.data(), grid.vel_y.data(), grid.vel_z.data()
    );
}

GasSimulator::StateSnapshot GasSimulator::captureState() const {
    StateSnapshot snapshot;
    snapshot.grid = grid;
    snapshot.grid_temp = grid_temp;
    snapshot.persistent_vorticity = persistent_vorticity;
    snapshot.current_frame = current_frame;
    snapshot.accumulated_time = accumulated_time;
    snapshot.valid = true;
    return snapshot;
}

bool GasSimulator::canUseFFTPressureSolver() const {
    if (!g_hasCUDA) {
        return false;
    }

    if (settings.backend != SolverBackend::CUDA) {
        return false;
    }

    if (settings.boundary_mode != GasSimulationSettings::BoundaryMode::Periodic) {
        return false;
    }

    const int min_dim = std::min({ grid.nx, grid.ny, grid.nz });
    const long long total_cells = (long long)grid.nx * grid.ny * grid.nz;
    if (min_dim < 8 || total_cells <= 0) {
        return false;
    }

    for (const auto& emitter : emitters) {
        if (emitter.enabled && emitter.fuel_rate > 0.0001f) {
            return false;
        }
    }

    return true;
}

void GasSimulator::restoreState(const StateSnapshot& snapshot) {
    if (!snapshot.valid) return;
    grid = snapshot.grid;
    grid_temp = snapshot.grid_temp;
    persistent_vorticity = snapshot.persistent_vorticity;
    current_frame = snapshot.current_frame;
    accumulated_time = snapshot.accumulated_time;
    gpu_data_valid = false;
    if (cuda_initialized && settings.backend == SolverBackend::CUDA) {
        uploadToGPU();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU FORCE FIELD SUPPORT
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::uploadForceFieldsToGPU() {
    if (!cuda_initialized || !external_force_field_manager) return;
    
    const auto& fields = external_force_field_manager->getForceFields();
    
    // Count active fields that affect gas
    std::vector<FluidSim::GPUForceField> gpu_fields;
    gpu_fields.reserve(fields.size());
    
    for (const auto& field : fields) {
        if (!field || !field->enabled || !field->affects_gas) continue;
        
        FluidSim::GPUForceField gf = {};
        
        // Type mapping
        gf.type = static_cast<int>(field->type);
        gf.shape = static_cast<int>(field->shape);
        gf.falloff_type = static_cast<int>(field->falloff_type);
        gf.enabled = 1;
        
        // Transform (convert degrees to radians)
        gf.pos_x = field->position.x;
        gf.pos_y = field->position.y;
        gf.pos_z = field->position.z;
        gf.rot_x = field->rotation.x * 0.0174533f;
        gf.rot_y = field->rotation.y * 0.0174533f;
        gf.rot_z = field->rotation.z * 0.0174533f;
        gf.scale_x = field->scale.x;
        gf.scale_y = field->scale.y;
        gf.scale_z = field->scale.z;
        
        // Force parameters
        gf.strength = field->strength;
        gf.dir_x = field->direction.x;
        gf.dir_y = field->direction.y;
        gf.dir_z = field->direction.z;
        
        // Falloff
        gf.falloff_radius = field->falloff_radius;
        gf.inner_radius = field->inner_radius;
        
        // Vortex
        gf.axis_x = field->axis.x;
        gf.axis_y = field->axis.y;
        gf.axis_z = field->axis.z;
        gf.inward_force = field->inward_force;
        
        // Noise
        gf.noise_frequency = field->noise.frequency;
        gf.noise_amplitude = field->noise.amplitude;
        gf.noise_speed = field->noise.speed;
        gf.noise_octaves = field->noise.octaves;
        
        // Drag (use linear_drag as coefficient)
        gf.drag_coefficient = field->linear_drag;
        
        gpu_fields.push_back(gf);
    }
    
    // Free old buffer if count changed
    if (gpu_force_field_count != (int)gpu_fields.size()) {
        freeGPUForceFields();
    }
    
    if (gpu_fields.empty()) {
        gpu_force_field_count = 0;
        return;
    }
    
    // Upload to GPU
    if (!d_force_fields) {
        d_force_fields = FluidSim::cuda_upload_force_fields(gpu_fields.data(), (int)gpu_fields.size());
    } else {
        // Update existing buffer
        cudaMemcpy(d_force_fields, gpu_fields.data(), 
                   gpu_fields.size() * sizeof(FluidSim::GPUForceField), 
                   cudaMemcpyHostToDevice);
    }
    
    gpu_force_field_count = (int)gpu_fields.size();
}

void GasSimulator::freeGPUForceFields() {
    if (d_force_fields) {
        FluidSim::cuda_free_force_fields((FluidSim::GPUForceField*)d_force_fields);
        d_force_fields = nullptr;
    }
    gpu_force_field_count = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT PRESSURE SOLVER (10-50x faster than iterative for grids > 64³)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::initFFTSolver() {
    if (fft_solver) {
        cleanupFFTSolver();
    }
    
    fft_solver = new FFTPressureSolver();
    
    if (!initFFTPressureSolver(*fft_solver, grid.nx, grid.ny, grid.nz)) {
        SCENE_LOG_ERROR("[GasSimulator] Failed to initialize FFT Pressure Solver");
        delete fft_solver;
        fft_solver = nullptr;
        use_fft_solver = false;
        return;
    }
    
    use_fft_solver = true;
    SCENE_LOG_INFO("[GasSimulator] FFT Pressure Solver initialized - 10-50x faster!");
}

void GasSimulator::cleanupFFTSolver() {
    if (fft_solver) {
        cleanupFFTPressureSolver(*fft_solver);
        delete fft_solver;
        fft_solver = nullptr;
    }
    use_fft_solver = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU ADVANCED EMITTERS (Noise, Falloff, Spray Cone)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::uploadAdvancedEmittersToGPU() {
    if (emitters.empty()) {
        freeGPUAdvancedEmitters();
        return;
    }
    
    // Convert Emitter to GPUAdvancedEmitter
    std::vector<GPUAdvancedEmitter> gpu_emitters;
    gpu_emitters.reserve(emitters.size());
    
    for (const auto& e : emitters) {
        if (!e.enabled) continue;
        
        GPUAdvancedEmitter ge = {};
        
        // Shape & Transform
        ge.shape = mapEmitterShapeToGPU(e.shape);
        ge.enabled = e.enabled ? 1 : 0;
        ge.pos_x = e.position.x;
        ge.pos_y = e.position.y;
        ge.pos_z = e.position.z;
        ge.rot_x = 0.0f;  // Basic Emitter doesn't have rotation
        ge.rot_y = 0.0f;
        ge.rot_z = 0.0f;
        ge.scale_x = 1.0f;
        ge.scale_y = 1.0f;
        ge.scale_z = 1.0f;
        
        // Shape dimensions
        ge.radius = e.radius;
        ge.size_x = e.size.x;
        ge.size_y = e.size.y;
        ge.size_z = e.size.z;
        ge.height = e.height;
        ge.inner_radius = e.inner_radius;
        ge.cone_angle = degreesToRadians(e.cone_angle);
        
        // Emission
        ge.density_rate = e.density_rate;
        ge.temperature = e.temperature;
        ge.fuel_rate = e.fuel_rate;
        ge.fuel_phase = static_cast<int>(e.fuel_phase);
        ge.phase_change_temperature = e.phase_change_temperature;
        ge.fuel_release_rate = e.fuel_release_rate;
        ge.flame_contact_sensitivity = e.flame_contact_sensitivity;
        ge.vel_x = e.velocity.x;
        ge.vel_y = e.velocity.y;
        ge.vel_z = e.velocity.z;
        ge.velocity_magnitude = e.velocity.length();
        
        // Falloff
        ge.falloff_type = static_cast<int>(e.falloff_type);
        ge.falloff_start = e.falloff_start;
        ge.falloff_end = e.falloff_end;
        
        // Noise modulation
        ge.noise_enabled = e.noise_enabled ? 1 : 0;
        ge.noise_frequency = e.noise_frequency;
        ge.noise_amplitude = e.noise_amplitude;
        ge.noise_speed = e.noise_speed;
        ge.noise_seed = e.noise_seed;
        ge.noise_modulate_density = e.noise_modulate_density ? 1 : 0;
        ge.noise_modulate_temperature = e.noise_modulate_temperature ? 1 : 0;
        ge.noise_modulate_velocity = e.noise_modulate_velocity ? 1 : 0;
        
        // Velocity variance
        ge.spray_cone_angle = degreesToRadians(e.spray_cone_angle);
        ge.speed_min = e.speed_min;
        ge.speed_max = e.speed_max;
        
        // Emission profile
        ge.emission_mode = static_cast<int>(e.emission_mode);
        ge.start_frame = e.start_frame;
        ge.end_frame = e.end_frame;
        ge.pulse_interval = e.pulse_interval;
        ge.pulse_duration = e.pulse_duration;
        
        gpu_emitters.push_back(ge);
    }
    
    if (gpu_emitters.empty()) {
        freeGPUAdvancedEmitters();
        return;
    }
    
    // Free old buffer if count changed
    if (gpu_advanced_emitter_count != static_cast<int>(gpu_emitters.size())) {
        freeGPUAdvancedEmitters();
    }
    
    // Upload to GPU
    if (!d_advanced_emitters) {
        d_advanced_emitters = cuda_upload_advanced_emitters(gpu_emitters.data(), 
                                                            static_cast<int>(gpu_emitters.size()));
    } else {
        // Update existing buffer
        cudaMemcpy(d_advanced_emitters, gpu_emitters.data(),
                   gpu_emitters.size() * sizeof(GPUAdvancedEmitter),
                   cudaMemcpyHostToDevice);
    }
    
    gpu_advanced_emitter_count = static_cast<int>(gpu_emitters.size());
}

void GasSimulator::freeGPUAdvancedEmitters() {
    if (d_advanced_emitters) {
        cuda_free_advanced_emitters(d_advanced_emitters);
        d_advanced_emitters = nullptr;
    }
    gpu_advanced_emitter_count = 0;
}

} // namespace FluidSim

