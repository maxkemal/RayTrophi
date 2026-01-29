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

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTER SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json Emitter::toJson() const {
    nlohmann::json j;
    j["shape"] = static_cast<int>(shape);
    j["position"] = {position.x, position.y, position.z};
    j["size"] = {size.x, size.y, size.z};
    j["radius"] = radius;
    j["density_rate"] = density_rate;
    j["fuel_rate"] = fuel_rate;
    j["temperature"] = temperature;
    j["velocity"] = {velocity.x, velocity.y, velocity.z};
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
    if (j.contains("density_rate")) density_rate = j["density_rate"];
    if (j.contains("fuel_rate")) fuel_rate = j["fuel_rate"];
    if (j.contains("temperature")) temperature = j["temperature"];
    if (j.contains("velocity")) {
        auto v = j["velocity"];
        velocity = Vec3(v[0], v[1], v[2]);
    }
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
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.major >= 5) {
            cuda_available = true;
        }
    } else {
        cudaGetLastError(); // Clear any error flag
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
    
    // CFL Adaptive Timestep: Calculate safe timestep based on max velocity
    float effective_dt = scaled_dt;
    if (settings.adaptive_timestep) {
        effective_dt = computeCFLTimestep(scaled_dt);
    }
    
    // Update active tiles for sparse processing (VDB-style optimization)
    grid.sparse_mode_enabled = settings.sparse_mode;
    grid.sparse_threshold = settings.sparse_threshold;
    if (settings.sparse_mode) {
        grid.updateActiveTiles(settings.ambient_temperature);
    }
    
    // Use CUDA if enabled and available
    if (settings.backend == SolverBackend::CUDA && cuda_initialized) {
        stepCUDA(effective_dt, world_matrix);
    } else {
        // CPU solver (OpenMP disabled to prevent deadlock after resolution change)
        float substep_dt = effective_dt / settings.substeps;
        
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
    
    auto end = std::chrono::high_resolution_clock::now();
    last_step_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    accumulated_time += dt;
    current_frame++;
    gpu_data_valid = false;
}

void GasSimulator::reset() {
    grid.clear();
    grid_temp.clear();
    std::fill(grid.temperature.begin(), grid.temperature.end(), settings.ambient_temperature);
    current_frame = 0;
    accumulated_time = 0.0f;
    
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
        if (!emitter.enabled) continue;
        
        // Calculate grid-space bounding box
        Vec3 min_pos, max_pos;
        if (emitter.shape == EmitterShape::Sphere || emitter.shape == EmitterShape::Point) {
            float r = (emitter.shape == EmitterShape::Point) ? grid.voxel_size : emitter.radius;
            min_pos = emitter.position - Vec3(r, r, r);
            max_pos = emitter.position + Vec3(r, r, r);
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
                    bool inside = false;
                    
                    if (emitter.shape == EmitterShape::Sphere) {
                        float dist = (cell_pos - emitter.position).length();
                        inside = (dist < emitter.radius);
                    } else if (emitter.shape == EmitterShape::Box) {
                        Vec3 d = cell_pos - emitter.position;
                        inside = (std::abs(d.x) <= emitter.size.x && 
                                  std::abs(d.y) <= emitter.size.y && 
                                  std::abs(d.z) <= emitter.size.z);
                    } else if (emitter.shape == EmitterShape::Point) {
                        float dist = (cell_pos - emitter.position).length();
                        inside = (dist < grid.voxel_size * 2.0f);
                    }
                    
                    if (inside) {
                        size_t idx = grid.cellIndex(i, j, k);
                        
                        // Inject with limits to prevent overflow
                        grid.density[idx] += emitter.density_rate * dt;
                        grid.density[idx] = std::min(grid.density[idx], settings.max_density);
                        
                        grid.fuel[idx] += emitter.fuel_rate * dt;
                        // Fuel limit: prevent infinite accumulation
                        grid.fuel[idx] = std::min(grid.fuel[idx], 100.0f);
                        
                        // Temperature: blend towards emitter temp, don't exceed max
                        float target_temp = std::min(emitter.temperature, settings.max_temperature * 0.9f);
                        if (grid.temperature[idx] < target_temp) {
                            // Gradual heating instead of instant set
                            grid.temperature[idx] = grid.temperature[idx] * 0.9f + target_temp * 0.1f;
                        }
                        
                        if (emitter.velocity.length() > 0.001f) {
                            if (i > 0) grid.vel_x[grid.velXIndex(i, j, k)] += emitter.velocity.x * dt;
                            if (j > 0) grid.vel_y[grid.velYIndex(i, j, k)] += emitter.velocity.y * dt;
                            if (k > 0) grid.vel_z[grid.velZIndex(i, j, k)] += emitter.velocity.z * dt;
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
                
                // Only apply turbulence where there's density (smoke)
                float density = grid.densityAt(i, j, k);
                if (density < 0.01f) continue;
                
                // Density-weighted strength - more turbulence in denser smoke
                float local_strength = strength * std::min(density, 1.0f);
                
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
    // Copy current velocity to temp
    grid_temp.vel_x = grid.vel_x;
    grid_temp.vel_y = grid.vel_y;
    grid_temp.vel_z = grid.vel_z;
    
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
                
                // Sample velocity at previous position
                grid.vel_x[grid.velXIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).x;
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
                grid.vel_y[grid.velYIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).y;
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
                grid.vel_z[grid.velZIndex(i, j, k)] = grid_temp.sampleVelocity(prev_pos).z;
            }
        }
    }
}

void GasSimulator::advectScalars(float dt) {
    // Copy to temp
    grid_temp.density = grid.density;
    grid_temp.temperature = grid.temperature;
    
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
                
                // FIX: Use trilinear interpolation for fuel and interaction fields
                // to avoid staircase artifacts and flicker.
                grid.fuel[idx] = grid_temp.sampleCellCentered(grid_temp.fuel, prev_pos);
                grid.interaction[idx] = grid_temp.sampleCellCentered(grid_temp.interaction, prev_pos);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRESSURE SOLVER (with SOR support for faster convergence)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::solvePressure() {
    float scale = 1.0f / grid.voxel_size;
    
    // Compute divergence of the velocity field
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float div = (grid.velXAt(i + 1, j, k) - grid.velXAt(i, j, k) +
                             grid.velYAt(i, j + 1, k) - grid.velYAt(i, j, k) +
                             grid.velZAt(i, j, k + 1) - grid.velZAt(i, j, k)) * scale;
                grid.divergence[grid.cellIndex(i, j, k)] = div;
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
        grid.temperature[i] = settings.ambient_temperature + (grid.temperature[i] - settings.ambient_temperature) * temp_factor;
    }
    
    for (int i = 0; i < (int)grid.fuel.size(); ++i) {
        grid.fuel[i] *= fuel_factor;
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
    
    #pragma omp parallel for
    for (long long i = 0; i < cell_count; ++i) {
        // Defensive check: reset NaNs if any
        if (!std::isfinite(grid.fuel[i])) grid.fuel[i] = 0.0f;
        if (!std::isfinite(grid.temperature[i])) grid.temperature[i] = settings.ambient_temperature;
        if (!std::isfinite(grid.density[i])) grid.density[i] = 0.0f;

        float f = grid.fuel[i];
        float t = grid.temperature[i];
        
        // Reset reaction each frame
        grid.interaction[i] = 0.0f;
        
        if (f > 0.001f && t > settings.ignition_temperature) {
            // Yanma gerçekleşir!
            float den = grid.density[i];
            float density_throttle = std::max(0.05f, 1.0f - (den / max_dens));
            
            float burned = f * settings.burn_rate * dt * density_throttle;
            if (burned > f) burned = f;
            
            // Soft throttle: strongly reduce burn rate as temperature approaches max
            float range = max_temp - settings.ignition_temperature;
            float temp_factor = (max_temp - t) / (range > 0 ? range : 1.0f);
            float temp_headroom = std::max(0.0f, std::min(1.0f, temp_factor));
            burned *= temp_headroom; 
            
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
    if (name == "Fire") {
        // ═══════════════════════════════════════════════════════════════════
        // FIRE PRESET - Houdini/EmberGen style realistic fire
        // Key: Strong buoyancy, fast fuel burn, moderate vorticity
        // ═══════════════════════════════════════════════════════════════════
        settings.time_scale = 1.0f;               // Normal speed
        settings.substeps = 2;                    // More substeps for stability
        
        // Dissipation - keep values high (close to 1.0) for persistence
        settings.density_dissipation = 0.995f;    // Smoke persists well
        settings.temperature_dissipation = 0.95f; // Heat cools faster to prevent runaway
        settings.velocity_dissipation = 0.98f;    // Velocity persists
        settings.fuel_dissipation = 0.92f;        // Fuel burns away faster
        
        // Buoyancy - STRONG upward force
        settings.buoyancy_density = -0.3f;        // Slight downward from density
        settings.buoyancy_temperature = 5.0f;     // Strong upward force from heat (was 8, too explosive)
        settings.ambient_temperature = 293.0f;    // Room temp (20°C)
        
        // Turbulence & Detail
        settings.vorticity_strength = 0.8f;       // Good flame detail
        settings.turbulence_strength = 0.3f;      // Add noise for flicker
        settings.turbulence_scale = 2.0f;
        
        // Combustion
        settings.ignition_temperature = 350.0f;   // Lower ignition for easy start
        settings.burn_rate = 2.0f;                // Moderate combustion (was 3)
        settings.heat_release = 100.0f;            // Increased for warmer fire
        settings.expansion_strength = 3.0f;       // Fire expands outward
        settings.smoke_generation = 0.8f;         // Moderate smoke from fire
        settings.soot_generation = 0.1f;
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack;
        settings.adaptive_timestep = true;
        settings.cfl_number = 0.5f;
        
        // Emitter - continuous flame source
        if (emitters.empty()) {
            // Create default emitter at grid center-bottom
            Emitter e;
            e.name = "Fire Emitter";
            e.shape = EmitterShape::Sphere;
            e.position = grid.origin + Vec3(grid.nx * grid.voxel_size * 0.5f, 
                                            grid.voxel_size * 2.0f,
                                            grid.nz * grid.voxel_size * 0.5f);
            e.radius = grid.voxel_size * 3.0f;
            emitters.push_back(e);
        }
        // Configure first emitter for fire (balanced values)
        emitters[0].fuel_rate = 30.0f;        
        emitters[0].density_rate = 3.0f;      
        emitters[0].temperature = 800.0f;     // Increased emitter temp
        emitters[0].velocity = Vec3(0, 3, 0); // Upward initial velocity (was 4)
        emitters[0].enabled = true;
    }
    else if (name == "Smoke") {
        // ═══════════════════════════════════════════════════════════════════
        // SMOKE PRESET - Dense, slow-rising smoke plume
        // Key: Moderate buoyancy, no combustion, slow dissipation
        // ═══════════════════════════════════════════════════════════════════
        settings.time_scale = 1.0f;
        settings.substeps = 1;
        
        // Dissipation - smoke lingers (values close to 1.0)
        settings.density_dissipation = 0.995f;    // Smoke stays much longer
        settings.temperature_dissipation = 0.98f; // Temperature persists for buoyancy
        settings.velocity_dissipation = 0.98f;    // Velocity persists
        settings.fuel_dissipation = 0.99f;
        
        // Buoyancy - warm smoke rises
        settings.buoyancy_density = -0.3f;        // Slight downward from mass
        settings.buoyancy_temperature = 6.0f;     // Strong thermal lift!
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
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack;
        settings.adaptive_timestep = true;
        
        // Emitter - continuous smoke source
        if (emitters.empty()) {
            Emitter e;
            e.name = "Smoke Emitter";
            e.shape = EmitterShape::Sphere;
            e.position = grid.origin + Vec3(grid.nx * grid.voxel_size * 0.5f, 
                                            grid.voxel_size * 2.0f,
                                            grid.nz * grid.voxel_size * 0.5f);
            e.radius = grid.voxel_size * 4.0f;
            emitters.push_back(e);
        }
        emitters[0].fuel_rate = 0.0f;         // No fuel
        emitters[0].density_rate = 40.0f;     // Heavy smoke injection
        emitters[0].temperature = 400.0f;     // Warm smoke
        emitters[0].velocity = Vec3(0, 2, 0); // Gentle upward push
        emitters[0].enabled = true;
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
        
        // Solver
        settings.pressure_solver = GasSimulationSettings::PressureSolverMode::SOR;
        settings.advection_mode = GasSimulationSettings::AdvectionMode::MacCormack; // High detail
        settings.adaptive_timestep = true;
        settings.cfl_number = 0.4f;               // More conservative for stability
        
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
        // Don't return - continue and mark as initialized, the buffers may still be valid
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

    float substep_dt = dt / settings.substeps;
    float inv_voxel_size = 1.0f / (settings.voxel_size > 0.0001f ? settings.voxel_size : 0.1f);
    
    for (int sub = 0; sub < settings.substeps; ++sub) {
        // 1. Apply Emitters
        for (const auto& emitter : emitters) {
             if (!emitter.enabled) continue;
             
             // Convert World Position -> Grid Index Space
             // IMPORTANT: Position must be relative to grid origin
             Vec3 pos_idx = (emitter.position - grid.origin) * inv_voxel_size;
             
             // Convert velocity from world space (m/s) to grid space (cells/s)
             // This is CRITICAL for advection to work correctly!
             Vec3 vel_grid = emitter.velocity * inv_voxel_size;
             
             // Size in voxels
             float radius_idx = emitter.radius * inv_voxel_size;
             Vec3 size_idx = emitter.size * inv_voxel_size;
             
             float sx = (emitter.shape == EmitterShape::Sphere) ? radius_idx : size_idx.x;
             
             // CRITICAL: Use grid dimensions (actual buffer size), not settings
             FluidSim::cuda_apply_emitter(
                grid.nx, grid.ny, grid.nz,
                substep_dt,
                (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                (int)emitter.shape,
                pos_idx.x, pos_idx.y, pos_idx.z,
                sx, size_idx.y, size_idx.z,
                emitter.density_rate, emitter.temperature, emitter.fuel_rate,
                vel_grid.x, vel_grid.y, vel_grid.z
             );
        }
        
        // 2. Apply External Force Fields (GPU)
        if (external_force_field_manager && external_force_field_manager->getActiveCount() > 0) {
            uploadForceFieldsToGPU();
            
            if (d_force_fields && gpu_force_field_count > 0) {
                // Convert world_matrix to column-major float array
                float world_mat[16];
                for (int c = 0; c < 4; ++c) {
                    for (int r = 0; r < 4; ++r) {
                        world_mat[c * 4 + r] = world_matrix.m[r][c];
                    }
                }
                
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
                    accumulated_time
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
        params.time = accumulated_time;
        
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
        
        // Call new unified simulation step
        FluidSim::cuda_step_simulation_v2(
            params,
            (float*)d_density, (float*)d_temperature, (float*)d_fuel,
            (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
            (float*)d_pressure, (float*)d_divergence,
            (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z,
            (float*)d_tmp1, (float*)d_tmp2, (float*)d_tmp3
        );
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
        ge.shape = static_cast<int>(e.shape);
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
        ge.height = e.size.y;  // Use Y size as height for cylinder/cone
        ge.inner_radius = 0.0f;
        ge.cone_angle = 0.5f;  // Default cone angle (radians)
        
        // Emission
        ge.density_rate = e.density_rate;
        ge.temperature = e.temperature;
        ge.fuel_rate = e.fuel_rate;
        ge.vel_x = e.velocity.x;
        ge.vel_y = e.velocity.y;
        ge.vel_z = e.velocity.z;
        ge.velocity_magnitude = e.velocity.length();
        
        // Falloff (default smooth falloff)
        ge.falloff_type = 2;  // Smooth
        ge.falloff_start = 0.7f;
        ge.falloff_end = 1.0f;
        
        // Noise modulation (disabled by default for basic emitters)
        ge.noise_enabled = 0;
        ge.noise_frequency = 1.0f;
        ge.noise_amplitude = 0.3f;
        ge.noise_speed = 0.5f;
        ge.noise_seed = 42;
        ge.noise_modulate_density = 1;
        ge.noise_modulate_temperature = 0;
        ge.noise_modulate_velocity = 0;
        
        // Velocity variance (default no spray)
        ge.spray_cone_angle = 0.0f;
        ge.speed_min = 1.0f;
        ge.speed_max = 1.0f;
        
        // Emission profile (continuous)
        ge.emission_mode = 0;  // Continuous
        ge.start_frame = 0.0f;
        ge.end_frame = -1.0f;  // Never ends
        ge.pulse_interval = 10.0f;
        ge.pulse_duration = 5.0f;
        
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

