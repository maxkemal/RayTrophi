/**
 * @file GasSimulator.cpp
 * @brief CPU implementation of gas/smoke simulation
 */

#include "GasSimulator.h"
#include "ForceField.h"
#include "gas_kernels.cuh"
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
    j["pressure_iterations"] = pressure_iterations;
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
    j["gravity"] = {gravity.x, gravity.y, gravity.z};
    j["wind"] = {wind.x, wind.y, wind.z};
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
    if (j.contains("pressure_iterations")) pressure_iterations = j["pressure_iterations"];
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
    if (j.contains("gravity")) {
        auto g = j["gravity"];
        gravity = Vec3(g[0], g[1], g[2]);
    }
    if (j.contains("wind")) {
        auto w = j["wind"];
        wind = Vec3(w[0], w[1], w[2]);
    }
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
    
    // Safety checks
    if (settings.resolution_x <= 0) settings.resolution_x = 32;
    if (settings.resolution_y <= 0) settings.resolution_y = 32;
    if (settings.resolution_z <= 0) settings.resolution_z = 32;
    
    // Calculate uniform voxel size based on the longest axis and its resolution
    // Industry Standard: voxel_size = domain_size / resolution
    float vx = settings.grid_size.x / (float)settings.resolution_x;
    float vy = settings.grid_size.y / (float)settings.resolution_y;
    float vz = settings.grid_size.z / (float)settings.resolution_z;
    
    // We use the smallest voxel size to ensure we fit within the requested grid_size boundaries
    // or we can allow non-uniform if we really want, but standard is UNIFORM.
    // Let's use VX as the master and adjust others to be square.
    settings.voxel_size = vx; 
    if (settings.voxel_size < 0.0001f) settings.voxel_size = 0.1f;
    
    // Create grids
    grid.resize(settings.resolution_x, settings.resolution_y, settings.resolution_z, settings.voxel_size, settings.grid_offset);
    grid_temp.resize(settings.resolution_x, settings.resolution_y, settings.resolution_z, settings.voxel_size, settings.grid_offset);
    
    // Initialize persistent buffers
    persistent_vorticity.assign(grid.getCellCount(), Vec3(0, 0, 0));
    
    // Initialize with ambient temperature
    std::fill(grid.temperature.begin(), grid.temperature.end(), settings.ambient_temperature);
    
    // Initialize CUDA if needed
    if (settings.backend == SolverBackend::CUDA) {
        initCUDA();
    }
    
    current_frame = 0;
    accumulated_time = 0.0f;
    initialized = true;
}

void GasSimulator::step(float dt, const Matrix4x4& world_matrix) {
    if (!initialized) return;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Use CUDA if enabled and available
    if (settings.backend == SolverBackend::CUDA && cuda_initialized) {
        stepCUDA(dt, world_matrix);
    } else {
        // CPU solver
        float substep_dt = dt / settings.substeps;
        
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
                        // No need for atomic if this emitter's loop is parallelized internally
                        grid.density[idx] += emitter.density_rate * dt;
                        grid.fuel[idx] += emitter.fuel_rate * dt;
                        if (grid.temperature[idx] < emitter.temperature) grid.temperature[idx] = emitter.temperature;
                        
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
    applyBuoyancy(dt);
    applyVorticity(dt);
    applyWind(dt);
    applyExternalForceFields(dt, world_matrix); // Now passes the world matrix
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

// ═══════════════════════════════════════════════════════════════════════════════
// EXTERNAL FORCE FIELDS (Scene-level)
// ═══════════════════════════════════════════════════════════════════════════════

void GasSimulator::applyExternalForceFields(float dt, const Matrix4x4& world_matrix) {
    if (!external_force_field_manager) return;
    
    float time = accumulated_time;
    Matrix4x4 inv_world = world_matrix.inverse();
    
    // Apply force fields to each staggered face component to avoid data races and ensure accuracy
    
    // X-velocity components (nx+1, ny, nz)
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i <= grid.nx; ++i) {
                // NORMALIZED POSITION [0, 1]: Matrix handles scaling/translation
                Vec3 face_pos_norm(i / (float)grid.nx, (j + 0.5f) / (float)grid.ny, (k + 0.5f) / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                // For world velocity evaluation, we need to sample velocity at the physical location
                Vec3 local_pos_physical = grid.origin + Vec3(i * grid.voxel_size, (j + 0.5f) * grid.voxel_size, (k + 0.5f) * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                Vec3 world_vel = world_matrix.transform_vector(local_vel);
                
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, world_vel, true, false, false, false);
                Vec3 local_force = inv_world.transform_vector(world_force);
                
                grid.vel_x[grid.velXIndex(i, j, k)] += local_force.x * dt;
            }
        }
    }

    // Y-velocity components (nx, ny+1, nz)
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j <= grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 face_pos_norm((i + 0.5f) / (float)grid.nx, j / (float)grid.ny, (k + 0.5f) / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                Vec3 local_pos_physical = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, j * grid.voxel_size, (k + 0.5f) * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                Vec3 world_vel = world_matrix.transform_vector(local_vel);
                
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, world_vel, true, false, false, false);
                Vec3 local_force = inv_world.transform_vector(world_force);
                
                grid.vel_y[grid.velYIndex(i, j, k)] += local_force.y * dt;
            }
        }
    }

    // Z-velocity components (nx, ny, nz+1)
    #pragma omp parallel for collapse(2)
    for (int k = 0; k <= grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                Vec3 face_pos_norm((i + 0.5f) / (float)grid.nx, (j + 0.5f) / (float)grid.ny, k / (float)grid.nz);
                Vec3 world_pos = world_matrix.transform_point(face_pos_norm);
                
                Vec3 local_pos_physical = grid.origin + Vec3((i + 0.5f) * grid.voxel_size, (j + 0.5f) * grid.voxel_size, k * grid.voxel_size);
                Vec3 local_vel = grid.sampleVelocity(local_pos_physical);
                Vec3 world_vel = world_matrix.transform_vector(local_vel);
                
                Vec3 world_force = external_force_field_manager->evaluateAtFiltered(world_pos, time, world_vel, true, false, false, false);
                Vec3 local_force = inv_world.transform_vector(world_force);
                
                grid.vel_z[grid.velZIndex(i, j, k)] += local_force.z * dt;
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
// PRESSURE SOLVER
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
    
    // WARM START: We don't reset pressure to 0 every frame anymore.
    // This significantly improves temporal stability (prevents flicker).
    // std::fill(grid.pressure.begin(), grid.pressure.end(), 0.0f);
    
    // Red-Black Gauss-Seidel for thread-safe parallel convergence
    for (int iter = 0; iter < settings.pressure_iterations; ++iter) {
        // Red cells update
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < grid.nz; ++k) {
            for (int j = 0; j < grid.ny; ++j) {
                for (int i = 0; i < grid.nx; ++i) {
                    if ((i + j + k) % 2 == 0) {
                        // OPEN BOUNDARY CONDITION: Surroundings outside return 0 pressure
                        // This allows fluid to flow out of the domain (outflow).
                        float p_sum = (i + 1 < grid.nx ? grid.pressureAt(i + 1, j, k) : 0.0f) +
                                      (i - 1 >= 0 ? grid.pressureAt(i - 1, j, k) : 0.0f) +
                                      (j + 1 < grid.ny ? grid.pressureAt(i, j + 1, k) : 0.0f) +
                                      (j - 1 >= 0 ? grid.pressureAt(i, j - 1, k) : 0.0f) +
                                      (k + 1 < grid.nz ? grid.pressureAt(i, j, k + 1) : 0.0f) +
                                      (k - 1 >= 0 ? grid.pressureAt(i, j, k - 1) : 0.0f);
                        
                        float div = grid.divergence[grid.cellIndex(i, j, k)];
                        grid.pressure[grid.cellIndex(i, j, k)] = (p_sum - div * h2) / 6.0f;
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
                         // OPEN BOUNDARY CONDITION: Surroundings outside return 0 pressure
                        float p_sum = (i + 1 < grid.nx ? grid.pressureAt(i + 1, j, k) : 0.0f) +
                                      (i - 1 >= 0 ? grid.pressureAt(i - 1, j, k) : 0.0f) +
                                      (j + 1 < grid.ny ? grid.pressureAt(i, j + 1, k) : 0.0f) +
                                      (j - 1 >= 0 ? grid.pressureAt(i, j - 1, k) : 0.0f) +
                                      (k + 1 < grid.nz ? grid.pressureAt(i, j, k + 1) : 0.0f) +
                                      (k - 1 >= 0 ? grid.pressureAt(i, j, k - 1) : 0.0f);
                        
                        float div = grid.divergence[grid.cellIndex(i, j, k)];
                        grid.pressure[grid.cellIndex(i, j, k)] = (p_sum - div * h2) / 6.0f;
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
    
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < (int)grid.density.size(); ++i) {
            grid.density[i] *= density_factor;
            if (grid.density[i] < 0.001f) grid.density[i] = 0.0f;
        }
        
        #pragma omp for
        for (int i = 0; i < (int)grid.temperature.size(); ++i) {
            grid.temperature[i] = settings.ambient_temperature + (grid.temperature[i] - settings.ambient_temperature) * temp_factor;
        }
        
        #pragma omp for
        for (int i = 0; i < (int)grid.fuel.size(); ++i) {
            grid.fuel[i] *= fuel_factor;
            if (grid.fuel[i] < 0.001f) grid.fuel[i] = 0.0f;
        }

        #pragma omp for
        for (int i = 0; i < (int)grid.interaction.size(); ++i) {
            grid.interaction[i] *= 0.5f;
            if (grid.interaction[i] < 0.001f) grid.interaction[i] = 0.0f;
        }
        
        #pragma omp for
        for (int i = 0; i < (int)grid.vel_x.size(); ++i) grid.vel_x[i] *= vel_factor;
        #pragma omp for
        for (int i = 0; i < (int)grid.vel_y.size(); ++i) grid.vel_y[i] *= vel_factor;
        #pragma omp for
        for (int i = 0; i < (int)grid.vel_z.size(); ++i) grid.vel_z[i] *= vel_factor;
    }
}

void GasSimulator::processCombustion(float dt) {
    // Yanma işlemi: Yakıt + Yüksek Sıcaklık -> Alev + Isı + Duman
    if (settings.burn_rate <= 0.0f) {
        std::fill(grid.interaction.begin(), grid.interaction.end(), 0.0f);
        return;
    }

    long long cell_count = (long long)grid.getCellCount(); 
    
    #pragma omp parallel for
    for (long long i = 0; i < cell_count; ++i) {
        float f = grid.fuel[i];
        float t = grid.temperature[i];
        
        // Reset reaction each frame
        grid.interaction[i] = 0.0f;
        
        if (f > 0.001f && t > settings.ignition_temperature) {
            // Yanma gerçekleşir!
            float burned = f * settings.burn_rate * dt;
            if (burned > f) burned = f;
            
            // Yakıtı tüket
            grid.fuel[i] -= burned;
            
            // Isı yay
            grid.temperature[i] += burned * settings.heat_release;
            
            // Duman üret
            grid.density[i] += burned * settings.smoke_generation;
            
            // Görsel alev yoğunluğu
            grid.interaction[i] = burned / dt;
            
            // Genişleme (patlama etkisi benzeri)
            if (settings.expansion_strength > 0) {
                grid.pressure[i] += burned * settings.expansion_strength;
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
        settings.density_dissipation = 0.98f;
        settings.temperature_dissipation = 0.95f;
        settings.fuel_dissipation = 0.99f;
        settings.buoyancy_density = -0.1f;
        settings.buoyancy_temperature = 0.8f;
        settings.vorticity_strength = 0.4f;
        settings.burn_rate = 1.2f;
        settings.heat_release = 250.0f;
        settings.smoke_generation = 0.3f;
        settings.ignition_temperature = 350.0f;
        
        if (!emitters.empty()) {
            emitters[0].fuel_rate = 50.0f;
            emitters[0].density_rate = 2.0f;
            emitters[0].temperature = 800.0f;
        }
    }
    else if (name == "Smoke") {
        settings.density_dissipation = 0.995f;
        settings.temperature_dissipation = 0.99f;
        settings.burn_rate = 0.0f; // No fire
        settings.buoyancy_density = -0.5f;
        settings.buoyancy_temperature = 0.2f;
        
        if (!emitters.empty()) {
            emitters[0].fuel_rate = 0.0f;
            emitters[0].density_rate = 20.0f;
            emitters[0].temperature = 350.0f;
        }
    }
    else if (name == "Explosion") {
        settings.density_dissipation = 0.94f;     // Smoke fades slightly faster
        settings.temperature_dissipation = 0.85f; // Fire cooling
        settings.burn_rate = 8.0f;                // Extremely fast combustion
        settings.heat_release = 2500.0f;          // Massive heat release
        settings.expansion_strength = 60.0f;     // Strong outward blast
        settings.smoke_generation = 1.5f;
        settings.vorticity_strength = 2.5f;      // Much more detail/swirls
        
        // Clear grid and set initial 'fireball'
        grid.clear();
        
        int cx = settings.resolution_x / 2;
        int cy = settings.resolution_y / 4; // Start near bottom
        int cz = settings.resolution_z / 2;
        int radius = std::max(2, settings.resolution_x / 16);
        
        for (int k = -radius; k <= radius; ++k) {
            for (int j = -radius; j <= radius; ++j) {
                for (int i = -radius; i <= radius; ++i) {
                    int x = cx + i, y = cy + j, z = cz + k;
                    if (x >= 0 && x < grid.nx && y >= 0 && y < grid.ny && z >= 0 && z < grid.nz) {
                        float dist = sqrtf((float)(i*i + j*j + k*k));
                        if (dist <= radius) {
                            size_t idx = grid.cellIndex(x, y, z);
                            float factor = 1.0f - (dist / (float)radius);
                            grid.fuel[idx] = 100.0f * factor;
                            grid.temperature[idx] = 3000.0f * factor;
                            grid.density[idx] = 10.0f * factor;
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
    
    // Create temperature grid
    openvdb::FloatGrid::Ptr temp_grid = openvdb::FloatGrid::create();
    temp_grid->setName("temperature");
    openvdb::FloatGrid::Accessor temp_accessor = temp_grid->getAccessor();
    
    for (int k = 0; k < grid.nz; ++k) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
                float t = grid.temperature[grid.cellIndex(i, j, k)];
                if (t > settings.ambient_temperature + 0.1f) {
                    temp_accessor.setValue(openvdb::Coord(i, j, k), t);
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
    
    // Write file
    openvdb::GridPtrVec grids;
    grids.push_back(density_grid);
    grids.push_back(temp_grid);
    
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
    if (cuda_initialized) freeCUDA();

    FluidSim::cuda_init_simulation(
        settings.resolution_x, settings.resolution_y, settings.resolution_z,
        (float**)&d_density, (float**)&d_temperature, (float**)&d_fuel,
        (float**)&d_vel_x, (float**)&d_vel_y, (float**)&d_vel_z,
        (float**)&d_pressure, (float**)&d_divergence,
        (float**)&d_vort_x, (float**)&d_vort_y, (float**)&d_vort_z
    );
    
    // Initial upload of CPU state
    uploadToGPU();
    
    cuda_initialized = true;
    gpu_data_valid = true;
}

void GasSimulator::freeCUDA() {
    if (!cuda_initialized) return;
    
    FluidSim::cuda_free_simulation(
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
        (float*)d_pressure, (float*)d_divergence,
        (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z
    );
    
    d_density = d_temperature = d_fuel = nullptr;
    d_vel_x = d_vel_y = d_vel_z = nullptr;
    d_pressure = d_divergence = nullptr;
    d_vort_x = d_vort_y = d_vort_z = nullptr;
    
    cuda_initialized = false;
    gpu_data_valid = false;
}

void GasSimulator::stepCUDA(float dt, const Matrix4x4& world_matrix) {
    if (!cuda_initialized) return;

    float substep_dt = dt / settings.substeps;
    float inv_voxel_size = 1.0f / (settings.voxel_size > 0.0001f ? settings.voxel_size : 0.1f);
    
    for (int sub = 0; sub < settings.substeps; ++sub) {
        // 1. Apply Emitters
        for (const auto& emitter : emitters) {
             if (!emitter.enabled) continue;
             
             // Convert World Position -> Grid Index Space
             // Assumes grid.origin is (0,0,0) relative to the domain transform if handled externally,
             // or if we handle world space here. 
             // GasSimulator usually treats 'grid_offset' as origin.
             Vec3 pos_idx = (emitter.position - settings.grid_offset) * inv_voxel_size;
             Vec3 vel_idx = emitter.velocity * inv_voxel_size;
             
             // Size in voxels
             float radius_idx = emitter.radius * inv_voxel_size;
             Vec3 size_idx = emitter.size * inv_voxel_size;
             
             float sx = (emitter.shape == EmitterShape::Sphere) ? radius_idx : size_idx.x;
             
             FluidSim::cuda_apply_emitter(
                settings.resolution_x, settings.resolution_y, settings.resolution_z,
                substep_dt,
                (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                (int)emitter.shape,
                pos_idx.x, pos_idx.y, pos_idx.z,
                sx, size_idx.y, size_idx.z,
                emitter.density_rate, emitter.temperature, emitter.fuel_rate,
                vel_idx.x, vel_idx.y, vel_idx.z
             );
        }
        
        // 2. Step Simulation
        // Wind conversion
        Vec3 wind_idx = settings.wind * inv_voxel_size;
        
        FluidSim::cuda_step_simulation(
             settings.resolution_x, settings.resolution_y, settings.resolution_z,
             substep_dt,
             (float*)d_density, (float*)d_temperature, (float*)d_fuel,
             (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
             (float*)d_pressure, (float*)d_divergence,
             (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z,
             
             settings.vorticity_strength,
             settings.buoyancy_density, settings.buoyancy_temperature,
             settings.ambient_temperature,
             settings.density_dissipation, settings.velocity_dissipation,
             settings.temperature_dissipation, settings.fuel_dissipation,
             settings.ignition_temperature, settings.burn_rate, settings.heat_release,
             settings.smoke_generation, settings.expansion_strength,
             settings.pressure_iterations,
             wind_idx.x, wind_idx.y, wind_idx.z
        );
    }
    
    // Mark valid so next upload won't be redundant (though step updates it anyway)
    gpu_data_valid = true;
}

void GasSimulator::uploadToGPU() {
    if (!cuda_initialized) return;
    
    FluidSim::cuda_upload_data(
        settings.resolution_x, settings.resolution_y, settings.resolution_z,
        grid.density.data(), grid.temperature.data(), grid.fuel.data(),
        grid.vel_x.data(), grid.vel_y.data(), grid.vel_z.data(),
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z
    );
    
    gpu_data_valid = true;
}

void GasSimulator::downloadFromGPU() {
    if (!cuda_initialized) return;
    
    FluidSim::cuda_download_data(
        settings.resolution_x, settings.resolution_y, settings.resolution_z,
        (float*)d_density, (float*)d_temperature, (float*)d_fuel,
        (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
        grid.density.data(), grid.temperature.data(), grid.fuel.data(),
        grid.vel_x.data(), grid.vel_y.data(), grid.vel_z.data()
    );
}

} // namespace FluidSim
