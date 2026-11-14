#include "GasSimulator.h"
#include "ForceField.h"
#include "gas_kernels.cuh"
#include "gas_fft_solver.cuh"
#include "globals.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace FluidSim {
namespace {
constexpr float kPi = 3.14159265359f;
float degreesToRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

int mapEmitterShapeToGPU(EmitterShape shape) {
    switch (shape) {
    case EmitterShape::Point:
        return 0;
    case EmitterShape::Sphere:
        return 1;
    case EmitterShape::Box:
        return 2;
    case EmitterShape::Cylinder:
        return 3;
    case EmitterShape::Cone:
        return 4;
    case EmitterShape::Disc:
        return 5;
    default:
        return 1;
    }
}
} // namespace

void GasSimulator::initCUDA() {
    SCENE_LOG_INFO(
        "[GasSimulator::initCUDA] START - Resolution: " + std::to_string(settings.resolution_x) +
        "x" + std::to_string(settings.resolution_y) + "x" + std::to_string(settings.resolution_z));

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
        settings.resolution_x, settings.resolution_y, settings.resolution_z, (float**)&d_density,
        (float**)&d_temperature, (float**)&d_fuel, (float**)&d_vel_x, (float**)&d_vel_y,
        (float**)&d_vel_z, (float**)&d_pressure, (float**)&d_divergence, (float**)&d_vort_x,
        (float**)&d_vort_y, (float**)&d_vort_z);

    // Allocate temporary buffers for advection
    size_t sz = (size_t)settings.resolution_x * settings.resolution_y * settings.resolution_z *
                sizeof(float);
    cudaMalloc(&d_tmp1, sz);
    cudaMalloc(&d_tmp2, sz);
    cudaMalloc(&d_tmp3, sz);

    cudaError_t allocErr = cudaGetLastError();
    if (allocErr != cudaSuccess) {
        SCENE_LOG_ERROR("[GasSimulator::initCUDA] CUDA allocation error: " +
                        std::string(cudaGetErrorString(allocErr)));
        FluidSim::cuda_free_simulation((float*)d_density, (float*)d_temperature, (float*)d_fuel,
                                       (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                                       (float*)d_pressure, (float*)d_divergence, (float*)d_vort_x,
                                       (float*)d_vort_y, (float*)d_vort_z);
        if (d_tmp1)
            cudaFree(d_tmp1);
        if (d_tmp2)
            cudaFree(d_tmp2);
        if (d_tmp3)
            cudaFree(d_tmp3);
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

    SCENE_LOG_INFO("[GasSimulator::initCUDA] COMPLETE - Buffer size: " + std::to_string(sz) +
                   " bytes");
}

void GasSimulator::freeCUDA() {
    if (!cuda_initialized)
        return;
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

    FluidSim::cuda_free_simulation((float*)d_density, (float*)d_temperature, (float*)d_fuel,
                                   (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                                   (float*)d_pressure, (float*)d_divergence, (float*)d_vort_x,
                                   (float*)d_vort_y, (float*)d_vort_z);

    if (d_tmp1)
        cudaFree(d_tmp1);
    if (d_tmp2)
        cudaFree(d_tmp2);
    if (d_tmp3)
        cudaFree(d_tmp3);

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
    if (!cuda_initialized)
        return;

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
    std::vector<float> ambient_temp_data((size_t)grid.nx * grid.ny * grid.nz,
                                         settings.ambient_temperature);
    cudaMemcpy(d_temperature, ambient_temp_data.data(), sz, cudaMemcpyHostToDevice);

    gpu_data_valid = true;
}

void GasSimulator::stepCUDA(float dt, const Matrix4x4& world_matrix) {
    if (!cuda_initialized) {
        SCENE_LOG_WARN("[GasSimulator::stepCUDA] Called but cuda_initialized=false!");
        return;
    }

    if (settings.pressure_solver == GasSimulationSettings::PressureSolverMode::FFT &&
        canUseFFTPressureSolver()) {
        if (!use_fft_solver || !fft_solver || fft_solver->nx != grid.nx ||
            fft_solver->ny != grid.ny || fft_solver->nz != grid.nz) {
            initFFTSolver();
        }
    } else if (use_fft_solver) {
        cleanupFFTSolver();
    }

    // SAFETY CHECK: Resolution mismatch should be handled in step(), but double-check here
    // This is a fallback safety check in case auto-reinit in step() fails
    if (settings.resolution_x != grid.nx || settings.resolution_y != grid.ny ||
        settings.resolution_z != grid.nz) {
        SCENE_LOG_ERROR(
            "[GasSimulator::stepCUDA] Resolution mismatch still present after step()! "
            "Settings: " +
            std::to_string(settings.resolution_x) + "x" + std::to_string(settings.resolution_y) +
            "x" + std::to_string(settings.resolution_z) + ", Grid: " + std::to_string(grid.nx) +
            "x" + std::to_string(grid.ny) + "x" + std::to_string(grid.nz) +
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
                grid.nx, grid.ny, grid.nz, substep_dt, (float*)d_density, (float*)d_temperature,
                (float*)d_fuel, (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z,
                d_advanced_emitters, gpu_advanced_emitter_count, world_mat,
                static_cast<float>(current_frame) +
                    static_cast<float>(sub) / std::max(1, settings.substeps),
                substep_time);
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
                    grid.nx, grid.ny, grid.nz,
                    substep_dt * inv_voxel_size, // Scale dt to convert force to grid units
                    (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z, world_mat,
                    (FluidSim::GPUForceField*)d_force_fields, gpu_force_field_count, substep_time);
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
                params, fft_solver, (float*)d_density, (float*)d_temperature, (float*)d_fuel,
                (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z, (float*)d_pressure,
                (float*)d_divergence, (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z,
                (float*)d_tmp1, (float*)d_tmp2, (float*)d_tmp3);
        } else {
            FluidSim::cuda_step_simulation_v2(
                params, (float*)d_density, (float*)d_temperature, (float*)d_fuel, (float*)d_vel_x,
                (float*)d_vel_y, (float*)d_vel_z, (float*)d_pressure, (float*)d_divergence,
                (float*)d_vort_x, (float*)d_vort_y, (float*)d_vort_z, (float*)d_tmp1,
                (float*)d_tmp2, (float*)d_tmp3);
        }
    }

    // NOTE: Removed post-step cudaDeviceSynchronize - was causing UI freezes
    // CUDA kernels run asynchronously, errors will be caught on next operation that requires sync

    // Mark valid so next upload won't be redundant (though step updates it anyway)
    gpu_data_valid = true;
}

void GasSimulator::uploadToGPU() {
    if (!cuda_initialized)
        return;

    // CRITICAL: Use grid dimensions (actual allocated size), not settings (user requested)
    FluidSim::cuda_upload_data(
        grid.nx, grid.ny, grid.nz, grid.density.data(), grid.temperature.data(), grid.fuel.data(),
        grid.vel_x.data(), grid.vel_y.data(), grid.vel_z.data(), (float*)d_density,
        (float*)d_temperature, (float*)d_fuel, (float*)d_vel_x, (float*)d_vel_y, (float*)d_vel_z);

    gpu_data_valid = true;
}

void GasSimulator::downloadFromGPU() {
    if (!cuda_initialized)
        return;

    // CRITICAL: Use grid dimensions (actual allocated size), not settings (user requested)
    FluidSim::cuda_download_data(grid.nx, grid.ny, grid.nz, (float*)d_density,
                                 (float*)d_temperature, (float*)d_fuel, (float*)d_vel_x,
                                 (float*)d_vel_y, (float*)d_vel_z, grid.density.data(),
                                 grid.temperature.data(), grid.fuel.data(), grid.vel_x.data(),
                                 grid.vel_y.data(), grid.vel_z.data());
}

void GasSimulator::uploadForceFieldsToGPU() {
    if (!cuda_initialized || !external_force_field_manager)
        return;

    const auto& fields = external_force_field_manager->getForceFields();

    // Count active fields that affect gas
    std::vector<FluidSim::GPUForceField> gpu_fields;
    gpu_fields.reserve(fields.size());

    for (const auto& field : fields) {
        if (!field || !field->enabled || !field->affects_gas)
            continue;

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
        d_force_fields =
            FluidSim::cuda_upload_force_fields(gpu_fields.data(), (int)gpu_fields.size());
    } else {
        // Update existing buffer
        cudaMemcpy(d_force_fields, gpu_fields.data(),
                   gpu_fields.size() * sizeof(FluidSim::GPUForceField), cudaMemcpyHostToDevice);
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

void GasSimulator::uploadAdvancedEmittersToGPU() {
    if (emitters.empty()) {
        freeGPUAdvancedEmitters();
        return;
    }

    // Convert Emitter to GPUAdvancedEmitter
    std::vector<GPUAdvancedEmitter> gpu_emitters;
    gpu_emitters.reserve(emitters.size());

    for (const auto& e : emitters) {
        if (!e.enabled)
            continue;

        GPUAdvancedEmitter ge = {};

        // Shape & Transform
        ge.shape = mapEmitterShapeToGPU(e.shape);
        ge.enabled = e.enabled ? 1 : 0;
        ge.pos_x = e.position.x;
        ge.pos_y = e.position.y;
        ge.pos_z = e.position.z;
        ge.rot_x = 0.0f; // Basic Emitter doesn't have rotation
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
                   gpu_emitters.size() * sizeof(GPUAdvancedEmitter), cudaMemcpyHostToDevice);
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

bool GasSimulator::cudaBackendAvailable() const {
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

    return cuda_available;
}

void GasSimulator::bindCudaThread() {
    cudaSetDevice(0);
}
} // namespace FluidSim
