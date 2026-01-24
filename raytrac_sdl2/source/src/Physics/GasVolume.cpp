#include "GasVolume.h"
#include "Ray.h"
#include "VDBVolumeManager.h"
#include <cmath>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════════

GasVolume::GasVolume() {
    transform = std::make_shared<Transform>();
    
    // Default settings
    settings.resolution_x = 64;
    settings.resolution_y = 64;
    settings.resolution_z = 64;
    settings.grid_size = Vec3(5, 5, 5); // 5 meter cube default
    settings.voxel_size = 5.0f / 64.0f;
}

GasVolume::GasVolume(const std::string& n) : GasVolume() {
    name = n;
}

GasVolume::~GasVolume() {
    freeGPUResources();
    simulator.shutdown();
}

// ═══════════════════════════════════════════════════════════════════════════════
// HITTABLE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

bool GasVolume::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    if (!visible || ignore_volumes) return false;
    // AABB intersection for ray-volume test
    Vec3 bounds_min, bounds_max;
    getWorldBounds(bounds_min, bounds_max);
    
    float tmin = t_min;
    float tmax = t_max;
    
    for (int d = 0; d < 3; ++d) {
        float lo = (bounds_min[d] - r.origin[d]) / r.direction[d];
        float hi = (bounds_max[d] - r.origin[d]) / r.direction[d];
        
        if (lo > hi) std::swap(lo, hi);
        
        tmin = std::max(tmin, lo);
        tmax = std::min(tmax, hi);
        
        if (tmin > tmax) return false;
    }
    
    if (tmin < t_min) tmin = t_min;
    if (tmin > t_max) return false;
    
    rec.t = tmin;
    rec.point = r.at(tmin);
    rec.set_face_normal(r, Vec3(0, 1, 0)); // Arbitrary normal
    rec.material = nullptr; // Will use VolumeShader separately
    rec.gas_volume = this; // Set pointer to this GasVolume
    
    return true;
}

bool GasVolume::bounding_box(float time0, float time1, AABB& output_box) const {
    Vec3 min_bound, max_bound;
    getWorldBounds(min_bound, max_bound);
    output_box = AABB(min_bound, max_bound);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSFORM
// ═══════════════════════════════════════════════════════════════════════════════

void GasVolume::setPosition(const Vec3& pos) {
    position = pos;
    if (transform) {
        transform->position = pos;
        transform->updateMatrix();
    }
    bounds_dirty = true;
    applyTransform();
}

void GasVolume::setRotation(const Vec3& rot) {
    rotation = rot;
    if (transform) {
        transform->rotation = rot;
        transform->updateMatrix();
    }
    bounds_dirty = true;
    applyTransform();
}

void GasVolume::setScale(const Vec3& s) {
    scale = s;
    if (transform) {
        transform->scale = s;
        transform->updateMatrix();
    }
    
    // Sync physical domain size with object scale
    settings.grid_size = s;
    
    // Update voxel_size metrics based on new scale
    if (settings.resolution_x > 0) {
        settings.voxel_size = settings.grid_size.x / (float)settings.resolution_x;
        if (initialized) {
            simulator.getGrid().voxel_size = settings.voxel_size;
        }
    }
    
    bounds_dirty = true;
    applyTransform();
    
    if (!is_playing && initialized) {
        initialize();
    }
}

void GasVolume::applyTransform() {
    // Note: Since we use Transform matrix for raytracing, 
    // the simulation internal grid should remain at local (0,0,0).
    // The world position is handled by the object transform matrix.
    settings.grid_offset = Vec3(0, 0, 0); 
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION CONTROL
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// GPU RENDERING
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>

void GasVolume::uploadToGPU(void* stream_ptr) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    if (!initialized) return;
    
    const auto& grid = simulator.getGrid();
    int width = grid.nx;
    int height = grid.ny;
    int depth = grid.nz;

    // Detect resolution change and free old textures
    if (width != gpu_res_x || height != gpu_res_y || depth != gpu_res_z) {
        freeGPUResources();
        gpu_res_x = width;
        gpu_res_y = height;
        gpu_res_z = depth;
    }
    
    auto uploadTexture = [&](void* d_sim_ptr, const float* h_ptr, void** d_array, unsigned long long& tex_obj) {
        // 1. Allocate CUDA Array if needed
        if (!*d_array) {
            cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
            cudaExtent extent = make_cudaExtent(width, height, depth);
            cudaError_t err = cudaMalloc3DArray((cudaArray**)d_array, &channelDesc, extent);
            if (err != cudaSuccess) {
                printf("GasVolume::uploadToGPU Error: Failed to allocate 3D array (%s)\n", cudaGetErrorString(err));
                return;
            }
        }
        
        // 2. Copy Data (CPU/GPU -> Array) (Using Stream for Flicker-Free)
        cudaMemcpy3DParms copyParams = {0};
        if (d_sim_ptr) {
            copyParams.srcPtr = make_cudaPitchedPtr(d_sim_ptr, width * sizeof(float), width, height);
            copyParams.kind = cudaMemcpyDeviceToDevice;
        } else {
            copyParams.srcPtr = make_cudaPitchedPtr((void*)h_ptr, width * sizeof(float), width, height);
            copyParams.kind = cudaMemcpyHostToDevice;
        }
        
        copyParams.dstArray = (cudaArray*)*d_array;
        copyParams.extent = make_cudaExtent(width, height, depth);
        
        cudaError_t copyErr;
        if (stream) {
            copyErr = cudaMemcpy3DAsync(&copyParams, stream);
        } else {
            copyErr = cudaMemcpy3D(&copyParams);
        }
        
        if (copyErr != cudaSuccess) {
            printf("GasVolume::uploadToGPU Error: Failed to copy data (%s)\n", cudaGetErrorString(copyErr));
        }
        
        // 3. Create Texture Object (if not exists)
        if (tex_obj == 0) {
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = (cudaArray*)*d_array;
            
            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp; 
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.addressMode[2] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;     
            texDesc.readMode = cudaReadModeElementType;   
            texDesc.normalizedCoords = 1;                  
            
            cudaCreateTextureObject((cudaTextureObject_t*)&tex_obj, &resDesc, &texDesc, nullptr);
        }
    };

    // Upload Density
    uploadTexture(simulator.getGPUDensityPtr(), grid.density.data(), &density_array, density_texture);
    
    // Upload Temperature (if blackbody is likely used)
    if (shader && shader->emission.mode == VolumeEmissionMode::Blackbody) {
        uploadTexture(simulator.getGPUTemperaturePtr(), grid.temperature.data(), &temperature_array, temperature_texture);
    }

    if (stream) {
        cudaStreamSynchronize(stream);
    }
    
    // Upload Velocity (if motion blur is likely used)
    // Note: Simulator provides 3 separate grids for velocity (d_vel_x, etc.)
    // For now we only support density and temperature. 
    // Velocity would need a float4 texture (Vec3 + padding).

    // -------------------------------------------------------------------------
    // VDB UNIFIED PIPELINE SYNC
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // VDB UNIFIED PIPELINE SYNC
    // -------------------------------------------------------------------------
    if (render_path == VolumeRenderPath::VDBUnified) {
        
        // Critical: If running on CUDA backend, the simulation data resides on GPU (d_density).
        // However, VDBVolumeManager current implementation expects CPU data (grid.density) 
        // to convert it to NanoVDB. We MUST download it here.
        // NOTE: Ideally VDBVolumeManager should accept GPU pointers to avoid this round-trip,
        // but for now this fixes the "No Quality Change" issue.
        if (settings.backend == FluidSim::SolverBackend::CUDA) {
             simulator.downloadFromGPU();
        }

        // Sync simulation arrays to VDB Volume Manager
        live_vdb_id = VDBVolumeManager::getInstance().registerOrUpdateLiveVolume(
            live_vdb_id,
            name + " [VDB Path]",
            grid.nx, grid.ny, grid.nz, 
            grid.voxel_size,
            grid.density.data(),
            (shader && shader->emission.mode == VolumeEmissionMode::Blackbody) ? grid.temperature.data() : nullptr,
            stream
        );
        
        // Also ensure individual volume data has the correct transform
        auto* vdb_data = VDBVolumeManager::getInstance().getVolume(live_vdb_id);
        if (vdb_data && transform) {
            vdb_data->transform = transform;
        }
    }
}

void GasVolume::freeGPUResources() {
    auto freeResources = [&](unsigned long long& tex_obj, void** d_array) {
        if (tex_obj != 0) {
            cudaDestroyTextureObject((cudaTextureObject_t)tex_obj);
            tex_obj = 0;
        }
        if (*d_array) {
            cudaFreeArray((cudaArray*)*d_array);
            *d_array = nullptr;
        }
    };

    freeResources(density_texture, &density_array);
    freeResources(temperature_texture, &temperature_array);
    freeResources(velocity_texture, &velocity_array);
}

void GasVolume::initialize() {
    // Ensure our transform scale matches the physical grid_size from settings
    scale = settings.grid_size;
    if (transform) {
        transform->scale = scale;
        transform->position = position;
        transform->rotation = rotation;
    }
    
    settings.grid_offset = Vec3(0, 0, 0); 
    simulator.initialize(settings);
    initialized = true;
    bounds_dirty = true;
    
    // Initial GPU upload
    uploadToGPU();
}

void GasVolume::stop() {
    is_playing = false;
    // Don't free GPU resources on stop, only on destruction
}

void GasVolume::stepFrame(float dt, void* stream) {
    if (initialized) {
        simulator.step(dt, transform ? transform->base : Matrix4x4::identity());
        uploadToGPU(stream); // Upload new frame data to GPU
    }
}

void GasVolume::update(float dt, void* stream) {
    // -------------------------------------------------------------------------
    // TRANSFORM SYNC (Gizmo & Manual movement)
    // -------------------------------------------------------------------------
    if (transform) {
        // Force update if transform changed externally (e.g. via Gizmo)
        if (transform->isDirty() || 
            position != transform->position || 
            rotation != transform->rotation || 
            scale != transform->scale) {
            
            position = transform->position;
            rotation = transform->rotation;
            
            // If scale changed, we must also update physical grid size
            if (scale != transform->scale) {
                scale = transform->scale;
                settings.grid_size = scale;
                
                // CRITICAL: Update voxel_size metrics so rendering follows scale
                if (settings.resolution_x > 0) {
                    settings.voxel_size = settings.grid_size.x / (float)settings.resolution_x;
                    if (initialized) {
                        simulator.getGrid().voxel_size = settings.voxel_size;
                    }
                }
            }
            
            bounds_dirty = true;
        }
    }

    if (is_playing && initialized && !simulator.isBaking()) {
        // SYNC LIVE SETTINGS: Push modified UI settings to internal simulator 
        // to avoid requiring a full simulation reset for non-resolution parameters.
        auto& sim_settings = simulator.getSettings();
        sim_settings.timestep = settings.timestep;
        sim_settings.substeps = settings.substeps;
        sim_settings.pressure_iterations = settings.pressure_iterations;
        sim_settings.density_dissipation = settings.density_dissipation;
        sim_settings.velocity_dissipation = settings.velocity_dissipation;
        sim_settings.temperature_dissipation = settings.temperature_dissipation;
        sim_settings.fuel_dissipation = settings.fuel_dissipation;
        sim_settings.buoyancy_density = settings.buoyancy_density;
        sim_settings.buoyancy_temperature = settings.buoyancy_temperature;
        sim_settings.ambient_temperature = settings.ambient_temperature;
        sim_settings.vorticity_strength = settings.vorticity_strength;
        sim_settings.wind = settings.wind;
        sim_settings.ignition_temperature = settings.ignition_temperature;
        sim_settings.burn_rate = settings.burn_rate;
        sim_settings.heat_release = settings.heat_release;
        sim_settings.expansion_strength = settings.expansion_strength;
        sim_settings.smoke_generation = settings.smoke_generation;
        
        simulator.step(dt, transform ? transform->base : Matrix4x4::identity());
        uploadToGPU(stream); // Upload new frame data to GPU
    }
    else if (!is_playing && bounds_dirty) {
        // Still upload to GPU if moved while paused to keep preview in sync
        uploadToGPU(stream);
    }
}

void GasVolume::reset() {
    if (initialized) {
        simulator.reset();
    }
}

void GasVolume::updateFromTimeline(int timeline_frame, void* stream) {
    if (!linked_to_timeline || !initialized) return;
    
    int target_frame = timeline_frame + frame_offset;
    if (target_frame < 0) target_frame = 0;
    
    // Apply keyframe animation to all emitters
    auto& emitters = simulator.getEmitters();
    for (auto& emitter : emitters) {
        // Check if emitter has keyframes
        if (!emitter.keyframes.empty()) {
            // Get interpolated keyframe for current frame
            auto kf = emitter.getInterpolatedKeyframe((float)timeline_frame);
            
            // Apply keyframe to emitter
            emitter.applyKeyframe(kf);
        }
    }
    
    if (settings.mode == FluidSim::SimulationMode::Baked) {
        if (target_frame != simulator.getCurrentFrame()) {
            simulator.loadBakedFrame(target_frame);
            uploadToGPU(stream);
        }
    } else {
        // Real-time sequential sim
        int current = simulator.getCurrentFrame();
        if (target_frame > current) {
            // Catch up if within reasonable range (e.g. 5 frames)
            // Otherwise it might hitch too much. Usually 1-2 frames.
            int delta = target_frame - current;
            if (delta > 0 && delta <= 10) {
                for (int i = 0; i < delta; ++i) {
                    simulator.step(settings.timestep, transform ? transform->base : Matrix4x4::identity());
                }
                uploadToGPU(stream);
            }
        } else if (target_frame < current && target_frame == 0) {
            // Auto-reset at frame 0
            simulator.reset();
            uploadToGPU(stream);
        }
    }
}

void GasVolume::setFrame(int frame) {
    // For baked simulations, load the specific frame
    if (settings.mode == FluidSim::SimulationMode::Baked) {
        simulator.loadBakedFrame(frame + frame_offset);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMITTERS
// ═══════════════════════════════════════════════════════════════════════════════

int GasVolume::addEmitter(const FluidSim::Emitter& emitter) {
    return simulator.addEmitter(emitter);
}

void GasVolume::removeEmitter(int index) {
    simulator.removeEmitter(index);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHADER
// ═══════════════════════════════════════════════════════════════════════════════

std::shared_ptr<VolumeShader> GasVolume::getOrCreateShader() {
    if (!shader) {
        shader = VolumeShader::createSmokePreset();
        shader->name = name + " Shader";
    }
    return shader;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════

float GasVolume::sampleDensity(const Vec3& world_pos) const {
    if (!initialized) return 0.0f;
    
    Vec3 local_pos = world_pos;
    if (transform) {
        transform->updateMatrix();
        Matrix4x4 inv_mat = transform->base.inverse();

        // Transform point (w=1) to Local Space (0..1 range because scale is in the matrix)
        float x = inv_mat.m[0][0] * world_pos.x + inv_mat.m[0][1] * world_pos.y + inv_mat.m[0][2] * world_pos.z + inv_mat.m[0][3];
        float y = inv_mat.m[1][0] * world_pos.x + inv_mat.m[1][1] * world_pos.y + inv_mat.m[1][2] * world_pos.z + inv_mat.m[1][3];
        float z = inv_mat.m[2][0] * world_pos.x + inv_mat.m[2][1] * world_pos.y + inv_mat.m[2][2] * world_pos.z + inv_mat.m[2][3];

        // Convert to physical simulation units (meters)
        local_pos = Vec3(x * settings.grid_size.x, y * settings.grid_size.y, z * settings.grid_size.z);
    }
    
    return simulator.sampleDensity(local_pos);
}

float GasVolume::sampleTemperature(const Vec3& world_pos) const {
    if (!initialized) return settings.ambient_temperature;
    
    Vec3 local_pos = world_pos;
    if (transform) {
        transform->updateMatrix();
        Matrix4x4 inv_mat = transform->base.inverse();

        float x = inv_mat.m[0][0] * world_pos.x + inv_mat.m[0][1] * world_pos.y + inv_mat.m[0][2] * world_pos.z + inv_mat.m[0][3];
        float y = inv_mat.m[1][0] * world_pos.x + inv_mat.m[1][1] * world_pos.y + inv_mat.m[1][2] * world_pos.z + inv_mat.m[1][3];
        float z = inv_mat.m[2][0] * world_pos.x + inv_mat.m[2][1] * world_pos.y + inv_mat.m[2][2] * world_pos.z + inv_mat.m[2][3];

        local_pos = Vec3(x * settings.grid_size.x, y * settings.grid_size.y, z * settings.grid_size.z);
    }
    return simulator.sampleTemperature(local_pos);
}

Vec3 GasVolume::sampleVelocity(const Vec3& world_pos) const {
    if (!initialized) return Vec3(0, 0, 0);
    
    Vec3 local_pos = world_pos;
    if (transform) {
        transform->updateMatrix();
        Matrix4x4 inv_mat = transform->base.inverse();

        float x = inv_mat.m[0][0] * world_pos.x + inv_mat.m[0][1] * world_pos.y + inv_mat.m[0][2] * world_pos.z + inv_mat.m[0][3];
        float y = inv_mat.m[1][0] * world_pos.x + inv_mat.m[1][1] * world_pos.y + inv_mat.m[1][2] * world_pos.z + inv_mat.m[1][3];
        float z = inv_mat.m[2][0] * world_pos.x + inv_mat.m[2][1] * world_pos.y + inv_mat.m[2][2] * world_pos.z + inv_mat.m[2][3];

        local_pos = Vec3(x * settings.grid_size.x, y * settings.grid_size.y, z * settings.grid_size.z);
    }
    return simulator.sampleVelocity(local_pos);
}

void GasVolume::getWorldBounds(Vec3& min_out, Vec3& max_out) const {
    if (bounds_dirty) {
        updateBounds();
    }
    min_out = cached_bounds_min;
    max_out = cached_bounds_max;
}

void GasVolume::updateBounds() const {
    // Local bounds: Unit box [0,1]^3
    // The scale in the transform matrix (synced with grid_size) defines the real world size.
    Vec3 local_min = Vec3(0, 0, 0);
    Vec3 local_max = Vec3(1, 1, 1);

    if (transform) {
        transform->updateMatrix();
        const Matrix4x4& m = transform->base;
        
        Vec3 corners[8] = {
            Vec3(0,0,0), Vec3(1,0,0), Vec3(0,1,0), Vec3(1,1,0),
            Vec3(0,0,1), Vec3(1,0,1), Vec3(0,1,1), Vec3(1,1,1)
        };
        
        Vec3 world_min(1e10f, 1e10f, 1e10f);
        Vec3 world_max(-1e10f, -1e10f, -1e10f);
        
        for(int i=0; i<8; ++i) {
            Vec3 p = corners[i];
            float tx = m.m[0][0]*p.x + m.m[0][1]*p.y + m.m[0][2]*p.z + m.m[0][3];
            float ty = m.m[1][0]*p.x + m.m[1][1]*p.y + m.m[1][2]*p.z + m.m[1][3];
            float tz = m.m[2][0]*p.x + m.m[2][1]*p.y + m.m[2][2]*p.z + m.m[2][3];
            
            world_min.x = std::min(world_min.x, tx);
            world_min.y = std::min(world_min.y, ty);
            world_min.z = std::min(world_min.z, tz);
            
            world_max.x = std::max(world_max.x, tx);
            world_max.y = std::max(world_max.y, ty);
            world_max.z = std::max(world_max.z, tz);
        }
        cached_bounds_min = world_min;
        cached_bounds_max = world_max;
    } else {
        cached_bounds_min = position;
        cached_bounds_max = position + scale;
    }
    
    bounds_dirty = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// BAKING
// ═══════════════════════════════════════════════════════════════════════════════

void GasVolume::startBake(int start_frame, int end_frame) {
    Matrix4x4 world_mat = transform ? transform->base : Matrix4x4::identity();
    simulator.startBake(start_frame, end_frame, settings.cache_directory, world_mat);
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════════════════

bool GasVolume::exportToVDB(const std::string& filepath) const {
    return simulator.exportToVDB(filepath);
}

bool GasVolume::exportSequence(const std::string& directory, int start_frame, int end_frame) {
    Matrix4x4 world_mat = transform ? transform->base : Matrix4x4::identity();
    return simulator.exportSequenceToVDB(directory, name, start_frame, end_frame, world_mat);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

nlohmann::json GasVolume::toJson() const {
    nlohmann::json j;
    
    j["name"] = name;
    j["id"] = id;
    j["visible"] = visible;
    
    // Transform
    j["position"] = {position.x, position.y, position.z};
    j["rotation"] = {rotation.x, rotation.y, rotation.z};
    j["scale"] = {scale.x, scale.y, scale.z};
    
    // Settings
    j["settings"] = settings.toJson();
    
    // Emitters
    nlohmann::json emitters_json = nlohmann::json::array();
    for (const auto& e : simulator.getEmitters()) {
        emitters_json.push_back(e.toJson());
    }
    j["emitters"] = emitters_json;
    
    // Shader
    if (shader) {
        j["shader"] = shader->toJson();
    }
    
    // Timeline
    j["linked_to_timeline"] = linked_to_timeline;
    j["frame_offset"] = frame_offset;
    j["is_playing"] = is_playing;
    
    j["render_path"] = static_cast<int>(render_path);
    
    return j;
}

void GasVolume::fromJson(const nlohmann::json& j) {
    if (j.contains("name")) name = j["name"];
    if (j.contains("id")) id = j["id"];
    if (j.contains("visible")) visible = j["visible"];
    
    // Transform
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

    if (j.contains("render_path")) {
        render_path = static_cast<VolumeRenderPath>(j["render_path"]);
    }
    
    // Settings
    if (j.contains("settings")) {
        settings.fromJson(j["settings"]);
    }
    
    // Initialize after loading settings
    initialize();
    
    // Emitters
    if (j.contains("emitters")) {
        uint32_t max_uid = 0;
        for (const auto& e_json : j["emitters"]) {
            FluidSim::Emitter e;
            e.fromJson(e_json);
            if (e.uid > max_uid) max_uid = e.uid;
            addEmitter(e);
        }
        simulator.emitter_id_counter = max_uid + 1;
    }
    
    // Shader
    if (j.contains("shader")) {
        shader = std::make_shared<VolumeShader>();
        shader->fromJson(j["shader"]);
    }
    
    // Timeline
    if (j.contains("linked_to_timeline")) linked_to_timeline = j["linked_to_timeline"];
    if (j.contains("frame_offset")) frame_offset = j["frame_offset"];
    if (j.contains("is_playing")) is_playing = j["is_playing"];
    
    // Update transform
    if (transform) {
        transform->position = position;
        transform->rotation = rotation;
        transform->scale = scale;
    }
    
    bounds_dirty = true;
}


