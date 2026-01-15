// Suppress C4146 warning globally for this file (OpenVDB/NanoVDB compatibility)
#pragma warning(disable: 4146)

// Fix for linker errors: Ensure Imath/OpenEXR are treated as DLLs
#define IMATH_DLL
#define OPENEXR_DLL

#include "VDBVolumeManager.h"
#include "Transform.h"
#include "globals.h"

// OpenVDB includes
// Explicitly include openvdb first
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Interpolation.h>

#include <nanovdb/NanoVDB.h>
// Enable OpenVDB support in NanoVDB tools
#define NANOVDB_USE_OPENVDB 
// Use standard CreateNanoGrid (compatible with latest lib)
#include <nanovdb/tools/CreateNanoGrid.h> 

// CUDA
#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>

// ============================================================================
// Internal Helpers
// ============================================================================
// (No adapters needed with updated library)

// ============================================================================
// VDBVolumeManager Implementation
// ============================================================================

VDBVolumeManager::VDBVolumeManager() {
    // Constructor - initialization done in initialize()
}

VDBVolumeManager::~VDBVolumeManager() {
    shutdown();
}

VDBVolumeManager& VDBVolumeManager::getInstance() {
    static VDBVolumeManager instance;
    return instance;
}

void VDBVolumeManager::initialize() {
    if (initialized) return;
    
    try {
        openvdb::initialize();
        initialized = true;
        SCENE_LOG_INFO("VDBVolumeManager: OpenVDB initialized successfully");
    } catch (const std::exception& e) {
        last_error = std::string("OpenVDB initialization failed: ") + e.what();
        SCENE_LOG_ERROR(last_error);
    }
}

void VDBVolumeManager::shutdown() {
    if (!initialized) return;
    
    // Free all GPU resources
    freeAllGPU();
    
    // Clear all volumes
    for (auto& vol : volumes) {
        // Free internal OpenVDB grid
        if (vol.internal_openvdb_grid) {
            auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol.internal_openvdb_grid);
            delete grid_ptr;
            vol.internal_openvdb_grid = nullptr;
        }
        
        // Free NanoVDB handle
        if (vol.internal_nano_handle) {
            auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
            delete handle;
            vol.internal_nano_handle = nullptr;
        }
    }
    
    volumes.clear();
    initialized = false;
    
    SCENE_LOG_INFO("VDBVolumeManager: Shutdown complete");
}

int VDBVolumeManager::loadVDB(const std::string& filepath) {
    if (!initialized) {
        initialize();
    }
    
    if (!initialized) {
        last_error = "VDBVolumeManager not initialized";
        return -1;
    }
    
    try {
        // Check if file exists
        if (!std::filesystem::exists(filepath)) {
            last_error = "VDB file not found: " + filepath;
            SCENE_LOG_ERROR(last_error);
            return -1;
        }

        // VDB loading started

        // Open VDB file
        openvdb::io::File file(filepath);
        file.open();

        // Get grid names (GridPtrVecPtr matches OpenVDB API)
        openvdb::GridPtrVecPtr grids = file.getGrids();
        if (!grids || grids->empty()) {
            last_error = "VDB file contains no grids";
            file.close();
            SCENE_LOG_ERROR(last_error);
            return -1;
        }

        // Look for density grid (common names: density, Density, smoke, Smoke)
        openvdb::FloatGrid::Ptr density_grid = nullptr;

        // Explicit iterator usage to avoid ambiguity
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr grid = (*grids)[i];
            std::string name = grid->getName();
            // Grid found: name, type

            if ((name == "density" || name == "Density" || name == "smoke" || name == "Smoke" || name.empty())
                && grid->isType<openvdb::FloatGrid>()) {
                density_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                break;
            }
        }

        // If no density grid found, try the first float grid
        if (!density_grid) {
            for (size_t i = 0; i < grids->size(); ++i) {
                openvdb::GridBase::Ptr grid = (*grids)[i];
                if (grid->isType<openvdb::FloatGrid>()) {
                    density_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                    SCENE_LOG_INFO("  Using first FloatGrid as density: " + grid->getName());
                    break;
                }
            }
        }

        file.close();

        if (!density_grid) {
            last_error = "No suitable density grid found in VDB file";
            SCENE_LOG_ERROR(last_error);
            return -1;
        }

        // Search for temperature grid (optional)
        openvdb::FloatGrid::Ptr temp_grid = nullptr;
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr grid = (*grids)[i];
            std::string name = grid->getName();
            if ((name == "temperature" || name == "Temperature" || name == "heat" || name == "Heat")
                && grid->isType<openvdb::FloatGrid>()) {
                temp_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                // Temperature grid found
                break;
            }
        }

        // Create volume data
        VDBVolumeData vol;
        vol.id = next_id++;
        vol.filepath = filepath;
        vol.name = std::filesystem::path(filepath).stem().string();
        vol.has_density = true;
        vol.has_temperature = (temp_grid != nullptr);

        // Store OpenVDB grid (heap allocated to avoid copy)
        auto* grid_ptr = new openvdb::FloatGrid::Ptr(density_grid);
        vol.internal_openvdb_grid = grid_ptr;

        // Get voxel size
        vol.voxel_size = static_cast<float>(density_grid->voxelSize()[0]);

        // Get bounding box
        openvdb::CoordBBox bbox = density_grid->evalActiveVoxelBoundingBox();
        openvdb::BBoxd world_bbox = density_grid->transform().indexToWorld(bbox);

        vol.bbox_min[0] = static_cast<float>(world_bbox.min().x());
        vol.bbox_min[1] = static_cast<float>(world_bbox.min().y());
        vol.bbox_min[2] = static_cast<float>(world_bbox.min().z());
        vol.bbox_max[0] = static_cast<float>(world_bbox.max().x());
        vol.bbox_max[1] = static_cast<float>(world_bbox.max().y());
        vol.bbox_max[2] = static_cast<float>(world_bbox.max().z());

        // Bounds stored

        // Convert to NanoVDB
        // NanoVDB conversion

        try {
            // Use standard createNanoGrid from the upgraded library
            auto temp_handle = nanovdb::tools::createNanoGrid(*density_grid);

            // Move buffer out of the stack handle into a new heap handle
            // This is robust against deleted copy/move constructors on GridHandle itself
            // as long as HostBuffer is moveable (which it is).
            auto* handle_ptr = new nanovdb::GridHandle<nanovdb::HostBuffer>(
                std::move(temp_handle.buffer())
            );

            vol.internal_nano_handle = handle_ptr;
            vol.gpu_buffer_size = handle_ptr->bufferSize();

            // NanoVDB buffer ready

        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR("NanoVDB conversion failed: " + std::string(e.what()));
            return -1;
        }

        // Temperature Grid Conversion
        if (temp_grid) {
            // Store OpenVDB temp grid
            auto* temp_ptr = new openvdb::FloatGrid::Ptr(temp_grid);
            vol.internal_openvdb_temperature = temp_ptr;

            try {
                // Convert to NanoVDB
                auto temp_handle = nanovdb::tools::createNanoGrid(*temp_grid);
                auto* handle_ptr = new nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(temp_handle.buffer()));
                vol.internal_nano_temperature_handle = handle_ptr;
                vol.gpu_temp_buffer_size = handle_ptr->bufferSize();
                // Temperature grid converted
            }
            catch (const std::exception& e) {
                SCENE_LOG_WARN("NanoVDB Temperature conversion failed: " + std::string(e.what()));
                vol.has_temperature = false; // Disable if conversion fails
            }
        }

        // Create default transform
        vol.transform = std::make_shared<Transform>();

        // Add to list
        volumes.push_back(std::move(vol));

        SCENE_LOG_INFO("VDB loaded successfully: " + filepath + " (ID: " + std::to_string(volumes.back().id) + ")");

        return volumes.back().id;
    }
     catch (const openvdb::Exception& e) {
        last_error = std::string("OpenVDB error: ") + e.what();
        SCENE_LOG_ERROR(last_error);
        return -1;
    } catch (const std::exception& e) {
        last_error = std::string("Error loading VDB: ") + e.what();
        SCENE_LOG_ERROR(last_error);
        return -1;
    }
}

// Update existing volume data (for animation sequences)
bool VDBVolumeManager::updateVolume(int volume_id, const std::string& filepath) {
    if (!initialized) {
        last_error = "VDBVolumeManager not initialized";
        return false;
    }
    
    // Find volume
    int idx = -1;
    for (int i = 0; i < volumes.size(); ++i) {
        if (volumes[i].id == volume_id) {
            idx = i;
            break;
        }
    }
    
    if (idx < 0) {
        last_error = "Volume ID not found: " + std::to_string(volume_id);
        return false;
    }
    
    VDBVolumeData& vol = volumes[idx];
    
    // Check if file is same? If so, skip?
    if (vol.filepath == filepath) return true; // Already loaded
    
    if (!std::filesystem::exists(filepath)) {
        last_error = "VDB file not found: " + filepath;
        return false;
    }
    
    try {
        // Open VDB file
        openvdb::io::File file(filepath);
        file.open();
        
        openvdb::GridPtrVecPtr grids = file.getGrids();
        if (!grids || grids->empty()) {
            file.close();
            return false;
        }
        
        // Find Density Grid
        openvdb::FloatGrid::Ptr density_grid = nullptr;
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr grid = (*grids)[i];
            std::string name = grid->getName();
            if ((name == "density" || name == "Density" || name == "smoke" || name == "Smoke" || name.empty())
                && grid->isType<openvdb::FloatGrid>()) {
                density_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                break;
            }
        }
        
        // Fallback
        if (!density_grid) {
            for (size_t i = 0; i < grids->size(); ++i) {
                if ((*grids)[i]->isType<openvdb::FloatGrid>()) {
                    density_grid = openvdb::gridPtrCast<openvdb::FloatGrid>((*grids)[i]);
                    break;
                }
            }
        }
        
        if (!density_grid) {
            file.close();
            return false;
        }
        
        // Find Temperature Grid (MUST be done before file.close()!)
        openvdb::FloatGrid::Ptr temp_grid = nullptr;
        for (size_t i = 0; i < grids->size(); ++i) {
            openvdb::GridBase::Ptr grid = (*grids)[i];
            std::string name = grid->getName();
            if ((name == "temperature" || name == "Temperature" || name == "heat" || name == "Heat") 
                && grid->isType<openvdb::FloatGrid>()) {
                temp_grid = openvdb::gridPtrCast<openvdb::FloatGrid>(grid);
                break;
            }
        }
        
        file.close();
        
        // --------------------------------------------------------------------
        // PREPARE NEW RESOURCES (Before freeing old ones to ensure safety)
        // --------------------------------------------------------------------
        
        // NanoVDB Conversion
        auto temp_handle = nanovdb::tools::createNanoGrid(*density_grid);
        auto* new_nano_handle = new nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(temp_handle.buffer()));
        size_t new_gpu_size = new_nano_handle->bufferSize();
        
        // Temperature NanoVDB
        nanovdb::GridHandle<nanovdb::HostBuffer>* new_nano_temp_handle = nullptr;
        size_t new_gpu_temp_size = 0;
        
        if (temp_grid) {
            try {
                auto t_handle = nanovdb::tools::createNanoGrid(*temp_grid);
                new_nano_temp_handle = new nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(t_handle.buffer()));
                new_gpu_temp_size = new_nano_temp_handle->bufferSize();
            } catch(...) {
                // Ignore temp failure
            }
        }
        
        // New OpenVDB Pointers
        auto* new_ovdb_grid = new openvdb::FloatGrid::Ptr(density_grid);
        auto* new_ovdb_temp = temp_grid ? new openvdb::FloatGrid::Ptr(temp_grid) : nullptr;
        
        // --------------------------------------------------------------------
        // FREE OLD RESOURCES
        // --------------------------------------------------------------------
        
        // Free old OpenVDB
        if (vol.internal_openvdb_grid) {
             delete static_cast<openvdb::FloatGrid::Ptr*>(vol.internal_openvdb_grid);
        }
        if (vol.internal_openvdb_temperature) {
             delete static_cast<openvdb::FloatGrid::Ptr*>(vol.internal_openvdb_temperature);
        }
        
        // Free old NanoVDB
        if (vol.internal_nano_handle) {
            delete static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
        }
        if (vol.internal_nano_temperature_handle) {
             delete static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_temperature_handle);
        }
        
        // --------------------------------------------------------------------
        // ASSIGN NEW RESOURCES
        // --------------------------------------------------------------------
        vol.filepath = filepath; // Update path to new frame
        vol.filepath = filepath;
        vol.internal_openvdb_grid = new_ovdb_grid;
        vol.internal_openvdb_temperature = new_ovdb_temp;
        vol.internal_nano_handle = new_nano_handle;
        vol.internal_nano_temperature_handle = new_nano_temp_handle;
        vol.gpu_buffer_size = new_gpu_size;
        vol.gpu_temp_buffer_size = new_gpu_temp_size;
        
        vol.has_temperature = (temp_grid != nullptr);
        
        // UPDATE BOUNDING BOX from new frame's density grid
        openvdb::CoordBBox bbox = density_grid->evalActiveVoxelBoundingBox();
        openvdb::BBoxd world_bbox = density_grid->transform().indexToWorld(bbox);
        
        vol.bbox_min[0] = static_cast<float>(world_bbox.min().x());
        vol.bbox_min[1] = static_cast<float>(world_bbox.min().y());
        vol.bbox_min[2] = static_cast<float>(world_bbox.min().z());
        vol.bbox_max[0] = static_cast<float>(world_bbox.max().x());
        vol.bbox_max[1] = static_cast<float>(world_bbox.max().y());
        vol.bbox_max[2] = static_cast<float>(world_bbox.max().z());
        
        // Update voxel size too (in case it changed)
        vol.voxel_size = static_cast<float>(density_grid->voxelSize()[0]);
        
        // Free OLD GPU Memory
        if (vol.d_nano_grid) {
            cudaFree(vol.d_nano_grid);
            vol.d_nano_grid = nullptr;
        }
        if (vol.d_nano_temperature) {
            cudaFree(vol.d_nano_temperature);
            vol.d_nano_temperature = nullptr;
        }
        vol.gpu_uploaded = false;
        
        // Re-Upload (allocates new GPU memory)
        uploadToGPU(volume_id);
        
        return true;
        
    } catch (const std::exception& e) {
        last_error = std::string("Error updating VDB: ") + e.what();
        SCENE_LOG_ERROR(last_error);
        return false;
    }
}

void VDBVolumeManager::unloadVDB(int volume_id) {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return;
    
    VDBVolumeData& vol = volumes[idx];
    
    // Free GPU
    freeGPU(volume_id);
    
    // Free OpenVDB grid
    if (vol.internal_openvdb_grid) {
        auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol.internal_openvdb_grid);
        delete grid_ptr;
        vol.internal_openvdb_grid = nullptr;
    }

    if (vol.internal_openvdb_temperature) {
        auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol.internal_openvdb_temperature);
        delete grid_ptr;
        vol.internal_openvdb_temperature = nullptr;
    }
    
    // Free NanoVDB handle
    if (vol.internal_nano_handle) {
        auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
        delete handle;
        vol.internal_nano_handle = nullptr;
    }

    if (vol.internal_nano_temperature_handle) {
        auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_temperature_handle);
        delete handle;
        vol.internal_nano_temperature_handle = nullptr;
    }
    
    // Remove from list
    volumes.erase(volumes.begin() + idx);
    
    SCENE_LOG_INFO("VDB unloaded: ID " + std::to_string(volume_id));
}

void VDBVolumeManager::unloadAll() {
    while (!volumes.empty()) {
        unloadVDB(volumes.back().id);
    }
}

int VDBVolumeManager::findVolumeIndex(int volume_id) const {
    for (size_t i = 0; i < volumes.size(); ++i) {
        if (volumes[i].id == volume_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

VDBVolumeData* VDBVolumeManager::getVolume(int volume_id) {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return nullptr;
    return &volumes[idx];
}

const VDBVolumeData* VDBVolumeManager::getVolume(int volume_id) const {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return nullptr;
    return &volumes[idx];
}

bool VDBVolumeManager::uploadToGPU(int volume_id) {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) {
        last_error = "Volume not found: " + std::to_string(volume_id);
        return false;
    }
    
    VDBVolumeData& vol = volumes[idx];
    
    if (vol.gpu_uploaded) {
        return true; // Already uploaded
    }
    
    if (!vol.internal_nano_handle) {
        last_error = "No NanoVDB handle available";
        return false;
    }
    
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
    
    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&vol.d_nano_grid, handle->bufferSize());
    if (err != cudaSuccess) {
        last_error = std::string("CUDA malloc failed: ") + cudaGetErrorString(err);
        SCENE_LOG_ERROR(last_error);
        return false;
    }
    
    // Copy to GPU
    err = cudaMemcpy(vol.d_nano_grid, handle->data(), handle->bufferSize(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(vol.d_nano_grid);
        vol.d_nano_grid = nullptr;
        last_error = std::string("CUDA memcpy failed: ") + cudaGetErrorString(err);
        SCENE_LOG_ERROR(last_error);
        return false;
    }
    
    vol.gpu_buffer_size = handle->bufferSize();

    // Upload Temperature Grid if available
    if (vol.internal_nano_temperature_handle) {
        auto* temp_handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_temperature_handle);
        
        err = cudaMalloc(&vol.d_nano_temperature, temp_handle->bufferSize());
        if (err == cudaSuccess) {
            err = cudaMemcpy(vol.d_nano_temperature, temp_handle->data(), temp_handle->bufferSize(), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                 cudaFree(vol.d_nano_temperature);
                 vol.d_nano_temperature = nullptr;
                 SCENE_LOG_WARN("Failed to upload temperature grid to GPU");
            }
        } else {
             SCENE_LOG_WARN("Failed to allocate GPU memory for temperature grid");
        }
    }
    
    vol.gpu_uploaded = true;
    vol.gpu_buffer_size = handle->bufferSize();
    
    SCENE_LOG_INFO("VDB uploaded to GPU: " + vol.name + " (" + std::to_string(vol.gpu_buffer_size / (1024*1024)) + " MB)");
    
    return true;
}

void VDBVolumeManager::freeGPU(int volume_id) {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return;
    
    VDBVolumeData& vol = volumes[idx];
    
    if (vol.d_nano_grid) {
        cudaFree(vol.d_nano_grid);
        vol.d_nano_grid = nullptr;
        vol.gpu_uploaded = false; // Mark false only if main grid freed
        // GPU memory freed
    }
    if (vol.d_nano_temperature) {
        cudaFree(vol.d_nano_temperature);
        vol.d_nano_temperature = nullptr;
    }
}

void VDBVolumeManager::freeAllGPU() {
    for (auto& vol : volumes) {
        if (vol.d_nano_grid) {
            cudaFree(vol.d_nano_grid);
            vol.d_nano_grid = nullptr;
            vol.gpu_uploaded = false;
        }
    }
}

void* VDBVolumeManager::getGPUGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->gpu_uploaded) return nullptr;
    return vol->d_nano_grid;
}

void* VDBVolumeManager::getGPUTemperatureGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->gpu_uploaded) return nullptr;
    return vol->d_nano_temperature;
}

float VDBVolumeManager::sampleDensityCPU(int volume_id, float x, float y, float z) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_openvdb_grid) return 0.0f;
    
    auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol->internal_openvdb_grid);
    if (!grid_ptr || !*grid_ptr) return 0.0f;
    
    openvdb::FloatGrid::Ptr grid = *grid_ptr;
    
    // Use trilinear interpolation sampler
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);
    
    openvdb::Vec3d world_pos(x, y, z);
    
    // Apply transform if available
    if (vol->transform) {
        // Transform is applied to world position
        // For now, we use the position directly
        // TODO: Apply inverse transform
    }
    
    float density = sampler.wsSample(world_pos);
    return density * vol->density_scale;
}

float VDBVolumeManager::sampleTemperatureCPU(int volume_id, float x, float y, float z) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_openvdb_temperature) return 0.0f;
    
    auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol->internal_openvdb_temperature);
    if (!grid_ptr || !*grid_ptr) return 0.0f;
    
    openvdb::FloatGrid::Ptr grid = *grid_ptr;
    openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*grid);
    openvdb::Vec3d world_pos(x, y, z);
    
    return sampler.wsSample(world_pos);
}

bool VDBVolumeManager::hasTemperatureGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    return vol && vol->has_temperature && vol->internal_openvdb_temperature;
}
