// Suppress C4146 warning globally for this file (OpenVDB/NanoVDB compatibility)
#pragma warning(disable: 4146)

#ifndef NOMINMAX
#define NOMINMAX
#endif

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
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/Interpolation.h>

// Undefine min and max macros that might have been brought in by Windows headers
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <nanovdb/NanoVDB.h>
// Enable OpenVDB support in NanoVDB tools
#define NANOVDB_USE_OPENVDB 
// Use standard CreateNanoGrid (compatible with latest lib)
#include <nanovdb/tools/CreateNanoGrid.h> 

// CUDA
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
        vol.content_version = 1;

        // Store OpenVDB grid (heap allocated to avoid copy)
        auto* grid_ptr = new openvdb::FloatGrid::Ptr(density_grid);
        vol.internal_openvdb_grid = grid_ptr;

        // Get voxel size
        vol.voxel_size = static_cast<float>(density_grid->voxelSize()[0]);

        // Get bounding box
        openvdb::CoordBBox bbox = density_grid->evalActiveVoxelBoundingBox();
        
        if (bbox.empty()) {
            // Empty grid fallback
            vol.bbox_min[0] = -0.5f; vol.bbox_min[1] = -0.5f; vol.bbox_min[2] = -0.5f;
            vol.bbox_max[0] = 0.5f;  vol.bbox_max[1] = 0.5f;  vol.bbox_max[2] = 0.5f;
        } else {
            openvdb::BBoxd world_bbox = density_grid->transform().indexToWorld(bbox);

            vol.bbox_min[0] = static_cast<float>(world_bbox.min().x());
            vol.bbox_min[1] = static_cast<float>(world_bbox.min().y());
            vol.bbox_min[2] = static_cast<float>(world_bbox.min().z());
            vol.bbox_max[0] = static_cast<float>(world_bbox.max().x());
            vol.bbox_max[1] = static_cast<float>(world_bbox.max().y());
            vol.bbox_max[2] = static_cast<float>(world_bbox.max().z());
        }

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
bool VDBVolumeManager::updateVolume(int volume_id, const std::string& filepath, void* stream) {
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
        
        vol.has_temperature = (temp_grid != nullptr);
        vol.content_version++;
        
        // UPDATE BOUNDING BOX from new frame's density grid
        openvdb::CoordBBox bbox = density_grid->evalActiveVoxelBoundingBox();
        if (bbox.empty()) {
            vol.bbox_min[0] = -0.5f; vol.bbox_min[1] = -0.5f; vol.bbox_min[2] = -0.5f;
            vol.bbox_max[0] = 0.5f;  vol.bbox_max[1] = 0.5f;  vol.bbox_max[2] = 0.5f;
        } else {
            openvdb::BBoxd world_bbox = density_grid->transform().indexToWorld(bbox);
            vol.bbox_min[0] = static_cast<float>(world_bbox.min().x());
            vol.bbox_min[1] = static_cast<float>(world_bbox.min().y());
            vol.bbox_min[2] = static_cast<float>(world_bbox.min().z());
            vol.bbox_max[0] = static_cast<float>(world_bbox.max().x());
            vol.bbox_max[1] = static_cast<float>(world_bbox.max().y());
            vol.bbox_max[2] = static_cast<float>(world_bbox.max().z());
        }
        
        // Update voxel size too (in case it changed)
        vol.voxel_size = static_cast<float>(density_grid->voxelSize()[0]);
        
        // SMART GPU SYNC: Only re-allocate if existing buffer is too small.
        // This prevents flickering caused by constant cudaFree/cudaMalloc during simulation.
        if (g_hasCUDA && vol.d_nano_grid && vol.gpu_buffer_size < new_gpu_size) {
             cudaFree(vol.d_nano_grid);
             vol.d_nano_grid = nullptr;
        }
        if (g_hasCUDA && vol.d_nano_temperature && vol.gpu_temp_buffer_size < new_gpu_temp_size) {
             cudaFree(vol.d_nano_temperature);
             vol.d_nano_temperature = nullptr;
        }
        
        // Reset GPU uploaded state to force a RE-COPY of data, 
        // but the pointers (if valid) will stay, skipping malloc.
        vol.gpu_uploaded = false; 
        
        // Re-Upload (allocates only if pointers were nulled)
        uploadToGPU(volume_id, true, stream); // Silent mode for sequence scrubbing
        
        return true;
        
    } catch (const std::exception& e) {
        last_error = std::string("Error updating VDB: ") + e.what();
        SCENE_LOG_ERROR(last_error);
        return false;
    }
}

int VDBVolumeManager::registerOrUpdateLiveVolume(int existing_id, const std::string& name, 
                                               int res_x, int res_y, int res_z, float voxel_size,
                                               const float* density_ptr, const float* temp_ptr, void* stream) {
    if (!initialized) initialize();
    if (!initialized) return -1;

    int volume_id = existing_id;
    int idx = findVolumeIndex(volume_id);
    
    // Create new volume entry if not found
    if (idx < 0) {
        VDBVolumeData vol;
        vol.id = next_id++;
        vol.name = name;
        vol.filepath = "[LIVE SIMULATION]";
        vol.transform = std::make_shared<Transform>();
        volumes.push_back(std::move(vol));
        volume_id = volumes.back().id;
        idx = static_cast<int>(volumes.size() - 1);
    }

    VDBVolumeData& vol = volumes[idx];
    vol.has_density = (density_ptr != nullptr);
    vol.has_temperature = (temp_ptr != nullptr);
    vol.voxel_size = voxel_size;
    
    try {
        // Dense → sparse population via openvdb::tools::copyFromDense. Internally
        // parallel (TBB) and replaces a host-side O(nx*ny*nz) accessor.setValue
        // hot loop — the prior bottleneck of the live fluid/gas NanoVDB bridge.
        // Layout matches the dense pointer indexing: density[i + nx*(j + ny*k)],
        // i.e. X fastest → openvdb::tools::LayoutXYZ. NOTE: LayoutXYZ is an enum
        // *value* in openvdb::tools::MemoryLayout, not a type — pass it directly
        // as the non-type template argument, do not try to alias it via `using`.
        using DenseFloat = openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ>;
        const openvdb::CoordBBox dense_bbox(
            openvdb::Coord(0, 0, 0),
            openvdb::Coord(res_x - 1, res_y - 1, res_z - 1));

        // 1. Density grid
        openvdb::FloatGrid::Ptr density_grid = openvdb::FloatGrid::create(0.0f);
        {
            // copyFromDense reads-only from the dense buffer; const_cast is safe.
            DenseFloat dense(dense_bbox, const_cast<float*>(density_ptr));
            openvdb::tools::copyFromDense(dense, *density_grid, /*tolerance*/ 0.0001f);
        }
        density_grid->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size));

        // 2. NanoVDB conversion for density
        auto temp_handle = nanovdb::tools::createNanoGrid(*density_grid);
        auto* new_nano_handle = new nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(temp_handle.buffer()));

        // Assign & free old
        if (vol.internal_nano_handle) delete static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
        vol.internal_nano_handle = new_nano_handle;

        // 3. Temperature grid (optional)
        if (temp_ptr) {
            openvdb::FloatGrid::Ptr temp_grid = openvdb::FloatGrid::create(0.0f);
            {
                DenseFloat dense(dense_bbox, const_cast<float*>(temp_ptr));
                // Threshold matches the old loop's `t > 300.0f` Kelvin floor:
                // values within 300 of the background (0) are dropped, keeping
                // only voxels hotter than ~300K.
                openvdb::tools::copyFromDense(dense, *temp_grid, /*tolerance*/ 300.0f);
            }
            temp_grid->setTransform(density_grid->transform().copy());
            auto t_handle = nanovdb::tools::createNanoGrid(*temp_grid);
            auto* new_t_nano_handle = new nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(t_handle.buffer()));

            if (vol.internal_nano_temperature_handle) delete static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_temperature_handle);
            vol.internal_nano_temperature_handle = new_t_nano_handle;
        }

        // Update bounds
        vol.bbox_min[0] = 0; vol.bbox_min[1] = 0; vol.bbox_min[2] = 0;
        vol.bbox_max[0] = res_x * voxel_size; 
        vol.bbox_max[1] = res_y * voxel_size; 
        vol.bbox_max[2] = res_z * voxel_size;
        vol.content_version++;

        // SEAMLESS UPDATE: Do NOT set gpu_uploaded = false here.
        // uploadToGPU will perform the copy and only update flags when ready.
        uploadToGPU(volume_id, true, stream);

        return volume_id;

    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("registerOrUpdateLiveVolume failed: " + std::string(e.what()));
        return -1;
    }
}

bool VDBVolumeManager::exportDenseGridToVDB(const std::string& filepath,
                                            int res_x, int res_y, int res_z, float voxel_size,
                                            float origin_x, float origin_y, float origin_z,
                                            const float* density, const float* temperature,
                                            const float* fuel, const float* flame) {
    if (res_x <= 0 || res_y <= 0 || res_z <= 0) {
        return false;
    }
    auto cellIndex = [res_x, res_y](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(res_x) * (static_cast<std::size_t>(j) +
               static_cast<std::size_t>(res_y) * static_cast<std::size_t>(k));
    };

    auto buildGrid = [&](const float* data, const char* name, float threshold) -> openvdb::FloatGrid::Ptr {
        if (!data) return nullptr;
        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
        grid->setName(name);
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        auto accessor = grid->getAccessor();
        for (int k = 0; k < res_z; ++k)
            for (int j = 0; j < res_y; ++j)
                for (int i = 0; i < res_x; ++i) {
                    const float v = data[cellIndex(i, j, k)];
                    if (v > threshold) {
                        accessor.setValue(openvdb::Coord(i, j, k), v);
                    }
                }
        return grid;
    };

    try {
        openvdb::GridPtrVec grids;
        openvdb::math::Transform::Ptr xform =
            openvdb::math::Transform::createLinearTransform(voxel_size);
        xform->postTranslate(openvdb::Vec3d(origin_x, origin_y, origin_z));

        if (auto g = buildGrid(density, "density", 1e-6f)) { g->setTransform(xform); grids.push_back(g); }
        if (auto g = buildGrid(temperature, "temperature", 1e-4f)) { g->setTransform(xform); grids.push_back(g); }
        if (auto g = buildGrid(fuel, "fuel", 1e-4f)) { g->setTransform(xform); grids.push_back(g); }
        if (auto g = buildGrid(flame, "flame", 1e-4f)) { g->setTransform(xform); grids.push_back(g); }

        if (grids.empty()) {
            return false;
        }
        openvdb::io::File file(filepath);
        file.write(grids);
        file.close();
        return true;
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR("exportDenseGridToVDB failed: " + std::string(e.what()));
        return false;
    } catch (...) {
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
    
    //SCENE_LOG_INFO("VDB unloaded: ID " + std::to_string(volume_id));
}

void VDBVolumeManager::unloadAll() {
    while (!volumes.empty()) {
        unloadVDB(volumes.back().id);
    }
    next_id = 0; // Reset ID counter for new project sessions
    SCENE_LOG_INFO("VDBVolumeManager: All volumes cleared and ID counter reset.");
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

bool VDBVolumeManager::uploadToGPU(int volume_id, bool silent, void* stream_ptr) {
    if (!g_hasCUDA) return false;

    // Check if the OptiX backend is actually active.
    // If not active, bypass CUDA allocations to avoid double allocation/redundant copying.
    if (!render_settings.use_optix) {
        // Release any existing CUDA VRAM allocations for this volume to prevent leaks
        freeGPU(volume_id);
        
        // Mark uploaded status as true because host grid buffer is fully processed & ready for Vulkan RT
        int idx = findVolumeIndex(volume_id);
        if (idx >= 0) {
            volumes[idx].gpu_uploaded = true;
        }
        return true;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    cudaGetLastError(); // Clear sticky error flag without forcing a global stall
    
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) {
        last_error = "Volume not found: " + std::to_string(volume_id);
        return false;
    }
    
    VDBVolumeData& vol = volumes[idx];
    
    // SEAMLESS SYNC: Even if gpu_uploaded is true, we proceed if we have a handle
    // to ensure live simulation frames are copied.
    
    if (!vol.internal_nano_handle) {
        last_error = "No NanoVDB handle available";
        return false;
    }
    
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_handle);
    size_t required_size = handle->bufferSize();

    if (required_size == 0) {
        last_error = "NanoVDB buffer size is 0";
        return false;
    }
    
    // Ensure correct CUDA context
    cudaError_t setDeviceErr = cudaSetDevice(0);
    if (setDeviceErr != cudaSuccess) {
        last_error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(setDeviceErr);
        if (!silent) SCENE_LOG_ERROR(last_error);
        return false;
    }
    
    // SMART REALLOC: Allocate GPU memory only if needed or if existing buffer is too small
    cudaError_t err = cudaSuccess;
    if (vol.d_nano_grid == nullptr || vol.gpu_buffer_size < required_size) {
        // CRITICAL: synchronize before freeing memory. Live grid resets can
        // replace the NanoVDB buffer while OptiX still has the old pointer in
        // its launch params / volume table from the previous frame.
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
        
        if (vol.d_nano_grid) {
            cudaFree(vol.d_nano_grid);
            vol.d_nano_grid = nullptr;
        }
        
        err = cudaMalloc(&vol.d_nano_grid, required_size);
        if (err != cudaSuccess) {
            vol.d_nano_grid = nullptr;
            vol.gpu_buffer_size = 0;
            last_error = std::string("CUDA malloc failed: ") + cudaGetErrorString(err);
            if (!silent) SCENE_LOG_ERROR(last_error);
            return false;
        }
        vol.gpu_buffer_size = required_size;
    }
    
    if (!vol.d_nano_grid) {
         last_error = "Destination GPU pointer is NULL after malloc";
         if (!silent) SCENE_LOG_ERROR(last_error);
         return false;
    }
    
    // Copy to GPU (use synchronous copy for safety)
    void* host_ptr = handle->data();
    if (!host_ptr) {
        last_error = "NanoVDB handle.data() is NULL";
        if (!silent) SCENE_LOG_ERROR(last_error);
        return false;
    }

    // Sanity checks and logging to help diagnose illegal memory accesses.
    if (required_size == 0 || required_size > (size_t)1024 * 1024 * 1024) { // >1GB suspicious
        last_error = "Suspicious NanoVDB buffer size: " + std::to_string(required_size);
        if (!silent) SCENE_LOG_WARN(last_error);
    }

    // NOTE: Use async upload when a stream is provided to avoid stalling the frame.
    err = stream
        ? cudaMemcpyAsync(vol.d_nano_grid, host_ptr, required_size, cudaMemcpyHostToDevice, stream)
        : cudaMemcpy(vol.d_nano_grid, host_ptr, required_size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        last_error = std::string("CUDA memcpy failed: ") + cudaGetErrorString(err) + 
                     " (Size: " + std::to_string(required_size) + 
                     ", Dst: " + std::to_string((unsigned long long)vol.d_nano_grid) + 
                     ", Src: " + std::to_string((unsigned long long)host_ptr) + ")";
        if (!silent) SCENE_LOG_ERROR(last_error);
        return false;
    }
    
    vol.gpu_buffer_size = handle->bufferSize();

    // Upload Temperature Grid if available
    if (vol.internal_nano_temperature_handle) {
        auto* temp_handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol.internal_nano_temperature_handle);
        size_t temp_required_size = temp_handle->bufferSize();
        
        if (temp_required_size > 0) {
            if (vol.d_nano_temperature == nullptr || vol.gpu_temp_buffer_size < temp_required_size) {
                if (stream) {
                    cudaStreamSynchronize(stream);
                } else {
                    cudaDeviceSynchronize();
                }
                if (vol.d_nano_temperature) cudaFree(vol.d_nano_temperature);
                cudaError_t tempAllocErr = cudaMalloc(&vol.d_nano_temperature, temp_required_size);
                if (tempAllocErr != cudaSuccess) {
                    vol.d_nano_temperature = nullptr;
                    vol.gpu_temp_buffer_size = 0;
                    last_error = std::string("CUDA temp malloc failed: ") + cudaGetErrorString(tempAllocErr);
                    if (!silent) SCENE_LOG_ERROR(last_error);
                    return false;
                }
                vol.gpu_temp_buffer_size = temp_required_size;
            }
            
            if (vol.d_nano_temperature != nullptr) {
                void* t_host_ptr = temp_handle->data();
                if (t_host_ptr) {
                    if (temp_required_size == 0 || temp_required_size > (size_t)1024 * 1024 * 1024) {
                        if (!silent) SCENE_LOG_WARN("Suspicious NanoVDB temperature buffer size: " + std::to_string(temp_required_size));
                    }
                    cudaError_t tempCopyErr = stream
                        ? cudaMemcpyAsync(vol.d_nano_temperature, t_host_ptr, temp_required_size, cudaMemcpyHostToDevice, stream)
                        : cudaMemcpy(vol.d_nano_temperature, t_host_ptr, temp_required_size, cudaMemcpyHostToDevice);
                    if (tempCopyErr != cudaSuccess) {
                        last_error = std::string("CUDA memcpy temp failed: ") + cudaGetErrorString(tempCopyErr);
                        if (!silent) SCENE_LOG_ERROR(last_error);
                        return false;
                    }
                }
            }
        }
    }
    
    vol.gpu_uploaded = true;
    
    std::string size_str;
    if (vol.gpu_buffer_size < 1024 * 1024) {
        size_str = std::to_string(vol.gpu_buffer_size / 1024) + " KB";
    } else {
        size_str = std::to_string(vol.gpu_buffer_size / (1024 * 1024)) + " MB";
    }
    
    if (!silent) {
        SCENE_LOG_INFO("VDB uploaded to GPU: " + vol.name + " (" + size_str + ")");
    }
    
    return true;
}

void VDBVolumeManager::freeGPU(int volume_id) {
    if (!g_hasCUDA) return;
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return;
    
    VDBVolumeData& vol = volumes[idx];
    
    // Check if there's anything to free first
    if (!vol.d_nano_grid && !vol.d_nano_temperature) {
        return;
    }
    
    // Clear any sticky CUDA errors first
    cudaGetLastError();
    
    // Check if CUDA is available before sync
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess && device_count > 0) {
        // CRITICAL: Sync device before freeing to ensure no async operations use these pointers
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            SCENE_LOG_INFO("[VDBVolumeManager] cudaDeviceSynchronize failed in freeGPU: " + 
                           std::string(cudaGetErrorString(err)));
            cudaGetLastError(); // Clear error
        }
    }
    
    if (vol.d_nano_grid) {
        cudaFree(vol.d_nano_grid);
        vol.d_nano_grid = nullptr;
        vol.gpu_buffer_size = 0; // Reset size so next upload reallocates
        vol.gpu_uploaded = false; // Mark false only if main grid freed
        // GPU memory freed
    }
    if (vol.d_nano_temperature) {
        cudaFree(vol.d_nano_temperature);
        vol.d_nano_temperature = nullptr;
        vol.gpu_temp_buffer_size = 0; // Reset size
    }
}

void VDBVolumeManager::freeAllGPU() {
    if (!g_hasCUDA) return;

    for (auto& vol : volumes) {
        freeGPU(vol.id);
    }
}

void* VDBVolumeManager::getGPUGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol) return nullptr;
    return vol->d_nano_grid;
}

void* VDBVolumeManager::getGPUTemperatureGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol) return nullptr;
    return vol->d_nano_temperature;
}

void* VDBVolumeManager::getHostGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_nano_handle) return nullptr;
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol->internal_nano_handle);
    return handle->data();
}

size_t VDBVolumeManager::getHostGridSize(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_nano_handle) return 0;
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol->internal_nano_handle);
    return handle->bufferSize();
}

void* VDBVolumeManager::getHostTemperatureGrid(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_nano_temperature_handle) return nullptr;
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol->internal_nano_temperature_handle);
    return handle->data();
}

size_t VDBVolumeManager::getHostTemperatureGridSize(int volume_id) const {
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_nano_temperature_handle) return 0;
    auto* handle = static_cast<nanovdb::GridHandle<nanovdb::HostBuffer>*>(vol->internal_nano_temperature_handle);
    return handle->bufferSize();
}

uint32_t VDBVolumeManager::getContentVersion(int volume_id) const {
    int idx = findVolumeIndex(volume_id);
    if (idx < 0) return 0;
    return volumes[idx].content_version;
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
    return density;
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

VDBDensityStats VDBVolumeManager::analyzeDensityStats(int volume_id, std::size_t max_samples) const {
    VDBDensityStats stats;
    const VDBVolumeData* vol = getVolume(volume_id);
    if (!vol || !vol->internal_openvdb_grid) return stats;

    auto* grid_ptr = static_cast<openvdb::FloatGrid::Ptr*>(vol->internal_openvdb_grid);
    if (!grid_ptr || !*grid_ptr) return stats;

    const openvdb::FloatGrid::Ptr& grid = *grid_ptr;
    if (!grid) return stats;

    std::vector<float> samples;
    samples.reserve(max_samples);

    double sum = 0.0;
    bool first_value = true;
    std::uint64_t seen = 0;

    for (auto it = grid->cbeginValueOn(); it.test(); ++it) {
        const float value = *it;
        if (!std::isfinite(value) || value <= 0.0f) {
            continue;
        }

        ++seen;
        sum += value;

        if (first_value) {
            stats.min_value = value;
            stats.max_value = value;
            first_value = false;
        } else {
            stats.min_value = (std::min)(stats.min_value, value);
            stats.max_value = (std::max)(stats.max_value, value);
        }

        if (samples.size() < max_samples) {
            samples.push_back(value);
        } else {
            const std::uint64_t hashed = (seen * 2654435761ull) ^ (seen >> 11);
            const std::size_t slot = static_cast<std::size_t>(hashed % seen);
            if (slot < max_samples) {
                samples[slot] = value;
            }
        }
    }

    if (seen == 0 || samples.empty()) {
        return stats;
    }

    std::sort(samples.begin(), samples.end());
    auto percentile = [&samples](float t) -> float {
        const float clamped = (std::max)(0.0f, (std::min)(1.0f, t));
        const std::size_t idx = static_cast<std::size_t>(clamped * static_cast<float>(samples.size() - 1));
        return samples[idx];
    };

    stats.valid = true;
    stats.active_voxel_count = seen;
    stats.mean_value = static_cast<float>(sum / static_cast<double>(seen));
    stats.p50_value = percentile(0.50f);
    stats.p95_value = percentile(0.95f);
    stats.p99_value = percentile(0.99f);

    const float reference_density = stats.p99_value > 1e-5f
        ? stats.p99_value
        : (stats.p95_value > 1e-5f ? stats.p95_value : stats.max_value);
    if (reference_density > 1e-6f) {
        // Bias towards visibly readable defaults for sparse caches.
        stats.recommended_smoke_multiplier = (std::max)(2.0f, (std::min)(35.0f, 0.35f / reference_density));
    }

    return stats;
}
