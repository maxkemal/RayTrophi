#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>



// Forward declare Vec3 and AABB
class Vec3;
class AABB;
class Transform;

/**
 * @brief VDB Volume data container
 * Holds both CPU (OpenVDB) and GPU (NanoVDB) representations
 */
struct VDBVolumeData {
    int id = -1;
    std::string name;
    std::string filepath;
    
    // Volume properties
    float density_scale = 1.0f;
    float voxel_size = 1.0f;
    
    // Bounding box in world space
    float bbox_min[3] = {0, 0, 0};
    float bbox_max[3] = {1, 1, 1};
    
    // Transform for positioning in scene
    std::shared_ptr<Transform> transform;
    
    // GPU buffer info
    void* d_nano_grid = nullptr;        // CUDA device pointer (Density)
    void* d_nano_temperature = nullptr; // CUDA device pointer (Temperature)
    
    size_t gpu_buffer_size = 0;
    size_t gpu_temp_buffer_size = 0;
    bool gpu_uploaded = false;
    
    // Grid type info
    bool has_density = false;
    bool has_temperature = false;
    bool has_velocity = false;
    
    // Internal handles (opaque to users)
    void* internal_openvdb_grid = nullptr;        // openvdb::FloatGrid::Ptr (Density)
    void* internal_openvdb_temperature = nullptr; // openvdb::FloatGrid::Ptr (Temperature)
    
    void* internal_nano_handle = nullptr;         // nanovdb::GridHandle (Density)
    void* internal_nano_temperature_handle = nullptr; // nanovdb::GridHandle (Temperature)
};

/**
 * @brief Manager for VDB volumes with NanoVDB GPU support
 * 
 * Usage:
 *   int id = VDBVolumeManager::getInstance().loadVDB("cloud.vdb");
 *   VDBVolumeManager::getInstance().uploadToGPU(id);
 *   void* gpu_grid = VDBVolumeManager::getInstance().getGPUGrid(id);
 */
class VDBVolumeManager {
public:
    static VDBVolumeManager& getInstance();
    
    // Lifecycle
    void initialize();
    void shutdown();
    
    // File I/O
    int loadVDB(const std::string& filepath);
    void unloadVDB(int volume_id);
    void unloadAll();
    
    // Update existing volume data (for animation sequences)
    bool updateVolume(int volume_id, const std::string& filepath);
    
    // Access
    VDBVolumeData* getVolume(int volume_id);
    const VDBVolumeData* getVolume(int volume_id) const;
    const std::vector<VDBVolumeData>& getAllVolumes() const { return volumes; }
    int getVolumeCount() const { return static_cast<int>(volumes.size()); }
    
    // GPU Management
    bool uploadToGPU(int volume_id);
    void freeGPU(int volume_id);
    void freeAllGPU();
    void* getGPUGrid(int volume_id) const;
    void* getGPUTemperatureGrid(int volume_id) const;
    
    // CPU Sampling (for CPU renderer)
    float sampleDensityCPU(int volume_id, float x, float y, float z) const;
    float sampleTemperatureCPU(int volume_id, float x, float y, float z) const;
    bool hasTemperatureGrid(int volume_id) const;
    
    // Utility
    bool isInitialized() const { return initialized; }
    std::string getLastError() const { return last_error; }
    
private:
    VDBVolumeManager();
    ~VDBVolumeManager();
    
    // Prevent copying
    VDBVolumeManager(const VDBVolumeManager&) = delete;
    VDBVolumeManager& operator=(const VDBVolumeManager&) = delete;
    
    std::vector<VDBVolumeData> volumes;
    int next_id = 0;
    bool initialized = false;
    std::string last_error;
    
    // Helper to find volume index by ID
    int findVolumeIndex(int volume_id) const;
};
