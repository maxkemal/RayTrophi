/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          OptixAccelManager.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include "Vec3.h"
#include "material_gpu.h"
#include "sbt_record.h"
#include "sbt_data.h"
#include <OptixTypes.h>
#include <functional> // Added for std::function
#include "PrincipledBSDF.h"
#include "Matrix4x4.h"
#include "Hittable.h"
#include "skinning_kernels.cuh" // GPU Skinning Kernels
// ═══════════════════════════════════════════════════════════════════════════
// OPTIX TLAS/BLAS ACCELERATION STRUCTURE MANAGER
// ═══════════════════════════════════════════════════════════════════════════
// Two-level acceleration structure for efficient scene updates:
// - BLAS (Bottom-Level AS): Per-mesh geometry, built once
// - TLAS (Top-Level AS): Instances with transforms, rebuilt on transform change
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

// Forward declaration
class Triangle;


// Per-mesh Bottom-Level Acceleration Structure
struct MeshBLAS {
    OptixTraversableHandle handle = 0;      // BLAS traversable handle
    CUdeviceptr d_vertices = 0;             // GPU vertex buffer (Deformed/Current)
    CUdeviceptr d_vertices_rest = 0;        // GPU vertex buffer (Rest Pose - Original)
    CUdeviceptr d_indices = 0;              // GPU index buffer
    CUdeviceptr d_normals = 0;              // GPU normal buffer
    CUdeviceptr d_uvs = 0;                  // GPU UV buffer  
    CUdeviceptr d_tangents = 0;             // GPU tangent buffer
    CUdeviceptr d_vertex_colors = 0;        // GPU Vertex colors (RGB)
    bool has_vertex_colors = false;
    CUdeviceptr d_gas_output = 0;           // GAS output buffer
    size_t gas_output_size = 0;             // GAS output buffer size (for refit)
    size_t vertex_count = 0;
    size_t index_count = 0;
    int sbt_offset = 0;                     // Base SBT offset (for RAY_TYPE_COUNT records)
    int material_id = 0;                    // Material ID for this mesh
    std::string mesh_name;                  // Unique name (e.g. "Car_mat_0")
    std::string original_name;              // Node name from OBJ/FBX (e.g. "Car")
    
    // Foliage Properties
    bool is_foliage = false;
    float mesh_height = 0.0f;
    float3 mesh_pivot = {0.0f, 0.0f, 0.0f};
    
    // GPU Skinning Buffers
    CUdeviceptr d_bindPoses = 0;            // Original Bind Pose Vertices
    CUdeviceptr d_bindNormals = 0;          // Original Bind Pose Normals
    CUdeviceptr d_boneIndices = 0;          // Bone Indices (int4)
    CUdeviceptr d_boneWeights = 0;          // Bone Weights (float4)
    bool hasSkinningData = false;
    
    void cleanup() {
        if (d_vertices) { cudaFree(reinterpret_cast<void*>(d_vertices)); d_vertices = 0; }
        if (d_vertices_rest) { cudaFree(reinterpret_cast<void*>(d_vertices_rest)); d_vertices_rest = 0; }
        if (d_indices) { cudaFree(reinterpret_cast<void*>(d_indices)); d_indices = 0; }
        if (d_normals) { cudaFree(reinterpret_cast<void*>(d_normals)); d_normals = 0; }
        if (d_uvs) { cudaFree(reinterpret_cast<void*>(d_uvs)); d_uvs = 0; }
        if (d_uvs) { cudaFree(reinterpret_cast<void*>(d_uvs)); d_uvs = 0; }
        if (d_tangents) { cudaFree(reinterpret_cast<void*>(d_tangents)); d_tangents = 0; }
        if (d_vertex_colors) { cudaFree(reinterpret_cast<void*>(d_vertex_colors)); d_vertex_colors = 0; }
        if (d_gas_output) { cudaFree(reinterpret_cast<void*>(d_gas_output)); d_gas_output = 0; }
        
        // Free Skinning Buffers
        if (d_bindPoses) { cudaFree(reinterpret_cast<void*>(d_bindPoses)); d_bindPoses = 0; }
        if (d_bindNormals) { cudaFree(reinterpret_cast<void*>(d_bindNormals)); d_bindNormals = 0; }
        if (d_boneIndices) { cudaFree(reinterpret_cast<void*>(d_boneIndices)); d_boneIndices = 0; }
        if (d_boneWeights) { cudaFree(reinterpret_cast<void*>(d_boneWeights)); d_boneWeights = 0; }
        
        handle = 0;
    }
};


// Per-hair curve Bottom-Level Acceleration Structure
struct CurveBLAS {
    OptixTraversableHandle handle = 0;
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_widths = 0;         // Separate per-vertex radius buffer (required by OptiX)
    CUdeviceptr d_indices = 0;
    CUdeviceptr d_tangents = 0;
    CUdeviceptr d_strand_ids = 0;
    CUdeviceptr d_gas_output = 0;
    size_t gas_output_size = 0;
    size_t vertex_count = 0;
    size_t segment_count = 0;
    int sbt_offset = 0;
    int material_id = 0;
    int mesh_material_id = -1;
    GpuHairMaterial hair_material;
    std::string name;
    bool use_bspline = false;
    CUdeviceptr d_root_uvs = 0;       // Per-segment root UV (float2)

    // Temporary storage for refit pointers to ensure they stay valid for async builds
    CUdeviceptr d_refit_vertices = 0;
    CUdeviceptr d_refit_widths = 0;

    void cleanup() {
        if (d_vertices) { cudaFree(reinterpret_cast<void*>(d_vertices)); d_vertices = 0; }
        if (d_widths) { cudaFree(reinterpret_cast<void*>(d_widths)); d_widths = 0; }
        if (d_indices) { cudaFree(reinterpret_cast<void*>(d_indices)); d_indices = 0; }
        if (d_tangents) { cudaFree(reinterpret_cast<void*>(d_tangents)); d_tangents = 0; }
        if (d_strand_ids) { cudaFree(reinterpret_cast<void*>(d_strand_ids)); d_strand_ids = 0; }
        if (d_root_uvs) { cudaFree(reinterpret_cast<void*>(d_root_uvs)); d_root_uvs = 0; }
        if (d_gas_output) { cudaFree(reinterpret_cast<void*>(d_gas_output)); d_gas_output = 0; }
        handle = 0;
    }
};

enum class InstanceType {
    Mesh,
    Curve
};

// Per-instance data for TLAS
struct SceneInstance {
    InstanceType type = InstanceType::Mesh;
    int blas_id = -1;                       // Index into mesh_blas_list or curve_blas_list
    int material_id = 0;                    // Material index for SBT
    float transform[12] = {                 // 3x4 row-major transform matrix
        1, 0, 0, 0,                         // Row 0: scale.x * right + tx
        0, 1, 0, 0,                         // Row 1: scale.y * up + ty
        0, 0, 1, 0                          // Row 2: scale.z * forward + tz
    };
    uint32_t visibility_mask = 255;         // Ray visibility mask (0 = invisible)
    uint32_t sbt_offset = 0;                // Computed SBT offset
    std::string node_name;                  // Node name for lookup
    void* source_hittable = nullptr;        // Pointer to CPU Hittable for transform sync
    bool visible = true;                    // CPU-side visibility flag for incremental delete
    
    // Set identity transform
    void setIdentity() {
        transform[0] = 1; transform[1] = 0; transform[2] = 0; transform[3] = 0;
        transform[4] = 0; transform[5] = 1; transform[6] = 0; transform[7] = 0;
        transform[8] = 0; transform[9] = 0; transform[10] = 1; transform[11] = 0;
    }
    
    // Set transform from position, rotation (euler degrees), scale
    void setTransform(const Vec3& pos, const Vec3& rot_deg, const Vec3& scale) {
        float rx = rot_deg.x * 3.14159265f / 180.0f;
        float ry = rot_deg.y * 3.14159265f / 180.0f;
        float rz = rot_deg.z * 3.14159265f / 180.0f;
        
        float cx = cosf(rx), sx = sinf(rx);
        float cy = cosf(ry), sy = sinf(ry);
        float cz = cosf(rz), sz = sinf(rz);
        
        // Rotation matrix (ZYX order) * scale
        transform[0] = scale.x * (cy * cz);
        transform[1] = scale.y * (sx * sy * cz - cx * sz);
        transform[2] = scale.z * (cx * sy * cz + sx * sz);
        transform[3] = pos.x;
        
        transform[4] = scale.x * (cy * sz);
        transform[5] = scale.y * (sx * sy * sz + cx * cz);
        transform[6] = scale.z * (cx * sy * sz - sx * cz);
        transform[7] = pos.y;
        
        transform[8] = scale.x * (-sy);
        transform[9] = scale.y * (sx * cy);
        transform[10] = scale.z * (cx * cy);
        transform[11] = pos.z;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// ACCELERATION STRUCTURE MANAGER CLASS
// ═══════════════════════════════════════════════════════════════════════════

class OptixAccelManager {
public:
    OptixAccelManager() = default;
    ~OptixAccelManager() { cleanup(); }
    
    // Initialize with OptiX context and stream
    void initialize(OptixDeviceContext ctx, CUstream str, 
                    OptixProgramGroup hit_pg, OptixProgramGroup hit_shadow_pg,
                    OptixProgramGroup hair_hit_pg, OptixProgramGroup hair_shadow_pg);
    
    // Mark that the scene structure (object list) has changed
    void markTopologyDirty() { m_topology_dirty = true; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // BLAS MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════
    
    // Build BLAS for a mesh, returns mesh_id
    int buildMeshBLAS(const MeshGeometry& geometry);
    
    // Build BLAS for hair curves
    int buildCurveBLAS(const CurveGeometry& geometry);
    
    // Update curve material parameters without rebuild
    void updateCurveMaterial(int curve_id, const GpuHairMaterial& material);

    // Fast update for hair curves (Refit)
    /**
     * @brief Fast update of curve acceleration structure (Refit)
     * @param d_external_vertices Optional new vertex buffer (interleaved supported via stride)
     * @param d_external_widths Optional new width buffer
     * @param vertex_stride Optional custom stride (default 0 means sizeof(float3))
     * @param width_stride Optional custom stride (default 0 means sizeof(float))
     */
    bool refitCurveBLAS(int curve_id, bool sync = true, 
                       CUdeviceptr d_external_vertices = 0, 
                       CUdeviceptr d_external_widths = 0,
                       size_t vertex_stride = 0,
                       size_t width_stride = 0);

    // Update BLAS vertices and refit (for transform updates)
    // Returns true if successful, false if refit not possible (needs full rebuild)
    bool updateMeshBLAS(int mesh_id, const MeshGeometry& geometry, bool skipCpuUpload = false, bool sync = true);
    
    // Partial update: Upload only a range of vertices/normals
    void uploadMeshVerticesPartial(int mesh_id, 
                                   const std::vector<float3>& vertices, 
                                   const std::vector<float3>& normals, 
                                   size_t offset_in_vertices);

    // Trigger BLAS refit after partial uploads
    bool refitMeshBLAS(int mesh_id, bool sync = true);
    
    // Update ALL BLAS structures from triangle list (for gizmo transform)
    // Updates all BLAS vertex buffers (Heavy: for deformation)
    // Update ALL BLAS structures from triangle list (for gizmo transform)
    // Updates all BLAS vertex buffers (Heavy: for deformation)
    void updateAllBLASFromTriangles(const std::vector<std::shared_ptr<Hittable>>& objects,
                                    const std::vector<Matrix4x4>& boneMatrices = {});
    
    // Updates only Instance Transforms (Lightweight: for rigid motion)
    void syncInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects, bool force_rebuild_cache = false);

    // Get BLAS by mesh_id
    const MeshBLAS* getBLAS(int mesh_id) const;
    
    // Find BLAS ID by name and material (returns -1 if not found)
    int findBLAS(const std::string& name, int material_id) const;
    
    // Static helper: Group triangles by nodeName for per-mesh BLAS building
    static std::vector<MeshData> groupTrianglesByMesh(
        const std::vector<std::shared_ptr<Triangle>>& triangles);
    
    // ═══════════════════════════════════════════════════════════════════════
    // INSTANCE MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════
    
    // Add instance, returns instance_id
    int addInstance(int blas_id, const float transform[12], int material_id, InstanceType type = InstanceType::Mesh, const std::string& name = "", void* source_hittable = nullptr);
    
    // Update instance transform (fast - only TLAS rebuild needed)
    void updateInstanceTransform(int instance_id, const float transform[12]);
    
    // Remove instance (marks for removal, call compactInstances to clean up)
    void removeInstance(int instance_id);
    
    // ═══════════════════════════════════════════════════════════════════════
    // INCREMENTAL UPDATES (Fast - No BLAS rebuild!)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Hide instance (for delete - just sets visibility_mask = 0)
    void hideInstance(int instance_id);
    
    // Show instance (restore visibility)
    void showInstance(int instance_id);
    void showAllInstances();
    // Hide all instances with matching node name
    void hideInstancesByNodeName(const std::string& nodeName);
    
    // Clone instance (for duplicate - copies instance, reuses BLAS)
    // Returns new instance_id
    int cloneInstance(int source_instance_id, const std::string& newNodeName);
    
    // Clone all instances with matching node name
    // Returns vector of new instance IDs  
    std::vector<int> cloneInstancesByNodeName(const std::string& sourceName, const std::string& newName);
    
    // ═══════════════════════════════════════════════════════════════════════
    // TLAS MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════
    
    // Build/rebuild TLAS from all instances
    void buildTLAS();
    
    // Update TLAS (refit only - faster if topology unchanged)
    void updateTLAS();
    
    // Get final traversable handle for rendering
    OptixTraversableHandle getTraversableHandle() const { return tlas_handle; }
    
    // ═══════════════════════════════════════════════════════════════════════
    // SBT MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════
    
    // Build SBT records for all meshes
    void buildSBT(const std::vector<GpuMaterial>& materials,
                  const std::vector<OptixGeometryData::TextureBundle>& textures,
                  const std::vector<OptixGeometryData::VolumetricInfo>& volumetrics);
                 
    // Synchronize material properties (emission, textures) into existing SBT without full rebuild
    void syncSBTMaterialData(const std::vector<GpuMaterial>& materials, bool sync_terrain = true);

    // Apply wind deformation to foliage meshes (Kernel Launch + BLAS Refit)
    void applyWindDeformation(int mesh_id, const Vec3& direction, float strength, float speed, float time);
    
    // Get SBT for rendering
    const OptixShaderBindingTable& getSBT() const { return sbt; }
    
    // Get hitgroup records for external modification
    std::vector<SbtRecord<HitGroupData>>& getHitGroupRecords() { return hitgroup_records; }
    
    // Upload modified hitgroup records to GPU
    void uploadHitGroupRecords();
    
    // Initialize PTX module
    void initFoliageKernel();
    
    // ═══════════════════════════════════════════════════════════════════════
    // CLEANUP
    // ═══════════════════════════════════════════════════════════════════════
    
    void cleanup();
    void clearInstances();  // Keep BLAS, clear instances and TLAS
    void clearCurves();     // Clear all curve BLAS and their instances
    
    // ═══════════════════════════════════════════════════════════════════════
    // STATS
    // ═══════════════════════════════════════════════════════════════════════
    


    size_t getMeshCount() const { return mesh_blas_list.size(); }
    size_t getInstanceCount() const { return instances.size(); }
    size_t getCurveInstanceCount() const {
        size_t count = 0;
        for (const auto& inst : instances) if (inst.type == InstanceType::Curve) count++;
        return count;
    }
    const std::vector<SceneInstance>& getInstances() const { return instances; }
    bool isBuilt() const { return tlas_handle != 0; }
    
    // Get mesh name by SBT index (for GPU picking)
    std::string getMeshNameByIndex(int mesh_idx) const {
        if (mesh_idx >= 0 && mesh_idx < static_cast<int>(mesh_blas_list.size())) {
            return mesh_blas_list[mesh_idx].original_name;  // Use original_name (node name)
        }
        return "";
    }
    
    // Callback for HUD messages
    // int argument is for message type: 0=Info, 1=Warning, 2=Error
    void setMessageCallback(std::function<void(const std::string&, int)> callback) {
        m_messageCallback = callback;
    }

private:
    std::function<void(const std::string&, int)> m_messageCallback;
    // OptiX context
    OptixDeviceContext context = nullptr;
    CUstream stream = nullptr;
    OptixProgramGroup hit_program_group = nullptr;
    OptixProgramGroup hit_shadow_program_group = nullptr;
    OptixProgramGroup hair_hit_program_group = nullptr;
    OptixProgramGroup hair_shadow_program_group = nullptr;
    
    // BLAS storage
    std::vector<MeshBLAS> mesh_blas_list;
    std::vector<CurveBLAS> curve_blas_list;
    
    // Instance storage
    std::vector<SceneInstance> instances;
    std::vector<int> free_instance_slots;  // Recycled instance IDs
    
    // TLAS
    OptixTraversableHandle tlas_handle = 0;
    CUdeviceptr d_instances = 0;            // OptixInstance array on GPU
    CUdeviceptr d_tlas_output = 0;          // TLAS output buffer
    CUdeviceptr d_tlas_temp = 0;            // TLAS temp buffer
    size_t tlas_output_size = 0;
    bool tlas_needs_rebuild = true;
    
    // Topology cache for animation performance
    bool m_topology_dirty = true;
    std::vector<std::shared_ptr<Triangle>> m_cached_triangles;
    struct MeshGroupCache {
        int blas_idx;
        std::vector<int> triangle_indices;
        std::vector<int> instance_ids; // Map of TLAS instances using this BLAS
    };
    std::vector<MeshGroupCache> m_cached_groups;
    
    // Mapping for fast transform sync
    struct InstanceTransformCache {
        int instance_id;
        std::string node_name;
        std::shared_ptr<Triangle> representative_tri; // Fetch transform from here
    };
    std::vector<InstanceTransformCache> m_instance_sync_cache;
    
    // SBT
    OptixShaderBindingTable sbt = {};
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;
    CUdeviceptr d_hitgroup_records = 0;
    size_t m_last_hitgroup_size = 0; // Track size to avoid reallocations
    
    // Foliage Deformation PTX Module
    CUmodule foliage_module = 0;
    CUfunction foliage_kernel = 0;
    bool foliage_kernel_initialized = false;
    
    // Global Bone Matrices (Persistent buffer to avoid per-frame allocation)
    CUdeviceptr d_globalBoneMatrices = 0;
    size_t globalBoneMatrices_capacity = 0; // In bytes
    
    // Helper: Build single GAS from geometry
    OptixTraversableHandle buildGAS(
        CUdeviceptr d_verts, size_t vert_count,
        CUdeviceptr d_idxs, size_t idx_count,
        CUdeviceptr& d_output,
        size_t& output_size
    );
};

