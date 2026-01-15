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
#include <OptixWrapper.h>
#include <functional> // Added for std::function
#include "PrincipledBSDF.h"
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

// Per-mesh geometry data for BLAS building
struct MeshGeometry {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<float3> tangents;
    int material_id = 0;
    std::string mesh_name;
    
    // Skinning Data (Optional)
    std::vector<int4> boneIndices;
    std::vector<float4> boneWeights;
};

// Per-mesh data for grouping triangles by nodeName
// Used to identify which triangles belong to the same mesh/object
struct MeshData {
    std::string mesh_name;                  // Unique name (e.g. "Car_mat_0")
    std::string original_name;              // Original node name (e.g. "Car")
    std::vector<int> triangle_indices;      // Indices into global triangle list
    int material_id = 0;                    // Primary material ID for this mesh
};

// Per-mesh Bottom-Level Acceleration Structure
struct MeshBLAS {
    OptixTraversableHandle handle = 0;      // BLAS traversable handle
    CUdeviceptr d_vertices = 0;             // GPU vertex buffer
    CUdeviceptr d_indices = 0;              // GPU index buffer
    CUdeviceptr d_normals = 0;              // GPU normal buffer
    CUdeviceptr d_uvs = 0;                  // GPU UV buffer
    CUdeviceptr d_tangents = 0;             // GPU tangent buffer
    CUdeviceptr d_gas_output = 0;           // GAS output buffer
    size_t gas_output_size = 0;             // GAS output buffer size (for refit)
    size_t vertex_count = 0;
    size_t index_count = 0;
    int sbt_offset = 0;                     // SBT hit group offset
    int material_id = 0;                    // Material ID for this mesh
    std::string mesh_name;                  // Unique name (e.g. "Car_mat_0")
    std::string original_name;              // Base node name (e.g. "Car")
    
    // GPU Skinning Buffers
    CUdeviceptr d_bindPoses = 0;            // Original Bind Pose Vertices
    CUdeviceptr d_bindNormals = 0;          // Original Bind Pose Normals
    CUdeviceptr d_boneIndices = 0;          // Bone Indices (int4)
    CUdeviceptr d_boneWeights = 0;          // Bone Weights (float4)
    bool hasSkinningData = false;
    
    void cleanup() {
        if (d_vertices) { cudaFree(reinterpret_cast<void*>(d_vertices)); d_vertices = 0; }
        if (d_indices) { cudaFree(reinterpret_cast<void*>(d_indices)); d_indices = 0; }
        if (d_normals) { cudaFree(reinterpret_cast<void*>(d_normals)); d_normals = 0; }
        if (d_uvs) { cudaFree(reinterpret_cast<void*>(d_uvs)); d_uvs = 0; }
        if (d_tangents) { cudaFree(reinterpret_cast<void*>(d_tangents)); d_tangents = 0; }
        if (d_gas_output) { cudaFree(reinterpret_cast<void*>(d_gas_output)); d_gas_output = 0; }
        
        // Free Skinning Buffers
        if (d_bindPoses) { cudaFree(reinterpret_cast<void*>(d_bindPoses)); d_bindPoses = 0; }
        if (d_bindNormals) { cudaFree(reinterpret_cast<void*>(d_bindNormals)); d_bindNormals = 0; }
        if (d_boneIndices) { cudaFree(reinterpret_cast<void*>(d_boneIndices)); d_boneIndices = 0; }
        if (d_boneWeights) { cudaFree(reinterpret_cast<void*>(d_boneWeights)); d_boneWeights = 0; }
        
        handle = 0;
    }
};

// Per-instance data for TLAS
struct SceneInstance {
    int mesh_id = -1;                       // Index into mesh_blas_list
    int material_id = 0;                    // Material index for SBT
    float transform[12] = {                 // 3x4 row-major transform matrix
        1, 0, 0, 0,                         // Row 0: scale.x * right + tx
        0, 1, 0, 0,                         // Row 1: scale.y * up + ty
        0, 0, 1, 0                          // Row 2: scale.z * forward + tz
    };
    uint32_t visibility_mask = 255;         // Ray visibility mask (0 = invisible)
    uint32_t sbt_offset = 0;                // Computed SBT offset
    std::string node_name;                  // Node name for lookup
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
    void initialize(OptixDeviceContext ctx, CUstream str, OptixProgramGroup hit_pg);
    
    // ═══════════════════════════════════════════════════════════════════════
    // BLAS MANAGEMENT
    // ═══════════════════════════════════════════════════════════════════════
    
    // Build BLAS for a mesh, returns mesh_id
    int buildMeshBLAS(const MeshGeometry& geometry);
    
    // Update BLAS vertices and refit (for transform updates)
    // Returns true if successful, false if refit not possible (needs full rebuild)
    bool updateMeshBLAS(int mesh_id, const MeshGeometry& geometry, bool skipCpuUpload = false, bool sync = true);
    
    // Update ALL BLAS structures from triangle list (for gizmo transform)
    // Updates all BLAS vertex buffers (Heavy: for deformation)
    // Update ALL BLAS structures from triangle list (for gizmo transform)
    // Updates all BLAS vertex buffers (Heavy: for deformation)
    void updateAllBLASFromTriangles(const std::vector<std::shared_ptr<Hittable>>& objects,
                                    const std::vector<Matrix4x4>& boneMatrices = {});
    
    // Updates only Instance Transforms (Lightweight: for rigid motion)
    void syncInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects);

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
    int addInstance(int mesh_id, const float transform[12], int material_id, const std::string& name = "");
    
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
    
    // Get SBT for rendering
    const OptixShaderBindingTable& getSBT() const { return sbt; }
    
    // Get hitgroup records for external modification
    std::vector<SbtRecord<HitGroupData>>& getHitGroupRecords() { return hitgroup_records; }
    
    // Upload modified hitgroup records to GPU
    void uploadHitGroupRecords();
    
    // ═══════════════════════════════════════════════════════════════════════
    // CLEANUP
    // ═══════════════════════════════════════════════════════════════════════
    
    void cleanup();
    void clearInstances();  // Keep BLAS, clear instances and TLAS
    
    // ═══════════════════════════════════════════════════════════════════════
    // STATS
    // ═══════════════════════════════════════════════════════════════════════
    


    size_t getMeshCount() const { return mesh_blas_list.size(); }
    size_t getInstanceCount() const { return instances.size(); }
    const std::vector<SceneInstance>& getInstances() const { return instances; }
    bool isBuilt() const { return tlas_handle != 0; }
    
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
    
    // BLAS storage
    std::vector<MeshBLAS> mesh_blas_list;
    
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
    
    // SBT
    OptixShaderBindingTable sbt = {};
    std::vector<SbtRecord<HitGroupData>> hitgroup_records;
    CUdeviceptr d_hitgroup_records = 0;
    
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
