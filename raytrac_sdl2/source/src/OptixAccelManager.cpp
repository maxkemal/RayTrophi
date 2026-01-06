#include "OptixAccelManager.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <iostream>
#include <algorithm>
#include "TerrainManager.h"
#include "Texture.h"
#include "Material.h"


// ═══════════════════════════════════════════════════════════════════════════
// OPTIX CHECK MACRO
// ═══════════════════════════════════════════════════════════════════════════

#define OPTIX_CHECK_ACCEL(call)                                                 \
    do {                                                                        \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS) {                                             \
            std::string msg = "[OptixAccelManager] OptiX call failed: " +       \
                           std::to_string(res);                                 \
            SCENE_LOG_ERROR(msg);                                               \
            if (m_messageCallback) m_messageCallback(msg, 2);                   \
            return;                                                             \
        }                                                                       \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

void OptixAccelManager::initialize(OptixDeviceContext ctx, CUstream str, OptixProgramGroup hit_pg) {
    context = ctx;
    stream = str;
    hit_program_group = hit_pg;
    // SCENE_LOG_INFO("[OptixAccelManager] Initialized with OptiX context");
}

// ═══════════════════════════════════════════════════════════════════════════
// MESH GROUPING - Group triangles by nodeName for per-mesh BLAS building
// ═══════════════════════════════════════════════════════════════════════════

std::vector<MeshData> OptixAccelManager::groupTrianglesByMesh(
    const std::vector<std::shared_ptr<Triangle>>& triangles) 
{
    std::unordered_map<std::string, MeshData> mesh_map;
    
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tri = triangles[i];
        if (!tri) continue;
        
        std::string base_name = tri->getNodeName();
        if (base_name.empty()) base_name = "default_mesh";
        
        int mat_id = tri->getMaterialID();
        
        // Critical Fix: Group by Name AND Material ID
        // This ensures parts with different materials are split into separate BLAS
        std::string unique_key = base_name + "_mat_" + std::to_string(mat_id);
        
        auto& mesh = mesh_map[unique_key];
        if (mesh.mesh_name.empty()) {
            mesh.mesh_name = unique_key; // Use unique name to distinguish sub-meshes
            mesh.original_name = base_name; // Store original name for instance mapping
            mesh.material_id = mat_id;
        }
        mesh.triangle_indices.push_back(static_cast<int>(i));
    }
    
    // Convert map to vector
    std::vector<MeshData> result;
    result.reserve(mesh_map.size());
    for (auto& [name, data] : mesh_map) {
        result.push_back(std::move(data));
    }
    
    // Sort by name to ensure deterministic order (Stable Mesh IDs)
    std::sort(result.begin(), result.end(), [](const MeshData& a, const MeshData& b) {
        return a.mesh_name < b.mesh_name;
    });

   // SCENE_LOG_INFO("[OptixAccelManager] Grouped " + std::to_string(triangles.size()) + 
    //               " triangles into " + std::to_string(result.size()) + " meshes");
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// BLAS BUILDING
// ═══════════════════════════════════════════════════════════════════════════

int OptixAccelManager::buildMeshBLAS(const MeshGeometry& geometry) {
    if (geometry.vertices.empty() || geometry.indices.empty()) {
        std::string msg = "[OptixAccelManager] Cannot build BLAS: empty geometry";
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        return -1;
    }
    
    MeshBLAS blas;
    blas.mesh_name = geometry.mesh_name;
    blas.material_id = geometry.material_id;  // Store material ID for SBT
    blas.vertex_count = geometry.vertices.size();
    blas.index_count = geometry.indices.size();
    
    // Upload vertices
    size_t v_size = geometry.vertices.size() * sizeof(float3);
    cudaMalloc(reinterpret_cast<void**>(&blas.d_vertices), v_size);
    cudaMemcpy(reinterpret_cast<void*>(blas.d_vertices), geometry.vertices.data(), v_size, cudaMemcpyHostToDevice);
    
    // Upload indices
    size_t i_size = geometry.indices.size() * sizeof(uint3);
    cudaMalloc(reinterpret_cast<void**>(&blas.d_indices), i_size);
    cudaMemcpy(reinterpret_cast<void*>(blas.d_indices), geometry.indices.data(), i_size, cudaMemcpyHostToDevice);
    
    // Upload normals
    if (!geometry.normals.empty()) {
        size_t n_size = geometry.normals.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_normals), n_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_normals), geometry.normals.data(), n_size, cudaMemcpyHostToDevice);
    }
    
    // Upload UVs
    if (!geometry.uvs.empty()) {
        size_t uv_size = geometry.uvs.size() * sizeof(float2);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_uvs), uv_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_uvs), geometry.uvs.data(), uv_size, cudaMemcpyHostToDevice);
    }
    
    // Upload tangents
    if (!geometry.tangents.empty()) {
        size_t t_size = geometry.tangents.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_tangents), t_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_tangents), geometry.tangents.data(), t_size, cudaMemcpyHostToDevice);
    }

    // Upload Skinning Data
    if (!geometry.boneIndices.empty() && !geometry.boneWeights.empty()) {
        size_t bi_size = geometry.boneIndices.size() * sizeof(int4);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_boneIndices), bi_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_boneIndices), geometry.boneIndices.data(), bi_size, cudaMemcpyHostToDevice);

        size_t bw_size = geometry.boneWeights.size() * sizeof(float4);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_boneWeights), bw_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_boneWeights), geometry.boneWeights.data(), bw_size, cudaMemcpyHostToDevice);
        
        // Also allocate/copy Bind Pose data (initial vertex/normal state)
        // Bind poses should be copied from the 'vertices' and 'normals' at build time
        
        size_t v_size = geometry.vertices.size() * sizeof(float3);
        cudaMalloc(reinterpret_cast<void**>(&blas.d_bindPoses), v_size);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_bindPoses), geometry.vertices.data(), v_size, cudaMemcpyHostToDevice);

        if (!geometry.normals.empty()) {
            size_t n_size = geometry.normals.size() * sizeof(float3);
            cudaMalloc(reinterpret_cast<void**>(&blas.d_bindNormals), n_size);
            cudaMemcpy(reinterpret_cast<void*>(blas.d_bindNormals), geometry.normals.data(), n_size, cudaMemcpyHostToDevice);
        }

        blas.hasSkinningData = true;
    }
    
    // Build GAS for this mesh
    blas.handle = buildGAS(blas.d_vertices, blas.vertex_count,
                           blas.d_indices, blas.index_count,
                           blas.d_gas_output, blas.gas_output_size);
    
    if (blas.handle == 0) {
        std::string msg = "[OptixAccelManager] Failed to build BLAS for: " + geometry.mesh_name;
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        blas.cleanup();
        return -1;
    }
    
    // Assign SBT offset (each mesh gets one SBT record)
    blas.sbt_offset = static_cast<int>(mesh_blas_list.size());
    
    int mesh_id = static_cast<int>(mesh_blas_list.size());
    mesh_blas_list.push_back(std::move(blas));
    
    // SCENE_LOG_INFO("[OptixAccelManager] Built BLAS #" + std::to_string(mesh_id) + 
    //                " for '" + geometry.mesh_name + "' (" + 
    //                std::to_string(geometry.vertices.size()) + " verts, " +
    //                std::to_string(geometry.indices.size()) + " tris)");
    
    return mesh_id;
}

const MeshBLAS* OptixAccelManager::getBLAS(int mesh_id) const {
    if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_blas_list.size())) {
        return nullptr;
    }
    return &mesh_blas_list[mesh_id];
}

bool OptixAccelManager::updateMeshBLAS(int mesh_id, const MeshGeometry& geometry, bool skipCpuUpload, bool sync) {
    if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_blas_list.size())) {
        return false;
    }
    
    MeshBLAS& blas = mesh_blas_list[mesh_id];
    
    // Check if vertex count matches (required for refit)
    if (geometry.vertices.size() != blas.vertex_count) {
        std::string msg = "[OptixAccelManager] Vertex count mismatch for BLAS update";
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        return false;
    }
    
    // Update vertex buffer on GPU (Only if not skipped)
    if (!skipCpuUpload) {
        size_t v_size = geometry.vertices.size() * sizeof(float3);
        cudaMemcpy(reinterpret_cast<void*>(blas.d_vertices), geometry.vertices.data(), v_size, cudaMemcpyHostToDevice);

        // Update normals if present
        if (!geometry.normals.empty() && blas.d_normals) {
            size_t n_size = geometry.normals.size() * sizeof(float3);
            cudaMemcpy(reinterpret_cast<void*>(blas.d_normals), geometry.normals.data(), n_size, cudaMemcpyHostToDevice);
        }
    }
    
    // Refit GAS with OPTIX_BUILD_OPERATION_UPDATE
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &blas.d_vertices;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(blas.vertex_count);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.indexBuffer = blas.d_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(blas.index_count);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;  // Refit, not rebuild
    
    OptixAccelBufferSizes buffer_sizes;
    OptixResult res = optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes);
    if (res != OPTIX_SUCCESS) {
        return false;
    }
    
    // Use temporary buffer for update
    CUdeviceptr d_temp;
    cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempUpdateSizeInBytes);
    
    res = optixAccelBuild(
        context, stream, &accel_options, &build_input, 1,
        d_temp, buffer_sizes.tempUpdateSizeInBytes,
        blas.d_gas_output, blas.gas_output_size,  // Use stored size
        &blas.handle, nullptr, 0
    );
    
    cudaFree(reinterpret_cast<void*>(d_temp));
    
    if (res != OPTIX_SUCCESS) {
        std::string msg = "[OptixAccelManager] BLAS refit failed: " + std::to_string(res);
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        return false;
    }
    
    if (sync) {
        cudaStreamSynchronize(stream);
    }
    return true;
}

// ----------------------------------------------------------------------------
// NEW: Lightweight Transform Sync (No Vertex Upload)
// ----------------------------------------------------------------------------
void OptixAccelManager::syncInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!context) return;
    
    // 1. Collect all Triangles
    std::vector<std::shared_ptr<Triangle>> triangles;
    for (const auto& obj : objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            triangles.push_back(tri);
        } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
            // Traverse... (Simplified: assume flattened list or use existing mesh grouping)
            // Actually, we need to map objects to meshes.
            // 'triangles' vector in 'groupTrianglesByMesh' was simple.
            // Re-doing grouping is slow.
            // BETTER: Iterate 'mesh_blas_list' and find one representative triangle?
            // We don't store pointer to Triangles in MeshBLAS.
            // We DO store 'mesh_name'.
            //
            // Optimization: Iterate the input 'objects' (which are Triangles), 
            // identify which mesh they belong to, and update that mesh's transform.
            // Since all triangles in a mesh share the same transform, finding the first one is enough.
        }
    }
    
    // Optimized approach: 
    // We already have 'instances' which store 'mesh_id'.
    // We have 'mesh_blas_list' which has 'original_name'.
    // We can iterate 'objects' (CPU Hittables), find their Name + Material, find matching Instance?
    // 
    // Or, we re-use the grouping logic but ONLY for 1 triangle per group?
    // 
    // Let's use the same logic as updateAllBLASFromTriangles but skip vertex collection.
    
    std::vector<std::shared_ptr<Triangle>> all_tris;
    all_tris.reserve(objects.size());
    
    // Flatten
    // (Assuming objects are triangles for now, or we support BVH traversal)
    // For simplicity, let's assume objects are flat or we use the helper logic.
    // Actually, 'updateAllBLASFromTriangles' takes 'objects'.
    
    for (const auto& obj : objects) {
         if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
             all_tris.push_back(tri);
         }
         // TODO: Handle BVH children if needed, but usually 'objects' passed here is the flat list or we flatten it.
    }
    
    // Group by Mesh (Lightweight: only need 1 tri per mesh to get transform)
    std::unordered_map<std::string, Matrix4x4> transform_map;
    
    for (const auto& tri : all_tris) {
        std::string base_name = tri->getNodeName();
        if (base_name.empty()) base_name = "default_mesh";
        // Convert to key used in grouping (Name + MatID)
        // Wait, instances are split by material.
        // We need to update ALL instances sharing the same Node Name (if rigid transform).
        // So we just need Map<NodeName, Matrix>.
        
        if (transform_map.find(base_name) == transform_map.end()) {
             transform_map[base_name] = tri->getTransformMatrix();
        }
    }
    
    // Update Instances
    for (size_t i = 0; i < instances.size(); ++i) {
        if (instances[i].mesh_id < 0 || !instances[i].visible) continue;  // Skip invalid/hidden
        
        // CRITICAL: Use instance's own node_name (supports cloned instances like "Car_1")
        // NOT the BLAS's original_name (which is "Car" for all clones sharing that BLAS)
        std::string name = instances[i].node_name;
        
        // Handle material suffix - strip "_mat_X" to get base name for lookup
        size_t mat_pos = name.find("_mat_");
        if (mat_pos != std::string::npos) {
            name = name.substr(0, mat_pos);
        }
        
        if (transform_map.count(name)) {
            Matrix4x4 m = transform_map[name];
            float t[12];
            t[0] = m.m[0][0]; t[1] = m.m[0][1]; t[2] = m.m[0][2]; t[3] = m.m[0][3];
            t[4] = m.m[1][0]; t[5] = m.m[1][1]; t[6] = m.m[1][2]; t[7] = m.m[1][3];
            t[8] = m.m[2][0]; t[9] = m.m[2][1]; t[10] = m.m[2][2]; t[11] = m.m[2][3];
            
            updateInstanceTransform(static_cast<int>(i), t);
        }
    }
    
    tlas_needs_rebuild = true;
}

void OptixAccelManager::updateAllBLASFromTriangles(const std::vector<std::shared_ptr<Hittable>>& objects, const std::vector<Matrix4x4>& boneMatrices) {
    if (!context) return;
    
    // Flatten objects to triangles
    std::vector<std::shared_ptr<Triangle>> triangles;
    triangles.reserve(objects.size());
    for (const auto& obj : objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri) triangles.push_back(tri);
    }
    
    // Group by mesh name (same as during initial build)
    std::vector<MeshData> mesh_groups = groupTrianglesByMesh(triangles);
    
    // Build name -> mesh_group map for fast lookup
    std::unordered_map<std::string, const MeshData*> name_to_group;
    for (const auto& group : mesh_groups) {
        name_to_group[group.mesh_name] = &group;
    }

    // Upload Bone Matrices (Reuse global buffer)
    CUdeviceptr d_boneMatricesPtr = 0;
    if (!boneMatrices.empty()) {
        size_t sz = boneMatrices.size() * 16 * sizeof(float);
        
        // Resize if needed
        if (sz > globalBoneMatrices_capacity || d_globalBoneMatrices == 0) {
            if (d_globalBoneMatrices) cudaFree(reinterpret_cast<void*>(d_globalBoneMatrices));
            cudaMalloc(reinterpret_cast<void**>(&d_globalBoneMatrices), sz);
            globalBoneMatrices_capacity = sz;
        }
        
        cudaMemcpy(reinterpret_cast<void*>(d_globalBoneMatrices), boneMatrices.data(), sz, cudaMemcpyHostToDevice);
        d_boneMatricesPtr = d_globalBoneMatrices;
    }
    
    int updated_count = 0;
    
    // For each existing BLAS, find matching mesh group by NAME and update
    for (size_t blas_idx = 0; blas_idx < mesh_blas_list.size(); ++blas_idx) {
        MeshBLAS& blas = mesh_blas_list[blas_idx];
        
        auto it = name_to_group.find(blas.mesh_name);
        if (it == name_to_group.end()) {
            // Mesh no longer exists in scene
            continue;
        }
        
        const MeshData& mesh = *(it->second);
        bool gpu_skinning_applied = false;

        // -----------------------------------------------------------------------
        // GPU SKINNING PATH
        // -----------------------------------------------------------------------
        if (blas.hasSkinningData && d_boneMatricesPtr) {
            launchSkinningKernel(
                reinterpret_cast<float3*>(blas.d_bindPoses),
                reinterpret_cast<float3*>(blas.d_bindNormals),
                reinterpret_cast<int4*>(blas.d_boneIndices),
                reinterpret_cast<float4*>(blas.d_boneWeights),
                reinterpret_cast<float*>(d_boneMatricesPtr),
                reinterpret_cast<float3*>(blas.d_vertices), // Output to BLAS directly
                reinterpret_cast<float3*>(blas.d_normals),  // Output to BLAS directly
                static_cast<int>(blas.vertex_count),
                static_cast<int>(boneMatrices.size()),
                stream
            );

            // Refit BLAS (Skip CPU Upload, NO Sync)
            // We just need a geometry object with correct size to pass the check inside updateMeshBLAS
            MeshGeometry dummyGeom;
            dummyGeom.vertices.resize(blas.vertex_count); 
            
            // Pass sync=false to batch updates
            if (updateMeshBLAS(static_cast<int>(blas_idx), dummyGeom, true, false)) {
                updated_count++;
            }
            gpu_skinning_applied = true;
        }

        // -----------------------------------------------------------------------
        // CPU PATH (Fallback)
        // -----------------------------------------------------------------------
        if (!gpu_skinning_applied) {
            MeshGeometry geom;
            geom.mesh_name = mesh.mesh_name;
            geom.material_id = mesh.material_id;

            for (int tri_idx : mesh.triangle_indices) {
                if (tri_idx < 0 || tri_idx >= static_cast<int>(triangles.size())) continue;
                auto& tri = triangles[tri_idx];
                
                uint32_t base = static_cast<uint32_t>(geom.vertices.size());
                
                // Get CURRENT vertex positions (skinned/transformed on CPU if GPU failed/inactive)
                Vec3 v0 = tri->getVertexPosition(0);
                Vec3 v1 = tri->getVertexPosition(1);
                Vec3 v2 = tri->getVertexPosition(2);
                geom.vertices.push_back({v0.x, v0.y, v0.z});
                geom.vertices.push_back({v1.x, v1.y, v1.z});
                geom.vertices.push_back({v2.x, v2.y, v2.z});
                
                geom.indices.push_back({base, base + 1, base + 2});
                
                Vec3 n0 = tri->getVertexNormal(0);
                Vec3 n1 = tri->getVertexNormal(1);
                Vec3 n2 = tri->getVertexNormal(2);
                geom.normals.push_back({n0.x, n0.y, n0.z});
                geom.normals.push_back({n1.x, n1.y, n1.z});
                geom.normals.push_back({n2.x, n2.y, n2.z});
            }
            
            // Pass sync=false to batch updates
            if (updateMeshBLAS(static_cast<int>(blas_idx), geom, false, false)) {
                updated_count++;
            }
        }
        
        // ═══════════════════════════════════════════════════════════════════════════
        // Update Instance Transform for this mesh
        // This is the key optimization: fast transform update without BLAS rebuild!
        // ═══════════════════════════════════════════════════════════════════════════
        if (!mesh.triangle_indices.empty()) {
            int first_tri_idx = mesh.triangle_indices[0];
            if (first_tri_idx >= 0 && first_tri_idx < static_cast<int>(triangles.size())) {
                const auto& tri = triangles[first_tri_idx];
                
                // Get current transform from triangle
                Matrix4x4 m = tri->getTransformMatrix();
                float transform[12];
                transform[0] = m.m[0][0]; transform[1] = m.m[0][1]; transform[2] = m.m[0][2]; transform[3] = m.m[0][3];
                transform[4] = m.m[1][0]; transform[5] = m.m[1][1]; transform[6] = m.m[1][2]; transform[7] = m.m[1][3];
                transform[8] = m.m[2][0]; transform[9] = m.m[2][1]; transform[10] = m.m[2][2]; transform[11] = m.m[2][3];
                
                // Find all instances using this BLAS and update them
                for (size_t i = 0; i < instances.size(); ++i) {
                    if (instances[i].mesh_id == static_cast<int>(blas_idx)) {
                        updateInstanceTransform(static_cast<int>(i), transform);
                    }
                }
            }
        }
    }
    
   // SCENE_LOG_INFO("[OptixAccelManager] Updated " + std::to_string(updated_count) + " BLAS structures");
    
    

    // Batch Sync at the end
    cudaStreamSynchronize(stream);
    
    // SCENE_LOG_INFO("[OptixAccelManager] Updated " + std::to_string(updated_count) + " BLAS structures");
    
    // Mark TLAS for rebuild since BLAS handles changed
    tlas_needs_rebuild = true;
}

// ═══════════════════════════════════════════════════════════════════════════
// INSTANCE MANAGEMENT
// ═══════════════════════════════════════════════════════════════════════════

int OptixAccelManager::addInstance(int mesh_id, const float transform[12], int material_id, const std::string& name) {
    if (mesh_id < 0 || mesh_id >= static_cast<int>(mesh_blas_list.size())) {
        SCENE_LOG_ERROR("[OptixAccelManager] Invalid mesh_id: " + std::to_string(mesh_id));
        return -1;
    }
    
    SceneInstance inst;
    inst.mesh_id = mesh_id;
    inst.material_id = material_id;
    inst.node_name = name;
    inst.sbt_offset = mesh_blas_list[mesh_id].sbt_offset;
    std::memcpy(inst.transform, transform, sizeof(float) * 12);
    
    int instance_id;
    if (!free_instance_slots.empty()) {
        instance_id = free_instance_slots.back();
        free_instance_slots.pop_back();
        instances[instance_id] = std::move(inst);
    } else {
        instance_id = static_cast<int>(instances.size());
        instances.push_back(std::move(inst));
    }
    
    tlas_needs_rebuild = true;
    return instance_id;
}

void OptixAccelManager::updateInstanceTransform(int instance_id, const float transform[12]) {
    if (instance_id < 0 || instance_id >= static_cast<int>(instances.size())) {
        return;
    }
    
    std::memcpy(instances[instance_id].transform, transform, sizeof(float) * 12);
    tlas_needs_rebuild = true;
}

void OptixAccelManager::removeInstance(int instance_id) {
    if (instance_id < 0 || instance_id >= static_cast<int>(instances.size())) {
        return;
    }
    
    instances[instance_id].mesh_id = -1;  // Mark as invalid
    free_instance_slots.push_back(instance_id);
    tlas_needs_rebuild = true;
}

// ═══════════════════════════════════════════════════════════════════════════
// INCREMENTAL UPDATES (Fast - No BLAS rebuild!)
// ═══════════════════════════════════════════════════════════════════════════

void OptixAccelManager::hideInstance(int instance_id) {
    if (instance_id < 0 || instance_id >= static_cast<int>(instances.size())) {
        return;
    }
    instances[instance_id].visible = false;
    instances[instance_id].visibility_mask = 0;  // OptiX will skip this instance
    tlas_needs_rebuild = true;  // Still need TLAS update, but very fast
}

void OptixAccelManager::showInstance(int instance_id) {
    if (instance_id < 0 || instance_id >= static_cast<int>(instances.size())) {
        return;
    }
    instances[instance_id].visible = true;
    instances[instance_id].visibility_mask = 255;
    tlas_needs_rebuild = true;
}

void OptixAccelManager::hideInstancesByNodeName(const std::string& nodeName) {
    for (size_t i = 0; i < instances.size(); ++i) {
        // Check if instance's node_name starts with nodeName (handles material suffix)
        if (instances[i].node_name == nodeName || 
            instances[i].node_name.find(nodeName + "_mat_") == 0) {
            hideInstance(static_cast<int>(i));
        }
    }
}

int OptixAccelManager::cloneInstance(int source_instance_id, const std::string& newNodeName) {
    if (source_instance_id < 0 || source_instance_id >= static_cast<int>(instances.size())) {
        return -1;
    }
    
    const auto& source = instances[source_instance_id];
    if (source.mesh_id < 0) return -1;  // Skip invalid instances
    
    SceneInstance newInst = source;  // Copy everything
    newInst.node_name = newNodeName;
    newInst.visible = true;
    newInst.visibility_mask = 255;
    
    int new_id;
    if (!free_instance_slots.empty()) {
        new_id = free_instance_slots.back();
        free_instance_slots.pop_back();
        instances[new_id] = std::move(newInst);
    } else {
        new_id = static_cast<int>(instances.size());
        instances.push_back(std::move(newInst));
    }
    
    tlas_needs_rebuild = true;
    return new_id;
}

std::vector<int> OptixAccelManager::cloneInstancesByNodeName(const std::string& sourceName, const std::string& newName) {
    std::vector<int> new_ids;
    
    // Find all instances matching source name (including material variants)
    for (size_t i = 0; i < instances.size(); ++i) {
        if (instances[i].mesh_id < 0 || !instances[i].visible) continue;  // Skip invalid/hidden
        
        // Match exact name or name_mat_X pattern
        std::string instName = instances[i].node_name;
        if (instName == sourceName || instName.find(sourceName + "_mat_") == 0) {
            // Derive new name preserving material suffix
            std::string suffix = "";
            if (instName.find("_mat_") != std::string::npos) {
                suffix = instName.substr(instName.find("_mat_"));
            }
            
            int new_id = cloneInstance(static_cast<int>(i), newName + suffix);
            if (new_id >= 0) {
                new_ids.push_back(new_id);
            }
        }
    }
    
    return new_ids;
}

// ═══════════════════════════════════════════════════════════════════════════
// TLAS BUILDING
// ═══════════════════════════════════════════════════════════════════════════

void OptixAccelManager::buildTLAS() {
    if (!context) {
        std::string msg = "[OptixAccelManager] Cannot build TLAS: not initialized";
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        return;
    }
    
    // Count valid instances
    std::vector<OptixInstance> optix_instances;
    optix_instances.reserve(instances.size());
    
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& inst = instances[i];
        if (inst.mesh_id < 0 || inst.mesh_id >= static_cast<int>(mesh_blas_list.size())) {
            continue;  // Skip invalid instances
        }
        
        const MeshBLAS& blas = mesh_blas_list[inst.mesh_id];
        if (blas.handle == 0) {
            continue;  // Skip if BLAS not built
        }
        
        OptixInstance optix_inst = {};
        
        // Copy transform (row-major 3x4)
        std::memcpy(optix_inst.transform, inst.transform, sizeof(float) * 12);
        
        optix_inst.instanceId = static_cast<unsigned int>(i);
        optix_inst.sbtOffset = blas.sbt_offset;
        optix_inst.visibilityMask = inst.visibility_mask;
        
        // DISABLE CULLING: Essential for double-sided materials and mirrored instances (negative scale)
        // We do NOT use OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING because it alters barycentrics 
        // (A-C-B order) while our shader reads attributes in A-B-C order, causing skewed interpolation.
        // Instead, we hit the backface and let the Shader flip the Normal.
        optix_inst.flags = OPTIX_INSTANCE_FLAG_DISABLE_TRIANGLE_FACE_CULLING;
        optix_inst.traversableHandle = blas.handle;
        
        optix_instances.push_back(optix_inst);
    }
    
    if (optix_instances.empty()) {
        std::string msg = "[OptixAccelManager] No valid instances for TLAS";
        SCENE_LOG_WARN(msg);
        // if (m_messageCallback) m_messageCallback(msg, 1); // Warn
        tlas_handle = 0;
        return;
    }
    
    // Upload instances to GPU (reuse buffer if size is same)
    size_t inst_size = optix_instances.size() * sizeof(OptixInstance);
    
    // Check if we can reuse the existing instance buffer
    static size_t last_inst_capacity = 0;
    if (!d_instances || inst_size > last_inst_capacity) {
        if (d_instances) cudaFree(reinterpret_cast<void*>(d_instances));
        cudaMalloc(reinterpret_cast<void**>(&d_instances), inst_size);
        last_inst_capacity = inst_size;
    }
    
    cudaMemcpy(reinterpret_cast<void*>(d_instances), optix_instances.data(), inst_size, cudaMemcpyHostToDevice);
    
    // Build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = d_instances;
    build_input.instanceArray.numInstances = static_cast<unsigned int>(optix_instances.size());
    
    // Build options & Optimization: Use UPDATE if handle exists and count matches
    bool can_update = (tlas_handle != 0) && (optix_instances.size() == instances.size()); 
    // Note: instances.size() might differ from last time if objects added/removed
    // Ideally we track 'last_num_instances'
    static size_t last_num_instances = 0;
    can_update = (tlas_handle != 0) && (optix_instances.size() == last_num_instances);
    last_num_instances = optix_instances.size();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = can_update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;
    
    // Compute memory requirements
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_CHECK_ACCEL(optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes));
    
    // Allocate buffers
    // Reuse temp buffer if large enough
    static size_t last_temp_size = 0;
    // For update, temp size might be different? usually smaller or same.
    size_t temp_size_needed = can_update ? buffer_sizes.tempUpdateSizeInBytes : buffer_sizes.tempSizeInBytes;
    
    if (!d_tlas_temp || temp_size_needed > last_temp_size) {
        if (d_tlas_temp) cudaFree(reinterpret_cast<void*>(d_tlas_temp));
        cudaMalloc(reinterpret_cast<void**>(&d_tlas_temp), temp_size_needed);
        last_temp_size = temp_size_needed;
    }
    
    if (tlas_output_size < buffer_sizes.outputSizeInBytes) {
        if (d_tlas_output) cudaFree(reinterpret_cast<void*>(d_tlas_output));
        cudaMalloc(reinterpret_cast<void**>(&d_tlas_output), buffer_sizes.outputSizeInBytes);
        tlas_output_size = buffer_sizes.outputSizeInBytes;
        // If reallocating output, we MUST do a full build
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    }
    
    // Build or Update TLAS
    OPTIX_CHECK_ACCEL(optixAccelBuild(
        context, stream, &accel_options, &build_input, 1,
        d_tlas_temp, temp_size_needed,
        d_tlas_output, buffer_sizes.outputSizeInBytes,
        &tlas_handle, nullptr, 0
    ));
    
    // NOTE: NO cudaStreamSynchronize here! Let GPU work asynchronously.
    // The next launch() or render call will naturally sync.
    // Removing this sync prevents UI freezes during TLAS updates.
    
    // NOTE: Don't free d_tlas_temp - reuse it next time for efficiency!
    
    tlas_needs_rebuild = false;
    
    //SCENE_LOG_INFO("[OptixAccelManager] Built TLAS with " + 
    //               std::to_string(optix_instances.size()) + " instances");
}

void OptixAccelManager::updateTLAS() {
    if (!tlas_needs_rebuild) return;
    
    // For now, full rebuild. Could optimize to use OPTIX_BUILD_OPERATION_UPDATE
    // if only transforms changed and instance count is same.
    buildTLAS();
}

// ═══════════════════════════════════════════════════════════════════════════
// SBT BUILDING
// ═══════════════════════════════════════════════════════════════════════════

void OptixAccelManager::buildSBT(const std::vector<GpuMaterial>& materials,
                                  const std::vector<OptixGeometryData::TextureBundle>& textures,
                                  const std::vector<OptixGeometryData::VolumetricInfo>& volumetrics) {
    if (!hit_program_group) {
        std::string msg = "[OptixAccelManager] Cannot build SBT: hit program group not set";
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        return;
    }
    
    hitgroup_records.clear();
    
    // One SBT record per mesh (BLAS)
    for (size_t mesh_idx = 0; mesh_idx < mesh_blas_list.size(); ++mesh_idx) {
        const MeshBLAS& blas = mesh_blas_list[mesh_idx];
        int mat_id = blas.material_id;  // Use actual material ID from mesh
        
        SbtRecord<HitGroupData> rec = {};
        
        // Geometry pointers
        rec.data.vertices = reinterpret_cast<float3*>(blas.d_vertices);
        rec.data.indices = reinterpret_cast<uint3*>(blas.d_indices);
        rec.data.normals = reinterpret_cast<float3*>(blas.d_normals);
        rec.data.uvs = reinterpret_cast<float2*>(blas.d_uvs);
        rec.data.tangents = reinterpret_cast<float3*>(blas.d_tangents);
        rec.data.has_normals = (blas.d_normals != 0);
        rec.data.has_uvs = (blas.d_uvs != 0);
        rec.data.has_tangents = (blas.d_tangents != 0);
        
        // Material ID - use actual material ID from mesh
        rec.data.material_id = mat_id;
        
        // Texture bundle (if available) - use material_id as index
        if (mat_id >= 0 && mat_id < static_cast<int>(textures.size())) {
            const auto& tex = textures[mat_id];
            rec.data.albedo_tex = tex.albedo_tex;
            rec.data.roughness_tex = tex.roughness_tex;
            rec.data.normal_tex = tex.normal_tex;
            rec.data.metallic_tex = tex.metallic_tex;
            rec.data.transmission_tex = tex.transmission_tex;
            rec.data.opacity_tex = tex.opacity_tex;
            rec.data.emission_tex = tex.emission_tex;
            rec.data.has_albedo_tex = tex.has_albedo_tex;
            rec.data.has_roughness_tex = tex.has_roughness_tex;
            rec.data.has_normal_tex = tex.has_normal_tex;
            rec.data.has_metallic_tex = tex.has_metallic_tex;
            rec.data.has_transmission_tex = tex.has_transmission_tex;
            rec.data.has_opacity_tex = tex.has_opacity_tex;
            rec.data.has_emission_tex = tex.has_emission_tex;
        }
        
        // Emission from material - use material_id as index
        if (mat_id >= 0 && mat_id < static_cast<int>(materials.size())) {
            rec.data.emission = materials[mat_id].emission;
        }
        
        // TERRAIN LAYER SYSTEM
        // Identify if this mesh belongs to a TerrainObject
        rec.data.is_terrain = 0;
        
        // Find terrain by name comparison (Simple & Effective)
        // TerrainManager uses names like "Terrain 1", "Terrain 2".
        // The mesh name in BLAS comes from Assimp or creation.
        // For our generated terrains, mesh_name is usually "Terrain_Mesh".
        // Let's iterate all terrains to find if this BLAS matches any terrain's mesh.
        // Since we don't store mesh pointer in BLAS, let's rely on name or material ID.
        
        // Better approach: Iterate TerrainManager's terrains.
        auto& terrains = TerrainManager::getInstance().getTerrains();
        
        for (auto& terrain : terrains) {
            if (terrain.material_id == mat_id) {
                rec.data.is_terrain = 1;
                
                // 1. Splat Map
                if (terrain.splatMap) {
                     if (!terrain.splatMap->is_gpu_uploaded) terrain.splatMap->upload_to_gpu();
                     rec.data.splat_map_tex = terrain.splatMap->get_cuda_texture();
                } else {
                     SCENE_LOG_WARN("[TERRAIN SBT] No splatMap for terrain '" + terrain.name + "'");
                }
                
                // 2. Layers - use dynamic_cast to access PrincipledBSDF texture properties
                for (int i = 0; i < 4; ++i) {
                    if (i < terrain.layers.size() && terrain.layers[i]) {
                        // Try PrincipledBSDF first (most common for surface materials)
                        PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(terrain.layers[i].get());
                        
                        if (pbsdf) {
                            // ALBEDO
                            if (pbsdf->albedoProperty.texture) {
                                 if (!pbsdf->albedoProperty.texture->is_gpu_uploaded) 
                                     pbsdf->albedoProperty.texture->upload_to_gpu();
                                 rec.data.layer_albedo_tex[i] = pbsdf->albedoProperty.texture->get_cuda_texture();
                            }
                            
                            // NORMAL
                            if (pbsdf->normalProperty.texture) {
                                 if (!pbsdf->normalProperty.texture->is_gpu_uploaded) 
                                     pbsdf->normalProperty.texture->upload_to_gpu();
                                 rec.data.layer_normal_tex[i] = pbsdf->normalProperty.texture->get_cuda_texture();
                            }
                            
                            // ROUGHNESS
                            if (pbsdf->roughnessProperty.texture) {
                                 if (!pbsdf->roughnessProperty.texture->is_gpu_uploaded) 
                                     pbsdf->roughnessProperty.texture->upload_to_gpu();
                                 rec.data.layer_roughness_tex[i] = pbsdf->roughnessProperty.texture->get_cuda_texture();
                            }
                        }
                        
                        // UV SCALE
                        if (i < terrain.layer_uv_scales.size()) {
                            rec.data.layer_uv_scale[i] = terrain.layer_uv_scales[i];
                        }
                    }
                }
                
                break; // Found the terrain for this mesh
            }
        }
        
        OPTIX_CHECK_ACCEL(optixSbtRecordPackHeader(hit_program_group, &rec));
        hitgroup_records.push_back(rec);
    }
    
    uploadHitGroupRecords();
}

void OptixAccelManager::uploadHitGroupRecords() {
    if (hitgroup_records.empty()) return;
    
    size_t sbt_size = hitgroup_records.size() * sizeof(SbtRecord<HitGroupData>);
    
    if (d_hitgroup_records) {
        cudaFree(reinterpret_cast<void*>(d_hitgroup_records));
    }
    cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records), sbt_size);
    cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records), hitgroup_records.data(), sbt_size, cudaMemcpyHostToDevice);
    
    sbt.hitgroupRecordBase = d_hitgroup_records;
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord<HitGroupData>);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_records.size());
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: BUILD GAS
// ═══════════════════════════════════════════════════════════════════════════

OptixTraversableHandle OptixAccelManager::buildGAS(
    CUdeviceptr d_verts, size_t vert_count,
    CUdeviceptr d_idxs, size_t idx_count,
    CUdeviceptr& d_output,
    size_t& output_size
) {
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &d_verts;
    build_input.triangleArray.numVertices = static_cast<unsigned int>(vert_count);
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    build_input.triangleArray.indexBuffer = d_idxs;
    build_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(idx_count);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    
    unsigned int flags = OPTIX_GEOMETRY_FLAG_NONE;
    build_input.triangleArray.flags = &flags;
    build_input.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes buffer_sizes;
    OptixResult res = optixAccelComputeMemoryUsage(context, &accel_options, &build_input, 1, &buffer_sizes);
    if (res != OPTIX_SUCCESS) {
        std::string msg = "[OptixAccelManager] Failed to compute GAS memory: " + std::to_string(res);
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        output_size = 0;
        return 0;
    }
    
    CUdeviceptr d_temp;
    cudaMalloc(reinterpret_cast<void**>(&d_temp), buffer_sizes.tempSizeInBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_output), buffer_sizes.outputSizeInBytes);
    output_size = buffer_sizes.outputSizeInBytes;  // Store for later refit
    
    OptixTraversableHandle handle = 0;
    res = optixAccelBuild(
        context, stream, &accel_options, &build_input, 1,
        d_temp, buffer_sizes.tempSizeInBytes,
        d_output, buffer_sizes.outputSizeInBytes,
        &handle, nullptr, 0
    );
    
    cudaFree(reinterpret_cast<void*>(d_temp));
    
    if (res != OPTIX_SUCCESS) {
        std::string msg = "[OptixAccelManager] Failed to build GAS: " + std::to_string(res);
        SCENE_LOG_ERROR(msg);
        if (m_messageCallback) m_messageCallback(msg, 2);
        cudaFree(reinterpret_cast<void*>(d_output));
        d_output = 0;
        output_size = 0;
        return 0;
    }
    
    cudaStreamSynchronize(stream);
    return handle;
}

// ═══════════════════════════════════════════════════════════════════════════
// CLEANUP
// ═══════════════════════════════════════════════════════════════════════════

void OptixAccelManager::cleanup() {
    // Cleanup BLAS
    for (auto& blas : mesh_blas_list) {
        blas.cleanup();
    }
    mesh_blas_list.clear();
    
    // Cleanup instances
    instances.clear();
    free_instance_slots.clear();
    
    // Cleanup TLAS
    if (d_instances) { cudaFree(reinterpret_cast<void*>(d_instances)); d_instances = 0; }
    if (d_tlas_output) { cudaFree(reinterpret_cast<void*>(d_tlas_output)); d_tlas_output = 0; }
    if (d_tlas_temp) { cudaFree(reinterpret_cast<void*>(d_tlas_temp)); d_tlas_temp = 0; }
    tlas_handle = 0;
    tlas_output_size = 0;
    
    // Cleanup Bone Matrices
    if (d_globalBoneMatrices) { cudaFree(reinterpret_cast<void*>(d_globalBoneMatrices)); d_globalBoneMatrices = 0; }
    globalBoneMatrices_capacity = 0;

    // Cleanup SBT
    if (d_hitgroup_records) { cudaFree(reinterpret_cast<void*>(d_hitgroup_records)); d_hitgroup_records = 0; }
    hitgroup_records.clear();
    sbt = {};
    
    tlas_needs_rebuild = true;
    
    SCENE_LOG_INFO("[OptixAccelManager] Cleaned up");
}

void OptixAccelManager::clearInstances() {
    instances.clear();
    free_instance_slots.clear();
    
    if (d_instances) { cudaFree(reinterpret_cast<void*>(d_instances)); d_instances = 0; }
    if (d_tlas_output) { cudaFree(reinterpret_cast<void*>(d_tlas_output)); d_tlas_output = 0; }
    tlas_handle = 0;
    tlas_output_size = 0;
    tlas_needs_rebuild = true;
}
