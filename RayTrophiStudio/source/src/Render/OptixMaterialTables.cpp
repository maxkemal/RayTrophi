/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Render/OptixMaterialTables.cpp
* =========================================================================
* Moved verbatim out of AssimpLoader.h (Faz 3). Behaviour unchanged; the only
* edits are the name, the loss of one level of indentation, and the includes
* that used to arrive through that header.
* =========================================================================
*/
#include "OptixMaterialTables.h"

#include "Triangle.h"
#include "Mesh.h"
#include "Material.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Dielectric.h"
#include "Volumetric.h"
#include "Texture.h"
#include "PBRMaterialSnapshot.h"
#include "sbt_data.h"
#include <globals.h>

#include <algorithm>
#include <cmath>

OptixGeometryData buildOptixMaterialTables(const std::vector<std::shared_ptr<Triangle>>& triangles) {
    OptixGeometryData data;
    
    // 1. Get Canonical Material List from MaterialManager
    auto& mgr = MaterialManager::getInstance();
    const auto& all_materials = mgr.getAllMaterials();
    
    // 2. Pre-allocate GPU buffers
    data.materials.resize(all_materials.size());
    data.textures.resize(all_materials.size());
    data.volumetric_info.resize(all_materials.size());
    
    // 3. Populate Material Data (O(M)) - Serial is fine (M is small)
    for (size_t i = 0; i < all_materials.size(); ++i) {
        const auto& mat = all_materials[i];
        if (!mat) continue; 
        
        // ... (Material Population Logic kept identical for safety, copied for context)
        GpuMaterial gpuMat = {}; 
        if (mat->type() == MaterialType::Volumetric) {
            // Volumetric defaults
            gpuMat.albedo = make_float3(1.0f, 1.0f, 1.0f);
            gpuMat.roughness = 1.0f;
            gpuMat.metallic = 0.0f;
            gpuMat.emission = make_float3(0.0f, 0.0f, 0.0f);
            gpuMat.ior = 1.0f;
            gpuMat.transmission = 0.0f;
            gpuMat.opacity = 1.0f;
        } else {
            // Helper to get CUDA texture object (uploads if needed)
            auto getCudaTex = [](const std::shared_ptr<Texture>& tex) -> cudaTextureObject_t {
                extern bool g_hasOptix; 
                if (tex && tex->is_loaded()) {
                     if (!tex->is_gpu_uploaded && g_hasOptix && isCudaTextureUploadAllowed()) {
                         tex->upload_to_gpu();
                     }
                     return tex->get_cuda_texture();
                }
                return 0;
            };

            // Calculate GpuMaterial properties from PrincipledBSDF
            if (mat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
                applyPBRMaterialSnapshotToGpuMaterial(snapshot, gpuMat);

                // Bindless Textures Population
                gpuMat.albedo_tex      = getCudaTex(pbsdf->albedoProperty.texture);
                gpuMat.normal_tex      = getCudaTex(pbsdf->normalProperty.texture);
                gpuMat.roughness_tex   = getCudaTex(pbsdf->roughnessProperty.texture);
                gpuMat.metallic_tex    = getCudaTex(pbsdf->metallicProperty.texture);
                gpuMat.specular_tex    = getCudaTex(pbsdf->specularProperty.texture);
                gpuMat.emission_tex    = getCudaTex(pbsdf->emissionProperty.texture);
                gpuMat.opacity_tex     = getCudaTex(pbsdf->opacityProperty.texture);
                gpuMat.transmission_tex= getCudaTex(pbsdf->transmissionProperty.texture);
                gpuMat.height_tex      = getCudaTex(pbsdf->heightProperty.texture); // Displacement
            } else if (mat->gpuMaterial) {
                gpuMat = *mat->gpuMaterial;
            } else {
                gpuMat.albedo = make_float3(0.8f, 0.8f, 0.8f);
                gpuMat.roughness = 0.5f;
                gpuMat.metallic = 0.0f;
                gpuMat.emission = make_float3(0.0f, 0.f, 0.f);
                gpuMat.ior = 1.5f;
                gpuMat.transmission = 0.0f;
                gpuMat.opacity = 1.0f;
            }
        }
        data.materials[i] = gpuMat;

        // Texture Bundle (Legacy / SBT access)
        OptixGeometryData::TextureBundle texBundle = {};
        
        if (mat->type() == MaterialType::PrincipledBSDF) {
            PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
            // Use same texture handles populated above
            if (gpuMat.albedo_tex) { texBundle.albedo_tex = gpuMat.albedo_tex; texBundle.has_albedo_tex = true; }
            if (gpuMat.roughness_tex) { texBundle.roughness_tex = gpuMat.roughness_tex; texBundle.has_roughness_tex = true; }
            if (gpuMat.normal_tex) { texBundle.normal_tex = gpuMat.normal_tex; texBundle.has_normal_tex = true; }
            if (gpuMat.metallic_tex) { texBundle.metallic_tex = gpuMat.metallic_tex; texBundle.has_metallic_tex = true; }
            if (gpuMat.specular_tex) { texBundle.specular_tex = gpuMat.specular_tex; texBundle.has_specular_tex = true; }
            if (gpuMat.emission_tex) { texBundle.emission_tex = gpuMat.emission_tex; texBundle.has_emission_tex = true; }
            if (gpuMat.opacity_tex) { 
                texBundle.opacity_tex = gpuMat.opacity_tex;
                texBundle.has_opacity_tex = true;
                texBundle.opacity_has_alpha = pbsdf->opacityProperty.texture->has_alpha ? 1 : 0;
            }
            if (gpuMat.transmission_tex) { texBundle.transmission_tex = gpuMat.transmission_tex; texBundle.has_transmission_tex = true; }
            
            // Height texture not in logic bundle yet, but available in GpuMaterial
        }
        data.textures[i] = texBundle;

        // Volumetric Info Init
        OptixGeometryData::VolumetricInfo volInfo = {};
        if (mat->type() == MaterialType::Volumetric) {
            Volumetric* vol_mat = static_cast<Volumetric*>(mat.get());
            volInfo.is_volumetric = 1;
            Vec3 albedo = vol_mat->getAlbedo();
            Vec3 emission = vol_mat->getEmissionColor();
            volInfo.density = static_cast<float>(vol_mat->getDensity());
            volInfo.absorption = static_cast<float>(vol_mat->getAbsorption());
            volInfo.scattering = static_cast<float>(vol_mat->getScattering());
            volInfo.albedo = make_float3(albedo.x, albedo.y, albedo.z);
            volInfo.emission = make_float3(emission.x, emission.y, emission.z);
            volInfo.g = static_cast<float>(vol_mat->getG());
            volInfo.step_size = vol_mat->getStepSize();
            volInfo.max_steps = vol_mat->getMaxSteps();
            volInfo.noise_scale = vol_mat->getNoiseScale();
            volInfo.multi_scatter = vol_mat->getMultiScatter();
            volInfo.g_back = vol_mat->getGBack();
            volInfo.lobe_mix = vol_mat->getLobeMix();
            volInfo.light_steps = vol_mat->getLightSteps();
            volInfo.shadow_strength = vol_mat->getShadowStrength();
            volInfo.aabb_min = make_float3(1e10f, 1e10f, 1e10f);
            volInfo.aabb_max = make_float3(-1e10f, -1e10f, -1e10f);
        }
        data.volumetric_info[i] = volInfo;
    }

    // 4. Parallel Geometry Extraction
    size_t nTris = triangles.size();
    if (nTris > 0) {
        data.vertices.resize(nTris * 3);
        data.normals.resize(nTris * 3);
        data.uvs.resize(nTris * 3);
        data.colors.resize(nTris * 3);
        data.indices.resize(nTris);
        data.material_indices.resize(nTris);
        data.boneIndices.resize(nTris * 3, make_int4(-1, -1, -1, -1));
        data.boneWeights.resize(nTris * 3, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
        if (nTris < 1000) num_threads = 1; // Don't thread for small meshes

        size_t chunk_size = nTris / num_threads;
        std::vector<std::future<std::vector<OptixGeometryData::VolumetricInfo>>> futures;

        for (unsigned int t = 0; t < num_threads; ++t) {
            size_t start = t * chunk_size;
            size_t end = (t == num_threads - 1) ? nTris : (start + chunk_size);
            
            // Copy current thread's volumetric buffer info to local lambda
            // (We need local accumulation to avoid mutex locking per triangle)
            
            futures.push_back(std::async(std::launch::async, 
                [&, start, end]() -> std::vector<OptixGeometryData::VolumetricInfo> {
                    // Local copy of volumetric info for thread-safe accumulation
                    std::vector<OptixGeometryData::VolumetricInfo> local_vol_info = data.volumetric_info;
                    
                    TriangleMesh* lastParentMesh = nullptr;
                    const Vec3* cachedPositions = nullptr;
                    const Vec3* cachedNormals = nullptr;
                    const Vec2* cachedUvs = nullptr;
                    const uint16_t* cachedMatIDs = nullptr;
                    const std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>>* cachedIndices = nullptr;
                    bool hasGeometry = false;

                    for (size_t i = start; i < end; ++i) {
                        const auto& tri = triangles[i];
                        
                        Vec3 verts[3];
                        Vec3 norms[3];
                        Vec2 uvs[3];
                        uint16_t matID = 0xFFFF;

                        if (tri->parentMesh) {
                            if (tri->parentMesh.get() != lastParentMesh) {
                                lastParentMesh = tri->parentMesh.get();
                                if (lastParentMesh->geometry) {
                                    cachedPositions = lastParentMesh->geometry->get_attribute_data<Vec3>("P");
                                    cachedNormals = lastParentMesh->geometry->get_attribute_data<Vec3>("N");
                                    cachedUvs = lastParentMesh->geometry->get_attribute_data<Vec2>("uv");
                                    cachedMatIDs = lastParentMesh->geometry->get_attribute_data<uint16_t>("materialID");
                                    cachedIndices = &lastParentMesh->geometry->indices;
                                    hasGeometry = (cachedPositions != nullptr) && (cachedIndices != nullptr) && (!cachedIndices->empty());
                                } else {
                                    hasGeometry = false;
                                }
                            }

                            if (hasGeometry) {
                                uint32_t faceIdx = tri->faceIndex;
                                uint32_t baseIdx = faceIdx * 3;
                                
                                uint32_t i0 = (*cachedIndices)[baseIdx + 0];
                                uint32_t i1 = (*cachedIndices)[baseIdx + 1];
                                uint32_t i2 = (*cachedIndices)[baseIdx + 2];
                                
                                verts[0] = cachedPositions[i0];
                                verts[1] = cachedPositions[i1];
                                verts[2] = cachedPositions[i2];
                                
                                norms[0] = cachedNormals ? cachedNormals[i0] : Vec3(0, 1, 0);
                                norms[1] = cachedNormals ? cachedNormals[i1] : Vec3(0, 1, 0);
                                norms[2] = cachedNormals ? cachedNormals[i2] : Vec3(0, 1, 0);
                                
                                uvs[0] = cachedUvs ? cachedUvs[i0] : Vec2(0, 0);
                                uvs[1] = cachedUvs ? cachedUvs[i1] : Vec2(0, 0);
                                uvs[2] = cachedUvs ? cachedUvs[i2] : Vec2(0, 0);
                                
                                matID = cachedMatIDs ? cachedMatIDs[i0] : tri->getMaterialID();
                            } else {
                                verts[0] = tri->getVertexPosition(0);
                                verts[1] = tri->getVertexPosition(1);
                                verts[2] = tri->getVertexPosition(2);
                                norms[0] = tri->getVertexNormal(0);
                                norms[1] = tri->getVertexNormal(1);
                                norms[2] = tri->getVertexNormal(2);
                                uvs[0] = tri->t_ref(0);
                                uvs[1] = tri->t_ref(1);
                                uvs[2] = tri->t_ref(2);
                                matID = tri->getMaterialID();
                            }
                        } else {
                            verts[0] = tri->getVertexPosition(0);
                            verts[1] = tri->getVertexPosition(1);
                            verts[2] = tri->getVertexPosition(2);
                            norms[0] = tri->getVertexNormal(0);
                            norms[1] = tri->getVertexNormal(1);
                            norms[2] = tri->getVertexNormal(2);
                            uvs[0] = tri->t_ref(0);
                            uvs[1] = tri->t_ref(1);
                            uvs[2] = tri->t_ref(2);
                            matID = tri->getMaterialID();
                        }

                        size_t base_v_idx = i * 3;
                        uint3 tri_idx_struct;
                        tri_idx_struct.x = (unsigned int)base_v_idx + 0;
                        tri_idx_struct.y = (unsigned int)base_v_idx + 1;
                        tri_idx_struct.z = (unsigned int)base_v_idx + 2;

                        for (int k = 0; k < 3; ++k) {
                            data.vertices[base_v_idx + k] = make_float3(verts[k].x, verts[k].y, verts[k].z);
                            data.normals[base_v_idx + k] = make_float3(norms[k].x, norms[k].y, norms[k].z);
                            data.uvs[base_v_idx + k] = make_float2(uvs[k].u, uvs[k].v);
                            data.colors[base_v_idx + k] = make_float3(0.0f, 0.0f, 0.0f); // default color

                            // Extract skinning data
                            if (tri->hasSkinData()) {
                                const auto& weights = tri->getSkinBoneWeights(k);
                                int4 bi = make_int4(-1, -1, -1, -1);
                                float4 bw = make_float4(0, 0, 0, 0);

                                // 1. Calculate sum of first 4 weights
                                float sum = 0.0f;
                                size_t numInfluences = std::min(weights.size(), (size_t)4);
                                for (size_t w = 0; w < numInfluences; ++w) {
                                    sum += weights[w].second;
                                }

                                // 2. Assign and normalize
                                if (sum > 1e-6f) {
                                    float invSum = 1.0f / sum;
                                    for (size_t w = 0; w < numInfluences; ++w) {
                                        float normalizedWeight = weights[w].second * invSum;
                                        if (w == 0) { bi.x = weights[w].first; bw.x = normalizedWeight; }
                                        else if (w == 1) { bi.y = weights[w].first; bw.y = normalizedWeight; }
                                        else if (w == 2) { bi.z = weights[w].first; bw.z = normalizedWeight; }
                                        else if (w == 3) { bi.w = weights[w].first; bw.w = normalizedWeight; }
                                    }
                                }

                                data.boneIndices[base_v_idx + k] = bi;
                                data.boneWeights[base_v_idx + k] = bw;
                            }
                        }
                        data.indices[i] = tri_idx_struct;

                        int gpuIndex = matID;
                        if (gpuIndex < 0 || gpuIndex >= static_cast<int>(data.materials.size())) {
                            gpuIndex = 0;
                        }
                        data.material_indices[i] = gpuIndex;

                        // Volumetric AABB (Local Accumulation)
                        if (gpuIndex < static_cast<int>(local_vol_info.size()) && 
                            local_vol_info[gpuIndex].is_volumetric) {
                            auto& vol = local_vol_info[gpuIndex];
                            for (int vi = 0; vi < 3; vi++) {
                                 const auto& v = verts[vi];
                                 vol.aabb_min.x = std::min(vol.aabb_min.x, (float)v.x);
                                 vol.aabb_min.y = std::min(vol.aabb_min.y, (float)v.y);
                                 vol.aabb_min.z = std::min(vol.aabb_min.z, (float)v.z);
                                 vol.aabb_max.x = std::max(vol.aabb_max.x, (float)v.x);
                                 vol.aabb_max.y = std::max(vol.aabb_max.y, (float)v.y);
                                 vol.aabb_max.z = std::max(vol.aabb_max.z, (float)v.z);
                            }
                        }
                    }
                    return local_vol_info;
                }
            ));
        }

        // JOIN and MERGE Volumetric Info
        for (auto& f : futures) {
            auto local_vol = f.get();
            // Merge into main data
            for (size_t i = 0; i < data.volumetric_info.size(); ++i) {
                if (data.volumetric_info[i].is_volumetric) {
                    auto& main_vol = data.volumetric_info[i];
                    const auto& thread_vol = local_vol[i];
                    
                    main_vol.aabb_min.x = std::min(main_vol.aabb_min.x, thread_vol.aabb_min.x);
                    main_vol.aabb_min.y = std::min(main_vol.aabb_min.y, thread_vol.aabb_min.y);
                    main_vol.aabb_min.z = std::min(main_vol.aabb_min.z, thread_vol.aabb_min.z);
                    main_vol.aabb_max.x = std::max(main_vol.aabb_max.x, thread_vol.aabb_max.x);
                    main_vol.aabb_max.y = std::max(main_vol.aabb_max.y, thread_vol.aabb_max.y);
                    main_vol.aabb_max.z = std::max(main_vol.aabb_max.z, thread_vol.aabb_max.z);
                }
            }
        }
    }
    
    // Finalize Volumetric AABBs (Apply padding)
    for (auto& vol : data.volumetric_info) {
        if (vol.is_volumetric) {
            if (vol.aabb_max.x < vol.aabb_min.x) {
                 vol.aabb_min = make_float3(0.f, 0.f, 0.f);
                 vol.aabb_max = make_float3(0.f, 0.f, 0.f);
            } else {
                 float padding = 0.001f;
                 vol.aabb_min.x -= padding; vol.aabb_min.y -= padding; vol.aabb_min.z -= padding;
                 vol.aabb_max.x += padding; vol.aabb_max.y += padding; vol.aabb_max.z += padding;
            }
        }
    }

    return data;
}
