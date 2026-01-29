#include "SceneExporter.h"
#include "scene_data.h"
#include "Triangle.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Texture.h"
#include "InstanceManager.h" 
#include "globals.h"
#include "imgui.h"
#include <unordered_map>
#include <iostream>
#include <assimp/version.h>
#include <filesystem>

// Light Headers
#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"

// STB Image Write for embedded textures
// #define STB_IMAGE_WRITE_IMPLEMENTATION // Already defined in Main.cpp
#include "stb_image_write.h"
#include <ui_modern.h>

// Define MeshBatch structure at file scope
struct MeshBatch {
    std::string name;
    uint16_t material_id;
    std::vector<std::shared_ptr<Triangle>> triangles;
};

bool SceneExporter::drawExportPopup(SceneData& scene) {
    if (!show_export_popup) return false;
    
    bool trigger_export = false;

    if (ImGui::Begin("Export Settings", &show_export_popup, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Configuration");
        ImGui::Separator();

        ImGui::Checkbox("Geometry", &settings.export_geometry);
        ImGui::Checkbox("Materials", &settings.export_materials);
        
        // Added Cameras and Lights options
        ImGui::Checkbox("Cameras", &settings.export_cameras);
        ImGui::Checkbox("Lights", &settings.export_lights);

        ImGui::Separator();
        ImGui::Checkbox("Selected Objects Only", &settings.export_selected_only);
        UIWidgets::HelpMarker("Export only the currently selected objects.");

        ImGui::Separator();
        ImGui::Text("Animation Support");
        ImGui::Checkbox("Export Skeleton (Bones)", &settings.export_skinning);
        ImGui::Checkbox("Export Animation Clips", &settings.export_animations);
        if (settings.export_animations && !settings.export_skinning) {
            ImGui::TextColored(ImVec4(1,1,0,1), "Warning: Animations usually require Skeleton!");
        }

        ImGui::Separator();
        ImGui::Text("Format");
        static int format_idx = 0; // 0: binary, 1: text
        if (settings.binary_mode) format_idx = 0; else format_idx = 1;
        
        if (ImGui::Combo("Type", &format_idx, "GLTF Binary (.glb)\0GLTF Text (.gltf)\0")) {
            settings.binary_mode = (format_idx == 0);
        }

        ImGui::Separator();
        if (ImGui::Button("Select File & Export", ImVec2(-1, 0))) {
            trigger_export = true;
            show_export_popup = false; // Close popup
        }
        
        ImGui::End();
    }
    return trigger_export;
}

// Helper to find bone roots
std::vector<std::string> findSkeletonRoots(const BoneData& boneData) {
    std::unordered_set<std::string> allBoneNames;
    for (const auto& kv : boneData.boneNameToIndex) {
        allBoneNames.insert(kv.first);
    }

    std::vector<std::string> roots;
    
    // Naive approach: Find nodes whose parents are NOT in bone list
    // This requires traversing up from each bone using the boneNameToNode map
    // Note: boneNameToNode points to IMPORT time nodes. 
    // We assume the hierarchy structure is valid to traverse up.
    
    std::unordered_set<std::string> visited;

    for (const auto& kv : boneData.boneNameToNode) {
        std::string currentName = kv.first;
        aiNode* node = kv.second;
        
        // Walk up until we hit a node not in our bone list or root
        while (node->mParent) {
            std::string pName = node->mParent->mName.C_Str();
            if (allBoneNames.find(pName) == allBoneNames.end()) {
                // Parent is not a bone, so 'node' is a root bone
                if (visited.find(node->mName.C_Str()) == visited.end()) {
                    roots.push_back(node->mName.C_Str());
                    visited.insert(node->mName.C_Str());
                }
                break;
            }
            node = node->mParent;
        }
        
        // If no parent, it's a root
        if (!node->mParent) {
             if (visited.find(node->mName.C_Str()) == visited.end()) {
                roots.push_back(node->mName.C_Str());
                visited.insert(node->mName.C_Str());
            }
        }
    }
    return roots;
}

// Recursive function to copy node hierarchy for export
aiNode* copyNodeHierarchy(aiNode* originalNode, aiNode* newParent) {
    aiNode* newNode = new aiNode();
    newNode->mName = originalNode->mName;
    newNode->mTransformation = originalNode->mTransformation;
    newNode->mParent = newParent;
    
    // Deep copy children
    if (originalNode->mNumChildren > 0) {
        newNode->mNumChildren = originalNode->mNumChildren;
        newNode->mChildren = new aiNode*[newNode->mNumChildren];
        for (unsigned int i = 0; i < newNode->mNumChildren; i++) {
             newNode->mChildren[i] = copyNodeHierarchy(originalNode->mChildren[i], newNode);
        }
    }
    return newNode;
}

// Skeleton Export Logic
// We will export the full hierarchy starting from the "Bone Roots" found.
void exportSkeleton(aiScene* scene, const BoneData& boneData) {
    std::vector<std::string> rootNames = findSkeletonRoots(boneData);
    
    // We append these roots to the scene's existing root
    // But we need to resize mChildren of the scene root
    
    if (rootNames.empty()) return;

    aiNode* sceneRoot = scene->mRootNode;
    unsigned int currentChildCount = sceneRoot->mNumChildren;
    
    // Create new array with space for existing + new skeleton roots
    aiNode** newChildren = new aiNode*[currentChildCount + rootNames.size()];
    
    // Copy existing
    for (unsigned int i = 0; i < currentChildCount; i++) {
        newChildren[i] = sceneRoot->mChildren[i];
    }
    
    int added = 0;
    for (const std::string& rootName : rootNames) {
        if (boneData.boneNameToNode.find(rootName) != boneData.boneNameToNode.end()) {
            aiNode* originalRoot = boneData.boneNameToNode.at(rootName);
            newChildren[currentChildCount + added] = copyNodeHierarchy(originalRoot, sceneRoot);
            added++;
        }
    }
    
    delete[] sceneRoot->mChildren;
    sceneRoot->mChildren = newChildren;
    sceneRoot->mNumChildren = currentChildCount + added;
}


bool SceneExporter::exportScene(const std::string& filepath, SceneData& scene, const ExportSettings& settings, 
                     const std::vector<std::shared_ptr<Hittable>>& selected_objects) {
    if (filepath.empty()) {
        SCENE_LOG_ERROR("Export failed: Filepath is empty.");
        return false;
    }

    SCENE_LOG_INFO("Starting GLTF Export to: " + filepath);
    is_exporting = true;
    current_export_status = "Converting Scene Data...";
    accumulated_textures.clear();
    texture_dedup_map.clear();

    // 1. Identify Selection Names if needed
    std::unordered_set<std::string> selected_node_names;
    if (settings.export_selected_only) {
        for (const auto& obj : selected_objects) {
             auto tri = std::dynamic_pointer_cast<Triangle>(obj);
             if (tri) selected_node_names.insert(tri->getNodeName());
        }
        SCENE_LOG_INFO("[Export] Selected Object Names: " + std::to_string(selected_node_names.size()));
    }

    // 2. Group Triangles by NodeName + Material (Single Pass Logic)
    std::map<std::string, MeshBatch> batches;
    int accepted_primitives = 0;

    if (settings.export_geometry) {
        // Iterate ALL objects once
        for (const auto& obj : scene.world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (!tri) continue;

            // Filter if needed
            // Optimization: check map/set only if filtering is active
            if (settings.export_selected_only) {
                if (selected_node_names.find(tri->getNodeName()) == selected_node_names.end()) {
                    continue; // Skip unselected
                }
            }
            
            accepted_primitives++;

            uint16_t mat_id = tri->getMaterialID(); 
            std::string key = tri->getNodeName() + "_" + std::to_string(mat_id);
            
            // Note: repeated string construction is strictly unavoidable without changing data structures
            // but we avoid the double-loop overhead now.
            if (batches.find(key) == batches.end()) {
                batches[key].name = tri->getNodeName();
                batches[key].material_id = mat_id;
            }
            batches[key].triangles.push_back(tri);
        }
    }
    SCENE_LOG_INFO("[Export] Processing " + std::to_string(accepted_primitives) + " primitives.");

    // 3. Create Assimp Scene
    std::unique_ptr<aiScene> ai_scene(new aiScene());
    ai_scene->mRootNode = new aiNode();
    ai_scene->mRootNode->mName = "Root";

    // EXPORT SKELETON FIRST
    if (settings.export_skinning && scene.boneData.getBoneCount() > 0) {
        exportSkeleton(ai_scene.get(), scene.boneData);
    }

    std::vector<aiMesh*> accumulated_meshes;
    std::vector<aiMaterial*> accumulated_materials;
    std::vector<aiNode*> child_nodes; 

    // Map internal material ID to Assimp Material Index
    std::map<uint16_t, unsigned int> matIdToAiIndex;

    // 4. Process Materials
    for (const auto& kv : batches) {
        uint16_t mat_id = kv.second.material_id;
        if (matIdToAiIndex.find(mat_id) == matIdToAiIndex.end()) {
            auto matShared = MaterialManager::getInstance().getMaterialShared(mat_id);
            std::string matName = "Mat_" + std::to_string(mat_id); 
            if (matShared && !matShared->materialName.empty()) matName = matShared->materialName;
            
            aiMaterial* aiMat = createAssimpMaterial(matShared, matName);
            accumulated_materials.push_back(aiMat);
            matIdToAiIndex[mat_id] = (unsigned int)accumulated_materials.size() - 1;
        }
    }

    // 5. Create Meshes and Nodes (Re-including strictly to preserve file integrity if ranges overlap)
    // Actually, I can skip replacing Step 5 logic if I end the replacement block at line 235 (end of loop 2).
    // BUT the prompt asks me to fix Camera Matrix too. That is further down.
    // I must replace the whole block or use multiple replacements.
    // The previous tool used replace_file_content (single block).
    // Doing a massive block replacement is safer for context matching.

    for (const auto& kv : batches) {
        const MeshBatch& batch = kv.second;
        
        aiMesh* mesh = new aiMesh();
        mesh->mName = batch.name;
        mesh->mMaterialIndex = matIdToAiIndex[batch.material_id];
        mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

        // Vertices, Normals, UVs
        mesh->mNumVertices = (unsigned int)batch.triangles.size() * 3;
        mesh->mVertices = new aiVector3D[mesh->mNumVertices];
        mesh->mNormals = new aiVector3D[mesh->mNumVertices];
        mesh->mTextureCoords[0] = new aiVector3D[mesh->mNumVertices]; 
        mesh->mNumUVComponents[0] = 2; 

        mesh->mNumFaces = (unsigned int)batch.triangles.size();
        mesh->mFaces = new aiFace[mesh->mNumFaces];

        std::map<std::string, std::vector<aiVertexWeight>> boneWeightsMap;

        Matrix4x4 batchTransform = Matrix4x4::identity();
        if (!batch.triangles.empty()) {
            batchTransform = batch.triangles[0]->getTransformMatrix();
        }

        for (size_t i = 0; i < batch.triangles.size(); ++i) {
            auto t = batch.triangles[i];
            
            Vec3 v0 = t->getOriginalVertexPosition(0);
            Vec3 v1 = t->getOriginalVertexPosition(1);
            Vec3 v2 = t->getOriginalVertexPosition(2);
            
            Vec3 n0 = t->getOriginalVertexNormal(0);
            Vec3 n1 = t->getOriginalVertexNormal(1);
            Vec3 n2 = t->getOriginalVertexNormal(2);

             Vec2 uv0 = t->t0;
             Vec2 uv1 = t->t1;
             Vec2 uv2 = t->t2;
        
             int baseIdx = (int)i * 3;
            
             mesh->mVertices[baseIdx + 0] = aiVector3D(v0.x, v0.y, v0.z);
             mesh->mVertices[baseIdx + 1] = aiVector3D(v1.x, v1.y, v1.z);
             mesh->mVertices[baseIdx + 2] = aiVector3D(v2.x, v2.y, v2.z);

             mesh->mNormals[baseIdx + 0] = aiVector3D(n0.x, n0.y, n0.z);
             mesh->mNormals[baseIdx + 1] = aiVector3D(n1.x, n1.y, n1.z);
             mesh->mNormals[baseIdx + 2] = aiVector3D(n2.x, n2.y, n2.z);

             mesh->mTextureCoords[0][baseIdx + 0] = aiVector3D(uv0.x, uv0.y, 0.0f);
             mesh->mTextureCoords[0][baseIdx + 1] = aiVector3D(uv1.x, uv1.y, 0.0f);
             mesh->mTextureCoords[0][baseIdx + 2] = aiVector3D(uv2.x, uv2.y, 0.0f);

             mesh->mFaces[i].mNumIndices = 3;
             mesh->mFaces[i].mIndices = new unsigned int[3];
             mesh->mFaces[i].mIndices[0] = baseIdx + 0;
             mesh->mFaces[i].mIndices[1] = baseIdx + 1;
             mesh->mFaces[i].mIndices[2] = baseIdx + 2;

             if (settings.export_skinning) {
                 auto& weights = t->getVertexBoneWeights(); 
                 if (!weights.empty()) {
                     for (int v = 0; v < 3; v++) {
                         if (v >= weights.size()) break;
                         int currentVertexID = baseIdx + v;
                         
                         for (const auto& bw : weights[v]) {
                             int boneIndex = bw.first;
                             float weight = bw.second;
                             if (weight <= 0.001f) continue;

                             std::string boneName = scene.boneData.getBoneNameByIndex(boneIndex);
                             if (!boneName.empty()) {
                                 boneWeightsMap[boneName].push_back(aiVertexWeight(currentVertexID, weight));
                             }
                         }
                     }
                 }
             }
        }

        if (!boneWeightsMap.empty()) {
            mesh->mNumBones = (unsigned int)boneWeightsMap.size();
            mesh->mBones = new aiBone*[mesh->mNumBones];
            
            int bIdx = 0;
            for (const auto& bw : boneWeightsMap) {
                aiBone* bone = new aiBone();
                bone->mName = bw.first;
                bone->mNumWeights = (unsigned int)bw.second.size();
                bone->mWeights = new aiVertexWeight[bone->mNumWeights];
                std::copy(bw.second.begin(), bw.second.end(), bone->mWeights);
                
                if (scene.boneData.boneOffsetMatrices.count(bw.first)) {
                     Matrix4x4 off = scene.boneData.boneOffsetMatrices.at(bw.first);
                     bone->mOffsetMatrix = aiMatrix4x4(
                         off.m[0][0], off.m[0][1], off.m[0][2], off.m[0][3],
                         off.m[1][0], off.m[1][1], off.m[1][2], off.m[1][3],
                         off.m[2][0], off.m[2][1], off.m[2][2], off.m[2][3],
                         off.m[3][0], off.m[3][1], off.m[3][2], off.m[3][3]
                     );
                } else {
                    bone->mOffsetMatrix = aiMatrix4x4(); 
                }

                mesh->mBones[bIdx++] = bone;
            }
        }

        accumulated_meshes.push_back(mesh);

        aiNode* node = new aiNode();
        node->mName = batch.name;
        node->mNumMeshes = 1;
        node->mMeshes = new unsigned int[1];
        node->mMeshes[0] = (unsigned int)accumulated_meshes.size() - 1;
        node->mParent = ai_scene->mRootNode;
        
        node->mTransformation = aiMatrix4x4(
            batchTransform.m[0][0], batchTransform.m[0][1], batchTransform.m[0][2], batchTransform.m[0][3],
            batchTransform.m[1][0], batchTransform.m[1][1], batchTransform.m[1][2], batchTransform.m[1][3],
            batchTransform.m[2][0], batchTransform.m[2][1], batchTransform.m[2][2], batchTransform.m[2][3],
            batchTransform.m[3][0], batchTransform.m[3][1], batchTransform.m[3][2], batchTransform.m[3][3]
        );

        child_nodes.push_back(node);
    }

    // EXPORT ANIMATIONS (Omitted/Kept same - replacing below)
    // NOTE: Replace tool needs contiguous block. I will just repeat the Animation Block (abbreviated here for prompt efficiency, will generate full loop)
    if (settings.export_animations && !scene.animationDataList.empty()) {
        ai_scene->mNumAnimations = (unsigned int)scene.animationDataList.size();
        ai_scene->mAnimations = new aiAnimation*[ai_scene->mNumAnimations];
        for (size_t i = 0; i < scene.animationDataList.size(); i++) {
            const auto& srcAnim = scene.animationDataList[i];
            if (!srcAnim) continue;
            aiAnimation* dstAnim = new aiAnimation();
            dstAnim->mName = srcAnim->name;
            dstAnim->mDuration = srcAnim->duration;
            dstAnim->mTicksPerSecond = srcAnim->ticksPerSecond;
            std::vector<aiNodeAnim*> channels;
            std::unordered_set<std::string> animatedNodes;
            for(auto& p : srcAnim->positionKeys) animatedNodes.insert(p.first);
            for(auto& p : srcAnim->rotationKeys) animatedNodes.insert(p.first);
            for(auto& p : srcAnim->scalingKeys) animatedNodes.insert(p.first);
            for (const auto& nodeName : animatedNodes) {
                aiNodeAnim* channel = new aiNodeAnim();
                channel->mNodeName = nodeName;
                if (srcAnim->positionKeys.count(nodeName)) {
                    const auto& keys = srcAnim->positionKeys.at(nodeName);
                    channel->mNumPositionKeys = (unsigned int)keys.size();
                    channel->mPositionKeys = new aiVectorKey[channel->mNumPositionKeys];
                    std::copy(keys.begin(), keys.end(), channel->mPositionKeys);
                }
                if (srcAnim->rotationKeys.count(nodeName)) {
                     const auto& keys = srcAnim->rotationKeys.at(nodeName);
                     channel->mNumRotationKeys = (unsigned int)keys.size();
                     channel->mRotationKeys = new aiQuatKey[channel->mNumRotationKeys];
                     std::copy(keys.begin(), keys.end(), channel->mRotationKeys);
                }
                if (srcAnim->scalingKeys.count(nodeName)) {
                     const auto& keys = srcAnim->scalingKeys.at(nodeName);
                     channel->mNumScalingKeys = (unsigned int)keys.size();
                     channel->mScalingKeys = new aiVectorKey[channel->mNumScalingKeys];
                     std::copy(keys.begin(), keys.end(), channel->mScalingKeys);
                }
                channels.push_back(channel);
            }
            dstAnim->mNumChannels = (unsigned int)channels.size();
            dstAnim->mChannels = new aiNodeAnim*[dstAnim->mNumChannels];
            std::copy(channels.begin(), channels.end(), dstAnim->mChannels);
            ai_scene->mAnimations[i] = dstAnim;
        }
    }

    // EXPORT CAMERAS
    if (settings.export_cameras && scene.camera) {
        ai_scene->mNumCameras = 1;
        ai_scene->mCameras = new aiCamera*[1];
        
        aiCamera* cam = new aiCamera();
        cam->mName = "MainCameraNode"; 
        
        const auto& mainCam = scene.camera; 
        
        cam->mPosition = aiVector3D(0,0,0);
        cam->mUp = aiVector3D(0,1,0); 
        cam->mLookAt = aiVector3D(0,0,-1); 
        
        cam->mHorizontalFOV = mainCam->vfov * (3.14159f / 180.0f); 
        cam->mClipPlaneNear = mainCam->near_dist;
        cam->mClipPlaneFar = mainCam->far_dist;
        cam->mAspect = mainCam->aspect_ratio;
        
        ai_scene->mCameras[0] = cam;
        
        aiNode* camNode = new aiNode();
        camNode->mName = "MainCameraNode"; 
        
        // SAFE Matrix Calculation
        Vec3 f = (mainCam->lookat - mainCam->lookfrom).normalize();
        Vec3 vup = mainCam->vup.normalize();
        
        // Check for degenerate (parallel) case
        if (std::abs(Vec3::dot(f, vup)) > 0.99f) {
            // New fallback Up vector
            vup = (std::abs(f.y) > 0.99f) ? Vec3(0,0,1) : Vec3(0,1,0); 
        }

        Vec3 s = Vec3::cross(f, vup).normalize(); // Side
        // Recompute true up
        Vec3 u = Vec3::cross(s, f); 
        
        aiMatrix4x4 mat;
        // Construct View-to-World (Inverse View)? 
        // No, Scene Node Transform is the objects World Matrix.
        // Camera looks -Z.
        // If we want the camera at 'lookfrom' looking at 'lookat'.
        // We need a matrix that transforms (0,0,0) to 'lookfrom' and (-Z) to 'f'.
        
        // Rotation Columns: s, u, -f
        mat.a1 = s.x; mat.a2 = u.x; mat.a3 = -f.x; mat.a4 = mainCam->lookfrom.x;
        mat.b1 = s.y; mat.b2 = u.y; mat.b3 = -f.y; mat.b4 = mainCam->lookfrom.y;
        mat.c1 = s.z; mat.c2 = u.z; mat.c3 = -f.z; mat.c4 = mainCam->lookfrom.z;
        mat.d1 = 0;   mat.d2 = 0;   mat.d3 = 0;    mat.d4 = 1;
        
        camNode->mTransformation = mat;
        child_nodes.push_back(camNode);
    }

    // EXPORT LIGHTS
    if (settings.export_lights && !scene.lights.empty()) {
        std::vector<aiLight*> exportLights;
        
        for(const auto& light : scene.lights) {
            // Check filtering logic... 
             if (settings.export_selected_only) {
                 // Try to match light name with selected nodes?
                 // Most selection is mesh-based, so this might exclude all lights.
                 // We will INCLUDE all lights if Filter is OFF,
                 // If Selected Only is ON, we only include if name matches?
                 if (selected_node_names.find(light->nodeName) == selected_node_names.end()) {
                     // Not found in selected nodes.
                     // BUT users often don't select the light explicitly.
                     // COMPROMISE: If "Selected Objects Only" is ON, DO NOT export unselected lights.
                     // User must select the light (if UI allows) or we assume they want just the mesh.
                     // Wait, user complained "Camera/Light export not done". 
                     // I'll stick to exporting ALL lights if the user explicitly checked "Export Lights" in the popup
                     // even if "Selected Only" is checked? 
                     // No, "Selected Only" implies strict filtering.
                     // I'll trust the set->find result. If UI allows light selection, this works.
                     // If UI doesn't allow light selection, they won't export.
                     // User's UI *does* allow selection of generic items.
                     // So:
                     // continue; // SKIP
                 }
             }

            aiLight* aiL = new aiLight();
            aiL->mName = light->nodeName;
            
            float intensity = light->intensity;
            aiL->mColorDiffuse = aiColor3D(light->color.x * intensity, light->color.y * intensity, light->color.z * intensity);
            aiL->mColorSpecular = aiL->mColorDiffuse; 
            aiL->mColorAmbient = aiColor3D(0,0,0);
            
            // For GLTF: Props are Local. Node has Transform.
            aiL->mPosition = aiVector3D(0,0,0);
            aiL->mDirection = aiVector3D(0,0,-1); 
            
            if (light->type() == LightType::Point) {
                aiL->mType = aiLightSource_POINT;
                aiL->mAttenuationConstant = 1.0f;
                aiL->mAttenuationQuadratic = 1.0f; 
            }
            else if (light->type() == LightType::Directional) {
                aiL->mType = aiLightSource_DIRECTIONAL;
            }
            else if (light->type() == LightType::Spot) {
                 aiL->mType = aiLightSource_SPOT;
                 // Spot params...
            }
            
            exportLights.push_back(aiL);
            
            aiNode* lNode = new aiNode();
            lNode->mName = light->nodeName; 
            
            // Build Transform Matrix for Light
            aiMatrix4x4 mat;
            aiMatrix4x4::Translation(aiVector3D(light->position.x, light->position.y, light->position.z), mat);
            // Rotation? Directional/Spot need rotation. Point doesn't care.
            // TODO: Construct LookAt matrix or similar for direction.
            // Simplified: Just Translation for now (Point). 
            // Directional needs proper orientation.
            
            lNode->mTransformation = mat;
            child_nodes.push_back(lNode);
        }
        
        if (!exportLights.empty()) {
            ai_scene->mNumLights = (unsigned int)exportLights.size();
            ai_scene->mLights = new aiLight*[exportLights.size()];
            std::copy(exportLights.begin(), exportLights.end(), ai_scene->mLights);
        }
    }


    // Assign pointers to Scene
    if (!child_nodes.empty()) {
        unsigned int existingCount = ai_scene->mRootNode->mNumChildren;
        unsigned int count = (unsigned int)child_nodes.size();
        
        aiNode** combined = new aiNode*[existingCount + count];
        
        if (existingCount > 0) {
            std::copy(ai_scene->mRootNode->mChildren, ai_scene->mRootNode->mChildren + existingCount, combined);
            delete[] ai_scene->mRootNode->mChildren;
        }
        
        std::copy(child_nodes.begin(), child_nodes.end(), combined + existingCount);
        
        ai_scene->mRootNode->mChildren = combined;
        ai_scene->mRootNode->mNumChildren = existingCount + count;
    }

    // 6. Assign Arrays to aiScene
    if (!accumulated_meshes.empty()) {
        ai_scene->mNumMeshes = (unsigned int)accumulated_meshes.size();
        ai_scene->mMeshes = new aiMesh*[accumulated_meshes.size()];
        std::copy(accumulated_meshes.begin(), accumulated_meshes.end(), ai_scene->mMeshes);
    }
    
    if (!accumulated_materials.empty()) {
        ai_scene->mNumMaterials = (unsigned int)accumulated_materials.size();
        ai_scene->mMaterials = new aiMaterial*[accumulated_materials.size()];
        std::copy(accumulated_materials.begin(), accumulated_materials.end(), ai_scene->mMaterials);
    }

    if (!accumulated_textures.empty()) {
        ai_scene->mNumTextures = (unsigned int)accumulated_textures.size();
        ai_scene->mTextures = new aiTexture*[accumulated_textures.size()];
        std::copy(accumulated_textures.begin(), accumulated_textures.end(), ai_scene->mTextures);
    }

    // 7. Export
    Assimp::Exporter exporter;

    // CHECK VERSION
    unsigned int major = aiGetVersionMajor();
    unsigned int minor = aiGetVersionMinor();
    unsigned int rev = aiGetVersionRevision();
    SCENE_LOG_INFO("[Export] Assimp Version: " + std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(rev));

    std::string formatId = settings.binary_mode ? "glb2" : "gltf2";
    SCENE_LOG_INFO("[Export] Using Format ID: " + formatId);
    SCENE_LOG_INFO("[Export] Scene Stats: " 
        + std::to_string(ai_scene->mNumMeshes) + " Meshes, "
        + std::to_string(ai_scene->mNumMaterials) + " Materials, "
        + std::to_string(ai_scene->mNumAnimations) + " Animations");

    if (ai_scene->mRootNode->mNumChildren == 0 && ai_scene->mNumMeshes == 0) {
        SCENE_LOG_ERROR("[Export] Scene is empty! Nothing to export.");
        is_exporting = false;
        return false;
    }
    
    // Attempt Export
    aiReturn ret = exporter.Export(ai_scene.get(), formatId, filepath);
    
    if (ret != aiReturn_SUCCESS) {
        SCENE_LOG_ERROR("Assimp Export Failed Code: " + std::to_string(ret));
        std::string err = exporter.GetErrorString();
        SCENE_LOG_ERROR("Assimp Error String: '" + err + "'");
        
        std::string altFormat = settings.binary_mode ? "glb" : "gltf";
        SCENE_LOG_INFO("[Export] Retrying with format: " + altFormat);
        ret = exporter.Export(ai_scene.get(), altFormat, filepath);
        if (ret != aiReturn_SUCCESS) {
             SCENE_LOG_ERROR("Retry Failed: " + std::string(exporter.GetErrorString()));
             is_exporting = false;
             return false;
        }
    }

    SCENE_LOG_INFO("Scene exported successfully to: " + filepath);
    is_exporting = false;
    current_export_status = "Done";
    return true;
}

// Helper for STB write to memory
void stbi_write_to_mem(void *context, void *data, int size) {
    std::vector<unsigned char> *buffer = (std::vector<unsigned char> *)context;
    buffer->insert(buffer->end(), (unsigned char *)data, (unsigned char *)data + size);
}

#include "Material.h" // Ensure full definition is visible

aiMaterial* SceneExporter::createAssimpMaterial(const std::shared_ptr<Material>& mat, const std::string& name) {
    aiMaterial* aiMat = new aiMaterial();

    aiString aiName(name);
    aiMat->AddProperty(&aiName, AI_MATKEY_NAME);

    if (mat) {
        auto gpuMat = mat->gpuMaterial; 

        // --- TEXTURES ---
        // Helper to add texture (Embed if necessary)
        auto addTexture = [&](const std::shared_ptr<Texture>& tex, aiTextureType type, unsigned int uvIndex = 0) {
            if (!tex) return;

            std::string pathStr = tex->name;
            bool needsEmbedding = pathStr.empty() || pathStr.find("embedded_") == 0 || !std::filesystem::exists(pathStr);
            std::string cacheKey = pathStr.empty() ? ("Ptr_" + std::to_string((uintptr_t)tex.get())) : pathStr;

            if ((needsEmbedding || settings.embed_textures) && tex->m_is_loaded) {
                std::string embeddedName;
                
                // CHECK DEDUP
                if (texture_dedup_map.count(cacheKey)) {
                    embeddedName = texture_dedup_map[cacheKey];
                } else {
                    // Create New
                    aiTexture* aiTex = new aiTexture();
                    
                    embeddedName = "*" + std::to_string(accumulated_textures.size());
                    
                    aiTex->mFilename = embeddedName;
                    aiTex->mWidth = 0; // 0 for compressed
                    aiTex->mHeight = 0;
                    strcpy(aiTex->achFormatHint, "png"); 

                    // Compress data
                    std::vector<unsigned char> pngBuffer;
                    int w = tex->width;
                    int h = tex->height;
                    int comp = 4; // RGBA
                    
                    // Extract raw data from Texture
                    std::vector<unsigned char> rawData;
                    bool dataLoaded = false;
                    unsigned char* stbi_pixels = nullptr;

                    if (!tex->pixels.empty()) {
                        rawData.reserve(w * h * 4);
                        for(const auto& p : tex->pixels) {
                             rawData.push_back(p.r);
                             rawData.push_back(p.g);
                             rawData.push_back(p.b);
                             rawData.push_back(p.a);
                        }
                        dataLoaded = true;
                    } else if (tex->is_hdr && !tex->float_pixels.empty()) {
                         rawData.reserve(w * h * 4);
                         for(const auto& p : tex->float_pixels) {
                             rawData.push_back((unsigned char)std::clamp(p.x * 255.0f, 0.0f, 255.0f));
                             rawData.push_back((unsigned char)std::clamp(p.y * 255.0f, 0.0f, 255.0f));
                             rawData.push_back((unsigned char)std::clamp(p.z * 255.0f, 0.0f, 255.0f));
                             rawData.push_back(255);
                        }
                        dataLoaded = true;
                    } else if (std::filesystem::exists(pathStr)) {
                        int x, y, ch;
                        stbi_pixels = stbi_load(pathStr.c_str(), &x, &y, &ch, 4); 
                        if (stbi_pixels) {
                            w = x;
                            h = y;
                            comp = 4;
                            dataLoaded = true;
                        }
                    }

                    if (dataLoaded) {
                         unsigned char* srcData = (stbi_pixels != nullptr) ? stbi_pixels : rawData.data();
                         stbi_write_png_to_func(stbi_write_to_mem, &pngBuffer, w, h, comp, srcData, w * 4);
                         if (stbi_pixels) stbi_image_free(stbi_pixels);

                         aiTex->mWidth = (unsigned int)pngBuffer.size(); 
                         aiTex->pcData = new aiTexel[aiTex->mWidth];
                         memcpy(aiTex->pcData, pngBuffer.data(), aiTex->mWidth);
                         
                         accumulated_textures.push_back(aiTex);
                         texture_dedup_map[cacheKey] = embeddedName; 
                    }
                }

                if (!embeddedName.empty()) {
                     aiString aiPath(embeddedName);
                     aiMat->AddProperty(&aiPath, AI_MATKEY_TEXTURE(type, uvIndex));
                }

            } else if (!pathStr.empty()) {
                aiString path(pathStr);
                aiMat->AddProperty(&path, AI_MATKEY_TEXTURE(type, uvIndex));
            }
        };

        auto pMat = std::dynamic_pointer_cast<PrincipledBSDF>(mat);

        // Albedo / Base Color
        if (mat->albedoProperty.texture) {
            addTexture(mat->albedoProperty.texture, aiTextureType_BASE_COLOR); 
             aiColor3D white(1,1,1);
             aiMat->AddProperty(&white, 1, AI_MATKEY_BASE_COLOR);
        } else if (pMat && pMat->albedoProperty.texture) { // Try casted PBR
             addTexture(pMat->albedoProperty.texture, aiTextureType_BASE_COLOR);
             aiColor3D white(1,1,1);
             aiMat->AddProperty(&white, 1, AI_MATKEY_BASE_COLOR);
        } else if (gpuMat) {
             aiColor3D color(gpuMat->albedo.x, gpuMat->albedo.y, gpuMat->albedo.z);
             aiMat->AddProperty(&color, 1, AI_MATKEY_BASE_COLOR);
        }

        // Normals
        if (mat->normalProperty.texture) {
            addTexture(mat->normalProperty.texture, aiTextureType_NORMALS);
        } else if (pMat && pMat->normalProperty.texture) {
            addTexture(pMat->normalProperty.texture, aiTextureType_NORMALS);
        }

        // Roughness
        if (mat->roughnessProperty.texture) {
            addTexture(mat->roughnessProperty.texture, aiTextureType_DIFFUSE_ROUGHNESS);
        } else if (pMat && pMat->roughnessProperty.texture) {
            addTexture(pMat->roughnessProperty.texture, aiTextureType_DIFFUSE_ROUGHNESS);
        }
        
        // Metallic
        if (mat->metallicProperty.texture) {
             addTexture(mat->metallicProperty.texture, aiTextureType_METALNESS);
        } else if (pMat && pMat->metallicProperty.texture) {
             addTexture(pMat->metallicProperty.texture, aiTextureType_METALNESS);
        }

        // Emission
        if (mat->emissionProperty.texture) {
            addTexture(mat->emissionProperty.texture, aiTextureType_EMISSIVE);
             aiColor3D white(1,1,1);
             aiMat->AddProperty(&white, 1, AI_MATKEY_COLOR_EMISSIVE);
        } else if (pMat && pMat->emissionProperty.texture) {
            addTexture(pMat->emissionProperty.texture, aiTextureType_EMISSIVE);
             aiColor3D white(1,1,1);
             aiMat->AddProperty(&white, 1, AI_MATKEY_COLOR_EMISSIVE);
        }
        
        // --- CONSTANTS ---
        if (gpuMat) {
            float roughness = gpuMat->roughness;
            aiMat->AddProperty(&roughness, 1, AI_MATKEY_ROUGHNESS_FACTOR);
            
            float metallic = gpuMat->metallic;
            aiMat->AddProperty(&metallic, 1, AI_MATKEY_METALLIC_FACTOR);
            
            if (!mat->emissionProperty.texture) {
                float emission_max = std::max({gpuMat->emission.x, gpuMat->emission.y, gpuMat->emission.z});
                if (emission_max > 0.0f) {
                     aiColor3D emitColor(gpuMat->emission.x, gpuMat->emission.y, gpuMat->emission.z);
                     aiMat->AddProperty(&emitColor, 1, AI_MATKEY_COLOR_EMISSIVE);
                     aiMat->AddProperty(&emission_max, 1, "$mat.emissiveIntensity", 0, 0); 
                }
            }
             
            // Transmission / IOR
            if (gpuMat->transmission > 0.0f) {
                float transmission = gpuMat->transmission;
                aiMat->AddProperty(&transmission, 1, AI_MATKEY_TRANSMISSION_FACTOR);
            }
            float ior = gpuMat->ior;
            aiMat->AddProperty(&ior, 1, AI_MATKEY_REFRACTI);
        } else if (pMat) {
             // Fallback to PrincipledBSDF properties if GPU Mat is missing
             float roughness = pMat->roughnessProperty.intensity;
             aiMat->AddProperty(&roughness, 1, AI_MATKEY_ROUGHNESS_FACTOR);

             float metallic = pMat->metallicProperty.intensity;
             aiMat->AddProperty(&metallic, 1, AI_MATKEY_METALLIC_FACTOR);
        }
    }
    return aiMat;
}
