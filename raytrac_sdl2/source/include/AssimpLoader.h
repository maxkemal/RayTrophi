#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <memory>
#include <format>
#include "Triangle.h"
#include "Camera.h"
#include "Mesh.h"
#include "Material.h"
#include "PrincipledBSDF.h"
#include "Dielectric.h"
#include "Texture.h"
#include "AreaLight.h"
#include "Light.h"
#include "DirectionalLight.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "Volumetric.h"
#include <map>
#include "Quaternion.h"
#include <globals.h>
#include "EmbreeBVH.h"
#include "sbt_data.h"
#include "MaterialManager.h"
#include <set>
struct MeshInstance {
    int meshIndex; // aiMesh ID
    Matrix4x4 transform;
    std::string nodeName;
};
struct AnimationData {
    std::string name;
    double duration;
    double ticksPerSecond;
    std::map<std::string, std::vector<aiVectorKey>> positionKeys;
    std::map<std::string, std::vector<aiQuatKey>> rotationKeys;
    std::map<std::string, std::vector<aiVectorKey>> scalingKeys;
    
    // Frame range bilgisi (Blender'dan gelen)
    int startFrame = 0;
    int endFrame = 0;

    Matrix4x4 calculateAnimationTransform(const AnimationData& animData, float time, const std::string& nodeName, const Matrix4x4& defaultTransform)
        const {
        // Animasyon süresini normalize et
        double animationTime = fmod(time * ticksPerSecond, duration);

        // Decompose default transform to retrieve Bind Pose values
        aiMatrix4x4 aiDef;
        aiDef.a1 = defaultTransform.m[0][0]; aiDef.a2 = defaultTransform.m[0][1]; aiDef.a3 = defaultTransform.m[0][2]; aiDef.a4 = defaultTransform.m[0][3];
        aiDef.b1 = defaultTransform.m[1][0]; aiDef.b2 = defaultTransform.m[1][1]; aiDef.b3 = defaultTransform.m[1][2]; aiDef.b4 = defaultTransform.m[1][3];
        aiDef.c1 = defaultTransform.m[2][0]; aiDef.c2 = defaultTransform.m[2][1]; aiDef.c3 = defaultTransform.m[2][2]; aiDef.c4 = defaultTransform.m[2][3];
        aiDef.d1 = defaultTransform.m[3][0]; aiDef.d2 = defaultTransform.m[3][1]; aiDef.d3 = defaultTransform.m[3][2]; aiDef.d4 = defaultTransform.m[3][3];

        aiVector3D defScale, defPos;
        aiQuaternion defRot;
        aiDef.Decompose(defScale, defRot, defPos);

        bool hasPos = positionKeys.count(nodeName) && !positionKeys.at(nodeName).empty();
        bool hasRot = rotationKeys.count(nodeName) && !rotationKeys.at(nodeName).empty();
        bool hasScl = scalingKeys.count(nodeName) && !scalingKeys.at(nodeName).empty();

        if (!hasPos && !hasRot && !hasScl) {
            return defaultTransform;
        }

        Matrix4x4 translationMatrix = Matrix4x4::identity();
        Matrix4x4 rotationMatrix = Matrix4x4::identity();
        Matrix4x4 scaleMatrix = Matrix4x4::identity();

        // --- POSITION ---
        if (hasPos) {
            Vec3 position(0, 0, 0);
            const auto& keys = positionKeys.at(nodeName);

            size_t frameIndex = 0;
            for (size_t i = 0; i < keys.size() - 1; i++) {
                if (animationTime < keys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = (float)(keys[nextFrameIndex].mTime - keys[frameIndex].mTime);
            if (deltaTime < 0) deltaTime += (float)duration;

            float factor = (deltaTime == 0) ? 0.0f :
                (float)(animationTime - keys[frameIndex].mTime) / deltaTime;

            const auto& start = keys[frameIndex].mValue;
            const auto& end = keys[nextFrameIndex].mValue;
            position = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
            translationMatrix = Matrix4x4::translation(position);
        } else {
             // Fallback to Bind Pose Position
             translationMatrix = Matrix4x4::translation(Vec3(defPos.x, defPos.y, defPos.z));
        }

        // --- ROTATION ---
        if (hasRot) {
            Quaternion rotation(0, 0, 0, 1);
            const auto& keys = rotationKeys.at(nodeName);

            size_t frameIndex = 0;
            for (size_t i = 0; i < keys.size() - 1; i++) {
                if (animationTime < keys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            double deltaTime = keys[nextFrameIndex].mTime - keys[frameIndex].mTime;
            if (deltaTime < 0) deltaTime += duration;

            double factor = (deltaTime == 0) ? 0.0 :
                (animationTime - keys[frameIndex].mTime) / deltaTime;

            Quaternion start(keys[frameIndex].mValue.x, keys[frameIndex].mValue.y, keys[frameIndex].mValue.z, keys[frameIndex].mValue.w);
            Quaternion end(keys[nextFrameIndex].mValue.x, keys[nextFrameIndex].mValue.y, keys[nextFrameIndex].mValue.z, keys[nextFrameIndex].mValue.w);
            rotation = Quaternion::slerp(start, end, factor);
            rotationMatrix = rotation.toMatrix();
        } else {
             // Fallback to Bind Pose Rotation (Fixes FBX flipped camera issues)
             Quaternion defQ(defRot.x, defRot.y, defRot.z, defRot.w);
             rotationMatrix = defQ.toMatrix();
        }

        // --- SCALING ---
        if (hasScl) {
            Vec3 scaling(1, 1, 1);
            const auto& keys = scalingKeys.at(nodeName);

            size_t frameIndex = 0;
            for (size_t i = 0; i < keys.size() - 1; i++) {
                if (animationTime < keys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = (float)(keys[nextFrameIndex].mTime - keys[frameIndex].mTime);
            if (deltaTime < 0) deltaTime += (float)duration;

            float factor = (deltaTime == 0) ? 0.0f :
                (float)(animationTime - keys[frameIndex].mTime) / deltaTime;

            const auto& start = keys[frameIndex].mValue;
            const auto& end = keys[nextFrameIndex].mValue;
            scaling = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
            scaleMatrix = Matrix4x4::scaling(scaling);
        } else {
            // Fallback to Bind Pose Scale
             scaleMatrix = Matrix4x4::scaling(Vec3(defScale.x, defScale.y, defScale.z));
        }

        // Matrisleri birleştir: Final = Translation * Rotation * Scale
        return translationMatrix * rotationMatrix * scaleMatrix;
    }
};
// Texture bilgilerini önceden al
struct TextureInfo {
    aiTextureType type;
    std::string path;
};
struct BoneData {
    std::unordered_map<std::string, unsigned int> boneNameToIndex;
    std::unordered_map<std::string, aiNode*> boneNameToNode;
    std::unordered_map<std::string, Matrix4x4> boneOffsetMatrices; // << DÜZELTİLDİ
    Matrix4x4 globalInverseTransform;
};


// Matrix4x4.h dosyasının sonunda olabilir:
inline Matrix4x4 convert(const aiMatrix4x4& m) {
    return Matrix4x4(
        m.a1, m.a2, m.a3, m.a4,
        m.b1, m.b2, m.b3, m.b4,
        m.c1, m.c2, m.c3, m.c4,
        m.d1, m.d2, m.d3, m.d4
    );
}
class AssimpLoader {
public:
    std::set<std::string> lightNodeNames;
    std::set<std::string> cameraNodeNames;
    std::vector<std::shared_ptr<Light>> lights;
    std::unordered_map<std::string, std::shared_ptr<Texture>> textureCache;
    std::vector<TextureInfo> textureInfos;
    std::vector<std::shared_ptr<Camera>> cameras; // Kamera listesi

    Assimp::Importer importer;
    const aiNode* getNodeByName(const std::string& name) const {
        auto it = nodeMap.find(name);
        return it != nodeMap.end() ? it->second : nullptr;
    }
    aiMatrix4x4 getGlobalParentTransform(const aiNode* node) const {
        aiMatrix4x4 transform;
        transform.IsIdentity();

        if (!node) return transform;

        const aiNode* current = node->mParent; //  Sadece parent'ları alacağız
        while (current) {
            transform = current->mTransformation * transform;
            current = current->mParent;
        }

        return transform;
    }

  
    // Tek bir kamera almak için
    std::shared_ptr<Camera> getCamera(size_t index) const {
        if (index < cameras.size()) {
            return cameras[index];
        }
        return nullptr;
    }
    // Tüm kameraları almak için
    const std::vector<std::shared_ptr<Camera>>& getCameras() const {
        return cameras;
    }

    // Kamera sayısını almak için
    size_t getCameraCount() const {
        return cameras.size();
    }

    // Varsayılan kamerayı almak için
    std::shared_ptr<Camera> getDefaultCamera() const {
        if (!cameras.empty()) {
            return cameras[0];
        }

        // Eğer hiç kamera yoksa, varsayılan bir kamera oluştur
        Vec3 vup(0, 1, 0);
        int blade_count = 6;
        float vfov = 20.0f;
        float focus_dist = 2.91f;
        Vec3 lookfrom1(7.35889f, 4.95831f, 6.92579f);
        Vec3 lookat1(0.0f, 0.0f, 0.0f);
        return std::make_shared<Camera>(
            lookfrom1,
            lookat1,
            vup,
            vfov,  // FOV
            aspect_ratio,  // aspect ratio
            aperture,  // aperture
            focus_dist,  // focus distance
            blade_count      // blade count
        );
    }

    // Hem Triangle hem de AnimationData döndüren metod
    std::tuple<std::vector<std::shared_ptr<Triangle>>, std::vector<AnimationData>, BoneData>
        loadModelToTriangles(const std::string& filename, const std::shared_ptr<Material>& material = nullptr) {
        SCENE_LOG_INFO("Starting model loading: " + filename);

        BoneData boneData;

        SCENE_LOG_INFO("Importing file with Assimp...");
        this->scene = importer.ReadFile(filename,
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_Triangulate |
            aiProcess_CalcTangentSpace
        );

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            SCENE_LOG_ERROR("Assimp import failed: " + std::string(importer.GetErrorString()));
            return {};
        }

        if (!scene || !scene->mRootNode) {
            SCENE_LOG_ERROR("Scene or root node is null: " + std::string(importer.GetErrorString()));
            return {};
        }

        SCENE_LOG_INFO("Assimp file loaded successfully. Processing scene data...");

        SCENE_LOG_INFO("Clearing existing cameras and lights...");
        cameras.clear();
        lights.clear();

        SCENE_LOG_INFO("Building node map from scene hierarchy...");
        std::function<void(aiNode*)> recurse = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                recurse(node->mChildren[i]);
            };
        recurse(scene->mRootNode);
        SCENE_LOG_INFO("Node map built with " + std::to_string(nodeMap.size()) + " nodes.");

        SCENE_LOG_INFO("Processing cameras...");
        processCameras(scene);
        SCENE_LOG_INFO("Cameras processed: " + std::to_string(cameras.size()) + " camera(s) found.");

        SCENE_LOG_INFO("Processing lights...");
        processLights(scene);
        SCENE_LOG_INFO("Lights processed: " + std::to_string(lights.size()) + " light(s) found.");

        SCENE_LOG_INFO("Building Bone Data...");
      
        buildBoneData(scene, boneData);

        std::vector<std::shared_ptr<Triangle>> triangles;
        OptixGeometryData geometry_data;

        SCENE_LOG_INFO("Processing nodes to extract triangles...");
        processNodeToTriangles(scene->mRootNode, scene, triangles, boneData, &geometry_data);
        SCENE_LOG_INFO("Triangle extraction completed: " + std::to_string(triangles.size()) + " triangles processed.");

        // NOTE: Bone weights now assigned inside processNodeToTriangles -> processTriangles
        // Old processBones call removed.

        SCENE_LOG_INFO("Processing animations...");
        std::vector<AnimationData> animationDataList;

        if (scene->mNumAnimations > 0) {
            SCENE_LOG_INFO("Found " + std::to_string(scene->mNumAnimations) + " animation(s) in model.");

            for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
                const aiAnimation* animation = scene->mAnimations[i];
                AnimationData animData;

                animData.name = animation->mName.C_Str();
                animData.duration = animation->mDuration;
                animData.ticksPerSecond = animation->mTicksPerSecond;

                SCENE_LOG_INFO("[Animation " + std::to_string(i + 1) + "] Name: " + animData.name +
                    ", Duration: " + std::to_string(animData.duration) +
                    ", TPS: " + std::to_string(animData.ticksPerSecond));

                unsigned int totalKeys = 0;

                for (unsigned int j = 0; j < animation->mNumChannels; ++j) {
                    const aiNodeAnim* channel = animation->mChannels[j];
                    std::string nodeName = channel->mNodeName.C_Str();

                    SCENE_LOG_INFO("  [Channel " + std::to_string(j + 1) + "] Node: " + nodeName +
                        ", Pos Keys: " + std::to_string(channel->mNumPositionKeys) +
                        ", Rot Keys: " + std::to_string(channel->mNumRotationKeys) +
                        ", Scale Keys: " + std::to_string(channel->mNumScalingKeys));

                    for (unsigned int k = 0; k < channel->mNumPositionKeys; ++k) {
                        animData.positionKeys[nodeName].push_back(channel->mPositionKeys[k]);
                    }

                    for (unsigned int k = 0; k < channel->mNumRotationKeys; ++k) {
                        animData.rotationKeys[nodeName].push_back(channel->mRotationKeys[k]);
                    }

                    for (unsigned int k = 0; k < channel->mNumScalingKeys; ++k) {
                        animData.scalingKeys[nodeName].push_back(channel->mScalingKeys[k]);
                    }

                    totalKeys += channel->mNumPositionKeys + channel->mNumRotationKeys + channel->mNumScalingKeys;
                }

                // Frame range hesapla (tüm keyframe'lerden min/max time bul)
                double minTime = std::numeric_limits<double>::max();
                double maxTime = std::numeric_limits<double>::lowest();
                
                // Position keys
                for (const auto& [nodeName, keys] : animData.positionKeys) {
                    for (const auto& key : keys) {
                        minTime = std::min(minTime, key.mTime);
                        maxTime = std::max(maxTime, key.mTime);
                    }
                }
                
                // Rotation keys
                for (const auto& [nodeName, keys] : animData.rotationKeys) {
                    for (const auto& key : keys) {
                        minTime = std::min(minTime, key.mTime);
                        maxTime = std::max(maxTime, key.mTime);
                    }
                }
                
                // Scaling keys
                for (const auto& [nodeName, keys] : animData.scalingKeys) {
                    for (const auto& key : keys) {
                        minTime = std::min(minTime, key.mTime);
                        maxTime = std::max(maxTime, key.mTime);
                    }
                }
                
                // Time'ı frame'e çevir
                if (animData.ticksPerSecond > 0) {
                    animData.startFrame = static_cast<int>(std::round(minTime / animData.ticksPerSecond * 24.0)); // Blender default 24 FPS
                    animData.endFrame = static_cast<int>(std::round(maxTime / animData.ticksPerSecond * 24.0));
                }

                SCENE_LOG_INFO("[Animation " + std::to_string(i + 1) + "] Total channels: " +
                    std::to_string(animation->mNumChannels) + ", Total keys: " +
                    std::to_string(totalKeys) + 
                    ", Frame range: " + std::to_string(animData.startFrame) + "-" + std::to_string(animData.endFrame));

                animationDataList.push_back(animData);
            }

            SCENE_LOG_INFO("All " + std::to_string(animationDataList.size()) + " animation(s) loaded successfully.");
        }
        else {
            SCENE_LOG_INFO("No animations found in file: " + filename);
        }

        SCENE_LOG_INFO("Model loading completed. Summary - Triangles: " +
            std::to_string(triangles.size()) +
            ", Animations: " + std::to_string(animationDataList.size()) +
            ", Cameras: " + std::to_string(cameras.size()) +
            ", Lights: " + std::to_string(lights.size()));

        return { triangles, animationDataList, boneData };
    }

    std::tuple<std::vector<Mesh>, std::vector<AnimationData>, BoneData>
        loadModelToMeshes(const std::string& filename) {
        Assimp::Importer importer;

        const aiScene* scene = importer.ReadFile(filename,
            aiProcess_GenSmoothNormals |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices|
            aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            SCENE_LOG_ERROR("Model load failed! Assimp: " + std::string(importer.GetErrorString()));
            return {};
        }

        // 1. Kamera / Işık / Node Ağacı
        nodeMap.clear();
        std::function<void(aiNode*)> recurse = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                recurse(node->mChildren[i]);
            };
        recurse(scene->mRootNode);

        processCameras(scene);
        processLights(scene);

        // 2. Materyalleri yükle
        std::vector<std::shared_ptr<Material>> materials;
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
            materials.push_back(processMaterial(scene->mMaterials[i],scene));

        // 3. Meshleri oluştur
        std::vector<Mesh> meshes;
        BoneData boneData;
        processBonesForMeshes(scene, meshes, boneData); // Triangle olmayan loader’da boş triangle listesi verilebilir
        processNodeToMeshes(scene->mRootNode, scene, meshes, boneData, materials);

        // 4. Animasyonları al
        std::vector<AnimationData> animationDataList;
        if (scene->mNumAnimations > 0) {
            for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
                const aiAnimation* animation = scene->mAnimations[i];
                AnimationData animData;
                animData.name = animation->mName.C_Str();
                animData.duration = animation->mDuration;
                animData.ticksPerSecond = animation->mTicksPerSecond;
                for (unsigned int j = 0; j < animation->mNumChannels; ++j) {
                    const aiNodeAnim* channel = animation->mChannels[j];
                    std::string nodeName = channel->mNodeName.C_Str();
                    for (unsigned int k = 0; k < channel->mNumPositionKeys; ++k)
                        animData.positionKeys[nodeName].push_back(channel->mPositionKeys[k]);
                    for (unsigned int k = 0; k < channel->mNumRotationKeys; ++k)
                        animData.rotationKeys[nodeName].push_back(channel->mRotationKeys[k]);
                    for (unsigned int k = 0; k < channel->mNumScalingKeys; ++k)
                        animData.scalingKeys[nodeName].push_back(channel->mScalingKeys[k]);
                }
                animationDataList.push_back(animData);
            }
        }

        return { meshes, animationDataList, boneData };
    }

    std::vector<std::shared_ptr<Light>> getLights() const {
        return lights;
    }

    const std::vector<std::shared_ptr<Camera>>& getCameras() {
        return cameras;
    }


    static Vec3 rotateVector(const Vec3& v, const aiQuaternion& q) {
        aiMatrix3x3 mat = aiMatrix3x3(q.GetMatrix());
        return Vec3(
            mat.a1 * v.x + mat.a2 * v.y + mat.a3 * v.z,
            mat.b1 * v.x + mat.b2 * v.y + mat.b3 * v.z,
            mat.c1 * v.x + mat.c2 * v.y + mat.c3 * v.z
        );
    }

     aiMatrix4x4 getGlobalTransform(const aiNode* node) {
        aiMatrix4x4 transform = node ? node->mTransformation : aiMatrix4x4();
        const aiNode* parent = node ? node->mParent : nullptr;

        while (parent) {
            transform = parent->mTransformation * transform;
            parent = parent->mParent;
        }

        return transform;
    }

    const aiScene* scene = nullptr;
    const aiScene* getScene() const { return scene; }

    bool is_texture_bundle_valid(const OptixGeometryData::TextureBundle& tex) {
        return tex.has_albedo_tex || tex.has_roughness_tex || tex.has_normal_tex ||
            tex.has_metallic_tex || tex.has_transmission_tex ||
            tex.has_opacity_tex || tex.has_emission_tex;
    }

    OptixGeometryData convertTrianglesToOptixData(const std::vector<std::shared_ptr<Triangle>>& triangles) {
        OptixGeometryData data;
        std::unordered_map<GpuMaterialWithTextures, int> materialMap;
        std::vector<GpuMaterial> gpuMaterials;

        for (const auto& tri : triangles) {
            uint3 tri_indices;

            // Use accessor methods for vertex positions and normals
            Vec3 verts[3] = { tri->getVertexPosition(0), tri->getVertexPosition(1), tri->getVertexPosition(2) };
            Vec3 norms[3] = { tri->getVertexNormal(0), tri->getVertexNormal(1), tri->getVertexNormal(2) };
            Vec2 uvs[3] = { tri->t0, tri->t1, tri->t2 };

            for (int i = 0; i < 3; ++i) {
                const Vec3& pos = verts[i];
                const Vec3& normal = norms[i];
                const Vec2& uv = uvs[i];

                data.vertices.push_back(make_float3(pos.x, pos.y, pos.z));
                data.normals.push_back(make_float3(normal.x, normal.y, normal.z));
                data.uvs.push_back(make_float2(uv.x, uv.y));

                unsigned int idx = static_cast<unsigned int>(data.vertices.size()) - 1;
                if (i == 0) tri_indices.x = idx;
                else if (i == 1) tri_indices.y = idx;
                else            tri_indices.z = idx;
            }

            data.indices.push_back(tri_indices);

            // Get material using the new accessor (falls back to mat_ptr for compatibility)
            auto material = tri->getMaterial();
            if (!material) continue;

            // Unique material + texture key
            GpuMaterialWithTextures key;
            key.material = *material->gpuMaterial;
            key.albedoTexID = static_cast<size_t>(tri->textureBundle.albedo_tex);
            key.normalTexID = static_cast<size_t>(tri->textureBundle.normal_tex);
            key.roughnessTexID = static_cast<size_t>(tri->textureBundle.roughness_tex);
            key.metallicTexID = static_cast<size_t>(tri->textureBundle.metallic_tex);
            key.opacityTexID = static_cast<size_t>(tri->textureBundle.opacity_tex);
            key.emissionTexID = static_cast<size_t>(tri->textureBundle.emission_tex);

            int gpuIndex = -1;
            auto it = materialMap.find(key);
            if (it != materialMap.end()) {
                gpuIndex = it->second;
            }
            else {
                gpuIndex = static_cast<int>(gpuMaterials.size());
                gpuMaterials.push_back(key.material);
                data.textures.push_back(tri->textureBundle);
                materialMap[key] = gpuIndex;
            }

            data.material_indices.push_back(gpuIndex);
        }

        data.materials = std::move(gpuMaterials);
        return data;
    }


    void clearTextureCache() {
        SCENE_LOG_INFO("[TEXTURE CLEANUP] Starting texture cache cleanup...");
        int gpu_cleaned = 0;
        int cpu_cleaned = 0;
        
        // 1. AssimpLoader'ın local cache'ini temizle
        for (auto& [name, tex] : textureCache) {
            if (tex) {
                tex->cleanup_gpu(); // GPU belleği temizle
                gpu_cleaned++;
            }
        }
        cpu_cleaned = textureCache.size();
        textureCache.clear(); // CPU cache'i temizle
        
        // 2. Global singleton cache'leri de temizle
        size_t global_texture_cache_size = TextureCache::instance().size();
        size_t global_file_cache_size = FileTextureCache::instance().size();
        
        TextureCache::instance().clear();
        FileTextureCache::instance().clear();
        
        SCENE_LOG_INFO("[TEXTURE CLEANUP] Complete! Stats:");
        SCENE_LOG_INFO("  - GPU textures cleaned: " + std::to_string(gpu_cleaned));
        SCENE_LOG_INFO("  - CPU cache entries removed: " + std::to_string(cpu_cleaned));
        SCENE_LOG_INFO("  - Global TextureCache cleared: " + std::to_string(global_texture_cache_size) + " entries");
        SCENE_LOG_INFO("  - Global FileTextureCache cleared: " + std::to_string(global_file_cache_size) + " entries");
    }
    // AssimpLoader sınıfı içinde veya uygun bir namespace'de
     void calculateAnimatedNodeTransformsRecursive(
        aiNode* node,
        const Matrix4x4& parentAnimatedGlobalTransform,
        const std::map<std::string, const AnimationData*>& animationMap,
        float currentTime,
        std::unordered_map<std::string, Matrix4x4>& animatedGlobalTransformsStore
    ) {
        std::string nodeName = node->mName.C_Str();
        
        // Varsayılan olarak node'un static (bind pose) local transform'unu al
        Matrix4x4 nodeLocalTransform = convert(node->mTransformation);

        // Eğer bu düğüm için animasyon verisi varsa, animasyonlu lokal transformu hesapla
        if (animationMap.count(nodeName) > 0) {
            const AnimationData* anim = animationMap.at(nodeName);
            // AnimationData::calculateAnimationTransform animasyon keyframe'lerinden transform oluşturur
            // Blender'dan gelen animasyon keyframe'leri zaten objenin doğru pozisyonunu içerir
            // YENİ: Bind pose'u (nodeLocalTransform) varsayılan olarak gönderiyoruz.
            nodeLocalTransform = anim->calculateAnimationTransform(*anim, currentTime, nodeName, nodeLocalTransform);
        }
        // Animasyon yoksa, static transform kullanılır (yukarıda zaten atandı)

        // Parent'ın global transform'u ile bu node'un local transform'unu birleştir
        Matrix4x4 currentAnimatedGlobalTransform = parentAnimatedGlobalTransform * nodeLocalTransform;
        animatedGlobalTransformsStore[nodeName] = currentAnimatedGlobalTransform;

        // Çocuk düğümler için rekürsif olarak devam et
        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            calculateAnimatedNodeTransformsRecursive(
                node->mChildren[i],
                currentAnimatedGlobalTransform,
                animationMap,
                currentTime,
                animatedGlobalTransformsStore
            );
        }
    }
// TriangleData ve GpuMaterial kullanan tam OptixGeometryData çıkarımı
private:
    Matrix4x4 convertMatrix(const aiMatrix4x4& m) {
        Matrix4x4 result;
        result.m[0][0] = m.a1; result.m[0][1] = m.a2; result.m[0][2] = m.a3; result.m[0][3] = m.a4;
        result.m[1][0] = m.b1; result.m[1][1] = m.b2; result.m[1][2] = m.b3; result.m[1][3] = m.b4;
        result.m[2][0] = m.c1; result.m[2][1] = m.c2; result.m[2][2] = m.c3; result.m[2][3] = m.c4;
        result.m[3][0] = m.d1; result.m[3][1] = m.d2; result.m[3][2] = m.d3; result.m[3][3] = m.d4;
        return result;
    }
    Vec3 toVec3(const aiVector3D& v) {
        return Vec3(v.x, v.y, v.z);
    }
    void processBonesForMeshes(
        const aiScene* scene,
        std::vector<Mesh>& meshes,
        BoneData& boneData
    ) {
        std::cout << "[processBonesForMeshes] Starting...\n";

        boneData.boneNameToIndex.clear();
        boneData.boneNameToNode.clear();
        boneData.boneOffsetMatrices.clear();

        boneData.globalInverseTransform = convert(scene->mRootNode->mTransformation).inverse();

        // 1. Node map
        std::unordered_map<std::string, aiNode*> nodeMap;
        std::function<void(aiNode*)> collectNodes = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                collectNodes(node->mChildren[i]);
            };
        collectNodes(scene->mRootNode);

        // 2. Meshlere göre işle
        for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
            Mesh& mesh = meshes[meshIdx];
            aiMesh* ai_mesh = scene->mMeshes[mesh.originalMeshIndex];
            if (!ai_mesh->HasBones()) continue;

            std::cout << "[Mesh " << meshIdx << "] Bone count: " << ai_mesh->mNumBones << std::endl;

            for (unsigned int b = 0; b < ai_mesh->mNumBones; ++b) {
                aiBone* bone = ai_mesh->mBones[b];
                std::string boneName = bone->mName.C_Str();

                // Bone index ata
                if (!boneData.boneNameToIndex.count(boneName))
                    boneData.boneNameToIndex[boneName] = static_cast<unsigned int>(boneData.boneNameToIndex.size());

                unsigned int boneIndex = boneData.boneNameToIndex[boneName];
                boneData.boneNameToNode[boneName] = nodeMap.count(boneName) ? nodeMap[boneName] : nullptr;
                boneData.boneOffsetMatrices[boneName] = convert(bone->mOffsetMatrix);

                // Ağırlıkları vertexlere işle
                for (unsigned int w = 0; w < bone->mNumWeights; ++w) {
                    unsigned int vertexId = bone->mWeights[w].mVertexId;
                    float weight = bone->mWeights[w].mWeight;

                    if (vertexId >= mesh.vertices.size()) {
                        std::cerr << "[WARN] Bone weight vertexId out of range: " << vertexId << " vs " << mesh.vertices.size() << std::endl;
                        continue;
                    }

                    mesh.vertices[vertexId].boneWeights.emplace_back(boneIndex, weight);
                }
            }
        }

        std::cout << "[processBonesForMeshes] Completed. Total bones: " << boneData.boneNameToIndex.size() << std::endl;
    }

    void buildBoneData(const aiScene* scene, BoneData& boneData) {
        SCENE_LOG_INFO("[buildBoneData] Starting bone map generation...");

        boneData.boneNameToIndex.clear();
        boneData.boneNameToNode.clear();
        boneData.boneOffsetMatrices.clear();
        boneData.globalInverseTransform = convert(scene->mRootNode->mTransformation).inverse();

        // Node map
        std::unordered_map<std::string, aiNode*> nodeMap;
        std::function<void(aiNode*)> collectNodes = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                collectNodes(node->mChildren[i]);
        };
        collectNodes(scene->mRootNode);

        // Iterate all meshes to find all unique bones
        for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
            aiMesh* mesh = scene->mMeshes[m];
            if (!mesh->HasBones()) continue;

            for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
                aiBone* bone = mesh->mBones[b];
                std::string boneName = bone->mName.C_Str();

                if (boneData.boneNameToIndex.find(boneName) == boneData.boneNameToIndex.end()) {
                    unsigned int id = static_cast<unsigned int>(boneData.boneNameToIndex.size());
                    boneData.boneNameToIndex[boneName] = id;
                    
                    // Haritayı doldur
                    if (nodeMap.find(boneName) != nodeMap.end()) {
                        boneData.boneNameToNode[boneName] = nodeMap[boneName];
                    } else {
                         // SCENE_LOG_WARN("Bone node not found in hierarchy: " + boneName);
                    }
                    boneData.boneOffsetMatrices[boneName] = convert(bone->mOffsetMatrix);
                }
            }
        }
        SCENE_LOG_INFO("[buildBoneData] Completed. Total unique bones: " + std::to_string(boneData.boneNameToIndex.size()));
    }


    std::unordered_map<std::string, aiNode*> nodeMap;
    Vec3 transformPosition(const aiMatrix4x4& matrix, const aiVector3D& position) {
       aiVector3D transformed = matrix * position;
       return Vec3(transformed.x, transformed.y, transformed.z);
   }

    Vec3 transformDirection(const aiMatrix4x4& matrix, const aiVector3D& direction) {
       aiMatrix3x3 rotation(matrix);
       aiVector3D transformed = rotation * direction;
       return Vec3(transformed.x, transformed.y, transformed.z);
   }
    void processNodeToMeshes(aiNode* node, const aiScene* scene, std::vector<Mesh>& outMeshes, const BoneData& boneData, const std::vector<std::shared_ptr<Material>>& materials) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* ai_mesh = scene->mMeshes[node->mMeshes[i]];
            Mesh mesh = processMesh(ai_mesh, node, scene, boneData, materials);
            outMeshes.push_back(std::move(mesh));
        }
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToMeshes(node->mChildren[i], scene, outMeshes, boneData, materials);
        }
    }

   // std::vector<std::shared_ptr<Camera>> cameras;
     void processNodeToTriangles(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Triangle>>& triangles, const BoneData& boneData, OptixGeometryData* geometry_data = nullptr) {
        aiMatrix4x4 globalTransform = getGlobalTransform(node);

        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

            // HitGroupData oluştur
            HitGroupData hit_data = {};

            // Material yükle ve textureları hit_data'ya yaz
            auto convertedMaterial = processMaterial(material, scene, &hit_data, geometry_data);

            // Mevcut üçgen sayısını not al
            size_t triangleStart = triangles.size();

            // Mesh'teki üçgenleri ekle
            processTriangles(mesh, globalTransform, node->mName.C_Str(), convertedMaterial, triangles, boneData);

            // Yeni eklenen tüm üçgenlere textureBundle kopyala
            for (size_t t = triangleStart; t < triangles.size(); ++t) {
                auto& tri = triangles[t];
                tri->textureBundle.albedo_tex = hit_data.albedo_tex;
                tri->textureBundle.has_albedo_tex = hit_data.has_albedo_tex;
                tri->textureBundle.roughness_tex = hit_data.roughness_tex;
                tri->textureBundle.has_roughness_tex = hit_data.has_roughness_tex;
                tri->textureBundle.normal_tex = hit_data.normal_tex;
                tri->textureBundle.has_normal_tex = hit_data.has_normal_tex;
                tri->textureBundle.metallic_tex = hit_data.metallic_tex;
                tri->textureBundle.has_metallic_tex = hit_data.has_metallic_tex;
                tri->textureBundle.transmission_tex = hit_data.transmission_tex;
                tri->textureBundle.has_transmission_tex = hit_data.has_transmission_tex;
                tri->textureBundle.opacity_tex = hit_data.opacity_tex;
                tri->textureBundle.has_opacity_tex = hit_data.has_opacity_tex;
                tri->textureBundle.emission_tex = hit_data.emission_tex;
                tri->textureBundle.has_emission_tex = hit_data.has_emission_tex;
            }

            // OptiX için textureBundle yedekle
            if (geometry_data) {
                OptixGeometryData::TextureBundle tex_bundle = {};
                tex_bundle.albedo_tex = hit_data.albedo_tex;
                tex_bundle.has_albedo_tex = hit_data.has_albedo_tex;
                tex_bundle.roughness_tex = hit_data.roughness_tex;
                tex_bundle.has_roughness_tex = hit_data.has_roughness_tex;
                tex_bundle.normal_tex = hit_data.normal_tex;
                tex_bundle.has_normal_tex = hit_data.has_normal_tex;
                tex_bundle.metallic_tex = hit_data.metallic_tex;
                tex_bundle.has_metallic_tex = hit_data.has_metallic_tex;
                tex_bundle.transmission_tex = hit_data.transmission_tex;
                tex_bundle.has_transmission_tex = hit_data.has_transmission_tex;
                tex_bundle.opacity_tex = hit_data.opacity_tex;
                tex_bundle.has_opacity_tex = hit_data.has_opacity_tex;
                tex_bundle.emission_tex = hit_data.emission_tex;
                tex_bundle.has_emission_tex = hit_data.has_emission_tex;
                geometry_data->textures.push_back(tex_bundle);
            }
        }

        // Çocuk node'ları işle
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToTriangles(node->mChildren[i], scene, triangles, boneData, geometry_data);
        }
    }
     Mesh processMesh(aiMesh* mesh, aiNode* node, const aiScene* scene, const BoneData& boneData, const std::vector<std::shared_ptr<Material>>& materials) {
         Mesh result;

         result.meshName = mesh->mName.C_Str();
         result.nodeName = node->mName.C_Str();
         result.localTransform = convertMatrix(node->mTransformation);

         // 1. Vertexler
         for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
             Vec3 pos = toVec3(mesh->mVertices[i]);
             Vec3 norm = mesh->HasNormals() ? toVec3(mesh->mNormals[i]) : Vec3(0);
             Vec2 uv = mesh->HasTextureCoords(0) ? Vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : Vec2(0);
             result.vertices.emplace_back(pos, norm, uv);
         }

         // 2. İndeksler
         for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
             const aiFace& face = mesh->mFaces[i];
             if (face.mNumIndices != 3) continue;
             result.indices.push_back({ face.mIndices[0], face.mIndices[1], face.mIndices[2] });
         }

         // 3. Materyal
         if (mesh->mMaterialIndex < materials.size()) {
             result.material = materials[mesh->mMaterialIndex];
             result.materialIndex = mesh->mMaterialIndex;
         }

         // 4. Bone Weights
         if (mesh->HasBones()) {
             result.hasSkinning = true;
             for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                 aiBone* bone = mesh->mBones[i];
                 std::string boneName(bone->mName.C_Str());
                 auto it = boneData.boneNameToIndex.find(boneName);
                 if (it != boneData.boneNameToIndex.end()) {
                     int boneIndex = it->second;

                     for (unsigned int j = 0; j < bone->mNumWeights; j++) {
                         const aiVertexWeight& vw = bone->mWeights[j];
                         if (vw.mVertexId < result.vertices.size()) {
                             result.vertices[vw.mVertexId].boneWeights.emplace_back(boneIndex, vw.mWeight);
                         }
                     }
                 }
             }
         }

         return result;
     }

     void processTriangles(
        aiMesh* mesh,
        const aiMatrix4x4& transform,
        const std::string& nodeName,
        const std::shared_ptr<Material>& material,
        std::vector<std::shared_ptr<Triangle>>& triangles,
        const BoneData& boneData)
    {
        aiMatrix4x4 normalTransform = transform;
        normalTransform.Inverse();
        normalTransform.Transpose();

        // Get or create material ID - saves 70+ bytes per triangle!
        uint16_t materialID = MaterialManager::INVALID_MATERIAL_ID;
        if (material) {
            std::string matName = material->materialName.empty() 
                ? "Material_" + nodeName + "_" + std::to_string(mesh->mMaterialIndex)
                : material->materialName;
            materialID = MaterialManager::getInstance().getOrCreateMaterialID(matName, material);
        }

        // Create shared transform for all triangles in this mesh - saves 248 bytes per triangle!
        auto sharedTransform = std::make_shared<Transform>();
        sharedTransform->setBase(convertMatrix(transform));

        // --- NEW: Pre-process bone weights for this mesh ---
        // This ensures every vertex gets its correct weight list before triangle splits
        std::vector<std::vector<std::pair<int, float>>> meshVertexWeights(mesh->mNumVertices);
        if (mesh->HasBones()) {
            for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                aiBone* bone = mesh->mBones[i];
                std::string boneName(bone->mName.C_Str());
                
                auto it = boneData.boneNameToIndex.find(boneName);
                if (it != boneData.boneNameToIndex.end()) {
                    int globalBoneIndex = it->second;
                    for (unsigned int w = 0; w < bone->mNumWeights; w++) {
                        const aiVertexWeight& vw = bone->mWeights[w];
                        if (vw.mVertexId < mesh->mNumVertices) {
                            meshVertexWeights[vw.mVertexId].emplace_back(globalBoneIndex, vw.mWeight);
                        }
                    }
                }
            }
        }
        // ---------------------------------------------------

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue;

            std::vector<Vec3> vertices;
            std::vector<Vec3> normals;
            std::vector<Vec2> texCoords;

            for (unsigned int j = 0; j < 3; j++) {
                unsigned int index = face.mIndices[j];

                // Vertex - Mesh-local space'de sakla, transform etme!
                // Transform sharedTransform tarafından uygulanacak
                aiVector3D vertex = mesh->mVertices[index];
                vertices.emplace_back(vertex.x, vertex.y, vertex.z);

                // Normal - Mesh-local space'de sakla
                if (mesh->HasNormals()) {
                    aiVector3D normal = mesh->mNormals[index];
                    normals.emplace_back(normal.x, normal.y, normal.z);
                }

                // UV
                if (mesh->HasTextureCoords(0)) {
                    aiVector3D texCoord = mesh->mTextureCoords[0][index];
                    texCoords.emplace_back(texCoord.x, texCoord.y);
                }
                else {
                    texCoords.emplace_back(0.0f, 0.0f);
                }
            }

            // Use optimized constructor with materialID
            auto triangle = std::make_shared<Triangle>(
                vertices[0], vertices[1], vertices[2],
                normals[0], normals[1], normals[2],
                texCoords[0], texCoords[1], texCoords[2],
                materialID
            );

            // Set shared transform (all triangles in this mesh share the same transform)
            triangle->setTransformHandle(sharedTransform);
            
            // İlk transform'u uygula (başlangıç pozisyonu için)
            triangle->updateTransformedVertices();


            triangle->setNodeName(nodeName);
            triangle->setAssimpVertexIndices(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            triangle->setFaceIndex(static_cast<int>(triangles.size()));

            // --- NEW: Assign weights to triangle vertices ---
            if (mesh->HasBones()) {
                triangle->initializeSkinData();
                triangle->setSkinBoneWeights(0, meshVertexWeights[face.mIndices[0]]);
                triangle->setSkinBoneWeights(1, meshVertexWeights[face.mIndices[1]]);
                triangle->setSkinBoneWeights(2, meshVertexWeights[face.mIndices[2]]);
            }
            // ------------------------------------------------

            triangles.push_back(triangle);
        }
    }

    void processCameras(const aiScene* scene) {
        try {
            if (!scene || !scene->HasCameras()) {
                SCENE_LOG_WARN("Does not include scene cameras...");
                return;
            }

            for (unsigned int i = 0; i < scene->mNumCameras; i++) {
                aiCamera* aiCam = scene->mCameras[i];
                if (!aiCam) continue;

                cameraNodeNames.insert(aiCam->mName.C_Str());
                aiNode* camNode = scene->mRootNode->FindNode(aiCam->mName.C_Str());
                if (!camNode) {
                    SCENE_LOG_ERROR("Camera node not found: " + std::basic_string(aiCam->mName.C_Str()));
                    continue;
                }

                aiMatrix4x4 globalTransform = getGlobalTransform(camNode);
                aiVector3D scaling, position;
                aiQuaternion rotation;
                globalTransform.Decompose(scaling, rotation, position);

                Vec3 lookfrom(position.x, position.y, position.z);
                Vec3 forward = rotateVector(Vec3(0, 0, -1), rotation);
                Vec3 lookat = lookfrom + forward;
                Vec3 vup = rotateVector(Vec3(0, 1, 0), rotation);

                // Aspect oranı sıfırsa fallback uygula
                double aspect = (aiCam->mAspect > 0.01) ? aiCam->mAspect : 1.0;
                double hfov_rad = aiCam->mHorizontalFOV;
                double vfov_rad = 2.0 * atan(tan(hfov_rad / 2.0) / aspect);
                double vfov = vfov_rad * 180.0 / M_PI;

                // Kamera oluştur
                auto camera = std::make_shared<Camera>(
                    lookfrom, lookat, vup, (float)vfov, (float)aspect, aperture, focusdistance, 5
                );
                camera->nodeName = std::string(aiCam->mName.C_Str());

                cameras.push_back(camera);
                SCENE_LOG_INFO("---- Camera Loaded: " + std::string(aiCam->mName.C_Str()));
                // Loglama
                SCENE_LOG_INFO("Pos: " + lookfrom.toString() +
                    " | LookAt: " + lookat.toString() +
                    " | Up: " + vup.toString() +
                    " | Aspect:" + std::to_string(aspect) +
                    " | FOV:" + std::to_string(vfov));

            }
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::format("Camera processing error: {}", e.what()));
        }
    }

    static std::string LightTypeToString(aiLightSourceType type)
    {
        switch (type)
        {
        case aiLightSource_DIRECTIONAL: return "Directional";
        case aiLightSource_POINT:       return "Point";
        case aiLightSource_SPOT:        return "Spot";
        case aiLightSource_AREA:        return "Area";
        case aiLightSource_AMBIENT:     return "Ambient";
        default:                        return "Unknown";
        }
    }

    void processLights(const aiScene* scene) {
        try {
            if (!scene) {
                SCENE_LOG_ERROR("Light processing failed: Scene is null");
                return;
            }

            if (!scene->HasLights()) {
                SCENE_LOG_WARN("Scene has no lights, skipping light processing");
                return;
            }

            SCENE_LOG_INFO("Total lights in scene: " + std::to_string(scene->mNumLights));

            for (unsigned int i = 0; i < scene->mNumLights; i++) {
                aiLight* aiLgt = scene->mLights[i];
                std::string name = aiLgt->mName.C_Str();
                std::string typeStr = LightTypeToString(aiLgt->mType);

                SCENE_LOG_INFO("Processing light: " + name + " (" + typeStr + ")");

                lightNodeNames.insert(name);

                const aiNode* node = scene->mRootNode->FindNode(aiLgt->mName);
                if (!node) {
                    SCENE_LOG_WARN("Node not found for light: " + name);
                    continue;
                }

                aiMatrix4x4 global = getGlobalTransform(node);

                Vec3 pos = transformPosition(global, aiLgt->mPosition);
                Vec3 dir = transformDirection(global, aiLgt->mDirection);

                if ((aiLgt->mType == aiLightSource_DIRECTIONAL || aiLgt->mType == aiLightSource_SPOT) && dir.length() == 0.0f) {
                    SCENE_LOG_WARN("Direction zero-length detected, fallback used for: " + name);
                    dir = Vec3(0, 0, -1);
                }
                else if (dir.length() > 0.0f) {
                    dir = dir.normalize();
                }

                aiColor3D col = aiLgt->mColorDiffuse;
                
                // Preserve color tones, extract intensity properly
                Vec3 color(col.r, col.g, col.b);
                float maxComp = std::max({col.r, col.g, col.b});
                float intensity = 1.0f;
                
                if (maxComp > 1.0f) {
                    // HDR: normalize color, keep intensity
                    intensity = maxComp;
                    color = color / maxComp;
                } else if (maxComp > 0.0f) {
                    // LDR: use as-is
                    intensity = 1.0f;
                    // FIX: Convert sRGB color to Linear space for correct rendering
                    // This prevents colors from looking washed out/dark
                    color.x = powf(color.x, 2.2f);
                    color.y = powf(color.y, 2.2f);
                    color.z = powf(color.z, 2.2f);
                } else {
                    SCENE_LOG_WARN("Light has zero color, using white fallback: " + name);
                    color = Vec3(1.0f);
                    intensity = 0.1f;
                }
                
                // Blender exports Watts. 
                // Dividing by 100 provides a better baseline brightness (100W ~ 1.0 unit)
                // This fixes the 'too dark' issue compared to Blender
                intensity /= 1000.0f;

                std::shared_ptr<Light> light = nullptr;

                switch (aiLgt->mType)
                {
                case aiLightSource_DIRECTIONAL:
                    // Radius = 0.05f (approx 3 degrees) for reasonable soft shadows
                    // Previous 10.0f was way too large, causing light direction to be random
                    light = std::make_shared<DirectionalLight>(dir, color * intensity, 0.05f);
                    light->position = pos;
                    break;

                case aiLightSource_POINT:
                    light = std::make_shared<PointLight>(pos, color * intensity, 0.1f);
                    break;

                case aiLightSource_SPOT:
                {
                    float angleDeg = aiLgt->mAngleInnerCone * (180.0f / M_PI);
                    // Radius = 0.0f for Spot lights (hard shadows default)
                    light = std::make_shared<SpotLight>(pos, dir, color * intensity, angleDeg, 0.0f);
                    break;
                }

                case aiLightSource_AREA:
                {
                    Vec3 forward = transformDirection(global, aiLgt->mDirection).normalize();
                    Vec3 up = transformDirection(global, aiLgt->mUp).normalize();

                    if (forward.length() == 0 || up.length() == 0)
                        SCENE_LOG_WARN("Area light axis transform issue: " + name);

                    Vec3 u = Vec3::cross(forward, up).normalize();
                    Vec3 v = Vec3::cross(u, forward).normalize();

                    light = std::make_shared<AreaLight>(
                        pos, u, v,
                        aiLgt->mSize.x,
                        aiLgt->mSize.y,
                        color * intensity
                    );
                    break;
                }

                default:
                    SCENE_LOG_WARN("Unsupported light type: " + std::to_string(aiLgt->mType));
                    break;
                }

                if (!light) {
                    SCENE_LOG_ERROR("Failed to create light object: " + name);
                    continue;
                }

                light->nodeName = name;
                light->initialDirection = dir;
                lights.push_back(light);

                SCENE_LOG_INFO(
                    "Light created: " + name +
                    " | Type: " + typeStr +
                    " | Pos: (" + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ", " + std::to_string(pos.z) + ")" +
                    " | Dir: (" + std::to_string(dir.x) + ", " + std::to_string(dir.y) + ", " + std::to_string(dir.z) + ")" +
                    " | Color: (" + std::to_string(color.x) + ", " + std::to_string(color.y) + ", " + std::to_string(color.z) + ")" +
                    " | Intensity: " + std::to_string(intensity)
                );
            }
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR("Light processing fatal error: " + std::string(e.what()));
        }
    }

     std::shared_ptr<Material> processMaterial(
        aiMaterial* aiMat,
        const aiScene* scene,
        HitGroupData* hit_data = nullptr,
        OptixGeometryData* geometry_data = nullptr // ← ekledik
    )

    {
       auto material = std::make_shared<PrincipledBSDF>();
       aiString str;
       textureInfos.clear();
       // Material type check
       aiString materialName;
      
       aiMat->Get(AI_MATKEY_NAME, materialName);
       std::string materialNameStr = materialName.C_Str();
       std::transform(materialNameStr.begin(), materialNameStr.end(), materialNameStr.begin(),
           [](unsigned char c) { return std::tolower(c); });
       const aiTextureType textureTypes[] = {
      aiTextureType_DIFFUSE, aiTextureType_SPECULAR, aiTextureType_EMISSIVE,
      aiTextureType_NORMALS, aiTextureType_OPACITY, aiTextureType_METALNESS,
      aiTextureType_DIFFUSE_ROUGHNESS, aiTextureType_AMBIENT_OCCLUSION
       };
	   material->materialName = materialNameStr;
       SCENE_LOG_INFO("Process Material: " + materialNameStr);
       for (auto type : textureTypes) {

           // Ensure texture count is greater than 0
           if (aiMat->GetTextureCount(type) > 0) {
               aiString str;
               int count = aiMat->GetTextureCount(type);
               if (count > 0) {
                   SCENE_LOG_INFO("Material has texture: type=" + std::to_string(type) +
                       " count=" + std::to_string(count));
               }
               // Retrieve the texture, check for success and ensure the texture path is valid
               if (AI_SUCCESS == aiMat->GetTexture(type, 0, &str) && str.length > 0) {
                   // Perform the sanitization and push to textureInfos if the texture name is not empty
                   std::string sanitizedName = sanitizeTextureName(str);
                   if (!sanitizedName.empty()) {
                       textureInfos.push_back({ type, sanitizedName });
                   }
               }
           }
       }

       aiColor3D color(0.0f, 0.0f, 0.0f);
       aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
       material->albedoProperty = MaterialProperty(Vec3(color.r, color.g, color.b), 1.0f);
      
       // Roughness
       float roughness = 0.0f;
       aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
       material->roughnessProperty = MaterialProperty( Vec3(roughness));
	  
       // Metallic
       aiColor3D colorM(0.0f, 0.0f, 0.0f);
       float metalicFactor = 0.0;
       aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metalicFactor);
       material->metallicProperty = MaterialProperty(Vec3(color.r, color.g, color.b), metalicFactor);

       // Emissive Color
       aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
       aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor);

       // Eğer `emissiveFactor` sıfırdan büyükse, bu değeri kullan
       float emissiveStrength = std::max({ emissiveColor.r, emissiveColor.g, emissiveColor.b });
	   
       // Materyale uygula
       material->emissionProperty = MaterialProperty(Vec3(emissiveColor.r, emissiveColor.g, emissiveColor.b), emissiveStrength);
	   //std::cout << "emissiveStrength : " << emissiveStrength << std::endl;
     
       // Specular Reflection (Glossiness)
       aiColor3D specularColor(0.0f, 0.0f, 0.0f);
       aiMat->Get(AI_MATKEY_COLOR_SPECULAR, specularColor);
       material->specularProperty = MaterialProperty(Vec3(specularColor.r, specularColor.g, specularColor.b), 1.0f);
      
       // Clearcoat Factor
       float clearcoatFactor = 0.0f;  // Default is no clearcoat
       float clearcoatRoughness = 0.0f;  // Default smooth      
       aiMat->Get(AI_MATKEY_CLEARCOAT_ROUGHNESS_FACTOR, clearcoatRoughness);
       aiMat->Get(AI_MATKEY_CLEARCOAT_FACTOR, clearcoatFactor);
       material->setClearcoat(clearcoatFactor, clearcoatRoughness);
       //material->setSubsurfaceScattering((color.r, color.g, color.b),0.4);
       // Anisotropic Properties
       float anisotropy = 0.0f;
       if (aiMat->Get(AI_MATKEY_ANISOTROPY_FACTOR, anisotropy) == AI_SUCCESS) {
           aiColor3D anisotropyDir(0.0f, 0.0f, 0.0f);
           aiMat->Get(AI_MATKEY_ANISOTROPY_FACTOR, anisotropyDir);
           material->setAnisotropic(anisotropy, Vec3(anisotropyDir.r, anisotropyDir.g, anisotropyDir.b));
       }
       float ior = 1.5f;
       if (!aiMat->Get(AI_MATKEY_REFRACTI, ior)) {
           aiMat->Get("IOR", 0, 0, ior);  // GLTF için fallback
       }
       material->ior = ior;
	  // std::cout << "Material IOR: " << ior << std::endl;
       float transmission = 0.0f;
       if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_TRANSMISSION_FACTOR, &transmission)) {
           material->setTransmission(transmission, ior);
          
       }

       else {
		   material->setTransmission(0.0f, ior);
       }
       aiString texPath;
       if (aiGetMaterialTexture(aiMat, AI_MATKEY_TRANSMISSION_TEXTURE, &texPath) == AI_SUCCESS) {
           auto texture = loadTextureWithCache(texPath.C_Str(), TextureType::Transmission);
           material->transmissionProperty.texture = texture;
           material->transmissionProperty.intensity = 1.0f;
       }
       // Opacity (transparency)
       float opacity = 1.0f;
           material->opacityProperty = MaterialProperty(Vec3(opacity));     
           if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_OPACITY, &opacity)) {
               material->opacityProperty.alpha = opacity;
              
           }

            std::shared_ptr<Texture> texture;

            for (const auto& texInfo : textureInfos)
            {
                const std::string& path = texInfo.path;
                TextureType ttype = convertToTextureType(texInfo.type);

                // Embedded mı?
                const aiTexture* emb = scene->GetEmbeddedTexture(path.c_str());

                if (emb) {
                    // Embedded texture için unique cache key oluştur
                    // Format: "embedded_<pointer_address>_<type>"
                    std::string embeddedKey = "embedded_" + 
                        std::to_string(reinterpret_cast<uintptr_t>(emb)) + 
                        "_" + std::to_string(static_cast<int>(ttype));

                    // Cache'de var mı kontrol et
                    auto cacheIt = textureCache.find(embeddedKey);
                    if (cacheIt != textureCache.end()) {
                        texture = cacheIt->second;
                        SCENE_LOG_INFO("[EMBEDDED CACHE HIT] Reusing texture: " + path + " | Key: " + embeddedKey);
                    } else {
                        // Cache'de yok, yeni oluştur ve cache'e ekle
                        texture = std::make_shared<Texture>(emb, ttype, embeddedKey);
                        textureCache[embeddedKey] = texture;
                        SCENE_LOG_INFO("[EMBEDDED CACHE MISS] Created new texture: " + path + " | Key: " + embeddedKey);
                    }
                }
                else
                    texture = loadTextureWithCache(path, ttype);

              
               if (!texture || !texture->is_loaded())
                   continue;

               // CPU tarafı materyale her durumda ata (CPU renderer için gerekli)
               switch (texInfo.type)
               {
               case aiTextureType_DIFFUSE:
                   material->albedoProperty.texture = texture;
                   if (texture->has_alpha) {
                       material->opacityProperty.texture = texture;
                       material->opacityProperty.alpha = opacity;
                   }
                   break;
               case aiTextureType_SPECULAR:
                   material->specularProperty.texture = texture;
                   break;
               case aiTextureType_NORMALS:
                   material->normalProperty.texture = texture;
                   break;
               case aiTextureType_EMISSIVE:
                   material->emissionProperty.texture = texture;
                   break;
               case aiTextureType_OPACITY:
                   material->opacityProperty.texture = texture;
                   material->opacityProperty.intensity = 1.0f;
                   break;
               case aiTextureType_METALNESS:
                   material->metallicProperty.texture = texture;
                   break;
               case aiTextureType_DIFFUSE_ROUGHNESS:
                   material->roughnessProperty.texture = texture;
                   break;
               case aiTextureType_TRANSMISSION:
                   material->transmissionProperty.texture = texture;
                   break;
               default:
                   break;
               }

               // GPU yüklemeyi dene (sadece bir kez, zaten is_gpu_uploaded flag'i varsa atla)
               bool gpu_ok = false;
               if (g_hasOptix && !texture->is_gpu_uploaded) {
                   if (!texture->upload_to_gpu()&& g_hasOptix) {
                       SCENE_LOG_ERROR("Texture GPU upload failed: " + texInfo.path);
                       gpu_ok = false;
                   }
                   else {
                       gpu_ok = true;
                   }
               }
               else {
                   gpu_ok = true;
               }

               // hit_data'ya GPU handle ataması: SADECE gpu_ok true ise
               if (hit_data) {
                   if (gpu_ok) {
                       auto cudaTex = texture->get_cuda_texture();
                       if (cudaTex) {
                           switch (texInfo.type)
                           {
                           case aiTextureType_DIFFUSE:
                               hit_data->albedo_tex = cudaTex;
                               hit_data->has_albedo_tex = 1;
                               break;
                           case aiTextureType_SPECULAR:
                               hit_data->roughness_tex = cudaTex;
                               hit_data->has_roughness_tex = 1;
                               break;
                           case aiTextureType_NORMALS:
                               hit_data->normal_tex = cudaTex;
                               hit_data->has_normal_tex = 1;
                               break;
                           case aiTextureType_EMISSIVE:
                               hit_data->emission_tex = cudaTex;
                               hit_data->has_emission_tex = 1;
                               break;
                           case aiTextureType_OPACITY:
                               hit_data->opacity_tex = cudaTex;
                               hit_data->has_opacity_tex = 1;
                               break;
                           case aiTextureType_METALNESS:
                               hit_data->metallic_tex = cudaTex;
                               hit_data->has_metallic_tex = 1;
                               break;
                           case aiTextureType_DIFFUSE_ROUGHNESS:
                               hit_data->roughness_tex = cudaTex;
                               hit_data->has_roughness_tex = 1;
                               break;
                           case aiTextureType_TRANSMISSION:
                               hit_data->transmission_tex = cudaTex;
                               hit_data->has_transmission_tex = 1;
                               break;
                           default:
                               break;
                           }
                       }
                       else {
                           SCENE_LOG_WARN("Texture marked uploaded but cuda texture is null: " + texInfo.path);
                       }
                   }
                   else {
                       // GPU yok/başarısız: hit_data alanlarını 0 bırak
                       switch (texInfo.type)
                       {
                       case aiTextureType_DIFFUSE:
                           hit_data->has_albedo_tex = 0;
                           break;
                       case aiTextureType_SPECULAR:
                       case aiTextureType_DIFFUSE_ROUGHNESS:
                           hit_data->has_roughness_tex = 0;
                           break;
                       case aiTextureType_NORMALS:
                           hit_data->has_normal_tex = 0;
                           break;
                       case aiTextureType_EMISSIVE:
                           hit_data->has_emission_tex = 0;
                           break;
                       case aiTextureType_OPACITY:
                           hit_data->has_opacity_tex = 0;
                           break;
                       case aiTextureType_METALNESS:
                           hit_data->has_metallic_tex = 0;
                           break;
                       case aiTextureType_TRANSMISSION:
                           hit_data->has_transmission_tex = 0;
                           break;
                       default:
                           break;
                       }
                   }
               }
           }




           if (materialNameStr.find("sss") != std::string::npos || materialNameStr.find("subsurface") != std::string::npos) {
              
               material->setSubsurfaceScattering((1.0f, 0.8f, 0.5f), (1.0f, 1.0f, 1.0f));
            
           }

           if (materialNameStr.find("volume") != std::string::npos || materialNameStr.find("volumetric") != std::string::npos) {
               // TEST SETUP: Smoke Cloud defaults
               Vec3 albedo = Vec3(0.8f, 0.8f, 0.8f); // Grey smoke
               float density = 1.0f;                 // Moderate density
               float scattering_factor = 0.6f;       // Forward scattering
               float absorption_probability = 0.1f;
               Vec3 emission = Vec3(0.0f);           // No emission for clearer smoke visualization

               // Gürültü oluştur
               auto noise = std::make_shared<Perlin>();

               // Volumetrik materyali oluştur
               auto volumetric_material = std::make_shared<Volumetric>(
                   albedo, density, absorption_probability, scattering_factor, emission, noise
               );

               // CRITICAL FIX: Ensure GPU material data is populated to prevent pointers errors
               auto gpu = std::make_shared<GpuMaterial>();
                gpu->albedo = make_float3(albedo.x, albedo.y, albedo.z);
                gpu->emission = make_float3(emission.x, emission.y, emission.z);
                gpu->roughness = 1.0f;
                gpu->metallic = 0.0f;
                gpu->opacity = 0.0f; // Surface invisible
                gpu->transmission = 1.0f; 
                gpu->ior = 1.0f;
                gpu->anisotropic = 1.0f; // FLAG: 1.0 means IS_VOLUMETRIC
                volumetric_material->gpuMaterial = gpu;

               // Maintain texture bundle alignment
               if (geometry_data) {
                   OptixGeometryData::TextureBundle tex_bundle = {};
                   geometry_data->textures.push_back(tex_bundle);
               }

               return volumetric_material;
           }
           auto gpu = std::make_shared<GpuMaterial>();

           // Texture varsa dummy, yoksa gerçek değer atıyoruz:
           if (hit_data && hit_data->has_albedo_tex)
               gpu->albedo = make_float3(1.0f, 1.0f, 1.0f);
           else
               gpu->albedo = make_float3(color.r, color.g, color.b);

           if (hit_data && hit_data->has_roughness_tex)
               gpu->roughness = 1.0f;
           else
               gpu->roughness = roughness;

           if (hit_data && hit_data->has_metallic_tex)
               gpu->metallic = 1.0f;
           else
               gpu->metallic = metalicFactor;

           if (hit_data && hit_data->has_emission_tex)
               gpu->emission = make_float3(1.0f, 1.0f, 1.0f);
           else
               gpu->emission = make_float3(emissiveColor.r, emissiveColor.g, emissiveColor.b);

           if (hit_data && hit_data->has_transmission_tex)
               gpu->transmission = 1.0f;
           else
               gpu->transmission = transmission;

           if (hit_data && hit_data->has_opacity_tex)
               gpu->opacity = 0.0f;
           else
               gpu->opacity = opacity;

           // Sabit IOR
           gpu->ior = ior;
		   gpu->clearcoat = clearcoatFactor;
           // GPU materyali materyale ata
           material->gpuMaterial = gpu;
           static int matCount = 0;
           matCount++;

        /*   std::cout << "[Material #" << matCount << "] "
               << material->materialName <<  std::endl;*/              

           // TextureBundle'ı doldur ve pushla (varsa geometry_data)
           if (geometry_data && hit_data) {
               OptixGeometryData::TextureBundle tex_bundle = {};

               tex_bundle.albedo_tex = hit_data->albedo_tex;
               tex_bundle.has_albedo_tex = hit_data->has_albedo_tex;

               tex_bundle.roughness_tex = hit_data->roughness_tex;
               tex_bundle.has_roughness_tex = hit_data->has_roughness_tex;

               tex_bundle.normal_tex = hit_data->normal_tex;
               tex_bundle.has_normal_tex = hit_data->has_normal_tex;

               tex_bundle.metallic_tex = hit_data->metallic_tex;
               tex_bundle.has_metallic_tex = hit_data->has_metallic_tex;

               tex_bundle.transmission_tex = hit_data->transmission_tex;
               tex_bundle.has_transmission_tex = hit_data->has_transmission_tex;

               tex_bundle.opacity_tex = hit_data->opacity_tex;
               tex_bundle.has_opacity_tex = hit_data->has_opacity_tex;

               tex_bundle.emission_tex = hit_data->emission_tex;
               tex_bundle.has_emission_tex = hit_data->has_emission_tex;

               geometry_data->textures.push_back(tex_bundle);
           }

		 
       return material;
   }
    std::shared_ptr<Texture> loadTexture(const std::string& filepath, TextureType type = TextureType::Unknown, const std::string& opacityMapPath = "") {
       std::string fullPath = baseDirectory + filepath;
       auto texture = std::make_shared<Texture>(fullPath, type); //  yeni parametre

       if (!opacityMapPath.empty()) {
           std::string fullOpacityMapPath = baseDirectory + opacityMapPath;
           texture->loadOpacityMap(fullOpacityMapPath);
       }

       return texture;
   }

    std::string sanitizeTextureName(const aiString& str)
    {
        std::string textureName = str.C_Str();

        for (size_t i = 0; i + 2 < textureName.size();)
        {
            if (textureName[i] == '%' &&
                std::isxdigit(textureName[i + 1]) &&
                std::isxdigit(textureName[i + 2]))
            {
                // Hex değeri çöz
                std::string hex = textureName.substr(i + 1, 2);
                char decoded = static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));

                // "%XX" yerine decoded karakter koy
                textureName.replace(i, 3, std::string(1, decoded));
                i += 1;
            }
            else
            {
                i++;
            }
        }

        return textureName;
    }
    std::shared_ptr<Texture> loadTextureWithCache(const std::string& name, TextureType type)
    {
        // Include type in cache key to prevent reusing Diffuse texture for Opacity etc.
        std::string cacheKey = name + "_" + std::to_string(static_cast<int>(type));

        auto it = textureCache.find(cacheKey);
        if (it != textureCache.end())
            return it->second;

        std::filesystem::path texPath(name);
        if (!texPath.is_absolute())
            texPath = std::filesystem::path(baseDirectory) / texPath;
        texPath = texPath.lexically_normal();

       // SCENE_LOG_INFO("Final texture path: " + texPath.string()); 
      //  SCENE_LOG_INFO("Path exists: " + std::to_string(std::filesystem::exists(texPath)));  

        auto tex = std::make_shared<Texture>(texPath.string(), type);

        if (!tex->is_loaded()) {  // ← Bu kontrolü ekle!
            SCENE_LOG_ERROR("Texture is_loaded() = false after constructor: " + texPath.string());
            return nullptr;
        }

        if (g_hasOptix) {
            if (!tex->upload_to_gpu()) {
                SCENE_LOG_ERROR("Disk texture GPU upload failed: " + texPath.string());
                return nullptr;
            }
        }

        textureCache[cacheKey] = tex;
        return tex;
    }
    TextureType convertToTextureType(aiTextureType type) {
        switch (type) {
        case aiTextureType_DIFFUSE: return TextureType::Albedo;
        case 12: /* aiTextureType_BASE_COLOR */ return TextureType::Albedo;
        case aiTextureType_SPECULAR: return TextureType::Unknown; // genelde linear
        case aiTextureType_EMISSIVE: return TextureType::Emission;
        case aiTextureType_NORMALS: return TextureType::Normal;
        case aiTextureType_OPACITY: return TextureType::Opacity;
        case aiTextureType_METALNESS: return TextureType::Metallic;
        case aiTextureType_DIFFUSE_ROUGHNESS: return TextureType::Roughness;
        case aiTextureType_AMBIENT_OCCLUSION: return TextureType::AO;
        default: return TextureType::Unknown;
        }
    }

};
