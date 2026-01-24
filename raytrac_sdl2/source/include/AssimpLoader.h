/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          AssimpLoader.h
* Author:        Kemal Demirtaş
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
* Description:
* This file contains the AssimpLoader class, responsible for loading 3D models 
* and their associated data (meshes, materials, animations, lights, cameras)
* using the Open Asset Import Library (Assimp).
*
* External Dependencies:
* - Assimp (Open Asset Import Library) - https://github.com/assimp/assimp
*   Assimp is used under the 3-Clause BSD License.
*   Copyright (c) 2006-2024, assimp team. All rights reserved.
* =========================================================================
* (Turkish Description)
* Proje:         RayTrophi Studio
* Dosya:         AssimpLoader.h
* Yazar:         Kemal Demirtaş
* Tarih:         Haziran 2024
* Lisans:        [Lisans Bilgisi - örn. Özel / MIT / vb.]
* =========================================================================
* Açıklama:
* Bu dosya, Open Asset Import Library (Assimp) kullanarak 3D modelleri ve
* ilişkili verileri (meshler, materyaller, animasyonlar, ışıklar, kameralar)
* yüklemekten sorumlu AssimpLoader sınıfını içerir.
*
* Dış Bağımlılıklar:
* - Assimp (Open Asset Import Library) - https://github.com/assimp/assimp
*   Assimp, 3-Maddeli BSD Lisansı altında kullanılmaktadır.
*   Telif Hakkı (c) 2006-2024, assimp ekibi. Tüm hakları saklıdır.
* =========================================================================
*/

#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <memory>
#include <format>
#include <algorithm>
#include <cmath>
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
#include <future>
#include <thread>
#include <mutex>
struct MeshInstance {
    int meshIndex; // aiMesh ID (aiMesh Kimliği)
    Matrix4x4 transform; // Global transform matrix (Global dönüşüm matrisi)
    std::string nodeName; // Node name in hierarchy (Hiyerarşideki düğüm adı)
};
struct AnimationData {
    std::string name;
    double duration;
    double ticksPerSecond;
    std::map<std::string, std::vector<aiVectorKey>> positionKeys;
    std::map<std::string, std::vector<aiQuatKey>> rotationKeys;
    std::map<std::string, std::vector<aiVectorKey>> scalingKeys;
    
   
    int startFrame = 0;
    int endFrame = 0;

    // Calculates the interpolated transform matrix for a node at a specific time
    // Belirli bir andaki node için enterpolasyonlu dönüşüm matrisini hesaplar
    Matrix4x4 calculateAnimationTransform(const AnimationData& animData, float time, const std::string& nodeName, const Matrix4x4& defaultTransform)
        const {
      
        // Calculate normalized animation time within duration
        // Animasyon süresini normalize et (döngüsel zaman)
        double animationTime = fmod(time * ticksPerSecond, duration);

        // Decompose default transform to retrieve Bind Pose values
        // Bind Pose değerlerini almak için varsayılan transformu ayır
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

        // --- ROTATION (Dönme) ---
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

            Quaternion start(keys[frameIndex].mValue.w, keys[frameIndex].mValue.x, keys[frameIndex].mValue.y, keys[frameIndex].mValue.z);
            Quaternion end(keys[nextFrameIndex].mValue.w, keys[nextFrameIndex].mValue.x, keys[nextFrameIndex].mValue.y, keys[nextFrameIndex].mValue.z);
            rotation = Quaternion::slerp(start, end, factor);
            rotationMatrix = rotation.toMatrix();
        } else {
             // Fallback to Bind Pose Rotation (Fixes FBX flipped camera issues)
             // Bind Pose Rotasyonuna dön (FBX ters kamera sorunlarını düzeltir)
             Quaternion defQ(defRot.w, defRot.x, defRot.y, defRot.z);
             rotationMatrix = defQ.toMatrix();
        }

        // --- SCALING (Ölçekleme) ---
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

        // Combine matrices: Final = Translation * Rotation * Scale
        // Matrisleri birleştir: Final = Çevirme * Dönme * Ölçekleme
        return translationMatrix * rotationMatrix * scaleMatrix;
    }
};

// Pre-fetch Texture Info (Doku bilgilerini önceden al)
struct TextureInfo {
    aiTextureType type;
    std::string path;
};
struct BoneData {
    std::unordered_map<std::string, unsigned int> boneNameToIndex;
    std::unordered_map<std::string, aiNode*> boneNameToNode;
    std::unordered_map<std::string, Matrix4x4> boneOffsetMatrices;
    Matrix4x4 globalInverseTransform;
    
    // =========================================================================
    // OPTIMIZATION: Reverse lookup table (bone index -> bone name)
    // Eliminates O(n²) complexity in animation updates
    // OPTİMİZASYON: Ters arama tablosu (kemik indeksi -> kemik adı)
    // Animasyon güncellemelerindeki O(n²) karmaşıklığını ortadan kaldırır
    // =========================================================================
    std::vector<std::string> boneIndexToName;
    
    // Rebuild reverse lookup table - call after all bones are added
    // Ters arama tablosunu yeniden oluştur (tüm kemikler eklendikten sonra çağır)
    void rebuildReverseLookup() {
        if (boneNameToIndex.empty()) {
            boneIndexToName.clear();
            return;
        }
        
        // Find max index to size the vector correctly
        unsigned int maxIndex = 0;
        for (const auto& [name, idx] : boneNameToIndex) {
            if (idx > maxIndex) maxIndex = idx;
        }
        
        boneIndexToName.resize(maxIndex + 1);
        for (const auto& [name, idx] : boneNameToIndex) {
            boneIndexToName[idx] = name;
        }
    }
    
    // Get bone name by index (O(1) lookup)
    // İndekse göre kemik adını al (O(1) arama)
    const std::string& getBoneNameByIndex(unsigned int index) const {
        static const std::string empty;
        if (index < boneIndexToName.size()) {
            return boneIndexToName[index];
        }
        return empty;
    }
    
    // Check if bone index is valid
    // Kemik indeksinin geçerli olup olmadığını kontrol et
    bool isValidBoneIndex(unsigned int index) const {
        return index < boneIndexToName.size() && !boneIndexToName[index].empty();
    }
    
    // Get total bone count (Toplam kemik sayısı)
    size_t getBoneCount() const {
        return boneNameToIndex.size();
    }
    
    // Clear all bone data (Tüm kemik verilerini temizle)
    void clear() {
        boneNameToIndex.clear();
        boneNameToNode.clear();
        boneOffsetMatrices.clear();
        boneIndexToName.clear();
        globalInverseTransform = Matrix4x4::identity();
    }
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
// =========================================================================
// ASSIMP LOADER - CLASS FOR LOADING AND PROCESSING MODELS
// Assimp Loader - Modelleri Yükleme ve İşleme Sınıfı
// =========================================================================
// This class loads 3D models (GLTF, OBJ, FBX, etc.) using the Assimp library.
// It parses the model, separating it into meshes, triangles, materials, lights, and cameras.
// It also manages skeletal animations and node hierarchy.
//
// Bu sınıf, 3D modelleri (GLTF, OBJ, FBX vb.) Assimp kütüphanesi kullanarak yükler.
// Modeli analiz eder; mesh'lere, üçgenlere, materyallere, ışıklara ve kameralara ayırır.
// Ayrıca iskelet animasyonlarını ve hiyerarşiyi yönetir.
// =========================================================================

class AssimpLoader {
public:
    std::string currentImportName; // Unique material naming to prevent collisions (Çakışma önleyici isim)
    // NOTE: Uses global ::baseDirectory from globals.h
    
    std::set<std::string> lightNodeNames;
    std::set<std::string> cameraNodeNames;
    std::vector<std::shared_ptr<Light>> lights;
    std::unordered_map<std::string, std::shared_ptr<Texture>> textureCache;
    std::vector<TextureInfo> textureInfos;
    std::vector<std::shared_ptr<Camera>> cameras; // Camera list (Kamera listesi)
    bool isFBX = false; // Track if current file is FBX (FBX formatı kontrolü)
    
    // Transform Cache: Ensures all meshes with the same nodeName share the same Transform.
    // Critical for Gizmo to move all object parts together.
    // Transform Önbelleği: Aynı isme sahip node'ların transformunu paylaştırır (Gizmo için kritik).
    std::unordered_map<std::string, std::shared_ptr<Transform>> nodeNameToTransform;
    
    // Clear transform cache (call before each import)
    // Transform önbelleğini temizle (her yüklemeden önce)
    void clearTransformCache() {
        nodeNameToTransform.clear();
    }
    
    // Get or create shared Transform for a node
    // Node için paylaşılan Transform'u al veya oluştur
    std::shared_ptr<Transform> getOrCreateNodeTransform(const std::string& nodeName, const Matrix4x4& baseTransform) {
        auto it = nodeNameToTransform.find(nodeName);
        if (it != nodeNameToTransform.end()) {
            return it->second;  // Return existing transform (Mevcut olanı döndür)
        }
        // Create new transform (Yeni oluştur)
        auto sharedTransform = std::make_shared<Transform>();
        sharedTransform->setBase(baseTransform);
        nodeNameToTransform[nodeName] = sharedTransform;
        return sharedTransform;
    }

    Assimp::Importer importer;

    // Returns node pointer by name
    // İsme göre node döndürür
    const aiNode* getNodeByName(const std::string& name) const {
        auto it = nodeMap.find(name);
        return it != nodeMap.end() ? it->second : nullptr;
    }

    // Calculates the global (world space) transform matrix of a node
    // Node'un global (dünya) transform matrisini hesaplar
    aiMatrix4x4 getGlobalParentTransform(const aiNode* node) const {
        aiMatrix4x4 transform;
        transform.IsIdentity();

        if (!node) return transform;

        const aiNode* current = node->mParent; // Take parents up to root (En üste kadar ebeveynleri al)
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

    // =========================================================================
    // FUNCTION TO LOAD MODEL INTO TRIANGLES
    // Modeli Üçgenlere Yükleme Fonksiyonu
    // =========================================================================
    // Loads the model, processes data, and returns a Triangle list for ray tracing.
    // Modeli yükler, işler ve ray tracing için Üçgen listesi döndürür.
    //
    // @param filename: Path of the file to load (Dosya yolu)
    // @param material: Optional default material (Opsiyonel varsayılan materyal)
    // @return: Triangles, Animation Data, Bone Data (Üçgenler, Animasyon, Kemik Verileri)
    std::tuple<std::vector<std::shared_ptr<Triangle>>, std::vector<AnimationData>, BoneData>
        loadModelToTriangles(const std::string& filename, const std::shared_ptr<Material>& material = nullptr) {
        
        // Generate unique prefix from filename to prevent material collisions
        // Materyal çakışmasını önlemek için benzersiz ön ek oluştur
        std::filesystem::path p(filename);
        this->currentImportName = p.parent_path().filename().string() + "_" + p.stem().string();

        SCENE_LOG_INFO("Starting model loading: " + filename);

        BoneData boneData;
        
        // Check if file is FBX format
        // FBX formatı kontrolü
        std::string lowerFilename = filename;
        std::transform(lowerFilename.begin(), lowerFilename.end(), lowerFilename.begin(), ::tolower);
        this->isFBX = (lowerFilename.find(".fbx") != std::string::npos);
        
        if (isFBX) {
            SCENE_LOG_INFO("FBX format detected");
            // Note: Assimp reads unit scale from FBX metadata automatically
        }

        SCENE_LOG_INFO("Importing file with Assimp...");
        
        unsigned int importFlags = 
            aiProcess_GenSmoothNormals |    // Generate smooth normals (Yumuşak normaller)
            aiProcess_JoinIdenticalVertices | // Optimization: Join same vertices (Optimizasyon: Vertex birleştir)
            aiProcess_Triangulate |         // Triangulate all faces (Tüm yüzeyleri üçgenle)
            aiProcess_CalcTangentSpace;     // Calculate tangent space (Tanjant uzayı hesapla)
        
        // Add global scale for FBX
        // FBX için global ölçekleme
        if (isFBX) {
            importFlags |= aiProcess_GlobalScale;
        }
        
        this->scene = importer.ReadFile(filename, importFlags);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            SCENE_LOG_ERROR("Assimp import failed: " + std::string(importer.GetErrorString()));
            return {};
        }

        SCENE_LOG_INFO("Assimp file loaded successfully. Processing scene data...");

        SCENE_LOG_INFO("Clearing existing cameras, lights, and transform cache...");
        cameras.clear();
        lights.clear();
        clearTransformCache();  // Clear transform cache (Transform önbelleğini temizle)

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
                // Calculate frame range (find min/max time from all keyframes)
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
                
                // Time'ı frame'e çevir (Time to frame conversion)
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

    // =========================================================================
    // FUNCTION TO LOAD MODEL INTO MESHES
    // Modeli Mesh'lere Yükleme Fonksiyonu
    // =========================================================================
    // Loads the model and splits it into Meshes (instead of Triangles).
    // Used for scenarios where logical mesh separation is needed.
    //
    // Modeli yükler ve Mesh'lere ayırır (Triangle listesi yerine).
    // Mantıksal mesh ayrımının gerektiği senaryolar için kullanılır.
    // =========================================================================
    std::tuple<std::vector<Mesh>, std::vector<AnimationData>, BoneData>
        loadModelToMeshes(const std::string& filename) {
        Assimp::Importer importer;

        // Flags for high-quality geometry import (Yüksek kaliteli geometri içe aktarımı için bayraklar)
        const aiScene* scene = importer.ReadFile(filename,
            aiProcess_GenSmoothNormals |
            aiProcess_GenNormals |
            aiProcess_JoinIdenticalVertices|
            aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            SCENE_LOG_ERROR("Model load failed! Assimp: " + std::string(importer.GetErrorString()));
            return {};
        }

        // 1. Camera / Light / Node Tree (Kamera / Işık / Node Ağacı)
        nodeMap.clear();
        std::function<void(aiNode*)> recurse = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                recurse(node->mChildren[i]);
            };
        recurse(scene->mRootNode);

        processCameras(scene);
        processLights(scene);

        // 2. Load Materials (Materyalleri yükle)
        std::vector<std::shared_ptr<Material>> materials;
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
            materials.push_back(processMaterial(scene->mMaterials[i],scene));

        // 3. Create Meshes (Meshleri oluştur)
        std::vector<Mesh> meshes;
        BoneData boneData; // Will be populated later (Daha sonra doldurulacak)
        
        // FIX: processNodeToMeshes creates the meshes first.
        // DÜZELTME: Önce mesh'ler oluşturulmalı.
        processNodeToMeshes(scene->mRootNode, scene, meshes, boneData, materials);
        
        // FIX: Then process bones using the created meshes (and their original indices).
        // DÜZELTME: Sonra oluşturulan mesh'leri kullanarak kemikleri işle.
        // This function reconstructs BoneData and assigns weights to mesh vertices.
        processBonesForMeshes(scene, meshes, boneData); 

        // 4. Animasyonları al (Extract animations)
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
            } else if (mat->gpuMaterial) {
                gpuMat = *mat->gpuMaterial;
            } else if (mat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                Vec3 alb = pbsdf->albedoProperty.color;
                gpuMat.albedo = make_float3((float)alb.x, (float)alb.y, (float)alb.z);
                gpuMat.roughness = (float)pbsdf->roughnessProperty.color.x;
                gpuMat.metallic = (float)pbsdf->metallicProperty.intensity;
                Vec3 em = pbsdf->emissionProperty.color;
                float emStr = pbsdf->emissionProperty.intensity;
                gpuMat.emission = make_float3((float)em.x * emStr, (float)em.y * emStr, (float)em.z * emStr);
                gpuMat.ior = pbsdf->ior;
                gpuMat.transmission = pbsdf->transmission;
                gpuMat.opacity = pbsdf->opacityProperty.alpha;
                gpuMat.subsurface = pbsdf->subsurface;
                Vec3 sssColor = pbsdf->subsurfaceColor;
                gpuMat.subsurface_color = make_float3((float)sssColor.x, (float)sssColor.y, (float)sssColor.z);
                Vec3 sssRadius = pbsdf->subsurfaceRadius;
                gpuMat.subsurface_radius = make_float3((float)sssRadius.x, (float)sssRadius.y, (float)sssRadius.z);
                gpuMat.subsurface_scale = pbsdf->subsurfaceScale;
                gpuMat.subsurface_anisotropy = pbsdf->subsurfaceAnisotropy;
                gpuMat.subsurface_ior = pbsdf->subsurfaceIOR;
                gpuMat.clearcoat = pbsdf->clearcoat;
                gpuMat.clearcoat_roughness = pbsdf->clearcoatRoughness;
                gpuMat.translucent = pbsdf->translucent;
                gpuMat.anisotropic = pbsdf->anisotropic;
            } else {
                gpuMat.albedo = make_float3(0.8f, 0.8f, 0.8f);
                gpuMat.roughness = 0.5f;
                gpuMat.metallic = 0.0f;
                gpuMat.emission = make_float3(0.0f, 0.f, 0.f);
                gpuMat.ior = 1.5f;
                gpuMat.transmission = 0.0f;
                gpuMat.opacity = 1.0f;
            }
            data.materials[i] = gpuMat;

            // Texture Bundle
            OptixGeometryData::TextureBundle texBundle = {};
            auto getCudaTex = [](const std::shared_ptr<Texture>& tex) -> cudaTextureObject_t {
                if (tex && tex->is_loaded()) {
                     if (!tex->is_gpu_uploaded && g_hasOptix) {
                         tex->upload_to_gpu();
                     }
                     return tex->get_cuda_texture();
                }
                return 0;
            };
            if (mat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                if (pbsdf->albedoProperty.texture) { texBundle.albedo_tex = getCudaTex(pbsdf->albedoProperty.texture); texBundle.has_albedo_tex = (texBundle.albedo_tex != 0); }
                if (pbsdf->roughnessProperty.texture) { texBundle.roughness_tex = getCudaTex(pbsdf->roughnessProperty.texture); texBundle.has_roughness_tex = (texBundle.roughness_tex != 0); }
                if (pbsdf->normalProperty.texture) { texBundle.normal_tex = getCudaTex(pbsdf->normalProperty.texture); texBundle.has_normal_tex = (texBundle.normal_tex != 0); }
                if (pbsdf->metallicProperty.texture) { texBundle.metallic_tex = getCudaTex(pbsdf->metallicProperty.texture); texBundle.has_metallic_tex = (texBundle.metallic_tex != 0); }
                if (pbsdf->emissionProperty.texture) { texBundle.emission_tex = getCudaTex(pbsdf->emissionProperty.texture); texBundle.has_emission_tex = (texBundle.emission_tex != 0); }
                if (pbsdf->opacityProperty.texture) { texBundle.opacity_tex = getCudaTex(pbsdf->opacityProperty.texture); texBundle.has_opacity_tex = (texBundle.opacity_tex != 0); }
                if (pbsdf->transmissionProperty.texture) { texBundle.transmission_tex = getCudaTex(pbsdf->transmissionProperty.texture); texBundle.has_transmission_tex = (texBundle.transmission_tex != 0); }
                
                // DEBUG: Trace albedo texture assignment
                // if (texBundle.has_albedo_tex) SCENE_LOG_INFO("AssimpLoader: Albedo texture assigned for material " + std::to_string(i));
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
                        
                        for (size_t i = start; i < end; ++i) {
                            const auto& tri = triangles[i];
                            
                            // Accessors (assuming thread-safe or read-only)
                            Vec3 verts[3] = { tri->getVertexPosition(0), tri->getVertexPosition(1), tri->getVertexPosition(2) };
                            Vec3 norms[3] = { tri->getVertexNormal(0), tri->getVertexNormal(1), tri->getVertexNormal(2) };
                            Vec2 uvs[3] = { tri->t0, tri->t1, tri->t2 };
                            Vec3 cols[3] = { tri->getVertexColor(0), tri->getVertexColor(1), tri->getVertexColor(2) };

                            size_t base_v_idx = i * 3;
                            uint3 tri_idx_struct;
                            tri_idx_struct.x = (unsigned int)base_v_idx + 0;
                            tri_idx_struct.y = (unsigned int)base_v_idx + 1;
                            tri_idx_struct.z = (unsigned int)base_v_idx + 2;

                            for (int k = 0; k < 3; ++k) {
                                data.vertices[base_v_idx + k] = make_float3(verts[k].x, verts[k].y, verts[k].z);
                                data.normals[base_v_idx + k] = make_float3(norms[k].x, norms[k].y, norms[k].z);
                                data.uvs[base_v_idx + k] = make_float2(uvs[k].u, uvs[k].v);
                                data.colors[base_v_idx + k] = make_float3(cols[k].x, cols[k].y, cols[k].z);
                            }
                            data.indices[i] = tri_idx_struct;

                            int gpuIndex = tri->getMaterialID();
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


    void clearTextureCache() {
      //  SCENE_LOG_INFO("[TEXTURE CLEANUP] Starting texture cache cleanup...");
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
        
       /* SCENE_LOG_INFO("[TEXTURE CLEANUP] Complete! Stats:");
        SCENE_LOG_INFO("  - GPU textures cleaned: " + std::to_string(gpu_cleaned));
        SCENE_LOG_INFO("  - CPU cache entries removed: " + std::to_string(cpu_cleaned));
        SCENE_LOG_INFO("  - Global TextureCache cleared: " + std::to_string(global_texture_cache_size) + " entries");
        SCENE_LOG_INFO("  - Global FileTextureCache cleared: " + std::to_string(global_file_cache_size) + " entries");*/
    }
    // AssimpLoader sınıfı içinde veya uygun bir namespace'de
    // Helper to generate unique names
    std::string getUniqueName(const std::string& originalName) const {
        if (currentImportName.empty()) return originalName;
        return currentImportName + "_" + originalName;
    }

    void calculateAnimatedNodeTransformsRecursive(
        aiNode* node,
        const Matrix4x4& parentAnimatedGlobalTransform,
        const std::map<std::string, const AnimationData*>& animationMap,
        float currentTime,
        std::unordered_map<std::string, Matrix4x4>& animatedGlobalTransformsStore
    ) {
        // Original name for Animation lookup
        std::string originalNodeName = node->mName.C_Str();
        
        // Varsayılan olarak node'un static (bind pose) local transform'unu al
        Matrix4x4 nodeLocalTransform = convert(node->mTransformation);

        // Eğer bu düğüm için animasyon verisi varsa, animasyonlu lokal transformu hesapla
        if (animationMap.count(originalNodeName) > 0) {
            const AnimationData* anim = animationMap.at(originalNodeName);
            // AnimationData::calculateAnimationTransform animasyon keyframe'lerinden transform oluşturur
            // Blender'dan gelen animasyon keyframe'leri zaten objenin doğru pozisyonunu içerir
            // YENİ: Bind pose'u (nodeLocalTransform) varsayılan olarak gönderiyoruz.
            nodeLocalTransform = anim->calculateAnimationTransform(*anim, currentTime, originalNodeName, nodeLocalTransform);
        }
        // Animasyon yoksa, static transform kullanılır (yukarıda zaten atandı)

        // Parent'ın global transform'u ile bu node'un local transform'unu birleştir
        Matrix4x4 currentAnimatedGlobalTransform = parentAnimatedGlobalTransform * nodeLocalTransform;
        
        // STORE WITH UNIQUE NAME to match BoneData keys
        std::string uniqueNodeName = getUniqueName(originalNodeName);
        animatedGlobalTransformsStore[uniqueNodeName] = currentAnimatedGlobalTransform;

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

        // 2. Meshlere göre işle (Process each mesh)
        for (size_t meshIdx = 0; meshIdx < meshes.size(); ++meshIdx) {
            Mesh& mesh = meshes[meshIdx];
            
            // CRITICAL CHECK: Ensure index is valid to prevent crash (Kritik: Çökmeyi önlemek için indeksi kontrol et)
            if (mesh.originalMeshIndex >= scene->mNumMeshes) {
                SCENE_LOG_ERROR("Mesh index out of bounds in processBonesForMeshes: " + 
                    std::to_string(mesh.originalMeshIndex) + " >= " + std::to_string(scene->mNumMeshes));
                continue;
            }

            aiMesh* ai_mesh = scene->mMeshes[mesh.originalMeshIndex];
            if (!ai_mesh->HasBones()) continue;

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
                      
                        continue;
                    }

                    mesh.vertices[vertexId].boneWeights.emplace_back(boneIndex, weight);
                }
            }
        }
      
    }

    void buildBoneData(const aiScene* scene, BoneData& boneData) {
       // SCENE_LOG_INFO("[buildBoneData] Starting bone map generation...");

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
                std::string originalBoneName = bone->mName.C_Str();
                std::string boneName = getUniqueName(originalBoneName);

                if (boneData.boneNameToIndex.find(boneName) == boneData.boneNameToIndex.end()) {
                    unsigned int id = static_cast<unsigned int>(boneData.boneNameToIndex.size());
                    boneData.boneNameToIndex[boneName] = id;
                    
                    // Haritayı doldur - Node map uses ORIGINAL node names in hierarchy traverse, 
                    // BUT our boneNameToNode map needs to find the node.
                    // The nodeMap key is also ORIGINAL name (from C_Str()).
                    if (nodeMap.find(originalBoneName) != nodeMap.end()) {
                        boneData.boneNameToNode[boneName] = nodeMap[originalBoneName];
                    } else {
                         // SCENE_LOG_WARN("Bone node not found in hierarchy: " + boneName);
                    }
                    boneData.boneOffsetMatrices[boneName] = convert(bone->mOffsetMatrix);
                }
            }
        }
       // SCENE_LOG_INFO("[buildBoneData] Completed. Total unique bones: " + std::to_string(boneData.boneNameToIndex.size()));
        
        // Build reverse lookup table for O(1) index-to-name queries
        boneData.rebuildReverseLookup();
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
    // =========================================================================
    // RECURSIVE FUNCTION TO EXTRACT MESHES
    // Mesh Çıkarma İçin Özyinelemeli Fonksiyon
    // =========================================================================
    // Recursively traverses the node hierarchy to create Mesh objects.
    // Mesh nesnelerini oluşturmak için node hiyerarşisini özyinelemeli olarak dolaşır.
    // =========================================================================
    void processNodeToMeshes(aiNode* node, const aiScene* scene, std::vector<Mesh>& outMeshes, const BoneData& boneData, const std::vector<std::shared_ptr<Material>>& materials) {
        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            unsigned int meshIndex = node->mMeshes[i];
            aiMesh* ai_mesh = scene->mMeshes[meshIndex];
            
            // CRITICAL FIX: Pass meshIndex to ensure valid originalMeshIndex is set (Kritik Düzeltme: meshIndex gönderilmeli)
            Mesh mesh = processMesh(ai_mesh, node, scene, boneData, materials, meshIndex);
            
            outMeshes.push_back(std::move(mesh));
        }
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToMeshes(node->mChildren[i], scene, outMeshes, boneData, materials);
        }
    }

   // std::vector<std::shared_ptr<Camera>> cameras;
    // =========================================================================
    // RECURSIVE FUNCTION TO EXTRACT TRIANGLES
    // Üçgen Çıkarma İçin Özyinelemeli Fonksiyon
    // =========================================================================
    // Extracts triangles for ray tracing, applying global transforms and materials.
    // Ray tracing için üçgenleri çıkarır, global transform ve materyalleri uygular.
    // =========================================================================
    void processNodeToTriangles(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Triangle>>& triangles, const BoneData& boneData, OptixGeometryData* geometry_data = nullptr) {
        aiMatrix4x4 globalTransform = getGlobalTransform(node);

        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

            // Create HitGroupData (HitGroupData oluştur)
            HitGroupData hit_data = {};

            // Convert material and assign texture handles to hit_data
            // Materyali dönüştür ve texture handle'larını hit_data'ya ata
            auto convertedMaterial = processMaterial(material, scene, &hit_data, geometry_data);

            // Record current triangle count start
            // Mevcut üçgen başlangıç sayısını not al
            size_t triangleStart = triangles.size();

            // Extract triangles from mesh
            // Mesh'teki üçgenleri çıkar
            processTriangles(mesh, globalTransform, node->mName.C_Str(), convertedMaterial, triangles, boneData);

            // Copy texture bundle to all newly added triangles
            // Yeni eklenen tüm üçgenlere texture paketini kopyala
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

            // Backup TextureBundle for OptiX usage
            // OptiX kullanımı için TextureBundle yedeği
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

        // Process child nodes recursively
        // Çocuk düğümleri özyinelemeli olarak işle
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToTriangles(node->mChildren[i], scene, triangles, boneData, geometry_data);
        }
    }

    // =========================================================================
    // PROCESS SINGLE MESH (Tek Mesh İşle)
    // =========================================================================
    Mesh processMesh(aiMesh* mesh, aiNode* node, const aiScene* scene, const BoneData& boneData, const std::vector<std::shared_ptr<Material>>& materials, unsigned int meshIndex) {
         Mesh result;

         result.meshName = mesh->mName.C_Str();
         result.nodeName = node->mName.C_Str();
         result.localTransform = convertMatrix(node->mTransformation);
         
         // CRITICAL SAFEGUARD: Set the original mesh index for bone processing (Kritik Güvenlik: Kemik işleme için orijinal indeks)
         result.originalMeshIndex = meshIndex;

         // 1. Vertices (Vertexler)
         for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
             Vec3 pos = toVec3(mesh->mVertices[i]);
             Vec3 norm = mesh->HasNormals() ? toVec3(mesh->mNormals[i]) : Vec3(0);
             Vec2 uv = mesh->HasTextureCoords(0) ? Vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y) : Vec2(0);
             result.vertices.emplace_back(pos, norm, uv);
         }

         // 2. Indices (İndeksler)
         for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
             const aiFace& face = mesh->mFaces[i];
             if (face.mNumIndices != 3) continue;
             result.indices.push_back({ face.mIndices[0], face.mIndices[1], face.mIndices[2] });
         }

         // 3. Material (Materyal)
         if (mesh->mMaterialIndex < materials.size()) {
             result.material = materials[mesh->mMaterialIndex];
             result.materialIndex = mesh->mMaterialIndex;
         }

         // 4. Bone Weights
         // NOTE: This logic is partially redundant with processBonesForMeshes but kept for direct Mesh processing usage
         // NOT: Bu mantık processBonesForMeshes ile kısmen tekrarlı ama doğrudan Mesh işleme için kullanımı korundu
         if (mesh->HasBones()) {
             result.hasSkinning = true;
             for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                 aiBone* bone = mesh->mBones[i];
                 std::string originalBoneName = bone->mName.C_Str();
                 std::string boneName = getUniqueName(originalBoneName);
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
            std::string baseMatName = material->materialName.empty() 
                ? "Material_" + nodeName + "_" + std::to_string(mesh->mMaterialIndex)
                : material->materialName;
            
            // Prepend import name to ensure uniqueness across different imports
            // This prevents "Mesh B" from using "Mesh A" textures just because they both have "Material.001"
            std::string uniqueMatName = currentImportName.empty() ? baseMatName : (currentImportName + "_" + baseMatName);

            materialID = MaterialManager::getInstance().getOrCreateMaterialID(uniqueMatName, material);
        }

        // Use shared Transform for all meshes with the same nodeName
        // This ensures gizmo moves all parts of an object together
        auto sharedTransform = getOrCreateNodeTransform(nodeName, convertMatrix(transform));

        // --- NEW: Pre-process bone weights for this mesh ---
        // This ensures every vertex gets its correct weight list before triangle splits
        std::vector<std::vector<std::pair<int, float>>> meshVertexWeights(mesh->mNumVertices);
        if (mesh->HasBones()) {
            for (unsigned int i = 0; i < mesh->mNumBones; i++) {
                aiBone* bone = mesh->mBones[i];
                std::string originalBoneName = bone->mName.C_Str();
                std::string boneName = getUniqueName(originalBoneName);
                
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
        // Pre-check Vertex Colors
        bool hasColors0 = mesh->HasVertexColors(0);
        bool hasColors1 = mesh->HasVertexColors(1);
        // ---------------------------------------------------

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue;

            std::vector<Vec3> vertices;
            std::vector<Vec3> normals;
            std::vector<Vec2> texCoords;
            std::vector<Vec3> colors;

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

                // Color
                if (hasColors0) {
                    aiColor4D col = mesh->mColors[0][index];
                    colors.emplace_back(col.r, col.g, col.b);
                } else if (hasColors1) {
                    aiColor4D col = mesh->mColors[1][index];
                    colors.emplace_back(col.r, col.g, col.b);
                } else {
                    colors.emplace_back(0.0f, 0.0f, 0.0f);
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
            // Paylaşılan transformu ata (bu mesh'teki tüm üçgenler aynı transformu paylaşır)
            triangle->setTransformHandle(sharedTransform);
            
            // Apply initial transform (for start position)
            // İlk transform'u uygula (başlangıç pozisyonu için)
            triangle->updateTransformedVertices();



            triangle->setNodeName(nodeName);
            triangle->setAssimpVertexIndices(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            triangle->setAssimpVertexIndices(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            triangle->setFaceIndex(static_cast<int>(triangles.size()));

            // Assign Colors
            triangle->setVertexColor(0, colors[0]);
            triangle->setVertexColor(1, colors[1]);
            triangle->setVertexColor(2, colors[2]);

            // --- NEW: Assign weights to triangle vertices ---
            // --- YENİ: Üçgen vertexlerine ağırlıkları ata ---
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

    // =========================================================================
    // PROCESS CAMERAS (Kameraları İşle)
    // =========================================================================
    // Extracts and creates Camera objects from the scene.
    // Sahnedeki kamera nesnelerini çıkarır ve oluşturur.
    // =========================================================================
    void processCameras(const aiScene* scene) {
        try {
            if (!scene || !scene->HasCameras()) {
              //  SCENE_LOG_WARN("Does not include scene cameras...");
                return;
            }

            for (unsigned int i = 0; i < scene->mNumCameras; i++) {
                aiCamera* aiCam = scene->mCameras[i];
                if (!aiCam) continue;

                cameraNodeNames.insert(aiCam->mName.C_Str());
                aiNode* camNode = scene->mRootNode->FindNode(aiCam->mName.C_Str());
                if (!camNode) {
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

                // Fallback if aspect ratio is zero
                // Aspect oranı sıfırsa fallback uygula
                double aspect = (aiCam->mAspect > 0.01) ? aiCam->mAspect : 1.0;
                double hfov_rad = aiCam->mHorizontalFOV;
                double vfov_rad = 2.0 * atan(tan(hfov_rad / 2.0) / aspect);
                double vfov = vfov_rad * 180.0 / M_PI;

                // Create camera (Kamera oluştur)
                auto camera = std::make_shared<Camera>(
                    lookfrom, lookat, vup, (float)vfov, (float)aspect, aperture, focusdistance, 5
                );
                camera->nodeName = std::string(aiCam->mName.C_Str());
                camera->save_initial_state();  // Save initial state for reset functionality
                cameras.push_back(camera);
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

    // =========================================================================
    // PROCESS LIGHTS (Işıkları İşle)
    // =========================================================================
    // Extracts and creates Light objects (Point, Directional, Spot, Area).
    // Sahnedeki Işık nesnelerini (Noktasal, Yönsel, Spot, Alan) çıkarır ve oluşturur.
    // =========================================================================
    void processLights(const aiScene* scene) {
        try {
            if (!scene) {
              //  SCENE_LOG_ERROR("Light processing failed: Scene is null");
                return;
            }

            if (!scene->HasLights()) {
              //  SCENE_LOG_WARN("Scene has no lights, skipping light processing");
                return;
            }

           // SCENE_LOG_INFO("Total lights in scene: " + std::to_string(scene->mNumLights));

            for (unsigned int i = 0; i < scene->mNumLights; i++) {
                aiLight* aiLgt = scene->mLights[i];
                std::string name = aiLgt->mName.C_Str();
                std::string typeStr = LightTypeToString(aiLgt->mType);

                // [VERBOSE] SCENE_LOG_INFO("Processing light: " + name + " (" + typeStr + ")"); // Per-light log

                lightNodeNames.insert(name);

                const aiNode* node = scene->mRootNode->FindNode(aiLgt->mName);
                if (!node) {
                  //  SCENE_LOG_WARN("Node not found for light: " + name);
                    continue;
                }

                aiMatrix4x4 global = getGlobalTransform(node);

                Vec3 pos = transformPosition(global, aiLgt->mPosition);
                Vec3 dir = transformDirection(global, aiLgt->mDirection);

                if ((aiLgt->mType == aiLightSource_DIRECTIONAL || aiLgt->mType == aiLightSource_SPOT) && dir.length() == 0.0f) {
                   // SCENE_LOG_WARN("Direction zero-length detected, fallback used for: " + name);
                    dir = Vec3(0, 0, -1);
                }
                else if (dir.length() > 0.0f) {
                    dir = dir.normalize();
                }

                aiColor3D col = aiLgt->mColorDiffuse;
                
                // Preserve color tones, extract intensity properly
                // Renk tonlarını koru, yoğunluğu doğru çıkar
                Vec3 color(col.r, col.g, col.b);
                float maxComp = std::max({col.r, col.g, col.b});
                float intensity = 1.0f;
                
                if (maxComp > 1.0f) {
                    // HDR: normalize color, keep intensity
                    // HDR: rengi normalize et, yoğunluğu koru
                    intensity = maxComp;
                    color = color / maxComp;
                } else if (maxComp > 0.0f) {
                    // LDR: use as-is
                    // LDR: olduğu gibi kullan
                    intensity = 1.0f;
                    // FIX: Convert sRGB color to Linear space for correct rendering
                    // DÜZELTME: Doğru render için sRGB rengi Linear uzaya çevir
                    // This prevents colors from looking washed out/dark
                    // Bu, renklerin soluk/koyu görünmesini önler
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
                // Blender Watt cinsinden çıktı verir. 100'e bölmek daha iyi bir temel parlaklık sağlar.
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
                  //  SCENE_LOG_WARN("Unsupported light type: " + std::to_string(aiLgt->mType));
                    break;
                }

                if (!light) {
                  //  SCENE_LOG_ERROR("Failed to create light object: " + name);
                    continue;
                }

                light->nodeName = name;
                light->initialDirection = dir;
                lights.push_back(light);

              /*  SCENE_LOG_INFO(
                    "Light created: " + name +
                    " | Type: " + typeStr +
                    " | Pos: (" + std::to_string(pos.x) + ", " + std::to_string(pos.y) + ", " + std::to_string(pos.z) + ")" +
                    " | Dir: (" + std::to_string(dir.x) + ", " + std::to_string(dir.y) + ", " + std::to_string(dir.z) + ")" +
                    " | Color: (" + std::to_string(color.x) + ", " + std::to_string(color.y) + ", " + std::to_string(color.z) + ")" +
                    " | Intensity: " + std::to_string(intensity)
                );*/
            }
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR("Light processing fatal error: " + std::string(e.what()));
        }
    }

    // =========================================================================
    // PROCESS MATERIAL (Materyali İşle)
    // =========================================================================
    // Converts Assimp material to internal PrincipledBSDF or Volumetric material.
    // Handles texture extraction and assignment.
    //
    // Assimp materyalini dahili PrincipledBSDF veya Volumetric materyale dönüştürür.
    // Doku çıkarımı ve atamasını yönetir.
    // =========================================================================
     std::shared_ptr<Material> processMaterial(
        aiMaterial* aiMat,
        const aiScene* scene,
        HitGroupData* hit_data = nullptr,
        OptixGeometryData* geometry_data = nullptr // ← added (eklendi)
    )

    {
       auto material = std::make_shared<PrincipledBSDF>();
       aiString str;
       textureInfos.clear();
       // Material type check (Materyal tipi kontrolü)
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
       // [VERBOSE] SCENE_LOG_INFO("Process Material: " + materialNameStr); // Disabled - high volume
       for (auto type : textureTypes) {

           // Ensure texture count is greater than 0
           // Texture sayısının 0'dan büyük olduğundan emin ol
           if (aiMat->GetTextureCount(type) > 0) {
               aiString str;
               int count = aiMat->GetTextureCount(type);
               if (count > 0) {
                   // [VERBOSE] SCENE_LOG_INFO("Material has texture: type=" + std::to_string(type) +
                   //     " count=" + std::to_string(count)); // Per-texture log
               }
               // Retrieve the texture, check for success and ensure the texture path is valid
               // Doku adını al, başarıyı kontrol et ve yolun geçerli olduğundan emin ol
               if (AI_SUCCESS == aiMat->GetTexture(type, 0, &str) && str.length > 0) {
                   // Perform the sanitization and push to textureInfos if the texture name is not empty
                   // Temizleme işlemini yap ve ad boş değilse textureInfos'a ekle
                   std::string sanitizedName = sanitizeTextureName(str);
                   if (!sanitizedName.empty()) {
                       textureInfos.push_back({ type, sanitizedName });
                   }
               }
           }
       }

       aiColor3D color(0.0f, 0.0f, 0.0f);
       aiMat->Get(AI_MATKEY_COLOR_DIFFUSE, color);
       // Convert sRGB to Linear for CPU material property
       // CPU materyal özelliği için sRGB'den Lineer'e dönüştür
       material->albedoProperty = MaterialProperty(Vec3(powf(color.r, 2.2f), powf(color.g, 2.2f), powf(color.b, 2.2f)), 1.0f);
      
       // Roughness (Pürüzlülük)
       float roughness = 0.0f;
       aiMat->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
       material->roughnessProperty = MaterialProperty( Vec3(std::clamp(roughness, 0.0f, 1.0f)));
	  
       // Metallic (Metalik)
       aiColor3D colorM(0.0f, 0.0f, 0.0f);
       float metalicFactor = 0.0;
       aiMat->Get(AI_MATKEY_METALLIC_FACTOR, metalicFactor);
       material->metallicProperty = MaterialProperty(Vec3(color.r, color.g, color.b), std::clamp(metalicFactor, 0.0f, 1.0f));

       // Emissive Color (Yayılan Işık Rengi)
       aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
       aiMat->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor);

       // Use this value if emissiveFactor is greater than zero
       // Eğer `emissiveFactor` sıfırdan büyükse, bu değeri kullan
       float emissiveStrength = std::max({ emissiveColor.r, emissiveColor.g, emissiveColor.b });
	   
       // Apply to material (Materyale uygula)
       // GPU uses emissiveColor directly (without multiplying by intensity)
       // So CPU's emissionProperty.intensity must be 1.0f for GPU parity
       // The color already contains the emission strength (max component)
       float hasEmission = (emissiveColor.r > 0.001f || emissiveColor.g > 0.001f || emissiveColor.b > 0.001f) ? 1.0f : 0.0f;
       material->emissionProperty = MaterialProperty(Vec3(emissiveColor.r, emissiveColor.g, emissiveColor.b), hasEmission);
	   //std::cout << "emissiveStrength : " << emissiveStrength << std::endl;
     
       // Specular Reflection (Glossiness) (Speküler Yansıma)
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
       // Clamp IOR to be at least 1.0f
       ior = std::max(ior, 1.0f);
       material->ior = ior;
	  // std::cout << "Material IOR: " << ior << std::endl;
       float transmission = 0.0f;
       if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_TRANSMISSION_FACTOR, &transmission)) {
           material->setTransmission(std::clamp(transmission, 0.0f, 1.0f), ior);
          
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
               material->opacityProperty.alpha = std::clamp(opacity, 0.0f, 1.0f);
              
           }

            std::shared_ptr<Texture> texture;

            for (const auto& texInfo : textureInfos)
            {
                const std::string& path = texInfo.path;
                TextureType ttype = convertToTextureType(texInfo.type);

                // Check if embedded texture (Embedded mi?)
                const aiTexture* emb = scene->GetEmbeddedTexture(path.c_str());

                if (emb) {
                    // Create unique cache key for embedded texture
                    // Embedded texture için unique cache key oluştur
                    // Format: "embedded_<ImportName>_<pointer_address>_<type>"
                    // Adding currentImportName is CRITICAL to prevent collisions if memory addresses are reused across imports (ABA Problem)
                    std::string embeddedKey = "embedded_" + 
                        currentImportName + "_" +
                        std::to_string(reinterpret_cast<uintptr_t>(emb)) + 
                        "_" + std::to_string(static_cast<int>(ttype));

                    // Check if in cache (Cache'de var mı kontrol et)
                    auto cacheIt = textureCache.find(embeddedKey);
                    if (cacheIt != textureCache.end()) {
                        texture = cacheIt->second;
                    } else {
                        // Not in cache, create new and add to cache
                        // Cache'de yok, yeni oluştur ve cache'e ekle
                        texture = std::make_shared<Texture>(emb, ttype, embeddedKey);
                        textureCache[embeddedKey] = texture;
                    }
                }
                else
                    texture = loadTextureWithCache(path, ttype);

              
               if (!texture || !texture->is_loaded())
                   continue;

               // Assign to CPU material in all cases (CPU tarafı materyale her durumda ata)
               switch (texInfo.type)
               {
               case aiTextureType_DIFFUSE:
               case 12: // aiTextureType_BASE_COLOR
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
                   // Fix: If emission color was black (causing intensity 0), but we have a texture,
                   // we must enable emission (intensity 1.0, color white) so the texture is visible.
                   if (material->emissionProperty.intensity < 0.001f) {
                       material->emissionProperty.intensity = 1.0f;
                       material->emissionProperty.color = Vec3(1.0f); 
                   }
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

               // Try GPU upload (only once, skip if already uploaded)
               // GPU yüklemeyi dene (sadece bir kez, zaten is_gpu_uploaded flag'i varsa atla)
               bool gpu_ok = false;
               if (g_hasOptix && !texture->is_gpu_uploaded) {
                   if (!texture->upload_to_gpu()&& g_hasOptix) {
                     //  SCENE_LOG_ERROR("Texture GPU upload failed: " + texInfo.path);
                       gpu_ok = false;
                   }
                   else {
                       gpu_ok = true;
                   }
               }
               else {
                   gpu_ok = true;
               }

               // Assign GPU handle to hit_data: ONLY if gpu_ok is true
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
                          // SCENE_LOG_WARN("Texture marked uploaded but cuda texture is null: " + texInfo.path);
                       }
                   }
                   else {
                       // No GPU/Failed: leave hit_data fields as 0
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

               // Create noise (Gürültü oluştur)
               auto noise = std::make_shared<Perlin>();

               // Create volumetric material (Volumetrik materyali oluştur)
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

           // Texture varsa dummy (1.0), yoksa gerçek değer (Linear Space'e çevirerek) atıyoruz:
           if (hit_data && hit_data->has_albedo_tex) {
               gpu->albedo = make_float3(1.0f, 1.0f, 1.0f);
           }
           else {
               // FIX: Convert scalar Albedo from sRGB to Linear to match lighting pipeline
               // Prevents "washed out" look for untextured materials
               gpu->albedo = make_float3(powf(color.r, 2.2f), powf(color.g, 2.2f), powf(color.b, 2.2f));
           }

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
               gpu->opacity = 1.0f; // FIX: Was 0.0f! Must be 1.0f to allow texture modulation.
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
        // Calculate Full Path FIRST to ensure uniqueness in cache across different imports
        std::filesystem::path texPath(name);
        if (!texPath.is_absolute())
            texPath = std::filesystem::path(baseDirectory) / texPath;
        texPath = texPath.lexically_normal();
        
        // Use Full Path in Cache Key
        // This fixes the issue where 'wood.jpg' in Model1 uses the texture from Model2 by mistake
        std::string cacheKey = texPath.string() + "_" + std::to_string(static_cast<int>(type));

        auto it = textureCache.find(cacheKey);
        if (it != textureCache.end())
            return it->second;

        // SCENE_LOG_INFO("Final texture path: " + texPath.string()); 
        // SCENE_LOG_INFO("Path exists: " + std::to_string(std::filesystem::exists(texPath)));  

        auto tex = std::make_shared<Texture>(texPath.string(), type);

        if (!tex->is_loaded()) {  // ← Bu kontrolü ekle!
           // SCENE_LOG_ERROR("Texture is_loaded() = false after constructor: " + texPath.string());
            return nullptr;
        }

        if (g_hasOptix) {
            if (!tex->upload_to_gpu()) {
               // SCENE_LOG_ERROR("Disk texture GPU upload failed: " + texPath.string());
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
