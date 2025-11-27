#pragma once

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <memory>
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

    Matrix4x4 calculateAnimationTransform(const AnimationData& animData, float time, const std::string& nodeName)
        const {
        // Animasyon süresini normalize et
        double animationTime = fmod(time * ticksPerSecond, duration);

        // Varsayılan dönüşüm değerleri
        Vec3 position(0, 0, 0);
        Quaternion rotation(0, 0, 0, 1);
        Vec3 scaling(1, 1, 1);

        // Position interpolasyonu
        auto posIt = positionKeys.find(nodeName);
        if (posIt != positionKeys.end() && !posIt->second.empty()) {
            const auto& keys = posIt->second;

            size_t frameIndex = 0;
            for (size_t i = 0; i < keys.size() - 1; i++) {
                if (animationTime < keys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = keys[nextFrameIndex].mTime - keys[frameIndex].mTime;
            if (deltaTime < 0) deltaTime += duration;

            float factor = (deltaTime == 0) ? 0.0 :
                (animationTime - keys[frameIndex].mTime) / deltaTime;

            const auto& start = keys[frameIndex].mValue;
            const auto& end = keys[nextFrameIndex].mValue;
            position = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
        }

        // Rotation interpolasyonu
        auto rotIt = rotationKeys.find(nodeName);
        if (rotIt != rotationKeys.end() && !rotIt->second.empty()) {
            const auto& keys = rotIt->second;

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

            Quaternion start(
                keys[frameIndex].mValue.x,
                keys[frameIndex].mValue.y,
                keys[frameIndex].mValue.z,
                keys[frameIndex].mValue.w
            );

            Quaternion end(
                keys[nextFrameIndex].mValue.x,
                keys[nextFrameIndex].mValue.y,
                keys[nextFrameIndex].mValue.z,
                keys[nextFrameIndex].mValue.w
            );

            rotation = Quaternion::slerp(start, end, factor);
        }

        // Scaling interpolasyonu
        auto scaleIt = scalingKeys.find(nodeName);
        if (scaleIt != scalingKeys.end() && !scaleIt->second.empty()) {
            const auto& keys = scaleIt->second;

            size_t frameIndex = 0;
            for (size_t i = 0; i < keys.size() - 1; i++) {
                if (animationTime < keys[i + 1].mTime) {
                    frameIndex = i;
                    break;
                }
            }

            size_t nextFrameIndex = (frameIndex + 1) % keys.size();
            float deltaTime = keys[nextFrameIndex].mTime - keys[frameIndex].mTime;
            if (deltaTime < 0) deltaTime += duration;

            float factor = (deltaTime == 0) ? 0.0 :
                (animationTime - keys[frameIndex].mTime) / deltaTime;

            const auto& start = keys[frameIndex].mValue;
            const auto& end = keys[nextFrameIndex].mValue;
            scaling = Vec3(
                start.x + (end.x - start.x) * factor,
                start.y + (end.y - start.y) * factor,
                start.z + (end.z - start.z) * factor
            );
        }

        // Dönüşüm matrislerini oluştur
        Matrix4x4 translationMatrix = Matrix4x4::translation(position);
        Matrix4x4 rotationMatrix = rotation.toMatrix();
        Matrix4x4 scaleMatrix = Matrix4x4::scaling(scaling);

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

        const aiNode* current = node->mParent; // 🔥 Sadece parent'ları alacağız
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
        float focus_dist = 2.91;
        Vec3 lookfrom1(7.35889, 4.95831, 6.92579);
        Vec3 lookat1(0.0, 0.0, 0.0);
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
        BoneData boneData;

        this->scene = importer.ReadFile(filename,
            //aiProcess_Triangulate |           // Önce üçgenlere böl
            aiProcess_GenSmoothNormals |      // Normalleri oluştur
             aiProcess_JoinIdenticalVertices  //Aynı olan vertexleri birleştir
          //  aiProcess_GenNormals              // Normal haritaları oluşturur
         //   aiProcess_CalcTangentSpace       // Tangent ve bitangent hesapla

        );
        //importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 45.0f);  // 45 derece örnek açıdır, bunu ihtiyacınıza göre değiştirebilirsiniz.
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
            return {};
        }
        if (!scene || !scene->mRootNode) {
            std::cerr << "Assimp load error: " << importer.GetErrorString() << std::endl;
            return {};
        }
        cameras.clear(); // ❗ Kameraları her model yüklemesinde sıfırla
        lights.clear();  // Aynı mantık ışıklar için de geçerli
      
        // Node ağacını tarayıp nodeMap'e doldur
        std::function<void(aiNode*)> recurse = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                recurse(node->mChildren[i]);
            };
        recurse(scene->mRootNode);
        processCameras(scene);
        processLights(scene);
        std::vector<std::shared_ptr<Triangle>> triangles;
       
        OptixGeometryData geometry_data;

        processNodeToTriangles(scene->mRootNode, scene, triangles, &geometry_data);
        std::unordered_map<std::string, unsigned int> boneNameToIndex;
        std::unordered_map<std::string, aiNode*> boneNameToNode;

        processBones(scene, triangles, boneData);
      
        std::vector<AnimationData> animationDataList;
        if (scene->mNumAnimations > 0) {
            std::cout << "Animations found, loading..." << std::endl;
            for (unsigned int i = 0; i < scene->mNumAnimations; ++i) {
                const aiAnimation* animation = scene->mAnimations[i];
                AnimationData animData;

                animData.name = animation->mName.C_Str();
                animData.duration = animation->mDuration;
                animData.ticksPerSecond = animation->mTicksPerSecond;

                std::cout << "Loaded Animation: " << animData.name
                    << ", Duration: " << animData.duration
                    << ", Ticks Per Second: " << animData.ticksPerSecond << std::endl;

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
                }
                animationDataList.push_back(animData);
            }
        }
        else {
            std::cout << "No animations found in the file: " << filename << std::endl;
        }
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
            std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
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

            Vec3 verts[3] = { tri->transformed_v0, tri->transformed_v1, tri->transformed_v2 };
            Vec3 norms[3] = { tri->transformed_n0, tri->transformed_n1, tri->transformed_n2 };
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

            // Benzersiz materyal + texture anahtarı
            GpuMaterialWithTextures key;
            key.material = *tri->mat_ptr->gpuMaterial;
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
                data.textures.push_back(tri->textureBundle); // textures paralel sırada tutulur
                materialMap[key] = gpuIndex;

               /* std::cout << "[GpuMaterial] Yeni materyal eklendi: "
                    << tri->mat_ptr->materialName << " -> Index " << gpuIndex << "\n";*/
            }

            data.material_indices.push_back(gpuIndex);
        }

        data.materials = std::move(gpuMaterials);
        return data;
    }


    void clearTextureCache() {
        for (auto& [name, tex] : textureCache) {
            if (tex) tex->cleanup_gpu(); // GPU belleği temizle
        }
        textureCache.clear(); // CPU cache'i temizle
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
        Matrix4x4 nodeLocalTransform = convert(node->mTransformation); // Varsayılan olarak bind pose lokal transform

        // Eğer bu düğüm için animasyon verisi varsa, animasyonlu lokal transformu hesapla
        if (animationMap.count(nodeName) > 0) {
            const AnimationData* anim = animationMap.at(nodeName);
            // AnimationData::calculateAnimationTransform zaten lokal animasyon transformunu döndürmeli
            nodeLocalTransform = anim->calculateAnimationTransform(*anim, currentTime, nodeName);
        }

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

    void processBones(
        const aiScene* scene,
        std::vector<std::shared_ptr<Triangle>>& triangles,
        BoneData& boneData
    ) {
        std::cout << "[processBones] Starting...\n";

        // 0. Temizle
        boneData.boneNameToIndex.clear();
        boneData.boneNameToNode.clear();
        boneData.boneOffsetMatrices.clear();

        // 1. Global inverse transform (root node'dan)
        boneData.globalInverseTransform = convert(scene->mRootNode->mTransformation).inverse();
        std::cout << "[processBones] Global inverse transform received.\n";

        // 2. Node map oluştur (bone isimlerini düğümle eşleştirmek için)
        std::unordered_map<std::string, aiNode*> nodeMap;
        std::function<void(aiNode*)> collectNodes = [&](aiNode* node) {
            nodeMap[node->mName.C_Str()] = node;
            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                collectNodes(node->mChildren[i]);
            };
        collectNodes(scene->mRootNode);
        std::cout << "[processBones] Node map created. Total: " << nodeMap.size() << " node.\n";

        // 3. Meshleri dolaş
        for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
            aiMesh* mesh = scene->mMeshes[meshIndex];
            if (!mesh->HasBones()) continue;

            std::cout << "[Mesh " << meshIndex << "] Number of bones: " << mesh->mNumBones << std::endl;

            for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex) {
                aiBone* bone = mesh->mBones[boneIndex];
                std::string boneName = bone->mName.C_Str();

                // 3.1. Bone index
                if (boneData.boneNameToIndex.count(boneName) == 0)
                    boneData.boneNameToIndex[boneName] = static_cast<unsigned int>(boneData.boneNameToIndex.size());

                unsigned int globalBoneIndex = boneData.boneNameToIndex[boneName];

                // 3.2. Node eşlemesi
                boneData.boneNameToNode[boneName] = nodeMap.count(boneName) ? nodeMap[boneName] : nullptr;

                // 3.3. Offset matrix
                boneData.boneOffsetMatrices[boneName] = convert(bone->mOffsetMatrix);

                // 3.4. Ağırlıkları işle
                for (unsigned int w = 0; w < bone->mNumWeights; ++w) {
                    unsigned int vertexId = bone->mWeights[w].mVertexId;
                    float weight = bone->mWeights[w].mWeight;
                    bool found = false;

                    for (auto& tri : triangles) {
                        const auto& indices = tri->getAssimpVertexIndices();

                        for (int vi = 0; vi < 3; ++vi) {
                            if (indices[vi] == vertexId) {
                                if (tri->vertexBoneWeights.size() != 3)
                                    tri->vertexBoneWeights.resize(3);
                                if (tri->originalVertexPositions.size() != 3)
                                    tri->originalVertexPositions.resize(3);

                                tri->vertexBoneWeights[vi].emplace_back(globalBoneIndex, weight);

                                if (vi == 0) tri->originalVertexPositions[0] = tri->original_v0;
                                if (vi == 1) tri->originalVertexPositions[1] = tri->original_v1;
                                if (vi == 2) tri->originalVertexPositions[2] = tri->original_v2;

                                found = true;
                                break;
                            }
                        }
                        if (found) break;
                    }

                    if (!found) {
                       // std::cerr << "[WARNING] vertexId " << vertexId << " için triangle bulunamadı!\n";
                    }
                }
            }
        }

        std::cout << "[processBones] Completed. Total number of bones: " << boneData.boneNameToIndex.size() << std::endl;
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
     void processNodeToTriangles(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Triangle>>& triangles, OptixGeometryData* geometry_data = nullptr) {
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
            processTriangles(mesh, globalTransform, node->mName.C_Str(), convertedMaterial, triangles);

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
            processNodeToTriangles(node->mChildren[i], scene, triangles, geometry_data);
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
        std::vector<std::shared_ptr<Triangle>>& triangles)
    {
        aiMatrix4x4 normalTransform = transform;
        normalTransform.Inverse();
        normalTransform.Transpose();

        for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
            const aiFace& face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue; // Sadece üçgen yüzleri işleyin

            std::vector<Vec3> vertices;
            std::vector<Vec3> normals;
            std::vector<Vec2> texCoords;
          

            for (unsigned int j = 0; j < 3; j++) {
                unsigned int index = face.mIndices[j];

                // Vertex
                aiVector3D vertex = mesh->mVertices[index];
                aiVector3D transformedVertex = transform * vertex;
                vertices.emplace_back(transformedVertex.x, transformedVertex.y, transformedVertex.z);

                // Normal
                if (mesh->HasNormals()) {
                    aiVector3D normal = mesh->mNormals[index];
                    aiVector3D transformedNormal = normalTransform * normal;
                    transformedNormal.Normalize();
                    normals.emplace_back(transformedNormal.x, transformedNormal.y, transformedNormal.z);
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

            auto triangle = std::make_shared<Triangle>(
                vertices[0], vertices[1], vertices[2],
                normals[0], normals[1], normals[2],
                texCoords[0], texCoords[1], texCoords[2],
              
                material,
                mesh->mMaterialIndex
            );

            triangle->mat_ptr = material;
            triangle->gpuMaterialPtr = material->gpuMaterial;
            triangle->setNodeName(nodeName);
            triangle->setAssimpVertexIndices(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
            triangle->setFaceIndex(static_cast<int>(triangles.size()));

            triangles.push_back(triangle);
        }
    }

    void processCameras(const aiScene* scene) {
        try {
            if (!scene || !scene->HasCameras()) {
                std::cout << "Does not include scene cameras..." << std::endl;
                return;
            }

            for (unsigned int i = 0; i < scene->mNumCameras; i++) {
                aiCamera* aiCam = scene->mCameras[i];
                if (!aiCam) continue;

                cameraNodeNames.insert(aiCam->mName.C_Str());
                aiNode* camNode = scene->mRootNode->FindNode(aiCam->mName.C_Str());
                if (!camNode) {
                    std::cerr << "Camera node not found: " << aiCam->mName.C_Str() << std::endl;
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
                    lookfrom, lookat, vup, vfov, aspect, aperture, focusdistance, 5
                );
                camera->nodeName = std::string(aiCam->mName.C_Str());

                cameras.push_back(camera);

                // Loglama
                std::cout << "[Camera Loaded] " << aiCam->mName.C_Str() << "\n";
                std::cout << "  Position: " << lookfrom.toString() << "\n";
                std::cout << "  LookAt  : " << lookat.toString() << "\n";
                std::cout << "  Up      : " << vup.toString() << "\n";
                std::cout << "  Aspect  : " << aspect << "\n";
                std::cout << "  FOV     : " << vfov << " degrees\n\n";
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Camera processing error: " << e.what() << std::endl;
        }
    }

     void processLights(const aiScene* scene) {
         try {
             if (!scene || !scene->HasLights()) {
                 std::cout << "There is no light on the stage, the process continues..." << std::endl;
                 return;
             }

             for (unsigned int i = 0; i < scene->mNumLights; i++) {
                 aiLight* aiLgt = scene->mLights[i];
                 lightNodeNames.insert(std::string(aiLgt->mName.C_Str()));

                 // --- Global transform al ---
                 const aiNode* node = scene->mRootNode->FindNode(aiLgt->mName);
                 aiMatrix4x4 globalTransform = node ? getGlobalTransform(node) : aiMatrix4x4();

                 // --- Pozisyon ve yönü transform et ---
                 Vec3 position = transformPosition(globalTransform, aiLgt->mPosition);
                 Vec3 direction = transformDirection(globalTransform, aiLgt->mDirection);

                 // --- Renk ve güç ayrımı ---
                 aiColor3D col = aiLgt->mColorDiffuse;
                 Vec3 raw_intensity(col.r, col.g, col.b);
                 float power = raw_intensity.length(); // toplam ışık gücü
                 Vec3 color = (power > 0.0f) ? raw_intensity / power : Vec3(1.0f);
                 power /= 1000.0f; // isteğe bağlı ölçeklendirme

                 // --- Ortak pointer ---
                 std::shared_ptr<Light> light = nullptr;

                 // --- Türüne göre ışık oluştur ---
                 if (aiLgt->mType == aiLightSource_DIRECTIONAL) {
                     light = std::make_shared<DirectionalLight>(direction, color * power, 10.0f);
                     light->position = position;
                 }
                 else if (aiLgt->mType == aiLightSource_POINT) {
                     light = std::make_shared<PointLight>(position, color * power, 0.1f);
                 }
                 else if (aiLgt->mType == aiLightSource_SPOT) {
                     float angle_degrees = aiLgt->mAngleInnerCone * (180.0f / M_PI);
                     float radius = 10.0f;
                     light = std::make_shared<SpotLight>(position, direction, color * power, angle_degrees, radius);
                 }
                 else if (aiLgt->mType == aiLightSource_AREA) {
                     Vec3 forward = transformDirection(globalTransform, aiLgt->mDirection).normalize();
                     Vec3 up = transformDirection(globalTransform, aiLgt->mUp).normalize();
                     Vec3 u = Vec3::cross(forward, up).normalize();
                     Vec3 v = Vec3::cross(u, forward).normalize();
                     double width = aiLgt->mSize.x;
                     double height = aiLgt->mSize.y;
                     light = std::make_shared<AreaLight>(position, u, v, width, height, color * power);
                 }

                 // --- Işık oluşturulduysa kaydet ---
                 if (light) {
                     light->nodeName = std::string(aiLgt->mName.C_Str());
                     light->initialDirection = direction;

                     lights.push_back(light);

                     std::cout << "Added light: " << aiLgt->mName.C_Str()
                         << " Type: " << aiLgt->mType
                         << " Position: (" << position.x << ", " << position.y << ", " << position.z << ")"
                         << " Color: " << color
                         << " Power: " << power << std::endl;
                 }
                 else {
                     std::cerr << "Unsupported light type: " << aiLgt->mType << std::endl;
                 }
             }

         }
         catch (const std::exception& e) {
             std::cerr << "Error occurred during light processing: " << e.what() << std::endl;
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

       for (auto type : textureTypes) {
           // Ensure texture count is greater than 0
           if (aiMat->GetTextureCount(type) > 0) {
               aiString str;

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

          
       // Texture'ları yükle ve ata
           for (const auto& texInfo : textureInfos) {
               auto texture = loadTextureWithCache(texInfo.path, convertToTextureType(texInfo.type));

               if (!texture || !texture->is_loaded())
                   continue; // ❌ Texture yüklenmemiş veya geçersiz

               // CPU materyale ata
               switch (texInfo.type) {
               case aiTextureType_DIFFUSE:
                   material->albedoProperty.texture = texture;

                   if (texture->has_alpha) {
                       material->opacityProperty.texture = texture;
                       material->opacityProperty.alpha = opacity;
                   }

                   if (hit_data) {
                       hit_data->albedo_tex = texture->get_cuda_texture();
                       hit_data->has_albedo_tex = 1;
                   }
                   break;

               case aiTextureType_SPECULAR: // roughness yerine
                   material->specularProperty.texture = texture;
                   if (hit_data) {
                       hit_data->roughness_tex = texture->get_cuda_texture();
                       hit_data->has_roughness_tex = 1;
                   }
                   break;

               case aiTextureType_NORMALS:
                   material->normalProperty.texture = texture;
                   if (hit_data) {
                       hit_data->normal_tex = texture->get_cuda_texture();
                       hit_data->has_normal_tex = 1;
                   }
                   break;

               case aiTextureType_EMISSIVE:
                   material->emissionProperty.texture = texture;
                   if (hit_data) {
                       hit_data->emission_tex = texture->get_cuda_texture();
                       hit_data->has_emission_tex = 1;
                   }
                   break;

               case aiTextureType_OPACITY:
                   material->opacityProperty.texture = texture;
                   material->opacityProperty.intensity = 1.0f;
                   if (hit_data) {
                       hit_data->opacity_tex = texture->get_cuda_texture();
                       hit_data->has_opacity_tex = 1;
                   }
                   break;

               case aiTextureType_METALNESS:
                   material->metallicProperty.texture = texture;
                   if (hit_data) {
                       hit_data->metallic_tex = texture->get_cuda_texture();
                       hit_data->has_metallic_tex = 1;
                   }
                   break;

               case aiTextureType_DIFFUSE_ROUGHNESS:
                   material->roughnessProperty.texture = texture;
                   if (hit_data) {
                       hit_data->roughness_tex = texture->get_cuda_texture();
                       hit_data->has_roughness_tex = 1;
                   }
                   break;

               case aiTextureType_AMBIENT_OCCLUSION:
                   // AO destekliyorsan buraya ekle
                   break;
               }
           }


           if (materialNameStr.find("sss") != std::string::npos || materialNameStr.find("subsurface") != std::string::npos) {
              
               material->setSubsurfaceScattering((1.0f, 0.8f, 0.5f), (1.0f, 1.0f, 1.0f));
            
           }

           if (materialNameStr.find("volume") != std::string::npos || materialNameStr.find("volumetric") != std::string::npos) {
               Vec3 albedo = Vec3(color.r, color.g, color.b); // Açık mavi hacim için uygun
               float density = 0.8f;               // Daha düşük yoğunluk genellikle görselde güzel durur
               float scattering_factor = 0.5f;     // Saçılma etkisi (Heney-Greenstein)
               float absorption_probability = 0.25f;
               Vec3 emission = albedo; // Hafif morumsu bir emisyon, daha doğal

               // Gürültü oluştur
               auto noise = std::make_shared<Perlin>();

               // Volumetrik materyali oluştur
               auto volumetric_material = std::make_shared<Volumetric>(
                   albedo, density, absorption_probability, scattering_factor, emission, noise
               );

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
       auto texture = std::make_shared<Texture>(fullPath, type); // 👈 yeni parametre

       if (!opacityMapPath.empty()) {
           std::string fullOpacityMapPath = baseDirectory + opacityMapPath;
           texture->loadOpacityMap(fullOpacityMapPath);
       }

       return texture;
   }

    std::string sanitizeTextureName(const aiString& str) {
       std::string textureName = str.C_Str();
       size_t pos = 0;
       while ((pos = textureName.find("%20", pos)) != std::string::npos) {
           textureName.replace(pos, 3, " ");
           pos += 1;
       }
       return textureName;
   }
     void processMaterialTexture(aiMaterial* aiMat, aiTextureType type, MaterialProperty& property) {
       aiString str;
       if (aiMat->GetTextureCount(type) > 0) {
           aiMat->GetTexture(type, 0, &str);
           std::string textureName = sanitizeTextureName(str);
           property.texture = loadTexture(textureName.c_str());
       }
   }
    TextureType convertToTextureType(aiTextureType type) {
       switch (type) {
       case aiTextureType_DIFFUSE: return TextureType::Albedo;
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

    std::shared_ptr<Texture> loadTextureWithCache(const std::string& textureName, TextureType type = TextureType::Unknown) {
       auto it = textureCache.find(textureName);
       if (it != textureCache.end()) {
           return it->second; //  Zaten yüklü -> direkt dön
       }

       // İlk kez yükleniyor
       auto texture = std::make_shared<Texture>(baseDirectory + textureName, type);
       textureCache[textureName] = texture;

       texture->upload_to_gpu(); // İlk ve tek GPU yükleme

       return texture;
   }

};
