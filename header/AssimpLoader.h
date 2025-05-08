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

static std::vector<std::shared_ptr<Camera>> cameras; // Kamera listesi
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

    std::vector<std::shared_ptr<Triangle>> loadMeshByIndex(int meshIndex) {
        std::vector<std::shared_ptr<Triangle>> triangles;

        if (!scene || meshIndex >= scene->mNumMeshes) {
            std::cerr << "Geçersiz mesh index: " << meshIndex << std::endl;
            return triangles;
        }

        aiMesh* mesh = scene->mMeshes[meshIndex];
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        auto convertedMaterial = processMaterial(material, scene);

        // Global transform'i sıfır alıyoruz, çünkü instancing için zaten ayrı uygulanacak
        aiMatrix4x4 identity;
        std::string nodeName = "mesh_" + std::to_string(meshIndex);

        AssimpLoader::processTriangles(mesh, identity, nodeName, convertedMaterial, triangles);

        return triangles;
    }

    // AssimpLoader'a eklenecek yeni metod
    std::vector<MeshInstance> loadModelToInstances(const std::string& filename) {
        std::vector<MeshInstance> instances;

        this->scene = importer.ReadFile(filename,
            aiProcess_GenSmoothNormals |
            aiProcess_GenNormals |
            aiProcess_CalcTangentSpace
        );

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
            return instances;
        }

        std::function<void(aiNode*)> recurse = [&](aiNode* node) {
            Matrix4x4 transform = convert(getGlobalTransform(node));

            for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
                int meshIndex = node->mMeshes[i];
                instances.push_back({ meshIndex, transform, node->mName.C_Str() });
            }

            for (unsigned int i = 0; i < node->mNumChildren; ++i)
                recurse(node->mChildren[i]);
            };
        recurse(scene->mRootNode);

        return instances;
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
    Matrix4x4 getBoneTransformAtTime(
        const AnimationData& animData,
        float time,
        const std::string& boneName,
        aiNode* boneNode)
    {
        // 1. Animasyon dönüşümü (animData'dan al)
        Matrix4x4 animTransform = animData.calculateAnimationTransform(animData, time, boneName);

        // 2. Node'un parent zincirinden gelen dönüşüm (static transformlar)
        aiMatrix4x4 nodeTransform = boneNode ? boneNode->mTransformation : aiMatrix4x4();

        aiNode* parent = boneNode ? boneNode->mParent : nullptr;
        while (parent)
        {
            nodeTransform = parent->mTransformation * nodeTransform;
            parent = parent->mParent;
        }

        Matrix4x4 globalTransform = convert(nodeTransform);
        return animTransform * globalTransform;
    }

    // Hem Triangle hem de AnimationData döndüren metod
    std::pair<std::vector<std::shared_ptr<Triangle>>, std::vector<AnimationData>>
        loadModelToTriangles(const std::string& filename, const std::shared_ptr<Material>& material = nullptr) {

        this->scene = importer.ReadFile(filename,
            //aiProcess_Triangulate |           // Önce üçgenlere böl
            aiProcess_GenSmoothNormals |      // Normalleri oluştur
            //  aiProcess_JoinIdenticalVertices | //Aynı olan vertexleri birleştir
            aiProcess_GenNormals |             // Normal haritaları oluşturur
            aiProcess_CalcTangentSpace       // Tangent ve bitangent hesapla

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

        processBones(scene, triangles, boneNameToIndex, boneNameToNode);

      
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
                    std::cout << "Node: " << nodeName
                        << ", Position Keys: " << channel->mNumPositionKeys
                        << ", Rotation Keys: " << channel->mNumRotationKeys
                        << ", Scaling Keys: " << channel->mNumScalingKeys << std::endl;
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
        return { triangles, animationDataList };
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

    static aiMatrix4x4 getGlobalTransform(const aiNode* node) {
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
    OptixGeometryData convertTrianglesToOptixData(const std::vector<std::shared_ptr<Triangle>>& triangles) {
        OptixGeometryData data;
        std::unordered_map<GpuMaterial, int> materialMap;
        std::unordered_map<int, OptixGeometryData::TextureBundle> textureMap;  //  int-based map
        std::vector<GpuMaterial> materialList;

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
                data.tangents.push_back(make_float3(tri->tangent0.x, tri->tangent0.y, tri->tangent0.z));
                data.tangents.push_back(make_float3(tri->tangent1.x, tri->tangent1.y, tri->tangent1.z));
                data.tangents.push_back(make_float3(tri->tangent2.x, tri->tangent2.y, tri->tangent2.z));

                unsigned int idx = static_cast<unsigned int>(data.vertices.size()) - 1;
                if (i == 0) tri_indices.x = idx;
                else if (i == 1) tri_indices.y = idx;
                else            tri_indices.z = idx;
            }

            data.indices.push_back(tri_indices);

            // GpuMaterial index eşleştirmesi
            GpuMaterial gpuMat = tri->gpuMaterial;
            int matIndex = -1;

            auto it = materialMap.find(gpuMat);
            if (it != materialMap.end()) {
                matIndex = it->second;
            }
            else {
                matIndex = static_cast<int>(materialList.size());
                materialMap[gpuMat] = matIndex;
                materialList.push_back(gpuMat);
            }

            //  INT tabanlı kayıt → garantili çalışır
            textureMap[matIndex] = tri->textureBundle;
            data.material_indices.push_back(matIndex);

        }

        for (size_t i = 0; i < materialList.size(); ++i) {
            if (textureMap.count((int)i)) {
                data.textures.push_back(textureMap[(int)i]);
              
            }
            else {
                OptixGeometryData::TextureBundle dummy = {};
                data.textures.push_back(dummy);
              
            }
        }

        data.materials = std::move(materialList);
        return data;
    }


  
// TriangleData ve GpuMaterial kullanan tam OptixGeometryData çıkarımı
private:
    void processBones(const aiScene* scene,
        std::vector<std::shared_ptr<Triangle>>& triangles,
        std::unordered_map<std::string, unsigned int>& boneNameToIndex,
        std::unordered_map<std::string, aiNode*>& boneNameToNode)
    {
        for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
        {
            aiMesh* mesh = scene->mMeshes[meshIndex];

            if (mesh->HasBones())
            {
                std::cout << "Mesh " << meshIndex << " has bones: " << mesh->mNumBones << std::endl;

                for (unsigned int boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex)
                {
                    aiBone* bone = mesh->mBones[boneIndex];
                    std::string boneName = bone->mName.C_Str();
                    aiMatrix4x4 offsetMatrix = bone->mOffsetMatrix;

                    // ---  Bone index ve node eşleşmelerini kaydet ---
                    boneNameToIndex[boneName] = boneIndex;

                    // nodeMap zaten loadModelToTriangles içinde dolmuştu
                    if (nodeMap.find(boneName) != nodeMap.end())
                        boneNameToNode[boneName] = nodeMap[boneName];
                    else
                        boneNameToNode[boneName] = nullptr;  // Bulunamazsa NULL

                    std::cout << "Bone: " << boneName
                        << ", Affected Vertices: " << bone->mNumWeights << std::endl;

                    for (unsigned int w = 0; w < bone->mNumWeights; ++w)
                    {
                        unsigned int vertexId = bone->mWeights[w].mVertexId;
                        float weight = bone->mWeights[w].mWeight;

                        unsigned int triIndex = vertexId / 3;
                        unsigned int vertInTri = vertexId % 3;

                        if (triIndex >= triangles.size()) continue;

                        auto& tri = triangles[triIndex];

                        // Eğer vertexBoneWeights yoksa, oluştur
                        if (tri->vertexBoneWeights.size() != 3)
                            tri->vertexBoneWeights = std::vector<std::vector<std::pair<int, float>>>(3);

                        tri->vertexBoneWeights[vertInTri].push_back({ boneIndex, weight });

                        if (tri->originalVertexPositions.size() != 3)
                        {
                            tri->originalVertexPositions = { tri->original_v0, tri->original_v1, tri->original_v2 };
                        }

                        std::cout << " -> Triangle " << triIndex
                            << " Vertex " << vertInTri
                            << " Bone " << boneIndex
                            << " Weight " << weight << std::endl;
                    }
                }
            }
        }
    }


    std::unordered_map<std::string, aiNode*> nodeMap;
   static Vec3 transformPosition(const aiMatrix4x4& matrix, const aiVector3D& position) {
       aiVector3D transformed = matrix * position;
       return Vec3(transformed.x, transformed.y, transformed.z);
   }

   static Vec3 transformDirection(const aiMatrix4x4& matrix, const aiVector3D& direction) {
       aiMatrix3x3 rotation(matrix);
       aiVector3D transformed = rotation * direction;
       return Vec3(transformed.x, transformed.y, transformed.z);
   }

    static std::vector<TextureInfo> textureInfos;
    static std::vector<std::shared_ptr<Light>> lights;
    static std::unordered_map<std::string, std::shared_ptr<Texture>> textureCache;
   // std::vector<std::shared_ptr<Camera>> cameras;
    static void processNodeToTriangles(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Triangle>>& triangles, OptixGeometryData* geometry_data = nullptr) {
        aiMatrix4x4 globalTransform = getGlobalTransform(node);

        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

            // 🔧 HitGroupData tanımla
            HitGroupData hit_data = {};

            //  Material ve GPU texture'ları yükle
            auto convertedMaterial = processMaterial(material, scene, &hit_data, geometry_data);

            // 🎯 Mesh'i üçgenlere dönüştür
            processTriangles(mesh, globalTransform, node->mName.C_Str(), convertedMaterial, triangles);

            // 🧵 Son üçgene GPU texture bundle'ı aktar
            if (!triangles.empty()) {
                auto& tri = triangles.back();
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

            // 🔗 geometry_data'ya textureBundle kaydet (OptiX için)
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

        // 🔁 Çocuk node'ları işle
        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToTriangles(node->mChildren[i], scene, triangles, geometry_data);
        }
    }
    static void processTriangles(
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
            aiFace face = mesh->mFaces[i];
            if (face.mNumIndices != 3) continue; // Sadece üçgen yüzleri işleyin

            std::vector<Vec3> vertices;
            std::vector<Vec3> normals;
            std::vector<Vec2> texCoords;
            std::vector<Vec3> tangents;
            std::vector<Vec3> bitangents;
            bool hasTangents = mesh->HasTangentsAndBitangents();

            for (unsigned int j = 0; j < 3; j++) {
                unsigned int index = face.mIndices[j];

                // Vertex işleme
                aiVector3D vertex = mesh->mVertices[index];
                aiVector3D transformedVertex = transform * vertex;
                vertices.emplace_back(transformedVertex.x, transformedVertex.y, transformedVertex.z);

                // Normal işleme
                if (mesh->HasNormals()) {
                    aiVector3D normal = mesh->mNormals[index];
                    aiVector3D transformedNormal = normalTransform * normal;
                    transformedNormal.Normalize();
                    normals.emplace_back(transformedNormal.x, transformedNormal.y, transformedNormal.z);
                }

                // UV koordinatları işleme
                if (mesh->HasTextureCoords(0)) {
                    aiVector3D texCoord = mesh->mTextureCoords[0][index];
                    float u = texCoord.x;
                    float v = texCoord.y;
                    texCoords.emplace_back(u, v);
                }
                else {
                    texCoords.emplace_back(0.0f, 0.0f);
                }

                // Tangent ve bitangent işleme
                if (hasTangents) {
                    aiVector3D tangent = mesh->mTangents[index];
                    aiVector3D bitangent = mesh->mBitangents[index];

                    aiMatrix3x3 normalMatrix(transform);
                    normalMatrix.Inverse();
                    normalMatrix.Transpose();

                    aiVector3D transformedTangent = normalMatrix * tangent;
                    aiVector3D transformedBitangent = normalMatrix * bitangent;

                    transformedTangent.Normalize();
                    transformedBitangent.Normalize();

                    tangents.emplace_back(transformedTangent.x, transformedTangent.y, transformedTangent.z);
                    bitangents.emplace_back(transformedBitangent.x, transformedBitangent.y, transformedBitangent.z);
                }
            }

            auto triangle = std::make_shared<Triangle>(
                vertices[0], vertices[1], vertices[2],
                normals[0], normals[1], normals[2],
                texCoords[0], texCoords[1], texCoords[2],
                tangents.empty() ? Vec3() : tangents[0],
                tangents.empty() ? Vec3() : tangents[1],
                tangents.empty() ? Vec3() : tangents[2],
                bitangents.empty() ? Vec3() : bitangents[0],
                bitangents.empty() ? Vec3() : bitangents[1],
                bitangents.empty() ? Vec3() : bitangents[2],
                hasTangents,
                material,
                mesh->mMaterialIndex
            );
            triangle->mat_ptr = material;
            triangle->gpuMaterial = *material->gpuMaterial; // struct kopyalanıyor

            triangle->setNodeName(nodeName);
            triangles.push_back(triangle);

        }
    }

     void processCameras(const aiScene* scene) {
        try {
            if (!scene || !scene->HasCameras()) {
                std::cout << "Sahne kameraları içermiyor..." << std::endl;
                return;
            }

            for (unsigned int i = 0; i < scene->mNumCameras; i++) {
                aiCamera* aiCam = scene->mCameras[i];
                if (!aiCam) continue;
                cameraNodeNames.insert(aiCam->mName.C_Str());
                aiNode* camNode = scene->mRootNode->FindNode(aiCam->mName.C_Str());
                if (!camNode) {
                    std::cerr << "Kamera düğümü bulunamadı: " << aiCam->mName.C_Str() << std::endl;
                    continue;
                }

                // **Global transform hesapla**
                aiMatrix4x4 globalTransform = getGlobalTransform(camNode);

                // **Dönüşüm matrisini parçala**
                aiVector3D scaling, position;
                aiQuaternion rotation;
                globalTransform.Decompose(scaling, rotation, position);

                // **Dönüşleri uygula**
                Vec3 lookfrom(position.x, position.y, position.z);
                Vec3 forward = rotateVector(Vec3(0, 0, -1), rotation);
                Vec3 lookat = lookfrom + forward;
                Vec3 vup = rotateVector(Vec3(0, 1, 0), rotation);

                // **FOV dönüşümü**
                double vfov = aiCam->mHorizontalFOV * 180.0 / M_PI;
                double aspect = aiCam->mAspect;
                vfov = 2.0 * atan(tan(vfov * 0.5 * M_PI / 180.0) / aspect) * 180.0 / M_PI;

                auto camera = std::make_shared<Camera>(
                    lookfrom, lookat, vup, vfov, aspect, aperture, focusdistance, 5
                );
                camera->nodeName = std::string(aiCam->mName.C_Str());  // 🔥 Bu şart!

                cameras.push_back(camera);

                // **Debug Çıkışı**
                std::cout << "Kamera " << i << " yüklendi:" << std::endl;
                std::cout << "Konum: " << lookfrom.toString() << std::endl;
                std::cout << "Hedef: " << lookat.toString() << std::endl;
                std::cout << "Up vektörü: " << vup.toString() << std::endl;
                std::cout << "FOV: " << vfov << "°" << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Kamera işleme hatası: " << e.what() << std::endl;
        }
    }
  
     void processLights(const aiScene* scene) {
        try {
            if (!scene || !scene->HasLights()) {
                std::cout << "Sahnede ışık bulunmuyor, işlem devam ediyor..." << std::endl;
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

                // --- Renk ve yoğunluk ---
                aiColor3D color = aiLgt->mColorDiffuse;
                Vec3 intensity = Vec3(color.r, color.g, color.b);

                // --- Ortak pointer ---
                std::shared_ptr<Light> light = nullptr;
               
                // --- Türüne göre ışık oluştur ---
                if (aiLgt->mType == aiLightSource_DIRECTIONAL) {
                    light = std::make_shared<DirectionalLight>(direction, intensity / 1000, 100.0);
                    light->position = position;
                   
                }
                else if (aiLgt->mType == aiLightSource_POINT) {
                    light = std::make_shared<PointLight>(position, intensity / 1000, 0.1f);
                    
                }
                else if (aiLgt->mType == aiLightSource_SPOT) {
                    float angle_degrees = aiLgt->mAngleInnerCone * (180.0f / M_PI);
                    float angle_radians = aiLgt->mAngleOuterCone;
                    float radius = 10.0f;

                    light = std::make_shared<SpotLight>(position, direction, intensity, angle_degrees, radius);
                  
                }
                else if (aiLgt->mType == aiLightSource_AREA) {
                    Vec3 forward = transformDirection(globalTransform, aiLgt->mDirection).normalize();
                    Vec3 up = transformDirection(globalTransform, aiLgt->mUp).normalize();

                    Vec3 u = Vec3::cross(forward, up).normalize();
                    Vec3 v = Vec3::cross(u, forward).normalize();

                    double width = aiLgt->mSize.x;
                    double height = aiLgt->mSize.y;

                    light = std::make_shared<AreaLight>(position, u, v, width, height, intensity);
                   
                }

                // --- Eğer ışık oluşturulduysa ayarları yap ---
                if (light) {
                    // Node adını ve başlangıç yönünü kaydet (animasyon eşleşmesi için)
                    light->nodeName = std::string(aiLgt->mName.C_Str()); // ✅ DOĞRU!
                    light->initialDirection = direction; // Yönlü ışıklar için önemli

                    // Listeye ekle
                    lights.push_back(light);

                    // DEBUG
                    std::cout << "Işık eklendi: " << aiLgt->mName.C_Str()
                        << " Tür: " << aiLgt->mType
                        << " Pozisyon: (" << position.x << ", " << position.y << ", " << position.z << ")"
                        << " Şiddet: " << intensity << std::endl;
                }
                else {
                    std::cerr << "Desteklenmeyen ışık türü: " << aiLgt->mType << std::endl;
                }
            }

        }
        catch (const std::exception& e) {
            std::cerr << "Işık işleme sırasında hata oluştu: " << e.what() << std::endl;
        }
    }



    static std::shared_ptr<Material> processMaterial(
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
       float emissiveStrength = std::max({ emissiveColor.r, emissiveColor.g, emissiveColor.b })*1.0;
	   
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
       float transmission = 0.0f;
       if (AI_SUCCESS == aiGetMaterialFloat(aiMat, AI_MATKEY_TRANSMISSION_FACTOR, &transmission)) {
           material->setTransmission(transmission,1.1);
          
       }

       else {
		   material->setTransmission(0.0f, 1.5);
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

       /*if (materialNameStr.find("glass") != std::string::npos || materialNameStr.find("dielectric") != std::string::npos) {
           Vec3 glassColor = Vec3(color.r,color.g,color.b);       
           float scratch_density=10.0f;
           auto dielectricMaterial = std::make_shared<Dielectric>(
            1.2f, glassColor,2.0, 1.0,roughness, scratch_density);//cam ayarları kırılma indisi,renk,kaostik,cam rengi,bulanıklık, çizikler
           // Transfer properties
           dielectricMaterial->albedoProperty = material->albedoProperty;
           dielectricMaterial->roughnessProperty = material->roughnessProperty;
           dielectricMaterial->metallicProperty = material->metallicProperty;
           dielectricMaterial->normalProperty = material->normalProperty;
           dielectricMaterial->opacityProperty = material->opacityProperty;
           return dielectricMaterial;
       }*/

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

           // Doğrudan renk verileriyle doldur
           gpu->albedo = make_float3(color.r, color.g, color.b);
           gpu->roughness = roughness;
           gpu->metallic = metalicFactor;
           gpu->emission = make_float3(emissiveColor.r, emissiveColor.g, emissiveColor.b);
           gpu->transmission = transmission;
           gpu->ior = 1.5f;
		   gpu->opacity = opacity;

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
               geometry_data->textures.push_back(tex_bundle); // 🚀
           }

           // GPU tarafına geçirilecek veriyi material içinde tut
           if (materialNameStr.find("pastel") != std::string::npos ) {
               material->artistic_albedo_response = 0.3f;
           }

           gpu->artistic_albedo_response = material->artistic_albedo_response;
           material->gpuMaterial = gpu;


       return material;
   }
   static std::shared_ptr<Texture> loadTexture(const std::string& filepath, TextureType type = TextureType::Unknown, const std::string& opacityMapPath = "") {
       std::string fullPath = baseDirectory + filepath;
       auto texture = std::make_shared<Texture>(fullPath, type); // 👈 yeni parametre

       if (!opacityMapPath.empty()) {
           std::string fullOpacityMapPath = baseDirectory + opacityMapPath;
           texture->loadOpacityMap(fullOpacityMapPath);
       }

       return texture;
   }

   static std::string sanitizeTextureName(const aiString& str) {
       std::string textureName = str.C_Str();
       size_t pos = 0;
       while ((pos = textureName.find("%20", pos)) != std::string::npos) {
           textureName.replace(pos, 3, " ");
           pos += 1;
       }
       return textureName;
   }
   static  void processMaterialTexture(aiMaterial* aiMat, aiTextureType type, MaterialProperty& property) {
       aiString str;
       if (aiMat->GetTextureCount(type) > 0) {
           aiMat->GetTexture(type, 0, &str);
           std::string textureName = sanitizeTextureName(str);
           property.texture = loadTexture(textureName.c_str());
       }
   }
   static TextureType convertToTextureType(aiTextureType type) {
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

   static std::shared_ptr<Texture> loadTextureWithCache(const std::string& textureName, TextureType type = TextureType::Unknown) {
       auto it = textureCache.find(textureName);
       if (it != textureCache.end()) {
           return it->second; // 📦 Zaten yüklü -> direkt dön
       }

       // İlk kez yükleniyor
       auto texture = std::make_shared<Texture>(baseDirectory + textureName, type);
       textureCache[textureName] = texture;

       texture->upload_to_gpu(); // 📦 İlk ve tek GPU yükleme

       return texture;
   }




};
