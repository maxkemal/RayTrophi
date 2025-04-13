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

class AssimpLoader {
public:
    Assimp::Importer importer;
    const aiNode* getNodeByName(const std::string& name) const {
        auto it = nodeMap.find(name);
        return it != nodeMap.end() ? it->second : nullptr;
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
        processNodeToTriangles(scene->mRootNode, scene, triangles);
       
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
        return { triangles, animationDataList};
    }

    std::vector<std::shared_ptr<Light>> getLights() const {
        return lights;
    }

    const std::vector<std::shared_ptr<Camera>>& getCameras()  {
        return cameras;
    }

 
    static std::vector<AnimationData> loadAnimations(const aiScene* scene) {
        std::vector<AnimationData> animations;

        try {
            for (unsigned int i = 0; i < scene->mNumAnimations; i++) {
                const aiAnimation* animation = scene->mAnimations[i];
                AnimationData animData;

                animData.name = animation->mName.C_Str();
                animData.duration = animation->mDuration;
                animData.ticksPerSecond = animation->mTicksPerSecond;

                std::cout << "Loaded Animation: " << animData.name
                    << ", Duration: " << animData.duration
                    << ", Ticks Per Second: " << animData.ticksPerSecond << std::endl;

                for (unsigned int j = 0; j < animation->mNumChannels; j++) {
                    const aiNodeAnim* channel = animation->mChannels[j];
                    std::string nodeName = channel->mNodeName.C_Str();

                    std::cout << "Node: " << nodeName
                        << ", Position Keys: " << channel->mNumPositionKeys
                        << ", Rotation Keys: " << channel->mNumRotationKeys
                        << ", Scaling Keys: " << channel->mNumScalingKeys << std::endl;

                    // Position keys
                    for (unsigned int k = 0; k < channel->mNumPositionKeys; k++) {
                        animData.positionKeys[nodeName].push_back(channel->mPositionKeys[k]);
                    }

                    // Rotation keys
                    for (unsigned int k = 0; k < channel->mNumRotationKeys; k++) {
                        animData.rotationKeys[nodeName].push_back(channel->mRotationKeys[k]);
                    }

                    // Scaling keys
                    for (unsigned int k = 0; k < channel->mNumScalingKeys; k++) {
                        animData.scalingKeys[nodeName].push_back(channel->mScalingKeys[k]);
                    }
                }

                animations.push_back(animData);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing animations: " << e.what() << std::endl;
        }

        return animations;
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


private:
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
    static void processNodeToTriangles(aiNode* node, const aiScene* scene, std::vector<std::shared_ptr<Triangle>>& triangles) {
        aiMatrix4x4 globalTransform = getGlobalTransform(node);

        for (unsigned int i = 0; i < node->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
            auto convertedMaterial = processMaterial(material, scene);

            processTriangles(mesh, globalTransform, node->mName.C_Str(), convertedMaterial, triangles);
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++) {
            processNodeToTriangles(node->mChildren[i], scene, triangles);
        }
    }

    static void processCameras(const aiScene* scene) {
        try {
            if (!scene || !scene->HasCameras()) {
                std::cout << "Sahne kameraları içermiyor..." << std::endl;
                return;
            }

            for (unsigned int i = 0; i < scene->mNumCameras; i++) {
                aiCamera* aiCam = scene->mCameras[i];
                if (!aiCam) continue;

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
  
    static void processLights(const aiScene* scene) {
        try {
            if (!scene || !scene->HasLights()) {
                std::cout << "Sahnede ışık bulunmuyor, işlem devam ediyor..." << std::endl;
                return;
            }

            for (unsigned int i = 0; i < scene->mNumLights; i++) {
                aiLight* aiLgt = scene->mLights[i];

                // Işığın bağlı olduğu düğümü bul
                const aiNode* node = scene->mRootNode->FindNode(aiLgt->mName);
                aiMatrix4x4 globalTransform = node ? getGlobalTransform(node) : aiMatrix4x4();

                // Pozisyon ve yönü transform et
                Vec3 position = transformPosition(globalTransform, aiLgt->mPosition);
                Vec3 direction = transformDirection(globalTransform, aiLgt->mDirection);

                // Renk ve yoğunluk
                aiColor3D color = aiLgt->mColorDiffuse;
                Vec3 intensity = Vec3(color.r, color.g, color.b);

                // Işık türüne göre ekleme
                if (aiLgt->mType == aiLightSource_DIRECTIONAL) {
                    auto light = std::make_shared<DirectionalLight>(direction, intensity / 1000, 100.0);
                    lights.push_back(light);
                    std::cout << "direction" << direction << std::endl;
                }
                else if (aiLgt->mType == aiLightSource_POINT) {
                    auto light = std::make_shared<PointLight>(position, intensity / 1000, 0.1f);
                    lights.push_back(light);
                }
                if (aiLgt->mType == aiLightSource_SPOT) {
                    // Işığın pozisyonu ve yönü
                    Vec3 position = transformPosition(globalTransform, aiLgt->mPosition);
                    Vec3 direction = transformDirection(globalTransform, aiLgt->mDirection);


                    // Işığın renk yoğunluğu
                    aiColor3D color = aiLgt->mColorDiffuse;
                    Vec3 intensity(color.r, color.g, color.b);

                    // Işığın açıları ve yayılma yarıçapı
                    float angle_degrees = aiLgt->mAngleInnerCone * (180.0f / M_PI); // İçe doğru koni açısı
                    float angle_radians = aiLgt->mAngleOuterCone;
                    float radius = 10.0f; // Varsayılan olarak bir değer, ihtiyaca göre ayarlanabilir

                    // SpotLight nesnesi oluştur
                    auto light = std::make_shared<SpotLight>(position, direction, intensity, angle_degrees, radius);
                    lights.push_back(light);

                    std::cout << "Spot Light yüklendi: "
                        << "Pozisyon(" << position.x << ", " << position.y << ", " << position.z << "), "
                        << "Açı(" << angle_degrees << "), "
                        << "Yarıçap(" << radius << ")" << std::endl;
                }

                if (aiLgt->mType == aiLightSource_AREA) {
                    Vec3 position = transformPosition(globalTransform, aiLgt->mPosition);
                    Vec3 forward = transformDirection(globalTransform, aiLgt->mDirection).normalize();
                    Vec3 up = transformDirection(globalTransform, aiLgt->mUp).normalize();

                    // Dik eksenler üret
                    Vec3 u = Vec3::cross(forward, up).normalize();
                    Vec3 v = Vec3::cross(u, forward).normalize();

                    double width = aiLgt->mSize.x;
                    double height = aiLgt->mSize.y;

                    aiColor3D color = aiLgt->mColorDiffuse;
                    Vec3 intensity(color.r, color.g, color.b);

                    auto light = std::make_shared<AreaLight>(position, u, v, width, height, intensity);
                    light->setWidth(width);
                    light->setHeight(height);
                    lights.push_back(light);


                    std::cout << "Area Light yüklendi: "
                        << "Pozisyon(" << position.x << ", " << position.y << ", " << position.z << "), "
                        << "Boyut(" << width << " x " << height << ")" << std::endl;
                }
                std::cout << "Işık: " << aiLgt->mName.C_Str() << " eklendi. Tür: " << aiLgt->mType
                    << " Pozisyon: " << position.x << ", " << position.y << ", " << position.z << std::endl;
               
                std::cout << "Işık şiddeti : " << intensity << std::endl;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Işık işleme sırasında hata oluştu: " << e.what() << std::endl;
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
			triangle->setNodeName(nodeName);
            triangles.push_back(triangle);
          
        }
    }
   

   static std::shared_ptr<Material> processMaterial(aiMaterial* aiMat, const aiScene* scene) {
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
       float emissiveStrength = std::max({ emissiveColor.r, emissiveColor.g, emissiveColor.b })*2.0;
	   
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
     
       // Opacity (transparency)
       float opacity = 1.0f;
           material->opacityProperty = MaterialProperty(Vec3(opacity));     
       // Texture'ları yükle ve ata
           for (const auto& texInfo : textureInfos) {
               auto texture = loadTextureWithCache(texInfo.path, convertToTextureType(texInfo.type));
               switch (texInfo.type) {
               case aiTextureType_DIFFUSE:
                   material->albedoProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Albedo); // gamma ✔️
                   if (texture->has_alpha) {
                       material->opacityProperty.texture = texture;
                       material->opacityProperty.intensity = 1.0f;
                   }
                   break;
               case aiTextureType_SPECULAR:
                   material->specularProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Roughness);
                   break;
             
               case aiTextureType_NORMALS:
                   material->normalProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Normal);  
                  
                   break;
               case aiTextureType_EMISSIVE:
                   material->emissionProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Emission);
                   break;
               case aiTextureType_OPACITY:
                   material->opacityProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Opacity);
                   material->opacityProperty.intensity = 1.0f;
                   break;
               case aiTextureType_METALNESS:
                   material->metallicProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Metallic);
                   break;
               case aiTextureType_DIFFUSE_ROUGHNESS:
                   material->roughnessProperty.texture = loadTextureWithCache(texInfo.path, TextureType::Roughness); 
				  
                   break;
             
               }
           }

           if (materialNameStr.find("sss") != std::string::npos || materialNameStr.find("subsurface") != std::string::npos) {
              
               material->setSubsurfaceScattering((1.0f, 0.8f, 0.5f), (1.0f, 1.0f, 1.0f));
              // material->setTransmission(1.0f, 1.5f);
           }

       if (materialNameStr.find("glass") != std::string::npos || materialNameStr.find("dielectric") != std::string::npos) {
           Vec3 glassColor = Vec3(color.r,color.g,color.b);       
           float scratch_density=10.0f;
           auto dielectricMaterial = std::make_shared<Dielectric>(
            1.18f, glassColor,2.0, 1.0,roughness, scratch_density);//cam ayarları kırılma indisi,renk,kaostik,cam rengi,bulanıklık, çizikler
           // Transfer properties
           dielectricMaterial->albedoProperty = material->albedoProperty;
           dielectricMaterial->roughnessProperty = material->roughnessProperty;
           dielectricMaterial->metallicProperty = material->metallicProperty;
           dielectricMaterial->normalProperty = material->normalProperty;
           dielectricMaterial->opacityProperty = material->opacityProperty;
           return dielectricMaterial;
       }

       if (materialNameStr.find("volume") != std::string::npos || materialNameStr.find("volumetric") != std::string::npos) {
           Vec3 albedo= Vec3(0.6, 0.6, 1);
           float density = 0.5f, scattering_factor = 0.5f, absorption_probability = 0.3f;
           Vec3 emission = albedo;
           auto volumetric_material = std::make_shared<Volumetric>(albedo, density, absorption_probability, scattering_factor,  emission);
           //volumetric_material->albedoProperty = material->albedoProperty;
           return volumetric_material;
       }

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
           return it->second;
       }

       auto texture = std::make_shared<Texture>(baseDirectory + textureName, type);
       textureCache[textureName] = texture;

       return texture;
   }

};
