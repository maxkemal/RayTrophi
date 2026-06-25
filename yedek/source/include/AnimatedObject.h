/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          AnimatedObject.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include "hittable.h"
#include "Ray.h"
#include "matrix4x4.h"
#include "vec3SIMD.h"
#include "quaternion.h"
#include "AssimpLoader.h" // AnimationData yapısının tanımlandığı header


class AnimatedObject : public Hittable {
public:
    struct Node {
        std::string name;
        Matrix4x4 transform;
        Matrix4x4 globalTransform;
        Node* parent;
        std::vector<Node*> children;
    };
    std::vector<std::shared_ptr<Hittable>> m_meshes;
    std::unordered_map<std::string, Node> m_nodeHierarchy;
    std::unordered_map<std::string, size_t> m_nodeMeshMap;
    std::vector<Matrix4x4> m_meshTransforms;
    template<typename T>
    int findKeyframeIndex(const std::vector<T>& keys, double animationTime) {
        for (size_t i = 0; i < keys.size() - 1; i++) {
            if (animationTime < keys[i + 1].mTime) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(keys.size() - 1);
    }
  
    AnimatedObject(const std::vector<std::shared_ptr<Hittable>>& meshes)
        : m_meshes(meshes) {
        m_meshTransforms.resize(meshes.size(), Matrix4x4()); // Varsayılan constructor zaten identity matrix oluşturuyor
    }

    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override {
        bool hit_anything = false;
        double closest_so_far = t_max;

        for (size_t i = 0; i < m_meshes.size(); ++i) {
            HitRecord temp_rec;
            if (m_meshes[i]->hit(r, t_min, closest_so_far, temp_rec, ignore_volumes)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
                rec.point = m_meshTransforms[i].transform_point(rec.point);
                rec.normal = m_meshTransforms[i].transform_vector(rec.normal).normalize();
            }
        }

        return hit_anything;
    }

    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override {
        if (m_meshes.empty()) return false;

        AABB temp_box;
        bool first_box = true;

        for (size_t i = 0; i < m_meshes.size(); ++i) {
            if (!m_meshes[i]->bounding_box(time0, time1, temp_box)) return false;
            output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
            first_box = false;
        }

        return true;
    }

    void updateTransform(const AnimationData& animation, float current_time) {
        // Animasyon süresini saniyeye çevir
        float animationDurationInSeconds = animation.duration / animation.ticksPerSecond;

        // Mevcut zamanı animasyon süresine göre normalize et
        float normalizedTimeInSeconds = std::fmod(current_time, animationDurationInSeconds);

        // Saniye cinsinden zamanı tick'lere çevir
        double normalizedTimeInTicks = normalizedTimeInSeconds * animation.ticksPerSecond;

        std::cout << "Animation Update:"
            << "\nCurrent time (s): " << current_time
            << "\nAnimation duration (s): " << animationDurationInSeconds
            << "\nNormalized time (s): " << normalizedTimeInSeconds
            << "\nNormalized time (ticks): " << normalizedTimeInTicks
            << "\nTicks per second: " << animation.ticksPerSecond
            << "\nDuration (ticks): " << animation.duration << std::endl;

        // Tüm node'lar için güncelleme yap
        for (const auto& [nodeName, nodeData] : m_nodeHierarchy) {
            Vec3 position = Vec3(0, 0, 0);
            Quaternion rotation = Quaternion(0, 0, 0, 1);
            Vec3 scaling = Vec3(1, 1, 1);

            // Debug için pozisyon değerlerini yazdır
            if (animation.positionKeys.count(nodeName) > 0) {
                const auto& keys = animation.positionKeys.at(nodeName);
                std::cout << "Node: " << nodeName << " has " << keys.size() << " position keys" << std::endl;
                std::cout << "First key time: " << keys.front().mTime << ", Last key time: " << keys.back().mTime << std::endl;

                position = interpolatePosition(keys, normalizedTimeInTicks);
                std::cout << "Interpolated position: " << position.x << ", " << position.y << ", " << position.z << std::endl;
            }

            if (animation.rotationKeys.count(nodeName) > 0) {
                rotation = interpolateRotation(animation.rotationKeys.at(nodeName), normalizedTimeInTicks);
            }

            if (animation.scalingKeys.count(nodeName) > 0) {
                scaling = interpolateScaling(animation.scalingKeys.at(nodeName), normalizedTimeInTicks);
            }

            Matrix4x4 translationMat = Matrix4x4::translation(position);
            Matrix4x4 rotationMat = quaternionToMatrix4x4(rotation);
            Matrix4x4 scalingMat = Matrix4x4::scaling(scaling);

            Matrix4x4 nodeTransform = translationMat * rotationMat * scalingMat;
            updateNodeTransform(nodeName, nodeTransform);
        }
    }
    size_t getMeshCount() const { return m_meshes.size(); }
    const Matrix4x4& getMeshTransform(size_t index) const { return m_meshTransforms[index]; }

    void setupNodeHierarchy(const std::unordered_map<std::string, Node>& nodeData) {
        m_nodeHierarchy = nodeData;
    }

    void setupMeshNodeMapping(const std::unordered_map<std::string, size_t>& meshNodeMapping) {
        m_nodeMeshMap = meshNodeMapping;
    }


    // Quaternion'u Matrix4x4'e çeviren yardımcı fonksiyon
    Matrix4x4 quaternionToMatrix4x4(const Quaternion& q) {
        // Bu fonksiyonu Quaternion sınıfınıza göre uyarlayın
        // Örnek bir implementasyon:
        float xx = q.x * q.x;
        float xy = q.x * q.y;
        float xz = q.x * q.z;
        float xw = q.x * q.w;
        float yy = q.y * q.y;
        float yz = q.y * q.z;
        float yw = q.y * q.w;
        float zz = q.z * q.z;
        float zw = q.z * q.w;

        Matrix4x4 result;
        result.m[0][0] = 1 - 2 * (yy + zz);
        result.m[0][1] = 2 * (xy - zw);
        result.m[0][2] = 2 * (xz + yw);
        result.m[1][0] = 2 * (xy + zw);
        result.m[1][1] = 1 - 2 * (xx + zz);
        result.m[1][2] = 2 * (yz - xw);
        result.m[2][0] = 2 * (xz - yw);
        result.m[2][1] = 2 * (yz + xw);
        result.m[2][2] = 1 - 2 * (xx + yy);

        return result;
    }
    void updateNodeTransform(const std::string& nodeName, const Matrix4x4& localTransform) {
        // Önce local transform'u kaydet
        m_nodeHierarchy[nodeName].transform = localTransform;

        // Global transform'u hesapla
        Matrix4x4 globalTransform = localTransform;
        Node* currentNode = &m_nodeHierarchy[nodeName];
        if (currentNode->parent != nullptr) {
            // Parent'ın global transform'unu kullan
            globalTransform = currentNode->parent->globalTransform * globalTransform;
        }

        // Global transform'u kaydet
        m_nodeHierarchy[nodeName].globalTransform = globalTransform;

        // Mesh transform'unu güncelle
        if (m_nodeMeshMap.count(nodeName) > 0) {
            size_t meshIndex = m_nodeMeshMap[nodeName];
            m_meshTransforms[meshIndex] = globalTransform;
        }
    }

   static  Vec3 interpolatePosition(const std::vector<aiVectorKey>& positionKeys, double time) {
        if (positionKeys.size() == 1) {
            return Vec3(positionKeys[0].mValue.x, positionKeys[0].mValue.y, positionKeys[0].mValue.z);
        }

        for (size_t i = 0; i < positionKeys.size() - 1; ++i) {
            if (time < positionKeys[i + 1].mTime) {
                float factor = static_cast<float>((time - positionKeys[i].mTime) / (positionKeys[i + 1].mTime - positionKeys[i].mTime));

                float startX = positionKeys[i].mValue.x;
                float startY = positionKeys[i].mValue.y;
                float startZ = positionKeys[i].mValue.z;

                float endX = positionKeys[i + 1].mValue.x;
                float endY = positionKeys[i + 1].mValue.y;
                float endZ = positionKeys[i + 1].mValue.z;

                float interpolatedX = startX + (endX - startX) * factor;
                float interpolatedY = startY + (endY - startY) * factor;
                float interpolatedZ = startZ + (endZ - startZ) * factor;

                return Vec3(interpolatedX, interpolatedY, interpolatedZ);
            }
        }

        return Vec3(positionKeys.back().mValue.x, positionKeys.back().mValue.y, positionKeys.back().mValue.z);
    }


   static  Quaternion interpolateRotation(const std::vector<aiQuatKey>& rotationKeys, double time) {
        if (rotationKeys.size() == 1)
            return Quaternion(rotationKeys[0].mValue.x, rotationKeys[0].mValue.y, rotationKeys[0].mValue.z, rotationKeys[0].mValue.w);

        for (size_t i = 0; i < rotationKeys.size() - 1; ++i) {
            if (time < rotationKeys[i + 1].mTime) {
                float factor = (time - rotationKeys[i].mTime) / (rotationKeys[i + 1].mTime - rotationKeys[i].mTime);
                Quaternion start(rotationKeys[i].mValue.x, rotationKeys[i].mValue.y, rotationKeys[i].mValue.z, rotationKeys[i].mValue.w);
                Quaternion end(rotationKeys[i + 1].mValue.x, rotationKeys[i + 1].mValue.y, rotationKeys[i + 1].mValue.z, rotationKeys[i + 1].mValue.w);
                return Quaternion::slerp(start, end, factor);
            }
        }
        return Quaternion(rotationKeys.back().mValue.x, rotationKeys.back().mValue.y, rotationKeys.back().mValue.z, rotationKeys.back().mValue.w);
    }

   static Vec3 interpolateScaling(const std::vector<aiVectorKey>& scalingKeys, double time) {
        if (scalingKeys.size() == 1) {
            return Vec3(scalingKeys[0].mValue.x, scalingKeys[0].mValue.y, scalingKeys[0].mValue.z);
        }

        for (size_t i = 0; i < scalingKeys.size() - 1; ++i) {
            if (time < scalingKeys[i + 1].mTime) {
                float factor = static_cast<float>((time - scalingKeys[i].mTime) / (scalingKeys[i + 1].mTime - scalingKeys[i].mTime));

                float startX = scalingKeys[i].mValue.x;
                float startY = scalingKeys[i].mValue.y;
                float startZ = scalingKeys[i].mValue.z;

                float endX = scalingKeys[i + 1].mValue.x;
                float endY = scalingKeys[i + 1].mValue.y;
                float endZ = scalingKeys[i + 1].mValue.z;

                float interpolatedX = startX + (endX - startX) * factor;
                float interpolatedY = startY + (endY - startY) * factor;
                float interpolatedZ = startZ + (endZ - startZ) * factor;

                return Vec3(interpolatedX, interpolatedY, interpolatedZ);
            }
        }

        return Vec3(scalingKeys.back().mValue.x, scalingKeys.back().mValue.y, scalingKeys.back().mValue.z);
    }


};

