#pragma once

#include <vector>
#include <memory>
#include <string>
#include <array>
#include "Vec2.h"
#include "Vec3.h"
#include "Matrix4x4.h"
#include "Material.h"
#include "Hittable.h"
#include "AABB.h"

struct Vertex {
    Vec3 position;
    Vec3 normal;
    Vec2 texcoord;
    Vec3 tangent;
    Vec3 bitangent;
    std::vector<std::pair<int, float>> boneWeights;
    Vec3 bindPosePosition;
    Vec3 bindPoseNormal;

    Vertex() = default;

    Vertex(const Vec3& pos, const Vec3& norm, const Vec2& uv)
        : position(pos), normal(norm), texcoord(uv),
        bindPosePosition(pos), bindPoseNormal(norm) {
    }
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<std::array<uint32_t, 3>> indices;

    std::shared_ptr<Material> material;
    std::string meshName;
    std::string nodeName;
    Matrix4x4 localTransform;
    unsigned int originalMeshIndex = 0;
    bool hasSkinning = false;
    int materialIndex = -1;

    std::vector<std::shared_ptr<Triangle>> toTriangles() const {
        std::vector<std::shared_ptr<Triangle>> tris;
        tris.reserve(indices.size());

        for (const auto& triIndices : indices) {
            uint32_t i0 = triIndices[0];
            uint32_t i1 = triIndices[1];
            uint32_t i2 = triIndices[2];

            const Vertex& v0 = vertices[i0];
            const Vertex& v1 = vertices[i1];
            const Vertex& v2 = vertices[i2];

            auto tri = std::make_shared<Triangle>(
                v0.position, v1.position, v2.position,
                v0.normal, v1.normal, v2.normal,
                v0.texcoord, v1.texcoord, v2.texcoord,
                material
            );

            tri->setOriginalVertexPosition(0, v0.bindPosePosition);
            tri->setOriginalVertexPosition(1, v1.bindPosePosition);
            tri->setOriginalVertexPosition(2, v2.bindPosePosition);

            tri->setOriginalVertexNormal(0, v0.bindPoseNormal);
            tri->setOriginalVertexNormal(1, v1.bindPoseNormal);
            tri->setOriginalVertexNormal(2, v2.bindPoseNormal);

            if (!v0.boneWeights.empty() || !v1.boneWeights.empty() || !v2.boneWeights.empty()) {
                tri->initializeSkinData();
                tri->setSkinBoneWeights(0, v0.boneWeights);
                tri->setSkinBoneWeights(1, v1.boneWeights);
                tri->setSkinBoneWeights(2, v2.boneWeights);
            }

            tri->setAssimpVertexIndices(i0, i1, i2);
            tri->setNodeName(nodeName);
            tri->setMaterial(material);

            tris.push_back(tri);
        }

        return tris;
    }
};
