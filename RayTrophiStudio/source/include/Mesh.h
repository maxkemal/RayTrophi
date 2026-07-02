/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Mesh.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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

    std::shared_ptr<TriangleMesh> toTriangleMesh() const {
        auto mesh = std::make_shared<TriangleMesh>();
        mesh->nodeName = nodeName;
        
        size_t v_count = vertices.size();
        mesh->geometry->resize_vertices(v_count);

        mesh->geometry->add_attribute<Vec3>("P");
        mesh->geometry->add_attribute<Vec3>("N");
        mesh->geometry->add_attribute<Vec2>("uv");
        mesh->geometry->add_attribute<uint16_t>("materialID");

        Vec3* positions = mesh->geometry->get_attribute_data_mut<Vec3>("P");
        Vec3* normals = mesh->geometry->get_attribute_data_mut<Vec3>("N");
        Vec2* uvs = mesh->geometry->get_attribute_data_mut<Vec2>("uv");
        uint16_t* matIDs = mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");

        for (size_t i = 0; i < v_count; ++i) {
            if (positions) positions[i] = vertices[i].position;
            if (normals) normals[i] = vertices[i].normal;
            if (uvs) uvs[i] = vertices[i].texcoord;
            if (matIDs) matIDs[i] = static_cast<uint16_t>(materialIndex != -1 ? materialIndex : 0xFFFF);
        }

        // Copy indices
        size_t tri_count = indices.size();
        mesh->geometry->indices.resize(tri_count * 3);
        for (size_t i = 0; i < tri_count; ++i) {
            mesh->geometry->indices[i * 3 + 0] = indices[i][0];
            mesh->geometry->indices[i * 3 + 1] = indices[i][1];
            mesh->geometry->indices[i * 3 + 2] = indices[i][2];
        }

        return mesh;
    }
};

