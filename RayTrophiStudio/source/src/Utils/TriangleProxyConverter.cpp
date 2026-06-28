#include "TriangleProxyConverter.h"
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace {
    struct QuantizedVertexKey {
        int x, y, z;
        int u, v;
        bool operator==(const QuantizedVertexKey& other) const {
            return x == other.x && y == other.y && z == other.z && u == other.u && v == other.v;
        }
    };
    struct QuantizedVertexKeyHasher {
        std::size_t operator()(const QuantizedVertexKey& key) const {
            std::size_t hx = std::hash<int>{}(key.x);
            std::size_t hy = std::hash<int>{}(key.y);
            std::size_t hz = std::hash<int>{}(key.z);
            std::size_t hu = std::hash<int>{}(key.u);
            std::size_t hv = std::hash<int>{}(key.v);
            return hx ^ (hy << 1) ^ (hz << 2) ^ (hu << 3) ^ (hv << 4);
        }
    };
    QuantizedVertexKey quantizeVertex(const Vec3& p, const Vec2& uv, float epsilon) {
        const float safeEpsilon = std::max(epsilon, 1e-6f);
        return QuantizedVertexKey{
            static_cast<int>(std::round(p.x / safeEpsilon)),
            static_cast<int>(std::round(p.y / safeEpsilon)),
            static_cast<int>(std::round(p.z / safeEpsilon)),
            static_cast<int>(std::round(uv.u / 1e-4f)), // fixed tolerance for UVs
            static_cast<int>(std::round(uv.v / 1e-4f))
        };
    }
}

std::shared_ptr<TriangleMesh> TriangleProxyConverter::convertToProxyMesh(const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (triangles.empty()) return nullptr;

    auto mesh = std::make_shared<TriangleMesh>();
    
    // Adaptive weld epsilon calculation based on shortest edge
    float minEdgeSq = 1e8f;
    for (const auto& tri : triangles) {
        if (!tri) continue;
        const Vec3 a = tri->getOriginalVertexPosition(0);
        const Vec3 b = tri->getOriginalVertexPosition(1);
        const Vec3 c = tri->getOriginalVertexPosition(2);
        const float e0 = (b - a).length_squared();
        const float e1 = (c - b).length_squared();
        const float e2 = (a - c).length_squared();
        if (e0 > 1e-20f) minEdgeSq = std::min(minEdgeSq, e0);
        if (e1 > 1e-20f) minEdgeSq = std::min(minEdgeSq, e1);
        if (e2 > 1e-20f) minEdgeSq = std::min(minEdgeSq, e2);
    }
    const float weldEdge = (minEdgeSq < 1e8f) ? std::sqrt(minEdgeSq) : 1e-4f;
    const float weldEps = std::clamp(weldEdge * 0.25f, 1e-6f, 1e-4f);

    std::unordered_map<QuantizedVertexKey, uint32_t, QuantizedVertexKeyHasher> vlookup;
    
    for (const auto& tri : triangles) {
        if (tri) {
            mesh->transform = tri->getTransformHandle();
            mesh->nodeName = tri->getNodeName();
            break;
        }
    }

    // Pass 1: Weld vertices and collect temporary lists
    std::vector<Vec3> tempPositions;
    std::vector<Vec3> tempOriginalPositions;
    std::vector<Vec3> tempNormals;
    std::vector<Vec3> tempOriginalNormals;
    std::vector<Vec2> tempUvs;
    
    std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>> meshIndices;
    meshIndices.reserve(triangles.size() * 3);

    for (const auto& tri : triangles) {
        if (!tri) continue;
        
        auto [uv0, uv1, uv2] = tri->getUVCoordinates();
        Vec2 triUVs[3] = { uv0, uv1, uv2 };
        
        uint32_t current_indices[3];
        for (int i = 0; i < 3; ++i) {
            Vec3 pos = tri->getVertexPosition(i);
            Vec3 orig_pos = tri->getOriginalVertexPosition(i);
            Vec3 norm = tri->getVertexNormal(i);
            Vec3 orig_norm = tri->getOriginalVertexNormal(i);
            Vec2 uv = triUVs[i];
            
            QuantizedVertexKey key = quantizeVertex(orig_pos, uv, weldEps);
            auto it = vlookup.find(key);
            uint32_t vid;
            if (it != vlookup.end()) {
                vid = it->second;
            } else {
                vid = static_cast<uint32_t>(tempPositions.size());
                vlookup[key] = vid;
                tempPositions.push_back(pos);
                tempOriginalPositions.push_back(orig_pos);
                tempNormals.push_back(norm);
                tempOriginalNormals.push_back(orig_norm);
                tempUvs.push_back(uv);
            }
            
            current_indices[i] = vid;
        }
        
        meshIndices.push_back(current_indices[0]);
        meshIndices.push_back(current_indices[1]);
        meshIndices.push_back(current_indices[2]);
    }

    // Pass 2: Allocate and copy into DNA::GeometryDetail SoA arrays
    size_t vCount = tempPositions.size();
    mesh->geometry->resize_vertices(vCount);
    mesh->geometry->add_attribute<Vec3>("P");
    mesh->geometry->add_attribute<Vec3>("N");
    mesh->geometry->add_attribute<Vec3>("P_orig");
    mesh->geometry->add_attribute<Vec3>("N_orig");
    mesh->geometry->add_attribute<Vec2>("uv");
    mesh->geometry->add_attribute<uint16_t>("materialID");

    Vec3* positions = mesh->geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* normals = mesh->geometry->get_attribute_data_mut<Vec3>("N");
    Vec3* origPositions = mesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* origNormals = mesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec2* uvs = mesh->geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* matIDs = mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");

    if (positions) std::copy(tempPositions.begin(), tempPositions.end(), positions);
    if (normals) std::copy(tempNormals.begin(), tempNormals.end(), normals);
    if (origPositions) std::copy(tempOriginalPositions.begin(), tempOriginalPositions.end(), origPositions);
    if (origNormals) std::copy(tempOriginalNormals.begin(), tempOriginalNormals.end(), origNormals);
    if (uvs) std::copy(tempUvs.begin(), tempUvs.end(), uvs);

    // Copy indices to geometry
    mesh->geometry->indices = std::move(meshIndices);

    // Populate materialID per vertex (querying from input triangles)
    if (matIDs) {
        // Initialize with default
        std::fill(matIDs, matIDs + vCount, static_cast<uint16_t>(MaterialManager::INVALID_MATERIAL_ID));
        
        size_t triIdx = 0;
        for (const auto& tri : triangles) {
            if (!tri) continue;
            uint16_t mId = tri->getMaterialID();
            uint32_t i0 = mesh->geometry->indices[triIdx * 3 + 0];
            uint32_t i1 = mesh->geometry->indices[triIdx * 3 + 1];
            uint32_t i2 = mesh->geometry->indices[triIdx * 3 + 2];
            matIDs[i0] = mId;
            matIDs[i1] = mId;
            matIDs[i2] = mId;
            triIdx++;
        }
    }

    return mesh;
}

void TriangleProxyConverter::convertFromRawArrays(
    const std::vector<Vec3>& pos,
    const std::vector<Vec3>& nrm,
    const std::vector<uint32_t>& triIndices,
    const std::vector<Vec2>& triUVs,
    const std::vector<uint16_t>& matIDs,
    const std::vector<int>& faceIdxs,
    std::shared_ptr<Transform> transform,
    const std::string& nodeName,
    std::vector<std::shared_ptr<Triangle>>& outTriangles)
{
    if (matIDs.empty()) return;

    const size_t nTris = matIDs.size();
    outTriangles.resize(nTris);

    // 1. Direct population of the shared original geometry.
    // The input pos and nrm vectors represent already welded unique vertices of the subdivided mesh.
    // Re-welding them with a hash map was redundant, extremely slow, and memory-intensive.
    auto sharedGeom = std::make_shared<OriginalMeshGeometry>();
    sharedGeom->positions = pos;
    sharedGeom->normals.resize(nrm.size());

    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(nrm.size() >= 4096)
    for (int i = 0; i < (int)nrm.size(); ++i) {
        float len = nrm[i].length();
        if (len > 1e-6f) {
            sharedGeom->normals[i] = nrm[i] / len;
        } else {
            sharedGeom->normals[i] = Vec3(0.0f, 1.0f, 0.0f);
        }
    }

    // 2. Parallel materialization of Triangle objects using OpenMP
    Matrix4x4 finalTransform = Matrix4x4::identity();
    Matrix4x4 normalTransform = Matrix4x4::identity();
    if (transform) {
        finalTransform = transform->getFinal();
        normalTransform = transform->getNormalTransform();
    }

    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(nTris >= 2048)
    for (int t = 0; t < (int)nTris; ++t) {
        const uint32_t idx0 = triIndices[t * 3 + 0];
        const uint32_t idx1 = triIndices[t * 3 + 1];
        const uint32_t idx2 = triIndices[t * 3 + 2];

        auto tri = std::make_shared<Triangle>(
            pos[idx0], pos[idx1], pos[idx2],
            nrm[idx0], nrm[idx1], nrm[idx2],
            triUVs[t * 3 + 0], triUVs[t * 3 + 1], triUVs[t * 3 + 2],
            matIDs[t]
        );

        const int fIdx = (t < (int)faceIdxs.size()) ? faceIdxs[t] : -1;
        tri->setFaceIndex(fIdx);
        tri->setTransformHandle(transform);
        tri->setNodeName(nodeName);
        
        tri->setOriginalGeometry(sharedGeom);
        tri->setAssimpVertexIndices(idx0, idx1, idx2);
        
        // Transform and update bounding box using pre-calculated matrices
        tri->updateTransformedVerticesWith(finalTransform, normalTransform);

        outTriangles[t] = std::move(tri);
    }
}

namespace {
    // Weld key for the facade path: (quantized position, quantized uv, materialID). Splits
    // UV islands AND material boundaries so per-vertex SoA uv/materialID stay correct.
    struct FacadeWeldKey {
        int x, y, z, u, v;
        uint16_t m;
        bool operator==(const FacadeWeldKey& o) const {
            return x == o.x && y == o.y && z == o.z && u == o.u && v == o.v && m == o.m;
        }
    };
    struct FacadeWeldHasher {
        std::size_t operator()(const FacadeWeldKey& k) const {
            std::size_t h = std::hash<int>{}(k.x);
            h ^= std::hash<int>{}(k.y) << 1;
            h ^= std::hash<int>{}(k.z) << 2;
            h ^= std::hash<int>{}(k.u) << 3;
            h ^= std::hash<int>{}(k.v) << 4;
            h ^= std::hash<uint16_t>{}(k.m) << 5;
            return h;
        }
    };
}

void TriangleProxyConverter::convertFromRawArraysToMesh(
    const std::vector<Vec3>& pos,
    const std::vector<Vec3>& nrm,
    const std::vector<uint32_t>& triIndices,
    const std::vector<Vec2>& triUVs,
    const std::vector<uint16_t>& matIDs,
    const std::vector<int>& faceIdxs,
    std::shared_ptr<Transform> transform,
    const std::string& nodeName,
    std::vector<std::shared_ptr<Triangle>>& outTriangles)
{
    outTriangles.clear();
    const size_t nTris = matIDs.size();
    if (nTris == 0) return;

    auto mesh = std::make_shared<TriangleMesh>();
    mesh->transform = transform;
    mesh->nodeName = nodeName;

    constexpr float POS_Q = 100000.0f; // 0.01 mm grid (matches cage vertex precision)
    constexpr float UV_Q  = 10000.0f;  // 1e-4 UV tolerance

    std::unordered_map<FacadeWeldKey, uint32_t, FacadeWeldHasher> vlookup;
    vlookup.reserve(nTris * 3);

    // Unique welded vertices (bind/original pose; current pose is derived via transform below).
    std::vector<Vec3> uPos, uNrm;
    std::vector<Vec2> uUV;
    std::vector<uint16_t> uMat;
    uPos.reserve(nTris * 3); uNrm.reserve(nTris * 3); uUV.reserve(nTris * 3); uMat.reserve(nTris * 3);

    std::vector<uint32_t, DNA::AlignedAllocator<uint32_t, 32>> meshIndices;
    meshIndices.reserve(nTris * 3);

    for (size_t t = 0; t < nTris; ++t) {
        const uint16_t m = matIDs[t];
        for (int c = 0; c < 3; ++c) {
            const uint32_t srcIdx = triIndices[t * 3 + c];
            const Vec3& p  = pos[srcIdx];
            const Vec3  n  = (srcIdx < nrm.size()) ? nrm[srcIdx] : Vec3(0.0f, 1.0f, 0.0f);
            const Vec2& uv = triUVs[t * 3 + c];

            FacadeWeldKey key{
                static_cast<int>(std::lround(p.x * POS_Q)),
                static_cast<int>(std::lround(p.y * POS_Q)),
                static_cast<int>(std::lround(p.z * POS_Q)),
                static_cast<int>(std::lround(uv.u * UV_Q)),
                static_cast<int>(std::lround(uv.v * UV_Q)),
                m
            };
            auto it = vlookup.find(key);
            uint32_t vid;
            if (it != vlookup.end()) {
                vid = it->second;
            } else {
                vid = static_cast<uint32_t>(uPos.size());
                vlookup.emplace(key, vid);
                uPos.push_back(p);
                uNrm.push_back(n);
                uUV.push_back(uv);
                uMat.push_back(m);
            }
            meshIndices.push_back(vid);
        }
    }

    const size_t vCount = uPos.size();
    mesh->geometry->resize_vertices(vCount);
    mesh->geometry->add_attribute<Vec3>("P");
    mesh->geometry->add_attribute<Vec3>("N");
    mesh->geometry->add_attribute<Vec3>("P_orig");
    mesh->geometry->add_attribute<Vec3>("N_orig");
    mesh->geometry->add_attribute<Vec2>("uv");
    mesh->geometry->add_attribute<uint16_t>("materialID");

    Vec3*     P  = mesh->geometry->get_attribute_data_mut<Vec3>("P");
    Vec3*     N  = mesh->geometry->get_attribute_data_mut<Vec3>("N");
    Vec3*     Po = mesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3*     No = mesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec2*     UV = mesh->geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* M  = mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");

    Matrix4x4 finalTransform  = Matrix4x4::identity();
    Matrix4x4 normalTransform = Matrix4x4::identity();
    if (transform) {
        finalTransform  = transform->getFinal();
        normalTransform = transform->getNormalTransform();
    }

    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static) if(vCount >= 4096)
    for (int v = 0; v < (int)vCount; ++v) {
        Vec3 origN = uNrm[v];
        float nl = origN.length();
        origN = (nl > 1e-6f) ? (origN / nl) : Vec3(0.0f, 1.0f, 0.0f);

        if (Po) Po[v] = uPos[v];
        if (No) No[v] = origN;
        if (UV) UV[v] = uUV[v];
        if (M)  M[v]  = uMat[v];

        Vec3 wp = finalTransform.transform_point(uPos[v]);
        Vec3 wn = normalTransform.transform_vector(origN);
        float wl = wn.length();
        if (P) P[v] = wp;
        if (N) N[v] = (wl > 1e-6f) ? (wn / wl) : Vec3(0.0f, 1.0f, 0.0f);
    }

    mesh->geometry->indices = std::move(meshIndices);

    // Materialize lightweight facade triangles referencing the shared SoA mesh. With the
    // facade-slim Triangle (lazy standalone payload), each of these is ~16 B of live state
    // (parentMesh + faceIndex) instead of a fat ~200 B standalone object.
    outTriangles.resize(nTris);
    for (size_t t = 0; t < nTris; ++t) {
        auto tri = std::make_shared<Triangle>(mesh, static_cast<uint32_t>(t));
        const int fIdx = (t < faceIdxs.size()) ? faceIdxs[t] : -1;
        tri->setFaceIndex(fIdx);
        outTriangles[t] = std::move(tri);
    }
}
