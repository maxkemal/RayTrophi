#ifndef TRIANGLE_PROXY_CONVERTER_H
#define TRIANGLE_PROXY_CONVERTER_H

#include <vector>
#include <memory>
#include "Triangle.h"
#include "TriangleMesh.h"

class TriangleProxyConverter {
public:
    /**
     * @brief Converts a vector of standalone Triangle objects into a flat TriangleMesh 
     *        and sets the original Triangles to act as proxies to the new mesh.
     * @param triangles The list of shared_ptr<Triangle> to convert
     * @return The resulting TriangleMesh holding the flat SoA data
     */
    static std::shared_ptr<TriangleMesh> convertToProxyMesh(const std::vector<std::shared_ptr<Triangle>>& triangles);

    /**
     * @brief Direct converter from subdivision plan output arrays to avoid intermediate legacy allocations
     * @param pos Evaluated cage positions
     * @param nrm Evaluated cage normals
     * @param triIndices Indices into pos/nrm per triangle (size = 3 * num_triangles)
     * @param triUVs UVs per face-vertex (size = 3 * num_triangles)
     * @param matIDs Material ID per triangle (size = num_triangles)
     * @param faceIdxs Original face index per triangle (size = num_triangles)
     * @param transform Transform to assign to the mesh
     * @param nodeName Name of the node
     * @param outTriangles The vector to fill with the lightweight proxy Triangle objects
     */
    static void convertFromRawArrays(
        const std::vector<Vec3>& pos,
        const std::vector<Vec3>& nrm,
        const std::vector<uint32_t>& triIndices,
        const std::vector<Vec2>& triUVs,
        const std::vector<uint16_t>& matIDs,
        const std::vector<int>& faceIdxs,
        std::shared_ptr<Transform> transform,
        const std::string& nodeName,
        std::vector<std::shared_ptr<Triangle>>& outTriangles);

    /**
     * @brief Faz 1 (B): same inputs as convertFromRawArrays, but emits LIGHTWEIGHT FACADE
     *        triangles over a single shared DNA::GeometryDetail (SoA) instead of fat
     *        standalone Triangles. Welds on (position, uv, materialID) so UV islands and
     *        material boundaries split into distinct SoA vertices (paint/texture sampling
     *        stays correct). The returned facades keep the mesh alive via parentMesh.
     *        Combined with the facade-slim Triangle (lazy standalone payload) this is the
     *        real RAM win for dense subdivide/CC output (the 12.5M / 25 GB case).
     */
    static void convertFromRawArraysToMesh(
        const std::vector<Vec3>& pos,
        const std::vector<Vec3>& nrm,
        const std::vector<uint32_t>& triIndices,
        const std::vector<Vec2>& triUVs,
        const std::vector<uint16_t>& matIDs,
        const std::vector<int>& faceIdxs,
        std::shared_ptr<Transform> transform,
        const std::string& nodeName,
        std::vector<std::shared_ptr<Triangle>>& outTriangles);
};

#endif // TRIANGLE_PROXY_CONVERTER_H
