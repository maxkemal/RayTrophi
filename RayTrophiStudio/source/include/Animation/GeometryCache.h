#pragma once

#include "Vec3.h"
#include "json.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

class TriangleMesh;

namespace Animation {

// Fixed-topology deformation cache. Topology, UVs, materials and all custom
// attributes remain on the canonical TriangleMesh; samples store only local
// vertex positions.
struct GeometryCacheSample {
    int frame = 0;
    std::vector<Vec3> positions;
};

struct GeometryCacheClip {
    std::string object_name;
    int start_frame = 0;
    int end_frame = 0;
    int frame_step = 1;
    uint64_t topology_hash = 0;
    uint64_t source_signature = 0;
    size_t vertex_count = 0;
    bool enabled = true;
    std::vector<GeometryCacheSample> samples;
};

struct GeometryCacheApplyResult {
    bool changed = false;
    bool topology_mismatch = false;
    std::string error;
};

uint64_t geometryTopologyHash(const TriangleMesh& mesh);
size_t geometryCacheMemoryBytes(const GeometryCacheClip& clip);
bool captureGeometryCacheSample(const TriangleMesh& mesh, int frame,
                                GeometryCacheSample& out, std::string* error = nullptr);
GeometryCacheApplyResult applyGeometryCacheFrame(const GeometryCacheClip& clip,
                                                  TriangleMesh& mesh, int frame);

void serializeGeometryCache(const GeometryCacheClip& clip, nlohmann::json& out,
                            std::ostream* binary);
bool deserializeGeometryCache(const nlohmann::json& in, GeometryCacheClip& out,
                              std::istream* binary, std::string* error = nullptr);
void serializeGeometryCaches(
    const std::unordered_map<std::string, GeometryCacheClip>& clips,
    nlohmann::json& out, std::ostream* binary);
void deserializeGeometryCaches(
    const nlohmann::json& in,
    std::unordered_map<std::string, GeometryCacheClip>& clips,
    std::istream* binary);

bool runGeometryCacheSelfTest(std::string* details = nullptr);

} // namespace Animation
