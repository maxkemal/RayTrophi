#include "Animation/GeometryCache.h"

#include "DNA/GeometryDetail.h"
#include "TriangleMesh.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

namespace Animation {
namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void hashBytes(uint64_t& hash, const void* bytes, size_t size) {
    const auto* data = static_cast<const unsigned char*>(bytes);
    for (size_t i = 0; i < size; ++i) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= kFnvPrime;
    }
}

bool finiteVec(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

Vec3 unitOrUp(const Vec3& value) {
    return value.length_squared() > 1.0e-12f
        ? value.normalize() : Vec3(0.0f, 1.0f, 0.0f);
}

void rebuildNormals(DNA::GeometryDetail& geometry) {
    const size_t count = geometry.get_vertex_count();
    Vec3* positions = geometry.get_attribute_data_mut<Vec3>("P");
    Vec3* normals = geometry.get_attribute_data_mut<Vec3>("N");
    Vec3* normalsOrig = geometry.get_attribute_data_mut<Vec3>("N_orig");
    if (!positions || (!normals && !normalsOrig)) return;
    if (normals) std::fill(normals, normals + count, Vec3(0.0f));
    if (normalsOrig) std::fill(normalsOrig, normalsOrig + count, Vec3(0.0f));
    for (size_t i = 0; i + 2 < geometry.indices.size(); i += 3) {
        const uint32_t a = geometry.indices[i];
        const uint32_t b = geometry.indices[i + 1];
        const uint32_t c = geometry.indices[i + 2];
        if (a >= count || b >= count || c >= count) continue;
        const Vec3 face = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
        if (normals) { normals[a] += face; normals[b] += face; normals[c] += face; }
        if (normalsOrig) {
            normalsOrig[a] += face; normalsOrig[b] += face; normalsOrig[c] += face;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const Vec3 source = normals ? normals[i] : normalsOrig[i];
        const Vec3 value = unitOrUp(source);
        if (normals) normals[i] = value;
        if (normalsOrig) normalsOrig[i] = value;
    }
}

} // namespace

uint64_t geometryTopologyHash(const TriangleMesh& mesh) {
    if (!mesh.geometry) return 0;
    uint64_t hash = kFnvOffset;
    const uint64_t vertexCount = static_cast<uint64_t>(mesh.geometry->get_vertex_count());
    const uint64_t indexCount = static_cast<uint64_t>(mesh.geometry->indices.size());
    hashBytes(hash, &vertexCount, sizeof(vertexCount));
    hashBytes(hash, &indexCount, sizeof(indexCount));
    if (!mesh.geometry->indices.empty()) {
        hashBytes(hash, mesh.geometry->indices.data(),
                  mesh.geometry->indices.size() * sizeof(uint32_t));
    }
    return hash;
}

size_t geometryCacheMemoryBytes(const GeometryCacheClip& clip) {
    size_t bytes = 0;
    for (const auto& sample : clip.samples) bytes += sample.positions.size() * sizeof(Vec3);
    return bytes;
}

bool captureGeometryCacheSample(const TriangleMesh& mesh, int frame,
                                GeometryCacheSample& out, std::string* error) {
    if (!mesh.geometry) {
        if (error) *error = "mesh has no flat geometry";
        return false;
    }
    const size_t count = mesh.geometry->get_vertex_count();
    const Vec3* positions = mesh.geometry->get_attribute_data<Vec3>("P");
    if (!positions) positions = mesh.geometry->get_attribute_data<Vec3>("P_orig");
    if (!positions || count == 0) {
        if (error) *error = "mesh has no canonical position attribute";
        return false;
    }
    out.frame = frame;
    out.positions.assign(positions, positions + count);
    if (!std::all_of(out.positions.begin(), out.positions.end(), finiteVec)) {
        if (error) *error = "mesh contains a non-finite vertex position";
        out.positions.clear();
        return false;
    }
    return true;
}

GeometryCacheApplyResult applyGeometryCacheFrame(const GeometryCacheClip& clip,
                                                  TriangleMesh& mesh, int frame) {
    GeometryCacheApplyResult result;
    if (!clip.enabled || clip.samples.empty()) return result;
    if (!mesh.geometry) { result.error = "target mesh has no geometry"; return result; }
    if (mesh.geometry->get_vertex_count() != clip.vertex_count ||
        geometryTopologyHash(mesh) != clip.topology_hash) {
        result.topology_mismatch = true;
        result.error = "geometry cache topology no longer matches target mesh";
        return result;
    }

    const GeometryCacheSample* left = &clip.samples.front();
    const GeometryCacheSample* right = left;
    if (frame >= clip.samples.back().frame) {
        left = right = &clip.samples.back();
    } else if (frame > clip.samples.front().frame) {
        const auto upper = std::upper_bound(
            clip.samples.begin(), clip.samples.end(), frame,
            [](int value, const GeometryCacheSample& sample) { return value < sample.frame; });
        right = &*upper;
        left = &*(upper - 1);
    }
    if (left->positions.size() != clip.vertex_count ||
        right->positions.size() != clip.vertex_count) {
        result.error = "geometry cache sample size is invalid";
        return result;
    }

    Vec3* positions = mesh.geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* positionsOrig = mesh.geometry->get_attribute_data_mut<Vec3>("P_orig");
    if (!positions && !positionsOrig) {
        result.error = "target mesh has no writable position attribute";
        return result;
    }
    const float denominator = static_cast<float>(right->frame - left->frame);
    const float t = denominator > 0.0f
        ? std::clamp((static_cast<float>(frame - left->frame) / denominator), 0.0f, 1.0f)
        : 0.0f;
    for (size_t i = 0; i < clip.vertex_count; ++i) {
        const Vec3 value = left->positions[i] * (1.0f - t) + right->positions[i] * t;
        if (positions) positions[i] = value;
        if (positionsOrig) positionsOrig[i] = value;
    }
    rebuildNormals(*mesh.geometry);
    result.changed = true;
    return result;
}

void serializeGeometryCache(const GeometryCacheClip& clip, nlohmann::json& out,
                            std::ostream* binary) {
    out = {{"object", clip.object_name}, {"start", clip.start_frame},
           {"end", clip.end_frame}, {"step", clip.frame_step},
           {"topology_hash", clip.topology_hash},
           {"source_signature", clip.source_signature},
           {"vertex_count", clip.vertex_count}, {"enabled", clip.enabled}};
    out["frames"] = nlohmann::json::array();
    for (const auto& sample : clip.samples) out["frames"].push_back(sample.frame);
    if (binary && !clip.samples.empty()) {
        const int64_t offset = static_cast<int64_t>(binary->tellp());
        for (const auto& sample : clip.samples) {
            binary->write(reinterpret_cast<const char*>(sample.positions.data()),
                          static_cast<std::streamsize>(sample.positions.size() * sizeof(Vec3)));
        }
        out["binary_offset"] = offset;
        out["binary_bytes"] = geometryCacheMemoryBytes(clip);
    } else {
        nlohmann::json samples = nlohmann::json::array();
        for (const auto& sample : clip.samples) {
            nlohmann::json positions = nlohmann::json::array();
            for (const Vec3& value : sample.positions)
                positions.push_back({value.x, value.y, value.z});
            samples.push_back(std::move(positions));
        }
        out["samples"] = std::move(samples);
    }
}

bool deserializeGeometryCache(const nlohmann::json& in, GeometryCacheClip& out,
                              std::istream* binary, std::string* error) {
    out = {};
    out.object_name = in.value("object", std::string());
    out.start_frame = in.value("start", 0);
    out.end_frame = in.value("end", out.start_frame);
    out.frame_step = std::max(1, in.value("step", 1));
    out.topology_hash = in.value("topology_hash", uint64_t(0));
    out.source_signature = in.value("source_signature", uint64_t(0));
    out.vertex_count = in.value("vertex_count", size_t(0));
    out.enabled = in.value("enabled", true);
    if (out.object_name.empty() || out.vertex_count == 0 || !in.contains("frames")) {
        if (error) *error = "geometry cache metadata is incomplete";
        return false;
    }
    const auto& frames = in["frames"];
    if (!frames.is_array() || frames.empty()) {
        if (error) *error = "geometry cache has no frame samples";
        return false;
    }
    out.samples.resize(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) out.samples[i].frame = frames[i].get<int>();

    if (binary && in.contains("binary_offset")) {
        const int64_t offset = in.value("binary_offset", int64_t(-1));
        if (offset < 0) { if (error) *error = "invalid geometry cache binary offset"; return false; }
        binary->clear();
        binary->seekg(offset);
        for (auto& sample : out.samples) {
            sample.positions.resize(out.vertex_count);
            const size_t bytes = out.vertex_count * sizeof(Vec3);
            binary->read(reinterpret_cast<char*>(sample.positions.data()),
                         static_cast<std::streamsize>(bytes));
            if (static_cast<size_t>(binary->gcount()) != bytes) {
                if (error) *error = "geometry cache binary payload is truncated";
                out = {};
                return false;
            }
        }
    } else if (in.contains("samples") && in["samples"].is_array() &&
               in["samples"].size() == out.samples.size()) {
        for (size_t sampleIndex = 0; sampleIndex < out.samples.size(); ++sampleIndex) {
            const auto& positions = in["samples"][sampleIndex];
            if (!positions.is_array() || positions.size() != out.vertex_count) {
                if (error) *error = "geometry cache JSON sample size is invalid";
                out = {};
                return false;
            }
            auto& target = out.samples[sampleIndex].positions;
            target.reserve(out.vertex_count);
            for (const auto& value : positions) {
                if (!value.is_array() || value.size() < 3) {
                    if (error) *error = "geometry cache JSON position is invalid";
                    out = {};
                    return false;
                }
                target.emplace_back(value[0].get<float>(), value[1].get<float>(),
                                    value[2].get<float>());
            }
        }
    } else {
        if (error) *error = "geometry cache has no readable payload";
        out = {};
        return false;
    }
    return true;
}

void serializeGeometryCaches(
    const std::unordered_map<std::string, GeometryCacheClip>& clips,
    nlohmann::json& out, std::ostream* binary) {
    out = nlohmann::json::object();
    std::vector<std::string> names;
    names.reserve(clips.size());
    for (const auto& [name, clip] : clips) { (void)clip; names.push_back(name); }
    std::sort(names.begin(), names.end());
    for (const std::string& name : names)
        serializeGeometryCache(clips.at(name), out[name], binary);
}

void deserializeGeometryCaches(
    const nlohmann::json& in,
    std::unordered_map<std::string, GeometryCacheClip>& clips,
    std::istream* binary) {
    clips.clear();
    if (!in.is_object()) return;
    for (auto it = in.begin(); it != in.end(); ++it) {
        GeometryCacheClip clip;
        std::string error;
        if (deserializeGeometryCache(it.value(), clip, binary, &error)) clips[it.key()] = std::move(clip);
    }
}

bool runGeometryCacheSelfTest(std::string* details) {
    TriangleMesh mesh;
    mesh.nodeName = "GeometryCacheSelfTest";
    mesh.geometry = std::make_shared<DNA::GeometryDetail>();
    for (const char* name : {"P", "P_orig", "N", "N_orig"})
        mesh.geometry->add_attribute<Vec3>(name);
    mesh.geometry->resize_vertices(3);
    mesh.geometry->indices = {0, 1, 2};
    Vec3* p = mesh.geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* po = mesh.geometry->get_attribute_data_mut<Vec3>("P_orig");
    p[0] = po[0] = Vec3(0, 0, 0); p[1] = po[1] = Vec3(1, 0, 0); p[2] = po[2] = Vec3(0, 1, 0);
    GeometryCacheClip clip;
    clip.object_name = mesh.nodeName;
    clip.vertex_count = 3;
    clip.topology_hash = geometryTopologyHash(mesh);
    GeometryCacheSample a, b;
    const bool ca = captureGeometryCacheSample(mesh, 0, a);
    p[0].z = po[0].z = 2.0f;
    const bool cb = captureGeometryCacheSample(mesh, 10, b);
    clip.samples = {a, b};
    const auto applied = applyGeometryCacheFrame(clip, mesh, 5);
    p = mesh.geometry->get_attribute_data_mut<Vec3>("P");
    const bool midpoint = applied.changed && std::abs(p[0].z - 1.0f) < 1.0e-5f;
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    nlohmann::json serialized;
    serializeGeometryCache(clip, serialized, &binary);
    GeometryCacheClip loaded;
    std::string roundTripError;
    const bool roundTrip = deserializeGeometryCache(
        serialized, loaded, &binary, &roundTripError) &&
        loaded.samples.size() == 2 && loaded.samples[1].positions[0].z == 2.0f;
    mesh.geometry->indices[2] = 1;
    const auto mismatch = applyGeometryCacheFrame(loaded, mesh, 5);
    const bool pass = ca && cb && midpoint && roundTrip && mismatch.topology_mismatch &&
        geometryCacheMemoryBytes(clip) == 6 * sizeof(Vec3);
    if (details) {
        std::ostringstream text;
        text << (pass ? "PASS" : "FAIL") << " midpoint_z=" << p[0].z
             << " bytes=" << geometryCacheMemoryBytes(clip)
             << " binary_roundtrip=" << (roundTrip ? "true" : "false")
             << " topology_guard=" << (mismatch.topology_mismatch ? "true" : "false");
        if (!roundTripError.empty()) text << " error=" << roundTripError;
        *details = text.str();
    }
    return pass;
}

} // namespace Animation
