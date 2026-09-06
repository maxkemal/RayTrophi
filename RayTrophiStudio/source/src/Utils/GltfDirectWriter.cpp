/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GltfDirectWriter.cpp
* Author:        Kemal Demirtas
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
* See GltfDirectWriter.h for why this replaces the Assimp export round trip.
*/
#include "GltfDirectWriter.h"

#include "SceneExporter.h"      // ExportSettings
#include "scene_data.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "HittableInstance.h"
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include "Material.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Texture.h"
#include "Transform.h"
#include "Matrix4x4.h"
#include "Camera.h"
#include "globals.h"
#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "json.hpp"
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::json;

// ★ The whole design rests on these three types being bit-identical to their
// glTF accessor layouts, because the writer streams the arrays to disk without
// converting them. If padding or a fourth component ever appears, every exported
// file silently becomes garbage - so fail the BUILD, not the file.
static_assert(sizeof(Vec3) == 12, "Vec3 must be 3 tightly-packed floats to stream as glTF VEC3/FLOAT");
static_assert(sizeof(Vec2) == 8,  "Vec2 must be 2 tightly-packed floats to stream as glTF VEC2/FLOAT");
static_assert(sizeof(uint32_t) == 4, "glTF SCALAR/UNSIGNED_INT indices are 4 bytes");

namespace rtgltf {
namespace {

// ---------------------------------------------------------------------------
// glTF constants
// ---------------------------------------------------------------------------
constexpr int kCompFloat  = 5126;
constexpr int kCompUShort = 5123;
constexpr int kCompUInt   = 5125;
constexpr int kTargetArrayBuffer        = 34962;
constexpr int kTargetElementArrayBuffer = 34963;

constexpr uint32_t kGlbMagic       = 0x46546C67u;
constexpr uint32_t kGlbVersion     = 2u;
constexpr uint32_t kGlbJsonChunk   = 0x4E4F534Au;
constexpr uint32_t kGlbBinChunk    = 0x004E4942u;

// A GLB chunk length is a uint32; refuse to emit a file no loader could read
// back rather than silently truncating the offset math.
constexpr uint64_t kGlbMaxTotalBytes = 0xFFFFFFF0ull;

// Above this triangle count a single mesh gets the triangle axis parallelised.
// Below it, the per-mesh work is microseconds and the mesh-axis loop is enough.
constexpr uint64_t kBigMeshTriangles = 65536;

// Streaming granularity for generated (not memcpy-able) payload.
constexpr uint64_t kGenChunkTriangles = 1ull << 21;   // 2M tris -> 24 MB scratch
constexpr size_t   kIoBlockBytes      = 8u << 20;     // 8 MB write blocks

float degToRadf(float d) { return d * 0.017453292519943295f; }
float clamp01f(float v) { return (std::max)(0.0f, (std::min)(1.0f, v)); }

// ---------------------------------------------------------------------------
// Parallel range helper. Every hot loop in this file splits the TRIANGLE (or
// vertex) axis, never the object axis - splitting by object count is exactly
// what collapsed to one core on a single-giant-mesh scene.
// ---------------------------------------------------------------------------
template <typename Fn>
void parallelRange(uint64_t count, uint64_t minPerThread, Fn&& fn) {
    if (count == 0) return;
    size_t threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;
    const uint64_t maxUseful = (count + minPerThread - 1) / minPerThread;
    threads = (size_t)(std::min<uint64_t>)((uint64_t)threads, (std::max<uint64_t>)(1ull, maxUseful));
    if (threads <= 1) { fn(0ull, count, (size_t)0); return; }

    const uint64_t chunk = (count + threads - 1) / threads;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        const uint64_t begin = (uint64_t)t * chunk;
        if (begin >= count) break;
        const uint64_t end = (std::min<uint64_t>)(begin + chunk, count);
        futures.push_back(std::async(std::launch::async,
            [&fn, begin, end, t]() { fn(begin, end, t); }));
    }
    for (auto& f : futures) f.get();
}

// Column-major glTF node matrix from a row-major Matrix4x4 (m[row][col]).
std::vector<float> toGltfMatrix(const Matrix4x4& M) {
    std::vector<float> out(16);
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            out[(size_t)c * 4 + r] = M.m[r][c];
    return out;
}

// Decompose a column-major glTF matrix into T/R/S. Folds a mirrored axis into
// one scale component so the remaining 3x3 is a proper rotation before the
// trace-based quaternion extraction (Shoemake).
void decomposeGltfMatrix(const float* m, float outT[3], float outQ[4], float outS[3]) {
    outT[0] = m[12]; outT[1] = m[13]; outT[2] = m[14];

    const float col0[3] = { m[0], m[1], m[2] };
    const float col1[3] = { m[4], m[5], m[6] };
    const float col2[3] = { m[8], m[9], m[10] };
    auto len3 = [](const float v[3]) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); };
    float sx = (std::max)(len3(col0), 1e-8f);
    float sy = (std::max)(len3(col1), 1e-8f);
    float sz = (std::max)(len3(col2), 1e-8f);

    float r[3][3] = {
        { col0[0]/sx, col1[0]/sy, col2[0]/sz },
        { col0[1]/sx, col1[1]/sy, col2[1]/sz },
        { col0[2]/sx, col1[2]/sy, col2[2]/sz }
    };
    const float det = r[0][0]*(r[1][1]*r[2][2] - r[1][2]*r[2][1])
                    - r[0][1]*(r[1][0]*r[2][2] - r[1][2]*r[2][0])
                    + r[0][2]*(r[1][0]*r[2][1] - r[1][1]*r[2][0]);
    if (det < 0.0f) {
        sy = -sy;
        r[0][1] = -r[0][1]; r[1][1] = -r[1][1]; r[2][1] = -r[2][1];
    }
    outS[0] = sx; outS[1] = sy; outS[2] = sz;

    const float trace = r[0][0] + r[1][1] + r[2][2];
    if (trace > 0.0f) {
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        outQ[3] = 0.25f * s;
        outQ[0] = (r[2][1] - r[1][2]) / s;
        outQ[1] = (r[0][2] - r[2][0]) / s;
        outQ[2] = (r[1][0] - r[0][1]) / s;
    } else if (r[0][0] > r[1][1] && r[0][0] > r[2][2]) {
        float s = std::sqrt(1.0f + r[0][0] - r[1][1] - r[2][2]) * 2.0f;
        outQ[3] = (r[2][1] - r[1][2]) / s; outQ[0] = 0.25f * s;
        outQ[1] = (r[0][1] + r[1][0]) / s; outQ[2] = (r[0][2] + r[2][0]) / s;
    } else if (r[1][1] > r[2][2]) {
        float s = std::sqrt(1.0f + r[1][1] - r[0][0] - r[2][2]) * 2.0f;
        outQ[3] = (r[0][2] - r[2][0]) / s; outQ[0] = (r[0][1] + r[1][0]) / s;
        outQ[1] = 0.25f * s;               outQ[2] = (r[1][2] + r[2][1]) / s;
    } else {
        float s = std::sqrt(1.0f + r[2][2] - r[0][0] - r[1][1]) * 2.0f;
        outQ[3] = (r[1][0] - r[0][1]) / s; outQ[0] = (r[0][2] + r[2][0]) / s;
        outQ[1] = (r[1][2] + r[2][1]) / s; outQ[2] = 0.25f * s;
    }
}

// ---------------------------------------------------------------------------
// Binary payload planning
//
// A piece describes bytes that WILL be written, and how to produce them, but
// holds the big ones by pointer. Planning walks the pieces to assign offsets;
// writing walks the same list once and streams. Nothing large is buffered.
// ---------------------------------------------------------------------------
struct BinPiece {
    enum class Kind : uint8_t {
        RawPointer,     // memcpy-able source that outlives the write (SoA arrays)
        Owned,          // small generated bytes held inline
        Zero,           // alignment padding
        MeshIndicesAll, // whole index buffer verbatim (single-material mesh)
        MeshIndicesMat, // indices of one material, filtered at write time
        SkinJoints,     // JOINTS_0 generated from skin_weights
        SkinWeights,    // WEIGHTS_0 generated from skin_weights
        UVFlipped       // TEXCOORD_0 with V flipped back to glTF's origin
    };

    Kind kind = Kind::Owned;
    uint64_t bytes = 0;
    uint64_t offset = 0;

    const void* src = nullptr;              // RawPointer
    std::vector<uint8_t> owned;             // Owned
    const DNA::GeometryDetail* geom = nullptr;  // Mesh*/Skin*
    uint16_t material = 0;                  // MeshIndicesMat
};

class BinPlan {
public:
    // Every accessor here is float- or uint32-typed, so keeping the cursor
    // 4-aligned satisfies the spec's component-size alignment rule everywhere.
    uint64_t add(BinPiece piece) {
        const uint64_t pad = (4ull - (cursor_ % 4ull)) % 4ull;
        if (pad != 0) {
            BinPiece p;
            p.kind = BinPiece::Kind::Zero;
            p.bytes = pad;
            p.offset = cursor_;
            cursor_ += pad;
            pieces_.push_back(std::move(p));
        }
        piece.offset = cursor_;
        cursor_ += piece.bytes;
        if (piece.kind == BinPiece::Kind::Owned) ownedBytes_ += piece.owned.size();
        const uint64_t off = piece.offset;
        pieces_.push_back(std::move(piece));
        return off;
    }

    uint64_t addRaw(const void* src, uint64_t bytes) {
        BinPiece p; p.kind = BinPiece::Kind::RawPointer; p.src = src; p.bytes = bytes;
        return add(std::move(p));
    }
    uint64_t addOwned(std::vector<uint8_t>&& data) {
        BinPiece p; p.kind = BinPiece::Kind::Owned; p.bytes = data.size(); p.owned = std::move(data);
        return add(std::move(p));
    }

    uint64_t totalBytes() const { return cursor_; }
    uint64_t ownedBytes() const { return ownedBytes_; }
    const std::vector<BinPiece>& pieces() const { return pieces_; }

private:
    std::vector<BinPiece> pieces_;
    uint64_t cursor_ = 0;
    uint64_t ownedBytes_ = 0;
};

// Per-vertex material ids, but ONLY if the buffer is actually long enough to be
// indexed by every vertex. A short MaterialID snapshot would otherwise be read
// past its end while classifying triangles - see get_core_attribute_count().
// A null result means "one material, id 0", which is the correct reading of a
// mesh that never got the attribute.
const uint16_t* materialIdsOrNull(const DNA::GeometryDetail& g) {
    const uint16_t* ids = g.get_material_ids();
    if (!ids) return nullptr;
    if (g.get_core_attribute_count(DNA::Attr::MaterialID) < g.get_vertex_count()) return nullptr;
    return ids;
}

// Emits one piece to the stream. Large raw sources are written in blocks so a
// single 1+ GB write never has to succeed atomically.
bool writePiece(std::ostream& out, const BinPiece& piece) {
    switch (piece.kind) {
    case BinPiece::Kind::Zero: {
        static const char zeros[4] = { 0, 0, 0, 0 };
        out.write(zeros, (std::streamsize)piece.bytes);
        return out.good();
    }
    case BinPiece::Kind::Owned: {
        if (!piece.owned.empty())
            out.write(reinterpret_cast<const char*>(piece.owned.data()),
                      (std::streamsize)piece.owned.size());
        return out.good();
    }
    case BinPiece::Kind::RawPointer: {
        const char* p = reinterpret_cast<const char*>(piece.src);
        uint64_t remaining = piece.bytes;
        while (remaining > 0) {
            const size_t block = (size_t)(std::min<uint64_t>)(remaining, kIoBlockBytes);
            out.write(p, (std::streamsize)block);
            if (!out.good()) return false;
            p += block;
            remaining -= block;
        }
        return true;
    }
    case BinPiece::Kind::MeshIndicesAll: {
        const auto& idx = piece.geom->indices;
        const char* p = reinterpret_cast<const char*>(idx.data());
        uint64_t remaining = (uint64_t)idx.size() * sizeof(uint32_t);
        while (remaining > 0) {
            const size_t block = (size_t)(std::min<uint64_t>)(remaining, kIoBlockBytes);
            out.write(p, (std::streamsize)block);
            if (!out.good()) return false;
            p += block;
            remaining -= block;
        }
        return true;
    }
    case BinPiece::Kind::MeshIndicesMat: {
        const auto& idx = piece.geom->indices;
        const uint16_t* matIds = materialIdsOrNull(*piece.geom);
        const uint64_t triCount = idx.size() / 3;
        std::vector<uint32_t> scratch;
        scratch.reserve((size_t)(std::min<uint64_t>)(triCount, kGenChunkTriangles) * 3);
        for (uint64_t base = 0; base < triCount; base += kGenChunkTriangles) {
            const uint64_t end = (std::min<uint64_t>)(base + kGenChunkTriangles, triCount);
            scratch.clear();
            for (uint64_t t = base; t < end; ++t) {
                uint16_t mat = matIds ? matIds[idx[t * 3]] : 0;
                if (mat == MaterialManager::INVALID_MATERIAL_ID) mat = 0;
                if (mat != piece.material) continue;
                scratch.push_back(idx[t * 3 + 0]);
                scratch.push_back(idx[t * 3 + 1]);
                scratch.push_back(idx[t * 3 + 2]);
            }
            if (!scratch.empty()) {
                out.write(reinterpret_cast<const char*>(scratch.data()),
                          (std::streamsize)(scratch.size() * sizeof(uint32_t)));
                if (!out.good()) return false;
            }
        }
        return true;
    }
    case BinPiece::Kind::UVFlipped: {
        // ★ The engine stores UVs with V up (v = 0 at the image BOTTOM); glTF puts
        // (0,0) at the UPPER LEFT. Every other attribute streams out by memcpy, but
        // this one has to be transformed, so it gets a generated piece rather than
        // a RawPointer. Blocked so a 100M-vertex mesh never materialises a full copy.
        const Vec2* uv = piece.geom->get_uvs();
        if (!uv) return false;
        const uint64_t vcount = piece.geom->get_vertex_count();
        constexpr uint64_t kBlockVerts = 1ull << 18;
        std::vector<Vec2> block;
        for (uint64_t base = 0; base < vcount; base += kBlockVerts) {
            const uint64_t end = (std::min<uint64_t>)(base + kBlockVerts, vcount);
            block.clear();
            block.reserve((size_t)(end - base));
            for (uint64_t v = base; v < end; ++v) block.emplace_back(uv[v].x, 1.0f - uv[v].y);
            out.write(reinterpret_cast<const char*>(block.data()),
                      (std::streamsize)(block.size() * sizeof(Vec2)));
            if (!out.good()) return false;
        }
        return true;
    }
    case BinPiece::Kind::SkinJoints:
    case BinPiece::Kind::SkinWeights: {
        // Top-4 influences per vertex, weights renormalised. Generated in blocks
        // so a 100M-vertex skinned mesh never materialises a full array.
        const bool wantJoints = (piece.kind == BinPiece::Kind::SkinJoints);
        const auto& sw = piece.geom->skin_weights;
        const uint64_t vcount = piece.geom->get_vertex_count();
        constexpr uint64_t kBlockVerts = 1ull << 18;
        std::vector<uint8_t> block;
        for (uint64_t base = 0; base < vcount; base += kBlockVerts) {
            const uint64_t end = (std::min<uint64_t>)(base + kBlockVerts, vcount);
            block.clear();
            block.reserve((size_t)(end - base) * (wantJoints ? 8u : 16u));
            for (uint64_t v = base; v < end; ++v) {
                uint16_t j[4] = { 0, 0, 0, 0 };
                float    w[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                if (v < sw.size()) {
                    // Partial sort by weight, keeping at most four influences.
                    for (const auto& infl : sw[v]) {
                        if (infl.second <= 0.0f) continue;
                        int slot = -1;
                        float worst = infl.second;
                        for (int k = 0; k < 4; ++k) {
                            if (w[k] < worst) { worst = w[k]; slot = k; }
                        }
                        if (slot < 0) continue;
                        j[slot] = (uint16_t)(std::max)(0, infl.first);
                        w[slot] = infl.second;
                    }
                    const float sum = w[0] + w[1] + w[2] + w[3];
                    if (sum > 0.0f) { for (int k = 0; k < 4; ++k) w[k] /= sum; }
                }
                if (wantJoints) {
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(j);
                    block.insert(block.end(), b, b + sizeof(j));
                } else {
                    const uint8_t* b = reinterpret_cast<const uint8_t*>(w);
                    block.insert(block.end(), b, b + sizeof(w));
                }
            }
            if (!block.empty()) {
                out.write(reinterpret_cast<const char*>(block.data()),
                          (std::streamsize)block.size());
                if (!out.good()) return false;
            }
        }
        return true;
    }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Scene collection
// ---------------------------------------------------------------------------
struct FlatEntry {
    const TriangleMesh* mesh = nullptr;
    const DNA::GeometryDetail* geom = nullptr;
    uint64_t vertexCount = 0;
    uint64_t triangleCount = 0;
    std::vector<std::pair<uint16_t, uint64_t>> matTriangles; // sorted, ascending
};

struct LegacyGroup {
    std::string name;
    Matrix4x4 transform = Matrix4x4::identity();
    std::vector<std::shared_ptr<Triangle>> triangles;
};

struct InstanceRequest {
    std::string node_name;
    Matrix4x4 transform;
    std::shared_ptr<std::vector<std::shared_ptr<Triangle>>> source;
};

// One scatter/foliage emission read straight from InstanceManager.
//
// ★ Scatter geometry is NOT in scene.world.objects on the Vulkan path.
// syncInstancesToScene() deliberately returns early there ("Never expand that
// data back into one CPU HittableInstance facade per placement"), so an exporter
// that only walks world.objects writes ZERO scatter — silently, with a
// perfectly valid file. Vulkan RT/raster read InstanceGroup directly, and so
// must we: same source, same transform convention, no CPU expansion.
//
// Transform convention copied verbatim from VulkanBackend.cpp:
//     final = instance.toMatrix() * translation(-source.mesh_center) * sourceWorld
struct ScatterEmission {
    std::string name;
    // ★★★ ALL SIBLING MESHES OF ONE SOURCE, NOT ONE EMISSION PER SIBLING.
    // A scattered tree is multi-material, so its ScatterSource holds ~9 flat
    // meshes. Emitting one instancing node PER MESH turned one forest layer into
    // nine on reopen — each carrying a fraction of the tree and re-centred on
    // its own bounding box, which is why the reimported forest also came back at
    // the wrong spacing and scale. One source is one node.
    std::vector<const TriangleMesh*> flatMeshes;                          // flat SoA source
    std::shared_ptr<std::vector<std::shared_ptr<Triangle>>> legacySource; // legacy facade source
    Matrix4x4 sourceToScatter = Matrix4x4::identity();
    std::vector<Matrix4x4> transforms;
};

// Vertex dedup key for the legacy Triangle facade path only. Flat SoA meshes
// never go through this - their vertex arrays are already shared and indexed.
struct VKey {
    float px, py, pz, nx, ny, nz, u, v;
    bool operator==(const VKey& o) const {
        return px == o.px && py == o.py && pz == o.pz &&
               nx == o.nx && ny == o.ny && nz == o.nz &&
               u == o.u && v == o.v;
    }
};
struct VKeyHash {
    size_t operator()(const VKey& k) const {
        std::hash<float> hf;
        size_t seed = hf(k.px);
        auto mix = [&](float f) { seed ^= hf(f) + 0x9e3779b9u + (seed << 6) + (seed >> 2); };
        mix(k.py); mix(k.pz); mix(k.nx); mix(k.ny); mix(k.nz); mix(k.u); mix(k.v);
        return seed;
    }
};

// Interleave-free CPU-side mesh produced from legacy Triangle objects.
struct BuiltLegacyMesh {
    std::vector<float> positions;   // 3 per vertex
    std::vector<float> normals;
    std::vector<float> uvs;         // 2 per vertex
    std::map<uint16_t, std::vector<uint32_t>> indicesByMaterial;
    float minPos[3] = { 0, 0, 0 };
    float maxPos[3] = { 0, 0, 0 };
    bool hasUV = false;
};

BuiltLegacyMesh buildLegacyMesh(const std::vector<std::shared_ptr<Triangle>>& tris) {
    BuiltLegacyMesh out;
    std::unordered_map<VKey, uint32_t, VKeyHash> cache;
    cache.reserve(tris.size() * 2);
    bool first = true;

    for (const auto& tri : tris) {
        if (!tri) continue;
        uint16_t mat = tri->getMaterialID();
        if (mat == MaterialManager::INVALID_MATERIAL_ID) mat = 0;
        auto& dst = out.indicesByMaterial[mat];

        for (int c = 0; c < 3; ++c) {
            const Vec3 p = tri->getOriginalVertexPosition(c);
            const Vec3 n = tri->getOriginalVertexNormal(c);
            const Vec2 uv = tri->t_ref(c);
            VKey key{ p.x, p.y, p.z, n.x, n.y, n.z, uv.x, uv.y };

            auto it = cache.find(key);
            uint32_t idx;
            if (it != cache.end()) {
                idx = it->second;
            } else {
                idx = (uint32_t)(out.positions.size() / 3);
                out.positions.insert(out.positions.end(), { p.x, p.y, p.z });
                out.normals.insert(out.normals.end(), { n.x, n.y, n.z });
                // ★ V flipped back to glTF's upper-left origin; see the matching
                // note in GltfDirectReader::emitPrimitive. Import and export must
                // move together — they were BOTH missing this, which is why the
                // round trip looked correct and hid the bug.
                out.uvs.insert(out.uvs.end(), { uv.x, 1.0f - uv.y });
                if (uv.x != 0.0f || uv.y != 0.0f) out.hasUV = true;
                cache.emplace(key, idx);
                if (first) {
                    out.minPos[0] = out.maxPos[0] = p.x;
                    out.minPos[1] = out.maxPos[1] = p.y;
                    out.minPos[2] = out.maxPos[2] = p.z;
                    first = false;
                } else {
                    out.minPos[0] = (std::min)(out.minPos[0], p.x);
                    out.minPos[1] = (std::min)(out.minPos[1], p.y);
                    out.minPos[2] = (std::min)(out.minPos[2], p.z);
                    out.maxPos[0] = (std::max)(out.maxPos[0], p.x);
                    out.maxPos[1] = (std::max)(out.maxPos[1], p.y);
                    out.maxPos[2] = (std::max)(out.maxPos[2], p.z);
                }
            }
            dst.push_back(idx);
        }
    }
    return out;
}

// Per-material triangle counts for one flat mesh, serial. collect() has a
// triangle-parallel version for the giant meshes; scatter SOURCES are assets
// (thousands of triangles), so the serial form is the right tool there.
void histogramFor(FlatEntry& e) {
    const uint16_t* matIds = materialIdsOrNull(*e.geom);
    if (!matIds) {
        e.matTriangles.emplace_back((uint16_t)0, e.triangleCount);
        return;
    }
    std::map<uint16_t, uint64_t> counts;
    const auto& idx = e.geom->indices;
    for (uint64_t t = 0; t < e.triangleCount; ++t) {
        uint16_t m = matIds[idx[t * 3]];
        if (m == MaterialManager::INVALID_MATERIAL_ID) m = 0;
        ++counts[m];
    }
    for (const auto& kv : counts) e.matTriangles.emplace_back(kv.first, kv.second);
}

std::vector<uint8_t> floatsToBytes(const std::vector<float>& v) {
    std::vector<uint8_t> out(v.size() * sizeof(float));
    if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}
std::vector<uint8_t> uintsToBytes(const std::vector<uint32_t>& v) {
    std::vector<uint8_t> out(v.size() * sizeof(uint32_t));
    if (!v.empty()) std::memcpy(out.data(), v.data(), out.size());
    return out;
}

} // namespace

// ===========================================================================
// Writer
// ===========================================================================
namespace {

class GltfWriter {
public:
    GltfWriter(SceneData& scene, const ExportSettings& settings,
               const std::map<uint16_t, std::shared_ptr<Material>>& overrides)
        : scene_(scene), settings_(settings), overrides_(overrides) {}

    bool run(const std::string& filepath,
             const std::vector<std::shared_ptr<Hittable>>& selected,
             WriteStats& stats, std::string& error);

private:
    // --- phases -------------------------------------------------------------
    void collect(const std::vector<std::shared_ptr<Hittable>>& selected);
    void collectScatter();
    // Appends this entry's primitives (one per material) to `primitives` and
    // plans its vertex buffers. Split out of emitFlatMesh so several sibling
    // meshes can land in ONE glTF mesh without their vertex data being written
    // twice — BinPlan has no dedup, every add() appends.
    bool buildFlatPrimitives(const FlatEntry& e, json& primitives, bool& wantSkinOut);
    int emitFlatMeshGroup(const std::vector<const FlatEntry*>& entries, const std::string& meshName);
    int  emitFlatMesh(const FlatEntry& e);   // -> glTF mesh index, or -1
    void planScatter();
    void buildMaterials();
    void planGeometry();
    void planSkinsAndAnimations();
    void planCamerasAndLights();
    bool emit(const std::string& filepath, std::string& error);

    // --- json helpers -------------------------------------------------------
    int addBufferView(uint64_t offset, uint64_t length, int target);
    int addAccessor(int bufferView, int componentType, uint64_t count,
                    const char* type, bool normalized = false);
    int addNode(json node, bool asSceneRoot);
    void useExtension(const char* name);

    // --- material helpers ---------------------------------------------------
    int  materialIndexFor(uint16_t materialId);
    int  imageIndexForTexture(const std::shared_ptr<Texture>& tex, bool srgbSource);
    int  imageIndexForPackedMR(const std::shared_ptr<Texture>& rough,
                               const std::shared_ptr<Texture>& metal);
    int  imageIndexForBaseColorWithAlpha(const std::shared_ptr<Texture>& albedo,
                                         const std::shared_ptr<Texture>& opacity,
                                         const Vec3& albedoFactor,
                                         float opacityScalar);
    int  textureIndexForImage(int imageIndex);
    json textureInfo(int textureIndex, const GpuMaterial* gpu);

    SceneData& scene_;
    const ExportSettings& settings_;
    const std::map<uint16_t, std::shared_ptr<Material>>& overrides_;

    json gltf_ = json::object();
    BinPlan plan_;

    std::vector<FlatEntry> flat_;
    std::vector<LegacyGroup> legacy_;
    std::vector<InstanceRequest> instances_;
    std::vector<ScatterEmission> scatter_;
    // A flat source mesh that is BOTH a scene object and a scatter source must
    // produce one glTF mesh, referenced by both its own node and the instancing
    // node - not two copies of the same vertex data.
    std::unordered_map<const TriangleMesh*, int> flatMeshIndex_;

    std::unordered_map<uint16_t, int> materialIdToIndex_;
    std::unordered_map<std::string, int> imageKeyToIndex_;
    std::unordered_map<int, int> imageToTextureIndex_;
    std::unordered_map<std::string, int> nodeNameToIndex_;
    // Nodes whose mesh carries JOINTS_0/WEIGHTS_0; the skin index is stamped on
    // them once the skeleton exists (phase 4 runs after geometry planning).
    std::vector<int> skinnedNodes_;
    std::unordered_set<int> skinnedMeshes_;

    int defaultSamplerIndex_ = -1;
    WriteStats stats_{};
};

// ---------------------------------------------------------------------------
int GltfWriter::addBufferView(uint64_t offset, uint64_t length, int target) {
    json v = { {"buffer", 0}, {"byteOffset", offset}, {"byteLength", length} };
    if (target != 0) v["target"] = target;
    gltf_["bufferViews"].push_back(std::move(v));
    return (int)gltf_["bufferViews"].size() - 1;
}

int GltfWriter::addAccessor(int bufferView, int componentType, uint64_t count,
                            const char* type, bool normalized) {
    json a = { {"bufferView", bufferView}, {"componentType", componentType},
               {"count", count}, {"type", type} };
    if (normalized) a["normalized"] = true;
    gltf_["accessors"].push_back(std::move(a));
    return (int)gltf_["accessors"].size() - 1;
}

int GltfWriter::addNode(json node, bool asSceneRoot) {
    gltf_["nodes"].push_back(std::move(node));
    const int index = (int)gltf_["nodes"].size() - 1;
    if (asSceneRoot) gltf_["scenes"][0]["nodes"].push_back(index);
    return index;
}

void GltfWriter::useExtension(const char* name) {
    for (const auto& e : gltf_["extensionsUsed"]) {
        if (e.is_string() && e.get<std::string>() == name) return;
    }
    gltf_["extensionsUsed"].push_back(name);
}

// ---------------------------------------------------------------------------
// Phase 1: collect
// ---------------------------------------------------------------------------
void GltfWriter::collect(const std::vector<std::shared_ptr<Hittable>>& selected) {
    if (!settings_.export_geometry) return;

    std::unordered_set<std::string> selectedNames;
    if (settings_.export_selected_only) {
        for (const auto& obj : selected) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj))            selectedNames.insert(tri->getNodeName());
            else if (auto m = std::dynamic_pointer_cast<TriangleMesh>(obj))     selectedNames.insert(m->nodeName);
            else if (auto i = std::dynamic_pointer_cast<HittableInstance>(obj)) selectedNames.insert(i->node_name);
        }
    }
    // ★ An object deleted in the editor is HIDDEN, not removed: it stays in
    // world.objects until the next save physically purges it. rtapi::listObjects
    // filters those out; the exporter never did, so every deleted object was
    // still written into the file. Nobody reports that as a bug - the export
    // "works", it just quietly contains things that are not in the scene any
    // more. Found the first time export got a script surface (scene.export_gltf):
    // list_objects said 2 objects, the file held 4 meshes.
    auto rejected = [&](const std::string& name) {
        if (scene_.isEditorPendingDeleteObjectName(name)) return true;
        return settings_.export_selected_only && selectedNames.find(name) == selectedNames.end();
    };

    std::map<std::string, LegacyGroup> legacyByNode;

    for (const auto& obj : scene_.world.objects) {
        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (!mesh->geometry || rejected(mesh->nodeName)) continue;
            const DNA::GeometryDetail* geom = mesh->geometry.get();
            if (geom->indices.empty() || geom->get_vertex_count() == 0) continue;
            if (!geom->get_positions_orig() && !geom->get_positions()) continue;

            FlatEntry e;
            e.mesh = mesh.get();
            e.geom = geom;
            e.vertexCount = geom->get_vertex_count();
            e.triangleCount = geom->indices.size() / 3;
            flat_.push_back(std::move(e));
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            const std::string name = tri->getNodeName();
            if (rejected(name)) continue;
            auto& g = legacyByNode[name];
            if (g.triangles.empty()) {
                g.name = name;
                g.transform = tri->getTransformMatrix();
            }
            g.triangles.push_back(tri);
        } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (rejected(inst->node_name)) continue;
            if (!inst->source_triangles || inst->source_triangles->empty()) continue;
            InstanceRequest r;
            r.node_name = inst->node_name;
            r.transform = inst->transform;
            r.source = inst->source_triangles;
            instances_.push_back(std::move(r));
        }
    }

    legacy_.reserve(legacyByNode.size());
    for (auto& kv : legacyByNode) legacy_.push_back(std::move(kv.second));

    // Per-material triangle histogram. This is the ONLY pass over the triangle
    // array before writing, and it is what makes plan-then-write possible: after
    // it, every accessor count and every byte offset is known.
    //
    // Small meshes: parallel over the mesh list. Big meshes: parallel over their
    // own triangles, one mesh at a time (splitting by object count is what
    // collapsed to a single core on a one-giant-mesh scene).
    std::vector<size_t> bigMeshes;
    std::vector<size_t> smallMeshes;
    for (size_t i = 0; i < flat_.size(); ++i) {
        (flat_[i].triangleCount >= kBigMeshTriangles ? bigMeshes : smallMeshes).push_back(i);
    }

    auto histogramSerial = [](FlatEntry& e) {
        const uint16_t* matIds = materialIdsOrNull(*e.geom);
        if (!matIds) {
            e.matTriangles.emplace_back((uint16_t)0, e.triangleCount);
            return;
        }
        std::map<uint16_t, uint64_t> counts;
        const auto& idx = e.geom->indices;
        for (uint64_t t = 0; t < e.triangleCount; ++t) {
            uint16_t m = matIds[idx[t * 3]];
            if (m == MaterialManager::INVALID_MATERIAL_ID) m = 0;
            ++counts[m];
        }
        for (const auto& kv : counts) e.matTriangles.emplace_back(kv.first, kv.second);
    };

    if (!smallMeshes.empty()) {
        parallelRange(smallMeshes.size(), 1, [&](uint64_t b, uint64_t en, size_t) {
            for (uint64_t i = b; i < en; ++i) histogramSerial(flat_[smallMeshes[(size_t)i]]);
        });
    }

    for (size_t mi : bigMeshes) {
        FlatEntry& e = flat_[mi];
        const uint16_t* matIds = materialIdsOrNull(*e.geom);
        if (!matIds) {
            e.matTriangles.emplace_back((uint16_t)0, e.triangleCount);
            continue;
        }
        const auto& idx = e.geom->indices;
        size_t threads = std::thread::hardware_concurrency();
        if (threads == 0) threads = 4;
        std::vector<std::vector<uint64_t>> perThread(threads);
        parallelRange(e.triangleCount, 8192, [&](uint64_t b, uint64_t en, size_t t) {
            auto& local = perThread[(std::min)(t, threads - 1)];
            local.assign(65536, 0);
            for (uint64_t i = b; i < en; ++i) {
                uint16_t m = matIds[idx[i * 3]];
                if (m == MaterialManager::INVALID_MATERIAL_ID) m = 0;
                ++local[m];
            }
        });
        std::vector<uint64_t> total(65536, 0);
        for (const auto& local : perThread) {
            if (local.size() != 65536) continue;
            for (size_t m = 0; m < 65536; ++m) total[m] += local[m];
        }
        for (size_t m = 0; m < 65536; ++m) {
            if (total[m] != 0) e.matTriangles.emplace_back((uint16_t)m, total[m]);
        }
        if (e.matTriangles.empty()) e.matTriangles.emplace_back((uint16_t)0, e.triangleCount);
    }
}

// ---------------------------------------------------------------------------
// Phase 2: materials, textures, images
// ---------------------------------------------------------------------------
namespace {

void pngWriteToVector(void* context, void* data, int size) {
    auto* buffer = static_cast<std::vector<uint8_t>*>(context);
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    buffer->insert(buffer->end(), bytes, bytes + size);
}

uint8_t linearToSrgbByte(float v) {
    v = clamp01f(v);
    const float s = (v <= 0.0031308f) ? (v * 12.92f)
                                      : (1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f);
    return (uint8_t)(clamp01f(s) * 255.0f + 0.5f);
}

// ★ ROW -> SAMPLE-V. Texture::get_color_bilinear maps v to source row
// (1 - v) * (height - 1), i.e. v = 0 is the LAST row of `pixels`. Everything the
// writer emits verbatim (encodeTexturePng streams `pixels` untouched) therefore
// puts source row 0 at image row 0, and any RESAMPLED image has to reproduce
// that same mapping or it lands in the file upside down.
//
// The packed metallic-roughness image did not: it sampled with v = (y+0.5)/h and
// wrote the source bottom row into image row 0. Nobody reported it because a
// flipped roughness map still looks like a roughness map — the exact failure
// class this repo keeps paying for ("plausible, therefore invisible").
inline float imageRowToSampleV(uint64_t y, int h) {
    return 1.0f - (((float)y + 0.5f) / (float)h);
}

// Encodes a Texture to PNG bytes. `srgbSource` requests linear->sRGB conversion
// (base colour and emissive only - normal/roughness/metallic stay linear).
bool encodeTexturePng(const Texture& tex, bool srgbSource, std::vector<uint8_t>& out) {
    const int w = tex.width;
    const int h = tex.height;
    if (w <= 0 || h <= 0) return false;

    if (!tex.pixels.empty()) {
        // CompactVec4 is {r,g,b,a} bytes, so the pixel array is already the
        // exact RGBA8 buffer stb wants - no per-pixel repacking.
        if (!srgbSource || tex.is_srgb) {
            stbi_write_png_to_func(pngWriteToVector, &out, w, h, 4,
                                   tex.pixels.data(), w * 4);
        } else {
            std::vector<uint8_t> conv((size_t)w * h * 4);
            parallelRange((uint64_t)w * h, 65536, [&](uint64_t b, uint64_t e, size_t) {
                for (uint64_t i = b; i < e; ++i) {
                    const auto& p = tex.pixels[(size_t)i];
                    conv[(size_t)i * 4 + 0] = linearToSrgbByte(p.r / 255.0f);
                    conv[(size_t)i * 4 + 1] = linearToSrgbByte(p.g / 255.0f);
                    conv[(size_t)i * 4 + 2] = linearToSrgbByte(p.b / 255.0f);
                    conv[(size_t)i * 4 + 3] = p.a;
                }
            });
            stbi_write_png_to_func(pngWriteToVector, &out, w, h, 4, conv.data(), w * 4);
        }
        return !out.empty();
    }

    if (tex.is_hdr && !tex.float_pixels.empty()) {
        std::vector<uint8_t> conv((size_t)w * h * 4);
        parallelRange((uint64_t)w * h, 65536, [&](uint64_t b, uint64_t e, size_t) {
            for (uint64_t i = b; i < e; ++i) {
                const auto& p = tex.float_pixels[(size_t)i];
                if (srgbSource) {
                    conv[(size_t)i * 4 + 0] = linearToSrgbByte(p.x);
                    conv[(size_t)i * 4 + 1] = linearToSrgbByte(p.y);
                    conv[(size_t)i * 4 + 2] = linearToSrgbByte(p.z);
                } else {
                    conv[(size_t)i * 4 + 0] = (uint8_t)(clamp01f(p.x) * 255.0f + 0.5f);
                    conv[(size_t)i * 4 + 1] = (uint8_t)(clamp01f(p.y) * 255.0f + 0.5f);
                    conv[(size_t)i * 4 + 2] = (uint8_t)(clamp01f(p.z) * 255.0f + 0.5f);
                }
                conv[(size_t)i * 4 + 3] = (uint8_t)(clamp01f(p.w) * 255.0f + 0.5f);
            }
        });
        stbi_write_png_to_func(pngWriteToVector, &out, w, h, 4, conv.data(), w * 4);
        return !out.empty();
    }

    return false;
}

// A texture that only exists as a file on disk is embedded VERBATIM when the
// container already matches a glTF-legal mime type. Decoding it just to
// re-encode a PNG would be slower, larger and lossy for JPEG sources.
bool readFileBytes(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return false;
    const std::streamoff size = in.tellg();
    if (size <= 0) return false;
    out.resize((size_t)size);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), size);
    return in.good() || in.gcount() == size;
}

const char* mimeTypeForExtension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    return nullptr;
}

} // namespace

int GltfWriter::textureIndexForImage(int imageIndex) {
    auto it = imageToTextureIndex_.find(imageIndex);
    if (it != imageToTextureIndex_.end()) return it->second;

    if (defaultSamplerIndex_ < 0) {
        gltf_["samplers"].push_back({ {"magFilter", 9729},        // LINEAR
                                      {"minFilter", 9987},        // LINEAR_MIPMAP_LINEAR
                                      {"wrapS", 10497}, {"wrapT", 10497} });
        defaultSamplerIndex_ = (int)gltf_["samplers"].size() - 1;
    }

    gltf_["textures"].push_back({ {"sampler", defaultSamplerIndex_}, {"source", imageIndex} });
    const int idx = (int)gltf_["textures"].size() - 1;
    imageToTextureIndex_[imageIndex] = idx;
    return idx;
}

int GltfWriter::imageIndexForTexture(const std::shared_ptr<Texture>& tex, bool srgbSource) {
    if (!tex) return -1;

    const std::string key = "tex:" + std::to_string((uintptr_t)tex.get()) + (srgbSource ? ":s" : ":l");
    auto it = imageKeyToIndex_.find(key);
    if (it != imageKeyToIndex_.end()) return it->second;

    std::vector<uint8_t> bytes;
    const char* mime = "image/png";

    if (!encodeTexturePng(*tex, srgbSource, bytes)) {
        // No decoded pixels in RAM: embed the on-disk file as-is if we can.
        const char* diskMime = tex->name.empty() ? nullptr : mimeTypeForExtension(tex->name);
        if (!diskMime || !std::filesystem::exists(tex->name) || !readFileBytes(tex->name, bytes)) {
            imageKeyToIndex_[key] = -1;
            return -1;
        }
        mime = diskMime;
    }

    const uint64_t offset = plan_.addOwned(std::move(bytes));
    const uint64_t length = plan_.pieces().back().bytes;
    const int view = addBufferView(offset, length, 0);
    gltf_["images"].push_back({ {"bufferView", view}, {"mimeType", mime} });
    const int index = (int)gltf_["images"].size() - 1;
    imageKeyToIndex_[key] = index;
    return index;
}

// glTF stores roughness in G and metallic in B of ONE texture. The Assimp path
// emitted them as two separate non-standard slots, which every conformant viewer
// ignores - so roughness/metallic maps were silently lost on export. Packing
// them here is a fidelity fix, not just a size win.
int GltfWriter::imageIndexForPackedMR(const std::shared_ptr<Texture>& rough,
                                      const std::shared_ptr<Texture>& metal) {
    if (!rough && !metal) return -1;

    const std::string key = "mr:" + std::to_string((uintptr_t)rough.get()) + "/" +
                            std::to_string((uintptr_t)metal.get());
    auto it = imageKeyToIndex_.find(key);
    if (it != imageKeyToIndex_.end()) return it->second;

    const int w = (std::max)(rough ? rough->width : 0, metal ? metal->width : 0);
    const int h = (std::max)(rough ? rough->height : 0, metal ? metal->height : 0);
    if (w <= 0 || h <= 0) { imageKeyToIndex_[key] = -1; return -1; }

    std::vector<uint8_t> rgb((size_t)w * h * 3, 255);
    parallelRange((uint64_t)h, 64, [&](uint64_t y0, uint64_t y1, size_t) {
        for (uint64_t y = y0; y < y1; ++y) {
            const float v = imageRowToSampleV(y, h);
            for (int x = 0; x < w; ++x) {
                const float u = ((float)x + 0.5f) / (float)w;
                const size_t o = ((size_t)y * w + x) * 3;
                rgb[o + 0] = 255; // unused (occlusion channel)
                if (rough) rgb[o + 1] = (uint8_t)(clamp01f(rough->get_color_bilinear(u, v).x) * 255.0f + 0.5f);
                if (metal) rgb[o + 2] = (uint8_t)(clamp01f(metal->get_color_bilinear(u, v).x) * 255.0f + 0.5f);
            }
        }
    });

    std::vector<uint8_t> png;
    stbi_write_png_to_func(pngWriteToVector, &png, w, h, 3, rgb.data(), w * 3);
    if (png.empty()) { imageKeyToIndex_[key] = -1; return -1; }

    const uint64_t offset = plan_.addOwned(std::move(png));
    const uint64_t length = plan_.pieces().back().bytes;
    const int view = addBufferView(offset, length, 0);
    gltf_["images"].push_back({ {"bufferView", view}, {"mimeType", "image/png"} });
    const int index = (int)gltf_["images"].size() - 1;
    imageKeyToIndex_[key] = index;
    return index;
}

// ---------------------------------------------------------------------------
// BASE COLOUR + ALPHA
//
// ★ glTF has NO opacity texture slot. Alpha lives in the ALPHA CHANNEL of
// baseColorTexture, and that is the only place a conformant importer looks.
// RayTrophi keeps opacity in its own MaterialProperty slot, and the writer used
// to read albedo/normal/rough/metal/emission and simply never look at it — so
// every masked material (foliage, decals, glass cards) exported fully opaque.
// The file was valid, the import "worked", and the leaves were solid quads.
//
// So the two maps are composited into one RGBA image here. RGB follows the
// same sRGB encoding the albedo path uses; A follows Material's own opacity
// semantics (MaterialProperty::evaluateOpacity): the texture's alpha channel
// when it has one, otherwise its red channel, scaled by the property alpha.
// The engine's 0.1 noise floor is mirrored deliberately - the exported file
// should match what RayTrophi renders, not a cleaner theory of it.
// ---------------------------------------------------------------------------
int GltfWriter::imageIndexForBaseColorWithAlpha(const std::shared_ptr<Texture>& albedo,
                                                const std::shared_ptr<Texture>& opacity,
                                                const Vec3& albedoFactor,
                                                float opacityScalar) {
    if (!opacity) return -1;

    const std::string key = "bca:" + std::to_string((uintptr_t)albedo.get()) + "/" +
                            std::to_string((uintptr_t)opacity.get()) + "/" +
                            std::to_string(opacityScalar);
    auto it = imageKeyToIndex_.find(key);
    if (it != imageKeyToIndex_.end()) return it->second;

    const int w = (std::max)(albedo ? albedo->width : 0, opacity->width);
    const int h = (std::max)(albedo ? albedo->height : 0, opacity->height);
    if (w <= 0 || h <= 0) { imageKeyToIndex_[key] = -1; return -1; }

    // Both maps must be sampleable. A texture that only exists on disk returns
    // black/1.0 from the samplers, which would quietly export a black cutout —
    // worse than losing the alpha. Refuse instead; the caller then embeds the
    // albedo verbatim and logs that opacity could not be merged.
    const bool albedoSampleable = !albedo ||
        !albedo->pixels.empty() || !albedo->float_pixels.empty();
    const bool opacitySampleable =
        !opacity->pixels.empty() || !opacity->float_pixels.empty();
    if (!albedoSampleable || !opacitySampleable) { imageKeyToIndex_[key] = -1; return -1; }

    const bool opacityHasAlpha = opacity->has_alpha && !opacity->pixels.empty();
    const float scale = clamp01f(opacityScalar);

    std::vector<uint8_t> rgba((size_t)w * h * 4, 255);
    parallelRange((uint64_t)h, 64, [&](uint64_t y0, uint64_t y1, size_t) {
        for (uint64_t y = y0; y < y1; ++y) {
            const float v = imageRowToSampleV(y, h);
            for (int x = 0; x < w; ++x) {
                const float u = ((float)x + 0.5f) / (float)w;
                const size_t o = ((size_t)y * w + x) * 4;

                const Vec3 lin = albedo ? albedo->get_color_bilinear(u, v) : albedoFactor;
                rgba[o + 0] = linearToSrgbByte(lin.x);
                rgba[o + 1] = linearToSrgbByte(lin.y);
                rgba[o + 2] = linearToSrgbByte(lin.z);

                float a = opacityHasAlpha ? opacity->get_alpha_bilinear(u, v)
                                          : clamp01f(opacity->get_color_bilinear(u, v).x);
                // Same noise floor MaterialProperty::evaluateOpacity applies.
                a = (a < 0.1f) ? 0.0f : a * scale;
                rgba[o + 3] = (uint8_t)(clamp01f(a) * 255.0f + 0.5f);
            }
        }
    });

    std::vector<uint8_t> png;
    stbi_write_png_to_func(pngWriteToVector, &png, w, h, 4, rgba.data(), w * 4);
    if (png.empty()) { imageKeyToIndex_[key] = -1; return -1; }

    const uint64_t offset = plan_.addOwned(std::move(png));
    const uint64_t length = plan_.pieces().back().bytes;
    const int view = addBufferView(offset, length, 0);
    gltf_["images"].push_back({ {"bufferView", view}, {"mimeType", "image/png"} });
    const int index = (int)gltf_["images"].size() - 1;
    imageKeyToIndex_[key] = index;
    return index;
}

json GltfWriter::textureInfo(int textureIndex, const GpuMaterial* gpu) {
    json info = { {"index", textureIndex} };
    if (gpu) {
        const bool hasOffset = (gpu->uv_offset_x != 0.0f || gpu->uv_offset_y != 0.0f);
        const bool hasScale  = (gpu->uv_scale_x != 1.0f || gpu->uv_scale_y != 1.0f ||
                                gpu->uv_tiling_x != 1.0f || gpu->uv_tiling_y != 1.0f);
        if (hasOffset || hasScale) {
            // The Assimp path dropped UV scale/offset entirely; a tiled material
            // exported as untiled looks correct-but-wrong, which is the failure
            // mode nobody reports. Carry it in the standard extension instead.
            useExtension("KHR_texture_transform");
            info["extensions"]["KHR_texture_transform"] = {
                {"offset", { gpu->uv_offset_x, gpu->uv_offset_y }},
                {"scale",  { gpu->uv_scale_x * gpu->uv_tiling_x,
                             gpu->uv_scale_y * gpu->uv_tiling_y }}
            };
        }
    }
    return info;
}

int GltfWriter::materialIndexFor(uint16_t materialId) {
    auto it = materialIdToIndex_.find(materialId);
    if (it != materialIdToIndex_.end()) return it->second;

    auto ov = overrides_.find(materialId);
    std::shared_ptr<Material> mat = (ov != overrides_.end())
        ? ov->second
        : MaterialManager::getInstance().getMaterialShared(materialId);

    std::string name = "Mat_" + std::to_string(materialId);
    if (mat && !mat->materialName.empty()) name = mat->materialName;

    json m = json::object();
    m["name"] = name;
    json pbr = json::object();
    pbr["baseColorFactor"] = { 1.0f, 1.0f, 1.0f, 1.0f };
    pbr["metallicFactor"] = 0.0f;
    pbr["roughnessFactor"] = 1.0f;

    if (mat && settings_.export_materials) {
        auto pMat = std::dynamic_pointer_cast<PrincipledBSDF>(mat);
        const GpuMaterial* gpu = mat->gpuMaterial ? mat->gpuMaterial.get() : nullptr;

        auto pick = [&](const MaterialProperty& a, const MaterialProperty* b) -> std::shared_ptr<Texture> {
            if (a.texture) return a.texture;
            if (b && b->texture) return b->texture;
            return nullptr;
        };

        auto albedoTex = pick(mat->albedoProperty, pMat ? &pMat->albedoProperty : nullptr);
        auto normalTex = pick(mat->normalProperty, pMat ? &pMat->normalProperty : nullptr);
        auto roughTex  = pick(mat->roughnessProperty, pMat ? &pMat->roughnessProperty : nullptr);
        auto metalTex  = pick(mat->metallicProperty, pMat ? &pMat->metallicProperty : nullptr);
        auto emitTex   = pick(mat->emissionProperty, pMat ? &pMat->emissionProperty : nullptr);
        auto opacityTex = pick(mat->opacityProperty, pMat ? &pMat->opacityProperty : nullptr);

        // GpuMaterial::opacity IS opacityProperty.alpha (PBRMaterialSnapshot.h),
        // so read one of them, never both - applying it twice would darken the
        // alpha by its own square and look like "a bit too transparent".
        const float opacity = gpu ? clamp01f(gpu->opacity)
                                  : clamp01f(mat->opacityProperty.alpha);

        // ★ Alpha is not a slot in glTF, it is baseColorTexture's A channel.
        // Composite first; only fall back to the plain albedo path when there is
        // no opacity map (or it cannot be sampled).
        bool alphaBakedIntoBaseColor = false;
        if (opacityTex) {
            const int img = imageIndexForBaseColorWithAlpha(
                albedoTex, opacityTex,
                gpu ? Vec3(gpu->albedo.x, gpu->albedo.y, gpu->albedo.z) : Vec3(1.0f, 1.0f, 1.0f), opacity);
            if (img >= 0) {
                pbr["baseColorTexture"] = textureInfo(textureIndexForImage(img), gpu);
                // The scalar is already inside the image; leaving it in the
                // factor too would apply it a second time.
                pbr["baseColorFactor"] = { 1.0f, 1.0f, 1.0f, 1.0f };
                alphaBakedIntoBaseColor = true;
            } else {
                // Say so. A dropped opacity map is invisible in the output file
                // (it is still a valid, opaque material) and this is the only
                // place that knows it happened.
                SCENE_LOG_WARN("[Export] '" + name + "': opacity map could not be merged "
                    "into baseColor alpha (source pixels not resident); material exports opaque.");
            }
        }

        if (!alphaBakedIntoBaseColor) {
            if (albedoTex) {
                const int img = imageIndexForTexture(albedoTex, /*srgbSource*/ true);
                if (img >= 0) pbr["baseColorTexture"] = textureInfo(textureIndexForImage(img), gpu);
                pbr["baseColorFactor"] = { 1.0f, 1.0f, 1.0f, opacity };
            } else if (gpu) {
                pbr["baseColorFactor"] = { gpu->albedo.x, gpu->albedo.y, gpu->albedo.z, opacity };
            }
        }

        if (roughTex || metalTex) {
            const int img = imageIndexForPackedMR(roughTex, metalTex);
            if (img >= 0) pbr["metallicRoughnessTexture"] = textureInfo(textureIndexForImage(img), gpu);
        }
        if (gpu) {
            pbr["roughnessFactor"] = clamp01f(gpu->roughness);
            pbr["metallicFactor"] = clamp01f(gpu->metallic);
        } else if (pMat) {
            pbr["roughnessFactor"] = clamp01f(pMat->roughnessProperty.intensity);
            pbr["metallicFactor"] = clamp01f(pMat->metallicProperty.intensity);
        }

        if (normalTex) {
            const int img = imageIndexForTexture(normalTex, /*srgbSource*/ false);
            if (img >= 0) {
                json ninfo = textureInfo(textureIndexForImage(img), gpu);
                if (gpu && gpu->normal_strength != 1.0f) ninfo["scale"] = gpu->normal_strength;
                m["normalTexture"] = std::move(ninfo);
            }
        }

        if (emitTex) {
            const int img = imageIndexForTexture(emitTex, /*srgbSource*/ true);
            if (img >= 0) m["emissiveTexture"] = textureInfo(textureIndexForImage(img), gpu);
            m["emissiveFactor"] = { 1.0f, 1.0f, 1.0f };
        } else if (gpu) {
            const float peak = (std::max)({ gpu->emission.x, gpu->emission.y, gpu->emission.z });
            if (peak > 0.0f) {
                // glTF clamps emissiveFactor to [0,1]; anything brighter has to
                // ride on KHR_materials_emissive_strength or it is silently lost.
                m["emissiveFactor"] = { gpu->emission.x / (std::max)(peak, 1.0f),
                                        gpu->emission.y / (std::max)(peak, 1.0f),
                                        gpu->emission.z / (std::max)(peak, 1.0f) };
                if (peak > 1.0f) {
                    useExtension("KHR_materials_emissive_strength");
                    m["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"] = peak;
                }
            }
        }

        if (gpu) {
            if (gpu->transmission > 0.0f) {
                useExtension("KHR_materials_transmission");
                m["extensions"]["KHR_materials_transmission"]["transmissionFactor"] = clamp01f(gpu->transmission);
            }
            if (gpu->ior > 0.0f && std::abs(gpu->ior - 1.5f) > 1e-4f) {
                useExtension("KHR_materials_ior");
                m["extensions"]["KHR_materials_ior"]["ior"] = gpu->ior;
            }
            if (gpu->clearcoat > 0.0f) {
                useExtension("KHR_materials_clearcoat");
                m["extensions"]["KHR_materials_clearcoat"]["clearcoatFactor"] = clamp01f(gpu->clearcoat);
                m["extensions"]["KHR_materials_clearcoat"]["clearcoatRoughnessFactor"] = clamp01f(gpu->clearcoat_roughness);
            }
        }

        // alphaMode gates whether an importer looks at alpha AT ALL: the glTF
        // default is OPAQUE, under which a perfectly correct alpha channel is
        // ignored. So it must follow the alpha we actually wrote, not only the
        // scalar - a fully-textured cutout has opacity == 1.0 and would still
        // have exported as OPAQUE.
        //
        // BLEND, not MASK: RayTrophi's opacity is a continuous value
        // (MaterialProperty::evaluateOpacity) and the path tracer treats it that
        // way, so a cutoff here would be a look change disguised as a format
        // choice. A viewer that prefers MASK can threshold it.
        if (alphaBakedIntoBaseColor || opacity < 1.0f) {
            m["alphaMode"] = "BLEND";
        }

    }

    m["pbrMetallicRoughness"] = std::move(pbr);
    m["doubleSided"] = true;
    gltf_["materials"].push_back(std::move(m));
    const int index = (int)gltf_["materials"].size() - 1;
    materialIdToIndex_[materialId] = index;
    return index;
}

void GltfWriter::buildMaterials() {
    // Deterministic order: every material referenced by geometry, ascending id.
    std::vector<uint16_t> used;
    std::unordered_set<uint16_t> seen;
    auto note = [&](uint16_t id) { if (seen.insert(id).second) used.push_back(id); };

    for (const auto& e : flat_) for (const auto& kv : e.matTriangles) note(kv.first);
    for (const auto& g : legacy_) for (const auto& t : g.triangles) {
        if (!t) continue;
        uint16_t m = t->getMaterialID();
        note(m == MaterialManager::INVALID_MATERIAL_ID ? (uint16_t)0 : m);
    }
    std::unordered_set<const void*> seenSources;
    for (const auto& r : instances_) {
        if (!seenSources.insert(r.source.get()).second) continue;
        for (const auto& t : *r.source) {
            if (!t) continue;
            uint16_t m = t->getMaterialID();
            note(m == MaterialManager::INVALID_MATERIAL_ID ? (uint16_t)0 : m);
        }
    }

    std::sort(used.begin(), used.end());
    for (uint16_t id : used) materialIndexFor(id);
    // Material/image COUNTS are taken in run() after planning, not here: a
    // scatter source can reference a material no world.objects mesh used, and
    // materialIndexFor registers it lazily during planScatter().
}

// ---------------------------------------------------------------------------
// Phase 3: plan geometry
//
// Flat SoA meshes contribute ZERO heap allocation on the vertex path: the
// position/normal/uv arrays are planned as raw pointers into the geometry that
// already exists, and streamed straight out at write time.
// ---------------------------------------------------------------------------
// Phase 1b: scatter / foliage, read from InstanceManager rather than the scene
//
// ★ THIS IS NOT A DUPLICATE OF THE HittableInstance PATH ABOVE. On the Vulkan
// backends - the primary GPU path - SceneUI::syncInstancesToScene() returns
// early and NEVER expands scatter into world.objects; Vulkan RT and raster read
// InstanceGroup directly. So an exporter that only walks world.objects writes a
// perfectly valid file containing zero scatter, and nothing reports it. The
// HittableInstance branch still matters for projects loaded with the CPU-compat
// expansion; both are needed, and neither covers the other.
//
// Measured 2026-09-04: scatter.fill spawned 1000 instances, scene.export_gltf
// reported instances=0.
// ---------------------------------------------------------------------------
void GltfWriter::collectScatter() {
    if (!settings_.export_geometry) return;

    const auto& groups = InstanceManager::getInstance().getGroups();
    for (const InstanceGroup& group : groups) {
        // Transient groups are runtime particle render bridges, not authored
        // scene content; they are rebuilt every frame and must not be baked
        // into an interchange file.
        if (group.transient) continue;
        if (group.instances.empty() || group.sources.empty()) continue;

        for (size_t si = 0; si < group.sources.size(); ++si) {
            const ScatterSource& source = group.sources[si];

            // Collect this source's instance transforms once.
            std::vector<Matrix4x4> transforms;
            transforms.reserve(group.instances.size());
            for (const auto& inst : group.instances) {
                int idx = inst.source_index;
                if (idx < 0 || idx >= (int)group.sources.size()) idx = 0;
                if ((size_t)idx != si) continue;
                transforms.push_back(inst.toMatrix());
            }
            if (transforms.empty()) continue;

            if (!source.flat_meshes.empty()) {
                // Canonical flat sources: same transform composition the Vulkan
                // backends use, so the exported placement matches the render.
                // ★ ONE emission for the whole source. The siblings differ only by
                // material; they share the node transform, so they share the
                // sourceToScatter term as well.
                ScatterEmission em;
                em.name = group.name.empty() ? ("Scatter" + std::to_string(group.id)) : group.name;
                bool haveWorld = false;
                Matrix4x4 sourceWorld = Matrix4x4::identity();
                for (const auto& mesh : source.flat_meshes) {
                    if (!mesh || !mesh->geometry || mesh->geometry->indices.empty()) continue;
                    if (!haveWorld) {
                        sourceWorld = mesh->transform ? mesh->transform->getFinal()
                                                      : Matrix4x4::identity();
                        em.name += "_" + (source.name.empty() ? mesh->nodeName : source.name);
                        haveWorld = true;
                    }
                    em.flatMeshes.push_back(mesh.get());
                }
                if (em.flatMeshes.empty()) continue;
                em.sourceToScatter = Matrix4x4::translation(-source.mesh_center) * sourceWorld;
                em.transforms = transforms;
                scatter_.push_back(std::move(em));
            } else {
                // Legacy facade sources. centered_triangles_ptr already has the
                // source transform baked and the pivot removed, so the instance
                // matrix alone places it.
                const auto& ptr = source.centered_triangles_ptr;
                if (!ptr || ptr->empty()) continue;
                ScatterEmission em;
                em.name = group.name.empty() ? ("Scatter" + std::to_string(group.id)) : group.name;
                em.name += "_" + source.name;
                em.legacySource = ptr;
                em.transforms = std::move(transforms);
                scatter_.push_back(std::move(em));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Writes ONE flat SoA mesh (vertex buffers + one primitive per material) and
// returns its glTF mesh index. Split out of planGeometry so a scatter source
// and a scene object that are the SAME mesh share one copy of the vertex data.
bool GltfWriter::buildFlatPrimitives(const FlatEntry& e, json& primitives, bool& wantSkinOut) {
    {
        const DNA::GeometryDetail& g = *e.geom;

        // ★ These buffers are streamed to disk VERBATIM, so a pointer alone is
        // not enough: a regenerating surface (fluid/gas isosurface, particle
        // mesh) can advance vertex_count while leaving a SHORTER snapshot buffer
        // behind. Reading vertexCount elements off such a buffer overruns it, and
        // the damage would land silently in the file. Every attribute here is
        // therefore length-checked, and a short snapshot falls back to the live
        // buffer rather than being trusted.
        using DNA::Attr;
        const uint64_t need = e.vertexCount;
        auto usable = [&](Attr a) { return g.get_core_attribute_count(a) >= need; };

        const Vec3* P = nullptr;
        bool usedOrig = false;
        if (g.get_positions_orig() && usable(Attr::P_orig)) { P = g.get_positions_orig(); usedOrig = true; }
        else if (g.get_positions() && usable(Attr::P))       { P = g.get_positions(); }
        if (!P) {
            SCENE_LOG_ERROR("[Export] '" + e.mesh->nodeName + "' skipped: no position buffer holds "
                + std::to_string(need) + " vertices (P_orig=" + std::to_string(g.get_core_attribute_count(Attr::P_orig))
                + ", P=" + std::to_string(g.get_core_attribute_count(Attr::P)) + ").");
            return false;
        }
        if (!usedOrig && g.get_positions_orig()) {
            SCENE_LOG_INFO("[Export] '" + e.mesh->nodeName +
                "' has a stale P_orig snapshot shorter than vertex_count; exporting live positions instead.");
        }

        // Normals pair with whichever position snapshot was accepted; mixing a
        // bind-pose position buffer with live normals would shade wrong.
        const Vec3* N = nullptr;
        if (usedOrig && g.get_normals_orig() && usable(Attr::N_orig)) N = g.get_normals_orig();
        else if (g.get_normals() && usable(Attr::N))                  N = g.get_normals();

        const Vec2* UV = (g.get_uvs() && usable(Attr::UV)) ? g.get_uvs() : nullptr;

        // POSITION min/max is mandatory; reduce it on the triangle axis.
        float mn[3] = { P[0].x, P[0].y, P[0].z };
        float mx[3] = { P[0].x, P[0].y, P[0].z };
        {
            size_t threads = std::thread::hardware_concurrency();
            if (threads == 0) threads = 4;
            const std::array<float, 6> seed{ { mn[0], mn[1], mn[2], mx[0], mx[1], mx[2] } };
            std::vector<std::array<float, 6>> partial(threads, seed);
            parallelRange(e.vertexCount, 32768, [&](uint64_t b, uint64_t en, size_t t) {
                auto& acc = partial[(std::min)(t, threads - 1)];
                for (uint64_t i = b; i < en; ++i) {
                    const Vec3& p = P[i];
                    acc[0] = (std::min)(acc[0], p.x); acc[1] = (std::min)(acc[1], p.y); acc[2] = (std::min)(acc[2], p.z);
                    acc[3] = (std::max)(acc[3], p.x); acc[4] = (std::max)(acc[4], p.y); acc[5] = (std::max)(acc[5], p.z);
                }
            });
            for (const auto& acc : partial) {
                for (int k = 0; k < 3; ++k) {
                    mn[k] = (std::min)(mn[k], acc[k]);
                    mx[k] = (std::max)(mx[k], acc[3 + k]);
                }
            }
        }

        json attributes = json::object();

        const uint64_t posOffset = plan_.addRaw(P, e.vertexCount * sizeof(Vec3));
        const int posView = addBufferView(posOffset, e.vertexCount * sizeof(Vec3), kTargetArrayBuffer);
        const int posAcc = addAccessor(posView, kCompFloat, e.vertexCount, "VEC3");
        gltf_["accessors"][posAcc]["min"] = { mn[0], mn[1], mn[2] };
        gltf_["accessors"][posAcc]["max"] = { mx[0], mx[1], mx[2] };
        attributes["POSITION"] = posAcc;

        if (N) {
            const uint64_t off = plan_.addRaw(N, e.vertexCount * sizeof(Vec3));
            const int view = addBufferView(off, e.vertexCount * sizeof(Vec3), kTargetArrayBuffer);
            attributes["NORMAL"] = addAccessor(view, kCompFloat, e.vertexCount, "VEC3");
        }
        if (UV) {
            // Not addRaw: V has to be flipped back to glTF's upper-left origin.
            BinPiece up; up.kind = BinPiece::Kind::UVFlipped; up.geom = &g;
            up.bytes = e.vertexCount * sizeof(Vec2);
            const uint64_t off = plan_.add(std::move(up));
            const int view = addBufferView(off, e.vertexCount * sizeof(Vec2), kTargetArrayBuffer);
            attributes["TEXCOORD_0"] = addAccessor(view, kCompFloat, e.vertexCount, "VEC2");
        }

        const bool wantSkin = settings_.export_skinning && !g.skin_weights.empty() &&
                              scene_.boneData.getBoneCount() > 0;
        wantSkinOut = wantSkinOut || wantSkin;
        if (wantSkin) {
            BinPiece jp; jp.kind = BinPiece::Kind::SkinJoints; jp.geom = &g;
            jp.bytes = e.vertexCount * 4 * sizeof(uint16_t);
            const uint64_t jOff = plan_.add(std::move(jp));
            const int jView = addBufferView(jOff, e.vertexCount * 4 * sizeof(uint16_t), kTargetArrayBuffer);
            attributes["JOINTS_0"] = addAccessor(jView, kCompUShort, e.vertexCount, "VEC4");

            BinPiece wp; wp.kind = BinPiece::Kind::SkinWeights; wp.geom = &g;
            wp.bytes = e.vertexCount * 4 * sizeof(float);
            const uint64_t wOff = plan_.add(std::move(wp));
            const int wView = addBufferView(wOff, e.vertexCount * 4 * sizeof(float), kTargetArrayBuffer);
            attributes["WEIGHTS_0"] = addAccessor(wView, kCompFloat, e.vertexCount, "VEC4");
        }

        // One shared vertex set, one primitive per material. The Assimp path
        // duplicated the whole vertex array once per material batch.
        const size_t primitivesBefore = primitives.size();
        const bool singleMaterial = (e.matTriangles.size() == 1);
        for (const auto& kv : e.matTriangles) {
            const uint64_t indexCount = kv.second * 3;
            if (indexCount == 0) continue;

            BinPiece ip;
            ip.kind = singleMaterial ? BinPiece::Kind::MeshIndicesAll : BinPiece::Kind::MeshIndicesMat;
            ip.geom = &g;
            ip.material = kv.first;
            ip.bytes = indexCount * sizeof(uint32_t);
            const uint64_t off = plan_.add(std::move(ip));
            const int view = addBufferView(off, indexCount * sizeof(uint32_t), kTargetElementArrayBuffer);

            json prim = json::object();
            prim["attributes"] = attributes;
            prim["indices"] = addAccessor(view, kCompUInt, indexCount, "SCALAR");
            prim["mode"] = 4; // TRIANGLES
            prim["material"] = materialIndexFor(kv.first);
            primitives.push_back(std::move(prim));
            ++stats_.primitive_count;
        }
        if (primitives.size() == primitivesBefore) return false;

        stats_.vertex_count += e.vertexCount;
        stats_.triangle_count += e.triangleCount;
        return true;
    }
}

// ---------------------------------------------------------------------------
// Writes ONE glTF mesh from one or more sibling FlatEntries and returns its
// index. Every member is registered in flatMeshIndex_, so a mesh that is BOTH a
// scene object and a scatter source resolves to the same glTF mesh instead of a
// second copy of its vertex data.
int GltfWriter::emitFlatMeshGroup(const std::vector<const FlatEntry*>& entries,
                                  const std::string& meshName) {
    if (entries.empty()) return -1;
    {
        auto cached = flatMeshIndex_.find(entries.front()->mesh);
        if (cached != flatMeshIndex_.end()) return cached->second;
    }

    json primitives = json::array();
    bool wantSkin = false;
    for (const FlatEntry* e : entries) {
        if (!e) continue;
        buildFlatPrimitives(*e, primitives, wantSkin);
    }
    if (primitives.empty()) return -1;

    gltf_["meshes"].push_back({ {"name", meshName}, {"primitives", std::move(primitives)} });
    const int meshIndex = (int)gltf_["meshes"].size() - 1;
    for (const FlatEntry* e : entries) if (e) flatMeshIndex_[e->mesh] = meshIndex;
    if (wantSkin) skinnedMeshes_.insert(meshIndex);
    ++stats_.mesh_count;
    return meshIndex;
}

int GltfWriter::emitFlatMesh(const FlatEntry& e) {
    return emitFlatMeshGroup({ &e }, e.mesh->nodeName);
}

// ---------------------------------------------------------------------------
void GltfWriter::planGeometry() {
    // ★★★ SIBLINGS SHARING A nodeName ARE ONE OBJECT, SO THEY ARE ONE glTF NODE.
    //
    // glTF splits a mesh into one primitive per material, and this engine mirrors
    // that with one TriangleMesh per material — all carrying the SAME nodeName,
    // because a multi-material import is ONE logical object (AssimpLoader says so
    // inline; gatherScatterSource and the material panel both rely on it).
    //
    // Writing one node per TriangleMesh threw that grouping away: a 9-material
    // tree left as 9 same-named nodes, and reopening had to make those names
    // unique again — so the object came back split, its foliage layer multiplied
    // by the material count, and each fragment re-centred on its own bounding
    // box. The file was valid every time.
    //
    // The transform pointer is part of the key on purpose: two objects that
    // merely share a name but sit at different places must NOT be merged.
    struct MeshGroup {
        std::string name;
        Transform* transform = nullptr;
        std::vector<const FlatEntry*> entries;
    };
    std::vector<MeshGroup> groups;
    std::unordered_map<std::string, std::vector<size_t>> byName;

    for (const FlatEntry& e : flat_) {
        if (!e.mesh) continue;
        Transform* xf = e.mesh->transform.get();
        size_t slot = groups.size();   // sentinel: "no group yet"
        auto it = byName.find(e.mesh->nodeName);
        if (it != byName.end()) {
            for (size_t idx : it->second) {
                if (groups[idx].transform == xf) { slot = idx; break; }
            }
        }
        if (slot == groups.size()) {
            groups.push_back(MeshGroup{ e.mesh->nodeName, xf, {} });
            byName[e.mesh->nodeName].push_back(slot);
        }
        groups[slot].entries.push_back(&e);
    }

    for (const MeshGroup& grp : groups) {
        const int meshIndex = emitFlatMeshGroup(grp.entries, grp.name);
        if (meshIndex < 0) continue;

        json node = json::object();
        node["name"] = grp.name;
        node["mesh"] = meshIndex;
        if (grp.transform) node["matrix"] = toGltfMatrix(grp.transform->getFinal());
        const int nodeIndex = addNode(std::move(node), /*asSceneRoot*/ true);
        nodeNameToIndex_[grp.name] = nodeIndex;
        if (skinnedMeshes_.count(meshIndex)) skinnedNodes_.push_back(nodeIndex);
    }

    planScatter();

    // Legacy Triangle facade groups: still supported, but they carry no index
    // buffer of their own, so this is the only place a dedup hash is used.
    for (const LegacyGroup& grp : legacy_) {
        BuiltLegacyMesh built = buildLegacyMesh(grp.triangles);
        if (built.positions.empty()) continue;
        const uint64_t vcount = built.positions.size() / 3;

        json attributes = json::object();
        {
            auto bytes = floatsToBytes(built.positions);
            const uint64_t len = bytes.size();
            const uint64_t off = plan_.addOwned(std::move(bytes));
            const int view = addBufferView(off, len, kTargetArrayBuffer);
            const int acc = addAccessor(view, kCompFloat, vcount, "VEC3");
            gltf_["accessors"][acc]["min"] = { built.minPos[0], built.minPos[1], built.minPos[2] };
            gltf_["accessors"][acc]["max"] = { built.maxPos[0], built.maxPos[1], built.maxPos[2] };
            attributes["POSITION"] = acc;
        }
        {
            auto bytes = floatsToBytes(built.normals);
            const uint64_t len = bytes.size();
            const uint64_t off = plan_.addOwned(std::move(bytes));
            const int view = addBufferView(off, len, kTargetArrayBuffer);
            attributes["NORMAL"] = addAccessor(view, kCompFloat, vcount, "VEC3");
        }
        if (built.hasUV) {
            auto bytes = floatsToBytes(built.uvs);
            const uint64_t len = bytes.size();
            const uint64_t off = plan_.addOwned(std::move(bytes));
            const int view = addBufferView(off, len, kTargetArrayBuffer);
            attributes["TEXCOORD_0"] = addAccessor(view, kCompFloat, vcount, "VEC2");
        }

        json primitives = json::array();
        uint64_t triangles = 0;
        for (auto& kv : built.indicesByMaterial) {
            if (kv.second.empty()) continue;
            const uint64_t count = kv.second.size();
            auto bytes = uintsToBytes(kv.second);
            const uint64_t len = bytes.size();
            const uint64_t off = plan_.addOwned(std::move(bytes));
            const int view = addBufferView(off, len, kTargetElementArrayBuffer);

            json prim = json::object();
            prim["attributes"] = attributes;
            prim["indices"] = addAccessor(view, kCompUInt, count, "SCALAR");
            prim["mode"] = 4;
            prim["material"] = materialIndexFor(kv.first);
            primitives.push_back(std::move(prim));
            triangles += count / 3;
            ++stats_.primitive_count;
        }
        if (primitives.empty()) continue;

        gltf_["meshes"].push_back({ {"name", grp.name}, {"primitives", std::move(primitives)} });
        const int meshIndex = (int)gltf_["meshes"].size() - 1;

        json node = json::object();
        node["name"] = grp.name;
        node["mesh"] = meshIndex;
        node["matrix"] = toGltfMatrix(grp.transform);
        const int nodeIndex = addNode(std::move(node), true);
        nodeNameToIndex_[grp.name] = nodeIndex;

        stats_.vertex_count += vcount;
        stats_.triangle_count += triangles;
        ++stats_.mesh_count;
    }

    // Scattered/foliage instances: geometry built ONCE per unique source, then
    // one EXT_mesh_gpu_instancing node per source carrying every transform.
    if (!instances_.empty()) {
        std::vector<const void*> order;
        std::unordered_map<const void*, std::vector<const InstanceRequest*>> bySource;
        for (const auto& r : instances_) {
            auto& list = bySource[r.source.get()];
            if (list.empty()) order.push_back(r.source.get());
            list.push_back(&r);
        }

        for (const void* key : order) {
            const auto& requests = bySource[key];
            BuiltLegacyMesh built = buildLegacyMesh(*requests.front()->source);
            if (built.positions.empty()) continue;
            const uint64_t vcount = built.positions.size() / 3;

            json attributes = json::object();
            {
                auto bytes = floatsToBytes(built.positions);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int view = addBufferView(off, len, kTargetArrayBuffer);
                const int acc = addAccessor(view, kCompFloat, vcount, "VEC3");
                gltf_["accessors"][acc]["min"] = { built.minPos[0], built.minPos[1], built.minPos[2] };
                gltf_["accessors"][acc]["max"] = { built.maxPos[0], built.maxPos[1], built.maxPos[2] };
                attributes["POSITION"] = acc;
            }
            {
                auto bytes = floatsToBytes(built.normals);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int view = addBufferView(off, len, kTargetArrayBuffer);
                attributes["NORMAL"] = addAccessor(view, kCompFloat, vcount, "VEC3");
            }
            if (built.hasUV) {
                auto bytes = floatsToBytes(built.uvs);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int view = addBufferView(off, len, kTargetArrayBuffer);
                attributes["TEXCOORD_0"] = addAccessor(view, kCompFloat, vcount, "VEC2");
            }

            json primitives = json::array();
            uint64_t triangles = 0;
            for (auto& kv : built.indicesByMaterial) {
                if (kv.second.empty()) continue;
                const uint64_t count = kv.second.size();
                auto bytes = uintsToBytes(kv.second);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int view = addBufferView(off, len, kTargetElementArrayBuffer);
                json prim = json::object();
                prim["attributes"] = attributes;
                prim["indices"] = addAccessor(view, kCompUInt, count, "SCALAR");
                prim["mode"] = 4;
                prim["material"] = materialIndexFor(kv.first);
                primitives.push_back(std::move(prim));
                triangles += count / 3;
                ++stats_.primitive_count;
            }
            if (primitives.empty()) continue;

            const std::string sourceName = requests.front()->node_name;
            gltf_["meshes"].push_back({ {"name", sourceName + "_Source"}, {"primitives", std::move(primitives)} });
            const int meshIndex = (int)gltf_["meshes"].size() - 1;
            stats_.vertex_count += vcount;
            stats_.triangle_count += triangles;
            ++stats_.mesh_count;
            stats_.instance_count += requests.size();

            if (settings_.use_gpu_instancing_extension && requests.size() > 1) {
                std::vector<float> T, R, S;
                T.reserve(requests.size() * 3);
                R.reserve(requests.size() * 4);
                S.reserve(requests.size() * 3);
                for (const InstanceRequest* r : requests) {
                    const std::vector<float> m = toGltfMatrix(r->transform);
                    float t[3], q[4], s[3];
                    decomposeGltfMatrix(m.data(), t, q, s);
                    T.insert(T.end(), t, t + 3);
                    R.insert(R.end(), q, q + 4);
                    S.insert(S.end(), s, s + 3);
                }
                auto addFloatAcc = [&](const std::vector<float>& v, const char* type, uint64_t count) {
                    auto bytes = floatsToBytes(v);
                    const uint64_t len = bytes.size();
                    const uint64_t off = plan_.addOwned(std::move(bytes));
                    return addAccessor(addBufferView(off, len, 0), kCompFloat, count, type);
                };
                const int tAcc = addFloatAcc(T, "VEC3", requests.size());
                const int rAcc = addFloatAcc(R, "VEC4", requests.size());
                const int sAcc = addFloatAcc(S, "VEC3", requests.size());

                useExtension("EXT_mesh_gpu_instancing");
                json node = json::object();
                node["name"] = sourceName + "_Instances";
                node["mesh"] = meshIndex;
                node["extensions"]["EXT_mesh_gpu_instancing"]["attributes"] = {
                    {"TRANSLATION", tAcc}, {"ROTATION", rAcc}, {"SCALE", sAcc}
                };
                addNode(std::move(node), true);
                ++stats_.instanced_group_count;
            } else {
                for (const InstanceRequest* r : requests) {
                    json node = json::object();
                    node["name"] = r->node_name;
                    node["mesh"] = meshIndex;
                    node["matrix"] = toGltfMatrix(r->transform);
                    addNode(std::move(node), true);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scatter emission: one glTF mesh per unique source (shared with the scene
// object when it is the same mesh) plus one EXT_mesh_gpu_instancing node
// carrying every placement.
// ---------------------------------------------------------------------------
void GltfWriter::planScatter() {
    for (ScatterEmission& em : scatter_) {
        int meshIndex = -1;

        if (!em.flatMeshes.empty()) {
            // Own the entries: emitFlatMeshGroup takes pointers into this vector.
            std::vector<FlatEntry> entries;
            entries.reserve(em.flatMeshes.size());
            for (const TriangleMesh* m : em.flatMeshes) {
                FlatEntry fe;
                fe.mesh = m;
                fe.geom = m->geometry.get();
                fe.vertexCount = fe.geom->get_vertex_count();
                fe.triangleCount = fe.geom->indices.size() / 3;
                if (fe.vertexCount == 0 || fe.triangleCount == 0) continue;
                histogramFor(fe);
                entries.push_back(std::move(fe));
            }
            if (entries.empty()) continue;
            std::vector<const FlatEntry*> ptrs;
            ptrs.reserve(entries.size());
            for (const FlatEntry& fe : entries) ptrs.push_back(&fe);
            // Hits flatMeshIndex_ when the prototype was already written as a
            // scene object, so the vertex data is written once, not twice.
            meshIndex = emitFlatMeshGroup(ptrs, em.name + "_Source");
        } else if (em.legacySource) {
            BuiltLegacyMesh built = buildLegacyMesh(*em.legacySource);
            if (built.positions.empty()) continue;
            const uint64_t vcount = built.positions.size() / 3;

            json attributes = json::object();
            {
                auto bytes = floatsToBytes(built.positions);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int acc = addAccessor(addBufferView(off, len, kTargetArrayBuffer),
                                            kCompFloat, vcount, "VEC3");
                gltf_["accessors"][acc]["min"] = { built.minPos[0], built.minPos[1], built.minPos[2] };
                gltf_["accessors"][acc]["max"] = { built.maxPos[0], built.maxPos[1], built.maxPos[2] };
                attributes["POSITION"] = acc;
            }
            {
                auto bytes = floatsToBytes(built.normals);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                attributes["NORMAL"] = addAccessor(addBufferView(off, len, kTargetArrayBuffer),
                                                   kCompFloat, vcount, "VEC3");
            }
            if (built.hasUV) {
                auto bytes = floatsToBytes(built.uvs);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                attributes["TEXCOORD_0"] = addAccessor(addBufferView(off, len, kTargetArrayBuffer),
                                                       kCompFloat, vcount, "VEC2");
            }

            json primitives = json::array();
            uint64_t triangles = 0;
            for (auto& kv : built.indicesByMaterial) {
                if (kv.second.empty()) continue;
                const uint64_t count = kv.second.size();
                auto bytes = uintsToBytes(kv.second);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                json prim = json::object();
                prim["attributes"] = attributes;
                prim["indices"] = addAccessor(addBufferView(off, len, kTargetElementArrayBuffer),
                                              kCompUInt, count, "SCALAR");
                prim["mode"] = 4;
                prim["material"] = materialIndexFor(kv.first);
                primitives.push_back(std::move(prim));
                triangles += count / 3;
                ++stats_.primitive_count;
            }
            if (primitives.empty()) continue;

            gltf_["meshes"].push_back({ {"name", em.name + "_Source"}, {"primitives", std::move(primitives)} });
            meshIndex = (int)gltf_["meshes"].size() - 1;
            stats_.vertex_count += vcount;
            stats_.triangle_count += triangles;
            ++stats_.mesh_count;
        }

        if (meshIndex < 0) continue;
        stats_.instance_count += em.transforms.size();

        if (settings_.use_gpu_instancing_extension && em.transforms.size() > 1) {
            std::vector<float> T, R, S;
            T.reserve(em.transforms.size() * 3);
            R.reserve(em.transforms.size() * 4);
            S.reserve(em.transforms.size() * 3);
            for (const Matrix4x4& t : em.transforms) {
                const std::vector<float> m = toGltfMatrix(t * em.sourceToScatter);
                float tr[3], q[4], sc[3];
                decomposeGltfMatrix(m.data(), tr, q, sc);
                T.insert(T.end(), tr, tr + 3);
                R.insert(R.end(), q, q + 4);
                S.insert(S.end(), sc, sc + 3);
            }
            auto addFloatAcc = [&](const std::vector<float>& v, const char* type) {
                auto bytes = floatsToBytes(v);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                return addAccessor(addBufferView(off, len, 0), kCompFloat, em.transforms.size(), type);
            };
            const int tAcc = addFloatAcc(T, "VEC3");
            const int rAcc = addFloatAcc(R, "VEC4");
            const int sAcc = addFloatAcc(S, "VEC3");

            useExtension("EXT_mesh_gpu_instancing");
            json node = json::object();
            node["name"] = em.name + "_Instances";
            node["mesh"] = meshIndex;
            node["extensions"]["EXT_mesh_gpu_instancing"]["attributes"] = {
                {"TRANSLATION", tAcc}, {"ROTATION", rAcc}, {"SCALE", sAcc}
            };
            addNode(std::move(node), true);
            ++stats_.instanced_group_count;
        } else {
            for (size_t i = 0; i < em.transforms.size(); ++i) {
                json node = json::object();
                node["name"] = em.name + "_" + std::to_string(i);
                node["mesh"] = meshIndex;
                node["matrix"] = toGltfMatrix(em.transforms[i] * em.sourceToScatter);
                addNode(std::move(node), true);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 4: skeleton, skin, animations
// ---------------------------------------------------------------------------
void GltfWriter::planSkinsAndAnimations() {
    const BoneData& bd = scene_.boneData;
    const bool wantSkin = settings_.export_skinning && bd.getBoneCount() > 0;

    std::unordered_map<std::string, int> boneNodeIndex;

    if (wantSkin) {
        // Joint order MUST match the bone indices baked into skin_weights, or
        // every vertex binds to the wrong bone - a failure that renders as a
        // plausible-looking but wrong pose, not as an error.
        size_t maxIndex = 0;
        for (const auto& kv : bd.boneNameToIndex) maxIndex = (std::max)(maxIndex, (size_t)kv.second);
        std::vector<std::string> jointOrder(maxIndex + 1);
        for (const auto& kv : bd.boneNameToIndex) jointOrder[kv.second] = kv.first;

        for (const std::string& name : jointOrder) {
            if (name.empty()) continue;
            json node = json::object();
            node["name"] = name;
            auto lt = bd.boneDefaultTransforms.find(name);
            if (lt != bd.boneDefaultTransforms.end()) node["matrix"] = toGltfMatrix(lt->second);
            gltf_["nodes"].push_back(std::move(node));
            const int idx = (int)gltf_["nodes"].size() - 1;
            boneNodeIndex[name] = idx;
            // Bone names win the animation-target lookup only if no object node
            // already claimed the name (object nodes are registered first).
            nodeNameToIndex_.emplace(name, idx);
        }

        // Hierarchy from VALUES (boneParents), not from the import-time aiNode
        // pointers the old exporter walked - those outlive nothing reliably.
        std::vector<int> roots;
        for (const auto& kv : boneNodeIndex) {
            auto p = bd.boneParents.find(kv.first);
            if (p != bd.boneParents.end()) {
                auto parentNode = boneNodeIndex.find(p->second);
                if (parentNode != boneNodeIndex.end()) {
                    gltf_["nodes"][parentNode->second]["children"].push_back(kv.second);
                    continue;
                }
            }
            roots.push_back(kv.second);
        }
        std::sort(roots.begin(), roots.end());
        for (int r : roots) gltf_["scenes"][0]["nodes"].push_back(r);

        if (!skinnedNodes_.empty()) {
            std::vector<int> joints;
            std::vector<float> ibm;
            joints.reserve(jointOrder.size());
            ibm.reserve(jointOrder.size() * 16);
            for (const std::string& name : jointOrder) {
                if (name.empty()) continue;
                joints.push_back(boneNodeIndex[name]);
                auto off = bd.boneOffsetMatrices.find(name);
                const std::vector<float> m = (off != bd.boneOffsetMatrices.end())
                    ? toGltfMatrix(off->second)
                    : toGltfMatrix(Matrix4x4::identity());
                ibm.insert(ibm.end(), m.begin(), m.end());
            }

            if (!joints.empty()) {
                auto bytes = floatsToBytes(ibm);
                const uint64_t len = bytes.size();
                const uint64_t off = plan_.addOwned(std::move(bytes));
                const int view = addBufferView(off, len, 0);
                const int acc = addAccessor(view, kCompFloat, joints.size(), "MAT4");

                json skin = json::object();
                skin["joints"] = joints;
                skin["inverseBindMatrices"] = acc;
                if (!roots.empty()) skin["skeleton"] = roots.front();
                gltf_["skins"].push_back(std::move(skin));
                const int skinIndex = (int)gltf_["skins"].size() - 1;
                for (int n : skinnedNodes_) gltf_["nodes"][n]["skin"] = skinIndex;
            }
        }
    }

    if (!settings_.export_animations || scene_.animationDataList.empty()) return;

    for (const auto& src : scene_.animationDataList) {
        if (!src) continue;
        const double tps = (src->ticksPerSecond > 0.0) ? src->ticksPerSecond : 25.0;

        json samplers = json::array();
        json channels = json::array();

        std::vector<std::string> animatedNodes;
        {
            std::unordered_set<std::string> seen;
            auto note = [&](const std::string& n) { if (seen.insert(n).second) animatedNodes.push_back(n); };
            for (const auto& kv : src->positionKeys) note(kv.first);
            for (const auto& kv : src->rotationKeys) note(kv.first);
            for (const auto& kv : src->scalingKeys)  note(kv.first);
        }

        for (const std::string& nodeName : animatedNodes) {
            auto nit = nodeNameToIndex_.find(nodeName);
            if (nit == nodeNameToIndex_.end()) continue; // nothing in the file to drive
            const int targetNode = nit->second;

            auto addSampler = [&](const std::vector<float>& times,
                                  const std::vector<float>& values,
                                  const char* valueType, uint64_t count) -> int {
                auto tBytes = floatsToBytes(times);
                const uint64_t tLen = tBytes.size();
                const uint64_t tOff = plan_.addOwned(std::move(tBytes));
                const int tView = addBufferView(tOff, tLen, 0);
                const int tAcc = addAccessor(tView, kCompFloat, count, "SCALAR");
                // An animation input accessor is required to carry min/max.
                gltf_["accessors"][tAcc]["min"] = { times.empty() ? 0.0f : times.front() };
                gltf_["accessors"][tAcc]["max"] = { times.empty() ? 0.0f : times.back() };

                auto vBytes = floatsToBytes(values);
                const uint64_t vLen = vBytes.size();
                const uint64_t vOff = plan_.addOwned(std::move(vBytes));
                const int vView = addBufferView(vOff, vLen, 0);
                const int vAcc = addAccessor(vView, kCompFloat, count, valueType);

                samplers.push_back({ {"input", tAcc}, {"output", vAcc}, {"interpolation", "LINEAR"} });
                return (int)samplers.size() - 1;
            };

            auto pit = src->positionKeys.find(nodeName);
            if (pit != src->positionKeys.end() && !pit->second.empty()) {
                std::vector<float> times, values;
                times.reserve(pit->second.size());
                values.reserve(pit->second.size() * 3);
                for (const auto& k : pit->second) {
                    times.push_back((float)(k.time / tps));
                    values.insert(values.end(), { k.value.x, k.value.y, k.value.z });
                }
                const int s = addSampler(times, values, "VEC3", pit->second.size());
                channels.push_back({ {"sampler", s}, {"target", {{"node", targetNode}, {"path", "translation"}}} });
            }

            auto rit = src->rotationKeys.find(nodeName);
            if (rit != src->rotationKeys.end() && !rit->second.empty()) {
                std::vector<float> times, values;
                times.reserve(rit->second.size());
                values.reserve(rit->second.size() * 4);
                for (const auto& k : rit->second) {
                    times.push_back((float)(k.time / tps));
                    // glTF stores quaternions xyzw; RayTrophi's Quaternion is wxyz.
                    values.insert(values.end(), { k.value.x, k.value.y, k.value.z, k.value.w });
                }
                const int s = addSampler(times, values, "VEC4", rit->second.size());
                channels.push_back({ {"sampler", s}, {"target", {{"node", targetNode}, {"path", "rotation"}}} });
            }

            auto sit = src->scalingKeys.find(nodeName);
            if (sit != src->scalingKeys.end() && !sit->second.empty()) {
                std::vector<float> times, values;
                times.reserve(sit->second.size());
                values.reserve(sit->second.size() * 3);
                for (const auto& k : sit->second) {
                    times.push_back((float)(k.time / tps));
                    values.insert(values.end(), { k.value.x, k.value.y, k.value.z });
                }
                const int s = addSampler(times, values, "VEC3", sit->second.size());
                channels.push_back({ {"sampler", s}, {"target", {{"node", targetNode}, {"path", "scale"}}} });
            }
        }

        if (channels.empty()) continue;
        gltf_["animations"].push_back({ {"name", src->name},
                                        {"samplers", std::move(samplers)},
                                        {"channels", std::move(channels)} });
    }
}

// ---------------------------------------------------------------------------
// Phase 5: cameras and punctual lights
// ---------------------------------------------------------------------------
void GltfWriter::planCamerasAndLights() {
    if (settings_.export_cameras && scene_.camera) {
        const auto& cam = scene_.camera;
        gltf_["cameras"].push_back({
            {"name", "MainCamera"},
            {"type", "perspective"},
            {"perspective", {
                {"yfov", degToRadf(cam->vfov)},
                {"znear", (std::max)(0.0001f, cam->near_dist)},
                {"zfar", (std::max)(cam->near_dist + 0.0001f, cam->far_dist)},
                {"aspectRatio", (std::max)(0.0001f, cam->aspect_ratio)}
            }}
        });

        Vec3 forward = cam->lookat - cam->lookfrom;
        Vec3 up = cam->vup;
        if (forward.length_squared() < 1e-8f) forward = Vec3(0.0f, 0.0f, -1.0f);
        forward = forward.normalize();
        if (up.length_squared() < 1e-8f) up = Vec3(0.0f, 1.0f, 0.0f);
        up = up.normalize();
        if (std::abs(Vec3::dot(forward, up)) > 0.99f)
            up = (std::abs(forward.y) > 0.99f) ? Vec3(0, 0, 1) : Vec3(0, 1, 0);
        const Vec3 right = Vec3::cross(forward, up).normalize();
        const Vec3 trueUp = Vec3::cross(right, forward).normalize();

        json node = json::object();
        node["name"] = "MainCameraNode";
        node["camera"] = (int)gltf_["cameras"].size() - 1;
        node["matrix"] = std::vector<float>{
            right.x, right.y, right.z, 0.0f,
            trueUp.x, trueUp.y, trueUp.z, 0.0f,
            -forward.x, -forward.y, -forward.z, 0.0f,
            cam->lookfrom.x, cam->lookfrom.y, cam->lookfrom.z, 1.0f
        };
        addNode(std::move(node), true);
    }

    if (!settings_.export_lights || scene_.lights.empty()) return;

    json lightDefs = json::array();
    for (const auto& light : scene_.lights) {
        if (!light) continue;

        json def = {
            {"name", light->nodeName.empty() ? "Light" : light->nodeName},
            {"color", { light->color.x, light->color.y, light->color.z }},
            {"intensity", (std::max)(0.0f, light->intensity)}
        };

        Vec3 dir = light->direction;
        if (light->type() == LightType::Point) {
            def["type"] = "point";
        } else if (light->type() == LightType::Directional) {
            def["type"] = "directional";
        } else if (light->type() == LightType::Spot) {
            def["type"] = "spot";
            auto spot = std::dynamic_pointer_cast<SpotLight>(light);
            const float outer = degToRadf(spot ? spot->getAngleDegrees() * 0.5f : 22.5f);
            const float inner = outer * (1.0f - clamp01f(spot ? spot->getFalloff() : 0.1f));
            def["spot"] = { {"innerConeAngle", (std::max)(0.0f, inner)},
                            {"outerConeAngle", (std::max)(inner, outer)} };
        } else {
            continue;
        }

        lightDefs.push_back(std::move(def));
        const int lightIndex = (int)lightDefs.size() - 1;

        json node = json::object();
        node["name"] = light->nodeName.empty() ? ("Light_" + std::to_string(lightIndex)) : light->nodeName;
        node["extensions"]["KHR_lights_punctual"]["light"] = lightIndex;

        if (light->type() == LightType::Point) {
            node["translation"] = { light->position.x, light->position.y, light->position.z };
        } else {
            if (dir.length_squared() < 1e-8f) dir = Vec3(0.0f, -1.0f, 0.0f);
            dir = dir.normalize();
            Vec3 up(0.0f, 1.0f, 0.0f);
            if (std::abs(Vec3::dot(dir, up)) > 0.99f) up = Vec3(0.0f, 0.0f, 1.0f);
            const Vec3 right = Vec3::cross(dir, up).normalize();
            const Vec3 trueUp = Vec3::cross(right, dir).normalize();
            node["matrix"] = std::vector<float>{
                right.x, right.y, right.z, 0.0f,
                trueUp.x, trueUp.y, trueUp.z, 0.0f,
                -dir.x, -dir.y, -dir.z, 0.0f,
                light->position.x, light->position.y, light->position.z, 1.0f
            };
        }
        addNode(std::move(node), true);
    }

    if (!lightDefs.empty()) {
        useExtension("KHR_lights_punctual");
        gltf_["extensions"]["KHR_lights_punctual"]["lights"] = std::move(lightDefs);
    }
}

// ---------------------------------------------------------------------------
// Phase 6: emit
//
// Everything above only PLANNED bytes. Here the JSON is already final (all
// offsets known), so the file is produced in a single forward pass with no
// temp file, no concatenation, and no full-payload buffer.
// ---------------------------------------------------------------------------
bool GltfWriter::emit(const std::string& filepath, std::string& error) {
    const std::filesystem::path outPath(filepath);
    std::string ext = outPath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    const bool binaryMode = (ext != ".gltf");

    const uint64_t binBytes = plan_.totalBytes();
    const uint64_t binPadded = binBytes + ((4ull - (binBytes % 4ull)) % 4ull);

    // Strip empty arrays: a "materials": [] is invalid glTF, not merely untidy.
    static const char* kOptionalArrays[] = {
        "meshes", "accessors", "bufferViews", "materials", "images", "samplers",
        "textures", "skins", "animations", "cameras", "nodes", "extensionsUsed"
    };
    for (const char* key : kOptionalArrays) {
        if (gltf_.contains(key) && gltf_[key].is_array() && gltf_[key].empty()) gltf_.erase(key);
    }
    if (gltf_.contains("extensions") && gltf_["extensions"].is_object() && gltf_["extensions"].empty())
        gltf_.erase("extensions");

    std::string binName;
    if (binPadded > 0) {
        json buffer = json::object();
        buffer["byteLength"] = binPadded;
        if (!binaryMode) {
            binName = outPath.stem().string() + ".bin";
            buffer["uri"] = binName;
        }
        gltf_["buffers"] = json::array({ std::move(buffer) });
    }

    std::string jsonText = gltf_.dump();
    stats_.json_bytes = jsonText.size();

    auto writePieces = [&](std::ostream& out) -> bool {
        for (const BinPiece& piece : plan_.pieces()) {
            if (!writePiece(out, piece)) return false;
        }
        for (uint64_t i = binBytes; i < binPadded; ++i) out.put('\0');
        return out.good();
    };

    if (!binaryMode) {
        // Text mode writes a sidecar .bin. The previous exporter base64-embedded
        // it, which is ~33% larger AND has to exist in RAM as one string.
        if (binPadded > 0) {
            const std::filesystem::path binPath = outPath.parent_path() / binName;
            std::ofstream binOut(binPath, std::ios::binary | std::ios::trunc);
            if (!binOut) { error = "Cannot open sidecar buffer for writing: " + binPath.string(); return false; }
            if (!writePieces(binOut)) { error = "Failed while streaming buffer to " + binPath.string(); return false; }
            binOut.close();
        }
        std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
        if (!out) { error = "Cannot open " + filepath + " for writing"; return false; }
        out.write(jsonText.data(), (std::streamsize)jsonText.size());
        if (!out.good()) { error = "Failed while writing glTF JSON"; return false; }
        stats_.bin_bytes = binPadded;
        stats_.file_bytes = jsonText.size();
        return true;
    }

    // GLB: JSON chunk is space-padded (spec), BIN chunk zero-padded.
    while ((jsonText.size() % 4u) != 0u) jsonText.push_back(' ');

    const uint64_t total = 12ull + 8ull + jsonText.size() + (binPadded > 0 ? 8ull + binPadded : 0ull);
    if (total > kGlbMaxTotalBytes) {
        error = "Scene needs " + std::to_string(total / (1024ull * 1024ull)) +
                " MB, which exceeds the 4 GB ceiling a GLB's 32-bit chunk length can address. "
                "Export as .gltf instead - that writes a sidecar .bin with no size limit.";
        return false;
    }

    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out) { error = "Cannot open " + filepath + " for writing"; return false; }

    auto writeU32 = [&](uint32_t v) {
        const uint8_t b[4] = { (uint8_t)(v & 0xFFu), (uint8_t)((v >> 8) & 0xFFu),
                               (uint8_t)((v >> 16) & 0xFFu), (uint8_t)((v >> 24) & 0xFFu) };
        out.write(reinterpret_cast<const char*>(b), 4);
    };

    writeU32(kGlbMagic);
    writeU32(kGlbVersion);
    writeU32((uint32_t)total);
    writeU32((uint32_t)jsonText.size());
    writeU32(kGlbJsonChunk);
    out.write(jsonText.data(), (std::streamsize)jsonText.size());
    if (!out.good()) { error = "Failed while writing GLB JSON chunk"; return false; }

    if (binPadded > 0) {
        writeU32((uint32_t)binPadded);
        writeU32(kGlbBinChunk);
        if (!writePieces(out)) { error = "Failed while streaming GLB binary chunk"; return false; }
    }

    out.flush();
    if (!out.good()) { error = "Stream error finalising " + filepath; return false; }
    stats_.bin_bytes = binPadded;
    stats_.file_bytes = total;
    return true;
}

bool GltfWriter::run(const std::string& filepath,
                     const std::vector<std::shared_ptr<Hittable>>& selected,
                     WriteStats& stats, std::string& error) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    gltf_["asset"] = { {"version", "2.0"}, {"generator", "RayTrophi Studio"} };
    gltf_["scene"] = 0;
    gltf_["scenes"] = json::array({ json::object({ {"nodes", json::array()} }) });
    gltf_["nodes"] = json::array();
    gltf_["meshes"] = json::array();
    gltf_["accessors"] = json::array();
    gltf_["bufferViews"] = json::array();
    gltf_["materials"] = json::array();
    gltf_["images"] = json::array();
    gltf_["samplers"] = json::array();
    gltf_["textures"] = json::array();
    gltf_["skins"] = json::array();
    gltf_["animations"] = json::array();
    gltf_["cameras"] = json::array();
    gltf_["extensionsUsed"] = json::array();

    collect(selected);
    collectScatter();
    const auto t1 = clock::now();

    buildMaterials();
    const auto t2 = clock::now();

    planGeometry();
    planSkinsAndAnimations();
    planCamerasAndLights();
    const auto t3 = clock::now();

    // Counted AFTER planning: scatter sources can introduce materials/images
    // that no world.objects mesh referenced.
    stats_.node_count = gltf_["nodes"].size();
    stats_.material_count = gltf_["materials"].size();
    stats_.image_count = gltf_["images"].size();

    if (stats_.mesh_count == 0 && gltf_["nodes"].empty()) {
        error = "Scene is empty: nothing to export.";
        stats = stats_;
        return false;
    }

    const bool ok = emit(filepath, error);
    const auto t4 = clock::now();

    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    stats_.seconds_collect = secs(t0, t1);
    stats_.seconds_materials = secs(t1, t2);
    stats_.seconds_plan = secs(t2, t3);
    stats_.seconds_write = secs(t3, t4);
    stats_.seconds_total = secs(t0, t4);
    // Honest number: bytes THIS writer allocated, not process RSS. Flat SoA
    // geometry contributes zero here - that is the whole point.
    stats_.peak_writer_mb = (double)plan_.ownedBytes() / (1024.0 * 1024.0);

    stats = stats_;
    return ok;
}

} // namespace

bool writeScene(const std::string& filepath,
                SceneData& scene,
                const ExportSettings& settings,
                const std::vector<std::shared_ptr<Hittable>>& selected_objects,
                const std::map<uint16_t, std::shared_ptr<Material>>& materialOverrides,
                WriteStats& stats,
                std::string& error) {
    try {
        GltfWriter writer(scene, settings, materialOverrides);
        return writer.run(filepath, selected_objects, stats, error);
    } catch (const std::exception& ex) {
        error = std::string("glTF writer threw: ") + ex.what();
        return false;
    } catch (...) {
        error = "glTF writer threw an unknown exception";
        return false;
    }
}

} // namespace rtgltf
