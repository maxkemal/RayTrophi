// GPU wet paint atlas: bake, device buffers, per-frame stamp/tick, publish.
// See include/Paint/WetPaintGpu.h and shaders/paint_wet.comp.

#include "Paint/WetPaintGpu.h"

#include "Backend/IBackend.h"
#include "Backend/VulkanBackend.h"
#include "Paint/MeshPaintAdapter.h"
#include "Paint/PaintHeightNormalMath.h"
#include "PrincipledBSDF.h"
#include "ParticleSimulation.h"
#include "SimulationComputeVulkanContext.h"
#include "SimulationWorld.h"
#include "SurfaceFlowField.h"
#include "Texture.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>

namespace Paint {

using RayTrophiSim::ComputeBackendType;
using RayTrophiSim::ComputeBufferDesc;
using RayTrophiSim::ComputeBufferHandle;
using RayTrophiSim::ComputeBufferUsage;
using RayTrophiSim::ComputeDispatch;
using RayTrophiSim::SimulationComputeContext;

namespace {

constexpr uint32_t kTile = 32u;
constexpr uint32_t kStatsStride = 16u;
constexpr uint32_t kBindingCount = 17u;
constexpr int kGutterTexels = 2;

enum WetPaintMode : uint32_t {
    kModeStamp = 0, kModeTick = 1, kModeResolve = 2, kModeRotate = 3,
    kModeConsume = 4, kModeCoverage = 5, kModeReset = 6,
    kModeChannels = 7, kModeSeedHeight = 8,
};
enum WetPaintSource : uint32_t { kSourceDabs = 0, kSourceEvents = 1 };
enum StatsSlot : uint32_t {
    kStActiveTiles = 0, kStDirtyMinX = 1, kStDirtyMinY = 2, kStDirtyMaxX = 3, kStDirtyMaxY = 4,
    kStStampedTexels = 5, kStTickedTiles = 6, kStSourcesDropped = 7, kStCoveredTexels = 8,
    kStValidTexels = 9, kStSumR = 10, kStSumG = 11, kStSumB = 12, kStWetTexels = 13,
    kStRoughnessTexels = 14, kStNormalTexels = 15,
};

// Mirrored field for field in paint_wet.comp (std430).
struct GpuSurface {
    uint32_t width = 0, height = 0, page_offset = 0, tile_offset = 0;
    uint32_t tiles_x = 0, tiles_y = 0, flags = 0, _pad0 = 0;
    float diffusion = 0, runoff = 0, absorption = 0, drip_head = 0;
    float lifetime = 1, brush_radius_px = 1, falloff = 0.65f, paint_load = 0.85f;
    float wetness = 0.65f, deposit_rate = 0.65f, pickup_rate = 0.35f, _pad2 = 0;
};
static_assert(sizeof(GpuSurface) == 80, "GpuSurface must match Surface in paint_wet.comp");

struct GpuTile {
    uint32_t surface = 0, tx = 0, ty = 0, valid_texels = 0;
    float bmin[4] = {0, 0, 0, 0};
    float bmax[4] = {0, 0, 0, 0};
};
static_assert(sizeof(GpuTile) == 48, "GpuTile must match Tile in paint_wet.comp");

struct GpuDab {
    float pos_radius[4];
    float color_amount[4];
    float normal_facing[4];
};
static_assert(sizeof(GpuDab) == 48, "GpuDab must match Dab in paint_wet.comp");

// Mirrored field for field in paint_wet.comp; kernel table: 64.
struct WetPaintGpuConstants {
    uint32_t mode = 0, surface_count = 0, tile_count = 0, source_kind = 0;
    uint32_t source_count = 0, event_capacity = 0, max_events = 0, lookup_count = 0;
    float dt = 0, splat_radius = 0, splat_amount = 0, splat_min_speed = 0;
    float facing_min = 0;
    uint32_t lut_rows = 0, _p0 = 0, _p1 = 0;
};
static_assert(sizeof(WetPaintGpuConstants) == 64,
              "paint_wet push constants: shader, kernel table (64) and this struct change together");

uint32_t packRgba8(const CompactVec4& c) {
    return static_cast<uint32_t>(c.r) | (static_cast<uint32_t>(c.g) << 8) |
           (static_cast<uint32_t>(c.b) << 16) | (static_cast<uint32_t>(c.a) << 24);
}

// Octahedral normal, packSnorm2x16 into a float's bits (octDecode in the shader).
float packOctNormal(Vec3 n) {
    const float len = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    if (len <= 1e-12f) n = Vec3(0.0f, 1.0f, 0.0f);
    else n = Vec3(n.x / len, n.y / len, n.z / len);
    float ex = n.x, ey = n.y;
    if (n.z < 0.0f) {
        ex = (1.0f - std::abs(n.y)) * (n.x >= 0.0f ? 1.0f : -1.0f);
        ey = (1.0f - std::abs(n.x)) * (n.y >= 0.0f ? 1.0f : -1.0f);
    }
    const auto snorm = [](float v) -> uint32_t {
        const int i = static_cast<int>(std::round(std::clamp(v, -1.0f, 1.0f) * 32767.0f));
        return static_cast<uint32_t>(static_cast<uint16_t>(static_cast<int16_t>(i)));
    };
    const uint32_t bits = snorm(ex) | (snorm(ey) << 16);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float linearToSrgb01(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    return v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

std::shared_ptr<Triangle> findPaintTriangle(SceneData& scene, const std::string& object,
                                            int material_id) {
    for (const auto& obj : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (!mesh || mesh->nodeName != object || !mesh->geometry) {
            continue;
        }
        const auto& geometry = *mesh->geometry;
        const auto* materials = geometry.get_attribute_data<uint16_t>("materialID");
        for (size_t face = 0; face < mesh->num_triangles(); ++face) {
            const uint32_t vertex = geometry.indices[face * 3];
            if (vertex >= geometry.get_vertex_count()) {
                continue;
            }
            const uint16_t material = materials ? materials[vertex] : 0;
            if (material_id < 0 || material == static_cast<uint16_t>(material_id)) {
                // One compatibility adapter for material/texture authoring only.
                // Geometry below is read directly from the canonical DNA arrays.
                return std::make_shared<Triangle>(mesh, static_cast<uint32_t>(face));
            }
        }
    }
    return nullptr;
}

struct FlatPaintFace {
    std::array<Vec3, 3> positions;
    std::array<Vec3, 3> normals;
    std::array<Vec2, 3> uvs;
};

std::vector<FlatPaintFace> paintFaces(SceneData& scene, const std::string& object,
                                     uint16_t material_id, int uv_set) {
    std::vector<FlatPaintFace> faces;
    for (const auto& obj : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (!mesh || mesh->nodeName != object || !mesh->geometry) {
            continue;
        }
        const auto& geometry = *mesh->geometry;
        const auto* positions = geometry.get_positions();
        const auto* normals = geometry.get_normals();
        const auto* materials = geometry.get_attribute_data<uint16_t>("materialID");
        const auto* uvs = uv_set > 0
            ? geometry.get_attribute_data<Vec2>("uv" + std::to_string(uv_set))
            : geometry.get_uvs();
        if (!uvs) {
            uvs = geometry.get_uvs();
        }
        if (!positions || !uvs) {
            continue;
        }
        for (size_t offset = 0; offset + 2 < geometry.indices.size(); offset += 3) {
            const uint32_t* indices = geometry.indices.data() + offset;
            const size_t count = geometry.get_vertex_count();
            if (indices[0] >= count || indices[1] >= count || indices[2] >= count ||
                (materials ? materials[indices[0]] : 0) != material_id) {
                continue;
            }
            FlatPaintFace face;
            for (size_t i = 0; i < 3; ++i) {
                face.positions[i] = positions[indices[i]];
                face.normals[i] = normals ? normals[indices[i]] : Vec3(0.0f);
                face.uvs[i] = uvs[indices[i]];
            }
            faces.push_back(face);
        }
    }
    return faces;
}

bool sameCompute(const ComputeBufferHandle& h, const SimulationComputeContext& c) {
    return h.valid() && h.backend == c.backendType();
}

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace

// ── state ───────────────────────────────────────────────────────────────────

struct WetPaintGpu::Surface {
    std::string object;
    int material_id = 0;
    int resolution = 0;
    uint32_t width = 0, height = 0;
    uint32_t page_offset = 0;
    uint32_t tile_offset = 0, tiles_x = 0, tiles_y = 0;
    std::weak_ptr<Texture> texture;
    std::string texture_name;
    std::vector<float> pos_valid;      // 4 per texel
    std::vector<float> flow_normal;    // 4 per texel
    std::vector<uint32_t> base;        // RGBA8, original base colour
    std::array<std::weak_ptr<Texture>, 3> channel_textures;
    std::array<std::vector<uint32_t>, 3> channel_base; // roughness, height mask, normal
    std::vector<Vec2> normal_spans;    // same Jacobian metric used by CPU Paint
    bool channels_refresh = false;
    std::vector<GpuTile> tiles;        // surface field filled at layout
    uint32_t valid_texels = 0, gutter_texels = 0, triangles = 0;
    WetPaintSurfaceSettings settings;
    int uv_set = 0;
    uint16_t paint_material = 0;
    std::array<double, 8> geometry_key{};   // geometryKey() at the last bake
    // Dirty texel rectangle not yet in each backend (0 render, 1 viewport).
    struct Rect { bool any = false; int x0 = 0, y0 = 0, x1 = -1, y1 = -1; };
    Rect pending[2];
    Rect channel_pending[2][3];
    WetPaintSurfaceInfo last;
};

struct WetPaintGpu::Device {
    SimulationComputeContext* compute = nullptr;
    ComputeBackendType backend = ComputeBackendType::CPU;
    ComputeBufferHandle surfaces, tiles, pos_valid, flow_normal, base, color, state,
        color_scratch, state_scratch, output, tile_flags, stats, dabs, lut, lookup,
        dummy_events, dummy_counter;
    std::size_t dab_capacity = 0, lut_capacity = 0, lookup_capacity = 0;
    uint32_t texel_count = 0, tile_count = 0;
    bool channels_allocated = false;
    bool channels_force = false;
    // The stats of the last dispatching tick, read one tick later
    // (SimulationComputeContext::submitWithReadback), with what that tick knew.
    uint64_t stats_ticket = 0;
    uint32_t stats_surfaces = 0;
    bool stats_force = false;      // channels_force at that dispatch
    bool stats_coverage = false;   // that tick measured coverage
    uint64_t stats_serial = 0;     // paint_serial_ of that tick

    void release() {
        stats_ticket = 0;   // a dropped ticket's slot is reused by the backend
        if (compute) {
            for (ComputeBufferHandle* h : {&surfaces, &tiles, &pos_valid, &flow_normal, &base, &color,
                                           &state, &color_scratch, &state_scratch, &output,
                                           &tile_flags, &stats, &dabs, &lut, &lookup,
                                           &dummy_events, &dummy_counter}) {
                if (sameCompute(*h, *compute)) compute->destroyBuffer(*h);
                *h = {};
            }
        }
        dab_capacity = lut_capacity = lookup_capacity = 0;
        texel_count = tile_count = 0;
        channels_allocated = channels_force = false;
        compute = nullptr;
    }
};

WetPaintGpu& wetPaintGpu() {
    static WetPaintGpu instance;
    return instance;
}

WetPaintGpu::WetPaintGpu() = default;
WetPaintGpu::~WetPaintGpu() {
    // The compute contexts may outlive their VkDevice at process exit; drop
    // handles only and leak our own backend rather than destroy on a dead device.
    if (device_) device_->compute = nullptr;
    (void)own_compute_.release();
}

WetPaintGpu::Surface* WetPaintGpu::find(const std::string& object, int material_id) {
    for (auto& s : surfaces_) {
        if (s->object == object && (material_id < 0 || s->material_id == material_id)) return s.get();
    }
    return nullptr;
}

// ── surfaces ────────────────────────────────────────────────────────────────

// Cheap fingerprint of an object's world geometry (flat meshes carry their
// transform in the positions): vertex count, position sum and bounds.
std::array<double, 8> WetPaintGpu::geometryKey(SceneData& scene, const std::string& object) {
    std::array<double, 8> key{};
    key[5] = key[6] = key[7] = 0.0;
    double mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30};
    for (const auto& obj : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (!mesh || mesh->nodeName != object || !mesh->geometry) continue;
        const auto* positions = mesh->geometry->get_positions();
        const size_t count = mesh->geometry->get_vertex_count();
        if (!positions) continue;
        key[0] += static_cast<double>(count);
        for (size_t i = 0; i < count; ++i) {
            const Vec3& v = positions[i];
            key[1] += v.x; key[2] += v.y; key[3] += v.z;
            mn[0] = std::min(mn[0], double(v.x)); mx[0] = std::max(mx[0], double(v.x));
            mn[1] = std::min(mn[1], double(v.y)); mx[1] = std::max(mx[1], double(v.y));
            mn[2] = std::min(mn[2], double(v.z)); mx[2] = std::max(mx[2], double(v.z));
        }
    }
    key[4] = mx[0] - mn[0]; key[5] = mx[1] - mn[1]; key[6] = mx[2] - mn[2];
    key[7] = mn[1];
    return key;
}

// Rasterises the surface's triangles into texel space: world position,
// flow and normal per texel, gutter and tile bounds. Flat meshes keep their
// world transform in the geometry, so moving or scaling the object changes
// what this produces - tick() re-runs it when geometryKey() changes.
bool WetPaintGpu::bakeGeometry(SceneData& scene, Surface& s, std::string& error) {
    const size_t texels = static_cast<size_t>(s.width) * s.height;
    s.pos_valid.assign(texels * 4, 0.0f);
    s.flow_normal.assign(texels * 4, 0.0f);
    s.normal_spans.assign(texels, Vec2(1.0f, 1.0f));
    s.valid_texels = s.gutter_texels = s.triangles = 0;
    s.geometry_key = geometryKey(scene, s.object);
    // ── bake: rasterize every triangle of (object, material) into texel space.
    // Texel (px, py) <-> uv (px / (W-1), 1 - py / (H-1)), the CPU model's map.
    const int uv_set = s.uv_set;
    const auto faces = paintFaces(scene, s.object, s.paint_material, uv_set);
    const float W1 = static_cast<float>(s.width - 1);
    const float H1 = static_cast<float>(s.height - 1);
    const Vec3 gravity(0.0f, -1.0f, 0.0f);
    for (const auto& tri : faces) {
        const auto& uv0 = tri.uvs[0];
        const auto& uv1 = tri.uvs[1];
        const auto& uv2 = tri.uvs[2];
        const float x[3] = {uv0.u * W1, uv1.u * W1, uv2.u * W1};
        const float y[3] = {(1.0f - uv0.v) * H1, (1.0f - uv1.v) * H1, (1.0f - uv2.v) * H1};
        const float area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
        if (std::abs(area) < 1e-12f) continue;
        ++s.triangles;
        const auto& p = tri.positions;
        Vec3 geom_n = (p[1] - p[0]).cross(p[2] - p[0]);
        const float gl = geom_n.length();
        geom_n = gl > 1e-20f ? geom_n / gl : Vec3(0.0f, 1.0f, 0.0f);
        const auto& n = tri.normals;
        const Vec2 normal_span = HeightNormalMath::twoTexelWorldSpan(
            p[0], p[1], p[2], uv0, uv1, uv2, s.width, s.height);
        RayTrophiSim::SurfaceFlowSample flow;
        const bool has_flow = RayTrophiSim::SurfaceFlowField::computePixelFlow(
            p[0], p[1], p[2], uv0, uv1, uv2, static_cast<int>(s.width),
            static_cast<int>(s.height), flow, gravity) && flow.valid();
        const int x0 = std::max(0, static_cast<int>(std::floor(std::min({x[0], x[1], x[2]}))));
        const int x1 = std::min(static_cast<int>(s.width) - 1, static_cast<int>(std::ceil(std::max({x[0], x[1], x[2]}))));
        const int y0 = std::max(0, static_cast<int>(std::floor(std::min({y[0], y[1], y[2]}))));
        const int y1 = std::min(static_cast<int>(s.height) - 1, static_cast<int>(std::ceil(std::max({y[0], y[1], y[2]}))));
        const float inv_area = 1.0f / area;
        for (int py = y0; py <= y1; ++py) {
            for (int px = x0; px <= x1; ++px) {
                const float fx = static_cast<float>(px), fy = static_cast<float>(py);
                const float w0 = ((x[1] - fx) * (y[2] - fy) - (x[2] - fx) * (y[1] - fy)) * inv_area;
                const float w1 = ((x[2] - fx) * (y[0] - fy) - (x[0] - fx) * (y[2] - fy)) * inv_area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < -1e-4f || w1 < -1e-4f || w2 < -1e-4f) continue;
                const size_t ti = static_cast<size_t>(py) * s.width + px;
                if (s.pos_valid[ti * 4 + 3] > 0.5f) continue;   // first triangle wins
                const Vec3 wp = p[0] * w0 + p[1] * w1 + p[2] * w2;
                Vec3 wn = n[0] * w0 + n[1] * w1 + n[2] * w2;
                const float nl = wn.length();
                wn = nl > 1e-6f ? wn / nl : geom_n;
                if (wn.dot(geom_n) < 0.0f && nl <= 1e-6f) wn = geom_n;
                s.pos_valid[ti * 4 + 0] = wp.x;
                s.pos_valid[ti * 4 + 1] = wp.y;
                s.pos_valid[ti * 4 + 2] = wp.z;
                s.pos_valid[ti * 4 + 3] = 1.0f;
                s.flow_normal[ti * 4 + 0] = has_flow ? flow.flow_x : 0.0f;
                s.flow_normal[ti * 4 + 1] = has_flow ? flow.flow_y : 0.0f;
                s.flow_normal[ti * 4 + 2] = has_flow ? flow.slope : 0.0f;
                s.flow_normal[ti * 4 + 3] = packOctNormal(wn);
                s.normal_spans[ti] = normal_span;
                ++s.valid_texels;
            }
        }
    }
    if (s.valid_texels == 0) {
        error = "no texel of " + s.object + " is covered by its UVs (uv_set " + std::to_string(uv_set) + ")";
        return false;
    }
    // Gutter: copy the nearest island texel outward so bilinear reads and
    // stamping cover the seams (w = 2: stamped, not counted as coverage).
    for (int pass = 0; pass < kGutterTexels; ++pass) {
        std::vector<float> pv = s.pos_valid, fn = s.flow_normal;
        auto spans = s.normal_spans;
        for (uint32_t py = 0; py < s.height; ++py) {
            for (uint32_t px = 0; px < s.width; ++px) {
                const size_t ti = static_cast<size_t>(py) * s.width + px;
                if (s.pos_valid[ti * 4 + 3] > 0.5f) continue;
                static constexpr int kN[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                for (const auto& o : kN) {
                    const int nx = static_cast<int>(px) + o[0], ny = static_cast<int>(py) + o[1];
                    if (nx < 0 || ny < 0 || nx >= static_cast<int>(s.width) || ny >= static_cast<int>(s.height)) continue;
                    const size_t ni = static_cast<size_t>(ny) * s.width + nx;
                    if (s.pos_valid[ni * 4 + 3] < 0.5f) continue;
                    for (int c = 0; c < 4; ++c) {
                        pv[ti * 4 + c] = s.pos_valid[ni * 4 + c];
                        fn[ti * 4 + c] = s.flow_normal[ni * 4 + c];
                    }
                    spans[ti] = s.normal_spans[ni];
                    pv[ti * 4 + 3] = 2.0f;
                    ++s.gutter_texels;
                    break;
                }
            }
        }
        s.pos_valid.swap(pv);
        s.flow_normal.swap(fn);
        s.normal_spans.swap(spans);
    }
    // Tiles: world AABB of their texels (island + gutter).
    s.tiles_x = (s.width + kTile - 1) / kTile;
    s.tiles_y = (s.height + kTile - 1) / kTile;
    s.tiles.assign(static_cast<size_t>(s.tiles_x) * s.tiles_y, GpuTile{});
    for (uint32_t ty = 0; ty < s.tiles_y; ++ty) {
        for (uint32_t tx = 0; tx < s.tiles_x; ++tx) {
            GpuTile& t = s.tiles[static_cast<size_t>(ty) * s.tiles_x + tx];
            t.tx = tx;
            t.ty = ty;
            float mn[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max()};
            float mx[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                           -std::numeric_limits<float>::max()};
            for (uint32_t ly = 0; ly < kTile; ++ly) {
                const uint32_t py = ty * kTile + ly;
                if (py >= s.height) break;
                for (uint32_t lx = 0; lx < kTile; ++lx) {
                    const uint32_t px = tx * kTile + lx;
                    if (px >= s.width) break;
                    const size_t ti = static_cast<size_t>(py) * s.width + px;
                    if (s.pos_valid[ti * 4 + 3] < 0.5f) continue;
                    ++t.valid_texels;
                    for (int c = 0; c < 3; ++c) {
                        mn[c] = std::min(mn[c], s.pos_valid[ti * 4 + c]);
                        mx[c] = std::max(mx[c], s.pos_valid[ti * 4 + c]);
                    }
                }
            }
            if (t.valid_texels > 0) {
                for (int c = 0; c < 3; ++c) { t.bmin[c] = mn[c]; t.bmax[c] = mx[c]; }
            }
        }
    }

    return true;
}

bool WetPaintGpu::addSurface(SceneData& scene, const std::string& object, int material_id,
                             int resolution, std::string& error) {
    if (resolution < 32 || resolution > 4096) {
        error = "wet paint resolution must be between 32 and 4096";
        return false;
    }
    auto triangle = findPaintTriangle(scene, object, material_id);
    if (!triangle) {
        error = "paint target mesh/material not found: " + object;
        return false;
    }
    MeshPaintAdapter adapter(&scene, triangle);
    if (!adapter.isValid()) {
        error = "paint target is not valid: " + object;
        return false;
    }
    const int mat = static_cast<int>(adapter.getMaterialID());
    // One material = one base colour texture. Two objects sharing a material
    // would bind their paint textures to the same slot and the last one wins on
    // every object (the CPU paint target has the same limit, silently).
    for (const auto& other : surfaces_) {
        if (other->material_id == mat && other->object != adapter.getNodeName()) {
            error = "material " + std::to_string(mat) + " is shared with wet paint surface '" +
                    other->object + "': give each painted object its own material (material.assign)";
            return false;
        }
    }
    // The same target paint.ensure_target creates, with a base colour texture
    // bound to the material (seeded from the original albedo).
    adapter.ensureTextureSet(resolution);
    if (!adapter.assignTextureToChannel(PaintChannel::BaseColor)) {
        error = "could not create the base colour paint texture for " + object;
        return false;
    }
    PaintTextureSet* set = adapter.getTextureSet();
    std::shared_ptr<Texture> texture = set ? set->getTexture(PaintChannel::BaseColor) : nullptr;
    if (!texture || texture->width <= 1 || texture->height <= 1 ||
        texture->pixels.size() != static_cast<size_t>(texture->width) * texture->height) {
        error = "paint texture has no host pixels for " + object;
        return false;
    }

    auto surface = std::make_unique<Surface>();
    Surface& s = *surface;
    s.object = adapter.getNodeName();
    s.material_id = mat;
    // Report the size actually painted: a surface re-added at another size
    // keeps its existing base texture, and a saved game design must reload the
    // same size it shows.
    s.resolution = texture->width;
    s.width = static_cast<uint32_t>(texture->width);
    s.height = static_cast<uint32_t>(texture->height);
    s.texture = texture;
    s.texture_name = texture->name;
    const size_t texels = static_cast<size_t>(s.width) * s.height;
    s.base.resize(texels);
    for (size_t i = 0; i < texels; ++i) s.base[i] = packRgba8(texture->pixels[i]);

    s.uv_set = adapter.getTarget().uv_set;
    s.paint_material = adapter.getMaterialID();
    if (!bakeGeometry(scene, s, error)) {
        return false;
    }

    // Replacing a surface keeps its settings.
    for (auto it = surfaces_.begin(); it != surfaces_.end(); ++it) {
        if ((*it)->object == s.object && (*it)->material_id == s.material_id) {
            s.settings = (*it)->settings;
            surfaces_.erase(it);
            break;
        }
    }
    surfaces_.push_back(std::move(surface));
    layout_dirty_ = true;
    settings_dirty_ = true;
    return true;
}

bool WetPaintGpu::removeSurface(const std::string& object, int material_id, SceneData* scene) {
    for (auto it = surfaces_.begin(); it != surfaces_.end(); ++it) {
        if ((*it)->object == object && (material_id < 0 || (*it)->material_id == material_id)) {
            if (scene) {
                // A removed surface must not keep painting its material: the bound
                // paint texture overrides base_color, so an object that was a
                // surface once (a balloon picked by mistake) rendered white
                // forever while material.get reported the colour set on it.
                if (auto triangle = findPaintTriangle(*scene, (*it)->object, (*it)->material_id)) {
                    MeshPaintAdapter adapter(scene, triangle);
                    adapter.releaseLayerStackFromScene();
                    if (adapter.restoreOriginalMaterialTextures()) {
                        g_materials_dirty = true;
                    }
                }
            }
            surfaces_.erase(it);
            layout_dirty_ = true;
            return true;
        }
    }
    return false;
}

void WetPaintGpu::clear() {
    surfaces_.clear();
    pending_dabs_.clear();
    if (device_) device_->release();
    device_.reset();
    stats_ = WetPaintStats{};
    reset_requested_ = false;
    layout_dirty_ = true;
    settings_dirty_ = true;
    channels_dirty_ = false;
}

bool WetPaintGpu::setSurfaceSettings(const std::string& object, int material_id,
                                     const WetPaintSurfaceSettings& settings) {
    Surface* s = find(object, material_id);
    if (!s) return false;
    const auto& previous = s->settings;
    const bool changed = previous.roughness_channel != settings.roughness_channel ||
        previous.height_normal_channel != settings.height_normal_channel ||
        previous.wet_roughness != settings.wet_roughness ||
        previous.dry_roughness != settings.dry_roughness ||
        previous.height_contribution != settings.height_contribution ||
        previous.normal_strength != settings.normal_strength;
    s->settings = settings;
    s->channels_refresh = s->channels_refresh || changed;
    channels_dirty_ = channels_dirty_ || changed;
    settings_dirty_ = true;
    return true;
}

void WetPaintGpu::setSplatSettings(const WetPaintSplatSettings& settings) {
    splat_ = settings;
    splat_.max_events_per_tick = std::max<uint32_t>(1u, splat_.max_events_per_tick);
}

void WetPaintGpu::queueDab(const WetPaintDab& dab) { pending_dabs_.push_back(dab); }

bool WetPaintGpu::resetPaint() {
    if (surfaces_.empty()) return false;
    reset_requested_ = true;
    if (device_) device_->channels_force = device_->channels_allocated;
    pending_dabs_.clear();
    return true;
}

std::vector<WetPaintSurfaceInfo> WetPaintGpu::surfaces() const {
    std::vector<WetPaintSurfaceInfo> out;
    for (const auto& s : surfaces_) {
        WetPaintSurfaceInfo info = s->last;
        info.object = s->object;
        info.material_id = s->material_id;
        info.resolution = s->resolution;
        info.tiles = static_cast<uint32_t>(s->tiles.size());
        info.valid_texels = s->valid_texels;
        info.gutter_texels = s->gutter_texels;
        info.triangles = s->triangles;
        info.texture = s->texture_name;
        info.settings = s->settings;
        out.push_back(std::move(info));
    }
    return out;
}

// ── device ──────────────────────────────────────────────────────────────────

bool WetPaintGpu::ensureCompute(SceneData& scene) {
    // The simulation context is on Vulkan only while something asks for it (a
    // GPU particle system or domain); SceneData::syncSimulationWorld returns it
    // to the CPU otherwise, on every getSimulationWorld(). Use it when it is on
    // Vulkan - the particle event buffers live there, so contacts are stamped
    // without a download. Otherwise run on a context of our own on the same
    // VkDevice. Switching between the two rebuilds the atlas (paint is lost).
    SimulationComputeContext* chosen = &scene.getSimulationWorld().compute();
    stats_.compute_context = "simulation";
    if (chosen->backendType() != ComputeBackendType::VulkanCompute || !chosen->supportsDispatch()) {
        void* vk_device = g_vulkan_sim_compute_ctx.device;
        if (own_compute_ && own_compute_device_ != vk_device) {
            // The VkDevice changed under us: the old backend's handles are dead,
            // so it is dropped without touching them (a small leak, never a crash).
            if (device_ && device_->compute == own_compute_.get()) {
                device_->compute = nullptr;
                device_.reset();
                layout_dirty_ = true;
            }
            (void)own_compute_.release();
        }
        if (!own_compute_ && vk_device) {
            auto backend = RayTrophiSim::createVulkanSimulationComputeBackend(g_vulkan_sim_compute_ctx);
            if (backend) {
                own_compute_ = std::make_unique<SimulationComputeContext>();
                own_compute_->setBackend(std::move(backend));
                own_compute_device_ = vk_device;
            }
        }
        if (!own_compute_ || own_compute_->backendType() != ComputeBackendType::VulkanCompute ||
            !own_compute_->supportsDispatch()) {
            stats_.device_ready = false;
            stats_.status = "no_vulkan_compute";
            return false;
        }
        chosen = own_compute_.get();
        stats_.compute_context = "own";
    }
    SimulationComputeContext& compute = *chosen;
    if (device_ && (device_->compute != &compute || device_->backend != compute.backendType())) {
        device_->release();
        device_.reset();
        layout_dirty_ = true;
    }
    if (!device_) {
        device_ = std::make_unique<Device>();
        device_->compute = &compute;
        device_->backend = compute.backendType();
        layout_dirty_ = true;
    }
    return true;
}

bool WetPaintGpu::rebuildDevice(SceneData& scene) {
    (void)scene;
    Device& d = *device_;
    SimulationComputeContext& c = *d.compute;
    SimulationComputeContext* keep = d.compute;
    d.release();
    d.compute = keep;
    d.backend = c.backendType();

    uint32_t texels = 0, tiles = 0;
    for (auto& s : surfaces_) {
        s->page_offset = texels;
        s->tile_offset = tiles;
        texels += s->width * s->height;
        tiles += static_cast<uint32_t>(s->tiles.size());
        for (int b = 0; b < 2; ++b) {
            s->pending[b] = Surface::Rect{};
            for (auto& r : s->channel_pending[b]) r = Surface::Rect{};
        }
    }
    d.texel_count = texels;
    d.tile_count = tiles;
    if (texels == 0) return true;

    const ComputeBufferUsage ro = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                                  ComputeBufferUsage::ReadOnly;
    const ComputeBufferUsage rw = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                                  ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
    const auto makeBuffer = [&c](ComputeBufferHandle& handle, std::size_t size_bytes,
                                 const char* debug_name, ComputeBufferUsage usage) -> bool {
        ComputeBufferDesc desc;
        desc.debug_name = debug_name;
        desc.size_bytes = std::max<std::size_t>(size_bytes, 16);
        desc.usage = usage;
        handle = c.createBuffer(desc);
        return handle.valid();
    };
    const std::size_t n = surfaces_.size();
    bool ok = makeBuffer(d.surfaces, n * sizeof(GpuSurface), "WetPaintSurfaces", ro) &&
              makeBuffer(d.tiles, static_cast<std::size_t>(tiles) * sizeof(GpuTile), "WetPaintTiles", ro) &&
              makeBuffer(d.pos_valid, static_cast<std::size_t>(texels) * 16, "WetPaintPosValid", ro) &&
              makeBuffer(d.flow_normal, static_cast<std::size_t>(texels) * 16, "WetPaintFlowNormal", ro) &&
              makeBuffer(d.base, static_cast<std::size_t>(texels) * 4, "WetPaintBase", ro) &&
              makeBuffer(d.color, static_cast<std::size_t>(texels) * 4, "WetPaintColor", rw) &&
              makeBuffer(d.state, static_cast<std::size_t>(texels) * 16, "WetPaintState", rw) &&
              makeBuffer(d.color_scratch, static_cast<std::size_t>(texels) * 4, "WetPaintColorScratch", rw) &&
              makeBuffer(d.state_scratch, static_cast<std::size_t>(texels) * 16, "WetPaintStateScratch", rw) &&
              makeBuffer(d.output, static_cast<std::size_t>(texels) * 4, "WetPaintOutput", rw) &&
              makeBuffer(d.tile_flags, static_cast<std::size_t>(tiles) * 8, "WetPaintTileFlags", rw) &&
              makeBuffer(d.stats, n * kStatsStride * 4, "WetPaintStats", rw) &&
              makeBuffer(d.dummy_events, 48, "WetPaintNoEvents", rw) &&
              makeBuffer(d.dummy_counter, 16, "WetPaintNoCounter", rw);
    if (!ok) {
        stats_.status = "buffer_allocation_failed";
        d.release();
        d.compute = keep;
        return false;
    }

    std::vector<GpuTile> all_tiles;
    all_tiles.reserve(tiles);
    std::vector<uint32_t> zero_u;
    std::vector<float> zero_f;
    for (uint32_t si = 0; si < n; ++si) {
        Surface& s = *surfaces_[si];
        for (GpuTile t : s.tiles) {
            t.surface = si;
            all_tiles.push_back(t);
        }
        const std::size_t off = s.page_offset;
        const std::size_t cnt = static_cast<std::size_t>(s.width) * s.height;
        ok = ok && c.uploadBuffer(d.pos_valid, s.pos_valid.data(), cnt * 16, off * 16) &&
             c.uploadBuffer(d.flow_normal, s.flow_normal.data(), cnt * 16, off * 16) &&
             c.uploadBuffer(d.base, s.base.data(), cnt * 4, off * 4) &&
             c.uploadBuffer(d.output, s.base.data(), cnt * 4, off * 4);
    }
    zero_u.assign(texels, 0u);
    zero_f.assign(static_cast<std::size_t>(texels) * 4, 0.0f);
    std::vector<uint32_t> zero_flags(static_cast<std::size_t>(tiles) * 2, 0u);
    std::vector<uint32_t> zero_events(12, 0u), zero_counter(4, 0u);
    ok = ok && c.uploadBuffer(d.tiles, all_tiles.data(), all_tiles.size() * sizeof(GpuTile)) &&
         c.uploadBuffer(d.color, zero_u.data(), zero_u.size() * 4) &&
         c.uploadBuffer(d.color_scratch, zero_u.data(), zero_u.size() * 4) &&
         c.uploadBuffer(d.state, zero_f.data(), zero_f.size() * 4) &&
         c.uploadBuffer(d.state_scratch, zero_f.data(), zero_f.size() * 4) &&
         c.uploadBuffer(d.tile_flags, zero_flags.data(), zero_flags.size() * 4) &&
         c.uploadBuffer(d.dummy_events, zero_events.data(), zero_events.size() * 4) &&
         c.uploadBuffer(d.dummy_counter, zero_counter.data(), zero_counter.size() * 4);
    if (!ok) {
        stats_.status = "upload_failed";
        d.release();
        d.compute = keep;
        return false;
    }
    layout_dirty_ = false;
    settings_dirty_ = true;
    return true;
}

// Channel targets use the existing PaintTextureSet adapter (including its UV
// bake and source texture bookkeeping). No CPU brush/composite is run per frame.
bool WetPaintGpu::prepareChannels(SceneData& scene) {
    Device& d = *device_;
    const bool enabled = std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
        return s->settings.roughness_channel || s->settings.height_normal_channel;
    });
    if (!enabled && !d.channels_allocated) {
        channels_dirty_ = false;
        return true;
    }
    if (!channels_dirty_ && d.channels_allocated) return true;
    bool upload = !d.channels_allocated;
    static constexpr PaintChannel channels[3] = {
        PaintChannel::Roughness, PaintChannel::Mask, PaintChannel::Normal
    };
    for (auto& surface : surfaces_) {
        Surface& s = *surface;
        const bool need_roughness = s.settings.roughness_channel && s.channel_base[0].empty();
        const bool need_normal = s.settings.height_normal_channel && s.channel_base[2].empty();
        if (!need_roughness && !need_normal) continue;
        auto triangle = findPaintTriangle(scene, s.object, s.material_id);
        if (!triangle || !triangle->getMaterial()) {
            stats_.status = "channel_material_missing: " + s.object;
            return false;
        }
        auto material = triangle->getMaterial();
        auto principled = std::dynamic_pointer_cast<PrincipledBSDF>(material);
        const bool had_roughness = material->roughnessProperty.texture != nullptr;
        const float roughness_scalar = material->roughnessProperty.color.x *
                                       material->roughnessProperty.intensity;
        const float roughness_factor = material->roughnessProperty.intensity;
        const float normal_strength = material->get_normal_strength();
        const int selector = principled ? principled->roughness_tex_channel : 0;
        const int roughness_component = selector == 1 ? 0 : (selector == 3 ? 2 : 1);
        MeshPaintAdapter adapter(&scene, triangle);
        // Channel planes are sized like the surface's base colour texture. The
        // set's resolution is the last REQUESTED one (a surface re-added at
        // another size keeps its existing base texture), so new channel textures
        // would be created at the wrong size; and a channel texture left bound
        // by an earlier surface of another size can never be used. Pin the set
        // to the base size and recreate mismatched channel textures.
        PaintTextureSet& channel_set = adapter.ensureTextureSet(static_cast<int>(s.width));
        for (int ch = 0; ch < 3; ++ch) {
            if ((ch == 0 && !need_roughness) || (ch != 0 && !need_normal)) continue;
            std::shared_ptr<Texture>& existing = channel_set.getTextureRef(channels[ch]);
            if (existing && (existing->width != static_cast<int>(s.width) ||
                             existing->height != static_cast<int>(s.height))) {
                existing.reset();
            }
        }
        for (int ch = 0; ch < 3; ++ch) {
            if ((ch == 0 && !need_roughness) || (ch != 0 && !need_normal)) continue;
            if (!adapter.assignTextureToChannel(channels[ch])) {
                stats_.status = "channel_target_failed: " + s.object;
                return false;
            }
        }
        auto* set = adapter.getTextureSet();
        if (!set) return false;
        const std::size_t count = static_cast<std::size_t>(s.width) * s.height;
        for (int ch = 0; ch < 3; ++ch) {
            if ((ch == 0 && !need_roughness) || (ch != 0 && !need_normal)) continue;
            auto texture = set->getTexture(channels[ch]);
            if (!texture || texture->width != static_cast<int>(s.width) ||
                texture->height != static_cast<int>(s.height) || texture->pixels.size() != count) {
                stats_.status = "channel_resolution_mismatch: " + s.object;
                return false;
            }
            texture->name = "generated/wet/" + std::to_string(texture->m_uid) + ".png";
            texture->is_srgb = false;
            s.channel_textures[ch] = texture;
            s.channel_base[ch].resize(count);
            for (std::size_t pixel = 0; pixel < count; ++pixel) {
                CompactVec4 value = texture->pixels[pixel];
                if (ch == 0) {
                    const uint8_t components[3] = {value.r, value.g, value.b};
                    const float scalar = had_roughness
                        ? components[roughness_component] / 255.0f * roughness_factor
                        : roughness_scalar;
                    const auto byte = static_cast<uint8_t>(
                        std::clamp(scalar, 0.0f, 1.0f) * 255.0f + 0.5f);
                    value = CompactVec4(byte, byte, byte, 255);
                } else if (ch == 2 && normal_strength != 1.0f) {
                    Vec3 n(value.r / 127.5f - 1.0f, value.g / 127.5f - 1.0f,
                           value.b / 127.5f - 1.0f);
                    n.x *= normal_strength;
                    n.y *= normal_strength;
                    n = n.normalize();
                    const auto byte = [](float v) {
                        return static_cast<uint8_t>(std::clamp(v * 0.5f + 0.5f, 0.0f, 1.0f) *
                                                    255.0f + 0.5f);
                    };
                    value = CompactVec4(byte(n.x), byte(n.y), byte(n.z), 255);
                }
                s.channel_base[ch][pixel] = packRgba8(value);
            }
        }
        // Source factors are baked into the generated maps, preserving the
        // unpainted appearance. Both CPU and GPU now sample effective values.
        if (need_roughness) {
            material->roughnessProperty.color = Vec3(1.0f);
            material->roughnessProperty.intensity = 1.0f;
            if (principled) principled->roughness_tex_channel = 1;
        }
        if (need_normal) material->set_normal_strength(1.0f);
        // A later channel assignment may flatten the original CPU stack again.
        // Reinstall all ORIGINAL channel pixels, never GPU paint pixels.
        for (int ch = 0; ch < 3; ++ch) {
            auto texture = s.channel_textures[ch].lock();
            if (!texture) continue;
            for (std::size_t pixel = 0; pixel < count; ++pixel) {
                const uint32_t v = s.channel_base[ch][pixel];
                texture->pixels[pixel] = CompactVec4(v & 255u, (v >> 8u) & 255u,
                                                     (v >> 16u) & 255u, v >> 24u);
            }
        }
        s.channels_refresh = true;
        g_materials_dirty = true;
        upload = true;
    }
    auto& compute = *d.compute;
    if (upload) {
        const std::size_t count = d.texel_count;
        std::vector<uint32_t> original(count * 6, 0u);
        for (const auto& surface : surfaces_) {
            const Surface& s = *surface;
            const std::size_t offset = s.page_offset;
            std::copy(s.base.begin(), s.base.end(), original.begin() + offset);
            for (int ch = 0; ch < 3; ++ch) {
                if (!s.channel_base[ch].empty()) {
                    std::copy(s.channel_base[ch].begin(), s.channel_base[ch].end(),
                              original.begin() + (ch + 1) * count + offset);
                }
            }
            for (std::size_t pixel = 0; pixel < s.normal_spans.size(); ++pixel) {
                std::memcpy(&original[4 * count + offset + pixel], &s.normal_spans[pixel].u, 4);
                std::memcpy(&original[5 * count + offset + pixel], &s.normal_spans[pixel].v, 4);
            }
        }
        ComputeBufferHandle new_base, new_output;
        if (!d.channels_allocated) {
            ComputeBufferDesc desc;
            desc.debug_name = "WetPaintChannelSources";
            desc.size_bytes = original.size() * 4;
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                         ComputeBufferUsage::ReadOnly;
            new_base = compute.createBuffer(desc);
            desc.debug_name = "WetPaintChannelOutputs";
            desc.size_bytes = count * 16;
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                         ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
            new_output = compute.createBuffer(desc);
            if (!new_base.valid() || !new_output.valid()) {
                if (new_base.valid()) compute.destroyBuffer(new_base);
                if (new_output.valid()) compute.destroyBuffer(new_output);
                stats_.status = "channel_allocation_failed";
                return false;
            }
        }
        auto& base = d.channels_allocated ? d.base : new_base;
        auto& output = d.channels_allocated ? d.output : new_output;
        compute.synchronize();
        const bool ok = compute.uploadBuffer(base, original.data(), original.size() * 4) &&
                        compute.uploadBuffer(output, original.data(), count * 16);
        if (!ok) {
            if (new_base.valid()) compute.destroyBuffer(new_base);
            if (new_output.valid()) compute.destroyBuffer(new_output);
            stats_.status = "channel_upload_failed";
            return false;
        }
        if (!d.channels_allocated) {
            compute.destroyBuffer(d.base);
            compute.destroyBuffer(d.output);
            d.base = new_base;
            d.output = new_output;
            d.channels_allocated = true;
        }
    }
    d.channels_force = true;
    channels_dirty_ = false;
    stats_.pbr_buffer_bytes = static_cast<uint64_t>(d.texel_count) * 32u;
    stats_.pbr_texture_bytes = 0;
    for (const auto& s : surfaces_) {
        for (const auto& texture : s->channel_textures) {
            if (!texture.expired()) stats_.pbr_texture_bytes +=
                static_cast<uint64_t>(s->width) * s->height * 4u;
        }
    }
    return true;
}

void WetPaintGpu::uploadSurfaceTable() {
    Device& d = *device_;
    std::vector<GpuSurface> table;
    table.reserve(surfaces_.size());
    for (const auto& sp : surfaces_) {
        const Surface& s = *sp;
        const WetPaintSurfaceSettings& w = s.settings;
        // tickWetPaint's derived terms.
        const float lifetime = std::max(0.05f, w.lifetime_seconds);
        const float lifetime_runoff = std::clamp((lifetime - 0.35f) / 1.5f, 0.0f, 1.0f);
        GpuSurface g;
        g.width = s.width;
        g.height = s.height;
        g.page_offset = s.page_offset;
        g.tile_offset = s.tile_offset;
        g.tiles_x = s.tiles_x;
        g.tiles_y = s.tiles_y;
        g.flags = (w.oil ? 1u : 0u) | (w.roughness_channel ? 2u : 0u) |
                  (w.height_normal_channel ? 4u : 0u);
        const auto unorm = [](float value, uint32_t scale) {
            return static_cast<uint32_t>(std::clamp(value, 0.0f, 1.0f) * scale + 0.5f);
        };
        g._pad0 = unorm(w.wet_roughness, 255u) | (unorm(w.dry_roughness, 255u) << 8u) |
                  (unorm(w.height_contribution, 65535u) << 16u);
        g._pad2 = w.normal_strength;
        g.diffusion = std::clamp(w.diffusion, 0.0f, 2.0f) * (w.oil ? 0.58f : 1.0f);
        g.runoff = std::clamp(w.runoff, 0.0f, 2.0f) * lifetime_runoff * (w.oil ? 0.18f : 1.0f);
        g.absorption = std::clamp(w.absorption, 0.0f, 1.0f) * (w.oil ? 0.55f : 1.0f);
        g.drip_head = std::clamp(w.drip_head, 0.0f, 1.5f) * lifetime_runoff * (w.oil ? 0.24f : 1.0f);
        g.lifetime = lifetime;
        g.brush_radius_px = std::max(0.1f, w.brush_radius * (static_cast<float>(s.width) / 1024.0f));
        g.falloff = std::clamp(w.falloff, 0.0f, 1.0f);
        g.paint_load = std::clamp(w.paint_load, 0.0f, 1.0f);
        g.wetness = std::clamp(w.wetness, 0.0f, 1.0f);
        g.deposit_rate = std::clamp(w.deposit_rate, 0.0f, 1.0f);
        g.pickup_rate = std::clamp(w.pickup_rate, 0.0f, 1.0f);
        table.push_back(g);
    }
    if (!table.empty()) {
        d.compute->uploadBuffer(d.surfaces, table.data(), table.size() * sizeof(GpuSurface));
    }
    settings_dirty_ = false;
}

// ── tick ────────────────────────────────────────────────────────────────────

bool WetPaintGpu::tick(SceneData& scene, float dt) {
    const auto t0 = std::chrono::steady_clock::now();
    // A surface whose texture went away (project change, target released).
    for (auto it = surfaces_.begin(); it != surfaces_.end();) {
        if ((*it)->texture.expired()) {
            it = surfaces_.erase(it);
            layout_dirty_ = true;
        } else {
            ++it;
        }
    }
    // A surface whose object was moved, scaled or edited since its bake would
    // keep painting where the object USED to be (positions are baked once):
    // the room raised to 4.7 m painted nothing above the old 3.4 m. Checked
    // twice a second; a changed surface is re-baked and the atlas rebuilt.
    geometry_check_clock_ += std::clamp(dt, 0.0f, 0.25f);
    if (geometry_check_clock_ >= 0.5f) {
        geometry_check_clock_ = 0.0f;
        for (auto& surface : surfaces_) {
            const auto key = geometryKey(scene, surface->object);
            bool changed = false;
            for (size_t k = 0; k < key.size(); ++k) {
                if (std::abs(key[k] - surface->geometry_key[k]) >
                    1e-4 * std::max(1.0, std::abs(key[k]))) {
                    changed = true;
                }
            }
            if (!changed) continue;
            std::string error;
            if (bakeGeometry(scene, *surface, error)) {
                ++stats_.geometry_rebakes;
                layout_dirty_ = true;
            } else {
                stats_.status = "rebake_failed: " + surface->object + ": " + error;
                surface->geometry_key = key;   // do not retry every check
            }
        }
    }
    stats_.surfaces = static_cast<uint32_t>(surfaces_.size());
    if (surfaces_.empty()) {
        if (device_) { device_->release(); device_.reset(); }
        pending_dabs_.clear();
        stats_.device_ready = false;
        stats_.status = "no_surfaces";
        return false;
    }
    if (!ensureCompute(scene)) return false;
    Device& d = *device_;
    SimulationComputeContext& c = *d.compute;
    const bool layout_fresh = layout_dirty_;
    if (layout_dirty_ && !rebuildDevice(scene)) return false;
    if (!prepareChannels(scene)) return false;
    if (settings_dirty_) uploadSurfaceTable();
    stats_.device_ready = true;
    stats_.tiles = d.tile_count;

    dt = dt > 0.0f ? std::min(dt, 2.5f / 60.0f) : (1.0f / 60.0f);

    WetPaintGpuConstants k;
    k.surface_count = n;
    k.tile_count = d.tile_count;
    k._p0 = d.channels_allocated ? d.texel_count : 0u;
    k._p1 = d.channels_force ? 1u : 0u;
    k.dt = dt;
    const uint32_t n = static_cast<uint32_t>(surfaces_.size());

    // The previous dispatching tick's stats (dirty rectangles, active tiles,
    // coverage). One frame later its submission has long finished, so this
    // wait is ~free; before 2026-11-18 the tick synchronized at its end, and on
    // the single queue that waited for the render frame queued ahead of it.
    if (d.stats_ticket != 0) {
        std::vector<uint32_t> prev(static_cast<std::size_t>(d.stats_surfaces) * kStatsStride, 0u);
        const bool got = c.collectReadback(d.stats_ticket, prev.data(), prev.size() * 4, /*wait=*/true);
        d.stats_ticket = 0;
        if (got && d.stats_surfaces == n) {
            applyStats(prev, d.stats_force, d.stats_coverage, d.stats_serial);
        } else {
            ++stats_.stats_readbacks_lost;
        }
    }
    k.splat_radius = std::max(0.0f, splat_.radius);
    k.splat_amount = std::clamp(splat_.amount, 0.0f, 1.0f);
    k.splat_min_speed = std::max(0.0f, splat_.min_speed);
    k.facing_min = std::clamp(splat_.facing_min, -1.0f, 1.0f);
    k.max_events = splat_.max_events_per_tick;

    ComputeBufferHandle bindings[kBindingCount] = {
        d.surfaces, d.tiles, d.pos_valid, d.flow_normal, d.base, d.color, d.state,
        d.color_scratch, d.state_scratch, d.output, d.tile_flags, d.stats,
        d.dummy_events /*dabs*/, d.dummy_events, d.dummy_counter, d.dummy_events /*lut*/,
        d.dummy_counter /*lookup*/,
    };
    bool ok = true;
    const auto run = [&](uint32_t mode, uint32_t groups) {
        k.mode = mode;
        ComputeDispatch cmd;
        cmd.kernel = "paint_wet";
        cmd.buffers = bindings;
        cmd.buffer_count = kBindingCount;
        cmd.constants = &k;
        cmd.constants_size = sizeof(k);
        cmd.groups.groups_x = std::max<uint32_t>(1u, groups);
        ok = c.dispatch(cmd) && ok;
    };
    const auto ensure = [&](ComputeBufferHandle& h, std::size_t& cap, std::size_t needed,
                            std::size_t elem, const char* name) {
        if (sameCompute(h, c) && cap >= needed) return true;
        if (sameCompute(h, c)) c.destroyBuffer(h);
        std::size_t want = 64;
        while (want < needed) want *= 2;
        ComputeBufferDesc desc;
        desc.debug_name = name;
        desc.size_bytes = want * elem;
        desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload | ComputeBufferUsage::ReadOnly;
        h = c.createBuffer(desc);
        cap = h.valid() ? want : 0;
        return h.valid();
    };

    // Particle systems: device buffers on this context, or host contacts.
    std::vector<RayTrophiSim::ParticleSimulationSystem*> device_sources;
    std::vector<GpuDab> dabs;
    dabs.reserve(pending_dabs_.size());
    for (const WetPaintDab& w : pending_dabs_) {
        GpuDab g;
        g.pos_radius[0] = w.position[0]; g.pos_radius[1] = w.position[1];
        g.pos_radius[2] = w.position[2]; g.pos_radius[3] = std::max(0.0f, w.radius);
        g.color_amount[0] = w.color[0]; g.color_amount[1] = w.color[1];
        g.color_amount[2] = w.color[2]; g.color_amount[3] = std::clamp(w.amount, 0.0f, 1.0f);
        g.normal_facing[0] = w.normal[0]; g.normal_facing[1] = w.normal[1];
        g.normal_facing[2] = w.normal[2]; g.normal_facing[3] = w.facing ? 1.0f : 0.0f;
        dabs.push_back(g);
    }
    stats_.dabs_applied += pending_dabs_.size();
    pending_dabs_.clear();
    if (splat_.consume_particles) {
        std::vector<RayTrophiSim::ParticleCollisionEvent> host_events;
        uint32_t host_taken = 0;
        for (auto& system : scene.particle_systems) {
            if (!system.runtime || !system.enabled) continue;
            if (!splat_.particle_system.empty() && system.name != splat_.particle_system) continue;
            RayTrophiSim::ParticleDeviceEventView view;
            if (system.runtime->deviceCollisionEventView(c, view)) {
                // A system with no live particle now and none at the last tick
                // cannot have a contact pending. Before 2026-11-18 every
                // recording system counted as "arriving", so the idle exit below
                // never fired: a full tick (7.5..13 ms, Player) every frame with
                // nothing wet. The tick after the last particle died still stamps.
                const bool alive = system.runtime->aliveCount() > 0;
                bool& was_alive = source_alive_[system.runtime.get()];
                if (alive || was_alive) device_sources.push_back(system.runtime.get());
                was_alive = alive;
            }
            host_events.clear();
            if (system.runtime->takeHostCollisionEvents(host_events) == 0) continue;
            // CPU-reference contacts: the same splat, coloured on the host.
            for (const auto& e : host_events) {
                if (e.speed < splat_.min_speed || host_taken >= splat_.max_events_per_tick) continue;
                ++host_taken;
                const float* row = system.runtime->appearanceLutRow(e.profile);
                float rgb[3] = {1.0f, 1.0f, 1.0f};
                if (row) {
                    const float s = std::clamp(e.age_fraction, 0.0f, 1.0f) * 63.0f;
                    const int i0 = static_cast<int>(std::floor(s));
                    const int i1 = std::min(i0 + 1, 63);
                    const float f = s - static_cast<float>(i0);
                    for (int ch = 0; ch < 3; ++ch) {
                        rgb[ch] = linearToSrgb01(row[i0 * 8 + ch] * (1.0f - f) + row[i1 * 8 + ch] * f);
                    }
                }
                GpuDab g;
                g.pos_radius[0] = e.position[0]; g.pos_radius[1] = e.position[1];
                g.pos_radius[2] = e.position[2]; g.pos_radius[3] = k.splat_radius;
                g.color_amount[0] = rgb[0]; g.color_amount[1] = rgb[1];
                g.color_amount[2] = rgb[2]; g.color_amount[3] = k.splat_amount;
                g.normal_facing[0] = e.normal[0]; g.normal_facing[1] = e.normal[1];
                g.normal_facing[2] = e.normal[2]; g.normal_facing[3] = 1.0f;
                dabs.push_back(g);
                ++stats_.host_events_applied;
            }
        }
    }
    // Idle: nothing wet, nothing arriving. No dispatch, no readback.
    if (dabs.empty() && device_sources.empty() && !reset_requested_ && stats_.active_tiles == 0 &&
        !layout_fresh && !d.channels_force) {
        stats_.status = "idle";
        stats_.ticked_tiles = stats_.stamped_texels = 0;
        stats_.last_tick_ms = msSince(t0);
        return true;
    }
    // Per-tick stat slots: dirty bbox min = max uint, the rest zero.
    std::vector<uint32_t> stats_init(static_cast<std::size_t>(n) * kStatsStride, 0u);
    for (uint32_t si = 0; si < n; ++si) {
        stats_init[si * kStatsStride + kStDirtyMinX] = 0xFFFFFFFFu;
        stats_init[si * kStatsStride + kStDirtyMinY] = 0xFFFFFFFFu;
    }
        ++stats_.idle_ticks;
        stats_.event_sources = 0;
    c.uploadBuffer(d.stats, stats_init.data(), stats_init.size() * 4);
    if (reset_requested_) {
        run(kModeReset, d.tile_count);
        reset_requested_ = false;
    ++paint_serial_;   // this tick dispatches: coverage() must measure again
    stats_.event_sources = static_cast<uint32_t>(device_sources.size());
    }

    if (!dabs.empty() && ensure(d.dabs, d.dab_capacity, dabs.size(), sizeof(GpuDab), "WetPaintDabs") &&
        c.uploadBuffer(d.dabs, dabs.data(), dabs.size() * sizeof(GpuDab))) {
        bindings[12] = d.dabs;
        k.source_kind = kSourceDabs;
        k.source_count = static_cast<uint32_t>(dabs.size());
        run(kModeStamp, d.tile_count);
        bindings[12] = d.dummy_events;
    }
    for (RayTrophiSim::ParticleSimulationSystem* system : device_sources) {
        RayTrophiSim::ParticleDeviceEventView view;
        if (!system->deviceCollisionEventView(c, view)) continue;
        // Colour table: this system's LUT rows and id -> row map.
        std::vector<float> lut;
        std::vector<uint32_t> lookup;
        uint32_t max_id = 0;
        for (const auto& profile : system->appearanceProfiles()) max_id = std::max(max_id, profile.id);
        lookup.assign(static_cast<std::size_t>(max_id) + 1, 0xFFFFFFFFu);
        for (const auto& profile : system->appearanceProfiles()) {
            const float* row = system->appearanceLutRow(profile.id);
            if (!row) continue;
            lookup[profile.id] = static_cast<uint32_t>(lut.size() / RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
            lut.insert(lut.end(), row, row + RayTrophiSim::kParticleAppearanceLutFloatsPerRow);
        }
        const bool tables = !lut.empty() &&
            ensure(d.lut, d.lut_capacity, lut.size(), sizeof(float), "WetPaintLut") &&
            ensure(d.lookup, d.lookup_capacity, lookup.size(), sizeof(uint32_t), "WetPaintLookup") &&
            c.uploadBuffer(d.lut, lut.data(), lut.size() * sizeof(float)) &&
            c.uploadBuffer(d.lookup, lookup.data(), lookup.size() * sizeof(uint32_t));
        bindings[13] = view.events;
        bindings[14] = view.counter;
        bindings[15] = tables ? d.lut : d.dummy_events;
        bindings[16] = tables ? d.lookup : d.dummy_counter;
        k.source_kind = kSourceEvents;
        k.source_count = 0;
        k.event_capacity = view.capacity;
        k.lut_rows = tables ? static_cast<uint32_t>(lut.size() / RayTrophiSim::kParticleAppearanceLutFloatsPerRow) : 0u;
        k.lookup_count = tables ? static_cast<uint32_t>(lookup.size()) : 0u;
        run(kModeStamp, d.tile_count);
        run(kModeConsume, 1);   // the consume: counter back to zero
        ++stats_.device_event_dispatches;
        bindings[13] = d.dummy_events;
        bindings[14] = d.dummy_counter;
        bindings[15] = d.dummy_events;
        bindings[16] = d.dummy_counter;
    }

    run(kModeTick, d.tile_count);
    run(kModeResolve, d.tile_count);
    if (d.channels_allocated) {
        if (d.channels_force) run(kModeSeedHeight, d.tile_count);
        const bool enabled = std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
            return s->settings.roughness_channel || s->settings.height_normal_channel;
        });
        if (enabled || d.channels_force) {
            run(kModeChannels, d.tile_count);
            ++stats_.pbr_dispatches;
        }
    }
    run(kModeRotate, (d.tile_count + 255u) / 256u);
    // A periodic reader got a stale coverage: measure it here, after this
    // tick's final state, inside the readback this tick pays anyway (the
    // stats_init upload above zeroed the coverage slots too).
    const bool measure_coverage = coverage_requested_;
    if (measure_coverage) run(kModeCoverage, d.tile_count);

    const bool force = d.channels_force;
    d.channels_force = false;   // consumed by the dispatches above
    if (!ok) {
        // ★ Still hand the recorded work over: on wet paint's own context
        // nobody else submits it, and a recording left open forever is the
        // "kernels never ran" failure.
        c.submitAsync();
        stats_.status = "dispatch_failed";
        stats_.last_tick_ms = msSince(t0);
        return false;
    }
    stats_.active_tiles = stats_.ticked_tiles = stats_.stamped_texels = 0;
    for (uint32_t si = 0; si < n; ++si) {
        Surface& s = *surfaces_[si];
        const uint32_t* v = st.data() + si * kStatsStride;
        s.last.active_tiles = v[kStActiveTiles];
        s.last.ticked_tiles = v[kStTickedTiles];
        s.last.stamped_texels = v[kStStampedTexels];
        s.last.sources_dropped = v[kStSourcesDropped];
        stats_.active_tiles += v[kStActiveTiles];
        stats_.ticked_tiles += v[kStTickedTiles];
    const std::size_t stats_bytes = static_cast<std::size_t>(n) * kStatsStride * 4;
    const uint64_t ticket = c.submitWithReadback(d.stats, stats_bytes, 0);
    if (ticket != 0) {
        d.stats_ticket = ticket;
        d.stats_surfaces = n;
        d.stats_force = force;
        d.stats_coverage = measure_coverage;
        d.stats_serial = paint_serial_;
        if (measure_coverage) coverage_requested_ = false;   // its answer is on the way
        ++stats_.stats_delayed;
    } else {
        // No delayed readback on this backend: the old synchronous read.
        // ★ downloadBuffer's immediate path drains IN-FLIGHT work only; the
        // recorded dispatches must be submitted first.
        std::vector<uint32_t> st(static_cast<std::size_t>(n) * kStatsStride, 0u);
        c.synchronize();
        if (!c.downloadBuffer(d.stats, st.data(), stats_bytes)) {
            stats_.status = "stats_download_failed";
            stats_.last_tick_ms = msSince(t0);
            return false;
        }
        applyStats(st, force, measure_coverage, paint_serial_);
    }
    ++stats_.ticks;
    stats_.last_dt = dt;
    stats_.status = "running";
    stats_.last_tick_ms = msSince(t0);
    return true;
}

// The stats of one dispatching tick: per-surface counters, dirty rectangles
// into the publish queues, and the coverage that tick measured.
void WetPaintGpu::applyStats(const std::vector<uint32_t>& st, bool force, bool measured_coverage,
                             uint64_t serial) {
    const uint32_t n = static_cast<uint32_t>(surfaces_.size());
    if (st.size() < static_cast<std::size_t>(n) * kStatsStride) return;
    if (measured_coverage) {
        readCoverage(st, coverage_cache_);
        coverage_serial_ = serial;
        ++stats_.coverage_in_tick;
    }
        stats_.stamped_texels += v[kStStampedTexels];
        stats_.sources_dropped += v[kStSourcesDropped];
        if (v[kStDirtyMaxX] >= v[kStDirtyMinX] && v[kStDirtyMinX] != 0xFFFFFFFFu) {
            const int x0 = static_cast<int>(v[kStDirtyMinX] * kTile);
            const int y0 = static_cast<int>(v[kStDirtyMinY] * kTile);
            const int x1 = std::min(static_cast<int>(s.width) - 1, static_cast<int>((v[kStDirtyMaxX] + 1) * kTile) - 1);
            const int y1 = std::min(static_cast<int>(s.height) - 1, static_cast<int>((v[kStDirtyMaxY] + 1) * kTile) - 1);
            const auto merge = [&](Surface::Rect& r) {
                if (!r.any) { r.any = true; r.x0 = x0; r.y0 = y0; r.x1 = x1; r.y1 = y1; }
                else { r.x0 = std::min(r.x0, x0); r.y0 = std::min(r.y0, y0);
                       r.x1 = std::max(r.x1, x1); r.y1 = std::max(r.y1, y1); }
            };
            for (auto& r : s.pending) merge(r);
            for (int channel = 0; channel < 3; ++channel) {
                const bool enabled = channel == 0 ? s.settings.roughness_channel
                                                  : s.settings.height_normal_channel;
                if ((enabled || s.channels_refresh || force) &&
                    !s.channel_textures[channel].expired()) {
                    for (auto& pending : s.channel_pending) merge(pending[channel]);
                }
            }
            s.channels_refresh = false;
        }
    }
}

// ── publish ─────────────────────────────────────────────────────────────────

int WetPaintGpu::publish(Backend::IBackend* render_backend, Backend::IBackend* viewport_backend) {
    host_materials_.clear();
    if (!device_ || !device_->compute || layout_dirty_ || surfaces_.empty()) return 0;
    Device& d = *device_;
    SimulationComputeContext& c = *d.compute;
    if (!sameCompute(d.output, c)) return 0;
    Backend::IBackend* targets[2] = {render_backend,
                                     viewport_backend != render_backend ? viewport_backend : nullptr};
    void* vk_device = c.nativeDevice();
    void* vk_buffer = c.nativeBufferPtr(d.output);
    bool any_pending = false;
    for (const auto& s : surfaces_) {
        any_pending = any_pending || s->pending[0].any || s->pending[1].any;
        for (const auto& per_backend : s->channel_pending) {
            for (const auto& r : per_backend) any_pending = any_pending || r.any;
        }
    }
    if (!any_pending) return 0;
    int changed = 0;
    // The device copy reads what the dispatches wrote. Submitted, not waited
    // for: the copy batch is queued after them on the same queue and orders
    // itself with a barrier (VulkanBackendAdapter::beginDeviceTextureCopies).
    // The rectangles come from the previous tick's stats; the buffer already
    // holds this tick's paint too, so a copy is never older than its rectangle.
    c.submitAsync();
    Backend::VulkanBackendAdapter* batches[2] = {};
    for (int b = 0; b < 2; ++b) {
        batches[b] = dynamic_cast<Backend::VulkanBackendAdapter*>(targets[b]);
        if (batches[b]) batches[b]->beginDeviceTextureCopies();
    }
    struct EndBatches {
        Backend::VulkanBackendAdapter** list;
        ~EndBatches() { for (int b = 0; b < 2; ++b) if (list[b]) list[b]->endDeviceTextureCopies(); }
    } end_batches{batches};

    // One plane of the output atlas (0 colour, 1 roughness, 2 height mask,
    // 3 normal) into the texture bound to that material slot. Planes above 0
    // exist only once the channel atlas is allocated.
    const auto publishPlane = [&](Surface& s, const std::shared_ptr<Texture>& tex, uint32_t plane,
                                  Surface::Rect& render_rect, Surface::Rect& viewport_rect) {
        Surface::Rect* rects[2] = {&render_rect, &viewport_rect};
        if (!tex || (plane > 0 && !d.channels_allocated)) {
            for (Surface::Rect* r : rects) *r = Surface::Rect{};
            return;
        }
        const std::size_t plane_offset = static_cast<std::size_t>(plane) * d.texel_count + s.page_offset;
        bool host_path = false;
        for (int b = 0; b < 2; ++b) {
            Surface::Rect& r = *rects[b];
            if (!r.any) continue;
            auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(targets[b]);
            if (!vk) { r = Surface::Rect{}; continue; }   // no Vulkan target in this slot
            const int rc = (vk_device && vk_buffer)
                ? vk->copyDeviceBufferToTexture(tex.get(), vk_device, vk_buffer,
                                                static_cast<uint64_t>(plane_offset) * 4ull, s.width,
                                                r.x0, r.y0, static_cast<uint32_t>(r.x1 - r.x0 + 1),
                                                static_cast<uint32_t>(r.y1 - r.y0 + 1))
                : -1;
            if (rc == 1) {
                if (plane == 0) ++stats_.publish_device_copies;
                else ++stats_.pbr_device_copies;
                stats_.publish_mode = "device_copy";
                changed |= kWetPaintPublishedDevice;
                r = Surface::Rect{};
            } else if (rc < 0) {
                host_path = true;
            }
            // rc == 0: the texture is not resident on that backend yet; keep the rectangle.
        }
        if (!host_path) return;
        // Host path: whole rows of the union rectangle into the texture pixels,
        // then the normal dirty-region upload (both backends read these pixels).
        int y0 = static_cast<int>(s.height), y1 = -1, x0 = static_cast<int>(s.width), x1 = -1;
        for (const Surface::Rect* r : rects) {
            if (!r->any) continue;
            y0 = std::min(y0, r->y0); y1 = std::max(y1, r->y1);
            x0 = std::min(x0, r->x0); x1 = std::max(x1, r->x1);
        }
        if (y1 < y0) return;
        const std::size_t rows = static_cast<std::size_t>(y1 - y0 + 1);
        std::vector<uint32_t> data(rows * s.width);
        if (!c.downloadBuffer(d.output, data.data(), data.size() * 4,
                              (plane_offset + static_cast<std::size_t>(y0) * s.width) * 4)) {
            return;
        }
        for (std::size_t j = 0; j < rows; ++j) {
            for (int x = x0; x <= x1; ++x) {
                const uint32_t v = data[j * s.width + x];
                CompactVec4& px = tex->pixels[(static_cast<std::size_t>(y0) + j) * s.width + x];
                px.r = static_cast<uint8_t>(v & 0xFFu);
                px.g = static_cast<uint8_t>((v >> 8) & 0xFFu);
                px.b = static_cast<uint8_t>((v >> 16) & 0xFFu);
                px.a = static_cast<uint8_t>((v >> 24) & 0xFFu);
            }
        }
        tex->upload_region_to_gpu(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
        // The dirty region only travels to a Vulkan image when the material
        // sync walks it (Renderer::updateBackendMaterials); the caller runs it
        // on kWetPaintPublishedHost, like the CPU brush does after a stroke.
        // (Studio: the material-preview viewport is a second VkDevice, so this
        // host path is its only path; the RT render device gets the device copy.)
        ++stats_.publish_host_downloads;
        stats_.publish_bytes_downloaded += data.size() * 4;
        stats_.publish_mode = "host_download";
        for (Surface::Rect* r : rects) *r = Surface::Rect{};
        if (std::find(host_materials_.begin(), host_materials_.end(),
                      static_cast<uint16_t>(s.material_id)) == host_materials_.end()) {
            host_materials_.push_back(static_cast<uint16_t>(s.material_id));
        }
        changed |= kWetPaintPublishedHost;
    };

    for (auto& sp : surfaces_) {
        Surface& s = *sp;
        publishPlane(s, s.texture.lock(), 0u, s.pending[0], s.pending[1]);
        for (uint32_t channel = 0; channel < 3; ++channel) {
            if (!s.channel_pending[0][channel].any && !s.channel_pending[1][channel].any) continue;
            publishPlane(s, s.channel_textures[channel].lock(), channel + 1u,
                         s.channel_pending[0][channel], s.channel_pending[1][channel]);
        }
    }
    return changed;
}

// ── coverage ────────────────────────────────────────────────────────────────

void WetPaintGpu::readCoverage(const std::vector<uint32_t>& st, WetPaintCoverage& out) const {
    out = WetPaintCoverage{};
    const uint32_t n = static_cast<uint32_t>(surfaces_.size());
    for (uint32_t si = 0; si < n; ++si) {
        const uint32_t* v = st.data() + si * kStatsStride;
        WetPaintCoverage::Surface s;
        s.object = surfaces_[si]->object;
        s.material_id = surfaces_[si]->material_id;
        s.covered_texels = v[kStCoveredTexels];
        s.valid_texels = v[kStValidTexels];
        s.wet_texels = v[kStWetTexels];
        s.roughness_texels = v[kStRoughnessTexels];
        s.normal_texels = v[kStNormalTexels];
        s.ratio = s.valid_texels ? static_cast<float>(s.covered_texels) / static_cast<float>(s.valid_texels) : 0.0f;
        if (s.covered_texels > 0) {
            s.mean_color[0] = static_cast<float>(v[kStSumR]) / (255.0f * s.covered_texels);
            s.mean_color[1] = static_cast<float>(v[kStSumG]) / (255.0f * s.covered_texels);
            s.mean_color[2] = static_cast<float>(v[kStSumB]) / (255.0f * s.covered_texels);
        }
        out.covered_texels += s.covered_texels;
        out.valid_texels += s.valid_texels;
        out.surfaces.push_back(s);
    }
    out.ratio = out.valid_texels ? static_cast<float>(static_cast<double>(out.covered_texels) /
                                                      static_cast<double>(out.valid_texels)) : 0.0f;
}

bool WetPaintGpu::coverage(SceneData& scene, WetPaintCoverage& out, bool allow_stale) {
    out = WetPaintCoverage{};
    if (surfaces_.empty() || !ensureCompute(scene)) return false;
    if (layout_dirty_ && !rebuildDevice(scene)) return false;
    if (!prepareChannels(scene)) return false;
    if (settings_dirty_) uploadSurfaceTable();
    Device& d = *device_;
    SimulationComputeContext& c = *d.compute;
    const uint32_t n = static_cast<uint32_t>(surfaces_.size());
    std::vector<uint32_t> st(static_cast<std::size_t>(n) * kStatsStride, 0u);
    // Only the coverage slots are cleared; the tick slots keep their values.
    for (uint32_t si = 0; si < n; ++si) {
        c.uploadBuffer(d.stats, st.data(), (kStatsStride - kStCoveredTexels) * 4,
    // Unchanged paint (every tick since the last measurement was idle, no
    // layout or settings change): the answer is the same. The room goal asks
    // twice a second; each measurement is a dispatch + synchronize + download.
    if (!layout_dirty_ && !settings_dirty_ && coverage_serial_ == paint_serial_) {
        out = coverage_cache_;
        ++stats_.coverage_cached;
        return true;
    }
    if (allow_stale && !layout_dirty_ && !settings_dirty_ && coverage_serial_ != ~0ull) {
        out = coverage_cache_;   // at most one request old
        coverage_requested_ = true;
        ++stats_.coverage_cached;
        return true;
    }
                       (static_cast<std::size_t>(si) * kStatsStride + kStCoveredTexels) * 4);
    }
    ComputeBufferHandle bindings[kBindingCount] = {
        d.surfaces, d.tiles, d.pos_valid, d.flow_normal, d.base, d.color, d.state,
        d.color_scratch, d.state_scratch, d.output, d.tile_flags, d.stats,
        d.dummy_events, d.dummy_events, d.dummy_counter, d.dummy_events, d.dummy_counter,
    };
    WetPaintGpuConstants k;
    k.mode = kModeCoverage;
    k.surface_count = n;
    k.tile_count = d.tile_count;
    k._p0 = d.channels_allocated ? d.texel_count : 0u;
    ComputeDispatch cmd;
    cmd.kernel = "paint_wet";
    cmd.buffers = bindings;
    cmd.buffer_count = kBindingCount;
    cmd.constants = &k;
    cmd.constants_size = sizeof(k);
    cmd.groups.groups_x = std::max<uint32_t>(1u, d.tile_count);
    if (!c.dispatch(cmd)) return false;
    c.synchronize();   // see tick(): the download does not submit recorded work
    if (!c.downloadBuffer(d.stats, st.data(), st.size() * 4)) return false;
    readCoverage(st, out);
    coverage_cache_ = out;
    coverage_serial_ = paint_serial_;
    return true;
}

} // namespace Paint
