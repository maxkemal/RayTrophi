// GPU wet paint atlas: bake, device buffers, per-frame stamp/tick, publish.
// See include/Paint/WetPaintGpu.h and shaders/paint_wet.comp.

#include "Paint/WetPaintGpu.h"
#include "Paint/WetPaintGeometryBake.h"
#include "Paint/WetPaintMobileBudget.h"
#include "WetPaintChannelEdits.h"
#include "Paint/HiddenGameObjects.h"

#include "Backend/IBackend.h"
#include "Backend/VulkanBackend.h"
#include "Paint/MeshPaintAdapter.h"
#include "Paint/PaintHeightNormalMath.h"
#include "MaterialManager.h"
#include "Material.h"
#include "PrincipledBSDF.h"
#include "ParticleSimulation.h"
#include "PerfProfile.h"
#include "SimulationComputeVulkanContext.h"
#include "SimulationWorld.h"
#include "SurfaceFlowField.h"
#include "Texture.h"
#include "ProjectManager.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "globals.h"
#include "scene_data.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>

namespace Paint {

using RayTrophiSim::ComputeBackendType;
using RayTrophiSim::ComputeBufferDesc;
using RayTrophiSim::ComputeBufferHandle;
using RayTrophiSim::ComputeBufferUsage;
using RayTrophiSim::ComputeDispatch;
using RayTrophiSim::SimulationComputeContext;

namespace {

constexpr uint32_t kTile = 32u;
// The drying/runoff simulation steps at most this often (2026-11-24). The
// kernel's rates are written per 60 Hz frame (pc.dt * 60), so a 60 Hz step is
// its reference cadence, not an approximation. Before, it stepped every
// rendered frame: 300-500 Hz on a desktop, where tick+resolve+channels (+ a
// texture publish per dirty plane) cost 0.6 ms per frame for one pop and
// 1.6 ms for four - for changes no one can see between two 60 Hz steps.
// STAMP and CONSUME still run every frame (contact events must not pile up);
// a stamp shows at the next step, at most one 60 Hz step later. Below 60 fps
// every frame is a step, exactly as before.
constexpr double kWetStepSeconds = WetPaintMobileBudget::kStepSeconds;
constexpr uint32_t kStatsStride = 18u;
constexpr uint32_t kBindingCount = 17u;
constexpr int kGutterTexels = 2;
// A painted surface's base follows its material colour once that colour has
// stayed put this long (a colour picker drag rebuilds nothing until it stops).
constexpr double kBaseSettleSeconds = 0.25;

enum WetPaintMode : uint32_t {
    kModeStamp = 0, kModeTick = 1, kModeResolve = 2, kModeRotate = 3,
    kModeConsume = 4, kModeCoverage = 5, kModeReset = 6,
    kModeChannels = 7, kModeSeedHeight = 8,
    kModeCompactChannels = 9, kModeChannelsList = 10,
};
constexpr const char* kWetPaintTimingLabels[] = {
    "paint_wet.stamp", "paint_wet.tick", "paint_wet.resolve", "paint_wet.rotate",
    "paint_wet.consume", "paint_wet.coverage", "paint_wet.reset",
    "paint_wet.channels", "paint_wet.seed_height", "paint_wet.channels_compact",
    "paint_wet.channels",
};
constexpr uint32_t kCompactChannels = 4u;
constexpr uint32_t kPictureBit = 8u;          // pc._p1: source plane 6 (hidden picture) exists
constexpr uint32_t kResetInitOnlyBit = 16u;   // pc._p1: MODE_RESET zeroes paint/scratch buffers only
constexpr uint32_t kSurfacePicture = 16u;     // GpuSurface.flags: this surface has a picture
constexpr uint32_t kSurfaceOverlay = 32u;     // GpuSurface.flags: premultiplied canvas over a tiled base
constexpr uint32_t kPicturePlanes = 7u;       // source atlas planes when a picture exists
constexpr std::size_t kIndirectGroupCountWords = 3u;
constexpr std::size_t channelDispatchOffset(uint32_t tiles) {
    return static_cast<std::size_t>(tiles) * 2u * sizeof(uint32_t);
}
enum WetPaintSource : uint32_t { kSourceDabs = 0, kSourceEvents = 1 };
enum StatsSlot : uint32_t {
    kStActiveTiles = 0, kStDirtyMinX = 1, kStDirtyMinY = 2, kStDirtyMaxX = 3, kStDirtyMaxY = 4,
    kStStampedTexels = 5, kStTickedTiles = 6, kStSourcesDropped = 7, kStCoveredTexels = 8,
    kStValidTexels = 9, kStSumR = 10, kStSumG = 11, kStSumB = 12, kStWetTexels = 13,
    kStRoughnessTexels = 14, kStNormalTexels = 15,
    kStPictureTexels = 16, kStPictureRevealed = 17,
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

// Loads a hidden-picture image and resamples it (bilinear, premultiplied-free
// per channel) to the surface page. Row convention = Surface::base: page row y
// is texture row y (Texture::pixels is row 0 first, exactly what base copies).
bool loadPictureImage(const std::string& path, uint32_t width, uint32_t height,
                      std::vector<uint32_t>& out) {
    out.clear();
    if (path.empty() || width == 0 || height == 0) return true;
    namespace fs = std::filesystem;
    std::string resolved = path;
    std::error_code ec;
    if (!fs::exists(fs::path(path), ec)) {
        const std::string project = ProjectManager::getInstance().getCurrentFilePath();
        if (!project.empty() && fs::path(path).is_relative()) {
            const fs::path candidate = fs::path(project).parent_path() / path;
            if (fs::exists(candidate, ec)) resolved = candidate.string();
        }
    }
    Texture image(resolved, TextureType::Albedo);
    if (!image.is_loaded() || image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<size_t>(image.width) * image.height) {
        return false;
    }
    const int sw = image.width, sh = image.height;
    const int dst_h = static_cast<int>(height);
    out.resize(static_cast<std::size_t>(width) * height);
#pragma omp parallel for schedule(static)
    for (int y = 0; y < dst_h; ++y) {
        const float fy = std::clamp((static_cast<float>(y) + 0.5f) * sh / height - 0.5f, 0.0f,
                                    static_cast<float>(sh - 1));
        const int y0 = static_cast<int>(fy), y1 = std::min(y0 + 1, sh - 1);
        const float ty = fy - y0;
        for (uint32_t x = 0; x < width; ++x) {
            const float fx = std::clamp((static_cast<float>(x) + 0.5f) * sw / width - 0.5f, 0.0f,
                                        static_cast<float>(sw - 1));
            const int x0 = static_cast<int>(fx), x1 = std::min(x0 + 1, sw - 1);
            const float tx = fx - x0;
            const CompactVec4 p[4] = {image.pixels[static_cast<size_t>(y0) * sw + x0],
                                      image.pixels[static_cast<size_t>(y0) * sw + x1],
                                      image.pixels[static_cast<size_t>(y1) * sw + x0],
                                      image.pixels[static_cast<size_t>(y1) * sw + x1]};
            const float w[4] = {(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty};
            float c[4] = {0, 0, 0, 0};
            for (int k = 0; k < 4; ++k) {
                c[0] += p[k].r * w[k]; c[1] += p[k].g * w[k];
                c[2] += p[k].b * w[k]; c[3] += p[k].a * w[k];
            }
            const auto b = [](float v) {
                return static_cast<uint8_t>(std::clamp(v + 0.5f, 0.0f, 255.0f));
            };
            out[static_cast<size_t>(y) * width + x] =
                static_cast<uint32_t>(b(c[0])) | (static_cast<uint32_t>(b(c[1])) << 8) |
                (static_cast<uint32_t>(b(c[2])) << 16) | (static_cast<uint32_t>(b(c[3])) << 24);
        }
    }
    return true;
}

uint32_t packRgba8(const CompactVec4& c) {
    return static_cast<uint32_t>(c.r) | (static_cast<uint32_t>(c.g) << 8) |
           (static_cast<uint32_t>(c.b) << 16) | (static_cast<uint32_t>(c.a) << 24);
}

CompactVec4 unpackRgba8(uint32_t c) {
    return CompactVec4(static_cast<uint8_t>(c & 255u), static_cast<uint8_t>((c >> 8) & 255u),
                       static_cast<uint8_t>((c >> 16) & 255u), static_cast<uint8_t>(c >> 24));
}

// A texture transform that maps UV to itself (the wrap mode is irrelevant: a
// canvas covers 0..1 once either way).
bool identityTransform(const PrincipledBSDF::TextureTransform& t) {
    const auto one = [](double v) { return std::abs(v - 1.0) <= 1e-4; };
    const auto zero = [](double v) { return std::abs(v) <= 1e-4; };
    return one(t.scale.u) && one(t.scale.v) && one(t.tilingFactor.u) && one(t.tilingFactor.v) &&
           zero(t.translation.u) && zero(t.translation.v) && zero(t.rotation_degrees);
}

// `texture` stretched once over a w x h page (the canvas seed's mapping,
// cloneTextureForPaint), opaque.
std::vector<uint32_t> stretchedBase(const Texture& texture, uint32_t w, uint32_t h) {
    std::vector<uint32_t> out(static_cast<std::size_t>(w) * h, packRgba8(CompactVec4(255, 255, 255, 255)));
    const int sw = texture.width, sh = texture.height;
    if (sw <= 0 || sh <= 0 || texture.pixels.size() < static_cast<std::size_t>(sw) * sh || w < 2 || h < 2) {
        return out;
    }
    for (uint32_t y = 0; y < h; ++y) {
        const float fy = static_cast<float>(y) / static_cast<float>(h - 1) * static_cast<float>(sh - 1);
        const int y0 = std::clamp(static_cast<int>(fy), 0, sh - 1), y1 = std::min(y0 + 1, sh - 1);
        const float ty = fy - static_cast<float>(y0);
        for (uint32_t x = 0; x < w; ++x) {
            const float fx = static_cast<float>(x) / static_cast<float>(w - 1) * static_cast<float>(sw - 1);
            const int x0 = std::clamp(static_cast<int>(fx), 0, sw - 1), x1 = std::min(x0 + 1, sw - 1);
            const float tx = fx - static_cast<float>(x0);
            const CompactVec4& a = texture.pixels[static_cast<std::size_t>(y0) * sw + x0];
            const CompactVec4& b = texture.pixels[static_cast<std::size_t>(y0) * sw + x1];
            const CompactVec4& c = texture.pixels[static_cast<std::size_t>(y1) * sw + x0];
            const CompactVec4& d = texture.pixels[static_cast<std::size_t>(y1) * sw + x1];
            const auto lerp = [&](uint8_t pa, uint8_t pb, uint8_t pc, uint8_t pd) {
                const float top = pa + (pb - pa) * tx, bottom = pc + (pd - pc) * tx;
                return static_cast<uint8_t>(std::clamp(std::lround(top + (bottom - top) * ty), 0L, 255L));
            };
            out[static_cast<std::size_t>(y) * w + x] =
                packRgba8(CompactVec4(lerp(a.r, b.r, c.r, d.r), lerp(a.g, b.g, c.g, d.g),
                                      lerp(a.b, b.b, c.b, d.b), 255));
        }
    }
    return out;
}

PrincipledBSDF* paintMaterial(int material_id) {
    return dynamic_cast<PrincipledBSDF*>(
        MaterialManager::getInstance().getMaterial(static_cast<uint16_t>(material_id)));
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

// Every distinct material slot an object's faces use, in first-use order.
std::vector<uint16_t> objectPaintMaterials(SceneData& scene, const std::string& object) {
    std::vector<uint16_t> out;
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
            if (std::find(out.begin(), out.end(), material) == out.end()) {
                out.push_back(material);
            }
        }
    }
    return out;
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
        const bool instance_pose = hiddenObjectUsesInstancePose(mesh->nodeName) &&
            mesh->transform && !mesh->hasSkinWeights() && geometry.get_positions_orig();
        const auto* positions = instance_pose
            ? geometry.get_positions_orig() : geometry.get_positions();
        const auto* normals = instance_pose ? geometry.get_normals_orig() : geometry.get_normals();
        const size_t normal_count = geometry.get_core_attribute_count(
            instance_pose ? DNA::Attr::N_orig : DNA::Attr::N);
        const auto world = instance_pose ? mesh->transform->getFinal() : Matrix4x4::identity();
        const auto normal_matrix = instance_pose ? world.inverse().transpose()
                                                : Matrix4x4::identity();
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
            const size_t count = instance_pose ? geometry.get_positions_orig_count()
                                               : geometry.get_vertex_count();
            if (indices[0] >= count || indices[1] >= count || indices[2] >= count ||
                (materials ? materials[indices[0]] : 0) != material_id) {
                continue;
            }
            FlatPaintFace face;
            for (size_t i = 0; i < 3; ++i) {
                face.positions[i] = world.transform_point(positions[indices[i]]);
                face.normals[i] = normals && indices[i] < normal_count
                    ? normal_matrix.transform_vector(normals[indices[i]]) : Vec3(0.0f);
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
    bool base_from_color = false;      // base = the material colour (no texture under it)
    // Canvas over a tiled base (docs/dev/WET_PAINT_CANVAS_UV.md). The material
    // keeps its UV transform for base_texture (PrincipledBSDF::paintBaseTexture),
    // the output page is the premultiplied paint alone, and `base` holds only
    // the pickup colour (alpha 0). Back in composite mode `base` is
    // base_texture stretched once over the page, as the canvas was seeded.
    bool overlay = false;
    std::shared_ptr<Texture> base_texture;   // the material's own base colour texture
    Vec3 base_linear = Vec3(1.0f);     // that colour when the base was filled
    bool geometry_upload = false; // Same dimensions: patch pages without retiring GPU handles.
    bool base_upload = false;          // base changed on the host: rewrite its device slice
    bool base_pending = false;         // material colour moved, waiting for it to settle
    Vec3 base_pending_color = Vec3(0.0f);
    std::chrono::steady_clock::time_point base_pending_since{};
    std::array<std::weak_ptr<Texture>, 4> channel_textures;
    std::array<std::vector<uint32_t>, 4> channel_base; // roughness, height, normal, opacity
    std::vector<Vec2> normal_spans;    // same Jacobian metric used by CPU Paint
    // Hidden picture (RGBA8 sRGB at the page size, empty = none). Source plane 6.
    std::vector<uint32_t> picture;
    std::string picture_path_loaded;
    // "@base" mode: the original base pixels, kept so the wall can be restored.
    // Non-empty exactly while s.base holds the flat blank colour.
    std::vector<uint32_t> base_original;
    bool picture_upload = false;
    bool channels_refresh = false;
    uint32_t edited_planes = 0;
    std::vector<GpuTile> tiles;        // surface field filled at layout
    uint32_t valid_texels = 0, gutter_texels = 0, triangles = 0;
    WetPaintSurfaceSettings settings;
    int uv_set = 0;
    uint16_t paint_material = 0;
    std::array<double, 24> geometry_key{};   // geometryKey() at the last bake
    struct CachedMeshTransform {
        const TriangleMesh* mesh = nullptr;
        Matrix4x4 transform = Matrix4x4::identity();
    };
    std::vector<CachedMeshTransform> cached_transforms;
    bool geometry_fast_valid = false;
    // Dirty texel rectangle not yet in each backend (0 render, 1 viewport).
    struct Rect { bool any = false; int x0 = 0, y0 = 0, x1 = -1, y1 = -1; };
    Rect pending[2];
    Rect channel_pending[2][4];
    WetPaintSurfaceInfo last;
};

struct WetPaintGpu::Device {
    SimulationComputeContext* compute = nullptr;
    ComputeBackendType backend = ComputeBackendType::CPU;
    ComputeBufferHandle surfaces, tiles, pos_valid, flow_normal, base, color, state,
        color_scratch, state_scratch, output, tile_flags, stats, dabs,
        dummy_events, dummy_counter;
    std::size_t dab_capacity = 0;

    struct AppearanceTables {
        std::weak_ptr<RayTrophiSim::ParticleSimulationSystem> owner;
        ComputeBufferHandle lut, lookup;
        std::size_t lut_capacity = 0, lookup_capacity = 0;
        uint32_t lut_rows = 0, lookup_count = 0;
        std::optional<uint64_t> revision;

        void release(SimulationComputeContext& context) {
            for (const auto& handle : {lut, lookup}) {
                if (sameCompute(handle, context)) {
                    context.destroyBuffer(handle);
                }
            }
        }
    };
    std::unordered_map<const RayTrophiSim::ParticleSimulationSystem*, AppearanceTables>
        appearance_tables;
    uint32_t texel_count = 0, tile_count = 0;
    bool channels_allocated = false;
    bool reveal_allocated = false;
    bool picture_allocated = false;
    bool channels_force = false;
    // Stats of dispatching ticks, read back later
    // (SimulationComputeContext::submitWithReadback), each with what its tick
    // knew. Oldest first; collected without waiting while they are not ready.
    struct PendingStats {
        uint64_t ticket = 0;
        uint32_t surfaces = 0;
        bool force = false;        // channels_force at that dispatch
        bool coverage = false;     // that tick measured coverage
        uint64_t serial = 0;       // paint_serial_ of that tick
    };
    std::deque<PendingStats> stats_pending;
    // Fixed-rate simulation (2026-11-24, see kWetStepSeconds): wall time not
    // yet simulated, and whether the stats slots must be zeroed before the
    // next dispatch (once per readback window, NOT per frame: stamp-only
    // frames add their counters into the window the next step reads back).
    double step_clock = 0.0;
    double stamp_clock = WetPaintMobileBudget::kStampIntervalSeconds;
    double coverage_clock = WetPaintMobileBudget::kMinCoverageIntervalSeconds;
    bool stats_reset_needed = true;
    bool zero_init_needed = false;
    uint64_t last_scene_geometry_gen = ~0ull;
    std::size_t last_world_object_count = 0;
    std::size_t last_surface_count = 0;
    std::vector<const Surface*> last_geometry_surfaces;
    std::vector<uint32_t> stats_prev_scratch;
    std::vector<uint32_t> stats_init_scratch;

    void release() {
        stats_pending.clear();   // a dropped ticket's slot is reused by the backend
        last_scene_geometry_gen = ~0ull;
        last_world_object_count = 0;
        last_surface_count = 0;
        last_geometry_surfaces.clear();
        if (compute) {
            for (auto& entry : appearance_tables) {
                entry.second.release(*compute);
            }
            for (ComputeBufferHandle* h : {&surfaces, &tiles, &pos_valid, &flow_normal, &base, &color,
                                           &state, &color_scratch, &state_scratch, &output,
                                           &tile_flags, &stats, &dabs,
                                           &dummy_events, &dummy_counter}) {
                if (sameCompute(*h, *compute)) compute->destroyBuffer(*h);
                *h = {};
            }
        }
        appearance_tables.clear();
        dab_capacity = 0;
        texel_count = tile_count = 0;
        channels_allocated = channels_force = reveal_allocated = picture_allocated = false;
        zero_init_needed = false;
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
std::array<double, 24> WetPaintGpu::geometryKey(SceneData& scene, const std::string& object) {
    std::array<double, 24> key{};
    key[5] = key[6] = key[7] = 0.0;
    double mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30};
    for (const auto& obj : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
        if (!mesh || mesh->nodeName != object || !mesh->geometry) continue;
        const auto& geometry = *mesh->geometry;
        const bool instance_pose = hiddenObjectUsesInstancePose(mesh->nodeName) &&
            mesh->transform && !mesh->hasSkinWeights() && geometry.get_positions_orig();
        const auto* positions = instance_pose
            ? geometry.get_positions_orig() : geometry.get_positions();
        const size_t count = instance_pose ? geometry.get_positions_orig_count()
                                           : geometry.get_vertex_count();
        const auto world = instance_pose ? mesh->transform->getFinal() : Matrix4x4::identity();
        if (instance_pose) {
            // Exact pose components also detect symmetric rotations whose
            // position sum/AABB stay unchanged while the UV-facing normals move.
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    key[8 + row * 4 + column] += world.m[row][column];
                }
            }
        }
        if (!positions) continue;
        key[0] += static_cast<double>(count);
        // The atlas IS the mesh's UV layout: a re-unwrap with the same
        // positions moves every texel, so the UVs are part of the key (folded
        // into slot 0; the key is compared for equality only).
        if (const Vec2* uvs = mesh->geometry->get_uvs()) {
            if (mesh->geometry->get_core_attribute_count(DNA::Attr::UV) == count) {
                for (size_t i = 0; i < count; ++i) {
                    key[0] += 1.000123 * uvs[i].u + 2.000457 * uvs[i].v;
                }
            }
        }
        for (size_t i = 0; i < count; ++i) {
            const Vec3 v = world.transform_point(positions[i]);
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
// world transform in the geometry. Managed hidden objects instead use local
// P_orig/N_orig + current instance pose, matching the SDF resolver and renderer.
// geometryKey reads that same world space to trigger rebaking after a move.
bool WetPaintGpu::bakeGeometry(SceneData& scene, Surface& s, std::string& error) {
    s.geometry_key = geometryKey(scene, s.object);
    s.geometry_fast_valid = false;
    const int uv_set = s.uv_set;
    const auto faces = paintFaces(scene, s.object, s.paint_material, uv_set);
    std::vector<WetPaintBakeFace> bake_faces;
    bake_faces.reserve(faces.size());
    for (const auto& f : faces) {
        bake_faces.push_back({f.positions, f.normals, f.uvs});
    }
    WetPaintBakeResult baked;
    if (!bakeWetPaintGeometry(bake_faces, s.width, s.height, kTile, kGutterTexels,
                              s.object, uv_set, baked, error)) {
        return false;
    }
    s.pos_valid = std::move(baked.pos_valid);
    s.flow_normal = std::move(baked.flow_normal);
    s.normal_spans = std::move(baked.normal_spans);
    s.tiles_x = baked.tiles_x;
    s.tiles_y = baked.tiles_y;
    s.valid_texels = baked.valid_texels;
    s.gutter_texels = baked.gutter_texels;
    s.triangles = baked.triangles;
    s.tiles.resize(baked.tiles.size());
    static_assert(sizeof(GpuTile) == sizeof(WetPaintBakeTile), "tile layout must match");
    if (!baked.tiles.empty()) {
        std::memcpy(s.tiles.data(), baked.tiles.data(), baked.tiles.size() * sizeof(GpuTile));
    }
    return true;
}

bool WetPaintGpu::addSurface(SceneData& scene, const std::string& object, int material_id,
                             int resolution, std::string& error) {
    if (resolution < 32 || resolution > 4096) {
        error = "wet paint resolution must be between 32 and 4096";
        return false;
    }
    // material_id < 0 = EVERY material slot of the object, one surface each.
    // It used to mean "the slot of the first triangle": an object with several
    // slots (a sofa's seat and base, a toy shelf's back and planks) got paint on
    // one of them only, and the others stayed clean with no message (2026-11-19).
    if (material_id < 0) {
        const auto slots = objectPaintMaterials(scene, object);
        if (slots.empty()) {
            error = "paint target mesh/material not found: " + object;
            return false;
        }
        // Check every slot before adding any, so a refusal leaves nothing half added.
        for (uint16_t slot : slots) {
            for (const auto& other : surfaces_) {
                if (other->material_id == slot && other->object != object) {
                    error = "material " + std::to_string(slot) + " (a slot of " + object +
                            ") is shared with wet paint surface '" + other->object +
                            "': give each painted object its own material (material.assign)";
                    return false;
                }
            }
        }
        for (uint16_t slot : slots) {
            if (!addSurface(scene, object, slot, resolution, error)) {
                return false;
            }
        }
        return true;
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
    // The material's UV transform BEFORE the adapter binds the canvas (binding
    // resets it to identity): a tiled base keeps it (canvas over a tiled base).
    PrincipledBSDF* pbsdf = paintMaterial(mat);
    const std::optional<PrincipledBSDF::TextureTransform> transform_before =
        pbsdf ? std::optional<PrincipledBSDF::TextureTransform>(pbsdf->textureTransform) : std::nullopt;
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
    const int64_t texel_count = static_cast<int64_t>(texels);
    #pragma omp parallel for schedule(static) if(texel_count >= 65536)
    for (int64_t i = 0; i < texel_count; ++i) {
        s.base[static_cast<size_t>(i)] = packRgba8(texture->pixels[static_cast<size_t>(i)]);
    }
    // No texture under the canvas: the base is the material colour and
    // follows it (tick). MeshPaintAdapter seeded the canvas with it.
    s.base_from_color = !set->wasSeededFromExisting(PaintChannel::BaseColor);
    if (const Material* material = MaterialManager::getInstance().getMaterial(mat)) {
        s.base_linear = material->albedoProperty.color;
    }

    s.uv_set = adapter.getTarget().uv_set;
    s.paint_material = adapter.getMaterialID();
    if (!bakeGeometry(scene, s, error)) {
        return false;
    }
    // A base colour texture under the canvas: the material's own texture, not a
    // canvas of ours (isGeneratedPaintTexture is refused as a source upstream).
    s.base_texture = set->getSourceTexture(PaintChannel::BaseColor);
    if (pbsdf && pbsdf->paintBaseTexture) {
        // Re-added surface already in overlay: the transform was never reset.
        s.base_texture = pbsdf->paintBaseTexture;
    } else if (pbsdf && transform_before && !identityTransform(*transform_before) && s.base_texture) {
        pbsdf->textureTransform = *transform_before;
    }
    if (pbsdf && s.base_texture && !identityTransform(pbsdf->textureTransform) &&
        s.settings.reveal_picture != "@base") {
        setCanvasOverlay(s, true);
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

// material_id < 0 removes every slot of the object (see addSurface).
bool WetPaintGpu::removeSurface(const std::string& object, int material_id, SceneData* scene) {
    bool removed = false;
    for (auto it = surfaces_.begin(); it != surfaces_.end();) {
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
            it = surfaces_.erase(it);
            layout_dirty_ = true;
            removed = true;
            continue;
        }
        ++it;
    }
    return removed;
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

// material_id < 0 sets every slot of the object (see addSurface).
bool WetPaintGpu::setSurfaceSettings(const std::string& object, int material_id,
                                     const WetPaintSurfaceSettings& settings) {
    bool found = false;
    for (auto& surface : surfaces_) {
        Surface* s = surface.get();
        if (s->object != object || (material_id >= 0 && s->material_id != material_id)) continue;
        found = true;
        const auto& previous = s->settings;
        const bool changed = previous.roughness_channel != settings.roughness_channel ||
            previous.height_normal_channel != settings.height_normal_channel ||
            previous.reveal_opacity != settings.reveal_opacity ||
            previous.reveal_picture != settings.reveal_picture;
        s->edited_planes |= WetPaintChannelEdits::changedPlanes(previous, settings);
        s->settings = settings;   // the picture file itself is (re)loaded in prepareChannels
        s->channels_refresh = s->channels_refresh || changed;
        channels_dirty_ = channels_dirty_ || changed;
    }
    if (found) settings_dirty_ = true;
    return found;
}

void WetPaintGpu::setSplatSettings(const WetPaintSplatSettings& settings) {
    splat_ = settings;
    splat_.max_events_per_tick =
        WetPaintMobileBudget::clampMaxEventsPerTick(splat_.max_events_per_tick);
}

void WetPaintGpu::queueDab(const WetPaintDab& dab) { pending_dabs_.push_back(dab); }

bool WetPaintGpu::resetPaint() {
    if (surfaces_.empty()) return false;
    reset_requested_ = true;
    if (device_) device_->channels_force = device_->channels_allocated;
    pending_dabs_.clear();
    // ★ The paint is clean from here on: the cached coverage must say so. Before
    // 2026-11-18 the cache kept the pre-reset ratio, the room goal's next
    // (allow_stale) poll read it, room_progress jumped back to 1 and all four
    // milestones fired in one tick - the finale cascade popped every balloon
    // right after a reset. A new serial with a zero answer; measurements of
    // older ticks still in flight are dropped by applyStats.
    ++paint_serial_;
    coverage_cache_.covered_texels = 0;
    coverage_cache_.ratio = 0.0f;
    for (auto& s : coverage_cache_.surfaces) {
        s.covered_texels = s.wet_texels = s.roughness_texels = s.normal_texels = 0;
        s.ratio = 0.0f;
        s.mean_color[0] = s.mean_color[1] = s.mean_color[2] = 0.0f;
    }
    coverage_serial_ = paint_serial_;
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
        info.canvas_overlay = s->overlay;
        info.base_texture = s->overlay && s->base_texture ? s->base_texture->name : std::string();
        info.settings = s->settings;
        out.push_back(std::move(info));
    }
    return out;
}

std::shared_ptr<Texture> WetPaintGpu::cleanBaseTextureForSave(uint16_t material_id) const {
    for (const auto& sp : surfaces_) {
        const Surface& s = *sp;
        if (s.material_id != static_cast<int>(material_id)) continue;
        // Prefer the real non-generated base texture under the canvas (both
        // overlay and composite modes).
        if (s.base_texture && s.base_texture->is_loaded() &&
            s.base_texture->name.rfind("generated/", 0) != 0) {
            return s.base_texture;
        }
        // For a surface whose texture already lives inside a generated/...
        // canvas, return a clean copy from s.base_original (while "@base" is
        // active) or s.base so saving never embeds the white-washed "@base"
        // blank output or transient wet paint splats.
        const std::vector<uint32_t>& clean =
            !s.base_original.empty() ? s.base_original : s.base;
        const std::size_t expected = static_cast<std::size_t>(s.width) * s.height;
        if (s.base_from_color || clean.size() != expected || expected == 0) {
            return nullptr;
        }
        auto copy = std::make_shared<Texture>(
            s.texture_name, static_cast<int>(s.width), static_cast<int>(s.height),
            TextureType::Albedo);
        copy->is_srgb = true;
        for (std::size_t i = 0; i < expected; ++i) {
            copy->pixels[i] = unpackRgba8(clean[i]);
        }
        copy->markSaveDirty();
        return copy;
    }
    return nullptr;
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
    Device& d = *device_;
    SimulationComputeContext& c = *d.compute;
    SimulationComputeContext* keep = d.compute;
    d.release();
    d.compute = keep;
    d.backend = c.backendType();

    // Sync geometry_key after GameDesign enables hidden-object instance poses;
    // rebake only when world positions/UVs (key[0..7]) actually changed.
    for (auto& s : surfaces_) {
        const auto key = geometryKey(scene, s->object);
        bool world_changed = false;
        for (size_t k = 0; k < 8; ++k) {
            if (std::abs(key[k] - s->geometry_key[k]) >
                1e-4 * std::max(1.0, std::abs(key[k]))) {
                world_changed = true;
                break;
            }
        }
        if (world_changed) {
            std::string error;
            if (!bakeGeometry(scene, *s, error)) {
                s->geometry_key = key;
            }
        } else {
            s->geometry_key = key;
        }
    }

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
    // prepareChannels() immediately replaces base and output at 6-7x size when
    // any PBR channel or picture is active; allocate tiny placeholders now to
    // skip the redundant full-atlas base/output allocation and staging upload.
    const bool will_use_channels = std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
        return s->settings.roughness_channel || s->settings.height_normal_channel ||
               s->settings.reveal_opacity || !s->picture.empty();
    });
    const std::size_t base_alloc_bytes = will_use_channels
        ? 16u : static_cast<std::size_t>(texels) * 4u;
    // [active, next] per tile | 3 indirect counts | channel list (tiles) |
    // changed bits (tiles, paint_wet.comp tileChangedIndex, 2026-11-24).
    const std::size_t tile_flag_bytes = channelDispatchOffset(tiles) +
        (kIndirectGroupCountWords + 2u * static_cast<std::size_t>(tiles)) * sizeof(uint32_t);
    bool ok = makeBuffer(d.surfaces, n * sizeof(GpuSurface), "WetPaintSurfaces", ro) &&
              makeBuffer(d.tiles, static_cast<std::size_t>(tiles) * sizeof(GpuTile), "WetPaintTiles", ro) &&
              makeBuffer(d.pos_valid, static_cast<std::size_t>(texels) * 16, "WetPaintPosValid", ro) &&
              makeBuffer(d.flow_normal, static_cast<std::size_t>(texels) * 16, "WetPaintFlowNormal", ro) &&
              makeBuffer(d.base, base_alloc_bytes, "WetPaintBase", ro) &&
              makeBuffer(d.color, static_cast<std::size_t>(texels) * 4, "WetPaintColor", rw) &&
              makeBuffer(d.state, static_cast<std::size_t>(texels) * 16, "WetPaintState", rw) &&
              makeBuffer(d.color_scratch, static_cast<std::size_t>(texels) * 4, "WetPaintColorScratch", rw) &&
              makeBuffer(d.state_scratch, static_cast<std::size_t>(texels) * 16, "WetPaintStateScratch", rw) &&
              makeBuffer(d.output, base_alloc_bytes, "WetPaintOutput", rw) &&
              makeBuffer(d.tile_flags, tile_flag_bytes, "WetPaintTileFlags",
                         rw | ComputeBufferUsage::Indirect) &&
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
    for (uint32_t si = 0; si < n; ++si) {
        Surface& s = *surfaces_[si];
        for (GpuTile t : s.tiles) {
            t.surface = si;
            all_tiles.push_back(t);
        }
        const std::size_t off = s.page_offset;
        const std::size_t cnt = static_cast<std::size_t>(s.width) * s.height;
        c.beginTransferBatch();
        ok = ok && c.uploadBuffer(d.pos_valid, s.pos_valid.data(), cnt * 16, off * 16) &&
             c.uploadBuffer(d.flow_normal, s.flow_normal.data(), cnt * 16, off * 16);
        if (!will_use_channels) {
            ok = ok && c.uploadBuffer(d.base, s.base.data(), cnt * 4, off * 4);
            if (s.overlay) {
                // Premultiplied transparent: nothing painted over the tiled base.
                zero_u.assign(cnt, 0u);
                ok = ok && c.uploadBuffer(d.output, zero_u.data(), cnt * 4, off * 4);
            } else {
                ok = ok && c.uploadBuffer(d.output, s.base.data(), cnt * 4, off * 4);
            }
        }
        c.endTransferBatch();
        c.synchronize();
    }
    // Whole tile_flags buffer, changed bits included (a stale bit only costs one
    // extra channel pass, but a garbage one would be read before any ROTATE).
    // color/state/scratch buffers are zeroed on GPU via MODE_RESET + kResetInitOnlyBit
    // on the first tick/coverage, avoiding ~80-320 MB of CPU zero alloc + staging.
    std::vector<uint32_t> zero_flags(tile_flag_bytes / sizeof(uint32_t), 0u);
    std::vector<uint32_t> zero_events(12, 0u), zero_counter(4, 0u);
    c.beginTransferBatch();
    ok = ok && c.uploadBuffer(d.tiles, all_tiles.data(), all_tiles.size() * sizeof(GpuTile)) &&
         c.uploadBuffer(d.tile_flags, zero_flags.data(), zero_flags.size() * 4) &&
         c.uploadBuffer(d.dummy_events, zero_events.data(), zero_events.size() * 4) &&
         c.uploadBuffer(d.dummy_counter, zero_counter.data(), zero_counter.size() * 4);
    c.endTransferBatch();
    c.synchronize();
    if (!ok) {
        stats_.status = "upload_failed";
        d.release();
        d.compute = keep;
        return false;
    }
    d.zero_init_needed = true;
    for (auto& surface : surfaces_) surface->geometry_upload = false;
    layout_dirty_ = false;
    settings_dirty_ = true;
    ++stats_.device_rebuilds;
    return true;
}

// Only geometric pages change on a rigid hidden-object move. Preserve colour,
// output/channel buffers and texture handles that asynchronous frames still use.
bool WetPaintGpu::patchGeometry() {
    Device& d = *device_;
    auto& compute = *d.compute;
    bool pending = false;
    for (const auto& surface : surfaces_) {
        pending = pending || surface->geometry_upload;
    }
    if (!pending) return true;
    compute.synchronize(); // Rare edit/reset boundary, never the normal paint tick.
    for (size_t index = 0; index < surfaces_.size(); ++index) {
        auto& surface = *surfaces_[index];
        if (!surface.geometry_upload) continue;
        const size_t count = static_cast<size_t>(surface.width) * surface.height;
        for (auto& tile : surface.tiles) tile.surface = static_cast<uint32_t>(index);
        bool ok = compute.uploadBuffer(d.pos_valid, surface.pos_valid.data(), count * 16,
                                       static_cast<size_t>(surface.page_offset) * 16) &&
                  compute.uploadBuffer(d.flow_normal, surface.flow_normal.data(), count * 16,
                                       static_cast<size_t>(surface.page_offset) * 16) &&
                  compute.uploadBuffer(d.tiles, surface.tiles.data(),
                                       surface.tiles.size() * sizeof(GpuTile),
                                       static_cast<size_t>(surface.tile_offset) * sizeof(GpuTile));
        if (ok && d.channels_allocated) {
            std::vector<float> metric(count);
            for (int axis = 0; axis < 2; ++axis) {
                for (size_t pixel = 0; pixel < count; ++pixel) {
                    metric[pixel] = axis == 0 ? surface.normal_spans[pixel].u
                                              : surface.normal_spans[pixel].v;
                }
                const size_t offset = (static_cast<size_t>(4 + axis) * d.texel_count +
                                       surface.page_offset) * 4;
                ok = compute.uploadBuffer(d.base, metric.data(), count * 4, offset) && ok;
            }
        }
        if (!ok) {
            stats_.status = "geometry_patch_failed: " + surface.object;
            return false;
        }
        surface.geometry_upload = false;
        ++stats_.geometry_uploads;
        surface.base_upload = true; // Existing per-surface reset/publish after the pose edit.
        settings_dirty_ = true;
    }
    return true;
}

// ── canvas over a tiled base ────────────────────────────────────────────────
// docs/dev/WET_PAINT_CANVAS_UV.md. Composite mode (the default, and every
// untiled surface): the output page is base + picture + paint, bound as the
// material's albedo, sampled through an identity transform. Overlay mode: the
// material keeps its UV transform and its base colour texture
// (PrincipledBSDF::paintBaseTexture -> GpuMaterial.paint_base_tex), the
// renderer samples every slot at the raw UV and puts the premultiplied output
// page over the tiled base. Switching resets this surface's paint (base slice
// rewritten + MODE_RESET of its tiles, the base colour edit's path).
void WetPaintGpu::setCanvasOverlay(Surface& s, bool overlay) {
    PrincipledBSDF* pbsdf = paintMaterial(s.material_id);
    auto texture = s.texture.lock();
    if (overlay && (!pbsdf || !s.base_texture || !texture)) return;
    const std::size_t texels = static_cast<std::size_t>(s.width) * s.height;
    if (overlay) {
        // `base` is only the pickup colour now: the base texture's mean, alpha 0.
        uint64_t sum[3] = {0, 0, 0};
        const auto& px = s.base_texture->pixels;
        for (const auto& p : px) { sum[0] += p.r; sum[1] += p.g; sum[2] += p.b; }
        const uint64_t n = std::max<uint64_t>(1, px.size());
        const uint32_t pickup = packRgba8(CompactVec4(static_cast<uint8_t>(sum[0] / n),
                                                      static_cast<uint8_t>(sum[1] / n),
                                                      static_cast<uint8_t>(sum[2] / n), 0));
        s.base.assign(texels, pickup);
        // The canvas texture's host copy: transparent (premultiplied zero). The
        // device copy is rewritten by the reset below; this is what any host
        // re-upload of the canvas shows.
        texture->pixels.assign(texels, CompactVec4(0, 0, 0, 0));
        texture->has_alpha = true;
        pbsdf->paintBaseTexture = s.base_texture;
    } else {
        if (s.base_texture) {
            s.base = stretchedBase(*s.base_texture, s.width, s.height);
        }
        if (texture && texture->pixels.size() == texels && s.base.size() == texels) {
            for (std::size_t i = 0; i < texels; ++i) texture->pixels[i] = unpackRgba8(s.base[i]);
        }
        if (pbsdf) {
            pbsdf->paintBaseTexture.reset();
            // The canvas is sampled through the material transform again.
            pbsdf->textureTransform = PrincipledBSDF::TextureTransform(
                Vec2(1.0, 1.0), 0.0f, Vec2(0.0, 0.0), Vec2(1.0, 1.0), pbsdf->textureTransform.wrapMode);
        }
    }
    s.overlay = overlay;
    s.base_from_color = false;
    s.base_upload = true;
    settings_dirty_ = true;
    g_materials_dirty = true;
}

// The material decides the mode: a base colour texture under the canvas AND a
// non-identity UV transform = overlay. Set after the surface was added (the
// material panel's tiling field, material.set uv_*) the canvas used to repeat
// with the wallpaper; now the surface switches. Without a base texture a
// transform can only tile the canvas itself, so it is undone.
void WetPaintGpu::updateCanvasModes() {
    for (auto& surface : surfaces_) {
        Surface& s = *surface;
        PrincipledBSDF* pbsdf = paintMaterial(s.material_id);
        if (!pbsdf) continue;
        // Detect a material albedo texture swap/clear in the Material Panel (or
        // via material.set) while this wet paint surface is active. Adopt the
        // new texture as the surface's base (and recapture "@base" if active),
        // then re-bind the canvas so live viewport, "@base" picture and project
        // save all stay in sync with the user's new texture.
        auto canvas = s.texture.lock();
        std::shared_ptr<Texture> cur_albedo = pbsdf->albedoProperty.texture;
        if (canvas && cur_albedo != canvas) {
            const std::size_t texels = static_cast<std::size_t>(s.width) * s.height;
            if (cur_albedo && cur_albedo->is_loaded()) {
                s.base_texture = cur_albedo;
                s.base_from_color = false;
                s.base = stretchedBase(*cur_albedo, s.width, s.height);
                if (canvas->pixels.size() == texels && s.base.size() == texels) {
                    for (std::size_t i = 0; i < texels; ++i) {
                        canvas->pixels[i] = unpackRgba8(s.base[i]);
                    }
                }
                canvas->markVulkanDirtyFull();
                pbsdf->albedoProperty.texture = canvas;
                if (s.overlay) {
                    pbsdf->paintBaseTexture = s.base_texture;
                    setCanvasOverlay(s, true);
                } else {
                    s.base_upload = true;
                    settings_dirty_ = true;
                    g_materials_dirty = true;
                }
                if (s.settings.reveal_picture == "@base") {
                    s.base_original.clear();
                }
                SCENE_LOG_INFO("[WetPaint] '" + s.object + "': adopted new base albedo texture '" +
                               cur_albedo->name + "'");
            } else if (!cur_albedo) {
                s.base_texture.reset();
                s.base_from_color = true;
                s.base_linear = pbsdf->albedoProperty.color;
                const CompactVec4 fill(
                    static_cast<uint8_t>(std::clamp(std::lround(linearToSrgb01(s.base_linear.x) * 255.0f), 0L, 255L)),
                    static_cast<uint8_t>(std::clamp(std::lround(linearToSrgb01(s.base_linear.y) * 255.0f), 0L, 255L)),
                    static_cast<uint8_t>(std::clamp(std::lround(linearToSrgb01(s.base_linear.z) * 255.0f), 0L, 255L)),
                    255);
                s.base.assign(texels, packRgba8(fill));
                canvas->pixels.assign(texels, fill);
                canvas->markVulkanDirtyFull();
                pbsdf->albedoProperty.texture = canvas;
                pbsdf->paintBaseTexture.reset();
                s.overlay = false;
                if (s.settings.reveal_picture == "@base") {
                    s.base_original.clear();
                }
                s.base_upload = true;
                settings_dirty_ = true;
                g_materials_dirty = true;
            }
        }
        const bool tiled = !identityTransform(pbsdf->textureTransform);
        // "@base" makes the base itself the hidden picture, laid over the page
        // once: there is no tiled base to put the canvas over. Left to the
        // composite path (a transform there tiles the canvas, as before).
        const bool base_picture = s.settings.reveal_picture == "@base";
        if (base_picture) {
            if (s.overlay) {
                setCanvasOverlay(s, false);
                SCENE_LOG_INFO("[WetPaint] '" + s.object + "': '@base' picture, canvas back to composite mode");
            }
            continue;
        }
        if (s.overlay) {
            if (!tiled) {
                setCanvasOverlay(s, false);
                SCENE_LOG_INFO("[WetPaint] '" + s.object + "': UV transform is identity, canvas back to composite mode");
            }
        } else if (tiled) {
            if (s.base_texture) {
                setCanvasOverlay(s, true);
                SCENE_LOG_INFO("[WetPaint] '" + s.object + "': tiled base texture, canvas switched to overlay mode");
            } else {
                pbsdf->textureTransform = PrincipledBSDF::TextureTransform(
                    Vec2(1.0, 1.0), 0.0f, Vec2(0.0, 0.0), Vec2(1.0, 1.0), pbsdf->textureTransform.wrapMode);
                g_materials_dirty = true;
                SCENE_LOG_WARN("[WetPaint] '" + s.object +
                               "': UV transform on a canvas without a base texture would tile the paint; reset to identity");
            }
        }
    }
}

// Channel targets use the existing PaintTextureSet adapter (including its UV
// bake and source texture bookkeeping). No CPU brush/composite is run per frame.
bool WetPaintGpu::prepareChannels(SceneData& scene) {
    Device& d = *device_;
    // Hidden pictures: (re)load whatever file a surface's settings name now.
    for (auto& surface : surfaces_) {
        Surface& s = *surface;
        if (s.settings.reveal_picture == s.picture_path_loaded) continue;
        if (s.settings.reveal_picture == "@base") continue;   // handled by syncBasePictures in tick
        s.picture_path_loaded = s.settings.reveal_picture;
        if (!loadPictureImage(s.settings.reveal_picture, s.width, s.height, s.picture)) {
            s.picture.clear();
            stats_.status = "picture_load_failed: " + s.object + ": " + s.settings.reveal_picture;
            // The status is overwritten by the next tick ("running"); the log
            // keeps the reason, so a wall with no picture is not a silent mystery.
            SCENE_LOG_WARN("[WetPaint] hidden picture could not be loaded for '" + s.object +
                           "': " + s.settings.reveal_picture);
        }
        s.picture_upload = true;
        s.channels_refresh = true;
        channels_dirty_ = true;
        settings_dirty_ = true;
    }
    const bool picture_atlas = d.picture_allocated ||
        std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
            return !s->picture.empty();
        });
    const bool enabled = picture_atlas || std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
        return s->settings.roughness_channel || s->settings.height_normal_channel ||
               s->settings.reveal_opacity;
    });
    if (!enabled && !d.channels_allocated) {
        channels_dirty_ = false;
        return true;
    }
    if (!channels_dirty_ && d.channels_allocated) return true;
    const bool reveal_atlas = d.reveal_allocated ||
        std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
            return s->settings.reveal_opacity;
        });
    const bool expand = !d.channels_allocated || (reveal_atlas && !d.reveal_allocated) ||
                        (picture_atlas && !d.picture_allocated);
    bool upload = expand;
    for (auto& surface : surfaces_) {
        if (surface->picture_upload) { upload = true; surface->picture_upload = false; }
    }
    static constexpr PaintChannel channels[4] = {
        PaintChannel::Roughness, PaintChannel::Mask, PaintChannel::Normal, PaintChannel::Opacity
    };
    for (auto& surface : surfaces_) {
        Surface& s = *surface;
        const bool need_roughness = s.settings.roughness_channel && s.channel_base[0].empty();
        const bool need_normal = s.settings.height_normal_channel && s.channel_base[2].empty();
        const bool need_opacity = s.settings.reveal_opacity && s.channel_base[3].empty();
        if (!need_roughness && !need_normal && !need_opacity) continue;
        auto triangle = findPaintTriangle(scene, s.object, s.material_id);
        if (!triangle || !triangle->getMaterial()) {
            stats_.status = "channel_material_missing: " + s.object;
            return false;
        }
        auto material = triangle->getMaterial();
        auto principled = std::dynamic_pointer_cast<PrincipledBSDF>(material);
        const auto& roughness_source = material->roughnessProperty.texture;
        const bool had_roughness = roughness_source && roughness_source->is_loaded() &&
                                   !roughness_source->pixels.empty();
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
        for (int ch = 0; ch < 4; ++ch) {
            if ((ch == 0 && !need_roughness) ||
                ((ch == 1 || ch == 2) && !need_normal) || (ch == 3 && !need_opacity)) {
                continue;
            }
            std::shared_ptr<Texture>& existing = channel_set.getTextureRef(channels[ch]);
            if (existing && (existing->width != static_cast<int>(s.width) ||
                             existing->height != static_cast<int>(s.height))) {
                existing.reset();
            }
        }
        for (int ch = 0; ch < 4; ++ch) {
            if ((ch == 0 && !need_roughness) ||
                ((ch == 1 || ch == 2) && !need_normal) || (ch == 3 && !need_opacity)) {
                continue;
            }
            if (!adapter.assignTextureToChannel(channels[ch])) {
                stats_.status = "channel_target_failed: " + s.object;
                return false;
            }
        }
        auto* set = adapter.getTextureSet();
        if (!set) return false;
        const std::size_t count = static_cast<std::size_t>(s.width) * s.height;
        for (int ch = 0; ch < 4; ++ch) {
            if ((ch == 0 && !need_roughness) ||
                ((ch == 1 || ch == 2) && !need_normal) || (ch == 3 && !need_opacity)) {
                continue;
            }
            auto texture = set->getTexture(channels[ch]);
            if (!texture || texture->width != static_cast<int>(s.width) ||
                texture->height != static_cast<int>(s.height) || texture->pixels.size() != count) {
                stats_.status = "channel_resolution_mismatch: " + s.object;
                return false;
            }
            texture->name = "generated/wet/" + std::to_string(texture->m_uid) + ".png";
            texture->is_srgb = false;
            if (ch == 3) {
                texture->has_alpha = true;
            }
            s.channel_textures[ch] = texture;
            s.channel_base[ch].resize(count);
            const int64_t pixel_count = static_cast<int64_t>(count);
            #pragma omp parallel for schedule(static) if(pixel_count >= 65536)
            for (int64_t pi = 0; pi < pixel_count; ++pi) {
                const std::size_t pixel = static_cast<std::size_t>(pi);
                CompactVec4 value = texture->pixels[pixel];
                if (ch == 0) {
                    const uint8_t components[3] = {value.r, value.g, value.b};
                    const float scalar = had_roughness
                        ? components[roughness_component] / 255.0f * roughness_factor
                        : roughness_scalar;
                    const auto byte = static_cast<uint8_t>(
                        std::clamp(scalar, 0.0f, 1.0f) * 255.0f + 0.5f);
                    value = CompactVec4(byte, byte, byte, 255);
                } else if (ch == 3) {
                    // Reveal owns opacity on solid geometry. Never seed from a
                    // saved transient reveal mask: a clean reload must be paintable.
                    value = CompactVec4(255, 255, 255, 255);
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
        for (int ch = 0; ch < 4; ++ch) {
            auto texture = s.channel_textures[ch].lock();
            if (!texture) continue;
            const int64_t pixel_count = static_cast<int64_t>(count);
            #pragma omp parallel for schedule(static) if(pixel_count >= 65536)
            for (int64_t pi = 0; pi < pixel_count; ++pi) {
                const std::size_t pixel = static_cast<std::size_t>(pi);
                const uint32_t v = (ch == 3 && s.settings.reveal_opacity)
                    ? 0u : s.channel_base[ch][pixel];
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
        std::vector<uint32_t> original(count * (picture_atlas ? kPicturePlanes : 6u), 0u);
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
            if (picture_atlas && !s.picture.empty()) {
                std::copy(s.picture.begin(), s.picture.end(), original.begin() + 6 * count + offset);
            }
            const int64_t span_count = static_cast<int64_t>(s.normal_spans.size());
            #pragma omp parallel for schedule(static) if(span_count >= 65536)
            for (int64_t pi = 0; pi < span_count; ++pi) {
                const std::size_t pixel = static_cast<std::size_t>(pi);
                std::memcpy(&original[4 * count + offset + pixel], &s.normal_spans[pixel].u, 4);
                std::memcpy(&original[5 * count + offset + pixel], &s.normal_spans[pixel].v, 4);
            }
        }
        // Source planes 4/5 remain the UV metric; opacity needs no extra source plane.
        std::vector<uint32_t> outputs(count * (reveal_atlas ? 5 : 4), 0u);
        std::copy_n(original.begin(), count * 4, outputs.begin());
        if (reveal_atlas) {
            std::fill(outputs.begin() + count * 4, outputs.end(), 0xFFFFFFFFu);
        }
        ComputeBufferHandle new_base, new_output;
        if (expand) {
            ComputeBufferDesc desc;
            desc.debug_name = "WetPaintChannelSources";
            desc.size_bytes = original.size() * 4;
            desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                         ComputeBufferUsage::ReadOnly;
            new_base = compute.createBuffer(desc);
            desc.debug_name = "WetPaintChannelOutputs";
            desc.size_bytes = outputs.size() * 4;
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
        auto& base = expand ? new_base : d.base;
        auto& output = expand ? new_output : d.output;
        compute.synchronize();
        const bool ok = compute.uploadBuffer(base, original.data(), original.size() * 4) &&
                        compute.uploadBuffer(output, outputs.data(), outputs.size() * 4);
        if (!ok) {
            if (new_base.valid()) compute.destroyBuffer(new_base);
            if (new_output.valid()) compute.destroyBuffer(new_output);
            stats_.status = "channel_upload_failed";
            return false;
        }
        if (expand) {
            compute.destroyBuffer(d.base);
            compute.destroyBuffer(d.output);
            d.base = new_base;
            d.output = new_output;
            d.channels_allocated = true;
            d.reveal_allocated = reveal_atlas;
            d.picture_allocated = picture_atlas;
        }
    }
    d.channels_force = true;
    channels_dirty_ = false;
    stats_.pbr_buffer_bytes = static_cast<uint64_t>(d.texel_count) * (d.reveal_allocated ? 36u : 32u);
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
                  (w.height_normal_channel ? 4u : 0u) | (w.reveal_opacity ? 8u : 0u) |
                  (s.overlay ? kSurfaceOverlay : 0u);
        if (!s.picture.empty()) {
            // bit 4 picture; bits 8..15 outline opacity; bits 16..23 hue tolerance (deg/180*255); bits 24..31 hint 0..1
            const float tolerance = std::clamp(w.picture_tolerance, 5.0f, 90.0f);
            g.flags |= kSurfacePicture |
                       (static_cast<uint32_t>(std::clamp(w.picture_outline, 0.0f, 1.0f) * 255.0f + 0.5f) << 8u) |
                       (static_cast<uint32_t>(tolerance / 180.0f * 255.0f + 0.5f) << 16u) |
                       (static_cast<uint32_t>(std::clamp(w.picture_hint, 0.0f, 1.0f) * 255.0f + 0.5f) << 24u);
        }
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
    // Canvas mode follows the material every tick (cheap: a lookup per
    // surface), BEFORE the "@base" picture sync below captures s.base.
    updateCanvasModes();
    // A surface whose object was moved, scaled or edited since its bake would
    // keep painting where the object USED to be (positions are baked once):
    // the room raised to 4.7 m painted nothing above the old 3.4 m. Checked
    // twice a second; a changed surface is re-baked and the atlas rebuilt.
    geometry_check_clock_ += std::clamp(dt, 0.0f, 0.25f);
    if (geometry_check_clock_ >= 0.5f) {
        geometry_check_clock_ = 0.0f;
        const uint64_t scene_gen =
            g_scene_geometry_generation.load(std::memory_order_acquire);
        const std::size_t world_count = scene.world.objects.size();
        const auto surfaceTransformsMatch = [&](const Surface& s) -> bool {
            if (!s.geometry_fast_valid) return false;
            std::size_t idx = 0;
            for (const auto& obj : scene.world.objects) {
                const auto* mesh = dynamic_cast<const TriangleMesh*>(obj.get());
                if (!mesh || mesh->nodeName != s.object || !mesh->geometry) continue;
                const Matrix4x4 world =
                    mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
                if (idx >= s.cached_transforms.size() ||
                    s.cached_transforms[idx].mesh != mesh ||
                    !(s.cached_transforms[idx].transform == world)) {
                    return false;
                }
                ++idx;
            }
            return idx == s.cached_transforms.size();
        };
        const auto captureSurfaceTransforms = [&](Surface& s) {
            s.cached_transforms.clear();
            for (const auto& obj : scene.world.objects) {
                const auto* mesh = dynamic_cast<const TriangleMesh*>(obj.get());
                if (!mesh || mesh->nodeName != s.object || !mesh->geometry) continue;
                const Matrix4x4 world =
                    mesh->transform ? mesh->transform->getFinal() : Matrix4x4::identity();
                s.cached_transforms.push_back({mesh, world});
            }
            s.geometry_fast_valid = true;
        };

        bool skip_geometry_check = false;
        if (device_ &&
            device_->last_scene_geometry_gen == scene_gen &&
            device_->last_world_object_count == world_count &&
            device_->last_surface_count == surfaces_.size() &&
            device_->last_geometry_surfaces.size() == surfaces_.size()) {
            skip_geometry_check = true;
            for (std::size_t si = 0; si < surfaces_.size(); ++si) {
                if (device_->last_geometry_surfaces[si] != surfaces_[si].get() ||
                    !surfaceTransformsMatch(*surfaces_[si])) {
                    skip_geometry_check = false;
                    break;
                }
            }
        }
        if (!skip_geometry_check) {
            for (auto& surface : surfaces_) {
                if (device_ &&
                    device_->last_scene_geometry_gen == scene_gen &&
                    device_->last_world_object_count == world_count &&
                    surfaceTransformsMatch(*surface)) {
                    continue;
                }
                const auto key = geometryKey(scene, surface->object);
                bool changed = false;
                for (size_t k = 0; k < key.size(); ++k) {
                    if (std::abs(key[k] - surface->geometry_key[k]) >
                        1e-4 * std::max(1.0, std::abs(key[k]))) {
                        changed = true;
                    }
                }
                if (!changed) {
                    captureSurfaceTransforms(*surface);
                    continue;
                }
                std::string error;
                const size_t previous_tiles = surface->tiles.size();
                const size_t previous_texels = surface->pos_valid.size();
                if (bakeGeometry(scene, *surface, error)) {
                    ++stats_.geometry_rebakes;
                    if (!layout_dirty_ && device_ && device_->compute &&
                        previous_tiles == surface->tiles.size() &&
                        previous_texels == surface->pos_valid.size()) {
                        surface->geometry_upload = true;
                    } else {
                        layout_dirty_ = true;
                    }
                } else {
                    stats_.status = "rebake_failed: " + surface->object + ": " + error;
                    surface->geometry_key = key;   // do not retry every check
                }
                captureSurfaceTransforms(*surface);
            }
            if (device_) {
                device_->last_scene_geometry_gen = scene_gen;
                device_->last_world_object_count = world_count;
                device_->last_surface_count = surfaces_.size();
                device_->last_geometry_surfaces.resize(surfaces_.size());
                for (std::size_t si = 0; si < surfaces_.size(); ++si) {
                    device_->last_geometry_surfaces[si] = surfaces_[si].get();
                }
            }
        }
        // A surface painted over its material COLOUR (no texture underneath)
        // follows that colour: made blue in the material panel, the wall must
        // turn blue (the bound canvas overrides the colour, so nothing else
        // would show it). The wet paint starts over (as it always did).
        //
        // ★★★★ Only the base slice is rewritten (2026-11-19). This used to set
        //   layout_dirty_ (the WHOLE atlas rebuilt: every surface's position,
        //   flow and normal pages re-uploaded, tens of MB) and g_materials_dirty
        //   (a full material-table sync with its drains) on every colour step.
        //   A colour picker held on a painted wall in a game scene hitched
        //   ~300 ms in loop.ui_draw whenever the colour settled; a scene
        //   without wet paint surfaces never did. Now: the new base goes into
        //   this surface's slice of the device base buffer below, and a reset
        //   of THIS surface's tile range (MODE_RESET: out = base, its tiles
        //   marked dirty) republishes it through the normal path - device copy
        //   to the render device, per-material host sync for the viewport.
        //   ★ Not resetPaint(): that resets and republishes EVERY surface, and
        //   in a 20-surface room the Studio preview (a second VkDevice, host
        //   path) re-read ~270 MB per colour step (measured 10 GB in 60 s).
        //   Paint on the other surfaces now survives the edit.
        for (auto& surface : surfaces_) {
            if (!surface->base_from_color) continue;
            const Material* material =
                MaterialManager::getInstance().getMaterial(static_cast<uint16_t>(surface->material_id));
            if (!material) continue;
            const Vec3 c = material->albedoProperty.color;
            const auto differs = [](const Vec3& a, const Vec3& b) {
                return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z) >= 1e-4f;
            };
            if (!differs(c, surface->base_linear)) {
                surface->base_pending = false;
                continue;
            }
            // ★★★ Applied once the colour SETTLES (kBaseSettleSeconds unchanged),
            //   not on every step of a drag: each application republishes the
            //   whole surface, and the Studio preview (a second VkDevice) reads
            //   it back through the host. While the picker moves nothing is
            //   rebuilt; the canvas shows the new colour a moment after it stops.
            const auto now = std::chrono::steady_clock::now();
            if (!surface->base_pending || differs(c, surface->base_pending_color)) {
                surface->base_pending = true;
                surface->base_pending_color = c;
                surface->base_pending_since = now;
                continue;
            }
            if (std::chrono::duration<double>(now - surface->base_pending_since).count() <
                kBaseSettleSeconds) {
                continue;
            }
            surface->base_pending = false;
            surface->base_linear = c;
            auto texture = surface->texture.lock();
            if (!texture) continue;
            auto srgb = [](float v) {
                v = std::clamp(v, 0.0f, 1.0f);
                const float s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                return static_cast<uint8_t>(std::clamp(std::lround(s * 255.0f), 0L, 255L));
            };
            const CompactVec4 fill(srgb(c.x), srgb(c.y), srgb(c.z), 255);
            for (auto& pixel : texture->pixels) pixel = fill;
            std::fill(surface->base.begin(), surface->base.end(), packRgba8(fill));
            surface->base_upload = true;
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
    if (!patchGeometry()) return false;
    // A base colour change (above): this surface's slice of the device base
    // only, and a reset of its own tiles below (out = base, paint cleared,
    // tiles marked dirty for publish). A fresh layout already uploaded every
    // base and starts clean.
    // "@base" hidden pictures: the surface's own base becomes the picture and
    // the base slice a flat blank colour (restored when the setting leaves @base).
    std::vector<std::pair<uint32_t, uint32_t>> base_resets;   // (first tile, tile count)
    for (auto& sp : surfaces_) {
        Surface& s = *sp;
        bool blank_edit = false;
        const bool want = (s.settings.reveal_picture == "@base");
        if (want) {
            if (s.base_original.empty()) {
                s.base_original = s.base;
                s.picture = s.base_original;
                s.picture_path_loaded = "@base";
                s.picture_upload = true;
                s.channels_refresh = true;
                channels_dirty_ = true;
                settings_dirty_ = true;
            }
            const auto channel = [](float v) {
                return static_cast<uint8_t>(std::clamp(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f), 0L, 255L));
            };
            const uint32_t blank = packRgba8(CompactVec4(channel(s.settings.picture_blank[0]),
                channel(s.settings.picture_blank[1]), channel(s.settings.picture_blank[2]), 255));
            if (!s.base.empty() && (s.base.front() != blank || s.base.back() != blank)) {
                blank_edit = !s.base_upload && !s.picture_upload &&
                    (s.edited_planes & WetPaintChannelEdits::Color) != 0;
                std::fill(s.base.begin(), s.base.end(), blank);
                s.base_upload = true;
            }
        } else if (!s.base_original.empty()) {
            s.base = s.base_original;
            s.base_original.clear();
            s.picture.clear();
            s.picture_path_loaded.clear();   // a file path in the new setting is loaded by prepareChannels
            s.picture_upload = true;
            s.base_upload = true;
            s.channels_refresh = true;
            channels_dirty_ = true;
            settings_dirty_ = true;
        }
        const bool picture_changed = !layout_fresh && s.settings.reveal_picture != s.picture_path_loaded &&
                                     s.settings.reveal_picture != "@base";
        if (picture_changed && d.texel_count != 0 && !s.tiles.empty()) {
            base_resets.emplace_back(s.tile_offset, static_cast<uint32_t>(s.tiles.size()));
        }
        if (!s.base_upload) continue;
        s.base_upload = false;
        if (layout_fresh || d.texel_count == 0 || s.tiles.empty()) continue;
        const std::size_t cnt = static_cast<std::size_t>(s.width) * s.height;
        if (!c.uploadBuffer(d.base, s.base.data(), cnt * 4, s.page_offset * 4)) {
            layout_dirty_ = true;   // could not patch: fall back to a full rebuild next tick
            continue;
        }
        if (!picture_changed && !blank_edit) {
            base_resets.emplace_back(s.tile_offset, static_cast<uint32_t>(s.tiles.size()));
        }
    }
    if (!prepareChannels(scene)) return false;
    if (settings_dirty_) uploadSurfaceTable();
    stats_.device_ready = true;
    stats_.tiles = d.tile_count;
    const uint32_t n = static_cast<uint32_t>(surfaces_.size());

    // Earlier dispatching ticks' stats (dirty rectangles, active tiles,
    // coverage). applyStats MERGES dirty rectangles into the pending uploads, so
    // a late readback loses nothing; it only arrives a frame later.
    // ★ Do not wait for the newest one. Since the Player submits its whole
    // compute tick once, asynchronously (RuntimeSimulationClock
    // defer_compute_submit), last tick's submission sits behind the render
    // frame queued ahead of it on the single queue; a blocking collect here was
    // 4.3 ms of a 14.7 ms burst frame (2026-11-23, 1080p, 10 pops) -- the same
    // stall the per-frame fence used to be. Only a full queue waits (oldest).
    // Three pending match the backend's submission ring (two in flight): the
    // oldest is already complete when it is waited for (2026-11-24).
    {
        RTPERF_FRAME_SCOPE("paint.wet.stats_collect");
        constexpr std::size_t kMaxPendingStats = 3;   // backend keeps 4 readback slots
        while (!d.stats_pending.empty()) {
            const auto p = d.stats_pending.front();
            d.stats_prev_scratch.assign(
                static_cast<std::size_t>(p.surfaces) * kStatsStride, 0u);
            const bool must_wait = d.stats_pending.size() >= kMaxPendingStats;
            if (!c.collectReadback(p.ticket, d.stats_prev_scratch.data(),
                                   d.stats_prev_scratch.size() * sizeof(uint32_t),
                                   must_wait)) {
                if (!must_wait) break;               // not ready yet: next tick
                d.stats_pending.pop_front();         // waited and still failed
                ++stats_.stats_readbacks_lost;
                continue;
            }
            d.stats_pending.pop_front();
            if (p.surfaces == n) {
                applyStats(d.stats_prev_scratch, p.force, p.coverage, p.serial);
            } else {
                ++stats_.stats_readbacks_lost;
            }
        }
    }

    // Wall time to simulate; the step's dt is decided below (kWetStepSeconds).
    const double frame_dt = dt > 0.0f ? static_cast<double>(dt) : 0.0;

    WetPaintGpuConstants k;
    const bool run_channels = d.channels_allocated &&
        (d.channels_force || d.picture_allocated || std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& s) {
            return s->settings.roughness_channel || s->settings.height_normal_channel ||
                   s->settings.reveal_opacity;
        }));
    const bool compact_channels = run_channels && !d.channels_force;
    const bool edited_channels = d.channels_allocated &&
        std::any_of(surfaces_.begin(), surfaces_.end(), [](const auto& surface) {
            return surface->edited_planes != 0;
        });
    k.surface_count = n;
    k.tile_count = d.tile_count;
    k._p0 = d.channels_allocated ? d.texel_count : 0u;
    k._p1 = (d.channels_force ? 1u : 0u) | (d.reveal_allocated ? 2u : 0u) |
            (compact_channels ? kCompactChannels : 0u) | (d.picture_allocated ? kPictureBit : 0u);
    k.dt = 1.0f / 60.0f;   // stamp does not read it; the step sets its own below
    k.splat_radius = std::max(0.0f, splat_.radius);
    k.splat_amount = std::clamp(splat_.amount, 0.0f, 1.0f);
    k.splat_min_speed = std::max(0.0f, splat_.min_speed);
    k.facing_min = std::clamp(splat_.facing_min, -1.0f, 1.0f);
    k.max_events = WetPaintMobileBudget::clampMaxEventsPerTick(splat_.max_events_per_tick);

    ComputeBufferHandle bindings[kBindingCount] = {
        d.surfaces, d.tiles, d.pos_valid, d.flow_normal, d.base, d.color, d.state,
        d.color_scratch, d.state_scratch, d.output, d.tile_flags, d.stats,
        d.dummy_events /*dabs*/, d.dummy_events, d.dummy_counter, d.dummy_events /*lut*/,
        d.dummy_counter /*lookup*/,
    };
    bool ok = true;
    const auto run = [&](WetPaintMode mode, uint32_t groups) {
        k.mode = mode;
        ComputeDispatch cmd;
        cmd.kernel = "paint_wet";
        const bool edited_range = mode == kModeChannels &&
            (k._p1 & WetPaintChannelEdits::DispatchRange) != 0;
        cmd.timing_label = edited_range ? "paint_wet.channels_edit" : kWetPaintTimingLabels[mode];
        cmd.buffers = bindings;
        cmd.buffer_count = kBindingCount;
        cmd.constants = &k;
        cmd.constants_size = sizeof(k);
        cmd.groups.groups_x = std::max<uint32_t>(1u, groups);
        if (mode == kModeChannelsList) {
            cmd.indirect_groups = d.tile_flags;
            cmd.indirect_offset = channelDispatchOffset(d.tile_count);
        }
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

    for (auto it = d.appearance_tables.begin(); it != d.appearance_tables.end();) {
        if (it->second.owner.expired()) {
            it->second.release(c);
            it = d.appearance_tables.erase(it);
        } else {
            ++it;
        }
    }

    // Particle systems: device buffers on this context, or host contacts.
    std::vector<std::shared_ptr<RayTrophiSim::ParticleSimulationSystem>> device_sources;
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
                if (alive || was_alive) device_sources.push_back(system.runtime);
                was_alive = alive;
            }
            host_events.clear();
            if (system.runtime->takeHostCollisionEvents(host_events) == 0) continue;
            // CPU-reference contacts: the same splat, coloured on the host.
            const uint32_t max_host_events =
                WetPaintMobileBudget::clampMaxEventsPerTick(splat_.max_events_per_tick);
            for (const auto& e : host_events) {
                if (e.speed < splat_.min_speed || host_taken >= max_host_events) continue;
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
    const bool simulation_idle = dabs.empty() && device_sources.empty() &&
        !reset_requested_ && base_resets.empty() &&
        stats_.active_tiles == 0 && !layout_fresh && !d.channels_force;
    if (simulation_idle && !edited_channels) {
        stats_.status = "idle";
        ++stats_.idle_ticks;
        stats_.event_sources = 0;
        stats_.ticked_tiles = stats_.stamped_texels = 0;
        stats_.last_tick_ms = msSince(t0);
        d.step_clock = 0.0;   // a new stain starts a fresh cadence
        return true;
    }
    // A prior stamp or a readback still in flight can carry simulation work
    // even when the last collected active-tile counter is zero.
    const bool edit_only = simulation_idle && d.stats_reset_needed && d.stats_pending.empty();
    // Step decision (kWetStepSeconds). Resets, a fresh layout and a forced
    // channel pass are one-shot work that must not wait for the cadence.
    d.step_clock += frame_dt;
    d.stamp_clock += frame_dt;
    const bool force_step = layout_fresh || reset_requested_ || !base_resets.empty() ||
                            d.channels_force || edited_channels;
    const double step_interval = WetPaintMobileBudget::stepIntervalSeconds(stats_.active_tiles);
    const bool step = force_step || d.step_clock >= step_interval - 1e-6;
    const bool stamp_events = !device_sources.empty() &&
        (step || d.stamp_clock >= WetPaintMobileBudget::stampIntervalSeconds() - 1e-6);
    // The step simulates all the time since the last one (rates are per
    // second), clamped as before so a stall cannot launch a huge step.
    const float step_dt = WetPaintMobileBudget::computeStepDt(d.step_clock);
    if (step) d.step_clock = 0.0;
    if (stamp_events) d.stamp_clock = 0.0;
    if (!step && dabs.empty() && !stamp_events) {
        // Drying between two steps: nothing to stamp, nothing to simulate yet.
        stats_.status = "between_steps";
        stats_.last_tick_ms = msSince(t0);
        return true;
    }
    ++paint_serial_;   // this tick dispatches: coverage() must measure again
    stats_.event_sources = static_cast<uint32_t>(device_sources.size());
    // Stat slots: dirty bbox min = max uint, the rest zero. Once per readback
    // window - a stamp-only frame adds its counters (sources dropped, stamped
    // texels) into the window the next step reads back.
    if (d.stats_reset_needed) {
        d.stats_init_scratch.assign(static_cast<std::size_t>(n) * kStatsStride, 0u);
        for (uint32_t si = 0; si < n; ++si) {
            d.stats_init_scratch[si * kStatsStride + kStDirtyMinX] = 0xFFFFFFFFu;
            d.stats_init_scratch[si * kStatsStride + kStDirtyMinY] = 0xFFFFFFFFu;
        }
        c.uploadBuffer(d.stats, d.stats_init_scratch.data(),
                       d.stats_init_scratch.size() * 4);
        d.stats_reset_needed = false;
    }
    // MODE_RESET reads source_count as its first tile (paint_wet.comp).
    const bool clearing_paint = reset_requested_;
    if (reset_requested_) {
        k.source_count = 0;
        run(kModeReset, d.tile_count);
        reset_requested_ = false;
        d.zero_init_needed = false;
    } else {
        if (d.zero_init_needed) {
            k.source_count = 0;
            k._p1 |= kResetInitOnlyBit;
            run(kModeReset, d.tile_count);
            k._p1 &= ~kResetInitOnlyBit;
            d.zero_init_needed = false;
        }
        for (const auto& range : base_resets) {
            k.source_count = range.first;
            run(kModeReset, range.second);
        }
    }
    k.source_count = 0;

    if (!dabs.empty() && ensure(d.dabs, d.dab_capacity, dabs.size(), sizeof(GpuDab), "WetPaintDabs") &&
        c.uploadBuffer(d.dabs, dabs.data(), dabs.size() * sizeof(GpuDab))) {
        bindings[12] = d.dabs;
        k.source_kind = kSourceDabs;
        k.source_count = static_cast<uint32_t>(dabs.size());
        run(kModeStamp, d.tile_count);
        bindings[12] = d.dummy_events;
    }
    if (stamp_events)
    for (const auto& system : device_sources) {
        RayTrophiSim::ParticleDeviceEventView view;
        if (!system->deviceCollisionEventView(c, view)) continue;
        auto& tables = d.appearance_tables[system.get()];
        const uint64_t revision = system->appearanceRevision();
        if (tables.revision != revision) {
            tables.owner = system;
            tables.revision.reset();
            tables.lut_rows = tables.lookup_count = 0;
            const auto& profiles = system->appearanceProfiles();
            if (profiles.empty()) {
                tables.revision = revision;
            } else {
                const auto max_profile = std::max_element(
                    profiles.begin(), profiles.end(),
                    [](const auto& a, const auto& b) { return a.id < b.id; });
                std::vector<uint32_t> lookup(static_cast<std::size_t>(max_profile->id) + 1,
                                             0xFFFFFFFFu);
                constexpr auto row_size = RayTrophiSim::kParticleAppearanceLutFloatsPerRow;
                std::vector<float> lut;
                lut.reserve(profiles.size() * row_size);
                for (const auto& profile : profiles) {
                    const float* row = system->appearanceLutRow(profile.id);
                    lookup[profile.id] = static_cast<uint32_t>(lut.size() / row_size);
                    lut.insert(lut.end(), row, row + row_size);
                }
                if (ensure(tables.lut, tables.lut_capacity, lut.size(), sizeof(float),
                           "WetPaintLut") &&
                    ensure(tables.lookup, tables.lookup_capacity, lookup.size(), sizeof(uint32_t),
                           "WetPaintLookup") &&
                    c.uploadBuffer(tables.lut, lut.data(), lut.size() * sizeof(float)) &&
                    c.uploadBuffer(tables.lookup, lookup.data(), lookup.size() * sizeof(uint32_t))) {
                    tables.lut_rows = static_cast<uint32_t>(lut.size() / row_size);
                    tables.lookup_count = static_cast<uint32_t>(lookup.size());
                    tables.revision = revision;
                }
            }
        }
        bindings[13] = view.events;
        bindings[14] = view.counter;
        bindings[15] = tables.lut_rows ? tables.lut : d.dummy_events;
        bindings[16] = tables.lookup_count ? tables.lookup : d.dummy_counter;
        k.source_kind = kSourceEvents;
        k.source_count = 0;
        k.event_capacity = view.capacity;
        k.lut_rows = tables.lut_rows;
        k.lookup_count = tables.lookup_count;
        if (!clearing_paint) run(kModeStamp, d.tile_count);
        run(kModeConsume, 1);   // the consume: counter back to zero
        ++stats_.device_event_dispatches;
        bindings[13] = d.dummy_events;
        bindings[14] = d.dummy_counter;
        bindings[15] = d.dummy_events;
        bindings[16] = d.dummy_counter;
    }

    if (!step) {
        // Stamp-only frame: the stamps (and consumed events) are recorded; the
        // next step simulates and publishes them. No readback - its counters
        // stay in the current stats window.
        if (!ok) {
            c.submitAsync();
            stats_.status = "dispatch_failed";
            stats_.last_tick_ms = msSince(t0);
            return false;
        }
        c.submitAsync();   // no-op inside the Player's deferred frame submit
        stats_.status = "stamping";
        stats_.last_tick_ms = msSince(t0);
        return true;
    }
    if (!edit_only) {
        k.dt = step_dt;
        run(kModeTick, d.tile_count);
        run(kModeResolve, d.tile_count);
        if (run_channels) {
            if (compact_channels) {
                run(kModeCompactChannels, (d.tile_count + 255u) / 256u);
                if (ok) run(kModeChannelsList, 0);
            } else {
                run(kModeSeedHeight, d.tile_count);
                run(kModeChannels, d.tile_count);
            }
            ++stats_.pbr_dispatches;
        }
        run(kModeRotate, (d.tile_count + 255u) / 256u);
    }
    if (edited_channels && !d.channels_force) {
        WetPaintChannelEdits::dispatchRanges(surfaces_, k, [&](uint32_t count, bool seed_height) {
            if (seed_height) {
                run(kModeSeedHeight, count);
            }
            run(kModeChannels, count);
            ++stats_.pbr_dispatches;
        });
    }
    // A periodic reader got a stale coverage: measure it here, after this
    // tick's final state, inside the readback this tick pays anyway (the
    // stats_init upload above zeroed the coverage slots too).
    const bool measure_coverage = WetPaintMobileBudget::allowPeriodicCoverage(
        coverage_requested_, d.coverage_clock, step_dt);
    if (measure_coverage) run(kModeCoverage, d.tile_count);

    const bool force = d.channels_force;
    if (!ok) {
        // ★ Still hand the recorded work over: on wet paint's own context
        // nobody else submits it, and a recording left open forever is the
        // "kernels never ran" failure.
        c.submitAsync();
        stats_.status = "dispatch_failed";
        stats_.last_tick_ms = msSince(t0);
        return false;
    }
    const std::size_t stats_bytes = static_cast<std::size_t>(n) * kStatsStride * 4;
    uint64_t ticket = 0;
    {
        // The tick's one submission. Its wall time is the host waiting for the
        // backend's submission ring (should be ~0 when the CPU runs ahead).
        RTPERF_FRAME_SCOPE("paint.wet.submit");
        ticket = c.submitWithReadback(d.stats, stats_bytes, 0);
    }
    // This window is read back (delayed or below): the next dispatch starts a new one.
    d.stats_reset_needed = true;
    if (ticket != 0) {
        Device::PendingStats pending;
        pending.ticket = ticket;
        pending.surfaces = n;
        pending.force = force;
        pending.coverage = measure_coverage;
        pending.serial = paint_serial_;
        d.stats_pending.push_back(pending);
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
    d.channels_force = false;
    if (edited_channels) {
        WetPaintChannelEdits::queuePublication(surfaces_, force);
    }
    ++stats_.ticks;   // simulation steps (at most 1 / kWetStepSeconds per second)
    stats_.last_dt = step_dt;
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
    // Never replace the cache with an OLDER measurement (a delayed readback of
    // a tick before a reset or before a synchronous coverage() call).
    if (measured_coverage && (coverage_serial_ == ~0ull || serial >= coverage_serial_)) {
        readCoverage(st, coverage_cache_);
        coverage_serial_ = serial;
        ++stats_.coverage_in_tick;
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
            for (int channel = 0; channel < 4; ++channel) {
                const bool enabled = channel == 0 ? s.settings.roughness_channel
                    : (channel == 3 ? s.settings.reveal_opacity : s.settings.height_normal_channel);
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

void* WetPaintGpu::publicationDevice() const {
    if (!device_ || !device_->compute || layout_dirty_ || surfaces_.empty()) return nullptr;
    const auto& compute = *device_->compute;
    return sameCompute(device_->output, compute) ? compute.nativeDevice() : nullptr;
}

int WetPaintGpu::publish(Backend::IBackend* render_backend, Backend::IBackend* viewport_backend,
                         bool allow_host_copy) {
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
        for (int b = 0; b < 2; ++b) {
            if (!allow_host_copy && !targets[b]) {
                continue;
            }
            any_pending = any_pending || s->pending[b].any;
            for (const auto& r : s->channel_pending[b]) {
                any_pending = any_pending || r.any;
            }
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
            if (!vk) {
                if (allow_host_copy) r = Surface::Rect{};
                continue;
            }
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
            } else if (rc < 0 && allow_host_copy) {
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
        for (uint32_t channel = 0; channel < 4; ++channel) {
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
        s.picture_texels = v[kStPictureTexels];
        s.picture_revealed_texels = v[kStPictureRevealed];
        s.picture_ratio = s.picture_texels
            ? static_cast<float>(s.picture_revealed_texels) / static_cast<float>(s.picture_texels) : 0.0f;
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
                       (static_cast<std::size_t>(si) * kStatsStride + kStCoveredTexels) * 4);
    }
    ComputeBufferHandle bindings[kBindingCount] = {
        d.surfaces, d.tiles, d.pos_valid, d.flow_normal, d.base, d.color, d.state,
        d.color_scratch, d.state_scratch, d.output, d.tile_flags, d.stats,
        d.dummy_events, d.dummy_events, d.dummy_counter, d.dummy_events, d.dummy_counter,
    };
    if (d.zero_init_needed) {
        WetPaintGpuConstants reset_k;
        reset_k.mode = kModeReset;
        reset_k.surface_count = n;
        reset_k.tile_count = d.tile_count;
        reset_k._p0 = d.channels_allocated ? d.texel_count : 0u;
        reset_k._p1 = kResetInitOnlyBit | (d.picture_allocated ? kPictureBit : 0u);
        ComputeDispatch reset_cmd;
        reset_cmd.kernel = "paint_wet";
        reset_cmd.timing_label = kWetPaintTimingLabels[kModeReset];
        reset_cmd.buffers = bindings;
        reset_cmd.buffer_count = kBindingCount;
        reset_cmd.constants = &reset_k;
        reset_cmd.constants_size = sizeof(reset_k);
        reset_cmd.groups.groups_x = std::max<uint32_t>(1u, d.tile_count);
        if (!c.dispatch(reset_cmd)) return false;
        d.zero_init_needed = false;
    }
    WetPaintGpuConstants k;
    k.mode = kModeCoverage;
    k.surface_count = n;
    k.tile_count = d.tile_count;
    k._p0 = d.channels_allocated ? d.texel_count : 0u;
    k._p1 = d.picture_allocated ? kPictureBit : 0u;
    ComputeDispatch cmd;
    cmd.kernel = "paint_wet";
    cmd.timing_label = kWetPaintTimingLabels[kModeCoverage];
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
