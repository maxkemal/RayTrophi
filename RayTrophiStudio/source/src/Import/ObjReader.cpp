/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Import/ObjReader.cpp
* =========================================================================
* Wavefront OBJ + MTL. See Import/ObjReader.h for the six traps this reader
* is written around; the comments below mark where each one is paid.
* =========================================================================
*/
#include "Import/ObjReader.h"

#include "Triangle.h"
#include "TriangleMesh.h"
#include "Transform.h"
// ImportedModel owns a shared_ptr<BoneData>, so the type has to be complete
// here. ★ This used to mean including AssimpLoader.h — a reader replacing
// Assimp depending on Assimp's header. Faz 3 moved the core types out.
#include "Animation/AnimationData.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "Texture.h"
#include "Vec2.h"
#include "globals.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtimport {
namespace {

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// ---------------------------------------------------------------------------
// ★★★ LOCALE-INDEPENDENT SCANNING.
//
// Main.cpp calls setlocale(LC_ALL, "Turkish") at startup, so the C locale's
// decimal separator is a COMMA. strtof/atof/sscanf("%f") would read "1.5" as
// 1.0 and stop at the dot: every fractional coordinate truncated, in a file
// that still parses and still produces a model. std::from_chars is specified to
// ignore the locale entirely, which is why it is used for every number here.
// ---------------------------------------------------------------------------
class Scanner {
public:
    Scanner(const char* begin, const char* end) : p_(begin), end_(end) {}

    void skipSpace() { while (p_ < end_ && (*p_ == ' ' || *p_ == '\t')) ++p_; }

    std::string_view token() {
        skipSpace();
        const char* start = p_;
        while (p_ < end_ && *p_ != ' ' && *p_ != '\t') ++p_;
        return std::string_view(start, size_t(p_ - start));
    }

    // from_chars rejects a leading '+', which OBJ files do occasionally write.
    template <typename T>
    bool value(T& out) {
        skipSpace();
        const char* start = (p_ < end_ && *p_ == '+') ? p_ + 1 : p_;
        const auto result = std::from_chars(start, end_, out);
        if (result.ec != std::errc()) return false;
        p_ = result.ptr;
        return true;
    }

    // A texture path is "whatever is left", never one token: MTL filenames are
    // allowed to contain spaces and frequently do on Windows exports.
    std::string_view rest() {
        skipSpace();
        const char* stop = end_;
        while (stop > p_ && (stop[-1] == ' ' || stop[-1] == '\t')) --stop;
        std::string_view value(p_, size_t(stop - p_));
        p_ = end_;
        return value;
    }

    char peek() const { return p_ < end_ ? *p_ : '\0'; }
    void advance() { if (p_ < end_) ++p_; }
    const char* mark() const { return p_; }
    void reset(const char* to) { p_ = to; }

private:
    const char* p_;
    const char* end_;
};

// ---------------------------------------------------------------------------
// Parsed OBJ payload
// ---------------------------------------------------------------------------
struct Corner {
    int      position  = 0;
    int      uv        = -1;
    int      normal    = -1;
    uint32_t smoothing = 0;
};

struct Part {
    std::string          material;          // MTL name; empty = file default
    std::vector<Corner>  corners;           // triangles, 3 corners each
    std::vector<Vec3>    generatedNormals;  // filled only where `vn` was absent
};

struct Object {
    std::string                            name;
    std::vector<Part>                      parts;
    std::unordered_map<std::string, size_t> index;

    Part& part(const std::string& material) {
        auto it = index.find(material);
        if (it == index.end()) {
            it = index.emplace(material, parts.size()).first;
            parts.push_back(Part{material, {}, {}});
        }
        return parts[it->second];
    }
};

struct Source {
    std::vector<Vec3> positions;
    std::vector<Vec2> uvs;
    std::vector<Vec3> normals;
    std::vector<Vec3> colors;   // only if the file used the 6-float `v` form
};

// OBJ indices are 1-based, and a NEGATIVE index is relative to how many
// elements have been declared SO FAR — which is why this resolves during
// parsing and not afterwards. Reading them as absolute scrambles geometry on
// exports that use the relative form (Blender and Houdini both can).
bool resolve(int raw, size_t count, int& out) {
    if (raw > 0)      out = raw - 1;
    else if (raw < 0) out = int(count) + raw;
    else              return false;
    return out >= 0 && size_t(out) < count;
}

// ---------------------------------------------------------------------------
// MTL
// ---------------------------------------------------------------------------
struct ObjMaterial {
    std::string name;
    Vec3  diffuse{0.8f, 0.8f, 0.8f};
    Vec3  specular{0.0f, 0.0f, 0.0f};
    Vec3  emissive{0.0f, 0.0f, 0.0f};
    float shininess   = 0.0f;  bool hasShininess = false;   // Ns
    float roughness   = 0.0f;  bool hasRoughness = false;   // Pr
    float metallic    = 0.0f;  bool hasMetallic  = false;   // Pm
    float opacity     = 1.0f;
    float ior         = 1.5f;
    int   illum       = -1;
    std::string mapDiffuse, mapSpecular, mapEmissive, mapNormal,
                mapOpacity, mapRoughness, mapMetallic;
};

void consumeNumbers(Scanner& s, int most) {
    for (int i = 0; i < most; ++i) {
        const char* save = s.mark();
        float ignored = 0.0f;
        if (!s.value(ignored)) { s.reset(save); return; }
    }
}

// `map_Kd -bm 0.2 -s 1 1 1 brick wall.png` — options are skipped and whatever
// remains, spaces included, is the filename.
std::string mapPath(Scanner& s) {
    for (;;) {
        s.skipSpace();
        if (s.peek() != '-') break;
        const auto option = s.token();
        if (option == "-blendu" || option == "-blendv" || option == "-cc" ||
            option == "-clamp" || option == "-imfchan" || option == "-type") {
            s.token();                       // one non-numeric argument
        } else if (option == "-mm") {
            consumeNumbers(s, 2);
        } else if (option == "-o" || option == "-s" || option == "-t") {
            consumeNumbers(s, 3);            // 1..3 numbers, whatever parses
        } else {
            consumeNumbers(s, 1);            // -bm, -boost, -texres, unknown
        }
    }
    std::string path(s.rest());
    std::replace(path.begin(), path.end(), '\\', '/');
    return path;
}

// ---------------------------------------------------------------------------
// Materials: parse, prefetch textures in parallel, then build engine materials.
// ---------------------------------------------------------------------------
class ObjMaterials {
public:
    ObjMaterials(const std::string& objPath, const ImportOptions& options, ImportStats& stats)
        : directory_(std::filesystem::path(objPath).parent_path()), options_(options), stats_(stats) {}

    void parseLibrary(const std::string& relative);
    void prefetch();
    uint16_t bind(const std::string& name);

private:
    using TextureKey = std::pair<std::string, int>;
    struct Slot { const std::string ObjMaterial::* path; TextureType type; };

    static std::vector<Slot> slots() {
        return {
            {&ObjMaterial::mapDiffuse,   TextureType::Albedo},
            {&ObjMaterial::mapNormal,    TextureType::Normal},
            {&ObjMaterial::mapSpecular,  TextureType::Specular},
            {&ObjMaterial::mapEmissive,  TextureType::Emission},
            {&ObjMaterial::mapOpacity,   TextureType::Opacity},
            {&ObjMaterial::mapRoughness, TextureType::Roughness},
            {&ObjMaterial::mapMetallic,  TextureType::Metallic},
        };
    }
    std::shared_ptr<Texture> decode(const std::string& path, TextureType type) const;
    std::shared_ptr<Texture> lookup(const std::string& path, TextureType type) const;

    std::filesystem::path                       directory_;
    const ImportOptions&                        options_;
    ImportStats&                                stats_;
    std::vector<std::filesystem::path>          searchPaths_;   // OBJ dir + each MTL dir
    std::map<std::string, ObjMaterial>          materials_;
    std::map<TextureKey, std::shared_ptr<Texture>> textures_;
    std::map<std::string, uint16_t>             bound_;
};

void ObjMaterials::parseLibrary(const std::string& relative) {
    std::string name = relative;
    std::replace(name.begin(), name.end(), '\\', '/');
    std::filesystem::path file(name);
    if (!file.is_absolute()) file = directory_ / file;

    std::ifstream stream(file, std::ios::binary);
    if (!stream) {
        // A missing .mtl is not fatal — the geometry is still valid — but it is
        // the difference between a textured model and a grey one, so it is
        // never silent.
        SCENE_LOG_WARN("[obj] mtllib not found: " + file.string());
        return;
    }
    const auto parent = file.parent_path();
    if (std::find(searchPaths_.begin(), searchPaths_.end(), parent) == searchPaths_.end())
        searchPaths_.push_back(parent);

    std::string data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    ObjMaterial* current = nullptr;
    const char* p = data.data();
    const char* end = p + data.size();
    while (p < end) {
        const char* lineEnd = static_cast<const char*>(std::memchr(p, '\n', size_t(end - p)));
        if (!lineEnd) lineEnd = end;
        const char* stop = lineEnd;
        if (stop > p && stop[-1] == '\r') --stop;
        Scanner s(p, stop);
        p = lineEnd < end ? lineEnd + 1 : end;

        const auto key = s.token();
        if (key.empty() || key[0] == '#') continue;

        if (key == "newmtl") {
            const std::string id(s.rest());
            current = &materials_[id];
            current->name = id;
            continue;
        }
        if (!current) continue;

        auto color = [&](Vec3& target) {
            float r = 0, g = 0, b = 0;
            if (!s.value(r)) return;
            if (!s.value(g)) { g = r; b = r; } else if (!s.value(b)) { b = 0.0f; }
            target = Vec3(r, g, b);
        };
        auto scalar = [&](float& target, bool* flag = nullptr) {
            float v = 0.0f;
            if (s.value(v)) { target = v; if (flag) *flag = true; }
        };

        if      (key == "Kd") color(current->diffuse);
        else if (key == "Ks") color(current->specular);
        else if (key == "Ke") color(current->emissive);
        else if (key == "Ns") scalar(current->shininess, &current->hasShininess);
        else if (key == "Pr") scalar(current->roughness, &current->hasRoughness);
        else if (key == "Pm") scalar(current->metallic,  &current->hasMetallic);
        else if (key == "Ni") scalar(current->ior);
        else if (key == "d")  scalar(current->opacity);
        else if (key == "Tr") { float t = 0.0f; if (s.value(t)) current->opacity = 1.0f - t; }
        else if (key == "illum") { int v = 0; if (s.value(v)) current->illum = v; }
        else if (key == "map_Kd")                       current->mapDiffuse  = mapPath(s);
        else if (key == "map_Ks")                       current->mapSpecular = mapPath(s);
        else if (key == "map_Ke")                       current->mapEmissive = mapPath(s);
        else if (key == "map_d")                        current->mapOpacity  = mapPath(s);
        else if (key == "map_Pr")                       current->mapRoughness = mapPath(s);
        else if (key == "map_Pm")                       current->mapMetallic  = mapPath(s);
        else if (key == "norm" || key == "map_Bump" ||
                 key == "map_bump" || key == "bump")    current->mapNormal   = mapPath(s);
    }
}

std::shared_ptr<Texture> ObjMaterials::decode(const std::string& path, TextureType type) const {
    const std::filesystem::path relative(path);
    std::vector<std::filesystem::path> candidates;
    if (relative.is_absolute()) candidates.push_back(relative);
    for (const auto& root : searchPaths_) {
        candidates.push_back(root / relative);
        // Exports routinely carry the authoring machine's absolute path or a
        // stale subfolder; the basename beside the model is the usual rescue.
        candidates.push_back(root / relative.filename());
    }
    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(candidate, ec))
            return std::make_shared<Texture>(candidate.string(), type);
    }
    return nullptr;
}

// ★ Assimp's importer decoded textures on a thread pool. The first direct
// reader in this repo dropped that and nothing reported it — a lost
// optimisation shows up as no error, no warning, just a slower import.
void ObjMaterials::prefetch() {
    if (!options_.loadMaterials) return;
    std::map<TextureKey, std::pair<std::string, TextureType>> unique;
    for (const auto& [name, material] : materials_)
        for (const auto& slot : slots()) {
            const std::string& path = material.*slot.path;
            if (!path.empty()) unique.emplace(TextureKey{path, int(slot.type)},
                                              std::make_pair(path, slot.type));
        }
    if (unique.empty()) return;

    std::vector<const std::pair<const TextureKey, std::pair<std::string, TextureType>>*> jobs;
    jobs.reserve(unique.size());
    for (const auto& job : unique) jobs.push_back(&job);
    std::vector<std::shared_ptr<Texture>> decoded(jobs.size());
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) return;
            try { decoded[i] = decode(jobs[i]->second.first, jobs[i]->second.second); }
            catch (...) { decoded[i].reset(); }
        }
    };
    // Bound transient decode memory as well as thread count on high-core CPUs.
    const size_t count = (std::min)(jobs.size(),
        size_t((std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()))));
    std::vector<std::thread> workers;
    workers.reserve(count);
    try { for (size_t i = 1; i < count; ++i) workers.emplace_back(worker); }
    catch (...) { /* Started workers and this thread still drain every job. */ }
    worker();
    for (auto& thread : workers) thread.join();

    size_t loaded = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        auto& image = decoded[i];
        if (image && image->is_loaded()) {
            // GPU upload stays serial: it is a driver call, not decode work.
            if (g_hasOptix && isCudaTextureUploadAllowed() && !image->upload_to_gpu())
                SCENE_LOG_WARN("[obj] CUDA texture upload failed: " + image->name);
            textures_[jobs[i]->first] = image;
            ++stats_.image_count;
            ++loaded;
        } else {
            textures_[jobs[i]->first] = nullptr;
            SCENE_LOG_WARN("[obj] Texture missing or undecodable: " + jobs[i]->second.first);
        }
    }
    SCENE_LOG_INFO("[obj] texture prefetch: " + std::to_string(loaded) + "/" +
        std::to_string(jobs.size()) + " image/type pair(s), " +
        std::to_string(workers.size() + 1) + " worker(s)");
}

std::shared_ptr<Texture> ObjMaterials::lookup(const std::string& path, TextureType type) const {
    if (path.empty()) return nullptr;
    const auto it = textures_.find(TextureKey{path, int(type)});
    return it == textures_.end() ? nullptr : it->second;
}

uint16_t ObjMaterials::bind(const std::string& name) {
    const auto cached = bound_.find(name);
    if (cached != bound_.end()) return cached->second;

    const auto found = materials_.find(name);
    const ObjMaterial* m = found == materials_.end() ? nullptr : &found->second;
    if (!name.empty() && !m)
        SCENE_LOG_WARN("[obj] usemtl '" + name + "' has no definition in any mtllib; using defaults.");

    auto pbr = std::make_shared<PrincipledBSDF>();
    const std::string id = options_.importPrefix + "_" +
        (name.empty() ? std::string("DefaultMaterial") : name);
    pbr->materialName = id;
    // OBJ has exactly one UV channel. Unlike glTF/FBX there is no per-texture
    // texCoord index to honour, so this is not a shortcut.
    pbr->selected_uv_set = 0;

    if (m && options_.loadMaterials) {
        // sRGB -> linear, matching AssimpLoader's OBJ path (powf(c, 2.2)) so the
        // two readers can be compared side by side without a gamma difference
        // masquerading as a bug.
        pbr->albedoProperty.color = Vec3(std::pow((std::max)(0.0f, m->diffuse.x), 2.2f),
                                         std::pow((std::max)(0.0f, m->diffuse.y), 2.2f),
                                         std::pow((std::max)(0.0f, m->diffuse.z), 2.2f));
        pbr->albedoProperty.intensity = 1.0f;
        pbr->albedoProperty.texture = lookup(m->mapDiffuse, TextureType::Albedo);

        // ★ EXPECTED DIFFERENCE FROM ASSIMP, AND IT IS DELIBERATE.
        // Assimp reads AI_MATKEY_ROUGHNESS_FACTOR, which for MTL is `Pr` only;
        // a classic MTL that specifies just `Ns` therefore arrives as roughness
        // 0.0 — a perfect mirror. Deriving from the specular exponent (the same
        // conversion UfbxMaterials uses for FBX) is the physically sensible
        // reading. Documented in docs/dev/NEXT_BUILD_CHECKS.md so the
        // difference is not mistaken for an import bug.
        const float fromExponent =
            std::sqrt(2.0f / ((std::max)(0.0f, m->hasShininess ? m->shininess : 6.0f) + 2.0f));
        pbr->roughnessProperty.color = Vec3(1.0f);
        pbr->roughnessProperty.intensity =
            std::clamp(m->hasRoughness ? m->roughness : fromExponent, 0.0f, 1.0f);
        pbr->roughnessProperty.texture = lookup(m->mapRoughness, TextureType::Roughness);

        pbr->metallicProperty.intensity = std::clamp(m->hasMetallic ? m->metallic : 0.0f, 0.0f, 1.0f);
        pbr->metallicProperty.texture = lookup(m->mapMetallic, TextureType::Metallic);

        pbr->specularProperty.intensity =
            std::clamp((std::max)({m->specular.x, m->specular.y, m->specular.z}), 0.0f, 1.0f);
        pbr->specularProperty.texture = lookup(m->mapSpecular, TextureType::Specular);

        pbr->normalProperty.texture = lookup(m->mapNormal, TextureType::Normal);
        pbr->normalProperty.intensity = 1.0f;

        pbr->emissionProperty.color = m->emissive;
        pbr->emissionProperty.intensity = 1.0f;
        pbr->emissionProperty.texture = lookup(m->mapEmissive, TextureType::Emission);
        if (pbr->emissionProperty.texture &&
            m->emissive.x == 0.0f && m->emissive.y == 0.0f && m->emissive.z == 0.0f)
            pbr->emissionProperty.color = Vec3(1.0f);

        pbr->opacityProperty.alpha = std::clamp(m->opacity, 0.0f, 1.0f);
        pbr->opacityProperty.texture = lookup(m->mapOpacity, TextureType::Opacity);
        // Cut-out foliage in OBJ almost always carries its mask in the diffuse
        // map's alpha and never writes map_d.
        if (!pbr->opacityProperty.texture && pbr->albedoProperty.texture &&
            pbr->albedoProperty.texture->has_alpha)
            pbr->opacityProperty.texture = pbr->albedoProperty.texture;

        pbr->setTransmission(0.0f, m->ior);
        // illum 4/6/7 declare refraction. The engine's transmission is not the
        // same parameter as MTL's, and guessing a value would look like a
        // working glass material while being invented — so it is reported.
        if (m->illum == 4 || m->illum == 6 || m->illum == 7)
            SCENE_LOG_WARN("[obj] material '" + id + "' requests illum " +
                std::to_string(m->illum) + " (refraction); transmission is not mapped, "
                "set it on the material if the surface should be glass.");
    }

    pbr->gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbr), *pbr->gpuMaterial);
    const uint16_t assigned = MaterialManager::getInstance().getOrCreateMaterialID(id, pbr);
    if (assigned == MaterialManager::INVALID_MATERIAL_ID)
        throw std::runtime_error("OBJ material capacity exhausted");
    bound_[name] = assigned;
    ++stats_.material_count;
    return assigned;
}

// ---------------------------------------------------------------------------
// ★ Missing normals are generated, and `s` smoothing groups are honoured.
//
// A file with no `vn` at all is completely ordinary. Generating a single
// smoothed normal per position would round off every hard edge on a mechanical
// part; ignoring smoothing groups entirely would facet a character. So the weld
// key is (vertex, smoothing group), and group 0 — "s off" — stays flat.
// ---------------------------------------------------------------------------
void generateNormals(std::vector<Object>& objects, const Source& source) {
    std::unordered_map<uint64_t, Vec3> accumulated;
    auto key = [](int vertex, uint32_t group) {
        return (uint64_t(uint32_t(vertex)) << 32) | uint64_t(group);
    };

    for (auto& object : objects)
        for (auto& part : object.parts) {
            bool needed = false;
            for (const auto& corner : part.corners)
                if (corner.normal < 0) { needed = true; break; }
            if (!needed) continue;

            part.generatedNormals.assign(part.corners.size(), Vec3(0.0f, 0.0f, 0.0f));
            for (size_t t = 0; t + 2 < part.corners.size(); t += 3) {
                const Vec3& a = source.positions[part.corners[t].position];
                const Vec3& b = source.positions[part.corners[t + 1].position];
                const Vec3& c = source.positions[part.corners[t + 2].position];
                // Left un-normalised on purpose: the magnitude is twice the
                // triangle area, which is the weighting a smooth normal wants.
                const Vec3 face = (b - a).cross(c - a);
                for (size_t k = 0; k < 3; ++k) {
                    part.generatedNormals[t + k] = face;
                    const auto& corner = part.corners[t + k];
                    if (corner.smoothing)
                        accumulated[key(corner.position, corner.smoothing)] += face;
                }
            }
        }

    for (auto& object : objects)
        for (auto& part : object.parts) {
            if (part.generatedNormals.empty()) continue;
            for (size_t i = 0; i < part.corners.size(); ++i) {
                const auto& corner = part.corners[i];
                if (corner.normal >= 0 || !corner.smoothing) continue;
                const Vec3 smooth = accumulated[key(corner.position, corner.smoothing)];
                if (smooth.x * smooth.x + smooth.y * smooth.y + smooth.z * smooth.z > 1e-20f)
                    part.generatedNormals[i] = smooth;
            }
            for (auto& normal : part.generatedNormals) normal = normal.normalize();
        }
}

// ---------------------------------------------------------------------------
// One TriangleMesh per (object, material).
// ---------------------------------------------------------------------------
void emit(const Part& part, const std::string& nodeName,
          const std::shared_ptr<Transform>& transform, uint16_t materialId,
          const Source& source, const ImportOptions& options, ImportedModel& out) {
    const size_t count = part.corners.size();
    if (!count) return;

    auto mesh = std::make_shared<TriangleMesh>();
    // ★★ EVERY MATERIAL SUB-MESH OF ONE `o` SHARES THIS NAME. nodeName is a
    // GROUPING KEY, not an identity — per-mesh identity is by pointer. Suffixing
    // it per material silently splits one object into N for scatter, selection
    // and the outliner (docs/dev/FAZ3_DEVIR_NOTU.md §3.1).
    mesh->nodeName = nodeName;
    mesh->transform = transform;

    auto& geo = *mesh->geometry;
    geo.resize_vertices(count);
    for (const char* attribute : {"P", "N", "P_orig", "N_orig"}) geo.add_attribute<Vec3>(attribute);
    geo.add_attribute<Vec2>("uv");
    geo.add_attribute<uint16_t>("materialID");
    auto* p   = geo.get_attribute_data_mut<Vec3>("P");
    auto* n   = geo.get_attribute_data_mut<Vec3>("N");
    auto* po  = geo.get_attribute_data_mut<Vec3>("P_orig");
    auto* no  = geo.get_attribute_data_mut<Vec3>("N_orig");
    auto* uv  = geo.get_attribute_data_mut<Vec2>("uv");
    auto* ids = geo.get_attribute_data_mut<uint16_t>("materialID");

    Vec3* color = nullptr;
    if (!source.colors.empty()) {
        geo.add_attribute<Vec3>("Cd");
        color = geo.get_attribute_data_mut<Vec3>("Cd");
    }

    for (size_t v = 0; v < count; ++v) {
        const auto& corner = part.corners[v];
        p[v] = po[v] = source.positions[corner.position];
        n[v] = no[v] = corner.normal >= 0 ? source.normals[corner.normal].normalize()
                                          : (part.generatedNormals.empty() ? Vec3(0.0f, 1.0f, 0.0f)
                                                                           : part.generatedNormals[v]);
        // ★ NO V FLIP. OBJ's texture origin is bottom-left, the same as FBX and
        // the same as this engine. glTF is the format that differs, which is
        // why GltfDirectReader flips and this one must not.
        uv[v] = corner.uv >= 0 ? source.uvs[corner.uv] : Vec2(0.0f, 0.0f);
        ids[v] = materialId;
        if (color) color[v] = source.colors[corner.position];
    }

    // Existing TriangleMesh consumers require an address table. It is strictly
    // sequential: the canonical geometry above is flat, one vertex per corner.
    geo.indices.resize(count);
    std::iota(geo.indices.begin(), geo.indices.end(), uint32_t(0));

    const size_t faces = count / 3;
    // OBJ carries no skinning, so the representative-facade contract applies
    // unconditionally here (see ImportOptions::emitSingleFacadePerMesh).
    const size_t facades = options.emitSingleFacadePerMesh ? 1 : faces;
    for (size_t f = 0; f < facades; ++f)
        out.objects.push_back(std::make_shared<Triangle>(mesh, uint32_t(f)));

    ++out.stats.mesh_count;
    out.stats.vertex_count += count;
    out.stats.triangle_count += faces;
}

} // namespace

// ---------------------------------------------------------------------------
// Counts and a box, without building a scene. See Import/ModelProbe.h.
//
// ★ Deliberately a second, much smaller pass rather than a call to readObj():
// a probe must not register materials in MaterialManager or allocate meshes.
// The asset browser scans a whole library; the old Assimp path built the entire
// scene to report a number.
// ---------------------------------------------------------------------------
bool probeObj(const std::string& path, bool, ModelProbe& out) {
    out = ModelProbe{};
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    std::string data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (data.empty()) return false;

    const std::filesystem::path directory = std::filesystem::path(path).parent_path();
    std::vector<std::string> libraries;
    std::unordered_map<std::string, bool> usedMaterials;
    std::unordered_map<std::string, bool> groups;
    float lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};

    const char* p = data.data();
    const char* end = p + data.size();
    while (p < end) {
        const char* lineEnd = static_cast<const char*>(std::memchr(p, '\n', size_t(end - p)));
        if (!lineEnd) lineEnd = end;
        const char* stop = lineEnd;
        if (stop > p && stop[-1] == '\r') --stop;
        Scanner s(p, stop);
        p = lineEnd < end ? lineEnd + 1 : end;

        const auto key = s.token();
        if (key.empty() || key[0] == '#') continue;

        if (key == "v") {
            float xyz[3] = {0, 0, 0};
            if (!s.value(xyz[0]) || !s.value(xyz[1]) || !s.value(xyz[2])) continue;
            for (int a = 0; a < 3; ++a) {
                if (!out.has_bounds) { lo[a] = hi[a] = xyz[a]; }
                else { lo[a] = (std::min)(lo[a], xyz[a]); hi[a] = (std::max)(hi[a], xyz[a]); }
            }
            out.has_bounds = true;
            ++out.vertex_count;
            continue;
        }
        if (key == "f") {
            // Corner count minus two, matching the fan triangulation the reader
            // performs — the browser must not report a different number from
            // the one the import produces.
            uint64_t corners = 0;
            for (;;) {
                s.skipSpace();
                const char c = s.peek();
                if (c != '-' && c != '+' && (c < '0' || c > '9')) break;
                int ignored = 0;
                if (!s.value(ignored)) break;
                while (s.peek() == '/' || (s.peek() >= '0' && s.peek() <= '9') ||
                       s.peek() == '-' || s.peek() == '+') s.advance();
                ++corners;
            }
            if (corners >= 3) out.triangle_count += corners - 2;
            continue;
        }
        if (key == "usemtl") { usedMaterials[std::string(s.rest())] = true; continue; }
        if (key == "o" || key == "g") { groups[std::string(s.rest())] = true; continue; }
        if (key == "mtllib") {
            for (;;) {
                const auto name = s.token();
                if (name.empty()) break;
                libraries.push_back(std::string(name));
            }
            continue;
        }
    }

    // Texture references come from the .mtl files, counted the same way the
    // reader consumes them: one per map_* statement it can actually use.
    for (const auto& library : libraries) {
        std::string name = library;
        std::replace(name.begin(), name.end(), '\\', '/');
        std::filesystem::path file(name);
        if (!file.is_absolute()) file = directory / file;
        std::ifstream mtl(file, std::ios::binary);
        if (!mtl) continue;
        std::string line;
        while (std::getline(mtl, line)) {
            Scanner s(line.data(), line.data() + line.size());
            const auto key = s.token();
            if (key == "map_Kd" || key == "map_Ks" || key == "map_Ke" || key == "map_d" ||
                key == "map_Pr" || key == "map_Pm" || key == "norm" || key == "map_Bump" ||
                key == "map_bump" || key == "bump")
                ++out.texture_reference_count;
        }
    }

    out.material_count = usedMaterials.size();
    // One mesh per (group, material) pair is what the reader emits; without
    // parsing that association here, the group count is the honest estimate and
    // matches the OBJECT count the outliner will show.
    out.mesh_count = groups.empty() ? (out.triangle_count ? 1 : 0) : groups.size();
    out.node_count = out.mesh_count;
    for (int a = 0; a < 3; ++a) { out.bounds_min[a] = lo[a]; out.bounds_max[a] = hi[a]; }
    return out.has_bounds || out.triangle_count > 0;
}

bool readObj(const std::string& path, const ImportOptions& requested,
             ImportedModel& out, std::string& error) {
    out = {};
    error.clear();
    try {
        const auto start = Clock::now();
        // OBJ carries nothing BUT geometry and materials, so a geometry-less
        // request has no meaning here. Refused up front rather than after the
        // parse: returning an empty model would look like a successful import
        // of a file that happened to contain nothing.
        if (!requested.loadGeometry) {
            error = "obj: loadGeometry=false leaves nothing to import from " + path;
            return false;
        }
        std::ifstream stream(path, std::ios::binary);
        if (!stream) { error = "obj: cannot open " + path; return false; }
        std::string data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        if (data.empty()) { error = "obj: empty file " + path; return false; }

        // OBJ allows a trailing backslash to continue a statement on the next
        // line. Rare, but unhandled it truncates faces rather than failing.
        if (data.find("\\\n") != std::string::npos) {
            std::string merged;
            merged.reserve(data.size());
            for (size_t i = 0; i < data.size();) {
                if (data[i] == '\\') {
                    size_t j = i + 1;
                    if (j < data.size() && data[j] == '\r') ++j;
                    if (j < data.size() && data[j] == '\n') { merged.push_back(' '); i = j + 1; continue; }
                }
                merged.push_back(data[i++]);
            }
            data.swap(merged);
        }

        ImportOptions options = requested;
        if (options.importPrefix.empty()) {
            const std::filesystem::path file(path);
            options.importPrefix = file.parent_path().filename().string() + "_" + file.stem().string();
        }

        ImportedModel model;
        model.importName = options.importPrefix;
        model.stats.reader = "obj";

        Source source;
        std::vector<Object> objects;
        std::unordered_map<std::string, size_t> objectIndex;
        ObjMaterials materials(path, options, model.stats);

        // Decided before parsing, because the grouping rule for `g` depends on
        // whether an `o` appears ANYWHERE in the file (see the `g` case below).
        bool hasObjectKeyword = false;
        for (size_t i = 0; i + 1 < data.size(); ++i) {
            if (data[i] != 'o') continue;
            if (i && data[i - 1] != '\n' && data[i - 1] != '\r') continue;
            if (data[i + 1] == ' ' || data[i + 1] == '\t') { hasObjectKeyword = true; break; }
        }

        std::string currentObject;
        std::string currentMaterial;
        uint32_t    currentSmoothing = 0;
        size_t      degenerate = 0;
        size_t      badIndices = 0;

        auto object = [&]() -> Object& {
            auto it = objectIndex.find(currentObject);
            if (it == objectIndex.end()) {
                it = objectIndex.emplace(currentObject, objects.size()).first;
                objects.push_back(Object{currentObject, {}, {}});
            }
            return objects[it->second];
        };

        const char* p = data.data();
        const char* end = p + data.size();
        std::vector<Corner> face;
        while (p < end) {
            const char* lineEnd = static_cast<const char*>(std::memchr(p, '\n', size_t(end - p)));
            if (!lineEnd) lineEnd = end;
            const char* stop = lineEnd;
            if (stop > p && stop[-1] == '\r') --stop;
            Scanner s(p, stop);
            p = lineEnd < end ? lineEnd + 1 : end;

            const auto key = s.token();
            if (key.empty() || key[0] == '#') continue;

            if (key == "v") {
                float x = 0, y = 0, z = 0;
                if (!s.value(x) || !s.value(y) || !s.value(z)) continue;
                source.positions.emplace_back(x, y, z);
                // The 6-float form `v x y z r g b` is a widely used extension.
                float r = 0, g = 0, b = 0;
                if (s.value(r) && s.value(g) && s.value(b)) {
                    source.colors.resize(source.positions.size() - 1, Vec3(1.0f, 1.0f, 1.0f));
                    source.colors.emplace_back(r, g, b);
                } else if (!source.colors.empty()) {
                    source.colors.emplace_back(1.0f, 1.0f, 1.0f);
                }
                continue;
            }
            if (key == "vt") {
                float u = 0, v = 0;
                if (!s.value(u) || !s.value(v)) continue;
                source.uvs.emplace_back(u, v);
                continue;
            }
            if (key == "vn") {
                float x = 0, y = 0, z = 0;
                if (!s.value(x) || !s.value(y) || !s.value(z)) continue;
                source.normals.emplace_back(x, y, z);
                continue;
            }
            if (key == "o") { currentObject = std::string(s.rest()); continue; }
            if (key == "g") {
                // ★★ `g` GROUPS ONLY WHEN THE FILE HAS NO `o` AT ALL.
                // Plenty of exporters emit both: one `o` for the object and a
                // `g` per material group INSIDE it. Treating every `g` as an
                // object would split a multi-material model into N objects —
                // precisely the failure nodeName grouping exists to prevent,
                // and one that looks perfectly fine in the viewport. Files that
                // only ever use `g` still group correctly.
                if (!hasObjectKeyword) currentObject = std::string(s.rest());
                continue;
            }
            if (key == "usemtl") { currentMaterial = std::string(s.rest()); continue; }
            if (key == "mtllib") {
                // A single line may list several libraries.
                for (;;) {
                    const auto name = s.token();
                    if (name.empty()) break;
                    materials.parseLibrary(std::string(name));
                }
                continue;
            }
            if (key == "s") {
                const auto value = s.token();
                if (value == "off" || value == "0" || value.empty()) currentSmoothing = 0;
                else {
                    uint32_t group = 0;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), group);
                    currentSmoothing = result.ec == std::errc() ? group : 1;
                }
                continue;
            }
            if (key != "f") continue;

            face.clear();
            for (;;) {
                s.skipSpace();
                const char digit = s.peek();
                if (digit != '-' && digit != '+' && (digit < '0' || digit > '9')) break;
                int rawP = 0, rawT = 0, rawN = 0;
                if (!s.value(rawP)) break;
                if (s.peek() == '/') {
                    s.advance();
                    if (s.peek() != '/') { if (!s.value(rawT)) rawT = 0; }
                    if (s.peek() == '/') { s.advance(); if (!s.value(rawN)) rawN = 0; }
                }
                Corner corner;
                corner.smoothing = currentSmoothing;
                if (!resolve(rawP, source.positions.size(), corner.position)) { ++badIndices; face.clear(); break; }
                if (rawT && !resolve(rawT, source.uvs.size(), corner.uv)) corner.uv = -1;
                if (rawN && !resolve(rawN, source.normals.size(), corner.normal)) corner.normal = -1;
                face.push_back(corner);
            }
            if (face.size() < 3) { if (face.size()) ++degenerate; continue; }

            // Fan triangulation. OBJ polygons are planar and convex by
            // specification; quads are by far the common case.
            auto& target = object().part(currentMaterial);
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                target.corners.push_back(face[0]);
                target.corners.push_back(face[i]);
                target.corners.push_back(face[i + 1]);
            }
        }
        model.stats.seconds_parse = elapsed(start);

        if (source.positions.empty()) { error = "obj: no vertices in " + path; return false; }
        if (objects.empty())          { error = "obj: no faces in " + path; return false; }
        if (badIndices)
            SCENE_LOG_WARN("[obj] " + std::to_string(badIndices) +
                " face(s) referenced an out-of-range vertex and were skipped.");
        if (degenerate)
            SCENE_LOG_WARN("[obj] " + std::to_string(degenerate) + " face(s) had fewer than 3 corners.");
        if (!source.colors.empty()) source.colors.resize(source.positions.size(), Vec3(1.0f, 1.0f, 1.0f));

        const auto materialStart = Clock::now();
        materials.prefetch();
        model.stats.seconds_materials = elapsed(materialStart);

        const auto geometryStart = Clock::now();
        generateNormals(objects, source);

        // OBJ has no node transforms and no animation: one root plus one node
        // per object, all identity.
        const int root = model.hierarchy.addNode(options.importPrefix, options.importPrefix,
                                                 Matrix4x4::identity(), -1);
        std::unordered_map<std::string, size_t> usedNames;
        for (auto& entry : objects) {
            const std::string raw = entry.name.empty() ? "Object" : entry.name;
            std::string nodeName = options.importPrefix + "_" + raw;
            const auto collision = usedNames.find(nodeName);
            if (collision == usedNames.end()) usedNames.emplace(nodeName, 0);
            else nodeName += "_" + std::to_string(++collision->second);
            model.hierarchy.addNode(raw, nodeName, Matrix4x4::identity(), root);

            // ★ ONE Transform for the whole object, shared by every material
            // sub-mesh, so moving the object in the editor moves all of it.
            auto transform = std::make_shared<Transform>();
            transform->setBase(Matrix4x4::identity());
            for (const auto& part : entry.parts) {
                const uint16_t id = options.loadMaterials ? materials.bind(part.material)
                                                          : materials.bind(std::string());
                emit(part, nodeName, transform, id, source, options, model);
            }
        }
        model.stats.node_count = model.hierarchy.size();
        model.stats.seconds_geometry = elapsed(geometryStart);
        model.stats.seconds_total = elapsed(start);

        out = std::move(model);
        return true;
    } catch (const std::exception& e) {
        out = {};
        error = std::string("obj: ") + e.what();
        return false;
    }
}

} // namespace rtimport
