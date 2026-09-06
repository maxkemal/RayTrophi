#include "UfbxMaterials.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "globals.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <thread>

namespace rtimport {
namespace {
std::string str(ufbx_string s) { return std::string(s.data ? s.data : "", s.length); }
bool present(const ufbx_material_map& m) { return m.has_value || (m.texture_enabled && m.texture); }
const ufbx_material_map& choose(const ufbx_material_map& pbr, const ufbx_material_map& fbx) {
    return present(pbr) ? pbr : fbx;
}
float scalar(const ufbx_material_map& m, float fallback) {
    return m.has_value ? float(m.value_real) : fallback;
}
Vec3 color(const ufbx_material_map& m, Vec3 fallback) {
    return m.has_value ? Vec3(float(m.value_vec3.x), float(m.value_vec3.y), float(m.value_vec3.z)) : fallback;
}
struct Slot { const ufbx_material_map* map; TextureType type; };
std::vector<Slot> slots(const ufbx_material& m) {
    return {
        {&choose(m.pbr.base_color, m.fbx.diffuse_color), TextureType::Albedo},
        {&m.pbr.roughness, TextureType::Roughness},
        {&m.pbr.metalness, TextureType::Metallic},
        {&choose(m.pbr.normal_map, m.fbx.normal_map), TextureType::Normal},
        {&choose(m.pbr.emission_color, m.fbx.emission_color), TextureType::Emission},
        {&choose(m.pbr.specular_factor, m.fbx.specular_factor), TextureType::Specular},
        {&m.pbr.opacity, TextureType::Opacity},
        {&m.pbr.transmission_factor, TextureType::Transmission},
    };
}
const ufbx_texture* fileTexture(const ufbx_material_map& map) {
    if (!map.texture_enabled || !map.texture) return nullptr;
    // Layer composition/procedurals cannot be represented by a single sampler.
    return map.texture->type == UFBX_TEXTURE_FILE ? map.texture : nullptr;
}
}
UfbxMaterials::UfbxMaterials(const std::string& path, const ImportOptions& options, ImportStats& stats)
    : directory_(std::filesystem::path(path).parent_path().string()), options_(options), stats_(stats) {}

UfbxMaterials::TextureKey UfbxMaterials::key(const ufbx_texture& t, TextureType type) const {
    // Embedded content IDs are scene-local. External paths are normalized so
    // multiple FBX texture objects pointing to one file share the decoded pixels.
    const std::string identity = t.content.size ?
        (t.has_file ? "embedded-file:" + std::to_string(t.file_index) : "embedded-texture:" + std::to_string(t.typed_id)) :
        std::filesystem::path(str(t.filename)).lexically_normal().generic_string();
    return {identity.empty() ? "texture:" + std::to_string(t.typed_id) : identity, int(type)};
}
std::shared_ptr<Texture> UfbxMaterials::decode(const ufbx_texture& t, TextureType type) const {
    if (t.content.size) {
        if (!t.content.data || t.content.size > size_t((std::numeric_limits<int>::max)())) return nullptr;
        const char* first = static_cast<const char*>(t.content.data);
        return std::make_shared<Texture>(std::vector<char>(first, first + t.content.size), type,
            options_.importPrefix + "_image_" + std::to_string(t.typed_id));
    }
    std::vector<std::filesystem::path> candidates;
    for (auto name : {t.filename, t.relative_filename, t.absolute_filename}) {
        if (!name.length) continue;
        std::string normalized = str(name);
        std::replace(normalized.begin(), normalized.end(), '\\', '/');
        const std::filesystem::path p(normalized);
        candidates.push_back(p.is_absolute() ? p : std::filesystem::path(directory_) / p);
        candidates.push_back(std::filesystem::path(directory_) / p.filename());
    }
    for (const auto& p : candidates) {
        std::error_code ec;
        if (std::filesystem::is_regular_file(p, ec)) return std::make_shared<Texture>(p.string(), type);
    }
    return nullptr;
}
void UfbxMaterials::prefetch(const ufbx_scene& scene) {
    if (!options_.loadMaterials) return;
    std::map<TextureKey, std::pair<const ufbx_texture*, TextureType>> unique;
    for (const auto* node : scene.nodes) {
        if (!node->mesh || !node->mesh->num_triangles) continue;
        for (const auto& part : node->mesh->material_parts) {
            if (!part.num_triangles || part.index >= node->materials.count) continue;
            for (const auto slot : slots(*node->materials[part.index]))
                if (const auto* t = fileTexture(*slot.map)) unique.emplace(key(*t, slot.type), std::make_pair(t, slot.type));
        }
    }
    std::vector<decltype(unique)::value_type const*> jobs;
    for (const auto& job : unique) jobs.push_back(&job);
    std::vector<std::shared_ptr<Texture>> decoded(jobs.size());
    std::atomic<size_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobs.size()) return;
            try { decoded[i] = decode(*jobs[i]->second.first, jobs[i]->second.second); }
            catch (...) { decoded[i].reset(); }
        }
    };
    // Bound transient decode memory as well as thread count on high-core CPUs.
    const size_t count = (std::min)(jobs.size(), size_t((std::min)(8u, (std::max)(1u, std::thread::hardware_concurrency()))));
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
            if (g_hasOptix && isCudaTextureUploadAllowed() && !image->upload_to_gpu())
                SCENE_LOG_WARN("[ufbx] CUDA texture upload failed: " + image->name);
            textures_[jobs[i]->first] = image;
            ++stats_.image_count;
            ++loaded;
        } else {
            textures_[jobs[i]->first] = nullptr;
            SCENE_LOG_WARN("[ufbx] Texture missing or undecodable: " + str(jobs[i]->second.first->filename));
        }
    }
    SCENE_LOG_INFO("[ufbx] texture prefetch: " + std::to_string(loaded) + "/" +
        std::to_string(jobs.size()) + " image/type pair(s), " + std::to_string(workers.size() + 1) + " worker(s)");
}
std::shared_ptr<Texture> UfbxMaterials::texture(const ufbx_material_map& map, TextureType type) {
    const auto* source = fileTexture(map);
    if (!source) return nullptr;
    const auto it = textures_.find(key(*source, type));
    return it == textures_.end() ? nullptr : it->second;
}
FbxMaterialBinding UfbxMaterials::bind(const ufbx_material* m, const ufbx_mesh& mesh) {
    size_t uvSet = 0;
    bool chosen = false;
    bool mismatch = false;
    if (options_.loadMaterials && m) for (const auto slot : slots(*m)) {
        const auto* t = fileTexture(*slot.map);
        if (!t) continue;
        size_t set = 0;
        if (t->uv_set.length) {
            bool found = false;
            for (size_t i = 0; i < mesh.uv_sets.count; ++i)
                if (str(mesh.uv_sets[i].name) == str(t->uv_set)) { set = i; found = true; break; }
            if (!found) SCENE_LOG_WARN("[ufbx] UV set '" + str(t->uv_set) + "' missing on mesh '" + str(mesh.name) + "'; using set 0.");
        }
        if (!chosen) { uvSet = set; chosen = true; }
        else mismatch |= uvSet != set;
    }
    const auto cacheKey = std::make_pair(m, uvSet);
    const auto previous = materials_.find(cacheKey);
    if (previous != materials_.end()) return {previous->second, uvSet};
    auto pbr = std::make_shared<PrincipledBSDF>();
    const std::string name = options_.importPrefix + "_" + (m ? str(m->name) + "_" + std::to_string(m->typed_id) : "DefaultMaterial") + "_uv" + std::to_string(uvSet);
    pbr->materialName = name;
    pbr->selected_uv_set = int(uvSet);
    if (m && options_.loadMaterials) {
        const auto& base = choose(m->pbr.base_color, m->fbx.diffuse_color);
        Vec3 baseColor = color(base, Vec3(0.8f));
        // Match the existing FBX diffuse-color convention (authored sRGB).
        baseColor = Vec3(std::pow((std::max)(0.0f, baseColor.x), 2.2f),
                         std::pow((std::max)(0.0f, baseColor.y), 2.2f),
                         std::pow((std::max)(0.0f, baseColor.z), 2.2f));
        pbr->albedoProperty.color = baseColor * scalar(choose(m->pbr.base_factor, m->fbx.diffuse_factor), 1.0f);
        pbr->albedoProperty.intensity = 1.0f;
        pbr->albedoProperty.texture = texture(base, TextureType::Albedo);
        pbr->roughnessProperty.color = Vec3(1.0f);
        pbr->roughnessProperty.intensity = std::clamp(scalar(m->pbr.roughness,
            std::sqrt(2.0f / ((std::max)(0.0f, scalar(m->fbx.specular_exponent, 6.0f)) + 2.0f))), 0.0f, 1.0f);
        pbr->roughnessProperty.texture = texture(m->pbr.roughness, TextureType::Roughness);
        pbr->metallicProperty.intensity = std::clamp(scalar(m->pbr.metalness, 0.0f), 0.0f, 1.0f);
        pbr->metallicProperty.texture = texture(m->pbr.metalness, TextureType::Metallic);
        const auto& specular = choose(m->pbr.specular_factor, m->fbx.specular_factor);
        pbr->specularProperty.intensity = scalar(specular, 0.5f);
        pbr->specularProperty.texture = texture(specular, TextureType::Specular);
        pbr->normalProperty.texture = texture(choose(m->pbr.normal_map, m->fbx.normal_map), TextureType::Normal);
        pbr->normalProperty.intensity = 1.0f;
        const auto& emission = choose(m->pbr.emission_color, m->fbx.emission_color);
        pbr->emissionProperty.color = color(emission, Vec3(0.0f));
        pbr->emissionProperty.intensity = scalar(choose(m->pbr.emission_factor, m->fbx.emission_factor), 1.0f);
        pbr->emissionProperty.texture = texture(emission, TextureType::Emission);
        if (pbr->emissionProperty.texture && !emission.has_value) pbr->emissionProperty.color = Vec3(1.0f);
        // Classic FBX transparency is color * factor. Factor alone can be 1
        // on an opaque material whose transparency color is black.
        const Vec3 transparent = color(m->fbx.transparency_color,
            Vec3(m->fbx.transparency_factor.has_value ? 1.0f : 0.0f));
        const float transparency = (std::max)({transparent.x, transparent.y, transparent.z}) *
            scalar(m->fbx.transparency_factor, 1.0f);
        pbr->opacityProperty.alpha = std::clamp(scalar(m->pbr.opacity, 1.0f - transparency), 0.0f, 1.0f);
        pbr->opacityProperty.texture = texture(m->pbr.opacity, TextureType::Opacity);
        if (!pbr->opacityProperty.texture && pbr->albedoProperty.texture && pbr->albedoProperty.texture->has_alpha)
            pbr->opacityProperty.texture = pbr->albedoProperty.texture;
        pbr->setTransmission(scalar(m->pbr.transmission_factor, 0.0f), scalar(m->pbr.specular_ior, 1.5f));
        pbr->transmissionProperty.texture = texture(m->pbr.transmission_factor, TextureType::Transmission);
        pbr->setClearcoat(scalar(m->pbr.coat_factor, 0.0f), scalar(m->pbr.coat_roughness, 0.03f));
        if (mismatch) SCENE_LOG_WARN("[ufbx] Material '" + name + "' uses different UV sets per slot; first textured slot wins.");
        for (const auto slot : slots(*m)) if (slot.map->texture_enabled && slot.map->texture) {
            if (!fileTexture(*slot.map)) SCENE_LOG_WARN("[ufbx] Layered/procedural texture unsupported on '" + name + "'.");
            else if (slot.map->texture->has_uv_transform)
                SCENE_LOG_WARN("[ufbx] Texture UV transform not represented on '" + name + "'.");
        }
        if (present(m->fbx.bump) || present(m->pbr.displacement_map) || present(m->pbr.ambient_occlusion))
            SCENE_LOG_WARN("[ufbx] Bump/displacement/AO maps are not mapped by this increment: '" + name + "'.");
        if (m->fbx.transparency_color.texture || m->fbx.specular_color.texture ||
            m->pbr.specular_color.texture || scalar(m->pbr.specular_anisotropy, 0.0f) != 0.0f)
            SCENE_LOG_WARN("[ufbx] Transparency/specular color maps or anisotropy are not mapped by this increment: '" + name + "'.");
    }
    pbr->gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbr), *pbr->gpuMaterial);
    const uint16_t id = MaterialManager::getInstance().getOrCreateMaterialID(name, pbr);
    if (id == MaterialManager::INVALID_MATERIAL_ID) throw std::runtime_error("FBX material capacity exhausted");
    materials_[cacheKey] = id;
    ++stats_.material_count;
    return {id, uvSet};
}
}
