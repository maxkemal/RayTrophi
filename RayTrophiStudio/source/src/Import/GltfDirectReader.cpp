/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Import/GltfDirectReader.cpp
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*
* Direct glTF 2.0 / GLB reader. See Import/GltfDirectReader.h for why cgltf and
* not tiny_gltf, and for the conventions that must not drift from the Assimp
* path.
* =========================================================================
*/
#include <charconv>
#include <cstring>
#include <system_error>

namespace {
// JSON always uses a decimal point. Main sets LC_ALL to Turkish, so cgltf's
// default atof truncates fractional node matrices/TRS while binary accessors
// remain intact. Do not change process locale around imports: other threads
// use it too. from_chars parses JSON numbers independently of locale.
thread_local bool gltfJsonNumberInvalid = false;

double gltfJsonNumber(const char* text) {
    const char* end = text + std::strlen(text);
    double value = 0.0;
    const auto parsed = std::from_chars(text, end, value, std::chars_format::general);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        // Let cgltf finish/release its C allocations before rejecting the file.
        gltfJsonNumberInvalid = true;
        return 0.0;
    }
    return value;
}
} // namespace

#define CGLTF_ATOF(text) gltfJsonNumber(text)
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#undef CGLTF_ATOF

#include "Import/GltfDirectReader.h"

#include "Animation/AnimationData.h"   // AnimationData, BoneData
#include "Triangle.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "Material.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "PBRMaterialSnapshot.h"
#include "Texture.h"
#include "Camera.h"
#include "Light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "globals.h"

#include <atomic>
#include <thread>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace rtimport {

namespace {

using Clock = std::chrono::steady_clock;
inline double secondsSince(const Clock::time_point& t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

// ---------------------------------------------------------------------------
// ★★★ WHY cgltf_validate IS NOT THE GATE ANY MORE.
//
// cgltf_validate answers "is this file spec-conformant?" — one bool for the
// whole file. We need a DIFFERENT question: "can we read this without running
// off the end of a buffer?" Those are not the same question, and treating the
// first as the second turned real assets into hard failures.
//
// Measured case (2026-09-05, a 627 MB Unreal Engine 5.2.1 export):
//   574 meshes / 1141 primitives, of which 566 are EMPTY — a second primitive
//   per mesh whose POSITION/NORMAL/TANGENT/TEXCOORD accessors all have
//   "count": 0 and no indices. UE writes one primitive per material slot and
//   emits the slot even when the LOD section has no geometry.
//
// The spec says accessor.count must be >= 1, so cgltf is RIGHT to refuse it —
// and the file is still perfectly readable: emitPrimitive() already returns
// early on a zero-count POSITION, so those 566 primitives simply produce
// nothing. Assimp read this file for years by being lenient; the whole file
// became unopenable the moment the fallback was removed, which is exactly the
// risk the Faz 2 checklist flagged.
//
// So: keep cgltf_validate as the fast path, and when it refuses, check the
// subset that actually protects MEMORY. Every rule below exists because the
// reader dereferences that data:
//   - accessor ranges: cgltf_accessor_unpack_* reads count*stride bytes with no
//     bounds check of its own,
//   - buffer view ranges: same, one level down,
//   - index component type: cgltf_accessor_read_index switches on it.
// Anything cgltf rejects that is NOT in this list is a spec violation we can
// survive, and we say so in the log rather than refusing the file.
// ---------------------------------------------------------------------------
bool validateReadSafety(const cgltf_data* d, std::string& why) {
    if (!d) { why = "no data"; return false; }

    for (cgltf_size i = 0; i < d->buffer_views_count; ++i) {
        const cgltf_buffer_view& v = d->buffer_views[i];
        if (!v.buffer) { why = "buffer_view " + std::to_string(i) + " has no buffer"; return false; }
        if (v.offset + v.size > v.buffer->size) {
            why = "buffer_view " + std::to_string(i) + " runs past the end of its buffer";
            return false;
        }
    }

    for (cgltf_size i = 0; i < d->accessors_count; ++i) {
        const cgltf_accessor& a = d->accessors[i];
        if (a.component_type == cgltf_component_type_invalid || a.type == cgltf_type_invalid) {
            why = "accessor " + std::to_string(i) + " has an invalid component type";
            return false;
        }
        if (!a.buffer_view || a.count == 0) continue;   // zero-count is the benign case
        const cgltf_size elem = cgltf_calc_size(a.type, a.component_type);
        const cgltf_size stride = a.stride ? a.stride : elem;
        const cgltf_size need = a.offset + stride * (a.count - 1) + elem;
        if (need > a.buffer_view->size) {
            why = "accessor " + std::to_string(i) + " reads " + std::to_string(need) +
                  " bytes from a " + std::to_string(a.buffer_view->size) + "-byte view";
            return false;
        }
    }

    for (cgltf_size m = 0; m < d->meshes_count; ++m) {
        const cgltf_mesh& mesh = d->meshes[m];
        for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
            const cgltf_accessor* idx = mesh.primitives[p].indices;
            if (!idx) continue;
            if (idx->type != cgltf_type_scalar) {
                why = "mesh " + std::to_string(m) + " primitive " + std::to_string(p) +
                      " has non-scalar indices";
                return false;
            }
            if (idx->component_type != cgltf_component_type_r_8u &&
                idx->component_type != cgltf_component_type_r_16u &&
                idx->component_type != cgltf_component_type_r_32u) {
                why = "mesh " + std::to_string(m) + " primitive " + std::to_string(p) +
                      " has a non-integer index accessor";
                return false;
            }
        }
    }

    return true;
}

// How many primitives carry no vertices at all. Reported, not fixed: they are
// legal to skip and the count is the difference between "the exporter emits
// empty material slots" and "we are silently dropping geometry".
cgltf_size countEmptyPrimitives(const cgltf_data* d) {
    cgltf_size empty = 0;
    for (cgltf_size m = 0; m < d->meshes_count; ++m) {
        const cgltf_mesh& mesh = d->meshes[m];
        for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
            const cgltf_primitive& prim = mesh.primitives[p];
            bool hasVerts = false;
            for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
                if (prim.attributes[a].type == cgltf_attribute_type_position &&
                    prim.attributes[a].data && prim.attributes[a].data->count > 0) {
                    hasVerts = true;
                    break;
                }
            }
            if (!hasVerts) ++empty;
        }
    }
    return empty;
}



// ---------------------------------------------------------------------------
// ★★★ WHICH UV SET A MATERIAL SAMPLES — glTF's `texCoord`, WHICH WE IGNORED.
//
// Every glTF textureInfo carries `texCoord: N`, meaning "sample TEXCOORD_N".
// It is plain spec, not an extension, and the reader always sampled TEXCOORD_0.
//
// Measured on a 627 MB Unreal Engine 5.2.1 asset: of 248 materials that own
// textures, **241 use texCoord 1** and only 7 use 0. So nearly every textured
// object in that file sampled the wrong UV set — which does not look like an
// error, it looks like "the texture did not come through".
//
// ★ AssimpLoader has ALWAYS read this (AI_MATKEY_UVWSRC -> selected_uv_set);
// it simply was not carried over to the direct reader. Same family as the lost
// parallel texture prefetch: nothing broke, the file just came in wrong.
//
// The engine models this PER MATERIAL (PrincipledBSDF::selected_uv_set) while
// glTF models it PER TEXTURE SLOT. Assimp resolves that the same way — first
// textured slot wins — so this mirrors it, and LOGS when the slots disagree
// rather than silently picking one.
// ---------------------------------------------------------------------------
int preferredUvSet(const cgltf_material* mat, bool* disagreed) {
    if (disagreed) *disagreed = false;
    if (!mat) return 0;

    const cgltf_texture_view* slots[] = {
        mat->has_pbr_metallic_roughness ? &mat->pbr_metallic_roughness.base_color_texture : nullptr,
        mat->has_pbr_metallic_roughness ? &mat->pbr_metallic_roughness.metallic_roughness_texture : nullptr,
        &mat->normal_texture,
        &mat->emissive_texture,
    };

    int chosen = -1;
    for (const cgltf_texture_view* v : slots) {
        if (!v || !v->texture) continue;
        const int set = v->texcoord > 0 ? static_cast<int>(v->texcoord) : 0;
        if (chosen < 0) chosen = set;
        else if (set != chosen && disagreed) *disagreed = true;
    }
    return chosen < 0 ? 0 : chosen;
}

// ---------------------------------------------------------------------------
// ★ glTF node matrices are COLUMN-major (OpenGL convention); Matrix4x4 here is
// row-major m[row][col]. Getting this backwards yields a transposed pose that
// still looks like a pose — it animates, it is just wrong, which is the failure
// class this repo keeps paying for.
// ---------------------------------------------------------------------------
Matrix4x4 mat4FromGl(const cgltf_float gl[16]) {
    Matrix4x4 m;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            m.m[r][c] = static_cast<float>(gl[c * 4 + r]);
    return m;
}

Matrix4x4 nodeLocalMatrix(const cgltf_node* node) {
    cgltf_float gl[16];
    cgltf_node_transform_local(node, gl);
    return mat4FromGl(gl);
}

Matrix4x4 nodeWorldMatrix(const cgltf_node* node) {
    cgltf_float gl[16];
    cgltf_node_transform_world(node, gl);
    return mat4FromGl(gl);
}

std::string lowerExtension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// ---------------------------------------------------------------------------
// Reader state. One instance per import; nothing outlives readGltf().
// ---------------------------------------------------------------------------
class GltfReader {
public:
    GltfReader(const std::string& path, const ImportOptions& opts, ImportedModel& out)
        : path_(path), opts_(opts), out_(out) {}

    ~GltfReader() {
        // Also release successfully parsed data if a later import stage throws.
        if (data_) cgltf_free(data_);
    }

    bool run(std::string& error);

private:
    std::string prefixed(const std::string& name) const {
        if (opts_.importPrefix.empty()) return name;
        if (name.rfind(opts_.importPrefix + "_", 0) == 0) return name;
        return opts_.importPrefix + "_" + name;
    }

    // Every glTF entity may be unnamed; a stable synthetic name matters because
    // animation channels, bones and the transform store are all keyed by name.
    std::string nodeName(const cgltf_node* node) const {
        auto resolved = meshNodeNames_.find(node);
        if (resolved != meshNodeNames_.end()) return resolved->second;
        if (node && node->name && node->name[0]) return prefixed(node->name);
        const size_t index = node ? static_cast<size_t>(node - data_->nodes) : 0;
        return prefixed("Node_" + std::to_string(index));
    }

    void buildHierarchy();
    void buildMaterials();
    void buildMeshes();
    // Every TriangleMesh emitted for a given glTF mesh, so an EXT_mesh_gpu_instancing
    // node that points at the SAME mesh as a scene node reuses it instead of
    // building a second copy of the vertex data (and a second BLAS).
    std::unordered_map<const cgltf_mesh*, std::vector<std::shared_ptr<TriangleMesh>>> meshesByGltfMesh_;
    bool readInstancePlacements(const cgltf_node* node,
                                std::vector<ImportedInstancePlacement>& out) const;
    bool primitiveIsSkinned(const cgltf_node* node, const cgltf_primitive& prim) const;
    std::string resolveUniqueMeshNodeName(const std::string& baseName);
    std::string meshNodeName(const cgltf_node* node);
    bool usesLegacyRayTrophiSkinSpace(const cgltf_node* node, const Matrix4x4& world) const;
    void buildSkins();
    void buildAnimations();
    void buildCamerasAndLights();

    std::shared_ptr<Texture> decodeImage(const cgltf_image* image, TextureType type);
    void prefetchTextures();
    std::shared_ptr<Texture> textureFor(const cgltf_texture* tex, TextureType type);
    uint16_t materialIdFor(const cgltf_material* mat);
    void emitPrimitive(const cgltf_node* node,
                       const cgltf_primitive& prim,
                       const std::string& name,
                       const std::shared_ptr<Transform>& transform);

    std::string path_;
    const ImportOptions& opts_;
    ImportedModel& out_;
    cgltf_data* data_ = nullptr;
    std::string baseDir_;

    std::unordered_map<const cgltf_material*, uint16_t> materialIds_;
    // Keyed by (image, type): the SAME image used as albedo and as emissive needs
    // two Texture objects because is_srgb / is_aces differ. ★ Do NOT fold the two
    // into one scalar by offsetting the pointer — that can collide with a
    // neighbouring image and silently return a texture decoded in the wrong
    // colour space.
    struct ImageKey {
        const cgltf_image* image;
        TextureType type;
        bool operator==(const ImageKey& o) const { return image == o.image && type == o.type; }
    };
    struct ImageKeyHash {
        size_t operator()(const ImageKey& k) const {
            return std::hash<const void*>()(k.image) ^
                   (std::hash<int>()(static_cast<int>(k.type)) << 1);
        }
    };
    std::unordered_map<ImageKey, std::shared_ptr<Texture>, ImageKeyHash> textureCache_;

    // One resolved identity per source node, shared by geometry, joints and channels.
    std::unordered_map<const cgltf_node*, std::string> meshNodeNames_;
    std::unordered_map<std::string, int> nodeNameUsage_;
};

// ---------------------------------------------------------------------------
// Hierarchy
// ---------------------------------------------------------------------------
// Mirrors AssimpLoader::resolveUniqueNodeName, including the zero-padded
// ".001" suffix, so the same file yields the same object names through either
// reader — which is what makes the parity probe a real comparison.
std::string GltfReader::resolveUniqueMeshNodeName(const std::string& baseName) {
    int& count = nodeNameUsage_[baseName];
    if (count == 0) { ++count; return baseName; }
    std::string candidate;
    do {
        std::string suffix = std::to_string(count++);
        while (suffix.size() < 3) suffix = "0" + suffix;
        candidate = baseName + "." + suffix;
    } while (nodeNameUsage_.count(candidate));
    // Reserve the generated name too, so a node genuinely AUTHORED as "X.001"
    // cannot end up sharing it with a generated one. Assimp checks its transform
    // map here instead and can still collide; producing two objects under one
    // name is worse than diverging in this corner.
    ++nodeNameUsage_[candidate];
    return candidate;
}

// Resolved during the hierarchy walk so the traversal ORDER decides which node
// keeps the bare name — the same rule Assimp's depth-first walk applies. Nodes
// outside the active scene are resolved on demand here.
std::string GltfReader::meshNodeName(const cgltf_node* node) {
    auto it = meshNodeNames_.find(node);
    if (it != meshNodeNames_.end()) return it->second;
    std::string resolved = resolveUniqueMeshNodeName(nodeName(node));
    meshNodeNames_[node] = resolved;
    return resolved;
}

void GltfReader::buildHierarchy() {
    out_.hierarchy.nodes.clear();

    // ★ glTF scenes may have MANY roots; NodeHierarchy walks from node 0. A
    // synthetic identity root keeps a multi-root file from losing every root but
    // the first — silently, as a model missing most of its parts.
    const cgltf_scene* scene = data_->scene ? data_->scene
                             : (data_->scenes_count ? &data_->scenes[0] : nullptr);
    std::vector<cgltf_node*> roots;
    if (scene) {
        for (cgltf_size i = 0; i < scene->nodes_count; ++i) roots.push_back(scene->nodes[i]);
    } else {
        for (cgltf_size i = 0; i < data_->nodes_count; ++i)
            if (!data_->nodes[i].parent) roots.push_back(&data_->nodes[i]);
    }
    if (roots.empty()) return;

    const int syntheticRoot = out_.hierarchy.addNode(
        "RootNode", resolveUniqueMeshNodeName(prefixed("__gltf_scene_root")), Matrix4x4::identity(), -1);
    std::unordered_set<const cgltf_node*> visited;

    std::function<void(cgltf_node*, int)> recurse = [&](cgltf_node* node, int parent) {
        if (!node || !visited.insert(node).second) return;
        // Consume a name slot for EVERY node, mesh-bearing or not, exactly as
        // AssimpLoader does (it calls resolveUniqueNodeName once per node in its
        // own walk). Skipping the empties here would shift which duplicate keeps
        // the bare name and silently rename objects between the two readers.
        meshNodeNames_[node] = resolveUniqueMeshNodeName(nodeName(node));
        const int self = out_.hierarchy.addNode(
            node->name && node->name[0] ? node->name
                                        : ("Node_" + std::to_string(node - data_->nodes)),
            nodeName(node), nodeLocalMatrix(node), parent);
        for (cgltf_size i = 0; i < node->children_count; ++i)
            recurse(node->children[i], self);
    };
    for (cgltf_node* r : roots) recurse(r, syntheticRoot);
    // Keep complete ancestor chains for joints/channels outside the active scene.
    for (cgltf_size i = 0; i < data_->nodes_count; ++i) {
        cgltf_node* root = &data_->nodes[i];
        if (!visited.count(root)) {
            while (root->parent) root = root->parent;
            recurse(root, syntheticRoot);
        }
    }

    out_.stats.node_count = out_.hierarchy.size();
}

// ---------------------------------------------------------------------------
// Textures
//
// glTF images live either in a bufferView (GLB / embedded) or behind a URI.
// cgltf_load_buffers has already resolved base64 data: URIs into buffer memory,
// so only external files still need disk I/O.
// ---------------------------------------------------------------------------
// Pure decode: no cache, no stats, no member mutation — so it is safe to run on
// a worker thread. Everything it touches (data_, baseDir_, the import prefix) is
// read-only for the duration of the import.
std::shared_ptr<Texture> GltfReader::decodeImage(const cgltf_image* image, TextureType type) {
    if (!image) return nullptr;

    if (image->buffer_view && image->buffer_view->buffer && image->buffer_view->buffer->data) {
        const auto* base = static_cast<const unsigned char*>(image->buffer_view->buffer->data);
        const unsigned char* start = base + image->buffer_view->offset;
        std::vector<char> bytes(reinterpret_cast<const char*>(start),
                                reinterpret_cast<const char*>(start) + image->buffer_view->size);
        const std::string name = prefixed(
            image->name && image->name[0] ? image->name
                                          : ("image_" + std::to_string(image - data_->images)));
        return std::make_shared<Texture>(bytes, type, name);
    }

    if (image->uri && image->uri[0]) {
        // cgltf already turned data: URIs into buffer_view memory above, so a
        // surviving URI is a real file next to the .gltf.
        std::filesystem::path p(image->uri);
        if (p.is_relative()) p = std::filesystem::path(baseDir_) / p;
        if (std::filesystem::exists(p)) return std::make_shared<Texture>(p.string(), type);
        SCENE_LOG_WARN("[glTF] texture file not found: " + p.string());
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// ★★★ PARALLEL TEXTURE DECODE — THE IMPORT'S DOMINANT COST.
//
// buildMaterials() walks materials one at a time and each texture slot decodes a
// PNG on the calling thread. On a vegetation asset that is ~20 atlases, and
// measured on this repo it was the bulk of a 2.4 s import: everything else in
// the reader is memory shuffling, stb_image is actual work.
//
// ★ This is not a new idea here — AssimpLoader has had exactly this pass
// (prefetchTextures) for a long time. Replacing Assimp with the direct reader
// therefore SILENTLY DROPPED an optimisation the old path had: same file, same
// pixels, one core instead of N. Nothing failed, the import just got slower, and
// a lost optimisation reports as nothing at all.
//
// Shape mirrors the Assimp one on purpose:
//   - bounded workers + an atomic job index, NOT std::async per job (async can
//     spawn a thread per call and oversubscribe on big scenes),
//   - decode in parallel, then do the GPU upload and the cache insert SERIALLY
//     on the calling thread, because those touch shared state.
// ---------------------------------------------------------------------------
void GltfReader::prefetchTextures() {
    if (!opts_.loadMaterials || !data_ || data_->materials_count == 0) return;

    // The four slots buildMaterials() actually reads. A slot missed here is not
    // a bug, it just decodes serially later; a slot listed with the WRONG type
    // would decode the same image twice, so this list mirrors materialIdFor().
    std::vector<ImageKey> jobs;
    std::unordered_set<ImageKey, ImageKeyHash> seen;
    auto want = [&](const cgltf_texture* tex, TextureType type) {
        if (!tex || !tex->image) return;
        const ImageKey key{ tex->image, type };
        if (textureCache_.count(key)) return;
        if (!seen.insert(key).second) return;
        jobs.push_back(key);
    };

    for (cgltf_size m = 0; m < data_->materials_count; ++m) {
        const cgltf_material& mat = data_->materials[m];
        if (mat.has_pbr_metallic_roughness) {
            want(mat.pbr_metallic_roughness.base_color_texture.texture, TextureType::Albedo);
            want(mat.pbr_metallic_roughness.metallic_roughness_texture.texture, TextureType::Roughness);
        }
        want(mat.normal_texture.texture, TextureType::Normal);
        want(mat.emissive_texture.texture, TextureType::Emission);
    }

    if (jobs.size() < 2) return;   // one image is not worth a thread

    const size_t jobCount = jobs.size();
    std::vector<std::shared_ptr<Texture>> results(jobCount);

    unsigned int hw = std::thread::hardware_concurrency();
    if (hw < 2u) hw = 2u;
    const size_t workerCount = (std::min)(static_cast<size_t>(hw), jobCount);
    std::atomic<size_t> nextJob{ 0 };

    const auto tPrefetch = Clock::now();
    auto worker = [&]() {
        for (;;) {
            const size_t i = nextJob.fetch_add(1, std::memory_order_relaxed);
            if (i >= jobCount) return;
            try {
                results[i] = decodeImage(jobs[i].image, jobs[i].type);
            } catch (...) {
                results[i].reset();
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(workerCount);
    for (size_t t = 0; t < workerCount; ++t) threads.emplace_back(worker);
    for (auto& t : threads) t.join();

    size_t decoded = 0;
    for (size_t i = 0; i < jobCount; ++i) {
        if (!results[i] || !results[i]->is_loaded()) continue;
        if (g_hasOptix && isCudaTextureUploadAllowed()) {
            // Upload failed: leave it OUT of the cache so the serial path in
            // textureFor() retries it rather than caching a half-live texture.
            if (!results[i]->upload_to_gpu()) continue;
        }
        textureCache_[jobs[i]] = results[i];
        ++out_.stats.image_count;
        ++decoded;
    }

    SCENE_LOG_INFO("[glTF] texture prefetch: " + std::to_string(decoded) + "/" +
                   std::to_string(jobCount) + " image(s) on " + std::to_string(workerCount) +
                   " worker(s) in " + std::to_string(secondsSince(tPrefetch)) + " s");
}

std::shared_ptr<Texture> GltfReader::textureFor(const cgltf_texture* tex, TextureType type) {
    if (!tex || !tex->image) return nullptr;

    const ImageKey key{ tex->image, type };
    auto it = textureCache_.find(key);
    if (it != textureCache_.end()) return it->second;

    // Miss: either prefetch skipped this slot, or its upload failed. Decode here.
    std::shared_ptr<Texture> result = decodeImage(tex->image, type);
    if (result) ++out_.stats.image_count;
    textureCache_[key] = result;
    return result;
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
uint16_t GltfReader::materialIdFor(const cgltf_material* mat) {
    if (!mat) return MaterialManager::INVALID_MATERIAL_ID;
    auto it = materialIds_.find(mat);
    if (it != materialIds_.end()) return it->second;

    const std::string name = prefixed(
        mat->name && mat->name[0] ? mat->name
                                  : ("Material_" + std::to_string(mat - data_->materials)));

    auto pbr = std::make_shared<PrincipledBSDF>();
    pbr->materialName = name;

    if (opts_.loadMaterials) {
        // Which TEXCOORD_n this material's textures address. emitPrimitive reads
        // the same helper so the geometry and the material cannot disagree.
        bool uvSetsDisagree = false;
        pbr->selected_uv_set = preferredUvSet(mat, &uvSetsDisagree);
        if (uvSetsDisagree) {
            SCENE_LOG_WARN("[glTF] material '" + name + "' addresses DIFFERENT UV sets per "
                "texture slot; the engine stores one set per material, so the first textured "
                "slot wins. Some maps on this material may sample the wrong UVs.");
        }

        if (mat->has_pbr_metallic_roughness) {
            const auto& mr = mat->pbr_metallic_roughness;
            pbr->albedoProperty.color = Vec3(mr.base_color_factor[0],
                                             mr.base_color_factor[1],
                                             mr.base_color_factor[2]);
            pbr->albedoProperty.intensity = 1.0f;
            // ★ Alpha lives in baseColorFactor[3] and in the base colour
            // texture's A channel — glTF has no separate opacity slot. The
            // writer composites it the same way; see GltfDirectWriter.cpp.
            pbr->opacityProperty.alpha = mr.base_color_factor[3];
            pbr->roughnessProperty.intensity = mr.roughness_factor;
            pbr->metallicProperty.intensity = mr.metallic_factor;

            if (auto t = textureFor(mr.base_color_texture.texture, TextureType::Albedo)) {
                pbr->albedoProperty.texture = t;
                // An alpha channel in base colour IS the opacity map here.
                if (t->has_alpha) pbr->opacityProperty.texture = t;
            }
            // glTF packs occlusion/roughness/metallic into ONE texture: R/G/B.
            // Both slots point at the same image; the shaders read the channel
            // they need. Splitting it into two decoded copies would double the
            // texture memory for no gain.
            if (auto t = textureFor(mr.metallic_roughness_texture.texture, TextureType::Roughness)) {
                pbr->roughnessProperty.texture = t;
                pbr->metallicProperty.texture = t;
            }
        }

        if (auto t = textureFor(mat->normal_texture.texture, TextureType::Normal)) {
            pbr->normalProperty.texture = t;
            pbr->normalProperty.intensity =
                mat->normal_texture.scale > 0.0f ? mat->normal_texture.scale : 1.0f;
        }

        pbr->emissionProperty.color = Vec3(mat->emissive_factor[0],
                                           mat->emissive_factor[1],
                                           mat->emissive_factor[2]);
        // KHR_materials_emissive_strength — the writer emits it for anything
        // brighter than 1.0, because emissiveFactor itself is clamped to [0,1].
        float emissiveStrength = 1.0f;
        if (mat->has_emissive_strength) emissiveStrength = mat->emissive_strength.emissive_strength;
        pbr->emissionProperty.intensity = emissiveStrength;
        if (auto t = textureFor(mat->emissive_texture.texture, TextureType::Emission)) {
            pbr->emissionProperty.texture = t;
        }

        // Transmission and IOR travel together through the one setter that owns
        // their relationship; writing the fields directly would bypass whatever
        // it keeps consistent.
        const float transmission = mat->has_transmission
            ? mat->transmission.transmission_factor : 0.0f;
        const float ior = mat->has_ior ? mat->ior.ior : 1.5f;
        if (transmission > 0.0f || mat->has_ior) pbr->setTransmission(transmission, ior);

        if (mat->has_clearcoat) {
            pbr->setClearcoat(mat->clearcoat.clearcoat_factor,
                              mat->clearcoat.clearcoat_roughness_factor);
        }
        // alphaMode MASK/BLEND both mean "alpha is meaningful". OPAQUE means an
        // alpha channel present in the image must be IGNORED, so honour it —
        // otherwise a stray alpha channel silently punches holes in a solid mesh.
        if (mat->alpha_mode == cgltf_alpha_mode_opaque) {
            pbr->opacityProperty.texture = nullptr;
            pbr->opacityProperty.alpha = 1.0f;
        }
    }

    // Single source of truth for the GPU side (PBRMaterialSnapshot.h), the same
    // helper every other path uses — hand-filling GpuMaterial here would be a
    // second definition of the same mapping.
    if (!pbr->gpuMaterial) pbr->gpuMaterial = std::make_shared<GpuMaterial>();
    applyPBRMaterialSnapshotToGpuMaterial(capturePBRMaterialSnapshot(*pbr), *pbr->gpuMaterial);

    const uint16_t id = MaterialManager::getInstance().getOrCreateMaterialID(name, pbr);
    materialIds_[mat] = id;
    ++out_.stats.material_count;
    return id;
}

void GltfReader::buildMaterials() {
    for (cgltf_size i = 0; i < data_->materials_count; ++i) materialIdFor(&data_->materials[i]);
}

// ---------------------------------------------------------------------------
// Geometry — straight into flat SoA, no per-face facade soup.
// ---------------------------------------------------------------------------
void GltfReader::emitPrimitive(const cgltf_node* node,
                               const cgltf_primitive& prim,
                               const std::string& name,
                               const std::shared_ptr<Transform>& transform) {
    if (prim.type != cgltf_primitive_type_triangles) return;   // points/lines are not geometry here

    const cgltf_accessor* posAcc = nullptr;
    const cgltf_accessor* nrmAcc = nullptr;
    const cgltf_accessor* uvAcc = nullptr;
    const cgltf_accessor* jointAcc = nullptr;
    const cgltf_accessor* weightAcc = nullptr;

    // ★ The PRIMARY "uv" attribute holds the set the material actually samples,
    // not TEXCOORD_0 by reflex. That is the engine's convention: applyUVSet(n)
    // copies set n into "uv" and selected_uv_set records which one it is — so
    // doing the selection here means an imported material is already consistent
    // with what the UI, the paint tools and the project file expect.
    const int wantedUvSet = preferredUvSet(prim.material, nullptr);
    const cgltf_accessor* uvSet0 = nullptr;

    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        switch (attr.type) {
            case cgltf_attribute_type_position: posAcc = attr.data; break;
            case cgltf_attribute_type_normal:   nrmAcc = attr.data; break;
            case cgltf_attribute_type_texcoord:
                if (attr.index == 0) uvSet0 = attr.data;
                if (attr.index == wantedUvSet) uvAcc = attr.data;
                break;
            case cgltf_attribute_type_joints:   if (attr.index == 0) jointAcc = attr.data; break;
            case cgltf_attribute_type_weights:  if (attr.index == 0) weightAcc = attr.data; break;
            default: break;
        }
    }
    // The material asked for a set this primitive does not carry. Fall back to
    // set 0 rather than shipping blank UVs, and say so — blank UVs render as a
    // single texel of the atlas, which reads as a flat-coloured object.
    if (!uvAcc && uvSet0) {
        if (wantedUvSet != 0) {
            SCENE_LOG_WARN("[glTF] '" + name + "': material wants TEXCOORD_" +
                std::to_string(wantedUvSet) + " but the primitive only has TEXCOORD_0; using set 0.");
        }
        uvAcc = uvSet0;
    }
    if (!posAcc || posAcc->count == 0) return;

    const size_t vCount = static_cast<size_t>(posAcc->count);

    auto triMesh = std::make_shared<TriangleMesh>();
    // ★★★ EVERY PRIMITIVE OF A NODE SHARES THE NODE'S NAME. NOT name+"_prim"N.
    //
    // glTF splits a mesh into one primitive PER MATERIAL, so a tree arrives as
    // 9 primitives. They are NOT 9 objects: a multi-material import is ONE
    // logical object / hierarchy entry, and the whole engine is written that
    // way — this is a documented contract, not an accident:
    //   • AssimpLoader::processNodeToTriangles gives every mesh of a node the
    //     same uniqueNodeName, with the reason spelled out inline.
    //   • scene_ui_scatter.cpp::gatherScatterSource collects EVERY TriangleMesh
    //     whose nodeName matches, because building a source from one sibling was
    //     already reported once as "object bütünlüğü yok".
    //   • scene_ui_materials.cpp says outright that a multi-material import
    //     produces several TriangleMesh objects under one name.
    // Suffixing the name made each material its OWN object: scatter picked up a
    // fraction of the tree, selection and the project's node bookkeeping split,
    // and nothing errored — the geometry was all there, just filed under N names.
    //
    // Uniqueness is still enforced where it belongs: BETWEEN NODES, by
    // resolveUniqueMeshNodeName. Anything that needs per-mesh identity keys by
    // POINTER, not by name (VulkanBackend's "[DirectMesh]-<name>-<ptr>" BLAS key,
    // the raster meshKey) — which is exactly why the shared name is safe.
    triMesh->nodeName = name;
    triMesh->transform = transform;

    triMesh->geometry->resize_vertices(vCount);
    triMesh->geometry->add_attribute<Vec3>("P");
    triMesh->geometry->add_attribute<Vec3>("N");
    triMesh->geometry->add_attribute<Vec3>("P_orig");
    triMesh->geometry->add_attribute<Vec3>("N_orig");
    triMesh->geometry->add_attribute<Vec2>("uv");
    triMesh->geometry->add_attribute<uint16_t>("materialID");

    Vec3* positions = triMesh->geometry->get_attribute_data_mut<Vec3>("P");
    Vec3* normals = triMesh->geometry->get_attribute_data_mut<Vec3>("N");
    Vec3* origPositions = triMesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
    Vec3* origNormals = triMesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
    Vec2* uvs = triMesh->geometry->get_attribute_data_mut<Vec2>("uv");
    uint16_t* matIDs = triMesh->geometry->get_attribute_data_mut<uint16_t>("materialID");
    if (!positions || !normals || !uvs || !matIDs) return;

    // Bulk unpack: cgltf handles stride, normalisation and sparse accessors, so
    // the awkward encodings do not each need a hand-written branch here.
    //
    // ★ UNPACKED STRAIGHT INTO THE SoA ARRAYS — no temp vector, no copy pass.
    // Vec3 is {float x,y,z} and Vec2 is a union of two 2-float structs, i.e. both
    // are tightly packed (the codebase already relies on this: it uploads
    // `n * sizeof(Vec3)` raw to the GPU). cgltf's unpack has a memcpy fast path
    // for float32 accessors with the natural stride, which is what every real
    // exporter writes — so this is a straight blit into the destination instead
    // of allocate + blit + element-wise rebuild. On a 356k-vertex tree the old
    // shape allocated and copied ~12 MB per mesh for nothing.
    static_assert(sizeof(Vec3) == 3 * sizeof(float), "Vec3 must be tightly packed to unpack in place");
    static_assert(sizeof(Vec2) == 2 * sizeof(float), "Vec2 must be tightly packed to unpack in place");

    cgltf_accessor_unpack_floats(posAcc, &positions[0].x, vCount * 3);

    if (nrmAcc && nrmAcc->count == posAcc->count) {
        cgltf_accessor_unpack_floats(nrmAcc, &normals[0].x, vCount * 3);
        // Renormalise in place: glTF only requires unit normals in principle, and
        // a non-uniform node scale baked by an exporter breaks that in practice.
        for (size_t v = 0; v < vCount; ++v) {
            const float len = normals[v].length();
            normals[v] = len > 1e-6f ? normals[v] / len : Vec3(0.0f);
        }
    } else {
        for (size_t v = 0; v < vCount; ++v) normals[v] = Vec3(0.0f);
    }
    if (uvAcc && uvAcc->count == posAcc->count) {
        cgltf_accessor_unpack_floats(uvAcc, &uvs[0].x, vCount * 2);
        // ★ V flip applies to EVERY UV set, not just this one; see buildExtraUvSets.
        // ★★★ V IS FLIPPED, AND THAT IS NOT A PREFERENCE.
        // glTF 2.0: "The origin of the UV coordinates (0, 0) corresponds to the
        // UPPER LEFT corner of a texture image", i.e. v grows downward. This
        // engine is the other convention throughout — Texture::get_color_bilinear
        // samples row (1 - v) * (h - 1), and the GPU mirrors it
        // (material_program.glsl: texture(..., vec2(t.x, 1.0 - t.y))). So v = 0 is
        // the BOTTOM of the image here, the OpenGL/FBX convention.
        //
        // ★ Why this survived so long: the WRITER made the same omission, so a
        // glTF -> RayTrophi -> glTF round trip came back correct and could not
        // detect it. Only content crossing the boundary once shows it: an FBX
        // imports right (FBX is bottom-left too) while a glTF painting arrives
        // upside down. GltfDirectWriter now flips on the way out as well; the two
        // MUST move together or the round trip breaks.
        for (size_t v = 0; v < vCount; ++v) uvs[v].y = 1.0f - uvs[v].y;
    } else {
        for (size_t v = 0; v < vCount; ++v) uvs[v] = Vec2(0.0f, 0.0f);
    }

    const uint16_t materialId = prim.material ? materialIdFor(prim.material) : (uint16_t)0;
    for (size_t v = 0; v < vCount; ++v) matIDs[v] = materialId;

    // ★★ EXTRA UV SETS (TEXCOORD_1..n).
    // The Assimp path copies every UV channel into "uv1", "uv2", ... and the
    // material panel's UV-set switcher reads them. Reading only TEXCOORD_0 here
    // would have made "remove Assimp from glTF" a SILENT DOWNGRADE for any
    // lightmapped or multi-UV asset: the sets do not vanish loudly, the panel
    // just stops offering them.
    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.type != cgltf_attribute_type_texcoord || attr.index == 0) continue;
        if (!attr.data || attr.data->count != posAcc->count) continue;

        const std::string attrName = "uv" + std::to_string(attr.index);
        triMesh->geometry->add_attribute<Vec2>(attrName);
        Vec2* dst = triMesh->geometry->get_attribute_data_mut<Vec2>(attrName);
        if (!dst) continue;
        cgltf_accessor_unpack_floats(attr.data, &dst[0].x, vCount * 2);
        for (size_t v = 0; v < vCount; ++v) dst[v].y = 1.0f - dst[v].y;   // same flip as set 0
    }

    // Indices. glTF is already indexed; a non-indexed primitive is a sequential
    // triple list, which is exactly what GeometryDetail wants anyway.
    auto& indices = triMesh->geometry->indices;
    if (prim.indices && prim.indices->count >= 3) {
        const size_t indexCount = (prim.indices->count / 3) * 3;
        indices.resize(indexCount);
        // ★ BULK, not one call per index. cgltf_accessor_read_index() is a
        // per-element call that re-derives the component type every time; on a
        // 356k-triangle tree that is over a million of them for what is, in the
        // overwhelmingly common case, a memcpy (uint32 source) or one tight
        // widening loop (uint16 source — what most exporters write).
        //
        // ★ It returns 0 for the cases it cannot do in bulk (sparse accessor, no
        // buffer view). That is not an error, it is the signal to take the
        // general path — so the fallback below stays, rather than being deleted
        // as "unreachable".
        const cgltf_size unpacked = cgltf_accessor_unpack_indices(
            prim.indices, indices.data(), sizeof(uint32_t), indexCount);
        if (unpacked < indexCount) {
            for (size_t i = 0; i < indexCount; ++i)
                indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, i));
        }
    } else {
        const size_t indexCount = (vCount / 3) * 3;
        indices.resize(indexCount);
        for (size_t i = 0; i < indexCount; ++i) indices[i] = static_cast<uint32_t>(i);
    }
    if (indices.empty()) return;

    // Missing normals: derive them from the faces rather than shipping zeros,
    // which shade black and read as a material bug.
    if (!nrmAcc) {
        std::vector<Vec3> accum(vCount, Vec3(0.0f));
        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
            if (a >= vCount || b >= vCount || c >= vCount) continue;
            Vec3 fn = Vec3::cross(positions[b] - positions[a], positions[c] - positions[a]);
            const float len = fn.length();
            if (len <= 1e-8f) continue;
            fn = fn / len;
            accum[a] += fn; accum[b] += fn; accum[c] += fn;
        }
        for (size_t v = 0; v < vCount; ++v) {
            const float len = accum[v].length();
            normals[v] = len > 1e-8f ? accum[v] / len : Vec3(0.0f, 1.0f, 0.0f);
        }
    }

    if (origPositions) std::memcpy(origPositions, positions, vCount * sizeof(Vec3));
    if (origNormals)   std::memcpy(origNormals, normals, vCount * sizeof(Vec3));

    // ── Skinning ────────────────────────────────────────────────────────────
    // ★ glTF hands over JOINTS_0/WEIGHTS_0 as fixed 4-wide arrays, while
    // GeometryDetail::skin_weights is a vector PER VERTEX. Reserving exactly 4
    // and dropping zero weights keeps this from becoming one small heap
    // allocation per vertex with room to spare — the same cost class the export
    // side spent this whole migration escaping, just on the way in.
    const bool skinned = primitiveIsSkinned(node, prim);
    if (skinned) {
        std::vector<std::vector<std::pair<int, float>>> weights(vCount);
        cgltf_float w[4];
        cgltf_uint j[4];
        for (size_t v = 0; v < vCount; ++v) {
            if (!cgltf_accessor_read_float(weightAcc, v, w, 4)) continue;
            if (!cgltf_accessor_read_uint(jointAcc, v, j, 4)) continue;
            auto& dst = weights[v];
            dst.reserve(4);
            for (int k = 0; k < 4; ++k) {
                if (w[k] <= 0.0f) continue;
                const cgltf_node* jointNode =
                    (j[k] < node->skin->joints_count) ? node->skin->joints[j[k]] : nullptr;
                if (!jointNode) continue;
                auto boneIt = out_.bones->boneNameToIndex.find(nodeName(jointNode));
                if (boneIt == out_.bones->boneNameToIndex.end()) continue;
                dst.emplace_back(static_cast<int>(boneIt->second), static_cast<float>(w[k]));
            }
            // Strongest influence first — the skinning paths assume this order.
            std::sort(dst.begin(), dst.end(),
                      [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                          return a.second > b.second;
                      });
        }
        triMesh->geometry->skin_weights = std::move(weights);
        ++out_.stats.skinned_mesh_count;
    }

    // ── Facade representative(s) ────────────────────────────────────────────
    // Matches AssimpLoader exactly: one representative for a flat mesh, the full
    // per-face set for a skinned one (Renderer's import-flat collapse excludes
    // skinned meshes — "SoA skinning is a later increment"). Diverging here
    // would change which meshes go flat as a side effect of swapping readers.
    const size_t faceCount = indices.size() / 3;
    if (opts_.emitSingleFacadePerMesh && !skinned) {
        out_.objects.push_back(std::make_shared<Triangle>(triMesh, 0u));
    } else {
        for (size_t f = 0; f < faceCount; ++f)
            out_.objects.push_back(std::make_shared<Triangle>(triMesh, static_cast<uint32_t>(f)));
    }

    if (node && node->mesh) meshesByGltfMesh_[node->mesh].push_back(triMesh);

    ++out_.stats.mesh_count;
    out_.stats.vertex_count += vCount;
    out_.stats.triangle_count += faceCount;
}

// Use the same predicate for transform selection and weight decoding.
bool GltfReader::primitiveIsSkinned(const cgltf_node* node, const cgltf_primitive& prim) const {
    if (!opts_.loadSkinning || !node || !node->skin ||
        prim.type != cgltf_primitive_type_triangles) return false;
    const cgltf_accessor* pos = nullptr;
    const cgltf_accessor* joints = nullptr;
    const cgltf_accessor* weights = nullptr;
    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
        const cgltf_attribute& attr = prim.attributes[a];
        if (attr.type == cgltf_attribute_type_position) pos = attr.data;
        else if (attr.type == cgltf_attribute_type_joints && attr.index == 0) joints = attr.data;
        else if (attr.type == cgltf_attribute_type_weights && attr.index == 0) weights = attr.data;
    }
    return pos && joints && weights && pos->count > 0 &&
           joints->count == pos->count && weights->count == pos->count;
}

// Older RayTrophi exports wrote engine-space offsets and kept the cancelling
// axis/unit correction on the mesh node. Standard glTF ignores that node for
// skinning. Recognize this legacy layout by provenance AND its bind equation,
// rather than applying a guessed axis rotation/scale to every imported rig.
bool GltfReader::usesLegacyRayTrophiSkinSpace(
    const cgltf_node* node, const Matrix4x4& world) const {
    if (!data_->asset.generator || std::strcmp(data_->asset.generator, "RayTrophi Studio") != 0 ||
        !node || !node->skin || node->skin->joints_count == 0 || world.isIdentity()) return false;

    for (cgltf_size j = 0; j < node->skin->joints_count; ++j) {
        const cgltf_node* joint = node->skin->joints[j];
        if (!joint) return false;
        const auto offset = out_.bones->boneOffsetMatrices.find(nodeName(joint));
        if (offset == out_.bones->boneOffsetMatrices.end()) return false;
        const Matrix4x4 bind = world * nodeWorldMatrix(joint) * offset->second;
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                const float value = bind.m[r][c];
                const float expected = r == c ? 1.0f : 0.0f;
                if (!std::isfinite(value) || std::abs(value - expected) > 2e-3f) return false;
            }
        }
    }
    return true;
}

// EXT_mesh_gpu_instancing: TRANSLATION / ROTATION / SCALE, one entry per
// instance. All three are optional and default to identity; cgltf has already
// verified they share a count.
bool GltfReader::readInstancePlacements(const cgltf_node* node,
                                        std::vector<ImportedInstancePlacement>& out) const {
    if (!node || !node->has_mesh_gpu_instancing) return false;
    const cgltf_mesh_gpu_instancing& gi = node->mesh_gpu_instancing;
    if (gi.attributes_count == 0) return false;

    const cgltf_accessor* tAcc = nullptr;
    const cgltf_accessor* rAcc = nullptr;
    const cgltf_accessor* sAcc = nullptr;
    for (cgltf_size a = 0; a < gi.attributes_count; ++a) {
        const char* n = gi.attributes[a].name;
        if (!n) continue;
        if (std::strcmp(n, "TRANSLATION") == 0)   tAcc = gi.attributes[a].data;
        else if (std::strcmp(n, "ROTATION") == 0) rAcc = gi.attributes[a].data;
        else if (std::strcmp(n, "SCALE") == 0)    sAcc = gi.attributes[a].data;
    }

    const cgltf_accessor* any = tAcc ? tAcc : (rAcc ? rAcc : sAcc);
    if (!any || any->count == 0) return false;
    const size_t count = static_cast<size_t>(any->count);

    std::vector<cgltf_float> t, r, s;
    if (tAcc && tAcc->count == count) {
        t.resize(count * 3);
        if (cgltf_accessor_unpack_floats(tAcc, t.data(), t.size()) == 0) t.clear();
    }
    if (rAcc && rAcc->count == count) {
        r.resize(count * 4);
        if (cgltf_accessor_unpack_floats(rAcc, r.data(), r.size()) == 0) r.clear();
    }
    if (sAcc && sAcc->count == count) {
        s.resize(count * 3);
        if (cgltf_accessor_unpack_floats(sAcc, s.data(), s.size()) == 0) s.clear();
    }

    // ★ The node's own transform is the instancing SPACE: placements are given in
    // it, so it multiplies each placement rather than being ignored. The writer
    // emits an untransformed node today, but a file from anywhere else may not.
    const Matrix4x4 nodeWorld = nodeWorldMatrix(node);
    const bool nodeIsIdentity = nodeWorld.isIdentity();

    out.clear();
    out.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const Vec3 pos = t.empty() ? Vec3(0.0f) : Vec3(t[i * 3 + 0], t[i * 3 + 1], t[i * 3 + 2]);
        // glTF quaternion order is (x, y, z, w); the engine's is (w, x, y, z).
        const Quaternion rot = r.empty()
            ? Quaternion(1.0f, 0.0f, 0.0f, 0.0f)
            : Quaternion(r[i * 4 + 3], r[i * 4 + 0], r[i * 4 + 1], r[i * 4 + 2]);
        const Vec3 scl = s.empty() ? Vec3(1.0f) : Vec3(s[i * 3 + 0], s[i * 3 + 1], s[i * 3 + 2]);

        Matrix4x4 local = Matrix4x4::translation(pos) * rot.toMatrix() * Matrix4x4::scaling(scl);
        if (!nodeIsIdentity) local = nodeWorld * local;

        // No decomposition here: the engine's instance Euler convention belongs
        // to InstanceTransform, and its inverse lives beside toMatrix().
        ImportedInstancePlacement p;
        p.transform = local;
        out.push_back(p);
    }
    return !out.empty();
}

void GltfReader::buildMeshes() {
    if (!opts_.loadGeometry) return;

    // ★ TWO PASSES, and the order is the point: an EXT_mesh_gpu_instancing node
    // may reference the same glTF mesh as an ordinary scene node, and the file
    // gives no guarantee about which comes first. Emitting the plain nodes first
    // means the instancing pass can REUSE that geometry instead of building a
    // second copy — which would also be a second BLAS of the same tree.
    for (int pass = 0; pass < 2; ++pass) {
    for (cgltf_size i = 0; i < data_->nodes_count; ++i) {
        cgltf_node* node = &data_->nodes[i];
        if (!node->mesh) continue;

        std::vector<ImportedInstancePlacement> placements;
        const bool instanced = readInstancePlacements(node, placements);
        if (instanced != (pass == 1)) continue;

        // Already emitted by a plain node in pass 0: reuse it, emit nothing.
        if (instanced) {
            auto existing = meshesByGltfMesh_.find(node->mesh);
            if (existing != meshesByGltfMesh_.end() && !existing->second.empty()) {
                ImportedInstanceGroup group;
                group.name = meshNodeName(node);
                group.sourceNodeName = existing->second.front()->nodeName;
                group.sourceMeshes = existing->second;
                group.placements = std::move(placements);
                out_.stats.instance_count += group.placements.size();
                ++out_.stats.instance_group_count;
                SCENE_LOG_INFO("[glTF] EXT_mesh_gpu_instancing: '" + group.name + "' -> " +
                               std::to_string(group.placements.size()) +
                               " placement(s), reusing the scene mesh '" +
                               group.sourceNodeName + "' as source");
                out_.instanceGroups.push_back(std::move(group));
                continue;
            }
        }

        const std::string name = meshNodeName(node);

        // Standard glTF skinning yields world coordinates, so its mesh base is
        // identity. Legacy RayTrophi engine-space exports are handled explicitly.
        bool skinnedMesh = false;
        for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p)
            skinnedMesh |= primitiveIsSkinned(node, node->mesh->primitives[p]);
        const Matrix4x4 nodeWorld = nodeWorldMatrix(node);

        auto transform = std::make_shared<Transform>();
        // setBase(), not `base =`: it also decomposes into position/rotation/scale,
        // which is what the inspector and gizmo read. Assigning the matrix directly
        // leaves those at 0/0/1, so the panel reports the wrong place for an object
        // that renders in the right one — and the first gizmo edit recomposes from
        // them and snaps the object to the origin.
        const bool legacySkinSpace = skinnedMesh && usesLegacyRayTrophiSkinSpace(node, nodeWorld);
        transform->setBase(skinnedMesh && !legacySkinSpace ? Matrix4x4::identity() : nodeWorld);

        if (legacySkinSpace) {
            SCENE_LOG_INFO("[glTF] legacy RayTrophi skin-space correction retained for '" + name +
                           "' (meshWorld * jointBindWorld * inverseBind = identity).");
        } else if (skinnedMesh && !nodeWorld.isIdentity()) {
            SCENE_LOG_INFO("[glTF] skinned mesh '" + name +
                           "' has a non-identity node transform; ignoring it per spec "
                           "(its joints already carry the ancestor chain).");
        }

        auto staticTransform = transform;
        if (skinnedMesh) {
            staticTransform = std::make_shared<Transform>();
            staticTransform->setBase(nodeWorld);
        }
        const size_t objectsBefore = out_.objects.size();

        for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
            const auto& prim = node->mesh->primitives[p];
            emitPrimitive(node, prim, name,
                          primitiveIsSkinned(node, prim) ? transform : staticTransform);
        }

        if (instanced) {
            // ★ The prototype STAYS in `objects`. Vulkan's flat scatter path takes
            // each source mesh's BLAS from the base world-object pass, so a mesh
            // that is not a world object has no BLAS and the whole group is
            // skipped — silently, which is exactly what "the forest did not come
            // back" looked like.
            ImportedInstanceGroup group;
            group.name = name;
            group.sourceNodeName = name;
            group.placements = std::move(placements);
            for (size_t k = objectsBefore; k < out_.objects.size(); ++k) {
                const auto& facade = out_.objects[k];
                if (facade && facade->parentMesh) group.sourceMeshes.push_back(facade->parentMesh);
            }

            out_.stats.instance_count += group.placements.size();
            ++out_.stats.instance_group_count;
            SCENE_LOG_INFO("[glTF] EXT_mesh_gpu_instancing: '" + name + "' -> " +
                           std::to_string(group.placements.size()) + " placement(s), " +
                           std::to_string(group.sourceMeshes.size()) + " source mesh(es)");
            out_.instanceGroups.push_back(std::move(group));
        }
    }
    }
}

// ---------------------------------------------------------------------------
// Skins → BoneData
// ---------------------------------------------------------------------------
void GltfReader::buildSkins() {
    if (!opts_.loadSkinning) return;
    BoneData& bones = *out_.bones;

    // ★★★★★ THE SKELETON IS THE JOINTS **PLUS EVERY ANCESTOR OF A JOINT**.
    //
    // AnimationController::getAnimatedGlobalTransform builds a joint's world
    // matrix by walking boneData.boneParents UPWARD, reading each ancestor's
    // local matrix out of boneDefaultTransforms. The walk stops at the first
    // name that has no entry. So registering only the skin's joints truncates
    // the chain at the armature/root node — and everything that node carried
    // (the Z-up→Y-up rotation, the unit scale) is silently dropped.
    //
    // ★ The symptom is NOT a broken pose: the joints still move correctly
    // relative to each other, so the animation looks right while the whole
    // character is giant and lying on the wrong axis. Nobody files that as
    // "the skeleton is missing a node".
    //
    // This mirrors AssimpLoader::buildBoneData, which builds exactly this
    // closure ("technicalBones") by walking mParent up to the scene root.
    // Porting the joints without the closure is what broke it.
    std::unordered_set<const cgltf_node*> technical;   // joints + all ancestors
    std::unordered_set<const cgltf_node*> indexed;     // needs a runtime bone slot
    std::unordered_map<const cgltf_node*, Matrix4x4> weightedOffsets;

    auto addChain = [&](const cgltf_node* n) {
        for (const cgltf_node* c = n; c; c = c->parent) technical.insert(c);
    };

    // 1. Weighted joints — these carry a real inverse bind matrix.
    for (cgltf_size s = 0; s < data_->skins_count; ++s) {
        const cgltf_skin& skin = data_->skins[s];

        std::vector<cgltf_float> ibm;
        if (skin.inverse_bind_matrices && skin.inverse_bind_matrices->count >= skin.joints_count) {
            ibm.resize(static_cast<size_t>(skin.inverse_bind_matrices->count) * 16);
            if (cgltf_accessor_unpack_floats(skin.inverse_bind_matrices, ibm.data(), ibm.size()) == 0) {
                ibm.clear();   // ★ unpack refused; do NOT ship the zero-filled buffer
                SCENE_LOG_WARN("[glTF] could not unpack inverseBindMatrices for skin " +
                               std::to_string(s) + "; bones fall back to identity offsets.");
            }
        }

        for (cgltf_size j = 0; j < skin.joints_count; ++j) {
            const cgltf_node* joint = skin.joints[j];
            if (!joint) continue;

            indexed.insert(joint);
            addChain(joint);
            bones.weightedBoneNames.insert(nodeName(joint));

            // ★ The inverse bind matrix IS the offset matrix. Falling back to
            // identity when a skin omits it is what the spec allows, but it is
            // also silent: an identity offset produces a plausible-looking pose
            // that is wrong, so say it happened.
            if (!ibm.empty()) {
                weightedOffsets[joint] = mat4FromGl(&ibm[j * 16]);
            } else {
                weightedOffsets[joint] = Matrix4x4::identity();
                SCENE_LOG_WARN("[glTF] skin has no usable inverseBindMatrices; bone '" +
                               nodeName(joint) + "' falls back to an identity offset.");
            }
        }
    }

    // 2. Animation targets that carry no weights still need a skeleton slot: an
    //    animation-only import (or a rig whose root is animated but unskinned)
    //    otherwise has no runtime representation at all. Assimp does the same,
    //    with an identity offset.
    for (cgltf_size a = 0; a < data_->animations_count; ++a) {
        const cgltf_animation& anim = data_->animations[a];
        for (cgltf_size c = 0; c < anim.channels_count; ++c) {
            const cgltf_node* target = anim.channels[c].target_node;
            if (!target) continue;
            indexed.insert(target);
            addChain(target);
        }
    }

    if (technical.empty()) {
        bones.globalInverseTransform = Matrix4x4::identity();
        if (!opts_.importPrefix.empty())
            bones.perModelInverses[opts_.importPrefix] = bones.globalInverseTransform;
        bones.rebuildReverseLookup();
        out_.stats.bone_count = 0;
        return;
    }

    // 3. Register the closure in HIERARCHY order, so bone indices are assigned
    //    the way Assimp's depth-first addSkeletonNode assigns them. The order is
    //    not cosmetic: OzzRuntime and the anim graph map their own joint arrays
    //    onto these indices by position.
    std::function<void(const cgltf_node*)> addSkeletonNode = [&](const cgltf_node* node) {
        if (!node) return;
        if (technical.count(node)) {
            const std::string name = nodeName(node);
            bones.boneDefaultTransforms[name] = nodeLocalMatrix(node);

            // Only link to a parent that is itself part of the skeleton, so the
            // chain terminates cleanly at the topmost ancestor instead of
            // pointing at a name the walk cannot resolve.
            if (node->parent && technical.count(node->parent))
                bones.boneParents[name] = nodeName(node->parent);

            if (indexed.count(node) && !bones.boneNameToIndex.count(name)) {
                bones.boneNameToIndex[name] =
                    static_cast<unsigned int>(bones.boneNameToIndex.size());
                auto off = weightedOffsets.find(node);
                bones.boneOffsetMatrices[name] =
                    (off != weightedOffsets.end()) ? off->second : Matrix4x4::identity();
            }
        }
        for (cgltf_size k = 0; k < node->children_count; ++k)
            addSkeletonNode(node->children[k]);
    };

    const cgltf_scene* activeScene = data_->scene ? data_->scene
                                   : (data_->scenes_count ? &data_->scenes[0] : nullptr);
    if (activeScene) {
        for (cgltf_size i = 0; i < activeScene->nodes_count; ++i) addSkeletonNode(activeScene->nodes[i]);
    } else {
        for (cgltf_size i = 0; i < data_->nodes_count; ++i)
            if (!data_->nodes[i].parent) addSkeletonNode(&data_->nodes[i]);
    }

    // 4. Safety net: a joint or channel target outside the active scene is never
    //    reached by the walk above. Assimp carries the same net, and its absence
    //    would show up as one limb frozen at the origin.
    for (cgltf_size i = 0; i < data_->nodes_count; ++i) {
        const cgltf_node* node = &data_->nodes[i];
        if (!technical.count(node)) continue;
        const std::string name = nodeName(node);
        if (!bones.boneDefaultTransforms.count(name))
            bones.boneDefaultTransforms[name] = nodeLocalMatrix(node);
        if (node->parent && technical.count(node->parent) && !bones.boneParents.count(name))
            bones.boneParents[name] = nodeName(node->parent);
        if (indexed.count(node) && !bones.boneNameToIndex.count(name)) {
            bones.boneNameToIndex[name] =
                static_cast<unsigned int>(bones.boneNameToIndex.size());
            auto off = weightedOffsets.find(node);
            bones.boneOffsetMatrices[name] =
                (off != weightedOffsets.end()) ? off->second : Matrix4x4::identity();
        }
    }


    // Joint matrices retain their source hierarchy space. Standard glTF meshes
    // use an identity base; recognized legacy RayTrophi files retain their
    // cancelling mesh correction. Neither route needs an extra global inverse.
    bones.globalInverseTransform = Matrix4x4::identity();
    if (!opts_.importPrefix.empty())
        bones.perModelInverses[opts_.importPrefix] = bones.globalInverseTransform;
    bones.rebuildReverseLookup();
    out_.stats.bone_count = bones.boneNameToIndex.size();
}

// ---------------------------------------------------------------------------
// Animations → AnimationData with RayTrophi key types (Faz 0 made this clean:
// there is no Assimp key type left to convert through).
// ---------------------------------------------------------------------------
void GltfReader::buildAnimations() {
    if (!opts_.loadAnimations) return;

    for (cgltf_size a = 0; a < data_->animations_count; ++a) {
        const cgltf_animation& anim = data_->animations[a];
        auto out = std::make_shared<AnimationData>();
        out->name = prefixed(anim.name && anim.name[0] ? anim.name
                                                       : ("Animation_" + std::to_string(a)));
        out->modelName = opts_.importPrefix;
        // ★ glTF sampler inputs are SECONDS. AnimationData is expressed in TICKS
        // with an explicit ticksPerSecond, so declaring 1 tick == 1 second keeps
        // the stored numbers untouched and the playback rate correct. Inventing
        // a 24 or 30 here would silently retime every imported clip.
        out->ticksPerSecond = 1.0;
        double maxTime = 0.0;

        for (cgltf_size c = 0; c < anim.channels_count; ++c) {
            const cgltf_animation_channel& ch = anim.channels[c];
            if (!ch.target_node || !ch.sampler) continue;
            const cgltf_animation_sampler& smp = *ch.sampler;
            if (!smp.input || !smp.output) continue;

            const size_t keyCount = static_cast<size_t>(smp.input->count);
            if (keyCount == 0) continue;

            std::vector<cgltf_float> times(keyCount);
            cgltf_accessor_unpack_floats(smp.input, times.data(), times.size());

            const size_t comps = cgltf_num_components(smp.output->type);
            // CUBICSPLINE stores in-tangent/value/out-tangent per key; taking
            // every third element samples the VALUES. Treating it as linear is a
            // known, stated approximation — not a silent one.
            const size_t stride = (smp.interpolation == cgltf_interpolation_type_cubic_spline) ? 3 : 1;
            const size_t valueCount = static_cast<size_t>(smp.output->count);
            std::vector<cgltf_float> values(valueCount * comps);
            cgltf_accessor_unpack_floats(smp.output, values.data(), values.size());
            if (stride == 3 && smp.interpolation == cgltf_interpolation_type_cubic_spline) {
                SCENE_LOG_WARN("[glTF] CUBICSPLINE channel sampled as LINEAR in clip '" +
                               out->name + "'.");
            }

            const std::string target = nodeName(ch.target_node);
            for (size_t k = 0; k < keyCount; ++k) {
                const double t = static_cast<double>(times[k]);
                maxTime = (std::max)(maxTime, t);
                const size_t base = (stride == 3 ? (k * 3 + 1) : k) * comps;
                if (base + comps > values.size()) break;

                if (ch.target_path == cgltf_animation_path_type_translation && comps >= 3) {
                    out->positionKeys[target].emplace_back(
                        t, Vec3(values[base + 0], values[base + 1], values[base + 2]));
                } else if (ch.target_path == cgltf_animation_path_type_rotation && comps >= 4) {
                    // ★ glTF quaternions are (x, y, z, w); Quaternion is (w, x, y, z).
                    out->rotationKeys[target].emplace_back(
                        t, Quaternion(values[base + 3], values[base + 0],
                                      values[base + 1], values[base + 2]));
                } else if (ch.target_path == cgltf_animation_path_type_scale && comps >= 3) {
                    out->scalingKeys[target].emplace_back(
                        t, Vec3(values[base + 0], values[base + 1], values[base + 2]));
                }
            }
            ++out_.stats.animation_channel_count;
        }

        out->duration = maxTime > 0.0 ? maxTime : 1.0;
        out->startFrame = 0;
        out->endFrame = static_cast<int>(out->duration * 24.0);
        if (!out->positionKeys.empty() || !out->rotationKeys.empty() || !out->scalingKeys.empty()) {
            out_.animations.push_back(std::move(out));
            ++out_.stats.animation_count;
        }
    }
}

// ---------------------------------------------------------------------------
// Cameras and KHR_lights_punctual
// ---------------------------------------------------------------------------
void GltfReader::buildCamerasAndLights() {
    for (cgltf_size i = 0; i < data_->nodes_count; ++i) {
        cgltf_node* node = &data_->nodes[i];
        const Matrix4x4 world = nodeWorldMatrix(node);
        const Vec3 origin(world.m[0][3], world.m[1][3], world.m[2][3]);
        // glTF convention: a camera/light looks down its own -Z.
        const Vec3 forward = Vec3(-world.m[0][2], -world.m[1][2], -world.m[2][2]).normalize();

        if (opts_.loadCameras && node->camera) {
            const cgltf_camera& cam = *node->camera;
            float vfov = 45.0f;
            float aspect = 16.0f / 9.0f;
            if (cam.type == cgltf_camera_type_perspective) {
                vfov = static_cast<float>(cam.data.perspective.yfov * 180.0 / 3.14159265358979);
                if (cam.data.perspective.has_aspect_ratio && cam.data.perspective.aspect_ratio > 0.0f)
                    aspect = cam.data.perspective.aspect_ratio;
            }
            auto camera = std::make_shared<Camera>(origin, origin + forward, Vec3(0, 1, 0),
                                                   vfov, aspect, 0.0f, 1.0f, 6);
            camera->nodeName = nodeName(node);
            out_.cameras.push_back(std::move(camera));
            ++out_.stats.camera_count;
        }

        if (opts_.loadLights && node->light) {
            const cgltf_light& l = *node->light;

            // ★★★ KHR_lights_punctual IS PHOTOMETRIC; THIS ENGINE IS NOT.
            //   point / spot : intensity is CANDELA  (lm/sr)
            //   directional  : intensity is LUX      (lm/m²)
            // The engine's Light::intensity is a radiometric-ish quantity in the
            // SAME geometric form: PointLight::getIntensity() returns
            // (color * intensity) / distance² — note there is NO 4π, so it is
            // already a per-steradian value, exactly like candela. Directional
            // lights are irradiance, exactly like lux.
            //
            // So the whole conversion is one constant: the luminous efficacy of
            // 555 nm light, 683 lm/W — the same number Khronos and Blender's
            // glTF I/O use. W/sr = cd / 683, W/m² = lux / 683.
            //
            // Sanity check on a Blender default: a 100 W point lamp exports as
            // 100 * 683 / (4π) = 5435 cd, and 5435 / 683 = 7.96 = 100 / (4π) W/sr,
            // i.e. it round-trips back to its authored wattage. The engine's own
            // default point light is ~17, so an imported Blender lamp now lands in
            // the same range instead of being ~5400.
            //
            // ★ This REPLACES the magnitude ladder the Assimp path used
            // (">1000 divide by 1000, >100 divide by 100, ..."). That ladder had no
            // unit behind it: it was fitted to make Blender defaults look right,
            // so any lamp near a threshold changed brightness by 10x for no
            // physical reason, and nothing in the file said which branch it took.
            constexpr float kLumensPerWatt = 683.0f;

            // RayTrophi writes its OWN native intensity (GltfDirectWriter emits
            // light->intensity verbatim), so converting our own files would divide
            // them by 683 on every reopen. Mirrors AssimpLoader::isRayTrophiGltfFile.
            const char* gen = data_->asset.generator;
            const bool nativeUnits = gen && std::string(gen).find("RayTrophi") != std::string::npos;
            const float unitScale = nativeUnits ? 1.0f : (1.0f / kLumensPerWatt);

            Vec3 color(l.color[0], l.color[1], l.color[2]);
            const float maxComp = (std::max)(color.x, (std::max)(color.y, color.z));
            if (maxComp > 1e-6f) color = color / maxComp;   // keep hue, magnitude lives in intensity
            else                 color = Vec3(1.0f);
            const float scalar = (std::max)(0.0f, l.intensity * unitScale * (maxComp > 1e-6f ? maxComp : 1.0f));

            std::shared_ptr<Light> light;
            switch (l.type) {
                case cgltf_light_type_directional:
                    light = std::make_shared<DirectionalLight>(forward, color, 0.05f);
                    break;
                case cgltf_light_type_spot:
                    light = std::make_shared<SpotLight>(
                        origin, forward, color,
                        // ★ glTF's outerConeAngle is the HALF angle; SpotLight takes the
                        // FULL cone in degrees (GltfDirectWriter halves it on the way
                        // out). Without the x2 every imported spot was half as wide.
                        static_cast<float>(l.spot_outer_cone_angle * 2.0 * 180.0 / 3.14159265358979),
                        0.0f);
                    break;
                case cgltf_light_type_point:
                default:
                    light = std::make_shared<PointLight>(origin, color, 0.1f);
                    break;
            }
            if (light) {
                // ★ Set color and intensity EXPLICITLY instead of folding them into
                // one Vec3. The punctual constructors do intensity = vec.length(),
                // so handing them colour x intensity inflates a white light by
                // sqrt(3) = 1.73 and tints the stored intensity by hue. The writer
                // exports light->intensity verbatim, so that error also compounded
                // on every export/import round trip.
                light->color = color;
                light->intensity = scalar;
                light->initialDirection = forward;
                light->nodeName = nodeName(node);
                out_.lights.push_back(std::move(light));
                ++out_.stats.light_count;
            }
        }
    }
}

// ---------------------------------------------------------------------------
bool GltfReader::run(std::string& error) {
    const auto tTotal = Clock::now();
    out_.stats.reader = "cgltf";
    out_.importName = opts_.importPrefix;
    if (!out_.bones) out_.bones = std::make_shared<BoneData>();
    baseDir_ = std::filesystem::path(path_).parent_path().string();

    const auto tParse = Clock::now();
    cgltf_options options{};
    gltfJsonNumberInvalid = false;
    cgltf_result r = cgltf_parse_file(&options, path_.c_str(), &data_);
    if (gltfJsonNumberInvalid) {
        error = "Invalid or out-of-range glTF JSON number";
        return false;
    }
    if (r != cgltf_result_success) {
        error = "cgltf_parse_file failed (" + std::to_string(static_cast<int>(r)) + ")";
        return false;
    }
    r = cgltf_load_buffers(&options, data_, path_.c_str());
    if (r != cgltf_result_success) {
        error = "cgltf_load_buffers failed (" + std::to_string(static_cast<int>(r)) + ")";
        cgltf_free(data_); data_ = nullptr;
        return false;
    }
    const cgltf_result vr = cgltf_validate(data_);
    if (vr != cgltf_result_success) {
        std::string why;
        if (!validateReadSafety(data_, why)) {
            error = "cgltf_validate rejected the file (result " +
                    std::to_string(static_cast<int>(vr)) + ") and it is not safe to read: " + why;
            cgltf_free(data_); data_ = nullptr;
            return false;
        }
        cgltf_size totalPrimitives = 0;
        for (cgltf_size m = 0; m < data_->meshes_count; ++m)
            totalPrimitives += data_->meshes[m].primitives_count;
        const cgltf_size empty = countEmptyPrimitives(data_);
        SCENE_LOG_WARN("[glTF] file is not spec-conformant (cgltf_validate result " +
            std::to_string(static_cast<int>(vr)) + ") but is safe to read; continuing. Generator: " +
            std::string(data_->asset.generator ? data_->asset.generator : "unknown") +
            ". Empty primitives (no vertices, skipped): " + std::to_string(empty) +
            " of " + std::to_string(totalPrimitives) + ".");
    }
    out_.stats.seconds_parse = secondsSince(tParse);

    buildHierarchy();

    const auto tMat = Clock::now();
    prefetchTextures();   // ★ decode every image in parallel before materials ask for them
    buildMaterials();
    out_.stats.seconds_materials = secondsSince(tMat);

    // Skins BEFORE meshes: emitPrimitive resolves joint indices through
    // boneNameToIndex, so the bone table has to exist first. Getting this order
    // wrong drops every weight and the mesh renders in bind pose — silently.
    buildSkins();

    const auto tGeo = Clock::now();
    buildMeshes();
    out_.stats.seconds_geometry = secondsSince(tGeo);

    const auto tAnim = Clock::now();
    buildAnimations();
    out_.stats.seconds_animation = secondsSince(tAnim);

    buildCamerasAndLights();

    cgltf_free(data_);
    data_ = nullptr;
    out_.stats.seconds_total = secondsSince(tTotal);
    return true;
}

} // namespace

namespace {

// Transforms the eight corners of [mn,mx] and grows the accumulator. A rotated
// box's axis-aligned hull is not the transform of its min/max corners, so the
// corners have to be enumerated — taking a shortcut here yields a box that is
// too small, which shows up as a preview that clips the model.
void growByTransformedBox(const Matrix4x4& m, const float mn[3], const float mx[3],
                          float outMin[3], float outMax[3], bool& any) {
    for (int c = 0; c < 8; ++c) {
        const Vec3 corner((c & 1) ? mx[0] : mn[0],
                          (c & 2) ? mx[1] : mn[1],
                          (c & 4) ? mx[2] : mn[2]);
        const Vec3 p = m.transform_point(corner);
        const float v[3] = { p.x, p.y, p.z };
        for (int k = 0; k < 3; ++k) {
            if (!any) { outMin[k] = v[k]; outMax[k] = v[k]; }
            else {
                outMin[k] = (v[k] < outMin[k]) ? v[k] : outMin[k];
                outMax[k] = (v[k] > outMax[k]) ? v[k] : outMax[k];
            }
        }
        any = true;
    }
}

const cgltf_accessor* positionAccessor(const cgltf_primitive& prim) {
    for (cgltf_size a = 0; a < prim.attributes_count; ++a)
        if (prim.attributes[a].type == cgltf_attribute_type_position) return prim.attributes[a].data;
    return nullptr;
}

} // namespace

bool probeGltf(const std::string& filepath, bool applyNodeTransforms, ModelProbe& out) {
    out = ModelProbe{};
    if (!isGltfPath(filepath)) return false;

    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, filepath.c_str(), &data) != cgltf_result_success) return false;

    // ★ No cgltf_load_buffers here on purpose — that is the whole saving. It is
    // called below ONLY if some POSITION accessor is missing its mandatory
    // min/max, i.e. for a file that is already out of spec.
    bool buffersLoaded = false;
    auto ensureBuffers = [&]() {
        if (buffersLoaded) return;
        buffersLoaded = true;
        if (cgltf_load_buffers(&options, data, filepath.c_str()) != cgltf_result_success) {
            SCENE_LOG_WARN("[glTF] probe: POSITION accessor without min/max and buffers "
                           "could not be loaded: " + filepath);
        }
    };

    out.mesh_count      = data->meshes_count;
    out.material_count  = data->materials_count;
    out.animation_count = data->animations_count;
    out.node_count      = data->nodes_count;
    out.skin_count      = data->skins_count;

    for (cgltf_size m = 0; m < data->materials_count; ++m) {
        const cgltf_material& mat = data->materials[m];
        const cgltf_texture* slots[] = {
            mat.pbr_metallic_roughness.base_color_texture.texture,
            mat.pbr_metallic_roughness.metallic_roughness_texture.texture,
            mat.normal_texture.texture,
            mat.occlusion_texture.texture,
            mat.emissive_texture.texture,
        };
        for (const cgltf_texture* t : slots) if (t) ++out.texture_reference_count;
    }

    float mn[3] = { 0, 0, 0 }, mx[3] = { 0, 0, 0 };
    bool any = false;

    // Per-primitive bounds, resolved once and reused by both space options.
    auto primitiveBounds = [&](const cgltf_primitive& prim, float pmn[3], float pmx[3]) -> bool {
        const cgltf_accessor* pos = positionAccessor(prim);
        if (!pos || pos->count == 0) return false;
        if (pos->has_min && pos->has_max) {
            for (int k = 0; k < 3; ++k) { pmn[k] = pos->min[k]; pmx[k] = pos->max[k]; }
            return true;
        }
        ensureBuffers();
        std::vector<cgltf_float> tmp(static_cast<size_t>(pos->count) * 3);
        if (cgltf_accessor_unpack_floats(pos, tmp.data(), tmp.size()) == 0) return false;
        for (int k = 0; k < 3; ++k) { pmn[k] = tmp[k]; pmx[k] = tmp[k]; }
        for (size_t v = 1; v < static_cast<size_t>(pos->count); ++v)
            for (int k = 0; k < 3; ++k) {
                const float c = tmp[v * 3 + k];
                if (c < pmn[k]) pmn[k] = c;
                if (c > pmx[k]) pmx[k] = c;
            }
        return true;
    };

    // Counts always cover EVERY mesh, node-referenced or not, so a file whose
    // meshes are not in the active scene still reports its real size.
    for (cgltf_size i = 0; i < data->meshes_count; ++i) {
        const cgltf_mesh& mesh = data->meshes[i];
        for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
            const cgltf_primitive& prim = mesh.primitives[p];
            if (prim.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor* pos = positionAccessor(prim);
            if (!pos) continue;
            out.vertex_count += pos->count;
            out.triangle_count += (prim.indices ? prim.indices->count : pos->count) / 3;

            if (!applyNodeTransforms) {
                float pmn[3], pmx[3];
                if (primitiveBounds(prim, pmn, pmx))
                    growByTransformedBox(Matrix4x4::identity(), pmn, pmx, mn, mx, any);
            }
        }
    }

    if (applyNodeTransforms) {
        for (cgltf_size i = 0; i < data->nodes_count; ++i) {
            const cgltf_node* node = &data->nodes[i];
            if (!node->mesh) continue;
            const Matrix4x4 world = nodeWorldMatrix(node);
            for (cgltf_size p = 0; p < node->mesh->primitives_count; ++p) {
                const cgltf_primitive& prim = node->mesh->primitives[p];
                if (prim.type != cgltf_primitive_type_triangles) continue;
                float pmn[3], pmx[3];
                if (primitiveBounds(prim, pmn, pmx))
                    growByTransformedBox(world, pmn, pmx, mn, mx, any);
            }
        }
    }

    out.has_bounds = any;
    for (int k = 0; k < 3; ++k) { out.bounds_min[k] = mn[k]; out.bounds_max[k] = mx[k]; }

    cgltf_free(data);
    return true;
}

bool isGltfPath(const std::string& filepath) {
    const std::string ext = lowerExtension(filepath);
    return ext == ".gltf" || ext == ".glb";
}

bool readGltf(const std::string& filepath,
              const ImportOptions& options,
              ImportedModel& out,
              std::string& error) {
    out = ImportedModel{};
    out.bones = std::make_shared<BoneData>();
    error.clear();
    try {
        GltfReader reader(filepath, options, out);
        if (reader.run(error)) return true;
    } catch (const std::exception& e) {
        error = std::string("glTF reader threw: ") + e.what();
    } catch (...) {
        error = "glTF reader threw an unknown exception";
    }
    // Leave a well-defined empty result so a caller that falls back cannot
    // accidentally merge half a scene.
    out.objects.clear();
    out.animations.clear();
    out.hierarchy.nodes.clear();
    out.lights.clear();
    out.cameras.clear();
    return false;
}

} // namespace rtimport
