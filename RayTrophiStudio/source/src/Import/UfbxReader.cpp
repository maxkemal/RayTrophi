#include "Import/UfbxReader.h"
#include "UfbxMaterials.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "Animation/AnimationData.h"
#include "globals.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace rtimport {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string str(ufbx_string s) { return std::string(s.data ? s.data : "", s.length); }
Vec3 vec(ufbx_vec3 v) { return Vec3(float(v.x), float(v.y), float(v.z)); }
Matrix4x4 matrix(const ufbx_matrix& src) {
    Matrix4x4 dst = Matrix4x4::identity();
    // ufbx is a column-major affine 3x4; the engine is row-major 4x4.
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 3; ++row)
            dst.m[row][col] = float(src.cols[col].v[row]);
    return dst;
}
struct Part {
    uint32_t material;
    std::vector<uint32_t> corners;
};
// Triangulate once per SOURCE mesh, not once per node. Corner indices preserve
// UV seams, split normals and polygon winding, including concave n-gons.
std::vector<Part> topology(const ufbx_mesh& mesh) {
    if (mesh.max_face_triangles > (std::numeric_limits<size_t>::max)() / 3)
        throw std::runtime_error("FBX face size overflow");
    std::vector<uint32_t> scratch(mesh.max_face_triangles * 3);
    std::vector<Part> result;
    result.reserve(mesh.material_parts.count);
    for (const auto& source : mesh.material_parts) {
        if (!source.num_triangles) continue;
        if (source.num_triangles > (std::numeric_limits<uint32_t>::max)() / 3)
            throw std::runtime_error("FBX material part exceeds the engine vertex limit");
        Part part{source.index, {}};
        part.corners.reserve(source.num_triangles * 3);
        for (uint32_t f : source.face_indices) {
            const ufbx_face face = mesh.faces[f];
            if (face.num_indices < 3) continue;
            const size_t count = ufbx_triangulate_face(scratch.data(), scratch.size(), &mesh, face);
            if (count != size_t(face.num_indices - 2))
                throw std::runtime_error("FBX polygon triangulation failed");
            part.corners.insert(part.corners.end(), scratch.begin(), scratch.begin() + count * 3);
        }
        result.push_back(std::move(part));
    }
    return result;
}

// ---------------------------------------------------------------------------
// Which engine bone index each ufbx cluster maps to, for one mesh.
// ---------------------------------------------------------------------------
struct SkinBinding {
    const ufbx_skin_deformer* deformer = nullptr;
    std::vector<int> clusterBone;   // cluster index -> engine bone index, -1 = unmapped
    bool valid() const { return deformer != nullptr; }
};

// ---------------------------------------------------------------------------
// ★★★★★ THE SKELETON IS THE BONES **PLUS EVERY ANCESTOR OF A BONE**.
//
// AnimationController::getAnimatedGlobalTransform builds a bone's world matrix
// by walking boneParents UPWARD, reading each ancestor's local matrix out of
// boneDefaultTransforms. The walk stops at the first name with no entry. So
// registering only the clusters' bone nodes truncates the chain at the armature
// node — and everything that node carried (the axis conversion ufbx applied,
// the unit scale) is silently dropped.
//
// ★ The symptom is NOT a broken pose: the bones still move correctly relative
// to each other, so the animation looks right while the character is giant and
// lying on the wrong axis. Nobody files that as "the skeleton lost a node".
// This cost a full batch on the glTF side; see GltfDirectReader::buildSkins,
// which builds exactly this closure, and docs/dev/FAZ3_DEVIR_NOTU.md.
// ---------------------------------------------------------------------------
void buildSkeleton(const ufbx_scene& scene,
                   const std::unordered_map<const ufbx_node*, std::string>& names,
                   const std::vector<ufbx_baked_anim*>& bakes,
                   const ImportOptions& options,
                   ImportedModel& out) {
    if (!out.bones) out.bones = std::make_shared<BoneData>();
    BoneData& bones = *out.bones;

    std::unordered_set<const ufbx_node*> technical;   // bones + every ancestor
    std::unordered_set<const ufbx_node*> indexed;     // needs a runtime bone slot
    std::unordered_map<const ufbx_node*, Matrix4x4> offsets;

    auto addChain = [&](const ufbx_node* n) {
        for (const ufbx_node* c = n; c; c = c->parent) technical.insert(c);
    };

    size_t conflicting = 0;
    for (const ufbx_skin_deformer* skin : scene.skin_deformers) {
        for (const ufbx_skin_cluster* cluster : skin->clusters) {
            const ufbx_node* bone = cluster->bone_node;
            if (!bone || !names.count(bone)) continue;
            indexed.insert(bone);
            addChain(bone);
            bones.weightedBoneNames.insert(names.at(bone));

            // ★ geometry_to_bone IS the offset matrix: ufbx documents the skinning
            // matrix as `bone->node_to_world * cluster->geometry_to_bone`, which is
            // exactly the engine's `jointWorld * offset`.
            const Matrix4x4 offset = matrix(cluster->geometry_to_bone);
            const auto existing = offsets.find(bone);
            if (existing == offsets.end()) {
                offsets.emplace(bone, offset);
            } else {
                // ★ A bone shared by two meshes has a DIFFERENT geometry_to_bone per
                // mesh, because each mesh has its own geometry space. BoneData keys
                // offsets by bone NAME — one global map — so that difference cannot
                // be represented. Assimp has the same limitation. Say it out loud
                // rather than letting the second mesh silently wear the first
                // mesh's bind pose.
                for (int r = 0; r < 4 && !conflicting; ++r)
                    for (int c = 0; c < 4; ++c)
                        if (std::fabs(existing->second.m[r][c] - offset.m[r][c]) > 1e-4f) {
                            ++conflicting;
                            break;
                        }
            }
        }
    }
    if (conflicting) {
        SCENE_LOG_WARN("[ufbx] " + std::to_string(conflicting) + " bone(s) are bound to more than "
            "one mesh with different bind matrices; BoneData stores one offset per bone NAME, so "
            "the first binding wins. Those meshes may deform against the wrong bind pose.");
    }

    // Animated nodes that carry no weights still need a skeleton slot: an
    // animation-only import (or a rig whose root is animated but unskinned)
    // would otherwise have no runtime representation at all.
    for (const ufbx_baked_anim* bake : bakes) {
        if (!bake) continue;
        for (const ufbx_baked_node& baked : bake->nodes) {
            if (baked.typed_id >= scene.nodes.count) continue;
            const ufbx_node* node = scene.nodes[baked.typed_id];
            if (!node || !names.count(node)) continue;
            indexed.insert(node);
            addChain(node);
        }
    }

    if (technical.empty()) {
        bones.globalInverseTransform = Matrix4x4::identity();
        if (!options.importPrefix.empty())
            bones.perModelInverses[options.importPrefix] = bones.globalInverseTransform;
        bones.rebuildReverseLookup();
        out.stats.bone_count = 0;
        return;
    }

    // Register the closure in HIERARCHY order. The order is not cosmetic:
    // OzzRuntime and the anim graph map their own joint arrays onto these
    // indices by position.
    std::function<void(const ufbx_node*)> visit = [&](const ufbx_node* node) {
        if (!node) return;
        if (technical.count(node) && names.count(node)) {
            const std::string name = names.at(node);
            bones.boneDefaultTransforms[name] = matrix(node->node_to_parent);

            // Only link to a parent that is itself in the skeleton, so the chain
            // terminates cleanly instead of pointing at an unresolvable name.
            if (node->parent && technical.count(node->parent) && names.count(node->parent))
                bones.boneParents[name] = names.at(node->parent);

            if (indexed.count(node) && !bones.boneNameToIndex.count(name)) {
                bones.boneNameToIndex[name] =
                    static_cast<unsigned int>(bones.boneNameToIndex.size());
                const auto off = offsets.find(node);
                bones.boneOffsetMatrices[name] =
                    (off != offsets.end()) ? off->second : Matrix4x4::identity();
            }
        }
        for (const ufbx_node* child : node->children) visit(child);
    };
    visit(scene.root_node);

    // Bind matrices already carry the mesh->bone space change, and bone world
    // matrices come from this same hierarchy, so no extra global inverse is
    // needed — same conclusion as the glTF reader.
    bones.globalInverseTransform = Matrix4x4::identity();
    if (!options.importPrefix.empty())
        bones.perModelInverses[options.importPrefix] = bones.globalInverseTransform;
    bones.rebuildReverseLookup();
    out.stats.bone_count = bones.boneNameToIndex.size();
}

// ---------------------------------------------------------------------------
// Animation: BAKED, not read curve by curve.
//
// ★ FBX transform animation is not a list of TRS keys. It carries pre/post
// rotation, per-node Euler order, rotation/scaling pivots and offsets, and
// inherit modes — and evaluating those by hand is how FBX importers get poses
// that are almost right. ufbx_bake_anim() resolves all of it and hands back
// plain translation/rotation/scale keys per node, which is exactly the shape
// AnimationData stores.
// ---------------------------------------------------------------------------
void buildAnimations(const ufbx_scene& scene,
                     const std::vector<ufbx_baked_anim*>& bakes,
                     const std::unordered_map<const ufbx_node*, std::string>& names,
                     const ImportOptions& options,
                     ImportedModel& out) {
    for (size_t i = 0; i < bakes.size(); ++i) {
        const ufbx_baked_anim* bake = bakes[i];
        if (!bake || i >= scene.anim_stacks.count) continue;
        const ufbx_anim_stack* stack = scene.anim_stacks[i];

        auto clip = std::make_shared<AnimationData>();
        const std::string raw = stack->name.length ? str(stack->name)
                                                   : ("Animation_" + std::to_string(i));
        clip->name = options.importPrefix.empty() ? raw : options.importPrefix + "_" + raw;
        clip->modelName = options.importPrefix;
        // Baked key times are SECONDS. Declaring 1 tick == 1 second keeps the
        // stored numbers untouched and the playback rate correct; inventing a
        // 24 or 30 here would silently retime every imported clip.
        clip->ticksPerSecond = 1.0;

        // ★ Rebase to zero. FBX stacks routinely start at a non-zero time (a
        // 1-based frame origin is common), and a clip whose first key sits at
        // t=41.7 s plays as "nothing happens" until the playhead gets there.
        const double origin = bake->key_time_min;
        double maxTime = 0.0;

        for (const ufbx_baked_node& baked : bake->nodes) {
            if (baked.typed_id >= scene.nodes.count) continue;
            const ufbx_node* node = scene.nodes[baked.typed_id];
            if (!node || !names.count(node)) continue;
            const std::string target = names.at(node);

            for (const ufbx_baked_vec3& key : baked.translation_keys) {
                const double t = key.time - origin;
                maxTime = (std::max)(maxTime, t);
                clip->positionKeys[target].emplace_back(t, vec(key.value));
            }
            for (const ufbx_baked_quat& key : baked.rotation_keys) {
                const double t = key.time - origin;
                maxTime = (std::max)(maxTime, t);
                // ufbx quaternions are (x, y, z, w); Quaternion is (w, x, y, z).
                clip->rotationKeys[target].emplace_back(
                    t, Quaternion(float(key.value.w), float(key.value.x),
                                  float(key.value.y), float(key.value.z)));
            }
            for (const ufbx_baked_vec3& key : baked.scale_keys) {
                const double t = key.time - origin;
                maxTime = (std::max)(maxTime, t);
                clip->scalingKeys[target].emplace_back(t, vec(key.value));
            }
            ++out.stats.animation_channel_count;
        }

        clip->duration = maxTime > 0.0 ? maxTime : 1.0;
        clip->startFrame = 0;
        clip->endFrame = static_cast<int>(clip->duration * 24.0);
        if (!clip->positionKeys.empty() || !clip->rotationKeys.empty() || !clip->scalingKeys.empty()) {
            out.animations.push_back(std::move(clip));
            ++out.stats.animation_count;
        }
    }
}

void emit(const ufbx_mesh& source, const Part& part, const std::string& name,
          const std::shared_ptr<Transform>& transform, FbxMaterialBinding material,
          const SkinBinding& skin, bool& renormalized,
          const ImportOptions& options, ImportedModel& out) {
    const size_t count = part.corners.size();
    if (!count) return;
    auto mesh = std::make_shared<TriangleMesh>();
    mesh->nodeName = name; // All material siblings belong to ONE logical node.
    mesh->transform = transform;
    auto& geo = *mesh->geometry;
    geo.resize_vertices(count);
    for (const char* attr : {"P", "N", "P_orig", "N_orig"}) geo.add_attribute<Vec3>(attr);
    geo.add_attribute<Vec2>("uv");
    geo.add_attribute<uint16_t>("materialID");
    auto* p = geo.get_attribute_data_mut<Vec3>("P");
    auto* n = geo.get_attribute_data_mut<Vec3>("N");
    auto* po = geo.get_attribute_data_mut<Vec3>("P_orig");
    auto* no = geo.get_attribute_data_mut<Vec3>("N_orig");
    auto* uv = geo.get_attribute_data_mut<Vec2>("uv");
    auto* ids = geo.get_attribute_data_mut<uint16_t>("materialID");
    std::vector<Vec2*> sets;
    for (size_t s = 0; s < source.uv_sets.count; ++s) {
        const std::string key = "uv" + std::to_string(s);
        geo.add_attribute<Vec2>(key);
        sets.push_back(geo.get_attribute_data_mut<Vec2>(key));
    }
    Vec3* color = nullptr;
    if (source.vertex_color.exists) {
        geo.add_attribute<Vec3>("Cd");
        color = geo.get_attribute_data_mut<Vec3>("Cd");
    }
    for (size_t v = 0; v < count; ++v) {
        const uint32_t corner = part.corners[v];
        p[v] = po[v] = vec(ufbx_get_vertex_vec3(&source.vertex_position, corner));
        n[v] = no[v] = vec(ufbx_get_vertex_vec3(&source.vertex_normal, corner)).normalize();
        ids[v] = material.id;
        uv[v] = Vec2(0.0f);
        for (size_t s = 0; s < sets.size(); ++s) {
            const auto& attr = source.uv_sets[s].vertex_uv;
            const auto value = attr.exists ? ufbx_get_vertex_vec2(&attr, corner) : ufbx_vec2{};
            sets[s][v] = Vec2(float(value.x), float(value.y)); // FBX and engine: bottom-left V.
            if (s == material.uvSet) uv[v] = sets[s][v];
        }
        if (color) {
            const auto c = ufbx_get_vertex_vec4(&source.vertex_color, corner);
            color[v] = Vec3(float(c.x), float(c.y), float(c.z));
        }
    }
    // ── Skinning ──────────────────────────────────────────────────────
    // ufbx stores weights PER VERTEX while this geometry is per CORNER, so the
    // corner is mapped back through vertex_indices. ufbx already sorts each
    // vertex's weights by decreasing influence, which is the order the skinning
    // paths assume — so the filtering below preserves it rather than re-sorting.
    if (skin.valid()) {
        std::vector<std::vector<std::pair<int, float>>> weights(count);
        for (size_t v = 0; v < count; ++v) {
            const uint32_t corner = part.corners[v];
            if (corner >= source.vertex_indices.count) continue;
            const uint32_t vertex = source.vertex_indices[corner];
            if (vertex >= skin.deformer->vertices.count) continue;
            const ufbx_skin_vertex& sv = skin.deformer->vertices[vertex];

            auto& dst = weights[v];
            dst.reserve(sv.num_weights);
            float sum = 0.0f;
            for (uint32_t w = 0; w < sv.num_weights; ++w) {
                const size_t index = size_t(sv.weight_begin) + w;
                if (index >= skin.deformer->weights.count) break;
                const ufbx_skin_weight& sw = skin.deformer->weights[index];
                if (sw.weight <= 0.0) continue;
                if (sw.cluster_index >= skin.clusterBone.size()) continue;
                const int bone = skin.clusterBone[sw.cluster_index];
                if (bone < 0) continue;
                dst.emplace_back(bone, float(sw.weight));
                sum += float(sw.weight);
            }
            // ★ ufbx states outright that FBX weights are NOT guaranteed
            // normalized, and an unnormalized set does not fail — it shrinks or
            // inflates the mesh around the bones, which reads as a rigging bug.
            // Correct files are left byte-identical; only broken ones are fixed,
            // and the caller says so once.
            if (sum > 1e-6f && std::fabs(sum - 1.0f) > 1e-3f) {
                for (auto& influence : dst) influence.second /= sum;
                renormalized = true;
            }
        }
        geo.skin_weights = std::move(weights);
        ++out.stats.skinned_mesh_count;
    }

    // Existing TriangleMesh consumers require an address table. It is strictly
    // sequential: the canonical geometry above is flat, one vertex per corner.
    geo.indices.resize(count);
    std::iota(geo.indices.begin(), geo.indices.end(), uint32_t(0));
    const size_t faces = count / 3;
    // ★ A skinned mesh keeps one facade PER FACE: Renderer's import-flat collapse
    // excludes skinned meshes ("SoA skinning is a later increment"), and the glTF
    // reader makes the same exception. Diverging here would leave a skinned mesh
    // represented by a single triangle.
    const size_t facades = (options.emitSingleFacadePerMesh && !skin.valid()) ? 1 : faces;
    for (size_t f = 0; f < facades; ++f)
        out.objects.push_back(std::make_shared<Triangle>(mesh, uint32_t(f)));
    ++out.stats.mesh_count;
    out.stats.vertex_count += count;
    out.stats.triangle_count += faces;
}
}

// ---------------------------------------------------------------------------
// Counts and a box, without building a scene. See Import/ModelProbe.h.
// ---------------------------------------------------------------------------
bool probeFbx(const std::string& path, bool applyNodeTransforms, ModelProbe& out) {
    out = ModelProbe{};
    ufbx_load_opts opts{};
    opts.file_format = UFBX_FILE_FORMAT_FBX;
    opts.target_axes = ufbx_axes_right_handed_y_up;
    opts.target_unit_meters = 1.0;
    opts.space_conversion = UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;
    opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
    // ★ The probe's reason to exist: skip embedded image payloads entirely.
    // Decoding them to report a triangle count is what made the old Assimp
    // scan slow enough to be noticeable on a whole library.
    opts.ignore_embedded = true;
    ufbx_error failure{};
    std::unique_ptr<ufbx_scene, decltype(&ufbx_free_scene)> scene(
        ufbx_load_file(path.c_str(), &opts, &failure), &ufbx_free_scene);
    if (!scene) return false;

    out.mesh_count = scene->meshes.count;
    out.material_count = scene->materials.count;
    out.animation_count = scene->anim_stacks.count;
    out.node_count = scene->nodes.count;
    out.skin_count = scene->skin_deformers.count;
    out.texture_reference_count = scene->textures.count;

    float lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    auto include = [&](const ufbx_vec3& v, const ufbx_matrix* place) {
        const ufbx_vec3 p = place ? ufbx_transform_position(place, v) : v;
        const float xyz[3] = {float(p.x), float(p.y), float(p.z)};
        for (int a = 0; a < 3; ++a) {
            if (!out.has_bounds) { lo[a] = hi[a] = xyz[a]; }
            else { lo[a] = (std::min)(lo[a], xyz[a]); hi[a] = (std::max)(hi[a], xyz[a]); }
        }
        out.has_bounds = true;
    };

    if (applyNodeTransforms) {
        // Scene-graph placed bounds: a mesh instanced by several nodes counts
        // at each of its placements, which is what preview framing needs.
        for (const ufbx_node* node : scene->nodes) {
            if (!node->mesh) continue;
            for (const ufbx_vec3& v : node->mesh->vertices) include(v, &node->geometry_to_world);
        }
    }
    for (const ufbx_mesh* mesh : scene->meshes) {
        out.triangle_count += mesh->num_triangles;
        out.vertex_count += mesh->num_vertices;
        if (!applyNodeTransforms)
            for (const ufbx_vec3& v : mesh->vertices) include(v, nullptr);
    }
    for (int a = 0; a < 3; ++a) { out.bounds_min[a] = lo[a]; out.bounds_max[a] = hi[a]; }
    return true;
}

bool readUfbx(const std::string& path, const ImportOptions& requested,
              ImportedModel& out, std::string& error) {
    out = {};
    error.clear();
    try {
        const auto start = Clock::now();
        ufbx_load_opts opts{};
        opts.file_format = UFBX_FILE_FORMAT_FBX;
        opts.target_axes = ufbx_axes_right_handed_y_up;
        opts.target_unit_meters = 1.0;
        opts.space_conversion = UFBX_SPACE_CONVERSION_ADJUST_TRANSFORMS;
        opts.geometry_transform_handling = UFBX_GEOMETRY_TRANSFORM_HANDLING_HELPER_NODES;
        opts.inherit_mode_handling = UFBX_INHERIT_MODE_HANDLING_HELPER_NODES;
        opts.generate_missing_normals = true;
        ufbx_error failure{};
        std::unique_ptr<ufbx_scene, decltype(&ufbx_free_scene)> scene(
            ufbx_load_file(path.c_str(), &opts, &failure), &ufbx_free_scene);
        if (!scene) {
            char detail[2048]{};
            ufbx_format_error(detail, sizeof(detail), &failure);
            error = std::string("ufbx: ") + detail;
            return false;
        }
        // Validate the increment's boundary BEFORE creating materials or objects.
        // ★ Increment 2 adds SKINNING and TRANSFORM ANIMATION. Blend shapes and
        // geometry caches are still unsupported, and that stays a LOUD failure on
        // purpose: reading such a file and quietly ignoring those deformers gives
        // a character frozen in its neutral expression with no error anywhere —
        // the "plausible-looking result" this repo keeps paying for.
        if (scene->blend_deformers.count || scene->cache_deformers.count) {
            error = "ufbx: this FBX uses blend shapes (morph targets) or geometry "
                    "caches, which are not supported yet. The file was NOT imported "
                    "partially - importing it without those deformers would give a "
                    "character frozen in its neutral pose and no error.";
            return false;
        }
        ImportOptions options = requested;
        if (options.importPrefix.empty()) {
            const std::filesystem::path file(path);
            options.importPrefix = file.parent_path().filename().string() + "_" + file.stem().string();
        }
        ImportedModel model;
        model.importName = options.importPrefix;
        model.stats.reader = "ufbx";
        model.stats.seconds_parse = elapsed(start);
        std::unordered_map<const ufbx_node*, std::string> names;
        std::unordered_set<std::string> used;
        std::unordered_map<std::string, size_t> nextSuffix;
        names.reserve(scene->nodes.count);
        used.reserve(scene->nodes.count);
        std::vector<std::pair<const ufbx_node*, int>> pending{{scene->root_node, -1}};
        while (!pending.empty()) {
            const auto entry = pending.back(); pending.pop_back();
            const auto* node = entry.first;
            const std::string raw = node->name.length ? str(node->name) : "Node_" + std::to_string(node->typed_id);
            const std::string base = options.importPrefix + "_" + raw;
            std::string name = base;
            while (!used.insert(name).second)
                name = base + "_" + std::to_string(++nextSuffix[base]);
            names[node] = name;
            const int index = model.hierarchy.addNode(raw, name, matrix(node->node_to_parent), entry.second);
            for (size_t c = node->children.count; c > 0; --c)
                pending.emplace_back(node->children[c - 1], index);
        }
        model.stats.node_count = model.hierarchy.size();
        if ((options.loadCameras && scene->cameras.count) || (options.loadLights && scene->lights.count))
            SCENE_LOG_WARN("[ufbx] FBX cameras/lights are not imported yet.");

        // Bake first: the skeleton closure needs to know which nodes are animated,
        // and the baked result is what buildAnimations consumes afterwards. Index
        // alignment with scene->anim_stacks is kept by pushing nullptr on failure.
        const auto animStart = Clock::now();
        std::vector<std::unique_ptr<ufbx_baked_anim, decltype(&ufbx_free_baked_anim)>> bakeOwners;
        std::vector<ufbx_baked_anim*> bakes;
        if (options.loadAnimations) {
            bakes.reserve(scene->anim_stacks.count);
            for (const ufbx_anim_stack* stack : scene->anim_stacks) {
                ufbx_bake_opts bakeOpts{};
                ufbx_error bakeError{};
                ufbx_baked_anim* baked = ufbx_bake_anim(scene.get(), stack->anim, &bakeOpts, &bakeError);
                if (!baked) {
                    char detail[1024]{};
                    ufbx_format_error(detail, sizeof(detail), &bakeError);
                    SCENE_LOG_WARN("[ufbx] could not bake animation stack '" + str(stack->name) +
                                   "': " + detail);
                    bakes.push_back(nullptr);
                    continue;
                }
                bakeOwners.emplace_back(baked, &ufbx_free_baked_anim);
                bakes.push_back(baked);
            }
        }

        // Skeleton before geometry: emit() resolves each cluster to a bone INDEX,
        // so the index table has to exist first. Getting this order wrong drops
        // every weight and the mesh renders in bind pose — silently.
        if (options.loadSkinning) buildSkeleton(*scene, names, bakes, options, model);
        buildAnimations(*scene, bakes, names, options, model);
        model.stats.seconds_animation = elapsed(animStart);
        const auto materialsStart = Clock::now();
        UfbxMaterials materials(path, options, model.stats);
        if (options.loadGeometry) materials.prefetch(*scene);
        model.stats.seconds_materials = elapsed(materialsStart);
        const auto geometryStart = Clock::now();
        double bindingSeconds = 0.0;
        std::unordered_map<const ufbx_mesh*, std::vector<Part>> cachedTopology;
        cachedTopology.reserve(scene->meshes.count);
        std::unordered_map<const ufbx_mesh*, SkinBinding> skinBindings;
        const SkinBinding noSkin;
        size_t repeatedNodes = 0;
        bool renormalized = false;
        if (options.loadGeometry) for (const auto* node : scene->nodes) {
            if (!node->mesh || !node->mesh->num_triangles) continue;
            const auto& source = *node->mesh;
            auto cached = cachedTopology.find(&source);
            if (cached == cachedTopology.end()) cached = cachedTopology.emplace(&source, topology(source)).first;
            else ++repeatedNodes;

            const SkinBinding* skin = &noSkin;
            if (options.loadSkinning && source.skin_deformers.count && model.bones) {
                auto known = skinBindings.find(&source);
                if (known == skinBindings.end()) {
                    SkinBinding built;
                    built.deformer = source.skin_deformers[0];
                    if (source.skin_deformers.count > 1) {
                        SCENE_LOG_WARN("[ufbx] mesh '" + str(source.name) + "' has " +
                            std::to_string(source.skin_deformers.count) + " skin deformers; the "
                            "engine binds one weight set per mesh, so only the first is used.");
                    }
                    built.clusterBone.assign(built.deformer->clusters.count, -1);
                    for (size_t c = 0; c < built.deformer->clusters.count; ++c) {
                        const ufbx_node* bone = built.deformer->clusters[c]->bone_node;
                        if (!bone || !names.count(bone)) continue;
                        const auto slot = model.bones->boneNameToIndex.find(names.at(bone));
                        if (slot != model.bones->boneNameToIndex.end())
                            built.clusterBone[c] = int(slot->second);
                    }
                    known = skinBindings.emplace(&source, std::move(built)).first;
                }
                skin = &known->second;
            }

            auto transform = std::make_shared<Transform>();
            // ★★ A SKINNED MESH GETS AN IDENTITY BASE, NOT geometry_to_world.
            // ufbx documents the skinning matrix as
            //     bone->node_to_world * cluster->geometry_to_bone
            // i.e. the bone chain already carries the mesh from geometry space to
            // world. Also applying geometry_to_world would apply that term TWICE —
            // the exact failure the glTF reader paid for with skinned meshes
            // (docs/dev/FAZ3_DEVIR_NOTU.md §3.5), and it does not crash: it
            // produces a character that is placed and scaled plausibly wrong.
            transform->setBase(skin->valid() ? Matrix4x4::identity()
                                             : matrix(node->geometry_to_world));
            for (const auto& part : cached->second) {
                const auto* material = part.material < node->materials.count ? node->materials[part.material] : nullptr;
                const auto bindStart = Clock::now();
                const auto binding = materials.bind(material, source);
                bindingSeconds += elapsed(bindStart);
                emit(source, part, names.at(node), transform, binding, *skin, renormalized, options, model);
            }
        }
        if (renormalized) {
            SCENE_LOG_WARN("[ufbx] some vertex weight sets did not sum to 1 and were normalized. "
                "FBX does not guarantee normalized weights; an unnormalized set does not fail, it "
                "shrinks or inflates the mesh around its bones.");
        }
        model.stats.seconds_geometry = (std::max)(0.0, elapsed(geometryStart) - bindingSeconds);
        model.stats.seconds_materials += bindingSeconds;
        model.stats.seconds_total = elapsed(start);
        if (repeatedNodes) SCENE_LOG_INFO("[ufbx] Reused triangulation for " + std::to_string(repeatedNodes) +
            " repeated mesh node(s); flat geometry still owned per node (no implicit instancing).");
        if (options.loadGeometry && model.objects.empty()) {
            error = "ufbx: no polygon geometry in file";
            return false;
        }
        out = std::move(model);
        return true;
    } catch (const std::exception& e) {
        error = std::string("ufbx import failed: ") + e.what();
        return false;
    }
}
}
