/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          Api/RtApiModifiers.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "RtApiInternal.h"
#include "MeshModifiers.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cctype>

namespace rtapi {

namespace {

// Returns the OWNING shared_ptr (not a raw pointer): base_mesh_cache stores facade Triangles
// whose parentMesh must keep the source TriangleMesh alive. syncModifierEvaluatedMesh reassigns
// ctx.scene.world.objects below, which drops the object's owning ref there — if the facades held
// only a non-owning alias, the mesh would be freed and the cached facades would dangle, so the
// next re-evaluation read freed memory (UAF → giant garbage geometry + heap corruption).
std::shared_ptr<TriangleMesh> findFlatMesh(UIContext& ctx, const std::string& name) {
    for (auto& obj : ctx.scene.world.objects) {
        if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (tm->nodeName == name) return tm;
        }
    }
    return nullptr;
}

void syncModifierEvaluatedMesh(UIContext& ctx, const std::string& object_name) {
    auto stack_it = ctx.scene.mesh_modifiers.find(object_name);
    if (stack_it == ctx.scene.mesh_modifiers.end()) return;

    if (ctx.scene.base_mesh_cache.find(object_name) == ctx.scene.base_mesh_cache.end() ||
        ctx.scene.base_mesh_cache[object_name].empty()) {
        std::vector<std::shared_ptr<Triangle>> baseTriangles;
        std::shared_ptr<TriangleMesh> tm = findFlatMesh(ctx, object_name);
        if (tm && tm->geometry) {
            size_t nTris = tm->num_triangles();
            // OWNING parentMesh keeps this source mesh alive in base_mesh_cache after the
            // world.objects reassignment below frees the scene's copy.
            for (size_t t = 0; t < nTris; ++t) {
                baseTriangles.push_back(std::make_shared<Triangle>(tm, static_cast<uint32_t>(t)));
            }
        } else {
            for (const auto& obj : ctx.scene.world.objects) {
                if (obj->isTriangle()) {
                    auto tri = std::static_pointer_cast<Triangle>(obj);
                    if (tri->getNodeName() == object_name) {
                        baseTriangles.push_back(tri);
                    }
                }
            }
        }
        if (!baseTriangles.empty()) {
            ctx.scene.base_mesh_cache[object_name] = baseTriangles;
        }
    }

    const auto& baseMesh = ctx.scene.base_mesh_cache[object_name];
    if (baseMesh.empty()) return;

    const auto& modifierStack = stack_it->second;
    bool forRender = !g_solid_viewport_active;

    std::shared_ptr<TriangleMesh> outMesh;
    std::vector<std::shared_ptr<Triangle>> evaluatedTriangles;

    if (!modifierStack.modifiers.empty()) {
        evaluatedTriangles = modifierStack.evaluate(baseMesh, forRender, &outMesh);
    } else {
        evaluatedTriangles = baseMesh;
    }

    std::vector<std::shared_ptr<Hittable>> newObjects;
    // world.objects has a single canonical representation for evaluated meshes:
    // one flat TriangleMesh. Facades remain an internal modifier input/view only.
    if (!outMesh && !evaluatedTriangles.empty()) {
        outMesh = MeshModifiers::facadesToFlatMesh(evaluatedTriangles);
    }
    newObjects.reserve(ctx.scene.world.objects.size() + (outMesh ? 1 : evaluatedTriangles.size()));

    for (const auto& obj : ctx.scene.world.objects) {
        if (obj->isTriangle()) {
            auto tri = std::static_pointer_cast<Triangle>(obj);
            if (tri->getNodeName() != object_name) {
                newObjects.push_back(obj);
            }
        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
            if (tm->nodeName != object_name) {
                newObjects.push_back(obj);
            }
        } else {
            newObjects.push_back(obj);
        }
    }

    if (outMesh) {
        newObjects.push_back(outMesh);
    } else {
        // Defensive fallback for malformed/empty conversion input. Normal modifier
        // paths must never persist these facades in world.objects.
        for (const auto& tri : evaluatedTriangles) {
            newObjects.push_back(tri);
        }
    }

    ctx.scene.world.objects = newObjects;

    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_vulkan_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    ::ui.mesh_cache_valid = false;

    scheduleSceneMutationRebuilds(ctx, true);
}

} // namespace

Result getModifierStack(const std::string& object_name, std::vector<rtapi::ModifierInfo>& out_stack) {
    out_stack.clear();
    if (!g_ctx) return notBound();

    auto it = g_ctx->scene.mesh_modifiers.find(object_name);
    if (it == g_ctx->scene.mesh_modifiers.end()) {
        return Result::success();
    }

    const auto& modifiers = it->second.modifiers;
    out_stack.reserve(modifiers.size());

    for (size_t i = 0; i < modifiers.size(); ++i) {
        const auto& mod = modifiers[i];
        rtapi::ModifierInfo info;
        info.index = static_cast<int>(i);
        info.name = mod.name;

        switch (mod.type) {
            case MeshModifiers::ModifierType::CatmullClark:
                info.type = "catmull_clark";
                break;
            case MeshModifiers::ModifierType::FlatSubdivision:
                info.type = "simple";
                break;
            case MeshModifiers::ModifierType::SmoothSubdivision:
                info.type = "smooth";
                break;
            case MeshModifiers::ModifierType::Bevel:
                info.type = "bevel";
                break;
            case MeshModifiers::ModifierType::WaterSurface:
                info.type = "water_surface";
                break;
            default:
                info.type = "unknown";
                break;
        }

        info.enabled = mod.enabled;
        info.levels = mod.levels;
        info.render_levels = mod.renderLevels;
        info.smooth_angle = mod.smoothAngle;
        out_stack.push_back(info);
    }

    return Result::success();
}

Result addModifier(const std::string& object_name, const std::string& type_in, const std::string& name,
                   int levels, int render_levels, rtapi::ModifierInfo& out_mod) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!objectExists(object_name)) {
        return Result::fail("object not found: " + object_name);
    }

    std::string type = type_in;
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    MeshModifiers::ModifierData mod;
    if (type == "catmull_clark" || type == "subdivision" || type == "catmullclark" || type == "cc") {
        mod.type = MeshModifiers::ModifierType::CatmullClark;
        mod.name = name.empty() ? "Subdivision" : name;
    } else if (type == "simple" || type == "flat" || type == "flatsubdivision") {
        mod.type = MeshModifiers::ModifierType::FlatSubdivision;
        mod.name = name.empty() ? "Simple Subdivision" : name;
    } else if (type == "smooth" || type == "smoothsubdivision") {
        mod.type = MeshModifiers::ModifierType::SmoothSubdivision;
        mod.name = name.empty() ? "Smooth Subdivision" : name;
    } else {
        return Result::fail("unsupported modifier type: " + type_in);
    }

    mod.levels = std::clamp(levels, 1, 10);
    mod.renderLevels = std::clamp(render_levels, 1, 10);
    mod.enabled = true;

    auto& stack = g_ctx->scene.mesh_modifiers[object_name];
    stack.modifiers.push_back(mod);

    size_t new_idx = stack.modifiers.size() - 1;
    out_mod.index = static_cast<int>(new_idx);
    out_mod.name = mod.name;
    out_mod.type = (mod.type == MeshModifiers::ModifierType::CatmullClark) ? "catmull_clark" :
                   (mod.type == MeshModifiers::ModifierType::FlatSubdivision) ? "simple" : "smooth";
    out_mod.enabled = mod.enabled;
    out_mod.levels = mod.levels;
    out_mod.render_levels = mod.renderLevels;
    out_mod.smooth_angle = mod.smoothAngle;

    syncModifierEvaluatedMesh(*g_ctx, object_name);
    return Result::success();
}

Result removeModifier(const std::string& object_name, int index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    auto it = g_ctx->scene.mesh_modifiers.find(object_name);
    if (it == g_ctx->scene.mesh_modifiers.end() || index < 0 || index >= static_cast<int>(it->second.modifiers.size())) {
        return Result::fail("invalid modifier index: " + std::to_string(index) + " for object: " + object_name);
    }

    it->second.modifiers.erase(it->second.modifiers.begin() + index);

    syncModifierEvaluatedMesh(*g_ctx, object_name);
    return Result::success();
}

Result updateModifier(const std::string& object_name, int index,
                       const std::string* new_name, const bool* enabled,
                       const int* levels, const int* render_levels,
                       const float* smooth_angle) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    auto it = g_ctx->scene.mesh_modifiers.find(object_name);
    if (it == g_ctx->scene.mesh_modifiers.end() || index < 0 || index >= static_cast<int>(it->second.modifiers.size())) {
        return Result::fail("invalid modifier index: " + std::to_string(index) + " for object: " + object_name);
    }

    auto& mod = it->second.modifiers[index];
    if (new_name) mod.name = *new_name;
    if (enabled) mod.enabled = *enabled;
    if (levels) mod.levels = std::clamp(*levels, 1, 10);
    if (render_levels) mod.renderLevels = std::clamp(*render_levels, 1, 10);
    if (smooth_angle) mod.smoothAngle = std::clamp(*smooth_angle, 0.0f, 1.0f);

    syncModifierEvaluatedMesh(*g_ctx, object_name);
    return Result::success();
}

Result applyModifier(const std::string& object_name, int index) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    auto it = g_ctx->scene.mesh_modifiers.find(object_name);
    if (it == g_ctx->scene.mesh_modifiers.end() || index < 0 || index >= static_cast<int>(it->second.modifiers.size())) {
        return Result::fail("invalid modifier index: " + std::to_string(index) + " for object: " + object_name);
    }

    if (g_ctx->scene.base_mesh_cache.find(object_name) == g_ctx->scene.base_mesh_cache.end() ||
        g_ctx->scene.base_mesh_cache[object_name].empty()) {
        std::vector<std::shared_ptr<Triangle>> baseTriangles;
        std::shared_ptr<TriangleMesh> tm = findFlatMesh(*g_ctx, object_name);
        if (tm && tm->geometry) {
            size_t nTris = tm->num_triangles();
            // OWNING parentMesh keeps this source mesh alive in base_mesh_cache after the
            // world.objects reassignment below frees the scene's copy.
            for (size_t t = 0; t < nTris; ++t) {
                baseTriangles.push_back(std::make_shared<Triangle>(tm, static_cast<uint32_t>(t)));
            }
        } else {
            for (const auto& obj : g_ctx->scene.world.objects) {
                if (obj->isTriangle()) {
                    auto tri = std::static_pointer_cast<Triangle>(obj);
                    if (tri->getNodeName() == object_name) {
                        baseTriangles.push_back(tri);
                    }
                }
            }
        }
        if (baseTriangles.empty()) {
            return Result::fail("base mesh for object empty: " + object_name);
        }
        g_ctx->scene.base_mesh_cache[object_name] = baseTriangles;
    }

    const auto& baseMesh = g_ctx->scene.base_mesh_cache[object_name];
    MeshModifiers::ModifierStack bakedStack;
    bakedStack.modifiers.assign(
        it->second.modifiers.begin(),
        it->second.modifiers.begin() + index + 1);
    bakedStack.edgeCreases = it->second.edgeCreases;

    std::vector<std::shared_ptr<Triangle>> bakedMesh = bakedStack.evaluate(baseMesh);
    g_ctx->scene.base_mesh_cache[object_name] = bakedMesh;

    it->second.modifiers.erase(
        it->second.modifiers.begin(),
        it->second.modifiers.begin() + index + 1);

    syncModifierEvaluatedMesh(*g_ctx, object_name);
    return Result::success();
}

} // namespace rtapi
