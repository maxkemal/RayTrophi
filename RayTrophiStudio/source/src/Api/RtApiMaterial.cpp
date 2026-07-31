/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiMaterial.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Material asset facade (Faz 5.5a). RtApi.cpp already exposed per-object
* parameter get/set; what was missing was the asset layer — listing, creating,
* assigning and texturing materials — without which a script cannot build a
* look from scratch.
*
* Materials are addressed by name. MaterialManager keeps Material::materialName
* and its registry key identical (addUniqueMaterial guarantees this), so the
* name is a stable handle for the lifetime of the asset.
*
* Assignment mirrors the material panel's own path: repaint the flat mesh's
* per-vertex materialID buffer, remap the live Embree BVH in place instead of
* paying for a full rebuild, then resync the backend. Not undoable — treated
* like the other bulk-authoring APIs.
*/

#include "RtApiInternal.h"

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "Perlin.h"
#include "Texture.h"
#include "EmbreeBVH.h"
#include "PBRMaterialSnapshot.h"
#include "ProjectManager.h"

namespace rtapi {
namespace {

const char* materialTypeName(const Material* material) {
    if (dynamic_cast<const PrincipledBSDF*>(material)) return "principled";
    if (dynamic_cast<const Volumetric*>(material))      return "volumetric";
    return "other";
}

// Same GPU-side refresh the material panel runs after an edit.
void syncGpuMaterial(PrincipledBSDF& material) {
    if (!material.gpuMaterial) material.gpuMaterial = std::make_shared<GpuMaterial>();
    const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(material);
    applyPBRMaterialSnapshotToGpuMaterial(snapshot, *material.gpuMaterial);
}

void resyncMaterial(uint16_t id) {
    g_ctx->renderer.resetCPUAccumulation();
    if (g_ctx->backend_ptr) {
        g_ctx->renderer.updateBackendMaterial(g_ctx->scene, id);
        g_ctx->backend_ptr->resetAccumulation();
    }
    g_ctx->start_render = true;
    ProjectManager::getInstance().markModified();
}

Result resolveMaterial(const std::string& name, uint16_t& out_id, Material*& out_material) {
    auto& manager = MaterialManager::getInstance();
    const uint16_t id = manager.getMaterialID(name);
    if (id == MaterialManager::INVALID_MATERIAL_ID)
        return Result::fail("material not found: " + name);
    Material* material = manager.getMaterial(id);
    if (!material) return Result::fail("material not found: " + name);
    out_id = id;
    out_material = material;
    return Result::success();
}

// Texture slot name -> (property, TextureType). Height has no dedicated
// TextureType in the engine, matching the material panel which passes Unknown.
struct TextureSlot {
    MaterialProperty* property = nullptr;
    TextureType type = TextureType::Unknown;
};

bool resolveTextureSlot(PrincipledBSDF& material, const std::string& slot, TextureSlot& out) {
    if (slot == "base_color" || slot == "albedo") {
        out = { &material.albedoProperty, TextureType::Albedo };
    } else if (slot == "roughness") {
        out = { &material.roughnessProperty, TextureType::Roughness };
    } else if (slot == "metallic") {
        out = { &material.metallicProperty, TextureType::Metallic };
    } else if (slot == "normal") {
        out = { &material.normalProperty, TextureType::Normal };
    } else if (slot == "emission") {
        out = { &material.emissionProperty, TextureType::Emission };
    } else if (slot == "opacity") {
        out = { &material.opacityProperty, TextureType::Opacity };
    } else if (slot == "specular") {
        out = { &material.specularProperty, TextureType::Specular };
    } else if (slot == "transmission") {
        out = { &material.transmissionProperty, TextureType::Transmission };
    } else if (slot == "height") {
        out = { &material.heightProperty, TextureType::Unknown };
    } else {
        return false;
    }
    return true;
}

const char* kTextureSlotList =
    "base_color|roughness|metallic|normal|emission|opacity|specular|transmission|height";

} // namespace

std::vector<MaterialInfo> listMaterials() {
    std::vector<MaterialInfo> out;
    auto& manager = MaterialManager::getInstance();
    const auto& materials = manager.getAllMaterials();
    out.reserve(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) {
        const Material* material = materials[i].get();
        if (!material) continue;
        MaterialInfo info;
        info.id = static_cast<uint16_t>(i);
        info.name = manager.getMaterialName(static_cast<uint16_t>(i));
        info.type = materialTypeName(material);
        out.push_back(std::move(info));
    }
    return out;
}

Result getMaterial(const std::string& name, MaterialInfo& out) {
    uint16_t id = 0;
    Material* material = nullptr;
    if (Result r = resolveMaterial(name, id, material); !r) return r;
    out.id = id;
    out.name = MaterialManager::getInstance().getMaterialName(id);
    out.type = materialTypeName(material);
    return Result::success();
}

Result createMaterial(const std::string& type, const std::string& requested_name,
                      std::string& out_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    std::shared_ptr<Material> material;
    std::string base_name = requested_name;
    if (type == "principled") {
        // Same defaults as the panel's "Create New Surface".
        material = std::make_shared<PrincipledBSDF>(Vec3(0.8f), 0.5f, 0.0f);
        if (base_name.empty()) base_name = "Surface";
    } else if (type == "volumetric") {
        material = std::make_shared<Volumetric>(Vec3(0.8f), 1.0f, 0.1f, 0.5f, Vec3(0.0f),
                                                std::make_shared<Perlin>());
        if (base_name.empty()) base_name = "Volume";
    } else {
        return Result::fail("unknown material type (principled|volumetric): " + type);
    }

    // addUniqueMaterial keeps materialName and the registry key identical, so the
    // returned name stays a valid handle even when the requested one was taken.
    const uint16_t id = MaterialManager::getInstance().addUniqueMaterial(base_name, material);
    if (id == MaterialManager::INVALID_MATERIAL_ID)
        return Result::fail("material registry is full");

    if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material.get())) syncGpuMaterial(*pbsdf);
    out_name = MaterialManager::getInstance().getMaterialName(id);
    ProjectManager::getInstance().markModified();
    return Result::success();
}

std::vector<std::string> objectMaterials(const std::string& object_name) {
    std::vector<std::string> out;
    if (!g_ctx) return out;
    auto& manager = MaterialManager::getInstance();
    for (uint16_t id : objectMaterialIds(*g_ctx, object_name)) out.push_back(manager.getMaterialName(id));
    return out;
}

Result assignMaterial(const std::string& object_name, const std::string& material_name) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (!objectExists(object_name)) return Result::fail("object not found: " + object_name);

    uint16_t new_id = 0;
    Material* replacement = nullptr;
    if (Result r = resolveMaterial(material_name, new_id, replacement); !r) return r;

    // Repaint every flat mesh carrying this node name. Unlike the panel, which
    // swaps one slot, this replaces the object's whole material assignment —
    // that is the useful primitive for a script building a look.
    bool touched = false;
    std::unordered_set<uint16_t> replaced_ids;
    auto embree = std::dynamic_pointer_cast<EmbreeBVH>(g_ctx->scene.bvh);
    for (auto& object : g_ctx->scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || mesh->nodeName != object_name || !mesh->geometry) continue;

        uint16_t* material_ids =
            mesh->geometry->get_attribute_data_mut<uint16_t>("materialID");
        if (!material_ids) continue;
        const size_t vertex_count = mesh->geometry->get_vertex_count();
        std::unordered_set<uint16_t> mesh_ids;
        for (size_t v = 0; v < vertex_count; ++v) {
            if (material_ids[v] == new_id) continue;
            mesh_ids.insert(material_ids[v]);
            material_ids[v] = new_id;
        }
        touched = true;

        // Remap the live CPU BVH in place; a dense mesh would otherwise pay a
        // multi-second full rebuild for what is a per-primitive id swap.
        if (embree) {
            for (uint16_t old_id : mesh_ids) embree->remapMeshMaterialID(mesh.get(), old_id, new_id);
        }
        replaced_ids.insert(mesh_ids.begin(), mesh_ids.end());
    }
    if (!touched) return Result::fail("object has no flat mesh geometry: " + object_name);

    for (uint16_t old_id : replaced_ids)
        g_ctx->renderer.updateMeshMaterialBinding(g_ctx->scene, object_name, old_id, new_id);
    resyncMaterial(new_id);
    return Result::success();
}

Result setMaterialTexture(const std::string& material_name, const std::string& slot,
                          const std::string& filepath) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    uint16_t id = 0;
    Material* material = nullptr;
    if (Result r = resolveMaterial(material_name, id, material); !r) return r;
    auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material);
    if (!pbsdf) return Result::fail("texture slots require a Principled BSDF material: " + material_name);

    TextureSlot target;
    if (!resolveTextureSlot(*pbsdf, slot, target))
        return Result::fail("unknown texture slot: " + slot + " (" + kTextureSlotList + ")");

    auto texture = std::make_shared<Texture>(filepath, target.type);
    if (!texture || !texture->is_loaded())
        return Result::fail("failed to load texture: " + filepath);
    texture->upload_to_gpu();

    target.property->texture = std::move(texture);
    syncGpuMaterial(*pbsdf);
    resyncMaterial(id);
    return Result::success();
}

Result clearMaterialTexture(const std::string& material_name, const std::string& slot) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    uint16_t id = 0;
    Material* material = nullptr;
    if (Result r = resolveMaterial(material_name, id, material); !r) return r;
    auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material);
    if (!pbsdf) return Result::fail("texture slots require a Principled BSDF material: " + material_name);

    TextureSlot target;
    if (!resolveTextureSlot(*pbsdf, slot, target))
        return Result::fail("unknown texture slot: " + slot + " (" + kTextureSlotList + ")");

    target.property->texture.reset();
    syncGpuMaterial(*pbsdf);
    resyncMaterial(id);
    return Result::success();
}

std::vector<std::string> materialTextureSlots(const std::string& material_name) {
    std::vector<std::string> out;
    uint16_t id = 0;
    Material* material = nullptr;
    if (!resolveMaterial(material_name, id, material)) return out;
    auto* pbsdf = dynamic_cast<PrincipledBSDF*>(material);
    if (!pbsdf) return out;

    static const char* kSlots[] = { "base_color", "roughness", "metallic", "normal",
                                    "emission", "opacity", "specular", "transmission", "height" };
    for (const char* slot : kSlots) {
        TextureSlot target;
        if (!resolveTextureSlot(*pbsdf, slot, target)) continue;
        if (target.property->texture) out.emplace_back(slot);
    }
    return out;
}

} // namespace rtapi
