/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiSelect.cpp
* Author:        Kemal Demirtas
* Date:          July 2026
* License:       MIT
* =========================================================================
*
* Selection facade. Most editor operations are selection-driven, so scripting
* the selection is what lets a script drive those paths instead of duplicating
* them (rtapi::duplicateObject already relies on this).
*
* Not undoable — selection is treated like viewport navigation, the same
* exception the camera API takes. Selection changes are picked up by the UI's
* per-frame validation pass, so nothing needs to be marked dirty here.
*/

#include "RtApiInternal.h"

#include <memory>
#include <string>
#include <vector>

#include "SceneSelection.h"
#include "Triangle.h"

namespace rtapi {
namespace {

const char* selectableTypeName(SelectableType t) {
    switch (t) {
        case SelectableType::Object:           return "object";
        case SelectableType::Light:            return "light";
        case SelectableType::Camera:           return "camera";
        case SelectableType::CameraTarget:     return "camera_target";
        case SelectableType::VDBVolume:        return "vdb_volume";
        case SelectableType::GasVolume:        return "gas_volume";
        case SelectableType::ForceField:       return "force_field";
        case SelectableType::ParticleSystem:   return "particle_system";
        case SelectableType::SimulationDomain: return "simulation_domain";
        case SelectableType::ParticleEmitter:  return "particle_emitter";
        case SelectableType::World:            return "world";
        default:                               return "none";
    }
}

// Resolves a flat mesh plus its index in scene.world.objects. Pending-delete
// objects are skipped so a script cannot select something the UI considers gone.
bool findSelectableMesh(const std::string& name, std::shared_ptr<TriangleMesh>& out_mesh,
                        int& out_index) {
    for (size_t i = 0; i < g_ctx->scene.world.objects.size(); ++i) {
        auto tm = std::dynamic_pointer_cast<TriangleMesh>(g_ctx->scene.world.objects[i]);
        if (!tm || tm->nodeName != name) continue;
        if (g_ctx->scene.isEditorPendingDeleteObjectName(name)) continue;
        out_mesh = std::move(tm);
        out_index = static_cast<int>(i);
        return true;
    }
    return false;
}

// Builds the same SelectableItem shape SceneSelection::selectObject produces,
// including the single-face Triangle facade legacy property tools still expect.
SelectableItem makeObjectItem(const std::shared_ptr<TriangleMesh>& mesh, int index,
                              const std::string& name) {
    SelectableItem item;
    item.type = SelectableType::Object;
    item.mesh_object = mesh;
    item.mesh_face_index = 0u;
    if (mesh && mesh->num_triangles() > 0) item.object = std::make_shared<Triangle>(mesh, 0u);
    item.object_index = index;
    item.name = name;
    return item;
}

} // namespace

std::vector<SelectionItem> listSelection() {
    std::vector<SelectionItem> out;
    if (!g_ctx) return out;
    for (const SelectableItem& item : g_ctx->selection.multi_selection) {
        if (!item.is_valid()) continue;
        SelectionItem info;
        info.type = selectableTypeName(item.type);
        info.name = item.name;
        info.index = (item.type == SelectableType::Light) ? item.light_index : item.object_index;
        info.primary = (item == g_ctx->selection.selected);
        out.push_back(std::move(info));
    }
    return out;
}

Result selectObject(const std::string& name, bool additive) {
    if (!g_ctx) return notBound();

    std::shared_ptr<TriangleMesh> mesh;
    int index = -1;
    if (!findSelectableMesh(name, mesh, index))
        return Result::fail("object not found: " + name);

    if (additive) {
        g_ctx->selection.addToSelection(makeObjectItem(mesh, index, name));
    } else {
        // SceneSelection::selectObject clears first and builds the facade itself.
        g_ctx->selection.selectObject(mesh, index, name, 0u, nullptr);
    }
    return Result::success();
}

Result deselectObject(const std::string& name) {
    if (!g_ctx) return notBound();

    std::shared_ptr<TriangleMesh> mesh;
    int index = -1;
    if (!findSelectableMesh(name, mesh, index))
        return Result::fail("object not found: " + name);

    g_ctx->selection.removeFromSelection(makeObjectItem(mesh, index, name));
    g_ctx->selection.syncPrimarySelection();
    return Result::success();
}

Result selectLight(int index, bool additive) {
    if (!g_ctx) return notBound();
    auto& lights = g_ctx->scene.lights;
    if (index < 0 || index >= static_cast<int>(lights.size()) || !lights[index])
        return Result::fail("light index out of range: " + std::to_string(index));

    if (additive) {
        SelectableItem item;
        item.type = SelectableType::Light;
        item.light = lights[index];
        item.light_index = index;
        item.name = lights[index]->nodeName;
        g_ctx->selection.addToSelection(item);
    } else {
        g_ctx->selection.selectLight(lights[index], index, lights[index]->nodeName);
    }
    return Result::success();
}

Result selectAllObjects(int& out_count) {
    if (!g_ctx) return notBound();
    out_count = 0;
    g_ctx->selection.clearSelection();
    for (size_t i = 0; i < g_ctx->scene.world.objects.size(); ++i) {
        auto tm = std::dynamic_pointer_cast<TriangleMesh>(g_ctx->scene.world.objects[i]);
        if (!tm) continue;
        if (g_ctx->scene.isEditorPendingDeleteObjectName(tm->nodeName)) continue;
        g_ctx->selection.addToSelection(
            makeObjectItem(tm, static_cast<int>(i), tm->nodeName));
        ++out_count;
    }
    g_ctx->selection.syncPrimarySelection();
    return Result::success();
}

Result clearSelection() {
    if (!g_ctx) return notBound();
    g_ctx->selection.clearSelection();
    return Result::success();
}

} // namespace rtapi
