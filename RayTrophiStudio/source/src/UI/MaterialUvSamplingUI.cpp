// SPDX-License-Identifier: MIT
#include "UI/MaterialUvSamplingUI.h"
#include "UI/UvEditorUI.h"
#include "UV/UvService.h"
#include "PrincipledBSDF.h"
#include "TriangleMesh.h"
#include "SceneCommand.h"
#include "scene_ui.h"
#include "ui_modern.h"

#include <algorithm>
#include <memory>
#include <unordered_set>
#include <vector>
#include <limits>
#include <utility>

namespace MaterialUvSamplingUI {

bool draw(UIContext& ctx, PrincipledBSDF& material, uint16_t material_id,
          const std::string& object) {
    if (!UIWidgets::BeginSection("Texture Coordinates", ImVec4(0.54f, 0.86f, 1, 1))) {
        return false;
    }
    bool changed = false;
    if (UIWidgets::SecondaryButton("Open UV Editor", ImVec2(-1, 0), !object.empty())) {
        UvEditorUI::open(object);
    }
    // Preserve the material's existing active-set behavior, using only flat data.
    // Set-zero/render-channel separation remains the follow-up described in the UV roadmap.
    static std::vector<std::weak_ptr<TriangleMesh>> meshes;
    static std::string cached_object;
    static uint16_t cached_material = 0;
    static const void* cached_scene = nullptr;
    static uint64_t cached_generation = (std::numeric_limits<uint64_t>::max)();
    static int common_sets = 0;
    const uint64_t generation = UV::sceneRevision();
    if (cached_scene != &ctx.scene || cached_object != object ||
        cached_material != material_id || cached_generation != generation) {
        cached_scene = &ctx.scene;
        cached_object = object;
        cached_material = material_id;
        cached_generation = generation;
        meshes.clear();
        common_sets = 0;
        std::unordered_set<TriangleMesh*> seen;
        for (const auto& entry : ctx.scene.world.objects) {
            auto mesh = std::dynamic_pointer_cast<TriangleMesh>(entry);
            if (!mesh || !mesh->geometry || mesh->nodeName != object ||
                ctx.scene.isEditorPendingDeleteObjectName(object) || !seen.insert(mesh.get()).second) {
                continue;
            }
            const auto& geometry = *mesh->geometry;
            const uint16_t* ids = geometry.get_material_ids();
            if (ids && geometry.get_core_attribute_count(DNA::Attr::MaterialID) !=
                           geometry.get_vertex_count()) {
                continue;
            }
            bool matches = !ids;
            if (ids) {
                for (size_t vertex = 0; vertex < geometry.get_vertex_count(); ++vertex) {
                    if (ids[vertex] == material_id) {
                        matches = true;
                        break;
                    }
                }
            }
            if (!matches) {
                continue;
            }
            int count = geometry.has_attribute("uv") ? 1 : 0;
            while (count > 0 && geometry.has_attribute("uv" + std::to_string(count))) {
                ++count;
            }
            common_sets = meshes.empty() ? count : (std::min)(common_sets, count);
            meshes.push_back(std::move(mesh));
        }
    }
    if (common_sets > 1) {
        const std::string active = "UV " + std::to_string(material.selected_uv_set);
        ImGui::TextUnformatted("UV Set");
        ImGui::SetNextItemWidth(-1);
        if (ImGui::BeginCombo("##material_uv_set", active.c_str())) {
            for (int set = 0; set < common_sets; ++set) {
                const std::string label = "UV " + std::to_string(set);
                if (ImGui::Selectable(label.c_str(), set == material.selected_uv_set) &&
                    set != material.selected_uv_set) {
                    for (const auto& handle : meshes) {
                        const auto mesh = handle.lock();
                        if (!mesh || !mesh->geometry) {
                            continue;
                        }
                        auto& geometry = *mesh->geometry;
                        const std::string name = set == 0 ? "uv" : "uv" + std::to_string(set);
                        const Vec2* source = geometry.get_attribute_data<Vec2>(name);
                        Vec2* destination = geometry.get_uvs_mut();
                        const uint16_t* ids = geometry.get_material_ids();
                        if (!source || !destination) {
                            continue;
                        }
                        for (size_t vertex = 0; vertex < geometry.get_vertex_count(); ++vertex) {
                            if (!ids || ids[vertex] == material_id) {
                                destination[vertex] = source[vertex];
                            }
                        }
                    }
                    material.selected_uv_set = set;
                    scheduleSceneMutationRebuilds(ctx, false);
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
    } else {
        ImGui::TextDisabled("UV Set: UV %d", material.selected_uv_set);
    }
    auto& transform = material.textureTransform;
    static bool lock_scale = true;
    static bool lock_offset = false;
    float scale[2] = {transform.scale.u, transform.scale.v};
    float offset[2] = {transform.translation.u, transform.translation.v};
    ImGui::Checkbox("Lock Scale XY", &lock_scale);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::DragFloat2("##sampling_scale", scale, 0.01f, 0.001f, 64, "Scale %.3f")) {
        if (lock_scale) {
            scale[1] = scale[0];
        }
        transform.scale = Vec2((std::max)(0.001f, scale[0]), (std::max)(0.001f, scale[1]));
        changed = true;
    }
    ImGui::Checkbox("Lock Offset XY", &lock_offset);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::DragFloat2("##sampling_offset", offset, 0.01f, -100, 100, "Offset %.3f")) {
        if (lock_offset) {
            offset[1] = offset[0];
        }
        transform.translation = Vec2(offset[0], offset[1]);
        changed = true;
    }
    ImGui::SetNextItemWidth(-1);
    changed |= ImGui::DragFloat("##sampling_rotation", &transform.rotation_degrees,
                               0.25f, -360, 360, "Rotation %.1f deg");
    const char* wraps[] = {"Repeat", "Mirror", "Clamp", "Planar", "Cubic"};
    int wrap = static_cast<int>(transform.wrapMode);
    ImGui::TextUnformatted("Wrap");
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("##sampling_wrap", &wrap, wraps, IM_ARRAYSIZE(wraps))) {
        transform.wrapMode = static_cast<WrapMode>(wrap);
        changed = true;
    }
    if (UIWidgets::SecondaryButton("Reset UV Transform", ImVec2(-1, 0))) {
        transform.scale = Vec2(1, 1);
        transform.translation = Vec2();
        transform.rotation_degrees = 0;
        transform.wrapMode = WrapMode::Repeat;
        changed = true;
    }
    ImGui::TextWrapped("These settings affect texture sampling. Edit stored mesh UVs in UV Editor.");
    UIWidgets::EndSection();
    return changed;
}

} // namespace MaterialUvSamplingUI
