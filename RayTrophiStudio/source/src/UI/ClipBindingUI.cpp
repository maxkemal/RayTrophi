#include "UI/ClipBindingUI.h"
#include "UI/ClipBindingUIState.h"
#include "Api/RtApi.h"
#include "Api/RtApiClipBinding.h"
#include "Animation/ClipBinding.h"
#include "scene_ui.h"
#include "imgui.h"
#include <unordered_map>
namespace RigUI {
ClipBindingUIState& clipBindingState(const std::string& target) {
    static std::unordered_map<std::string, ClipBindingUIState> states;
    return states[target];
}
void drawClipBindingContents(UIContext& ctx, const std::string& target, bool manual) {
    auto& state = clipBindingState(target);
    ImGui::TextWrapped("Copy a clip onto this rig, with optional rest-basis correction.");
    if (ImGui::BeginCombo("Source rig", state.source.empty() ? "Choose source" : state.source.c_str())) {
        for (const auto& model : ctx.scene.importedModelContexts) {
            if (model.importName == target || !model.hasAnimation || !model.hasSkeletonRepresentation) continue;
            if (ImGui::Selectable(model.importName.c_str(), state.source == model.importName)) {
                state.source = model.importName; state.clip.clear(); state.nodeMap.clear(); state.previewed = false; state.message.clear();
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::BeginCombo("Source clip", state.clip.empty() ? "Choose clip" : state.clip.c_str())) {
        for (const auto& clip : ctx.scene.animationDataList) if (clip && clip->modelName == state.source)
            if (ImGui::Selectable(clip->name.c_str(), state.clip == clip->name)) { state.clip = clip->name; state.previewed = false; state.message.clear(); }
        ImGui::EndCombo();
    }
    if (ImGui::Checkbox("Correct rest-pose / bone basis", &state.restBasis)) {
        state.previewed = false; state.message.clear(); state.translationScale = 1.f;
    }
    if (state.restBasis) {
        if (ImGui::InputFloat("Translation motion scale", &state.translationScale, 0.1f, 1.f)) {
            state.previewed = false; state.message.clear();
        }
        ImGui::TextWrapped("Matching parent chains and positive uniform rest scales required. No foot-contact IK.");
    }
    const SceneData::ImportedModelContext* from = nullptr; const SceneData::ImportedModelContext* to = nullptr;
    for (const auto& model : ctx.scene.importedModelContexts) {
        if (model.importName == state.source) from = &model;
        if (model.importName == target) to = &model;
    }
    if (manual && from && to && ImGui::TreeNode("Manual node mapping")) {
        ImGui::TextWrapped("Override node names, including parents/helpers. Auto uses original exported names.");
        ImGui::BeginChild("NodeMappingRows", ImVec2(0, 200), true);
        for (const auto& node : from->nodeHierarchy.nodes) {
            ImGui::PushID(node.uniqueName.c_str());
            const auto selected = state.nodeMap.find(node.uniqueName);
            const std::string label = selected == state.nodeMap.end() ? "Auto by name" : selected->second;
            if (ImGui::BeginCombo(node.uniqueName.c_str(), label.c_str())) {
                if (ImGui::Selectable("Auto by name", selected == state.nodeMap.end())) {
                    state.nodeMap.erase(node.uniqueName); state.previewed = false; state.message.clear();
                }
                for (const auto& candidate : to->nodeHierarchy.nodes)
                    if (ImGui::Selectable(candidate.uniqueName.c_str(), label == candidate.uniqueName)) {
                        state.nodeMap[node.uniqueName] = candidate.uniqueName; state.previewed = false; state.message.clear();
                    }
                ImGui::EndCombo();
            }
            ImGui::PopID();
        }
        ImGui::EndChild(); ImGui::TreePop();
    }
    if (!state.source.empty() && !state.clip.empty() && ImGui::Button("Check mapping")) {
        const auto r = rtapi::previewClipBinding(state.source, state.clip, target, state.report, state.nodeMap, state.restBasis ? "rest_basis" : "same_rig", state.translationScale);
        state.previewed = r.ok; state.message = r.ok ? std::string() : r.error;
    }
    if (state.previewed) {
        ImGui::Text("Mapped: %zu | Missing: %zu | Ambiguous: %zu", state.report.matches.size(), state.report.unmapped.size(), state.report.ambiguous.size());
        ImGui::Text("Hierarchy issues: %zu", state.report.hierarchy_mismatches.size());
        for (const auto& key : state.report.unmapped) ImGui::TextWrapped("Missing: %s", key.c_str());
        for (const auto& key : state.report.ambiguous) ImGui::TextWrapped("Ambiguous: %s", key.c_str());
        for (const auto& key : state.report.hierarchy_mismatches) ImGui::TextWrapped("Parent mismatch: %s", key.c_str());
        if (!state.restBasis && state.report.rest_difference_count) ImGui::TextWrapped("%d default transforms differ. Direct transfer keeps clip TRS; it does not correct a different rig/rest basis.", state.report.rest_difference_count);
        if (state.report.ready && ImGui::Button(state.restBasis ? "Bake retargeted clip to this character" : "Add same-rig clip to this character")) {
            const auto r = rtapi::bindAnimationClip(state.source, state.clip, target, {}, state.report, state.nodeMap, state.restBasis ? "rest_basis" : "same_rig", state.translationScale);
            state.message = r.ok ? "Added: " + state.report.output_clip + " (choose this clip on the target AnimGraph)" : r.error;
        }
        if (!state.report.ready) ImGui::TextWrapped("Mapping is incompatible. Check node and parent overrides; a different rig basis requires retargeting.");
    }
    if (!state.message.empty()) ImGui::TextWrapped("%s", state.message.c_str());
}
void drawClipBinding(UIContext& ctx, const std::string& target) {
    if (!ImGui::TreeNode("Add clip from imported rig")) return;
    drawClipBindingContents(ctx, target, true);
    ImGui::TextWrapped("For visual mapping, open Animation > Retarget in the bottom editor.");
    ImGui::TreePop();
}
}
