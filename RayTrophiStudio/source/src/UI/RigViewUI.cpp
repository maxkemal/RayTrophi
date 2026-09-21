#include "UI/RigViewUI.h"
#include "Animation/RigSelection.h"
#include "scene_ui.h"
#include "Api/RtApi.h"
#include "imgui.h"
#include "UI/ClipBindingUI.h"
namespace RigUI {
void drawBoneTree(UIContext& ctx, const SceneData::ImportedModelContext& model, int index) {
    if (index < 0 || static_cast<size_t>(index) >= model.skeletonNodes.size()) return;
    const auto& node = model.skeletonNodes[index];
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow;
    if (node.children.empty()) flags |= ImGuiTreeNodeFlags_Leaf;
    if (RigAuthoring::isBoneSelected(ctx.scene.rigView,model.importName,node.name)) flags |= ImGuiTreeNodeFlags_Selected;
    const std::string label = node.name + (node.weightedBone ? " [skinned]" : node.boneIndex >= 0 ? " [joint]" : " [helper]");
    const bool open = ImGui::TreeNodeEx(label.c_str(), flags);
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {const auto& io=ImGui::GetIO();rtapi::selectRigBones(model.importName,{node.name},"",io.KeyShift?"range":io.KeyCtrl?"toggle":"replace");}
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Joint index: %d\nClick: select; Ctrl: toggle; Shift: hierarchy range.", node.boneIndex);
    if (open) { for (int child : node.children) drawBoneTree(ctx, model, child); ImGui::TreePop(); }
}
void drawModelControls(UIContext& ctx, const SceneData::ImportedModelContext& model) {
    if (rtapi::renderStatus().state == rtapi::RenderJobState::Rendering) return;
    bool visible = ctx.scene.rigView.visible;
    if (ImGui::Checkbox("Skeleton overlay", &visible)) rtapi::setRigOverlayVisible(visible);
    drawClipBinding(ctx, model.importName);
    RigAuthoring::BoneView b;
    if (!RigAuthoring::selectedBone(ctx.scene, b) || b.character != model.importName) return;
    ImGui::TextWrapped("Active: %s", b.name.c_str());
    ImGui::Text("Selected joints: %llu",static_cast<unsigned long long>(RigAuthoring::selectedBones(ctx.scene).size()));
    ImGui::TextWrapped("Parent: %s", b.parent.empty() ? "(root)" : b.parent.c_str());
    ImGui::TextDisabled("Index: %d | %s | Pose: %s", b.bone_index, b.weighted ? "weighted" : "unweighted", b.pose_source.c_str());
    ImGui::Text("Position: %.3f, %.3f, %.3f", b.world.m[0][3], b.world.m[1][3], b.world.m[2][3]);
    if (ImGui::SmallButton("Clear bone selection")) rtapi::clearRigSelection();
}
}
