#include "UI/RigEnvelopeEditorUI.h"

#include "Animation/RigEnvelopeWeights.h"
#include "Api/RtApi.h"
#include "imgui.h"
#include "scene_ui.h"
#include <algorithm>
#include <cmath>

namespace RigUI {
namespace {
struct EditorState {
    bool open = false;
    int load = -1;
    int drag = -1;
    std::string character;
    std::string bone;
    std::string message;
    uint64_t revision = 0;
    RigAuthoring::EnvelopeBoneProfile profile;
} editor;

void loadProfile(UIContext &ctx, const std::string &bone) {
    nlohmann::json value;
    const auto result = rtapi::getRigBoneEnvelope(editor.character, bone, value);
    editor.drag = -1;
    editor.message.clear();
    if (!result.ok) {
        editor.bone.clear();
        editor.message = result.error;
        return;
    }
    editor.bone = bone;
    editor.revision = value["rig_revision"].get<uint64_t>();
    editor.profile.bone = bone;
    editor.profile.startRadius = value["start_radius"].get<float>();
    editor.profile.endRadius = value["end_radius"].get<float>();
    editor.profile.startExtension = value["start_extension"].get<float>();
    editor.profile.endExtension = value["end_extension"].get<float>();
    editor.profile.falloff = value["falloff"].get<float>();
    editor.load = ctx.scene.load_counter;
}

void drawCanvas() {
    const ImVec2 size((std::max)(320.f, ImGui::GetContentRegionAvail().x), 230.f);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("EnvelopeProfileCanvas", size, ImGuiButtonFlags_MouseButtonLeft);
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 center(origin.x + size.x * .5f, origin.y + size.y * .55f);
    const float baseLength = size.x * .42f;
    const float radiusScale = size.y * .72f;
    const float baseStart = center.x - baseLength * .5f;
    const float baseEnd = center.x + baseLength * .5f;
    const float startX = baseStart - editor.profile.startExtension * baseLength;
    const float endX = baseEnd + editor.profile.endExtension * baseLength;
    const float startR = editor.profile.startRadius * radiusScale;
    const float endR = editor.profile.endRadius * radiusScale;
    const ImVec2 handles[2] = {{startX, center.y - startR}, {endX, center.y - endR}};
    auto *draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y),
                        IM_COL32(20, 24, 31, 255), 8.f);
    draw->AddLine(ImVec2(baseStart, center.y), ImVec2(baseEnd, center.y),
                  IM_COL32(145, 155, 170, 150), 2.f);
    const ImVec2 p0(startX, center.y - startR), p1(endX, center.y - endR);
    const ImVec2 p2(endX, center.y + endR), p3(startX, center.y + startR);
    draw->AddQuadFilled(p0, p1, p2, p3, IM_COL32(65, 205, 235, 35));
    draw->AddLine(p0, p1, IM_COL32(75, 220, 245, 230), 2.f);
    draw->AddLine(p3, p2, IM_COL32(75, 220, 245, 230), 2.f);
    draw->AddCircle(ImVec2(startX, center.y), startR, IM_COL32(75, 220, 245, 230), 40, 2.f);
    draw->AddCircle(ImVec2(endX, center.y), endR, IM_COL32(75, 220, 245, 230), 40, 2.f);
    draw->AddText(ImVec2(origin.x + 12.f, origin.y + 10.f), IM_COL32(205, 215, 225, 220),
                  "Drag either upper handle: horizontal = extension, vertical = radius");
    for (int index = 0; index < 2; ++index)
        draw->AddCircleFilled(handles[index], 7.f,
                              editor.drag == index ? IM_COL32(255, 176, 55, 255)
                                                   : IM_COL32(100, 235, 255, 255));
    auto &io = ImGui::GetIO();
    if (hovered && editor.drag < 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        for (int index = 0; index < 2; ++index) {
            const float dx = io.MousePos.x - handles[index].x;
            const float dy = io.MousePos.y - handles[index].y;
            if (dx * dx + dy * dy <= 144.f) {
                editor.drag = index;
                break;
            }
        }
    }
    if (editor.drag >= 0 && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const float radius = std::clamp((center.y - io.MousePos.y) / radiusScale, .005f, .5f);
        if (editor.drag == 0) {
            editor.profile.startRadius = radius;
            editor.profile.startExtension =
                std::clamp((baseStart - io.MousePos.x) / baseLength, 0.f, .75f);
        } else {
            editor.profile.endRadius = radius;
            editor.profile.endExtension =
                std::clamp((io.MousePos.x - baseEnd) / baseLength, 0.f, .75f);
        }
    } else {
        editor.drag = -1;
    }
}
} // namespace

void openRigEnvelopeEditor(const std::string &character) {
    editor.open = true;
    if (editor.character != character) {
        editor.character = character;
        editor.bone.clear();
    }
}

void drawRigEnvelopeEditor(UIContext &ctx) {
    if (!editor.open)
        return;
    ImGui::SetNextWindowSize(ImVec2(660, 520), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Envelope Editor", &editor.open)) {
        ImGui::End();
        return;
    }
    const auto selectedCharacter = ctx.scene.rigView.character;
    const auto selectedBone = ctx.scene.rigView.bone;
    if (selectedCharacter == editor.character && !selectedBone.empty() &&
        (editor.bone != selectedBone || editor.load != ctx.scene.load_counter)) {
        loadProfile(ctx, selectedBone);
    }
    ImGui::Text("Character: %s", editor.character.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Select a bone in the viewport to change the profile.");
    if (editor.bone.empty()) {
        ImGui::TextWrapped("Select a bone that owns a visible segment.");
        if (!editor.message.empty())
            ImGui::TextWrapped("%s", editor.message.c_str());
        ImGui::End();
        return;
    }
    ImGui::SeparatorText(editor.bone.c_str());
    drawCanvas();
    ImGui::SetNextItemWidth(180.f);
    ImGui::DragFloat("Start radius", &editor.profile.startRadius, .001f, .005f, .5f, "%.3f H");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::DragFloat("End radius", &editor.profile.endRadius, .001f, .005f, .5f, "%.3f H");
    ImGui::SetNextItemWidth(180.f);
    ImGui::DragFloat("Start extension", &editor.profile.startExtension, .005f, 0.f, .75f,
                     "%.3f bone");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.f);
    ImGui::DragFloat("End extension", &editor.profile.endExtension, .005f, 0.f, .75f,
                     "%.3f bone");
    ImGui::SetNextItemWidth(180.f);
    ImGui::DragFloat("Falloff", &editor.profile.falloff, .05f, .5f, 8.f, "%.2f");
    if (ImGui::Button("Apply profile and rebuild weights")) {
        const auto result =
            rtapi::applyRigBoneEnvelope(editor.character, editor.profile, editor.revision);
        if (result.ok) {
            editor.message = "Profile applied; weights rebuilt in one undo step.";
            rtapi::setRigWeightMapVisible(true);
            nlohmann::json binding;
            if (rtapi::getRigMeshBinding(editor.character, binding).ok)
                editor.revision = binding["rig_revision"].get<uint64_t>();
        } else {
            editor.message = result.error;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reload"))
        loadProfile(ctx, editor.bone);
    if (!editor.message.empty())
        ImGui::TextWrapped("%s", editor.message.c_str());
    ImGui::End();
}
} // namespace RigUI
