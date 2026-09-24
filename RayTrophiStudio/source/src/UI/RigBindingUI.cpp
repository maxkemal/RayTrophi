#include "UI/RigBindingUI.h"
#include "UI/RigEnvelopeEditorUI.h"
#include "Api/RtApi.h"
#include "ImGuizmo.h"
#include "imgui.h"
#include "scene_ui.h"
namespace RigUI {
namespace {
struct Draft {
    std::string character, mesh, message;
    int load = -1;
    bool confirmed = false;
    bool envelopesLoaded = false;
    uint64_t envelopeRevision = 0;
    nlohmann::json preview, envelopePreview;
    RigAuthoring::EnvelopeWeightSettings envelopes;
} draft;
} // namespace
void drawRigBinding(UIContext &ctx, const std::string &character, const std::string &mesh,
                    bool defaultOpen) {
    const auto flags = defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None;
    if (character.empty() || !UIWidgets::CollapsingHeader("Skin binding and weights", flags))
        return;
    if (draft.character != character || draft.mesh != mesh ||
        draft.load != ctx.scene.load_counter) {
        draft = Draft{};
        draft.character = character;
        draft.mesh = mesh;
        draft.load = ctx.scene.load_counter;
    }
    nlohmann::json binding;
    auto info = rtapi::getRigMeshBinding(character, binding);
    if (!info.ok) {
        ImGui::TextWrapped("%s", info.error.c_str());
        return;
    }
    if (binding.value("bound", false)) {
        const auto revision = binding["rig_revision"].get<uint64_t>();
        if (!draft.envelopesLoaded || draft.envelopeRevision != revision) {
            draft.envelopes = {};
            if (binding.contains("envelope_settings")) {
                const auto &settings = binding["envelope_settings"];
                draft.envelopes.torsoRadius = settings["torso_radius"].get<float>();
                draft.envelopes.limbRadius = settings["limb_radius"].get<float>();
                draft.envelopes.extremityRadius = settings["extremity_radius"].get<float>();
                draft.envelopes.falloff = settings["falloff"].get<float>();
            }
            draft.envelopesLoaded = true;
            draft.envelopeRevision = revision;
        }
        ImGui::TextWrapped("Bound parts retain their names and separate geometry.");
        for (const auto &part : binding["parts"])
            ImGui::BulletText("%s%s", part["mesh"].get<std::string>().c_str(),
                              part.value("present", false) ? "" : " (missing)");
        ImGui::TextWrapped(
            "Inspect weights above. Rest/topology editing requires an unskinned skeleton.");
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.48f, 0.20f, 0.16f, 1.0f));
        if (ImGui::Button("Unbind Mesh")) {
            const auto result = rtapi::unbindRigMesh(character);
            draft.message = result.ok
                                ? "Mesh unbound; skeleton and animation clips preserved"
                                : result.error;
            draft.preview = nullptr;
            draft.envelopePreview = nullptr;
        }
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(
                "Remove skin weights and detach mesh parts. The skeleton, controls,\n"
                "authored clips and rest placement remain. Undo restores the binding.");
        }
        if (UIWidgets::CollapsingHeader("Automatic influence envelopes",
                                    ImGuiTreeNodeFlags_DefaultOpen)) {
            nlohmann::json overlayState;
            const auto overlayResult = rtapi::getRigEnvelopeOverlay(overlayState);
            bool overlayVisible =
                overlayResult.ok && overlayState.value("visible", false) &&
                overlayState.value("character", std::string()) == character;
            if (ImGui::Checkbox("Show capsule overlay", &overlayVisible)) {
                const auto result =
                    rtapi::setRigEnvelopeOverlay(character, draft.envelopes, overlayVisible);
                if (!result.ok)
                    draft.message = result.error;
            }
            ImGui::SameLine();
            if (ImGui::Button("Open Envelope Editor"))
                openRigEnvelopeEditor(character);
            bool changed = false;
            changed |= ImGui::SliderFloat("Torso radius (% height)", &draft.envelopes.torsoRadius,
                                          .02f, .5f, "%.3f");
            changed |= ImGui::SliderFloat("Limb radius (% height)", &draft.envelopes.limbRadius,
                                          .01f, .3f, "%.3f");
            changed |= ImGui::SliderFloat("Hand/foot radius (% height)",
                                          &draft.envelopes.extremityRadius, .005f, .2f, "%.3f");
            changed |=
                ImGui::SliderFloat("Envelope falloff", &draft.envelopes.falloff, .5f, 8.f, "%.2f");
            if (changed) {
                draft.envelopePreview = nullptr;
                draft.message.clear();
                if (overlayVisible) {
                    const auto result =
                        rtapi::setRigEnvelopeOverlay(character, draft.envelopes, true);
                    if (!result.ok)
                        draft.message = result.error;
                }
            }
            if (ImGui::Button("Preview envelope weights")) {
                const auto result = rtapi::previewRigEnvelopeWeights(character, draft.envelopes,
                                                                     draft.envelopePreview);
                if (result.ok) {
                    const auto overlay =
                        rtapi::setRigEnvelopeOverlay(character, draft.envelopes, true);
                    draft.message = overlay.ok
                                        ? "Envelope preview ready; cyan capsules show proposed "
                                          "boundaries and weights are unchanged"
                                        : overlay.error;
                } else {
                    draft.message = result.error;
                }
            }
            if (draft.envelopePreview.is_object()) {
                ImGui::Text("Vertices: %llu  fallback: %llu",
                            draft.envelopePreview["vertex_count"].get<unsigned long long>(),
                            draft.envelopePreview["fallback_vertices"].get<unsigned long long>());
                ImGui::SameLine();
                if (ImGui::Button("Apply envelope weights")) {
                    const auto revision = draft.envelopePreview["rig_revision"].get<uint64_t>();
                    const auto result =
                        rtapi::applyRigEnvelopeWeights(character, draft.envelopes, revision);
                    draft.message =
                        result.ok ? "Envelope weights applied (one undo step)" : result.error;
                    draft.envelopePreview = nullptr;
                }
            }
            ImGui::TextWrapped(
                "Capsule radii limit distant limb influence. Vertices outside all "
                "capsules fall back to their nearest bone so no vertex is unweighted.");
            ImGui::TextDisabled("Selected bone: amber boundary. Other bones: cyan boundary.");
        }
        if (!draft.message.empty())
            ImGui::TextWrapped("%s", draft.message.c_str());
        return;
    }
    if (mesh.empty()) {
        ImGui::TextWrapped("Select an unskinned mesh or model group above.");
        return;
    }
    ImGui::TextWrapped(
        "Align the skeleton first. Nearest bone segments give initial weights; inspect deformation "
        "afterward. Surface visibility and interior are not verified.");
    ImGui::BeginDisabled(ImGuizmo::IsUsing());
    if (ImGui::Checkbox("Axes and rest alignment checked", &draft.confirmed))
        draft.preview = nullptr;
    ImGui::BeginDisabled(!draft.confirmed);
    if (ImGui::Button("Preview initial weights")) {
        auto result = rtapi::previewRigMeshBinding(character, mesh, draft.confirmed, draft.preview);
        draft.message = result.ok ? "Preview ready; geometry is unchanged" : result.error;
    }
    ImGui::EndDisabled();
    if (draft.preview.is_object()) {
        const auto &report = draft.preview;
        ImGui::Text("Parts: %llu  Vertices: %llu", report["part_count"].get<unsigned long long>(),
                    report["vertex_count"].get<unsigned long long>());
        ImGui::Text("Weighted bones: %llu  Max influences: %llu",
                    report["weighted_bone_count"].get<unsigned long long>(),
                    report["max_influences"].get<unsigned long long>());
        ImGui::Text("Weight sums: %.6f - %.6f", report["min_weight_sum"].get<double>(),
                    report["max_weight_sum"].get<double>());
        ImGui::Text("Joints outside bounds: %llu",
                    report["outside_bounds_joints"].get<unsigned long long>());
        ImGui::BeginDisabled(!report.value("can_bind", false));
        if (ImGui::Button("Bind and apply initial weights")) {
            auto result = rtapi::bindRigMesh(character, mesh, draft.preview);
            draft.message = result.ok ? "Mesh bound (one undo step)" : result.error;
            draft.preview = nullptr;
        }
        ImGui::EndDisabled();
    }
    ImGui::EndDisabled();
    if (!draft.message.empty())
        ImGui::TextWrapped("%s", draft.message.c_str());
}
} // namespace RigUI
