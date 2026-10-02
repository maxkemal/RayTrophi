#include "UI/RigStudioUI.h"
#include "UI/RigStudioUI_Header.h"
#include "UI/RigStudioUI_MotionStrip.h"
#include "UI/RigStudioUI_MarkingMenu.h"
#include "UI/RigJointProfileUI.h"
#include "UI/RigFittingUI.h"
#include "UI/RigAnatomyUI.h"
#include "UI/RigPoseAuthoringUI.h"
#include "UI/RigIKUI.h"
#include "UI/RigMirrorUI.h"
#include "UI/ClipBindingUI.h"
#include "UI/RigBindingUI.h"
#include "UI/RigEnvelopeEditorUI.h"
#include "UI/RigWeightMapUI.h"
#include "Animation/RigStudioServices.h"
#include "scene_ui.h"
#include "scene_ui_animgraph.hpp"
#include "imgui.h"

namespace RayTrophi {

void RigStudioUI::draw(UIContext& ctx) {
    RigStudioServices::syncContextWithRigView(ctx.scene);
    auto& studioCtx = RigStudioServices::getContext(ctx.scene);

    if (!studioCtx.windowOpen) {
        return;
    }

    ImGui::SetNextWindowSize(ImVec2(760, 800), ImGuiCond_FirstUseEver);

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 6));

    if (ImGui::Begin("Rig & Character Studio Workspace###RigStudioWorkspace", &studioCtx.windowOpen, ImGuiWindowFlags_NoCollapse)) {
        // Studio Context Header
        RigStudioUI_Header::draw(ctx);
        ImGui::Spacing();

        const std::string character = studioCtx.characterId;
        const int reqTab = studioCtx.requestedTab;
        studioCtx.requestedTab = -1; // Consume one-shot programmatic switch request

        if (ImGui::BeginTabBar("RigStudioTabsMain", ImGuiTabBarFlags_Reorderable | ImGuiTabBarFlags_TabListPopupButton)) {
            
            // -----------------------------------------------------------------
            // Tab 0: Animate & Motion
            // -----------------------------------------------------------------
            ImGuiTabItemFlags tab0Flags = (reqTab == 0) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("  Animate & Motion  ", nullptr, tab0Flags)) {
                studioCtx.activeTab = 0;
                studioCtx.mode = RigStudioMode::Animate;

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.3f, 0.75f, 0.95f, 1.0f), "Locomotion Motion Strip");
                ImGui::Separator();
                RigStudioUI_MotionStrip::draw(ctx);

                ImGui::Spacing();
                ImGui::Spacing();

                if (!character.empty()) {
                    if (ImGui::CollapsingHeader("Clip Retargeting & Track Assignment", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawClipBinding(ctx, character);
                    }
                } else {
                    ImGui::TextDisabled("Select an active character to bind and retarget animation clips.");
                }

                ImGui::EndTabItem();
            }

            // -----------------------------------------------------------------
            // Tab 1: Rest & Alignment
            // -----------------------------------------------------------------
            ImGuiTabItemFlags tab1Flags = (reqTab == 1) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("  Rest & Alignment  ", nullptr, tab1Flags)) {
                studioCtx.activeTab = 1;
                studioCtx.mode = RigStudioMode::Rest;

                ImGui::Spacing();
                if (!character.empty()) {
                    ImGui::TextColored(ImVec4(0.3f, 0.75f, 0.95f, 1.0f), "Skeleton Rest Pose & Mesh Alignment");
                    ImGui::Separator();

                    RigUI::drawRigJointProfile(ctx, character);
                    RigUI::drawRigFitting(ctx, character, "");

                    if (ImGui::CollapsingHeader("Landmark Alignment Canvas", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawRigAlignmentInline(ctx);
                    }

                    RigUI::drawRigAnatomy(ctx, character);
                } else {
                    ImGui::TextDisabled("Select a character from the top header to configure rest skeleton alignment.");
                }

                ImGui::EndTabItem();
            }

            // -----------------------------------------------------------------
            // Tab 2: Pose & IK
            // -----------------------------------------------------------------
            ImGuiTabItemFlags tab2Flags = (reqTab == 2) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("  Pose & IK  ", nullptr, tab2Flags)) {
                studioCtx.activeTab = 2;
                studioCtx.mode = RigStudioMode::Pose;

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.3f, 0.75f, 0.95f, 1.0f), "Pose Authoring & IK Solvers");
                ImGui::Separator();

                RigUI::drawRigPoseAuthoring(ctx);

                if (!character.empty()) {
                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader("Inverse Kinematics (IK) Solvers", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawRigIKControls(ctx, character);
                    }
                    if (ImGui::CollapsingHeader("Pose Mirroring & Symmetry", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawRigMirror(ctx, character);
                    }
                }

                ImGui::EndTabItem();
            }

            // -----------------------------------------------------------------
            // Tab 3: Skin & Envelope
            // -----------------------------------------------------------------
            ImGuiTabItemFlags tab3Flags = (reqTab == 3) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("  Skin & Envelope  ", nullptr, tab3Flags)) {
                studioCtx.activeTab = 3;
                studioCtx.mode = RigStudioMode::Skin;

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.3f, 0.75f, 0.95f, 1.0f), "Mesh Skin Binding & Envelope Weights");
                ImGui::Separator();

                RigUI::drawRigWeightMapControls(ctx);

                if (!character.empty()) {
                    ImGui::Spacing();
                    if (ImGui::CollapsingHeader("Rig Skinning & Influence Weights", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawRigBinding(ctx, character, "");
                    }
                    if (ImGui::CollapsingHeader("Bone Envelope Deformers", ImGuiTreeNodeFlags_DefaultOpen)) {
                        RigUI::drawRigEnvelopeEditorContent(ctx);
                    }
                }

                ImGui::EndTabItem();
            }

            // -----------------------------------------------------------------
            // Tab 4: Anim Graph
            // -----------------------------------------------------------------
            ImGuiTabItemFlags tab4Flags = (reqTab == 4) ? ImGuiTabItemFlags_SetSelected : 0;
            if (ImGui::BeginTabItem("  Anim Graph  ", nullptr, tab4Flags)) {
                studioCtx.activeTab = 4;

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.3f, 0.75f, 0.95f, 1.0f), "Animation Graph & Retarget State Machine");
                ImGui::Separator();

                drawAnimationGraphPanel(ctx);

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }

        // RMB Radial Marking Menu in Viewport when studio open
        RigStudioUI_MarkingMenu::draw(ctx);
    }
    ImGui::End();
    ImGui::PopStyleVar(2);
}

} // namespace RayTrophi

