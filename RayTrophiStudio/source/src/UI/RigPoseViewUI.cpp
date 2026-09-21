#include "UI/RigPoseViewUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"
namespace RigUI {
void drawRigPoseViewControls(UIContext& ctx,const std::string& fixedCharacter, bool compactMode) {
    static std::string target,message;
    ImGui::PushID("RigPoseView");
    if(fixedCharacter.empty()) {
        if(target.empty())target=ctx.scene.rigView.character;
        if(compactMode) ImGui::TextDisabled("Character:");
        ImGui::SetNextItemWidth(compactMode ? -1.0f : 200.0f);
        if(ImGui::BeginCombo(compactMode ? "##PoseViewChar" : "Pose view character",target.empty()?"Choose skeleton":target.c_str())) {
            for(const auto& model:ctx.scene.importedModelContexts)if(model.hasSkeletonRepresentation)
                if(ImGui::Selectable(model.importName.c_str(),target==model.importName)){target=model.importName;message.clear();}
            ImGui::EndCombo();
        }
    }
    const auto character=fixedCharacter.empty()?target:fixedCharacter;
    if(!character.empty()) {
        std::string mode,effective;const auto result=rtapi::getRigPoseView(character,mode,effective);
        if(result.ok) {
            const bool locked=ctx.scene.rigView.edit_mode && ctx.scene.rigView.edit_character==character;
            ImGui::BeginDisabled(locked);
            if(compactMode) ImGui::TextDisabled("Pose Mode:");
            ImGui::SetNextItemWidth(compactMode ? -1.0f : 160.0f);
            if(ImGui::BeginCombo(compactMode ? "##PoseViewMode" : "Pose view",locked?"rest (Rig Edit)":mode.c_str())) {
                for(const char* option:{"animated","rest"})if(ImGui::Selectable(option,mode==option)) {
                    const auto changed=rtapi::setRigPoseView(character,option);message=changed.ok?"":changed.error;
                }
                ImGui::EndCombo();
            }
            ImGui::EndDisabled();
            if(locked)ImGui::TextDisabled("Rig Edit always shows stored rest/bind pose.");
        } else ImGui::TextWrapped("%s",result.error.c_str());
    }
    if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
    ImGui::PopID();
}
}
