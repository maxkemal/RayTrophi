#include "UI/RigMirrorUI.h"
#include "Api/RtApi.h"
#include "Animation/RigSelection.h"
#include "Animation/RigEditing.h"
#include "scene_ui.h"
#include "imgui.h"
#include "ImGuizmo.h"
namespace RigUI {
void drawRigMirror(UIContext& ctx,const std::string& character) {
    if(character.empty() || !UIWidgets::CollapsingHeader("Mirror rest / creation"))return;
    static std::string owner,message;static int load=-1,axis=0,side=0;static float offset=0;static char name[129]="MirroredJoint";
    if(owner!=character || load!=ctx.scene.load_counter){owner=character;load=ctx.scene.load_counter;axis=side=0;offset=0;message.clear();}
    const char* axes[]={"x","y","z"};ImGui::SetNextItemWidth(80);ImGui::Combo("Rig axis",&axis,axes,3);
    ImGui::SetNextItemWidth(150);ImGui::InputFloat("Plane offset (rig space)",&offset);
    std::string error;const bool editable=RigAuthoring::canEditRig(ctx.scene,character,error);
    if(!editable)ImGui::TextWrapped("%s",error.c_str());
    uint64_t revision=0;for(const auto& m:ctx.scene.importedModelContexts)if(m.importName==character)revision=m.rigRevision;
    const auto selected=RigAuthoring::selectedBones(ctx.scene);
    ImGui::BeginDisabled(!editable || ImGuizmo::IsUsing());
    auto copy=[&](const std::vector<std::string>& bones,const char* direction){auto r=rtapi::mirrorRigRest(character,bones,direction,axes[axis],offset,revision);message=r.ok?"Mirrored rest applied (one undo step)":r.error;};
    ImGui::BeginDisabled(selected.empty());if(ImGui::Button("Mirror selected"))copy(selected,"selected");ImGui::EndDisabled();
    if(ImGui::Button("All Left -> Right"))copy({},"left_to_right");ImGui::SameLine();
    if(ImGui::Button("All Right -> Left"))copy({},"right_to_left");
    ImGui::TextWrapped("Uses anatomy symmetry pairs. Selected mode requires one source per pair. Unpaired center bones are not mirrored.");
    ImGui::InputText("New opposite bone name",name,sizeof(name));
    const char* sides[]={"Left","Right"};ImGui::SetNextItemWidth(100);ImGui::Combo("Source side",&side,sides,2);
    ImGui::BeginDisabled(selected.size()!=1);
    if(ImGui::Button("Create opposite bone")) {
        auto r=rtapi::createMirroredRigBone(character,selected.front(),name,side==0?"left":"right",axes[axis],offset,revision);
        message=r.ok?"Opposite bone and symmetry pair created (one undo step)":r.error;
    }
    ImGui::EndDisabled();ImGui::EndDisabled();
    ImGui::TextWrapped("Creation requires one unpaired non-root source. Its parent maps to the paired parent, or stays shared when unpaired. Roles/chains are configured separately.");
    if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
}
}
