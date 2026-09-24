#include "UI/RigAnatomyUI.h"
#include "Api/RtApi.h"
#include "scene_ui.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <cstdio>
#include <vector>
namespace RigUI {
void drawRigAnatomy(UIContext& ctx,const std::string& character) {
    if(character.empty()||!UIWidgets::CollapsingHeader("Rig anatomy"))return;
    ImGui::PushID("RigAnatomy");
    static std::string owner,rightBone,message;static int load=-1,roleIndex=-1,pairIndex=-1,chainIndex=-1;
    static bool editMetadata=false;static char roleName[129]="left_arm.upper",chainName[129]="left_arm";
    static std::vector<std::string> draft;
    if(owner!=character||load!=ctx.scene.load_counter){owner=character;load=ctx.scene.load_counter;rightBone.clear();draft.clear();message.clear();roleIndex=pairIndex=chainIndex=-1;editMetadata=false;}
    nlohmann::json value;auto result=rtapi::getRigAnatomy(character,value);
    if(!result.ok){ImGui::TextWrapped("%s",result.error.c_str());ImGui::PopID();return;}
    std::vector<RigAuthoring::BoneView> bones;result=rtapi::listRigBones(character,bones);
    if(!result.ok){ImGui::TextWrapped("%s",result.error.c_str());ImGui::PopID();return;}
    auto shortName=[&](const std::string& name){const auto prefix=character+"_";return name.find(prefix)==0?name.substr(prefix.size()):name;};
    const std::string selected=ctx.scene.rigView.character==character?ctx.scene.rigView.bone:"";
    const auto family=value["family"].get<std::string>();
    ImGui::Text("%s | %zu roles | %zu pairs | %zu chains",family.c_str(),value["roles"].size(),value["symmetry"].size(),value["chains"].size());
    ImGui::TextWrapped("Templates supply these records. Normal fitting does not require editing them.");
    ImGui::Checkbox("Edit anatomy metadata",&editMetadata);
    bool apply=false;auto edited=value;
    ImGui::BeginDisabled(ImGuizmo::IsUsing());
    if(editMetadata) {
        const char* families[]={"custom","humanoid","quadruped","insect","avian"};
        if(ImGui::BeginCombo("Family",family.c_str())){for(const char* option:families)if(ImGui::Selectable(option,family==option)){edited["family"]=option;apply=true;}ImGui::EndCombo();}
        ImGui::TextWrapped("Selected bone: %s",selected.empty()?"Choose a bone":shortName(selected).c_str());
    }
    if(UIWidgets::CollapsingHeader("Roles")) {
        const auto& roles=value["roles"];if(roleIndex>=static_cast<int>(roles.size()))roleIndex=-1;
        if(ImGui::BeginListBox("##Roles",ImVec2(-1,125))) {
            for(size_t i=0;i<roles.size();++i){const auto label=roles[i]["role"].get<std::string>()+" = "+shortName(roles[i]["bone"].get<std::string>());if(ImGui::Selectable(label.c_str(),roleIndex==static_cast<int>(i)))roleIndex=static_cast<int>(i);}
            ImGui::EndListBox();
        }
        if(editMetadata) {
            ImGui::BeginDisabled(roleIndex<0);if(ImGui::Button("Remove selected role")){edited["roles"].erase(edited["roles"].begin()+roleIndex);roleIndex=-1;apply=true;}ImGui::EndDisabled();
            if(ImGui::TreeNode("Assign role")) {
                ImGui::InputText("Role identifier",roleName,sizeof(roleName));ImGui::BeginDisabled(selected.empty());
                if(ImGui::Button("Assign to selected bone")){auto& rows=edited["roles"];for(size_t i=rows.size();i>0;--i)if(rows[i-1]["role"]==roleName)rows.erase(rows.begin()+static_cast<nlohmann::json::difference_type>(i-1));rows.push_back({{"role",roleName},{"bone",selected}});apply=true;}
                ImGui::EndDisabled();ImGui::TreePop();
            }
        }
    }
    if(UIWidgets::CollapsingHeader("Symmetry pairs")) {
        const auto& pairs=value["symmetry"];if(pairIndex>=static_cast<int>(pairs.size()))pairIndex=-1;
        if(ImGui::BeginListBox("##Pairs",ImVec2(-1,125))) {
            for(size_t i=0;i<pairs.size();++i){const auto label=shortName(pairs[i]["left"].get<std::string>())+" <-> "+shortName(pairs[i]["right"].get<std::string>());if(ImGui::Selectable(label.c_str(),pairIndex==static_cast<int>(i)))pairIndex=static_cast<int>(i);}
            ImGui::EndListBox();
        }
        if(editMetadata) {
            ImGui::BeginDisabled(pairIndex<0);if(ImGui::Button("Remove selected pair")){edited["symmetry"].erase(edited["symmetry"].begin()+pairIndex);pairIndex=-1;apply=true;}ImGui::EndDisabled();
            if(ImGui::TreeNode("Add symmetry pair")) {
                if(ImGui::BeginCombo("Counterpart",rightBone.empty()?"Choose bone":shortName(rightBone).c_str())){for(const auto& b:bones)if(ImGui::Selectable(shortName(b.name).c_str(),rightBone==b.name))rightBone=b.name;ImGui::EndCombo();}
                ImGui::BeginDisabled(selected.empty()||rightBone.empty());if(ImGui::Button("Pair selected bone")){edited["symmetry"].push_back({{"left",selected},{"right",rightBone}});apply=true;}ImGui::EndDisabled();ImGui::TreePop();
            }
        }
    }
    if(UIWidgets::CollapsingHeader("Limb chains")) {
        const auto& chains=value["chains"];if(chainIndex>=static_cast<int>(chains.size()))chainIndex=-1;
        if(ImGui::BeginListBox("##Chains",ImVec2(-1,125))) {
            for(size_t i=0;i<chains.size();++i){const auto label=chains[i]["name"].get<std::string>()+" ("+std::to_string(chains[i]["bones"].size())+" joints)";if(ImGui::Selectable(label.c_str(),chainIndex==static_cast<int>(i)))chainIndex=static_cast<int>(i);}
            ImGui::EndListBox();
        }
        if(chainIndex>=0){std::string path;for(const auto& b:chains[chainIndex]["bones"]){if(!path.empty())path+=" -> ";path+=shortName(b.get<std::string>());}ImGui::TextWrapped("%s",path.c_str());}
        if(editMetadata) {
            ImGui::BeginDisabled(chainIndex<0);if(ImGui::Button("Remove selected chain")){edited["chains"].erase(edited["chains"].begin()+chainIndex);chainIndex=-1;apply=true;}ImGui::EndDisabled();
            if(ImGui::TreeNode("Build chain")) {
                ImGui::InputText("Chain identifier",chainName,sizeof(chainName));ImGui::BeginDisabled(selected.empty());if(ImGui::Button("Append selected bone"))draft.push_back(selected);ImGui::EndDisabled();
                std::string path;for(const auto& b:draft){if(!path.empty())path+=" -> ";path+=shortName(b);}ImGui::TextWrapped("%s",path.empty()?"Select joints from base to tip":path.c_str());
                if(ImGui::Button("Clear draft"))draft.clear();ImGui::SameLine();ImGui::BeginDisabled(draft.size()<2);
                if(ImGui::Button("Save chain")){edited["chains"].push_back({{"name",chainName},{"bones",draft}});apply=true;}ImGui::EndDisabled();ImGui::TreePop();
            }
        }
    }
    ImGui::EndDisabled();
    if(apply){auto saved=rtapi::setRigAnatomy(character,edited);message=saved.ok?"Anatomy saved":saved.error;}
    if(!message.empty())ImGui::TextWrapped("%s",message.c_str());ImGui::PopID();
}
}
