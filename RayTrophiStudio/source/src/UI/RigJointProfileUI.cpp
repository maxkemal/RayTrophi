#include "UI/RigJointProfileUI.h"
#include "Api/RtApi.h"
#include "Animation/RigJointRules.h"
#include "scene_ui.h"
#include "imgui.h"
#include "ImGuizmo.h"
namespace RigUI {
void drawRigJointProfile(UIContext& ctx,const std::string& character){
 if(character.empty()||!ImGui::CollapsingHeader("Joint Motion"))return;
 static std::string owner,bone,message;static int load=-1;static uint64_t revision=0;static RigAuthoring::JointRule draft;
 nlohmann::json state;const auto result=rtapi::getRigJointProfile(character,state);if(!result.ok){ImGui::TextWrapped("%s",result.error.c_str());return;}
 bool owned=false;for(const auto& m:ctx.scene.importedModelContexts)if(m.importName==character)owned=m.authoringOwned;
 const auto selected=ctx.scene.rigView.character==character?ctx.scene.rigView.bone:std::string();
 const auto rev=state["rig_revision"].get<uint64_t>();
 if(owner!=character||bone!=selected||load!=ctx.scene.load_counter||revision!=rev){owner=character;bone=selected;load=ctx.scene.load_counter;revision=rev;message.clear();draft=RigAuthoring::JointRule();draft.bone=bone;for(const auto& row:state["profile"]["joints"])if(row["bone"]==bone){draft.type=row["type"].get<std::string>();draft.enabled=row["enabled"].get<bool>();draft.lockTranslation=row["lock_translation"].get<bool>();draft.axis=Vec3(row["axis"][0].get<float>(),row["axis"][1].get<float>(),row["axis"][2].get<float>());draft.minimum=row["minimum"].get<float>();draft.maximum=row["maximum"].get<float>();draft.swing=row["swing"].get<float>();}}
 ImGui::PushID("JointProfile");
 bool overlay=ctx.scene.rigView.joint_limits_visible,edit=ctx.scene.rigView.joint_limits_edit;
 if(ImGui::Checkbox("Show joint axes / limits",&overlay)){const auto r=rtapi::setRigJointLimitOverlay(overlay,overlay&&edit);message=r.ok?"":r.error;}
 ImGui::BeginDisabled(!overlay||!owned||bone.empty()||ctx.scene.rigView.pose.hasPreview);
 if(ImGui::Checkbox("Edit limits in viewport",&edit)){const auto r=rtapi::setRigJointLimitOverlay(overlay,edit);message=r.ok?"":r.error;}
 ImGui::EndDisabled();
 if(edit)ImGui::TextWrapped("Drag the blue/orange angle handles or green swing handle. Escape cancels; release saves one undo step. Enable the joint rule to enforce it.");
 ImGui::Text("Family: %s",state["family"].get<std::string>().c_str());
 ImGui::BeginDisabled(!owned||ImGuizmo::IsUsing()||ctx.scene.rigView.pose.hasPreview);
 if(ImGui::Button("Add anatomy suggestions")){
  nlohmann::json proposal;auto r=rtapi::suggestRigJointProfile(character,proposal);
  if(r.ok){auto profile=state["profile"];size_t added=0;for(const auto& row:proposal["profile"]["joints"]){bool exists=false;for(const auto& old:profile["joints"])exists=exists||old["bone"]==row["bone"];if(!exists){profile["joints"].push_back(row);++added;}}if(added)r=rtapi::setRigJointProfile(character,profile,rev);message=r.ok?(added?"Suggestions added. Review a joint and enable its rule.":"No new suggestions for these anatomy roles."):r.error;}
  else message=r.error;
 }
 ImGui::TextWrapped("Suggestions remain disabled. Select a joint to review its motion.");
 ImGui::BeginDisabled(bone.empty());
 ImGui::TextWrapped("Joint: %s",bone.empty()?"Select a bone":bone.c_str());
 if(ImGui::BeginCombo("Behavior",draft.type.c_str())){for(const auto* type:{"free","hinge","ball","fixed"})if(ImGui::Selectable(type,draft.type==type))draft.type=type;ImGui::EndCombo();}
 ImGui::Checkbox("Enable joint rule",&draft.enabled);
 ImGui::Checkbox("Keep joint position",&draft.lockTranslation);
 if(draft.type=="hinge"||draft.type=="ball"){
  ImGui::SetNextItemWidth(-1);ImGui::SliderFloat("Minimum angle",&draft.minimum,-180,0,"%.0f deg");
  ImGui::SetNextItemWidth(-1);ImGui::SliderFloat("Maximum angle",&draft.maximum,0,180,"%.0f deg");
  if(draft.type=="ball"){ImGui::SetNextItemWidth(-1);ImGui::SliderFloat("Swing cone",&draft.swing,0,180,"%.0f deg");}
  if(ImGui::TreeNode("Axis settings")){float axis[3]={draft.axis.x,draft.axis.y,draft.axis.z};if(ImGui::InputFloat3("Rest-local axis",axis))draft.axis=Vec3(axis[0],axis[1],axis[2]);ImGui::TextWrapped("Hinge bending / ball twisting axis, relative to this joint's rest rotation. Angles are relative to neutral.");ImGui::TreePop();}
 }
 if(ImGui::Button("Apply joint settings")){
  auto row=draft;if(row.axis.length_squared()>1e-8f)row.axis=row.axis.normalize();auto profile=state["profile"];auto encoded=RigAuthoring::serializeJointRules({row})["joints"][0];bool replaced=false;for(auto& old:profile["joints"])if(old["bone"]==bone){old=encoded;replaced=true;}if(!replaced)profile["joints"].push_back(encoded);
  const auto r=rtapi::setRigJointProfile(character,profile,rev);message=r.ok?"Joint settings saved":r.error;
 }
 if(ImGui::Button("Remove joint settings")){auto profile=state["profile"];auto& rows=profile["joints"];for(size_t i=rows.size();i>0;--i)if(rows[i-1]["bone"]==bone)rows.erase(rows.begin()+static_cast<nlohmann::json::difference_type>(i-1));const auto r=rtapi::setRigJointProfile(character,profile,rev);message=r.ok?"Joint settings removed":r.error;}
 ImGui::EndDisabled();ImGui::EndDisabled();
 if(!owned)ImGui::TextDisabled("Joint authoring requires an owned rig.");
 if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
 ImGui::PopID();
}
}
