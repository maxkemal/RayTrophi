#include "UI/RigJointProfileUI.h"
#include "UI/RigEditingUI.h"
#include "UI/RigAnatomyUI.h"
#include "UI/RigPoseViewUI.h"
#include "UI/RigFittingUI.h"
#include "UI/RigBindingUI.h"
#include "UI/RigMirrorUI.h"
#include "UI/RigPoseAuthoringUI.h"
#include "UI/RigWeightMapUI.h"
#include "Animation/RigSelection.h"
#include "Api/RtApi.h"
#include "Animation/RigTemplates.h"
#include "scene_ui.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "TriangleMesh.h"
#include <cstdint>
#include <algorithm>
#include <cstdio>

namespace RigUI {
void drawRigEditing(UIContext& ctx) {
    if (ctx.scene.rigView.pose.active) {
        drawRigPoseAuthoring(ctx);
        drawRigJointProfile(ctx, ctx.scene.rigView.pose.character);
        ImGui::PushID("PoseSecondary");
        drawRigBinding(ctx, ctx.scene.rigView.pose.character, "", false);
        ImGui::PopID();
        return;
    }
    drawRigPoseAuthoring(ctx);
    static char rigName[129]="Rig",jointName[129]="Joint1";
    static char renameName[129]="",copyName[129]="EditableRig";
    static std::string copySource;
    static std::string topologyKey,newParent;
    static uint64_t topologyRevision=0;
    static bool automaticName=true,automaticRigName=true;
    static char rigSeed[129]="Rig";
    static int preset=0;static float height=1.8f;
    static std::string character,message,editingKey;
    static uint64_t editingRevision=0;
    static Vec3 addPosition(0,.3f,0),position,rotation;
    for(const auto& model:ctx.scene.importedModelContexts)
        if(model.authoringOwned && model.importName==ctx.scene.rigView.character)character=model.importName;
    const auto selected=RigAuthoring::selectedBones(ctx.scene);
    ImGui::Text("Selected bones: %llu",static_cast<unsigned long long>(selected.size()));
    ImGui::TextDisabled("Ctrl: toggle, Shift: range. Rig Edit: Shift-drag blank space for box select.");
    if(ImGui::RadioButton("Active bone pivot",ctx.scene.rigView.selection_pivot=="active"))rtapi::setRigSelectionPivot("active");
    ImGui::SameLine();if(ImGui::RadioButton("Selection center",ctx.scene.rigView.selection_pivot=="center"))rtapi::setRigSelectionPivot("center");
    drawRigWeightMapControls(ctx);
    const bool editing=ctx.scene.rigView.edit_mode;
    ImGui::Text("Viewport mode: %s",editing?"Rig Edit (bone selection locked)":"Scene");
    if(editing) {
        ImGui::TextWrapped("Editing: %s",ctx.scene.rigView.edit_character.c_str());
        if(ImGui::Button("Return to Scene")){auto r=rtapi::setRigMode("scene");message=r.ok?"Scene mode":r.error;}
    } else if(!character.empty() && ImGui::Button("Enter Rig Edit")) {
        auto r=rtapi::setRigMode("edit",character);message=r.ok?"Rig Edit: meshes cannot be picked":r.error;
    }
    if(ctx.scene.rigView.edit_mode || !character.empty()) {
        ImGui::TextDisabled(editing?"Drag edits selected bone rest pose.":"Drag moves/rotates/scales the entire selected rig (G/R/S).");
        ImGui::TextDisabled("Escape cancels; one undo step on release.");
    }
    ImGui::SeparatorText("Rig setup");
    ImGui::TextWrapped("Create a meshless rig, then add joints and edit local rest position/rotation.");
    bool overlay=ctx.scene.rigView.visible;
    if(ImGui::Checkbox("Skeleton overlay",&overlay))rtapi::setRigOverlayVisible(overlay);
    bool validRigName=true;
    ImGui::Checkbox("Automatic rig names",&automaticRigName);
    if(automaticRigName) {
        ImGui::InputText("Rig name seed",rigSeed,sizeof(rigSeed));
        if(!editing){std::string next;auto r=rtapi::getNextRigName(rigSeed,next);validRigName=r.ok;if(r.ok)std::snprintf(rigName,sizeof(rigName),"%s",next.c_str());else ImGui::TextWrapped("%s",r.error.c_str());}
        ImGui::Text("Next rig: %s",rigName);
    } else {ImGui::SetNextItemWidth(180);ImGui::InputText("Rig name",rigName,sizeof(rigName));}
    if(editing)ImGui::TextDisabled("Return to Scene to create/select another rig.");
    const auto& templates=RigAuthoring::rigTemplateCatalogue();
    if(preset<0 || static_cast<size_t>(preset)>=templates.size())preset=0;
    ImGui::SetNextItemWidth(180);
    if(ImGui::BeginCombo("Template",templates[preset].label.c_str())) {
        for(size_t i=0;i<templates.size();++i)if(ImGui::Selectable(templates[i].label.c_str(),preset==static_cast<int>(i))) {
            preset=static_cast<int>(i);height=templates[i].default_height;
        }
        ImGui::EndCombo();
    }
    if(templates[preset].id!="root"){ImGui::SetNextItemWidth(180);ImGui::InputFloat("Rig height",&height,.1f,1.f);}
    ImGui::Text("%zu joints | %s",templates[preset].joint_count,templates[preset].family.c_str());
    ImGui::BeginDisabled(editing || !validRigName || ImGuizmo::IsUsing());
    if(ImGui::Button("Create rig")) {
        const auto r=rtapi::createRig(rigName,templates[preset].id,height);
        message=r.ok?"Created: "+std::string(rigName):r.error;
        if(r.ok){character=rigName;ctx.selection.clearSelection();ctx.selection.transform_mode=TransformMode::Translate;}
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText("Mesh preparation");
    static std::string targetMesh;static nlohmann::json preflight;static int preflightLoad=-1;
    if(preflightLoad!=ctx.scene.load_counter){targetMesh.clear();preflight=nullptr;preflightLoad=ctx.scene.load_counter;}
    if(ImGui::BeginCombo("Target mesh",targetMesh.empty()?"Choose mesh":targetMesh.c_str())) {
        nlohmann::json targets;auto listed=rtapi::listRigFitTargets(targets);
        if(listed.ok)for(const auto& item:targets) {
            const auto id=item["target"].get<std::string>();
            if(ImGui::Selectable(item["label"].get<std::string>().c_str(),targetMesh==id)){targetMesh=id;preflight=nullptr;}
        }
        ImGui::EndCombo();
    }
    ImGui::TextWrapped("Character groups include all imported parts. Choose the body mesh if accessories distort alignment bounds.");
    ImGui::BeginDisabled(targetMesh.empty() || ImGuizmo::IsUsing());
    if(ImGui::Button("Inspect mesh for fitting")) {
        auto r=rtapi::preflightRigMesh(targetMesh,preflight);message=r.ok?"Mesh diagnostics updated":r.error;
    }
    ImGui::EndDisabled();
    if(preflight.is_object()) {
        ImGui::Text("Vertices: %llu | Triangles: %llu",preflight["vertex_count"].get<unsigned long long>(),preflight["triangle_count"].get<unsigned long long>());
        ImGui::TextWrapped("%s",preflight["can_start_landmarks"].get<bool>()?"Geometry can start landmark setup; axes, pose and interior still need confirmation.":"Geometry or existing skin needs attention before landmark setup.");
        ImGui::Text("Nonfinite: %llu | Invalid: %llu | Degenerate: %llu",preflight["nonfinite_vertices"].get<unsigned long long>(),preflight["invalid_triangles"].get<unsigned long long>(),preflight["degenerate_triangles"].get<unsigned long long>());
        if(preflight["bounds"].is_object()) {
            const auto e=preflight["bounds"]["extent"];ImGui::Text("Extent: %.3f / %.3f / %.3f",e[0].get<float>(),e[1].get<float>(),e[2].get<float>());
        }
        if(preflight.contains("part_count"))ImGui::Text("Target parts: %llu",preflight["part_count"].get<unsigned long long>());
        if(preflight.contains("blockers"))for(const auto& reason:preflight["blockers"])ImGui::TextWrapped("Needs attention: %s",reason.get<std::string>().c_str());
        ImGui::TextDisabled("Snapshot report; inspect again after changing mesh. Fitting/binding is pending.");
    }
    if(ImGui::CollapsingHeader("Skin weight diagnostics")) {
        static nlohmann::json stats,vertexReport;static int weightVertex=0;static std::string inspected;static int weightLoad=-1;
        if(inspected!=targetMesh || weightLoad!=ctx.scene.load_counter){stats=nullptr;vertexReport=nullptr;weightVertex=0;inspected=targetMesh;weightLoad=ctx.scene.load_counter;}
        ImGui::TextDisabled("Choose an individual mesh; character groups are for fitting.");
        if(!targetMesh.empty() && ImGui::Button("Inspect skin weights")){auto r=rtapi::getRigWeightStats(targetMesh,stats);message=r.ok?"Weight snapshot refreshed":r.error;}
        if(stats.is_object()){
            ImGui::Text("Max influences: %llu | Unweighted: %llu",stats["max_influences"].get<unsigned long long>(),stats["unweighted_vertices"].get<unsigned long long>());
            ImGui::TextWrapped("%s",stats["contract_valid"].get<bool>()?"Stored influence contract is valid.":"Stored influences need attention.");
            if(stats["min_weight_sum"].is_number())ImGui::Text("Weight sums: %.6f to %.6f",stats["min_weight_sum"].get<double>(),stats["max_weight_sum"].get<double>());
            else ImGui::TextDisabled("No nonempty weight rows.");
            ImGui::Text("Invalid entries: %llu | Duplicate IDs: %llu",stats["invalid_entries"].get<unsigned long long>(),stats["duplicate_entries"].get<unsigned long long>());
            ImGui::Text("Over four: %llu | Unnormalized: %llu | Unsorted: %llu",stats["over_limit_vertices"].get<unsigned long long>(),stats["unnormalized_vertices"].get<unsigned long long>(),stats["unsorted_vertices"].get<unsigned long long>());
            ImGui::Text("Extra weight rows: %llu",stats["extra_weight_rows"].get<unsigned long long>());
            ImGui::Text("Unknown bones: %llu | Ambiguous: %llu | Foreign: %llu",stats["unknown_bone_entries"].get<unsigned long long>(),stats["ambiguous_bone_entries"].get<unsigned long long>(),stats["foreign_bone_entries"].get<unsigned long long>());
            if(stats["character"].is_string())ImGui::TextWrapped("Character: %s",stats["character"].get<std::string>().c_str());
            ImGui::TextWrapped("%s",stats["bone_indices_verified"].get<bool>()?"Bone indices belong to this character.":"Bone ownership or indices could not be verified.");
        }
        ImGui::SetNextItemWidth(150);
        if(ImGui::InputInt("Flat vertex index",&weightVertex)){weightVertex=std::max(0,weightVertex);vertexReport=nullptr;}
        if(!targetMesh.empty() && ImGui::Button("Inspect vertex weights")) {
            auto r=rtapi::getRigVertexWeights(targetMesh,static_cast<uint64_t>(weightVertex),vertexReport);
            message=r.ok?"Vertex weight snapshot refreshed":r.error;
        }
        if(vertexReport.is_object()) {
            ImGui::Text("Influences: %llu | Sum: %.6f",vertexReport["influence_count"].get<unsigned long long>(),vertexReport["weight_sum"].get<double>());
            if(vertexReport["unweighted"].get<bool>())ImGui::TextDisabled("This vertex has no stored influences.");
            if(ImGui::BeginTable("VertexWeightRows",4,ImGuiTableFlags_BordersInnerV|ImGuiTableFlags_RowBg|ImGuiTableFlags_ScrollY,ImVec2(0,140))) {
                ImGui::TableSetupColumn("Index");ImGui::TableSetupColumn("Bone");ImGui::TableSetupColumn("Weight");ImGui::TableSetupColumn("Ownership");ImGui::TableHeadersRow();
                for(const auto& influence:vertexReport["influences"]) {
                    ImGui::TableNextRow();ImGui::TableNextColumn();ImGui::Text("%d",influence["bone_index"].get<int>());
                    ImGui::TableNextColumn();ImGui::TextUnformatted(influence["bone"].is_string()?influence["bone"].get<std::string>().c_str():"Unknown or ambiguous");
                    ImGui::TableNextColumn();if(influence["weight"].is_number())ImGui::Text("%.6f",influence["weight"].get<double>());else ImGui::TextUnformatted("Nonfinite");
                    ImGui::TableNextColumn();ImGui::TextUnformatted(influence["belongs_to_character"].is_boolean()?(influence["belongs_to_character"].get<bool>()?"This character":"Foreign bone"):"Unverified");
                }
                ImGui::EndTable();
            }
        }
        ImGui::TextDisabled("Read-only snapshots. Inspect again after mesh or rig edits.");
    }
    ImGui::SeparatorText("Skeleton source and topology");
    ImGui::TextDisabled("Create from a template above or copy an eligible imported skeleton.");
    ImGui::BeginDisabled(editing || ImGuizmo::IsUsing());
    if(copySource.empty())for(const auto& model:ctx.scene.importedModelContexts)
        if(!model.authoringOwned && model.importName==ctx.scene.rigView.character)copySource=model.importName;
    if(ImGui::BeginCombo("Source skeleton",copySource.empty()?"Choose meshless skeleton":copySource.c_str())) {
        for(const auto& model:ctx.scene.importedModelContexts)
            if(model.members.empty() && !model.weightedBoneCount && model.hasSkeletonRepresentation)
                if(ImGui::Selectable(model.importName.c_str(),copySource==model.importName))copySource=model.importName;
        ImGui::EndCombo();
    }
    ImGui::InputText("Copy as",copyName,sizeof(copyName));
    ImGui::BeginDisabled(copySource.empty());
    if(ImGui::Button("Create editable copy")) {
        std::vector<RigAuthoring::RigCopyBone> mapping;
        const auto result=rtapi::copyRigFrom(copySource,copyName,mapping);
        message=result.ok?"Editable rest copy created: "+std::string(copyName)+" ("+std::to_string(mapping.size())+" source joints)":result.error;
        if(result.ok)character=copyName;
    }
    ImGui::EndDisabled();ImGui::EndDisabled();
    ImGui::TextDisabled("Copies rest skeleton only; source and its clips remain intact.");
    ImGui::Separator();
    ImGui::BeginDisabled(editing);
    if(ImGui::BeginCombo("Owned rig",character.empty()?"Choose rig":character.c_str())) {
        for(const auto& model:ctx.scene.importedModelContexts)if(model.authoringOwned)
            if(ImGui::Selectable(model.importName.c_str(),character==model.importName)) {
                character=model.importName;
                if(!model.nodeHierarchy.empty())rtapi::selectRigBone(character,model.nodeHierarchy.nodes[0].uniqueName);
            }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if(!character.empty()) {
        std::vector<RigAuthoring::BoneView> bones;
        auto listed=rtapi::listRigBones(character,bones);
        if(!listed.ok)ImGui::TextWrapped("%s",listed.error.c_str());
        else {
            RigAuthoring::BoneView selected;
            bool found=false;
            for(const auto& bone:bones)if(ctx.scene.rigView.character==character && ctx.scene.rigView.bone==bone.name){selected=bone;found=true;break;}
            if(!found && !bones.empty()){selected=bones.front();found=true;}
            if(ImGui::BeginCombo("Selected / parent bone",found?selected.name.c_str():"Choose bone")) {
                for(const auto& bone:bones)if(ImGui::Selectable(bone.name.c_str(),found && selected.name==bone.name)) {
                    rtapi::selectRigBone(character,bone.name);selected=bone;found=true;
                }
                ImGui::EndCombo();
            }
            if(found) {
                ImGui::Text("Revision: %llu | BoneData %s | Hierarchy %s | Overlay %s | Ozz %s",
                    static_cast<unsigned long long>(selected.rig_revision),selected.in_bonedata?"OK":"missing",
                    selected.in_node_hierarchy?"OK":"missing",selected.in_skeleton_nodes?"OK":"missing",selected.in_ozz_skeleton?"OK":"missing");
                const std::string selectedKey=character+"|"+selected.name;
                if(topologyKey!=selectedKey || topologyRevision!=selected.rig_revision) {
                    const auto prefix=character+"_";
                    const auto authored=selected.name.find(prefix)==0?selected.name.substr(prefix.size()):selected.name;
                    std::snprintf(renameName,sizeof(renameName),"%s",authored.c_str());
                    topologyKey=selectedKey;topologyRevision=selected.rig_revision;newParent=selected.parent;
                }
                if(!editing) {
                    static std::string placementKey;static uint64_t placementRevision=0;
                    static Vec3 scenePosition,sceneRotation;static float sceneScale=1.f;
                    if(placementKey!=character || placementRevision!=selected.rig_revision) {
                        Vec3 scale;selected.scene_transform.decompose(scenePosition,sceneRotation,scale);sceneScale=scale.x;
                        placementKey=character;placementRevision=selected.rig_revision;
                    }
                    ImGui::SeparatorText("Whole rig scene placement");
                    ImGui::InputFloat3("Scene position",&scenePosition.x);
                    ImGui::InputFloat3("Scene rotation (degrees)",&sceneRotation.x);
                    ImGui::InputFloat("Uniform rig scale",&sceneScale,.05f,.5f,"%.4f");
                    ImGui::BeginDisabled(ImGuizmo::IsUsing());
                    if(ImGui::Button("Apply rig placement")) {
                        auto r=rtapi::setRigSceneTransform(character,Matrix4x4::composeTRS(scenePosition,sceneRotation,Vec3(sceneScale,sceneScale,sceneScale)));
                        message=r.ok?"Whole rig placement updated":r.error;
                    }
                    ImGui::EndDisabled();
                    ImGui::TextDisabled("Select any joint to place its whole rig. Enter Rig Edit for bone edits.");
                }
                ImGui::BeginDisabled(!editing);
                ImGui::SeparatorText("Bone topology");
                ImGui::BeginDisabled(ImGuizmo::IsUsing());
                ImGui::InputText("Authored bone name",renameName,sizeof(renameName));
                if(ImGui::Button("Rename bone")) {
                    const auto result=rtapi::renameRigBone(character,selected.name,renameName);
                    message=result.ok?"Bone renamed":result.error;
                    if(result.ok){ImGui::EndDisabled();ImGui::EndDisabled();return;}
                }
                const bool root=selected.parent.empty();
                ImGui::BeginDisabled(root);
                if(ImGui::BeginCombo("New parent",newParent.c_str())) {
                    for(const auto& bone:bones)if(bone.name!=selected.name)
                        if(ImGui::Selectable(bone.name.c_str(),bone.name==newParent))newParent=bone.name;
                    ImGui::EndCombo();
                }
                if(ImGui::Button("Reparent (keep world pose)")) {
                    const auto result=rtapi::reparentRigBone(character,selected.name,newParent);
                    message=result.ok?"Parent changed; world rest pose preserved":result.error;
                    if(result.ok){ImGui::EndDisabled();ImGui::EndDisabled();ImGui::EndDisabled();return;}
                }
                bool hasChildren=false;
                for(const auto& bone:bones)if(bone.parent==selected.name){hasChildren=true;break;}
                ImGui::BeginDisabled(hasChildren);
                if(ImGui::Button("Delete leaf bone")) {
                    const auto result=rtapi::deleteRigBone(character,selected.name);
                    message=result.ok?"Leaf deleted; parent selected":result.error;
                    if(result.ok){ImGui::EndDisabled();ImGui::EndDisabled();ImGui::EndDisabled();ImGui::EndDisabled();return;}
                }
                ImGui::EndDisabled();ImGui::EndDisabled();ImGui::EndDisabled();
                if(root)ImGui::TextDisabled("Root can be renamed; reparent/delete are blocked.");
                else if(hasChildren)ImGui::TextDisabled("Reparent children or delete leaves first.");
                ImGui::Separator();
                ImGui::TextWrapped("New child parent: %s",selected.name.c_str());
                ImGui::Checkbox("Automatic joint names",&automaticName);
                if(automaticName) {
                    std::string next;
                    const auto result=rtapi::getNextRigBoneName(character,jointName[0]?jointName:"Joint1",next);
                    if(result.ok)std::snprintf(jointName,sizeof(jointName),"%s",next.c_str());
                }
                ImGui::InputText("New joint name",jointName,sizeof(jointName));
                ImGui::InputFloat3("New joint local position",&addPosition.x);
                if(ImGui::Button("Add child joint")) {
                    const auto r=rtapi::addRigBone(character,jointName,selected.name,Matrix4x4::translation(addPosition));
                    message=r.ok?"Added: "+std::string(jointName)+"; selected as next parent":r.error;
                    if(r.ok && automaticName) {
                        std::string next;
                        if(rtapi::getNextRigBoneName(character,jointName,next).ok)
                            std::snprintf(jointName,sizeof(jointName),"%s",next.c_str());
                    }
                }
                const std::string key=character+"|"+selected.name;
                if(editingKey!=key || editingRevision!=selected.rig_revision) {
                    Vec3 scale;selected.local_rest.decompose(position,rotation,scale);
                    editingKey=key;editingRevision=selected.rig_revision;
                }
                ImGui::SeparatorText("Local rest transform");
                ImGui::InputFloat3("Position",&position.x);ImGui::InputFloat3("Rotation (degrees)",&rotation.x);
                if(ImGui::Button("Apply rest transform")) {
                    const auto rest=Matrix4x4::composeTRS(position,rotation,Vec3(1,1,1));
                    const auto r=rtapi::setRigRestTransform(character,selected.name,rest);
                    message=r.ok?"Rest pose updated":r.error;
                }
                ImGui::EndDisabled();
            }
        }
    }
    if(!character.empty())drawRigPoseViewControls(ctx,character);
    drawRigJointProfile(ctx,character);
    drawRigAnatomy(ctx,character);
    ImGui::SeparatorText("Alignment and deformation");
    drawRigMirror(ctx,character);
    drawRigFitting(ctx,character,targetMesh);
    drawRigBinding(ctx,character,targetMesh);
    ImGui::TextDisabled("Owned, unskinned rigs without bound clips; rigid rest transforms and leaf deletion.");
    if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
}
}
