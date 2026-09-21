#include "UI/RigFittingUI.h"
#include "UI/RigBindingUI.h"
#include "Api/RtApi.h"
#include "Animation/RigBindMath.h"
#include "scene_ui.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <algorithm>
#include <cmath>
namespace RigUI {
namespace {
struct View {float x=0,y=0,width=1,height=1,zoom=1;};
struct Draft {
    nlohmann::json setup,report,dragStart,anatomy;
    std::string character,mesh,bone,message;
    std::vector<std::string> bones;
    View views[3];int load=-1,dragPlane=-1,panPlane=-1;
    bool open=false,confirmed=false,showTop=false,showMesh=true,showSkeleton=true,showLabels=true;
    float opacity=.45f,mirrorOffset=0;int mirrorAxis=0,guideMode=0;bool mirrorLive=false;
} draft;
const int horizontal[3]={0,2,0},vertical[3]={1,1,2};
// Match the viewport's Front, Right and Top camera directions.
const float horizontalSign[3]={1.f,-1.f,1.f},verticalSign[3]={1.f,1.f,-1.f};
const char* captions[3]={"Front (X/Y)","Side / Right (-Z/Y)","Top (X/-Z)"};
void fitViews() {
    if(!draft.setup.is_object())return;
    for(int plane=0;plane<3;++plane) {
        const int a=horizontal[plane],b=vertical[plane];
        float loX=draft.setup["bounds"]["min"][a].get<float>(),hiX=draft.setup["bounds"]["max"][a].get<float>();
        float loY=draft.setup["bounds"]["min"][b].get<float>(),hiY=draft.setup["bounds"]["max"][b].get<float>();
        for(const auto& mark:draft.setup["landmarks"].items()) {
            const float x=mark.value()[a].get<float>(),y=mark.value()[b].get<float>();
            loX=std::min(loX,x);hiX=std::max(hiX,x);loY=std::min(loY,y);hiY=std::max(hiY,y);
        }
        auto& v=draft.views[plane];v.x=loX*.5f+hiX*.5f;v.y=loY*.5f+hiY*.5f;
        v.width=std::max(hiX-loX,1e-5f);v.height=std::max(hiY-loY,1e-5f);v.zoom=1;
    }
}
void cancelDrag() {
    if(draft.dragPlane>=0 && draft.setup.is_object() && !draft.bone.empty()) {
        draft.setup["landmarks"]=draft.dragStart;draft.report=nullptr;
    }
    draft.dragPlane=-1;draft.panPlane=-1;
}
void prepare() {
    cancelDrag();nlohmann::json setup;
    const auto r=rtapi::getRigFitSetup(draft.character,draft.mesh,setup);
    draft.report=nullptr;draft.confirmed=false;draft.bone.clear();draft.bones.clear();
    draft.setup=r.ok?std::move(setup):nlohmann::json();
    draft.message=r.ok?"Adjust landmarks, confirm axes, then preview.":r.error;
    draft.mirrorLive=false;draft.mirrorAxis=0;draft.mirrorOffset=0;draft.anatomy=nullptr;
    if(r.ok) {
        rtapi::getRigAnatomy(draft.character,draft.anatomy);
        Matrix4x4 placement,inverse;
        if(rtapi::getRigSceneTransform(draft.character,placement).ok && RigAuthoring::bindAffineInverse(placement,inverse)) {
            const auto& b=draft.setup["bounds"];Vec3 center;
            center.x=b["min"][0].get<float>()*.5f+b["max"][0].get<float>()*.5f;
            center.y=b["min"][1].get<float>()*.5f+b["max"][1].get<float>()*.5f;
            center.z=b["min"][2].get<float>()*.5f+b["max"][2].get<float>()*.5f;
            draft.mirrorOffset=inverse.transform_point(center).x;
        }
        fitViews();
    }
}
void discard() {cancelDrag();draft.setup=nullptr;draft.report=nullptr;draft.confirmed=false;draft.open=false;}
bool outsideJoint(const std::string& name) {
    if(draft.report.is_object())for(const auto& joint:draft.report["joints"])
        if(joint["name"]==name)return !joint["inside_bounds"].get<bool>();
    return false;
}
bool pairedLandmark(const std::string& name) {
    if(draft.anatomy.is_object())for(const auto& p:draft.anatomy["symmetry"])
        if(p["left"]==name || p["right"]==name)return true;
    return false;
}
const nlohmann::json* jointRow(const std::string& name) {
    if(!draft.setup.is_object())return nullptr;
    for(const auto& joint:draft.setup["joints"])
        if(joint.value("name",std::string())==name)return &joint;
    return nullptr;
}
bool selectableGuide(const std::string& name) {
    const auto* joint=jointRow(name);
    if(!joint)return false;
    if(!joint->value("fit_editable",true))return false;
    const auto group=joint->value("fit_group",std::string("primary"));
    return draft.guideMode==2 || (draft.guideMode==0 && group=="primary") ||
           (draft.guideMode==1 && group=="hands");
}
void resolveDraftRules(nlohmann::json& landmarks);
void mirrorDraft(const std::vector<std::string>& bones,const std::string& direction) {
    const char* axes[]={"x","y","z"};nlohmann::json mirrored;
    auto r=rtapi::mirrorRigLandmarks(draft.character,draft.setup["landmarks"],bones,direction,axes[draft.mirrorAxis],draft.mirrorOffset,draft.setup["rig_revision"].get<uint64_t>(),mirrored);
    if(r.ok){resolveDraftRules(mirrored);draft.setup["landmarks"]=std::move(mirrored);draft.report=nullptr;draft.message="Draft mirrored. Apply publishes it to scene.";}else draft.message=r.error;
}
void resolveDraftRules(nlohmann::json& landmarks) {
    if(!draft.anatomy.is_object() || !draft.anatomy.contains("fit_rules"))return;
    for(const auto& rule:draft.anatomy["fit_rules"]) {
        const auto start=rule["start"].get<std::string>();
        const auto end=rule["end"].get<std::string>();
        const auto bone=rule["bone"].get<std::string>();
        const float t=rule["position"].get<float>();
        if(!landmarks.contains(start)||!landmarks.contains(end))continue;
        nlohmann::json value=nlohmann::json::array();
        for(int axis=0;axis<3;++axis) {
            const float a=landmarks[start][axis].get<float>();
            const float b=landmarks[end][axis].get<float>();
            value.push_back(a+(b-a)*t);
        }
        landmarks[bone]=std::move(value);
    }
}
bool updateLandmark(const std::string& name,const nlohmann::json& position) {
    auto candidate=draft.setup["landmarks"];candidate[name]=position;
    if(draft.mirrorLive && pairedLandmark(name)) {
        const char* axes[]={"x","y","z"};nlohmann::json mirrored;
        const auto r=rtapi::mirrorRigLandmarks(draft.character,candidate,{name},"selected",axes[draft.mirrorAxis],draft.mirrorOffset,draft.setup["rig_revision"].get<uint64_t>(),mirrored);
        if(!r.ok){draft.message=r.error;return false;}candidate=std::move(mirrored);
    }
    resolveDraftRules(candidate);
    draft.setup["landmarks"]=std::move(candidate);draft.report=nullptr;return true;
}
bool updateLandmarks(nlohmann::json candidate,const std::vector<std::string>& sources) {
    if(draft.mirrorLive) {
        std::vector<std::string> paired;
        for(const auto& name:sources)if(pairedLandmark(name))paired.push_back(name);
        if(!paired.empty()) {
            const char* axes[]={"x","y","z"};nlohmann::json mirrored;
            const auto r=rtapi::mirrorRigLandmarks(
                draft.character,candidate,paired,"selected",axes[draft.mirrorAxis],
                draft.mirrorOffset,draft.setup["rig_revision"].get<uint64_t>(),mirrored);
            if(!r.ok){draft.message=r.error;return false;}candidate=std::move(mirrored);
        }
    }
    resolveDraftRules(candidate);
    draft.setup["landmarks"]=std::move(candidate);draft.report=nullptr;return true;
}
void canvas(int plane,float height) {
    const int a=horizontal[plane],b=vertical[plane];auto& view=draft.views[plane];
    const float sx=horizontalSign[plane],sy=verticalSign[plane];
    ImGui::PushID(plane);ImGui::TextUnformatted(captions[plane]);
    const ImVec2 size(std::max(80.f,ImGui::GetContentRegionAvail().x),height),origin=ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("AlignmentCanvas",size,ImGuiButtonFlags_MouseButtonLeft|ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered=ImGui::IsItemHovered();auto& io=ImGui::GetIO();
    const float fit=std::min((size.x-24.f)/view.width,(size.y-24.f)/view.height);
    float scale=std::max(fit*view.zoom,1e-10f);
    const ImVec2 center(origin.x+size.x*.5f,origin.y+size.y*.5f);
    if(hovered && draft.dragPlane<0 && draft.panPlane<0 && io.MouseWheel!=0) {
        const float x=view.x+sx*(io.MousePos.x-center.x)/scale,y=view.y-sy*(io.MousePos.y-center.y)/scale;
        view.zoom=std::clamp(view.zoom*std::pow(1.2f,io.MouseWheel),.05f,100.f);scale=std::max(fit*view.zoom,1e-10f);
        view.x=x-sx*(io.MousePos.x-center.x)/scale;view.y=y+sy*(io.MousePos.y-center.y)/scale;
    }
    if(hovered && draft.dragPlane<0 && !ImGuizmo::IsUsing() && ImGui::IsMouseClicked(ImGuiMouseButton_Middle))draft.panPlane=plane;
    if(draft.panPlane==plane && ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        view.x-=sx*io.MouseDelta.x/scale;view.y+=sy*io.MouseDelta.y/scale;
    }
    auto project=[&](const nlohmann::json& p) {return ImVec2(center.x+sx*(p[a].get<float>()-view.x)*scale,center.y-sy*(p[b].get<float>()-view.y)*scale);};
    if(draft.dragPlane==plane && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        const float x=view.x+sx*(io.MousePos.x-center.x)/scale,y=view.y-sy*(io.MousePos.y-center.y)/scale;
        if(std::isfinite(x)&&std::isfinite(y)) {
            auto candidate=draft.dragStart;const auto active=draft.dragStart[draft.bone];
            const float dx=x-active[a].get<float>(),dy=y-active[b].get<float>();
            const auto selected=draft.bones.empty()?std::vector<std::string>{draft.bone}:draft.bones;
            for(const auto& name:selected) {
                auto p=candidate[name];p[a]=p[a].get<float>()+dx;p[b]=p[b].get<float>()+dy;
                candidate[name]=std::move(p);
            }
            updateLandmarks(std::move(candidate),selected);
        }
    }
    auto* draw=ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin,ImVec2(origin.x+size.x,origin.y+size.y),IM_COL32(22,25,32,255));
    draw->PushClipRect(origin,ImVec2(origin.x+size.x,origin.y+size.y),true);
    const auto zero=project(nlohmann::json::array({0.f,0.f,0.f}));
    draw->AddLine(ImVec2(origin.x,zero.y),ImVec2(origin.x+size.x,zero.y),IM_COL32(85,90,105,90));
    draw->AddLine(ImVec2(zero.x,origin.y),ImVec2(zero.x,origin.y+size.y),IM_COL32(85,90,105,90));
    const auto lo=project(draft.setup["bounds"]["min"]),hi=project(draft.setup["bounds"]["max"]);
    draw->AddRect(ImVec2(std::min(lo.x,hi.x),std::min(lo.y,hi.y)),ImVec2(std::max(lo.x,hi.x),std::max(lo.y,hi.y)),IM_COL32(100,115,135,100));
    if(draft.showMesh)for(const auto& p:draft.setup["mesh_points"])
        draw->AddCircleFilled(project(p),1.25f,IM_COL32(160,175,195,static_cast<int>(draft.opacity*255.f)));
    auto& marks=draft.setup["landmarks"];std::string pick;float nearest=9.f;
    if(draft.showSkeleton)for(const auto& joint:draft.setup["joints"]) {
        const auto name=joint["name"].get<std::string>(),parent=joint["parent"].get<std::string>();const auto p=project(marks[name]);
        const bool selected=std::find(draft.bones.begin(),draft.bones.end(),name)!=draft.bones.end();
        const bool guide=selectableGuide(name);
        const auto color=selected?IM_COL32(255,170,45,255):(outsideJoint(name)?IM_COL32(255,95,85,255):(guide?IM_COL32(120,215,235,255):IM_COL32(95,110,125,120)));
        if(!parent.empty())draw->AddLine(project(marks[parent]),p,color,selected?2.5f:1.5f);
        draw->AddCircleFilled(p,selected?5.f:3.5f,color);
        if(draft.showLabels && selected)draw->AddText(ImVec2(p.x+8,p.y-8),color,name.c_str());
        const float dx=p.x-io.MousePos.x,dy=p.y-io.MousePos.y,d=std::sqrt(dx*dx+dy*dy);
        if(guide && d<nearest){nearest=d;pick=name;}
    }
    if(hovered && draft.dragPlane<0 && draft.panPlane<0 && !ImGuizmo::IsUsing() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !pick.empty()) {
        const bool additive=io.KeyCtrl||io.KeyShift;
        auto found=std::find(draft.bones.begin(),draft.bones.end(),pick);
        if(additive) {
            if(io.KeyCtrl && found!=draft.bones.end())draft.bones.erase(found);
            else if(found==draft.bones.end())draft.bones.push_back(pick);
        } else if(found==draft.bones.end())draft.bones={pick};
        if(std::find(draft.bones.begin(),draft.bones.end(),pick)!=draft.bones.end()) {
            draft.bone=pick;draft.dragStart=marks;draft.dragPlane=plane;
        }
    }
    draw->PopClipRect();draw->AddRect(origin,ImVec2(origin.x+size.x,origin.y+size.y),IM_COL32(75,85,100,255));
    ImGui::TextDisabled("Zoom %.0f%%",view.zoom*100.f);ImGui::PopID();
}
void preview() {
    nlohmann::json current;auto r=rtapi::getRigFitSetup(draft.character,draft.mesh,current);
    if(!r.ok){draft.report=nullptr;draft.message=r.error;return;}
    if(current["rig_revision"]!=draft.setup["rig_revision"] || current["mesh_token"]!=draft.setup["mesh_token"]) {
        draft.report=nullptr;draft.message="Mesh or skeleton changed. Prepare alignment again.";return;
    }
    r=rtapi::previewRigFit(draft.character,draft.mesh,draft.setup["landmarks"],draft.confirmed,draft.report);
    if(r.ok && draft.report.contains("landmarks"))draft.setup["landmarks"]=draft.report["landmarks"];
    draft.message=r.ok?(draft.report.value("can_commit",false)?"Preview ready. Apply publishes the rest landmarks to the scene.":"Alignment blocked: some joints are outside mesh bounds. Select the listed joints and adjust them."):r.error;
}
}
void drawRigFitting(UIContext& ctx,const std::string& character,const std::string& mesh) {
    if(character.empty()||mesh.empty())return;
    if(!ImGui::CollapsingHeader("Fit skeleton to mesh"))return;
    ImGui::TextWrapped("Edit manual landmarks in the Alignment panel.");
    ImGui::BeginDisabled(ImGuizmo::IsUsing());
    if(ImGui::Button("Open Alignment")) {
        if(draft.character!=character || draft.mesh!=mesh || draft.load!=ctx.scene.load_counter) {
            draft=Draft{};draft.character=character;draft.mesh=mesh;draft.load=ctx.scene.load_counter;
        }
        draft.open=true;if(!draft.setup.is_object())prepare();ImGui::SetWindowFocus("Skeleton Alignment###RigAlignment");
    }
    ImGui::EndDisabled();
    if(draft.character==character && draft.mesh==mesh && !draft.message.empty())ImGui::TextWrapped("%s",draft.message.c_str());
}
void drawRigAlignmentWindow(UIContext& ctx) {
    if(!draft.open)return;
    if(draft.load!=ctx.scene.load_counter){discard();return;}
    const auto display=ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowSize(ImVec2(std::min(1150.f,display.x*.9f),std::min(760.f,display.y*.9f)),ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(std::min(620.f,display.x*.9f),std::min(420.f,display.y*.9f)),ImVec2(display.x,display.y));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(8,6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,ImVec2(6,4));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,ImVec2(6,3));
    bool open=true,closeDrawn=false;
    if(ImGui::Begin("Skeleton Alignment###RigAlignment",&open,ImGuiWindowFlags_NoScrollWithMouse)) {
        if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && ImGui::IsKeyPressed(ImGuiKey_Escape))cancelDrag();
        ImGui::BeginDisabled(ImGuizmo::IsUsing() || draft.dragPlane>=0 || draft.panPlane>=0);
        if(ImGui::Button("Prepare"))prepare();ImGui::SameLine();
        if(ImGui::Button("Fit to View"))fitViews();ImGui::SameLine();ImGui::Checkbox("Top",&draft.showTop);
        ImGui::SameLine();ImGui::SetNextItemWidth(105.f);
        const char* guideModes[]={"Primary","Hands","All guides"};
        if(ImGui::Combo("##FitGuides",&draft.guideMode,guideModes,3)) {
            cancelDrag();draft.bone.clear();draft.bones.clear();
        }
        ImGui::EndDisabled();ImGui::SameLine();ImGui::TextDisabled("(?)");
        if(ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::Text("Skeleton: %s",draft.character.c_str());ImGui::Text("Mesh: %s",draft.mesh.c_str());
            ImGui::TextUnformatted("Wheel: cursor zoom | Middle drag: pan | Left drag: landmark\nEscape: cancel drag. Apply publishes draft rest joints to scene.\nBinding generates initial weights after alignment.");
            ImGui::EndTooltip();
        }
        if(draft.setup.is_object()) {
            ImGui::Checkbox("Mesh",&draft.showMesh);ImGui::SameLine();
            if(ImGui::Checkbox("Skeleton",&draft.showSkeleton) && !draft.showSkeleton)cancelDrag();
            ImGui::SameLine();ImGui::Checkbox("Label",&draft.showLabels);ImGui::SameLine();
            ImGui::SetNextItemWidth(100);ImGui::SliderFloat("Opacity",&draft.opacity,.05f,1.f);
            ImGui::SameLine();ImGui::BeginDisabled(draft.dragPlane>=0 || draft.panPlane>=0);
            ImGui::Checkbox("Mirror",&draft.mirrorLive);ImGui::SameLine();
            if(ImGui::Button("Mirror..."))ImGui::OpenPopup("AlignmentMirror");
            if(ImGui::BeginPopup("AlignmentMirror")) {
                const char* axes[]={"x","y","z"};ImGui::SetNextItemWidth(80);ImGui::Combo("Rig axis",&draft.mirrorAxis,axes,3);
                ImGui::SetNextItemWidth(140);ImGui::InputFloat("Plane offset (rig space)",&draft.mirrorOffset);
                ImGui::TextWrapped("Offset starts at the mesh bounds center in rig space. Live mirror uses anatomy pairs; center joints edit independently.");
                if(ImGui::Button("Left -> Right"))mirrorDraft({},"left_to_right");ImGui::SameLine();
                if(ImGui::Button("Right -> Left"))mirrorDraft({},"right_to_left");
                ImGui::EndPopup();
            }
            ImGui::EndDisabled();
            // Reserve actual compact control rows and the canvas caption/zoom
            // lines, rather than a fixed 245-pixel footer on every window size.
            const float status=draft.message.empty()?0.f:ImGui::CalcTextSize(draft.message.c_str(),nullptr,false,ImGui::GetContentRegionAvail().x).y+ImGui::GetStyle().ItemSpacing.y;
            const float outside=draft.report.is_object() && draft.report.value("outside_bounds",0ull)>0?ImGui::GetFrameHeightWithSpacing():0.f;
            const float confirmation=draft.confirmed?0.f:ImGui::GetTextLineHeightWithSpacing();
            const float footer=ImGui::GetFrameHeightWithSpacing()*2.f+ImGui::GetTextLineHeightWithSpacing()*2.f+status+outside+confirmation+8.f;
            const float height=std::max(80.f,ImGui::GetContentRegionAvail().y-footer);
            if(ImGui::BeginTable("AlignmentViews",draft.showTop?3:2,ImGuiTableFlags_Resizable|ImGuiTableFlags_SizingStretchSame)) {
                for(int plane=0;plane<(draft.showTop?3:2);++plane){ImGui::TableNextColumn();canvas(plane,height);}
                ImGui::EndTable();
            }
            if(!ImGui::IsMouseDown(ImGuiMouseButton_Left))draft.dragPlane=-1;
            if(!ImGui::IsMouseDown(ImGuiMouseButton_Middle))draft.panPlane=-1;
            ImGui::BeginDisabled(draft.dragPlane>=0 || draft.panPlane>=0 || ImGuizmo::IsUsing());
            const float width=ImGui::GetContentRegionAvail().x;
            ImGui::SetNextItemWidth(std::min(200.f,width*.35f));
            if(ImGui::BeginCombo("##Joint",draft.bone.empty()?"Choose joint":draft.bone.c_str())) {
                for(const auto& joint:draft.setup["joints"]) {
                    const auto name=joint["name"].get<std::string>();if(!selectableGuide(name))continue;
                    if(ImGui::Selectable(name.c_str(),name==draft.bone)){draft.bone=name;draft.bones={name};}
                }
                ImGui::EndCombo();
            }
            if(!draft.bone.empty()) {
                ImGui::SameLine();auto& p=draft.setup["landmarks"][draft.bone];float xyz[3]={p[0].get<float>(),p[1].get<float>(),p[2].get<float>()};
                ImGui::SetNextItemWidth(std::max(100.f,std::min(350.f,ImGui::GetContentRegionAvail().x-35.f)));
                if(ImGui::InputFloat3("XYZ",xyz)) {
                    if(std::isfinite(xyz[0])&&std::isfinite(xyz[1])&&std::isfinite(xyz[2])){updateLandmark(draft.bone,nlohmann::json::array({xyz[0],xyz[1],xyz[2]}));}
                    else draft.message="Landmark coordinates must be finite.";
                }
                if(ImGui::IsItemHovered())ImGui::SetTooltip("Landmark world position");
            }
            if(draft.bones.size()>1)ImGui::TextDisabled(
                "%llu guides selected; drag a selected guide to move the group.",
                static_cast<unsigned long long>(draft.bones.size()));
            if(ImGui::Checkbox("Axes/rest checked",&draft.confirmed))draft.report=nullptr;
            if(ImGui::IsItemHovered())ImGui::SetTooltip("Confirm mesh is +Y up, +Z forward and in rest pose.");
            ImGui::SameLine();ImGui::BeginDisabled(!draft.confirmed);
            if(ImGui::Button("Preview"))preview();ImGui::SameLine();
            if(ImGui::Button("Apply")) {
                preview();
                if(draft.report.is_object() && draft.report.value("can_commit",false)) {
                    const auto r=rtapi::commitRigFit(draft.character,draft.mesh,draft.report);
                    draft.message=r.ok?"Rest alignment applied to scene. Bind below to generate initial weights.":r.error;
                    if(r.ok){draft.setup=nullptr;draft.report=nullptr;draft.confirmed=false;}
                }
            }
            ImGui::EndDisabled();ImGui::EndDisabled();ImGui::SameLine();
            if(ImGui::Button("Cancel / Close"))open=false;closeDrawn=true;
            if(draft.setup.is_object() && !draft.confirmed)ImGui::TextDisabled("Apply requires axes/rest confirmation.");
            if(draft.report.is_object() && draft.report.value("outside_bounds",0ull)>0) {
                ImGui::SetNextItemWidth(220);
                const auto label="Outside bounds: "+std::to_string(draft.report["outside_bounds"].get<unsigned long long>());
                if(ImGui::BeginCombo("##OutsideJoints",label.c_str())) {
                    for(const auto& joint:draft.report["joints"])if(!joint["inside_bounds"].get<bool>()) {
                        const auto name=joint["name"].get<std::string>();if(ImGui::Selectable(name.c_str(),name==draft.bone)){draft.bone=name;draft.bones={name};}
                    }
                    ImGui::EndCombo();
                }
            }
        }
        if(!draft.setup.is_object())drawRigBinding(ctx,draft.character,draft.mesh);
        if(!closeDrawn && ImGui::Button("Cancel / Close"))open=false;
        if(!draft.message.empty())ImGui::TextWrapped("%s",draft.message.c_str());
    }
    ImGui::End();ImGui::PopStyleVar(3);if(!open)discard();
}
}
