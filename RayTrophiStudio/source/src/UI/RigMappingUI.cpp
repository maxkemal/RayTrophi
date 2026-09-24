#include "UI/RigMappingUI.h"
#include "UI/RigPoseViewUI.h"
#include "UI/ClipBindingUIState.h"
#include "Api/RtApi.h"
#include "Animation/RigPosePreview.h"
#include "scene_ui.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace RigUI {
namespace {
struct MappingView {
    std::string target, sourceNode, targetNode, fitKey;
    bool sourceRest=false,targetRest=false;
    bool playing=false, unresolvedOnly=false, fitted=false;
    float time=0, duration=1, zoom=1;
    int plane=0;
    char filter[128]={};
    ImVec2 centers[2]; float extentX=1, extentY=1;
    ImVec2 panOffset={0,0};
};
ImVec2 position(const RigAuthoring::PreviewJoint& j, int plane) {
    return ImVec2(j.world.m[plane==0?0:2][3], j.world.m[plane==2?0:1][3]);
}
void fit(MappingView& v, const RigAuthoring::ClipPosePreview& p) {
    v.extentX=1e-3f; v.extentY=1e-3f;
    const std::vector<RigAuthoring::PreviewJoint>* sets[]={&p.source,&p.target};
    for (int i=0;i<2;++i) {
        ImVec2 lo(std::numeric_limits<float>::max(),std::numeric_limits<float>::max()), hi(-lo.x,-lo.y);
        for(const auto& j:*sets[i]) { const auto xy=position(j,v.plane); lo.x=std::min(lo.x,xy.x);lo.y=std::min(lo.y,xy.y);hi.x=std::max(hi.x,xy.x);hi.y=std::max(hi.y,xy.y); }
        if(sets[i]->empty()) { v.centers[i]=ImVec2(); continue; }
        v.centers[i]=ImVec2((lo.x+hi.x)*.5f,(lo.y+hi.y)*.5f);
        v.extentX=std::max(v.extentX,hi.x-lo.x); v.extentY=std::max(v.extentY,hi.y-lo.y);
    }
    v.panOffset = ImVec2(0, 0);
    v.fitted=true;
}
void selectSource(MappingView& v, ClipBindingUIState& state, const std::string& name) {
    v.sourceNode=name;v.targetNode.clear();
    const auto manual=state.nodeMap.find(name);
    if(manual!=state.nodeMap.end())v.targetNode=manual->second;
    else for(const auto& match:state.report.matches)if(match.source==name){v.targetNode=match.target;break;}
    rtapi::selectRigBone(state.source,name); // Helpers may have no selectable BoneData entry.
}
bool issue(const RigAuthoring::ClipBindingReport& r,const std::string& key) {
    const auto has=[&](const std::vector<std::string>& values){return std::find(values.begin(),values.end(),key)!=values.end();};
    return has(r.unmapped)||has(r.ambiguous)||has(r.hierarchy_mismatches);
}
void drawMapping(UIContext& ctx) {
    static MappingView v;
    static float leftPanelWidth = 310.0f;
    static bool isPanningCanvas = false;

    if(ImGui::BeginCombo("Target character",v.target.empty()?"Choose target":v.target.c_str())) {
        for(const auto& model:ctx.scene.importedModelContexts) if(model.hasSkeletonRepresentation)
            if(ImGui::Selectable(model.importName.c_str(),v.target==model.importName)) {
                v.target=model.importName;v.sourceNode.clear();v.targetNode.clear();v.fitted=false;v.playing=false;v.time=0;
            }
        ImGui::EndCombo();
    }
    if(v.target.empty()) { ImGui::TextWrapped("Choose a target, then its source rig and clip. Pick a source joint followed by a target joint to override mapping.");return; }
    auto& state=clipBindingState(v.target);
    float availHeight = ImGui::GetContentRegionAvail().y;

    ImGui::BeginChild("RetargetControls",ImVec2(leftPanelWidth,0),true);
    drawClipBindingContents(ctx,v.target,false);
    const std::string fitKey=v.target+"|"+state.source+"|"+state.clip;
    if(v.fitKey!=fitKey){v.fitKey=fitKey;v.fitted=false;v.sourceNode.clear();v.targetNode.clear();v.time=0;}
    ImGui::Separator(); ImGui::InputText("Find node",v.filter,sizeof(v.filter));
    ImGui::Checkbox("Unresolved source only",&v.unresolvedOnly);
    ImGui::TextWrapped("Source: %s",v.sourceNode.c_str());ImGui::TextWrapped("Target: %s",v.targetNode.c_str());
    if(!v.sourceNode.empty() && ImGui::Button("Use auto for selected source")) {
        state.nodeMap.erase(v.sourceNode);state.previewed=false;state.message.clear();
    }
    if(ImGui::Button("Clear all overrides")) {state.nodeMap.clear();state.previewed=false;state.message.clear();}
    // List fallback handles overlapping joints/helpers in orthographic projections.
    for(int side=0;side<2;++side) {
        if (UIWidgets::CollapsingHeader(side==0?"Source nodes":"Target nodes", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& character=side==0?state.source:v.target;
            for(const auto& model:ctx.scene.importedModelContexts) if(model.importName==character)
                for(const auto& node:model.nodeHierarchy.nodes) {
                    if(v.filter[0] && node.uniqueName.find(v.filter)==std::string::npos) continue;
                    if(side==0 && v.unresolvedOnly && !issue(state.report,node.uniqueName)) continue;
                    ImGui::PushID(side);
                    if(ImGui::Selectable(node.uniqueName.c_str(),(side==0?v.sourceNode:v.targetNode)==node.uniqueName)) {
                        if(side==0) selectSource(v,state,node.uniqueName);
                        else if(!v.sourceNode.empty()) {v.targetNode=node.uniqueName;state.nodeMap[v.sourceNode]=node.uniqueName;rtapi::selectRigBone(v.target,node.uniqueName);state.previewed=false;state.message.clear();}
                    }
                    ImGui::PopID();
                }
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Resize handle (splitter)
    ImGui::InvisibleButton("##RetargetPanelResize", ImVec2(6.0f, availHeight));
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    if (ImGui::IsItemActive()) {
        leftPanelWidth += ImGui::GetIO().MouseDelta.x;
        leftPanelWidth = std::clamp(leftPanelWidth, 180.0f, 600.0f);
    }

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    ImVec2 handleMin = ImGui::GetItemRectMin();
    ImVec2 handleMax = ImGui::GetItemRectMax();
    drawList->AddRectFilled(handleMin, handleMax,
        ImGui::IsItemHovered() ? IM_COL32(100, 100, 100, 255) : IM_COL32(60, 60, 60, 255));

    ImGui::SameLine();

    ImGui::BeginChild("RetargetView",ImVec2(0,0),true);
    if(ImGui::Checkbox("Source rest",&v.sourceRest))v.fitted=false;ImGui::SameLine();
    if(ImGui::Checkbox("Target rest",&v.targetRest))v.fitted=false;
    ImGui::BeginDisabled(v.sourceRest && v.targetRest);
    if(ImGui::Button(v.playing?"Pause preview":"Play preview")) v.playing=!v.playing;
    ImGui::EndDisabled();
    ImGui::SameLine(); if(ImGui::Button("Fit")) {v.fitted=false;v.zoom=1;v.panOffset=ImVec2(0,0);}
    ImGui::SameLine();const char* planes[]={"Front XY","Side ZY","Top ZX"};
    ImGui::SetNextItemWidth(110);if(ImGui::Combo("##Plane",&v.plane,planes,3))v.fitted=false;
    ImGui::SameLine();
    ImGui::TextDisabled("| Zoom: %.0f%%",v.zoom*100.f);
    if(v.playing && !(v.sourceRest && v.targetRest))v.time=std::fmod(v.time+ImGui::GetIO().DeltaTime,std::max(v.duration,1e-6f));
    ImGui::SliderFloat("Time (seconds)",&v.time,0,std::max(v.duration,1e-6f));
    RigAuthoring::ClipPosePreview p;
    const auto result=rtapi::sampleClipBinding(state.source,state.clip,v.target,v.time,p,state.nodeMap,state.restBasis?"rest_basis":"same_rig",state.translationScale,v.sourceRest?"rest":"animated",v.targetRest?"rest":"animated");
    if(result.ok) {state.report=p.binding;state.previewed=true;v.duration=static_cast<float>(p.duration_seconds);}
    else {
        state.previewed=false;
        ImGui::TextWrapped("Preview: %s (showing bind skeletons)",result.error.c_str());
        // Keep invalid mappings editable; canonical bind sampler supplies fallback geometry.
        std::string error;
        for(const auto& model:ctx.scene.importedModelContexts) {
            if(model.importName==state.source) RigAuthoring::sampleRigPose(model.nodeHierarchy,nullptr,0,p.source,error);
            if(model.importName==v.target) RigAuthoring::sampleRigPose(model.nodeHierarchy,nullptr,0,p.target,error);
        }
    }
    ImGui::Text("Source: %s | Target: %s",p.source_pose_source.empty()?"rest fallback":p.source_pose_source.c_str(),p.target_pose_source.empty()?"rest fallback":p.target_pose_source.c_str());
    const auto size=ImGui::GetContentRegionAvail();const auto origin=ImGui::GetCursorScreenPos();
    if(size.x>30 && size.y>30) {
        if(!v.fitted)fit(v,p);
        ImGui::InvisibleButton("MappingCanvas",size);auto* draw=ImGui::GetWindowDrawList();

        // Mouse Scroll Zoom & Middle/Right Drag Pan
        if(ImGui::IsItemHovered()) {
            float wheel = ImGui::GetIO().MouseWheel;
            if(wheel != 0.0f) {
                float oldZoom = v.zoom;
                v.zoom = std::clamp(v.zoom * (wheel > 0.0f ? 1.15f : (1.0f / 1.15f)), 0.15f, 8.0f);
                ImVec2 mouse = ImGui::GetIO().MousePos;
                ImVec2 relMouse = ImVec2(mouse.x - origin.x - size.x * 0.5f, mouse.y - origin.y - size.y * 0.5f);
                v.panOffset.x = relMouse.x - (relMouse.x - v.panOffset.x) * (v.zoom / oldZoom);
                v.panOffset.y = relMouse.y - (relMouse.y - v.panOffset.y) * (v.zoom / oldZoom);
            }
            if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                v.fitted = false;
                v.zoom = 1.0f;
                v.panOffset = ImVec2(0, 0);
            }
            if(ImGui::IsMouseClicked(ImGuiMouseButton_Middle) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                isPanningCanvas = true;
            }
        }
        if(isPanningCanvas) {
            if(ImGui::IsMouseDown(ImGuiMouseButton_Middle) || ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                ImVec2 delta = ImGui::GetIO().MouseDelta;
                v.panOffset.x += delta.x;
                v.panOffset.y += delta.y;
            } else {
                isPanningCanvas = false;
            }
        }

        draw->PushClipRect(origin,ImVec2(origin.x+size.x,origin.y+size.y),true);
        draw->AddLine(ImVec2(origin.x+size.x*.5f+v.panOffset.x,origin.y),ImVec2(origin.x+size.x*.5f+v.panOffset.x,origin.y+size.y),IM_COL32(80,80,80,255));
        const float scale=std::min(size.x*.42f/v.extentX,size.y*.8f/v.extentY)*v.zoom;
        std::unordered_map<std::string,ImVec2> points[2];
        const std::vector<RigAuthoring::PreviewJoint>* sets[]={&p.source,&p.target};
        for(int side=0;side<2;++side) for(const auto& j:*sets[side]) {
            const auto xy=position(j,v.plane);
            points[side][j.name]=ImVec2(origin.x+size.x*(side==0?.25f:.75f)+(xy.x-v.centers[side].x)*scale+v.panOffset.x,origin.y+size.y*.5f-(xy.y-v.centers[side].y)*scale+v.panOffset.y);
        }
        for(const auto& match:p.binding.matches) {
            const auto a=points[0].find(match.source),b=points[1].find(match.target);
            if(a!=points[0].end() && b!=points[1].end())draw->AddLine(a->second,b->second,IM_COL32(80,150,140,70));
        }
        std::string pick;int pickSide=-1;float nearest=81;
        const auto mouse=ImGui::GetIO().MousePos;
        for(int side=0;side<2;++side)for(const auto& j:*sets[side]) {
            const auto pt=points[side].at(j.name);const auto parent=points[side].find(j.parent);
            if(parent!=points[side].end())draw->AddLine(parent->second,pt,side==0?IM_COL32(90,195,235,255):IM_COL32(190,140,240,255),2);
            const bool selected=(side==0?v.sourceNode:v.targetNode)==j.name;
            draw->AddCircleFilled(pt,selected?6.f:3.f,selected?IM_COL32(255,180,50,255):issue(p.binding,j.name)?IM_COL32(250,85,80,255):IM_COL32(190,215,230,255));
            if(selected)draw->AddText(ImVec2(pt.x+8,pt.y),IM_COL32(255,210,100,255),j.name.c_str());
            const float d=(mouse.x-pt.x)*(mouse.x-pt.x)+(mouse.y-pt.y)*(mouse.y-pt.y);
            if(d<nearest && (mouse.x<origin.x+size.x*.5f+v.panOffset.x)==(side==0)){nearest=d;pick=j.name;pickSide=side;}
        }
        draw->PopClipRect();
        if(ImGui::IsItemHovered() && pickSide>=0) {
            ImGui::SetTooltip("%s",pick.c_str());
            if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                if(pickSide==0)selectSource(v,state,pick);
                else if(!v.sourceNode.empty()){v.targetNode=pick;state.nodeMap[v.sourceNode]=pick;rtapi::selectRigBone(v.target,pick);state.previewed=false;state.message.clear();}
            }
        }
    }
    ImGui::EndChild();
}
}
void drawAnimationWorkspace(UIContext& ctx,const std::function<void()>& drawGraph) {
    if(ImGui::BeginTabBar("AnimationWorkspace")) {
        if(ImGui::BeginTabItem("Graph")){drawGraph();ImGui::EndTabItem();}
        if(ImGui::BeginTabItem("Retarget")){drawMapping(ctx);ImGui::EndTabItem();}
        ImGui::EndTabBar();
    }
}
}
