#include "UI/RigViewportUI.h"
#include "UI/RigIKUI.h"
#include "Api/RtApi.h"
#include "Animation/RigEditing.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigSelection.h"
#include "Animation/RigBatchRest.h"
#include "Animation/RigBindMath.h"
#include <algorithm>
#include "Animation/RigPosePreview.h"
#include "scene_ui.h"
#include "globals.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include <cmath>
using namespace RayTrophi;

namespace RigUI {
namespace {
struct Drag {
    bool active=false, blocked=false,editing=false,posing=false;
    int load=-1;
    uint64_t revision=0,poseSerial=0;int frame=0;
    TransformMode operation=TransformMode::Translate;
    TransformSpace space=TransformSpace::World;
    std::string error;
    std::string character,bone,pivot;
    std::vector<std::string> bones;
    Matrix4x4 initialGizmo,delta=Matrix4x4::identity();
    Matrix4x4 world,parent=Matrix4x4::identity(),initialLocal,placement=Matrix4x4::identity();
    NodeHierarchy hierarchy;
    std::vector<RigAuthoring::PreviewJoint> preview;
} drag;
void cancel() {
    if(drag.active || ImGuizmo::IsUsing()){ImGuizmo::Enable(false);ImGuizmo::Enable(true);}
    if(drag.active && drag.posing)rtapi::cancelRigPosePreview(drag.character);
    drag.active=false;drag.preview.clear();drag.blocked=ImGui::IsMouseDown(0);
}
bool finite(const Matrix4x4& m) {
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(m.m[r][c]))return false;
    return true;
}
}
void applyRigDragPreview(const SceneData& scene,std::vector<RigAuthoring::BoneView>& bones) {
    if(!drag.active || (scene.rigView.edit_mode || scene.rigView.pose.active)!=drag.editing || scene.load_counter!=drag.load ||
       scene.rigView.character!=drag.character || scene.rigView.bone!=drag.bone ||
       (drag.editing && (scene.rigView.selection_pivot!=drag.pivot || RigAuthoring::selectedBones(scene)!=drag.bones)))return;
    for(auto& bone:bones)if(bone.character==drag.character && bone.rig_revision==drag.revision) {
        if(!drag.editing)bone.world=drag.world*drag.placement.inverse()*bone.world;
        else for(const auto& joint:drag.preview)if(joint.name==bone.name){bone.world=drag.placement*joint.world;break;}
    }
}
void drawRigRestGizmo(UIContext& ctx,int shadingMode,bool enabled,bool& hit) {
    const bool posing=ctx.scene.rigView.pose.active;
    if(posing)RigAuthoring::synchronizePoseFrame(ctx.scene);
    const bool editing=ctx.scene.rigView.edit_mode || posing;
    const bool available=(editing || !ctx.selection.hasSelection()) && enabled && ctx.scene.camera &&
        ctx.scene.rigView.visible && !rtapi::renderOutputPending() &&
        rtapi::renderStatus().state!=rtapi::RenderJobState::Rendering;
    if(ctx.scene.rigView.joint_limits_visible&&ctx.scene.rigView.joint_limits_edit){if(drag.active)cancel();drawRigIKGizmo(ctx,shadingMode,false,hit);return;}
    if(drawRigIKGizmo(ctx,shadingMode,available,hit)){if(drag.active)cancel();return;}
    if(!available){if(drag.active)cancel();return;}
    ImGuizmo::BeginFrame();
    if(!drag.error.empty())ImGui::GetForegroundDrawList()->AddText(ImVec2(30,90),IM_COL32(255,120,80,255),drag.error.c_str());
    auto& io=ImGui::GetIO();
    // Floating authoring windows own their mouse gestures.
    // ImGuizmo requests WantCaptureMouse for its own hover/drag. Testing that
    // flag here hides the gizmo on the next frame and can prevent drag startup.
    if(!drag.active && ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))return;
    if(drag.blocked){if(!ImGui::IsMouseDown(0))drag.blocked=false;else return;}
    RigAuthoring::BoneView bone;
    if(!RigAuthoring::selectedBone(ctx.scene,bone) || (editing && bone.character!=(posing?ctx.scene.rigView.pose.character:ctx.scene.rigView.edit_character))){if(drag.active)cancel();return;}
    std::string error;
    if(!(editing?(posing?RigAuthoring::canAuthorPose(ctx.scene,bone.character,error):RigAuthoring::canEditRig(ctx.scene,bone.character,error)):RigAuthoring::canPlaceRig(ctx.scene,bone.character,error))){if(drag.active)cancel();return;}
    if(drag.active && (drag.editing!=editing || drag.posing!=posing || (posing && (drag.frame!=ctx.scene.timeline.current_frame || drag.poseSerial!=ctx.scene.rigView.pose.serial)) || drag.load!=ctx.scene.load_counter || drag.revision!=bone.rig_revision ||
        drag.character!=bone.character || drag.bone!=bone.name ||
        drag.operation!=ctx.selection.transform_mode || drag.space!=ctx.selection.transform_space ||
        (editing && (drag.pivot!=ctx.scene.rigView.selection_pivot || drag.bones!=RigAuthoring::selectedBones(ctx.scene))))){cancel();return;}
    if(drag.active && ImGui::IsKeyPressed(ImGuiKey_Escape)){cancel();hit=true;return;}
    if(!drag.active && !io.WantCaptureKeyboard && !io.WantTextInput && !io.KeyCtrl && !io.KeyAlt) {
        if(ImGui::IsKeyPressed(ImGuiKey_G))ctx.selection.transform_mode=TransformMode::Translate;
        else if(ImGui::IsKeyPressed(ImGuiKey_R))ctx.selection.transform_mode=TransformMode::Rotate;
        else if(!editing && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_S))ctx.selection.transform_mode=TransformMode::Scale;
    }
    if(editing && ctx.selection.transform_mode==TransformMode::Scale){
        if(drag.active){cancel();return;}
        ctx.selection.transform_mode=TransformMode::Translate;
    }
    const Camera& cam=*ctx.scene.camera;
    const Vec3 f=(cam.lookat-cam.lookfrom).normalize(),r=f.cross(cam.vup).normalize(),u=r.cross(f);
    const float view[16]={r.x,u.x,-f.x,0,r.y,u.y,-f.y,0,r.z,u.z,-f.z,0,
        -r.dot(cam.lookfrom),-u.dot(cam.lookfrom),f.dot(cam.lookfrom),1};
    const bool ortho=cam.orthographic && shadingMode!=2;
    const float aspect=image_height>0?float(image_width)/image_height:io.DisplaySize.x/io.DisplaySize.y;
    if(!std::isfinite(aspect) || aspect<=0)return;
    constexpr float nearZ=.1f,farZ=10000.f;
    float projection[16]={};
    if(ortho) {
        const float h=cam.ortho_height>1e-4f?cam.ortho_height:10.f;
        projection[0]=2/(h*aspect);projection[5]=2/h;projection[10]=-2/(farZ-nearZ);
        projection[14]=-(farZ+nearZ)/(farZ-nearZ);projection[15]=1;
    } else {
        const float tangent=std::tan(cam.vfov*3.14159265359f/360.f);
        if(!std::isfinite(tangent) || tangent<=0)return;
        projection[0]=1/(aspect*tangent);projection[5]=1/tangent;
        projection[10]=-(farZ+nearZ)/(farZ-nearZ);projection[11]=-1;
        projection[14]=-2*farZ*nearZ/(farZ-nearZ);
    }
    Matrix4x4 world=drag.active?drag.world:(editing?bone.world:bone.scene_transform);
    if(editing && !drag.active && ctx.scene.rigView.selection_pivot=="center") {
        std::vector<RigAuthoring::BoneView> views;if(!RigAuthoring::listBones(ctx.scene,bone.character,views,error))return;
        Vec3 center(0,0,0);size_t count=0;
        for(const auto& b:views)if(RigAuthoring::isBoneSelected(ctx.scene.rigView,b.character,b.name)){center+=Vec3(b.world.m[0][3],b.world.m[1][3],b.world.m[2][3]);++count;}
        if(!count)return;center=center/static_cast<float>(count);
        world.m[0][3]=center.x;world.m[1][3]=center.y;world.m[2][3]=center.z;
    }
    float matrix[16];for(int row=0;row<4;++row)for(int col=0;col<4;++col)matrix[col*4+row]=world.m[row][col];
    ImGuizmo::SetOrthographic(ortho);ImGuizmo::SetRect(0,0,io.DisplaySize.x,io.DisplaySize.y);
    const bool scaling=ctx.selection.transform_mode==TransformMode::Scale;
    const auto operation=scaling?ImGuizmo::SCALE:(ctx.selection.transform_mode==TransformMode::Rotate?ImGuizmo::ROTATE:ImGuizmo::TRANSLATE);
    ImGuizmo::Manipulate(view,projection,operation,
        (scaling || ctx.selection.transform_space==TransformSpace::Local)?ImGuizmo::LOCAL:ImGuizmo::WORLD,matrix);
    const bool usingNow=ImGuizmo::IsUsing();
    hit=hit || usingNow || ImGuizmo::IsOver();
    if(usingNow && !drag.active) {
        drag.initialGizmo=world;drag.delta=Matrix4x4::identity();drag.bones=RigAuthoring::selectedBones(ctx.scene);drag.pivot=ctx.scene.rigView.selection_pivot;
        drag.error.clear();drag.operation=ctx.selection.transform_mode;drag.space=ctx.selection.transform_space;
        drag.character=bone.character;drag.bone=bone.name;drag.load=ctx.scene.load_counter;
        drag.editing=editing;drag.posing=posing;drag.frame=ctx.scene.timeline.current_frame;drag.poseSerial=ctx.scene.rigView.pose.serial;drag.revision=bone.rig_revision;drag.placement=bone.scene_transform;
        drag.initialLocal=editing?bone.local_rest:bone.scene_transform;drag.parent=editing?bone.scene_transform:Matrix4x4::identity();
        std::vector<RigAuthoring::BoneView> bones;
        if(!RigAuthoring::listBones(ctx.scene,bone.character,bones,error)){cancel();return;}
        bool parentFound=!editing || bone.parent.empty();
        for(const auto& b:bones)if(editing && b.name==bone.parent){drag.parent=b.world;parentFound=true;break;}
        if(!parentFound){cancel();return;}
        if(posing){if(!RigAuthoring::currentPoseHierarchy(ctx.scene,bone.character,drag.hierarchy,error)){cancel();return;}}
        else for(const auto& model:ctx.scene.importedModelContexts)if(model.importName==bone.character)drag.hierarchy=model.nodeHierarchy;
        drag.active=true;
    }
    if(!drag.active)return;
    for(int row=0;row<4;++row)for(int col=0;col<4;++col)drag.world.m[row][col]=matrix[col*4+row];
    if(scaling) {
        // Every axis handle adjusts the same actor scale, preserving anatomy proportions.
        float factor=1.f,largestChange=0.f;
        for(int col=0;col<3;++col) {
            float dot=0.f,length2=0.f;
            for(int row=0;row<3;++row) {
                dot+=drag.world.m[row][col]*drag.placement.m[row][col];
                length2+=drag.placement.m[row][col]*drag.placement.m[row][col];
            }
            if(!std::isfinite(length2) || length2<=0){cancel();return;}
            const float ratio=dot/length2;
            if(std::fabs(ratio-1.f)>largestChange){factor=ratio;largestChange=std::fabs(ratio-1.f);}
        }
        if(!std::isfinite(factor) || factor<=0){cancel();return;}
        for(int row=0;row<3;++row)for(int col=0;col<3;++col)
            drag.world.m[row][col]=drag.placement.m[row][col]*factor;
    }
    const auto local=drag.parent.inverse()*drag.world;
    if(!finite(local)){cancel();return;}
    if(editing) {
        Matrix4x4 inverse;if(!RigAuthoring::bindAffineInverse(drag.initialGizmo,inverse)){cancel();return;}
        drag.delta=drag.world*inverse;NodeHierarchy staged;
        if(!RigAuthoring::transformRestHierarchy(drag.hierarchy,drag.placement,drag.bones,drag.delta,staged,error) ||
           !RigAuthoring::sampleRigPose(staged,nullptr,0,drag.preview,error)){drag.error=error;cancel();return;}
        if(posing){const auto r=rtapi::previewRigPoseTransform(drag.character,drag.bones,drag.delta,drag.revision);if(!r.ok){drag.error=r.error;cancel();return;}NodeHierarchy actual;if(!RigAuthoring::currentPoseHierarchy(ctx.scene,drag.character,actual,error,true)||!RigAuthoring::sampleRigPose(actual,nullptr,0,drag.preview,error)){drag.error=error;cancel();return;}}
    }
    if(!usingNow) {
        bool changed=false;
        for(int row=0;row<4;++row)for(int col=0;col<4;++col)
            changed=changed || std::fabs(editing?drag.delta.m[row][col]-(row==col?1.f:0.f):local.m[row][col]-drag.initialLocal.m[row][col])>1e-6f;
        if(changed) {
            const auto result=editing?(posing?rtapi::applyRigPosePreview(drag.character):rtapi::transformRigRest(drag.character,drag.bones,drag.delta,drag.revision)):rtapi::setRigSceneTransform(drag.character,local);
            if(!result.ok){drag.error=result.error;if(posing)rtapi::cancelRigPosePreview(drag.character);}
        }
        if(posing && !changed)rtapi::cancelRigPosePreview(drag.character);
        drag.active=false;drag.preview.clear();
    }
}
}
