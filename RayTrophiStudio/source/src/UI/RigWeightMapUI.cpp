#include "UI/RigWeightMapUI.h"
#include "UI/ScalarFieldOverlay.h"
#include "Api/RtApi.h"
#include "Api/RtApiRigWeightMap.h"
#include "Animation/RigWeights.h"
#include "Animation/RigBindingScope.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Camera.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include <cmath>
#include <unordered_set>
extern SceneUI ui;
namespace RigUI {
void drawRigWeightMapControls(UIContext& ctx) {
    static std::string message;bool visible=ctx.scene.rigView.weight_map_visible;
    if(ImGui::Checkbox("Show selected bone weights",&visible)){auto result=rtapi::setRigWeightMapVisible(visible);message=result.ok?"":result.error;}
    if(!message.empty())ImGui::TextWrapped("%s",message.c_str());
    if(visible) {
        ImGui::TextWrapped("Blue tint: stronger weight. Select another bone to change the map. Display only; sculpt protection mask stays separate.");
        ImGui::TextDisabled("Surface tint has no depth test; overlapping surfaces may show through.");
        ImGui::TextDisabled("Dense models may be partially displayed (2M vertices / 120k faces).");
        if(ctx.scene.rigView.bone.empty())ImGui::TextDisabled("Select a bone first.");
        else ImGui::TextWrapped("Bone: %s",ctx.scene.rigView.bone.c_str());
    }
}
void drawRigWeightMapOverlay(UIContext& ctx) {
    const auto& view=ctx.scene.rigView;
    if(!view.weight_map_visible || view.bone.empty() || view.character.empty() || !ctx.scene.camera)return;
    const auto& camera=*ctx.scene.camera;const auto size=ImGui::GetIO().DisplaySize;
    const auto forward=(camera.lookat-camera.lookfrom).normalize();const auto right=forward.cross(camera.vup).normalize();const auto up=right.cross(forward).normalize();
    const bool ortho=camera.orthographic && ui.viewport_settings.shading_mode!=2;
    auto project=[&](const Vec3& world,ImVec2& screen) {
        const auto delta=world-camera.lookfrom;const float depth=delta.dot(forward);
        if(!ortho && depth<=.1f)return false;
        const float h=ortho?camera.ortho_height*.5f:depth*std::tan(camera.vfov*3.14159265f/360.f);const float w=h*camera.aspect_ratio;
        if(!std::isfinite(h) || !std::isfinite(w) || h<=1e-6f || w<=1e-6f)return false;
        screen=ImVec2((delta.dot(right)/w*.5f+.5f)*size.x,(.5f-delta.dot(up)/h*.5f)*size.y);
        return std::isfinite(screen.x)&&std::isfinite(screen.y)&&screen.x>=-64&&screen.y>=-64&&screen.x<=size.x+64&&screen.y<=size.y+64;
    };
    auto* draw=ImGui::GetBackgroundDrawList();size_t examined=0,vertices=0;std::unordered_set<const TriangleMesh*> seen;
    for(const auto& object:ctx.scene.world.objects) {
        auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(!mesh || !mesh->visible || !mesh->geometry || !mesh->hasSkinWeights() || !RigAuthoring::meshBelongsToRig(ctx.scene,view.character,*mesh) || ctx.scene.isEditorPendingDeleteObjectName(mesh->nodeName))continue;
        if(!seen.insert(mesh.get()).second)continue;
        const auto count=mesh->num_vertices();if(count>2000000-vertices)continue;vertices+=count;
        RigAuthoring::BoneWeightField field;std::string error;
        if(!RigAuthoring::boneWeightField(ctx.scene,mesh->nodeName,view.character,view.bone,field,error))continue;
        const auto& geometry=*mesh->geometry;const auto* p=geometry.get_positions();const auto* n=geometry.get_normals();
        if(!p || geometry.get_core_attribute_count(DNA::Attr::P)<count)continue;
        const auto matrix=mesh->transform?mesh->transform->getFinal():Matrix4x4::identity();
        const auto normal=mesh->transform?mesh->transform->getNormalTransform():Matrix4x4::identity();
        for(size_t i=0;i+2<geometry.indices.size();i+=3) {
            if(++examined>120000)return;
            const auto a=geometry.indices[i],b=geometry.indices[i+1],c=geometry.indices[i+2];if(a>=count||b>=count||c>=count)continue;
            if(field.values[a]<=.02f && field.values[b]<=.02f && field.values[c]<=.02f)continue;
            const auto wa=matrix.transform_point(p[a]);
            if(n && geometry.get_core_attribute_count(DNA::Attr::N)>=count && normal.transform_vector(n[a]).dot(camera.lookfrom-wa)<0)continue;
            ImVec2 sa,sb,sc;if(!project(wa,sa)||!project(matrix.transform_point(p[b]),sb)||!project(matrix.transform_point(p[c]),sc))continue;
            ScalarFieldOverlay::triangle(draw,sa,sb,sc,field.values[a],field.values[b],field.values[c]);
        }
    }
}
}
