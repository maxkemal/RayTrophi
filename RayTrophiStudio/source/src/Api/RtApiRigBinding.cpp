#include "RtApiInternal.h"
#include "Api/RtApiRigBinding.h"
#include "Animation/RigBinding.h"
#include "Animation/RigSelection.h"
#include "Animation/RigClipRuntimeSync.h"
#include "ProjectManager.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include "Renderer.h"
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <exception>
namespace {
class RigBindCommand final:public SceneCommand {
    RigAuthoring::RigBindState pending;
    std::string character;
    std::unordered_map<std::string,std::string>::node_type pose;
    std::unordered_set<std::string>::node_type wake;
    size_t triangles=0;
    std::string description;
    void exchange(UIContext& ctx) {
        prepare(ctx); // All fallible allocations precede canonical state changes.
        auto oldPose=ctx.scene.rigView.pose_views.extract(character);
        if(!pose.empty())ctx.scene.rigView.pose_views.insert(std::move(pose));pose=std::move(oldPose);
        auto inserted=ctx.scene.rigView.pose_view_dirty.insert(std::move(wake));wake=std::move(inserted.node);
        for(auto& model:ctx.scene.importedModelContexts)if(model.importName==character){
            std::swap(model,pending.rig.model);
            if(ctx.scene.rigView.pose.active &&
               ctx.scene.rigView.pose.character==character) {
                ctx.scene.rigView.pose.revision=model.rigRevision;
                ctx.scene.rigView.pose.invalidateEvaluation();
            }
            break;
        }
        for(auto& membership:pending.memberships)
            for(auto& model:ctx.scene.importedModelContexts)if(model.importName==membership.character){model.members.swap(membership.members);break;}
        std::swap(ctx.scene.boneData,pending.rig.bones);
        auto& view=ctx.scene.rigView;
        RigAuthoring::exchangeSelection(view,pending.view);
        std::swap(view.edit_mode,pending.view.edit_mode);std::swap(view.edit_character,pending.view.edit_character);
        for(auto& part:pending.parts) {
            part.mesh->geometry.swap(part.geometry);part.mesh->transform.swap(part.transform);
            part.mesh->local_bvh.reset();part.mesh->pointiness.clear();part.mesh->material_attribs.clear();
            auto& geometry=*part.mesh->geometry;geometry.last_skinned_pose_hash=0;
            // Redo may hold a previously animated buffer. Start in canonical Rest.
            if(part.mesh->hasSkinWeights()) {
                const auto* bindP=geometry.get_positions_orig();const auto* bindN=geometry.get_normals_orig();
                auto* p=geometry.get_positions_mut();auto* n=geometry.get_normals_mut();
                for(size_t i=0;i<geometry.get_vertex_count();++i){p[i]=bindP[i];if(n&&bindN)n[i]=bindN[i];}
            }
        }
        for(const auto& node:pending.rig.model.nodeHierarchy.nodes) {
            const auto found=ctx.scene.boneData.boneNameToIndex.find(node.uniqueName);
            if(found!=ctx.scene.boneData.boneNameToIndex.end() && found->second<ctx.renderer.finalBoneMatrices.size())ctx.renderer.finalBoneMatrices[found->second]=Matrix4x4::identity();
        }
        ctx.renderer.invalidateAnimationGeometry();
        g_geometry_dirty=true;g_bvh_rebuild_pending=true;g_optix_rebuild_pending=true;
        g_vulkan_rebuild_pending=true;g_viewport_raster_rebuild_pending=true;
        ui.mesh_cache_valid=false;ProjectManager::getInstance().markModified();
        RigAuthoring::synchronizeClipRuntime(ctx.scene,character);
        // Canonical mutation cannot fail because an optional immediate refresh fails.
        try{scheduleSceneMutationRebuilds(ctx,true);}catch(...){g_scene_geometry_generation.fetch_add(1,std::memory_order_release);}
    }
public:
    RigBindCommand(RigAuthoring::RigBindState state,std::string label):pending(std::move(state)),character(pending.rig.model.importName),description(std::move(label)) {
        pose=pending.view.pose_views.extract(character);
        wake=pending.view.pose_view_dirty.extract(character);
        for(const auto& part:pending.parts)triangles+=part.mesh->num_triangles();
    }
    void prepare(UIContext& ctx) {
        ctx.scene.rigView.pose_views.reserve(ctx.scene.rigView.pose_views.size()+1);
        ctx.scene.rigView.pose_view_dirty.reserve(ctx.scene.rigView.pose_view_dirty.size()+1);
        if(wake.empty()){std::unordered_set<std::string> temporary;temporary.insert(character);wake=temporary.extract(character);}
        if(ctx.renderer.finalBoneMatrices.size()<ctx.scene.boneData.getBoneIndexCapacity())ctx.renderer.finalBoneMatrices.resize(ctx.scene.boneData.getBoneIndexCapacity(),Matrix4x4::identity());
    }
    void execute(UIContext& ctx)override{exchange(ctx);}
    void undo(UIContext& ctx)override{exchange(ctx);}
    Type getType()const override{return Type::Heavy;}
    bool isHeavyGeometry()const override{return true;}
    size_t getTriangleCount()const override{return triangles;}
    std::string getDescription()const override{return description+character;}
};
}
namespace rtapi {
Result previewRigMeshBinding(const std::string& character,const std::string& mesh,bool confirmed,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::previewMeshBinding(g_ctx->scene,character,mesh,confirmed,output,error)?Result::success():Result::fail(error);}
    catch(const std::exception&){return Result::fail("rig_bind_failed");}
}
Result getRigMeshBinding(const std::string& character,nlohmann::json& output) {
    output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
    try{std::string error;return RigAuthoring::meshBindingInfo(g_ctx->scene,character,output,error)?Result::success():Result::fail(error);}
    catch(const std::exception&){return Result::fail("rig_bind_failed");}
}
Result bindRigMesh(const std::string& character,const std::string& mesh,const nlohmann::json& preview) {
    if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");if(!g_history)return Result::fail("history_not_bound");
    try {
        RigAuthoring::RigBindState state;std::string error;
        if(!RigAuthoring::stageMeshBinding(g_ctx->scene,character,mesh,preview,state,error))return Result::fail(error);
        auto command=std::make_unique<RigBindCommand>(std::move(state),"Bind mesh to rig: ");auto* recorded=command.get();
        recorded->prepare(*g_ctx);g_history->record(std::move(command));recorded->execute(*g_ctx);
        return Result::success();
    }catch(const std::exception&){return Result::fail("rig_bind_failed");}
}
Result unbindRigMesh(const std::string& character) {
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    if(!g_history)return Result::fail("history_not_bound");
    try {
        RigAuthoring::RigBindState state;std::string error;
        if(!RigAuthoring::stageMeshUnbinding(g_ctx->scene,character,state,error))
            return Result::fail(error);
        auto command=std::make_unique<RigBindCommand>(std::move(state),"Unbind mesh from rig: ");
        auto* recorded=command.get();
        recorded->prepare(*g_ctx);g_history->record(std::move(command));recorded->execute(*g_ctx);
        return Result::success();
    }catch(const std::exception&){return Result::fail("rig_unbind_failed");}
}
}
