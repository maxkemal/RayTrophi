#include "RtApiInternal.h"
#include "Api/RtApiRigPoseView.h"
#include "Animation/RigPoseView.h"
#include "Backend/IBackend.h"
namespace rtapi {
Result setRigPoseView(const std::string& character,const std::string& mode) {
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    const auto previous=g_ctx->scene.rigView.pose_views.find(character);
    const bool changed=(previous==g_ctx->scene.rigView.pose_views.end()?"animated":previous->second)!=mode;
    std::string error;
    if(!RigAuthoring::setPoseView(g_ctx->scene,character,mode,error))return Result::fail(error);
    if(changed) {
        g_ctx->start_render=true;
        g_ctx->renderer.resetCPUAccumulation();
        if(g_ctx->backend_ptr)g_ctx->backend_ptr->resetAccumulation();
    }
    return Result::success();
}
Result getRigPoseView(const std::string& character,std::string& mode,std::string& effectiveMode) {
    if(!g_ctx)return Result::fail("api_not_bound");
    if(renderJobActive())return Result::fail("scene_locked");
    for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character) {
        const auto found=g_ctx->scene.rigView.pose_views.find(character);
        mode=found==g_ctx->scene.rigView.pose_views.end()?"animated":found->second;
        effectiveMode=RigAuthoring::isRestPoseView(g_ctx->scene,character)?"rest":"animated";
        return Result::success();
    }
    return Result::fail("unknown_character");
}
}
