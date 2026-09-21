#include "RtApiInternal.h"
#include "Api/RtApiRig.h"
#include "Animation/RigEditing.h"
namespace rtapi {
namespace { Result rigNotBound() { return Result::fail("api_not_bound"); } }
Result setRigMode(const std::string& mode,const std::string& character) {
    if(!g_ctx)return rigNotBound();
    if(renderJobActive())return Result::fail("scene_locked");
    if((mode=="edit" || mode=="pose") && (ui.sculpt_mode_state.enabled || ui.paint_mode_state.enabled || ui.mesh_overlay_settings.edit_mode))
        return Result::fail("viewport_edit_mode_conflict");
    if(mode=="pose" && (g_ctx->render_settings.animation_fps<1 || g_ctx->render_settings.animation_fps>240))return Result::fail("invalid_clip_timing");
    try {
    std::string error;
    if(!RigAuthoring::setInteractionMode(g_ctx->scene,mode,character,error))return Result::fail(error);
    if(mode=="edit" || mode=="pose")ui.active_properties_tab=14;
    if(mode=="pose"){ui.timeline.pausePlayback();g_ctx->scene.rigView.pose.fps=static_cast<float>(g_ctx->render_settings.animation_fps);setAnimPaused(character,true);g_ctx->selection.clearSelection();g_ctx->start_render=true;}
    if(mode=="scene")g_ctx->start_render=true;
    return Result::success();
    }catch(const std::exception&){return Result::fail("rig_mode_failed");}
}
Result getRigMode(std::string& mode,std::string& character) {
    if(!g_ctx)return rigNotBound();
    if(renderJobActive())return Result::fail("scene_locked");
    mode=g_ctx->scene.rigView.pose.active?"pose":(g_ctx->scene.rigView.edit_mode?"edit":"scene");character=g_ctx->scene.rigView.pose.active?g_ctx->scene.rigView.pose.character:g_ctx->scene.rigView.edit_character;
    return Result::success();
}
Result listRigCharacters(std::vector<std::string>& out) {
    out.clear(); if (!g_ctx) return rigNotBound(); if (renderJobActive()) return Result::fail("scene_locked");
    out = RigAuthoring::listCharacters(g_ctx->scene); return Result::success();
}
Result listRigBones(const std::string& character, std::vector<RigAuthoring::BoneView>& out) {
    out.clear();
    if (!g_ctx) return rigNotBound();
    if (renderJobActive()) return Result::fail("scene_locked");
    std::string error;
    return RigAuthoring::listBones(g_ctx->scene, character, out, error) ? Result::success() : Result::fail(error);
}
Result selectRigBone(const std::string& character, const std::string& bone) {
    if (!g_ctx) return rigNotBound();
    if (renderJobActive()) return Result::fail("scene_locked");
    std::string error;
    if(!RigAuthoring::selectBone(g_ctx->scene, character, bone, error))return Result::fail(error);
    g_ctx->selection.clearSelection(); // Bone gizmo owns viewport selection until another scene object is picked.
    return Result::success();
}
Result clearRigSelection() {
    if (!g_ctx) return rigNotBound();
    if (renderJobActive()) return Result::fail("scene_locked");
    RigAuthoring::clearSelection(g_ctx->scene);
    return Result::success();
}
Result getSelectedRigBone(RigAuthoring::BoneView& out, bool& hasSelection) {
    hasSelection = false;
    if (!g_ctx) return rigNotBound();
    if (renderJobActive()) return Result::fail("scene_locked");
    hasSelection = RigAuthoring::selectedBone(g_ctx->scene, out);
    return Result::success();
}
Result getRigOverlayVisible(bool& out) { if (!g_ctx) return rigNotBound(); out = g_ctx->scene.rigView.visible; return Result::success(); }
Result setRigOverlayVisible(bool visible) { if (!g_ctx) return rigNotBound(); if(!visible && (g_ctx->scene.rigView.edit_mode||g_ctx->scene.rigView.pose.active))return Result::fail("rig_edit_overlay_required"); g_ctx->scene.rigView.visible = visible; return Result::success(); }
}
