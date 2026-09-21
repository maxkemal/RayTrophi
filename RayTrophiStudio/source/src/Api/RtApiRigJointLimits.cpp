#include "RtApiInternal.h"
#include "Api/RtApiRigJointLimits.h"
#include "Animation/RigJointLimits.h"
namespace rtapi {
Result getRigJointLimitView(const std::string& character,const std::string& bone,nlohmann::json& output){
 output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
 try{RigAuthoring::JointLimitView view;std::string error;if(!RigAuthoring::getJointLimitView(g_ctx->scene,character,bone,view,error))return Result::fail(error);output=RigAuthoring::jointLimitViewJson(view);return Result::success();}catch(const std::exception&){return Result::fail("rig_joint_failed");}
}
Result setRigJointLimits(const std::string& character,const std::string& bone,float minimum,float maximum,float swing,uint64_t revision){
 if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
 try{for(const auto& model:g_ctx->scene.importedModelContexts)if(model.importName==character){
  if(!model.authoringOwned)return Result::fail("rig_not_owned");if(model.rigRevision!=revision)return Result::fail("rig_edit_stale_revision");
  std::vector<RigAuthoring::JointRule> staged;std::string error;if(!RigAuthoring::replaceJointLimits(model.rigAnatomy.joints,model.nodeHierarchy,bone,minimum,maximum,swing,staged,error))return Result::fail(error);
  return setRigJointProfile(character,RigAuthoring::serializeJointRules(staged),revision);
 }return Result::fail("unknown_character");}catch(const std::exception&){return Result::fail("rig_joint_failed");}
}
Result getRigJointLimitOverlay(nlohmann::json& output){output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");output={{"visible",g_ctx->scene.rigView.joint_limits_visible},{"edit",g_ctx->scene.rigView.joint_limits_edit},{"scope","active_joint"}};return Result::success();}
Result setRigJointLimitOverlay(bool visible,bool edit){
 if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
 if(edit&&!visible)return Result::fail("rig_joint_overlay_hidden");if(edit&&g_ctx->scene.rigView.pose.hasPreview)return Result::fail("rig_pose_preview_active");
 g_ctx->scene.rigView.joint_limits_visible=visible;g_ctx->scene.rigView.joint_limits_edit=edit;if(edit)ui.timeline.pausePlayback();g_ctx->start_render=true;return Result::success();
}
}
