#include "RtApiInternal.h"
#include "Api/RtApiRigJointProfile.h"
#include "Animation/RigJointRules.h"
#include "Animation/RigPosePreview.h"
#include "ProjectManager.h"
#include "Backend/IBackend.h"
#include <algorithm>
#include <limits>
namespace rtapi {
namespace {
void wakeProfile(UIContext& ctx){auto& pose=ctx.scene.rigView.pose;pose.preview.clear();pose.hasPreview=false;pose.limitHits.clear();++pose.serial;pose.invalidateEvaluation();ctx.start_render=true;ctx.renderer.resetCPUAccumulation();if(ctx.backend_ptr)ctx.backend_ptr->resetAccumulation();}
class ProfileCommand final:public SceneCommand {
 std::string character;std::vector<RigAuthoring::JointRule> pending;uint64_t revision;
 void exchange(UIContext& ctx){for(auto& model:ctx.scene.importedModelContexts)if(model.importName==character){model.rigAnatomy.joints.swap(pending);std::swap(model.rigRevision,revision);wakeProfile(ctx);ProjectManager::getInstance().markModified();break;}}
public:
 ProfileCommand(std::string name,std::vector<RigAuthoring::JointRule> rules,uint64_t next):character(std::move(name)),pending(std::move(rules)),revision(next){}
 void execute(UIContext& ctx)override{exchange(ctx);}void undo(UIContext& ctx)override{exchange(ctx);}
 Type getType()const override{return Type::Generic;}std::string getDescription()const override{return "Rig joint profile";}
};
}
Result getRigJointProfile(const std::string& character,nlohmann::json& output){
 output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
 try{for(const auto& m:g_ctx->scene.importedModelContexts)if(m.importName==character){output={{"character",character},{"family",m.rigAnatomy.family},{"rig_revision",m.rigRevision},{"profile",RigAuthoring::serializeJointRules(m.rigAnatomy.joints)}};return Result::success();}return Result::fail("unknown_character");}catch(const std::exception&){return Result::fail("rig_joint_failed");}
}
Result suggestRigJointProfile(const std::string& character,nlohmann::json& output){
 output=nullptr;if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");
 try{for(const auto& m:g_ctx->scene.importedModelContexts)if(m.importName==character){
  std::vector<RigAuthoring::PreviewJoint> globals;std::string error;if(!RigAuthoring::sampleRigPose(m.nodeHierarchy,nullptr,0,globals,error))return Result::fail(error);
  std::vector<RigAuthoring::JointRule> rules;std::vector<std::string> seen;
  for(const auto& role:m.rigAnatomy.roles){
   if(std::find(seen.begin(),seen.end(),role.bone)!=seen.end())continue;seen.push_back(role.bone);
   const bool lower=role.role.find("_arm.lower")!=std::string::npos||role.role.find("_leg.lower")!=std::string::npos;
   const bool upper=role.role.find("_arm.upper")!=std::string::npos||role.role.find("_leg.upper")!=std::string::npos;
   if(!lower&&!upper)continue;
   const auto* node=m.nodeHierarchy.find(role.bone);if(!node)return Result::fail("unknown_bone");
   RigAuthoring::JointRule rule;rule.bone=role.bone;rule.type=lower?"hinge":"ball";rule.swing=lower?0:120;rule.minimum=-150;rule.maximum=150;
   const auto i=static_cast<size_t>(node-m.nodeHierarchy.nodes.data());const auto& world=globals[i].world;
   Vec3 axis(1,0,0);
   if(lower&&node->parent>=0){axis=Vec3(0,0,0);const auto& parent=globals[static_cast<size_t>(node->parent)].world;const Vec3 incoming(world.m[0][3]-parent.m[0][3],world.m[1][3]-parent.m[1][3],world.m[2][3]-parent.m[2][3]);
    for(const auto& child:m.nodeHierarchy.nodes)if(child.parent==static_cast<int>(i)){const auto ci=static_cast<size_t>(&child-m.nodeHierarchy.nodes.data());const auto& cw=globals[ci].world;axis=incoming.cross(Vec3(cw.m[0][3]-world.m[0][3],cw.m[1][3]-world.m[1][3],cw.m[2][3]-world.m[2][3]));break;}
    if(axis.length_squared()<1e-8f)axis=incoming.cross(Vec3(0,0,1));if(axis.length_squared()<1e-8f)axis=Vec3(1,0,0);
   }else{for(const auto& child:m.nodeHierarchy.nodes)if(child.parent==static_cast<int>(i)){axis=Vec3(child.localBind.m[0][3],child.localBind.m[1][3],child.localBind.m[2][3]);axis=Quaternion::fromMatrix(world).rotate(axis);break;}}
   if(axis.length_squared()<1e-8f)axis=Vec3(1,0,0);auto rotation=Quaternion::fromMatrix(world);rotation.normalize();rule.axis=rotation.conjugate().rotate(axis.normalize()).normalize();
   rules.push_back(rule); // Proposed axes/ranges are disabled until reviewed.
  }
  if(!RigAuthoring::validateJointRules(rules,m.nodeHierarchy,error))return Result::fail(error);
  output={{"character",character},{"family",m.rigAnatomy.family},{"rig_revision",m.rigRevision},{"needs_review",true},{"profile",RigAuthoring::serializeJointRules(rules)}};return Result::success();
 }return Result::fail("unknown_character");}catch(const std::exception&){return Result::fail("rig_joint_failed");}
}
Result setRigJointProfile(const std::string& character,const nlohmann::json& profile,uint64_t expected){
 if(!g_ctx)return Result::fail("api_not_bound");if(renderJobActive())return Result::fail("scene_locked");if(!g_history)return Result::fail("history_not_bound");
 try{for(const auto& m:g_ctx->scene.importedModelContexts)if(m.importName==character){
  if(!m.authoringOwned)return Result::fail("rig_not_owned");if(m.rigRevision!=expected)return Result::fail("rig_edit_stale_revision");if(expected==std::numeric_limits<uint64_t>::max())return Result::fail("rig_revision_overflow");
  if(g_ctx->scene.rigView.pose.active&&g_ctx->scene.rigView.pose.hasPreview)return Result::fail("rig_pose_preview_active");
  std::vector<RigAuthoring::JointRule> staged;std::string error;if(!RigAuthoring::deserializeJointRules(profile,m.nodeHierarchy,staged,error))return Result::fail(error);
  if(RigAuthoring::serializeJointRules(staged)==RigAuthoring::serializeJointRules(m.rigAnatomy.joints))return Result::fail("rig_joint_no_change");
  auto command=std::make_unique<ProfileCommand>(character,std::move(staged),expected+1);auto* raw=command.get();g_history->record(std::move(command));raw->execute(*g_ctx);return Result::success();
 }return Result::fail("unknown_character");}catch(const std::exception&){return Result::fail("rig_joint_failed");}
}
}
