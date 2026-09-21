#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigIKTimeline.h"
#include "Animation/RigPoseView.h"
#include "Animation/RigBindingScope.h"
#include "TriangleMesh.h"
#include <cmath>
namespace RigAuthoring {
namespace {
const AnimationData* selectedClip(const SceneData& scene,const std::string& character) {
 const auto key=scene.rigView.pose.clips.find(character);if(key==scene.rigView.pose.clips.end())return nullptr;
 for(const auto& clip:scene.animationDataList)if(clip&&clip->rigAuthoring&&clip->modelName==character&&clip->name==key->second)return clip.get();return nullptr;
}
}
bool canAuthorPose(const SceneData& scene,const std::string& character,std::string& error) {
    error.clear();
    if(scene.rigView.edit_mode) {
        error="rig_edit_active";
        return false;
    }
    for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
        if(!model.authoringOwned) {
            error="rig_pose_requires_owned_rig";
            return false;
        }
        if(model.nodeHierarchy.empty()||!model.hasSkeletonRepresentation) {
            error="character_has_no_skeleton";
            return false;
        }
        if(model.nodeHierarchy.size()>4096) {
            error="rig_pose_limit";
            return false;
        }
        bool deforming=false;
        for(const auto& object:scene.world.objects) {
            auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
            if(mesh&&meshBelongsToRig(scene,character,*mesh)&&mesh->hasSkinWeights()) {
                deforming=true;
                break;
            }
        }
        for(const auto& node:model.nodeHierarchy.nodes) {
            const auto index=scene.boneData.boneNameToIndex.find(node.uniqueName);
            const auto offset=scene.boneData.boneOffsetMatrices.find(node.uniqueName);
            if(index==scene.boneData.boneNameToIndex.end() ||
               (deforming&&offset==scene.boneData.boneOffsetMatrices.end())) {
                error="rig_pose_invalid_binding";
                return false;
            }
        }
        std::vector<PreviewJoint> joints;
        return sampleRigPose(model.nodeHierarchy,nullptr,0,joints,error);
    }
    error="unknown_character";
    return false;
}
bool enterPose(SceneData& scene,const std::string& character,std::string& error) {
 if(!canAuthorPose(scene,character,error))return false;
 auto view=scene.rigView;auto& state=view.pose;state.active=true;state.invalidateEvaluation();++state.serial;state.character=character;state.locals.clear();state.preview.clear();state.limitHits.clear();state.hasPreview=false;state.frame=scene.timeline.current_frame;
 state.ik.clear();state.previewIK.clear();
 state.control.clear();state.controlHandle="target";
 for(const auto& model:scene.importedModelContexts)if(model.importName==character)state.revision=model.rigRevision;
 if(!state.clips.count(character))for(const auto& clip:scene.animationDataList)if(clip&&clip->rigAuthoring&&clip->modelName==character){state.clips[character]=clip->name;break;}
 view.visible=true;
 if(view.character!=character)for(const auto& m:scene.importedModelContexts)if(m.importName==character){view.character=character;view.bone=m.nodeHierarchy.nodes.front().uniqueName;view.selected_bones={view.bone};view.selection_character=character;view.selection_anchor=view.bone;}
 scene.rigView=std::move(view);return true;
}
void leavePose(SceneData& scene) {
 auto& s=scene.rigView.pose;if(!s.character.empty())scene.rigView.pose_view_dirty.insert(s.character);s.active=false;++s.serial;s.locals.clear();s.preview.clear();s.limitHits.clear();s.hasPreview=false;
 s.ik.clear();s.previewIK.clear();
 s.control.clear();
}
void synchronizePoseFrame(SceneData& scene) {
 auto& s=scene.rigView.pose;if(s.frame!=scene.timeline.current_frame){s.locals.clear();s.preview.clear();s.previewIK.clear();for(auto i=s.ik.begin();i!=s.ik.end();)if(!i->second.contact)i=s.ik.erase(i);else ++i;s.limitHits.clear();s.hasPreview=false;s.frame=scene.timeline.current_frame;s.invalidateEvaluation();++s.serial;}
}
bool needsIKPlacementEvaluation(const SceneData& scene) {
 const auto& s=scene.rigView.pose;if(!s.active)return false;IKPoses effective;std::string controlError;if(!effectiveIKControls(scene,s.character,true,effective,controlError)||!hasEnabledIK(effective))return false;
 Matrix4x4 placement;std::string error;if(!rigScenePlacement(scene,s.character,placement,error))return false;
 if(!s.placementAcknowledged)return true;
 for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(placement.m[r][c]!=s.evaluatedPlacement.m[r][c])return true;
 return false;
}
bool currentPoseHierarchy(const SceneData& scene,const std::string& character,RayTrophi::NodeHierarchy& output,std::string& error,bool preview,std::vector<std::string>* limitHits,bool evaluateIK) {
 const auto& state=scene.rigView.pose;
 if(!std::isfinite(state.fps)||state.fps<=0||scene.timeline.current_frame<0){error="invalid_preview_time";return false;}
 for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
  const JointGlobals empty;const bool editing=state.active&&state.character==character&&state.frame==scene.timeline.current_frame;
  const auto& overrides=editing?(preview&&state.hasPreview?state.preview:state.locals):empty;
  RayTrophi::NodeHierarchy sampled;std::vector<std::string> hits;
  if(!poseHierarchy(model.nodeHierarchy,selectedClip(scene,character),double(scene.timeline.current_frame)/state.fps,overrides,sampled,error))return false;
  if(evaluateIK) {
   IKPoses controls;if(!effectiveIKControls(scene,character,preview,controls,error))return false;
   if(hasEnabledIK(controls)){Matrix4x4 placement;if(!rigScenePlacement(scene,character,placement,error))return false;RayTrophi::NodeHierarchy solved;if(!solveIKPose(sampled,model.rigAnatomy.controls,controls,placement,solved,error))return false;sampled=std::move(solved);}
  }
  if(!constrainJointPose(model.nodeHierarchy,model.rigAnatomy.joints,sampled,output,hits,error))return false;
  if(limitHits)*limitHits=std::move(hits);return true;
 }error="unknown_character";return false;
}
bool applyAuthoringPose(SceneData& scene,SceneData::ImportedModelContext& model,std::vector<Matrix4x4>& skin,bool cpu,bool& changed) {
 auto& state=scene.rigView.pose;const bool active=state.active&&state.character==model.importName;
 if(!active && (isRestPoseView(scene,model.importName)||!selectedClip(scene,model.importName)))return false;
 changed=false;synchronizePoseFrame(scene);RayTrophi::NodeHierarchy pose;std::string error;
 if(!currentPoseHierarchy(scene,model.importName,pose,error,true))return true;
 std::vector<PreviewJoint> joints;if(!sampleRigPose(pose,nullptr,0,joints,error))return true;
 if(skin.size()<scene.boneData.getBoneIndexCapacity())skin.resize(scene.boneData.getBoneIndexCapacity(),Matrix4x4::identity());
 auto inverse=model.globalInverseTransform;const auto found=scene.boneData.perModelInverses.find(model.importName);if(found!=scene.boneData.perModelInverses.end())inverse=found->second;
 JointGlobals globals;std::vector<std::pair<unsigned int,Matrix4x4>> matrices;
 for(const auto& joint:joints) {
  globals[joint.name]=joint.world;
  const auto i=scene.boneData.boneNameToIndex.find(joint.name);const auto offset=scene.boneData.boneOffsetMatrices.find(joint.name);
  if(i==scene.boneData.boneNameToIndex.end()||offset==scene.boneData.boneOffsetMatrices.end())continue;
  const auto matrix=inverse*joint.world*offset->second;
  for(int r=0;r<4;++r)for(int c=0;c<4;++c){if(!std::isfinite(matrix.m[r][c]))return true;changed=changed||skin[i->second].m[r][c]!=matrix.m[r][c];}
  matrices.emplace_back(i->second,matrix);
 }
 for(const auto& p:matrices)skin[p.first]=p.second;
 captureGlobals(scene,model.importName,globals,active?"pose":"authored_clip");
 (void)cpu; // Keep canonical flat CPU positions/picking and weight overlays in sync too.
 for(const auto& o:scene.world.objects){auto mesh=std::dynamic_pointer_cast<TriangleMesh>(o);if(mesh&&mesh->hasSkinWeights()&&meshBelongsToRig(scene,model.importName,*mesh))changed=mesh->applySkinning(skin)||changed;}
 model.rigPoseViewCpuApplied=false;model.rigPoseViewCpuRestorePending=true;
 if(active){state.acknowledgeEvaluation(scene.timeline.current_frame);state.placementAcknowledged=rigScenePlacement(scene,model.importName,state.evaluatedPlacement,error);scene.rigView.pose_view_dirty.erase(model.importName);}
 return true;
}
}
