#include "Animation/RigPoseView.h"
#include "Animation/RigPoseAuthoring.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigBindingScope.h"
#include "TriangleMesh.h"
#include <cmath>
#include <unordered_map>

namespace RigAuthoring {
namespace {
bool restSkin(const SceneData& scene,const SceneData::ImportedModelContext& model,
              std::vector<PreviewJoint>& joints,std::unordered_map<unsigned int,Matrix4x4>& matrices,std::string& error) {
    if(model.nodeHierarchy.empty() || !model.hasSkeletonRepresentation){error="rig_pose_view_requires_hierarchy";return false;}
    if(!sampleRigPose(model.nodeHierarchy,nullptr,0,joints,error))return false;
    std::unordered_map<std::string,Matrix4x4> globals;
    for(const auto& joint:joints)globals.emplace(joint.name,joint.world);
    auto inverse=model.globalInverseTransform;
    const auto inv=scene.boneData.perModelInverses.find(model.importName);
    if(inv!=scene.boneData.perModelInverses.end())inverse=inv->second;
    const auto prefix=model.importName+"_";
    for(const auto& entry:scene.boneData.boneNameToIndex)if(entry.first.find(prefix)==0) {
        const auto world=globals.find(entry.first);
        const auto offset=scene.boneData.boneOffsetMatrices.find(entry.first);
        if(world==globals.end() || offset==scene.boneData.boneOffsetMatrices.end()) {
            if(scene.boneData.weightedBoneNames.count(entry.first)){error="rig_pose_view_incomplete_bind";return false;}
            continue;
        }
        const auto skin=inverse*world->second*offset->second;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(skin.m[r][c])){error="rig_pose_view_invalid_bind";return false;}
        matrices.emplace(entry.second,skin);
    }
    return true;
}
void skinFlatMeshes(SceneData& scene,const std::string& character,const std::vector<Matrix4x4>& skin) {
    for(const auto& object:scene.world.objects) {
        const auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(mesh && meshBelongsToRig(scene,character,*mesh) && mesh->hasSkinWeights())mesh->applySkinning(skin);
    }
}
}
bool synchronizeCpuSkinning(SceneData& scene,const std::vector<Matrix4x4>& skin) {
    if(skin.empty())return false;
    bool changed=false;
    for(const auto& object:scene.world.objects) {
        const auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(!mesh || !mesh->hasSkinWeights() || !mesh->geometry)continue;
        // Generic transform synchronization may have replaced P independently
        // of the pose hash. Rebuild from P_orig and the current pose once.
        mesh->geometry->last_skinned_pose_hash=0;
        changed=mesh->applySkinning(skin) || changed;
    }
    for(auto& model:scene.importedModelContexts)if(isRestPoseView(scene,model.importName)) {
        model.rigPoseViewCpuApplied=true;
        model.rigPoseViewCpuRestorePending=true;
    }
    return changed;
}
void initializeRestPoseDefaults(SceneData& scene) {
    for(const auto& model:scene.importedModelContexts) {
        if(scene.rigView.pose_views.count(model.importName))continue; // Preserve explicit views during append/reinitialization.
        bool weighted=model.weightedBoneCount!=0;
        if(!weighted)for(const auto& object:scene.world.objects) {
            auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
            if(mesh && meshBelongsToRig(scene,model.importName,*mesh) && mesh->hasSkinWeights()){weighted=true;break;}
        }
        if(weighted){std::string error;setPoseView(scene,model.importName,"rest",error);}
    }
}
bool needsFileAnimationEvaluation(const SceneData& scene) {
    const auto& pose=scene.rigView.pose;
    if(pose.needsEvaluation(scene.timeline.current_frame))return true;
    if(needsIKPlacementEvaluation(scene))return true;
    for(const auto& character:scene.rigView.pose_view_dirty)if(!pose.active || character!=pose.character)return true; // Apply a view transition once before idle.
    for(const auto& clip:scene.animationDataList)if(clip) {
        bool resolved=false;
        for(const auto& model:scene.importedModelContexts)
            if(clip->modelName==model.importName || (clip->modelName.empty() && scene.importedModelContexts.size()==1)) {
                resolved=true;if((!pose.active || model.importName!=pose.character) && !isRestPoseView(scene,model.importName))return true;break;
            }
        if(!resolved)return true; // Preserve legacy unowned animation evaluation.
    }
    return false;
}
bool isRestPoseView(const SceneData& scene,const std::string& character) {
    if(scene.rigView.pose.active && scene.rigView.pose.character==character)return false;
    if(scene.rigView.edit_mode && scene.rigView.edit_character==character)return true;
    const auto view=scene.rigView.pose_views.find(character);
    return view!=scene.rigView.pose_views.end() && view->second=="rest";
}
bool setPoseView(SceneData& scene,const std::string& character,const std::string& mode,std::string& error) {
    error.clear();
    if(scene.rigView.pose.active && scene.rigView.pose.character==character){error="rig_pose_mode_active";return false;}
    if(mode!="rest" && mode!="animated"){error="invalid_rig_pose_view";return false;}
    for(auto& model:scene.importedModelContexts)if(model.importName==character) {
        std::vector<PreviewJoint> joints;std::unordered_map<unsigned int,Matrix4x4> matrices;
        if(mode=="rest" && !restSkin(scene,model,joints,matrices,error))return false;
        const auto found=scene.rigView.pose_views.find(character);
        const std::string previous=found==scene.rigView.pose_views.end()?"animated":found->second;
        if(previous==mode)return true;
        scene.rigView.pose_views[character]=mode;scene.rigView.pose_view_dirty.insert(character);
        model.restPoseApplied=false;model.rigPoseViewCpuApplied=false;return true;
    }
    error="unknown_character";return false;
}
bool applyRestPoseView(SceneData& scene,SceneData::ImportedModelContext& model,std::vector<Matrix4x4>& skin,bool cpu,bool& changed) {
    if(applyAuthoringPose(scene,model,skin,cpu,changed))return true;
    changed=false;if(!isRestPoseView(scene,model.importName))return false;
    const bool cached=!scene.rigView.pose_view_dirty.count(model.importName) && model.rigPoseSource=="rest_view" && (!cpu || model.rigPoseViewCpuApplied);
    std::vector<PreviewJoint> joints;std::unordered_map<unsigned int,Matrix4x4> matrices;std::string error;
    if(!restSkin(scene,model,joints,matrices,error))return true; // Never fall through into playback while Rest owns evaluation.
    if(skin.size()<scene.boneData.getBoneIndexCapacity())skin.resize(scene.boneData.getBoneIndexCapacity(),Matrix4x4::identity());
    // The renderer buffer can be rebuilt independently of the pose snapshot.
    // Restore only this character's slots even when Rest itself has not changed.
    bool matricesChanged=false;
    for(const auto& entry:matrices) {
        auto& target=skin[entry.first];
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)
            if(target.m[r][c]!=entry.second.m[r][c])matricesChanged=true;
        target=entry.second;
    }
    if(cached && !matricesChanged)return true;
    JointGlobals globals;for(const auto& joint:joints)globals.emplace(joint.name,joint.world);
    captureGlobals(scene,model.importName,globals,"rest_view");
    if(model.hasAnimation)for(const auto& object:scene.world.objects) {
        const auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(!mesh || mesh->hasSkinWeights() || !mesh->transform)continue;
        const auto rest=globals.find(mesh->nodeName);
        if(rest!=globals.end())mesh->transform->setBase(rest->second);
    }
    if(cpu || model.rigPoseViewCpuRestorePending){skinFlatMeshes(scene,model.importName,skin);model.rigPoseViewCpuRestorePending=false;}
    model.rigPoseViewCpuApplied=cpu;
    scene.rigView.pose_view_dirty.erase(model.importName);changed=true;return true;
}
bool applyPoseViewResume(SceneData& scene,SceneData::ImportedModelContext& model,const std::vector<Matrix4x4>& skin,bool cpu) {
    const bool dirty=scene.rigView.pose_view_dirty.erase(model.importName)!=0;
    if(!dirty && !(cpu && model.rigPoseViewCpuRestorePending))return false;
    if(cpu){skinFlatMeshes(scene,model.importName,skin);model.rigPoseViewCpuRestorePending=false;}
    model.rigPoseViewCpuApplied=false;return true;
}
}
