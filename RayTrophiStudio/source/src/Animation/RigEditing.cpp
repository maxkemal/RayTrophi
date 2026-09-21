#include "Animation/RigEditing.h"
#include "Animation/RigBatchRest.h"
#include "Animation/RigWeights.h"
#include "Animation/RigBindingScope.h"
#include "Animation/RigSerialization.h"
#include "Animation/RigAnatomy.h"
#include "Animation/RigTemplates.h"
#include "Animation/RigFitting.h"
#include "Animation/RigPoseAuthoring.h"
#include "json.hpp"
#include "Animation/RigPosePreview.h"
#include "OzzRuntime.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <unordered_set>
using namespace RayTrophi;

namespace RigAuthoring {
namespace {
bool validName(const std::string& name) {
    if(name.empty() || name.size()>128)return false;
    for(unsigned char c:name)if(!((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'))return false;
    return true;
}
bool rigid(const Matrix4x4& m,std::string& error) {
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(m.m[r][c])){error="invalid_rest_transform";return false;}
    Vec3 p,s;Quaternion q;RayTrophi::decomposeTRS(m,p,q,s);q.normalize();
    const auto rebuilt=Matrix4x4::translation(p)*q.toMatrix();
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(std::fabs(m.m[r][c]-rebuilt.m[r][c])>1e-4f){error="rig_rest_requires_rigid_transform";return false;}
    return true;
}
// Actor placement allows positive uniform scale; joint local rest remains rigid.
bool scenePlacement(const Matrix4x4& m,std::string& error) {
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)
        if(!std::isfinite(m.m[r][c])){error="invalid_rig_scene_transform";return false;}
    for(int c=0;c<4;++c)if(std::fabs(m.m[3][c]-(c==3?1.f:0.f))>1e-6f) {
        error="rig_scene_requires_uniform_scale";return false;
    }
    Vec3 p,s;Quaternion q;RayTrophi::decomposeTRS(m,p,q,s);q.normalize();
    if(!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z) ||
       s.x<1e-4f || s.x>10000.f || s.y<=0 || s.z<=0 ||
       std::fabs(s.x-s.y)>1e-5f*s.x || std::fabs(s.x-s.z)>1e-5f*s.x) {
        error="rig_scene_requires_uniform_scale";return false;
    }
    const auto rebuilt=Matrix4x4::translation(p)*q.toMatrix()*Matrix4x4::scaling(Vec3(s.x,s.x,s.x));
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)
        if(!std::isfinite(rebuilt.m[r][c]) || std::fabs(m.m[r][c]-rebuilt.m[r][c])>1e-5f*std::max(1.f,s.x)) {
            error="rig_scene_requires_uniform_scale";return false;
        }
    return true;
}
bool validateNewRig(const SceneData& scene,const std::string& character,std::string& error) {
    if(scene.rigView.edit_mode){error="rig_edit_active";return false;}
    if(!validName(character)){error="invalid_rig_name";return false;}
    const std::string prefix=character+"_";
    for(const auto& model:scene.importedModelContexts)
        if(model.importName==character || model.importName.find(prefix)==0 || character.find(model.importName+"_")==0){error="rig_name_conflict";return false;}
    for(const auto& bone:scene.boneData.boneDefaultTransforms)if(bone.first.find(prefix)==0){error="rig_name_conflict";return false;}
    for(const auto& bone:scene.boneData.boneNameToIndex)if(bone.first.find(prefix)==0){error="rig_name_conflict";return false;}
    for(const auto& object:scene.world.objects) {
        const auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(mesh && mesh->nodeName.find(prefix)==0){error="rig_name_conflict";return false;}
    }
    return true;
}
bool editable(const SceneData& scene,const std::string& character,RigEditState& out,std::string& error) {
    if(!canEditRig(scene,character,error))return false;
    const SceneData::ImportedModelContext* model=nullptr;
    for(const auto& m:scene.importedModelContexts)if(m.importName==character){model=&m;break;}
    if(!model){error="unknown_character";return false;}
    if(hasFlatSkinReferences(scene,model->nodeHierarchy)){error="rig_edit_requires_unskinned";return false;}
    serializeRigHierarchy(model->nodeHierarchy); // Validate before topology indexing.
    if(model->rigRevision==std::numeric_limits<uint64_t>::max()){error="rig_revision_overflow";return false;}
    out.bones=scene.boneData;out.model=*model;return true;
}
bool finish(RigEditState& state,std::string& error) {
    const auto& h=state.model.nodeHierarchy;
    // Validate parents/keys and rebuild local globals without touching live scene.
    serializeRigHierarchy(h);
    if(!validateRigAnatomy(state.model.rigAnatomy,h,error))return false;
    for(const auto& node:h.nodes)if(!rigid(node.localBind,error))return false;
    std::vector<PreviewJoint> globals;if(!sampleRigPose(h,nullptr,0,globals,error))return false;
    unsigned int next=0;
    for(const auto& entry:state.bones.boneNameToIndex) {
        if(entry.second==std::numeric_limits<unsigned int>::max()){error="bone_index_overflow";return false;}
        next=std::max(next,entry.second+1);
    }
    for(size_t i=0;i<h.size();++i) {
        const auto& n=h.nodes[i];
        if(!state.bones.boneNameToIndex.count(n.uniqueName)) {
            if(next>=static_cast<unsigned int>(std::numeric_limits<int>::max())){error="bone_index_overflow";return false;}
            state.bones.boneNameToIndex[n.uniqueName]=next++;
        }
        state.bones.boneDefaultTransforms[n.uniqueName]=n.localBind;
        state.bones.boneParents[n.uniqueName]=n.parent<0?"":h.nodes[n.parent].uniqueName;
        const auto offset=globals[i].world.inverse();
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(offset.m[r][c])){error="invalid_rest_transform";return false;}
        state.bones.boneOffsetMatrices[n.uniqueName]=offset;
    }
    state.bones.perModelInverses[state.model.importName]=Matrix4x4::identity();
    state.bones.boneIndexToName.clear(); // Removed slots must not retain stale names.
    state.bones.rebuildReverseLookup();state.model.rebuildSkeletonRepresentation(state.bones);
    state.model.rigJointGlobals.clear();state.model.rigPoseSource="bind";state.model.restPoseApplied=false;
    state.model.animator=std::make_shared<AnimationController>();
    state.model.ozzAnimationSet=OzzRuntime::buildStubAnimationSet(state.model.importName,state.bones,{});
    ++state.model.rigRevision;return true;
}
}
bool nextRigName(const SceneData& scene,const std::string& seed,std::string& name,std::string& error) {
    name.clear();error.clear();
    if(!validName(seed)){error="invalid_rig_name";return false;}
    if(scene.rigView.edit_mode){error="rig_edit_active";return false;}
    for(unsigned int i=0;i<100000;++i) {
        const auto candidate=seed+(i?std::to_string(i):"");
        if(candidate.size()>128){error="invalid_rig_name";return false;}
        if(validateNewRig(scene,candidate,error)){name=candidate;error.clear();return true;}
        if(error!="rig_name_conflict")return false;
    }
    error="rig_name_exhausted";return false;
}
bool canPlaceRig(const SceneData& scene,const std::string& character,std::string& error) {
    error.clear();if(scene.rigView.edit_mode){error="rig_edit_active";return false;}
    for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
        if(!model.authoringOwned){error="rig_not_owned";return false;}
        if(!model.rigBoundMeshes.empty() || !model.members.empty() || model.weightedBoneCount){error="rig_placement_requires_meshless_unskinned";return false;}
        for(const auto& node:model.nodeHierarchy.nodes)if(scene.boneData.weightedBoneNames.count(node.uniqueName)) {
            error="rig_placement_requires_meshless_unskinned";return false;
        }
        return true;
    }
    error="unknown_character";return false;
}
nlohmann::json serializeRigPlacement(const Matrix4x4& matrix) {
    nlohmann::json value=nlohmann::json::array();
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)value.push_back(matrix.m[r][c]);
    return value;
}
bool deserializeRigPlacement(const nlohmann::json& value,Matrix4x4& out,std::string& error) {
    error.clear();if(!value.is_array() || value.size()!=16){error="invalid_rig_scene_transform";return false;}
    Matrix4x4 matrix;
    for(int r=0;r<4;++r)for(int c=0;c<4;++c) {
        if(!value[r*4+c].is_number()){error="invalid_rig_scene_transform";return false;}
        matrix.m[r][c]=value[r*4+c].get<float>();
    }
    if(!scenePlacement(matrix,error))return false;
    out=matrix;return true;
}
bool stageRigPlacement(const SceneData& scene,const std::string& character,const Matrix4x4& matrix,
                       RigEditState& out,std::string& error) {
    if(!canPlaceRig(scene,character,error))return false;
    if(!scenePlacement(matrix,error))return false;
    for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
        if(hasFlatSkinReferences(scene,model.nodeHierarchy)){error="rig_placement_requires_meshless_unskinned";return false;}
        if(model.rigRevision==std::numeric_limits<uint64_t>::max()){error="rig_revision_overflow";return false;}
        bool changed=false;
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)changed=changed || std::fabs(matrix.m[r][c]-model.rigSceneTransform.m[r][c])>1e-6f;
        if(!changed){error="rig_edit_no_change";return false;}
        out.bones=scene.boneData;out.model=model;out.model.rigSceneTransform=matrix;++out.model.rigRevision;
        out.selectedBone=scene.rigView.character==character?scene.rigView.bone:
            (model.nodeHierarchy.empty()?"":model.nodeHierarchy.nodes.front().uniqueName);
        return true;
    }
    error="unknown_character";return false;
}
bool canEditRig(const SceneData& scene,const std::string& character,std::string& error) {
    error.clear();
    if(scene.rigView.edit_mode && scene.rigView.edit_character!=character){error="rig_edit_character_locked";return false;}
    for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
        if(!model.authoringOwned){error="rig_not_owned";return false;}
        if(!model.rigBoundMeshes.empty() || !model.members.empty() || model.weightedBoneCount){error="rig_edit_requires_unskinned";return false;}
        for(const auto& node:model.nodeHierarchy.nodes)if(scene.boneData.weightedBoneNames.count(node.uniqueName)){error="rig_edit_requires_unskinned";return false;}
        for(const auto& clip:scene.animationDataList)if(clip && clip->modelName==character){error="rig_edit_requires_no_clips";return false;}
        return true;
    }
    error="unknown_character";return false;
}
bool setInteractionMode(SceneData& scene,const std::string& mode,const std::string& character,std::string& error) {
    error.clear();
    if(mode=="pose")return enterPose(scene,character,error);
    if(mode=="scene") {if(scene.rigView.pose.active)leavePose(scene);scene.rigView.edit_mode=false;scene.rigView.edit_character.clear();return true;}
    if(mode!="edit"){error="unsupported_rig_mode";return false;}
    if(scene.rigView.pose.active){error="rig_pose_mode_active";return false;}
    if(!canEditRig(scene,character,error))return false;
    for(const auto& model:scene.importedModelContexts)if(model.importName==character && hasFlatSkinReferences(scene,model.nodeHierarchy)){error="rig_edit_requires_unskinned";return false;}
    scene.rigView.edit_mode=true;scene.rigView.edit_character=character;scene.rigView.visible=true;
    if(scene.rigView.character!=character) {
        scene.rigView.character=character;
        for(const auto& model:scene.importedModelContexts)if(model.importName==character && !model.nodeHierarchy.empty())
            scene.rigView.bone=model.nodeHierarchy.nodes.front().uniqueName;
    }
    return true;
}
bool stageSetRigAnatomy(const SceneData& scene,const std::string& character,const nlohmann::json& value,
                        RigEditState& out,std::string& error) {
    error.clear();if(!editable(scene,character,out,error))return false;
    RigAnatomy anatomy;
    if(!deserializeRigAnatomy(value,out.model.nodeHierarchy,anatomy,error))return false;
    if(serializeRigAnatomy(anatomy)==serializeRigAnatomy(out.model.rigAnatomy)){error="rig_edit_no_change";return false;}
    out.model.rigAnatomy=std::move(anatomy);++out.model.rigRevision;
    out.selectedBone=scene.rigView.character==character?scene.rigView.bone:
        (out.model.nodeHierarchy.empty()?"":out.model.nodeHierarchy.nodes.front().uniqueName);
    return true;
}
bool stageCommitRigFit(const SceneData& scene,const std::string& character,const std::string& mesh,const nlohmann::json& preview,RigEditState& out,std::string& error) {
    RayTrophi::NodeHierarchy h;if(!fittedHierarchy(scene,character,mesh,preview,h,error))return false;
    if(!editable(scene,character,out,error))return false;
    bool changed=false;
    for(size_t i=0;i<h.size();++i)for(int row=0;row<4;++row)for(int col=0;col<4;++col)
        changed=changed || std::fabs(h.nodes[i].localBind.m[row][col]-out.model.nodeHierarchy.nodes[i].localBind.m[row][col])>1e-6f;
    if(!changed){error="rig_edit_no_change";return false;}
    out.model.nodeHierarchy=std::move(h);
    out.selectedBone=scene.rigView.character==character?scene.rigView.bone:out.model.nodeHierarchy.nodes.front().uniqueName;
    return finish(out,error);
}
bool stageCreateRig(const SceneData& scene,const std::string& character,const std::string& id,
                    float height,RigEditState& out,std::string& error) {
    error.clear();out={};
    if(scene.rigView.edit_mode){error="rig_edit_active";return false;}
    if(!validName(character)){error="invalid_rig_name";return false;}
    if(!std::isfinite(height) || height<=0 || height>10000){error="invalid_rig_height";return false;}
    if(!validateNewRig(scene,character,error))return false;
    out.bones=scene.boneData;out.model.importName=character;out.model.authoringOwned=true;out.model.rigTemplateId=id;
    out.model.globalInverseTransform=Matrix4x4::identity();out.model.animGraphAssetKey=character;
    if(!buildRigTemplate(id,character,height,out.model.nodeHierarchy,out.model.rigAnatomy,error))return false;
    for(const auto& info:rigTemplateCatalogue())if(info.id==id){out.model.rigTemplateVersion=info.version;break;}
    out.selectedBone=out.model.nodeHierarchy.nodes.front().uniqueName;return finish(out,error);
}
bool stageCopyRig(const SceneData& scene,const std::string& sourceCharacter,const std::string& character,
                  RigEditState& out,std::vector<RigCopyBone>& mapping,std::string& error) {
    error.clear();out={};mapping.clear();
    if(!validateNewRig(scene,character,error))return false;
    const SceneData::ImportedModelContext* source=nullptr;
    for(const auto& model:scene.importedModelContexts)if(model.importName==sourceCharacter){source=&model;break;}
    if(!source){error="unknown_source_character";return false;}
    if(!source->rigBoundMeshes.empty() || !source->members.empty() || source->weightedBoneCount){error="rig_copy_requires_meshless_unskinned";return false;}
    if(!source->hasSkeletonRepresentation || source->skeletonNodes.empty() || source->nodeHierarchy.empty()) {
        error="rig_copy_requires_skeleton_hierarchy";return false;
    }
    const auto& h=source->nodeHierarchy;
    std::vector<PreviewJoint> globals;
    if(!sampleRigPose(h,nullptr,0,globals,error))return false; // Rest, not the current animated pose.
    std::vector<bool> included(h.size(),false);
    for(const auto& joint:source->skeletonNodes) {
        const auto* node=h.find(joint.name);
        if(!node){error="rig_copy_incomplete_hierarchy";return false;}
        int index=static_cast<int>(node-h.nodes.data());
        while(index>=0 && !included[index]){included[index]=true;index=h.nodes[index].parent;}
    }
    // Bake positive uniform scale into global joint positions. Preserve proper rotations;
    // reject shear, reflection and nonuniform scale rather than silently changing the rest shape.
    auto rigidFrame=[&](const Matrix4x4& m,Matrix4x4& frame) {
        Vec3 p,scale;Quaternion q;decomposeTRS(m,p,q,scale);q.normalize();
        if(!std::isfinite(scale.x) || !std::isfinite(scale.y) || !std::isfinite(scale.z) || scale.x<=1e-8f ||
           std::fabs(scale.x-scale.y)>1e-4f*scale.x || std::fabs(scale.x-scale.z)>1e-4f*scale.x) {
            error="rig_copy_unsupported_rest";return false;
        }
        frame=Matrix4x4::translation(p)*q.toMatrix();
        const auto rebuilt=frame*Matrix4x4::scaling(scale);
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)
            if(!std::isfinite(m.m[r][c]) || !std::isfinite(rebuilt.m[r][c]) ||
               std::fabs(m.m[r][c]-rebuilt.m[r][c])>1e-4f*std::max(1.f,std::fabs(m.m[r][c]))) {
                error="rig_copy_unsupported_rest";return false;
            }
        return rigid(frame,error);
    };
    std::vector<Matrix4x4> frames(h.size());
    std::vector<std::vector<int>> children(h.size());std::vector<int> order;
    for(size_t i=0;i<h.size();++i)if(included[i]) {
        if(scene.boneData.weightedBoneNames.count(h.nodes[i].uniqueName)){error="rig_copy_requires_meshless_unskinned";return false;}
        Matrix4x4 unused;
        if(!rigidFrame(h.nodes[i].localBind,unused) || !rigidFrame(globals[i].world,frames[i]))return false;
        if(h.nodes[i].parent<0)order.push_back(static_cast<int>(i));
        else children[h.nodes[i].parent].push_back(static_cast<int>(i));
    }
    const bool forest=order.size()>1;
    for(size_t cursor=0;cursor<order.size();++cursor)
        for(int child:children[order[cursor]])order.push_back(child);
    if(order.empty()){error="rig_copy_requires_skeleton_hierarchy";return false;}
    out.bones=scene.boneData;out.model.importName=character;out.model.authoringOwned=true;
    out.model.rigTemplateId="import_copy";out.model.globalInverseTransform=Matrix4x4::identity();out.model.animGraphAssetKey=character;
    std::unordered_set<std::string> names;
    if(forest){out.model.nodeHierarchy.addNode("SceneRoot",character+"_SceneRoot",Matrix4x4::identity(),-1);names.insert("SceneRoot");}
    std::vector<int> indices(h.size(),-1);std::vector<RigCopyBone> staged;
    for(int index:order) {
        const auto& node=h.nodes[index];std::string name;
        for(unsigned char c:node.name) {
            if(name.size()==128)break;
            name+=((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-')?char(c):'_';
        }
        if(name.empty())name="Joint";
        const auto stem=name;
        for(size_t suffix=1;names.count(name);++suffix) {
            const auto tail="_"+std::to_string(suffix);name=stem.substr(0,128-tail.size())+tail;
        }
        names.insert(name);const auto key=character+"_"+name;
        const int parent=node.parent<0?(forest?0:-1):indices[node.parent];
        const auto local=node.parent<0?frames[index]:frames[node.parent].inverse()*frames[index];
        indices[index]=out.model.nodeHierarchy.addNode(name,key,local,parent);
        staged.push_back({node.uniqueName,key});
    }
    out.selectedBone=out.model.nodeHierarchy.nodes.front().uniqueName;
    out.model.rigAnatomy=source->rigAnatomy;
    for(const auto& entry:staged)renameAnatomyBone(out.model.rigAnatomy,entry.source_bone,entry.target_bone);
    if(!finish(out,error))return false;
    mapping=std::move(staged);return true;
}
bool nextRigBoneName(const SceneData& scene,const std::string& character,const std::string& seed,std::string& name,std::string& error) {
    name.clear();error.clear();
    if(!validName(seed)){error="invalid_bone_name";return false;}
    if(!canEditRig(scene,character,error))return false;
    const NodeHierarchy* hierarchy=nullptr;
    for(const auto& model:scene.importedModelContexts)if(model.importName==character){hierarchy=&model.nodeHierarchy;break;}
    auto available=[&](const std::string& candidate) {
        const auto key=character+"_"+candidate;
        return !hierarchy->find(key) && !scene.boneData.boneNameToIndex.count(key) && !scene.boneData.boneDefaultTransforms.count(key);
    };
    if(available(seed)){name=seed;return true;}
    size_t end=seed.size();while(end>0 && seed[end-1]>='0' && seed[end-1]<='9')--end;
    const std::string stem=seed.substr(0,end);
    uint64_t number=0;
    for(size_t i=end;i<seed.size();++i) {
        const unsigned digit=seed[i]-'0';
        if(number>(std::numeric_limits<uint64_t>::max()-digit)/10){error="bone_name_exhausted";return false;}
        number=number*10+digit;
    }
    const size_t attempts=hierarchy->size()+scene.boneData.boneNameToIndex.size()+scene.boneData.boneDefaultTransforms.size()+1;
    for(size_t i=0;i<attempts;++i) {
        if(number==std::numeric_limits<uint64_t>::max()){error="bone_name_exhausted";return false;}
        const auto candidate=stem+std::to_string(++number);
        if(candidate.size()>128){error="bone_name_exhausted";return false;}
        if(available(candidate)){name=candidate;return true;}
    }
    error="bone_name_exhausted";return false;
}
bool stageAddRigBone(const SceneData& scene,const std::string& character,const std::string& name,
                     const std::string& parent,const Matrix4x4& rest,RigEditState& out,std::string& error) {
    error.clear();std::string resolved=name;
    if(resolved.empty() && !nextRigBoneName(scene,character,"Joint1",resolved,error))return false;
    if(!validName(resolved)){error="invalid_bone_name";return false;}
    if(!rigid(rest,error) || !editable(scene,character,out,error))return false;
    auto& h=out.model.nodeHierarchy;const auto* p=h.find(parent);
    if(!p){error="unknown_parent_bone";return false;}
    const std::string key=character+"_"+resolved;
    if(h.find(key) || out.bones.boneNameToIndex.count(key) || out.bones.boneDefaultTransforms.count(key)){error="bone_name_conflict";return false;}
    const auto parentIndex=static_cast<int>(p-h.nodes.data());
    h.addNode(resolved,key,rest,parentIndex);out.selectedBone=key;return finish(out,error);
}
bool stageMirrorRigRest(const SceneData& scene,const std::string& character,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane& plane,uint64_t revision,RigEditState& out,std::string& error) {
    if(!editable(scene,character,out,error))return false;
    if(out.model.rigRevision!=revision){error="rig_edit_stale_revision";return false;}
    RayTrophi::NodeHierarchy mirrored;std::vector<std::string> targets;
    if(!mirrorRest(out.model.nodeHierarchy,out.model.rigAnatomy,sources,direction,plane,mirrored,targets,error))return false;
    bool changed=false;for(size_t i=0;i<mirrored.size();++i)for(int r=0;r<4;++r)for(int c=0;c<4;++c)
        changed=changed || std::fabs(mirrored.nodes[i].localBind.m[r][c]-out.model.nodeHierarchy.nodes[i].localBind.m[r][c])>1e-6f;
    if(!changed){error="rig_edit_no_change";return false;}
    out.model.nodeHierarchy=std::move(mirrored);out.selectedBones=std::move(targets);out.selectedBone=out.selectedBones.back();return finish(out,error);
}
bool stageCreateMirrorBone(const SceneData& scene,const std::string& character,const std::string& source,const std::string& name,const std::string& sourceSide,const MirrorPlane& plane,uint64_t revision,RigEditState& out,std::string& error) {
    if(!validName(name)){error="invalid_bone_name";return false;}if(!editable(scene,character,out,error))return false;
    if(out.model.rigRevision!=revision){error="rig_edit_stale_revision";return false;}
    const auto key=character+"_"+name;
    if(out.bones.boneNameToIndex.count(key) || out.bones.boneDefaultTransforms.count(key) || out.bones.boneOffsetMatrices.count(key) || out.bones.boneParents.count(key)){error="bone_name_conflict";return false;}
    RayTrophi::NodeHierarchy h;RigAnatomy a;
    if(!createMirrorBone(out.model.nodeHierarchy,out.model.rigAnatomy,source,key,name,sourceSide,plane,h,a,error))return false;
    out.model.nodeHierarchy=std::move(h);out.model.rigAnatomy=std::move(a);out.selectedBone=key;return finish(out,error);
}
bool stageBatchRigRest(const SceneData& scene,const std::string& character,const std::vector<std::string>& bones,const Matrix4x4& delta,uint64_t expected,RigEditState& out,std::string& error) {
    if(!editable(scene,character,out,error))return false;
    if(out.model.rigRevision!=expected){error="rig_edit_stale_revision";return false;}
    NodeHierarchy staged;
    if(!transformRestHierarchy(out.model.nodeHierarchy,out.model.rigSceneTransform,bones,delta,staged,error))return false;
    bool changed=false;
    for(size_t i=0;i<staged.size();++i)for(int r=0;r<4;++r)for(int c=0;c<4;++c)
        changed=changed || std::fabs(staged.nodes[i].localBind.m[r][c]-out.model.nodeHierarchy.nodes[i].localBind.m[r][c])>1e-6f;
    if(!changed){error="rig_edit_no_change";return false;}
    out.model.nodeHierarchy=std::move(staged);out.selectedBones=bones;
    out.selectedBone=scene.rigView.character==character && std::find(bones.begin(),bones.end(),scene.rigView.bone)!=bones.end()?scene.rigView.bone:bones.back();
    return finish(out,error);
}
bool stageRigRestEdit(const SceneData& scene,const std::string& character,const std::string& bone,
                      const Matrix4x4& rest,RigEditState& out,std::string& error) {
    error.clear();if(!rigid(rest,error) || !editable(scene,character,out,error))return false;
    for(auto& n:out.model.nodeHierarchy.nodes)if(n.uniqueName==bone){n.localBind=rest;out.selectedBone=bone;return finish(out,error);}
    error="unknown_bone";return false;
}
namespace {
int jointIndex(const NodeHierarchy& h,const std::string& bone) {
    for(size_t i=0;i<h.size();++i)if(h.nodes[i].uniqueName==bone)return static_cast<int>(i);
    return -1;
}
void rebuildChildren(NodeHierarchy& h) {
    for(auto& n:h.nodes)n.children.clear();
    for(size_t i=0;i<h.size();++i)if(h.nodes[i].parent>=0)h.nodes[h.nodes[i].parent].children.push_back(static_cast<int>(i));
}
bool externalReference(const SceneData& scene,const std::string& character,const std::string& bone) {
    const auto prefix=character+"_";
    for(const auto& entry:scene.boneData.boneParents)
        if(entry.first.find(prefix)!=0 && entry.second==bone)return true;
    for(const auto& model:scene.importedModelContexts)if(model.importName!=character)
        if(model.nodeHierarchy.find(bone))return true;
    return false;
}
}
bool stageRenameRigBone(const SceneData& scene,const std::string& character,const std::string& bone,
                        const std::string& name,RigEditState& out,std::string& error) {
    error.clear();if(!validName(name)){error="invalid_bone_name";return false;}
    if(!editable(scene,character,out,error))return false;
    auto& h=out.model.nodeHierarchy;const int index=jointIndex(h,bone);
    if(index<0){error="unknown_bone";return false;}
    const auto key=character+"_"+name;
    if(key==bone){error="rig_edit_no_change";return false;}
    if(h.find(key) || out.bones.boneNameToIndex.count(key) || out.bones.boneDefaultTransforms.count(key) ||
       out.bones.boneOffsetMatrices.count(key) || out.bones.boneParents.count(key)){error="bone_name_conflict";return false;}
    if(externalReference(scene,character,bone)){error="rig_bone_external_reference";return false;}
    // Keep scene indices stable: skinned rigs elsewhere retain their original weight indices.
    auto rekey=[&](auto& map) {auto it=map.find(bone);if(it!=map.end()){auto value=it->second;map.erase(it);map.emplace(key,std::move(value));}};
    rekey(out.bones.boneNameToIndex);rekey(out.bones.boneDefaultTransforms);
    rekey(out.bones.boneOffsetMatrices);rekey(out.bones.boneParents);
    for(auto& entry:out.bones.boneParents)if(entry.second==bone)entry.second=key;
    renameAnatomyBone(out.model.rigAnatomy,bone,key);
    h.nodes[index].name=name;h.nodes[index].uniqueName=key;out.selectedBone=key;
    return finish(out,error);
}
bool stageReparentRigBone(const SceneData& scene,const std::string& character,const std::string& bone,
                          const std::string& parent,RigEditState& out,std::string& error) {
    error.clear();if(!editable(scene,character,out,error))return false;
    auto& h=out.model.nodeHierarchy;const int index=jointIndex(h,bone),newParent=jointIndex(h,parent);
    if(index<0){error="unknown_bone";return false;}
    if(newParent<0){error="unknown_parent_bone";return false;}
    if(h.nodes[index].parent<0){error="rig_root_edit_blocked";return false;}
    if(h.nodes[index].parent==newParent){error="rig_edit_no_change";return false;}
    if(externalReference(scene,character,bone)){error="rig_bone_external_reference";return false;}
    // Reject self-parent and descendant-parent cycles before sampling or mutating topology.
    int ancestor=newParent;
    for(size_t visits=0;ancestor>=0;++visits) {
        if(ancestor==index){error="rig_parent_cycle";return false;}
        if(visits>=h.size() || static_cast<size_t>(ancestor)>=h.size()){error="rig_invalid_hierarchy";return false;}
        ancestor=h.nodes[ancestor].parent;
    }
    std::vector<PreviewJoint> globals;if(!sampleRigPose(h,nullptr,0,globals,error))return false;
    const auto local=globals[newParent].world.inverse()*globals[index].world;
    if(!rigid(local,error))return false;
    h.nodes[index].parent=newParent;h.nodes[index].localBind=local;rebuildChildren(h);
    out.selectedBone=bone;return finish(out,error);
}
bool stageDeleteRigBone(const SceneData& scene,const std::string& character,const std::string& bone,
                        RigEditState& out,std::string& error) {
    error.clear();if(!editable(scene,character,out,error))return false;
    auto& h=out.model.nodeHierarchy;const int index=jointIndex(h,bone);
    if(index<0){error="unknown_bone";return false;}
    const int parent=h.nodes[index].parent;
    if(parent<0){error="rig_root_edit_blocked";return false;}
    for(const auto& n:h.nodes)if(n.parent==index){error="rig_delete_requires_leaf";return false;}
    if(externalReference(scene,character,bone)){error="rig_bone_external_reference";return false;}
    if(anatomyReferencesBone(out.model.rigAnatomy,bone)){error="rig_bone_anatomy_referenced";return false;}
    out.selectedBone=h.nodes[parent].uniqueName;
    h.nodes.erase(h.nodes.begin()+index);
    for(auto& n:h.nodes)if(n.parent>index)--n.parent;
    rebuildChildren(h);
    out.bones.boneNameToIndex.erase(bone);out.bones.boneDefaultTransforms.erase(bone);
    out.bones.boneOffsetMatrices.erase(bone);out.bones.boneParents.erase(bone);
    out.bones.weightedBoneNames.erase(bone);
    return finish(out,error);
}
bool restoreOwnedRigRuntime(SceneData& scene,std::string& error) {
    error.clear();
    try {
        if(!restoreRigBindingMembers(scene,error))return false;
        for(auto& model:scene.importedModelContexts)if(model.authoringOwned) {
            bool hasClips=false;
            for(const auto& clip:scene.animationDataList)if(clip && clip->modelName==model.importName){hasClips=true;break;}
            if(hasClips)continue; // The existing animation initializer handles these.
            auto runtime=OzzRuntime::buildStubAnimationSet(model.importName,scene.boneData,{});
            auto animator=std::make_shared<AnimationController>();
            model.ozzAnimationSet=std::move(runtime);model.animator=std::move(animator);
        }
        return true;
    }catch(const std::exception&){error="rig_runtime_restore_failed";return false;}
}
}
