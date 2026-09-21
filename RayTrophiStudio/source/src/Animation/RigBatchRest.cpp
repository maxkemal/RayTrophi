#include "Animation/RigBatchRest.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigBindMath.h"
#include <cmath>
#include <functional>
#include <unordered_set>
namespace RigAuthoring {
bool transformRestHierarchy(const RayTrophi::NodeHierarchy& source,const Matrix4x4& placement,const std::vector<std::string>& bones,const Matrix4x4& worldDelta,RayTrophi::NodeHierarchy& output,std::string& error) {
    error.clear();if(bones.empty()){error="rig_selection_empty";return false;}
    if(bones.size()>4096 || source.size()>4096){error="rig_selection_limit";return false;}
    // Only a rigid delta; scale remains actor placement, not joint rest authoring.
    for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(worldDelta.m[r][c])){error="invalid_rest_transform";return false;}
    for(int c=0;c<4;++c)if(std::fabs(worldDelta.m[3][c]-(c==3?1.f:0.f))>1e-5f){error="rig_rest_requires_rigid_transform";return false;}
    for(int a=0;a<3;++a)for(int b=0;b<3;++b) {
        double dot=0;for(int r=0;r<3;++r)dot+=double(worldDelta.m[r][a])*worldDelta.m[r][b];
        if(std::fabs(dot-(a==b?1.:0.))>1e-4){error="rig_rest_requires_rigid_transform";return false;}
    }
    Matrix4x4 inversePlacement,inverseDelta;
    if(!bindAffineInverse(placement,inversePlacement) || !bindAffineInverse(worldDelta,inverseDelta)){error="rig_rest_requires_rigid_transform";return false;}
    std::vector<PreviewJoint> globals;if(!sampleRigPose(source,nullptr,0,globals,error))return false;
    std::unordered_set<std::string> selected;
    for(const auto& name:bones) {
        if(!source.find(name)){error="unknown_bone";return false;}
        if(!selected.insert(name).second){error="rig_selection_duplicate_bone";return false;}
    }
    auto staged=source;std::vector<Matrix4x4> targets(source.size());std::vector<bool> done(source.size(),false);
    const auto delta=inversePlacement*worldDelta*placement;
    std::function<bool(size_t)> solve=[&](size_t i) {
        if(done[i])return true;
        const auto parent=source.nodes[i].parent;Matrix4x4 parentGlobal=Matrix4x4::identity();
        if(parent>=0){if(!solve(static_cast<size_t>(parent)))return false;parentGlobal=targets[static_cast<size_t>(parent)];}
        if(selected.count(source.nodes[i].uniqueName)) {
            Matrix4x4 inverse;if(!bindAffineInverse(parentGlobal,inverse)){error="rig_batch_rest_invalid_parent";return false;}
            targets[i]=delta*globals[i].world;staged.nodes[i].localBind=inverse*targets[i];
        } else targets[i]=parentGlobal*source.nodes[i].localBind;
        done[i]=true;return true;
    };
    for(size_t i=0;i<source.size();++i)if(!solve(i))return false;
    std::vector<PreviewJoint> validated;if(!sampleRigPose(staged,nullptr,0,validated,error))return false;
    output=std::move(staged);return true;
}
}
