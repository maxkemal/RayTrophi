#include "Animation/RigMirror.h"
#include "Animation/RigPosePreview.h"
#include "Animation/RigBindMath.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <unordered_set>
#include <unordered_map>
namespace RigAuthoring {
bool mirrorReflection(const MirrorPlane& plane,Matrix4x4& pointReflection,Matrix4x4& basisReflection,std::string& error) {
    error.clear();int a=plane.axis=="x"?0:plane.axis=="y"?1:plane.axis=="z"?2:-1;
    if(a<0 || !std::isfinite(plane.offset) || !std::isfinite(plane.offset*2.f)){error="rig_mirror_invalid_plane";return false;}
    pointReflection=basisReflection=Matrix4x4::identity();pointReflection.m[a][a]=basisReflection.m[a][a]=-1;
    pointReflection.m[a][3]=2.f*plane.offset;return true;
}
bool mirrorPairs(const RayTrophi::NodeHierarchy& h,const RigAnatomy& anatomy,const std::vector<std::string>& sources,const std::string& direction,std::vector<RigSymmetry>& copies,std::string& error) {
    error.clear();if(direction!="selected" && direction!="left_to_right" && direction!="right_to_left"){error="rig_mirror_invalid_direction";return false;}
    if(h.size()>4096 || sources.size()>4096){error="rig_mirror_limit";return false;}
    if(!validateRigAnatomy(anatomy,h,error))return false;
    std::unordered_set<std::string> requested,paired;
    for(const auto& p:anatomy.symmetry){paired.insert(p.left);paired.insert(p.right);}
    for(const auto& name:sources) {
        if(!h.find(name)){error="unknown_bone";return false;}
        if(!requested.insert(name).second){error="rig_selection_duplicate_bone";return false;}
        if(!paired.count(name)){error="rig_mirror_unpaired_bone";return false;}
    }
    if(direction=="selected" && requested.empty()){error="rig_selection_empty";return false;}
    std::vector<RigSymmetry> staged;
    for(const auto& p:anatomy.symmetry) {
        if(direction=="selected") {
            if(requested.count(p.left) && requested.count(p.right)){error="rig_mirror_ambiguous_pair";return false;}
            if(requested.count(p.left))staged.push_back(p);
            if(requested.count(p.right))staged.push_back({p.right,p.left});
        } else {
            const auto& from=direction=="left_to_right"?p.left:p.right;
            const auto& to=direction=="left_to_right"?p.right:p.left;
            if(requested.empty() || requested.count(from))staged.push_back({from,to});
        }
    }
    if(staged.empty()){error="rig_mirror_no_pairs";return false;}copies=std::move(staged);return true;
}
bool mirrorRest(const RayTrophi::NodeHierarchy& h,const RigAnatomy& anatomy,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane& plane,RayTrophi::NodeHierarchy& output,std::vector<std::string>& targets,std::string& error) {
    Matrix4x4 reflection,basis;if(!mirrorReflection(plane,reflection,basis,error))return false;
    std::vector<RigSymmetry> copies;if(!mirrorPairs(h,anatomy,sources,direction,copies,error))return false;
    std::vector<PreviewJoint> joints;if(!sampleRigPose(h,nullptr,0,joints,error))return false;
    std::unordered_map<std::string,size_t> indices;for(size_t i=0;i<h.size();++i)indices[h.nodes[i].uniqueName]=i;
    std::unordered_map<size_t,Matrix4x4> desired;std::vector<std::string> changed;
    for(const auto& p:copies){desired[indices.at(p.right)]=reflection*joints[indices.at(p.left)].world*basis;changed.push_back(p.right);}
    auto staged=h;std::vector<Matrix4x4> globals(h.size());std::vector<bool> done(h.size(),false);
    std::function<bool(size_t)> solve=[&](size_t i) {
        if(done[i])return true;Matrix4x4 parent=Matrix4x4::identity();const int p=h.nodes[i].parent;
        if(p>=0){if(!solve(static_cast<size_t>(p)))return false;parent=globals[p];}
        const auto target=desired.find(i);
        if(target!=desired.end()) {
            Matrix4x4 inverse;if(!bindAffineInverse(parent,inverse)){error="rig_mirror_invalid_transform";return false;}
            globals[i]=target->second;staged.nodes[i].localBind=inverse*globals[i];
        } else globals[i]=parent*h.nodes[i].localBind;
        done[i]=true;return true;
    };
    for(size_t i=0;i<h.size();++i)if(!solve(i))return false;
    if(!sampleRigPose(staged,nullptr,0,joints,error))return false;
    output=std::move(staged);targets=std::move(changed);return true;
}
bool mirrorPose(const RayTrophi::NodeHierarchy& rest,const RayTrophi::NodeHierarchy& pose,const RigAnatomy& anatomy,const std::vector<std::string>& sources,const std::string& direction,const std::string& axis,RayTrophi::NodeHierarchy& output,std::string& error) {
    Matrix4x4 reflection,basis;if(!mirrorReflection({axis,0},reflection,basis,error))return false;
    std::vector<RigSymmetry> copies;if(!mirrorPairs(rest,anatomy,sources,direction,copies,error))return false;
    if(rest.size()!=pose.size()){error="invalid_preview_hierarchy";return false;}
    for(size_t i=0;i<rest.size();++i)if(rest.nodes[i].uniqueName!=pose.nodes[i].uniqueName || rest.nodes[i].parent!=pose.nodes[i].parent){error="invalid_preview_hierarchy";return false;}
    std::vector<PreviewJoint> globals,validation;
    if(!sampleRigPose(rest,nullptr,0,globals,error) || !sampleRigPose(pose,nullptr,0,validation,error))return false;
    std::unordered_map<std::string,size_t> indices;for(size_t i=0;i<rest.size();++i)indices[rest.nodes[i].uniqueName]=i;
    auto staged=pose;
    for(const auto& pair:copies) {
        const auto s=indices.at(pair.left),t=indices.at(pair.right);
        // Strip joint origins: the mirrored displacement is relative to the
        // target's own rest, preserving asymmetric fitting and bone lengths.
        auto sourceBasis=globals[s].world,targetBasis=globals[t].world;
        for(int r=0;r<3;++r){sourceBasis.m[r][3]=0;targetBasis.m[r][3]=0;}
        Matrix4x4 inverseLocal,inverseSource,inverseTarget;
        if(!bindAffineInverse(rest.nodes[s].localBind,inverseLocal) || !bindAffineInverse(sourceBasis,inverseSource) || !bindAffineInverse(targetBasis,inverseTarget)){error="rig_mirror_invalid_transform";return false;}
        const auto delta=inverseLocal*pose.nodes[s].localBind;
        staged.nodes[t].localBind=rest.nodes[t].localBind*inverseTarget*basis*sourceBasis*delta*inverseSource*basis*targetBasis;
    }
    if(!sampleRigPose(staged,nullptr,0,validation,error))return false;
    output=std::move(staged);return true;
}
bool mirrorLandmarks(const RayTrophi::NodeHierarchy& h,const RigAnatomy& anatomy,const Matrix4x4& placement,const nlohmann::json& marks,const std::vector<std::string>& sources,const std::string& direction,const MirrorPlane& plane,nlohmann::json& output,std::string& error) {
    Matrix4x4 reflection,basis,inverse;if(!mirrorReflection(plane,reflection,basis,error))return false;
    std::vector<RigSymmetry> copies;if(!mirrorPairs(h,anatomy,sources,direction,copies,error))return false;
    if(!bindAffineInverse(placement,inverse)){error="rig_mirror_invalid_transform";return false;}
    if(!marks.is_object() || marks.size()!=h.size()){error="rig_fit_incomplete_landmarks";return false;}
    for(const auto& node:h.nodes) {
        if(!marks.contains(node.uniqueName)){error="rig_fit_incomplete_landmarks";return false;}
        const auto& p=marks[node.uniqueName];if(!p.is_array() || p.size()!=3){error="rig_fit_invalid_landmark";return false;}
        for(const auto& v:p)if(!v.is_number() || !std::isfinite(v.get<float>())){error="rig_fit_invalid_landmark";return false;}
    }
    auto staged=marks;const auto worldReflection=placement*reflection*inverse;
    for(const auto& p:copies) {
        const auto& v=marks[p.left];const auto target=worldReflection.transform_point(Vec3(v[0].get<float>(),v[1].get<float>(),v[2].get<float>()));
        if(!std::isfinite(target.x)||!std::isfinite(target.y)||!std::isfinite(target.z)){error="rig_mirror_invalid_transform";return false;}
        staged[p.right]={target.x,target.y,target.z};
    }
    output=std::move(staged);return true;
}
bool createMirrorBone(const RayTrophi::NodeHierarchy& h,const RigAnatomy& metadata,const std::string& source,const std::string& key,const std::string& label,const std::string& sourceSide,const MirrorPlane& plane,RayTrophi::NodeHierarchy& output,RigAnatomy& anatomy,std::string& error) {
    if(sourceSide!="left" && sourceSide!="right"){error="rig_mirror_invalid_side";return false;}
    if(h.size()>=4096){error="rig_mirror_limit";return false;}
    if(!validateRigAnatomy(metadata,h,error))return false;
    Matrix4x4 reflection,basis;if(!mirrorReflection(plane,reflection,basis,error))return false;
    std::vector<PreviewJoint> globals;if(!sampleRigPose(h,nullptr,0,globals,error))return false;
    int index=-1;for(size_t i=0;i<h.size();++i)if(h.nodes[i].uniqueName==source)index=static_cast<int>(i);
    if(index<0){error="unknown_bone";return false;}if(h.find(key)){error="bone_name_conflict";return false;}
    for(const auto& p:metadata.symmetry)if(p.left==source || p.right==source){error="rig_mirror_already_paired";return false;}
    int parent=h.nodes[index].parent;if(parent<0){error="rig_root_edit_blocked";return false;}
    const auto parentKey=h.nodes[parent].uniqueName;
    for(const auto& p:metadata.symmetry) {
        const auto target=p.left==parentKey?p.right:p.right==parentKey?p.left:std::string();
        if(!target.empty())for(size_t i=0;i<h.size();++i)if(h.nodes[i].uniqueName==target)parent=static_cast<int>(i);
    }
    Matrix4x4 inverse;if(!bindAffineInverse(globals[parent].world,inverse)){error="rig_mirror_invalid_transform";return false;}
    auto staged=h;staged.addNode(label,key,inverse*reflection*globals[index].world*basis,parent);
    auto a=metadata;a.symmetry.push_back(sourceSide=="left"?RigSymmetry{source,key}:RigSymmetry{key,source});
    if(!validateRigAnatomy(a,staged,error) || !sampleRigPose(staged,nullptr,0,globals,error))return false;
    output=std::move(staged);anatomy=std::move(a);return true;
}
}
