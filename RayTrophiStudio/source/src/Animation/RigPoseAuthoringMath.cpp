#include "Animation/RigPoseAuthoringMath.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
namespace RigAuthoring {
bool poseHierarchy(const RayTrophi::NodeHierarchy& rest,const AnimationData* clip,double seconds,const JointGlobals& locals,RayTrophi::NodeHierarchy& output,std::string& error) {
 if(rest.size()>4096 || locals.size()>4096){error="rig_pose_limit";return false;}
 std::vector<PreviewJoint> validate;if(!sampleRigPose(rest,clip,seconds,validate,error))return false;
 auto staged=rest;
 for(const auto& p:locals)if(!rest.find(p.first)){error="unknown_bone";return false;}
 for(auto& node:staged.nodes) {
  if(clip)node.localBind=clip->calculateAnimationTransform(
      *clip,static_cast<float>(seconds),node.uniqueName,node.localBind,false);
  const auto found=locals.find(node.uniqueName);if(found!=locals.end())node.localBind=found->second;
 }
 if(!sampleRigPose(staged,nullptr,0,validate,error))return false;
 output=std::move(staged);return true;
}
bool insertPoseKeys(AnimationData& clip,const RayTrophi::NodeHierarchy& pose,const std::vector<std::string>& bones,double seconds,std::string& error) {
 error.clear();if(!clip.rigAuthoring){error="rig_pose_clip_not_editable";return false;}
 if(!std::isfinite(seconds)||seconds<0||!std::isfinite(clip.ticksPerSecond)||clip.ticksPerSecond<=0||!std::isfinite(clip.duration)||clip.duration<=0){error="invalid_clip_timing";return false;}
 const double time=seconds*clip.ticksPerSecond;if(!std::isfinite(time)||time>1000000){error="rig_pose_time_limit";return false;}
 if(bones.empty()){error="rig_selection_empty";return false;}if(bones.size()>4096){error="rig_pose_limit";return false;}
 auto staged=clip;std::unordered_set<std::string> names;
 for(const auto& name:bones) {
  if(!names.insert(name).second){error="rig_selection_duplicate_bone";return false;}
  const auto* node=pose.find(name);if(!node){error="unknown_bone";return false;}
  Vec3 position,scale;Quaternion rotation;RayTrophi::decomposeTRS(node->localBind,position,rotation,scale);rotation.normalize();
  const auto rebuilt=Matrix4x4::translation(position)*rotation.toMatrix();
  for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(node->localBind.m[r][c])||std::fabs(node->localBind.m[r][c]-rebuilt.m[r][c])>1e-4f){error="rig_pose_requires_rigid_transform";return false;}
  auto& pos=staged.positionKeys[name];auto p=std::lower_bound(pos.begin(),pos.end(),time,[](const auto& k,double t){return k.time<t;});
  if(p!=pos.end()&&std::fabs(p->time-time)<1e-8)p->value=position;else pos.insert(p,{time,position});
  auto& rot=staged.rotationKeys[name];auto q=std::lower_bound(rot.begin(),rot.end(),time,[](const auto& k,double t){return k.time<t;});
  if(q!=rot.end()&&std::fabs(q->time-time)<1e-8)q->value=rotation;else rot.insert(q,{time,rotation});
  for(size_t i=1;i<rot.size();++i) {
   const auto& a=rot[i-1].value;auto& b=rot[i].value;
   if(a.w*b.w+a.x*b.x+a.y*b.y+a.z*b.z<0){b.w=-b.w;b.x=-b.x;b.y=-b.y;b.z=-b.z;}
  }
 }
 staged.duration=std::max(staged.duration,time+1.0);staged.endFrame=std::max(staged.endFrame,static_cast<int>(std::ceil(time)));
 clip=std::move(staged);return true;
}
bool removePoseKeys(AnimationData& clip, const RayTrophi::NodeHierarchy& pose,
                    const std::vector<std::string>& bones, double seconds, std::string& error) {
    error.clear();
    if (!clip.rigAuthoring) {
        error = "rig_pose_clip_not_editable";
        return false;
    }
    if (!std::isfinite(seconds) || seconds < 0 || !std::isfinite(clip.ticksPerSecond) ||
        clip.ticksPerSecond <= 0) {
        error = "invalid_clip_timing";
        return false;
    }
    const double time = seconds * clip.ticksPerSecond;
    if (!std::isfinite(time) || time > 1000000) {
        error = "rig_pose_time_limit";
        return false;
    }
    if (bones.empty()) {
        error = "rig_selection_empty";
        return false;
    }
    if (bones.size() > 4096) {
        error = "rig_pose_limit";
        return false;
    }
    auto staged = clip;
    std::unordered_set<std::string> names;
    bool changed = false;
    for (const auto& name : bones) {
        if (!names.insert(name).second) {
            error = "rig_selection_duplicate_bone";
            return false;
        }
        if (!pose.find(name)) {
            error = "unknown_bone";
            return false;
        }
        auto eraseAt = [&](auto& channels) {
            const auto found = channels.find(name);
            if (found == channels.end())
                return;
            auto& keys = found->second;
            const auto oldSize = keys.size();
            keys.erase(std::remove_if(keys.begin(), keys.end(), [&](const auto& key) {
                           return std::fabs(key.time - time) < 1e-8;
                       }),
                       keys.end());
            changed = changed || keys.size() != oldSize;
            if (keys.empty())
                channels.erase(found);
        };
        eraseAt(staged.positionKeys);
        eraseAt(staged.rotationKeys);
    }
    if (!changed) {
        error = "rig_edit_no_change";
        return false;
    }
    clip = std::move(staged);
    return true;
}
}
