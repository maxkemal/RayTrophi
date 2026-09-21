#include "Animation/RigJointLimits.h"
#include "Animation/RigView.h"
#include "Animation/RigBindMath.h"
#include "Animation/AnimationKeys.h"
#include "scene_data.h"
#include <algorithm>
#include <cmath>
namespace RigAuthoring {
namespace {
constexpr float rad=3.14159265359f/180.f;
Vec3 origin(const Matrix4x4& m){return Vec3(m.m[0][3],m.m[1][3],m.m[2][3]);}
Vec3 vector(const Matrix4x4& m,const Vec3& v){return Vec3(m.m[0][0]*v.x+m.m[0][1]*v.y+m.m[0][2]*v.z,m.m[1][0]*v.x+m.m[1][1]*v.y+m.m[1][2]*v.z,m.m[2][0]*v.x+m.m[2][1]*v.y+m.m[2][2]*v.z).normalize();}
bool rotation(const Matrix4x4& m,Quaternion& q){Vec3 p,s;RayTrophi::decomposeTRS(m,p,q,s);q.normalize();return std::isfinite(q.w)&&std::isfinite(q.x)&&std::isfinite(q.y)&&std::isfinite(q.z);}
}
Vec3 jointLimitAxis(const JointLimitView& v){return vector(v.neutralWorld,v.rule.axis);}
Vec3 jointLimitReference(const JointLimitView& v){const auto a=v.rule.axis;const auto seed=std::fabs(a.x)<.7f?Vec3(1,0,0):Vec3(0,1,0);return vector(v.neutralWorld,(seed-a*Vec3::dot(seed,a)).normalize());}
Vec3 jointLimitArcPoint(const JointLimitView& v,float radius,float degrees){const auto a=jointLimitAxis(v),r=jointLimitReference(v),t=a.cross(r).normalize();return origin(v.jointWorld)+(r*std::cos(degrees*rad)+t*std::sin(degrees*rad))*radius;}
Vec3 jointLimitConePoint(const JointLimitView& v,float radius,float swing,float azimuth){const auto a=jointLimitAxis(v),r=jointLimitReference(v),t=a.cross(r).normalize();return origin(v.jointWorld)+(a*std::cos(swing*rad)+(r*std::cos(azimuth*rad)+t*std::sin(azimuth*rad))*std::sin(swing*rad))*radius;}
bool jointOutsideLimits(const JointRule& rule,float twist,float swing){if(!rule.enabled)return false;if(rule.type=="fixed")return std::fabs(twist)>.1f||swing>.1f;return (rule.type=="hinge"||rule.type=="ball")&&(twist<rule.minimum-.1f||twist>rule.maximum+.1f||swing>(rule.type=="hinge"?.1f:rule.swing+.1f));}
bool replaceJointLimits(const std::vector<JointRule>& rules,const RayTrophi::NodeHierarchy& h,const std::string& bone,float minimum,float maximum,float swing,std::vector<JointRule>& output,std::string& error){
 if(!h.find(bone)){error="unknown_bone";return false;}auto staged=rules;
 auto found=std::find_if(staged.begin(),staged.end(),[&](const auto& r){return r.bone==bone;});
 if(found==staged.end()){error="rig_joint_rule_required";return false;}
 if(found->type!="hinge"&&found->type!="ball"){error="rig_joint_limits_not_applicable";return false;}
 found->minimum=minimum;found->maximum=maximum;found->swing=swing;
 if(!validateJointRules(staged,h,error))return false;output=std::move(staged);return true;
}
bool getJointLimitView(const SceneData& scene,const std::string& character,const std::string& bone,JointLimitView& output,std::string& error){
 error.clear();std::vector<BoneView> bones;if(!listBones(scene,character,bones,error))return false;
 const auto found=std::find_if(bones.begin(),bones.end(),[&](const auto& b){return b.name==bone;});if(found==bones.end()){error="unknown_bone";return false;}
 for(const auto& model:scene.importedModelContexts)if(model.importName==character){
  const auto* node=model.nodeHierarchy.find(bone);if(!node){error="rig_joint_hierarchy_required";return false;}
  if(node->parent<-1||node->parent>=static_cast<int>(model.nodeHierarchy.size())){error="rig_joint_hierarchy_required";return false;}
  JointLimitView view;view.character=character;view.bone=bone;view.rule.bone=bone;view.owned=model.authoringOwned;view.revision=model.rigRevision;view.poseSource=found->pose_source;view.jointWorld=found->world;
  for(const auto& rule:model.rigAnatomy.joints)if(rule.bone==bone){view.rule=rule;view.hasRule=true;break;}
  if(!validateJointRules({view.rule},model.nodeHierarchy,error))return false;
  auto parent=found->scene_transform;
  if(node->parent>=0){const auto& key=model.nodeHierarchy.nodes[static_cast<size_t>(node->parent)].uniqueName;const auto p=std::find_if(bones.begin(),bones.end(),[&](const auto& b){return b.name==key;});if(p==bones.end()){error="rig_joint_hierarchy_required";return false;}parent=p->world;}
  view.neutralWorld=parent*node->localBind;for(int r=0;r<3;++r)view.neutralWorld.m[r][3]=view.jointWorld.m[r][3];
  Matrix4x4 inverse;if(!bindAffineInverse(parent,inverse)){error="rig_joint_invalid_transform";return false;}
  Quaternion rest,current;if(!rotation(node->localBind,rest)||!rotation(inverse*view.jointWorld,current)){error="rig_joint_invalid_transform";return false;}
  auto delta=rest.conjugate()*current;delta.normalize();if(delta.w<0){delta.w=-delta.w;delta.x=-delta.x;delta.y=-delta.y;delta.z=-delta.z;}
  const float projection=delta.x*view.rule.axis.x+delta.y*view.rule.axis.y+delta.z*view.rule.axis.z;
  Quaternion twist(delta.w,view.rule.axis.x*projection,view.rule.axis.y*projection,view.rule.axis.z*projection);
  if(twist.w*twist.w+projection*projection<1e-10f)twist=Quaternion();else twist.normalize();
  view.twist=2*std::atan2(twist.x*view.rule.axis.x+twist.y*view.rule.axis.y+twist.z*view.rule.axis.z,twist.w)/rad;
  auto swing=delta*twist.conjugate();swing.normalize();view.swing=2*std::acos((std::max)(0.f,(std::min)(1.f,std::fabs(swing.w))))/rad;
  view.outside=jointOutsideLimits(view.rule,view.twist,view.swing);
  if(scene.rigView.pose.active&&scene.rigView.pose.character==character){const auto& p=scene.rigView.pose;auto controls=p.hasPreview&&p.frame==scene.timeline.current_frame?p.previewIK:p.ik;if(p.frame!=scene.timeline.current_frame)for(auto i=controls.begin();i!=controls.end();)if(!i->second.contact)i=controls.erase(i);else ++i;view.ik=ikDrivesBone(model.rigAnatomy.controls,controls,bone);}
  output=std::move(view);return true;
 }error="unknown_character";return false;
}
nlohmann::json jointLimitViewJson(const JointLimitView& v){auto matrix=[](const Matrix4x4& m){auto value=nlohmann::json::array();for(int r=0;r<4;++r)for(int c=0;c<4;++c)value.push_back(m.m[r][c]);return value;};return {{"character",v.character},{"bone",v.bone},{"rig_revision",v.revision},{"owned",v.owned},{"has_rule",v.hasRule},{"rule",serializeJointRules({v.rule})["joints"][0]},{"neutral_world",matrix(v.neutralWorld)},{"joint_world",matrix(v.jointWorld)},{"twist_degrees",v.twist},{"swing_degrees",v.swing},{"outside_limits",v.outside},{"ik_driven",v.ik},{"pose_source",v.poseSource},{"limit_space","joint_rest_relative"}};}
}
