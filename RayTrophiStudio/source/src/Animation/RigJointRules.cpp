#include "Animation/RigJointRules.h"
#include "Animation/AnimationKeys.h"
#include "Animation/RigPosePreview.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>
namespace RigAuthoring {
namespace {
constexpr float pi=3.14159265359f;
float clamp(float v,float lo,float hi){return (std::max)(lo,(std::min)(hi,v));}
bool fields(const nlohmann::json& value,std::initializer_list<const char*> allowed){if(!value.is_object())return false;for(const auto& p:value.items()){bool found=false;for(const auto* key:allowed)found=found||p.key()==key;if(!found)return false;}return true;}
bool rigid(const Matrix4x4& m,Vec3& p,Quaternion& q){Vec3 scale;RayTrophi::decomposeTRS(m,p,q,scale);q.normalize();const auto rebuilt=Matrix4x4::translation(p)*q.toMatrix();for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(m.m[r][c])||!std::isfinite(rebuilt.m[r][c])||std::fabs(m.m[r][c]-rebuilt.m[r][c])>1e-4f)return false;return true;}
Quaternion axisAngle(const Vec3& axis,float radians){const auto s=std::sin(radians*.5f);return Quaternion(std::cos(radians*.5f),axis.x*s,axis.y*s,axis.z*s);}
}
bool validateJointRules(const std::vector<JointRule>& rules,const RayTrophi::NodeHierarchy& h,std::string& error){
 error.clear();if(rules.size()>4096){error="rig_joint_limit";return false;}std::unordered_set<std::string> bones;
 for(const auto& rule:rules){
  if(!h.find(rule.bone)){error="unknown_bone";return false;}
  if(!bones.insert(rule.bone).second){error="rig_joint_duplicate_bone";return false;}
  if(rule.type!="free"&&rule.type!="hinge"&&rule.type!="ball"&&rule.type!="fixed"){error="rig_joint_invalid_type";return false;}
  if(!std::isfinite(rule.axis.x)||!std::isfinite(rule.axis.y)||!std::isfinite(rule.axis.z)||std::fabs(rule.axis.length_squared()-1.f)>1e-4f){error="rig_joint_invalid_axis";return false;}
  if(!std::isfinite(rule.minimum)||!std::isfinite(rule.maximum)||!std::isfinite(rule.swing)||rule.minimum<-180||rule.maximum>180||rule.minimum>0||rule.maximum<0||rule.swing<0||rule.swing>180){error="rig_joint_invalid_range";return false;}
 }return true;
}
nlohmann::json serializeJointRules(const std::vector<JointRule>& rules){auto rows=nlohmann::json::array();for(const auto& r:rules)rows.push_back({{"bone",r.bone},{"type",r.type},{"enabled",r.enabled},{"lock_translation",r.lockTranslation},{"axis",{r.axis.x,r.axis.y,r.axis.z}},{"minimum",r.minimum},{"maximum",r.maximum},{"swing",r.swing}});return {{"version",1},{"joints",rows}};}
bool deserializeJointRules(const nlohmann::json& value,const RayTrophi::NodeHierarchy& h,std::vector<JointRule>& out,std::string& error){
 error.clear();try{
  if(!fields(value,{"version","joints"})||!value.contains("version")||!value["version"].is_number_integer()||value["version"]!=1||!value.contains("joints")||!value["joints"].is_array()){error="rig_joint_invalid_schema";return false;}
  if(value["joints"].size()>4096){error="rig_joint_limit";return false;}std::vector<JointRule> staged;
  for(const auto& row:value["joints"]){
   if(!fields(row,{"bone","type","enabled","lock_translation","axis","minimum","maximum","swing"})||!row.contains("bone")||!row["bone"].is_string()){error="rig_joint_invalid_schema";return false;}
   JointRule r;r.bone=row["bone"].get<std::string>();
   if(row.contains("type")){if(!row["type"].is_string()){error="rig_joint_invalid_schema";return false;}r.type=row["type"].get<std::string>();}
   for(const auto* key:{"enabled","lock_translation"})if(row.contains(key)&&!row[key].is_boolean()){error="rig_joint_invalid_schema";return false;}
   r.enabled=row.value("enabled",false);r.lockTranslation=row.value("lock_translation",true);
   if(row.contains("axis")){if(!row["axis"].is_array()||row["axis"].size()!=3){error="rig_joint_invalid_axis";return false;}for(const auto& v:row["axis"])if(!v.is_number()){error="rig_joint_invalid_axis";return false;}r.axis=Vec3(row["axis"][0].get<float>(),row["axis"][1].get<float>(),row["axis"][2].get<float>());}
   for(const auto* key:{"minimum","maximum","swing"})if(row.contains(key)&&!row[key].is_number()){error="rig_joint_invalid_range";return false;}
   r.minimum=row.value("minimum",-180.f);r.maximum=row.value("maximum",180.f);r.swing=row.value("swing",180.f);staged.push_back(r);
  }
  if(!validateJointRules(staged,h,error))return false;out=std::move(staged);return true;
 }catch(const nlohmann::json::exception&){error="rig_joint_invalid_schema";return false;}
}
bool constrainJointPose(const RayTrophi::NodeHierarchy& rest,const std::vector<JointRule>& rules,const RayTrophi::NodeHierarchy& input,RayTrophi::NodeHierarchy& output,std::vector<std::string>& hits,std::string& error){
 hits.clear();if(!validateJointRules(rules,rest,error))return false;
 if(input.size()!=rest.size()){error="rig_joint_hierarchy_mismatch";return false;}
 for(size_t i=0;i<rest.size();++i)if(input.nodes[i].uniqueName!=rest.nodes[i].uniqueName||input.nodes[i].parent!=rest.nodes[i].parent){error="rig_joint_hierarchy_mismatch";return false;}
 auto staged=input;std::vector<std::string> limited;
 for(const auto& r:rules){if(!r.enabled)continue;
  auto* node=&staged.nodes[static_cast<size_t>(rest.find(r.bone)-rest.nodes.data())];const auto* reference=rest.find(r.bone);Vec3 p,rp;Quaternion q,rq;
  if(!rigid(node->localBind,p,q)||!rigid(reference->localBind,rp,rq)){error="rig_joint_requires_rigid_pose";return false;}
  if(r.lockTranslation)p=rp;
  if(r.type=="fixed")q=rq;
  else if(r.type!="free"){
   auto delta=rq.conjugate()*q;delta.normalize();if(delta.w<0){delta.w=-delta.w;delta.x=-delta.x;delta.y=-delta.y;delta.z=-delta.z;}
   const float projection=delta.x*r.axis.x+delta.y*r.axis.y+delta.z*r.axis.z;
   Quaternion twist(delta.w,r.axis.x*projection,r.axis.y*projection,r.axis.z*projection);
   if(twist.w*twist.w+projection*projection<1e-10f)twist=Quaternion();else twist.normalize();
   float angle=2*std::atan2(twist.x*r.axis.x+twist.y*r.axis.y+twist.z*r.axis.z,twist.w)*180/pi;
   auto swing=delta*twist.conjugate();swing.normalize();if(swing.w<0){swing.w=-swing.w;swing.x=-swing.x;swing.y=-swing.y;swing.z=-swing.z;}
   const float swingAngle=2*std::acos(clamp(swing.w,-1,1));const float cone=(r.type=="hinge"?0:r.swing)*pi/180;
   if(swingAngle>cone && swingAngle>1e-6f){const float length=std::sqrt(swing.x*swing.x+swing.y*swing.y+swing.z*swing.z);swing=length>1e-6f?axisAngle(Vec3(swing.x/length,swing.y/length,swing.z/length),cone):Quaternion();}
   q=rq*swing*axisAngle(r.axis,clamp(angle,r.minimum,r.maximum)*pi/180);q.normalize();
  }
  const auto result=Matrix4x4::translation(p)*q.toMatrix();bool changed=false;
  for(int a=0;a<4;++a)for(int b=0;b<4;++b)changed=changed||std::fabs(node->localBind.m[a][b]-result.m[a][b])>1e-6f;
  if(changed){node->localBind=result;limited.push_back(r.bone);}
 }
 std::vector<PreviewJoint> validate;if(!sampleRigPose(staged,nullptr,0,validate,error))return false;
 output=std::move(staged);hits=std::move(limited);return true;
}
}
