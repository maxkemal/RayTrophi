#include "Animation/RigFitting.h"
#include "Animation/RigMeshSpace.h"
#include "Animation/RigPreflight.h"
#include "Animation/RigEditing.h"
#include "Animation/RigPosePreview.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <utility>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>
namespace RigAuthoring {
namespace {
Vec3 point(const Matrix4x4& m,const Vec3& p) {
    return Vec3(m.m[0][0]*p.x+m.m[0][1]*p.y+m.m[0][2]*p.z+m.m[0][3],m.m[1][0]*p.x+m.m[1][1]*p.y+m.m[1][2]*p.z+m.m[1][3],m.m[2][0]*p.x+m.m[2][1]*p.y+m.m[2][2]*p.z+m.m[2][3]);
}
bool vec(const nlohmann::json& j,Vec3& p) {
    if(!j.is_array()||j.size()!=3)return false;
    for(int i=0;i<3;++i)if(!j[i].is_number())return false;
    p=Vec3(j[0].get<float>(),j[1].get<float>(),j[2].get<float>());
    return std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z);
}
bool input(const SceneData& scene,const std::string& character,const std::string& name,
           const SceneData::ImportedModelContext*& model,std::shared_ptr<TriangleMesh>& mesh,nlohmann::json& pre,std::string& error) {
    if(!canEditRig(scene,character,error))return false;
    if(!preflightMesh(scene,name,pre,error))return false;
    if(!pre["can_start_landmarks"].get<bool>()){error="rig_fit_mesh_not_ready";return false;}
    for(const auto& m:scene.importedModelContexts)if(m.importName==character){model=&m;break;}
    mesh=resolveFitMesh(scene,name,error);if(!mesh)return false;
    if(!model||!mesh){error="rig_fit_input_missing";return false;}
    if(model->nodeHierarchy.size()>4096){error="rig_fit_limit";return false;}
    return true;
}
std::string meshToken(const TriangleMesh& mesh) {
    uint64_t hash=1469598103934665603ull;
    auto add=[&](uint32_t value){hash^=value;hash*=1099511628211ull;};
    const auto& g=*mesh.geometry;add(static_cast<uint32_t>(g.get_vertex_count()));
    const auto* p=meshSpacePositions(g,mesh.hasSkinWeights());
    for(size_t i=0;i<g.get_vertex_count();++i)for(float v:{p[i].x,p[i].y,p[i].z}){uint32_t bits;std::memcpy(&bits,&v,sizeof(bits));add(bits);}
    for(auto i:g.indices)add(i);
    const auto m=mesh.transform?mesh.transform->getFinal():Matrix4x4::identity();
    for(int r=0;r<4;++r)for(int c=0;c<4;++c){uint32_t bits;std::memcpy(&bits,&m.m[r][c],sizeof(bits));add(bits);}
    return std::to_string(hash);
}
bool resolveLandmarks(const SceneData::ImportedModelContext& model,const nlohmann::json& input,
                      nlohmann::json& resolved,std::string& error) {
    if(!input.is_object()){error="rig_fit_incomplete_landmarks";return false;}
    std::unordered_set<std::string> derived;
    std::unordered_set<std::string> known;
    for(const auto& rule:model.rigAnatomy.fitRules)derived.insert(rule.bone);
    resolved=nlohmann::json::object();
    for(const auto& node:model.nodeHierarchy.nodes) {
        known.insert(node.uniqueName);
        if(derived.count(node.uniqueName))continue;
        Vec3 p;if(!input.contains(node.uniqueName)||!vec(input.at(node.uniqueName),p)) {
            error="rig_fit_incomplete_landmarks";return false;
        }
        resolved[node.uniqueName]={p.x,p.y,p.z};
    }
    for(const auto& item:input.items())if(!known.count(item.key())) {
        error="rig_fit_invalid_landmark";return false;
    }
    for(const auto& rule:model.rigAnatomy.fitRules) {
        Vec3 start,end;
        if(!resolved.contains(rule.start)||!resolved.contains(rule.end)||
           !vec(resolved.at(rule.start),start)||!vec(resolved.at(rule.end),end)) {
            error="rig_anatomy_fit_rule_dependency";return false;
        }
        const auto p=start+(end-start)*rule.position;
        resolved[rule.bone]={p.x,p.y,p.z};
    }
    if(resolved.size()!=model.nodeHierarchy.size()) {
        error="rig_fit_incomplete_landmarks";return false;
    }
    return true;
}
}
bool fitSetup(const SceneData& scene,const std::string& character,const std::string& name,nlohmann::json& out,std::string& error) {
    out=nullptr;const SceneData::ImportedModelContext* model=nullptr;std::shared_ptr<TriangleMesh> mesh;nlohmann::json pre;
    if(!input(scene,character,name,model,mesh,pre,error))return false;
    std::vector<PreviewJoint> joints;if(!sampleRigPose(model->nodeHierarchy,nullptr,0,joints,error))return false;
    const float inf=std::numeric_limits<float>::infinity();Vec3 lo(inf,inf,inf),hi(-inf,-inf,-inf);
    std::vector<Vec3> world;
    for(const auto& j:joints){auto p=point(model->rigSceneTransform,Vec3(j.world.m[0][3],j.world.m[1][3],j.world.m[2][3]));if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){error="invalid_rig_scene_transform";return false;}world.push_back(p);lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);}
    if(!std::isfinite(hi.y-lo.y)||hi.y-lo.y<=1e-6f){error="rig_fit_requires_height";return false;}
    Vec3 ml,mh;vec(pre["bounds"]["min"],ml);vec(pre["bounds"]["max"],mh);
    const float scale=(mh.y-ml.y)/(hi.y-lo.y);
    const Vec3 origin((lo.x+hi.x)*.5f,lo.y,(lo.z+hi.z)*.5f),target((ml.x+mh.x)*.5f,ml.y,(ml.z+mh.z)*.5f);
    nlohmann::json marks=nlohmann::json::object(),rows=nlohmann::json::array(),points=nlohmann::json::array();
    std::unordered_set<std::string> derived;
    for(const auto& rule:model->rigAnatomy.fitRules)derived.insert(rule.bone);
    std::unordered_map<std::string,std::string> roles;
    for(const auto& role:model->rigAnatomy.roles)roles[role.bone]=role.role;
    for(size_t i=0;i<joints.size();++i){
        const auto p=(world[i]-origin)*scale+target;
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){error="rig_fit_invalid_landmark";return false;}
        marks[joints[i].name]={p.x,p.y,p.z};const int parent=model->nodeHierarchy.nodes[i].parent;
        const auto role=roles.find(joints[i].name);const std::string roleName=role==roles.end()?"":role->second;
        rows.push_back({{"name",joints[i].name},{"parent",parent<0?"":model->nodeHierarchy.nodes[parent].uniqueName},
                        {"fit_editable",derived.count(joints[i].name)==0},
                        {"fit_group",roleName.find("_hand.")!=std::string::npos?"hands":"primary"}});
    }
    auto guideMarks=marks;
    for(const auto& rule:model->rigAnatomy.fitRules)guideMarks.erase(rule.bone);
    nlohmann::json resolvedMarks;
    if(!resolveLandmarks(*model,guideMarks,resolvedMarks,error))return false;
    marks=std::move(resolvedMarks);
    const auto t=mesh->transform?mesh->transform->getFinal():Matrix4x4::identity();const auto* positions=meshSpacePositions(*mesh->geometry,mesh->hasSkinWeights());const size_t count=mesh->num_vertices(),stride=std::max(size_t(1),(count+8191)/8192);
    for(size_t i=0;i<count;i+=stride){auto p=point(t,positions[i]);points.push_back({p.x,p.y,p.z});}
    out={{"character",character},{"mesh",name},{"rig_revision",model->rigRevision},{"mesh_token",meshToken(*mesh)},
         {"landmarks",marks},{"joints",rows},{"mesh_points",points},{"bounds",pre["bounds"]},{"axes", "+Y up, +Z forward; user confirmation required"}};
    return true;
}
bool previewFit(const SceneData& scene,const std::string& character,const std::string& name,const nlohmann::json& marks,bool confirmed,nlohmann::json& out,std::string& error) {
    out=nullptr;if(!confirmed){error="rig_fit_axes_unconfirmed";return false;}
    const SceneData::ImportedModelContext* model=nullptr;std::shared_ptr<TriangleMesh> mesh;nlohmann::json pre;
    if(!input(scene,character,name,model,mesh,pre,error))return false;
    nlohmann::json resolved;if(!resolveLandmarks(*model,marks,resolved,error))return false;
    Vec3 lo,hi;vec(pre["bounds"]["min"],lo);vec(pre["bounds"]["max"],hi);
    for(const auto& node:model->nodeHierarchy.nodes) {
        Vec3 p;if(!vec(resolved.at(node.uniqueName),p)){error="rig_fit_invalid_landmark";return false;}
    }
    const float tolerance=1e-5f*std::max({hi.x-lo.x,hi.y-lo.y,hi.z-lo.z})+
        2.f*std::numeric_limits<float>::epsilon()*std::max({std::fabs(lo.x),std::fabs(lo.y),std::fabs(lo.z),std::fabs(hi.x),std::fabs(hi.y),std::fabs(hi.z)});
    nlohmann::json rows=nlohmann::json::array();size_t outside=0;
    for(const auto& node:model->nodeHierarchy.nodes) {
        Vec3 p;if(!vec(resolved.at(node.uniqueName),p)){error="rig_fit_invalid_landmark";return false;}
        const bool in=p.x>=lo.x-tolerance&&p.x<=hi.x+tolerance&&p.y>=lo.y-tolerance&&p.y<=hi.y+tolerance&&p.z>=lo.z-tolerance&&p.z<=hi.z+tolerance;
        outside+=!in;float length=0;
        if(node.parent>=0){Vec3 parent;if(!vec(resolved.at(model->nodeHierarchy.nodes[node.parent].uniqueName),parent)){error="rig_fit_invalid_landmark";return false;}const auto delta=p-parent;length=std::sqrt(delta.dot(delta));if(!std::isfinite(length)||length<=1e-6f){error="rig_fit_zero_length_bone";return false;}}
        rows.push_back({{"name",node.uniqueName},{"world_position",{p.x,p.y,p.z}},{"segment_length",length},{"inside_bounds",in}});
    }
    out={{"character",character},{"mesh",name},{"rig_revision",model->rigRevision},{"mesh_token",meshToken(*mesh)},{"landmarks",resolved},
         {"axes_confirmed",true},{"mode",model->rigAnatomy.fitRules.empty()?"manual_landmarks":"guided_landmarks"},{"joints",rows},{"outside_bounds",outside},{"bounds_tolerance",tolerance},{"can_commit",outside==0},
         {"interior_verified",false},{"warnings",{"Bounds are not a surface interior test. Inspect both projections before applying.","This applies manual rest landmarks only; no automatic fitting, binding or weights."}}};
    return true;
}
bool fittedHierarchy(const SceneData& scene,const std::string& character,const std::string& name,const nlohmann::json& preview,RayTrophi::NodeHierarchy& out,std::string& error) {
    const SceneData::ImportedModelContext* model=nullptr;std::shared_ptr<TriangleMesh> mesh;nlohmann::json pre;
    if(!input(scene,character,name,model,mesh,pre,error))return false;
    if(!preview.is_object()||!preview.contains("character")||!preview["character"].is_string()||!preview.contains("mesh")||!preview["mesh"].is_string()||
       !preview.contains("axes_confirmed")||!preview["axes_confirmed"].is_boolean()||preview.value("character",std::string())!=character||preview.value("mesh",std::string())!=name||
       !preview.contains("rig_revision")||!preview["rig_revision"].is_number_unsigned()||!preview.contains("mesh_token")||!preview["mesh_token"].is_string()||!preview.contains("landmarks")) {error="rig_fit_invalid_preview";return false;}
    if(preview["rig_revision"].get<uint64_t>()!=model->rigRevision||preview["mesh_token"].get<std::string>()!=meshToken(*mesh)){error="rig_fit_stale_preview";return false;}
    nlohmann::json verified;if(!previewFit(scene,character,name,preview["landmarks"],preview.value("axes_confirmed",false),verified,error))return false;
    if(!verified["can_commit"].get<bool>()){error="rig_fit_landmarks_outside_bounds";return false;}
    auto h=model->nodeHierarchy;std::vector<PreviewJoint> current;if(!sampleRigPose(h,nullptr,0,current,error))return false;
    const auto inverse=model->rigSceneTransform.inverse();std::vector<Matrix4x4> globals;
    for(size_t i=0;i<h.size();++i){Vec3 p;vec(verified["landmarks"][h.nodes[i].uniqueName],p);p=point(inverse,p);auto g=current[i].world;g.m[0][3]=p.x;g.m[1][3]=p.y;g.m[2][3]=p.z;globals.push_back(g);}
    for(size_t i=0;i<h.size();++i){const int parent=h.nodes[i].parent;h.nodes[i].localBind=parent<0?globals[i]:globals[parent].inverse()*globals[i];}
    out=std::move(h);return true;
}
}
