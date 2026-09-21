#include "Animation/RigPreflight.h"
#include "Animation/RigMeshSpace.h"
#include "scene_data.h"
#include "TriangleMesh.h"
#include "Transform.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
namespace RigAuthoring {
namespace {
std::vector<std::shared_ptr<TriangleMesh>> groupParts(const SceneData& scene,const std::string& name,std::string& error) {
    const SceneData::ImportedModelContext* model=nullptr;
    for(const auto& m:scene.importedModelContexts)if(m.importName==name){if(model){error="ambiguous_mesh_group";return {};}model=&m;}
    if(!model){error="unknown_mesh_group";return {};}
    std::unordered_set<const TriangleMesh*> members,seen;
    for(const auto& o:model->members){auto mesh=std::dynamic_pointer_cast<TriangleMesh>(o);if(mesh)members.insert(mesh.get());}
    std::vector<std::shared_ptr<TriangleMesh>> parts;
    for(const auto& o:scene.world.objects){auto mesh=std::dynamic_pointer_cast<TriangleMesh>(o);if(mesh&&(members.count(mesh.get())||model->nodeHierarchy.find(mesh->nodeName))&&seen.insert(mesh.get()).second)parts.push_back(mesh);}
    if(parts.empty())error="mesh_group_has_no_flat_parts";
    return parts;
}
}
nlohmann::json fitTargets(const SceneData& scene) {
    nlohmann::json out=nlohmann::json::array();std::unordered_set<const TriangleMesh*> seen;
    for(const auto& o:scene.world.objects){auto m=std::dynamic_pointer_cast<TriangleMesh>(o);if(m&&!m->nodeName.empty()&&seen.insert(m.get()).second)out.push_back({{"target",m->nodeName},{"label",m->nodeName},{"kind","mesh"},{"part_count",1}});}
    for(const auto& model:scene.importedModelContexts){std::string error;auto parts=groupParts(scene,model.importName,error);if(parts.size()>1&&error.empty())out.push_back({{"target","model:"+model.importName},{"label","Character: "+model.importName+" ("+std::to_string(parts.size())+" parts)"},{"kind","model_group"},{"part_count",parts.size()}});}
    return out;
}
bool resolveFitParts(const SceneData& scene,const std::string& target,std::vector<std::shared_ptr<TriangleMesh>>& output,std::string& error) {
    output.clear();error.clear();if(target.empty()){error="invalid_mesh_name";return false;}
    std::shared_ptr<TriangleMesh> found;
    for(const auto& object:scene.world.objects) {
        auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);if(!mesh || mesh->nodeName!=target)continue;
        if(found && found!=mesh){error="ambiguous_mesh_name";return false;}found=mesh;
    }
    if(found){output.push_back(found);return true;}
    if(target.find("model:")!=0){error="unknown_mesh";return false;}
    output=groupParts(scene,target.substr(6),error);return error.empty();
}
std::shared_ptr<TriangleMesh> resolveFitMesh(const SceneData& scene,const std::string& target,std::string& error,bool* existingSkin) {
    error.clear();if(existingSkin)*existingSkin=false;if(target.empty()){error="invalid_mesh_name";return {};}
    // Exact mesh names take precedence over the reserved model: group syntax.
    std::shared_ptr<TriangleMesh> mesh;
    for(const auto& o:scene.world.objects){auto m=std::dynamic_pointer_cast<TriangleMesh>(o);if(!m||m->nodeName!=target)continue;if(mesh&&mesh!=m){error="ambiguous_mesh_name";return {};}mesh=m;}
    if(mesh){if(existingSkin)*existingSkin=mesh->hasSkinWeights();return mesh;}
    if(target.find("model:")!=0){error="unknown_mesh";return {};}
    auto parts=groupParts(scene,target.substr(6),error);if(!error.empty())return {};
    size_t count=0;bool weighted=false;
    for(const auto& part:parts) {
        nlohmann::json report;if(!preflightMesh(scene,part->nodeName,report,error))return {};
        if(report["nonfinite_vertices"].get<size_t>()||report["invalid_triangles"].get<size_t>()||report["trailing_indices"].get<size_t>()){error="mesh_group_invalid_geometry";return {};}
        if(part->num_vertices()>std::numeric_limits<uint32_t>::max()-count){error="mesh_group_too_large";return {};}
        count+=part->num_vertices();weighted=weighted||part->hasSkinWeights();
    }
    auto merged=std::make_shared<TriangleMesh>();merged->nodeName=target;merged->geometry->resize_vertices(count);
    auto* positions=merged->geometry->get_positions_mut();size_t offset=0;
    for(const auto& part:parts) {
        const auto t=part->transform?part->transform->getFinal():Matrix4x4::identity();const auto* p=meshSpacePositions(*part->geometry,part->hasSkinWeights());
        for(size_t i=0;i<part->num_vertices();++i)positions[offset+i]=Vec3(t.m[0][0]*p[i].x+t.m[0][1]*p[i].y+t.m[0][2]*p[i].z+t.m[0][3],t.m[1][0]*p[i].x+t.m[1][1]*p[i].y+t.m[1][2]*p[i].z+t.m[1][3],t.m[2][0]*p[i].x+t.m[2][1]*p[i].y+t.m[2][2]*p[i].z+t.m[2][3]);
        for(auto index:part->geometry->indices)merged->geometry->indices.push_back(static_cast<uint32_t>(offset)+index);
        offset+=part->num_vertices();
    }
    if(existingSkin)*existingSkin=weighted;
    return merged;
}

bool preflightMesh(const SceneData& scene,const std::string& name,nlohmann::json& report,std::string& error) {
    report=nullptr;error.clear();bool existingSkin=false;auto mesh=resolveFitMesh(scene,name,error,&existingSkin);
    if(!mesh)return false;
    if(!mesh->geometry){error="mesh_has_no_geometry";return false;}
    const auto& g=*mesh->geometry;const size_t count=g.get_vertex_count();
    const Vec3* positions=meshSpacePositions(g,mesh->hasSkinWeights());
    if(!count || !positions){error="mesh_has_no_positions";return false;}
    if(meshSpacePositionCount(g,mesh->hasSkinWeights())<count){error="mesh_position_buffer_incomplete";return false;}
    const auto transform=mesh->transform?mesh->transform->getFinal():Matrix4x4::identity();
    for(int row=0;row<4;++row)for(int col=0;col<4;++col)
        if(!std::isfinite(transform.m[row][col])){error="invalid_mesh_transform";return false;}
    for(int col=0;col<4;++col)if(std::fabs(transform.m[3][col]-(col==3?1.f:0.f))>1e-6f) {
        error="invalid_mesh_transform";return false;
    }
    auto point=[&](const Vec3& p) {
        return Vec3(transform.m[0][0]*p.x+transform.m[0][1]*p.y+transform.m[0][2]*p.z+transform.m[0][3],
                    transform.m[1][0]*p.x+transform.m[1][1]*p.y+transform.m[1][2]*p.z+transform.m[1][3],
                    transform.m[2][0]*p.x+transform.m[2][1]*p.y+transform.m[2][2]*p.z+transform.m[2][3]);
    };
    const float inf=std::numeric_limits<float>::infinity();Vec3 lo(inf,inf,inf),hi(-inf,-inf,-inf);
    size_t nonfinite=0,invalidTriangles=0,degenerate=0;
    for(size_t i=0;i<count;++i) {
        const auto p=point(positions[i]);
        if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)){++nonfinite;continue;}
        lo.x=std::min(lo.x,p.x);lo.y=std::min(lo.y,p.y);lo.z=std::min(lo.z,p.z);
        hi.x=std::max(hi.x,p.x);hi.y=std::max(hi.y,p.y);hi.z=std::max(hi.z,p.z);
    }
    const auto& indices=g.indices;
    for(size_t i=0;i+2<indices.size();i+=3) {
        const auto a=indices[i],b=indices[i+1],c=indices[i+2];
        if(a>=count||b>=count||c>=count){++invalidTriangles;continue;}
        const Vec3 pa=point(positions[a]),pb=point(positions[b]),pc=point(positions[c]);
        const auto cross=(pb-pa).cross(pc-pa);const float area2=cross.dot(cross);
        if(!std::isfinite(area2)){++invalidTriangles;continue;}
        if(area2<=0)++degenerate;
    }
    const bool finiteBounds=nonfinite<count;
    const Vec3 extent=finiteBounds?hi-lo:Vec3(0,0,0);
    const bool volumetric=finiteBounds && std::isfinite(extent.x) && std::isfinite(extent.y) && std::isfinite(extent.z) && extent.x>0 && extent.y>0 && extent.z>0;
    const bool weighted=existingSkin;
    const bool inspectable=!nonfinite && !invalidTriangles && degenerate<indices.size()/3 && indices.size()>=3 && indices.size()%3==0 && volumetric;
    report={{"mesh",name},{"vertex_count",count},{"triangle_count",indices.size()/3},
        {"nonfinite_vertices",nonfinite},{"invalid_triangles",invalidTriangles},{"degenerate_triangles",degenerate},
        {"trailing_indices",indices.size()%3},{"has_skin_weights",weighted},{"geometry_valid",inspectable},
        {"can_start_landmarks",inspectable&&!weighted},{"fit_ready",false},
        {"position_source","flat_viewport_positions_with_final_transform"},{"closed_surface","not_evaluated"},
        {"pose","not_evaluated"},{"axes","not_confirmed"},
        {"warnings",{"Confirm axes and rest pose before fitting.","Surface interior and symmetry have not been evaluated.","No fitting or mesh binding is performed."}}};
    report["bounds"]=(finiteBounds && std::isfinite(extent.x) && std::isfinite(extent.y) && std::isfinite(extent.z))?nlohmann::json{{"min",{lo.x,lo.y,lo.z}},{"max",{hi.x,hi.y,hi.z}},{"extent",{extent.x,extent.y,extent.z}}}:nlohmann::json(nullptr);
    report["blockers"]=nlohmann::json::array();
    if(nonfinite)report["blockers"].push_back("nonfinite_vertices");
    if(invalidTriangles)report["blockers"].push_back("invalid_triangles");
    if(indices.size()%3)report["blockers"].push_back("trailing_indices");
    if(indices.size()<3 || degenerate==indices.size()/3)report["blockers"].push_back("no_valid_faces");
    if(!volumetric)report["blockers"].push_back("bounds_have_no_volume");
    if(weighted)report["blockers"].push_back("target_already_skinned");
    if(name.find("model:")==0) {
        std::string groupError;auto parts=groupParts(scene,name.substr(6),groupError);
        if(groupError.empty()){report["part_count"]=parts.size();report["parts"]=nlohmann::json::array();for(const auto& p:parts)report["parts"].push_back(p->nodeName);}
    }
    if(degenerate)report["warnings"].push_back("Zero-area faces are reported; manual landmark setup can continue when valid faces remain.");
    return true;
}
}
