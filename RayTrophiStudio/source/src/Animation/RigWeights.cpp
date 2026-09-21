#include "Animation/RigWeights.h"
#include "Animation/RigBindingScope.h"
#include "scene_data.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
namespace RigAuthoring {
namespace {
struct WeightSource {
    std::shared_ptr<TriangleMesh> mesh;
    const SceneData::ImportedModelContext* owner = nullptr;
    std::string ownership = "unresolved";
    std::unordered_map<int, std::string> names;
    std::unordered_set<int> ambiguousIndices;
    std::unordered_set<std::string> ownerKeys;
};
bool resolveWeightSource(const SceneData& scene, const std::string& name, WeightSource& source, std::string& error) {
    error.clear();
    if (name.empty()) { error = "invalid_mesh_name"; return false; }
    for (const auto& object : scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (!mesh || mesh->nodeName != name) continue;
        if (source.mesh && source.mesh != mesh) { error = "ambiguous_mesh_name"; return false; }
        source.mesh = mesh;
    }
    if (!source.mesh) { error = "unknown_mesh"; return false; }
    if (!source.mesh->geometry) { error = "mesh_has_no_geometry"; return false; }
    const auto* explicitOwner=explicitMeshRig(scene,name);
    size_t owners = 0;
    for (const auto& model : scene.importedModelContexts) {
        // Exact canonical identity only: neither authored-name nor prefix guessing.
        bool member = std::any_of(model.nodeHierarchy.nodes.begin(), model.nodeHierarchy.nodes.end(),
            [&](const auto& node) { return node.uniqueName == name; });
        for (const auto& object : model.members) {
            if (object.get() == source.mesh.get()) { member = true; break; }
        }
        if (explicitOwner) member=model.importName==*explicitOwner;
        if (member) { ++owners; source.owner = &model; }
    }
    if (owners == 1) {
        source.ownership = "resolved";
        for (const auto& node : source.owner->nodeHierarchy.nodes) source.ownerKeys.insert(node.uniqueName);
    } else {
        source.owner = nullptr;
        if (owners > 1) source.ownership = "ambiguous";
    }
    // Forward map is authoritative; reverse cache may contain stale removed slots.
    for (const auto& entry : scene.boneData.boneNameToIndex) {
        if (entry.second > static_cast<unsigned int>(std::numeric_limits<int>::max())) continue;
        const int index = static_cast<int>(entry.second);
        if (!source.names.emplace(index, entry.first).second) source.ambiguousIndices.insert(index);
    }
    return true;
}
const std::string* boneName(const WeightSource& source, int index) {
    if (source.ambiguousIndices.count(index)) return nullptr;
    const auto found = source.names.find(index);
    return found == source.names.end() ? nullptr : &found->second;
}
nlohmann::json ownershipReport(const WeightSource& source) {
    return {{"ownership", source.ownership},
        {"character", source.owner ? nlohmann::json(source.owner->importName) : nlohmann::json(nullptr)},
        {"ownership_verified", source.owner && !source.ownerKeys.empty()}};
}
}
bool hasFlatSkinReferences(const SceneData& scene,const RayTrophi::NodeHierarchy& hierarchy) {
    std::unordered_set<int> indices;
    for(const auto& node:hierarchy.nodes) {
        const auto found=scene.boneData.boneNameToIndex.find(node.uniqueName);
        if(found!=scene.boneData.boneNameToIndex.end() &&
           found->second<=static_cast<unsigned int>(std::numeric_limits<int>::max()))
            indices.insert(static_cast<int>(found->second));
    }
    if(indices.empty())return false;
    std::unordered_set<const DNA::GeometryDetail*> seen;
    for(const auto& object:scene.world.objects) {
        const auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);
        if(!mesh || !mesh->geometry || !seen.insert(mesh->geometry.get()).second)continue;
        // Any stored reference blocks the unskinned subset, including invalid/zero
        // entries and malformed extra rows. Never silently orphan an index.
        for(const auto& row:mesh->geometry->skin_weights)
            for(const auto& influence:row)if(indices.count(influence.first))return true;
    }
    return false;
}
bool weightStats(const SceneData& scene,const std::string& name,nlohmann::json& report,std::string& error) {
    report=nullptr;WeightSource source;
    if(!resolveWeightSource(scene,name,source,error))return false;
    const auto& g=*source.mesh->geometry;const size_t count=g.get_vertex_count();
    size_t empty=0,invalid=0,duplicates=0,unsorted=0,overLimit=0,unnormalized=0,maxInfluences=0;
    size_t unknownBones=0,ambiguousBones=0,foreignBones=0;
    double minSum=std::numeric_limits<double>::infinity(),maxSum=0;
    for(size_t vertex=0;vertex<count;++vertex) {
        if(vertex>=g.skin_weights.size() || g.skin_weights[vertex].empty()){++empty;continue;}
        const auto& weights=g.skin_weights[vertex];maxInfluences=std::max(maxInfluences,weights.size());
        if(weights.size()>4)++overLimit;
        std::unordered_set<int> ids;double sum=0;float previous=std::numeric_limits<float>::infinity();
        bool badOrder=false;
        for(const auto& w:weights) {
            if(!ids.insert(w.first).second)++duplicates;
            const auto* key=boneName(source,w.first);
            if(source.ambiguousIndices.count(w.first))++ambiguousBones;
            else if(!key)++unknownBones;
            else if(source.owner && !source.ownerKeys.empty() && !source.ownerKeys.count(*key))++foreignBones;
            if(w.first<0 || !std::isfinite(w.second) || w.second<=0){++invalid;continue;}
            if(w.second>previous)badOrder=true;previous=w.second;sum+=w.second;
        }
        if(badOrder)++unsorted;
        if(std::fabs(sum-1.0)>1e-5)++unnormalized;
        minSum=std::min(minSum,sum);maxSum=std::max(maxSum,sum);
    }
    const size_t extra=g.skin_weights.size()>count?g.skin_weights.size()-count:0;
    report={{"mesh",name},{"vertex_count",count},{"weight_rows",g.skin_weights.size()},
        {"unweighted_vertices",empty},{"max_influences",maxInfluences},
        {"min_weight_sum",std::isfinite(minSum)?nlohmann::json(minSum):nlohmann::json(nullptr)},
        {"max_weight_sum",std::isfinite(minSum)?nlohmann::json(maxSum):nlohmann::json(nullptr)},
        {"invalid_entries",invalid},{"duplicate_entries",duplicates},{"unsorted_vertices",unsorted},
        {"over_limit_vertices",overLimit},{"unnormalized_vertices",unnormalized},{"extra_weight_rows",extra},
        {"contract_valid",!invalid&&!duplicates&&!unsorted&&!overLimit&&!unnormalized&&!extra},
        {"fully_weighted",count>0 && empty==0},
        {"unknown_bone_entries",unknownBones},{"ambiguous_bone_entries",ambiguousBones},{"foreign_bone_entries",foreignBones},
        {"index_range_valid",!unknownBones&&!ambiguousBones},
        {"bone_indices_verified",source.owner && !source.ownerKeys.empty() && !unknownBones && !ambiguousBones && !foreignBones}};
    report.update(ownershipReport(source));
    report["weights_valid"]=report["contract_valid"].get<bool>() && report["fully_weighted"].get<bool>() && report["bone_indices_verified"].get<bool>();
    return true;
}

bool boneWeightField(const SceneData& scene,const std::string& mesh,const std::string& character,const std::string& bone,BoneWeightField& output,std::string& error) {
    output=BoneWeightField{};WeightSource source;
    if(!resolveWeightSource(scene,mesh,source,error))return false;
    if(!source.owner || source.ownerKeys.empty()){error="rig_weight_map_unverified_owner";return false;}
    if(source.owner->importName!=character){error="rig_weight_map_foreign_character";return false;}
    if(!source.ownerKeys.count(bone)){error="unknown_bone";return false;}
    const auto found=scene.boneData.boneNameToIndex.find(bone);
    if(found==scene.boneData.boneNameToIndex.end() || found->second>static_cast<unsigned int>(std::numeric_limits<int>::max()) || source.ambiguousIndices.count(static_cast<int>(found->second))){error="rig_weight_map_invalid_index";return false;}
    const auto& geometry=*source.mesh->geometry;
    if(geometry.get_vertex_count()>2000000){error="rig_weight_map_limit";return false;}
    BoneWeightField staged;staged.bone_index=static_cast<int>(found->second);staged.values.resize(geometry.get_vertex_count(),0.f);
    for(size_t vertex=0;vertex<staged.values.size() && vertex<geometry.skin_weights.size();++vertex) {
        double sum=0;
        for(const auto& w:geometry.skin_weights[vertex])if(w.first==staged.bone_index) {
            if(!std::isfinite(w.second) || w.second<0){++staged.invalid_entries;continue;}sum+=w.second;
        }
        staged.values[vertex]=static_cast<float>(std::clamp(sum,0.,1.));
    }
    output=std::move(staged);return true;
}

bool vertexWeights(const SceneData& scene,const std::string& name,uint64_t vertex,nlohmann::json& report,std::string& error) {
    report=nullptr;WeightSource source;
    if(!resolveWeightSource(scene,name,source,error))return false;
    const auto& geometry=*source.mesh->geometry;
    if(vertex>=geometry.get_vertex_count()){error="rig_vertex_out_of_range";return false;}
    nlohmann::json influences=nlohmann::json::array();
    double sum=0;bool validValues=true;
    if(vertex<geometry.skin_weights.size())for(const auto& weight:geometry.skin_weights[static_cast<size_t>(vertex)]) {
        const auto* key=boneName(source,weight.first);
        const bool finite=std::isfinite(weight.second);
        const bool valid=finite && weight.second>0 && weight.first>=0;
        validValues=validValues && valid;
        if(valid)sum+=weight.second;
        influences.push_back({{"bone_index",weight.first},{"bone",key?nlohmann::json(*key):nlohmann::json(nullptr)},
            {"weight",finite?nlohmann::json(weight.second):nlohmann::json(nullptr)},
            {"value_valid",valid},{"index_known",key!=nullptr},{"index_ambiguous",source.ambiguousIndices.count(weight.first)>0},
            {"belongs_to_character",source.owner && !source.ownerKeys.empty() && key
                ?nlohmann::json(source.ownerKeys.count(*key)>0):nlohmann::json(nullptr)}});
    }
    report=ownershipReport(source);
    report.update({{"object",name},{"vertex",vertex},{"vertex_count",geometry.get_vertex_count()},
        {"influences",influences},{"influence_count",influences.size()},
        {"weight_sum",sum},{"values_valid",validValues},{"unweighted",influences.empty()},
        {"weight_row_present",vertex<geometry.skin_weights.size()}});
    return true;
}
}
