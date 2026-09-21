#include "Animation/RigSerialization.h"
#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <stdexcept>

namespace RigAuthoring {
namespace {
bool validate(const RayTrophi::NodeHierarchy& h, std::string& error) {
    if(h.size()>static_cast<size_t>(std::numeric_limits<int>::max())){error="invalid_rig_hierarchy_format";return false;}
    std::set<std::string> keys; std::vector<int> state(h.size(),0);
    for(const auto& node:h.nodes) {
        if(node.uniqueName.empty() || !keys.insert(node.uniqueName).second) {error="invalid_rig_node_key";return false;}
        if(node.parent < -1 || (node.parent>=0 && static_cast<size_t>(node.parent)>=h.size())) {error="invalid_rig_parent";return false;}
        for(int r=0;r<4;++r)for(int c=0;c<4;++c)if(!std::isfinite(node.localBind.m[r][c])) {error="invalid_rig_transform";return false;}
    }
    for(size_t start=0;start<h.size();++start) {
        std::vector<size_t> path;int i=static_cast<int>(start);
        while(i>=0 && state[i]==0){state[i]=1;path.push_back(static_cast<size_t>(i));i=h.nodes[i].parent;}
        if(i>=0 && state[i]==1){error="cyclic_rig_hierarchy";return false;}
        for(const auto node:path)state[node]=2;
    }
    return true;
}
void children(RayTrophi::NodeHierarchy& h) {
    for(auto& node:h.nodes)node.children.clear();
    for(size_t i=0;i<h.size();++i)if(h.nodes[i].parent>=0)h.nodes[h.nodes[i].parent].children.push_back(static_cast<int>(i));
}
}
nlohmann::json serializeRigHierarchy(const RayTrophi::NodeHierarchy& h) {
    std::string error;if(!validate(h,error))throw std::runtime_error(error);
    auto nodes=nlohmann::json::array();
    for(const auto& node:h.nodes) {
        auto matrix=nlohmann::json::array();for(int r=0;r<4;++r)for(int c=0;c<4;++c)matrix.push_back(node.localBind.m[r][c]);
        nodes.push_back({{"name",node.name},{"uniqueName",node.uniqueName},{"parent",node.parent},{"localBind",matrix}});
    }
    return {{"version",1},{"nodes",nodes}};
}
bool deserializeRigHierarchy(const nlohmann::json& data, RayTrophi::NodeHierarchy& output, std::string& error) {
    error.clear();RayTrophi::NodeHierarchy staged;
    try {
        if(!data.is_object() || !data.contains("version") || !data["version"].is_number_integer() || data["version"]!=1) {error="unsupported_rig_hierarchy_version";return false;}
        if(!data.contains("nodes") || !data["nodes"].is_array()){error="invalid_rig_hierarchy_format";return false;}
        if(data["nodes"].size()>static_cast<size_t>(std::numeric_limits<int>::max())){error="invalid_rig_hierarchy_format";return false;}
        for(const auto& entry:data["nodes"]) {
            if(!entry.is_object() || !entry.contains("name") || !entry["name"].is_string() ||
               !entry.contains("uniqueName") || !entry["uniqueName"].is_string() ||
               !entry.contains("parent") || !entry["parent"].is_number_integer() ||
               !entry.contains("localBind") || !entry["localBind"].is_array() || entry["localBind"].size()!=16) {
                error="invalid_rig_hierarchy_format";return false;
            }
            RayTrophi::SceneNode node;node.name=entry["name"].get<std::string>();node.uniqueName=entry["uniqueName"].get<std::string>();
            if(entry["parent"].is_number_unsigned() && entry["parent"].get<uint64_t>()>static_cast<uint64_t>(std::numeric_limits<int>::max())){
                error="invalid_rig_parent";return false;
            }
            const auto parent=entry["parent"].get<int64_t>();
            if(parent < -1 || parent>=static_cast<int64_t>(data["nodes"].size())){error="invalid_rig_parent";return false;}
            node.parent=static_cast<int>(parent);
            for(int r=0;r<4;++r)for(int c=0;c<4;++c) {
                const auto& v=entry["localBind"][r*4+c];if(!v.is_number()){error="invalid_rig_transform";return false;}
                node.localBind.m[r][c]=v.get<float>();
            }
            staged.nodes.push_back(std::move(node));
        }
        if(!validate(staged,error))return false;
        children(staged);output=std::move(staged);return true;
    }catch(const std::exception&){error="invalid_rig_hierarchy_format";return false;}
}
bool rebuildLegacyRigHierarchy(const BoneData& bones,const std::string& character,
                               RayTrophi::NodeHierarchy& output,std::string& error) {
    error.clear();const std::string prefix=character+"_";std::map<std::string,Matrix4x4> locals;
    for(const auto& entry:bones.boneDefaultTransforms)if(entry.first.find(prefix)==0)locals[entry.first]=entry.second;
    for(const auto& entry:bones.boneNameToIndex)if(entry.first.find(prefix)==0 && !locals.count(entry.first))locals[entry.first]=Matrix4x4::identity();
    RayTrophi::NodeHierarchy staged;std::map<std::string,int> lookup;
    for(const auto& entry:locals) {
        lookup[entry.first]=static_cast<int>(staged.size());staged.addNode(entry.first.substr(prefix.size()),entry.first,entry.second,-1);
    }
    for(auto& node:staged.nodes) {
        const auto parent=bones.boneParents.find(node.uniqueName);
        if(parent==bones.boneParents.end() || parent->second.empty())continue;
        const auto found=lookup.find(parent->second);
        if(found==lookup.end()){error="missing_legacy_rig_parent";return false;}
        node.parent=found->second;
    }
    if(!validate(staged,error))return false;
    children(staged);
    // Existing tree walkers start at node zero. Put parents first; join forests
    // with an identity helper when the original scene root was not retained.
    std::vector<int> order;for(size_t i=0;i<staged.size();++i)if(staged.nodes[i].parent<0)order.push_back(static_cast<int>(i));
    const bool forest=order.size()>1;RayTrophi::NodeHierarchy rebuilt;
    if(forest) {
        std::string root=prefix+"__LegacyRoot";while(lookup.count(root))root+="_";
        rebuilt.addNode("__LegacyRoot",root,Matrix4x4::identity(),-1);
    }
    std::map<int,int> indices;
    for(size_t i=0;i<order.size();++i) {
        const auto& node=staged.nodes[order[i]];
        const int parent=node.parent<0?(forest?0:-1):indices.at(node.parent);
        indices[order[i]]=rebuilt.addNode(node.name,node.uniqueName,node.localBind,parent);
        order.insert(order.end(),node.children.begin(),node.children.end());
    }
    output=std::move(rebuilt);return true;
}
}
