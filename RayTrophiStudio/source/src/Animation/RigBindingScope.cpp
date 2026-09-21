#include "Animation/RigBindingScope.h"
#include "scene_data.h"
#include "TriangleMesh.h"
#include <algorithm>
#include <unordered_set>
namespace RigAuthoring {
const std::string* explicitMeshRig(const SceneData& scene,const std::string& mesh) {
    for(const auto& model:scene.importedModelContexts)
        if(std::find(model.rigBoundMeshes.begin(),model.rigBoundMeshes.end(),mesh)!=model.rigBoundMeshes.end())return &model.importName;
    return nullptr;
}
bool meshBelongsToRig(const SceneData& scene,const std::string& character,const TriangleMesh& mesh) {
    if(const auto* owner=explicitMeshRig(scene,mesh.nodeName))return *owner==character;
    for(const auto& model:scene.importedModelContexts)if(model.importName==character) {
        for(const auto& member:model.members)if(member.get()==&mesh)return true;
        for(const auto& node:model.nodeHierarchy.nodes)if(node.uniqueName==mesh.nodeName)return true;
    }
    return mesh.nodeName.find(character+"_")==0; // Existing imported model convention.
}
bool readRigBoundMeshes(const nlohmann::json& input,std::vector<std::string>& output,std::string& error) {
    error.clear();std::vector<std::string> staged;std::unordered_set<std::string> names;
    if(!input.is_array() || input.size()>4096){error="rig_binding_invalid_format";return false;}
    for(const auto& value:input) {
        if(!value.is_string()){error="rig_binding_invalid_format";return false;}
        auto name=value.get<std::string>();
        if(name.empty() || name.size()>1024 || !names.insert(name).second){error="rig_binding_invalid_format";return false;}
        staged.push_back(std::move(name));
    }
    output=std::move(staged);return true;
}
bool restoreRigBindingMembers(SceneData& scene,std::string& error) {
    error.clear();std::unordered_set<std::string> claimed;
    std::vector<std::vector<std::shared_ptr<Hittable>>> staged(scene.importedModelContexts.size());
    for(size_t i=0;i<scene.importedModelContexts.size();++i) {
        const auto& model=scene.importedModelContexts[i];
        staged[i]=model.members;
        if(!model.rigBoundMeshes.empty() && !model.authoringOwned){error="rig_binding_requires_owned";return false;}
        for(const auto& name:model.rigBoundMeshes) {
            if(!claimed.insert(name).second){error="rig_binding_duplicate_mesh";return false;}
            std::shared_ptr<TriangleMesh> found;
            for(const auto& object:scene.world.objects) {
                auto mesh=std::dynamic_pointer_cast<TriangleMesh>(object);if(!mesh || mesh->nodeName!=name)continue;
                if(found && found!=mesh){error="ambiguous_mesh_name";return false;}found=mesh;
            }
            // Deleted/missing parts keep their registry identity for undo; do not
            // manufacture geometry or silently attach a diagnostic merged mesh.
            if(found && std::none_of(staged[i].begin(),staged[i].end(),[&](const auto& p){return p.get()==found.get();}))staged[i].push_back(found);
        }
    }
    for(size_t i=0;i<scene.importedModelContexts.size();++i) {
        auto& members=staged[i];const auto& character=scene.importedModelContexts[i].importName;
        members.erase(std::remove_if(members.begin(),members.end(),[&](const auto& p){
            auto mesh=std::dynamic_pointer_cast<TriangleMesh>(p);
            const auto* owner=mesh?explicitMeshRig(scene,mesh->nodeName):nullptr;
            return owner && *owner!=character;
        }),members.end());
    }
    for(size_t i=0;i<staged.size();++i)scene.importedModelContexts[i].members.swap(staged[i]);
    return true;
}
}
