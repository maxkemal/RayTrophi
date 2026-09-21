#pragma once
#include "json.hpp"
#include <string>
#include <vector>
struct SceneData;
class TriangleMesh;
namespace RigAuthoring {
bool meshBelongsToRig(const SceneData&,const std::string& character,const TriangleMesh&);
const std::string* explicitMeshRig(const SceneData&,const std::string& mesh);
bool restoreRigBindingMembers(SceneData&,std::string& error);
bool readRigBoundMeshes(const nlohmann::json&,std::vector<std::string>&,std::string& error);
}
