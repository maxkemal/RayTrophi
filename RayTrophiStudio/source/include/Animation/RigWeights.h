#pragma once
#include "json.hpp"
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
struct SceneData;
namespace RayTrophi { class NodeHierarchy; }
namespace RigAuthoring {
struct BoneWeightField {std::vector<float> values;int bone_index=-1;std::size_t invalid_entries=0;};
bool boneWeightField(const SceneData&,const std::string& mesh,const std::string& character,const std::string& bone,BoneWeightField&,std::string& error);
// Fail closed for unskinned authoring even when skin metadata is incomplete.
bool hasFlatSkinReferences(const SceneData&, const RayTrophi::NodeHierarchy&);
// Read exact stored flat weights; never normalize or repair on a query.
bool weightStats(const SceneData&, const std::string& mesh, nlohmann::json& report, std::string& error);
bool vertexWeights(const SceneData&, const std::string& object, uint64_t vertex, nlohmann::json& report, std::string& error);
}
