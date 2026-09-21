#pragma once
#include "json.hpp"
#include <string>
#include <memory>
#include <vector>
class TriangleMesh;
struct SceneData;
namespace RigAuthoring {
bool resolveFitParts(const SceneData&,const std::string& target,std::vector<std::shared_ptr<TriangleMesh>>&,std::string& error);
std::shared_ptr<TriangleMesh> resolveFitMesh(const SceneData&,const std::string& target,std::string& error,bool* existingSkin=nullptr);
nlohmann::json fitTargets(const SceneData&);
// Read-only mesh diagnostics. Does not certify pose, watertightness or fit quality.
bool preflightMesh(const SceneData&,const std::string& mesh,nlohmann::json& report,std::string& error);
}
