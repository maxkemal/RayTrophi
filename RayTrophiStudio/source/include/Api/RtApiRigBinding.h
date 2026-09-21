#pragma once
#include "json.hpp"
#include <string>
namespace rtapi {
struct Result;
Result previewRigMeshBinding(const std::string& character,const std::string& mesh,bool axesConfirmed,nlohmann::json&);
Result bindRigMesh(const std::string& character,const std::string& mesh,const nlohmann::json& preview);
Result unbindRigMesh(const std::string& character);
Result getRigMeshBinding(const std::string& character,nlohmann::json&);
}
