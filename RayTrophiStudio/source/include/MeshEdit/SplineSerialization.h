#pragma once

#include "MeshEdit/SplineObject.h"
#include <json.hpp>

namespace MeshEdit {

// Versioned, lossless authoring payload shared by project save, scripting and IPC.
nlohmann::json serializeSpline(const SplineObject& object);
bool deserializeSpline(const nlohmann::json& payload, SplineObject& object,
                       std::string& error);

const char* splineCurveTypeName(SplineCurveType type);
bool parseSplineCurveType(const std::string& value, SplineCurveType& out);

} // namespace MeshEdit
