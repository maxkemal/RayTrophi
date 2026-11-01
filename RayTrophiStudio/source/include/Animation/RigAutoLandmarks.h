#pragma once
#include "json.hpp"
#include <string>
struct SceneData;
namespace RigAuthoring {
// Proposes fit landmarks for a rig on a mesh, in the same fit space and format as fitSetup, so
// the result feeds previewFit / commitRigFit unchanged. Starts from the template scaled into
// the mesh bounds; for humanoids it detects the head top, hand and foot tips from the mesh and
// centres every joint in its local cross-section. Each joint gets a confidence in 0..1 and the
// low ones are listed, so the user only corrects those. Read-only.
bool autoLandmarks(const SceneData& scene, const std::string& character, const std::string& mesh,
                   nlohmann::json& output, std::string& error);
} // namespace RigAuthoring
