#pragma once
#include <vector>
#include <memory>
#include "Triangle.h"
#include "json.hpp"

namespace MeshModifiers {
    
    // Subdivides a mesh evenly (1 level = each triangle split into 4)
    // Performs a linear subdivision without smoothing (flat subdivide).
    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1);

    // Subdivides a mesh and applies Laplacian smoothing to the new vertices.
    // Approximates evaluating a smooth subdivision surface (like Loop subdivision for triangles).
    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1, float smoothAngle = 0.5f);
    enum class ModifierType {
        FlatSubdivision,
        SmoothSubdivision
    };

    struct ModifierData {
        std::string name = "Modifier";
        ModifierType type = ModifierType::FlatSubdivision;
        bool enabled = true;
        int levels = 1;
        float smoothAngle = 0.5f;

        void serialize(nlohmann::json& j) const;
        void deserialize(const nlohmann::json& j);
    };

    struct ModifierStack {
        std::vector<ModifierData> modifiers;

        // Evaluate modifiers sequentially on a base mesh
        std::vector<std::shared_ptr<Triangle>> evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh) const;

        void serialize(nlohmann::json& j) const;
        void deserialize(const nlohmann::json& j);
    };

} 
