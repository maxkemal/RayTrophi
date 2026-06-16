#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <map>
#include <array>
#include "Triangle.h"
#include "json.hpp"

namespace MeshModifiers {

    // Subdivides a mesh evenly (1 level = each triangle split into 4)
    // Performs a linear subdivision without smoothing (flat subdivide).
    std::vector<std::shared_ptr<Triangle>> SubdivideSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1);

    // Subdivides a mesh and applies Laplacian smoothing to the new vertices.
    // Approximates evaluating a smooth subdivision surface (like Loop subdivision for triangles).
    // NOTE: superseded by CatmullClarkSubD for the Smooth Subdivision modifier; kept for reference.
    std::vector<std::shared_ptr<Triangle>> SmoothSubD(const std::vector<std::shared_ptr<Triangle>>& inputMesh, int levels = 1, float smoothAngle = 0.5f);

    // Per-edge crease sharpness provider: given the two edge endpoints in LOCAL space,
    // returns sharpness in [0,1] (0 = smooth, 1 = fully sharp). Seeds level-1 creases;
    // sharpness is then propagated to child edges across levels. Boundary edges are
    // always sharp regardless. An empty function => smooth everywhere except boundaries.
    using EdgeCreaseFn = std::function<float(const Vec3&, const Vec3&)>;

    // True Catmull-Clark subdivision surface. The original vertices act as a control
    // cage and each level converges toward the CC limit surface (Blender's Subdivision
    // Surface behaviour). Operates on reconstructed topology, so it is immune to the
    // position-quantize normal cancellation of the old SmoothSubD path.
    std::vector<std::shared_ptr<Triangle>> CatmullClarkSubD(
        const std::vector<std::shared_ptr<Triangle>>& inputMesh,
        int levels = 1,
        const EdgeCreaseFn& creaseLookup = {});
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

        // Per-edge crease sharpness in [0,1], authored on the control cage and consumed
        // by the Catmull-Clark evaluator. Keyed by the quantized LOCAL positions of the
        // two edge endpoints, so it is topology-independent: it survives re-welding and
        // maps cleanly onto a future GPU subdivider that rebuilds its own connectivity.
        std::map<std::array<int, 6>, float> edgeCreases;

        // Evaluate modifiers sequentially on a base mesh. Creases (if any) are applied
        // automatically inside Catmull-Clark — no extra plumbing at the call sites.
        std::vector<std::shared_ptr<Triangle>> evaluate(const std::vector<std::shared_ptr<Triangle>>& baseMesh) const;

        // Crease authoring (positions in mesh LOCAL space). weight<=0 clears the edge.
        static std::array<int, 6> makeCreaseKey(const Vec3& a, const Vec3& b);
        void setEdgeCrease(const Vec3& a, const Vec3& b, float weight);
        float getEdgeCrease(const Vec3& a, const Vec3& b) const;

        void serialize(nlohmann::json& j) const;
        void deserialize(const nlohmann::json& j);
    };

} 
