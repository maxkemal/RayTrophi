#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// ROAD ASSIGNMENTS - which authored curves are roads, and under which profile
// ═══════════════════════════════════════════════════════════════════════════════
// This registry stores NO curve geometry. A road assignment is a reference to a
// SplineObject by name plus the road semantics layered on top: profile, crossing
// mode, enabled state. Curve editing stays entirely on the `spline.*` surface, so
// there is one authority for the geometry of a curve.
//
// It also answers the question that made the graph explode: "which splines are
// roads". Before this, every road segment needed its own Curve Input node feeding
// its own Curve to Mask, because nothing in the system knew a curve was a road.

#include "TerrainRoadProfile.h"
#include "json.hpp"

#include <functional>
#include <string>
#include <vector>

namespace TerrainNodesV2 {

struct RoadAssignment {
    // The SplineObject's nodeName. Names are the scene's own identity for these
    // objects and are what `spline.*` addresses, so the registry speaks the same
    // language the rest of the API does.
    std::string splineObject;
    std::string profileId = "dirt_road";
    RoadCrossingMode crossingMode = RoadCrossingMode::Auto;
    bool enabled = true;
    // Per-assignment override of the profile's carve settings. Empty means "use
    // the profile as shipped"; the override exists so one segment can be widened
    // without cloning a whole profile.
    bool hasOverride = false;
    RoadCarveSettings overrideCarve;
    // The generated surface mesh this assignment owns, empty when none was
    // built. Stored so a rebuild REPLACES the geometry of that object instead of
    // adding a second road beside the first - repeated generation leaving stale
    // duplicates behind is the failure this one string prevents.
    std::string meshObject;

    // The settings this assignment actually solves with.
    RoadCarveSettings effectiveCarve() const;
};

class RoadNetworkRegistry {
public:
    static RoadNetworkRegistry& getInstance();

    // Assign or re-assign. Returns false with an error when the profile id is
    // unknown - never substitutes a default, because carving a main road where a
    // footpath was asked for produces a result that looks entirely plausible.
    bool assignProfile(const std::string& splineObject, const std::string& profileId,
                       std::string* error = nullptr);
    bool setCrossingMode(const std::string& splineObject, RoadCrossingMode mode,
                         std::string* error = nullptr);
    bool setEnabled(const std::string& splineObject, bool enabled,
                    std::string* error = nullptr);
    bool setCarveOverride(const std::string& splineObject, const RoadCarveSettings& settings,
                          std::string* error = nullptr);
    bool clearCarveOverride(const std::string& splineObject, std::string* error = nullptr);
    bool clearProfile(const std::string& splineObject);
    // Records which scene object carries this road's generated surface. Bumps
    // the generation like every other mutation so a save is marked dirty.
    bool setMeshObject(const std::string& splineObject, const std::string& meshObject,
                       std::string* error = nullptr);

    const RoadAssignment* find(const std::string& splineObject) const;
    const std::vector<RoadAssignment>& assignments() const { return assignments_; }

    // Assignments whose curve no longer exists. A segment that quietly stops
    // grading because its spline was deleted is exactly the failure nobody
    // reports, so this is surfaced rather than skipped.
    std::vector<std::string> danglingAssignments(
        const std::function<bool(const std::string&)>& splineExists) const;

    // Bumped on every mutation. Consumers hash it so a graph re-solves when an
    // assignment changes without having to diff the whole registry.
    uint64_t generation() const { return generation_; }

    void clear();
    nlohmann::json serialize() const;
    void deserialize(const nlohmann::json& value);

private:
    RoadNetworkRegistry() = default;
    RoadAssignment* mutableFind(const std::string& splineObject);

    std::vector<RoadAssignment> assignments_;
    uint64_t generation_ = 1;
};

} // namespace TerrainNodesV2
