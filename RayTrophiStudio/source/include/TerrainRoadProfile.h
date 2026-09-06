#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// ROAD PROFILES - one data-driven cross-section for every road class
// ═══════════════════════════════════════════════════════════════════════════════
// "Path" and "road" are the SAME measurement under a different profile, not two
// fields and not two solvers. A footpath is a narrow, steep-capable profile with
// a small cut/fill budget; a main road is a wide, shallow-grade one. Making them
// separate code paths would guarantee they drift.

#include "TerrainRoadCarve.h"
#include "json.hpp"

#include <string>
#include <vector>

namespace TerrainNodesV2 {

// RoadCrossingMode itself now lives in TerrainRoadCarve.h: the solver resolves a
// crossing per route sample, so the enum belongs with the solve rather than with
// the cross-section library. Naming and parsing stay here, next to the rest of
// the data-driven road vocabulary.
const char* roadCrossingModeName(RoadCrossingMode mode);
bool parseRoadCrossingMode(const std::string& text, RoadCrossingMode& out);

struct RoadProfile {
    // Stable key. Serialized and used by script; the display name may change
    // without invalidating a saved project.
    std::string id;
    std::string displayName;
    // The profile IS the carve settings. There is deliberately no second copy of
    // width/grade/cut/fill semantics: a profile that could disagree with the
    // solver's own parameters is a lie waiting to be told.
    RoadCarveSettings carve;
};

// The three shipped presets. Built-ins are returned by value so a caller cannot
// mutate the library out from under another consumer.
std::vector<RoadProfile> builtinRoadProfiles();

// Returns false when the id matches no built-in profile. Callers must treat that
// as a diagnostic: silently substituting a default would carve a main road where
// the author asked for a footpath, and nothing about the result would look wrong.
bool findRoadProfile(const std::string& id, RoadProfile& out);

// ONE validator for every writer: the shipped profile library, a script override
// and the panel. A second copy of these bounds would eventually accept what the
// solver refuses, and the symptom is the worst kind - a road that carves nothing
// with no error raised anywhere. `label` names the road in the message.
bool validateRoadCarveSettings(const RoadCarveSettings& settings, const std::string& label,
                               std::string* error);

nlohmann::json roadCarveSettingsToJson(const RoadCarveSettings& settings);
RoadCarveSettings roadCarveSettingsFromJson(const nlohmann::json& value,
                                            const RoadCarveSettings& fallback);

} // namespace TerrainNodesV2
