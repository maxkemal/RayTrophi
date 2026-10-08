#pragma once

// Editing side of the substance library (lookups live in MaterialStateField.h).
//
// Built-ins are read-only. "Edit" means derive: a project substance names the
// profile it inherits from and stores ONLY the fields it overrides, so a fix to
// a built-in reaches every substance derived from it. Resolution order:
//
//   built-in  ->  project substance (this file, saved with the project)
//             ->  per-object override (SubstanceOverride, MSF)
//
// Design note: docs/dev/MADDE_TIPLERI_TASARIMI.md §4.

#include "MaterialStateField.h"
#include "json.hpp"

#include <string>
#include <vector>

namespace RayTrophiSim {

struct ProjectSubstance {
    std::string name;
    std::string based_on;
    nlohmann::json overrides = nlohmann::json::object();
};

enum class SubstanceFieldKind : uint8_t { Float, Bool, Model, Category, Color };

// One authorable field of SubstanceProfile. The key is the stable name used by
// the project file, IPC, Python and the panel; they all go through this table,
// so a field cannot be editable in one and missing in another.
struct SubstanceFieldSpec {
    const char* key;
    SubstanceFieldKind kind;
    float min;
    float max;
    const char* unit;
    const char* group;   // panel section
};
const std::vector<SubstanceFieldSpec>& substanceFieldSpecs();

// Every field of a resolved profile, keyed as in substanceFieldSpecs().
nlohmann::json substanceFieldsToJson(const SubstanceProfile& profile);

bool isBuiltinSubstance(const std::string& name);
// Keys the named project substance overrides; empty for a built-in or unknown.
std::vector<std::string> substanceOverriddenFields(const std::string& name);

// Creates a project substance identical to `based_on` (no overrides yet).
bool deriveSubstance(const std::string& name, const std::string& based_on,
                     std::string& error);
// Sets fields on a project substance. A null value drops that override (the
// field follows its base again). Transactional: one invalid field rejects the
// whole patch and nothing changes.
bool patchSubstance(const std::string& name, const nlohmann::json& fields,
                    std::string& error);
// Refuses while another project substance derives from it. Scene references
// (flow sources, colliders, domain bindings) are checked by the caller, which
// can see the scene.
bool removeSubstance(const std::string& name, std::string& error);

// Project file round trip, in creation order (a base always precedes what
// derives from it).
std::vector<ProjectSubstance> projectSubstances();
// Replaces every project substance. Transactional: an unknown base, a cycle,
// an unknown field or an out-of-range value rejects the whole set, names the
// substance, and leaves the library as it was.
bool replaceProjectSubstances(const std::vector<ProjectSubstance>& substances,
                              std::string& error);

} // namespace RayTrophiSim
