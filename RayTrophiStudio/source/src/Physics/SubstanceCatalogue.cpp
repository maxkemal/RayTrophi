#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "SubstanceCatalogue.h"
#include "SubstanceLibrary.h"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace RayTrophiSim {
namespace {

struct Foundation {
    const char* name;
    const char* example;
    const char* family;
    const char* label;
};

constexpr Foundation foundations[] = {
    {"Generic metal", "Steel", "metals", "Metals"},
    {"Generic mineral", "Stone", "minerals", "Minerals and ceramics"},
    {"Generic polymer", "Plastic (PE)", "polymers", "Polymers and elastomers"},
    {"Generic fibrous solid", "Wood (Oak)", "fibrous", "Fibrous and porous materials"},
    {"Generic liquid", "Water", "liquids", "Liquids"},
    {"Generic mineral grains", "Sand", "minerals", "Minerals and ceramics"},
};

std::pair<std::string, std::string> family(const SubstanceProfile& profile) {
    const auto& name = profile.name;
    for (const auto& entry : foundations) {
        if (name == entry.name) {
            return {entry.family, entry.label};
        }
    }
    if (name == "Iron" || name == "Steel" || name == "Copper") {
        return {"metals", "Metals"};
    }
    if (name == "Stone" || name == "Sand" || name == "Gravel" || name == "Soil") {
        return {"minerals", "Minerals and ceramics"};
    }
    if (name == "Plastic (PE)" || name == "Wax") {
        return {"polymers", "Polymers and elastomers"};
    }
    if (name == "Wood (Oak)" || name == "Paper" || name == "Cloth") {
        return {"fibrous", "Fibrous and porous materials"};
    }
    if (name == "Flesh") {
        return {"biological", "Biological examples"};
    }
    // Identity is not a runtime phase: frozen water remains in the liquid family.
    if (profile.category == SubstanceCategory::Liquid ||
        profile.category == SubstanceCategory::Fuel || name == "Ice" || name == "Snow") {
        return {"liquids", "Liquids"};
    }
    return {"custom", "Other materials"};
}

} // namespace

void appendFoundationSubstances(std::vector<SubstanceProfile>& profiles) {
    for (const auto& entry : foundations) {
        const auto source = std::find_if(profiles.begin(), profiles.end(),
            [&](const SubstanceProfile& value) { return value.name == entry.example; });
        if (source == profiles.end()) {
            throw std::logic_error("Foundation substance example is unavailable");
        }
        // Copy the already configured representative, including its solver hints.
        // No existing named material or saved binding is renamed or retuned.
        auto profile = *source;
        profile.name = entry.name;
        profile.based_on.clear();
        profiles.push_back(std::move(profile));
    }
}

nlohmann::json substanceCatalogueInfo(const SubstanceProfile& profile) {
    const auto* root = &profile;
    // Library publication rejects cycles; follow every valid inheritance level.
    while (!root->based_on.empty()) {
        const auto* parent = tryFindSubstance(root->based_on);
        if (!parent) {
            break;
        }
        root = parent;
    }
    const auto [key, label] = family(*root);
    std::string tier = profile.based_on.empty() ? "example" : "project";
    std::string representative;
    for (const auto& entry : foundations) {
        if (root->name == entry.name) {
            representative = entry.example;
            if (profile.based_on.empty()) {
                tier = "foundation";
            }
            break;
        }
    }
    if (profile.based_on.empty() && root->name == "Flesh") {
        tier = "test";
    }
    return {{"family", key}, {"family_label", label}, {"tier", tier},
        {"root_profile", root->name}, {"representative", representative},
        {"values_kind", "representative_starting_values"},
        {"acoustic_surface_defined", profile.acoustic_defined},
        {"acoustic_medium_defined", profile.acoustic_medium_defined}};
}

} // namespace RayTrophiSim
