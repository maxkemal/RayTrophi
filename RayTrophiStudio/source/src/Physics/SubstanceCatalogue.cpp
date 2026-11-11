#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "SubstanceCatalogue.h"
#include "SubstanceLibrary.h"
#include <string>
#include <utility>

namespace RayTrophiSim {
namespace {

struct Foundation {
    const char* name;
    const char* family;
    const char* label;
};

constexpr Foundation foundations[] = {
    {"Steel", "metals", "Metals"},
    {"Stone", "minerals", "Minerals and ceramics"},
    {"Plastic (PE)", "polymers", "Polymers and elastomers"},
    {"Wood (Oak)", "fibrous", "Fibrous and porous materials"},
    {"Water", "liquids", "Liquids"},
    {"Oil", "liquids", "Liquids"},
    {"Honey", "liquids", "Liquids"},
    {"Sand", "minerals", "Minerals and ceramics"},
    {"Soil", "minerals", "Minerals and ceramics"},
    {"Snow", "liquids", "Liquids"},
    {"Mud", "soft_materials", "Soft solids and pastes"},
    {"Gelatin", "soft_materials", "Soft solids and pastes"},
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
    if (name == "Mud Slurry" || name == "Dough" || name == "Chewing Gum") {
        return {"soft_materials", "Soft solids and pastes"};
    }
    // Identity is not a runtime phase: frozen water remains in the liquid family.
    if (profile.category == SubstanceCategory::Liquid ||
        profile.category == SubstanceCategory::Fuel || name == "Ice" || name == "Snow") {
        return {"liquids", "Liquids"};
    }
    return {"custom", "Other materials"};
}

} // namespace

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
            representative = entry.name;
            if (profile.based_on.empty()) {
                tier = "foundation";
            }
            break;
        }
    }
    if (profile.based_on.empty() && root->name == "Flesh") {
        tier = "test";
    }
    std::string model_notes;
    if (root->name == "Chocolate") {
        model_notes = "Molten chocolate, constant-viscosity approximation; "
            "yield stress and shear thinning are not implemented.";
    } else if (root->name == "Honey" || root->name == "Oil") {
        model_notes = "Fixed reference viscosity; temperature and composition "
            "dependence are not fitted. Resolution can skip the viscous solve.";
    } else if (root->name == "Stone") {
        model_notes = "Molten rock uses the existing Lava starting viscosity and "
            "an equal-density approximation, not a fitted magma rheology.";
    } else if (root->name == "Plastic (PE)") {
        model_notes = "Constant-viscosity polymer melt approximation; "
            "temperature and shear-rate dependence are not fitted.";
    } else if (root->name == "Alcohol") {
        model_notes = "Ethanol reference composition; not a generic alcohol mixture.";
    } else if (root->name == "Chewing Gum") {
        model_notes = "Viscoelastoplastic starting recipe: heating reduces stiffness, "
            "viscosity, yield strain and relaxation time. Not fitted to measured gum data.";
    }
    return {{"family", key}, {"family_label", label}, {"tier", tier},
        {"model_notes", model_notes},
        {"root_profile", root->name}, {"representative", representative},
        {"values_kind", "representative_starting_values"},
        {"acoustic_surface_defined", profile.acoustic_defined},
        {"acoustic_medium_defined", profile.acoustic_medium_defined}};
}

} // namespace RayTrophiSim
