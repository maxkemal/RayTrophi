#include "Fluid/MatterMaterialPreset.h"
#include "MaterialStateField.h"
#include "PrincipledBSDF.h"

namespace RayTrophiSim::Fluid {

std::shared_ptr<Material> createMatterMaterialPreset(const std::string& substance,
                                                    std::string& error) {
    if (!tryFindSubstance(substance)) {
        error = "Unknown substance: " + substance + "; use the canonical catalogue name";
        return {};
    }
    Vec3 color(0.65f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float transmission = 0.0f;
    float ior = 1.45f;
    if (substance == "Water" || substance == "Ice" || substance == "Alcohol") {
        color = Vec3(0.92f, 0.97f, 1.0f);
        roughness = substance == "Ice" ? 0.18f : 0.02f;
        transmission = 1.0f;
        ior = substance == "Water" ? 1.333f : 1.31f;
    } else if (substance == "Sand" || substance == "Soil" || substance == "Gravel") {
        color = substance == "Sand" ? Vec3(0.64f, 0.43f, 0.19f)
            : (substance == "Soil" ? Vec3(0.18f, 0.09f, 0.035f) : Vec3(0.35f));
        roughness = 0.88f;
    } else if (substance == "Iron" || substance == "Steel" || substance == "Copper") {
        color = substance == "Copper" ? Vec3(0.95f, 0.64f, 0.54f) : Vec3(0.65f);
        metallic = 1.0f;
        roughness = 0.3f;
    } else if (substance == "Paper" || substance == "Cloth") {
        color = Vec3(0.85f, 0.82f, 0.72f);
        roughness = 0.9f;
    } else if (substance == "Wood (Oak)") {
        color = Vec3(0.35f, 0.16f, 0.06f);
        roughness = 0.7f;
    } else if (substance == "Oil" || substance == "Gasoline") {
        color = Vec3(0.7f, 0.5f, 0.12f);
        roughness = 0.04f;
        transmission = 0.8f;
        ior = 1.46f;
    } else if (substance == "Snow") {
        color = Vec3(0.92f, 0.94f, 0.97f);
        roughness = 0.85f;
    } else if (substance == "Wax") {
        color = Vec3(0.88f, 0.8f, 0.61f);
        roughness = 0.35f;
    }
    auto material = std::make_shared<PrincipledBSDF>(color, roughness, metallic);
    material->setTransmission(transmission, ior);
    error.clear();
    return material;
}

} // namespace RayTrophiSim::Fluid
