#include "DemBondControls.h"
#include "Api/DemBondApi.h"
#include "Fluid/DemBondNetwork.h"
#include "ParticleSimulation.h"
#include "imgui.h"

#include <array>
#include <map>
#include <sstream>
#include <stdexcept>

namespace RayTrophiSim::Fluid {
namespace {
struct Editor {
    float young = 1.0e6f;
    float poisson = 0.25f;
    float tensile = 1.0e4f;
    float shear = 1.0e4f;
    float reach = 1.25f;
    float section = 0.5f;
    bool all = false;
    std::array<char, 512> ids{};
    uint64_t remove_body = 1;
    std::string error;
};
std::map<std::string, Editor> editors;
}

bool editDemBondSeed(nlohmann::json& settings) {
    bool enabled = settings.at("enabled").get<bool>();
    bool changed = ImGui::Checkbox("Seed as one breakable DEM body", &enabled);
    settings["enabled"] = enabled;
    if (enabled) {
        ImGui::TextWrapped("Each seed layer is a separate body. Turn Seed Settle off. "
            "Bond properties require material calibration.");
        const std::pair<const char*, const char*> fields[]{
            {"young_pa", "Bond Young modulus (Pa)"}, {"poisson", "Bond Poisson ratio"},
            {"tensile_strength_pa", "Bond tensile strength (Pa)"},
            {"shear_strength_pa", "Bond shear strength (Pa)"},
            {"reach_ratio", "Bonding reach / diameter"},
            {"section_ratio", "Bond section / grain radius"}};
        for (const auto& [key, label] : fields) {
            double value = settings.at(key).get<double>();
            if (ImGui::InputDouble(label, &value, 0.0, 0.0, "%.6g")) {
                settings[key] = value;
                changed = true;
            }
        }
    }
    return changed;
}

void drawDemBondControls(const SimulationGridDomainDesc& domain,
                         const SimulationGridDomainState* state) {
    if (!domain.fluid_params.grain.enabled ||
        !ImGui::TreeNode("Breakable DEM bodies")) {
        return;
    }
    auto& e = editors[domain.name];
    ImGui::TextWrapped("Bind an existing group into one breakable body. "
        "Future births stay loose. Bond material values require calibration.");
    if (state) {
        const auto info = DemBond::summary(state->particles.dem_bonds);
        ImGui::Text("Bodies: %zu  Bonds: %zu  Broken: %zu",
            info.at("bodies").get<std::size_t>(), info.at("bonds").get<std::size_t>(),
            info.at("broken").get<std::size_t>());
    }
    ImGui::InputFloat("Bond Young modulus (Pa)", &e.young, 0.0f, 0.0f, "%.6g");
    ImGui::InputFloat("Bond Poisson ratio", &e.poisson);
    ImGui::InputFloat("Bond tensile strength (Pa)", &e.tensile, 0.0f, 0.0f, "%.6g");
    ImGui::InputFloat("Bond shear strength (Pa)", &e.shear, 0.0f, 0.0f, "%.6g");
    ImGui::SliderFloat("Bonding reach / diameter", &e.reach, 1.0f, 2.0f);
    ImGui::SliderFloat("Section / grain radius", &e.section, 0.05f, 1.0f);
    ImGui::Checkbox("Select all current DEM grains", &e.all);
    if (!e.all) {
        ImGui::InputText("Particle IDs (comma separated)", e.ids.data(), e.ids.size());
    }
    if (ImGui::Button("Bind selected grains")) {
        try {
            nlohmann::json request{{"action", "bind"},
                {"material", {{"young_pa", e.young}, {"poisson", e.poisson},
                    {"tensile_strength_pa", e.tensile}, {"shear_strength_pa", e.shear}}},
                {"reach_ratio", e.reach}, {"section_ratio", e.section}};
            if (e.all) {
                request["all_current_grains"] = true;
            } else {
                auto ids = nlohmann::json::array();
                std::string text(e.ids.data());
                for (char& c : text) {
                    if (c == ',') {
                        c = ' ';
                    }
                }
                std::istringstream input(text);
                std::string token;
                while (input >> token) {
                    if (token.find_first_not_of("0123456789") != std::string::npos) {
                        throw std::runtime_error("Particle IDs must be positive integers");
                    }
                    ids.push_back(static_cast<uint64_t>(std::stoull(token)));
                }
                request["particle_ids"] = ids;
            }
            const auto result = rtapi::demBodies(domain.name, request);
            e.error = result.value("checkpoint", false) ? std::string() :
                "Body updated; replay checkpoint was not captured. Use a seed recipe to save the setup.";
        } catch (const std::exception& error) {
            e.error = error.what();
        }
    }
    ImGui::InputScalar("Body ID", ImGuiDataType_U64, &e.remove_body);
    if (ImGui::Button("Release body bonds")) {
        try {
            const auto result = rtapi::demBodies(domain.name,
                {{"action", "remove"}, {"body", e.remove_body}});
            e.error = result.value("checkpoint", false) ? std::string() :
                "Body released; replay checkpoint was not captured.";
        } catch (const std::exception& error) {
            e.error = error.what();
        }
    }
    if (!e.error.empty()) {
        ImGui::TextWrapped("%s", e.error.c_str());
    }
    ImGui::TreePop();
}
} // namespace RayTrophiSim::Fluid
