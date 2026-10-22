#include "UI/GrainGeneratorUI.h"

#include "Api/GrainGeneratorApi.h"
#include "GrainGenerator.h"
#include "MaterialManager.h"
#include <imgui.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace GrainGeneratorUI {
namespace {

struct Editor {
    bool requested = false;
    RayTrophiSim::GrainGeneratorSettings settings;
    char name[64] = "Grain";
    float minimum = 0.02f;
    float maximum = 0.03f;
    Vec3 origin = Vec3(0.0f);
    int distribution = 0;
    std::string material;
    bool weighted = false;
    std::vector<std::pair<std::string, float>> palette;
    std::string domain;
    bool assign = false;
    std::string message;
    bool failed = false;
};

Editor editor;

void materialPicker(const char* label, std::string& selected) {
    const auto& materials = MaterialManager::getInstance().getAllMaterials();
    if (ImGui::BeginCombo(label, selected.empty() ? "Scene default" : selected.c_str())) {
        if (ImGui::Selectable("Scene default", selected.empty())) {
            selected.clear();
        }
        for (const auto& material : materials) {
            if (material && ImGui::Selectable(material->materialName.c_str(),
                    selected == material->materialName)) {
                selected = material->materialName;
            }
        }
        ImGui::EndCombo();
    }
}

} // namespace

void requestOpen(const std::string& domain) {
    editor.requested = true;
    editor.domain = domain;
    editor.assign = !domain.empty();
}

bool consumeOpenRequest() {
    const bool requested = editor.requested;
    editor.requested = false;
    return requested;
}

void drawPanel(UIContext&) {
    auto& settings = editor.settings;
    const char* labels[] = {"Rock / Sand", "Voronoi Fragments", "Snow Crystal", "Mineral Crystal"};
    const char* types[] = {"rock", "voronoi", "snow", "crystal"};
    int kind = settings.type == "rock" ? 0 : settings.type == "snow" ? 2 :
        settings.type == "crystal" ? 3 : 1;
    if (ImGui::Combo("Shape", &kind, labels, 4)) {
        settings.type = types[kind];
        settings.roundness = kind == 0 ? 1.0f : 0.0f;
        settings.aspect = kind == 3 ? Vec3(1, 2, 1) : Vec3(1.0f);
        editor.message.clear();
    }
    ImGui::InputText("Name", editor.name, sizeof(editor.name));
    ImGui::SliderInt("Examples", &settings.count, 1, 64);
    int seed = static_cast<int>(settings.seed);
    if (ImGui::InputInt("Seed", &seed)) {
        settings.seed = static_cast<uint32_t>(std::max(seed, 0));
    }
    if (kind == 0) {
        ImGui::SliderInt("Mesh Resolution", &settings.resolution, 2, 16);
        ImGui::SliderFloat("Roundness", &settings.roundness, 0, 1);
        ImGui::SliderFloat("Surface Irregularity", &settings.roughness, 0, 0.5f);
    } else if (kind == 1) {
        ImGui::SliderFloat("Roundness", &settings.roundness, 0, 1);
        ImGui::SliderInt("Edge Detail", &settings.resolution, 2, 16);
        ImGui::TextWrapped("Roundness softens cell edges; Edge Detail adds real surface geometry.");
    } else if (kind == 2) {
        ImGui::SliderInt("Arms", &settings.arms, 3, 12);
        ImGui::SliderInt("Branches per Side", &settings.branches, 0, 5);
        ImGui::SliderFloat("Branch Length", &settings.branch_length, 0.001f, 0.4f, "%.3f");
        ImGui::SliderFloat("Plate Thickness", &settings.thickness, 0.005f, 0.3f, "%.3f");
        ImGui::SliderFloat("Arm Irregularity", &settings.roughness, 0, 0.5f);
        ImGui::TextWrapped("Branch length is limited by branch spacing to keep a closed outline.");
    } else {
        if (ImGui::Button("Quartz / Ice")) {
            settings.crystal_sides = 6;
            settings.tip_height = 0.3f;
            settings.aspect = Vec3(1, 2, 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("Salt")) {
            settings.crystal_sides = 4;
            settings.tip_height = 0;
            settings.aspect = Vec3(1.0f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Sugar")) {
            settings.crystal_sides = 5;
            settings.tip_height = 0.15f;
            settings.aspect = Vec3(1, 1.5f, 1);
        }
        ImGui::SliderInt("Prism Sides", &settings.crystal_sides, 3, 12);
        ImGui::SliderFloat("Tip Height", &settings.tip_height, 0, 0.9f);
    }
    ImGui::SliderFloat3("Aspect X / Y / Z", &settings.aspect.x, 0.05f, 5.0f);
    ImGui::Separator();
    ImGui::DragFloatRange2("Diameter (m)", &editor.minimum, &editor.maximum,
        0.0001f, 0.00001f, 100.0f, "Min %.5f", "Max %.5f");
    const char* distributions[] = {"Uniform", "Log Uniform"};
    ImGui::Combo("Size Distribution", &editor.distribution, distributions, 2);
    ImGui::InputFloat3("Library Origin", &editor.origin.x);
    ImGui::TextWrapped("Creates editable scene meshes in a spaced grid. Near Grain Meshes matches "
                       "each example to its carrier volume; scene size does not change physics.");
    ImGui::Checkbox("Weighted Material Palette", &editor.weighted);
    if (!editor.weighted) {
        materialPicker("Material", editor.material);
    } else {
        for (std::size_t index = 0; index < editor.palette.size(); ++index) {
            ImGui::PushID(static_cast<int>(index));
            materialPicker("Material", editor.palette[index].first);
            ImGui::SliderFloat("Weight", &editor.palette[index].second, 0.01f, 10.0f);
            if (ImGui::SmallButton("Remove")) {
                editor.palette.erase(editor.palette.begin() + index);
                ImGui::PopID();
                break;
            }
            ImGui::PopID();
        }
        if (editor.palette.size() < 32 && ImGui::SmallButton("Add Material")) {
            editor.palette.emplace_back(editor.material, 1.0f);
        }
    }
    if (!editor.domain.empty()) {
        ImGui::Checkbox("Use Library in Mesh Grains", &editor.assign);
        ImGui::TextWrapped("Target: %s", editor.domain.c_str());
    }
    if (ImGui::Button("Generate Grain / Crystal Library")) {
        nlohmann::json input = {{"type", settings.type}, {"name", editor.name},
            {"count", settings.count}, {"seed", settings.seed}, {"resolution", settings.resolution},
            {"roundness", settings.roundness}, {"roughness", settings.roughness},
            {"aspect", {settings.aspect.x, settings.aspect.y, settings.aspect.z}},
            {"arms", settings.arms}, {"branches", settings.branches},
            {"branch_length", settings.branch_length}, {"thickness", settings.thickness},
            {"crystal_sides", settings.crystal_sides}, {"tip_height", settings.tip_height},
            {"diameter_min", editor.minimum}, {"diameter_max", editor.maximum},
            {"size_distribution", editor.distribution == 0 ? "uniform" : "log_uniform"},
            {"origin", {editor.origin.x, editor.origin.y, editor.origin.z}}};
        if (editor.weighted) {
            input["material_weights"] = nlohmann::json::array();
            for (const auto& entry : editor.palette) {
                input["material_weights"].push_back({{"material", entry.first},
                    {"weight", entry.second}});
            }
        } else {
            input["material"] = editor.material;
        }
        if (editor.assign) {
            input["domain"] = editor.domain;
        }
        nlohmann::json output;
        const auto result = rtapi::generateGrains(input, output);
        editor.failed = !result.ok;
        editor.message = result.ok
            ? "Created " + std::to_string(output.at("count").get<int>()) +
                " editable objects. Undo removes this library in one step."
            : result.error;
    }
    if (!editor.message.empty()) {
        const ImVec4 color = editor.failed
            ? ImVec4(1, 0.5f, 0.35f, 1) : ImVec4(0.6f, 0.9f, 0.7f, 1);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        ImGui::TextWrapped("%s", editor.message.c_str());
        ImGui::PopStyleColor();
    }
}

} // namespace GrainGeneratorUI
